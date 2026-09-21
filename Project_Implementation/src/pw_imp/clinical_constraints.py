"""Clinical constraint validation and projection utilities for the tabular PCOS pipeline.

This module enforces a formal clinical guardrail layer for counterfactual generation:
- immutable feature protection
- range checks
- max-step checks
- binary/categorical validation
- non-actionable modification prevention
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "clinical_constraints.yaml"


def load_clinical_constraints(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clinical constraint config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    feature_map = raw.get("feature", {}) if isinstance(raw, dict) else {}
    return feature_map


class ClinicalConstraintEngine:
    """Validate and project candidate feature edits against a clinical constraint map."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config = load_clinical_constraints(config_path)

    def get_feature_spec(self, feature_name: str) -> Dict[str, Any]:
        if feature_name not in self.config:
            return {}
        return self.config[feature_name]

    def get_counterfactual_features(self, current_values: Mapping[str, Any] | None = None) -> list[str]:
        """Return configured features eligible for counterfactual modification."""
        return [
            name for name, spec in self.config.items()
            if bool(spec.get("actionable"))
            and bool(spec.get("changeable"))
            and (current_values is None or name in current_values)
        ]

    def _coerce_numeric(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _values_equal(self, left: Any, right: Any) -> bool:
        if left is right:
            return True
        try:
            if bool(left != left) and bool(right != right):
                return True
        except (TypeError, ValueError):
            pass
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def _is_binary_like(self, spec: Mapping[str, Any]) -> bool:
        return str(spec.get("type", "")).lower() in {"binary", "categorical"}

    def _validate_direction(self, feature_name: str, current_value, new_value, spec: Mapping[str, Any]):
        allowed = str(spec.get("allowed_direction") or "").lower()
        if not allowed or allowed == "both":
            return
        if current_value is None:
            return
        current_num = self._coerce_numeric(current_value)
        new_num = self._coerce_numeric(new_value)
        if current_num is None or new_num is None:
            return
        if allowed == "increase" and new_num < current_num:
            raise ValueError(f"Feature '{feature_name}' may only increase; requested {new_value} from {current_value}.")
        if allowed == "decrease" and new_num > current_num:
            raise ValueError(f"Feature '{feature_name}' may only decrease; requested {new_value} from {current_value}.")

    def _validate_binary_or_categorical(self, feature_name: str, value, spec: Mapping[str, Any]):
        t = str(spec.get("type") or "").lower()
        if t not in {"binary", "categorical"}:
            return
        if "allowed_values" in spec and value not in spec["allowed_values"]:
            raise ValueError(f"Feature '{feature_name}' has unsupported categorical value {value!r}.")
        lower = spec.get("lower_bound")
        upper = spec.get("upper_bound")
        if lower is not None and upper is not None:
            val = self._coerce_numeric(value)
            if val is not None and (val < float(lower) or val > float(upper)):
                raise ValueError(f"Feature '{feature_name}' value {value} is out of range [{lower}, {upper}].")

    def _dependency_names_for_feature(self, feature_name: str):
        spec = self.get_feature_spec(feature_name)
        dependency = spec.get("dependency") if spec else None
        if dependency is None:
            return []
        if isinstance(dependency, str):
            return [part.strip() for part in dependency.split(",") if part.strip()]
        if isinstance(dependency, (list, tuple)):
            return [str(part).strip() for part in dependency if str(part).strip()]
        return []

    def _recompute_derived_value(self, feature_name: str, values: Mapping[str, Any]):
        if feature_name == "BMI":
            weight = self._coerce_numeric(values.get("Weight (Kg)"))
            height_cm = self._coerce_numeric(values.get("Height(Cm)"))
            if weight is not None and height_cm is not None and height_cm > 0:
                height_m = height_cm / 100.0
                return weight / (height_m ** 2)
            return None

        if feature_name == "FSH/LH":
            fsh = self._coerce_numeric(values.get("FSH(mIU/mL)"))
            lh = self._coerce_numeric(values.get("LH(mIU/mL)"))
            if fsh is not None and lh is not None and lh != 0:
                return fsh / lh
            return None

        if feature_name == "Waist:Hip Ratio":
            waist = self._coerce_numeric(values.get("Waist(inch)"))
            hip = self._coerce_numeric(values.get("Hip(inch)"))
            if waist is not None and hip is not None and hip != 0:
                return waist / hip
            return None

        return None

    def _validate_dependency_consistency(self, feature_name: str, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]):
        dependencies = self._dependency_names_for_feature(feature_name)
        if not dependencies:
            return

        merged = {**current_values, **candidate_values}
        recomputed = self._recompute_derived_value(feature_name, merged)
        if recomputed is None:
            return

        proposed = self._coerce_numeric(candidate_values.get(feature_name, merged.get(feature_name)))
        if proposed is None:
            return

        if abs(proposed - recomputed) > 1e-6:
            raise ValueError(
                f"Dependency mismatch for '{feature_name}': expected {recomputed:.6f} from "
                f"{dependencies}, received {proposed}."
            )

    def validate_feature_change(self, feature_name: str, current_value: Any, proposed_value: Any) -> bool:
        spec = self.get_feature_spec(feature_name)
        if not spec:
            if proposed_value != current_value:
                raise ValueError(f"Constraint definition missing for feature '{feature_name}'; it is non-actionable.")
            return True

        immutable = bool(spec.get("immutable", False))
        if immutable and not self._values_equal(proposed_value, current_value):
            raise ValueError(f"Feature '{feature_name}' is immutable and cannot be modified.")

        actionable = bool(spec.get("actionable", False))
        if (not actionable) and not self._values_equal(proposed_value, current_value) and not immutable:
            # This guard prevents non-actionable or measurement features from being changed automatically.
            if str(spec.get("type", "")).lower() not in {"binary", "categorical"}:
                raise ValueError(f"Feature '{feature_name}' is non-actionable and cannot be modified.")

        lower = spec.get("lower_bound")
        upper = spec.get("upper_bound")
        if lower is not None or upper is not None:
            val = self._coerce_numeric(proposed_value)
            if val is None:
                raise ValueError(f"Feature '{feature_name}' requires a numeric value for range validation.")
            if lower is not None:
                if val < float(lower):
                    raise ValueError(f"Feature '{feature_name}' proposed value {proposed_value} is below lower bound {lower}.")
            if upper is not None:
                if val > float(upper):
                    raise ValueError(f"Feature '{feature_name}' proposed value {proposed_value} is above upper bound {upper}.")

        if current_value is not None and proposed_value is not None:
            max_step = spec.get("max_step")
            if max_step is not None:
                current_num = self._coerce_numeric(current_value)
                proposed_num = self._coerce_numeric(proposed_value)
                if current_num is not None and proposed_num is not None:
                    if abs(proposed_num - current_num) > float(max_step):
                        raise ValueError(
                            f"Feature '{feature_name}' exceeds max step {max_step} "
                            f"(delta={abs(proposed_num - current_num)})."
                        )

        self._validate_direction(feature_name, current_value, proposed_value, spec)
        self._validate_binary_or_categorical(feature_name, proposed_value, spec)
        return True

    def validate_candidate(self, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]) -> Dict[str, Any]:
        merged = dict(current_values)
        changed = set(self.changed_features(current_values, candidate_values))
        for feature_name in changed:
            proposed_value = candidate_values[feature_name]
            if feature_name in current_values:
                current_value = current_values[feature_name]
            else:
                current_value = None
            spec = self.get_feature_spec(feature_name)
            if str(spec.get("type", "")).lower() == "derived":
                self._validate_dependency_consistency(feature_name, current_values, candidate_values)
            else:
                self.validate_feature_change(feature_name, current_value, proposed_value)
            merged[feature_name] = proposed_value

        for feature_name, spec in self.config.items():
            if str(spec.get("type", "")).lower() != "derived" or feature_name not in candidate_values:
                continue
            dependencies = set(self._dependency_names_for_feature(feature_name))
            if feature_name in changed or dependencies.intersection(changed):
                self._validate_dependency_consistency(feature_name, current_values, candidate_values)

        for feature_name in current_values:
            if feature_name not in changed:
                merged[feature_name] = current_values[feature_name]
        return merged

    def immutable_features_unchanged(self, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]) -> bool:
        """Return whether every configured immutable value is exactly preserved."""
        return all(
            self._values_equal(candidate_values.get(feature_name, current_values.get(feature_name)), current_values.get(feature_name))
            for feature_name, spec in self.config.items()
            if spec.get("immutable") and feature_name in current_values
        )

    def changed_features(self, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]) -> list[str]:
        """Return changed fields while preserving NaN equality semantics."""
        changed = []
        for feature_name in current_values:
            current = current_values.get(feature_name)
            proposed = candidate_values.get(feature_name, current)
            try:
                equal = bool((current == proposed) or (current != current and proposed != proposed))
            except (TypeError, ValueError):
                equal = current == proposed
            if not equal:
                changed.append(feature_name)
        return changed

    def audit_candidate(self, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]) -> Dict[str, Any]:
        """Return candidate-aware constraint failures without projecting the row."""
        changed = self.changed_features(current_values, candidate_values)
        failures: list[Dict[str, str]] = []

        for feature_name in candidate_values:
            if feature_name not in current_values and not self._values_equal(candidate_values[feature_name], None):
                failures.append({
                    "constraint": "constraint_definition",
                    "feature": feature_name,
                    "reason": f"Feature '{feature_name}' is not present in the original patient.",
                })

        for feature_name, spec in self.config.items():
            if not spec.get("immutable") or feature_name not in current_values:
                continue
            proposed = candidate_values.get(feature_name, current_values[feature_name])
            if not self._values_equal(proposed, current_values[feature_name]):
                failures.append({
                    "constraint": "immutable",
                    "feature": feature_name,
                    "reason": f"Feature '{feature_name}' is immutable and changed.",
                })

        for feature_name in changed:
            spec = self.get_feature_spec(feature_name)
            if not spec:
                failures.append({
                    "constraint": "constraint_definition",
                    "feature": feature_name,
                    "reason": f"Actionable feature '{feature_name}' has no constraint definition.",
                })
                continue
            if bool(spec.get("immutable")):
                continue
            if str(spec.get("type", "")).lower() == "derived":
                failures.append({
                    "constraint": "derived_feature_edit",
                    "feature": feature_name,
                    "reason": f"Derived feature '{feature_name}' cannot be independently changed.",
                })
                continue
            if not bool(spec.get("actionable", False)):
                failures.append({
                    "constraint": "non_actionable",
                    "feature": feature_name,
                    "reason": f"Feature '{feature_name}' is non-actionable and changed.",
                })
                continue
            try:
                self.validate_feature_change(feature_name, current_values.get(feature_name), candidate_values.get(feature_name))
            except (ValueError, TypeError) as exc:
                message = str(exc)
                if "max step" in message:
                    category = "maximum_change"
                elif "below lower" in message or "above upper" in message or "out of range" in message:
                    category = "bounds"
                elif "may only" in message:
                    category = "allowed_direction"
                elif "categorical" in message:
                    category = "categorical_validity"
                else:
                    category = "clinical_constraint"
                failures.append({"constraint": category, "feature": feature_name, "reason": message})

        changed_set = set(changed)
        for feature_name, spec in self.config.items():
            if str(spec.get("type", "")).lower() != "derived" or feature_name not in candidate_values:
                continue
            dependencies = set(self._dependency_names_for_feature(feature_name))
            if feature_name not in changed_set and not dependencies.intersection(changed_set):
                continue
            try:
                self._validate_dependency_consistency(feature_name, current_values, candidate_values)
            except (ValueError, TypeError) as exc:
                failures.append({"constraint": "dependency_consistency", "feature": feature_name, "reason": str(exc)})

        return {"valid": not failures, "changed_features": changed, "failures": failures}

    def project_candidate(self, current_values: Mapping[str, Any], candidate_values: Mapping[str, Any]) -> Dict[str, Any]:
        projected = dict(current_values)
        changed = set(self.changed_features(current_values, candidate_values))
        for feature_name, proposed_value in candidate_values.items():
            if feature_name not in current_values:
                if not self._values_equal(proposed_value, current_values.get(feature_name)):
                    raise ValueError(f"Feature '{feature_name}' is not present in the original patient and cannot be modified.")
                continue
            current_value = current_values[feature_name]
            spec = self.get_feature_spec(feature_name)
            if not spec:
                if not self._values_equal(proposed_value, current_value):
                    raise ValueError(f"Constraint definition missing for feature '{feature_name}'; it is non-actionable.")
                projected[feature_name] = proposed_value
                continue

            if bool(spec.get("immutable", False)):
                if not self._values_equal(proposed_value, current_value):
                    raise ValueError(f"Feature '{feature_name}' is immutable and cannot be modified.")
                projected[feature_name] = current_value
                continue

            if str(spec.get("type", "")).lower() == "derived":
                projected[feature_name] = current_value
                continue

            if not bool(spec.get("actionable", False)) and not self._values_equal(proposed_value, current_value):
                raise ValueError(f"Feature '{feature_name}' is non-actionable and cannot be modified.")

            if self._values_equal(proposed_value, current_value):
                projected[feature_name] = current_value
                continue

            val = proposed_value
            lower = spec.get("lower_bound")
            upper = spec.get("upper_bound")
            if lower is not None or upper is not None:
                num = self._coerce_numeric(val)
                if num is not None:
                    if lower is not None:
                        num = max(num, float(lower))
                    if upper is not None:
                        num = min(num, float(upper))
                    val = num

            max_step = spec.get("max_step")
            if max_step is not None:
                current_num = self._coerce_numeric(current_value)
                proposed_num = self._coerce_numeric(val)
                if current_num is not None and proposed_num is not None:
                    delta = proposed_num - current_num
                    limit = float(max_step)
                    if abs(delta) > limit:
                        if delta > 0:
                            val = current_num + limit
                        else:
                            val = current_num - limit

            projected[feature_name] = val

        # Recalculate only configured derived fields affected by a changed dependency.
        for feature_name, spec in self.config.items():
            if str(spec.get("type", "")).lower() != "derived" or feature_name not in projected:
                continue
            dependencies = set(self._dependency_names_for_feature(feature_name))
            if not dependencies.intersection(changed):
                continue
            repair_value = self._recompute_derived_value(feature_name, projected)
            if repair_value is not None:
                projected[feature_name] = repair_value

        # Ensure no inconsistent dependency remains. This may raise if candidate violates the formula.
        for feature_name, spec in self.config.items():
            if str(spec.get("type", "")).lower() != "derived" or feature_name not in candidate_values:
                continue
            dependencies = set(self._dependency_names_for_feature(feature_name))
            if feature_name in changed or dependencies.intersection(changed):
                self._validate_dependency_consistency(feature_name, current_values, projected)

        return projected


def validate_candidate(current_values: Mapping[str, Any], candidate_values: Mapping[str, Any], config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    engine = ClinicalConstraintEngine(config_path)
    return engine.validate_candidate(current_values, candidate_values)


def project_candidate(current_values: Mapping[str, Any], candidate_values: Mapping[str, Any], config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    engine = ClinicalConstraintEngine(config_path)
    return engine.project_candidate(current_values, candidate_values)
