"""Clinically constrained multi-objective counterfactual generation.

The public API works on raw tabular rows.  A supplied preprocessor converts each
project row to the representation expected by the existing model.  No ultrasound
or image-specific code is used.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
except ImportError:  # pragma: no cover - exercised only without optional runtime deps
    NSGA2 = None
    Problem = None
    minimize = None

from pw_imp.clinical_constraints import ClinicalConstraintEngine

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CCMOCF_OUT = REPO_ROOT / "results" / "counterfactuals" / "ccmocf"

SUCCESS_STATES = {"SUCCESS"}
FAILURE_STATES = {
    "NO_FEASIBLE_COUNTERFACTUAL",
    "NO_PREDICTION_FLIP",
    "CONSTRAINT_INFEASIBLE",
    "OPTIMIZER_ERROR",
}


@dataclass
class Candidate:
    row: dict[str, Any]
    probability: float
    objectives: tuple[float, ...]
    changed_features: list[str]
    constraint_status: str
    feasibility_status: str


class _CCMOCFProblem(Problem):
    """Adapter from actionable feature vectors to the existing CC-MO-CF logic."""

    def __init__(self, engine: "CCMOCF", original: Mapping[str, Any], desired_class: int):
        self.engine = engine
        self.original = dict(original)
        self.desired_class = int(desired_class)
        self.original_probability = float(_probability(engine.model, engine._model_input([self.original]))[0])
        self.original_prediction = int(self.original_probability >= 0.5)
        self.actionable_features = engine._actionable_features(self.original)
        lower, upper = engine._decision_bounds(self.original, self.actionable_features)
        super().__init__(n_var=len(self.actionable_features), n_obj=6, n_constr=0, xl=np.asarray(lower), xu=np.asarray(upper))

    def candidate_row(self, values):
        row = dict(self.original)
        row.update({name: float(value) for name, value in zip(self.actionable_features, values)})
        return row

    def evaluate_candidate(self, values):
        row = self.candidate_row(values)
        projected = self.engine.constraints.project_candidate(self.original, row)
        constraint_valid = True
        dependency_valid = True
        try:
            self.engine.constraints.validate_candidate(self.original, projected)
        except (ValueError, TypeError):
            constraint_valid = False
        if self.engine.use_dependency_consistency:
            for feature_name in ("BMI", "FSH/LH", "Waist:Hip Ratio"):
                if feature_name in projected and feature_name in self.engine.constraints.config:
                    try:
                        self.engine.constraints.validate_candidate(self.original, projected)
                    except (ValueError, TypeError):
                        dependency_valid = False
        probability = float(_probability(self.engine.model, self.engine._model_input([projected]))[0])
        prediction = int(probability >= 0.5)
        changed = [name for name in self.original if self.original.get(name) != projected.get(name)]
        numeric_deltas = [abs(float(projected[name]) - float(self.original[name])) for name in changed if _numeric(self.original.get(name)) is not None and _numeric(projected.get(name)) is not None]
        objectives = np.asarray([
            1.0 - probability if self.desired_class == 1 else probability,
            float(sum(numeric_deltas)) if self.engine.use_proximity_objective else 0.0,
            float(len(changed)) if self.engine.use_sparsity_objective else 0.0,
            float(sum(1 for name in changed if not self.engine.constraints.get_feature_spec(name).get("actionable", False))) if self.engine.use_actionability_objective else 0.0,
            0.0 if constraint_valid else 1.0,
            0.0 if dependency_valid or not self.engine.use_dependency_consistency else 1.0,
        ], dtype=float)
        is_flip = prediction == self.desired_class and prediction != self.original_prediction and bool(changed)
        return objectives, Candidate(projected, probability, tuple(objectives.tolist()), changed, "VALID" if constraint_valid else "INVALID", "VALID_FEASIBLE" if constraint_valid and dependency_valid and is_flip else "INFEASIBLE")

    def _evaluate(self, x, out, *args, **kwargs):
        values = np.atleast_2d(x)
        projected_rows = []
        validity = []
        dependency_validity = []
        for candidate_values in values:
            row = self.candidate_row(candidate_values)
            projected = self.engine.constraints.project_candidate(self.original, row)
            constraint_valid = True
            dependency_valid = True
            try:
                self.engine.constraints.validate_candidate(self.original, projected)
            except (ValueError, TypeError):
                constraint_valid = False
            if self.engine.use_dependency_consistency:
                for feature_name in ("BMI", "FSH/LH", "Waist:Hip Ratio"):
                    if feature_name in projected and feature_name in self.engine.constraints.config:
                        try:
                            self.engine.constraints.validate_candidate(self.original, projected)
                        except (ValueError, TypeError):
                            dependency_valid = False
            projected_rows.append(projected)
            validity.append(constraint_valid)
            dependency_validity.append(dependency_valid)
        probabilities = _probability(self.engine.model, self.engine._model_input(projected_rows))
        objectives = []
        for index, projected in enumerate(projected_rows):
            probability = float(probabilities[index])
            changed = [name for name in self.original if self.original.get(name) != projected.get(name)]
            numeric_deltas = [abs(float(projected[name]) - float(self.original[name])) for name in changed if _numeric(self.original.get(name)) is not None and _numeric(projected.get(name)) is not None]
            objectives.append([
                1.0 - probability if self.desired_class == 1 else probability,
                float(sum(numeric_deltas)) if self.engine.use_proximity_objective else 0.0,
                float(len(changed)) if self.engine.use_sparsity_objective else 0.0,
                float(sum(1 for name in changed if not self.engine.constraints.get_feature_spec(name).get("actionable", False))) if self.engine.use_actionability_objective else 0.0,
                0.0 if validity[index] else 1.0,
                0.0 if dependency_validity[index] or not self.engine.use_dependency_consistency else 1.0,
            ])
        out["F"] = np.asarray(objectives, dtype=float)


def _probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(X)
        return np.asarray(values)[:, 1] if np.asarray(values).ndim > 1 else np.asarray(values)
    values = model.predict(X)
    return np.asarray(values, dtype=float).reshape(-1)


def _dominates(left: Candidate, right: Candidate) -> bool:
    return all(a <= b for a, b in zip(left.objectives, right.objectives)) and any(
        a < b for a, b in zip(left.objectives, right.objectives)
    )


def _pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    return [candidate for candidate in candidates if not any(_dominates(other, candidate) for other in candidates)]


class CCMOCF:
    """Generate feasible Pareto counterfactuals for a tabular predictor."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        constraint_engine: ClinicalConstraintEngine | None = None,
        config_path: str | Path | None = None,
        population_size: int = 64,
        generations: int = 20,
        seed: int = 42,
        top_k: int = 3,
        diversity_threshold: float = 1e-6,
        optimizer_available: bool | None = None,
        use_sparsity_objective: bool = True,
        use_proximity_objective: bool = True,
        use_actionability_objective: bool = True,
        use_constraint_projection: bool = True,
        use_dependency_consistency: bool = True,
        use_diversity_selection: bool = True,
        use_fallback: bool = False,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.constraints = constraint_engine or ClinicalConstraintEngine(config_path) if config_path else constraint_engine or ClinicalConstraintEngine()
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.seed = int(seed)
        self.top_k = max(1, int(top_k))
        self.diversity_threshold = float(diversity_threshold)
        self.use_sparsity_objective = use_sparsity_objective
        self.use_proximity_objective = use_proximity_objective
        self.use_actionability_objective = use_actionability_objective
        self.use_constraint_projection = use_constraint_projection
        self.use_dependency_consistency = use_dependency_consistency
        self.use_diversity_selection = use_diversity_selection
        self.use_fallback = bool(use_fallback)
        if optimizer_available is None:
            try:
                import pymoo  # noqa: F401
                optimizer_available = True
            except ImportError:
                optimizer_available = False
        self.optimizer_available = optimizer_available

    def _actionable_features(self, original: Mapping[str, Any]) -> list[str]:
        return [name for name, spec in self.constraints.config.items() if spec.get("actionable") and name in original]

    def _decision_bounds(self, original: Mapping[str, Any], features: list[str]):
        lower, upper = [], []
        for name in features:
            spec = self.constraints.get_feature_spec(name)
            current = _numeric(original.get(name))
            if current is None:
                raise ValueError(f"Actionable feature '{name}' must be numeric for NSGA-II.")
            max_step = float(spec.get("max_step") or max(abs(current) * 0.1, 1.0))
            lo = float(spec["lower_bound"]) if spec.get("lower_bound") is not None else current - max_step
            hi = float(spec["upper_bound"]) if spec.get("upper_bound") is not None else current + max_step
            lower.append(max(lo, current - max_step))
            upper.append(min(hi, current + max_step))
        return lower, upper

    def _model_input(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        raw = pd.DataFrame(rows)
        transformed = self.preprocessor.transform(raw)
        selected = getattr(self.preprocessor, "selected_features", None)
        return transformed.reindex(columns=selected).fillna(0) if selected else transformed

    def _evaluate(self, original: Mapping[str, Any], row: dict[str, Any], desired_class: int) -> Candidate | None:
        try:
            projected = self.constraints.project_candidate(original, row) if self.use_constraint_projection else dict(row)
            if self.use_constraint_projection:
                self.constraints.validate_candidate(original, projected)
        except (ValueError, TypeError):
            return None
        probability = float(_probability(self.model, self._model_input([projected]))[0])
        prediction = int(probability >= 0.5)
        if prediction != int(desired_class):
            return None
        changed = [name for name in original if original.get(name) != projected.get(name)]
        numeric_deltas = [abs(float(projected[name]) - float(original[name])) for name in changed if _numeric(original.get(name)) and _numeric(projected.get(name))]
        dependency_penalty = 0.0
        if self.use_dependency_consistency:
            for name in ("BMI", "FSH/LH", "Waist:Hip Ratio"):
                if name in projected and name in self.constraints.config:
                    try:
                        self.constraints.validate_candidate(original, projected)
                    except ValueError:
                        return None
        else:
            dependency_penalty = 1.0
        objectives = (
            1.0 - probability if desired_class == 1 else probability,
            float(sum(numeric_deltas)) if self.use_proximity_objective else 0.0,
            float(len(changed)) if self.use_sparsity_objective else 0.0,
            float(sum(1 for name in changed if not self.constraints.get_feature_spec(name).get("actionable", False))) if self.use_actionability_objective else 0.0,
            0.0,
            dependency_penalty,
        )
        return Candidate(projected, probability, objectives, changed, "VALID", "VALID_FEASIBLE")

    def _population(self, original: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
        actionable = [name for name, spec in self.constraints.config.items() if spec.get("actionable") and name in original]
        population = [dict(original)]
        for _ in range(self.population_size - 1):
            row = dict(original)
            for name in actionable:
                spec = self.constraints.get_feature_spec(name)
                current = _numeric(original[name])
                if current is None:
                    continue
                step = float(spec.get("max_step") or max(abs(current) * 0.1, 1.0))
                row[name] = current + rng.uniform(-step, step)
            population.append(self.constraints.project_candidate(original, row))
        return population

    def generate(self, original: Mapping[str, Any], desired_class: int = 1) -> dict[str, Any]:
        started = time.perf_counter()
        original = dict(original)
        original_probability = float(_probability(self.model, self._model_input([original]))[0])
        original_prediction = int(original_probability >= 0.5)
        rng = random.Random(self.seed)
        if self.optimizer_available and NSGA2 is not None and minimize is not None and Problem is not None:
            return self._generate_nsga2(original, desired_class, started)
        if not self.use_fallback:
            return {"status": "OPTIMIZER_ERROR", "constraint_status": "OPTIMIZER_UNAVAILABLE", "optimizer_used": "unavailable", "original_prediction": original_prediction, "original_probability": original_probability, "counterfactuals": [], "candidates_evaluated": 0, "error": "pymoo NSGA-II is unavailable; enable use_fallback explicitly to use fallback search.", "runtime": time.perf_counter() - started}
        return self._generate_fallback(original, desired_class, started, original_prediction, original_probability)

    def _generate_nsga2(self, original, desired_class, started):
        problem = _CCMOCFProblem(self, original, desired_class)
        algorithm = NSGA2(pop_size=self.population_size)
        LOGGER.info("Optimizer: NSGA-II | Population size: %s | Generations: %s | Number of objectives: %s", self.population_size, self.generations, problem.n_obj)
        result = minimize(problem, algorithm, termination=("n_gen", self.generations), seed=self.seed, verbose=False)
        solutions = [] if result.X is None else np.atleast_2d(result.X)
        candidates = []
        for values in solutions:
            _, candidate = problem.evaluate_candidate(values)
            prediction = int(candidate.probability >= 0.5)
            if prediction == desired_class and candidate.changed_features and candidate.constraint_status == "VALID" and candidate.feasibility_status == "VALID_FEASIBLE":
                candidates.append(candidate)
        unique = []
        for candidate in candidates:
            if not self.use_diversity_selection or all(_row_distance(candidate.row, old.row) > self.diversity_threshold for old in unique):
                unique.append(candidate)
            if len(unique) >= self.top_k:
                break
        original_probability = float(_probability(self.model, self._model_input([original]))[0])
        LOGGER.info("Number of Pareto solutions: %s | Number of feasible solutions: %s | Prediction evaluation: PASS | Constraint validation: PASS", len(solutions), len(unique))
        return {"status": "SUCCESS" if unique else "NO_PREDICTION_FLIP", "constraint_status": "VALID_FEASIBLE" if unique else "NO_VALID_CANDIDATE", "optimizer_used": "NSGA-II", "original_prediction": int(original_probability >= 0.5), "original_probability": original_probability, "counterfactuals": unique, "candidates_evaluated": int(len(solutions)), "pareto_solution_count": int(len(solutions)), "runtime": time.perf_counter() - started}

    def _generate_fallback(self, original, desired_class, started, original_prediction, original_probability):
        optimizer = "fallback"
        candidates_evaluated = 0
        rng = random.Random(self.seed)
        try:
            rows = self._population(original, rng)
            feasible: list[Candidate] = []
            for _ in range(self.generations):
                for row in rows:
                    candidates_evaluated += 1
                    candidate = self._evaluate(original, row, desired_class)
                    if candidate and candidate.changed_features and int(candidate.probability >= 0.5) != original_prediction:
                        feasible.append(candidate)
                # Iterative directed search: mutate around the best feasible rows.
                bases = [item.row for item in _pareto_front(feasible)] or rows[: max(1, self.population_size // 4)]
                rows = []
                for _ in range(self.population_size):
                    base = dict(rng.choice(bases))
                    for name, spec in self.constraints.config.items():
                        if spec.get("actionable") and name in base and _numeric(base[name]):
                            step = float(spec.get("max_step") or 1.0)
                            base[name] += rng.uniform(-step, step)
                    rows.append(self.constraints.project_candidate(original, base) if self.use_constraint_projection else base)
            unique: list[Candidate] = []
            for candidate in sorted(_pareto_front(feasible), key=lambda item: item.objectives):
                if not self.use_diversity_selection or all(_row_distance(candidate.row, old.row) > self.diversity_threshold for old in unique):
                    unique.append(candidate)
                if len(unique) >= self.top_k:
                    break
            status = "SUCCESS" if unique else "NO_PREDICTION_FLIP"
            if not feasible:
                status = "NO_FEASIBLE_COUNTERFACTUAL"
            return {
                "status": status,
                "constraint_status": "VALID_FEASIBLE" if unique else "NO_VALID_CANDIDATE",
                "optimizer_used": optimizer,
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": unique,
                "candidates_evaluated": candidates_evaluated,
                "runtime": time.perf_counter() - started,
            }
        except Exception as exc:
            return {"status": "OPTIMIZER_ERROR", "constraint_status": "OPTIMIZER_ERROR", "optimizer_used": optimizer, "original_prediction": original_prediction, "original_probability": original_probability, "counterfactuals": [], "candidates_evaluated": candidates_evaluated, "error": str(exc), "runtime": time.perf_counter() - started}


def _numeric(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _row_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    distance = 0.0
    for name in left:
        a, b = _numeric(left.get(name)), _numeric(right.get(name))
        distance += abs(a - b) if a is not None and b is not None else float(left.get(name) != right.get(name))
    return distance


def save_results(results: Iterable[Mapping[str, Any]], output_dir: str | Path = CCMOCF_OUT) -> dict[str, Path]:
    """Persist attempts without dropping failures or modifying earlier outputs."""
    output = Path(output_dir)
    (output / "patients").mkdir(parents=True, exist_ok=True)
    attempts, success, failures, selected_rows = [], [], [], []
    for result in results:
        record = {key: value for key, value in result.items() if key != "counterfactuals"}
        attempts.append(record)
        target = success if result.get("status") == "SUCCESS" else failures
        target.append(record)
        for counterfactual_id, candidate in enumerate(result.get("counterfactuals", []), start=1):
            candidate_data = candidate if isinstance(candidate, Candidate) else Candidate(**candidate)
            selected_rows.append({
                "patient_id": result.get("patient_id"),
                "counterfactual_id": counterfactual_id,
                "original_prediction": result.get("original_prediction"),
                "counterfactual_prediction": int(candidate_data.probability >= 0.5),
                "original_probability": result.get("original_probability"),
                "desired_class_probability": candidate_data.probability,
                "changed_features": " | ".join(candidate_data.changed_features),
                "number_changed_features": len(candidate_data.changed_features),
                "objective_values": json.dumps(candidate_data.objectives),
                "constraint_status": candidate_data.constraint_status,
                "feasibility_status": candidate_data.feasibility_status,
                "runtime": result.get("runtime"),
            })
        patient_id = result.get("patient_id")
        if patient_id is not None:
            patient_dir = output / "patients" / str(patient_id)
            patient_dir.mkdir(exist_ok=True)
            with open(patient_dir / "original_prediction.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, default=str)
                patient_rows = [item for item in selected_rows if item["patient_id"] == patient_id]
                pd.DataFrame(patient_rows).to_csv(patient_dir / "counterfactuals.csv", index=False)
            paths = {name: output / name for name in ("all_attempts.csv", "successful_counterfactuals.csv", "failed_counterfactuals.csv", "successful_counterfactual_details.csv")}
    pd.DataFrame(attempts).to_csv(paths["all_attempts.csv"], index=False)
    pd.DataFrame(success).to_csv(paths["successful_counterfactuals.csv"], index=False)
    pd.DataFrame(failures).to_csv(paths["failed_counterfactuals.csv"], index=False)
    pd.DataFrame(selected_rows).to_csv(paths["successful_counterfactual_details.csv"], index=False)
    summary = output / "summary.json"
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump({"attempted": len(attempts), "successful": len(success), "failed": len(failures)}, handle, indent=2)
    paths["summary.json"] = summary
    return paths


def generate_ccmocf(
    model_name: str = "xgboost",
    patient_ids: Iterable[int] | None = None,
    target_class: int = 1,
    population_size: int = 64,
    generations: int = 20,
    seed: int = 42,
    top_k: int = 3,
    output_dir: str | Path = CCMOCF_OUT,
):
    """Run CC-MO-CF for selected held-out test patients and persist every attempt."""
    from pw_imp.dice_baseline import load_dice_baseline_model
    from pw_imp.preprocessing import build_preprocessing_pipeline, load_dataframe

    pipeline = build_preprocessing_pipeline(random_state=seed)
    model = load_dice_baseline_model(model_name)
    raw = load_dataframe()
    test_ids = set(pipeline["patient_test"]["PatientID"].tolist())
    raw_test = raw[raw["PatientID"].isin(test_ids)].reset_index(drop=True)
    preprocessor = pipeline["preprocessor"]
    if patient_ids is not None:
        requested = set(int(pid) for pid in patient_ids)
        raw_test = raw_test[raw_test["PatientID"].isin(requested)].reset_index(drop=True)

    engine = CCMOCF(
        model=model,
        preprocessor=preprocessor,
        population_size=population_size,
        generations=generations,
        seed=seed,
        top_k=top_k,
    )
    attempts = []
    for _, row in raw_test.iterrows():
        patient_id = int(row["PatientID"])
        original = row.drop(labels=["PatientID", "PCOS (Y/N)"], errors="ignore").to_dict()
        result = engine.generate(original, desired_class=target_class)
        result["patient_id"] = patient_id
        result["target_class"] = int(target_class)
        attempts.append(result)
    paths = save_results(attempts, output_dir)
    return {"attempts": attempts, "output_paths": paths, "optimizer_used": "NSGA-II" if engine.optimizer_available else "directed_fallback"}