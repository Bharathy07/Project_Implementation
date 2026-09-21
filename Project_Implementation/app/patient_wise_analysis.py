"""Patient-wise PMOS prediction, explanation, and counterfactual analysis.

Run with: ``python -m streamlit run app/patient_wise_analysis.py``.
The page reads the original workbook record by PatientID and uses the saved models
with the project's fitted-on-training preprocessing procedure.  It never creates
synthetic patient records or relabels an optimisation failure as infeasibility.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pw_imp.ccmocf import CCMOCF
from pw_imp.clinical_constraints import ClinicalConstraintEngine
from pw_imp.preprocessing import build_preprocessing_pipeline, load_dataframe


TARGET = "PCOS (Y/N)"


def label(value: int) -> str:
    return "PMOS" if int(value) else "Non-PMOS"


def numeric(value: Any) -> float | None:
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def clinical_unit(feature: str) -> str:
    units = {
        "Age (yrs)": "years",
        "Weight (Kg)": "kg",
        "Height(Cm)": "cm",
        "BMI": "",
        "Cycle length(days)": "days",
        "Follicle No. (L)": "count",
        "Follicle No. (R)": "count",
        "Avg. F size (L) (mm)": "mm",
    }
    return units.get(feature, "")


def clinical_value(value: Any, feature: str) -> Any:
    if pd.isna(value):
        return value
    unit = clinical_unit(feature)
    if unit:
        return f"{value} {unit}"
    return value


def clinical_changed_rows(original: dict[str, Any], candidate: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in original:
        before, after = original.get(feature), candidate.get(feature)
        changed = bool((pd.isna(before) != pd.isna(after)) or (not pd.isna(before) and before != after))
        if changed:
            rows.append({
                "Feature": feature,
                "Original Value": clinical_value(before, feature),
                "Suggested Value": clinical_value(after, feature),
                "Unit": clinical_unit(feature),
                "Changed?": "Yes",
            })
    return pd.DataFrame(rows)


def available_features(original: dict[str, Any], terms: tuple[str, ...]) -> list[str]:
    return [name for name in original if any(term.lower() in name.lower() for term in terms)]


def show_clinical_context(original: dict[str, Any], engine: ClinicalConstraintEngine) -> None:
    st.header("Clinical Interpretation Context")
    st.info("ML prediction only - clinical diagnosis requires clinician assessment and appropriate diagnostic criteria.")
    st.write({
        "Model Prediction": "Shown above from the trained model",
        "Clinical Diagnosis": "Not established by this application",
        "Diagnostic framework": "PMOS diagnosis commonly considers 2 of 3 domains: ovulatory dysfunction, hyperandrogenism, and polycystic ovarian morphology.",
    })
    diagnostic_rows = [
        {"Diagnostic domain": "Ovulatory dysfunction", "Dataset evidence": ", ".join(available_features(original, ("Cycle",))) or "Unavailable", "Interpretation": "Diagnostic criterion cannot be independently assessed from the available dataset."},
        {"Diagnostic domain": "Hyperandrogenism", "Dataset evidence": ", ".join(available_features(original, ("hair", "Skin", "Pimples", "FSH", "LH", "AMH"))) or "Unavailable", "Interpretation": "Diagnostic criterion cannot be independently assessed from the available dataset."},
        {"Diagnostic domain": "Polycystic ovarian morphology", "Dataset evidence": ", ".join(available_features(original, ("Follicle", "F size"))) or "Unavailable", "Interpretation": "Ultrasound morphology is displayed as observed data only; it is not converted into an automatic diagnosis."},
    ]
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
    st.warning("Exclusion conditions are not assessed by this model because the required clinical variables are not available. This includes thyroid disease, hyperprolactinemia, non-classical congenital adrenal hyperplasia, Cushing's syndrome, and androgen-secreting tumors. No absence of these conditions is assumed.")

    st.subheader("Clinical Constraint Layers")
    layer_rows = [
        {"Layer": "Diagnostic constraints", "Status": "Context only; no clinical diagnosis is produced"},
        {"Layer": "Immutable features", "Status": f"{sum(bool(spec.get('immutable')) for spec in engine.config.values())} configured; Age and PatientID locked"},
        {"Layer": "Non-actionable features", "Status": f"{sum(not spec.get('actionable', False) and not spec.get('immutable', False) and spec.get('type') != 'derived' for spec in engine.config.values())} configured"},
        {"Layer": "Modifiable/actionable features", "Status": ", ".join(name for name, spec in engine.config.items() if spec.get('actionable') and name in original) or "None represented for this patient"},
        {"Layer": "Derived-feature dependencies", "Status": ", ".join(name for name, spec in engine.config.items() if spec.get('type') == 'derived') or "None configured"},
        {"Layer": "Clinical plausibility", "Status": "Configured bounds, maximum changes, directions, and categorical rules"},
        {"Layer": "Counterfactual optimization", "Status": "NSGA-II/CC-MO-CF objectives with hard clinical validation"},
    ]
    st.dataframe(pd.DataFrame(layer_rows), use_container_width=True, hide_index=True)

    st.subheader("Clinically Modifiable Factors Not Represented in Current Model")
    unavailable = [
        ("Dietary pattern", ("diet", "fiber", "protein", "glycemic"), "Dietary pattern is clinically modifiable, but it is not represented as an editable feature in the current prediction dataset."),
        ("Physical activity", ("activity", "exercise", "steps"), "Physical activity is clinically modifiable, but it is not represented as an editable feature in the current prediction dataset."),
        ("Sleep", ("sleep",), "Sleep is clinically modifiable, but it is not represented as an editable feature in the current prediction dataset."),
        ("Environmental exposure", ("bpa", "phthalate", "exposure", "plastic"), "Environmental exposure is clinically relevant, but it is not represented as an input feature of the current trained model and cannot be evaluated as a model counterfactual."),
        ("Medication or treatment", ("medication", "treatment", "letrozole", "clomiphene"), "Treatment selection requires clinician assessment and treatment variables are not counterfactual actions in this model."),
        ("Metabolic surveillance", ("glucose", "hba1c", "lipid", "cholesterol", "insulin"), "Metabolic surveillance variables are not available in the current dataset."),
        ("Fertility outcomes", ("fertility", "pregnancy", "infertility", "treatment response"), "Fertility outcomes are not modeled and must not be inferred from PMOS prediction."),
    ]
    unavailable_rows = []
    for domain, terms, explanation in unavailable:
        represented = available_features(original, terms)
        if not represented:
            unavailable_rows.append({"Factor": domain, "Status": "Not represented", "Explanation": explanation})
    st.dataframe(pd.DataFrame(unavailable_rows), use_container_width=True, hide_index=True)
    st.caption("Observed dataset fields such as cycle values, ultrasound measurements, symptoms, and fast-food indicators remain patient context or non-actionable inputs according to the configuration. They are not treatment recommendations.")

    st.subheader("Clinical Safety Note")
    st.warning("This system provides machine-learning predictions and computational counterfactual analysis. Counterfactual changes are hypothetical model scenarios and are not medical treatment recommendations, diagnostic confirmation, or personalized clinical advice. Clinical diagnosis and treatment decisions require evaluation by a qualified healthcare professional. SHAP is not causality; a counterfactual is not treatment; model prediction is not clinical diagnosis; and model probability is not actual patient risk.")


@st.cache_resource(show_spinner="Loading the saved models and preprocessing pipeline…")
def runtime() -> tuple[dict, Any, Any, float]:
    pipeline = build_preprocessing_pipeline(random_state=42)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(ROOT / "results" / "models" / "xgboost_model.json"))
    lightgbm_model = joblib.load(ROOT / "results" / "models" / "lightgbm_model.joblib")
    config = json.loads((ROOT / "results" / "ensemble_config.json").read_text(encoding="utf-8"))
    return pipeline, xgb_model, lightgbm_model, float(config["alpha"])


@st.cache_data(show_spinner=False)
def patient_record(patient_id: int) -> pd.DataFrame:
    data = load_dataframe()
    return data.loc[pd.to_numeric(data["PatientID"], errors="coerce").eq(patient_id)].copy()


def model_input(pipeline: dict, raw_features: dict[str, Any]) -> pd.DataFrame:
    transformed = pipeline["preprocessor"].transform(pd.DataFrame([raw_features]))
    return transformed.reindex(columns=pipeline["X_train"].columns).fillna(0)


def probabilities(pipeline: dict, xgb_model: Any, lightgbm_model: Any, alpha: float, raw: dict[str, Any]) -> dict[str, float]:
    X = model_input(pipeline, raw)
    xp = float(xgb_model.predict_proba(X)[:, 1][0])
    lp = float(lightgbm_model.predict_proba(X)[:, 1][0])
    return {"XGBoost": xp, "LightGBM": lp, "Hybrid Ensemble": alpha * xp + (1 - alpha) * lp}


def _normalise_shap_values(raw: Any, feature_count: int) -> np.ndarray:
    """Normalize SHAP's binary-class output without truncating mismatched data."""
    values = getattr(raw, "values", raw)
    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP returned an empty list")
        class_values = values[1] if len(values) > 1 else values[0]
        array = np.asarray(class_values)
    else:
        array = np.asarray(values)

    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"SHAP sample dimension {array.shape[0]} does not match one patient")
        if array.shape[2] < 2:
            raise ValueError(f"SHAP class dimension {array.shape[2]} does not contain a PMOS class")
        array = array[0, :, 1]
    elif array.ndim == 2:
        if array.shape == (1, feature_count):
            array = array[0]
        elif array.shape == (feature_count, 2):
            array = array[:, 1]
        elif array.shape[0] == 2 and array.shape[1] == feature_count:
            array = array[1]
        else:
            raise ValueError(f"SHAP 2D shape {array.shape} does not match one patient and {feature_count} features")
    elif array.ndim != 1:
        raise ValueError(f"SHAP returned unsupported shape {array.shape}")

    if len(array) != feature_count:
        raise ValueError(f"SHAP feature count {len(array)} does not match model input feature count {feature_count}")
    return np.asarray(array, dtype=float)


def shap_values(xgb_model: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    import shap

    explainer = shap.TreeExplainer(xgb_model.get_booster())
    values = _normalise_shap_values(explainer.shap_values(X), len(X.columns))
    base = explainer.expected_value
    base = float(np.asarray(base).reshape(-1)[-1])
    return values, base


def shap_patient_value(feature: str, original: dict[str, Any]) -> Any:
    """Map encoded model features back to an observed patient field when possible."""
    if feature in original:
        return original[feature]
    source = feature.rsplit("_", 1)[0] if "_" in feature else feature
    return original.get(source, "Unavailable")


def show_dice_baseline(
    pipeline: dict,
    xgb_model: Any,
    original: dict[str, Any],
    original_probability: float,
    original_prediction: int,
    target: int,
) -> dict[str, Any]:
    """Render the existing DiCE baseline without inventing raw clinical values."""
    st.header("5. DiCE Baseline Counterfactuals")
    dice_frame, dice_error = dice_for_patient(
        pipeline["X_test"], xgb_model, model_input(pipeline, original), target
    )
    if dice_error:
        st.error("DICE GENERATION FAILED")
        return {"status": "failed", "frame": pd.DataFrame(), "error": dice_error}
    if dice_frame.empty:
        st.info("NO DICE COUNTERFACTUAL FOUND")
        return {"status": "none", "frame": dice_frame, "error": None}

    rows = []
    for number, (_, candidate) in enumerate(dice_frame.iterrows(), 1):
        candidate_input = candidate.reindex(model_input(pipeline, original).columns).to_frame().T
        probability = float(xgb_model.predict_proba(candidate_input)[:, 1][0])
        changed = [name for name in candidate_input.columns if candidate_input.iloc[0][name] != model_input(pipeline, original).iloc[0][name]]
        target_achieved = int(probability >= 0.5) == target
        rows.append({
            "#": number,
            "Target Prediction": label(int(probability >= 0.5)),
            "Target Probability": f"{probability:.2%}",
            "Changed Features": " | ".join(changed) or "None",
            "Number of Changes": len(changed),
            "Constraint Status": "NOT APPLIED",
            "Final Status": "TARGET ACHIEVED" if target_achieved else "TARGET NOT ACHIEVED",
        })
    st.success("DICE COUNTERFACTUAL FOUND")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("DiCE: Original vs Counterfactual PMOS Probability")
    dice_probability = float(xgb_model.predict_proba(dice_frame.iloc[[0]].reindex(columns=model_input(pipeline, original).columns))[:, 1][0])
    st.bar_chart(pd.DataFrame({"Original": [original_probability * 100], "DiCE": [dice_probability * 100]}, index=["Probability (%)"]))
    st.caption("DiCE candidates are displayed in the existing model feature space. Original clinical values are retained; no inverse transformation is assumed where the pipeline does not provide one.")
    return {"status": "generated", "frame": dice_frame, "error": None, "rows": rows}


def changed_rows(original: dict[str, Any], candidate: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in original:
        before, after = original.get(feature), candidate.get(feature)
        changed = bool((pd.isna(before) != pd.isna(after)) or (not pd.isna(before) and before != after))
        rows.append({"Feature": feature, "Original Value": before, "Counterfactual Value": after, "Changed?": "Yes" if changed else "No"})
    return pd.DataFrame(rows)


def changed_feature_names(original: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    return [
        feature for feature in original
        if bool((pd.isna(original.get(feature)) != pd.isna(candidate.get(feature))) or (
            not pd.isna(original.get(feature)) and original.get(feature) != candidate.get(feature)
        ))
    ]


def dice_for_patient(X_reference: pd.DataFrame, xgb_model: Any, row: pd.DataFrame, desired: int) -> tuple[pd.DataFrame, str | None]:
    """Run the existing DiCE library against the same XGBoost feature space."""
    try:
        import dice_ml
        outcome = np.zeros(len(X_reference), dtype=int)
        data = dice_ml.Data(dataframe=X_reference.assign(_outcome=outcome), continuous_features=X_reference.columns.tolist(), outcome_name="_outcome")
        model = dice_ml.Model(model=xgb_model, backend="sklearn", model_type="classifier")
        result = dice_ml.Dice(data, model, method="random").generate_counterfactuals(
            row, total_CFs=3, desired_class=int(desired),
            permitted_range={name: [float(X_reference[name].min()), float(X_reference[name].max())] for name in X_reference.columns},
        )
        frame = result.cf_examples_list[0].final_cfs_df
        return (frame.copy() if frame is not None else pd.DataFrame()), None
    except Exception as exc:  # DiCE availability/runtime is a real reported outcome.
        return pd.DataFrame(), str(exc)


def constraint_table(engine: ClinicalConstraintEngine, original: dict[str, Any], candidate: dict[str, Any], target: int, probability: float, require_target: bool = True) -> pd.DataFrame:
    changed = changed_feature_names(original, candidate)
    immutable_pass = engine.immutable_features_unchanged(original, candidate)
    actionable_pass = all(
        bool(engine.get_feature_spec(name).get("actionable", False))
        or str(engine.get_feature_spec(name).get("type", "")).lower() == "derived"
        for name in changed
    )
    feature_change_pass = True
    validation_error = ""
    for name in changed:
        if str(engine.get_feature_spec(name).get("type", "")).lower() == "derived":
            continue
        try:
            engine.validate_feature_change(name, original.get(name), candidate.get(name))
        except (ValueError, TypeError) as exc:
            feature_change_pass = False
            validation_error = str(exc)
    try:
        if changed:
            engine.validate_candidate(original, candidate)
        clinical = "✅ PASS"
    except Exception as exc:
        clinical = f"❌ FAIL: {exc}"
        validation_error = str(exc)
    dependency = "✅ PASS"
    changed_set = set(changed)
    for name, spec in engine.config.items():
        if str(spec.get("type", "")).lower() != "derived" or name not in candidate:
            continue
        dependencies = set(engine._dependency_names_for_feature(name))
        if name not in changed_set and not dependencies.intersection(changed_set):
            continue
        try:
            engine.validate_candidate(original, candidate)
        except Exception as exc:
            dependency = f"❌ FAIL: {exc}"
            validation_error = str(exc)
            break
    target_pass = int(probability >= .5) == target
    complete_pass = (target_pass or not require_target) and immutable_pass and actionable_pass and feature_change_pass and clinical == "✅ PASS" and dependency == "✅ PASS"
    return pd.DataFrame([
        {"Constraint": "Target prediction", "Status": "PASS" if target_pass else "FAIL", "Reason": f"Probability: {probability:.2%}"},
        {"Constraint": "Immutable features", "Status": "PASS" if immutable_pass else "FAIL", "Reason": "Age, PatientID, and configured immutable values are unchanged"},
        {"Constraint": "Actionability", "Status": "PASS" if actionable_pass else "FAIL", "Reason": "Only configured actionable features changed"},
        {"Constraint": "Clinical constraints", "Status": "PASS" if feature_change_pass and clinical == "✅ PASS" else "FAIL", "Reason": validation_error or "Configured bounds, maximum changes, directions, and categorical rules passed"},
        {"Constraint": "Dependency consistency", "Status": "PASS" if dependency == "✅ PASS" else "FAIL", "Reason": "Derived features were recalculated and validated"},
        {"Constraint": "Overall feasibility", "Status": "FEASIBLE" if complete_pass else "INFEASIBLE", "Reason": "All hard constraints passed" if complete_pass else "At least one hard constraint failed"},
    ])


def before_after_constraint_audit(
    engine: ClinicalConstraintEngine,
    original: dict[str, Any],
    candidate: dict[str, Any],
) -> pd.DataFrame:
    """Audit one raw model-flipping row without projecting or repairing it."""
    changed = changed_feature_names(original, candidate)
    rows = [{
        "Constraint": "Immutable features",
        "Status": "PASS" if engine.immutable_features_unchanged(original, candidate) else "FAIL",
        "Reason": "Age, PatientID, and configured immutable values are unchanged",
    }]
    for feature_name in changed:
        spec = engine.get_feature_spec(feature_name)
        if not spec:
            rows.append({"Constraint": f"{feature_name} constraint definition", "Status": "FAIL", "Reason": "Actionable feature has no constraint definition."})
            continue
        if str(spec.get("type", "")).lower() == "derived":
            rows.append({"Constraint": f"{feature_name} dependency", "Status": "FAIL", "Reason": "Derived feature cannot be independently changed."})
            continue
        if not bool(spec.get("actionable", False)):
            rows.append({"Constraint": f"{feature_name} actionability", "Status": "FAIL", "Reason": "Changed feature is non-actionable."})
            continue
        try:
            engine.validate_feature_change(feature_name, original.get(feature_name), candidate.get(feature_name))
            rows.append({"Constraint": f"{feature_name} constraints", "Status": "PASS", "Reason": "Bounds, maximum change, direction, and categorical validity passed."})
        except (ValueError, TypeError) as exc:
            rows.append({"Constraint": f"{feature_name} constraints", "Status": "FAIL", "Reason": str(exc)})

    changed_set = set(changed)
    for feature_name, spec in engine.config.items():
        if str(spec.get("type", "")).lower() != "derived" or feature_name not in candidate:
            continue
        dependencies = set(engine._dependency_names_for_feature(feature_name))
        if feature_name not in changed_set and not dependencies.intersection(changed_set):
            continue
        try:
            engine._validate_dependency_consistency(feature_name, original, candidate)
            rows.append({"Constraint": f"{feature_name} dependency", "Status": "PASS", "Reason": "Derived-feature dependency is consistent."})
        except (ValueError, TypeError) as exc:
            rows.append({"Constraint": f"{feature_name} dependency", "Status": "FAIL", "Reason": str(exc)})

    rows.append({
        "Constraint": "Overall",
        "Status": "PASS" if all(row["Status"] == "PASS" for row in rows) else "FAIL",
        "Reason": "All applicable configured constraints passed" if all(row["Status"] == "PASS" for row in rows) else "Candidate rejected by one or more configured constraints",
    })
    return pd.DataFrame(rows)


def show_before_after_constraints(
    engine: ClinicalConstraintEngine,
    original: dict[str, Any],
    original_probability: float,
    original_prediction: int,
    model_flip_candidates: list[dict[str, Any]],
) -> None:
    st.subheader("Counterfactual Before vs After Clinical Constraints")
    if not model_flip_candidates:
        st.info("No model-flipping counterfactual was generated by the existing optimizer.")
        return
    st.caption("Before clinical constraints, the counterfactual is evaluated only according to whether the trained model changes its prediction. After clinical constraints, the same counterfactual is checked against the configured clinical rules, immutable features, actionability, bounds, and dependencies.")
    st.caption("MODEL FLIP DOES NOT EQUAL FEASIBLE CLINICAL COUNTERFACTUAL")
    for number, item in enumerate(model_flip_candidates[:5], 1):
        candidate = item["row"]
        probability = float(item["probability"])
        changes = clinical_changed_rows(original, candidate).drop(columns=["Changed?"], errors="ignore")
        if not changes.empty:
            changes = changes.rename(columns={"Suggested Value": "Counterfactual Value"})
            changes["Change"] = [
                f"{candidate[feature] - original[feature]:+g}"
                if numeric(original[feature]) is not None and numeric(candidate[feature]) is not None
                else "changed"
                for feature in changed_feature_names(original, candidate)
            ]
        audit = before_after_constraint_audit(engine, original, candidate)
        feasible = audit.loc[audit["Constraint"].eq("Overall"), "Status"].iloc[0] == "PASS"
        st.subheader(f"Candidate {number}")
        st.markdown("**Before Applying Clinical Constraints**")
        st.dataframe(changes, use_container_width=True, hide_index=True)
        st.write({
            "Original Prediction": label(original_prediction),
            "Original PMOS Probability": f"{original_probability:.2%}",
            "Counterfactual Prediction": label(int(item["prediction"])),
            "Counterfactual PMOS Probability": f"{probability:.2%}",
            "Probability Change": f"{probability - original_probability:+.2%}",
            "Prediction Flip": "YES",
            "Clinical Constraint Validation": "NOT YET APPLIED",
            "Status": "MODEL FLIP FOUND",
        })
        st.markdown("**After Applying Clinical Constraints**")
        st.dataframe(changes, use_container_width=True, hide_index=True)
        st.write({
            "Original Prediction": label(original_prediction),
            "Counterfactual Prediction": label(int(item["prediction"])),
            "Original PMOS Probability": f"{original_probability:.2%}",
            "Counterfactual PMOS Probability": f"{probability:.2%}",
            "Prediction Flip": "YES",
            "Clinical Constraints": "PASS" if feasible else "FAIL",
            "Overall Feasibility": "FEASIBLE" if feasible else "INFEASIBLE",
        })
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.markdown("**Counterfactual Status Comparison**")
        st.dataframe(pd.DataFrame([
            {"": "Prediction", "Before Constraints": label(int(item["prediction"])), "After Constraints": label(int(item["prediction"]))},
            {"": "PMOS Probability", "Before Constraints": f"{probability:.2%}", "After Constraints": f"{probability:.2%}"},
            {"": "Prediction Flip", "Before Constraints": "YES", "After Constraints": "YES"},
            {"": "Clinical Validation", "Before Constraints": "Not applied", "After Constraints": "PASS" if feasible else "FAIL"},
            {"": "Feasible", "Before Constraints": "Not evaluated", "After Constraints": "YES" if feasible else "NO"},
        ]), use_container_width=True, hide_index=True)


def show_feasible_card(number: int, engine: ClinicalConstraintEngine, original: dict[str, Any], candidate: Any, target: int, original_prob: float) -> None:
    candidate_probability = float(candidate.probability)
    checks = constraint_table(engine, original, candidate.row, target, candidate_probability)
    target_achieved = int(candidate_probability >= .5) == target
    feasible = target_achieved and checks.iloc[-1]["Status"] == "FEASIBLE" and candidate.feasibility_status == "VALID_FEASIBLE"
    if not feasible:
        return
    st.success(f"Feasible Counterfactual {number}")
    st.dataframe(clinical_changed_rows(original, candidate.row).drop(columns=["Changed?"], errors="ignore"), use_container_width=True, hide_index=True)
    st.write({
        "Original PMOS Probability": f"{original_prob:.2%}",
        "New PMOS Probability": f"{candidate_probability:.2%}",
        "Probability Change": f"{candidate_probability - original_prob:+.2%}",
        "Prediction": f"{label(int(original_prob >= .5))} → {label(int(candidate_probability >= .5))}",
        "Clinical Constraints": "PASS",
        "Overall Feasibility": "FEASIBLE",
    })
    st.write({
        "Passed constraints": checks.loc[
            (checks["Status"] == "PASS") & (checks["Constraint"] != "Target prediction"),
            "Constraint",
        ].tolist(),
    })


def show_ccmocf(
    pipeline: dict,
    xgb_model: Any,
    original: dict[str, Any],
    original_prob: float,
    original_prediction: int,
    desired: int,
    patient_id: int,
) -> dict:
    engine = CCMOCF(xgb_model, pipeline["preprocessor"], ClinicalConstraintEngine(), population_size=64, generations=20, seed=42, top_k=3)
    result = engine.generate(original, desired_class=desired)
    candidates = result.get("counterfactuals", [])
    evaluated = result.get("evaluated_candidates", candidates)
    model_flip_candidates = result.get("model_flip_candidates", [])
    feasible_candidates = []
    for candidate in evaluated:
        checks = constraint_table(engine.constraints, original, candidate.row, desired, candidate.probability)
        flip = int(candidate.probability >= .5) == desired
        passed = bool(
            flip
            and checks.loc[checks["Constraint"].eq("Overall feasibility"), "Status"].iloc[0] == "FEASIBLE"
            and candidate.constraint_status == "VALID"
            and candidate.feasibility_status == "VALID_FEASIBLE"
        )
        if passed:
            feasible_candidates.append(candidate)
    show_before_after_constraints(engine, original, original_prob, int(result.get("original_prediction", original_prob >= 0.5)), model_flip_candidates)
    st.header(f"6. Feasible Counterfactuals — Patient {patient_id}")
    st.write({
        "Current prediction": label(original_prediction),
        "Current probability": f"{original_prob:.2%}",
        "Target prediction": label(desired),
    })
    st.subheader("CC-MO-CF Decision")
    status = result.get("status", "OPTIMIZER_ERROR")
    if status == "OPTIMIZER_ERROR":
        decision = "OPTIMIZATION FAILED"
        st.error(decision)
    elif not evaluated:
        decision = "NO CANDIDATES GENERATED"
        st.warning(decision)
    elif feasible_candidates:
        decision = "FEASIBLE PREDICTION FLIP"
        st.success(decision)
    elif model_flip_candidates:
        decision = "TARGET ACHIEVED BUT NO FEASIBLE SOLUTION"
        st.warning(decision)
    else:
        decision = "NO FEASIBLE PREDICTION FLIP"
        st.warning(decision)
    flips = len(model_flip_candidates)
    st.write({"Candidates tested": result.get("candidates_evaluated", 0), "Prediction flips": flips, "Feasible solutions": len(feasible_candidates)})
    audit_records = result.get("candidate_audit", [])
    if audit_records:
        with st.expander("Candidate Evaluation Details", expanded=False):
            detail_rows = []
            for record in audit_records:
                failures = record.get("failed_constraints", [])
                categories = {failure.get("constraint") for failure in failures}
                detail_rows.append({
                    "Candidate": record.get("candidate_id"),
                    "Target Achieved": "PASS" if record.get("prediction_flip") else "FAIL",
                    "Target Probability": record.get("probability"),
                    "Changed Features": " | ".join(record.get("changed_features", [])) or "None",
                    "Actionability": "FAIL" if "non_actionable" in categories or "constraint_definition" in categories else "PASS",
                    "Bounds": "FAIL" if "bounds" in categories else "PASS",
                    "Max Change": "FAIL" if "maximum_change" in categories else "PASS",
                    "Dependency": "FAIL" if "dependency_consistency" in categories else "PASS",
                    "Clinical Constraints": "FAIL" if failures else "PASS",
                    "Final Status": "FEASIBLE" if record.get("constraint_valid") and record.get("prediction_flip") else "REJECTED",
                    "Exact Reason": " | ".join(f"{failure['constraint']}: {failure['reason']}" for failure in failures) or "Target not achieved" if not record.get("prediction_flip") else "All applicable checks passed",
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    if feasible_candidates:
        for number, candidate in enumerate(feasible_candidates, 1):
            show_feasible_card(number, engine.constraints, original, candidate, desired, original_prob)
    else:
        st.warning(f"No feasible counterfactual found for Patient {patient_id}.")
        tested = int(result.get("candidates_evaluated", 0))
        st.write({
            "Candidates tested": tested,
            "Prediction flips": flips,
            "Feasible counterfactuals": 0,
        })
        if flips == 0:
            st.info("No feasible counterfactual was generated because none of the tested candidates achieved the target prediction.")
        else:
            st.info("Target prediction was achieved, but all target-achieving candidates failed one or more configured clinical constraints.")
    return {"result": result, "decision": decision, "feasible": len(feasible_candidates), "evaluated": evaluated, "engine": engine}


def show_counterfactual_decision_diagram(
    original_prediction: int,
    target_prediction: int,
    candidates_tested: int,
    prediction_flips: int,
    feasible_solutions: int,
    decision: str,
) -> None:
    """Show a runtime-derived counterfactual decision flow."""
    st.subheader("Counterfactual Decision Flow")
    nodes = [
        f"Original prediction\n{label(original_prediction)}",
        f"Counterfactual search\n{candidates_tested} candidates",
        f"Target achieved?\n{prediction_flips} flips\nTarget: {label(target_prediction)}",
        f"Clinical constraints?\n{feasible_solutions} feasible",
        f"Final decision\n{decision}",
    ]
    fig, axis = plt.subplots(figsize=(12, 2.4))
    axis.set_xlim(0, len(nodes))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, node in enumerate(nodes):
        x_position = index + 0.5
        axis.text(
            x_position,
            0.5,
            node,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#eef3f8", "edgecolor": "#54708c"},
        )
        if index < len(nodes) - 1:
            axis.annotate("", xy=(x_position + 0.38, 0.5), xytext=(x_position + 0.12, 0.5), arrowprops={"arrowstyle": "->", "color": "#54708c", "lw": 1.5})
    st.pyplot(fig)
    plt.close(fig)


def show_what_if(
    pipeline: dict,
    xgb_model: Any,
    lightgbm_model: Any,
    alpha: float,
    original: dict[str, Any],
    original_prob: float,
    original_prediction: int,
    engine: ClinicalConstraintEngine,
) -> None:
    st.header("8. What-If Analysis")
    actionable = engine.get_counterfactual_features(original)
    if not actionable:
        st.info("No actionable features are configured for what-if analysis.")
        return

    feature = st.selectbox("Actionable feature", actionable)
    spec = engine.get_feature_spec(feature)
    current = numeric(original[feature])
    if current is None:
        st.error(f"What-if analysis is unavailable because {feature} is not numeric.")
        return
    max_step = float(spec.get("max_step") or 0.0)
    lower = max(float(spec["lower_bound"]) if spec.get("lower_bound") is not None else current - max_step, current - max_step)
    upper = min(float(spec["upper_bound"]) if spec.get("upper_bound") is not None else current + max_step, current + max_step)
    direction = str(spec.get("allowed_direction") or "both").lower()
    if direction == "increase":
        lower = current
    elif direction == "decrease":
        upper = current
    if lower > upper:
        st.error(f"No valid configured what-if range exists for {feature}.")
        return
    selected = st.number_input(
        f"{feature} ({clinical_unit(feature) or 'clinical units'})",
        min_value=float(lower),
        max_value=float(upper),
        value=float(current),
        step=1.0 if current.is_integer() else 0.1,
    )
    no_change = engine._values_equal(current, selected)
    if no_change:
        st.info("NO CHANGE")
    candidate_input = dict(original)
    candidate_input[feature] = selected
    audit = {"valid": True, "failures": []}
    try:
        if no_change:
            candidate = dict(original)
        else:
            candidate = engine.project_candidate(original, candidate_input)
        audit = engine.audit_candidate(original, candidate)
        if not audit["valid"]:
            reason = "; ".join(item["reason"] for item in audit["failures"])
            raise ValueError(reason)
        engine.validate_candidate(original, candidate)
    except (ValueError, TypeError) as exc:
        candidate = dict(candidate_input)
        audit = {"valid": False, "failures": [{"reason": str(exc)}]}
    try:
        what_if_prob = original_prob if no_change else probabilities(pipeline, xgb_model, lightgbm_model, alpha, candidate)["Hybrid Ensemble"]
    except Exception as exc:
        st.error(f"Prediction evaluation failed: {exc}")
        return
    what_if_prediction = int(what_if_prob >= .5)
    target_achieved = what_if_prediction == (1 - original_prediction)
    feasible = bool(target_achieved and audit["valid"])
    changed = "NO CHANGE" if no_change else feature
    table = pd.DataFrame([{
        "Scenario": "Scenario 1",
        "Feature Changed": changed,
        "Original": clinical_value(current, feature),
        "Suggested": clinical_value(selected, feature),
        "Original PMOS Probability": f"{original_prob:.2%}",
        "New PMOS Probability": f"{what_if_prob:.2%}",
        "Probability Change": f"{what_if_prob - original_prob:+.2%} pp",
        "Prediction": f"{label(original_prediction)} → {label(what_if_prediction)}",
        "Target Achieved": "YES" if target_achieved else "NO",
        "Feasible": "YES" if feasible else "NO",
    }])
    st.dataframe(table, use_container_width=True, hide_index=True)
    if not audit["valid"]:
        st.caption("Clinical feasibility: FAIL. " + "; ".join(item["reason"] for item in audit["failures"]))
    elif not no_change:
        st.caption("Clinical feasibility: PASS. Feasible is YES only when the configured target prediction is also achieved.")

    if table.empty or not audit["valid"]:
        st.info("No What-If scenarios available.")
        return

    plot_frame = table.copy()
    plot_frame["Original Probability"] = original_prob * 100.0
    plot_frame["What-If Probability"] = what_if_prob * 100.0
    plot_frame["Probability Change (pp)"] = what_if_prob * 100.0 - original_prob * 100.0

    st.subheader("Patient-wise What-If PMOS Probability")
    probability_chart = plot_frame.set_index("Scenario")[["Original Probability", "What-If Probability"]]
    st.bar_chart(probability_chart, y_label="PMOS Probability (%)")

    st.subheader("Change in PMOS Probability")
    change_chart = plot_frame.set_index("Scenario")[["Probability Change (pp)"]]
    st.bar_chart(change_chart, y_label="Probability Change (percentage points)")
    st.caption("Positive value = PMOS probability increased; negative value = PMOS probability decreased; zero = no change.")


def main() -> None:
    st.set_page_config(page_title="PMOS Patient Analysis", layout="wide")
    st.title("PMOS Patient-wise Prediction & Counterfactual Analysis")
    st.caption("Enter Patient No. → Get Complete Analysis. Outputs are model-based and are not medical advice or treatment recommendations.")
    requested = st.text_input("Patient No. / PatientID", placeholder="248")
    if not requested.strip():
        return
    try:
        patient_id = int(requested.strip())
    except ValueError:
        st.error("Patient ID not found. Please enter a valid Patient ID.")
        return
    record = patient_record(patient_id)
    if record.empty:
        st.error("Patient ID not found. Please enter a valid Patient ID.")
        return
    try:
        pipeline, xgb_model, lightgbm_model, alpha = runtime()
    except Exception as exc:
        st.error(f"The saved pipeline/models could not be loaded: {exc}")
        return
    row = record.iloc[0]
    original = row.drop(labels=["PatientID", TARGET], errors="ignore").to_dict()
    st.header("1. Patient Information")
    patient_values = {"Patient ID": patient_id}
    for feature in ("Age (yrs)", "Weight (Kg)", "BMI", "Cycle length(days)", "Height(Cm)", "FSH(mIU/mL)", "LH(mIU/mL)", "AMH(ng/mL)", "Follicle No. (L)", "Follicle No. (R)"):
        if feature in record:
            patient_values[feature] = record.iloc[0][feature]
    st.dataframe(pd.DataFrame({"Feature": list(patient_values), "Observed value": list(patient_values.values())}), use_container_width=True, hide_index=True)
    processed = model_input(pipeline, original).iloc[0]
    probs = probabilities(pipeline, xgb_model, lightgbm_model, alpha, original)
    prediction_table = pd.DataFrame([{"Model": name, "Prediction": label(value >= .5), "PMOS Probability": value} for name, value in probs.items()])
    st.header("2. Final PMOS Prediction")
    final_probability = probs["Hybrid Ensemble"]
    final_prediction = int(final_probability >= .5)
    st.success(f"Prediction: {label(final_prediction)}")
    st.metric("PMOS Probability", f"{final_probability:.2%}")
    st.write("Model: Hybrid Ensemble")
    st.subheader("3. Model Agreement")
    st.dataframe(pd.DataFrame([{"Model": name, "Prediction": label(value >= .5), "PMOS Probability": f"{value:.2%}"} for name, value in probs.items()]), use_container_width=True, hide_index=True)
    clinical_engine = ClinicalConstraintEngine()
    desired = 1 - final_prediction
    st.header("4. SHAP Explanation")
    shap_error = None
    shap_frame = pd.DataFrame()
    try:
        values, base = shap_values(xgb_model, processed.to_frame().T)
        order = np.argsort(np.abs(values))[::-1]
        shap_frame = pd.DataFrame({
            "Feature": processed.index[order],
            "Patient Value": [shap_patient_value(name, original) for name in processed.index[order]],
            "SHAP Value": values[order],
        })
        shap_frame["Effect"] = np.where(shap_frame["SHAP Value"] >= 0, "Toward PMOS", "Away from PMOS")
        st.write({"Patient ID": patient_id, "Final PMOS Prediction": label(final_prediction), "PMOS Probability": f"{probs['XGBoost']:.2%}", "Model": "XGBoost", "Target class": label(desired)})
        st.write("Features contributing toward PMOS")
        st.dataframe(shap_frame[shap_frame["SHAP Value"] > 0][["Feature", "Patient Value", "SHAP Value", "Effect"]].head(10), use_container_width=True, hide_index=True)
        st.write("Features contributing away from PMOS")
        st.dataframe(shap_frame[shap_frame["SHAP Value"] < 0][["Feature", "Patient Value", "SHAP Value", "Effect"]].head(10), use_container_width=True, hide_index=True)
        st.subheader("Patient-Level SHAP Contributions")
        top = shap_frame.head(10).sort_values("SHAP Value")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top["Feature"], top["SHAP Value"], color=np.where(top["SHAP Value"] >= 0, "#c94c4c", "#3b72a8"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title("Patient-Level SHAP Contributions")
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("SHAP Waterfall — Patient-Level Prediction")
        waterfall = shap_frame.head(10).sort_values("SHAP Value")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(waterfall["Feature"], waterfall["SHAP Value"], color=np.where(waterfall["SHAP Value"] >= 0, "#c94c4c", "#3b72a8"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Base value {base:.4f} + top patient contributions")
        ax.set_xlabel("Contribution to model output")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        shap_error = str(exc)
        st.warning("SHAP explanation is unavailable for this patient.")
    st.caption("SHAP values explain the model's prediction and should not be interpreted as causal or medical evidence.")
    dice = show_dice_baseline(pipeline, xgb_model, original, probs["XGBoost"], final_prediction, desired)
    cf = show_ccmocf(pipeline, xgb_model, original, probs["XGBoost"], final_prediction, desired, patient_id)


if __name__ == "__main__":
    main()
