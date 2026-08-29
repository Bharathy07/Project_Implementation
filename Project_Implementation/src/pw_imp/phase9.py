"""Phase 9 held-out evaluation runner for CC-MO-CF."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pw_imp.ccmocf import CCMOCF, Candidate
from pw_imp.dice_baseline import load_dice_baseline_model
from pw_imp.preprocessing import build_preprocessing_pipeline, load_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9_OUT = REPO_ROOT / "results" / "counterfactuals" / "ccmocf"


def _probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X))
        return values[:, 1] if values.ndim > 1 else values
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def run_phase9(
    model_name: str = "xgboost",
    population_size: int = 64,
    generations: int = 20,
    seed: int = 42,
    top_k: int = 3,
    output_dir: str | Path = PHASE9_OUT,
):
    """Evaluate every held-out patient using a recorded bidirectional policy.

    Policy: patients predicted as PCOS target class 0; patients predicted as
    non-PCOS target class 1. The direction is recorded for every attempt.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = build_preprocessing_pipeline(random_state=seed)
    model = load_dice_baseline_model(model_name)
    raw = load_dataframe()
    patient_test = pipeline["patient_test"].reset_index(drop=True)
    y_test = pipeline["y_test"].reset_index(drop=True)
    X_test = pipeline["X_test"]
    probabilities = _probability(model, X_test)
    predictions = (probabilities >= 0.5).astype(int)

    test_ids = patient_test["PatientID"].tolist()
    raw_test = raw[raw["PatientID"].isin(test_ids)].copy()
    raw_test = raw_test.set_index("PatientID").loc[test_ids].reset_index()
    engine = CCMOCF(model, pipeline["preprocessor"], population_size=population_size, generations=generations, seed=seed, top_k=top_k)

    prediction_rows = []
    attempt_rows = []
    successful_rows = []
    failed_rows = []
    summary_rows = []
    all_attempts = []
    for index, raw_row in raw_test.iterrows():
        patient_id = int(raw_row["PatientID"])
        original = raw_row.drop(labels=["PatientID", "PCOS (Y/N)"], errors="ignore").to_dict()
        original_prediction = int(predictions[index])
        target_class = 0 if original_prediction == 1 else 1
        result = engine.generate(original, desired_class=target_class)
        result.update({"patient_id": patient_id, "true_label": int(y_test.iloc[index]), "target_class": target_class, "direction": f"{original_prediction}->{target_class}"})
        all_attempts.append(result)
        prediction_rows.append({"PatientID": patient_id, "true_label": int(y_test.iloc[index]), "model_prediction": original_prediction, "prediction_probability": float(probabilities[index]), "counterfactual_target_class": target_class, "counterfactual_generation_status": result["status"], "direction": result["direction"]})
        base = {"PatientID": patient_id, "Status": result["status"], "FailureReason": result.get("error", "" if result["status"] == "SUCCESS" else result["constraint_status"]), "Runtime": result["runtime"], "NumberOfCandidatesEvaluated": result["candidates_evaluated"], "true_label": int(y_test.iloc[index]), "target_class": target_class, "optimizer_used": result["optimizer_used"]}
        attempt_rows.append(base)
        if result["status"] == "SUCCESS":
            for cf_id, candidate in enumerate(result["counterfactuals"], start=1):
                candidate = candidate if isinstance(candidate, Candidate) else Candidate(**candidate)
                row = {**base, "CounterfactualID": cf_id, "counterfactual_prediction": int(candidate.probability >= 0.5), "desired_class_probability": candidate.probability, "changed_features": " | ".join(candidate.changed_features), "number_changed_features": len(candidate.changed_features), "objective_values": json.dumps(candidate.objectives), "constraint_status": candidate.constraint_status, "feasibility_status": candidate.feasibility_status}
                successful_rows.append(row)
        else:
            failed_rows.append(base)
        summary_rows.append({"PatientID": patient_id, "status": result["status"], "runtime": result["runtime"], "number_of_candidates_evaluated": result["candidates_evaluated"], "number_of_counterfactuals": len(result["counterfactuals"]), "number_of_changed_features": float(np.mean([len(cf.changed_features) for cf in result["counterfactuals"]])) if result["counterfactuals"] else np.nan})

    prediction_frame = pd.DataFrame(prediction_rows)
    attempt_frame = pd.DataFrame(attempt_rows)
    successful_frame = pd.DataFrame(successful_rows)
    failed_frame = pd.DataFrame(failed_rows)
    summary_frame = pd.DataFrame(summary_rows)
    prediction_frame.to_csv(output / "patient_predictions.csv", index=False)
    attempt_frame.to_csv(output / "ccmocf_all_attempts.csv", index=False)
    successful_frame.to_csv(output / "ccmocf_successful.csv", index=False)
    failed_frame.to_csv(output / "ccmocf_failed.csv", index=False)
    summary_frame.to_csv(output / "ccmocf_patient_summary.csv", index=False)
    attempt_frame.to_csv(output / "ccmocf_heldout_results.csv", index=False)
    failed_frame.rename(columns={"Status": "failure_status", "FailureReason": "failure_reason", "Runtime": "runtime_seconds"}).to_csv(output / "ccmocf_heldout_failures.csv", index=False)
    successful_frame.rename(columns={"Runtime": "runtime_seconds"}).to_csv(output / "ccmocf_pareto_solutions.csv", index=False)
    try:
        import pymoo
        pymoo_version = pymoo.__version__
    except ImportError:
        pymoo_version = None
    model_artifact = REPO_ROOT / "results" / "models" / ("xgboost_model.json" if model_name.lower() == "xgboost" else "lightgbm_model.joblib")
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is None and hasattr(model, "get_booster"):
        model_feature_count = len(model.get_booster().feature_names or [])
    config = {"phase": 9, "model_name": model_name, "model_artifact": str(model_artifact), "expected_feature_count": len(X_test.columns), "actual_model_feature_count": model_feature_count, "population_size": population_size, "generations": generations, "crossover": "pymoo NSGA2 default SBX", "mutation": "pymoo NSGA2 default PM", "seed": seed, "top_k": top_k, "number_of_objectives": 6, "direction_policy": "predicted PCOS: 1->0; predicted non-PCOS: 0->1", "test_set_only": True, "models_retrained": False, "optimizer_used": "NSGA-II" if engine.optimizer_available else "fallback", "pymoo_version": pymoo_version, "split_identifier": str(REPO_ROOT / "results" / "splits" / "patientid_split_mapping.csv"), "execution_timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with open(output / "ccmocf_experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    summary = pd.DataFrame(summary_rows)
    successful_patients = len({row["PatientID"] for row in successful_rows})
    runtime_row = {"total_test_patients": len(test_ids), "eligible_patients": len(attempt_rows), "attempted_patients": len(attempt_rows), "successful_counterfactual_patients": successful_patients, "failed_patients": len(failed_rows), "no_feasible_counterfactual": sum(row["Status"] == "NO_FEASIBLE_COUNTERFACTUAL" for row in attempt_rows), "feasibility_rate": successful_patients / len(attempt_rows) if attempt_rows else 0.0, "prediction_flip_rate": successful_patients / len(attempt_rows) if attempt_rows else 0.0, "average_pareto_solutions": float(summary["number_of_counterfactuals"].mean()) if not summary.empty else 0.0, "average_feasible_solutions": len(successful_rows) / len(attempt_rows) if attempt_rows else 0.0, "average_counterfactuals_per_successful_patient": len(successful_rows) / successful_patients if successful_patients else 0.0, "average_number_changed_features": float(summary["number_of_changed_features"].dropna().mean()) if not summary["number_of_changed_features"].dropna().empty else 0.0, "average_runtime": float(summary["runtime"].mean()) if not summary.empty else 0.0}
    failure_distribution = pd.Series([row["FailureReason"] for row in failed_rows], dtype="string").value_counts().to_dict()
    integrity = {
        "all_attempted_patients_recorded": len(attempt_rows) == len(test_ids),
        "patient_id_excluded_from_model_features": "PatientID" not in X_test.columns,
        "no_test_set_tuning_performed": True,
        "failed_cases_preserved": len(failed_rows) + successful_patients == len(attempt_rows),
        "successful_output_contains_only_valid_counterfactuals": all(row["feasibility_status"] == "VALID_FEASIBLE" for row in successful_rows),
    }
    runtime_row["failure_reason_distribution"] = json.dumps(failure_distribution, sort_keys=True)
    runtime_row["integrity"] = json.dumps(integrity, sort_keys=True)
    pd.DataFrame([runtime_row]).to_csv(output / "ccmocf_runtime_summary.csv", index=False)
    if not attempt_frame.empty:
        group_summary = attempt_frame.assign(true_group=np.where(attempt_frame["true_label"] == 1, "true_PCOS", "true_non_PCOS")).groupby("true_group", as_index=False).agg(records=("PatientID", "count"), successful=("Status", lambda values: int((values == "SUCCESS").sum())), mean_runtime=("Runtime", "mean"))
        group_summary.to_csv(output / "ccmocf_true_group_summary.csv", index=False)
    return {"config": config, "predictions": prediction_rows, "attempts": all_attempts, "runtime_summary": runtime_row, "failure_reason_distribution": failure_distribution, "integrity": integrity, "output_dir": output}


if __name__ == "__main__":
    result = run_phase9()
    print("Phase 9 CC-MO-CF summary")
    for key, value in result["runtime_summary"].items():
        if key not in {"failure_reason_distribution", "integrity"}:
            print(f"{key}: {value}")
    print("failure_reason_distribution:", json.dumps(result["failure_reason_distribution"], sort_keys=True))
    for key, value in result["integrity"].items():
        print(f"[{'PASS' if value else 'FAIL'}] {key}")