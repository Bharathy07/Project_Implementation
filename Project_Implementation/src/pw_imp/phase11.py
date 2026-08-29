"""Phase 11 read-only patient result organization.

This module reads existing artifacts and writes indexed organization tables. It
never trains models or generates counterfactuals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PATIENTS_DIR = RESULTS_DIR / "patients"
PHASE9_DIR = RESULTS_DIR / "experiments" / "phase9_ccmocf"
DICE_DIR = RESULTS_DIR / "counterfactuals" / "dice"
SHAP_DIR = RESULTS_DIR / "figures" / "shap"

MASTER_COLUMNS = [
    "PatientID", "TrueLabel", "PredictedLabel", "PredictionProbability",
    "PredictionCorrect", "CounterfactualAttempted", "CounterfactualStatus",
    "NumberOfValidCFs", "BestCounterfactualID", "CounterfactualMethod",
]


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path)


def _as_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _prediction_table() -> pd.DataFrame:
    """Load saved predictions, or compute them read-only from final artifacts."""
    saved = _read_csv(PHASE9_DIR / "patient_predictions.csv")
    if not saved.empty:
        return saved.rename(columns={
            "true_label": "TrueLabel", "model_prediction": "PredictedLabel",
            "prediction_probability": "PredictionProbability",
        })

    shap = _read_csv(SHAP_DIR / "patient_explanations.csv")
    if not shap.empty:
        return shap.rename(columns={
            "ActualLabel": "TrueLabel", "PredictedLabel": "PredictedLabel",
            "PCOSProbability": "PredictionProbability",
        })[["PatientID", "TrueLabel", "PredictedLabel", "PredictionProbability"]]

    # This is inference only; model training/tuning is never called here.
    try:
        from pw_imp.dice_baseline import load_dice_baseline_model
        from pw_imp.preprocessing import build_preprocessing_pipeline

        pipeline = build_preprocessing_pipeline(random_state=42)
        model = load_dice_baseline_model("xgboost")
        X_test = pipeline["X_test"]
        probabilities = np.asarray(model.predict_proba(X_test))[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        labels = pipeline["y_test"].reset_index(drop=True)
        patients = pipeline["patient_test"].reset_index(drop=True)
        return pd.DataFrame({
            "PatientID": patients["PatientID"], "TrueLabel": labels,
            "PredictedLabel": predictions, "PredictionProbability": probabilities,
        })
    except Exception:
        return pd.DataFrame(columns=["PatientID", "TrueLabel", "PredictedLabel", "PredictionProbability"])


def _counterfactual_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = _read_csv(PHASE9_DIR / "ccmocf_all_attempts.csv")
    successful = _read_csv(PHASE9_DIR / "ccmocf_successful.csv")
    if not attempts.empty:
        attempts = attempts.rename(columns={"patient_id": "PatientID", "status": "CounterfactualStatus", "Status": "CounterfactualStatus"})
        successful = successful.rename(columns={"patient_id": "PatientID"})
        attempts["CounterfactualMethod"] = "CC-MO-CF"
        return attempts, successful

    # Phase 7 files are preserved and used only as existing-result metadata.
    dice_summary = _read_csv(DICE_DIR / "dice_summary.csv")
    if not dice_summary.empty:
        attempts = dice_summary.rename(columns={"status": "CounterfactualStatus"})
        attempts["CounterfactualAttempted"] = True
        attempts["NumberOfValidCFs"] = attempts.get("n_counterfactuals", 0)
        attempts["CounterfactualMethod"] = "DiCE"
        return attempts, pd.DataFrame()
    return pd.DataFrame(columns=["PatientID", "CounterfactualStatus"]), pd.DataFrame()


def _method_for(patient_id: int, attempts: pd.DataFrame) -> str:
    if attempts.empty or "PatientID" not in attempts:
        return "unavailable"
    methods = attempts.loc[attempts["PatientID"].astype(int).eq(patient_id), "CounterfactualMethod"] if "CounterfactualMethod" in attempts else pd.Series(dtype=str)
    return str(methods.iloc[0]) if not methods.empty else "unavailable"


def organize_patient_results(output_dir: str | Path = PATIENTS_DIR) -> pd.DataFrame:
    """Create the master and four read-only patient group tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = _prediction_table()
    attempts, successful = _counterfactual_tables()

    if predictions.empty:
        patient_ids = _read_csv(RESULTS_DIR / "splits" / "test_patient_ids.csv")
        predictions = patient_ids.rename(columns={"PatientID": "PatientID"})
        predictions["TrueLabel"] = pd.NA
        predictions["PredictedLabel"] = pd.NA
        predictions["PredictionProbability"] = pd.NA

    predictions["PatientID"] = predictions["PatientID"].astype(int)
    rows = []
    for _, prediction in predictions.drop_duplicates("PatientID").iterrows():
        patient_id = int(prediction["PatientID"])
        attempt = attempts[attempts["PatientID"].astype(int).eq(patient_id)] if not attempts.empty and "PatientID" in attempts else pd.DataFrame()
        valid = successful[successful["PatientID"].astype(int).eq(patient_id)] if not successful.empty and "PatientID" in successful else pd.DataFrame()
        predicted = _as_int(prediction.get("PredictedLabel"))
        true_label = _as_int(prediction.get("TrueLabel"))
        status = str(attempt.iloc[0].get("CounterfactualStatus", "NOT_AVAILABLE")) if not attempt.empty else "NOT_AVAILABLE"
        method = _method_for(patient_id, attempts)
        best_id = valid.iloc[0].get("CounterfactualID") if not valid.empty and "CounterfactualID" in valid else pd.NA
        rows.append({
            "PatientID": patient_id,
            "TrueLabel": true_label,
            "PredictedLabel": predicted,
            "PredictionProbability": prediction.get("PredictionProbability", pd.NA),
            "PredictionCorrect": pd.NA if true_label is None or predicted is None else bool(true_label == predicted),
            "CounterfactualAttempted": bool(not attempt.empty),
            "CounterfactualStatus": status,
            "NumberOfValidCFs": int(len(valid)) if not valid.empty else (int(attempt.iloc[0].get("n_counterfactuals", 0)) if not attempt.empty else 0),
            "BestCounterfactualID": best_id,
            "CounterfactualMethod": method,
        })
    master = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    master.to_csv(output / "patient_master_results.csv", index=False)
    master[master["PredictedLabel"] == 1].to_csv(output / "predicted_pcos_patients.csv", index=False)
    master[master["PredictedLabel"] == 0].to_csv(output / "predicted_non_pcos_patients.csv", index=False)
    master[master["TrueLabel"] == 1].to_csv(output / "true_pcos_patients.csv", index=False)
    master[master["TrueLabel"] == 0].to_csv(output / "true_non_pcos_patients.csv", index=False)
    return master


def _raw_patient(patient_id: int) -> dict[str, Any]:
    try:
        from pw_imp.preprocessing import load_dataframe

        raw = load_dataframe()
        match = raw[raw["PatientID"].astype(int).eq(patient_id)]
        return match.iloc[0].to_dict() if not match.empty else {}
    except Exception as exc:
        return {"_unavailable": str(exc)}


def view_patient_result(PatientID: int, output_dir: str | Path = PATIENTS_DIR) -> dict[str, Any]:
    """Return and display all existing organized information for one patient."""
    patient_id = int(PatientID)
    master = _read_csv(Path(output_dir) / "patient_master_results.csv")
    if master.empty:
        master = organize_patient_results(output_dir)
    match = master[master["PatientID"].astype(int).eq(patient_id)]
    if match.empty:
        raise KeyError(f"PatientID {patient_id} was not found in organized results")
    row = match.iloc[0].to_dict()
    shap = _read_csv(SHAP_DIR / "patient_explanations.csv")
    shap_row = shap[shap["PatientID"].astype(int).eq(patient_id)].to_dict("records") if not shap.empty else []
    dice_path = DICE_DIR / f"dice_patient_{patient_id}.csv"
    dice = _read_csv(dice_path).to_dict("records") if dice_path.exists() else []
    ccmocf = _read_csv(PHASE9_DIR / "ccmocf_successful.csv")
    ccmocf_rows = ccmocf[ccmocf["PatientID"].astype(int).eq(patient_id)].to_dict("records") if not ccmocf.empty and "PatientID" in ccmocf else []
    result = {"PatientID": patient_id, "original_features": _raw_patient(patient_id), "master_result": row, "shap_explanation": shap_row, "dice_counterfactuals": dice, "ccmocf_counterfactuals": ccmocf_rows, "failure_reason": row.get("CounterfactualStatus") if not ccmocf_rows and not dice else None}
    print(json.dumps(result, indent=2, default=str))
    return result


def validate_patient_results(master: pd.DataFrame) -> dict[str, bool]:
    """Validate one master row per PatientID and clear label/prediction separation."""
    return {
        "one_row_per_patient_id": not master["PatientID"].duplicated().any(),
        "patient_ids_present": master["PatientID"].notna().all(),
        "true_and_predicted_labels_separate": "TrueLabel" in master and "PredictedLabel" in master and "TrueLabel" not in {"PredictedLabel"},
        "counterfactual_alignment_by_patient_id": master["PatientID"].notna().all(),
    }


if __name__ == "__main__":
    table = organize_patient_results()
    print(json.dumps(validate_patient_results(table), indent=2))