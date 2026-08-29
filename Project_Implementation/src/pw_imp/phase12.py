"""Phase 12 full finalized held-out experiment orchestrator.

The runner loads finalized artifacts only. It never calls model training or
hyperparameter search. Component failures are recorded per patient rather than
silently removing patients from the final experiment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pw_imp.dice_baseline import generate_dice_counterfactuals, load_dice_baseline_model
from pw_imp.preprocessing import build_preprocessing_pipeline, load_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_OUT = REPO_ROOT / "results" / "final_experiment"


def _model_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X))
        return values[:, 1] if values.ndim > 1 else values
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _run_shap(model_name: str, patient_ids: list[int]) -> str:
    try:
        from pw_imp.shap_explainer import generate_shap_explanations

        generate_shap_explanations(model_name=model_name, selected_patient_ids=patient_ids, num_patients=len(patient_ids))
        return "SUCCESS"
    except Exception as exc:
        return f"FAILED: {exc}"


def _run_dice(model_name: str, patient_ids: list[int], target_class: int, num_counterfactuals: int) -> dict[str, Any]:
    try:
        result = generate_dice_counterfactuals(
            model_name=model_name,
            patient_ids=patient_ids,
            target_class=target_class,
            num_counterfactuals=num_counterfactuals,
            max_patients=len(patient_ids),
        )
        return {"status": "SUCCESS", "result": result}
    except Exception as exc:
        return {"status": f"FAILED: {exc}", "result": None}


def _run_ccmocf(
    model_name: str,
    population_size: int,
    generations: int,
    seed: int,
    top_k: int,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        from pw_imp.phase9 import run_phase9

        result = run_phase9(
            model_name=model_name,
            population_size=population_size,
            generations=generations,
            seed=seed,
            top_k=top_k,
            output_dir=output_dir,
        )
        return {"status": "SUCCESS", "result": result}
    except Exception as exc:
        return {"status": f"FAILED: {exc}", "result": None}


def run_phase12(
    model_name: str = "xgboost",
    seed: int = 42,
    population_size: int = 64,
    generations: int = 20,
    top_k: int = 3,
    dice_counterfactuals: int = 3,
    output_dir: str | Path = FINAL_OUT,
) -> dict[str, Any]:
    """Run the fixed Phase 12 pipeline on every held-out test patient."""
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": 12,
        "model_name": model_name,
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "top_k": top_k,
        "dice_counterfactuals": dice_counterfactuals,
        "held_out_test_set_only": True,
        "models_retrained": False,
        "hyperparameters_changed": False,
        "test_set_tuning": False,
        "counterfactual_direction_policy": "existing Phase 7 DiCE target policy is recorded; Phase 9 CC-MO-CF uses predicted-class reversal",
    }
    with open(output / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    pipeline = None
    model = None
    predictions = pd.DataFrame()
    setup_error = None
    try:
        pipeline = build_preprocessing_pipeline(random_state=seed)
        model = load_dice_baseline_model(model_name)
        probabilities = _model_probability(model, pipeline["X_test"])
        labels = pipeline["y_test"].reset_index(drop=True)
        patients = pipeline["patient_test"].reset_index(drop=True)
        predictions = pd.DataFrame({
            "PatientID": patients["PatientID"].astype(int),
            "TrueLabel": labels.astype(int),
            "PredictedLabel": (probabilities >= 0.5).astype(int),
            "PredictionProbability": probabilities.astype(float),
        })
    except Exception as exc:
        setup_error = str(exc)

    if predictions.empty:
        test_ids = _read(REPO_ROOT / "results" / "splits" / "test_patient_ids.csv")
        predictions = test_ids.rename(columns={"PatientID": "PatientID"})
        predictions["TrueLabel"] = pd.NA
        predictions["PredictedLabel"] = pd.NA
        predictions["PredictionProbability"] = pd.NA

    patient_ids = predictions["PatientID"].astype(int).tolist()
    shap_status = _run_shap(model_name, patient_ids) if setup_error is None else f"SKIPPED: {setup_error}"
    dice_result = _run_dice(model_name, patient_ids, target_class=1, num_counterfactuals=dice_counterfactuals) if setup_error is None else {"status": f"SKIPPED: {setup_error}", "result": None}
    ccmocf_dir = output / "phase9_ccmocf"
    ccmocf_result = _run_ccmocf(model_name, population_size, generations, seed, top_k, ccmocf_dir) if setup_error is None else {"status": f"SKIPPED: {setup_error}", "result": None}

    dice_summary = _read(REPO_ROOT / "results" / "counterfactuals" / "dice" / "dice_summary.csv")
    ccmocf_attempts = _read(ccmocf_dir / "ccmocf_all_attempts.csv")
    ccmocf_success = _read(ccmocf_dir / "ccmocf_successful.csv")
    patient_rows = []
    failures = []
    for _, patient in predictions.iterrows():
        patient_id = int(patient["PatientID"])
        ccmocf_patient = ccmocf_attempts[ccmocf_attempts["PatientID"].astype(int).eq(patient_id)] if not ccmocf_attempts.empty and "PatientID" in ccmocf_attempts else pd.DataFrame()
        dice_patient = dice_summary[dice_summary["PatientID"].astype(int).eq(patient_id)] if not dice_summary.empty else pd.DataFrame()
        status = str(ccmocf_patient.iloc[0].get("Status", "NOT_RUN")) if not ccmocf_patient.empty else "NOT_RUN"
        patient_rows.append({"PatientID": patient_id, "TrueLabel": patient.get("TrueLabel"), "PredictedLabel": patient.get("PredictedLabel"), "PredictionProbability": patient.get("PredictionProbability"), "SHAPStatus": shap_status, "DiCEStatus": "generated" if not dice_patient.empty and str(dice_patient.iloc[0].get("status", "")).lower() == "generated" else dice_result["status"], "CCMOCFStatus": status, "CCMOCFValidCount": int(len(ccmocf_success[ccmocf_success["PatientID"].astype(int).eq(patient_id)])) if not ccmocf_success.empty else 0, "DiCEValidCount": int(dice_patient.iloc[0].get("n_counterfactuals", 0)) if not dice_patient.empty else 0})
        for component, component_status in (("SHAP", shap_status), ("DiCE", dice_result["status"]), ("CC-MO-CF", ccmocf_result["status"])):
            if component_status.startswith("FAILED") or component_status.startswith("SKIPPED"):
                failures.append({"PatientID": patient_id, "Component": component, "FailureReason": component_status})

    all_counterfactuals = []
    if not ccmocf_success.empty:
        ccmocf_copy = ccmocf_success.copy()
        ccmocf_copy["Method"] = "CC-MO-CF"
        all_counterfactuals.append(ccmocf_copy)
    if not dice_summary.empty:
        for patient_id in dice_summary["PatientID"].astype(int):
            path = REPO_ROOT / "results" / "counterfactuals" / "dice" / f"dice_patient_{patient_id}.csv"
            if path.exists():
                dice_copy = pd.read_csv(path)
                dice_copy["Method"] = "DiCE"
                all_counterfactuals.append(dice_copy)
    pd.DataFrame(patient_rows).to_csv(output / "all_patient_results.csv", index=False)
    pd.concat(all_counterfactuals, ignore_index=True).to_csv(output / "all_counterfactual_results.csv", index=False) if all_counterfactuals else pd.DataFrame().to_csv(output / "all_counterfactual_results.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    predictions.assign(PredictionCorrect=predictions["TrueLabel"] == predictions["PredictedLabel"]).to_csv(output / "model_results.csv", index=False)
    summary = {"total_test_patients": len(predictions), "shap_status": shap_status, "dice_status": dice_result["status"], "ccmocf_status": ccmocf_result["status"], "patients_with_ccmocf_success": int(predictions["PatientID"].isin(ccmocf_success["PatientID"] if not ccmocf_success.empty else []).sum()), "failure_rows": len(failures), "runtime_seconds": time.perf_counter() - started, "no_ablation": True, "no_statistical_tests": True}
    with open(output / "experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return {"config": config, "summary": summary, "output_dir": output}


if __name__ == "__main__":
    result = run_phase12()
    print(json.dumps(result["summary"], indent=2))