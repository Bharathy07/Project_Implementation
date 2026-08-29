"""Standard DiCE counterfactual baseline for the tabular PCOS model.

This module is intentionally a controlled baseline wrapper around dice-ml and does not
implement the full CC-MO-CF method. It uses the same model and data splits as the main
experiment, preserves PatientID, and records honest failure cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

try:
    import dice_ml
except Exception as exc:  # pragma: no cover
    dice_ml = None
    DICE_IMPORT_ERROR = exc
else:
    DICE_IMPORT_ERROR = None

from pw_imp.preprocessing import build_preprocessing_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DICE_OUT = REPO_ROOT / "results" / "counterfactuals" / "dice"
DICE_OUT.mkdir(parents=True, exist_ok=True)


def _ensure_dice_available():
    if dice_ml is None:
        raise ImportError(f"dice-ml could not be imported: {DICE_IMPORT_ERROR}")


def load_dice_baseline_model(model_name: str = "xgboost"):
    """Return the same final selected model used for the main experiment."""
    from pw_imp.models import MODELS_OUT

    if model_name.lower() == "xgboost":
        import xgboost as xgb
        model_path = MODELS_OUT / "xgboost_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        return model

    if model_name.lower() == "lightgbm":
        from pw_imp.models import MODELS_OUT
        import joblib
        path = MODELS_OUT / "lightgbm_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"LightGBM model not found at {path}")
        return joblib.load(path)

    raise ValueError(f"Unsupported model: {model_name}")


def generate_dice_counterfactuals(
    model_name: str = "xgboost",
    patient_ids: Optional[Iterable[int]] = None,
    target_class: int = 1,
    num_counterfactuals: int = 3,
    max_patients: int = 3,
):
    """Generate a small controlled standard DiCE counterfactual set for held-out test patients."""
    _ensure_dice_available()

    pipeline = build_preprocessing_pipeline(random_state=2)
    X_test = pipeline["X_test"].copy()
    patient_test = pipeline["patient_test"].copy().reset_index(drop=True)
    y_test = pipeline["y_test"].reset_index(drop=True)

    model = load_dice_baseline_model(model_name)

    if hasattr(model, "predict_proba"):
        pred_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict"):
        pred_scores = model.predict(X_test)
    else:
        pred_scores = model.predict(X_test)

    if patient_ids is None:
        patient_ids = patient_test.iloc[np.argsort(pred_scores)[::-1][:max_patients]]["PatientID"].tolist()

    patient_ids = [int(pid) for pid in patient_ids]
    selected = patient_test[patient_test["PatientID"].isin(patient_ids)].copy().reset_index(drop=True)
    selected["ActualLabel"] = y_test.iloc[selected.index].to_numpy()

    results = []
    failures = []
    features = X_test.columns.tolist()
    continuous_features = [c for c in features if X_test[c].dtype.kind in {"i", "f"}]

    # Build a standard DiCE data interface using the held-out test data only as the data reference.
    data_interface = dice_ml.Data(
        dataframe=X_test.assign(y=y_test).copy(),
        continuous_features=continuous_features,
        outcome_name="y",
    )

    model_interface = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    explainer = dice_ml.Dice(data_interface, model_interface, method="random")

    for pid in selected["PatientID"].tolist():
        patient_index = int(patient_test[patient_test["PatientID"] == pid].index[0])
        row = X_test.iloc[[patient_index]].copy()
        try:
            explanation = explainer.generate_counterfactuals(
                row,
                total_CFs=num_counterfactuals,
                desired_class=target_class,
                permitted_range={feat: [float(X_test[feat].min()), float(X_test[feat].max())] for feat in continuous_features},
            )
            cf_df = explanation.cf_examples_list[0].final_cfs_df
            if cf_df is not None and not cf_df.empty:
                cf_df.insert(0, "PatientID", pid)
                cf_path = DICE_OUT / f"dice_patient_{pid}.csv"
                cf_df.to_csv(cf_path, index=False)
                n_cf = len(cf_df)
                status = "generated"
            else:
                n_cf = 0
                status = "no_counterfactuals"
                failures.append({"PatientID": pid, "reason": "no counterfactuals generated"})
            results.append({
                "PatientID": pid,
                "status": status,
                "n_counterfactuals": n_cf,
                "target_class": int(target_class),
            })
        except Exception as exc:
            failures.append({"PatientID": pid, "reason": str(exc)})
            results.append({"PatientID": pid, "status": "failed", "reason": str(exc)})

    pd.DataFrame(results).to_csv(DICE_OUT / "dice_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(DICE_OUT / "dice_failures.csv", index=False)
    return {"results": results, "failures": failures, "output_dir": str(DICE_OUT)}


if __name__ == "__main__":
    print(generate_dice_counterfactuals(model_name="xgboost", max_patients=2))
