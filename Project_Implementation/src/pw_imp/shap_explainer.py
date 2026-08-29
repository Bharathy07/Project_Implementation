"""SHAP explainability for the final selected tree-based PCOS model.

This module preserves PatientID metadata and explains the final XGBoost tree model
using TreeSHAP, while keeping the feature matrix free of the identifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from pw_imp.preprocessing import build_preprocessing_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_OUT = REPO_ROOT / "results" / "models"
SHAP_OUT = REPO_ROOT / "results" / "figures" / "shap"
SHAP_OUT.mkdir(parents=True, exist_ok=True)


def _load_xgboost_model(model_path: Path | None = None):
    path = model_path or MODELS_OUT / "xgboost_model.json"
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {path}")
    model = xgb.Booster()
    model.load_model(str(path))
    return model


def _load_lightgbm_model(model_path: Path | None = None):
    path = model_path or MODELS_OUT / "lightgbm_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"LightGBM model not found at {path}")
    return joblib.load(path)


def load_selected_tree_model(model_name: str = "xgboost"):
    """Return the final tree model selected for SHAP explanation.

    Default is XGBoost because TreeSHAP is directly supported and it is the tree-based
    model most commonly used in the phase-3 experiments.
    """
    model_name = (model_name or "xgboost").lower()
    if model_name == "xgboost":
        return _load_xgboost_model(), "xgboost"
    if model_name == "lightgbm":
        return _load_lightgbm_model(), "lightgbm"
    raise ValueError(f"Unsupported tree model: {model_name}")


def _extract_positive_class_values(raw_values):
    if isinstance(raw_values, list):
        arr = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])
    else:
        arr = np.asarray(raw_values)

    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    return arr


def _extract_base_value(raw_base):
    if isinstance(raw_base, (list, tuple, np.ndarray)):
        if len(raw_base) == 0:
            return 0.0
        return raw_base[1] if len(raw_base) > 1 else raw_base[0]
    return raw_base


def _make_explanation(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    values = _extract_positive_class_values(raw)
    base_values = _extract_base_value(explainer.expected_value)
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X.to_numpy(),
        feature_names=X.columns.tolist(),
    )
    return explanation, explainer


def _save_figure(fig_path: Path, fig):
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_force_plot(explanation, patient_index: int, output_path: Path):
    try:
        force_plot = shap.plots.force(explanation[patient_index], matplotlib=False, show=False)
        if hasattr(force_plot, "data"):
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(force_plot.data)
            return True
        if hasattr(force_plot, "_repr_html_"):
            html_text = force_plot._repr_html_()
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(html_text)
            return True
    except Exception:
        return False
    return False


def generate_shap_explanations(
    model_name: str = "xgboost",
    selected_patient_ids: Iterable[int] | None = None,
    num_patients: int = 10,
):
    """Generate global and individual SHAP explanations for the selected tree model.

    The patient metadata from the test split is preserved separately from the model
    feature matrix. The output directory is results/figures/shap/.
    """
    pipeline = build_preprocessing_pipeline(random_state=2)
    X_test = pipeline["X_test"].copy()
    y_test = pipeline["y_test"].reset_index(drop=True)
    patient_test = pipeline["patient_test"].reset_index(drop=True).copy()

    model, selected_model_name = load_selected_tree_model(model_name)

    explanation, explainer = _make_explanation(model, X_test)
    shap_values = explanation.values

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(len(X_test), len(X_test.columns))

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict") and not isinstance(model, xgb.Booster):
        probas = model.predict(X_test)
    else:
        dtest = xgb.DMatrix(X_test)
        probas = model.predict(dtest)

    probas = np.asarray(probas, dtype=float)
    if probas.ndim > 1 and probas.shape[1] > 1:
        probas = probas[:, 1]

    patient_test["ActualLabel"] = y_test.astype(int).to_numpy()
    patient_test["PredictedLabel"] = (probas >= 0.5).astype(int)
    patient_test["PCOSProbability"] = probas.astype(float)

    # Save global feature importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = pd.DataFrame({
        "feature": X_test.columns.tolist(),
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_importance.to_csv(SHAP_OUT / "global_feature_importance.csv", index=False)

    # Save global SHAP plots
    plt.figure(figsize=(10, 7), dpi=300)
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.savefig(SHAP_OUT / "shap_bar_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8), dpi=300)
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.savefig(SHAP_OUT / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    if selected_patient_ids is None:
        top_idx = np.argsort(probas)[::-1][:num_patients]
        selected_patient_ids = patient_test.iloc[top_idx]["PatientID"].tolist()

    selected_patient_ids = [int(pid) for pid in selected_patient_ids]
    selected_patient_df = patient_test[patient_test["PatientID"].isin(selected_patient_ids)].copy()
    selected_patient_df = selected_patient_df.sort_values("PCOSProbability", ascending=False).reset_index(drop=True)

    records = []
    for idx, row in selected_patient_df.iterrows():
        patient_id = int(row["PatientID"])
        row_index = patient_test[patient_test["PatientID"] == patient_id].index[0]

        local_values = shap_values[row_index]
        top_idx = np.argsort(np.abs(local_values))[::-1][:5]
        top_features = [X_test.columns[j] for j in top_idx]
        contribution_dirs = ["positive" if local_values[j] > 0 else "negative" for j in top_idx]
        magnitudes = [float(abs(local_values[j])) for j in top_idx]

        records.append({
            "PatientID": patient_id,
            "ActualLabel": int(row["ActualLabel"]),
            "PredictedLabel": int(row["PredictedLabel"]),
            "PCOSProbability": float(row["PCOSProbability"]),
            "TopContributingFeatures": " | ".join(top_features),
            "ContributionDirection": " | ".join(contribution_dirs),
            "ContributionMagnitude": " | ".join(f"{val:.4f}" for val in magnitudes),
        })

        local_explanation = shap.Explanation(
            values=np.asarray(local_values).reshape(1, -1),
            base_values=np.asarray([float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else float(explainer.expected_value)]),
            data=X_test.iloc[[row_index]].to_numpy(),
            feature_names=X_test.columns.tolist(),
        )

        plt.figure(figsize=(12, 8), dpi=300)
        shap.plots.waterfall(local_explanation[0], max_display=10, show=False)
        plt.savefig(SHAP_OUT / f"waterfall_patient_{patient_id}.png", dpi=300, bbox_inches="tight")
        plt.close()

        force_path = SHAP_OUT / f"force_patient_{patient_id}.html"
        _safe_force_plot(local_explanation, 0, force_path)

    patient_summary = pd.DataFrame(records)
    patient_summary.to_csv(SHAP_OUT / "patient_explanations.csv", index=False)

    return {
        "model_name": selected_model_name,
        "global_feature_importance": global_importance,
        "patient_summary": patient_summary,
        "shap_output_dir": SHAP_OUT,
    }


if __name__ == "__main__":
    result = generate_shap_explanations(model_name="xgboost", num_patients=10)
    print(f"Generated SHAP outputs for {result['model_name']} at {result['shap_output_dir']}")
