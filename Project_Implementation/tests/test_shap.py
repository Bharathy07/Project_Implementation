import sys
from pathlib import Path

import pandas as pd

# ensure src is on path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

from pw_imp.preprocessing import build_preprocessing_pipeline
from pw_imp.shap_explainer import SHAP_OUT, generate_shap_explanations


def test_shap_outputs_created():
    result = generate_shap_explanations(model_name="xgboost", num_patients=3)
    assert result["model_name"] == "xgboost"
    expected = [
        "global_feature_importance.csv",
        "shap_bar_plot.png",
        "shap_beeswarm.png",
        "patient_explanations.csv",
    ]
    for name in expected:
        assert (SHAP_OUT / name).exists(), f"Missing SHAP output: {name}"

    patient_summary = pd.read_csv(SHAP_OUT / "patient_explanations.csv")
    assert "PatientID" in patient_summary.columns
    assert patient_summary["PatientID"].notna().all()
    assert len(patient_summary) >= 1


def test_patientid_preserved_in_explanations():
    pipeline = build_preprocessing_pipeline(random_state=2)
    X_test = pipeline["X_test"]
    patient_test = pipeline["patient_test"]
    assert "PatientID" not in X_test.columns
    assert "PatientID" in patient_test.columns

    result = generate_shap_explanations(model_name="xgboost", num_patients=2)
    patient_summary = result["patient_summary"]
    assert set(patient_summary["PatientID"]).issubset(set(patient_test["PatientID"]))
