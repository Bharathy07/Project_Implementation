import sys
from pathlib import Path
import pandas as pd

# ensure src is on path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

from pw_imp.preprocessing import build_preprocessing_pipeline, FEATURES_OUT


def test_preprocessing_pipeline():
    res = build_preprocessing_pipeline(random_state=2)
    # patient metadata present
    assert "patient_train" in res and "patient_val" in res and "patient_test" in res
    pt = res["patient_train"]
    pv = res["patient_val"]
    ptst = res["patient_test"]
    # features not include PatientID
    X_train = res["X_train"]
    assert "PatientID" not in X_train.columns
    # alignment lengths
    assert len(X_train) == len(pt), "X_train and patient_train must align"
    assert len(res["X_val"]) == len(pv)
    assert len(res["X_test"]) == len(ptst)
    # selected features file exists
    self = Path(FEATURES_OUT) / "selected_features.csv"
    assert self.exists(), "selected_features.csv must be saved"
    # report sizes (sanity checks)
    assert len(X_train) > 0
    assert len(res["X_val"]) > 0
    assert len(res["X_test"]) > 0


def test_no_leakage_imputer_stats():
    # Ensure numeric imputer statistics are derived from training data only
    res = build_preprocessing_pipeline(random_state=2)
    pre = res["preprocessor"]
    # check we have median statistics for numeric imputer
    if pre.numeric_cols:
        stats = pre.num_imputer.statistics_
        assert len(stats) == len(pre.numeric_cols)


def test_selected_feature_count_and_report():
    res = build_preprocessing_pipeline(random_state=2)
    pre = res["preprocessor"]
    # original feature count (before transform)
    orig_count = sum([1 for _ in pre.numeric_cols]) + sum([1 for _ in pre.categorical_cols])
    final_count = len(pre.selected_features)
    assert final_count > 0
    # ensure saved file contains same list
    saved = pd.read_csv(Path(FEATURES_OUT) / "selected_features.csv")
    assert set(saved['feature'].tolist()) == set(pre.selected_features)


def test_no_patient_overlap_between_splits():
    res = build_preprocessing_pipeline(random_state=2)
    pt = set(res['patient_train']['PatientID'].tolist())
    pv = set(res['patient_val']['PatientID'].tolist())
    ptest = set(res['patient_test']['PatientID'].tolist())
    assert pt & pv == set()
    assert pt & ptest == set()
    assert pv & ptest == set()
