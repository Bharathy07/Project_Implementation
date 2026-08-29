import sys
from pathlib import Path
import pandas as pd

# ensure src is on path so tests can import package
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

from pw_imp.splitter import load_raw_excel, create_patient_splits, get_feature_matrix, REPO_ROOT


def test_patientid_preserved():
    df = load_raw_excel()
    assert "PatientID" in df.columns, "PatientID column must exist after loading"


def test_patientid_unique():
    df = load_raw_excel()
    assert df["PatientID"].nunique() == len(df), "Each row should have a unique PatientID in this dataset"


def test_splits_no_overlap():
    import tempfile
    df = load_raw_excel()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "splits"
        splits = create_patient_splits(df=df, out_dir=out_dir, random_state=1, train_frac=0.6, val_frac=0.2)
        s_train = set(splits["train"])
        s_val = set(splits["validation"])
        s_test = set(splits["test"])
        assert s_train & s_val == set(), "Train and validation splits overlap"
        assert s_train & s_test == set(), "Train and test splits overlap"
        assert s_val & s_test == set(), "Validation and test splits overlap"
        # check files exist
        assert (out_dir / "train_patient_ids.csv").exists()
        assert (out_dir / "validation_patient_ids.csv").exists()
        assert (out_dir / "test_patient_ids.csv").exists()


def test_patientid_not_in_features():
    df = load_raw_excel()
    X, y, metadata = get_feature_matrix(df)
    assert "PatientID" not in X.columns, "PatientID must not be present in feature matrix X"
    assert "PatientID" in df.columns, "PatientID must still be present in the original dataframe"
    assert "patient_ids" in metadata and len(metadata["patient_ids"]) == len(df)


def test_default_saves_results_splits():
    # run with default out_dir (repo results/splits)
    df = load_raw_excel()
    out_dir = REPO_ROOT / "results" / "splits"
    # cleanup any existing files from previous runs
    if out_dir.exists():
        for f in out_dir.glob("*.csv"):
            try:
                f.unlink()
            except Exception:
                pass
    create_patient_splits(df=df, out_dir=out_dir, random_state=2)
    assert (out_dir / "train_patient_ids.csv").exists()
    assert (out_dir / "validation_patient_ids.csv").exists()
    assert (out_dir / "test_patient_ids.csv").exists()
