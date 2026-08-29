"""PatientID-preserving data ingestion and patient-level splits.

Functions:
- load_raw_excel(path, sheet_name)
- get_feature_matrix(df, target_col)
- create_patient_splits(df, out_dir, random_state, train_frac, val_frac)

Saves split CSVs to results/splits by default.
"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "PCOS_data_without_infertility.xlsx"
SHEET_NAME = "Full_new"


def load_raw_excel(path: Path = DATA_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Load the raw Excel sheet, normalize column names, and preserve PatientID.

    - Renames "Patient File No." to "PatientID" if present.
    - Strips whitespace from column names.
    - Replaces "#NAME?" placeholders with NA.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    # normalize column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    # preserve patient identifier
    if "Patient File No." in df.columns:
        df = df.rename(columns={"Patient File No.": "PatientID"})
    if "PatientID" not in df.columns:
        raise KeyError("Patient identifier column not found in sheet")
    # replace common Excel formula errors with NA
    df = df.replace("#NAME?", pd.NA)
    return df


def get_feature_matrix(df: pd.DataFrame, target_col: str = "PCOS (Y/N)"):
    """Return X (features), y (target), and metadata containing patient IDs.

    Ensures PatientID is not part of X but is returned in metadata.
    """
    if "PatientID" not in df.columns:
        raise KeyError("PatientID column missing from dataframe")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    df2 = df.copy()
    patient_ids = df2["PatientID"].tolist()
    y = df2[target_col].copy()
    X = df2.drop(columns=["PatientID", target_col])
    metadata = {"patient_ids": patient_ids}
    return X, y, metadata


def create_patient_splits(
    df: pd.DataFrame = None,
    out_dir: Path = None,
    random_state: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """Create reproducible patient-level train/validation/test splits.

    - Ensures no patient overlap between splits.
    - Saves CSVs to out_dir (defaults to <repo>/results/splits).
    - Returns a dict with 'train','validation','test' lists and a mapping DataFrame.
    """
    if df is None:
        df = load_raw_excel()
    if out_dir is None:
        out_dir = REPO_ROOT / "results" / "splits"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = list(pd.Series(df["PatientID"].unique()))
    rng = np.random.RandomState(random_state)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val

    train = patient_ids[:n_train]
    val = patient_ids[n_train : n_train + n_val]
    test = patient_ids[n_train + n_val :]

    # save split files
    pd.DataFrame({"PatientID": train}).to_csv(out_dir / "train_patient_ids.csv", index=False)
    pd.DataFrame({"PatientID": val}).to_csv(out_dir / "validation_patient_ids.csv", index=False)
    pd.DataFrame({"PatientID": test}).to_csv(out_dir / "test_patient_ids.csv", index=False)

    mapping = pd.DataFrame(
        {"PatientID": train + val + test, "split": ["train"] * len(train) + ["validation"] * len(val) + ["test"] * len(test)}
    )
    mapping.to_csv(out_dir / "patientid_split_mapping.csv", index=False)

    return {"train": train, "validation": val, "test": test, "mapping": mapping}
