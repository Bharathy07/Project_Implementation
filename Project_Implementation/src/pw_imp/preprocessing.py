"""Leakage-free preprocessing pipeline for tabular PCOS dataset.

Implements:
- Patient-aware train/val/test splitting loader
- Preprocessor class which fits on training data only and transforms validation/test
- Missing-value handling (median for numeric, most frequent for categorical)
- Recompute BMI, FSH/LH, Waist:Hip Ratio when missing from component columns
- Outlier clipping using train percentiles (1st/99th)
- One-hot encoding for categorical features
- Scaling (StandardScaler) for numeric features
- Feature selection using mutual information (SelectKBest) fitted on train labels only

Saves selected features to results/features/selected_features.csv
"""
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

# sklearn imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_MAPPING = REPO_ROOT / "results" / "splits" / "patientid_split_mapping.csv"
DATA_PATH = REPO_ROOT / "PCOS_data_without_infertility.xlsx"
SHEET_NAME = "Full_new"
FEATURES_OUT = REPO_ROOT / "results" / "features"
FEATURES_OUT.mkdir(parents=True, exist_ok=True)


def load_dataframe():
    # reuse splitter loader logic lightly here to ensure PatientID rename
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if "Patient File No." in df.columns:
        df = df.rename(columns={"Patient File No.": "PatientID"})
    df = df.replace("#NAME?", pd.NA)
    return df


def load_splits():
    if not SPLIT_MAPPING.exists():
        raise FileNotFoundError("Split mapping not found. Run splitter.create_patient_splits first.")
    mapping = pd.read_csv(SPLIT_MAPPING)
    return mapping


@dataclass
class Preprocessor:
    numeric_cols: list = None
    categorical_cols: list = None
    num_imputer: SimpleImputer = None
    cat_imputer: SimpleImputer = None
    encoder: OneHotEncoder = None
    scaler: StandardScaler = None
    clip_min: dict = None
    clip_max: dict = None
    selected_features: list = None
    selector: SelectKBest = None

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Recompute BMI if missing: Weight (Kg) and Height(Cm)
        if "BMI" in df.columns and "Weight (Kg)" in df.columns and "Height(Cm)" in df.columns:
            # height in cm to meters
            h_m = df["Height(Cm)"].astype(float) / 100.0
            bmi_calc = df["Weight (Kg)"].astype(float) / (h_m ** 2)
            # if BMI missing, fill with calc
            df["BMI"] = df["BMI"].astype(float)
            missing_bmi = df["BMI"].isna()
            df.loc[missing_bmi, "BMI"] = bmi_calc[missing_bmi]
        # Recompute FSH/LH if missing
        if "FSH/LH" in df.columns and "FSH(mIU/mL)" in df.columns and "LH(mIU/mL)" in df.columns:
            df["FSH(mIU/mL)"] = pd.to_numeric(df["FSH(mIU/mL)"], errors="coerce")
            df["LH(mIU/mL)"] = pd.to_numeric(df["LH(mIU/mL)"], errors="coerce")
            ratio = df["FSH(mIU/mL)"] / df["LH(mIU/mL)"]
            df["FSH/LH"] = df["FSH/LH"].astype(float)
            missing_ratio = df["FSH/LH"].isna()
            df.loc[missing_ratio, "FSH/LH"] = ratio[missing_ratio]
        # Recompute Waist:Hip Ratio if missing
        if "Waist:Hip Ratio" in df.columns and "Waist(inch)" in df.columns and "Hip(inch)" in df.columns:
            df["Waist(inch)"] = pd.to_numeric(df["Waist(inch)"], errors="coerce")
            df["Hip(inch)"] = pd.to_numeric(df["Hip(inch)"], errors="coerce")
            wh = df["Waist(inch)"] / df["Hip(inch)"]
            df["Waist:Hip Ratio"] = df["Waist:Hip Ratio"].astype(float)
            missing_wh = df["Waist:Hip Ratio"].isna()
            df.loc[missing_wh, "Waist:Hip Ratio"] = wh[missing_wh]
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # X is a dataframe that excludes PatientID and target
        df = X.copy()
        df = self._engineer(df)

        # determine categorical vs numeric
        # treat low-cardinality numeric as categorical
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        nonnumeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        # low cardinality numerics to categorical
        for col in numeric[:]:
            if df[col].nunique(dropna=True) <= 10:
                nonnumeric.append(col)
                numeric.remove(col)
        self.numeric_cols = numeric
        self.categorical_cols = nonnumeric

        # imputers
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        if len(self.numeric_cols) > 0:
            self.num_imputer.fit(df[self.numeric_cols])
        if len(self.categorical_cols) > 0:
            self.cat_imputer.fit(df[self.categorical_cols])

        # clip percentiles
        self.clip_min = {}
        self.clip_max = {}
        for col in self.numeric_cols:
            col_ser = pd.to_numeric(df[col], errors="coerce")
            lo = np.nanpercentile(col_ser, 1)
            hi = np.nanpercentile(col_ser, 99)
            self.clip_min[col] = lo
            self.clip_max[col] = hi

        # encode
        # handle sklearn API differences for sparse vs sparse_output
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if len(self.categorical_cols) > 0:
            cat_filled = pd.DataFrame(self.cat_imputer.transform(df[self.categorical_cols]), columns=self.categorical_cols)
            # ensure uniform types for encoder
            cat_filled = cat_filled.astype(str).fillna("NA")
            self.encoder.fit(cat_filled)

        # scale
        self.scaler = StandardScaler()
        if len(self.numeric_cols) > 0:
            num_filled = pd.DataFrame(self.num_imputer.transform(df[self.numeric_cols]), columns=self.numeric_cols)
            # clip
            for col in self.numeric_cols:
                num_filled[col] = num_filled[col].clip(self.clip_min[col], self.clip_max[col])
            self.scaler.fit(num_filled)

        # feature selection using mutual information if y provided
        if y is not None:
            # transform numeric and categorical to a full dataframe
            X_trans = self._transform_df(df)
            # use mutual_info_classif
            # Drop NaNs in X_trans and y alignment
            Xn = X_trans.fillna(0)
            mi = mutual_info_classif(Xn, y, discrete_features=False, random_state=0)
            # select top k features
            k = min(20, Xn.shape[1])
            self.selector = SelectKBest(k=k)
            self.selector.fit(Xn, y)
            mask = self.selector.get_support()
            self.selected_features = list(Xn.columns[mask])
        else:
            # default to all features if no y
            X_trans = self._transform_df(df)
            self.selected_features = list(X_trans.columns)
        # save selected features
        pd.DataFrame({"feature": self.selected_features}).to_csv(FEATURES_OUT / "selected_features.csv", index=False)

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2 = self._engineer(df2)
        # numeric
        num = pd.DataFrame()
        if len(self.numeric_cols) > 0:
            num_arr = self.num_imputer.transform(df2[self.numeric_cols])
            num = pd.DataFrame(num_arr, columns=self.numeric_cols, index=df2.index)
            for col in self.numeric_cols:
                num[col] = num[col].clip(self.clip_min[col], self.clip_max[col])
        # categorical
        cat = pd.DataFrame()
        if len(self.categorical_cols) > 0:
            cat_filled = pd.DataFrame(self.cat_imputer.transform(df2[self.categorical_cols]), columns=self.categorical_cols, index=df2.index)
            # ensure same dtype treatment
            cat_filled = cat_filled.astype(str).fillna("NA")
            cat_arr = self.encoder.transform(cat_filled)
            cat_cols = list(self.encoder.get_feature_names_out(self.categorical_cols))
            cat = pd.DataFrame(cat_arr, columns=cat_cols, index=df2.index)
        # combine
        combined = pd.concat([num, cat], axis=1)
        # scale numeric columns only
        if len(self.numeric_cols) > 0:
            combined[self.numeric_cols] = self.scaler.transform(combined[self.numeric_cols])
        return combined

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform_df(X)


# High-level orchestration
def build_preprocessing_pipeline(random_state: int = 42):
    # load mapping and raw data
    mapping = load_splits()
    df = load_dataframe()
    # rename in df if needed
    if "PatientID" not in df.columns and "Patient File No." in df.columns:
        df = df.rename(columns={"Patient File No.": "PatientID"})
    # target
    target_col = "PCOS (Y/N)"
    # create patient groups
    train_ids = mapping[mapping["split"] == "train"]["PatientID"].tolist()
    val_ids = mapping[mapping["split"] == "validation"]["PatientID"].tolist()
    test_ids = mapping[mapping["split"] == "test"]["PatientID"].tolist()

    df_train = df[df["PatientID"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["PatientID"].isin(val_ids)].reset_index(drop=True)
    df_test = df[df["PatientID"].isin(test_ids)].reset_index(drop=True)

    # preserve patient metadata
    patient_train = df_train[["PatientID"]].copy()
    patient_val = df_val[["PatientID"]].copy()
    patient_test = df_test[["PatientID"]].copy()

    # build X and y (drop PatientID and target from X)
    X_train = df_train.drop(columns=["PatientID", target_col])
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=["PatientID", target_col])
    y_val = df_val[target_col]
    X_test = df_test.drop(columns=["PatientID", target_col])
    y_test = df_test[target_col]

    # fit preprocessor on train only
    pre = Preprocessor()
    pre.fit(X_train, y_train)

    # transform datasets
    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # select features
    if pre.selected_features is not None:
        X_train_sel = X_train_t[pre.selected_features]
        X_val_sel = X_val_t.reindex(columns=pre.selected_features).fillna(0)
        X_test_sel = X_test_t.reindex(columns=pre.selected_features).fillna(0)
    else:
        X_train_sel = X_train_t
        X_val_sel = X_val_t
        X_test_sel = X_test_t

    # save selected features already handled in pre.fit

    return {
        "preprocessor": pre,
        "X_train": X_train_sel,
        "y_train": y_train.reset_index(drop=True),
        "X_val": X_val_sel,
        "y_val": y_val.reset_index(drop=True),
        "X_test": X_test_sel,
        "y_test": y_test.reset_index(drop=True),
        "patient_train": patient_train.reset_index(drop=True),
        "patient_val": patient_val.reset_index(drop=True),
        "patient_test": patient_test.reset_index(drop=True),
        "preprocessor": pre,
    }
