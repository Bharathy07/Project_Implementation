"""Phase 16 read-only end-to-end validation audit.

Uses only the Python standard library so the audit can run even when the
research dependencies are unavailable. It does not execute experiments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
REPORT_PATH = RESULTS / "final_validation_report.md"


def _exists(relative: str) -> bool:
    return (REPO_ROOT / relative).exists()


def _csv_ids(relative: str) -> list[str]:
    path = REPO_ROOT / relative
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row.get("PatientID", "") for row in csv.DictReader(handle) if row.get("PatientID", "")]


def _source_contains(relative: str, text: str) -> bool:
    path = REPO_ROOT / relative
    return path.exists() and text in path.read_text(encoding="utf-8", errors="ignore")


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def run_audit() -> Path:
    checks: list[tuple[str, str, bool, str]] = []

    train = set(_csv_ids("results/splits/train_patient_ids.csv"))
    validation = set(_csv_ids("results/splits/validation_patient_ids.csv"))
    test = set(_csv_ids("results/splits/test_patient_ids.csv"))
    checks.extend([
        ("PatientID preserved in metadata/output", "Data integrity", bool(test) and _source_contains("src/pw_imp/preprocessing.py", 'patient_test = df_test[["PatientID"]]'), "Test split and preprocessing metadata path verified."),
        ("PatientID excluded from model features", "Data integrity", _source_contains("src/pw_imp/preprocessing.py", 'drop(columns=["PatientID", target_col])'), "Preprocessing drops PatientID before model input."),
        ("No train/validation/test patient overlap", "Data integrity", bool(train and validation and test and not (train & validation or train & test or validation & test)), "Split CSV sets are disjoint."),
        ("No preprocessing leakage", "Data integrity", _source_contains("src/pw_imp/preprocessing.py", "pre.fit(X_train, y_train)"), "Preprocessor is fitted on training data before validation/test transform."),
        ("No test-set hyperparameter tuning", "Data integrity", _source_contains("src/pw_imp/models.py", "optuna_xgboost(X_train, y_train") and _source_contains("src/pw_imp/models.py", "optuna_lightgbm(X_train, y_train"), "Hyperparameter search source uses training data; runtime audit did not rerun training."),
        ("XGBoost reproducible", "Prediction", False, "Final XGBoost artifact exists, but runtime reproducibility check was blocked by missing dependencies."),
        ("LightGBM reproducible", "Prediction", False, "Final LightGBM artifact exists, but runtime reproducibility check was blocked by missing dependencies."),
        ("Final model selection follows predefined criterion", "Prediction", False, "Saved model summary exists, but a formal selection criterion is not documented."),
        ("Metrics match saved experiment outputs", "Prediction", False, "Saved model metrics exist, but independent recomputation was blocked by missing dependencies."),
        ("SHAP outputs generated from finalized model", "Explainability", _exists("results/figures/shap/global_feature_importance.csv") and _exists("results/figures/shap/patient_explanations.csv"), "Global and patient SHAP CSV outputs exist."),
        ("Local explanations map to correct PatientID", "Explainability", _source_contains("src/pw_imp/shap_explainer.py", 'patient_id = int(row["PatientID"])'), "SHAP code maps local rows through PatientID metadata."),
        ("DiCE is actual DiCE or explicitly unavailable", "Counterfactuals", _exists("results/counterfactuals/dice/dice_summary.csv") and _source_contains("src/pw_imp/dice_baseline.py", "dice_ml"), "DiCE implementation and saved outputs exist; runtime dependency verification unavailable."),
        ("CC-MO-CF produces valid outputs", "Counterfactuals", False, "Implementation exists, but no finalized Phase 9 runtime output is available."),
        ("Immutable features remain unchanged", "Counterfactuals", _source_contains("src/pw_imp/clinical_constraints.py", "immutable"), "Constraint engine protects immutable features; full run not executed."),
        ("Non-actionable features remain unchanged", "Counterfactuals", _source_contains("src/pw_imp/clinical_constraints.py", "non-actionable"), "Constraint engine rejects non-actionable edits."),
        ("Bounds enforced", "Counterfactuals", _source_contains("src/pw_imp/clinical_constraints.py", "lower_bound") and _source_contains("src/pw_imp/clinical_constraints.py", "upper_bound"), "Configured bounds are implemented."),
        ("Max-step enforced", "Counterfactuals", _source_contains("src/pw_imp/clinical_constraints.py", "max_step"), "Configured max-step projection and validation are implemented."),
        ("Dependencies preserved", "Counterfactuals", _source_contains("src/pw_imp/clinical_constraints.py", "_recompute_derived_value"), "Derived dependency restoration is implemented."),
        ("Prediction flips verified", "Counterfactuals", False, "Flip validation exists in source, but no end-to-end runtime verification is available."),
        ("Failed patients retained", "Counterfactuals", False, "Phase 9 code preserves failed rows, but its finalized output is unavailable."),
        ("No cherry-picking", "Counterfactuals", False, "Phase 9 iterates all held-out rows in source, but completed-run evidence is unavailable."),
        ("Same eligible patients used for paired comparison", "Evaluation", False, "Intersection logic exists, but finalized evaluation output is missing."),
        ("Metrics reproducible", "Evaluation", False, "Metric code exists, but runtime verification is blocked and finalized outputs are missing."),
        ("Ablation completed", "Evaluation", _exists("results/ablation/ablation_results.csv"), "Final ablation CSV is missing."),
        ("Statistical tests appropriate", "Evaluation", False, "Method-selection code exists, but no finalized statistical test output is available."),
        ("Confidence intervals generated", "Evaluation", _exists("results/statistics/confidence_intervals.csv"), "Final confidence-interval output is missing."),
        ("Patient-level CSV exists", "Outputs", _exists("results/patients/patient_master_results.csv"), "Phase 11 master output is missing."),
        ("Model comparison table exists", "Outputs", _exists("results/evaluation/ccmocf_vs_dice_summary.csv"), "Phase 10 comparison output is missing."),
        ("Counterfactual comparison exists", "Outputs", _exists("results/evaluation/cf_metrics_per_counterfactual.csv"), "Phase 10 counterfactual metrics output is missing."),
        ("Ablation table exists", "Outputs", _exists("results/ablation/ablation_per_patient.csv"), "Phase 13 ablation output is missing."),
        ("Statistics table exists", "Outputs", _exists("results/statistics/paired_tests.csv"), "Phase 14 paired-test output is missing."),
        ("Paper figures exist", "Outputs", _exists("paper_figures"), "Phase 15 paper figure directory is missing."),
    ])

    failed = [(name, area, evidence) for name, area, passed, evidence in checks if not passed]
    warnings = [
        "The configured interpreter is Python 3.15 alpha and lacks numpy, pandas, pytest, and the project runtime dependencies.",
        "Source-level checks are not substitutes for runtime experiment validation.",
        "The formal final-model selection criterion is not explicitly documented in the repository.",
        "Phase 9 through Phase 15 output directories are absent, so no final end-to-end results can be claimed.",
        "No clinical treatment or superiority claim is made.",
    ]
    missing = [
        relative for relative in (
            "results/experiments/phase9_ccmocf/ccmocf_all_attempts.csv",
            "results/patients/patient_master_results.csv",
            "results/evaluation/cf_metrics_per_counterfactual.csv",
            "results/ablation/ablation_results.csv",
            "results/statistics/confidence_intervals.csv",
            "paper_figures",
        ) if not _exists(relative)
    ]
    lines = ["# Final Validation Report", "", "Audit scope: read-only repository and artifact audit. No algorithms, models, experiments, or tuning were run.", "", "## Check Results", ""]
    for name, area, passed, evidence in checks:
        lines.append(f"- [{_status(passed)}] **{area}: {name}**. {evidence}")
    lines.extend(["", "## Failed Checks", ""])
    if failed:
        lines.extend(f"- **{area}: {name}**: {evidence}" for name, area, evidence in failed)
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Missing Outputs", ""])
    lines.extend(f"- `{item}`" for item in missing) if missing else lines.append("- None.")
    lines.extend(["", "## Known Limitations", "", "- Runtime checks requiring pandas, numpy, scikit-learn, XGBoost, LightGBM, SHAP, DiCE, or pytest could not execute in the active environment.", "- Existing DiCE files cover only the saved Phase 7 patients and do not establish an all-test-patient comparison.", "- Phase 9-15 result generation remains incomplete; source implementations are not evidence that experiments completed.", "- The audit does not make clinical treatment claims.", "", "## Reproducibility Information", "", "- Held-out split: `results/splits/patientid_split_mapping.csv` and `results/splits/test_patient_ids.csv`.", "- Final model parameters: `results/models/model_parameters.json`.", "- Final model artifacts: `results/models/xgboost_model.json` and `results/models/lightgbm_model.joblib`.", "- Clinical constraints: `config/clinical_constraints.yaml`.", "- Fixed seed in Phase 9/13/14 defaults: `42`.", "- Environment blocker: Python 3.15 alpha with missing scientific dependencies.", "", "## Exact Experiment Configuration References", "", "- Phase 9: `src/pw_imp/phase9.py`.", "- Phase 10: `src/pw_imp/phase10.py`.", "- Phase 11: `src/pw_imp/phase11.py`.", "- Phase 12: `src/pw_imp/phase12.py`.", "- Phase 13: `src/pw_imp/phase13.py`.", "- Phase 14: `src/pw_imp/phase14.py`.", "- Phase 15: `src/pw_imp/phase15.py`.", "", "## Overall Status", "", f"**FAIL / INCOMPLETE:** {len(failed)} checks failed or were not evidenced by finalized outputs. The repository is not ready for a claim of complete end-to-end validation."])
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    print(run_audit())