# Final Validation Report

Audit scope: read-only repository and artifact audit. No algorithms, models, experiments, or tuning were run.

## Check Results

- [PASS] **Data integrity: PatientID preserved in metadata/output**. Test split and preprocessing metadata path verified.
- [PASS] **Data integrity: PatientID excluded from model features**. Preprocessing drops PatientID before model input.
- [PASS] **Data integrity: No train/validation/test patient overlap**. Split CSV sets are disjoint.
- [PASS] **Data integrity: No preprocessing leakage**. Preprocessor is fitted on training data before validation/test transform.
- [PASS] **Data integrity: No test-set hyperparameter tuning**. Hyperparameter search source uses training data; runtime audit did not rerun training.
- [FAIL] **Prediction: XGBoost reproducible**. Final XGBoost artifact exists, but runtime reproducibility check was blocked by missing dependencies.
- [FAIL] **Prediction: LightGBM reproducible**. Final LightGBM artifact exists, but runtime reproducibility check was blocked by missing dependencies.
- [FAIL] **Prediction: Final model selection follows predefined criterion**. Saved model summary exists, but a formal selection criterion is not documented.
- [FAIL] **Prediction: Metrics match saved experiment outputs**. Saved model metrics exist, but independent recomputation was blocked by missing dependencies.
- [PASS] **Explainability: SHAP outputs generated from finalized model**. Global and patient SHAP CSV outputs exist.
- [PASS] **Explainability: Local explanations map to correct PatientID**. SHAP code maps local rows through PatientID metadata.
- [PASS] **Counterfactuals: DiCE is actual DiCE or explicitly unavailable**. DiCE implementation and saved outputs exist; runtime dependency verification unavailable.
- [FAIL] **Counterfactuals: CC-MO-CF produces valid outputs**. Implementation exists, but no finalized Phase 9 runtime output is available.
- [PASS] **Counterfactuals: Immutable features remain unchanged**. Constraint engine protects immutable features; full run not executed.
- [PASS] **Counterfactuals: Non-actionable features remain unchanged**. Constraint engine rejects non-actionable edits.
- [PASS] **Counterfactuals: Bounds enforced**. Configured bounds are implemented.
- [PASS] **Counterfactuals: Max-step enforced**. Configured max-step projection and validation are implemented.
- [PASS] **Counterfactuals: Dependencies preserved**. Derived dependency restoration is implemented.
- [FAIL] **Counterfactuals: Prediction flips verified**. Flip validation exists in source, but no end-to-end runtime verification is available.
- [FAIL] **Counterfactuals: Failed patients retained**. Phase 9 code preserves failed rows, but its finalized output is unavailable.
- [FAIL] **Counterfactuals: No cherry-picking**. Phase 9 iterates all held-out rows in source, but completed-run evidence is unavailable.
- [FAIL] **Evaluation: Same eligible patients used for paired comparison**. Intersection logic exists, but finalized evaluation output is missing.
- [FAIL] **Evaluation: Metrics reproducible**. Metric code exists, but runtime verification is blocked and finalized outputs are missing.
- [FAIL] **Evaluation: Ablation completed**. Final ablation CSV is missing.
- [FAIL] **Evaluation: Statistical tests appropriate**. Method-selection code exists, but no finalized statistical test output is available.
- [FAIL] **Evaluation: Confidence intervals generated**. Final confidence-interval output is missing.
- [FAIL] **Outputs: Patient-level CSV exists**. Phase 11 master output is missing.
- [FAIL] **Outputs: Model comparison table exists**. Phase 10 comparison output is missing.
- [FAIL] **Outputs: Counterfactual comparison exists**. Phase 10 counterfactual metrics output is missing.
- [FAIL] **Outputs: Ablation table exists**. Phase 13 ablation output is missing.
- [FAIL] **Outputs: Statistics table exists**. Phase 14 paired-test output is missing.
- [FAIL] **Outputs: Paper figures exist**. Phase 15 paper figure directory is missing.

## Failed Checks

- **Prediction: XGBoost reproducible**: Final XGBoost artifact exists, but runtime reproducibility check was blocked by missing dependencies.
- **Prediction: LightGBM reproducible**: Final LightGBM artifact exists, but runtime reproducibility check was blocked by missing dependencies.
- **Prediction: Final model selection follows predefined criterion**: Saved model summary exists, but a formal selection criterion is not documented.
- **Prediction: Metrics match saved experiment outputs**: Saved model metrics exist, but independent recomputation was blocked by missing dependencies.
- **Counterfactuals: CC-MO-CF produces valid outputs**: Implementation exists, but no finalized Phase 9 runtime output is available.
- **Counterfactuals: Prediction flips verified**: Flip validation exists in source, but no end-to-end runtime verification is available.
- **Counterfactuals: Failed patients retained**: Phase 9 code preserves failed rows, but its finalized output is unavailable.
- **Counterfactuals: No cherry-picking**: Phase 9 iterates all held-out rows in source, but completed-run evidence is unavailable.
- **Evaluation: Same eligible patients used for paired comparison**: Intersection logic exists, but finalized evaluation output is missing.
- **Evaluation: Metrics reproducible**: Metric code exists, but runtime verification is blocked and finalized outputs are missing.
- **Evaluation: Ablation completed**: Final ablation CSV is missing.
- **Evaluation: Statistical tests appropriate**: Method-selection code exists, but no finalized statistical test output is available.
- **Evaluation: Confidence intervals generated**: Final confidence-interval output is missing.
- **Outputs: Patient-level CSV exists**: Phase 11 master output is missing.
- **Outputs: Model comparison table exists**: Phase 10 comparison output is missing.
- **Outputs: Counterfactual comparison exists**: Phase 10 counterfactual metrics output is missing.
- **Outputs: Ablation table exists**: Phase 13 ablation output is missing.
- **Outputs: Statistics table exists**: Phase 14 paired-test output is missing.
- **Outputs: Paper figures exist**: Phase 15 paper figure directory is missing.

## Warnings

- The configured interpreter is Python 3.15 alpha and lacks numpy, pandas, pytest, and the project runtime dependencies.
- Source-level checks are not substitutes for runtime experiment validation.
- The formal final-model selection criterion is not explicitly documented in the repository.
- Phase 9 through Phase 15 output directories are absent, so no final end-to-end results can be claimed.
- No clinical treatment or superiority claim is made.

## Missing Outputs

- `results/experiments/phase9_ccmocf/ccmocf_all_attempts.csv`
- `results/patients/patient_master_results.csv`
- `results/evaluation/cf_metrics_per_counterfactual.csv`
- `results/ablation/ablation_results.csv`
- `results/statistics/confidence_intervals.csv`
- `paper_figures`

## Known Limitations

- Runtime checks requiring pandas, numpy, scikit-learn, XGBoost, LightGBM, SHAP, DiCE, or pytest could not execute in the active environment.
- Existing DiCE files cover only the saved Phase 7 patients and do not establish an all-test-patient comparison.
- Phase 9-15 result generation remains incomplete; source implementations are not evidence that experiments completed.
- The audit does not make clinical treatment claims.

## Reproducibility Information

- Held-out split: `results/splits/patientid_split_mapping.csv` and `results/splits/test_patient_ids.csv`.
- Final model parameters: `results/models/model_parameters.json`.
- Final model artifacts: `results/models/xgboost_model.json` and `results/models/lightgbm_model.joblib`.
- Clinical constraints: `config/clinical_constraints.yaml`.
- Fixed seed in Phase 9/13/14 defaults: `42`.
- Environment blocker: Python 3.15 alpha with missing scientific dependencies.

## Exact Experiment Configuration References

- Phase 9: `src/pw_imp/phase9.py`.
- Phase 10: `src/pw_imp/phase10.py`.
- Phase 11: `src/pw_imp/phase11.py`.
- Phase 12: `src/pw_imp/phase12.py`.
- Phase 13: `src/pw_imp/phase13.py`.
- Phase 14: `src/pw_imp/phase14.py`.
- Phase 15: `src/pw_imp/phase15.py`.

## Overall Status

**FAIL / INCOMPLETE:** 19 checks failed or were not evidenced by finalized outputs. The repository is not ready for a claim of complete end-to-end validation.
