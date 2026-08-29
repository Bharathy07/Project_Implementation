# Final Experiment Report

## Environment
- Python: 3.14.7
- Execution is measured from the current run; no synthetic results were used.

## Dataset and splitting
- Train/validation/test: 378/81/82; PatientID is preserved as metadata and excluded from model features.

## Predictive results
```csv
Model,Accuracy,Precision,Recall,Sensitivity,Specificity,F1-score,ROC-AUC,PR-AUC,MCC,Cohen's Kappa,Balanced Accuracy,Log Loss,Brier Score
XGBoost,0.8780487804878049,0.8846153846153846,0.7666666666666667,0.7666666666666667,0.9423076923076923,0.8214285714285714,0.9679487179487178,0.9469906615325054,0.7338581813775421,0.7295514511873351,0.8544871794871796,0.25880989416760103,0.08482595851270013
LightGBM,0.8536585365853658,0.8214285714285714,0.7666666666666667,0.7666666666666667,0.9038461538461539,0.7931034482758621,0.9544871794871794,0.9308867797804579,0.6810727271689808,0.6801040312093629,0.8352564102564103,0.2761104043496429,0.08999453272932759
Hybrid Ensemble,0.8780487804878049,0.8846153846153846,0.7666666666666667,0.7666666666666667,0.9423076923076923,0.8214285714285714,0.9679487179487178,0.9469906615325054,0.7338581813775421,0.7295514511873351,0.8544871794871796,0.25880989416760103,0.08482595851270013
```

## Counterfactual results
- Selected PatientIDs: 248, 451, 27, 169, 324.
- DiCE counterfactual rows: 15; CC-MO-CF feasible Pareto rows: 0.
- Counterfactuals are model-based hypothetical scenarios, not clinical prescriptions.

## Limitations
- Clinical configuration entries marked REQUIRES_CLINICAL_VALIDATION remain so.
- Ablation was not automatically executed because a constraint-removal variant is not safe for finalized research claims.

## Reproducibility
- Command: E:\IV IEEE RESEARCH\project\Project_Implementation\.venv\Scripts\python.exe run_experiment.py
- Runtime seconds: 41.82
