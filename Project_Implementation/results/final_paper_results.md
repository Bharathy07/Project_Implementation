# Final Paper Results

All numeric values below are loaded from existing finalized output files. No experiments were retrained for this export.

## Dataset Results

```csv
Statistic,Value
Records,541
Features excluding target/PatientID,43
Missing values,1904
Duplicate records,0
Positive class,177
Negative class,364
```

## Classification Performance

```csv
Model,Accuracy,Precision,Recall,Sensitivity,Specificity,F1-score,ROC-AUC,PR-AUC,MCC,Cohen's Kappa,Balanced Accuracy,Log Loss,Brier Score
XGBoost,0.8780487804878049,0.8846153846153846,0.7666666666666667,0.7666666666666667,0.9423076923076924,0.8214285714285714,0.9679487179487178,0.9469906615325054,0.7338581813775421,0.7295514511873351,0.8544871794871796,0.258809894167601,0.0848259585127001
LightGBM,0.8536585365853658,0.8214285714285714,0.7666666666666667,0.7666666666666667,0.903846153846154,0.7931034482758621,0.9544871794871794,0.930886779780458,0.6810727271689808,0.6801040312093629,0.8352564102564103,0.2761104043496429,0.0899945327293275
Hybrid Ensemble,0.8780487804878049,0.8846153846153846,0.7666666666666667,0.7666666666666667,0.9423076923076924,0.8214285714285714,0.9679487179487178,0.9469906615325054,0.7338581813775421,0.7295514511873351,0.8544871794871796,0.258809894167601,0.0848259585127001
```

## DiCE Results

```csv
PatientID,OriginalPrediction,OriginalProbability,CounterfactualPrediction,CounterfactualProbability,ChangedFeatures,Sparsity,Proximity,Diversity,Validity,Runtime
248,1,0.9951229691505432,0,0.0,BMI | Cycle length(days) | Follicle No. (L) | Follicle No. (R) | Cycle(R/I)_4 | Pimples(Y/N)_0,6,,,1.0,
248,1,0.9951229691505432,0,0.0,Weight (Kg) | BMI | Cycle length(days) | Follicle No. (L) | Follicle No. (R) | hair growth(Y/N)_0 | Pimples(Y/N)_0,7,,,1.0,
248,1,0.9951229691505432,0,0.0,BMI | Cycle length(days) | Follicle No. (L) | Follicle No. (R) | Cycle(R/I)_4 | Skin darkening (Y/N)_0 | Pimples(Y/N)_0,7,,,1.0,
451,1,0.992528200149536,0,0.0,Age (yrs) | BMI | Cycle length(days) | Follicle No. (R) | Pimples(Y/N)_0,5,,,1.0,
451,1,0.992528200149536,0,0.0,Age (yrs) | BMI | Cycle length(days) | Follicle No. (R) | Weight gain(Y/N)_1,5,,,1.0,
451,1,0.992528200149536,0,0.0,Age (yrs) | BMI | Cycle length(days) | Follicle No. (L) | Follicle No. (R),5,,,1.0,
27,1,0.9917660355567932,0,0.0,Age (yrs) | BMI | Follicle No. (L) | Follicle No. (R) | Pimples(Y/N)_0,5,,,1.0,
27,1,0.9917660355567932,0,0.0,Age (yrs) | BMI | Follicle No. (R) | Avg. F size (L) (mm) | hair growth(Y/N)_0,5,,,1.0,
27,1,0.9917660355567932,0,0.0,Age (yrs) | BMI | Follicle No. (L) | Follicle No. (R),4,,,1.0,
169,1,0.991647481918335,0,0.0,Age (yrs) | FSH(mIU/mL) | Follicle No. (R) | Fast food (Y/N)_0.0,4,,,1.0,
169,1,0.991647481918335,0,0.0,Age (yrs) | Follicle No. (L) | Follicle No. (R) | Cycle(R/I)_2,4,,,1.0,
169,1,0.991647481918335,0,0.0,Age (yrs) | FSH(mIU/mL) | Follicle No. (R),3,,,1.0,
324,1,0.9904378056526184,0,0.0,BMI | FSH(mIU/mL) | Follicle No. (R) | Avg. F size (L) (mm),4,,,1.0,
324,1,0.9904378056526184,0,0.0,BMI | FSH(mIU/mL) | Follicle No. (R),3,,,1.0,
324,1,0.9904378056526184,0,0.0,BMI | FSH(mIU/mL) | Follicle No. (L) | Follicle No. (R),4,,,1.0,
```

## CC-MO-CF and Five-Profile Results

```csv
PatientID,OriginalPrediction,OriginalProbability,CounterfactualFound,NumberOfCounterfactuals,BestCounterfactualProbability,ChangedFeatures,Sparsity,Proximity,Plausibility,Diversity,ConstraintSatisfaction,DependencyConsistency,Runtime,FailureReason
248,1,0.9223580956459044,False,0,,,,,,,NO_VALID_CANDIDATE,NOT_APPLICABLE,0.1749917999841272,NO_PREDICTION_FLIP
451,1,0.8918277025222778,False,0,,,,,,,NO_VALID_CANDIDATE,NOT_APPLICABLE,0.1712902999715879,NO_PREDICTION_FLIP
27,1,0.895735502243042,False,0,,,,,,,NO_VALID_CANDIDATE,NOT_APPLICABLE,0.2331237000180408,NO_PREDICTION_FLIP
169,1,0.9075535535812378,False,0,,,,,,,NO_VALID_CANDIDATE,NOT_APPLICABLE,0.2719557000091299,NO_PREDICTION_FLIP
324,1,0.9054586291313172,False,0,,,,,,,NO_VALID_CANDIDATE,NOT_APPLICABLE,0.2340258000185713,NO_PREDICTION_FLIP
```

## DiCE vs CC-MO-CF

```csv
Method,Validity,Proximity,Sparsity,Plausibility,Diversity,Constraint Satisfaction,Dependency Consistency,Runtime
DiCE,1.0,,4.733333333333333,,,,,
CC-MO-CF,0.0,,,,,0.0,,
```

## Ablation Results

```csv
Variant,Status,Reason
NOT IMPLEMENTED,NOT IMPLEMENTED,Ablation is not run automatically because removing clinical constraints cannot safely produce research results.
```

## Important Observations

- DiCE rows: 15.
- CC-MO-CF feasible rows: 0.
- Counterfactuals are model-based hypothetical scenarios, not clinical recommendations.

## Limitations

- Clinical constraints marked `REQUIRES_CLINICAL_VALIDATION` remain unvalidated by this software dashboard.
- Ablation entries are displayed only when actually executed.
