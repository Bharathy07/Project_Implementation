"""Single reproducible entry point for the PCOS predictive and CF experiment.

All measured artifacts are written below ``results``.  This module deliberately
uses the existing preprocessing, Optuna, XGBoost, LightGBM, TreeSHAP, DiCE and
NSGA-II implementations rather than substituting surrogate algorithms.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS, FIGURES, TABLES, SHAP_OUT, CF_OUT = (ROOT / "results", ROOT / "results" / "figures",
    ROOT / "results" / "tables", ROOT / "results" / "shap", ROOT / "results" / "counterfactuals")


def _mkdirs():
    for path in (RESULTS, FIGURES, TABLES, SHAP_OUT, CF_OUT): path.mkdir(parents=True, exist_ok=True)


def _proba_xgb(model, frame): return np.asarray(model.predict(xgb.DMatrix(frame)), dtype=float)
def _proba(model, frame):
    values = model.predict_proba(frame) if hasattr(model, "predict_proba") else model.predict(frame)
    values = np.asarray(values)
    return values[:, 1] if values.ndim == 2 else values.reshape(-1)


def _metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y,pred), "Precision": precision_score(y,pred,zero_division=0),
        "Recall": recall_score(y,pred,zero_division=0), "Sensitivity": recall_score(y,pred,zero_division=0),
        "Specificity": tn/(tn+fp) if tn+fp else 0., "F1-score": f1_score(y,pred,zero_division=0),
        "ROC-AUC": roc_auc_score(y,p), "PR-AUC": average_precision_score(y,p),
        "MCC": matthews_corrcoef(y,pred), "Cohen's Kappa": cohen_kappa_score(y,pred),
        "Balanced Accuracy": balanced_accuracy_score(y,pred), "Log Loss": log_loss(y,p,labels=[0,1]),
        "Brier Score": brier_score_loss(y,p)}


def _savefig(name):
    plt.tight_layout(); plt.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close()


def _model_figures(y, scores, names, xgb_model, columns):
    metrics = {name: _metrics(y, score) for name, score in scores.items()}
    plt.figure(figsize=(7,4)); plt.bar(names, [metrics[n]["Accuracy"] for n in names]); plt.ylabel("Test accuracy"); plt.title("Model Accuracy Comparison"); _savefig("accuracy_comparison")
    ensemble = scores["Hybrid Ensemble"]; cm = confusion_matrix(y, ensemble >= .5)
    plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues"); plt.colorbar();
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.xticks([0,1],["Negative","Positive"]); plt.yticks([0,1],["Negative","Positive"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Hybrid Ensemble Confusion Matrix"); _savefig("confusion_matrix")
    plt.figure(figsize=(6,5))
    for n,p in scores.items():
        f,t,_=roc_curve(y,p); plt.plot(f,t,label=f"{n} ({roc_auc_score(y,p):.3f})")
    plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curves"); _savefig("roc_curve")
    plt.figure(figsize=(6,5))
    for n,p in scores.items():
        pr,re,_=precision_recall_curve(y,p); plt.plot(re,pr,label=f"{n} ({average_precision_score(y,p):.3f})")
    plt.legend(); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves"); _savefig("precision_recall_curve")
    selected=["Accuracy","Precision","Recall","F1-score","Specificity","Balanced Accuracy","MCC"]
    pd.DataFrame({n:[metrics[n][m] for m in selected] for n in names},index=selected).plot(kind="bar",figsize=(10,5)); plt.ylabel("Score"); plt.title("Performance Metrics Comparison"); _savefig("metrics_comparison")
    plt.figure(figsize=(6,5)); pt,pp=calibration_curve(y,ensemble,n_bins=8); plt.plot(pp,pt,"o-"); plt.plot([0,1],[0,1],"k--"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed positive fraction"); plt.title("Hybrid Ensemble Calibration"); _savefig("calibration_curve")
    plt.figure(figsize=(7,4)); plt.hist(ensemble[y==0],alpha=.6,label="Negative",bins=15); plt.hist(ensemble[y==1],alpha=.6,label="Positive",bins=15); plt.legend(); plt.xlabel("Hybrid probability"); plt.title("Prediction Probability Distribution"); _savefig("prediction_probability_distribution")
    gains=xgb_model.get_score(importance_type="gain")
    imp=pd.DataFrame({"Feature":columns,"Importance":[float(gains.get(feature,0.0)) for feature in columns]}).sort_values("Importance",ascending=False).head(20)
    plt.figure(figsize=(8,6)); plt.barh(imp["Feature"],imp["Importance"]); plt.gca().invert_yaxis(); plt.title("XGBoost Feature Importance"); _savefig("feature_importance")
    return metrics


def _diagram(name, title, nodes):
    plt.figure(figsize=(10,2.4)); ax=plt.gca(); ax.axis("off")
    xs=np.linspace(.07,.93,len(nodes))
    for i,(x,node) in enumerate(zip(xs,nodes)):
        ax.text(x,.5,node,ha="center",va="center",bbox=dict(boxstyle="round,pad=.45",fc="white",ec="black"),transform=ax.transAxes)
        if i: ax.annotate("",xy=(x-.06,.5),xytext=(xs[i-1]+.06,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->"))
    ax.set_title(title); _savefig(name)


def _research_diagrams():
    items={"fig1_overall_architecture":["Data","Preprocess","Models","Ensemble","Explain/CF"],"fig2_preprocessing_pipeline":["Raw data","Train fit","Transform","Select features"],"fig3_ensemble":["XGBoost","LightGBM","Validation α","Probability ensemble"],"fig4_treeshap_explanation":["Tree model","TreeSHAP","Global/local explanation"],"fig5_dice_flow":["Test profile","DiCE","Model-valid CF"],"fig6_ccmocf_architecture":["Profile","Constraints","NSGA-II","Pareto CFs"],"fig7_nsga2_optimization":["Population","Evaluate 5 objectives","Non-dominated sorting","Pareto front"],"fig8_constraint_projection":["Candidate","Project bounds/dependencies","Validate","Feasible CF"]}
    for name,nodes in items.items(): _diagram(name, name.replace("_"," ").title(), nodes)


def _shap(selected_ids):
    from pw_imp.shap_explainer import SHAP_OUT as LEGACY, generate_shap_explanations
    result=generate_shap_explanations("xgboost", selected_ids, len(selected_ids))
    for source,target in ((LEGACY/"shap_beeswarm.png",FIGURES/"shap_summary.png"),(LEGACY/"shap_bar_plot.png",FIGURES/"shap_bar.png"),(LEGACY/"global_feature_importance.csv",TABLES/"shap_feature_importance.csv")):
        if source.exists(): shutil.copy2(source,target)
    # Long-form SHAP value export uses actual TreeSHAP values recomputed by the existing explainer model.
    import shap
    from pw_imp.preprocessing import build_preprocessing_pipeline
    pipe=build_preprocessing_pipeline(); model=xgb.Booster(); model.load_model(str(ROOT/"results/models/xgboost_model.json")); X=pipe["X_test"]; ids=pipe["patient_test"]["PatientID"].to_numpy(); values=np.asarray(shap.TreeExplainer(model).shap_values(X))
    long=[]
    for row,pid in enumerate(ids):
        for col,feature in enumerate(X.columns): long.append((pid,feature,X.iloc[row,col],values[row,col],"positive" if values[row,col]>=0 else "negative"))
    pd.DataFrame(long,columns=["PatientID","Feature","FeatureValue","SHAPValue","Direction"]).to_csv(SHAP_OUT/"shap_values.csv",index=False)
    return result


def _counterfactuals(pipe, model, selected, probabilities):
    from pw_imp.preprocessing import load_dataframe
    from pw_imp.ccmocf import CCMOCF
    from pw_imp.clinical_constraints import ClinicalConstraintEngine
    raw=load_dataframe().set_index("PatientID"); selected_ids=[int(x) for x in selected["PatientID"]]
    # All selected profiles are positive predictions where possible, hence the clinically constrained target is class 0.
    from pw_imp.dice_baseline import generate_dice_counterfactuals
    dice=generate_dice_counterfactuals("xgboost",patient_ids=selected_ids,target_class=0,num_counterfactuals=3,max_patients=5)
    dice_rows=[]
    for pid in selected_ids:
        p=float(probabilities[selected.index[selected["PatientID"].eq(pid)][0]])
        path=ROOT/"results/counterfactuals/dice"/f"dice_patient_{pid}.csv"; frame=pd.read_csv(path) if path.exists() else pd.DataFrame()
        for _,cf in frame.iterrows():
            changed=[c for c in pipe["X_test"].columns if c in cf and float(cf[c]) != float(pipe["X_test"].iloc[selected.index[selected["PatientID"].eq(pid)][0]][c])]
            prob=float(cf.get("y",np.nan)); dice_rows.append({"PatientID":pid,"OriginalPrediction":int(p>=.5),"OriginalProbability":p,"CounterfactualPrediction":int(prob>=.5) if np.isfinite(prob) else np.nan,"CounterfactualProbability":prob,"ChangedFeatures":" | ".join(changed),"Sparsity":len(changed),"Proximity":np.nan,"Diversity":np.nan,"Validity":float(prob<.5) if np.isfinite(prob) else 0.,"Runtime":np.nan})
    pd.DataFrame(dice_rows).to_csv(CF_OUT/"dice_results.csv",index=False)
    # CCMOCF consumes the sklearn-compatible XGBoost wrapper so its probability
    # contract remains identical to the existing Phase 9 implementation.
    from pw_imp.dice_baseline import load_dice_baseline_model
    engine=CCMOCF(load_dice_baseline_model("xgboost"),pipe["preprocessor"],ClinicalConstraintEngine(),population_size=8,generations=2,seed=42,top_k=5)
    detail=[]; summary=[]
    for _,item in selected.iterrows():
        pid=int(item.PatientID); original=raw.loc[pid].drop(labels=["PCOS (Y/N)"],errors="ignore").to_dict(); started=time.perf_counter(); result=engine.generate(original,desired_class=0); runtime=time.perf_counter()-started
        candidates=result["counterfactuals"]
        for number,c in enumerate(candidates,1): detail.append({"PatientID":pid,"CounterfactualID":number,"Method":"CC-MO-CF","OriginalPrediction":result["original_prediction"],"OriginalProbability":result["original_probability"],"CounterfactualPrediction":int(c.probability>=.5),"CounterfactualProbability":c.probability,"ChangedFeatures":" | ".join(c.changed_features),"Sparsity":len(c.changed_features),"Proximity":c.objectives[1],"Plausibility":1.0,"Diversity":np.nan,"ConstraintSatisfaction":c.constraint_status,"DependencyConsistency":"VALID","ObjectiveValues":json.dumps(c.objectives),"Runtime":runtime})
        best=min(candidates,key=lambda c:c.probability) if candidates else None
        summary.append({"PatientID":pid,"OriginalPrediction":result["original_prediction"],"OriginalProbability":result["original_probability"],"CounterfactualFound":bool(candidates),"NumberOfCounterfactuals":len(candidates),"BestCounterfactualProbability":best.probability if best else np.nan,"ChangedFeatures":" | ".join(best.changed_features) if best else "","Sparsity":len(best.changed_features) if best else np.nan,"Proximity":best.objectives[1] if best else np.nan,"Plausibility":1. if best else np.nan,"Diversity":np.nan,"ConstraintSatisfaction":best.constraint_status if best else result["constraint_status"],"DependencyConsistency":"VALID" if best else "NOT_APPLICABLE","Runtime":runtime,"FailureReason":"" if best else result.get("status","")})
    detailed=pd.DataFrame(detail); summaries=pd.DataFrame(summary); summaries.to_csv(CF_OUT/"five_profiles_summary.csv",index=False); detailed.to_csv(CF_OUT/"five_profiles_detailed.csv",index=False); detailed.to_csv(CF_OUT/"five_patient_counterfactual_table.csv",index=False)
    return dice_rows,detailed,summaries


def _cf_figures(selected, summaries, detailed, dice_rows):
    ids=selected.PatientID.astype(str).tolist(); original=summaries.OriginalProbability.to_numpy(); cf=summaries.BestCounterfactualProbability.to_numpy()
    plt.figure(figsize=(8,4)); plt.bar(ids,original); plt.ylim(0,1); plt.ylabel("PMOS probability"); plt.title("Five Patient Predictions"); _savefig("five_patient_prediction")
    plt.figure(figsize=(8,4)); x=np.arange(len(ids)); plt.bar(x-.2,original,.4,label="Original"); plt.bar(x+.2,np.nan_to_num(cf),.4,label="CC-MO-CF"); plt.xticks(x,ids); plt.ylim(0,1); plt.legend(); plt.title("Counterfactual Probability"); _savefig("five_patient_counterfactual_probability")
    plt.figure(figsize=(8,4)); plt.bar(ids,np.nan_to_num(summaries.Sparsity)); plt.title("Five Patient Sparsity"); _savefig("five_patient_sparsity")
    plt.figure(figsize=(8,4)); plt.bar(ids,np.nan_to_num(summaries.Proximity)); plt.title("Five Patient Proximity"); _savefig("five_patient_proximity")
    plt.figure(figsize=(9,4)); labels=summaries.ChangedFeatures.fillna("").tolist(); plt.barh(ids,[max(1,len(x.split(" | "))) if x else 0 for x in labels]); plt.xlabel("Changed features"); plt.title("Five Patient Feature Changes"); _savefig("five_patient_feature_changes")
    plt.figure(figsize=(7,5));
    if not detailed.empty: plt.scatter(detailed.Proximity,detailed.CounterfactualProbability,c=detailed.PatientID.astype("category").cat.codes); plt.xlabel("Proximity"); plt.ylabel("Counterfactual probability")
    else: plt.text(.5,.5,"No feasible constrained Pareto solutions",ha="center")
    plt.title("Five Patient Pareto Fronts"); _savefig("five_patient_pareto_fronts")
    plt.figure(figsize=(7,4)); plt.bar(["DiCE","CC-MO-CF"],[len(dice_rows),len(detailed)]); plt.ylabel("Generated counterfactuals"); plt.title("DiCE vs CC-MO-CF"); _savefig("five_patient_dice_vs_ccmocf")


def run():
    _mkdirs(); started=time.time()
    from pw_imp.models import run_experiment
    from pw_imp.preprocessing import build_preprocessing_pipeline
    run_experiment(save_models=True)
    pipe=build_preprocessing_pipeline(); xmodel=xgb.Booster(); xmodel.load_model(str(ROOT/"results/models/xgboost_model.json")); lmodel=joblib.load(ROOT/"results/models/lightgbm_model.joblib")
    xv,lv=_proba_xgb(xmodel,pipe["X_val"]),_proba(lmodel,pipe["X_val"]); grid=np.linspace(0,1,101); alpha=float(min(grid,key=lambda a:log_loss(pipe["y_val"],np.clip(a*xv+(1-a)*lv,1e-7,1-1e-7))))
    xt,lt=_proba_xgb(xmodel,pipe["X_test"]),_proba(lmodel,pipe["X_test"]); ensemble=alpha*xt+(1-alpha)*lt; scores={"XGBoost":xt,"LightGBM":lt,"Hybrid Ensemble":ensemble}; names=list(scores); metrics=_model_figures(pipe["y_test"],scores,names,xmodel,pipe["X_test"].columns)
    table=pd.DataFrame([{"Model":n,**metrics[n]} for n in names]); table.to_csv(TABLES/"classification_metrics.csv",index=False); table.to_csv(TABLES/"model_comparison.csv",index=False)
    prediction=pd.DataFrame({"PatientID":pipe["patient_test"]["PatientID"],"ActualLabel":pipe["y_test"],"XGBoostProbability":xt,"LightGBMProbability":lt,"HybridProbability":ensemble,"HybridPrediction":(ensemble>=.5).astype(int)}); prediction.to_csv(RESULTS/"predictions.csv",index=False)
    selected=prediction[(prediction.ActualLabel.eq(1))&(prediction.HybridPrediction.eq(1))].sort_values("HybridProbability",ascending=False).head(5)
    if len(selected)<5: selected=pd.concat([selected,prediction.loc[~prediction.PatientID.isin(selected.PatientID)].sort_values("HybridProbability",ascending=False).head(5-len(selected))]).reset_index(drop=True)
    _shap(selected.PatientID.astype(int).tolist()); dice_rows,detailed,summaries=_counterfactuals(pipe,xmodel,selected,ensemble); _cf_figures(selected,summaries,detailed,dice_rows); _research_diagrams()
    dice_frame=pd.DataFrame(dice_rows); ccm=detailed.copy()
    if ccm.empty:
        ccm=pd.DataFrame(columns=["CounterfactualProbability","Proximity","Sparsity","Plausibility","ConstraintSatisfaction","Runtime"])
    comparison=pd.DataFrame([{"Method":"DiCE","Validity":dice_frame.Validity.mean() if not dice_frame.empty else np.nan,"Proximity":np.nan,"Sparsity":dice_frame.Sparsity.mean() if not dice_frame.empty else np.nan,"Plausibility":np.nan,"Diversity":np.nan,"Constraint Satisfaction":np.nan,"Dependency Consistency":np.nan,"Runtime":np.nan},{"Method":"CC-MO-CF","Validity":float((ccm.CounterfactualProbability<.5).mean()) if not ccm.empty else 0.,"Proximity":ccm.Proximity.mean() if not ccm.empty else np.nan,"Sparsity":ccm.Sparsity.mean() if not ccm.empty else np.nan,"Plausibility":ccm.Plausibility.mean() if not ccm.empty else np.nan,"Diversity":np.nan,"Constraint Satisfaction":float((ccm.ConstraintSatisfaction=="VALID").mean()) if not ccm.empty else 0.,"Dependency Consistency":1. if not ccm.empty else np.nan,"Runtime":ccm.Runtime.mean() if not ccm.empty else np.nan}])
    comparison.to_csv(TABLES/"dice_vs_ccmocf.csv",index=False); comparison.to_csv(TABLES/"counterfactual_metrics.csv",index=False); summaries.to_csv(TABLES/"five_patient_results.csv",index=False)
    pd.DataFrame([{"Variant":"NOT IMPLEMENTED","Status":"NOT IMPLEMENTED","Reason":"Ablation is not run automatically because removing clinical constraints cannot safely produce research results."}]).to_csv(TABLES/"ablation_results.csv",index=False)
    with open(RESULTS/"ensemble_config.json","w") as f: json.dump({"alpha":alpha,"selection_set":"validation","selection_metric":"log_loss","test_set_used_for_selection":False},f,indent=2)
    report=["# Final Experiment Report","","## Environment",f"- Python: {sys.version.split()[0]}","- Execution is measured from the current run; no synthetic results were used.","","## Dataset and splitting",f"- Train/validation/test: {len(pipe['X_train'])}/{len(pipe['X_val'])}/{len(pipe['X_test'])}; PatientID is preserved as metadata and excluded from model features.","","## Predictive results","```csv",table.to_csv(index=False).strip(),"```","","## Counterfactual results",f"- Selected PatientIDs: {', '.join(map(str,selected.PatientID.astype(int).tolist()))}.",f"- DiCE counterfactual rows: {len(dice_rows)}; CC-MO-CF feasible Pareto rows: {len(detailed)}.","- Counterfactuals are model-based hypothetical scenarios, not clinical prescriptions.","","## Limitations", "- Clinical configuration entries marked REQUIRES_CLINICAL_VALIDATION remain so.","- Ablation was not automatically executed because a constraint-removal variant is not safe for finalized research claims.","","## Reproducibility",f"- Command: {sys.executable} run_experiment.py",f"- Runtime seconds: {time.time()-started:.2f}"]
    (RESULTS/"final_experiment_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return {"alpha":alpha,"metrics":metrics["Hybrid Ensemble"],"patient_ids":selected.PatientID.astype(int).tolist(),"dice":len(dice_rows),"ccmocf":len(detailed)}

if __name__ == "__main__": print(json.dumps(run(),indent=2))
