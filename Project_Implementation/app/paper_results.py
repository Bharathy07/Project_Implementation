"""Read-only Streamlit dashboard for finalized paper artifacts.

Run: python -m streamlit run app/paper_results.py
"""
from __future__ import annotations

import html
import platform
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES, TABLES, CF = RESULTS / "figures", RESULTS / "tables", RESULTS / "counterfactuals"

TABLE_FILES = {
    "Table 1 — Dataset Statistics": "dataset_statistics.csv",
    "Table 2 — Classification Performance": "classification_metrics.csv",
    "Table 3 — Model Comparison": "model_comparison.csv",
    "Table 4 — SHAP Feature Importance": "shap_feature_importance.csv",
    "Table 5 — DiCE vs CC-MO-CF": "dice_vs_ccmocf.csv",
    "Table 6 — Counterfactual Metrics": "counterfactual_metrics.csv",
    "Table 7 — Five-Profile Counterfactual Results": "five_patient_results.csv",
    "Table 8 — Ablation Results": "ablation_results.csv",
}
FIGURE_FILES = [
    ("Fig. 1 Overall Architecture", "fig1_overall_architecture.png"), ("Fig. 2 Preprocessing Pipeline", "fig2_preprocessing_pipeline.png"),
    ("Fig. 3 Hybrid XGBoost-LightGBM Ensemble", "fig3_ensemble.png"), ("Fig. 4 TreeSHAP Explanation", "fig4_treeshap_explanation.png"),
    ("Fig. 5 DiCE Counterfactual Generation", "fig5_dice_flow.png"), ("Fig. 6 CC-MO-CF Architecture", "fig6_ccmocf_architecture.png"),
    ("Fig. 7 NSGA-II Optimization", "fig7_nsga2_optimization.png"), ("Fig. 8 Clinical Constraint Projection", "fig8_constraint_projection.png"),
    ("Accuracy Comparison", "accuracy_comparison.png"), ("Confusion Matrix", "confusion_matrix.png"), ("ROC Curve", "roc_curve.png"),
    ("Precision-Recall Curve", "precision_recall_curve.png"), ("Calibration Curve", "calibration_curve.png"), ("Feature Importance", "feature_importance.png"),
    ("SHAP Summary", "shap_summary.png"), ("SHAP Bar", "shap_bar.png"), ("Five-Profile Prediction", "five_patient_prediction.png"),
    ("Five-Profile Counterfactual", "five_patient_counterfactual_probability.png"), ("Feature Changes", "five_patient_feature_changes.png"),
    ("Pareto Front", "five_patient_pareto_fronts.png"), ("DiCE vs CC-MO-CF", "five_patient_dice_vs_ccmocf.png"), ("Ablation Study", "ablation_comparison.png"),
]


def read_csv(path: Path) -> pd.DataFrame:
    try: return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def dataset_stats() -> pd.DataFrame:
    try:
        raw = pd.read_excel(ROOT / "PCOS_data_without_infertility.xlsx", sheet_name="Full_new", engine="openpyxl")
        target = "PCOS (Y/N)"
        return pd.DataFrame({"Statistic": ["Records", "Features excluding target/PatientID", "Missing values", "Duplicate records", "Positive class", "Negative class"], "Value": [len(raw), max(0, len(raw.columns)-2), int(raw.isna().sum().sum()), int(raw.duplicated().sum()), int((raw[target] == 1).sum()) if target in raw else "N/A", int((raw[target] == 0).sum()) if target in raw else "N/A"]})
    except Exception as exc:
        return pd.DataFrame({"Statistic": ["Dataset read status"], "Value": [f"Unavailable: {exc}"]})


def constraints() -> pd.DataFrame:
    with open(ROOT / "config" / "clinical_constraints.yaml", encoding="utf-8") as f: raw = yaml.safe_load(f) or {}
    rows=[]
    for feature, spec in raw.get("feature", {}).items():
        rows.append({"Feature":feature,"Constraint type":spec.get("type"),"Lower bound":spec.get("lower_bound"),"Upper bound":spec.get("upper_bound"),"Maximum change":spec.get("max_step"),"Actionability":spec.get("actionable"),"Dependency rule":spec.get("dependency"),"Validation status":spec.get("validation_status")})
    return pd.DataFrame(rows)


def _text_table(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False) if not frame.empty else "No actual output was generated."


def export_results() -> tuple[Path, Path]:
    metrics=read_csv(TABLES / "classification_metrics.csv"); dice=read_csv(CF / "dice_results.csv"); five=read_csv(CF / "five_profiles_summary.csv"); compare=read_csv(TABLES / "dice_vs_ccmocf.csv"); ablation=read_csv(TABLES / "ablation_results.csv")
    sections=[("Dataset Results",dataset_stats()),("Classification Performance",metrics),("DiCE Results",dice),("CC-MO-CF and Five-Profile Results",five),("DiCE vs CC-MO-CF",compare),("Ablation Results",ablation)]
    lines=["# Final Paper Results", "", "All numeric values below are loaded from existing finalized output files. No experiments were retrained for this export.", ""]
    for title, frame in sections: lines += [f"## {title}", "", "```csv", _text_table(frame).strip(), "```", ""]
    lines += ["## Important Observations", "", f"- DiCE rows: {len(dice)}.", f"- CC-MO-CF feasible rows: {int(five.get('NumberOfCounterfactuals', pd.Series(dtype=float)).sum()) if not five.empty else 0}.", "- Counterfactuals are model-based hypothetical scenarios, not clinical recommendations.", "", "## Limitations", "", "- Clinical constraints marked `REQUIRES_CLINICAL_VALIDATION` remain unvalidated by this software dashboard.", "- Ablation entries are displayed only when actually executed."]
    markdown="\n".join(lines)+"\n"; md=RESULTS / "final_paper_results.md"; md.write_text(markdown,encoding="utf-8")
    html_sections="".join(f"<h2>{html.escape(title)}</h2><pre>{html.escape(_text_table(frame))}</pre>" for title,frame in sections)
    page=f"<!doctype html><html><head><meta charset='utf-8'><title>Final Paper Results</title><style>body{{font-family:Arial;margin:2rem}}pre{{overflow:auto;border:1px solid #ddd;padding:1rem}}</style></head><body><h1>Final Paper Results</h1><p>Loaded from finalized project outputs; no experiments rerun.</p>{html_sections}</body></html>"
    hp=RESULTS / "final_paper_results.html"; hp.write_text(page,encoding="utf-8")
    return md,hp


def main():
    import streamlit as st
    st.set_page_config(page_title="PMOS Paper Results", layout="wide")
    st.title("Explainable PMOS Prediction Through Clinically Constrained Multi-Objective Counterfactual Optimization")
    metrics=read_csv(TABLES/"classification_metrics.csv"); five=read_csv(CF/"five_profiles_summary.csv"); dice=read_csv(CF/"dice_results.csv"); comparison=read_csv(TABLES/"dice_vs_ccmocf.csv")
    st.header("1. Experiment Overview")
    ds=dataset_stats(); splits={n:len(read_csv(RESULTS/"splits"/f"{n}_patient_ids.csv")) for n in ("train","validation","test")}; c1,c2,c3,c4=st.columns(4); c1.metric("Records",ds.loc[ds.Statistic.eq("Records"),"Value"].iloc[0] if not ds.empty else "N/A"); c2.metric("Train",splits["train"]); c3.metric("Validation",splits["validation"]); c4.metric("Test",splits["test"])
    st.write("Algorithms: XGBoost, LightGBM, Hybrid Ensemble, TreeSHAP, DiCE, CC-MO-CF, NSGA-II")
    import numpy, pymoo
    st.caption(f"Python {platform.python_version()} · NumPy {numpy.__version__} · pandas {pd.__version__} · pymoo {pymoo.__version__}")
    st.header("2. Dataset Results"); st.dataframe(ds, use_container_width=True)
    st.header("3. Model Performance"); st.dataframe(metrics, use_container_width=True); show_images(st, ["accuracy_comparison.png","metrics_comparison.png","confusion_matrix.png","roc_curve.png","precision_recall_curve.png","calibration_curve.png","prediction_probability_distribution.png"])
    st.header("4. Feature Importance"); imp=read_csv(TABLES/"shap_feature_importance.csv"); st.dataframe(imp.head(15),use_container_width=True); show_images(st,["feature_importance.png"])
    st.header("5. TreeSHAP Explainability"); show_images(st,["shap_summary.png","shap_bar.png"]); patient=read_csv(FIGURES/"shap"/"patient_explanations.csv"); st.dataframe(patient,use_container_width=True)
    if not patient.empty:
        pid=st.selectbox("PatientID",patient.PatientID.tolist()); row=patient[patient.PatientID.eq(pid)].iloc[0]; st.json(row.to_dict()); image=FIGURES/"shap"/f"waterfall_patient_{pid}.png"; st.image(str(image),caption="Positive SHAP values push prediction toward PMOS; negative values push it away. They are not causal effects.") if image.exists() else st.info("Individual waterfall plot not generated.")
    st.header("6. DiCE Counterfactual Baseline"); st.metric("DiCE counterfactuals",len(dice)); st.dataframe(dice,use_container_width=True)
    st.header("7. CC-MO-CF Results"); st.metric("Feasible CC-MO-CF counterfactuals",int(five.get("NumberOfCounterfactuals",pd.Series(dtype=float)).sum()) if not five.empty else 0); st.dataframe(five,use_container_width=True)
    st.header("8. DiCE vs CC-MO-CF"); st.dataframe(comparison,use_container_width=True); show_images(st,["five_patient_dice_vs_ccmocf.png"])
    st.header("9. Five Individual Profiles"); st.dataframe(five,use_container_width=True)
    st.header("10–13. Five-Profile and Pareto Visualizations"); show_images(st,["five_patient_prediction.png","five_patient_counterfactual_probability.png","five_patient_feature_changes.png","five_patient_pareto_fronts.png"]); st.caption("The Pareto plot is a 2D projection of the higher-dimensional objective space.")
    st.header("14. Clinical Constraints"); st.dataframe(constraints(),use_container_width=True)
    st.header("15. Ablation Study"); st.dataframe(read_csv(TABLES/"ablation_results.csv"),use_container_width=True); show_images(st,["ablation_comparison.png"])
    st.header("16. Final Paper Tables")
    for label,name in TABLE_FILES.items():
        frame=dataset_stats() if name=="dataset_statistics.csv" else read_csv(TABLES/name); st.subheader(label); st.dataframe(frame,use_container_width=True); st.download_button(f"Download {name}",frame.to_csv(index=False),name,"text/csv",key=name)
    st.header("17. Final Paper Figures")
    for label,name in FIGURE_FILES:
        st.subheader(label); path=FIGURES/name; st.image(str(path)) if path.exists() else st.info("Figure not generated")
    st.header("18. Paper-Ready Results Summary")
    final=metrics[metrics.Model.eq("Hybrid Ensemble")].iloc[0] if not metrics.empty else pd.Series(dtype=object); st.write({k:final.get(k,"N/A") for k in ["Model","Accuracy","Precision","Recall","F1-score","ROC-AUC","MCC","Balanced Accuracy"]}); st.write({"Profiles analyzed":len(five),"Feasible CC-MO-CF":int(five.get("NumberOfCounterfactuals",pd.Series(dtype=float)).sum()) if not five.empty else 0})
    md,hp=export_results(); st.header("19. Export"); st.download_button("Download final_paper_results.md",md.read_text(encoding="utf-8"),"final_paper_results.md"); st.download_button("Download final_paper_results.html",hp.read_text(encoding="utf-8"),"final_paper_results.html")


def show_images(st, names):
    for name in names:
        path=FIGURES/name
        if path.exists(): st.image(str(path), caption=name.replace("_"," ").replace(".png",""))
        else: st.info(f"Figure not generated: {name}")


if __name__ == "__main__": main()
