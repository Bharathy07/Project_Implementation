"""Read-only five-case presentation of existing CC-MO-CF results.

Run: python -m streamlit run app/counterfactual_cases.py
Use ``python app/counterfactual_cases.py --export`` to refresh the CSV/report
from the stored artifacts without running any model or optimizer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CF = ROOT / "results" / "counterfactuals"
FIGURES = ROOT / "results" / "figures"


def _read(path: Path) -> pd.DataFrame:
    try: return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def cases() -> pd.DataFrame:
    """The five actual finalized profile attempts, never duplicated or synthesized."""
    return _read(CF / "five_profiles_summary.csv").head(5).copy()


def raw_profiles(patient_ids: list[int]) -> dict[int, dict]:
    try:
        raw = pd.read_excel(ROOT / "PCOS_data_without_infertility.xlsx", sheet_name="Full_new", engine="openpyxl")
        raw.columns = [str(column).strip() for column in raw.columns]
        if "Patient File No." in raw: raw = raw.rename(columns={"Patient File No.": "PatientID"})
        raw = raw.set_index("PatientID")
        return {int(pid): raw.loc[pid].drop(labels=["PCOS (Y/N)"], errors="ignore").dropna().to_dict() for pid in patient_ids if pid in raw.index}
    except Exception:
        return {}


def _constraints() -> dict:
    with open(ROOT / "config" / "clinical_constraints.yaml", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("feature", {})


def _successful() -> pd.DataFrame:
    """Only stored valid Pareto rows; absent rows must not become suggestions."""
    saved = _read(CF / "ccmocf" / "ccmocf_pareto_solutions.csv")
    if saved.empty or "feasibility_status" not in saved: return pd.DataFrame()
    return saved[saved["feasibility_status"].eq("VALID_FEASIBLE")].copy()


def export_artifacts() -> tuple[Path, Path]:
    selected = cases(); profiles = raw_profiles(selected.PatientID.astype(int).tolist()) if not selected.empty else {}; valid = _successful()
    rows=[]; report=["# Five-Case CC-MO-CF Counterfactual Suggestions", "", "Model-based hypothetical counterfactual scenarios generated using clinically constrained multi-objective optimization.", "", "## Availability", "", "No feasible CC-MO-CF counterfactual was stored for the five selected finalized attempts. Therefore no feature change is presented as a suggestion.", ""]
    for number, (_, case) in enumerate(selected.iterrows(), 1):
        pid=int(case.PatientID); matching=valid[valid.PatientID.astype(int).eq(pid)] if not valid.empty and "PatientID" in valid else pd.DataFrame()
        report += [f"## Case {number}", "", f"- PatientID: {pid}", f"- Original prediction: {case.OriginalPrediction}", f"- Original probability: {case.OriginalProbability}", f"- Counterfactual found: {bool(case.CounterfactualFound)}", ""]
        if matching.empty:
            report += ["No valid CC-MO-CF suggestion was generated: `NO_PREDICTION_FLIP` under the configured clinical, actionability, and dependency constraints.", ""]
            rows.append({"Case":f"Case {number}","PatientID":pid,"OriginalPrediction":case.OriginalPrediction,"OriginalProbability":case.OriginalProbability,"CounterfactualID":pd.NA,"CounterfactualPrediction":pd.NA,"CounterfactualProbability":pd.NA,"ProbabilityChange":pd.NA,"ChangedFeatures":"","OriginalValues":json.dumps({}),"CounterfactualValues":json.dumps({}),"Sparsity":pd.NA,"Proximity":pd.NA,"Plausibility":pd.NA,"Diversity":pd.NA,"ConstraintSatisfaction":case.ConstraintSatisfaction,"DependencyConsistency":case.DependencyConsistency})
        else:
            for identifier, (_, solution) in enumerate(matching.head(3).iterrows(), 1):
                # Stored Phase-9 rows do not preserve raw candidate feature values; do not invent them.
                report += [f"### Counterfactual {'ABC'[identifier-1]}", "", f"- Changed features: {solution.get('changed_features','')}", f"- Predicted probability: {solution.get('desired_class_probability','')}", ""]
                rows.append({"Case":f"Case {number}","PatientID":pid,"OriginalPrediction":case.OriginalPrediction,"OriginalProbability":case.OriginalProbability,"CounterfactualID":solution.get("CounterfactualID",identifier),"CounterfactualPrediction":solution.get("counterfactual_prediction"),"CounterfactualProbability":solution.get("desired_class_probability"),"ProbabilityChange":solution.get("desired_class_probability",pd.NA)-case.OriginalProbability,"ChangedFeatures":solution.get("changed_features",""),"OriginalValues":json.dumps({}),"CounterfactualValues":json.dumps({}),"Sparsity":solution.get("number_changed_features"),"Proximity":pd.NA,"Plausibility":pd.NA,"Diversity":pd.NA,"ConstraintSatisfaction":solution.get("constraint_status"),"DependencyConsistency":"NOT_RECORDED"})
    output=pd.DataFrame(rows); csv_path=CF / "five_case_counterfactual_suggestions.csv"; output.to_csv(csv_path,index=False)
    report += ["## Five-Case Summary", "", "```csv", output.to_csv(index=False).strip(), "```", "", "These outputs are model-based hypothetical scenarios and are not medical prescriptions."]
    report_path=CF / "five_case_counterfactual_report.md"; report_path.write_text("\n".join(report)+"\n",encoding="utf-8")
    return csv_path, report_path


def main():
    import streamlit as st
    st.set_page_config(page_title="CC-MO-CF Five Case Analysis",layout="wide")
    st.title("CC-MO-CF Counterfactual Suggestions — Five Case Analysis")
    st.caption("Model-based hypothetical counterfactual scenarios generated using clinically constrained multi-objective optimization.")
    selected=cases(); profiles=raw_profiles(selected.PatientID.astype(int).tolist()) if not selected.empty else {}; specs=_constraints(); valid=_successful()
    if selected.empty:
        st.error("No finalized five-profile counterfactual output is available."); return
    st.warning("No feasible CC-MO-CF solution is stored for these five finalized attempts. The cases below are displayed as actual failed attempts; no counterfactual values are fabricated.")
    for number, (_, case) in enumerate(selected.iterrows(),1):
        pid=int(case.PatientID); st.header(f"CASE {number}")
        c1,c2,c3,c4=st.columns(4); c1.metric("PatientID (metadata)",pid); c2.metric("Original PMOS Prediction",int(case.OriginalPrediction)); c3.metric("Original PMOS Probability",f"{case.OriginalProbability:.4f}"); c4.metric("Counterfactual Found",str(bool(case.CounterfactualFound)))
        st.subheader("Original Profile — predictive variables")
        profile=profiles.get(pid,{})
        if profile: st.dataframe(pd.DataFrame({"Feature":list(profile),"Original Value":list(profile.values())}),use_container_width=True)
        else: st.info("Original profile values are unavailable in the stored data artifact.")
        matching=valid[valid.PatientID.astype(int).eq(pid)] if not valid.empty and "PatientID" in valid else pd.DataFrame()
        if matching.empty:
            st.info(f"No valid suggestion: {case.get('FailureReason','NO_PREDICTION_FLIP')}. Rejected candidates are not displayed as feasible counterfactuals.")
        else:
            for identifier, (_, solution) in enumerate(matching.head(3).iterrows()):
                st.subheader(f"Counterfactual {'ABC'[identifier]}"); st.write({"Prediction":solution.get("counterfactual_prediction"),"Probability":solution.get("desired_class_probability"),"Changed Features":solution.get("changed_features"),"Sparsity":solution.get("number_changed_features"),"Clinical Constraints":"PASS" if solution.get("constraint_status")=="VALID" else "FAIL","Dependency Consistency":"PASS" if solution.get("feasibility_status")=="VALID_FEASIBLE" else "FAIL"})
    st.header("Five-Case Comparison")
    summary=selected[[column for column in ["PatientID","OriginalProbability","BestCounterfactualProbability","ChangedFeatures","Sparsity","Proximity","Plausibility","ConstraintSatisfaction","DependencyConsistency"] if column in selected]]; st.dataframe(summary,use_container_width=True)
    st.header("Visualizations")
    for name in ("five_patient_counterfactual_probability.png","five_patient_feature_changes.png","five_patient_pareto_fronts.png"):
        path=FIGURES/name
        st.image(str(path),caption=name.replace("_"," ")) if path.exists() else st.info(f"Figure not generated: {name}")
    csv_path, report_path=export_artifacts(); st.download_button("Download five-case suggestions CSV",csv_path.read_text(encoding="utf-8"),csv_path.name,"text/csv"); st.download_button("Download five-case report",report_path.read_text(encoding="utf-8"),report_path.name,"text/markdown")
    st.caption("Counterfactual outputs are model-based hypothetical scenarios, not medical recommendations.")


if __name__ == "__main__":
    if "--export" in sys.argv: print(*export_artifacts(),sep="\n")
    else: main()
