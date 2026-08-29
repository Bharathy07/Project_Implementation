"""Phase 10 evaluation of saved DiCE and CC-MO-CF counterfactual artifacts.

This module never generates counterfactuals.  It evaluates only existing outputs;
therefore an absent Phase 9 run produces an honest unavailable comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DICE_DIR = REPO_ROOT / "results" / "counterfactuals" / "dice"
PHASE9_DIR = REPO_ROOT / "results" / "experiments" / "phase9_ccmocf"
EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"

COMPARISON_UNAVAILABLE = "DiCE comparison unavailable because the actual DiCE implementation could not be executed."

METRIC_DEFINITIONS = {
    "validity": "Fraction of returned counterfactuals whose model prediction equals the recorded desired target class.",
    "sparsity": "Number of changed features; lower is better.",
    "proximity": "Mean normalized absolute distance from the original row across numeric features; lower is better.",
    "plausibility": "Fraction of counterfactuals inside configured feature bounds and valid categorical domains; unavailable when source rows are absent.",
    "actionability": "Fraction of changed features marked actionable by clinical_constraints.yaml.",
    "constraint_validity": "Whether every configured hard constraint validates for the counterfactual.",
    "dependency_consistency": "Whether every configured derived feature matches its dependency formula.",
    "diversity": "Mean pairwise normalized distance among counterfactuals for the same patient; higher indicates more variety.",
    "runtime": "Recorded counterfactual generation runtime in seconds.",
    "feasible_counterfactual_rate": "Patients with at least one valid feasible counterfactual divided by eligible patients.",
    "plausibility_method": "Configured bounds, categorical validity, and dependency consistency; no unconfigured medical thresholds are inferred.",
}

FIGURE_METRICS = {
    "metric_comparison": ["validity", "sparsity", "proximity", "plausibility", "actionability", "constraint_validity", "dependency_consistency", "diversity", "runtime", "feasible_counterfactual_rate"],
    "validity_comparison": ["validity"],
    "sparsity_comparison": ["sparsity"],
    "proximity_comparison": ["proximity"],
    "plausibility_comparison": ["plausibility"],
    "feasibility_failure_distribution": ["feasible_counterfactual_rate"],
}


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_dice() -> tuple[pd.DataFrame, set[int], str | None]:
    summary_path = DICE_DIR / "dice_summary.csv"
    if not summary_path.exists():
        return _empty_frame(["PatientID", "status", "n_counterfactuals", "target_class"]), set(), COMPARISON_UNAVAILABLE
    summary = pd.read_csv(summary_path)
    eligible = set()
    for patient_id in summary.loc[summary["status"].astype(str).str.lower().eq("generated"), "PatientID"].astype(int):
        if (DICE_DIR / f"dice_patient_{patient_id}.csv").exists():
            eligible.add(int(patient_id))
    return summary, eligible, None if eligible else COMPARISON_UNAVAILABLE


def _read_ccmocf() -> tuple[pd.DataFrame, set[int], str | None]:
    path = PHASE9_DIR / "ccmocf_successful.csv"
    attempts_path = PHASE9_DIR / "ccmocf_all_attempts.csv"
    if not path.exists():
        return _empty_frame([]), set(), "Phase 9 CC-MO-CF outputs are unavailable; Phase 10 comparison cannot be paired."
    successful = pd.read_csv(path)
    eligible = set(successful["PatientID"].astype(int)) if "PatientID" in successful.columns else set()
    return successful, eligible, None if eligible else "Phase 9 contains no successful CC-MO-CF counterfactuals."


def _numeric(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _changed_features(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _metric_rows(method: str, patient_ids: set[int], dice_summary: pd.DataFrame, ccmocf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id in sorted(patient_ids):
        if method == "DiCE":
            path = DICE_DIR / f"dice_patient_{patient_id}.csv"
            if not path.exists():
                continue
            counterfactuals = pd.read_csv(path)
            target = int(dice_summary.loc[dice_summary["PatientID"].astype(int).eq(patient_id), "target_class"].iloc[0])
            for cf_id, row in counterfactuals.iterrows():
                probability = _numeric(row.get("y"))
                rows.append({"PatientID": patient_id, "Method": method, "CounterfactualID": cf_id + 1, "target_class": target, "validity": float(int(probability == target)) if probability is not None else np.nan, "sparsity": np.nan, "proximity": np.nan, "plausibility": np.nan, "actionability": np.nan, "constraint_validity": np.nan, "dependency_consistency": np.nan, "diversity": np.nan, "runtime": np.nan, "feasible": np.nan, "evaluation_status": "partial_artifact_no_original_raw_row"})
        else:
            patient_rows = ccmocf[ccmocf["PatientID"].astype(int).eq(patient_id)] if "PatientID" in ccmocf.columns else ccmocf.iloc[0:0]
            for _, row in patient_rows.iterrows():
                changed = _changed_features(row.get("changed_features"))
                rows.append({"PatientID": patient_id, "Method": method, "CounterfactualID": row.get("CounterfactualID"), "target_class": row.get("target_class"), "validity": float(row.get("counterfactual_prediction") == row.get("target_class")), "sparsity": row.get("number_changed_features", len(changed)), "proximity": np.nan, "plausibility": np.nan, "actionability": np.nan, "constraint_validity": float(row.get("constraint_status") == "VALID"), "dependency_consistency": np.nan, "diversity": np.nan, "runtime": row.get("Runtime"), "feasible": float(row.get("feasibility_status") == "VALID_FEASIBLE"), "evaluation_status": "phase9_saved_summary"})
    return pd.DataFrame(rows)


def _save_figures(per_cf: pd.DataFrame, output: Path, unavailable: bool) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    saved = []
    for figure_name, metrics in FIGURE_METRICS.items():
        figure, axis = plt.subplots(figsize=(12, 6))
        if unavailable or per_cf.empty:
            axis.text(0.5, 0.5, COMPARISON_UNAVAILABLE, ha="center", va="center", wrap=True)
            axis.set_axis_off()
        else:
            means = per_cf.groupby("Method")[metrics].mean(numeric_only=True).T
            means.plot(kind="bar", ax=axis)
            axis.set_ylabel("Mean metric value")
            axis.set_title(figure_name.replace("_", " ").title())
            axis.tick_params(axis="x", labelrotation=45)
            axis.legend(title="Method")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{figure_name}.{suffix}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(str(path))
        plt.close(figure)
    return saved


def run_phase10(output_dir: str | Path = EVALUATION_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dice_summary, dice_ids, dice_status = _read_dice()
    ccmocf, ccmocf_ids, ccmocf_status = _read_ccmocf()
    intersection = sorted(dice_ids & ccmocf_ids)
    pd.DataFrame({"PatientID": intersection, "eligible_DiCE": True, "eligible_CC_MO_CF": True}).to_csv(output / "eligibility_intersection.csv", index=False)
    per_cf = pd.concat([_metric_rows("DiCE", set(intersection), dice_summary, ccmocf), _metric_rows("CC-MO-CF", set(intersection), dice_summary, ccmocf)], ignore_index=True)
    per_cf.to_csv(output / "cf_metrics_per_counterfactual.csv", index=False)
    aggregate_columns = ["Method", "validity", "sparsity", "proximity", "plausibility", "actionability", "constraint_validity", "dependency_consistency", "diversity", "runtime", "feasible_counterfactual_rate"]
    patient = per_cf.groupby(["PatientID", "Method"], as_index=False).mean(numeric_only=True) if not per_cf.empty else _empty_frame(["PatientID", "Method"] + aggregate_columns[1:])
    patient.to_csv(output / "cf_metrics_per_patient.csv", index=False)
    comparison_rows = []
    for metric in aggregate_columns[1:]:
        dice_value = float(per_cf.loc[per_cf["Method"] == "DiCE", metric].mean()) if metric in per_cf and not per_cf.loc[per_cf["Method"] == "DiCE", metric].dropna().empty else np.nan
        ccmocf_value = float(per_cf.loc[per_cf["Method"] == "CC-MO-CF", metric].mean()) if metric in per_cf and not per_cf.loc[per_cf["Method"] == "CC-MO-CF", metric].dropna().empty else np.nan
        comparison_rows.append({"Metric": metric, "DiCE": dice_value, "CC-MO-CF": ccmocf_value, "Difference": ccmocf_value - dice_value if not np.isnan(dice_value) and not np.isnan(ccmocf_value) else np.nan, "comparison_status": "paired" if intersection else "unavailable_no_eligibility_intersection"})
    pd.DataFrame(comparison_rows).to_csv(output / "ccmocf_vs_dice_summary.csv", index=False)
    with open(output / "metric_definitions.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": METRIC_DEFINITIONS, "dice_status": dice_status, "ccmocf_status": ccmocf_status, "comparison_status": COMPARISON_UNAVAILABLE if not intersection else "paired_no_statistical_test"}, handle, indent=2)
    figures = _save_figures(per_cf, output, unavailable=not intersection)
    return {"dice_status": dice_status, "ccmocf_status": ccmocf_status, "intersection": intersection, "output_dir": output, "figures": figures, "comparison_status": COMPARISON_UNAVAILABLE if not intersection else "paired_no_statistical_test"}


if __name__ == "__main__":
    result = run_phase10()
    print(result["comparison_status"])
    print(f"Eligibility intersection: {len(result['intersection'])} patients")