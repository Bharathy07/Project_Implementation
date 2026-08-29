"""Phase 14 read-only statistical analysis of finalized result tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"
ABLATION_DIR = REPO_ROOT / "results" / "ablation"
STATISTICS_DIR = REPO_ROOT / "results" / "statistics"

DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 42
ALPHA = 0.05


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _bootstrap_ci(values: np.ndarray, iterations: int, seed: int, statistic=np.mean) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True)
    estimates = statistic(samples, axis=1)
    return float(statistic(values)), float(np.quantile(estimates, ALPHA / 2)), float(np.quantile(estimates, 1 - ALPHA / 2))


def _holm(pvalues: Iterable[float]) -> list[float]:
    indexed = sorted((index, value) for index, value in enumerate(pvalues) if np.isfinite(value))
    adjusted = [np.nan] * len(list(pvalues)) if not isinstance(pvalues, list) else [np.nan] * len(pvalues)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        corrected = min(1.0, (total - rank) * value)
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def _paired_test(left: np.ndarray, right: np.ndarray, metric: str, bootstrap_iterations: int, seed: int) -> dict:
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    base = {"Metric": metric, "SampleSize": int(left.size), "TestUsed": "NOT_RUN", "Difference": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan, "PValue": np.nan, "MissingExcluded": int((~finite).sum()), "ExclusionReason": "non-finite paired values removed" if (~finite).sum() else ""}
    if left.size < 5:
        base["ExclusionReason"] = "paired sample size < 5; test inappropriate"
        return base
    differences = right - left
    base["Difference"] = float(np.median(differences))
    _, base["CI_Lower"], base["CI_Upper"] = _bootstrap_ci(differences, bootstrap_iterations, seed + len(metric))
    try:
        from scipy.stats import binomtest, wilcoxon

        if metric.lower() in {"validity", "feasibility", "feasibilityrate", "feasible_counterfactual_rate"}:
            left_binary = left >= 0.5
            right_binary = right >= 0.5
            discordant_left = int(np.sum(left_binary & ~right_binary))
            discordant_right = int(np.sum(~left_binary & right_binary))
            discordant = discordant_left + discordant_right
            base["TestUsed"] = "McNemar exact binomial"
            base["PValue"] = float(1.0 if discordant == 0 else binomtest(min(discordant_left, discordant_right), discordant, 0.5).pvalue)
        elif np.allclose(differences, 0):
            base["TestUsed"] = "Wilcoxon signed-rank (all differences zero)"
            base["PValue"] = 1.0
        else:
            statistic, pvalue = wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
            base["TestUsed"] = "Wilcoxon signed-rank"
            base["PValue"] = float(pvalue)
    except ImportError:
        base["ExclusionReason"] = "scipy unavailable; paired test not executed"
    return base


def run_phase14(
    output_dir: str | Path = STATISTICS_DIR,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Analyze existing finalized evaluation/ablation outputs without rerunning them."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ci_rows = []
    comparison_rows = []
    effect_rows = []
    missing = []

    per_patient = _read(EVALUATION_DIR / "cf_metrics_per_patient.csv")
    per_counterfactual = _read(EVALUATION_DIR / "cf_metrics_per_counterfactual.csv")
    if per_patient.empty:
        missing.append({"Input": str(EVALUATION_DIR / "cf_metrics_per_patient.csv"), "Reason": "finalized Phase 10 patient metrics are absent"})
    else:
        method_column = "Method"
        metrics = [name for name in ("validity", "sparsity", "proximity", "plausibility", "actionability", "constraint_validity", "dependency_consistency", "diversity", "runtime", "feasible_counterfactual_rate") if name in per_patient.columns]
        for method in per_patient[method_column].dropna().unique():
            method_values = per_patient[per_patient[method_column] == method]
            for metric in metrics:
                values = pd.to_numeric(method_values[metric], errors="coerce").to_numpy()
                estimate, lower, upper = _bootstrap_ci(values, bootstrap_iterations, seed + len(ci_rows))
                ci_rows.append({"Method": method, "Metric": metric, "SampleSize": int(np.isfinite(values).sum()), "Estimate": estimate, "CI_Lower": lower, "CI_Upper": upper, "BootstrapIterations": bootstrap_iterations, "Seed": seed, "MissingExcluded": int((~np.isfinite(values)).sum())})
        if {"PatientID", method_column}.issubset(per_patient.columns):
            pivot = per_patient.pivot_table(index="PatientID", columns=method_column, values=metrics, aggfunc="mean")
            for metric in metrics:
                if (metric, "DiCE") in pivot.columns and (metric, "CC-MO-CF") in pivot.columns:
                    comparison_rows.append(_paired_test(pivot[(metric, "DiCE")].to_numpy(), pivot[(metric, "CC-MO-CF")].to_numpy(), metric, bootstrap_iterations, seed))
                    differences = pivot[(metric, "CC-MO-CF")].to_numpy() - pivot[(metric, "DiCE")].to_numpy()
                    effect_rows.append({"Metric": metric, "Effect": float(np.nanmedian(differences)), "EffectDefinition": "median paired CC-MO-CF minus DiCE", "SampleSize": int(np.isfinite(differences).sum())})
    if per_counterfactual.empty:
        missing.append({"Input": str(EVALUATION_DIR / "cf_metrics_per_counterfactual.csv"), "Reason": "finalized Phase 10 counterfactual metrics are absent"})

    ablation = _read(ABLATION_DIR / "ablation_per_patient.csv")
    if ablation.empty:
        missing.append({"Input": str(ABLATION_DIR / "ablation_per_patient.csv"), "Reason": "finalized Phase 13 ablation results are absent"})
    else:
        for metric in [name for name in ("Validity", "Sparsity", "Proximity", "Plausibility", "Actionability", "ConstraintValidity", "DependencyConsistency", "Diversity", "Runtime", "FeasibilityRate") if name in ablation.columns]:
            for variant in ablation["Variant"].dropna().unique():
                values = pd.to_numeric(ablation.loc[ablation["Variant"] == variant, metric], errors="coerce").to_numpy()
                estimate, lower, upper = _bootstrap_ci(values, bootstrap_iterations, seed + len(ci_rows))
                ci_rows.append({"Method": variant, "Metric": metric, "SampleSize": int(np.isfinite(values).sum()), "Estimate": estimate, "CI_Lower": lower, "CI_Upper": upper, "BootstrapIterations": bootstrap_iterations, "Seed": seed, "MissingExcluded": int((~np.isfinite(values)).sum())})

    raw_pvalues = [row["PValue"] for row in comparison_rows]
    adjusted = _holm(raw_pvalues)
    for row, value in zip(comparison_rows, adjusted):
        row["PValueHolmAdjusted"] = value
        row["MultipleComparisonPolicy"] = "Holm step-down correction across executed paired tests"
    pd.DataFrame(ci_rows).to_csv(output / "confidence_intervals.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "paired_tests.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(output / "effect_sizes.csv", index=False)

    lines = ["# Phase 14 Statistical Analysis", "", "Read-only analysis of finalized result files. No experiments, model training, or tuning were run.", "", f"Bootstrap iterations: {bootstrap_iterations}", f"Bootstrap seed: {seed}", "Confidence level: 95%", "Multiple comparisons: Holm step-down correction across executed paired tests.", ""]
    if missing:
        lines.append("## Missing or Excluded Inputs")
        for item in missing:
            lines.append(f"- `{item['Input']}`: {item['Reason']}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("Numerical differences are descriptive only. No statistical superiority is claimed.")
    if not comparison_rows:
        lines.append("No paired DiCE vs CC-MO-CF tests were executed because finalized paired results were unavailable or insufficient.")
    else:
        lines.append(f"Paired tests executed or assessed: {len(comparison_rows)}. Tests with insufficient sample size or missing dependencies are explicitly marked.")
    (output / "statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_dir": output, "confidence_intervals": len(ci_rows), "paired_tests": len(comparison_rows), "missing_inputs": missing}


if __name__ == "__main__":
    print(json.dumps(run_phase14(), indent=2, default=str))