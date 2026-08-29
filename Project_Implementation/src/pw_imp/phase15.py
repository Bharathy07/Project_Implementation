"""Phase 15 paper-ready tables and figures from finalized saved artifacts only."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
PAPER_TABLES = RESULTS / "paper_tables"
PAPER_FIGURES = REPO_ROOT / "paper_figures"


def _read_csv(path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _na_frame(columns):
    import pandas as pd

    return pd.DataFrame([{column: "N/A" for column in columns}])


def _save_table(frame, name):
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_TABLES / f"{name}.csv", index=False)
    try:
        (PAPER_TABLES / f"{name}.tex").write_text(frame.to_latex(index=False, na_rep="N/A"), encoding="utf-8")
    except Exception:
        pass


def _copy_figure(source, target_stem):
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in ("png", "pdf"):
        source_path = source.with_suffix(f".{suffix}")
        if source_path.exists():
            target = PAPER_FIGURES / f"{target_stem}.{suffix}"
            shutil.copy2(source_path, target)
            copied.append(str(target))
    return copied


def _save_plot(name, title, frame=None, x=None, y=None, hue=None, text=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    if text or frame is None or frame.empty:
        axis.text(0.5, 0.5, text or "N/A: finalized result unavailable", ha="center", va="center", wrap=True)
        axis.set_axis_off()
    elif x and y and hue and hue in frame:
        for label, group in frame.groupby(hue):
            axis.plot(group[x], group[y], marker="o", label=str(label))
        axis.legend()
        axis.set_xlabel(x)
        axis.set_ylabel(y)
    elif x and y:
        axis.bar(frame[x].astype(str), frame[y])
        axis.tick_params(axis="x", labelrotation=45)
        axis.set_ylabel(y)
    axis.set_title(title)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = PAPER_FIGURES / f"{name}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def _table_inputs():
    import pandas as pd

    raw_path = REPO_ROOT / "PCOS_data_without_infertility.xlsx"
    if raw_path.exists():
        try:
            raw = pd.read_excel(raw_path, sheet_name="Full_new", engine="openpyxl")
            labels = raw["PCOS (Y/N)"].value_counts(dropna=False).rename_axis("Class").reset_index(name="Count") if "PCOS (Y/N)" in raw else _na_frame(["Class", "Count"])
            dataset = pd.DataFrame({"Statistic": ["Rows", "Columns", "Unique patients"], "Value": [len(raw), len(raw.columns), raw.iloc[:, 0].nunique()]})
        except Exception:
            labels = _na_frame(["Class", "Count"])
            dataset = _na_frame(["Statistic", "Value"])
    else:
        labels = _na_frame(["Class", "Count"])
        dataset = _na_frame(["Statistic", "Value"])
    selected = _read_csv(RESULTS / "features" / "selected_features.csv")
    if selected.empty:
        selected = _na_frame(["feature"])
    parameters_path = RESULTS / "models" / "model_parameters.json"
    if parameters_path.exists():
        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        hyperparameters = pd.DataFrame([{"Model": model, "Hyperparameters": json.dumps(values, sort_keys=True)} for model, values in parameters.items()])
    else:
        hyperparameters = _na_frame(["Model", "Hyperparameters"])
    model_summary = _read_csv(RESULTS / "final_experiment" / "model_results.csv")
    if model_summary.empty:
        model_summary = _read_csv(RESULTS / "models" / "results_summary.csv")
    metrics = model_summary if not model_summary.empty else _na_frame(["Model", "Metric", "Value", "N"])
    definitions = _read_csv(RESULTS / "evaluation" / "cf_metrics_per_counterfactual.csv")
    metric_definitions = _na_frame(["Metric", "Definition", "N"]) if definitions.empty else pd.DataFrame({"Metric": definitions.columns, "Definition": ["Defined in Phase 10 evaluation module"] * len(definitions.columns), "N": [len(definitions)] * len(definitions.columns)})
    comparison = _read_csv(RESULTS / "evaluation" / "ccmocf_vs_dice_summary.csv")
    comparison = comparison if not comparison.empty else _na_frame(["Metric", "DiCE", "CC-MO-CF", "Difference", "N"])
    ablation = _read_csv(RESULTS / "ablation" / "ablation_per_patient.csv")
    ablation = ablation if not ablation.empty else _na_frame(["Variant", "Metric", "Mean", "SD", "N"])
    stats = _read_csv(RESULTS / "statistics" / "paired_tests.csv")
    stats = stats if not stats.empty else _na_frame(["Metric", "SampleSize", "TestUsed", "Difference", "PValue", "PValueHolmAdjusted"])
    patient = _read_csv(RESULTS / "patients" / "patient_master_results.csv")
    cases = patient.head(5) if not patient.empty else _na_frame(["PatientID", "TrueLabel", "PredictedLabel", "CounterfactualStatus"])
    return dataset, labels, selected, hyperparameters, metrics, metric_definitions, comparison, ablation, stats, cases


def run_phase15():
    """Generate paper artifacts from saved results; never invokes an experiment runner."""
    dataset, labels, selected, hyperparameters, metrics, definitions, comparison, ablation, stats, cases = _table_inputs()
    for name, frame in {
        "table_I_dataset_characteristics": dataset,
        "table_I_class_distribution": labels,
        "table_II_selected_clinical_features": selected,
        "table_III_model_hyperparameters": hyperparameters,
        "table_IV_predictive_model_performance": metrics,
        "table_V_counterfactual_metric_definitions": definitions,
        "table_VI_dice_vs_ccmocf": comparison,
        "table_VII_ablation": ablation,
        "table_VIII_statistical_analysis": stats,
        "table_IX_patient_case_studies": cases,
    }.items():
        _save_table(frame, name)

    figure_sources = {
        "figure_7_shap_global": RESULTS / "figures" / "shap" / "global_feature_importance.png",
        "figure_8_shap_beeswarm": RESULTS / "figures" / "shap" / "shap_beeswarm.png",
        "figure_10_cf_metric_comparison": RESULTS / "evaluation" / "metric_comparison.png",
        "figure_11_cf_feasibility": RESULTS / "evaluation" / "feasibility_failure_distribution.png",
        "figure_14_ablation": RESULTS / "ablation" / "ablation_metric_figure.png",
    }
    copied = []
    for name, source in figure_sources.items():
        copied.extend(_copy_figure(source, name))
    copied.extend(_save_plot("figure_2_dataset_class_distribution", "Dataset and class distribution", labels, "Class", "Count"))
    copied.extend(_save_plot("figure_3_predictive_model_comparison", "Predictive model comparison", metrics if "Model" in metrics and "Value" in metrics else None, "Model", "Value", text=None if "Model" in metrics and "Value" in metrics else "N/A: finalized model performance unavailable"))
    copied.extend(_save_plot("figure_1_framework_architecture", "Overall framework architecture", text="Test patient -> Final model -> SHAP -> DiCE -> CC-MO-CF -> Validation -> Metrics"))
    for name, title in (("figure_4_roc_curve", "ROC curve"), ("figure_5_precision_recall_curve", "Precision-recall curve"), ("figure_6_calibration", "Calibration analysis"), ("figure_9_local_shap", "Local SHAP explanation"), ("figure_12_representative_ccmocf", "Representative CC-MO-CF patient"), ("figure_13_pareto_front", "Pareto front / multi-objective trade-off")):
        copied.extend(_save_plot(name, title, text="N/A: finalized figure data unavailable"))
    manifest = {"tables_dir": str(PAPER_TABLES), "figures_dir": str(PAPER_FIGURES), "tables_actual_or_na": True, "figures_generated_or_na": True, "experiments_rerun": False, "figure_count_requested": 14, "figure_files_created": len(copied)}
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / "paper_output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_phase15(), indent=2))