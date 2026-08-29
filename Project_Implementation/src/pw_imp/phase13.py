"""Phase 13 ablation study for CC-MO-CF."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pw_imp.ccmocf import CCMOCF, Candidate
from pw_imp.clinical_constraints import ClinicalConstraintEngine
from pw_imp.dice_baseline import load_dice_baseline_model
from pw_imp.preprocessing import build_preprocessing_pipeline, load_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_OUT = REPO_ROOT / "results" / "ablation"

VARIANTS = {
    "full_ccmocf": {},
    "without_sparsity": {"use_sparsity_objective": False},
    "without_proximity": {"use_proximity_objective": False},
    "without_actionability": {"use_actionability_objective": False},
    "without_constraint_projection": {"use_constraint_projection": False},
    "without_dependency_consistency": {"use_dependency_consistency": False},
    "without_diversity_selection": {"use_diversity_selection": False},
}


def _distance(original: dict, candidate: Candidate) -> float:
    values = []
    for name, value in candidate.row.items():
        try:
            original_value = float(original[name])
            candidate_value = float(value)
            values.append(abs(candidate_value - original_value))
        except (KeyError, TypeError, ValueError):
            values.append(float(original.get(name) != value))
    return float(np.mean(values)) if values else 0.0


def _candidate_distance(left: Candidate, right: Candidate) -> float:
    values = []
    for name, value in left.row.items():
        try:
            values.append(abs(float(value) - float(right.row[name])))
        except (KeyError, TypeError, ValueError):
            values.append(float(value != right.row.get(name)))
    return float(np.mean(values)) if values else 0.0


def _candidate_record(variant: str, patient_id: int, original: dict, candidate: Candidate, target: int, runtime: float) -> dict:
    engine = ClinicalConstraintEngine()
    constraint_valid = True
    dependency_valid = True
    try:
        engine.validate_candidate(original, candidate.row)
    except ValueError:
        constraint_valid = False
    for feature in ("BMI", "FSH/LH", "Waist:Hip Ratio"):
        if feature in candidate.row:
            try:
                engine.validate_candidate(original, candidate.row)
            except ValueError:
                dependency_valid = False
    changed = candidate.changed_features
    actionability = float(sum(bool(engine.get_feature_spec(name).get("actionable", False)) for name in changed) / len(changed)) if changed else 1.0
    return {"Variant": variant, "PatientID": patient_id, "CounterfactualID": None, "TargetClass": target, "Validity": float(int(candidate.probability >= 0.5) == target), "Sparsity": len(changed), "Proximity": _distance(original, candidate), "Plausibility": float(constraint_valid and dependency_valid), "Actionability": actionability, "ConstraintValidity": float(constraint_valid), "DependencyConsistency": float(dependency_valid), "Diversity": np.nan, "FeasibilityRate": float(constraint_valid and dependency_valid), "Runtime": runtime, "FeasibilityStatus": candidate.feasibility_status if constraint_valid and dependency_valid else "INVALID_ABLATION_OUTPUT", "ChangedFeatures": " | ".join(changed)}


def run_phase13(model_name: str = "xgboost", seed: int = 42, population_size: int = 64, generations: int = 20, top_k: int = 3, output_dir: str | Path = ABLATION_OUT) -> dict:
    """Run fixed-parameter ablations on the same held-out test patients."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = build_preprocessing_pipeline(random_state=seed)
    model = load_dice_baseline_model(model_name)
    raw = load_dataframe()
    patient_test = pipeline["patient_test"].reset_index(drop=True)
    X_test = pipeline["X_test"]
    probabilities = np.asarray(model.predict_proba(X_test))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    test_ids = patient_test["PatientID"].astype(int).tolist()
    raw_test = raw.set_index("PatientID").loc[test_ids].reset_index()
    per_cf = []
    per_patient = []
    for variant, switches in VARIANTS.items():
        for index, raw_row in raw_test.iterrows():
            started = time.perf_counter()
            patient_id = int(raw_row["PatientID"])
            original = raw_row.drop(labels=["PatientID", "PCOS (Y/N)"], errors="ignore").to_dict()
            target = 0 if int(predictions[index]) == 1 else 1
            engine = CCMOCF(model, pipeline["preprocessor"], population_size=population_size, generations=generations, seed=seed, top_k=top_k, **switches)
            result = engine.generate(original, desired_class=target)
            runtime = time.perf_counter() - started
            candidates = result["counterfactuals"]
            records = [_candidate_record(variant, patient_id, original, candidate, target, runtime) for candidate in candidates]
            diversity = float(np.mean([_candidate_distance(left, right) for index_left, left in enumerate(candidates) for right in candidates[index_left + 1:]])) if len(candidates) > 1 else 0.0
            for counterfactual_id, record in enumerate(records, start=1):
                record["CounterfactualID"] = counterfactual_id
                record["Diversity"] = diversity
            per_cf.extend(records)
            per_patient.append({"Variant": variant, "PatientID": patient_id, "Status": result["status"], "NumberOfCandidatesEvaluated": result["candidates_evaluated"], "NumberOfCFs": len(records), "Validity": float(np.mean([r["Validity"] for r in records])) if records else 0.0, "Sparsity": float(np.mean([r["Sparsity"] for r in records])) if records else np.nan, "Proximity": float(np.mean([r["Proximity"] for r in records])) if records else np.nan, "Plausibility": float(np.mean([r["Plausibility"] for r in records])) if records else 0.0, "Actionability": float(np.mean([r["Actionability"] for r in records])) if records else np.nan, "ConstraintValidity": float(np.mean([r["ConstraintValidity"] for r in records])) if records else 0.0, "DependencyConsistency": float(np.mean([r["DependencyConsistency"] for r in records])) if records else 0.0, "Diversity": diversity, "FeasibilityRate": float(bool(records) and all(r["FeasibilityStatus"] == "VALID_FEASIBLE" for r in records)), "Runtime": runtime, "ConstraintsRemovedExperimental": variant == "without_constraint_projection"})
    per_cf_frame = pd.DataFrame(per_cf)
    per_patient_frame = pd.DataFrame(per_patient)
    per_cf_frame.to_csv(output / "ablation_results.csv", index=False)
    per_patient_frame.to_csv(output / "ablation_per_patient.csv", index=False)
    config = {"model_name": model_name, "seed": seed, "population_size": population_size, "generations": generations, "top_k": top_k, "eligible_patient_ids": test_ids, "fixed_predictive_model": True, "variants": VARIANTS, "constraints_removed_is_experimental_ablation": True}
    with open(output / "ablation_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    try:
        import matplotlib.pyplot as plt
        means = per_patient_frame.groupby("Variant")[["Validity", "Sparsity", "Proximity", "Plausibility", "Actionability", "ConstraintValidity", "DependencyConsistency", "FeasibilityRate", "Runtime"]].mean(numeric_only=True)
        axis = means.plot(kind="bar", figsize=(15, 7), title="CC-MO-CF Ablation Comparison")
        axis.set_ylabel("Mean value")
        axis.tick_params(axis="x", labelrotation=45)
        axis.figure.tight_layout()
        axis.figure.savefig(output / "ablation_metric_figure.png", dpi=300, bbox_inches="tight")
        axis.figure.savefig(output / "ablation_metric_figure.pdf", bbox_inches="tight")
        plt.close(axis.figure)
    except ImportError:
        pass
    return {"output_dir": output, "patients": len(test_ids), "variants": list(VARIANTS)}


if __name__ == "__main__":
    print(json.dumps(run_phase13(), default=str, indent=2))