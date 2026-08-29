import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pw_imp.ccmocf import CCMOCF, _CCMOCFProblem, save_results
from pw_imp.clinical_constraints import ClinicalConstraintEngine
from pw_imp.ccmocf import CCMOCF_OUT


CONFIG = Path(__file__).resolve().parents[1] / "config" / "clinical_constraints.yaml"


class DummyPreprocessor:
    selected_features = ["Weight (Kg)", "PatientID"]

    def transform(self, frame):
        return frame[self.selected_features].astype(float)


class WeightModel:
    def predict_proba(self, frame):
        probability = (frame["Weight (Kg)"].to_numpy() < 65).astype(float)
        return pd.DataFrame({0: 1 - probability, 1: probability}).to_numpy()


def _engine():
    return ClinicalConstraintEngine(CONFIG)


def test_ccmocf_preserves_immutable_and_patient_id():
    engine = _engine()
    current = {"PatientID": 27, "Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 70 / 1.7**2}
    projected = engine.project_candidate(current, {**current, "PatientID": 999, "Weight (Kg)": 75.0})
    assert projected["PatientID"] == 27
    assert projected["Weight (Kg)"] == 75.0
    assert abs(projected["BMI"] - 75 / 1.7**2) < 1e-6
    engine.validate_candidate(current, projected)


def test_ccmocf_bounds_and_max_step_are_enforced():
    engine = _engine()
    current = {"Weight (Kg)": 70.0}
    projected = engine.project_candidate(current, {"Weight (Kg)": 250.0})
    assert projected["Weight (Kg)"] == 80.0
    assert engine.validate_candidate(current, projected)


def test_ccmocf_finds_valid_flip_and_reports_nsga_path():
    engine = CCMOCF(WeightModel(), DummyPreprocessor(), _engine(), population_size=8, generations=3, seed=3, top_k=2, optimizer_available=True)
    result = engine.generate({"PatientID": 27, "Weight (Kg)": 70.0}, desired_class=1)
    assert result["optimizer_used"] == "NSGA-II"
    assert result["status"] == "SUCCESS"
    assert all(cf.row["PatientID"] == 27 for cf in result["counterfactuals"])
    assert all(cf.row["Weight (Kg)"] < 65.0 for cf in result["counterfactuals"])


def test_actual_pymoo_nsga2_problem_and_objectives():
    engine = CCMOCF(WeightModel(), DummyPreprocessor(), _engine(), population_size=10, generations=2, seed=42, top_k=3)
    problem = _CCMOCFProblem(engine, {"PatientID": 27, "Weight (Kg)": 70.0}, desired_class=1)
    assert problem.n_obj == 6
    algorithm = NSGA2(pop_size=10)
    result = minimize(problem, algorithm, termination=("n_gen", 2), seed=42, verbose=False)
    assert result.X is not None
    assert result.F is not None
    generated = engine.generate({"PatientID": 27, "Weight (Kg)": 70.0}, desired_class=1)
    assert generated["optimizer_used"] == "NSGA-II"
    assert generated["pareto_solution_count"] >= 1
    assert all(candidate.row["PatientID"] == 27 for candidate in generated["counterfactuals"])
    smoke_rows = [{
        "test_id": "synthetic_patient_27",
        "optimizer_used": generated["optimizer_used"],
        "population_size": 10,
        "generations": 2,
        "number_of_objectives": 6,
        "pareto_solution_count": generated["pareto_solution_count"],
        "feasible_solution_count": len(generated["counterfactuals"]),
        "original_prediction": generated["original_prediction"],
        "original_probability": generated["original_probability"],
        "candidate_prediction": int(candidate.probability >= 0.5),
        "candidate_probability": candidate.probability,
        "changed_features": " | ".join(candidate.changed_features),
        "objective_values": str(candidate.objectives),
        "constraint_valid": candidate.constraint_status == "VALID",
        "runtime_seconds": generated["runtime"],
        "error": "",
        "validation_type": "SOFTWARE RUNTIME VALIDATION ONLY - NOT RESEARCH RESULTS",
    } for candidate in generated["counterfactuals"]]
    CCMOCF_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(smoke_rows).to_csv(CCMOCF_OUT / "ccmocf_nsga2_smoke_test.csv", index=False)


def test_ccmocf_fallback_and_failure_are_explicit(tmp_path):
    engine = CCMOCF(WeightModel(), DummyPreprocessor(), _engine(), population_size=8, generations=2, seed=3, optimizer_available=False, use_fallback=True)
    result = engine.generate({"PatientID": 27, "Weight (Kg)": 70.0}, desired_class=0)
    assert result["optimizer_used"] == "fallback"
    assert result["status"] in {"NO_PREDICTION_FLIP", "NO_FEASIBLE_COUNTERFACTUAL"}
    paths = save_results([{"patient_id": 27, **result}], tmp_path)
    assert all(path.exists() for path in paths.values())
    assert pd.read_csv(paths["all_attempts.csv"]).iloc[0]["status"] in {"NO_PREDICTION_FLIP", "NO_FEASIBLE_COUNTERFACTUAL"}


def test_fallback_is_opt_in():
    engine = CCMOCF(WeightModel(), DummyPreprocessor(), _engine(), optimizer_available=False)
    result = engine.generate({"PatientID": 27, "Weight (Kg)": 70.0}, desired_class=1)
    assert result["optimizer_used"] == "unavailable"


def test_ccmocf_rejects_non_actionable_change():
    engine = _engine()
    current = {"FSH(mIU/mL)": 5.4}
    try:
        engine.validate_candidate(current, {"FSH(mIU/mL)": 6.0})
    except ValueError:
        return
    raise AssertionError("non-actionable feature change was accepted")