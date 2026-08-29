import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pw_imp.clinical_constraints import ClinicalConstraintEngine


ENGINE = ClinicalConstraintEngine(Path(__file__).resolve().parents[1] / "config" / "clinical_constraints.yaml")


def test_immutable_violation():
    with pytest.raises(ValueError):
        ENGINE.validate_feature_change("PatientID", 101, 102)


def test_out_of_range():
    with pytest.raises(ValueError):
        ENGINE.validate_feature_change("Weight (Kg)", 70.0, 250.0)


def test_max_step_violation():
    with pytest.raises(ValueError):
        ENGINE.validate_feature_change("Weight (Kg)", 70.0, 90.0)


def test_non_actionable_modification():
    with pytest.raises(ValueError):
        ENGINE.validate_feature_change("FSH(mIU/mL)", 5.4, 6.0)


def test_valid_candidate():
    assert ENGINE.validate_feature_change("Weight (Kg)", 70.0, 75.0) is True
    assert ENGINE.validate_feature_change("Cycle(R/I)_2", 0.0, 1.0) is True


def test_projection_preserves_immutable_fields():
    current = {"PatientID": 101, "Weight (Kg)": 70.0, "FSH(mIU/mL)": 5.4}
    candidate = {"PatientID": 999, "Weight (Kg)": 75.0, "FSH(mIU/mL)": 6.5}
    projected = ENGINE.project_candidate(current, candidate)
    assert projected["PatientID"] == 101
    assert projected["Weight (Kg)"] == 75.0
    assert projected["FSH(mIU/mL)"] == 5.4


def test_valid_dependency():
    current = {"Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 24.22}
    candidate = {"Weight (Kg)": 72.0, "Height(Cm)": 170.0}
    projected = ENGINE.project_candidate(current, candidate)
    assert abs(projected["BMI"] - 24.91) < 0.2


def test_invalid_dependency():
    current = {"Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 24.22}
    candidate = {"Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 40.0}
    with pytest.raises(ValueError):
        ENGINE.validate_candidate(current, candidate)


def test_projection_repair_dependency():
    current = {"Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 24.22}
    candidate = {"Weight (Kg)": 80.0, "Height(Cm)": 170.0, "BMI": 50.0}
    projected = ENGINE.project_candidate(current, candidate)
    expected = 80.0 / (1.7 ** 2)
    assert abs(projected["BMI"] - expected) < 1e-6


def test_rejection_of_inconsistent_dependency():
    current = {"Weight (Kg)": 70.0, "Height(Cm)": 170.0, "BMI": 24.22}
    candidate = {"Weight (Kg)": 80.0, "Height(Cm)": 170.0, "BMI": 20.0}
    with pytest.raises(ValueError):
        ENGINE.validate_candidate(current, candidate)
