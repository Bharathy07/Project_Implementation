import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
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


def test_projection_rejects_non_actionable_modification():
    current = {"Cycle length(days)": 30.0}
    with pytest.raises(ValueError, match="non-actionable"):
        ENGINE.project_candidate(current, {"Cycle length(days)": 35.0})


def test_unchanged_missing_value_is_not_treated_as_an_edit():
    current = {"Weight (Kg)": 70.0, "Waist:Hip Ratio": np.nan}
    candidate = {"Weight (Kg)": 75.0, "Waist:Hip Ratio": np.nan}
    projected = ENGINE.project_candidate(current, candidate)
    assert projected["Weight (Kg)"] == 75.0


def test_missing_constraint_definition_is_non_actionable():
    with pytest.raises(ValueError, match="Constraint definition missing"):
        ENGINE.validate_feature_change("Unsupported feature", 0.0, 1.0)


def test_valid_candidate():
    assert ENGINE.validate_feature_change("Weight (Kg)", 70.0, 75.0) is True
    assert ENGINE.validate_feature_change("Cycle(R/I)_2", 0.0, 1.0) is True


def test_projection_rejects_immutable_fields():
    current = {"PatientID": 101, "Weight (Kg)": 70.0, "FSH(mIU/mL)": 5.4}
    candidate = {"PatientID": 999, "Weight (Kg)": 75.0, "FSH(mIU/mL)": 6.5}
    with pytest.raises(ValueError, match="PatientID.*immutable"):
        ENGINE.project_candidate(current, candidate)


def test_projection_rejects_age_change():
    current = {"Age (yrs)": 20.0, "Weight (Kg)": 70.0}
    candidate = {"Age (yrs)": 21.0, "Weight (Kg)": 75.0}
    with pytest.raises(ValueError, match=r"Age \(yrs\).*immutable"):
        ENGINE.project_candidate(current, candidate)


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
