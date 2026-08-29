import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pw_imp.dice_baseline import DICE_OUT, generate_dice_counterfactuals


def test_dice_generation_for_small_sample():
    result = generate_dice_counterfactuals(model_name="xgboost", max_patients=2, num_counterfactuals=2)
    assert result["failures"] == []
    assert len(result["results"]) == 2
    summary_path = DICE_OUT / "dice_summary.csv"
    assert summary_path.exists()
    assert (DICE_OUT / "dice_patient_27.csv").exists()
