"""Native and generic OOLONG paths share the existing auxiliary score."""

import importlib.util
from pathlib import Path


def test_existing_oolong_score_cases():
    path = Path(__file__).parents[1] / "training/environments/oolong/oolong/scoring.py"
    spec = importlib.util.spec_from_file_location("oolong_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.synth_score({"answer": "[3]", "answer_type": "ANSWER_TYPE.NUMERIC"}, "Answer: 3")
        == 1.0
    )
    assert (
        module.synth_score({"answer": "[3]", "answer_type": "ANSWER_TYPE.NUMERIC"}, "Answer: 4")
        == 0.75
    )
    assert (
        module.synth_score(
            {"answer": "['less common than']"}, "location is less common than entity"
        )
        == 1.0
    )
    assert module.synth_score({"answer": "['less common than']"}, "more common than") == 0.0
