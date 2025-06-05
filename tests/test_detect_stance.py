from stance_llm.base import StanceClassification, ALLOWED_STANCE_CATEGORIES

# RUN TESTS ---------------


def test_detect_stance_returns_correct_object_trf(stance_detection_runs_trf):
    """Test if stance detection runs with Transformers backend return an object of type StanceClassification"""
    for run in stance_detection_runs_trf:
        run.language = "de"
    assert all(
        isinstance(run, StanceClassification) for run in stance_detection_runs_trf
    )


def test_detect_stance_returns_stance(
    stance_detection_runs_trf
):
    """Test if stance detection runs return a stance as a string and that the string is in an allowed category"""
    assert all(isinstance(run.stance, str) for run in stance_detection_runs_trf)
    assert all(
        run.stance in ALLOWED_STANCE_CATEGORIES
        for run in stance_detection_runs_trf
    )
