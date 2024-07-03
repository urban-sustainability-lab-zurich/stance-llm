from stance_llm.base import StanceClassification

# from dotenv import load_dotenv
# load_dotenv(".env")

# RUN TESTS ---------------


def test_detect_stance_returns_correct_object_openai(stance_detection_runs_openai):
    """Test if stance detection runs with OpenAI backend return an object of type StanceClassification"""
    assert all(
        isinstance(run, StanceClassification) for run in stance_detection_runs_openai
    )


def test_detect_stance_returns_correct_object_trf(stance_detection_runs_trf):
    """Test if stance detection runs with Transformers backend return an object of type StanceClassification"""
    assert all(
        isinstance(run, StanceClassification) for run in stance_detection_runs_trf
    )


def test_detect_stance_returns_stance(
    stance_detection_runs_openai, stance_detection_runs_trf
):
    """Test if stance detection runs return a stance as a string and that the string is in an allowed category"""
    assert all(isinstance(run.stance, str) for run in stance_detection_runs_openai)
    assert all(
        run.stance in ["support", "opposition", "irrelevant"]
        for run in stance_detection_runs_openai
    )
    assert all(isinstance(run.stance, str) for run in stance_detection_runs_trf)
    assert all(
        run.stance in ["support", "opposition", "irrelevant"]
        for run in stance_detection_runs_trf
    )


def test_detect_stance_returns_correct_entity(
    stance_detection_runs_openai, test_examples
):
    """Test if stance detection runs return the correct entity string"""
    assert ([run.entity for run in stance_detection_runs_openai] == 
            int(len(stance_detection_runs_openai)/
            len(test_examples))*[eg["ent_text"] for eg in test_examples])


def test_detect_stance_masked_returns_correct_entity(
    stance_detection_run_masked_openai, test_examples
):
    """Test if masked stance detection runs return the correct entity string"""
    assert stance_detection_run_masked_openai.entity == test_examples[0]["ent_text"]
