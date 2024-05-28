from stance_llm.base import StanceClassification

# from dotenv import load_dotenv
# load_dotenv(".env")

# RUN TESTS ---------------

def test_detect_stance_returns_correct_object_openai(stance_detection_run_openai,
                                              stance_detection_run_trf):
    """Test if stance detection runs with OpenAI backend return an object of type StanceClassification
    """
    assert isinstance(stance_detection_run_openai,StanceClassification)

def test_detect_stance_returns_correct_object_trf(stance_detection_run_openai,
                                              stance_detection_run_trf):
    """Test if stance detection runs with Transformers backend return an object of type StanceClassification
    """
    assert isinstance(stance_detection_run_trf,StanceClassification)

def test_detect_stance_returns_stance(stance_detection_run_openai,
                                      stance_detection_run_trf):
    """Test if stance detection runs return a stance as a string and that the string is in an allowed category
    """
    assert isinstance(stance_detection_run_openai.stance,str)
    assert stance_detection_run_openai.stance in ["support","opposition","irrelevant"]
    assert isinstance(stance_detection_run_trf.stance,str)
    assert stance_detection_run_trf.stance in ["support","opposition","irrelevant"]

def test_detect_stance_returns_correct_entity(stance_detection_run_openai,
                                              test_examples):
    """Test if stance detection runs return the correct entity string
    """
    assert stance_detection_run_openai.entity == test_examples[0]["org_text"]

def test_detect_stance_masked_returns_correct_entity(stance_detection_run_masked_openai,
                                                     test_examples):
    """Tet if masked stance detection runs return the correct entity string
    """
    assert stance_detection_run_masked_openai.entity == test_examples[0]["org_text"]