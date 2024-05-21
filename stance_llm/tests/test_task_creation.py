from guidance import models
import os
from stance_llm.process import detect_stance
from stance_llm.base import StanceClassification

from dotenv import load_dotenv
load_dotenv(".env")

# OPENAI api key is loaded from a .env file in tests/ via pytest-dotenv
gpt35 = models.OpenAI("gpt-3.5-turbo",api_key=os.environ["OPEN_AI_KEY"])

test_entity_str = "Stadt Bern"
test_examples = [
    {"text":"Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
     "org_text":test_entity_str,
     "statement":"Das Fahrrad als Mobilitätsform soll gefördert werden."}]

stance_detection_run = detect_stance(
            eg = test_examples[0],
            lm = gpt35,
            chain_label = "s2is"
        )

stance_detection_run_masked = detect_stance(
            eg = test_examples[0],
            lm = gpt35,
            chain_label = "s2is",
            entity_mask="Organisation X"
        )

def test_detect_stance_returns_correct_object():
    """Test if a stance detection run returns an object of type StanceClassification
    """
    assert isinstance(stance_detection_run,StanceClassification)

def test_detect_stance_returns_stance():
    """Test if a stance detection run retruns a stance as a string and that the string is in an allowed category
    """
    assert isinstance(stance_detection_run.stance,str)
    assert stance_detection_run.stance in ["support","opposition","irrelevant"]

def test_detect_stance_returns_correct_entity():
    assert stance_detection_run.entity == test_entity_str

def test_detect_stance_masked_returns_correct_entity():
    assert stance_detection_run_masked.entity == test_entity_str