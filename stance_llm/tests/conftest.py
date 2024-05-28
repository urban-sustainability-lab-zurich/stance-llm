import pytest

from guidance import models
import os
from stance_llm.process import detect_stance

# PREPARE EXAMPLE DATA ----------------

@pytest.fixture(scope="module")
def test_examples():
    test_egs = [
    {"text":"Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
     "org_text":"Stadt Bern",
     "statement":"Das Fahrrad als Mobilitätsform soll gefördert werden."},
     {"text":"Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
     "org_text":"FDP",
     "statement":"Das Fahrrad als Mobilitätsform soll gefördert werden."}]
    return(test_egs)

# LOAD MODELS ------------

@pytest.fixture(scope="module")
def gpt35_openai():
    # OPENAI api key is loaded from a .env file in tests/ via pytest-dotenv
    gpt35 = models.OpenAI("gpt-3.5-turbo",api_key=os.environ["OPEN_AI_KEY"])
    return(gpt35)

@pytest.fixture(scope="module")
def gpt2_trf():
    # gpt2 model from huggingface as example of Transformer model
    gpt2_trf = models.Transformers('gpt2')
    return(gpt2_trf)

# RUN STANCE DETECTIONS --------------

# non-masked

@pytest.fixture(scope="module")
def stance_detection_run_openai(
    test_examples,
    gpt35_openai):
    run = detect_stance(
            eg = test_examples[0],
            llm = gpt35_openai,
            chain_label = "s2is"
        )
    return(run)

@pytest.fixture(scope="module")
def stance_detection_run_trf(
    test_examples,
    gpt2_trf):
    run = detect_stance(
            eg = test_examples[0],
            llm = gpt2_trf,
            chain_label = "s2"
        )
    return(run)

# masked

@pytest.fixture(scope="module")
def stance_detection_run_masked_openai(
    test_examples,
    gpt35_openai):
    run = detect_stance(
            eg = test_examples[0],
            llm = gpt35_openai,
            chain_label = "s2is",
            entity_mask="Organisation X"
        )
    return(run)

@pytest.fixture(scope="module")
def stance_detection_run_masked_trf(
    test_examples,
    gpt2_trf):
    run = detect_stance(
            eg = test_examples[0],
            llm = gpt2_trf,
            chain_label = "s2",
            entity_mask="Organisation X"
        )
    return(run)