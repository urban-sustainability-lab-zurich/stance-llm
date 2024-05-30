import pytest

from guidance import models
import os
from stance_llm.process import detect_stance
from stance_llm.base import REGISTERED_LLM_CHAINS, CONSTRAINED_GRAMMAR_CHAINS

# PREPARE EXAMPLE DATA ----------------


@pytest.fixture(scope="module")
def test_examples():
    test_egs = [
        {
            "text": "Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
            "ent_text": "Stadt Bern",
            "statement": "Das Fahrrad als Mobilitätsform soll gefördert werden.",
            "stance_true": "support",
        },
        {
            "text": "Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
            "ent_text": "FDP",
            "statement": "Das Fahrrad als Mobilitätsform soll gefördert werden.",
            "stance_true": "opposition",
        },
    ]
    return test_egs


# LOAD MODELS ------------


@pytest.fixture(scope="module")
def gpt35_openai():
    # OPENAI api key is loaded from a .env file in tests/ via pytest-dotenv
    gpt35 = models.OpenAI("gpt-3.5-turbo", api_key=os.environ["OPEN_AI_KEY"])
    return gpt35


@pytest.fixture(scope="module")
def gpt2_trf():
    # gpt2 model from huggingface as example of Transformer model
    gpt2_trf = models.Transformers("gpt2")
    return gpt2_trf


# RUN STANCE DETECTIONS FOR ALL CHAINS --------------


@pytest.fixture(scope="module")
def allowed_openai_chains():
    allowed = [
        chain
        for chain in REGISTERED_LLM_CHAINS
        if chain not in CONSTRAINED_GRAMMAR_CHAINS
    ]
    return allowed


# non-masked, iterate across all chains


@pytest.fixture(scope="module")
def stance_detection_runs_openai(test_examples, gpt35_openai, allowed_openai_chains):
    classifications = []
    for chain in allowed_openai_chains:
        classification = detect_stance(
            eg=test_examples[0], llm=gpt35_openai, chain_label=chain
        )
        classifications.append(classification)
    return classifications


@pytest.fixture(scope="module")
def stance_detection_runs_trf(test_examples, gpt2_trf):
    classifications = []
    for chain in REGISTERED_LLM_CHAINS:
        classification = detect_stance(
            eg=test_examples[0], llm=gpt2_trf, chain_label=chain
        )
        classifications.append(classification)
    return classifications


# masked, single chain as example


@pytest.fixture(scope="module")
def stance_detection_run_masked_openai(test_examples, gpt35_openai):
    run = detect_stance(
        eg=test_examples[0],
        llm=gpt35_openai,
        chain_label="s2is",
        entity_mask="Organisation X",
    )
    return run


@pytest.fixture(scope="module")
def stance_detection_run_masked_trf(test_examples, gpt2_trf):
    run = detect_stance(
        eg=test_examples[0],
        llm=gpt2_trf,
        chain_label="s2",
        entity_mask="Organisation X",
    )
    return run


# PROCESSING TESTS


@pytest.fixture(scope="module")
def test_output_dir():
    out_dir = os.path.join(os.getcwd(), "tests/test_output")
    return out_dir
