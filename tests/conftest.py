import pytest

from guidance import models
import os
from stance_llm.process import detect_stance
from stance_llm.base import REGISTERED_LLM_CHAINS

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
        {
            "text": "Emily will Papageien zähmen.",
            "ent_text": "Emily",
            "statement": "Das Fahrrad als Mobilitätsform soll gefördert werden.",
            "stance_true": "irrelevant",
        },
    ]
    return test_egs

@pytest.fixture(scope="module")
def english_examples():
    english_egs = [
        {
            "text": "The city of Bern supports building more bike lanes. However, this is controversial. The FDP is clearly against it.",
            "ent_text": "city of Bern",
            "statement": "Cycling as a mode of transport should be promoted.",
            "stance_true": "support",
        },
        {
            "text": "The city of Bern supports building more bike lanes. However, this is controversial. The FDP is clearly against it.",
            "ent_text": "FDP",
            "statement": "Cycling as a mode of transport should be promoted.",
            "stance_true": "opposition",
        },
        {
            "text": "Emily wants to tame parrots.",
            "ent_text": "Emily",
            "statement": "Cycling as a mode of transport should be promoted.",
            "stance_true": "irrelevant",
        },
    ]
    return english_egs

# LOAD MODELS ------------

@pytest.fixture(scope="module")
def gpt2_trf():
    # gpt2 model from huggingface as example of Transformer model
    gpt2_trf = models.Transformers("openai-community/gpt2")
    return gpt2_trf


# RUN STANCE DETECTIONS FOR ALL CHAINS --------------

# non-masked, iterate across all chains

@pytest.fixture(scope="module")
def stance_detection_runs_trf(test_examples, gpt2_trf):
    classifications = []
    for chain in REGISTERED_LLM_CHAINS:
        for test_eg in test_examples:
            classification = detect_stance(
                eg=test_eg, llm=gpt2_trf, chain_label=chain
            )
            classifications.append(classification)
    return classifications

@pytest.fixture(scope="module")
def stance_detection_runs_english_trf(english_examples, gpt2_trf):
    classifications = []
    for chain in REGISTERED_LLM_CHAINS:
        for test_eg in english_examples:
            classification = detect_stance(
                eg=test_eg, llm=gpt2_trf, chain_label=chain, language="en"
            )
            classifications.append(classification)
    return classifications

# masked, single chain as example

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
