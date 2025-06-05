import pytest
import re
import os
import shutil
from unittest.mock import Mock

from stance_llm.base import StanceClassification, get_registered_chains, get_registered_chains_keys, REGISTERED_LLM_CHAINS
from stance_llm.process import detect_stance, make_export_folder


def test_mask_entity_replaces_entity_and_returns_self():
    text = "Die Stadtverwaltung von Bern unterstützt den Bau von neuen Fahrradwegen"
    entity = "Stadtverwaltung von Bern"
    mask = "Organization A"
    stance_obj = StanceClassification(input_text=text, statement="Das Fahrrad als Mobilitätsform sollte unterstützt werden", entity=entity)

    returned_obj = stance_obj.mask_entity(mask)

    assert isinstance(returned_obj, StanceClassification)
    assert returned_obj is stance_obj  # returns self

    # The masked_input_text should replace "Stadtverwaltung von Bern" with "Organization A"
    assert "Stadtverwaltung von Bern" not in returned_obj.masked_input_text
    assert mask in returned_obj.masked_input_text
    # The masked_entity should be updated
    assert returned_obj.masked_entity == mask


@pytest.mark.parametrize("chain_label", get_registered_chains_keys())
def test_detect_stance_calls_chain_method(monkeypatch, chain_label):
    # Prepare example input
    example = {
        "text": "Example text mentioning entity.",
        "ent_text": "entity",
        "statement": "Test statement.",
    }
    # Mock a dummy LLM model
    mock_llm = Mock()
    # Static correct mapping from chain_label to method name in StanceClassification
    chain_method_map = {
        "sis": "summarize_irrelevant_stance_chain",
        "is": "irrelevant_stance_chain",
        "nise": "nested_irrelevant_summary_explicit",
        "s2is": "summarize_v2_irrelevant_stance_chain",
        "s2": "summarize_v2_chain",
        "is2": "irrelevant_summarize_v2_chain",
        "nis2e": "nested_irrelevant_summary_v2_explicit",
    }
    method_name = chain_method_map.get(chain_label, None)
    if method_name is None:
        # Skip chain labels not covered in detect_stance function logic
        pytest.skip(f"Chain label {chain_label} not covered")

    called = False

    def mock_method(self, *args, **kwargs):
        nonlocal called
        called = True
        self.stance = "dummy_stance"
        return self

    monkeypatch.setattr(StanceClassification, method_name, mock_method)

    classification = detect_stance(example, mock_llm, chain_label)

    assert called, f"Method {method_name} was not called"
    assert classification.stance == "dummy_stance"


def test_detect_stance_invalid_chain_label_raises():
    example = {
        "text": "some text",
        "ent_text": "entity",
        "statement": "a statement",
    }
    mock_llm = Mock()
    with pytest.raises(NameError):
        detect_stance(example, mock_llm, "invalid_chain_label")


def test_detect_stance_invalid_dual_llm_chain_raises():
    example = {
        "text": "some text",
        "ent_text": "entity",
        "statement": "a statement",
    }
    mock_llm = Mock()
    mock_llm2 = Mock()

    # We'll use a chain that is registered but block dual usage: "is"
    with pytest.raises(NameError):
        detect_stance(example, mock_llm, "is", llm2=mock_llm2)


def test_make_export_folder_creates_and_returns_path(tmp_path):
    base_folder = tmp_path / "exports"
    model_used = "modelx"
    chain_used = "chainy"
    run_alias = "run1"

    folder_path = make_export_folder(str(base_folder), model_used, chain_used, run_alias)

    expected_path = base_folder / chain_used / model_used / str(folder_path.split(os.sep)[-2]) / run_alias
    # Check folder is created
    assert os.path.exists(folder_path)
    # Check returned path is the same as created one
    assert folder_path == str(expected_path)

    # Call again, should return without creating again
    folder_path2 = make_export_folder(str(base_folder), model_used, chain_used, run_alias)
    assert folder_path2 == folder_path