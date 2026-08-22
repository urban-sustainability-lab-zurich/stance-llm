"""Unit tests for the StanceClassification prompt-chain control flow.

These monkeypatch ``stance_llm.base._run_turn`` so no model is loaded: each chain
step just pops the next scripted response, letting us assert the branching logic
(including the language-specific answer mappings) in isolation.
"""

import pytest

from stance_llm import base
from stance_llm.base import (
    StanceClassification,
    IRRELEVANCE_ANSWERS,
    IRRELEVANCE_ANSWERS2,
    SUMMARY_STANCE_OPTIONS,
)


def _make_task():
    return StanceClassification(
        input_text="Some text about Org.",
        statement="A statement.",
        entity="Org",
    )


def _install_script(monkeypatch, responses):
    """Replace _run_turn with a scripted stub returning `responses` in order."""
    state = {"i": 0, "prompts": []}

    def fake_run_turn(llm, chat, prompt, continuation):
        state["prompts"].append(prompt)
        resp = responses[state["i"]]
        state["i"] += 1
        return resp

    monkeypatch.setattr(base, "_run_turn", fake_run_turn)
    return state


# --- answer-mapping orientation guards (would catch the 47bb08f DE swap) ------


def test_answer_mapping_orientation():
    assert "keine" in IRRELEVANCE_ANSWERS["de"]["irrelevant"]
    assert "not" in IRRELEVANCE_ANSWERS["en"]["irrelevant"]
    assert "keine" in IRRELEVANCE_ANSWERS2["de"]["irrelevant"]
    assert "not" in IRRELEVANCE_ANSWERS2["en"]["irrelevant"]
    # "stance" side must NOT carry the negation.
    assert "keine" not in IRRELEVANCE_ANSWERS["de"]["stance"]
    assert "keine" not in IRRELEVANCE_ANSWERS2["de"]["stance"]


# --- nested chains general gate ------------------------------------------------


@pytest.mark.parametrize("language", ["de", "en"])
def test_nise_general_gate_irrelevant_short_circuits(monkeypatch, language):
    state = _install_script(
        monkeypatch,
        [{"answer_general": IRRELEVANCE_ANSWERS2[language]["irrelevant"]}],
    )
    task = _make_task()
    task.nested_irrelevant_summary_explicit(llm=object(), chat=False, language=language)
    assert task.stance == "irrelevant"
    assert state["i"] == 1  # only the general gate ran


@pytest.mark.parametrize("language", ["de", "en"])
def test_nise_general_gate_stance_proceeds(monkeypatch, language):
    # A stance-taking general answer must NOT be classified irrelevant at gate 1;
    # it proceeds to the relatedness gate. Guards the German key orientation.
    state = _install_script(
        monkeypatch,
        [
            {"answer_general": IRRELEVANCE_ANSWERS2[language]["stance"]},
            {"answer": IRRELEVANCE_ANSWERS[language]["irrelevant"]},
        ],
    )
    task = _make_task()
    task.nested_irrelevant_summary_explicit(llm=object(), chat=False, language=language)
    assert state["i"] == 2  # proceeded past the general gate
    assert task.stance == "irrelevant"  # decided by the relatedness gate


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize(
    "support_yes,opposition_yes,expected",
    [
        (True, None, "support"),
        (False, True, "opposition"),
        (False, False, "irrelevant"),
    ],
)
def test_nise_full_path(monkeypatch, language, support_yes, opposition_yes, expected):
    yes = "Ja" if language == "de" else "Yes"
    no = "Nein" if language == "de" else "No"
    responses = [
        {"answer_general": IRRELEVANCE_ANSWERS2[language]["stance"]},
        {"answer": IRRELEVANCE_ANSWERS[language]["stance"]},
        {"summary": "position summary"},
        {"answer": yes if support_yes else no},
    ]
    if opposition_yes is not None:
        responses.append({"answer": yes if opposition_yes else no})
    _install_script(monkeypatch, responses)
    task = _make_task()
    task.nested_irrelevant_summary_explicit(llm=object(), chat=False, language=language)
    assert task.stance == expected
    assert task.meta["llms"]["irrelevance_general"] is not None


# --- irrelevant_summarize_v2 (is2) --------------------------------------------


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize("idx,expected", [(0, "irrelevant"), (1, "support"), (2, "opposition")])
def test_is2_summary_stance_select_maps(monkeypatch, language, idx, expected):
    responses = [
        {"answer": IRRELEVANCE_ANSWERS[language]["stance"]},
        {"stance": SUMMARY_STANCE_OPTIONS[language][idx], "summary": "..."},
    ]
    _install_script(monkeypatch, responses)
    task = _make_task()
    task.irrelevant_summarize_v2_chain(llm=object(), chat=False, language=language)
    assert task.stance == expected


@pytest.mark.parametrize("language", ["de", "en"])
def test_is2_irrelevance_short_circuits(monkeypatch, language):
    state = _install_script(
        monkeypatch,
        [{"answer": IRRELEVANCE_ANSWERS[language]["irrelevant"]}],
    )
    task = _make_task()
    task.irrelevant_summarize_v2_chain(llm=object(), chat=False, language=language)
    assert task.stance == "irrelevant"
    assert state["i"] == 1
    assert task.meta["llms"]["summary"] is None


# --- is / sis / s2 quick sanity via helpers -----------------------------------


@pytest.mark.parametrize("language", ["de", "en"])
def test_is_support(monkeypatch, language):
    yes = "Ja" if language == "de" else "Yes"
    _install_script(
        monkeypatch,
        [
            {"answer": IRRELEVANCE_ANSWERS[language]["stance"]},
            {"answer": yes},
        ],
    )
    task = _make_task()
    task.irrelevant_stance_chain(llm=object(), chat=False, language=language)
    assert task.stance == "support"


@pytest.mark.parametrize("language", ["de", "en"])
def test_s2_maps_summary_select(monkeypatch, language):
    _install_script(
        monkeypatch,
        [{"stance": SUMMARY_STANCE_OPTIONS[language][2], "summary": "..."}],
    )
    task = _make_task()
    task.summarize_v2_chain(llm=object(), chat=False, language=language)
    assert task.stance == "opposition"


# --- mask_entity literal replacement (no regex) -------------------------------


def test_mask_entity_literal_with_regex_chars():
    task = StanceClassification(
        input_text="Die AG (Zürich) unterstützt das Projekt.",
        statement="x",
        entity="AG (Zürich)",
    )
    returned = task.mask_entity("Organisation X")
    assert returned is task
    assert "AG (Zürich)" not in task.masked_input_text
    assert "Organisation X" in task.masked_input_text
    assert task.masked_entity == "Organisation X"
