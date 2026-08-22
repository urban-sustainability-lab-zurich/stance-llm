"""Unit tests for evaluation, prodigy prep, and detect_stance input validation.

None of these load a model.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from stance_llm.process import (
    evaluate,
    prepare_prodigy_egs,
    detect_stance,
    process,
    process_evaluate,
)


def test_evaluate_metrics():
    egs = [
        {"stance_pred": "support", "stance_true": "support"},
        {"stance_pred": "support", "stance_true": "support"},
        {"stance_pred": "opposition", "stance_true": "support"},
        {"stance_pred": "error", "stance_true": "irrelevant"},
    ]
    m = evaluate(egs)
    assert m["error_count"] == 1
    # error rows are excluded from y_true/y_pred: 3 true supports, 2 predicted correctly.
    assert m["support"]["precision"] == pytest.approx(1.0)
    assert m["support"]["recall"] == pytest.approx(2 / 3)


def test_prepare_prodigy_egs_flag_handling():
    egs_in = [
        {"par_id": 1, "text": "a", "meta": {"org_text": "O1"}, "statement_de": "s", "accept": ["support"], "flagged": True},
        {"par_id": 2, "text": "b", "meta": {"org_text": "O2"}, "statement_de": "s", "accept": ["opposition"], "flagged": False},
        {"par_id": 3, "text": "c", "meta": {"org_text": "O3"}, "statement_de": "s", "accept": ["irrelevant"]},
    ]
    kept = prepare_prodigy_egs(egs_in, remove_flagged=True)
    assert len(kept) == 2  # flagged=True dropped
    assert {"id", "text", "ent_text", "statement", "stance_true"} <= set(kept[0].keys())
    assert kept[0]["ent_text"] == "O2"
    all_egs = prepare_prodigy_egs(egs_in, remove_flagged=False)
    assert len(all_egs) == 3


def test_detect_stance_missing_key_raises():
    with pytest.raises(KeyError):
        detect_stance({"text": "x", "ent_text": "y"}, llm=Mock(), chain_label="is")


def test_process_error_path(tmp_path):
    # A bogus backend makes both the grammar preflight (best-effort, no raise) and
    # the chain fail; process must record an "error" stance instead of crashing.
    eg = {"text": "t", "ent_text": "e", "statement": "s"}
    out = process(
        egs=[eg],
        llm=SimpleNamespace(),
        export_folder=str(tmp_path),
        model_used="m",
        chain_used="is",
        stream_out=False,
        wait_time=0,
    )
    assert out[0]["stance_pred"] == "error"
    assert out[0]["meta"]["prompt_history"] is None


def test_process_evaluate_empty_raises():
    with pytest.raises(ValueError):
        process_evaluate(egs=[], llm=SimpleNamespace(), model_used="m", chain_used="is")
