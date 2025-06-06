from stance_llm.base import StanceClassification, ALLOWED_STANCE_CATEGORIES
from stance_llm.process import detect_stance, process

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

def test_detect_stance_english_prompt(english_examples, gpt2_trf):
    """Test detect_stance returns correct object and stance for English prompt"""
    for eg in english_examples:
        result = detect_stance(eg, llm=gpt2_trf, chain_label="s2is", language="en")
        assert isinstance(result, StanceClassification)
        assert result.stance in ALLOWED_STANCE_CATEGORIES

def test_process_language_selection(test_examples, english_examples, gpt2_trf, tmp_path):
    """Test process runs with both German and English prompt chains"""
    # German (default)
    results_de = process(
        egs=test_examples,
        llm=gpt2_trf,
        export_folder=str(tmp_path),
        chain_used="s2is",
        model_used="s2is",
        stream_out=False,
        language="de"
    )
    assert all(eg["stance_classification"].stance in ALLOWED_STANCE_CATEGORIES for eg in results_de)
    # English
    results_en = process(
        egs=english_examples,
        llm=gpt2_trf,
        export_folder=str(tmp_path),
        chain_used="s2is",
        model_used="s2is",
        stream_out=False,
        language="en"
    )
    assert all(eg["stance_classification"].stance in ALLOWED_STANCE_CATEGORIES for eg in results_en)
