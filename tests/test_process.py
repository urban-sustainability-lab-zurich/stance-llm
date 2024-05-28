import os
import shutil
import pathlib
import srsly

from stance_llm.process import process, process_evaluate


def test_process_creates_folder_contents(test_examples,
                                 gpt35_openai,
                                 test_output_dir):
    process(
        egs=test_examples,
        llm=gpt35_openai,
        export_folder=test_output_dir,
        chain_used="s2is",
        model_used="gpt35",
        stream_out=True
    )
    # list all files
    out = pathlib.Path(test_output_dir)
    file_list = [str(item.name) for item in list(out.rglob("*")) if item.is_file()]
    shutil.rmtree(test_output_dir)
    assert "classifications.jsonl" in file_list
    assert "meta.json" in file_list

def test_process_outputs_classifications(test_examples,
                                 gpt35_openai,
                                 test_output_dir):
    process(
        egs=test_examples,
        llm=gpt35_openai,
        export_folder=test_output_dir,
        chain_used="s2is",
        model_used="gpt35",
        stream_out=True
    )
    # list all files
    out = pathlib.Path(test_output_dir)
    classifications_file = [item for item in list(out.rglob("classifications.jsonl")) if item.is_file()]
    egs_with_classifications = list(srsly.read_jsonl(classifications_file[0]))
    shutil.rmtree(test_output_dir)
    assert len(classifications_file) == 1
    assert all([eg["stance_pred"] in ["support","opposition","irrelevant"] for eg in egs_with_classifications])
    assert len(egs_with_classifications) == len(test_examples)

def test_process_evaluate_creates_folder_contents(test_examples,
                                 gpt35_openai,
                                 test_output_dir):
    process_evaluate(
        egs=test_examples,
        llm=gpt35_openai,
        export_folder=test_output_dir,
        chain_used="s2is",
        model_used="gpt35",
    )
    # list all files
    out = pathlib.Path(test_output_dir)
    file_list = [str(item.name) for item in list(out.rglob("*")) if item.is_file()]
    shutil.rmtree(test_output_dir)
    assert "classifications.jsonl" in file_list
    assert "meta.json" in file_list
    assert "metrics.json" in file_list