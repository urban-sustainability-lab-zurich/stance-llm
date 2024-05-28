import os
import shutil
import pathlib

from stance_llm.process import process


def test_process_creates_folder(test_examples,
                                 gpt35_openai):
    test_output_dir = "tests/test_output"
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