import os
import time
from datetime import date
from typing_extensions import Self

import srsly
from tqdm import tqdm
from sklearn.metrics import classification_report
from loguru import logger
from wonderwords import RandomWord

from stance_llm.base import (
    StanceClassification,
    get_registered_chains,
    get_allowed_dual_llm_chains,
)


def detect_stance(
    eg: dict, llm, chain_label: str, llm2=None, chat=True, entity_mask=None
) -> Self:
    """Detect stance of an entity in a dictionary input

    Expects a dictionary item with a "text" key containing text to classify, a key "ent_text"
    containing a string matching the entity to detect stance for and a key "statement"
    containing the statement to evaluate the stance against

    Args:
        eg: A dictionary item with a "text" key containing text to classify and a "ent_text" key containing a string matching the organizational entity to predict stance for and a key "statement" containing the statement to evaluate the stance against
        llm: A guidance model backend from guidance.models
        chain_label: A implemented llm chain. See stance_llm.base.get_registered_chains for list

    Returns:
        A StanceClassification class object with a stance and meta data
    """
    if 'text' not in eg.keys():
        logger.error("Input dictionary for classification has not text key")
    if 'ent_text' not in eg.keys():
        logger.error("Input dictionary for classification has not ent_text key")
    if 'statement' not in eg.keys():
        logger.error("Input dictionary for classification has not statement key")
    chain_labels = get_registered_chains()
    if chain_label not in chain_labels:
        raise NameError("Chain label is not registered")
    if llm2 is not None:
        allowed_dual_llm_labels = get_allowed_dual_llm_chains()
        if chain_label not in allowed_dual_llm_labels:
            raise NameError(
                f"Prompt chain is not set up for using two llm backends. Allowed are {allowed_dual_llm_labels}"
            )
    entity = eg["ent_text"]
    text = eg["text"]
    statement = eg["statement"]
    task = StanceClassification(input_text=text, statement=statement, entity=entity)
    if entity_mask is not None:
        task = task.mask_entity(entity_mask=entity_mask)
    if chain_label == "sis":
        classification = task.summarize_irrelevant_stance_chain(
            llm=llm, chat=chat, llm2=llm2
        )
    if chain_label == "is":
        classification = task.irrelevant_stance_chain(llm=llm, chat=chat, llm2=llm2)
    if chain_label == "nise":
        classification = task.nested_irrelevant_summary_explicit(
            llm=llm, chat=chat, llm2=llm2
        )
    if chain_label == "s2is":
        classification = task.summarize_v2_irrelevant_stance_chain(
            llm=llm, chat=chat, llm2=llm2
        )
    if chain_label == "s2":
        classification = task.summarize_v2_chain(llm=llm, chat=chat, llm2=llm2)
    if chain_label == "is2":
        classification = task.irrelevant_summarize_v2_chain(
            llm=llm, chat=chat, llm2=llm2
        )
    if chain_label == "nis2e":
        classification = task.nested_irrelevant_summary_v2_explicit(
            llm=llm, chat=chat, llm2=llm2
        )
    return classification


def make_export_folder(
    export_folder: str, model_used, chain_used: str, run_alias: str
) -> str:
    """creates folder of format <export_folder/<chain_used>/<model_used>/<current date>/<run_alias>

    Args:
        export_folder: directory to which all outputs are saved
        model_used: llm model name
        chain_used: prompt chain (short name)
        run_alias: name of the classification run to be saved
    """
    today = str(date.today())
    folder_path = os.path.join(export_folder, chain_used, model_used, today, run_alias)
    if os.path.exists(folder_path):
        return folder_path
    else:
        logger.info(f"Creating folder at {folder_path}")
        os.makedirs(folder_path)
        return folder_path


def get_prompt_texts_from_meta(classification: StanceClassification) -> dict:
    """pulls prompt texts from meta data and returns it

    Args:
        classification: StanceClassification class object, with predictions ideally

    Returns:
        dict: gets all the meta information - created during the stance classification with a prompt chain - and returns it as a string (instead of a nested dictionary)
    """
    if "llms" not in classification.meta:
        return {}
    else:
        components = {}
        chain_components = classification.meta["llms"]
        for component_key in chain_components.keys():
            components[component_key] = {
                "prompt_text": str(chain_components[component_key])
            }
    return components


def process(
    egs,
    llm,
    export_folder: str,
    model_used: str,
    chain_used: str,
    true_stance_key=None,
    wait_time=5,
    stream_out=True,
    id_key=None,
    chat=True,
    llm2=None,
    entity_mask=None,
):
    r_word = RandomWord()
    run_alias = "-".join(r_word.random_words(2))
    logger.info(f"Starting run {run_alias}")
    """serves like a main function that
     - sends data together with constructed prompts to the llm (detect_stance())
     - assigns run alias (specific name) and saves classifications together with prompt texts (get_prompt_texts_from_meta() & save_classifications_jsonl())
    
    Args:
        egs: list of examples to classify as dictionaries with at least keys "text","ent_text","statement" (see detect_stance())
        llm: A guidance model backend from guidance.models
        export_folder: Folder for evaluation output.
        model_used: name of the currently employed llm
        chain_used: name of propt chain of the current execution
        true_stance_key: contains true stance. Defaults to None.
        wait_time: Wait time between two prompts sent to the llm. Defaults to 5.
        id_key = id of the instance. Defaults to None.

    Return:
        Returns the classifications (with text, statement, etc.) together with the extracted predicted stance ("pred_stance") from out of the StanceClassification class attribute "stance" as well as the prompt texts from the attribute "meta"
    
    """
    pred_egs = []
    for eg in tqdm(egs):
        try:
            eg["stance_classification"] = detect_stance(
                eg,
                llm=llm,
                chain_label=chain_used,
                chat=chat,
                llm2=llm2,
                entity_mask=entity_mask,
            )
            eg["run_alias"] = run_alias
            eg["stance_pred"] = eg["stance_classification"].stance
            eg["meta"] = {
                "prompt_history": get_prompt_texts_from_meta(
                    classification=eg["stance_classification"]
                )
            }
            if entity_mask is not None:
                eg["meta"] = eg["meta"] | {"entity_mask": entity_mask}
            pred_egs.append(eg)
            if stream_out:
                save_classifications_jsonl(
                    export_folder=export_folder,
                    egs_with_classifications=pred_egs,
                    model_used=model_used,
                    chain_used=chain_used,
                    run_alias=run_alias,
                    true_stance_key=true_stance_key,
                    id_key=id_key,
                )
            time.sleep(wait_time)
        except:
            # if error return error stance classification
            logger.error(f"Classification failed for task. Writing error to stance_pred.")
            eg["run_alias"] = run_alias
            eg["stance_pred"] = "error"
            eg["meta"] = {
                "prompt_history": None
            }
            if entity_mask is not None:
                eg["meta"] = eg["meta"] | {"entity_mask": entity_mask}
            pred_egs.append(eg)
            if stream_out:
                save_classifications_jsonl(
                    export_folder=export_folder,
                    egs_with_classifications=pred_egs,
                    model_used=model_used,
                    chain_used=chain_used,
                    run_alias=run_alias,
                    true_stance_key=true_stance_key,
                    id_key=id_key,
                )
            time.sleep(wait_time)
    if stream_out:
        save_run_meta_info_json(
            export_folder=export_folder,
            model_used=model_used,
            chain_used=chain_used,
            run_alias=run_alias,
            entity_mask=entity_mask,
        )
    logger.info(f"finished run {run_alias}")
    return pred_egs


def evaluate(egs_with_preds):
    """creates and outputs evaluation metrics: for each stance class: precision, recall, f1, accuracy, and macro (precision, recall, F1, accuracy) and micro (precision, recall, F1, accuracy)

    Args:
        egs_with_preds: list of dictionaries containing predicted stances at a key "stance_pred" and true stances under at key "stance_true"
    """
    y_true = []
    y_pred = []
    for eg in egs_with_preds:
        if eg["stance_pred"] != "error":
            y_pred.append(eg["stance_pred"])
            y_true.append(eg["stance_true"])
    classes = ["support", "opposition", "irrelevant"]
    logger.info("Creating evaluation report")
    eval_metrics = classification_report(
        y_true, y_pred, labels=classes, output_dict=True
    )
    eval_metrics["error_count"] = len([eg for eg in egs_with_preds if eg["stance_pred"] == "error"])
    logger.info(
        f"----------- Evaluation metrics ------------ \n {eval_metrics} \n ------------------"
    )
    return eval_metrics


def save_evaluations_json(
    export_folder: str, eval_metrics, chain_used: str, model_used: str, run_alias: str
) -> None:
    """serializes metrics to metrics.json file at <export_folder/<chain_used>/<model_used>/<current date>/<run_alias>

    Args:
        export_folder: directory target for serialization
        eval_metrics: dictionary with evaluation metrics per stance class and macro and average (for each: precision ,recall, F1, accuracy)
        chain_used: prompt chain (short name)
        model_used: llm model name
        run_alias: name of the classification run to be saved
    """
    export_folder_path = make_export_folder(
        export_folder=export_folder,
        chain_used=chain_used,
        model_used=model_used,
        run_alias=run_alias,
    )
    out_dict = {"run_alias": run_alias, "metrics": eval_metrics}
    logger.info(f"Saving evaluation report to {str(export_folder_path)}")
    srsly.write_json(os.path.join(export_folder_path, "metrics.json"), out_dict)


def save_run_meta_info_json(
    export_folder: str,
    chain_used: str,
    model_used: str,
    run_alias: str,
    entity_mask: str,
) -> None:
    """serializes run meta information to meta.json file at <export_folder/<chain_used>/<model_used>/<current date>/<run_alias>

    Args:
        export_folder: directory target for serialization
        chain_used: prompt chain (short name)
        model_used: llm model name
        run_alias: name of the classification run to be saved
        entity_mask: string used to mask the original entity string in the classified text, if any is given

    """
    export_folder_path = make_export_folder(
        export_folder=export_folder,
        chain_used=chain_used,
        model_used=model_used,
        run_alias=run_alias,
    )
    if entity_mask is not None:
        entity_masking = entity_mask
    else:
        entity_masking = "None"
    out_dict = {
        "run_alias": run_alias,
        "chain_used": chain_used,
        "model_used": model_used,
        "date_run": str(date.today()),
        "entity_masking": entity_masking,
    }
    logger.info(f"Saving run meta-information to {str(export_folder_path)}")
    srsly.write_json(os.path.join(export_folder_path, "meta.json"), out_dict)


def save_classifications_jsonl(
    export_folder: str,
    egs_with_classifications,
    model_used: str,
    chain_used: str,
    run_alias: str,
    id_key=None,
    true_stance_key=None,
) -> None:
    """serializes a list of stance classifications to JSONL in a classifications.jsonl file at <export_folder/<chain_used>/<model_used>/<current date>/<run_alias>

    Args:
        export_folder: directory target for serialization
        egs_with_classifications (list): List of stance classifications
        model_used: llm model name
        chain_used: prompt chain (short name)
        run_alias: name of the classification run to be saved
        id_key (optional): id of the example. Defaults to None.
        true_stance_key (optional): contains true stance. Defaults to None.
    """
    to_export = []
    for eg in egs_with_classifications:
        if "stance_classification" in eg.keys():
            export_dict = {
                "text": eg["text"],
                "ent_text": eg["ent_text"],
                "statement": eg["statement"],
                "stance_pred": eg["stance_classification"].stance,
                "model_used": model_used,
                "chain_used": chain_used,
                "run_alias": run_alias,
                "meta": eg["meta"],
            }
            if id_key is not None:
                export_dict = export_dict | {"id": eg[id_key]}
            if true_stance_key is not None:
                export_dict = export_dict | {"stance_true": eg[true_stance_key]}
            to_export.append(export_dict)
    export_subfolder = make_export_folder(
        export_folder=export_folder,
        model_used=model_used,
        chain_used=chain_used,
        run_alias=run_alias,
    )
    filepath = os.path.join(export_subfolder, "classifications.jsonl")
    srsly.write_jsonl(filepath, to_export)


def prepare_prodigy_egs(prodigy_egs, remove_flagged=True):
    """Helper to convert exports from prodigy annotations to simpler list to pass to stance annotations tasks.


    Args:
        prodigy_egs: List of dicts exported from prodigy db with keys
        "par_id","text",["meta"]["org"],"statement_de"
        and annotated stances at ["accept"][0]
        remove_flagged (bool, optional): remove examples marked as flagged. defaults to True
    """
    egs = []
    if remove_flagged:
        logger.info(
            "Removing flagged evaluation examples. To avoid this, set remove_flagged to False"
        )
    for eg in prodigy_egs:
        refactored = {
            "id": eg["par_id"],
            "text": eg["text"],
            "ent_text": eg["meta"]["org_text"],
            "statement": eg["statement_de"],
            "stance_true": eg["accept"][0],
        }
        if remove_flagged:
            if "flagged" in eg.keys():
                if eg["flagged"] is not True:
                    egs.append(refactored)
            if "flagged" not in eg.keys():
                egs.append(refactored)
        if remove_flagged is not True:
            egs.append(refactored)
    return egs


def process_evaluate(
    egs,
    llm,
    model_used: str,
    chain_used: str,
    chat=True,
    wait_time=0.5,
    export_folder="./evaluations",
    llm2=None,
    entity_mask=None,
):
    """Process a list of examples to via a llm backend, stream out results, evaluate against true values and save evaluations

    Args:
        egs: A list of dictionary items with a "text" key containing text to classify, a "ent_text" key containing a string for the organizational entity to predict stance for and a "stance_true" key containing a true stance to evaluate against
        llm: A guidance model backend from guidance.models
        model_used: String giving label for model backend
        chain_used: An implemented llm chain. See stance_llm.base.get_registered_chains for list
        chat (bool, optional): Should a chat model variant be used? Defaults to True.
        wait_time (int): Wait time (in seconds) between two prompts sent to the llm. Defaults to 5.
        export_folder (str, optional): Folder for evaluation output. Defaults to "./evaluations".
    """
    preds = process(
        egs=egs,
        llm=llm,
        export_folder=export_folder,
        model_used=model_used,
        chain_used=chain_used,
        wait_time=wait_time,
        true_stance_key="stance_true",
        stream_out=True,
        chat=chat,
        llm2=llm2,
        entity_mask=entity_mask,
    )
    eval_metrics = evaluate(preds)
    run_alias = preds[0]["run_alias"]
    assert all([pred["run_alias"] == run_alias for pred in preds])
    save_evaluations_json(
        export_folder=export_folder,
        eval_metrics=eval_metrics,
        model_used=model_used,
        chain_used=chain_used,
        run_alias=run_alias,
    )
    save_run_meta_info_json(
        export_folder=export_folder,
        model_used=model_used,
        chain_used=chain_used,
        run_alias=run_alias,
        entity_mask=entity_mask,
    )
    return preds
