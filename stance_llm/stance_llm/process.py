import os
import time
from datetime import date

import srsly
from tqdm import tqdm
from sklearn.metrics import classification_report
from loguru import logger
from wonderwords import RandomWord

from stance_llm.base import StanceClassification, get_registered_chains, get_allowed_dual_lm_chains

def detect_stance(eg, lm, chain_label: str, lm2=None, chat=True, entity_mask=None):
    """Detect stance of an entity in a dictionary input

    Expects a dictionary item with a "text" key containing text to classify and
    a "meta" key containing a dictionary with at least a key "org_text" containing
    a string for the organizational entity to detect stance for and a key "statement"
    containing the statement to evaluate the stance for

    Args:
        eg: A dictionary item with a "text" key containing text to classify and a "org_text" key containing a string for the organizational entity to predict stance for and a key "statement" containing the statement to evaluate the stance for
        lm: A guidance model backend from guidance.models
        chain_label: A implemented lm chain. See stance_llm.base.get_registered_chains for list
    """
    chain_labels = get_registered_chains()
    if chain_label not in chain_labels:
        raise NameError('Chain label is not registered')
    if lm2 is not None:
        allowed_dual_lm_labels = get_allowed_dual_lm_chains()
        if chain_label not in allowed_dual_lm_labels:
            raise NameError(f"Prompt chain is not set up for using two lm backends. Allowed are {allowed_dual_lm_labels}")
    entity = eg["org_text"]
    text = eg["text"]
    statement = eg["statement"]
    task = StanceClassification(
        input_text=text,
        statement=statement,
        entity=entity)
    if entity_mask is not None:
        task = task.mask_entity(entity_mask=entity_mask)
    if chain_label == "sis":
        classification = task.summarize_irrelevant_stance_chain(lm=lm,chat = chat,lm2=lm2)
    if chain_label == "is":
        classification = task.irrelevant_stance_chain(lm=lm,chat = chat,lm2=lm2)
    if chain_label == "nise":
        classification = task.nested_irrelevant_summary_explicit(lm=lm,chat = chat,lm2=lm2)
    if chain_label == "s2is":
        classification = task.summarize_v2_irrelevant_stance_chain(lm=lm,chat = chat,lm2=lm2)
    if chain_label == "s2":
        classification = task.summarize_v2_chain(lm=lm,chat = chat,lm2=lm2)
    if chain_label == "is2":
        classification = task.irrelevant_summarize_v2_chain(lm=lm, chat=chat,lm2=lm2)
    if chain_label == "nis2e":
        classification = task.nested_irrelevant_summary_v2_explicit(lm=lm,chat = chat,lm2=lm2)
    return(classification)

def make_export_folder(export_folder,
                          model_used,
                          chain_used,
                          run_alias):
    today = str(date.today())
    folder_path = os.path.join(export_folder,chain_used,model_used,today,run_alias)
    if os.path.exists(folder_path):
        return(folder_path)
    else:
        logger.info(f"Creating folder at {folder_path}")
        os.makedirs(folder_path)
        return(folder_path)

def get_prompt_texts_from_meta(classification: StanceClassification):
    if "lms" not in classification.meta:
        return({})
    else:
        components = {}
        chain_components = classification.meta["lms"]
        for component_key in chain_components.keys():
            components[component_key] = {"prompt_text":str(chain_components[component_key])}
    return(components)


def process(
    egs,
    lm,
    export_folder,
    model_used,
    chain_used,
    true_stance_key=None,
    wait_time=5,
    stream_out=True,
    id_key = "id",
    chat=True,
    lm2=None,
    entity_mask=None):
    r_word = RandomWord()
    run_alias = "-".join(r_word.random_words(2))
    logger.info(f"Starting run {run_alias}")
    pred_egs = []
    for eg in tqdm(egs):
        eg["stance_classification"] = detect_stance(eg,lm=lm,
                                                    chain_label=chain_used,
                                                    chat=chat,
                                                    lm2=lm2,
                                                    entity_mask=entity_mask)
        eg["run_alias"] = run_alias
        eg["stance_pred"] = eg["stance_classification"].stance
        eg["meta"] = {
            "prompt_history": get_prompt_texts_from_meta(classification=eg["stance_classification"])
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
                id_key=id_key)
        time.sleep(wait_time)
    logger.info(f"finished run {run_alias}")
    return(pred_egs)

def evaluate(egs_with_preds):
    y_true = []
    y_pred = []
    for eg in egs_with_preds:
        y_pred.append(eg["stance_pred"])
        y_true.append(eg["stance_true"])
    classes = ["support","opposition","irrelevant"]
    logger.info("Creating evaluation report")
    eval_metrics = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True)
    logger.info(f"----------- Evaluation metrics ------------ \n {eval_metrics} \n ------------------")
    return(eval_metrics)

def save_evaluations_json(export_folder,
                          eval_metrics,
                          chain_used,
                          model_used,
                          run_alias):
    export_folder_path = make_export_folder(export_folder=export_folder,
                                            chain_used=chain_used,
                                            model_used=model_used,
                                            run_alias=run_alias)
    out_dict = {
        "run_alias": run_alias,
        "metrics": eval_metrics
    }
    logger.info(f"Saving evaluation report to {str(export_folder_path)}")
    srsly.write_json(os.path.join(export_folder_path,"metrics.jsonl"),
                     out_dict)
    
def save_run_meta_info_json(export_folder,
                        chain_used,
                        model_used,
                        run_alias,
                        entity_mask):
    export_folder_path = make_export_folder(export_folder=export_folder,
                                            chain_used=chain_used,
                                            model_used=model_used,
                                            run_alias=run_alias)
    if entity_mask is not None:
        entity_masking = entity_mask
    else:
        entity_masking = "None"
    out_dict = {
        "run_alias": run_alias,
        "chain_used": chain_used,
        "model_used": model_used,
        "date_run": str(date.today()),
        "entity_masking": entity_masking
    }
    logger.info(f"Saving run meta-information to {str(export_folder_path)}")
    srsly.write_json(os.path.join(export_folder_path,"meta.jsonl"),
                     out_dict)

def save_classifications_jsonl(export_folder,
                               egs_with_classifications, 
                               model_used, 
                               chain_used,
                               run_alias, 
                               id_key = None,
                               true_stance_key=None):
    to_export = []
    for eg in egs_with_classifications:
        if "stance_classification" in eg.keys():
            export_dict = {
                "text": eg["text"],
                "org_text": eg["org_text"],
                "statement": eg["statement"],
                "stance_pred": eg["stance_classification"].stance,
                "model_used": model_used,
                "chain_used": chain_used,
                "run_alias": run_alias,
                "meta": eg["meta"]
                }
            if id_key is not None:
                export_dict = export_dict | {"id": eg[id_key]}
            if true_stance_key is not None:
                export_dict = export_dict | {"stance_true": eg[true_stance_key]}
            to_export.append(export_dict)
    export_subfolder = make_export_folder(export_folder=export_folder,
                                          model_used=model_used,
                                          chain_used=chain_used,
                                          run_alias=run_alias)
    filepath = os.path.join(export_subfolder,"classifications.jsonl")
    srsly.write_jsonl(filepath,to_export)

def prepare_prodigy_egs(prodigy_egs, remove_flagged = True):
    """Helper to convert exports from prodigy to simpler list to pass to stance annotations tasks.

    Args:
        prodigy_egs: List of dicts exported from prodigy db
    """
    egs = []
    if remove_flagged:
        logger.info("Removing flagged evaluation examples. To avoid this, set remove_flagged to False")
    for eg in prodigy_egs:
        refactored = { 
            "id": eg["par_id"],
            "text": eg["text"],
            "org_text": eg["meta"]["org_text"],
            "statement": eg["statement_de"],
            "stance_true": eg["accept"][0]
        }
        if remove_flagged:
            if 'flagged' in eg.keys():
                if eg['flagged'] is not True:
                    egs.append(refactored)
            if 'flagged' not in eg.keys():
                    egs.append(refactored)
        if remove_flagged is not True:
            egs.append(refactored)
    return(egs)

def process_evaluate(egs,
               lm,
               model_used,
               chain_used,
               chat=True,
               wait_time=0.5,
               export_folder="./evaluations",
               lm2=None,
               entity_mask=None):
    """Process a list of examples to via a lm backend, stream out results, evaluate against true values and save evaluations

    Args:
        egs: A list of dictionary items with a "text" key containing text to classify, a "org_text" key containing a string for the organizational entity to predict stance for and a "stance_true" key containing a true stance to evaluate against
        lm: A guidance model backend from guidance.models
        model_used: String giving label for model backend
        chain_used: An implemented lm chain. See stance_llm.base.get_registered_chains for list
        chat (bool, optional): Should a chat model variant be used? Defaults to True.
        export_folder (str, optional): Folder for evaluation output. Defaults to "./evaluations".
    """
    preds = process(
        egs = egs,
        lm=lm,
        export_folder=export_folder,
        model_used=model_used,
        chain_used=chain_used,
        wait_time=wait_time,
        true_stance_key="stance_true",
        stream_out=True,
        chat=chat,
        lm2=lm2,
        entity_mask=entity_mask
        )
    eval_metrics = evaluate(preds)
    run_alias = preds[0]["run_alias"]
    assert all([pred["run_alias"] == run_alias for pred in preds])
    save_evaluations_json(export_folder=export_folder, 
                          eval_metrics=eval_metrics,
                          model_used=model_used,
                          chain_used=chain_used,
                          run_alias=run_alias)
    save_run_meta_info_json(export_folder=export_folder, 
                        model_used=model_used,
                        chain_used=chain_used,
                        run_alias=run_alias,
                        entity_mask = entity_mask)
    return(preds)