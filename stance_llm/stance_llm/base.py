from loguru import logger
import re

from guidance import models, gen, select, user, system, assistant, instruction

REGISTERED_LM_CHAINS = {
    "sis": "summarize_irrelevant_stance",
    "s2is": "summarize_v2_irrelevant_stance",
    "s2": "summarize_v2",
    "is": "irrelevant_stance",
    "is2": "irrelevant_summarize_v2",
    "nise": "nested_irrelevant_summary_explicit",
    "nis2e": "nested_irrelevant_summary_v2_explicit"
    }

ALLOWED_DUAL_LM_CHAINS = ["is2"]

IRRELEVANCE_ANSWERS = {
    "irrelevant": 'Bezieht keine Stellung',
    "stance": 'Bezieht Stellung'
}
IRRELEVANCE_ANSWERS2 = {
    "irrelevant": 'Bezieht keine Haltung',
    "stance": 'Bezieht eine Haltung'
}

def construct_irrelevance_prompt(input_text,
                                 entity,
                                 statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Bezieht die Organisation {entity} Stellung zur folgenden Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS['irrelevant']} oder {IRRELEVANCE_ANSWERS['stance']}"
    return(prompt)

def construct_summary_prompt(input_text,
                             entity):
    prompt = f'Fasse die Position der Organisation {entity} im folgenden Text zusammen: \n {input_text}. Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}...'
    return(prompt)

def construct_summary_statementspecific_prompt(input_text,
                                               entity,
                                               statement):
    prompt = f'Fasse die Position der Organisation {entity} im folgenden Text in Bezug auf die Aussage "{statement}" zusammen: \n {input_text}. \n Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}...'
    return(prompt)

def construct_general_stance_prompt(input_text,
                                        entity):
    prompt = f"Analysiere den folgenden Text: {input_text}. Äussert die Organisation {entity} eine implizite oder explizite Haltung? Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS2['irrelevant']} oder {IRRELEVANCE_ANSWERS2['stance']}"
    return(prompt)

def construct_support_stance_prompt(input_text,
                            entity,
                            statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Befürwortet die Organisation {entity} die Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
    return(prompt)

def construct_opposition_stance_prompt(input_text,
                            entity,
                            statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Lehnt die Organisation {entity} folgende die Aussage ab: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
    return(prompt)

def get_registered_chains():
    return(REGISTERED_LM_CHAINS)

def get_allowed_dual_lm_chains():
    return(ALLOWED_DUAL_LM_CHAINS)

class StanceClassification:
    """constructs German prompt chains
    """
    def __init__(self,input_text,statement,entity):
        self.input_text = input_text
        self.statement = statement
        self.entity = entity
        self.stance = None
        self.meta = None
        self.masked_entity = entity
        self.masked_input_text = input_text
    def __str__(self):
        return("The stance of entity {} towards the statement {} given text {} is {}".format(
            self.entity,
            self.statement,
            self.input_text,
            self.stance))
    
    def mask_entity(self:"StanceClassification", entity_mask:str)-> "StanceClassification":
        """replaces the entity/actor within the entire prompt with a placeholder name like "Organisation X"
           Serves as a check for an actor bias

        Args:
            entity_mask (str): a string that will replace the original/true detected actor
        """
        self.masked_input_text = re.sub(self.entity, entity_mask, self.input_text)
        self.masked_entity = entity_mask
        return(self)

    def summarize_irrelevant_stance_chain(self:"StanceClassification", lm, chat:bool,lm2=None, log=True) -> "StanceClassification":
        """prompt chain that:
           1. summarises text (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object)
           2. classifies whether the detected actor has a stance in the summary related to the statement, or not (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           3. if actor has a related stance: classify stance as opposition or support, not related stance: stance=irrelevant (saves stance as meta data and class attribute)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text, the irrelevance prompt text, and the stance prompt text)
        """
        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_prompt(input_text=self.masked_input_text,
                                                  entity=self.masked_entity)
        if chat:
            with user():
                summary = lm + summary_prompt
            with assistant():
                summary += gen(name="summary", max_tokens=120)
        if not chat: 
                summary = lm + summary_prompt + gen(name="summary", max_tokens=80)
        if log:
            logger.info(f"Basing classification on position summary: {summary['summary']}")
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(input_text=summary['summary'],
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
        if chat:
            with user():
                irrelevance = lm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if not chat:
            irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(input_text=summary['summary'],
                                                    entity=self.masked_entity,
                                                    statement=self.statement)
            if chat:
                with user():
                    stance = lm + stance_prompt
                with assistant():
                    stance = stance + select(['Ja','Nein'], name = 'answer')
            if not chat:
                stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            "irrelevance":irrelevance,
            "stance":stance
            }
        }
        return(self)
    
    def summarize_v2_irrelevant_stance_chain(self:"StanceClassification", lm, chat:bool, lm2=None, log=True) -> "StanceClassification":
        """prompt chain that:
           1. summarises text in relation to the statement (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object)
           2. classifies whether the detected actor has a stance in the summary related to the statement, or not (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           3. if actor has a related stance: classify stance as opposition or support, if no related stance: stance=irrelevant (saves stance as meta data and class attribute)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text, the irrelevance prompt text, and the stance prompt text)
        """

        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_statementspecific_prompt(input_text=self.masked_input_text,
                                                                    entity=self.masked_entity,
                                                                    statement=self.statement)
        if chat:
            with user():
                summary = lm + summary_prompt
            with assistant():
                summary += gen(name="summary", max_tokens=120)
        if not chat: 
                summary = lm + summary_prompt + gen(name="summary", max_tokens=80)
        if log:
            logger.info(f"Basing classification on position summary: {summary['summary']}")
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(input_text=summary['summary'],
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
        if chat:
            with user():
                irrelevance = lm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if not chat:
            irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(input_text=summary['summary'],
                                                    entity=self.masked_entity,
                                                    statement=self.statement)
            if chat:
                with user():
                    stance = lm + stance_prompt
                with assistant():
                    stance = stance + select(['Ja','Nein'], name = 'answer')
            if not chat:
                stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            "irrelevance":irrelevance,
            "stance":stance
            }
        }
        return(self)
    
    def summarize_v2_chain(self:"StanceClassification", lm, chat:bool, lm2=None, log=True)-> "StanceClassification":
        """prompt chain that:
           1. summarises text in relation to the statement (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object)
           2. prompts llm directly to classify the detected actor's stance based on the summary, stance class labels to select from: irrelevant, opposition, support

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text)
        """

        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_statementspecific_prompt(input_text=self.masked_input_text,
                                                                    entity=self.masked_entity,
                                                                    statement=self.statement)
        if chat:
            with user():
                summary = lm + summary_prompt
            with assistant():
                summary += f"Die Organisation {self.masked_entity} " + select(["drückt keine Haltung aus dazu, dass",
                                                                        "unterstützt, dass",
                                                                        "lehnt ab, dass"], name="stance") + gen(name="summary", max_tokens=80)
        if not chat: 
                summary = lm + summary_prompt + f"Die Organisation {self.masked_entity} " + select(["drückt keine Haltung aus dazu, dass",
                                                            "unterstützt, dass",
                                                            "lehnt ab, dass"], 
                                                            name="stance") + gen(name="summary", max_tokens=80)
        if log:
            logger.info(f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}")
        if summary["stance"] == "drückt keine Haltung aus dazu, dass":
            self.stance = "irrelevant"
        if summary["stance"] == "unterstützt, dass":
            self.stance = "support"
        if summary["stance"] == "lehnt ab, dass":
            self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            }
        }
        return(self)
    
    def irrelevant_summarize_v2_chain(self, lm, chat, lm2=None, log=True):
        """prompt chain that:
           1. classifies whether the detected actor has a stance in the text related to the statement, or not (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           2. if actor has a related stance: continue with 3., if not: stance=irrelevance (saved as a new class attribute called stance)
           3. summarises text in relation to the statement and prompts in the same prompt text/step for the stance classification for either opposition or support (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object and the classified stance as a new class attribute called self.stance)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text, and the irrelevance prompt text)
        """
        if lm2 is None:
            lm2 = lm
        if log:
            logger.info(f"Summarizing position of {self.entity}")
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
        if chat:
            with user():
                irrelevance = lm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if not chat:
            irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            summary = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            summary_prompt = construct_summary_statementspecific_prompt(input_text=self.masked_input_text,
                                                                        entity=self.masked_entity,
                                                                        statement=self.statement)
            if chat:
                with user():
                    summary = lm2 + summary_prompt
                with assistant():
                    summary += f"Die Organisation {self.masked_entity} " + select(["drückt keine Haltung aus dazu, dass",
                                                                            "unterstützt, dass",
                                                                            "lehnt ab, dass"], name="stance") + gen(name="summary", 
                                                                                                                    max_tokens=80)
            if not chat: 
                summary = lm2 + summary_prompt + f"Die Organisation {self.masked_entity} " + select(["drückt keine Haltung aus dazu, dass",
                                                                                             "unterstützt, dass",
                                                                                             "lehnt ab, dass"],
                                                                                             name="stance") + gen(name="summary", 
                                                                                                                  max_tokens=80)
            if log:
                logger.info(f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}")
            if summary["stance"] == "drückt keine Haltung aus dazu, dass":
                self.stance = "irrelevant"
            if summary["stance"] == "unterstützt, dass":
                self.stance = "support"
            if summary["stance"] == "lehnt ab, dass":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            "irrelevance":irrelevance,
            }
        }
        return(self)
    
    def irrelevant_stance_chain(self:"StanceClassification", lm, chat:bool, lm2=None, log=True) -> "StanceClassification":
        """prompt chain that:
           1. classifies whether the detected actor has a stance in the text related to the statement, or not (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           2. if actor has a related stance: classify stance as support or not support, if no related stance: stance=irrelevant (saves stance as meta data and class attribute)
           3. if the stance is not support: the stance=opposition (saves stance as meta data and class attribute)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the irrelevance prompt text and the stance prompt text)
        """
        
        if log:
            logger.info(f"Analyzing position of {self.entity} regarding statement {self.statement}")
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
        if chat:
            with user():
                irrelevance = lm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if not chat:
            irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(input_text=self.masked_input_text,
                                                    entity=self.masked_entity,
                                                    statement=self.statement)
            if chat:
                with user():
                    stance = lm + stance_prompt
                with assistant():
                    stance = stance + select(['Ja','Nein'], name = 'answer')
            if not chat:
                stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "irrelevance":irrelevance,
            "stance":stance
            }
        }
        return(self)
    
    def nested_irrelevant_summary_explicit(self:"StanceClassification", lm, chat:bool, lm2=None, log=True)-> "StanceClassification":
        """prompt chain that:
           1. checks if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
           2. checks whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           3. summarises text (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object)
           4. prompts llm explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support (saves stance as meta data and class attribute)
           5. prompts llm explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition (saves stance as meta data and class attribute)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text, the irrelevance prompt text, and the stance prompt text)
        """
        if log:
            logger.info(f"Analyzing if {self.entity} has position")
            logger.info("Checking potential stance...")
        general_prompt = construct_general_stance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity)
        if chat:
            with user():
                irrelevance = lm + general_prompt
            with assistant():
                irrelevance = irrelevance + select(["Ja", "Nein"], name='answer')
        if not chat:
            irrelevance = lm + general_prompt + select(["Ja", "Nein"], name='answer')
        if irrelevance["answer"] == "Nein":
            self.stance = "irrelevant"
            stance = None
            summary = None
        if irrelevance["answer"] == "Ja":
            if log:
                logger.info(f"Analyzing if {self.entity} supports statement {self.statement}")
                logger.info("Checking irrelevance...")
            irrelevance_prompt = construct_irrelevance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
            if chat:
                with user():
                    irrelevance = lm + irrelevance_prompt
                with assistant():
                    irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
            if not chat:
                irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
                self.stance = "irrelevant"
                stance = None
                summary = None
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
                if log:
                    logger.info(f"Summarizing position of {self.entity}")
                summary_prompt = construct_summary_prompt(input_text=self.masked_input_text,
                                                        entity=self.masked_entity)
                if chat:
                    with user():
                        summary = lm + summary_prompt
                    with assistant():
                        summary += gen(name="summary", max_tokens=120)
                if not chat: 
                        summary = lm + summary_prompt + gen(name="summary", max_tokens=80)
                if log:
                    logger.info(f"Basing classification on position summary: {summary['summary']}")
                    logger.info("Checking irrelevance...")

                stance_prompt = construct_support_stance_prompt(input_text=summary['summary'],
                                                        entity=self.masked_entity,
                                                        statement=self.statement)
                if chat:
                    with user():
                        stance = lm + stance_prompt
                    with assistant():
                        stance = stance + select(['Ja','Nein'], name = 'answer')
                if not chat:
                    stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
                if stance["answer"] == "Ja":
                    self.stance = "support"
                if stance["answer"] == "Nein":
                    stance_prompt = construct_opposition_stance_prompt(input_text=summary['summary'],
                                                            entity=self.masked_entity,
                                                            statement=self.statement)
                    if chat:
                        with user():
                            stance = lm + stance_prompt
                        with assistant():
                            stance = stance + select(['Ja','Nein'], name = 'answer')
                    if not chat:
                        stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
                    if stance["answer"] == "Ja":
                        self.stance = "opposition"
                    if stance["answer"] == "Nein":
                        self.stance = "irrelevant"  
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            "irrelevance":irrelevance,
            "stance":stance
            }
        }
        return(self)
    
    def nested_irrelevant_summary_v2_explicit(self:"StanceClassification", lm, chat:bool, lm2=None, log=True)-> "StanceClassification":
        """prompt chain that:
           1. checks if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
           2. checks whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant (creates and stores the entire prompt text sent to the lm in the parameter irrelevance, returned as meta data in the class object)
           3. summarises text in relation to the statement (creates and stores the entire prompt text sent to the lm in the parameter summary, returned as meta data in the class object)
           4. prompts llm explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support (saves stance as meta data and class attribute)
           5. prompts llm explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition (saves stance as meta data and class attribute)

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            lm: A guidance model backend from guidance.models
            chat (bool): whether lm it is a chat LLM or not
            lm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. (meta is a dictionary containing the summary prompt text, the irrelevance prompt text, and the stance prompt text)
        """
        if log:
            logger.info(f"Analyzing if {self.entity} has position")
            logger.info("Checking potential stance...")
        general_prompt = construct_general_stance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity)
        if chat:
            with user():
                irrelevance = lm + general_prompt
            with assistant():
                irrelevance = irrelevance + select(["Ja", "Nein"], name='answer')
        if not chat:
            irrelevance = lm + general_prompt + select(["Ja", "Nein"], name='answer')
        if irrelevance["answer"] == "Nein":
            self.stance = "irrelevant"
            stance = None
            summary = None
        if irrelevance["answer"] == "Ja":
            if log:
                logger.info(f"Analyzing if {self.entity} supports statement {self.statement}")
                logger.info("Checking irrelevance...")
            irrelevance_prompt = construct_irrelevance_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity,
                                                          statement=self.statement)
            if chat:
                with user():
                    irrelevance = lm + irrelevance_prompt
                with assistant():
                    irrelevance = irrelevance + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
            if not chat:
                irrelevance = lm + irrelevance_prompt + select(list(IRRELEVANCE_ANSWERS.values()), name='answer')
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
                self.stance = "irrelevant"
                stance = None
                summary = None
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
                if log:
                    logger.info(f"Summarizing position of {self.entity}")
                summary_prompt = construct_summary_statementspecific_prompt(input_text=self.masked_input_text,
                                                                            entity=self.masked_entity,
                                                                            statement=self.statement)
                if chat:
                    with user():
                        summary = lm + summary_prompt
                    with assistant():
                        summary += gen(name="summary", max_tokens=120)
                if not chat: 
                        summary = lm + summary_prompt + gen(name="summary", max_tokens=80)
                if log:
                    logger.info(f"Basing classification on position summary: {summary['summary']}")
                    logger.info("Checking irrelevance...")
                stance_prompt = construct_support_stance_prompt(input_text=summary['summary'],
                                                        entity=self.masked_entity,
                                                        statement=self.statement)
                if chat:
                    with user():
                        stance = lm + stance_prompt
                    with assistant():
                        stance = stance + select(['Ja','Nein'], name = 'answer')
                if not chat:
                    stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
                if stance["answer"] == "Ja":
                    self.stance = "support"
                if stance["answer"] == "Nein":
                    stance_prompt = construct_opposition_stance_prompt(input_text=summary['summary'],
                                                            entity=self.masked_entity,
                                                            statement=self.statement)
                    if chat:
                        with user():
                            stance = lm + stance_prompt
                        with assistant():
                            stance = stance + select(['Ja','Nein'], name = 'answer')
                    if not chat:
                        stance = lm + stance_prompt + select(['Ja','Nein'], name = 'answer')
                    if stance["answer"] == "Ja":
                        self.stance = "opposition"
                    if stance["answer"] == "Nein":
                        self.stance = "irrelevant"  
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "lms": {
            "summary":summary,
            "irrelevance":irrelevance,
            "stance":stance
            }
        }
        return(self)