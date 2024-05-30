from loguru import logger
from typing_extensions import Self
import re

from guidance import gen, select, user, assistant

REGISTERED_LLM_CHAINS = {
    "sis": "summarize_irrelevant_stance",
    "s2is": "summarize_v2_irrelevant_stance",
    "s2": "summarize_v2",
    "is": "irrelevant_stance",
    "is2": "irrelevant_summarize_v2",
    "nise": "nested_irrelevant_summary_explicit",
    "nis2e": "nested_irrelevant_summary_v2_explicit",
}

ALLOWED_DUAL_LLM_CHAINS = ["is2"]

CONSTRAINED_GRAMMAR_CHAINS = ["s2", "is2"]

IRRELEVANCE_ANSWERS = {
    "irrelevant": "Bezieht keine Stellung",
    "stance": "Bezieht Stellung",
}
IRRELEVANCE_ANSWERS2 = {
    "irrelevant": "Bezieht keine Haltung",
    "stance": "Bezieht eine Haltung",
}


def construct_irrelevance_prompt(input_text, entity, statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Bezieht die Organisation {entity} Stellung zur folgenden Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS['irrelevant']} oder {IRRELEVANCE_ANSWERS['stance']}"
    return prompt


def construct_summary_prompt(input_text, entity):
    prompt = f"Fasse die Position der Organisation {entity} im folgenden Text zusammen: \n {input_text}. Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}..."
    return prompt


def construct_summary_statementspecific_prompt(input_text, entity, statement):
    prompt = f'Fasse die Position der Organisation {entity} im folgenden Text in Bezug auf die Aussage "{statement}" zusammen: \n {input_text}. \n Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}...'
    return prompt


def construct_general_stance_prompt(input_text, entity):
    prompt = f"Analysiere den folgenden Text: {input_text}. Äussert die Organisation {entity} eine implizite oder explizite Haltung? Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS2['irrelevant']} oder {IRRELEVANCE_ANSWERS2['stance']}"
    return prompt


def construct_support_stance_prompt(input_text, entity, statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Befürwortet die Organisation {entity} die Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
    return prompt


def construct_opposition_stance_prompt(input_text, entity, statement):
    prompt = f"Analysiere den folgenden Text: {input_text}. Lehnt die Organisation {entity} folgende die Aussage ab: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
    return prompt


def get_registered_chains():
    return REGISTERED_LLM_CHAINS


def get_allowed_dual_llm_chains():
    return ALLOWED_DUAL_LLM_CHAINS


class StanceClassification:
    """Class used for LLM-based classifications of stances by a specified entity in a text regarding a statement.

    Attributes:
        entity (str): entity to classify stance of
        statement (str): statement to classify stance toward
        input_text (str): text to classify stance of entity in

    Methods:
        # TODO
    """

    def __init__(self, input_text, statement, entity):
        self.input_text = input_text
        self.statement = statement
        self.entity = entity
        self.stance = None
        self.meta = None
        self.masked_entity = entity
        self.masked_input_text = input_text

    def __str__(self):
        return "The stance of entity {} towards the statement {} given text {} is {}".format(
            self.entity, self.statement, self.input_text, self.stance
        )

    def mask_entity(self, entity_mask: str) -> Self:
        """replaces the entity/actor within the entire prompt with a placeholder name like "Organisation X"
           Serves as a check for an actor bias

        Args:
            entity_mask (str): a string that will mask the original entity
        """
        self.masked_input_text = re.sub(self.entity, entity_mask, self.input_text)
        self.masked_entity = entity_mask
        return self

    def summarize_irrelevant_stance_chain(
        self, llm, chat: bool, llm2=None, log=True
    ) -> Self:
        """prompt chain that:
           1. summarises text (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           2. classifies whether the detected actor has a stance in the summary related to the statement, or not (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance"])
           3. if actor has a related stance: classify stance as opposition or support, not related stance: stance=irrelevant (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned: e.g. meta["llms"]["irrelevance"].
        """
        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_prompt(
            input_text=self.masked_input_text, entity=self.masked_entity
        )
        if chat:
            with user():
                summary = llm + summary_prompt
            with assistant():
                summary += gen(name="summary", max_tokens=120)
        if not chat:
            summary = llm + summary_prompt + gen(name="summary", max_tokens=80)
        if log:
            logger.info(
                f"Basing classification on position summary: {summary['summary']}"
            )
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(
            input_text=summary["summary"],
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                irrelevance = llm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(
                    list(IRRELEVANCE_ANSWERS.values()), name="answer"
                )
        if not chat:
            irrelevance = (
                llm
                + irrelevance_prompt
                + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
            )
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(
                input_text=summary["summary"],
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    stance = llm + stance_prompt
                with assistant():
                    stance = stance + select(["Ja", "Nein"], name="answer")
            if not chat:
                stance = llm + stance_prompt + select(["Ja", "Nein"], name="answer")
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {"summary": summary, "irrelevance": irrelevance, "stance": stance}
        }
        return self

    def summarize_v2_irrelevant_stance_chain(
        self, llm, chat: bool, llm2=None, log=True
    ) -> Self:
        """prompt chain that:
           1. summarises text in relation to the statement (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           2. classifies whether the detected actor has a stance in the summary related to the statement, or not (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance"])
           3. if actor has a related stance: classify stance as opposition or support, if no related stance: stance=irrelevant (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned: e.g. meta["llms"]["irrelevance"].
        """

        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_statementspecific_prompt(
            input_text=self.masked_input_text,
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                summary = llm + summary_prompt
            with assistant():
                summary += gen(name="summary", max_tokens=120)
        if not chat:
            summary = llm + summary_prompt + gen(name="summary", max_tokens=80)
        if log:
            logger.info(
                f"Basing classification on position summary: {summary['summary']}"
            )
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(
            input_text=summary["summary"],
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                irrelevance = llm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(
                    list(IRRELEVANCE_ANSWERS.values()), name="answer"
                )
        if not chat:
            irrelevance = (
                llm
                + irrelevance_prompt
                + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
            )
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(
                input_text=summary["summary"],
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    stance = llm + stance_prompt
                with assistant():
                    stance = stance + select(["Ja", "Nein"], name="answer")
            if not chat:
                stance = llm + stance_prompt + select(["Ja", "Nein"], name="answer")
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {"summary": summary, "irrelevance": irrelevance, "stance": stance}
        }
        return self

    def summarize_v2_chain(self, llm, chat: bool, llm2=None, log=True) -> Self:
        """prompt chain that:
           1. summarises text in relation to the statement (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           2. prompts llm directly to classify the detected actor's stance based on the summary, stance class labels to select from: irrelevant, opposition, support

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The summary prompt text is stored in a dictionary value at the key ["llms"]["summary"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned.
        """

        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary_prompt = construct_summary_statementspecific_prompt(
            input_text=self.masked_input_text,
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                summary = llm + summary_prompt
            with assistant():
                summary += (
                    f"Die Organisation {self.masked_entity} "
                    + select(
                        [
                            "drückt keine Haltung aus dazu, dass",
                            "unterstützt, dass",
                            "lehnt ab, dass",
                        ],
                        name="stance",
                    )
                    + gen(name="summary", max_tokens=80)
                )
        if not chat:
            summary = (
                llm
                + summary_prompt
                + f"Die Organisation {self.masked_entity} "
                + select(
                    [
                        "drückt keine Haltung aus dazu, dass",
                        "unterstützt, dass",
                        "lehnt ab, dass",
                    ],
                    name="stance",
                )
                + gen(name="summary", max_tokens=80)
            )
        if log:
            logger.info(
                f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}"
            )
        if summary["stance"] == "drückt keine Haltung aus dazu, dass":
            self.stance = "irrelevant"
        if summary["stance"] == "unterstützt, dass":
            self.stance = "support"
        if summary["stance"] == "lehnt ab, dass":
            self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "summary": summary,
            }
        }
        return self

    def irrelevant_summarize_v2_chain(self, llm, chat, llm2=None, log=True) -> Self:
        """prompt chain that:
           1. classifies whether the detected actor has a stance in the text related to the statement, or not (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance"])
           2. if actor has a related stance: continue with 3., if not: stance=irrelevance (saved as a new class attribute called stance)
           3. summarises text in relation to the statement and prompts in the same prompt text/step for the stance classification for either opposition or support

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Generates the summary and classifies the stance. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance and summary prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned: e.g. meta["llms"]["irrelevance"].
        """
        if llm2 is None:
            llm2 = llm
        if log:
            logger.info(f"Summarizing position of {self.entity}")
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(
            input_text=self.masked_input_text,
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                irrelevance = llm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(
                    list(IRRELEVANCE_ANSWERS.values()), name="answer"
                )
        if not chat:
            irrelevance = (
                llm
                + irrelevance_prompt
                + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
            )
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            summary = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            summary_prompt = construct_summary_statementspecific_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    summary = llm2 + summary_prompt
                with assistant():
                    summary += (
                        f"Die Organisation {self.masked_entity} "
                        + select(
                            [
                                "drückt keine Haltung aus dazu, dass",
                                "unterstützt, dass",
                                "lehnt ab, dass",
                            ],
                            name="stance",
                        )
                        + gen(name="summary", max_tokens=80)
                    )
            if not chat:
                summary = (
                    llm2
                    + summary_prompt
                    + f"Die Organisation {self.masked_entity} "
                    + select(
                        [
                            "drückt keine Haltung aus dazu, dass",
                            "unterstützt, dass",
                            "lehnt ab, dass",
                        ],
                        name="stance",
                    )
                    + gen(name="summary", max_tokens=80)
                )
            if log:
                logger.info(
                    f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}"
                )
            if summary["stance"] == "drückt keine Haltung aus dazu, dass":
                self.stance = "irrelevant"
            if summary["stance"] == "unterstützt, dass":
                self.stance = "support"
            if summary["stance"] == "lehnt ab, dass":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "summary": summary,
                "irrelevance": irrelevance,
            }
        }
        return self

    def irrelevant_stance_chain(self, llm, chat: bool, llm2=None, log=True) -> Self:
        """prompt chain that:
           1. classifies whether the detected actor has a stance in the text related to the statement, or not (the irrelevance prompt text is stored in a dictionary value at the key ["llms"]["irrelevance"] in the "meta" attribute)
           2. if actor has a related stance: classify stance as support or not support, if no related stance: stance=irrelevant (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")
           3. if the stance is not support: the stance=opposition (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned, e.g. meta["llms"]["stance"]
        """

        if log:
            logger.info(
                f"Analyzing position of {self.entity} regarding statement {self.statement}"
            )
            logger.info("Checking irrelevance...")
        irrelevance_prompt = construct_irrelevance_prompt(
            input_text=self.masked_input_text,
            entity=self.masked_entity,
            statement=self.statement,
        )
        if chat:
            with user():
                irrelevance = llm + irrelevance_prompt
            with assistant():
                irrelevance = irrelevance + select(
                    list(IRRELEVANCE_ANSWERS.values()), name="answer"
                )
        if not chat:
            irrelevance = (
                llm
                + irrelevance_prompt
                + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
            )
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
            stance_prompt = construct_support_stance_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    stance = llm + stance_prompt
                with assistant():
                    stance = stance + select(["Ja", "Nein"], name="answer")
            if not chat:
                stance = llm + stance_prompt + select(["Ja", "Nein"], name="answer")
            if stance["answer"] == "Ja":
                self.stance = "support"
            if stance["answer"] == "Nein":
                self.stance = "opposition"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {"llms": {"irrelevance": irrelevance, "stance": stance}}
        return self

    def nested_irrelevant_summary_explicit(
        self, llm, chat: bool, llm2=None, log=True
    ) -> Self:
        """prompt chain that:
           1. checks if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance_general"])
           2. checks whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance"])
           3. summarises text (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           4. prompts llm explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")
           5. prompts llm explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned: e.g. meta["llms"]["irrelevance"].
        """
        if log:
            logger.info(f"Analyzing if {self.entity} has position")
            logger.info("Checking potential stance...")
        general_prompt = construct_general_stance_prompt(
            input_text=self.masked_input_text, entity=self.masked_entity
        )
        if chat:
            with user():
                irrelevance_general = llm + general_prompt
            with assistant():
                irrelevance_general = irrelevance_general + select(
                    list(IRRELEVANCE_ANSWERS2.values()), name="answer_general"
                )
        if not chat:
            irrelevance_general = (
                llm
                + general_prompt
                + select(list(IRRELEVANCE_ANSWERS2.values()), name="answer_general")
            )
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
            summary = None
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2["stance"]:
            if log:
                logger.info(
                    f"Analyzing if {self.entity} supports statement {self.statement}"
                )
                logger.info("Checking irrelevance...")
            irrelevance_prompt = construct_irrelevance_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    irrelevance = llm + irrelevance_prompt
                with assistant():
                    irrelevance = irrelevance + select(
                        list(IRRELEVANCE_ANSWERS.values()), name="answer"
                    )
            if not chat:
                irrelevance = (
                    llm
                    + irrelevance_prompt
                    + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
                )
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
                self.stance = "irrelevant"
                stance = None
                summary = None
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
                if log:
                    logger.info(f"Summarizing position of {self.entity}")
                summary_prompt = construct_summary_prompt(
                    input_text=self.masked_input_text, entity=self.masked_entity
                )
                if chat:
                    with user():
                        summary = llm + summary_prompt
                    with assistant():
                        summary += gen(name="summary", max_tokens=120)
                if not chat:
                    summary = llm + summary_prompt + gen(name="summary", max_tokens=80)
                if log:
                    logger.info(
                        f"Basing classification on position summary: {summary['summary']}"
                    )
                    logger.info("Checking irrelevance...")

                stance_prompt = construct_support_stance_prompt(
                    input_text=summary["summary"],
                    entity=self.masked_entity,
                    statement=self.statement,
                )
                if chat:
                    with user():
                        stance = llm + stance_prompt
                    with assistant():
                        stance = stance + select(["Ja", "Nein"], name="answer")
                if not chat:
                    stance = llm + stance_prompt + select(["Ja", "Nein"], name="answer")
                if stance["answer"] == "Ja":
                    self.stance = "support"
                if stance["answer"] == "Nein":
                    stance_prompt = construct_opposition_stance_prompt(
                        input_text=summary["summary"],
                        entity=self.masked_entity,
                        statement=self.statement,
                    )
                    if chat:
                        with user():
                            stance = llm + stance_prompt
                        with assistant():
                            stance = stance + select(["Ja", "Nein"], name="answer")
                    if not chat:
                        stance = (
                            llm + stance_prompt + select(["Ja", "Nein"], name="answer")
                        )
                    if stance["answer"] == "Ja":
                        self.stance = "opposition"
                    if stance["answer"] == "Nein":
                        self.stance = "irrelevant"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "irrelevance_general": irrelevance_general,
                "irrelevance": irrelevance,
                "summary": summary,
                "stance": stance,
            }
        }
        return self

    def nested_irrelevant_summary_v2_explicit(
        self, llm, chat: bool, llm2=None, log=True
    ) -> Self:
        """prompt chain that:
           1. checks if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance_general"])
           2. checks whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["irrelevance"])
           3. summarises text in relation to the statement (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           4. prompts llm explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")
           5. prompts llm explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition (saves stance prompt in the "meta" attribute at the dictionary key ["llms"]["stance"] and the predicted stance separately in the class attribute "stance")

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm it is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in the "meta" attribute of the returned StanceClassification object: e.g. meta["llms"]["irrelevance"].
        """
        if log:
            logger.info(f"Analyzing if {self.entity} has position")
            logger.info("Checking potential stance...")
        general_prompt = construct_general_stance_prompt(
            input_text=self.masked_input_text, entity=self.masked_entity
        )
        if chat:
            with user():
                irrelevance_general = llm + general_prompt
            with assistant():
                irrelevance_general = irrelevance_general + select(
                    list(IRRELEVANCE_ANSWERS2.values()), name="answer_general"
                )
        if not chat:
            irrelevance_general = (
                llm
                + general_prompt
                + select(list(IRRELEVANCE_ANSWERS2.values()), name="answer_general")
            )
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2["irrelevant"]:
            self.stance = "irrelevant"
            stance = None
            summary = None
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2["stance"]:
            if log:
                logger.info(
                    f"Analyzing if {self.entity} supports statement {self.statement}"
                )
                logger.info("Checking irrelevance...")
            irrelevance_prompt = construct_irrelevance_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                statement=self.statement,
            )
            if chat:
                with user():
                    irrelevance = llm + irrelevance_prompt
                with assistant():
                    irrelevance = irrelevance + select(
                        list(IRRELEVANCE_ANSWERS.values()), name="answer"
                    )
            if not chat:
                irrelevance = (
                    llm
                    + irrelevance_prompt
                    + select(list(IRRELEVANCE_ANSWERS.values()), name="answer")
                )
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["irrelevant"]:
                self.stance = "irrelevant"
                stance = None
                summary = None
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS["stance"]:
                if log:
                    logger.info(f"Summarizing position of {self.entity}")
                summary_prompt = construct_summary_statementspecific_prompt(
                    input_text=self.masked_input_text,
                    entity=self.masked_entity,
                    statement=self.statement,
                )
                if chat:
                    with user():
                        summary = llm + summary_prompt
                    with assistant():
                        summary += gen(name="summary", max_tokens=120)
                if not chat:
                    summary = llm + summary_prompt + gen(name="summary", max_tokens=80)
                if log:
                    logger.info(
                        f"Basing classification on position summary: {summary['summary']}"
                    )
                    logger.info("Checking irrelevance...")
                stance_prompt = construct_support_stance_prompt(
                    input_text=summary["summary"],
                    entity=self.masked_entity,
                    statement=self.statement,
                )
                if chat:
                    with user():
                        stance = llm + stance_prompt
                    with assistant():
                        stance = stance + select(["Ja", "Nein"], name="answer")
                if not chat:
                    stance = llm + stance_prompt + select(["Ja", "Nein"], name="answer")
                if stance["answer"] == "Ja":
                    self.stance = "support"
                if stance["answer"] == "Nein":
                    stance_prompt = construct_opposition_stance_prompt(
                        input_text=summary["summary"],
                        entity=self.masked_entity,
                        statement=self.statement,
                    )
                    if chat:
                        with user():
                            stance = llm + stance_prompt
                        with assistant():
                            stance = stance + select(["Ja", "Nein"], name="answer")
                    if not chat:
                        stance = (
                            llm + stance_prompt + select(["Ja", "Nein"], name="answer")
                        )
                    if stance["answer"] == "Ja":
                        self.stance = "opposition"
                    if stance["answer"] == "Nein":
                        self.stance = "irrelevant"
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "irrelevance_general": irrelevance_general,
                "irrelevance": irrelevance,
                "summary": summary,
                "stance": stance,
            }
        }
        return self
