from loguru import logger
from typing import Self

from guidance import gen, select, user, assistant

REGISTERED_LLM_CHAINS = {
    "sis": "summarize_irrelevant_stance_chain",
    "s2is": "summarize_v2_irrelevant_stance_chain",
    "s2": "summarize_v2_chain",
    "is": "irrelevant_stance_chain",
    "is2": "irrelevant_summarize_v2_chain",
    "nise": "nested_irrelevant_summary_explicit",
    "nis2e": "nested_irrelevant_summary_v2_explicit",
}

ALLOWED_DUAL_LLM_CHAINS = ["is2"]


IRRELEVANCE_ANSWERS = {
    "de": {
        "irrelevant": "Bezieht keine Stellung",
        "stance": "Bezieht Stellung",
    },
    "en": {
        "irrelevant": "Does not take a stance",
        "stance": "Takes a stance",
    },
}
IRRELEVANCE_ANSWERS2 = {
    "de": {
        "irrelevant": "Bezieht keine Stellung",
        "stance": "Bezieht Stellung",
    },
    "en": {
        "irrelevant": "Does not express a position",
        "stance": "Expresses a position",
    },
}

ALLOWED_STANCE_CATEGORIES = ["support", "opposition", "irrelevant", "error"]

YES_NO_ANSWERS = {"de": ["Ja", "Nein"], "en": ["Yes", "No"]}

SUMMARY_STANCE_OPTIONS = {
    "en": ["does not express a position that", "supports that", "opposes that"],
    "de": ["drückt keine Haltung aus dazu, dass", "unterstützt, dass", "lehnt ab, dass"],
}

SUMMARY_STANCE_PREFIX = {
    "en": "The organization {entity} ",
    "de": "Die Organisation {entity} ",
}


def construct_irrelevance_prompt(input_text, entity, statement, language="de"):
    if language == "en":
        prompt = (
            f"Analyze the following text: {input_text}. "
            f"Does the organization {entity} take a stance on the following statement: {statement}? "
            f"Refer only to the text. Answer with {IRRELEVANCE_ANSWERS['en']['irrelevant']} or {IRRELEVANCE_ANSWERS['en']['stance']}"
        )
    else:
        prompt = (
            f"Analysiere den folgenden Text: {input_text}. "
            f"Bezieht die Organisation {entity} Stellung zur folgenden Aussage: {statement}? "
            f"Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS['de']['irrelevant']} oder {IRRELEVANCE_ANSWERS['de']['stance']}"
        )
    return prompt


def construct_summary_prompt(input_text, entity, language="de"):
    if language == "en":
        prompt = (
            f"Summarize the position of the organization {entity} in the following text:\n {input_text}. "
            f"Be brief and start your summary with: The organization {entity}..."
        )
    else:
        prompt = (
            f"Fasse die Position der Organisation {entity} im folgenden Text zusammen: \n {input_text}. "
            f"Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}..."
        )
    return prompt


def construct_summary_statementspecific_prompt(input_text, entity, statement, language="de"):
    if language == "en":
        prompt = (
            f'Summarize the position of the organization {entity} in the following text with respect to the statement "{statement}":\n {input_text}.\n '
            f"Be brief and start your summary with: The organization {entity}..."
        )
    else:
        prompt = (
            f'Fasse die Position der Organisation {entity} im folgenden Text in Bezug auf die Aussage "{statement}" zusammen: \n {input_text}. \n '
            f"Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}..."
        )
    return prompt


def construct_general_stance_prompt(input_text, entity, language="de"):
    if language == "en":
        prompt = (
            f"Analyze the following text: {input_text}. "
            f"Does the organization {entity} express an implicit or explicit position regarding an issue? "
            f"Refer only to the text. Answer with {IRRELEVANCE_ANSWERS2['en']['irrelevant']} or {IRRELEVANCE_ANSWERS2['en']['stance']}"
        )
    else:
        prompt = (
            f"Analysiere den folgenden Text: {input_text}. "
            f"Bezieht die Organisation {entity} implizit oder explizit Stellung zu einem Sachverhalt? "
            f"Beziehe dich nur auf den Text. Antworte mit {IRRELEVANCE_ANSWERS2['de']['irrelevant']} oder {IRRELEVANCE_ANSWERS2['de']['stance']}"
        )
    return prompt


def construct_support_stance_prompt(input_text, entity, statement, language="de"):
    if language == "en":
        prompt = (
            f"Analyze the following text: {input_text}. "
            f"Does the organization {entity} support the statement: {statement}? "
            f"Refer only to the text. Answer with Yes or No"
        )
    else:
        prompt = (
            f"Analysiere den folgenden Text: {input_text}. "
            f"Befürwortet die Organisation {entity} die Aussage: {statement}? "
            f"Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
        )
    return prompt


def construct_opposition_stance_prompt(input_text, entity, statement, language="de"):
    if language == "en":
        prompt = (
            f"Analyze the following text: {input_text}. "
            f"Does the organization {entity} oppose the statement: {statement}? "
            f"Refer only to the text. Answer with Yes or No"
        )
    else:
        prompt = (
            f"Analysiere den folgenden Text: {input_text}. "
            f"Lehnt die Organisation {entity} die folgende Aussage ab: {statement}? "
            f"Beziehe dich nur auf den Text. Antworte mit Ja oder Nein"
        )
    return prompt


def _run_turn(llm, chat: bool, prompt: str, continuation):
    """Send `prompt` as a turn to `llm` and append `continuation` (a guidance
    grammar object, e.g. select(...)/gen(...)/a sum of both) to the response,
    honoring chat vs. plain-completion mode."""
    if chat:
        with user():
            state = llm + prompt
        with assistant():
            state = state + continuation
    else:
        state = llm + prompt + continuation
    return state


def get_registered_chains():
    return REGISTERED_LLM_CHAINS

def get_registered_chains_keys():
    return [key for key in REGISTERED_LLM_CHAINS.keys()]

def get_allowed_dual_llm_chains():
    return ALLOWED_DUAL_LLM_CHAINS


class StanceClassification:
    """Class used for LLM-based classifications of stances by a specified entity in a text regarding a statement.

    Attributes:
        entity (str): entity to classify stance of
        statement (str): statement to classify stance toward
        input_text (str): text to classify stance of entity in

    Methods:
        mask_entity: replace the entity string with a placeholder in all prompts
        summarize_irrelevant_stance_chain, summarize_v2_irrelevant_stance_chain,
        summarize_v2_chain, irrelevant_summarize_v2_chain, irrelevant_stance_chain,
        nested_irrelevant_summary_explicit, nested_irrelevant_summary_v2_explicit:
        prompt chains classifying the stance (see get_registered_chains)
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
        self.masked_input_text = self.input_text.replace(self.entity, entity_mask)
        self.masked_entity = entity_mask
        return self

    def _run_irrelevance(self, llm, chat: bool, input_text, language):
        """Run the irrelevance check (statement-related stance or not) on `input_text`."""
        prompt = construct_irrelevance_prompt(
            input_text=input_text,
            entity=self.masked_entity,
            statement=self.statement,
            language=language,
        )
        return _run_turn(
            llm, chat, prompt,
            select(list(IRRELEVANCE_ANSWERS[language].values()), name="answer"),
        )

    def _run_yes_no(self, llm, chat: bool, prompt: str, language):
        """Run a Ja/Nein (Yes/No) constrained answer for `prompt`."""
        return _run_turn(
            llm, chat, prompt,
            select(YES_NO_ANSWERS[language], name="answer"),
        )

    def _run_summary(self, llm, chat: bool, language, statement_specific: bool):
        """Free-text position summary of the (masked) input text."""
        if statement_specific:
            prompt = construct_summary_statementspecific_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                statement=self.statement,
                language=language,
            )
        else:
            prompt = construct_summary_prompt(
                input_text=self.masked_input_text,
                entity=self.masked_entity,
                language=language,
            )
        return _run_turn(
            llm, chat, prompt,
            gen(name="summary", max_tokens=120 if chat else 80),
        )

    def _run_summary_stance_select(self, llm, chat: bool, language):
        """Statement-specific summary that selects the stance inline via a fixed
        set of sentence openers, then generates the rest of the summary."""
        summary_prompt = construct_summary_statementspecific_prompt(
            input_text=self.masked_input_text,
            entity=self.masked_entity,
            statement=self.statement,
            language=language,
        )
        prefix = SUMMARY_STANCE_PREFIX[language].format(entity=self.masked_entity)
        return _run_turn(
            llm, chat, summary_prompt,
            prefix
            + select(SUMMARY_STANCE_OPTIONS[language], name="stance")
            + gen(name="summary", max_tokens=80),
        )

    def _stance_from_summary_select(self, stance_answer, language):
        """Map a SUMMARY_STANCE_OPTIONS selection to a stance category."""
        return ("irrelevant", "support", "opposition")[
            SUMMARY_STANCE_OPTIONS[language].index(stance_answer)
        ]

    def _resolve_support_opposition(self, llm, chat: bool, input_text, language, nested: bool):
        """Classify support vs. opposition on `input_text`, setting self.stance.

        Non-nested: a Yes -> support, a No -> opposition.
        Nested: a Yes -> support; otherwise a second opposition question decides
        opposition (Yes) vs. irrelevant (No). Returns the last stance state.
        """
        support_prompt = construct_support_stance_prompt(
            input_text=input_text,
            entity=self.masked_entity,
            statement=self.statement,
            language=language,
        )
        stance = self._run_yes_no(llm, chat, support_prompt, language)
        if stance["answer"] in ["Ja", "Yes"]:
            self.stance = "support"
            return stance
        if not nested:
            self.stance = "opposition"
            return stance
        opposition_prompt = construct_opposition_stance_prompt(
            input_text=input_text,
            entity=self.masked_entity,
            statement=self.statement,
            language=language,
        )
        stance = self._run_yes_no(llm, chat, opposition_prompt, language)
        if stance["answer"] in ["Ja", "Yes"]:
            self.stance = "opposition"
        if stance["answer"] in ["Nein", "No"]:
            self.stance = "irrelevant"
        return stance

    def _summarize_then_irrelevant_stance(
        self, llm, chat: bool, log: bool, language, statement_specific: bool
    ) -> Self:
        """Shared body of the sis/s2is chains: summarize, check irrelevance on the
        summary, then classify support/opposition when a stance is present."""
        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary = self._run_summary(llm, chat, language, statement_specific=statement_specific)
        if log:
            logger.info(
                f"Basing classification on position summary: {summary['summary']}"
            )
            logger.info("Checking irrelevance...")
        irrelevance = self._run_irrelevance(llm, chat, summary["summary"], language)
        stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["irrelevant"]:
            self.stance = "irrelevant"
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["stance"]:
            stance = self._resolve_support_opposition(
                llm, chat, summary["summary"], language, nested=False
            )
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {"summary": summary, "irrelevance": irrelevance, "stance": stance}
        }
        return self

    def summarize_irrelevant_stance_chain(
        self, llm, chat: bool, llm2=None, log=True, language="de"
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
            language (str): "de" or "en"

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned: e.g. meta["llms"]["irrelevance"].
        """
        return self._summarize_then_irrelevant_stance(
            llm, chat, log, language, statement_specific=False
        )

    def summarize_v2_irrelevant_stance_chain(
        self, llm, chat: bool, llm2=None, log=True, language="de"
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
            language (str): "de" or "en"

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance, summary, and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification class object returned: e.g. meta["llms"]["irrelevance"].
        """
        return self._summarize_then_irrelevant_stance(
            llm, chat, log, language, statement_specific=True
        )

    def summarize_v2_chain(self, llm, chat: bool, llm2=None, log=True, language="de") -> Self:
        """prompt chain that:
           1. summarises text in relation to the statement (stored in the "meta" attribute of the StanceClassification class object in a dictionary value at the key ["llms"]["summary"])
           2. prompts llm directly to classify the detected actor's stance based on the summary, stance class labels to select from: irrelevant, opposition, support

        Args:
            self: StanceClassification class object, contains: entity, statement, input_text, stance
            llm: A guidance model backend from guidance.models
            chat (bool): whether llm is a chat llm or not
            llm2 (optional): A second guidance model backend from guidance.models. Defaults to None.
            log (bool, optional): To log or not. Defaults to True.
            language (str): "de" or "en"

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The summary prompt text is stored in a dictionary value at the key ["llms"]["summary"] in a dictionary stored in the "meta" attribute of the StanceClassification class object returned.
        """
        if log:
            logger.info(f"Summarizing position of {self.entity}")
        summary = self._run_summary_stance_select(llm, chat, language)
        if log:
            logger.info(
                f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}"
            )
        self.stance = self._stance_from_summary_select(summary["stance"], language)
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "summary": summary,
            }
        }
        return self

    def irrelevant_summarize_v2_chain(self, llm, chat, llm2=None, log=True, language="de") -> Self:
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
            language (str): "de" or "en"

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance and summary prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification class object returned: e.g. meta["llms"]["irrelevance"].
        """
        if llm2 is None:
            llm2 = llm
        if log:
            logger.info(f"Summarizing position of {self.entity}")
            logger.info("Checking irrelevance...")
        irrelevance = self._run_irrelevance(llm, chat, self.masked_input_text, language)
        summary = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["irrelevant"]:
            self.stance = "irrelevant"
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["stance"]:
            summary = self._run_summary_stance_select(llm2, chat, language)
            if log:
                logger.info(
                    f"Basing classification on position summary: {self.entity} {summary['stance']} {summary['summary']}"
                )
            self.stance = self._stance_from_summary_select(summary["stance"], language)
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {
            "llms": {
                "summary": summary,
                "irrelevance": irrelevance,
            }
        }
        return self

    def irrelevant_stance_chain(self, llm, chat: bool, llm2=None, log=True, language="de") -> Self:
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
            language (str): "de" or "en"

        Returns:
            StanceClassification class object with new class object attributes: meta and stance. The irrelevance and stance prompt texts are stored in a dictionary value at the key ["llms"] in a dictionary stored in the "meta" attribute of the StanceClassification object returned, e.g. meta["llms"]["stance"]
        """
        if log:
            logger.info(
                f"Analyzing position of {self.entity} regarding statement {self.statement}"
            )
            logger.info("Checking irrelevance...")
        irrelevance = self._run_irrelevance(llm, chat, self.masked_input_text, language)
        stance = None
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["irrelevant"]:
            self.stance = "irrelevant"
        if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["stance"]:
            stance = self._resolve_support_opposition(
                llm, chat, self.masked_input_text, language, nested=False
            )
        if log:
            logger.info(f"classified as {self.stance}")
        self.meta = {"llms": {"irrelevance": irrelevance, "stance": stance}}
        return self

    def _nested_irrelevant_summary(
        self, llm, chat: bool, log: bool, language, statement_specific: bool
    ) -> Self:
        """Shared body of the nise/nis2e chains: a general-stance gate, a
        statement-relatedness gate, a summary, then nested support/opposition."""
        if log:
            logger.info(f"Analyzing if {self.entity} has position")
            logger.info("Checking potential stance...")
        general_prompt = construct_general_stance_prompt(
            input_text=self.masked_input_text, entity=self.masked_entity, language=language
        )
        irrelevance_general = _run_turn(
            llm, chat, general_prompt,
            select(list(IRRELEVANCE_ANSWERS2[language].values()), name="answer_general"),
        )
        irrelevance = None
        stance = None
        summary = None
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2[language]["irrelevant"]:
            self.stance = "irrelevant"
        if irrelevance_general["answer_general"] == IRRELEVANCE_ANSWERS2[language]["stance"]:
            if log:
                logger.info(
                    f"Analyzing if {self.entity} supports statement {self.statement}"
                )
                logger.info("Checking irrelevance...")
            irrelevance = self._run_irrelevance(llm, chat, self.masked_input_text, language)
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["irrelevant"]:
                self.stance = "irrelevant"
            if irrelevance["answer"] == IRRELEVANCE_ANSWERS[language]["stance"]:
                if log:
                    logger.info(f"Summarizing position of {self.entity}")
                summary = self._run_summary(
                    llm, chat, language, statement_specific=statement_specific
                )
                if log:
                    logger.info(
                        f"Basing classification on position summary: {summary['summary']}"
                    )
                    logger.info("Checking irrelevance...")
                stance = self._resolve_support_opposition(
                    llm, chat, summary["summary"], language, nested=True
                )
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

    def nested_irrelevant_summary_explicit(
        self, llm, chat: bool, llm2=None, log=True, language="de"
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
        return self._nested_irrelevant_summary(
            llm, chat, log, language, statement_specific=False
        )

    def nested_irrelevant_summary_v2_explicit(
        self, llm, chat: bool, llm2=None, log=True, language="de"
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
        return self._nested_irrelevant_summary(
            llm, chat, log, language, statement_specific=True
        )
