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
    "nis2e": "nested_irrelevant_summary_v2_explicit",
    "cb": "codebook",
    "nis2eq": "nested_irrelevant_summary_v2_explicit_questioning",
    "nis2em": "nested_irrelevant_summary_v2_explicit_masked"}

ALLOWED_DUAL_LM_CHAINS = ["is2"]

IRRELEVANCE_ANSWERS = {
    "irrelevant": 'Bezieht keine Stellung',
    "stance": 'Bezieht Stellung'
}
IRRELEVANCE_ANSWERS2 = {
    "irrelevant": 'Bezieht keine Haltung',
    "stance": 'Bezieht eine Haltung'
}

CODEBOOKS = {
    "cb1": """
        # Codebuch
        ## Ziel der Annotation
        Ziel der Annotation ist es, basierend auf einem Text die Haltung eines Akteurs gegenüber einer Aussage zu bestimmen.

        ## Mögliche Kategorien:
        support (Unterstützung): Der Akteur befürwortet die Aussage ausdrücklich oder allgemein.
        opposition (Ablehnung): Der Akteur lehnt die Aussage ab.
        irrelevant (keine Haltung): Der Akteur drückt im vorliegenden Text keine Haltung zur Aussage aus oder der Text hat nichts mit der Aussage zu tun.

        Beispiele:
        Beispiel 1:
        Aussage "Es sollten mehr Bäume gepflanzt werden"
        Akteur: "GGU"
        Text: "Die GGU lehnt das Pflanzen von Bäumen ab"
        Korrekte Antwort: "opposition"

        Beispiel 2:
        Aussage "Gurken essen sollte gefördert werden"
        Akteur: "Pflanzverein Neuhausen"
        Text: "Der Pflanzverein Neuhausen äussert sich verhalten positiv dazu, dass Gurkengewächse mehr verzehrt werden sollten."
        Korrekte Antwort: "support"

        Beispiel 3:
        Aussage "Tomaten essen sollte gefördert werden"
        Akteur: "Kartenspielerverband"
        Text: "Der Kartenspielerverband will ein Jassturnier durchführen."
        Korrekte Antwort: "irrelevant"
        """,
    "cb2": """
        # Codebuch
        ## Ziel der Annotation
        Ziel der Annotation ist es, basierend auf einem Text die Haltung eines Akteurs gegenüber einer Aussage zu bestimmen.

        ## Mögliche Kategorien:
        support (Unterstützung): Der Akteur befürwortet die Aussage ausdrücklich oder allgemein.
        opposition (Ablehnung): Der Akteur lehnt die Aussage ab.
        irrelevant (keine Haltung): Der Akteur drückt im vorliegenden Text keine Haltung zur Aussage aus oder der Text hat nichts mit der Aussage zu tun.
        """
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
    # hat eine Stellung?
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

def construct_codebook_prompt(input_text,
                              entity):
    prompt = f"Du bist ein Text-Klassifizierer. Du klassifizierst die Haltungen von Akteuren, die in einem Text zum Ausdruck kommen. Halte dich dabei an das folgende Annotations-Codebuch: {CODEBOOKS['cb2']}. Annotiere folgenden Text: {input_text}. Wie lautet die richtige Annotations-Kategorie für den Akteur {entity}? Antwort mit einer der drei Kategorien irrelevant, support, oder opposition."
    return(prompt)

def get_registered_chains():
    return(REGISTERED_LM_CHAINS)

def get_allowed_dual_lm_chains():
    return(ALLOWED_DUAL_LM_CHAINS)

class StanceClassification:
    def __init__(self,input_text,statement,entity):
        self.input_text = input_text
        self.statement = statement
        self.entity = entity
        self.stance = None
        self.meta = None
        self.masked_entity = entity
        self.masked_input_text = input_text
    def __str__(self):
        return("The stance of entity {} toward the statement {} given text {} is {}".format(
            self.entity,
            self.statement,
            self.input_text,
            self.stance))
    
    def mask_entity(self, entity_mask:str):
        self.masked_input_text = re.sub(self.entity, entity_mask, self.input_text)
        self.masked_entity = entity_mask
        return(self)

    def summarize_irrelevant_stance_chain(self, lm, chat,lm2=None, log=True):
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
    def summarize_v2_irrelevant_stance_chain(self, lm, chat, lm2=None, log=True):
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
    def summarize_v2_chain(self, lm, chat, lm2=None, log=True):
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
        """Stance detection prompt chain with stages "irrelevance check" -> "stance specific summary"
        
        Stance classification is taken directly from constraining grammar for the summary.
        Chain does not work with models that do not allow constraining grammar (such as OpenAI)

        Args:
            lm (_type_): language model backend.
            chat (_type_): boolean - should a chat variant be used?
            lm2 (_type_, optional): language model backend for "stance specific summary" stage. Defaults to None, in which case lm is used.
            log (bool, optional): Output logs during run? Defaults to True.
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
    def irrelevant_stance_chain(self, lm, chat, lm2=None, log=True):
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
    
    def nested_irrelevant_summary_explicit(self, lm, chat, lm2=None, log=True):
        """
        has the following prompt chains:
        construct_general_stance_prompt
        construct_irrelevance_prompt
        construct_summary_prompt 
        construct_support_stance_prompt
        construct_opposition_stance_prompt, else: irrelevant
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
    
    def nested_irrelevant_summary_v2_explicit(self, lm, chat, lm2=None, log=True):
        """
        has the following prompt chains:
        construct_general_stance_prompt
        construct_irrelevance_prompt
        construct_summary_prompt (v2)
        construct_support_stance_prompt
        construct_opposition_stance_prompt, else: irrelevant
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
    
    def codebook(self, lm, chat, lm2=None, log=True):
        if log:
            logger.info(f"Annotating according to codebook...")
        codebook_prompt = construct_codebook_prompt(input_text=self.masked_input_text,
                                                          entity=self.masked_entity)
        if chat:
            with user():
                stance = lm + codebook_prompt
            with assistant():
                stance = stance + select(["support", "opposition", "irrelevant"], name='answer')
        if not chat:
            stance = lm + codebook_prompt + select(["support", "opposition", "irrelevant"], name='answer')

        self.stance = stance["answer"]
        if log:
            logger.info(f"annotated stance: {self.stance}")
        
        self.meta = {
            "lms": {
                "stance":stance
            }
        }

        return(self)
    
    
    def nested_irrelevant_summary_v2_explicit_questioning(self, lm, chat, lm2=None, log=True):
        """
        has the following prompt chains:
        construct_general_stance_prompt (n) +q
        construct_irrelevance_prompt (i)
        construct_summary_prompt (v2) (s2)
        construct_support_stance_prompt (e)
        construct_opposition_stance_prompt, else: irrelevant
        questioning llm classifications --> after each "select"-prompt, (q)
        the LLM will be questioned if it is sure about its decision.
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
                with user():
                    questioning = irrelevance + "Bist du dir sicher?"
                with assistant():
                    questioning = questioning + select(["Ja", "Nein"], name='answer')
        if not chat:
            irrelevance = lm + general_prompt + select(["Ja", "Nein"], name='answer')
            questioning = irrelevance + "Bist du dir sicher?" + select(["Ja", "Nein"], name='answer')

        while questioning["answer"] == "Nein":
            if log:
                logger.info(f"LLM is not sure yet")
            if chat:
                with user():
                    irrelevance = lm + general_prompt
                with assistant():
                    irrelevance = irrelevance + select(["Ja", "Nein"], name='answer')
                    with user():
                        questioning = irrelevance + "Bist du dir sicher?"
                    with assistant():
                        questioning = questioning + select(["Ja", "Nein"], name='answer')
            if not chat:
                irrelevance = lm + general_prompt + select(["Ja", "Nein"], name='answer')
                questioning = irrelevance + "Bist du dir sicher?" + select(["Ja", "Nein"], name='answer')
                
        if log:
            logger.info(f"you questioned a chatbot's decision, congrats")
        if questioning["answer"] == "Ja" and irrelevance["answer"] == "Nein":
            self.stance = "irrelevant"
            stance = None
            summary = None
        if questioning["answer"] == "Ja" and irrelevance["answer"] == "Ja":
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
            "questioning":questioning,
            "stance":stance
            }
        }
        return(self)