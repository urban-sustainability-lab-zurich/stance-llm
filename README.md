# stance-llm: German LLM prompts for stance detection

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/stance-llm)
![PyPI](https://img.shields.io/pypi/v/stance-llm?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/stance-llm)

Classify stances of entities related to a statement in German text using large language models (LLMs). 

stance-llm is built on [guidance](https://github.com/guidance-ai/guidance), which provides a unified interface to different LLMs and enables constrained grammar and structured output.

stance-llm offers several prompt chains to choose from to classify stances (see [implemented prompt chains](#implemented-prompt-chains)). At its core, in terms of input and output, you choose an LLM, choose one of several prompt chain options, and then you feed stance-llm a list of dictionaries in the form:

```
[{"text":<German-text-to-analyze>, 
"ent_text": <entity string to classify stance for>, 
"statement": <the (German) statement to evaluate stance of entity toward>}]
```

for example:

```
[{"text":"Emily will LLMs in den Papageienzoo sperren und streng beaufsichtigen. Das ist gut so.", 
"ent_text": "Emily", 
"statement": "LLMs sollten in den Papageienzoo gesperrt werden."}]
```

And you get a list of `StanceClassification` objects back containing 
- your original data
- a predicted stance (here likely "support", if all went well), currently one of "support","opposition" or "irrelevant"
- meta-information on the the prompts and generated text by the LLM during processing

Going beyond basic functionality, stance-llm currently allows for [entity masking](#entity-masking), using [two different LLMs](#use-of-multiple-llms-in-one-prompt-chain) in different parts of a prompt chain, as well as [serializing classifications](#serialize-classifications), optionally alongside evaluation metrics compared to "true" stances.

## Motivation

We developed and evaluated a number of different German LLM prompts during a research project (preprint with detailed evaluation results forthcoming). 
At this stage, stance-llm provides an interface to easily use these specific, [different prompt chains](#implemented-prompt-chains) on your own data and getting structured output back.
Thus we provide a way to easily leverage LLMs for stance classification of named entities regarding an arbitrary statement in German text.

We generally believe that the hype around LLMs for many NLP tasks is overblown (for many reasons).

However, the NLP task of zero-shot classifying the stance of an entity toward any arbitrary statement is incredibly general and hard to satisfyingly solve with current tools. 
For this general, hard task, the use of LLMs, as task-unspecific, general models seemed to have a use case to us.

## Installation

> ⚠️ `stance-llm` requires **Python 3.10** or higher.

stance-llm is available through PyPI:
```bash
pip install stance-llm
```


## How to use `stance-llm`

### Data

Your data could look like this:


| id | statement                      | text            | ent_text |
|----|--------------------------------|-----------------------|----------------|
| 1  | "Mehr Bäume sollten gepflanzt werden" | "Die vereinigten Waldelfen haben eine Kampagne organisiert, die die Bevölkerung für die Vorteile des Baumpflanzens sensibilisieren soll" | "vereinigten Waldelfen"|
| 2  | "Sport ist Mord"     | "Das Sportministerium spricht sich gegen übermässigen Konsum von Zucker im Rahmen von Fahrradfahrten aus" | "Sportministerium"|
> ⚠️ Your data must at least include: a text to classify a stance in, a string giving a detected actor in the text, and a statement giving a statement to evaluate the stance of the actor against

To use the data with stance-llm, turn your data into a list of dictionaries of the form:

```
[{"text":<German-text-to-analyze>, 
"ent_text": <entity string to classify stance for>, 
"statement": <the (German) statement to evaluate stance of entity toward>}]
```

Optionally, per item in the list of dictionaries:
- if you want stance-llm to also evaluate its classifications against test data, you may provide a true stance as the value of a "stance_true" key.
- you may supply an "id" key for your examples, which is serialized when using `stance_llm.process.process` or `stance_llm.process.process_evaluate` to enable you to keep track of your examples.

### Choose your LLM

stance-llm is built on top of [guidance](https://github.com/guidance-ai/guidance), making it possible to use a variety of LLMs through the [guidance.models.Model](https://guidance.readthedocs.io/en/stable/generated/guidance.models.Model.html#guidance-models-model) class, which can be either externally hosted (eg. OpenAI, VertexAI...) or running locally. 

> ⚠️ Prompt chain compatibility: Models accessed through an API (eg. OpenAI or ...) will reject some prompt chains due to not allowing for constrained grammar. If you want to make use of all available prompt chains, use an LLM running locally (eg. through `guidance.models.Transformers` or `guidance.models.LlamaCpp`), which does **not** use constrained grammar. See table below for an overview.

| prompt chain | constrained grammar    | second llm option     |
|--------------|------------------------|-----------------------|
| [is](#is)           | X                      | X                     |
| [sis](#sis)          | X                      | X                     |
| [nise](#nise)         | X                      | X                     |
| [s2](#s2)           | ✓                      | X                     |
| [is2](#is2)          | ✓                      | ✓                     |
| [s2is](#s2is)         | X                      | X                     |
| [nis2e](#nis2e)        | X                      | X                     |


## Get started

### Basic usage

Load an LLM - for example, we might use [Disco LM German 7b v1](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1).

> ⚠️ This downloads a number of large files and you'll probably need a GPU and set up your system to utilize it


```python
from guidance import models

disco7b = models.Transformers("DiscoResearch/DiscoLM_German_7b_v1")
```

or maybe you want to use OpenAI's servers to do the work for you (*wiederwillig* or *zähneknirschend*, as we say in German).

```python
from guidance import models

gpt35 = models.OpenAI("gpt-3.5-turbo",api_key=<your-API-key>)
```

Let's create some test data:

```python
test_examples = [
    {"text":"Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
     "ent_text":"Stadt Bern",
     "statement":"Das Fahrrad als Mobilitätsform soll gefördert werden.",
     "stance_true": "support"},
     {"text":"Die Stadt Bern spricht sich dafür aus, mehr Velowege zu bauen. Dies ist allerdings umstritten. Die FDP ist klar dagegen.",
     "ent_text":"FDP",
     "statement":"Das Fahrrad als Mobilitätsform soll gefördert werden.",
     "stance_true": "opposition"}]
```

Now we can run stance detection on a single item in the list.

Here, we can choose among a number of different prompt chains. Below, we use the [is](#is) chain.

```python
from stance_llm.process import detect_stance

classification = detect_stance(
            eg = test_examples[0], #we run this on the first example only
            llm = gpt35,
            chain_label = "is" # This is where we choose our prompt chain
        )
```

Et voilà, this should return a `StanceClassification` object containing (among other things, a predicted stance).

```python
classification.stance
```

### Serialize classifications

To process a list of dictionaries and serialize the classifications to a folder as `classifications.jsonl`, we have `process`:

```python
from stance_llm.process import process

process(
    egs=test_examples,
    llm=gpt35,
    export_folder=<folder-to-your-output-folder>,
    chain_used="is", #here, we choose our prompt chain
    model_used="openai-gpt35", #the label you want to give the LLM used
    stream_out=True)
```

If your examples to classify have a "stance_true" key (for example containing manually annotated stances for your examples - they must be one of "support","opposition" or "irrelevant"), you can also evaluate results of classifications with `process_evaluate`, which will create an additional `metrics.json` file in the output folder:

```python
from stance_llm.process import process_evaluate

process_evaluate(
    egs=test_examples,
    llm=gpt35,
    export_folder=<folder-to-your-output-folder>,
    chain_used="is",
    model_used="openai-gpt35", #the label you want to give the LLM used
    stream_out=True)
```

### Entity masking

LLMs are trained on large amounts of (sometimes stolen, hrrmpf) data. Given this, if you want to classify stances of entities that are relatively visible it might make sense to "mask" them. stance-llm provides a way to do so by providing an `entity_mask` option to its main functions (`detect_stance`, `process` and `process_evaluate`). You can supply a more neutral string to this option (e.g. "Organisation X") and this will hide the actual entity name from the LLM in all prompts.

```python
process(
    egs=test_examples,
    llm=gpt35,
    export_folder=<path-to-your-output-folder>,
    chain_used="is",
    model_used="openai-gpt35", #the label you want to give the LLM used
    stream_out=True,
    entity_mask="Organisation X" #this string will be used to mask the entity
    )
```

### Chat models

Some LLMs loadable as guidance models are "chat" models requiring a different form of prompting.
All prompt chains in stance-llm are implemented in both chat and non-chat versions. You can choose which version to use by specifying the boolean (True/ False) `chat` option to its main functions (`detect_stance`, `process` and `process_evaluate`). It defaults to "True".
Generally, you should get a warning (via guidance) if you use a chat version with a non-chat LLM model.

### Use of multiple LLMs in one prompt chain

Theoretically, prompt chains (currently only implemented for [is2](#is2)) can use a different LLM for different parts of the prompt chain, for example, in [is2](#is2), a locally hosted model (like Disco LM) for the classification part and a model accessed through an API for the irrelevance check part (like GPT-3.5). Using dual LLMs in this way can be enabled by passing a second `guidance.models.Model` object via the option `llm2` in `detect_stance`, `process` and `process_evaluate`.

## Implemented prompt chains

Feel free to play around with those. We will have a preprint out soon on which chains worked best on our specific data (which might be really different from yours).

### is
![is_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/is_prompt_illu.png)
1. prompts the LLM to check, if there is a stance in the text related to the statement, or not
2. if the stance in the text is found to be related to the statement, the LLM is prompted to classify the stance as support not support, if not related stance: stance=irrelevant
3. if the stance is not support: the stance=opposition 


### sis
![sis_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/sis_prompt_illu.png)
1. prompt the LLM to summarise the input text
2. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
3. if actor has a related stance: the LLM is prompted to classify the stance as opposition or support, if not related stance: stance=irrelevant


### nise
![nise_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/nise_prompt_illu.png)
1. prompts the LLM if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
2. prompts the LLM whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant
3. prompt the LLM to summarise the input text
4. prompts the LLM explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support
5. prompts the LLM explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition


### s2
![s2_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/s2_prompt_illu.png)
1. prompts the LLM to summarise the input text in relation to the statement
2. prompts the LLM to classify the detected actor's stance based on the summary. Stance class labels to select from: irrelevant, opposition, support


### is2
![is2_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/is2_prompt_illu.png)
1. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
2. if actor has a related stance: continue with 3., if not: stance=irrelevance
3. summarises text in relation to the statement and prompts in the same prompt text/step for the stance classification for either opposition or support


### s2is
![s2is_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/s2is_prompt_illu.png)
1. prompt the LLM to summarise the input text in relation to the statement
2. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
3. if actor has a related stance: the LLM is prompted to classify the stance as opposition or support, if not related stance: stance=irrelevant


### nis2e
![nis2e_prompt_illu](https://github.com/urban-sustainability-lab-zurich/stance-llm/docs/figures/nis2e_prompt_illu.png)
1. prompts the LLM if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
2. prompts the LLM whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant
3. prompt the LLM to summarise the input text in relation to the statement
4. prompts the LLM explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support
5. prompts the LLM explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition

## Roadmap

For future releases, we could envision at least:
- a closer integration with spacy doc objects
- extending chains to different languages
- providing an interface for providing custom prompt chains

Get in touch if you want to contribute.
