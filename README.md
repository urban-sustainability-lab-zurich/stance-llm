# stance-llm: German LLM prompts for stance detection

Classify stances of entities related to a statement in German text using large language models through guidance-llm. The package offers several prompt chains to choose from (explained below). Is built on [guidance-llm](https://github.com/guidance-ai/guidance/tree/main) which is an effective programming paradigm allowing to control the structure of LLMs outputs.

**Stance Detection: Task Definition**

Stance detection is the identification of attitudes expressed in a text towards a specific topic, organisation, or person. Opposition, support, in-favour, against, irrelevant, none, neutral, are all previously used stance classes (Mohammad et al., 2016).


### Overview: Implemented prompt chains

#### is
1. prompts the LLM to check, if there is a stance in the text related to the statement, or not
2. if the stance in the text is found to be related to the statement, the LLM is prompted to classify the stance as support not support, if not related stance: stance=irrelevant
3. if the stance is not support: the stance=opposition 


#### sis
1. prompt the LLM to summarise the input text
2. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
3. if actor has a related stance: the LLM is prompted to classify the stance as opposition or support, if not related stance: stance=irrelevant


#### nise
1. prompts the LLM if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
2. prompts the LLM whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant
3. prompt the LLM to summarise the input text
4. prompts the LLM explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support
5. prompts the LLM explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition


#### s2
1. prompts the LLM to summarise the input text in relation to the statement
2. prompts the LLM to classify the detected actor's stance based on the summary. Stance class labels to select from: irrelevant, opposition, support


#### is2
1. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
2. if actor has a related stance: continue with 3., if not: stance=irrelevance
3. summarises text in relation to the statement and prompts in the same prompt text/step for the stance classification for either opposition or support


#### s2is
1. prompt the LLM to summarise the input text in relation to the statement
2. prompts the LLM to classify whether the detected actor has a stance in the summary related to the statement, or not
3. if actor has a related stance: the LLM is prompted to classify the stance as opposition or support, if not related stance: stance=irrelevant


#### nis2e
1. prompts the LLM if there is a (general) stance of the detected actor in the text, if not: stance=irrelevant
2. prompts the LLM whether the stance of the actor has a relation to the statement, or not, if not: stance=irrelevant
3. prompt the LLM to summarise the input text in relation to the statement
4. prompts the LLM explicitly, if the stance in the summary text is in support of the statement, if not: continue with 4., if yes: stance=support if actor has a related stance: classify stance as opposition or support
5. prompts the LLM explicitly, if the stance in the summary text is in opposition of the statement, if not: stance=irrelevant, if yes: stance=opposition


## Motivation

simple and usable interface for specific German LLM prompts built on [guidance-llm](https://github.com/guidance-ai/guidance/tree/main). TODO: write some more?...


## Install

> ⚠️ Note that `stance-llm` is compatible from **Python 3.10** or higher.

stance-llm is available through PyPI
```bash
pip install stance-llm
```


## How to use `stance-llm`

### Pre-requisites

#### Data
Could look like this:


| id | statement                      | input_text            | detected_actor |
|----|--------------------------------|-----------------------|----------------|
| 1  | "More trees should be planted" | "The Health Department launched a new campaign to educate citizens about the dangers of excessive sugar consumption, featuring engaging stories and testimonials to promote healthier lifestyle choices." | "health departement"|
| 2  | "Exercise is good for you"     | "The Sports Department organized a community-wide fitness challenge, encouraging residents to participate in various activities." | "sports departement"|
> ⚠️ Your data must at least include: text, detected actor in the text, id, statement



#### LLM
Below we provide an example for an OpenAI and a LLM hosted on Hugging Face.

> ⚠️ Special case OpenAI LLMs: their models reject the following prompt chains due to constraint grammar:


| prompt chain | constrained grammar    | second llm option     |
|--------------|------------------------|-----------------------|
| is           | X                      | X                     |
| sis          | X                      | X                     |
| nise         | X                      | X                     |
| s2           | ✓                      | X                     |
| is2          | ✓                      | ✓                     |
| s2is         | X                      | X                     |
| nis2e        | X                      | X                     |




TODO: mention that the user has to know if their LLM is a chat model, or not



### Get started
TODO:

Load LLM
```python
from guidance import models

gpt35 = models.OpenAI("gpt-3.5-turbo",api_key=os.environ["OPEN_API_KEY"])
```

or
```python
from guidance import models
os.environ["HF_HOME"]
disco7b = models.Transformers("DiscoResearch/DiscoLM_German_7b_v1", 
                    cache_dir=os.environ["HF_HOME"],
                    device_map='auto',
                    torch_dtype=torch.float16)

```

Stance detection
```python
from stance_llm.process import detect_stance

detect_stance(eg:dict, llm, chain_label:str, llm2=None, chat=True, entity_mask=None)
```

Evaluation
```python
from stance_llm.process import evaluate

evaluate(egs_with_preds)
```

Process, classify and evaluation
```python
from stance_llm.process import process_evaluate
process_evaluate(egs=random.sample(egs,sample),
                lm=gpt35,
                model_used="gpt35",
                chain_used=chain,
                entity_mask=entity_mask)
```


## References

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. [Semeval-2016 task 6: Detecting stance in tweets](https://aclanthology.org/S16-1003/). In Proceedings of the 10th international workshop on semantic evaluation (SemEval-2016), pages 31–41.