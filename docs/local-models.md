# Using stance-llm with local models

stance-llm's prompt chains rely on **guidance constrained generation** (`select()` /
`gen()`). That constraint is only enforceable by *in-process* guidance backends:

- `guidance.models.Transformers` (HuggingFace weights) — the backend we test against
- `guidance.models.LlamaCpp` (GGUF weights, incl. models pulled with ollama)
- `guidance.models.OnnxRuntimeGenAI` (ONNX weights)

> ⚠️ **API/HTTP backends do not work.** Models reached over an API — OpenAI, and
> importantly **ollama's OpenAI-compatible server** (`http://localhost:11434/v1`) — cannot
> enforce a grammar. Every chain will fail with
> `UnsupportedNodeError: ... does not support SelectNode(...)`. To use a model you pulled
> with ollama, load its **weights** in-process (see below), not its server.

`process()` / `process_evaluate()` run a one-line capability probe up front and raise a
clear error if the backend can't do constrained generation, so you find out immediately
rather than deep inside a chain.

## Install the backend you want

The relevant libraries are packaged as optional extras:

```bash
pip install "stance-llm[transformers]"   # HuggingFace + torch
pip install "stance-llm[llamacpp]"       # llama-cpp-python (GGUF, incl. ollama models)
pip install "stance-llm[onnx]"           # onnxruntime-genai
```

## Recipe A — HuggingFace Transformers (the tested path)

```python
from guidance import models
from stance_llm.process import detect_stance

llm = models.Transformers("DiscoResearch/DiscoLM_German_7b_v1")  # downloads weights; GPU recommended

eg = {"text": "...", "ent_text": "Stadt Bern", "statement": "..."}
result = detect_stance(eg, llm=llm, chain_label="is")  # chat is auto-detected
print(result.stance)
```

## Recipe B — A GGUF file via LlamaCpp

Download a `.gguf` (e.g. from a `bartowski` / `TheBloke` HuggingFace repo) and point
`LlamaCpp` at it:

```python
from guidance import models
llm = models.LlamaCpp("/path/to/model.gguf", n_ctx=4096)
```

## Recipe C — A model you already pulled with ollama

ollama stores each model as an OCI-style manifest plus content-addressed blobs; the actual
weights are a GGUF blob with an opaque `sha256-…` filename. stance-llm ships a helper that
finds that blob and loads it with `LlamaCpp`:

```python
from stance_llm.backends import from_ollama
from stance_llm.process import detect_stance

llm = from_ollama("qwen2.5:1.5b")          # resolves the GGUF ollama pulled, loads via LlamaCpp
# from_ollama("hrbrmstr/ornith-9b-fixed")  # namespaced names work too
# from_ollama("qwen2.5:1.5b", n_gpu_layers=-1)  # extra kwargs forwarded to LlamaCpp

result = detect_stance(
    {"text": "...", "ent_text": "GreenMobility", "statement": "..."},
    llm=llm, chain_label="is", language="en",
)
```

If you'd rather resolve the path yourself (or ollama lives somewhere unusual):

```python
from stance_llm.backends import resolve_ollama_gguf
path = resolve_ollama_gguf("qwen2.5:1.5b")          # searches ~/.ollama and /usr/share/ollama
path = resolve_ollama_gguf("qwen2.5:1.5b", root="/custom/ollama/models")
```

`resolve_ollama_gguf` also honours the `OLLAMA_MODELS` environment variable. On system-wide
installs the root is typically `/usr/share/ollama/.ollama/models` (owned by the `ollama`
user — you need read access to the blobs).

### ⚠️ Gotcha: llama-cpp-python must match the model's architecture

`llama-cpp-python` bundles a pinned copy of llama.cpp. Brand-new model architectures fail to
load on older wheels, e.g.:

```
error loading model hyperparameters: key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3
```

or `unknown model architecture`. If you hit this, upgrade `llama-cpp-python` (a newer wheel
bundles a newer llama.cpp), or use a model whose architecture your installed build already
supports. ollama often ships a newer llama.cpp than the latest published
`llama-cpp-python` wheel, so a model that runs under `ollama run` may still be too new for
`LlamaCpp`.

## chat vs. completion mode

`detect_stance` / `process` / `process_evaluate` take a `chat` argument that defaults to
`"auto"`: stance-llm inspects the model's chat template and picks chat or plain-completion
prompting for you. Pass `chat=True` or `chat=False` to override the detection.
