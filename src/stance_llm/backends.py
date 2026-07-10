"""Helpers for loading and validating local guidance model backends.

stance-llm's prompt chains rely on guidance constrained generation (``select``/
``gen``), which only works with *in-process* guidance backends
(``guidance.models.Transformers``, ``guidance.models.LlamaCpp``,
``guidance.models.OnnxRuntimeGenAI``). API/HTTP backends -- including ollama's
OpenAI-compatible server -- cannot enforce a grammar and are unsupported.

This module keeps heavy imports (``guidance``, ``llama_cpp``) inside functions so
that ``import stance_llm`` stays cheap and does not require those extras.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

# Common on-disk locations of the ollama models root (the directory that holds
# the ``manifests`` and ``blobs`` subfolders).
_DEFAULT_OLLAMA_ROOTS = [
    Path.home() / ".ollama" / "models",
    Path("/usr/share/ollama/.ollama/models"),
    Path("/var/lib/ollama/.ollama/models"),
]


def _ollama_roots(root: str | os.PathLike | None = None) -> list[Path]:
    if root is not None:
        return [Path(root)]
    roots: list[Path] = []
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        roots.append(Path(env))
    roots.extend(_DEFAULT_OLLAMA_ROOTS)
    return roots


def resolve_ollama_gguf(
    model_name: str, root: str | os.PathLike | None = None
) -> Path:
    """Resolve the GGUF blob path for a model already pulled with ollama.

    ollama stores an OCI-style manifest per model and content-addressed blobs.
    This finds the manifest for ``model_name`` (``name`` or ``name:tag``; the tag
    defaults to ``latest``), locates the model layer, and returns the path to its
    GGUF blob. No model is loaded and no heavy dependency is imported, so this is
    safe to unit-test.

    Args:
        model_name: ollama model reference, e.g. ``"qwen2.5:1.5b"`` or
            ``"llama3.2"`` or ``"hrbrmstr/ornith-9b-fixed"``.
        root: optional ollama models root (the directory containing ``manifests``
            and ``blobs``). If omitted, ``$OLLAMA_MODELS`` and common user/system
            locations are searched.

    Returns:
        ``pathlib.Path`` to the GGUF blob.

    Raises:
        FileNotFoundError: if no matching manifest/blob can be found.
    """
    name, _, tag = model_name.partition(":")
    tag = tag or "latest"
    # Un-namespaced names live under the default ``library`` namespace.
    namespaced = name if "/" in name else f"library/{name}"

    tried: list[str] = []
    for models_root in _ollama_roots(root):
        manifests = models_root / "manifests"
        candidates = [manifests / "registry.ollama.ai" / namespaced / tag]
        # Fall back to a glob across registry hosts.
        if not any(c.is_file() for c in candidates):
            candidates += sorted(manifests.glob(f"*/{namespaced}/{tag}"))
        for manifest_path in candidates:
            tried.append(str(manifest_path))
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            for layer in manifest.get("layers", []):
                if "model" in layer.get("mediaType", ""):
                    blob = models_root / "blobs" / layer["digest"].replace(":", "-")
                    if blob.is_file():
                        return blob
                    tried.append(str(blob))
    raise FileNotFoundError(
        f"Could not resolve a GGUF blob for ollama model {model_name!r}. "
        f"Checked: {tried}. Is the model pulled (`ollama pull {model_name}`), and "
        f"is the ollama models root discoverable? Set OLLAMA_MODELS or pass root= "
        f"if it lives in an unusual location."
    )


def from_ollama(
    model_name: str, root: str | os.PathLike | None = None, **llama_kwargs
):
    """Build a ``guidance.models.LlamaCpp`` from a model pulled with ollama.

    Resolves the GGUF blob ollama stored (see :func:`resolve_ollama_gguf`) and
    loads it with llama.cpp -- the supported way to use an ollama model with
    stance-llm, since ollama's own server cannot enforce guidance grammars.

    Args:
        model_name: ollama model reference, e.g. ``"qwen2.5:1.5b"``.
        root: optional ollama models root (see :func:`resolve_ollama_gguf`).
        **llama_kwargs: forwarded to ``guidance.models.LlamaCpp`` (e.g. ``n_ctx``,
            ``n_gpu_layers``). ``n_ctx`` defaults to 4096.

    Returns:
        A ``guidance.models.LlamaCpp`` instance.

    Raises:
        ImportError: if ``llama-cpp-python`` is not installed.
        FileNotFoundError: if the model cannot be resolved.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Using ollama models with stance-llm needs llama-cpp-python. Install "
            'it with `pip install "stance-llm[llamacpp]"` (or `pip install '
            "llama-cpp-python`). Note: the installed llama-cpp-python must bundle a "
            "llama.cpp new enough for the model's architecture, or loading the GGUF "
            "will fail with an error like 'unknown architecture' / 'wrong array "
            "length'."
        ) from e
    from guidance import models

    blob = resolve_ollama_gguf(model_name, root=root)
    llama_kwargs.setdefault("n_ctx", 4096)
    logger.info(f"Loading ollama model {model_name!r} from {blob} via LlamaCpp")
    return models.LlamaCpp(str(blob), **llama_kwargs)


def is_chat_model(llm) -> bool | None:
    """Best-effort detection of whether a guidance model is a chat/instruct model.

    Returns ``True``/``False`` when a chat template can be located, or ``None``
    when it cannot be determined. Used to resolve ``chat="auto"``.

    Note: guidance wraps non-chat models with a ChatML *fallback* template, so we
    inspect the *underlying* tokenizer/metadata rather than guidance's wrapper.
    """
    engine = getattr(llm, "engine", None)
    tokenizer = getattr(engine, "tokenizer", None)
    # Transformers: the wrapped HF tokenizer carries the authoritative chat_template
    # (None for a plain completion model, a Jinja string for a chat model).
    hf_tokenizer = getattr(tokenizer, "_orig_tokenizer", None)
    if hf_tokenizer is not None:
        return getattr(hf_tokenizer, "chat_template", None) is not None
    # LlamaCpp: the GGUF metadata carries the chat template (absent for base models).
    metadata = getattr(getattr(engine, "model_obj", None), "metadata", None)
    if isinstance(metadata, dict):
        return bool(metadata.get("tokenizer.chat_template"))
    return None


def resolve_chat(llm, chat):
    """Resolve the ``chat`` argument, supporting ``chat="auto"``.

    ``True``/``False`` are returned as-is (explicit override). ``"auto"`` triggers
    :func:`is_chat_model`; if capability cannot be determined, it defaults to
    ``True`` -- guidance applies a ChatML fallback, matching stance-llm's
    historical default.
    """
    if chat == "auto":
        detected = is_chat_model(llm)
        resolved = True if detected is None else detected
        logger.debug(f"chat='auto' resolved to {resolved} (detected={detected})")
        return resolved
    return chat


def assert_constrained_generation(llm) -> None:
    """Verify a backend can enforce constrained generation (guidance grammars).

    stance-llm's chains rely on ``select``/``gen`` constraints, which only work
    with in-process guidance backends. This runs a tiny probe and, if the backend
    rejects the grammar (as API/HTTP backends like ollama's server do), raises a
    clear error instead of letting a cryptic failure surface deep inside a chain.

    Best-effort: if the probe cannot run for unrelated reasons, it does nothing.

    Raises:
        RuntimeError: if the backend cannot enforce a grammar.
    """
    from guidance import assistant, select

    try:
        # The grammar rejection only surfaces inside an active role, so probe there.
        with assistant():
            _ = llm + "stance-llm capability probe: " + select(
                ["a", "b"], name="_stance_llm_probe"
            )
    except Exception as e:  # noqa: BLE001 - best-effort diagnostic
        if type(e).__name__ == "UnsupportedNodeError" or "does not support" in str(e).lower():
            raise RuntimeError(
                "This model backend cannot enforce constrained generation "
                "(guidance grammars), which stance-llm requires. Use a local "
                "in-process backend such as guidance.models.Transformers, "
                "guidance.models.LlamaCpp, or guidance.models.OnnxRuntimeGenAI. "
                "API/HTTP backends -- including ollama's OpenAI-compatible server "
                "-- are not supported; to use an ollama model, load its GGUF "
                "weights in-process via stance_llm.backends.from_ollama."
            ) from e
        logger.debug(
            f"constrained-generation probe inconclusive: {type(e).__name__}: {e}"
        )
