"""Unit tests for stance_llm.backends.

These are fast: no model is downloaded or loaded. The ollama resolver is tested
against a fabricated manifest/blob tree, and chat detection against lightweight
stand-in objects that mimic the guidance model attribute layout.
"""

import importlib.util
import json
from types import SimpleNamespace

import pytest

from stance_llm.backends import (
    from_ollama,
    is_chat_model,
    resolve_chat,
    resolve_ollama_gguf,
)


# --- resolve_ollama_gguf ---------------------------------------------------


def _make_ollama_tree(
    tmp_path,
    model="mymodel",
    tag="latest",
    namespace="library",
    registry="registry.ollama.ai",
    digest="sha256:abc123",
    write_blob=True,
):
    """Build a minimal fake ollama models root and return its path."""
    root = tmp_path / "models"
    manifest_dir = root / "manifests" / registry / namespace / model
    manifest_dir.mkdir(parents=True)
    manifest = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tmpl"},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest},
        ]
    }
    (manifest_dir / tag).write_text(json.dumps(manifest))
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    blob_path = blobs / digest.replace(":", "-")
    if write_blob:
        blob_path.write_bytes(b"GGUF-fake")
    return root, blob_path


def test_resolve_ollama_gguf_finds_blob(tmp_path):
    root, blob_path = _make_ollama_tree(tmp_path)
    assert resolve_ollama_gguf("mymodel", root=root) == blob_path


def test_resolve_ollama_gguf_defaults_to_latest_tag(tmp_path):
    root, blob_path = _make_ollama_tree(tmp_path, tag="latest")
    # No tag given -> should resolve the "latest" manifest.
    assert resolve_ollama_gguf("mymodel", root=root) == blob_path


def test_resolve_ollama_gguf_honors_explicit_tag(tmp_path):
    root, blob_path = _make_ollama_tree(tmp_path, model="qwen2.5", tag="1.5b")
    assert resolve_ollama_gguf("qwen2.5:1.5b", root=root) == blob_path


def test_resolve_ollama_gguf_namespaced_model(tmp_path):
    root, blob_path = _make_ollama_tree(
        tmp_path, namespace="hrbrmstr", model="ornith-9b-fixed"
    )
    assert resolve_ollama_gguf("hrbrmstr/ornith-9b-fixed", root=root) == blob_path


def test_resolve_ollama_gguf_glob_fallback_other_registry(tmp_path):
    # A non-default registry host is found via the glob fallback.
    root, blob_path = _make_ollama_tree(tmp_path, registry="my.registry.example")
    assert resolve_ollama_gguf("mymodel", root=root) == blob_path


def test_resolve_ollama_gguf_missing_model_raises(tmp_path):
    root, _ = _make_ollama_tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_ollama_gguf("does-not-exist", root=root)


def test_resolve_ollama_gguf_missing_blob_raises(tmp_path):
    # Manifest present but the referenced blob file is absent.
    root, _ = _make_ollama_tree(tmp_path, write_blob=False)
    with pytest.raises(FileNotFoundError):
        resolve_ollama_gguf("mymodel", root=root)


def test_from_ollama_missing_model_raises(tmp_path):
    # If llama-cpp-python is installed we get past the import guard and fail at
    # resolution (FileNotFoundError); otherwise the import guard fires first.
    llama_installed = importlib.util.find_spec("llama_cpp") is not None
    expected = FileNotFoundError if llama_installed else ImportError
    with pytest.raises(expected):
        from_ollama("does-not-exist", root=tmp_path)


# --- chat detection / resolution -------------------------------------------


def _transformers_like(chat_template):
    """Mimic a guidance Transformers model: engine.tokenizer._orig_tokenizer."""
    return SimpleNamespace(
        engine=SimpleNamespace(
            tokenizer=SimpleNamespace(
                _orig_tokenizer=SimpleNamespace(chat_template=chat_template)
            )
        )
    )


def _llamacpp_like(metadata):
    """Mimic a guidance LlamaCpp model: engine.model_obj.metadata, no _orig_tokenizer."""
    return SimpleNamespace(
        engine=SimpleNamespace(
            tokenizer=SimpleNamespace(),  # no _orig_tokenizer
            model_obj=SimpleNamespace(metadata=metadata),
        )
    )


def test_is_chat_model_transformers_chat():
    assert is_chat_model(_transformers_like("{% for m in messages %}...")) is True


def test_is_chat_model_transformers_completion():
    assert is_chat_model(_transformers_like(None)) is False


def test_is_chat_model_llamacpp_chat():
    assert is_chat_model(_llamacpp_like({"tokenizer.chat_template": "{{...}}"})) is True


def test_is_chat_model_llamacpp_base():
    assert is_chat_model(_llamacpp_like({})) is False


def test_is_chat_model_unknown_returns_none():
    assert is_chat_model(SimpleNamespace()) is None


def test_resolve_chat_explicit_bool_passthrough():
    obj = _transformers_like(None)
    # explicit bool always wins, regardless of what detection would say
    assert resolve_chat(obj, True) is True
    assert resolve_chat(obj, False) is False


def test_resolve_chat_auto_detects_chat():
    assert resolve_chat(_transformers_like("{% ... %}"), "auto") is True
    assert resolve_chat(_llamacpp_like({"tokenizer.chat_template": "x"}), "auto") is True


def test_resolve_chat_auto_detects_completion():
    assert resolve_chat(_transformers_like(None), "auto") is False


def test_resolve_chat_auto_unknown_defaults_true():
    # When capability can't be determined, fall back to the historical default.
    assert resolve_chat(SimpleNamespace(), "auto") is True
