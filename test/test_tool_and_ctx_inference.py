"""Regression tests for the two binding-settings model-profile actions.

Covered behaviours:

1. **Rule-based inference** of context size, vision and *tool calling* from
   the model name (used both when creating profiles and when re-applying
   inference to existing ones).
2. **Auto-generation** now writes ``TOOL_CALLING_ENABLED`` and, when the
   binding declares ``ctx_size`` as a per-model parameter (llama_cpp_server
   style), populates it from the *inferred* context of the model instead of
   the binding default.
3. **Re-apply inference** honours user divergences: a field a user explicitly
   toggled is not overwritten by a later inference pass.
"""
from pathlib import Path
from typing import Any, Dict

import pytest

from lollms_client import lollms_config_cli_env as wizard


# A shrunken llama_cpp_server description: enough to exercise both actions
# without touching the real binding.
_DESC: Dict[str, Any] = {
    "global_input_parameters": [
        {"name": "host", "type": "str", "default": "127.0.0.1"},
    ],
    "model_input_parameters": [
        {"name": "model_name", "type": "str", "default": ""},
    ],
    "model_profiles_input_parameters": [
        {"name": "ctx_size", "type": "int", "default": 4096},
        {"name": "n_gpu_layers", "type": "int", "default": -1},
    ],
}


@pytest.fixture
def fake_desc(monkeypatch):
    monkeypatch.setattr(
        wizard, "_get_binding_description",
        lambda b_name, b_type: _DESC if b_name == "llama_cpp_server" else None,
    )
    yield


@pytest.fixture
def seeded_config_map() -> Dict[str, str]:
    return {
        "LLM_BINDINGS_MAIN_BINDING_NAME": "llama_cpp_server",
    }


# ── Inference rules (regex-based) ────────────────────────────────────────────

@pytest.mark.parametrize("model, expected_tools", [
    ("Qwen2.5-7B-Instruct", True),
    ("qwen3-32b-instruct", True),
    ("Llama-3.1-8B-Instruct", True),
    ("Llama-3-8B-Instruct", False),
    ("Mistral-Small-3", True),
    ("DeepSeek-R1-Distill", False),  # reasoning-only, no tool calls
    ("deepseek-v3-chat", True),
    ("Hermes-3-Llama", True),
    ("phi-3-mini", False),
    ("phi-4", True),
    ("command-r-plus", True),
    ("gpt-4o-mini", True),
    ("tinyllama", False),
])
def test_tool_calling_inference(model, expected_tools):
    assert wizard._infer_model_capabilities(model)["tool_calling_enabled"] is expected_tools


@pytest.mark.parametrize("model, expected_ctx", [
    ("qwen2.5-7b", 32_768),
    ("qwen3-32b", 131_072),
    ("llama-3.3-70b", 128_000),
    ("phi-4", 16_384),
    ("gemma-3-27b-it", 128_000),
])
def test_ctx_inference(model, expected_ctx):
    assert wizard._infer_model_capabilities(model)["forced_context_size"] == expected_ctx


def test_vision_and_thinking_still_inferred():
    assert wizard._infer_model_capabilities("Qwen2.5-VL-7B")["vision_enabled"] is True
    assert wizard._infer_model_capabilities("DeepSeek-R1")["thinking_enabled"] is True


# ── Auto-generation now populates tool_calling & per-profile ctx_size ───────

def test_autogenerate_writes_tool_calling_and_ctx_size(
    fake_desc, seeded_config_map, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path.cwd() / "_tmp_tool_test_home"))
    monkeypatch.setattr(wizard, "_safe_confirm", lambda prompt, default=False: True)
    monkeypatch.setattr(
        wizard, "_fetch_available_models",
        lambda *a, **kw: ["Qwen2.5-7B-Instruct", "Llama-3-8B-Instruct"],
    )

    wizard._autogenerate_profiles_for_binding("llm", "main", seeded_config_map)

    prefixes = [
        p for p in seeded_config_map
        if p.startswith("LLM_PROFILES_") and p.endswith("_MODEL_NAME")
    ]
    assert len(prefixes) == 2

    # Qwen2.5 → tool calling ON, ctx_size = 32768 (inferred, not 4096 fallback)
    qwen_prefix = next(p for p in prefixes if "QWEN2_5" in p).replace("MODEL_NAME", "")
    assert seeded_config_map[qwen_prefix + "TOOL_CALLING_ENABLED"] == "true"
    assert seeded_config_map[qwen_prefix + "CTX_SIZE"] == "32768"

    # Llama-3 (vanilla) → tool calling OFF, ctx still inferred (8k)
    llama_prefix = next(p for p in prefixes if "LLAMA_3_8B" in p).replace("MODEL_NAME", "")
    assert seeded_config_map[llama_prefix + "TOOL_CALLING_ENABLED"] == "false"
    assert seeded_config_map[llama_prefix + "CTX_SIZE"] == "8192"


# ── Re-apply inference preserves user divergence ────────────────────────────

def test_reapply_preserves_user_overrides_but_fills_new_fields(
    fake_desc, seeded_config_map, monkeypatch
):
    # Existing profile that a user already tuned: tool_calling explicitly off,
    # ctx_size bumped to a custom value; nothing else set (pre-dates new
    # inference rules for VISION/THINKING/TOOL).
    seeded_config_map.update({
        "LLM_PROFILES_MAIN_QWEN_BINDING_ALIAS": "main",
        "LLM_PROFILES_MAIN_QWEN_MODEL_NAME": "Qwen2.5-7B-Instruct",
        "LLM_PROFILES_MAIN_QWEN_TOOL_CALLING_ENABLED": "false",  # user override
        "LLM_PROFILES_MAIN_QWEN_CTX_SIZE": "65536",              # user override
    })

    monkeypatch.setattr(wizard, "_safe_confirm", lambda prompt, default=False: True)

    wizard._reapply_inference_to_existing_profiles("llm", "main", seeded_config_map)

    prefix = "LLM_PROFILES_MAIN_QWEN_"
    # Preserved: user explicitly disabled tool calling and set 64k context.
    assert seeded_config_map[prefix + "TOOL_CALLING_ENABLED"] == "false"
    assert seeded_config_map[prefix + "CTX_SIZE"] == "65536"
    # Filled in: fields that were empty get the inferred defaults.
    # (Qwen2.5 has no vision/video, has thinking table hit via Qwen3? No →
    #  thinking stays false; and TOOL_CALLING stays as the preserved false.)
    assert seeded_config_map[prefix + "VISION_ENABLED"] == "false"
    assert seeded_config_map[prefix + "VIDEO_ENABLED"] == "false"
    assert seeded_config_map[prefix + "IMAGE_TOKEN_COST"] == "256"

def test_reapply_updates_when_value_matches_previous_default(
    fake_desc, seeded_config_map, monkeypatch
):
    # Values were auto-filled at creation time and never touched → re-inference
    # may now refresh them (e.g. tool_calling default false → true for qwen).
    seeded_config_map.update({
        "LLM_PROFILES_MAIN_QWEN_BINDING_ALIAS": "main",
        "LLM_PROFILES_MAIN_QWEN_MODEL_NAME": "Qwen2.5-7B-Instruct",
        "LLM_PROFILES_MAIN_QWEN_TOOL_CALLING_ENABLED": "false",  # matches framework default
        "LLM_PROFILES_MAIN_QWEN_VISION_ENABLED": "false",
    })
    monkeypatch.setattr(wizard, "_safe_confirm", lambda prompt, default=False: True)

    wizard._reapply_inference_to_existing_profiles("llm", "main", seeded_config_map)

    prefix = "LLM_PROFILES_MAIN_QWEN_"
    # Tool calling was at the framework default ("false") so inference upgrades it.
    assert seeded_config_map[prefix + "TOOL_CALLING_ENABLED"] == "true"

def test_reapply_is_idempotent(fake_desc, seeded_config_map, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path.cwd() / "_tmp_idem_home"))
    monkeypatch.setattr(wizard, "_safe_confirm", lambda prompt, default=False: True)
    monkeypatch.setattr(
        wizard, "_fetch_available_models",
        lambda *a, **kw: ["Qwen2.5-7B-Instruct"],
    )
    wizard._autogenerate_profiles_for_binding("llm", "main", seeded_config_map)

    # Second pass should report zero changes for the same inputs.
    before = dict(seeded_config_map)
    wizard._reapply_inference_to_existing_profiles("llm", "main", seeded_config_map)
    assert seeded_config_map == before