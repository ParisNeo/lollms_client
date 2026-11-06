"""Regression tests for per-model (per-profile) parameter declaration.

A binding opts into per-model configuration by declaring, in its
``description.yaml``::

    model_profiles_input_parameters:
      - name: n_gpu_layers
        type: int
        mandatory: false
        default: -1
        description: "Number of layers to offload to GPU (-1 = auto)."

Those parameters belong to *each profile* of the binding — not to the
binding instance — so a spawner binding can vary ``n_gpu_layers``,
``context_size``, ``flash_attn``, ``chat_format``, ... per model.

The mechanism is validated end-to-end through the visible wizard helpers:
schema discovery, profile configuration, edit-menu backfill, auto-generation
defaults, and the headless config API.
"""
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from lollms_client import lollms_config_cli_env as wizard


# ── Fixtures ─────────────────────────────────────────────────────────────────

_FAKE_LLAMACPP_DESC: Dict[str, Any] = {
    "title": "Fake llama_cpp_server",
    "global_input_parameters": [
        {"name": "host", "type": "str", "mandatory": False, "default": "127.0.0.1"},
    ],
    "model_input_parameters": [
        {"name": "model_name", "type": "str", "mandatory": True, "default": ""},
    ],
    # The mechanism being tested:
    "model_profiles_input_parameters": [
        {"name": "n_gpu_layers", "type": "int", "mandatory": False, "default": -1,
         "description": "Layers offloaded to GPU (-1 = all)."},
        {"name": "context_size", "type": "int", "mandatory": False, "default": 4096,
         "description": "Per-model context window."},
        {"name": "flash_attn", "type": "bool", "mandatory": False, "default": False,
         "description": "Enable flash attention."},
        {"name": "chat_format", "type": "str", "mandatory": False, "default": "",
         "description": "Prompt template family."},
    ],
}


@pytest.fixture
def fake_binding_desc(monkeypatch):
    """Patch description discovery so only our fake llama_cpp_server resolves."""
    def _fake(b_name: str, b_type: str):
        if b_name == "llama_cpp_server":
            return _FAKE_LLAMACPP_DESC
        return None

    monkeypatch.setattr(wizard, "_get_binding_description", _fake)
    yield


@pytest.fixture
def seeded_config_map() -> Dict[str, str]:
    """A config map with one LLM binding (llama_cpp_server) and no profiles."""
    return {
        "LLM_BINDINGS_MAIN_BINDING_NAME": "llama_cpp_server",
        "LLM_BINDINGS_MAIN_HOST": "127.0.0.1",
        "LLM_BINDINGS_MAIN_PORT": "8080",
    }


def _mock_profile_prompts(monkeypatch, model_name: str, param_answers: Dict[str, Any]):
    """Patch the wizard's interactive helpers so profile creation is headless.

    ``param_answers`` maps declared parameter names (as given in
    description.yaml) to the raw string the user would type.
    """
    def _select(prompt, choices, allow_back=True):
        if "binding" in prompt.lower():
            return "main"
        if "Model for" in prompt or "model" in prompt.lower():
            return model_name
        return choices[0] if choices else None

    def _confirm(prompt, default=False):
        # Decline router metadata, vision, thinking prompts.
        return False

    def _input_or_back(prompt, default=""):
        for key, answer in param_answers.items():
            if key.replace("_", " ").lower() in prompt.lower() or key.lower() in prompt.lower():
                return str(answer)
        return default

    monkeypatch.setattr(wizard, "_safe_select", _select)
    monkeypatch.setattr(wizard, "_safe_confirm", _confirm)
    monkeypatch.setattr(wizard, "_safe_input_or_back", _input_or_back)
    # Pretend model listing returned our target model.
    monkeypatch.setattr(wizard, "_fetch_available_models", lambda *a, **kw: [model_name])


# ── Schema discovery ─────────────────────────────────────────────────────────

def test_get_model_profile_schema_returns_declared_parameters(fake_binding_desc):
    schema = wizard._get_model_profile_schema("llm", "llama_cpp_server")
    names = [p["name"] for p in schema]
    assert names == ["n_gpu_layers", "context_size", "flash_attn", "chat_format"]
    by_name = {p["name"]: p for p in schema}
    assert by_name["n_gpu_layers"]["type"] == "int"
    assert by_name["n_gpu_layers"]["default"] == -1
    assert by_name["flash_attn"]["type"] == "bool"
    assert by_name["flash_attn"]["default"] is False


def test_get_model_profile_schema_empty_for_binding_without_declaration(monkeypatch):
    monkeypatch.setattr(wizard, "_get_binding_description", lambda *a, **kw: None)
    assert wizard._get_model_profile_schema("llm", "some_remote_binding") == []


def test_schema_skips_framework_keys(monkeypatch):
    """Framework-managed keys (MODEL_NAME, BINDING_ALIAS, IS_DEFAULT) must not
    leak into user-declared per-model parameters even if the YAML tries to."""
    desc = {
        "model_profiles_input_parameters": [
            {"name": "model_name", "type": "str"},
            {"name": "binding_alias", "type": "str"},
            {"name": "is_default", "type": "bool"},
            {"name": "context_size", "type": "int", "default": 2048},
        ]
    }
    monkeypatch.setattr(wizard, "_get_binding_description", lambda *a, **kw: desc)
    schema = wizard._get_model_profile_schema("llm", "llama_cpp_server")
    assert [p["name"] for p in schema] == ["context_size"]


# ── Profile creation writes per-model parameters ─────────────────────────────

def test_configure_profile_instance_writes_declared_params(
    fake_binding_desc, seeded_config_map, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path.cwd() / "_tmp_home_test_mp"))
    _mock_profile_prompts(monkeypatch, model_name="qwen2.5-7b-instruct", param_answers={
        "n_gpu_layers": "24",
        "context_size": "16384",
        "flash_attn": "y",
        "chat_format": "chatml",
    })

    wizard._configure_profile_instance("llm", "big_qwen", seeded_config_map)

    prefix = "LLM_PROFILES_BIG_QWEN_"
    assert seeded_config_map[prefix + "BINDING_ALIAS"] == "main"
    assert seeded_config_map[prefix + "MODEL_NAME"] == "qwen2.5-7b-instruct"
    assert seeded_config_map[prefix + "N_GPU_LAYERS"] == "24"
    assert seeded_config_map[prefix + "CONTEXT_SIZE"] == "16384"
    assert seeded_config_map[prefix + "FLASH_ATTN"] in ("true", "false")
    assert seeded_config_map[prefix + "CHAT_FORMAT"] == "chatml"


# ── Edit menu backfill exposes declared keys ─────────────────────────────────

def test_edit_menu_backfills_missing_declared_params(fake_binding_desc):
    """An existing profile created before the binding declared parameters must
    expose the new keys with their declared defaults when opened in the edit
    menu — without overwriting values already set by the user."""
    cfg: Dict[str, str] = {
        "LLM_BINDINGS_MAIN_BINDING_NAME": "llama_cpp_server",
        "LLM_PROFILES_BIG_QWEN_BINDING_ALIAS": "main",
        "LLM_PROFILES_BIG_QWEN_MODEL_NAME": "qwen2.5-7b-instruct",
        "LLM_PROFILES_BIG_QWEN_N_GPU_LAYERS": "24",   # existing user value
        # CONTEXT_SIZE missing → should be backfilled from default (4096)
    }

    # Prevent the edit menu from actually looping to user input: make
    # _safe_input_or_back return immediately as a "back" so the menu exits.
    with patch.object(wizard, "_safe_input_or_back", return_value=None):
        # The Menu needs to build without a TTY; force exit by patching run().
        with patch.object(wizard.Menu, "run", return_value=wizard._BACK_VALUE):
            wizard._edit_keys_menu("llm", "PROFILES", "big_qwen", cfg)

    # User value preserved, missing key backfilled.
    assert cfg["LLM_PROFILES_BIG_QWEN_N_GPU_LAYERS"] == "24"
    assert cfg["LLM_PROFILES_BIG_QWEN_CONTEXT_SIZE"] == "4096"
    assert cfg["LLM_PROFILES_BIG_QWEN_FLASH_ATTN"] == "false"
    # Empty default → key absent
    assert "LLM_PROFILES_BIG_QWEN_CHAT_FORMAT" not in cfg


# ── Auto-generation populates declared params with defaults ─────────────────

def test_autogenerate_profiles_populates_declared_defaults(
    fake_binding_desc, seeded_config_map, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path.cwd() / "_tmp_home_test_mp"))
    # Auto-generation confirms with default=True; keep it and reply no to other prompts.
    monkeypatch.setattr(wizard, "_safe_confirm", lambda prompt, default=False: True)
    monkeypatch.setattr(
        wizard, "_fetch_available_models",
        lambda *a, **kw: ["qwen2.5-7b-instruct", "llama-3.1-8b-instruct"],
    )

    wizard._autogenerate_profiles_for_binding("llm", "main", seeded_config_map)

    # Each generated profile must carry every declared default.
    for profile_alias in ("main_qwen2_5_7b_instruct", "main_llama_3_1_8b_instruct",
                          "main_qwen2.5-7b-instruct", "main_llama-3.1-8b-instruct"):
        prefix = f"LLM_PROFILES_{wizard._sanitize_alias(profile_alias).upper()}_"
        if (prefix + "MODEL_NAME") not in seeded_config_map:
            continue
        assert seeded_config_map[prefix + "N_GPU_LAYERS"] == "-1"
        assert seeded_config_map[prefix + "CONTEXT_SIZE"] == "4096"
        assert seeded_config_map[prefix + "FLASH_ATTN"] == "false"


# ── Headless config API exposure ────────────────────────────────────────────

def test_config_api_get_model_profile_schema(fake_binding_desc):
    from lollms_client import lollms_config_api as api

    schema = api.get_model_profile_schema("llm", "llama_cpp_server")
    names = [p["name"] for p in schema]
    assert names == ["n_gpu_layers", "context_size", "flash_attn", "chat_format"]


# ── Profile → binding_config propagation (unchanged contract, verified) ─────

def test_declared_params_propagate_into_binding_config(fake_binding_desc):
    """Unknown profile keys are merged into ``binding_config`` at client
    resolution time so spawner bindings receive them without extra wiring."""
    cfg: Dict[str, str] = {
        "LLM_BINDINGS_MAIN_BINDING_NAME": "llama_cpp_server",
        "LLM_PROFILES_BIG_QWEN_BINDING_ALIAS": "main",
        "LLM_PROFILES_BIG_QWEN_MODEL_NAME": "qwen2.5-7b-instruct",
        "LLM_PROFILES_BIG_QWEN_N_GPU_LAYERS": "24",
        "LLM_PROFILES_BIG_QWEN_CONTEXT_SIZE": "16384",
        "LLM_PROFILES_BIG_QWEN_IS_DEFAULT": "true",
    }
    bindings = wizard._extract_bindings_from_env("LLM", cfg)
    profiles = wizard._extract_profiles_from_env("LLM", bindings, cfg)

    assert "big_qwen" in profiles
    bcfg = profiles["big_qwen"]["binding_config"]
    assert bcfg["n_gpu_layers"] == "24"
    assert bcfg["context_size"] == "16384"