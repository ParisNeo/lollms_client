import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Import LollmsClient at module level so patch.object can resolve it in the fixture.

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile  # noqa: E402

@pytest.fixture
def mock_all_binding_managers():
    """Mocks *all nine* modality binding managers to prevent real instantiation.

    Why all nine and not just the six "obvious" modals?

    ``LollmsClient.__init__`` constructs every manager eagerly, and each real
    manager scans its on-disk binding directory and imports every discovered
    binding module. ``tools_bindings/`` contains ~25 nested toolsets,
    ``connection_bindings/`` pulls in Discord/Slack/Telegram/WhatsApp SDKs, and
    ``rag_bindings/`` can drag in ``torch`` / ``sentence_transformers``. When
    those three managers were left unmocked, every single ``LollmsClient()``
    call paid that full import/scan cost — which is what made this file feel
    pathologically slow. Mocking them is both correct (unit isolation) and
    dramatically faster.

    The factory additionally configures each mock to behave like a *remote*
    binding (``is_local() -> False``, ``is_model_loaded() -> True``) so the
    client's VRAM-management / model-reload code paths short-circuit.
    """
    patches = [
        patch("lollms_client.lollms_core.LollmsLLMBindingManager"),
        patch("lollms_client.lollms_core.LollmsTTIBindingManager"),
        patch("lollms_client.lollms_core.LollmsTTSBindingManager"),
        patch("lollms_client.lollms_core.LollmsSTTBindingManager"),
        patch("lollms_client.lollms_core.LollmsTTVBindingManager"),
        patch("lollms_client.lollms_core.LollmsTTMBindingManager"),
        patch("lollms_client.lollms_core.LollmsTOOLBindingManager"),
        patch("lollms_client.lollms_core.LollmsConnectionBindingManager"),
        patch("lollms_client.lollms_core.LollmsRAGBindingManager"),
    ]
    started = [p.start() for p in patches]

    def create_side_effect(binding_name, **kwargs):
        mock = MagicMock()
        mock.binding_name = binding_name
        mock.model_name = kwargs.get("model_name", "test-model")
        mock.reset_cancel = MagicMock()
        mock.vision_enabled = kwargs.get("vision_enabled", False)
        # Remote-binding behaviour: skip ensure_model_loaded / VRAM reclaim.
        mock.is_local = MagicMock(return_value=False)
        mock.is_model_loaded = MagicMock(return_value=True)
        mock.has_active_resources = MagicMock(return_value=False)
        mock.get_loaded_models = MagicMock(return_value=[])
        mock.load_model = MagicMock(return_value=True)
        mock.unload_model = MagicMock(return_value=True)
        return mock

    for mgr in started:
        mgr.return_value.create_binding.side_effect = create_side_effect

    # Safety net: never touch ~/.lollms_client/config.yaml during unit tests.
    with patch.object(LollmsClient, "_save_single_default_profiles_to_disk", return_value=None):
        try:
            llm_mgr, tti_mgr, tts_mgr, stt_mgr, ttv_mgr, ttm_mgr, tools_mgr, conn_mgr, rag_mgr = started
            yield {
                "llm": llm_mgr, "tti": tti_mgr, "tts": tts_mgr,
                "stt": stt_mgr, "ttv": ttv_mgr, "ttm": ttm_mgr,
                "tools": tools_mgr, "connection": conn_mgr, "rag": rag_mgr,
            }
        finally:
            for p in reversed(patches):
                p.stop()


def test_lollms_binding_and_model_profile_dataclasses():
    """Tests the two-tier profile dataclasses."""
    b_profile = LollmsBindingProfile(
        name="local_ollama",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434"},
        is_default=True
    )
    assert b_profile.name == "local_ollama"
    assert b_profile.binding_name == "ollama"
    assert b_profile.is_default is True

    m_profile = LollmsModelProfile(
        name="llama_chat",
        binding_profile_name="local_ollama",
        model_name="llama3",
        is_default=True,
        vision_enabled=True,
        forced_context_size=8192
    )
    assert m_profile.name == "llama_chat"
    assert m_profile.binding_profile_name == "local_ollama"
    assert m_profile.model_name == "llama3"
    assert m_profile.vision_enabled is True
    assert m_profile.forced_context_size == 8192

def test_universal_lazy_loading_all_modalities(mock_all_binding_managers):
    """Ensures non-default profiles for all modalities are registered but NOT instantiated at init."""
    client = LollmsClient(
        llm_binding_profiles={
            "local_ollama": LollmsBindingProfile(name="local_ollama", binding_name="ollama", is_default=True)
        },
        llm_model_profiles={
            "master": LollmsModelProfile(name="master", binding_profile_name="local_ollama", model_name="llama3", is_default=True),
            "coder": LollmsModelProfile(name="coder", binding_profile_name="local_ollama", model_name="qwen2.5-coder")
        },
        tti_binding_profiles={
            "local_diffusers": LollmsBindingProfile(name="local_diffusers", binding_name="diffusers", is_default=True),
            "cloud_openai": LollmsBindingProfile(name="cloud_openai", binding_name="openai")
        },
        tti_model_profiles={
            "local_sd": LollmsModelProfile(name="local_sd", binding_profile_name="local_diffusers", model_name="sd-1.5", is_default=True),
            "cloud_dalle": LollmsModelProfile(name="cloud_dalle", binding_profile_name="cloud_openai", model_name="dall-e-3")
        },
        tts_binding_profiles={
            "local_piper": LollmsBindingProfile(name="local_piper", binding_name="piper", is_default=True),
            "cloud_bark": LollmsBindingProfile(name="cloud_bark", binding_name="bark")
        },
        tts_model_profiles={
            "default_piper": LollmsModelProfile(name="default_piper", binding_profile_name="local_piper", is_default=True),
            "alt_bark": LollmsModelProfile(name="alt_bark", binding_profile_name="cloud_bark")
        }
    )
    
    # LLM
    assert "master" in client.llms
    assert "coder" not in client.llms
    assert client.llm is not None
    assert client._active_llm_alias == "master"
    
    # TTI
    assert "local_sd" in client.tti_model_profiles_registry
    assert "cloud_dalle" in client.tti_model_profiles_registry
    assert "local_sd" in client.ttis  # Default instantiated
    assert "cloud_dalle" not in client.ttis  # Non-default lazy
    assert client.tti is not None
    assert client._active_tti_alias == "local_sd"
    
    # TTS
    assert "default_piper" in client.tts_model_profiles_registry
    assert "alt_bark" in client.tts_model_profiles_registry
    assert "default_piper" in client.tts_bindings  # Default instantiated
    assert "alt_bark" not in client.tts_bindings  # Non-default lazy
    assert client.tts is not None
    assert client._active_tts_alias == "default_piper"

def test_switch_tti_lazy_instantiation(mock_all_binding_managers):
    """Tests switching TTI modality triggers lazy instantiation."""
    client = LollmsClient(
        tti_binding_profiles={
            "local_engine": LollmsBindingProfile(name="local_engine", binding_name="diffusers", is_default=True),
            "cloud_engine": LollmsBindingProfile(name="cloud_engine", binding_name="openai")
        },
        tti_model_profiles={
            "default_tti": LollmsModelProfile(name="default_tti", binding_profile_name="local_engine", is_default=True),
            "secondary_tti": LollmsModelProfile(name="secondary_tti", binding_profile_name="cloud_engine", model_name="dall-e-3")
        }
    )
    
    # Switch to non-default TTI
    result = client.switch_tti("secondary_tti")
    assert result is True
    assert "secondary_tti" in client.ttis
    assert client._active_tti_alias == "secondary_tti"
    assert client.tti == client.ttis["secondary_tti"]
    
    # Verify the manager was called to instantiate it
    mock_tti_mgr = mock_all_binding_managers["tti"]
    assert mock_tti_mgr.return_value.create_binding.call_count >= 2

def test_switch_tts_caching(mock_all_binding_managers):
    """Tests that switching back to an already instantiated TTS uses the cache."""
    client = LollmsClient(
        tts_binding_profiles={
            "local_engine": LollmsBindingProfile(name="local_engine", binding_name="piper", is_default=True),
            "alt_engine": LollmsBindingProfile(name="alt_engine", binding_name="bark")
        },
        tts_model_profiles={
            "default_tts": LollmsModelProfile(name="default_tts", binding_profile_name="local_engine", is_default=True),
            "secondary_tts": LollmsModelProfile(name="secondary_tts", binding_profile_name="alt_engine")
        }
    )
    
    # Switch to secondary
    client.switch_tts("secondary_tts")
    first_instance = client.tts_bindings["secondary_tts"]
    
    # Switch back to default
    client.switch_tts("default_tts")
    
    # Switch to secondary again
    client.switch_tts("secondary_tts")
    second_instance = client.tts_bindings["secondary_tts"]
    
    # Should be the exact same object (cached)
    assert first_instance is second_instance

def test_stt_ttv_ttm_profiles(mock_all_binding_managers):
    """Tests STT, TTV, and TTM profile registration and lazy loading."""
    client = LollmsClient(
        stt_binding_profiles={
            "local_stt": LollmsBindingProfile(name="local_stt", binding_name="whisper", is_default=True),
            "alt_stt_b": LollmsBindingProfile(name="alt_stt_b", binding_name="lollms")
        },
        stt_model_profiles={
            "default_stt": LollmsModelProfile(name="default_stt", binding_profile_name="local_stt", is_default=True),
            "alt_stt": LollmsModelProfile(name="alt_stt", binding_profile_name="alt_stt_b")
        },
        ttv_binding_profiles={
            "ttv_b": LollmsBindingProfile(name="ttv_b", binding_name="diffusers", is_default=True)
        },
        ttv_model_profiles={
            "default_ttv": LollmsModelProfile(name="default_ttv", binding_profile_name="ttv_b", is_default=True)
        },
        ttm_binding_profiles={
            "ttm_b": LollmsBindingProfile(name="ttm_b", binding_name="audiocraft", is_default=True)
        },
        ttm_model_profiles={
            "default_ttm": LollmsModelProfile(name="default_ttm", binding_profile_name="ttm_b", is_default=True)
        }
    )
    
    # STT
    assert "default_stt" in client.stts
    assert "alt_stt" not in client.stts
    client.switch_stt("alt_stt")
    assert "alt_stt" in client.stts
    assert client._active_stt_alias == "alt_stt"
    
    # TTV
    assert "default_ttv" in client.ttvs
    assert client._active_ttv_alias == "default_ttv"
    
    # TTM
    assert "default_ttm" in client.ttms
    assert client._active_ttm_alias == "default_ttm"

def test_invalid_modality_switch(mock_all_binding_managers):
    """Tests switching to an invalid alias returns False."""
    client = LollmsClient(
        tti_binding_profiles={
            "b_tti": LollmsBindingProfile(name="b_tti", binding_name="diffusers", is_default=True)
        },
        tti_model_profiles={
            "default_tti": LollmsModelProfile(name="default_tti", binding_profile_name="b_tti", is_default=True)
        }
    )
    
    result = client.switch_tti("nonexistent_tti")
    assert result is False
    assert client._active_tti_alias == "default_tti"

def test_tools_profiles_and_switching(mock_all_binding_managers):
    """Tests tools binding/model profile registration, lazy loading, switching, and eager instantiation."""
    client = LollmsClient(
        tools_binding_profiles={
            "mcp_main": LollmsBindingProfile(name="mcp_main", binding_name="mcp", binding_config={"host": "127.0.0.1"})
        },
        tools_model_profiles={
            "default_tools": LollmsModelProfile(name="default_tools", binding_profile_name="mcp_main", model_name="web-tools", is_default=True),
            "secondary_tools": LollmsModelProfile(name="secondary_tools", binding_profile_name="mcp_main", model_name="filesystem")
        }
    )

    assert "default_tools" in client.toolss
    assert "secondary_tools" not in client.toolss
    assert client.tools is not None
    assert client._active_tools_alias == "default_tools"

    res = client.switch_tools("secondary_tools")
    assert res is True
    assert "secondary_tools" in client.toolss
    assert client._active_tools_alias == "secondary_tools"
    assert client.tools == client.toolss["secondary_tools"]

    first_inst = client.toolss["secondary_tools"]
    client.switch_tools("default_tools")
    client.switch_tools("secondary_tools")
    assert client.toolss["secondary_tools"] is first_inst

def test_update_tools_binding_replaces_master(mock_all_binding_managers):
    client = LollmsClient(
        tools_binding_name="mcp",
        tools_binding_config={"model_name": "web-tools"}
    )
    assert "master" in client.tools_binding_profiles_registry
    assert client._active_tools_alias == "master"

    client.update_tools_binding("lcp", {"model_name": "filesystem"})
    assert client.tools_binding_profiles_registry["master"].binding_name == "lcp"
    assert client.tools_binding_profiles_registry["master"].binding_config["model_name"] == "filesystem"
    assert "master" in client.toolss
    assert client._active_tools_alias == "master"

def test_update_tti_binding_replaces_master(mock_all_binding_managers):
    """Tests updating a modality binding replaces the master profile and reinstantiates."""
    client = LollmsClient(
        llm_binding_name="ollama",
        tti_binding_name="diffusers",
        tti_binding_config={"model_name": "sd-v1.5"}
    )
    
    # Initial state
    assert "master" in client.tti_binding_profiles_registry
    assert client._active_tti_alias == "master"
    
    # Update TTI binding
    client.update_tti_binding("openai", {"model_name": "dall-e-3"})
    
    # Master should be updated and reinstantiated
    assert client.tti_binding_profiles_registry["master"].binding_name == "openai"
    assert client.tti_binding_profiles_registry["master"].binding_config["model_name"] == "dall-e-3"
    assert "master" in client.ttis  # Re-instantiated
    assert client._active_tti_alias == "master"