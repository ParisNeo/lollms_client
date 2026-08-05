import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile

@pytest.fixture
def mock_all_binding_managers():
    """Mocks all modality binding managers to prevent real instantiation."""
    with patch("lollms_client.lollms_core.LollmsLLMBindingManager") as llm_mgr, \
         patch("lollms_client.lollms_core.LollmsTTIBindingManager") as tti_mgr, \
         patch("lollms_client.lollms_core.LollmsTTSBindingManager") as tts_mgr, \
         patch("lollms_client.lollms_core.LollmsSTTBindingManager") as stt_mgr, \
         patch("lollms_client.lollms_core.LollmsTTVBindingManager") as ttv_mgr, \
         patch("lollms_client.lollms_core.LollmsTTMBindingManager") as ttm_mgr:
        
        def create_side_effect(binding_name, **kwargs):
            mock = MagicMock()
            mock.binding_name = binding_name
            mock.model_name = kwargs.get("model_name", "test-model")
            mock.reset_cancel = MagicMock()
            mock.vision_enabled = kwargs.get("vision_enabled", False)
            return mock
        
        for mgr in [llm_mgr, tti_mgr, tts_mgr, stt_mgr, ttv_mgr, ttm_mgr]:
            mgr.return_value.create_binding.side_effect = create_side_effect
            
        yield {
            "llm": llm_mgr, "tti": tti_mgr, "tts": tts_mgr,
            "stt": stt_mgr, "ttv": ttv_mgr, "ttm": ttm_mgr
        }

def test_lollms_binding_profile_dataclass():
    """Tests the universal profile dataclass."""
    profile = LollmsBindingProfile(
        name="test",
        binding_name="ollama",
        binding_config={"model_name": "llama3"},
        is_default=True,
        vision_enabled=True,
        forced_context_size=8192
    )
    assert profile.name == "test"
    assert profile.binding_name == "ollama"
    assert profile.is_default is True
    assert profile.vision_enabled is True

def test_universal_lazy_loading_all_modalities(mock_all_binding_managers):
    """Ensures non-default profiles for all modalities are registered but NOT instantiated at init."""
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={"model_name": "llama3"},
        tti_profiles={
            "local_sd": LollmsBindingProfile(name="local_sd", binding_name="diffusers", is_default=True),
            "cloud_dalle": LollmsBindingProfile(name="cloud_dalle", binding_name="openai")
        },
        tts_profiles={
            "local_piper": LollmsBindingProfile(name="local_piper", binding_name="piper", is_default=True),
            "cloud_bark": LollmsBindingProfile(name="cloud_bark", binding_name="openai")
        }
    )
    
    # LLM (Legacy master)
    assert "master" in client.llms
    assert client.llm is not None
    
    # TTI
    assert "local_sd" in client.tti_profiles_registry
    assert "cloud_dalle" in client.tti_profiles_registry
    assert "local_sd" in client.ttis  # Default instantiated
    assert "cloud_dalle" not in client.ttis  # Non-default lazy
    assert client.tti is not None
    assert client._active_tti_alias == "local_sd"
    
    # TTS
    assert "local_piper" in client.tts_profiles_registry
    assert "cloud_bark" in client.tts_profiles_registry
    assert "local_piper" in client.tts_bindings  # Default instantiated
    assert "cloud_bark" not in client.tts_bindings  # Non-default lazy
    assert client.tts is not None
    assert client._active_tts_alias == "local_piper"

def test_switch_tti_lazy_instantiation(mock_all_binding_managers):
    """Tests switching TTI modality triggers lazy instantiation."""
    client = LollmsClient(
        llm_binding_name="ollama",
        tti_profiles={
            "default_tti": LollmsBindingProfile(name="default_tti", binding_name="diffusers", is_default=True),
            "secondary_tti": LollmsBindingProfile(name="secondary_tti", binding_name="openai")
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
    # Once for default at init, once for secondary on switch
    assert mock_tti_mgr.return_value.create_binding.call_count >= 2

def test_switch_tts_caching(mock_all_binding_managers):
    """Tests that switching back to an already instantiated TTS uses the cache."""
    client = LollmsClient(
        llm_binding_name="ollama",
        tts_profiles={
            "default_tts": LollmsBindingProfile(name="default_tts", binding_name="piper", is_default=True),
            "secondary_tts": LollmsBindingProfile(name="secondary_tts", binding_name="bark")
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
        llm_binding_name="ollama",
        stt_profiles={
            "default_stt": LollmsBindingProfile(name="default_stt", binding_name="whisper", is_default=True),
            "alt_stt": LollmsBindingProfile(name="alt_stt", binding_name="lollms")
        },
        ttv_profiles={
            "default_ttv": LollmsBindingProfile(name="default_ttv", binding_name="diffusers", is_default=True)
        },
        ttm_profiles={
            "default_ttm": LollmsBindingProfile(name="default_ttm", binding_name="audiocraft", is_default=True)
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
        llm_binding_name="ollama",
        tti_profiles={
            "default_tti": LollmsBindingProfile(name="default_tti", binding_name="diffusers", is_default=True)
        }
    )
    
    result = client.switch_tti("nonexistent_tti")
    assert result is False
    assert client._active_tti_alias == "default_tti"

def test_update_tti_binding_replaces_master(mock_all_binding_managers):
    """Tests updating a modality binding replaces the master profile and reinstantiates."""
    client = LollmsClient(
        llm_binding_name="ollama",
        tti_binding_name="diffusers",
        tti_binding_config={"model_name": "sd-v1.5"}
    )
    
    # Initial state
    assert "master" in client.tti_profiles_registry
    assert client._active_tti_alias == "master"
    
    # Update TTI binding
    client.update_tti_binding("openai", {"model_name": "dall-e-3"})
    
    # Master should be updated and reinstantiated
    assert client.tti_profiles_registry["master"].binding_name == "openai"
    assert client.tti_profiles_registry["master"].binding_config["model_name"] == "dall-e-3"
    assert "master" in client.ttis  # Re-instantiated
    assert client._active_tti_alias == "master"