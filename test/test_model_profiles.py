import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile

@pytest.fixture
def mock_binding_manager():
    with patch("lollms_client.lollms_core.LollmsLLMBindingManager") as manager:
        mock_binding = MagicMock()
        mock_binding.model_name = "test-model"
        mock_binding.binding_name = "test-binding"
        mock_binding.reset_cancel = MagicMock()
        manager.return_value.create_binding.return_value = mock_binding
        yield manager

def test_lollms_model_profile_dataclass():
    """Tests the new decoupled dataclasses."""
    b_profile = LollmsBindingProfile(
        name="local_server",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434"},
        is_default=True
    )
    assert b_profile.binding_name == "ollama"
    
    m_profile = LollmsModelProfile(
        name="fast_model",
        binding_profile_name="local_server",
        model_name="llama3",
        is_default=True,
        vision_enabled=False,
        forced_context_size=8192,
        routing_config={"strategy": "tfidf"}
    )
    assert m_profile.name == "fast_model"
    assert m_profile.binding_profile_name == "local_server"
    assert m_profile.model_name == "llama3"
    assert m_profile.forced_context_size == 8192

def test_legacy_backward_compatibility(mock_binding_manager):
    """Ensures passing llm_binding_name registers a 'master' binding and model profile, and instantiates it."""
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={"model_name": "llama3"}
    )
    
    # Check Binding Registry
    assert "master" in client.llm_binding_profiles_registry
    assert client.llm_binding_profiles_registry["master"].is_default is True
    assert client.llm_binding_profiles_registry["master"].binding_name == "ollama"
    
    # Check Model Registry
    assert "master" in client.llm_model_profiles_registry
    assert client.llm_model_profiles_registry["master"].binding_profile_name == "master"
    
    # Should be eagerly instantiated
    assert "master" in client.llms
    assert client._active_llm_alias == "master"
    assert client.llm is not None

def test_lazy_loading_with_two_tier_profiles(mock_binding_manager):
    """Ensures non-default profiles are registered but NOT instantiated at init."""
    client = LollmsClient(
        llm_binding_profiles={
            "local_ollama": {
                "binding_name": "ollama",
                "binding_config": {"host_address": "http://localhost:11434"}
            }
        },
        llm_model_profiles={
            "coder": {
                "binding_profile_name": "local_ollama",
                "model_name": "qwen-coder",
                "vision_enabled": False,
                "is_default": True # Make one default to ensure eager loading works
            },
            "vision": {
                "binding_profile_name": "local_ollama",
                "model_name": "llava",
                "vision_enabled": True,
                "is_default": False
            }
        }
    )
    
    # Coder should be instantiated (default)
    assert "coder" in client.llms
    assert client.llm is not None
    assert client._active_llm_alias == "coder"
    
    # Vision should be in registry but NOT instantiated
    assert "vision" in client.llm_model_profiles_registry
    assert "vision" not in client.llms
    
    # Switch to vision
    result = client.switch_model("vision")
    assert result is True
    assert "vision" in client.llms
    assert client._active_llm_alias == "vision"
    assert client.llm == client.llms["vision"]

def test_switch_model_invalid_alias(mock_binding_manager):
    client = LollmsClient(llm_binding_name="ollama")
    result = client.switch_model("nonexistent_model")
    assert result is False
    assert client._active_llm_alias == "master" # Should remain on master

def test_switch_model_uses_cached_instance(mock_binding_manager):
    client = LollmsClient(
        llm_binding_profiles={"b1": {"binding_name": "ollama"}},
        llm_model_profiles={
            "default_m": {"binding_profile_name": "b1", "is_default": True},
            "secondary": {"binding_profile_name": "b1"}
        }
    )
    
    # First switch instantiates
    client.switch_model("secondary")
    first_instance = client.llms["secondary"]
    
    # Switch back to master
    client.switch_model("default_m")
    
    # Switch to secondary again
    client.switch_model("secondary")
    second_instance = client.llms["secondary"]
    
    # Should be the exact same object (cached)
    assert first_instance is second_instance

def test_profile_object_input(mock_binding_manager):
    """Tests passing actual dataclass objects instead of dicts."""
    b_profile = LollmsBindingProfile(name="b_obj", binding_name="ollama")
    m_profile = LollmsModelProfile(
        name="custom", 
        binding_profile_name="b_obj", 
        model_name="custom-model",
        forced_context_size=4096
    )
    
    client = LollmsClient(
        llm_binding_profiles={"b_obj": b_profile},
        llm_model_profiles={"custom": m_profile, "master": {"binding_profile_name": "b_obj", "is_default": True}}
    )
    
    assert "custom" in client.llm_model_profiles_registry
    assert client.llm_model_profiles_registry["custom"].forced_context_size == 4096

def test_update_llm_binding_replaces_master(mock_binding_manager):
    client = LollmsClient(llm_binding_name="ollama")
    
    # Update to a new binding
    client.update_llm_binding("openai", {"model_name": "gpt-4"})
    
    assert client.llm_binding_profiles_registry["master"].binding_name == "openai"
    assert client.llm_binding_profiles_registry["master"].binding_config["model_name"] == "gpt-4"
    assert "master" in client.llms  # Should be re-instantiated
    assert client._active_llm_alias == "master"

def test_extra_llms_legacy_support(mock_binding_manager):
    """Ensures the legacy extra_llms parameter is converted to the two-tier system."""
    client = LollmsClient(
        llm_binding_name="ollama",
        extra_llms={
            "legacy_extra": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "legacy-model"}
            }
        }
    )

    assert "legacy_extra" in client.llm_model_profiles_registry
    # It should map to an auto-created binding profile
    b_name = client.llm_model_profiles_registry["legacy_extra"].binding_profile_name
    assert b_name in client.llm_binding_profiles_registry
    assert client.llm_binding_profiles_registry[b_name].binding_name == "ollama"