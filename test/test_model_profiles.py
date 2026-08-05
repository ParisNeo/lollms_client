import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_core import LollmsClient, LollmsModelProfile

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
    profile = LollmsModelProfile(
        name="test",
        binding_name="ollama",
        binding_config={"model_name": "llama3"},
        is_default=True,
        vision_enabled=True,
        forced_context_size=8192,
        routing_config={"strategy": "tfidf"}
    )
    assert profile.name == "test"
    assert profile.binding_name == "ollama"
    assert profile.binding_config["model_name"] == "llama3"
    assert profile.is_default is True
    assert profile.vision_enabled is True
    assert profile.forced_context_size == 8192
    assert profile.routing_config["strategy"] == "tfidf"

def test_legacy_backward_compatibility(mock_binding_manager):
    """Ensures passing llm_binding_name registers a 'master' profile and instantiates it."""
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={"model_name": "llama3"}
    )
    
    assert "master" in client.llm_profiles_registry
    assert client.llm_profiles_registry["master"].is_default is True
    assert client.llm_profiles_registry["master"].binding_name == "ollama"
    
    # Should be eagerly instantiated
    assert "master" in client.llms
    assert client._active_llm_alias == "master"
    assert client.llm is not None

def test_lazy_loading_with_profiles(mock_binding_manager):
    """Ensures non-default profiles are registered but NOT instantiated at init."""
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={"model_name": "llama3"},
        llm_profiles={
            "coder": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "qwen-coder"},
                "vision_enabled": False
            },
            "vision": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llava"},
                "vision_enabled": True,
                "is_default": False
            }
        }
    )
    
    # Master should be instantiated
    assert "master" in client.llms
    assert client.llm is not None
    
    # Coder and Vision should be in registry but NOT instantiated
    assert "coder" in client.llm_profiles_registry
    assert "vision" in client.llm_profiles_registry
    assert "coder" not in client.llms
    assert "vision" not in client.llms
    
    # Switch to coder
    result = client.switch_model("coder")
    assert result is True
    assert "coder" in client.llms
    assert client._active_llm_alias == "coder"
    assert client.llm == client.llms["coder"]
    
    # Verify vision attributes were attached
    # (Note: Since we mock the binding, we just check it was instantiated)
    assert mock_binding_manager.return_value.create_binding.call_count >= 2

def test_switch_model_invalid_alias(mock_binding_manager):
    client = LollmsClient(llm_binding_name="ollama")
    result = client.switch_model("nonexistent_model")
    assert result is False
    assert client._active_llm_alias == "master" # Should remain on master

def test_switch_model_uses_cached_instance(mock_binding_manager):
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_profiles={"secondary": {"binding_name": "ollama"}}
    )
    
    # First switch instantiates
    client.switch_model("secondary")
    first_instance = client.llms["secondary"]
    
    # Switch back to master
    client.switch_model("master")
    
    # Switch to secondary again
    client.switch_model("secondary")
    second_instance = client.llms["secondary"]
    
    # Should be the exact same object (cached)
    assert first_instance is second_instance
    # Should not have called create_binding again for secondary
    # (call count should be 2: master at init + secondary first switch)

def test_profile_object_input(mock_binding_manager):
    """Tests passing actual LollmsModelProfile objects instead of dicts."""
    profile = LollmsModelProfile(
        name="custom",
        binding_name="ollama",
        binding_config={"model_name": "custom-model"},
        forced_context_size=4096
    )
    
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_profiles={"custom": profile}
    )
    
    assert "custom" in client.llm_profiles_registry
    assert client.llm_profiles_registry["custom"].forced_context_size == 4096

def test_update_llm_binding_replaces_master(mock_binding_manager):
    client = LollmsClient(llm_binding_name="ollama")
    
    # Update to a new binding
    client.update_llm_binding("openai", {"model_name": "gpt-4"})
    
    assert client.llm_profiles_registry["master"].binding_name == "openai"
    assert client.llm_profiles_registry["master"].binding_config["model_name"] == "gpt-4"
    assert "master" in client.llms  # Should be re-instantiated
    assert client._active_llm_alias == "master"

def test_extra_llms_legacy_support(mock_binding_manager):
    """Ensures the legacy extra_llms parameter is converted to profiles and eagerly instantiated."""
    client = LollmsClient(
        llm_binding_name="ollama",
        extra_llms={
            "legacy_extra": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "legacy-model"}
            }
        }
    )

    assert "legacy_extra" in client.llm_profiles_registry
    assert client.llm_profiles_registry["legacy_extra"].binding_name == "ollama"
    # Legacy extra_llms are eagerly instantiated for backward compatibility
    assert "legacy_extra" in client.llms