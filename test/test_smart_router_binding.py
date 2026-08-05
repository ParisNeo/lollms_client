import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.llm_bindings.smart_router import LollmsSmartBinding

@pytest.fixture
def mock_binding_manager():
    with patch("lollms_client.llm_bindings.smart_router.LollmsLLMBindingManager") as manager:
        def create_side_effect(binding_name, **kwargs):
            mock = MagicMock()
            mock.binding_name = binding_name
            mock.model_name = kwargs.get("model_name", "test-model")
            mock.generate_text.return_value = "Mocked response"
            mock.generate_from_messages.return_value = "Mocked message response"
            mock.count_tokens.return_value = 10
            mock.get_ctx_size.return_value = 8192
            return mock
        
        manager.return_value.create_binding.side_effect = create_side_effect
        yield manager

def test_smart_router_initialization(mock_binding_manager):
    """Tests if the router correctly instantiates child bindings from profiles."""
    router = LollmsSmartBinding(
        model_profiles={
            "child_a": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "routing_profile": {"description": "general tasks"}
            },
            "child_b": {
                "binding_name": "openai",
                "binding_config": {"model_name": "gpt-4"},
                "routing_profile": {"description": "complex reasoning"}
            }
        }
    )
    
    assert len(router.child_bindings) == 2
    assert "child_a" in router.child_bindings
    assert "child_b" in router.child_bindings
    # Ensure vision flags are attached
    assert router.child_bindings["child_a"].vision_enabled == False

def test_tfidf_subject_routing(mock_binding_manager):
    """Verifies that TF-IDF similarity correctly routes prompts to the best matching model."""
    router = LollmsSmartBinding(
        model_profiles={
            "coder": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "qwen-coder"},
                "routing_profile": {"description": "python javascript typescript code programming debugging"}
            },
            "writer": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "routing_profile": {"description": "creative writing stories poems emails blog marketing"}
            }
        }
    )
    
    # Act: Coding prompt
    chosen = router._select_model("Write a python script to sort a list.")
    assert chosen == "coder"
    
    # Act: Writing prompt
    chosen = router._select_model("Write a poem about the ocean.")
    assert chosen == "writer"

def test_complexity_routing(mock_binding_manager):
    """Verifies that complexity heuristics route to the correct tier."""
    router = LollmsSmartBinding(
        model_profiles={
            "simple": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3.2:3b"},
                "routing_profile": {"description": "general", "complexity_tier": 1}
            },
            "complex": {
                "binding_name": "openai",
                "binding_config": {"model_name": "gpt-4"},
                "routing_profile": {"description": "general", "complexity_tier": 3}
            }
        }
    )
    
    # Simple prompt
    assert router._select_model("What is 2+2?") == "simple"
    
    # Complex prompt (trigger word 'architect' + length)
    long_prompt = "Architect a distributed system. " + " ".join(["word"] * 150)
    assert router._select_model(long_prompt) == "complex"

def test_vision_hard_filter(mock_binding_manager):
    """Ensures non-vision models are filtered out when images are present."""
    router = LollmsSmartBinding(
        model_profiles={
            "text_only": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "vision_enabled": False,
                "routing_profile": {"description": "general text"}
            },
            "vlm": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llava"},
                "vision_enabled": True,
                "routing_profile": {"description": "general image"}
            }
        }
    )
    
    # Text prompt -> should pick text_only (assuming default weights favor subject match)
    # Both have 'general', so either could win. Let's make text slightly more specific.
    # Actually, let's just test the image filter.
    
    # Image prompt -> must pick vlm
    chosen = router._select_model("Describe this image.", images=["base64data..."])
    assert chosen == "vlm"

def test_strategy_weights_selection(mock_binding_manager):
    """Tests if cost_optimized strategy correctly penalizes expensive models."""
    router = LollmsSmartBinding(
        routing_strategy="cost_optimized",
        model_profiles={
            "cheap": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "routing_profile": {"description": "general", "cost_per_1k_tokens": 0.0}
            },
            "expensive": {
                "binding_name": "openai",
                "binding_config": {"model_name": "gpt-4"},
                "routing_profile": {"description": "general", "cost_per_1k_tokens": 0.03}
            }
        }
    )
    
    # Both match subject 'general', but cost strategy should heavily favor 'cheap'
    chosen = router._select_model("Tell me a joke.")
    assert chosen == "cheap"

def test_generate_text_delegation(mock_binding_manager):
    """Tests that generate_text correctly delegates to the chosen child."""
    router = LollmsSmartBinding(
        model_profiles={
            "child_a": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "routing_profile": {"description": "general tasks"}
            }
        }
    )
    
    response = router.generate_text("Hello")
    assert response == "Mocked response"
    router.child_bindings["child_a"].generate_text.assert_called_once_with("Hello")

def test_generate_from_messages_delegation(mock_binding_manager):
    """Tests that generate_from_messages correctly delegates to the chosen child."""
    router = LollmsSmartBinding(
        model_profiles={
            "child_a": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "routing_profile": {"description": "general tasks"}
            }
        }
    )
    
    messages = [{"role": "user", "content": "Hello"}]
    response = router.generate_from_messages(messages)
    assert response == "Mocked message response"
    router.child_bindings["child_a"].generate_from_messages.assert_called_once_with(messages)

def test_graceful_degradation_to_text_model(mock_binding_manager):
    """Tests that the router falls back to the highest priority text model if no VLM matches."""
    router = LollmsSmartBinding(
        model_profiles={
            "text_only": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "vision_enabled": False,
                "routing_profile": {"description": "general text", "priority": 1}
            }
        }
    )

    # Instead of raising an error, the router should gracefully degrade to the text_only model
    response = router.generate_text("Describe this image.", images=["base64data..."])
    assert response == "Mocked response"

    # Verify the text_only model was selected despite the image prompt
    selected_model = router._select_model("Describe this image.", images=["base64data..."])
    assert selected_model == "text_only"