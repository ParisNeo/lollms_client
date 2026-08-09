# Universal Lazy Profiles & Multi-Model Routing

The `lollms_client` library features a powerful, memory-efficient multi-binding architecture called **Universal Lazy Profiles**. This system allows you to define declarative registries of `LollmsBindingProfile` configurations for *any* modality (LLM, TTI, TTS, STT, TTV, TTM).

Instead of eagerly instantiating all models or engines at startup (which wastes RAM and VRAM), only the binding marked as `is_default` is loaded. Other bindings are instantiated lazily *on-demand* when you switch to them. This system is 100% backward compatible.

---

## 1. The `LollmsBindingProfile` Dataclass

At the heart of the system is the `LollmsBindingProfile` dataclass. It is a declarative configuration container for a single binding instance.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class LollmsBindingProfile:
    """A declarative profile for any modality binding (LLM, TTI, TTS, etc.)."""
    name: str
    binding_name: str
    binding_config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    vision_enabled: bool = False
    forced_context_size: Optional[int] = None
    routing_config: Optional[Dict[str, Any]] = None
```

**Attributes:**
*   `name` (str): The unique alias for this profile (e.g., "local_coder", "cloud_vision"). You will use this name to switch to it.
*   `binding_name` (str): The actual binding implementation to use (e.g., "ollama", "openai", "diffusers").
*   `binding_config` (Dict): The configuration dictionary passed to the binding's constructor (e.g., `{"model_name": "llama3", "host_address": "..."}`).
*   `is_default` (bool): If `True`, this profile is eagerly instantiated at `LollmsClient` startup.
*   `vision_enabled` (bool): A hint flag indicating if this model supports vision (used by Smart Router and VLM tools).
*   `forced_context_size` (Optional[int]): Manually overrides the context size detection protocol.
*   `routing_config` (Optional[Dict]): Advanced configuration for the `smart_router` binding (e.g., cost, latency, complexity).

---

## 2. Multi-Model LLM Routing

You can define multiple LLM profiles to handle different domains, complexity tiers, or cost constraints. 

### Example: Local Coder + Cloud Vision

```python
from lollms_client import LollmsClient, LollmsBindingProfile

# 1. Define LLM profiles declaratively (None of these are instantiated yet)
llm_profiles = {
    "cloud_vision": LollmsBindingProfile(
        name="cloud_vision",
        binding_name="openai",
        binding_config={"model_name": "gpt-4o", "service_key": "your-key"},
        vision_enabled=True,
        is_default=True  # This one will be loaded at startup
    ),
    "local_coder": LollmsBindingProfile(
        name="local_coder",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "qwen2.5-coder:7b"},
        forced_context_size=32768
    ),
    "local_fast": LollmsBindingProfile(
        name="local_fast",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "llama3.2:3b"}
    )
}

# 2. Initialize the client (Only "cloud_vision" is instantiated)
lc = LollmsClient(llm_profiles=llm_profiles)

# 3. Use the default model
response1 = lc.generate_text("Explain quantum physics.")

# 4. Dynamically switch to the local coder (Instantiated on-the-fly and cached)
lc.switch_model("local_coder")
response2 = lc.generate_text("Write a Python script to sort a list.")

# 5. Switch back to the default cloud vision model (Retrieved from cache, no re-instantiation)
lc.switch_model("cloud_vision")
```

### Backward Compatibility

If you use the legacy `llm_binding_name` or `extra_llms` parameters, they are automatically registered as the `"master"` profiles and eagerly instantiated, ensuring older code runs without modification.

```python
# Legacy style - still works perfectly
lc = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "llama3"}
)

# This creates a "master" profile internally
assert "master" in lc.llm_profiles_registry
```

---

## 3. Multi-Engine Modality Routing (TTI, TTS, etc.)

The profile system is universal. You can define profiles for image generation engines, text-to-speech, etc., and switch between a local Stable Diffusion model and a cloud DALL-E API seamlessly.

### Example: Local Stable Diffusion + Cloud DALL-E

```python
# Define TTI profiles
tti_profiles = {
    "local_sd": LollmsBindingProfile(
        name="local_sd",
        binding_name="diffusers",
        binding_config={"model_name": "stable-diffusion-v1-5"},
        is_default=True
    ),
    "cloud_dalle": LollmsBindingProfile(
        name="cloud_dalle",
        binding_name="openai",
        binding_config={"model_name": "dall-e-3", "service_key": "your-key"}
    )
}

lc = LollmsClient(llm_binding_name="ollama", tti_profiles=tti_profiles)

# Generate with default local SD
img_bytes = lc.generate_image("A cyberpunk cat")

# Switch to DALL-E 3 for a specific prompt
lc.switch_tti("cloud_dalle")
img_bytes = lc.generate_image("A hyperrealistic oil painting of a dog")
```

**Available Switch Methods:**
*   `lc.switch_model(alias)`
*   `lc.switch_tti(alias)`
*   `lc.switch_tts(alias)`
*   `lc.switch_stt(alias)`
*   `lc.switch_ttv(alias)`
*   `lc.switch_ttm(alias)`

---

## 4. The Smart Router Binding (`smart_router`)

For automated routing, the `smart_router` meta-binding evaluates incoming prompts using **TF-IDF (subject matching)**, **complexity heuristics**, and **weighted constraints (cost/latency)** to delegate generation to the optimal child model automatically.

### Configuration

The `smart_router` accepts a dictionary of `model_profiles` in its `llm_binding_config`. Each profile contains a `routing_profile` dictionary with metadata for the router's decision engine.

```python
from lollms_client import LollmsClient

router_config = {
    "routing_strategy": "cost_optimized",  # or "balanced", "quality_optimized"
    "model_profiles": {
        "cheap_fast": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5:3b"},
            "routing_profile": {
                "description": "fast simple tasks math formatting",
                "cost_per_1k_tokens": 0.0,
                "avg_latency_ms": 20,
                "complexity_tier": 1
            }
        },
        "smart_coder": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5-coder:7b"},
            "routing_profile": {
                "description": "python javascript rust code debugging algorithms",
                "cost_per_1k_tokens": 0.0,
                "avg_latency_ms": 40,
                "complexity_tier": 2
            }
        }
    }
}

lc = LollmsClient(
    llm_binding_name="smart_router",
    llm_binding_config=router_config
)

# The router automatically selects "cheap_fast" for simple prompts
response = lc.generate_text("What is 2+2?")

# The router automatically selects "smart_coder" for coding prompts
response = lc.generate_text("Write a Python script to sort a list.")
```

### Routing Strategies
*   `balanced`: Weights subject match, complexity, and cost equally.
*   `cost_optimized`: Heavily penalizes expensive models unless strictly necessary.
*   `quality_optimized`: Prefers higher complexity tiers and lower latency regardless of cost.

---

## 5. VLM as a Tool (`vlm_query`)

When using a text-only LLM (or Smart Router without a VLM child), you can enable VLM collaboration via the `vlm_query` LCP tool. This allows the text LLM to delegate visual analysis to a secondary VLM on-demand.

The `LollmsDiscussion` chat loop conditionally mounts this tool as a fallback mechanism. It is only mounted if the active binding **lacks** vision capabilities, another instantiated binding **supports** vision, and the user explicitly passes `enable_vlm_query=True` to `discussion.chat()`. The LLM can then call `tool_vlm_query(image_index, query)` on-demand to ask a VLM specific questions about images, preserving context and saving compute.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager

# Client with a Smart Router that includes a VLM profile
lc = LollmsClient(
    llm_binding_name="smart_router",
    llm_binding_config={
        "model_profiles": {
            "text_llm": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "vision_enabled": False
            },
            "vlm": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llava"},
                "vision_enabled": True
            }
        }
    }
)

db_manager = LollmsDataManager("sqlite:///discussion.db")
discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db_manager)

# The user sends an image and asks a question
discussion.chat(
    user_message="Look at the diagram in the image. What are the main components?",
    images=["base64_encoded_image_data..."],
    enable_vlm_query=True, # CRITICAL: Explicitly enable the fallback tool
    max_reasoning_steps=10
)

# The LLM (llama3) will realize it needs vision, call `tool_vlm_query(0, "Identify the main components in this diagram.")`,
# receive the text description from the VLM (llava), and synthesize the final answer.
```