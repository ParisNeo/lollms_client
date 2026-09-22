---
title: "LollmsClient Creation and Two-Tier Profile Mastery"
description: "Authoritative doctrine for configuring and instantiating LollmsClient across all modalities using the Two-Tier Profile Architecture, global configuration files, and fast token estimation."
category: "core_architecture"
tags: [lollms_client, configuration, profiles, bindings, modalities, performance]
visibility: visible
modifiable: true
---

# LollmsClient Creation & Profile Architecture

The `LollmsClient` is the unified gateway to all large language model (LLM), vision, speech, audio, video, and communication backend engines in the LOLLMS ecosystem.

---

## 1. Universal Two-Tier Profile Architecture

To prevent duplicate connection definitions and enable lazy-loaded multi-model routing without wasting VRAM, `LollmsClient` separates configurations into two distinct layers:

```
┌────────────────────────────────────────────────────────┐
│ 1. Connection Layer (*_binding_profiles)               │
│    • Host address, API key, SSL cert, backend library  │
│    • Declared ONCE per server/provider                 │
└───────────────────────────┬────────────────────────────┘
                            │ Referenced by binding_profile_name
┌───────────────────────────▼────────────────────────────┐
│ 2. Execution Layer (*_model_profiles)                  │
│    • Specific model name, vision flags, context budget │
│    • Lazy-loaded: instantiated on first use            │
└────────────────────────────────────────────────────────┘
```

### Supported Modalities
Both layers exist across all seven modalities:
- **`llm`**: Large Language Models (Text & Chat)
- **`tti`**: Text-to-Image (Image Generation & Inpainting)
- **`tts`**: Text-to-Speech (Voice Synthesis)
- **`stt`**: Speech-to-Text (Audio Transcription)
- **`ttv`**: Text-to-Video (Video Generation)
- **`ttm`**: Text-to-Music (Audio & Music Synthesis)
- **`connection`**: Communication Bridges (Discord, Slack, Telegram, Webhooks)

---

## 2. Programmatic Client Instantiation

### A. Two-Tier Declarative Initialization (Recommended)

```python
from lollms_client import LollmsClient

client = LollmsClient(
    # ── 1. Connection Layer (Servers & Endpoints) ──
    llm_binding_profiles={
        "local_ollama": {
            "binding_name": "ollama",
            "binding_config": {
                "host_address": "http://localhost:11434",
                "verify_ssl_certificate": False,
            },
        },
        "cloud_openai": {
            "binding_name": "openai",
            "binding_config": {
                "service_key": "sk-proj-...",
                "host_address": "https://api.openai.com/v1",
            },
        },
    },

    # ── 2. Execution Layer (Model Targets) ──
    llm_model_profiles={
        "fast_coder": {
            "binding_profile_name": "local_ollama",
            "model_name": "qwen2.5-coder:7b",
            "forced_context_size": 32768,
            "is_default": True,  # Eagerly initialized on startup
        },
        "reasoning_master": {
            "binding_profile_name": "cloud_openai",
            "model_name": "o3-mini",
            "is_default": False,  # Lazy-loaded on demand
        },
    },

    # ── Tool Binding ──
    tools_binding_name="lcp",
    tools_binding_config={"tools_folders": ["./tools"]},
    debug=False,
)
```

### B. Fast Single-Model Initialization (Quick Scripts & CLI)

For standalone utilities, pass direct binding parameters. The client automatically registers them as the `"master"` connection and model profiles:

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={
        "host_address": "http://localhost:11434",
        "model_name": "qwen2.5-coder:7b",
        "ctx_size": 16384,
    },
)
```

---

## 3. Global Configuration Hierarchy (`config.yaml` & `.env`)

`LollmsClient` automatically resolves settings from standardized configuration files when parameters are omitted in code.

### Resolution Priority (Top to Bottom)
1. Direct keyword arguments passed to `LollmsClient(...)`
2. CLI argument overrides (`--profile`, `--model`, `--host`, `--api-key`)
3. `./.lollms_code/config.yaml` or `./.lollms_code/.env` (Project Workspace)
4. `~/.lollms_client/config.yaml` or `~/.lollms-client/.env` (User Global)
5. Operating system environment variables

### Structure of `~/.lollms_client/config.yaml`
```yaml
llm:
  bindings:
    local_ollama:
      binding_name: ollama
      host_address: http://localhost:11434
      verify_ssl_certificate: false
    openai_server:
      binding_name: openai
      service_key: sk-...
  profiles:
    default_coder:
      binding_alias: local_ollama
      model_name: qwen2.5-coder:7b
      forced_context_size: 32768
      is_default: true
    gpt4o:
      binding_alias: openai_server
      model_name: gpt-4o
      vision_enabled: true

tti:
  bindings:
    diffusers_local:
      binding_name: diffusers
      host_address: http://localhost:9642
  profiles:
    sdxl:
      binding_alias: diffusers_local
      model_name: stabilityai/sdxl-turbo
      is_default: true
```

---

## 4. Fast Token Estimation Heuristics

Remote tokenizers can introduce network latency before each generation turn. `LollmsClient` includes a local heuristic estimation engine designed for latency-sensitive applications.

```python
# Activate fast heuristic estimation (no server round-trips)
client.enable_fast_token_estimate(coefficient=1.0)

# Estimate tokens locally in sub-millisecond time
token_count = client.count_tokens("def calculate(x: int) -> int:\n    return x * 2")

# Disable and restore exact remote tokenization
client.disable_fast_token_estimate()
```

*Formula*:
$$\text{Tokens} \approx \left(\text{Word Count} + \frac{\text{Indentation Whitespace}}{4}\right) \times \text{coefficient}$$

---

## 5. Runtime Modality & Model Switching

Switch models mid-task without reconstructing the client:

```python
# Switch active LLM profile
client.switch_model("reasoning_master")

# Switch active Image Generation (TTI) engine
client.switch_tti("sdxl")

# Switch active Speech Synthesis (TTS) engine
client.switch_tts("voice_narrator")
```

---

## 6. Vision Capability Discovery & Automatic VLM Routing

When images are passed to a non-vision active model, `LollmsClient` discovers and routes image descriptions through an available Vision-Language Model (VLM):

```python
# Check if active model natively supports images
if not client.has_vision_capability():
    # Automatically extracts image description using available VLM
    description = client.get_or_generate_image_description(image_bytes)
```