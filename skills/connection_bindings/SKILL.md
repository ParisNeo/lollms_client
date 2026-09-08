---
name: LoLLMS Connection Bindings
description: Teaches how to use LoLLMS connection bindings to send messages to communication services (Discord, Telegram, Slack, WhatsApp, webhooks). Covers the two-tier profile system for connections and how LollmsPersonality auto-discovers connection tools.
author: LoLLMS
version: 1.0.0
category: lollms_client/connections
---

# LoLLMS Connection Bindings

Connection bindings bridge LoLLMS to communication platforms (Discord, Telegram, Slack, WhatsApp, webhooks, etc.). They allow a `LollmsClient` or `LollmsPersonality` to send messages to configured channels, following the same two-tier profile architecture as LLM/TTI/TTS bindings.

## 1. Architecture

```
LollmsClient
├── connection_binding_manager (discovery + instantiation)
├── connection (active binding instance)
├── connections {} (multi-instance registry)
├── connection_binding_profiles_registry {} (connection layer)
└── connection_model_profiles_registry {} (instance layer)
```

**Two-Tier Profiles** for connections:
- **Connection Profile** (`LOLLMS_CONNECTION_BINDINGS_<alias>_`): API key, host, platform specifics.
- **Instance Profile** (`LOLLMS_CONNECTION_PROFILES_<alias>_`): `INSTANCE_NAME` = which channel/chat to target.

## 2. Configuring a Connection Binding

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gemma3:latest", "host_address": "http://localhost:11434"},
    connection_binding_profiles={
        "my-webhook": {
            "binding_name": "generic_webhook",
            "binding_config": {
                "service_key": "https://hooks.slack.com/services/T.../B.../XXX",
                "timeout": 30,
            },
            "is_default": True,
        }
    },
    connection_model_profiles={
        "default-channel": {
            "binding_profile_name": "my-webhook",
            "model_name": "general",
            "is_default": True,
        },
        "alerts-channel": {
            "binding_profile_name": "my-webhook",
            "model_name": "alerts",
            "is_default": False,
        },
    },
)
```

## 3. Direct Send (No LLM)

```python
result = client.send_connection_message("Hello from LoLLMS!")
if result["sent"]:
    print(f"Sent to {result['channel']}: {result['message_id']}")

# Switch to a different channel
client.switch_connection("alerts-channel")
client.send_connection_message("⚠️ Critical alert!")
```

## 4. Agent-Powered Send (LLM decides when to send)

A `LollmsPersonality` with a `LollmsClient` that has connection profiles automatically discovers `tool_send_connection`:

```python
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality, CapabilityFlags

persona = LollmsPersonality(
    name="Notifier",
    system_prompt="You can send messages to communication channels when asked.",
    lollms_client=client,
    capabilities=CapabilityFlags(enable_networking=True),
)

result = persona.chat(
    prompt="Please notify the team that the build succeeded.",
    max_nb_rounds=1,
)
# The LLM will emit: <tool>{"name": "