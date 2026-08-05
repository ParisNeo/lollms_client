# 🚀 LollmsClient Examples

This directory contains runnable examples demonstrating how to use the `lollms_client` library for LLM generation, agentic workflows, and multimodal operations.

## Quick Start (`.env` Configuration)

All examples in this folder are designed to be environment-agnostic. They leverage a 4-tier configuration resolution protocol to load LLM settings without requiring hardcoded credentials. The library checks for configurations in the following order:

1. CLI arguments (if applicable)
2. `.env` file in the current working directory
3. `.env` file in `~/.lollms-client/`
4. OS environment variables
5. Interactive Wizard fallback

### 1. Generate your `.env` file

You can use the shared interactive wizard to automatically generate your `.env` file in `~/.lollms-client/`. The wizard supports configuring multiple bindings (LLM, TTI, TTS, STT, TTM, TTV) dynamically:

```bash
python -m lollms_client.lollms_config_cli_env
```

### 2. Configure your Bindings

Open the generated `.env` file and set the variables according to your target bindings. All binding parameters are namespaced with their type prefix (e.g., `LLM_`, `TTI_`, `TTS_`).

Here is a typical configuration for an **Ollama** LLM binding and a **Diffusers** TTI binding:

```env
LLM_BINDING_NAME=ollama
LLM_MODEL_NAME=llama3
LLM_HOST_ADDRESS=http://localhost:11434
LLM_VERIFY_SSL_CERTIFICATE=true

TTI_BINDING_NAME=diffusers
TTI_MODEL_NAME=runwayml/stable-diffusion-v1-5
```

### 3. Write an Example Script

When writing your own scripts, use the `get_client_from_env` helper to seamlessly execute the 4-tier protocol and instantiate the client. You can selectively load bindings by passing boolean flags, preventing the unnecessary initialization of heavy multimodal libraries.

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client import LollmsClient

def main():
    # 1. Automatically resolve .env file or launch the wizard, then build the client
    # Only load the LLM and TTI bindings. TTS, STT, TTM, and TTV will be ignored.
    client = get_client_from_env(create_llm=True, create_tti=True)

    # 2. Generate text
    response = client.generate_text(
        prompt="Explain the concept of sovereignty in software architecture.",
        temperature=0.7
    )
    print(response)

if __name__ == "__main__":
    main()
```

## Available Examples

- `agentic_personality_tools_example.py`: A full agentic workflow demonstrating custom personality definitions, multi-step reasoning, and external tool chaining (arXiv + Wikipedia).
- `universal_profiles_example.py`: Demonstrates the Universal Lazy Profile System (multi-model LLM/TTI routing) dynamically loaded from the `.env` file. Shows how to switch between models configured via the wizard.
- `example.env`: A template configuration file showing the syntax for defining Master Bindings and Universal Lazy Profiles.
- `test_discussion.py`: A smoke test for the `LollmsDiscussion` package, demonstrating in-memory and DB-backed chats, as well as artefact management.
- `llama_cpp_server_example.py`: Basic usage specific to the local `llama_cpp_server` binding.
- `ollama_vision_flux_klein_loop.py`: Demonstrates a multimodal loop using Ollama for vision and Flux for image generation.