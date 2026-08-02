# 🚀 LollmsClient Examples

This directory contains runnable examples demonstrating how to use the `lollms_client` library for LLM generation, agentic workflows, and multimodal operations.

## Quick Start (`.env` Configuration)

All examples in this folder are designed to be environment-agnostic. They read configuration variables from a `.env` file located in this same directory. This allows you to seamlessly switch between local bindings (like `llama_cpp_server`) and remote APIs (like `openai`, `groq`, or `ollama`) without changing the code.

### 1. Generate your `.env` file

You can use the provided interactive wizard to automatically generate your `.env` file:

```bash
python examples/configure_env.py
```

Alternatively, you can copy the template manually:

```bash
cp examples/.env.example examples/.env
```

### 2. Configure your Binding

Open the `.env` file and set the variables according to your target binding. Here is a typical configuration for a remote **Ollama** binding:

```env
LLM_BINDING_NAME=ollama
MODEL_NAME=llama3
HOST_ADDRESS=http://localhost:11434
VERIFY_SSL=false
```

And here is a configuration for a gated cloud service like **OpenAI**:

```env
LLM_BINDING_NAME=openai
MODEL_NAME=gpt-4o-mini
HOST_ADDRESS=https://api.openai.com/v1
API_KEY=sk-your-api-key-here
VERIFY_SSL=true
```

### 3. Write an Example Script

When writing your own scripts, use the following boilerplate to safely load the environment and construct the `LollmsClient`. This pattern ensures your script runs out-of-the-box with safe fallback defaults, isolates execution logic inside a `main()` function, and protects the execution via the `if __name__ == "__main__":` guard.

```python
import os
import sys
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient

# 1. Safely load the .env file
# Falls back gracefully if python-dotenv is not installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# 2. Read configuration with safe defaults for local execution
LLM_BINDING_NAME = os.getenv("LLM_BINDING_NAME", "llama_cpp_server")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf")
HOST_ADDRESS = os.getenv("HOST_ADDRESS", "http://localhost:11434")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("true", "1", "yes")
API_KEY = os.getenv("API_KEY")

def main():
    # 3. Dynamically construct the binding configuration
    if LLM_BINDING_NAME == "llama_cpp_server":
        # Configuration for local bindings
        BINDING_CONFIG = {
            "models_path": os.getenv("MODELS_PATH", str(Path.home() / ".lollms_hub" / "models")),
            "binaries_path": os.getenv("BINARIES_PATH", str(Path.home() / ".lollms_hub" / "bin")),
            "ctx_size": int(os.getenv("CONTEXT_SIZE", "8192")),
            "n_gpu_layers": int(os.getenv("N_GPU_LAYERS", "-1")),
        }
    else:
        # Configuration for remote bindings (ollama, openai, groq, etc.)
        BINDING_CONFIG = {
            "model_name": MODEL_NAME,
            "host_address": HOST_ADDRESS,
            "verify_ssl_certificate": VERIFY_SSL,
        }
        # Conditionally inject API key for gated services
        if API_KEY:
            BINDING_CONFIG["service_key"] = API_KEY

    # 4. Initialize the client
    client = LollmsClient(
        llm_binding_name=LLM_BINDING_NAME,
        llm_binding_config=BINDING_CONFIG,
        debug=True
    )

    # 5. Generate text
    response = client.generate_text(
        prompt="Explain the concept of sovereignty in software architecture.",
        temperature=0.7
    )
    print(response)

if __name__ == "__main__":
    main()
```

## Available Examples

- `configure_env.py`: Interactive wizard to generate the `.env` file.
- `test_discussion.py`: A smoke test for the `LollmsDiscussion` package, demonstrating in-memory and DB-backed chats, as well as artefact management.
- `agentic_personality_tools_example.py`: A full agentic workflow demonstrating custom personality definitions, multi-step reasoning, and external tool chaining (arXiv + Wikipedia).
- `llama_cpp_server_example.py`: Basic usage specific to the local `llama_cpp_server` binding.
- `ollama_vision_flux_klein_loop.py`: Demonstrates a multimodal loop using Ollama for vision and Flux for image generation.