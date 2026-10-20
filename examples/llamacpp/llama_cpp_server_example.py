#!/usr/bin/env python3
"""
llama_cpp_server_example.py
============================
A complete, self-contained example showing how to:

1. Create a LollmsClient configured to use the llama_cpp_server binding
2. Download a small GGUF model from the built-in model zoo
3. Load the model into the llama.cpp server
4. Generate a simple "hello" response

Requirements
------------
pip install lollms_client ascii_colors

The first run will also auto-download llama.cpp binaries (~20-100 MB)
and the GGUF model (~2.2 GB for the smallest one).
"""

import sys
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE


def progress_callback(payload: dict):
    """
    Called during model download to report progress.
    """
    status = payload.get("status", "unknown")
    message = payload.get("message", "")
    completed = payload.get("completed", 0)
    total = payload.get("total", 100)

    if status == "downloading":
        pct = (completed / total * 100) if total else 0
        print(f"⬇️  [{pct:5.1f}%] {message}")
    elif status == "success":
        print(f"✅ {message}")
    elif status == "error":
        print(f"❌ ERROR: {message}")
    else:
        print(f"ℹ️  [{status.upper()}] {message}")


# ── Generation parameters (tune these) ───────────────────────────────────────
N_PREDICT = 128
TEMPERATURE = 0.7


def main():
    # ------------------------------------------------------------------
    # 1. Configure the llama_cpp_server binding
    # ------------------------------------------------------------------
    # The binding will auto-install llama.cpp binaries on first use.
    # models_path: where GGUF files are stored / looked up.
    # ctx_size: context window in tokens.
    # n_gpu_layers: -1 = offload all layers to GPU (if CUDA/Metal available).
    #               0  = CPU-only.
    # n_parallel: number of parallel decoding slots.
    # idle_timeout: auto-shutdown server after N seconds of inactivity (-1 = never).
    binding_config = {
        "models_path": "data/models/llama_cpp_models",   # GGUF storage
        "binaries_path": "data/bin/llm/llama_cpp_server", # llama.cpp binaries
        "ctx_size": 4096,
        "n_gpu_layers": -1,      # Change to 0 for CPU-only
        "n_threads": 4,          # CPU threads for generation
        "n_parallel": 1,
        "batch_size": 512,
        "idle_timeout": 300,     # Auto-unload after 5 min idle
    }

    print("🚀 Creating LollmsClient with llama_cpp_server binding...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=binding_config,
        user_name="user",
        ai_name="assistant",
    )

    # ------------------------------------------------------------------
    # 2. Pick & download a small model from the built-in zoo
    # ------------------------------------------------------------------
    # The binding ships with a curated zoo. Index 1 is the smallest:
    #   Ministral-3-3B-Instruct-2512-GGUF  (~2.2 GB Q4_K_M)
    #
    # If you already have a GGUF file locally, skip this step and set
    #   client.llm.load_model("your-model.gguf")
    print("\n📋 Available models in zoo:")
    zoo = client.llm.get_zoo()
    for idx, entry in enumerate(zoo):
        print(f"   [{idx}] {entry['name']:40s}  {entry['size']}")

    # Download the smallest model (index 1)
    model_index = 1
    chosen = zoo[model_index]
    print(f"\n⬇️  Downloading: {chosen['name']} ({chosen['size']})")
    result = client.llm.download_from_zoo(model_index, progress_callback=progress_callback)

    if not result.get("status"):
        print(f"Failed to download model: {result.get('error', result.get('message'))}")
        sys.exit(1)

    model_filename = chosen["filename"]
    print(f"\n📁 Model saved. Filename: {model_filename}")

    # ------------------------------------------------------------------
    # 3. Load the model into the llama.cpp server
    # ------------------------------------------------------------------
    print("\n🔌 Loading model into llama.cpp server (this may take a moment)...")
    success = client.llm.load_model(model_filename)
    if not success:
        print("❌ Failed to load model.")
        sys.exit(1)

    print("✅ Model loaded and server is ready!")

    # ------------------------------------------------------------------
    # 4. Generate a simple "hello" response
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📝 Prompt: 'Hello! Please introduce yourself in one sentence.'")
    print("=" * 60 + "\n")

    response = client.generate_text(
        prompt="Hello! Please introduce yourself in one sentence.",
        n_predict=N_PREDICT,
        temperature=TEMPERATURE,
        top_p=0.9,
        stream=False,  # Set True and pass streaming_callback for streaming
    )

    if isinstance(response, dict) and "error" in response:
        print(f"❌ Generation failed: {response['error']}")
        sys.exit(1)

    print("🤖 Response:")
    print(response)

    # ------------------------------------------------------------------
    # 5. (Optional) Inspect server status
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 Server status:")
    for proc in client.llm.ps():
        print(f"   - {proc['model_name']}  PID:{proc['pid']}  "
              f"Port:{proc['port']}  RSS:{proc['rss_mb']} MB")

    # ------------------------------------------------------------------
    # 6. Cleanup: unload the model when done
    # ------------------------------------------------------------------
    print("\n🧹 Unloading model...")
    client.llm.unload_model()
    print("👋 Done!")


if __name__ == "__main__":
    main()
