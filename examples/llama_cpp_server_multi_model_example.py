#!/usr/bin/env python3
"""
llama_cpp_server_multi_model_example.py
======================================
An advanced example demonstrating:

1. Two LollmsClient instances with DIFFERENT small models
2. Automatic model download if missing (zoo + pull_model)
3. Concurrent generation across TWO PROCESSES
4. Two separate llama-server processes running simultaneously
5. Process-safe server registry (FileLock + JSON registry)

Requirements
------------
pip install lollms_client ascii_colors

This downloads TWO small models:
  • Ministral-3-3B  (~2.2 GB from built-in zoo)
  • Llama-3.2-1B    (~0.7 GB from Hugging Face via pull_model)

Total download: ~2.9 GB.  Both servers run concurrently on separate ports.
"""

import sys
import time
import multiprocessing
from pathlib import Path
from typing import Dict, Any

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Worker A: built-in zoo model (index 1 = Ministral 3B)
MODEL_A_ZOO_INDEX = 1

# Generation parameters (tune these)
N_PREDICT = 4096
TEMPERATURE = 0.7

# Worker B: pulled from Hugging Face — a tiny 1B model
MODEL_B_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_B_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

BINDING_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 4096,
    "n_gpu_layers": -1,      # 0 for CPU-only
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "max_active_models": 2,  # Allow BOTH models to run simultaneously
    "idle_timeout": -1,      # Never auto-unload during the demo
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (must be top-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def worker_generate(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs in a SEPARATE PROCESS.
    Creates its own LollmsClient, ensures the model is downloaded,
    loads it, generates text, and returns the result.
    """
    worker_id = args["worker_id"]
    prompt = args["prompt"]

    print(f"[Worker {worker_id}] 🚀 Starting in PID {multiprocessing.current_process().pid}")

    # Each process builds its own client (binding is process-safe via FileLock)
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── Download model if missing ─────────────────────────────────────────
    if worker_id == "A":
        # Worker A uses the built-in zoo
        zoo = client.llm.get_zoo()
        chosen = zoo[MODEL_A_ZOO_INDEX]
        model_filename = chosen["filename"]
        model_display_name = chosen["name"]

        model_path = Path(BINDING_CONFIG["models_path"]) / model_filename
        if not model_path.exists():
            print(f"[Worker {worker_id}] ⬇️  Downloading {chosen['name']} from zoo...")
            result = client.llm.download_from_zoo(MODEL_A_ZOO_INDEX)
            if not result.get("status"):
                return {
                    "worker_id": worker_id,
                    "error": f"Download failed: {result.get('error', result.get('message'))}",
                }
            print(f"[Worker {worker_id}] ✅ Download complete.")
        else:
            print(f"[Worker {worker_id}] 📁 Model already exists locally.")

    else:
        # Worker B pulls a tiny model from Hugging Face
        model_filename = MODEL_B_FILENAME
        model_display_name = "Llama-3.2-1B-Instruct-Q4_K_M"

        model_path = Path(BINDING_CONFIG["models_path"]) / model_filename
        if not model_path.exists():
            print(f"[Worker {worker_id}] ⬇️  Pulling {model_display_name} from Hugging Face...")
            result = client.llm.pull_model(MODEL_B_REPO, MODEL_B_FILENAME)
            if not result.get("status"):
                return {
                    "worker_id": worker_id,
                    "error": f"Pull failed: {result.get('error', result.get('message'))}",
                }
            print(f"[Worker {worker_id}] ✅ Pull complete.")
        else:
            print(f"[Worker {worker_id}] 📁 Model already exists locally.")

    # ── Load the model (spawns llama-server on a free port) ───────────────
    print(f"[Worker {worker_id}] 🔌 Loading model '{model_filename}' ...")
    t0 = time.time()
    success = client.llm.load_model(model_filename)
    if not success:
        return {"worker_id": worker_id, "error": "Failed to load model."}
    load_time = time.time() - t0
    print(f"[Worker {worker_id}] ✅ Model loaded in {load_time:.1f}s")

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"[Worker {worker_id}] 📝 Generating ...")
    t0 = time.time()
    response = client.generate_text(
        prompt=prompt,
        n_predict=N_PREDICT,
        temperature=TEMPERATURE,
        top_p=0.9,
        stream=False,
    )
    gen_time = time.time() - t0

    if isinstance(response, dict) and "error" in response:
        return {"worker_id": worker_id, "error": response["error"]}

    # Gather server info
    ps_info = client.llm.ps()

    return {
        "worker_id": worker_id,
        "model_name": model_display_name,
        "model_filename": model_filename,
        "load_time_sec": round(load_time, 2),
        "gen_time_sec": round(gen_time, 2),
        "response": response,
        "server_info": ps_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🧪 MULTI-MODEL CONCURRENT GENERATION DEMO")
    print("=" * 70)
    print(f"Worker A: Ministral-3-3B  (~2.2 GB, from built-in zoo)")
    print(f"Worker B: Llama-3.2-1B    (~0.7 GB, from Hugging Face)")
    print(f"max_active_models: {BINDING_CONFIG['max_active_models']}")
    print("-" * 70 + "\n")

    # ── Prepare two independent generation tasks ──────────────────────────
    tasks = [
        {
            "worker_id": "A",
            "prompt": (
                "You are a helpful coding assistant. "
                "Write a Python function that computes the factorial of a number. "
                "Keep it short and add a docstring."
            ),
        },
        {
            "worker_id": "B",
            "prompt": (
                "You are a creative writer. "
                "Write a haiku about artificial intelligence. "
                "Only output the haiku, nothing else."
            ),
        },
    ]

    # ── Launch both workers IN PARALLEL (separate processes) ────────────
    print("🚀 Launching TWO workers in parallel processes...\n")
    overall_t0 = time.time()

    with multiprocessing.Pool(processes=2) as pool:
        async_result = pool.map_async(worker_generate, tasks)
        results = async_result.get(timeout=600)  # 10 min max

    overall_elapsed = time.time() - overall_t0

    # ── Display results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    for res in results:
        wid = res["worker_id"]
        if "error" in res:
            print(f"\n[Worker {wid}] ❌ ERROR: {res['error']}")
            continue

        print(f"\n[Worker {wid}] ✅ SUCCESS")
        print(f"   Model:      {res['model_name']}")
        print(f"   Load time:  {res['load_time_sec']}s")
        print(f"   Gen time:   {res['gen_time_sec']}s")
        print(f"   Response:")
        for line in str(res["response"]).strip().splitlines():
            print(f"      {line}")

        for srv in res.get("server_info", []):
            print(f"   Server:     PID {srv['pid']} | Port {srv['port']} | "
                  f"RSS {srv['rss_mb']} MB")

    print("\n" + "=" * 70)
    print(f"⏱️  Total wall-clock time (parallel): {overall_elapsed:.1f}s")
    print("=" * 70)

    # ── Cleanup: inspect then unload all ──────────────────────────────────
    print("\n🧹 Inspecting running servers before cleanup...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
    )

    print("\n   Active servers seen from main process:")
    servers = client.llm.ps()
    if servers:
        for srv in servers:
            print(f"      - {srv['model_name']} on port {srv['port']} "
                  f"(PID {srv['pid']}, RSS {srv['rss_mb']} MB)")
    else:
        print("      (none)")

    # Unload both models
    for res in results:
        if "error" not in res:
            fname = res["model_filename"]
            print(f"\n   Unloading '{fname}' ...")
            client.llm.unload_model(fname)

    print("\n👋 Demo complete!")


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    multiprocessing.set_start_method("spawn", force=True)
    main()
