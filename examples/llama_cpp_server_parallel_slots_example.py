#!/usr/bin/env python3
"""
llama_cpp_server_parallel_slots_example.py
=========================================
Demonstrates llama.cpp server's **native parallel generation**
using a SINGLE server process with multiple decoding slots.

Key concept: llama.cpp server supports `--parallel N` which creates N
independent KV-cache slots within ONE process.  This is far more
memory-efficient than running N separate server processes.

Comparison
----------
| Approach          | Processes | RAM per model | Best for |
|-------------------|-----------|---------------|----------|
| n_parallel=2      | 1         | 1×            | Same model, concurrent users |
| 2× load_model()   | 2         | 2×            | Different models |

Requirements
------------
pip install lollms_client ascii_colors

Downloads: ~2.2 GB (Ministral 3B, the smallest zoo model)
"""

import sys
import time
import concurrent.futures
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO_INDEX = 1   # Ministral-3-3B-Instruct-2512  (~2.2 GB)

# Generation parameters (tune these)
N_PREDICT = 256
TEMPERATURE = 0.7

BINDING_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 4096,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_parallel": 2,       # ← KEY: 2 concurrent decoding slots
    "batch_size": 512,
    "max_active_models": 1,
    "idle_timeout": -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (runs in a THREAD, not a process)
# ─────────────────────────────────────────────────────────────────────────────

def thread_generate(args: dict) -> dict:
    """
    Runs in a SEPARATE THREAD within the SAME process.
    All threads share the same LollmsClient and the same llama-server.
    The server handles concurrency via its internal slot mechanism.
    """
    worker_id = args["worker_id"]
    prompt = args["prompt"]

    # We use the SAME client instance — the OpenAI client is thread-safe
    client = args["client"]

    print(f"[Worker {worker_id}] 📝 Sending request ...")
    t0 = time.time()

    response = client.generate_text(
        prompt=prompt,
        n_predict=N_PREDICT,
        temperature=TEMPERATURE,
        top_p=0.9,
        stream=False,
    )

    elapsed = time.time() - t0

    if isinstance(response, dict) and "error" in response:
        return {"worker_id": worker_id, "error": response["error"]}

    return {
        "worker_id": worker_id,
        "elapsed_sec": round(elapsed, 2),
        "response": response,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🧪 PARALLEL SLOTS DEMO  (single server, n_parallel=2)")
    print("=" * 70)
    print("This uses ONE llama-server process with 2 decoding slots.")
    print("Both requests run concurrently inside the SAME process.\n")

    # ── Create client ─────────────────────────────────────────────────────
    print("🚀 Creating LollmsClient with n_parallel=2 ...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── Download model if missing ─────────────────────────────────────────
    zoo = client.llm.get_zoo()
    chosen = zoo[MODEL_ZOO_INDEX]
    model_filename = chosen["filename"]

    model_path = Path(BINDING_CONFIG["models_path"]) / model_filename
    if not model_path.exists():
        print(f"⬇️  Downloading {chosen['name']} ({chosen['size']}) ...")
        result = client.llm.download_from_zoo(MODEL_ZOO_INDEX)
        if not result.get("status"):
            print(f"❌ Download failed: {result.get('error')}")
            sys.exit(1)
        print("✅ Download complete.")
    else:
        print(f"📁 Model already exists: {model_filename}")

    # ── Load model (spawns ONE server with 2 slots) ───────────────────────
    print(f"\n🔌 Loading model with --parallel {BINDING_CONFIG['n_parallel']} ...")
    t0 = time.time()
    success = client.llm.load_model(model_filename)
    if not success:
        print("❌ Failed to load model.")
        sys.exit(1)
    print(f"✅ Model loaded in {time.time() - t0:.1f}s")

    # Show server info
    for srv in client.llm.ps():
        print(f"   Server: PID {srv['pid']} | Port {srv['port']} | "
              f"RSS {srv['rss_mb']} MB")

    # ── Prepare two concurrent generation tasks ───────────────────────────
    tasks = [
        {
            "worker_id": "A",
            "client": client,
            "prompt": (
                "You are a Python tutor. "
                "Explain list comprehensions in one paragraph."
            ),
        },
        {
            "worker_id": "B",
            "client": client,
            "prompt": (
                "You are a math tutor. "
                "Explain the Pythagorean theorem in one paragraph."
            ),
        },
    ]

    # ── Launch both workers CONCURRENTLY (threads, not processes) ─────────
    print("\n🚀 Launching TWO concurrent requests to the SAME server ...\n")
    overall_t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(thread_generate, t) for t in tasks]
        results = [f.result(timeout=300) for f in futures]

    overall_elapsed = time.time() - overall_t0

    # ── Display results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    for res in results:
        wid = res["worker_id"]
        if "error" in res:
            print(f"\n[Worker {wid}] ❌ ERROR: {res['error']}")
            continue

        print(f"\n[Worker {wid}] ✅ SUCCESS")
        print(f"   Latency:    {res['elapsed_sec']}s")
        print(f"   Response:")
        for line in str(res["response"]).strip().splitlines():
            print(f"      {line}")

    print("\n" + "=" * 70)
    print(f"⏱️  Total wall-clock time: {overall_elapsed:.1f}s")
    print("=" * 70)

    # ── Verify only ONE server process exists ─────────────────────────────
    print("\n📊 Server status (should show exactly ONE process):")
    servers = client.llm.ps()
    print(f"   Active servers: {len(servers)}")
    for srv in servers:
        print(f"   - {srv['model_name']}  PID:{srv['pid']}  "
              f"Port:{srv['port']}  RSS:{srv['rss_mb']} MB")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print("\n🧹 Unloading model ...")
    client.llm.unload_model()
    print("👋 Done!")


if __name__ == "__main__":
    main()
