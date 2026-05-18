#!/usr/bin/env python3
"""
llama_cpp_server_ensemble_example.py
====================================
Demonstrates an **ensemble / mixture-of-experts** pattern:

1. User enters a prompt
2. The prompt is sent SIMULTANEOUSLY to two different models
3. Both models answer independently
4. A third "judge" model sees both answers and synthesizes a better one
5. The final refined response is shown

Architecture
------------
┌─────────────┐     ┌─────────────┐
│  Model A    │     │  Model B    │
│ Ministral 3B│     │ Llama 3.2 1B│
│  (fast)     │     │  (tiny)     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌─────────────────┐
       │  Both outputs   │
       │  + original     │
       │  prompt         │
       └────────┬────────┘
                ▼
       ┌─────────────────┐
       │  Model C        │
       │  (Ministral 3B) │
       │  "Judge"        │
       └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │  Final improved │
       │  answer         │
       └─────────────────┘

Requirements
------------
pip install lollms_client ascii_colors

Downloads: ~2.9 GB total (Ministral 3B + Llama 3.2 1B)
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

MODEL_A_ZOO_INDEX = 1   # Ministral-3-3B-Instruct-2512  (~2.2 GB)
MODEL_B_REPO = "unsloth/Qwen3.5-2B-MTP-GGUF"
MODEL_B_FILENAME = "Qwen3.5-2B-IQ4_NL.gguf"  # ~0.7 GB

# Generation parameters (tune these)
N_PREDICT = 4096
TEMPERATURE = 0.7
JUDGE_N_PREDICT = 4096
JUDGE_TEMPERATURE = 0.5   # Lower for more deterministic synthesis

# Shared binding config — both clients use the same registry directory
# so they can see each other's servers via the FileLock-protected JSON registry
SHARED_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 8192,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "max_active_models": 2,   # Allow both models to run simultaneously
    "idle_timeout": -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model_a(client: LollmsClient) -> str:
    """Downloads Model A if missing, loads it, returns filename."""
    zoo = client.llm.get_zoo()
    chosen = zoo[MODEL_A_ZOO_INDEX]
    fname = chosen["filename"]

    path = Path(SHARED_CONFIG["models_path"]) / fname
    if not path.exists():
        print(f"⬇️  Downloading {chosen['name']} ({chosen['size']}) ...")
        result = client.llm.download_from_zoo(MODEL_A_ZOO_INDEX)
        if not result.get("status"):
            raise RuntimeError(f"Download failed: {result.get('error')}")
        print("✅ Download complete.")
    else:
        print(f"📁 Model A already exists: {fname}")

    print(f"🔌 Loading Model A ({chosen['name']}) ...")
    if not client.llm.load_model(fname):
        raise RuntimeError("Failed to load Model A.")
    print("✅ Model A ready.")
    return fname


def ensure_model_b(client: LollmsClient) -> str:
    """Downloads Model B if missing, loads it, returns filename."""
    fname = MODEL_B_FILENAME
    display = "Llama-3.2-1B-Instruct-Q4_K_M"

    path = Path(SHARED_CONFIG["models_path"]) / fname
    if not path.exists():
        print(f"⬇️  Pulling {display} from Hugging Face ...")
        result = client.llm.pull_model(MODEL_B_REPO, fname)
        if not result.get("status"):
            raise RuntimeError(f"Pull failed: {result.get('error')}")
        print("✅ Pull complete.")
    else:
        print(f"📁 Model B already exists: {fname}")

    print(f"🔌 Loading Model B ({display}) ...")
    if not client.llm.load_model(fname):
        raise RuntimeError("Failed to load Model B.")
    print("✅ Model B ready.")
    return fname


def generate_with_model(client: LollmsClient, prompt: str, label: str) -> dict:
    """Sends *prompt* to *client* and returns timing + response."""
    print(f"[{label}] 📝 Generating ...")
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
        return {"label": label, "error": response["error"], "elapsed_sec": 0}

    return {
        "label": label,
        "response": response,
        "elapsed_sec": round(elapsed, 2),
    }


def build_judge_prompt(original_prompt: str, answer_a: str, answer_b: str) -> str:
    """Constructs the synthesis prompt for the judge model."""
    return (
        "You are an expert critic and synthesizer. Two AI assistants were asked "
        "the same question. Your job is to analyze both answers, take the best "
        "parts from each, correct any errors, and produce a single superior response.\n\n"
        "=== ORIGINAL QUESTION ===\n"
        f"{original_prompt}\n\n"
        "=== ANSWER FROM ASSISTANT A ===\n"
        f"{answer_a}\n\n"
        "=== ANSWER FROM ASSISTANT B ===\n"
        f"{answer_b}\n\n"
        "=== YOUR TASK ===\n"
        "Write a final answer that is better than both individual responses. "
        "Be concise but thorough. Do not mention the assistants or that you are "
        "synthesizing — just provide the best possible answer."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🧪 ENSEMBLE / MIXTURE-OF-EXPERTS DEMO")
    print("=" * 70)
    print("Two models answer independently, then a judge synthesizes the best.")
    print()

    # ── Create clients ────────────────────────────────────────────────────
    # We create TWO separate clients.  Because they share the same models_path,
    # they use the same FileLock + JSON registry and can see each other's servers.
    print("🚀 Initializing clients ...")
    client_a = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=SHARED_CONFIG,
        user_name="user",
        ai_name="assistant",
    )
    client_b = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=SHARED_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── Ensure both models are downloaded and loaded ──────────────────────
    model_a_fname = ensure_model_a(client_a)
    model_b_fname = ensure_model_b(client_b)

    # Show running servers
    print("\n📊 Active servers:")
    for srv in client_a.llm.ps():
        print(f"   - {srv['model_name']}  PID:{srv['pid']}  "
              f"Port:{srv['port']}  RSS:{srv['rss_mb']} MB")

    # ── Get user prompt ───────────────────────────────────────────────────
    print("\n" + "-" * 70)
    default_prompt = (
        "Explain the concept of recursion in programming. "
        "Give a simple example in Python."
    )
    user_input = input(
        f"Enter your prompt (press Enter for default):\n> "
    ).strip()
    prompt = user_input if user_input else default_prompt
    print("-" * 70)

    # ── Phase 1: Send to BOTH models concurrently ─────────────────────────
    print("\n🚀 Phase 1: Querying both models in parallel ...\n")
    phase1_t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(generate_with_model, client_a, prompt, "Model A")
        future_b = executor.submit(generate_with_model, client_b, prompt, "Model B")
        result_a = future_a.result(timeout=300)
        result_b = future_b.result(timeout=300)

    phase1_elapsed = time.time() - phase1_t0

    # ── Display individual answers ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 PHASE 1 RESULTS")
    print("=" * 70)

    for res in (result_a, result_b):
        if "error" in res:
            print(f"\n[{res['label']}] ❌ ERROR: {res['error']}")
            continue
        print(f"\n[{res['label']}] ✅ ({res['elapsed_sec']}s)")
        for line in str(res["response"]).strip().splitlines():
            print(f"   {line}")

    if "error" in result_a or "error" in result_b:
        print("\n❌ One or both models failed. Cannot proceed to synthesis.")
        return

    # ── Phase 2: Judge synthesizes the best answer ────────────────────────
    print("\n" + "=" * 70)
    print("🚀 Phase 2: Judge synthesis")
    print("=" * 70)
    print("Sending both answers to Model A (the judge) for refinement ...\n")

    judge_prompt = build_judge_prompt(
        prompt,
        result_a["response"],
        result_b["response"],
    )

    phase2_t0 = time.time()
    final_response = client_a.generate_text(
        prompt=judge_prompt,
        n_predict=JUDGE_N_PREDICT,
        temperature=JUDGE_TEMPERATURE,
        top_p=0.9,
        stream=False,
    )
    phase2_elapsed = time.time() - phase2_t0

    if isinstance(final_response, dict) and "error" in final_response:
        print(f"❌ Judge synthesis failed: {final_response['error']}")
        return

    print("✅ Final answer ready!\n")
    print("-" * 70)
    print(final_response)
    print("-" * 70)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 TIMING SUMMARY")
    print("=" * 70)
    print(f"   Model A generation:  {result_a['elapsed_sec']}s")
    print(f"   Model B generation:  {result_b['elapsed_sec']}s")
    print(f"   Phase 1 (parallel):  {phase1_elapsed:.1f}s  (wall-clock)")
    print(f"   Phase 2 (judge):     {phase2_elapsed:.1f}s")
    print(f"   Total:               {phase1_elapsed + phase2_elapsed:.1f}s")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n🧹 Unloading models ...")
    client_a.llm.unload_model(model_a_fname)
    client_b.llm.unload_model(model_b_fname)
    print("👋 Done!")


if __name__ == "__main__":
    main()
