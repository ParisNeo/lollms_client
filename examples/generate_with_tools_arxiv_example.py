#!/usr/bin/env python3
"""
generate_with_tools_arxiv_example.py
====================================
Demonstrates LollmsClient.generate_with_tools() with the llama_cpp_server binding.

This example:
1. Creates an arXiv search tool in lollms format (Python script with tool_* functions)
2. Downloads and loads the Mistral-3-3B-Instruct model
3. Uses generate_with_tools() to perform an agentic arXiv search with tool calling

Requirements
------------
pip install lollms_client ascii_colors

Downloads: ~2.2 GB (Mistral 3B Q4_K_M)
"""

import sys
import json
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO_INDEX = 1   # Ministral-3-3B-Instruct-2512 in the built-in zoo

BINDING_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 8192,        # Larger context for tool use
    "n_gpu_layers": -1,      # 0 for CPU-only
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "idle_timeout": 300,     # Auto-unload after 5 min idle
}

# Where to store our custom tool
TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"
ARXIV_TOOL_PATH = TOOLS_DIR / "arxiv_search.py"


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv tool definition (lollms format)
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_TOOL_CONTENT = '''TOOL_LIBRARY_NAME = 'ArXiv Explorer'
TOOL_LIBRARY_DESC = 'Search scientific papers and pre-prints on ArXiv.'
TOOL_LIBRARY_ICON = '🔬'

def init_tool_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'arxiv': '>=2.1.0'})

def tool_search_papers(args: dict):
    """
    Search for scientific papers on ArXiv.

    Args:
        args: dict with keys:
            - query (str): Scientific keywords or paper ID
            - count (int, optional): Number of papers to fetch (default: 3)
            - year_start (int, optional): Start year for filtering papers (inclusive)
            - year_end (int, optional): End year for filtering papers (inclusive)
    """
    import arxiv
    try:
        query = args.get('query')
        if not query:
            return "Error: Query is required."

        count = args.get('count', 3)
        year_start = args.get('year_start')
        year_end = args.get('year_end')

        # Fetch more results initially to allow for filtering
        search = arxiv.Search(query=query, max_results=100)
        client = arxiv.Client()

        results = []
        for res in client.results(search):
            # Extract year from published date (format: YYYY-MM-DD)
            try:
                pub_year = int(res.published.strftime('%Y'))
            except Exception:
                pub_year = None

            # Apply year filters if specified
            if year_start is not None and pub_year is not None and pub_year < year_start:
                continue
            if year_end is not None and pub_year is not None and pub_year > year_end:
                continue

            authors = ', '.join(author.name for author in res.authors)
            # Format the date as YYYY-MM-DD
            pub_date = res.published.strftime('%Y-%m-%d') if res.published else "Unknown date"
            results.append(
                f"[{res.entry_id}] {res.title}\\n"
                f"Authors: {authors}\\n"
                f"Published: {pub_date}\\n"
                f"Abstract: {res.summary[:500]}..."
            )

            # Stop if we have enough results
            if len(results) >= count:
                break

        return "\\n\\n".join(results) if results else "No papers found matching the criteria."
    except Exception as e:
        return f"Error: {str(e)}"
'''


def ensure_arxiv_tool():
    """Create the arXiv tool file if it doesn't exist."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if not ARXIV_TOOL_PATH.exists():
        print(f"📝 Creating arXiv tool at: {ARXIV_TOOL_PATH}")
        ARXIV_TOOL_PATH.write_text(ARXIV_TOOL_CONTENT, encoding="utf-8")
        print("✅ ArXiv tool created.")
    else:
        print(f"📁 ArXiv tool already exists: {ARXIV_TOOL_PATH}")
    return str(ARXIV_TOOL_PATH)


def progress_callback(payload: dict):
    """Called during model download to report progress."""
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


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to the console as they arrive."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🔬 LollmsClient.generate_with_tools() — ArXiv Search Demo")
    print("=" * 70)
    print()

    # ── 1. Ensure the arXiv tool exists ─────────────────────────────────
    tool_path = ensure_arxiv_tool()

    # ── 2. Create LollmsClient with llama_cpp_server binding ────────────
    print("\n🚀 Creating LollmsClient with llama_cpp_server binding...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── 3. Download model if missing ──────────────────────────────────────
    zoo = client.llm.get_zoo()
    chosen = zoo[MODEL_ZOO_INDEX]
    model_filename = chosen["filename"]

    model_path = Path(BINDING_CONFIG["models_path"]) / model_filename
    if not model_path.exists():
        print(f"\n⬇️  Downloading {chosen['name']} ({chosen['size']}) ...")
        result = client.llm.download_from_zoo(MODEL_ZOO_INDEX, progress_callback=progress_callback)
        if not result.get("status"):
            print(f"❌ Download failed: {result.get('error')}")
            sys.exit(1)
        print("✅ Download complete.")
    else:
        print(f"\n📁 Model already exists: {model_filename}")

    # ── 4. Load the model ─────────────────────────────────────────────────
    print(f"\n🔌 Loading model '{model_filename}' ...")
    success = client.llm.load_model(model_filename)
    if not success:
        print("❌ Failed to load model.")
        sys.exit(1)
    print("✅ Model loaded and server is ready!")

    # Show server info
    for srv in client.llm.ps():
        print(f"   Server: PID {srv['pid']} | Port {srv['port']} | RSS {srv['rss_mb']} MB")

    # ── 5. Define the search query ──────────────────────────────────────
    search_query = (
        "Find recent papers about large language models and reasoning. "
        "I want 3 papers from 2024 or 2025."
    )

    print("\n" + "=" * 70)
    print("📝 USER PROMPT:")
    print("=" * 70)
    print(search_query)
    print()

    # ── 6. Call generate_with_tools ─────────────────────────────────────
    print("=" * 70)
    print("🤖 AGENT RESPONSE (streaming):")
    print("=" * 70)

    result = client.generate_with_tools(
        prompt=search_query,
        tools=[tool_path],           # Pass the path to our arXiv tool script
        system_prompt=(
            "You are a helpful research assistant. "
            "When the user asks about scientific papers, use the arxiv_search tool. "
            "After receiving tool results, summarize the findings clearly."
        ),
        temperature=0.7,
        n_predict=2048,
        max_tool_rounds=5,
        streaming_callback=streaming_callback,
        auto_execute=True,
    )

    print("\n")  # Newline after streaming

    # ── 7. Display results metadata ─────────────────────────────────────
    print("=" * 70)
    print("📊 RESULT METADATA")
    print("=" * 70)
    print(f"Rounds:        {result['rounds']}")
    print(f"Tool calls:    {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"  • Round {tc['round']}: {tc['name']}({json.dumps(tc['parameters'])})")
    print(f"Tool results:  {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        res = tr["result"]
        status = "✅" if res.get("success") else "❌"
        print(f"  • {status} {tr['name']}: {str(res.get('output', res))[:100]}...")

    # ── 8. Final response ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📝 FINAL RESPONSE (captured):")
    print("=" * 70)
    print(result["response"])

    # ── 9. Cleanup ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🧹 Cleanup")
    print("=" * 70)
    print("Unloading model...")
    client.llm.unload_model()
    print("👋 Done!")


if __name__ == "__main__":
    main()
