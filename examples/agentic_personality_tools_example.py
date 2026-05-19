#!/usr/bin/env python3
"""
agentic_personality_tools_example.py
====================================
A full agentic workflow demonstrating:
- Custom personality definition with research-focused system prompts
- Multiple file-based tools in lollms format (arXiv + Wikipedia)
- Multi-step reasoning with automatic tool chaining
- Agent.generate_with_tools() with rich execution metadata

Architecture
------------
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Query    │────→│  ResearchAgent  │────→│  arXiv Search   │
│  (LLM reasoning)│     │  (Personality)  │     │  (Tool #1)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                           │
                              │←──────────────────────────┘
                              │     (papers returned)
                              ↓
                       ┌─────────────────┐
                       │ Wikipedia Search│
                       │   (Tool #2)     │
                       └─────────────────┘
                              │
                              └──────────────────────────→┐
                                                          ↓
                                              ┌─────────────────┐
                                              │  Synthesis      │
                                              │  (Final Answer) │
                                              └─────────────────┘

Requirements
------------
pip install lollms_client ascii_colors

Downloads: ~2.2 GB (Ministral 3B Q4_K_M)
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (lollms format)
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
            - year_start (int, optional): Start year for filtering (inclusive)
            - year_end (int, optional): End year for filtering (inclusive)
    """
    import arxiv
    try:
        query = args.get('query')
        if not query:
            return "Error: Query is required."

        count = args.get('count', 3)
        year_start = args.get('year_start')
        year_end = args.get('year_end')

        search = arxiv.Search(query=query, max_results=100)
        client = arxiv.Client()

        results = []
        for res in client.results(search):
            try:
                pub_year = int(res.published.strftime('%Y'))
            except Exception:
                pub_year = None

            if year_start is not None and pub_year is not None and pub_year < year_start:
                continue
            if year_end is not None and pub_year is not None and pub_year > year_end:
                continue

            authors = ', '.join(author.name for author in res.authors)
            pub_date = res.published.strftime('%Y-%m-%d') if res.published else "Unknown date"
            results.append(
                f"[{res.entry_id}] {res.title}\\n"
                f"Authors: {authors}\\n"
                f"Published: {pub_date}\\n"
                f"Abstract: {res.summary[:500]}..."
            )

            if len(results) >= count:
                break

        return "\\n\\n".join(results) if results else "No papers found matching the criteria."
    except Exception as e:
        return f"Error: {str(e)}"
'''

WIKIPEDIA_TOOL_CONTENT = '''TOOL_LIBRARY_NAME = 'Wikipedia Search'
TOOL_LIBRARY_DESC = 'Search and retrieve article summaries from Wikipedia.'
TOOL_LIBRARY_ICON = '📖'

def init_tool_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'wikipedia': '>=1.4.0'})

def tool_search_wikipedia(args: dict):
    """
    Search Wikipedia for articles matching a query and return summaries.

    Args:
        args: dict with keys:
            - query (str): The search term or phrase
            - max_results (int, optional): Maximum number of results (default: 3)
    """
    import wikipedia
    try:
        query = args.get('query')
        limit = args.get('max_results', 3)
        search_results = wikipedia.search(query)
        output = []
        for title in search_results[:limit]:
            try:
                page = wikipedia.summary(title, sentences=5)
                output.append(f"--- {title} ---\\n{page}")
            except Exception:
                continue
        return "\\n\\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error: {str(e)}"
'''


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO_INDEX = 1   # Ministral-3-3B-Instruct-2512

BINDING_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 8192,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "idle_timeout": 300,
}

TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_tools() -> tuple:
    """Create tool files if they don't exist. Returns (arxiv_path, wiki_path)."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    arxiv_path = TOOLS_DIR / "arxiv_search.py"
    wiki_path = TOOLS_DIR / "wikipedia_search.py"
    
    if not arxiv_path.exists():
        print(f"📝 Creating arXiv tool: {arxiv_path}")
        arxiv_path.write_text(ARXIV_TOOL_CONTENT, encoding="utf-8")
    
    if not wiki_path.exists():
        print(f"📝 Creating Wikipedia tool: {wiki_path}")
        wiki_path.write_text(WIKIPEDIA_TOOL_CONTENT, encoding="utf-8")
    
    return str(arxiv_path), str(wiki_path)


def progress_callback(payload: dict):
    """Called during model download."""
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


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


def print_tool_execution_summary(result: Dict[str, Any]):
    """Pretty-print the tool execution metadata."""
    print("\n" + "=" * 70)
    print("📊 EXECUTION METADATA")
    print("=" * 70)
    print(f"Total agentic rounds:  {result['rounds']}")
    print(f"Tool calls executed:   {len(result['tool_calls'])}")
    
    if not result['tool_calls']:
        print("  (No tools were called — model answered directly)")
        return
    
    for tc in result['tool_calls']:
        print(f"\n  🔹 Round {tc['round']}: {tc['name']}")
        print(f"     Parameters: {json.dumps(tc['parameters'], indent=2, ensure_ascii=False)}")
        
        # Find matching result
        tr = next((r for r in result['tool_results'] if r['round'] == tc['round']), None)
        if tr:
            res = tr['result']
            status = "✅ SUCCESS" if res.get('success') else "❌ FAILED"
            output = str(res.get('output', res.get('error', 'No output')))
            # Truncate very long outputs for display
            if len(output) > 300:
                output = output[:300] + f"... [{len(output) - 300} more chars]"
            print(f"     Result: {status}")
            print(f"     Output: {output}")
    
    if result.get('pending_tool'):
        print(f"\n  ⏸️  PENDING (manual execution):")
        pt = result['pending_tool']
        print(f"     {pt['name']}({json.dumps(pt['parameters'])})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🔬 Research Agent — Multi-Step Reasoning with Personality + Tools")
    print("=" * 70)
    print("This demo shows an agent that:")
    print("  1. Searches arXiv for recent academic papers")
    print("  2. Searches Wikipedia for background concepts")
    print("  3. Synthesizes a comprehensive report with citations")
    print()

    # ── 1. Ensure tools exist ─────────────────────────────────────────
    arxiv_path, wiki_path = ensure_tools()
    print(f"📁 Tools ready:")
    print(f"   • {arxiv_path}")
    print(f"   • {wiki_path}")

    # ── 2. Create LollmsClient ────────────────────────────────────────
    print("\n🚀 Creating LollmsClient with llama_cpp_server binding...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── 3. Download model if missing ──────────────────────────────────
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

    # ── 4. Load the model ─────────────────────────────────────────────
    print(f"\n🔌 Loading model '{model_filename}' ...")
    t0 = time.time()
    success = client.llm.load_model(model_filename)
    if not success:
        print("❌ Failed to load model.")
        sys.exit(1)
    load_time = time.time() - t0
    print(f"✅ Model loaded in {load_time:.1f}s")

    for srv in client.llm.ps():
        print(f"   Server: PID {srv['pid']} | Port {srv['port']} | RSS {srv['rss_mb']} MB")

    # ── 5. Create Research Personality ──────────────────────────────────
    print("\n🎭 Creating ResearchAgent personality...")
    personality = LollmsPersonality(
        name="ResearchAgent",
        author="lollms-client",
        category="Research",
        description=(
            "An expert research assistant specializing in computer science "
            "and artificial intelligence literature synthesis."
        ),
        system_prompt=(
            "You are ResearchAgent, an expert research assistant with deep "
            "knowledge of computer science and artificial intelligence.\n\n"
            "Your workflow for answering research queries:\n"
            "1. SEARCH arXiv for the latest academic papers on the topic\n"
            "2. SEARCH Wikipedia for foundational concepts and background\n"
            "3. SYNTHESIZE findings into a comprehensive, well-structured report\n"
            "4. CITE sources clearly and highlight key insights\n\n"
            "Rules:\n"
            "• Always use available tools — never rely solely on training data\n"
            "• Chain multiple searches if the topic has sub-components\n"
            "• Be thorough but concise in synthesis\n"
            "• Use markdown formatting for readability"
        ),
    )

    # ── 6. Create Agent ───────────────────────────────────────────────
    print("🤖 Creating Agent with personality and tools...")
    agent = Agent(
        lc=client,
        personality=personality,
        name="ResearchAgent",
        role=AgentRole.DOMAIN_EXPERT,
        model_params={"temperature": 0.7},
        max_tokens_per_turn=4096,
        metadata={"specialization": "AI/CS research synthesis"},
    )
    print(f"   Agent: {agent.display_name} | Role: {agent.role} | ID: {agent._agent_id[:8]}")

    # ── 7. Define multi-step research query ─────────────────────────────
    research_query = (
        "I want to understand the current state of reasoning in large language models. "
        "Specifically:\n"
        "1. Find 3 recent papers from 2024-2025 about LLM reasoning or chain-of-thought\n"
        "2. Look up background on 'chain-of-thought reasoning' on Wikipedia\n"
        "3. Synthesize a comprehensive overview that explains the concept, "
        "summarizes the latest research directions, and highlights open challenges"
    )

    print("\n" + "-" * 70)
    print("📝 RESEARCH QUERY:")
    print("-" * 70)
    print(research_query)
    print("-" * 70)

    # ── 8. Execute agentic generation ─────────────────────────────────
    print("\n🔍 Starting multi-step research (streaming enabled)...\n")
    print("=" * 70)
    print("🤖 AGENT RESPONSE (streaming):")
    print("=" * 70)

    overall_t0 = time.time()
    
    result = agent.generate_with_tools(
        prompt=research_query,
        tools=[arxiv_path, wiki_path],
        system_prompt=personality.system_prompt,
        temperature=0.7,
        n_predict=4096,
        max_tool_rounds=10,
        streaming_callback=streaming_callback,
        auto_execute=True,
    )

    overall_elapsed = time.time() - overall_t0

    print("\n")  # Newline after streaming

    # ── 9. Display execution metadata ─────────────────────────────────
    print_tool_execution_summary(result)

    # ── 10. Display final report ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📝 FINAL SYNTHESIZED REPORT")
    print("=" * 70)
    print(result["response"])

    # ── 11. Performance summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("⏱️  PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Model load time:       {load_time:.1f}s")
    print(f"Total generation time:   {overall_elapsed:.1f}s")
    print(f"Agentic rounds:          {result['rounds']}")
    print(f"Tools utilized:          {len(result['tool_calls'])}")
    print(f"Final response length:   {len(result['response'])} chars")

    # ── 12. Cleanup ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🧹 Cleanup")
    print("=" * 70)
    print("Unloading model...")
    client.llm.unload_model()
    print("👋 Done!")


if __name__ == "__main__":
    main()
