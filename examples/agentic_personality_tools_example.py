#!/usr/bin/env python3
"""
agentic_personality_tools_example.py
====================================
A full agentic workflow demonstrating:
- Custom personality definition with research-focused system prompts
- Environment file (.env) configuration support with safe fallbacks
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
                              └──────────────────────────→┌
                                                          ↓
                                              ┌─────────────────┐
                                              │  Synthesis      │
                                              │  (Final Answer) │
                                              └─────────────────┘

Requirements
------------
pip install lollms_client ascii_colors
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ascii_colors import ASCIIColors
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality, AgentRole
from lollms_client.lollms_types import MSG_TYPE


ARXIV_TOOL_CONTENT = '''TOOL_LIBRARY_NAME = 'ArXiv Explorer'
TOOL_LIBRARY_DESC = 'Search scientific papers and pre-prints on ArXiv.'
TOOL_LIBRARY_ICON = '🔬'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'arxiv': '>=2.1.0'})

def tool_search_papers(query: str, count: int = 3, year_start: int = None, year_end: int = None):
    """
    Search for scientific papers on ArXiv based on a query.

    Args:
        query (str): Scientific keywords or paper ID to search for.
        count (int, optional): Number of papers to fetch. Defaults to 3.
        year_start (int, optional): Start year for filtering papers (inclusive).
        year_end (int, optional): End year for filtering papers (inclusive).
    """
    import arxiv
    import time
    try:
        if not query:
            return "Error: Query is required."

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
            time.sleep(1)  # Be polite to the API

        return "\\n\\n".join(results) if results else "No papers found matching the criteria."
    except Exception as e:
        return f"Error: {str(e)}"
'''

WIKIPEDIA_TOOL_CONTENT = '''TOOL_LIBRARY_NAME = 'Wikipedia Search'
TOOL_LIBRARY_DESC = 'Search and retrieve article summaries from Wikipedia.'
TOOL_LIBRARY_ICON = '📖'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'wikipedia': '>=1.4.0'})

def tool_search_wikipedia(query: str, max_results: int = 3):
    """
    Search Wikipedia for articles matching a query and return their summaries.

    Args:
        query (str): The search term or phrase to look up.
        max_results (int, optional): Maximum number of results to return. Defaults to 3.
    """
    import wikipedia
    import time
    try:
        if not query:
            return "Error: Query is required."
        
        time.sleep(1.0)  # Be polite to the API
        
        search_results = wikipedia.search(query)
        output = []
        for title in search_results[:max_results]:
            try:
                time.sleep(0.5)
                page = wikipedia.summary(title, sentences=5)
                output.append(f"--- {title} ---\\n{page}")
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    time.sleep(0.5)
                    page = wikipedia.summary(e.options[0], sentences=5)
                    output.append(f"--- {e.options[0]} ---\\n{page}")
            except Exception as inner_e:
                continue
                
        if not output:
            return "Error: Could not retrieve Wikipedia summaries. The API may be rate-limiting requests or no exact match was found. Please rely on the arxiv results or use another tool."
        return "\\n\\n".join(output)
    except Exception as e:
        return f"Error: Wikipedia search failed ({str(e)}). The API may be temporarily unavailable. Please proceed using the available ArXiv papers or other tools."
'''

TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"

def print_config_panel(client: LollmsClient):
    """Prints a formatted panel of all configuration variables at startup."""
    binding_name = client.llm.binding_name
    model_name = client.llm.model_name if hasattr(client.llm, 'model_name') else "N/A"
    
    config_content = (
        f"[cyan]LLM Binding:[/cyan]   {binding_name}\n"
        f"[cyan]Model Name:[/cyan]    {model_name}\n"
        f"[cyan]Tools Directory:[/cyan] {TOOLS_DIR}"
    )
    ASCIIColors.panel(config_content, title="[bold]CONFIGURATION SUMMARY[/bold]", border_style="cyan")

def ensure_tools() -> tuple:
    """Create tool files if they don't exist. Returns (arxiv_path, wiki_path)."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    arxiv_path = TOOLS_DIR / "arxiv_search.py"
    wiki_path = TOOLS_DIR / "wikipedia_search.py"
    
    if not arxiv_path.exists():
        ASCIIColors.info(f"📝 Creating arXiv tool: {arxiv_path}")
        arxiv_path.write_text(ARXIV_TOOL_CONTENT, encoding="utf-8")
    
    if not wiki_path.exists():
        ASCIIColors.info(f"📝 Creating Wikipedia tool: {wiki_path}")
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
        ASCIIColors.cyan(f"⬇️  [{pct:5.1f}%] {message}")
    elif status == "success":
        ASCIIColors.green(f"✅ {message}")
    elif status == "error":
        ASCIIColors.red(f"❌ ERROR: {message}")

def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        ASCIIColors.rich_print(chunk, end="", flush=True)
    return True

def print_tool_execution_summary(result: Dict[str, Any]):
    """Pretty-print the tool execution metadata."""
    ASCIIColors.rule("[bold cyan]📊 EXECUTION METADATA[/bold cyan]")
    
    metrics_table = ASCIIColors.table(
        "Metric", "Value",
        rows=[
            ["Total agentic rounds", str(result['rounds'])],
            ["Tool calls executed", str(len(result['tool_calls']))]
        ],
        title="[bold]Execution Summary[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(metrics_table)
    
    if not result['tool_calls']:
        ASCIIColors.yellow("  (No tools were called — model answered directly)")
        return
    
    tool_rows = []
    for tc in result['tool_calls']:
        tool_rows.append([f"Round {tc['round']}", tc['name'], json.dumps(tc['parameters'], indent=2, ensure_ascii=False)])
        
        tr = next((r for r in result['tool_results'] if r['round'] == tc['round']), None)
        if tr:
            res = tr['result']
            status = "✅ SUCCESS" if res.get('success') else "❌ FAILED"
            output = str(res.get('output', res.get('error', 'No output')))
            if len(output) > 300:
                output = output[:300] + f"... [{len(output) - 300} more chars]"
            tool_rows.append(["", "Result", f"{status}\n{output}"])
            
    tools_table = ASCIIColors.table(
        "Round", "Tool / Status", "Parameters / Output",
        rows=tool_rows,
        title="[bold magenta]🛠️ Tool Execution Details[/bold magenta]",
        box="round",
        show_lines=True
    )
    ASCIIColors.rich_print(tools_table)

    if result.get('pending_tool'):
        ASCIIColors.yellow(f"\n  ⏸️  PENDING (manual execution):")
        pt = result['pending_tool']
        ASCIIColors.cyan(f"     {pt['name']}({json.dumps(pt['parameters'])})")

def main():
    ASCIIColors.panel(
        "[bold]🔬 Research Agent — Multi-Step Reasoning with Personality + Tools[/bold]\n[dim]This demo shows an agent that:\n  1. Searches arXiv for recent academic papers\n  2. Searches Wikipedia for background concepts\n  3. Synthesizes a comprehensive report with citations[/dim]",
        title="[bold green]🤖 AGENTIC PERSONALITY DEMO[/bold green]",
        border_style="green"
    )

    try:
        with ASCIIColors.status("[cyan]Initializing client...[/cyan]", spinner="dots"):
            client = get_client_from_env()
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration failed: {e}")
        sys.exit(1)

    print_config_panel(client)

    arxiv_path, wiki_path = ensure_tools()
    ASCIIColors.green(f"📁 Tools ready:")
    ASCIIColors.cyan(f"   • {arxiv_path}")
    ASCIIColors.cyan(f"   • {wiki_path}")

    if client.llm.binding_name == "llama_cpp_server":
        model_name = client.llm.model_name
        models_path = client.llm.models_path
        model_path = Path(models_path) / model_name
        
        if not model_path.exists():
            ASCIIColors.panel(f"[yellow]⬇️  Downloading {model_name} ...[/yellow]", title="[bold]Model Download[/bold]", border_style="yellow")
            result = client.llm.download_from_zoo(MODEL_ZOO_INDEX, progress_callback=progress_callback)
            if not result.get("status"):
                ASCIIColors.red(f"❌ Download failed: {result.get('error')}")
                sys.exit(1)
            ASCIIColors.green("✅ Download complete.")
        else:
            ASCIIColors.green(f"\n📁 Model already exists: {model_name}")

        ASCIIColors.panel(f"[cyan]🔌 Loading model '{model_name}' ...[/cyan]", title="[bold]Model Loading[/bold]", border_style="cyan")
        t0 = time.time()
        success = client.llm.load_model(model_name)
        if not success:
            ASCIIColors.red("❌ Failed to load model.")
            sys.exit(1)
        load_time = time.time() - t0
        ASCIIColors.green(f"✅ Model loaded in {load_time:.1f}s")

        for srv in client.llm.ps():
            ASCIIColors.cyan(f"   Server: PID {srv['pid']} | Port {srv['port']} | RSS {srv['rss_mb']} MB")
    else:
        load_time = 0.0
        ASCIIColors.blue("\n☁️  Using remote binding. No model download or local loading required.")

    ASCIIColors.panel("[magenta]Creating ResearchAgent personality...[/magenta]", title="[bold]🎭 Personality[/bold]", border_style="magenta")
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

    ASCIIColors.panel("[blue]Configuring ResearchAgent personality with agentic parameters...[/blue]", title="[bold]🤖 Agent[/bold]", border_style="blue")
    personality.lollms_client = client
    personality.role = AgentRole.DOMAIN_EXPERT
    personality.model_params = {"temperature": 0.7}
    personality.max_tokens_per_turn = 4096
    ASCIIColors.cyan(f"   Agent: {personality.display_name} | Role: {personality.role} | ID: {personality._agent_id[:8]}")

    research_query = (
        "I want to understand the current state of reasoning in large language models. "
        "Specifically:\n"
        "1. Find 3 recent papers from 2024-2025 about LLM reasoning or chain-of-thought\n"
        "2. Look up background on 'chain-of-thought reasoning' on Wikipedia\n"
        "3. Synthesize a comprehensive overview that explains the concept, "
        "summarizes the latest research directions, and highlights open challenges"
    )

    ASCIIColors.panel(f"[yellow]{research_query}[/yellow]", title="[bold]📝 RESEARCH QUERY[/bold]", border_style="yellow")

    ASCIIColors.panel("[cyan]🔍 Starting multi-step research (streaming enabled)...[/cyan]", title="[bold]🚀 EXECUTION[/bold]", border_style="cyan")
    overall_t0 = time.time()
    
    try:
        result = personality.generate_with_tools(
            prompt=research_query,
            tools=[arxiv_path, wiki_path],
            system_prompt=personality.system_prompt,
            temperature=0.7,
            n_predict=4096,
            max_tool_rounds=10,
            streaming_callback=streaming_callback,
            auto_execute=True,
        )
    except Exception as e:
        ASCIIColors.red(f"\n💥 Fatal error during agent execution: {e}")
        sys.exit(1)

    overall_elapsed = time.time() - overall_t0

    print_tool_execution_summary(result)

    final_response = result.get("response", "")
    if not final_response.strip():
        ASCIIColors.panel(
            "[red]The agent did not produce a final response. This usually happens if the LLM connection fails or the model returns an empty string.[/red]",
            title="[bold red]⚠️ EMPTY RESPONSE[/bold red]",
            border_style="red"
        )
    else:
        ASCIIColors.rule("[bold cyan]📝 FINAL SYNTHESIZED REPORT[/bold cyan]")
        ASCIIColors.rich_print(final_response)

    perf_table = ASCIIColors.table(
        "Metric", "Value",
        rows=[
            ["Model load time", f"{load_time:.1f}s"],
            ["Total generation time", f"{overall_elapsed:.1f}s"],
            ["Agentic rounds", str(result['rounds'])],
            ["Tools utilized", str(len(result['tool_calls']))],
            ["Final response length", f"{len(final_response)} chars"]
        ],
        title="[bold]⏱️ PERFORMANCE SUMMARY[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(perf_table)

    ASCIIColors.panel("[magenta]🧹 Cleanup[/magenta]", title="[bold]🧹 CLEANUP[/bold]", border_style="magenta")
    if client.llm.binding_name == "llama_cpp_server" and hasattr(client.llm, "unload_model"):
        ASCIIColors.cyan("Unloading model...")
        client.llm.unload_model()
    ASCIIColors.green("👋 Done!")

if __name__ == "__main__":
    main()