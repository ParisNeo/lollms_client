#!/usr/bin/env python3
"""
handbag_agent_example.py
========================
A comprehensive example demonstrating the Handbag architecture.

The Handbag is a self-contained folder that carries ALL of an agent's resources:
- Personalities (SOUL.md)
- Tools (LCP .py files)
- Skills (SKILL.md files)
- RAG sources (text documents)
- Memory database (SQLite)
- Workspace (isolated working directory)

This example:
1. Programmatically creates a complete Handbag structure on disk.
2. Populates it with a custom personality, a custom tool, a RAG document, and a skill.
3. Instantiates an Agent using ONLY the handbag_path (auto-configuration).
4. Runs a multi-step agentic task that utilizes the handbag's resources.

Requirements
------------
pip install lollms_client ascii_colors python-dotenv
"""

import sys
import os
import shutil
import json
from pathlib import Path

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── ENVIRONMENT LOADING ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality, AgentRole, CapabilityFlags, Handbag
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Read variables from the environment with safe defaults
LLM_BINDING_NAME = os.getenv("LLM_BINDING_NAME", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
HOST_ADDRESS = os.getenv("HOST_ADDRESS", "http://localhost:11434")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("true", "1", "yes")
API_KEY = os.getenv("API_KEY")

# Define the path where our Handbag will be created
HANDBAG_DIR = PROJECT_ROOT / "data_workspace" / "my_research_handbag"


# ─────────────────────────────────────────────────────────────────────────────
# Handbag Content Definitions
# ─────────────────────────────────────────────────────────────────────────────

SOUL_MD_CONTENT = """---
name: ResearchBot
author: lollms-client
version: '1.0'
category: research
description: An autonomous research agent that uses tools and RAG to answer questions.
---

You are ResearchBot, an autonomous AI agent specialized in data retrieval and synthesis.
Your workspace contains a set of project rules in your RAG knowledge base.
You have access to a custom tool to fetch information.

Follow these steps to answer user questions:
1. Consult your RAG knowledge base (project_rules.md) for project constraints.
2. Use the `tool_fetch_mock_data` tool to retrieve any specific data you need.
3. Synthesize the information and provide a comprehensive answer.
Always cite the rules from your knowledge base when making decisions.
"""

PROJECT_RULES_MD_CONTENT = """# Project Rules and Constraints

1. All responses must be concise and structured.
2. The user's name is Alex, and the project codename is "Operation Sunrise".
3. You must always use the `tool_fetch_mock_data` tool if the user asks for data, rather than hallucinating.
4. If the user asks for a summary, you must output exactly 3 bullet points.
"""

WEB_FETCH_TOOL_CONTENT = """TOOL_LIBRARY_NAME = 'Mock Data Fetcher'
TOOL_LIBRARY_DESC = 'Fetches mock JSON data for the research agent.'
TOOL_LIBRARY_ICON = '🔌'

def init_tools_library() -> None:
    \"\"\"No external dependencies required.\"\"\"
    pass

def tool_fetch_mock_data(data_type: str = "metrics"):
    \"\"\"
    Fetches mock JSON data based on the requested data type.

    Args:
        data_type (str, optional): The type of data to fetch ('metrics' or 'users'). Defaults to 'metrics'.
    \"\"\"
    import json
    if data_type == "users":
        return json.dumps([
            {"id": 1, "name": "Alice", "role": "Admin"},
            {"id": 2, "name": "Bob", "role": "User"}
        ])
    else:
        return json.dumps({
            "cpu_usage": "12%",
            "memory_usage": "45%",
            "active_connections": 1024,
            "status": "Operational"
        })
"""

SKILL_MD_CONTENT = """---
title: "Data Analysis Pattern"
description: "Standard workflow for analyzing datasets with pandas"
category: "data_analysis"
tags: [python, pandas, csv, data]
always_visible: false
---

# Data Analysis Pattern

## Workflow
1. Load with `pd.read_csv(file_name, encoding='utf-8')`
2. Inspect with `df.head()`, `df.info()`, `df.describe()`
3. Filter with boolean indexing
4. Aggregate with `df.groupby()`
5. Save results with `df.to_csv()`
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_handbag_structure():
    """Programmatically creates the handbag folder and populates it with resources."""
    ASCIIColors.cyan(f"\n📁 Creating Handbag structure at: {HANDBAG_DIR}")
    
    # Clean up existing handbag for a fresh run
    if HANDBAG_DIR.exists():
        shutil.rmtree(HANDBAG_DIR)
    
    # 1. Create the base structure using the Handbag utility
    Handbag.create_structure(HANDBAG_DIR, name="Research Handbag")
    
    # 2. Populate the Personality (SOUL.md)
    pers_dir = HANDBAG_DIR / "personalities" / "researcher"
    pers_dir.mkdir(parents=True, exist_ok=True)
    (pers_dir / "SOUL.md").write_text(SOUL_MD_CONTENT, encoding="utf-8")
    
    # 3. Populate the RAG knowledge base
    rag_dir = HANDBAG_DIR / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    (rag_dir / "project_rules.md").write_text(PROJECT_RULES_MD_CONTENT, encoding="utf-8")
    
    # 4. Populate the Custom Tool
    tools_dir = HANDBAG_DIR / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "web_fetch.py").write_text(WEB_FETCH_TOOL_CONTENT, encoding="utf-8")
    
    # 5. Populate a Skill
    skills_dir = HANDBAG_DIR / "skills" / "data_analysis"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD_CONTENT, encoding="utf-8")
    
    # 6. Update manifest to set default personality
    manifest_path = HANDBAG_DIR / "handbag.yaml"
    try:
        import yaml
        manifest = {
            "name": "Research Handbag",
            "version": "1.0",
            "description": "A handbag containing all resources for the research agent.",
            "default_personality": "researcher",
            "skills_mode": "mixed"
        }
        manifest_path.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False), encoding="utf-8")
    except ImportError:
        manifest_path.write_text("name: Research Handbag\\nversion: '1.0'\\ndefault_personality: researcher\\nskills_mode: mixed\\n", encoding="utf-8")

    ASCIIColors.green("✅ Handbag populated with Personality, RAG, Tools, and Skills.")


def print_config_panel():
    """Prints a formatted panel of all configuration variables at startup."""
    panel_width = 64
    print("\n" + "┌" + "─" * panel_width + "┐")
    print("│" + " HANDBAG EXAMPLE CONFIGURATION".center(panel_width) + "│")
    print("├" + "─" * panel_width + "┤")

    rows = [
        ("LLM Binding", LLM_BINDING_NAME),
        ("Model Name", MODEL_NAME),
        ("Host Address", HOST_ADDRESS),
        ("Verify SSL", VERIFY_SSL),
        ("API Key", "Loaded" if API_KEY else "None"),
        ("Handbag Path", HANDBAG_DIR),
    ]

    for label, value in rows:
        line = f" {label}: {value}"
        if len(line) > panel_width:
            line = line[: panel_width - 3] + "..."
        print(f"│{line.ljust(panel_width)}│")

    print("└" + "─" * panel_width + "┘\n")


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to the console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🎒 Handbag Agent Example — Unified Resource Architecture")
    print("=" * 70)
    print("This demo shows how to package an agent's entire context into a single folder.")
    print("The Agent will automatically load its personality, tools, RAG, and memory.")
    print()

    # ── 1. Print Configuration Panel ───────────────────────────────────
    print_config_panel()

    # ── 2. Create the Handbag on disk ──────────────────────────────────
    create_handbag_structure()

    # ── 3. Create LollmsClient ─────────────────────────────────────────
    print(f"\n🚀 Creating LollmsClient with {LLM_BINDING_NAME} binding...")
    
    llm_config = {
        "model_name": MODEL_NAME,
        "host_address": HOST_ADDRESS,
        "verify_ssl_certificate": VERIFY_SSL
    }
    if API_KEY:
        llm_config["service_key"] = API_KEY

    client = LollmsClient(
        llm_binding_name=LLM_BINDING_NAME,
        llm_binding_config=llm_config,
        user_name="user",
        ai_name="assistant",
    )

    # ── 4. Instantiate Personality using ONLY the Handbag ──────────────
    print("\n🤖 Instantiating Personality from Handbag...")

    caps = CapabilityFlags(
        enable_code_execution=False,
        enable_sub_agents=False,
        enable_skill_loading=True,  # Allow loading the skill we created
        enable_skill_creation=False,
        skills_mode="mixed",        # Match the manifest setting
    )

    agent = LollmsPersonality.from_handbag(str(HANDBAG_DIR), lollms_client=client)
    agent.role = AgentRole.DOMAIN_EXPERT
    agent.capabilities = caps
    agent.max_tokens_per_turn = 2048

    print(f"✅ Agent initialized: {agent.display_name}")
    print(f"   • Personality: {agent.name}")
    print(f"   • Has RAG: {agent.has_data}")
    print(f"   • Memory DB: {agent.memory_manager is not None}")

    # ── 5. Define a multi-step prompt ──────────────────────────────────
    # This prompt requires the agent to:
    # 1. Use its custom tool (tool_fetch_mock_data)
    # 2. Consult its RAG knowledge base (project_rules.md) for the user's name
    # 3. Format the output according to the rules (3 bullet points)
    prompt = (
        "Hello! Can you fetch the current system metrics for me, and then summarize them? "
        "Please also remind me of my name and the project codename based on our project rules."
    )

    print("\n" + "-" * 70)
    print("📝 USER PROMPT:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)

    # ── 6. Execute agentic generation ──────────────────────────────────
    print("\n🔍 Starting multi-step agentic workflow (streaming)...\n")
    print("=" * 70)
    print("🤖 AGENT RESPONSE:")
    print("=" * 70)

    result = agent.chat(
        prompt=prompt,
        streaming_callback=streaming_callback,
        max_reasoning_steps=10,
        temperature=0.3,
        n_predict=2048,
    )

    print("\n")  # Newline after streaming

    # ── 7. Display execution metadata ──────────────────────────────────
    print("=" * 70)
    print("📊 EXECUTION METADATA")
    print("=" * 70)
    print(f"Total agentic rounds:  {result['rounds']}")
    print(f"Tool calls executed:   {len(result['tool_calls'])}")
    
    for tc in result['tool_calls']:
        print(f"\n  🔹 Round {tc['round']}: {tc['name']}")
        print(f"     Parameters: {json.dumps(tc['parameters'], indent=2)}")
        
        tr = next((r for r in result['tool_results'] if r['round'] == tc['round']), None)
        if tr:
            res = tr['result']
            status = "✅ SUCCESS" if res.get('success') else "❌ FAILED"
            output = str(res.get('output', res.get('error', 'No output')))[:150]
            print(f"     Result: {status}")
            print(f"     Output: {output}...")

    print("\n" + "=" * 70)
    print("🎉 Handbag Agent Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()