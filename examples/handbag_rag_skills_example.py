#!/usr/bin/env python3
"""
handbag_rag_skills_example.py
==============================
Demonstrates the unified Handbag architecture in lollms_client:
  1. Default client configuration resolution via get_client_from_env() (like lollms-code).
  2. Programmatic creation of a complete Handbag folder:
     - SOUL.md: Agent core identity & persona.
     - skills/:
       • Text-only skill (no frontmatter) -> Automatically loaded into active context.
       • Loadable skill (with YAML frontmatter) -> Listed under available skills, loaded via tool_load_skill.
     - rag/: Domain documents for knowledge retrieval.
     - tools/: Standalone LCP tools.
     - memory/: Independent SQLite memory database.
  3. LollmsPersonality.from_handbag() initialization.
  4. Multi-turn reasoning verifying RAG retrieval, skill application, and tool execution.

Requirements
------------
pip install lollms_client ascii_colors
"""

import os
import sys
import shutil
import json
from pathlib import Path

# Ensure src/ is importable when running from repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ascii_colors import ASCIIColors
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_personality import LollmsPersonality, CapabilityFlags, AgentRole
from lollms_client.lollms_types import MSG_TYPE


# ── HANDBAG DIRECTORY ────────────────────────────────────────────────────────
HANDBAG_PATH = PROJECT_ROOT / "data_workspace" / "devops_handbag"


# ── HANDBAG ASSETS DEFINITIONS ───────────────────────────────────────────────

SOUL_CONTENT = """---
name: DevOpsArchitect
author: ParisNeo
version: '2.0'
category: devops_engineering
description: Expert DevOps & Site Reliability Engineer specializing in resilient systems, microservices, and database clustering.
---

You are DevOpsArchitect, an autonomous infrastructure and SRE specialist.
Follow your workflow:
1. Always follow the guidelines in your active skills.
2. If you need specialized patterns, load the corresponding skill via `tool_load_skill`.
3. Query your attached RAG knowledge base for specific cluster policies and credentials.
4. Use your tools to perform measurements or operations.
5. Provide concise, production-ready recommendations and end with `<done/>`.
"""

# 1. Text-only Skill (No YAML frontmatter) -> Always auto-loaded into context
TEXT_ONLY_SKILL_CONTENT = """# Incident Response Protocol

## Severity Classification
- **P0 (Critical)**: Total service outage or data corruption. Immediate rollback required.
- **P1 (High)**: Degraded performance affecting > 20% of users. Failover to replica.
- **P2 (Medium)**: Non-blocking bug or partial redundancy loss.

## Execution Rules
1. Never apply hotfixes directly to main database instances without an active backup.
2. In P0 scenarios, always verify the standby node state before issuing a failover command.
"""

# 2. Structured Skill (With YAML frontmatter) -> Loadable on demand
LOADABLE_SKILL_CONTENT = """---
title: "Database Failover Procedure"
description: "Step-by-step commands to promote a standby database replica to primary."
category: "database_admin"
tags: [postgresql, replication, failover]
visibility: loadable
---

# Database Failover Procedure

## Step-by-Step Recovery
1. Verify lag on standby: `SELECT now() - pg_last_xact_replay_timestamp();`
2. Promote standby node: `pg_ctl promote -D /var/lib/postgresql/data`
3. Reconfigure DNS endpoint `db.internal.net` to point to the new master IP.
4. Notify the operations team via notification webhook.
"""

# 3. RAG Knowledge Document
RAG_CLUSTER_SPECS = """# Production Cluster Specs & Policy

## Node Topology
- **Primary Database**: `db-primary.infra.internal` (IP: `10.0.1.10`, Region: `eu-west-1`)
- **Standby Database**: `db-replica.infra.internal` (IP: `10.0.1.11`, Region: `eu-west-1`)
- **Prometheus Endpoint**: `http://monitoring.infra.internal:9090`

## Access Policy
- Maximum allowed replication lag: 3000ms.
- Any latency exceeding 5000ms triggers automatic P1 alert escalation.
"""

# 4. Custom LCP Tool
TELEMETRY_TOOL_CONTENT = """TOOL_LIBRARY_NAME = 'Cluster Telemetry Fetcher'
TOOL_LIBRARY_DESC = 'Retrieves live status and replication metrics for database clusters.'
TOOL_LIBRARY_ICON = '📡'

def init_tools_library() -> None:
    pass

def tool_get_cluster_metrics(cluster_name: str = "prod-db"):
    \"\"\"
    Queries real-time telemetry metrics for the requested cluster.

    Args:
        cluster_name (str, optional): Target cluster identifier. Defaults to 'prod-db'.
    \"\"\"
    import json
    return {
        "cluster": cluster_name,
        "status": "Degraded",
        "primary_node": "db-primary.infra.internal",
        "standby_node": "db-replica.infra.internal",
        "replication_lag_ms": 6200,
        "active_connections": 1420,
        "cpu_load_percent": 94.5
    }
"""


def setup_handbag_files():
    """Builds the complete Handbag folder layout programmatically."""
    ASCIIColors.info(f"Building Handbag at: {HANDBAG_PATH}")
    if HANDBAG_PATH.exists():
        shutil.rmtree(HANDBAG_PATH, ignore_errors=True)

    HANDBAG_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Write SOUL.md
    (HANDBAG_PATH / "SOUL.md").write_text(SOUL_CONTENT, encoding="utf-8")

    # 2. Write Skills
    # Subfolder 1: Text-only (auto-visible)
    s1_dir = HANDBAG_PATH / "skills" / "incident_response"
    s1_dir.mkdir(parents=True, exist_ok=True)
    (s1_dir / "SKILL.md").write_text(TEXT_ONLY_SKILL_CONTENT, encoding="utf-8")

    # Subfolder 2: Loadable (on-demand)
    s2_dir = HANDBAG_PATH / "skills" / "database_failover"
    s2_dir.mkdir(parents=True, exist_ok=True)
    (s2_dir / "SKILL.md").write_text(LOADABLE_SKILL_CONTENT, encoding="utf-8")

    # 3. Write RAG documents
    rag_dir = HANDBAG_PATH / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    (rag_dir / "cluster_specs.md").write_text(RAG_CLUSTER_SPECS, encoding="utf-8")

    # 4. Write Tools
    tools_dir = HANDBAG_PATH / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "metrics_fetcher.py").write_text(TELEMETRY_TOOL_CONTENT, encoding="utf-8")

    # 5. Create Memory directory
    (HANDBAG_PATH / "memory").mkdir(parents=True, exist_ok=True)

    # 6. Create Workspace directory
    (HANDBAG_PATH / "workspace").mkdir(parents=True, exist_ok=True)

    ASCIIColors.success("✅ Handbag successfully created with SOUL, Skills, RAG, and Tools.")


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Streams generated text to console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        ASCIIColors.rich_print(chunk, end="", flush=True)
    return True


def main():
    ASCIIColors.panel(
        "[bold]DevOps Agent — Handbag RAG + Skills Multi-Step Demonstration[/bold]\n"
        "[dim]Demonstrating auto-loaded text skills, loadable on-demand skills, RAG specs, and LCP tools.[/dim]",
        title="[bold green]🎒 LOLLMS HANDBAG DEMO[/bold green]",
        border_style="green"
    )

    # Step 1: Populate the handbag structure on disk
    setup_handbag_files()

    # Step 2: Initialize LollmsClient from default environment configuration (like lollms-code)
    ASCIIColors.info("\nInitializing LollmsClient from environment...")
    try:
        client = get_client_from_env(create_llm=True)
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration error: {e}")
        ASCIIColors.yellow("Please run 'python -m lollms_client.lollms_config_cli_env' to configure your connection.")
        sys.exit(1)

    ASCIIColors.green(f"✅ Connected using binding: [bold]{client.llm.binding_name}[/bold] | Model: [bold]{getattr(client.llm, 'model_name', 'N/A')}[/bold]")

    # Step 3: Instantiate Personality directly from the Handbag
    ASCIIColors.info("\nLoading Personality from Handbag folder...")
    agent = LollmsPersonality.from_handbag(
        path=HANDBAG_PATH,
        lollms_client=client
    )

    # Configure capabilities
    agent.capabilities = CapabilityFlags(
        enable_code_execution=True,
        enable_workspace_tools=True,
        enable_sub_agents=False,
        enable_skill_loading=True,
        skills_mode="mixed"
    )
    agent.role = AgentRole.DOMAIN_EXPERT
    agent.max_tokens_per_turn = 4096

    ASCIIColors.green(f"✅ Personality '[bold]{agent.name}[/bold]' loaded successfully.")
    ASCIIColors.cyan(f"   • Skills Loaded: {len(agent.skills_manager.skills)}")
    ASCIIColors.cyan(f"   • Memory Attached: {agent.memory_manager is not None}")
    ASCIIColors.cyan(f"   • RAG Active: {agent.has_data}")

    # Inspect the automatically generated skills context zone
    skills_context_preview = agent.skills_manager.build_context()
    ASCIIColors.panel(
        skills_context_preview,
        title="[bold blue]🧠 Skills Context Injected Into Prompt[/bold blue]",
        border_style="blue"
    )

    # Step 4: Execute a Multi-Step Task
    task_prompt = (
        "We received an alert regarding cluster `prod-db`.\n"
        "1. Query the live telemetry metrics for `prod-db` using your tool.\n"
        "2. Cross-reference the replication lag against our policy in `cluster_specs.md`.\n"
        "3. Check your incident response rules to classify the severity.\n"
        "4. Load the `Database Failover Procedure` skill to extract the exact promotion command.\n"
        "5. Formulate a remediation plan and end with `<done/>`."
    )

    ASCIIColors.panel(task_prompt, title="[bold yellow]📝 User Task[/bold yellow]", border_style="yellow")
    ASCIIColors.rule("[bold green]🤖 Agent Deliberation & Execution Stream[/bold green]")

    result = agent.chat(
        prompt=task_prompt,
        streaming_callback=streaming_callback,
        max_reasoning_steps=10,
        temperature=0.2,
        use_internal_history=False,
    )

    print("\n")
    ASCIIColors.rule("[bold cyan]📊 EXECUTION SUMMARY[/bold cyan]")

    summary_table = ASCIIColors.table(
        "Metric", "Value",
        rows=[
            ["Total Reasoning Rounds", str(result.get("rounds", 0))],
            ["Tool Calls Made", str(len(result.get("tool_calls", [])))],
            ["Was Cancelled", str(result.get("was_cancelled", False))],
            ["Final Response Size", f"{len(result.get('response', ''))} characters"]
        ],
        title="[bold]Session Metrics[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(summary_table)

    if result.get("tool_calls"):
        tool_rows = []
        for tc in result["tool_calls"]:
            tool_rows.append([
                f"Round {tc.get('round', '?')}",
                tc.get("name", "unknown"),
                json.dumps(tc.get("parameters", {}), ensure_ascii=False)
            ])
        tools_table = ASCIIColors.table(
            "Round", "Tool", "Parameters",
            rows=tool_rows,
            title="[bold magenta]🛠️ Tools Executed[/bold magenta]",
            box="round"
        )
        ASCIIColors.rich_print(tools_table)

    ASCIIColors.success("\n🎉 Handbag RAG + Skills demonstration completed successfully!")


if __name__ == "__main__":
    main()