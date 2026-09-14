"""
lollms_agentic: Two-stage Orchestrator/Worker execution doctrine.

The Orchestrator plans and verifies; it NEVER executes tools itself.
It spawns workers via the functional `<agent>` tag (never a tool call).
The Worker runs through the SAME chat() engine with the full tool doctrine.
Progress is tracked in a verifiable progress.md file (checkmark list).

Exports:
    AgenticRunner      — entry point used by ChatMixin when agent mode is active.
    OrchestratorAgent  — planning + delegation + verification persona.
    WorkerAgent        — single atomic task executor.
    AgenticPlan        — serializable plan structure (steps + progress.md).
    build_spinoff_agent_tools — in-process sub-agent tool factory.
    parse_agent_tag / run_sub_agent — <agent> tag spawner for ChatMixin.
"""

from .orchestrator import OrchestratorAgent, AgenticPlan
from .worker import WorkerAgent
from .runner import AgenticRunner
from .spinoff_tools import build_spinoff_agent_tools
from .sub_agent_spawner import parse_agent_tag, run_sub_agent

__all__ = [
    "OrchestratorAgent",
    "WorkerAgent",
    "AgenticRunner",
    "AgenticPlan",
    "build_spinoff_agent_tools",
    "parse_agent_tag",
    "run_sub_agent",
]