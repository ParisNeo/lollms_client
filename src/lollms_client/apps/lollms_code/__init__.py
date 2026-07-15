"""
lollms_code — Autonomous CLI Coding Agent powered by lollms_client.

A fully autonomous coding assistant that leverages:
  - LollmsAgent with multi-step reasoning loops (build → test → fix)
  - Persistent Skills (SKILL.md) for cross-session learning
  - LollmsMemoryManager for episodic and semantic memory
  - Sub-agent delegation for complex tasks
  - Model switching for adaptive performance
  - Workspace isolation and intelligent context management
"""

from lollms_client.apps.lollms_code.cli import main

__all__ = ["main"]
