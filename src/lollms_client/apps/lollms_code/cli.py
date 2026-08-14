#!/usr/bin/env python3
"""
lollms_code — Autonomous CLI Coding Agent
==========================================

A production-grade CLI tool that turns a single prompt into a full autonomous
coding session. It uses the LollmsPersonality system and the Handbag architecture
to autonomously:
  1. Analyze the target codebase (workspace context injection)
  2. Plan the implementation strategy
  3. Write code, execute tests, and fix failures iteratively
  4. Create persistent Skills (SKILL.md) from lessons learned
  5. Save episodic memories for cross-session continuity
  6. Delegate sub-tasks to focused child agents when needed
  7. Switch models mid-task for optimal performance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
import signal
import threading
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ascii_colors import ASCIIColors
from ascii_colors import questionary

from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality, PersonalityBundle
from lollms_client.lollms_personality.skills_manager import SkillsManager
from lollms_client.lollms_agent.lollms_agent import CapabilityFlags
from lollms_client.lollms_types import MSG_TYPE, EventMode

APP_NAME = "lollms_code"
APP_VERSION = "2.0.0"
APP_CONFIG_DIR = Path.home() / ".lollms_client" / "lollms_code"
APP_CONFIG_FILE = APP_CONFIG_DIR / "config.json"
APP_HISTORY_FILE = APP_CONFIG_DIR / "history.json"
APP_USER_PROFILE_FILE = Path.home() / ".lollms_client" / "user_profile.md"
APP_DEFAULT_WORKSPACE = Path.cwd()
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"
APP_DEFAULT_HANDSAG_DIR = APP_CONFIG_DIR / "handbags"

CODING_SYSTEM_PROMPT = """\
You are lollms_code, an elite autonomous software engineering agent.

## YOUR IDENTITY
You are not a chatbot. You are a hands-on engineer that writes, tests, and ships code.
You operate in a fully autonomous loop — no human intervention is required.

## WORKFLOW (MANDATORY)
For every task, follow this structured pipeline:

### Phase 1: RECONNAISSANCE
- Use `<unlock_file>filename</unlock_file>` to load key files into your context.
- The workspace tree is already visible in your system prompt. Check the [C], [M], [U] markers.
- If the workspace is empty, start fresh.
- If files exist, understand the architecture before modifying anything.

### Phase 2: PLANNING
- Before writing ANY code, state your plan in 3-5 bullet points.
- Identify which files need to be created, modified, or deleted.
- Identify potential risks or edge cases.

### Phase 3: IMPLEMENTATION
- Use `<artifact>` tags to create or overwrite files.
- For EXISTING files with small changes, use SEARCH/REPLACE blocks inside `<artifact>` tags.
- Write clean, production-quality code with proper error handling.
- Include docstrings and type hints where appropriate.

### Phase 4: TESTING & VERIFICATION
- Use `tool_execute_shell_command` to run tests (e.g., `python -m pytest`).
- Read the test output carefully. If tests fail, FIX THE ROOT CAUSE.
- Do NOT mask errors with try/except — fix the actual bug.
- Re-run tests after each fix until ALL pass.

### Phase 5: SKILL CREATION (CRITICAL FOR LEARNING)
- After completing a non-trivial task, ALWAYS use `tool_create_skill` to save:
  - The pattern or methodology you used
  - Any gotchas or edge cases you discovered
  - Best practices specific to this codebase
- If you discover a BETTER way to do something you previously saved as a skill,
  use `tool_update_skill` to refine it.

### Phase 6: TERMINATION
- When ALL objectives are met and tests pass, write a brief summary:
  - What was created/modified
  - What tests pass
  - Any remaining TODOs or known limitations
- End with `<done/>` on a new line.

## AUTONOMY RULES
1. **NEVER ask the user for help.** You are autonomous. Make decisions.
2. **If stuck after 5 attempts on the same bug**, emit `<done/>` with a clear
   explanation of what failed and what you tried.
3. **If a tool is not available**, adapt and use what you have.
4. **Prefer correctness over speed.** A slow correct solution beats a fast broken one.

## CODE QUALITY STANDARDS
- All Python code must be PEP 8 compliant.
- All functions must have docstrings (Google or Sphinx style).
- All public functions must have type hints.
- Error handling: use specific exceptions, not bare `except:`.
- File encoding: always use `encoding='utf-8'` when opening files.
- Never leave debug `print()` statements in production code.

## CONTEXT MANAGEMENT
- The workspace tree is visible in your system prompt with markers: [C]=loaded, [U]=unlockable, [L]=locked.
- Use `<unlock_file>filename</unlock_file>` to load a file into context.
- Use `<lock_file>filename</lock_file>` when done to free context space.
- Do NOT read the same file repeatedly — it stays in your context after unlocking.

## SUB-AGENT DELEGATION
- If `tool_spawn_sub_agent` is available and the task has independent sub-components,
  delegate each to a focused sub-agent.
- Examples: "write the frontend" + "write the backend" → two sub-agents.
- Always provide clear, specific instructions to sub-agents.
- After sub-agents complete, synthesize their outputs into a unified result.

## SKILL SYSTEM USAGE
- Before starting a task, use `tool_list_skills` to check if a relevant skill exists.
- If found, use `tool_load_skill` to get the full content.
- After completing a task, ALWAYS create or update a skill.
- Skills are your long-term memory — they make you better over time.

## PERSISTENT MEMORY SYSTEM (CRITICAL FOR CONTINUITY)
You have access to a persistent memory database that survives across sessions.
1. **STORE FACTS**: When the user shares personal information (name, preferences, project details), you MUST save it immediately:
   <mem_new content="The user's name is Saif" tags="identity,user_profile" level="2" />
2. **UPDATE FACTS**: If information changes, update the memory:
   <mem_update id="memory_id" content="New information" />
3. **AUTOMATIC RECALL**: Relevant memories are automatically injected into your context. You do not need to query them manually.
4. **MANDATORY**: Always use memory tags for non-trivial user facts. If the user tells you their name, you MUST emit `<mem_new>` in your response.
5. **USE MEMORIES**: When asked "do you remember my name?", check the ACTIVE MEMORIES section in your context. If the user's name is there, use it.

## SANDBOX & WORKSPACE ISOLATION (CRITICAL)
You are operating inside the project workspace at `./` (which resolves to the project root).
1. **PROJECT FILES**: You have full access to read, modify, and create files in the workspace.
2. **TRANSIENT SCRIPTS**: All test scripts, temporary files, and experimental code MUST be written to the `.lollms_code/scripts/` subdirectory. This directory is automatically cleaned on every restart.
3. **PERSISTENT NOTES**: A `.lollms_code/scratchpad.md` file exists. Use it to store long-term context, architectural decisions, or task state. This file survives restarts.
4. **NO WORKSPACE BLOAT**: Do not leave temporary files in the root project directory. Use the `.lollms_code/` folder for all non-essential outputs.

## SYSTEM SHELL EXECUTION (PRIMARY EXECUTION METHOD)
You have access to the `tool_execute_shell_command` tool. This is your PRIMARY and ONLY method for interacting with the system, executing code, and managing files.

### WORKFLOW RULES
1. **FILE CREATION**: To create or overwrite files, use shell redirection or `python -c` via the tool.
   - Windows (cmd): `tool_execute_shell_command(command="echo content > scripts/file.py")`
   - For multi-line files: `tool_execute_shell_command(command="python -c \"with open('scripts/test.py', 'w', encoding='utf-8') as f: f.write('print(1)')\"")`
2. **FILE READING**: To read files for context, use `type scripts\file.py` (Windows) or `cat scripts/file.py` (Unix).
3. **CODE EXECUTION**: To execute Python code, use `python scripts/script.py` or `python -c "import math; print(math.pi)"`.
4. **PACKAGE MANAGEMENT**: If a package is missing, use `pip install package_name`.
5. **TESTING**: Run tests using `python -m pytest` or `python -m unittest`.

### GIT OPERATIONS (HIGH-EFFICIENCY PROTOCOL)
When asked to "commit", "push", or perform any git operation, you MUST follow this 2-round protocol:
- **Round 1**: Run `git diff` (or `git diff --stat` for large changes) to inspect what changed. DO NOT unlock or load any files into context.
- **Round 2**: Run `git add -A && git commit -m "message"` with a meaningful message based on the diff. Then emit `<done/>`.
You are STRICTLY FORBIDDEN from using `<unlock_file>` before a git commit. The diff output is sufficient to write a commit message.

### SAFETY
- The host application controls the autonomy level of the shell tool.
- If a command is blocked because it requires elevated privileges, inform the user that they need to adjust the `system_shell` configuration in the host application settings.
"""


# ── CLI INTERACTIVE HELP MANUAL ─────────────────────────────────────────────
HELP_SECTIONS = {
    "1": {
        "title": "🚀 1. Quick Start & Core Concepts",
        "content": """\
[cyan]Welcome to lollms_code![/cyan]

lollms_code is an autonomous coding agent that operates directly in your terminal.
Unlike standard LLM wrappers, it executes a [bold]Plan -> Code -> Test -> Fix[/bold] loop automatically.

[bold yellow]Basic Usage:[/bold yellow]
  [green]lollms-code "Implement a user authentication system using JWT"[/green]
  [green]lollms-code -i[/green]  # Starts interactive REPL mode

[bold yellow]Core Concepts:[/bold yellow]
  1. [cyan]Workspace[/cyan]: The agent reads your local directory tree automatically.
  2. [cyan]Context Tree[/cyan]: Files are marked [C] (Loaded), [M] (Metadata), [U] (Unlockable).
  3. [cyan]Autonomy[/cyan]: It uses shell tools to run commands, tests, and fix bugs iteratively until it succeeds.
  4. [cyan]Memory[/cyan]: It remembers facts across sessions (e.g., your name, preferences).
  5. [cyan]Skills[/cyan]: It saves reusable coding patterns as SKILL.md files for future use.
"""
    },
    "2": {
        "title": "💻 2. Coding & Refactoring",
        "content": """\
[bold magenta]Use Case: Writing new features, fixing bugs, or refactoring.[/bold magenta]

The agent is designed to write production-ready code. It uses an [bold]Aider-style SEARCH/REPLACE protocol[/bold] for surgical edits to existing files.

[bold yellow]Examples:[/bold yellow]
  - "Add pagination to the `UserList` component in `src/components.py`. Use a limit of 20."
  - "Find the memory leak in `image_processor.py` and fix it."
  - "Write unit tests for all functions in `utils/math.py` using pytest. Run the tests and fix any failures."

[bold green]Pro-Tip:[/bold green]
If the task is large, ask it to plan first:
  "Plan the architecture for a REST API for a blog. Then implement the models and database connection."
The agent will outline the plan, then create the files using `<artifact>` tags.
"""
    },
    "3": {
        "title": "📚 3. Documentation & Content Organization",
        "content": """\
[bold magenta]Use Case: Generating docs, READMEs, or organizing markdown content.[/bold magenta]

The agent can read your entire codebase and extract structural information to write accurate documentation.

[bold yellow]Examples:[/bold yellow]
  - "Read all Python files in the `src/` directory and generate a comprehensive `README.md` with architecture diagrams."
  - "Add Google-style docstrings to all public classes and functions in `main.py`."
  - "Scan the `data/` folder, analyze the CSV headers, and create a `schema.md` file documenting the data structures."

[bold green]Pro-Tip:[/bold green]
For massive codebases, unlock specific files first in your prompt:
  "Read `src/api/router.py` and `src/api/auth.py`, then write an API reference document."
"""
    },
    "4": {
        "title": "🧠 4. Autonomous Learning & Skills",
        "content": """\
[bold magenta]Use Case: Building a persistent knowledge base of coding patterns.[/bold magenta]

lollms_code automatically creates [cyan]Skills[/cyan] (SKILL.md files) when it solves a non-trivial problem. In future sessions, it loads these skills to solve similar problems instantly.

[bold yellow]Examples:[/bold yellow]
  - "Figure out how to integrate Stripe payment webhooks into this Flask app."
  (After succeeding, the agent saves a "stripe_integration" skill).
  
  - "What skills do you know?"
  (In interactive mode, type `skills` to list all learned skills).

[bold green]Pro-Tip:[/bold green]
You can explicitly ask the agent to create a skill:
  "Create a skill named 'git_conflict_resolution' documenting the best way to resolve complex merge conflicts."
"""
    },
    "5": {
        "title": "🤖 5. Sub-Agent Delegation (Complex Tasks)",
        "content": """\
[bold magenta]Use Case: Breaking down massive tasks into parallel work streams.[/bold magenta]

If enabled, the agent can spawn "child" agents to handle independent parts of a task simultaneously.

[bold yellow]Examples:[/bold yellow]
  - "Build a full-stack weather app. Delegate the frontend (HTML/JS) to one sub-agent, and the backend (Python API) to another. Then integrate them."
  - "Translate the UI into Spanish, French, and German simultaneously using sub-agents."

[bold green]Pro-Tip:[/bold green]
Sub-agents share the same workspace but have isolated context windows. They are perfect for heavy, independent operations like data processing or file generation.
"""
    },
    "6": {
        "title": "⚙️ 6. Configuration & Memory",
        "content": """\
[bold magenta]Use Case: Customizing behavior and ensuring cross-session continuity.[/bold magenta]

[bold yellow]Interactive Commands:[/bold yellow]
  - [cyan]config[/cyan]: Runs the Lollms configuration wizard (changes models, bindings).
  - [cyan]forget[/cyan]: Permanently wipes the agent's associative memory (use with caution!).
  - [cyan]skills[/cyan]: Lists all stored skills in your `~/.lollms_client/lollms_code/skills/` directory.

[bold yellow]Memory Management:[/bold yellow]
The agent remembers facts about you. If you say "My name is Alex", it will save it.
Next session, you can ask "What is my name?" and it will know.

[bold green]Pro-Tip:[/bold green]
Use `--workspace ./path/to/project` to target a specific directory without changing your current terminal path.
"""
    },
    "q": {
        "title": "Exit Help",
        "content": "Returning to the agent..."
    }
}

def show_interactive_help():
    """Displays the interactive, multi-page help manual."""
    current_page = "1"
    while True:
        section = HELP_SECTIONS[current_page]
        ASCIIColors.rule(f"[bold blue]{section['title']}[/bold blue]")
        ASCIIColors.rich_print(section["content"])
        
        ASCIIColors.rich_print("\n[bold]Navigation:[/bold]")
        ASCIIColors.rich_print("  [cyan]1-6[/cyan] - Jump to a specific section")
        ASCIIColors.rich_print("  [cyan]q[/cyan]   - Quit help and return to the agent")
        
        try:
            choice = input("\n  Choice> ").strip().lower()
            if choice in HELP_SECTIONS:
                current_page = choice
                if choice == "q":
                    break
            else:
                ASCIIColors.yellow("  Invalid choice. Please enter 1-6 or q.")
        except (EOFError, KeyboardInterrupt):
            break

class PersistentHistory:
    """Manages a persistent JSON-backed history of prompts for the REPL."""
    
    def __init__(self, history_file: Path, max_entries: int = 100):
        self.history_file = history_file
        self.max_entries = max_entries
        self.entries: List[str] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.entries = data
            except Exception:
                self.entries = []

    def _save(self):
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(
                json.dumps(self.entries, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
        except Exception as e:
            ASCIIColors.warning(f"Failed to save history: {e}")

    def add(self, prompt: str):
        prompt = prompt.strip()
        if not prompt:
            return
        if self.entries and self.entries[-1] == prompt:
            return
        self.entries.append(prompt)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._save()


class CodeAgentConfig:
    def __init__(self):
        self.llm_binding: str = "ollama"
        self.model_name: str = "qwen3:32b"
        self.host_address: str = "http://localhost:11434"
        self.api_key: Optional[str] = None
        self.verify_ssl: bool = False
        self.context_size: int = 8192
        self.n_gpu_layers: int = -1
        self.models_path: str = ""
        self.binaries_path: str = ""
        self.wizard_completed: bool = False
        self.llm_binding_config: Dict[str, Any] = {}
        self.max_reasoning_steps: int = 100
        self.temperature: float = 0.3
        self.max_tokens_per_turn: int = 8192
        self.enable_shell_execution: bool = True
        self.shell_autonomy_level: str = "safe"
        self.enable_sub_agents: bool = True
        self.enable_model_switching: bool = False
        self.enable_skill_creation: bool = True
        self.enable_skill_loading: bool = True
        self.enable_memory: bool = True
        self.skills_mode: str = "mixed"
        self.max_sub_agent_depth: int = 2
        self.max_sub_agents_per_turn: int = 3
        self.workspace_path: str = str(APP_DEFAULT_WORKSPACE)
        self.skills_dir: str = str(APP_DEFAULT_SKILLS_DIR)
        self.memory_db: str = f"sqlite:///{APP_DEFAULT_MEMORY_DB}"
        self.handbag_path: str = str(APP_DEFAULT_HANDSAG_DIR / "default_coder")
        self.show_tool_calls: bool = True
        self.show_workspace_changes: bool = True
        self.show_skills: bool = True
        self.show_progress: bool = True
        self.debug: bool = False

    @classmethod
    def load(cls, cli_args: argparse.Namespace) -> "CodeAgentConfig":
        config = cls()
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if APP_CONFIG_FILE.exists():
            try:
                file_config = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
                for key, val in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, val)
            except Exception as e:
                ASCIIColors.warning(f"Failed to read config file: {e}")

        try:
            get_client_from_env(
                create_llm=True,
                create_tti=False,
                create_tts=False,
                create_stt=False,
                create_ttm=False,
                create_ttv=False,
                run_wizard_if_fail=False
            )
        except Exception:
            pass

        default_binding_alias = None
        for k, v in os.environ.items():
            if k.startswith("LLM_PROFILES_") and k.endswith("_IS_DEFAULT") and v.lower() in ("true", "1", "yes"):
                default_binding_alias = k[len("LLM_PROFILES_"):-len("_IS_DEFAULT")]
                break

        if default_binding_alias:
            binding_alias = os.getenv(f"LLM_PROFILES_{default_binding_alias}_BINDING_ALIAS", default_binding_alias)
            config.llm_binding = os.getenv(f"LLM_BINDINGS_{binding_alias}_BINDING_NAME", config.llm_binding)
            config.model_name = os.getenv(f"LLM_PROFILES_{default_binding_alias}_MODEL_NAME", config.model_name)
            config.host_address = os.getenv(f"LLM_BINDINGS_{binding_alias}_HOST_ADDRESS", config.host_address)
            config.api_key = os.getenv(f"LLM_BINDINGS_{binding_alias}_SERVICE_KEY", config.api_key)

            ssl_env = os.getenv(f"LLM_BINDINGS_{binding_alias}_VERIFY_SSL_CERTIFICATE")
            if ssl_env is not None:
                config.verify_ssl = ssl_env.lower() in ("true", "1", "yes")
        else:
            config.llm_binding = os.getenv("LLM_BINDING_NAME", config.llm_binding)
            config.model_name = os.getenv("MODEL_NAME", config.model_name)
            config.host_address = os.getenv("HOST_ADDRESS", config.host_address)
            config.api_key = os.getenv("API_KEY", config.api_key)

            ssl_env = os.getenv("VERIFY_SSL_CERTIFICATE")
            if ssl_env is not None:
                config.verify_ssl = ssl_env.lower() in ("true", "1", "yes")

        if cli_args.llm_binding:
            config.llm_binding = cli_args.llm_binding
        if cli_args.model:
            config.model_name = cli_args.model
        if cli_args.host:
            config.host_address = cli_args.host
        if cli_args.api_key:
            config.api_key = cli_args.api_key
        if cli_args.workspace:
            config.workspace_path = str(Path(cli_args.workspace).resolve())
        if cli_args.max_steps:
            config.max_reasoning_steps = cli_args.max_steps
        if cli_args.temperature is not None:
            config.temperature = cli_args.temperature
        if cli_args.max_tokens:
            config.max_tokens_per_turn = cli_args.max_tokens
        if cli_args.context_size:
            config.context_size = cli_args.context_size
        if cli_args.debug is not None:
            config.debug = cli_args.debug
        else:
            config.debug = False
        if cli_args.enable_model_switching:
            config.enable_model_switching = True
        if cli_args.no_shell_execution:
            config.enable_shell_execution = False
        if cli_args.shell_autonomy:
            config.shell_autonomy_level = cli_args.shell_autonomy
        if cli_args.no_sub_agents:
            config.enable_sub_agents = False
        if cli_args.no_memory:
            config.enable_memory = False
        if cli_args.skills_dir:
            config.skills_dir = str(Path(cli_args.skills_dir).resolve())
        if cli_args.handbag_path:
            config.handbag_path = str(Path(cli_args.handbag_path).resolve())

        return config

    def save(self):
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        try:
            APP_CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"Failed to save config: {e}")

    def is_configured(self) -> bool:
        if not self.llm_binding or not self.model_name:
            return False
        if self.llm_binding_config:
            return True
            
        host_address = os.getenv("HOST_ADDRESS")
        if host_address:
            return True
            
        for k, v in os.environ.items():
            if k.startswith("LLM_BINDINGS_") and "_HOST_ADDRESS" in k and v:
                return True
                
        return False
    

def create_client(config: CodeAgentConfig) -> LollmsClient:
    if config.llm_binding_config:
        llm_config = dict(config.llm_binding_config)
    else:
        llm_config: Dict[str, Any] = {
            "model_name": config.model_name,
            "host_address": config.host_address,
            "verify_ssl_certificate": config.verify_ssl,
        }
        if config.llm_binding == "llama_cpp_server":
            llm_config["ctx_size"] = config.context_size
            llm_config["n_gpu_layers"] = config.n_gpu_layers
            if config.models_path:
                llm_config["models_path"] = config.models_path
            if config.binaries_path:
                llm_config["binaries_path"] = config.binaries_path

    if config.model_name and config.model_name != llm_config.get("model_name"):
        llm_config["model_name"] = config.model_name
    if config.host_address and config.host_address != llm_config.get("host_address"):
        llm_config["host_address"] = config.host_address
    if config.api_key:
        llm_config["service_key"] = config.api_key
    if "verify_ssl_certificate" not in llm_config:
        llm_config["verify_ssl_certificate"] = config.verify_ssl

    import lollms_client
    package_root = Path(lollms_client.__file__).resolve().parent
    default_tools_path = package_root / "tools_bindings" / "lcp" / "default_tools"

    tools_folders = [str(default_tools_path)] if default_tools_path.exists() else []

    host_tool_configs = {
        "system_shell": {
            "autonomy_level": config.shell_autonomy_level
        }
    }

    client = LollmsClient(
        llm_binding_name=config.llm_binding,
        llm_binding_config=llm_config,
        tools_binding_name="lcp",
        tools_binding_config={
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs
        },
    )

    if config.enable_shell_execution and client.tools:
        try:
            client.tools.mount_tool_library('system_shell')
            ASCIIColors.success("[CLI] ✅ System Shell library mounted.")
        except Exception as e:
            ASCIIColors.warning(f"Failed to pre-mount system_shell library: {e}")

    return client


def ensure_handbag_structure(config: CodeAgentConfig):
    """Ensures that the handbag directory and SOUL.md exist and are up-to-date."""
    handbag_path = Path(config.handbag_path)
    handbag_path.mkdir(parents=True, exist_ok=True)

    soul_path = handbag_path / "SOUL.md"
    metadata = {
        "name": "lollms_code",
        "author": "ParisNeo",
        "category": "software_engineering",
        "description": "An elite autonomous software engineering agent that writes, tests, and fixes code iteratively.",
        "temperature": str(config.temperature)
    }
    yaml_lines = [f"{k}: {v}" for k, v in metadata.items()]
    soul_content = f"---\n{chr(10).join(yaml_lines)}\n---\n\n{CODING_SYSTEM_PROMPT}"

    if not soul_path.exists() or soul_path.read_text(encoding="utf-8") != soul_content:
        soul_path.write_text(soul_content, encoding="utf-8")
        ASCIIColors.info("[CLI] SOUL.md updated to latest system prompt standard.")
        
    coworkers_dir = handbag_path / "coworkers"
    coworkers_dir.mkdir(exist_ok=True)
    
    tools_dir = handbag_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    
    skills_dir = handbag_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    
    memory_dir = handbag_path / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    workspace_dir = handbag_path / "workspace"
    workspace_dir.mkdir(exist_ok=True)


def ensure_sandbox_structure(config: CodeAgentConfig):
    """Ensures the .lollms_code sandbox exists and cleans transient scripts."""
    sandbox_dir = Path(config.workspace_path) / ".lollms_code"
    scripts_dir = sandbox_dir / "scripts"
    scratchpad = sandbox_dir / "scratchpad.md"

    sandbox_dir.mkdir(parents=True, exist_ok=True)

    if scripts_dir.exists():
        for f in scripts_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
    scripts_dir.mkdir(exist_ok=True)

    if not scratchpad.exists():
        scratchpad.write_text("# Agent Scratchpad\n\nUse this space to store long-term notes, code snippets, and task context.\n", encoding="utf-8")

def build_environment_context(config: CodeAgentConfig) -> str:
    """Builds a dynamic system prompt block describing the execution environment."""
    is_windows = platform.system() == "Windows"
    os_name = platform.system()
    os_version = platform.version()
    python_version = platform.python_version()

    workspace_root = Path(config.workspace_path).resolve()
    sandbox_dir = workspace_root / ".lollms_code"
    scripts_dir = sandbox_dir / "scripts"

    shell_cmd = "cmd / powershell" if is_windows else "bash/sh"
    list_cmd = "dir" if is_windows else "ls"
    read_cmd = "type" if is_windows else "cat"
    path_sep = "\\" if is_windows else "/"

    git_branch_info = ""
    git_dir = workspace_root / ".git"
    if git_dir.exists():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(workspace_root),
                capture_output=True, text=True, encoding="utf-8", errors="ignore"
            )
            if result.returncode == 0:
                branch_name = result.stdout.strip()
                if branch_name:
                    git_branch_info = f"\n- Git Branch: {branch_name}"
        except Exception:
            pass

    return f"""
=== ENVIRONMENT CONTEXT (CRITICAL) ===
You are operating in the following environment:
- Operating System: {os_name} {os_version}
- Python Version: {python_version}
- Shell: {shell_cmd}
- Path Separator: `{path_sep}`{git_branch_info}

### OS-SPECIFIC RULES (MANDATORY)
1. **SHELL COMMANDS**: You MUST use {os_name}-compatible shell commands.
   - To list files: Use `{list_cmd}`
   - To read files: Use `{read_cmd} filename.ext`
   - To execute scripts: Use `python script.py` (not `python3` on Windows)
2. **PATHS**: Always use `{path_sep}` for file paths in shell commands. ALL paths must be relative to the Workspace Root. NEVER attempt to access absolute paths outside the workspace.
3. **TRANSIENT SCRIPTS**: When writing test scripts or temporary files, you MUST save them to the Sandbox Directory (`.lollms_code/scripts/`).
   - Example: `python -c "with open('.lollms_code{path_sep}scripts{path_sep}test.py', 'w') as f: f.write('print(1)')"`
   - NEVER create `.py` or `.log` files in the Workspace Root.
4. **SANDBOX ISOLATION**: The Workspace Root contains the user's actual project. Do not modify project files unless explicitly instructed. Use the Sandbox Directory for all experimental work.
=== END ENVIRONMENT CONTEXT ===
"""

def create_coding_personality(config: CodeAgentConfig, client: LollmsClient) -> LollmsPersonality:
    """Creates a coding personality from the handbag structure, injecting client and capabilities."""
    ensure_handbag_structure(config)
    ensure_sandbox_structure(config)

    caps = CapabilityFlags(
        enable_sub_agents=config.enable_sub_agents,
        enable_model_switching=config.enable_model_switching,
        enable_skill_creation=config.enable_skill_creation,
        enable_skill_loading=config.enable_skill_loading,
        enable_workspace_tools=True,
        skills_mode=config.skills_mode,
        max_sub_agent_depth=config.max_sub_agent_depth,
        max_sub_agents_per_turn=config.max_sub_agents_per_turn
    )

    personality = LollmsPersonality.from_handbag(config.handbag_path)
    personality.lollms_client = client

    personality.workspace_path = Path(config.workspace_path)

    env_context = build_environment_context(config)
    personality.system_prompt = personality.system_prompt + "\n" + env_context

    personality.capabilities = caps
    personality.max_tokens_per_turn = config.max_tokens_per_turn

    personality.debug_mode = config.debug

    personality._init_user_profile(APP_USER_PROFILE_FILE)

    try:
        if hasattr(personality, "_init_artefact_system"):
            personality._init_artefact_system()
        if hasattr(personality, "_sync_artefact_index_with_disk"):
            personality._sync_artefact_index_with_disk()
    except Exception as e:
        ASCIIColors.warning(f"Failed to pre-initialize artefact system for stats: {e}")

    return personality


def _format_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def _render_files_table(files_data: List[Dict[str, Any]], title: str):
    from rich.console import Console
    from rich.table import Table
    console = Console()
    sorted_files = sorted(
        files_data, 
        key=lambda x: x.get("size", 0), 
        reverse=True
    )
    files_table = Table(title=f"[bold]{title}[/bold]", box=None)
    files_table.add_column("Size", style="cyan", no_wrap=True)
    files_table.add_column("Relative Path", style="white")
    for f in sorted_files:
        files_table.add_row(_format_bytes(f["size"]), f["path"])
    console.print(files_table)

def get_workspace_stats(personality: LollmsPersonality) -> Dict[str, Any]:
    """
    Calculates statistics about the indexed workspace files.
    Returns total files, loaded files count, and a list of relative paths of loaded files.
    """
    stats = {
        "total_indexed": 0,
        "total_loaded": 0,
        "loaded_files": []
    }
    
    if not hasattr(personality, '_artefact_manager') or not personality._artefact_manager:
        return stats
        
    try:
        from lollms_client.lollms_artefact import ArtefactVisibility
        all_arts = personality._artefact_manager._get_all_raw()
        
        stats["total_indexed"] = len([a for a in all_arts if not a.get("title", "").endswith("::images")])
        
        for art in all_arts:
            if art.get("visibility") == ArtefactVisibility.FULL:
                rel_path = art.get("physical_path") or art.get("title", "")
                if rel_path:
                    ws_root = str(personality._resolved_workspace)
                    if rel_path.startswith(ws_root):
                        rel_path = rel_path[len(ws_root):].lstrip("\\/")

                    file_size = art.get("size", 0)
                    if not file_size:
                        try:
                            abs_path = personality._resolved_workspace / rel_path
                            if abs_path.exists() and abs_path.is_file():
                                file_size = abs_path.stat().st_size
                        except Exception:
                            file_size = 0

                    stats["loaded_files"].append({
                        "path": rel_path,
                        "size": file_size
                    })

        stats["total_loaded"] = len(stats["loaded_files"])
    except Exception as e:
        ASCIIColors.warning(f"Failed to calculate workspace stats: {e}")
        
    return stats


class StreamRenderer:
    def __init__(self, config: CodeAgentConfig):
        self.config = config
        self._processing_buffer = ""
        self._in_processing = False

    def _render_processing_block(self, block_content: str):
        """Parses and renders a <processing> block as a rich panel."""
        import re
        import json as _json

        type_match = re.search(r'type="([^"]+)"', block_content)
        title_match = re.search(r'title="([^"]+)"', block_content)
        params_match = re.search(r'params="([^"]+)"', block_content)

        proc_type = type_match.group(1) if type_match else "action"
        title = title_match.group(1) if title_match else "Processing"

        if proc_type == "tool":
            params_str = params_match.group(1) if params_match else "{}"
            try:
                params_dict = _json.loads(params_str)
                params_str_formatted = _json.dumps(params_dict, indent=2, ensure_ascii=False)
            except Exception:
                params_str_formatted = params_str

            body_match = re.search(r'>(.*)', block_content, re.DOTALL)
            body_text = body_match.group(1).strip() if body_match else ""

            panel_content = f"[cyan]Parameters:[/cyan]\n[dim]{params_str_formatted}[/dim]\n\n[cyan]Execution Log:[/cyan]\n{body_text}"
            ASCIIColors.panel(
                panel_content,
                title=f"[bold blue]🛠️ Tool Execution: {title}[/bold blue]",
                border_style="blue"
            )
        else:
            body_match = re.search(r'>(.*)', block_content, re.DOTALL)
            body_text = body_match.group(1).strip() if body_match else ""
            ASCIIColors.panel(
                body_text,
                title=f"[bold magenta]⚙️ {title}[/bold magenta]",
                border_style="magenta"
            )

    def _render_callback_event(self, msg_type: Any, meta: Optional[Dict]):
        """Renders structured MSG_TYPE events as Rich panels for FULL_CALLBACK_MODE."""
        if not meta:
            return

        if msg_type == MSG_TYPE.MSG_TYPE_TOOL_START:
            tool_name = meta.get("tool_name", "unknown")
            params = meta.get("parameters", {})
            
            command_str = params.get("command", "")
            autonomy = params.get("autonomy_level", "safe")
            
            if tool_name == "tool_execute_shell_command" and command_str:
                panel_content = (
                    f"[cyan]Command:[/cyan] [yellow]{command_str}[/yellow]\n"
                    f"[cyan]Autonomy:[/cyan] [dim]{autonomy}[/dim]\n"
                    f"\n[cyan]Status:[/cyan] [yellow]⏳ Executing...[/yellow]"
                )
            else:
                params_str = json.dumps(params, indent=2, ensure_ascii=False) if params else "{}"
                panel_content = (
                    f"[cyan]Parameters:[/cyan]\n[dim]{params_str}[/dim]\n"
                    f"\n[cyan]Status:[/cyan] [yellow]⏳ Executing...[/yellow]"
                )

            ASCIIColors.panel(
                panel_content,
                title=f"[bold blue]🛠️ Executing: {tool_name}[/bold blue]",
                border_style="blue"
            )

        elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_END:
            tool_name = meta.get("tool_name", "unknown")
            success = meta.get("success", False)
            output = meta.get("output", "")
            error = meta.get("error")

            status_str = "[green]✅ Success[/green]" if success else "[red]❌ Failed[/red]"
            
            max_log_lines = 15
            log_lines = (output or error or "").splitlines()
            if len(log_lines) > max_log_lines:
                log_content = "\n".join(log_lines[:max_log_lines]) + f"\n[dim]... ({len(log_lines) - max_log_lines} more lines truncated)[/dim]"
            else:
                log_content = output or error or ""

            panel_content = f"[cyan]Status:[/cyan] {status_str}\n\n[cyan]Execution Log:[/cyan]\n{log_content}"
            ASCIIColors.panel(
                panel_content,
                title=f"[bold blue]🛠️ Finished: {tool_name}[/bold blue]",
                border_style="green" if success else "red"
            )

        elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START:
            title = meta.get("title", "artifact")
            art_type = meta.get("art_type", "code")
            lang = meta.get("language", "")
            is_patch = meta.get("is_patch", False)

            action = "📝 Rewriting" if not is_patch else "🔧 Patching"
            ASCIIColors.panel(
                f"[cyan]Action:[/cyan] {action}\n[cyan]Language:[/cyan] {lang or 'N/A'}",
                title=f"[bold magenta]⚙️ Artifact: {title}[/bold magenta]",
                border_style="magenta"
            )

        elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END:
            title = meta.get("title", "artifact")
            success = meta.get("success", False)
            version = meta.get("version", 1)
            error = meta.get("error")

            status_str = f"[green]✅ Saved (v{version})[/green]" if success else f"[red]❌ Failed: {error}[/red]"
            ASCIIColors.panel(
                status_str,
                title=f"[bold magenta]⚙️ Artifact Complete: {title}[/bold magenta]",
                border_style="green" if success else "red"
            )

        elif msg_type == MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE:
            action = meta.get("action", "update")
            status = meta.get("status", "")

            # Suppress intermediate stream events; only render the final execution event
            if status in ("streaming", "stream_complete"):
                return

            files = meta.get("files", [])
            error = meta.get("error")

            if not files and error:
                files_display = f"[red]{error}[/red]"
            elif not files:
                files_display = "[dim]No files specified[/dim]"
            else:
                files_display = "\n".join(f"  - {f}" for f in files)

            status_color = "green" if status == "success" else "red"
            panel_content = f"[cyan]Files:[/cyan]\n{files_display}\n\n[cyan]Status:[/cyan] [{status_color}]{status}[/{status_color}]"
            if error:
                panel_content += f"\n[cyan]Error:[/cyan] [red]{error}[/red]"

            ASCIIColors.panel(
                panel_content,
                title=f"[bold yellow]📂 Context {action.replace('_', ' ').capitalize()}[/bold yellow]",
                border_style="yellow"
            )

    def __call__(self, chunk: str, msg_type: Any = None, meta: Optional[Dict] = None) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_NEW_MESSAGE:
            ASCIIColors.rich_print("\n[bold green]🤖 Generating...[/bold green]")
            return True

        if msg_type in [
            MSG_TYPE.MSG_TYPE_TOOL_START,
            MSG_TYPE.MSG_TYPE_TOOL_END,
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START,
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END,
            MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE
        ]:
            # Reset processing buffer state to prevent UI bleeding between rapid events
            self._processing_buffer = ""
            self._in_processing = False
            self._render_callback_event(msg_type, meta)
            return True

        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("was_processed"):
                self._processing_buffer += chunk

                if "<processing" in chunk:
                    self._in_processing = True

                if "</processing>" in chunk or (self._in_processing and "<!-- status:" in chunk):
                    self._in_processing = False
                    try:
                        self._render_processing_block(self._processing_buffer)
                    except Exception as e:
                        ASCIIColors.rich_print(self._processing_buffer, end="")
                    self._processing_buffer = ""
            elif meta and (meta.get("live_tool_chunk") or meta.get("live_artifact_chunk")):
                # Suppress raw JSON/XML chunks of tools and artifacts to prevent UI bleeding.
                return True
            else:
                # Intercept and suppress <done/> tags from raw text stream
                if "<done" in chunk and "/>" in chunk:
                    return True

                if self._in_processing:
                    self._processing_buffer += chunk
                else:
                    ASCIIColors.rich_print(chunk, end="")
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            ASCIIColors.rich_print(f"[dim]{chunk}[/dim]", end="")
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            if meta and meta.get("done_intercepted"):
                ASCIIColors.rule("[bold green]✅ Task Completed (<done/>)[/bold green]")
                return True
            else:
                ASCIIColors.rich_print(f"\n[blue][INFO] {chunk}[/blue]")
            return True
    

def display_result(result: Dict[str, Any], config: CodeAgentConfig, elapsed: float):
    ASCIIColors.rule("[bold cyan]📊 SESSION REPORT[/bold cyan]")

    summary_rows = [
        ["Total rounds", str(result.get('rounds', 0))],
        ["Tool calls", str(len(result.get('tool_calls', [])))],
        ["Was cancelled", str(result.get('was_cancelled', False))],
        ["Elapsed time", f"{elapsed:.1f}s"]
    ]

    ctx_health = result.get("context_health")
    if ctx_health and ctx_health.get("max_tokens", 0) > 0:
        used = ctx_health.get("used_tokens", 0)
        max_t = ctx_health.get("max_tokens", 0)
        pct = ctx_health.get("fill_percentage", 0.0)
        summary_rows.append(["Context used", f"{used:,} / {max_t:,} tokens"])
        summary_rows.append(["Context fill", f"{pct:.1f}%"])

    summary_table = ASCIIColors.table(
        "Metric", "Value",
        rows=summary_rows,
        title="[bold]Execution Summary[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(summary_table)

    if config.show_workspace_changes and result.get("workspace_changes"):
        changes_table = ASCIIColors.table(
            "Action", "Path", "Size",
            rows=[[c.get("action", "?"), c.get("path", "?"), f"{c.get('size', 0):,} bytes"] for c in result["workspace_changes"]],
            title="[bold green]📁 Workspace Changes[/bold green]",
            box="round"
        )
        ASCIIColors.rich_print(changes_table)

    if config.show_skills and (result.get("skills_created") or result.get("skills_updated")):
        skills_table = ASCIIColors.table(
            "Action", "Skill",
            rows=([["Created", s] for s in result.get("skills_created", [])] + 
                  [["Updated", s] for s in result.get("skills_updated", [])]),
            title="[bold yellow]🎓 Skills Activity[/bold yellow]",
            box="round"
        )
        ASCIIColors.rich_print(skills_table)

    if result.get("sub_agents_spawned", 0) > 0:
        ASCIIColors.magenta(f"\n  🧠 Sub-agents spawned: {result['sub_agents_spawned']}")

    if result.get("model_switches"):
        ASCIIColors.blue(f"  🔄 Model switches: {result['model_switches']}")


def run_single_prompt(personality: LollmsPersonality, client: LollmsClient, prompt: str, config: CodeAgentConfig) -> int:
    if config.debug:
        dump_startup_context(personality, client)

    renderer = StreamRenderer(config)

    config_panel_content = (
        f"[cyan]Workspace:[/cyan] {config.workspace_path}\n"
        f"[cyan]Model:[/cyan]      {config.model_name}\n"
        f"[cyan]Binding:[/cyan]    {config.llm_binding}\n"
        f"[cyan]Max steps:[/cyan]  {config.max_reasoning_steps}\n"
        f"[cyan]Memory:[/cyan]     {'enabled' if config.enable_memory else 'disabled'}\n"
        f"[cyan]Skills:[/cyan]     {config.skills_mode}\n"
        f"[cyan]Sub-agents:[/cyan] {'enabled' if config.enable_sub_agents else 'disabled'}"
    )
    ASCIIColors.panel(config_panel_content, title=f"[bold green]🚀 lollms_code v{APP_VERSION}[/bold green]", border_style="green")
    
    ASCIIColors.panel(f"[magenta]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/magenta]", title="[bold]📝 Task[/bold]", border_style="magenta")
    
    # ── 📂 WORKSPACE STATS ──
    ws_stats = get_workspace_stats(personality)
    if ws_stats["total_indexed"] > 0:
        stats_content = (
            f"[cyan]Indexed Files:[/cyan] {ws_stats['total_indexed']}\n"
            f"[cyan]Loaded in Context:[/cyan] {ws_stats['total_loaded']}"
        )
        ASCIIColors.panel(stats_content, title="[bold blue]📂 Workspace Telemetry[/bold blue]", border_style="blue")
        
        if ws_stats["loaded_files"]:
            _render_files_table(ws_stats["loaded_files"], "Pre-loaded Context Files [C]")

    ASCIIColors.rule("[bold]🤖 Agent output[/bold]")

    start_time = time.time()
    ASCIIColors.rich_print("") # Ensure output starts on a new line

    def _signal_handler(sig, frame):
        ASCIIColors.yellow("\n\n⚠️  Interrupt received. Cancelling generation...")
        if hasattr(client, 'cancel'):
            client.cancel()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        result = personality.chat(
            prompt=prompt,
            lollms_client=client,
            streaming_callback=renderer,
            max_reasoning_steps=config.max_reasoning_steps,
            temperature=config.temperature,
            n_predict=config.max_tokens_per_turn,
            enable_artefacts=True,
            use_internal_history=False,
            event_mode=EventMode.FULL_CALLBACK_MODE,
        )
    except KeyboardInterrupt:
        if hasattr(client, 'cancel'):
            client.cancel()
        ASCIIColors.yellow("\n\n⚠️  Generation cancelled by user.")
        return 130
    except Exception as e:
        ASCIIColors.red(f"\n\n💥 Fatal error: {e}")
        return 1

    elapsed = time.time() - start_time
    display_result(result, config, elapsed)

    ASCIIColors.panel(result.get("response", ""), title="[bold cyan]📝 FINAL OUTPUT[/bold cyan]", border_style="cyan")

    return 0 if not result.get("was_cancelled") else 130


def get_context_fill_status(personality: LollmsPersonality, client: LollmsClient) -> Optional[Dict[str, Any]]:
    """Safely calculates the initial system prompt context fill status."""
    try:
        max_ctx = client.get_ctx_size() or 0
        if max_ctx <= 0:
            return None

        active_tools = personality._discover_tools(None, [])
        full_system_prompt = personality._build_system_prompt(active_tools)

        used_tokens = client.count_tokens(full_system_prompt) or 0
        fill_pct = round((used_tokens / max_ctx) * 100, 1)

        return {
            "used_tokens": used_tokens,
            "max_tokens": max_ctx,
            "fill_percentage": fill_pct
        }
    except Exception:
        return None

def dump_startup_context(personality: LollmsPersonality, client: LollmsClient):
    """
    🐛 DEBUG INSTRUMENTATION: Writes the initial system prompt and active tools
    to a debug log BEFORE the agentic loop begins. This captures the "Zero State"
    for diagnosing initialization hangs or context bloat.
    """
    if not getattr(personality, 'debug_mode', False):
        return

    try:
        ws_path = personality._resolved_workspace
        if not ws_path:
            return

        debug_dir = ws_path / ".lollms_code" / "_debug_dumps"
        debug_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        for item in debug_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass
            elif item.is_dir():
                try:
                    shutil.rmtree(str(item))
                except Exception:
                    pass

        debug_log_path = debug_dir / "startup_context.log"

        active_tools = personality._discover_tools(None, [])
        full_system_prompt = personality._build_system_prompt(active_tools)

        # Capture Memory and Scratchpad context if they exist
        mem_ctx = ""
        if hasattr(personality, '_build_user_profile_context'):
            mem_ctx += personality._build_user_profile_context()
        if hasattr(personality, '_build_scratchpad_context'):
            mem_ctx += personality._build_scratchpad_context()

        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("🐛 [DEBUG] STARTUP CONTEXT DUMP (ZERO-STATE)\n")
            f.write("=" * 80 + "\n\n")

            f.write("--- ACTIVE TOOLS ---\n")
            for t_name, t_spec in active_tools.items():
                f.write(f"- {t_name}: {t_spec.get('description', '')[:100]}\n")
            f.write("\n")

            f.write("--- FULL SYSTEM PROMPT (STABLE PREFIX) ---\n")
            f.write(full_system_prompt + "\n\n")

            if mem_ctx.strip():
                f.write("--- MEMORY & SCRATCHPAD CONTEXT ---\n")
                f.write(mem_ctx + "\n\n")

            f.write("--- WORKSPACE CONTEXT SNAPSHOT ---\n")
            ws_ctx = personality._build_workspace_context_block()
            f.write(ws_ctx + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("📊 CONTEXT STATS\n")
            f.write("=" * 80 + "\n")
            ctx_stats = get_context_fill_status(personality, client)
            if ctx_stats:
                f.write(f"Used Tokens: {ctx_stats['used_tokens']:,}\n")
                f.write(f"Max Tokens:  {ctx_stats['max_tokens']:,}\n")
                f.write(f"Fill %:      {ctx_stats['fill_percentage']}%\n")
            else:
                f.write("Context stats unavailable.\n")

        ASCIIColors.info(f"[CLI] 🐛 Startup context dumped to: {debug_log_path}")
    except Exception as e:
        ASCIIColors.warning(f"[CLI] Failed to dump startup context: {e}")

def _advanced_prompt(history: PersistentHistory, commands: List[str]) -> Optional[str]:
    """
    Cross-platform raw key-capture prompt with Ghost-Text Autocomplete.
    - Type '/': shows '/exit' in gray. Cursor stays after '/'.
    - Type 'f': shows '/files' in gray. Cursor stays after 'f'.
    - Press Tab or Right Arrow: accepts the gray suggestion.
    - Press Up/Down: cycles through matching commands (or history if no match).
    """
    import sys

    PROMPT_TEXT = "👤 You> "
    PROMPT_LEN = len(PROMPT_TEXT)

    def _draw_line(buffer: str, cursor_pos: int, ghost: str = ""):
        sys.stdout.write("\r\033[K")
        if ghost:
            sys.stdout.write(f"{PROMPT_TEXT}{buffer}\033[90m{ghost}\033[0m")
        else:
            sys.stdout.write(f"{PROMPT_TEXT}{buffer}")

        # Move cursor back to correct position
        if cursor_pos < len(buffer + ghost):
            # +1 because terminal positions are 1-based
            target_col = PROMPT_LEN + cursor_pos + 1
            sys.stdout.write(f"\033[{target_col}G")
        sys.stdout.flush()

    def _native_prompt_unix() -> Optional[str]:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        buffer = ""
        cursor_pos = 0
        history_idx = len(history.entries)
        current_input = ""
        active_suggestion_idx = -1

        def get_suggestions():
            if not buffer.startswith("/"):
                return []
            return [c for c in commands if c.startswith(buffer)]

        try:
            tty.setraw(fd)
            sys.stdout.write(PROMPT_TEXT)
            sys.stdout.flush()

            while True:
                ch = sys.stdin.read(1)
                sugg = get_suggestions()
                ghost_text = ""

                if ch == '\x03' or ch == '\x04':
                    sys.stdout.write("\n")
                    return None
                elif ch in ('\r', '\n'):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return buffer
                elif ch in ('\x7f', '\b'):
                    if cursor_pos > 0:
                        buffer = buffer[:cursor_pos-1] + buffer[cursor_pos:]
                        cursor_pos -= 1
                        active_suggestion_idx = -1
                        sugg = get_suggestions()
                        ghost_text = sugg[0][len(buffer):] if sugg and sugg[0].startswith(buffer) else ""
                        _draw_line(buffer, cursor_pos, ghost_text)
                elif ch == '\t':  # Tab accepts ghost text
                    if sugg:
                        buffer = sugg[0]
                        cursor_pos = len(buffer)
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos)
                elif ch in ('\r', '\n'):
                    sugg = get_suggestions()
                    if sugg and sugg[0].startswith(buffer):
                        buffer = sugg[0]
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return buffer
                elif ch == '\x1b':
                    ch2 = sys.stdin.read(1)
                    ch3 = sys.stdin.read(1)
                    if ch2 == '[':
                        if ch3 == 'A':  # Up
                            if sugg:
                                active_suggestion_idx = (active_suggestion_idx - 1) % len(sugg)
                                buffer = sugg[active_suggestion_idx]
                                cursor_pos = len(buffer)
                                _draw_line(buffer, cursor_pos)
                            elif history.entries:
                                if history_idx == len(history.entries):
                                    current_input = buffer
                                history_idx = max(0, history_idx - 1)
                                buffer = history.entries[history_idx]
                                cursor_pos = len(buffer)
                                _draw_line(buffer, cursor_pos)
                        elif ch3 == 'B':  # Down
                            if sugg:
                                active_suggestion_idx = (active_suggestion_idx + 1) % len(sugg)
                                buffer = sugg[active_suggestion_idx]
                                cursor_pos = len(buffer)
                                _draw_line(buffer, cursor_pos)
                            elif history_idx < len(history.entries):
                                history_idx += 1
                                if history_idx == len(history.entries):
                                    buffer = current_input
                                else:
                                    buffer = history.entries[history_idx]
                                cursor_pos = len(buffer)
                                _draw_line(buffer, cursor_pos)
                        elif ch3 == 'C':  # Right accepts ghost text
                            if sugg and cursor_pos == len(buffer):
                                buffer = sugg[0]
                                cursor_pos = len(buffer)
                                active_suggestion_idx = -1
                                _draw_line(buffer, cursor_pos)
                            elif cursor_pos < len(buffer):
                                cursor_pos += 1
                                _draw_line(buffer, cursor_pos, ghost_text)
                        elif ch3 == 'D':  # Left
                            if cursor_pos > 0:
                                cursor_pos -= 1
                                _draw_line(buffer, cursor_pos, ghost_text)
                elif len(ch) == 1 and ch.isprintable():
                    buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                    cursor_pos += 1
                    active_suggestion_idx = -1
                    sugg = get_suggestions()
                    ghost_text = sugg[0][len(buffer):] if sugg and sugg[0].startswith(buffer) else ""
                    _draw_line(buffer, cursor_pos, ghost_text)
        except Exception:
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _native_prompt_windows() -> Optional[str]:
        import msvcrt

        buffer = ""
        cursor_pos = 0
        history_idx = len(history.entries)
        current_input = ""
        active_suggestion_idx = -1

        def get_suggestions():
            if not buffer.startswith("/"):
                return []
            return [c for c in commands if c.startswith(buffer)]

        sys.stdout.write(PROMPT_TEXT)
        sys.stdout.flush()

        while True:
            if not msvcrt.kbhit():
                continue

            ch = msvcrt.getwch()
            sugg = get_suggestions()
            ghost_text = ""

            if ch in ('\x03', '\x1b'):
                sys.stdout.write("\n")
                return None
            elif ch in ('\r', '\n'):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return buffer
            elif ch in ('\x08', '\x7f'):
                if cursor_pos > 0:
                    buffer = buffer[:cursor_pos-1] + buffer[cursor_pos:]
                    cursor_pos -= 1
                    active_suggestion_idx = -1
                    sugg = get_suggestions()
                    ghost_text = sugg[0][len(buffer):] if sugg and sugg[0].startswith(buffer) else ""
                    _draw_line(buffer, cursor_pos, ghost_text)
            elif ch == '\t':
                if sugg:
                    buffer = sugg[0]
                    cursor_pos = len(buffer)
                    active_suggestion_idx = -1
                    _draw_line(buffer, cursor_pos)
            elif ch in ('\r', '\n'):
                sugg = get_suggestions()
                if sugg and sugg[0].startswith(buffer):
                    buffer = sugg[0]
                sys.stdout.write("\n")
                sys.stdout.flush()
                return buffer
            elif ch == '\x00' or ch == '\xe0':
                ch2 = msvcrt.getwch()
                if ch2 == 'H': # Up
                    if sugg:
                        active_suggestion_idx = (active_suggestion_idx - 1) % len(sugg)
                        buffer = sugg[active_suggestion_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos)
                    elif history.entries:
                        if history_idx == len(history.entries): current_input = buffer
                        history_idx = max(0, history_idx - 1)
                        buffer = history.entries[history_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos)
                elif ch2 == 'P': # Down
                    if sugg:
                        active_suggestion_idx = (active_suggestion_idx + 1) % len(sugg)
                        buffer = sugg[active_suggestion_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos)
                    elif history_idx < len(history.entries):
                        history_idx += 1
                        buffer = current_input if history_idx == len(history.entries) else history.entries[history_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos)
                elif ch2 == 'M': # Right
                    if sugg and cursor_pos == len(buffer):
                        buffer = sugg[0]
                        cursor_pos = len(buffer)
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos)
                    elif cursor_pos < len(buffer):
                        cursor_pos += 1
                        _draw_line(buffer, cursor_pos, ghost_text)
                elif ch2 == 'K': # Left
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        _draw_line(buffer, cursor_pos, ghost_text)
            elif ch.isprintable():
                buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                cursor_pos += 1
                active_suggestion_idx = -1
                sugg = get_suggestions()
                ghost_text = sugg[0][len(buffer):] if sugg and sugg[0].startswith(buffer) else ""
                _draw_line(buffer, cursor_pos, ghost_text)

    try:
        if sys.platform == 'win32':
            return _native_prompt_windows()
        else:
            return _native_prompt_unix()
    except Exception:
        try:
            return input(PROMPT_TEXT)
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            return None

def run_interactive(personality: LollmsPersonality, client: LollmsClient, config: CodeAgentConfig) -> int:
    if config.debug:
        dump_startup_context(personality, client)

    renderer = StreamRenderer(config)
    history = PersistentHistory(APP_HISTORY_FILE)

    slash_commands = ["/exit", "/quit", "/help", "/config", "/forget", "/skills", "/clear", "/models", "/files"]
    
    # Display a safe, truncated workspace path to the user
    ws_path_display = Path(config.workspace_path).resolve()
    try:
        # Attempt to show a relative path if it's under the home directory
        ws_path_display = ws_path_display.relative_to(Path.home())
        ws_path_display = f"~/{ws_path_display}"
    except ValueError:
        pass # Keep absolute if outside home directory

    header_lines = [
        f"[cyan]Workspace:[/cyan] {ws_path_display}",
        f"[cyan]Model:[/cyan]      {config.model_name}",
        f"[cyan]Binding:[/cyan]    {config.llm_binding}",
        f"[dim]Commands: 'exit' (quit), 'help' (manual), 'config' (wizard), 'forget' (wipe memory), 'skills' (list)[/dim]"
    ]

    ctx_status = get_context_fill_status(personality, client)
    if ctx_status:
        used = ctx_status["used_tokens"]
        max_t = ctx_status["max_tokens"]
        pct = ctx_status["fill_percentage"]
        header_lines.append(f"[cyan]Context:[/cyan]    {used:,} / {max_t:,} tokens ({pct:.1f}%)")

    ASCIIColors.panel(
        "\n".join(header_lines),
        title=f"[bold green]🚀 lollms_code v{APP_VERSION} — Interactive Mode[/bold green]",
        border_style="green"
    )

    # ── 📂 WORKSPACE TELEMETRY ──
    ws_stats = get_workspace_stats(personality)
    if ws_stats["total_indexed"] > 0:
        stats_content = (
            f"[cyan]Indexed Files:[/cyan] {ws_stats['total_indexed']}\n"
            f"[cyan]Loaded in Context:[/cyan] {ws_stats['total_loaded']}"
        )
        ASCIIColors.panel(stats_content, title="[bold blue]📂 Workspace Telemetry[/bold blue]", border_style="blue")
        
        if ws_stats["loaded_files"]:
            _render_files_table(ws_stats["loaded_files"], "Pre-loaded Context Files [C]")

    while True:
        try:
            user_input = _advanced_prompt(history, slash_commands)
        except (EOFError, KeyboardInterrupt):
            ASCIIColors.cyan("\n👋 Goodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            ASCIIColors.cyan("👋 Goodbye!")
            return 0

        if user_input.lower() == "/help":
            show_interactive_help()
            continue

        if user_input.lower() == "/forget":
            ASCIIColors.red("\n  ⚠️  WARNING: You are about to PERMANENTLY DELETE ALL agent memories.")
            ASCIIColors.red("  This includes user preferences, learned facts, and episodic history.")
            ASCIIColors.yellow("  Type 'CONFIRM WIPE' to proceed, or anything else to abort.")

            try:
                confirm = input("  ❓ Confirmation> ").strip()
            except (EOFError, KeyboardInterrupt):
                ASCIIColors.yellow("\n  ❌ Wipe aborted.")
                continue

            if confirm == "CONFIRM WIPE":
                if personality and hasattr(personality, "wipe_all_memories"):
                    if personality.wipe_all_memories():
                        ASCIIColors.red("  🧠 All episodic and associative memories have been permanently wiped.")
                    else:
                        ASCIIColors.yellow("  Memory manager not initialized or failed to wipe.")
                else:
                    ASCIIColors.yellow("  Personality does not support memory wiping.")
            else:
                ASCIIColors.green("  ✅ Wipe aborted. Memories are safe.")
            continue

        if user_input.lower() == "/skills":
            if personality.skills_manager:
                skills = personality.skills_manager.list_skills()
                if not skills:
                    ASCIIColors.yellow("  No skills learned yet.")
                else:
                    skills_table = ASCIIColors.table(
                        "Title", "Category", "Description",
                        rows=[[s['title'], s.get('category', ''), s.get('description', '')] for s in skills],
                        title="[bold yellow]📚 Learned Skills[/bold yellow]",
                        box="round"
                    )
                    ASCIIColors.rich_print(skills_table)
            else:
                ASCIIColors.yellow("  Skills manager not initialized.")
            continue

        if user_input.lower() == "/clear":
            personality._conversation = []
            ASCIIColors.green("  Conversation history cleared.")
            continue

        if user_input.lower() == "/models":
            ASCIIColors.yellow("  Model switching is managed via LollmsClient profiles in this version.")
            continue

        if user_input.lower() == "/config":
            from lollms_client.lollms_config_cli_env import run_wizard_and_save
            run_wizard_and_save()
            ASCIIColors.green("  Configuration updated. Restart lollms-code for changes to take effect.")
            continue

        if user_input.lower() == "/files":
            ws_stats = get_workspace_stats(personality)
            if not ws_stats["loaded_files"]:
                ASCIIColors.yellow("  No files are currently loaded in context [C].")
            else:
                _render_files_table(ws_stats["loaded_files"], "Loaded Context Files [C]")
            continue
        
        history.add(user_input)
        ASCIIColors.rule("[bold green]🤖 Agent[/bold green]")
        ASCIIColors.rich_print("") # Ensure output starts on a new line

        start_time = time.time()
        try:
            result = personality.chat(
                prompt=user_input,
                lollms_client=client,
                streaming_callback=renderer,
                max_reasoning_steps=config.max_reasoning_steps,
                temperature=config.temperature,
                n_predict=config.max_tokens_per_turn,
                enable_artefacts=True,
                use_internal_history=True,
                event_mode=EventMode.FULL_CALLBACK_MODE,
            )
        except KeyboardInterrupt:
            if hasattr(client, 'cancel'):
                client.cancel()
            ASCIIColors.yellow("\n\n⚠️  Cancelled.")
            continue
        except Exception as e:
            ASCIIColors.red(f"\n💥 Error: {e}")
            continue

        if hasattr(client, 'llm') and hasattr(client.llm, 'flush_stream'):
            try:
                client.llm.flush_stream()
            except Exception:
                pass

        elapsed = time.time() - start_time
        ctx_h = result.get("context_health", {})
        ctx_str = ""
        if ctx_h and ctx_h.get("max_tokens", 0) > 0:
            ctx_str = f" | Ctx: {ctx_h.get('fill_percentage', 0.0):.1f}%"
        ASCIIColors.rich_print(f"\n[dim]⏱️  {elapsed:.1f}s | Rounds: {result.get('rounds', 0)} | Tools: {len(result.get('tool_calls', []))}{ctx_str}[/dim]")


def list_skills(config: CodeAgentConfig):
    skills_dir = Path(config.skills_dir)
    if not skills_dir.exists():
        ASCIIColors.yellow("No skills directory found. Run a task first to generate skills.")
        return

    mgr = SkillsManager(skills_dirs=[str(skills_dir)], mode="loadable")
    skills = mgr.list_skills()
    if not skills:
        ASCIIColors.yellow("No skills learned yet.")
        return

    skills_table = ASCIIColors.table(
        "Title", "Category", "Description",
        rows=[[s['title'], s.get('category', ''), s.get('description', '')] for s in skills],
        title="[bold yellow]📚 Learned Skills[/bold yellow]",
        box="round"
    )
    ASCIIColors.rich_print(skills_table)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lollms-code",
        description=f"lollms_code v{APP_VERSION} — Autonomous CLI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  lollms-code "Implement a REST API client with retry logic"
  lollms-code -i
  lollms-code --workspace ./myproject "add unit tests"
  lollms-code --model qwen3:32b "refactor the database layer"
  lollms-code --enable-model-switching "build a CLI tool"
  lollms-code --list-skills
""",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="The task prompt for the autonomous agent.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start in interactive REPL mode.")
    parser.add_argument("--workspace", type=str, default=None, help="Path to the workspace directory.")
    parser.add_argument("--handbag-path", type=str, default=None, help="Path to the Handbag folder containing agent resources.")
    parser.add_argument("--model", type=str, default=None, help="Model name to use.")
    parser.add_argument("--llm-binding", type=str, default=None, help="LLM binding name.")
    parser.add_argument("--host", type=str, default=None, help="Host address for remote bindings.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for gated services.")
    parser.add_argument("--context-size", type=int, default=None, help="Context window size for local models.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum reasoning steps.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens per generation turn.")
    parser.add_argument("--skills-dir", type=str, default=None, help="Directory for SKILL.md files.")
    parser.add_argument("--enable-model-switching", action="store_true", help="Allow the agent to switch models.")
    parser.add_argument("--no-shell-execution", action="store_true", help="Disable autonomous shell command execution.")
    parser.add_argument("--shell-autonomy", type=str, default="safe", choices=["safe", "full_access"], help="Autonomy level for shell execution.")
    parser.add_argument("--no-sub-agents", action="store_true", help="Disable sub-agent delegation.")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory.")
    parser.add_argument("--list-skills", action="store_true", help="List all learned skills and exit.")
    parser.add_argument("--clear-history", action="store_true", help="Clear conversation history and exit.")
    parser.add_argument("--config", action="store_true", help="Run configuration wizard and exit.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"lollms_code v{APP_VERSION}")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = CodeAgentConfig.load(args)

    if args.config or not config.is_configured():
        from lollms_client.lollms_config_cli_env import run_wizard_and_save
        run_wizard_and_save()
        config = CodeAgentConfig.load(args)
        config.wizard_completed = True
        config.save()
        if args.config:
            ASCIIColors.green("\n✅ Configuration saved successfully!")
            return 0

    if args.list_skills:
        list_skills(config)
        return 0

    if args.clear_history:
        history_file = APP_CONFIG_DIR / "conversation.json"
        if history_file.exists():
            history_file.unlink()
            ASCIIColors.green("Conversation history cleared.")
        else:
            ASCIIColors.yellow("No conversation history found.")
        return 0

    if args.interactive:
        mode = "interactive"
    elif args.prompt:
        mode = "single"
    else:
        mode = "interactive"

    try:
        client = create_client(config)
    except Exception as e:
        ASCIIColors.red(f"Failed to create LollmsClient: {e}")
        return 1

    personality = create_coding_personality(config, client)

    config.save()

    if mode == "single":
        return run_single_prompt(personality, client, args.prompt, config)
    else:
        return run_interactive(personality, client, config)


if __name__ == "__main__":
    sys.exit(main())