#!/usr/bin/env python3
"""
lollms_code — Autonomous CLI Coding Agent
==========================================

A production-grade CLI tool that turns a single prompt into a full autonomous
coding session. It uses the lollms_client Agent system to:

  1. Analyze the target codebase (workspace context injection)
  2. Plan the implementation strategy
  3. Write code, execute tests, and fix failures iteratively
  4. Create persistent Skills (SKILL.md) from lessons learned
  5. Save episodic memories for cross-session continuity
  6. Delegate sub-tasks to focused child agents when needed
  7. Switch models mid-task for optimal performance

Usage
-----
  # Single-prompt autonomous mode
  lollms-code "Implement a REST API client with retry logic"

  # Interactive REPL mode
  lollms-code -i

  # Target a specific project directory
  lollms-code --workspace ./myproject "add unit tests for all modules"

  # Use a specific model
  lollms-code --model qwen3:32b "refactor the database layer"

  # Enable model switching
  lollms-code --enable-model-switching "build and test a CLI tool"

  # List learned skills
  lollms-code --list-skills

  # Clear conversation history
  lollms-code --clear-history

Configuration
-------------
  The tool reads configuration from (in priority order):
    1. CLI arguments
    2. Environment variables
    3. .env file in the current directory
    4. config.json in ~/.lollms_hub/lollms_code/
    5. Built-in defaults

  Key environment variables:
    LOLLMS_CODE_LLM_BINDING     (default: ollama)
    LOLLMS_CODE_MODEL            (default: qwen3:32b)
    LOLLMS_CODE_HOST             (default: http://localhost:11434)
    LOLLMS_CODE_API_KEY          (default: None)
    LOLLMS_CODE_VERIFY_SSL       (default: false)
    LOLLMS_CODE_CONTEXT_SIZE     (default: 8192)
    LOLLMS_CODE_MAX_STEPS        (default: 100)
    LOLLMS_CODE_TEMPERATURE      (default: 0.3)
    LOLLMS_CODE_MAX_TOKENS       (default: 8192)
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
from pathlib import Path
from typing import Optional, Dict, Any, List

# ── Ensure project source is importable ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ascii_colors import ASCIIColors

from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags, SkillsManager
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE

# ── Constants ──────────────────────────────────────────────────────────────

APP_NAME = "lollms_code"
APP_VERSION = "1.0.0"
APP_CONFIG_DIR = Path.home() / ".lollms_hub" / "lollms_code"
APP_CONFIG_FILE = APP_CONFIG_DIR / "config.json"
APP_DEFAULT_WORKSPACE = Path.cwd()
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"

# ── Default Coding Personality ─────────────────────────────────────────────

CODING_SYSTEM_PROMPT = """\
You are lollms_code, an elite autonomous software engineering agent.

## YOUR IDENTITY
You are not a chatbot. You are a hands-on engineer that writes, tests, and ships code.
You operate in a fully autonomous loop — no human intervention is required.

## WORKFLOW (MANDATORY)
For every task, follow this structured pipeline:

### Phase 1: RECONNAISSANCE
- Use `tool_list_files` to see what already exists in the workspace.
- Use `tool_read_file` to inspect key files relevant to the task.
- If the workspace is empty, start fresh.
- If files exist, understand the architecture before modifying anything.

### Phase 2: PLANNING
- Before writing ANY code, state your plan in 3-5 bullet points.
- Identify which files need to be created, modified, or deleted.
- Identify potential risks or edge cases.

### Phase 3: IMPLEMENTATION
- Use `tool_write_file` to create or overwrite files.
- For EXISTING files with small changes, prefer surgical edits over full rewrites.
- Write clean, production-quality code with proper error handling.
- Include docstrings and type hints where appropriate.

### Phase 4: TESTING & VERIFICATION
- If `tool_execute_python_code` is available, write and run tests.
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
- If the workspace has many files, use `tool_list_files` first.
- Only `tool_read_file` the files you actually need for the current step.
- Do NOT read the same file repeatedly — cache it in your context.
- If you need to reference a file you already read, use the content from your
  previous context, not a new tool call.

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

## MEMORY AWARENESS
- The system injects active memories into your context automatically.
- Use these memories as background knowledge.
- Do NOT output memory markers like `[MEMORY_CONTEXT]` — they are infrastructure.
"""


# ── Configuration ──────────────────────────────────────────────────────────

class CodeAgentConfig:
    """Holds all configuration for the lollms_code CLI."""

    def __init__(self):
        # LLM settings
        self.llm_binding: str = "ollama"
        self.model_name: str = "qwen3:32b"
        self.host_address: str = "http://localhost:11434"
        self.api_key: Optional[str] = None
        self.verify_ssl: bool = False
        self.context_size: int = 8192
        self.n_gpu_layers: int = -1
        self.models_path: str = ""
        self.binaries_path: str = ""

        # Agent settings
        self.max_reasoning_steps: int = 100
        self.temperature: float = 0.3
        self.max_tokens_per_turn: int = 8192

        # Feature flags
        self.enable_code_execution: bool = True
        self.enable_sub_agents: bool = True
        self.enable_model_switching: bool = False
        self.enable_skill_creation: bool = True
        self.enable_skill_loading: bool = True
        self.enable_memory: bool = True
        self.skills_mode: str = "mixed"
        self.max_sub_agent_depth: int = 2
        self.max_sub_agents_per_turn: int = 3

        # Paths
        self.workspace_path: str = str(APP_DEFAULT_WORKSPACE)
        self.skills_dir: str = str(APP_DEFAULT_SKILLS_DIR)
        self.memory_db: str = f"sqlite:///{APP_DEFAULT_MEMORY_DB}"

        # Display
        self.show_tool_calls: bool = True
        self.show_workspace_changes: bool = True
        self.show_skills: bool = True
        self.show_progress: bool = True
        self.debug: bool = False

    @classmethod
    def load(cls, cli_args: argparse.Namespace) -> "CodeAgentConfig":
        """Loads configuration from CLI args, env vars, config file, and defaults."""
        config = cls()

        # 1. Load config file if it exists
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if APP_CONFIG_FILE.exists():
            try:
                file_config = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
                for key, val in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, val)
            except Exception as e:
                ASCIIColors.warning(f"Failed to read config file {APP_CONFIG_FILE}: {e}")

        # 2. Load .env file if python-dotenv is available
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

        # 3. Override with environment variables
        env_map = {
            "LOLLMS_CODE_LLM_BINDING": ("llm_binding", str),
            "LOLLMS_CODE_MODEL": ("model_name", str),
            "LOLLMS_CODE_HOST": ("host_address", str),
            "LOLLMS_CODE_API_KEY": ("api_key", str),
            "LOLLMS_CODE_VERIFY_SSL": ("verify_ssl", lambda v: v.lower() in ("true", "1", "yes")),
            "LOLLMS_CODE_CONTEXT_SIZE": ("context_size", int),
            "LOLLMS_CODE_MAX_STEPS": ("max_reasoning_steps", int),
            "LOLLMS_CODE_TEMPERATURE": ("temperature", float),
            "LOLLMS_CODE_MAX_TOKENS": ("max_tokens_per_turn", int),
            "LOLLMS_CODE_WORKSPACE": ("workspace_path", str),
            "LOLLMS_CODE_SKILLS_DIR": ("skills_dir", str),
            "LOLLMS_CODE_DEBUG": ("debug", lambda v: v.lower() in ("true", "1", "yes")),
        }
        for env_key, (attr, caster) in env_map.items():
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    setattr(config, attr, caster(env_val))
                except (ValueError, TypeError):
                    pass

        # 4. Override with CLI arguments (highest priority)
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
        if cli_args.debug:
            config.debug = True
        if cli_args.enable_model_switching:
            config.enable_model_switching = True
        if cli_args.no_code_execution:
            config.enable_code_execution = False
        if cli_args.no_sub_agents:
            config.enable_sub_agents = False
        if cli_args.no_memory:
            config.enable_memory = False
        if cli_args.skills_dir:
            config.skills_dir = str(Path(cli_args.skills_dir).resolve())

        return config

    def save(self):
        """Persists the current config to the config file."""
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        try:
            APP_CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"Failed to save config: {e}")


# ── Client Creation ────────────────────────────────────────────────────────

def create_client(config: CodeAgentConfig) -> LollmsClient:
    """Creates a LollmsClient based on the configuration."""
    llm_config: Dict[str, Any] = {
        "model_name": config.model_name,
        "host_address": config.host_address,
        "verify_ssl_certificate": config.verify_ssl,
    }
    if config.api_key:
        llm_config["service_key"] = config.api_key

    # For llama_cpp_server, add local-specific config
    if config.llm_binding == "llama_cpp_server":
        llm_config["ctx_size"] = config.context_size
        llm_config["n_gpu_layers"] = config.n_gpu_layers
        if config.models_path:
            llm_config["models_path"] = config.models_path
        if config.binaries_path:
            llm_config["binaries_path"] = config.binaries_path

    default_tools_path = PROJECT_ROOT / "src" / "lollms_client" / "tools_bindings" / "lcp" / "default_tools"

    client = LollmsClient(
        llm_binding_name=config.llm_binding,
        llm_binding_config=llm_config,
        tools_binding_name="lcp",
        tools_binding_config={
            "tools_folders": [str(default_tools_path)] if default_tools_path.exists() else []
        },
    )
    return client


# ── Memory Manager Creation ────────────────────────────────────────────────

def create_memory_manager(config: CodeAgentConfig) -> Optional[Any]:
    """Creates a LollmsMemoryManager if memory is enabled."""
    if not config.enable_memory:
        return None
    try:
        from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
        memory_db_path = config.memory_db
        # Ensure parent directory exists
        db_file = memory_db_path.replace("sqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

        mem_config = MemoryConfig(working_token_budget=2000)
        manager = LollmsMemoryManager(
            db_path=memory_db_path,
            owner_id=f"lollms_code_{Path(config.workspace_path).name}",
            config=mem_config,
        )
        return manager
    except Exception as e:
        ASCIIColors.warning(f"Failed to initialize memory manager: {e}")
        return None


# ── Personality Creation ────────────────────────────────────────────────────

def create_coding_personality() -> LollmsPersonality:
    """Creates the default coding agent personality."""
    return LollmsPersonality(
        name="lollms_code",
        author="ParisNeo",
        category="software_engineering",
        description=(
            "An elite autonomous software engineering agent that writes, tests, "
            "and fixes code iteratively while learning from each session."
        ),
        system_prompt=CODING_SYSTEM_PROMPT,
    )


# ── Agent Creation ──────────────────────────────────────────────────────────

def create_agent(
    config: CodeAgentConfig,
    client: LollmsClient,
    personality: LollmsPersonality,
    memory_manager: Optional[Any],
) -> Agent:
    """Creates a fully configured Agent with all subsystems enabled."""
    # Ensure workspace exists
    workspace = Path(config.workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    # Ensure skills directory exists
    skills_dir = Path(config.skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)

    capabilities = CapabilityFlags(
        enable_code_execution=config.enable_code_execution,
        enable_external_file_access=False,
        enable_networking=False,
        enable_image_generation=False,
        enable_image_editing=False,
        enable_tts=False,
        enable_stt=False,
        enable_ttm=False,
        enable_ttv=False,
        enable_sub_agents=config.enable_sub_agents,
        enable_model_switching=config.enable_model_switching,
        enable_skill_creation=config.enable_skill_creation,
        enable_skill_loading=config.enable_skill_loading,
        skills_mode=config.skills_mode,
        max_sub_agent_depth=config.max_sub_agent_depth,
        max_sub_agents_per_turn=config.max_sub_agents_per_turn,
        enable_workspace_tools=True,
    )

    agent = Agent(
        lc=client,
        personality=personality,
        name="lollms_code",
        role=AgentRole.IMPLEMENTER,
        workspace_path=str(workspace.resolve()),
        capabilities=capabilities,
        skills_dirs=[str(skills_dir.resolve())],
        model_params={"temperature": config.temperature},
        max_tokens_per_turn=config.max_tokens_per_turn,
        memory_manager=memory_manager,
        metadata={
            "version": APP_VERSION,
            "workspace": str(workspace.resolve()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return agent


# ── Streaming Callback ──────────────────────────────────────────────────────

class StreamRenderer:
    """Renders streaming tokens to the terminal with color coding."""

    def __init__(self, config: CodeAgentConfig):
        self.config = config
        self._in_processing = False
        self._tool_call_count = 0

    def __call__(self, chunk: str, msg_type: Any = None, meta: Optional[Dict] = None) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("was_processed"):
                # Processing block content — render in dim color
                if "<processing" in chunk:
                    self._in_processing = True
                    ASCIIColors.cyan(chunk, end="", flush=True)
                elif "</processing>" in chunk:
                    self._in_processing = False
                    ASCIIColors.cyan(chunk, end="", flush=True)
                elif self._in_processing:
                    ASCIIColors.cyan(chunk, end="", flush=True)
                else:
                    # Regular processed content
                    ASCIIColors.white(chunk, end="", flush=True)
            elif meta and meta.get("is_heartbeat"):
                ASCIIColors.yellow(f"\n{chunk}", end="", flush=True)
            else:
                # Normal conversational text — bright white
                ASCIIColors.white(chunk, end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            ASCIIColors.gray(chunk, end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            ASCIIColors.blue(f"\n[INFO] {chunk}", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            if self.config.show_workspace_changes and meta:
                art_type = meta.get("type", "update")
                title = meta.get("title", "")
                if art_type == "artifact_created":
                    ASCIIColors.green(f"\n  📄 Created: {title}")
                elif art_type == "artifact_updated":
                    ASCIIColors.green(f"\n  📝 Updated: {title}")
        elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
            if self.config.show_tool_calls and meta:
                tool_name = meta.get("tool", "unknown")
                self._tool_call_count += 1
                ASCIIColors.magenta(f"\n  🔧 Tool call #{self._tool_call_count}: {tool_name}")
        return True


# ── Result Display ──────────────────────────────────────────────────────────

def display_result(result: Dict[str, Any], config: CodeAgentConfig, elapsed: float):
    """Pretty-prints the structured result from Agent.chat()."""
    print()
    ASCIIColors.cyan("=" * 70)
    ASCIIColors.cyan("📊 SESSION REPORT")
    ASCIIColors.cyan("=" * 70)

    ASCIIColors.white(f"  Total rounds:              {result['rounds']}")
    ASCIIColors.white(f"  Tool calls:                {len(result['tool_calls'])}")
    ASCIIColors.white(f"  Was cancelled:             {result['was_cancelled']}")
    ASCIIColors.white(f"  Elapsed time:              {elapsed:.1f}s")

    if config.show_workspace_changes and result.get("workspace_changes"):
        ASCIIColors.green("\n  📁 Workspace changes:")
        for change in result["workspace_changes"]:
            action = change.get("action", "unknown")
            path = change.get("path", "?")
            size = change.get("size", 0)
            icon = "✨" if action == "created" else "📝" if action == "modified" else "🗑️"
            ASCIIColors.green(f"    {icon} {action}: {path} ({size:,} bytes)")

    if config.show_skills:
        if result.get("skills_created"):
            ASCIIColors.yellow("\n  🎓 Skills created:")
            for skill in result["skills_created"]:
                ASCIIColors.yellow(f"    + {skill}")
        if result.get("skills_updated"):
            ASCIIColors.yellow("\n  🔄 Skills updated:")
            for skill in result["skills_updated"]:
                ASCIIColors.yellow(f"    ↻ {skill}")

    if result.get("sub_agents_spawned", 0) > 0:
        ASCIIColors.magenta(f"\n  🧠 Sub-agents spawned:      {result['sub_agents_spawned']}")

    if result.get("model_switches"):
        ASCIIColors.blue(f"\n  🔄 Model switches:         {result['model_switches']}")

    ASCIIColors.cyan("=" * 70)


# ── Single-Prompt Mode ─────────────────────────────────────────────────────

def run_single_prompt(agent: Agent, prompt: str, config: CodeAgentConfig) -> int:
    """Runs a single autonomous prompt and returns exit code."""
    renderer = StreamRenderer(config)

    ASCIIColors.cyan("\n" + "=" * 70)
    ASCIIColors.cyan(f"🚀 lollms_code v{APP_VERSION} — Autonomous Coding Agent")
    ASCIIColors.cyan("=" * 70)
    ASCIIColors.white(f"  Workspace:  {config.workspace_path}")
    ASCIIColors.white(f"  Model:      {config.model_name}")
    ASCIIColors.white(f"  Binding:    {config.llm_binding}")
    ASCIIColors.white(f"  Max steps:  {config.max_reasoning_steps}")
    ASCIIColors.white(f"  Memory:     {'enabled' if config.enable_memory else 'disabled'}")
    ASCIIColors.white(f"  Skills:     {config.skills_mode}")
    ASCIIColors.white(f"  Sub-agents: {'enabled' if config.enable_sub_agents else 'disabled'}")
    ASCIIColors.cyan("-" * 70)
    ASCIIColors.magenta(f"\n  📝 Task: {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n")
    ASCIIColors.cyan("-" * 70)
    ASCIIColors.white("\n🤖 Agent output:\n")
    ASCIIColors.cyan("-" * 70 + "\n")

    start_time = time.time()

    # Handle Ctrl+C gracefully
    def _signal_handler(sig, frame):
        ASCIIColors.yellow("\n\n⚠️  Interrupt received. Cancelling generation...")
        agent.cancel_generation()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        result = agent.chat(
            prompt=prompt,
            streaming_callback=renderer,
            max_reasoning_steps=config.max_reasoning_steps,
            temperature=config.temperature,
            n_predict=config.max_tokens_per_turn,
            enable_memory=config.enable_memory,
            use_internal_history=False,
        )
    except KeyboardInterrupt:
        agent.cancel_generation()
        ASCIIColors.yellow("\n\n⚠️  Generation cancelled by user.")
        return 130
    except Exception as e:
        ASCIIColors.red(f"\n\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start_time

    display_result(result, config, elapsed)

    # Show the final response
    ASCIIColors.cyan("\n" + "=" * 70)
    ASCIIColors.cyan("📝 FINAL OUTPUT")
    ASCIIColors.cyan("=" * 70)
    print(result.get("response", ""))
    ASCIIColors.cyan("=" * 70)

    return 0 if not result.get("was_cancelled") else 130


# ── Interactive REPL Mode ──────────────────────────────────────────────────

def run_interactive(agent: Agent, config: CodeAgentConfig) -> int:
    """Runs an interactive REPL session."""
    renderer = StreamRenderer(config)

    ASCIIColors.cyan("\n" + "=" * 70)
    ASCIIColors.cyan(f"🚀 lollms_code v{APP_VERSION} — Interactive Mode")
    ASCIIColors.cyan("=" * 70)
    ASCIIColors.white(f"  Workspace:  {config.workspace_path}")
    ASCIIColors.white(f"  Model:      {config.model_name}")
    ASCIIColors.white(f"  Type 'exit' or Ctrl+C to quit.")
    ASCIIColors.white(f"  Type 'skills' to list learned skills.")
    ASCIIColors.white(f"  Type 'clear' to clear conversation history.")
    ASCIIColors.white(f"  Type 'models' to list available models.")
    ASCIIColors.cyan("=" * 70 + "\n")

    while True:
        try:
            ASCIIColors.magenta("\n👤 You> ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            ASCIIColors.cyan("\n\n👋 Goodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", ":q"):
            ASCIIColors.cyan("👋 Goodbye!")
            return 0
        if user_input.lower() == "skills":
            skills = agent.list_skills()
            if not skills:
                ASCIIColors.yellow("  No skills learned yet.")
            else:
                ASCIIColors.green(f"\n  📚 Learned Skills ({len(skills)}):")
                for s in skills:
                    vis = " [always visible]" if s.get("always_visible") else ""
                    cat = f" [{s.get('category', '')}]" if s.get("category") else ""
                    ASCIIColors.white(f"    • {s['title']}{cat}{vis}: {s.get('description', '')}")
            continue
        if user_input.lower() == "clear":
            agent.clear_conversation()
            ASCIIColors.green("  Conversation history cleared.")
            continue
        if user_input.lower() == "models":
            models = agent.list_available_models()
            current = agent.get_current_model()
            if not models:
                ASCIIColors.yellow("  No models available for switching.")
            else:
                ASCIIColors.green(f"\n  📋 Available models (current: {current}):")
                for m in models:
                    marker = " ← current" if m == current else ""
                    ASCIIColors.white(f"    • {m}{marker}")
            continue

        # Run the prompt
        ASCIIColors.cyan("\n🤖 Agent> ")
        ASCIIColors.cyan("-" * 50)

        start_time = time.time()
        try:
            result = agent.chat(
                prompt=user_input,
                streaming_callback=renderer,
                max_reasoning_steps=config.max_reasoning_steps,
                temperature=config.temperature,
                n_predict=config.max_tokens_per_turn,
                enable_memory=config.enable_memory,
                use_internal_history=True,
            )
        except KeyboardInterrupt:
            agent.cancel_generation()
            ASCIIColors.yellow("\n\n⚠️  Cancelled.")
            continue
        except Exception as e:
            ASCIIColors.red(f"\n💥 Error: {e}")
            continue

        elapsed = time.time() - start_time

        ASCIIColors.cyan(f"\n  ⏱️  {elapsed:.1f}s | Rounds: {result['rounds']} | Tools: {len(result['tool_calls'])}")

        if result.get("skills_created") or result.get("skills_updated"):
            all_skills = (result.get("skills_created", []) or []) + (result.get("skills_updated", []) or [])
            ASCIIColors.yellow(f"  🎓 Skills: {', '.join(all_skills)}")

        if result.get("workspace_changes"):
            changes = result["workspace_changes"]
            created = [c for c in changes if c["action"] == "created"]
            modified = [c for c in changes if c["action"] == "modified"]
            if created:
                ASCIIColors.green(f"  ✨ Created: {', '.join(c['path'] for c in created)}")
            if modified:
                ASCIIColors.green(f"  📝 Modified: {', '.join(c['path'] for c in modified)}")


# ── Skills Listing ─────────────────────────────────────────────────────────

def list_skills(config: CodeAgentConfig):
    """Lists all learned skills."""
    skills_dir = Path(config.skills_dir)
    if not skills_dir.exists():
        ASCIIColors.yellow("No skills directory found. Run a task first to generate skills.")
        return

    mgr = SkillsManager(skills_dirs=[str(skills_dir)], mode="loadable", default_skills_dir=str(skills_dir))
    skills = mgr.list_skills()
    if not skills:
        ASCIIColors.yellow("No skills learned yet.")
        return

    ASCIIColors.green(f"\n📚 Learned Skills ({len(skills)}):\n")
    for s in skills:
        vis = " [always visible]" if s.get("always_visible") else ""
        cat = f" [{s.get('category', '')}]" if s.get("category") else ""
        desc = s.get("description", "No description")
        ASCIIColors.white(f"  • {s['title']}{cat}{vis}")
        ASCIIColors.gray(f"    {desc}")
    print()


# ── Argument Parser ────────────────────────────────────────────────────────

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

    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The task prompt for the autonomous agent. If omitted, enters interactive mode (unless -i is set).",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start in interactive REPL mode.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Path to the workspace directory (default: current directory).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., qwen3:32b, gpt-4o).",
    )
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=None,
        help="LLM binding name (ollama, openai, llama_cpp_server, etc.).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address for remote bindings (e.g., http://localhost:11434).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for gated services (OpenAI, Mistral, etc.).",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Context window size for local models (default: 8192).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum reasoning steps (default: 100). Increase for complex loops.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.3 for code, 0.7 for creative).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per generation turn (default: 8192).",
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        default=None,
        help="Directory for SKILL.md files (default: ~/.lollms_hub/lollms_code/skills).",
    )
    parser.add_argument(
        "--enable-model-switching",
        action="store_true",
        help="Allow the agent to switch models mid-task.",
    )
    parser.add_argument(
        "--no-code-execution",
        action="store_true",
        help="Disable Python code execution (safer but less capable).",
    )
    parser.add_argument(
        "--no-sub-agents",
        action="store_true",
        help="Disable sub-agent delegation.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent memory (no cross-session learning).",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List all learned skills and exit.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear the agent's conversation history file and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lollms_code v{APP_VERSION}",
    )

    return parser


# ── Main Entry Point ───────────────────────────────────────────────────────

def main():
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load configuration
    config = CodeAgentConfig.load(args)

    # Handle non-agent commands
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

    # Determine mode
    if args.interactive:
        mode = "interactive"
    elif args.prompt:
        mode = "single"
    else:
        # No prompt and no -i flag — default to interactive
        mode = "interactive"

    # Create components
    try:
        client = create_client(config)
    except Exception as e:
        ASCIIColors.red(f"Failed to create LollmsClient: {e}")
        import traceback
        traceback.print_exc()
        return 1

    personality = create_coding_personality()
    memory_manager = create_memory_manager(config)
    agent = create_agent(config, client, personality, memory_manager)

    # Save config for future sessions
    config.save()

    # Run
    if mode == "single":
        return run_single_prompt(agent, args.prompt, config)
    else:
        return run_interactive(agent, config)


if __name__ == "__main__":
    sys.exit(main())
