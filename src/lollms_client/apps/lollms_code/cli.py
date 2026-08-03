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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ascii_colors import ASCIIColors
from ascii_colors import questionary

from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags, SkillsManager
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE

APP_NAME = "lollms_code"
APP_VERSION = "1.0.0"
APP_CONFIG_DIR = Path.home() / ".lollms_hub" / "lollms_code"
APP_CONFIG_FILE = APP_CONFIG_DIR / "config.json"
APP_DEFAULT_WORKSPACE = Path.cwd()
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"

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
        self.enable_code_execution: bool = True
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
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        try:
            APP_CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"Failed to save config: {e}")

    def is_configured(self) -> bool:
        return self.wizard_completed and bool(self.llm_binding) and bool(self.llm_binding_config)


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


def create_memory_manager(config: CodeAgentConfig) -> Optional[Any]:
    if not config.enable_memory:
        return None
    try:
        from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
        db_file = config.memory_db.replace("sqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        mem_config = MemoryConfig(working_token_budget=2000)
        manager = LollmsMemoryManager(
            db_path=config.memory_db,
            owner_id=f"lollms_code_{Path(config.workspace_path).name}",
            config=mem_config,
        )
        return manager
    except Exception as e:
        ASCIIColors.warning(f"Failed to initialize memory manager: {e}")
        return None


def create_coding_personality() -> LollmsPersonality:
    return LollmsPersonality(
        name="lollms_code",
        author="ParisNeo",
        category="software_engineering",
        description="An elite autonomous software engineering agent that writes, tests, and fixes code iteratively.",
        system_prompt=CODING_SYSTEM_PROMPT,
    )


def create_agent(
    config: CodeAgentConfig,
    client: LollmsClient,
    personality: LollmsPersonality,
    memory_manager: Optional[Any],
) -> Agent:
    workspace = Path(config.workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
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
        metadata={"version": APP_VERSION, "workspace": str(workspace.resolve())},
    )
    return agent


class StreamRenderer:
    def __init__(self, config: CodeAgentConfig):
        self.config = config
        self._in_processing = False
        self._tool_call_count = 0

    def __call__(self, chunk: str, msg_type: Any = None, meta: Optional[Dict] = None) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("was_processed"):
                if "<processing" in chunk:
                    self._in_processing = True
                    ASCIIColors.rich_print(f"\n[dim cyan]{chunk}[/dim cyan]", end="")
                elif "</processing>" in chunk:
                    self._in_processing = False
                    ASCIIColors.rich_print(f"[dim cyan]{chunk}[/dim cyan]", end="")
                elif self._in_processing:
                    ASCIIColors.rich_print(f"[dim cyan]{chunk}[/dim cyan]", end="")
                else:
                    ASCIIColors.rich_print(chunk, end="")
            else:
                ASCIIColors.rich_print(chunk, end="")
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            ASCIIColors.rich_print(f"[dim]{chunk}[/dim]", end="")
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            ASCIIColors.rich_print(f"\n[blue][INFO] {chunk}[/blue]")
        return True


def display_result(result: Dict[str, Any], config: CodeAgentConfig, elapsed: float):
    ASCIIColors.rule("[bold cyan]📊 SESSION REPORT[/bold cyan]")
    
    summary_table = ASCIIColors.table(
        "Metric", "Value",
        rows=[
            ["Total rounds", str(result['rounds'])],
            ["Tool calls", str(len(result['tool_calls']))],
            ["Was cancelled", str(result['was_cancelled'])],
            ["Elapsed time", f"{elapsed:.1f}s"]
        ],
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
            rows=([["Created", s] for s in result["skills_created"]] + 
                  [["Updated", s] for s in result["skills_updated"]]),
            title="[bold yellow]🎓 Skills Activity[/bold yellow]",
            box="round"
        )
        ASCIIColors.rich_print(skills_table)

    if result.get("sub_agents_spawned", 0) > 0:
        ASCIIColors.magenta(f"\n  🧠 Sub-agents spawned: {result['sub_agents_spawned']}")

    if result.get("model_switches"):
        ASCIIColors.blue(f"  🔄 Model switches: {result['model_switches']}")


def run_single_prompt(agent: Agent, prompt: str, config: CodeAgentConfig) -> int:
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
    
    ASCIIColors.rule("[bold]🤖 Agent output[/bold]")

    start_time = time.time()

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
        return 1

    elapsed = time.time() - start_time
    display_result(result, config, elapsed)

    ASCIIColors.panel(result.get("response", ""), title="[bold cyan]📝 FINAL OUTPUT[/bold cyan]", border_style="cyan")

    return 0 if not result.get("was_cancelled") else 130


def run_interactive(agent: Agent, config: CodeAgentConfig) -> int:
    renderer = StreamRenderer(config)

    ASCIIColors.panel(
        f"[cyan]Workspace:[/cyan] {config.workspace_path}\n[cyan]Model:[/cyan] {config.model_name}\n[dim]Type 'exit' or Ctrl+C to quit.[/dim]",
        title=f"[bold green]🚀 lollms_code v{APP_VERSION} — Interactive Mode[/bold green]",
        border_style="green"
    )

    while True:
        try:
            user_input = questionary.text("👤 You>").ask()
        except (EOFError, KeyboardInterrupt):
            ASCIIColors.cyan("\n👋 Goodbye!")
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
                skills_table = ASCIIColors.table(
                    "Title", "Category", "Description",
                    rows=[[s['title'], s.get('category', ''), s.get('description', '')] for s in skills],
                    title="[bold yellow]📚 Learned Skills[/bold yellow]",
                    box="round"
                )
                ASCIIColors.rich_print(skills_table)
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
                models_table = ASCIIColors.table(
                    "Model", "Status",
                    rows=[[m, "← current" if m == current else ""] for m in models],
                    title="[bold blue]📋 Available Models[/bold blue]",
                    box="round"
                )
                ASCIIColors.rich_print(models_table)
            continue
        if user_input.lower() == "config":
            from lollms_client.lollms_config_cli_env import run_wizard_and_save
            run_wizard_and_save()
            ASCIIColors.green("  Configuration updated. Restart lollms-code for changes to take effect.")
            continue

        ASCIIColors.rule("[bold green]🤖 Agent[/bold green]")

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
        ASCIIColors.rich_print(f"\n[dim]⏱️  {elapsed:.1f}s | Rounds: {result['rounds']} | Tools: {len(result['tool_calls'])}[/dim]")


def list_skills(config: CodeAgentConfig):
    skills_dir = Path(config.skills_dir)
    if not skills_dir.exists():
        ASCIIColors.yellow("No skills directory found. Run a task first to generate skills.")
        return

    mgr = SkillsManager(skills_dirs=[str(skills_dir)], mode="loadable", default_skills_dir=str(skills_dir))
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
    parser.add_argument("--no-code-execution", action="store_true", help="Disable Python code execution.")
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

    personality = create_coding_personality()
    memory_manager = create_memory_manager(config)
    agent = create_agent(config, client, personality, memory_manager)

    config.save()

    if mode == "single":
        return run_single_prompt(agent, args.prompt, config)
    else:
        return run_interactive(agent, config)


if __name__ == "__main__":
    sys.exit(main())