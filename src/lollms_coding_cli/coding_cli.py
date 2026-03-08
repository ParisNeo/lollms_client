#!/usr/bin/env python3
# coding_cli.py  –  lollms Agentic Coding Assistant
#
# Install:  pip install rich questionary pyyaml ascii_colors lollms_client
# Run:      python coding_cli.py
#
# All state lives under ~/.lollms/coding_cli/   (never committed to repos)
#   config.yaml        –  LLM binding + runtime settings
#   projects.yaml      –  project registry
#   scratchpad.yaml    –  global inter-agent key-value store
#   projects/<n>.db    –  SQLite discussion databases
#
# Per-project:   <output_folder>/.lollms/memory.yaml
#
# ── Slash commands ──────────────────────────────────────────────────────────
#
#  Setup
#   /setup                   re-run full setup wizard
#   /setup show              print current config (secrets masked)
#
#  Projects
#   /project new <n> [dir]   create & switch to a new project
#   /project switch <n>      switch to an existing project
#   /project ls              list all projects
#   /project info            current project details
#   /project delete <n>      remove from registry (files kept)
#
#  Scratchpad  (global inter-agent / inter-task key-value store)
#   /scratch set <key> <value>   write an entry
#   /scratch get <key>           read an entry
#   /scratch ls                  list all entries
#   /scratch del <key>           delete an entry
#   /scratch clear               wipe everything (with confirmation)
#
#  Project long-term memory  (per-project, injected into LLM context)
#   /memory add <text>           append a memory entry
#   /memory ls                   list all entries
#   /memory del <id>             delete by numeric id
#   /memory clear                wipe (with confirmation)
#   /memory search <query>       case-insensitive keyword search
#
#  Artefacts
#   /ls                          list artefacts in current discussion
#   /show <n>                    syntax-highlighted view
#   /activate <n>                inject into LLM context
#   /deactivate <n>              remove from context
#
#  Files, execution & tests
#   /commit [folder]             write code artefacts to disk
#   /run <file> [args…]          execute in project folder
#   /shell <command>             shell command in project folder
#   /test [path]                 run pytest
#
#  Git  (runs inside the project output folder)
#   /git init | status | add [files] | commit <msg>
#       diff [file] | log [n] | branch [name] | checkout <branch>
#
#  Conversation
#   /clear                       new discussion (same project)
#   /history                     show conversation
#   /help                        this list
#   /quit                        exit
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml  # PyYAML  (pip install pyyaml)
import json
# ── optional rich / questionary ───────────────────────────────────────────────
# Both are soft-required: we degrade gracefully if missing but tell the user.

from ascii_colors import Console, Markdown, Panel, Confirm, Prompt, Rule, Syntax, Table
import re 
# Prompt toolkit for advanced REPL features
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter, NestedCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False

# questionary — used for interactive selection menus
_HAS_QUESTIONARY = True
# ascii_colors — optional; just used for colour helpers
from ascii_colors import ASCIIColors, trace_exception, questionary, Style as QStyle
from ascii_colors.rich import Text

from lollms_client import LollmsClient

from lollms_client.lollms_discussion import (
    ArtefactType,
    LollmsDataManager,
    LollmsDiscussion,
)
from lollms_client.lollms_types import MSG_TYPE

# ═════════════════════════════════════════════════════════════════════════════
# Paths (must be defined before Soul & Skills)
# ═════════════════════════════════════════════════════════════════════════════

def _documents_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes, ctypes.wintypes
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            p = Path(buf.value)
            if p.exists():
                return p
        except Exception:
            pass
        return Path.home() / "Documents"
    elif system == "Darwin":
        return Path.home() / "Documents"
    else:
        xdg = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
        return Path(xdg) if xdg else Path.home() / "Documents"


CONFIG_DIR      = Path.home() / ".lollms" / "coding_cli"
CONFIG_FILE     = CONFIG_DIR / "config.yaml"
PROJECTS_FILE   = CONFIG_DIR / "projects.yaml"
SCRATCHPAD_FILE = CONFIG_DIR / "scratchpad.yaml"
HISTORY_FILE    = CONFIG_DIR / ".cli_history"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Soul & Skills (Claude Code-style architecture)
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_SOUL = """\
You are an expert software engineer and coding assistant. Your purpose is to help users write, understand, and improve code efficiently.

CORE PRINCIPLES:
- Write clean, maintainable, well-documented code
- Prefer explicit over implicit; avoid clever tricks that obscure logic
- Use appropriate design patterns; don't over-engineer simple tasks
- Always consider security, performance, and edge cases
- Explain your reasoning when making architectural decisions

COMMUNICATION STYLE:
- Be concise but thorough; no unnecessary verbosity
- Admit uncertainty rather than hallucinating
- Ask clarifying questions when requirements are ambiguous
- Use the environment context to avoid generating invalid commands

When given a task, analyze requirements, propose an approach, then implement with the tools available.
"""

SOUL_FILE = CONFIG_DIR / "soul.md"
SKILLS_DIR = CONFIG_DIR / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def load_soul() -> str:
    """Load persistent system personality (the 'soul')."""
    if SOUL_FILE.exists():
        try:
            return SOUL_FILE.read_text(encoding="utf-8")
        except Exception:
            pass
    # Create default soul
    SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
    return DEFAULT_SOUL


def save_soul(content: str) -> None:
    """Save soul to disk."""
    SOUL_FILE.write_text(content, encoding="utf-8")


class Skill:
    """A reusable skill module with name, description, and system prompt additions."""
    
    def __init__(self, name: str, description: str, prompt_addition: str,
                 category: str = "general", enabled: bool = True):
        self.name = name
        self.description = description
        self.prompt_addition = prompt_addition
        self.category = category
        self.enabled = enabled
        self.file_path = SKILLS_DIR / f"{name}.md"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "path": str(self.file_path),
        }
    
    def save(self) -> None:
        """Persist skill to disk."""
        lines = [
            f"<!-- SKILL: {self.name} -->",
            f"<!-- CATEGORY: {self.category} -->",
            f"<!-- DESCRIPTION: {self.description} -->",
            f"<!-- ENABLED: {self.enabled} -->",
            "",
            "# System Prompt Addition",
            "",
            self.prompt_addition,
        ]
        self.file_path.write_text("\n".join(lines), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> "Skill":
        """Load skill from markdown file with YAML-like frontmatter."""
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # Parse frontmatter
        name = path.stem
        description = ""
        category = "general"
        enabled = True
        prompt_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith("<!-- SKILL:"):
                name = line.split(":", 1)[1].replace("-->", "").strip()
            elif line.startswith("<!-- DESCRIPTION:"):
                description = line.split(":", 1)[1].replace("-->", "").strip()
            elif line.startswith("<!-- CATEGORY:"):
                category = line.split(":", 1)[1].replace("-->", "").strip()
            elif line.startswith("<!-- ENABLED:"):
                enabled = line.split(":", 1)[1].replace("-->", "").strip().lower() == "true"
            elif line == "# System Prompt Addition":
                prompt_start = i + 1
                break
        
        prompt_addition = "\n".join(lines[prompt_start:]).strip()
        skill = cls(name, description, prompt_addition, category, enabled)
        skill.file_path = path
        return skill
    
    def render(self) -> str:
        """Render skill as text for inclusion in system prompt."""
        if not self.enabled:
            return ""
        return f"\n## Skill: {self.name}\n{self.prompt_addition}\n"


def load_skills() -> list[Skill]:
    """Load all available skills."""
    skills = []
    if SKILLS_DIR.exists():
        for path in sorted(SKILLS_DIR.glob("*.md")):
            try:
                skills.append(Skill.load(path))
            except Exception:
                pass
    return skills


def get_enabled_skills_prompt(skills: list[Skill]) -> str:
    """Concatenate enabled skills into prompt section."""
    enabled = [s for s in skills if s.enabled]
    if not enabled:
        return ""
    parts = ["## Active Skills"]
    for s in enabled:
        parts.append(f"\n### {s.name} ({s.category})\n{s.prompt_addition}")
    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# Reset marker file — if present, force first-run experience
# ═════════════════════════════════════════════════════════════════════════════

RESET_MARKER = CONFIG_DIR / ".reset_marker"


def _is_fresh_install() -> bool:
    """Check if a reset marker exists (indicating user requested fresh start)."""
    if RESET_MARKER.exists():
        try:
            RESET_MARKER.unlink()
        except Exception:
            pass
        return True
    return not CONFIG_FILE.exists()

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
# Binding catalogue
# All bindings from the lollms_client llm_bindings directory + their config.
# ═════════════════════════════════════════════════════════════════════════════

# Each entry:  name  →  {label, fields: [{name, label, secret, default, required}], needs_model_list}
# fields with secret=True are masked in /setup show.
# needs_model_list=True  →  we try to list models after connecting.

BINDING_CATALOG: dict[str, dict] = {
    "lollms": {
        "label": "LoLLMs (local server)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address",          "label": "Host address",           "secret": False, "default": "http://localhost:9642"},
            {"name": "service_key",            "label": "Service key (optional)", "secret": True,  "default": ""},
            {"name": "verify_ssl_certificate", "label": "Verify SSL cert",        "secret": False, "default": "true"},
        ],
    },
    "ollama": {
        "label": "Ollama (local server)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "Host address", "secret": False, "default": "http://localhost:11434"},
        ],
    },
    "openai": {
        "label": "OpenAI",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "API key (sk-…)", "secret": True, "default": ""},
        ],
    },
    "azure_openai": {
        "label": "Azure OpenAI",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key",   "label": "Azure API key",      "secret": True,  "default": ""},
            {"name": "host_address",  "label": "Azure endpoint URL",  "secret": False, "default": ""},
            {"name": "api_version",   "label": "API version",         "secret": False, "default": "2024-02-01"},
        ],
    },
    "claude": {
        "label": "Anthropic Claude",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Anthropic API key", "secret": True, "default": ""},
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Google AI Studio key", "secret": True, "default": ""},
        ],
    },
    "groq": {
        "label": "Groq",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Groq API key", "secret": True, "default": ""},
        ],
    },
    "open_router": {
        "label": "OpenRouter",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "OpenRouter key (sk-or-…)", "secret": True, "default": ""},
        ],
    },
    "mistral": {
        "label": "Mistral AI",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Mistral API key", "secret": True, "default": ""},
        ],
    },
    "hugging_face_inference_api": {
        "label": "Hugging Face Inference API",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "HF access token (hf_…)", "secret": True, "default": ""},
        ],
    },
    "perplexity": {
        "label": "Perplexity AI",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Perplexity API key", "secret": True, "default": ""},
        ],
    },
    "novita_ai": {
        "label": "Novita AI",
        "needs_model_list": False,
        "fields": [
            {"name": "service_key", "label": "Novita API key", "secret": True, "default": ""},
        ],
    },
    "openllm": {
        "label": "OpenLLM (local server)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "Server address", "secret": False, "default": "http://localhost:3000"},
        ],
    },
    "openwebui": {
        "label": "Open WebUI (local server)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "Server address", "secret": False, "default": "http://localhost:3000"},
            {"name": "service_key",  "label": "API key (optional)", "secret": True, "default": ""},
        ],
    },
    "lollms_webui": {
        "label": "LoLLMs WebUI",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "WebUI address", "secret": False, "default": "http://localhost:9600"},
            {"name": "service_key",  "label": "Service key (optional)", "secret": True, "default": ""},
        ],
    },
    "llama_cpp_server": {
        "label": "llama.cpp server (local)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "Server address", "secret": False, "default": "http://localhost:8080"},
        ],
    },
    "litellm": {
        "label": "LiteLLM proxy",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "LiteLLM proxy address", "secret": False, "default": "http://localhost:4000"},
            {"name": "service_key",  "label": "API key (optional)",    "secret": True,  "default": ""},
        ],
    },
    "pythonllamacpp": {
        "label": "llama-cpp-python (local GGUF)",
        "needs_model_list": False,
        "fields": [
            {"name": "model_path",   "label": "Path to .gguf file",      "secret": False, "default": ""},
            {"name": "n_gpu_layers", "label": "GPU layers (-1 = all)",    "secret": False, "default": "-1"},
            {"name": "n_ctx",        "label": "Context size",             "secret": False, "default": "4096"},
            {"name": "chat_format",  "label": "Chat format (e.g. chatml)","secret": False, "default": "chatml"},
        ],
    },
    "transformers": {
        "label": "Hugging Face Transformers (local)",
        "needs_model_list": False,
        "fields": [
            {"name": "model_path",    "label": "Model name or local path", "secret": False, "default": ""},
            {"name": "device",        "label": "Device (cpu/cuda/mps)",    "secret": False, "default": "cpu"},
        ],
    },
    "tensor_rt": {
        "label": "TensorRT-LLM (local)",
        "needs_model_list": False,
        "fields": [
            {"name": "model_path", "label": "Path to TRT engine", "secret": False, "default": ""},
        ],
    },
    "vllm": {
        "label": "vLLM (local server)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "vLLM server address", "secret": False, "default": "http://localhost:8000"},
            {"name": "service_key",  "label": "API key (optional)",  "secret": True,  "default": ""},
        ],
    },
}

# Binding name → default model names (used when live listing is unavailable)
BINDING_DEFAULT_MODELS: dict[str, list[str]] = {
    "openai":       ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "azure_openai": ["gpt-4o", "gpt-4-turbo", "gpt-35-turbo"],
    "claude":       ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    "gemini":       ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro"],
    "groq":         ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    "open_router":  ["anthropic/claude-3-haiku", "openai/gpt-4o", "meta-llama/llama-3-70b-instruct"],
    "mistral":      ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
    "hugging_face_inference_api": ["google/gemma-1.1-7b-it", "meta-llama/Llama-2-7b-chat-hf"],
    "perplexity":   ["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
    "novita_ai":    ["meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct"],
}

EXT_MAP: dict[str, str] = {
    "python": ".py",      "py": ".py",
    "javascript": ".js",  "js": ".js",
    "typescript": ".ts",  "ts": ".ts",
    "html": ".html",      "css": ".css",
    "bash": ".sh",        "shell": ".sh",
    "rust": ".rs",        "go": ".go",
    "c": ".c",            "cpp": ".cpp",  "c++": ".cpp",
    "java": ".java",      "kotlin": ".kt",
    "json": ".json",      "yaml": ".yml", "toml": ".toml",
    "markdown": ".md",    "md": ".md",    "sql": ".sql",
}
CODE_TYPES = {ArtefactType.CODE, ArtefactType.DOCUMENT, ArtefactType.FILE}

_SENSITIVE = ("key", "secret", "token", "password", "passwd", "credential")

SYSTEM_PROMPT_TEMPLATE = """\
{base_personality}

## Environment Context (CRITICAL — always check before executing commands)
{env_section}
{skills_section}

## Project memory
{memory_section}

## Scratchpad (shared inter-agent notes)
{scratchpad_section}

## Artefacts
Wrap all code in artefact tags (always use a filename with its extension):

<artefact name="filename.py" type="code" language="python">
[code here]
</artefact>

To PATCH existing code use aider SEARCH/REPLACE inside the artefact tag:

<artefact name="filename.py">
<<<<<<< SEARCH
[exact lines to find]
=======
[replacement lines]
>>>>>>> REPLACE
</artefact>

## Agent tools — MANDATORY USAGE
When the user asks you to perform an action (run code, execute a file, 
make a git commit, etc.), you MUST use the appropriate tool. 
Do NOT describe what you would do — actually invoke the tool.

Tool invocation format:
<tool_call>{{"name": "tool_name", "parameters": {{"key": "value"}}}}</tool_call>

Available tools:
- run_code(filename, args="")   execute a file in the project folder
- run_shell(command)            shell command in the project folder
- run_tests(path=".")           run pytest and get results
- read_file(path)               read any project file
- write_file(path, content)     write or overwrite a project file
- list_files(path=".")          list files in a directory
- scratchpad_read(key)          read a global scratchpad entry
- scratchpad_write(key, value)  write a global scratchpad entry
- memory_add(text)              add an entry to project long-term memory
- git_status()                  current git status
- git_diff(file="")             uncommitted changes
- git_add(files=".")            stage files
- git_commit(message)           commit staged changes
- git_log(n=10)                 recent commit history

## Workflow
1. Write the code as artefacts.
2. Call run_code() or run_tests() to verify.
3. If it fails, read the error, patch the artefact, run again.
4. Repeat until tests pass, then call final_answer().

## IMPORTANT: Running Servers and Long-Running Processes
When the user asks to "run the server", "start the app", or any indefinite process:
- ALWAYS use run_code(filename="main.py", capture_seconds=5.0, wait_for_exit=False)
- This returns immediately with a PID you can poll/stop later
- NEVER use wait_for_exit=True for servers (it will timeout!)
- After starting, use read_process_output(pid) to check status
"""


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mask(key: str, val: str) -> str:
    return "****" if any(w in key.lower() for w in _SENSITIVE) else val


def _q_style() -> "QStyle":
    return QStyle([
        ("qmark",        "fg:#5f87ff bold"),
        ("question",     "bold"),
        ("answer",       "fg:#5fffff bold"),
        ("pointer",      "fg:#5f87ff bold"),
        ("highlighted",  "fg:#5f87ff bold"),
        ("selected",     "fg:#5fffff"),
        ("separator",    "fg:#5f87ff"),
        ("instruction",  "fg:#858585"),
    ])


def _ask_select(message: str, choices: list[str], default: str = None) -> str:
    """Use questionary select if available, else Rich numbered menu."""
    if _HAS_QUESTIONARY:
        result = questionary.select(
            message, choices=choices, default=default, style=_q_style()
        ).ask()
        return result or (default or choices[0])
    # Fallback: numbered list
    console.print(f"\n[bold]{message}[/bold]")
    for i, c in enumerate(choices, 1):
        marker = " [green]◀[/green]" if c == default else ""
        console.print(f"  [cyan]{i:2}[/cyan].  {c}{marker}")
    while True:
        raw = Prompt.ask("  Enter number or value").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        console.print(f"  [red]Invalid choice. Enter 1–{len(choices)} or the exact name.[/red]")


def _ask_text(message: str, default: str = "") -> str:
    # Build prompt with default clearly visible
    if default:
        display = f"{message} [dim](default: {default})[/dim]"
    else:
        display = message
    
    if _HAS_QUESTIONARY:
        result = questionary.text(message, default=default, style=_q_style()).ask()
        return result if result is not None else default
    
    # Rich fallback - show default in brackets
    result = Prompt.ask(display, default=default, show_default=False)
    return result.strip() if result else default


def _ask_password(message: str) -> str:
    if _HAS_QUESTIONARY:
        result = questionary.password(message, style=_q_style()).ask()
        return result or ""
    return Prompt.ask(message, password=True)


def _ask_confirm(message: str, default: bool = False) -> bool:
    if _HAS_QUESTIONARY:
        result = questionary.confirm(message, default=default, style=_q_style()).ask()
        return result if result is not None else default
    return Confirm.ask(message, default=default)


# ═════════════════════════════════════════════════════════════════════════════
# Config  (~/.lollms/coding_cli/config.yaml)
# ═════════════════════════════════════════════════════════════════════════════

def _default_config() -> dict:
    docs = _documents_dir()
    return {
        "binding_name":      "lollms",
        "binding_config":    {},
        "projects_root":     str(docs / "lollms_projects"),
        "python_executable": sys.executable,
        "git_auto_commit":   False,
        "shell_timeout":     30,
        "max_output_chars":  8000,
        "show_thinking":     True,
    }


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
            base = _default_config()
            base.update(data)
            return base
        except Exception:
            pass
    return _default_config()


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _try_list_models(binding_name: str, bc: dict) -> list[dict]:
    """
    Connect to the binding and return a list of model dicts.
    Each dict is guaranteed to have at least 'model_name'.
    Extra fields (owned_by, context_length, max_generation, created) are
    preserved when the server provides them.
    Returns [] on any failure.
    """
    try:
        lc_tmp = LollmsClient(
            llm_binding_name=binding_name,
            llm_binding_config=bc,
        )
        raw = lc_tmp.list_models()
        if not isinstance(raw, list) or not raw:
            return []
        models: list[dict] = []
        for m in raw:
            if isinstance(m, dict):
                name = (m.get("model_name") or m.get("name") or
                        m.get("id") or "").strip()
                if not name:
                    continue
                models.append({
                    "model_name":      name,
                    "owned_by":        m.get("owned_by", ""),
                    "context_length":  m.get("context_length"),
                    "max_generation":  m.get("max_generation"),
                })
            elif isinstance(m, str) and m.strip():
                models.append({"model_name": m.strip(), "owned_by": "",
                               "context_length": None, "max_generation": None})
        return models
    except Exception:
        return []


def _show_models_table(models: list[dict]) -> None:
    """Print a rich table of available models (no truncation of names)."""
    t = Table(show_header=True, header_style="bold cyan",
            expand=False, show_lines=False)
    t.add_column("#",              style="dim",        justify="right", width=4)
    t.add_column("Model name",     style="bold white",  no_wrap=True)
    t.add_column("Owner",          style="cyan",        no_wrap=True)
    t.add_column("Context",        style="green",       justify="right")
    t.add_column("Max gen",        style="yellow",      justify="right")
    for i, m in enumerate(models, 1):
        ctx = str(m["context_length"]) if m["context_length"] is not None else "—"
        mxg = str(m["max_generation"]) if m["max_generation"] is not None else "—"
        t.add_row(str(i), m["model_name"], m["owned_by"] or "—", ctx, mxg)
    console.print(t)


def _pick_model(models: list[dict], current_model: str) -> str:
    """
    Show a rich table, then let the user pick with questionary autocomplete
    (type to filter by name) or fall back to a numbered prompt.
    Returns the chosen model_name string.
    """
    _show_models_table(models)
    names = [m["model_name"] for m in models]
    
    # Determine what to show as default
    default_name = current_model if current_model in names else (names[0] if names else "")
    default_hint = f" [dim](press Enter for: {default_name})[/dim]" if default_name else ""

    if _HAS_QUESTIONARY:
        # autocomplete: user types part of a name, list filters live
        result = questionary.autocomplete(
            f"Select model (type to filter ↑↓ to navigate):{default_hint}",
            choices=names,
            default=default_name,
            style=_q_style(),
            validate=lambda v: v in names or f"'{v}' is not in the list",
        ).ask()
        return result or current_model or default_name
    else:
        # Fallback: accept index or exact name, show default clearly
        while True:
            prompt_text = f"  Enter model number (1–{len(names)}) or name"
            if default_name:
                prompt_text += f" [dim](default: {default_name})[/dim]"
            raw = Prompt.ask(prompt_text, default=default_name, show_default=False).strip()
            if not raw and default_name:
                return default_name
            if raw.isdigit() and 1 <= int(raw) <= len(names):
                return names[int(raw) - 1]
            if raw in names:
                return raw
            console.print(f"  [red]Not found. Try again.[/red]")


def run_setup_wizard(existing: Optional[dict] = None, first_run: bool = False) -> dict:
    """
    Full interactive setup wizard.
    Uses questionary when available, otherwise falls back to Rich prompts.
    """
    cfg = existing or _default_config()

    console.print(Panel(
        "[bold cyan]lollms Coding Assistant – Setup Wizard[/bold cyan]\n"
        "[dim]Config is saved to [bold]~/.lollms/coding_cli/config.yaml[/bold]\n"
        "and is NEVER committed to any repository.[/dim]",
        border_style="cyan",
    ))

    if not _HAS_QUESTIONARY:
        console.print(
            "[yellow]Tip:[/yellow] Install [bold]questionary[/bold] for a nicer wizard:\n"
            "       [dim]pip install questionary[/dim]\n"
        )
    
    if not _HAS_PROMPT_TOOLKIT:
        console.print(
            "[yellow]Tip:[/yellow] Install [bold]prompt-toolkit[/bold] for history & autocompletion:\n"
            "       [dim]pip install prompt-toolkit[/dim]\n"
        )
    else:
        console.print(
            "[dim]History: [bold]↑/↓[/bold] arrows  |  Autocomplete: [bold]Tab[/bold]  |  Exit: [bold]Ctrl+C/D[/bold][/dim]\n"
        )

    # ── 1. Choose binding ─────────────────────────────────────────────────────
    console.print("\n[bold]── Step 1: LLM Binding ──[/bold]")
    binding_choices = [
        f"{name}  —  {info['label']}"
        for name, info in BINDING_CATALOG.items()
    ]
    current_binding = cfg.get("binding_name", "lollms")
    # Find current index for default
    default_choice = next(
        (c for c in binding_choices if c.startswith(current_binding + " ")),
        binding_choices[0],
    )
    chosen_label = _ask_select("Select LLM binding:", binding_choices, default=default_choice)
    binding_name = chosen_label.split("  —  ")[0].strip()
    cfg["binding_name"] = binding_name
    catalog_entry = BINDING_CATALOG.get(binding_name, {})

    # ── 2. Binding-specific fields ────────────────────────────────────────────
    console.print(f"\n[bold]── Step 2: Configure '{binding_name}' ──[/bold]")
    bc = dict(cfg.get("binding_config") or {})

    for field in catalog_entry.get("fields", []):
        fname     = field["name"]
        flabel    = field["label"]
        is_secret = field.get("secret", False)
        # Show saved value if exists, otherwise catalog default
        saved_val = bc.get(fname, "")
        catalog_default = field.get("default", "")
        fdefault = str(saved_val) if saved_val else str(catalog_default)

        if is_secret:
            # For secrets: show hint if value exists, but don't pre-fill for security
            hint = " [dim](saved)[/dim]" if saved_val else f" [dim](default: {catalog_default or 'empty'})[/dim]"
            console.print(f"  {flabel}{hint}")
            val = _ask_password(f"    Enter value")
            if not val and saved_val:          # keep existing if blank
                val = saved_val
        else:
            val = _ask_text(f"  {flabel}", default=fdefault)

        if val:
            bc[fname] = val
        elif fname in bc and not val:
            # User cleared a field that had a value - remove it so default applies next time
            del bc[fname]
    cfg["binding_config"] = bc

    # ── 3. Model selection ────────────────────────────────────────────────────
    console.print(f"\n[bold]── Step 3: Model ──[/bold]")
    current_model = bc.get("model_name", "")

    if catalog_entry.get("needs_model_list"):
        console.print("  [dim]Connecting to server and fetching model list…[/dim]")
        live_model_dicts = _try_list_models(binding_name, bc)
        if live_model_dicts:
            console.print(
                f"  [green]✓[/green]  [bold]{len(live_model_dicts)}[/bold] model(s) available\n"
            )
            chosen_model = _pick_model(live_model_dicts, current_model)
        else:
            console.print(
                "  [yellow]⚠  Could not reach server — enter model name manually.[/yellow]"
            )
            fallback_names = BINDING_DEFAULT_MODELS.get(binding_name, [])
            if fallback_names:
                fallback_dicts = [{"model_name": n, "owned_by": "", 
                                   "context_length": None, "max_generation": None}
                                  for n in fallback_names]
                console.print("  [dim]Showing known defaults:[/dim]")
                chosen_model = _pick_model(fallback_dicts, current_model)
            else:
                chosen_model = _ask_text("  Model name:", default=current_model)
    else:
        fallback_names = BINDING_DEFAULT_MODELS.get(binding_name, [])
        if fallback_names:
            fallback_dicts = [{"model_name": n, "owned_by": "",
                               "context_length": None, "max_generation": None}
                              for n in fallback_names]
            chosen_model = _pick_model(fallback_dicts, current_model)
        else:
            chosen_model = _ask_text("  Model name:", default=current_model)

    if chosen_model:
        cfg["binding_config"]["model_name"] = chosen_model

    # ── 4. Paths ──────────────────────────────────────────────────────────────
    console.print("\n[bold]── Step 4: Paths ──[/bold]")
    cfg["projects_root"] = _ask_text(
        "  Projects root folder",
        default=cfg.get("projects_root", str(_documents_dir() / "lollms_projects")),
    )

    # ── 5. Runtime ────────────────────────────────────────────────────────────
    console.print("\n[bold]── Step 5: Runtime ──[/bold]")
    cfg["python_executable"] = _ask_text(
        "  Python executable",
        default=cfg.get("python_executable", sys.executable),
    )
    cfg["shell_timeout"] = int(
        _ask_text("  Shell command timeout (seconds)",
                  default=str(cfg.get("shell_timeout", 30)))
    )
    cfg["max_output_chars"] = int(
        _ask_text("  Max captured output chars per tool call",
                  default=str(cfg.get("max_output_chars", 8000)))
    )

    # ── 6. Git ────────────────────────────────────────────────────────────────
    console.print("\n[bold]── Step 6: Git ──[/bold]")
    cfg["git_auto_commit"] = _ask_confirm(
        "  Auto git-commit after /commit?",
        default=cfg.get("git_auto_commit", False),
    )

    save_config(cfg)
    console.print("\n[bold green]✅  Config saved.[/bold green]\n")
    return cfg


def show_config(cfg: dict) -> None:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="yellow")
    for k, v in cfg.items():
        if k == "binding_config":
            continue
        t.add_row(k, str(v))
    t.add_row("", "")
    t.add_row("[dim]binding_config[/dim]", "")
    for k, v in (cfg.get("binding_config") or {}).items():
        t.add_row(f"  [dim]{k}[/dim]", _mask(k, str(v)))
    console.print(Panel(t, title="Current config", border_style="dim"))


def build_client(cfg: dict) -> LollmsClient:
    return LollmsClient(
        llm_binding_name=cfg["binding_name"],
        llm_binding_config=cfg.get("binding_config", {}),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scratchpad  (~/.lollms/coding_cli/scratchpad.yaml)
# Global inter-agent / inter-task key-value store
# ═════════════════════════════════════════════════════════════════════════════

def load_scratchpad() -> dict:
    if SCRATCHPAD_FILE.exists():
        try:
            return yaml.safe_load(SCRATCHPAD_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


def save_scratchpad(data: dict) -> None:
    SCRATCHPAD_FILE.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def cmd_scratch(sub: str, arg: str, scratch: dict) -> dict:
    """Handle /scratch sub-commands. Returns possibly-modified scratch dict."""

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    if sub == "set":
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[red]Usage: /scratch set <key> <value>[/red]")
            return scratch
        key, value = parts[0], parts[1]
        scratch[key] = {"value": value, "updated_at": _ts()}
        save_scratchpad(scratch)
        console.print(f"  [green]✓[/green]  [bold]{key}[/bold] = {value!r}")

    elif sub == "get":
        key = arg.strip()
        if not key:
            console.print("[red]Usage: /scratch get <key>[/red]")
        elif key not in scratch:
            console.print(f"  [dim]Key '{key}' not found.[/dim]")
        else:
            entry = scratch[key]
            console.print(
                f"  [bold cyan]{key}[/bold cyan]  [dim]{entry.get('updated_at','')}[/dim]\n"
                f"  {entry['value']}"
            )

    elif sub == "ls":
        if not scratch:
            console.print("[dim]Scratchpad is empty.[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("Key",        style="bold white", no_wrap=True)
            t.add_column("Value",      style="yellow")
            t.add_column("Updated",    style="dim")
            for k, v in sorted(scratch.items()):
                val = str(v["value"])[:80] + ("…" if len(str(v["value"])) > 80 else "")
                t.add_row(k, val, v.get("updated_at", ""))
            console.print(t)

    elif sub == "del":
        key = arg.strip()
        if not key:
            console.print("[red]Usage: /scratch del <key>[/red]")
        elif key not in scratch:
            console.print(f"  [yellow]Key '{key}' not found.[/yellow]")
        else:
            del scratch[key]
            save_scratchpad(scratch)
            console.print(f"  [yellow]Deleted '{key}'.[/yellow]")

    elif sub == "clear":
        if _ask_confirm("  Clear the entire scratchpad?", default=False):
            scratch.clear()
            save_scratchpad(scratch)
            console.print("  [yellow]Scratchpad cleared.[/yellow]")

    else:
        console.print("[red]Unknown /scratch sub-command.[/red]  set | get | ls | del | clear")

    return scratch


# ═════════════════════════════════════════════════════════════════════════════
# Per-project long-term memory  (<output_folder>/.lollms/memory.yaml)
# ═════════════════════════════════════════════════════════════════════════════

def _memory_path(out_folder: Path) -> Path:
    mp = out_folder / ".lollms"
    mp.mkdir(exist_ok=True)
    return mp / "memory.yaml"


def load_memory(out_folder: Path) -> list:
    mp = _memory_path(out_folder)
    if mp.exists():
        try:
            return yaml.safe_load(mp.read_text(encoding="utf-8")) or []
        except Exception:
            pass
    return []


def save_memory(out_folder: Path, entries: list) -> None:
    _memory_path(out_folder).write_text(
        yaml.dump(entries, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _memory_summary(entries: list, max_chars: int = 3000) -> str:
    if not entries:
        return "(no project memory yet)"
    lines = []
    for e in entries:
        ts  = e.get("created_at", "")[:10]
        txt = e.get("text", "").strip()
        lines.append(f"[{ts}] {txt}")
    combined = "\n".join(lines)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n… [memory truncated]"
    return combined


def cmd_memory(sub: str, arg: str, out_folder: Path, mem: list) -> list:
    """Handle /memory sub-commands. Returns possibly-modified entries list."""

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    if sub == "add":
        text = arg.strip()
        if not text:
            console.print("[red]Usage: /memory add <text>[/red]")
        else:
            entry = {"id": len(mem) + 1, "text": text, "created_at": _ts()}
            mem.append(entry)
            save_memory(out_folder, mem)
            console.print(f"  [green]✓[/green]  Memory #{entry['id']} added.")

    elif sub == "ls":
        if not mem:
            console.print("[dim]No memory entries yet.[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("ID",      style="dim",        justify="right", width=4)
            t.add_column("Date",    style="dim",         width=12)
            t.add_column("Text",    style="white")
            for e in mem:
                txt = e.get("text", "")
                short = txt[:100] + ("…" if len(txt) > 100 else "")
                t.add_row(str(e.get("id", "?")), e.get("created_at", "")[:10], short)
            console.print(t)

    elif sub == "del":
        try:
            eid = int(arg.strip())
        except ValueError:
            console.print("[red]Usage: /memory del <id>[/red]")
            return mem
        before = len(mem)
        mem = [e for e in mem if e.get("id") != eid]
        if len(mem) < before:
            save_memory(out_folder, mem)
            console.print(f"  [yellow]Memory #{eid} deleted.[/yellow]")
        else:
            console.print(f"  [red]Entry #{eid} not found.[/red]")

    elif sub == "clear":
        if _ask_confirm("  Clear all project memory entries?", default=False):
            mem.clear()
            save_memory(out_folder, mem)
            console.print("  [yellow]Project memory cleared.[/yellow]")

    elif sub == "search":
        q = arg.strip().lower()
        if not q:
            console.print("[red]Usage: /memory search <query>[/red]")
        else:
            hits = [e for e in mem if q in e.get("text", "").lower()]
            if not hits:
                console.print(f"  [dim]No entries matching '{q}'.[/dim]")
            else:
                t = Table(show_header=True, header_style="bold cyan", expand=False)
                t.add_column("ID",   style="dim", justify="right", width=4)
                t.add_column("Date", style="dim", width=12)
                t.add_column("Text", style="white")
                for e in hits:
                    txt = e.get("text", "")
                    short = txt[:100] + ("…" if len(txt) > 100 else "")
                    t.add_row(str(e.get("id", "?")), e.get("created_at", "")[:10], short)
                console.print(t)
                console.print(f"  [dim]{len(hits)} result(s)[/dim]")

    else:
        console.print("[red]Unknown /memory sub-command.[/red]  add | ls | del | clear | search")

    return mem


# ═════════════════════════════════════════════════════════════════════════════
# Projects  (~/.lollms/coding_cli/projects.yaml)
# ═════════════════════════════════════════════════════════════════════════════

def load_projects() -> dict:
    if PROJECTS_FILE.exists():
        try:
            return yaml.safe_load(PROJECTS_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


def save_projects(projects: dict) -> None:
    PROJECTS_FILE.write_text(
        yaml.dump(projects, allow_unicode=True, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Subprocess helper
# ═════════════════════════════════════════════════════════════════════════════

def _run_proc(args, cwd: Path, timeout: int, max_chars: int,
              shell: bool = False) -> dict:
    try:
        r = subprocess.run(
            args, capture_output=True, text=True,
            cwd=str(cwd), timeout=timeout, errors="replace", shell=shell,
        )
        combined = (r.stdout + r.stderr).strip()
        if len(combined) > max_chars:
            combined = combined[:max_chars] + f"\n… [truncated at {max_chars} chars]"
        return {"success": r.returncode == 0, "returncode": r.returncode, "output": combined}
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "output": f"Timed out after {timeout}s"}
    except FileNotFoundError as e:
        return {"success": False, "returncode": -1, "output": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# Agent tool factory
# ═════════════════════════════════════════════════════════════════════════════

def build_agent_tools(project_root: Path, cfg: dict,
                      scratch: dict, mem: list) -> dict:
    """
    Build the tools dict for LollmsDiscussion.chat(tools=…).
    scratch / mem are passed by reference so tool writes are immediately visible.
    """
    py    = cfg.get("python_executable", sys.executable)
    tout  = cfg.get("shell_timeout", 30)
    maxch = cfg.get("max_output_chars", 8000)
    project_root.mkdir(parents=True, exist_ok=True)

    def _safe(p: str) -> Path:
        resolved = (project_root / p).resolve()
        try:
            resolved.relative_to(project_root.resolve())
        except ValueError:
            raise PermissionError(f"Path '{p}' escapes the project folder.")
        return resolved

    # ═════════════════════════════════════════════════════════════════════════
    # Process Manager — tracks background processes for LLM control
    # ═════════════════════════════════════════════════════════════════════════
    
    _process_registry: dict[int, dict] = {}  # pid -> process info
    
    def _register_process(pid: int, proc: Any, cmd: list, filename: str,
                          log_path: Path, start_time: float) -> None:
        _process_registry[pid] = {
            "proc": proc,
            "cmd": cmd,
            "filename": filename,
            "log_path": log_path,
            "start_time": start_time,
            "last_read_position": 0,
        }
    
    def _cleanup_process(pid: int) -> None:
        if pid in _process_registry:
            del _process_registry[pid]

    # ── execution ─────────────────────────────────────────────────────────────

    def run_code(filename: str, args: str = "", 
                 capture_seconds: float = 0.0,
                 wait_for_exit: bool = True,
                 timeout: Optional[int] = None) -> dict:
        """
        Execute code with flexible output handling.
        
        **CRITICAL**: For servers, web apps, or any long-running process:
        - ALWAYS use capture_seconds=5.0 (or higher) 
        - Set wait_for_exit=False to run in background
        
        - capture_seconds > 0: Run async, capture output for N seconds, then return
        - wait_for_exit=True:  Block until process completes (BAD for servers!)
        - wait_for_exit=False: Return immediately with PID, let LLM poll/stop later
        
        For UI apps: combine with list_windows() and screenshot_window() tools.
        """
        import subprocess as sp
        import threading
        import time as time_mod
        
        fp = _safe(filename)
        if not fp.exists():
            return {"success": False, "output": f"File not found: {filename}"}
        
        cmd = [py, str(fp)] + (args.split() if args.strip() else [])
        effective_timeout = timeout or tout
        
        # Determine execution mode
        is_async = capture_seconds > 0 or not wait_for_exit
        
        if not is_async:
            # ═══ Synchronous mode (original behavior) ═══
            r = _run_proc(cmd, project_root, effective_timeout, maxch)
            console.print(Panel(r["output"] or "(no output)",
                                title=f"[cyan]run {filename}[/cyan]",
                                border_style="green" if r["success"] else "red"))
            return r
        
        # ═══ Async mode: start process, capture/manage output ═══
        
        # Setup log file for persistent output
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = project_root / f"{Path(filename).stem}_{timestamp}.log"
        
        try:
            # Start process with pipes for live capture
            if platform.system() == "Windows":
                proc = sp.Popen(
                    cmd, cwd=str(project_root),
                    stdout=sp.PIPE, stderr=sp.STDOUT,  # Merge stderr into stdout
                    creationflags=sp.CREATE_NEW_PROCESS_GROUP,
                    text=True, errors="replace",
                    bufsize=1  # Line buffered
                )
            else:
                proc = sp.Popen(
                    cmd, cwd=str(project_root),
                    stdout=sp.PIPE, stderr=sp.STDOUT,
                    start_new_session=True,
                    text=True, errors="replace",
                    bufsize=1
                )
            
            pid = proc.pid
            start_time = time_mod.time()
            
            # Register for management
            _register_process(pid, proc, cmd, filename, log_file, start_time)
            
            # Write header to log
            log_file.write_text(
                f"# Started: {datetime.now(timezone.utc).isoformat()}\n"
                f"# PID: {pid}\n"
                f"# Command: {' '.join(cmd)}\n"
                f"# {'='*50}\n\n",
                encoding="utf-8"
            )
            
            captured_output: list[str] = []
            
            def _read_stream():
                """Background thread to continuously read and log output."""
                with open(log_file, "a", encoding="utf-8") as log_f:
                    try:
                        for line in proc.stdout:
                            line_str = line.rstrip('\n')
                            captured_output.append(line_str)
                            log_f.write(line_str + '\n')
                            log_f.flush()
                    except Exception:
                        pass
            
            reader_thread = threading.Thread(target=_read_stream, daemon=True)
            reader_thread.start()
            
            # Mode 1: Capture for fixed duration, then return
            if capture_seconds > 0:
                console.print(Panel(
                    f"Capturing output for {capture_seconds}s... (PID: {pid})",
                    title=f"[cyan]{filename}[/cyan]",
                    border_style="yellow"
                ))
                
                # Wait for capture period or process exit
                try:
                    proc.wait(timeout=capture_seconds)
                    exit_code = proc.returncode
                    still_running = False
                except sp.TimeoutExpired:
                    exit_code = None
                    still_running = True
                
                # Collect captured output
                time_mod.sleep(0.1)  # Let reader flush
                output_text = '\n'.join(captured_output[-500:])  # Last 500 lines max
                if len(output_text) > maxch:
                    output_text = "...[truncated]...\n" + output_text[-maxch:]
                
                status = "running" if still_running else f"exited ({exit_code})"
                
                result = {
                    "success": exit_code in (None, 0),
                    "pid": pid,
                    "status": status,
                    "captured_seconds": capture_seconds,
                    "output_preview": output_text,
                    "log_file": str(log_file),
                    "stop_command": f"stop_process({pid})",
                    "read_more": f"read_process_output({pid}, lines=100)",
                }
                
                console.print(Panel(
                    f"Status: {status}\n"
                    f"Captured {len(captured_output)} lines\n"
                    f"Log: {log_file.name}",
                    title=f"[green]✓ {filename}[/green]" if exit_code in (None, 0) else f"[red]✗ {filename}[/red]",
                    border_style="green" if exit_code in (None, 0) else "red"
                ))
                
                # Keep process running in registry if still alive
                if not still_running:
                    _cleanup_process(pid)
                
                return result
            
            # Mode 2: Fire-and-forget (wait_for_exit=False)
            else:
                # Quick health check — did it start or crash immediately?
                time_mod.sleep(0.5)
                if proc.poll() is not None:
                    # Process exited immediately — probably an error
                    exit_code = proc.returncode
                    output_text = '\n'.join(captured_output)
                    _cleanup_process(pid)
                    return {
                        "success": False,
                        "pid": pid,
                        "status": f"crashed immediately (exit {exit_code})",
                        "output": output_text[:maxch],
                        "log_file": str(log_file),
                    }
                
                console.print(Panel(
                    f"PID: {pid}\n"
                    f"Log: {log_file.name}\n"
                    f"Use read_process_output({pid}) to check status",
                    title=f"[green]✓ {filename} started[/green]",
                    border_style="green"
                ))
                
                return {
                    "success": True,
                    "pid": pid,
                    "status": "running",
                    "log_file": str(log_file),
                    "read_output": f"read_process_output({pid}, lines=50)",
                    "stop_command": f"stop_process({pid})",
                }
                
        except Exception as e:
            trace_exception(e)
            return {"success": False, "output": f"Failed to start: {e}"}

    def read_process_output(pid: int, lines: int = 50, 
                            since_start: bool = False) -> dict:
        """
        Read recent output from a running background process.
        
        - lines: number of recent lines to return
        - since_start: if True, return all output from beginning (may be large!)
        """
        if pid not in _process_registry:
            # Try to read from log file even if process not in registry
            # (e.g., if CLI was restarted)
            log_files = list(project_root.glob(f"*{pid}*.log"))
            if log_files:
                log_file = max(log_files, key=lambda p: p.stat().st_mtime)
                content = log_file.read_text(encoding="utf-8", errors="replace")
                all_lines = content.split('\n')
                recent = '\n'.join(all_lines[-lines:] if not since_start else all_lines)
                return {
                    "success": True,
                    "pid": pid,
                    "status": "unknown (not in active registry)",
                    "output": recent[-maxch:],
                    "total_lines": len(all_lines),
                    "log_file": str(log_file),
                }
            return {"success": False, "output": f"Process {pid} not found"}
        
        info = _process_registry[pid]
        proc = info["proc"]
        log_file = info["log_path"]
        
        # Check if still running
        exit_code = proc.poll()
        is_running = exit_code is None
        
        # Read from log file (more reliable than trying to read pipe)
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            all_lines = content.split('\n')
            
            if since_start:
                start_idx = 0
            else:
                # Start from last read position or last N lines
                start_idx = max(0, len(all_lines) - lines)
            
            selected = all_lines[start_idx:]
            output_text = '\n'.join(selected)
            
            # Update last read position for incremental reads
            info["last_read_position"] = len(all_lines)
        else:
            output_text = "(log file not yet created)"
        
        if len(output_text) > maxch:
            output_text = "...[truncated]...\n" + output_text[-maxch:]
        
        runtime = time.time() - info["start_time"]
        
        result = {
            "success": True,
            "pid": pid,
            "status": "running" if is_running else f"exited ({exit_code})",
            "runtime_seconds": round(runtime, 1),
            "output": output_text,
            "total_lines": len(all_lines) if log_file.exists() else 0,
            "log_file": str(log_file),
        }
        
        if not is_running:
            _cleanup_process(pid)
            result["stop_command"] = None
        else:
            result["actions"] = {
                "read_more": f"read_process_output({pid}, lines={lines})",
                "stop": f"stop_process({pid})",
            }
        
        console.print(Panel(
            f"Status: {result['status']} | Runtime: {runtime:.1f}s\n"
            f"Lines: {result['total_lines']}",
            title=f"[cyan]Process {pid}[/cyan]",
            border_style="green" if is_running else "dim"
        ))
        
        return result

    def stop_process(pid: int, graceful: bool = True,
                     grace_period: float = 5.0) -> dict:
        """
        Stop a background process.
        
        - graceful: send SIGTERM first, then SIGKILL
        - grace_period: seconds to wait between SIGTERM and SIGKILL
        """
        import signal as sig_module
        
        if pid not in _process_registry:
            # Try to kill anyway
            try:
                os.kill(pid, sig_module.SIGTERM if graceful else sig_module.SIGKILL)
                return {"success": True, "output": f"Sent signal to PID {pid} (was not in registry)"}
            except ProcessLookupError:
                return {"success": False, "output": f"Process {pid} not found"}
            except Exception as e:
                return {"success": False, "output": str(e)}
        
        info = _process_registry[pid]
        proc = info["proc"]
        
        console.print(Panel(f"Stopping PID {pid}...", 
                           title="[yellow]stop[/yellow]", border_style="yellow"))
        
        try:
            if platform.system() == "Windows":
                # Windows: CTRL_BREAK_EVENT to process group, then terminate
                try:
                    proc.send_signal(sig_module.CTRL_BREAK_EVENT)
                    proc.wait(timeout=grace_period)
                except:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except:
                        proc.kill()
            else:
                # Unix: SIGTERM, wait, then SIGKILL
                proc.terminate()
                try:
                    proc.wait(timeout=grace_period)
                except sp.TimeoutExpired:
                    console.print(f"[yellow]SIGTERM timeout, sending SIGKILL...[/yellow]")
                    proc.kill()
                    proc.wait(timeout=5)
            
            # Final output capture
            final_read = read_process_output(pid, lines=100)
            _cleanup_process(pid)
            
            console.print(Panel(
                f"Process {pid} stopped\n"
                f"Exit code: {proc.returncode}",
                title="[green]✓ Stopped[/green]",
                border_style="green"
            ))
            
            return {
                "success": True,
                "pid": pid,
                "exit_code": proc.returncode,
                "final_output": final_read.get("output", "")[:maxch],
            }
            
        except Exception as e:
            _cleanup_process(pid)
            return {"success": False, "output": f"Error stopping process: {e}"}

    def list_running_processes() -> dict:
        """List all processes started by this session."""
        if not _process_registry:
            return {"success": True, "processes": [], "count": 0}
        
        processes = []
        now = time.time()
        
        for pid, info in list(_process_registry.items()):
            proc = info["proc"]
            exit_code = proc.poll()
            is_running = exit_code is None
            
            processes.append({
                "pid": pid,
                "filename": info["filename"],
                "runtime_seconds": round(now - info["start_time"], 1),
                "status": "running" if is_running else f"exited ({exit_code})",
                "log_file": str(info["log_path"]),
            })
            
            if not is_running:
                _cleanup_process(pid)
        
        return {
            "success": True,
            "processes": processes,
            "count": len(processes),
        }

    # ── UI app support ───────────────────────────────────────────────────────

    def list_windows(pattern: str = "") -> dict:
        """
        List visible windows. Useful for finding UI apps the LLM started.
        
        - pattern: optional filter by window title
        """
        system = platform.system()
        
        try:
            if system == "Windows":
                # Use PowerShell to enumerate windows
                ps_cmd = '''
                Add-Type @"
                using System;
                using System.Runtime.InteropServices;
                public class WinAPI {
                    [DllImport("user32.dll")] public static extern bool EnumWindows(IntPtr callback, IntPtr lParam);
                    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
                    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
                }
"@
                $windows = @()
                $callback = {
                    param($hwnd, $lParam)
                    if ([WinAPI]::IsWindowVisible($hwnd)) {
                        $title = New-Object System.Text.StringBuilder 256
                        [WinAPI]::GetWindowText($hwnd, $title, 256) | Out-Null
                        $t = $title.ToString()
                        if ($t -and ($t -match "'"$pattern"'")) {
                            $windows += $t
                        }
                    }
                    return $true
                }
                $delegate = [System.Delegate]::CreateDelegate([Func[IntPtr,IntPtr,bool]], $callback)
                [WinAPI]::EnumWindows($delegate, 0) | Out-Null
                $windows
                '''
                r = subprocess.run(
                    ["powershell", "-Command", ps_cmd.replace("'", '"')],
                    capture_output=True, text=True, timeout=10
                )
                titles = [l.strip() for l in r.stdout.split('\n') if l.strip()]
                
            elif system == "Darwin":
                # macOS: use osascript
                r = subprocess.run(
                    ["osascript", "-e", 'tell application "System Events" to get name of every window of every process'],
                    capture_output=True, text=True, timeout=10
                )
                titles = [t.strip() for t in r.stdout.split(',') if t.strip()]
                if pattern:
                    titles = [t for t in titles if pattern.lower() in t.lower()]
                    
            else:
                # Linux: try xdotool or wmctrl
                try:
                    r = subprocess.run(
                        ["xdotool", "search", "--onlyvisible", "--name", pattern or ".*", "getwindowname"],
                        capture_output=True, text=True, timeout=5
                    )
                    titles = [l.strip() for l in r.stdout.split('\n') if l.strip()]
                except FileNotFoundError:
                    try:
                        r = subprocess.run(
                            ["wmctrl", "-l"],
                            capture_output=True, text=True, timeout=5
                        )
                        titles = [' '.join(l.split()[3:]) for l in r.stdout.split('\n') if l.strip()]
                        if pattern:
                            titles = [t for t in titles if pattern.lower() in t.lower()]
                    except FileNotFoundError:
                        titles = []
            
            return {
                "success": True,
                "platform": system,
                "windows": titles[:50],  # Limit results
                "count": len(titles),
            }
            
        except Exception as e:
            return {"success": False, "output": f"Could not list windows: {e}"}

    def screenshot_window(title_pattern: str = "", 
                        output_path: str = "screenshot.png") -> dict:
        """
        Capture a screenshot of a window or the full screen.
        
        - title_pattern: specific window (empty = full screen)
        - output_path: where to save (relative to project folder)
        """
        system = platform.system()
        dest = _safe(output_path)
        
        try:
            if system == "Windows":
                # Use PowerShell + .NET for screenshot
                ps_cmd = f'''
                Add-Type -AssemblyName System.Windows.Forms
                Add-Type -AssemblyName System.Drawing
                $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
                $bitmap.Save("{dest}")
                '''
                subprocess.run(["powershell", "-Command", ps_cmd], check=True, timeout=10)
                
            elif system == "Darwin":
                # macOS: use screencapture
                if title_pattern:
                    # Try to find and capture specific window
                    subprocess.run(
                        ["screencapture", "-w", str(dest)],
                        check=True, timeout=10
                    )
                else:
                    subprocess.run(
                        ["screencapture", str(dest)],
                        check=True, timeout=10
                    )
                    
            else:
                # Linux: try gnome-screenshot, import (ImageMagick), or ffmpeg
                tools = [
                    ["gnome-screenshot", "-f", str(dest)],
                    ["import", "-window", "root", str(dest)],
                    ["ffmpeg", "-f", "x11grab", "-i", ":0", "-frames:v", "1", str(dest)],
                ]
                for tool in tools:
                    try:
                        subprocess.run(tool, check=True, timeout=10)
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
                else:
                    raise RuntimeError("No screenshot tool found (tried: gnome-screenshot, ImageMagick, ffmpeg)")
            
            # Read and encode for LLM vision
            import base64
            img_bytes = dest.read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            
            console.print(Panel(
                f"Saved: {dest}\n"
                f"Size: {len(img_bytes):,} bytes",
                title="[green]✓ Screenshot[/green]",
                border_style="green"
            ))
            
            return {
                "success": True,
                "path": str(dest),
                "size_bytes": len(img_bytes),
                "base64_image": b64[:100] + "...[truncated]" if len(b64) > 100 else b64,
                "full_base64_available": True,  # LLM can request full if needed
            }
            
        except Exception as e:
            return {"success": False, "output": f"Screenshot failed: {e}"}

    def run_shell(command: str) -> dict:
        on_win = platform.system() == "Windows"
        r = _run_proc(command if on_win else ["bash", "-c", command],
                      project_root, tout, maxch, shell=on_win)
        console.print(Panel(r["output"] or "(no output)",
                            title=f"[cyan]$ {command[:70]}[/cyan]",
                            border_style="green" if r["success"] else "red"))
        return r

    def run_tests(path: str = ".") -> dict:
        target = _safe(path)
        if not target.exists():
            return {"success": False, "output": f"Path not found: {path}"}
        r = _run_proc([py, "-m", "pytest", str(target), "-v", "--tb=short", "--no-header"],
                      project_root, tout * 2, maxch)
        console.print(Panel(r["output"] or "(no output)", title="[cyan]pytest[/cyan]",
                            border_style="green" if r["success"] else "red"))
        return r

    # ── file system ───────────────────────────────────────────────────────────

    def read_file(path: str) -> dict:
        try:
            content = _safe(path).read_text(encoding="utf-8", errors="replace")
            if len(content) > maxch:
                content = content[:maxch] + "\n… [truncated]"
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "output": str(e)}

    def write_file(path: str, content: str) -> dict:
        try:
            fp = _safe(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            console.print(f"  [green]✓[/green]  wrote [bold]{path}[/bold]  ({len(content)} chars)")
            return {"success": True, "path": path, "bytes": len(content.encode())}
        except Exception as e:
            return {"success": False, "output": str(e)}

    def list_files(path: str = ".") -> dict:
        try:
            target  = _safe(path)
            entries = sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name))
            lines   = [
                f"{'📄' if e.is_file() else '📁'}  {e.name}"
                + (f"  {e.stat().st_size:>9} B" if e.is_file() else "")
                for e in entries
            ]
            return {"success": True, "listing": "\n".join(lines),
                    "files": [e.name for e in entries]}
        except Exception as e:
            return {"success": False, "output": str(e)}

    # ── scratchpad tools ──────────────────────────────────────────────────────

    def scratchpad_read(key: str) -> dict:
        if key not in scratch:
            return {"success": False, "output": f"Key '{key}' not found in scratchpad."}
        return {"success": True, "key": key, "value": scratch[key]["value"]}

    def scratchpad_write(key: str, value: str) -> dict:
        scratch[key] = {"value": value,
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_scratchpad(scratch)
        console.print(f"  [green]✓[/green]  scratchpad[{key!r}] = {value!r}")
        return {"success": True, "key": key}

    # ── memory tool ───────────────────────────────────────────────────────────

    def memory_add(text: str) -> dict:
        entry = {
            "id": len(mem) + 1,
            "text": text.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        mem.append(entry)
        save_memory(project_root, mem)
        console.print(f"  [green]✓[/green]  Memory #{entry['id']} added.")
        return {"success": True, "id": entry["id"]}

    # ── git ───────────────────────────────────────────────────────────────────

    def _g(*args) -> dict:
        return _run_proc(["git"] + list(args), project_root, tout, maxch)

    def git_status() -> dict:
        r = _g("status")
        console.print(Panel(r["output"], title="[cyan]git status[/cyan]", border_style="dim"))
        return r

    def git_diff(file: str = "") -> dict:
        r = _g(*filter(None, ["diff", file]))
        if r["output"]:
            console.print(Syntax(r["output"], "diff", theme="monokai"))
        return r

    def git_add(files: str = ".") -> dict:
        targets = files.split() if files.strip() != "." else ["."]
        r = _g("add", *targets)
        if r["success"]:
            console.print(f"[green]git add {files} ✓[/green]")
        return r

    def git_commit(message: str) -> dict:
        r = _g("commit", "-m", message)
        if r["success"]:
            console.print(f"[green]git commit ✓[/green]  {message!r}")
        else:
            console.print(f"[yellow]git commit:[/yellow] {r['output']}")
        return r

    def git_log(n: int = 10) -> dict:
        r = _g("log", "--oneline", f"-{n}")
        if r["output"]:
            console.print(Panel(r["output"], title=f"[cyan]git log -{n}[/cyan]", border_style="dim"))
        return r

    def git_branch(name: str = "") -> dict:
        r = _g(*filter(None, ["branch", name]))
        console.print(Panel(r["output"] or "(no branches)", title="[cyan]git branch[/cyan]", border_style="dim"))
        return r

    def git_checkout(branch: str) -> dict:
        r = _g("checkout", branch)
        if r["success"]:
            console.print(f"[green]git checkout {branch} ✓[/green]")
        return r

    def _llm_query(query: str) -> dict:
        return {"answer": query, "success": True}

    def _python_exec(code: str) -> dict:
        return _run_proc([py, "-c", code], project_root, tout, maxch)

    # ── assemble ──────────────────────────────────────────────────────────────

    def _t(name, desc, params, out, fn):
        return {name: {"name": name, "description": desc,
                       "parameters": params, "output": out, "callable": fn}}

    tools: dict = {}
    # ── process management ────────────────────────────────────────────────────
    
    tools.update(_t("run_code",
        "Execute code. For servers/long-running processes: MUST use capture_seconds=5+ and wait_for_exit=False",
        [{"name":"filename","type":"str","optional":False},
         {"name":"args",    "type":"str","optional":True,"default":""},
         {"name":"capture_seconds","type":"float","optional":True,"default":0.0,
          "description":"REQUIRED for servers: seconds to capture output before returning (e.g., 5.0)"},
         {"name":"wait_for_exit","type":"bool","optional":True,"default":True,
          "description":"MUST be False for servers! True=block until exit, False=background with PID"},
         {"name":"timeout","type":"int","optional":True,"default":30,
          "description":"Timeout for synchronous mode only"}],
        [{"name":"success","type":"bool"},{"name":"pid","type":"int","optional":True},
         {"name":"status","type":"str"},{"name":"output_preview","type":"str","optional":True},
         {"name":"log_file","type":"str"},{"name":"read_more","type":"str","optional":True},
         {"name":"stop_command","type":"str","optional":True}], run_code))
    
    tools.update(_t("read_process_output",
        "Read recent output from a running background process",
        [{"name":"pid","type":"int","optional":False},
         {"name":"lines","type":"int","optional":True,"default":50,
          "description":"Number of recent lines to return"},
         {"name":"since_start","type":"bool","optional":True,"default":False,
          "description":"Return all output from beginning (may be large!)"}],
        [{"name":"success","type":"bool"},{"name":"pid","type":"int"},
         {"name":"status","type":"str"},{"name":"output","type":"str"},
         {"name":"runtime_seconds","type":"float"},{"name":"actions","type":"dict","optional":True}],
        read_process_output))
    
    tools.update(_t("stop_process",
        "Stop a background process started with run_code",
        [{"name":"pid","type":"int","optional":False},
         {"name":"graceful","type":"bool","optional":True,"default":True,
          "description":"Try SIGTERM first before SIGKILL"},
         {"name":"grace_period","type":"float","optional":True,"default":5.0,
          "description":"Seconds to wait for graceful shutdown"}],
        [{"name":"success","type":"bool"},{"name":"pid","type":"int"},
         {"name":"exit_code","type":"int","optional":True},
         {"name":"final_output","type":"str","optional":True}], stop_process))
    
    tools.update(_t("list_running_processes",
        "List all processes started by this session",
        [],[{"name":"success","type":"bool"},{"name":"processes","type":"list"},
         {"name":"count","type":"int"}], list_running_processes))
    
    # ── UI app support ─────────────────────────────────────────────────────
    
    tools.update(_t("list_windows",
        "List visible windows (useful for finding UI apps)",
        [{"name":"pattern","type":"str","optional":True,"default":"",
          "description":"Optional filter by window title"}],
        [{"name":"success","type":"bool"},{"name":"platform","type":"str"},
         {"name":"windows","type":"list"},{"name":"count","type":"int"}], list_windows))
    
    tools.update(_t("screenshot_window",
        "Capture screenshot of window or full screen for LLM to 'see' UI",
        [{"name":"title_pattern","type":"str","optional":True,"default":"",
          "description":"Specific window title (empty = full screen)"},
         {"name":"output_path","type":"str","optional":True,"default":"screenshot.png"}],
        [{"name":"success","type":"bool"},{"name":"path","type":"str"},
         {"name":"size_bytes","type":"int"},
         {"name":"base64_image","type":"str","optional":True}], screenshot_window))
    tools.update(_t("run_shell",  "Shell command in project folder",
        [{"name":"command","type":"str","optional":False}],
        [{"name":"output","type":"str"}], run_shell))
    tools.update(_t("run_tests",  "Run pytest",
        [{"name":"path","type":"str","optional":True,"default":"."}],
        [{"name":"output","type":"str"},{"name":"returncode","type":"int"}], run_tests))
    tools.update(_t("read_file",  "Read a project file",
        [{"name":"path","type":"str","optional":False}],
        [{"name":"content","type":"str"}], read_file))
    tools.update(_t("write_file", "Write/overwrite a project file",
        [{"name":"path",   "type":"str","optional":False},
         {"name":"content","type":"str","optional":False}],
        [{"name":"bytes","type":"int"}], write_file))
    tools.update(_t("list_files", "List project directory",
        [{"name":"path","type":"str","optional":True,"default":"."}],
        [{"name":"listing","type":"str"},{"name":"files","type":"list"}], list_files))
    tools.update(_t("scratchpad_read",  "Read a global scratchpad entry",
        [{"name":"key","type":"str","optional":False}],
        [{"name":"value","type":"str"}], scratchpad_read))
    tools.update(_t("scratchpad_write", "Write a global scratchpad entry",
        [{"name":"key",  "type":"str","optional":False},
         {"name":"value","type":"str","optional":False}],
        [{"name":"success","type":"bool"}], scratchpad_write))
    tools.update(_t("memory_add", "Append an entry to project long-term memory",
        [{"name":"text","type":"str","optional":False}],
        [{"name":"id","type":"int"}], memory_add))
    tools.update(_t("git_status",   "git status",  [], [{"name":"output","type":"str"}], git_status))
    tools.update(_t("git_diff",     "Show uncommitted diff",
        [{"name":"file","type":"str","optional":True,"default":""}],
        [{"name":"output","type":"str"}], git_diff))
    tools.update(_t("git_add",      "Stage files",
        [{"name":"files","type":"str","optional":True,"default":"."}],
        [{"name":"output","type":"str"}], git_add))
    tools.update(_t("git_commit",   "Commit staged changes",
        [{"name":"message","type":"str","optional":False}],
        [{"name":"output","type":"str"}], git_commit))
    tools.update(_t("git_log",      "Recent commit history",
        [{"name":"n","type":"int","optional":True,"default":10}],
        [{"name":"output","type":"str"}], git_log))
    tools.update(_t("git_branch",   "List or create a branch",
        [{"name":"name","type":"str","optional":True,"default":""}],
        [{"name":"output","type":"str"}], git_branch))
    tools.update(_t("git_checkout", "Switch branch",
        [{"name":"branch","type":"str","optional":False}],
        [{"name":"output","type":"str"}], git_checkout))
    tools.update(_t("llm_query",    "Internal sub-LLM query",
        [{"name":"query","type":"str","optional":False}],
        [{"name":"answer","type":"str"}], _llm_query))
    tools.update(_t("python_exec",  "Execute a Python snippet",
        [{"name":"code","type":"str","optional":False}],
        [{"name":"output","type":"str"}], _python_exec))
    return tools


# ═════════════════════════════════════════════════════════════════════════════
# ProjectSession
# ═════════════════════════════════════════════════════════════════════════════

class ProjectSession:
    """Everything belonging to the currently active project."""

    def __init__(self, name: str, projects: dict, lc: LollmsClient,
                 cfg: dict, scratch: dict):
        self.name        = name
        self.projects    = projects
        self._lc         = lc
        self._cfg        = cfg
        self._scratch    = scratch
        meta             = projects[name]
        self.db          = LollmsDataManager(meta["db_path"])
        self.out_folder  = Path(meta["output_path"])
        self.out_folder.mkdir(parents=True, exist_ok=True)
        self._mem: list  = load_memory(self.out_folder)
        self._disc: Optional[LollmsDiscussion] = None
        self._tools: dict = {}
        self._rebuild_tools()
        self._restore_or_new()

    # ── tools ─────────────────────────────────────────────────────────────────

    def _rebuild_tools(self) -> None:
        self._tools = build_agent_tools(
            self.out_folder, self._cfg, self._scratch, self._mem
        )

    @property
    def tools(self) -> dict:
        return self._tools

    @property
    def mem(self) -> list:
        return self._mem

    def reload_memory(self) -> None:
        self._mem = load_memory(self.out_folder)
        self._rebuild_tools()

    # ── discussion lifecycle ──────────────────────────────────────────────────

    def _make_system_prompt(self) -> str:
        mem_text = _memory_summary(self._mem)
        scratch_lines = "\n".join(
            f"  {k} = {v['value']}" for k, v in sorted(self._scratch.items())
        ) or "  (empty)"
        env_text = self._detect_environment()
        soul = load_soul()
        skills = load_skills()
        skills_text = get_enabled_skills_prompt(skills)
        return SYSTEM_PROMPT_TEMPLATE.format(
            base_personality=soul,
            env_section=env_text,
            skills_section=skills_text,
            memory_section=mem_text,
            scratchpad_section=scratch_lines,
        )

    def _restore_or_new(self) -> None:
        saved_id = self.projects[self.name].get("active_discussion_id")
        if saved_id and self.db.discussion_exists(saved_id):
            try:
                disc = self.db.get_discussion(self._lc, saved_id)
                if disc:
                    self._disc = disc
                    # Always refresh system prompt to pick up memory/scratchpad changes
                    self.refresh_system_prompt()
                    console.print(f"[dim]Resumed discussion {saved_id[:8]}…[/dim]")
                    return
            except Exception:
                pass
        self._new_discussion()

    def _new_discussion(self) -> None:
        if self._disc:
            self._disc.close()
        self._disc = LollmsDiscussion.create_new(
            lollms_client=self._lc,
            db_manager=self.db,
            autosave=True,
            system_prompt=self._make_system_prompt(),
        )
        self.projects[self.name]["active_discussion_id"] = self._disc.id
        save_projects(self.projects)
        console.print(f"[dim]New discussion {self._disc.id[:8]}…[/dim]")

    def _detect_environment(self) -> str:
        """Detect and format current runtime environment."""
        import sys
        import shutil
        import subprocess
        
        # Basic OS info
        system = platform.system()
        release = platform.release()
        machine = platform.machine()
        
        # User and paths
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
        cwd = os.getcwd()
        home = Path.home()
        
        # Shell detection
        shell = "unknown"
        if system == "Windows":
            shell = os.environ.get("COMSPEC", "cmd.exe").split("\\")[-1].replace(".exe", "")
            # PowerShell detection via parent process or PSModulePath
            if "PSModulePath" in os.environ or "POWERSHELL_DISTRIBUTION_CHANNEL" in os.environ:
                shell = "powershell"
        else:
            shell = os.environ.get("SHELL", "/bin/sh").split("/")[-1]
        
        # Python environment
        py_version = sys.version.replace("\n", " ")
        py_executable = sys.executable
        venv = getattr(sys, "real_prefix", None) or getattr(sys, "base_prefix", None)
        if venv and venv != sys.prefix:
            venv_info = f"venv: {sys.prefix}"
        else:
            venv_info = "system Python (no venv)"
        
        # Check common tools
        tools = []
        for cmd in ["git", "gcc", "g++", "clang", "cl", "cmake", "make", "ninja", "node", "npm", "cargo", "rustc", "go", "javac", "docker", "kubectl", "terraform", "ansible"]:
            path = shutil.which(cmd)
            if path:
                try:
                    ver = subprocess.run([cmd, "--version"], capture_output=True, text=True, errors="ignore", timeout=2)
                    ver_line = ver.stdout.split("\n")[0][:50] if ver.stdout else "version unknown"
                    tools.append(f"  - {cmd}: {ver_line}")
                except:
                    tools.append(f"  - {cmd}: available")
        
        # Check key Python packages
        py_packages = []
        for pkg in ["fastapi", "uvicorn", "flask", "django", "requests", "numpy", "pandas", "pytest", "sqlalchemy", "pydantic", "httpx", "aiohttp", "asyncio", "tqdm", "rich", "typer", "click"]:
            try:
                __import__(pkg)
                py_packages.append(pkg)
            except ImportError:
                pass
        
        # Git repository status
        git_info = []
        git_path = shutil.which("git")
        if git_path:
            try:
                # Check if inside a git repo
                r = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    capture_output=True, text=True, errors="ignore", timeout=2,
                    cwd=os.getcwd()
                )
                if r.returncode == 0:
                    # Get repo info
                    branch_r = subprocess.run(
                        ["git", "branch", "--show-current"],
                        capture_output=True, text=True, errors="ignore", timeout=2
                    )
                    branch = branch_r.stdout.strip() or "detached HEAD"
                    
                    # Check for uncommitted changes
                    status_r = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True, text=True, errors="ignore", timeout=2
                    )
                    dirty = "dirty" if status_r.stdout.strip() else "clean"
                    
                    # Get last commit
                    log_r = subprocess.run(
                        ["git", "log", "-1", "--oneline"],
                        capture_output=True, text=True, errors="ignore", timeout=2
                    )
                    last_commit = log_r.stdout.strip()[:50] if log_r.stdout else "unknown"
                    
                    git_info = [
                        f"  - Git available: {git_path}",
                        f"  - Repository: yes (branch: {branch}, {dirty})",
                        f"  - Last commit: {last_commit}",
                    ]
                else:
                    git_info = [
                        f"  - Git available: {git_path}",
                        f"  - Repository: no (not inside a git repo)",
                    ]
            except Exception:
                git_info = [f"  - Git available: {git_path}", "  - Repository: detection failed"]
        else:
            git_info = ["  - Git: not installed or not in PATH"]
        
        lines = [
            f"- OS: {system} {release} ({machine})",
            f"- User: {user}",
            f"- Shell: {shell}",
            f"- Working directory: {cwd}",
            f"- Home directory: {home}",
            f"- Date/Time: {datetime.now(timezone.utc).isoformat()} UTC",
            f"",
            f"- Python: {py_version}",
            f"  Executable: {py_executable}",
            f"  Environment: {venv_info}",
            f"",
            f"- Installed Python packages detected: {', '.join(py_packages) if py_packages else '(none checked)'}",
            f"",
            f"- Available system tools:",
        ]
        lines.extend(tools if tools else ["  (none detected)"])
        lines.append("")
        lines.extend(["- Version control:"] + git_info)
        lines.append("")
        lines.append("⚠️ CRITICAL: Always prefer Windows-compatible commands (e.g., `Get-Content` instead of `cat`, `Select-Object -Last` instead of `tail`). Use Python for cross-platform scripts when possible.")
        
        return "\n".join(lines)

    def refresh_system_prompt(self) -> None:
        """Re-inject memory + scratchpad + environment into the current discussion's system prompt."""
        if self._disc:
            self._disc.system_prompt = self._make_system_prompt()
            self._disc.commit()

    @property
    def disc(self) -> LollmsDiscussion:
        return self._disc

    def new_discussion(self) -> None:
        self._new_discussion()

    def update_config(self, cfg: dict) -> None:
        self._cfg = cfg
        self._rebuild_tools()

    def close(self) -> None:
        if self._disc:
            self._disc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Artefact / file helpers
# ═════════════════════════════════════════════════════════════════════════════

def _artefact_filename(a: dict) -> str:
    title = a.get("title", "artefact")
    if Path(title).suffix:
        return title
    ext = EXT_MAP.get((a.get("language") or "").lower(), ".txt")
    return title + ext


def commit_artefacts(disc: LollmsDiscussion, folder: Path, cfg: dict) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    tout  = cfg.get("shell_timeout", 30)
    maxch = cfg.get("max_output_chars", 8000)
    written = []
    for a in disc.artefacts.list():
        if a.get("type") not in CODE_TYPES:
            continue
        content = (a.get("content") or "").strip()
        if not content:
            continue
        dest = folder / _artefact_filename(a)
        dest.write_text(content, encoding="utf-8")
        written.append(str(dest))
    if written and cfg.get("git_auto_commit"):
        _run_proc(["git", "add", "."], folder, tout, maxch)
        msg = f"auto-commit: {len(written)} file(s)"
        r   = _run_proc(["git", "commit", "-m", msg], folder, tout, maxch)
        if r["success"]:
            console.print(f"[dim]git auto-commit ✓  '{msg}'[/dim]")
    return written


# ═════════════════════════════════════════════════════════════════════════════
# Project CRUD
# ═════════════════════════════════════════════════════════════════════════════

def cmd_project_new(arg: str, projects: dict, lc: LollmsClient,
                    cfg: dict, scratch: dict) -> Optional[ProjectSession]:
    parts = arg.split(maxsplit=1)
    if not parts or not parts[0]:
        console.print("[red]Usage: /project new <n> [output_dir][/red]")
        return None
    name = parts[0].strip()
    if name in projects:
        console.print(f"[red]Project '{name}' already exists.[/red]")
        return None
    projects_root = Path(cfg.get("projects_root",
                                 str(_documents_dir() / "lollms_projects")))
    default_out   = projects_root / name
    out_path = Path(parts[1].strip()) if len(parts) > 1 else \
               Path(_ask_text("  Output folder", default=str(default_out)))
    default_db = CONFIG_DIR / "projects" / f"{name}.db"
    db_file    = Path(_ask_text("  DB file", default=str(default_db)))
    db_file.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    projects[name] = {
        "db_path":     f"sqlite:///{db_file}",
        "output_path": str(out_path),
        "created_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_projects(projects)
    console.print(f"[green]✅  Project '[bold]{name}[/bold]' created → {out_path}[/green]")
    return ProjectSession(name, projects, lc, cfg, scratch)


def cmd_project_switch(name: str, projects: dict, lc: LollmsClient,
                       cfg: dict, scratch: dict,
                       current: Optional[ProjectSession]) -> Optional[ProjectSession]:
    if not name:
        console.print("[red]Usage: /project switch <n>[/red]")
        return current
    if name not in projects:
        console.print(f"[red]Project '{name}' not found.[/red]")
        return current
    if current:
        current.close()
    s = ProjectSession(name, projects, lc, cfg, scratch)
    console.print(f"[green]Switched to '[bold]{name}[/bold]'.[/green]")
    return s


def cmd_project_ls(projects: dict, current_name: Optional[str]) -> None:
    if not projects:
        console.print("[dim]No projects. Use /project new <n>[/dim]")
        return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("",        width=2)
    t.add_column("Name",    style="bold white", no_wrap=True)
    t.add_column("Output",  style="dim", no_wrap=True)
    t.add_column("Created", style="dim")
    for n, meta in sorted(projects.items()):
        marker  = "[green]▶[/green]" if n == current_name else " "
        created = meta.get("created_at", "")[:10]
        t.add_row(marker, n, meta.get("output_path", "?"), created)
    console.print(t)


def cmd_project_info(session: ProjectSession) -> None:
    meta = session.projects[session.name]
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="yellow")
    t.add_row("Project",     session.name)
    t.add_row("Output",      meta.get("output_path", "?"))
    t.add_row("DB",          meta.get("db_path", "?").replace("sqlite:///", ""))
    t.add_row("Created",     meta.get("created_at", "?")[:19])
    t.add_row("Discussion",  session.disc.id)
    t.add_row("Artefacts",   str(len(session.disc.artefacts.list())))
    t.add_row("Memory",      f"{len(session.mem)} entries")
    t.add_row("Git repo",    "yes" if (session.out_folder / ".git").exists()
                              else "not initialised — /git init to start")
    console.print(Panel(t, title=f"Project: {session.name}", border_style="cyan"))


def cmd_project_delete(name: str, projects: dict,
                       current: Optional[ProjectSession]) -> Optional[ProjectSession]:
    if not name:
        console.print("[red]Usage: /project delete <n>[/red]")
        return current
    if name not in projects:
        console.print(f"[red]Project '{name}' not found.[/red]")
        return current
    if not _ask_confirm(
            f"  Remove project '{name}' from registry? (files are kept)"):
        return current
    if current and current.name == name:
        current.close()
        current = None
    del projects[name]
    save_projects(projects)
    console.print(f"[yellow]Project '{name}' removed from registry.[/yellow]")
    return current


# ═════════════════════════════════════════════════════════════════════════════
# Rich display helpers
# ═════════════════════════════════════════════════════════════════════════════

def show_artefact_table(disc: LollmsDiscussion) -> None:
    arts = disc.artefacts.list()
    if not arts:
        console.print("[dim]No artefacts yet.[/dim]")
        return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("Title",  style="bold white", no_wrap=True)
    t.add_column("Type",   style="cyan")
    t.add_column("Lang",   style="green")
    t.add_column("Ver",    justify="right")
    t.add_column("Active", justify="center")
    t.add_column("Lines",  justify="right")
    for a in arts:
        lines = len((a.get("content") or "").splitlines())
        t.add_row(
            a.get("title", "?"), a.get("type", "?"),
            a.get("language") or "–", str(a.get("version", 1)),
            "[green]✓[/green]" if a.get("active") else "[dim]·[/dim]",
            str(lines),
        )
    console.print(t)


def show_history(disc: LollmsDiscussion) -> None:
    branch = disc.get_branch(disc.active_branch_id)
    if not branch:
        console.print("[dim]No messages yet.[/dim]")
        return
    for msg in branch:
        label = "[bold blue]You[/bold blue]" if msg.sender_type == "user" \
                else "[bold green]Assistant[/bold green]"
        console.print(label)
        console.print(Markdown(msg.content or ""))
        console.print()


def print_welcome_banner() -> None:
    """Display a pleasant ASCII welcome banner."""
    banner_text = "\n".join([
    "╔══════════════════════════════════════════════════════════╗",
    "║                                                          ║",
    "║   [cyan]██╗      ██████╗ ██╗     ██╗     ███╗   ███╗███████╗[/cyan]   ║",
    "║   [cyan]██║     ██╔═══██╗██║     ██║     ████╗ ████║██╔════╝[/cyan]   ║",
    "║   [cyan]██║     ██║   ██║██║     ██║     ██╔████╔██║███████╗[/cyan]   ║",
    "║   [cyan]██║     ██║   ██║██║     ██║     ██║╚██╔╝██║╚════██║[/cyan]   ║",
    "║   [cyan]███████╗╚██████╔╝███████╗███████╗██║ ╚═╝ ██║███████║[/cyan]   ║",
    "║   [cyan]╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝[/cyan]   ║",
    "║                                                          ║",
    "║      [yellow] ██████╗ ██████╗ ██████╗ ███████╗██████╗ [/yellow]           ║",
    "║      [yellow]██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗[/yellow]           ║",
    "║      [yellow]██║     ██║   ██║██║  ██║█████╗  ██████╔╝[/yellow]           ║",
    "║      [yellow]██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗[/yellow]           ║",
    "║      [yellow]╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║[/yellow]           ║",
    "║      [yellow] ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝[/yellow]           ║",
    "║                                                          ║",
    "║                [bold bright_yellow]✨ Lollms Coder ✨[/bold bright_yellow]                        ║",
    "║             [bold]Built with love by ParisNeo[/bold]                  ║",
    "║                                                          ║",
    "╚══════════════════════════════════════════════════════════╝",
    ])
    ASCIIColors.rich_print(banner_text)
    ASCIIColors.rich_print(
        "[italic green]         Agentic Coding Assistant — Your AI Pair Programmer[/italic green]"
    )


def cmd_config(sub: str, arg: str, cfg: dict) -> dict:
    """Handle /config sub-commands for runtime configuration updates."""
    
    editable_keys = {
        "git_auto_commit": ("bool", "Auto-commit after /commit"),
        "shell_timeout": ("int", "Shell command timeout (seconds)"),
        "max_output_chars": ("int", "Max captured output chars"),
        "python_executable": ("str", "Python executable path"),
        "show_thinking": ("bool", "Show thinking process and tool calls"),
    }
    
    if sub in ("", "ls", "list"):
        t = Table(show_header=True, header_style="bold cyan", expand=False)
        t.add_column("Key", style="bold white", no_wrap=True)
        t.add_column("Type", style="dim")
        t.add_column("Current Value", style="yellow")
        t.add_column("Description", style="dim")
        
        for key, (typ, desc) in editable_keys.items():
            val = cfg.get(key, "—")
            t.add_row(key, typ, str(val), desc)
        
        console.print(Panel(t, title="Editable Configuration", border_style="cyan"))
        console.print("\n[dim]Use [bold]/config set <key> <value>[/bold] to modify.[/dim]")
        
    elif sub == "set":
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[red]Usage: /config set <key> <value>[/red]")
            console.print(f"[dim]Editable keys: {', '.join(editable_keys.keys())}[/dim]")
            return cfg
            
        key, raw_val = parts[0], parts[1].strip()
        
        if key not in editable_keys:
            console.print(f"[red]Unknown key: '{key}'[/red]")
            console.print(f"[dim]Available: {', '.join(editable_keys.keys())}[/dim]")
            return cfg
            
        typ, desc = editable_keys[key]
        
        # Type conversion
        try:
            if typ == "bool":
                val = raw_val.lower() in ("true", "yes", "1", "on")
            elif typ == "int":
                val = int(raw_val)
            else:  # str
                val = raw_val
        except ValueError as e:
            console.print(f"[red]Invalid value for {typ}: {raw_val}[/red]")
            return cfg
        
        # Update and save
        cfg[key] = val
        save_config(cfg)
        console.print(f"  [green]✓[/green]  [bold]{key}[/bold] = {val!r}  [dim]({desc})[/dim]")
        
    elif sub == "get":
        key = arg.strip()
        if not key:
            console.print("[red]Usage: /config get <key>[/red]")
        elif key not in editable_keys:
            console.print(f"[red]Unknown key: '{key}'[/red]")
        else:
            val = cfg.get(key, "—")
            typ, desc = editable_keys[key]
            console.print(f"  [bold cyan]{key}[/bold cyan]  [dim]({typ})[/dim]")
            console.print(f"  Value: [yellow]{val!r}[/yellow]")
            console.print(f"  {desc}")
            
    else:
        console.print("[red]Unknown /config sub-command.[/red]  ls | set | get")
        
    return cfg


def cmd_soul(sub: str, arg: str) -> None:
    """Handle /soul sub-commands for personality management."""
    if sub in ("", "show", "cat"):
        soul = load_soul()
        console.print(Panel(Markdown(soul), title="Current Soul", border_style="magenta"))
        
    elif sub == "edit":
        import tempfile, subprocess, shutil
        
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or (
            "notepad" if platform.system() == "Windows" else "nano"
        )
        soul = load_soul()
        
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
            f.write(soul)
            tmp_path = f.name
        
        try:
            subprocess.call([editor, tmp_path])
            new_soul = Path(tmp_path).read_text(encoding="utf-8")
            if new_soul != soul:
                save_soul(new_soul)
                console.print("[green]✓ Soul updated.[/green]")
            else:
                console.print("[dim]No changes made.[/dim]")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    elif sub == "reset":
        if _ask_confirm("Reset soul to default?", default=False):
            save_soul(DEFAULT_SOUL)
            console.print("[yellow]Soul reset to default.[/yellow]")
            
    else:
        console.print("[red]Unknown /soul sub-command.[/red]  show | edit | reset")


def cmd_skill(sub: str, arg: str, skills: list[Skill]) -> list[Skill]:
    """Handle /skill sub-commands for skill management."""
    
    if sub in ("", "ls", "list"):
        if not skills:
            console.print("[dim]No skills defined. Use /skill new <name>[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("", width=2)  # enabled marker
            t.add_column("Name", style="bold white", no_wrap=True)
            t.add_column("Category", style="cyan")
            t.add_column("Description", style="dim")
            
            for s in skills:
                marker = "[green]✓[/green]" if s.enabled else "[dim]·[/dim]"
                t.add_row(marker, s.name, s.category, s.description[:50])
            console.print(Panel(t, title=f"Skills ({len(skills)})", border_style="cyan"))
            
    elif sub == "new":
        name = arg.strip().replace(" ", "_").lower()
        if not name:
            console.print("[red]Usage: /skill new <name>[/red]")
            return skills
            
        if any(s.name == name for s in skills):
            console.print(f"[red]Skill '{name}' already exists.[/red]")
            return skills
            
        description = _ask_text("  Description")
        category = _ask_text("  Category", default="general")
        
        console.print("  Enter system prompt addition (Ctrl+D or EOF to finish):")
        lines = []
        try:
            while True:
                line = input("  > ")
                lines.append(line)
        except EOFError:
            pass
        prompt_addition = "\n".join(lines)
        
        skill = Skill(name, description, prompt_addition, category, enabled=True)
        skill.save()
        skills.append(skill)
        console.print(f"[green]✓ Skill '{name}' created and enabled.[/green]")
        
    elif sub == "show":
        name = arg.strip()
        skill = next((s for s in skills if s.name == name), None)
        if not skill:
            console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            console.print(Panel(
                Markdown(f"## {skill.name} ({skill.category})\n\n{skill.description}\n\n---\n\n{skill.prompt_addition}"),
                title=f"Skill: {skill.name}",
                border_style="cyan" if skill.enabled else "dim"
            ))
            
    elif sub == "toggle":
        name = arg.strip()
        skill = next((s for s in skills if s.name == name), None)
        if not skill:
            console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            skill.enabled = not skill.enabled
            skill.save()
            state = "enabled" if skill.enabled else "disabled"
            console.print(f"[green]Skill '{name}' {state}.[/green]")
            
    elif sub == "delete":
        name = arg.strip()
        skill = next((s for s in skills if s.name == name), None)
        if not skill:
            console.print(f"[red]Skill '{name}' not found.[/red]")
        elif _ask_confirm(f"Delete skill '{name}'?", default=False):
            skill.file_path.unlink(missing_ok=True)
            skills.remove(skill)
            console.print(f"[yellow]Skill '{name}' deleted.[/yellow]")
            
    elif sub == "edit":
        name = arg.strip()
        skill = next((s for s in skills if s.name == name), None)
        if not skill:
            console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            import tempfile, subprocess
            
            editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or (
                "notepad" if platform.system() == "Windows" else "nano"
            )
            
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
                f.write(skill.file_path.read_text())
                tmp_path = f.name
            
            try:
                subprocess.call([editor, tmp_path])
                # Reload to get changes
                updated = Skill.load(Path(tmp_path))
                updated.file_path = skill.file_path
                updated.save()
                # Update in list
                idx = skills.index(skill)
                skills[idx] = updated
                console.print(f"[green]✓ Skill '{name}' updated.[/green]")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
    else:
        console.print("[red]Unknown /skill sub-command.[/red]  ls | new | show | toggle | delete | edit")
        
    return skills


def print_help() -> None:
    sections = [
        ("Setup & Personality", [
            ("/setup",               "Re-run full setup wizard (binding, model, paths…)"),
            ("/setup show",          "Print current config (secrets masked)"),
            ("/config",              "List editable runtime settings"),
            ("/config set <k> <v>",  "Update a config value"),
            ("/config get <k>",      "View specific config value"),
            ("/soul",                "Show current personality (the 'soul')"),
            ("/soul edit",           "Edit personality in $EDITOR"),
            ("/soul reset",          "Reset personality to default"),
            ("/skill",               "List all skills"),
            ("/skill new <name>",    "Create a new skill"),
            ("/skill toggle <name>", "Enable/disable a skill"),
            ("/skill show <name>",   "View skill details"),
            ("/skill edit <name>",   "Edit skill in $EDITOR"),
            ("/skill delete <name>", "Delete a skill"),
            ("/reset_configuration", "[red]⚠️ Delete all configs & exit (fresh install)[/red]"),
        ]),
        ("Projects", [
            ("/project new <n> [dir]",  "Create & switch to a new project"),
            ("/project switch <n>",     "Switch to an existing project"),
            ("/project ls",               "List all projects"),
            ("/project info",             "Current project details"),
            ("/project delete <n>",    "Remove from registry"),
        ]),
        ("Scratchpad  (global inter-agent / inter-task key-value store)", [
            ("/scratch set <key> <value>", "Write an entry"),
            ("/scratch get <key>",         "Read an entry"),
            ("/scratch ls",                  "List all entries"),
            ("/scratch del <key>",         "Delete an entry"),
            ("/scratch clear",               "Wipe everything"),
        ]),
        ("Project memory  (per-project long-term memory)", [
            ("/memory add <text>",    "Append a memory entry"),
            ("/memory ls",              "List all entries"),
            ("/memory del <id>",      "Delete by id"),
            ("/memory clear",           "Wipe all entries"),
            ("/memory search <query>","Keyword search"),
        ]),
        ("Artefacts", [
            ("/ls",               "List artefacts"),
            ("/show <n>",      "Pretty-print one artefact"),
            ("/activate <n>",  "Inject into LLM context"),
            ("/deactivate <n>","Remove from LLM context"),
        ]),
        ("Files, execution & tests", [
            ("/commit [folder]",   "Write artefacts to disk"),
            ("/run <file> [args]", "Execute a file"),
            ("/shell <command>",   "Shell command in project folder"),
            ("/test [path]",       "Run pytest"),
        ]),
        ("Git", [
            ("/git init",              "git init"),
            ("/git status",            "git status"),
            ("/git add [files]",       "Stage files (default: all)"),
            ("/git commit <message>",  "Commit staged changes"),
            ("/git diff [file]",       "Uncommitted changes"),
            ("/git log [n]",           "Last n commits"),
            ("/git branch [name]",     "List or create branch"),
            ("/git checkout <branch>", "Switch branch"),
        ]),
        ("Conversation", [
            ("/clear",   "New discussion (same project)"),
            ("/history", "Show conversation"),
            ("/help",    "This help"),
            ("/quit",    "Exit"),
            ("/reset_configuration", "[red]⚠️  NUCLEAR: wipe all data & exit[/red]"),
        ]),
    ]
    for title, rows in sections:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="bold yellow", no_wrap=True)
        t.add_column(style="white")
        for c, d in rows:
            t.add_row(c, d)
        console.print(Panel(t, title=f"[bold]{title}[/bold]", border_style="dim"))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def prompt_marker(session: Optional[ProjectSession]) -> str:
    proj = f"[dim cyan]{session.name}[/dim cyan] " if session else ""
    return f"{proj}[bold blue]❯[/bold blue]"


def build_completer(projects: dict, session: Optional[ProjectSession]) -> Optional:
    """Build nested completer for all slash commands."""
    if not _HAS_PROMPT_TOOLKIT:
        return None
    
    commands = {
        "/quit": None,
        "/exit": None,
        "/help": None,
        "/clear": None,
        "/history": None,
        "/ls": None,
        "/files": None,
        "/commit": None,
        "/setup": {"show": None},
        "/config": {"set": None, "get": None, "ls": None, "list": None},
        "/soul":   {"show": None, "cat": None, "edit": None, "reset": None},
        "/skill":  {"ls": None, "list": None, "new": None, "show": None,
                    "toggle": None, "delete": None, "del": None, "edit": None},
        "/project":{"new": None, "switch": None, "ls": None, "list": None,
                    "info": None, "delete": None, "del": None},
        "/scratch":{"set": None, "get": None, "ls": None, "list": None,
                    "del": None, "delete": None, "clear": None},
        "/memory": {"add": None, "ls": None, "list": None, "del": None,
                    "delete": None, "clear": None, "search": None},
        "/reset_configuration": None,
        "/activate": None,
        "/deactivate": None,
        "/show": None,
        "/run": None,
        "/shell": None,
        "/test": None,
        "/git": {"init": None, "status": None, "add": None, "commit": None,
                 "diff": None, "log": None, "branch": None, "checkout": None},
    }
    
    if session:
        commands["/project"]["switch"] = {p: None for p in projects.keys()}
        try:
            arts = session.disc.artefacts.list()
            art_names = {a.get("title", ""): None for a in arts if a.get("title")}
            commands["/activate"] = art_names
            commands["/deactivate"] = art_names
            commands["/show"] = art_names
        except:
            pass
    
    return NestedCompleter.from_nested_dict(commands)


def get_input_with_history(session: Optional[ProjectSession], projects: dict) -> str:
    """Get user input with history and autocompletion."""
    marker_text = (prompt_marker(session)
                   .replace("[/]", "").replace("[dim cyan]", "")
                   .replace("[bold blue]", "").replace("[/dim cyan]", "")
                   .replace("[/bold blue]", ""))
    
    if not _HAS_PROMPT_TOOLKIT:
        try:
            return Prompt.ask(marker_text).strip()
        except (EOFError, KeyboardInterrupt):
            return "/quit"
    
    completer = build_completer(projects, session)
    prompt_session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        completer=completer,
        complete_style=CompleteStyle.READLINE_LIKE,
        key_bindings=KeyBindings(),
    )
    
    try:
        result = prompt_session.prompt(marker_text)
        return result.strip()
    except (EOFError, KeyboardInterrupt):
        return "/quit"


def main() -> None:
    # ── config ────────────────────────────────────────────────────────────────
    cfg = load_config()
    first_run = _is_fresh_install() or not CONFIG_FILE.exists() or not cfg.get("binding_config")
    
    print_welcome_banner()
    if first_run:
        console.print("\n[yellow]Welcome! Let's set up your LLM binding.[/yellow]\n")
        cfg = run_setup_wizard(cfg, first_run=True)

    lc = build_client(cfg)

    # ── global state ─────────────────────────────────────────────────────────
    projects = load_projects()
    scratch  = load_scratchpad()
    skills   = load_skills()
    session: Optional[ProjectSession] = None

    if projects:
        latest  = max(projects, key=lambda n: projects[n].get("created_at", ""))
        session = ProjectSession(latest, projects, lc, cfg, scratch)
        console.print(
            f"[dim]Binding: [bold]{cfg['binding_name']}[/bold]  "
            f"model: {cfg['binding_config'].get('model_name','?')}  "
            f"|  Project: [bold]{latest}[/bold][/dim]\n"
        )
    else:
        console.print(
            f"[dim]Binding: [bold]{cfg['binding_name']}[/bold]  "
            f"model: {cfg['binding_config'].get('model_name','?')}[/dim]\n"
            "[yellow]No projects yet.[/yellow]  "
            "Create one: [bold]/project new <n>[/bold]\n"
        )

    # ── REPL ─────────────────────────────────────────────────────────────────
    while True:
        raw = get_input_with_history(session, projects)
        if raw == "/quit":
            console.print("\n[dim]Bye![/dim]")
            break
        if not raw:
            continue

        cmd, _, arg = raw.partition(" ")
        arg = arg.strip()

        # ── /quit ─────────────────────────────────────────────────────────────
        if cmd in ("/quit", "/exit"):
            console.print("[dim]Bye![/dim]")
            break

        # ── /reset_configuration ──────────────────────────────────────────────
        if cmd == "/reset_configuration":
            console.print(Panel(
                "[bold red]⚠️  DESTRUCTIVE OPERATION[/bold red]\n\n"
                "This will permanently delete:\n"
                "  • All configuration files (~/.lollms/coding_cli/)\n"
                "  • Project registry and databases\n"
                "  • Scratchpad contents\n"
                "  • Any cached or temporary data\n\n"
                "This action cannot be undone.",
                title="[bold red]RESET CONFIGURATION[/bold red]",
                border_style="red",
            ))
            if _ask_confirm("  [red]Type YES to permanently delete all data:[/red]", default=False):
                try:
                    import shutil
                    if session:
                        console.print("[dim]Closing database connections...[/dim]")
                        session.close()
                        session = None
                    import gc
                    gc.collect()
                    if CONFIG_DIR.exists():
                        marker_path = CONFIG_DIR.parent / ".coding_cli_reset_marker"
                        try:
                            marker_path.parent.mkdir(parents=True, exist_ok=True)
                            marker_path.write_text("reset_requested", encoding="utf-8")
                        except Exception as me:
                            console.print(f"[yellow]Warning: Could not create marker: {me}[/yellow]")
                        import time
                        time.sleep(0.5)
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                shutil.rmtree(CONFIG_DIR)
                                break
                            except PermissionError as pe:
                                if attempt < max_retries - 1:
                                    console.print(f"[dim]Waiting for file release (attempt {attempt + 1}/{max_retries})...[/dim]")
                                    time.sleep(1.0)
                                    gc.collect()
                                else:
                                    raise
                        console.print(f"[red]✓ Deleted:[/red] {CONFIG_DIR}")
                    else:
                        console.print(f"[yellow]Config directory not found: {CONFIG_DIR}[/yellow]")
                    RESET_MARKER.parent.mkdir(parents=True, exist_ok=True)
                    RESET_MARKER.touch()
                    console.print("\n[bold red]All configuration data has been erased.[/bold red]")
                    console.print("[dim]The program will now exit. Next run will be a fresh install.[/dim]")
                except Exception as e:
                    console.print(f"[red]Error during cleanup: {e}[/red]")
                    console.print("[yellow]Some files may remain. Please manually delete:[/yellow]")
                    console.print(f"  {CONFIG_DIR}")
                    console.print("[dim]Bye![/dim]")
                    break
            else:
                console.print("[dim]Reset cancelled.[/dim]")
            console.print("[dim]Bye![/dim]")
            break

        # ── /help ─────────────────────────────────────────────────────────────
        if cmd == "/help":
            print_help()
            continue

        # ── /config ───────────────────────────────────────────────────────────
        if cmd == "/config":
            sub, _, sub_arg = arg.partition(" ")
            cfg = cmd_config(sub.strip(), sub_arg.strip(), cfg)
            if session:
                session.update_config(cfg)
            continue

        # ── /soul ─────────────────────────────────────────────────────────────
        if cmd == "/soul":
            sub, _, sub_arg = arg.partition(" ")
            cmd_soul(sub.strip(), sub_arg.strip())
            if session:
                session.refresh_system_prompt()
            continue

        # ── /skill ────────────────────────────────────────────────────────────
        if cmd == "/skill":
            sub, _, sub_arg = arg.partition(" ")
            skills = cmd_skill(sub.strip(), sub_arg.strip(), skills)
            if session:
                session.refresh_system_prompt()
            continue

        # ── /setup ────────────────────────────────────────────────────────────
        if cmd == "/setup":
            if arg == "show":
                show_config(cfg)
            else:
                cfg = run_setup_wizard(cfg)
                lc  = build_client(cfg)
                if session:
                    session._lc = lc
                    session.update_config(cfg)
                console.print("[dim]LLM client and tools rebuilt.[/dim]")
            continue

        # ── /project ──────────────────────────────────────────────────────────
        if cmd == "/project":
            sub, _, sub_arg = arg.partition(" ")
            sub_arg = sub_arg.strip()
            if sub == "new":
                ns = cmd_project_new(sub_arg, projects, lc, cfg, scratch)
                if ns:
                    if session:
                        session.close()
                    session = ns
            elif sub == "switch":
                session = cmd_project_switch(sub_arg, projects, lc, cfg, scratch, session)
            elif sub == "ls":
                cmd_project_ls(projects, session.name if session else None)
            elif sub == "info":
                if session:
                    cmd_project_info(session)
                else:
                    console.print("[red]No active project.[/red]")
            elif sub == "delete":
                session = cmd_project_delete(sub_arg, projects, session)
                if session is None and projects:
                    other   = max(projects, key=lambda n: projects[n].get("created_at", ""))
                    session = ProjectSession(other, projects, lc, cfg, scratch)
                    console.print(f"[dim]Auto-switched to '[bold]{other}[/bold]'.[/dim]")
            else:
                console.print("[red]Unknown /project sub-command.[/red]  new | switch | ls | info | delete")
            continue

        # ── /scratch ──────────────────────────────────────────────────────────
        if cmd == "/scratch":
            sub, _, sub_arg = arg.partition(" ")
            scratch = cmd_scratch(sub.strip(), sub_arg.strip(), scratch)
            if session:
                session._scratch = scratch
                session.refresh_system_prompt()
            continue

        # ── require active project ────────────────────────────────────────────
        if session is None:
            console.print(
                "[red]No active project.[/red]  Create one: [bold]/project new <n>[/bold]"
            )
            continue

        disc   = session.disc
        tools  = session.tools
        folder = session.out_folder
        tout   = cfg.get("shell_timeout", 30)
        maxch  = cfg.get("max_output_chars", 8000)

        # ── /memory ───────────────────────────────────────────────────────────
        if cmd == "/memory":
            sub, _, sub_arg = arg.partition(" ")
            session._mem = cmd_memory(sub.strip(), sub_arg.strip(),
                                      folder, session._mem)
            session._rebuild_tools()
            session.refresh_system_prompt()
            continue

        # ── /clear ────────────────────────────────────────────────────────────
        if cmd == "/clear":
            session.new_discussion()
            continue

        # ── /history ──────────────────────────────────────────────────────────
        if cmd == "/history":
            show_history(disc)
            continue

        # ── /ls ───────────────────────────────────────────────────────────────
        if cmd == "/ls":
            show_artefact_table(disc)
            continue

        # ── /files ────────────────────────────────────────────────────────────
        if cmd == "/files":
            r = tools["list_files"]["callable"](path=arg or ".")
            if r["success"]:
                console.print(Panel(r["listing"], title="[cyan]project files[/cyan]", border_style="dim"))
            else:
                console.print(f"[red]{r['output']}[/red]")
            continue

        # ── /show ─────────────────────────────────────────────────────────────
        if cmd == "/show":
            if not arg:
                console.print("[red]Usage: /show <artefact_name>[/red]")
                continue
            a = disc.artefacts.get(arg)
            if a is None:
                console.print(f"[red]Artefact '{arg}' not found.[/red]")
            else:
                console.print(Syntax(a.get("content", ""),
                                     a.get("language") or "text",
                                     theme="monokai", line_numbers=True))
            continue

        # ── /activate / /deactivate ───────────────────────────────────────────
        if cmd == "/activate":
            if not arg:
                console.print("[red]Usage: /activate <n>[/red]"); continue
            try:
                disc.artefacts.activate(arg)
                console.print(f"[green]Activated '{arg}'.[/green]")
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue

        if cmd == "/deactivate":
            if not arg:
                console.print("[red]Usage: /deactivate <n>[/red]"); continue
            try:
                disc.artefacts.deactivate(arg)
                console.print(f"[dim]Deactivated '{arg}'.[/dim]")
            except Exception as e:
                console.print(f"[red]{e}[/red]")
            continue

        # ── /commit ───────────────────────────────────────────────────────────
        if cmd == "/commit":
            target  = Path(arg) if arg else folder
            written = commit_artefacts(disc, target, cfg)
            if not written:
                console.print("[yellow]No code artefacts to commit.[/yellow]")
            else:
                console.print(Rule(f"[green]Committed {len(written)} file(s) → {target}[/green]"))
                for p in written:
                    console.print(f"  [green]✓[/green]  {p}")
            continue

        # ── /run ──────────────────────────────────────────────────────────────
        if cmd == "/run":
            if not arg:
                console.print("[red]Usage: /run <file> [args…][/red]"); continue
            parts = arg.split(maxsplit=1)
            tools["run_code"]["callable"](filename=parts[0],
                                          args=parts[1] if len(parts) > 1 else "")
            continue

        # ── /shell ────────────────────────────────────────────────────────────
        if cmd == "/shell":
            if not arg:
                console.print("[red]Usage: /shell <command>[/red]"); continue
            tools["run_shell"]["callable"](command=arg)
            continue

        # ── /test ─────────────────────────────────────────────────────────────
        if cmd == "/test":
            r = tools["run_tests"]["callable"](path=arg or ".")
            if r["success"]:
                console.print("[bold green]✅  All tests passed.[/bold green]")
            else:
                console.print("[bold red]❌  Tests failed.[/bold red]\n"
                              "[dim]Paste the error above and ask me to fix it.[/dim]")
            continue

        # ── /git ──────────────────────────────────────────────────────────────
        if cmd == "/git":
            sub, _, sub_arg = arg.partition(" ")
            sub_arg = sub_arg.strip()
            g = tools
            if sub == "init":
                r = _run_proc(["git", "init"], folder, tout, maxch)
                console.print(Panel(r["output"], border_style="green" if r["success"] else "red"))
            elif sub == "status":
                g["git_status"]["callable"]()
            elif sub == "add":
                g["git_add"]["callable"](files=sub_arg or ".")
            elif sub == "commit":
                if not sub_arg:
                    console.print("[red]Usage: /git commit <message>[/red]")
                else:
                    g["git_commit"]["callable"](message=sub_arg)
            elif sub == "diff":
                g["git_diff"]["callable"](file=sub_arg)
            elif sub == "log":
                g["git_log"]["callable"](n=int(sub_arg) if sub_arg.isdigit() else 10)
            elif sub == "branch":
                g["git_branch"]["callable"](name=sub_arg)
            elif sub == "checkout":
                if not sub_arg:
                    console.print("[red]Usage: /git checkout <branch>[/red]")
                else:
                    g["git_checkout"]["callable"](branch=sub_arg)
            else:
                console.print("[red]Unknown git sub-command.[/red]  "
                              "init | status | add | commit | diff | log | branch | checkout")
            continue

        # ── unknown slash command ─────────────────────────────────────────────
        if cmd.startswith("/"):
            console.print(f"[red]Unknown command '{cmd}'.[/red]  Type [bold]/help[/bold].")
            continue

        # ══════════════════════════════════════════════════════════════════════
        # LLM turn
        # ══════════════════════════════════════════════════════════════════════
        console.print(Rule("[bold green]Assistant[/bold green]"))
        console.print()

        show_thinking = cfg.get("show_thinking", True)
        buffer = ""

        # Detect if user is requesting code execution but didn't explicitly mention "tool"
        action_patterns = [
            r'\brun\s+(?:the\s+)?(?:server|app|application|code|script|main)',
            r'\bstart\s+(?:the\s+)?(?:server|app|application)',
            r'\bexecute\s+(?:the\s+)?(?:code|script|file)',
            r'\blaunch\s+(?:the\s+)?(?:server|app)',
            r'\bbuild\s+(?:and\s+run|\s+the\s+project)',
        ]
        needs_explicit_tools = any(
            re.search(p, raw, re.IGNORECASE) for p in action_patterns
        )
        
        if needs_explicit_tools:
            # Prepend a reminder to actually use tools
            raw = f"[System reminder: The user wants you to EXECUTE code using the run_code tool. Do not describe the action - invoke the tool.]\n\n{raw}"

        # ── Spinner: shown while waiting for first token ──────────────────────
        # ASCIIColors.Live lets us display animated text that we can stop/replace.
        _spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        _spinner_idx    = [0]
        _spinner_active = [True]   # mutable so the closure can flip it

        import threading, time as _time

        def _spin():
            while _spinner_active[0]:
                frame = _spinner_frames[_spinner_idx[0] % len(_spinner_frames)]
                # Overwrite the current line with the spinner + label
                ASCIIColors.rich_print(f"\r  {frame} [dim]Thinking…[/dim]  ", end="", flush=True)
                _spinner_idx[0] += 1
                _time.sleep(0.08)

        _spinner_thread = threading.Thread(target=_spin, daemon=True)
        _spinner_thread.start()

        def _stop_spinner():
            """Kill the spinner and erase its line."""
            if _spinner_active[0]:
                _spinner_active[0] = False
                _spinner_thread.join(timeout=0.3)
                # Erase the spinner line completely
                print("\r" + " " * 40 + "\r", end="", flush=True)

        # ── Callback ──────────────────────────────────────────────────────────
        # stream_cb is called as callback(text, msg_type, meta) — always 3 args.
        # It stops the spinner on the first real content (chunk or step marker).
        def stream_cb(text: str, msg_type=None, meta=None) -> bool:
            nonlocal buffer

            # ── Step / phase notifications (agentic mode) ─────────────────
            if msg_type in (MSG_TYPE.MSG_TYPE_STEP_START,
                            MSG_TYPE.MSG_TYPE_STEP_END):
                _stop_spinner()
                prefix = "▷" if msg_type == MSG_TYPE.MSG_TYPE_STEP_START else "◁"
                ASCIIColors.cyan(f"  {prefix} {text}", end="\n")
                return True

            # ── Tool call notification ────────────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
                import json as _json
                _stop_spinner()
                try:
                    td = (_json.loads(text)
                          if isinstance(text, str) and text.strip().startswith("{")
                          else {"name": str(text)})
                    name = td.get("name", "tool")
                except Exception:
                    name = str(text)[:60]
                ASCIIColors.cyan(f"  ▶ Running {name}...", end="\n")
                return True

            # ── Tool output notification ──────────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
                _stop_spinner()
                # Display actual tool result if present in meta
                result_preview = ""
                if isinstance(meta, dict) and "result" in meta:
                    r = meta["result"]
                    if isinstance(r, dict):
                        # Show key info: success status, output, or error
                        if r.get("output"):
                            result_preview = str(r["output"])[:500]
                        elif r.get("content"):
                            result_preview = str(r["content"])[:500]
                        elif r.get("error"):
                            result_preview = f"Error: {str(r.get('error'))[:300]}"
                        else:
                            result_preview = json.dumps(r, default=str)[:500]
                    else:
                        result_preview = str(r)[:500]
                    if len(result_preview) >= 500:
                        result_preview += "…"
                ASCIIColors.green(f"  ✓ Tool complete", end="")
                if result_preview:
                    ASCIIColors.dim(f" → {result_preview}", end="")
                ASCIIColors.print("")  # newline
                return True

            # ── Exception/Error messages ──────────────────────────────────
            if msg_type in (MSG_TYPE.MSG_TYPE_EXCEPTION, MSG_TYPE.MSG_TYPE_ERROR):
                _stop_spinner()
                ASCIIColors.red(f"  ✗ Error: {text}", end="\n")
                return True

            # ── Info messages ─────────────────────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_INFO:
                _stop_spinner()
                ASCIIColors.dim(f"  ℹ  {text}", end="\n")
                return True

            # ── Warnings ─────────────────────────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_WARNING:
                _stop_spinner()
                ASCIIColors.yellow(f"  ⚠  {text}", end="\n")
                return True

            # ── Scratchpad snapshots — silent ─────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_SCRATCHPAD:
                return True

            # ── Text chunks ───────────────────────────────────────────────
            # Stop the spinner on first chunk, then accumulate silently.
            # We do NOT print chunks raw — we render once as Markdown after
            # the full response is received, avoiding the double-display bug.
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                _stop_spinner()
                buffer += text
                return True

            return True

        result = disc.chat(
            user_message=raw,
            tools=tools,
            streaming_callback=stream_cb,
            auto_activate_artefacts=True,
        )

        # Make sure spinner is gone even if no tokens arrived
        _stop_spinner()

        # Render the complete accumulated response as Markdown (once only)
        if buffer.strip():
            console.print(Markdown(buffer))
        console.print()

        # ── post-turn housekeeping ────────────────────────────────────────────
        # Re-inject memory/scratchpad into system prompt in case tools wrote to them
        session.reload_memory()
        session.refresh_system_prompt()

        # Check if tools were actually invoked vs. hallucinated
        tool_calls = result.get("tool_calls", []) if isinstance(result, dict) else []
        affected = result.get("artefacts", []) if isinstance(result, dict) else []
        
        # Heuristic: if user asked to run/start/execute something and no tools were called,
        # the model likely hallucinated instead of using the available tools
        action_verbs = ("run", "start", "execute", "launch", "build", "test", "commit", "deploy")
        lower_input = raw.lower()
        looks_like_action = any(v in lower_input for v in action_verbs)
        
        if looks_like_action and not tool_calls:
            console.print()
            console.print(Panel(
                "[yellow]⚠️  The assistant responded without using any tools.[/yellow]\n\n"
                "The model may have [bold]hallucinated[/bold] the result rather than actually "
                "executing the requested action.\n\n"
                "[dim]If you need the action performed, try being more explicit:[/dim]\n"
                "  • 'Use the run_code tool to execute the server'\n"
                "  • 'Run python main.py using the available tool'",
                title="[yellow]Possible Hallucination Detected[/yellow]",
                border_style="yellow"
            ))
            console.print()
        
        if affected:
            names = ", ".join(a.get("title", "?") for a in affected)
            console.print(
                f"[dim]📦  Artefact(s) saved: [bold]{names}[/bold]"
                "  —  /commit to write to disk[/dim]"
            )
        
        if tool_calls:
            console.print(
                f"[dim]🔧  Tools used: {len(tool_calls)}  "
                f"({', '.join(t.get('name', '?') for t in tool_calls)})[/dim]"
            )

    if session:
        session.close()


if __name__ == "__main__":
    main()