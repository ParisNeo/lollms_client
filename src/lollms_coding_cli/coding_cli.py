#!/usr/bin/env python3
# coding_cli.py  –  lollms Agentic Coding Assistant
#
# Install:  pip install rich questionary pyyaml ascii_colors lollms_client prompt-toolkit
# Run:      python coding_cli.py
#
# All state lives under ~/.lollms/coding_cli/   (never committed to repos)
#   config.yaml        –  LLM binding + runtime settings
#   projects.yaml      –  project registry
#   scratchpad.yaml    –  global inter-agent key-value store
#   soul.md            –  persistent AI personality
#   skills/            –  file-based reusable skill modules (legacy, still works)
#   projects/<n>.db    –  SQLite discussion databases
#
# Per-project:   <output_folder>/.lollms/memory.yaml
#
# ── Slash commands ──────────────────────────────────────────────────────────
#
#  Setup & Personality
#   /setup                   re-run full setup wizard
#   /setup show              print current config (secrets masked)
#   /config                  list editable runtime settings
#   /config set <k> <v>      update a setting (inc. enable_notes, enable_skills)
#   /soul                    show current personality
#   /soul edit               edit personality in $EDITOR
#   /soul reset              reset to default
#
#  Legacy file-based Skills  (prompt-engineering modules stored as .md files)
#   /fskill                  list file-based skills
#   /fskill new <name>       create a new file skill
#   /fskill show <name>      show a file skill
#   /fskill toggle <name>    enable / disable a file skill
#   /fskill edit <name>      edit in $EDITOR
#   /fskill delete <name>    delete permanently
#
#  Projects
#   /project new <n> [dir]   create & switch to a new project
#   /project switch <n>      switch to an existing project
#   /project ls              list all projects
#   /project info            current project details (inc. note/skill counts)
#   /project delete <n>      remove from registry (files kept)
#
#  Scratchpad  (global inter-agent / inter-task key-value store)
#   /scratch set <key> <val> write an entry
#   /scratch get <key>       read an entry
#   /scratch ls              list all entries
#   /scratch del <key>       delete an entry
#   /scratch clear           wipe everything (with confirmation)
#
#  Project long-term memory  (per-project, injected into LLM context)
#   /memory add <text>       append a memory entry
#   /memory ls               list all entries
#   /memory del <id>         delete by numeric id
#   /memory clear            wipe (with confirmation)
#   /memory search <query>   case-insensitive keyword search
#
#  Code Artefacts  (code, documents, files saved by the LLM)
#   /ls                      list code/document artefacts
#   /show <n>                syntax-highlighted view
#   /activate <n>            inject into LLM context
#   /deactivate <n>          remove from context
#   /commit [folder]         write code artefacts to disk
#
#  Notes  (structured notes saved by LLM via <note> tags)
#   /notes                   list all notes
#   /note show <title>       view a note (prefix match OK)
#   /note save <title> [path] export note to Markdown file
#
#  Skills  (reusable patterns saved by LLM via <skill> tags)
#   /skills                  list all skills
#   /skill show <title>      view a skill (prefix match OK)
#   /skill toggle <title>    enable / disable a skill
#   /skill delete <title>    permanently remove a skill
#
#  Files, execution & tests
#   /run <file> [args]       execute in project folder
#   /shell <command>         shell command in project folder
#   /test [path]             run pytest
#   /files [path]            list project files
#
#  Process management  (background / long-running processes)
#   /ps                      list running processes
#   /kill <pid>              stop a background process
#   /proc <pid>              read recent output from a background process
#
#  Git  (runs inside the project output folder)
#   /git init | status | add [files] | commit <msg>
#       diff [file] | log [n] | branch [name] | checkout <branch>
#
#  Conversation
#   /clear                   new discussion (same project)
#   /history                 show conversation
#   /help                    this list
#   /quit                    exit
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import base64
import platform
import shutil
import subprocess
import sys
import time
import threading
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from ascii_colors import (
    ASCIIColors, Console, Markdown, Panel, Confirm, Prompt,
    Rule, Syntax, Table, trace_exception, questionary, Style as QStyle,
)
from ascii_colors.rich import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False

_HAS_QUESTIONARY = True

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import (
    ArtefactType,
    LollmsDataManager,
    LollmsDiscussion,
)
from lollms_client.lollms_types import MSG_TYPE

# ═════════════════════════════════════════════════════════════════════════════
# Paths
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
SOUL_FILE       = CONFIG_DIR / "soul.md"
SKILLS_DIR      = CONFIG_DIR / "skills"        # file-based (legacy) skills

for _d in (CONFIG_DIR, SKILLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESET_MARKER = CONFIG_DIR / ".reset_marker"

# ═════════════════════════════════════════════════════════════════════════════
# Soul  (persistent AI personality)
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

When given a task, analyse requirements, propose an approach, then implement with the tools available.
"""


def load_soul() -> str:
    if SOUL_FILE.exists():
        try:
            return SOUL_FILE.read_text(encoding="utf-8")
        except Exception:
            pass
    SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
    return DEFAULT_SOUL


def save_soul(content: str) -> None:
    SOUL_FILE.write_text(content, encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# File-based Skills  (legacy prompt-engineering modules, stored in ~/.lollms/
#                     coding_cli/skills/*.md)
#
# These are distinct from library-backed SKILL artefacts (§6 of the docs).
# File skills inject permanent extra system-prompt sections; artefact skills
# are created by the LLM on demand and stored in the discussion database.
# Both coexist and complement each other.
# ═════════════════════════════════════════════════════════════════════════════

class FileSkill:
    """A reusable system-prompt module stored as a Markdown file."""

    def __init__(self, name: str, description: str, prompt_addition: str,
                 category: str = "general", enabled: bool = True):
        self.name             = name
        self.description      = description
        self.prompt_addition  = prompt_addition
        self.category         = category
        self.enabled          = enabled
        self.file_path        = SKILLS_DIR / f"{name}.md"

    def save(self) -> None:
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
    def load(cls, path: Path) -> "FileSkill":
        content = path.read_text(encoding="utf-8")
        lines   = content.split("\n")
        name    = path.stem
        description = ""
        category    = "general"
        enabled     = True
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
        obj = cls(name, description, prompt_addition, category, enabled)
        obj.file_path = path
        return obj


def load_file_skills() -> list[FileSkill]:
    skills = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        try:
            skills.append(FileSkill.load(path))
        except Exception:
            pass
    return skills


def get_file_skills_prompt(skills: list[FileSkill]) -> str:
    enabled = [s for s in skills if s.enabled]
    if not enabled:
        return ""
    parts = ["## Active Skills (file-based)"]
    for s in enabled:
        parts.append(f"\n### {s.name} ({s.category})\n{s.prompt_addition}")
    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# Reset marker
# ═════════════════════════════════════════════════════════════════════════════

def _is_fresh_install() -> bool:
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
# ═════════════════════════════════════════════════════════════════════════════

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
            {"name": "service_key",  "label": "Azure API key",      "secret": True,  "default": ""},
            {"name": "host_address", "label": "Azure endpoint URL",  "secret": False, "default": ""},
            {"name": "api_version",  "label": "API version",         "secret": False, "default": "2024-02-01"},
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
            {"name": "host_address", "label": "Server address",      "secret": False, "default": "http://localhost:3000"},
            {"name": "service_key",  "label": "API key (optional)", "secret": True,  "default": ""},
        ],
    },
    "lollms_webui": {
        "label": "LoLLMs WebUI",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "WebUI address",          "secret": False, "default": "http://localhost:9600"},
            {"name": "service_key",  "label": "Service key (optional)", "secret": True,  "default": ""},
        ],
    },
    "llama_cpp_server": {
        "label": "llama.cpp server (local)",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "Server address", "secret": False, "default": "http://localhost:8080"},
        ],
    },
    "pythonllamacpp": {
        "label": "llama-cpp-python (local GGUF)",
        "needs_model_list": False,
        "fields": [
            {"name": "model_path",   "label": "Path to .gguf file",       "secret": False, "default": ""},
            {"name": "n_gpu_layers", "label": "GPU layers (-1 = all)",     "secret": False, "default": "-1"},
            {"name": "n_ctx",        "label": "Context size",              "secret": False, "default": "4096"},
            {"name": "chat_format",  "label": "Chat format (e.g. chatml)", "secret": False, "default": "chatml"},
        ],
    },
    "transformers": {
        "label": "Hugging Face Transformers (local)",
        "needs_model_list": False,
        "fields": [
            {"name": "model_path", "label": "Model name or local path", "secret": False, "default": ""},
            {"name": "device",     "label": "Device (cpu/cuda/mps)",    "secret": False, "default": "cpu"},
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
    "litellm": {
        "label": "LiteLLM proxy",
        "needs_model_list": True,
        "fields": [
            {"name": "host_address", "label": "LiteLLM proxy address", "secret": False, "default": "http://localhost:4000"},
            {"name": "service_key",  "label": "API key (optional)",    "secret": True,  "default": ""},
        ],
    },
}

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
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css",
    "bash": ".sh", "shell": ".sh",
    "rust": ".rs", "go": ".go",
    "c": ".c", "cpp": ".cpp", "c++": ".cpp",
    "java": ".java", "kotlin": ".kt",
    "json": ".json", "yaml": ".yml", "toml": ".toml",
    "markdown": ".md", "md": ".md", "sql": ".sql",
}
CODE_TYPES  = {ArtefactType.CODE, ArtefactType.DOCUMENT, ArtefactType.FILE}
_SENSITIVE  = ("key", "secret", "token", "password", "passwd", "credential")

# ─────────────────────────────────────────────────────────────────────────────
# Intent detection — pre-injection of task-specific reminders
#
# When the user's message implies an action that requires specific tools, we
# prepend a concise reminder to user_message before calling chat().  This is
# not a workaround — it's the correct way to steer a stateless LLM toward the
# right tool sequence when the conversation context alone is ambiguous.
#
# Intent classes and their canonical tool sequences:
#
#  EXECUTE  — user wants code to run
#             → run_code(filename, capture_seconds, wait_for_exit)
#
#  TEST_APP — user wants to verify a running service
#             → list_running_processes → (run_code if not running)
#               → wait_for_server → http_get → stop_process
#
#  COMMIT   — user wants files written to disk
#             → run_code / artefacts → /commit
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_EXECUTE = [
    r'\brun\s+(?:the\s+)?(?:server|app|application|code|script|main)',
    r'\bstart\s+(?:the\s+)?(?:server|app|application)',
    r'\bexecute\s+(?:the\s+)?(?:code|script|file)',
    r'\blaunch\s+(?:the\s+)?(?:server|app)',
    r'\bbuild\s+(?:and\s+(?:run|test)|the\s+project)',
]

_INTENT_TEST_APP = [
    r'\btest\s+(?:the\s+)?(?:app|application|server|api|endpoint|code|it)',
    r'\bcheck\s+(?:the\s+)?(?:app|application|server|api|endpoint|health)',
    r'\bverify\s+(?:the\s+)?(?:app|application|server|api|endpoint|code)',
    r'\btry\s+(?:the\s+)?(?:app|application|server|api|endpoint)',
    r'\bprobe\s+(?:the\s+)?(?:server|api|endpoint)',
    r'\bping\s+(?:the\s+)?(?:server|api|endpoint)',
    r'\bis\s+(?:the\s+)?(?:server|app|application)\s+(?:running|working|up|alive)',
    r'\bdoes\s+(?:the\s+)?(?:server|app|api)\s+work',
    r'\bsend\s+(?:a\s+)?(?:request|GET|POST|curl)',
    r'\bcall\s+(?:the\s+)?(?:\w+\s+)*(?:api|endpoint)',
]

_INTENT_COMMIT = [
    r'\bcommit\b',
    r'\bsave\s+(?:the\s+)?(?:file|code|artefact)',
    r'\bwrite\s+(?:to\s+)?disk',
]

_REMINDER_EXECUTE = (
    "[SYSTEM REMINDER — TOOL USAGE REQUIRED]\n"
    "The user wants you to EXECUTE code or start a process.\n"
    "You MUST use run_code().  Do NOT describe what you would do — invoke the tool.\n"
    "For servers: run_code(filename='main.py', capture_seconds=5.0, wait_for_exit=False)\n"
    "Then use wait_for_server() before probing any endpoint.\n"
    "[END REMINDER]\n\n"
)

_REMINDER_TEST_APP = (
    "[SYSTEM REMINDER — SERVER TESTING WORKFLOW REQUIRED]\n"
    "The user wants to TEST a running application.  Follow these steps IN ORDER:\n"
    "  1. list_running_processes()  — check if the server is already running\n"
    "  2. If NOT running: run_code('main.py', capture_seconds=5.0, wait_for_exit=False)\n"
    "  3. wait_for_server('http://localhost:<port>/health')  — wait until ready\n"
    "  4. http_get('http://localhost:<port>/<endpoint>')  — test each endpoint\n"
    "  5. stop_process(<pid>)  — clean up when done\n"
    "NEVER use run_shell('curl ...') or any shell command to probe HTTP endpoints.\n"
    "NEVER skip step 1 — always verify whether the server is already running first.\n"
    "[END REMINDER]\n\n"
)

_REMINDER_COMMIT = (
    "[SYSTEM REMINDER — USE /commit TO WRITE FILES]\n"
    "If you have created artefacts, use the /commit slash command to write them to disk,\n"
    "or call write_file() for individual files.\n"
    "[END REMINDER]\n\n"
)


def _detect_intent_reminder(text: str) -> str:
    """
    Return a task-specific reminder string to prepend to the user message,
    or an empty string if no action intent is detected.

    Priority: TEST_APP > EXECUTE > COMMIT
    (TEST_APP is checked first because 'test the app' also contains 'app' which
    could weakly match EXECUTE patterns — we want the more specific class to win.)
    """
    low = text.lower()

    if any(re.search(p, low, re.IGNORECASE) for p in _INTENT_TEST_APP):
        return _REMINDER_TEST_APP

    if any(re.search(p, low, re.IGNORECASE) for p in _INTENT_EXECUTE):
        return _REMINDER_EXECUTE

    if any(re.search(p, low, re.IGNORECASE) for p in _INTENT_COMMIT):
        return _REMINDER_COMMIT

    return ""

# ─────────────────────────────────────────────────────────────────────────────
# System prompt template
# NOTE: <artefact>, <note>, <skill> instructions are injected automatically
# by the library (chat() calls _build_artefact_instructions, etc.).
# We only include project/environment context here.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
{base_personality}

## Environment Context (CRITICAL — always check before executing commands)
{env_section}
{file_skills_section}

## Project memory
{memory_section}

## Scratchpad (shared inter-agent notes)
{scratchpad_section}

## Shell dispatch — READ THIS BEFORE WRITING ANY run_shell() COMMAND
{shell_rules_section}

## Agent tools — MANDATORY USAGE
When the user asks you to perform an action (run code, execute a file,
make a git commit, etc.), you MUST use the appropriate tool.
Do NOT describe what you would do — actually invoke the tool.

Tool invocation format:
<tool_call>{{"name": "tool_name", "parameters": {{"key": "value"}}}}</tool_call>

Available tools:
- run_code(filename, args="", capture_seconds=0.0, wait_for_exit=True)
- read_process_output(pid, lines=50)   read output from a running background process
- stop_process(pid)                    stop a background process
- list_running_processes()             list all background processes started this session
- http_get(url, timeout=5.0)           HTTP GET — use instead of curl on ALL platforms
- http_post(url, body="", content_type="application/json", timeout=5.0)  HTTP POST
- wait_for_server(url, timeout=30.0, interval=1.0)  wait until a server is ready
- list_windows(pattern="")            list visible UI windows
- screenshot_window(title="", output_path="screenshot.png")
- run_shell(command)                   shell command — NEVER use curl here, use http_get()
- run_tests(path=".")
- read_file(path)
- write_file(path, content)
- list_files(path=".")
- scratchpad_read(key) / scratchpad_write(key, value)
- memory_add(text)
- git_status() | git_diff([file]) | git_add([files]) | git_commit(message)
- git_log([n]) | git_branch([name]) | git_checkout(branch)

## Workflow: testing a web server
ALWAYS follow this exact sequence — do NOT skip steps:
1. Check if server is already running:  list_running_processes()
2. If NOT running — start it:           run_code("main.py", capture_seconds=5.0, wait_for_exit=False)
3. Wait until it accepts connections:   wait_for_server("http://localhost:<port>/health")
4. Test endpoints:                      http_get("http://localhost:<port>/endpoint")
5. When done — stop it:                 stop_process(<pid>)

NEVER call http_get() or run_shell() to test an endpoint before confirming
the server is running.  NEVER use curl in run_shell() — always use http_get().

## Workflow: general coding
1. Write code as artefacts using <artefact> tags.
2. Save structured findings as notes using <note> tags.
3. Save reusable patterns as skills using <skill> tags (only when asked).
4. Call run_code() or run_tests() to verify.
5. If it fails, read the error, patch, run again. Repeat until tests pass.
"""

# ═════════════════════════════════════════════════════════════════════════════
# Shell rules — injected into system prompt so the LLM knows exactly which
# shell run_shell() dispatches to and what syntax to use.
# ═════════════════════════════════════════════════════════════════════════════

def _shell_rules() -> str:
    """
    Return a system-prompt section that tells the LLM exactly which shell
    run_shell() will use on this machine.  Called once at startup and on
    every system-prompt refresh so the model is never confused.
    """
    system = platform.system()
    # Curl ban is platform-independent — applies everywhere
    curl_ban = """\

ABSOLUTE RULE — NEVER USE curl IN run_shell():
  • On Windows, curl is aliased to Invoke-WebRequest, which does NOT accept -s, -X, or bare URLs.
  • On all platforms, use the http_get() or http_post() tools instead.
  • http_get() and http_post() work identically on Windows, Linux, and macOS with zero shell syntax.
  • Correct pattern:  http_get("http://localhost:8000/health")
  • Wrong pattern:    run_shell("curl -s http://localhost:8000/health")   ← NEVER DO THIS"""

    if system == "Windows":
        pwsh = shutil.which("pwsh")
        exe  = "pwsh.exe" if pwsh else "powershell.exe"
        return f"""\
SHELL: Windows PowerShell  (executable: {exe})
All run_shell() commands are executed as:  {exe} -NoProfile -NonInteractive -Command <your_command>

MANDATORY RULES FOR run_shell() ON WINDOWS:
1. Use PowerShell syntax ONLY.  cmd.exe idioms (e.g. `type`, `dir /b`, `findstr`) are NOT available.
2. Correct equivalents:
   • cat / type            →  Get-Content <file>
   • ls / dir              →  Get-ChildItem  or  dir  (dir is a PS alias for Get-ChildItem)
   • grep <pat> <file>     →  Select-String -Pattern '<pat>' -Path <file>
   • echo $VAR             →  Write-Output $env:VAR   or   $env:VAR
   • export VAR=val        →  $env:VAR = 'val'   (session-local only)
   • touch file.txt        →  New-Item -ItemType File file.txt
   • rm -rf dir            →  Remove-Item -Recurse -Force dir
   • mkdir -p a/b/c        →  New-Item -ItemType Directory -Force a/b/c
   • tail -n 20 file       →  Get-Content file -Tail 20
   • head -n 5 file        →  Get-Content file -TotalCount 5
   • wc -l file            →  (Get-Content file).Count
   • which cmd             →  Get-Command cmd | Select-Object -ExpandProperty Source
   • ps aux                →  Get-Process
   • kill <pid>            →  Stop-Process -Id <pid>
3. Pipeline objects, not text:  PowerShell pipelines pass objects, not raw strings.
4. String quoting: use single quotes for literals ('value'), double quotes only when you need variable expansion.
5. Multi-line commands: use backtick (`) as line continuation, or semicolons (;) on one line.
6. NEVER mix bash syntax into a run_shell() call.  It will fail silently or produce garbage output.
{curl_ban}"""
    elif system == "Darwin":
        bash = shutil.which("bash") or "/bin/bash"
        return f"""\
SHELL: bash  (executable: {bash})
All run_shell() commands are executed as:  bash -c <your_command>
Use standard bash/POSIX syntax.  macOS ships zsh as default login shell, but run_shell always uses bash.
{curl_ban}"""
    else:
        bash  = shutil.which("bash") or "/bin/sh"
        sname = "bash" if shutil.which("bash") else "sh"
        return f"""\
SHELL: {sname}  (executable: {bash})
All run_shell() commands are executed as:  {bash} -c <your_command>
Use standard bash/POSIX syntax.
{curl_ban}"""


# ═════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mask(key: str, val: str) -> str:
    return "****" if any(w in key.lower() for w in _SENSITIVE) else val


def _q_style():
    return QStyle([
        ("qmark",       "fg:#5f87ff bold"),
        ("question",    "bold"),
        ("answer",      "fg:#5fffff bold"),
        ("pointer",     "fg:#5f87ff bold"),
        ("highlighted", "fg:#5f87ff bold"),
        ("selected",    "fg:#5fffff"),
        ("separator",   "fg:#5f87ff"),
        ("instruction", "fg:#858585"),
    ])


def _ask_select(message: str, choices: list[str], default: str = None) -> str:
    if _HAS_QUESTIONARY:
        r = questionary.select(message, choices=choices, default=default, style=_q_style()).ask()
        return r or (default or choices[0])
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
        console.print(f"  [red]Invalid.[/red]")


def _ask_text(message: str, default: str = "") -> str:
    if _HAS_QUESTIONARY:
        r = questionary.text(message, default=default, style=_q_style()).ask()
        return r if r is not None else default
    display = f"{message} [dim](default: {default})[/dim]" if default else message
    r = Prompt.ask(display, default=default, show_default=False)
    return r.strip() if r else default


def _ask_password(message: str) -> str:
    if _HAS_QUESTIONARY:
        r = questionary.password(message, style=_q_style()).ask()
        return r or ""
    return Prompt.ask(message, password=True)


def _ask_confirm(message: str, default: bool = False) -> bool:
    if _HAS_QUESTIONARY:
        r = questionary.confirm(message, default=default, style=_q_style()).ask()
        return r if r is not None else default
    return Confirm.ask(message, default=default)


# ═════════════════════════════════════════════════════════════════════════════
# Config  (~/.lollms/coding_cli/config.yaml)
# ═════════════════════════════════════════════════════════════════════════════

def _default_config() -> dict:
    return {
        "binding_name":           "lollms",
        "binding_config":         {},
        "projects_root":          str(_documents_dir() / "lollms_projects"),
        "python_executable":      sys.executable,
        "git_auto_commit":        False,
        "shell_timeout":          30,
        "max_output_chars":       8000,
        "show_thinking":          True,
        # ── new library feature flags ────────────────────────────────────────
        "enable_notes":           True,   # <note> → NOTE artefacts
        "enable_skills":          True,   # <skill> → SKILL artefacts
        "enable_inline_widgets":  False,  # <lollms_inline> (needs browser renderer)
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
    try:
        lc_tmp = LollmsClient(llm_binding_name=binding_name, llm_binding_config=bc)
        raw = lc_tmp.list_models()
        if not isinstance(raw, list) or not raw:
            return []
        models: list[dict] = []
        for m in raw:
            if isinstance(m, dict):
                name = (m.get("model_name") or m.get("name") or m.get("id") or "").strip()
                if not name:
                    continue
                models.append({
                    "model_name":     name,
                    "owned_by":       m.get("owned_by", ""),
                    "context_length": m.get("context_length"),
                    "max_generation": m.get("max_generation"),
                })
            elif isinstance(m, str) and m.strip():
                models.append({"model_name": m.strip(), "owned_by": "",
                               "context_length": None, "max_generation": None})
        return models
    except Exception:
        return []


def _show_models_table(models: list[dict]) -> None:
    t = Table(show_header=True, header_style="bold cyan", expand=False, show_lines=False)
    t.add_column("#",         style="dim",        justify="right", width=4)
    t.add_column("Model",     style="bold white",  no_wrap=True)
    t.add_column("Owner",     style="cyan",        no_wrap=True)
    t.add_column("Context",   style="green",       justify="right")
    t.add_column("Max gen",   style="yellow",      justify="right")
    for i, m in enumerate(models, 1):
        t.add_row(str(i), m["model_name"], m["owned_by"] or "—",
                  str(m["context_length"]) if m["context_length"] else "—",
                  str(m["max_generation"]) if m["max_generation"] else "—")
    console.print(t)


def _pick_model(models: list[dict], current_model: str) -> str:
    _show_models_table(models)
    names = [m["model_name"] for m in models]
    default_name = current_model if current_model in names else (names[0] if names else "")
    hint = f" [dim](Enter for: {default_name})[/dim]" if default_name else ""
    if _HAS_QUESTIONARY:
        r = questionary.autocomplete(
            f"Select model{hint}", choices=names, default=default_name, style=_q_style(),
            validate=lambda v: v in names or f"'{v}' not in list",
        ).ask()
        return r or current_model or default_name
    while True:
        raw = Prompt.ask(f"  Enter number or name{hint}", default=default_name,
                         show_default=False).strip()
        if not raw and default_name:
            return default_name
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in names:
            return raw
        console.print("  [red]Not found.[/red]")


def run_setup_wizard(existing: Optional[dict] = None, first_run: bool = False) -> dict:
    cfg = existing or _default_config()
    console.print(Panel(
        "[bold cyan]lollms Coding Assistant – Setup Wizard[/bold cyan]\n"
        "[dim]Config saved to [bold]~/.lollms/coding_cli/config.yaml[/bold][/dim]",
        border_style="cyan",
    ))

    # Step 1 — binding
    console.print("\n[bold]── Step 1: LLM Binding ──[/bold]")
    choices = [f"{n}  —  {i['label']}" for n, i in BINDING_CATALOG.items()]
    cur = cfg.get("binding_name", "lollms")
    default_c = next((c for c in choices if c.startswith(cur + " ")), choices[0])
    chosen = _ask_select("Select LLM binding:", choices, default=default_c)
    binding_name = chosen.split("  —  ")[0].strip()
    cfg["binding_name"] = binding_name
    entry = BINDING_CATALOG.get(binding_name, {})

    # Step 2 — fields
    console.print(f"\n[bold]── Step 2: Configure '{binding_name}' ──[/bold]")
    bc = dict(cfg.get("binding_config") or {})
    for field in entry.get("fields", []):
        fname, flabel = field["name"], field["label"]
        is_secret = field.get("secret", False)
        saved = bc.get(fname, "")
        fdefault = str(saved) if saved else str(field.get("default", ""))
        if is_secret:
            hint = " [dim](saved)[/dim]" if saved else f" [dim](default: {field.get('default','empty')})[/dim]"
            console.print(f"  {flabel}{hint}")
            val = _ask_password("    Value")
            if not val and saved:
                val = saved
        else:
            val = _ask_text(f"  {flabel}", default=fdefault)
        if val:
            bc[fname] = val
        elif fname in bc and not val:
            del bc[fname]
    cfg["binding_config"] = bc

    # Step 3 — model
    console.print(f"\n[bold]── Step 3: Model ──[/bold]")
    cur_model = bc.get("model_name", "")
    if entry.get("needs_model_list"):
        console.print("  [dim]Connecting…[/dim]")
        live = _try_list_models(binding_name, bc)
        if live:
            chosen_model = _pick_model(live, cur_model)
        else:
            console.print("  [yellow]⚠  Cannot reach server — using defaults.[/yellow]")
            fb = [{"model_name": n, "owned_by": "", "context_length": None, "max_generation": None}
                  for n in BINDING_DEFAULT_MODELS.get(binding_name, [])]
            chosen_model = _pick_model(fb, cur_model) if fb else _ask_text("  Model name", default=cur_model)
    else:
        fb = [{"model_name": n, "owned_by": "", "context_length": None, "max_generation": None}
              for n in BINDING_DEFAULT_MODELS.get(binding_name, [])]
        chosen_model = _pick_model(fb, cur_model) if fb else _ask_text("  Model name", default=cur_model)
    if chosen_model:
        cfg["binding_config"]["model_name"] = chosen_model

    # Step 4 — paths
    console.print("\n[bold]── Step 4: Paths ──[/bold]")
    cfg["projects_root"] = _ask_text("  Projects root folder",
                                     default=cfg.get("projects_root", str(_documents_dir() / "lollms_projects")))

    # Step 5 — runtime
    console.print("\n[bold]── Step 5: Runtime ──[/bold]")
    cfg["python_executable"] = _ask_text("  Python executable", default=cfg.get("python_executable", sys.executable))
    cfg["shell_timeout"]     = int(_ask_text("  Shell timeout (s)", default=str(cfg.get("shell_timeout", 30))))
    cfg["max_output_chars"]  = int(_ask_text("  Max output chars",  default=str(cfg.get("max_output_chars", 8000))))

    # Step 6 — git
    console.print("\n[bold]── Step 6: Git ──[/bold]")
    cfg["git_auto_commit"] = _ask_confirm("  Auto git-commit after /commit?",
                                          default=cfg.get("git_auto_commit", False))

    # Step 7 — AI feature flags
    console.print("\n[bold]── Step 7: AI Features ──[/bold]")
    cfg["enable_notes"] = _ask_confirm(
        "  Enable <note> tags (LLM saves structured notes)?",
        default=cfg.get("enable_notes", True))
    cfg["enable_skills"] = _ask_confirm(
        "  Enable <skill> tags (LLM saves reusable patterns)?",
        default=cfg.get("enable_skills", True))
    cfg["enable_inline_widgets"] = _ask_confirm(
        "  Enable <lollms_inline> widgets (requires browser renderer)?",
        default=cfg.get("enable_inline_widgets", False))

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
# Scratchpad
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
    def _ts(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if sub == "set":
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[red]Usage: /scratch set <key> <value>[/red]"); return scratch
        scratch[parts[0]] = {"value": parts[1], "updated_at": _ts()}
        save_scratchpad(scratch)
        console.print(f"  [green]✓[/green]  [bold]{parts[0]}[/bold] = {parts[1]!r}")
    elif sub == "get":
        k = arg.strip()
        if not k: console.print("[red]Usage: /scratch get <key>[/red]")
        elif k not in scratch: console.print(f"  [dim]Key '{k}' not found.[/dim]")
        else:
            e = scratch[k]
            console.print(f"  [bold cyan]{k}[/bold cyan]  [dim]{e.get('updated_at','')}[/dim]\n  {e['value']}")
    elif sub == "ls":
        if not scratch: console.print("[dim]Scratchpad is empty.[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("Key",     style="bold white", no_wrap=True)
            t.add_column("Value",   style="yellow")
            t.add_column("Updated", style="dim")
            for k, v in sorted(scratch.items()):
                val = str(v["value"])[:80] + ("…" if len(str(v["value"])) > 80 else "")
                t.add_row(k, val, v.get("updated_at", ""))
            console.print(t)
    elif sub == "del":
        k = arg.strip()
        if not k: console.print("[red]Usage: /scratch del <key>[/red]")
        elif k not in scratch: console.print(f"  [yellow]Key '{k}' not found.[/yellow]")
        else:
            del scratch[k]; save_scratchpad(scratch)
            console.print(f"  [yellow]Deleted '{k}'.[/yellow]")
    elif sub == "clear":
        if _ask_confirm("  Clear the entire scratchpad?", default=False):
            scratch.clear(); save_scratchpad(scratch)
            console.print("  [yellow]Scratchpad cleared.[/yellow]")
    else:
        console.print("[red]Unknown /scratch sub-command.[/red]  set | get | ls | del | clear")
    return scratch


# ═════════════════════════════════════════════════════════════════════════════
# Per-project long-term memory
# ═════════════════════════════════════════════════════════════════════════════

def _memory_path(folder: Path) -> Path:
    mp = folder / ".lollms"; mp.mkdir(exist_ok=True); return mp / "memory.yaml"

def load_memory(folder: Path) -> list:
    mp = _memory_path(folder)
    if mp.exists():
        try: return yaml.safe_load(mp.read_text(encoding="utf-8")) or []
        except Exception: pass
    return []

def save_memory(folder: Path, entries: list) -> None:
    _memory_path(folder).write_text(
        yaml.dump(entries, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

def _memory_summary(entries: list, max_chars: int = 3000) -> str:
    if not entries: return "(no project memory yet)"
    lines = [f"[{e.get('created_at','')[:10]}] {e.get('text','').strip()}" for e in entries]
    combined = "\n".join(lines)
    return (combined[:max_chars] + "\n… [truncated]") if len(combined) > max_chars else combined

def cmd_memory(sub: str, arg: str, folder: Path, mem: list) -> list:
    def _ts(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if sub == "add":
        text = arg.strip()
        if not text: console.print("[red]Usage: /memory add <text>[/red]")
        else:
            e = {"id": len(mem)+1, "text": text, "created_at": _ts()}
            mem.append(e); save_memory(folder, mem)
            console.print(f"  [green]✓[/green]  Memory #{e['id']} added.")
    elif sub == "ls":
        if not mem: console.print("[dim]No memory entries yet.[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("ID",  style="dim", justify="right", width=4)
            t.add_column("Date",style="dim", width=12)
            t.add_column("Text",style="white")
            for e in mem:
                txt = e.get("text","")
                t.add_row(str(e.get("id","?")), e.get("created_at","")[:10],
                          txt[:100]+("…" if len(txt)>100 else ""))
            console.print(t)
    elif sub == "del":
        try: eid = int(arg.strip())
        except ValueError: console.print("[red]Usage: /memory del <id>[/red]"); return mem
        old = len(mem); mem = [e for e in mem if e.get("id") != eid]
        if len(mem) < old: save_memory(folder, mem); console.print(f"  [yellow]Memory #{eid} deleted.[/yellow]")
        else: console.print(f"  [red]Entry #{eid} not found.[/red]")
    elif sub == "clear":
        if _ask_confirm("  Clear all project memory?", default=False):
            mem.clear(); save_memory(folder, mem); console.print("  [yellow]Cleared.[/yellow]")
    elif sub == "search":
        q = arg.strip().lower()
        if not q: console.print("[red]Usage: /memory search <query>[/red]")
        else:
            hits = [e for e in mem if q in e.get("text","").lower()]
            if not hits: console.print(f"  [dim]No entries matching '{q}'.[/dim]")
            else:
                t = Table(show_header=True, header_style="bold cyan", expand=False)
                t.add_column("ID",  style="dim", justify="right", width=4)
                t.add_column("Date",style="dim", width=12)
                t.add_column("Text",style="white")
                for e in hits:
                    txt = e.get("text","")
                    t.add_row(str(e.get("id","?")), e.get("created_at","")[:10],
                              txt[:100]+("…" if len(txt)>100 else ""))
                console.print(t); console.print(f"  [dim]{len(hits)} result(s)[/dim]")
    else:
        console.print("[red]Unknown /memory sub-command.[/red]  add | ls | del | clear | search")
    return mem


# ═════════════════════════════════════════════════════════════════════════════
# Projects
# ═════════════════════════════════════════════════════════════════════════════

def load_projects() -> dict:
    if PROJECTS_FILE.exists():
        try: return yaml.safe_load(PROJECTS_FILE.read_text(encoding="utf-8")) or {}
        except Exception: pass
    return {}

def save_projects(projects: dict) -> None:
    PROJECTS_FILE.write_text(
        yaml.dump(projects, allow_unicode=True, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )

# ═════════════════════════════════════════════════════════════════════════════
# Subprocess helper
# ═════════════════════════════════════════════════════════════════════════════

def _run_proc(args, cwd: Path, timeout: int, max_chars: int, shell: bool = False) -> dict:
    try:
        r = subprocess.run(args, capture_output=True, text=True, cwd=str(cwd),
                           timeout=timeout, errors="replace", shell=shell)
        combined = (r.stdout + r.stderr).strip()
        if len(combined) > max_chars:
            combined = combined[:max_chars] + f"\n… [truncated at {max_chars}]"
        return {"success": r.returncode == 0, "returncode": r.returncode, "output": combined}
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "output": f"Timed out after {timeout}s"}
    except FileNotFoundError as e:
        return {"success": False, "returncode": -1, "output": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# Agent tool factory
# ═════════════════════════════════════════════════════════════════════════════

def build_agent_tools(project_root: Path, cfg: dict, scratch: dict, mem: list) -> dict:
    py    = cfg.get("python_executable", sys.executable)
    tout  = cfg.get("shell_timeout", 30)
    maxch = cfg.get("max_output_chars", 8000)
    project_root.mkdir(parents=True, exist_ok=True)

    def _safe(p: str) -> Path:
        resolved = (project_root / p).resolve()
        try: resolved.relative_to(project_root.resolve())
        except ValueError: raise PermissionError(f"Path '{p}' escapes the project folder.")
        return resolved

    # ── Process registry ──────────────────────────────────────────────────────
    _procs: dict[int, dict] = {}

    def _reg(pid, proc, cmd, filename, log_path, start_time):
        _procs[pid] = {"proc": proc, "cmd": cmd, "filename": filename,
                       "log_path": log_path, "start_time": start_time}

    def _drop(pid):
        _procs.pop(pid, None)

    # ── run_code ─────────────────────────────────────────────────────────────
    def run_code(filename: str, args: str = "",
                 capture_seconds: float = 0.0,
                 wait_for_exit: bool = True,
                 timeout: Optional[int] = None) -> dict:
        """
        Execute code with flexible output handling.

        Synchronous (wait_for_exit=True, capture_seconds=0):
          Blocks until process exits; returns full output.

        Async+capture (capture_seconds > 0, wait_for_exit=False):
          Starts process, reads for N seconds, returns preview + PID.
          Use for servers: capture_seconds=5.0, wait_for_exit=False.

        Fire-and-forget (wait_for_exit=False, capture_seconds=0):
          Returns immediately with PID. Use read_process_output() to poll.
        """
        fp = _safe(filename)
        if not fp.exists():
            return {"success": False, "output": f"File not found: {filename}"}
        cmd = [py, str(fp)] + (args.split() if args.strip() else [])
        effective_timeout = timeout or tout
        is_async = capture_seconds > 0 or not wait_for_exit

        if not is_async:
            r = _run_proc(cmd, project_root, effective_timeout, maxch)
            console.print(Panel(r["output"] or "(no output)",
                                title=f"[cyan]run {filename}[/cyan]",
                                border_style="green" if r["success"] else "red"))
            return r

        # Async: start with pipes
        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = project_root / f"{Path(filename).stem}_{ts}.log"
        try:
            kwargs = dict(cwd=str(project_root), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            if platform.system() == "Windows":
                proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **kwargs)
            else:
                proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)

            pid = proc.pid
            t0  = time.time()
            _reg(pid, proc, cmd, filename, log_file, t0)
            log_file.write_text(
                f"# Started: {datetime.now(timezone.utc).isoformat()}\n# PID: {pid}\n"
                f"# Command: {' '.join(cmd)}\n\n", encoding="utf-8")
            captured: list[str] = []

            def _read():
                with open(log_file, "a", encoding="utf-8") as lf:
                    try:
                        for line in proc.stdout:
                            s = line.rstrip("\n"); captured.append(s)
                            lf.write(s + "\n"); lf.flush()
                    except Exception: pass

            threading.Thread(target=_read, daemon=True).start()

            if capture_seconds > 0:
                console.print(Panel(f"Capturing for {capture_seconds}s…  (PID {pid})",
                                    title=f"[cyan]{filename}[/cyan]", border_style="yellow"))
                try:
                    proc.wait(timeout=capture_seconds); exit_code = proc.returncode; running = False
                except subprocess.TimeoutExpired:
                    exit_code = None; running = True
                time.sleep(0.1)
                out = "\n".join(captured[-500:])
                if len(out) > maxch: out = "…[truncated]…\n" + out[-maxch:]
                status = "running" if running else f"exited ({exit_code})"
                console.print(Panel(f"Status: {status}\n{len(captured)} lines\nLog: {log_file.name}",
                                    title=f"[{'green' if exit_code in (None,0) else 'red'}]{filename}[/]",
                                    border_style="green" if exit_code in (None,0) else "red"))
                if not running: _drop(pid)
                return {"success": exit_code in (None,0), "pid": pid, "status": status,
                        "output_preview": out, "log_file": str(log_file)}
            else:  # fire-and-forget
                time.sleep(0.5)
                if proc.poll() is not None:
                    _drop(pid)
                    return {"success": False, "pid": pid,
                            "status": f"crashed ({proc.returncode})",
                            "output": "\n".join(captured)[:maxch]}
                console.print(Panel(f"PID: {pid}\nLog: {log_file.name}",
                                    title=f"[green]✓ {filename} started[/green]", border_style="green"))
                return {"success": True, "pid": pid, "status": "running", "log_file": str(log_file)}

        except Exception as e:
            trace_exception(e)
            return {"success": False, "output": f"Failed to start: {e}"}

    # ── read_process_output ───────────────────────────────────────────────────
    def read_process_output(pid: int, lines: int = 50, since_start: bool = False) -> dict:
        """Read recent output from a running background process."""
        if pid not in _procs:
            logs = list(project_root.glob(f"*{pid}*.log"))
            if logs:
                lf = max(logs, key=lambda p: p.stat().st_mtime)
                content = lf.read_text(encoding="utf-8", errors="replace")
                all_lines = content.split("\n")
                selected = all_lines if since_start else all_lines[-lines:]
                out = "\n".join(selected)
                return {"success": True, "pid": pid, "status": "unknown (not in registry)",
                        "output": out[-maxch:], "total_lines": len(all_lines), "log_file": str(lf)}
            return {"success": False, "output": f"Process {pid} not found"}

        info = _procs[pid]; proc = info["proc"]; lf = info["log_path"]
        exit_code = proc.poll(); is_running = exit_code is None
        out = ""
        if lf.exists():
            content = lf.read_text(encoding="utf-8", errors="replace")
            all_lines = content.split("\n")
            selected = all_lines if since_start else all_lines[-lines:]
            out = "\n".join(selected)
            if len(out) > maxch: out = "…[truncated]…\n" + out[-maxch:]
        else:
            all_lines = []
        runtime = time.time() - info["start_time"]
        status  = "running" if is_running else f"exited ({exit_code})"
        console.print(Panel(f"Status: {status} | Runtime: {runtime:.1f}s",
                            title=f"[cyan]Process {pid}[/cyan]",
                            border_style="green" if is_running else "dim"))
        if not is_running: _drop(pid)
        return {"success": True, "pid": pid, "status": status, "output": out,
                "runtime_seconds": round(runtime, 1), "total_lines": len(all_lines),
                "log_file": str(lf)}

    # ── stop_process ─────────────────────────────────────────────────────────
    def stop_process(pid: int, graceful: bool = True, grace_period: float = 5.0) -> dict:
        """Stop a background process."""
        import signal as _sig
        if pid not in _procs:
            try:
                os.kill(pid, _sig.SIGTERM if graceful else _sig.SIGKILL)
                return {"success": True, "output": f"Signal sent to PID {pid}"}
            except ProcessLookupError:
                return {"success": False, "output": f"PID {pid} not found"}
            except Exception as e:
                return {"success": False, "output": str(e)}

        info = _procs[pid]; proc = info["proc"]
        console.print(Panel(f"Stopping PID {pid}…", title="[yellow]stop[/yellow]", border_style="yellow"))
        try:
            if platform.system() == "Windows":
                try:
                    proc.send_signal(_sig.CTRL_BREAK_EVENT); proc.wait(timeout=grace_period)
                except Exception:
                    proc.terminate()
                    try: proc.wait(timeout=2)
                    except Exception: proc.kill()
            else:
                proc.terminate()
                try: proc.wait(timeout=grace_period)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=5)
            final = read_process_output(pid, lines=100)
            _drop(pid)
            console.print(Panel(f"Stopped. Exit: {proc.returncode}",
                                title="[green]✓ Stopped[/green]", border_style="green"))
            return {"success": True, "pid": pid, "exit_code": proc.returncode,
                    "final_output": final.get("output","")[:maxch]}
        except Exception as e:
            _drop(pid); return {"success": False, "output": f"Error: {e}"}

    # ── list_running_processes ────────────────────────────────────────────────
    def list_running_processes() -> dict:
        """List all background processes started this session."""
        if not _procs:
            return {"success": True, "processes": [], "count": 0}
        now = time.time(); result = []
        for pid, info in list(_procs.items()):
            ec = info["proc"].poll(); running = ec is None
            result.append({"pid": pid, "filename": info["filename"],
                           "runtime_seconds": round(now - info["start_time"], 1),
                           "status": "running" if running else f"exited ({ec})",
                           "log_file": str(info["log_path"])})
            if not running: _drop(pid)
        return {"success": True, "processes": result, "count": len(result)}

    # ── list_windows ─────────────────────────────────────────────────────────
    def list_windows(pattern: str = "") -> dict:
        """List visible windows — useful for finding UI apps the LLM started."""
        system = platform.system()
        try:
            if system == "Windows":
                ps = f'''
                Add-Type @"
                using System; using System.Runtime.InteropServices;
                public class W {{
                    [DllImport("user32.dll")] public static extern bool EnumWindows(IntPtr cb, IntPtr lp);
                    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder t, int c);
                    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
                }}
"@
                $out=@(); [W]::EnumWindows({{
                    param($h,$l); if([W]::IsWindowVisible($h)){{
                        $t=New-Object System.Text.StringBuilder 256
                        [W]::GetWindowText($h,$t,256)|Out-Null; $s=$t.ToString()
                        if($s -and $s -match '{pattern}'){{$out+=$s}}
                    }}; $true
                }},0)|Out-Null; $out
                '''
                r = subprocess.run(["powershell","-Command",ps], capture_output=True, text=True, timeout=10)
                titles = [l.strip() for l in r.stdout.split("\n") if l.strip()]
            elif system == "Darwin":
                r = subprocess.run(["osascript","-e",
                    'tell application "System Events" to get name of every window of every process'],
                    capture_output=True, text=True, timeout=10)
                titles = [t.strip() for t in r.stdout.split(",") if t.strip()]
                if pattern: titles = [t for t in titles if pattern.lower() in t.lower()]
            else:
                for cmd in [["xdotool","search","--onlyvisible","--name",pattern or ".*","getwindowname"],
                             ["wmctrl","-l"]]:
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        titles = [l.strip() for l in r.stdout.split("\n") if l.strip()]
                        if cmd[0]=="wmctrl" and pattern:
                            titles = [t for t in titles if pattern.lower() in t.lower()]
                        break
                    except FileNotFoundError:
                        continue
                else:
                    titles = []
            return {"success": True, "platform": system, "windows": titles[:50], "count": len(titles)}
        except Exception as e:
            return {"success": False, "output": f"Could not list windows: {e}"}

    # ── screenshot_window ─────────────────────────────────────────────────────
    def screenshot_window(title_pattern: str = "", output_path: str = "screenshot.png") -> dict:
        """Capture a screenshot of a window or the full screen."""
        system = platform.system(); dest = _safe(output_path)
        try:
            if system == "Windows":
                ps = f'''
                Add-Type -AssemblyName System.Windows.Forms,System.Drawing
                $b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bm=New-Object System.Drawing.Bitmap $b.Width,$b.Height
                $g=[System.Drawing.Graphics]::FromImage($bm)
                $g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size)
                $bm.Save("{dest}")
                '''
                subprocess.run(["powershell","-Command",ps], check=True, timeout=15)
            elif system == "Darwin":
                subprocess.run(["screencapture",str(dest)], check=True, timeout=10)
            else:
                for tool in [["gnome-screenshot","-f",str(dest)],
                              ["import","-window","root",str(dest)],
                              ["ffmpeg","-f","x11grab","-i",":0","-frames:v","1",str(dest)]]:
                    try: subprocess.run(tool, check=True, timeout=10); break
                    except (FileNotFoundError, subprocess.CalledProcessError): continue
                else: raise RuntimeError("No screenshot tool found")
            img_bytes = dest.read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            console.print(Panel(f"Saved: {dest}\n{len(img_bytes):,} bytes",
                                title="[green]✓ Screenshot[/green]", border_style="green"))
            return {"success": True, "path": str(dest), "size_bytes": len(img_bytes),
                    "base64_image": b64[:200]+"…[use read_file for full]"}
        except Exception as e:
            return {"success": False, "output": f"Screenshot failed: {e}"}

    # ── file + shell ─────────────────────────────────────────────────────────
    def run_shell(command: str) -> dict:
        # Dispatch to the correct shell for the current platform.
        #
        # Windows:  powershell.exe -NoProfile -NonInteractive -Command <cmd>
        #   • Gives the LLM a real PowerShell session so all PS cmdlets work
        #     (ConvertFrom-Json, Get-Content, Select-String, etc.)
        #   • -NoProfile: faster startup, no user profile side-effects
        #   • -NonInteractive: no prompts that would hang the subprocess
        #
        # Unix/Mac: bash -c <cmd>   (fallback: sh -c if bash not found)
        if platform.system() == "Windows":
            shell_label = "PS>"
            args = ["powershell.exe", "-NoProfile", "-NonInteractive",
                    "-Command", command]
        else:
            shell_label = "$"
            bash = shutil.which("bash") or "sh"
            args = [bash, "-c", command]
        r = _run_proc(args, project_root, tout, maxch, shell=False)
        console.print(Panel(r["output"] or "(no output)",
                            title=f"[cyan]{shell_label} {command[:70]}[/cyan]",
                            border_style="green" if r["success"] else "red"))
        return r

    def run_tests(path: str = ".") -> dict:
        tgt = _safe(path)
        if not tgt.exists(): return {"success": False, "output": f"Not found: {path}"}
        r = _run_proc([py,"-m","pytest",str(tgt),"-v","--tb=short","--no-header"],
                      project_root, tout*2, maxch)
        console.print(Panel(r["output"] or "(no output)", title="[cyan]pytest[/cyan]",
                            border_style="green" if r["success"] else "red"))
        return r

    def read_file(path: str) -> dict:
        try:
            c = _safe(path).read_text(encoding="utf-8", errors="replace")
            if len(c) > maxch: c = c[:maxch]+"\n… [truncated]"
            return {"success": True, "content": c, "path": path}
        except Exception as e: return {"success": False, "output": str(e)}

    def write_file(path: str, content: str) -> dict:
        try:
            fp = _safe(path); fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            console.print(f"  [green]✓[/green]  wrote [bold]{path}[/bold]  ({len(content)} chars)")
            return {"success": True, "path": path, "bytes": len(content.encode())}
        except Exception as e: return {"success": False, "output": str(e)}

    def list_files(path: str = ".") -> dict:
        try:
            tgt = _safe(path)
            entries = sorted(tgt.iterdir(), key=lambda e: (e.is_file(), e.name))
            lines = [
                f"{'📄' if e.is_file() else '📁'}  {e.name}"
                + (f"  {e.stat().st_size:>9} B" if e.is_file() else "")
                for e in entries
            ]
            return {"success": True, "listing": "\n".join(lines),
                    "files": [e.name for e in entries]}
        except Exception as e: return {"success": False, "output": str(e)}

    # ── scratchpad + memory tools ─────────────────────────────────────────────
    def scratchpad_read(key: str) -> dict:
        if key not in scratch: return {"success": False, "output": f"Key '{key}' not found."}
        return {"success": True, "key": key, "value": scratch[key]["value"]}

    def scratchpad_write(key: str, value: str) -> dict:
        scratch[key] = {"value": value,
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_scratchpad(scratch)
        console.print(f"  [green]✓[/green]  scratch[{key!r}] = {value!r}")
        return {"success": True, "key": key}

    def memory_add(text: str) -> dict:
        e = {"id": len(mem)+1, "text": text.strip(),
             "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        mem.append(e); save_memory(project_root, mem)
        console.print(f"  [green]✓[/green]  Memory #{e['id']} added.")
        return {"success": True, "id": e["id"]}

    # ── HTTP probing ──────────────────────────────────────────────────────────
    # These tools replace curl/Invoke-WebRequest for HTTP testing.
    # They use Python's stdlib urllib — zero deps, works identically on all
    # platforms (Windows/Linux/Mac) without any shell syntax concerns.

    def http_get(url: str, timeout: float = 5.0,
                 headers: Optional[dict] = None) -> dict:
        """
        Make a plain HTTP GET request and return the status code + body.

        Use this instead of run_shell('curl ...') or Invoke-WebRequest.
        Works identically on Windows, Linux, and macOS.

        Returns a dict with:
          success        bool   True if status < 400
          status_code    int    HTTP status code (or -1 on connection error)
          body           str    Response body (truncated to max_output_chars)
          content_type   str    Content-Type header value
          elapsed_ms     float  Round-trip time in milliseconds
          server_running bool   True if the server accepted the connection
          error          str    Error message on failure (absent on success)
        """
        import urllib.request
        import urllib.error
        import json as _json

        req = urllib.request.Request(url, headers=headers or {})
        t0  = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = (time.time() - t0) * 1000
                raw     = resp.read()
                try:
                    body = raw.decode("utf-8", errors="replace")
                except Exception:
                    body = repr(raw[:200])
                ct = resp.headers.get("Content-Type", "")
                # Pretty-print JSON bodies
                if "json" in ct:
                    try:
                        body = _json.dumps(_json.loads(body), indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                if len(body) > maxch:
                    body = body[:maxch] + f"\n… [truncated at {maxch} chars]"
                result = {
                    "success":        True,
                    "status_code":    resp.status,
                    "body":           body,
                    "content_type":   ct,
                    "elapsed_ms":     round(elapsed, 1),
                    "server_running": True,
                }
                border = "green"
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - t0) * 1000
            try:
                body = e.read().decode("utf-8", errors="replace")[:maxch]
            except Exception:
                body = str(e)
            result = {
                "success":        False,
                "status_code":    e.code,
                "body":           body,
                "content_type":   "",
                "elapsed_ms":     round(elapsed, 1),
                "server_running": True,   # server is up, just returned an error
                "error":          f"HTTP {e.code} {e.reason}",
            }
            border = "yellow"
        except (ConnectionRefusedError, OSError) as e:
            elapsed = (time.time() - t0) * 1000
            result = {
                "success":        False,
                "status_code":    -1,
                "body":           "",
                "content_type":   "",
                "elapsed_ms":     round(elapsed, 1),
                "server_running": False,
                "error":          f"Connection refused / server not running: {e}",
            }
            border = "red"
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            result = {
                "success":        False,
                "status_code":    -1,
                "body":           "",
                "content_type":   "",
                "elapsed_ms":     round(elapsed, 1),
                "server_running": False,
                "error":          str(e),
            }
            border = "red"

        # Display
        status_str = (str(result["status_code"])
                      if result["status_code"] > 0 else "NO RESPONSE")
        console.print(Panel(
            result.get("body") or result.get("error") or "(empty)",
            title=f"[cyan]GET {url}[/cyan]  [{status_str}]  "
                  f"[dim]{result['elapsed_ms']} ms[/dim]",
            border_style=border,
        ))
        return result

    def http_post(url: str, body: str = "", content_type: str = "application/json",
                  timeout: float = 5.0, headers: Optional[dict] = None) -> dict:
        """
        Make an HTTP POST request with a string body.

        Use this instead of curl -X POST / Invoke-WebRequest -Method POST.
        Works identically on all platforms.
        """
        import urllib.request
        import urllib.error
        import json as _json

        extra_headers = {"Content-Type": content_type}
        extra_headers.update(headers or {})
        data = body.encode("utf-8") if body else b""
        req  = urllib.request.Request(url, data=data, headers=extra_headers, method="POST")
        t0   = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = (time.time() - t0) * 1000
                raw     = resp.read().decode("utf-8", errors="replace")
                ct      = resp.headers.get("Content-Type", "")
                if "json" in ct:
                    try:
                        raw = _json.dumps(_json.loads(raw), indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                if len(raw) > maxch:
                    raw = raw[:maxch] + "\n… [truncated]"
                result = {"success": True, "status_code": resp.status,
                          "body": raw, "elapsed_ms": round(elapsed, 1),
                          "server_running": True}
                border = "green"
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - t0) * 1000
            try: body_err = e.read().decode("utf-8", errors="replace")[:maxch]
            except Exception: body_err = str(e)
            result = {"success": False, "status_code": e.code,
                      "body": body_err, "elapsed_ms": round(elapsed, 1),
                      "server_running": True, "error": f"HTTP {e.code} {e.reason}"}
            border = "yellow"
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            result = {"success": False, "status_code": -1,
                      "body": "", "elapsed_ms": round(elapsed, 1),
                      "server_running": False, "error": str(e)}
            border = "red"

        console.print(Panel(
            result.get("body") or result.get("error") or "(empty)",
            title=f"[cyan]POST {url}[/cyan]  [{result['status_code']}]  "
                  f"[dim]{result['elapsed_ms']} ms[/dim]",
            border_style=border,
        ))
        return result

    def wait_for_server(url: str, timeout: float = 30.0,
                        interval: float = 1.0) -> dict:
        """
        Poll a URL until it returns any HTTP response (even 4xx/5xx counts —
        that means the server is up).  Waits up to `timeout` seconds,
        checking every `interval` seconds.

        Use this AFTER run_code(..., wait_for_exit=False) to know when the
        server is ready before calling http_get() or http_post() to test it.

        Returns:
          ready         bool   True if server responded within timeout
          url           str    The URL polled
          elapsed_s     float  Seconds waited
          status_code   int    Last HTTP status (-1 = never responded)
          attempts      int    Number of probe attempts made
        """
        import urllib.request
        import urllib.error

        t0       = time.time()
        attempts = 0
        console.print(f"  [dim]Waiting for server at [bold]{url}[/bold] "
                      f"(timeout {timeout}s)…[/dim]")
        while True:
            attempts += 1
            elapsed = time.time() - t0
            try:
                with urllib.request.urlopen(url, timeout=min(interval, 2.0)) as resp:
                    sc = resp.status
                    console.print(f"  [green]✓ Server ready[/green]  "
                                  f"[dim]{url}  [{sc}]  {elapsed:.1f}s  "
                                  f"{attempts} attempt(s)[/dim]")
                    return {"ready": True, "url": url,
                            "elapsed_s": round(elapsed, 2),
                            "status_code": sc, "attempts": attempts}
            except urllib.error.HTTPError as e:
                # HTTP error = server is actually up
                console.print(f"  [green]✓ Server ready[/green]  "
                              f"[dim]{url}  [{e.code}]  {elapsed:.1f}s[/dim]")
                return {"ready": True, "url": url,
                        "elapsed_s": round(elapsed, 2),
                        "status_code": e.code, "attempts": attempts}
            except Exception:
                pass

            if elapsed >= timeout:
                console.print(f"  [red]✗ Server did not start within {timeout}s[/red]  "
                              f"[dim]({attempts} attempts)[/dim]")
                return {"ready": False, "url": url,
                        "elapsed_s": round(elapsed, 2),
                        "status_code": -1, "attempts": attempts}

            # Progress dots
            remaining = timeout - elapsed
            console.print(f"  [dim]  … {elapsed:.0f}s elapsed, "
                          f"{remaining:.0f}s remaining[/dim]")
            time.sleep(interval)

    # ── git ───────────────────────────────────────────────────────────────────
    def _g(*args) -> dict: return _run_proc(["git"]+list(args), project_root, tout, maxch)

    def git_status():
        r=_g("status"); console.print(Panel(r["output"],title="[cyan]git status[/cyan]",border_style="dim")); return r
    def git_diff(file=""):
        r=_g(*filter(None,["diff",file]))
        if r["output"]: console.print(Syntax(r["output"],"diff",theme="monokai"))
        return r
    def git_add(files="."):
        r=_g("add",*(files.split() if files.strip()!="." else ["."]));
        if r["success"]: console.print(f"[green]git add {files} ✓[/green]")
        return r
    def git_commit(message):
        r=_g("commit","-m",message)
        if r["success"]: console.print(f"[green]git commit ✓[/green]  {message!r}")
        else: console.print(f"[yellow]{r['output']}[/yellow]")
        return r
    def git_log(n=10):
        r=_g("log","--oneline",f"-{n}")
        if r["output"]: console.print(Panel(r["output"],title=f"[cyan]git log -{n}[/cyan]",border_style="dim"))
        return r
    def git_branch(name=""):
        r=_g(*filter(None,["branch",name]))
        console.print(Panel(r["output"] or "(no branches)",title="[cyan]git branch[/cyan]",border_style="dim"))
        return r
    def git_checkout(branch):
        r=_g("checkout",branch)
        if r["success"]: console.print(f"[green]git checkout {branch} ✓[/green]")
        return r
    def _llm_query(query: str) -> dict: return {"answer": query, "success": True}
    def _python_exec(code: str) -> dict: return _run_proc([py,"-c",code], project_root, tout, maxch)

    # ── assemble ──────────────────────────────────────────────────────────────
    def _t(n, d, p, o, fn): return {n: {"name":n,"description":d,"parameters":p,"output":o,"callable":fn}}

    tools: dict = {}
    tools.update(_t("run_code",
        "Execute code. For servers: capture_seconds=5+, wait_for_exit=False",
        [{"name":"filename","type":"str","optional":False},
         {"name":"args","type":"str","optional":True,"default":""},
         {"name":"capture_seconds","type":"float","optional":True,"default":0.0},
         {"name":"wait_for_exit","type":"bool","optional":True,"default":True},
         {"name":"timeout","type":"int","optional":True,"default":30}],
        [{"name":"success","type":"bool"},{"name":"pid","type":"int","optional":True},
         {"name":"status","type":"str"},{"name":"output_preview","type":"str","optional":True},
         {"name":"log_file","type":"str","optional":True}], run_code))
    tools.update(_t("read_process_output",
        "Read recent output from a running background process",
        [{"name":"pid","type":"int","optional":False},
         {"name":"lines","type":"int","optional":True,"default":50},
         {"name":"since_start","type":"bool","optional":True,"default":False}],
        [{"name":"success","type":"bool"},{"name":"output","type":"str"},
         {"name":"status","type":"str"},{"name":"runtime_seconds","type":"float"}],
        read_process_output))
    tools.update(_t("stop_process",
        "Stop a background process",
        [{"name":"pid","type":"int","optional":False},
         {"name":"graceful","type":"bool","optional":True,"default":True},
         {"name":"grace_period","type":"float","optional":True,"default":5.0}],
        [{"name":"success","type":"bool"},{"name":"exit_code","type":"int","optional":True}],
        stop_process))
    tools.update(_t("list_running_processes",
        "List all background processes started this session",
        [], [{"name":"processes","type":"list"},{"name":"count","type":"int"}],
        list_running_processes))
    tools.update(_t("list_windows",
        "List visible UI windows",
        [{"name":"pattern","type":"str","optional":True,"default":""}],
        [{"name":"windows","type":"list"},{"name":"count","type":"int"}], list_windows))
    tools.update(_t("screenshot_window",
        "Screenshot a window or full screen",
        [{"name":"title_pattern","type":"str","optional":True,"default":""},
         {"name":"output_path","type":"str","optional":True,"default":"screenshot.png"}],
        [{"name":"success","type":"bool"},{"name":"path","type":"str"},
         {"name":"base64_image","type":"str","optional":True}], screenshot_window))
    tools.update(_t("run_shell",  "Shell command in project folder. NEVER use curl — use http_get() instead.",
        [{"name":"command","type":"str","optional":False}],
        [{"name":"output","type":"str"}], run_shell))
    tools.update(_t("http_get",
        "HTTP GET request to a URL. Use this instead of curl/Invoke-WebRequest — works on all platforms.",
        [{"name":"url",    "type":"str",  "optional":False},
         {"name":"timeout","type":"float","optional":True,"default":5.0}],
        [{"name":"success",       "type":"bool"},
         {"name":"status_code",   "type":"int"},
         {"name":"body",          "type":"str"},
         {"name":"elapsed_ms",    "type":"float"},
         {"name":"server_running","type":"bool"}], http_get))
    tools.update(_t("http_post",
        "HTTP POST request with a string body. Use instead of curl -X POST / Invoke-WebRequest -Method POST.",
        [{"name":"url",          "type":"str",  "optional":False},
         {"name":"body",         "type":"str",  "optional":True, "default":""},
         {"name":"content_type", "type":"str",  "optional":True, "default":"application/json"},
         {"name":"timeout",      "type":"float","optional":True, "default":5.0}],
        [{"name":"success",    "type":"bool"},
         {"name":"status_code","type":"int"},
         {"name":"body",       "type":"str"}], http_post))
    tools.update(_t("wait_for_server",
        "Poll a URL until the server responds or timeout expires. "
        "Call this after run_code(..., wait_for_exit=False) before testing endpoints.",
        [{"name":"url",     "type":"str",  "optional":False},
         {"name":"timeout", "type":"float","optional":True,"default":30.0},
         {"name":"interval","type":"float","optional":True,"default":1.0}],
        [{"name":"ready",      "type":"bool"},
         {"name":"elapsed_s",  "type":"float"},
         {"name":"status_code","type":"int"},
         {"name":"attempts",   "type":"int"}], wait_for_server))
    tools.update(_t("run_tests",  "Run pytest",
        [{"name":"path","type":"str","optional":True,"default":"."}],
        [{"name":"output","type":"str"},{"name":"returncode","type":"int"}], run_tests))
    tools.update(_t("read_file",  "Read a project file",
        [{"name":"path","type":"str","optional":False}],
        [{"name":"content","type":"str"}], read_file))
    tools.update(_t("write_file", "Write/overwrite a project file",
        [{"name":"path","type":"str","optional":False},
         {"name":"content","type":"str","optional":False}],
        [{"name":"bytes","type":"int"}], write_file))
    tools.update(_t("list_files", "List project directory",
        [{"name":"path","type":"str","optional":True,"default":"."}],
        [{"name":"listing","type":"str"},{"name":"files","type":"list"}], list_files))
    tools.update(_t("scratchpad_read",  "Read a global scratchpad entry",
        [{"name":"key","type":"str","optional":False}],
        [{"name":"value","type":"str"}], scratchpad_read))
    tools.update(_t("scratchpad_write", "Write a global scratchpad entry",
        [{"name":"key","type":"str","optional":False},
         {"name":"value","type":"str","optional":False}],
        [{"name":"success","type":"bool"}], scratchpad_write))
    tools.update(_t("memory_add", "Add an entry to project long-term memory",
        [{"name":"text","type":"str","optional":False}],
        [{"name":"id","type":"int"}], memory_add))
    tools.update(_t("git_status",   "git status",   [], [{"name":"output","type":"str"}], git_status))
    tools.update(_t("git_diff",     "Show diff",
        [{"name":"file","type":"str","optional":True,"default":""}],
        [{"name":"output","type":"str"}], git_diff))
    tools.update(_t("git_add",      "Stage files",
        [{"name":"files","type":"str","optional":True,"default":"."}],
        [{"name":"output","type":"str"}], git_add))
    tools.update(_t("git_commit",   "Commit staged changes",
        [{"name":"message","type":"str","optional":False}],
        [{"name":"output","type":"str"}], git_commit))
    tools.update(_t("git_log",      "Recent commits",
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

    def _rebuild_tools(self) -> None:
        self._tools = build_agent_tools(
            self.out_folder, self._cfg, self._scratch, self._mem)

    @property
    def tools(self) -> dict: return self._tools

    @property
    def mem(self) -> list: return self._mem

    def reload_memory(self) -> None:
        self._mem = load_memory(self.out_folder)
        self._rebuild_tools()

    def _make_system_prompt(self) -> str:
        mem_text = _memory_summary(self._mem)
        scratch_lines = "\n".join(
            f"  {k} = {v['value']}" for k, v in sorted(self._scratch.items())
        ) or "  (empty)"
        env_text      = self._detect_environment()
        soul          = load_soul()
        file_skills   = load_file_skills()
        fskills_text  = get_file_skills_prompt(file_skills)
        return SYSTEM_PROMPT_TEMPLATE.format(
            base_personality=soul,
            env_section=env_text,
            file_skills_section=fskills_text,
            shell_rules_section=_shell_rules(),
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
        system  = platform.system()
        release = platform.release()
        machine = platform.machine()
        user    = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
        shell   = "unknown"
        if system == "Windows":
            shell = os.environ.get("COMSPEC","cmd.exe").split("\\")[-1].replace(".exe","")
            if "PSModulePath" in os.environ: shell = "powershell"
        else:
            shell = os.environ.get("SHELL","/bin/sh").split("/")[-1]
        py_version = sys.version.replace("\n"," ")
        venv = getattr(sys,"real_prefix",None) or getattr(sys,"base_prefix",None)
        venv_info = f"venv: {sys.prefix}" if (venv and venv != sys.prefix) else "system Python"
        tools_found = []
        for cmd in ["git","gcc","g++","clang","cmake","make","node","npm","cargo","rustc",
                    "go","javac","docker","kubectl","terraform"]:
            p = shutil.which(cmd)
            if p:
                try:
                    v = subprocess.run([cmd,"--version"],capture_output=True,text=True,timeout=2)
                    tools_found.append(f"  - {cmd}: {v.stdout.split(chr(10))[0][:50]}")
                except Exception:
                    tools_found.append(f"  - {cmd}: available")
        py_pkgs = []
        for pkg in ["fastapi","uvicorn","flask","django","requests","numpy","pandas",
                    "pytest","sqlalchemy","pydantic","httpx","aiohttp"]:
            try: __import__(pkg); py_pkgs.append(pkg)
            except ImportError: pass
        git_info = []
        if shutil.which("git"):
            try:
                r = subprocess.run(["git","rev-parse","--is-inside-work-tree"],
                                   capture_output=True,text=True,timeout=2,cwd=os.getcwd())
                if r.returncode == 0:
                    branch_r = subprocess.run(["git","branch","--show-current"],
                                              capture_output=True,text=True,timeout=2)
                    status_r = subprocess.run(["git","status","--porcelain"],
                                              capture_output=True,text=True,timeout=2)
                    log_r    = subprocess.run(["git","log","-1","--oneline"],
                                              capture_output=True,text=True,timeout=2)
                    git_info = [
                        f"  - Branch: {branch_r.stdout.strip()} ({'dirty' if status_r.stdout.strip() else 'clean'})",
                        f"  - Last commit: {log_r.stdout.strip()[:50]}",
                    ]
                else:
                    git_info = ["  - Not inside a git repo"]
            except Exception:
                git_info = ["  - Detection failed"]
        lines = [
            f"- OS: {system} {release} ({machine})",
            f"- User: {user}",
            f"- Shell: {shell}",
            f"- Working dir: {os.getcwd()}",
            f"- Date/Time: {datetime.now(timezone.utc).isoformat()} UTC",
            "",
            f"- Python: {py_version}",
            f"  Environment: {venv_info}",
            f"  Installed packages: {', '.join(py_pkgs) if py_pkgs else '(none)'}",
            "",
            "- System tools:",
        ]
        lines.extend(tools_found if tools_found else ["  (none)"])
        lines.extend(["", "- Git:"] + (git_info if git_info else ["  (not available)"]))
        return "\n".join(lines)

    def refresh_system_prompt(self) -> None:
        if self._disc:
            self._disc.system_prompt = self._make_system_prompt()
            self._disc.commit()

    @property
    def disc(self) -> LollmsDiscussion: return self._disc

    def new_discussion(self) -> None: self._new_discussion()

    def update_config(self, cfg: dict) -> None:
        self._cfg = cfg; self._rebuild_tools()

    def close(self) -> None:
        if self._disc: self._disc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Artefact helpers
# ═════════════════════════════════════════════════════════════════════════════

def _artefact_filename(a: dict) -> str:
    title = a.get("title", "artefact")
    if Path(title).suffix: return title
    return title + EXT_MAP.get((a.get("language") or "").lower(), ".txt")


def commit_artefacts(disc: LollmsDiscussion, folder: Path, cfg: dict) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    tout = cfg.get("shell_timeout", 30); maxch = cfg.get("max_output_chars", 8000)
    written = []
    for a in disc.artefacts.list():
        if a.get("type") not in CODE_TYPES: continue
        content = (a.get("content") or "").strip()
        if not content: continue
        dest = folder / _artefact_filename(a)
        dest.write_text(content, encoding="utf-8")
        written.append(str(dest))
    if written and cfg.get("git_auto_commit"):
        _run_proc(["git","add","."], folder, tout, maxch)
        r = _run_proc(["git","commit","-m",f"auto-commit: {len(written)} file(s)"], folder, tout, maxch)
        if r["success"]: console.print(f"[dim]git auto-commit ✓[/dim]")
    return written


# ═════════════════════════════════════════════════════════════════════════════
# NOTE artefact commands  (saved by LLM via <note> tags)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_notes(disc: LollmsDiscussion) -> None:
    notes = disc.artefacts.list(artefact_type=ArtefactType.NOTE)
    if not notes:
        console.print("[dim]No notes yet.  The LLM creates notes automatically "
                      "when [bold]enable_notes=True[/bold] (the default).[/dim]")
        return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("Title",   style="bold white", no_wrap=True)
    t.add_column("Ver",     justify="right", style="dim")
    t.add_column("Active",  justify="center")
    t.add_column("Lines",   justify="right")
    t.add_column("Updated", style="dim")
    for n in notes:
        lines = len((n.get("content") or "").splitlines())
        t.add_row(n.get("title","?"), str(n.get("version",1)),
                  "[green]✓[/green]" if n.get("active") else "[dim]·[/dim]",
                  str(lines), (n.get("updated_at") or "")[:16])
    console.print(Panel(t, title=f"[bold]Notes[/bold] ({len(notes)})", border_style="yellow"))


def _resolve_artefact(disc: LollmsDiscussion, title: str, atype: str) -> Optional[dict]:
    """Return artefact by exact title or case-insensitive prefix match."""
    exact = disc.artefacts.get(title)
    if exact and exact.get("type") == atype:
        return exact
    all_of_type = disc.artefacts.list(artefact_type=atype)
    matches = [x for x in all_of_type if x.get("title","").lower().startswith(title.lower())]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous — matches: "
                      f"{', '.join(x['title'] for x in matches)}[/yellow]")
    else:
        console.print(f"[red]{atype.capitalize()} '{title}' not found.[/red]")
    return None


def cmd_note_show(disc: LollmsDiscussion, title: str) -> None:
    n = _resolve_artefact(disc, title, ArtefactType.NOTE)
    if n is None: return
    console.print(Panel(Markdown(n.get("content","(empty)")),
                        title=f"[bold yellow]📝 {n['title']}[/bold yellow]  "
                              f"[dim]v{n.get('version',1)}[/dim]", border_style="yellow"))


def cmd_note_save(disc: LollmsDiscussion, title: str, path: Optional[str] = None) -> None:
    n = _resolve_artefact(disc, title, ArtefactType.NOTE)
    if n is None: return
    dest = Path(path) if path else Path(title.replace(" ","_")+".md")
    dest.write_text(n.get("content",""), encoding="utf-8")
    console.print(f"  [green]✓[/green]  Note saved → [bold]{dest}[/bold]")


# ═════════════════════════════════════════════════════════════════════════════
# SKILL artefact commands  (saved by LLM via <skill> tags)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_skills(disc: LollmsDiscussion) -> None:
    skills = disc.artefacts.list(artefact_type=ArtefactType.SKILL)
    if not skills:
        console.print("[dim]No skills yet.  The LLM creates skills when "
                      "[bold]enable_skills=True[/bold] and asked to save a pattern.[/dim]")
        return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("Title",       style="bold white", no_wrap=True)
    t.add_column("Category",    style="cyan",       no_wrap=True)
    t.add_column("Description", style="dim")
    t.add_column("Active",      justify="center")
    t.add_column("Ver",         justify="right", style="dim")
    for s in skills:
        t.add_row(s.get("title","?"), s.get("category","—"),
                  (s.get("description") or "—")[:50],
                  "[green]✓[/green]" if s.get("active") else "[dim]·[/dim]",
                  str(s.get("version",1)))
    console.print(Panel(t, title=f"[bold]Skills[/bold] ({len(skills)})", border_style="magenta"))


def cmd_skill_show(disc: LollmsDiscussion, title: str) -> None:
    s = _resolve_artefact(disc, title, ArtefactType.SKILL)
    if s is None: return
    meta_line = ""
    if s.get("category"):    meta_line += f"  Category: {s['category']}"
    if s.get("description"): meta_line += f"  |  {s['description']}"
    console.print(Panel(Markdown(s.get("content","(empty)")),
                        title=f"[bold magenta]🎓 {s['title']}[/bold magenta]  [dim]{meta_line}[/dim]",
                        border_style="magenta"))


def cmd_skill_toggle(disc: LollmsDiscussion, title: str) -> None:
    s = _resolve_artefact(disc, title, ArtefactType.SKILL)
    if s is None: return
    new_state = disc.artefacts.toggle(title)
    label = "[green]active[/green]" if new_state else "[dim]inactive[/dim]"
    console.print(f"  Skill '[bold]{title}[/bold]' is now {label}.")


def cmd_skill_delete(disc: LollmsDiscussion, title: str) -> None:
    s = _resolve_artefact(disc, title, ArtefactType.SKILL)
    if s is None: return
    if _ask_confirm(f"  Delete skill '{title}'?", default=False):
        disc.artefacts.remove(title)
        console.print(f"  [yellow]Skill '{title}' deleted.[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# File-based skill CLI  (/fskill …)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_fskill(sub: str, arg: str, skills: list[FileSkill]) -> list[FileSkill]:
    if sub in ("", "ls", "list"):
        if not skills:
            console.print("[dim]No file-based skills. /fskill new <name>[/dim]")
        else:
            t = Table(show_header=True, header_style="bold cyan", expand=False)
            t.add_column("",        width=2)
            t.add_column("Name",    style="bold white", no_wrap=True)
            t.add_column("Category",style="cyan")
            t.add_column("Description", style="dim")
            for s in skills:
                marker = "[green]✓[/green]" if s.enabled else "[dim]·[/dim]"
                t.add_row(marker, s.name, s.category, s.description[:50])
            console.print(Panel(t, title=f"File-based skills ({len(skills)})", border_style="cyan"))

    elif sub == "new":
        name = arg.strip().replace(" ","_").lower()
        if not name: console.print("[red]Usage: /fskill new <name>[/red]"); return skills
        if any(s.name == name for s in skills):
            console.print(f"[red]Skill '{name}' already exists.[/red]"); return skills
        description = _ask_text("  Description")
        category    = _ask_text("  Category", default="general")
        console.print("  Enter system prompt addition (Ctrl+D to finish):")
        lines = []
        try:
            while True:
                line = input("  > "); lines.append(line)
        except EOFError:
            pass
        skill = FileSkill(name, description, "\n".join(lines), category, enabled=True)
        skill.save(); skills.append(skill)
        console.print(f"[green]✓ File skill '{name}' created.[/green]")

    elif sub == "show":
        name = arg.strip()
        s = next((x for x in skills if x.name == name), None)
        if not s: console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            console.print(Panel(
                Markdown(f"## {s.name}  ({s.category})\n\n{s.description}\n\n---\n\n{s.prompt_addition}"),
                title=f"File skill: {s.name}", border_style="cyan" if s.enabled else "dim"))

    elif sub == "toggle":
        name = arg.strip()
        s = next((x for x in skills if x.name == name), None)
        if not s: console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            s.enabled = not s.enabled; s.save()
            console.print(f"[green]Skill '{name}' {'enabled' if s.enabled else 'disabled'}.[/green]")

    elif sub in ("delete","del"):
        name = arg.strip()
        s = next((x for x in skills if x.name == name), None)
        if not s: console.print(f"[red]Skill '{name}' not found.[/red]")
        elif _ask_confirm(f"Delete file skill '{name}'?", default=False):
            s.file_path.unlink(missing_ok=True); skills.remove(s)
            console.print(f"[yellow]File skill '{name}' deleted.[/yellow]")

    elif sub == "edit":
        name = arg.strip()
        s = next((x for x in skills if x.name == name), None)
        if not s: console.print(f"[red]Skill '{name}' not found.[/red]")
        else:
            import tempfile
            editor = (os.environ.get("EDITOR") or
                      ("notepad" if platform.system()=="Windows" else "nano"))
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
                f.write(s.file_path.read_text()); tmp = f.name
            try:
                subprocess.call([editor, tmp])
                updated = FileSkill.load(Path(tmp))
                updated.file_path = s.file_path; updated.save()
                idx = skills.index(s); skills[idx] = updated
                console.print(f"[green]✓ File skill '{name}' updated.[/green]")
            finally:
                Path(tmp).unlink(missing_ok=True)

    else:
        console.print("[red]Unknown /fskill sub-command.[/red]  ls | new | show | toggle | delete | edit")
    return skills


# ═════════════════════════════════════════════════════════════════════════════
# Project CRUD
# ═════════════════════════════════════════════════════════════════════════════

def cmd_project_new(arg: str, projects: dict, lc: LollmsClient,
                    cfg: dict, scratch: dict) -> Optional[ProjectSession]:
    parts = arg.split(maxsplit=1)
    if not parts or not parts[0]:
        console.print("[red]Usage: /project new <n> [output_dir][/red]"); return None
    name = parts[0].strip()
    if name in projects:
        console.print(f"[red]Project '{name}' already exists.[/red]"); return None
    projects_root = Path(cfg.get("projects_root", str(_documents_dir()/"lollms_projects")))
    out_path = (Path(parts[1].strip()) if len(parts) > 1
                else Path(_ask_text("  Output folder", default=str(projects_root/name))))
    db_file  = Path(_ask_text("  DB file",
                               default=str(CONFIG_DIR/"projects"/f"{name}.db")))
    db_file.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    projects[name] = {"db_path":     f"sqlite:///{db_file}",
                      "output_path": str(out_path),
                      "created_at":  datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save_projects(projects)
    console.print(f"[green]✅  Project '[bold]{name}[/bold]' created → {out_path}[/green]")
    return ProjectSession(name, projects, lc, cfg, scratch)


def cmd_project_switch(name: str, projects: dict, lc: LollmsClient,
                       cfg: dict, scratch: dict,
                       current: Optional[ProjectSession]) -> Optional[ProjectSession]:
    if not name: console.print("[red]Usage: /project switch <n>[/red]"); return current
    if name not in projects: console.print(f"[red]Project '{name}' not found.[/red]"); return current
    if current: current.close()
    s = ProjectSession(name, projects, lc, cfg, scratch)
    console.print(f"[green]Switched to '[bold]{name}[/bold]'.[/green]")
    return s


def cmd_project_ls(projects: dict, current_name: Optional[str]) -> None:
    if not projects:
        console.print("[dim]No projects. Use /project new <n>[/dim]"); return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("",        width=2)
    t.add_column("Name",    style="bold white", no_wrap=True)
    t.add_column("Output",  style="dim",        no_wrap=True)
    t.add_column("Created", style="dim")
    for n, meta in sorted(projects.items()):
        t.add_row("[green]▶[/green]" if n == current_name else " ",
                  n, meta.get("output_path","?"), meta.get("created_at","")[:10])
    console.print(t)


def cmd_project_info(session: ProjectSession) -> None:
    meta   = session.projects[session.name]; disc = session.disc
    notes  = disc.artefacts.list(artefact_type=ArtefactType.NOTE)
    skills = disc.artefacts.list(artefact_type=ArtefactType.SKILL)
    code   = [a for a in disc.artefacts.list() if a.get("type") in CODE_TYPES]
    t = Table(show_header=False, box=None, padding=(0,2))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="yellow")
    t.add_row("Project",        session.name)
    t.add_row("Output",         meta.get("output_path","?"))
    t.add_row("DB",             meta.get("db_path","?").replace("sqlite:///",""))
    t.add_row("Created",        meta.get("created_at","?")[:19])
    t.add_row("Discussion",     disc.id)
    t.add_row("Code artefacts", str(len(code)))
    t.add_row("Notes",          str(len(notes)))
    t.add_row("Skills",         str(len(skills)))
    t.add_row("Memory",         f"{len(session.mem)} entries")
    t.add_row("Git repo",       "yes" if (session.out_folder/".git").exists() else "not initialised")
    console.print(Panel(t, title=f"Project: {session.name}", border_style="cyan"))


def cmd_project_delete(name: str, projects: dict,
                       current: Optional[ProjectSession]) -> Optional[ProjectSession]:
    if not name: console.print("[red]Usage: /project delete <n>[/red]"); return current
    if name not in projects: console.print(f"[red]Project '{name}' not found.[/red]"); return current
    if not _ask_confirm(f"  Remove project '{name}' from registry? (files kept)"): return current
    if current and current.name == name:
        current.close(); current = None
    del projects[name]; save_projects(projects)
    console.print(f"[yellow]Project '{name}' removed.[/yellow]")
    return current


# ═════════════════════════════════════════════════════════════════════════════
# Rich display helpers
# ═════════════════════════════════════════════════════════════════════════════

def show_artefact_table(disc: LollmsDiscussion) -> None:
    """List code/doc/file artefacts only (notes+skills have their own commands)."""
    arts = [a for a in disc.artefacts.list() if a.get("type") in CODE_TYPES]
    if not arts: console.print("[dim]No code artefacts yet.[/dim]"); return
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    t.add_column("Title",  style="bold white", no_wrap=True)
    t.add_column("Type",   style="cyan")
    t.add_column("Lang",   style="green")
    t.add_column("Ver",    justify="right")
    t.add_column("Active", justify="center")
    t.add_column("Lines",  justify="right")
    for a in arts:
        lines = len((a.get("content") or "").splitlines())
        t.add_row(a.get("title","?"), a.get("type","?"),
                  a.get("language") or "–", str(a.get("version",1)),
                  "[green]✓[/green]" if a.get("active") else "[dim]·[/dim]", str(lines))
    console.print(t)


def show_history(disc: LollmsDiscussion) -> None:
    branch = disc.get_branch(disc.active_branch_id)
    if not branch: console.print("[dim]No messages yet.[/dim]"); return
    for msg in branch:
        label = "[bold blue]You[/bold blue]" if msg.sender_type == "user" \
                else "[bold green]Assistant[/bold green]"
        console.print(label); console.print(Markdown(msg.content or "")); console.print()


def print_welcome_banner() -> None:
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
    editable = {
        "git_auto_commit":        ("bool", "Auto-commit after /commit"),
        "shell_timeout":          ("int",  "Shell command timeout (seconds)"),
        "max_output_chars":       ("int",  "Max captured output chars"),
        "python_executable":      ("str",  "Python executable path"),
        "show_thinking":          ("bool", "Show thinking process and tool calls"),
        "enable_notes":           ("bool", "LLM saves <note> tags as NOTE artefacts"),
        "enable_skills":          ("bool", "LLM saves <skill> tags as SKILL artefacts"),
        "enable_inline_widgets":  ("bool", "LLM embeds <lollms_inline> widgets (needs browser)"),
    }
    if sub in ("","ls","list"):
        t = Table(show_header=True, header_style="bold cyan", expand=False)
        t.add_column("Key",           style="bold white", no_wrap=True)
        t.add_column("Type",          style="dim")
        t.add_column("Current Value", style="yellow")
        t.add_column("Description",   style="dim")
        for key,(typ,desc) in editable.items():
            t.add_row(key, typ, str(cfg.get(key,"—")), desc)
        console.print(Panel(t, title="Editable Configuration", border_style="cyan"))
        console.print("\n[dim]/config set <key> <value> to modify[/dim]")
    elif sub == "set":
        parts = arg.split(maxsplit=1)
        if len(parts) < 2: console.print("[red]Usage: /config set <key> <value>[/red]"); return cfg
        key, raw_val = parts[0], parts[1].strip()
        if key not in editable: console.print(f"[red]Unknown key '{key}'.[/red]"); return cfg
        typ, desc = editable[key]
        try:
            val = (raw_val.lower() in ("true","yes","1","on")) if typ=="bool" else \
                  int(raw_val) if typ=="int" else raw_val
        except ValueError:
            console.print(f"[red]Invalid {typ} value: {raw_val}[/red]"); return cfg
        cfg[key] = val; save_config(cfg)
        console.print(f"  [green]✓[/green]  [bold]{key}[/bold] = {val!r}  [dim]({desc})[/dim]")
    elif sub == "get":
        key = arg.strip()
        if key not in editable: console.print(f"[red]Unknown key '{key}'.[/red]"); return cfg
        val = cfg.get(key,"—"); typ,desc = editable[key]
        console.print(f"  [bold cyan]{key}[/bold cyan] ({typ})\n  Value: [yellow]{val!r}[/yellow]\n  {desc}")
    else:
        console.print("[red]Unknown /config sub-command.[/red]  ls | set | get")
    return cfg


def cmd_soul(sub: str, arg: str) -> None:
    if sub in ("","show","cat"):
        console.print(Panel(Markdown(load_soul()), title="Current Soul", border_style="magenta"))
    elif sub == "edit":
        import tempfile
        editor = (os.environ.get("EDITOR") or
                  ("notepad" if platform.system()=="Windows" else "nano"))
        soul = load_soul()
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
            f.write(soul); tmp = f.name
        try:
            subprocess.call([editor, tmp])
            new_soul = Path(tmp).read_text(encoding="utf-8")
            if new_soul != soul: save_soul(new_soul); console.print("[green]✓ Soul updated.[/green]")
            else: console.print("[dim]No changes.[/dim]")
        finally: Path(tmp).unlink(missing_ok=True)
    elif sub == "reset":
        if _ask_confirm("Reset soul to default?", default=False):
            save_soul(DEFAULT_SOUL); console.print("[yellow]Soul reset.[/yellow]")
    else:
        console.print("[red]Unknown /soul sub-command.[/red]  show | edit | reset")


def print_help() -> None:
    sections = [
        ("Setup & Personality", [
            ("/setup",                "Re-run setup wizard"),
            ("/setup show",           "Print current config"),
            ("/config",               "List editable runtime settings"),
            ("/config set <k> <v>",   "Update a config value (inc. enable_notes, enable_skills)"),
            ("/soul",                 "Show personality"),
            ("/soul edit",            "Edit personality in $EDITOR"),
            ("/soul reset",           "Reset personality to default"),
            ("/reset_configuration",  "[red]⚠️  NUCLEAR: wipe all data and exit[/red]"),
        ]),
        ("File-based Skills  (prompt-engineering modules)", [
            ("/fskill",               "List file-based skills"),
            ("/fskill new <name>",    "Create a new file skill"),
            ("/fskill show <name>",   "View a file skill"),
            ("/fskill toggle <name>", "Enable / disable a file skill"),
            ("/fskill edit <name>",   "Edit in $EDITOR"),
            ("/fskill delete <name>", "Delete permanently"),
        ]),
        ("Projects", [
            ("/project new <n> [dir]", "Create & switch to a new project"),
            ("/project switch <n>",    "Switch to an existing project"),
            ("/project ls",            "List all projects"),
            ("/project info",          "Current project details (inc. note/skill counts)"),
            ("/project delete <n>",    "Remove from registry"),
        ]),
        ("Scratchpad  (global inter-agent key-value store)", [
            ("/scratch set <key> <val>", "Write an entry"),
            ("/scratch get <key>",       "Read an entry"),
            ("/scratch ls",              "List all entries"),
            ("/scratch del <key>",       "Delete an entry"),
            ("/scratch clear",           "Wipe everything"),
        ]),
        ("Project memory  (per-project long-term memory)", [
            ("/memory add <text>",     "Append a memory entry"),
            ("/memory ls",             "List all entries"),
            ("/memory del <id>",       "Delete by id"),
            ("/memory clear",          "Wipe all entries"),
            ("/memory search <query>", "Keyword search"),
        ]),
        ("Code Artefacts", [
            ("/ls",               "List code/document artefacts"),
            ("/show <n>",         "Syntax-highlighted view"),
            ("/activate <n>",     "Inject into LLM context"),
            ("/deactivate <n>",   "Remove from LLM context"),
            ("/commit [folder]",  "Write code artefacts to disk"),
        ]),
        ("Notes  (saved by LLM via <note> tags)", [
            ("/notes",                     "List all notes"),
            ("/note show <title>",         "View a note (prefix match OK)"),
            ("/note save <title> [path]",  "Export note to Markdown file"),
        ]),
        ("Skills  (saved by LLM via <skill> tags)", [
            ("/skills",               "List all skills"),
            ("/skill show <title>",   "View a skill (prefix match OK)"),
            ("/skill toggle <title>", "Enable / disable a skill"),
            ("/skill delete <title>", "Permanently remove a skill"),
        ]),
        ("Files, execution & tests", [
            ("/run <file> [args]",  "Execute a file"),
            ("/shell <command>",    "Shell command in project folder"),
            ("/test [path]",        "Run pytest"),
            ("/files [path]",       "List project files"),
        ]),
        ("Process management", [
            ("/ps",           "List running background processes"),
            ("/kill <pid>",   "Stop a background process"),
            ("/proc <pid>",   "Read recent output from a process"),
        ]),
        ("HTTP tools  (no curl needed — works on all platforms)", [
            ("/get <url>",               "HTTP GET a URL and display the response"),
            ("/post <url> [body]",       "HTTP POST with optional JSON body"),
            ("/wait <url> [timeout]",    "Wait until a server is ready to accept connections"),
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
        ]),
    ]
    for title, rows in sections:
        t = Table(show_header=False, box=None, padding=(0,2))
        t.add_column(style="bold yellow", no_wrap=True)
        t.add_column(style="white")
        for c, d in rows: t.add_row(c, d)
        console.print(Panel(t, title=f"[bold]{title}[/bold]", border_style="dim"))


# ═════════════════════════════════════════════════════════════════════════════
# Prompt-toolkit input
# ═════════════════════════════════════════════════════════════════════════════

def build_completer(projects: dict, session: Optional[ProjectSession]) -> Optional[Any]:
    if not _HAS_PROMPT_TOOLKIT: return None
    commands = {
        "/quit":None, "/exit":None, "/help":None, "/clear":None, "/history":None,
        "/ls":None, "/files":None, "/commit":None, "/ps":None,
        "/get":None, "/post":None, "/wait":None, "/kill":None, "/proc":None,
        "/notes":None, "/note":{"show":None,"save":None},
        "/skills":None, "/skill":{"show":None,"toggle":None,"delete":None},
        "/fskill":{"ls":None,"new":None,"show":None,"toggle":None,"delete":None,"edit":None},
        "/setup":{"show":None},
        "/config":{"set":None,"get":None,"ls":None},
        "/soul":{"show":None,"edit":None,"reset":None},
        "/project":{"new":None,"switch":None,"ls":None,"info":None,"delete":None},
        "/scratch":{"set":None,"get":None,"ls":None,"del":None,"clear":None},
        "/memory":{"add":None,"ls":None,"del":None,"clear":None,"search":None},
        "/activate":None, "/deactivate":None, "/show":None,
        "/run":None, "/shell":None, "/test":None,
        "/git":{"init":None,"status":None,"add":None,"commit":None,
                "diff":None,"log":None,"branch":None,"checkout":None},
        "/reset_configuration":None,
    }
    if session:
        commands["/project"]["switch"] = {p:None for p in projects.keys()}
        try:
            arts = session.disc.artefacts.list()
            all_names  = {a.get("title",""):None for a in arts if a.get("title")}
            note_names = {a.get("title",""):None for a in arts
                          if a.get("type")==ArtefactType.NOTE and a.get("title")}
            sk_names   = {a.get("title",""):None for a in arts
                          if a.get("type")==ArtefactType.SKILL and a.get("title")}
            commands["/activate"] = all_names
            commands["/deactivate"] = all_names
            commands["/show"] = all_names
            commands["/note"]  = {"show": note_names, "save": note_names}
            commands["/skill"] = {"show": sk_names, "toggle": sk_names, "delete": sk_names}
        except Exception: pass
    return NestedCompleter.from_nested_dict(commands)


def get_input_with_history(session: Optional[ProjectSession], projects: dict) -> str:
    marker_text = (
        (f"{session.name} " if session else "") + "❯ "
    )
    if not _HAS_PROMPT_TOOLKIT:
        try: return Prompt.ask(marker_text).strip()
        except (EOFError, KeyboardInterrupt): return "/quit"
    completer = build_completer(projects, session)
    ps = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        completer=completer,
        complete_style=CompleteStyle.READLINE_LIKE,
        key_bindings=KeyBindings(),
    )
    try: return ps.prompt(marker_text).strip()
    except (EOFError, KeyboardInterrupt): return "/quit"


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg       = load_config()
    first_run = _is_fresh_install() or not cfg.get("binding_config")
    print_welcome_banner()
    if first_run:
        console.print("\n[yellow]Welcome! Let's set up your LLM binding.[/yellow]\n")
        cfg = run_setup_wizard(cfg, first_run=True)

    lc         = build_client(cfg)
    projects   = load_projects()
    scratch    = load_scratchpad()
    file_skills: list[FileSkill] = load_file_skills()
    session: Optional[ProjectSession] = None

    if projects:
        latest  = max(projects, key=lambda n: projects[n].get("created_at",""))
        session = ProjectSession(latest, projects, lc, cfg, scratch)
        console.print(
            f"[dim]Binding: [bold]{cfg['binding_name']}[/bold]  "
            f"model: {cfg['binding_config'].get('model_name','?')}  "
            f"|  Project: [bold]{latest}[/bold][/dim]\n")
    else:
        console.print(
            f"[dim]Binding: [bold]{cfg['binding_name']}[/bold]  "
            f"model: {cfg['binding_config'].get('model_name','?')}[/dim]\n"
            "[yellow]No projects yet.[/yellow]  "
            "Create one: [bold]/project new <n>[/bold]\n")

    while True:
        raw = get_input_with_history(session, projects)
        if raw == "/quit":
            console.print("\n[dim]Bye![/dim]"); break
        if not raw:
            continue
        cmd, _, arg = raw.partition(" ")
        arg = arg.strip()

        if cmd in ("/quit","/exit"):
            console.print("[dim]Bye![/dim]"); break

        # ── /reset_configuration ──────────────────────────────────────────────
        if cmd == "/reset_configuration":
            console.print(Panel(
                "[bold red]⚠️  DESTRUCTIVE OPERATION[/bold red]\n\n"
                "Permanently deletes all config, databases, and cached data.",
                title="[bold red]RESET CONFIGURATION[/bold red]", border_style="red"))
            if _ask_confirm("  Permanently delete all data?", default=False):
                try:
                    import shutil
                    if session: session.close(); session = None
                    import gc; gc.collect()
                    if CONFIG_DIR.exists():
                        for attempt in range(3):
                            try: shutil.rmtree(CONFIG_DIR); break
                            except PermissionError:
                                if attempt < 2: time.sleep(1.0); gc.collect()
                                else: raise
                    RESET_MARKER.parent.mkdir(parents=True, exist_ok=True)
                    RESET_MARKER.touch()
                    console.print("[bold red]All data erased.[/bold red]")
                except Exception as e:
                    console.print(f"[red]Cleanup error: {e}[/red]")
            else:
                console.print("[dim]Reset cancelled.[/dim]")
            console.print("[dim]Bye![/dim]"); break

        if cmd == "/help":   print_help(); continue
        if cmd == "/setup":
            if arg == "show": show_config(cfg)
            else:
                cfg = run_setup_wizard(cfg); lc = build_client(cfg)
                if session: session._lc = lc; session.update_config(cfg)
            continue
        if cmd == "/config":
            sub, _, sub_arg = arg.partition(" ")
            cfg = cmd_config(sub.strip(), sub_arg.strip(), cfg)
            if session: session.update_config(cfg)
            continue
        if cmd == "/soul":
            sub, _, sub_arg = arg.partition(" ")
            cmd_soul(sub.strip(), sub_arg.strip())
            if session: session.refresh_system_prompt()
            continue
        if cmd == "/fskill":
            sub, _, sub_arg = arg.partition(" ")
            file_skills = cmd_fskill(sub.strip(), sub_arg.strip(), file_skills)
            if session: session.refresh_system_prompt()
            continue
        if cmd == "/project":
            sub, _, sub_arg = arg.partition(" "); sub_arg = sub_arg.strip()
            if sub == "new":
                ns = cmd_project_new(sub_arg, projects, lc, cfg, scratch)
                if ns:
                    if session: session.close()
                    session = ns
            elif sub == "switch":
                session = cmd_project_switch(sub_arg, projects, lc, cfg, scratch, session)
            elif sub == "ls":
                cmd_project_ls(projects, session.name if session else None)
            elif sub == "info":
                if session: cmd_project_info(session)
                else: console.print("[red]No active project.[/red]")
            elif sub == "delete":
                session = cmd_project_delete(sub_arg, projects, session)
                if session is None and projects:
                    other = max(projects, key=lambda n: projects[n].get("created_at",""))
                    session = ProjectSession(other, projects, lc, cfg, scratch)
                    console.print(f"[dim]Auto-switched to '[bold]{other}[/bold]'.[/dim]")
            else: console.print("[red]Unknown /project sub-command.[/red]")
            continue
        if cmd == "/scratch":
            sub, _, sub_arg = arg.partition(" ")
            scratch = cmd_scratch(sub.strip(), sub_arg.strip(), scratch)
            if session: session._scratch = scratch; session.refresh_system_prompt()
            continue

        if session is None:
            console.print("[red]No active project.[/red]  Create one: [bold]/project new <n>[/bold]")
            continue

        disc   = session.disc
        tools  = session.tools
        folder = session.out_folder
        tout   = cfg.get("shell_timeout", 30)
        maxch  = cfg.get("max_output_chars", 8000)

        if cmd == "/memory":
            sub, _, sub_arg = arg.partition(" ")
            session._mem = cmd_memory(sub.strip(), sub_arg.strip(), folder, session._mem)
            session._rebuild_tools(); session.refresh_system_prompt()
            continue
        if cmd == "/clear":   session.new_discussion(); continue
        if cmd == "/history": show_history(disc); continue
        if cmd == "/ls":      show_artefact_table(disc); continue

        # ── /notes / /note ────────────────────────────────────────────────────
        if cmd == "/notes":
            cmd_notes(disc); continue
        if cmd == "/note":
            sub, _, sub_arg = arg.partition(" "); sub, sub_arg = sub.strip(), sub_arg.strip()
            if sub == "show":
                if not sub_arg: console.print("[red]Usage: /note show <title>[/red]")
                else: cmd_note_show(disc, sub_arg)
            elif sub == "save":
                parts = sub_arg.split(maxsplit=1)
                if not parts: console.print("[red]Usage: /note save <title> [path][/red]")
                else: cmd_note_save(disc, parts[0], parts[1] if len(parts)>1 else None)
            else: console.print("[red]Unknown /note sub-command.[/red]  show | save")
            continue

        # ── /skills / /skill ──────────────────────────────────────────────────
        if cmd == "/skills":
            cmd_skills(disc); continue
        if cmd == "/skill":
            sub, _, sub_arg = arg.partition(" "); sub, sub_arg = sub.strip(), sub_arg.strip()
            if sub == "show":
                if not sub_arg: console.print("[red]Usage: /skill show <title>[/red]")
                else: cmd_skill_show(disc, sub_arg)
            elif sub == "toggle":
                if not sub_arg: console.print("[red]Usage: /skill toggle <title>[/red]")
                else: cmd_skill_toggle(disc, sub_arg)
            elif sub == "delete":
                if not sub_arg: console.print("[red]Usage: /skill delete <title>[/red]")
                else: cmd_skill_delete(disc, sub_arg)
            else: console.print("[red]Unknown /skill sub-command.[/red]  show | toggle | delete")
            continue

        # ── /files ────────────────────────────────────────────────────────────
        if cmd == "/files":
            r = tools["list_files"]["callable"](path=arg or ".")
            if r["success"]: console.print(Panel(r["listing"], title="[cyan]project files[/cyan]", border_style="dim"))
            else: console.print(f"[red]{r['output']}[/red]")
            continue

        # ── /show ─────────────────────────────────────────────────────────────
        if cmd == "/show":
            if not arg: console.print("[red]Usage: /show <artefact_name>[/red]"); continue
            a = disc.artefacts.get(arg)
            if a is None: console.print(f"[red]Artefact '{arg}' not found.[/red]")
            else: console.print(Syntax(a.get("content",""), a.get("language") or "text",
                                       theme="monokai", line_numbers=True))
            continue

        # ── /activate / /deactivate ───────────────────────────────────────────
        if cmd == "/activate":
            if not arg: console.print("[red]Usage: /activate <n>[/red]"); continue
            try: disc.artefacts.activate(arg); console.print(f"[green]Activated '{arg}'.[/green]")
            except Exception as e: console.print(f"[red]{e}[/red]")
            continue
        if cmd == "/deactivate":
            if not arg: console.print("[red]Usage: /deactivate <n>[/red]"); continue
            try: disc.artefacts.deactivate(arg); console.print(f"[dim]Deactivated '{arg}'.[/dim]")
            except Exception as e: console.print(f"[red]{e}[/red]")
            continue

        # ── /commit ───────────────────────────────────────────────────────────
        if cmd == "/commit":
            target  = Path(arg) if arg else folder
            written = commit_artefacts(disc, target, cfg)
            if not written: console.print("[yellow]No code artefacts to commit.[/yellow]")
            else:
                console.print(Rule(f"[green]Committed {len(written)} file(s) → {target}[/green]"))
                for p in written: console.print(f"  [green]✓[/green]  {p}")
            continue

        # ── /run ──────────────────────────────────────────────────────────────
        if cmd == "/run":
            if not arg: console.print("[red]Usage: /run <file> [args…][/red]"); continue
            parts = arg.split(maxsplit=1)
            tools["run_code"]["callable"](filename=parts[0],
                                          args=parts[1] if len(parts)>1 else "")
            continue

        # ── /shell ────────────────────────────────────────────────────────────
        if cmd == "/shell":
            if not arg: console.print("[red]Usage: /shell <command>[/red]"); continue
            tools["run_shell"]["callable"](command=arg); continue

        # ── /test ─────────────────────────────────────────────────────────────
        if cmd == "/test":
            r = tools["run_tests"]["callable"](path=arg or ".")
            if r["success"]: console.print("[bold green]✅  All tests passed.[/bold green]")
            else:            console.print("[bold red]❌  Tests failed.[/bold red]")
            continue

        # ── Process management ────────────────────────────────────────────────
        if cmd == "/ps":
            r = tools["list_running_processes"]["callable"]()
            if not r["processes"]:
                console.print("[dim]No background processes.[/dim]")
            else:
                t = Table(show_header=True, header_style="bold cyan", expand=False)
                t.add_column("PID",     style="bold", justify="right")
                t.add_column("File",    style="white")
                t.add_column("Status",  style="green")
                t.add_column("Runtime", justify="right", style="dim")
                t.add_column("Log",     style="dim")
                for p in r["processes"]:
                    t.add_row(str(p["pid"]), p["filename"], p["status"],
                              f"{p['runtime_seconds']}s",
                              Path(p["log_file"]).name if p.get("log_file") else "—")
                console.print(t)
            continue
        if cmd == "/kill":
            if not arg: console.print("[red]Usage: /kill <pid>[/red]"); continue
            try: tools["stop_process"]["callable"](pid=int(arg))
            except ValueError: console.print("[red]PID must be an integer.[/red]")
            continue
        if cmd == "/proc":
            if not arg: console.print("[red]Usage: /proc <pid>[/red]"); continue
            try:
                r = tools["read_process_output"]["callable"](pid=int(arg), lines=50)
                if r["success"]:
                    console.print(Panel(r.get("output","(no output)"),
                                        title=f"[cyan]Process {arg}  [{r.get('status','?')}][/cyan]",
                                        border_style="green" if r.get("status")=="running" else "dim"))
                else:
                    console.print(f"[red]{r.get('output','Error')}[/red]")
            except ValueError: console.print("[red]PID must be an integer.[/red]")
            continue

        # ── /get /post /wait  (HTTP tools — no curl needed) ──────────────────
        if cmd == "/get":
            if not arg: console.print("[red]Usage: /get <url>[/red]"); continue
            tools["http_get"]["callable"](url=arg)
            continue

        if cmd == "/post":
            # /post <url> [json_body]
            parts = arg.split(maxsplit=1)
            if not parts: console.print("[red]Usage: /post <url> [json_body][/red]"); continue
            url_arg  = parts[0]
            body_arg = parts[1] if len(parts) > 1 else ""
            tools["http_post"]["callable"](url=url_arg, body=body_arg)
            continue

        if cmd == "/wait":
            if not arg: console.print("[red]Usage: /wait <url> [timeout_seconds][/red]"); continue
            parts = arg.split(maxsplit=1)
            url_arg     = parts[0]
            timeout_arg = float(parts[1]) if len(parts) > 1 else 30.0
            tools["wait_for_server"]["callable"](url=url_arg, timeout=timeout_arg)
            continue


        if cmd == "/git":
            sub, _, sub_arg = arg.partition(" "); sub_arg = sub_arg.strip(); g = tools
            if sub == "init":
                r = _run_proc(["git","init"], folder, tout, maxch)
                console.print(Panel(r["output"], border_style="green" if r["success"] else "red"))
            elif sub == "status":   g["git_status"]["callable"]()
            elif sub == "add":      g["git_add"]["callable"](files=sub_arg or ".")
            elif sub == "commit":
                if not sub_arg: console.print("[red]Usage: /git commit <message>[/red]")
                else: g["git_commit"]["callable"](message=sub_arg)
            elif sub == "diff":     g["git_diff"]["callable"](file=sub_arg)
            elif sub == "log":      g["git_log"]["callable"](n=int(sub_arg) if sub_arg.isdigit() else 10)
            elif sub == "branch":   g["git_branch"]["callable"](name=sub_arg)
            elif sub == "checkout":
                if not sub_arg: console.print("[red]Usage: /git checkout <branch>[/red]")
                else: g["git_checkout"]["callable"](branch=sub_arg)
            else: console.print("[red]Unknown git sub-command.[/red]  "
                                 "init | status | add | commit | diff | log | branch | checkout")
            continue

        if cmd.startswith("/"):
            console.print(f"[red]Unknown command '{cmd}'.[/red]  Type [bold]/help[/bold].")
            continue

        # ══════════════════════════════════════════════════════════════════════
        # LLM turn
        # ══════════════════════════════════════════════════════════════════════
        console.print(Rule("[bold green]Assistant[/bold green]")); console.print()
        buffer = ""

        # Pre-check: detect action intent and prepend the appropriate reminder
        # so the model knows exactly which tools to use and in what order.
        reminder = _detect_intent_reminder(raw)
        user_message = (reminder + raw) if reminder else raw

        # Track which intent class fired (used for hallucination warning below)
        _intent_is_action = bool(reminder)

        # Spinner
        _spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        _spinner_idx    = [0]
        _spinner_active = [True]

        def _spin():
            while _spinner_active[0]:
                f = _spinner_frames[_spinner_idx[0] % len(_spinner_frames)]
                ASCIIColors.rich_print(f"\r  {f} [dim]Thinking…[/dim]  ", end="", flush=True)
                _spinner_idx[0] += 1; time.sleep(0.08)

        _spinner_thread = threading.Thread(target=_spin, daemon=True)
        _spinner_thread.start()

        def _stop_spinner():
            if _spinner_active[0]:
                _spinner_active[0] = False; _spinner_thread.join(timeout=0.3)
                print("\r"+" "*40+"\r", end="", flush=True)

        # Callback ─────────────────────────────────────────────────────────────
        def stream_cb(text: str, msg_type=None, meta=None) -> bool:
            nonlocal buffer

            if msg_type in (MSG_TYPE.MSG_TYPE_STEP_START, MSG_TYPE.MSG_TYPE_STEP_END):
                _stop_spinner()
                prefix = "▷" if msg_type==MSG_TYPE.MSG_TYPE_STEP_START else "◁"
                ASCIIColors.cyan(f"  {prefix} {text}", end="\n"); return True

            if msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
                _stop_spinner()
                try:
                    td = (json.loads(text)
                          if isinstance(text,str) and text.strip().startswith("{")
                          else {"name": str(text)})
                    name = td.get("name","tool")
                except Exception: name = str(text)[:60]
                ASCIIColors.cyan(f"  ▶ Running {name}…", end="\n"); return True

            if msg_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
                _stop_spinner()
                preview = ""
                if isinstance(meta, dict) and "result" in meta:
                    r = meta["result"]
                    if isinstance(r, dict):
                        preview = str(r.get("output") or r.get("content") or
                                      r.get("error",""))[:500]
                    else:
                        preview = str(r)[:500]
                    if len(preview) >= 500: preview += "…"
                ASCIIColors.green("  ✓ Tool complete", end="")
                if preview: ASCIIColors.dim(f" → {preview}", end="")
                ASCIIColors.print(""); return True

            if msg_type == MSG_TYPE.MSG_TYPE_NEW_MESSAGE:
                # Library pre-created the ai_message; nothing to display here
                return True

            if msg_type == MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
                _stop_spinner(); return True

            # ── New inline streaming events ────────────────────────────────
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and not text and isinstance(meta, dict):
                t = meta.get("type"); c = meta.get("content", {})
                if t == "note_start":
                    _stop_spinner()
                    ASCIIColors.yellow(f"  📝 Saving note: {c.get('title','…')}", end="\n")
                elif t == "skill_start":
                    _stop_spinner()
                    cat = c.get("category","")
                    ASCIIColors.magenta(
                        f"  🎓 Saving skill: {c.get('title','…')}"
                        + (f"  [{cat}]" if cat else ""), end="\n")
                elif t == "artefact_update":
                    _stop_spinner()
                    ASCIIColors.cyan(f"  📄 Updating artefact: {c.get('title','…')}", end="\n")
                elif t == "inline_widget_start":
                    _stop_spinner()
                    ASCIIColors.blue(
                        f"  🎛  Widget: {c.get('title','…')} ({c.get('widget_type','html')})",
                        end="\n")
                return True

            if msg_type in (MSG_TYPE.MSG_TYPE_EXCEPTION, MSG_TYPE.MSG_TYPE_ERROR):
                _stop_spinner(); ASCIIColors.red(f"  ✗ {text}", end="\n"); return True
            if msg_type == MSG_TYPE.MSG_TYPE_WARNING:
                _stop_spinner(); ASCIIColors.yellow(f"  ⚠  {text}", end="\n"); return True
            if msg_type == MSG_TYPE.MSG_TYPE_INFO:
                _stop_spinner(); ASCIIColors.dim(f"  ℹ  {text}", end="\n"); return True
            if msg_type == MSG_TYPE.MSG_TYPE_SCRATCHPAD:
                return True  # suppress — internal state, not for display

            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                _stop_spinner(); buffer += text; return True

            return True

        # ── chat() — all new library flags wired in ────────────────────────
        result = disc.chat(
            user_message          = user_message,
            tools                 = tools,
            streaming_callback    = stream_cb,
            auto_activate_artefacts = True,
            enable_notes          = cfg.get("enable_notes",          True),
            enable_skills         = cfg.get("enable_skills",         True),
            enable_inline_widgets = cfg.get("enable_inline_widgets", False),
            # Image generation auto-disabled when lollmsClient.tti is None
            enable_image_generation = True,
            enable_image_editing    = True,
        )

        _stop_spinner()
        if buffer.strip(): console.print(Markdown(buffer))
        console.print()

        # Post-turn housekeeping
        session.reload_memory()
        session.refresh_system_prompt()

        # Extract metadata from correct location (ai_message.metadata)
        ai_msg      = result.get("ai_message") if isinstance(result, dict) else None
        affected    = result.get("artefacts",  []) if isinstance(result, dict) else []
        meta_data   = (ai_msg.metadata if ai_msg else {}) or {}
        tool_calls  = meta_data.get("tool_calls", [])

        notes_saved  = [a for a in affected if a.get("type") == ArtefactType.NOTE]
        skills_saved = [a for a in affected if a.get("type") == ArtefactType.SKILL]
        code_saved   = [a for a in affected if a.get("type") in CODE_TYPES]
        widgets      = meta_data.get("inline_widgets", [])

        summary_lines = []
        if code_saved:
            names = ", ".join(a.get("title","?") for a in code_saved)
            summary_lines.append(f"  [cyan]📦 Code artefacts:[/cyan] [bold]{names}[/bold]  — /commit to write to disk")
        if notes_saved:
            names = ", ".join(a.get("title","?") for a in notes_saved)
            summary_lines.append(f"  [yellow]📝 Notes saved:[/yellow] [bold]{names}[/bold]  — /notes to view")
        if skills_saved:
            names = ", ".join(a.get("title","?") for a in skills_saved)
            summary_lines.append(f"  [magenta]🎓 Skills saved:[/magenta] [bold]{names}[/bold]  — /skills to view")
        if widgets:
            names = ", ".join(w.get("title","?") for w in widgets)
            summary_lines.append(f"  [blue]🎛  Widgets:[/blue] [bold]{names}[/bold]")
        if tool_calls:
            summary_lines.append(
                f"  [dim]🔧 Tools: {len(tool_calls)}  "
                f"({', '.join(t.get('name','?') for t in tool_calls)})[/dim]")

        if summary_lines:
            console.print(Rule(style="dim"))
            for line in summary_lines: console.print(line)
            console.print()

        # Hallucination warning — fires when an action intent was detected but no
        # tools were used.  Suggestion is tailored to the intent class.
        if _intent_is_action and not tool_calls:
            reminder_lower = reminder.lower()
            if "test" in reminder_lower or "server testing" in reminder_lower:
                suggestion = (
                    "For testing a server, the correct sequence is:\n"
                    "  1. list_running_processes()\n"
                    "  2. run_code('main.py', capture_seconds=5.0, wait_for_exit=False)\n"
                    "  3. wait_for_server('http://localhost:PORT/health')\n"
                    "  4. http_get('http://localhost:PORT/endpoint')\n\n"
                    "Try: 'Use list_running_processes then start the server and test the health endpoint'"
                )
            else:
                suggestion = "Try: 'Use the run_code tool to execute main.py'"
            console.print(Panel(
                f"[yellow]⚠️  The assistant responded without using any tools.[/yellow]\n\n"
                f"The model may have [bold]hallucinated[/bold] the result.\n\n"
                f"{suggestion}",
                title="[yellow]Possible Hallucination[/yellow]", border_style="yellow"))
            console.print()

    if session:
        session.close()


if __name__ == "__main__":
    main()