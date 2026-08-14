"""
app.agent_bridge — wraps LollmsClient + LollmsPersonality creation and the
streaming chat call so the NiceGUI layer never touches lollms_client directly.

This reuses the same handbag/sandbox/system-prompt construction as the
original lollms_code CLI (create_client, ensure_handbag_structure,
create_coding_personality, build_environment_context) so behavior stays
identical — only the I/O layer (terminal -> browser UI) changes.
"""
from __future__ import annotations

import platform
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from gui_prefs import GuiPrefs
from env_config import EnvStore

CODING_SYSTEM_PROMPT_PATH_NOTE = (
    # Import the full CODING_SYSTEM_PROMPT constant from your existing
    # lollms_code CLI module instead of duplicating it here, e.g.:
    #   from lollms_code_cli import CODING_SYSTEM_PROMPT
    # Left as a placeholder import below — point it at your real module.
)

try:
    from lollms_client import LollmsClient
    from lollms_client.lollms_personality import LollmsPersonality
    from lollms_client.lollms_agent.lollms_agent import CapabilityFlags
    from lollms_client.lollms_types import MSG_TYPE, EventMode
except ImportError:
    # Allows the GUI to at least launch (Settings page) before lollms_client
    # is on the path, so users get a friendly error instead of a crash.
    LollmsClient = None
    LollmsPersonality = None
    CapabilityFlags = None
    MSG_TYPE = None
    EventMode = None

# Point this at wherever CODING_SYSTEM_PROMPT actually lives in your project.
# Simplest fix: `from lollms_code.cli import CODING_SYSTEM_PROMPT`
try:
    from lollms_code_cli import CODING_SYSTEM_PROMPT  # type: ignore
except ImportError:
    CODING_SYSTEM_PROMPT = (
        "You are lollms_code, an elite autonomous software engineering agent.\n"
        "(!) CODING_SYSTEM_PROMPT import failed — replace the import in "
        "app/agent_bridge.py with the real path to your CLI module."
    )


class AgentEvent:
    """One item pushed onto the UI queue by the streaming callback."""

    def __init__(self, kind: str, **data: Any):
        self.kind = kind  # "chunk" | "thought" | "tool_start" | "tool_end" |
                           # "artefact_start" | "artefact_end" | "context_update" |
                           # "info" | "done" | "error"
        self.data = data


def build_environment_context(workspace_path: str) -> str:
    is_windows = platform.system() == "Windows"
    os_name = platform.system()
    os_version = platform.version()
    python_version = platform.python_version()
    workspace_root = Path(workspace_path).resolve()

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
                cwd=str(workspace_root), capture_output=True, text=True,
                encoding="utf-8", errors="ignore",
            )
            if result.returncode == 0 and result.stdout.strip():
                git_branch_info = f"\n- Git Branch: {result.stdout.strip()}"
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
1. Use {os_name}-compatible shell commands ({list_cmd} to list, {read_cmd} to read).
2. Always use `{path_sep}` for paths. Stay relative to the Workspace Root.
3. Transient scripts go in `.lollms_code/scripts/` — never the workspace root.
4. Do not modify project files unless explicitly instructed.
=== END ENVIRONMENT CONTEXT ===
"""


def ensure_handbag_structure(prefs: GuiPrefs) -> None:
    handbag_path = Path(prefs.handbag_path)
    handbag_path.mkdir(parents=True, exist_ok=True)
    soul_path = handbag_path / "SOUL.md"
    metadata = {
        "name": "lollms_code",
        "author": "ParisNeo",
        "category": "software_engineering",
        "description": "An elite autonomous software engineering agent.",
        "temperature": str(prefs.temperature),
    }
    yaml_lines = [f"{k}: {v}" for k, v in metadata.items()]
    soul_content = f"---\n{chr(10).join(yaml_lines)}\n---\n\n{CODING_SYSTEM_PROMPT}"
    if not soul_path.exists() or soul_path.read_text(encoding="utf-8") != soul_content:
        soul_path.write_text(soul_content, encoding="utf-8")
    for sub in ("coworkers", "tools", "skills", "memory", "workspace"):
        (handbag_path / sub).mkdir(exist_ok=True)


def ensure_sandbox_structure(prefs: GuiPrefs) -> None:
    sandbox_dir = Path(prefs.workspace_path) / ".lollms_code"
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
        scratchpad.write_text(
            "# Agent Scratchpad\n\nLong-term notes and task state.\n", encoding="utf-8"
        )


def _to_bool(v: Any) -> bool:
    return v.lower().strip() in ("true", "1", "yes", "y") if isinstance(v, str) else bool(v)


def create_client(env: EnvStore, prefs: GuiPrefs):
    """Builds the LollmsClient from the active .env's default LLM profile —
    same resolution CodeAgentConfig.load() used in the CLI — then attaches
    the shell tool binding on top, exactly like the CLI's create_client()."""
    if LollmsClient is None:
        raise RuntimeError(
            "lollms_client is not importable in this environment. "
            "Install/point PYTHONPATH at your package and restart the app."
        )

    resolved = env.resolve_default_connection("llm")
    if not resolved.get("binding_name"):
        raise RuntimeError(
            "No LLM binding configured yet. Open Settings and add a binding + profile."
        )

    llm_config: Dict[str, Any] = {
        "model_name": resolved.get("model_name") or "",
        "host_address": resolved.get("host_address") or "",
        "verify_ssl_certificate": _to_bool(resolved.get("verify_ssl")),
    }
    if resolved.get("api_key"):
        llm_config["service_key"] = resolved["api_key"]

    import lollms_client
    package_root = Path(lollms_client.__file__).resolve().parent
    default_tools_path = package_root / "tools_bindings" / "lcp" / "default_tools"
    tools_folders = [str(default_tools_path)] if default_tools_path.exists() else []

    host_tool_configs = {"system_shell": {"autonomy_level": prefs.shell_autonomy_level}}

    client = LollmsClient(
        llm_binding_name=resolved["binding_name"],
        llm_binding_config=llm_config,
        tools_binding_name="lcp",
        tools_binding_config={
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs,
        },
    )

    if prefs.enable_shell_execution and client.tools:
        try:
            client.tools.mount_tool_library("system_shell")
        except Exception:
            pass

    return client


def create_personality(prefs: GuiPrefs, client):
    ensure_handbag_structure(prefs)
    ensure_sandbox_structure(prefs)

    caps = CapabilityFlags(
        enable_sub_agents=prefs.enable_sub_agents,
        enable_model_switching=prefs.enable_model_switching,
        enable_skill_creation=prefs.enable_skill_creation,
        enable_skill_loading=prefs.enable_skill_loading,
        enable_workspace_tools=True,
        skills_mode=prefs.skills_mode,
        max_sub_agent_depth=prefs.max_sub_agent_depth,
        max_sub_agents_per_turn=prefs.max_sub_agents_per_turn,
    )

    personality = LollmsPersonality.from_handbag(prefs.handbag_path)
    personality.lollms_client = client
    personality.workspace_path = Path(prefs.workspace_path)
    personality.system_prompt = (
        personality.system_prompt + "\n" + build_environment_context(prefs.workspace_path)
    )
    personality.capabilities = caps
    personality.max_tokens_per_turn = prefs.max_tokens_per_turn
    return personality


class QueueStreamingCallback:
    """Same event surface as the CLI's StreamRenderer, but pushes AgentEvent
    objects onto a thread-safe queue instead of printing to the terminal.
    A ui.timer on the GUI side drains this queue and updates widgets."""

    def __init__(self, event_queue: "queue.Queue[AgentEvent]"):
        self.q = event_queue

    def __call__(self, chunk: str, msg_type: Any = None, meta: Optional[Dict] = None) -> bool:
        if MSG_TYPE is None:
            self.q.put(AgentEvent("chunk", text=chunk))
            return True

        mapping = {
            MSG_TYPE.MSG_TYPE_TOOL_START: "tool_start",
            MSG_TYPE.MSG_TYPE_TOOL_END: "tool_end",
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START: "artefact_start",
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END: "artefact_end",
            MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE: "context_update",
        }
        if msg_type in mapping:
            self.q.put(AgentEvent(mapping[msg_type], **(meta or {})))
            return True
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            self.q.put(AgentEvent("chunk", text=chunk, was_processed=bool(meta and meta.get("was_processed"))))
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            self.q.put(AgentEvent("thought", text=chunk))
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            self.q.put(AgentEvent("info", text=chunk))
        return True


def run_agent_turn_in_thread(
    personality, client, prompt: str, prefs: GuiPrefs,
    event_queue: "queue.Queue[AgentEvent]", use_history: bool = True,
) -> threading.Thread:
    """Runs personality.chat(...) in a background thread so the NiceGUI
    event loop never blocks, and reports completion/errors via the queue."""

    callback = QueueStreamingCallback(event_queue)

    def _worker():
        try:
            result = personality.chat(
                prompt=prompt,
                lollms_client=client,
                streaming_callback=callback,
                max_reasoning_steps=prefs.max_reasoning_steps,
                temperature=prefs.temperature,
                n_predict=prefs.max_tokens_per_turn,
                enable_artefacts=True,
                use_internal_history=use_history,
                event_mode=EventMode.FULL_CALLBACK_MODE,
            )
            event_queue.put(AgentEvent("done", result=result))
        except Exception as e:
            event_queue.put(AgentEvent("error", message=str(e)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t