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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    from lollms_client.lollms_personality.lollms_personality import CapabilityFlags
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
    from lollms_client.apps.lollms_code.cli import CODING_SYSTEM_PROMPT, CODING_EXECUTION_HARNESS
except ImportError:
    try:
        from lollms_code_cli import CODING_SYSTEM_PROMPT, CODING_EXECUTION_HARNESS  # type: ignore
    except ImportError:
        CODING_SYSTEM_PROMPT = (
            "You are lollms_code, an elite autonomous software engineering agent."
        )
        CODING_EXECUTION_HARNESS = ""


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

    shell_cmd = "cmd.exe (Windows Command Prompt, NOT PowerShell or bash)" if is_windows else "bash/sh"
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
1. **FILE READING**: Use `<unlock_file>` to read ANY file (text, PDF, DOCX, etc.). Do NOT use shell commands for reading.
2. **SHELL COMMANDS**: Use shell commands only for execution (running tests, git, pip).
   - To execute scripts: Use `python script.py` (not `python3` on Windows)
3. **PATHS**: Always use `{path_sep}` for file paths in shell commands. ALL paths must be relative to the Workspace Root. NEVER attempt to access absolute paths outside the workspace.
4. **TRANSIENT SCRIPTS**: When writing test scripts or temporary files, you MUST save them to the Sandbox Directory (`.lollms_code/scripts/`).
   - Example: `python -c "with open('.lollms_code{path_sep}scripts{path_sep}test.py', 'w') as f: f.write('print(1)')"`
   - NEVER create `.py` or `.log` files in the Workspace Root.
5. **SANDBOX ISOLATION**: The Workspace Root contains the user's actual project. Do not modify project files unless explicitly instructed. Use the Sandbox Directory for all experimental work.
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
    current_plan = sandbox_dir / "CURRENT.md"
    sub_ws_dir = sandbox_dir / "sub_workspace"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sub_ws_dir.mkdir(parents=True, exist_ok=True)
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
    if not current_plan.exists():
        current_plan.write_text(
            "# Current Task\n\nNo active task plan defined yet.\n", encoding="utf-8"
        )


def _to_bool(v: Any) -> bool:
    return v.lower().strip() in ("true", "1", "yes", "y") if isinstance(v, str) else bool(v)


def create_client(env: EnvStore, prefs: GuiPrefs):
    """Builds the LollmsClient from the unified two-tier profile architecture
    (Connection Layer + Execution Layer) across all modalities."""
    if LollmsClient is None:
        raise RuntimeError(
            "lollms_client is not importable in this environment. "
            "Install/point PYTHONPATH at your package and restart the app."
        )

    llm_bindings = env.get_binding_profiles("llm")
    llm_profiles = env.get_model_profiles("llm")

    if not llm_bindings or not llm_profiles:
        raise RuntimeError(
            "No LLM binding configured yet. Open Settings and add a binding + profile."
        )

    import lollms_client
    package_root = Path(lollms_client.__file__).resolve().parent
    default_tools_path = package_root / "tools_bindings" / "lcp" / "default_tools"
    tools_folders = [str(default_tools_path)] if default_tools_path.exists() else []

    host_tool_configs = {
        "system_shell": {"autonomy_level": prefs.shell_autonomy_level},
        "execute_python": {"autonomy_level": prefs.shell_autonomy_level},
    }

    client_kwargs: Dict[str, Any] = {
        "llm_binding_profiles": llm_bindings,
        "llm_model_profiles": llm_profiles,
        "tools_binding_name": "lcp",
        "tools_binding_config": {
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs,
        },
        "debug": prefs.debug,
    }

    # ── Other Modalities (TTI, TTS, STT, TTV, TTM) using unified profiles ──
    for modality in ("tti", "tts", "stt", "ttv", "ttm"):
        b_profs = env.get_binding_profiles(modality)
        m_profs = env.get_model_profiles(modality)
        if b_profs and m_profs:
            client_kwargs[f"{modality}_binding_profiles"] = b_profs
            client_kwargs[f"{modality}_model_profiles"] = m_profs

    client = LollmsClient(**client_kwargs)

    if prefs.enable_shell_execution and client.tools:
        try:
            client.tools.mount_tool_library("system_shell")
        except Exception:
            pass

    # ⚡ Enable fast token estimation (heuristic, no remote tokenizer HTTP calls)
    client.enable_fast_token_estimate()

    return client


def create_personality(prefs: GuiPrefs, client):
    ensure_handbag_structure(prefs)
    ensure_sandbox_structure(prefs)

    has_tti = hasattr(client, 'tti') and client.tti is not None
    has_tts = hasattr(client, 'tts') and client.tts is not None
    has_stt = hasattr(client, 'stt') and client.stt is not None

    caps = CapabilityFlags(
        enable_sub_agents=prefs.enable_sub_agents,
        enable_model_switching=prefs.enable_model_switching,
        enable_skill_creation=prefs.enable_skill_creation,
        enable_skill_loading=prefs.enable_skill_loading,
        enable_workspace_tools=True,
        skills_mode=prefs.skills_mode,
        max_sub_agent_depth=prefs.max_sub_agent_depth,
        max_sub_agents_per_turn=prefs.max_sub_agents_per_turn,
        enable_image_generation=has_tti,
        enable_image_editing=has_tti,
        enable_tts=has_tts,
        enable_stt=has_stt,
    )

    personality = LollmsPersonality.from_handbag(prefs.handbag_path, lollms_client=client)
    personality.lollms_client = client
    personality.workspace_path = Path(prefs.workspace_path)
    personality.system_prompt = (
        personality.system_prompt + "\n" + build_environment_context(prefs.workspace_path)
    )

    if has_tti:
        personality.system_prompt += (
            "\n\n=== IMAGE GENERATION CAPABILITY (ACTIVE) ===\n"
            "You have access to a Text-to-Image (TTI) binding. You CAN generate images.\n"
            "Use the `tool_generate_image` tool to create images from text prompts.\n"
            "Use the `tool_edit_image` tool to modify existing images in the workspace.\n"
            "Generated images are saved to the workspace automatically.\n"
            "When a user asks you to generate, draw, create, or make an image, you MUST use `tool_generate_image`.\n"
            "=== END IMAGE GENERATION CAPABILITY ==="
        )

    if has_tts:
        personality.system_prompt += (
            "\n\n=== TEXT-TO-SPEECH CAPABILITY (ACTIVE) ===\n"
            "You have access to a Text-to-Speech (TTS) binding. You CAN generate speech audio.\n"
            "Use the `tool_text_to_speech` tool to convert text to speech.\n"
            "=== END TEXT-TO-SPEECH CAPABILITY ==="
        )

    if has_stt:
        personality.system_prompt += (
            "\n\n=== SPEECH-TO-TEXT CAPABILITY (ACTIVE) ===\n"
            "You have access to a Speech-to-Text (STT) binding. You CAN transcribe audio.\n"
            "Use the `tool_speech_to_text` tool to transcribe audio files.\n"
            "=== END SPEECH-TO-TEXT CAPABILITY ==="
        )

    personality.capabilities = caps
    personality.max_tokens_per_turn = prefs.max_tokens_per_turn
    personality.debug_mode = prefs.debug

    # Grant autonomous workspace authority for coding tasks (exempt from git prompt blocks)
    object.__setattr__(personality, "_git_autonomy_granted", True)

    # ── Ensure Artefact System is initialized and synced with disk ──
    try:
        if hasattr(personality, "_init_artefact_system"):
            personality._init_artefact_system()
        if hasattr(personality, "_sync_artefact_index_with_disk"):
            personality._sync_artefact_index_with_disk()
    except Exception:
        pass

    # ── Project-Local Memory Setup (matching CLI) ──
    if prefs.enable_memory:
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
            project_memory_db = Path(prefs.workspace_path) / ".lollms_code" / "memory" / "memory.db"
            project_memory_db.parent.mkdir(parents=True, exist_ok=True)
            personality.memory_manager = LollmsMemoryManager(
                db_path=f"sqlite:///{project_memory_db}",
                owner_id=f"project_{Path(prefs.workspace_path).name}",
                config=MemoryConfig(working_token_budget=2000)
            )
        except Exception:
            pass

    if CODING_EXECUTION_HARNESS and "## MACRO STEPS PLANNING (CURRENT.md)" not in personality.system_prompt:
        personality.system_prompt += "\n\n" + CODING_EXECUTION_HARNESS

    sub_ws_instructions = (
        "\n\n=== SUB-WORKSPACE (REFERENCE & DOCUMENTATION) ===\n"
        "You have access to a reference sub-workspace stored in `.lollms_code/sub_workspace/`.\n"
        "This area holds external documentation, reference code, specifications, or datasets that do not belong to the project codebase itself.\n"
        "- Reference files are listed in your prompt under `=== SUB-WORKSPACE (REFERENCE & DOCUMENTATION) ===`.\n"
        "- To load a reference file into your context, use `<unlock_file>sub_workspace/filename.ext</unlock_file>`.\n"
        "- To unload when done, use `<lock_file>sub_workspace/filename.ext</lock_file>`.\n"
        "- You can read and reference these files, but NEVER modify them unless explicitly instructed.\n"
    )
    if "=== SUB-WORKSPACE (REFERENCE & DOCUMENTATION) ===" not in personality.system_prompt:
        personality.system_prompt += sub_ws_instructions

    # ── Universal Skills Discovery (Bundled + Global + Handbag) ──
    collected_skill_dirs = []

    # 1. Project / Repository bundled skills
    repo_skills = Path(__file__).resolve().parent.parent.parent.parent.parent / "skills"
    if repo_skills.exists() and repo_skills.is_dir():
        collected_skill_dirs.append(repo_skills.resolve())

    # 2. Package skills
    import lollms_client
    pkg_skills = Path(lollms_client.__file__).resolve().parent / "skills"
    if pkg_skills.exists() and pkg_skills.is_dir():
        collected_skill_dirs.append(pkg_skills.resolve())

    # 3. User global skills directory
    if prefs.skills_dir and Path(prefs.skills_dir).exists():
        collected_skill_dirs.append(Path(prefs.skills_dir).resolve())

    # 4. Workspace-local skills directory
    ws_skills = Path(prefs.workspace_path) / ".lollms_code" / "skills"
    if ws_skills.exists():
        collected_skill_dirs.append(ws_skills.resolve())

    if personality.skills_manager:
        for s_dir in collected_skill_dirs:
            if s_dir not in [d.resolve() for d in personality.skills_manager._skills_dirs]:
                personality.skills_manager._skills_dirs.append(s_dir)
        personality.skills_manager.reload()
    elif collected_skill_dirs:
        from lollms_client.lollms_personality.skills_manager import SkillsManager
        personality.skills_manager = SkillsManager(skills_dirs=collected_skill_dirs, mode=prefs.skills_mode)

    return personality


def switch_workspace(prefs: GuiPrefs, client, new_workspace_path: str):
    """Equivalent of the CLI's /workspace command: point prefs at the new
    directory, persist it, and rebuild the personality against it."""
    new_path = Path(new_workspace_path).resolve()
    if not new_path.exists() or not new_path.is_dir():
        raise ValueError(f"Not a directory: {new_path}")
    prefs.workspace_path = str(new_path)
    prefs.save()
    return create_personality(prefs, client)


def get_workspace_stats(personality) -> Dict[str, Any]:
    """Same logic as the CLI's get_workspace_stats() — indexed/loaded file
    counts and sizes, used by the /files command."""
    stats: Dict[str, Any] = {"total_indexed": 0, "total_loaded": 0, "loaded_files": []}
    if personality is None:
        return stats
    if not hasattr(personality, "_artefact_manager") or not personality._artefact_manager:
        if hasattr(personality, "_init_artefact_system") and getattr(personality, "_resolved_workspace", None):
            personality._init_artefact_system()
    if not hasattr(personality, "_artefact_manager") or not personality._artefact_manager:
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
                    stats["loaded_files"].append({"path": rel_path, "size": file_size})
        stats["total_loaded"] = len(stats["loaded_files"])
    except Exception:
        pass
    return stats


def change_file_visibility(personality, targets: list, action: str) -> Dict[str, Any]:
    """Wraps personality.change_file_visibility(), same as the CLI's
    /load, /unload, /lock, /hide, /unhide commands. `action` is one of
    'load', 'unload', 'lock', 'hide', 'unhide'."""
    if personality is not None:
        if not hasattr(personality, "_artefact_manager") or not personality._artefact_manager:
            if hasattr(personality, "_init_artefact_system") and getattr(personality, "_resolved_workspace", None):
                personality._init_artefact_system()
    result = personality.change_file_visibility(targets, action) if personality else {"status_str": "Personality not ready."}
    try:
        object.__setattr__(personality, "_last_ws_sync_time", 0.0)
    except Exception:
        pass
    return result


def clear_all_loaded_files(personality) -> Dict[str, Any]:
    """Same as the CLI's /clear-files: unloads every currently [C]-loaded
    file from context in one shot."""
    if personality is not None:
        if not hasattr(personality, "_artefact_manager") or not personality._artefact_manager:
            if hasattr(personality, "_init_artefact_system") and getattr(personality, "_resolved_workspace", None):
                personality._init_artefact_system()
    if not hasattr(personality, "_artefact_manager") or not personality._artefact_manager:
        return {"status_str": "Artefact system not initialized."}
    from lollms_client.lollms_artefact import ArtefactVisibility
    all_arts = personality._artefact_manager._get_all_raw()
    loaded_files = [
        a.get("title", "") for a in all_arts
        if a.get("visibility") == ArtefactVisibility.FULL and not a.get("title", "").endswith("::images")
    ]
    if not loaded_files:
        return {"status_str": "No files are currently loaded in context."}
    return change_file_visibility(personality, loaded_files, "unload")


def get_scratchpad_content(personality, workspace_path: Optional[str] = None) -> str:
    """Reads the live scratchpad file for the current workspace so the GUI
    can display the agent's persistent notes and intermediate thoughts."""
    scratchpad_path = getattr(personality, "_scratchpad_path", None)
    if scratchpad_path and Path(scratchpad_path).exists():
        try:
            return Path(scratchpad_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    ws = getattr(personality, "_resolved_workspace", None) or getattr(personality, "workspace_path", None) or workspace_path
    if ws:
        p = Path(ws) / ".lollms_code" / "scratchpad.md"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return ""


def get_current_plan_content(personality=None, workspace_path: Optional[str] = None) -> str:
    """Reads the live CURRENT.md plan file for the current workspace so the GUI
    can display the agent's macro-steps plan."""
    ws = None
    if personality is not None:
        ws = getattr(personality, "_resolved_workspace", None) or getattr(personality, "workspace_path", None)
    if not ws and workspace_path:
        ws = Path(workspace_path)

    if not ws:
        return ""

    plan_path = Path(ws).resolve() / ".lollms_code" / "CURRENT.md"
    if not plan_path.exists():
        return ""
    try:
        return plan_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def get_live_skills(personality) -> List[Dict[str, Any]]:
    """Returns the personality's current skills list with handbag provenance."""
    if personality is None:
        return []
    if hasattr(personality, "list_skills_structured"):
        try:
            return personality.list_skills_structured() or []
        except Exception:
            pass
    mgr = getattr(personality, "skills_manager", None)
    if mgr is None:
        return []
    try:
        return mgr.list_skills() or []
    except Exception:
        return []


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
            MSG_TYPE.MSG_TYPE_ARTEFACT_SYMBOL_DETECTED: "artefact_symbol",
            MSG_TYPE.MSG_TYPE_ROUND_START: "round_start",
            MSG_TYPE.MSG_TYPE_ROUND_END: "round_end",
            MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE: "context_update",
            MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE: "scratchpad_update",
            MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START: "worker_spawn_start",
            MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END: "worker_spawn_end",
        }
        if msg_type in mapping:
            self.q.put(AgentEvent(mapping[msg_type], **(meta or {})))
            return True
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            is_internal_chunk = bool(
                meta and (
                    meta.get("was_processed")
                    or meta.get("live_tool_chunk")
                    or meta.get("live_artifact_chunk")
                )
            )
            # In FULL_CALLBACK_MODE, suppress internal streaming chunks, raw tool tags, and processing tags
            if is_internal_chunk or (chunk and ("<processing" in chunk or "</processing>" in chunk or "<!-- status:" in chunk or "<tool>" in chunk or "</tool>" in chunk)):
                return True
            self.q.put(AgentEvent("chunk", text=chunk, was_processed=False))
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            self.q.put(AgentEvent("thought", text=chunk))
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            self.q.put(AgentEvent("info", text=chunk))
        return True


def cancel_agent_turn(personality, client=None) -> bool:
    """Cancels an active agent turn across personality, client, and low-level LLM bindings."""
    global _CURRENT_RESP_QUEUE
    if _CURRENT_RESP_QUEUE is not None:
        try:
            _CURRENT_RESP_QUEUE.put(("reject", "Generation cancelled by user."))
        except Exception:
            pass
        _CURRENT_RESP_QUEUE = None

    cancelled = False
    if personality is not None:
        if hasattr(personality, "cancel_generation"):
            personality.cancel_generation()
            cancelled = True
        elif hasattr(personality, "cancel"):
            personality.cancel()
            cancelled = True

    if client is not None:
        if hasattr(client, "cancel"):
            try:
                client.cancel()
                cancelled = True
            except Exception:
                pass
        if hasattr(client, "llm") and hasattr(client.llm, "cancel"):
            try:
                client.llm.cancel()
                cancelled = True
            except Exception:
                pass
    elif personality is not None and getattr(personality, "lollms_client", None) is not None:
        lc = personality.lollms_client
        if hasattr(lc, "cancel"):
            try:
                lc.cancel()
                cancelled = True
            except Exception:
                pass
        if hasattr(lc, "llm") and hasattr(lc.llm, "cancel"):
            try:
                lc.llm.cancel()
                cancelled = True
            except Exception:
                pass

    return cancelled


_CURRENT_RESP_QUEUE: Optional[queue.Queue] = None


def make_gui_python_confirm_handler(event_queue: "queue.Queue[AgentEvent]", prefs: GuiPrefs):
    """
    Creates a confirmation handler for LCP execution tools that dispatches a modal dialog
    request to NiceGUI and blocks the worker thread until the user decides.
    """
    def _handler(*args, **kwargs) -> Tuple[str, str]:
        global _CURRENT_RESP_QUEUE
        if getattr(prefs, "auto_approve_python", False):
            return "allow", ""

        source = ""
        script_label = "script.py"
        argv = None

        if len(args) == 1 and isinstance(args[0], dict):
            req = args[0]
            source = req.get("source") or req.get("content") or ""
            script_label = req.get("script_label") or req.get("label") or "script.py"
            argv = req.get("argv") or req.get("metadata", {}).get("argv")
        elif len(args) >= 2:
            source = str(args[0])
            script_label = str(args[1])
            if len(args) > 2:
                argv = args[2]
        elif kwargs:
            source = kwargs.get("source") or kwargs.get("content") or ""
            script_label = kwargs.get("script_label") or kwargs.get("label") or "script.py"
            argv = kwargs.get("argv")

        resp_queue = queue.Queue(maxsize=1)
        _CURRENT_RESP_QUEUE = resp_queue

        event_queue.put(AgentEvent(
            "python_approval_request",
            source=source,
            script_label=script_label,
            argv=argv,
            response_queue=resp_queue,
        ))

        try:
            # Wait for user decision with safety timeout to avoid hanging if the frontend has no interaction
            decision, reason = resp_queue.get(timeout=180.0)
            _CURRENT_RESP_QUEUE = None
            if decision == "always":
                # Enable auto-approval in-memory for this active session only (do not persist to disk)
                prefs.auto_approve_python = True
            return decision, reason
        except queue.Empty:
            _CURRENT_RESP_QUEUE = None
            return "reject", "Authorization timed out: no operator response received from the user interface."
        except Exception as e:
            _CURRENT_RESP_QUEUE = None
            return "reject", f"Approval interrupted or no user interaction channel available: {e}"

    return _handler


def run_agent_turn_in_thread(
    personality, client, prompt: str, prefs: GuiPrefs,
    event_queue: "queue.Queue[AgentEvent]", use_history: bool = True,
) -> threading.Thread:
    """Runs personality.chat(...) in a background thread so the NiceGUI
    event loop never blocks, and reports completion/errors via the queue."""

    gui_confirm_handler = make_gui_python_confirm_handler(event_queue, prefs)

    # Register GUI validation handler with execute_python and system_shell across all tool modules
    if client and hasattr(client, "tools") and client.tools:
        if hasattr(client.tools, "set_confirm_handler"):
            client.tools.set_confirm_handler(gui_confirm_handler)
        if hasattr(client.tools, "host_tool_configs") and isinstance(client.tools.host_tool_configs, dict):
            client.tools.host_tool_configs.setdefault("execute_python", {})["confirm_handler"] = gui_confirm_handler
            client.tools.host_tool_configs.setdefault("system_shell", {})["confirm_handler"] = gui_confirm_handler

    try:
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as _ep_mod
        _ep_mod.set_confirm_handler(gui_confirm_handler)
        _ep_mod._AUTO_APPROVE_PYTHON = getattr(prefs, "auto_approve_python", False)
    except Exception:
        pass

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
                context_compaction_threshold=getattr(prefs, "context_compaction_threshold", 0.85),
                enable_artefacts=True,
                use_internal_history=use_history,
                enable_shell=getattr(prefs, "enable_shell_execution", True),
                enable_python_exec=True,
                enable_workspace_tools=True,
                enforce_end_tag=True,
                event_mode=EventMode.FULL_CALLBACK_MODE,
                shell_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
                python_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
                auto_approve_python=getattr(prefs, "auto_approve_python", False),
                confirm_handler=gui_confirm_handler,
                debug=prefs.debug,
                debug_export=prefs.debug,
            )
            event_queue.put(AgentEvent("done", result=result))
        except Exception as e:
            event_queue.put(AgentEvent("error", message=str(e)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t