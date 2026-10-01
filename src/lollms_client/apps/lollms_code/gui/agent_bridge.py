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
import re
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
            "You are lollms_code, an elite autonomous software engineering agent.\n\n"
            "## CORE OPERATIONAL DIRECTIVES\n"
            "1. **SAME-RESPONSE ACTION EXECUTION**: Prose does NOT execute tools or modify files. Emit functional tags (`<tool>`, `<artifact>`, `<unlock_file>`) in the exact same response immediately.\n"
            "2. **SKILL-FIRST MANDATE**: When a task matches an available skill (e.g. `file_organization`), call `tool_load_skill` in Round 1 before taking ad-hoc steps.\n"
            "3. **PASSIVE MEMORY BOUNDARY**: Memories provide background context and user preferences only. On casual greetings, reply conversationally first, then emit `<done/>`.\n"
            "4. **WORKSPACE FILES**: Use `<unlock_file>` to load files into context [C], and `<lock_file>` to unload when finished.\n"
            "5. **ARTIFACTS & PATCHES**: Create files with `<artifact name=\"...\" type=\"code\">...code...</artifact>`. For updates, use Aider SEARCH/REPLACE patches.\n"
            "6. **COMPLETION CONTRACT**: Conclude all completed tasks with `<done/>` on a new line.\n"
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
    python_version = platform.python_version()
    workspace_root = Path(workspace_path).resolve()
    shell_cmd = "cmd.exe" if is_windows else "bash/sh"
    path_sep = "\\" if is_windows else "/"

    git_branch_info = ""
    git_dir = workspace_root / ".git"
    if git_dir.exists():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(workspace_root), capture_output=True, text=True,
                encoding="utf-8", errors="ignore", timeout=1.5
            )
            if result.returncode == 0 and result.stdout.strip():
                git_branch_info = f" | Git Branch: `{result.stdout.strip()}`"
        except Exception:
            pass

    return f"""
=== ENVIRONMENT CONTEXT ===
- OS: {os_name} | Python: {python_version} | Shell: `{shell_cmd}` | Separator: `{path_sep}`{git_branch_info}
- Execution: Use `python script.py` to run scripts. All relative paths resolve from workspace root '.'.
- Transient Scripts: Place temporary experiments in `.lollms_code{path_sep}scripts{path_sep}`.
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
    }
    yaml_lines = [f"{k}: {v}" for k, v in metadata.items()]
    soul_content = f"---\n{chr(10).join(yaml_lines)}\n---\n\n{CODING_SYSTEM_PROMPT}"
    if not soul_path.exists() or soul_path.read_text(encoding="utf-8") != soul_content:
        soul_path.write_text(soul_content, encoding="utf-8")
    for sub in ("coworkers", "tools", "skills", "memory", "workspace"):
        (handbag_path / sub).mkdir(exist_ok=True)

    # Seed modular skills from project root into handbag and ensure loadable visibility
    try:
        def _find_root() -> Path:
            p = Path(__file__).resolve().parent
            for parent in [p] + list(p.parents):
                if (parent / "pyproject.toml").exists():
                    return parent
            return Path.cwd().resolve()

        root_dir = _find_root()
        candidate_sources = [
            root_dir / "skills",
            Path(__file__).resolve().parents[4] / "skills",
            Path(__file__).resolve().parent.parent / "skills",
            Path.home() / ".lollms_client" / "skills",
        ]

        hb_skills = handbag_path / "skills"
        hb_skills.mkdir(parents=True, exist_ok=True)

        for src_dir in candidate_sources:
            if src_dir.exists() and src_dir.is_dir():
                for s_dir in src_dir.iterdir():
                    if s_dir.is_dir() and (s_dir / "SKILL.md").exists():
                        dest_d = hb_skills / s_dir.name
                        dest_d.mkdir(parents=True, exist_ok=True)
                        dest_f = dest_d / "SKILL.md"
                        content = (s_dir / "SKILL.md").read_text(encoding="utf-8")
                        # Enforce that all seeded skills default to loadable
                        if "visibility: visible" in content or "always_visible: true" in content:
                            content = re.sub(r'visibility:\s*visible', 'visibility: loadable', content)
                            content = re.sub(r'always_visible:\s*true', 'always_visible: false', content)
                        dest_f.write_text(content, encoding="utf-8")
    except Exception:
        pass


def ensure_sandbox_structure(prefs: GuiPrefs) -> None:
    sandbox_dir = Path(prefs.workspace_path) / ".lollms_code"
    scripts_dir = sandbox_dir / "scripts"
    scratchpad = sandbox_dir / "scratchpad.md"
    current_plan = sandbox_dir / "CURRENT.md"
    sub_ws_dir = sandbox_dir / "sub_workspace"
    ws_tools_dir = sandbox_dir / "tools"
    ws_skills_dir = sandbox_dir / "skills"
    ws_handbags_dir = sandbox_dir / "handbags"

    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sub_ws_dir.mkdir(parents=True, exist_ok=True)
    ws_tools_dir.mkdir(parents=True, exist_ok=True)
    ws_skills_dir.mkdir(parents=True, exist_ok=True)
    ws_handbags_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    # Scratchpad is strictly ephemeral per session
    scratchpad.write_text(
        "# Scratchpad\n\n(Empty - session notes only)\n", encoding="utf-8"
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

    # ── Register Global and Project-Local Tools Directories ──
    global_tools_dir = Path.home() / ".lollms_client" / "lollms_code" / "tools"
    if global_tools_dir.exists():
        tools_folders.append(str(global_tools_dir.resolve()))

    ws_tools_dir = Path(prefs.workspace_path) / ".lollms_code" / "tools"
    if ws_tools_dir.exists():
        tools_folders.append(str(ws_tools_dir.resolve()))

    host_tool_configs = {
        "system_shell": {"autonomy_level": prefs.shell_autonomy_level},
        "execute_python": {"autonomy_level": prefs.shell_autonomy_level},
    }

    lollms_system_dir = (Path.home() / ".lollms_client").resolve()
    lollms_system_dir.mkdir(parents=True, exist_ok=True)

    client_kwargs: Dict[str, Any] = {
        "system_dir": str(lollms_system_dir),
        "llm_binding_profiles": llm_bindings,
        "llm_model_profiles": llm_profiles,
        "tools_binding_name": "lcp",
        "tools_binding_config": {
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs,
            "system_dir": str(lollms_system_dir),
            "cwd": str(lollms_system_dir),
        },
        "debug": prefs.debug,
    }

    # ── Other Modalities (TTI, TTS, STT, CONNECTION, RAG) using unified profiles ──
    # Exclude media synthesis modalities (TTM/TTV) to prevent unwanted background daemon spawns on workspace load
    for modality in ("tti", "tts", "stt", "connection", "rag"):
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
    has_rag = (hasattr(client, 'rag') and client.rag is not None) or bool(getattr(client, 'rag_model_profiles_registry', None))

    caps = CapabilityFlags(
        enable_sub_agents=prefs.enable_sub_agents,
        enable_model_switching=prefs.enable_model_switching,
        enable_skill_creation=prefs.enable_skill_creation,
        enable_skill_loading=prefs.enable_skill_loading,
        enable_workspace_tools=True,
        allow_computer_use=getattr(prefs, "allow_computer_use", False) or getattr(prefs, "enable_computer_use", False),
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

    if has_rag:
        personality.system_prompt += (
            "\n\n=== RAG KNOWLEDGE BASE CAPABILITY (ACTIVE) ===\n"
            "You have access to a persistent RAG knowledge base & semantic store via the `safe_store` binding.\n"
            "Use `tool_query_rag` to execute dense vector + BM25 hybrid searches over indexed documents.\n"
            "Use `tool_sparql_query` to query the knowledge graph using W3C SPARQL 1.1.\n"
            "Use `tool_add_document_to_rag` to index workspace files into the knowledge base.\n"
            "=== END RAG KNOWLEDGE BASE CAPABILITY ==="
        )

    personality.capabilities = caps
    personality.max_tokens_per_turn = prefs.max_tokens_per_turn
    personality.debug_mode = prefs.debug

    # Grant autonomous workspace authority for coding tasks (exempt from git prompt blocks)
    object.__setattr__(personality, "_git_autonomy_granted", True)

    # ── Ensure Artefact System is initialized ──
    try:
        if hasattr(personality, "_init_artefact_system"):
            personality._init_artefact_system()
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
            personality.memory_manager.deduplicate_all()
            personality.memory_manager.clean_task_backlog_memories()
        except Exception:
            pass
    else:
        personality.memory_manager = None


    # ── Universal Skills Discovery (Bundled + Global + Handbag) ──
    collected_skill_dirs = []

    # 1. Synchronize package skills into user home (~/.lollms_client/skills/)
    try:
        from lollms_client.apps.lollms_code.cli import sync_default_skills_to_user_home
        sync_default_skills_to_user_home()
    except Exception:
        pass

    # 2. User home global skills directory (~/.lollms_client/skills/)
    global_user_skills = (Path.home() / ".lollms_client" / "skills").resolve()
    if global_user_skills.exists():
        collected_skill_dirs.append(global_user_skills)

    # 3. Custom skills directory from prefs if distinct
    if prefs.skills_dir and Path(prefs.skills_dir).exists():
        p_c = Path(prefs.skills_dir).resolve()
        if p_c not in collected_skill_dirs:
            collected_skill_dirs.append(p_c)

    # 4. Workspace-local skills directory (.lollms_code/skills/)
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


def get_context_preview(*args, **kwargs) -> Dict[str, Any]:
    """
    Generates a full diagnostic preview of the exact messages, system prompt,
    active tools, and generation parameters that will be sent to the LLM backend.

    Supports both:
      - get_context_preview(session, prefs, prompt_text="")
      - get_context_preview(personality, client, prefs, prompt_text="")
    """
    personality = None
    client = None
    prefs = None
    prompt_text = ""
    session_obj = None

    if len(args) == 4:
        personality, client, prefs, prompt_text = args
    elif len(args) == 3:
        if hasattr(args[0], "ensure_ready") or hasattr(args[0], "personality"):
            session_obj = args[0]
            session_obj.ensure_ready()
            personality = session_obj.personality
            client = session_obj.client
            prefs = args[1]
            prompt_text = args[2]
        else:
            personality, client, prefs = args
            prompt_text = kwargs.get("prompt_text", "")
    elif len(args) == 2:
        if hasattr(args[0], "ensure_ready") or hasattr(args[0], "personality"):
            session_obj = args[0]
            session_obj.ensure_ready()
            personality = session_obj.personality
            client = session_obj.client
            prefs = args[1]
        else:
            personality, client = args
            prefs = kwargs.get("prefs")
    elif len(args) == 1:
        session_candidate = args[0]
        if hasattr(session_candidate, "ensure_ready"):
            session_obj = session_candidate
            session_obj.ensure_ready()
            personality = session_obj.personality
            client = session_obj.client
            prefs = getattr(session_obj, "prefs", None)
        else:
            personality = session_candidate

    if personality is None and "personality" in kwargs:
        personality = kwargs["personality"]
    if client is None and "client" in kwargs:
        client = kwargs["client"]
    if prefs is None and "prefs" in kwargs:
        prefs = kwargs["prefs"]
    if not prompt_text and "prompt_text" in kwargs:
        prompt_text = kwargs["prompt_text"]

    if prefs is None:
        try:
            prefs = GuiPrefs.load()
        except Exception:
            prefs = GuiPrefs()

    resolved_llm = {}
    try:
        from lollms_client.lollms_config_api import load_config_map
        cfg_map = load_config_map()
        alias = getattr(personality, "_active_llm_alias", None)
        if not alias and hasattr(client, "_active_llm_alias"):
            alias = client._active_llm_alias
        resolved_llm["alias"] = alias or "default"
        if hasattr(client, "llm") and client.llm:
            resolved_llm["model_name"] = getattr(client.llm, "model_name", "unknown")
            resolved_llm["binding_name"] = getattr(client.llm, "binding_name", "unknown")
            resolved_llm["host_address"] = getattr(client.llm, "host_address", "http://localhost")
    except Exception:
        resolved_llm = {
            "alias": "default",
            "model_name": getattr(getattr(client, "llm", None), "model_name", "unknown"),
            "binding_name": getattr(getattr(client, "llm", None), "binding_name", "unknown"),
            "host_address": "unknown",
        }

    # Discover active tools
    active_tools = personality._discover_tools(
        enable_data_tools=True,
        enable_workspace_tools=True,
        enable_shell=getattr(prefs, "enable_shell_execution", True),
        enable_python_exec=True,
        enable_web_tools=True,
        auto_load_document_editor=True,
        enable_computer_use=False,
        shell_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
        python_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
        auto_approve_python=getattr(prefs, "auto_approve_python", False),
    )

    stable_system_prompt = personality._build_system_prompt(active_tools, dynamic_effort=getattr(prefs, "dynamic_effort", False))
    user_prof = personality._build_user_profile_context()
    if user_prof:
        stable_system_prompt += user_prof

    dynamic_suffix_parts = []
    ws_ctx = personality._build_workspace_context_block()
    if ws_ctx:
        dynamic_suffix_parts.append(ws_ctx.strip())

    scratchpad_ctx = personality._build_scratchpad_context()
    if scratchpad_ctx:
        dynamic_suffix_parts.append(scratchpad_ctx.strip())

    mem_working = ""
    mem_handles = ""
    mem_count = {"working": 0, "deep": 0, "archived": 0, "total": 0}
    if personality.memory_manager:
        try:
            mem_working = personality.memory_manager.build_working_zone() or "(No active Level 1 memories yet)"
            mem_handles = personality.memory_manager.build_handles_zone() or "(No Level 2 deep memory handles yet)"
            all_mems = personality.memory_manager.list_all(level=None, page=1, page_size=0, ignore_owner=True)
            mem_list = all_mems.get("memories", [])
            mem_count["total"] = len(mem_list)
            for m in mem_list:
                lvl = m.get("level", 1)
                if lvl == 1: mem_count["working"] += 1
                elif lvl == 2: mem_count["deep"] += 1
                elif lvl >= 3: mem_count["archived"] += 1
        except Exception as ex:
            mem_working = f"(Memory error: {ex})"
            mem_handles = ""

        if mem_working and "(No active Level 1 memories yet)" not in mem_working:
            dynamic_suffix_parts.append(mem_working.strip())
        if mem_handles and "(No Level 2 deep memory handles yet)" not in mem_handles:
            dynamic_suffix_parts.append(mem_handles.strip())

    plan_ctx = personality._build_current_plan_context()
    if plan_ctx:
        dynamic_suffix_parts.append(plan_ctx.strip())

    dynamic_suffix = "\n\n".join(dynamic_suffix_parts)
    if dynamic_suffix:
        stable_system_prompt += "\n\n" + dynamic_suffix

    base_conversation = list(personality._conversation) if personality and hasattr(personality, "_conversation") else []
    if not base_conversation and session_obj and hasattr(session_obj, "reconstruct_conversation_from_debug_log"):
        reconstructed = session_obj.reconstruct_conversation_from_debug_log()
        if reconstructed:
            if personality:
                personality._conversation = reconstructed
            base_conversation = list(reconstructed)

    effective_prompt = prompt_text.strip() if prompt_text else ""
    if effective_prompt:
        base_conversation.append({"role": "user", "content": effective_prompt})
    elif not base_conversation or base_conversation[-1].get("role") != "user":
        base_conversation.append({"role": "user", "content": "(Awaiting user prompt in input box...)"})

    from lollms_client.lollms_personality.lollms_personality import _HistoryContextAdapter, _normalize_messages
    from lollms_client.lollms_history import HistoryManager

    context_adapter = _HistoryContextAdapter(personality, stable_system_prompt)
    messages = HistoryManager.export(
        context=context_adapter,
        format_type="openai_chat",
        branch=base_conversation,
        virtual_history=[],
        system_prompt_override=stable_system_prompt
    )
    messages = _normalize_messages(messages)

    # Token counting
    total_tokens = 0
    tokenized_messages = []
    for msg in messages:
        c = msg.get("content", "")
        c_str = c if isinstance(c, str) else json.dumps(c, default=str)
        t_count = client.count_tokens(c_str) if hasattr(client, "count_tokens") else len(c_str) // 4
        total_tokens += t_count
        tokenized_messages.append({
            "role": msg.get("role", "user"),
            "content": c,
            "tokens": t_count,
        })

    max_ctx = getattr(client, "get_ctx_size", lambda: 8192)() or 8192
    fill_pct = round((total_tokens / max_ctx) * 100, 1)

    effort_display = (
        "Dynamic (Auto-scaling)"
        if getattr(prefs, "dynamic_effort", False)
        else (getattr(prefs, "reasoning_effort", None) or "Model Default")
    )

    # Build complete verbatim assembled context as rendered for the LLM
    assembled_parts = []
    for idx, msg in enumerate(messages):
        r = msg.get("role", "user").upper()
        c = msg.get("content", "")
        if isinstance(c, list):
            c_str = "\n".join(
                item.get("text", "") for item in c
                if isinstance(item, dict) and item.get("type") == "text"
            )
            if any(isinstance(item, dict) and item.get("type") == "image_url" for item in c):
                c_str += "\n[IMAGE ATTACHED]"
        else:
            c_str = str(c)
        assembled_parts.append(f"==================== [{idx}] ROLE: {r} ====================\n{c_str}\n")

    full_assembled_context = "\n".join(assembled_parts)

    return {
        "configuration": {
            "model_alias": resolved_llm.get("alias", "default"),
            "model_name": resolved_llm.get("model_name", "unknown"),
            "binding_name": resolved_llm.get("binding_name", "unknown"),
            "host_address": resolved_llm.get("host_address", "http://localhost"),
            "temperature": prefs.temperature,
            "max_tokens_per_turn": prefs.max_tokens_per_turn,
            "max_reasoning_steps": "∞ (Infinite)" if prefs.max_reasoning_steps <= 0 else prefs.max_reasoning_steps,
            "reasoning_effort": effort_display,
            "dynamic_effort": getattr(prefs, "dynamic_effort", False),
            "memory_enabled": getattr(prefs, "enable_memory", True),
            "memory_manager_attached": bool(personality.memory_manager),
            "memory_db_path": getattr(getattr(personality, "memory_manager", None), "resolved_disk_path", "None"),
            "memory_counts": mem_count,
            "shell_autonomy": prefs.shell_autonomy_level,
            "auto_approve_python": getattr(prefs, "auto_approve_python", False),
            "workspace_path": prefs.workspace_path,
            "handbag_name": getattr(personality, "name", "lollms_code"),
            "handbag_path": prefs.handbag_path,
            "total_tokens": total_tokens,
            "max_ctx": max_ctx,
            "fill_pct": fill_pct,
        },
        "full_assembled_context": full_assembled_context,
        "messages": tokenized_messages,
        "system_prompt": stable_system_prompt,
        "active_tools": active_tools,
        "memory_working_zone": mem_working,
        "memory_handles_zone": mem_handles,
        "workspace_tree": ws_ctx,
    }


def switch_persona_handbag(prefs: GuiPrefs, client, handbag_path: str):
    """Switches the active persona handbag and rebuilds the personality."""
    target_p = Path(handbag_path).resolve()
    if not target_p.exists() or not target_p.is_dir():
        raise ValueError(f"Handbag folder does not exist: {target_p}")
    prefs.handbag_path = str(target_p)
    prefs.save()
    return create_personality(prefs, client)


def get_subws_tools_and_skills(personality, prefs: GuiPrefs, client=None) -> Dict[str, Any]:
    """
    Assembles a complete, structured view of active session capabilities,
    strictly segregating native Handbag assets from Project Extra assets (.lollms_code/).
    """
    ws_root = Path(prefs.workspace_path).resolve()
    hb_root = Path(prefs.handbag_path).resolve() if prefs.handbag_path else None
    default_hb_root = (Path.home() / ".lollms_client" / "lollms_code" / "handbags" / "default_coder").resolve()

    # 1. Persona State
    p_name = getattr(personality, "name", "lollms_code")
    p_cat = getattr(personality, "category", "software_engineering")
    p_desc = getattr(personality, "description", "")
    p_soul = ""
    soul_p = hb_root / "SOUL.md" if hb_root else None
    if soul_p and soul_p.exists():
        try:
            p_soul = soul_p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    persona_source = "handbag"
    if hb_root:
        if hb_root == default_hb_root:
            persona_source = "default"
        elif str(hb_root).startswith(str(ws_root / ".lollms_code")):
            persona_source = "project"
        elif ".lollms_client" in str(hb_root):
            persona_source = "global"

    # 2. Tools Segregation (with canonical deduplication)
    handbag_tools: List[Dict[str, Any]] = []
    project_tools: List[Dict[str, Any]] = []
    builtin_tools: List[Dict[str, Any]] = []

    raw_tools = []
    if personality and hasattr(personality, "list_tools_structured"):
        try:
            raw_tools = personality.list_tools_structured()
        except Exception:
            pass

    seen_tool_keys = set()
    for t in raw_tools:
        name = t.get("name", "")
        seen_tool_keys.add(name.lower())
        src_file = t.get("source_file") or ""
        if src_file:
            p_obj = Path(src_file).resolve()
            seen_tool_keys.add(p_obj.stem.lower())
            seen_tool_keys.add(p_obj.name.lower())
            seen_tool_keys.add(p_obj.parent.name.lower())
            seen_tool_keys.add(str(p_obj).lower())

        is_in_project_tools = bool(src_file and str(ws_root / ".lollms_code" / "tools").lower() in str(src_file).lower())
        is_in_handbag_tools = bool(t.get("is_handbag") or (hb_root and src_file and str(hb_root).lower() in str(src_file).lower()))

        if is_in_handbag_tools:
            handbag_tools.append(t)
        elif is_in_project_tools:
            project_tools.append(t)
        else:
            # All other tools (LCP default tools, document editor, execution, spinoff sub-agents, etc.) are built-in
            builtin_tools.append(t)

    # Disk scan for project extra tools strictly in .lollms_code/tools
    proj_tools_dir = ws_root / ".lollms_code" / "tools"
    if proj_tools_dir.exists():
        for entry in sorted(proj_tools_dir.iterdir()):
            if entry.name.startswith(".") or entry.name in ("__pycache__",):
                continue
            entry_res = entry.resolve()
            item_name = entry.stem if entry.is_file() else entry.name

            # Check if any tool in project_tools already matches this path or name
            already_tracked = any(
                p_tool.get("name") == item_name
                or p_tool.get("source_file") == str(entry_res)
                or (p_tool.get("source_file") and str(entry_res) in p_tool.get("source_file"))
                for p_tool in project_tools
            )

            if already_tracked:
                continue

            desc = f"Project tool package in .lollms_code/tools/{entry.name}"
            doc_p = entry / "README.md" if entry.is_dir() else None
            py_p = entry / f"{entry.name}.py" if entry.is_dir() else entry

            if doc_p and doc_p.exists():
                try:
                    for line in doc_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if line.strip() and not line.startswith("#"):
                            desc = line.strip()[:140]
                            break
                except Exception:
                    pass

            project_tools.append({
                "name": item_name,
                "description": desc,
                "is_handbag": False,
                "source": "project_extra",
                "source_file": str(entry_res),
                "parameters": [],
            })

    # 3. Skills Segregation (with title, slug, and canonical path deduplication)
    handbag_skills: List[Dict[str, Any]] = []
    project_skills: List[Dict[str, Any]] = []
    other_skills: List[Dict[str, Any]] = []

    raw_skills = []
    if personality and hasattr(personality, "list_skills_structured"):
        try:
            raw_skills = personality.list_skills_structured(include_content=True)
        except Exception:
            pass

    seen_skill_keys = set()
    for s in raw_skills:
        title = s.get("title", "").strip()
        fp = s.get("file_path") or ""
        canon_title = title.lower()

        canon_path = str(Path(fp).resolve()).lower() if fp else ""
        if canon_title in seen_skill_keys or (canon_path and canon_path in seen_skill_keys):
            continue

        seen_skill_keys.add(canon_title)
        if canon_path:
            seen_skill_keys.add(canon_path)

        resolved_fp = str(Path(fp).resolve()) if fp else ""
        resolved_hb = str(hb_root.resolve()) if hb_root else ""
        resolved_ws = str((ws_root / ".lollms_code" / "skills").resolve())

        if s.get("is_handbag") or (resolved_hb and resolved_fp and resolved_hb in resolved_fp):
            handbag_skills.append(s)
        elif s.get("source") == "workspace" or (resolved_fp and resolved_ws in resolved_fp):
            project_skills.append(s)
        else:
            other_skills.append(s)

    # Disk scan for project extra skills in .lollms_code/skills
    proj_skills_dir = ws_root / ".lollms_code" / "skills"
    if proj_skills_dir.exists():
        for entry in sorted(proj_skills_dir.iterdir()):
            if entry.name.startswith(".") or entry.name in ("__pycache__", "README.md"):
                continue
            entry_res = entry.resolve()
            skill_slug = entry.stem if entry.is_file() else entry.name

            # Skip if already tracked by in-memory registry
            if (
                skill_slug.lower() in seen_skill_keys
                or str(entry_res).lower() in seen_skill_keys
                or str(entry_res / "SKILL.md").lower() in seen_skill_keys
            ):
                continue

            skill_md = entry / "SKILL.md" if entry.is_dir() else entry
            content_preview = ""
            desc = "Project skill"
            final_title = skill_slug

            if skill_md.exists():
                try:
                    content_preview = skill_md.read_text(encoding="utf-8", errors="ignore")
                    # Check YAML frontmatter for real title
                    if content_preview.startswith("---"):
                        fm_match = re.match(r"^---\n(.*?)\n---", content_preview, re.DOTALL)
                        if fm_match:
                            for line in fm_match.group(1).splitlines():
                                if line.startswith("title:"):
                                    final_title = line.split(":", 1)[1].strip().strip('"\'')
                                elif line.startswith("description:"):
                                    desc = line.split(":", 1)[1].strip().strip('"\'')

                    if desc == "Project skill":
                        for line in content_preview.splitlines():
                            if line.strip() and not line.startswith("#") and not line.startswith("---"):
                                desc = line.strip()[:100]
                                break
                except Exception:
                    pass

            if final_title.lower() in seen_skill_keys:
                continue

            project_skills.append({
                "title": final_title,
                "description": desc,
                "category": "project_skills",
                "tags": ["project"],
                "visibility": "loadable",
                "source": "workspace",
                "is_handbag": False,
                "file_path": str(skill_md.resolve()),
                "content_preview": content_preview[:200],
            })

    # 4. Available Handbags (for switching)
    available_handbags: List[Dict[str, Any]] = []
    # A. Default Coder
    if default_hb_root.exists():
        available_handbags.append({
            "name": "default_coder",
            "title": "Default Coder (lollms_code)",
            "path": str(default_hb_root),
            "scope": "default",
        })
    # B. Project Handbags (.lollms_code/handbags/)
    ws_hb_dir = ws_root / ".lollms_code" / "handbags"
    if ws_hb_dir.exists():
        for item in sorted(ws_hb_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                available_handbags.append({
                    "name": item.name,
                    "title": item.name.replace("_", " ").title(),
                    "path": str(item.resolve()),
                    "scope": "project",
                })
    # C. Global Handbags (~/.lollms_client/lollms_code/handbags/)
    glob_hb_dir = Path.home() / ".lollms_client" / "lollms_code" / "handbags"
    if glob_hb_dir.exists():
        for item in sorted(glob_hb_dir.iterdir()):
            if item.is_dir() and not item.name.startswith(".") and item.name != "default_coder":
                available_handbags.append({
                    "name": item.name,
                    "title": item.name.replace("_", " ").title(),
                    "path": str(item.resolve()),
                    "scope": "global",
                })

    return {
        "persona": {
            "name": p_name,
            "category": p_cat,
            "description": p_desc,
            "handbag_path": str(hb_root) if hb_root else "",
            "source": persona_source,
            "soul_content": p_soul,
        },
        "tools": {
            "handbag": handbag_tools,
            "project": project_tools,
            "builtin": builtin_tools,
        },
        "skills": {
            "handbag": handbag_skills,
            "project": project_skills,
            "other": other_skills,
        },
        "available_handbags": available_handbags,
    }


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
        if meta and meta.get("type") == "effort_change":
            self.q.put(AgentEvent("effort_change", **meta))
            return True
        if msg_type in mapping:
            self.q.put(AgentEvent(mapping[msg_type], **(meta or {})))
            return True
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("tool_progress"):
                self.q.put(AgentEvent("info", text=chunk, **meta))
                return True

            # If this is a live artifact streaming chunk, dispatch as a dedicated artefact_chunk event
            if meta and meta.get("live_artifact_chunk"):
                title = meta.get("artifact_title", "artifact")
                lang = meta.get("artifact_lang", "")
                self.q.put(AgentEvent("artefact_chunk", text=chunk, title=title, language=lang))
                return True

            is_internal_chunk = bool(
                meta and (
                    meta.get("was_processed")
                    or meta.get("live_tool_chunk")
                )
            )
            # In FULL_CALLBACK_MODE, suppress internal streaming chunks, raw tool tags, context action tags, processing tags, raw tool JSON, and orphan delimiter chunks
            is_action_tag = bool(re.search(r'</?(?:processing|tool|unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update|mem_load|mem_delete|mem_search|mem_tag)\b', chunk or "", re.IGNORECASE))
            is_tool_json = bool(re.search(r'^\s*>?(?:```(?:json)?\s*)?\{"name":\s*"tool_', chunk or "", re.IGNORECASE))
            is_delimiter_chunk = bool(re.match(r'^\s*[`>]{1,4}\s*$', chunk or ""))
            if is_internal_chunk or is_action_tag or is_tool_json or is_delimiter_chunk or (chunk and "<!-- status:" in chunk):
                return True
            self.q.put(AgentEvent("chunk", text=chunk, was_processed=False))
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            self.q.put(AgentEvent("thought", text=chunk))
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            self.q.put(AgentEvent("info", text=chunk, **(meta or {})))
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
        if hasattr(personality, "_sub_agent_spawner") and personality._sub_agent_spawner:
            try:
                personality._sub_agent_spawner.cancel_active_child()
            except Exception:
                pass
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
    resume_turn: bool = False,
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
        nonlocal prompt
        try:
            chat_kwargs: Dict[str, Any] = {}

            # If resuming a turn from checkpoint, restore virtual history & prompt
            if resume_turn and hasattr(personality, "load_turn_checkpoint"):
                chk = personality.load_turn_checkpoint()
                if chk:
                    prompt = chk.get("prompt") or prompt
                    vh_raw = chk.get("virtual_history", [])
                    # Inject continuation guidance to virtual history so LLM immediately resumes
                    if vh_raw and vh_raw[-1].get("sender_type") == "assistant":
                        vh_raw.append({
                            "sender_type": "user",
                            "content": "[SYSTEM: Turn resumed from checkpoint. Continue your previous task to completion.]"
                        })
                    chat_kwargs["resume_virtual_history"] = vh_raw
                    chat_kwargs["starting_round"] = chk.get("round_count", 1)
                    ASCIIColors.success(f"[AgentBridge] Resuming turn from checkpoint (Round {chk.get('round_count', 1)}).")

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
                allow_computer_use=getattr(prefs, "allow_computer_use", False) or getattr(prefs, "enable_computer_use", False),
                enforce_end_tag=True,
                event_mode=EventMode.FULL_CALLBACK_MODE,
                shell_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
                python_autonomy_level=getattr(prefs, "shell_autonomy_level", "safe"),
                auto_approve_python=getattr(prefs, "auto_approve_python", False),
                confirm_handler=gui_confirm_handler,
                debug=prefs.debug,
                debug_export=prefs.debug,
                reasoning_effort=getattr(prefs, "reasoning_effort", None),
                dynamic_effort=getattr(prefs, "dynamic_effort", False),
                **chat_kwargs
            )
            event_queue.put(AgentEvent("done", result=result))
        except Exception as e:
            event_queue.put(AgentEvent("error", message=str(e)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t