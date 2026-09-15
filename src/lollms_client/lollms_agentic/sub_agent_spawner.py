# lollms_agentic/sub_agent_spawner.py
# Sub-agent spawning via the functional <agent> tag.
#
# DOCTRINE: The SAME chat() engine runs both the orchestrator and the worker.
# The orchestrator emits `<agent ...>...</agent>` (a functional tag, NOT a
# tool call). _StreamState buffers it like any other secondary tag and
# dispatches it here. The spawner:
#   1. Parses the tag attributes + body into a SubAgentConfig.
#   2. Streams spawn telemetry to the user (processing block + structured events).
#   3. Runs the worker through discussion.chat() with a bounded budget,
#      relaying its live chunks to the user's callback.
#   4. Seals a compact run record (the verbatim <agent> tag + the worker's
#      final report envelope) into ai_msg.metadata["sub_agent_runs"].
#
# The worker's internal transcript (tool calls, artifact bodies, corrections)
# is NEVER sealed: the next-turn orchestrator context shows only the tag it
# emitted, the report envelope it received, and its own final answer.

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE

_AGENT_ATTR_RE = re.compile(r'(\w+)=["\']([^"\']*)["\']')
_TASK_RE = re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE)
_CONTEXT_FILES_RE = re.compile(
    r'<context_files>(.*?)</context_files>', re.DOTALL | re.IGNORECASE
)
_REPORT_OPEN_RE = re.compile(r'<report\b[^>]*>', re.IGNORECASE)
_REPORT_CLOSE_RE = re.compile(r'</report>', re.IGNORECASE)

_MAX_CONTEXT_FILES = 8
_MAX_REPORT_CHARS = 6000
_MAX_TASK_CHARS = 4000

WORKER_TASK_WRAPPER = """\
=== TASK ===
{task}
=== END TASK ===

{context_files_block}
"""


_TASK_WRAPPER_RE = re.compile(
    r'={2,}\s*TASK\s*={2,}(.*?)(?:={2,}\s*END\s+TASK\s*={2,}|$)',
    re.DOTALL | re.IGNORECASE,
)

_KNOWN_TASK_NEIGHBOR_TAGS_RE = re.compile(
    r'<(?:task|context_files|report|system_prompt|agent)\b[^>]*>.*?</(?:task|context_files|report|system_prompt|agent)>',
    re.DOTALL | re.IGNORECASE,
)

_KNOWN_TASK_NEIGHBOR_VOID_RE = re.compile(
    r'<(?:task|context_files|report|system_prompt|agent)\b[^>]*/?>',
    re.IGNORECASE,
)


def _extract_task_body(body: str) -> str:
    """
    Tolerant task extraction, in priority order:
      1. <task>...</task> XML (canonical grammar).
      2. '=== TASK === ... === END TASK ===' plain-text wrapper (WORKER_TASK_WRAPPER
         mimicry), with or without the closing wrapper.
      3. Raw body text with known functional XML fragments removed (bare-prose
         delegation that accidentally embeds <context_files> or similar tags).
    Returns an empty string only when nothing usable remains.
    """
    stripped = (body or "").strip()
    if not stripped:
        return ""

    task_match = _TASK_RE.search(stripped)
    if task_match:
        return task_match.group(1).strip()

    wrapper_match = _TASK_WRAPPER_RE.search(stripped)
    if wrapper_match:
        extracted = wrapper_match.group(1).strip()
        if extracted:
            return extracted

    cleaned = _KNOWN_TASK_NEIGHBOR_TAGS_RE.sub('', stripped)
    cleaned = _KNOWN_TASK_NEIGHBOR_VOID_RE.sub('', cleaned)
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned

    return ""


class SubAgentConfig:
    """Parsed, validated configuration for one <agent> spawn."""

    __slots__ = (
        "name", "task", "context_files", "max_rounds",
        "system_prompt", "spawn_tag_verbatim",
    )

    def __init__(
        self,
        name: str,
        task: str,
        context_files: List[str],
        max_rounds: int,
        system_prompt: Optional[str],
        spawn_tag_verbatim: str,
    ):
        self.name = name
        self.task = task
        self.context_files = context_files
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt
        self.spawn_tag_verbatim = spawn_tag_verbatim


def parse_agent_tag(opening_tag: str, body: str) -> Optional[SubAgentConfig]:
    """
    Parses `<agent name="..." max_rounds="...">` + `<task>...</task>` +
    optional `<context_files>` block. Returns None when the block is
    structurally unusable (no task or empty task).
    """
    attrs = {
        m.group(1).lower(): m.group(2)
        for m in _AGENT_ATTR_RE.finditer(opening_tag)
    }
    task = _extract_task_body(body)
    if not task:
        return None

    context_files: List[str] = []
    files_match = _CONTEXT_FILES_RE.search(body)
    if files_match:
        for line in files_match.group(1).splitlines():
            cleaned = line.strip().strip('`').strip()
            if cleaned and cleaned not in context_files:
                context_files.append(cleaned)
    context_files = context_files[:_MAX_CONTEXT_FILES]

    try:
        max_rounds = int(attrs.get("max_rounds", "6"))
    except ValueError:
        max_rounds = 6
    max_rounds = max(2, min(max_rounds, 12))

    system_prompt = attrs.get("system_prompt") or attrs.get("persona") or None
    if system_prompt:
        system_prompt = system_prompt[:2000]

    if len(task) > _MAX_TASK_CHARS:
        task = task[:_MAX_TASK_CHARS] + "\n... [task truncated by spawner]"

    return SubAgentConfig(
        name=attrs.get("name") or f"sub_agent_{len(body) % 97}",
        task=task,
        context_files=context_files,
        max_rounds=max_rounds,
        system_prompt=system_prompt,
        spawn_tag_verbatim=opening_tag,
    )


def repair_agent_tag_body(opening_tag: str, body: str) -> Tuple[str, List[str]]:
    """
    Best-effort structural repair of a malformed <agent> body.

    Returns (repaired_task_text, notes). repaired_task_text is "" when nothing
    can be salvaged. The repair never invents instructions: it only strips
    wrapper noise and re-encloses the prose the model clearly intended as the
    task. Notes describe the transformations applied (for telemetry only).
    """
    notes: List[str] = []
    stripped = (body or "").strip()
    if not stripped:
        return "", notes

    original_len = len(stripped)

    wrapper_match = _TASK_WRAPPER_RE.search(stripped)
    if wrapper_match and wrapper_match.group(1).strip():
        notes.append("extracted prose from '=== TASK ===' wrapper")
        return wrapper_match.group(1).strip(), notes

    task_match = _TASK_RE.search(stripped)
    if task_match and task_match.group(1).strip():
        notes.append("extracted <task> body")
        return task_match.group(1).strip(), notes

    cleaned = _KNOWN_TASK_NEIGHBOR_TAGS_RE.sub('', stripped)
    cleaned = _KNOWN_TASK_NEIGHBOR_VOID_RE.sub('', cleaned)
    cleaned = cleaned.strip()
    if cleaned and len(cleaned) < original_len:
        notes.append("stripped stray functional XML fragments from bare prose")
        return cleaned, notes
    if cleaned:
        notes.append("used bare prose as task")
        return cleaned, notes

    return "", notes


def build_worker_context(discussion, config: "SubAgentConfig") -> str:
    """
    Renders the <context_files> block for the sub-agent.

    Files are resolved against the workspace root with path-confusion guards:
    backslash normalization, workspace prefix stripping, and rejection of
    parent-traversal segments. Content is read from disk, never synthesized.
    """
    from pathlib import Path as _P

    if not config.context_files:
        return "No context files were provided."

    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []

    artefacts = getattr(discussion, "artefacts", None)
    for name in config.context_files:
        if artefacts:
            art = artefacts.get(name)
            if art is not None:
                resolved.append(art)
                continue

        rel_path = name.replace("\\", "/").lstrip("/")
        if rel_path.startswith("workspace_data/"):
            rel_path = rel_path[len("workspace_data/"):]
        if not rel_path or ".." in _P(rel_path).parts:
            missing.append(name)
            continue
        try:
            if artefacts and hasattr(artefacts, "_get_workspace_root"):
                ws_root = artefacts._get_workspace_root()
            else:
                ws_root = _P(
                    getattr(discussion, "workspace_data_path", None)
                    or getattr(discussion, "workspace_path", None)
                    or "."
                ).resolve()
            candidate = (ws_root / rel_path).resolve()
            candidate.relative_to(ws_root)
            if candidate.is_file():
                content = candidate.read_text(encoding="utf-8", errors="ignore")
                resolved.append({
                    "title": rel_path,
                    "type": "document",
                    "content": content,
                    "language": _P(rel_path).suffix.lstrip(".") or None,
                })
                continue
        except (PermissionError, ValueError, OSError):
            pass
        missing.append(name)

    parts: List[str] = []
    if resolved:
        files_rendered = []
        for art in resolved:
            title = art.get("title", "untitled")
            content = art.get("content", "") or ""
            files_rendered.append(f'<file path="{title}">\n{content}\n</file>')
        parts.append(
            "=== FILES PROVIDED ===\n"
            + "\n\n".join(files_rendered)
            + "\n=== END FILES ==="
        )
    if missing:
        listed = ", ".join(f"`{m}`" for m in missing)
        parts.append(
            f"NOTE: The following requested files were NOT found in the "
            f"workspace and are unavailable: {listed}. Proceed with what "
            f"you have or report the gap."
        )
    if not parts:
        return "No context files were provided."
    return "\n\n".join(parts)


def _escape_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cb(callback: Optional[Callable], text: str, msg_type: MSG_TYPE, meta: Optional[Dict[str, Any]] = None) -> bool:
    if callback is None:
        return True
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and not (meta or {}).get("was_processed"):
        if isinstance(text, str) and text.strip():
            return callback(text, msg_type, meta or {})
        return True
    try:
        result = callback(text, msg_type, meta or {})
        return result is not False
    except Exception as ex:
        trace_exception(ex)
    return True


def _tag_mode_active(event_mode: Any) -> bool:
    return event_mode in (None, EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE)


def extract_worker_report(raw_output: str) -> str:
    """
    Extracts the <report>...</report> body from the worker's full output.
    Falls back to the scrubbed full text when the tags are missing.
    """
    if not raw_output:
        return ""
    open_match = _REPORT_OPEN_RE.search(raw_output)
    if open_match:
        rest = raw_output[open_match.end():]
        close_match = _REPORT_CLOSE_RE.search(rest)
        if close_match:
            return rest[:close_match.start()].strip()
        return rest.strip()
    return raw_output.strip()


def build_report_envelope(config: SubAgentConfig, worker_result: Dict[str, Any], worker_index: int) -> str:
    """
    Wraps the worker's report into the single compact envelope the
    orchestrator receives. Plain data: no executable grammar.
    """
    report = (worker_result.get("report") or "(empty report)")[:_MAX_REPORT_CHARS]
    files = worker_result.get("files") or []
    files_line = ", ".join(f"`{f}`" for f in files) if files else "none"
    status = "SUCCESS" if worker_result.get("success") else "FAILURE"
    return (
        f"[WORKER REPORT {worker_index} — {status}]\n"
        f"Agent: {config.name}\n"
        f"Files created/modified: {files_line}\n"
        f"---\n"
        f"{report}\n"
        f"---\n"
    )


def run_sub_agent(
    discussion,
    config: SubAgentConfig,
    callback: Optional[Callable],
    worker_index: int,
    event_mode: Any = None,
    parent_personality=None,
) -> Dict[str, Any]:
    """
    Runs one worker through the standard discussion.chat() engine.

    Streams to the user:
      1. <processing type="agent_spawn"> block announcing the spawn.
      2. The worker's live chunks (relayed verbatim through the callback).
      3. A closing status block with the sealed report digest.

    Returns the sealed run record:
      {"agent_tag": str, "report_envelope": str, "files": [...], "success": bool}
    """
    from lollms_client.lollms_agentic.prompts import WORKER_SYSTEM_PROMPT
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

    system_prompt = WORKER_SYSTEM_PROMPT
    if config.system_prompt:
        system_prompt = config.system_prompt + "\n\n" + WORKER_SYSTEM_PROMPT
    worker_persona = LollmsPersonality(
        name=config.name,
        system_prompt=system_prompt,
        lollms_client=discussion.lollmsClient,
    )

    context_body = build_worker_context(discussion, config)
    full_worker_prompt = WORKER_TASK_WRAPPER.format(
        task=config.task, context_files_block=context_body
    )

    spawn_meta = {
        "worker_index": worker_index,
        "name": config.name,
        "task": config.task[:500],
        "context_files": config.context_files,
        "max_rounds": config.max_rounds,
    }
    _cb(callback, "", MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START, {**spawn_meta, "event_mode": event_mode})

    # ── User-facing spawn announcement (processing block) ──
    task_preview = config.task[:200].replace("\n", " ")
    spawn_block = (
        f'\n<processing type="agent_spawn" title="Spawning agent: {_escape_attr(config.name)}"'
        f' task="{_escape_attr(task_preview)}"'
        f' context_files="{_escape_attr(", ".join(config.context_files))}"'
        f' max_rounds="{config.max_rounds}">\n'
        f"* 🤖 Spawning sub-agent **{config.name}** (budget: {config.max_rounds} rounds).\n"
    )
    if config.context_files:
        spawn_block += f"* 📎 Context files: {', '.join(config.context_files)}.\n"
    if config.system_prompt:
        spawn_block += "* 🎭 Custom specialization applied.\n"
    spawn_block += "* ⏳ The agent is now working; its output streams below.\n"

    ai_message = None
    worker_error: Optional[str] = None
    spawned_files: List[str] = []
    raw_report = ""

    def worker_stream_relay(chunk: str, msg_type=None, meta=None):
        if callback is None:
            return True
        try:
            mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
            return callback(chunk, mt, meta or {})
        except Exception:
            return True

    pre_run_tip = getattr(discussion, "active_branch_id", None)
    try:
        result = discussion.chat(
            user_message=full_worker_prompt,
            personality=worker_persona,
            add_user_message=True,
            tools=None,
            enable_artefacts=True,
            enable_memory=False,
            enable_episodic_memory=False,
            enable_auto_dream=False,
            enable_deep_memory_pulling=False,
            prehydrate_rag=False,
            max_nb_rounds=config.max_rounds,
            enable_notes=True,
            enable_skills=False,
            enable_forms=False,
            orchestrator_mode=False,
            event_mode=event_mode,
            streaming_callback=worker_stream_relay,
        )
        ai_message = (result or {}).get("ai_message")
        full_text = getattr(ai_message, "content", "") or ""
        from lollms_client.lollms_discussion._mixin_chat import _scrub_for_llm_context
        raw_report = extract_worker_report(_scrub_for_llm_context(full_text))
        spawned_files = [
            a.get("title", "")
            for a in (result or {}).get("artefacts", [])
            if isinstance(a, dict)
        ]
    except Exception as ex:
        trace_exception(ex)
        worker_error = f"Sub-agent runtime crashed: {ex}"
    finally:
        try:
            current_tip = getattr(discussion, "active_branch_id", None)
            if current_tip and current_tip != pre_run_tip:
                discussion.remove_message(current_tip)
            if pre_run_tip and getattr(discussion, "active_branch_id", None) != pre_run_tip:
                discussion.active_branch_id = pre_run_tip
        except Exception as detach_ex:
            trace_exception(detach_ex)
            ASCIIColors.warning(f"[SubAgentSpawner] Branch detach failed: {detach_ex}")

    # Guarantee physical disk synchronization of worker-created files.
    try:
        discussion.artefacts.sync_all_active_to_disk()
    except Exception as sync_err:
        ASCIIColors.warning(f"[SubAgentSpawner] Post-run disk sync: {sync_err}")

    success = worker_error is None and bool(raw_report.strip())
    if worker_error and not raw_report:
        raw_report = f"Sub-agent failed before producing a report. {worker_error}"
    elif worker_error:
        raw_report = f"{raw_report}\n\n[Sub-agent runtime error: {worker_error}]"

    report_envelope = build_report_envelope(config, {
        "report": raw_report,
        "files": spawned_files,
        "success": success,
    }, worker_index)

    # ── User-facing closing block: the report the agent sends back ──
    closing_block = (
        f"* {'✅' if success else '❌'} Agent **{config.name}** finished"
        f"{' — files: ' + ', '.join(spawned_files) if spawned_files else ''}.\n"
        f"<!-- status:{'success' if success else 'failure'} -->\n</processing>\n\n"
    )
    if _tag_mode_active(event_mode):
        _cb(callback, spawn_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
        _cb(callback, closing_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
    _cb(callback, "", MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END, {
        **spawn_meta,
        "success": success,
        "report_digest": raw_report[:2000],
        "files": spawned_files,
        "error": worker_error,
    })

    return {
        "agent_tag": config.spawn_tag_verbatim,
        "report_envelope": report_envelope,
        "files": spawned_files,
        "success": success,
    }