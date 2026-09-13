# lollms_agentic/worker.py
# WorkerAgent: executes exactly ONE atomic task under the full tool doctrine.

import re
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import EventMode, MSG_TYPE

from .prompts import (
    WORKER_CONTEXT_FILES_TEMPLATE,
    WORKER_SYSTEM_PROMPT,
    WORKER_TASK_TEMPLATE,
)

from lollms_client.lollms_discussion._mixin_chat import _sanitize_host_paths, _scrub_for_llm_context

_REPORT_OPEN_RE = re.compile(r'<report\b[^>]*>', re.IGNORECASE)
_REPORT_CLOSE_RE = re.compile(r'</report>', re.IGNORECASE)

_FUNCTIONAL_TAG_BLOCK_RE = re.compile(
    r'<(?:tool|art(?:ifact|efact)|skill|note|scratchpad|delegate|unlock_file|lock_file|hide_file|lollms_inline|lollms_form|generate_image|edit_image)\b[^>]*>.*?</(?:tool|art(?:ifact|efact)|skill|note|scratchpad|delegate|unlock_file|lock_file|hide_file|lollms_inline|lollms_form|generate_image|edit_image)>',
    re.DOTALL | re.IGNORECASE,
)
_FUNCTIONAL_TAG_VOID_RE = re.compile(
    r'<(?:tool|art(?:ifact|efact)|skill|note|scratchpad|delegate|unlock_file|lock_file|hide_file|lollms_inline|lollms_form|generate_image|edit_image)\b[^>]*/?>',
    re.IGNORECASE,
)


def _scrub_functional_tags_from_report(text: str) -> str:
    if not text:
        return ""
    cleaned = _FUNCTIONAL_TAG_BLOCK_RE.sub("", text)
    cleaned = _FUNCTIONAL_TAG_VOID_RE.sub("", cleaned)
    cleaned = re.sub(r'<\s*(?:done|end)\s*/?>', '', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


_MAX_CONTEXT_FILES = 8
_MAX_REPORT_CHARS = 12000


def _cb(callback: Optional[Callable], text: str, msg_type: MSG_TYPE, meta: Optional[Dict[str, Any]] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        return result is not False
    except Exception as ex:
        trace_exception(ex)
    return True


class WorkerAgent:
    """
    Stage 2 of the two-stage doctrine: atomic execution.

    Runs within either a LollmsDiscussion or LollmsPersonality context in worker mode:
      * full tool/artifact XML doctrine,
      * real-time telemetry streaming to user callback,
      * fresh, small context: task + named files only,
      * verified physical persistence to the workspace filesystem.
    """

    def __init__(
        self,
        context=None,
        tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        callback: Optional[Callable] = None,
        event_mode: Any = None,
        max_rounds: int = 8,
        discussion: Any = None,
    ):
        self.context = context or discussion
        self.discussion = self.context
        self.tools_registry = tools_registry or {}
        self.callback = callback
        self.event_mode = event_mode
        self.max_rounds = max_rounds

    def _detach_worker_branch(self, pre_run_tip: Optional[str]) -> None:
        if not hasattr(self.context, "active_branch_id"):
            return
        try:
            current_tip = getattr(self.context, "active_branch_id", None)
            if current_tip and current_tip != pre_run_tip:
                self.context.remove_message(current_tip)
            if pre_run_tip and getattr(self.context, "active_branch_id", None) != pre_run_tip:
                self.context.active_branch_id = pre_run_tip
        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.warning(f"[Worker] Branch detach failed: {ex}")

    # ───────────────────────────── telemetry ──────────────────────────────

    def _tag_mode(self) -> bool:
        return self.event_mode in (None, EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE)

    def _event_mode(self) -> bool:
        return self.event_mode in (None, EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)

    def _emit(self, msg_type: MSG_TYPE, meta: Dict[str, Any]) -> None:
        if self.event_mode is EventMode.SILENT_MODE:
            return
        _cb(self.callback, "", msg_type, meta)

    def _emit_tag(self, text: str) -> None:
        if self._tag_mode():
            _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

    # ───────────────────────────── context files ──────────────────────────

    def _resolve_context_files(
        self, context_files: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []

        artefacts = getattr(self.context, "artefacts", None)

        for name in context_files:
            if artefacts:
                art = artefacts.get(name)
                if art is not None:
                    resolved.append(art)
                    continue

            rel_path = name.replace("\\", "/").lstrip("/")
            if rel_path.startswith("workspace_data/"):
                rel_path = rel_path[len("workspace_data/"):]
            if not rel_path or ".." in Path(rel_path).parts:
                missing.append(name)
                continue
            try:
                if artefacts and hasattr(artefacts, "_get_workspace_root"):
                    ws_root = artefacts._get_workspace_root()
                else:
                    ws_root = Path(
                        getattr(self.context, "workspace_data_path", None)
                        or getattr(self.context, "_resolved_workspace", None)
                        or getattr(self.context, "workspace_path", None)
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
                        "language": Path(rel_path).suffix.lstrip(".") or None,
                    })
                    continue
            except (PermissionError, ValueError, OSError) as ex:
                ASCIIColors.warning(
                    f"[Worker] Context file '{name}' rejected: {ex}"
                )
            missing.append(name)

        return resolved, missing

    def _build_context_files_block(self, resolved: List[Dict[str, Any]], missing: List[str]) -> str:
        parts: List[str] = []
        if resolved:
            files_rendered = []
            for art in resolved:
                title = art.get("title", "untitled")
                content = self._artifact_content(art)
                files_rendered.append(f'<file path="{title}">\n{content}\n</file>')
            parts.append(WORKER_CONTEXT_FILES_TEMPLATE.format(files_block="\n\n".join(files_rendered)))

        if missing:
            listed = ", ".join(f"`{m}`" for m in missing)
            parts.append(
                f"NOTE: The following requested files were NOT found in the "
                f"workspace and are unavailable: {listed}. Proceed with what "
                f"you have or report the gap."
            )
        return "\n\n".join(parts)

    def _artifact_content(self, art: Dict[str, Any]) -> str:
        artefacts = getattr(self.context, "artefacts", None)
        content = art.get("content") or ""
        if artefacts:
            if art.get("type") == "data" and hasattr(artefacts, "_get_lam_content"):
                content = artefacts._get_lam_content(art) or content
            elif not content and hasattr(artefacts, "_read_content_from_disk"):
                content = artefacts._read_content_from_disk(art)
        if not content and "title" in art:
            ws_root = Path(
                getattr(self.context, "workspace_data_path", None)
                or getattr(self.context, "_resolved_workspace", None)
                or getattr(self.context, "workspace_path", None)
                or "."
            ).resolve()
            candidate = ws_root / art["title"]
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        return content or ""

    # ───────────────────────────── execution ──────────────────────────────

    def run(self, task: str, context_files: List[str]) -> Dict[str, Any]:
        resolved, missing = self._resolve_context_files(context_files)
        files_block = self._build_context_files_block(resolved, missing)
        worker_prompt = WORKER_TASK_TEMPLATE.format(
            task=task,
            context_files_block=files_block,
        )

        self._emit(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START, {
            "task": task[:500],
            "context_files": context_files,
            "max_rounds": self.max_rounds,
        })
        self._emit_tag(f'\n<processing type="worker" title="Specialist Worker" task="{_attr_escape(task[:120])}">\n')
        self._emit_tag(f"* 🛠️ Specialist spawned for task: {task[:80]}... (budget: {self.max_rounds} rounds).\n")
        if resolved:
            self._emit_tag(f"* 📎 Loaded {len(resolved)} context file(s): {', '.join(a.get('title', '?') for a in resolved)}\n")
        if missing:
            self._emit_tag(f"* ⚠️ Missing context file(s): {', '.join(missing)}\n")

        raw_report = ""
        worker_error: Optional[str] = None
        spawned_artifacts: List[str] = []
        is_discussion = hasattr(self.context, "add_message") and hasattr(self.context, "active_branch_id")

        # Relay callback so the user receives real-time progress of the worker's inner loop
        def worker_stream_relay(chunk: str, msg_type=None, meta=None):
            if self.callback is None:
                return True
            try:
                mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
                return self.callback(chunk, mt, meta or {})
            except Exception:
                return True

        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        worker_persona = LollmsPersonality(
            name="SpecialistWorker",
            system_prompt=WORKER_SYSTEM_PROMPT,
            lollms_client=getattr(self.context, "lollmsClient", None) or getattr(self.context, "lollms_client", None)
        )

        if is_discussion:
            pre_run_tip = getattr(self.context, "active_branch_id", None)
            try:
                result = self.context.chat(
                    user_message=worker_prompt,
                    personality=worker_persona,
                    add_user_message=True,
                    tools=self.tools_registry,
                    enable_artefacts=True,
                    enable_memory=False,
                    enable_episodic_memory=False,
                    enable_auto_dream=False,
                    enable_deep_memory_pulling=False,
                    prehydrate_rag=False,
                    max_nb_rounds=self.max_rounds,
                    enable_notes=True,
                    enable_skills=False,
                    enable_forms=False,
                    orchestrator_mode=False,
                    event_mode=self.event_mode,
                    streaming_callback=worker_stream_relay,
                )
                ai_message = (result or {}).get("ai_message")
                full_text = getattr(ai_message, "content", "") or ""
                raw_report = _scrub_for_llm_context(full_text)
                spawned_artifacts = [
                    a.get("title", "")
                    for a in (result or {}).get("artefacts", [])
                    if isinstance(a, dict)
                ]
            except Exception as ex:
                trace_exception(ex)
                worker_error = f"Worker runtime crashed: {ex}"
            finally:
                self._detach_worker_branch(pre_run_tip)
        else:
            try:
                result = self.context.chat(
                    prompt=worker_prompt,
                    tools=self.tools_registry,
                    enable_artefacts=True,
                    use_internal_history=False,
                    max_nb_rounds=self.max_rounds,
                    orchestrator_mode=False,
                    event_mode=self.event_mode,
                    streaming_callback=worker_stream_relay,
                )
                full_text = (result or {}).get("response", "") or ""
                raw_report = _scrub_for_llm_context(full_text)
                spawned_artifacts = [
                    ch.get("path", "")
                    for ch in (result or {}).get("workspace_changes", [])
                    if isinstance(ch, dict)
                ]
            except Exception as ex:
                trace_exception(ex)
                worker_error = f"Worker runtime crashed: {ex}"

        # Guarantee physical disk synchronization after worker execution
        if hasattr(self.context, "artefacts") and self.context.artefacts:
            try:
                self.context.artefacts.sync_all_active_to_disk()
            except Exception as sync_err:
                ASCIIColors.warning(f"[Worker] Post-execution disk sync warning: {sync_err}")

        # If spawned_artifacts is empty, scan artefacts manager for active files
        if not spawned_artifacts and hasattr(self.context, "artefacts") and self.context.artefacts:
            for art in self.context.artefacts.list(active_only=True):
                title = art.get("title", "")
                if title and not title.endswith("::images") and title != "progress.md":
                    spawned_artifacts.append(title)

        report_text = self._extract_report(raw_report)
        report_text = _scrub_functional_tags_from_report(report_text)
        if worker_error and not report_text:
            report_text = f"Worker failed before producing a report. {worker_error}"
        elif worker_error:
            report_text = f"{report_text}\n\n[Worker runtime error: {worker_error}]"

        if len(report_text) > _MAX_REPORT_CHARS:
            report_text = (
                report_text[:_MAX_REPORT_CHARS]
                + f"\n... [report truncated, {len(report_text) - _MAX_REPORT_CHARS} more chars]"
            )

        report_text = _sanitize_host_paths(report_text)
        success = worker_error is None and bool(report_text.strip())

        files_summary = f" — files: {', '.join(spawned_artifacts)}" if spawned_artifacts else ""
        self._emit_tag(f"* {'✅' if success else '❌'} Specialist finished{files_summary}.\n")
        self._emit_tag(f'<!-- status:{"success" if success else "failure"} -->\n</processing>\n\n')

        self._emit(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END, {
            "success": success,
            "report_digest": report_text[:2000],
            "files": spawned_artifacts,
            "error": worker_error,
        })

        return {
            "report": report_text,
            "files": spawned_artifacts,
            "success": success,
        }

    @staticmethod
    def _extract_report(raw_output: str) -> str:
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


def _attr_escape(value: str) -> str:
    return (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )