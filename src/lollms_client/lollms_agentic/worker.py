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
    """
    Strips every functional XML tag from a Worker report before it is fed
    back to the Orchestrator.

    The King's context must remain grammar-free by construction: a single
    leaked <tool> or <artifact> example in its history is an in-context
    template it can learn to mimic. Reports carry facts (what was done,
    which files were touched), never executable grammar.
    """
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

    The Worker reuses the discussion's chat() loop in worker mode:
      * full tool/artifact XML doctrine,
      * all existing loop guards (one-action-per-turn, phantom interception,
        epoch gating, FailureMemory),
      * fresh, small context: task + named files only.

    Telemetry contract:
      * PROCESSING_TAG_MODE / MIXED_MODE: emits a <processing type="worker">
        block into the UI stream so the specialist's lifecycle is visible.
      * FULL_CALLBACK_MODE / MIXED_MODE: emits MSG_TYPE_WORKER_SPAWN_START/END.
      * SILENT_MODE: everything suppressed.
    """

    def __init__(
        self,
        discussion,
        tools_registry: Dict[str, Dict[str, Any]],
        callback: Optional[Callable] = None,
        event_mode: Any = None,
        max_rounds: int = 8,
    ):
        self.discussion = discussion
        self.tools_registry = tools_registry or {}
        self.callback = callback
        self.event_mode = event_mode
        self.max_rounds = max_rounds

    def _detach_worker_branch(self, pre_run_tip: Optional[str]) -> None:
        """
        Removes the worker's ephemeral user/assistant message pair from the
        shared discussion branch after the run.

        The Worker framed its task as a real user message (so it saw its own
        raw execution path verbatim during the loop), but the pair must not
        persist in the King's conversation: only the report envelope is fed
        back to the Orchestrator. Branch state is restored to the pre-run tip.
        """
        try:
            current_tip = getattr(self.discussion, "active_branch_id", None)
            if current_tip and current_tip != pre_run_tip:
                self.discussion.remove_message(current_tip)
            if pre_run_tip and getattr(self.discussion, "active_branch_id", None) != pre_run_tip:
                self.discussion.active_branch_id = pre_run_tip
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
        """Appends a processing-tag line to the worker's UI block."""
        if self._tag_mode():
            _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

    # ───────────────────────────── context files ──────────────────────────

    def _resolve_context_files(
        self, context_files: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolves the requested context files against the artifact registry
        and the sandboxed workspace. Returns (resolved, missing).
        """
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []

        for name in context_files:
            art = self.discussion.artefacts.get(name)
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
                ws_root = self.discussion.artefacts._get_workspace_root()
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
        """Renders resolved files + missing-file notice into the worker prompt body."""
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
        """Resolves the LLM-facing content of a resolved artifact (disk-first for data twins)."""
        content = art.get("content") or ""
        if art.get("type") == "data":
            content = self.discussion.artefacts._get_lam_content(art) or content
        elif not content:
            content = self.discussion.artefacts._read_content_from_disk(art)
        return content or ""

    # ───────────────────────────── execution ──────────────────────────────

    def run(self, task: str, context_files: List[str]) -> Dict[str, Any]:
        """
        Runs the worker loop to completion and returns its report envelope.
        Never raises: all failures become structured failure reports.
        """
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
        self._emit_tag(f"* 🛠️ Specialist spawned (rounds budget: {self.max_rounds}).\n")
        if resolved:
            self._emit_tag(f"* 📎 Loaded {len(resolved)} context file(s): {', '.join(a.get('title', '?') for a in resolved)}\n")
        if missing:
            self._emit_tag(f"* ⚠️ Missing context file(s): {', '.join(missing)}\n")

        raw_report = ""
        worker_error: Optional[str] = None
        spawned_artifacts: List[str] = []
        pre_run_tip = getattr(self.discussion, "active_branch_id", None)

        try:
            result = self.discussion.chat(
                user_message=worker_prompt,
                personality=None,
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

        self._emit_tag(f"* {'✅' if success else '❌'} Specialist finished"
                       f"{' — files: ' + ', '.join(spawned_artifacts) if spawned_artifacts else ''}.\n")
        self._emit_tag(f'<!-- status:{"success" if success else "failure"} -->\n</processing>\n')

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
        """
        Extracts the <report> body from Worker output. Falls back to the
        scrubbed full text when the tags are missing (bounded by the caller).
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


def _attr_escape(value: str) -> str:
    """Escapes a string for safe embedding into a processing-tag attribute."""
    return (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )