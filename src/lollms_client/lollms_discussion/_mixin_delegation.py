# lollms_agentic/worker.py
# WorkerAgent: stage 2 of the two-stage Orchestrator/Worker doctrine.
#
# The WORKER is a short-lived sub-agent spawned per delegation. It receives
# a fresh, small context (the task + the explicitly named files) and runs a
# bounded chat() loop in worker mode under the full tool/artifact doctrine.
# All existing loop-guards (_StreamState one-action-per-turn, phantom
# interception, epoch gating, FailureMemory) protect the Worker for free.
#
# The Worker's final output is captured inside <report>...</report> and fed
# back to the Orchestrator as ONE compact "[WORKER REPORT]" envelope that
# contains no executable grammar (zero tool syntax) — closing the
# report-injection mimicry vector.

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE

from ._mixin_chat import (
    _sanitize_host_paths,
    _scrub_for_llm_context,
)

_REPORT_OPEN_RE = re.compile(r'<report\b[^>]*>', re.IGNORECASE)
_REPORT_CLOSE_RE = re.compile(r'</report>', re.IGNORECASE)

_MAX_CONTEXT_FILES = 8
_MAX_REPORT_CHARS_DEFAULT = 12000


class DelegationMixin:
    """
    Adds Orchestrator→Worker delegation to the discussion chat loop.

    The mixin is intentionally stateless: all per-turn state lives on the
    ChatMixin instance and the spawned worker runtime, never on the class.
    """

    # ------------------------------------------------------------------ prompt
    # Prompt doctrine lives in PromptMixin (single source of truth).
    # DelegationMixin consumes it via self._build_orchestrator_instructions()
    # and self._build_worker_doctrine() resolved through the MRO.

    # ------------------------------------------------------------ tag parsing

    @staticmethod
    def _parse_delegation_tag(
        opening_tag: str, body: str
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Parses a delegation block into (task, context_files).
        Returns None when the block is structurally unusable.
        """
        attrs = {
            m.group(1).lower(): m.group(2)
            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', opening_tag)
        }
        task_match = re.search(
            r'<task>(.*?)</task>', body, re.DOTALL | re.IGNORECASE
        )
        if not task_match:
            return None
        task = task_match.group(1).strip()
        if not task:
            return None

        files_match = re.search(
            r'<context_files>(.*?)</context_files>', body, re.DOTALL | re.IGNORECASE
        )
        context_files: List[str] = []
        if files_match:
            for line in files_match.group(1).splitlines():
                cleaned = line.strip().strip('`').strip()
                if cleaned and cleaned not in context_files:
                    context_files.append(cleaned)
        context_files = context_files[:_MAX_CONTEXT_FILES]
        return task, context_files

    # --------------------------------------------------------------- worker io

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
            art = self.artefacts.get(name)
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
                ws_root = self.artefacts._get_workspace_root()
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
                    f"[Delegation] Context file '{name}' rejected: {ex}"
                )
            missing.append(name)

        return resolved, missing

    def _build_worker_context(
        self, resolved_files: List[Dict[str, Any]], missing: List[str]
    ) -> str:
        """
        Builds the Worker's complete prompt body: named file contents only.
        The task doctrine is prepended by the caller.
        """
        parts: List[str] = []
        if resolved_files:
            parts.append("=== FILES PROVIDED FOR THIS TASK ===")
            for art in resolved_files:
                title = art.get("title", "untitled")
                content = self.artefacts._get_lam_content(art) if art.get("type") == "data" else (art.get("content") or "")
                if not content:
                    content = self.artefacts._read_content_from_disk(art) if art.get("title") in {a.get("title") for a in self.artefacts.list()} else ""
                if not content:
                    content = art.get("content") or ""
                parts.append(f'<file path="{title}">\n{content}\n</file>')
            parts.append("=== END FILES ===\n")

        if missing:
            listed = ", ".join(f"`{m}`" for m in missing)
            parts.append(
                f"NOTE: The following requested files were NOT found in the "
                f"workspace and are unavailable: {listed}. Proceed with what "
                f"you have or report the gap."
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------- worker run

    def _run_worker(
        self,
        task: str,
        context_files: List[str],
        worker_tools: Optional[Dict[str, Dict[str, Any]]],
        max_worker_rounds: int,
        callback: Optional[Callable],
        event_meta: Dict[str, Any],
        think: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs one Worker to completion and returns its report envelope.
        Never raises: all failures become structured failure reports.
        """
        resolved, missing = self._resolve_context_files(context_files)
        context_body = self._build_worker_doctrine(task) + "\n\n" + self._build_worker_context(resolved, missing)

        report_limit = event_meta.get("report_char_limit", _MAX_REPORT_CHARS_DEFAULT)

        _cb(callback, "", MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START, {
            **event_meta,
            "task": task[:500],
            "context_files": context_files,
            "max_rounds": max_worker_rounds,
        })

        raw_report = ""
        worker_error: Optional[str] = None
        spawned_artifacts: List[str] = []

        def worker_stream_relay(chunk: str, msg_type=None, meta=None):
            if callback is None:
                return True
            try:
                mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
                return callback(chunk, mt, meta or {})
            except Exception:
                return True

        from lollms_client.lollms_agentic.prompts import WORKER_SYSTEM_PROMPT
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        worker_persona = LollmsPersonality(
            name="SpecialistWorker",
            system_prompt=WORKER_SYSTEM_PROMPT,
            lollms_client=self.lollmsClient
        )

        try:
            result = self.chat(
                user_message=context_body,
                personality=worker_persona,
                add_user_message=False,
                tools=worker_tools,
                enable_artefacts=True,
                enable_memory=False,
                enable_episodic_memory=False,
                enable_auto_dream=False,
                enable_deep_memory_pulling=False,
                prehydrate_rag=False,
                max_nb_rounds=max_worker_rounds,
                enable_notes=True,
                enable_skills=False,
                enable_forms=False,
                orchestrator_mode=False,
                event_mode=event_meta.get("event_mode"),
                streaming_callback=worker_stream_relay,
                think=think,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
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

        # Ensure physical workspace disk sync
        try:
            self.artefacts.sync_all_active_to_disk()
        except Exception as sync_err:
            ASCIIColors.warning(f"[DelegationMixin] Post-worker disk sync: {sync_err}")

        report_text = self._extract_worker_report(raw_report)
        if worker_error and not report_text:
            report_text = f"Worker failed before producing a report. {worker_error}"
        elif worker_error:
            report_text = f"{report_text}\n\n[Worker runtime error: {worker_error}]"

        if len(report_text) > report_limit:
            report_text = (
                report_text[:report_limit]
                + f"\n... [report truncated, {len(report_text) - report_limit} more chars]"
            )

        report_text = _sanitize_host_paths(report_text)

        _cb(callback, "", MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END, {
            **event_meta,
            "success": worker_error is None and bool(report_text.strip()),
            "report_digest": report_text[:2000],
            "files": spawned_artifacts,
            "error": worker_error,
        })

        return {
            "report": report_text,
            "files": spawned_artifacts,
            "success": worker_error is None and bool(report_text.strip()),
        }

    @staticmethod
    def _extract_worker_report(raw_output: str) -> str:
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

    # ------------------------------------------------------- orchestrator feed

    def _build_worker_report_envelope(
        self, worker_result: Dict[str, Any], worker_index: int
    ) -> str:
        """
        Wraps the Worker report into the single compact message the
        Orchestrator receives. Plain data envelope: no executable grammar.
        """
        report = worker_result.get("report", "") or "(empty report)"
        files = worker_result.get("files") or []
        files_line = ", ".join(f"`{f}`" for f in files) if files else "none"
        status = "SUCCESS" if worker_result.get("success") else "FAILURE"
        return (
            f"[WORKER REPORT {worker_index} — {status}]\n"
            f"Files created/modified: {files_line}\n"
            f"---\n"
            f"{report}\n"
            f"---\n"
            f"Choose your next move: ANSWER the user (then `<done/>`), or "
            f"DELEGATE a follow-up task if this report is insufficient."
        )