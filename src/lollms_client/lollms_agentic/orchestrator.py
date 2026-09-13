# lollms_agentic/orchestrator.py
# OrchestratorAgent: plans, delegates, verifies. Never executes tools.

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE

from .agentic_types import AgenticPlan, PlanStep, StepStatus
from .prompts import ORCHESTRATOR_SYSTEM_PROMPT, VERIFICATION_TASK_TEMPLATE
from .worker import WorkerAgent


_PLAN_OPEN_RE = re.compile(r'<plan\b[^>]*>', re.IGNORECASE)
_PLAN_CLOSE_RE = re.compile(r'</plan>', re.IGNORECASE)
_DELEGATE_OPEN_RE = re.compile(r'<delegate\b[^>]*>', re.IGNORECASE)
_DELEGATE_CLOSE_RE = re.compile(r'</delegate>', re.IGNORECASE)
_TASK_RE = re.compile(r'<task>(.*?)</task>', re.DOTALL | re.IGNORECASE)
_CONTEXT_FILES_RE = re.compile(
    r'<context_files>(.*?)</context_files>', re.DOTALL | re.IGNORECASE
)
_VERIFY_RE = re.compile(
    r'<verify\b[^>]*step\s*=\s*["\']?(\d+)["\']?[^>]*>(.*?)</verify>',
    re.DOTALL | re.IGNORECASE,
)
_DONE_RE = re.compile(r'<(?:done|end)\s*/?>', re.IGNORECASE)

_PASS_RE = re.compile(r'\bPASS\b', re.IGNORECASE)
_FAIL_RE = re.compile(r'\bFAIL(?:ED)?\b', re.IGNORECASE)

_MAX_CONTEXT_FILES = 8
_MAX_PLAN_STEPS = 20
_MAX_REPORT_CHARS = 2000


def _build_worker_specialties(tools_registry: Dict[str, Dict[str, Any]]) -> str:
    """
    Renders the tool registry as a plain-language capability list for the
    Orchestrator. Tool names appear as plain backticked words, never as
    executable tool syntax, so the King can describe what to delegate without
    ever learning invocation grammar.
    """
    if not tools_registry:
        return "(no worker tools are registered in this session)"
    entries: List[str] = []
    for name, spec in tools_registry.items():
        entries.append(f"- `{name}`: {spec.get('description', 'no description provided')}")
    return "\n".join(entries)


class OrchestratorAgent:
    """
    Two-stage doctrine stage 1: coordination.

    Contract:
      * Context contains NO tool syntax or execution templates.
      * Delegates work to WorkerAgent instances.
      * Progress is persisted to progress.md after every state transition.
      * VERIFY delegates ground-truth inspection to an Inspector Worker.
    """

    def __init__(
        self,
        context=None,
        tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        worker_factory: Optional[Callable[[], "WorkerAgent"]] = None,
        callback: Optional[Callable] = None,
        event_mode: Any = None,
        max_worker_rounds: int = 8,
        discussion: Any = None,
    ):
        self.context = context or discussion
        self.discussion = self.context
        self.tools_registry = tools_registry or {}
        self.callback = callback
        self.event_mode = event_mode
        self.max_worker_rounds = max_worker_rounds

        self.worker_factory = worker_factory or (
            lambda: WorkerAgent(
                context=self.context,
                tools_registry=self.tools_registry,
                callback=callback,
                event_mode=event_mode,
                max_rounds=self.max_worker_rounds,
            )
        )

        self._worker_specialties: str = _build_worker_specialties(self.tools_registry)
        self.plan: Optional[AgenticPlan] = None
        self.worker_counter = 0
        self.step_reports: Dict[int, Dict[str, Any]] = {}
        self._history: List[Dict[str, str]] = []
        self._workspace_tree_cache_revision: int = -1
        self._workspace_tree_cache: str = ""

    # ─────────────────────────────── telemetry ─────────────────────────────

    def _emit(self, msg_type: MSG_TYPE, meta: Dict[str, Any]) -> None:
        if self.event_mode is not None and "SILENT" in str(self.event_mode).upper():
            return
        if self.callback is None:
            return
        try:
            self.callback("", msg_type, meta)
        except Exception as ex:
            trace_exception(ex)

    def _progress_markdown(self) -> str:
        return self.plan.markdown() if self.plan else "# Progress — (no plan yet)"

    def _bump_workspace_revision(self) -> None:
        try:
            object.__setattr__(
                self.context,
                "_workspace_write_revision",
                int(getattr(self.context, "_workspace_write_revision", 0)) + 1,
            )
        except Exception:
            pass

    def _cached_workspace_tree(self) -> str:
        """
        Compact, cached directory tree of the sandboxed workspace.
        """
        revision = int(getattr(self.context, "_workspace_write_revision", 0))
        if self._workspace_tree_cache_revision == revision:
            return self._workspace_tree_cache

        lines: List[str] = []
        try:
            artefacts = getattr(self.context, "artefacts", None)
            if artefacts and hasattr(artefacts, "_get_workspace_root"):
                ws_root = artefacts._get_workspace_root()
            else:
                ws_root = Path(
                    getattr(self.context, "workspace_data_path", None)
                    or getattr(self.context, "_resolved_workspace", None)
                    or getattr(self.context, "workspace_path", None)
                    or "."
                ).resolve()

            if ws_root and ws_root.exists():
                for f in sorted(ws_root.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(ws_root)
                    rel_str = str(rel).replace("\\", "/")
                    if rel_str.startswith(".versions") or rel_str.startswith(".git"):
                        continue
                    if any(
                        part.startswith(".") and part not in (".", "..")
                        for part in rel.parts[:-1]
                    ):
                        continue
                    lines.append(rel_str)
        except Exception as ex:
            trace_exception(ex)

        self._workspace_tree_cache_revision = revision
        self._workspace_tree_cache = "\n".join(lines)
        return self._workspace_tree_cache

    # ───────────────────────────── progress.md ─────────────────────────────

    def _write_progress_file(self) -> None:
        """Persists the plan state as progress.md."""
        if self.plan is None:
            return
        try:
            artefacts = getattr(self.context, "artefacts", None)
            content = self._progress_markdown()
            if artefacts:
                existing = artefacts.get(self.plan.progress_artifact_title)
                if existing is None:
                    artefacts.add(
                        title=self.plan.progress_artifact_title,
                        artefact_type="document",
                        content=content,
                        active=True,
                        visibility="full",
                    )
                else:
                    artefacts.update(
                        title=self.plan.progress_artifact_title,
                        new_content=content,
                        bump_version=True,
                        active=True,
                    )
            else:
                ws_root = Path(
                    getattr(self.context, "workspace_data_path", None)
                    or getattr(self.context, "_resolved_workspace", None)
                    or getattr(self.context, "workspace_path", None)
                    or "."
                ).resolve()
                ws_root.mkdir(parents=True, exist_ok=True)
                (ws_root / self.plan.progress_artifact_title).write_text(content, encoding="utf-8")

            if hasattr(self.context, "commit"):
                self.context.commit()
        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.warning(f"[Orchestrator] Failed to persist progress.md: {ex}")

    def _mark_step(self, step: PlanStep, status: StepStatus, report: Optional[str] = None) -> None:
        step.status = status
        if report is not None:
            step.report = report
        self._write_progress_file()
        self._emit(MSG_TYPE.MSG_TYPE_INFO, {
            "type": "plan_step_status",
            "step": step.index,
            "status": status.value,
        })

    # ───────────────────────────── tag parsing ─────────────────────────────

    @staticmethod
    def _extract_block(text: str, open_re, close_re) -> Optional[str]:
        open_match = open_re.search(text)
        if not open_match:
            return None
        rest = text[open_match.end():]
        close_match = close_re.search(rest)
        if close_match:
            return rest[:close_match.start()].strip()
        return rest.strip()

    def _parse_plan(self, text: str) -> Optional[AgenticPlan]:
        block = self._extract_block(text, _PLAN_OPEN_RE, _PLAN_CLOSE_RE)
        if not block:
            return None

        steps: List[PlanStep] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r'^[-*+]\s*\[[xX~! -]\]\s*', '', line)
            line = re.sub(r'^\d+[.)]\s*', '', line)
            if not line or len(steps) >= _MAX_PLAN_STEPS:
                break
            steps.append(PlanStep(index=len(steps) + 1, description=line))

        if not steps:
            return None
        return AgenticPlan(goal="user_request", steps=steps)

    def _parse_delegate(self, text: str) -> Optional[Tuple[str, List[str]]]:
        block = self._extract_block(text, _DELEGATE_OPEN_RE, _DELEGATE_CLOSE_RE)
        if not block:
            return None
        task_match = _TASK_RE.search(block)
        if not task_match or not task_match.group(1).strip():
            return None
        task = task_match.group(1).strip()

        files: List[str] = []
        files_match = _CONTEXT_FILES_RE.search(block)
        if files_match:
            for line in files_match.group(1).splitlines():
                cleaned = line.strip().strip('`').strip()
                if cleaned and cleaned not in files:
                    files.append(cleaned)
        return task, files[:_MAX_CONTEXT_FILES]

    def _parse_verify(self, text: str) -> Optional[Tuple[int, str]]:
        match = _VERIFY_RE.search(text)
        if not match:
            return None
        return int(match.group(1)), match.group(2).strip()

    # ───────────────────────────── worker spawn ────────────────────────────

    def _spawn_worker(
        self, task: str, context_files: List[str], step: Optional[PlanStep]
    ) -> Dict[str, Any]:
        self.worker_counter += 1
        worker_index = self.worker_counter
        worker = self.worker_factory()

        self._emit(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START, {
            "round_id": step.index if step else 0,
            "worker_index": worker_index,
            "task": task[:500],
            "context_files": context_files,
            "max_rounds": self.max_worker_rounds,
        })

        result = worker.run(task=task, context_files=context_files)

        self._emit(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END, {
            "round_id": step.index if step else 0,
            "worker_index": worker_index,
            "success": result.get("success", False),
            "report_digest": (result.get("report") or "")[:2000],
            "files": result.get("files") or [],
            "error": result.get("error"),
        })

        self._bump_workspace_revision()
        return result

    def _spawn_inspector(
        self, step: PlanStep, worker_result: Dict[str, Any]
    ) -> bool:
        task = VERIFICATION_TASK_TEMPLATE.format(
            step_index=step.index,
            step_description=step.description,
            worker_report=(worker_result.get("report") or "(no worker report)")[:_MAX_REPORT_CHARS],
        )
        result = self._spawn_worker(
            task=task,
            context_files=[],
            step=step,
        )
        verdict_text = (result.get("report") or "").strip().upper()
        passed = (
            result.get("success")
            and _FAIL_RE.search(verdict_text) is None
            and _PASS_RE.search(verdict_text) is not None
        )
        return bool(passed)

    # ───────────────────────────── main loop ───────────────────────────────

    def run(self, user_message: str, max_rounds: int = 12) -> Dict[str, Any]:
        """
        Executes the PLAN → DELEGATE → VERIFY cycle until <done/> or budget.
        Returns a structured summary; never raises.
        """
        worker_turn_info: List[Dict[str, Any]] = []
        round_count = 0

        while round_count < max_rounds:
            round_count += 1

            system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
                worker_specialties=self._worker_specialties
            )
            progress_block = self._progress_markdown()
            history = self._build_history(user_message)

            messages = self._build_messages(
                system_prompt=system_prompt,
                progress_block=progress_block,
                history=history,
            )

            response = self._generate(messages)
            if response is None:
                break
            text = response.strip()
            if not text:
                break
            self._history.append({"role": "assistant", "content": text})

            if _DONE_RE.search(text):
                self._final_answer = _strip_done(text)
                break

            plan = self._parse_plan(text)
            if plan is not None and (
                self.plan is None or self._plan_content_changed(plan)
            ):
                self.plan = plan
                self._write_progress_file()
                continue

            delegation = self._parse_delegate(text)
            if delegation is not None:
                task, context_files = delegation
                step = self.plan.next_pending() if self.plan else None
                result = self._spawn_worker(task, context_files, step)
                worker_turn_info.append({
                    "task": task,
                    "files": result.get("files", []),
                    "success": result.get("success", False),
                })
                if step is not None:
                    self.step_reports[step.index] = result
                    if step.status == StepStatus.PENDING:
                        step.status = StepStatus.IN_PROGRESS
                self._feed_report(result)
                continue

            verification = self._parse_verify(text)
            if verification is not None and self.plan is not None:
                step_idx, judgement = verification
                target = next(
                    (s for s in self.plan.steps if s.index == step_idx), None
                )
                if target is None or target.status == StepStatus.DONE:
                    continue

                prior_report = self.step_reports.get(step_idx) or {}
                passed = self._spawn_inspector(target, prior_report)
                if passed:
                    self._mark_step(target, StepStatus.DONE, report=judgement)
                else:
                    self._mark_step(
                        target,
                        StepStatus.FAILED,
                        report=f"Inspector verdict: FAIL. {judgement}",
                    )
                continue

            self._nudge()

        return {
            "plan": self.plan,
            "progress_markdown": self._progress_markdown(),
            "worker_runs": worker_turn_info,
            "rounds_used": round_count,
            "final_answer": getattr(self, "_final_answer", ""),
        }

    # ───────────────────────────── context builders ────────────────────────

    def _plan_content_changed(self, new_plan: AgenticPlan) -> bool:
        if self.plan is None:
            return True
        return [s.description for s in new_plan.steps] != [
            s.description for s in self.plan.steps
        ]

    def _build_history(self, user_message: str) -> List[Dict[str, str]]:
        if not self._history:
            self._history.append({"role": "user", "content": user_message})
        return list(self._history)

    def _build_messages(
        self,
        system_prompt: str,
        progress_block: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        tree_block = self._cached_workspace_tree()
        tree_section = (
            "\n\n=== WORKSPACE TREE (name ONLY files from this list in <context_files>) ===\n"
            f"{tree_block}\n=== END WORKSPACE TREE ==="
            if tree_block
            else ""
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n=== CURRENT PROGRESS ===\n"
                    f"{progress_block}{tree_section}"
                ),
            }
        ]
        merged = self._merge_same_role(history)
        last_role = merged[-1]["role"] if merged else None
        if last_role != "user":
            messages.append({
                "role": "user",
                "content": (
                    "Current progress is shown in the system block. "
                    "Choose your next move: PLAN (if none), DELEGATE the next pending step, "
                    "VERIFY a completed step, or FINISH."
                ),
            })
        messages.extend(merged)
        return messages

    @staticmethod
    def _merge_same_role(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        for msg in history:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1] = {
                    "role": msg["role"],
                    "content": merged[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                merged.append(dict(msg))
        return merged

    def _generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        client = getattr(self.context, "lollmsClient", None) or getattr(self.context, "lollms_client", None)
        try:
            return client.generate_from_messages(
                messages=messages,
                stream=False,
                streaming_callback=None,
            )
        except Exception as ex:
            trace_exception(ex)
            return None

    def _feed_report(self, result: Dict[str, Any]) -> None:
        report = (result.get("report") or "(empty report)")[:_MAX_REPORT_CHARS]
        status = "SUCCESS" if result.get("success") else "FAILURE"
        files = result.get("files") or []
        files_line = ", ".join(files) if files else "none"
        envelope = (
            f"[WORKER REPORT — {status}]\n"
            f"Files created/modified: {files_line}\n"
            f"---\n"
            f"{report}\n"
            f"---\n"
            f"VERIFY the step against progress.md, then continue."
        )
        self._history.append({"role": "user", "content": envelope})

    def _nudge(self) -> None:
        self._history.append({"role": "user", "content": (
            "[SYSTEM: Your last response contained no plan, delegation, or verification. "
            "Emit <plan>, <delegate>, or <verify> now. Prose performs nothing.]"
        )})


def _strip_done(text: str) -> str:
    cleaned = _DONE_RE.sub("", text).strip()
    return cleaned