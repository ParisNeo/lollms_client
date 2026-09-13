# lollms_agentic/agentic_types.py
# Shared value types for the two-stage Orchestrator/Worker execution engine.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """One atomic unit of work selected and verified by the Orchestrator."""
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    assigned_worker: Optional[int] = None
    report: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgenticPlan:
    """
    The Orchestrator's roadmap. Serialized into progress.md so both the
    host application and the LLM can verify progress deterministically.
    """
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    progress_artifact_title: str = "progress.md"

    def markdown(self) -> str:
        lines = [
            f"# Progress — {self.goal}",
            "",
        ]
        for step in self.steps:
            mark = {
                StepStatus.PENDING: "[ ]",
                StepStatus.IN_PROGRESS: "[~]",
                StepStatus.DONE: "[x]",
                StepStatus.FAILED: "[!]",
                StepStatus.SKIPPED: "[-]",
            }[step.status]
            lines.append(f"- {mark} {step.description}")
        return "\n".join(lines)

    def next_pending(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.status in (StepStatus.PENDING, StepStatus.FAILED):
                return step
        return None

    def has_actionable_step(self) -> bool:
        return self.next_pending() is not None

    def all_done(self) -> bool:
        return all(s.status == StepStatus.DONE for s in self.steps) and bool(self.steps)