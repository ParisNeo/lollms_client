# lollms_agentic/runner.py
# AgenticRunner: the single entry point ChatMixin calls when agent mode is on.

from typing import Any, Callable, Dict, List, Optional

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE

from .orchestrator import OrchestratorAgent
from .worker import WorkerAgent


def _cb(callback: Optional[Callable], text: str, msg_type: MSG_TYPE, meta: Optional[Dict[str, Any]] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        return result is not False
    except Exception as ex:
        trace_exception(ex)
    return True


class AgenticRunner:
    """
    Bridges the conversational ChatMixin and the two-stage doctrine:
      1. Resolves the active tool registry exactly once (reuses ChatMixin logic).
      2. Runs the Orchestrator (plan/delegate/verify) with mandatory delegation.
      3. Persists the King's OWN final answer verbatim into the discussion
         branch — never a synthetic template. No placeholders, no processing
         blocks, no system-generated mission-report text ever enters the
         LLM-facing context.

    The orchestrator persona never sees the tools; every execution is a Worker.
    """

    def __init__(
        self,
        discussion,
        tools_registry: Dict[str, Dict[str, Any]],
        callback: Optional[Callable] = None,
        event_mode: Any = None,
        max_orchestrator_rounds: int = 12,
        max_worker_rounds: int = 8,
    ):
        self.discussion = discussion
        self.tools_registry = tools_registry or {}
        self.callback = callback
        self.event_mode = event_mode
        self.max_orchestrator_rounds = max_orchestrator_rounds
        self.max_worker_rounds = max_worker_rounds

    def run(self, user_message: str) -> Dict[str, Any]:
        client = self.discussion.lollmsClient

        user_msg = self.discussion.add_message(
            sender="user",
            sender_type="user",
            content=user_message,
        )
        ai_msg = self.discussion.add_message(
            sender=getattr(client, "ai_name", "assistant"),
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=getattr(getattr(client, "llm", None), "model_name", "unknown"),
            binding_name=getattr(getattr(client, "llm", None), "binding_name", "unknown"),
        )
        _cb(self.callback, ai_msg.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_msg.id})

        orchestrator = OrchestratorAgent(
            discussion=self.discussion,
            tools_registry=self.tools_registry,
            callback=self.callback,
            event_mode=self.event_mode,
            max_worker_rounds=self.max_worker_rounds,
        )
        try:
            outcome = orchestrator.run(
                user_message=user_message,
                max_rounds=self.max_orchestrator_rounds,
            )
        except Exception as ex:
            trace_exception(ex)
            outcome = {
                "plan": None,
                "progress_markdown": "",
                "worker_runs": [],
                "rounds_used": 0,
                "final_answer": "",
            }

        final_text = (outcome.get("final_answer") or "").strip()
        if final_text:
            ai_msg.content = final_text
            _cb(self.callback, final_text, MSG_TYPE.MSG_TYPE_CHUNK)

        ai_msg.metadata = {
            "mode": "agentic_orchestrator",
            "orchestrator_report": {
                "plan": outcome.get("plan"),
                "worker_runs": outcome.get("worker_runs") or [],
                "progress_markdown": outcome.get("progress_markdown") or "",
                "rounds_used": outcome.get("rounds_used", 0),
            },
        }

        if self.discussion._is_db_backed and self.discussion.autosave:
            self.discussion.commit()

        return {
            "user_message": user_msg,
            "ai_message": ai_msg,
            "sources": [],
            "artefacts": [],
            "memory_report": {},
            "dream_report": None,
            "was_cancelled": False,
            "orchestrator_report": outcome,
        }