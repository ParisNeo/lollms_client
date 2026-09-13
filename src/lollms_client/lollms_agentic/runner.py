# lollms_agentic/runner.py
# AgenticRunner: the single entry point called when agentic orchestrator mode is active.

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
    Bridges conversational chat loops (LollmsDiscussion and LollmsPersonality)
    and the two-stage doctrine:
      1. Resolves the active tool registry.
      2. Runs the Orchestrator (plan/delegate/verify) with mandatory delegation.
      3. Persists the Orchestrator's final answer verbatim into the context.
      4. Respects versioning in LollmsDiscussion while using live filesystem/git
         in LollmsPersonality.
    """

    def __init__(
        self,
        context=None,
        tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        callback: Optional[Callable] = None,
        event_mode: Any = None,
        max_orchestrator_rounds: int = 12,
        max_worker_rounds: int = 8,
        discussion: Any = None,
    ):
        self.context = context or discussion
        self.discussion = self.context
        self.tools_registry = tools_registry or {}
        self.callback = callback
        self.event_mode = event_mode
        self.max_orchestrator_rounds = max_orchestrator_rounds
        self.max_worker_rounds = max_worker_rounds

    def run(self, user_message: str) -> Dict[str, Any]:
        client = getattr(self.context, "lollmsClient", None) or getattr(self.context, "lollms_client", None)
        is_discussion = hasattr(self.context, "add_message") and hasattr(self.context, "commit")

        user_msg = None
        ai_msg = None

        if is_discussion:
            user_msg = self.context.add_message(
                sender="user",
                sender_type="user",
                content=user_message,
            )
            ai_msg = self.context.add_message(
                sender=getattr(client, "ai_name", "assistant"),
                sender_type="assistant",
                content="",
                parent_id=user_msg.id,
                model_name=getattr(getattr(client, "llm", None), "model_name", "unknown"),
                binding_name=getattr(getattr(client, "llm", None), "binding_name", "unknown"),
            )
            _cb(self.callback, ai_msg.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_msg.id})

        orchestrator = OrchestratorAgent(
            context=self.context,
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
            if ai_msg:
                ai_msg.content = final_text
            _cb(self.callback, final_text, MSG_TYPE.MSG_TYPE_CHUNK)

        if is_discussion:
            ai_msg.metadata = {
                "mode": "agentic_orchestrator",
                "orchestrator_report": {
                    "plan": outcome.get("plan"),
                    "worker_runs": outcome.get("worker_runs") or [],
                    "progress_markdown": outcome.get("progress_markdown") or "",
                    "rounds_used": outcome.get("rounds_used", 0),
                },
            }

            if self.context._is_db_backed and self.context.autosave:
                self.context.commit()

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
        else:
            if hasattr(self.context, "_conversation") and getattr(self.context, "use_internal_history", True):
                self.context._conversation.append({"role": "user", "content": user_message})
                self.context._conversation.append({"role": "assistant", "content": final_text})

            tool_calls = []
            for wr in outcome.get("worker_runs", []):
                tool_calls.append({"name": "delegate_worker", "parameters": {"task": wr.get("task", "")}})

            _has_tti = False
            if client:
                _has_tti = getattr(client, "tti", None) is not None or bool(getattr(client, "tti_model_profiles_registry", None))

            return {
                "response": final_text,
                "tool_calls": tool_calls,
                "tool_results": outcome.get("worker_runs", []),
                "rounds": outcome.get("rounds_used", 0),
                "workspace_changes": [],
                "was_cancelled": False,
                "context_health": {},
                "tti_available": _has_tti,
                "orchestrator_report": outcome,
            }