# lollms_discussion/_mixin_history.py
# HistoryMixin: Thin wrapper that delegates to the shared HistoryManager.

from typing import Any, Optional

from lollms_client.lollms_history import HistoryManager


class HistoryMixin:
    """
    Handles the export of discussion history, integration of agentic virtual_history,
    and normalization of messages for specific LLM APIs (e.g., OpenAI).
    Delegates to the shared HistoryManager to ensure unification with LollmsPersonality.
    """

    def export(self, format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None,
               suppress_system_prompt: bool = False, suppress_images: bool = False, 
               virtual_history: Optional[list] = None, debug: bool = False, 
               system_prompt_override: Optional[str] = None) -> Any:
        """
        Exports the discussion history. 
        If virtual_history is provided, it appends the active agentic context (unstripped)
        to the sanitized historical branch.
        """
        branch_tip_id = branch_tip_id or self.active_branch_id
        branch = self.get_branch(branch_tip_id)
        
        return HistoryManager.export(
            context=self,
            format_type=format_type,
            branch=branch,
            branch_tip_id=branch_tip_id,
            max_allowed_tokens=max_allowed_tokens,
            suppress_system_prompt=suppress_system_prompt,
            suppress_images=suppress_images,
            virtual_history=virtual_history,
            debug=debug,
            system_prompt_override=system_prompt_override
        )

    def _normalize_openai_messages(self, messages: list) -> list:
        """Delegates to the shared HistoryManager for OpenAI message normalization."""
        return HistoryManager._normalize_openai_messages(messages)