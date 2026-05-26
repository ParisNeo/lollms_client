# lollms_discussion/_mixin_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# MemoryMixin — integrates LollmsMemoryManager into LollmsDiscussion.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors

if TYPE_CHECKING:
    from .lollms_memory import LollmsMemoryManager

class MemoryMixin:
    """Adds multi-level persistent memory to LollmsDiscussion."""

    def _init_memory(self, memory_manager: Optional['LollmsMemoryManager'] = None):
        object.__setattr__(self, 'memory_manager', memory_manager)

    def _get_memory_manager(self, override: Optional['LollmsMemoryManager']) -> Optional['LollmsMemoryManager']:
        return override if override is not None else getattr(self, 'memory_manager', None)

    def _memory_pre_turn(self, mm: Optional['LollmsMemoryManager'], user_message: Optional[str] = None, enable_deep_memory_pulling: bool = True, token_counter=None):
        if mm is None: return
        mm.apply_decay()
        if user_message and enable_deep_memory_pulling:
            mm.auto_pull_deep_memories(user_message)
        mm.enforce_budget(token_counter=token_counter)

    def _build_memory_system_instructions(self, mm: Optional['LollmsMemoryManager']) -> str:
        return mm.build_system_instructions() if mm else ""

    def _build_memory_context_block(self, mm: Optional['LollmsMemoryManager'], token_counter=None) -> str:
        if mm is None: return ""
        working = mm.build_working_zone(token_counter=token_counter)
        handles = mm.build_handles_zone(token_counter=token_counter)
        parts = [p for p in (working, handles) if p]
        return "\n".join(parts) if parts else ""

    def _process_memory_tags(self, text: str, mm: Optional['LollmsMemoryManager'], callback=None) -> tuple:
        if mm is None: return text, {}
        from lollms_client.lollms_types import MSG_TYPE
        cleaned, report = mm.process_llm_output(text)
        if any(report.values()):
            count = sum(len(v) for v in report.values() if isinstance(v, list))
            ASCIIColors.cyan(f"[Memory] Turn processed — operations: {count}")
            if callback:
                try: callback(json.dumps(report, default=str), MSG_TYPE.MSG_TYPE_INFO, {"type": "memory_update", "report": report})
                except Exception: pass
        return cleaned, report

    def _inject_memory_into_messages(self, messages: list, mm: Optional['LollmsMemoryManager'], format_type: str, token_counter=None) -> list:
        if mm is None or format_type not in ("openai_chat", "ollama_chat"): return messages
        block = self._build_memory_context_block(mm, token_counter)
        if not block: return messages
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                mem_msg = {"role": "user", "content": f"[MEMORY CONTEXT]\n{block}\n[/MEMORY CONTEXT]"}
                return messages[:i] + [mem_msg] + messages[i:]
        return messages

    # ── High-Level Memory Management Delegator Methods (UI-Ready) ──────────

    def add_memory(
        self,
        content: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        subject_group: Optional[str] = None,
        level: int = 1
    ) -> Optional[Dict]:
        """Manually add a memory directly to the persistent layer."""
        if self.memory_manager:
            return self.memory_manager.add(content, importance, tags, subject_group, level)
        return None

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Retrieve an individual memory by its ID."""
        if self.memory_manager:
            return self.memory_manager.get(memory_id)
        return None

    def update_memory(self, memory_id: str, new_content: str) -> Optional[Dict]:
        """Manually update the content of an active memory."""
        if self.memory_manager:
            return self.memory_manager.update(memory_id, new_content)
        return None

    def edit_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        level: Optional[int] = None,
        tags: Optional[List[str]] = None,
        subject_group: Optional[str] = None
    ) -> Optional[Dict]:
        """Manually edit any aspect of a memory (content, weight, tags, tier level)."""
        if self.memory_manager:
            return self.memory_manager.edit_memory(memory_id, content, importance, level, tags, subject_group)
        return None

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete/forget a memory."""
        if self.memory_manager:
            return self.memory_manager.delete(memory_id)
        return False

    def load_memory_to_working(self, memory_id: str) -> Optional[Dict]:
        """Promote a Deep Memory handle back to Working Memory (Level 1)."""
        if self.memory_manager:
            return self.memory_manager.load_to_working(memory_id)
        return None

    def list_all_memories(
        self,
        level: Optional[int] = None,
        search_query: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """List all memories with optional level filtering, searching, and pagination."""
        if self.memory_manager:
            return self.memory_manager.list_all(level, search_query, page, page_size)
        return {"total": 0, "page": page, "page_size": page_size, "pages": 0, "memories": []}

    def query_memories(self, text: str, top_k: int = 5, level: Optional[int] = None) -> List[Dict]:
        """Perform a keyword search on the database memories."""
        if self.memory_manager:
            return self.memory_manager.query(text, top_k, level)
        return []

    def dream_memories(self, lollms_client: Optional[Any] = None) -> Optional[Dict]:
        """Trigger an on-demand subconscious dream consolidation pass."""
        if self.memory_manager:
            return self.memory_manager.dream(lollms_client)
        return None
