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

    def _memory_pre_turn(self, mm: Optional['LollmsMemoryManager'], token_counter=None):
        if mm is None: return
        mm.apply_decay()
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