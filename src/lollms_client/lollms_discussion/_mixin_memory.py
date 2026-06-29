# lollms_discussion/_mixin_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# MemoryMixin — integrates LollmsMemoryManager into LollmsDiscussion.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import re
from ascii_colors import ASCIIColors
from datetime import datetime
from lollms_client.lollms_memory import FailureMemory

if TYPE_CHECKING:
    from lollms_client.lollms_memory import LollmsMemoryManager

class MemoryMixin:
    """Adds multi-level persistent memory to LollmsDiscussion."""

    def _init_memory(self, memory_manager: Optional['LollmsMemoryManager'] = None):
        object.__setattr__(self, 'memory_manager', memory_manager)
        # Initialize the transient short-term FailureMemory
        if not hasattr(self, "failure_memory") or not self.failure_memory:
            object.__setattr__(self, "failure_memory", FailureMemory())

    def _trigger_evolutionary_reflection(self, tool_name: str, params: Dict[str, Any], result: str):
        """Generates a high-density lesson learned from successful recovery and saves it to long-term memory."""
        if not getattr(self, "memory_manager", None) or not getattr(self, "lollmsClient", None):
            return

        failures = [f for f in self.failure_memory.failures if f["tool_name"] == tool_name]
        if not failures:
            return

        failure_details = "\n".join([
            f"Attempt {idx+1}: Params={json.dumps(f['params'])} | Error={f['error']}"
            for idx, f in enumerate(failures)
        ])

        prompt = (
            "You are a Synaptic Reflection Engine.\n"
            f"A tool execution for '{tool_name}' just succeeded after previous failures in this session.\n\n"
            "=== PREVIOUS FAILURES ===\n"
            f"{failure_details}\n\n"
            "=== SUCCESSFUL ATTEMPT ===\n"
            f"Params: {json.dumps(params)}\n"
            f"Result: {str(result)[:1000]}\n\n"
            "Your task is to write a concise, high-density technical lesson summarizing why "
            "the previous attempts failed and how the successful configuration resolved it. "
            "Focus on specific rules, parameter mappings, or platform-specific constraints.\n\n"
            "Requirements:\n"
            "1. Output ONLY the technical lesson as a clear, standalone paragraph.\n"
            "2. Keep it professional, objective, and action-oriented.\n"
            "3. Do not write preamble, conversational text, or explanations."
        )
        try:
            lesson = self.lollmsClient.generate_text(prompt=prompt, temperature=0.1).strip()
            if lesson:
                self.memory_manager.add(
                    content=f"Technical Lesson Learned: {lesson}",
                    importance=1.0,
                    tags=["technical_lesson", "recovery", tool_name],
                    subject=tool_name,
                    predicate="IMPLEMENTS",
                    obj="recovery_pattern"
                )
                # Clear failures for this tool once reflected to prevent duplicate reflections
                self.failure_memory.failures = [f for f in self.failure_memory.failures if f["tool_name"] != tool_name]
        except Exception as e:
            ASCIIColors.warning(f"Failed to generate evolutionary reflection: {e}")

    def _trigger_evolutionary_reflection(self, tool_name: str, params: Dict[str, Any], result: str):
        """Generates a high-density lesson learned from successful recovery and saves it to long-term memory."""
        if not getattr(self, "memory_manager", None) or not getattr(self, "lollmsClient", None):
            return

        failures = [f for f in self.failure_memory.failures if f["tool_name"] == tool_name]
        if not failures:
            return

        failure_details = "\n".join([
            f"Attempt {idx+1}: Params={json.dumps(f['params'])} | Error={f['error']}"
            for idx, f in enumerate(failures)
        ])

        prompt = (
            "You are a Synaptic Reflection Engine.\n"
            f"A tool execution for '{tool_name}' just succeeded after previous failures in this session.\n\n"
            "=== PREVIOUS FAILURES ===\n"
            f"{failure_details}\n\n"
            "=== SUCCESSFUL ATTEMPT ===\n"
            f"Params: {json.dumps(params)}\n"
            f"Result: {str(result)[:1000]}\n\n"
            "Your task is to write a concise, high-density technical lesson summarizing why "
            "the previous attempts failed and how the successful configuration resolved it. "
            "Focus on specific rules, parameter mappings, or platform-specific constraints.\n\n"
            "Requirements:\n"
            "1. Output ONLY the technical lesson as a clear, standalone paragraph.\n"
            "2. Keep it professional, objective, and action-oriented.\n"
            "3. Do not write preamble, conversational text, or explanations."
        )
        try:
            lesson = self.lollmsClient.generate_text(prompt=prompt, temperature=0.1).strip()
            if lesson:
                self.memory_manager.add(
                    content=f"Technical Lesson Learned: {lesson}",
                    importance=1.0,
                    tags=["technical_lesson", "recovery", tool_name],
                    subject=tool_name,
                    predicate="IMPLEMENTS",
                    obj="recovery_pattern"
                )
                ASCIIColors.success(f"[FailureMemory] Promoted successful recovery lesson to long-term memory: '{lesson[:60]}...'")
                # Clear failures for this tool once reflected to prevent duplicate reflections
                self.failure_memory.failures = [f for f in self.failure_memory.failures if f["tool_name"] != tool_name]
        except Exception as e:
            ASCIIColors.warning(f"Failed to generate evolutionary reflection: {e}")

    def _get_memory_manager(self, override: Optional['LollmsMemoryManager']) -> Optional['LollmsMemoryManager']:
        return override if override is not None else getattr(self, 'memory_manager', None)

    def _memory_pre_turn(self, mm: Optional['LollmsMemoryManager'], user_message: Optional[str] = None, enable_deep_memory_pulling: bool = True, token_counter=None):
        if mm is None: return
        import time
        start_time = time.time()
        try:
            mm.apply_decay()
            if time.time() - start_time > 4.0:
                raise TimeoutError("Memory operations exceeded pre-turn safety budget.")

            if user_message and enable_deep_memory_pulling:
                mm.auto_pull_deep_memories(user_message)

            if time.time() - start_time > 4.5:
                raise TimeoutError("Memory operations exceeded pre-turn safety budget.")

            mm.enforce_budget(token_counter=token_counter)
        except Exception:
            pass

    def _build_memory_system_instructions(self, mm: Optional['LollmsMemoryManager']) -> str:
        return mm.build_system_instructions() if mm else ""

    def _build_memory_context_block(self, mm: Optional['LollmsMemoryManager'], token_counter=None) -> str:
        if mm is None: return ""
        working = mm.build_working_zone(token_counter=token_counter)
        handles = mm.build_handles_zone(token_counter=token_counter)
        parts = [p for p in (working, handles) if p]
        return "\n".join(parts) if parts else ""

    def _is_turn_worth_memorizing(self, user_text: str, ai_text: str) -> bool:
        """Filters out conversational noise, small talk, errors, and status updates."""
        u_clean = user_text.lower().strip()
        a_clean = ai_text.lower().strip()

        # 1. Skip empty or extremely short turns
        if len(u_clean) < 10 or len(a_clean) < 15:
            return False

        # 2. Filter common greetings, pleasantries, and trivial social fluff
        fluff_patterns = [
            r"^(hello|hi|hey|bonjour|salut|good morning|good afternoon|good evening|howdy|yo)\b",
            r"^(thanks|thank you|merci|cool|ok|okay|yes|no|oui|non|perfect|awesome|great|super|agreed)\b",
            r"^(bye|goodbye|au revoir|see you)\b"
        ]
        for pattern in fluff_patterns:
            if re.match(pattern, u_clean) and len(u_clean) < 30:
                return False
            if re.match(pattern, a_clean) and len(a_clean) < 40:
                return False

        # 3. Filter structural execution errors, rejections, and raw debugging/diff blocks
        noise_terms = [
            "rejected:", "failed to match", "search/replace", "error executing", 
            "invalid token", "traceback", "syntaxerror", "connection refused",
            "log mimicry", "processing_open", "processing_close"
        ]
        if any(term in u_clean or term in a_clean for term in noise_terms):
            return False

        # 4. Filter code-heavy responses (e.g. containing markdown code blocks, script blocks, or CSS)
        if "```" in a_clean or "<script" in a_clean or "canvas" in a_clean or "const " in a_clean:
            return False

        return True

    def _save_episodic_memory_turn(self, user_text: str, ai_text: str, mm: Optional['LollmsMemoryManager']):
        if mm is None:
            return
        # Clean AI text of processing tags or HTML tags
        clean_ai = re.sub(r'<processing.*?>.*?</processing>', '', ai_text, flags=re.DOTALL)
        clean_ai = re.sub(r'<[^>]+>', '', clean_ai).strip()
        clean_user = user_text.strip()
        if not clean_user or not clean_ai:
            return

        if not self._is_turn_worth_memorizing(clean_user, clean_ai):
            return

        episode_content = f"Event/Interaction on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC:\nUser asked: \"{clean_user}\"\nAI responded: \"{clean_ai}\""
        mm.add(content=episode_content, importance=0.8, tags=["episode", "interaction"], level=1)

    def _process_memory_tags(self, text: str, mm: Optional['LollmsMemoryManager'], callback=None) -> tuple:
        if mm is None: return text, {}
        from lollms_client.lollms_types import MSG_TYPE
        cleaned, report = mm.process_llm_output(text)
        if any(report.values()):
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

    def clear_memories_of_level(self, level: int) -> bool:
        """Delete all memories of the specified level."""
        if self.memory_manager:
            self.memory_manager.clear_level(level)
            return True
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

    def add_memory_relationship(self, source_id: str, target_id: str, relationship_type: str = "RELATED_TO", weight: float = 1.0, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Create an explicit graph relationship between two memories."""
        if self.memory_manager:
            return self.memory_manager.add_relationship(source_id, target_id, relationship_type, weight, metadata)
        return None

    def remove_memory_relationship(self, source_id: str, target_id: str) -> bool:
        """Remove a graph relationship between two memories."""
        if self.memory_manager:
            return self.memory_manager.remove_relationship(source_id, target_id)
        return False

    def get_memory_relationships(self, memory_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Get all relationships for a memory node."""
        if self.memory_manager:
            return self.memory_manager.get_relationships(memory_id, relationship_type)
        return []

    def traverse_memory_graph(self, start_id: str, max_depth: int = 3, relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Perform BFS traversal of the memory graph from a starting node."""
        if self.memory_manager:
            return self.memory_manager.traverse_graph(start_id, max_depth, relationship_types)
        return []

    def get_high_centrality_memories(self, top_k: int = 10, level: Optional[int] = None) -> List[Dict]:
        """Get memories with highest graph centrality (most connected/important)."""
        if self.memory_manager:
            return self.memory_manager.get_high_centrality_memories(top_k, level)
        return []

    def recalculate_memory_centrality(self):
        """Recalculate centrality scores for all memories (expensive operation)."""
        if self.memory_manager:
            self.memory_manager.recalculate_all_centrality()
