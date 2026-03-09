# lollms_discussion/_mixin_chat.py
# ChatMixin: simplified_chat() and chat() — the two high-level conversation methods.

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE
from ._message import LollmsMessage

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal callback helpers
# ---------------------------------------------------------------------------

def _cb(callback, text: str, msg_type: MSG_TYPE, meta: Optional[Dict] = None) -> bool:
    """Fire callback if present; returns False to signal stop if the callback does."""
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        # Callbacks may return None (fire-and-forget) or a bool (stop signal).
        if result is False:
            return False
    except Exception:
        pass
    return True


def _step_start(callback, text: str, meta: Optional[Dict] = None) -> Optional[str]:
    """Emit a STEP_START event; returns an event-id for later pairing with STEP_END."""
    event_id = str(uuid.uuid4())
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_START, {"id": event_id, **(meta or {})})
    return event_id


def _step_end(callback, text: str, event_id: Optional[str] = None,
              meta: Optional[Dict] = None):
    """Emit a STEP_END event, pairing with a previous STEP_START id."""
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_END, {"id": event_id, **(meta or {})})


def _info(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_INFO, meta)


def _warning(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_WARNING, meta)


# ---------------------------------------------------------------------------
# ChatMixin
# ---------------------------------------------------------------------------

class ChatMixin:
    """
    Provides simplified_chat() (intent-routing, RAG) and chat() (full agentic loop).

    Callback usage
    --------------
    Both methods now emit structured callback events throughout their lifecycle:

      MSG_TYPE_STEP_START / MSG_TYPE_STEP_END  – named reasoning / execution phases
      MSG_TYPE_INFO                            – lightweight status messages
      MSG_TYPE_WARNING                         – non-fatal issues
      MSG_TYPE_TOOL_CALL / MSG_TYPE_TOOL_OUTPUT– agentic tool calls
      MSG_TYPE_SCRATCHPAD                      – scratchpad state snapshots
      MSG_TYPE_SOURCES_LIST                    – RAG source lists
      MSG_TYPE_CHUNK                           – **streaming final answer tokens**

    The final answer is always *streamed* via MSG_TYPE_CHUNK when a callback is
    present, regardless of whether the overall flow was simple or agentic.
    """

    # ------------------------------------------------------------------ helpers

    def _stream_final_answer(self, callback, images, branch_tip_id, temperature, **kwargs):
        """
        Internal helper to handle the streaming bridge between the binding and UI.
        Fix: Correctly propagates stop signals (return False) from the callback to the binding.
        """
        caller_stream = kwargs.pop("stream", None)
        kwargs.pop("callback", None)
        kwargs.pop("streaming_callback", None)

        do_stream = (callback is not None) and (caller_stream is not False)
        collected = []

        def _streaming_relay(chunk, msg_type=None, meta=None):
            if isinstance(chunk, str):
                collected.append(chunk)
                # Important: return the result of the callback so the LLM knows if it should stop
                return _cb(callback, chunk, MSG_TYPE.MSG_TYPE_CHUNK, meta)
            return True

        result = self.lollmsClient.chat(
            self,
            images=images,
            branch_tip_id=branch_tip_id,
            stream=do_stream,
            streaming_callback=_streaming_relay if do_stream else None,
            temperature=temperature,
            **kwargs,
        )

        if do_stream:
            if isinstance(result, str) and result and not collected:
                _cb(callback, result, MSG_TYPE.MSG_TYPE_CHUNK)
                return result
            return "".join(collected) if collected else (result or "")
        else:
            return result if isinstance(result, str) else (result or "")

    # ------------------------------------------------------------------ simplified_chat

    def simplified_chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        mcps=None,
        rag_data_stores=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        use_rlm: bool = False,
        decision_temperature: float = 0.2,
        final_answer_temperature: float = 0.7,
        rag_top_k: int = 5,
        rag_min_similarity_percent: float = 0.5,
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simplified chat with structured callback events and streamed final answer.
        """
        callback = kwargs.get("streaming_callback")

        def is_fast(msg):
            m = msg.lower().strip()
            if len(m) < 20 and any(x in m for x in ["bonjour", "salut", "hello", "hi", "hey"]):
                return True
            return m in ["ok", "merci", "thanks", "cool", "yes", "no", "oui", "non"]

        # ── Inject extra system-prompt instructions ───────────────────────────
        extra_instructions = self._build_artefact_instructions()
        if enable_image_generation or enable_image_editing:
            extra_instructions += self._build_image_generation_instructions()
        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        # ── Inject agentic-mode hint so the model knows when to opt in ────────
        # This is built lazily (needs tools to be known) and appended later,
        # just before the first generation.  We set a flag here so the lazy
        # injection knows it is expected.
        _agentic_hint_needed = bool(tools)

        # ── Add user message ──────────────────────────────────────────────────
        user_msg = None
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=user_message,
                images=images,
                **kwargs,
            )

        # ── Fast-path: greeting / trivial ────────────────────────────────────
        if is_fast(user_message):
            _info(callback, "💬 Simple response path")
            text = self._stream_final_answer(
                callback, images, branch_tip_id, 0.1, **kwargs
            )
            ai = self.add_message(
                sender=personality.name if personality else "assistant",
                sender_type="assistant",
                content=text,
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
            )
        cleaned, affected = self._post_process_llm_response(
            text, ai, enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
        )
        if cleaned != text:
            ai.content = cleaned
            
        if affected and callback:
            # Notify the calling application immediately that artefacts were modified
            _cb(callback, json.dumps([a.get("title") for a in affected]), MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
            
        return {"user_message": user_msg, "ai_message": ai, "sources":[], "artefacts": affected}

        # ── Memory hit ────────────────────────────────────────────────────────
        if self.memory and user_message.lower() in self.memory.lower():
            _info(callback, "🧠 Answering from memory")
            text = self._stream_final_answer(
                callback, images, branch_tip_id, final_answer_temperature, **kwargs
            )
            ai = self.add_message(
                sender=personality.name if personality else "assistant",
                sender_type="assistant",
                content=text,
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
            )
        cleaned, affected = self._post_process_llm_response(
            text, ai, enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
        )
        if cleaned != text:
            ai.content = cleaned

        if affected and callback:
            # Notify the calling application immediately that artefacts were modified
            _cb(callback, json.dumps([a.get("title") for a in affected]), MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})

        return {
            "user_message": user_msg, "ai_message": ai,
            "sources":[], "artefacts": affected,
        }

        # ── Intent detection ──────────────────────────────────────────────────
        intent_id = _step_start(callback, "🔍 Analyzing intent…")
        intent = self.lollmsClient.generate_structured_content(
            prompt=user_message,
            schema={
                "needs_internal_knowledge": "boolean",
                "needs_full_documents":     "boolean",
                "needs_external_search":    "boolean",
                "reasoning":                "string",
            },
            temperature=decision_temperature,
        )
        _step_end(callback, "Intent analysis complete", intent_id, {"intent": intent})

        # ── RAG / data-zone injection ─────────────────────────────────────────
        scratchpad = ""
        sources: List[str] = []

        if intent and intent.get("needs_full_documents"):
            docs_id = _step_start(callback, "📚 Loading context documents…")
            for zone_name, zone_content in [
                ("user_data",         self.user_data_zone),
                ("discussion_data",   self.discussion_data_zone),
                ("personality_data",  self.personality_data_zone),
            ]:
                if zone_content:
                    scratchpad += f"\n--- {zone_name} ---\n{zone_content}\n"
                    sources.append(zone_name)
            _step_end(callback, f"Loaded {len(sources)} zone(s)", docs_id)

        if intent and intent.get("needs_external_search") and rag_data_stores:
            rag_id = _step_start(callback, "🔎 Searching external knowledge…")
            for name, fn in rag_data_stores.items():
                if callable(fn):
                    try:
                        res = fn(user_message)
                        if res:
                            scratchpad += f"\n--- {name} ---\n{str(res)}\n"
                            sources.append(name)
                            _info(callback, f"  ✅ Retrieved results from `{name}`")
                    except Exception as e:
                        _warning(callback, f"  ⚠️ `{name}` search error: {e}")
            if sources:
                _cb(callback, sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            _step_end(callback, "External search complete", rag_id, {"sources": sources})

        if scratchpad:
            self.personality_data_zone = scratchpad.strip()

        # ── Stream final answer ───────────────────────────────────────────────
        answer_id = _step_start(callback, "✍️ Generating answer…")
        final_text = self._stream_final_answer(
            callback, images, branch_tip_id, final_answer_temperature, **kwargs
        )
        _step_end(callback, "Answer generation complete", answer_id)

        if remove_thinking_blocks:
            final_text = self.lollmsClient.remove_thinking_blocks(final_text)

        ai = self.add_message(
            sender=personality.name if personality else "assistant",
            sender_type="assistant",
            content=final_text,
            parent_id=user_msg.id if user_msg else None,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"sources": sources} if sources else {},
        )
        cleaned, affected = self._post_process_llm_response(
            final_text, ai, enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
        )
        if cleaned != final_text:
            ai.content = cleaned
            
        if affected and callback:
            # Notify the calling application immediately that artefacts were modified
            _cb(callback, json.dumps([a.get("title") for a in affected]), MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})

        if scratchpad:
            self.personality_data_zone = ""

        return {
            "user_message": user_msg,
            "ai_message":   ai,
            "sources":      sources,
            "artefacts":    affected,
        }

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat with opt-in agentic mode.

        DEFAULT BEHAVIOUR — direct streaming
        ─────────────────────────────────────
        The model responds immediately via streaming tokens (MSG_TYPE_CHUNK).
        No reasoning loop, no tool calls, no overhead.

        OPT-IN AGENTIC MODE
        ────────────────────
        If the model decides it needs tools, it emits the self-closing tag
        ``<agentic/>`` anywhere in its first response.  The framework detects
        this, discards the partial text, and re-runs the full agentic loop
        (tool calls, RAG, scratchpad, composable answer) before streaming
        the final answer.

        The system prompt always describes the available tools so the model
        can make an informed decision about whether to activate agentic mode.
        """
        callback = kwargs.get("streaming_callback")

        # ── Inject extra system-prompt instructions ───────────────────────────
        extra_instructions = self._build_artefact_instructions()
        if enable_image_generation or enable_image_editing:
            extra_instructions += self._build_image_generation_instructions()
        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        # ── Inject agentic-mode hint so the model knows when to opt in ────────
        # This is built lazily (needs tools to be known) and appended later,
        # just before the first generation.  We set a flag here so the lazy
        # injection knows it is expected.
        _agentic_hint_needed = bool(tools)

        # ── Resolved generation parameters ───────────────────────────────────
        decision_temperature       = kwargs.get("decision_temperature",       0.3)
        final_answer_temperature   = kwargs.get("final_answer_temperature",   0.7)
        rag_top_k                  = kwargs.get("rag_top_k",                  5)
        rag_min_similarity_percent = kwargs.get("rag_min_similarity_percent", 0.5)
        max_rag_queries            = kwargs.get("max_rag_queries",            10)
        preflight_rag_enabled      = kwargs.get("preflight_rag",              True)

        rlm_enabled = False
        rlm_context_var_name = "USER_INPUT_CONTEXT"
        actual_user_content_for_llm = user_message

        if tools:
            has_python_exec = any(
                t.get("name") == "python_exec"
                for t in tools.values() if isinstance(t, dict)
            )
            has_llm_query = any(
                t.get("name") == "llm_query"
                for t in tools.values() if isinstance(t, dict)
            )
            rlm_enabled = has_python_exec and has_llm_query

        if rlm_enabled and len(user_message) > 10000:
            preview_length = 500
            actual_user_content_for_llm = "\n".join([
                "<RLM_STUB>",
                f"Large input ({len(user_message):,} chars) stored in `{rlm_context_var_name}`.",
                "PREVIEW:", user_message[:preview_length], "...",
                "Use python_exec() to access the full content.",
                "</RLM_STUB>",
            ])

        # ── Add user message ──────────────────────────────────────────────────
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=actual_user_content_for_llm,
                images=images,
                **kwargs,
            )
            if rlm_enabled and len(user_message) > 10000:
                user_msg.metadata["rlm_full_content"] = user_message
                user_msg.metadata["rlm_var_name"] = rlm_context_var_name
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images = user_msg.get_active_images()

        # ── Document extraction helper ────────────────────────────────────────
        def extract_documents(zone_content):
            if not zone_content:
                return []
            docs = []
            for m in re.findall(
                r"--- Document: (.+?) ---\n(.*?)\n--- End Document: \1 ---",
                zone_content, re.DOTALL,
            ):
                docs.append({
                    "name":        m[0].strip(),
                    "content":     m[1].strip(),
                    "size":        len(m[1].strip()),
                    "token_count": self.lollmsClient.count_tokens(m[1].strip()),
                })
            return docs

        all_documents = []
        for zone_content, zone_label in [
            (self.discussion_data_zone, "discussion"),
            (self.user_data_zone, "user"),
            (
                self.personality_data_zone
                if not callable(getattr(personality, "data_source", None))
                else None,
                "personality",
            ),
        ]:
            if zone_content:
                docs = extract_documents(zone_content)
                for d in docs:
                    d["zone"] = zone_label
                all_documents.extend(docs)

        # ── Build tool registry ───────────────────────────────────────────────
        tool_registry:     Dict[str, Any] = {}
        tool_descriptions: List[str]      = []
        rag_registry:      Dict[str, Any] = {}
        rag_tool_specs:    Dict[str, Any] = {}

        # Composable answer state
        composable_answer = {"sections": [], "complete": False, "last_updated": None}
        scratchpad_state  = {"notes": {}, "history": [], "assumptions": {}, "corrections": []}
        collected_sources: List[Dict]     = []
        queries_performed: List[Dict]     = []
        self_corrections:  List[Dict]     = []

        def append_to_answer(content, section_id=None, sources=None):
            if section_id is None:
                section_id = f"section_{len(composable_answer['sections']) + 1}"
            section = {
                "id":           section_id,
                "content":      content,
                "timestamp":    datetime.now().isoformat(),
                "sources_used": sources or [],
                "status":       "active",
            }
            composable_answer["sections"].append(section)
            composable_answer["last_updated"] = datetime.now().isoformat()
            return {
                "success":        True,
                "section_id":     section_id,
                "total_sections": len(composable_answer["sections"]),
                "current_length": sum(
                    len(s["content"]) for s in composable_answer["sections"]
                    if s["status"] == "active"
                ),
            }

        def update_answer_section(section_id, new_content, reason=None):
            for s in composable_answer["sections"]:
                if s["id"] == section_id:
                    old = s["content"]
                    s["content"]      = new_content
                    s["updated_at"]   = datetime.now().isoformat()
                    s["update_reason"]= reason or "Self-correction"
                    corr = {
                        "section_id":   section_id,
                        "old_content":  old[:200],
                        "new_content":  new_content[:200],
                        "reason":       reason,
                        "timestamp":    datetime.now().isoformat(),
                    }
                    self_corrections.append(corr)
                    scratchpad_state["corrections"].append(corr)
                    return {"success": True, "section_id": section_id}
            return {"success": False, "error": f"Section '{section_id}' not found"}

        def remove_answer_section(section_id, reason=None):
            for s in composable_answer["sections"]:
                if s["id"] == section_id:
                    s["status"] = "removed"
                    return {"success": True, "section_id": section_id}
            return {"success": False, "error": f"Section '{section_id}' not found"}

        def get_current_answer():
            active    = [s for s in composable_answer["sections"] if s.get("status") == "active"]
            full_text = "\n\n".join(s["content"] for s in active)
            return {
                "success":        True,
                "full_text":      full_text,
                "sections":       active,
                "total_sections": len(active),
                "total_length":   len(full_text),
                "last_updated":   composable_answer.get("last_updated"),
            }

        def update_scratchpad(key, value, category="notes"):
            if category == "assumptions":
                scratchpad_state["assumptions"][key] = {
                    "value":     value,
                    "status":    "UNCERTAIN",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                scratchpad_state["notes"][key] = value
            scratchpad_state["history"].append({
                "action": "update", "category": category,
                "key": key, "value": value,
                "timestamp": datetime.now().isoformat(),
            })
            return {"success": True, "key": key, "category": category}

        def update_assumption_status(assumption_key, status, reason=None):
            if assumption_key not in scratchpad_state["assumptions"]:
                return {"success": False, "error": f"Assumption '{assumption_key}' not found"}
            scratchpad_state["assumptions"][assumption_key]["status"] = status
            if reason:
                scratchpad_state["assumptions"][assumption_key]["reason"] = reason
            return {"success": True, "assumption": assumption_key, "new_status": status}

        def get_scratchpad(category=None):
            if category == "assumptions":
                return {"success": True, "assumptions": scratchpad_state["assumptions"]}
            elif category == "notes":
                return {"success": True, "notes": scratchpad_state["notes"]}
            elif category == "history":
                return {"success": True, "history": scratchpad_state["history"]}
            return {"success": True, "scratchpad": scratchpad_state}

        def remove_scratchpad_entry(key, category="notes"):
            target = (
                scratchpad_state["assumptions"]
                if category == "assumptions"
                else scratchpad_state["notes"]
            )
            if key in target:
                del target[key]
                return {"success": True, "removed": key}
            return {"success": False, "error": f"'{key}' not found in {category}"}

        # ── Process tools parameter ───────────────────────────────────────────
        if tools:
            for tool_name, tool_spec in tools.items():
                if not isinstance(tool_spec, dict):
                    continue
                name        = tool_spec.get("name", tool_name)
                description = tool_spec.get("description", f"Execute {name}")
                parameters  = tool_spec.get("parameters", [])
                output_spec = tool_spec.get("output", [])
                callable_fn = tool_spec.get("callable")
                if not callable(callable_fn):
                    continue
                param_strs = []
                for p in parameters:
                    pn, pt = p.get("name", "arg"), p.get("type", "any")
                    opt, dv = p.get("optional", False), p.get("default")
                    if opt and dv is not None:
                        param_strs.append(f"{pn}: {pt} = {dv}")
                    elif opt:
                        param_strs.append(f"{pn}: {pt} (optional)")
                    else:
                        param_strs.append(f"{pn}: {pt}")
                is_rag_tool = any(out.get("name") == "sources" for out in output_spec)

                def create_wrapper(fn, params_spec, is_rag):
                    def wrapped(**kw):
                        try:
                            call_args = {}
                            for p in params_spec:
                                pn = p.get("name")
                                if pn in kw:
                                    call_args[pn] = kw[pn]
                                elif not p.get("optional", False):
                                    return {"error": f"Missing required parameter: {pn}", "success": False}
                                elif "default" in p:
                                    call_args[pn] = p["default"]
                            result = fn(**call_args)
                            if "success" not in result:
                                result["success"] = "error" not in result or not result.get("error")
                            return result
                        except Exception as e:
                            return {"error": str(e), "success": False}
                    return wrapped

                wrapped = create_wrapper(callable_fn, parameters, is_rag_tool)
                tool_registry[name] = wrapped
                tool_descriptions.append(f"- {name}({', '.join(param_strs)}): {description}")
                if is_rag_tool:
                    rag_registry[name] = wrapped
                    rag_tool_specs[name] = {
                        "default_top_k":    rag_top_k,
                        "default_min_sim":  rag_min_similarity_percent,
                    }

        # ── Legacy personality.data_source support ────────────────────────────
        if personality and callable(getattr(personality, "data_source", None)):
            tool_name = "search_personality_knowledge"

            def create_legacy_rag(rag_fn, kb_name):
                def wrapped_rag(query):
                    try:
                        results = rag_fn(query)
                        formatted = []
                        if isinstance(results, list):
                            for chunk in results:
                                score = float(chunk.get("score", chunk.get("value", 1.0)))
                                if score > 1.0:
                                    score = 1.0 if score > 100.0 else score / 100.0
                                formatted.append({
                                    "content":  chunk.get("content", str(chunk)),
                                    "score":    score,
                                    "source":   kb_name,
                                    "metadata": chunk.get("metadata", {}),
                                })
                            formatted.sort(key=lambda x: x["score"], reverse=True)
                            formatted = [
                                r for r in formatted[:rag_top_k]
                                if r["score"] >= rag_min_similarity_percent
                            ]
                        else:
                            formatted = [{
                                "content":  str(results),
                                "score":    1.0,
                                "source":   kb_name,
                                "metadata": {},
                            }]
                        return {
                            "status": "success", "sources": formatted,
                            "count": len(formatted), "query": query, "success": True,
                        }
                    except Exception as e:
                        return {"status": "error", "error": str(e), "query": query, "success": False}
                return wrapped_rag

            wrapped_fn = create_legacy_rag(personality.data_source, "Personality Knowledge Base")
            tool_registry[tool_name] = wrapped_fn
            rag_registry[tool_name]  = wrapped_fn
            rag_tool_specs[tool_name]= {
                "default_top_k":   rag_top_k,
                "default_min_sim": rag_min_similarity_percent,
            }
            tool_descriptions.append(
                f"- {tool_name}(query: str): Search personality knowledge base"
            )

        # ── Personality: veracity + preflight RAG ────────────────────────────
        if personality:
            veracity_addendum = "\n".join([
                "",
                "=== VERACITY & ATTRIBUTION REQUIREMENTS ===",
                "Cite retrieved sources as [1],[2]… Use 'From my understanding…' for general knowledge.",
                "Never fabricate facts. Say 'I don't know' when uncertain.",
                "=== END ===", "",
            ])
            object.__setattr__(
                self, "_system_prompt",
                (personality.system_prompt or "") + veracity_addendum + extra_instructions,
            )

            if preflight_rag_enabled and callable(getattr(personality, "data_source", None)):
                preflight_id = _step_start(callback, "🔍 Pre-flight knowledge retrieval…")
                context_for_query = self.export("markdown", suppress_system_prompt=True)
                try:
                    query_json = self.lollmsClient.generate_structured_content(
                        prompt=context_for_query[-2000:]
                               + "\nGenerate a concise search query (JSON).",
                        schema={"query": "Your concise search query string"},
                        system_prompt="Output only JSON.",
                        temperature=0.1,
                    )
                    if query_json and "query" in query_json:
                        rag_tool = rag_registry.get("search_personality_knowledge")
                        if rag_tool:
                            result = rag_tool(query=query_json["query"])
                            if result.get("success"):
                                fmt = ""
                                for idx, chunk in enumerate(result.get("sources", [])):
                                    fmt += (
                                        f"[Source {idx+1}] ({chunk.get('source','?')},"
                                        f" {chunk.get('score',0):.2f})\n"
                                        f"{chunk.get('content','')}\n\n"
                                    )
                                    collected_sources.append({
                                        "title":           f"Preflight #{idx+1}",
                                        "content":         chunk.get("content", ""),
                                        "source":          chunk.get("source", ""),
                                        "query":           query_json["query"],
                                        "relevance_score": chunk.get("score", 0),
                                        "index":           idx + 1,
                                        "phase":           "preflight",
                                    })
                                if fmt:
                                    self.personality_data_zone = (
                                        fmt.strip()
                                        + "\n\nIMPORTANT: Cite sources as [1],[2],…"
                                    )
                                if collected_sources:
                                    _cb(
                                        callback,
                                        collected_sources,
                                        MSG_TYPE.MSG_TYPE_SOURCES_LIST,
                                    )
                except Exception as e:
                    trace_exception(e)
                _step_end(callback, "Pre-flight retrieval complete", preflight_id,
                          {"source_count": len(collected_sources)})

        # ── Control tools ─────────────────────────────────────────────────────
        # NOTE: "final_answer" is a *signal* only — the actual answer generation
        #       is handled below via streaming chat, not via structured content.
        tool_registry["final_answer"] = lambda: {
            "status":  "final",
            "answer":  get_current_answer()["full_text"]
                       if composable_answer["sections"] else None,
            "success": True,
        }
        tool_descriptions.append("- final_answer(): Signal that the answer is ready to be streamed")
        tool_registry["request_clarification"] = lambda question: {
            "status": "clarification", "question": question, "success": True,
        }
        tool_descriptions.append(
            "- request_clarification(question: str): Ask user for clarification"
        )

        # ====================================================================
        #  UNIFIED STREAMING WITH INLINE TOOL CALLS
        #
        #  The model streams freely.  Whenever it emits a tool call tag:
        #    <tool_call>{"name":"...","parameters":{...}}</tool_call>
        #  we:
        #    1. Stop receiving tokens (return False from callback)
        #    2. Execute the tool
        #    3. Inject a <tool_result> block into the conversation
        #    4. Resume generation (call _stream_final_answer again)
        #  This continues until the model finishes without a tool call.
        #
        #  The system prompt describes available tools in this format so the
        #  model knows exactly how to invoke them.
        # ====================================================================
        _has_docs = len(all_documents) > 0

        # ── Inject tool catalogue into system prompt ─────────────────────────
        if tools and tool_descriptions:
            _tc = [
                "\n\n## Available Tools",
                "To use a tool, emit a tool call tag anywhere in your response:",
                '  <tool_call>{"name": "tool_name", "parameters": {"key": "value"}}</tool_call>',
                "The framework will execute the tool and inject the result so you can continue.",
                "You may call multiple tools sequentially.",
                "Use tools whenever the task requires real-world actions",
                "(running commands, reading/writing files, git, installing packages, etc.).",
                "",
                "### Tool list",
            ] + tool_descriptions
            _tool_catalogue = "\n".join(_tc)
            _cur_sp = self._system_prompt or ""
            if "<tool_call>" not in _cur_sp:
                object.__setattr__(self, "_system_prompt", _cur_sp + _tool_catalogue)

        tools_summary = "\n".join(tool_descriptions)
        if self.max_context_size is not None:
            self.summarize_and_prune(self.max_context_size)

        start_time            = datetime.now()
        is_agentic_turn       = False
        tool_calls_this_turn: List[Dict] = []
        final_content         = ""
        final_raw_response    = ""
        final_scratchpad      = None

        # ── Streaming loop — runs until model produces no more tool calls ─────
        _TC_OPEN  = "<tool_call>"
        _TC_CLOSE = "</tool_call>"
        _accumulated_full = ""   # everything the model has said across all passes

        # Context injected between passes (tool results)
        _injected_results: List[str] = []

        _max_tool_rounds = max_reasoning_steps
        _round = 0

        while _round < _max_tool_rounds:
            _round += 1
            _stream_buf   = []          # chunks for this pass
            _tool_pending = []          # chars after <tool_call> open tag
            _in_tool_call = False
            _tool_trigger = False       # set True when </tool_call> is found

            def _inline_relay(chunk, msg_type=None, meta=None):
                nonlocal _in_tool_call, _tool_trigger
                if not isinstance(chunk, str):
                    return True

                _stream_buf.append(chunk)
                _so_far = "".join(_stream_buf)

                if not _in_tool_call:
                    if _TC_OPEN in _so_far:
                        _in_tool_call = True
                        _pre = _so_far[:_so_far.index(_TC_OPEN)]
                        if _pre and callback is not None:
                            _cb(callback, _pre, MSG_TYPE.MSG_TYPE_CHUNK)
                        return True 
                    else:
                        # Ensure chunks are passed through immediately unless they look like a partial tag
                        _could_be_partial = any(
                            _TC_OPEN.startswith(_so_far[-i:])
                            for i in range(1, len(_TC_OPEN) + 1)
                            if i <= len(_so_far)
                        )
                        if not _could_be_partial and callback is not None:
                            return _cb(callback, chunk, MSG_TYPE.MSG_TYPE_CHUNK)
                        return True
                else:
                    if _TC_CLOSE in _so_far:
                        _tool_trigger = True
                        return False  # Signal to _stream_final_answer to stop the LLM

                    return True

            # Build context with any injected tool results from previous rounds
            _extra_context = ""
            if _injected_results:
                _extra_context = "\n\n".join(_injected_results)
                # Temporarily append to personality_data_zone
                _prior_pdz = self.personality_data_zone
                self.personality_data_zone = (
                    (_prior_pdz or "") + "\n\n" + _extra_context
                )

            _pass_text = self._stream_final_answer(
                _inline_relay,
                images, branch_tip_id, final_answer_temperature, **kwargs
            )

            # Restore data zone
            if _injected_results:
                self.personality_data_zone = _prior_pdz

            _so_far = "".join(_stream_buf)
            _accumulated_full += _so_far

            if not _tool_trigger:
                # No tool call — model is done
                break

            # ── Parse and execute the tool call ──────────────────────────────
            is_agentic_turn = True
            try:
                _open_idx  = _so_far.index(_TC_OPEN) + len(_TC_OPEN)
                _close_idx = _so_far.index(_TC_CLOSE)
                _call_json = _so_far[_open_idx:_close_idx].strip()
                _call_data = json.loads(_call_json)
                _tool_name = _call_data.get("name", "")
                _tool_params = _call_data.get("parameters", {})
            except Exception as e:
                _warning(callback, f"⚠️ Failed to parse tool call: {e}")
                break

            # Emit a UI-friendly Tool Call event
            _cb(callback, f"🔧 {_tool_name}",
                MSG_TYPE.MSG_TYPE_TOOL_CALL, {"tool": _tool_name, "params": _tool_params})

            # Start a visual "Step" for the tool execution
            _step_label = f"Running tool: {_tool_name.replace('_', ' ').title()}"
            _tool_step_id = _step_start(callback, _step_label, {"tool": _tool_name})

            if _tool_name not in tool_registry:
                _warning(callback, f"⚠️ Unknown tool: {_tool_name}")
                _step_end(callback, f"Error: Tool '{_tool_name}' not found", _tool_step_id, {"status": "failed"})
                _injected_results.append(
                    f'<tool_result name="{_tool_name}">Error: tool not found</tool_result>'
                )
            else:
                try:
                    _tool_result = tool_registry[_tool_name](**_tool_params)
                    _result_str  = json.dumps(_tool_result, indent=2)[:2000]
                    tool_calls_this_turn.append({
                        "name":   _tool_name,
                        "params": _tool_params,
                        "result": _tool_result,
                    })
                    
                    # Log internal tool output event
                    _cb(callback, _result_str,
                        MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, {"tool": _tool_name, "result": _tool_result})
                    
                    # Finish the visual Step
                    _step_end(callback, f"Completed: {_tool_name}", _tool_step_id, {"status": "success"})
                    
                    _injected_results.append(
                        f'<tool_result name="{_tool_name}">{_result_str}</tool_result>'
                    )

                    # RAG bookkeeping
                    if _tool_name in rag_registry:
                        _q_str = _tool_params.get("query", "")
                        _res_count = len(_tool_result.get("sources", []))
                        queries_performed.append({
                            "step": _round, "tool": _tool_name,
                            "query": _q_str, "result_count": _res_count,
                        })
                        if _tool_result.get("success"):
                            for _idx, _doc in enumerate(_tool_result.get("sources", [])):
                                collected_sources.append({
                                    "title":           _doc.get("source", f"{_tool_name}#{len(collected_sources)+1}"),
                                    "content":         _doc.get("content", ""),
                                    "source":          _doc.get("source", ""),
                                    "query":           _q_str,
                                    "relevance_score": _doc.get("score", 0),
                                    "index":           len(collected_sources) + 1,
                                    "tool":            _tool_name,
                                })
                except Exception as e:
                    trace_exception(e)
                    _warning(callback, f"⚠️ Tool error ({_tool_name}): {e}")
                    # Close the step with error information
                    _step_end(callback, f"Error running {_tool_name}: {e}", _tool_step_id, {"status": "error"})
                    _injected_results.append(
                        f'<tool_result name="{_tool_name}">Error: {e}</tool_result>'
                    )

        # ── Clean up tool call tags from the final displayed text ─────────────
        import re as _re
        _clean = _re.sub(
            r"<tool_call>.*?</tool_call>", "", _accumulated_full,
            flags=_re.DOTALL
        ).strip()

        if remove_thinking_blocks:
            _clean = self.lollmsClient.remove_thinking_blocks(_clean)

        final_content      = _clean
        final_raw_response = _accumulated_full
        final_scratchpad   = scratchpad_state if is_agentic_turn else None

        # ====================================================================
        #  Assemble message & post-process
        # ====================================================================
        end_time    = datetime.now()
        duration    = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(final_content)
        tok_per_sec = (token_count / duration) if duration > 0 else 0

        message_meta: Dict[str, Any] = {
            "mode": (
                "rlm_agentic" if rlm_enabled
                else ("agentic" if is_agentic_turn else "direct")
            ),
            "duration_seconds":  duration,
            "token_count":       token_count,
            "tokens_per_second": tok_per_sec,
        }
        if tool_calls_this_turn:
            message_meta["tool_calls"] = tool_calls_this_turn
        if collected_sources:
            message_meta["sources"] = collected_sources
        if queries_performed:
            message_meta["query_history"] = queries_performed
        if is_agentic_turn:
            message_meta["scratchpad"] = scratchpad_state
        if self_corrections:
            message_meta["self_corrections"] = self_corrections

        ai_message_obj = self.add_message(
            sender=personality.name if personality else "assistant",
            sender_type="assistant",
            content=final_content,
            raw_content=final_raw_response,
            scratchpad=json.dumps(final_scratchpad, indent=2) if final_scratchpad else None,
            tokens=token_count,
            generation_speed=tok_per_sec,
            parent_id=user_msg.id,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata=message_meta,
        )

        cleaned_content, affected_artefacts = self._post_process_llm_response(
            final_content, ai_message_obj,
            enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
        )
        if cleaned_content != final_content:
            ai_message_obj.content = cleaned_content

        if affected_artefacts:
            message_meta["artefacts_modified"] =[a.get("title") for a in affected_artefacts]
            ai_message_obj.metadata = message_meta
            
            if callback:
                # Notify the calling application immediately that artefacts were modified
                _cb(callback, json.dumps(message_meta["artefacts_modified"]), MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected_artefacts})

        if self._is_db_backed and self.autosave:
            self.commit()

        return {
            "user_message":     user_msg,
            "ai_message":       ai_message_obj,
            "sources":          collected_sources,
            "scratchpad":       scratchpad_state if is_agentic_turn else None,
            "self_corrections": self_corrections if self_corrections else None,
            "artefacts":        affected_artefacts,
        }