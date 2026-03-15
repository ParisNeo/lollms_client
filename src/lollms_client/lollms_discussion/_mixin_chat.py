# lollms_discussion/_mixin_chat.py
# ChatMixin: simplified_chat() and chat() — the two high-level conversation methods.

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_personality import NullPersonality
from ._message import LollmsMessage

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal callback helpers
# ---------------------------------------------------------------------------

def _cb(callback, text: str, msg_type: MSG_TYPE, meta: Optional[Dict] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        if result is False:
            return False
    except Exception as e:
        trace_exception(e)
    return True


def _step_start(callback, text: str, meta: Optional[Dict] = None) -> Optional[str]:
    event_id = str(uuid.uuid4())
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_START, {"id": event_id, **(meta or {})})
    return event_id


def _step_end(callback, text: str, event_id: Optional[str] = None,
              meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_END, {"id": event_id, **(meta or {})})


def _info(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_INFO, meta)


def _warning(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_WARNING, meta)


def _extract_content_title(content: str, max_len: int = 80) -> Optional[str]:
    if not content or not content.strip():
        return None
    lines     = content.splitlines()
    non_blank = [l for l in lines if l.strip()]
    if not non_blank:
        return None

    def _clean(s: str) -> str:
        s = re.sub(r"[*_`#~]+", "", s).strip(" \t\r\n|>")
        if len(s) > max_len:
            s = s[:max_len].rsplit(" ", 1)[0] + "…"
        return s or ""

    for line in non_blank[:30]:
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    for i, line in enumerate(non_blank[:-1]):
        nxt = non_blank[i + 1]
        if re.match(r"^[=\-~^\"'`#*+]{4,}$", nxt.strip()) and line.strip():
            t = _clean(line)
            if t:
                return t

    fm = re.search(
        r'(?:^|[\n,{])\s*["\']?title["\']?\s*[:=]\s*["\']?([^\n"\'},]+)',
        content[:1000], re.IGNORECASE,
    )
    if fm:
        t = _clean(fm.group(1))
        if t:
            return t

    for pat in [r"<title[^>]*>([^<]+)</title>", r"<h1[^>]*>([^<]+)</h1>"]:
        m = re.search(pat, content[:2000], re.IGNORECASE)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    m = re.match(r"^\s*[*_]{1,2}([^*_\n]{4,60})[*_]{1,2}", content.lstrip())
    if m:
        t = _clean(m.group(1))
        if t:
            return t

    candidate = non_blank[0].strip()
    stripped  = _clean(candidate).rstrip(".")
    if stripped and len(stripped) <= max_len and not re.search(r"[.!?;]", stripped):
        return stripped

    return _clean(non_blank[0])


# ---------------------------------------------------------------------------
# ChatMixin
# ---------------------------------------------------------------------------

class ChatMixin:
    """
    Provides simplified_chat() and chat().

    Changes vs previous version
    ---------------------------
    * _active_callback is stored on self at the start of every chat turn so
      that _post_process_llm_response can fire artefact create/update events
      in real time, and cleared at the end to avoid leaking across turns.
    * enable_silent_artefact_explanation parameter (default True) is threaded
      through both chat paths and both _post_process_llm_response call sites.
    * enable_inline_widgets is wired through the fast path and agentic path
      identically to enable_notes / enable_skills.
    """

    # ------------------------------------------------------------------ helpers

    def _stream_final_answer(self, callback, images, branch_tip_id, temperature, **kwargs):
        """
        Calls the LLM to generate the next response.
        branch_tip_id controls which message is the tip of the exported branch.
        """
        caller_stream = kwargs.pop("stream", None)
        kwargs.pop("callback", None)
        kwargs.pop("streaming_callback", None)

        do_stream = (callback is not None) and (caller_stream is not False)
        collected = []

        def _streaming_relay(chunk, msg_type=None, meta=None):
            if isinstance(chunk, str):
                collected.append(chunk)
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
        return result if isinstance(result, str) else (result or "")

    # ------------------------------------------------------------------ context compression

    def _compress_context(
        self,
        callback,
        max_context_size: int,
        answer_reserve_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        import hashlib

        budget        = int(max_context_size * (1.0 - answer_reserve_ratio))
        status        = self.get_context_status()
        tokens_before = status.get("current_tokens", 0)

        if tokens_before <= budget:
            return {
                "needed": False, "artefact_pressure": False,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "budget": budget, "cache_hit": False, "summary_generated": False,
            }

        zones           = status.get("zones", {})
        history_tokens  = zones.get("message_history", {}).get("tokens", 0)
        artefact_tokens = (
            zones.get("system_context", {})
                 .get("breakdown", {})
                 .get("artefacts_zone", {})
                 .get("tokens", 0)
        )
        artefact_pressure = (
            artefact_tokens > 0
            and artefact_tokens > history_tokens
            and tokens_before > budget
        )

        _info(callback,
              f"Context at {tokens_before:,} / {budget:,} tokens "
              f"({'artefact-heavy' if artefact_pressure else 'history-heavy'}) -- compressing...")

        active_ids = sorted(a.get("id", "") for a in self.artefacts.list(active_only=True))
        key_src    = (self.active_branch_id or "") + "|" + ",".join(active_ids)
        cache_key  = hashlib.sha1(key_src.encode()).hexdigest()

        meta   = dict(self.metadata or {})
        cache  = meta.get("_compression_cache", {})
        cached = cache.get(cache_key)

        if cached:
            self.pruning_summary  = cached.get("summary", "")
            self.pruning_point_id = cached.get("pruning_point_id", "")
            self.touch()
            tokens_after = self.lollmsClient.count_tokens(
                self.export("lollms_text", self.active_branch_id, 999999)
            )
            _cb(callback,
                json.dumps({"type": "cache_hit", "tokens_before": tokens_before,
                            "tokens_after": tokens_after, "budget": budget,
                            "cache_key": cache_key,
                            "artefact_pressure": artefact_pressure}),
                MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
                {"tokens_before": tokens_before, "tokens_after": tokens_after,
                 "budget": budget, "cache_hit": True,
                 "artefact_pressure": artefact_pressure})
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_after,
                    "budget": budget, "cache_hit": True, "summary_generated": False}

        branch          = self.get_branch(self.active_branch_id)
        preserve_last_n = max(4, len(branch) // 4)

        if len(branch) <= preserve_last_n:
            _info(callback, "  History too short to prune -- reporting artefact pressure")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        to_prune      = branch[:-preserve_last_n]
        pruning_point = branch[-preserve_last_n]
        text_to_sum   = "\n\n".join(f"{m.sender}: {m.content}" for m in to_prune)

        try:
            summary = self.lollmsClient.generate_text(
                "Produce a concise but complete summary of the following conversation "
                "segment, preserving all technical decisions, code snippets, file names, "
                "variable names, and key facts. The summary will replace the original "
                "messages in the context window.\n\n"
                f"--- BEGIN SEGMENT ---\n{text_to_sum}\n--- END SEGMENT ---\n\nSUMMARY:",
                n_predict=1024, temperature=0.1,
            )
        except Exception as e:
            _warning(callback, f"  Compression failed: {e}")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        self.pruning_summary = (
            ((self.pruning_summary or "").rstrip()
             + "\n\n--- Summary ---\n" + summary.strip()).strip()
        )
        self.pruning_point_id = pruning_point.id
        self.touch()

        cache[cache_key] = {
            "summary":          self.pruning_summary,
            "pruning_point_id": self.pruning_point_id,
            "tokens_before":    tokens_before,
            "created_at":       datetime.utcnow().isoformat(),
        }
        if len(cache) > 10:
            del cache[next(iter(cache))]
        meta["_compression_cache"] = cache
        self.metadata = meta
        self.commit()

        tokens_after = self.lollmsClient.count_tokens(
            self.export("lollms_text", self.active_branch_id, 999999)
        )
        _cb(callback,
            json.dumps({"type": "summary_generated", "messages_pruned": len(to_prune),
                        "tokens_before": tokens_before, "tokens_after": tokens_after,
                        "budget": budget, "cache_key": cache_key,
                        "artefact_pressure": artefact_pressure}),
            MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
            {"tokens_before": tokens_before, "tokens_after": tokens_after,
             "budget": budget, "cache_hit": False, "summary_generated": True,
             "messages_pruned": len(to_prune), "artefact_pressure": artefact_pressure})
        _info(callback,
              f"  Compressed {len(to_prune)} messages: "
              f"{tokens_before:,} -> {tokens_after:,} tokens")

        return {"needed": True, "artefact_pressure": artefact_pressure,
                "tokens_before": tokens_before, "tokens_after": tokens_after,
                "budget": budget, "cache_hit": False, "summary_generated": True}

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
        enable_inline_widgets:   bool = True,
        enable_notes:            bool = True,
        enable_skills:           bool = False,
        enable_silent_artefact_explanation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self.scratchpad = ""
        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        # Stash callback so _post_process_llm_response can fire artefact events
        object.__setattr__(self, '_active_callback', callback)

        def is_fast(msg):
            m = msg.lower().strip()
            if len(m) < 20 and any(x in m for x in ["bonjour", "salut", "hello", "hi", "hey"]):
                return True
            return m in ["ok", "merci", "thanks", "cool", "yes", "no", "oui", "non"]

        extra_instructions = self._build_artefact_instructions()
        if enable_image_generation or enable_image_editing:
            extra_instructions += self._build_image_generation_instructions()
        if enable_inline_widgets:
            extra_instructions += self._build_inline_widget_instructions()
        if enable_notes:
            extra_instructions += self._build_note_instructions()
        if enable_skills:
            extra_instructions += self._build_skill_instructions()
        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        user_msg = None
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=user_message,
                images=images,
                **kwargs,
            )

        def _finish(text):
            ai = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content=text,
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
            )
            cleaned, affected = self._post_process_llm_response(
                text, ai,
                enable_image_generation, enable_image_editing,
                auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation,
            )
            if cleaned != text:
                ai.content = cleaned
            if affected and callback:
                _cb(callback, json.dumps([a.get("title") for a in affected]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
            object.__setattr__(self, '_active_callback', None)
            return {"user_message": user_msg, "ai_message": ai,
                    "sources": [], "artefacts": affected}

        if is_fast(user_message):
            _info(callback, "Simple response path")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, 0.1, **kwargs))

        if self.memory and user_message.lower() in self.memory.lower():
            _info(callback, "Answering from memory")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, final_answer_temperature, **kwargs))

        intent_id = _step_start(callback, "Analyzing intent...")
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

        scratchpad = ""
        sources: List[str] = []

        if intent and intent.get("needs_full_documents"):
            docs_id = _step_start(callback, "Loading context documents...")
            for zone_name, zone_content in [
                ("user_data",        self.user_data_zone),
                ("discussion_data",  self.discussion_data_zone),
                ("personality_data", self.personality_data_zone),
            ]:
                if zone_content:
                    scratchpad += f"\n--- {zone_name} ---\n{zone_content}\n"
                    sources.append(zone_name)
            _step_end(callback, f"Loaded {len(sources)} zone(s)", docs_id)

        if intent and intent.get("needs_external_search") and rag_data_stores:
            rag_id = _step_start(callback, "Searching external knowledge...")
            for name, fn in rag_data_stores.items():
                if callable(fn):
                    try:
                        res = fn(user_message)
                        if res:
                            scratchpad += f"\n--- {name} ---\n{str(res)}\n"
                            sources.append(name)
                            _info(callback, f"  Retrieved results from `{name}`")
                    except Exception as e:
                        _warning(callback, f"  `{name}` search error: {e}")
            if sources:
                _cb(callback, sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            _step_end(callback, "External search complete", rag_id, {"sources": sources})

        if scratchpad:
            self.scratchpad = scratchpad.strip()

        answer_id  = _step_start(callback, "Generating answer...")
        final_text = self._stream_final_answer(
            callback, images, branch_tip_id, final_answer_temperature, **kwargs)
        _step_end(callback, "Answer generation complete", answer_id)

        if remove_thinking_blocks:
            final_text = self.lollmsClient.remove_thinking_blocks(final_text)

        ai = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content=final_text,
            parent_id=user_msg.id if user_msg else None,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"sources": sources} if sources else {},
        )
        cleaned, affected = self._post_process_llm_response(
            final_text, ai,
            enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets,
            enable_notes=enable_notes,
            enable_skills=enable_skills,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation,
        )
        if cleaned != final_text:
            ai.content = cleaned
        if affected and callback:
            _cb(callback, json.dumps([a.get("title") for a in affected]),
                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        return {"user_message": user_msg, "ai_message": ai,
                "sources": sources, "artefacts": affected}

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        swarm=None,
        swarm_config=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = True,
        enable_image_editing:    bool = True,
        auto_activate_artefacts: bool = True,
        enable_show_tools:            bool = True,
        enable_extract_artefact:      bool = True,
        enable_final_answer:          bool = True,
        enable_request_clarification: bool = True,
        enable_repl_tools:            bool = True,
        enable_inline_widgets:        bool = True,
        enable_notes:                 bool = True,
        enable_skills:                bool = False,
        enable_silent_artefact_explanation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat with opt-in agentic tool use.

        Fast path
        ---------
        If no tools are registered, the method skips all agentic scaffolding
        and goes directly to a single LLM call.  The model may still emit
        <artefact>, <note>, <skill>, and <lollms_inline> XML tags inline.

        Callback lifecycle
        ------------------
        self._active_callback is set to the streaming callback at the start of
        every turn and cleared at the end.  _post_process_llm_response reads it
        to fire artefact create/update events in real time as each XML tag is
        processed, so the UI never has to wait for the full response.

        Scratchpad placement
        --------------------
        self.scratchpad is written to as tool results accumulate.  export()
        injects it as a role:system message immediately after the last user
        message, so the model always sees full tool-output context in the most
        relevant position.

        Anti-duplication
        ----------------
        _completed_tool_calls and _created_artefact_titles track every call
        and every produced title this turn.  An AGENT STATE block is prepended
        to the scratchpad (temporarily) at the top of each round.

        Runaway guard
        -------------
        _identical_call_counts[signature] tracks how many times each exact
        call has been made.  On the second repeat the call is blocked with a
        synthetic result; on the third the loop breaks hard.  Different params
        on the same tool are always allowed (multi-query investigation).

        Inline widgets
        --------------
        When enable_inline_widgets=True (default) the model may emit
        <lollms_inline type="html|react|svg" title="…">…</lollms_inline> tags.
        These are buffered during streaming and extracted into
        ai_message.metadata["inline_widgets"].

        Notes
        -----
        When enable_notes=True (default) the model may emit
        <note title="…">…</note> tags.  Each is saved as an ArtefactType.NOTE
        artefact, active immediately, and stripped from the visible message text.

        Skills
        ------
        When enable_skills=True (default False) the model may emit
        <skill title="…" description="…" category="…">…</skill> tags.
        Each is saved as an ArtefactType.SKILL artefact.

        Silent artefact guard
        ---------------------
        When enable_silent_artefact_explanation=True (default), if the LLM
        response consists entirely of XML tags with no human-readable text,
        a concise auto-generated summary of what was created/updated is shown
        so the chat never presents the user with an empty bubble.

        Swarm mode
        ----------
        When swarm= is a non-empty list of Agent objects, the discussion
        delegates the entire turn to SwarmOrchestrator, which runs a
        multi-agent deliberation session.  swarm_config= is an optional
        SwarmConfig instance; if omitted, SwarmConfig() defaults are used.
        The return dict gains two extra keys:
          ``agent_messages``  — list of LollmsMessage objects, one per agent per round
          ``hlf_log``         — list of HLF message dicts (inter-agent protocol log)
          ``swarm_meta``      — run statistics dict
        """
        self.scratchpad = ""

        # ── Normalise personality ────────────────────────────────────────────
        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        # Stash callback for real-time artefact events in _post_process_llm_response
        object.__setattr__(self, '_active_callback', callback)

        # ====================================================================
        #  SWARM DISPATCH — delegate to SwarmOrchestrator when agents present
        # ====================================================================
        if swarm:
            from lollms_client.lollms_swarm import SwarmOrchestrator, SwarmConfig as _SC

            _swarm_config = swarm_config if swarm_config is not None else _SC()

            # Add user message first so agents can read it
            if add_user_message:
                user_msg = self.add_message(
                    sender=kwargs.get("user_name", "user"),
                    sender_type="user",
                    content=user_message,
                    images=images,
                    **kwargs,
                )
            else:
                user_msg = None

            orchestrator = SwarmOrchestrator(
                discussion  = self,
                agents      = swarm,
                config      = _swarm_config,
                callback    = callback,
                user_msg_id = user_msg.id if user_msg else self.active_branch_id,
            )
            result = orchestrator.run(user_message, **kwargs)
            result["user_message"] = user_msg
            object.__setattr__(self, '_active_callback', None)
            self.scratchpad = ""
            if self._is_db_backed and self.autosave:
                self.commit()
            return result

        # ── Effective image flags (auto-resolve against TTI availability) ────
        _tti_available = getattr(self.lollmsClient, 'tti', None) is not None
        _eff_img_gen   = enable_image_generation and _tti_available
        _eff_img_edit  = enable_image_editing     and _tti_available

        # ── System-prompt instructions ───────────────────────────────────────
        extra_instructions = self._build_artefact_instructions()
        if _eff_img_gen or _eff_img_edit:
            extra_instructions += self._build_image_generation_instructions()
        if enable_inline_widgets:
            extra_instructions += self._build_inline_widget_instructions()
        if enable_notes:
            extra_instructions += self._build_note_instructions()
        if enable_skills:
            extra_instructions += self._build_skill_instructions()
        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        # ── Generation parameters ────────────────────────────────────────────
        decision_temperature       = kwargs.get("decision_temperature",       0.3)
        final_answer_temperature   = kwargs.get("final_answer_temperature",   0.7)
        rag_top_k                  = kwargs.get("rag_top_k",                  5)
        rag_min_similarity_percent = kwargs.get("rag_min_similarity_percent", 0.5)
        preflight_rag_enabled      = kwargs.get("preflight_rag",              True)

        # ── RLM detection ────────────────────────────────────────────────────
        rlm_enabled          = False
        rlm_context_var_name = "USER_INPUT_CONTEXT"
        actual_user_content  = user_message

        if tools:
            rlm_enabled = (
                any(t.get("name") == "python_exec" for t in tools.values() if isinstance(t, dict))
                and
                any(t.get("name") == "llm_query"   for t in tools.values() if isinstance(t, dict))
            )

        if rlm_enabled and len(user_message) > 10000:
            actual_user_content = "\n".join([
                "<RLM_STUB>",
                f"Large input ({len(user_message):,} chars) stored in `{rlm_context_var_name}`.",
                "PREVIEW:", user_message[:500], "...",
                "Use python_exec() to access the full content.",
                "</RLM_STUB>",
            ])

        # ── Add user message ─────────────────────────────────────────────────
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=actual_user_content,
                images=images,
                **kwargs,
            )
            if rlm_enabled and len(user_message) > 10000:
                user_msg.metadata["rlm_full_content"] = user_message
                user_msg.metadata["rlm_var_name"]     = rlm_context_var_name
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images   = user_msg.get_active_images()

        # ── Source inference helper ──────────────────────────────────────────
        def _infer_sources_from_json(data: Any, tool_name: str) -> List[Dict]:
            found_sources = []

            def looks_like_source(d: dict) -> bool:
                title_keys = {'title', 'name', 'label', 'header', 'id', 'filename'}
                return any(k in d for k in title_keys)

            def scan(obj: Any):
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and looks_like_source(item):
                            found_sources.append(item)
                        else:
                            scan(item)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in ['results', 'sources', 'data', 'items', 'content'] \
                                and isinstance(v, list):
                            scan(v)
                        else:
                            scan(v)

            scan(data)
            if not found_sources and isinstance(data, dict) and looks_like_source(data):
                found_sources.append(data)

            normalized = []
            for item in found_sources:
                title = (item.get('title') or item.get('name') or item.get('label') or
                         item.get('id') or item.get('filename') or f"{tool_name} Result")
                content = (item.get('content') or item.get('summary') or item.get('snippet') or
                           item.get('text') or item.get('description') or "")
                link = (item.get('url') or item.get('link') or item.get('href') or
                        item.get('source') or item.get('pdf_url') or "")
                raw_score = item.get('score') or item.get('relevance', 100 if content else 0)
                try:
                    score = float(raw_score) * 100 if 0 < float(raw_score) <= 1 \
                        else float(raw_score)
                except (TypeError, ValueError):
                    score = 0.0
                metadata = {k: v for k, v in item.items()
                            if k not in ['title', 'content', 'url', 'link', 'score']}
                normalized.append({
                    "title": str(title), "content": str(content), "source": str(link),
                    "relevance_score": score, "metadata": metadata, "tool": tool_name
                })
            return normalized

        # ── Document zone extraction helper ──────────────────────────────────
        def _extract_docs(zone_content):
            if not zone_content:
                return []
            return [
                {
                    "name":        m[0].strip(),
                    "content":     m[1].strip(),
                    "size":        len(m[1].strip()),
                    "token_count": self.lollmsClient.count_tokens(m[1].strip()),
                }
                for m in re.findall(
                    r"--- Document: (.+?) ---\n(.*?)\n--- End Document: \1 ---",
                    zone_content, re.DOTALL,
                )
            ]

        all_documents: List[Dict] = []
        for zone_content, zone_label in [
            (self.discussion_data_zone, "discussion"),
            (self.user_data_zone,       "user"),
            (None if personality.has_data else self.personality_data_zone, "personality"),
        ]:
            if zone_content:
                for d in _extract_docs(zone_content):
                    d["zone"] = zone_label
                    all_documents.append(d)

        # ====================================================================
        #  Tool registry
        # ====================================================================
        tool_registry:     Dict[str, Any] = {}
        tool_descriptions: List[str]      = []
        rag_registry:      Dict[str, Any] = {}
        rag_tool_specs:    Dict[str, Any] = {}

        composable_answer  = {"sections": [], "complete": False, "last_updated": None}
        scratchpad_state   = {"notes": {}, "history": [], "assumptions": {}, "corrections": []}
        collected_sources: List[Dict] = []
        queries_performed: List[Dict] = []
        self_corrections:  List[Dict] = []

        def get_current_answer():
            active    = [s for s in composable_answer["sections"] if s.get("status") == "active"]
            full_text = "\n\n".join(s["content"] for s in active)
            return {"success": True, "full_text": full_text, "sections": active,
                    "total_sections": len(active), "total_length": len(full_text),
                    "last_updated": composable_answer.get("last_updated")}

        # ── Validated wrapper factory ────────────────────────────────────────
        def _make_wrapper(fn: Any, params_spec: List[Dict]) -> Any:
            def _wrapped(**kw):
                try:
                    call_args: Dict[str, Any] = {}
                    for p in params_spec:
                        pn = p.get("name")
                        if pn in kw:
                            call_args[pn] = kw[pn]
                        elif not p.get("optional", False):
                            return {"error": f"Missing required parameter: {pn}",
                                    "success": False}
                        elif "default" in p:
                            call_args[pn] = p["default"]
                    result = fn(**call_args)
                    if not isinstance(result, dict):
                        result = {"output": result}
                    if "success" not in result:
                        result["success"] = "error" not in result or not result.get("error")
                    return result
                except Exception as exc:
                    return {"error": str(exc), "success": False}
            return _wrapped

        def _sig(params: List[Dict]) -> str:
            parts = []
            for p in params:
                pn, pt = p.get("name", "arg"), p.get("type", "any")
                opt, dv = p.get("optional", False), p.get("default")
                if opt and dv is not None:
                    parts.append(f"{pn}: {pt} = {dv}")
                elif opt:
                    parts.append(f"{pn}: {pt} (optional)")
                else:
                    parts.append(f"{pn}: {pt}")
            return ", ".join(parts)

        def _register(name, fn, params, description, output=None):
            tool_registry[name] = _make_wrapper(fn, params)
            tool_descriptions.append(f"- {name}({_sig(params)}): {description}")
            if any(o.get("name") == "sources" for o in (output or [])):
                rag_registry[name]   = tool_registry[name]
                rag_tool_specs[name] = {
                    "default_top_k":   rag_top_k,
                    "default_min_sim": rag_min_similarity_percent,
                }

        # ── Layer 1: caller-supplied tools ───────────────────────────────────
        for tool_name, tool_spec in (tools or {}).items():
            if not isinstance(tool_spec, dict):
                continue
            fn = tool_spec.get("callable")
            if not callable(fn):
                continue
            _register(
                name        = tool_spec.get("name", tool_name),
                fn          = fn,
                params      = tool_spec.get("parameters", []),
                description = tool_spec.get("description", f"Execute {tool_name}"),
                output      = tool_spec.get("output", []),
            )

        # ── Layer 2: personality.tool_specs() ────────────────────────────────
        _pt_specs = {}
        try:
            _pt_specs = personality.tool_specs(
                client_binding=getattr(self.lollmsClient, "tools", None)
            )
        except Exception as _pte:
            _warning(callback, f"  Personality tool discovery failed: {_pte}")
            trace_exception(_pte)

        if _pt_specs:
            _pt_step_id = _step_start(callback,
                                      f"Loading {len(_pt_specs)} personality tool(s)...")
            for pt_name, pt_spec in _pt_specs.items():
                fn = pt_spec.get("callable")
                if not callable(fn):
                    continue
                _register(
                    name        = pt_name,
                    fn          = fn,
                    params      = pt_spec.get("parameters", []),
                    description = pt_spec.get("description", f"Execute {pt_name}"),
                    output      = pt_spec.get("output", []),
                )
            _step_end(callback, f"{len(_pt_specs)} personality tool(s) ready", _pt_step_id,
                      {"tool_count": len(_pt_specs)})

        # ── Layer 3: personality RAG tool ────────────────────────────────────
        if personality.has_data:
            def _personality_rag(query: str) -> Dict[str, Any]:
                result = personality.query_data(query)
                sources_filtered = sorted(
                    (s for s in result.get("sources", [])),
                    key=lambda x: x.get("score", 0), reverse=True
                )
                for s in sources_filtered:
                    score = float(s.get("score", 1.0))
                    if score > 1.0:
                        score = 1.0 if score > 100.0 else score / 100.0
                    s["score"] = score
                sources_filtered = [
                    s for s in sources_filtered[:rag_top_k]
                    if s.get("score", 0) >= rag_min_similarity_percent
                ]
                result["sources"] = sources_filtered
                result["count"]   = len(sources_filtered)
                return result

            _register(
                name        = "search_personality_knowledge",
                fn          = _personality_rag,
                params      = [{"name": "query", "type": "str",
                                "description": "Search query"}],
                description = "Search the personality's knowledge base",
                output      = [{"name": "sources", "type": "list"}],
            )

        # ── Personality system prompt + veracity ─────────────────────────────
        if personality.system_prompt:
            veracity = (
                "\n=== VERACITY & ATTRIBUTION REQUIREMENTS ===\n"
                "Cite retrieved sources as [1],[2]... "
                "Use 'From my understanding...' for general knowledge.\n"
                "Never fabricate facts. Say 'I don't know' when uncertain.\n"
                "=== END ===\n"
            )
            object.__setattr__(
                self, "_system_prompt",
                personality.system_prompt + veracity + extra_instructions,
            )

        # ── Pre-flight RAG ───────────────────────────────────────────────────
        if preflight_rag_enabled and personality.has_data:
            preflight_id = _step_start(callback, "Pre-flight knowledge retrieval...")
            ctx = self.export("markdown", suppress_system_prompt=True)
            try:
                query_json = self.lollmsClient.generate_structured_content(
                    prompt=ctx[-2000:] + "\nGenerate a concise search query (JSON).",
                    schema={"query": "Your concise search query string"},
                    system_prompt="Output only JSON.",
                    temperature=0.1,
                )
                if query_json and "query" in query_json:
                    rag_result = personality.query_data(query_json["query"])
                    if rag_result.get("success"):
                        fmt = ""
                        for idx, chunk in enumerate(rag_result.get("sources", [])):
                            src  = chunk.get("source", "")
                            meta = chunk.get("metadata", {})
                            title = (
                                chunk.get("title")
                                or meta.get("title")
                                or meta.get("filename")
                                or meta.get("name")
                                or (src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if src else "")
                                or f"Source {idx + 1}"
                            )
                            fmt += (
                                f"[Source {idx+1}] ({src}, "
                                f"{chunk.get('score', 0):.2f})\n"
                                f"{chunk.get('content', '')}\n\n"
                            )
                            collected_sources.append({
                                "title":           title,
                                "content":         chunk.get("content", ""),
                                "source":          src,
                                "query":           query_json["query"],
                                "relevance_score": chunk.get("score", 0),
                                "index":           idx + 1,
                                "phase":           "preflight",
                                "metadata":        meta,
                            })
                        if fmt:
                            self.scratchpad = (
                                fmt.strip() + "\n\nIMPORTANT: Cite sources as [1],[2],..."
                            )
                        if collected_sources:
                            _cb(callback, collected_sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            except Exception as e:
                trace_exception(e)
            _step_end(callback, "Pre-flight retrieval complete", preflight_id,
                      {"source_count": len(collected_sources)})

        # ====================================================================
        #  FAST PATH — no external tools registered
        # ====================================================================
        _has_external_tools = bool(tool_registry)

        if not _has_external_tools:
            fast_buf: List[str] = []

            def _fast_relay(chunk, msg_type=None, meta=None):
                if isinstance(chunk, str):
                    fast_buf.append(chunk)
                return _cb(callback, chunk, msg_type or MSG_TYPE.MSG_TYPE_CHUNK, meta)

            raw_text = self._stream_final_answer(
                _fast_relay, images,
                branch_tip_id or self.active_branch_id,
                final_answer_temperature, **kwargs,
            )
            if not raw_text and fast_buf:
                raw_text = "".join(fast_buf)

            if remove_thinking_blocks:
                raw_text = self.lollmsClient.remove_thinking_blocks(raw_text)

            ai_message = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content=raw_text,
                parent_id=user_msg.id,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
                metadata={"mode": "direct"},
            )
            cleaned, affected = self._post_process_llm_response(
                raw_text, ai_message, _eff_img_gen, _eff_img_edit, auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation,
            )
            if cleaned != raw_text:
                ai_message.content = cleaned
            if affected and callback:
                _cb(callback, json.dumps([a.get("title") for a in affected]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
            if self._is_db_backed and self.autosave:
                self.commit()
            self.scratchpad = ""
            object.__setattr__(self, '_active_callback', None)
            return {
                "user_message":     user_msg,
                "ai_message":       ai_message,
                "sources":          collected_sources,
                "scratchpad":       None,
                "self_corrections": None,
                "artefacts":        affected,
            }

        # ====================================================================
        #  Built-in tools (only when external tools exist)
        # ====================================================================

        # show_tools ----------------------------------------------------------
        if enable_show_tools:
            def _show_tools_impl():
                catalogue: List[Dict[str, Any]] = []
                for _tname, _tspec in (tools or {}).items():
                    if not isinstance(_tspec, dict):
                        continue
                    catalogue.append({
                        "name":        _tspec.get("name", _tname),
                        "description": _tspec.get("description", ""),
                        "parameters":  _tspec.get("parameters", []),
                        "output":      _tspec.get("output", []),
                        "source":      "user",
                    })
                for pt_name, pt_spec in _pt_specs.items():
                    catalogue.append({
                        "name":        pt_name,
                        "description": pt_spec.get("description", ""),
                        "parameters":  pt_spec.get("parameters", []),
                        "output":      pt_spec.get("output", []),
                        "source":      "personality",
                        "binding":     pt_spec.get("_binding", ""),
                    })
                builtins: List[Dict] = [
                    {"name": "show_tools",
                     "description": "Display the full list of available tools",
                     "parameters": [], "output": [{"name": "tools", "type": "list"}],
                     "source": "builtin"},
                ]
                if enable_extract_artefact:
                    builtins.append({
                        "name": "extract_artefact_text",
                        "description": "Extract a range from an artefact by line-hint anchors",
                        "parameters": [
                            {"name": "source_title",    "type": "str"},
                            {"name": "new_title",       "type": "str"},
                            {"name": "start_line_hint", "type": "str"},
                            {"name": "end_line_hint",   "type": "str"},
                            {"name": "occurrence",      "type": "int",
                             "optional": True, "default": 1},
                            {"name": "artefact_type",   "type": "str",
                             "optional": True, "default": "document"},
                            {"name": "language",        "type": "str",
                             "optional": True, "default": ""},
                        ],
                        "output": [{"name": "lines_extracted", "type": "int"}],
                        "source": "builtin",
                    })
                if enable_final_answer:
                    builtins.append({
                        "name": "final_answer",
                        "description": "Signal that the answer is ready",
                        "parameters": [], "output": [], "source": "builtin",
                    })
                if enable_request_clarification:
                    builtins.append({
                        "name": "request_clarification",
                        "description": "Ask the user for clarification",
                        "parameters": [{"name": "question", "type": "str"}],
                        "output": [], "source": "builtin",
                    })
                if _eff_img_gen:
                    builtins.append({
                        "name": "generate_image (XML tag)",
                        "description": "Generate an image via <generate_image> XML tag",
                        "parameters": [
                            {"name": "width",  "type": "int", "optional": True, "default": 1024},
                            {"name": "height", "type": "int", "optional": True, "default": 1024},
                            {"name": "prompt", "type": "str"},
                        ],
                        "output": [{"name": "image", "type": "base64"}],
                        "source": "builtin",
                    })
                if _eff_img_edit:
                    builtins.append({
                        "name": "edit_image (XML tag)",
                        "description": "Edit an image via <edit_image> XML tag",
                        "parameters": [
                            {"name": "name",   "type": "str", "optional": True},
                            {"name": "prompt", "type": "str"},
                        ],
                        "output": [{"name": "image", "type": "base64"}],
                        "source": "builtin",
                    })
                for b in builtins:
                    if b not in catalogue:
                        catalogue.append(b)
                _cb(callback, json.dumps(catalogue, indent=2),
                    MSG_TYPE.MSG_TYPE_TOOLS_LIST, {"tools": catalogue})
                return {"success": True, "tool_count": len(catalogue), "tools": catalogue}

            tool_registry["show_tools"] = _show_tools_impl
            tool_descriptions.append("- show_tools(): Display the full list of available tools")

        # extract_artefact_text -----------------------------------------------
        if enable_extract_artefact:
            def _extract_artefact_text_impl(
                source_title: str, new_title: str,
                start_line_hint: str, end_line_hint: str,
                occurrence: int = 1,
                artefact_type: str = "document",
                language: str = "",
            ) -> Dict[str, Any]:
                source = self.artefacts.get(source_title)
                if source is None:
                    return {"success": False,
                            "error": f"Artefact '{source_title}' not found."}
                all_lines = source.get("content", "").splitlines()
                total = len(all_lines)
                if total == 0:
                    return {"success": False,
                            "error": f"Artefact '{source_title}' is empty."}
                sh = start_line_hint.strip().lower()
                eh = end_line_hint.strip().lower()
                if not sh:
                    return {"success": False, "error": "start_line_hint must not be empty."}
                if not eh:
                    return {"success": False, "error": "end_line_hint must not be empty."}
                start_idx, hit = None, 0
                for i, ln in enumerate(all_lines):
                    if ln.strip().lower().startswith(sh):
                        hit += 1
                        if hit == occurrence:
                            start_idx = i
                            break
                if start_idx is None:
                    return {"success": False,
                            "error": (f"start_line_hint {start_line_hint!r} not found "
                                      f"(occurrence {occurrence} of {hit} found)."),
                            "total_lines": total}
                end_idx = None
                for i in range(start_idx, total):
                    if all_lines[i].strip().lower().startswith(eh):
                        end_idx = i
                        break
                if end_idx is None:
                    return {"success": False,
                            "error": (f"end_line_hint {end_line_hint!r} not found "
                                      f"after line {start_idx + 1}."),
                            "start_line_no": start_idx + 1, "total_lines": total}
                from ._artefacts import ArtefactType as _AT
                resolved_type = (artefact_type if artefact_type in _AT.ALL
                                 else source.get("type", _AT.DOCUMENT))
                extracted = "\n".join(all_lines[start_idx:end_idx + 1])
                new_art   = self.artefacts.add(
                    title=new_title, artefact_type=resolved_type,
                    content=extracted,
                    language=language or source.get("language") or None,
                    active=True,
                )
                return {
                    "success": True, "source_title": source_title, "new_title": new_title,
                    "start_line_no": start_idx + 1, "end_line_no": end_idx + 1,
                    "total_lines": total, "lines_extracted": end_idx - start_idx + 1,
                    "artefact_id": new_art.get("id"),
                }

            tool_registry["extract_artefact_text"] = _extract_artefact_text_impl
            tool_descriptions.append(
                "- extract_artefact_text(source_title: str, new_title: str, "
                "start_line_hint: str, end_line_hint: str, occurrence: int = 1, "
                "artefact_type: str = 'document', language: str = ''): "
                "Extract a range from an artefact by line-prefix anchors"
            )

        # REPL text tools -----------------------------------------------------
        if enable_repl_tools:
            try:
                from ._repl_tools import TextBuffer, register_repl_tools as _reg_repl
                _reg_repl(tool_registry, tool_descriptions, TextBuffer(), self.artefacts)
            except ImportError as _e:
                _warning(callback, f"REPL text tools unavailable: {_e}")

        # final_answer --------------------------------------------------------
        if enable_final_answer:
            tool_registry["final_answer"] = lambda: {
                "status":  "final",
                "answer":  get_current_answer()["full_text"]
                           if composable_answer["sections"] else None,
                "success": True,
            }
            tool_descriptions.append("- final_answer(): Signal that the answer is ready")

        # request_clarification -----------------------------------------------
        if enable_request_clarification:
            tool_registry["request_clarification"] = lambda question: {
                "status": "clarification", "question": question, "success": True,
            }
            tool_descriptions.append(
                "- request_clarification(question: str): Ask user for clarification"
            )

        # ── Inject tool catalogue into system prompt ─────────────────────────
        if tool_descriptions:
            _tc = [
                "\n\n## Available Tools & Action Protocol",
                "You are an agent with access to external tools.",
                "CRITICAL: To perform an action, you MUST emit a <tool_call> tag. "
                "Do NOT use markdown code blocks for tool calls.",
                "IMPORTANT: Call each tool ONCE. Check the AGENT STATE in the scratchpad "
                "before calling any tool — if a call is already listed there, do NOT repeat it.",
                "",
                "### Call Syntax",
                '  <tool_call>{"name": "tool_name", "parameters": {"key": "value"}}</tool_call>',
                "",
                "The system will stop generation, execute the tool, and provide a "
                "<tool_result> in the next turn.",
                "Call one tool per turn. After receiving the result, either call the "
                "next tool OR write your final answer — never repeat a call you already made.",
                "",
                "### Tool Catalogue",
            ] + tool_descriptions
            _cur_sp   = self._system_prompt or ""
            _clean_sp = (_cur_sp.split("\n\n## Available Tools")[0]
                                .split("### FUNCTION CALLING INSTRUCTIONS")[0])
            object.__setattr__(self, "_system_prompt", _clean_sp + "\n".join(_tc))

        # ── Context compression ──────────────────────────────────────────────
        if self.max_context_size is not None:
            _cr = self._compress_context(callback, self.max_context_size)
            if _cr["needed"] and _cr["artefact_pressure"]:
                def _deactivate_artefacts_impl(titles: List[str]) -> Dict[str, Any]:
                    deactivated, not_found = [], []
                    for t in titles:
                        if self.artefacts.get(t) is None:
                            not_found.append(t)
                        else:
                            self.artefacts.deactivate(t)
                            deactivated.append(t)
                    if deactivated:
                        if callback:
                            _cb(callback, json.dumps(deactivated),
                                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                                {"artefacts": deactivated,
                                 "action": "deactivated_for_compression"})
                        tokens_after = self._compress_context(
                            callback, self.max_context_size
                        ).get("tokens_after", _cr["tokens_before"])
                    else:
                        tokens_after = _cr["tokens_before"]
                    return {
                        "success": bool(deactivated), "deactivated": deactivated,
                        "not_found": not_found,
                        "tokens_freed_estimate": _cr["tokens_before"] - tokens_after,
                    }

                tool_registry["deactivate_artefacts"] = _deactivate_artefacts_impl
                tool_descriptions.insert(0,
                    "- deactivate_artefacts(titles: list[str]): "
                    "CONTEXT PRESSURE -- deactivate unneeded artefacts first")
                _tc2 = [
                    "\n\n## Available Tools",
                    '  <tool_call>{"name": "tool_name", "parameters": {"key": "value"}}</tool_call>',
                    "", "### Tool list",
                ] + tool_descriptions
                object.__setattr__(
                    self, "_system_prompt",
                    (self._system_prompt or "").split("\n\n## Available Tools")[0]
                    + "\n".join(_tc2)
                )

        # ====================================================================
        #  Streaming / agentic loop
        # ====================================================================
        start_time            = datetime.now()
        is_agentic_turn       = False
        tool_calls_this_turn: List[Dict] = []
        all_events:           List[Dict] = []
        _TC_OPEN              = "<tool_call>"
        _TC_CLOSE             = "</tool_call>"
        _accumulated_full     = ""
        _clean_text_so_far    = ""
        _round                = 0
        _temp_msg_ids:        List[str] = []
        _current_branch_tip   = branch_tip_id or self.active_branch_id

        # Anti-duplication tracking
        _completed_tool_calls:    List[str] = []
        _created_artefact_titles: List[str] = []

        # Runaway guard
        _MAX_IDENTICAL_REPEATS              = 2
        _identical_call_counts: Dict[str, int] = {}

        # Pre-create the assistant message so the UI shows a bubble immediately
        ai_message = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"mode": "agentic", "events": []}
        )
        if self._is_db_backed:
            self.commit()
        _cb(callback, ai_message.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE,
            {"message_id": ai_message.id})

        # Tags that must be buffered / hidden from the live chat UI stream
        _TAG_STARTS = [
            _TC_OPEN, "<think>", "<think ", "<artefact", "<generate_image",
            "<edit_image", "<revert_artefact", "<generate_slides",
            "<street_view", "<schedule_task", "<note", "<skill", "<lollms_event",
            "<lollms_inline",
        ]

        # ── Secondary-stream tag type mapping ────────────────────────────────
        # Maps tag prefix → (open_meta_type, chunk_MSG_TYPE, done_MSG_TYPE, close_tag)
        # open_meta_type is the string put in the MSG_TYPE_CHUNK announcement meta.
        _SECONDARY_TAG_MAP = {
            "<artefact":    ("artefact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK,
                             MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,  "</artefact>"),
            "<note":        ("note_start",           MSG_TYPE.MSG_TYPE_NOTE_CHUNK,
                             MSG_TYPE.MSG_TYPE_NOTE_DONE,      "</note>"),
            "<skill":       ("skill_start",          MSG_TYPE.MSG_TYPE_SKILL_CHUNK,
                             MSG_TYPE.MSG_TYPE_SKILL_DONE,     "</skill>"),
            "<lollms_inline":("inline_widget_start", MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,
                              MSG_TYPE.MSG_TYPE_WIDGET_DONE,   "</lollms_inline>"),
        }

        while _round < max_reasoning_steps:
            _round += 1
            _stream_buf: List[str] = []

            # ── Streaming flow-control state — reset every round ─────────────
            _in_logic_tag         = False
            _is_hidden_tag        = False
            _tool_trigger         = False
            _bracket_buf: List[str] = []
            _is_buffering_bracket = False

            # Secondary-stream state: which XML tag are we currently inside?
            _sec_tag_prefix:  str        = ""   # e.g. "<artefact"
            _sec_chunk_type:  Any        = None  # MSG_TYPE for chunk events
            _sec_done_type:   Any        = None  # MSG_TYPE for done event
            _sec_close_tag:   str        = ""    # e.g. "</artefact>"
            _sec_open_meta:   Dict       = {}    # attrs extracted from opening tag
            _sec_content_buf: List[str]  = []    # accumulates raw content inside tag
            # Buffer for partial close-tag detection across chunk boundaries
            _sec_close_scan:  str        = ""

            def _parse_tag_attrs(attr_str: str) -> Dict[str, str]:
                return {m.group(1): m.group(2)
                        for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

            def _build_open_meta_and_announce(b_str: str) -> None:
                """
                Called once when an opening tag's full `<tag attrs>` is seen.
                Fires the MSG_TYPE_CHUNK announcement and populates _sec_open_meta.
                """
                nonlocal _sec_tag_prefix, _sec_chunk_type, _sec_done_type
                nonlocal _sec_close_tag, _sec_open_meta, _sec_content_buf, _sec_close_scan

                for prefix, (ann_type, chunk_mt, done_mt, close_tag) \
                        in _SECONDARY_TAG_MAP.items():
                    if not b_str.startswith(prefix):
                        continue

                    # Only handle tags that are enabled
                    if prefix == "<note"         and not enable_notes:            break
                    if prefix == "<skill"        and not enable_skills:           break
                    if prefix == "<lollms_inline" and not enable_inline_widgets:  break

                    attrs = _parse_tag_attrs(b_str)
                    _sec_tag_prefix  = prefix
                    _sec_chunk_type  = chunk_mt
                    _sec_done_type   = done_mt
                    _sec_close_tag   = close_tag
                    _sec_open_meta   = attrs
                    _sec_content_buf = []
                    _sec_close_scan  = ""

                    # Build the announcement meta (same shape as before — backwards compat)
                    if prefix == "<artefact":
                        title = attrs.get('name') or attrs.get('title', '')
                        if title and title not in _created_artefact_titles:
                            _created_artefact_titles.append(title)
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                            "type": "artefact_update",
                            "content": {"title": title}
                        })

                    elif prefix == "<note":
                        title = attrs.get('title') or attrs.get('name', 'Note')
                        if title and title not in _created_artefact_titles:
                            _created_artefact_titles.append(title)
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                            "type": "note_start",
                            "content": {"title": title}
                        })

                    elif prefix == "<skill":
                        title    = attrs.get('title') or attrs.get('name', 'Skill')
                        category = attrs.get('category', '')
                        if title and title not in _created_artefact_titles:
                            _created_artefact_titles.append(title)
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                            "type": "skill_start",
                            "content": {"title": title, "category": category}
                        })

                    elif prefix == "<lollms_inline":
                        title       = attrs.get('title', 'Interactive Widget')
                        widget_type = attrs.get('type', 'html')
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                            "type":    "inline_widget_start",
                            "content": {"title": title, "widget_type": widget_type}
                        })
                    break

            def _fire_sec_chunk(raw_chunk: str) -> None:
                """Fire a secondary-stream chunk event for the current tag type."""
                if not _sec_chunk_type or not raw_chunk:
                    return
                attrs = _sec_open_meta
                if _sec_tag_prefix == "<artefact":
                    _cb(callback, raw_chunk, _sec_chunk_type, {
                        "title":    attrs.get('name') or attrs.get('title', ''),
                        "chunk":    raw_chunk,
                        "art_type": attrs.get('type', 'document'),
                        "language": attrs.get('language'),
                    })
                elif _sec_tag_prefix == "<note":
                    _cb(callback, raw_chunk, _sec_chunk_type, {
                        "title": attrs.get('title') or attrs.get('name', 'Note'),
                        "chunk": raw_chunk,
                    })
                elif _sec_tag_prefix == "<skill":
                    _cb(callback, raw_chunk, _sec_chunk_type, {
                        "title":       attrs.get('title') or attrs.get('name', 'Skill'),
                        "chunk":       raw_chunk,
                        "category":    attrs.get('category', ''),
                        "description": attrs.get('description', ''),
                    })
                elif _sec_tag_prefix == "<lollms_inline":
                    _cb(callback, raw_chunk, _sec_chunk_type, {
                        "title":       attrs.get('title', 'Interactive Widget'),
                        "chunk":       raw_chunk,
                        "widget_type": attrs.get('type', 'html'),
                    })

            def _fire_sec_done() -> None:
                """Fire the secondary-stream done event and reset secondary state."""
                nonlocal _sec_tag_prefix, _sec_chunk_type, _sec_done_type
                nonlocal _sec_close_tag, _sec_open_meta, _sec_content_buf, _sec_close_scan

                if not _sec_done_type:
                    return

                full_content = "".join(_sec_content_buf)
                attrs        = _sec_open_meta

                if _sec_tag_prefix == "<artefact":
                    is_patch = bool(re.search(r'<{6,8}\s*SEARCH', full_content, re.I))
                    _cb(callback, full_content, _sec_done_type, {
                        "title":    attrs.get('name') or attrs.get('title', ''),
                        "content":  full_content,
                        "art_type": attrs.get('type', 'document'),
                        "language": attrs.get('language'),
                        "is_patch": is_patch,
                        "attrs":    {k: v for k, v in attrs.items()
                                     if k not in ('name', 'title', 'type', 'language')},
                    })
                elif _sec_tag_prefix == "<note":
                    _cb(callback, full_content, _sec_done_type, {
                        "title":   attrs.get('title') or attrs.get('name', 'Note'),
                        "content": full_content,
                    })
                elif _sec_tag_prefix == "<skill":
                    _cb(callback, full_content, _sec_done_type, {
                        "title":       attrs.get('title') or attrs.get('name', 'Skill'),
                        "content":     full_content,
                        "category":    attrs.get('category', ''),
                        "description": attrs.get('description', ''),
                    })
                elif _sec_tag_prefix == "<lollms_inline":
                    _cb(callback, full_content, _sec_done_type, {
                        "title":       attrs.get('title', 'Interactive Widget'),
                        "content":     full_content,
                        "widget_type": attrs.get('type', 'html'),
                    })

                # Reset secondary state
                _sec_tag_prefix  = ""
                _sec_chunk_type  = None
                _sec_done_type   = None
                _sec_close_tag   = ""
                _sec_open_meta   = {}
                _sec_content_buf = []
                _sec_close_scan  = ""

            def _inline_relay(chunk, msg_type=None, meta=None):
                nonlocal _in_logic_tag, _is_hidden_tag, _tool_trigger
                nonlocal _is_buffering_bracket, _sec_close_scan

                # Non-chunk events pass through unchanged
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    if msg_type in [MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, MSG_TYPE.MSG_TYPE_REASONING]:
                        ai_message.thoughts = (ai_message.thoughts or "") + chunk
                    return _cb(callback, chunk, msg_type, meta)

                if not isinstance(chunk, str):
                    return True

                _stream_buf.append(chunk)
                _so_far = "".join(_stream_buf)

                # ── Inside <tool_call>: buffer silently; stop on close tag ────
                if _in_logic_tag:
                    if _TC_CLOSE in _so_far:
                        _tool_trigger = True
                        return False
                    return True

                # ── Inside a secondary-stream tag (artefact/note/skill/widget) ─
                # Forward each chunk on the secondary stream, detect closing tag.
                if _is_hidden_tag and _sec_tag_prefix:
                    # Check for tool_call nested inside (shouldn't happen but be safe)
                    if _TC_OPEN in chunk:
                        _in_logic_tag = True
                        return True

                    # Accumulate close-tag scanner across chunk boundaries
                    _sec_close_scan += chunk
                    close_tag = _sec_close_tag  # e.g. "</artefact>"
                    close_len = len(close_tag)

                    if close_tag in _sec_close_scan:
                        # Found the closing tag — split content at it
                        idx      = _sec_close_scan.index(close_tag)
                        pre_part = _sec_close_scan[:idx]
                        # Everything after the close tag goes back to normal stream
                        post_part = _sec_close_scan[idx + close_len:]

                        # Emit any remaining content before the close
                        if pre_part:
                            _sec_content_buf.append(pre_part)
                            _fire_sec_chunk(pre_part)

                        # Fire the done event
                        _fire_sec_done()

                        # Reset hidden-tag state
                        _is_hidden_tag = False
                        _sec_close_scan = ""

                        # Route any post-close content back through normal processing
                        if post_part:
                            # Directly append to visible stream (simple case: prose after tag)
                            ai_message.content += post_part
                            _cb(callback, post_part, MSG_TYPE.MSG_TYPE_CHUNK)
                    else:
                        # Close tag not yet complete — keep scanning
                        # Only forward content that cannot be the start of the close tag
                        # (to avoid emitting partial close-tag fragments)
                        safe_len = max(0, len(_sec_close_scan) - close_len + 1)
                        safe_content = _sec_close_scan[:safe_len]
                        if safe_content:
                            _sec_content_buf.append(safe_content)
                            _fire_sec_chunk(safe_content)
                            _sec_close_scan = _sec_close_scan[safe_len:]

                    return True

                # ── Inside <think> or other silently-buffered hidden tags ──────
                if _is_hidden_tag:
                    if _TC_OPEN in _so_far:
                        _in_logic_tag = True
                    return True

                # ── Normal content: scan for tag openings ────────────────────
                rem = chunk
                while rem:
                    if _is_buffering_bracket:
                        if ">" in rem:
                            idx     = rem.index(">") + 1
                            tag_bit = rem[:idx]
                            rem     = rem[idx:]
                            _bracket_buf.append(tag_bit)
                            b_str = "".join(_bracket_buf)

                            if _TC_OPEN in b_str:
                                _in_logic_tag         = True
                                _is_buffering_bracket = False
                                return True

                            if any(b_str.startswith(s) for s in _TAG_STARTS):
                                _is_hidden_tag        = True
                                _is_buffering_bracket = False

                                # Fire opening announcement + set up secondary stream
                                _build_open_meta_and_announce(b_str)

                                # Reset close-tag scanner for the new tag
                                _sec_close_scan = ""

                                # Any content remaining in this chunk after the
                                # opening tag belongs to the secondary stream, not
                                # the chat bubble.  Route it through the secondary
                                # handler and stop the while-rem loop — subsequent
                                # chunks will enter via the _is_hidden_tag branch.
                                if rem and _sec_tag_prefix:
                                    _sec_close_scan += rem
                                    close_tag = _sec_close_tag
                                    close_len = len(close_tag)
                                    if close_tag in _sec_close_scan:
                                        idx       = _sec_close_scan.index(close_tag)
                                        pre_part  = _sec_close_scan[:idx]
                                        post_part = _sec_close_scan[idx + close_len:]
                                        if pre_part:
                                            _sec_content_buf.append(pre_part)
                                            _fire_sec_chunk(pre_part)
                                        _fire_sec_done()
                                        _is_hidden_tag  = False
                                        _sec_close_scan = ""
                                        if post_part:
                                            ai_message.content += post_part
                                            _cb(callback, post_part,
                                                MSG_TYPE.MSG_TYPE_CHUNK)
                                    else:
                                        safe_len = max(0, len(_sec_close_scan) - close_len + 1)
                                        safe_content = _sec_close_scan[:safe_len]
                                        if safe_content:
                                            _sec_content_buf.append(safe_content)
                                            _fire_sec_chunk(safe_content)
                                            _sec_close_scan = _sec_close_scan[safe_len:]
                                rem = ""   # consumed — exit the while loop
                                continue

                            # Not a special tag — flush buffered bracket content
                            if not any(s.startswith(b_str) for s in _TAG_STARTS):
                                ai_message.content += b_str
                                _cb(callback, b_str, MSG_TYPE.MSG_TYPE_CHUNK)
                                _is_buffering_bracket = False
                                _bracket_buf.clear()
                        else:
                            _bracket_buf.append(rem)
                            b_str = "".join(_bracket_buf)
                            # Fail-safe: flush if obviously not a tag
                            if (any(c in b_str for c in [" ", "\n", "\r", "\t"])
                                    or len(b_str) > 100):
                                if not any(s.startswith(b_str) for s in _TAG_STARTS):
                                    ai_message.content += b_str
                                    _cb(callback, b_str, MSG_TYPE.MSG_TYPE_CHUNK)
                                    _is_buffering_bracket = False
                                    _bracket_buf.clear()
                            rem = ""
                    else:
                        if "<" in rem:
                            idx  = rem.index("<")
                            pre  = rem[:idx]
                            post = rem[idx:]
                            if pre:
                                ai_message.content += pre
                                _cb(callback, pre, MSG_TYPE.MSG_TYPE_CHUNK)
                            _is_buffering_bracket = True
                            _bracket_buf.clear()
                            rem = post
                        else:
                            ai_message.content += rem
                            _cb(callback, rem, MSG_TYPE.MSG_TYPE_CHUNK)
                            rem = ""
                return True

            # ── Build agent-state header and prepend to scratchpad ───────────
            _saved_scratchpad = self.scratchpad
            if _completed_tool_calls or _created_artefact_titles:
                _state_lines = [
                    "=== AGENT STATE (already completed this turn — DO NOT repeat) ==="
                ]
                if _completed_tool_calls:
                    _state_lines.append("Tool calls already made:")
                    _state_lines.extend(f"  ✓ {c}" for c in _completed_tool_calls)
                if _created_artefact_titles:
                    _state_lines.append("Artefacts / notes already created:")
                    _state_lines.extend(f"  ✓ {t}" for t in _created_artefact_titles)
                _state_lines.append("=== END AGENT STATE ===\n")
                self.scratchpad = "\n".join(_state_lines) + (self.scratchpad or "")

            self._stream_final_answer(
                _inline_relay, images, _current_branch_tip, final_answer_temperature, **kwargs
            )

            # Restore undecorated scratchpad
            self.scratchpad = _saved_scratchpad

            _so_far            = "".join(_stream_buf)
            _accumulated_full += _so_far
            _clean_segment     = (_so_far[:_so_far.index(_TC_OPEN)]
                                  if _tool_trigger else _so_far)
            _clean_text_so_far += _clean_segment

            if not _tool_trigger:
                break

            # ── Parse tool call ──────────────────────────────────────────────
            is_agentic_turn = True
            try:
                _open_idx    = _so_far.index(_TC_OPEN) + len(_TC_OPEN)
                _close_idx   = _so_far.index(_TC_CLOSE)
                _call_data   = json.loads(_so_far[_open_idx:_close_idx].strip())
                _tool_name   = _call_data.get("name", "")
                _tool_params = _call_data.get("parameters", {})
            except Exception as e:
                trace_exception(e)
                _warning(callback, f"Failed to parse tool call: {e}")
                break

            # ── Deduplication + runaway guard ────────────────────────────────
            _params_summary = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in _tool_params.items()
            )
            _call_signature = f"{_tool_name}({_params_summary})"
            _call_tag       = f"round {_round}: {_call_signature}"

            _identical_call_counts[_call_signature] = \
                _identical_call_counts.get(_call_signature, 0) + 1
            _sig_count = _identical_call_counts[_call_signature]

            if _sig_count > _MAX_IDENTICAL_REPEATS:
                _warning(callback,
                         f"[RUNAWAY] Identical call '{_call_signature}' seen "
                         f"{_sig_count} times — breaking loop.")
                break

            if _sig_count > 1:
                _warning(callback,
                         f"[DEDUP] Blocking duplicate tool call: {_call_signature}")
                _dup_result_str = (
                    f"DUPLICATE CALL BLOCKED: '{_call_signature}' was already executed "
                    f"this turn. Check the AGENT STATE in the scratchpad and either "
                    f"proceed to the next step or write your final answer."
                )
                _temp_call = self.add_message(
                    sender=personality.name, sender_type="assistant",
                    content=_so_far, parent_id=_current_branch_tip
                )
                _temp_res = self.add_message(
                    sender="system", sender_type="user",
                    content=(f"<tool_result name=\"{_tool_name}\">"
                             f"{_dup_result_str}</tool_result>"),
                    parent_id=_temp_call.id
                )
                _temp_msg_ids.extend([_temp_call.id, _temp_res.id])
                _current_branch_tip = _temp_res.id
                continue

            # ── Emit tool-call event ─────────────────────────────────────────
            _current_offset = len(_clean_text_so_far)
            _call_id        = str(uuid.uuid4())
            _call_evt = {
                "type": "tool_call", "content": f"Calling {_tool_name}",
                "id": _call_id, "tool": _tool_name,
                "params": _tool_params, "offset": _current_offset,
            }
            _cb(callback, _call_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_CALL, _call_evt)
            all_events.append(_call_evt)

            _marker = f"\n<lollms_event id=\"{_call_id}\" />\n"
            _clean_text_so_far    += _marker
            ai_message.content     = _clean_text_so_far
            ai_message.metadata["events"] = list(all_events)

            _step_lbl = f"Running: {_tool_name.replace('_', ' ').title()}"
            _step_id  = _step_start(callback, _step_lbl,
                                    {"tool": _tool_name, "offset": _current_offset})
            all_events.append({"type": "step_start", "content": _step_lbl,
                               "id": _step_id, "offset": _current_offset})

            # ── Execute tool ─────────────────────────────────────────────────
            if _tool_name not in tool_registry:
                _warning(callback, f"Unknown tool: {_tool_name}")
                _result_str = f"Error: tool '{_tool_name}' not found"
                _result     = {"error": _result_str}
                _err_evt = {
                    "type": "tool_output", "content": _result_str,
                    "id": str(uuid.uuid4()), "tool": _tool_name,
                    "result": _result, "offset": _current_offset,
                }
                _cb(callback, _err_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _err_evt)
                _step_end(callback, f"Unknown tool '{_tool_name}'",
                          _step_id, {"status": "failed"})
                all_events.extend([_err_evt,
                    {"type": "step_end",
                     "content": f"Unknown tool '{_tool_name}'",
                     "id": _step_id, "offset": _current_offset, "status": "failed"}])
            else:
                try:
                    _result = tool_registry[_tool_name](**_tool_params)

                    inferred_srcs = _infer_sources_from_json(_result, _tool_name)
                    res_label     = _tool_name.replace('_', ' ').title()
                    llm_block     = f"### [Source List: {res_label}]\n"
                    for s in inferred_srcs:
                        s["index"] = len(collected_sources) + 1
                        collected_sources.append(s)
                        llm_block += f"[[{s['index']}]] {s['title']}\n"
                        if s['content']:
                            llm_block += f"Content: {s['content']}\n"
                        if s['metadata']:
                            llm_block += f"Metadata: {json.dumps(s['metadata'])}\n"
                        llm_block += "---\n"
                    if not inferred_srcs:
                        llm_block += json.dumps(_result, indent=2)

                    self.scratchpad = (self.scratchpad or "") + (
                        f"\n--- Tool: {res_label} (round {_round}) ---\n"
                        f"{llm_block}\n"
                        f"--- End {res_label} ---\n"
                    )

                    _completed_tool_calls.append(_call_tag)
                    _created_title = (
                        _result.get("title") or _result.get("name") or
                        _result.get("note_title") or _result.get("artefact_title") or
                        _tool_params.get("title") or _tool_params.get("name") or
                        _tool_params.get("note_title")
                    )
                    if _created_title and str(_created_title) not in _created_artefact_titles:
                        _created_artefact_titles.append(str(_created_title))

                    _result_str = json.dumps(_result, indent=2)[:2000]
                    tool_calls_this_turn.append({
                        "name": _tool_name, "params": _tool_params, "result": _result,
                    })

                    _out_evt = {
                        "type": "tool_output", "content": _result_str,
                        "id": str(uuid.uuid4()), "tool": _tool_name,
                        "result": _result, "offset": _current_offset,
                    }
                    _cb(callback, _out_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _out_evt)
                    _step_end(callback, f"Done: {_tool_name}",
                              _step_id, {"status": "success"})
                    all_events.extend([
                        _out_evt,
                        {"type": "step_end", "content": f"Done: {_tool_name}",
                         "id": _step_id, "offset": _current_offset, "status": "success"},
                    ])

                    ai_message.metadata["events"]  = list(all_events)
                    ai_message.metadata["sources"] = list(collected_sources)
                    if self._is_db_backed:
                        self.commit()

                    _q       = _tool_params.get("query", _tool_params.get("prompt", ""))
                    raw_srcs = _result.get("sources", [])
                    if not raw_srcs:
                        cand = _result.get(
                            "results", _result.get(
                                "content",
                                _result if isinstance(_result, list) else []
                            )
                        )
                        if isinstance(cand, list):
                            raw_srcs = cand
                    if raw_srcs and isinstance(raw_srcs, list):
                        queries_performed.append({
                            "step": _round, "tool": _tool_name, "query": _q,
                            "result_count": len(raw_srcs),
                        })
                        for _doc in raw_srcs:
                            if not isinstance(_doc, dict):
                                continue
                            _doc_title   = (_doc.get("title") or _doc.get("name") or
                                            _doc.get("source") or f"{_tool_name} Result")
                            _doc_content = (_doc.get("content") or _doc.get("summary") or
                                            _doc.get("snippet") or "")
                            _doc_link    = (_doc.get("link") or _doc.get("url") or
                                            _doc.get("source") or "")
                            collected_sources.append({
                                "title":           _doc_title,
                                "content":         _doc_content,
                                "source":          _doc_link,
                                "query":           _q,
                                "relevance_score": _doc.get("score",
                                    _doc.get("relevance", 100 if _doc_content else 0)),
                                "index":           len(collected_sources) + 1,
                                "tool":            _tool_name,
                            })

                except Exception as e:
                    trace_exception(e)
                    _warning(callback, f"Tool error ({_tool_name}): {e}")
                    _step_end(callback, f"Error: {e}", _step_id, {"status": "error"})
                    all_events.append({
                        "type": "step_end", "content": f"Error: {e}",
                        "id": _step_id, "offset": _current_offset, "status": "error"
                    })
                    _result_str = f"Error: {e}"
                    _result     = {"error": _result_str}

            _temp_call = self.add_message(
                sender=personality.name, sender_type="assistant",
                content=_so_far, parent_id=_current_branch_tip
            )
            _temp_res = self.add_message(
                sender="system", sender_type="user",
                content=f"<tool_result name=\"{_tool_name}\">{_result_str}</tool_result>",
                parent_id=_temp_call.id
            )
            _temp_msg_ids.extend([_temp_call.id, _temp_res.id])
            _current_branch_tip = _temp_res.id

        # ====================================================================
        #  Forced final-answer pass
        # ====================================================================
        if is_agentic_turn and not _clean_text_so_far.strip():
            _final_id  = _step_start(callback, "Generating final answer...")
            _final_buf: List[str] = []

            def _final_relay(chunk, msg_type=None, meta=None):
                if isinstance(chunk, str):
                    _final_buf.append(chunk)
                    if callback is not None:
                        return _cb(callback, chunk, MSG_TYPE.MSG_TYPE_CHUNK, meta)
                return True

            _scratch_before_final = self.scratchpad
            self.scratchpad = (
                (_scratch_before_final or "")
                + "\n\n[SYSTEM INSTRUCTION] All tool calls are complete. "
                "Write your final answer to the user in plain text. "
                "Do NOT emit any more <tool_call> tags."
            )
            self._stream_final_answer(
                _final_relay, images, _current_branch_tip, final_answer_temperature, **kwargs
            )
            self.scratchpad = _scratch_before_final

            _final_text        = "".join(_final_buf)
            _accumulated_full += _final_text
            _clean_text_so_far = _final_text
            _step_end(callback, "Final answer generated", _final_id)

        # ── Clean up temporary tool-history messages ─────────────────────────
        for mid in reversed(_temp_msg_ids):
            if hasattr(self, 'remove_message'):
                self.remove_message(mid)
            elif hasattr(self, 'delete_message'):
                self.delete_message(mid)
            else:
                self.db_manager.delete_message(mid)

        # ── Strip tool-call tags; keep lollms_event markers ──────────────────
        import re as _re
        _clean = _re.sub(
            r"<tool_call>.*?(?:</tool_call>|$)", "", _clean_text_so_far, flags=_re.DOTALL
        ).strip()
        if remove_thinking_blocks:
            _clean = self.lollmsClient.remove_thinking_blocks(_clean)

        end_time    = datetime.now()
        duration    = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(_clean)
        tok_per_sec = (token_count / duration) if duration > 0 else 0

        message_meta: Dict[str, Any] = {
            "mode": ("rlm_agentic" if rlm_enabled
                     else ("agentic" if is_agentic_turn else "direct")),
            "duration_seconds":  duration,
            "token_count":       token_count,
            "tokens_per_second": tok_per_sec,
        }
        if tool_calls_this_turn:
            message_meta["tool_calls"]           = tool_calls_this_turn
        if all_events:
            message_meta["events"]               = all_events
        if collected_sources:
            message_meta["sources"]              = collected_sources
        if queries_performed:
            message_meta["query_history"]        = queries_performed
        if is_agentic_turn:
            message_meta["scratchpad"]           = scratchpad_state
        if self_corrections:
            message_meta["self_corrections"]     = self_corrections
        if _pt_specs:
            message_meta["personality_tools_used"] = [
                n for n in _pt_specs
                if any(tc["name"] == n for tc in tool_calls_this_turn)
            ]

        # Clear volatile scratchpad
        self.scratchpad = ""

        ai_message.content          = _clean
        ai_message.raw_content      = _accumulated_full
        ai_message.tokens           = token_count
        ai_message.generation_speed = tok_per_sec
        ai_message.metadata         = message_meta

        cleaned_content, affected_artefacts = self._post_process_llm_response(
            _clean, ai_message, _eff_img_gen, _eff_img_edit, auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets,
            enable_notes=enable_notes,
            enable_skills=enable_skills,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation,
        )
        if cleaned_content != _clean:
            ai_message.content = cleaned_content

        if affected_artefacts:
            message_meta["artefacts_modified"] = [a.get("title") for a in affected_artefacts]
            ai_message.metadata = message_meta
            if callback:
                _cb(callback, json.dumps(message_meta["artefacts_modified"]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                    {"artefacts": affected_artefacts})

        if self._is_db_backed and self.autosave:
            self.commit()

        # Clear the stashed callback — must not leak into the next turn
        object.__setattr__(self, '_active_callback', None)

        return {
            "user_message":     user_msg,
            "ai_message":       ai_message,
            "sources":          collected_sources,
            "scratchpad":       scratchpad_state if is_agentic_turn else None,
            "self_corrections": self_corrections or None,
            "artefacts":        affected_artefacts,
        }