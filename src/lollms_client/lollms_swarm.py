# lollms_swarm.py
# SwarmOrchestrator: multi-agent collaboration engine for LollmsDiscussion.
#
# Architecture
# ------------
# A swarm run takes one user question and routes it through N agents across
# R rounds, producing a single discussion message per agent per round plus
# a final synthesised answer.
#
# Two communication layers
# ------------------------
# HLF (High-Level Format): structured JSON exchanged between agents internally.
#   Invisible to the human user by default.  Contains structured proposals,
#   critiques, votes, artefact references, confidence scores.
#
# NLP: natural-language text streamed to the user callback via MSG_TYPE_CHUNK.
#   Each agent's visible contribution is saved as a LollmsMessage with
#   sender = agent.display_name, sender_type = "assistant".
#
# Anti-sycophancy
# ---------------
# Every agent receives an explicit prompt section that forbids empty agreement.
# Agents must cite specific points they disagree with or add genuinely new
# content.  Confidence scores self-reported in HLF drive convergence detection
# so that low-confidence agreement doesn't masquerade as consensus.
#
# Modes
# -----
# quality      Brainstorm → critique → synthesise. Default, best for most tasks.
# debate       Two teams argue for/against; moderator judges.
# simulation   Agents play characters in a persistent world; user steers.
# game         Structured game (cards, strategy, etc.); rules in system prompt.
# freeform     No prescribed structure; agents decide their own dynamics.

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_types import MSG_TYPE

# We import Agent lazily to avoid circular imports at module level
# from lollms_client.lollms_agent import Agent, AgentRole


# ---------------------------------------------------------------------------
# New MSG_TYPE values (appended to the enum at runtime so lollms_types.py
# doesn't need to be recompiled — we extend the enum dynamically)
# ---------------------------------------------------------------------------

def _extend_msg_type():
    """Add swarm-specific event types to MSG_TYPE if not already present."""
    additions = {
        "MSG_TYPE_SWARM_AGENT_START": 32,
        "MSG_TYPE_SWARM_AGENT_END":   33,
        "MSG_TYPE_SWARM_ROUND_START": 34,
        "MSG_TYPE_SWARM_ROUND_END":   35,
        "MSG_TYPE_SWARM_HLF":         36,
        "MSG_TYPE_SWARM_CONSENSUS":   37,
    }
    for name, value in additions.items():
        if not hasattr(MSG_TYPE, name):
            # Extend the Enum at runtime
            try:
                from enum import Enum
                MSG_TYPE._value2member_map_[value] = \
                    MSG_TYPE._member_map_[name] = \
                    MSG_TYPE.__new__(MSG_TYPE, value)
                MSG_TYPE._member_map_[name]._name_  = name
                MSG_TYPE._member_map_[name]._value_ = value
            except Exception:
                pass   # graceful degradation — fall back to numeric constants

_extend_msg_type()

# Safe references (fall back to raw int if enum extension failed)
def _mt(name: str, fallback: int) -> Any:
    return getattr(MSG_TYPE, name, fallback)

MT_SWARM_AGENT_START = _mt("MSG_TYPE_SWARM_AGENT_START", 32)
MT_SWARM_AGENT_END   = _mt("MSG_TYPE_SWARM_AGENT_END",   33)
MT_SWARM_ROUND_START = _mt("MSG_TYPE_SWARM_ROUND_START", 34)
MT_SWARM_ROUND_END   = _mt("MSG_TYPE_SWARM_ROUND_END",   35)
MT_SWARM_HLF         = _mt("MSG_TYPE_SWARM_HLF",         36)
MT_SWARM_CONSENSUS   = _mt("MSG_TYPE_SWARM_CONSENSUS",   37)


# ---------------------------------------------------------------------------
# HLF message types
# ---------------------------------------------------------------------------

class HLFType:
    PROPOSAL      = "proposal"       # agent introduces a new idea or draft
    CRITIQUE      = "critique"       # agent challenges a specific point
    QUESTION      = "question"       # agent asks another agent for clarification
    ANSWER        = "answer"         # agent responds to a question
    VOTE          = "vote"           # agent votes on a proposal (game/debate)
    ARTEFACT_PATCH= "artefact_patch" # agent signals intent to patch a shared artefact
    DIRECTIVE     = "directive"      # orchestrator steers all agents
    STEER         = "steer"          # user steering injected by app layer
    SYNTHESIS     = "synthesis"      # final answer being assembled
    GAME_ACTION   = "game_action"    # a player's move in a game/simulation
    NARRATION     = "narration"      # narrator describes world state


@dataclass
class HLFMessage:
    """
    A single inter-agent message in the High-Level Format.
    Never shown directly to the user; used to build agent context.
    """
    from_agent:    str                       # display_name of sender
    msg_type:      str                       # HLFType constant
    content:       str                       # the structured payload (prose or JSON)
    to_agent:      str           = "all"     # display_name or "all"
    round_num:     int           = 0
    artefact_ref:  Optional[str] = None      # title of affected artefact
    confidence:    float         = 1.0       # self-reported 0–1
    msg_id:        str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:     str           = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "id":           self.msg_id,
            "from":         self.from_agent,
            "to":           self.to_agent,
            "type":         self.msg_type,
            "round":        self.round_num,
            "content":      self.content,
            "artefact_ref": self.artefact_ref,
            "confidence":   self.confidence,
            "ts":           self.timestamp,
        }

    def to_context_str(self) -> str:
        """Compact rendering injected into agent context windows."""
        to_str = f" → {self.to_agent}" if self.to_agent != "all" else ""
        conf   = f" [conf={self.confidence:.2f}]" if self.confidence < 1.0 else ""
        art    = f" [artefact: {self.artefact_ref}]" if self.artefact_ref else ""
        return (f"[{self.msg_type.upper()}]{to_str}{art}{conf}\n"
                f"{self.from_agent}: {self.content}")


# ---------------------------------------------------------------------------
# SwarmConfig
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """
    Controls every aspect of a swarm run.

    Parameters
    ----------
    mode : str
        ``"quality"``    — brainstorm → critique → synthesise (default)
        ``"debate"``     — two sides argue; moderator judges
        ``"simulation"`` — agents play characters in a persistent world
        ``"game"``       — structured game with explicit rules
        ``"freeform"``   — no prescribed structure
    max_rounds : int
        Maximum deliberation rounds before forced synthesis.  Each round
        has every agent speak once.
    convergence_threshold : float
        If the mean self-reported confidence across all agents exceeds this
        value AND all agents produced only ``proposal`` or ``answer`` type
        messages (no outstanding critiques), the orchestrator stops early.
        Set to 1.0 to disable early stopping.
    show_deliberation : bool
        If True, every agent's NLP output is streamed to the user in real
        time.  If False, only the final synthesis is shown.
    moderator_index : int
        Index of the agent in the list that will produce the final synthesis
        message.  Defaults to 0 (first agent).
    allow_artefact_collaboration : bool
        If True, all agents share the discussion's artefact store and may
        create, patch, and revert artefacts freely.
    user_steer_prefix : str
        When a user steering message is injected mid-swarm, it is prepended
        with this prefix so agents recognise it as an external directive.
    synthesis_prompt_suffix : str
        Extra instructions appended to the synthesiser agent's final-round
        system prompt.  Use this to shape the output format.
    game_rules : str
        For ``"game"`` mode — description of rules injected into every
        agent's system prompt at the start of the run.
    world_state : str
        For ``"simulation"`` mode — initial world description.
    max_nlp_tokens_per_agent : int
        Soft limit on visible NLP output per agent per round.
    anti_sycophancy_strength : str
        ``"light"``  — remind agents to add new value (gentle)
        ``"medium"`` — explicit prohibition on agreement without substance
        ``"strong"`` — adversarial framing; agents must find fault before agreeing
    """
    mode:                       str   = "quality"
    max_rounds:                 int   = 3
    convergence_threshold:      float = 0.85
    show_deliberation:          bool  = True
    moderator_index:            int   = 0
    allow_artefact_collaboration: bool = True
    user_steer_prefix:          str   = "⚡ USER DIRECTIVE:"
    synthesis_prompt_suffix:    str   = ""
    game_rules:                 str   = ""
    world_state:                str   = ""
    max_nlp_tokens_per_agent:   int   = 512
    anti_sycophancy_strength:   str   = "medium"  # "light" | "medium" | "strong"

    def validate(self):
        assert self.mode in ("quality", "debate", "simulation", "game", "freeform"), \
            f"Unknown swarm mode: {self.mode!r}"
        assert 1 <= self.max_rounds <= 20, "max_rounds must be 1–20"
        assert 0.0 < self.convergence_threshold <= 1.0
        assert self.anti_sycophancy_strength in ("light", "medium", "strong")


# ---------------------------------------------------------------------------
# Anti-sycophancy prompt fragments
# ---------------------------------------------------------------------------

_ANTI_SYC = {
    "light": (
        "When reviewing other agents' contributions, add genuinely new value. "
        "Brief acknowledgement of agreement is fine, but always extend or qualify."
    ),
    "medium": (
        "ANTI-SYCOPHANCY RULE: You MUST NOT simply confirm or rephrase what a previous "
        "agent said. If you agree on a specific point, state it in one sentence then "
        "IMMEDIATELY add something new: a different angle, an edge case, a counter-example, "
        "a risk, or a concrete improvement. Generic praise ('great point!') is forbidden. "
        "Your confidence score must reflect genuine uncertainty — use values below 0.7 "
        "whenever you are not fully certain."
    ),
    "strong": (
        "ADVERSARIAL REVIEW RULE: Before accepting any claim from another agent you must "
        "actively try to find a flaw, limitation, or missing consideration. Only after "
        "you have articulated what could go wrong may you indicate partial agreement. "
        "If you cannot find a flaw, say so explicitly — do not pretend to disagree. "
        "Your confidence score must be ≤ 0.6 on your first pass for any proposal "
        "not yet stress-tested."
    ),
}

_ROLE_HINTS = {
    "proposer":       "Your job is to introduce bold, concrete ideas. Be specific and actionable.",
    "critic":         "Your job is to find weaknesses, missing cases, and unstated assumptions.",
    "devil_advocate": "Your job is to argue the opposite position as forcefully as possible, even if you privately agree.",
    "domain_expert":  "Your job is to bring deep, specific domain knowledge that others may lack.",
    "synthesizer":    "Your job is to integrate all perspectives into a coherent, balanced conclusion.",
    "moderator":      "Your job is to keep the discussion productive, resolve conflicts, and ask clarifying questions.",
    "implementer":    "Your job is to turn plans into working artefacts (code, documents). Be precise and complete.",
    "tester":         "Your job is to probe artefacts and proposals for bugs, edge cases, and correctness.",
    "narrator":       "Your job is to describe the world state and events in vivid, coherent prose.",
    "player":         "You are a character in a simulation. Stay in character. React authentically.",
    "freeform":       "You decide your own approach. Bring whatever perspective feels most valuable.",
}


# ---------------------------------------------------------------------------
# SwarmOrchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """
    Runs a multi-agent deliberation session inside a LollmsDiscussion.

    Usage (called from ChatMixin.chat when swarm= is provided):
    -----------------------------------------------------------
    orchestrator = SwarmOrchestrator(
        discussion=discussion,
        agents=swarm,
        config=swarm_config or SwarmConfig(),
        callback=streaming_callback,
    )
    result = orchestrator.run(user_message, **kwargs)
    """

    def __init__(
        self,
        discussion:  Any,              # LollmsDiscussion
        agents:      List[Any],        # List[Agent]
        config:      SwarmConfig,
        callback:    Optional[Callable] = None,
        user_msg_id: Optional[str]     = None,
    ):
        if len(agents) < 1:
            raise ValueError("Swarm requires at least one agent.")
        config.validate()

        self.discussion  = discussion
        self.agents      = agents
        self.config      = config
        self.callback    = callback
        self.user_msg_id = user_msg_id

        # HLF log — full history, never pruned
        self.hlf_log: List[HLFMessage] = []

        # Saved agent messages (LollmsMessage objects) per round
        self.agent_messages: List[Any] = []

        # Accumulated artefact titles touched this run
        self.touched_artefacts: List[str] = []

        # Steering messages injected by the application layer mid-run
        self._pending_steers: List[str] = []

        # Run-level stats
        self._start_time = datetime.now()

    # ---------------------------------------------------------------- public API

    def run(self, user_message: str, **chat_kwargs) -> Dict[str, Any]:
        """
        Execute the full swarm session for one user message.
        Returns a result dict compatible with chat()'s return value.
        """
        ASCIIColors.cyan(
            f"\n{'='*60}\n"
            f"SWARM  mode={self.config.mode}  agents={len(self.agents)}"
            f"  max_rounds={self.config.max_rounds}\n"
            f"{'='*60}"
        )

        # ── Mode-specific init ────────────────────────────────────────────
        if self.config.mode == "game" and self.config.game_rules:
            self._inject_directive(
                f"GAME RULES:\n{self.config.game_rules}", round_num=0)
        if self.config.mode == "simulation" and self.config.world_state:
            self._inject_directive(
                f"INITIAL WORLD STATE:\n{self.config.world_state}", round_num=0)

        self._emit(MT_SWARM_ROUND_START, "Swarm session starting",
                   {"mode": self.config.mode, "agents": [a.display_name for a in self.agents]})

        # ── Main deliberation loop ────────────────────────────────────────
        final_synthesis: Optional[Any] = None  # LollmsMessage
        all_affected_artefacts: List[Dict] = []

        for round_num in range(1, self.config.max_rounds + 1):
            # Emit synthesis processing tag opening
            synthesis_attrs = f' type="synthesis" round="{round_num}" agents="{len(self.agents)}" mode="{self.config.mode}"'
            _cb(self.callback, f"<processing{synthesis_attrs}>", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_open",
                "processing_type": "synthesis",
                "round": round_num,
                "agents": len(self.agents),
                "mode": self.config.mode,
            })
            
            self._emit(MT_SWARM_ROUND_START, f"Round {round_num}/{self.config.max_rounds}",
                       {"round": round_num})
            _cb(self.callback, f"\n\n---\n**Round {round_num}**\n", MSG_TYPE.MSG_TYPE_CHUNK)

            round_messages: List[Any] = []
            round_confidences: List[float] = []
            has_unresolved_critique = False

            for agent in self.agents:
                # ── Apply any pending steers ──────────────────────────────
                if self._pending_steers:
                    for steer in self._pending_steers:
                        self._inject_directive(
                            f"{self.config.user_steer_prefix} {steer}",
                            round_num=round_num)
                    self._pending_steers.clear()

                # ── Build agent context for this turn ─────────────────────
                system_prompt = self._build_agent_system_prompt(
                    agent, round_num, user_message)
                hlf_context   = self._build_hlf_context(agent, round_num)

                # ── Stream NLP output ─────────────────────────────────────
                self._emit(MT_SWARM_AGENT_START,
                           f"{agent.display_name} ({agent.role})",
                           {"agent": agent.display_name, "role": agent.role,
                            "round": round_num})

                if self.config.show_deliberation:
                    _cb(self.callback,
                        f"\n\n**{agent.display_name}** *(round {round_num})*\n",
                        MSG_TYPE.MSG_TYPE_CHUNK)

                nlp_buf: List[str] = []

                def _agent_relay(chunk, msg_type=None, meta=None):
                    if isinstance(chunk, str) and self.config.show_deliberation:
                        nlp_buf.append(chunk)
                        return _cb(self.callback, chunk,
                                   msg_type or MSG_TYPE.MSG_TYPE_CHUNK, meta)
                    return True

                nlp_text = agent.generate(
                    prompt=self._build_nlp_prompt(
                        agent, round_num, user_message, hlf_context),
                    system_prompt=system_prompt,
                    temperature=agent.model_params.get("temperature", 0.75),
                    n_predict=self.config.max_nlp_tokens_per_agent,
                    streaming_callback=_agent_relay,
                )
                if not nlp_text and nlp_buf:
                    nlp_text = "".join(nlp_buf)
                nlp_text = nlp_text.strip()

                # ── Save NLP as a discussion message ──────────────────────
                agent_msg = self.discussion.add_message(
                    sender=agent.display_name,
                    sender_type="assistant",
                    content=nlp_text,
                    parent_id=self.user_msg_id,
                    model_name=getattr(agent.lc.llm, "model_name", ""),
                    binding_name=getattr(agent.lc.llm, "binding_name", ""),
                    metadata={
                        "swarm_round":  round_num,
                        "swarm_role":   agent.role,
                        "swarm_agent":  agent.display_name,
                        "agent_id":     agent._agent_id,
                    },
                )
                round_messages.append(agent_msg)
                self.agent_messages.append(agent_msg)

                # ── Post-process for artefacts ────────────────────────────
                # Each agent can create/patch artefacts via XML tags
                old_cb = getattr(self.discussion, '_active_callback', None)
                object.__setattr__(self.discussion, '_active_callback', self.callback)
                cleaned, affected = self.discussion._post_process_llm_response(
                    nlp_text, agent_msg,
                    enable_image_generation=False,
                    enable_image_editing=False,
                    auto_activate_artefacts=True,
                    enable_inline_widgets=True,
                    enable_notes=True,
                    enable_skills=False,
                    enable_silent_artefact_explanation=False,
                )
                object.__setattr__(self.discussion, '_active_callback', old_cb)
                if cleaned != nlp_text:
                    agent_msg.content = cleaned
                all_affected_artefacts.extend(affected)
                for art in affected:
                    t = art.get("title", "")
                    if t and t not in self.touched_artefacts:
                        self.touched_artefacts.append(t)

                # ── Generate HLF self-assessment ──────────────────────────
                hlf = self._generate_hlf(agent, nlp_text, round_num)
                if hlf:
                    self.hlf_log.append(hlf)
                    self._emit(MT_SWARM_HLF, json.dumps(hlf.to_dict()),
                               {"hlf": hlf.to_dict()})
                    round_confidences.append(hlf.confidence)
                    if hlf.msg_type == HLFType.CRITIQUE:
                        has_unresolved_critique = True
                else:
                    round_confidences.append(0.8)

                self._emit(MT_SWARM_AGENT_END,
                           f"{agent.display_name} done",
                           {"agent": agent.display_name, "round": round_num})

            # Emit synthesis status for this round
            status_msg = f"Round {round_num} complete"
            if round_confidences:
                mean_conf = sum(round_confidences) / len(round_confidences)
                status_msg += f" (mean confidence: {mean_conf:.2f})"
            _cb(self.callback, f"\n* {status_msg}", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_status",
                "processing_type": "synthesis",
                "status": status_msg,
            })
            
            self._emit(MT_SWARM_ROUND_END, f"Round {round_num} complete",
                       {"round": round_num,
                        "mean_confidence": (sum(round_confidences) / len(round_confidences))
                                           if round_confidences else 1.0})

            # Close synthesis processing tag for this round
            _cb(self.callback, "</processing>", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_close",
                "processing_type": "synthesis",
                "round": round_num,
            })

            # ── Convergence check ─────────────────────────────────────────
            if (not has_unresolved_critique
                    and round_confidences
                    and (sum(round_confidences) / len(round_confidences))
                        >= self.config.convergence_threshold
                    and round_num >= 2):
                self._emit(MT_SWARM_CONSENSUS,
                           f"Consensus reached after {round_num} rounds",
                           {"round": round_num,
                            "confidence": sum(round_confidences)/len(round_confidences)})
                ASCIIColors.green(f"  [Swarm] Consensus at round {round_num}")
                break

        # ── Final synthesis ───────────────────────────────────────────────
        # Emit final synthesis processing
        _cb(self.callback, '<processing type="synthesis" stage="final">', MSG_TYPE.MSG_TYPE_CHUNK, {
            "type": "processing_open",
            "processing_type": "synthesis",
            "stage": "final",
        })
        moderator = self.agents[
            min(self.config.moderator_index, len(self.agents) - 1)]
        final_synthesis = self._synthesise(
            moderator, user_message, all_affected_artefacts, **chat_kwargs)

        if self.config.show_deliberation:
            _cb(self.callback,
                f"\n\n---\n**{moderator.display_name} (synthesis)**\n",
                MSG_TYPE.MSG_TYPE_CHUNK)
        if final_synthesis:
            _cb(self.callback, f"\n* {moderator.display_name} synthesizing final answer...", 
                MSG_TYPE.MSG_TYPE_CHUNK, {
                    "type": "processing_status",
                    "processing_type": "synthesis",
                    "status": f"{moderator.display_name} synthesizing final answer...",
                })
            
            syn_buf: List[str] = []

            def _syn_relay(chunk, msg_type=None, meta=None):
                if isinstance(chunk, str):
                    syn_buf.append(chunk)
                    return _cb(self.callback, chunk,
                               msg_type or MSG_TYPE.MSG_TYPE_CHUNK, meta)
                return True

            syn_text = moderator.generate(
                prompt=self._build_synthesis_prompt(moderator, user_message),
                system_prompt=self._build_agent_system_prompt(
                    moderator, 99, user_message,
                    suffix=self.config.synthesis_prompt_suffix),
                temperature=moderator.model_params.get("temperature", 0.6),
                n_predict=moderator.max_tokens_per_turn,
                streaming_callback=_syn_relay,
            )
            if not syn_text and syn_buf:
                syn_text = "".join(syn_buf)
            syn_text = syn_text.strip()

            final_synthesis.content = syn_text

            # Post-process synthesis for artefacts too
            old_cb = getattr(self.discussion, '_active_callback', None)
            object.__setattr__(self.discussion, '_active_callback', self.callback)
            cleaned_syn, syn_affected = self.discussion._post_process_llm_response(
                syn_text, final_synthesis,
                enable_silent_artefact_explanation=True,
            )
            object.__setattr__(self.discussion, '_active_callback', old_cb)
            if cleaned_syn != syn_text:
                final_synthesis.content = cleaned_syn
            all_affected_artefacts.extend(syn_affected)
            
            _cb(self.callback, f"\n* Synthesis complete", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_status",
                "processing_type": "synthesis",
                "status": "Synthesis complete",
            })
            
            # Close final synthesis processing
            _cb(self.callback, "</processing>", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_close",
                "processing_type": "synthesis",
                "stage": "final",
            })

        if self.discussion._is_db_backed and self.discussion.autosave:
            self.discussion.commit()

        duration = (datetime.now() - self._start_time).total_seconds()
        return {
            "user_message":     None,    # filled by chat()
            "ai_message":       final_synthesis,
            "agent_messages":   self.agent_messages,
            "hlf_log":          [m.to_dict() for m in self.hlf_log],
            "sources":          [],
            "scratchpad":       None,
            "self_corrections": None,
            "artefacts":        all_affected_artefacts,
            "swarm_meta": {
                "mode":             self.config.mode,
                "rounds_run":       min(len(self.hlf_log), self.config.max_rounds),
                "agents":           [a.display_name for a in self.agents],
                "touched_artefacts": self.touched_artefacts,
                "duration_seconds": duration,
            },
        }

    def steer(self, directive: str):
        """
        Inject a user steering directive that will be delivered to all
        agents at the start of the next round.  Call from the application
        layer (e.g. on a user interrupt during a simulation).
        """
        self._pending_steers.append(directive)

    # ---------------------------------------------------------------- internals

    def _emit(self, msg_type: Any, text: str, meta: Optional[Dict] = None):
        _cb(self.callback, text, msg_type, meta or {})

    def _inject_directive(self, content: str, round_num: int):
        msg = HLFMessage(
            from_agent="orchestrator",
            msg_type=HLFType.DIRECTIVE,
            content=content,
            round_num=round_num,
            confidence=1.0,
        )
        self.hlf_log.append(msg)

    def _build_agent_system_prompt(
        self,
        agent: Any,
        round_num: int,
        user_question: str,
        suffix: str = "",
    ) -> str:
        parts: List[str] = []

        # 1. Personality system prompt
        if agent.system_prompt:
            parts.append(agent.system_prompt)

        # 2. Swarm context
        parts.append(
            f"\n=== SWARM SESSION ===\n"
            f"Mode: {self.config.mode}\n"
            f"Your role: {agent.role} — {_ROLE_HINTS.get(agent.role, '')}\n"
            f"Participants: {', '.join(a.display_name for a in self.agents)}\n"
            f"=== END SWARM CONTEXT ==="
        )

        # 3. Mode-specific rules
        if self.config.mode == "game" and self.config.game_rules:
            parts.append(f"\n=== GAME RULES ===\n{self.config.game_rules}\n=== END ===")
        if self.config.mode == "simulation" and self.config.world_state:
            parts.append(f"\n=== WORLD STATE ===\n{self.config.world_state}\n=== END ===")

        # 4. Anti-sycophancy
        if round_num > 1:
            parts.append(
                f"\n=== COLLABORATION RULES ===\n"
                f"{_ANTI_SYC[self.config.anti_sycophancy_strength]}\n"
                f"=== END ==="
            )

        # 5. Active artefacts (shared)
        artefact_zone = self.discussion.artefacts.build_artefacts_context_zone()
        if artefact_zone:
            parts.append(f"\n{artefact_zone}")

        # 6. Suffix (synthesis instructions etc.)
        if suffix:
            parts.append(f"\n{suffix}")

        return "\n".join(parts)

    def _build_hlf_context(self, agent: Any, round_num: int) -> str:
        """
        Builds the HLF context string injected into the agent's prompt.
        Includes messages from the current and previous rounds, filtered to
        messages addressed to this agent or to "all".
        """
        relevant = [
            m for m in self.hlf_log
            if (m.round_num >= round_num - 1
                and (m.to_agent == "all" or m.to_agent == agent.display_name)
                and m.from_agent != agent.display_name)
        ]
        if not relevant:
            return ""
        lines = ["=== INTER-AGENT MESSAGES (HLF) ==="]
        for m in relevant[-12:]:   # cap at 12 to stay lean
            lines.append(m.to_context_str())
        lines.append("=== END HLF ===")
        return "\n".join(lines)

    def _build_nlp_prompt(
        self,
        agent: Any,
        round_num: int,
        user_question: str,
        hlf_context: str,
    ) -> str:
        """
        Builds the full text prompt the agent sees when generating its
        visible NLP response.
        """
        parts: List[str] = []

        # User question
        parts.append(f"User question:\n{user_question}\n")

        # Previous NLP contributions from other agents this round
        prior_nlp = [
            f"{m.sender}: {m.content}"
            for m in self.agent_messages
            if (getattr(m, 'metadata', {}).get('swarm_round') == round_num
                and m.sender != agent.display_name)
        ]
        if prior_nlp:
            parts.append("\n--- Other agents' contributions this round ---")
            parts.extend(prior_nlp[-6:])    # cap to keep context lean
            parts.append("--- End contributions ---\n")

        # HLF context
        if hlf_context:
            parts.append(hlf_context)
            parts.append("")

        # Role instruction
        if round_num == 1:
            parts.append(
                f"This is round 1. As the {agent.role}, introduce your perspective "
                f"on the question above. Be specific and substantive."
            )
        else:
            parts.append(
                f"This is round {round_num}. Review what others have said, "
                f"then contribute as the {agent.role}. "
                f"Do not simply agree — bring new value or a specific challenge."
            )

        return "\n".join(parts)

    def _generate_hlf(
        self,
        agent: Any,
        nlp_text: str,
        round_num: int,
    ) -> Optional[HLFMessage]:
        """
        Asks the agent to self-assess its contribution and produce a
        structured HLF message.  This is a small, fast structured call.
        """
        schema = {
            "type":       f"One of: {', '.join([HLFType.PROPOSAL, HLFType.CRITIQUE, HLFType.QUESTION, HLFType.ANSWER, HLFType.GAME_ACTION, HLFType.NARRATION])}",
            "to":         "Target agent name or 'all'",
            "artefact":   "Artefact title if relevant, else empty string",
            "confidence": "Float 0.0–1.0: how confident are you in this contribution?",
            "summary":    "One sentence summary of your contribution for the HLF log",
        }
        result = agent.generate_structured(
            prompt=(
                f"You just wrote this response in round {round_num}:\n"
                f"---\n{nlp_text[:800]}\n---\n"
                f"Classify your contribution for the HLF log."
            ),
            schema=schema,
            temperature=0.1,
        )
        if not result:
            return None
        try:
            confidence = float(result.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.8

        return HLFMessage(
            from_agent   = agent.display_name,
            to_agent     = result.get("to", "all"),
            msg_type     = result.get("type", HLFType.PROPOSAL),
            content      = result.get("summary", nlp_text[:200]),
            artefact_ref = result.get("artefact") or None,
            confidence   = confidence,
            round_num    = round_num,
        )

    def _build_synthesis_prompt(self, moderator: Any, user_question: str) -> str:
        """
        Builds the prompt for the final synthesis pass.
        Includes all NLP contributions and the HLF summary.
        """
        parts: List[str] = [
            f"Original question: {user_question}\n",
            "=== ALL AGENT CONTRIBUTIONS ===",
        ]
        for msg in self.agent_messages:
            meta  = getattr(msg, 'metadata', {}) or {}
            rnd   = meta.get('swarm_round', '?')
            parts.append(
                f"\n[Round {rnd}] {msg.sender} ({meta.get('swarm_role','')}):\n"
                f"{msg.content}\n"
            )
        parts.append("=== END CONTRIBUTIONS ===\n")

        # Key HLF signals
        critiques = [m for m in self.hlf_log if m.msg_type == HLFType.CRITIQUE]
        if critiques:
            parts.append("Key critiques raised:")
            for c in critiques[-5:]:
                parts.append(f"  • [{c.from_agent}] {c.content[:120]}")
            parts.append("")

        if self.touched_artefacts:
            parts.append(
                f"Artefacts collaboratively produced: "
                f"{', '.join(self.touched_artefacts)}\n"
            )

        mode_instruction = {
            "quality":    (
                "Synthesise all perspectives into the best possible answer. "
                "Acknowledge unresolved disagreements honestly. "
                "Do not repeat every point — distil the key conclusions."
            ),
            "debate":     (
                "Judge which side made the stronger case and explain why. "
                "Note the strongest counter-argument and how the winning side addressed it."
            ),
            "simulation": (
                "Describe the current world state after this round of events. "
                "Be vivid and specific about what changed and what remains uncertain."
            ),
            "game":       (
                "Describe the result of this round, update scores or state, "
                "and set up the next round if the game continues."
            ),
            "freeform":   (
                "Integrate all contributions into a coherent response that serves "
                "the user's original intent."
            ),
        }.get(self.config.mode, "Synthesise all contributions.")

        parts.append(
            f"=== YOUR TASK ===\n"
            f"{mode_instruction}\n"
            f"{self.config.synthesis_prompt_suffix}\n"
            f"=== END ==="
        )
        return "\n".join(parts)

    def _synthesise(
        self,
        moderator: Any,
        user_question: str,
        affected_artefacts: List[Dict],
        **kwargs,
    ) -> Any:          # LollmsMessage
        """Creates the final synthesis message stub in the discussion."""
        msg = self.discussion.add_message(
            sender=moderator.display_name,
            sender_type="assistant",
            content="",
            parent_id=self.user_msg_id,
            model_name=getattr(moderator.lc.llm, "model_name", ""),
            binding_name=getattr(moderator.lc.llm, "binding_name", ""),
            metadata={
                "swarm_mode":     self.config.mode,
                "swarm_synthesis": True,
                "swarm_agents":   [a.display_name for a in self.agents],
                "swarm_rounds":   self.config.max_rounds,
                "artefacts_modified": [a.get("title") for a in affected_artefacts],
            },
        )
        return msg


# ---------------------------------------------------------------------------
# Helper (mirrors the one in _mixin_chat.py)
# ---------------------------------------------------------------------------

def _cb(callback, text: str, msg_type: Any, meta: Optional[Dict] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        if result is False:
            return False
    except Exception as e:
        trace_exception(e)
    return True