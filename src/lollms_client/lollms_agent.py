# lollms_agent.py
# Agent: a named, role-bearing wrapper around a LollmsClient + LollmsPersonality.
# Used as the unit of participation in a SwarmOrchestrator run.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lollms_client.lollms_core import LollmsClient
    from lollms_client.lollms_personality import LollmsPersonality


# ---------------------------------------------------------------------------
# AgentRole — semantic label for an agent's function inside the swarm
# ---------------------------------------------------------------------------

class AgentRole:
    """
    Well-known role labels.  Custom strings are also accepted; these are
    just constants to avoid typos and to let the orchestrator apply
    role-specific prompt rules automatically.
    """
    PROPOSER        = "proposer"        # introduces initial ideas / drafts
    CRITIC          = "critic"          # challenges, finds flaws, stress-tests
    DEVIL_ADVOCATE  = "devil_advocate"  # argues the unpopular / opposite position
    DOMAIN_EXPERT   = "domain_expert"   # deep specialist contribution
    SYNTHESIZER     = "synthesizer"     # integrates all viewpoints into a conclusion
    MODERATOR       = "moderator"       # drives the session, asks clarifying questions
    IMPLEMENTER     = "implementer"     # turns plans into code / artefacts
    TESTER          = "tester"          # reviews artefacts for correctness / edge cases
    NARRATOR        = "narrator"        # describes events in simulation / game modes
    PLAYER          = "player"          # participates in a game or simulation as a character
    FREEFORM        = "freeform"        # no prescribed role; acts on its own judgment


@dataclass
class Agent:
    """
    A single participant in a swarm run.

    Parameters
    ----------
    lc : LollmsClient
        The LLM binding this agent uses.  Multiple agents may share the same
        binding or use completely different ones (e.g. one local, one cloud).
    personality : LollmsPersonality
        The personality that defines the agent's system prompt, knowledge base,
        and any personality-level tools.  The personality's ``name`` attribute
        is used as the agent's display name in the discussion unless ``name``
        is explicitly provided.
    name : str | None
        Override the display name.  Defaults to ``personality.name``.
    role : str
        Semantic role label (see ``AgentRole``).  Influences the anti-sycophancy
        prompt injected by the orchestrator.
    tools : dict | None
        External tool specs in the same format accepted by ``chat(tools=...)``.
    model_params : dict
        Extra kwargs forwarded to every LLM call this agent makes
        (e.g. ``{"temperature": 0.8, "n_predict": 2048}``).
    max_tokens_per_turn : int
        Soft cap on how many tokens this agent generates per round.
        Passed as ``n_predict`` if the binding honours it.
    metadata : dict
        Arbitrary application-layer metadata (avatar URL, colour, etc.).
    """
    lc:                  Any                   # LollmsClient
    personality:         Any                   # LollmsPersonality
    name:                Optional[str]         = None
    role:                str                   = AgentRole.FREEFORM
    tools:               Optional[Dict]        = None
    model_params:        Dict[str, Any]        = field(default_factory=dict)
    max_tokens_per_turn: int                   = 1024
    metadata:            Dict[str, Any]        = field(default_factory=dict)

    # set by the orchestrator at the start of a swarm run
    _agent_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)

    # ---------------------------------------------------------------- derived

    @property
    def display_name(self) -> str:
        return self.name or getattr(self.personality, "name", "Agent")

    @property
    def system_prompt(self) -> str:
        return getattr(self.personality, "system_prompt", "") or ""

    def has_knowledge(self) -> bool:
        return getattr(self.personality, "has_data", False)

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return (f"<Agent name={self.display_name!r} role={self.role!r} "
                f"id={self._agent_id[:8]}>")

    # ---------------------------------------------------------------- helpers

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 1024,
        streaming_callback: Optional[Callable] = None,
        **extra,
    ) -> str:
        """
        Direct (non-discussion) generation from this agent's binding.
        Used by the orchestrator for HLF self-assessment calls and for
        agents whose turn does not produce a persistent discussion message.
        """
        kwargs: Dict[str, Any] = {
            **self.model_params,
            "temperature":        temperature,
            "n_predict":          min(n_predict, self.max_tokens_per_turn),
            **extra,
        }
        if streaming_callback:
            kwargs["streaming_callback"] = streaming_callback

        full_prompt = ""
        if system_prompt:
            full_prompt = f"!@>system:\n{system_prompt}\n!@>user:\n{prompt}\n!@>assistant:\n"
        else:
            full_prompt = prompt

        try:
            return self.lc.generate_text(full_prompt, **kwargs) or ""
        except Exception as e:
            from ascii_colors import trace_exception
            trace_exception(e)
            return f"[{self.display_name} generation error: {e}]"

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, str],
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Optional[Dict]:
        """Structured JSON generation for HLF messages."""
        try:
            return self.lc.generate_structured_content(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt or self.system_prompt,
                temperature=temperature,
            )
        except Exception as e:
            from ascii_colors import trace_exception
            trace_exception(e)
            return None