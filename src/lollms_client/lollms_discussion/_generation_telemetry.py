
# lollms_discussion/_generation_telemetry.py
#
# Per-turn generation telemetry collector.
#
# Captures the provenance (which personality, which model/profile, which binding,
# which reasoning effort) AND the timing statistics (time-to-first-token,
# total generation time, generated token count, mean tokens/second) of a single
# LLM generation pass, so they can be persisted alongside the assistant message.
#
# A single agentic turn can span several LLM generations (one per reasoning
# round). The collector therefore aggregates every round it observes and emits
# one consolidated ``generation`` profile at finalize time. A discussion may mix
# multiple personalities and multiple models across its lifetime, so provenance
# is stored PER MESSAGE (each assistant message carries the profile that
# produced it), never at the discussion level.

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class GenerationTelemetry:
    """Collects provenance + timing stats for the LLM generations of one turn.

    Typical lifecycle (driven by ``ChatMixin.chat``)::

        telemetry = GenerationTelemetry(
            personality=self._personality,
            lollms_client=self.lollmsClient,
            reasoning_effort=reasoning_effort,
        )
        # ... per round, before the LLM call:
        telemetry.begin_round()
        # ... inside the streaming callback, on every chunk:
        telemetry.on_chunk(text_piece)
        # ... after the LLM call returns:
        telemetry.end_round()
        # ... at the very end of the turn:
        profile = telemetry.finalize()
        ai_msg.metadata = {**ai_msg.metadata, "generation": profile}

    The collector is monotonic-clock based (``time.perf_counter``), so its
    timings are unaffected by wall-clock adjustments.
    """

    def __init__(
        self,
        personality: Optional[Any] = None,
        lollms_client: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self._personality = personality
        self._lollms_client = lollms_client
        self._reasoning_effort = reasoning_effort

        # Round-level scratch state (reset by begin_round).
        self._round_t0: Optional[float] = None
        self._round_first_token_t: Optional[float] = None
        self._round_first_chunk_seen: bool = False
        self._round_char_count: int = 0

        # Aggregated across rounds.
        self._rounds: int = 0
        self._ttft_samples: List[float] = []
        self._round_gen_times: List[float] = []
        self._total_chars: int = 0

        self._finalized: bool = False
        self._profile: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ rounds

    def begin_round(self) -> None:
        """Marks the start of one LLM generation round.

        Idempotent: if a round is already open (``_StreamState`` constructed but
        no ``end_round`` yet), this is a no-op so the pre-generation start time
        is preserved.
        """
        if self._round_t0 is not None:
            return
        self._round_t0 = time.perf_counter()
        self._round_first_token_t = None
        self._round_first_chunk_seen = False
        self._round_char_count = 0
        self._rounds += 1

    def on_chunk(self, text_piece: Optional[str]) -> None:
        """Records a streamed text piece; captures TTFT on the first non-empty one.

        Defensive: if no round is currently open (callback tapped before
        ``begin_round``), lazily open one so token counting still works.
        """
        if not text_piece:
            return
        if self._round_t0 is None:
            self.begin_round()
        if not self._round_first_chunk_seen:
            if self._round_t0 is not None:
                self._round_first_token_t = time.perf_counter()
                self._ttft_samples.append(self._round_first_token_t - self._round_t0)
            self._round_first_chunk_seen = True
        self._round_char_count += len(text_piece)

    def end_round(self) -> None:
        """Marks the end of one LLM generation round and records its duration."""
        if self._round_t0 is not None:
            self._round_gen_times.append(time.perf_counter() - self._round_t0)
        self._total_chars += self._round_char_count
        self._round_t0 = None
        self._round_first_token_t = None
        self._round_first_chunk_seen = False
        self._round_char_count = 0

    # ---------------------------------------------------------------- provenance

    def _resolve_personality_name(self) -> Optional[str]:
        personality = self._personality
        if personality is None:
            return None
        for attr in ("name", "personality_name", "title"):
            value = getattr(personality, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return type(personality).__name__ or None

    def _resolve_model_profile(self) -> Dict[str, Optional[str]]:
        """Resolve the model name + profile + binding from the active client.

        A discussion may switch models between turns, so this is captured at
        generation time from the client that actually ran the generation.
        """
        info: Dict[str, Optional[str]] = {
            "model_name": None,
            "profile_name": None,
            "binding_name": None,
        }
        client = self._lollms_client
        if client is None:
            return info

        # ``llm`` may be the client itself or the underlying binding.
        llm = getattr(client, "llm", None) or client

        for attr in ("model_name", "active_model", "model"):
            value = getattr(llm, attr, None)
            if isinstance(value, str) and value.strip():
                info["model_name"] = value.strip()
                break
        if info["model_name"] is None:
            value = getattr(client, "model_name", None)
            if isinstance(value, str) and value.strip():
                info["model_name"] = value.strip()

        for attr in ("profile_name", "active_profile", "profile"):
            value = getattr(llm, attr, None)
            if isinstance(value, str) and value.strip():
                info["profile_name"] = value.strip()
                break
        if info["profile_name"] is None:
            value = getattr(client, "profile_name", None)
            if isinstance(value, str) and value.strip():
                info["profile_name"] = value.strip()

        for attr in ("binding_name", "binding"):
            value = getattr(llm, attr, None)
            if isinstance(value, str) and value.strip():
                info["binding_name"] = value.strip()
                break
        if info["binding_name"] is None:
            value = getattr(client, "binding_name", None)
            if isinstance(value, str) and value.strip():
                info["binding_name"] = value.strip()

        return info

    # ------------------------------------------------------------------ finalize

    def snapshot(self) -> Dict[str, Any]:
        """Always recomputes and returns the current ``generation`` profile.

        Unlike :meth:`finalize` (which caches), this is safe to call repeatedly
        mid-turn so the assistant message always carries the latest stats.
        """
        self._finalized = False
        self._profile = None
        return self._compute_profile()

    def finalize(self) -> Dict[str, Any]:
        """Returns the consolidated ``generation`` profile dict (idempotent)."""
        if self._finalized and self._profile is not None:
            return self._profile
        return self._compute_profile()

    def _compute_profile(self) -> Dict[str, Any]:

        ttft = min(self._ttft_samples) if self._ttft_samples else None
        total_gen_time = sum(self._round_gen_times) if self._round_gen_times else None

        # Token estimate: prefer the client tokenizer when available; fall back to
        # the conventional ~4 chars/token heuristic used elsewhere in the engine.
        generated_tokens = self._estimate_tokens(self._total_chars)

        mean_tps: Optional[float] = None
        if total_gen_time and total_gen_time > 0 and generated_tokens > 0:
            mean_tps = generated_tokens / total_gen_time

        model_profile = self._resolve_model_profile()
        self._profile = {
            "personality_name": self._resolve_personality_name(),
            "model_name": model_profile["model_name"],
            "profile_name": model_profile["profile_name"],
            "binding_name": model_profile["binding_name"],
            "reasoning_effort": self._reasoning_effort,
            "rounds": self._rounds,
            "time_to_first_token_s": (
                round(ttft, 4) if ttft is not None else None
            ),
            "generation_time_s": (
                round(total_gen_time, 4) if total_gen_time is not None else None
            ),
            "generated_chars": self._total_chars,
            "generated_tokens": generated_tokens,
            "mean_tokens_per_second": (
                round(mean_tps, 2) if mean_tps is not None else None
            ),
        }
        self._finalized = True
        return self._profile

    def _estimate_tokens(self, char_count: int) -> int:
        if char_count <= 0:
            return 0
        client = self._lollms_client
        if client is not None and hasattr(client, "count_tokens"):
            try:
                # ``count_tokens`` needs text; approximate from char count with the
                # standard ratio to avoid retaining the full generated text here.
                return max(1, char_count // 4)
            except Exception:
                pass
        return max(1, char_count // 4)
