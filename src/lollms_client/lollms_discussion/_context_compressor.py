# lollms_discussion/_context_compressor.py
# Pure-heuristic (NO LLM) context-poisoning guard for the prompt builder.
#
# PURPOSE
# -------
# Long, tool-heavy agent turns accumulate: (a) large pinned file payloads that
# were loaded many rounds ago, (b) image packs in conversation history, and
# (c) a growing message history. Left unchecked the prompt balloons, dilutes
# the model's attention (context poisoning) and eventually overflows the window.
#
# This module decides, using ONLY cheap heuristics (char counts + recency +
# type), how to shrink a context back under a target budget. It NEVER calls an
# LLM. It is deterministic and side-effect free.
#
# STRATEGY (highest leverage first)
# ---------------------------------
# Three levers, spent cheapest/safest first:
#
#   Lever C — IMAGE CAP. Images dominate token cost. Cap retained images at
#             `max_images` (default 4, clamped to [1, 4]); the OLDEST image
#             packs are dropped first. Always safe, so it runs first.
#
#   Lever A — UNLOAD pinned files. Stale/large files loaded many rounds ago are
#             the classic poison. Unload OLDEST-and-largest first (recency-
#             weighted score), never blindly the biggest. The most-recent N
#             files stay pinned so the working set survives. BUDGETED: reclaims
#             at most `max_file_unload_ratio` of the remaining overflow, so a
#             small shortfall never wipes the whole working set.
#
#   Lever B — COMPRESS history. Whatever overflow the file lever did not (or
#             could not) reclaim is folded: the OLDEST compressible message
#             span becomes a single short note, while the system prompt, the
#             last K messages, and ALL tool-call/result structure stay verbatim.
#
# `plan()` weighs the levers and returns a CompressionPlan of concrete actions
# plus the projected post-compression size. `apply_plan()` resolves a plan into
# exact keep/drop sets so the host can rebuild the prompt directly. Neither
# method mutates its inputs — the module stays pure and unit-testable.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ── Tunable defaults (all overridable per-call) ────────────────────────────
DEFAULT_CONTEXT_THRESHOLD_CHARS = 180_000     # trigger compression above this
DEFAULT_TARGET_CHARS = 120_000                # aim below this after compression
DEFAULT_MAX_IMAGES = 4                        # keep at most 4 images at once
DEFAULT_KEEP_RECENT_FILES = 2                 # never unload the N newest files
DEFAULT_KEEP_RECENT_MESSAGES = 6              # never compress the N newest msgs
DEFAULT_MAX_FILE_UNLOAD_RATIO = 0.60          # files may free at most 60% of the
                                             # required overflow; the rest comes
                                             # from history compression, so we
                                             # never gut the working set.

# Approx tokens ≈ chars / 4 (English/code heuristic). Only used for reporting.
_CHARS_PER_TOKEN = 4.0

# Heuristic weights for file-unload priority.
_RECENCY_WEIGHT = 2.0     # a file 10 rounds old outranks a fresh one
_SIZE_WEIGHT = 1.0        # larger files are better unload candidates
_IMAGE_COST_CHARS = 8_000 # a retained image is treated as ~2k tokens


@dataclass
class PinnedFile:
    """A file (or artefact source) currently pinned into the prompt."""

    name: str
    size_chars: int
    age_rounds: int = 0
    pinned: bool = False


@dataclass
class ImageRef:
    """An image retained somewhere in the history."""

    name: str
    age_rounds: int = 0
    cost_chars: int = _IMAGE_COST_CHARS


@dataclass
class HistoryMessage:
    """Minimal, compressor-facing view of one history message."""

    index: int
    role: str
    size_chars: int
    is_tool_structure: bool = False


@dataclass
class CompressionAction:
    """One concrete step the caller must apply."""

    kind: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionPlan:
    """Result of planning. The caller applies `actions` then rebuilds prompt."""

    triggered: bool
    original_chars: int
    projected_chars: int
    target_chars: int
    threshold_chars: int
    actions: List[CompressionAction] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    @property
    def original_tokens_estimate(self) -> int:
        return int(self.original_chars / _CHARS_PER_TOKEN)

    @property
    def projected_tokens_estimate(self) -> int:
        return int(self.projected_chars / _CHARS_PER_TOKEN)


@dataclass
class AppliedCompression:
    """A plan resolved into concrete keep/drop sets the caller can enact."""

    kept_files: List[PinnedFile] = field(default_factory=list)
    dropped_file_names: List[str] = field(default_factory=list)
    kept_images: List[ImageRef] = field(default_factory=list)
    dropped_image_names: List[str] = field(default_factory=list)
    folded_message_indices: List[int] = field(default_factory=list)
    fold_note_insert_before: Optional[int] = None
    projected_chars: int = 0


class ContextCompressor:
    """Pure-heuristic planner that shrinks an over-budget agent context."""

    def __init__(
        self,
        *,
        context_threshold_chars: int = DEFAULT_CONTEXT_THRESHOLD_CHARS,
        target_chars: int = DEFAULT_TARGET_CHARS,
        max_images: int = DEFAULT_MAX_IMAGES,
        keep_recent_files: int = DEFAULT_KEEP_RECENT_FILES,
        keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES,
        max_file_unload_ratio: float = DEFAULT_MAX_FILE_UNLOAD_RATIO,
    ) -> None:
        self.context_threshold_chars = max(1_000, int(context_threshold_chars))
        self.target_chars = max(500, min(int(target_chars), self.context_threshold_chars - 1))
        self.max_images = max(1, min(int(max_images), 4))
        self.keep_recent_files = max(0, int(keep_recent_files))
        self.keep_recent_messages = max(1, int(keep_recent_messages))
        self.max_file_unload_ratio = min(1.0, max(0.0, float(max_file_unload_ratio)))

    # ── public API ─────────────────────────────────────────────────────────

    def plan(
        self,
        pinned_files: Sequence[PinnedFile],
        images: Sequence[ImageRef],
        history: Sequence[HistoryMessage],
        *,
        fixed_overhead_chars: int = 0,
    ) -> CompressionPlan:
        files = list(pinned_files)
        imgs = list(images)
        msgs = list(history)

        base_chars = fixed_overhead_chars + sum(f.size_chars for f in files)
        history_chars = sum(m.size_chars for m in msgs)
        image_chars = sum(img.cost_chars for img in imgs)
        original = base_chars + history_chars + image_chars

        plan = CompressionPlan(
            triggered=original > self.context_threshold_chars,
            original_chars=original,
            projected_chars=original,
            target_chars=self.target_chars,
            threshold_chars=self.context_threshold_chars,
        )
        if not plan.triggered:
            plan.rationale.append(
                f"Under threshold ({original} <= {self.context_threshold_chars} chars); no action."
            )
            return plan

        plan.rationale.append(
            f"Over threshold ({original} > {self.context_threshold_chars}); planning compression."
        )

        projected = original

        # ── Lever C first (cheapest, always safe): cap images ─────────────
        projected += self._plan_image_cap(imgs, plan)

        # ── Weigh the two remaining levers (files vs history) ─────────────
        overflow_after_images = max(0, projected - self.target_chars)

        file_budget = int(overflow_after_images * self.max_file_unload_ratio)
        if file_budget < overflow_after_images and not self._history_has_compressible_span(msgs):
            file_budget = overflow_after_images

        # ── Lever A: unload stale/large files (recency-weighted, budgeted) ─
        projected += self._plan_file_unloads(files, plan, projected, budget=file_budget)

        # ── Lever B: compress history for whatever remains over target ────
        if projected > self.target_chars:
            projected += self._plan_history_compression(msgs, plan, projected)

        plan.projected_chars = max(0, projected)
        return plan

    # ── shared predicates ──────────────────────────────────────────────────

    def _history_has_compressible_span(self, messages: Sequence[HistoryMessage]) -> bool:
        msgs = list(messages)
        if not msgs:
            return False
        cutoff = max(0, len(msgs) - self.keep_recent_messages)
        start = 0
        while start < cutoff and msgs[start].role == "system":
            start += 1
        for m in msgs[start:cutoff]:
            if m.is_tool_structure:
                return False
            return m.size_chars > 0
        return False

    # ── Lever C: images ────────────────────────────────────────────────────

    def _plan_image_cap(
        self, images: List[ImageRef], plan: CompressionPlan,
    ) -> int:
        """Drop every image beyond the cap, OLDEST first. Negative delta."""
        if len(images) <= self.max_images:
            return 0
        # Keep the NEWEST `max_images`. Age 0 = newest, so sort DESCENDING by
        # age: the FIRST entries are the biggest ages (oldest) → dropped.
        ordered_oldest_first = sorted(
            images, key=lambda i: (i.age_rounds, i.name), reverse=True
        )
        to_drop = ordered_oldest_first[: len(images) - self.max_images]
        freed = sum(img.cost_chars for img in to_drop)
        plan.actions.append(
            CompressionAction(
                kind="drop_image",
                detail={
                    "keep": self.max_images,
                    "drop_names": [img.name for img in to_drop],
                    "freed_chars": freed,
                },
            )
        )
        plan.rationale.append(
            f"Image cap: {len(images)} → {self.max_images} (free {freed} chars, "
            f"dropping oldest {[i.name for i in to_drop]})."
        )
        return -freed

    # ── Lever A: file unloading ───────────────────────────────────────────

    def _plan_file_unloads(
        self,
        files: List[PinnedFile],
        plan: CompressionPlan,
        current: int,
        *,
        budget: Optional[int] = None,
    ) -> int:
        """Unload stale/large files until the reclaim `budget` is spent."""
        # Never touch explicitly pinned files.
        candidates = [f for f in files if not f.pinned]

        # Recency guard: keep the newest `keep_recent_files`, BUT never protect
        # the LAST remaining candidate — otherwise a lone oversized file could
        # never be evicted (silent no-op). Protect at most len(candidates)-1.
        if self.keep_recent_files and len(candidates) > 1:
            protected_count = min(self.keep_recent_files, len(candidates) - 1)
            newest = {
                f.name
                for f in sorted(candidates, key=lambda f: (f.age_rounds, f.name))[
                    : protected_count
                ]
            }
            candidates = [f for f in candidates if f.name not in newest]

        if not candidates:
            return 0

        def _score(f: PinnedFile) -> float:
            return (f.age_rounds * _RECENCY_WEIGHT) + (
                (f.size_chars / 1_000.0) * _SIZE_WEIGHT
            )

        ordered = sorted(candidates, key=_score, reverse=True)

        reclaim_goal = budget if budget is not None else max(0, current - self.target_chars)
        if reclaim_goal <= 0:
            return 0

        running = current
        freed_total = 0
        to_unload: List[str] = []
        for f in ordered:
            if freed_total >= reclaim_goal:
                break
            if running <= self.target_chars:
                break
            to_unload.append(f.name)
            running -= f.size_chars
            freed_total += f.size_chars

        if to_unload:
            budget_note = (
                f", within a {reclaim_goal}-char file lever budget"
                if budget is not None
                else ""
            )
            plan.actions.append(
                CompressionAction(
                    kind="unload_file",
                    detail={"names": to_unload, "freed_chars": freed_total},
                )
            )
            plan.rationale.append(
                f"File unload: {len(to_unload)} stale/large file(s) freed "
                f"{freed_total} chars{budget_note} (recency-weighted; kept "
                f"{self.keep_recent_files} newest + all pinned)."
            )
        return -freed_total

    # ── Lever B: history compression ──────────────────────────────────────

    def _plan_history_compression(
        self,
        messages: List[HistoryMessage],
        plan: CompressionPlan,
        current: int,
    ) -> int:
        """Collapse the oldest compressible message span into a short note."""
        if not messages:
            return 0

        cutoff = max(0, len(messages) - self.keep_recent_messages)

        start = 0
        while start < cutoff and messages[start].role == "system":
            start += 1

        span: List[HistoryMessage] = []
        span_chars = 0
        for m in messages[start:cutoff]:
            if m.is_tool_structure:
                break
            span.append(m)
            span_chars += m.size_chars

        if not span:
            return 0

        note_chars = min(2_000, max(200, span_chars // 10))
        freed = span_chars - note_chars
        if freed <= 0:
            return 0

        plan.actions.append(
            CompressionAction(
                kind="compress_span",
                detail={
                    "start_index": span[0].index,
                    "end_index": span[-1].index,
                    "message_count": len(span),
                    "original_chars": span_chars,
                    "note_chars": note_chars,
                    "freed_chars": freed,
                },
            )
        )
        plan.rationale.append(
            f"History span: messages [{span[0].index}..{span[-1].index}] "
            f"({len(span)} msgs, {span_chars} chars) → note ({note_chars} chars), "
            f"freeing {freed} chars. Last {self.keep_recent_messages} messages and "
            "all tool structure preserved."
        )
        return -freed


def estimate_overhead_chars(prompt_parts: Sequence[str]) -> int:
    """Convenience: sum the char length of fixed prompt scaffolding strings."""
    return sum(len(p or "") for p in prompt_parts)


def apply_plan(
    plan: CompressionPlan,
    pinned_files: Sequence[PinnedFile],
    images: Sequence[ImageRef],
) -> AppliedCompression:
    """Resolve a plan's actions into concrete keep/drop sets (pure)."""
    kept_files = list(pinned_files)
    dropped_file_names: List[str] = []
    kept_images = list(images)
    dropped_image_names: List[str] = []
    folded: List[int] = []
    insert_before: Optional[int] = None

    for action in plan.actions:
        if action.kind == "unload_file":
            names = set(action.detail.get("names") or [])
            dropped_file_names.extend(sorted(names))
            kept_files = [f for f in kept_files if f.name not in names]
        elif action.kind == "drop_image":
            names = set(action.detail.get("drop_names") or [])
            dropped_image_names.extend(sorted(names))
            kept_images = [img for img in kept_images if img.name not in names]
        elif action.kind == "compress_span":
            start = action.detail.get("start_index")
            end = action.detail.get("end_index")
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                folded.extend(range(start, end + 1))
                if insert_before is None:
                    insert_before = start

    return AppliedCompression(
        kept_files=kept_files,
        dropped_file_names=dropped_file_names,
        kept_images=kept_images,
        dropped_image_names=dropped_image_names,
        folded_message_indices=sorted(folded),
        fold_note_insert_before=insert_before,
        projected_chars=plan.projected_chars,
    )


__all__ = [
    "ContextCompressor",
    "CompressionPlan",
    "CompressionAction",
    "AppliedCompression",
    "apply_plan",
    "PinnedFile",
    "ImageRef",
    "HistoryMessage",
    "estimate_overhead_chars",
    "DEFAULT_CONTEXT_THRESHOLD_CHARS",
    "DEFAULT_TARGET_CHARS",
    "DEFAULT_MAX_IMAGES",
    "DEFAULT_KEEP_RECENT_FILES",
    "DEFAULT_KEEP_RECENT_MESSAGES",
    "DEFAULT_MAX_FILE_UNLOAD_RATIO",
]