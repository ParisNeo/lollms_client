# lollms_discussion/_context_sanitizer.py
# Pure-functional execution layer for Context Sanitization and Tag Hygiene,
# plus the Context Budget Guard (segment accounting, priority eviction, and
# diagnostic telemetry used by `.lollms_code/_debug_dumps/startup_context.log`).

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence

_ARTIFACT_PATTERN = re.compile(
    r'<art(?:ifact|efact)\s+([^>]*)>(.*?)</art(?:ifact|efact)>',
    re.DOTALL | re.IGNORECASE
)

_SKILL_PATTERN = re.compile(
    r'<skill\s+([^>]*)>(.*?)</skill>',
    re.DOTALL | re.IGNORECASE
)

_NOTE_PATTERN = re.compile(
    r'<note\s+([^>]*)>(.*?)</note>',
    re.DOTALL | re.IGNORECASE
)

_TOOL_PATTERN = re.compile(
    r'<tool>(.*?)</tool>',
    re.DOTALL | re.IGNORECASE
)

_PROCESSING_PATTERN = re.compile(
    r'<processing[^>]*>.*?(?:</processing>|$)',
    re.DOTALL | re.IGNORECASE
)

_ORPHAN_PROCESSING_PATTERN = re.compile(
    r'<processing[^>]*>.*$',
    re.DOTALL | re.IGNORECASE
)

_ARTEFACT_IMAGE_PATTERN = re.compile(
    r'<artefact_image\s+[^/]*/>',
    re.IGNORECASE
)

_ACTION_RESULT_PATTERN = re.compile(
    r'<action_result[^>]*>.*?</action_result>',
    re.DOTALL | re.IGNORECASE
)


def _extract_attrs(attr_str: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}


def scrub_processing_and_status_blocks(text: str) -> str:
    """
    Removes system-generated execution logs (<processing> blocks, status comments, thoughts)
    without touching the LLM's conversational text or functional action tags.
    """
    if not text:
        return ""
    text = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought\b[^>]*>.*?(?:</thought>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<effort\b[^>]*(?:/>|>.*?</effort>)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = _PROCESSING_PATTERN.sub('', text)
    text = _ORPHAN_PROCESSING_PATTERN.sub('', text)
    text = re.sub(r'<!--\s*status:[^>]*-->', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</processing>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<lollms_artifact[^/]*/>', '', text, flags=re.IGNORECASE)
    text = _ARTEFACT_IMAGE_PATTERN.sub('', text)
    return text.strip()


def sanitize_context_for_llm(text: str) -> str:
    """
    Sanitizes older messages by stripping processing logs while preserving functional tags.
    """
    if not text:
        return ""
    return scrub_processing_and_status_blocks(text)


def build_anti_mimicry_directives() -> str:
    """
    Returns concise, positive directives for tool and artifact emission,
    including the placeholder-free history contract.
    """
    return (
        "=== ACTION & OUTPUT INTEGRITY PROTOCOL (MANDATORY) ===\n"
        "1. **EMIT REAL FUNCTIONAL TAGS**: To create or edit files use `<artifact name=\"...\">` with complete content or SEARCH/REPLACE blocks. To invoke tools use `<tool>`. To create skills use `<skill title=\"...\">`.\n"
        "2. **PROSE IS NOT ACTION**: Simply saying 'I will write the code' or 'I created the file' in natural text DOES NOT create or modify files. You MUST output the actual XML tags.\n To create artifacts use `<artifact name=\"...\">`, to create skills use `<skill title=\"...\">`, to save notes use `<note title=\"...\">`, and to invoke tools use `<tool>`. Do NOT simulate results or mimic past system messages.\n"
        "3. **DIRECT ENGAGEMENT**: Respond naturally and directly to the user's request. Do not provide meta-commentary about turns or system statuses.\n"
        "4. **NEVER REPRODUCE SYSTEM TEXT**: Bracketed system markers (e.g. `[SYSTEM: ...]`, `[COMPLETED ACTIONS DIGEST]`, `[SYSTEM NOTIFICATION]`), self-closing stub tags (e.g. `<artifact ... status=\"saved\" />`, `<tool_called ... />`), and digest narratives are generated exclusively by the system. NEVER emit them yourself. If you want to reference a past action, describe it in your own words or re-execute it with a real tag.\n"
        "=== END ACTION PROTOCOL ==="
    )


# ---------------------------------------------------------------------------
# Context Budget Guard
# ---------------------------------------------------------------------------
#
# The startup context is a multi-segment payload (stable prefix, skills list,
# tool descriptions, user profile, workspace tree, loaded files, memories).
# Before the guard, every segment was rendered unconditionally; on a model
# whose prompt already enumerates dozens of tools, the startup payload alone
# compressed the conversation budget. The guard caps the payload at
# `budget_tokens`, reserves room for the model's output, and evicts segments
# by a fixed retention ladder. Token accounting is persisted to
# `.lollms_code/context_state.db` so a per-segment drift (e.g. a toolset
# description getting longer) is observable and demotable before the first turn.


class ContextBudgetExceeded(RuntimeError):
    """Raised when the non-evictable prefix alone exceeds the budget.

    The guard MUST fail loudly: silently truncating the stable system prefix
    or the latest user turn would produce a prompt with corrupted directives,
    which is strictly worse than aborting the turn with an explicit error.
    """


# Segment priorities: HIGHER value = retained LONGER. The ladder is fixed and
# documented once, here — every caller (system prompt assembler, discussion
# chat mixin, personality chat) references the same numbers.
PRIORITY_STABLE_PREFIX = 1000
PRIORITY_CURRENT_USER_TURN = 900
PRIORITY_ACTIVE_SKILLS = 800
PRIORITY_USER_PROFILE = 700
PRIORITY_LOADED_FILES = 600
PRIORITY_TOOL_DESCRIPTIONS = 500
PRIORITY_SKILL_NAMES = 400
PRIORITY_STARTUP_ARTIFACTS = 300

# Segment IDs agreed across emitters so DB rows are joinable at query time.
SEG_ID_STABLE_PREFIX = "stable_prefix"
SEG_ID_USER_TURN = "current_user_turn"
SEG_ID_ACTIVE_SKILLS = "active_skills"
SEG_ID_USER_PROFILE = "user_profile"
SEG_ID_LOADED_FILES = "loaded_files"
SEG_ID_TOOL_DESCRIPTIONS = "tool_descriptions"
SEG_ID_SKILL_NAMES = "skill_names"
SEG_ID_STARTUP_ARTIFACTS = "startup_artifacts"

# Action vocabulary produced by `enforce_budget`. The DB stores the exact
# literal, and the startup log emitter tags each rendered segment with it.
ACTION_KEEP = "keep"
ACTION_DROP = "drop"
ACTION_TRUNCATE_MIDDLE = "truncate_middle"
ACTION_SUMMARIZE_NAMES = "summarize_to_names"
ACTION_POINTER = "replace_with_pointer"

# Pointer hint rendered in place of an evicted tool block. The canonical
# introspection tool is `tool_describe_tools`, which every personality mounts.
TOOL_POINTER_HINT = "call tool_describe_tools for schema"

# Fallback chars-per-token when the caller provides no tokenizer. Matches the
# fast heuristic used elsewhere in the codebase (`len(text) // 4`).
_DEFAULT_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str, counter: Optional[Any] = None) -> int:
    """Returns a token estimate for `text`.

    Order of preference:
      1. `counter(text)` when a callable tokenizer is supplied.
      2. Deterministic `len(text) // 4` fallback for tests / tokenizer-less
         backends.

    The fallback is deliberately deterministic so the budget-guard tests can
    assert exact numbers across runs.
    """
    if not text:
        return 0
    if callable(counter):
        try:
            value = counter(text)
            if isinstance(value, int) and value >= 0:
                return value
        except Exception:
            pass
    return len(text) // _DEFAULT_CHARS_PER_TOKEN


def _content_hash(content: str) -> str:
    """Stable short hash used to correlate two runs of the same segment."""
    return hashlib.sha1((content or "").encode("utf-8")).hexdigest()[:16]


def _truncate_middle(text: str, budget_tokens: int, counter: Optional[Any] = None) -> str:
    """Cuts `text` down to <= `budget_tokens` by dropping its middle section.

    Keeps the first and last thirds of the character budget so both the
    opening (headers) and closing (footers / sentinels) remain legible.
    Returns "" when `budget_tokens` <= 0.
    """
    if not text or budget_tokens <= 0:
        return ""
    target_chars = budget_tokens * _DEFAULT_CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return text
    head_chars = target_chars // 3
    tail_chars = target_chars // 3
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n[... {omitted} chars omitted to fit the context budget ...]\n"
        + text[len(text) - tail_chars:]
    )


def _render_names_only(text: str) -> str:
    """Collapses a rendered tool/skill block into a name-only listing.

    We do NOT try to parse the formatted prompt back into structure; the
    guard is fed by the caller, which is expected to pass a segment whose
    content is a newline-delimited list of `name: description` entries.
    Anything that looks like `• **tool_x(...)**: desc` or `- tool_x: desc`
    keeps its identifier and drops the tail. Unknown lines are preserved
    untouched so we never silently lose a hand-authored directive.
    """
    if not text:
        return ""
    out_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        # Pattern 1: "• **`tool_x(a: str)`**: description"
        m = re.match(r'^[•\-\*]?\s*\*\*`([^`(]+)\(', stripped)
        if m:
            out_lines.append(m.group(1).strip())
            continue
        # Pattern 2: "• **`tool_x`**: description"
        m = re.match(r'^[•\-\*]?\s*\*\*`([^`]+)`\*\*', stripped)
        if m:
            out_lines.append(m.group(1).strip())
            continue
        # Pattern 3: "- tool_x: description"
        m = re.match(r'^[\-\*]\s*`?([A-Za-z_][\w:]+)`?\s*:', stripped)
        if m:
            out_lines.append(m.group(1).strip())
            continue
        # Pattern 4: plain comma-separated names line ("`a`, `b`, `c`")
        tokens = re.findall(r'`([^`]+)`', stripped)
        if tokens and stripped.count("`") == len(tokens) * 2:
            out_lines.extend(t.strip() for t in tokens)
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


class ContextSegment:
    """Immutable descriptor for one addressable chunk of the context payload.

    `evictable=False` marks a HARD invariant (stable prefix, current user
    turn). The guard never touches non-evictable segments; if their combined
    cost exceeds the budget, `enforce_budget` raises `ContextBudgetExceeded`.
    """

    __slots__ = ("id", "priority", "content", "evictable")

    def __init__(self, id: str, priority: int, content: str, evictable: bool = True):
        if not id:
            raise ValueError("ContextSegment requires a non-empty id.")
        if not isinstance(priority, int):
            raise TypeError(f"ContextSegment priority must be int, got {type(priority).__name__}.")
        self.id = id
        self.priority = priority
        self.content = content or ""
        self.evictable = bool(evictable)

    def __repr__(self) -> str:
        return (
            f"ContextSegment(id={self.id!r}, priority={self.priority}, "
            f"evictable={self.evictable}, chars={len(self.content)})"
        )


class SanitizedSegment:
    """Result of applying the guard to one segment."""

    __slots__ = ("id", "priority", "action", "content", "tokens", "evictable")

    def __init__(self, id: str, priority: int, action: str, content: str,
                 tokens: int, evictable: bool):
        self.id = id
        self.priority = priority
        self.action = action
        self.content = content
        self.tokens = tokens
        self.evictable = evictable

    def __repr__(self) -> str:
        return (
            f"SanitizedSegment(id={self.id!r}, action={self.action!r}, "
            f"tokens={self.tokens}, evictable={self.evictable})"
        )


class SanitizedContext:
    """Result of `enforce_budget`."""

    __slots__ = ("segments", "total_tokens", "budget", "reserve", "status")

    def __init__(self, segments: Sequence[SanitizedSegment], total_tokens: int,
                 budget: int, reserve: int):
        self.segments = list(segments)
        self.total_tokens = total_tokens
        self.budget = budget
        self.reserve = reserve
        self.status = "OK" if total_tokens <= (budget - reserve) else "EVICTED"

    @property
    def evicted_ids(self) -> List[str]:
        """IDs of segments whose content was modified or dropped."""
        return [
            s.id for s in self.segments
            if s.action not in (ACTION_KEEP,)
        ]

    def render(self, separator: str = "\n\n") -> str:
        """Assembles the surviving segments into a single prompt string.

        Empty segments are omitted so the rendered output never accumulates
        blank `separator` runs.
        """
        parts = [s.content for s in self.segments if s.content.strip()]
        return separator.join(parts)


def _allocate_tool_block(
    segment: ContextSegment,
    available_tokens: int,
    counter: Optional[Any],
) -> SanitizedSegment:
    """Resolves a tool-description segment against its allocation.

    Rules:
      * Fits entirely → keep.
      * Oversized but non-empty → demote to `name + pointer`. Tools remain
        discoverable via `tool_describe_tools`; only the parameter schemas
        are dropped. This is non-negotiable: a missing tool name hides the
        capability from the LLM entirely.
      * Empty segment → drop (no content to keep).
    """
    tokens = _estimate_tokens(segment.content, counter)
    if tokens <= available_tokens:
        return SanitizedSegment(
            id=segment.id, priority=segment.priority, action=ACTION_KEEP,
            content=segment.content, tokens=tokens, evictable=segment.evictable,
        )
    if not segment.content.strip():
        return SanitizedSegment(
            id=segment.id, priority=segment.priority, action=ACTION_DROP,
            content="", tokens=0, evictable=segment.evictable,
        )
    demoted = (
        "=== TOOL NAMES (schemas demoted to fit the context budget) ===\n"
        f"Call `tool_describe_tools(tool_name='<name>')` for full schema. {TOOL_POINTER_HINT}.\n"
        + _render_names_only(segment.content)
    )
    return SanitizedSegment(
        id=segment.id, priority=segment.priority, action=ACTION_SUMMARIZE_NAMES,
        content=demoted, tokens=_estimate_tokens(demoted, counter),
        evictable=segment.evictable,
    )


def enforce_budget(
    segments: Sequence[ContextSegment],
    budget_tokens: int,
    reserve_output: int = 0,
    token_counter: Optional[Any] = None,
) -> SanitizedContext:
    """Applies the retention ladder and returns the surviving context.

    Algorithm:
      1. Split segments into non-evictable (HARD) and evictable.
      2. Sum the HARD cost. If it alone exceeds `budget_tokens`, raise
         `ContextBudgetExceeded` — the caller must fix the prefix, not the
         guard.
      3. Walk evictable segments in DESCENDING priority (highest first).
      4. Accumulate their cost until the remaining budget is exhausted, then
         apply the per-segment fallback (tool block → names-only, generic
         segment → middle truncation, zero-budget segment → drop).

    Returns a `SanitizedContext` whose `.status` is `"OK"` when the total
    (including reserve) fits, `"EVICTED"` otherwise. Failure to fit even
    after every eviction is still reported — the caller decides whether to
    raise or to log, matching the existing fail-soft convention of the
    sanitizer layer.
    """
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be > 0, got {budget_tokens}.")
    if reserve_output < 0:
        raise ValueError(f"reserve_output must be >= 0, got {reserve_output}.")

    available = budget_tokens - reserve_output
    if available <= 0:
        raise ValueError(
            f"reserve_output ({reserve_output}) must be < budget_tokens ({budget_tokens})."
        )

    hard: List[ContextSegment] = [s for s in segments if not s.evictable]
    soft: List[ContextSegment] = [s for s in segments if s.evictable]

    hard_cost = sum(_estimate_tokens(s.content, token_counter) for s in hard)
    if hard_cost > available:
        non_evictable_ids = ", ".join(s.id for s in hard)
        raise ContextBudgetExceeded(
            f"Non-evictable segments ({non_evictable_ids}) consume {hard_cost} tokens, "
            f"exceeding the available budget of {available} "
            f"(budget={budget_tokens}, reserve={reserve_output}). "
            f"Shorten the stable prefix or raise the budget."
        )

    remaining = available - hard_cost

    ordered = sorted(soft, key=lambda s: (-s.priority, s.id))
    results: Dict[str, SanitizedSegment] = {}
    for segment in ordered:
        cost = _estimate_tokens(segment.content, token_counter)
        if cost <= remaining:
            results[segment.id] = SanitizedSegment(
                id=segment.id, priority=segment.priority, action=ACTION_KEEP,
                content=segment.content, tokens=cost, evictable=True,
            )
            remaining -= cost
            continue

        if segment.id == SEG_ID_TOOL_DESCRIPTIONS:
            resolved = _allocate_tool_block(segment, remaining, token_counter)
            results[segment.id] = resolved
            remaining = max(0, remaining - resolved.tokens)
            continue

        if remaining <= 0:
            results[segment.id] = SanitizedSegment(
                id=segment.id, priority=segment.priority, action=ACTION_DROP,
                content="", tokens=0, evictable=True,
            )
            continue

        truncated = _truncate_middle(segment.content, remaining, token_counter)
        results[segment.id] = SanitizedSegment(
            id=segment.id, priority=segment.priority, action=ACTION_TRUNCATE_MIDDLE,
            content=truncated, tokens=_estimate_tokens(truncated, token_counter),
            evictable=True,
        )
        remaining = 0

    for segment in hard:
        results[segment.id] = SanitizedSegment(
            id=segment.id, priority=segment.priority, action=ACTION_KEEP,
            content=segment.content,
            tokens=_estimate_tokens(segment.content, token_counter),
            evictable=False,
        )

    # Preserve the caller's original ordering so the rendered prompt keeps
    # its intended section sequence (the ladder only decides WHAT survives).
    ordered_results = [results[s.id] for s in segments if s.id in results]
    total = sum(s.tokens for s in ordered_results)
    return SanitizedContext(
        segments=ordered_results,
        total_tokens=total,
        budget=budget_tokens,
        reserve=reserve_output,
    )


class ContextBudgetStore:
    """Persists per-run and per-segment token accounting to a SQLite file.

    Schema:
        runs(run_id TEXT PK, budget INT, reserve INT, total_tokens INT,
             status TEXT, timestamp REAL)
        segments(run_id TEXT, segment_id TEXT, priority INT, token_count INT,
                 action TEXT, content_hash TEXT, timestamp REAL)
        PRIMARY KEY (run_id, segment_id)
    """

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            budget INTEGER NOT NULL,
            reserve INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            status TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS segments (
            run_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            priority INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            action TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            timestamp REAL NOT NULL,
            PRIMARY KEY (run_id, segment_id)
        )
        """,
    )

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for stmt in self._SCHEMA:
                conn.execute(stmt)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(self, run_id: str, sanitized: SanitizedContext) -> None:
        """Upserts the run header + one row per segment in one transaction.

        Re-using a `run_id` overwrites the prior rows for that run so tests
        can call `record` repeatedly with the same run_id to prove
        idempotency without leaving duplicates.
        """
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "REPLACE INTO runs (run_id, budget, reserve, total_tokens, status, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, sanitized.budget, sanitized.reserve,
                 sanitized.total_tokens, sanitized.status, now),
            )
            for seg in sanitized.segments:
                conn.execute(
                    "REPLACE INTO segments "
                    "(run_id, segment_id, priority, token_count, action, content_hash, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, seg.id, seg.priority, seg.tokens, seg.action,
                     _content_hash(seg.content), now),
                )
            conn.commit()

    def latest_run_for(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Returns the most recent segment row for `segment_id`, or None."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT run_id, timestamp, token_count, action "
                "FROM segments WHERE segment_id = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (segment_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"run_id": row[0], "timestamp": row[1],
                "token_count": row[2], "action": row[3]}

    def growth_violations(
        self,
        sanitized: SanitizedContext,
        threshold_ratio: float = 0.15,
    ) -> List[str]:
        """Returns segment IDs whose token count grew beyond `threshold_ratio`
        compared to their most recent stored run.

        Used by the startup path to auto-demote a toolset whose description
        ballooned between sessions BEFORE the first turn is sent to the model.
        """
        if threshold_ratio <= 0:
            return []
        violations: List[str] = []
        for seg in sanitized.segments:
            if seg.id in (SEG_ID_STABLE_PREFIX, SEG_ID_USER_TURN):
                continue  # HARD segments are validated by enforce_budget itself.
            prior = self.latest_run_for(seg.id)
            if prior is None or prior["token_count"] <= 0:
                continue
            growth = (seg.tokens - prior["token_count"]) / prior["token_count"]
            if growth > threshold_ratio:
                violations.append(seg.id)
        return violations


def render_budget_header(sanitized: SanitizedContext) -> str:
    """Renders the mandatory `BUDGET:` header for startup_context.log."""
    return (
        f"BUDGET: total={sanitized.total_tokens}/{sanitized.budget} "
        f"reserve={sanitized.reserve} status={sanitized.status}"
    )


def render_segment_tags(sanitized: SanitizedContext) -> List[str]:
    """Renders one tag line per segment for the log's `ACTIVE TOOLS` section.

    Format: `[tokens=<n> priority=<p> action=<a>]` — matches the format the
    Librarian directive pinned for `.lollms_code/_debug_dumps/startup_context.log`.
    """
    return [
        f"[tokens={seg.tokens} priority={seg.priority} action={seg.action}] {seg.id}"
        for seg in sanitized.segments
    ]