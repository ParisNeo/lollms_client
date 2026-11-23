
"""
Mesh bot runtime (phase 3b).

This module is what the ``MeshSupervisor`` actually launches — one Python
process per bot, wired to a shared mesh bus, serving its ``spec.inbound_routes``
by pumping each incoming bus message through ``personality.chat()`` and
publishing the reply back on the bus.

It intentionally does NOT use ``BotDaemon``. BotDaemon is a "chat on telegram /
discord / etc" host — chain of async connection bindings contending for one
personality — utterly inappropriate here. A mesh bot is a *service*:

    while alive:
        if got bus message on a subscribed topic:
            run it through personality.chat()  (blocking, in a worker thread)
            reply on <topic>.reply (or correlation_id)

Design choices, justified
-------------------------
* **Single-threaded message pump, work offloaded to a thread pool.** The
  pump must stay responsive to heartbeats and shutdown; a chat() call is a
  multi-second LLM round trip, so each message is submitted to a
  ``ThreadPoolExecutor`` with ``max_workers=1`` — one bot answers one
  message at a time. Concurrent chat() on the same LollmsPersonality
  corrupts its internal state, so single-worker is a correctness constraint,
  not a perf one.

* **Graceful shutdown via SIGTERM/SIGINT.** The pump checks a stop-event on
  every bus poll cycle (~250ms). On SIGTERM, we cancel the in-flight chat
  if possible (via ``personality.cancel_generation()``), then exit with 0.
  This gives ``MeshSupervisor.stop()`` a fast, clean transition.

* **Personality built once, reused for every message.** Building it costs
  seconds (handbag load, tool/skill discovery, RAG connecting, LLM warmup).
  We build on startup, fail loudly if that fails, and reuse it — a bot is
  a long-lived agent, not a stateless function.

* **Per-message disk volatility is bounded.** Each chat call's disk sync
  is done inside ``personality.chat``; we do NOT save the personality's
  whole history per message (that's what the ``BotDaemon`` does and why it's
  ~1MB/s). Instead we keep the personality alive and let it manage its own
  working set. This bounds disk churn at mesh scale.

* **Profile override.** The ``MeshSupervisor`` launches ``mesh-bot`` as a
  module with ``--bot-id``; this module reads the BotSpec from the mesh
  registry, and when ``spec.llm_profile`` is set, builds the LollmsClient
  with that profile via ``get_client_from_env(llm_profile=...)``. This is
  how a mesh bot on another model coexists with the "default" ones.

CLI
---
    python -m lollms_client.apps.lollms_bot.mesh.runtime \
        --mesh-root <parent> --mesh-id <id> --bot-id <bot_id>

Exit codes:
    0 — clean shutdown (SIGTERM/SIGINT or empty subscriptions)
    1 — fatal runtime error (personality build failed, mesh missing, ...)
    2 — invalid arguments (argparse default)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_bot.mesh.bus import BusMessage, MessageBus
from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.types import BotSpec, BotStatus, MeshLayout

# LollmsDiscussion is the ONLY container that drives a LollmsPersonality
# through its full agentic loop. A mesh bot MUST hold one per bot_id, shared
# across all incoming messages, so capabilities (tools, artefacts, memory,
# sub-personas) actually fire. Calling personality.chat(prompt) directly
# silently drops all of this — that was the pre-fix runtime bug.
try:  # pragma: no cover - imported lazily so module import stays cheap
    from lollms_client.lollms_discussion import LollmsDiscussion
    _DISCUSSION_OK = True
except Exception:
    _DISCUSSION_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPLY_TOPIC_SUFFIX = ".reply"           # where a bot publishes its answer
_ERROR_TOPIC_SUFFIX = ".error"           # where a bot publishes a failure
_HEARTBEAT_INTERVAL_SECONDS = 3.0        # registry heartbeat cadence
_PUMP_POLL_INTERVAL_SECONDS = 0.25       # bus poll cadence
_MAX_MESSAGE_BODY_CHARS = 32_000         # refuse obviously oversized tasks
_STOP_GRACE_SECONDS = 15.0               # time the pump allows chat() to cancel


# ---------------------------------------------------------------------------
# Logging helper (uniform with the supervisor's expectations)
# ---------------------------------------------------------------------------

def _log(tag: str, msg: str) -> None:
    print(f"[mesh-bot {tag}] {msg}", flush=True)


def _log_error(tag: str, msg: str, exc: Optional[BaseException] = None) -> None:
    print(f"[mesh-bot {tag}] ERROR: {msg}", file=sys.stderr, flush=True)
    if exc is not None:
        traceback.print_exc(file=sys.stderr)


# ---------------------------------------------------------------------------
# Bot identity: read BotSpec from the registry (registered by supervisor
# before the child process launched, so it's already on disk).
# ---------------------------------------------------------------------------

def _load_bot_spec(layout: MeshLayout, bot_id: str) -> BotSpec:
    """
    Reads the spec that the supervisor placed in the registry before spawning
    us. If missing we fail fast — a bot without a spec is a supervisor bug,
    and silently running would hide it.
    """
    reg = MeshRegistry(layout)
    spec = reg.get_bot_spec(bot_id)
    if spec is None:
        raise RuntimeError(
            f"Bot {bot_id!r} is not registered in mesh {layout.root.name!r}. "
            f"Register it before launching."
        )
    return spec


# ---------------------------------------------------------------------------
# LLM client builder
# ---------------------------------------------------------------------------

def _build_client_with_profile(profile_name: Optional[str]) -> Any:
    """
    Builds the LollmsClient for this bot.

    ``profile_name`` is honored via ``get_client_from_env(llm_profile=...)``
    when supported; when the helper does not accept that kwarg (older
    lollms-client), we fall back to the default profile and note it. The bot
    still runs, but on the default model — this is a graceful degradation
    that lets mesh bots coexist with older client versions.
    """
    from lollms_client.lollms_config_cli_env import get_client_from_env

    kwargs: Dict[str, Any] = dict(
        create_llm=True,
        create_connection=False,  # mesh bots do NOT chat on telegram/discord
        run_wizard_if_fail=False,
    )
    if profile_name and profile_name != "default":
        try:
            client = get_client_from_env(llm_profile=profile_name, **kwargs)
            _log("client", f"LLM profile requested: {profile_name} (applied)")
            return client
        except TypeError:
            # Older lollms-client: no llm_profile kwarg. Fall through.
            _log(
                "client",
                f"llm_profile kwarg unsupported; ignoring requested "
                f"profile {profile_name!r} and using the default.",
            )
        except Exception as e:
            _log_error(
                "client",
                f"Failed to build client with profile {profile_name!r}: {e}. "
                f"Falling back to default.",
            )

    client = get_client_from_env(**kwargs)
    if getattr(client, "llm", None) is None:
        raise RuntimeError(
            "No LLM available. Configure lollms-client first: "
            "python -m lollms_client.lollms_config_cli_env"
        )
    _log(
        "client",
        f"LLM: binding={getattr(client.llm, 'binding_name', '?')} "
        f"model={getattr(client.llm, 'model_name', '?')}",
    )
    return client


# ---------------------------------------------------------------------------
# Personality builder (reuses daemon._finish_personality for parity with the
# single-daemon path — same capability flags, role, and token budget.)
# ---------------------------------------------------------------------------

def _build_personality(client: Any, spec: BotSpec):
    """
    Builds the bot's personality from its ``handbag_path``. Capabilities
    and per-turn token budget come from ``spec.capabilities`` (which the
    hub's "New Bot" dialog sets) — nothing else.

    The handbag must exist; we treat a missing handbag as a fatal startup
    error. There's no "make it on the fly" fallback here — a mesh is an
    explicit-identity system, not a "create me a default bot" daemon.
    """
    from lollms_client.lollms_personality import LollmsPersonality
    from lollms_client.lollms_personality.handbag import Handbag

    if spec.handbag_path:
        handbag_path = Path(spec.handbag_path).expanduser().resolve()
    else:
        raise RuntimeError(
            f"Bot {spec.bot_id!r} has no handbag_path. Set one in the spec."
        )
    if not handbag_path.exists():
        raise RuntimeError(
            f"Handbag path for bot {spec.bot_id!r} does not exist: {handbag_path}"
        )

    # Confirm the folder is actually a handbag (has SOUL.md or handbag.yaml)
    # so a typo in handbag_path fails fast rather than loading an empty folder.
    has_identity = (handbag_path / "SOUL.md").is_file() or (
        handbag_path / "handbag.yaml"
    ).is_file()
    if not has_identity:
        raise RuntimeError(
            f"Folder {handbag_path} is not a handbag "
            f"(no SOUL.md, no handbag.yaml)"
        )
    Handbag(handbag_path)  # validates structure

    personality = LollmsPersonality.from_handbag(
        str(handbag_path), lollms_client=client
    )
    if personality is None:
        raise RuntimeError(f"LollmsPersonality.from_handbag returned None for {spec.bot_id!r}")

    # Apply capabilities from the spec. We deliberately DO NOT reconstruct
    # CapabilityFlags here — its field set is owned by lollms_personality and
    # has changed across versions; constructing it with guessed kwargs is a
    # fatal TypeError waiting to happen (this was exactly the rc=1 crash that
    # burned the restart budget on every start).
    #
    # Instead, we treat ``spec.capabilities`` as a set of OVERRIDES applied
    # attribute-by-attribute, skipping any key the live CapabilityFlags
    # object doesn't actually expose. This is resilient, DRY, and keeps
    # mesh bots aligned with the same flags the single-daemon path uses.
    caps_dict = dict(spec.capabilities or {})

    # Hard safety invariant regardless of spec: a mesh bot MUST NOT spawn
    # sub-agents (that would be an unbounded fork bomb at mesh scale; the
    # mesh leader pattern replaces it with bus messages). Override any spec
    # attempt to the contrary.
    caps_dict.setdefault("enable_sub_agents", False)

    cf = getattr(personality, "capabilities", None)
    if cf is not None:
        applied: Dict[str, Any] = {}
        ignored: List[str] = []
        for key, value in caps_dict.items():
            if hasattr(cf, key):
                try:
                    setattr(cf, key, value)
                    applied[key] = value
                except Exception:
                    ignored.append(key)
            else:
                ignored.append(key)
        if ignored:
            _log(
                "capabilities",
                f"ignored {len(ignored)} unknown flag(s): {', '.join(ignored)}",
            )
        if applied:
            _log(
                "capabilities",
                f"applied {len(applied)} override(s): {', '.join(applied)}",
            )

    # max_tokens_per_turn is a personality attribute, not a capability flag.
    max_tokens = caps_dict.get("max_tokens_per_turn")
    if isinstance(max_tokens, int) and max_tokens > 0:
        try:
            personality.max_tokens_per_turn = max_tokens
        except Exception:
            pass
    return personality


# ---------------------------------------------------------------------------
# Message handling (invoked off the pump thread)
# ---------------------------------------------------------------------------

def _serialize_reply(bot_id: str, topic: str, msg: BusMessage, text: str) -> Dict[str, Any]:
    """
    Envelope a bot publishes back. The contract:
      * ``topic`` prefix ``.reply`` — correlation preserved via
        ``correlation_id`` so ``wait_for_reply`` on the *sender* side works.
      * ``payload.text`` — the reply string.
      * ``payload.in_reply_to_msg_id`` — provenance for the UI graph.
    """
    return {
        "text": text,
        "in_reply_to_msg_id": msg.id,
        "in_reply_to_topic": msg.topic,
        "original_sender": msg.sender,
        "bot_id": bot_id,
    }


def _serialize_error(bot_id: str, msg: BusMessage, error: str) -> Dict[str, Any]:
    return {
        "error": error,
        "in_reply_to_msg_id": msg.id,
        "in_reply_to_topic": msg.topic,
        "original_sender": msg.sender,
        "bot_id": bot_id,
    }


def _build_discussion(client: Any, db_path: Path) -> Any:
    """
    Constructs a persisted ``LollmsDiscussion`` for a mesh bot.

    Why this isn't a one-liner
    --------------------------
    ``LollmsDiscussion`` does not expose a stable, documented constructor
    across lollms-client versions. In particular:

      * ``LollmsDiscussion(db_path=...)``  → TypeError (CoreMixin.__init__
        does not accept ``db_path``). Observed in the wild.
      * ``LollmsDiscussion(lollmsClient=...)`` → accepted in some versions,
        not in others.
      * ``LollmsDiscussion.create(...)``, ``create_new(...)``,
        ``create_at_path(...)`` — factory names have moved around.

    Rather than fight that by guessing another signature (which historically
    burns the supervisor's restart budget on startup), we **introspect the
    real constructor at runtime** via :func:`inspect.signature` and hand it
    ONLY the kwargs it actually accepts. This is the correct, DRY way to
    adapt to whatever the current client exposes.

    Fallback chain, each attempt logged so operators can see what ran:
      1. Introspected keyword constructor (``LollmsDiscussion(**filtered)``).
      2. Class-level factory helpers, in a documented order:
         ``create_at_path`` → ``create_new`` → ``create``.
      3. ``LollmsDiscussion(lollmsClient=client)`` if it accepts *arg only.

    Raises ``RuntimeError`` with the full list of attempts if none work.
    The caller (``main``) marks the bot FAILED and exits cleanly, so the
    supervisor will *not* restart (self-declared FAILED is terminal).
    """
    import inspect

    attempts: List[str] = []

    def _try(label: str, fn):
        try:
            obj = fn()
            if obj is None:
                attempts.append(f"{label}: returned None")
                return None
            attempts.append(f"{label}: OK")
            return obj
        except TypeError as e:
            attempts.append(f"{label}: TypeError({e})")
            return None
        except Exception as e:
            attempts.append(f"{label}: {type(e).__name__}({e})")
            return None

    # ---- Attempt 1: introspect the real constructor -----------------------
    try:
        sig = inspect.signature(LollmsDiscussion.__init__)
        accepted = {
            name for name, p in sig.parameters.items()
            if name != "self"
            and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
    except (TypeError, ValueError):
        accepted = set()

    # Candidate kwargs — only the ones accepted will be sent.
    candidates = {
        "db_path": str(db_path),
        "database_path": str(db_path),
        "path": str(db_path),
        "db_file": str(db_path),
        "lollmsClient": client,          # camelCase — current client
        "lollms_client": client,         # snake_case — older
        "client": client,                # oldest
    }
    filtered = {k: v for k, v in candidates.items() if k in accepted}

    if filtered:
        obj = _try(
            f"LollmsDiscussion({','.join(sorted(filtered))})",
            lambda: LollmsDiscussion(**filtered),
        )
        if obj is not None:
            return obj

    # ---- Attempt 2: class-level factories, in preference order -----------
    factories = [
        ("LollmsDiscussion.create_at_path", "create_at_path", (str(db_path),)),
        ("LollmsDiscussion.create_new",     "create_new",     (str(db_path),)),
        ("LollmsDiscussion.create",         "create",         (str(db_path),)),
    ]
    for label, name, positional in factories:
        fn_attr = getattr(LollmsDiscussion, name, None)
        if fn_attr is None:
            attempts.append(f"{label}: not present")
            continue
        # Try with the client kwarg — camelCase first, then snake_case, then
        # bare positional (some factories only take the path).
        for client_kw in ("lollmsClient", "lollms_client", "client", None):
            if client_kw is None:
                obj = _try(label, lambda f=fn_attr, a=positional: f(*a))
            else:
                obj = _try(
                    f"{label}({client_kw}=...)",
                    lambda f=fn_attr, a=positional, kw=client_kw: f(*a, **{kw: client}),
                )
            if obj is not None:
                return obj

    # ---- Attempt 3: zero-arg constructor + attach client ----------------
    # Some clients keep discussion storage in a global default path and
    # expose the client via ``.lollmsClient``. Try the bare instantiation
    # and set the client attribute directly if it's writable.
    try:
        obj = LollmsDiscussion()
        try:
            setattr(obj, "lollmsClient", client)
        except Exception:
            try:
                setattr(obj, "lollms_client", client)
            except Exception:
                pass
        attempts.append("LollmsDiscussion(): OK")
        return obj
    except Exception as e:
        attempts.append(f"LollmsDiscussion(): {type(e).__name__}({e})")

    raise RuntimeError(
        "Could not construct a LollmsDiscussion for this bot. "
        "Attempts (in order):\n  - "
        + "\n  - ".join(attempts)
        + "\n\nThe mesh bot cannot run without a discussion container. "
        "This usually means the lollms-client version in use has a "
        "different LollmsDiscussion constructor than this build expects. "
        "Inspect it with: python -c \"import inspect; "
        "from lollms_client.lollms_discussion import LollmsDiscussion; "
        "print(inspect.signature(LollmsDiscussion.__init__))\""
    )


# ---------------------------------------------------------------------------
# The bot loop
# ---------------------------------------------------------------------------

class MeshBot:
    """
    One long-lived mesh bot process.

    Lifecycle:
        __init__    captures configuration (mesh layout, bot id, spec, personality, bus)
        run()       pump loop (blocking); returns on stop-event
        stop()      sets stop-event and cancels in-flight chat

    All public methods are safe to call from the main thread. The pump
    itself is the main thread; chat() runs on the executor thread.
    """

    def __init__(
        self,
        layout: MeshLayout,
        bot_id: str,
        spec: BotSpec,
        client: Any,
        personality: Any,
        discussion: Any,
    ):
        self.layout = layout
        self.bot_id = bot_id
        self.spec = spec
        self.client = client
        self.personality = personality
        # A mesh bot is CONVERSATIONAL: it must carry conversation state
        # across messages so memory, agentic rounds, tool history, and
        # artefacts persist. The discovery is done in main() and passed
        # here; constructing one per message would defeat the point.
        self.discussion = discussion

        self.registry = MeshRegistry(layout)
        self.bus = MessageBus(layout)
        self._stop = threading.Event()
        self._heartbeat_stop = threading.Event()
        # A single-worker executor guarantees discussion.chat() runs strictly
        # serially on the personality — the only safe concurrency model
        # given LollmsPersonality/LollmsDiscussion internal state.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mesh-bot-{bot_id}-chat"
        )
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        # Reception subscriber name is deterministic per (bot_id, topic),
        # so a bot that restarts resumes from its persisted cursor — a bot
        # NEVER misses a message across a crash, by design of the bus.
        self._subscriber_id = f"inbox::{bot_id}"

    # ---------------------------------------------------- subscription setup
    def _ensure_subscriptions(self) -> List[str]:
        """
        Subscribes the bot to each of its declared inbound routes. Returns
        the topics actually subscribed. Empty route list means "the operator
        hasn't wired this bot yet" — we log it and idle; we do NOT fall back
        to "*" (that would be an accidental broadcast listener).
        """
        topics = [t.strip() for t in (self.spec.inbound_routes or []) if t.strip()]
        for topic in topics:
            self.bus.subscribe(self._subscriber_id, topic, from_now=False)
        return topics

    # ------------------------------------------------------------- heartbeat
    def _heartbeat_loop(self) -> None:
        """
        Updates the registry's ``last_seen`` periodically. This is what
        lets the hub distinguish "running but silent" (bot is fine, no
        messages) from "dead" (process gone, tears down its entry via
        the supervisor's reconcile).
        """
        while not self._heartbeat_stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            try:
                self.registry.update_status(self.bot_id, heartbeat=True)
            except Exception as e:
                _log_error("heartbeat", f"failed: {e}")

    # ----------------------------------------------------------------- loop
    def run(self) -> int:
        """The main pump. Blocks; returns a process exit code."""
        topics = self._ensure_subscriptions()
        if not topics:
            _log(
                "pump",
                f"no inbound_routes on spec — idling. "
                f"Set inbound_routes in the bot spec to receive messages.",
            )
        else:
            _log("pump", f"subscribed to topics: {', '.join(topics)}")

        # Enter RUNNING state and start the heartbeat.
        try:
            self.registry.update_status(
                self.bot_id, status=BotStatus.RUNNING, pid=os.getpid()
            )
        except Exception as e:
            _log_error("pump", f"failed to enter RUNNING state: {e}")
            return 1

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"mesh-bot-{self.bot_id}-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        _log("pump", "running — Ctrl+C to stop")

        try:
            while not self._stop.is_set():
                # Poll every subscribed topic once per interval. We do a
                # small batch from each so a burst on one topic doesn't
                # starve the others.
                for topic in topics:
                    if self._stop.is_set():
                        break
                    try:
                        batch = self.bus.poll_topic(
                            self._subscriber_id, topic, max_batch=8
                        )
                    except Exception as e:
                        _log_error("pump", f"poll {topic!r} failed: {e}")
                        continue
                    for msg in batch:
                        if self._stop.is_set():
                            break
                        self._dispatch(msg)

                # Sleep in small slices so SIGTERM is honored promptly.
                self._stop.wait(_PUMP_POLL_INTERVAL_SECONDS)
        finally:
            self._shutdown()

        _log("pump", "stopped cleanly")
        return 0

    def stop(self) -> None:
        """
        Signals the pump to stop and cancels an in-flight chat if any. The
        cancel is unconditional: LollmsPersonality.cancel_generation() is
        idempotent, and a leak here (waiting out a slow LLM call on SIGTERM)
        is exactly what the supervisor reconciles against as "zombie".
        """
        self._cancel.set()
        self._stop.set()
        try:
            self.personality.cancel_generation()
        except Exception:
            pass

    # -------------------------------------------------------------- dispatch
    def _dispatch(self, msg: BusMessage) -> None:
        """
        Submits a message to the single-worker executor.

        Backpressure: if the single worker is already busy (a prior chat is
        mid-flight), the message has ALREADY been consumed from the bus
        (its cursor advanced). We must not block the pump — a busy bot has
        to keep heart-beating and honor SIGTERM — but we also must not let
        the executor queue grow unbounded. So: if the worker is busy, we
        DROP this message and publish an `.error` reply on its `.error`
        topic explaining the backpressure. Callers can retry or route to
        another bot. This is a real design decision: bus-level buffering
        happens on the SENDER side via subscriptions, not by accumulating
        work in a bot's process.
        """
        try:
            # ``self._executor._work_queue.qsize()`` is the cleanest signal
            # of "the single worker has a queue". Both CPython and PyPy
            # expose it; guarded defensively anyway.
            queued = getattr(getattr(self._executor, "_work_queue", None), "qsize", lambda: 0)()
        except Exception:
            queued = 0
        if queued >= 1:
            _log(
                "pump",
                f"backpressure: dropping msg {msg.id} "
                f"(worker has {queued} pending)",
            )
            self._publish_error(
                msg,
                "bot is busy (single-worker backpressure); "
                "retry or route to another bot",
            )
            return
        self._executor.submit(self._handle_message, msg)

    # -------------------------------------------------------------- worker
    def _handle_message(self, msg: BusMessage) -> None:
        """Runs inside the single-worker executor thread."""
        start = time.time()
        if self._cancel.is_set():
            return
        try:
            _log(
                "handle",
                f"from={msg.sender!r} topic={msg.topic!r} id={msg.id}",
            )
            prompt = self._extract_prompt(msg)
            if prompt is None:
                self._publish_error(
                    msg, "message payload must contain a string field 'text'"
                )
                return
            if len(prompt) > _MAX_MESSAGE_BODY_CHARS:
                self._publish_error(
                    msg,
                    f"message body too large ({len(prompt)} chars, "
                    f"limit {_MAX_MESSAGE_BODY_CHARS})",
                )
                return

            # ── Drive the discussion, NOT the personality directly. ──
            # discussion.chat() runs the full agentic loop: tools, artefacts,
            # skills, memory — everything the mesh bot was built to do. A
            # direct personality.chat() call silently bypasses all of it.
            try:
                result = self.discussion.chat(
                    user_message=prompt,
                    personality=self.personality,
                    add_user_message=True,
                    # Mesh bots run non-streaming: the caller is another bot
                    # or the mesh bus, not a UI, so a streaming callback
                    # would just be discarded bandwidth.
                    streaming_callback=None,
                    # Sensible defaults for a service-style bot. Operators
                    # can override via spec.capabilities later.
                    enable_artefacts=True,
                    enable_memory=True,
                    enable_episodic_memory=False,
                    enable_auto_dream=False,
                    enable_deep_memory_pulling=False,
                    enable_notes=True,
                    enable_skills=True,
                )
            except Exception as e:
                _log_error("handle", f"discussion.chat() raised: {e}", e)
                self._publish_error(
                    msg, f"chat() raised {type(e).__name__}: {e}"
                )
                return

            reply_text = self._extract_reply_text(result)
            if not reply_text:
                self._publish_error(msg, "chat() returned an empty reply")
                return

            self._publish_reply(msg, reply_text)
            duration = time.time() - start
            _log(
                "handle",
                f"replied to {msg.id} in {duration:.2f}s "
                f"({len(reply_text)} chars)",
            )
        finally:
            pass

    # ------------------------------------------------------------ envelopes
    @staticmethod
    def _extract_prompt(msg: BusMessage) -> Optional[str]:
        """
        Prompt extraction. Accepts ``payload = str`` directly, or
        ``payload = {"text": str, ...}``. Anything else is a bus-level
        protocol violation and is surfaced as an error reply.
        """
        payload = msg.payload
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            candidate = payload.get("text")
            if isinstance(candidate, str):
                return candidate
        return None

    @staticmethod
    def _extract_reply_text(result: Any) -> str:
        """
        The personality's chat() return shape is documented as a dict with
        ``ai_message`` — but very old / synthetic paths return a string.
        We handle both rather than raising a misleading "empty reply".
        """
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            ai = result.get("ai_message")
            if ai is None:
                return ""
            content = getattr(ai, "content", None)
            if content is None and isinstance(ai, dict):
                content = ai.get("content")
            if isinstance(content, str):
                return content.strip()
        return ""

    def _publish_reply(self, msg: BusMessage, text: str) -> None:
        """
        Publishes on ``<topic>.reply``; preserves correlation_id.

        If the inbound message has NO bus-level ``correlation_id`` but its
        payload carries a ``reply_token`` (that's the swarm-connector shape),
        we use the token as the correlation id so the connector's
        ``wait_for_reply`` matches. This is the piece that makes a bot
        subscribed to the connector's ingress topic answer a human on
        Discord/Slack without any orchestration glue.
        """
        reply_topic = f"{msg.topic}{_REPLY_TOPIC_SUFFIX}"
        corr = msg.correlation_id
        if corr is None and isinstance(msg.payload, dict):
            token = msg.payload.get("reply_token")
            if isinstance(token, str) and token:
                corr = token
        try:
            self.bus.send_message(
                reply_topic,
                _serialize_reply(self.bot_id, msg.topic, msg, text),
                sender=self.bot_id,
                correlation_id=corr,
                headers={"in_reply_to_msg_id": msg.id},
            )
        except Exception as e:
            _log_error("handle", f"publish reply failed: {e}")

    def _publish_error(self, msg: BusMessage, error: str) -> None:
        error_topic = f"{msg.topic}{_ERROR_TOPIC_SUFFIX}"
        corr = msg.correlation_id
        if corr is None and isinstance(msg.payload, dict):
            token = msg.payload.get("reply_token")
            if isinstance(token, str) and token:
                corr = token
        try:
            self.bus.send_message(
                error_topic,
                _serialize_error(self.bot_id, msg, error),
                sender=self.bot_id,
                correlation_id=corr,
                headers={"in_reply_to_msg_id": msg.id},
            )
        except Exception as e:
            _log_error("handle", f"publish error failed: {e}")

    # ------------------------------------------------------------ shutdown
    def _shutdown(self) -> None:
        """Drains what we can, then releases resources. Idempotent."""
        # Stop the heartbeat first so its thread doesn't touch a closed bus.
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

        # Give any in-flight chat a bounded window to notice the cancel.
        self._executor.shutdown(wait=False)
        # Try to mark STOPPED (the supervisor also does this on our exit;
        # doing it here makes the transition visible to the hub slightly
        # sooner and works even if the supervisor crashed).
        try:
            self.registry.update_status(self.bot_id, status=BotStatus.STOPPED)
        except Exception:
            pass
        try:
            self.bus.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mesh-bot",
        description="Run one lollms-bot mesh bot (called by MeshSupervisor).",
    )
    parser.add_argument(
        "--mesh-root",
        required=True,
        help="Parent directory holding <mesh_id>/ folders.",
    )
    parser.add_argument(
        "--mesh-id",
        required=True,
        help="Mesh identifier (a subfolder of mesh-root).",
    )
    parser.add_argument(
        "--bot-id",
        required=True,
        help="Bot identifier within the mesh.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Install signal handlers BEFORE we build the (slow) client so Ctrl+C
    # during startup is graceful too.
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        _log("signal", f"received {signum}; shutting down")
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        # Not on the main thread; ignore (rare embedded usage).
        pass

    mesh_root = Path(args.mesh_root).expanduser().resolve()
    layout = MeshLayout.under(mesh_root, args.mesh_id)
    layout.materialize()

    # --- Startup sequence (each step has a clear failure mode) ---
    try:
        spec = _load_bot_spec(layout, args.bot_id)
    except Exception as e:
        _log_error("startup", f"cannot load bot spec: {e}")
        return 1

    _log(
        "startup",
        f"bot={args.bot_id!r} mesh={args.mesh_id!r} handbag={spec.handbag_path!r}",
    )

    # Enter STARTING immediately so the hub sees movement even before the
    # client/personality finish warming up.
    try:
        MeshRegistry(layout).update_status(
            args.bot_id, status=BotStatus.STARTING, pid=os.getpid()
        )
    except Exception as e:
        _log_error("startup", f"failed to enter STARTING: {e}")
        return 1

    try:
        client = _build_client_with_profile(spec.llm_profile)
    except Exception as e:
        _log_error("startup", f"client build failed: {e}", e)
        try:
            MeshRegistry(layout).update_status(
                args.bot_id, status=BotStatus.FAILED, error=f"client: {e}"
            )
        except Exception:
            pass
        return 1

    if stop_event.is_set():
        _log("startup", "stopped by user during client build")
        return 0

    try:
        personality = _build_personality(client, spec)
    except Exception as e:
        _log_error("startup", f"personality build failed: {e}", e)
        try:
            MeshRegistry(layout).update_status(
                args.bot_id, status=BotStatus.FAILED, error=f"personality: {e}"
            )
        except Exception:
            pass
        return 1

    if stop_event.is_set():
        _log("startup", "stopped by user during personality build")
        return 0

    # ── Build the per-bot discussion BEFORE constructing MeshBot. ──
    # LollmsDiscussion is what turns a LollmsPersonality into a real agent;
    # without it, capabilities cannot fire. We persist it under
    # <bot_dir>/discussion/ so the bot's conversation state survives
    # restarts (matching the daemon's behaviour, minus the per-message disk
    # syncing that makes BotDaemon unusable at mesh scale).
    if not _DISCUSSION_OK:
        _log_error("startup", "LollmsDiscussion is unavailable — cannot run mesh bot")
        return 1

    discussion_dir = layout.bot_dir(args.bot_id) / "discussion"
    discussion_dir.mkdir(parents=True, exist_ok=True)
    db_path = discussion_dir / "discussion.db"

    try:
        discussion = _build_discussion(client, db_path)
        _log("startup", f"discussion at {discussion_dir}")
    except Exception as e:
        _log_error("startup", f"discussion build failed: {e}", e)
        try:
            MeshRegistry(layout).update_status(
                args.bot_id, status=BotStatus.FAILED, error=f"discussion: {e}"
            )
        except Exception:
            pass
        return 1

    if stop_event.is_set():
        _log("startup", "stopped by user during discussion build")
        return 0

    # --- The pump ---
    try:
        bot = MeshBot(layout, args.bot_id, spec, client, personality, discussion)
    except Exception as e:
        _log_error("startup", f"MeshBot init failed: {e}", e)
        return 1

    # Bridge our stop-event into the bot's stop():
    watcher = threading.Thread(
        target=lambda: (stop_event.wait(), bot.stop()),
        name=f"mesh-bot-{args.bot_id}-signal-watch",
        daemon=True,
    )
    watcher.start()

    try:
        return bot.run()
    except KeyboardInterrupt:
        _log("run", "keyboard interrupt")
        bot.stop()
        return 0
    except Exception as e:
        _log_error("run", f"fatal: {e}", e)
        try:
            MeshRegistry(layout).update_status(
                args.bot_id, status=BotStatus.FAILED, error=f"runtime: {e}"
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
