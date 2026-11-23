
"""
Swarm connector: bridges external connection bindings to the mesh bus.

The mesh is a federation of agents that only speak to each other over the
:class:`MessageBus`. A human talking to the mesh from Discord, Slack,
lollms.com, or any other lollms-client *connection binding* needs a bridge
from those platforms onto the bus — that bridge is this module.

Architecture
------------
Two directions, ONE hub-side daemon:

    [Discord/Slack/lollms.com user]
              │  (platform message)
              ▼
    connection_binding  ──on_message──►  SwarmConnector
              │                            │
              │                            ├─ publish to <ingress_topic>
              │                            │   (payload: {text, sender,
              │                            │    platform, channel_id,
              │                            │    reply_token})
              ▼                            ▼
        [personality reply]  ◄──  subscribe <ingress_topic>.reply
              ▲                            │
              │                            ├─ correlate reply_token
              │  (platform send)           ▼
              └────────────────────  send back to the SAME channel

Design decisions (and why)
--------------------------
* **Ingress topic, not a special bot.** The connector publishes on a normal
  mesh topic (default ``swarm.inbound``). Whoever is subscribed to that topic
  — typically the *Swarm Orchestrator* agent — handles it. The connector
  never decides routing; that is the orchestrator's job. This keeps the
  connector a dumb, testable transport adapter.

* **Reply-token correlation, not per-user subscriptions.** Each inbound
  message carries a fresh ``reply_token`` in its payload and the connector
  waits for a message on the ``.reply`` topic whose ``correlation_id`` is
  that token. That is exactly the bus's request/reply primitive
  (``wait_for_reply``), so we reuse it rather than invent a second scheme.

* **One worker thread per inbound message, bounded.** A human waiting 20 s
  for a big LLM answer must not block the next human. We run each exchange
  on a thread from a bounded pool so a burst degrades gracefully (queued,
  not lost) and an idle connector costs nothing.

* **The connection binding owns the platform.** The connector does not know
  how Discord frames a message vs Slack vs lollms.com. Platform adapters
  implement :class:`SwarmPlatformAdapter`, whose :meth:`publish_reply` hook
  the connector invokes when a mesh reply lands. If no adapter is supplied,
  the connector still *ingests* (external clients already publishing on the
  ingress topic work), it simply cannot push replies back over the platform.

* **Fail-closed when no binding is configured.** With no adapter AND no
  ingress traffic configured, the hub reports "swarm connector: off", rather
  than silently swallowing platform messages.

Configuration (persisted under ``<mesh_root>/swarm_connector.json``)
--------------------------------------------------------------------
    {
      "enabled": true,
      "ingress_topic": "swarm.inbound",
      "reply_timeout_seconds": 120,
      "max_concurrent": 4,
      "announce_identity": true,
      "connection_name": "discord"
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_bot.mesh.bus import MessageBus
from lollms_client.apps.lollms_bot.mesh.types import MeshLayout


class SwarmConnectorError(RuntimeError):
    """Raised when the swarm connector cannot start or is misused."""


# ---------------------------------------------------------------------------
# Platform adapter contract
# ---------------------------------------------------------------------------

@dataclass
class InboundMessage:
    """
    A normalized inbound message from a platform. Platform adapters produce
    these before handing them to :meth:`SwarmConnector.ingest`; the connector
    only understands this shape, never any platform specifics.

    ``reply_token`` is assigned by the connector (adapters leave it unset).
    ``channel_id`` is opaque to the connector — it is echoed back to
    ``adapter.publish_reply`` so the adapter can route the answer to the right
    channel/thread.
    """

    text: str
    sender: str
    platform: str = "unknown"
    channel_id: Optional[str] = None
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class SwarmPlatformAdapter:
    """
    Minimal contract a platform adapter must implement so the swarm
    connector can push a mesh reply back to the originating conversation.

    A concrete adapter might wrap an lollms-client connection binding (Discord,
    Slack, lollms.com, ...) whose inbound loop calls ``connector.ingest(...)``
    for every platform message and whose outbound path is delegated to
    ``publish_reply`` here.

    We use an abstract base (not a Protocol) because we want a *stable*,
    introspectable type for the hub's REST surface: ``isinstance(x, SwarmPlatformAdapter)``
    is used to accept/reject an adapter cleanly.
    """

    #: Human-readable platform name, used in the mesh payload and logs.
    platform: str = "unknown"

    def publish_reply(self, channel_id: Optional[str], text: str, metadata: Dict[str, Any]) -> None:
        """
        Called from a worker thread when a mesh reply (or error) arrives for
        an inbound message that originated on this platform.

        Implementations MUST be thread-safe and MUST NOT block: the connector
        calls this off the ingest path, but a slow ``publish_reply`` still
        delays the pool worker's next job. Where possible, defer the actual
        platform send to the platform's own async loop / queue.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _config_path(layout: MeshLayout) -> Path:
    return layout.root / "swarm_connector.json"


def load_connector_config(layout: MeshLayout) -> Dict[str, Any]:
    """Loads the connector config (defaults if absent/corrupt)."""
    defaults = {
        "enabled": False,
        "ingress_topic": "swarm.inbound",
        "reply_timeout_seconds": 120,
        "max_concurrent": 4,
        "announce_identity": True,
        "connection_name": "",
    }
    p = _config_path(layout)
    if not p.is_file():
        return defaults
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(raw, dict):
        return defaults
    cfg = dict(defaults)
    for k in defaults:
        if k in raw:
            cfg[k] = raw[k]
    return cfg


def save_connector_config(layout: MeshLayout, cfg: Dict[str, Any]) -> None:
    """Atomically persists the connector config. Raises on write failure."""
    p = _config_path(layout)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class SwarmConnector:
    """
    Bridges external platform messages onto the mesh bus (and back).

    Lifecycle::

        conn = SwarmConnector(layout, adapter=my_discord_adapter)
        conn.start()          # starts polling for replies on ingress.reply
        conn.ingest(inbound)  # call this from the adapter's inbound loop
        conn.stop()           # releases the bus and joins the worker

    The connector owns ONE bus connection and a bounded worker pool. It is
    safe to call :meth:`ingest` from any thread (the adapter's platform loop).
    """

    _POLL_INTERVAL_SECONDS = 0.25

    def __init__(
        self,
        layout: MeshLayout,
        *,
        adapter: Optional[SwarmPlatformAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.layout = layout
        self.layout.materialize()
        self.cfg = config or load_connector_config(layout)
        self.adapter = adapter
        # If the adapter is a ConnectionBindingAdapter (or any caller-
        # supplied adapter that exposes the optional ``attach`` hook), wire
        # the back-reference here so callers never forget it. Adapters
        # without the hook are ignored — they don't need it.
        if adapter is not None:
            attach_fn = getattr(adapter, "attach", None)
            if callable(attach_fn):
                try:
                    attach_fn(self)
                except Exception:
                    # Never fail construction on an adapter wiring bug — the
                    # caller still gets a functioning connector; they'll see
                    # the failure in the adapter's own logs when it tries to
                    # forward.
                    pass

        self.ingress_topic = str(self.cfg.get("ingress_topic") or "swarm.inbound")
        self.reply_topic = f"{self.ingress_topic}.reply"
        self.error_topic = f"{self.ingress_topic}.error"
        self.reply_timeout = float(self.cfg.get("reply_timeout_seconds", 120))
        self.max_concurrent = max(1, int(self.cfg.get("max_concurrent", 4)))

        self._bus: Optional[MessageBus] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._subscriber = f"swarm-connector::{uuid.uuid4().hex[:8]}"
        self._lock = threading.RLock()
        self._running = False
        # One dedicated thread drains the reply subscription; it is NOT the
        # pool, so a slow worker cannot delay delivery of an unrelated
        # channel's reply.
        self._reply_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Tracks in-flight tokens so a late reply for an abandoned request is
        # ignored cleanly (the token will not be in this set).
        self._pending: set = set()

    # ---------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Opens the bus and starts the reply-drain thread. Idempotent."""
        with self._lock:
            if self._running:
                return
            if not self.cfg.get("enabled", False) and self.adapter is None:
                raise SwarmConnectorError(
                    "swarm connector is disabled and no adapter was supplied; "
                    "enable it in the hub or pass a SwarmPlatformAdapter."
                )
            self._bus = MessageBus(self.layout)
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_concurrent,
                thread_name_prefix="swarm-connector",
            )
            # Subscribe to the reply topic so we capture every answer. We use
            # the bus's pub/sub semantics (group=None) — one connector per
            # mesh is the deployment assumption; multiple connectors would
            # each need their own diagnosis topic to avoid cross-talk.
            #
            # We also subscribe to the error topic so a bot that fails to
            # produce a real reply can surface its diagnostic to the human
            # instead of the generic timeout notice. Both subscriptions use
            # from_now=True: only FUTURE messages matter (the token is fresh).
            self._bus.subscribe(self._subscriber, self.reply_topic, from_now=True)
            self._bus.subscribe(self._subscriber, self.error_topic, from_now=True)
            self._stop_event.clear()
            self._running = True
            self._reply_thread = threading.Thread(
                target=self._reply_loop,
                name="swarm-connector-reply-drain",
                daemon=True,
            )
            self._reply_thread.start()
        ASCIIColors.info(
            f"[SwarmConnector] listening ingress='{self.ingress_topic}' "
            f"replies='{self.reply_topic}'"
        )

    def stop(self) -> None:
        """Stops the reply drain and joins the worker pool. Idempotent."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            thread = self._reply_thread
            self._reply_thread = None
        if thread is not None:
            thread.join(timeout=3.0)
        with self._lock:
            if self._pool is not None:
                # Do NOT join the pool: an in-flight LLM call can legitimately
                # take longer than shutdown should wait. It will notice the
                # closed bus when it tries to publish and exit through the
                # exception path.
                self._pool.shutdown(wait=False)
                self._pool = None
            if self._bus is not None:
                try:
                    self._bus.unsubscribe(self._subscriber, self.reply_topic)
                except Exception:
                    pass
                try:
                    self._bus.unsubscribe(self._subscriber, self.error_topic)
                except Exception:
                    pass
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None
            self._pending.clear()

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------- ingest
    def ingest(self, inbound: InboundMessage) -> str:
        """
        Accepts one inbound platform message. Publishes it on the ingress
        topic with a fresh ``reply_token`` and schedules a worker that waits
        for the corresponding reply, then hands it to the adapter.

        Returns the ``reply_token`` (useful for tests and for adapters that
        want to show "awaiting reply" state).

        Raises :class:`SwarmConnectorError` if the connector is not running.
        """
        if not self._running or self._bus is None or self._pool is None:
            raise SwarmConnectorError("connector is not running; call start() first")

        token = uuid.uuid4().hex
        # Register the token BEFORE publish so an extremely fast responder
        # cannot race us into "unknown token" filtering.
        with self._lock:
            self._pending.add(token)

        payload = {
            "text": inbound.text,
            "sender": inbound.sender,
            "platform": inbound.platform,
            "channel_id": inbound.channel_id,
            "reply_token": token,
            "metadata": dict(inbound.metadata or {}),
        }
        try:
            self._bus.send_message(
                self.ingress_topic,
                payload,
                sender=f"swarm:{inbound.sender}",
                # The token is the correlation id: the orchestrator's reply
                # carries it back so wait_for_reply can match it.
                correlation_id=token,
                headers={"swarm": True, "platform": inbound.platform},
                ttl_seconds=self.reply_timeout + 30,
            )
        except Exception as e:
            with self._lock:
                self._pending.discard(token)
            raise SwarmConnectorError(f"publish to ingress failed: {e}") from e

        # Capture the bus reference now: stop() may null it later, but the
        # worker needs the same instance to wait on its reply.
        bus = self._bus
        self._pool.submit(self._await_and_reply, inbound, token, bus)
        return token

    # ------------------------------------------------------------ internals
    def _await_and_reply(
        self, inbound: InboundMessage, token: str, bus: MessageBus
    ) -> None:
        """
        Worker: waits for the reply (or an error), then hands the answer to
        the adapter's ``publish_reply``.

        Both reply and error topics are scanned with the SAME deadline. A bot
        that fails during handling publishes on ``.error``; surfacing that
        text to the human is far better than the generic timeout notice —
        they can see WHY the mesh could not answer.
        """
        reply = None
        deadline = self._now() + float(self.reply_timeout)
        try:
            while self._now() < deadline:
                if self._stop_event.is_set():
                    break
                for topic in (self.reply_topic, self.error_topic):
                    try:
                        candidate = bus.wait_for_reply(
                            token,
                            topic=topic,
                            timeout=min(0.5, max(0.0, deadline - self._now())),
                            should_stop=lambda: self._stop_event.is_set(),
                        )
                    except Exception as e:
                        ASCIIColors.warning(
                            f"[SwarmConnector] reply wait on {topic!r} "
                            f"failed: {e}"
                        )
                        continue
                    if candidate is not None:
                        reply = candidate
                        break
                if reply is not None:
                    break
        finally:
            with self._lock:
                self._pending.discard(token)

        if reply is None:
            # Timed out or the connector is stopping. Only notify the platform
            # if the connector is still running — during shutdown a timeout
            # message would be noise.
            if self._running and self.adapter is not None:
                self._safe_publish_reply(
                    inbound, "(no reply from the mesh within "
                    f"{int(self.reply_timeout)}s — the agents may be busy)"
                )
            return

        if self.adapter is None:
            # No adapter: we ingested traffic but cannot push the reply back.
            # Log once per reply so an operator sees why nothing landed.
            ASCIIColors.info(
                f"[SwarmConnector] reply for token {token[:8]} received but "
                f"no adapter is attached to deliver it."
            )
            return

        text = self._extract_reply_text(reply.payload)
        self._safe_publish_reply(inbound, text)

    @staticmethod
    def _extract_reply_text(payload: Any) -> str:
        """
        Mesh replies (from runtime.py / orchestrator) put the answer in
        ``payload.text``. Be liberal: accept a bare string too, so a
        hand-built reply still reaches the human.
        """
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text
            err = payload.get("error")
            if isinstance(err, str):
                return f"(agent error) {err}"
        return "(empty reply)"

    def _safe_publish_reply(self, inbound: InboundMessage, text: str) -> None:
        """
        Calls the adapter's ``publish_reply`` defensively. A platform adapter
        bug must NEVER crash the connector worker — the token is already
        dropped, so the worst case is a lost reply for one message, which we
        log loudly.
        """
        assert self.adapter is not None
        try:
            self.adapter.publish_reply(
                inbound.channel_id,
                text,
                {
                    "platform": inbound.platform,
                    "sender": inbound.sender,
                    **(inbound.metadata or {}),
                },
            )
        except Exception as e:
            ASCIIColors.warning(
                f"[SwarmConnector] adapter.publish_reply failed "
                f"(channel={inbound.channel_id!r}): {e}"
            )

    def _reply_loop(self) -> None:
        """
        Drains the reply subscription so its cursor advances and stale
        messages do not accumulate. The actual delivery is done by the
        per-message workers (``_await_and_reply``) via ``wait_for_reply`` on
        the reply subscriber — those calls are what consume the messages.

        This loop exists to keep the connector's OWN presence on the reply
        topic healthy and to give the hub a place to observe liveness. It
        deliberately does NOT poll the reply topic (that would steal messages
        from the workers' ``wait_for_reply`` subscriptions); instead it just
        idles until stop.
        """
        while not self._stop_event.wait(self._POLL_INTERVAL_SECONDS * 4):
            # No-op heartbeat tick: keeps the thread cheaply alive and makes
            # the intent explicit (this is a coordinator thread, not a poller).
            pass

    @staticmethod
    def _now() -> float:
        """Single source of time for deadlines — imported lazily via `time`."""
        return time.time()

    # ---------------------------------------------------------- introspect
    def status(self) -> Dict[str, Any]:
        """Snapshot for the hub REST surface."""
        with self._lock:
            return {
                "running": self._running,
                "ingress_topic": self.ingress_topic,
                "reply_topic": self.reply_topic,
                "error_topic": self.error_topic,
                "reply_timeout_seconds": self.reply_timeout,
                "max_concurrent": self.max_concurrent,
                "pending": len(self._pending),
                "adapter_attached": self.adapter is not None,
                "adapter_platform": (
                    self.adapter.platform if self.adapter is not None else None
                ),
            }
