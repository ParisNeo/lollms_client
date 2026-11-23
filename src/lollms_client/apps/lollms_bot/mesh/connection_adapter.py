
"""
Swarm connector adapter for lollms-client connection bindings.

The :class:`SwarmConnector` speaks one generic protocol: ``SwarmPlatformAdapter``
(publish_reply + a shared "ingest" method the platform loop calls). Concrete
platforms — Discord, Slack, lollms.com, Telegram, ... — are implemented as
*lollms-client connection bindings*. This module glues the two so that a mesh
operator can pick any configured connection binding and have inbound messages
routed onto the mesh bus, with mesh replies pushed back to the originating
channel.

Design notes
------------
* **One adapter per binding.** Each ``LollmsConnectionBinding`` owns its own
  transport loop, credentials, and channel space. Binding two of them to the
  same connector would make reply routing ambiguous (a Discord reply must not
  land in a Slack channel with the same id). So the adapter instance is bound
  to exactly one binding.

* **Polling receive, not wake-up callbacks.** The base binding exposes both
  ``on_message`` (callback) and ``receive_message`` (pull). A connection
  binding's inbound loop lives on its own thread and may call
  ``connector.ingest(...)`` directly if we pass the connector in; but
  different bindings wire this differently (Discord uses ``on_message`` while
  a webhook-based transactional binding has no thread at all). To remain
  robust across all of them, this adapter *pulls* via ``receive_message`` on
  a small daemon thread and hands non-empty results to the connector. That
  works for any binding that implements the contract; nothing platform-
  specific leaks in.

* **Reply delivery is best-effort, thread-safe.** ``publish_reply`` is called
  from a connector worker thread. Connection bindings' ``send_message``
  must therefore be safe from off-loop threads. This is already the case
  for Discord/Slack bindings (they marshal to their own async loop), but
  the adapter still wraps the call in a try/except so a broken binding
  cannot crash the connector pool.

* **Loopback protection.** If the adapter ever ingests a message it should
  not (its own outbound, an empty payload), it silently drops it. The mesh
  bus identity is the *final* word on routing — the adapter only forwards
  what looks like a genuine inbound user message.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_bot.mesh.connector import (
    InboundMessage,
    SwarmPlatformAdapter,
)


# How often the pull loop wakes up to ask the binding for messages. Connection
# bindings that already push via on_message() will have an empty receive queue;
# this is only the fallback polling interval for pull-based bindings. Small
# enough to feel instant, large enough to be negligible CPU.
_DEFAULT_POLL_INTERVAL_SECONDS = 0.5


class ConnectionBindingAdapter(SwarmPlatformAdapter):
    """
    Bridges a lollms-client ``LollmsConnectionBinding`` onto a ``SwarmConnector``.

    Typical usage::

        connector = SwarmConnector(layout, adapter=ConnectionBindingAdapter(binding))
        connector.start()
        adapter.start()          # starts pulling inbound messages
        ...
        adapter.stop()
        connector.stop()

    ``binding`` must expose:
        * ``send_message(channel_id, text, ...)``   → push a reply
        * ``receive_message(timeout=...)``          → pull one (or None)
        * ``start()`` / ``is_connected()``          → optional lifecycle hints

    Every dependency on the base binding is defensive (``getattr`` + try/
    except): a binding missing an optional hook degrades gracefully instead
    of exploding the mesh.
    """

    def __init__(
        self,
        binding: Any,
        *,
        platform: Optional[str] = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.binding = binding
        self.platform = platform or getattr(binding, "binding_name", None) or "connection"
        self.poll_interval = float(poll_interval)
        self._connector: Optional[Any] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- wiring
    def attach(self, connector: Any) -> None:
        """
        Records the connector that owns this adapter so the pull loop can
        call ``connector.ingest(...)``. Called by the hub when it composes
        the pair; if no connector is attached, the pull loop is a no-op.
        """
        self._connector = connector

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Starts the inbound pull loop. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._pull_loop,
            name=f"swarm-adapter-{self.platform}",
            daemon=True,
        )
        self._thread.start()
        ASCIIColors.info(
            f"[ConnectionBindingAdapter] pulling inbound from platform '{self.platform}'"
        )

    def stop(self) -> None:
        """Stops the pull loop. Idempotent."""
        self._stop_event.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=3.0)

    # ------------------------------------------------------ platform contract
    def publish_reply(
        self,
        channel_id: Optional[str],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Pushes a mesh reply back to the platform via ``binding.send_message``.

        The binding's own signature varies a little across versions — we try
        the common ``send_message(channel_id, text)`` and fall back to a
        keyword form. If neither works we log loudly so the operator sees
        why a reply did not land (rather than silently swallowing it).
        """
        if not text:
            return
        target_channel = channel_id or self._default_channel()
        if not target_channel:
            ASCIIColors.warning(
                f"[ConnectionBindingAdapter] dropping reply on '{self.platform}': "
                f"no channel id supplied and the binding has no default channel."
            )
            return
        try:
            self.binding.send_message(target_channel, text)
            return
        except TypeError:
            pass
        except Exception as e:
            ASCIIColors.warning(
                f"[ConnectionBindingAdapter] send_message(channel, text) failed: {e}; "
                f"retrying keyword form."
            )
        try:
            self.binding.send_message(channel_id=target_channel, text=text)
        except Exception as e:
            ASCIIColors.warning(
                f"[ConnectionBindingAdapter] {self.platform}.send_message failed "
                f"(channel={target_channel!r}): {e}"
            )

    # ------------------------------------------------------------ internals
    def _default_channel(self) -> Optional[str]:
        """
        Best-effort default channel for a binding that addresses channels by
        configuration. Discord bindings expose ``allowed_channel_ids`` via
        ``_parse_ids``; we look for the first one. Other bindings expose
        ``channel_id`` directly. If none, we return None and let the caller
        decide (as opposed to silently guessing a channel name).
        """
        for attr in ("channel_id", "default_channel"):
            value = getattr(self.binding, attr, None)
            if isinstance(value, str) and value:
                return value
        allowed = getattr(self.binding, "allowed_channel_ids", None)
        if isinstance(allowed, (list, tuple, set)) and allowed:
            return str(next(iter(allowed)))
        return None

    def _pull_loop(self) -> None:
        """
        Periodically asks the binding for one message and forwards it to the
        connector. ``receive_message`` returns None on timeout — that's the
        normal, non-exceptional outcome and MUST NOT be logged as an error.

        A binding that raises in ``receive_message`` (transient network
        hiccup) is tolerated: we log once, sleep, and retry on the next tick.
        A persistent failure should not crash the mesh.
        """
        while not self._stop_event.wait(self.poll_interval):
            # A binding that isn't connected has nothing to deliver yet.
            is_connected = getattr(self.binding, "is_connected", None)
            if is_connected is not None:
                try:
                    if not is_connected():
                        continue
                except Exception:
                    pass
            try:
                message = self.binding.receive_message(timeout=self.poll_interval)
            except TypeError:
                # Older bindings may not accept a timeout kwarg.
                try:
                    message = self.binding.receive_message()
                except Exception as e:
                    ASCIIColors.warning(
                        f"[ConnectionBindingAdapter] {self.platform}.receive_message: {e}"
                    )
                    continue
            except Exception as e:
                ASCIIColors.warning(
                    f"[ConnectionBindingAdapter] {self.platform}.receive_message: {e}"
                )
                continue

            if not message:
                continue
            self._forward(message)

    def _forward(self, message: Any) -> None:
        """
        Normalizes a binding's inbound message and hands it to the connector.

        Bindings return different shapes (a dataclass with ``content`` +
        ``channel_id``, a plain dict, a string). We do an attribute-then-key
        probe — this is the whole reason the adapter is worth keeping in the
        codebase: it collocates every "duck-typing" assumption about
        lollms_client connection bindings in one auditable place.
        """
        connector = self._connector
        if connector is None:
            return

        channel_id = (
            getattr(message, "channel_id", None)
            or _get_dict(message, "channel_id")
        )
        author = (
            getattr(message, "author", None)
            or getattr(message, "sender", None)
            or _get_dict(message, "author")
            or _get_dict(message, "sender")
            or "user"
        )
        # Prefer an explicit ``content`` (matches ConnectionReceiveResult);
        # fall back to ``text`` / ``message`` / ``body`` — different bindings
        # name the field differently.
        text = (
            getattr(message, "content", None)
            or getattr(message, "text", None)
            or _get_dict(message, "content")
            or _get_dict(message, "text")
            or _get_dict(message, "message")
            or ""
        )
        if not isinstance(text, str) or not text.strip():
            return

        inbound = InboundMessage(
            text=text,
            sender=str(author),
            platform=self.platform,
            channel_id=str(channel_id) if channel_id is not None else None,
            metadata={},
        )
        try:
            connector.ingest(inbound)
        except Exception as e:
            ASCIIColors.warning(
                f"[ConnectionBindingAdapter] connector.ingest failed: {e}"
            )


# ---------------------------------------------------------------------------
# Factory: build adapters from lollms_client connection bindings in one call
# ---------------------------------------------------------------------------

def build_adapters_from_client(
    client: Any,
    *,
    platform_whitelist: Optional[List[str]] = None,
) -> List[ConnectionBindingAdapter]:
    """
    Returns one :class:`ConnectionBindingAdapter` per ACTIVE connection binding
    on an :class:`LollmsClient`, honoring an optional platform whitelist.

    This is the single shim a host (the hub, a daemon, a test) needs in order
    to prep a swarm connector for real platforms: it asks the client for its
    ``connections`` registry — mirroring ``connection_runtime._active_
    connection_bindings`` — and wraps each into an adapter. Adapters must
    still be attached to a connector and started; this function only
    constructs them.
    """
    registry = getattr(client, "connections", None)
    if isinstance(registry, dict) and registry:
        candidates = list(registry.values())
    else:
        active = getattr(client, "connection", None)
        candidates = [active] if active is not None else []

    whitelist = {str(w).lower() for w in (platform_whitelist or [])}

    adapters: List[ConnectionBindingAdapter] = []
    for binding in candidates:
        if binding is None:
            continue
        name = (
            getattr(binding, "binding_name", None)
            or getattr(binding, "name", None)
            or type(binding).__name__
        ).lower()
        if whitelist and name not in whitelist:
            continue
        adapters.append(ConnectionBindingAdapter(binding, platform=str(name)))
    return adapters


def _get_dict(obj: Any, key: str) -> Optional[Any]:
    """Returns ``obj.get(key)`` when ``obj`` is a Mapping, else None."""
    if isinstance(obj, dict):
        return obj.get(key)
    return None
