
"""
Session recorder / replayer for the hub (phase 6a).

Records the LIVE communication flow of a mesh (bus topics + registry status
transitions) as a stream of timestamped events, and lets the hub replay it
later as a visual animation. This is what powers "watch an animation of what
the agents are doing" and "record and replay visually what they do".

Storage
-------
One JSONL file per session under ``<mesh_root>/sessions/``. Each line is one
event envelope::

    {"t": 1699999999.123, "kind": "message", "topic": "tasks", "sender": "a",
     "payload": {...}, "headers": {...}, "correlation_id": "..."}
    {"t": 1699999999.456, "kind": "status", "bot_id": "b",
     "status": "running"}

JSONL is append-only (crash-safe), greppable, and cheap to stream back to the
browser one line at a time. Recording is fire-and-forget from the hub's
background watcher — nothing in the hot path (bus publish, registry update)
is blocked by the recorder.

Replay modes
------------
``Replayer.iter_events()`` yields events in order with the DELAYS PRESERVED
(delta between event timestamps) OR compressed by a ``speed`` multiplier.
This is exactly what the UI needs to redraw the graph over time.

The recorder is a *follower*, not a *source*. It polls the bus for new
messages on subscribed topics and the registry for status changes, both via
short durable polls at a configurable interval. This design keeps it
completely decoupled: mode "record" can be toggled without touching a bot,
and recording simply observes the same transport every bot uses.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from lollms_client.apps.lollms_bot.mesh.bus import BusMessage, MessageBus
from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.types import MeshLayout


class RecorderError(RuntimeError):
    """Raised for recorder I/O failures and misuse."""


@dataclass
class RecordedEvent:
    """One recorded event, decoded from a JSONL line."""

    t: float
    kind: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"t": self.t, "kind": self.kind, **self.data}


def _sessions_dir(layout: MeshLayout) -> Path:
    d = layout.root / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(layout: MeshLayout, session_id: str) -> Path:
    return _sessions_dir(layout) / f"{session_id}.jsonl"


def _now() -> float:
    return time.time()


class SessionRecorder:
    """
    Live recorder bound to a mesh. Start/stop is idempotent; each started
    recording gets a fresh ``session_id``. Thread-safe: the background
    poller only appends to a single file under its own lock.

    Usage
    -----
    ::

        rec = SessionRecorder(layout, topics=["*"])
        rec.start()                 # background follower starts
        rec.stop()                  # stops the follower, keeps the file

    The recorder intentionally does NOT auto-start. The hub calls ``start()``
    when the operator presses "Record" and ``stop()`` when they press "Stop".
    """

    _POLL_INTERVAL_SECONDS = 0.4

    def __init__(
        self,
        layout: MeshLayout,
        *,
        topics: Optional[List[str]] = None,
        all_topics: bool = True,
    ):
        self.layout = layout
        self.topics = list(topics or [])
        # ``all_topics=True`` means: subscribe to "*" — capture every message
        # the bus carries on this mesh, regardless of topic. This is what the
        # hub wants for full visual recording.
        self.all_topics = all_topics

        self._bus: Optional[MessageBus] = None
        self._registry: Optional[MeshRegistry] = None
        self._session_id: Optional[str] = None
        self._fh: Optional[Any] = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_statuses: Dict[str, str] = {}

    # ---------------------------------------------------------- lifecycle
    def start(self) -> str:
        """
        Starts a new session and the background poller. Returns the session
        id. Calling ``start()`` while already running raises — recording must
        be explicitly stopped first so session boundaries are unambiguous.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RecorderError("a recording is already in progress")

            self._session_id = uuid.uuid4().hex[:12]
            self._fh = open(_session_path(self.layout, self._session_id), "a", encoding="utf-8")
            self._bus = MessageBus(self.layout)
            self._registry = MeshRegistry(self.layout)

            # Subscribe the recorder to the "*" catch-all topic; the bus does
            # not have a wildcard subscription in phase 2, so we subscribe to
            # every topic currently known AND re-subscribe on the fly as new
            # topics appear (below).
            self._ensure_subscription("*")
            self._ensure_subscription("_all")
            self._ensure_subscription("__system__")
            for topic in self._known_topics():
                self._ensure_subscription(topic)

            self._last_statuses = {
                bot_id: entry.get("status", "")
                for bot_id, entry in self._registry.list_bots().items()
            }

            # Emit a sentinel "session_started" event so replay has a clear
            # origin marker (useful for the UI to render a timeline start).
            self._write_event("session_started", {
                "session_id": self._session_id,
                "topics": self.topics,
            })

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"recorder-{self._session_id}", daemon=True
            )
            self._thread.start()
            return self._session_id

    def stop(self) -> Optional[str]:
        """
        Stops the follower and flushes the file. Returns the session id that
        was recorded, or None if no recording was in progress.
        """
        with self._lock:
            if self._thread is None:
                return None
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
            session_id = self._session_id
            self._write_event("session_stopped", {"session_id": session_id})
            try:
                if self._fh is not None:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                    self._fh.close()
            except OSError:
                pass
            self._fh = None
            if self._bus is not None:
                self._bus.close()
                self._bus = None
            self._registry = None
            self._session_id = None
            return session_id

    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def current_session_id(self) -> Optional[str]:
        return self._session_id

    # ----------------------------------------------------------- internals
    def _known_topics(self) -> List[str]:
        if self._bus is None:
            return []
        try:
            return [t["topic"] for t in self._bus.list_topics()]
        except Exception:
            return []

    def _ensure_subscription(self, topic: str) -> None:
        if self._bus is None:
            return
        try:
            self._bus.subscribe(
                subscriber=f"recorder::{self._session_id}",
                topic=topic,
                from_now=False,
            )
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                # Recording must NEVER crash the hub; drop a poison interval
                # and retry at the next tick.
                pass
            self._stop_event.wait(self._POLL_INTERVAL_SECONDS)

    def _poll_once(self) -> None:
        if self._bus is None or self._registry is None:
            return

        # 1) New topics: subscribe on the fly so recording keeps up with bots
        #    that appear mid-session.
        for topic in self._known_topics():
            self._ensure_subscription(topic)

        # 2) Drain every recorder subscription.
        topics = self._known_topics()
        for topic in topics:
            try:
                batch = self._bus.poll_topic(
                    f"recorder::{self._session_id}", topic, max_batch=200
                )
            except Exception:
                continue
            for msg in batch:
                self._write_event("message", {
                    "topic": msg.topic,
                    "sender": msg.sender,
                    "payload": msg.payload,
                    "headers": msg.headers,
                    "correlation_id": msg.correlation_id,
                })

        # 3) Status transitions.
        for bot_id, entry in self._registry.list_bots().items():
            status = entry.get("status", "")
            prev = self._last_statuses.get(bot_id)
            if status and status != prev:
                self._last_statuses[bot_id] = status
                self._write_event("status", {
                    "bot_id": bot_id,
                    "status": status,
                    "host": entry.get("host"),
                })

    def _write_event(self, kind: str, data: Dict[str, Any]) -> None:
        # Serialize writes through the same lock that guards start/stop so a
        # stop() can't close the fh under the poller.
        with self._lock:
            if self._fh is None:
                return
            line = json.dumps(
                {"t": _now(), "kind": kind, **data},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Replayer — reads a recorded session, in order, with optional speed-up.
# ---------------------------------------------------------------------------

class Replayer:
    """
    Reads a session file and yields events with delays preserved.

    ``iter_events(block=True, speed=1.0)`` yields ``(delay, event)`` tuples
    where ``delay`` is the seconds to sleep BEFORE that event (matching the
    original recording), divided by ``speed``. ``block=False`` yields all
    events with ``delay=0`` — useful for computing a static summary.

    The replay is a pure, read-only operation: it never mutates the file.
    """

    def __init__(self, layout: MeshLayout, session_id: str):
        self.layout = layout
        self.session_id = session_id
        self.path = _session_path(layout, session_id)

    def exists(self) -> bool:
        return self.path.is_file()

    def _read_all(self) -> List[RecordedEvent]:
        if not self.path.is_file():
            raise RecorderError(f"session {self.session_id} not found")
        events: List[RecordedEvent] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    # Skip corrupt lines rather than aborting a replay: a
                    # partially written tail is normal after a crash.
                    continue
                if not isinstance(obj, dict) or "t" not in obj or "kind" not in obj:
                    continue
                t = float(obj.pop("t"))
                kind = str(obj.pop("kind"))
                events.append(RecordedEvent(t=t, kind=kind, data=obj))
        events.sort(key=lambda e: e.t)
        return events

    def iter_events(
        self,
        *,
        block: bool = True,
        speed: float = 1.0,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[tuple]:
        """
        Yields ``(delay_seconds, RecordedEvent)`` in replay order.

        ``block=True`` honours the recorded timing (sleeps between yields in
        the caller — see ``replay_sync`` for a helper that does this).
        ``block=False`` skips delays (delay always 0).
        ``should_stop`` lets a caller abandon the replay mid-stream.
        """
        events = self._read_all()
        if speed <= 0:
            raise RecorderError("speed must be > 0")
        last_t: Optional[float] = None
        for event in events:
            if should_stop is not None:
                try:
                    if should_stop():
                        return
                except Exception:
                    pass
            if block and last_t is not None:
                delay = max(0.0, (event.t - last_t) / speed)
            else:
                delay = 0.0
            yield delay, event
            last_t = event.t

    def replay_sync(
        self,
        *,
        speed: float = 1.0,
        should_stop: Optional[Callable[[], bool]] = None,
        on_event: Optional[Callable[[RecordedEvent], None]] = None,
    ) -> int:
        """
        Convenience synchronous replay: sleeps the recorded delay before
        invoking ``on_event`` for each event. Returns the number of events
        replayed. Blocks — call from a worker thread on the server.
        """
        count = 0
        for delay, event in self.iter_events(block=True, speed=speed, should_stop=should_stop):
            if delay > 0:
                time.sleep(delay)
            if on_event is not None:
                on_event(event)
            count += 1
        return count


def list_sessions(layout: MeshLayout) -> List[Dict[str, Any]]:
    """
    Enumerates recorded sessions, newest first. Each entry::
        {"session_id": "...", "size_bytes": ..., "mtime": ...}
    """
    d = _sessions_dir(layout)
    out: List[Dict[str, Any]] = []
    for entry in d.iterdir():
        if not entry.is_file() or entry.suffix != ".jsonl":
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        out.append({
            "session_id": entry.stem,
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
        })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def delete_session(layout: MeshLayout, session_id: str) -> bool:
    """Deletes one recorded session file. Returns True if a file was removed."""
    path = _session_path(layout, session_id)
    if not path.is_file():
        return False
    return bool(path.unlink(missing_ok=True))
