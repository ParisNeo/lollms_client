
"""
Mesh message bus: durable, multi-process, topic-based messaging for lollms-bot.

One ``bus.db`` (SQLite, WAL mode) per mesh. Every bot in the mesh talks to
the same file. The bus is a **transport** only — it carries JSON-serializable
payloads between bots and knows nothing about LLMs, personalities, or the
registry. The phase-4 workflow router and the phase-5 mesh tools decide what
messages mean and who gets them.

Why SQLite (not a socket server)
--------------------------------
* Zero extra dependency, portable to Windows (where the project lives).
* WAL mode = concurrent readers don't block a writer; SQLite serializes
  writers for us, so the bus is multi-process safe without a broker.
* Durability for free: a bot that restarts resumes from its persisted
  cursor, so no in-flight message is lost when a peer crashes.

Two delivery modes, ONE schema
------------------------------
Both are cursor semantics over the same ``messages`` table:

* **Queue**  — subscriptions sharing a ``consumer_group`` compete for
  messages; each message is delivered to exactly one member of the group.
  Use this to load-balance work across N equivalent bots. Implemented with
  a "checkpoint = MAX(cursor) over the group" rule (see
  ``_effective_cursor``), which requires no extra table and serializes
  correctly thanks to SQLite's write lock.

* **Pub/Sub** — subscriptions with NO ``consumer_group`` each get their own
  cursor and therefore see every message on the topic. Use this for
  broadcast (game moves, health pings, world-state updates).

Request/reply
-------------
``send_message(..., reply_to=token)`` tags an outbound message with a
correlation key. The caller then waits for a reply with ``correlation_id ==
token``. No special-case table: a reply subscriber is auto-registered.

Backpressure
------------
* ``poll_topic`` returns at most ``max_batch`` messages.
* ``pending_count`` lets a bot check its backlog and throttle.
* ``expire_old`` purges messages past their TTL so a stalled consumer can't
  grow the DB without bound.

Cursor invariant
----------------
A cursor advances **only forward**, to the max message id it has *observed*
(including expired-but-retained rows). This guarantees:

  * no message is ever delivered twice to the same subscription;
  * expired rows above the cursor cannot stall polling — they are "consumed"
    by the cursor advance even when they can't be delivered;
  * ``expire_old`` is always safe to run concurrently with polling.

Public surface (phase 2)
------------------------
    MessageBus(layout)                              # opens/creates bus.db
    .send_message(topic, payload, ...)              # publish
    .subscribe(subscriber, topic, ...)              # register durable reader
    .unsubscribe(subscriber, topic=None)            # drop reader + cursor
    .get_subscription(subscriber, topic)            # inspect
    .reset_cursor(subscriber, topic, cursor)        # replay control
    .poll_topic(subscriber, topic, max_batch)       # fetch + advance
    .wait_for(subscriber, topic, timeout, ...)      # poll loop w/ backoff
    .pending_count(subscriber, topic)               # backlog size
    .reply_subscriber(token)                        # canonical reply name
    .wait_for_reply(token, topic, timeout, ...)     # request/reply helper
    .expire_old(max_age_seconds=None)               # janitor
    .list_topics()                                  # introspection
    .close()                                        # release the connection
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterator, List, Optional

from lollms_client.apps.lollms_bot.mesh.types import MeshLayout


class BusError(RuntimeError):
    """Raised when a bus operation cannot be completed safely."""


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

@dataclass
class BusMessage:
    """
    One delivered message.

    ``id`` is the monotonic SQLite row id (used for cursor ordering).
    ``payload`` is the JSON-decoded body. ``headers`` is an optional
    JSON-decoded dict used by the router (message type, hop count, ...).
    """

    id: int
    topic: str
    sender: str
    payload: Any
    headers: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    created_at: float = 0.0
    expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Subscription:
    """
    A durable reader identity. The pair (``subscriber``, ``topic``) is the
    primary key; ``consumer_group`` decides queue vs pub/sub semantics.
    ``cursor`` is the last delivered/observed message id.
    """

    subscriber: str
    topic: str
    consumer_group: Optional[str] = None
    cursor: int = 0
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# NOTE: see the module docstring for the cursor invariant. The ``messages``
# table is append-only except for janitor deletions via ``expire_old``.
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    topic         TEXT    NOT NULL,
    sender        TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    headers       TEXT    NOT NULL DEFAULT '{}',
    correlation_id TEXT,
    consumer_group TEXT,
    created_at    REAL    NOT NULL,
    expires_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_topic_id
    ON messages(topic, id);
CREATE INDEX IF NOT EXISTS idx_messages_corr
    ON messages(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_expires
    ON messages(expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS subscriptions (
    subscriber     TEXT    NOT NULL,
    topic          TEXT    NOT NULL,
    consumer_group TEXT,
    cursor         INTEGER NOT NULL DEFAULT 0,
    created_at     REAL    NOT NULL,
    PRIMARY KEY (subscriber, topic)
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_topic
    ON subscriptions(topic);
"""


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class MessageBus:
    """
    Durable, multi-process topic bus backed by one SQLite file per mesh.

    Connection model
    ----------------
    Each :class:`MessageBus` instance owns ONE :class:`sqlite3.Connection`
    guarded by an in-process ``RLock``. Cross-process safety is provided by
    SQLite itself in WAL mode. A thread that needs concurrency may open a
    second ``MessageBus`` (each opens its own connection); OR call methods
    freely since every call is internally serialized.

    Resources are released via :meth:`close` or the context-manager form.
    """

    # Delay between empty polls in ``wait_for``. Responsive (250ms) without
    # busy-spinning the DB.
    _POLL_SLEEP_SECONDS = 0.25

    def __init__(self, layout: MeshLayout):
        self.layout = layout
        self.layout.materialize()
        self._lock = threading.RLock()
        self._closed = False
        self._db = self._connect(layout.bus_db)
        with self._acquire() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ io
    def _connect(self, path) -> sqlite3.Connection:
        """
        Opens the SQLite DB with WAL + busy timeout. ``check_same_thread=False``
        is safe because every access goes through :meth:`_acquire`.
        """
        try:
            conn = sqlite3.connect(
                str(path),
                timeout=30.0,          # busy_timeout for lock contention
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage transactions
            )
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise BusError(f"Failed to open bus DB {path}: {e}") from e

    @contextmanager
    def _acquire(self) -> Iterator[sqlite3.Connection]:
        """Serializes access to the connection within the process (re-entrant)."""
        if self._closed:
            raise BusError("MessageBus is closed.")
        with self._lock:
            yield self._db

    def close(self) -> None:
        """Releases the underlying SQLite connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._db.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _encode_json(value: Any, what: str) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            raise BusError(f"{what} is not JSON-serializable: {e}") from e

    @staticmethod
    def _decode_json(raw: Optional[str], what: str, *, default: Any = None) -> Any:
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def reply_subscriber(token: str) -> str:
        """Canonical subscriber name for a pending request/reply exchange."""
        return f"reply::{token}"

    # ---------------------------------------------------------- group state
    def _effective_cursor(
        self, conn: sqlite3.Connection, subscriber: str, topic: str,
        consumer_group: Optional[str],
    ) -> int:
        """
        Returns the cursor a subscriber must poll FROM.

        Pub/Sub: the subscriber's own ``cursor``.

        Queue: the MAXIMUM cursor across every member of the consumer_group on
        this topic. This is the classic "checkpoint = MAX over group" pattern
        that gives competing-consumer semantics: whichever member polls next
        sees only messages that NO member of the group has observed yet, so
        every message is delivered to exactly one member. The MAX query is
        re-evaluated inside the same locked connection as the poll+advance, so
        concurrent workers serialize correctly under SQLite's write lock.
        """
        if consumer_group is None:
            row = conn.execute(
                "SELECT cursor FROM subscriptions WHERE subscriber=? AND topic=?",
                (subscriber, topic),
            ).fetchone()
            return int(row["cursor"]) if row else 0
        row = conn.execute(
            "SELECT COALESCE(MAX(cursor), 0) AS m FROM subscriptions "
            "WHERE topic=? AND consumer_group=?",
            (topic, consumer_group),
        ).fetchone()
        return int(row["m"]) if row else 0

    # ---------------------------------------------------------- publishing
    def send_message(
        self,
        topic: str,
        payload: Any,
        *,
        sender: str,
        headers: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        consumer_group: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> int:
        """
        Publishes one message on ``topic``. Returns the assigned message id.

        Parameters
        ----------
        topic
            Routing key. Subscribers subscribe to a topic exactly.
        payload
            JSON-serializable body.
        sender
            Bot id (or ``"orchestrator"`` / ``"lollms-code"`` for system
            senders). Free-form.
        headers
            JSON-serializable routing metadata (message type, hop count, ...).
        correlation_id
            Correlation key a reply must echo to be matched.
        reply_to
            Sugar over ``correlation_id``: when provided, the reply subscriber
            for the eventual answer is registered automatically, so the
            responder just does ``send_message(topic, ..., correlation_id=token)``.
        consumer_group
            When set, the message is deliverable ONLY to queue subscribers of
            the same group (a pub/sub observer still sees it).
        ttl_seconds
            Optional lifetime. Expired rows are skipped and purged.
        """
        if not isinstance(topic, str) or not topic:
            raise BusError("topic must be a non-empty string.")
        if not isinstance(sender, str) or not sender:
            raise BusError("sender must be a non-empty string.")
        if reply_to is not None:
            if correlation_id is not None and correlation_id != reply_to:
                raise BusError(
                    "reply_to and correlation_id must agree when both are set."
                )
            correlation_id = reply_to

        now = self._now()
        expires_at = now + float(ttl_seconds) if ttl_seconds is not None else None
        payload_json = self._encode_json(payload, "payload")
        headers_json = self._encode_json(headers or {}, "headers")

        with self._acquire() as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO messages
                        (topic, sender, payload, headers, correlation_id,
                         consumer_group, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic,
                        sender,
                        payload_json,
                        headers_json,
                        correlation_id,
                        consumer_group,
                        now,
                        expires_at,
                    ),
                )
                msg_id = int(cur.lastrowid)
            except sqlite3.Error as e:
                raise BusError(f"send_message failed on topic {topic!r}: {e}") from e

        if reply_to is not None:
            # Idempotent — safe regardless of who created the subscription.
            # CRITICAL: from_now=True so the reply cursor starts PAST the
            # request itself. Otherwise wait_for_reply would match its own
            # outbound request as the "reply" (the request carries the same
            # correlation_id by construction).
            self._ensure_subscription(
                subscriber=self.reply_subscriber(reply_to),
                topic=topic,
                consumer_group=None,
                from_now=True,
            )
        return msg_id

    # ---------------------------------------------------------- subscribing
    def subscribe(
        self,
        subscriber: str,
        topic: str,
        *,
        consumer_group: Optional[str] = None,
        from_now: bool = False,
    ) -> Subscription:
        """
        Registers (or re-affirms) a durable subscription.

        ``from_now=True`` starts the cursor at the current max id so the
        subscriber sees only future messages. ``from_now=False`` (default)
        starts at 0 so the subscriber replays the retained backlog — the safe
        default for a bot restarting after a crash.

        Calling ``subscribe`` on an existing subscription is a no-op: it does
        NOT reset the cursor. Use :meth:`reset_cursor` to replay explicitly.
        """
        self._ensure_subscription(
            subscriber=subscriber,
            topic=topic,
            consumer_group=consumer_group,
            from_now=from_now,
        )
        sub = self.get_subscription(subscriber, topic)
        if sub is None:
            # Should be impossible after _ensure_subscription succeeded; treat
            # as a defensive error rather than returning a fake object.
            raise BusError(
                f"subscribe({subscriber!r}, {topic!r}) produced no record."
            )
        return sub

    def _ensure_subscription(
        self,
        *,
        subscriber: str,
        topic: str,
        consumer_group: Optional[str],
        from_now: bool,
    ) -> None:
        if not subscriber or not topic:
            raise BusError("subscriber and topic must be non-empty.")
        now = self._now()
        with self._acquire() as conn:
            try:
                row = conn.execute(
                    "SELECT 1 FROM subscriptions WHERE subscriber=? AND topic=?",
                    (subscriber, topic),
                ).fetchone()
                if row is not None:
                    return
                cursor = 0
                if from_now:
                    max_row = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) AS m FROM messages WHERE topic=?",
                        (topic,),
                    ).fetchone()
                    cursor = int(max_row["m"]) if max_row else 0
                conn.execute(
                    "INSERT INTO subscriptions "
                    "(subscriber, topic, consumer_group, cursor, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (subscriber, topic, consumer_group, cursor, now),
                )
            except sqlite3.Error as e:
                raise BusError(
                    f"subscribe({subscriber!r}, {topic!r}) failed: {e}"
                ) from e

    def unsubscribe(self, subscriber: str, topic: Optional[str] = None) -> int:
        """
        Removes a subscriber (all its topics when ``topic`` is None). Returns
        the number of rows removed. Idempotent.
        """
        with self._acquire() as conn:
            try:
                if topic is None:
                    cur = conn.execute(
                        "DELETE FROM subscriptions WHERE subscriber=?",
                        (subscriber,),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM subscriptions WHERE subscriber=? AND topic=?",
                        (subscriber, topic),
                    )
                return cur.rowcount or 0
            except sqlite3.Error as e:
                raise BusError(f"unsubscribe({subscriber!r}) failed: {e}") from e

    def reset_cursor(self, subscriber: str, topic: str, cursor: int = 0) -> None:
        """Sets a subscriber's cursor explicitly (e.g. to replay from a point)."""
        with self._acquire() as conn:
            try:
                conn.execute(
                    "UPDATE subscriptions SET cursor=? WHERE subscriber=? AND topic=?",
                    (int(cursor), subscriber, topic),
                )
            except sqlite3.Error as e:
                raise BusError(f"reset_cursor failed: {e}") from e

    def get_subscription(self, subscriber: str, topic: str) -> Optional[Subscription]:
        with self._acquire() as conn:
            row = conn.execute(
                "SELECT subscriber, topic, consumer_group, cursor, created_at "
                "FROM subscriptions WHERE subscriber=? AND topic=?",
                (subscriber, topic),
            ).fetchone()
        if row is None:
            return None
        return Subscription(
            subscriber=row["subscriber"],
            topic=row["topic"],
            consumer_group=row["consumer_group"],
            cursor=int(row["cursor"]),
            created_at=float(row["created_at"]),
        )

    # ------------------------------------------------------------- polling
    def poll_topic(
        self,
        subscriber: str,
        topic: str,
        *,
        max_batch: int = 50,
    ) -> List[BusMessage]:
        """
        Fetches up to ``max_batch`` messages for ``(subscriber, topic)`` and
        advances the cursor atomically.

        Semantics
        ---------
        Queue (``subscriptions.consumer_group IS NOT NULL``):
          Delivers messages whose ``consumer_group`` equals the subscription
          group. Pub/sub messages (``group IS NULL``) are invisible to a queue
          subscriber — the queue is a strict work-distribution channel.

        Pub/Sub (``consumer_group IS NULL``):
          Delivers all messages on the topic regardless of the message's
          ``consumer_group`` — a passive observer.

        Expired messages are skipped AND consumed via the cursor advance so
        they cannot block delivery. See the module docstring's cursor
        invariant.
        """
        if max_batch <= 0:
            return []
        now = self._now()
        with self._acquire() as conn:
            try:
                row = conn.execute(
                    "SELECT consumer_group, cursor FROM subscriptions "
                    "WHERE subscriber=? AND topic=?",
                    (subscriber, topic),
                ).fetchone()
                if row is None:
                    raise BusError(
                        f"No subscription for subscriber={subscriber!r} "
                        f"on topic={topic!r}. Call subscribe() first."
                    )
                group = row["consumer_group"]
                # For queue subs the effective cursor is the group's max cursor
                # (see _effective_cursor) — this is what gives competing-consumer
                # semantics. For pub/sub it's the subscriber's own cursor.
                cursor = self._effective_cursor(conn, subscriber, topic, group)

                # Compute the highest id visible to this subscription in the
                # SAME query as the fetch, so a single shot captures both the
                # delivered rows and any expired tail that should be skipped.
                if group is None:
                    head_row = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) AS m FROM messages "
                        "WHERE topic=? AND id>?",
                        (topic, cursor),
                    ).fetchone()
                    head = int(head_row["m"]) if head_row else 0
                    rows = conn.execute(
                        "SELECT id, topic, sender, payload, headers, "
                        "       correlation_id, created_at, expires_at "
                        "FROM messages "
                        "WHERE topic=? AND id>? "
                        "  AND (expires_at IS NULL OR expires_at>?) "
                        "ORDER BY id ASC LIMIT ?",
                        (topic, cursor, now, int(max_batch)),
                    ).fetchall()
                else:
                    head_row = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) AS m FROM messages "
                        "WHERE topic=? AND id>? AND consumer_group=?",
                        (topic, cursor, group),
                    ).fetchone()
                    head = int(head_row["m"]) if head_row else 0
                    rows = conn.execute(
                        "SELECT id, topic, sender, payload, headers, "
                        "       correlation_id, created_at, expires_at "
                        "FROM messages "
                        "WHERE topic=? AND id>? AND consumer_group=? "
                        "  AND (expires_at IS NULL OR expires_at>?) "
                        "ORDER BY id ASC LIMIT ?",
                        (topic, cursor, group, now, int(max_batch)),
                    ).fetchall()

                if not rows and head == cursor:
                    # Nothing pending; leave cursor untouched.
                    return []

                # Advance cursor to the max id OBSERVED in this scan. This is
                # always >= rows[-1].id (delivered) and >= any expired tail
                # above the boundary because ``head`` is computed from the raw
                # message table with no expiry filter.
                #
                # IMPORTANT: only advance when *every* delivery-eligible row
                # up to ``head`` was consumed, OR when ``head`` itself is
                # eligible. If ``rows`` hit the LIMIT while head-bound messages
                # remain above, advancing to ``head`` would silently LOSE those
                # messages. So: advance to ``rows[-1].id`` when LIMIT hit;
                # otherwise advance to ``head`` (which covers skipped expired
                # rows at the tail).
                limit_hit = len(rows) >= int(max_batch)
                new_cursor = int(rows[-1]["id"]) if rows else cursor
                if not limit_hit:
                    new_cursor = max(new_cursor, head)
                conn.execute(
                    "UPDATE subscriptions SET cursor=? "
                    "WHERE subscriber=? AND topic=?",
                    (new_cursor, subscriber, topic),
                )
            except sqlite3.Error as e:
                raise BusError(
                    f"poll_topic({subscriber!r}, {topic!r}) failed: {e}"
                ) from e

        return [self._row_to_message(r) for r in rows]

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> BusMessage:
        return BusMessage(
            id=int(row["id"]),
            topic=row["topic"],
            sender=row["sender"],
            payload=MessageBus._decode_json(row["payload"], "payload"),
            headers=MessageBus._decode_json(row["headers"], "headers", default={}),
            correlation_id=row["correlation_id"],
            created_at=float(row["created_at"]),
            expires_at=(
                float(row["expires_at"]) if row["expires_at"] is not None else None
            ),
        )

    def wait_for(
        self,
        subscriber: str,
        topic: str,
        *,
        timeout: float = 30.0,
        max_batch: int = 1,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[BusMessage]:
        """
        Blocking poll loop. Returns as soon as at least one message arrives,
        or ``[]`` on timeout. ``should_stop`` lets a supervisor abandon the
        wait when the target bot dies.

        Never raises on timeout — that is a normal, non-exceptional outcome
        for the supervisor pattern. Raises BusError only on backend failures.
        """
        deadline = self._now() + max(0.0, float(timeout))
        while True:
            batch = self.poll_topic(subscriber, topic, max_batch=max_batch)
            if batch:
                return batch
            if should_stop is not None:
                try:
                    if should_stop():
                        return []
                except Exception:
                    return []
            if self._now() >= deadline:
                return []
            time.sleep(self._POLL_SLEEP_SECONDS)

    # --------------------------------------------------- request/reply sugar
    def wait_for_reply(
        self,
        token: str,
        *,
        topic: str,
        timeout: float = 30.0,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[BusMessage]:
        """
        Waits for the message on ``topic`` whose ``correlation_id`` is
        ``token``. Returns that message or ``None`` on timeout.

        Non-matching replies that arrive on the same topic are consumed (their
        cursor is advanced) but discarded — only the matching one is returned.
        This keeps memory bounded when many replies share a topic.
        """
        subscriber = self.reply_subscriber(token)
        # from_now=True is REQUIRED here too (defence-in-depth): even if
        # send_message(..., reply_to=token) was never called, we want the
        # reply cursor positioned at the current head so we only see FUTURE
        # messages — never any pre-existing request tagged with this token.
        self._ensure_subscription(
            subscriber=subscriber,
            topic=topic,
            consumer_group=None,
            from_now=True,
        )
        deadline = self._now() + max(0.0, float(timeout))
        while True:
            batch = self.poll_topic(subscriber, topic, max_batch=50)
            for msg in batch:
                if msg.correlation_id == token:
                    return msg
            if should_stop is not None:
                try:
                    if should_stop():
                        return None
                except Exception:
                    return None
            if self._now() >= deadline:
                return None
            if not batch:
                time.sleep(self._POLL_SLEEP_SECONDS)

    # --------------------------------------------------------- introspection
    def pending_count(self, subscriber: str, topic: str) -> int:
        """
        Counts messages a subscriber would receive RIGHT NOW (id > cursor,
        group-filtered, not expired). Use it for backpressure decisions.
        """
        now = self._now()
        with self._acquire() as conn:
            row = conn.execute(
                "SELECT consumer_group, cursor FROM subscriptions "
                "WHERE subscriber=? AND topic=?",
                (subscriber, topic),
            ).fetchone()
            if row is None:
                return 0
            group = row["consumer_group"]
            cursor = self._effective_cursor(conn, subscriber, topic, group)
            if group is None:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages "
                    "WHERE topic=? AND id>? "
                    "  AND (expires_at IS NULL OR expires_at>?)",
                    (topic, cursor, now),
                ).fetchone()
            else:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages "
                    "WHERE topic=? AND id>? AND consumer_group=? "
                    "  AND (expires_at IS NULL OR expires_at>?)",
                    (topic, cursor, group, now),
                ).fetchone()
            return int(count_row["n"]) if count_row else 0

    def list_topics(self) -> List[Dict[str, Any]]:
        """
        One entry per topic: ``{"topic", "messages", "subscribers", "oldest",
        "newest"}``. Dashboard/health surface.
        """
        with self._acquire() as conn:
            rows = conn.execute(
                "SELECT topic, COUNT(*) AS n, "
                "  MIN(created_at) AS oldest, MAX(created_at) AS newest "
                "FROM messages GROUP BY topic ORDER BY topic"
            ).fetchall()
            counts = {r["topic"]: r for r in rows}
            sub_rows = conn.execute(
                "SELECT topic, COUNT(*) AS s FROM subscriptions GROUP BY topic"
            ).fetchall()
            sub_counts = {r["topic"]: int(r["s"]) for r in sub_rows}
        topics = set(counts) | set(sub_counts)
        out: List[Dict[str, Any]] = []
        for t in sorted(topics):
            c = counts.get(t)
            out.append({
                "topic": t,
                "messages": int(c["n"]) if c else 0,
                "subscribers": sub_counts.get(t, 0),
                "oldest": float(c["oldest"]) if c and c["oldest"] is not None else None,
                "newest": float(c["newest"]) if c and c["newest"] is not None else None,
            })
        return out

    # ----------------------------------------------------------- janitor
    def expire_old(self, *, max_age_seconds: Optional[float] = None) -> int:
        """
        Purges messages whose ``expires_at`` has passed (or, given
        ``max_age_seconds``, rows older than that regardless of per-message
        TTL). Returns the number of rows deleted.

        Safe to call from any bot. Deleting rows is invisible to subscribers
        whose cursor is already past them (guaranteed by the cursor
        invariant); a message id below a live cursor cannot be re-delivered.
        """
        now = self._now()
        with self._acquire() as conn:
            try:
                if max_age_seconds is not None:
                    cutoff = now - float(max_age_seconds)
                    cur = conn.execute(
                        "DELETE FROM messages "
                        "WHERE (expires_at IS NOT NULL AND expires_at<=?) "
                        "   OR created_at<?",
                        (now, cutoff),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM messages "
                        "WHERE expires_at IS NOT NULL AND expires_at<=?",
                        (now,),
                    )
                return cur.rowcount or 0
            except sqlite3.Error as e:
                raise BusError(f"expire_old failed: {e}") from e

    # --------------------------------------------------------- context mgr
    def __enter__(self) -> "MessageBus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
