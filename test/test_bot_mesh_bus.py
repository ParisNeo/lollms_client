
"""
Unit tests for the lollms-bot mesh message bus (phase 2).

Covers the contract the supervisor, workflow router, and mesh tools will rely
on:

  * registry CRUD + discovery (phase 1 semantics, re-verified);
  * pub/sub fan-out (each subscriber sees every message);
  * queue competing-consumer semantics (each message to exactly one member);
  * cursor persistence across close/reopen (durability);
  * request/reply correlation (matching reply, timeout, non-matching skip);
  * TTL expiry + janitor purge;
  * backpressure (pending_count, max_batch);
  * cursor invariant under an expired message inside a batch.

The tests deliberately use a temporary layout directory per test so no state
leaks between them. No network, no LLM, no threads — pure transport semantics.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lollms_client.apps.lollms_bot.mesh import (
    BotInstance,
    BotSpec,
    BotStatus,
    MeshLayout,
    MeshRegistry,
)
from lollms_client.apps.lollms_bot.mesh.bus import BusError, BusMessage, MessageBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mesh_layout(tmp_path: Path) -> MeshLayout:
    layout = MeshLayout.under(tmp_path, "test_mesh")
    layout.materialize()
    return layout


@pytest.fixture()
def bus(mesh_layout: MeshLayout):
    b = MessageBus(mesh_layout)
    try:
        yield b
    finally:
        b.close()


# ---------------------------------------------------------------------------
# Registry (phase 1 regression coverage)
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_discover(self, mesh_layout):
        reg = MeshRegistry(mesh_layout)
        spec = BotSpec(
            bot_id="intake",
            tags=["email", "front"],
            capabilities={"enable_computer_use": True},
        )
        BotInstance.create(reg, spec)
        assert reg.list_bots()["intake"]["status"] == BotStatus.PENDING
        assert len(reg.find_by_tag("email")) == 1
        assert len(reg.find_by_capability("enable_computer_use")) == 1
        assert reg.running_bots() == []

    def test_status_transition_and_remove_guard(self, mesh_layout):
        reg = MeshRegistry(mesh_layout)
        inst = BotInstance.create(reg, BotSpec(bot_id="relay"))
        reg.update_status("relay", status=BotStatus.RUNNING)
        assert inst.is_running()
        with pytest.raises(RuntimeError):
            inst.remove()   # cannot remove a running bot
        reg.update_status("relay", status=BotStatus.STOPPED)
        inst.remove()
        assert reg.get_bot("relay") is None


# ---------------------------------------------------------------------------
# Pub/Sub
# ---------------------------------------------------------------------------

class TestPubSub:
    def test_single_fanout(self, bus: MessageBus):
        bus.subscribe("a", "news")
        bus.send_message("news", {"headline": "h1"}, sender="pub")
        msgs = bus.poll_topic("a", "news")
        assert len(msgs) == 1
        assert msgs[0].payload == {"headline": "h1"}
        assert bus.poll_topic("a", "news") == []

    def test_each_subscriber_sees_every_message(self, bus: MessageBus):
        bus.subscribe("a", "updates")
        bus.subscribe("b", "updates")
        for i in range(3):
            bus.send_message("updates", {"i": i}, sender="pub")
        a = [m.payload["i"] for m in bus.poll_topic("a", "updates")]
        b = [m.payload["i"] for m in bus.poll_topic("b", "updates")]
        assert a == [0, 1, 2]
        assert b == [0, 1, 2]

    def test_max_batch_caps_delivery(self, bus: MessageBus):
        bus.subscribe("a", "queue")
        for i in range(10):
            bus.send_message("queue", {"i": i}, sender="pub")
        first = bus.poll_topic("a", "queue", max_batch=3)
        assert [m.payload["i"] for m in first] == [0, 1, 2]
        rest = bus.poll_topic("a", "queue", max_batch=100)
        assert [m.payload["i"] for m in rest] == [3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Queue (competing consumers)
# ---------------------------------------------------------------------------

class TestQueue:
    def test_message_to_exactly_one_member(self, bus: MessageBus):
        bus.subscribe("w1", "tasks", consumer_group="workers")
        bus.subscribe("w2", "tasks", consumer_group="workers")
        for i in range(4):
            bus.send_message(
                "tasks", {"i": i}, sender="pub", consumer_group="workers"
            )
        w1 = [m.payload["i"] for m in bus.poll_topic("w1", "tasks")]
        w2 = [m.payload["i"] for m in bus.poll_topic("w2", "tasks")]
        # Competing-consumer invariant: every task delivered to EXACTLY one
        # worker (no overlap), and together the two cover the whole set.
        assert len(set(w1) & set(w2)) == 0, (
            f"queue overlap: w1={w1}, w2={w2}"
        )
        assert sorted(w1 + w2) == [0, 1, 2, 3]

    def test_queue_alternating_polls_keep_order(self, bus: MessageBus):
        bus.subscribe("w1", "tasks", consumer_group="workers")
        bus.subscribe("w2", "tasks", consumer_group="workers")
        for i in range(6):
            bus.send_message(
                "tasks", {"i": i}, sender="pub", consumer_group="workers"
            )
        # First worker drains the first 2, then the second worker drains 2,
        # then the first drains the rest. No message may be seen twice.
        seen = []
        seen += [m.payload["i"] for m in bus.poll_topic("w1", "tasks", max_batch=2)]
        seen += [m.payload["i"] for m in bus.poll_topic("w2", "tasks", max_batch=2)]
        seen += [m.payload["i"] for m in bus.poll_topic("w1", "tasks", max_batch=2)]
        assert seen == [0, 1, 2, 3, 4, 5], f"unexpected order or dups: {seen}"

    def test_queue_after_drain_returns_empty(self, bus: MessageBus):
        bus.subscribe("w1", "tasks", consumer_group="workers")
        bus.subscribe("w2", "tasks", consumer_group="workers")
        bus.send_message("tasks", {"i": 0}, sender="pub", consumer_group="workers")
        w1 = bus.poll_topic("w1", "tasks")
        w2 = bus.poll_topic("w2", "tasks")
        assert len(w1) == 1
        assert w2 == []                      # already claimed by w1
        # But a NEW message after both drained is picked up by the next poller.
        bus.send_message("tasks", {"i": 1}, sender="pub", consumer_group="workers")
        w3 = bus.poll_topic("w1", "tasks")
        w4 = bus.poll_topic("w2", "tasks")
        assert len(w3) + len(w4) == 1
        assert (w3 + w4)[0].payload["i"] == 1

    def test_queue_subscriber_ignores_ungrouped_messages(self, bus: MessageBus):
        bus.subscribe("w1", "tasks", consumer_group="workers")
        bus.send_message("tasks", {"n": 1}, sender="pub")   # no group
        assert bus.poll_topic("w1", "tasks") == []


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------

class TestDurability:
    def test_cursor_persists_across_reopen(self, mesh_layout: MeshLayout):
        b1 = MessageBus(mesh_layout)
        b1.subscribe("reader", "feed")
        b1.send_message("feed", {"x": 1}, sender="pub")
        first = b1.poll_topic("reader", "feed")
        assert len(first) == 1
        b1.close()

        b2 = MessageBus(mesh_layout)
        try:
            b2.subscribe("reader", "feed")   # is a no-op; does NOT reset cursor
            b2.send_message("feed", {"x": 2}, sender="pub")
            second = b2.poll_topic("reader", "feed")
            assert [m.payload["x"] for m in second] == [2]
        finally:
            b2.close()

    def test_backlog_replay_from_zero(self, mesh_layout: MeshLayout):
        b1 = MessageBus(mesh_layout)
        b1.send_message("feed", {"x": 1}, sender="pub")
        b1.send_message("feed", {"x": 2}, sender="pub")
        b1.close()

        b2 = MessageBus(mesh_layout)
        try:
            # Subscribe AFTER the messages exist with default from_now=False.
            b2.subscribe("late", "feed")
            replay = b2.poll_topic("late", "feed")
            assert [m.payload["x"] for m in replay] == [1, 2]
        finally:
            b2.close()

    def test_from_now_skips_history(self, mesh_layout: MeshLayout):
        b = MessageBus(mesh_layout)
        try:
            b.send_message("feed", {"x": 1}, sender="pub")
            b.subscribe("late", "feed", from_now=True)
            b.send_message("feed", {"x": 2}, sender="pub")
            replay = b.poll_topic("late", "feed")
            assert [m.payload["x"] for m in replay] == [2]
        finally:
            b.close()


# ---------------------------------------------------------------------------
# Request / reply
# ---------------------------------------------------------------------------

class TestRequestReply:
    def test_matching_reply(self, bus: MessageBus):
        token = "req-42"
        # Producer registers reply subscriber via reply_to.
        bus.send_message(
            "rpc", {"q": "ping"}, sender="client", reply_to=token, ttl_seconds=5
        )
        # Responder (in the same process for the test) answers.
        bus.send_message(
            "rpc", {"a": "pong"}, sender="server", correlation_id=token
        )
        reply = bus.wait_for_reply(token, topic="rpc", timeout=1.0)
        assert reply is not None
        assert reply.payload == {"a": "pong"}
        assert reply.correlation_id == token

    def test_timeout_returns_none(self, bus: MessageBus):
        token = "req-timeout"
        bus.send_message(
            "rpc", {"q": "ping"}, sender="client", reply_to=token, ttl_seconds=5
        )
        reply = bus.wait_for_reply(token, topic="rpc", timeout=0.3)
        assert reply is None

    def test_non_matching_replies_are_skipped(self, bus: MessageBus):
        token = "req-A"
        bus.send_message("rpc", {"q": "A"}, sender="c", reply_to=token, ttl_seconds=5)
        # Someone else's reply arrives first, on the same topic.
        bus.send_message("rpc", {"a": "other"}, sender="s", correlation_id="req-B")
        # Our answer arrives second.
        bus.send_message("rpc", {"a": "A"}, sender="s", correlation_id=token)
        reply = bus.wait_for_reply(token, topic="rpc", timeout=1.0)
        assert reply is not None
        assert reply.correlation_id == token
        assert reply.payload == {"a": "A"}

    def test_reply_subscriber_starts_past_request(self, bus: MessageBus):
        # Publishing the request then reading the raw reply subscription
        # must NOT surface the request itself.
        token = "req-boundary"
        bus.send_message(
            "rpc", {"q": "ping"}, sender="c", reply_to=token, ttl_seconds=5
        )
        subscriber = MessageBus.reply_subscriber(token)
        later = bus.poll_topic(subscriber, "rpc")
        assert later == [], (
            "reply subscriber re-read its own request; from_now=True missing"
        )
        # An actual reply is delivered.
        bus.send_message(
            "rpc", {"a": "pong"}, sender="s", correlation_id=token
        )
        got = bus.poll_topic(subscriber, "rpc")
        assert len(got) == 1
        assert got[0].payload == {"a": "pong"}


# ---------------------------------------------------------------------------
# TTL / Janitor
# ---------------------------------------------------------------------------

class TestTTL:
    def test_expired_messages_are_not_delivered(self, bus: MessageBus):
        bus.subscribe("a", "ephemeral")
        bus.send_message("ephemeral", {"x": 1}, sender="pub", ttl_seconds=0.05)
        time.sleep(0.1)
        assert bus.poll_topic("a", "ephemeral") == []

    def test_expired_rows_between_live_rows_do_not_stall(self, bus: MessageBus):
        bus.subscribe("a", "mixed")
        bus.send_message("mixed", {"x": 0}, sender="pub")                # live
        bus.send_message("mixed", {"x": 1}, sender="pub", ttl_seconds=0.01)  # expires
        bus.send_message("mixed", {"x": 2}, sender="pub")                # live
        time.sleep(0.05)
        delivered = [m.payload["x"] for m in bus.poll_topic("a", "mixed")]
        assert delivered == [0, 2]

    def test_expire_old_purges(self, bus: MessageBus):
        bus.send_message("t", {"x": 1}, sender="pub", ttl_seconds=0.01)
        time.sleep(0.05)
        removed = bus.expire_old()
        assert removed == 1
        assert bus.list_topics() == [] or all(
            t["messages"] == 0 for t in bus.list_topics()
        )


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_pending_count(self, bus: MessageBus):
        bus.subscribe("a", "work")
        assert bus.pending_count("a", "work") == 0
        for i in range(5):
            bus.send_message("work", {"i": i}, sender="pub")
        assert bus.pending_count("a", "work") == 5
        bus.poll_topic("a", "work", max_batch=2)
        assert bus.pending_count("a", "work") == 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_non_json_payload_rejected(self, bus: MessageBus):
        with pytest.raises(BusError):
            bus.send_message("t", {"x": object()}, sender="pub")

    def test_poll_unknown_subscriber_raises(self, bus: MessageBus):
        with pytest.raises(BusError):
            bus.poll_topic("ghost", "t")

    def test_closed_bus_raises(self, mesh_layout: MeshLayout):
        b = MessageBus(mesh_layout)
        b.close()
        with pytest.raises(BusError):
            b.send_message("t", {"x": 1}, sender="pub")
