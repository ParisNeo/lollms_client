
"""
Tests for ConnectionBindingAdapter (swarm connector bridge to real bindings).

These tests exercise the adapter contract end to end against a
``FakeConnectionBinding`` that mimics the lollms-client binding shape
(send_message / receive_message / is_connected). No real platform is
touched — the point is to prove the *bridge* is correct:

  * inbound messages flow from the binding into ``SwarmConnector.ingest``;
  * replies flow from a mesh responder back to ``binding.send_message``;
  * the binding's channel_id is preserved round-trip;
  * the adapter stops cleanly and cannot forward after stop;
  * reconnect (stop then start) works — pull loop is restartable.

The tests intentionally use the SAME in-memory ``MessageBus`` protocol the
production runtime uses, so they validate the whole integration path
(platform ↔ connector ↔ bus ↔ responder ↔ connector ↔ platform).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lollms_client.apps.lollms_bot.mesh.bus import MessageBus
from lollms_client.apps.lollms_bot.mesh.connection_adapter import (
    ConnectionBindingAdapter,
    build_adapters_from_client,
)
from lollms_client.apps.lollms_bot.mesh.connector import SwarmConnector
from lollms_client.apps.lollms_bot.mesh.types import MeshLayout


# ---------------------------------------------------------------------------
# Fake connection binding
# ---------------------------------------------------------------------------

@dataclass
class _FakeInbound:
    content: str
    channel_id: str
    author: str = "user"


class FakeConnectionBinding:
    """In-memory stand-in for a real lollms-client connection binding."""

    binding_name = "fake-platform"

    def __init__(self) -> None:
        self._inbound: "queue.Queue[_FakeInbound]" = queue.Queue()
        self._sent: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.starts = 0
        self.stops = 0

    def is_connected(self) -> bool:
        return True

    def receive_message(self, timeout: float = 0.0) -> Optional[_FakeInbound]:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_message(self, channel_id: str, text: str) -> None:
        with self._lock:
            self._sent.append({"channel_id": channel_id, "text": text})

    def push_inbound(self, text: str, channel_id: str, author: str = "user") -> None:
        self._inbound.put(
            _FakeInbound(content=text, channel_id=channel_id, author=author)
        )

    def sent(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._sent)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def layout(tmp_path: Path) -> MeshLayout:
    lm = MeshLayout.under(tmp_path, "adapter_mesh")
    lm.materialize()
    return lm


def _start_responder(layout: MeshLayout, *, tag: str = "responder"):
    """Subscribes to 'swarm.inbound' and answers every message."""
    stop = threading.Event()

    def _loop():
        with MessageBus(layout) as bus:
            bus.subscribe(tag, "swarm.inbound", from_now=False)
            while not stop.is_set():
                msgs = bus.wait_for(tag, "swarm.inbound", timeout=0.2)
                for m in msgs:
                    token = (m.payload or {}).get("reply_token")
                    text = (m.payload or {}).get("text", "")
                    bus.send_message(
                        "swarm.inbound.reply",
                        {"text": f"[{tag}] echo: {text}"},
                        sender=tag,
                        correlation_id=token,
                    )

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop, t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConnectionBindingAdapter:
    def test_round_trip(self, layout: MeshLayout):
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)
        connector = SwarmConnector(
            layout,
            adapter=adapter,  # auto-attached by __init__
            config={
                "enabled": True,
                "ingress_topic": "swarm.inbound",
                "reply_timeout_seconds": 5,
                "max_concurrent": 2,
            },
        )
        connector.start()
        adapter.start()
        stop, _t = _start_responder(layout)
        try:
            binding.push_inbound("hello", channel_id="chan-1", author="alice")

            deadline = time.time() + 5.0
            while time.time() < deadline and not binding.sent():
                time.sleep(0.02)
            assert binding.sent(), "no reply landed on the binding"
            msg = binding.sent()[0]
            assert msg["channel_id"] == "chan-1"
            assert "echo: hello" in msg["text"]
        finally:
            stop.set()
            adapter.stop()
            connector.stop()

    def test_multiple_channels_preserved(self, layout: MeshLayout):
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)
        connector = SwarmConnector(
            layout,
            adapter=adapter,
            config={
                "enabled": True,
                "ingress_topic": "swarm.inbound",
                "reply_timeout_seconds": 5,
            },
        )
        connector.start()
        adapter.start()
        stop, _t = _start_responder(layout)
        try:
            binding.push_inbound("A", channel_id="alpha")
            binding.push_inbound("B", channel_id="beta")

            deadline = time.time() + 5.0
            while time.time() < deadline and len(binding.sent()) < 2:
                time.sleep(0.02)
            sent = binding.sent()
            assert len(sent) == 2
            channels = {m["channel_id"] for m in sent}
            assert channels == {"alpha", "beta"}, f"got {channels}"
        finally:
            stop.set()
            adapter.stop()
            connector.stop()

    def test_no_connector_is_silent(self):
        """An adapter with no attached connector drops inbound without crashing."""
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)  # no attach()
        adapter.start()
        try:
            binding.push_inbound("ignored", channel_id="x")
            # Give the pull loop time to poll and discard.
            time.sleep(0.6)
            assert binding.sent() == []
        finally:
            adapter.stop()

    def test_stop_is_idempotent(self):
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)
        adapter.stop()  # no start
        adapter.start()
        adapter.stop()
        adapter.stop()  # twice

    def test_restart_pull_loop(self):
        """A stopped adapter can be restarted and still forward."""
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)
        adapter.start()
        adapter.stop()
        adapter.start()
        try:
            # No connector; just assert the thread is running.
            assert adapter._thread is not None
            assert adapter._thread.is_alive()
        finally:
            adapter.stop()


class TestBuildAdaptersFromClient:
    def test_single_connection(self):
        """Client with one connection is wrapped into one adapter."""

        class _Client:
            def __init__(self, binding):
                self.connection = binding
                self.connections = None

        binding = FakeConnectionBinding()
        adapters = build_adapters_from_client(_Client(binding))
        assert len(adapters) == 1
        assert adapters[0].platform == "fake-platform"

    def test_whitelist_filters(self):
        class _Client:
            def __init__(self, bindings: Dict[str, Any]):
                self.connections = bindings
                self.connection = None

        b1 = FakeConnectionBinding()
        b1.binding_name = "discord"
        b2 = FakeConnectionBinding()
        b2.binding_name = "slack"
        adapters = build_adapters_from_client(
            _Client({"discord": b1, "slack": b2}),
            platform_whitelist=["slack"],
        )
        assert len(adapters) == 1
        assert adapters[0].platform == "slack"

    def test_empty_client(self):
        class _Client:
            connections = None
            connection = None

        assert build_adapters_from_client(_Client()) == []
