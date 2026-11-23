
"""
Tests for the lollms-bot swarm connector.

The connector bridges external platform messages onto the mesh bus and routes
replies back to the platform adapter. These tests exercise the *transport*
contract without any real platform:

  * ingest() publishes on the ingress topic with a correlation token;
  * an "orchestrator" (a plain bus subscriber in the test) reads the ingress
    message and publishes a reply on ingress.reply carrying the token;
  * the connector's adapter receives that reply text on its publish_reply
    hook, with the correct channel_id;
  * a timed-out request does not call publish_reply with a bogus answer when
    the connector is stopping, but DOES report a timeout when running;
  * ingest() on a non-running connector raises a clear error.

An in-memory test adapter records every publish_reply call so assertions are
deterministic. No threads are left running across tests: each test stops the
connector in a finally block.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lollms_client.apps.lollms_bot.mesh.bus import MessageBus
from lollms_client.apps.lollms_bot.mesh.connector import (
    InboundMessage,
    SwarmConnector,
    SwarmConnectorError,
    SwarmPlatformAdapter,
)
from lollms_client.apps.lollms_bot.mesh.types import MeshLayout


# ---------------------------------------------------------------------------
# Fixtures + test adapter
# ---------------------------------------------------------------------------

@pytest.fixture()
def mesh_layout(tmp_path: Path) -> MeshLayout:
    layout = MeshLayout.under(tmp_path, "swarmmesh")
    layout.materialize()
    return layout


class RecordingAdapter(SwarmPlatformAdapter):
    """Records every reply the connector sends back, in arrival order."""

    platform = "test"

    def __init__(self) -> None:
        self.replies: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def publish_reply(
        self,
        channel_id: Optional[str],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        with self._lock:
            self.replies.append(
                {"channel_id": channel_id, "text": text, "metadata": metadata}
            )

    def wait_for_replies(self, n: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Polls until ``n`` replies have been recorded or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self.replies) >= n:
                    return list(self.replies)
            time.sleep(0.02)
        with self._lock:
            return list(self.replies)


def _make_config(**overrides) -> Dict[str, Any]:
    cfg = {
        "enabled": True,
        "ingress_topic": "swarm.inbound",
        "reply_timeout_seconds": 5,
        "max_concurrent": 2,
        "announce_identity": True,
        "connection_name": "test",
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSwarmConnector:
    def test_reply_round_trip(self, mesh_layout: MeshLayout):
        """
        Full happy path: ingest -> ingress message -> orchestrator reply ->
        adapter receives the answer, with the original channel id preserved.
        """
        adapter = RecordingAdapter()
        conn = SwarmConnector(mesh_layout, adapter=adapter, config=_make_config())
        conn.start()
        try:
            # Stand in for the orchestrator: consume ingress and answer.
            with MessageBus(mesh_layout) as orch_bus:
                orch_bus.subscribe("orch", "swarm.inbound", from_now=False)

                token = conn.ingest(
                    InboundMessage(
                        text="hello mesh",
                        sender="alice",
                        platform="test",
                        channel_id="chan-42",
                    )
                )

                # Orchestrator sees the message and replies with the token.
                msgs = orch_bus.wait_for("orch", "swarm.inbound", timeout=3.0)
                assert len(msgs) == 1
                got = msgs[0]
                assert got.payload["text"] == "hello mesh"
                assert got.payload["reply_token"] == token

                orch_bus.send_message(
                    "swarm.inbound.reply",
                    {"text": "hi alice, the mesh heard you"},
                    sender="swarm-orchestrator",
                    correlation_id=token,
                )

            replies = adapter.wait_for_replies(1, timeout=5.0)
            assert len(replies) == 1
            assert replies[0]["channel_id"] == "chan-42"
            assert "hi alice" in replies[0]["text"]
        finally:
            conn.stop()

    def test_ingest_requires_running(self, mesh_layout: MeshLayout):
        conn = SwarmConnector(mesh_layout, adapter=RecordingAdapter(), config=_make_config())
        # Not started yet.
        with pytest.raises(SwarmConnectorError):
            conn.ingest(InboundMessage(text="x", sender="y"))

    def test_timeout_reports_back_to_platform(self, mesh_layout: MeshLayout):
        """
        No orchestrator answers -> after reply_timeout_seconds the adapter
        receives a timeout notice (connector still running).
        """
        adapter = RecordingAdapter()
        conn = SwarmConnector(
            mesh_layout,
            adapter=adapter,
            config=_make_config(reply_timeout_seconds=1),
        )
        conn.start()
        try:
            conn.ingest(
                InboundMessage(
                    text="anyone there?",
                    sender="bob",
                    platform="test",
                    channel_id="chan-timeout",
                )
            )
            replies = adapter.wait_for_replies(1, timeout=4.0)
            assert len(replies) == 1
            assert replies[0]["channel_id"] == "chan-timeout"
            assert "no reply" in replies[0]["text"].lower()
        finally:
            conn.stop()

    def test_error_reply_surfaces_to_platform(self, mesh_layout: MeshLayout):
        """
        A responder that publishes on ``swarm.inbound.error`` must reach the
        adapter with the error text (NOT a generic timeout).
        """
        adapter = RecordingAdapter()
        conn = SwarmConnector(
            mesh_layout,
            adapter=adapter,
            config=_make_config(reply_timeout_seconds=3),
        )
        conn.start()
        try:
            with MessageBus(mesh_layout) as orch_bus:
                orch_bus.subscribe("orch-err", "swarm.inbound", from_now=False)
                conn.ingest(
                    InboundMessage(
                        text="will fail", sender="alice", platform="test",
                        channel_id="chan-err",
                    )
                )
                msgs = orch_bus.wait_for("orch-err", "swarm.inbound", timeout=2.0)
                assert len(msgs) == 1
                token = msgs[0].payload["reply_token"]
                orch_bus.send_message(
                    "swarm.inbound.error",
                    {"error": "model timed out", "text": "(agent error) model timed out"},
                    sender="orch-err",
                    correlation_id=token,
                )
            replies = adapter.wait_for_replies(1, timeout=5.0)
            assert len(replies) == 1
            assert replies[0]["channel_id"] == "chan-err"
            assert "model timed out" in replies[0]["text"]
        finally:
            conn.stop()

    def test_disabled_without_adapter_refuses_start(self, mesh_layout: MeshLayout):
        conn = SwarmConnector(
            mesh_layout, adapter=None, config=_make_config(enabled=False)
        )
        with pytest.raises(SwarmConnectorError):
            conn.start()

    def test_status_snapshot(self, mesh_layout: MeshLayout):
        adapter = RecordingAdapter()
        conn = SwarmConnector(mesh_layout, adapter=adapter, config=_make_config())
        conn.start()
        try:
            status = conn.status()
            assert status["running"] is True
            assert status["ingress_topic"] == "swarm.inbound"
            assert status["adapter_attached"] is True
            assert status["adapter_platform"] == "test"
        finally:
            conn.stop()
        assert conn.status()["running"] is False
