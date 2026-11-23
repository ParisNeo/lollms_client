
"""
Example 3 — Swarm connector: bridge an external platform onto the mesh.

The swarm connector lets a user on Discord / Slack / lollms.com reach the
mesh. This example simulates that end-to-end WITHOUT a real platform:

  * a fake ``RecordingAdapter`` stands in for what a Discord connection
    binding would provide (it needs only ``publish_reply``);
  * an "orchestrator bot" is simulated by a plain bus subscriber that
    waits for ingress traffic and answers on the reply topic;
  * the hub-side connector ingests a message, correlating the reply back to
    the platform adapter — proving the full round trip.

This is the exact code path the hub's ``POST /api/swarm/ingest`` and its
``SwarmConnector`` instance use in production.

Run:
    python examples/mesh/03_swarm_connector.py
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lollms_client.apps.lollms_bot.mesh import (
    InboundMessage,
    MeshLayout,
    MessageBus,
    SwarmConnector,
    SwarmPlatformAdapter,
)


class ConsoleAdapter(SwarmPlatformAdapter):
    """Stand-in for a Discord/Slack adapter: prints every reply it receives."""

    platform = "console"

    def __init__(self) -> None:
        self.received: list[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def publish_reply(
        self, channel_id: Optional[str], text: str, metadata: Dict[str, Any]
    ) -> None:
        with self._lock:
            self.received.append(
                {"channel": channel_id, "text": text, "metadata": metadata}
            )
            print(f"[platform] ⇠ #{channel_id}: {text}")


def fake_orchestrator_bot(layout: MeshLayout, stop: threading.Event) -> None:
    """
    Simulates the Swarm Orchestrator agent: reads ``swarm.inbound`` and
    replies on ``swarm.inbound.reply`` with the same correlation id.
    """
    with MessageBus(layout) as bus:
        bus.subscribe("orch-demo", "swarm.inbound", from_now=False)
        print("[orchestrator] listening on swarm.inbound ...")
        while not stop.is_set():
            msgs = bus.wait_for("orch-demo", "swarm.inbound", timeout=0.3)
            for m in msgs:
                text = (m.payload or {}).get("text", "")
                token = (m.payload or {}).get("reply_token")
                print(f"[orchestrator] ⇢ received: {text!r}  token={token[:8]}...")
                bus.send_message(
                    "swarm.inbound.reply",
                    {"text": f"Echo from the mesh: {text.upper()}"},
                    sender="orch-demo",
                    correlation_id=token,
                )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        layout = MeshLayout.under(Path(tmpdir), "swarm_mesh")
        layout.materialize()

        adapter = ConsoleAdapter()
        connector = SwarmConnector(
            layout,
            adapter=adapter,
            config={
                "enabled": True,
                "ingress_topic": "swarm.inbound",
                "reply_timeout_seconds": 5,
                "max_concurrent": 2,
            },
        )
        connector.start()

        stop = threading.Event()
        orch = threading.Thread(
            target=fake_orchestrator_bot, args=(layout, stop), daemon=True
        )
        orch.start()
        try:
            # 1) A user on "the platform" sends a message.
            print("[platform] ⇢ alice: summarize the Q3 report")
            token = connector.ingest(
                InboundMessage(
                    text="summarize the Q3 report",
                    sender="alice",
                    platform="console",
                    channel_id="general",
                )
            )
            print(f"[connector] dispatched, reply_token={token[:8]}...")

            # Wait for the adapter to record exactly one reply (bounded).
            deadline = time.time() + 5.0
            while time.time() < deadline and not adapter.received:
                time.sleep(0.05)
            assert adapter.received, "connector never delivered a reply"
            print(f"\n[result] adapter received: {adapter.received[0]['text']!r}")

            # 2) A second message to prove the pool serves requests serially
            #    while remaining responsive.
            connector.ingest(
                InboundMessage(
                    text="thanks", sender="bob", platform="console",
                    channel_id="general",
                )
            )
            deadline = time.time() + 5.0
            while time.time() < deadline and len(adapter.received) < 2:
                time.sleep(0.05)
            print(f"[result] total replies delivered: {len(adapter.received)}")
        finally:
            stop.set()
            connector.stop()
            print("[demo] connector stopped.")


if __name__ == "__main__":
    main()
