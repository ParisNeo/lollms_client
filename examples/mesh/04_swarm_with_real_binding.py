
"""
Example 4 — Swarm connector driven by a REAL connection binding contract.

Building on example 3, this example wires a *live connection binding* into the
swarm connector — exactly the way the hub's `/api/swarm/config` endpoint does
in production. It uses a minimal in-process fake binding that mimics the
``LollmsConnectionBinding`` base contract (``send_message`` /
``receive_message`` / ``is_connected``), so the example is self-contained
and never touches a real Discord/Slack/lollms.com API.

What it proves (and why it matters):
  * the connector correctly pulls from a binding's ``receive_message`` loop;
  * mesh replies round-trip back to the binding's ``send_message``;
  * the whole path — human → binding → connector → orchestrator bot → reply
    → connector → binding → human — works end to end.

This is the "one endpoint the operator enables, and real users can talk to
the mesh" path.

Run:
    python examples/mesh/04_swarm_with_real_binding.py
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from lollms_client.apps.lollms_bot.mesh import (
    ConnectionBindingAdapter,
    InboundMessage,
    MeshLayout,
    MessageBus,
    SwarmConnector,
)


# ---------------------------------------------------------------------------
# A fake connection binding matching the lollms-client contract.
# ---------------------------------------------------------------------------

@dataclass
class _FakeInboundMessage:
    """Mimics ``ConnectionReceiveResult``-ish shape a binding returns."""
    content: str
    channel_id: str
    author: str = "user"


class FakeConnectionBinding:
    """
    Minimal LollmsConnectionBinding-compatible stand-in.

    The real bindings (Discord, Slack, lollms.com) provide one inbound queue
    and one outbound path. This fake does the same, in memory, so tests
    and examples stay hermetic:

      * ``push_inbound``     — simulate a user sending a message;
      * ``receive_message``  — pull the next one (blocking with timeout);
      * ``send_message``     — record a reply (or the test prints it);
      * ``is_connected``     — always True here.
    """

    binding_name = "fake-discord"

    def __init__(self) -> None:
        self._inbound: "queue.Queue[_FakeInboundMessage]" = queue.Queue()
        self._sent: list[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # ── observer hooks (binding contract) ────────────────────────────────
    def is_connected(self) -> bool:
        return True

    def receive_message(self, timeout: float = 0.0) -> Optional[_FakeInboundMessage]:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_message(self, channel_id: str, text: str) -> None:
        with self._lock:
            self._sent.append({"channel_id": channel_id, "text": text})
        print(f"[{self.binding_name}] ⇠ #{channel_id}: {text}")

    # ── test hooks ───────────────────────────────────────────────────────
    def push_inbound(self, text: str, channel_id: str, author: str = "user") -> None:
        self._inbound.put(
            _FakeInboundMessage(content=text, channel_id=channel_id, author=author)
        )

    def sent_messages(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._sent)


# ---------------------------------------------------------------------------
# Fake orchestrator bot: consumes swarm.inbound, replies on .reply
# ---------------------------------------------------------------------------

def fake_orchestrator(layout: MeshLayout, stop: threading.Event) -> None:
    with MessageBus(layout) as bus:
        bus.subscribe("orch", "swarm.inbound", from_now=False)
        while not stop.is_set():
            msgs = bus.wait_for("orch", "swarm.inbound", timeout=0.25)
            for m in msgs:
                payload = m.payload or {}
                text = payload.get("text", "")
                platform = payload.get("platform", "?")
                token = payload.get("reply_token")
                print(f"[orchestrator] ⇢ from {platform}: {text!r}")
                bus.send_message(
                    "swarm.inbound.reply",
                    {
                        "text": (
                            f"[mesh] Hi {payload.get('sender')} — "
                            f"acknowledged: {text.upper()}"
                        )
                    },
                    sender="orchestrator",
                    correlation_id=token,
                )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        layout = MeshLayout.under(Path(tmpdir), "swarm_binding_mesh")
        layout.materialize()

        # 1) Build a real-ish connection binding + wire it to the connector.
        binding = FakeConnectionBinding()
        adapter = ConnectionBindingAdapter(binding)
        connector = SwarmConnector(
            layout,
            adapter=adapter,
            config={
                "enabled": True,
                "ingress_topic": "swarm.inbound",
                "reply_timeout_seconds": 5,
                "max_concurrent": 4,
            },
        )
        adapter.attach(connector)
        connector.start()
        adapter.start()

        # 2) Stand up the "orchestrator agent" the mesh would normally run.
        stop = threading.Event()
        orch_thread = threading.Thread(
            target=fake_orchestrator, args=(layout, stop), daemon=True
        )
        orch_thread.start()

        try:
            # 3) Simulate a Discord user asking a question.
            print("[fake-discord] ⇢ alice: what's the status of the build?")
            binding.push_inbound(
                text="what's the status of the build?",
                channel_id="team-eng",
                author="alice",
            )

            # 4) Wait for the round trip (platform → connector → orchestrator
            #    → connector → platform).
            deadline = time.time() + 5.0
            while time.time() < deadline and not binding.sent_messages():
                time.sleep(0.05)

            assert binding.sent_messages(), "no reply reached the binding"
            reply = binding.sent_messages()[0]
            print(f"\n[result] binding received → #{reply['channel_id']}: {reply['text']}")
            assert reply["channel_id"] == "team-eng"
            assert "acknowledged" in reply["text"].lower()

            # 5) A second user from a different channel — proves the connector
            #    routes replies to the correct originating channel.
            print("\n[fake-discord] ⇢ bob: ping from another channel")
            binding.push_inbound(
                text="ping from another channel",
                channel_id="team-ops",
                author="bob",
            )
            deadline = time.time() + 5.0
            while time.time() < deadline and len(binding.sent_messages()) < 2:
                time.sleep(0.05)
            sent = binding.sent_messages()
            assert len(sent) == 2, f"expected 2 replies, got {len(sent)}"
            print(f"[result] channels: {[m['channel_id'] for m in sent]}")
            assert sent[1]["channel_id"] == "team-ops"
        finally:
            stop.set()
            adapter.stop()
            connector.stop()
            print("[demo] connector + adapter stopped cleanly.")


if __name__ == "__main__":
    main()
