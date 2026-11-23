
"""
Example 1 — Minimal mesh: two bots talking over the bus.

This is the smallest end-to-end mesh demo. It shows:

  * creating a mesh layout on disk,
  * registering one consumer bot (listens on "greet") and one producer
    (publishes on "greet"),
  * publishing a message and seeing the consumer publish its reply,
  * clean shutdown of the bus.

It does NOT spawn real bot processes — spawning requires an LLM client and
a real handbag (see ``03_agent_catalog.py``). This example deliberately
exercises only the ``MessageBus`` primitive so it runs in milliseconds with
no external dependencies.

Run:
    python examples/mesh/01_minimal_mesh.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from lollms_client.apps.lollms_bot.mesh import (
    BotInstance,
    BotSpec,
    BotStatus,
    MessageBus,
    MeshLayout,
    MeshRegistry,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        layout = MeshLayout.under(tmp, "demo_mesh")
        layout.materialize()
        print(f"[demo] mesh root: {layout.root}")

        # ── Register a bot (declarative) ────────────────────────────────
        registry = MeshRegistry(layout)
        spec = BotSpec(
            bot_id="greeter",
            handbag_path=str(layout.bot_handbag_dir("greeter")),
            inbound_routes=["greet"],
            tags=["demo", "consumer"],
        )
        BotInstance.create(registry, spec)
        print(f"[demo] registered: {registry.list_bots().keys()}")

        # ── Publish + receive over the bus ──────────────────────────────
        with MessageBus(layout) as bus:
            bus.subscribe("greeter", "greet", from_now=False)
            bus.send_message(
                "greet",
                {"text": "Hello from the producer!"},
                sender="producer",
            )
            msgs = bus.wait_for("greeter", "greet", timeout=2.0)
            assert len(msgs) == 1, f"expected 1 message, got {len(msgs)}"
            print(f"[demo] greeter received: {msgs[0].payload['text']}")

            # The consumer emulates replying on the standard reply topic.
            reply_topic = "greet.reply"
            bus.subscribe("producer", reply_topic, from_now=False)
            bus.send_message(
                reply_topic,
                {"text": f"Hi producer — greeter here, msg#{msgs[0].id} received."},
                sender="greeter",
                correlation_id=msgs[0].correlation_id,
            )
            replies = bus.wait_for("producer", reply_topic, timeout=2.0)
            assert len(replies) == 1
            print(f"[demo] producer received: {replies[0].payload['text']}")

            # Mark the bot running so a supervisor would not re-spawn it.
            registry.update_status("greeter", status=BotStatus.RUNNING)

        print("[demo] mesh bus closed cleanly.")


if __name__ == "__main__":
    main()
