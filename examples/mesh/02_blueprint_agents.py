
"""
Example 2 — Starter blueprints: instant pre-configured agents.

The blueprint catalog ships a curated set of starter agents (Email Triager,
Code Reviewer, Swarm Orchestrator, ...). This example shows:

  * listing the catalog (pure data, no I/O),
  * materializing a blueprint's starter handbag on disk,
  * creating an agent definition from that handbag,
  * registering it in the mesh.

Nothing is spawned here — that would require a live LLM. This is the exact
sequence the hub UI runs when the operator clicks an "Example Agents" card.

Run:
    python examples/mesh/02_blueprint_agents.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from lollms_client.apps.lollms_bot.mesh import (
    AgentCatalog,
    AgentDefinition,
    MeshLayout,
    MeshRegistry,
    ensure_blueprint_handbag,
    list_blueprints,
)


def main() -> None:
    print("[demo] starter blueprint catalog:")
    for bp in list_blueprints():
        print(f"  {bp.emoji} {bp.blueprint_id:22} — {bp.description}")

    with tempfile.TemporaryDirectory() as tmpdir:
        layout = MeshLayout.under(Path(tmpdir), "demo_mesh")
        layout.materialize()

        # ── Materialize two blueprints as real handbags on disk ─────────
        hb_reviewer = ensure_blueprint_handbag(layout, "code-reviewer")
        hb_orchestr = ensure_blueprint_handbag(layout, "swarm-orchestrator")
        print(f"\n[demo] synthesized handbags:\n  {hb_reviewer}\n  {hb_orchestr}")

        # Idempotent — calling again re-uses the same folder (no churn).
        again = ensure_blueprint_handbag(layout, "code-reviewer")
        assert again == hb_reviewer, "idempotency broken"
        print("[demo] re-materialization is idempotent ✓")

        # ── Register two agents from those blueprints ───────────────────
        catalog = AgentCatalog(layout)
        reviewer = catalog.create_agent(
            AgentDefinition(
                agent_id="reviewer",
                handbag_path=str(hb_reviewer),
                mission_topics=["code.review.request"],
                tags=["code", "quality"],
            )
        )
        orchestrator = catalog.create_agent(
            AgentDefinition(
                agent_id="orchestrator",
                handbag_path=str(hb_orchestr),
                mission_topics=["swarm.task"],
                tags=["orchestration"],
            )
        )
        print(f"\n[demo] created agents:")
        print(f"  reviewer     status={reviewer['status']!r}")
        print(f"  orchestrator status={orchestrator['status']!r}")

        # Both are registered in the SAME registry, so the supervisor can
        # spawn either/both and they will coordinate over the mesh bus.
        registry = MeshRegistry(layout)
        registered = sorted(registry.list_bots().keys())
        print(f"\n[demo] registry: {registered}")
        assert "reviewer" in registered and "orchestrator" in registered


if __name__ == "__main__":
    main()
