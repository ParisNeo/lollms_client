
# Mesh Examples

Three runnable examples showing how to use the lollms-bot mesh, from the
lowest-level primitive to the highest-level composition.

| # | File | What it shows | Dependencies |
|---|------|---------------|--------------|
| 1 | `01_minimal_mesh.py` | Register a bot + publish/receive on the `MessageBus`. | none (stdlib only) |
| 2 | `02_blueprint_agents.py` | List blueprints, materialize handbags, register agents. | none |
| 3 | `03_swarm_connector.py` | Bridge a fake platform onto the mesh via `SwarmConnector`. | none |
| 4 | `04_swarm_with_real_binding.py` | Wire a *live connection binding* (Discord-shaped) into the swarm connector. | none |

None of the examples below spawn a real subprocess or call an LLM. They run
in **under a second** and can be run as smoke tests in CI:

```bash
python examples/mesh/01_minimal_mesh.py
python examples/mesh/02_blueprint_agents.py
python examples/mesh/03_swarm_connector.py
python examples/mesh/04_swarm_with_real_binding.py
```

Example 4 specifically shows the exact path the hub uses when the operator
picks a real connection binding (Discord, Slack, lollms.com) in the swarm
config dialog: a `ConnectionBindingAdapter` pulls inbound messages from the
binding and calls `SwarmConnector.ingest`; replies land back in the
originating channel via the binding's own `send_message`.

## Running a real mesh

To run bot subprocesses that actually answer LLM-backed questions, use the
hub or the CLI:

```bash
# Materialize a starter blueprint, register an agent, and spawn it
python -m lollms_client.apps.lollms_bot.cli hub launch --token secret

# Or headless, on a machine meant to serve as a mesh node:
python -m lollms_client.apps.lollms_bot.cli hub serve --token secret --port 8123
```

Then open `http://localhost:8123/` for Mission Control.

## Concepts at a glance

* **Handbag** — an agent's identity bundle: `SOUL.md`, `tools/`, `skills/`,
  `memory/`, `rag/`. Lives on disk; portable.
* **Blueprint** — a starter template that synthesizes a valid handbag. Lives
  under `<mesh_root>/blueprints/<id>/`.
* **Agent** — a handbag bound to a model profile + mission topics. Every
  mesh bot IS an agent.
* **Bus** — a durable SQLite-WAL topic bus. Supports pub/sub, competing
  consumers (queue), and request/reply.
* **Supervisor** — spawns / watches / restarts agent subprocesses.
* **Swarm connector** — bridges external platforms (Discord, Slack,
  lollms.com) onto the ingress topic, so a human can talk to the mesh.
