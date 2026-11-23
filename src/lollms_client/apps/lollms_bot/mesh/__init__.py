
"""
lollms-bot mesh: decentralized multi-instance bot runtime.

Public surface
--------------
Two layers are exposed from this package:

**Foundation primitives** (phases 1-3) — everything needed to run a mesh of
bot processes that coordinate over a durable bus:

  * :class:`MeshLayout`      — on-disk layout for one mesh.
  * :class:`BotSpec`         — declarative definition of a bot.
  * :class:`BotStatus`       — lifecycle states (PENDING→RUNNING→...).
  * :class:`MeshRegistry`    — cross-process registry (registry.json).
  * :class:`BotInstance`     — a registered bot + its registry operations.
  * :class:`MessageBus`      — SQLite-WAL topic bus (pub/sub + queue + RPC).
  * :class:`MeshSupervisor`  — process spawn/stop/kill + bounded restart.
  * :class:`SessionRecorder` — records mesh activity for replay.
  * :class:`Replayer`        — replays a recorded session.

**High-level composition** — the operator-facing layer that turns primitives
into a usable Mission Control:

  * :class:`AgentCatalog`      — validate/create agents (handbag + model).
  * :class:`AgentDefinition`   — high-level agent inputs (UI form shape).
  * :class:`Blueprint`         — a curated starter-agent template.
  * :func:`list_blueprints`    — the full starter catalog.
  * :class:`SwarmConnector`    — bridge external platforms onto the bus.
  * :class:`SwarmPlatformAdapter` — the adapter a platform bridge implements.
  * :class:`InboundMessage`    — normalized inbound message from a platform.

The composition submodules (``agents``, ``blueprints``, ``connector``) each
import from the foundation submodules but never from each other. This keeps
the import graph acyclic and allows importing only what you need:

    from lollms_client.apps.lollms_bot.mesh.agents import AgentCatalog
    from lollms_client.apps.lollms_bot.mesh.blueprints import list_blueprints

The re-exports below are provided for ergonomics — a caller who wants the
common pieces (``from lollms_client.apps.lollms_bot.mesh import BotSpec,
MessageBus, SwarmConnector``) does not have to know which submodule owns what.

Heavy / optional submodules are NOT re-exported at package level on purpose:
  * ``gui``       — requires FastAPI/uvicorn; imported explicitly.
  * ``remote``    — requires urllib + ssl (stdlib, but kept explicit).
  * ``handbag_io``/``recorder`` — optional for pure-runtime users.
Importing them here would force every ``import lollms_bot.mesh`` to pull in
their transitive dependencies. Use the submodule path to access them.
"""

from __future__ import annotations

# ── Foundation: types / registry / instance ──────────────────────────────
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshConfig,
    MeshLayout,
    validate_bot_id,
)
from lollms_client.apps.lollms_bot.mesh.registry import (
    MeshRegistry,
    RegistryError,
)
from lollms_client.apps.lollms_bot.mesh.instance import BotInstance

# ── Foundation: transport / supervision ──────────────────────────────────
from lollms_client.apps.lollms_bot.mesh.bus import (
    BusError,
    BusMessage,
    MessageBus,
    Subscription,
)
from lollms_client.apps.lollms_bot.mesh.supervisor import (
    BotProcess,
    MeshSupervisor,
    SupervisorError,
)

# ── Composition: agents / blueprints ─────────────────────────────────────
from lollms_client.apps.lollms_bot.mesh.agents import (
    AgentCatalog,
    AgentDefinition,
    AgentError,
    discover_handbags,
    discover_model_profiles,
)
from lollms_client.apps.lollms_bot.mesh.blueprints import (
    Blueprint,
    BlueprintError,
    ensure_blueprint_handbag,
    get_blueprint,
    list_blueprints,
)

# ── Composition: swarm connector ─────────────────────────────────────────
from lollms_client.apps.lollms_bot.mesh.connector import (
    InboundMessage,
    SwarmConnector,
    SwarmConnectorError,
    SwarmPlatformAdapter,
    load_connector_config,
    save_connector_config,
)
from lollms_client.apps.lollms_bot.mesh.connection_adapter import (
    ConnectionBindingAdapter,
    build_adapters_from_client,
)

__all__ = [
    # types / layout / registry
    "BotSpec",
    "BotStatus",
    "MeshConfig",
    "MeshLayout",
    "validate_bot_id",
    "MeshRegistry",
    "RegistryError",
    "BotInstance",
    # transport
    "BusError",
    "BusMessage",
    "MessageBus",
    "Subscription",
    # supervision
    "BotProcess",
    "MeshSupervisor",
    "SupervisorError",
    # agents
    "AgentCatalog",
    "AgentDefinition",
    "AgentError",
    "discover_handbags",
    "discover_model_profiles",
    # blueprints
    "Blueprint",
    "BlueprintError",
    "ensure_blueprint_handbag",
    "get_blueprint",
    "list_blueprints",
    # swarm connector
    "InboundMessage",
    "SwarmConnector",
    "SwarmConnectorError",
    "SwarmPlatformAdapter",
    "load_connector_config",
    "save_connector_config",
    # swarm connector — connection-binding bridge
    "ConnectionBindingAdapter",
    "build_adapters_from_client",
]
