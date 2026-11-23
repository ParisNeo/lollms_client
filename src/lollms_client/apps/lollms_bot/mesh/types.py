
"""
Declarative types for the lollms-bot mesh.

Everything here is a pure data structure: no I/O, no threads, no side
effects beyond creating directories on disk when a ``MeshLayout`` is
materialized. These objects are the wire format for the registry, the
workflow designer UI, and the mesh LCP tools.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# A mesh_id is used as a directory name, so it must be a safe slug.
_MESH_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
_BOT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")


def validate_mesh_id(mesh_id: str) -> str:
    """Validates a mesh identifier for use as a filesystem-safe directory name."""
    if not isinstance(mesh_id, str) or not _MESH_ID_RE.match(mesh_id):
        raise ValueError(
            f"Invalid mesh_id {mesh_id!r}: must match [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
        )
    return mesh_id


def validate_bot_id(bot_id: str) -> str:
    """Validates a bot identifier for use as a filesystem-safe directory name."""
    if not isinstance(bot_id, str) or not _BOT_ID_RE.match(bot_id):
        raise ValueError(
            f"Invalid bot_id {bot_id!r}: must match [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
        )
    return bot_id


class BotStatus:
    """Lifecycle states of a bot in a mesh. Values are the on-disk strings."""

    PENDING = "pending"        # registered but not yet spawned
    STARTING = "starting"      # supervisor launched it; not ready yet
    RUNNING = "running"        # ready to receive messages
    STOPPING = "stopping"      # graceful shutdown in progress
    STOPPED = "stopped"        # exited cleanly
    FAILED = "failed"          # crashed or launch failure

    _VALID = {PENDING, STARTING, RUNNING, STOPPING, STOPPED, FAILED}

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        return status in (cls.STOPPED, cls.FAILED)


@dataclass
class BotSpec:
    """
    Declarative description of one bot in a mesh.

    This is what a workflow designer edits and what the supervisor consumes.
    It is deliberately model/handbag-agnostic: a bot is defined by WHAT it is
    given (handbag + LLM profile), not by any particular binding.

    Fields
    ------
    bot_id
        Unique, filesystem-safe identifier within the mesh.
    handbag_path
        Path to the bot's Handbag. Relative paths resolve against the mesh
        root; absolute paths are used verbatim. If None, ``LollmsPersonality``
        default is used.
    llm_profile
        Name of an LollmsClient LLM profile to launch the bot with. When
        None, the bot inherits the global default profile. When the string
        ``"default"`` it explicitly requests the default profile.
    connection_profile
        Name of an LollmsClient CONNECTION profile (telegram, discord, ...).
        Multiple connection profiles are joined by reaching the same bus —
        for phase 1 this is informational metadata.
    capabilities
        Overrides for ``CapabilityFlags`` (dict). Applied on top of defaults.
    tags
        Free-form labels used for discovery (e.g. ``["intake", "validator"]``).
    inbound_routes
        Declared subscription topics (for the phase-4 workflow router). Each
        entry is a topic string or ``"*"``. Informational in phase 1.
    metadata
        Arbitrary JSON-serializable data the designer attaches (position in
        the graph, colors, etc.).
    """

    bot_id: str
    handbag_path: Optional[str] = None
    llm_profile: Optional[str] = None
    connection_profile: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    inbound_routes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_bot_id(self.bot_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MeshConfig:
    """
    Mesh-wide configuration, persisted once at ``<mesh_root>/mesh.yaml``.

    ``defaults`` holds fallback values applied to every bot that doesn't
    override them (e.g. ``temperature``, ``max_tokens_per_turn``). The
    registry stores it only for reference; the supervisor (phase 3) is the
    authoritative reader.
    """

    mesh_id: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    defaults: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_mesh_id(self.mesh_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class MeshLayout:
    """
    Canonical on-disk layout of one mesh. Materialization is idempotent and
    safe to call from any process.

    Directory tree:
        <root>/
            mesh.yaml            # MeshConfig
            registry.json        # live presence (see MeshRegistry)
            bus.db               # message bus (phase 2, SQLite WAL)
            workflow.yaml        # workflow graph (phase 4)
            bots/<bot_id>/       # per-bot data root
                handbag/         # default handbag location, if the spec omits it
                workspace/       # per-bot writable workspace
                logs/            # stdout/stderr, run traces
    """

    root: Path

    # --- derived, deterministic paths -------------------------------------
    @property
    def mesh_yaml(self) -> Path:
        return self.root / "mesh.yaml"

    @property
    def registry_json(self) -> Path:
        return self.root / "registry.json"

    @property
    def bus_db(self) -> Path:
        return self.root / "bus.db"

    @property
    def workflow_yaml(self) -> Path:
        return self.root / "workflow.yaml"

    @property
    def bots_dir(self) -> Path:
        return self.root / "bots"

    def bot_dir(self, bot_id: str) -> Path:
        validate_bot_id(bot_id)
        return self.bots_dir / bot_id

    def bot_handbag_dir(self, bot_id: str) -> Path:
        return self.bot_dir(bot_id) / "handbag"

    def bot_workspace_dir(self, bot_id: str) -> Path:
        return self.bot_dir(bot_id) / "workspace"

    def bot_logs_dir(self, bot_id: str) -> Path:
        return self.bot_dir(bot_id) / "logs"

    def materialize(self) -> "MeshLayout":
        """Creates root + subdirectories with mkdir(exist_ok=True)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.bots_dir.mkdir(exist_ok=True)
        return self

    def ensure_bot_dirs(self, bot_id: str) -> None:
        """Ensures all directories needed by one bot exist."""
        for d in (
            self.bot_dir(bot_id),
            self.bot_handbag_dir(bot_id),
            self.bot_workspace_dir(bot_id),
            self.bot_logs_dir(bot_id),
        ):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def under(cls, parent: Path, mesh_id: str) -> "MeshLayout":
        validate_mesh_id(mesh_id)
        return cls(root=Path(parent) / mesh_id)
