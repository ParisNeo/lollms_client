
"""
BotInstance — a lightweight handle onto one live bot in the registry.

This is the object user code (and, in later phases, the mesh LCP tools and
the supervisor CLI) manipulates. It never owns a thread or process itself;
``.start()``/``.stop()``/``.kill()`` dispatch to the configured supervisor
implementation (phase 3's ``MeshSupervisor``). In phase 1 they raise a clear
``NotImplementedError`` so early adopters get a coherent message rather than
silent no-op.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.types import BotSpec, BotStatus


class BotInstance:
    """
    A handle that binds a ``BotSpec`` to its live registry entry.

    Construct one of:
      * ``BotInstance.create(registry, spec)`` — register (as pending) AND
        return the handle. The spec is persisted.
      * ``BotInstance.attach(registry, bot_id)`` — read an existing spec from
        the registry and return the handle.
    """

    def __init__(self, registry: MeshRegistry, spec: BotSpec):
        self.registry = registry
        self.spec = spec

    # ------------------------------------------------------------ factories
    @classmethod
    def create(cls, registry: MeshRegistry, spec: BotSpec) -> "BotInstance":
        """Registers the bot as PENDING (endpoint/pid not assigned yet)."""
        registry.register(spec, status=BotStatus.PENDING)
        return cls(registry, spec)

    @classmethod
    def attach(cls, registry: MeshRegistry, bot_id: str) -> "BotInstance":
        spec = registry.get_bot_spec(bot_id)
        if spec is None:
            raise KeyError(f"Bot {bot_id!r} not present in mesh registry.")
        return cls(registry, spec)

    # ------------------------------------------------------------- accessors
    @property
    def bot_id(self) -> str:
        return self.spec.bot_id

    def entry(self) -> Optional[Dict[str, Any]]:
        return self.registry.get_bot(self.spec.bot_id)

    def status(self) -> str:
        entry = self.entry()
        return (entry or {}).get("status", BotStatus.PENDING)

    def is_running(self) -> bool:
        return self.status() == BotStatus.RUNNING

    # ---------------------------------------------------------------- actions
    def heartbeat(self) -> None:
        """Marks the bot as alive right now (last_seen = now)."""
        self.registry.update_status(self.spec.bot_id, heartbeat=True)

    def set_status(self, status: str) -> None:
        self.registry.update_status(self.spec.bot_id, status=status)

    def start(self) -> None:
        """
        Launches this bot as a supervised process. Wired in phase 3
        (MeshSupervisor); a coherent error is raised today so callers aren't
        surprised by a silent no-op.
        """
        raise NotImplementedError(
            "BotInstance.start() requires the MeshSupervisor (phase 3). "
            "For now, launch the bot manually via `lollms-bot --config-path ...`."
        )

    def stop(self) -> None:
        """Requests graceful shutdown. Wired in phase 3."""
        raise NotImplementedError(
            "BotInstance.stop() requires the MeshSupervisor (phase 3)."
        )

    def kill(self) -> None:
        """Force-terminates the bot. Wired in phase 3."""
        raise NotImplementedError(
            "BotInstance.kill() requires the MeshSupervisor (phase 3)."
        )

    def remove(self) -> None:
        """
        Removes the bot from the registry entirely. Safe to call on a
        stopped/failed bot; the supervisor is responsible for stopping a
        running one first (mesh design decision: `remove` never kills).
        """
        if not BotStatus.is_terminal(self.status()):
            raise RuntimeError(
                f"Refusing to remove bot {self.bot_id!r} in status "
                f"{self.status()!r}; stop or kill it first."
            )
        self.registry.unregister(self.bot_id)

    def __repr__(self) -> str:
        return f"BotInstance(bot_id={self.bot_id!r}, status={self.status()!r})"
