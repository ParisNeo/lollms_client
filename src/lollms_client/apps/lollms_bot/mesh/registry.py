
"""
Persistent mesh registry: presence + discovery for the bots in one mesh.

Design constraints
------------------
* **Multi-process safe.** Multiple supervisor processes (or manual CLI
  invocations) may read/write the registry concurrently. Every mutation is
  done under an exclusive file lock (``.<mesh>/registry.json.lock``) and
  writes are atomic (write-temp + ``os.replace``), so a reader never sees
  half a file and a writer never corrupts another writer's entry.

* **No central server.** The registry is just a JSON file per mesh. Any
  participant can compute the path from ``mesh_id`` and read/write it.

* **Fail-open reads, fail-closed writes.** A missing/corrupt registry reads
  as empty; a failing write raises ``RegistryError`` so a caller never
  silently loses a presence update.

File format (registry.json):
    {
      "mesh_id": "...",
      "generation": 42,
      "bots": {
        "<bot_id>": {
          "spec": { ... },          # BotSpec.to_dict()
          "status": "running",      # BotStatus
          "endpoint": null,         # reserved for phase 3 (pid, unix socket)
          "pid": 12345,             # informational
          "last_seen": 1732000000,  # epoch seconds
          "error": null             # last fatal error string, if any
        }
      }
    }

Note: this module deliberately has ZERO knowledge of the bus or the LLM.
It is the presence directory — the bus (phase 2) delivers the messages.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
    validate_bot_id,
)


class RegistryError(RuntimeError):
    """Raised when a registry mutation cannot be completed safely."""


def _now() -> float:
    return time.time()


def _current_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


class MeshRegistry:
    """
    File-backed presence registry for one mesh.

    ``MeshRegistry(layout)`` accepts a pre-built layout (see ``MeshLayout``).

    All public methods are thread-safe within a process (an in-process
    ``RLock`` protects the file lock acquisition/release cycle) AND
    cross-process safe (the OS-level lock is the real serialization point).
    """

    _INPROC_LOCK = threading.RLock()
    _LOCK_SUFFIX = ".registry.lock"

    def __init__(self, layout: MeshLayout):
        self.layout = layout
        self.layout.materialize()

    # ------------------------------------------------------------------ io
    @contextmanager
    def _locked(self) -> Iterator[None]:
        """
        Acquires the cross-process file lock for the registry. Lock lifetime
        covers an entire read-modify-write cycle so increments (generation)
        cannot race. Uses ``fcntl`` on POSIX, ``msvcrt`` on Windows.
        """
        lock_path = self.layout.root / f".registry{self._LOCK_SUFFIX}"
        with self._INPROC_LOCK:
            fh = open(lock_path, "a+b")
            try:
                self._os_lock(fh)
                yield
            finally:
                try:
                    self._os_unlock(fh)
                finally:
                    fh.close()

    @staticmethod
    def _os_lock(fh) -> None:
        if os.name == "nt":  # Windows
            import msvcrt

            # msvcrt.locking() requires 1 byte in the region to lock.
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _os_unlock(fh) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

    def _read_raw(self) -> Dict[str, Any]:
        path = self.layout.registry_json
        if not path.exists():
            return self._empty()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Fail-open read: a corrupt registry is treated as empty so a
            # supervisor can rebuild presence from scratch rather than crash.
            return self._empty()
        if not isinstance(data, dict) or "bots" not in data:
            return self._empty()
        return data

    def _write_raw(self, data: Dict[str, Any]) -> None:
        """
        Atomic write via temp+replace. The temp file lives in the same
        directory so ``os.replace`` is atomic on every supported FS.

        Windows caveat: ``os.replace`` raises ``WinError 5 (Access Denied)``
        when the destination is momentarily open by another process’s reader
        handle. On a busy mesh (multiple bots + the supervisor + the hub all
        writing status/heartbeats) this collision is *routine*, not a bug —
        so we retry the replace with short exponential backoff instead of
        surfacing a spurious failure that would stall a running bot’s
        heartbeat thread. POSIX ``os.replace`` never needs the retry, but the
        loop is harmless there (it succeeds on the first attempt).
        """
        path = self.layout.registry_json
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".registry.", suffix=".tmp", dir=str(path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                self._replace_with_retry(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            raise RegistryError(f"Failed to write registry {path}: {e}") from e

    #: Windows error codes that mean "transient — retry". ``5`` is
    #: ERROR_ACCESS_DENIED and is what ``os.replace`` commonly raises when the
    #: destination is momentarily open by a concurrent reader; ``32`` is
    #: ERROR_SHARING_VIOLATION; ``33`` is ERROR_LOCK_VIOLATION. Anything else
    #: is a real failure and must propagate immediately (retrying it would
    #: just delay a guaranteed error).
    _WINDOWS_RETRYABLE_WINERRORS = frozenset({5, 32, 33})

    @classmethod
    def _is_transient_replace_error(cls, exc: BaseException) -> bool:
        """
        True for the transient "destination is momentarily in use" condition
        that is ROUTINE on a busy mesh (multiple bots + supervisor + hub all
        heartbeating into the SAME registry.json).

        Why ``hasattr(exc, 'winerror')`` rather than ``except PermissionError``:
        on CPython for Windows, ``os.replace`` surfaces Error 5 as a plain
        ``OSError`` in some builds and as a ``PermissionError`` subclass in
        others. Catching only ``PermissionError`` lets the plain-``OSError``
        form escape the retry loop entirely — which is precisely the bug that
        made a bot's heartbeat thread die with ``[WinError 5] Accès refusé``
        while another process held the file for a few milliseconds.

        On POSIX, ``os.replace`` is atomic and effectively never transiently
        fails; we return False for any non-Windows error so real POSIX
        failures surface immediately (never silently swallowed by a retry).
        """
        if os.name != "nt":
            return False
        return getattr(exc, "winerror", None) in cls._WINDOWS_RETRYABLE_WINERRORS

    @classmethod
    def _replace_with_retry(cls, src: str, dst: Path, *, attempts: int = 40) -> None:
        """
        ``os.replace(src, dst)`` with bounded exponential backoff on transient
        Windows share/lock violations (WinError 5/32/33).

        ``attempts`` is deliberately generous (40) with a 20ms base: the
        total worst-case wait is ~1.4s, comfortably under the supervisor's
        reconcile interval, so a burst of concurrent heartbeats serializes
        through the retry loop instead of crashing any one bot's heartbeat
        thread. Genuinely-unretryable errors re-raise on the first attempt.
        """
        last_error: Optional[OSError] = None
        for i in range(attempts):
            try:
                os.replace(src, dst)
                return
            except OSError as e:  # includes PermissionError on all platforms
                if not cls._is_transient_replace_error(e):
                    # Not a transient Windows sharing violation — a real
                    # failure. Never mask it with a retry loop.
                    raise
                last_error = e
                time.sleep(min(0.02 * (2 ** i), 0.5))
        if last_error is not None:
            raise last_error

    def _empty(self) -> Dict[str, Any]:
        return {"mesh_id": self.layout.root.name, "generation": 0, "bots": {}}

    # --------------------------------------------------------------- public
    def list_bots(self) -> Dict[str, Dict[str, Any]]:
        """Returns a snapshot of the current bot entries (read-only)."""
        return dict(self._read_raw().get("bots", {}))

    def get_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        validate_bot_id(bot_id)
        return self._read_raw().get("bots", {}).get(bot_id)

    def get_bot_spec(self, bot_id: str) -> Optional[BotSpec]:
        entry = self.get_bot(bot_id)
        if entry is None:
            return None
        spec = entry.get("spec")
        if not isinstance(spec, dict):
            return None
        try:
            return BotSpec.from_dict(spec)
        except Exception:
            return None

    def register(self, spec: BotSpec, *, status: str = BotStatus.PENDING) -> None:
        """
        Registers or replaces a bot's presence record. Registers as ``pending``
        by default — the supervisor flips it to ``running`` after a successful
        launch. Replacing an existing bot is allowed (designer edits).
        """
        if status not in BotStatus._VALID:
            raise RegistryError(f"Invalid status {status!r}")
        with self._locked():
            data = self._read_raw()
            data["bots"][spec.bot_id] = {
                "spec": spec.to_dict(),
                "status": status,
                "endpoint": None,
                "pid": None,
                "host": _current_host(),
                "updated_at": _now(),
                "last_seen": _now(),
                "error": None,
            }
            data["generation"] = int(data.get("generation", 0)) + 1
            self._write_raw(data)

    def update_status(
        self,
        bot_id: str,
        *,
        status: Optional[str] = None,
        pid: Optional[int] = None,
        endpoint: Optional[str] = None,
        error: Optional[str] = None,
        heartbeat: bool = False,
    ) -> None:
        """
        Updates fields on an existing bot. Silently no-ops if the bot has
        been concurrently deleted (register() wins in that case).
        """
        validate_bot_id(bot_id)
        if status is not None and status not in BotStatus._VALID:
            raise RegistryError(f"Invalid status {status!r}")
        with self._locked():
            data = self._read_raw()
            entry = data.get("bots", {}).get(bot_id)
            if entry is None:
                return
            if status is not None:
                entry["status"] = status
            if pid is not None:
                entry["pid"] = int(pid)
            if endpoint is not None:
                entry["endpoint"] = endpoint
            if error is not None:
                entry["error"] = error
            if heartbeat:
                entry["last_seen"] = _now()
            entry["updated_at"] = _now()
            data["generation"] = int(data.get("generation", 0)) + 1
            self._write_raw(data)

    def unregister(self, bot_id: str) -> None:
        """Removes a bot from the registry (idempotent)."""
        validate_bot_id(bot_id)
        with self._locked():
            data = self._read_raw()
            if data.get("bots", {}).pop(bot_id, None) is not None:
                data["generation"] = int(data.get("generation", 0)) + 1
                self._write_raw(data)

    def find_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Returns all bots on the mesh tagged with ``tag`` (exact match,
        case-sensitive). Used by the mesh discovery tools (phase 5).
        """
        out: List[Dict[str, Any]] = []
        for entry in self.list_bots().values():
            spec = entry.get("spec") or {}
            tags = spec.get("tags") or []
            if tag in tags:
                out.append(entry)
        return out

    def find_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Returns all bots whose spec.capabilities[capability] is truthy.
        Enables routing like "send this to the first bot with
        enable_computer_use=True".
        """
        out: List[Dict[str, Any]] = []
        for entry in self.list_bots().values():
            caps = (entry.get("spec") or {}).get("capabilities") or {}
            if bool(caps.get(capability, False)):
                out.append(entry)
        return out

    def running_bots(self) -> List[str]:
        """Convenience: ids of bots whose status is RUNNING."""
        return [
            bid
            for bid, entry in self.list_bots().items()
            if entry.get("status") == BotStatus.RUNNING
        ]
