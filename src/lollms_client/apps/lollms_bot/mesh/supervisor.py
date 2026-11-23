
"""
MeshSupervisor — process-spawning supervisor for the lollms-bot mesh (phase 3a).

Turns a declarative ``BotSpec`` into a real running ``lollms-bot mesh-bot``
subprocess wired to the shared mesh bus, and provides the operator (via the
hub UI, the CLI, or the mesh tools) with a single, robust control surface:

    spawn(bot_id)     launch a bot subprocess
    stop(bot_id)      graceful shutdown (SIGTERM, then SIGKILL after grace)
    kill(bot_id)      force-terminate process tree
    restart(bot_id)   stop (if running) then spawn
    status(bot_id)    full record: pid, status, started_at, restarts, ...

Design principles
-----------------
* **Supervision is separate from the bot's own logic.** This module only
  knows (1) "spawn a subprocess running a known module with a known config",
  (2) "watch it", (3) "update its registry entry". The bot side runs in
  ``mesh/runtime.py`` and is completely independent.

* **Registry is the source of truth.** Every state change writes to
  ``MeshRegistry`` before mutating internal state, so another process (the
  hub UI, another supervisor, a mesh tool) sees a consistent view. Statuses
  transition strictly PENDING→STARTING→RUNNING→STOPPING→STOPPED|FAILED.

* **Logs go to disk, not to pipes.** Each bot's stdout+stderr redirect to
  ``<bot_dir>/logs/stdout.log``. This means (a) no deadlock risk from
  unconsumed PIPE buffers, (b) the hub can tail logs across processes, and
  (c) a bot that crashes leaves a full trace without needing the supervisor.

* **Restart-on-failure is bounded.** A bot that exits non-zero is
  relaunched up to ``max_restarts`` times with exponential backoff (capped).
  Clean exits (status STOPPING→STOPPED) are NOT restarted. A bot that has
  failed ``max_restarts`` times becomes FAILED and stays there until the
  operator explicitly restarts it. This prevents runaway restart loops.

* **Cross-platform.** Uses ``subprocess.Popen`` + per-OS primitives:
  POSIX uses ``os.killpg``; Windows uses ``taskkill /T /F`` for the tree.

* **One supervisor per process.** Launching a bot twice is refused; the
  caller receives a coherent error. Two supervisors in two terminals is
  supported (they both target the same registry), as long as they don't
  target the same bots — the caller is responsible for this, and the
  registry pid field makes double-launch visible.

Nothing here requires FastAPI. The supervisor is usable from a plain
``python -m lollms_client.apps.lollms_bot.cli supervisor ...`` command or
embedded in the hub.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry, RegistryError
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
    validate_bot_id,
)


class SupervisorError(RuntimeError):
    """Raised when a supervisory operation cannot be completed safely."""


# ---------------------------------------------------------------------------
# Per-bot in-memory record
# ---------------------------------------------------------------------------

@dataclass
class BotProcess:
    """
    In-memory supervisory record for one bot launch.

    Decoupled from the registry on purpose: the registry is the *serialized*
    state (cross-process view); ``BotProcess`` is the *live* handle the
    supervisor itself manipulates (subprocess, log file, restart counter).
    """
    bot_id: str
    process: subprocess.Popen
    log_path: Path
    started_at: float
    restarts: int = 0
    stdout_handle: Any = None    # file object opened for the child
    intentional_stop: bool = False  # True once stop()/kill() called; disables auto-restart
    watch_thread: Optional[threading.Thread] = None

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "pid": self.process.pid,
            "started_at": self.started_at,
            "restarts": self.restarts,
            "alive": self.is_alive(),
            "returncode": self.process.returncode,
            "log_path": str(self.log_path),
            "intentional_stop": self.intentional_stop,
        }


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class MeshSupervisor:
    """
    Launches and supervises bot subprocesses for one mesh.

    We deliberately avoid re-implementing a "daemon" here: the CLI already
    spawns ``lollms-bot mesh-bot <mesh_id> <bot_id>`` and that subprocess is
    responsible for its own loop. The supervisor's job is lifecycle only —
    spawn it, watch it, and re-spawn it if it politely failed.

    Thread safety: all mutating methods take ``self._lock``. ``watch_loop``
    runs in a single background thread on the whole supervisor (not per bot)
    so the cross-bot state transitions stay serialized.
    """

    # Restart policy. These values are deliberately conservative: a mesh is a
    # small federation, and restart storms are worse than a slow recovery.
    _MAX_RESTARTS = 3
    _BACKOFF_BASE_SECONDS = 2.0    # 2s, 4s, 8s
    _BACKOFF_MAX_SECONDS = 30.0
    _STOP_GRACE_SECONDS = 10.0     # SIGTERM → wait → SIGKILL after this
    _WATCH_INTERVAL_SECONDS = 1.0
    # Grace window before a live child is promoted STARTING → RUNNING. The
    # child (runtime.py) enters RUNNING itself after the personality is
    # built; this grace is only a fallback so that a supervisor whose child
    # has no registry-write privileges can still make forward progress.
    _STARTUP_GRACE_SECONDS = 3.0
    # Cap on restarts within a rolling window. Prevents (MAX_RESTARTS/hour ×
    # a bot that crashes in a loop) from saturating a machine.
    _RESTART_WINDOW_SECONDS = 600.0

    def __init__(
        self,
        layout: MeshLayout,
        *,
        python_executable: Optional[str] = None,
        autostart_watchdog: bool = True,
    ):
        self.layout = layout
        self.layout.materialize()
        self.registry = MeshRegistry(layout)
        self._python = python_executable or sys.executable
        self._lock = threading.RLock()
        self._procs: Dict[str, BotProcess] = {}
        # Rolling restart timestamps so a fast crash-loop can't bypass the
        # backoff indefinitely.
        self._restart_history: Dict[str, Deque[float]] = {}
        self._watch_stop = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None
        # The watchdog MUST be running for restart-on-failure and the
        # STARTING→RUNNING promotion to ever happen. Historically the caller
        # had to remember to call start_watchdog(); autostart makes the
        # default safe (tests can pass autostart_watchdog=False).
        if autostart_watchdog:
            self.start_watchdog()

    # --------------------------------------------------------- lifecycle
    def start_watchdog(self) -> None:
        """
        Begins the background watchdog. Idempotent — calling twice does
        nothing. The watchdog reconciles the registry with actual process
        state and applies the restart policy.
        """
        with self._lock:
            if self._watch_thread is not None and self._watch_thread.is_alive():
                return
            self._watch_stop.clear()
            self._watch_thread = threading.Thread(
                target=self._watch_loop, name="mesh-supervisor-watch", daemon=True
            )
            self._watch_thread.start()

    def stop_watchdog(self) -> None:
        """Stops the watchdog. Any still-running bots keep running."""
        with self._lock:
            self._watch_stop.set()
            t = self._watch_thread
            self._watch_thread = None
        if t is not None:
            t.join(timeout=5.0)

    # ----------------------------------------------------------- queries
    def list_running(self) -> List[Dict[str, Any]]:
        """Returns the in-memory record for every tracked (alive or dead) bot."""
        with self._lock:
            return [p.to_dict() for p in self._procs.values()]

    def get(self, bot_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            proc = self._procs.get(bot_id)
            return proc.to_dict() if proc is not None else None

    # --------------------------------------------------------- spawn/stop
    def spawn(self, bot_id: str) -> Dict[str, Any]:
        """
        Launches the bot as a subprocess. Refuses if already tracked and
        alive. The registry entry is transitioned PENDING/STOPPED/FAILED →
        STARTING → RUNNING (RUNNING only after we can confirm the process is
        alive past a brief stabilization delay, giving the supervisor's
        caller a real answer).

        Returns the ``BotProcess.to_dict()`` snapshot.
        """
        validate_bot_id(bot_id)
        with self._lock:
            existing = self._procs.get(bot_id)
            if existing is not None and existing.is_alive():
                raise SupervisorError(
                    f"Bot {bot_id!r} is already running (pid {existing.process.pid})."
                )
            entry = self.registry.get_bot(bot_id)
            if entry is None:
                raise SupervisorError(
                    f"Bot {bot_id!r} is not registered in the mesh. Register it first."
                )
            spec = BotSpec.from_dict(entry["spec"]) if isinstance(entry.get("spec"), dict) else None
            if spec is None:
                raise SupervisorError(f"Bot {bot_id!r} has no valid BotSpec.")

            self.layout.ensure_bot_dirs(bot_id)
            log_path = self.layout.bot_logs_dir(bot_id) / "stdout.log"

            try:
                stdout_handle = open(log_path, "ab", buffering=0)
            except OSError as e:
                raise SupervisorError(f"Cannot open stdout log {log_path}: {e}") from e

            try:
                cmd = self._build_command(bot_id)
                env = self._build_env(bot_id, spec)
                proc = self._launch_process(cmd, env, stdout_handle)
            except Exception as e:
                # Ensure the log handle never leaks on ANY launch failure
                # (cmd construction, env build, Popen).
                try:
                    stdout_handle.close()
                except OSError:
                    pass
                self.registry.update_status(
                    bot_id,
                    status=BotStatus.FAILED,
                    error=f"launch failed: {e}",
                )
                raise SupervisorError(f"Failed to launch {bot_id}: {e}") from e

            started_at = time.time()
            restarts = self._prior_restarts(bot_id)
            bp = BotProcess(
                bot_id=bot_id,
                process=proc,
                log_path=log_path,
                started_at=started_at,
                restarts=restarts,
                stdout_handle=stdout_handle,
            )
            self._procs[bot_id] = bp
            # Guard against registry regression: the child may already have
            # written FAILED (fast config error) before we get here. In that
            # case, mark STARTING is WRONG — it would make the reconcile pass
            # below skip the terminal FAILED branch and try to restart the
            # broken bot. So we re-read the current status and only write
            # STARTING if the child hasn't already declared FAILED.
            current = self.registry.get_bot(bot_id) or {}
            if current.get("status") != BotStatus.FAILED:
                self.registry.update_status(
                    bot_id,
                    status=BotStatus.STARTING,
                    pid=proc.pid,
                    error=None,
                )
            return bp.to_dict()

    def stop(self, bot_id: str) -> Dict[str, Any]:
        """
        Requests graceful shutdown: sets ``intentional_stop`` (disables auto-
        restart), sends a terminate signal, and waits up to
        ``_STOP_GRACE_SECONDS``. If the process hasn't exited, it is killed.
        """
        validate_bot_id(bot_id)
        with self._lock:
            bp = self._procs.get(bot_id)
            if bp is None or not bp.is_alive():
                self.registry.update_status(bot_id, status=BotStatus.STOPPED)
                return {
                    "bot_id": bot_id,
                    "status": BotStatus.STOPPED,
                    "was_running": False,
                }
            bp.intentional_stop = True
            self.registry.update_status(bot_id, status=BotStatus.STOPPING)
        try:
            self._signal_terminate(bp)
        except Exception as e:
            ASCIIColors.warning(f"[Supervisor] terminate {bot_id} failed: {e}")
        exited = self._wait_for_exit(bp, self._STOP_GRACE_SECONDS)
        if not exited:
            try:
                self._kill_tree(bp)
            except Exception as e:
                ASCIIColors.warning(f"[Supervisor] kill {bot_id} failed: {e}")
            # One more bounded wait after SIGKILL.
            self._wait_for_exit(bp, 5.0)
        self.registry.update_status(bot_id, status=BotStatus.STOPPED)
        return {
            "bot_id": bot_id,
            "status": BotStatus.STOPPED,
            "was_running": True,
            "pid": bp.process.pid,
        }

    def kill(self, bot_id: str) -> Dict[str, Any]:
        """
        Force-terminates a bot's process tree and marks the registry FAILED
        (killing is not a clean stop). Used when the operator demands an
        immediate halt without grace.

        Semantics: kill ALWAYS terminates the bot in the FAILED state — even
        if the process had already exited on its own. This is the operator's
        explicit "abnormal termination" signal; conflating it with STOPPED
        would let the *next* watchdog reconcile auto-restart a bot the
        operator just killed.
        """
        validate_bot_id(bot_id)
        # Kill ALWAYS terminates in FAILED — even if the process had already
        # exited on its own. This is the operator's explicit "abnormal
        # termination" signal; conflating it with STOPPED would let a
        # subsequent reconcile auto-restart a bot the operator just killed.
        with self._lock:
            bp = self._procs.get(bot_id)
            was_running = bp is not None and bp.is_alive()
            pid: Optional[int] = bp.process.pid if bp is not None else None
            if bp is not None:
                # Set intentional_stop BEFORE killing so any concurrent
                # reconcile sees it and skips auto-restart.
                bp.intentional_stop = True
            if was_running:
                self.registry.update_status(bot_id, status=BotStatus.STOPPING)
        if was_running and bp is not None:
            try:
                self._kill_tree(bp)
            except Exception as e:
                ASCIIColors.warning(f"[Supervisor] kill {bot_id} failed: {e}")
            self._wait_for_exit(bp, 5.0)
        self.registry.update_status(bot_id, status=BotStatus.FAILED)
        return {
            "bot_id": bot_id,
            "status": BotStatus.FAILED,
            "was_running": was_running,
            "pid": pid,
        }

    def restart(self, bot_id: str) -> Dict[str, Any]:
        """Stop (if needed) then spawn. Clears the restart budget."""
        with self._lock:
            bp = self._procs.get(bot_id)
            if bp is not None and bp.is_alive():
                bp.intentional_stop = True
                try:
                    self._signal_terminate(bp)
                except Exception:
                    pass
                self._wait_for_exit(bp, self._STOP_GRACE_SECONDS)
                if bp.is_alive():
                    self._kill_tree(bp)
                    self._wait_for_exit(bp, 5.0)
            self._restart_history.pop(bot_id, None)
        return self.spawn(bot_id)

    # ------------------------------------------------------------ internals
    def _build_command(self, bot_id: str) -> List[str]:
        """
        Builds the child command. Runs the mesh-bot bridge as a module so the
        child re-imports cleanly (no pickled parent state, no fork issues on
        Windows). The mesh_id is derived from the layout's directory name —
        this is the canonical identifier.
        """
        return [
            self._python,
            "-m",
            "lollms_client.apps.lollms_bot.mesh.runtime",
            "--mesh-root",
            str(self.layout.root.parent),
            "--mesh-id",
            self.layout.root.name,
            "--bot-id",
            bot_id,
        ]

    def _build_env(self, bot_id: str, spec: BotSpec) -> Dict[str, str]:
        """
        Child environment. Propagates the current env (so the child inherits
        the same LOLLMS_* and PATH state) and adds the mesh-specific
        variables the runtime needs.
        """
        env = dict(os.environ)
        env["LOLLMS_BOT_MESH_ROOT"] = str(self.layout.root.parent)
        env["LOLLMS_BOT_MESH_ID"] = self.layout.root.name
        env["LOLLMS_BOT_MESH_BOT_ID"] = bot_id
        env["PYTHONUNBUFFERED"] = "1"
        # Prevent recursion if the bot's env ever leaks into another spawn.
        return env

    def _launch_process(
        self, cmd: List[str], env: Dict[str, str], stdout_handle
    ) -> subprocess.Popen:
        """
        Spawns the child. ``start_new_session=True`` on POSIX guarantees the
        child is in its own process group so ``killpg`` targets exactly it;
        on Windows we use CREATE_NEW_PROCESS_GROUP for the equivalent.
        """
        kwargs: Dict[str, Any] = dict(
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP lets taskkill /T target the tree.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(cmd, **kwargs)

    def _signal_terminate(self, bp: BotProcess) -> None:
        """Sends SIGTERM (POSIX) / CTRL_BREAK (Windows) to the child tree."""
        if not bp.is_alive():
            return
        if os.name == "nt":
            # On Windows, CTRL_BREAK_EVENT is the closest polite signal. Some
            # child stacks ignore it and require kill_tree; _wait_for_exit
            # enforces the grace timeout, so this stays predictable.
            try:
                bp.process.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                bp.process.terminate()
        else:
            try:
                os.killpg(os.getpgid(bp.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _kill_tree(self, bp: BotProcess) -> None:
        """Forcefully kills a bot and its children."""
        if os.name == "nt":
            taskkill = shutil.which("taskkill")
            if taskkill:
                try:
                    subprocess.run(
                        [taskkill, "/T", "/F", "/PID", str(bp.process.pid)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception:
                    pass
            # Fallback: best-effort terminate.
            try:
                bp.process.kill()
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(bp.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _wait_for_exit(self, bp: BotProcess, timeout: float) -> bool:
        """Blocks up to ``timeout`` seconds; returns True if the process exited."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not bp.is_alive():
                return True
            time.sleep(0.1)
        return not bp.is_alive()

    def _prior_restarts(self, bot_id: str) -> int:
        bp = self._procs.get(bot_id)
        return bp.restarts if bp is not None else 0

    def _record_restart_attempt(self, bot_id: str) -> int:
        """Records a restart attempt and returns the count within the window."""
        with self._lock:
            now = time.time()
            dq = self._restart_history.setdefault(bot_id, deque())
            cutoff = now - self._RESTART_WINDOW_SECONDS
            while dq and dq[0] < cutoff:
                dq.popleft()
            dq.append(now)
            return len(dq)

    def _backoff_seconds(self, restarts: int) -> float:
        return min(
            self._BACKOFF_BASE_SECONDS * (2 ** max(0, restarts - 1)),
            self._BACKOFF_MAX_SECONDS,
        )

    # -------------------------------------------------------- watchdog loop
    def _watch_loop(self) -> None:
        while not self._watch_stop.wait(self._WATCH_INTERVAL_SECONDS):
            try:
                self._reconcile_once()
            except Exception as e:
                ASCIIColors.warning(f"[Supervisor] reconcile error: {e}")

    def _reconcile_once(self) -> None:
        """
        One pass of reconciliation:
          1. For every tracked bot, sync registry status to PID liveness.
          2. For any bot whose process died non-intentionally, apply the
             restart policy.
          3. Fresh bots whose registry says RUNNING but we don't track them
             are left alone — this supervisor only oversees bots it spawned.
        """
        now = time.time()
        with self._lock:
            snapshot = list(self._procs.items())
        for bot_id, bp in snapshot:
            alive = bp.is_alive()
            rc = bp.process.returncode
            if alive:
                # Startup grace period: a fresh process that survived
                # _STARTUP_GRACE_SECONDS is considered RUNNING. Before that
                # it stays STARTING (the child might still be importing
                # lollms_client, which on a cold install is not instantaneous).
                if now - bp.started_at >= self._STARTUP_GRACE_SECONDS:
                    self.registry.update_status(
                        bot_id, status=BotStatus.RUNNING, heartbeat=True
                    )
                else:
                    self.registry.update_status(bot_id, heartbeat=True)
                continue

            # Process has exited.
            try:
                bp.stdout_handle.close()
            except Exception:
                pass

            if bp.intentional_stop:
                # stop()/kill() already set the terminal status; nothing to do.
                continue

            # If the CHILD already declared FAILED (e.g. personality build
            # error, missing handbag, bad profile), that is an explicit,
            # terminal signal: the bot knows it cannot run as configured.
            # NEVER auto-restart a self-declared FAILED — otherwise a
            # permanently mis-configured bot burns the whole restart budget
            # in seconds and spams the whole mesh.
            entry = self.registry.get_bot(bot_id) or {}
            if entry.get("status") == BotStatus.FAILED:
                ASCIIColors.warning(
                    f"[Supervisor] Bot {bot_id!r} exited rc={rc} after "
                    f"declaring FAILED — not restarting. "
                    f"Fix the spec, then restart manually."
                )
                continue

            # Unintentional exit — decide: restart or give up?
            attempted = self._record_restart_attempt(bot_id)
            if attempted > self._MAX_RESTARTS:
                self.registry.update_status(
                    bot_id,
                    status=BotStatus.FAILED,
                    error=(
                        f"exited rc={rc} after {attempted} restart attempts "
                        f"within {int(self._RESTART_WINDOW_SECONDS)}s; giving up"
                    ),
                )
                continue

            backoff = self._backoff_seconds(attempted)
            ASCIIColors.yellow(
                f"[Supervisor] Bot {bot_id!r} exited rc={rc}; "
                f"restart in {backoff:.1f}s (attempt {attempted}/{self._MAX_RESTARTS})"
            )
            self.registry.update_status(
                bot_id,
                status=BotStatus.STARTING,
                error=f"restart pending (rc={rc}, attempt {attempted})",
            )
            self._restart_history.setdefault(bot_id, deque())  # ensure exists
            # Schedule the respawn from a fresh thread — the watchdog must
            # never sleep on one bot and starve the others.
            threading.Thread(
                target=self._delayed_respawn,
                args=(bot_id, backoff, attempted),
                daemon=True,
                name=f"mesh-respawn-{bot_id}",
            ).start()

    def _delayed_respawn(self, bot_id: str, delay: float, attempt: int) -> None:
        """Waits ``delay`` then respawns. Ignores stop/kill issued meanwhile."""
        time.sleep(delay)
        with self._lock:
            bp = self._procs.get(bot_id)
            if bp is not None and (bp.is_alive() or bp.intentional_stop):
                return
            entry = self.registry.get_bot(bot_id)
            if entry is None:
                return
            status = entry.get("status", "")
            if status in (BotStatus.STOPPING, BotStatus.STOPPED):
                return
        try:
            new_bp = self.spawn(bot_id)
            with self._lock:
                self._procs[bot_id].restarts = attempt
            ASCIIColors.green(
                f"[Supervisor] Bot {bot_id!r} respawned (pid {new_bp['pid']})"
            )
        except Exception as e:
            ASCIIColors.warning(f"[Supervisor] respawn {bot_id} failed: {e}")

    # ------------------------------------------------------------- janitor
    def stop_all(self) -> List[Dict[str, Any]]:
        """Stops every bot this supervisor spawned. Returns their final states."""
        with self._lock:
            ids = list(self._procs.keys())
        results: List[Dict[str, Any]] = []
        for bot_id in ids:
            try:
                results.append(self.stop(bot_id))
            except Exception as e:
                results.append({"bot_id": bot_id, "error": str(e)})
        return results
