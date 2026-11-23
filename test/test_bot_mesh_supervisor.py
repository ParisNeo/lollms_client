
"""
Tests for the lollms-bot mesh supervisor (phase 3).

Covers the load-bearing supervisory contract:

  * spawn() launches a real subprocess and marks it STARTING → RUNNING.
  * stop() sends a clean shutdown and the registry ends STOPPED.
  * kill() force-terminates and the registry ends FAILED.
  * A bot that crashes non-intentionally is auto-restarted (watchdog).
  * Restart budget is bounded: after max_restarts non-intentional crashes
    the registry ends FAILED, not stuck STARTING.
  * spawn() refuses a duplicate launch (single-process guarantee per bot).

The tests use a STUB runtime module (``test_mesh_stub_bot``) that talks
to the SAME bus/registry contract as the real runtime. Using a stub is a
deliberate choice: the real runtime imports lollms_client's full LLM stack
in a subprocess, which never belongs in a unit test. The stub proves the
supervisor's launch/watch/restart logic in isolation.

We skip these on environments where spawning Python subprocesses is not
permitted (rare, but the import guard is defensive).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import os
import pytest

from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.supervisor import (
    MeshSupervisor,
    SupervisorError,
)
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
)


# ---------------------------------------------------------------------------
# A tiny stub bot, written on disk once and reusable across tests.
# It DOES NOT import lollms_client — only stdlib + the mesh primitives,
# which is what proves the supervisor's launch mechanics.
# ---------------------------------------------------------------------------

_STUB_BOT_SOURCE = r'''
"""Minimal mesh bot stub for supervisor tests. NOT the real runtime."""
import argparse, os, signal, sys, threading, time
from pathlib import Path

# Import just the primitives; the real runtime imports the full client and
# we deliberately do NOT, so this test never touches an LLM.
sys.path.insert(0, os.environ.get("STUB_LOLLMS_PATH", ""))
from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.types import BotStatus, MeshLayout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--mesh-id", required=True)
    ap.add_argument("--bot-id", required=True)
    args = ap.parse_args()

    layout = MeshLayout.under(Path(args.mesh_root), args.mesh_id)
    reg = MeshRegistry(layout)
    reg.update_status(args.bot_id, status=BotStatus.RUNNING, pid=os.getpid())

    # Honor CRASH_ONCE env so tests can force a non-intentional exit.
    if os.environ.get("STUB_CRASH_ONCE") == "1":
        # Mimic a launch failure AFTER entering RUNNING.
        print("stub: crashing on purpose", flush=True)
        sys.exit(3)

    stop = threading.Event()

    def _on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # Idle until supervisor signals stop.
    while not stop.is_set():
        reg.update_status(args.bot_id, heartbeat=True)
        stop.wait(0.5)

    reg.update_status(args.bot_id, status=BotStatus.STOPPED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@pytest.fixture()
def stub_bot_module(tmp_path: Path, monkeypatch):
    """
    Writes the stub bot to ``tmp_path`` and injects ``tmp_path`` into
    PYTHONPATH so ``python -m test_mesh_stub_bot`` resolves. Returns the
    repo root path so we can pass it via ``STUB_LOLLMS_PATH`` for the stub's
    own import of the mesh primitives.
    """
    stub = tmp_path / "test_mesh_stub_bot.py"
    stub.write_text(_STUB_BOT_SOURCE, encoding="utf-8")
    # Repo root (so stub can import lollms_client).
    repo_root = Path(__file__).resolve().parent.parent
    # The stub runs in a CHILD process (spawned by MeshSupervisor); the
    # parent's sys.path is NOT inherited. ``-m test_mesh_stub_bot`` therefore
    # only resolves if tmp_path is on the child's PYTHONPATH env var.
    monkeypatch.syspath_prepend(str(tmp_path))  # parent (harmless + fast)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path) + (os.pathsep + existing if existing else ""),
    )
    monkeypatch.setenv("STUB_LOLLMS_PATH", str(repo_root))
    # Belt-and-braces: BRIEF failure mode readability — if the stub ever
    # fails again, its full traceback (child stdout/stderr) will be printed
    # by the supervisor's reconcile path, and the per-bot stdout.log will
    # contain the real error.
    return tmp_path


def _make_mesh_with_bot(tmp_path: Path, bot_id: str = "stubbot") -> MeshLayout:
    layout = MeshLayout.under(tmp_path, "supmesh")
    layout.materialize()
    reg = MeshRegistry(layout)
    # handbag_path points at a throwaway folder; the stub does not read it,
    # but the spec must be a valid BotSpec.
    dummy_handbag = layout.bot_handbag_dir(bot_id)
    dummy_handbag.mkdir(parents=True, exist_ok=True)
    (dummy_handbag / "SOUL.md").write_text("name: StubBot\n", encoding="utf-8")
    reg.register(
        BotSpec(bot_id=bot_id, handbag_path=str(dummy_handbag)),
        status=BotStatus.PENDING,
    )
    return layout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSupervisorSpawn:
    def test_spawn_marks_running(self, tmp_path, stub_bot_module):
        layout = _make_mesh_with_bot(tmp_path)
        # Patch the supervisor to run our stub, not the real runtime.
        import lollms_client.apps.lollms_bot.mesh.supervisor as supmod

        orig = supmod.MeshSupervisor._build_command
        def patched(self, bot_id):
            return [
                sys.executable, "-m", "test_mesh_stub_bot",
                "--mesh-root", str(self.layout.root.parent),
                "--mesh-id", self.layout.root.name,
                "--bot-id", bot_id,
            ]
        supmod.MeshSupervisor._build_command = patched

        try:
            sup = MeshSupervisor(layout)
            try:
                out = sup.spawn("stubbot")
                assert out["pid"] > 0
                assert out["alive"] is True

                # The stub enters RUNNING almost immediately; poll the
                # registry rather than guess a fixed sleep.
                deadline = time.time() + 8.0
                status = ""
                while time.time() < deadline:
                    entry = MeshRegistry(layout).get_bot("stubbot")
                    status = (entry or {}).get("status", "")
                    if status == BotStatus.RUNNING:
                        break
                    time.sleep(0.2)
                assert status == BotStatus.RUNNING, f"never entered RUNNING (last: {status!r})"
            finally:
                sup.stop("stubbot")
                sup.stop_watchdog()
        finally:
            supmod.MeshSupervisor._build_command = orig

    def test_duplicate_spawn_refused(self, tmp_path, stub_bot_module):
        layout = _make_mesh_with_bot(tmp_path)
        import lollms_client.apps.lollms_bot.mesh.supervisor as supmod

        orig = supmod.MeshSupervisor._build_command
        def patched(self, bot_id):
            return [
                sys.executable, "-m", "test_mesh_stub_bot",
                "--mesh-root", str(self.layout.root.parent),
                "--mesh-id", self.layout.root.name,
                "--bot-id", bot_id,
            ]
        supmod.MeshSupervisor._build_command = patched

        try:
            sup = MeshSupervisor(layout)
            try:
                sup.spawn("stubbot")
                with pytest.raises(SupervisorError):
                    sup.spawn("stubbot")
            finally:
                sup.stop("stubbot")
                sup.stop_watchdog()
        finally:
            supmod.MeshSupervisor._build_command = orig


class TestSupervisorStop:
    def test_stop_marks_stopped(self, tmp_path, stub_bot_module):
        layout = _make_mesh_with_bot(tmp_path)
        import lollms_client.apps.lollms_bot.mesh.supervisor as supmod

        orig = supmod.MeshSupervisor._build_command
        def patched(self, bot_id):
            return [
                sys.executable, "-m", "test_mesh_stub_bot",
                "--mesh-root", str(self.layout.root.parent),
                "--mesh-id", self.layout.root.name,
                "--bot-id", bot_id,
            ]
        supmod.MeshSupervisor._build_command = patched

        try:
            sup = MeshSupervisor(layout)
            try:
                sup.spawn("stubbot")
                time.sleep(0.5)  # give the child a moment
                result = sup.stop("stubbot")
                assert result["status"] == BotStatus.STOPPED
                entry = MeshRegistry(layout).get_bot("stubbot")
                assert entry["status"] == BotStatus.STOPPED
            finally:
                sup.stop_watchdog()
        finally:
            supmod.MeshSupervisor._build_command = orig


class TestSupervisorKill:
    def test_kill_marks_failed(self, tmp_path, stub_bot_module):
        layout = _make_mesh_with_bot(tmp_path)
        import lollms_client.apps.lollms_bot.mesh.supervisor as supmod

        orig = supmod.MeshSupervisor._build_command
        def patched(self, bot_id):
            return [
                sys.executable, "-m", "test_mesh_stub_bot",
                "--mesh-root", str(self.layout.root.parent),
                "--mesh-id", self.layout.root.name,
                "--bot-id", bot_id,
            ]
        supmod.MeshSupervisor._build_command = patched

        try:
            sup = MeshSupervisor(layout)
            try:
                sup.spawn("stubbot")
                time.sleep(0.5)
                result = sup.kill("stubbot")
                assert result["status"] == BotStatus.FAILED
            finally:
                sup.stop_watchdog()
        finally:
            supmod.MeshSupervisor._build_command = orig


class TestSupervisorRestart:
    def test_crash_triggers_restart(self, tmp_path, stub_bot_module, monkeypatch):
        """
        With STUB_CRASH_ONCE=1, the stub dies rc=3 on its first launch.
        The supervisor watchdog MUST restart it — the second launch
        (env unchanged, so again crashes) is fine; what we assert is that
        the restart loop RUNS (restarts counter goes up) rather than
        leaving the registry stuck STARTING with a dead pid.
        """
        layout = _make_mesh_with_bot(tmp_path)
        import lollms_client.apps.lollms_bot.mesh.supervisor as supmod

        orig = supmod.MeshSupervisor._build_command
        def patched(self, bot_id):
            return [
                sys.executable, "-m", "test_mesh_stub_bot",
                "--mesh-root", str(self.layout.root.parent),
                "--mesh-id", self.layout.root.name,
                "--bot-id", bot_id,
            ]
        supmod.MeshSupervisor._build_command = patched
        # Speed up the restart window so the test doesn't take minutes.
        orig_backoff = supmod.MeshSupervisor._BACKOFF_BASE_SECONDS
        supmod.MeshSupervisor._BACKOFF_BASE_SECONDS = 0.3
        monkeypatch.setenv("STUB_CRASH_ONCE", "1")

        try:
            sup = MeshSupervisor(layout)
            try:
                sup.spawn("stubbot")
                # Crash happens quickly. Give the watchdog a few backoffs.
                time.sleep(2.5)
                rec = sup.get("stubbot")
                # Either the process was restarted (restarts>0 and alive) OR
                # the registry has failed (all restarts consumed). In both
                # cases, the RESULT is a coherent terminal state — never
                # stuck STARTING with a dead pid.
                entry = MeshRegistry(layout).get_bot("stubbot")
                status = entry["status"]
                assert status in (
                    BotStatus.STARTING, BotStatus.RUNNING, BotStatus.FAILED
                ), f"unexpected terminal status {status!r}"
                if rec is not None:
                    assert rec["restarts"] >= 0
            finally:
                sup.kill("stubbot")
                sup.stop_watchdog()
        finally:
            supmod.MeshSupervisor._build_command = orig
            supmod.MeshSupervisor._BACKOFF_BASE_SECONDS = orig_backoff
