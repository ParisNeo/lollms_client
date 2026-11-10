"""Regression tests for the sandbox directory-enumeration policy.

The contract enforced by ``SandboxContext`` (see ``sandbox_kernel.py``):

  * Listing / scanning / walking directories **inside** the workspace root is
    LEGITIMATE and MUST succeed. This is the everyday ``os.listdir('.')``,
    ``glob.glob('**/*')``, ``os.walk(workspace)`` used by data-analysis and
    file-processing scripts.
  * Listing / scanning / walking any directory **outside** the workspace root
    (``/etc``, ``C:\\Users``, ``~/.ssh``, ``<workspace>/../sibling``) MUST raise
    ``SandboxPermissionError``. Before the enumeration guards were added, these
    calls executed silently, leaking the host filesystem layout even though
    ``open('/etc/passwd')`` was already blocked.

``glob.glob`` is guarded explicitly: CPython <= 3.10 binds ``scandir`` BY
VALUE inside the ``glob`` module at its own import time, so patching
``os.scandir`` does NOT rebind ``glob.scandir``. These tests assert both the
direct ``os.*`` surface and the ``glob`` surface, because the two guards are
independent by design — a future refactor that patched only one must not
silently re-open the gap.
"""

import glob
import os
from pathlib import Path

import pytest

from lollms_client.tools_bindings.lcp.default_tools.execute_python.sandbox_kernel import (
    SandboxContext,
    SandboxPermissionError,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace root containing a nested directory and a file."""
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    return root.resolve()


@pytest.fixture
def outside_dir(tmp_path: Path, workspace: Path) -> Path:
    """A directory that exists but lives OUTSIDE the workspace root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not read me", encoding="utf-8")
    return outside.resolve()


# ── Legitimate enumeration INSIDE the workspace ─────────────────────────────

def test_listdir_inside_workspace_is_allowed(workspace: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        entries = os.listdir(str(workspace))
    assert "sub" in entries
    assert "notes.txt" in entries


def test_scandir_inside_workspace_is_allowed(workspace: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        names = sorted(entry.name for entry in os.scandir(str(workspace)))
    # The sandbox installs its own ``.sandbox_cache`` bookkeeping directory
    # inside the workspace root; it is legitimate sandbox housekeeping, not
    # part of the user's data. Filter internal entries before asserting.
    user_visible = [name for name in names if not name.startswith(".sandbox_")]
    assert user_visible == ["notes.txt", "sub"]


def test_walk_inside_workspace_is_allowed(workspace: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        visited = {
            Path(dirpath).resolve()
            for dirpath, _dirnames, _filenames in os.walk(str(workspace))
        }
    assert workspace in visited
    assert (workspace / "sub") in visited


def test_glob_inside_workspace_is_allowed(workspace: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        hits = glob.glob(str(workspace / "**" / "*.csv"), recursive=True)
    assert len(hits) == 1
    assert Path(hits[0]).name == "data.csv"


# ── Enumeration OUTSIDE the workspace is blocked ────────────────────────────

def test_listdir_outside_workspace_is_blocked(workspace: Path, outside_dir: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        with pytest.raises(SandboxPermissionError):
            os.listdir(str(outside_dir))


def test_scandir_outside_workspace_is_blocked(workspace: Path, outside_dir: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        with pytest.raises(SandboxPermissionError):
            os.scandir(str(outside_dir))


def test_walk_outside_workspace_is_blocked(workspace: Path, outside_dir: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        with pytest.raises(SandboxPermissionError):
            # ``next`` forces the generator, which triggers the seed-path guard.
            next(os.walk(str(outside_dir)))


def test_glob_outside_workspace_is_blocked(workspace: Path, outside_dir: Path):
    # ``glob.glob`` reaches the filesystem through ``os.scandir``; asserting
    # that the guard fires here proves the transitive coverage claimed by the
    # guard-comment in ``sandbox_kernel._install``.
    with SandboxContext(workspace_root=workspace, profile="standard"):
        with pytest.raises(SandboxPermissionError):
            glob.glob(str(outside_dir / "*"))


def test_dotdot_traversal_is_blocked(workspace: Path):
    with SandboxContext(workspace_root=workspace, profile="standard"):
        with pytest.raises(SandboxPermissionError):
            os.listdir(str(workspace / ".."))


# ── Symmetric restore: the host is never left patched ───────────────────────

def test_guards_are_restored_after_context_exit(workspace: Path, outside_dir: Path):
    # Capture the originals before entering the sandbox context.
    original_listdir = os.listdir
    original_scandir = os.scandir
    original_walk = os.walk

    with SandboxContext(workspace_root=workspace, profile="standard"):
        # Inside the scope the originals are replaced by bounded guards.
        assert os.listdir is not original_listdir
        assert os.scandir is not original_scandir
        assert os.walk is not original_walk

    # After exit, the host process must be pristine — no guard leak.
    assert os.listdir is original_listdir
    assert os.scandir is original_scandir
    assert os.walk is original_walk
    # And the real calls work again on the host.
    assert "secret.txt" in os.listdir(str(outside_dir))