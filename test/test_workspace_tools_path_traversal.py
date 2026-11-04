"""
Exploit / regression suite for workspace_tools sandboxing.

Covers CWE-22 (path traversal), CWE-59 (link following), and CWE-200 (info
disclosure via grep/find crawling outside the workspace).

Run with:  pytest test/test_workspace_tools_path_traversal.py -v
"""
import os
import sys
from pathlib import Path

import pytest

# Make the *inner* `src/` directory importable so `lollms_client` resolves
# directly. We deliberately do NOT put the repo root on sys.path and do NOT
# import via `src.lollms_client`: the top-level name `src` is hijackable by an
# unrelated PyPI package of the same name (`site-packages/src/__init__.py`),
# which would shadow the repo tree and crash on its own missing deps
# (`ModuleNotFoundError: No module named 'VirtEnv'`). See scripts/verify_sandbox.py
# and test/test_workspace_persistence.py for the canonical import pattern.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lollms_client.tools_bindings.lcp.default_tools.workspace_tools import workspace_tools as wt


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Chdir into an isolated workspace root and restore afterwards."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("INSIDE_MARKER", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("OUTSIDE_SECRET_MARKER", encoding="utf-8")

    monkeypatch.chdir(workspace)
    yield {"workspace": workspace, "outside": outside}


# ---------------------------------------------------------------------------
# 1. Direct path traversal on read/write must be rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "../outside/secret.txt",
    "../../etc/passwd",
    "..\\..\\outside\\secret.txt",
    "subdir/../../../outside/secret.txt",
    "./../../outside/secret.txt",
])
def test_read_file_blocks_relative_traversal(sandbox, payload):
    result = wt.tool_read_file(payload)
    assert result["success"] is False, f"Traversal not blocked for payload: {payload!r}"
    assert "escape" in result["error"].lower() or "traversal" in result["error"].lower()


@pytest.mark.parametrize("payload", [
    "../outside/evil.txt",
    "../../outside/evil.txt",
])
def test_write_file_blocks_relative_traversal(sandbox, payload):
    result = wt.tool_write_file(payload, "pwned")
    assert result["success"] is False
    assert not (sandbox["outside"] / "evil.txt").exists()


def test_read_file_blocks_absolute_outside_path(sandbox):
    abs_outside = str(sandbox["outside"] / "secret.txt")
    result = wt.tool_read_file(abs_outside)
    assert result["success"] is False
    assert "traversal" in result["error"].lower() or "escape" in result["error"].lower()


def test_write_file_blocks_absolute_outside_path(sandbox):
    abs_outside = str(sandbox["outside"] / "evil.txt")
    result = wt.tool_write_file(abs_outside, "pwned")
    assert result["success"] is False
    assert not (sandbox["outside"] / "evil.txt").exists()


# ---------------------------------------------------------------------------
# 2. Home-expansion & null-byte & device-name rejection.
# ---------------------------------------------------------------------------

def test_read_file_blocks_home_expansion(sandbox):
    result = wt.tool_read_file("~/.bashrc")
    assert result["success"] is False


def test_read_file_blocks_null_byte(sandbox):
    result = wt.tool_read_file("inside.txt\x00../../outside/secret.txt")
    assert result["success"] is False
    assert "null byte" in result["error"].lower()


# ---------------------------------------------------------------------------
# 3. Symlink escape (CWE-59) — grep and find must not leak outside content.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privilege on Windows")
def test_read_file_blocks_symlink_to_outside(sandbox):
    link = sandbox["workspace"] / "link_to_secret.txt"
    link.symlink_to(sandbox["outside"] / "secret.txt")
    result = wt.tool_read_file("link_to_secret.txt")
    assert result["success"] is False
    assert "traversal" in result["error"].lower() or "escape" in result["error"].lower()


@pytest.mark.skipif(os.name == "nt", reason="symlink directory creation requires privilege on Windows")
def test_grep_files_does_not_follow_symlinked_directory(sandbox):
    (sandbox["workspace"] / "leakdir").symlink_to(sandbox["outside"], target_is_directory=True)
    result = wt.tool_grep_files(pattern="OUTSIDE_SECRET_MARKER", search_mode="case_sensitive")
    assert result["success"] is True
    assert result["matches_count"] == 0, (
        "grep leaked content from outside the workspace via symlinked directory: "
        f"{result.get('output')!r}"
    )


@pytest.mark.skipif(os.name == "nt", reason="symlink directory creation requires privilege on Windows")
def test_find_files_does_not_escape_via_symlink(sandbox):
    (sandbox["workspace"] / "leakdir").symlink_to(sandbox["outside"], target_is_directory=True)
    result = wt.tool_find_files(pattern="secret.txt", path=".")
    assert result["success"] is True
    for match in result["files"]:
        assert "outside" not in match, f"find escaped workspace: {match!r}"


# ---------------------------------------------------------------------------
# 4. Positive path: genuine workspace files still reachable.
# ---------------------------------------------------------------------------

def test_read_inside_workspace_still_works(sandbox):
    result = wt.tool_read_file("inside.txt")
    assert result["success"] is True
    assert "INSIDE_MARKER" in result["output"]


def test_write_inside_workspace_still_works(sandbox):
    result = wt.tool_write_file("newdir/newfile.txt", "hello")
    assert result["success"] is True
    assert (sandbox["workspace"] / "newdir" / "newfile.txt").read_text(encoding="utf-8") == "hello"


def test_grep_inside_workspace_still_works(sandbox):
    result = wt.tool_grep_files(pattern="INSIDE_MARKER", search_mode="case_sensitive")
    assert result["success"] is True
    assert result["matches_count"] >= 1


def test_find_inside_workspace_still_works(sandbox):
    result = wt.tool_find_files(pattern="inside.txt", path=".")
    assert result["success"] is True
    assert any("inside.txt" in m for m in result["files"])