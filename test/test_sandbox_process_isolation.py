"""
Parent-process isolation contract for the sandbox.

The `SandboxContext` mutates builtins/os/shutil/socket/subprocess/importlib/
pathlib AND 13 cache env vars. Those mutations MUST NEVER reach the parent
process. The task tool must (a) run user code in a child, and (b) leave the
parent byte-for-byte unmodified when the child finishes — success, error,
sandbox violation, or crash.

These tests are the direct contract for that architectural decision. If any
of them fails, isolation has regressed and the sensitive tools are once again
mutating process-global state.
"""
from __future__ import annotations

import builtins
import os
import socket
import subprocess
from pathlib import Path

import pytest


_ISOLATION_ATTRS = [
    (builtins, "compile"),
    (builtins, "exec"),
    (builtins, "eval"),
    (builtins, "open"),
    (builtins, "__import__"),
    (os, "system"),
    (os, "remove"),
    (os, "mkdir"),
    (os, "chdir"),
    (subprocess, "run"),
    (subprocess, "Popen"),
    (socket, "create_connection"),
    (Path, "mkdir"),
    (Path, "write_text"),
    (Path, "open"),
]


def _snapshot_attrs():
    return {(id(obj), name): getattr(obj, name, None) for obj, name in _ISOLATION_ATTRS}


def _assert_unchanged(before, after):
    dirty = []
    for (obj_id, name), prev in before.items():
        # Resolve the object back from the global namespace.
        obj = next(o for o, n in _ISOLATION_ATTRS if id(o) == obj_id and n == name)
        current = getattr(obj, name, None)
        if current is not prev:
            dirty.append(f"{obj.__name__ if hasattr(obj, '__name__') else obj}.{name}")
    assert not dirty, (
        f"Sandbox leaked into the parent process. These attributes were "
        f"mutated and not restored: {dirty}"
    )


def test_data_query_success_leaves_parent_untouched(tmp_path):
    pytest.importorskip("pandas")
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )

    before = _snapshot_attrs()

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = epdq.tool_execute_python_data_query(
            code="import pandas as pd\nprint(int(pd.DataFrame({'a':[1,2,3]}).sum().iloc[0]))",
            env=None,
        )
        assert result["success"] is True, result.get("error")
        assert "6" in result["output"]
    finally:
        os.chdir(cwd)

    _assert_unchanged(before, _snapshot_attrs())


def test_data_query_static_rejection_leaves_parent_untouched(tmp_path):
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )

    before = _snapshot_attrs()

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = epdq.tool_execute_python_data_query(
            code="import subprocess\nsubprocess.run(['whoami'])",
            env=None,
        )
        assert result["success"] is False
        assert "subprocess" in result["error"].lower()
    finally:
        os.chdir(cwd)

    _assert_unchanged(before, _snapshot_attrs())


def test_data_query_runtime_error_leaves_parent_untouched(tmp_path):
    pytest.importorskip("pandas")
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )

    before = _snapshot_attrs()

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = epdq.tool_execute_python_data_query(
            code="import pandas as pd\nraise RuntimeError('intentional test failure')",
            env=None,
        )
        assert result["success"] is False
        assert "intentional test failure" in result["error"]
    finally:
        os.chdir(cwd)

    _assert_unchanged(before, _snapshot_attrs())


def test_execute_python_success_leaves_parent_untouched():
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep

    before = _snapshot_attrs()
    result = ep.tool_execute_python_code(code="print(2+2)")
    assert result["success"] is True, result.get("error")
    assert "4" in result["output"]
    _assert_unchanged(before, _snapshot_attrs())


def test_execute_python_runtime_error_leaves_parent_untouched():
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep

    before = _snapshot_attrs()
    result = ep.tool_execute_python_code(code="raise ValueError('boom')")
    assert result["success"] is False
    assert "boom" in result["error"]
    _assert_unchanged(before, _snapshot_attrs())


def test_execute_python_static_rejection_leaves_parent_untouched():
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep

    before = _snapshot_attrs()
    result = ep.tool_execute_python_code(code="import ctypes")
    assert result["success"] is False
    _assert_unchanged(before, _snapshot_attrs())


def test_execute_python_still_blocks_exploits_after_isolation():
    """Sandbox invariants survive the move to a child process."""
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep

    for exploit in (
        "import subprocess\nsubprocess.run(['whoami'])",
        "import socket\ns=socket.socket()\ns.connect(('127.0.0.1', 65535))",
        "import ctypes",
        "eval('1+1')",
        "exec('print(1)')",
    ):
        result = ep.tool_execute_python_code(code=exploit)
        assert result["success"] is False, (
            f"Exploit was not blocked post-isolation: {exploit!r}. "
            f"Got: {result!r}"
        )


def test_data_query_isolation_blocks_escape(tmp_path):
    pytest.importorskip("pandas")
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = epdq.tool_execute_python_data_query(
            code="import os\nos.system('whoami')",
            env=None,
        )
        assert result["success"] is False
    finally:
        os.chdir(cwd)