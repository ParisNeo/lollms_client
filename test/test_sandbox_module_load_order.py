"""
Regression: module-load-order corruption of sandbox_kernel's TRUE_* builtins.

BUG: when `sandbox_kernel` was imported for the FIRST time while a
`SandboxContext` was already active, `builtins.compile`/`exec`/`eval` were the
sandbox's own blocking stubs. `_capture_true_builtins()` then cached the block
stub as the "true" builtin, permanently breaking every tool that needed
exec/compile — most visibly `execute_python_data_query`, which raised:

    SandboxViolation: 🛑 BLOCKED BY SANDBOX: dynamic code execution
    (eval/exec/compile) is forbidden.

FIX: version 2 of `_capture_true_builtins()` validates each candidate via
`_is_block_stub()` and recovers the genuine builtin from
`builtins.__dict__` when a sentinel is missing or corrupted. Additionally,
`execute_python` and `execute_python_data_query` re-resolve their exec
primitives at call time via `_resolve_exec_builtins()`.

This test exercises the exact failure mode observed in
`test_agent_cognitive_decisions::test_scenario_c_multistep_data_query`.
"""
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def fresh_workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td).resolve()


def _reload_kernel_inside_sandbox(fresh_workspace):
    """Force `sandbox_kernel` to re-import while a sandbox is already active.

    This mimics the real-world condition: a tool call is already inside a
    `SandboxContext` when the kernel module is (re)loaded — e.g. a test that
    runs before the module is warmed up, a run under a test that unloads and
    reloads modules, or an app path where a `SandboxContext` from
    `execute_python_data_query` is entered before `execute_python` triggers
    its own module import.
    """
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as sk

    # Snapshot the module's current state so we can revert after the test.
    import builtins as _b
    saved_compile = _b.compile
    saved_exec = _b.exec
    saved_eval = _b.eval
    saved_tc = getattr(_b, "__lollms_true_compile__", None)
    saved_te = getattr(_b, "__lollms_true_exec__", None)
    saved_tv = getattr(_b, "__lollms_true_eval__", None)
    saved_sys_module = sys.modules.get(sk.__name__)
    saved_tc_const = sk.TRUE_COMPILE
    saved_te_const = sk.TRUE_EXEC
    saved_tv_const = sk.TRUE_EVAL
    try:
        # Enter a sandbox (which replaces builtins.compile/exec/eval with
        # block stubs) and simulate a "fresh import" of sandbox_kernel.
        with sk.SandboxContext(workspace_root=fresh_workspace, profile="standard"):
            # Force a reimport: drop from sys.modules and reload.
            sys.modules.pop(sk.__name__, None)
            import importlib
            fresh_module = importlib.import_module(
                "lollms_client.tools_bindings.lcp.default_tools.execute_python.sandbox_kernel"
            )
            # THE ASSERTION: the fresh module must NOT have captured stubs.
            assert not fresh_module._is_block_stub(fresh_module.TRUE_COMPILE), (
                "TRUE_COMPILE was corrupted to a block stub — module load-order "
                "bug regressed."
            )
            assert not fresh_module._is_block_stub(fresh_module.TRUE_EXEC), (
                "TRUE_EXEC was corrupted to a block stub — module load-order bug "
                "regressed."
            )
            # And it must actually execute real Python.
            exec_result = {}
            fresh_module.TRUE_EXEC(
                fresh_module.TRUE_COMPILE("x = 1 + 1", "<reload_check>", "exec"),
                exec_result,
            )
            assert exec_result["x"] == 2
    finally:
        # Restore module identity and builtins so other tests are unaffected.
        _b.compile = saved_compile
        _b.exec = saved_exec
        _b.eval = saved_eval
        if saved_tc is not None:
            _b.__lollms_true_compile__ = saved_tc
        if saved_te is not None:
            _b.__lollms_true_exec__ = saved_te
        if saved_tv is not None:
            _b.__lollms_true_eval__ = saved_tv
        if saved_sys_module is not None:
            sys.modules[sk.__name__] = saved_sys_module
            saved_sys_module.TRUE_COMPILE = saved_tc_const
            saved_sys_module.TRUE_EXEC = saved_te_const
            saved_sys_module.TRUE_EVAL = saved_tv_const


def test_reimport_inside_live_sandbox_does_not_capture_block_stub(fresh_workspace):
    """Reproducer: importing the kernel mid-sandbox must not cache the stub."""
    _reload_kernel_inside_sandbox(fresh_workspace)


def test_data_query_tool_survives_module_load_order_corruption(fresh_workspace):
    """End-to-end: after the corruption scenario, data_query tool still runs."""
    pytest.importorskip("pandas")
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )

    # Force a corrupted TRUE_EXEC on the data-query module (simulates the
    # exact failure state). `_resolve_exec_builtins()` must reject it.
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as sk
    epdq._TRUE_COMPILE = sk._make_block_stub()
    epdq._TRUE_EXEC = sk._make_block_stub()

    # A benign data-query call must still succeed — the tool re-resolves.
    result = epdq.tool_execute_python_data_query(
        code="import pandas as pd\nprint(int(pd.DataFrame({'a':[1,2,3]}).sum().iloc[0]))",
        env=None,
    )
    assert result["success"] is True, result.get("error")
    assert "6" in result["output"]


def test_execute_python_tool_survives_module_load_order_corruption(fresh_workspace):
    """Same end-to-end sanity for the standard profile tool."""
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as sk

    ep._TRUE_COMPILE = sk._make_block_stub()
    ep._TRUE_EXEC = sk._make_block_stub()

    result = ep.tool_execute_python_code(code="print(2+2)")
    assert result["success"] is True, result.get("error")
    assert "4" in result["output"]