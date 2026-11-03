"""
test_sandbox_python_kernel.py

Exploit reproduction + validation suite for the shared Python sandbox kernel
used by `execute_python` and `execute_python_data_query`.

Covers two layers:
  - Layer A — static AST validation (`validate_source`)
  - Layer B — runtime interception (`SandboxContext`)
  - Layer C — end-to-end verification through the tool entrypoints.

The tests run the exploits against the CURRENT kernel; each test asserts that
the attempt is blocked. If the sandbox regresses, these tests FAIL loudly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# The lollms_client package is installed under the plain `lollms_client`
# namespace. Importing via `src.lollms_client` collides with unrelated packages
# that install a top-level `src` module on PyPI. Fall back to explicitly
# prepending the repo's `src/` folder to `sys.path` for dev-checkout runs.
try:
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
    from lollms_client.tools_bindings.lcp.default_tools.execute_python.sandbox_kernel import (
        validate_source,
        SandboxContext,
        SandboxPermissionError,
        TRUE_COMPILE as _TRUE_COMPILE,
        TRUE_EXEC as _TRUE_EXEC,
    )
except ImportError:  # pragma: no cover - dev checkout fallback
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
    from lollms_client.tools_bindings.lcp.default_tools.execute_python.sandbox_kernel import (
        validate_source,
        SandboxContext,
        SandboxPermissionError,
        TRUE_COMPILE as _TRUE_COMPILE,
        TRUE_EXEC as _TRUE_EXEC,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layer A — Static AST validation
# ──────────────────────────────────────────────────────────────────────────────

class TestStaticLayer:
    def test_data_query_blocks_subprocess_import(self):
        v = validate_source("import subprocess\nsubprocess.run(['whoami'])", "data_query")
        assert v is not None
        assert "subprocess" in v.reason.lower()

    def test_data_query_blocks_os_system(self):
        v = validate_source("import os\nos.system('dir')", "data_query")
        assert v is not None
        assert "os.system" in v.reason.lower() or "os" in v.reason.lower()

    def test_data_query_blocks_network_libs(self):
        for lib in ("requests", "urllib", "socket", "httpx", "aiohttp"):
            v = validate_source(f"import {lib}", "data_query")
            assert v is not None, f"{lib} was not blocked in data_query"

    def test_data_query_blocks_eval_and_exec(self):
        v = validate_source("eval(\"1+1\")", "data_query")
        assert v is not None
        v = validate_source("exec('print(1)')", "data_query")
        assert v is not None

    def test_data_query_blocks_getattr_chain_to_os(self):
        # from os import system must be blocked
        v = validate_source("from os import system", "data_query")
        assert v is not None

    def test_data_query_allows_data_analysis(self):
        good = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "df = pd.DataFrame({'a':[1,2,3]})\n"
            "print(df['a'].sum())\n"
        )
        assert validate_source(good, "data_query") is None

    def test_standard_blocks_ctypes(self):
        v = validate_source("import ctypes", "standard")
        assert v is not None
        assert "ctypes" in v.reason.lower()

    def test_standard_blocks_socket(self):
        v = validate_source("import socket\nsocket.socket()", "standard")
        assert v is not None

    def test_standard_blocks_subprocess(self):
        v = validate_source("import subprocess\nsubprocess.run(['ls'])", "standard")
        assert v is not None

    def test_standard_allows_benign(self):
        good = (
            "import pandas as pd\n"
            "df = pd.DataFrame({'x':[1,2,3]})\n"
            "print(df)\n"
        )
        assert validate_source(good, "standard") is None


# ──────────────────────────────────────────────────────────────────────────────
# Layer B — Runtime interception
# ──────────────────────────────────────────────────────────────────────────────

class TestRuntimeLayer:
    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td).resolve()

    def _exec_as_user(self, source: str, label: str = "<execute_python>"):
        """Execute `source` with a user-code frame so guard logic engages."""
        return _TRUE_EXEC(
            _TRUE_COMPILE(source, label, "exec"),
            {"__builtins__": __builtins__},
        )

    def test_open_outside_root_blocked(self, workspace, tmp_path):
        """Direct open() of a path outside the workspace must be blocked."""
        escape_file = tmp_path / "escape.txt"
        escape_file.write_text("secret")
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user(f"open(r'{escape_file}').read()")

    def test_open_inside_root_allowed(self, workspace):
        target = workspace / "inside.txt"
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with open(str(target), "w") as f:
                f.write("ok")
        assert target.exists()

    def test_os_system_blocked(self, workspace):
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                os.system("echo hacked")

    def test_os_remove_outside_root_blocked(self, workspace, tmp_path):
        victim = tmp_path / "tobedeleted.txt"
        victim.write_text("x")
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user(
                    f"import os\nos.remove(r'{victim}')"
                )
        assert victim.exists()

    def test_path_write_outside_root_blocked(self, workspace, tmp_path):
        target = tmp_path / "escape.txt"
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user(
                    f"from pathlib import Path\nPath(r'{target}').write_text('nope')"
                )
        assert not target.exists()

    def test_socket_connect_blocked(self, workspace):
        import socket
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect(("127.0.0.1", 65535))
                finally:
                    s.close()

    def test_eval_blocked(self, workspace):
        """`eval` is blocked for user-code frames — same gate as exec/compile.

        The absolute `eval` is not the vector to guard: from the test frame
        (filename `test_sandbox_python_kernel.py`) the frame gate correctly
        declines to raise, because `eval` from a host frame is a legitimate
        operation (CPython's `collections.namedtuple` uses it during normal
        library imports). The exploit vector is `eval` reached from a
        user-code frame — i.e. from `<execute_python*>` — and that is what
        this test asserts.
        """
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user("eval('1+1')")

    def test_import_subprocess_blocked(self, workspace):
        """User-frame `import subprocess` must be blocked at runtime."""
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user("__import__('subprocess')")

    def test_importlib_import_module_blocked(self, workspace):
        with SandboxContext(workspace_root=workspace, profile="data_query"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                # From a user frame, importlib.import_module('subprocess') must fail.
                self._exec_as_user(
                    "import importlib\nimportlib.import_module('subprocess')",
                    label="<execute_python_data_query>",
                )

    def test_data_query_blocks_non_allowlisted_import(self, workspace):
        with SandboxContext(workspace_root=workspace, profile="data_query"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user(
                    "__import__('ctypes')", label="<execute_python_data_query>"
                )
            with pytest.raises((SandboxPermissionError, PermissionError)):
                self._exec_as_user(
                    "__import__('socket')", label="<execute_python_data_query>"
                )

    def test_data_query_allows_pandas(self, workspace):
        # pandas may or may not be installed; guard accordingly
        pytest.importorskip("pandas")
        with SandboxContext(workspace_root=workspace, profile="data_query"):
            import pandas  # noqa: F401

    def test_data_query_allows_library_internal_import(self, workspace):
        """Library-internal imports (not user frame) must pass through.

        `import pandas` triggers pandas' own `import importlib` / `tempfile`
        etc. Those are FORBIDDEN_MODULES for user code but perfectly safe for
        a vetted library's load. The frame-gated filter must let them through.
        """
        pytest.importorskip("pandas")
        # Simulate a library-frame import: call via __import__ from THIS test
        # frame (not a <execute_python_data_query> frame), which is what a
        # library's own __init__ looks like to the filter.
        with SandboxContext(workspace_root=workspace, profile="data_query"):
            import importlib  # noqa: F401
            import tempfile   # noqa: F401
            import copy       # noqa: F401

    def test_restores_pathlib_methods_after_exit(self, workspace):
        """Regression: the __pathlib__ snapshot MUST be restored on exit.

        If Path.mkdir/write_text/etc. leak past SandboxContext.__exit__, every
        later call in the host (and every subsequent test) gets path-checked
        against a stale workspace_root — the exact cascade that broke 190 tests.
        """
        from pathlib import Path as P
        originals = {
            name: getattr(P, name)
            for name in ("mkdir", "write_text", "open", "unlink", "rename", "replace")
        }
        with SandboxContext(workspace_root=workspace, profile="standard"):
            assert P.mkdir is not originals["mkdir"]
            assert P.write_text is not originals["write_text"]
        for name, orig in originals.items():
            assert getattr(P, name) is orig, f"Path.{name} was not restored"

    def test_os_chdir_escape_blocked(self, workspace, tmp_path):
        """chdir to a path outside the workspace root must be rejected."""
        original_cwd = os.getcwd()
        try:
            with SandboxContext(workspace_root=workspace, profile="standard"):
                with pytest.raises((SandboxPermissionError, PermissionError)):
                    os.chdir(str(tmp_path))
        finally:
            os.chdir(original_cwd)

    def test_os_chdir_inside_root_allowed(self, workspace):
        """chdir to a subdirectory of the workspace is legitimate."""
        original_cwd = os.getcwd()
        try:
            sub = workspace / "sub"
            sub.mkdir()
            with SandboxContext(workspace_root=workspace, profile="standard"):
                os.chdir(str(sub))
                assert Path.cwd().resolve() == sub.resolve()
        finally:
            os.chdir(original_cwd)

    def test_library_cache_writes_land_inside_workspace(self, workspace):
        """Legitimate library cache writes MUST succeed — via cache redirect.

        Every tempfile call (mkstemp, NamedTemporaryFile, TemporaryDirectory)
        and every XDG/matplotlib/pip cache write inside the sandbox is
        automatically redirected into `<workspace>/.sandbox_cache/`, so it
        lands inside the perimeter and is allowed by the frame-free guard.
        This is what makes matplotlib/pandas/sklearn usable in the data-query
        tool while keeping the escqape boundary strict for everything else.
        """
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                tmp.write("library cache")
                tmp_path = tmp.name
            # Redirected: file must live under .sandbox_cache, i.e. inside root.
            assert Path(tmp_path).resolve().is_relative_to(workspace.resolve())
            assert Path(tmp_path).read_text() == "library cache"
            Path(tmp_path).unlink()
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "libdir").mkdir()
                assert (Path(td) / "libdir").is_dir()
                # Same invariant for TemporaryDirectory.
                assert Path(td).resolve().is_relative_to(workspace.resolve())

    def test_pandas_read_outside_workspace_blocked(self, workspace, tmp_path):
        """pandas-mediated path escapes MUST be blocked (frame-independent).

        Regression for the frame-window hole: `pd.read_csv('../etc/passwd')`
        reaches `open()` through pandas.io.parsers → pandas.io.common →
        builtins.open — a call stack deeper than the old 3-frame heuristic
        scanned, so the escape succeeded. The new model blocks by path
        argument alone, regardless of who is deep in the stack.
        """
        pytest.importorskip("pandas")
        victim = tmp_path / "secret.csv"
        victim.write_text("classified\n")
        exploit_src = (
            "import pandas as pd\n"
            f"df = pd.read_csv(r'{victim}')\n"
            "print(df)\n"
        )
        with SandboxContext(workspace_root=workspace, profile="data_query"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                _TRUE_EXEC(
                    _TRUE_COMPILE(exploit_src, "<execute_python_data_query>", "exec"),
                    {"__builtins__": __builtins__},
                )

    def test_sandbox_cache_env_created_once_per_workspace(self, workspace):
        """Regression: cache dirs must be created at most once per workspace.

        The 600s regression was caused by _redirect_cache_env() running 13
        mkdir syscalls on every tool call. Once-per-workspace memoization
        eliminates it.
        """
        import time
        t0 = time.perf_counter()
        for _ in range(30):
            with SandboxContext(workspace_root=workspace, profile="standard"):
                pass
        elapsed = time.perf_counter() - t0
        # 30 enter/exit round-trips must complete well under 1s on any modern
        # machine. Regression = ~13 mkdir syscalls × 30 = 390 syscalls plus
        # guard resolves, which historically pushed this past 3-5s.
        assert elapsed < 1.0, (
            f"30 SandboxContext enter/exit cycles took {elapsed:.2f}s — "
            "cache-env redirection is likely being re-run per call."
        )
        # Sanity: exactly one .sandbox_cache directory exists.
        assert (workspace / ".sandbox_cache").is_dir()

    def test_relative_path_guard_is_fast(self, workspace):
        """Regression: relative Path.mkdir must not trigger resolve() syscalls.

        Fast-path check is O(1); slow-path resolve() is O(syscalls). 200 mkdir
        calls in the sandbox should complete in milliseconds, not seconds.
        """
        import time
        with SandboxContext(workspace_root=workspace, profile="standard"):
            t0 = time.perf_counter()
            for i in range(200):
                (workspace / f"dir_{i}").mkdir()
            elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, (
            f"200 relative Path.mkdir calls took {elapsed:.2f}s — the guard is "
            "resolving relative paths instead of using the fast path."
        )

    def test_absolute_path_guard_still_enforced(self, workspace, tmp_path):
        """Security invariant: absolute path outside root must be blocked."""
        target = tmp_path / "attack"
        with SandboxContext(workspace_root=workspace, profile="standard"):
            with pytest.raises((SandboxPermissionError, PermissionError)):
                _TRUE_EXEC(
                    _TRUE_COMPILE(
                        f"from pathlib import Path\nPath(r'{target}').mkdir()\n",
                        "<execute_python>",
                        "exec",
                    ),
                    {"__builtins__": __builtins__},
                )
        assert not target.exists()

    def test_restores_builtins_after_exit(self, workspace):
        original_open = open
        original_eval = eval
        original_import = __import__
        with SandboxContext(workspace_root=workspace, profile="standard"):
            assert open is not original_open
            assert eval is not original_eval
        assert open is original_open
        assert eval is original_eval
        assert __import__ is original_import


# ──────────────────────────────────────────────────────────────────────────────
# Layer C — End-to-end tool entrypoints
# ──────────────────────────────────────────────────────────────────────────────

class TestExecutePythonDataQueryEndpoint:
    def _accept_all(self, *_a, **_kw):
        return "allow", ""

    def test_blocks_os_system_before_running(self, monkeypatch):
        # Static layer must reject BEFORE the operator prompt is even consulted.
        result = epdq.tool_execute_python_data_query(
            code="import os\nos.system('whoami')",
            env=None,
        )
        assert result["success"] is False
        assert "sandbox" in result["error"].lower() or "blocked" in result["error"].lower()

    def test_blocks_subprocess_import(self):
        result = epdq.tool_execute_python_data_query(
            code="import subprocess\nsubprocess.run(['echo', 'pwn'])",
            env=None,
        )
        assert result["success"] is False
        assert "subprocess" in result["error"].lower()

    def test_blocks_eval(self):
        result = epdq.tool_execute_python_data_query(
            code="print(eval('1+1'))",
            env=None,
        )
        assert result["success"] is False

    def test_allows_pandas_query_with_handler(self, monkeypatch):
        pytest.importorskip("pandas")
        # Register a handler AND accept; this is the "operator-enabled" path.
        monkeypatch.setattr(
            epdq,
            "_has_registered_confirm_handler",
            lambda env=None: True,
        )
        monkeypatch.setattr(
            epdq,
            "_confirm_data_query_execution",
            lambda src, env=None: ("allow", ""),
        )
        result = epdq.tool_execute_python_data_query(
            code="import pandas as pd\nprint(pd.DataFrame({'a':[1,2]}).sum().to_dict())",
            env=None,
        )
        assert result["success"] is True, result.get("error")

    def test_operator_reject_blocks_execution(self, monkeypatch):
        # When an operator handler is registered AND rejects, execution is blocked.
        monkeypatch.setattr(
            epdq,
            "_has_registered_confirm_handler",
            lambda env=None: True,
        )
        monkeypatch.setattr(
            epdq,
            "_confirm_data_query_execution",
            lambda src, env=None: ("reject", "test rejection"),
        )
        result = epdq.tool_execute_python_data_query(
            code="print('should not run')",
            env=None,
        )
        assert result["success"] is False
        assert "operator" in result["error"].lower() or "reject" in result["error"].lower()

    def test_headless_runs_without_operator_handler(self):
        """Without a registered handler, the sandbox itself is the control.

        Headless execution of safe code MUST succeed — this is what makes the
        tool usable in CI, servers, and notebooks. The static+runtime sandbox
        still rejects anything unsafe.
        """
        pytest.importorskip("pandas")
        result = epdq.tool_execute_python_data_query(
            code="import pandas as pd\nprint(int(pd.DataFrame({'a':[1,2,3]}).sum().iloc[0]))",
            env=None,
        )
        assert result["success"] is True, result.get("error")
        assert "6" in result["output"]

    def test_still_blocks_exploits_in_headless_mode(self):
        """Sandbox safety is NOT contingent on the operator prompt."""
        result = epdq.tool_execute_python_data_query(
            code="import subprocess\nsubprocess.run(['whoami'])",
            env=None,
        )
        assert result["success"] is False
        assert "subprocess" in result["error"].lower()


class TestExecutePythonEndpoint:
    def test_blocks_ctypes_import(self):
        result = ep.tool_execute_python_code(code="import ctypes")
        assert result["success"] is False
        assert "ctypes" in result["error"].lower()

    def test_blocks_subprocess_import(self):
        result = ep.tool_execute_python_code(code="import subprocess")
        assert result["success"] is False

    def test_blocks_socket_import(self):
        result = ep.tool_execute_python_code(code="import socket")
        assert result["success"] is False

    def test_allows_benign_code(self):
        result = ep.tool_execute_python_code(code="print(2+2)")
        assert result["success"] is True, result.get("error")
        assert "4" in result["output"]

    def test_no_path_mkdir_leak_after_rejected_call(self):
        """Regression: a rejected/errored tool call must restore Path.mkdir.

        If a SandboxContext leaks on any exit path, the next host-side
        Path.mkdir() in the same process is path-checked against a stale
        workspace boundary — the exact cascade that broke 190+ tests.
        """
        from pathlib import Path as _P
        original_mkdir = _P.mkdir
        ep.tool_execute_python_code(code="import subprocess")  # static rejection
        ep.tool_execute_python_code(code="open('../../../etc/passwd')")  # runtime block
        assert _P.mkdir is original_mkdir, "Path.mkdir leaked from a rejected tool call"
        # Sanity: host-side mkdir still works
        with tempfile.TemporaryDirectory() as td:
            (_P(td) / "test").mkdir()

    def test_blocks_eval(self):
        # `eval` is unconditionally rejected while the sandbox is active.
        # The static AST layer rejects the source before the child even starts,
        # and the runtime layer's unconditional stub covers dynamic paths.
        result = ep.tool_execute_python_code(code="print(eval('1+1'))")
        assert result["success"] is False
        assert "eval" in result["error"].lower() or "sandbox" in result["error"].lower()

    def test_user_frame_ops_are_sandboxed(self, tmp_path):
        """User exec code attempting Path.mkdir outside root MUST be blocked.

        Complements test_library_cache_writes_not_blocked: host frames pass
        through, but the `<execute_python>` frame is enforced.
        """
        victim_parent = tmp_path / "outside"
        user_src = (
            "from pathlib import Path\n"
            f"Path(r'{victim_parent}').mkdir()\n"
        )
        sandbox = SandboxContext(workspace_root=tmp_path / "inside", profile="standard")
        (tmp_path / "inside").mkdir()
        with sandbox:
            # Use TRUE_COMPILE / TRUE_EXEC so the sandbox's patched builtins
            # (which would raise earlier for a different reason) don't mask
            # the path-guard behaviour we are actually trying to test.
            compiled = _TRUE_COMPILE(user_src, "<execute_python>", "exec")
            with pytest.raises((SandboxPermissionError, PermissionError)):
                _TRUE_EXEC(compiled, {"__builtins__": __builtins__})
        assert not victim_parent.exists()

    def test_blocks_file_escape(self, tmp_path):
        target = tmp_path / "victim.txt"
        target.write_text("secret")
        # Attempt to read a path outside the workspace via open()
        result = ep.tool_execute_python_code(
            code=f"with open(r'{target}') as f: print(f.read())"
        )
        assert result["success"] is False, (
            f"Sandbox must block file escape. Got success=True, "
            f"output={result.get('output')!r}, stderr={result.get('stderr')!r}"
        )
        assert "outside" in result["error"].lower() or "blocked" in result["error"].lower()

    def test_blocks_network_socket_call(self):
        # import already blocked; this passes the AST but the runtime layer must
        # also refuse to run subprocess/os.system calls that slip through.
        result = ep.tool_execute_python_code(code="import os\nos.system('echo hacked')")
        assert result["success"] is False