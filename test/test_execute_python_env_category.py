"""Regression tests for the `env` risk category in `execute_python`.

Reported defect
---------------
`configure_ngspice47()` writes ``os.environ["SPICE_SCRIPTS"] = ...`` and was
rejected by the sandbox with a bare ``BLOCKED BY SANDBOX`` and **no prompt**:
the operator had no way to say "yes, allow this". The root cause was twofold:

* `execute_python` did not classify `os.environ[...] = ...` as a risk, so
  `safe` mode never prompted, and `"always"` would not have persisted the
  consent even if the operator had been asked.
* `sandbox_kernel._gated_putenv` / `_gated_unsetenv` unconditionally refused
  user-code environment mutations, and the freshly-spawned child could not
  observe the parent's approval state anyway.

These tests pin the fix at three layers:

1. **Detection** (`_detect_env_mutation_ops`) — every mutation form is named.
2. **Prompting** (`_is_category_approved`) — env forces a prompt until the
   operator consents via `_approve_categories({"env"})`.
3. **Runtime honouring** (`run_in_sandboxed_child`) — with
   `approved_categories={"env"}` transported to the child, the mutation
   succeeds; without it, the block remains (the security property).
"""

import sys
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Import wiring — mirror how the rest of the suite reaches the tools
# ──────────────────────────────────────────────────────────────────────────────
def _import_modules():
    try:
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import (
            TOOLS as ep,
        )
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import (
            sandbox_kernel as sk,
        )
        return ep, sk
    except ImportError:
        # Dev-checkout fallback: `src/` on sys.path.
        repo_src = Path(__file__).resolve().parents[1] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import (  # type: ignore
            TOOLS as ep,
        )
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import (  # type: ignore
            sandbox_kernel as sk,
        )
        return ep, sk


execute_python, sandbox_kernel = _import_modules()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def isolated_approval_state():
    """Resets the process-wide approval set before AND after each test.

    The approval registry is mirrored on `sys`; state must not leak between
    tests (a stray `{"env"}` would make the "prompts when unapproved"
    assertions vacuous).
    """
    execute_python._reset_approved_categories()
    try:
        yield
    finally:
        execute_python._reset_approved_categories()


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """A scratch workspace the child can safely mutate."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1 — detection
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvMutationDetection:
    """`_detect_env_mutation_ops` must name every mutation form."""

    def test_subscript_assignment(self):
        ops = execute_python._detect_env_mutation_ops('os.environ["K"] = "V"')
        assert ops == ["os.environ[...] ="]

    def test_augmented_subscript_assignment(self):
        ops = execute_python._detect_env_mutation_ops('os.environ["K"] += "V"')
        assert ops == ["os.environ[...] ="]

    def test_delete_subscript(self):
        ops = execute_python._detect_env_mutation_ops('del os.environ["K"]')
        assert ops == ["del os.environ[...]"]

    def test_putenv_and_unsetenv(self):
        source = 'os.putenv("A", "1")\nos.unsetenv("B")'
        ops = execute_python._detect_env_mutation_ops(source)
        assert ops == ["os.putenv()", "os.unsetenv()"]

    @pytest.mark.parametrize("method", ["update", "setdefault", "pop", "popitem", "clear"])
    def test_environ_methods(self, method):
        ops = execute_python._detect_env_mutation_ops(f"os.environ.{method}(...)")
        assert ops == [f"os.environ.{method}()"]

    def test_from_import_style(self):
        ops = execute_python._detect_env_mutation_ops('environ["K"] = "V"')
        assert ops == ["environ[...] ="]

    def test_readonly_access_is_not_flagged(self):
        """Reading `os.environ[...]` / `os.getenv(...)` must NOT prompt."""
        assert execute_python._detect_env_mutation_ops('x = os.environ["K"]') == []
        assert execute_python._detect_env_mutation_ops('x = os.getenv("K")') == []
        assert execute_python._detect_env_mutation_ops("y = dict(os.environ)") == []

    def test_syntax_error_returns_empty(self):
        assert execute_python._detect_env_mutation_ops("def broken(:") == []


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2 — prompting decision
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvCategoryPrompting:
    """Env mutations must force a prompt until the operator consents."""

    def test_env_is_a_known_category(self):
        assert "env" in execute_python._RISK_CATEGORY_SIGNALS

    def test_unapproved_env_is_not_auto_approved(self, isolated_approval_state):
        # No risk reason — the ONLY unusual surface is the env mutation.
        approved = execute_python._is_category_approved(
            None, source='os.environ["K"] = "V"',
        )
        assert approved is False, (
            "An env mutation with no prior consent MUST trigger the prompt."
        )

    def test_approved_env_is_auto_approved(self, isolated_approval_state):
        execute_python._approve_categories(frozenset({"env"}))
        approved = execute_python._is_category_approved(
            None, source='os.environ["K"] = "V"',
        )
        assert approved is True

    def test_categories_for_includes_env(self):
        cats = execute_python._categories_for('os.putenv("A", "1")')
        assert "env" in cats

    def test_env_survives_approve_categories_persistence(self, isolated_approval_state):
        """`_approve_categories` persists every category except `delete`."""
        execute_python._approve_categories(frozenset({"env"}))
        assert "env" in execute_python._get_approved_categories()

    def test_delete_and_env_are_independent(self, isolated_approval_state):
        """Approving `env` must NOT license deletions, and vice-versa."""
        execute_python._approve_categories(frozenset({"env"}))
        delete_approved = execute_python._is_category_approved(
            None, source="import os\nos.remove('victim.txt')",
        )
        assert delete_approved is False


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3 — runtime honouring (end-to-end through the real child process)
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvRuntimeGate:
    """The spawned child must honour the transported consent set."""

    def _run(self, code: str, workspace_root: Path, approved):
        return sandbox_kernel.run_in_sandboxed_child(
            source=code,
            workspace_root=workspace_root,
            profile="standard",
            script_label="<env_test>",
            hard_timeout=30.0,
            approved_categories=approved,
        )

    def test_env_mutation_blocked_without_approval(self, workspace_root: Path):
        envelope = self._run(
            'import os\nos.environ["LOLLMS_ENV_TEST"] = "1"\nprint("after")',
            workspace_root,
            approved=frozenset(),
        )
        assert envelope["success"] is False
        assert envelope["sandbox_violation"] is True
        # The block message points the operator at the consent path.
        assert "env" in envelope["traceback"].lower()

    def test_env_mutation_succeeds_with_approval(self, workspace_root: Path):
        envelope = self._run(
            'import os\n'
            'os.environ["LOLLMS_ENV_TEST"] = "1"\n'
            'print(os.environ.get("LOLLMS_ENV_TEST"))',
            workspace_root,
            approved=frozenset({"env"}),
        )
        assert envelope["success"] is True, envelope.get("traceback")
        assert "1" in envelope["stdout"]

    def test_del_environ_succeeds_with_approval(self, workspace_root: Path):
        envelope = self._run(
            'import os\n'
            'os.environ["LOLLMS_ENV_TEST"] = "1"\n'
            'del os.environ["LOLLMS_ENV_TEST"]\n'
            'print("gone" if "LOLLMS_ENV_TEST" not in os.environ else "still")',
            workspace_root,
            approved=frozenset({"env"}),
        )
        assert envelope["success"] is True, envelope.get("traceback")
        assert "gone" in envelope["stdout"]


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end through the public tool — the exact reported scenario shape
# ──────────────────────────────────────────────────────────────────────────────
class TestToolEndToEnd:
    """Drive `tool_create_and_execute_python_file` with a pre-approving handler.

    This reproduces the reported `configure_ngspice47()` shape: a file that
    writes `os.environ[...]` and then does useful work. With a confirm handler
    that answers "always", the write+run must succeed and persist the `env`
    consent.
    """

    def test_write_and_run_env_snippet_with_always_consent(
        self, isolated_approval_state, workspace_root: Path, monkeypatch,
    ):
        # Point the tool's workspace resolution at our scratch directory.
        monkeypatch.setattr(
            execute_python, "_get_workspace_root",
            lambda tool_ctx=None: workspace_root,
        )

        seen_prompts = []

        def handler(source, script_label=None, argv=None):
            # `_prompt_user_validation` tries the positional (source, label, argv)
            # signature first (see its TypeError-fallback chain). Accept it.
            seen_prompts.append(script_label or "")
            return "always"

        monkeypatch.setattr(execute_python, "AUTONOMY_LEVEL", "safe")
        # Set the handler directly on the module (monkeypatch reverts it),
        # instead of via `set_confirm_handler` which also mutates sibling
        # module instances and would not be auto-reverted.
        monkeypatch.setattr(execute_python, "_CONFIRM_HANDLER", handler)

        code = (
            "import os\n"
            'os.environ["LOLLMS_SPICE_SCRIPTS"] = "enabled"\n'
            'print(os.environ["LOLLMS_SPICE_SCRIPTS"])\n'
        )
        result = execute_python.tool_create_and_execute_python_file(
            code=code,
            file_name="configure_env_demo.py",
            env=None,
        )

        assert result.get("success") is True, result.get("error")
        assert "enabled" in (result.get("output") or "")
        # Consent was prompted AND then persisted for future env snippets.
        assert seen_prompts, "The env mutation should have triggered a prompt."
        assert "env" in execute_python._get_approved_categories()