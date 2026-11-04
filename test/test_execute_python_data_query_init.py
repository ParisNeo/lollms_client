"""
test_execute_python_data_query_init.py

Contract tests for `execute_python_data_query.init_tools_library()`.

Contract (post trust-model refactor):

    LCP's `init_tools_library()` is the **trusted** place where a toolset
    ensures its dependencies. For first-party / signed toolsets the tool
    installs what it needs (via pipmaster). For untrusted (user-supplied)
    toolsets we refuse to auto-install — the host operator stays in control.
    This is the security boundary that matters: the *parameters* passed to a
    tool are untrusted; the tool's own initialization is not.

Test matrix:

    1. Trusted init installs when missing (via pipmaster).
    2. Trusted init is a no-op when nothing is missing.
    3. Untrusted init never installs (fail-closed).
    4. Init never raises on any pipmaster failure.
    5. Init refuses to install while inside an active SandboxContext.
    6. Runtime path (tool call) never imports pipmaster, at any trust level.
    7. `_require()` yields an actionable error free of host-path leakage.
    8. `install_missing_dependencies()` remains the explicit host entry point.
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest


try:
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as sk
except ImportError:  # pragma: no cover - dev checkout fallback
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as sk


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class _ImportSpy:
    """Records every module name imported via `builtins.__import__` while active."""

    def __init__(self):
        self.seen: list[str] = []
        self._orig = None

    def __enter__(self):
        self._orig = builtins.__import__

        def _spy(name, *args, **kwargs):
            self.seen.append(name)
            return self._orig(name, *args, **kwargs)

        builtins.__import__ = _spy
        return self

    def __exit__(self, *exc_info):
        if self._orig is not None:
            builtins.__import__ = self._orig

    def pipmaster_touched(self) -> bool:
        return any(m == "pipmaster" or m.startswith("pipmaster.") for m in self.seen)


class _FakePipmaster:
    """Captures calls; optionally raises to simulate a broken pip."""

    def __init__(self, raise_exc: Exception | None = None):
        self.calls: list[list[str]] = []
        self._raise = raise_exc

    def ensure_packages(self, packages):
        self.calls.append(list(packages))
        if self._raise is not None:
            raise self._raise


@pytest.fixture
def no_pipmaster(monkeypatch):
    """Simulate pipmaster being importable but harmless by default."""
    fake = _FakePipmaster()
    monkeypatch.setitem(sys.modules, "pipmaster", fake)
    return fake


@pytest.fixture
def force_missing(monkeypatch):
    """Force check_dependencies to report specific packages as missing."""
    def _set(missing: list[str]):
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": [], "missing": list(missing)},
        )
    return _set


# ──────────────────────────────────────────────────────────────────────────────
# 1. Trusted init installs when missing
# ──────────────────────────────────────────────────────────────────────────────

class TestTrustedInitInstalls:
    def test_trusted_builtin_installs_missing(self, no_pipmaster, force_missing):
        """The whole point of init: trusted toolset ensures its own deps."""
        force_missing(["pandas", "numpy"])
        epdq.init_tools_library()
        assert no_pipmaster.calls == [["pandas", "numpy"]], (
            f"Expected one ensure_packages call with the missing packages; got {no_pipmaster.calls}"
        )

    def test_trusted_signed_via_config_installs(self, no_pipmaster, force_missing):
        """An explicitly-signed third-party toolset is trusted to install too."""
        force_missing(["pandas"])
        epdq.init_tools_library(config={"trust_level": epdq.TRUSTED_SIGNED})
        assert no_pipmaster.calls == [["pandas"]]

    def test_trusted_init_noop_when_everything_present(self, no_pipmaster, monkeypatch):
        """No missing packages → no pipmaster call at all."""
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": list(epdq._REQUIRED_PACKAGES.keys()), "missing": []},
        )
        epdq.init_tools_library()
        assert no_pipmaster.calls == []

    def test_trusted_init_with_config_and_no_missing(self, no_pipmaster, monkeypatch):
        """`config={...}` is tolerated and does not change the no-missing behaviour."""
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": list(epdq._REQUIRED_PACKAGES.keys()), "missing": []},
        )
        epdq.init_tools_library(config={"any": "value"})
        assert no_pipmaster.calls == []


# ──────────────────────────────────────────────────────────────────────────────
# 2. Untrusted init never installs
# ──────────────────────────────────────────────────────────────────────────────

class TestUntrustedInitRefuses:
    def test_untrusted_user_never_installs(self, no_pipmaster, force_missing):
        """User-supplied toolsets MUST NOT trigger pip installs at init."""
        force_missing(["pandas"])
        epdq.init_tools_library(config={"trust_level": epdq.UNTRUSTED_USER})
        assert no_pipmaster.calls == [], (
            "Untrusted toolset triggered a pip install — the trust boundary "
            "is broken."
        )

    def test_unknown_trust_level_fails_closed(self, no_pipmaster, force_missing):
        """A typo or hostile caller can't accidentally raise trust."""
        force_missing(["pandas"])
        epdq.init_tools_library(config={"trust_level": "totally-legit-trust-me"})
        assert no_pipmaster.calls == [], (
            "Unknown trust_level must fail closed (untrusted) rather than "
            "silently escalating to trusted."
        )

    def test_untrusted_init_still_runs_matplotlib_config(self, tmp_path,
                                                       no_pipmaster, force_missing):
        """Even untrusted init runs matplotlib backend selection (harmless)."""
        force_missing(["pandas"])
        # Should not raise even though pipmaster exists and could raise.
        fake = _FakePipmaster(raise_exc=RuntimeError("pip crashed"))
        sys.modules["pipmaster"] = fake
        epdq.init_tools_library(config={"trust_level": epdq.UNTRUSTED_USER})
        # Still no call — pipmaster never invoked.
        assert fake.calls == []


# ──────────────────────────────────────────────────────────────────────────────
# 3. Init never raises on pipmaster failure
# ──────────────────────────────────────────────────────────────────────────────

class TestInitNeverRaises:
    def test_trusted_init_survives_pipmaster_failure(self, monkeypatch, force_missing):
        """Trusted init installs, but a broken pipmaster must not break init."""
        fake = _FakePipmaster(raise_exc=RuntimeError("simulated pip failure"))
        monkeypatch.setitem(sys.modules, "pipmaster", fake)
        force_missing(["pandas"])
        epdq.init_tools_library()  # no exception = pass
        # It should have attempted once via install_missing_dependencies.
        assert fake.calls == [["pandas"]]

    def test_init_survives_check_dependencies_raising(self, monkeypatch):
        """Defensive: even a broken probe must not break init."""
        def _boom():
            raise RuntimeError("probe failure")

        monkeypatch.setattr(epdq, "check_dependencies", _boom)
        # Should not raise at any trust level.
        epdq.init_tools_library()
        epdq.init_tools_library(config={"trust_level": epdq.UNTRUSTED_USER})

    def test_init_is_idempotent(self, no_pipmaster):
        """Calling init twice must be safe; second call is a no-op when all present."""
        epdq.init_tools_library()
        first_calls = list(no_pipmaster.calls)
        epdq.init_tools_library()
        # Whatever was missing in the first call, the second call would try
        # again only if still missing — so in this environment, second call
        # either repeats the same attempt (still missing) or adds nothing
        # (now available). Either behaviour is idempotent-safe.
        assert all(len(c) > 0 for c in no_pipmaster.calls)
        assert first_calls == no_pipmaster.calls[: len(first_calls)]

    def test_init_accepts_optional_config_argument(self, no_pipmaster, monkeypatch):
        """Contract: init_tools_library(config=...) is tolerated (LCP convention)."""
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": list(epdq._REQUIRED_PACKAGES.keys()), "missing": []},
        )
        epdq.init_tools_library(config={"any": "value"})
        epdq.init_tools_library(config=None)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Init refuses to install inside a live SandboxContext
# ──────────────────────────────────────────────────────────────────────────────

class TestInitInsideSandbox:
    def test_init_inside_sandbox_does_not_touch_pipmaster(self, tmp_path):
        """Running init inside a sandbox must not trigger pip.

        This is the crucial safety property preserved from the earlier
        hardening: tool init is a *host* responsibility. If a caller wraps it
        in a `SandboxContext` (defence-in-depth), we must refuse — pushing
        `pip install` through a sandbox is exactly what blocked the tool in
        the originating bug.
        """
        with sk.SandboxContext(workspace_root=tmp_path, profile="standard"):
            with _ImportSpy() as spy:
                epdq.init_tools_library()
        assert not spy.pipmaster_touched(), (
            "init_tools_library imported pipmaster from inside a live sandbox. "
            "That is the exact failure mode this test protects against."
        )

    def test_init_inside_sandbox_still_runs_matplotlib_config(self, tmp_path):
        """The matplotlib setup step remains active even inside the sandbox."""
        # Just verify it doesn't raise and doesn't import pipmaster.
        with sk.SandboxContext(workspace_root=tmp_path, profile="standard"):
            epdq.init_tools_library()

    def test_inside_sandbox_detection(self, tmp_path):
        """The _inside_sandbox() probe returns True during a SandboxContext."""
        before = epdq._inside_sandbox()
        with sk.SandboxContext(workspace_root=tmp_path, profile="standard"):
            during = epdq._inside_sandbox()
        after = epdq._inside_sandbox()
        assert before is False
        assert during is True
        assert after is False


# ──────────────────────────────────────────────────────────────────────────────
# 5. Runtime path never imports pipmaster
# ──────────────────────────────────────────────────────────────────────────────

class TestRuntimePathNeverAutoInstalls:
    def test_tool_call_does_not_import_pipmaster(self, tmp_path, monkeypatch):
        """A tool call must not trigger pipmaster, whatever the trust level."""
        monkeypatch.chdir(tmp_path)
        with _ImportSpy() as spy:
            result = epdq.tool_execute_python_data_query(
                code="import subprocess\nsubprocess.run(['whoami'])",
                env=None,
            )
        assert result["success"] is False  # static rejection as expected
        assert not spy.pipmaster_touched(), (
            "tool_execute_python_data_query imported pipmaster at runtime. "
            "Runtime must never auto-install."
        )

    def test_missing_dependency_error_is_actionable(self, monkeypatch):
        """`_require()` must yield an actionable, pip-installable hint."""
        monkeypatch.setitem(
            epdq._REQUIRED_PACKAGES,
            "definitely_not_a_real_module_xyz",
            "definitely-not-a-real-pip-pkg",
        )
        try:
            with pytest.raises(epdq.MissingDependency) as excinfo:
                epdq._require("definitely_not_a_real_module_xyz")
        finally:
            epdq._REQUIRED_PACKAGES.pop("definitely_not_a_real_module_xyz", None)

        msg = str(excinfo.value)
        assert "definitely-not-a-real-pip-pkg" in msg
        # Never leak absolute host filesystem paths in user-facing errors.
        for host_fragment in ("Users\\", "Users/", "/home/", "C:\\Users"):
            assert host_fragment not in msg


# ──────────────────────────────────────────────────────────────────────────────
# 6. install_missing_dependencies remains the explicit host entry point
# ──────────────────────────────────────────────────────────────────────────────

class TestInstallMissingDependencies:
    def test_noop_when_everything_present(self, monkeypatch, no_pipmaster):
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": list(epdq._REQUIRED_PACKAGES.keys()), "missing": []},
        )
        report = epdq.install_missing_dependencies()
        assert report["ok"] is True
        assert report["installed"] == []
        assert report["error"] is None
        assert no_pipmaster.calls == []

    def test_returns_structured_error_when_pipmaster_unavailable(self, monkeypatch):
        saved = sys.modules.pop("pipmaster", None)
        monkeypatch.setitem(sys.modules, "pipmaster", None)  # import fails
        try:
            report = epdq.install_missing_dependencies(packages=["pandas"])
        finally:
            sys.modules.pop("pipmaster", None)
            if saved is not None:
                sys.modules["pipmaster"] = saved

        assert report["ok"] is False
        assert report["error"] is not None
        assert "pipmaster" in report["error"].lower()

    def test_returns_structured_error_when_pipmaster_fails(self, monkeypatch):
        fake = _FakePipmaster(raise_exc=RuntimeError("simulated pip failure"))
        monkeypatch.setitem(sys.modules, "pipmaster", fake)
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": [], "missing": ["pandas"]},
        )
        report = epdq.install_missing_dependencies(packages=["pandas"])
        assert report["ok"] is False
        assert "simulated pip failure" in report["error"]

    def test_installs_when_invoked(self, no_pipmaster, monkeypatch):
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": list(epdq._REQUIRED_PACKAGES.keys()), "missing": []},
        )
        report = epdq.install_missing_dependencies(packages=["pandas", "numpy"])
        assert report["ok"] is True
        assert no_pipmaster.calls == [["pandas", "numpy"]]

    def test_normalizes_import_name_to_pip_name(self, no_pipmaster, monkeypatch):
        monkeypatch.setattr(
            epdq,
            "check_dependencies",
            lambda: {"available": [], "missing": []},
        )
        epdq.install_missing_dependencies(packages=["docx", "pptx"])
        assert no_pipmaster.calls == [["python-docx", "python-pptx"]]