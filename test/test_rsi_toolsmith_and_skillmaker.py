"""
RSI Toolsmith + SkillMaker — two-tier split safety net.

These tests pin the invariants that live in ``mission_briefing``:

  * tool_maker:
      - basic tier (read/write/test/list_versions/revert/discard) has NO flag gate
      - RSI tier (run_test_agent/commit_toolset) HARD-gated by enable_toolsmith
      - auto-discovery visibility filter drops ONLY the RSI tools when flag is off
      - .rsi_backup/ + .rsi/commits/ lifecycle actually produces restore targets
      - Aider SEARCH/REPLACE patch is applied in memory; partial patch is refused
      - revert-verify-and-rollback never leaves broken code on disk
      - _FORBIDDEN_TEST_TARGETS refusal fires before the child runner is spawned
      - _HALT_REQUESTED short-circuits every mutating tool

  * skill_maker:
      - identical split with its own enable_skillmaker flag
      - YAML frontmatter validation rejects broken skill files pre-write pre-commit
      - .rsi_backup/ + .rsi/commits/ lifecycle mirrors the toolsmith
      - _FORBIDDEN_TEST_SKILL_TOOLS refusal fires before the child is spawned
      - revert-verify-and-rollback mirrors the toolsmith

  * LCPBinding wiring (via stub):
      - _filter_toolsmith_visibility uses _library_name == 'tool_maker' scope
      - _filter_skillmaker_visibility uses _library_name == 'skill_maker' scope
      - the two filters do not cross-trigger

The suite is entirely deterministic, mock-based, offline, git-free. It does
NOT call any LLM, does NOT spawn a child process, does NOT touch a real
LCPBinding — every dependency is a stub that only exposes the exact attribute
surface the tools read.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# ──────────────────────────────── imports ────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# We import the tool modules via importlib so the test file works regardless
# of whether the package is installed (editable install on the workstation
# is not required — the test only needs the source tree).
import importlib.util


def _load_module(rel_path: str, name: str):
    path = ROOT / rel_path
    assert path.exists(), f"missing fixture file: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tool_maker = _load_module(
    "src/lollms_client/tools_bindings/lcp/default_tools/tool_maker/TOOLS.py",
    "_test_tool_maker",
)
skill_maker = _load_module(
    "src/lollms_client/tools_bindings/lcp/default_tools/skill_maker/TOOLS.py",
    "_test_skill_maker",
)


# ──────────────────────────────── stubs ────────────────────────────────


class StubBinding:
    """Minimal LCPBinding stand-in: only the attributes the tools read."""

    def __init__(self, *, discovered_tools: Optional[List[Dict[str, Any]]] = None,
                 flags: Optional[Dict[str, Any]] = None):
        self.discovered_tools = discovered_tools or []
        self.session_flags = dict(flags or {})
        self.remount_calls: List[str] = []
        self._remount_result = True

    def _remount_toolset_after_change(self, library_name: str) -> bool:
        self.remount_calls.append(library_name)
        return self._remount_result

    # The RSI visibility filters live on the LCPBinding class itself,
    # imported as unbound functions in the tests that need them.


class StubClient:
    def __init__(self, binding: StubBinding, *, personality=None):
        self.tools = binding
        self._active_personality = personality
        self.llm_binding_profiles: Dict[str, Any] = {}
        self.llm_model_profiles: Dict[str, Any] = {}
        self._system_dir = "."


class StubEnv:
    def __init__(self, *, client=None, discussion=None):
        self.client = client
        self.discussion = discussion


def _make_toolset(tmp_path: Path, *, filename: str = "TOOLS.py", content: str) -> Path:
    toolset_dir = tmp_path / "my_toolset"
    toolset_dir.mkdir()
    py = toolset_dir / filename
    py.write_text(content, encoding="utf-8")
    return py


def _seed_discovered(binding: StubBinding, py_file: Path, *tool_names: str) -> None:
    for tn in tool_names:
        binding.discovered_tools.append({
            "name": tn,
            "description": f"{tn} test tool",
            "input_schema": {"type": "object", "properties": {}, "required": []},
            "_python_file_path": str(py_file.resolve()),
            "_library_name": py_file.parent.name,
        })


GOOD_TOOLSET_SOURCE = textwrap.dedent(
    """
    TOOL_LIBRARY_NAME = "my_toolset"
    GATED_TASK_CATEGORY = "code_execution"

    def tool_hello(name: str = "world") -> dict:
        return {"success": True, "output": f"hello {name}"}
    """
).lstrip()


# ──────────────────────────── tool_maker: flag gate ────────────────────────────


def test_tool_maker_basic_tier_has_no_flag_gate(tmp_path):
    """Every basic tool must run without enable_toolsmith=True."""
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding(flags={})  # flag OFF
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    # read
    r = tool_maker.tool_read_tool_source("my_toolset", env=env)
    assert r["success"] is True
    assert "tool_hello" in r["source"]

    # write (full replacement — no patch)
    r = tool_maker.tool_write_tool_source(
        "my_toolset",
        GOOD_TOOLSET_SOURCE + "\n\ndef tool_world() -> dict:\n    return {'ok': True}\n",
        env=env,
    )
    assert r["success"] is True
    assert r["mode"] == "full"

    # test (AST + isolated import; must succeed)
    r = tool_maker.tool_test_tool_source("my_toolset", env=env)
    assert r["success"] is True
    assert r["syntax_ok"] is True
    assert r["import_ok"] is True

    # list versions
    r = tool_maker.tool_list_tool_versions("my_toolset", env=env)
    assert r["success"] is True
    assert r["git_available"] in (True, False)  # env-agnostic
    assert r["backup_count"] >= 1  # write auto-snapshots

    # revert to the pre-write snapshot (undo last write)
    r = tool_maker.tool_revert_tool_source("my_toolset", backup_index=0, env=env)
    assert r["success"] is True
    assert r["restored_from"].startswith("backup:")
    assert r["test_ok"] is True

    # discard (no release + no git → oldest snapshot)
    r = tool_maker.tool_discard_changes("my_toolset", env=env)
    assert r["success"] is True


def test_tool_maker_rsi_tier_hard_errors_without_flag(tmp_path):
    """run_test_agent + commit_toolset return a hard error without the flag."""
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding(flags={})  # enable_toolsmith absent
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    r = tool_maker.tool_run_test_agent("my_toolset", "test", env=env)
    assert r["success"] is False
    assert r["requires_flag"] == "enable_toolsmith"
    assert "enable_toolsmith" in r["error"]

    r = tool_maker.tool_commit_toolset("my_toolset", env=env)
    assert r["success"] is False
    assert r["requires_flag"] == "enable_toolsmith"


def test_tool_maker_rsi_tier_allowed_with_flag(tmp_path):
    """With the flag on, the RSI tools are admitted past the gate."""
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding(flags={"enable_toolsmith": True})
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    # commit_toolset works end-to-end without a real LLM or git.
    r = tool_maker.tool_commit_toolset("my_toolset", "unit test", env=env)
    assert r["success"] is True, r
    assert r["release_id"], "release manifest sidecar must be created"
    assert Path(r["release_file"]).is_file()
    assert binding.remount_calls == ["my_toolset"]
    # release manifest embeds the source.
    manifest = json.loads(Path(r["release_file"]).read_text(encoding="utf-8"))
    assert manifest["source"] == GOOD_TOOLSET_SOURCE or "tool_hello" in manifest["source"]


# ──────────────────────── tool_maker: patch safety ────────────────────────


def test_tool_maker_aider_patch_rejected_on_partial_match(tmp_path):
    """A patch where one hunk's SEARCH block does not match must abort the write."""
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding()
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    # First hunk matches, second hunk does not.
    patch = (
        "<<<<<<< SEARCH\n"
        "TOOL_LIBRARY_NAME = \"my_toolset\"\n"
        "=======\n"
        "TOOL_LIBRARY_NAME = \"renamed\"\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "THIS_LINE_DOES_NOT_EXIST = True\n"
        "=======\n"
        "THIS_LINE_ALSO_DOES_NOT_EXIST = False\n"
        ">>>>>>> REPLACE\n"
    )
    original_before = py.read_text(encoding="utf-8")
    r = tool_maker.tool_write_tool_source("my_toolset", patch, env=env)
    assert r["success"] is False
    assert "partial patch" in r["error"].lower() or "not found" in r["error"].lower()
    # On-disk file MUST be untouched (partial application forbidden).
    assert py.read_text(encoding="utf-8") == original_before


def test_tool_maker_aider_patch_applies_cleanly(tmp_path):
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding()
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    patch = (
        "<<<<<<< SEARCH\n"
        "def tool_hello(name: str = \"world\") -> dict:\n"
        "    return {\"success\": True, \"output\": f\"hello {name}\"}\n"
        "=======\n"
        "def tool_hello(name: str = \"world\") -> dict:\n"
        "    return {\"success\": True, \"output\": f\"HELLO {name.upper()}\"}\n"
        ">>>>>>> REPLACE"
    )
    r = tool_maker.tool_write_tool_source("my_toolset", patch, env=env)
    assert r["success"] is True
    assert r["mode"] == "patch"
    assert "HELLO" in py.read_text(encoding="utf-8")


# ──────────────────────── tool_maker: revert durability ────────────────────────


def test_tool_maker_revert_rolls_back_if_restored_snapshot_is_broken(tmp_path, monkeypatch):
    """If a chosen snapshot is syntactically broken, revert restores pre-revert content."""
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding()
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    # Simulate a snapshot that is broken.
    backup_dir = py.parent / ".rsi_backup"
    backup_dir.mkdir()
    broken = backup_dir / f"{py.name}.9999999999.bak"
    broken.write_text("def tool_x(:\n  pass\n", encoding="utf-8")  # syntax error

    original_on_disk = py.read_text(encoding="utf-8")
    r = tool_maker.tool_revert_tool_source("my_toolset", backup_index=0, env=env)
    assert r["success"] is False
    # Revert-verify-rollback guarantee: current content is re-written.
    assert py.read_text(encoding="utf-8") == original_on_disk
    # The pre-revert snapshot the tool created first is still in place.
    assert len(list(backup_dir.glob("*.bak"))) >= 2


# ──────────────────── tool_maker: forbidden + HALT gates ────────────────────


def test_tool_maker_refuses_forbidden_test_targets(tmp_path):
    """_FORBIDDEN_TEST_TARGETS must be refused before spawning the child."""
    py = tmp_path / "system_shell" / "TOOLS.py"
    py.parent.mkdir()
    py.write_text(GOOD_TOOLSET_SOURCE, encoding="utf-8")
    binding = StubBinding(flags={"enable_toolsmith": True})
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    r = tool_maker.tool_run_test_agent("system_shell", "test", env=env)
    assert r["success"] is False
    assert "code-execution toolset" in r["error"]


@pytest.mark.parametrize("tool_attr", [
    "tool_write_tool_source",
    "tool_test_tool_source",
    "tool_run_test_agent",
    "tool_commit_toolset",
    "tool_revert_tool_source",
    "tool_discard_changes",
])
def test_tool_maker_halt_short_circuits_mutating_tools(tmp_path, tool_attr):
    py = _make_toolset(tmp_path, content=GOOD_TOOLSET_SOURCE)
    binding = StubBinding(flags={"enable_toolsmith": True})
    _seed_discovered(binding, py, "tool_hello")
    env = StubEnv(client=StubClient(binding))

    original_before = py.read_text(encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(tool_maker, "_HALT_REQUESTED", True, raising=False)
    try:
        fn = getattr(tool_maker, tool_attr)
        if tool_attr == "tool_run_test_agent":
            # Signature: (tool_name_or_stem, test_task, max_steps=..., timeout_sec=...)
            r = fn("my_toolset", "test_task_placeholder", env=env)
        elif tool_attr == "tool_commit_toolset":
            # Signature: (tool_name_or_stem, commit_message=..., ...)
            r = fn("my_toolset", env=env)
        elif tool_attr == "tool_write_tool_source":
            r = fn("my_toolset", GOOD_TOOLSET_SOURCE, env=env)
        else:
            r = fn("my_toolset", env=env)
        assert r["success"] is False
        assert "HALT" in r["error"]
        assert py.read_text(encoding="utf-8") == original_before
    finally:
        monkeypatch.undo()


# ─────────────────────────── LCP discovery filters ───────────────────────────


def _build_filters_module():
    """Import the filters via a source-level AST extraction to avoid pulling the
    entire LCPBinding into the test — but we need the real implementation, so
    we import the class and call its methods on a bare stub object that only
    has the attributes the filters touch."""
    from lollms_client.tools_bindings.lcp import LCPBinding

    class Bare:
        session_flags: Dict[str, Any]

        # Bind the two filters + helpers directly from the real class.
        _is_toolsmith_flag_on = LCPBinding._is_toolsmith_flag_on
        _is_skillmaker_flag_on = LCPBinding._is_skillmaker_flag_on
        _filter_toolsmith_visibility = LCPBinding._filter_toolsmith_visibility
        _filter_skillmaker_visibility = LCPBinding._filter_skillmaker_visibility
        _TOOLSMITH_FLAG_GATED_TOOL_NAMES = LCPBinding._TOOLSMITH_FLAG_GATED_TOOL_NAMES
        _SKILLMAKER_FLAG_GATED_TOOL_NAMES = LCPBinding._SKILLMAKER_FLAG_GATED_TOOL_NAMES

    return Bare


def _seeded_tools() -> List[Dict[str, Any]]:
    return [
        # tool_maker basic tier
        {"name": "tool_read_tool_source", "_library_name": "tool_maker"},
        {"name": "tool_write_tool_source", "_library_name": "tool_maker"},
        {"name": "tool_test_tool_source", "_library_name": "tool_maker"},
        {"name": "tool_list_tool_versions", "_library_name": "tool_maker"},
        {"name": "tool_revert_tool_source", "_library_name": "tool_maker"},
        {"name": "tool_discard_changes", "_library_name": "tool_maker"},
        # tool_maker RSI tier
        {"name": "tool_run_test_agent", "_library_name": "tool_maker"},
        {"name": "tool_commit_toolset", "_library_name": "tool_maker"},
        # skill_maker basic tier
        {"name": "tool_read_skill_source", "_library_name": "skill_maker"},
        {"name": "tool_write_skill_source", "_library_name": "skill_maker"},
        {"name": "tool_test_skill_source", "_library_name": "skill_maker"},
        {"name": "tool_list_skill_versions", "_library_name": "skill_maker"},
        {"name": "tool_revert_skill", "_library_name": "skill_maker"},
        {"name": "tool_discard_skill_changes", "_library_name": "skill_maker"},
        # skill_maker RSI tier
        {"name": "tool_run_skill_test_agent", "_library_name": "skill_maker"},
        {"name": "tool_commit_skill", "_library_name": "skill_maker"},
        # unrelated toolset that happens to share a name (defence-in-depth)
        {"name": "tool_run_test_agent", "_library_name": "some_other_toolset"},
    ]


def test_filter_hides_only_toolsmith_rsi_when_flag_off():
    Bare = _build_filters_module()
    b = Bare()
    b.session_flags = {}
    out = b._filter_toolsmith_visibility(_seeded_tools())
    names = {t["name"] for t in out}
    assert "tool_read_tool_source" in names           # basic tier retained
    assert "tool_write_tool_source" in names
    assert "tool_run_skill_test_agent" in names       # unrelated RSI: retained
    # The two tool_maker RSI tools dropped:
    assert "tool_commit_toolset" not in names
    # But the OTHER toolset's tool with the same name survives:
    assert any(
        t["name"] == "tool_run_test_agent" and t["_library_name"] == "some_other_toolset"
        for t in out
    )
    assert not any(
        t["name"] == "tool_run_test_agent" and t["_library_name"] == "tool_maker"
        for t in out
    )


def test_filter_keeps_all_when_flag_on():
    Bare = _build_filters_module()
    b = Bare()
    b.session_flags = {"enable_toolsmith": True, "enable_skillmaker": True}
    out = b._filter_toolsmith_visibility(_seeded_tools())
    out = b._filter_skillmaker_visibility(out)
    assert {t["name"] for t in out} == {t["name"] for t in _seeded_tools()}


def test_filter_skillmaker_independent_from_toolsmith():
    Bare = _build_filters_module()
    b = Bare()
    b.session_flags = {"enable_toolsmith": False, "enable_skillmaker": False}
    out = b._filter_toolsmith_visibility(_seeded_tools())
    out = b._filter_skillmaker_visibility(out)
    names = {t["name"] for t in out}
    # Both RSI tiers hidden
    assert "tool_run_test_agent" not in {
        t["name"] for t in out if t["_library_name"] == "tool_maker"
    }
    assert "tool_commit_skill" not in names
    assert "tool_run_skill_test_agent" not in names
    # Both basic tiers retained
    assert "tool_read_tool_source" in names
    assert "tool_write_skill_source" in names


# ──────────────────────── skill_maker: flag gate ────────────────────────


GOOD_SKILL_MD = textwrap.dedent(
    """
    ---
    name: demo_skill
    title: "Demo skill for unit tests"
    description: "A minimal skill used to exercise skill_maker without model calls."
    category: test
    tools:
      - tool_load_skill
    ---

    # Demo skill

    This skill only exists to test the skill_maker two-tier split.
    """
).lstrip()


def _make_skill(tmp_path: Path) -> Path:
    """Idempotent: safe to call multiple times on the same tmp_path.

    Tests that call it both directly and via ``_skill_env(...)`` must not
    blow up on the second invocation — the skill directory is a fixture,
    not a resource to be consumed once.
    """
    skill_dir = tmp_path / "skills_root" / "demo_skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    f = skill_dir / "SKILL.md"
    f.write_text(GOOD_SKILL_MD, encoding="utf-8")
    return f


def _stub_skill_personality(skill_path: Path):
    """Personality stub whose skills_manager reports one registered skill."""
    class _Mgr:
        def list_skills(self):
            return [{
                "title": "Demo skill",
                "description": "…",
                "file_path": str(skill_path),
                "is_handbag": False,
            }]
        def reload(self):
            return None

    return SimpleNamespace(skills_manager=_Mgr())


def _skill_env(tmp_path: Path, *, flags: Optional[Dict[str, Any]] = None) -> StubEnv:
    skill = _make_skill(tmp_path)
    personality = _stub_skill_personality(skill)
    binding = StubBinding(flags=flags or {})
    client = StubClient(binding, personality=personality)
    return StubEnv(client=client)


def test_skill_maker_basic_tier_has_no_flag_gate(tmp_path):
    env = _skill_env(tmp_path, flags={})

    r = skill_maker.tool_read_skill_source("demo_skill", env=env)
    assert r["success"] is True
    assert "name: demo_skill" in r["source"]

    r = skill_maker.tool_test_skill_source("demo_skill", env=env)
    assert r["success"] is True
    assert r["frontmatter_ok"] is True
    assert r["structure_ok"] is True
    assert r["declared_tools"] == ["tool_load_skill"]

    r = skill_maker.tool_write_skill_source(
        "demo_skill", GOOD_SKILL_MD + "\n\n## Notes\n\nmore content here.\n", env=env,
    )
    assert r["success"] is True
    assert r["mode"] == "full"

    r = skill_maker.tool_list_skill_versions("demo_skill", env=env)
    assert r["success"] is True
    assert r["backup_count"] >= 1

    r = skill_maker.tool_revert_skill("demo_skill", env=env)
    assert r["success"] is True
    assert r["test_ok"] is True

    r = skill_maker.tool_discard_skill_changes("demo_skill", env=env)
    assert r["success"] is True


def test_skill_maker_rsi_tier_hard_errors_without_flag(tmp_path):
    env = _skill_env(tmp_path, flags={})
    r = skill_maker.tool_run_skill_test_agent("demo_skill", "vet", env=env)
    assert r["success"] is False
    assert r["requires_flag"] == "enable_skillmaker"
    r = skill_maker.tool_commit_skill("demo_skill", env=env)
    assert r["success"] is False
    assert r["requires_flag"] == "enable_skillmaker"


def test_skill_maker_commit_writes_release_manifest_without_git(tmp_path):
    env = _skill_env(tmp_path, flags={"enable_skillmaker": True})

    r = skill_maker.tool_commit_skill("demo_skill", "unit test", env=env)
    assert r["success"] is True, r
    release_file = Path(r["release_file"])
    assert release_file.is_file()
    manifest = json.loads(release_file.read_text(encoding="utf-8"))
    assert manifest["skill_slug"] == "demo_skill"
    assert "name: demo_skill" in manifest["source"]
    # remounted on the personality
    assert r["remounted"] is True


# ──────────────────────── skill_maker: validation gates ────────────────────────


def test_skill_maker_write_rejects_broken_frontmatter(tmp_path):
    env = _skill_env(tmp_path, flags={})
    bad = "---\nname: demo_skill\n---\n\nshort\n"  # missing description, body too short
    r = skill_maker.tool_write_skill_source("demo_skill", bad, env=env)
    assert r["success"] is False
    assert "rejected" in r["error"].lower() or "description" in r["error"].lower()


def test_skill_maker_revert_rolls_back_on_broken_snapshot(tmp_path):
    env = _skill_env(tmp_path, flags={})
    skill_file = _make_skill(tmp_path)  # path re-created; matches personality lookup
    backup_dir = skill_file.parent / ".rsi_backup"
    backup_dir.mkdir(exist_ok=True)
    broken = backup_dir / f"{skill_file.name}.9999999999.bak"
    broken.write_text("---\nname:\n---\n", encoding="utf-8")  # invalid

    original = skill_file.read_text(encoding="utf-8")
    r = skill_maker.tool_revert_skill("demo_skill", backup_index=0, env=env)
    assert r["success"] is False
    assert skill_file.read_text(encoding="utf-8") == original


# ──────────────────── skill_maker: forbidden + HALT gates ────────────────────


def test_skill_maker_refuses_forbidden_test_skill_tools(tmp_path):
    env = _skill_env(tmp_path, flags={"enable_skillmaker": True})
    skill_file = _make_skill(tmp_path)
    skill_file.write_text(
        GOOD_SKILL_MD.replace(
            "tools:\n  - tool_load_skill",
            "tools:\n  - tool_load_skill\n  - execute_python",
        ),
        encoding="utf-8",
    )
    r = skill_maker.tool_run_skill_test_agent("demo_skill", "test", env=env)
    assert r["success"] is False
    assert "code-execution" in r["error"].lower() or "execute_python" in r["error"]


@pytest.mark.parametrize("tool_attr", [
    "tool_write_skill_source",
    "tool_test_skill_source",
    "tool_run_skill_test_agent",
    "tool_commit_skill",
    "tool_revert_skill",
    "tool_discard_skill_changes",
])
def test_skill_maker_halt_short_circuits_mutating_tools(tmp_path, tool_attr):
    env = _skill_env(tmp_path, flags={"enable_skillmaker": True})
    skill_file = _make_skill(tmp_path)

    mp = pytest.MonkeyPatch()
    mp.setattr(skill_maker, "_HALT_REQUESTED", True, raising=False)
    original_before = skill_file.read_text(encoding="utf-8")
    try:
        fn = getattr(skill_maker, tool_attr)
        if tool_attr == "tool_run_skill_test_agent":
            # Signature: (skill_name_or_path, test_task, max_steps=..., timeout_sec=...)
            r = fn("demo_skill", "test_task_placeholder", env=env)
        elif tool_attr == "tool_write_skill_source":
            r = fn("demo_skill", GOOD_SKILL_MD, env=env)
        else:
            r = fn("demo_skill", env=env)
        assert r["success"] is False
        assert "HALT" in r["error"]
        assert skill_file.read_text(encoding="utf-8") == original_before
    finally:
        mp.undo()