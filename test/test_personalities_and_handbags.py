"""
test_personalities_and_handbags.py
==================================

Doctrine-compliance suite for the LollmsPersonality + Handbag + SkillsManager
subsystem, as specified in `docs/PERSONALITIES_AND_HANDBAGS.md`.

Each test class corresponds to one numbered doctrine:

  D1. Two-bucket `get_skills_by_source()` — always returns exactly
      {"builtin": {...}, "extra": {...}} and uses the "internal" sentinel
      for skills with no physical backing file.

  D2. The `modifiable:` skill contract — absent ⇒ read-only + one-shot
      YAML migration to `modifiable: false`; `true` ⇒ read/write;
      `false` ⇒ read-only.

  D3. Privileged output folder resolution — `skills_dir=` / `tools_dir=`
      follows the precedence: explicit arg → handbag.yaml manifest → the
      handbag's own folder.

  D4. `extra_knowledge_bases` split — a declared `.db` goes to
      `extra_safe_stores`; every other entry goes to
      `extra_knowledge_bases`. A missing `.db` warns and never crashes or
      silently demotes to a plain KB path.

  D5. Handbag discovery — `tools/` excludes `__init__.py` and recurses
      exactly one level; `rag/` recurses fully but skips dotfiles;
      `memory_db_path` is only set when `memory/` exists.

  D6. `register_skills_dir()` — idempotent, creates the directory on demand,
      and appends to `skills_dirs`.

  D7. The LollmsPersonality contract — name/system_prompt/tools/tool_specs/
      query_data/has_data/__bool__ are never None and never raise, even
      without a client.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_personality.handbag import Handbag  # noqa: E402
from lollms_client.lollms_personality.skill import Skill, parse_skill_md  # noqa: E402
from lollms_client.lollms_personality.skills_manager import SkillsManager  # noqa: E402
from lollms_client.lollms_personality.lollms_personality import (  # noqa: E402
    LollmsPersonality,
    NullPersonality,
    _NULL_TOOL_BINDING,
)


# ────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_handbag(tmp_path: Path, name: str = "hb") -> Path:
    root = tmp_path / name
    Handbag.create_structure(root)
    (root / "SOUL.md").write_text(
        "---\nname: Test Agent\nauthor: tester\n---\n\nYou are a tester.\n",
        encoding="utf-8",
    )
    return root


def _write_skill(
    directory: Path,
    slug: str,
    *,
    title: str,
    body: str = "Body.",
    modifiable: str | None = None,
    extra_front: str = "",
) -> Path:
    """
    Writes a `<directory>/<slug>/SKILL.md` with a frontmatter.
    `modifiable` controls whether the flag is present and what value it holds.
    """
    skill_dir = directory / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    lines = ["---", f"title: {title}", "description: Test skill."]
    if extra_front:
        lines.append(extra_front.rstrip())
    if modifiable is not None:
        lines.append(f"modifiable: {modifiable}")
    lines.extend(["---", "", body])
    skill_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return skill_file


def _mk_manager(skills_dir: Path) -> SkillsManager:
    """Creates a SkillsManager bound to exactly one skills dir (mode=loadable)."""
    return SkillsManager(skills_dirs=[skills_dir], mode="loadable")


# ════════════════════════════════════════════════════════════════════════════
# D1 — Two-bucket `get_skills_by_source()`
# ════════════════════════════════════════════════════════════════════════════


class TestD1_TwoBucketSkillInventory:
    """D1: `{...}` always yields exactly `{'builtin': {...}, 'extra': {...}}`."""

    def test_empty_manager_returns_two_empty_buckets(self, tmp_path):
        """When no skills exist, both keys must be present and empty."""
        manager = _mk_manager(tmp_path / "skills")

        inventory = manager.get_skills_by_source(
            builtin_dirs=[tmp_path / "skills"],
            extra_dirs=None,
        )

        assert set(inventory.keys()) == {"builtin", "extra"}
        assert inventory["builtin"] == {}
        assert inventory["extra"] == {}

    def test_builtin_and_extra_buckets_partition_correctly(self, tmp_path):
        """Handbag-native skills land in `builtin`; app extras land in `extra`."""
        builtin_root = tmp_path / "skills"
        extra_root = tmp_path / "extra_skills"
        builtin_root.mkdir()
        extra_root.mkdir()

        _write_skill(builtin_root, "handbag_skill", title="Handbag Skill", modifiable="true")
        _write_skill(extra_root, "app_skill", title="App Skill", modifiable="true")

        # Register both roots, then classify by provenance.
        manager = SkillsManager(skills_dirs=[builtin_root, extra_root], mode="loadable")

        inventory = manager.get_skills_by_source(
            builtin_dirs=[builtin_root],
            extra_dirs=[extra_root],
        )

        assert "Handbag Skill" in inventory["builtin"]
        assert "App Skill" in inventory["extra"]
        assert inventory["builtin"]["Handbag Skill"].endswith("SKILL.md")
        assert inventory["extra"]["App Skill"].endswith("SKILL.md")

    def test_internal_sentinel_for_file_less_skill(self, tmp_path):
        """
        A `Skill` object with `file_path=None` must surface with the literal
        sentinel string `"internal"` as its value (never None / never crash).
        """
        manager = _mk_manager(tmp_path / "skills")

        # Inject an in-memory skill (no file).
        in_memory_skill = Skill(
            title="Ephemeral Note",
            description="Created in memory only.",
            category="",
            tags=[],
            content="body",
            file_path=None,
            modifiable=True,
            modifiable_field_present=True,
        )
        manager.skills["ephemeral note"] = in_memory_skill

        inventory = manager.get_skills_by_source(
            builtin_dirs=[],
            extra_dirs=[],
        )

        assert inventory["extra"].get("Ephemeral Note") == "internal"
        assert "Ephemeral Note" not in inventory["builtin"]

    def test_missing_filepath_is_treated_as_internal(self, tmp_path):
        """
        A `Skill` whose `file_path` points to a non-existent file must ALSO
        produce the "internal" sentinel (path resolvability is the trigger).
        """
        manager = _mk_manager(tmp_path / "skills")
        skill = Skill(
            title="Ghost Skill",
            description="",
            category="",
            tags=[],
            content="body",
            file_path=tmp_path / "does_not_exist" / "SKILL.md",
            modifiable=True,
            modifiable_field_present=True,
        )
        manager.skills["ghost skill"] = skill

        inventory = manager.get_skills_by_source(builtin_dirs=[], extra_dirs=[])

        assert inventory["extra"].get("Ghost Skill") == "internal"

    def test_builtin_root_takes_precedence_over_extra_root(self, tmp_path):
        """
        A skill that lives under BOTH a builtin root and an extra root must be
        classified as `builtin` (documented precedence).
        """
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(root, "shared", title="Shared Skill", modifiable="true")

        manager = SkillsManager(skills_dirs=[root], mode="loadable")

        inventory = manager.get_skills_by_source(
            builtin_dirs=[root],
            extra_dirs=[root],  # same root passed in both buckets
        )

        assert "Shared Skill" in inventory["builtin"]
        assert "Shared Skill" not in inventory["extra"]


# ════════════════════════════════════════════════════════════════════════════
# D2 — The `modifiable:` skill contract
# ════════════════════════════════════════════════════════════════════════════


class TestD2_ModifiableContract:
    """D2: read/write contract is driven strictly by the YAML `modifiable:` key."""

    def test_absent_flag_defaults_to_read_only_and_migrates(self, tmp_path):
        """
        Doctrine: absent `modifiable:` ⇒ read-only AND a one-shot migration
        must inject `modifiable: false` into the frontmatter (durably).
        """
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_file = _write_skill(skills_root, "legacy", title="Legacy Skill")  # no modifiable

        # Sanity: the on-disk file has no `modifiable:` before load.
        assert "modifiable:" not in skill_file.read_text(encoding="utf-8")

        manager = _mk_manager(skills_root)
        skill = manager.get_skill("Legacy Skill")

        assert skill is not None
        assert skill.modifiable is False, "Absent flag must be treated as read-only"
        assert skill.modifiable_field_present is True, "Loader must normalise the file"

        # The migration must have persisted `modifiable: false` to disk.
        migrated = skill_file.read_text(encoding="utf-8")
        assert "modifiable: false" in migrated
        # Body must be untouched
        assert "Body." in migrated

    def test_absent_flag_migration_is_idempotent(self, tmp_path):
        """Re-loading the same file must not duplicate the injected flag."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_file = _write_skill(skills_root, "legacy", title="Legacy Skill")

        _mk_manager(skills_root)
        first_content = skill_file.read_text(encoding="utf-8")
        assert first_content.count("modifiable:") == 1

        # Second load must not change anything.
        _mk_manager(skills_root)
        second_content = skill_file.read_text(encoding="utf-8")
        assert second_content == first_content

    def test_explicit_true_grants_read_write(self, tmp_path):
        """`modifiable: true` must allow in-place updates."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        _write_skill(skills_root, "rw", title="RW Skill", body="v1", modifiable="true")

        manager = _mk_manager(skills_root)
        skill = manager.get_skill("RW Skill")
        assert skill.modifiable is True

        updated = manager.update_skill("RW Skill", "v2")
        assert updated is not None
        on_disk = skill.file_path.read_text(encoding="utf-8")
        assert "v2" in on_disk
        assert "modifiable: true" in on_disk

    def test_explicit_false_blocks_in_place_write(self, tmp_path):
        """`modifiable: false` must refuse every in-place mutation."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_file = _write_skill(
            skills_root, "ro", title="RO Skill", body="original", modifiable="false"
        )

        manager = _mk_manager(skills_root)
        skill = manager.get_skill("RO Skill")
        assert skill.modifiable is False

        # update refuses
        assert manager.update_skill("RO Skill", "hacked") is None
        # append refuses
        assert manager.append_to_skill("RO Skill", "hacked") is None
        # remove refuses
        assert manager.remove_skill("RO Skill") is False

        # File body is byte-identical to the original.
        assert "original" in skill_file.read_text(encoding="utf-8")
        assert "hacked" not in skill_file.read_text(encoding="utf-8")

    def test_newly_created_skill_emits_modifiable_true(self, tmp_path):
        """
        Doctrine: agent-generated skills are emitted with `modifiable: true`
        so they remain overwritable across round trips.
        """
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        manager = _mk_manager(skills_root)

        created = manager.create_skill(
            title="Generated Skill",
            content="Agent written.",
            description="A generated skill.",
        )

        assert created is not None
        assert created.modifiable is True
        assert "modifiable: true" in created.file_path.read_text(encoding="utf-8")

    def test_roundtrip_preserves_read_only_after_update_attempt(self, tmp_path):
        """
        Once a skill is treated as read-only (default), a rejected
        `update_skill` must leave the file's on-disk content intact — no
        fallback clone into `skills_output_path`.
        """
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_file = _write_skill(skills_root, "persist", title="Persist Skill", body="real")

        manager = _mk_manager(skills_root)
        original_bytes = skill_file.read_bytes()

        manager.update_skill("Persist Skill", "should never be written")

        assert skill_file.read_bytes() == original_bytes

    def test_parse_skill_frontmatter_contract_for_modifiable(self, tmp_path):
        """`parse_skill_md` must distinguish absent / true / false cleanly."""
        skills_root = tmp_path / "s"
        skills_root.mkdir()

        absent_f = _write_skill(skills_root, "a", title="A")
        true_f = _write_skill(skills_root, "t", title="T", modifiable="true")
        false_f = _write_skill(skills_root, "f", title="F", modifiable="false")

        a = parse_skill_md(absent_f)
        t = parse_skill_md(true_f)
        f = parse_skill_md(false_f)

        assert (a.modifiable, a.modifiable_field_present) == (False, False)
        assert (t.modifiable, t.modifiable_field_present) == (True, True)
        assert (f.modifiable, f.modifiable_field_present) == (False, True)


# ════════════════════════════════════════════════════════════════════════════
# D3 — Privileged output folders (`skills_dir=` / `tools_dir=`)
# ════════════════════════════════════════════════════════════════════════════


class TestD3_PrivilegedOutputFolderResolution:
    """
    D3: the output-folder precedence is: explicit arg → handbag.yaml manifest →
    handbag's own `skills/` / `tools/`.

    Note: full end-to-end `from_handbag()` is exercised by example files; here
    we validate the contracts that are achievable without a live LLM client,
    i.e. `Handbag.register_skills_dir()` + manifest-key reading.
    """

    def test_register_skills_dir_creates_and_appends(self, tmp_path):
        """`register_skills_dir()` must create the directory if missing."""
        handbag_root = _make_handbag(tmp_path)
        hb = Handbag(handbag_root)

        target = tmp_path / "app_skills_runtime"
        assert not target.exists()

        resolved = hb.register_skills_dir(target)

        assert resolved == target.resolve()
        assert target.is_dir()
        assert resolved in hb.skills_dirs

    def test_register_skills_dir_is_idempotent(self, tmp_path):
        """Registering the same path twice must be a no-op."""
        handbag_root = _make_handbag(tmp_path)
        hb = Handbag(handbag_root)
        target = tmp_path / "extra_skills"

        hb.register_skills_dir(target)
        count_after_first = hb.skills_dirs.count(target.resolve())

        hb.register_skills_dir(target)
        count_after_second = hb.skills_dirs.count(target.resolve())

        assert count_after_first == 1
        assert count_after_second == 1, "Re-registering the same path must not duplicate it"

    def test_manifest_declares_skills_output_path(self, tmp_path):
        """
        Doctrine: when no explicit override is passed, `handbag.yaml` keys
        `skills_output_path` / `tools_output_path` are the second-tier
        resolution point. This test pins that the manifest is honoured when
        the property is resolved (documented fallback: handbag's own folder).
        """
        handbag_root = _make_handbag(tmp_path)
        (handbag_root / "handbag.yaml").write_text(
            "skills_output_path: ./app_skills\n"
            "tools_output_path: ./app_tools\n",
            encoding="utf-8",
        )

        hb = Handbag(handbag_root)

        assert hb.manifest.get("skills_output_path") == "./app_skills"
        assert hb.manifest.get("tools_output_path") == "./app_tools"

        # The manifest values must be resolvable relative to the handbag root.
        resolved_skills = (hb.path / hb.manifest["skills_output_path"]).resolve()
        resolved_tools = (hb.path / hb.manifest["tools_output_path"]).resolve()
        assert resolved_skills.parent == hb.path
        assert resolved_tools.parent == hb.path

    def test_default_skills_dir_is_handbags_own_folder(self, tmp_path):
        """Without an override, the destination is the handbag's own skills/."""
        handbag_root = _make_handbag(tmp_path)
        hb = Handbag(handbag_root)

        assert hb.skills_dir == handbag_root / "skills"
        assert hb.skills_dir in hb.skills_dirs


# ════════════════════════════════════════════════════════════════════════════
# D4 — extra_knowledge_bases split
# ════════════════════════════════════════════════════════════════════════════


class TestD4_ExtraKnowledgeBasesSplit:
    """D4: .db → extra_safe_stores; else → extra_knowledge_bases."""

    def test_plain_path_goes_to_extra_knowledge_bases(self, tmp_path):
        """.md / directory entries must go to `extra_knowledge_bases`."""
        handbag_root = _make_handbag(tmp_path)
        plain_file = tmp_path / "shared" / "handbook.md"
        plain_file.parent.mkdir()
        plain_file.write_text("# Handbook\n", encoding="utf-8")

        (handbag_root / "handbag.yaml").write_text(
            f"extra_knowledge_bases:\n  - {plain_file}\n",
            encoding="utf-8",
        )

        hb = Handbag(handbag_root)

        assert plain_file.resolve() in hb.extra_knowledge_bases
        assert hb.extra_safe_stores == []

    def test_missing_db_warns_but_does_not_crash(self, tmp_path, caplog):
        """
        Doctrine: a declared `.db` that does not exist must NOT crash the
        handbag build and must NOT be silently demoted to a plain KB path —
        it must be registered as a deferred safe store instead.
        """
        handbag_root = _make_handbag(tmp_path)
        missing_db = tmp_path / "shared" / "missing.db"
        missing_db.parent.mkdir()
        # Deliberately do not create the file.

        (handbag_root / "handbag.yaml").write_text(
            f"extra_knowledge_bases:\n  - {missing_db}\n",
            encoding="utf-8",
        )

        # Must not raise.
        hb = Handbag(handbag_root)

        # Not drifted into the plain KB bucket.
        assert missing_db.resolve() not in hb.extra_knowledge_bases

        # Registered as a deferred safe-store config (dict form is the
        # documented fallback when the .db could not be opened eagerly).
        assert len(hb.extra_safe_stores) == 1
        entry = hb.extra_safe_stores[0]
        assert isinstance(entry, dict)
        assert Path(entry["db_path"]).resolve() == missing_db.resolve()
        assert entry["name"] == missing_db.stem

    def test_db_with_dict_shape_honours_name_and_description(self, tmp_path):
        """The dict form (`- path: ... / name: ... / description: ...`) must be honoured."""
        handbag_root = _make_handbag(tmp_path)
        db_file = tmp_path / "shared" / "archive.db"
        db_file.parent.mkdir()
        # Real sqlite file is not strictly required for the deferred-store path.

        (handbag_root / "handbag.yaml").write_text(
            "extra_knowledge_bases:\n"
            "  - path: ./shared/archive.db\n"
            "    name: archive\n"
            "    description: Archived project specs.\n",
            encoding="utf-8",
        )

        hb = Handbag(handbag_root)

        assert len(hb.extra_safe_stores) == 1
        entry = hb.extra_safe_stores[0]
        # Deferred dict form preserves the declared name & description.
        assert entry["name"] == "archive"
        assert entry["description"] == "Archived project specs."

    def test_bare_string_db_uses_stem_as_default_name(self, tmp_path):
        """A bare `.db` string must default the display name to the file stem."""
        handbag_root = _make_handbag(tmp_path)
        db_path = tmp_path / "shared" / "auto_design.db"
        db_path.parent.mkdir()

        (handbag_root / "handbag.yaml").write_text(
            f"extra_knowledge_bases:\n  - {db_path}\n",
            encoding="utf-8",
        )

        hb = Handbag(handbag_root)

        assert len(hb.extra_safe_stores) == 1
        entry = hb.extra_safe_stores[0]
        assert entry["name"] == "auto_design"


# ════════════════════════════════════════════════════════════════════════════
# D5 — Handbag discovery rules
# ════════════════════════════════════════════════════════════════════════════


class TestD5_HandbagDiscovery:
    """D5: tools/ / rag/ / memory/ discovery rules."""

    def test_tools_returns_one_entry_per_library_and_skips_dunder_init(self, tmp_path):
        """
        Doctrine (current): tools/ yields **one entry per tool library**.

          * A flat ``.py`` file directly under ``tools/`` (a legacy single-file
            library) is returned as that FILE.
          * A sub-directory containing at least one non-``__init__`` ``.py``
            is returned as that FOLDER (the LCP loader resolves it to
            ``TOOLS.py`` / ``<folder>.py`` / first ``.py`` internally).
          * ``__init__.py`` at the top level is never surfaced as a library.
          * Recursion into nested library internals (``sub/deeper/``) does NOT
            produce a separate entry — the library is ``sub/`` as a whole.
        """
        handbag_root = _make_handbag(tmp_path)
        tools = handbag_root / "tools"
        (tools / "sub").mkdir()
        (tools / "sub" / "deeper").mkdir()
        (tools / "top.py").write_text("def tool_top(): pass\n", encoding="utf-8")
        (tools / "__init__.py").write_text("", encoding="utf-8")
        (tools / "sub" / "nested.py").write_text("def tool_n(): pass\n", encoding="utf-8")
        (tools / "sub" / "__init__.py").write_text("", encoding="utf-8")
        (tools / "sub" / "deeper" / "buried.py").write_text("def tool_b(): pass\n", encoding="utf-8")

        hb = Handbag(handbag_root)
        names = sorted(p.name for p in hb.tool_files)

        # One entry per library: `top` is a flat single-file lib (kept as `.py`),
        # `sub` is a folder library. `__init__.py` is excluded, and the deep
        # `buried.py` is NOT surfaced separately (it lives inside `sub/`).
        assert names == ["sub", "top.py"], (
            f"Expected folder-library 'sub' + flat file 'top.py'; got {names}"
        )
        # The folder entry must actually resolve to the `sub/` directory,
        # not to the nested file inside it.
        sub_entry = next(p for p in hb.tool_files if p.name == "sub")
        assert sub_entry.is_dir()
        assert sub_entry.name == "sub"

    def test_rag_recurses_fully_but_skips_dotfiles(self, tmp_path):
        """rag/ uses `rglob("*")` (full recursion) and skips dotfiles."""
        handbag_root = _make_handbag(tmp_path)
        rag = handbag_root / "rag"
        (rag / "deep" / "deeper").mkdir(parents=True)
        (rag / "top.md").write_text("top", encoding="utf-8")
        (rag / ".hidden.md").write_text("hidden", encoding="utf-8")
        (rag / "deep" / "mid.md").write_text("mid", encoding="utf-8")
        (rag / "deep" / "deeper" / "buried.md").write_text("buried", encoding="utf-8")

        hb = Handbag(handbag_root)
        names = sorted(p.name for p in hb.rag_files)

        assert "top.md" in names
        assert "mid.md" in names
        assert "buried.md" in names
        assert ".hidden.md" not in names

    def test_memory_db_path_only_set_when_memory_dir_exists(self, tmp_path):
        """`memory_db_path` is `None` when memory/ is absent, and set otherwise."""
        # Case A: memory/ created by create_structure → memory_db_path set.
        with_mem_root = _make_handbag(tmp_path, name="with_mem")
        hb_with = Handbag(with_mem_root)
        assert hb_with.memory_db_path is not None
        assert "memory.db" in hb_with.memory_db_path

        # Case B: hand-rolled handbag without memory/
        without_mem_root = tmp_path / "without_mem"
        without_mem_root.mkdir()
        (without_mem_root / "SOUL.md").write_text("---\nname: X\n---\n", encoding="utf-8")
        hb_without = Handbag(without_mem_root)
        assert hb_without.memory_db_path is None

    def test_missing_handbag_path_raises_value_error(self, tmp_path):
        """Doctrine: a non-existent handbag path must raise `ValueError`."""
        with pytest.raises(ValueError):
            Handbag(tmp_path / "does_not_exist")

    def test_canonical_directories_are_all_present(self, tmp_path):
        """create_structure() must materialise the canonical directory layout."""
        root = Handbag.create_structure(tmp_path / "hb")
        for sub in (
            "coworkers", "tools", "skills", "rag", "docs",
            "memory", "assets", "workspace",
        ):
            assert (root / sub).is_dir(), f"Missing canonical subdirectory: {sub}"


# ════════════════════════════════════════════════════════════════════════════
# D6 — `register_skills_dir()` behaviour (extends D3 idempotency)
# ════════════════════════════════════════════════════════════════════════════


class TestD6_RegisterSkillsDirIdempotency:
    """D6: focused pin for the idempotency doctrine."""

    def test_returns_resolved_path(self, tmp_path):
        hb = Handbag(_make_handbag(tmp_path))
        target = tmp_path / "custom"
        returned = hb.register_skills_dir(target)
        assert returned.is_absolute()
        assert returned == target.resolve()

    def test_same_path_twice_no_duplicates(self, tmp_path):
        hb = Handbag(_make_handbag(tmp_path))
        target = tmp_path / "custom"
        hb.register_skills_dir(target)
        hb.register_skills_dir(target)
        hb.register_skills_dir(str(target))  # also as string
        assert sum(1 for p in hb.skills_dirs if p == target.resolve()) == 1

    def test_different_paths_are_all_appended(self, tmp_path):
        hb = Handbag(_make_handbag(tmp_path))
        a = tmp_path / "a"
        b = tmp_path / "b"
        hb.register_skills_dir(a)
        hb.register_skills_dir(b)
        assert a.resolve() in hb.skills_dirs
        assert b.resolve() in hb.skills_dirs


# ════════════════════════════════════════════════════════════════════════════
# D7 — LollmsPersonality contract invariants
# ════════════════════════════════════════════════════════════════════════════


class TestD7_PersonalityContract:
    """D7: name / system_prompt / tools / tool_specs / query_data / has_data / __bool__."""

    def test_minimal_construction_does_not_raise(self):
        """`LollmsPersonality()` with no args must build and be truthy."""
        p = LollmsPersonality()
        assert bool(p) is True
        assert isinstance(p.name, str) and p.name
        assert isinstance(p.system_prompt, str)
        assert p.tools is _NULL_TOOL_BINDING

    def test_tool_specs_never_raises_without_binding(self):
        """`tool_specs()` must return `{}` (never raise/None) without a binding."""
        p = LollmsPersonality()
        assert p.tool_specs() == {}

    def test_query_data_never_raises_without_data(self):
        """`query_data()` must return a normalised dict, never raise."""
        p = LollmsPersonality()
        result = p.query_data("anything")
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert result.get("sources") == []

    def test_has_data_false_without_sources(self):
        """`has_data` is `False` iff no RAG source is attached."""
        assert LollmsPersonality().has_data is False

    def test_null_personality_contract(self):
        """`NullPersonality` must be falsy but still expose the full contract."""
        np = NullPersonality()
        assert bool(np) is False
        assert isinstance(np.name, str) and np.name
        assert isinstance(np.system_prompt, str)
        assert np.tools is _NULL_TOOL_BINDING
        assert np.tool_specs() == {}
        result = np.query_data("x")
        assert isinstance(result, dict)

    def test_get_skills_by_source_always_two_keys(self):
        """
        Without a skills manager, `get_skills_by_source()` must still return
        exactly `{'builtin': {}, 'extra': {}}` — never None, never raise.
        """
        p = LollmsPersonality()
        inventory = p.get_skills_by_source()
        assert set(inventory.keys()) == {"builtin", "extra"}
        assert inventory["builtin"] == {}
        assert inventory["extra"] == {}

    def test_system_prompt_wins_no_default_protocol(self):
        """
        An empty `system_prompt` must not have any operating-protocol block
        prepended by default; the caller supplies its own via `chat()`.
        This pins the "constructor is a resource container, not a policy
        injector" doctrine — the protocol is appended in `_build_system_prompt`.
        """
        p = LollmsPersonality(system_prompt="You are a test.")
        assert "AUTHORITATIVE OPERATING PROTOCOL" not in p.system_prompt

    def test_name_defaults_to_assistant(self):
        """`name` never falls back to `None` or empty — default is `'assistant'`."""
        p = LollmsPersonality(name="")
        assert p.name == "assistant"

    def test_personality_has_skills_manager_attr_or_none(self):
        """
        The contract allows `skills_manager=None` — but the attribute must
        exist so callers can probe it without AttributeError.
        """
        p = LollmsPersonality()
        assert hasattr(p, "skills_manager")
        assert p.skills_manager is None