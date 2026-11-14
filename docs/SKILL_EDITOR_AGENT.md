# Building a Skills-Editor Agent (Handbag + `extra_skills_dirs`)

A guide to constructing a **dedicated editor persona** whose single job is to inspect,
modify, and validate existing `SKILL.md` skills — standing only on the verified mechanics
of `LollmsPersonality`, `Handbag`, `SkillsManager`, and the `skill_maker` LCP toolset.

> **Scope**: This is the skills analogue of
> [`docs/TOOL_EDITOR_AGENT.md`](./TOOL_EDITOR_AGENT.md). Read that document first if you
> are new to handbag-based editor personas — the wiring pattern is the same, but the
> **hooks, persistence, and refresh callbacks differ** in important, verifiable ways. This
> document only advocates behaviors the codebase implements.

---

## 1. Overview — why a dedicated skills-editor persona + `extra_skills_dirs`

The **skills analogue** of the tool-editor pattern:

1. **Handbag-native skills** are scanned from `<handbag>/skills/` on every construction
   and on every refresh.
2. **A target skills ROOT** is hooked via `extra_skills_dirs=[<root>]`. Each root is a
   *directory* that gets scanned for `<slug>/SKILL.md` subfolders — **not** a list of
   individual `.md` files.
3. All roots are merged into a single `SkillsManager` which indexes every `SKILL.md` by
   title and by folder slug.
4. The persona's built-in skill-inspection tools are constructed by
   `SkillsManager.build_skill_tools()` and become part of the agent's tool registry:
   `tool_list_skills`, `tool_load_skill`, `tool_unload_skill`, `tool_search_skills`,
   `tool_create_skill`, `tool_update_skill`, `tool_append_to_skill`, `tool_remove_skill`.
   These are the *in-agent* skill API — distinct from `skill_maker`'s RSI tools.
5. For read/write editing that goes beyond the manager API — patch-based edits, snapshots,
   revert, discard, commit — the **`skill_maker` LCP toolset** provides the RSI surface.

`extra_skills_dirs` is therefore a **directory hook**, not a file hook. The parallel to
`extra_tools` is structural, not literal.

> **Critical default**: a `SKILL.md` whose frontmatter **omits** `modifiable:` is treated
> as **read-only**. `SkillsManager._reconcile_skill_metadata` migrates such files on load by
> *injecting* an explicit `modifiable: false` line — a one-shot metadata normalization that
> never rewrites the body. **If you want to edit a skill in place, its frontmatter must
> declare `modifiable: true`.**

---

## 2. Prerequisites

- A working `LollmsClient` (any LLM binding). Optional but expected for real editing.
- A **handbag folder** (see §4) containing at least `SOUL.md`.
- A **target skills ROOT directory** containing at least one `<slug>/SKILL.md`, with
  `modifiable: true` in the target skill's frontmatter.
- (Recommended) the `skill_maker` toolset folder —
  `src/lollms_client/tools_bindings/lcp/default_tools/skill_maker/` — hooked via
  `extra_tools=[...]` if the persona is to patch, revert, or commit skills.

**Read-only default warning** — say it twice:

- `modifiable: true` → writable in place.
- `modifiable: false` → read-only; `update_skill` / `tool_update_skill` refuse writes.
- `modifiable:` **absent** → treated as read-only. `SkillsManager._reconcile_skill_metadata`
  rewrites the file once to inject `modifiable: false` so the decision is durable.

```python
from pathlib import Path

skills_root = Path("workspace/skills")     # scanned for <slug>/SKILL.md
target_skill = skills_root / "csv_inspector" / "SKILL.md"
assert target_skill.is_file(), "Target skill folder must contain a SKILL.md"
assert "modifiable: true" in target_skill.read_text(encoding="utf-8"), (
    "Target SKILL.md must declare `modifiable: true` to be editable in place."
)
```

---

## 3. Step 1 — Write the editor persona (`SOUL.md`)

`from_handbag` parses `SOUL.md` (`--- ... ---` YAML frontmatter, then a markdown body).
`PersonalityBundle.parse_soul_md` extracts the frontmatter and uses the body as the
system prompt.

Scope the body **strictly to skill editing**: forbid unrelated tasks, mandate the edit
loop, and forbid touching read-only skills.

```markdown
---
name: SkillSmith
author: lollms-client
version: '1.0'
category: engineering
description: A focused agent that inspects, edits, and validates a single SKILL.md skill.
skills_mode: loadable
---

You are SkillSmith, a dedicated SKILL.md editor.

Your ONLY job is to inspect, edit, and validate the target skill attached to you
via `extra_skills_dirs`. You must NOT perform unrelated tasks (no web research,
no document writing, no code generation outside the skill folder).

## The mandatory edit loop

1. READ the target skill: `tool_load_skill(title="<slug>")`, then
   `tool_read_skill_source(skill_name_or_path="<slug>")`.
2. UNDERSTAND the current workflow: state in one sentence what the skill currently
   instructs the agent to do, and what is wrong with it.
3. EDIT the source via `tool_write_skill_source` — either a full replacement
   or an Aider SEARCH/REPLACE block. Never write a partial patch.
4. RELOAD the skills index by asking the host to call
   `refresh_tools_and_skills()` (or doing so yourself if you have that capability).
   Never claim an edit is "hot-loaded" before that happens.
5. VALIDATE with `tool_test_skill_source`. If it fails, iterate from step 3.

## Hard rules

- Never edit any file outside the attached skills root.
- Never touch a SKILL.md whose frontmatter lacks `modifiable: true`.
- Never weaken `required_tools` or `tools:` of the skill unless explicitly instructed.
- Every edit must keep the YAML frontmatter valid (name, description required).
```

Save this at `skills_editor_handbag/SOUL.md`.

---

## 4. Step 2 — Handbag layout

`from_handbag` calls `Handbag(handbag_path)` internally. The layout it consumes:

```text
skills_editor_handbag/
├── SOUL.md                  # persona (required)
├── handbag.yaml             # optional manifest (skills_mode, skills_output_path)
├── tools/                   # optional native toolsets
├── skills/<slug>/SKILL.md   # handbag-native skills (scanned)
├── coworkers/<name>/SOUL.md # optional crew
├── rag/                     # files auto-exposed as RAGDataSource
├── memory/memory.db         # memory (auto-created)
└── workspace/               # default local sandbox
```

Only `SOUL.md` is strictly required. `handbag.yaml` is where `skills_mode` lives:

```yaml
name: Skills Editor Handbag
version: '1.0'
description: A handbag dedicated to SKILL.md-editing personas.
skills_mode: loadable
```

`from_handbag` reads `name / author / category / description / skills_mode / temperature`
from the SOUL frontmatter — but note that `handbag.yaml` is where `SkillsManager` derives
its `mode` (`loadable` | `searchable` | `visible` | `mixed`).

Workspace resolution precedence (verbatim, from the implementation):

1. `workspace_path=` argument,
2. `workspace=` (legacy alias),
3. `hb.workspace_dir` if it exists,
4. `None`.

---

## 5. Step 3 — Hook the target skills root via `extra_skills_dirs=[...]`

The verified signature (`LollmsPersonality.from_handbag`, `lollms_personality.py`):

```python
@classmethod
def from_handbag(
    cls,
    handbag_path: Union[str, Path],
    lollms_client: Optional[Any] = None,
    extra_skills_dirs: Optional[List[Union[str, Path]]] = None,    # <-- THE HOOK
    extra_tools: Optional[List[Union[str, Path]]] = None,          # for skill_maker
    extra_knowledge_bases: Optional[List[Union[str, Path]]] = None,
    extra_safe_stores: Optional[List[Any]] = None,
    workspace_path: Optional[Union[str, Path]] = None,
    capabilities: Optional[Any] = None,
    model_params: Optional[Dict[str, Any]] = None,
    max_tokens_per_turn: int = 4096,
    workspace: Optional[Union[str, Path]] = None,                  # legacy alias
    **kwargs: Any,
) -> "LollmsPersonality":
    ...
```

Full, copy-pasteable construction — the analogue of the tool editor's constructor, with
**two hooks**: `extra_skills_dirs` for the skills root, `extra_tools` for `skill_maker`.

```python
from pathlib import Path
import lollms_client
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

handbag_path   = Path("skills_editor_handbag").resolve()
skills_root    = Path("workspace/skills").resolve()    # <slug>/SKILL.md under here
workspace_root = Path("workspace").resolve()

skill_maker_dir = (
    Path(lollms_client.__file__).parent
    / "tools_bindings" / "lcp" / "default_tools" / "skill_maker"
)

editor = LollmsPersonality.from_handbag(
    handbag_path=handbag_path,
    lollms_client=client,
    extra_skills_dirs=[skills_root],     # <-- the skills root becomes scanned
    extra_tools=[skill_maker_dir],       # <-- RSI surface for patch/revert/commit
    workspace_path=workspace_root,
    max_tokens_per_turn=8192,
)

print([s.title for s in editor.skills_manager.get_unique_skills()])
```

### What `extra_skills_dirs` actually does (verified)

Persistence, verbatim from `from_handbag`'s tail:

```python
object.__setattr__(pers, "_extra_tool_files",
    [Path(t).resolve() for t in (extra_tools or []) if Path(t).resolve().exists()])
object.__setattr__(pers, "_extra_skills_dirs",
    [Path(d).resolve() for d in (extra_skills_dirs or []) if Path(d).resolve().exists()])
```

Key facts to internalise:

- **`extra_skills_dirs` takes DIRECTORIES, not files.** Each entry will be handed to a
  `SkillsManager`, which scans that directory for `<slug>/SKILL.md` subfolders. Passing a
  `.md` file path is not a supported shape.
- Each entry is `Path(d).resolve()`d and **existence-checked**. A path that does not exist
  at construction is silently dropped (`if Path(d).resolve().exists()`); this is symmetric
  to `extra_tools`.
- `_extra_skills_dirs` is a **plain Python list**, so `.append(Path(root).resolve())` works
  after construction — the same mechanism as `_extra_tool_files`. It is re-read on every
  `refresh_tools_and_skills()` call; entries that no longer exist are filtered out again at
  refresh time.
- `extra_skills_dirs` and `extra_tools` are **independent**. A persona can hook a skills
  root (editable `SKILL.md`) *and* the `skill_maker` toolset simultaneously.

---

## 6. Step 4 — (Optional) enable the RSI act phase

`skill_maker` gates its heavy *act-phase* tools (`tool_commit_skill`,
`tool_run_skill_test_agent`) behind a session flag. Verified from `_skillmaker_flag_is_on`:

```python
def _skillmaker_flag_is_on(binding: Any) -> bool:
    try:
        flags = getattr(binding, "session_flags", {}) or {}
        return bool(flags.get("enable_skillmaker"))
    except Exception:
        return False
```

- **Flag name: `enable_skillmaker`.**
- When the flag is absent, `LCPBinding.discover_tools()` **hides** `tool_commit_skill`
  and `tool_run_skill_test_agent` (see `skill_maker/description.yaml`). The lighter
  `tool_read_skill_source` / `tool_write_skill_source` / `tool_test_skill_source` /
  revert / discard tools remain callable.

Enable it on the current binding:

```python
editor._tool_binding.set_session_flags({"enable_skillmaker": True})
```

> Session flags live on the binding object. `refresh_tools_and_skills()` **replaces**
> `_tool_binding` on every call — see pitfall #1 in §9 — so if you rely on the flag,
> re-apply it after every refresh.

Additionally, every `skill_maker` tool declares `GATED_TASK_CATEGORY = "code_execution"`,
which routes each call through the host's confirmation modal. A headless script that wants
these tools to run without an interactive operator must install a confirm handler:

```python
from lollms_client.tools_bindings.lcp.default_tools.skill_maker import TOOLS as skill_maker_tools
skill_maker_tools.set_confirm_handler(lambda req: True)
```

In an interactive `chat()` / GUI session the host already handles this modal.

---

## 7. Step 5 — The edit loop

### 7.1 — Inspect (`tool_list_skills` + `tool_load_skill`)

These two tools are provided by `SkillsManager.build_skill_tools()` and mounted by the
persona. Drive them via chat:

```python
editor.chat(
    prompt=(
        "List all skills currently registered and load the one named "
        "'CSV Inspector' into your context. Do not edit anything yet."
    ),
    enable_workspace_tools=False,
)
```

Under the hood, `tool_list_skills` groups by visibility tier, and `tool_load_skill`
promotes the requested skill into the `=== ACTIVE SKILLS ===` section of the prompt (it
also calls `SkillsManager.set_skill_visibility` to persist the promotion when the skill is
writable).

### 7.2 — Read the raw source (`tool_read_skill_source`)

`tool_read_skill_source` is provided by `skill_maker` and is strictly scoped to
registered skills — it resolves slugs through the personality's `skills_manager` and only
returns `*/SKILL.md` files under a registered root:

```python
result = editor._tool_binding.execute_tool(
    tool_name="tool_read_skill_source",
    params={"skill_name_or_path": "csv_inspector"},
)
# result["source"]  — full SKILL.md text
# result["path"]    — absolute path
# result["line_count"], result["size_chars"] — stats
```

### 7.3 — Edit (`tool_write_skill_source`)

Verified signature (`skill_maker/TOOLS.py`):

```python
def tool_write_skill_source(
    skill_name_or_path: str,
    new_source: str,
    reason: str = "",
    env: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
```

On success the return dict contains:

```python
{
  "success": True,
  "slug": "csv_inspector",
  "path": "/abs/path/.../csv_inspector/SKILL.md",
  "mode": "full",            # or "patch"
  "line_count": 42,
  "reason": "fix read step", # or "(no reason provided)"
  "message": "Skill 'csv_inspector' written (full).",
}
```

On validation / patch failure:

```python
{"success": False, "slug": "csv_inspector", "mode": "patch", "error": "..."}
```

On a top-level exception: `{"success": False, "error": "Failed to write skill source: ..."}`
The `_HALT_REQUESTED` guard at the top also returns `{"success": False, "error": "..."}`.

Key behaviors that matter for the editor loop:

- **Validated before writing.** `_validate_skill_body` parses the frontmatter and requires
  `name`, `description`, and a non-trivial body. A rejected candidate never touches disk.
- **Aider SEARCH/REPLACE support.** Pass a block containing `<<<<<<< SEARCH` and
  `=======`; each hunk must match verbatim; a partial patch is refused. `new_source` may
  otherwise be a full-file replacement.
- **Auto-snapshot.** Before writing, the tool copies the current content to
  `<skill_dir>/.rsi_backup/<SKILL.md>.<epoch>.bak` (i.e.
  `skills/<slug>/.rsi_backup/SKILL.md.<epoch>.bak`). That folder is the on-disk revert
  source of truth even with no git (git is opportunistic).

Example patch call:

```python
patch = """\
<<<<<<< SEARCH
        1. Read the file.
=======
        1. Read the file only, never modify it.
>>>>>>> REPLACE"""

result = editor._tool_binding.execute_tool(
    tool_name="tool_write_skill_source",
    params={
        "skill_name_or_path": "csv_inspector",
        "new_source": patch,
        "reason": "clarify read-only instruction",
    },
)
```

### 7.4 — Reload — `refresh_tools_and_skills()` (no arguments)

Exactly like the tool-editor path: `LollmsPersonality` exposes the **same**
`refresh_tools_and_skills()` method, and it is the **only** refresh entry point. There is
**no separate `refresh_skills()`**.

Verified body of the relevant portion:

```python
def refresh_tools_and_skills(self) -> None:
    if not self.handbag_path or not self.handbag_path.exists():
        ASCIIColors.warning(f"[{self.name}] Cannot refresh tools/skills: handbag path not found.")
        return

    # 1. Tools — rebuild the LCPBinding from native + extra tool files.
    hb = Handbag(self.handbag_path)
    native_tool_files = hb.tool_files
    extra_tool_files  = [f for f in getattr(self, "_extra_tool_files", []) if f.exists()]
    all_tool_files    = native_tool_files + extra_tool_files
    if all_tool_files:
        self._tool_binding = LCPBinding(tool_files=[str(f) for f in all_tool_files])
    else:
        self._tool_binding = _NULL_TOOL_BINDING

    # 2. Skills — re-scan native + extra skills dirs.
    native_skills_dirs = hb.skills_dirs
    extra_skills_dirs  = [d for d in getattr(self, "_extra_skills_dirs", []) if d.exists()]
    all_skills_dirs    = native_skills_dirs + extra_skills_dirs

    if all_skills_dirs:
        if self.skills_manager:
            self.skills_manager._skills_dirs = all_skills_dirs
            self.skills_manager.reload()                # ← REUSES existing manager
        else:
            mode = "loadable"
            if getattr(self, "capabilities", None):
                mode = getattr(self.capabilities, "skills_mode", "loadable")
            self.skills_manager = SkillsManager(all_skills_dirs, mode=mode)   # ← NEW manager
```

Consequences you must design around — **read the two branches carefully**:

- **Manager reused (normal case).** When `self.skills_manager` already exists, the method
  rewrites `_skills_dirs` and calls `reload()`. Any per-manager configuration you set
  earlier — `allow_llm_skill_writing=False`, `max_visible_tokens`, `skills_output_path` —
  **survives** the refresh because the object is kept.
- **New manager (edge case).** When `self.skills_manager` is `None`, a fresh
  `SkillsManager(skills_dirs, mode=mode)` is constructed. That path **does not** carry
  through `skills_output_path` or `allow_llm_skill_writing` — set those manually before
  the first refresh if you need them (see pitfall #7).
- **Hot-reload is "re-scan every registered skills root," not "re-read the one file I just
  edited."** The edited skill is picked up on the next `_scan_directory` pass.

Standard re-hook recipe:

```python
# (Only needed if you are adding a NEW skills root — an existing root is already in
#  _extra_skills_dirs.)
from pathlib import Path
editor._extra_skills_dirs.append(Path(skills_root).resolve())

editor.refresh_tools_and_skills()
```

For an edit of a skill **inside an already-hooked root**, no `.append(...)` is needed —
the root is on disk and `_skills_dirs` is unchanged; `refresh_tools_and_skills()` alone
picks up the new `SKILL.md` content.

### 7.5 — Re-apply the RSI flag (if you used it)

Because `refresh_tools_and_skills()` **rebuilds `_tool_binding` from scratch**, any
`set_session_flags` you made is lost on refresh. Re-apply:

```python
if hasattr(editor._tool_binding, "set_session_flags"):
    editor._tool_binding.set_session_flags({"enable_skillmaker": True})
```

If your host has a factory hook for `LCPBinding`, install the flag there instead so you
don't have to remember.

### 7.6 — Validate (`tool_test_skill_source` + reload)

```python
res = editor._tool_binding.execute_tool(
    tool_name="tool_test_skill_source",
    params={"skill_name_or_path": "csv_inspector"},
)
# res == {
#   "success": True,
#   "slug": "csv_inspector",
#   "path": "/abs/.../SKILL.md",
#   "frontmatter_ok": True,
#   "structure_ok": True,
#   "declared_tools": [],
#   "error": None,
# }
```

Follow with a reload-to-observe check: after `tool_load_skill` (or re-loading the persona
prompt), the newly written body should surface. Diff the prompt before/after if you need a
hard PASS/FAIL.

---

## 8. `skill_maker` reference

`skill_maker` is the RSI toolset that mirrors `tool_maker` for `SKILL.md` files.

### 8.1 — Module constants and gating (verified)

```python
TOOL_LIBRARY_NAME = "skill_maker"
TOOL_LIBRARY_DESCRIPTION = (
    "RSI toolset for skills: read, write, test, revert, discard and commit "
    "SKILL.md files. RSI act-phase requires enable_skillmaker=True."
)
GATED_TASK_CATEGORY = "code_execution"
```

- **`GATED_TASK_CATEGORY = "code_execution"`** routes each call through the host's
  confirmation modal. A headless script must install a confirm handler
  (`skill_maker_tools.set_confirm_handler(...)`) — see §6.
- **`_skillmaker_flag_is_on`** (verbatim, reproduced above) is the flag check.
  `_require_skillmaker_flag` is the hard gate that blocks RSI act-phase tools when the
  flag is off.

### 8.2 — Tool surface

| Tool | Purpose |
|------|---------|
| `tool_read_skill_source(skill_name_or_path)` | Read the raw `SKILL.md`. |
| `tool_write_skill_source(skill_name_or_path, new_source, reason="")` | Full replacement or Aider SEARCH/REPLACE patch; auto-snapshots before write. |
| `tool_test_skill_source(skill_name_or_path)` | Parse frontmatter + validate shape. No execution. |
| `tool_run_skill_test_agent(...)` | RSI act-phase only. Spawns a constrained child agent. |
| `tool_list_skill_versions(skill_name_or_path)` | List revert targets (backups + release manifests + git). |
| `tool_revert_skill(skill_name_or_path, backup_index=0)` | Restore a prior snapshot; validates post-restore. |
| `tool_discard_skill_changes(skill_name_or_path)` | Reset to last committed state. |
| `tool_commit_skill(skill_name_or_path, commit_message="")` | Write release manifest; opportunistically git-commit; remount the skill. |

### 8.3 — `tool_write_skill_source` full return contract

- **Success**:
  `{"success": True, "slug": str, "path": str, "mode": "full"|"patch", "line_count": int, "reason": str, "message": str}`
- **Failure (validation or patch)**:
  `{"success": False, "slug": str, "mode": "full"|"patch", "error": str}`
- **Failure (top-level exception)**:
  `{"success": False, "error": str}`

### 8.4 — Path scoping (`_resolve_skill_file`)

Every read/write path is resolved by `_resolve_skill_file`, which:

1. Looks up the skill by slug or title in the personality's `skills_manager`.
2. Rejects paths whose basename is not `SKILL.md`.
3. Rejects paths not under a registered skills root.
4. Falls back to scanning `.lollms_client/skills` and `.lollms_code/skills` when no
   personality is attached (CLI convenience).

---

## 9. Pitfalls (verified)

1. **`refresh_tools_and_skills()` rebuilds the tool binding.** After a refresh,
   `editor._tool_binding` is a **new** `LCPBinding`, so any `session_flags` set on the old
   binding (e.g. `enable_skillmaker=True`) are lost. Re-apply them after every refresh,
   or install them via a factory hook.
2. **`modifiable:` absent = read-only (safe default).** `SkillsManager._reconcile_skill_metadata`
   migrates the frontmatter on load by *injecting* `modifiable: false` (it never rewrites
   the body). `update_skill` / `tool_update_skill` refuse in-place writes on such files.
   Editor personas must ensure the target `SKILL.md` declares `modifiable: true` BEFORE
   any edit, or writes are refused.
3. **`SkillsManager.__init__` mkdirs every `skills_dirs` entry** (`exist_ok=True`). A
   typo in a skills root silently becomes an empty directory the agent now treats as a
   valid skills root. Verify `Path(root).is_dir()` before passing.
4. **`skill_maker` tools are gated as `GATED_TASK_CATEGORY = "code_execution"`.** Headless
   runs need `set_confirm_handler(...)` (module-level, imported from
   `skill_maker/TOOLS.py`). Interactive `chat()` / GUI sessions auto-handle this.
5. **RSI ACT-phase requires `enable_skillmaker=True`.** Without it,
   `LCPBinding.discover_tools()` **hides** `tool_commit_skill` and
   `tool_run_skill_test_agent`. Non-RSI tools remain — but a persona that expects to
   "commit a release" will silently fail to find those tools.
6. **`extra_skills_dirs` entries that do not exist at construction are silently dropped**
   (`if Path(d).resolve().exists()`), and the refresh call filters them again on every
   pass (`[d for d in getattr(self, "_extra_skills_dirs", []) if d.exists()]`). A typo
   produces no error — only a skill root that is never scanned.
7. **`refresh_tools_and_skills()` does not preserve custom `SkillsManager` config when
   constructing a new manager.** If `self.skills_manager` was `None` at refresh time, the
   method calls `SkillsManager(skills_dirs=all_skills_dirs, mode=mode)` — dropping
   `skills_output_path`, `allow_llm_skill_writing`, and `max_visible_tokens` you may have
   wanted. To keep them, set
   `editor.skills_manager = SkillsManager(..., skills_output_path=..., allow_llm_skill_writing=...)`
   manually before the first refresh; the refresh will then **reuse** that manager.
8. **`refresh_tools_and_skills()` requires the handbag folder to still exist.** If it is
   deleted between construction and refresh, the call warns and returns without touching
   either tools or skills.

### Do NOT

- **Do NOT call `refresh_skills()`** — it does not exist. `refresh_tools_and_skills()` is
  the single refresh entry point, and it handles both tools and skills in one call.
- **Do NOT pass individual `.md` files to `extra_skills_dirs`** — it accepts **directories**
  (each scanned for `<slug>/SKILL.md`). Individual-file hooks are not supported.
- **Do NOT use `LCPBinding(tools_folders=...)`** — the constructor kwarg used by
  `from_handbag` / `refresh_tools_and_skills` is `tool_files=[...]` (this is the same
  warning as `docs/TOOL_EDITOR_AGENT.md` §9; it applies symmetrically here).

---

## 10. Minimal end-to-end example

Builds a handbag, writes a sample `SKILL.md` with `modifiable: true`, constructs the
editor with both `extra_skills_dirs` and `extra_tools=[skill_maker_dir]`, edits the skill
on disk, calls `refresh_tools_and_skills()`, and re-applies the RSI flag.

```python
"""skill_editor_demo.py — minimal runnable skills-editor round trip.

Run:
    python skill_editor_demo.py
"""
from pathlib import Path
import textwrap

from lollms_client.lollms_personality.lollms_personality import LollmsPersonality


ROOT      = Path("skills_editor_demo").resolve()
HANDBAG   = ROOT / "skills_editor_handbag"
WORKSPACE = ROOT / "skills_workspace"
SKILLS    = WORKSPACE / "skills"
TARGET    = SKILLS / "csv_inspector" / "SKILL.md"


def _write_handbag() -> None:
    HANDBAG.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    (HANDBAG / "SOUL.md").write_text(textwrap.dedent("""\
        ---
        name: SkillSmith
        author: lollms-client
        version: '1.0'
        category: engineering
        description: Focused editor for a single SKILL.md skill.
        skills_mode: loadable
        ---

        You are SkillSmith. Your ONLY job is to inspect, edit, and validate one SKILL.md.

        Mandatory loop: READ -> UNDERSTAND -> EDIT -> RELOAD -> VALIDATE.
        Never edit a SKILL.md that lacks `modifiable: true`.
        Never touch files outside the attached skills root.
        Never claim an edit is hot-loaded before refresh_tools_and_skills.
    """), encoding="utf-8")

    (HANDBAG / "handbag.yaml").write_text(textwrap.dedent("""\
        name: Skills Editor Handbag
        version: '1.0'
        description: Handbag for SKILL.md-editing personas.
        skills_mode: loadable
    """), encoding="utf-8")


def _write_target_skill() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(textwrap.dedent("""\
        ---
        name: CSV Inspector
        description: Inspect a CSV file and report its structure.
        category: data
        tags: [csv, data, inspection]
        visibility: loadable
        modifiable: true
        ---

        # CSV Inspector

        ## Steps
        1. Read the CSV file.
        2. Report row count and column names.
    """), encoding="utf-8")


def _edit_target_skill() -> None:
    """Appends an instruction without rewriting the existing body."""
    body = TARGET.read_text(encoding="utf-8").rstrip()
    body += "\n\nIMPORTANT: never modify the source CSV in place.\n"
    TARGET.write_text(body, encoding="utf-8")


def _print_skills(personality: LollmsPersonality, label: str) -> None:
    skills = personality.skills_manager.get_unique_skills()
    print(f"\n--- {label}: {len(skills)} skill(s) ---")
    for s in skills:
        print(f"  [{'rw' if s.modifiable else 'ro'}] {s.title} ({s.file_path})")


def _skill_maker_dir() -> Path:
    import lollms_client
    return (
        Path(lollms_client.__file__).parent
        / "tools_bindings" / "lcp" / "default_tools" / "skill_maker"
    )


def main() -> None:
    _write_handbag()
    _write_target_skill()

    # Optional: real client enables the agent loop.
    # client = LollmsClient(llm_binding_name="ollama",
    #                       llm_binding_config={"model_name": "qwen2.5-coder:7b"})
    client = None

    editor = LollmsPersonality.from_handbag(
        handbag_path=HANDBAG,
        lollms_client=client,
        extra_skills_dirs=[SKILLS],
        extra_tools=[_skill_maker_dir()],
        workspace_path=WORKSPACE,
    )

    # Optional: enable RSI act-phase (commit/test-agent tools).
    if hasattr(editor._tool_binding, "set_session_flags"):
        editor._tool_binding.set_session_flags({"enable_skillmaker": True})

    _print_skills(editor, "BEFORE edit")

    # --- Edit the skill source on disk -------------------------------------
    _edit_target_skill()

    # --- Re-hook: re-scan every registered skills root ---------------------
    editor.refresh_tools_and_skills()

    # --- Re-apply RSI flag (the binding was rebuilt) -----------------------
    if hasattr(editor._tool_binding, "set_session_flags"):
        editor._tool_binding.set_session_flags({"enable_skillmaker": True})

    _print_skills(editor, "AFTER  edit + refresh")

    # If a real client is present, you can now drive the editor via chat:
    # editor.chat(
    #     prompt=(
    #         "Using tool_read_skill_source, read 'csv_inspector'. Then call "
    #         "tool_write_skill_source with a SEARCH/REPLACE patch that appends "
    #         "'never modify the source CSV' to the Steps list. Then call "
    #         "tool_test_skill_source to validate."
    #     ),
    #     enable_workspace_tools=True,
    # )


if __name__ == "__main__":
    main()
```

Notes:

- The demo edits the file directly on disk (no LLM dependency), so it can be a hard
  PASS/FAIL check for the wiring itself.
- Because there is no `refresh_skills()` method, both hook types (tools + skills) refresh
  together with one call to `refresh_tools_and_skills()`. If you were editing both a
  toolset and a skill, one call refreshes both.

---

## 11. Verified against

Source symbols and files this document was checked against:

- `LollmsPersonality.from_handbag` — `src/lollms_client/lollms_personality/lollms_personality.py`
  - `extra_skills_dirs` resolution / existence filter and `_extra_skills_dirs` persistence
    via `object.__setattr__`
  - `extra_tools` (parallel hook, used here for `skill_maker`)
- `LollmsPersonality.refresh_tools_and_skills()` — same file
  - No-argument signature; rebuilds `_tool_binding` and refreshes skills via
    `skills_manager._skills_dirs = all_skills_dirs; skills_manager.reload()`
  - Two-branch behavior: reuse existing manager vs. construct a new one from
    `capabilities.skills_mode`
- `SkillsManager` — `src/lollms_client/lollms_personality/skills_manager.py`
  - `__init__` (mkdirs each `skills_dirs`, `VALID_VISIBILITIES`, `_resolve_visibility`)
  - `reload`, `_scan_directory`, `_reconcile_skill_metadata` (the `modifiable: false`
    migration)
  - `build_skill_tools` — the in-agent tool surface (`tool_list_skills`,
    `tool_load_skill`, `tool_unload_skill`, `tool_search_skills`, `tool_create_skill`,
    `tool_update_skill`, `tool_append_to_skill`, `tool_remove_skill`)
  - `update_skill` / `append_to_skill` / `remove_skill` — refuse when `not skill.modifiable`
- `Skill` dataclass and `parse_skill_md` — `src/lollms_client/lollms_personality/skill.py`
  - `modifiable` / `modifiable_field_present` read/write contract
  - Frontmatter keys surfaced: `name`, `description`, `category`, `tags`, `visibility`,
    `modifiable` (plus optional `required_tools` / `tools:`)
- `skill_maker` toolset — `src/lollms_client/tools_bindings/lcp/default_tools/skill_maker/TOOLS.py`
  - `GATED_TASK_CATEGORY = "code_execution"`
  - `_skillmaker_flag_is_on` (checks `flags.get("enable_skillmaker")`)
  - `_require_skillmaker_flag` (blocks RSI act-phase tools when the flag is off)
  - `_resolve_skill_file` (skills-root scoping)
  - `tool_read_skill_source`, `tool_write_skill_source`, `tool_test_skill_source`,
    `tool_run_skill_test_agent`, `tool_list_skill_versions`, `tool_revert_skill`,
    `tool_discard_skill_changes`, `tool_commit_skill`
  - `tool_write_skill_source` return contract and `.rsi_backup/<SKILL.md>.<epoch>.bak`
    auto-snapshot before write
- `Handbag.skills_dirs` / `Handbag.tool_files` — invoked by `refresh_tools_and_skills`
  (see `src/lollms_client/lollms_personality/handbag.py`)
- Companion doc: `docs/TOOL_EDITOR_AGENT.md` (the same `refresh_tools_and_skills()`
  invariant, the same `tool_files=` kwarg warning, the same handbag-layout contract).