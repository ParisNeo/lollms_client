# `chat()` — Toolset Editing (Staging) Parameters

`LollmsDiscussion.chat()` exposes two parameters that mount the
**`toolset_staging`** LCP toolset and give the agent a callable
read / patch / rewrite / test loop over LCP toolsets, inside the turn.

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `enable_toolset_editing` | `bool` | `False` | Mount `toolset_staging` and admit its `tool_stage_*` tools |
| `toolset_editing_paths` | `Optional[List[str]]` | `None` | Explicit extra edit-target folders (pushed via `set_staging_paths` after mount) |

Both default to off. They are **additive**: the existing office / computer-use /
workspace toolsets and their flags are unaffected.

---

## 0. Full example — a handbag persona that edits its own tools

`examples/simple/self_editing_toolset_agent.py` is the runnable reference. The
shape is:

```python
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

agent = LollmsPersonality.from_handbag(
    handbag_path=HANDBAG_DIR,
    lollms_client=client,
    # (1) Toolset the agent MAY CALL — its own tools.
    extra_tools=[TARGET_DIR],
    # (2) Editing surface: the staging toolset + workspace (contains target).
    extra_tool_staging_paths=[STAGING_DIR, WORKSPACE_DIR],
    # (3) Workspace MUST be an ancestor of the target for rglob discovery.
    workspace_path=WORKSPACE_DIR,
)

result = agent.discussion.chat(
    user_message="Extend the `units` toolset with a temperature converter.",
    personality=agent,
    # (4) The chat parameter: mounts `toolset_staging` and admits tool_stage_*.
    enable_toolset_editing=True,
    max_nb_rounds=30,
)
```

Three invariants that MUST be satisfied for this to work:

1. **`TARGET_DIR` exists on disk when `from_handbag` runs.** Paths in
   `extra_tools` are existence-checked; non-existent entries are silently
   dropped.
2. **`STAGING_DIR` is the shipped `toolset_staging` folder** (locate it via
   `importlib.util.find_spec` — see `_resolve_toolset_staging_dir()` in the
   example). It goes in `extra_tool_staging_paths`, **never** in `extra_tools`
   (the two lists convey different intents).
3. **`WORKSPACE_DIR` contains `TARGET_DIR`**, directly or at any depth.
   `tool_stage_list_stageable` scans `workspace.rglob("TOOLS.py")`.

If the agent improvises into `execute_python` instead of staging, the wiring
is wrong. Check `tool_stage_list_stageable` — it returns a `diagnosis` field
naming the exact cause (missing `env`, lost sub-agent binding, single-file
toolset).

---

## 0.1 Toolset self-protection (`TOOL_LIBRARY_READONLY`)

A toolset can **decline editing** by declaring a top-level constant in its
`TOOLS.py`:

```python
TOOL_LIBRARY_READONLY = True
```

Enforcement points:

| Where | Behavior |
|---|---|
| `tool_stage_list_stageable` | candidate is listed with `protected: true`; also surfaced under `protected: [...]`; `editable_count` excludes it |
| `tool_stage(name=...)` | hard refusal — returns `success: False, protected: True` with an explanatory error |
| `tool_stage_patch` | re-checks the flag **at write time** (defends against a protection added after staging); refuses if present |
| `tool_stage_rewrite` | same write-time re-check |

**Why it exists**: LCP default toolset authors (and any library author) may
want their toolset to be inspectable but not mutatable by the agent. Silent
agent-authored edits to a widely-used toolset are hard to review and easy to
break. This flag is an **author intent declaration**, not a sandbox boundary
— a tool with filesystem access can bypass it. Its purpose is to prevent
well-intentioned but destructive edits, not to defeat an adversarial tool.

Example file:

```python
"""locked_units — read-only demo toolset."""

TOOL_LIBRARY_NAME = "Locked Units"
TOOL_LIBRARY_READONLY = True   # <-- protected

def init_tools_library(config=None): ...

def tool_inch_to_cm(inches: float) -> dict:
    ...
```

When the agent tries `tool_stage(name="locked_units")`, it receives:

```json
{
  "success": false,
  "protected": true,
  "name": "locked_units",
  "error": "'locked_units' is READ-ONLY. Its TOOLS.py declares `TOOL_LIBRARY_READONLY = True`, so it is protected against agent-driven edits. ..."
}
```

The doctrine block injected into the system prompt tells the model to report
the refusal rather than attempt to bypass it.

---

## 1. Minimal usage

```python
result = discussion.chat(
    user_message="Add a `tool_count_rows` function to the `csv_utils` toolset.",
    personality=agent,
    enable_toolset_editing=True,          # mount toolset_staging
    toolset_editing_paths=["../../tools"],  # optional extra edit surfaces
    max_nb_rounds=30,
)
```

When `enable_toolset_editing=True` succeeds, the agent sees, in addition to its
own tools:

| Tool | Job |
|---|---|
| `tool_stage_list_stageable` | enumerate candidates (`source`: workspace / staging / external) + `diagnosis` when count is 0 |
| `tool_stage` | stage exactly ONE toolset (single focus slot; multi-staging refused) |
| `tool_stage_read` | return the current `TOOLS.py` source directly (no file tool needed) |
| `tool_stage_patch` | apply Aider `SEARCH/REPLACE` blocks ATOMICALLY |
| `tool_stage_rewrite` | full-file `TOOLS.py` replacement |
| `tool_stage_test` | run a `tool_*` from the on-disk source to verify |
| `tool_stage_unstage` | release the focus slot (file stays on disk) |
| `tool_stage_status` | non-mutating session view |

The agent's system prompt receives a dedicated **TOOLSET EDITING PROTOCOL**
block **only when at least one `tool_stage_*` tool was actually admitted**.
When the mount produces 0 tools, the doctrine is withheld (fail-closed
instruction gating, mirroring the office-mount contract) and a single
`[ChatMixin] ... produced 0 tool specs` diagnostic is logged.

---

## 2. Where the target toolset must live

The staging toolset finds editable targets from three sources, in order:

1. **workspace** — `workspace.rglob("TOOLS.py")` under the discussion's
   `workspace_data_path`.
2. **staging** — each folder passed in `toolset_editing_paths` (this file's
   terminal purpose; a declared path may be a toolset folder OR a parent whose
   immediate children are toolset folders).
3. **external** — toolsets mounted on the live LCP binding via
   `extra_tools=[...]` on the personality, whose folder lies OUTSIDE the
   workspace.

So a target is stageable if **any** of the following is true:

- its folder is an ancestor-or-descendant of the workspace root, OR
- its folder (or its parent) is listed in `toolset_editing_paths`, OR
- it is mounted on the personality's LCP binding.

`workspace_path` must be an **ancestor** of a workspace-resident target, never
the toolset folder itself — pointing it at the toolset makes the `rglob` scan
find zero candidates.

---

## 3. The one-owner rule (sub-agent trap)

A **sub-agent** is a fresh process: its `env` does not carry the parent's
`client.tools` binding, so `_list_external_toolsets` returns `[]` inside it.
`tool_stage_list_stageable` then reports `count: 0` **in the sub-agent** even
though the parent sees the toolset fine — and the `diagnosis` field names
exactly this cause.

Do the editing on the agent that owns the binding. If a sub-agent genuinely
must edit, its target has to live under **that sub-agent's** `workspace_path`
so the filesystem scan finds it without the parent binding.

---

## 4. Folder-layout only

`toolset_staging` edits **folder-layout** toolsets — `<name>/TOOLS.py`.
Single-file toolsets (`<name>.py` not named `TOOLS.py`) are not enumerated and
cannot be staged. This is by design: the atomic SEARCH/REPLACE + `ast.parse`
validation loop assumes a canonical entry point.

---

## 5. Prompt refresh — no reload needed

Once a toolset is staged, `LollmsPersonality._build_staged_toolsets_context`
calls `toolset_staging_state.build_staged_prompt_block(workspace)` **each
turn**. That function reads `TOOLS.py` from disk (the staging registry stores
only identity: name, folder, timestamp). Therefore:

- a `tool_stage_patch` write appears in the model's context on the **next**
  round with no reload call;
- re-staging is never needed to see an edit.

---

## 6. Validation gate

Both `tool_stage_patch` and `tool_stage_rewrite`:

- run the candidate source through `ast.parse`,
- require at least one top-level `def tool_*(...)`,
- for `patch`, apply all SEARCH/REPLACE blocks **atomically** — an ambiguous
  or missing SEARCH block aborts the entire write.

A broken source never reaches disk.

---

## 7. Interaction with other flags

| Flag | Effect when combined |
|---|---|
| `enable_python_exec` / `enable_code_execution` | Independent. Staging reads/patch source directly; execute_python is for running code, not editing toolsets. |
| `enable_workspace_tools` | Independent. Staging has its own `tool_stage_read`; the workspace read tools are not required. |
| `compact_workspace_tree` | Independent. Staging is unaffected by tree compaction. |
| `allow_dynamic_tools` | Independent. `allow_dynamic_tools` gates `<artifact type="tool">` registration; staging gates `tool_stage_*` authorship. |
| `orchestrator_persona=True` | **Staging is disabled** — the orchestrator persona has no executable tools by design, so the mount helper is skipped and no doctrine is injected. |

---

## 8. Failure modes & diagnostics

| Symptom | Cause | Fix |
|---|---|---|
| No `tool_stage_*` tools appear | LCP binding absent, or the `toolset_staging` library is not discoverable | Configure `create_tools=True`; check `default_tools/toolset_staging/TOOLS.py` exists |
| `'toolset_staging' mounted but produced 0 tool specs` | Fail-closed session-flag gate refused the mount | Check that a caller did not clear the flag after `chat()` set it; the mount helper sets both `tools_binding_config` and `host_tool_configs` before mount |
| `tool_stage_list_stageable` returns `count: 0` | See the `diagnosis` field: no env handle / sub-agent lost the binding / single-file toolset | Run on the owning agent; use a folder-layout target |
| `tool_editing_paths requested but 'set_staging_paths' could not be reached` | Binding has no `_get_or_load_tool_module` seam | Workspace-resident toolsets still work; out-of-workspace declared targets may not be stageable |
| Second `tool_stage` refused | Multi-staging is refused by design | Unstage the current toolset first |
| `tool_stage` returns `protected: true` | Toolset declares `TOOL_LIBRARY_READONLY = True` | This is intentional; the protection must be removed from source by a human operator |

---

## 8.1 Skill editing (`enable_skill_editing` + `skill_editing_paths`)

A parallel pair of parameters adds **skill editing** to the same chat call.
This is the SKILL.md analogue of toolset editing — but it is deliberately
simpler, because skills do not need a staging registry when the agent already
holds them.

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `enable_skill_editing` | `bool` | `False` | Mount the `skill_editor` toolset and admit its `tool_edit_skill_*` tools |
| `skill_editing_paths` | `Optional[List[str]]` | `None` | Explicit extra skill folder(s) or parent(s) with skill folders as immediate children |

### 8.1.1 Two editing modes

| Mode | When | How |
|---|---|---|
| **Direct** (skills the agent already holds) | `enable_skill_editing=True`, no explicit path or activation | The `skill_editor`'s `_iter_manager_skills(env)` exposes every skill registered on the agent's `skills_manager`. When exactly ONE candidate exists, the editor's tools **auto-activate** against it — so `tool_edit_skill_read` / `_patch` / `_rewrite` work on the held skill with a single call. The skill must still be editable (`modifiable: true`), see §8.1.3. |
| **Staged** (skills anywhere you declare) | `enable_skill_editing=True`, `skill_editing_paths=[...]` | Each declared folder is a valid target. Pass either the skill folder itself (contains `SKILL.md`) or a parent whose immediate children are skill folders. |

The direct mode is what makes "the LLM can still edit the skills it has" work
without any staging scaffold: if the personality has skills wired via
`skills_manager` (from `from_handbag`'s `skills/` folder, `extra_skills_dirs`,
or builtin skills), those skills flow into `tool_edit_skill_list` and are
auto-activatable. **They are also subject to the canonical
`modifiable: false` default** — the agent can *read* them, but editing requires
the skill to explicitly opt in. See §8.1.3.

The declared-path mode is the analogue of `toolset_editing_paths`: it lets the
agent edit skills that are **not** on its manager but whose folder lives in a
known location (e.g. a shared skill library the host mounts read-edit).

### 8.1.2 Workflow

```text
tool_edit_skill_list          -> confirm the target skill, read `diagnosis`
                                 if `count == 0`
tool_edit_skill_activate      -> OPTIONAL — auto-activates on the sole candidate
tool_edit_skill_read          -> SKILL.md source (frontmatter + body)
tool_edit_skill_patch         -> atomic Aider SEARCH/REPLACE blocks
tool_edit_skill_rewrite       -> full-file replacement
tool_edit_skill_test          -> exercise the edited skill
tool_edit_skill_close         -> release the session; file stays on disk
```

### 8.1.3 Self-protection — the canonical default is read-only

A skill declines agent editing by declaring either of two accepted flags.
**Both are honoured identically**, both are re-checked at write time, and the
**canonical default is protected** (a skill that omits `modifiable:` is
treated as `modifiable: false`).

**Flag 1 (canonical) — `modifiable: false` in the frontmatter:**

```yaml
---
name: pii_guard
description: A hardened skill that no agent may edit.
modifiable: false                     # <-- protected
---

... body ...
```

This is the flag `SkillsManager` already publishes: it is the gate used by
`update_skill`, `append_to_skill`, `remove_skill`, and `set_skill_visibility`,
and it is the **default when the key is absent** (`_reconcile_skill_metadata`
injects `modifiable: false` on first load of a frontmatter skill that omitted
it). `skill_editor` writes directly to disk and therefore bypasses
`SkillsManager` — the write-time re-check in the editors closes that gap,
never requiring the agent to route through the manager.

**Flag 2 (explicit, toolset-style) — `SKILL_LIBRARY_READONLY`:**

```yaml
---
name: pii_guard
description: A hardened skill that no agent may edit.
SKILL_LIBRARY_READONLY: true          # frontmatter form
---
```

```markdown
...
SKILL_LIBRARY_READONLY = True         # body constant form
```

This mirrors `TOOL_LIBRARY_READONLY` in `toolset_staging` — an unmistakable,
toolset-style marker for authors who prefer it. Either form is accepted.

Both flags (and the absent-`modifiable:` default) are detected by the single
helper `_is_skill_protected`, defined at
`src/lollms_client/tools_bindings/lcp/default_tools/skill_editor/TOOLS.py:720`.
It scans the frontmatter for `SKILL_LIBRARY_READONLY: true`, the body for the
`SKILL_LIBRARY_READONLY = True` constant, and the frontmatter for
`modifiable: <truthy>` — treating an absent `modifiable:` key as protected
(the canonical SkillsManager default). Enforcement points:

| Where | Behavior |
|---|---|
| `tool_edit_skill_list` | candidate is listed with `protected: true`; surfaced under `protected: [...]`; `editable_count` excludes it |
| `tool_edit_skill_activate` | hard refusal — returns `success: False, protected: True` |
| `tool_edit_skill_patch` | **write-time** re-check (defends against protection added after activation); refuses if present |
| `tool_edit_skill_rewrite` | same write-time re-check |

**Why it exists**: skill authors (LCP defaults, app-embedded skills) may want
a skill to be inspectable but immutable by the agent. Precisely like
`TOOL_LIBRARY_READONLY`, this is an **author intent declaration**, not a
sandbox boundary — a tool with filesystem access can bypass it. Its purpose is
to prevent well-intentioned but destructive agent edits, not to defeat an
adversarial tool.

**The default is protected, not editable.** `SkillsManager` publishes
`modifiable: false` as the default on first load of a frontmatter skill that
omits the key (`_reconcile_skill_metadata`). `skill_editor` writes directly to
disk, bypassing the manager — but it re-reads the on-disk frontmatter at every
activate/patch/rewrite and honours the same default. So exposing a skill as
editable requires **explicit opt-in** (`modifiable: true`).

### 8.1.4 Interaction with `enable_toolset_editing`

Both parameters are **independent** and may be combined:

```python
result = discussion.chat(
    user_message="Add a helper to the `units` toolset AND teach the `unit_qa` "
                 "skill about it.",
    personality=agent,
    enable_toolset_editing=True,     # tool_stage_* admitted
    enable_skill_editing=True,       # tool_edit_skill_* admitted
    max_nb_rounds=40,
)
```

Skills and toolsets are disjoint here: a skill is a prompt fragment; a toolset
is a callable Python library. The agent sees both families and can update both
within a single chat turn because their prompt blocks and their write paths do
not collide.

### 8.1.5 Wiring an agent that edits BOTH

```python
agent = LollmsPersonality.from_handbag(
    handbag_path=HANDBAG_DIR,
    lollms_client=client,
    extra_tools=[TARGET_TOOLSET],
    extra_tool_staging_paths=[TOOLSET_STAGING_DIR, WORKSPACE_DIR],
    extra_skills_dirs=[SHARED_SKILLS_DIR],   # mount skills the agent holds
    workspace_path=WORKSPACE_DIR,
)
```

The agent's `skills/` folder plus anything in `extra_skills_dirs` flows into
`skills_manager`. When chat() is called with `enable_skill_editing=True`, those
skills become the auto-activation candidates — the agent can update any of them
by name, subject to `SKILL_LIBRARY_READONLY`.

### 8.1.6 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| No `tool_edit_skill_*` tools appear | LCP binding absent, or `skill_editor` is not discoverable | Configure `create_tools=True`; check `default_tools/skill_editor/TOOLS.py` exists |
| `tool_edit_skill_list` returns `count: 0` | The agent holds no skills AND no paths were declared | Mount skills via `extra_skills_dirs`, wire a handbag `skills/` folder, or declare `skill_editing_paths` |
| `tool_edit_skill_activate` returns `protected: true` | The skill's SKILL.md declares `modifiable: false` OR `SKILL_LIBRARY_READONLY`, or its frontmatter omits `modifiable:` (canonical read-only default) | Intentional; opt in by setting `modifiable: true` in the frontmatter (or, for app-controlled skills, adjust how the skill is registered) |
| `tool_edit_skill_patch` returns `protected: true` | The protection was added between activate and patch | Same as above — write-time re-check is designed to catch this |
| Auto-activation refuses with "ambiguous" | Two or more editable candidates exist and no activation was performed | Call `tool_edit_skill_activate(skill_name="<target>")` explicitly |

---

## 9. Related documents

- [`docs/TOOLSET_STAGING_AND_SANDBOX.md`](./TOOLSET_STAGING_AND_SANDBOX.md) —
  the authoritative operator + author guide for the staging toolset and the
  `execute_python` sandbox.
- [`docs/lcp/10_TOOLSET_AUTHORING_GUIDE.md`](./lcp/10_TOOLSET_AUTHORING_GUIDE.md) —
  how to author the `TOOLS.py` the agent will edit.
- [`docs/TOOL_EDITOR_AGENT.md`](./TOOL_EDITOR_AGENT.md) — architectural
  rationale (legacy tool names; superseded by the staging toolset).