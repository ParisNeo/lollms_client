# `enable_skill_editing`

```python
enable_skill_editing: bool = False
# companion: skill_editing_paths: Optional[List[str]] = None
```

**Category**: Tools & Security
**Parent flags**: none
**Gate**: requires a live LCP tools binding (refused with a diagnostic when
none is configured)

## What it does

Mounts the `skill_editor` LCP library and admits its `tool_edit_skill_*` tools.
This is the **callable** skill-editing surface: the agent can inspect, patch,
validate, and reload `SKILL.md` files *inside the turn* — including skills it
already holds.

The admitted tool family:

| Tool | Purpose |
|---|---|
| `tool_edit_skill_list` | Enumerate editable skills (each entry carries a `source` and, when count is 0, a `diagnosis` string) |
| `tool_edit_skill_activate` | Explicitly focus one skill by name (optional when only one candidate exists — see auto-activation below) |
| `tool_edit_skill_read` | Return the current `SKILL.md` source |
| `tool_edit_skill_patch` | Apply one or more SEARCH/REPLACE blocks atomically (frontmatter + body validated) |
| `tool_edit_skill_rewrite` | Full-file replacement with the same validation |
| `tool_edit_skill_reload` | Re-read the file from disk |
| `tool_edit_skill_test` | Exercise the skill against a scripted input |
| `tool_edit_skill_load_status` | Report the load state |
| `tool_edit_skill_close` | Release the session (file stays on disk) |

## What to expect

- **Auto-activation**: when exactly ONE editable skill is a candidate, every
  editing tool operates on it WITHOUT an explicit activate call. This is the
  canonical path for an agent to update a skill it already holds: the
  `skills_manager` exposes the held skill as the sole candidate, and the agent
  can patch it directly with `tool_edit_skill_patch`.
- **Read-only protection**: a skill can protect itself via:
  - `modifiable: false` (frontmatter — the canonical `SkillsManager` flag;
    also the default when the key is absent), OR
  - `SKILL_LIBRARY_READONLY: true` (frontmatter), OR
  - `SKILL_LIBRARY_READONLY = True` (body).
  Activate/patch/rewrite refusals are surfaced in the processing block; the
  agent must report them, never bypass them.
- **All paths validated once per turn** — same contract as
  `toolset_editing_paths`: `Path.expanduser().resolve()`, directory-checked,
  deduplicated. Invalid entries are warned and skipped.
- **`skill_editing_paths` is only consulted when `enable_skill_editing=True`.**
- **Refused without an LCP binding**: logs an actionable diagnostic; the turn
  proceeds with whatever tools it has.
- **Validation**: patch/rewrite validate frontmatter and body — a broken
  `SKILL.md` never reaches disk. Malformed YAML frontmatter, missing required
  keys, or invalid markdown structures are rejected atomically.
- **Post-condition check**: the mount verifies `tool_edit_skill_read` is
  admitted (a READ tool must exist before any write). When absent, the mount is
  refused with an actionable diagnostic naming the cause.

## Gate

The mount requires a live LCP tools binding (`client.tools` with
`mount_tool_library_if_absent`). When absent:

```
[ChatMixin] enable_skill_editing=True but no LCP tools binding is
available — skill_editor NOT mounted. Configure a tools binding
(e.g. 'lcp') via create_tools=True to enable it.
```

The turn proceeds with whatever tools it already has. No exception is raised.

The fail-closed `enable_skill_editing` session flag is pre-authorized by the
chat layer via `tools_binding_config` + `host_tool_configs[stem]` +
`set_session_flags` before the mount, so the toolset is guaranteed to be
discovered in normal operation.

## Usage

```python
# Update a skill the agent already holds (no explicit activate needed)
result = discussion.chat(
    "Update the data_analysis skill: mention the new parquet helpers",
    enable_skill_editing=True,
)
# The agent calls:
#   tool_edit_skill_list → tool_edit_skill_read → tool_edit_skill_patch
#   → tool_edit_skill_test → tool_edit_skill_close
# The edited SKILL.md lives on disk in the skill folder.

# Edit an external skill declared as a path
result = discussion.chat(
    "Fix a bug in the bibliography skill",
    enable_skill_editing=True,
    skill_editing_paths=["/path/to/external/skills"],
)
```

## Related flags

- `enable_toolset_editing` — the toolset-editing equivalent (mounts
  `toolset_staging` with `tool_stage_*`; companion `toolset_editing_paths`).
- `enable_skills` — controls whether the model can CREATE new `<skill>` tags
  during normal turns; independent from this editing flag.
- `enable_skillmaker` (kwarg-only) — gates the heavy RSI act-phase tools
  (`tool_run_skill_test_agent`, `tool_commit_skill`); independent of
  `enable_skill_editing`.

## Recovering generated stuff

- The edited `SKILL.md` lives on disk in the skill folder — the workspace and
  the `.rsi/commits/<epoch>.json` release manifests (if skillmaker is also
  enabled) are the source of truth.
- A successful mount emits a `SKILL EDITING PROTOCOL` doctrine block into the
  system prompt on the current turn.
- See [`../SKILL_EDITOR_AGENT.md`](../SKILL_EDITOR_AGENT.md) for the full
  editing lifecycle, worked examples, and rollback behavior.