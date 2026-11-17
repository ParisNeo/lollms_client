# `enable_toolset_editing`

```python
enable_toolset_editing: bool = False
# companion: toolset_editing_paths: Optional[List[str]] = None
```

**Category**: Tools & Security
**Parent flags**: none
**Gate**: requires a live LCP tools binding (refused with a diagnostic when
none is configured)

## What it does

Mounts the `toolset_staging` LCP library and admits its `tool_stage_*` tools.
This is the **callable** editing surface: the agent can inspect, patch, test,
and stage LCP toolsets *inside the turn*.

The admitted tool family:

| Tool | Purpose |
|---|---|
| `tool_stage_list_stageable` | Enumerate candidate toolsets (each entry carries a `source` and, when the count is 0, a `diagnosis` string) |
| `tool_stage` | Stage ONE toolset — the source appears in the prompt's STAGED TOOLSETS zone |
| `tool_stage_patch` | Apply one or more SEARCH/REPLACE blocks atomically (`ast.parse` + `def tool_*` guardrails) |
| `tool_stage_rewrite` | Full-file replacement with the same guardrails |
| `tool_stage_test` | Execute a `tool_*` function from the edited on-disk source |
| `tool_stage_unstage` | Release the focus slot (the file stays on disk) |
| `tool_stage_status` | Report the current staging state |

## What to expect

- **All paths validated once per turn** (`Path.expanduser().resolve()`,
  directory-checked), deduplicated, and pushed into the staging module via
  `set_staging_paths(...)` after the mount. Invalid entries (non-string, empty,
  unresolvable, not a directory) are warned and skipped — never raised.
- **`toolset_editing_paths` is only consulted when `enable_toolset_editing=True`.**
- **Refused without an LCP binding**: logs an actionable diagnostic. Configure
  a tools binding (e.g. `'lcp'`) via `create_tools=True`.
- **Only folder-layout toolsets** (`<name>/TOOLS.py`) are editable; single-file
  toolsets are not.
- **Strict focus cycle**: ONE toolset staged at a time (multi-staging is refused
  — two sources corrupt the model's file focus).
- **Atomic edits**: `tool_stage_patch` applies all hunks or none — a broken
  source never reaches disk. Each edit is validated via `ast.parse` and requires
  at least one top-level `def tool_*(` function.
- **Refresh without a reload**: the STAGED TOOLSETS prompt region is refreshed
  from disk every round — a patch is visible on the next round WITHOUT a
  re-stage.
- **Forbidden writers**: `execute_python`, `tool_write_file`, and `<artifact>`
  tags bypass the guardrails and are rejected when used against a staged
  toolset. The ONLY sanctioned writers are `tool_stage_patch` and
  `tool_stage_rewrite`.

## Gate

The mount requires a live LCP tools binding (`client.tools` with
`mount_tool_library_if_absent`). When absent:

```
[ChatMixin] enable_toolset_editing=True but no LCP tools binding is
available — toolset_staging NOT mounted. Configure a tools binding
(e.g. 'lcp') via create_tools=True to enable it.
```

The turn proceeds with whatever tools it already has. No exception is raised.

When the mount completes but produces 0 tool specs, an additional diagnostic
names the cause (usually the fail-closed `enable_toolset_editing` session flag
was refused — the chat layer pre-authorizes it via `tools_binding_config` +
`host_tool_configs[stem]` + `set_session_flags` before the mount, so this should
not occur in normal operation).

## Usage

```python
result = discussion.chat(
    "Fix the unit_converter toolset: handle negative temperatures",
    enable_toolset_editing=True,
    toolset_editing_paths=["/path/to/handbags/toolsets"],
    enable_code_execution=True,  # for tool_stage_test to run the fixed tool
)
# The agent calls:
#   tool_stage_list_stageable → tool_stage → tool_stage_patch
#   → tool_stage_test → tool_stage_unstage
# The edited TOOLS.py lives on disk in the toolset folder.
```

## Related flags

- `enable_skill_editing` — the skill-editing equivalent (mounts `skill_editor`
  with `tool_edit_skill_*`; companion `skill_editing_paths`).
- `enable_code_execution` — needed for `tool_stage_test` to actually execute
  the fixed `tool_*` function.
- `allow_dynamic_tools` — independent security gate for tool registration; does
  NOT imply toolsmith/staging and vice versa.
- `enable_toolsmith` (kwarg-only) — gates the heavy RSI act-phase tools
  (`tool_run_test_agent`, `tool_commit_toolset`); independent of
  `enable_toolset_editing`.

## Recovering generated stuff

- The edited `TOOLS.py` lives on disk in the toolset folder — the workspace
  and the `.rsi/commits/<epoch>.json` release manifests (if toolsmith is also
  enabled) are the source of truth.
- A successful mount emits a `TOOLSET EDITING PROTOCOL` doctrine block into the
  system prompt on the current turn.
- See [`../TOOLSET_STAGING_AND_SANDBOX.md`](../TOOLSET_STAGING_AND_SANDBOX.md)
  for the full staging lifecycle, sandbox details, and troubleshooting guide.