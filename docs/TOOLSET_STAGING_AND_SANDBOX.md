# Toolset Staging & the Sandbox — Operator and Author Guide

This document is the **current** ground truth for two things that are commonly
confused, and for the failure they produce together:

1. **Editing a toolset from inside an agent** — the `toolset_staging` surface.
2. **Running Python from inside an agent** — the `execute_python` sandbox,
   including its operator-consent model for environment mutations.

It supersedes the older `docs/TOOL_EDITOR_AGENT.md` advice wherever the two
disagree. In particular, the editing toolset that ships today and is exercised
by the test suite is **`toolset_staging`** (folder
`src/lollms_client/tools_bindings/lcp/default_tools/toolset_staging/`), not
`toolset_editor`.

---

## Part 1 — The core confusion: *callable* is not *editable*

`LollmsPersonality.from_handbag(..., extra_tools=[X])` makes the `tool_*`
functions of toolset `X` **callable** by the agent. It does **not** give the
agent a way to *read or write* `X`'s source. Those are two separate
capabilities, and conflating them is the root cause of the most common
"the agent produced a review instead of an edit" bug.

| Capability        | Provided by                                   |
|-------------------|-----------------------------------------------|
| Call `X`'s tools  | `extra_tools=[X]`                             |
| Edit `X`'s source | `extra_tool_staging_paths=[toolset_staging]` **and** the target exposed via `workspace_path` or a declared staging path |

If you mount only `X` and ask the agent to change it, the agent has no legal
write path. It will typically improvise: it reads the source through some
other tool, then tries to rewrite it via `execute_python` — which is sandboxed
and **blocks**. That is the "blocks after spawning a test agent" symptom.

---

## Part 2 — Wiring an agent that can edit a toolset

There are exactly three things that must be true. Miss any one and staging
yields zero candidates.

### 2.1 Declare the toolsets you CALL and the surfaces you EDIT separately

`extra_tools` means exactly one thing: **the toolsets the agent may call**. Do
not smuggle the editing surface into it — that conflates two intents and can
mount `toolset_staging` in contexts that never edit. The editing surface is a
first-class, parallel parameter: **`extra_tool_staging_paths`**.

```python
from lollms_client import LollmsPersonality

agent = LollmsPersonality.from_handbag(
    handbag,
    extra_tools=[TARGET_TOOLSET],                  # toolsets the agent CALLS
    extra_tool_staging_paths=[STAGING_TOOLSET],    # editing surface(s) the agent EDITS through
    workspace_path=PARENT_FOLDER,                  # see 2.2
)
```

Both lists are mounted in the **same** single LCP binding (so the staging tool
is callable), but they are tracked separately on the personality:

- `_extra_tool_files` — callable toolsets (`extra_tools`),
- `_extra_staging_files` — editing surfaces (`extra_tool_staging_paths`).

Declaring a folder in `extra_tool_staging_paths` also registers it with the
staging toolset via `set_staging_paths(...)`, so those folders are valid edit
targets **even when they live outside `workspace_path`** (see 2.2 for why that
matters). You may pass several staging paths; each entry is one editing surface.

### 2.2 `workspace_path` (or a declared staging path) must expose the target

`tool_stage_list_stageable` enumerates candidates from two sources:

- **workspace**: `workspace.rglob("TOOLS.py")` under `workspace_path`;
- **live binding**: `discovered_tools[*]["_python_file_path"]` from
  `env.client.tools`.

The workspace scan only finds a toolset if `workspace_path` is an **ancestor**
of its folder. Point `workspace_path` at the toolset itself and the scan finds
zero candidates. Point it at the parent that *contains* the toolset.

```
PARENT_FOLDER/            <-- workspace_path
    target_toolset/
        TOOLS.py          <-- discovered
        description.yaml
```

### 2.3 The staging toolset must run on the agent that owns the binding

A **sub-agent** is spawned as a fresh process. Its `env` does **not** carry the
parent's `client.tools` binding, so `_list_external_toolsets` returns nothing
there. If you spawn a test agent and expect it to edit an out-of-workspace
toolset, staging will report `count: 0` **in the sub-agent** even though the
parent sees the toolset fine.

Do the editing on the agent constructed with
`from_handbag(extra_tool_staging_paths=[toolset_staging])`. If a sub-agent
genuinely needs to edit, the target must live **under that sub-agent's
`workspace_path`** so the filesystem scan finds it.

---

## Part 3 — The staging lifecycle

The staging toolset is a *focus* surface: exactly **one** toolset is staged at
a time. Multi-staging is refused by design (two staged sources corrupt the
model's context and confuse which file a patch targets).

```
tool_stage_list_stageable   -> pick ONE candidate
tool_stage(name=...)        -> stage it (refuses if another is already staged)
tool_stage_read             -> inspect current TOOLS.py (also injected each turn)
tool_stage_patch            -> apply Aider SEARCH/REPLACE blocks (atomic)
tool_stage_rewrite          -> replace whole TOOLS.py
tool_stage_test             -> run a tool_* from the on-disk source to validate
tool_stage_unstage          -> release the focus slot (file stays on disk)
tool_stage_status           -> non-mutating view
```

### 3.1 The prompt region refreshes from disk every turn

Once a toolset is staged, `LollmsPersonality._build_staged_toolsets_context`
calls `toolset_staging_state.build_staged_prompt_block(workspace)` **each
turn**. That function reads `TOOLS.py` from disk (the registry stores only
identity: name, folder, timestamp). Therefore:

- a `tool_stage_patch` write appears in the model's context on the **next**
  round with no reload call;
- you never need to "re-stage" to see an edit.

### 3.2 Validation gate

Both `tool_stage_patch` and `tool_stage_rewrite` run the candidate source
through `ast.parse` **and** require at least one top-level `def tool_*(...)`.
A broken source never reaches disk. On a `patch`, all blocks apply atomically
or none do — an ambiguous or missing SEARCH block aborts the whole write.

### 3.3 Hot reload

On a successful write, `tool_stage_patch`/`tool_stage_rewrite` call
`personality.refresh_tools_and_skills()` (best-effort, via
`_reload_personality`) so the edited `tool_*` is immediately callable through
the live binding.

---

## Part 4 — Diagnosing "it blocks after spawning a test agent"

`tool_stage_list_stageable` returns `count: 0` in three distinct situations,
and the agent **cannot** tell them apart from a bare count. It therefore now
returns a `diagnosis` string that names the cause:

| `diagnosis` mentions…                                     | Cause | Fix |
|-----------------------------------------------------------|-------|-----|
| "No `env` handle was provided"                            | tool called without a host context | run it with the injected `env` |
| "exposes no .client.tools binding … sub-agent spawn does not inherit" | sub-agent lost the parent binding | edit on the owning agent, or put the target under the sub-agent's `workspace_path` |
| "live binding is reachable but advertised no TOOLS.py entries" | binding present, but the target is a single-file toolset (no `TOOLS.py` folder) | staging only works on folder-layout toolsets |

If `count > 0` but the agent still improvises into `execute_python`, the prompt
was not refreshed — confirm the personality build path
(`_build_staged_toolsets_context`) is active for that discussion.

---

## Part 5 — The `execute_python` sandbox and the `env` category

Separate from toolset editing: when an agent runs Python, it does so through
`execute_python`, which executes user code in an isolated child process under
`SandboxContext`. The child dies at the end of the call; nothing is patched in
the parent.

### 5.1 Risk categories and operator consent

Risky operations are grouped into categories (`network`, `process`, `dynamic`,
`system`, `filesystem`, `delete`, and now `env`). In `safe` autonomy mode the
operator is prompted for any category they have not yet approved. Answering
**always** persists every category in the snippet **except** `delete`
(deletions always re-prompt).

### 5.2 The `env` category

`os.environ[...] = ...`, `del os.environ[...]`, `os.putenv(...)`,
`os.unsetenv(...)` and the `os.environ.update/setdefault/pop/popitem/clear`
methods are detected by `execute_python._detect_env_mutation_ops` and routed to
the `env` category. **Reading** the environment (`os.environ["K"]`,
`os.getenv`, `dict(os.environ)`) is never flagged.

This is the category that unblocks the classic `configure_ngspice47()` case:

```python
os.environ["SPICE_SCRIPTS"] = str(scripts_dir)   # -> ENV category, prompts in safe mode
```

Consent flow:

1. First run in `safe` mode → operator sees an `ENV: os.environ[...] =` prompt.
2. `[y]es` → runs **this** execution. `[a]lways` → persists `env` for the
   session (`execute_python._approve_categories({"env"})`).
3. Consent is carried into the child process. `run_in_sandboxed_child` serializes
   the approved set into the spawn config (`sandbox_state`); the child
   publishes it on `sys.__lollms_execute_python_approved_categories__`
   **before** importing the kernel. The runtime guards
   `_gated_putenv` / `_gated_unsetenv` / `_Environ.__setitem__` /
   `_Environ.__delitem__` allow the mutation iff `"env"` is present.

Host-side pre-approval (no prompt at all):

```python
from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
ep._approve_categories(frozenset({"env"}))
```

### 5.3 Headless stays denied

In pytest/CI (`_is_headless_environment()`), the "allow" branch deliberately
does **not** set `user_authorized_risky`, so the child never receives `env`
consent and an environment mutation is still refused. This is the intended
security property, not a bug: with no human present, the sandbox is the
control.

### 5.4 The user-frame label contract (author pitfall)

`SandboxContext` guards decide "is this user code?" with
`_caller_is_user_code(extra_label=...)`. The `extra_label` is the
`co_filename` the child used to `compile()` the user source — i.e. the
`script_label` passed to `run_in_sandboxed_child`, and set on
`SandboxContext.user_frame_label`:

```python
with SandboxContext(workspace_root=..., profile=..., user_frame_label=script_label) as sbx:
    ...
```

**Never** key user-frame detection on a fixed literal set if the filename is
caller-controlled. An earlier version matched only
`{"<execute_python>", "<execute_python_file>", "<execute_python_data_query>"}`;
any caller passing another `script_label` (a workspace filename, a test label,
…) compiled user code under a name none of the guards recognised, so **every**
guard passed through and the sandbox silently permitted everything. Binding
`user_frame_label` from the compilation site fixes this class of bug for all
tool callers.

---

## Part 6 — Checklist

Before asking an agent to edit a toolset, verify:

- [ ] `extra_tools` contains the target toolset(s) the agent **calls**;
      `extra_tool_staging_paths` contains the editing surface(s)
      (e.g. `toolset_staging`). They are two different intents — never put
      `toolset_staging` into `extra_tools`.
- [ ] The target toolset is reachable either because `workspace_path` is an
      **ancestor** of it, OR because its folder is listed in
      `extra_tool_staging_paths`.
- [ ] The work happens on the agent that owns the binding (not a sub-agent
      whose `env` lacks `client.tools`) — or the target is inside the
      sub-agent's `workspace_path`.
- [ ] `tool_stage_list_stageable` returns `count >= 1` (read `diagnosis` if it
      returns 0).
- [ ] The target is a **folder-layout** toolset (`<name>/TOOLS.py`), not a
      single-file toolset.
- [ ] The target does **not** declare `TOOL_LIBRARY_READONLY = True` — a
      protected toolset refuses `tool_stage` and re-checks the flag at write
      time, so edits are blocked even if it was staged earlier.

Before authorizing Python that mutates the environment:

- [ ] Expect an `ENV:` prompt in `safe` mode; answer `always` to persist.
- [ ] Do not expect env mutations to run in headless/CI — that is by design.