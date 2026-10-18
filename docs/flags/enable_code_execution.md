# `enable_code_execution`

```python
enable_code_execution: bool = False
```

**Category**: Tools & Security

## What it does

Mounts the Python execution toolset: `tool_execute_python_code` (inline code,
max 2000 chars), `tool_execute_python_file` (runs an existing workspace script),
plus file-reading helpers (`tool_read_lines`, `tool_read_chars`,
`tool_grep_file`). **Security-sensitive — off by default.**

## What to expect

- Pair with `python_autonomy_level` (default `"safe"`), `auto_approve_python`,
  and a `confirm_handler` to control execution approval.
- Execution runs with the discussion workspace as CWD.

## Recovering generated stuff

Scripts run with the working directory as CWD; any files they write are
auto-registered as artifacts by the workspace snapshot diffing
(`_sync_tool_artifacts`) — recovered via `discussion.artefacts` like any
artifact.

## Usage

```python
result = discussion.chat(
    "Run a script that computes fibonacci(30)",
    enable_code_execution=True,
)
```

## Related flags

- `auto_approve_python` — skips confirmation prompts.
- `python_autonomy_level` (non-boolean) — autonomy tier.