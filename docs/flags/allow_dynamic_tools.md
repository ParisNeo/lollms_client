# `allow_dynamic_tools`

```python
allow_dynamic_tools: bool = False
```

**Category**: Tools & Security

**Parent master switch**: none. This is a standalone security gate.

**Gate**: requires an active LCP tools binding (`client.tools`). Without one,
the flag is inert — dynamic tool artifacts are written to disk but never
registered.

## What it does

Security gate for **tool-authored tools**. When `True`, an artifact of
`type="tool"` (a `.py` file declaring one or more `tool_*` functions) becomes
registrable as an executable LCP tool on subsequent rounds. When `False` (the
default), tool artifacts still exist as filesystem content — they just never
register into the tool registry, so the model cannot call them.

This is one half of the RSI tool-building loop. The other half — editing
**existing** registered toolsets — is `enable_toolsmith` (see the master table).
The two flags are independent and orthogonal:

| Scenario | `allow_dynamic_tools` | `enable_toolsmith` | Enable this pair when… |
|---|---|---|---|
| Model authors a **brand-new** tool as an artifact | ✅ required | — | you trust the model with net-new executable code |
| Model patches an **already-registered** toolset | — | ✅ required | you want the RSI read→write→test→commit cycle |
| Model wants to test a fresh build **and** register it | ✅ | ✅ | full self-improvement loop |

## What to expect

- **Security-sensitive.** This opens a **code-execution** surface — the model
  writes arbitrary Python that will be imported. In any non-isolated deployment
  it MUST be paired with:
  - `enable_code_execution=True` (the parent code-execution capability), AND
  - a `confirm_handler` (so the operator sees the execution approval modal).
- **Prompter gate.** Even with `allow_dynamic_tools=True`, the resulting tool
  call still passes through the `code_execution` `GATED_TASK_CATEGORY` path
  and requires operator confirmation through `_gated_confirm`.
- **Tool runtime context.** Dynamic `tool_*` functions receive the standard LCP
  `env` (`env.llm`, `env.tti`, their `*_model_profiles`, `*_switch` functions).
  They may declare reserved parameters `lollms_client_instance` /
  `discussion_instance` to receive the injected client/discussion at call time.
  They MUST NOT construct their own client — the injected instance is the only
  supported surface.
- **Silent skip on malformed tools.** If the artifact cannot be imported
  (syntax error, missing `tool_*` function, signature-contract violation), the
  registration is skipped with a console warning and the failure is reported
  back to the model as a tool-creation failure. **The turn is not aborted** —
  the model sees the error and can correct the artifact.
- **Non-persistent across client restarts.** Dynamically registered tools live
  on the current `LCPBinding` session. A new `LollmsClient` re-discovers tools
  from configured `tools_folders` only — dynamic tool artifacts on disk are
  not auto-loaded unless they land in a configured tools folder. To persist a
  dynamic tool across sessions, use `enable_toolsmith` +
  `tool_commit_toolset`, which writes into the registered toolset folder.

### Failure-mode table

| Symptom | Cause | Fix |
|---|---|---|
| Model writes tool artifact but never calls it | `allow_dynamic_tools=False` (default). Check by inspecting `client.tools.discovered_tools`. | Set the flag. |
| `execute_tool` returns `"Tool '<name>' not found"` | LCP binding rejected the artifact (syntax error, missing `tool_*`, or signature violation). | Check console for the skip warning; fix the artifact source and retry. |
| Tool call goes through but no confirmation modal appears | No `confirm_handler` wired. | Provide `confirm_handler=` in `chat()`. |
| Tool runs but has no access to `env.llm` / `env.tti` | Tool function signature omits reserved params. | Declare `lollms_client_instance=None` / `discussion_instance=None` in the tool signature. |
| Tool exists after turn N, gone after restarting the client | Dynamic tools are per-session. | Use `tool_commit_toolset` (with `enable_toolsmith=True`) to persist into a registered toolset folder. |

## Usage

Minimal enablement:

```python
result = discussion.chat(
    "Author a tool that fetches weather from wttr.in and register it, then call it for Paris.",
    enable_code_execution=True,      # parent code-execution capability
    allow_dynamic_tools=True,        # opens the register-a-new-tool gate
    confirm_handler=my_confirm,      # operator sees the modal for the code execution
)
```

Authoring + registering + running in one turn:

```python
result = discussion.chat(
    """Create a class-note-taking tool: it takes a topic and returns a 3-bullet
    summary retrieved from the local notes.md file. Save it as a tool artifact,
    register it, then invoke it once with topic='budget'.""",
    enable_code_execution=True,
    allow_dynamic_tools=True,
    confirm_handler=my_confirm,
)
# The model emits:
#   <artifact type="tool" title="notes_summarizer.py"> ... </artifact>
# The LCP binding registers tool_summarize_notes.
# The model then calls <tool>{"name": "tool_summarize_notes", "parameters": {"topic": "budget"}}</tool>.
```

## Related flags

- `enable_artefacts` — tool artifacts are created through the artifact system.
  With `enable_artefacts=False`, the file cannot even be authored, so
  `allow_dynamic_tools` has nothing to register.
- `enable_code_execution` — the parent code-execution capability. In production
  deployments, always pair the two.
- `auto_approve_python` — controls whether the resulting executions skip the
  confirmation modal entirely. Leave `False` unless fully sandboxed.
- `enable_toolsmith` — the RSI-assisted counterpart for editing already-
  registered toolsets (independent flag; no cascade).

## Recovering generated stuff

The dynamic tool artifact is recoverable through the standard artifact API:

- `discussion.artefacts.get("<title>")` — returns the authored source.
- `client.tools.discovered_tools` — lists the registered tool dicts (with
  `name`, `description`, `input_schema`, `_python_file_path`).
- `client.tools.execute_tool("<name>", {...})` — direct invocation, bypassing
  the model.

For a persisted toolset (across restarts), use the toolsmith RSI path:
`tool_commit_toolset` writes into the registered toolset folder and creates a
release manifest under `.rsi/commits/<epoch>.json`.

## Implementation reference

Dynamic tools follow the same LCP signature rules as file-based toolsets:

```python
def tool_my_tool(arg: str, lollms_client_instance=None, discussion_instance=None) -> dict:
    # `lollms_client_instance` is the active client; `discussion_instance` is
    # the active discussion. Never construct your own client inside a tool.
    return {"success": True, "output": f"got: {arg}"}
```

See `examples/simple/personality_with_tti_tool.py` for a concrete example of
attaching a tool to an `AIPersonality` that accesses the injected
`LollmsClient` instance (reserved parameter `lollms_client_instance`) to call
the active TTI binding.