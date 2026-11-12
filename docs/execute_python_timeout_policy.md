# Sandboxed Execution Timeout Policy

How the LCP `execute_python` and `execute_python_data_query` tools govern the
wall-clock runtime of the child process that executes user code, and how the
operator stays in control of long-running executions.

> **Applies to**: `execute_python/sandbox_kernel.py ≥ current`,
> `execute_python/execute_python.py`, `execute_python_data_query/execute_python_data_query.py`.

---

## 1. The problem this solves

The sandboxed Python tools run the user's code inside an isolated **child
process** (see `sandbox_kernel.run_in_sandboxed_child`). Historically the
parent killed that child unconditionally at a fixed wall-clock deadline:

| Tool | Old hard timeout | Failure mode |
|---|---|---|
| `execute_python` (all three tools) | **180 s** | Child killed, generic `TimeoutError` envelope |
| `execute_python_data_query` | **120 s** | Child killed, generic `TimeoutError` envelope |

Legitimate long-running work — model fitting, large-file processing, batch
plot rendering — routinely exceeded these budgets. The child was discarded with
no recourse, and the operator had no way to say *"keep going — this is expected."*

---

## 2. The new policy in one sentence

> The child runs up to a **1-hour default budget**. When that budget elapses,
> the **operator is asked** *"keep waiting or stop?"* instead of the child being
> silently killed. *"Continue"* extends the wait by another full budget slice
> (repeatable). *"Stop"* terminates the child and returns a `TimeoutError`.

---

## 3. Defaults

| Tool | Default budget | Constant |
|---|---|---|
| `execute_python` | **3600 s** (1 h) | `_DEFAULT_EXECUTION_TIMEOUT_SECONDS` |
| `execute_python_data_query` | **3600 s** (1 h) | `_DEFAULT_DATA_QUERY_TIMEOUT_SECONDS` |

The budget is a **slice**, not a total cap: every time the operator answers
*"continue"*, the child gets another full slice. An operator who keeps
answering *"continue"* — or a handler that always returns *"continue"* — can
let a child run arbitrarily long; termination remains in the operator's hands.

---

## 4. Decision flow

```text
                  ┌──────────────────────────────┐
                  │   Child started              │
                  │   budget = 3600 s (slice)    │
                  └──────────────┬───────────────┘
                                 │
                      wait(slice)
                                 │
             ┌───────────────────┴───────────────────┐
             │ child exits in time?                  │
             └───────┬───────────────────────┬───────┘
                    yes                      no
                     │                        │
              return envelope           timeout_decision_cb(elapsed)
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                       "continue"          "stop"           (any other)
                          │                   │                   │
                     extend slice       kill child           kill child
                     wait again         return                return
                                       TimeoutError         TimeoutError
```

The elapsed time passed to the callback is the **real wall-clock wait**
(`time.monotonic()` delta), so a slow prompt for consent does not inflate the
reported value.

---

## 5. How the operator is asked

The operator-prompting logic lives in the **parent** tool module (never inside
the sandboxed child) and follows the same priority ladder used for the
existing authorization prompt:

### 5.1 `execute_python` — `_timeout_decision_prompt(script_label, elapsed)`

1. **Registered `_CONFIRM_HANDLER`** — the host-provided callback (NiceGUI,
   WebUI, VS Code extension, a test harness, …). It receives a structured
   dict:

   ```python
   {
       "tool_name": "execute_python",
       "action_type": "timeout",
       "script_label": "train_model.py",
       "elapsed_seconds": 3600.4,
   }
   ```

   The handler may return:
   - `"continue"` / `"wait"` / `"allow"` / `"always"` / `True` → **keep waiting**
   - `"stop"` / `"kill"` / `"reject"` / `False` / anything else → **terminate**

   Legacy handlers that already implement the authorization prompts
   (`"allow"`/`"always"`/`"reject"`/`True`/`False`) work unchanged — the
   decision strings are mapped onto the timeout protocol.

2. **Headless (pytest / CI)** → returns `"stop"`. A blocking prompt in a
   headless suite would hang the entire run; the sandbox is the control and no
   human is present to consent to a longer wait anyway.

3. **Interactive TTY** → prompts the operator directly:

   ```text
   ⏱️  'train_model.py' has been running for 3600s.
   Keep waiting? [c]ontinue / [s]top (default: continue):
   ```

4. **No interactive channel and no handler** → returns `"stop"` (fail-safe:
   never extend silently).

A broken handler is treated as `"stop"` — a hang is never the correct
consequence of a callback error.

### 5.2 `execute_python_data_query` — `_on_dq_timeout(elapsed)`

The data-query tool uses the **existing operator channel** it already requires
for execution authorization:

- If a confirm handler is present (`_has_registered_confirm_handler(env)`),
  the operator is prompted via `_confirm_data_query_execution` and `"allow"`/
  `"always"` → `"continue"`, anything else → `"stop"`.
- Otherwise (headless / no handler) → `"stop"`.

Because a null handler already causes the data-query tool to **refuse to run
at all** in headless mode (the strict static + runtime sandbox has no
operator-free authorization path), in practice this means: **a data-query
that starts will always prompt on timeout**.

---

## 6. Implementation notes (why this is correct)

The timeout retry loop is **not** implemented with `Popen.communicate()`.
Calling `communicate()` a second time on a process that has already raised
`TimeoutExpired` is undefined — the pipe is half-consumed and stdin may be
re-closed. The correct portable pattern is used instead:

1. **stdout / stderr are redirected to temp FILES, not pipes.** A child that
   writes more than the OS pipe buffer (64 KiB on many systems) to a pipe the
   parent is not draining blocks forever. A child writing to a file never
   blocks. This is what makes the retried wait loop safe on every platform.
2. **`proc.wait(timeout=slice)` in a loop.** `wait()` is well-defined and
   repeatable on a live process on both POSIX and Windows; on timeout, the
   decision callback runs, then either the loop breaks (stop) or repeats
   (continue).
3. **Output files are drained after exit** and cleaned up unconditionally —
   no temp file is ever left behind.

This design guarantees: no deadlock, no half-consumed stream, clean
termination on kill, and identical behavior on Windows and POSIX.

---

## 7. Writing a timeout-aware confirm handler

A handler that serves both normal authorization **and** timeout prompts just
inspects `action_type`:

```python
def my_confirm_handler(request):
    if isinstance(request, dict) and request.get("action_type") == "timeout":
        # Fine-grained policy: auto-continue the first three slices,
        # then ask the human.
        slices = request.get("elapsed_seconds", 0) // 3600
        if slices < 3:
            return "continue"
        return ask_human_ui(request["script_label"], request["elapsed_seconds"])

    # ... existing authorization branch (allow / always / reject) ...
    return "allow"
```

Register it once, wherever you already register the authorization handler:

```python
from lollms_client import LollmsClient

client = LollmsClient(...)

client.discussion.chat(
    ...,
    confirm_handler=my_confirm_handler,   # authorizes execution AND governs timeouts
)
```

Handlers that only implement the authorization path need **no changes** — the
timeout protocol maps their returns (`allow`/`always` → continue,
`reject` → stop) and the safe default (`"stop"`) applies when no decision can
be made.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Child still killed at ~exactly 1 h | No confirm handler, headless env — expected fail-safe | Register a `confirm_handler`, or run interactively |
| Prompt never appears in a GUI app | `sys.stdin.isatty()` is `False` inside GUI hosts | Register a `confirm_handler` — the interactive prompt requires a TTY |
| `TimeoutError: … exceeded the 3600s hard timeout (total Ns)` | Operator (or handler) answered *"stop"* | Increase `hard_timeout` at the call site, or make the handler auto-continue |
| Handler always auto-continues, child never stops | Handler returns `"continue"` unconditionally | Enforce a maximum slice count in the handler (see §7) |

---

## 9. Cross-references

- [`LOLLMS_CODE_SHELL_USAGE.md`](LOLLMS_CODE_SHELL_USAGE.md) §8 — general
  cross-reference index, including flag ↔ `chat()` boolean mapping and the
  `env` (tool runtime context) contract.
- [`lcp_tool_resource_access.md`](lcp_tool_resource_access.md) — how tool
  code reaches LLM / TTI / TTS / discussion resources through the injected
  client. The confirm-handler registry (`_CONFIRM_HANDLER` /
  `set_confirm_handler`) is the same seam used by the timeout policy.
- [`lcp/00_OVERVIEW.md`](lcp/00_OVERVIEW.md) §6 — the canonical `env`
  contract; `env.confirm_handler` and `env.tool_context` are read by both
  the authorization and timeout prompts.