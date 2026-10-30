# 🛠️ Tool Calling Format in LoLLMS

This document is the authoritative reference for **how an LLM calls tools inside a LoLLMS discussion**. It covers the wire format (the XML tags the model emits), the JSON parameter schema, the result envelope the runtime feeds back, the termination protocol, the tool registry rules, and the anti-loop guards that keep the agentic loop honest.

> **Scope**: This describes the *Worker-tier* tool grammar. The *Orchestrator-tier* persona never sees tool syntax — it only delegates (see [Orchestrator Mode](#-orchestrator-mode-no-tool-grammar)).

---

## 1. The `<tool>` Tag

A tool call is a single XML block emitted by the model:

```xml
<tool>{"name": "tool_execute_python_code", "parameters": {"code": "print(2+2)"}}</tool>
```

### Rules

| Rule | Detail |
| :--- | :--- |
| **One action per turn** | The stream parser (`_StreamState`) halts generation the instant a functional tag is dispatched. The executor takes the next round. |
| **Closed tag required** | The block must be closed with `</tool>`. Unclosed tags are force-flushed at end-of-stream (`flush_remaining_buffer`) and dispatched if the JSON is valid. |
| **JSON body** | The body is a JSON object with exactly two keys: `name` (string) and `parameters` (object). |
| **No prose inside** | The tag body must be pure JSON. Prose before/after the tag is allowed and is preserved as the assistant's conversational text. |
| **Verbatim preservation** | The raw `<tool>` block is appended to `virtual_history` exactly as emitted — this preserves KV-cache alignment across rounds. |

### Minimal example

```xml
I'll compute that now.
<tool>{"name": "tool_execute_python_code", "parameters": {"code": "import math; print(math.factorial(10))"}}</tool>
```

---

## 2. Parameter Schema

The `parameters` object is a flat JSON map. Each key must match a parameter declared in the tool's schema (discovered via AST introspection for LCP tools, or the explicit spec dict for callables).

```json
{
  "name": "tool_query_database_sql",
  "parameters": {
    "query": "SELECT id, name FROM users LIMIT 5",
    "database": "app.db"
  }
}
```

### Type mapping

| Schema type | JSON value |
| :--- | :--- |
| `str` | string |
| `int` | number (integer) |
| `float` | number |
| `bool` | `true` / `false` |
| `list` | array |
| `dict` | object |

### Reserved / stripped keys

The runtime **strips** any `tool_context` key the model emits inside `parameters` before execution. `tool_context` is a host-only injection channel (used for live progress streaming and cooperative cancellation) and is never part of the LLM-visible schema.

---

## 3. Tool Result Envelope

After execution, the runtime feeds the result back to the model inside a `<tool_result>` envelope appended to `virtual_history` as a **user-role** message:

```xml
<tool_result name="tool_execute_python_code" status="SUCCESS">
4
</tool_result>

⚠️ The JSON above is a RESULT, not a new tool call. Do not re-emit it.
```

### Envelope contract

| Field | Meaning |
| :--- | :--- |
| `name` | The tool that was called (echoed back for correlation). |
| `status` | `SUCCESS` or `FAILURE`. |
| **Body** | The sanitized tool output (base64 blobs stripped, oversized payloads offloaded). |
| **Anti-loop header** | A warning that the JSON is a result, not a new call — prevents the model from mimicking its own output as a fresh tool call. |

### Failure envelope

On failure, the body carries the actual error message (never masked as success):

```xml
<tool_result name="tool_execute_python_code" status="FAILURE">
Error: NameError: name 'foo' is not defined
</tool_result>

⚠️ **Tool Execution Failed.**
The tool 'tool_execute_python_code' encountered an error. Diagnose the cause,
fix the code, and retry — or explain the failure to the user.
```

---

## 4. The `<done/>` Termination Protocol

The agentic loop gives the model **sovereign control** over termination via the `<done/>` tag (alias `<end/>`).

```xml
The report is complete and saved to report.md.
<done/>
```

### Rules

1. **Sovereign termination** — any round ending in `<done/>` terminates the loop immediately with `status: "done"`. The tag is stripped from `ai_msg.content` before saving so it never appears in the UI.
2. **No analysis gate** — the runtime does not second-guess the model's termination signal.
3. **Plain text = the answer** — a text-only round *without* `<done/>` is treated as the model's final conversational answer (`status: "conversational"`). There is no stall-counter and no forced-continuation directive.
4. **Action continuation** — if the model emits a functional tag instead, the action executes and the loop continues with a completion mandate.

### Exit statuses

| Status | Trigger |
| :--- | :--- |
| `done` | Model emitted `<done/>`. |
| `conversational` | Plain-text round without `<done/>`. |
| `cancelled` | `cancel_generation()` observed at a safe point. |
| `action` | Tool ran / artifact built / correction injected. |
| `loop_break` | Duplicate artifact, repeated phantom/malformed call. |
| `max_rounds` | Round budget exhausted. |

---

## 5. Tool Registry & Sovereign Opt-In

The tool registry follows a **Strict Sovereign Opt-In Doctrine**. The system will **never** automatically expose all default LCP tools to the LLM. Tools are activated only via one of these mechanisms:

| # | Mechanism | How |
| :-- | :--- | :--- |
| 1 | **Personality Handbag tools** | `LollmsPersonality.from_handbag()` with an `LCPBinding` in `personality.tools`. Extra RAG sources can be attached via `extra_knowledge_bases=[...]` — see [07_PERSONALITY_INTEGRATION.md §7](./lcp/07_PERSONALITY_INTEGRATION.md#7-building-a-personality-from-a-handbag-with-extra-knowledge-bases). |
| 2 | **Explicit `tools=` parameter** | A dict of callables, or a list of default LCP tool names. |
| 3 | **Auto-mounted data tools** | `enable_data_tools=True` + data files in workspace → mounts `semantic_data_engineer`. |
| 4 | **Spinoff sub-agents** | Worker-tier only, no flag — `tool_spinoff_code_specialist`, `tool_spinoff_presentation_designer`. |

### Example: explicit tool list

```python
discussion.chat(
    user_message="Search the web for the latest AI news and summarize it.",
    tools=["tool_web_search", "tool_web_fetch"],
)
```

### Example: explicit callable

```python
def my_executor(code: str) -> dict:
    return {"success": True, "output": "done"}

discussion.chat(
    user_message="Run this code.",
    tools={
        "tool_my_executor": {
            "name": "tool_my_executor",
            "description": "Executes Python code.",
            "parameters": [{"name": "code", "type": "str", "description": "The code."}],
            "callable": my_executor,
        }
    },
)
```

---

## 6. LCP Tool Discovery

LCP (Local LollmsCommunicationProtocol) tools are discovered by scanning folders and files for `tool_*` functions. The AST parser builds the JSON schema automatically.

```python
client = LollmsClient(
    tools_binding_name="lcp",
    tools_binding_config={
        "tools_folders": ["./my_tools", "C:/shared/lcp_library"],
        "tool_files": ["C:/projects/utilities/matter_lock_controller.py"],
    },
)
```

### Tool authoring contract

```python
def tool_analyze_data(file_name: str) -> dict:
    """
    Analyzes a file in the workspace.

    Args:
        file_name (str): The file to analyze.
    """
    from pathlib import Path
    path = Path(file_name)  # CWD is already the discussion workspace
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}
    return {"success": True, "output": "Analysis complete."}
```

**Rules for tool authors:**

- Tools must be **agnostic** — never accept `discussion_instance` / `lollms_client_instance` unless explicitly required for advanced patterns (the AST parser filters these out of the LLM schema).
- Tools rely on the **CWD switch** — the orchestrator sets CWD to the discussion workspace before execution.
- Tools communicate via **return dicts only** — never manipulate the DB or commit artifacts directly.
- New files created by a tool are **auto-detected** and registered as artifacts after execution.

### Optional `tool_context` injection

If a tool declares `tool_context` in its signature, the runtime injects a `ToolContext` for live progress streaming and cooperative cancellation:

```python
def tool_long_task(duration: int = 5, tool_context=None) -> dict:
    import time
    for i in range(duration):
        time.sleep(1)
        if tool_context:
            tool_context.emit_progress(f"Step {i+1}/{duration}", {"stream": "stdout"})
    return {"success": True, "output": "done"}
```

### Reserved host-injected parameters

Beyond `tool_context`, the dispatcher strips four reserved names from the
AST-generated schema and injects live objects at call time. They never reach
the LLM:

| Parameter | Injected object | None when |
| :--- | :--- | :--- |
| `lollms_client_instance` | Active `LollmsClient` | dispatcher has no client handle |
| `discussion_instance` | Active `LollmsDiscussion` | tool runs outside a discussion |
| `tool_context` | Host `ToolContext` (progress/cancel) | host provides none |
| `user_interaction_handler` | Host confirm/Q&A handler | host provides none |

Using the injected client (TTI example; never construct your own client):

```python
def tool_generate_image(lollms_client_instance=None, subject: str = "") -> dict:
    """Generate an image via the host's TTI binding.
    Args:
        subject (str): What to depict.
    """
    lc = lollms_client_instance
    if lc is None or getattr(lc, "tti_binding", None) is None:
        return {"success": False, "error": "No TTI binding on the client."}
    result = lc.tti_binding.generate_image(prompt=subject)
    out = {"success": True, "output": f"Generated '{subject}'."}
    if isinstance(result, str):
        out["file_name"] = result
        out["context_files"] = [result]
    return out
```

Accessing the current discussion (history and attached files):

```python
def tool_recall(discussion_instance=None, keyword: str = "") -> dict:
    """Search conversation history.
    Args:
        keyword (str): Keyword to look for.
    """
    if discussion_instance is None:
        return {"success": False, "error": "No discussion context."}
    hits = [m.content for m in discussion_instance.messages
            if keyword.lower() in (m.content or "").lower()]
    files = getattr(discussion_instance, "files", []) or []
    return {"success": True, "output": f"{len(hits)} hits, {len(files)} attached files."}
```

Declare reserved params with `=None` defaults and do NOT document them in the
docstring's `Args:` block — they are stripped automatically and must stay
invisible to the model.

---

## 7. Error Handling & Anti-Loop Guards

The loop implements a **three-pronged defense** against infinite tool-retry loops.

### 7.1 Raw dict failure detection

The runtime checks the raw result dict for `success: False` or a non-200 `status_code` **before** looking at the sanitized string — so `prompt_injection` text can never mask a failure.

### 7.2 Error-aware sanitization

When a tool returns `prompt_injection` alongside `success: False`, the sanitizer includes the actual `error` message in the text fed back to the model.

### 7.3 FailureMemory loop guard

A per-turn `FailureMemory` tracks call signatures:

- **Signature**: `f"{tool_name}::{json.dumps(tool_params, sort_keys=True)}"`
- **First failure**: recorded, error fed back, loop continues (model gets one chance to self-correct).
- **Second identical call**: blocked, "Loop Intercepted" error returned, loop breaks.

### 7.4 Phantom tool guard

If the model calls a tool **not in the active registry**, the runtime injects a correction listing the exact available tool names and forces a continuation round. Repeated phantom calls (same signature 2+ times) hard-break the loop.

### 7.5 Malformed JSON guard

Malformed JSON or a missing `name` triggers a format-error correction. Repeated malformed signatures hard-break the loop.

### 7.6 Environment epoch gating

A failing tool may be retried identically only if the workspace changed in between (artifact built, different tool ran, files mutated). Otherwise the second identical failure is hard-blocked.

---

## 8. Streaming Events

Tool lifecycle is visible to host applications through structured events (in `EventMode.FULL_CALLBACK_MODE` or `MIXED_MODE`):

| Event | Meta |
| :--- | :--- |
| `MSG_TYPE_TOOL_START` | `{"tool_name": str, "parameters": dict}` |
| `MSG_TYPE_TOOL_END` | `{"tool_name": str, "success": bool, "output": str, "error": str\|None}` |

### Live progress streaming

Long-running tools can stream intermediate output via `ToolContext.emit_progress()`:

| EventMode | Progress event | Meta |
| :--- | :--- | :--- |
| `PROCESSING_TAG_MODE` | `MSG_TYPE_CHUNK` | `{"tool_progress": True, "stream": "stdout"\|"stderr"}` |
| `FULL_CALLBACK_MODE` | `MSG_TYPE_INFO` | `{"tool_progress": True, "stream": ...}` |
| `MIXED_MODE` | Both | as above |
| `SILENT_MODE` | Nothing | — |

**Cancellation**: if the streaming callback returns `False` during progress, the tool receives that signal and should abort cooperatively.

---

## 9. Complete Examples

### Example A — Python execution

```xml
<tool>{"name": "tool_execute_python_code", "parameters": {"code": "import math; print(math.factorial(10))"}}</tool>
```

Result:

```xml
<tool_result name="tool_execute_python_code" status="SUCCESS">
3628800
</tool_result>
```

### Example B — SQL query

```xml
<tool>{"name": "tool_query_database_sql", "parameters": {"query": "SELECT COUNT(*) FROM orders", "database": "shop.db"}}</tool>
```

### Example C — Web search

```xml
<tool>{"name": "tool_web_search", "parameters": {"query": "latest AI news 2026"}}</tool>
```

### Example D — Multi-step chain

```xml
I'll fetch the data, then analyze it.
<tool>{"name": "tool_web_fetch", "parameters": {"url": "https://example.com/data.json"}}</tool>
```

*(next round, after result)*

```xml
<tool>{"name": "tool_execute_python_code", "parameters": {"code": "import json; d=json.load(open('data.json')); print(len(d))"}}</tool>
```

*(next round)*

```xml
The dataset contains 42 records.
<done/>
```

---

## 10. Security Gates

Two critical gates control whether dangerous tools are even visible to the model:

| Gate | Default | Filters out |
| :--- | :--- | :--- |
| `allow_dynamic_tools` | `False` | `tool_execute_python_data_query` (LLM-authored Python execution). |
| `enable_code_execution` | `False` | `tool_execute_python_code` (arbitrary Python strings). |

If the model attempts to call a filtered-out tool, the runtime intercepts it **before execution**, injects a correction listing available tools, and forces a continuation round.

### Additional gates

| Gate | Default | Effect |
| :--- | :--- | :--- |
| `enable_computer_use` | `False` | Mounts desktop automation tools — **only if the model has vision**. |
| `enable_vlm_query` | `False` | Mounts `tool_vlm_query` for blind models to route image questions. |
| `shell_autonomy_level` | `"safe"` | Whitelist checks for shell commands. |
| `python_autonomy_level` | `"safe"` | Prohibits subprocess escapes in Python. |
| `auto_approve_python` | `False` | Bypasses human-in-the-loop confirmation. |

---

## 11. Orchestrator Mode (No Tool Grammar)

When `orchestrator_mode=True`, the persona is **structurally forbidden** from emitting tool calls:

- `tools_prompt` is **empty** — tool syntax never enters its context.
- Mimicry is **structurally impossible** — it cannot imitate a tool call it has never seen.
- Allowed moves per round: **ANSWER** (→ `<done/>`) or **DELEGATE** (→ `<delegate>` block).

### Delegation grammar

```xml
<delegate>
<task>
Precise, self-contained instructions for the specialist.
</task>
<context_files>
data.csv
analysis.py
</context_files>
</delegate>
```

The spawned Worker runs with `orchestrator_mode=False` and the full tool grammar. Its report returns as a single `[WORKER REPORT]` envelope containing **no executable grammar** — closing the report-injection mimicry vector.

---

## 12. Quick Reference

```text
┌─────────────────────────────────────────────────────────────┐
│  TOOL CALL                                                   │
│  <tool>{"name": "...", "parameters": {...}}</tool>          │
├─────────────────────────────────────────────────────────────┤
│  TOOL RESULT                                                 │
│  <tool_result name="..." status="SUCCESS|FAILURE">          │
│  ...body...                                                  │
│  </tool_result>                                              │
├─────────────────────────────────────────────────────────────┤
│  TERMINATION                                                 │
│  <done/>                                                     │
├─────────────────────────────────────────────────────────────┤
│  DELEGATION (orchestrator only)                              │
│  <delegate><task>...</task><context_files>...</context_files></delegate> │
└─────────────────────────────────────────────────────────────┘
```

### Invariants

1. **One action per turn** — the parser halts on the first functional tag.
2. **Verbatim preservation** — raw tags go into `virtual_history` unchanged.
3. **Sovereign termination** — `<done/>` is never rejected.
4. **Plain text = answer** — no forced continuation.
5. **Failure is data** — errors are always visible to the model.
6. **Environment epoch gating** — identical retries require a workspace change.
7. **Structural immunity** — the Orchestrator never sees tool syntax.

---

## See Also

- [`lollms_discussion/README.md`](../src/lollms_client/lollms_discussion/README.md) — the full agentic loop, virtual history, and context compression pipeline.
- [`docs/flags/README.md`](flags/README.md) — per-flag documentation for `chat()`.
- [`docs/lcp/`](lcp/) — LCP binding deep dives and toolset authoring guide.