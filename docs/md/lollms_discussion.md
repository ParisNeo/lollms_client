# lollms_discussion — Developer Reference

> **Scope** — This document covers every subsystem of `lollms_discussion` in detail:
> the layered context model, the event/callback system, conversation branching,
> artefacts (and how the LLM interacts with them), the full tool catalogue (external
> and built-in), the REPL text tools, and context compression.

---

## Table of Contents

1. [Core Architecture](#1-core-architecture)
2. [The Event System](#2-the-event-system)
3. [The Branching System](#3-the-branching-system)
4. [The Artefact System](#4-the-artefact-system)
5. [The Tooling System](#5-the-tooling-system)
6. [REPL Text Tools](#6-repl-text-tools)
7. [Context Compression](#7-context-compression)
8. [Source Title Extraction](#8-source-title-extraction)
9. [chat() Return Value Reference](#9-chat-return-value-reference)
10. [Worked Example](#10-worked-example)

---

## 1. Core Architecture

### 1.1 What Is a `LollmsDiscussion`?

A `LollmsDiscussion` is not a simple list of messages. It is a **stateful, database-backed,
branch-aware object** that:

- assembles a complete LLM context from multiple independent layers every turn,
- manages versioned artefacts (code, documents) that the LLM can create and patch inline,
- routes RAG queries through preflight retrieval and agentic tool calls,
- executes inline `<tool_call>` tags in a streaming loop,
- compresses history automatically when the context window gets full.

All of this happens without any orchestration code in the application — `chat()` handles it.

### 1.2 Data Zones — Layered Context Assembly

Every generation pass assembles context from six distinct layers in a deterministic order.
Each has a different lifetime and ownership model.

| Layer | Attribute | Lifetime | Who writes it |
|---|---|---|---|
| Memory | `discussion.memory` | Persistent, cross-session | App, after `memorize()` |
| User data | `discussion.user_data_zone` | Session-wide | App at startup |
| Discussion data | `discussion.discussion_data_zone` | Per-discussion | App as task evolves |
| Personality data | `discussion.personality_data_zone` | Per-turn, transient | Framework (RAG injection) |
| Active artefacts | `discussion.artefacts` (active) | Per-discussion, versioned | LLM via XML tags, or app |
| System prompt | `discussion._system_prompt` | Constant + per-turn augmentation | App at creation, framework |

> **Rule of thumb** — Put facts that survive resets in `memory`. Session preferences go
> in `user_data_zone`. Task-specific state goes in `discussion_data_zone`. Never manually
> write to `personality_data_zone` — the framework owns it for RAG chunk injection.

### 1.3 Creation and Persistence

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager

lc = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={"host_address": "http://localhost:9642"}
)

# Persistent discussion (recommended)
db = LollmsDataManager("sqlite:///my_vault.db")
discussion = LollmsDiscussion.create_new(
    lollms_client=lc,
    db_manager=db,
    autosave=True,                                    # commit after every chat()
    system_prompt="You are a collaborative coding assistant.",
    max_context_size=8192,                            # activates auto-compression
)

# Reload an existing discussion
discussion = LollmsDiscussion.load(db, discussion_id=42)
```

`max_context_size` is the only parameter that controls automatic context compression.
Without it, no pruning or summarisation ever runs.

---

## 2. The Event System

### 2.1 The Callback Contract

Both `chat()` and `simplified_chat()` accept an optional `streaming_callback` keyword
argument. The signature is:

```python
def callback(text: str, msg_type: MSG_TYPE, meta: dict) -> bool | None:
    ...
    return True    # continue streaming
    # return False # stop streaming immediately
```

- Called **synchronously** on every event during generation.
- Returning `False` sends a stop signal to the binding. `None` or `True` means continue.
- Exceptions inside the callback are silently swallowed to protect the generation loop.
- `text` is always a `str` (the payload), even for events that primarily communicate
  through `meta`.

> **Note on stop signals** — Returning `False` does not guarantee immediate stop.
> It depends on the binding. llama.cpp honours it within one chunk boundary; some
> network bindings drain the current generation pass first.

### 2.2 Complete MSG_TYPE Reference

| Value | Name | `text` payload | `meta` dict |
|---|---|---|---|
| 0 | `MSG_TYPE_CHUNK` | One streaming token chunk | `{}` |
| 1 | `MSG_TYPE_CONTENT` | Complete message (bulk) | `{}` |
| 2 | `MSG_TYPE_CONTENT_INVISIBLE_TO_AI` | Shown to user only | `{}` |
| 3 | `MSG_TYPE_CONTENT_INVISIBLE_TO_USER` | Shown to AI only | `{}` |
| 4 | `MSG_TYPE_THOUGHT_CHUNK` | Streaming `<think>` chunk | `{}` |
| 5 | `MSG_TYPE_THOUGHT_CONTENT` | Complete `<think>` block | `{}` |
| 6 | `MSG_TYPE_EXCEPTION` | Exception message | `{}` |
| 7 | `MSG_TYPE_WARNING` | Warning message | `{}` |
| 8 | `MSG_TYPE_INFO` | Status info string | `{}` |
| 9 | `MSG_TYPE_STEP` | Instant step label | `{}` |
| 10 | `MSG_TYPE_STEP_START` | Phase name / description | `{id: uuid, ...extra}` |
| 11 | `MSG_TYPE_STEP_PROGRESS` | Percentage string | `{percent: float}` |
| 12 | `MSG_TYPE_STEP_END` | Phase completion text | `{id: uuid, status: str, ...}` |
| 13 | `MSG_TYPE_JSON_INFOS` | JSON generation summary | `{}` |
| 14 | `MSG_TYPE_REF` | Reference `[text](path)` | `{}` |
| 15 | `MSG_TYPE_CODE` | JS to execute client-side | `{}` |
| 16 | `MSG_TYPE_UI` | Vue.js component markup | `{}` |
| 17 | `MSG_TYPE_NEW_MESSAGE` | Start of new message | `{}` |
| 18 | `MSG_TYPE_FINISHED_MESSAGE` | End of current message | `{}` |
| 19 | `MSG_TYPE_TOOL_CALL` | `"🔧 tool_name"` | `{tool: str, params: dict}` |
| 20 | `MSG_TYPE_TOOL_OUTPUT` | JSON result (≤ 2 000 chars) | `{tool: str, result: dict}` |
| 21 | `MSG_TYPE_REASONING` | AI reasoning text | `{}` |
| 22 | `MSG_TYPE_SCRATCHPAD` | Scratchpad state snapshot | `{}` |
| 23 | `MSG_TYPE_OBSERVATION` | AI observation text | `{}` |
| 24 | `MSG_TYPE_ERROR` | Severe error | `{}` |
| 25 | `MSG_TYPE_GENERATING_TITLE_START` | `""` | `{}` |
| 26 | `MSG_TYPE_GENERATING_TITLE_END` | Generated title | `{}` |
| 27 | `MSG_TYPE_SOURCES_LIST` | JSON source list | `[{title, content, source, score, ...}]` |
| 28 | `MSG_TYPE_INIT_PROGRESS` | Init status string | `{percent: float}` |
| 29 | `MSG_TYPE_ARTEFACTS_STATE_CHANGED` | JSON list of affected titles | `{artefacts: [...], action?: str}` |
| 30 | `MSG_TYPE_TOOLS_LIST` | JSON tool catalogue | `{tools: [{name, description, parameters, source}, ...]}` |
| 31 | `MSG_TYPE_CONTEXT_COMPRESSION` | JSON compression report | `{tokens_before, tokens_after, budget, cache_hit, summary_generated, artefact_pressure}` |

### 2.3 Step Events — Named Phases

`STEP_START` and `STEP_END` always arrive in matched pairs linked by a UUID in
`meta["id"]`. A UI can use this to show collapsible progress indicators.

Phases emitted during `chat()` with an agentic tool round:

```
🔍 Pre-flight knowledge retrieval…     [STEP_START id=abc…]
  (tool execution)
  ◀ Pre-flight retrieval complete       [STEP_END   id=abc…, source_count=N]

🔧 Running tool: Read File              [STEP_START id=def…, tool="read_file"]
  ◀ Completed: read_file               [STEP_END   id=def…, status="success"]
```

Phases emitted during `simplified_chat()`:

```
🔍 Analyzing intent…                   [STEP_START / STEP_END]
📚 Loading context documents…          [STEP_START / STEP_END]
🔎 Searching external knowledge…       [STEP_START / STEP_END]
✍️  Generating answer…                  [STEP_START / STEP_END]
```

### 2.4 A Complete Callback Implementation

```python
from lollms_client.lollms_types import MSG_TYPE

def on_event(text, msg_type, meta):
    match msg_type:
        case MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n  ▶ {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"  ◀ {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n  🔧 {meta['tool']}({meta['params']})")

        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  ✓  {meta['tool']} → {text[:120]}")

        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            action = meta.get("action", "updated")
            print(f"\n  📄 Artefacts {action}: {text}")

        case MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION:
            if meta["cache_hit"]:
                print(f"\n  ⚡ Compression cache hit: "
                      f"{meta['tokens_before']:,} → {meta['tokens_after']:,}")
            elif meta["summary_generated"]:
                print(f"\n  ⚡ Summarised {meta['messages_pruned']} messages: "
                      f"{meta['tokens_before']:,} → {meta['tokens_after']:,}")
            elif meta["artefact_pressure"]:
                print(f"\n  ⚡ Artefact pressure — LLM will deactivate artefacts")

        case MSG_TYPE.MSG_TYPE_SOURCES_LIST:
            for src in (meta if isinstance(meta, list) else []):
                score = src.get("relevance_score", src.get("score", 0))
                print(f"  📚 [{src['title']}]  score={score:.2f}")

        case MSG_TYPE.MSG_TYPE_TOOLS_LIST:
            tools = meta.get("tools", [])
            print(f"\n  🔩 Tool catalogue: {len(tools)} tools")

        case MSG_TYPE.MSG_TYPE_WARNING:
            print(f"\n  ⚠️  {text}")

        case MSG_TYPE.MSG_TYPE_INFO:
            print(f"  ℹ  {text}")

result = discussion.chat("Explain quicksort.", streaming_callback=on_event)
```

---

## 3. The Branching System

### 3.1 Concept

Every message forms a node in a directed acyclic graph. Each node has a single
`parent_id`. A **branch** is the linear path from the root to any leaf node. At any
moment one branch is **active** — this is the sequence of messages the LLM sees as
"the conversation."

Branching enables **what-if exploration**: instead of deleting and rewriting, you
create an alternative continuation that leaves the original intact.

### 3.2 What Creates a Branch

| Trigger | How |
|---|---|
| **Regeneration** | `chat(add_user_message=False)` after a user message creates a new assistant response at the same parent node |
| **Explicit fork** | `add_message(parent_id=<any earlier node id>)` |
| **Agentic tool loop** | Does **not** create branches — all tool rounds in one `chat()` call are a single linear turn from the graph's perspective |

### 3.3 Navigation API

```python
# The full message DAG
tree = discussion.get_tree()

# Ordered list of LollmsMessage objects on the active branch
branch = discussion.get_branch(discussion.active_branch_id)

# Switch the default active branch
discussion.switch_branch(branch_tip_id="<uuid of any leaf node>")

# Tip node of the active branch
tip = discussion.get_active_branch_tip()

# All leaf nodes (all branch tips)
tips = discussion.get_all_branch_tips()

# Generate on a specific branch without changing the default
result = discussion.chat(
    "What if we used Redis instead?",
    branch_tip_id="<uuid>",
)
```

### 3.4 Branch-Aware Context Assembly

When `chat()` assembles context, it walks from the root to `branch_tip_id` (defaulting
to `active_branch_id`). Only messages on that path are included. Messages on sibling
branches are invisible to the model. The pruning summary from context compression is
also stored per-branch.

> A discussion can have dozens of branches representing different architectural
> decisions, alternative story paths, or competing code implementations — all within
> a single database entry, all sharing the same artefact namespace.

---

## 4. The Artefact System

### 4.1 What Is an Artefact?

An artefact is a **named, versioned, typed document** that lives inside the discussion
but outside the message history. It is stored in a separate `ArtefactManager`
(`discussion.artefacts`) and **injected verbatim** into the context window alongside
(not inside) the message thread. Active artefacts are always visible to the LLM;
inactive ones are excluded from the context but remain in the database.

| Field | Description |
|---|---|
| `title` | Human-readable identifier, unique per discussion. Used in all XML tags and tool calls. |
| `type` | Declared type: `document`, `code`, `json`, `csv`, `html`, `markdown`, etc. Guides post-processing. |
| `language` | Optional language hint for `code` artefacts (`python`, `javascript`, `sql`, …). |
| `content` | Current text content. |
| `version` | Integer incremented on every successful patch or full replacement. |
| `active` | Boolean. Only `True` artefacts are injected into the context window. |
| `description` | Optional free-text. Stored but not injected into context. |
| `author` | Optional author metadata. |
| `id` | UUID assigned at creation. Stable across versions. |

### 4.2 LLM Interaction — XML Tags

The LLM communicates with the artefact system entirely through **XML tags embedded
in its text output**. These are post-processed by the framework after generation:

1. The raw tags are stripped from the displayed message.
2. The framework applies the operation to the artefact store.
3. `MSG_TYPE_ARTEFACTS_STATE_CHANGED` is fired with a list of affected artefact titles.

#### 4.2.1 Create or Replace (Full Content)

```xml
<artefact name="app.py" type="code" language="python"
          author="assistant" description="FastAPI entry point">
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}
</artefact>
```

If an artefact named `app.py` already exists this **replaces** its content in full and
increments the version. If it is new, it is created at version 1 with `active=True`.

#### 4.2.2 Patch an Existing Artefact — SEARCH/REPLACE (Aider Format)

The SEARCH/REPLACE format lets the LLM modify a specific range without re-transmitting
the entire content. This is the primary token-efficient editing mechanism.

```xml
<artefact name="app.py">
<<<<<<< SEARCH
@app.get("/")
def root():
    return {"status": "ok"}
=======
@app.get("/")
def root():
    return {"status": "ok", "version": "1.0"}

@app.get("/health")
def health():
    return {"healthy": True}
>>>>>>> REPLACE
</artefact>
```

> The `SEARCH` block must match the existing content **exactly** (including whitespace).
> If it does not match, the patch is rejected and the artefact is unchanged. On the
> next turn the LLM receives the rejection in its context.

#### 4.2.3 Activate / Deactivate

Inactive artefacts are excluded from the context window, saving tokens. The LLM manages
this with the `<artefact_control>` tag:

```xml
<!-- Deactivate — removes from context window for subsequent turns -->
<artefact_control name="old_schema.sql" action="deactivate"/>

<!-- Reactivate when needed again -->
<artefact_control name="old_schema.sql" action="activate"/>
```

Under context pressure, the framework can also dynamically register a
`deactivate_artefacts` tool (see §5.3.5) so the LLM can shed artefacts as a tool call
before the current generation pass overflows the context budget.

#### 4.2.4 Revert to a Previous Version

```xml
<artefact_revert name="app.py" version="2"/>
```

```python
# Equivalent Python API
discussion.artefacts.revert("app.py", target_version=2)
```

#### 4.2.5 Image Generation and Editing (TTI)

When a TTI (text-to-image) module is configured and the caller passes
`enable_image_generation=True` to `chat()`, the LLM can generate and edit images.
These tags are only described to the LLM when `lollmsClient.tti is not None`:

```xml
<!-- Generate a new image artefact -->
<generate_image width="1024" height="1024">
    A photorealistic sunset over a mountain lake, golden hour lighting
</generate_image>

<!-- Edit an existing image artefact -->
<edit_image name="hero_image.png">
    Add a small wooden boat in the foreground
</edit_image>
```

### 4.3 The ArtefactManager Python API

Application code can manipulate artefacts directly without going through the LLM:

```python
mgr = discussion.artefacts

# List all artefacts
all_arts    = mgr.list(active_only=False)
active_arts = mgr.list(active_only=True)

# Get one artefact by title (returns dict or None)
art = mgr.get("app.py")

# Create / replace
mgr.add(
    title="readme.md",
    artefact_type="document",
    content="# My Project",
    active=True,
)

# Patch (SEARCH/REPLACE)
mgr.patch("app.py", search="return ok", replace='return {"ok": True}')

# Activate / deactivate
mgr.activate("readme.md")
mgr.deactivate("old_draft.md")

# Revert
mgr.revert("app.py", target_version=1)

# Delete permanently
mgr.delete("scratch.txt")
```

---

## 5. The Tooling System

### 5.1 Mechanics

Tools are registered at the start of each `chat()` call and described to the LLM via
an `## Available Tools` section injected into the system prompt. The LLM calls a tool
by emitting a `<tool_call>` JSON tag anywhere in its streaming output.

**The inline streaming tool-call loop:**

```
1. LLM streams text freely.
2. Framework detects <tool_call>…</tool_call> mid-stream.
3. Streaming stops. Pre-tag text is forwarded to the callback as MSG_TYPE_CHUNK.
4. JSON is parsed, the tool is dispatched, result is serialised (≤ 2 000 chars).
5. MSG_TYPE_TOOL_CALL and MSG_TYPE_TOOL_OUTPUT events are fired.
6. A <tool_result name="…">…</tool_result> block is appended to the context.
7. Generation resumes. Loop continues up to max_reasoning_steps times.
8. When the LLM produces no more tool_call tags, the loop exits.
9. tool_call / tool_result tags are stripped from the final displayed text.
```

```
# LLM emits:
<tool_call>{"name": "read_file", "parameters": {"path": "src/main.py"}}</tool_call>

# Framework injects and resumes:
<tool_result name="read_file">{"success": true, "content": "…"}</tool_result>
```

> Tool call tags are stripped from the final displayed message. The user only sees
> the clean text produced around and after the tool interactions.

### 5.2 Registering External Tools

External tools are passed to `chat()` as a `tools` dict. Each key is a unique tool
name and each value is a spec dict:

| Spec field | Type | Description |
|---|---|---|
| `name` | `str` | Tool name used in the `<tool_call>` tag |
| `description` | `str` | Natural-language description injected into the system prompt |
| `parameters` | `list[dict]` | Each dict: `name`, `type`, `optional` (bool), `default`, `description` |
| `output` | `list[dict]` | Output field specs. If any has `name="sources"`, tool is treated as a RAG tool |
| `callable` | callable | Python function. Must return a `dict` with at least a `success` key |

```python
def list_directory(path: str) -> dict:
    import os
    try:
        entries = os.listdir(path)
        return {"success": True, "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"success": False, "error": str(e)}

tools = {
    "list_dir": {
        "name":        "list_directory",
        "description": "List files in a directory on the local filesystem",
        "parameters": [
            {"name": "path", "type": "str", "optional": False,
             "description": "Absolute or relative directory path"}
        ],
        "output":   [{"name": "entries", "type": "list"}],
        "callable": list_directory,
    }
}

result = discussion.chat(
    "What files are in /tmp?",
    tools=tools,
    streaming_callback=on_event,
)
```

#### 5.2.1 RAG Tools

Any tool whose `output` spec contains a field named `"sources"` is automatically
registered as a **RAG tool**. Its results are:

- tracked in `result["sources"]`,
- emitted as `MSG_TYPE_SOURCES_LIST` events,
- subject to `rag_top_k` and `rag_min_similarity_percent` filtering.

```python
# marks this as a RAG tool — framework handles top-k and min-similarity filtering
"output": [{"name": "sources", "type": "list"}],

# each item in sources should be:
# {"content": str, "score": float (0–1), "source": str, "metadata": dict}
```

#### 5.2.2 Legacy `personality.data_source` Support

If the personality object has a callable `data_source` attribute, it is automatically
wrapped as the `search_personality_knowledge` RAG tool. Each chunk's display title
is resolved by walking: `chunk["title"]` → `metadata["title"]` → `metadata["filename"]`
→ `metadata["name"]` → `_extract_content_title(chunk_content)` → `kb_name`.

### 5.3 Built-in Framework Tools

The following tools are registered automatically by `chat()`. Each has a toggle flag
on the `chat()` signature. All default to `True`.

#### 5.3.1 `show_tools`

**Toggle:** `enable_show_tools=True`

Assembles a complete structured catalogue of all registered tools (user-supplied and
built-in) and fires `MSG_TYPE_TOOLS_LIST` so a UI can render a live tool panel without
screen-scraping the system prompt. Image tools only appear when TTI is available.

```
# LLM calls:
<tool_call>{"name": "show_tools", "parameters": {}}</tool_call>

# UI receives MSG_TYPE_TOOLS_LIST with meta:
# {"tools": [{"name": "…", "description": "…", "source": "user"|"builtin", …}, …]}
```

#### 5.3.2 `extract_artefact_text`

**Toggle:** `enable_extract_artefact=True`

Extracts a line range from an existing artefact and saves it as a new artefact. The
range is specified by **text anchors** — the first few words of the opening and closing
lines — rather than line numbers, so the LLM does not need to know exact positions.

```json
{
  "name": "extract_artefact_text",
  "parameters": {
    "source_title":    "app.py",
    "new_title":       "app.py — router section",
    "start_line_hint": "# ── Router",
    "end_line_hint":   "# ── End Router",
    "occurrence":      1,
    "artefact_type":   "code",
    "language":        "python"
  }
}
```

| Parameter | Description |
|---|---|
| `source_title` | Title of the source artefact |
| `new_title` | Title for the new extracted artefact |
| `start_line_hint` | First ≥ 4 words of the opening line (case/indent insensitive prefix match) |
| `end_line_hint` | First ≥ 4 words of the closing line (searched forward from start) |
| `occurrence` | Which occurrence of `start_line_hint` to use (1-based, default 1). Useful for repeated patterns like multiple `def __init__` |
| `artefact_type` | Type for the new artefact (defaults to source's type) |
| `language` | Language hint (defaults to source's language) |

Both anchors are included in the extracted slice. The tool returns `start_line_no`,
`end_line_no`, `total_lines`, and `lines_extracted` (all 1-based for human readability).

#### 5.3.3 `final_answer`

**Toggle:** `enable_final_answer=True`

A signal tool used in composable-answer workflows. When the LLM has built up the
answer by calling `append_to_answer` / `update_answer_section` (internal scratchpad
tools), it calls `final_answer()` to signal that the composed answer is ready. Returns
the assembled `full_text`.

#### 5.3.4 `request_clarification`

**Toggle:** `enable_request_clarification=True`

Lets the LLM request clarification from the user mid-turn. The agentic loop stops
and the question surfaces in the response metadata.

```json
{
  "name": "request_clarification",
  "parameters": {"question": "Which database engine should I target — Postgres or SQLite?"}
}
```

#### 5.3.5 `deactivate_artefacts` (context-pressure only)

This tool is **conditionally registered** — it only appears in the tool catalogue when
context compression detects that artefacts are the dominant token pressure source
(`artefact_tokens > history_tokens` AND total > budget). It is inserted at position 0
in the tool list so the LLM sees it first, prefixed with `⚠️ CONTEXT PRESSURE`.

```json
{
  "name": "deactivate_artefacts",
  "parameters": {"titles": ["old_draft.md", "deprecated_schema.sql"]}
}
```

Returns `{success, deactivated, not_found, tokens_freed_estimate}`. After deactivation
the framework automatically re-runs compression to verify the window is now within budget.

### 5.4 The Composable Answer / Scratchpad

During an agentic turn, the LLM maintains a **composable answer** — a list of named
sections that can be built up, revised, and removed across tool rounds before the final
answer is assembled. This lets the LLM correct earlier sections after learning new
information from later tool calls.

The following internal functions are available as the scratchpad tool suite:

```python
# Build the answer incrementally
append_to_answer(content, section_id=None, sources=None)
    # → {success, section_id, total_sections, current_length}

# Revise a section (records a self-correction entry)
update_answer_section(section_id, new_content, reason=None)
    # → {success, section_id}

# Remove a section
remove_answer_section(section_id, reason=None)

# Inspect the current answer
get_current_answer()
    # → {full_text, sections, total_sections, total_length, last_updated}

# Scratchpad key-value store
update_scratchpad(key, value, category="notes"|"assumptions")
get_scratchpad(category=None)
remove_scratchpad_entry(key, category)

# Track uncertain assumptions and their resolution
update_assumption_status(assumption_key, status, reason=None)
```

The scratchpad and self-correction log are returned in `result["scratchpad"]` and
`result["self_corrections"]`.

### 5.5 Preflight RAG (Personality)

When a personality is provided, `chat()` runs a **preflight RAG pass** before the
first generation step. It generates a concise search query from the current conversation
context, queries `search_personality_knowledge`, and injects the results into
`personality_data_zone`. Sources found during preflight appear in `result["sources"]`
with `phase="preflight"` and fire `MSG_TYPE_SOURCES_LIST`.

### 5.6 `simplified_chat()` — Lighter RAG Path

`simplified_chat()` is a lighter alternative that skips the agentic tool-call loop.
It does a single structured intent-detection call, then injects relevant data zones
or external search results before generating.

```python
result = discussion.simplified_chat(
    user_message="What are the best indexing strategies for our schema?",
    rag_data_stores={
        "db_docs": lambda q: vector_db.search(q, top_k=5),
        "wiki":    lambda q: wiki_search(q),
    },
    streaming_callback=on_event,
    rag_top_k=5,
    rag_min_similarity_percent=0.5,
)
```

Intent detection returns `needs_internal_knowledge`, `needs_full_documents`,
`needs_external_search`, and `reasoning`. The first two control data zone injection;
the third controls whether `rag_data_stores` is queried.

`simplified_chat()` has three code paths:

- **Fast path** — greeting / trivial message (`"hi"`, `"ok"`, etc.): streams directly, no intent detection.
- **Memory hit** — user message found verbatim in `discussion.memory`: streams directly.
- **Full path** — intent detection → optional RAG → stream final answer.

---

## 6. REPL Text Tools

### 6.1 Purpose

MCP tools and RAG pipelines can return very large payloads. Stuffing the whole payload
into a `<tool_result>` block is wasteful and may overflow the context budget. The
REPL text tools solve this with a **named in-session buffer**:

1. The LLM calls `text_store(handle, content)` immediately after a large result.
2. The payload is indexed server-side. Only a compact summary enters the context
   (record count, schema, first 3 previews).
3. The LLM navigates with targeted calls: `text_search`, `text_get_range`, etc.
4. Structured data can be filtered and aggregated without re-injecting the full payload.
5. When done, `text_to_artefact` persists the result as a discussion artefact.

All buffers are **ephemeral** — they exist only for the duration of a single `chat()`
call. To persist data, use `text_to_artefact`.

**Toggle:** `enable_repl_tools=True` (default). A single `TextBuffer` instance is
created per `chat()` call and shared across all nine tools.

### 6.2 Format Auto-Detection

When `text_store()` ingests a payload it automatically detects the format and builds
the appropriate index (priority order):

| Format | Detection rule |
|---|---|
| `json_array` | Top-level JSON `[{…}, …]` |
| `jsonl` | ≥ 3 of the first 10 lines parse as standalone JSON objects |
| `csv` | `csv.Sniffer` detects a consistent delimiter; `DictReader` yields ≥ 2 fields per row |
| `md_table` | Pipe-delimited lines with a `|---|` separator row |
| `numbered_list` | ≥ 5 lines matching `1. …` / `- …` / `• …` / `* …` patterns |
| `sections` | ≥ 3 `## Heading` blocks |
| `lines` | Fallback — every non-blank line is one record |

### 6.3 The Nine REPL Tools

| Tool | Signature | Purpose |
|---|---|---|
| `text_store` | `(handle, content)` | Ingest a large payload; returns format, total_records, schema, and a 3-record preview. **Call this first on any large tool output.** |
| `text_search` | `(handle, query, max_results=10, field=None)` | Keyword/regex search across all records, or just one field. Returns hit indices, summaries, and snippets. |
| `text_get_range` | `(handle, start, end)` | Return records `[start..end]` inclusive (0-based). For reading contiguous blocks. |
| `text_get_record` | `(handle, index)` | Return one full record by 0-based index, with per-field soft truncation at 800 chars. |
| `text_list_records` | `(handle, page=1, page_size=20)` | Paginated one-line-per-record directory listing. Use to scan for records of interest. |
| `text_filter` | `(handle, field, op, value, new_handle)` | Filter structured records where `field <op> value` and save to `new_handle`. ops: `eq ne gt lt gte lte contains startswith regex` |
| `text_aggregate` | `(handle, operation, field)` | Aggregate over a field. ops: `count sum min max avg unique unique_count` |
| `text_to_artefact` | `(handle, title, artefact_type="document", language="")` | Persist the buffer (or filtered subset) as a discussion artefact. Format is preserved (JSON stays JSON, CSV stays CSV). |
| `text_list_buffers` | `()` | List all active handles with format and record count. |

### 6.4 Typical Workflow — 50 Scientific Articles

```
# 1. A tool returns 50 articles as a JSON array (large payload)
<tool_call>{"name": "search_papers", "parameters": {"query": "climate ML"}}
  → 50 articles, ~40 000 tokens

# 2. Store immediately — only a compact summary enters the context
<tool_call>{"name": "text_store", "parameters": {"handle": "papers", "content": "<…50 articles…>"}}
  → {"format": "json_array", "total_records": 50, "schema": {"title":"str","year":"int",…}, "preview": […3 items…]}

# 3. Search for relevant items
<tool_call>{"name": "text_search", "parameters": {"handle": "papers", "query": "neural scaling", "max_results": 8}}
  → {"hits": [{"index": 3, "summary": "Scaling Laws for Neural Language Models", …}, …]}

# 4. Read one full record
<tool_call>{"name": "text_get_record", "parameters": {"handle": "papers", "index": 3}}
  → {"record": {"title": "Scaling Laws for Neural Language Models", "abstract": "…", "year": 2020}}

# 5. Filter to recent papers
<tool_call>{"name": "text_filter", "parameters": {"handle": "papers", "field": "year", "op": "gte", "value": 2022, "new_handle": "recent"}}
  → {"matched": 31, "total_checked": 50}

# 6. Aggregate to understand the filtered set
<tool_call>{"name": "text_aggregate", "parameters": {"handle": "recent", "operation": "unique_count", "field": "journal"}}
  → {"result": 8}

# 7. Persist the filtered subset as an artefact
<tool_call>{"name": "text_to_artefact", "parameters": {"handle": "recent", "title": "Recent climate ML papers", "artefact_type": "json"}}
  → {"artefact_id": "…", "records_saved": 31}
```

---

## 7. Context Compression

### 7.1 Trigger

Compression runs automatically at the start of every `chat()` call when
`max_context_size` is set and the current context token count exceeds
`max_context_size × 0.80`. The 20% headroom is reserved for the model's answer.

### 7.2 Compression Strategy

```
1. Measure: discussion.get_context_status() → tokens_before

2. If tokens_before ≤ budget → fast-path return (nothing to do)

3. Classify pressure:
   artefact_pressure = (artefact_tokens > history_tokens) AND (total > budget)

4. Cache lookup:
   key = SHA-1(branch_tip_id + "|" + sorted(active_artefact_ids))
   If cache[key] exists → re-apply stored summary without any LLM call (cache_hit=True)

5a. History compression (cache miss, not artefact-heavy):
    - Keep newest max(4, len(branch) // 4) turns intact
    - Ask the LLM to summarise the rest (preserving code, filenames, variable names)
    - Append to any existing pruning_summary (supports multi-round stacking)
    - Persist to cache (max 10 entries, FIFO eviction), commit to DB

5b. Artefact pressure (history too short to prune):
    - Return artefact_pressure=True
    - Caller (chat()) dynamically injects deactivate_artefacts into the tool registry

6. Fire MSG_TYPE_CONTEXT_COMPRESSION with full stats
```

### 7.3 Compression Cache

The cache lives in `discussion.metadata["_compression_cache"]` and is keyed on a
SHA-1 fingerprint of the current branch tip + active artefact set. A cache hit
re-applies the stored summary without any LLM call, making repeated compression on
the same conversation state essentially free. The cache self-evicts to 10 entries
(oldest first).

### 7.4 `MSG_TYPE_CONTEXT_COMPRESSION` Meta Fields

| Field | Type | Description |
|---|---|---|
| `tokens_before` | `int` | Token count before this compression attempt |
| `tokens_after` | `int` | Token count after (0 if not needed, same as before on cache hit) |
| `budget` | `int` | Effective budget (`max_context_size × 0.80`) |
| `cache_hit` | `bool` | True if summary was re-applied from cache |
| `summary_generated` | `bool` | True if a new LLM summary was generated this call |
| `artefact_pressure` | `bool` | True if artefacts are the dominant pressure source |
| `messages_pruned` | `int` | Number of messages that were summarised (summary_generated only) |

### 7.5 Long-term Memory via `memorize()`

While compression is reactive, `memorize()` is proactive. After a meaningful session,
call it to ask the LLM to identify durable insights and return them as a structured
object. The caller is responsible for persisting the summary:

```python
facts = discussion.memorize()
# facts: {title, key_decisions, technical_facts, preferences, summary}

if facts:
    discussion.memory = facts.get("summary", "")
    discussion.commit()

# Carry forward to the next discussion
new_discussion.memory = discussion.memory
```

---

## 8. Source Title Extraction

When RAG chunks arrive without explicit metadata titles, the framework runs
`_extract_content_title()` — a pure-regex, zero-LLM text analyser — on the chunk
content to derive a human-readable title. The priority chain for title resolution is:

1. `chunk["title"]`
2. `chunk["metadata"]["title"]`
3. `chunk["metadata"]["filename"]`
4. `chunk["metadata"]["name"]`
5. `src.rsplit("/", 1)[-1]` — basename of the source path
6. **`_extract_content_title(content)`** — content analysis (see below)
7. `"Source N"` — last-resort fallback

`_extract_content_title` tries these signals in order, all via regex, no LLM:

| Priority | Signal | Example |
|---|---|---|
| 1 | Markdown `#` / `##` / `###` heading | `## Results and Discussion` → `Results and Discussion` |
| 2 | RST underline (`====`, `----`) | `My Paper\n=======` → `My Paper` |
| 3 | YAML/JSON `title:` / `"title":` field | `{"title": "Deep Learning Survey"}` → `Deep Learning Survey` |
| 4 | HTML `<title>` or `<h1>` tag | `<title>Intro to ML</title>` → `Intro to ML` |
| 5 | `**Bold**` or `*italic*` opening phrase | `**Key Finding**\n…` → `Key Finding` |
| 6 | First line heuristic (short, no mid-sentence punctuation) | `Quantum Entanglement Review\n…` → `Quantum Entanglement Review` |
| 7 | First non-blank line, truncated to 80 chars | any | 

---

## 9. `chat()` Return Value Reference

```python
result = discussion.chat(…)
```

| Key | Type | Description |
|---|---|---|
| `user_message` | `LollmsMessage` | The user turn message object |
| `ai_message` | `LollmsMessage` | The assistant turn message object |
| `sources` | `list[dict]` | All RAG sources collected during this turn |
| `scratchpad` | `dict \| None` | Full scratchpad state if the turn was agentic, else `None` |
| `self_corrections` | `list \| None` | Log of `{section_id, old_content, new_content, reason, timestamp}` |
| `artefacts` | `list[dict]` | Every artefact created or modified this turn |

`result["sources"]` items:

| Field | Description |
|---|---|
| `title` | Human-readable title (after extraction chain) |
| `content` | Chunk text |
| `source` | Raw source identifier / path |
| `query` | The query that retrieved this chunk |
| `relevance_score` | Float 0–1 |
| `index` | Sequential index within this turn |
| `tool` | Tool that produced this source (agentic phase) |
| `phase` | `"preflight"` or `"agentic"` |

`result["ai_message"].metadata` fields:

| Field | Description |
|---|---|
| `mode` | `"direct"` / `"agentic"` / `"rlm_agentic"` |
| `duration_seconds` | Wall-clock time for the full turn |
| `token_count` | Tokens in the final response |
| `tokens_per_second` | Generation speed |
| `tool_calls` | `[{name, params, result}]` for every tool call (agentic only) |
| `sources` | Same as `result["sources"]` |
| `query_history` | `[{step, tool, query, result_count}]` for every RAG query |
| `scratchpad` | Scratchpad state (agentic only) |
| `self_corrections` | Self-correction log (agentic only) |
| `artefacts_modified` | List of titles of artefacts changed this turn |

---

## 10. Worked Example

Full agentic turn: external tool, REPL text tools, artefact creation, and event callback.

```python
import json
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE

lc = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={"host_address": "http://localhost:9642"}
)
db = LollmsDataManager("sqlite:///vault.db")

discussion = LollmsDiscussion.create_new(
    lollms_client=lc,
    db_manager=db,
    autosave=True,
    system_prompt="You are a research analyst.",
    max_context_size=8192,
)


# ── External tool ────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 20) -> dict:
    papers = arxiv_api.search(query, max_results=max_results)
    return {"success": True, "content": json.dumps(papers), "count": len(papers)}

tools = {
    "search_arxiv": {
        "name":        "search_arxiv",
        "description": "Search arXiv for academic papers by keyword",
        "parameters": [
            {"name": "query",       "type": "str", "optional": False},
            {"name": "max_results", "type": "int", "optional": True, "default": 20},
        ],
        "output":   [{"name": "content", "type": "str"}],
        "callable": search_arxiv,
    }
}


# ── Callback ─────────────────────────────────────────────────────────────────

def cb(text, msg_type, meta):
    match msg_type:
        case MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)
        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n▶ {text}")
        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n🔧 {meta['tool']}")
        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  ✓ {text[:80]}")
        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            print(f"\n📄 {text}")
        case MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION:
            print(f"\n⚡ {meta['tokens_before']:,} → {meta['tokens_after']:,} tokens")


# ── Chat ──────────────────────────────────────────────────────────────────────

result = discussion.chat(
    "Search arXiv for papers on LLM context compression (2023–2024). "
    "Store the results, filter to those published after 2023, "
    "summarise the top 5 by citation count, and save a structured "
    "report as an artefact.",
    tools=tools,
    streaming_callback=cb,
    enable_repl_tools=True,
    enable_extract_artefact=True,
    max_reasoning_steps=15,
)

print("\n\nArtefacts:", [a["title"] for a in result["artefacts"]])
print("Sources:  ", [s["title"] for s in result["sources"]])
print("Mode:     ", result["ai_message"].metadata["mode"])
print("Duration: ", result["ai_message"].metadata["duration_seconds"], "s")
```

During this turn the LLM will typically:

1. Call `search_arxiv(query="LLM context compression", max_results=30)` — large JSON result.
2. Call `text_store(handle="papers", content=<result>)` — gets a compact summary.
3. Call `text_filter(handle="papers", field="year", op="gt", value=2023, new_handle="recent")`.
4. Call `text_list_records(handle="recent")` — browse to identify top papers.
5. Call `text_get_record` a few times for full abstracts.
6. Write a `<artefact name="context_compression_report.md" type="document">…</artefact>`
   tag in its final response — parsed and saved automatically by `_post_process_llm_response`.
