# lollms_discussion — Developer Reference

> **Scope** — This document covers every subsystem of `lollms_discussion` in detail:
> the layered context model, the event/callback system, conversation branching,
> artefacts (and how the LLM interacts with them), notes, skills, inline interactive
> teaching widgets, the full tool catalogue (external and built-in), the REPL text
> tools, context compression, the scratchpad placement model, the silent artefact
> guard, the aider patch system, and the swarm multi-agent system.

---

## Table of Contents

1. [Core Architecture](#1-core-architecture)
2. [The Event System](#2-the-event-system)
3. [The Branching System](#3-the-branching-system)
4. [The Artefact System](#4-the-artefact-system)
5. [Notes](#5-notes)
6. [Skills](#6-skills)
7. [Inline Interactive Teaching Widgets](#7-inline-interactive-teaching-widgets)
   - 7.1 Concept
   - 7.2 Widget Types (html / react / svg)
   - 7.3 Attribute Reference
   - 7.4 Post-Processing
   - 7.5 Streaming Notification
   - 7.6 Accessing Widgets After the Turn
   - 7.7 Frontend Rendering Guide
   - 7.8 Approved CDN Libraries
   - 7.9 Rules and Constraints
   - 7.10 Teaching-Companion Text Rule
   - 7.11 Widget vs Artefact Decision Table
   - 7.12 Multiple Widgets in One Response
   - 7.13 Disabling / 7.14 simplified_chat()
8. [The Silent Artefact Guard](#8-the-silent-artefact-guard)
9. [The Tooling System](#9-the-tooling-system)
10. [REPL Text Tools](#10-repl-text-tools)
11. [Context Compression](#11-context-compression)
12. [Scratchpad Placement Model](#12-scratchpad-placement-model)
13. [Source Title Extraction](#13-source-title-extraction)
14. [`chat()` Parameter Reference](#14-chat-parameter-reference)
15. [`chat()` Return Value Reference](#15-chat-return-value-reference)
16. [The Swarm System](#16-the-swarm-system)
    - 16.1 Concept and Use Cases
    - 16.2 The Agent
    - 16.3 AgentRole Reference
    - 16.4 The HLF Protocol
    - 16.5 SwarmConfig Reference
    - 16.6 Swarm Modes
    - 16.7 Anti-Sycophancy System
    - 16.8 Execution Flow
    - 16.9 Shared Artefact Collaboration
    - 16.10 User Steering
    - 16.11 Swarm Events (MSG_TYPE)
    - 16.12 Result Dict (swarm turn)
    - 16.13 Message Metadata (swarm turn)
    - 16.14 Worked Examples
17. [Worked Example (single-agent)](#17-worked-example)

---

## 1. Core Architecture

### 1.1 What Is a `LollmsDiscussion`?

A `LollmsDiscussion` is not a simple list of messages. It is a **stateful, database-backed,
branch-aware object** that:

- assembles a complete LLM context from multiple independent layers every turn,
- manages versioned artefacts (code, documents) that the LLM can create and patch inline,
- saves structured notes and reusable skills as first-class typed artefacts,
- embeds live interactive teaching widgets directly inside chat messages,
- routes RAG queries through preflight retrieval and agentic tool calls,
- executes inline `<tool_call>` tags in a streaming loop with anti-duplication guards,
- compresses history automatically when the context window gets full,
- fires real-time artefact create/update events to the UI as each XML tag is processed,
- orchestrates multi-agent swarm sessions where specialised agents brainstorm, critique, collaborate on artefacts, play games, and run simulations.

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

### 1.3 Scratchpad — Volatile Tool-Output Context

`discussion.scratchpad` is a **per-turn volatile string** that accumulates the full text
of every tool result during an agentic loop. It is **not** stored in the database; it is
cleared at the end of every `chat()` call.

Its key property is *placement*: instead of being injected into the system-prompt header
(far from the user's question), `export()` injects it as a dedicated `role: system`
message immediately **after the last user message**. An empty scratchpad is suppressed
entirely — no blank system messages are ever emitted.

```
[system:  instructions + data zones + active artefacts]
...conversation history...
[user:    current question                             ]  <- last user turn
[system:  == TOOL OUTPUT SCRATCHPAD ==                 ]  <- injected only when non-empty
                                                           <- model continues here
```

### 1.4 Creation and Persistence

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

### 2.2 The `_active_callback` Lifecycle

At the start of every `chat()` and `simplified_chat()` call the streaming callback is
stashed on the discussion object:

```python
object.__setattr__(self, '_active_callback', callback)
```

This makes it available to `_post_process_llm_response` so that artefact create/update
events can be fired in real time as each XML tag is processed — **before** the full
response has been assembled. It is cleared unconditionally at every exit point:

```python
object.__setattr__(self, '_active_callback', None)
```

Application code should never read or write `_active_callback` directly. It is an
internal synchronisation channel between the streaming loop and the post-processor.

### 2.3 Complete MSG_TYPE Reference

| Value | Name | `text` payload | `meta` dict |
|---|---|---|---|
| 0 | `MSG_TYPE_CHUNK` | One streaming token chunk | `{}` or `{type, content}` for inline events |
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
| 17 | `MSG_TYPE_NEW_MESSAGE` | Start of new message | `{message_id: str}` |
| 18 | `MSG_TYPE_FINISHED_MESSAGE` | End of current message | `{}` |
| 19 | `MSG_TYPE_TOOL_CALL` | `"Calling tool_name"` | `{id, tool, params, offset}` |
| 20 | `MSG_TYPE_TOOL_OUTPUT` | JSON result (<=2000 chars) | `{id, tool, result, offset}` |
| 21 | `MSG_TYPE_REASONING` | AI reasoning text | `{}` |
| 22 | `MSG_TYPE_SCRATCHPAD` | Scratchpad state snapshot | `{}` |
| 23 | `MSG_TYPE_OBSERVATION` | AI observation text | `{}` |
| 24 | `MSG_TYPE_ERROR` | Severe error | `{}` |
| 25 | `MSG_TYPE_GENERATING_TITLE_START` | `""` | `{}` |
| 26 | `MSG_TYPE_GENERATING_TITLE_END` | Generated title | `{}` |
| 27 | `MSG_TYPE_SOURCES_LIST` | JSON source list | `[{title, content, source, score, ...}]` |
| 28 | `MSG_TYPE_INIT_PROGRESS` | Init status string | `{percent: float}` |
| 29 | `MSG_TYPE_ARTEFACTS_STATE_CHANGED` | JSON payload (see §2.5) | `{artefact, is_new}` or `{artefacts, action?}` |
| 30 | `MSG_TYPE_TOOLS_LIST` | JSON tool catalogue | `{tools: [{name, description, parameters, source}, ...]}` |
| 31 | `MSG_TYPE_CONTEXT_COMPRESSION` | JSON compression report | `{tokens_before, tokens_after, budget, cache_hit, summary_generated, artefact_pressure}` |
| 32 | `MSG_TYPE_SWARM_AGENT_START` | `"AgentName (role)"` | `{agent: str, role: str, round: int}` |
| 33 | `MSG_TYPE_SWARM_AGENT_END` | `"AgentName done"` | `{agent: str, round: int}` |
| 34 | `MSG_TYPE_SWARM_ROUND_START` | `"Round N/M"` | `{round: int}` or `{mode: str, agents: list}` for round 0 |
| 35 | `MSG_TYPE_SWARM_ROUND_END` | `"Round N complete"` | `{round: int, mean_confidence: float}` |
| 36 | `MSG_TYPE_SWARM_HLF` | JSON HLF message dict | `{hlf: {id, from, to, type, round, content, artefact_ref, confidence, ts}}` |
| 37 | `MSG_TYPE_SWARM_CONSENSUS` | `"Consensus reached after N rounds"` | `{round: int, confidence: float}` |
| 38 | `MSG_TYPE_ARTEFACT_CHUNK` | Raw chunk of artefact content being streamed | `{title, chunk, art_type, language}` |
| 39 | `MSG_TYPE_ARTEFACT_DONE` | Complete raw artefact content (pre-post-processing) | `{title, content, art_type, language, is_patch, attrs}` |
| 40 | `MSG_TYPE_NOTE_CHUNK` | Raw chunk of note content being streamed | `{title, chunk}` |
| 41 | `MSG_TYPE_NOTE_DONE` | Complete raw note content | `{title, content}` |
| 42 | `MSG_TYPE_SKILL_CHUNK` | Raw chunk of skill content being streamed | `{title, chunk, category, description}` |
| 43 | `MSG_TYPE_SKILL_DONE` | Complete raw skill content | `{title, content, category, description}` |
| 44 | `MSG_TYPE_WIDGET_CHUNK` | Raw chunk of widget source being streamed | `{title, chunk, widget_type}` |
| 45 | `MSG_TYPE_WIDGET_DONE` | Complete raw widget source (do NOT mount directly) | `{title, content, widget_type}` |

### 2.4 Inline Streaming Events (MSG_TYPE_CHUNK with meta)

Some events arrive as `MSG_TYPE_CHUNK` with `text=""` but a non-empty `meta` dict.
These fire in real-time as the LLM streams, before `_post_process_llm_response` runs,
and serve as **opening announcements** — they tell the UI *that* a tag has started,
but carry no content. Content arrives on the secondary stream (§2.5).

| `meta["type"]` | When fired | `meta["content"]` fields |
|---|---|---|
| `"artefact_update"` | Opening `<artefact name="...">` tag detected | `{title: str}` |
| `"note_start"` | Opening `<note title="...">` tag detected | `{title: str}` |
| `"skill_start"` | Opening `<skill title="..." category="...">` tag detected | `{title: str, category: str}` |
| `"inline_widget_start"` | Opening `<lollms_inline ...>` tag detected | `{title: str, widget_type: "html"|"react"|"svg"}` |

### 2.5 Secondary Content Streams — CHUNK and DONE Events

**Why secondary streams exist**

When the LLM generates an artefact, note, skill, or widget, the raw content must not
appear in the main chat bubble stream (`MSG_TYPE_CHUNK`). It would either break the UI
(a partially-rendered widget appearing mid-message) or corrupt the prose text the user
is reading. Instead, the content is routed to a **separate, parallel stream** using
dedicated event types.

This separation gives the application full control:

- A code editor panel can live-preview an artefact as it is typed.
- A side panel can accumulate note content without affecting the chat bubble.
- A widget builder can ignore widget source entirely and wait for the final
  `<lollms_widget id="...">` anchor that appears in the finished message.
- The app decides — it can render the secondary stream immediately, buffer it,
  display a progress indicator, or discard it entirely.

**Event sequence for a single tag**

For every XML tag the LLM opens, the framework emits exactly three stages:

```
1.  MSG_TYPE_CHUNK  text=""  meta={type:"artefact_update"|"note_start"|...}
       ↑ opening announcement — no content, just the title / attrs

2.  MSG_TYPE_ARTEFACT_CHUNK  (or NOTE_CHUNK / SKILL_CHUNK / WIDGET_CHUNK)
       ↑ zero or more of these, one per raw chunk arriving from the LLM
       ↑ fires INSTEAD of MSG_TYPE_CHUNK (content never goes to the chat bubble)

3.  MSG_TYPE_ARTEFACT_DONE  (or NOTE_DONE / SKILL_DONE / WIDGET_DONE)
       ↑ fires ONCE when the closing </artefact> tag is detected
       ↑ carries the COMPLETE raw content
       ↑ fires BEFORE _post_process_llm_response runs
```

After stage 3, `_post_process_llm_response` runs as normal and fires
`MSG_TYPE_ARTEFACTS_STATE_CHANGED` when the artefact is committed to the database.

**`meta` dict shapes for every secondary event:**

```python
# ── Artefact ──────────────────────────────────────────────────────────────
# CHUNK:
{"title": "app.py", "chunk": "def main():\n", "art_type": "code", "language": "python"}

# DONE:
{
    "title":    "app.py",
    "content":  "def main():\n    pass\n",  # complete raw content
    "art_type": "code",
    "language": "python",
    "is_patch": False,   # True when content contains <<<<<<< SEARCH markers
    "attrs":    {"description": "Entry point", "author": "assistant"},  # extra XML attrs
}

# ── Note ──────────────────────────────────────────────────────────────────
# CHUNK:  {"title": "Price Analysis", "chunk": "| Route | ..."}
# DONE:   {"title": "Price Analysis", "content": "| Route | ...\n| TUN→LYS | 89 |"}

# ── Skill ─────────────────────────────────────────────────────────────────
# CHUNK:  {"title": "Python Async HTTP", "chunk": "import aiohttp\n",
#           "category": "programming/python/async", "description": "Using aiohttp..."}
# DONE:   {"title": "Python Async HTTP", "content": "...",
#           "category": "programming/python/async", "description": "Using aiohttp..."}

# ── Widget ────────────────────────────────────────────────────────────────
# CHUNK:  {"title": "Circle Explorer", "chunk": "<html>\n", "widget_type": "html"}
# DONE:   {"title": "Circle Explorer", "content": "<!DOCTYPE html>...",
#           "widget_type": "html"}
# NOTE: Do NOT attempt to render/mount the widget source from WIDGET_DONE.
#       Wait for the lollms_widget anchor in the final message and use
#       ai_message.metadata["inline_widgets"] to render safely.
```

**Important: the `is_patch` field on `MSG_TYPE_ARTEFACT_DONE`**

When `is_patch=True`, the content contains one or more `<<<<<<< SEARCH / ======= /
>>>>>>> REPLACE` blocks rather than full file content. The application should NOT try
to display this as the complete file. Use `_post_process_llm_response`'s output
(the `MSG_TYPE_ARTEFACTS_STATE_CHANGED` event that follows) to get the patched result.

**Partial close-tag safety**

The framework uses a sliding-window scan to avoid emitting the closing tag characters
(`</artefact>`, `</note>`, etc.) as content chunks. There is at most a one-close-tag-
length delay between the LLM generating a character and it appearing in a CHUNK event,
which is imperceptible in practice.

### 2.6 `MSG_TYPE_ARTEFACTS_STATE_CHANGED` — Two Distinct Modes

This event is fired in two distinct contexts with different `meta` shapes.

**Per-artefact real-time event** (fired by `_post_process_llm_response` via
`_active_callback` as each XML tag is processed):

```python
# text — JSON string with operation details
# meta — {artefact: dict, is_new: bool}
{
    "text":  '{"type": "artefact_created", "title": "app.py", "version": 1, "art_type": "code"}',
    "meta":  {"artefact": {...full artefact dict...}, "is_new": True}
}
# or for an update:
{
    "text":  '{"type": "artefact_updated", "title": "app.py", "version": 3, "art_type": "code"}',
    "meta":  {"artefact": {...full artefact dict...}, "is_new": False}
}
```

**Batch summary event** (fired once at the end of `chat()` / `simplified_chat()` after
all XML has been processed, listing everything that changed this turn):

```python
# text — JSON list of affected titles
# meta — {artefacts: [list of artefact dicts], action?: str}
{
    "text":  '["app.py", "README.md", "Transavia Note"]',
    "meta":  {"artefacts": [...], "action": "updated"}
}
# For artefacts deactivated by context compression:
{
    "text":  '["old_draft.md"]',
    "meta":  {"artefacts": ["old_draft.md"], "action": "deactivated_for_compression"}
}
```

A UI that needs to update a single artefact panel immediately should listen to the
per-artefact event. A UI that refreshes the whole artefact list at turn-end can use
the batch event.

### 2.7 Step Events — Named Phases

`STEP_START` and `STEP_END` always arrive in matched pairs linked by a UUID in
`meta["id"]`. A UI can use this to show collapsible progress indicators.

Phases emitted during `chat()` with an agentic tool round:

```
> Loading N personality tool(s)...      [STEP_START id=abc...]
  < N personality tool(s) ready         [STEP_END   id=abc...]

> Pre-flight knowledge retrieval...     [STEP_START id=def...]
  < Pre-flight retrieval complete       [STEP_END   id=def..., source_count=N]

> Running: Read File                    [STEP_START id=ghi..., tool="read_file"]
  < Done: read_file                     [STEP_END   id=ghi..., status="success"]
```

> **Step ID correlation** — Every `STEP_END` carries the same `id` as its matching
> `STEP_START`. The `id` is also attached to `TOOL_CALL` and `TOOL_OUTPUT` events
> via the `offset` field so a UI can visually link tool calls to their step bubbles.

Phases emitted during `simplified_chat()`:

```
> Analyzing intent...                   [STEP_START / STEP_END]
> Loading context documents...          [STEP_START / STEP_END]
> Searching external knowledge...       [STEP_START / STEP_END]
> Generating answer...                  [STEP_START / STEP_END]
```

### 2.8 A Complete Callback Implementation

```python
from lollms_client.lollms_types import MSG_TYPE

def on_event(text, msg_type, meta):
    match msg_type:

        # ── Main chat bubble ──────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_CHUNK:
            t = meta.get("type")
            if not text and t == "artefact_update":
                # Opening announcement — content follows on ARTEFACT_CHUNK/DONE
                print(f"\n  [Artefact building] {meta['content']['title']}")
            elif not text and t == "note_start":
                print(f"\n  [Note building] {meta['content']['title']}")
            elif not text and t == "skill_start":
                c = meta["content"]
                print(f"\n  [Skill building] {c['title']}  [{c['category']}]")
            elif not text and t == "inline_widget_start":
                c = meta["content"]
                print(f"\n  [Widget building] {c['title']} ({c['widget_type']})")
            else:
                print(text, end="", flush=True)   # prose goes to chat bubble

        # ── Secondary stream: artefact ───────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK:
            print(text, end="", flush=True)        # live preview in side panel

        case MSG_TYPE.MSG_TYPE_ARTEFACT_DONE:
            label = "[patch]" if meta["is_patch"] else f"[{meta['art_type']}]"
            lang  = f" ({meta['language']})" if meta.get("language") else ""
            print(f"\n  Artefact ready: {meta['title']} {label}{lang}")
            # meta["content"] = complete raw content (patch block if is_patch=True)
            # Do NOT display as final — wait for MSG_TYPE_ARTEFACTS_STATE_CHANGED.

        # ── Secondary stream: note ────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_NOTE_CHUNK:
            print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_NOTE_DONE:
            print(f"\n  Note ready: {meta['title']}")

        # ── Secondary stream: skill ───────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_SKILL_CHUNK:
            print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_SKILL_DONE:
            print(f"\n  Skill ready: {meta['title']} [{meta['category']}]")

        # ── Secondary stream: widget ──────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_WIDGET_CHUNK:
            pass   # do NOT render; source is raw and incomplete

        case MSG_TYPE.MSG_TYPE_WIDGET_DONE:
            print(f"\n  Widget source ready: {meta['title']} ({meta['widget_type']})")
            # Do NOT mount from here — use lollms_widget anchor +
            # ai_message.metadata["inline_widgets"] for safe rendering.

        # ── Artefact committed ────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            if "artefact" in meta:
                op  = "created" if meta["is_new"] else "updated"
                art = meta["artefact"]
                print(f"\n  [Artefact {op}] {art['title']} v{art['version']}")
            else:
                action = meta.get("action", "updated")
                print(f"\n  [Artefacts {action}] {text}")

        # ── Steps ─────────────────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n  > {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"  < {text}  [{meta['id'][:8]}]")

        # ── Tools ─────────────────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n  [Tool] {meta['tool']}({meta['params']})")

        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  [Result] {meta['tool']} -> {text[:120]}")

        # ── Context compression ───────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION:
            if meta["cache_hit"]:
                print(f"\n  [Compression cache hit] "
                      f"{meta['tokens_before']:,} -> {meta['tokens_after']:,}")
            elif meta["summary_generated"]:
                print(f"\n  [Compressed {meta['messages_pruned']} messages] "
                      f"{meta['tokens_before']:,} -> {meta['tokens_after']:,}")

        # ── Sources ───────────────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_SOURCES_LIST:
            for src in (meta if isinstance(meta, list) else []):
                score = src.get("relevance_score", src.get("score", 0))
                print(f"  [Source] [{src['title']}]  score={score:.2f}")

        # ── Diagnostics ───────────────────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_WARNING:
            print(f"\n  [Warning] {text}")

        case MSG_TYPE.MSG_TYPE_INFO:
            print(f"  [Info] {text}")
```

---

## 3. The Branching System

### 3.1 Concept

Every message forms a node in a directed acyclic graph. Each node has a single
`parent_id`. A **branch** is the linear path from the root to any leaf node. At any
moment one branch is **active** — this is the sequence of messages the LLM sees as
"the conversation."

### 3.2 What Creates a Branch

| Trigger | How |
|---|---|
| **Regeneration** | `chat(add_user_message=False)` after a user message creates a new assistant response at the same parent node |
| **Explicit fork** | `add_message(parent_id=<any earlier node id>)` |
| **Agentic tool loop** | Does **not** create permanent branches — all tool rounds in one `chat()` call use temporary messages that are deleted when the loop exits |

### 3.3 Navigation API

```python
branch = discussion.get_branch(discussion.active_branch_id)
discussion.switch_branch(branch_tip_id="<uuid>")
result  = discussion.chat("What if we used Redis instead?", branch_tip_id="<uuid>")
```

### 3.4 Temporary Tool-History Messages

During each round of the agentic loop, two **temporary** messages are inserted into the
branch so the model can see its own prior tool calls and their results:

1. `assistant` — the raw LLM output including the `<tool_call>` tag.
2. `user/system` — a `<tool_result name="...">...</tool_result>` block.

These are deleted from the branch immediately after the loop exits. The persisted
conversation contains only the clean final assistant message.

---

## 4. The Artefact System

### 4.1 What Is an Artefact?

An artefact is a **named, versioned, typed document** that lives inside the discussion
but outside the message history. It is stored in `ArtefactManager` (`discussion.artefacts`)
and injected verbatim into the context window. Active artefacts are always visible to the
LLM; inactive ones are excluded from context but remain in the database.

Notes and skills (see §5 and §6) are also stored as artefacts with their own types.

| Field | Description |
|---|---|
| `title` | Human-readable identifier, unique per discussion |
| `type` | `document`, `code`, `note`, `skill`, `image`, `file`, `search_result` |
| `language` | Optional language hint for `code` artefacts |
| `content` | Current text content |
| `version` | Integer incremented on every update |
| `active` | Only `True` artefacts are injected into the context window |
| `description` | Optional free-text (stored, not injected into context) |
| `category` | Optional forward-slash category path — primarily used by `skill` type |
| `author` | Optional author metadata |
| `id` | Stable UUID (survives renames and version bumps) |

### 4.2 LLM Interaction — XML Tags

The LLM communicates with the artefact system through **XML tags** embedded in its
output. Post-processing strips the tags from the displayed message, applies the
operation, and fires `MSG_TYPE_ARTEFACTS_STATE_CHANGED`.

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

If `app.py` already exists, this replaces its content and increments the version.
If it is new, it is created at version 1 with `active=True`.

#### 4.2.2 Fuzzy Title Matching

The framework uses **fuzzy title matching** to locate the right artefact to update even
when the LLM does not reproduce the stored title exactly. This is the mechanism that
prevents the common failure mode of a renamed tag silently creating a duplicate artefact
instead of updating the existing one.

Matching runs in priority order:

1. **Exact match** on the `name` attribute — always preferred when available.
2. **Fuzzy match** — bigram overlap + substring containment + normalised
   casing/extension-stripping, scored in `[0.0, 1.0]`. A match is accepted when the
   score meets the **threshold of 0.60**.
3. **No match** (score < 0.60) — a brand-new artefact is created.

Examples of what fuzzy matching handles transparently:

| Tag name | Existing title | Score | Outcome |
|---|---|---|---|
| `"Réservation Vol Transavia - Tunis → L..."` | `"Réservation Vol - Tunis → Lyon"` | ~0.78 | Update existing |
| `"app"` | `"app.py"` | ~0.88 | Update existing |
| `"MyFile.PY"` | `"myfile.py"` | ~0.95 | Update existing |
| `"README"` | `"requirements.txt"` | ~0.18 | Create new |

When a fuzzy match fires, a log line is emitted:

```
[INFO] Fuzzy title match: 'tag_name' → 'stored_title' (updating in place)
```

The known-title list is refreshed after each artefact tag in the same response, so
multiple tags in one turn do not interfere with each other.

#### 4.2.3 Renaming an Artefact

To change an artefact's title while simultaneously updating its content, use the
`rename` attribute. This is the correct way to rename — it deactivates all versions
under the old title and creates the new version under the new title, preserving the
stable UUID.

```xml
<artefact name="old_flight_note" rename="Réservation Vol Transavia - Tunis → Lyon">
<<<<<<< SEARCH
Airline: unknown
=======
Airline: Transavia
>>>>>>> REPLACE
</artefact>
```

The `rename` attribute can also be used with a full-content replacement:

```xml
<artefact name="draft_v1" rename="Final Report v2">
[new full content here]
</artefact>
```

If `name` matches an existing artefact (exactly or via fuzzy matching) and `rename` is
present, the stored artefact is updated under the new title. A rename without any content
change is not currently supported — include either a SEARCH/REPLACE block or the full
new content.

#### 4.2.4 Patch an Existing Artefact — SEARCH/REPLACE (Aider Format)

The SEARCH/REPLACE format lets the LLM modify specific ranges without re-transmitting
the entire content. Multiple independent blocks may appear inside one `<artefact>` tag.

```xml
<artefact name="app.py">
<<<<<<< SEARCH
def greet():
    return 'hello'
=======
def greet(name: str = 'world'):
    return f'hello {name}'
>>>>>>> REPLACE
<<<<<<< SEARCH
PORT = 8000
=======
PORT = int(os.getenv('PORT', 8000))
>>>>>>> REPLACE
</artefact>
```

**Critical rules for SEARCH/REPLACE patches:**

1. Keep each `SEARCH` block as **short as possible** — 1 to 5 lines is ideal. Longer
   blocks have a higher chance of whitespace or punctuation mismatch.
2. You may include **multiple blocks** inside one tag for several independent edits.
3. Each block **must** follow this exact structure — do not skip or reorder markers:
   ```
   <<<<<<< SEARCH
   [exact lines to find]
   =======
   [replacement lines]
   >>>>>>> REPLACE
   ```
4. The `SEARCH` text must match **character for character** (spaces, indentation, blank lines).
5. Do **not** place the replacement lines where `=======` should be.
6. Do **not** omit the `>>>>>>> REPLACE` marker at the end.

**Fault tolerance** — The framework repairs several common hallucination patterns:

| Hallucination | Repair |
|---|---|
| Missing `>>>>>>> REPLACE` at end | Appended automatically |
| `>>>>>>> REPLACE` used where `=======` should be | Treated as empty replacement (deletion) |
| 6- or 8-chevron variants (`<<<<<<`, `>>>>>>>>`) | Normalised via regex |
| `======` (5 equals) as separator | Accepted |
| CRLF line endings on markers | Normalised to LF |
| `SEARCH` text not found verbatim | `ValueError` with closest-line hint shown |

When a patch fails, the **existing artefact is left untouched**. The error is logged,
and the artefact is still included in `result["artefacts"]` so the UI can surface it.

#### 4.2.5 Revert to a Previous Version

```xml
<revert_artefact name="app.py" version="2" />
```

Fuzzy matching applies here too — the `name` attribute is matched against existing titles
using the same threshold.

```python
# Equivalent Python API
discussion.artefacts.revert("app.py", target_version=2)
```

#### 4.2.6 Image Generation and Editing (TTI)

`enable_image_generation` and `enable_image_editing` default to `True` and are
**automatically disabled** when `lollmsClient.tti is None` — no explicit configuration needed.

```xml
<generate_image width="1024" height="1024">
    A photorealistic sunset over a mountain lake
</generate_image>

<edit_image name="hero_image.png">
    Add a small wooden boat in the foreground
</edit_image>
```

### 4.3 The ArtefactManager Python API

```python
mgr = discussion.artefacts

# List all, or filter by type
all_arts = mgr.list(active_only=False)
notes    = mgr.list(artefact_type="note")
skills   = mgr.list(artefact_type="skill")

# Get one (returns dict or None)
art = mgr.get("app.py")

# Create / replace
mgr.add(title="readme.md", artefact_type="document", content="# My Project", active=True)

# Update content (optionally rename at the same time)
mgr.update("readme.md", new_content="# My Project v2", new_title="README.md")

# Activate / deactivate
mgr.activate("readme.md")
mgr.deactivate("old_draft.md")

# Revert
mgr.revert("app.py", target_version=1)

# Delete permanently
mgr.remove("scratch.txt")
```

#### `update()` signature

```python
mgr.update(
    title,                  # current stored title (exact match)
    new_content=None,       # replace content; None = keep existing
    new_type=None,          # change artefact type
    new_images=None,        # replace images list
    new_tags=None,          # replace tags list
    language=None,          # update language hint
    url=None,               # update URL
    new_title=None,         # rename the artefact (deactivates old title)
    bump_version=True,      # increment version counter
    active=None,            # override active flag (None = inherit)
    **extra_data            # any additional metadata fields
)
```

### 4.4 Real-Time Artefact Events

Every time `_post_process_llm_response` creates or updates an artefact it fires
`MSG_TYPE_ARTEFACTS_STATE_CHANGED` immediately via `self._active_callback`, before the
full response is assembled. See §2.5 for the event payload shapes.

This is implemented through an `event_callback` parameter on `_apply_artefact_xml`.
The post-processor builds a closure over `self._active_callback` and passes it down:

```python
def _artefact_event(artefact: dict, is_new: bool):
    # fires once per processed <artefact> / <revert_artefact> tag
    _active_cb(
        json.dumps({"type": "artefact_created" if is_new else "artefact_updated",
                    "title": artefact["title"], "version": artefact["version"],
                    "art_type": artefact["type"]}),
        MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
        {"artefact": artefact, "is_new": is_new},
    )

self.artefacts._apply_artefact_xml(
    cleaned,
    auto_activate=auto_activate_artefacts,
    replacements=code_blocks,
    event_callback=_artefact_event,   # ← per-artefact real-time callback
)
```

This means a UI can update an individual artefact card as soon as the LLM finishes
streaming that artefact's closing `</artefact>` tag, rather than waiting for the entire
turn to complete. The same mechanism fires for `<revert_artefact>` tags.

---

## 5. Notes

### 5.1 Concept

Notes are **lightweight, named, persistent documents** saved as `ArtefactType.NOTE`
artefacts. They are designed for structured summaries, analysis results, comparison
tables, action item lists, research findings — content the user wants to save and
reference later, but which is neither code nor a large full document.

- **Enable flag:** `enable_notes=True` (default `True`)
- **Artefact type:** `note`
- **Active by default:** yes — notes appear immediately in the artefact panel
- **Versioned:** yes — re-emitting a note with the same title creates a new version

### 5.2 The `<note>` Tag

```xml
<note title="Transavia — Price Analysis">
| Route     | Base | +Baggage | Total |
|-----------|------|----------|-------|
| TUN -> LYS | 89   | 35       | 124   |
| TUN -> CDG | 79   | 35       | 114   |
</note>
```

| Attribute | Required | Description |
|---|---|---|
| `title` | yes | Human-readable name (also accepted as `name=`) |

Multiple `<note>` tags may appear in a single response — each is saved independently.
A note with the same title as an existing one replaces it (new version created).

### 5.3 What Happens During Post-Processing

`_post_process_llm_response` processes `<note>` tags in the same pass as artefacts:

1. Content is extracted and restored from any masked code blocks.
2. `artefacts.add(artefact_type="note", title=..., content=..., active=True)` is called.
3. The tag is stripped from the visible message text.
4. The note artefact is included in `result["artefacts"]` and `MSG_TYPE_ARTEFACTS_STATE_CHANGED` is fired.

### 5.4 Streaming Notification

While the note body is streaming (hidden from the live text), the framework fires:

```python
# MSG_TYPE_CHUNK with text="" and meta:
{"type": "note_start", "content": {"title": "Transavia — Price Analysis"}}
```

A UI can use this to show a "Saving note…" placeholder immediately.

### 5.5 When to Use Notes vs Artefacts

| Use `<note>` for | Use `<artefact>` for |
|---|---|
| Summaries, analysis, comparison tables | Code files, scripts, full documents |
| Key findings from research | Structured data files (JSON, CSV) |
| Action item lists | Anything that needs syntax highlighting |
| Price comparisons, quick reference | Large documents with versioned diffs |
| Short persistent reminders | Anything > ~50 lines of structured content |

### 5.6 Disabling Notes

```python
result = discussion.chat("Compare these flights.", enable_notes=False)
```

When disabled, `<note>` tags are left as literal text (not processed or stripped).

### 5.7 Application-Side Note Parsing Is No Longer Needed

Previously, application code had to parse `<note>` tags from the raw LLM output and
save them to the database manually. This is now handled entirely by the library.
The app just receives note artefacts in `result["artefacts"]` with `type == "note"`.

### 5.8 Notes in `simplified_chat()`

`enable_notes` is also available on `simplified_chat()` with the same default (`True`)
and identical behaviour.

---

## 6. Skills

### 6.1 Concept

Skills are **reusable knowledge capsules** saved as `ArtefactType.SKILL` artefacts.
Unlike notes (which are results of one conversation), skills are intended to be
retrieved and injected into future sessions when relevant — code patterns, workflows,
techniques, domain recipes, language rules.

- **Enable flag:** `enable_skills=False` (default `False` — opt-in)
- **Artefact type:** `skill`
- **Extra metadata:** `description` and `category` stored on the artefact dict
- **Active by default:** yes

Skills default to `False` because they represent intentional knowledge capture. Setting
`enable_skills=True` describes the system to the LLM and enables `<skill>` tag processing.

### 6.2 The `<skill>` Tag

```
<skill title="Python Async HTTP Requests"
       description="Using aiohttp for concurrent HTTP calls"
       category="programming/python/async">
# Async HTTP with aiohttp

Use aiohttp instead of requests when you need concurrent I/O:

    import aiohttp, asyncio

    async def fetch_all(urls: list[str]) -> list[str]:
        async with aiohttp.ClientSession() as session:
            tasks = [session.get(url) for url in urls]
            responses = await asyncio.gather(*tasks)
            return [await r.text() for r in responses]

When to use: any time you have 3 or more independent HTTP calls in the same function.
Performance: typically 5-20x faster than sequential requests.get() calls.
</skill>
```

| Attribute | Required | Description |
|---|---|---|
| `title` | yes | Human-readable skill name |
| `description` | recommended | One-sentence summary of what this teaches |
| `category` | recommended | Forward-slash hierarchical path |

### 6.3 Category Convention

Categories follow a forward-slash hierarchy with 2–4 levels:

```
programming/python/async
programming/javascript/react/hooks
language/french/grammar/subjunctive
cooking/baking/bread/sourdough
devops/docker/networking
data-science/ml/feature-engineering
```

The application layer uses categories to index, search, and auto-inject relevant
skills into future discussions via `personality_data_zone` or data sources.

### 6.4 What Happens During Post-Processing

`_post_process_llm_response` processes `<skill>` tags after notes:

1. Content is extracted and restored from any masked code blocks.
2. `artefacts.add(artefact_type="skill", title=..., content=..., description=..., category=..., active=True)` is called. The `description` and `category` attributes are stored as extra fields on the artefact dict.
3. The tag is stripped from the visible message text.
4. The skill artefact is included in `result["artefacts"]` and `MSG_TYPE_ARTEFACTS_STATE_CHANGED` is fired.

### 6.5 Streaming Notification

```python
# MSG_TYPE_CHUNK with text="" and meta:
{
    "type": "skill_start",
    "content": {
        "title":    "Python Async HTTP Requests",
        "category": "programming/python/async"
    }
}
```

### 6.6 Accessing Skills from the Application Layer

```python
# List all skills in this discussion
skills = discussion.artefacts.list(artefact_type="skill")
for sk in skills:
    print(sk["title"], sk.get("category"), sk.get("description"))

# Filter by category prefix
python_skills = [s for s in skills
                 if s.get("category", "").startswith("programming/python")]
```

### 6.7 Enabling Skills

```python
result = discussion.chat(
    "Teach me this async pattern and save it as a skill.",
    enable_skills=True,
)
# result["artefacts"] will contain the skill artefact(s)
```

### 6.8 Application-Side Skill Parsing Is No Longer Needed

Previously, application code had to parse `<skill>` tags from LLM output and persist
them manually. This is now fully handled by the library. The app receives skill artefacts
in `result["artefacts"]` with `type == "skill"` and `category` / `description` fields.

### 6.9 Skills in `simplified_chat()`

`enable_skills` is also available on `simplified_chat()` with the same default (`False`)
and identical behaviour.

---

## 7. Inline Interactive Teaching Widgets

### 7.1 Concept

Inline widgets let the LLM embed a **live, self-contained interactive learning element**
directly inside a chat message. The primary purpose is *teaching by doing* — giving the
user something to interact with that makes an abstract concept tangible, rather than just
describing it in text.

Unlike artefacts (which go to a side panel), widgets render *in-place* inside the chat
bubble. Unlike notes (which are static text), widgets have controls, animations, and
dynamic output. Unlike artefacts, widgets are **not versioned** — each widget is created
once and stored immutably. If an updated version is needed, ask the LLM to produce a new
widget with a different title.

Enabled by `enable_inline_widgets=True` (the default). No TTI or external tool required.

**When the LLM should use a widget:**
- Physics / math simulations with parameter sliders ("drag the mass slider to see how period changes")
- Algorithm step-through visualisers ("click Next to advance bubble-sort one step")
- Signal / wave / Fourier explorers
- Probability and statistics sandboxes
- Colour / geometry / trigonometry explorers
- Mini quizzes or flashcard drills
- Data-entry forms that compute a result live

**When the LLM should NOT use a widget:**
- Full applications that need persistence → use `<artefact>` instead
- Static charts with no interactivity → describe in text or save as a note
- Anything requiring a server or file I/O

The LLM is also instructed that a **plain-text explanation must accompany every widget**,
telling the user what the widget demonstrates, which controls to interact with, and the
key insight to take away.

### 7.2 Widget Types

Three source formats are supported. The framework normalises any unknown value to `html`.

#### 7.2.1 `type="html"` (default)

A complete, self-contained HTML document. Full access to the DOM, Canvas API, Web Audio,
CSS animations, and any CDN-hosted library. The most flexible type — use this when in doubt.

```xml
<lollms_inline type="html" title="Circle Circumference Explorer">
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>
</head>
<body style="font-family:sans-serif;padding:12px">
  <div id="formula"></div>
  <label>r = <input id="r" type="range" min="0.5" max="10" step="0.01" value="4.5">
    <span id="rv">4.50</span></label>
  <p id="circ"></p>
  <canvas id="c" width="300" height="300"></canvas>
  <script>
    const slider = document.getElementById('r');
    katex.render('C = 2\\pi r', document.getElementById('formula'), {throwOnError:false});
    function draw() {
      const r = parseFloat(slider.value);
      document.getElementById('rv').textContent = r.toFixed(2);
      document.getElementById('circ').textContent =
        'C ≈ ' + (2 * Math.PI * r).toFixed(3);
      const ctx = document.getElementById('c').getContext('2d');
      ctx.clearRect(0, 0, 300, 300);
      ctx.beginPath();
      ctx.arc(150, 150, Math.min(r * 20, 130), 0, 2 * Math.PI);
      ctx.strokeStyle = '#4A9EFF'; ctx.lineWidth = 2; ctx.stroke();
    }
    slider.oninput = draw; draw();
  </script>
</body>
</html>
</lollms_inline>
```

#### 7.2.2 `type="react"`

A JSX component rendered client-side. The source must be a **single default-exported
functional component**. The frontend is responsible for providing a React runtime
(e.g. via a Babel standalone transform or a pre-compiled bundle). The component receives
no props — all state must be internal.

```xml
<lollms_inline type="react" title="Bubble Sort Step-Through">
import { useState } from "react";

export default function BubbleSort() {
  const initial = [5, 3, 8, 1, 9, 2, 7, 4, 6];
  const [arr, setArr]       = useState([...initial]);
  const [i, setI]           = useState(0);
  const [j, setJ]           = useState(0);
  const [done, setDone]     = useState(false);
  const [swapped, setSwapped] = useState(null);

  function step() {
    if (done) return;
    const a = [...arr];
    let ni = i, nj = j;
    if (a[nj] > a[nj + 1]) {
      [a[nj], a[nj + 1]] = [a[nj + 1], a[nj]];
      setSwapped(nj);
    } else {
      setSwapped(null);
    }
    nj++;
    if (nj >= a.length - ni - 1) { nj = 0; ni++; }
    if (ni >= a.length - 1) setDone(true);
    setArr(a); setI(ni); setJ(nj);
  }

  return (
    <div style={{fontFamily:"sans-serif",padding:12}}>
      <div style={{display:"flex",gap:6,marginBottom:12}}>
        {arr.map((v, idx) => (
          <div key={idx} style={{
            width:36, height:36, lineHeight:"36px", textAlign:"center",
            borderRadius:4, fontSize:14,
            background: idx === j || idx === j+1 ? "#4A9EFF" : "#e0e0e0",
            color: idx === j || idx === j+1 ? "#fff" : "#333",
            transition: "background 0.2s",
          }}>{v}</div>
        ))}
      </div>
      <button onClick={step} disabled={done}
        style={{padding:"6px 16px",cursor:"pointer"}}>
        {done ? "Sorted ✓" : "Next Step →"}
      </button>
      {swapped !== null && !done &&
        <span style={{marginLeft:12,color:"#e05"}}>Swapped positions {swapped} ↔ {swapped+1}</span>}
    </div>
  );
}
</lollms_inline>
```

> **React runtime note** — The framework stores the JSX source verbatim. The frontend
> must transpile and mount it. A common pattern is wrapping the source in a Blob URL,
> running it through Babel standalone, and mounting with `ReactDOM.createRoot`.
> The source is guaranteed to be a single default export with no external imports
> (CDN `<script>` tags in the surrounding HTML are acceptable for pure-JS libs, but
> `import` statements in JSX are not supported unless the frontend provides a module map).

#### 7.2.3 `type="svg"`

A self-contained SVG document. Best for geometric diagrams, animated illustrations, and
lightweight interactive drawings. SVG event handlers (`onclick`, `onmouseover`, etc.)
and `<animate>` / `<animateTransform>` elements work fully inside an inline SVG block.
For complex interactivity with state, prefer `type="html"` instead.

```xml
<lollms_inline type="svg" title="Unit Circle — Angle Explorer">
<svg viewBox="-1.4 -1.4 2.8 2.8" width="320" height="320"
     xmlns="http://www.w3.org/2000/svg"
     style="background:#1a1a2e;border-radius:8px">
  <!-- axes -->
  <line x1="-1.3" y1="0" x2="1.3" y2="0" stroke="#555" stroke-width="0.02"/>
  <line x1="0" y1="-1.3" x2="0" y2="1.3" stroke="#555" stroke-width="0.02"/>
  <!-- unit circle -->
  <circle cx="0" cy="0" r="1" fill="none" stroke="#4A9EFF" stroke-width="0.03"/>
  <!-- animated radius pointer -->
  <line id="ptr" x1="0" y1="0" x2="1" y2="0"
        stroke="#FFD700" stroke-width="0.04" stroke-linecap="round">
    <animateTransform attributeName="transform" type="rotate"
      from="0 0 0" to="360 0 0" dur="4s" repeatCount="indefinite"/>
  </line>
  <!-- labels -->
  <text x="1.1"  y="0.05" fill="#aaa" font-size="0.12">1</text>
  <text x="-1.3" y="0.05" fill="#aaa" font-size="0.12">-1</text>
  <text x="0.03" y="-1.1" fill="#aaa" font-size="0.12">i</text>
  <text x="0.03" y="1.25" fill="#aaa" font-size="0.12">-i</text>
</svg>
</lollms_inline>
```

### 7.3 The `<lollms_inline>` Tag — Attribute Reference

| Attribute | Values | Default | Description |
|---|---|---|---|
| `type` | `html`, `react`, `svg` | `html` | Widget source format. Any other value is normalised to `html`. |
| `title` | any string | `"Interactive Widget"` | Displayed above the widget in the UI and used as the widget's human-readable identifier within the turn. |

### 7.4 What Happens During Post-Processing

1. All masked code blocks inside the widget source are restored.
2. Type is normalised (`html` / `react` / `svg`; anything else → `html`).
3. A UUID is generated for this widget instance.
4. Widget entry stored in `ai_message.metadata["inline_widgets"]`:
   ```python
   {
       "id":     "<uuid>",
       "type":   "html",          # normalised
       "title":  "Circle Circumference Explorer",
       "source": "<!DOCTYPE html>...",   # full, unmodified source
   }
   ```
5. The `<lollms_inline>…</lollms_inline>` tag is **replaced** in the stored message
   text with a lightweight anchor:
   ```html
   <lollms_widget id="<uuid>" />
   ```
6. Frontend locates the anchor by `id` and renders the widget in-place.

> **Persistence** — Widget source is stored in `ai_message.metadata["inline_widgets"]`
> which is a JSON column in the `messages` table. It persists across sessions. Reloading
> a discussion and reading `message.metadata["inline_widgets"]` gives back the complete
> list with full source for every widget in that message.

> **No `MSG_TYPE_ARTEFACTS_STATE_CHANGED`** — Unlike artefacts, notes, and skills,
> inline widgets do **not** fire `MSG_TYPE_ARTEFACTS_STATE_CHANGED`. The only event
> they produce is the streaming `inline_widget_start` chunk (§7.5) and their presence
> in `result["ai_message"].metadata["inline_widgets"]` at turn end.

### 7.5 Streaming Notification

While the widget body is being streamed (hidden from the live text stream), the
framework fires a `MSG_TYPE_CHUNK` event with `text=""` and a populated `meta`:

```python
{
    "type":    "inline_widget_start",
    "content": {
        "title":       "Circle Circumference Explorer",
        "widget_type": "html"        # "html" | "react" | "svg"
    }
}
```

A UI can use this to show a "Rendering widget…" placeholder immediately, before the
closing tag is received and the widget is mounted.

### 7.6 Accessing Widgets After the Turn

```python
result = discussion.chat("Explain the unit circle interactively.")

widgets = result["ai_message"].metadata.get("inline_widgets", [])
for w in widgets:
    print(w["id"])      # UUID anchor — matches <lollms_widget id="..."> in message text
    print(w["type"])    # "html" | "react" | "svg"
    print(w["title"])   # human-readable label
    # w["source"] — full widget source code
```

To access widgets from a previously saved message (e.g. on reload):

```python
msg = discussion.get_message(message_id)
widgets = (msg.metadata or {}).get("inline_widgets", [])
```

### 7.7 Frontend Rendering Guide

The framework does not dictate how the frontend renders widgets — that is entirely the
application's responsibility. The contract is:

1. The stored message text contains `<lollms_widget id="<uuid>" />` anchors.
2. `ai_message.metadata["inline_widgets"]` contains a list of `{id, type, title, source}` dicts.
3. The frontend replaces each anchor with the rendered widget.

**Recommended approach by type:**

| Type | Recommended rendering strategy |
|---|---|
| `html` | Mount in a sandboxed `<iframe srcdoc="...">` with `sandbox="allow-scripts"`. This is the safest isolation model. |
| `react` | Transpile source with Babel standalone; mount with `ReactDOM.createRoot` inside a `<div>`. Add error boundary. |
| `svg` | Inject directly into the DOM as an inline `<svg>` element, or use an `<img>` with a Blob URL for isolation. |

**Security note** — Widget source comes from the LLM. Always render in an isolated
context (iframe sandbox or shadow DOM). Never use `eval()` on widget source directly in
the main page context.

### 7.8 Approved CDN Libraries

The LLM is instructed to use CDN links only for well-known libraries. The following are
explicitly supported and produce reliable results:

| Library | CDN URL | Best for |
|---|---|---|
| KaTeX | `https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js` | Math formulas |
| Chart.js | `https://cdn.jsdelivr.net/npm/chart.js` | Bar, line, pie charts |
| D3.js | `https://cdn.jsdelivr.net/npm/d3` | Custom data visualisations |
| Three.js | `https://cdn.jsdelivr.net/npm/three/build/three.min.js` | 3D scenes |
| p5.js | `https://cdn.jsdelivr.net/npm/p5/lib/p5.min.js` | Creative / generative art |
| Plotly | `https://cdn.plot.ly/plotly-latest.min.js` | Scientific / statistical plots |
| Tone.js | `https://cdn.jsdelivr.net/npm/tone` | Audio synthesis / music theory |
| MathJax | `https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js` | LaTeX rendering alternative |

Libraries that require build steps, ES module imports, or backend APIs are not suitable
for inline widgets.

### 7.9 Rules and Constraints

- **Fully self-contained** — CDN links (see §7.8) are the only allowed external references. No fetch/XHR at runtime.
- **Compact** — aim for <= 460 px height, full container width.
- **Label every control** clearly so the user knows what to do.
- **No `alert()` / `confirm()` / `prompt()`.**
- **Prefer smooth transitions / animations** over abrupt jumps.
- **Not for large applications** — use `<artefact>` for anything needing persistence or export.
- **Not versioned** — widgets are immutable once created. A new widget must have a different title.

### 7.10 Teaching-Companion Text Rule

The system prompt instructs the LLM:

> Always write a short explanation in your reply alongside the widget. Tell the user:
> - what the widget demonstrates,
> - which controls to interact with and what to look for,
> - the key insight they should walk away with.

This rule is enforced at the prompt level; the silent artefact guard (§8) provides a
safety net for cases where the LLM ignores it and produces only the widget tag.

### 7.11 Widget vs Artefact Decision Table

| Need | Use |
|---|---|
| Live interactive demo, simulation, explorer | `<lollms_inline>` widget |
| Code the user wants to download or run | `<artefact type="code">` |
| A full multi-file application | `<artefact>` (one per file) |
| A static chart or diagram (no interaction) | Describe in text or `<note>` |
| A document the user wants to edit later | `<artefact type="document">` |
| A visualisation that needs server data | Not possible inline — use `<artefact>` |

### 7.12 Multiple Widgets in One Response

Multiple `<lollms_inline>` tags in a single response are all processed. Each gets its
own UUID and its own entry in `metadata["inline_widgets"]`. The anchors appear in the
stored message text at the positions where the tags originally were, so the widgets
render interleaved with the surrounding prose.

```python
# A response with two widgets
widgets = result["ai_message"].metadata.get("inline_widgets", [])
# len(widgets) == 2
# result["ai_message"].content contains two <lollms_widget id="..."> anchors
```

### 7.13 Disabling Inline Widgets

```python
result = discussion.chat("Explain quicksort.", enable_inline_widgets=False)
```

When disabled, `<lollms_inline>` tags are left as literal text in the message (not
processed, not stripped). The widget system-prompt instructions are also not injected,
so the LLM will not attempt to produce widget tags at all.

### 7.14 Inline Widgets in `simplified_chat()`

`enable_inline_widgets` is available on `simplified_chat()` with the same default
(`True`) and identical post-processing behaviour. The widget system-prompt instructions
are injected when enabled on either method.

---

## 8. The Silent Artefact Guard

### 8.1 Problem

When the LLM response consists **entirely** of XML action tags — `<artefact>`, `<note>`,
`<skill>`, `<lollms_inline>` — the post-processor strips all of them, leaving the stored
message text blank. The user sees an empty chat bubble. This is confusing and looks like
a crash.

### 8.2 Solution

`_post_process_llm_response` includes a **silent artefact guard** that fires after all
tags have been processed. If `cleaned.strip()` is empty, it auto-generates a concise,
human-readable confirmation of what was produced:

```
📄 Created **app.py** (python) [code]: FastAPI entry point.
📝 Saved note **Transavia — Price Analysis** — | Route | Base | +Baggage | Total |…
🎓 Skill saved **Python Async HTTP Requests** [programming/python/async] — Using aiohttp for concurrent HTTP calls.
🎛️ Interactive widget ready: **Circle Circumference Explorer** (html) — use the controls below to explore the concept.
```

The guard is controlled by the `enable_silent_artefact_explanation` parameter (default `True`).

### 8.3 Parameter

| Parameter | Default | Description |
|---|---|---|
| `enable_silent_artefact_explanation` | `True` | Auto-generate a summary message when the LLM response is entirely XML tags |

### 8.4 Format

One line per produced item, in creation order, grouped by type:

- **Artefacts** (code, document, file, …): `📄 Created **title** (language) [type — version N]: description.`
  - Version N is omitted on version 1 (initial creation).
  - Language is omitted when not set.
  - Description is omitted when not set.
- **Notes**: `📝 Saved note **title** — [first line of content, up to 80 chars]…`
- **Skills**: `🎓 Skill saved **title** [category] — description.`
- **Inline widgets**: `🎛️ Interactive widget ready: **title** (type) — use the controls below to explore the concept.`

### 8.5 Disabling

```python
result = discussion.chat(
    "Create a Flask server.",
    enable_silent_artefact_explanation=False,
)
```

When disabled and the LLM produces only XML tags, the stored message content will be
empty. This is only appropriate in application contexts where the frontend handles
empty messages explicitly via `MSG_TYPE_ARTEFACTS_STATE_CHANGED` events.

### 8.6 Relationship to the Teaching-Companion Text Rule

The silent artefact guard and the teaching-companion text rule (§7.6) are complementary:

- The **teaching rule** is a prompt-level instruction that prevents silent responses
  *before* they happen by asking the model to explain its work.
- The **guard** is a post-processing safety net that produces a minimal explanation
  *after* the fact, in case the model ignores the instruction.

Both apply to all XML tag types, not just inline widgets.

---

## 9. The Tooling System

### 9.1 Mechanics

Tools are registered at the start of each `chat()` call and described via an
`## Available Tools` section injected into the system prompt. The LLM calls a tool
by emitting a `<tool_call>` JSON tag anywhere in its streaming output.

```
1.  LLM streams text freely.
2.  Framework detects <tool_call>...</tool_call> mid-stream.
3.  Streaming stops. Pre-tag text forwarded as MSG_TYPE_CHUNK.
4.  JSON is parsed; the tool is dispatched; result serialised (<=2000 chars).
5.  MSG_TYPE_TOOL_CALL and MSG_TYPE_TOOL_OUTPUT events are fired.
6.  Tool output is appended to self.scratchpad.
7.  A temporary <tool_result> message is added to the branch for the next pass.
8.  Generation resumes. Loop continues up to max_reasoning_steps times.
9.  When the LLM produces no more tool_call tags, the loop exits.
10. Temporary messages are deleted; <tool_call> tags are stripped from text.
```

### 9.2 Fast Path — No External Tools

When no external tools, personality tools, or RAG sources are registered, the framework
**skips all agentic scaffolding** and goes directly to a single LLM call. The model may
still emit `<artefact>`, `<note>`, `<skill>`, and `<lollms_inline>` XML tags, which are
post-processed as normal.

This eliminates the runaway-loop problem for straightforward requests (e.g. "analyse
these screenshots and create a note") where no external data is needed.

### 9.3 Anti-Duplication Guards

#### 9.3.1 Agent State Header

At the top of every agentic round, a temporary block is prepended to the scratchpad:

```
=== AGENT STATE (already completed this turn — DO NOT repeat) ===
Tool calls already made:
  v round 1: create_note(title=Transavia Analysis, content=...)
Artefacts / notes already created:
  v Transavia Analysis
=== END AGENT STATE ===
```

Removed immediately after the LLM call so it does not accumulate across rounds.

#### 9.3.2 Hard Deduplication Gate

Before any tool executes, `_identical_call_counts[signature]` is checked:

| Count | Behaviour |
|---|---|
| 1 (first call) | Executes normally |
| 2 (first repeat) | Blocked; synthetic `DUPLICATE CALL BLOCKED` result injected |
| 3+ | Loop breaks unconditionally |

**The same tool with different parameters is always allowed.** Multi-step research
calling the same search tool with six different queries will execute all six.

### 9.4 Registering External Tools

```python
def list_directory(path: str) -> dict:
    import os
    try:
        return {"success": True, "entries": os.listdir(path)}
    except Exception as e:
        return {"success": False, "error": str(e)}

tools = {
    "list_dir": {
        "name":        "list_directory",
        "description": "List files in a directory on the local filesystem",
        "parameters": [{"name": "path", "type": "str", "optional": False}],
        "output":     [{"name": "entries", "type": "list"}],
        "callable":   list_directory,
    }
}

result = discussion.chat("What files are in /tmp?", tools=tools)
```

| Spec field | Type | Description |
|---|---|---|
| `name` | `str` | Tool name used in the `<tool_call>` tag |
| `description` | `str` | Natural-language description injected into the system prompt |
| `parameters` | `list[dict]` | Each dict: `name`, `type`, `optional`, `default`, `description` |
| `output` | `list[dict]` | If any entry has `name="sources"`, tool is treated as a RAG tool |
| `callable` | callable | Must return a `dict` with at least a `success` key |

#### 9.4.1 RAG Tools

Any tool whose `output` spec contains a field named `"sources"` is automatically
registered as a RAG tool. Its results are tracked in `result["sources"]`, emitted
as `MSG_TYPE_SOURCES_LIST` events, and subject to `rag_top_k` / `rag_min_similarity_percent`.

### 9.5 Built-in Framework Tools

All built-in tools default to `True`. The personality tools step event is only emitted
when at least one personality tool is found.

#### 9.5.1 `show_tools` (`enable_show_tools=True`)

Assembles a complete tool catalogue and fires `MSG_TYPE_TOOLS_LIST`.

#### 9.5.2 `extract_artefact_text` (`enable_extract_artefact=True`)

Extracts a line range from an existing artefact and saves it as a new artefact using
text anchors rather than line numbers.

```json
{
  "name": "extract_artefact_text",
  "parameters": {
    "source_title":    "app.py",
    "new_title":       "app.py - router section",
    "start_line_hint": "# -- Router",
    "end_line_hint":   "# -- End Router",
    "occurrence":      1,
    "artefact_type":   "code",
    "language":        "python"
  }
}
```

#### 9.5.3 `final_answer` (`enable_final_answer=True`)

Signal tool for composable-answer workflows.

#### 9.5.4 `request_clarification` (`enable_request_clarification=True`)

Lets the LLM pause and ask the user a question mid-turn.

#### 9.5.5 `deactivate_artefacts` (context-pressure only)

Conditionally registered when `artefact_tokens > history_tokens AND total > budget`.

```json
{"name": "deactivate_artefacts", "parameters": {"titles": ["old_draft.md"]}}
```

### 9.6 Preflight RAG (Personality)

When a personality is provided, `chat()` runs a preflight RAG pass before the first
generation step. Results are injected into `self.scratchpad` and placed after the last
user message by `export()`. Sources appear in `result["sources"]` with `phase="preflight"`.

### 9.7 `simplified_chat()` — Lighter RAG Path

```python
result = discussion.simplified_chat(
    user_message="What are the best indexing strategies for our schema?",
    rag_data_stores={
        "db_docs": lambda q: vector_db.search(q, top_k=5),
        "wiki":    lambda q: wiki_search(q),
    },
    streaming_callback=on_event,
    enable_inline_widgets=True,
    enable_notes=True,
    enable_skills=False,
)
```

Three code paths: **fast path** (trivial message), **memory hit**, **full path**
(intent detection -> optional RAG -> stream final answer).

`simplified_chat()` now accepts the same XML-processing flags as `chat()`:
`enable_inline_widgets`, `enable_notes`, `enable_skills`, and
`enable_silent_artefact_explanation`. The corresponding system-prompt instructions
are injected when each flag is `True`.

---

## 10. REPL Text Tools

### 10.1 Purpose

MCP tools and RAG pipelines can return very large payloads. The REPL text tools solve
this with a **named in-session buffer** — the model stores a large result, gets back
only a compact summary, then navigates it with targeted calls.

**Toggle:** `enable_repl_tools=True` (default). All buffers are ephemeral — they exist
only for the duration of a single `chat()` call. Use `text_to_artefact` to persist.

### 10.2 Format Auto-Detection

| Format | Detection rule |
|---|---|
| `json_array` | Top-level JSON `[{...}, ...]` |
| `jsonl` | >= 3 of the first 10 lines parse as standalone JSON objects |
| `csv` | `csv.Sniffer` detects a consistent delimiter; `DictReader` yields >= 2 fields |
| `md_table` | Pipe-delimited lines with a `|---|` separator row |
| `numbered_list` | >= 5 lines matching `1. ...` / `- ...` / `* ...` patterns |
| `sections` | >= 3 `## Heading` blocks |
| `lines` | Fallback — every non-blank line is one record |

### 10.3 The Nine REPL Tools

| Tool | Signature | Purpose |
|---|---|---|
| `text_store` | `(handle, content)` | Ingest a large payload; returns format, total_records, schema, 3-record preview. **Call first on any large tool output.** |
| `text_search` | `(handle, query, max_results=10, field=None)` | Keyword/regex search across all records or one field. |
| `text_get_range` | `(handle, start, end)` | Return records `[start..end]` inclusive (0-based). |
| `text_get_record` | `(handle, index)` | Return one full record, with per-field soft truncation at 800 chars. |
| `text_list_records` | `(handle, page=1, page_size=20)` | Paginated one-line-per-record listing. |
| `text_filter` | `(handle, field, op, value, new_handle)` | Filter where `field <op> value`; ops: `eq ne gt lt gte lte contains startswith regex` |
| `text_aggregate` | `(handle, operation, field)` | Aggregate: `count sum min max avg unique unique_count` |
| `text_to_artefact` | `(handle, title, artefact_type="document", language="")` | Persist buffer as a discussion artefact. |
| `text_list_buffers` | `()` | List all active handles with format and record count. |

---

## 11. Context Compression

### 11.1 Trigger

Compression runs automatically when `max_context_size` is set and the current token
count exceeds `max_context_size x 0.80`.

### 11.2 Strategy

```
1. Measure current tokens.
2. If <= budget -> fast-path return.
3. Classify: artefact_pressure = (artefact_tokens > history_tokens) AND (total > budget)
4. Cache lookup: key = SHA-1(branch_tip_id + "|" + sorted(active_artefact_ids))
   Cache hit -> re-apply stored summary (no LLM call).
5a. History compression (cache miss, not artefact-heavy):
    - Keep newest max(4, len(branch) // 4) turns intact.
    - LLM summarises the rest (preserving code, filenames, variable names).
    - Append to existing pruning_summary (supports multi-round stacking).
    - Persist to cache (max 10 entries, FIFO).
5b. Artefact pressure -> return artefact_pressure=True
    -> chat() injects the deactivate_artefacts tool.
6. Fire MSG_TYPE_CONTEXT_COMPRESSION.
```

### 11.3 `MSG_TYPE_CONTEXT_COMPRESSION` Meta Fields

| Field | Type | Description |
|---|---|---|
| `tokens_before` | `int` | Token count before compression |
| `tokens_after` | `int` | Token count after |
| `budget` | `int` | `max_context_size x 0.80` |
| `cache_hit` | `bool` | Summary re-applied from cache (no LLM call) |
| `summary_generated` | `bool` | New LLM summary generated |
| `artefact_pressure` | `bool` | Artefacts are dominant pressure source |
| `messages_pruned` | `int` | Messages summarised (`summary_generated` only) |

---

## 12. Scratchpad Placement Model

### 12.1 Why Position Matters

The scratchpad holds the full text of every tool result accumulated during an agentic
turn. Placing it in the system-prompt header (far from the user's question) makes the
model less likely to attend to it. Placing it immediately before the model's continuation
point maximises its impact.

### 12.2 Empty Scratchpad Suppression

An empty scratchpad produces **no output at all** — no blank system message, no empty
block. `export()` checks `_scratchpad.strip()` before injecting anything.

### 12.3 Injection Per Format

| `export()` format | Injection mechanism |
|---|---|
| `lollms_text` | `!@>system:\n== TOOL OUTPUT SCRATCHPAD ==\n...` after the last `!@>user:` block |
| `openai_chat` | `{"role": "system", "content": "== TOOL OUTPUT SCRATCHPAD ==\n..."}` after the last user dict |
| `ollama_chat` | Same structure as `openai_chat` |
| `markdown` | `**system**: == TOOL OUTPUT SCRATCHPAD ==\n...` after the last user line |

### 12.4 Lifecycle Within a Single `chat()` Call

```
chat() starts
  -> self.scratchpad = ""                    # reset
  -> self._active_callback = callback        # stash for real-time artefact events

  [preflight RAG]
  -> self.scratchpad = "source1...source2..."   # written by preflight (if any)

  [agentic loop round N]
  -> _saved = self.scratchpad               # save base
  -> self.scratchpad = AGENT_STATE + base   # prepend state header temporarily
  -> LLM call  (export() injects scratchpad after last user msg)
  -> self.scratchpad = _saved               # restore base
  -> tool executes
  -> self.scratchpad += "--- Tool: X ---\n" # append result

  [forced final-answer pass, if needed]
  -> _saved = self.scratchpad
  -> self.scratchpad = base + "\n[SYSTEM INSTRUCTION] All tool calls complete..."
  -> LLM call
  -> self.scratchpad = _saved

  [loop cleanup]
  -> temporary messages deleted
  -> self.scratchpad = ""                   # cleared before commit
  -> self._active_callback = None           # cleared — must not leak to next turn
```

The scratchpad is **never written to the database**. It is purely a per-turn
communication channel between the tool executor and the LLM.

---

## 13. Source Title Extraction

When RAG chunks arrive without explicit metadata titles, `_extract_content_title()`
derives a human-readable title via regex (no LLM). Priority chain:

1. `chunk["title"]`
2. `chunk["metadata"]["title"]`
3. `chunk["metadata"]["filename"]`
4. `chunk["metadata"]["name"]`
5. `src.rsplit("/", 1)[-1]` — basename of the source path
6. Content analysis (all via regex):
   - Markdown `#` / `##` / `###` heading
   - RST underline (`====`, `----`)
   - YAML/JSON `title:` / `"title":` field
   - HTML `<title>` or `<h1>` tag
   - `**Bold**` or `*italic*` opening phrase
   - First-line heuristic (short, no mid-sentence punctuation)
7. `"Source N"` — last-resort fallback

---

## 14. `chat()` Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `user_message` | — | The user's input text |
| `personality` | `None` | `LollmsPersonality` instance; defaults to `NullPersonality` |
| `branch_tip_id` | `None` | Explicit branch tip; defaults to `active_branch_id` |
| `tools` | `None` | Dict of external tool specs (see §9.4) |
| `swarm` | `None` | List of `Agent` objects. When non-empty, delegates the entire turn to `SwarmOrchestrator` (see §16). |
| `swarm_config` | `None` | `SwarmConfig` instance controlling mode, rounds, anti-sycophancy, etc. Defaults to `SwarmConfig()` if `swarm` is set. |
| `add_user_message` | `True` | If `False`, re-uses the current branch tip as the user message |
| `max_reasoning_steps` | `20` | Maximum agentic loop iterations |
| `images` | `None` | List of base64 image strings for the user message |
| `debug` | `False` | Reserved for future use |
| `remove_thinking_blocks` | `True` | Strip `<think>...</think>` from the final response |
| `enable_image_generation` | `True` | Auto-disabled if `lollmsClient.tti is None` |
| `enable_image_editing` | `True` | Auto-disabled if `lollmsClient.tti is None` |
| `auto_activate_artefacts` | `True` | New artefacts / notes / skills are `active=True` |
| `enable_show_tools` | `True` | Register the `show_tools` built-in |
| `enable_extract_artefact` | `True` | Register the `extract_artefact_text` built-in |
| `enable_final_answer` | `True` | Register the `final_answer` signal built-in |
| `enable_request_clarification` | `True` | Register the `request_clarification` built-in |
| `enable_repl_tools` | `True` | Register the nine REPL text tools |
| `enable_inline_widgets` | `True` | Process `<lollms_inline>` tags; inject teaching widget instructions |
| `enable_notes` | `True` | Process `<note>` tags; save as NOTE artefacts |
| `enable_skills` | `False` | Process `<skill>` tags; save as SKILL artefacts |
| `enable_silent_artefact_explanation` | `True` | Auto-generate a summary message when the LLM response consists entirely of XML tags |
| `streaming_callback` | `None` | Callback function (see §2.1) |
| `decision_temperature` | `0.3` | Temperature for intent-detection / structured calls |
| `final_answer_temperature` | `0.7` | Temperature for the final generation pass |
| `rag_top_k` | `5` | Maximum RAG chunks per query |
| `rag_min_similarity_percent` | `0.5` | Minimum similarity score for RAG results |
| `preflight_rag` | `True` | Run preflight RAG pass when personality has a data source |

> **`simplified_chat()` parameter parity** — `simplified_chat()` accepts
> `enable_inline_widgets`, `enable_notes`, `enable_skills`, and
> `enable_silent_artefact_explanation` with identical defaults and behaviour.
> Parameters specific to the agentic loop (`tools`, `enable_show_tools`,
> `enable_repl_tools`, etc.) are not available on `simplified_chat()`.

---

## 15. `chat()` Return Value Reference

```python
result = discussion.chat(...)
```

| Key | Type | Description |
|---|---|---|
| `user_message` | `LollmsMessage` | The user turn message object |
| `ai_message` | `LollmsMessage` | The final synthesis message (swarm) or assistant turn (single-agent) |
| `sources` | `list[dict]` | All RAG sources collected during this turn |
| `scratchpad` | `dict or None` | Scratchpad state if the turn was agentic, else `None` |
| `self_corrections` | `list or None` | Log of self-correction events |
| `artefacts` | `list[dict]` | Every artefact created or modified this turn, including notes and skills |
| `agent_messages` | `list[LollmsMessage]` | **Swarm only.** One message per agent per round, in order. |
| `hlf_log` | `list[dict]` | **Swarm only.** Full HLF message log (see §16.4). |
| `swarm_meta` | `dict` | **Swarm only.** Run statistics: `mode`, `rounds_run`, `agents`, `touched_artefacts`, `duration_seconds`. |

`result["sources"]` items:

| Field | Description |
|---|---|
| `title` | Human-readable title (after extraction chain) |
| `content` | Chunk text |
| `source` | Raw source identifier / path |
| `query` | The query that retrieved this chunk |
| `relevance_score` | Float 0–100 |
| `index` | Sequential index within this turn |
| `tool` | Tool that produced this source |
| `phase` | `"preflight"` or `"agentic"` |

`result["ai_message"].metadata` fields:

| Field | Description |
|---|---|
| `mode` | `"direct"` / `"agentic"` / `"rlm_agentic"` |
| `duration_seconds` | Wall-clock time for the full turn |
| `token_count` | Tokens in the final response |
| `tokens_per_second` | Generation speed |
| `tool_calls` | `[{name, params, result}]` for every tool call |
| `events` | Full event log with `{type, content, id, tool, offset, status}` |
| `sources` | Same as `result["sources"]` |
| `query_history` | `[{step, tool, query, result_count}]` for every RAG query |
| `scratchpad` | Scratchpad state dict (agentic only) |
| `self_corrections` | Self-correction log (agentic only) |
| `artefacts_modified` | Titles of artefacts changed this turn — includes notes and skills |
| `inline_widgets` | `[{id, type, title, source}]` — populated when `<lollms_inline>` tags present |
| `personality_tools_used` | Names of personality tools actually called this turn |
| `swarm_mode` | **Synthesis message only.** Swarm mode string (e.g. `"quality"`) |
| `swarm_synthesis` | **Synthesis message only.** Always `True` — marks this as the synthesiser's output |
| `swarm_agents` | **Synthesis message only.** List of all agent display names |
| `swarm_rounds` | **Synthesis message only.** `max_rounds` value from config |
| `swarm_round` | **Per-agent messages.** Which round this agent spoke in |
| `swarm_role` | **Per-agent messages.** The agent's role string |
| `swarm_agent` | **Per-agent messages.** The agent's display name |
| `agent_id` | **Per-agent messages.** The agent's stable UUID |

---

## 16. The Swarm System

### 16.1 Concept and Use Cases

The swarm system turns a single `chat()` call into a **multi-agent deliberation session**.
Instead of one model answering the user, a team of `Agent` instances each contribute their
own perspective across multiple rounds, collaborate on shared artefacts, and converge on a
synthesised answer produced by a designated moderator.

**Default use case — quality enhancement:** the most common reason to use a swarm is to
get a better answer by leveraging the diversity of multiple models, personas, or knowledge
bases. A proposer drafts, a critic stress-tests, a domain expert adds specifics, and a
synthesizer distils the best of all three.

**Other use cases:**

| Use case | Swarm mode | What happens |
|---|---|---|
| Adversarial review | `quality` + `anti_sycophancy="strong"` | Agents actively try to break each other's proposals |
| Structured debate | `debate` | Two sides argue for/against; moderator judges |
| Collaborative coding | `quality` with `IMPLEMENTER` + `TESTER` roles | One agent writes code, another reviews for bugs |
| Simulated world | `simulation` | Agents play characters; world state persists across turns |
| Game playing | `game` | Agents play card games, strategy games, etc. |
| Creative writing | `freeform` | Multiple "character" agents build a story together |

The swarm shares the **same discussion object** as single-agent turns, so artefacts,
notes, memory, and message history are all continuous across swarm and non-swarm turns.

### 16.2 The Agent

`Agent` is a dataclass defined in `lollms_agent.py`. Import it:

```python
from lollms_client.lollms_agent import Agent, AgentRole
```

| Field | Type | Default | Description |
|---|---|---|---|
| `lc` | `LollmsClient` | required | The LLM binding this agent uses. Multiple agents may share one binding or use different ones. |
| `personality` | `LollmsPersonality` | required | Defines the agent's system prompt and knowledge base. The personality's `name` is the default display name. |
| `name` | `str \| None` | `None` | Display name override. Falls back to `personality.name`. |
| `role` | `str` | `"freeform"` | Semantic role — see §16.3. |
| `tools` | `dict \| None` | `None` | External tool specs in the same format as `chat(tools=...)`. Currently stored on the agent; future versions will pass them per-agent to the orchestrator. |
| `model_params` | `dict` | `{}` | Extra kwargs forwarded to every generation call (e.g. `{"temperature": 0.8}`). |
| `max_tokens_per_turn` | `int` | `1024` | Soft cap on tokens generated per round. |
| `metadata` | `dict` | `{}` | Application-layer metadata (avatar URL, colour, etc.). Not used by the orchestrator. |

```python
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_personality import LollmsPersonality

architect = Agent(
    lc          = lc_local,
    personality = LollmsPersonality(name="Architect",
                    system_prompt="You are a senior distributed systems architect."),
    role        = AgentRole.PROPOSER,
    model_params= {"temperature": 0.8},
    max_tokens_per_turn = 600,
)
```

### 16.3 AgentRole Reference

| Constant | String value | Orchestrator behaviour |
|---|---|---|
| `AgentRole.PROPOSER` | `"proposer"` | Introduces bold, concrete ideas. Round 1 focus. |
| `AgentRole.CRITIC` | `"critic"` | Finds weaknesses, missing cases, unstated assumptions. |
| `AgentRole.DEVIL_ADVOCATE` | `"devil_advocate"` | Argues the opposite position as forcefully as possible. |
| `AgentRole.DOMAIN_EXPERT` | `"domain_expert"` | Brings deep specialist knowledge. |
| `AgentRole.SYNTHESIZER` | `"synthesizer"` | Integrates all perspectives into a coherent conclusion. |
| `AgentRole.MODERATOR` | `"moderator"` | Keeps discussion productive; asks clarifying questions. |
| `AgentRole.IMPLEMENTER` | `"implementer"` | Turns plans into working artefacts (code, documents). |
| `AgentRole.TESTER` | `"tester"` | Probes artefacts for bugs and edge cases. |
| `AgentRole.NARRATOR` | `"narrator"` | Describes world state in simulation / game modes. |
| `AgentRole.PLAYER` | `"player"` | Plays a character in a simulation or game. |
| `AgentRole.FREEFORM` | `"freeform"` | No prescribed role; acts on its own judgment. |

Custom role strings are accepted — the orchestrator uses them as labels but will not apply
the built-in hint for unknown values.

### 16.4 The HLF Protocol

**HLF (High-Level Format)** is the structured inter-agent communication layer. It runs
in parallel with the NLP (visible) layer and is **never shown directly to the user**
unless the application explicitly reads `result["hlf_log"]`.

After each agent finishes its NLP turn, the orchestrator makes a small structured call
asking the agent to self-classify its contribution:

```python
# Schema of the HLF self-assessment call
{
    "type":       "One of: proposal, critique, question, answer, game_action, narration",
    "to":         "Target agent name or 'all'",
    "artefact":   "Artefact title if relevant, else empty string",
    "confidence": "Float 0.0–1.0",
    "summary":    "One sentence summary for the HLF log",
}
```

The resulting `HLFMessage` is stored in `orchestrator.hlf_log` and in
`result["hlf_log"]`. The fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Short UUID (8 chars) |
| `from` | `str` | Agent display name |
| `to` | `str` | Target agent name or `"all"` |
| `type` | `str` | HLF type (see table below) |
| `round` | `int` | Which round this was generated in |
| `content` | `str` | One-sentence summary of the contribution |
| `artefact_ref` | `str \| None` | Title of artefact being referenced |
| `confidence` | `float` | Self-reported 0.0–1.0 |
| `ts` | `str` | UTC ISO timestamp |

**HLF message types:**

| Type | When used |
|---|---|
| `proposal` | Agent introduces a new idea or initial draft |
| `critique` | Agent challenges a specific point from another agent |
| `question` | Agent asks another agent for clarification |
| `answer` | Agent responds to a question |
| `vote` | Agent votes on a proposal (debate / game modes) |
| `artefact_patch` | Agent signals intent to patch a shared artefact |
| `directive` | Orchestrator steers all agents (injected by framework) |
| `steer` | User steering directive (injected via `orchestrator.steer()`) |
| `synthesis` | Final answer being assembled |
| `game_action` | A player's move in a game or simulation |
| `narration` | Narrator describes world state |

**Confidence scores** drive convergence detection. If mean confidence ≥
`convergence_threshold` AND there are no outstanding `critique`-type messages, the
orchestrator exits the loop early.

**HLF context injection:** at the start of each agent's turn, the HLF messages from the
current and previous round addressed to that agent (or to `"all"`) are rendered as:

```
=== INTER-AGENT MESSAGES (HLF) ===
[CRITIQUE] → Architect [conf=0.72]
SecurityExpert: The proposed cache design has no TTL invalidation strategy — stale reads are likely under partition.
=== END HLF ===
```

This is injected into the agent's prompt context without appearing in the user-facing stream.

### 16.5 SwarmConfig Reference

```python
from lollms_client.lollms_swarm import SwarmConfig
```

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"quality"` | Session mode. One of `"quality"`, `"debate"`, `"simulation"`, `"game"`, `"freeform"`. |
| `max_rounds` | `int` | `3` | Maximum deliberation rounds. Each round has every agent speak once. Range 1–20. |
| `convergence_threshold` | `float` | `0.85` | Mean confidence required for early stopping. Set to `1.0` to disable. |
| `show_deliberation` | `bool` | `True` | Stream each agent's NLP output to the user in real time. If `False`, only the final synthesis is shown. |
| `moderator_index` | `int` | `0` | Index of the agent (in the swarm list) that produces the final synthesis. |
| `allow_artefact_collaboration` | `bool` | `True` | All agents share the discussion's artefact store. |
| `user_steer_prefix` | `str` | `"⚡ USER DIRECTIVE:"` | Prefix prepended to user steering messages so agents recognise them. |
| `synthesis_prompt_suffix` | `str` | `""` | Extra instructions appended to the synthesiser's final-round prompt. |
| `game_rules` | `str` | `""` | For `"game"` mode — rule description injected into every agent's system prompt. |
| `world_state` | `str` | `""` | For `"simulation"` mode — initial world description injected into every agent's system prompt. |
| `max_nlp_tokens_per_agent` | `int` | `512` | Soft cap on visible NLP output per agent per round. |
| `anti_sycophancy_strength` | `str` | `"medium"` | One of `"light"`, `"medium"`, `"strong"`. Controls how aggressively agents are instructed to disagree. |

### 16.6 Swarm Modes

#### `"quality"` (default)

The most useful mode for everyday tasks. Optimises output quality through diverse
perspectives and structured critique.

Flow: **Round 1** — each agent independently proposes their angle. **Rounds 2+** —
agents read all prior contributions and critique, extend, or challenge them. **Synthesis**
— moderator integrates the best ideas, acknowledges unresolved disagreements.

Best setup: mix `PROPOSER`, `CRITIC`, `DOMAIN_EXPERT`, `SYNTHESIZER` roles.

#### `"debate"`

Two groups argue opposing positions. The moderator judges which side made the stronger case.

Best setup: half the agents as `PROPOSER` arguing one side, half as `DEVIL_ADVOCATE`
arguing the other, one `MODERATOR` at the end of the list.

#### `"simulation"`

Agents play named characters in a persistent fictional world. The world state is injected
via `world_state` and updated in the synthesis each round. The user can steer events.

Best setup: `PLAYER` agents plus one `NARRATOR`. Set `world_state` with an initial scene.

#### `"game"`

A structured game with explicit rules. Rules are injected via `game_rules`. The synthesis
each round describes the game state and sets up the next turn.

Best setup: `PLAYER` agents. The `game_rules` field is critical — without it the agents
will invent their own rules inconsistently.

#### `"freeform"`

No prescribed structure. Agents decide their own dynamics. The synthesiser integrates
whatever emerges. Best for open-ended creative or exploratory tasks.

### 16.7 Anti-Sycophancy System

Automatic agreement is the primary quality killer in multi-agent systems. The swarm
enforces substantive disagreement at the prompt level.

Anti-sycophancy rules are injected starting from **round 2** (there is nothing to agree
with in round 1). Three strengths are available:

#### `"light"`
```
When reviewing other agents' contributions, add genuinely new value.
Brief acknowledgement of agreement is fine, but always extend or qualify.
```

#### `"medium"` (default)
```
ANTI-SYCOPHANCY RULE: You MUST NOT simply confirm or rephrase what a previous
agent said. If you agree on a specific point, state it in one sentence then
IMMEDIATELY add something new: a different angle, an edge case, a counter-example,
a risk, or a concrete improvement. Generic praise ('great point!') is forbidden.
Your confidence score must reflect genuine uncertainty — use values below 0.7
whenever you are not fully certain.
```

#### `"strong"`
```
ADVERSARIAL REVIEW RULE: Before accepting any claim from another agent you must
actively try to find a flaw, limitation, or missing consideration. Only after
you have articulated what could go wrong may you indicate partial agreement.
If you cannot find a flaw, say so explicitly — do not pretend to disagree.
Your confidence score must be ≤ 0.6 on your first pass for any proposal
not yet stress-tested.
```

Use `"strong"` for high-stakes design decisions, security reviews, or when you want to
stress-test an idea as aggressively as possible. Avoid it for creative or simulation modes
where it will break immersion.

### 16.8 Execution Flow

```
chat(swarm=[...], swarm_config=...) called
  → user message added to discussion
  → SwarmOrchestrator.run() called

  [Optional] Inject game_rules or world_state directive into HLF log (round 0)

  MSG_TYPE_SWARM_ROUND_START  {mode, agents}

  FOR round in 1..max_rounds:
    MSG_TYPE_SWARM_ROUND_START  {round}

    FOR each agent in swarm list:
      Build system_prompt  (personality + swarm context + role hint + anti-sycophancy)
      Build hlf_context    (HLF messages from this and previous round, addressed to this agent)
      Build nlp_prompt     (user question + prior NLP contributions + hlf_context + role instruction)

      MSG_TYPE_SWARM_AGENT_START  {agent, role, round}
      agent.generate(nlp_prompt, streaming_callback)   → stream NLP to user
      discussion.add_message(sender=agent.display_name, content=nlp_text, ...)
      _post_process_llm_response(nlp_text, ...)         → artefacts, notes, widgets
      agent.generate_structured(nlp_text → HLF schema)  → HLFMessage
      MSG_TYPE_SWARM_HLF  {hlf message}
      MSG_TYPE_SWARM_AGENT_END  {agent, round}

    Compute mean_confidence from this round's HLF messages
    MSG_TYPE_SWARM_ROUND_END  {round, mean_confidence}

    IF mean_confidence >= threshold AND no unresolved critiques AND round >= 2:
      MSG_TYPE_SWARM_CONSENSUS  {round, confidence}
      BREAK

  [Synthesis pass]
  moderator.generate(synthesis_prompt)                  → stream final NLP to user
  _post_process_llm_response(synthesis_text, ...)        → artefacts, notes, widgets
  discussion.add_message(sender=moderator.display_name, metadata={swarm_synthesis:True})

  return {user_message, ai_message, agent_messages, hlf_log, artefacts, swarm_meta}
```

Every agent's visible NLP output is saved as a permanent `LollmsMessage` in the
discussion, so the full deliberation is preserved in the database and visible in the
chat history.

### 16.9 Shared Artefact Collaboration

All agents share the same `discussion.artefacts` store. Any agent may:
- Create a new artefact via `<artefact name="...">` in its NLP output
- Patch an existing artefact via SEARCH/REPLACE blocks
- Revert via `<revert_artefact>`
- Save notes via `<note>`
- Embed inline widgets via `<lollms_inline>`

The fuzzy title matching (§4.2.2) applies across all agents. If the `IMPLEMENTER` creates
`server.py` in round 1 and the `TESTER` patches it in round 2, fuzzy matching ensures the
patch targets the right artefact even if the tester spells the name slightly differently.

All artefacts created or modified during the swarm are collected in
`result["artefacts"]` and listed in `result["swarm_meta"]["touched_artefacts"]`.

Real-time `MSG_TYPE_ARTEFACTS_STATE_CHANGED` events fire per-artefact as each agent's
output is post-processed, giving the UI an immediate update.

### 16.10 User Steering

The user (or application layer) can inject a directive mid-swarm that all agents will
receive at the start of the next round:

```python
# Direct access to the orchestrator (only during an active run)
orchestrator.steer("Focus only on the security implications from here on.")
```

For application-level steering across turns (i.e. steering the *next* chat() call),
simply pass a different `user_message`. For mid-run steering during a simulation or
long game, call `orchestrator.steer()` from a separate thread or async task while
`chat()` is running.

Steered directives appear in the HLF log as `type="steer"` from `"orchestrator"`, and
are rendered in each agent's context as:

```
⚡ USER DIRECTIVE: Focus only on the security implications from here on.
```

### 16.11 Swarm Events (MSG_TYPE)

The six swarm-specific events are added to `MSG_TYPE` at import time by `lollms_swarm.py`.
They extend the existing enum dynamically; no change to `lollms_types.py` is required.

```python
from lollms_client.lollms_types import MSG_TYPE

def cb(text, msg_type, meta):
    match msg_type:
        # ── Swarm events ─────────────────────────────────────────
        case MSG_TYPE.MSG_TYPE_SWARM_ROUND_START:
            if "mode" in meta:
                print(f"\n{'='*50}")
                print(f"SWARM  mode={meta['mode']}  agents={meta['agents']}")
            else:
                print(f"\n--- {text} ---")

        case MSG_TYPE.MSG_TYPE_SWARM_ROUND_END:
            conf = meta.get("mean_confidence", 0)
            print(f"\n[Round done] mean_confidence={conf:.2f}")

        case MSG_TYPE.MSG_TYPE_SWARM_AGENT_START:
            print(f"\n\n**{meta['agent']}** ({meta['role']}, round {meta['round']})")

        case MSG_TYPE.MSG_TYPE_SWARM_AGENT_END:
            pass   # optional: close agent bubble in UI

        case MSG_TYPE.MSG_TYPE_SWARM_HLF:
            # Raw HLF message — log it or update a debug panel
            hlf = meta.get("hlf", {})
            print(f"  [HLF] {hlf.get('from')} → {hlf.get('to')}: "
                  f"{hlf.get('type')} conf={hlf.get('confidence', 1):.2f}")

        case MSG_TYPE.MSG_TYPE_SWARM_CONSENSUS:
            print(f"\n✓ Consensus after {meta['round']} rounds "
                  f"(confidence={meta['confidence']:.2f})")

        # ── All existing events still fire normally ───────────────
        case MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            if "artefact" in meta:
                op = "created" if meta["is_new"] else "updated"
                print(f"\n  [Artefact {op}] {meta['artefact']['title']}")
```

### 16.12 Result Dict (swarm turn)

A swarm turn returns everything a single-agent turn returns, plus three extra keys:

```python
result = discussion.chat("...", swarm=[...])

# Standard keys (same as single-agent)
result["user_message"]   # LollmsMessage — the user's input
result["ai_message"]     # LollmsMessage — the synthesiser's final output
result["artefacts"]      # list[dict]    — all artefacts modified this turn
result["sources"]        # list[dict]    — empty in most swarm modes

# Swarm-only keys
result["agent_messages"] # list[LollmsMessage] — one per agent per round, in order
result["hlf_log"]        # list[dict] — full HLF message history
result["swarm_meta"]     # dict — run statistics

# swarm_meta fields:
# {
#   "mode":              "quality",
#   "rounds_run":        2,
#   "agents":            ["Architect", "SecurityExpert", "PerfEngineer"],
#   "touched_artefacts": ["server.py", "Architecture Review"],
#   "duration_seconds":  47.3,
# }
```

### 16.13 Message Metadata (swarm turn)

Per-agent messages (from `result["agent_messages"]`) carry:

```python
msg.metadata == {
    "swarm_round":  2,              # which round
    "swarm_role":   "critic",       # agent role
    "swarm_agent":  "SecurityExpert",
    "agent_id":     "a3f1b9c2",     # stable UUID prefix
}
```

The synthesis message (from `result["ai_message"]`) carries:

```python
msg.metadata == {
    "swarm_mode":      "quality",
    "swarm_synthesis": True,
    "swarm_agents":    ["Architect", "SecurityExpert", "PerfEngineer"],
    "swarm_rounds":    3,
    "artefacts_modified": ["server.py"],
}
```

### 16.14 Worked Examples

#### Quality Enhancement — Architecture Review

```python
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_swarm import SwarmConfig
from lollms_client.lollms_personality import LollmsPersonality

# Build agents — can share the same LC or use different ones
def make_agent(name, prompt, role, lc_instance, temp=0.75):
    return Agent(
        lc=lc_instance,
        personality=LollmsPersonality(name=name, system_prompt=prompt),
        role=role,
        model_params={"temperature": temp},
        max_tokens_per_turn=600,
    )

agents = [
    make_agent("Architect",
               "You are a senior distributed systems architect. Propose concrete designs.",
               AgentRole.PROPOSER, lc),
    make_agent("SecurityExpert",
               "You are a security engineer. Focus on attack surface, auth, and data exposure.",
               AgentRole.CRITIC, lc),
    make_agent("PerfEngineer",
               "You are a performance engineer. Focus on latency, throughput, and resource usage.",
               AgentRole.DOMAIN_EXPERT, lc),
    make_agent("TechLead",
               "You integrate all perspectives into a pragmatic, actionable recommendation.",
               AgentRole.SYNTHESIZER, lc, temp=0.5),
]

result = discussion.chat(
    "Design a fault-tolerant distributed cache for our e-commerce checkout service.",
    swarm=agents,
    swarm_config=SwarmConfig(
        mode="quality",
        max_rounds=3,
        anti_sycophancy_strength="medium",
        show_deliberation=True,
        moderator_index=3,   # TechLead synthesises
    ),
    streaming_callback=cb,
)

print("\nArtefacts produced:", result["swarm_meta"]["touched_artefacts"])
print("Rounds:", result["swarm_meta"]["rounds_run"])
print("Duration:", result["swarm_meta"]["duration_seconds"], "s")
```

#### Simulation — Persistent World

```python
EUROPA_STATE = """
Year 2087. Europa Research Station Kappa-7.
Crew of three: Commander Chen, Engineer Volkov, Scientist Park.
Power at 73%. Comms blackout for next 6 hours.
Outside temperature: -170°C. Ice drilling at 2.3 km depth.
Last entry: Drilling sensors picked up anomalous sonar returns at 2.1 km.
"""

sim_agents = [
    make_agent("Commander Chen",
               "You are Commander Chen, methodical, cautious, responsible for crew safety.",
               AgentRole.PLAYER, lc),
    make_agent("Engineer Volkov",
               "You are Engineer Volkov, pragmatic, resourceful, keeps the station running.",
               AgentRole.PLAYER, lc),
    make_agent("Scientist Park",
               "You are Scientist Park, curious, excited by the anomaly, sometimes reckless.",
               AgentRole.PLAYER, lc),
    make_agent("Station AI",
               "You are the station's AI narrator. Describe events, update world state, "
               "introduce complications. Stay consistent with established facts.",
               AgentRole.NARRATOR, lc, temp=0.9),
]

result = discussion.chat(
    "Begin the morning shift. The drilling sensors show the anomaly is moving.",
    swarm=sim_agents,
    swarm_config=SwarmConfig(
        mode="simulation",
        world_state=EUROPA_STATE,
        max_rounds=2,
        show_deliberation=True,
        moderator_index=3,      # Station AI narrates the final state
        synthesis_prompt_suffix=(
            "End your narration by updating the world state and posing "
            "one decision point for the crew."
        ),
    ),
    streaming_callback=cb,
)

# Steer mid-turn (call from a UI button, for example):
# orchestrator.steer("An alarm sounds — pressure drop in Module C.")
```

#### Collaborative Coding — Implementer + Tester

```python
dev_agents = [
    make_agent("Dev",
               "You are a Python developer. Write clean, well-documented code. "
               "Use <artefact> tags to create and update files.",
               AgentRole.IMPLEMENTER, lc),
    make_agent("Reviewer",
               "You are a senior code reviewer. Find bugs, missing edge cases, "
               "and security issues. Propose fixes using SEARCH/REPLACE patches.",
               AgentRole.TESTER, lc),
]

result = discussion.chat(
    "Implement a thread-safe LRU cache in Python with a max-size limit and TTL per key.",
    swarm=dev_agents,
    swarm_config=SwarmConfig(
        mode="quality",
        max_rounds=3,
        anti_sycophancy_strength="strong",  # Reviewer must find real problems
        moderator_index=0,    # Dev produces the final synthesised implementation
        synthesis_prompt_suffix=(
            "Produce the final, corrected implementation incorporating all review feedback. "
            "Update the artefact with the complete file."
        ),
    ),
    streaming_callback=cb,
    enable_notes=True,
)

# After the run, the artefact store contains the collaboratively refined implementation
impl = discussion.artefacts.get("lru_cache.py")
print(impl["content"])
print(f"Version: {impl['version']}")  # will be > 1 after tester patches
```

---

## 17. Worked Example (single-agent)

Full turn: external tool, REPL text tools, notes, skills, artefact, inline teaching
widget, and event callback including real-time artefact events.

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


# -- External tool -------------------------------------------------------------

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


# -- Callback ------------------------------------------------------------------

def cb(text, msg_type, meta):
    match msg_type:
        case MSG_TYPE.MSG_TYPE_CHUNK:
            t = meta.get("type")
            if   not text and t == "note_start":
                print(f"\n  [Note] {meta['content']['title']}")
            elif not text and t == "skill_start":
                c = meta["content"]
                print(f"\n  [Skill] {c['title']}  [{c['category']}]")
            elif not text and t == "inline_widget_start":
                c = meta["content"]
                print(f"\n  [Widget] {c['title']} ({c['widget_type']})")
            elif not text and t == "artefact_update":
                print(f"\n  [Artefact streaming] {meta['content']['title']}")
            else:
                print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n> {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"< {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n[Tool] {meta['tool']}")

        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  [Result] {text[:80]}")

        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            # Per-artefact real-time event (fires immediately as each tag is processed)
            if "artefact" in meta:
                op = "created" if meta["is_new"] else "updated"
                art = meta["artefact"]
                print(f"\n  [Artefact {op}] {art['title']} v{art['version']}")
            # Batch end-of-turn event
            else:
                action = meta.get("action", "updated")
                print(f"\n[Artefacts {action}] {text}")

        case MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION:
            print(f"\n[Compression] {meta['tokens_before']:,} -> {meta['tokens_after']:,}")


# -- Chat ----------------------------------------------------------------------

result = discussion.chat(
    "Search arXiv for papers on LLM context compression (2023-2024). "
    "Filter to those published after 2023, summarise the top 5, "
    "save a structured report as an artefact, "
    "save a comparison table as a note, "
    "save the search pattern as a skill, "
    "and embed an interactive bar chart of publication counts per year.",
    tools=tools,
    streaming_callback=cb,
    enable_repl_tools=True,
    enable_extract_artefact=True,
    enable_inline_widgets=True,
    enable_notes=True,
    enable_skills=True,
    enable_silent_artefact_explanation=True,
    max_reasoning_steps=15,
)

# Inspect results
artefacts = result["artefacts"]
notes  = [a for a in artefacts if a["type"] == "note"]
skills = [a for a in artefacts if a["type"] == "skill"]
docs   = [a for a in artefacts if a["type"] not in ("note", "skill", "image")]

print("\n\nDocuments:", [a["title"] for a in docs])
print("Notes:    ", [a["title"] for a in notes])
print("Skills:   ", [(a["title"], a.get("category")) for a in skills])
print("Widgets:  ", [w["title"] for w in
                     result["ai_message"].metadata.get("inline_widgets", [])])
print("Sources:  ", [s["title"] for s in result["sources"]])
print("Mode:     ", result["ai_message"].metadata["mode"])
print("Duration: ", result["ai_message"].metadata["duration_seconds"], "s")
```

During this turn the LLM will typically:

1. Call `search_arxiv(query="LLM context compression", max_results=30)` — large JSON result.
2. Call `text_store(handle="papers", content=<r>)` — compact summary enters context.
3. Call `text_filter(handle="papers", field="year", op="gt", value=2023, new_handle="recent")`.
4. Call `text_list_records` / `text_get_record` to browse and read abstracts of top papers.
5. Call `text_to_artefact(handle="recent", title="LLM Context Compression Papers 2024")`.
6. Emit `<note title="Top 5 Summary">...</note>` — saved as a NOTE artefact, stripped from text. Real-time `MSG_TYPE_ARTEFACTS_STATE_CHANGED` fires with `is_new=True`.
7. Emit `<skill title="arXiv Research Workflow" category="research/workflow" description="...">...</skill>` — saved as SKILL artefact. Real-time event fires.
8. Emit `<artefact name="context_compression_report.md" type="document">...</artefact>` — saved as DOCUMENT artefact. Real-time event fires with `is_new=True` (or `False` if it already existed, in which case the title is found via fuzzy matching).
9. Emit `<lollms_inline type="html" title="Publication Counts">...</lollms_inline>` — bar chart rendered live in the chat bubble.
10. If the LLM emits only the artefact/note/skill/widget tags with no surrounding text, the silent artefact guard produces a summary line for each item so the message bubble is never empty.