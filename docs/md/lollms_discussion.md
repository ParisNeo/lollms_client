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
   - 2.4.1 [Unified Processing Tag Protocol](#241-unified-processing-tag-protocol-new--recommended)
3. [The Branching System](#3-the-branching-system)
   - 3.1 Concept
   - 3.2 What Creates a Branch
   - 3.3 Data Structures — BranchInfo and MessageNode
   - 3.4 Discovery API
   - 3.5 Navigation API
   - 3.6 Forking
   - 3.7 Deletion
   - 3.8 Merging
   - 3.9 Labelling and Diffing
   - 3.10 Temporary Tool-History Messages
   - 3.11 Complete Branch Management Example
4. [The Artefact System](#4-the-artefact-system)
   - 4.5 [Artefact Images](#45-artefact-images--embedding-visual-content-in-documents)
5. [Notes](#5-notes)
6. [Skills](#6-skills)
7. [Inline Interactive Teaching Widgets](#7-inline-interactive-teaching-widgets)
8. [Interactive Forms](#8-interactive-forms)
9. [The Silent Artefact Guard](#9-the-silent-artefact-guard)
10. [The Tooling System](#10-the-tooling-system)
11. [The Streaming State Machine](#11-the-streaming-state-machine)
12. [REPL Text Tools](#12-repl-text-tools)
13. [Context Compression](#13-context-compression)
14. [Scratchpad Placement Model](#14-scratchpad-placement-model)
15. [Source Title Extraction](#15-source-title-extraction)
16. [`chat()` Parameter Reference](#16-chat-parameter-reference)
17. [`chat()` Return Value Reference](#17-chat-return-value-reference)
18. [The Swarm System](#18-the-swarm-system)
19. [UI Implementation Guide](#20-ui-implementation-guide)
20. [Migration Guide](#21-migration-guide-legacy-to-unified-streaming)
21. [Worked Example (single-agent)](#22-worked-example-single-agent)

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
| 46 | `MSG_TYPE_FORM_READY` | Complete parsed form descriptor ready for rendering | `{title, form}` |
| 47 | `MSG_TYPE_FORM_SUBMITTED` | User answers injected back into generation context | `{form_id, answers}` |

> **Note on Processing Events**: The unified `<processing>` tag protocol (§2.4.1)
> delivers all secondary content through `MSG_TYPE_CHUNK` with special `meta` dicts:
> - `meta["type"]`: `"processing_open"`, `"processing_status"`, `"processing_close"`,
>   or `"processing_final_content"`
> - `meta["processing_type"]`: `"artefact_building"`, `"widget_building"`,
>   `"form_building"`, `"note_building"`, `"skill_building"`, or `"synthesis"`

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
| `"inline_widget_start"` | Opening `<lollms_inline ...>` tag detected | `{title: str, widget_type: "html"\|"react"\|"svg"}` |
| `"form_start"` | Opening `<lollms_form ...>` tag detected | `{title: str}` |

### 2.4.1 Unified Processing Tag Protocol (NEW — Recommended)

The chat streaming system now uses a unified single-channel approach with `<processing>` tags to indicate ongoing operations. Status messages stream inside these tags, and final content is delivered in bulk after the processing tag closes. This replaces the separate secondary stream channels (MSG_TYPE_ARTEFACT_CHUNK, etc.) while maintaining backward compatibility through optional legacy events.

**Processing Tag Format**

```xml
<processing type="PROCESSING_TYPE" title="..." [attr1="..."] ...>
* Status message 1...
* Status message 2...
</processing>
[BULK FINAL CONTENT FOLLOWS - for widgets/forms only]
```

**Meta Dict for Processing Events**

| `meta["type"]` | Description | Additional `meta` fields |
|---|---|---|
| `"processing_open"` | Opening tag detected | `processing_type`, `title`, `attrs` |
| `"processing_status"` | Status line streamed | `processing_type`, `status` |
| `"processing_close"` | Closing tag detected | `processing_type` |
| `"processing_final_content"` | Bulk content after close | `processing_type`, `content_length` |

#### Standard Processing Types Reference

The following processing types are standardized across the framework. UIs should implement appropriate rendering for each:

| Processing Type | Description | Typical Attributes | UI Rendering Guidance |
|---|---|---|---|
| `artefact_building` | Creating or updating an artefact (code, document, etc.) | `art_type`, `language` | Show file icon + name, progress indicator, syntax-highlighted preview on completion |
| `note_building` | Creating a persistent note | — | Show note icon with title, expandable preview of content |
| `skill_building` | Creating a reusable skill capsule | `category`, `description` | Show graduation cap icon, category badge, description tooltip |
| `widget_building` | Building an interactive HTML/React/SVG widget | `widget_type` | Show widget icon, iframe sandbox indicator, live preview on completion |
| `form_building` | Creating an interactive user form | — | Show form icon with field count, render interactive form on completion |
| `tool_execution` | Executing a tool/function call | `tool`, `params` | Show tool/wrench icon, tool name, animated execution state, success/fail indicator |
| `synthesis` | Multi-agent synthesis (swarm mode) | `round`, `agents`, `mode`, `stage` | Show collaboration icon, round counter, agent avatars, confidence meter |

**Attribute Reference by Type**

```python
# artefact_building
{
    "type": "artefact_building",
    "title": "app.py",
    "attrs": {
        "art_type": "code",           # document, code, note, skill, image, etc.
        "language": "python",         # for code artefacts
    }
}

# skill_building
{
    "type": "skill_building",
    "title": "Async Patterns",
    "attrs": {
        "category": "programming/python/async",
        "description": "Common async patterns in Python",
    }
}

# widget_building
{
    "type": "widget_building",
    "title": "Circle Explorer",
    "attrs": {
        "widget_type": "html",        # html, react, svg
    }
}

# tool_execution
{
    "type": "tool_execution",
    "title": "search_arxiv",          # tool name as title
    "attrs": {
        "tool": "search_arxiv",
        "params": '{"query": "LLM context compression", "max_results": 20}',
    }
}

# synthesis (swarm)
{
    "type": "synthesis",
    "title": "Round 2",               # or "Final Synthesis"
    "attrs": {
        "round": "2",
        "agents": "3",
        "mode": "quality",            # quality, debate, simulation, game, freeform
        "stage": "final",             # optional: final vs intermediate
    }
}
```

**Recommended UI Patterns**

| Pattern | Implementation |
|---|---|
| **Inline Progress Cards** | Render a compact card in the chat stream showing: icon, title, status lines as bullet list, completion state |
| **Floating Status Indicators** | For long operations, show a sticky header with current processing type and title |
| **Collapsible Details** | Completed processing blocks can be collapsed to save space, showing only title and success/fail state |
| **Type-Specific Icons** | Use distinct icons: 📄 artefact, 📝 note, 🎓 skill, 🎛️ widget, 📋 form, 🔧 tool, 🐝 synthesis |
| **Animated States** | Pulse/spinner during `processing_open` → `processing_close`, solid icon after completion |
| **Error Highlighting** | Red accent for failed operations, with expandable error details |

**Complete Processing Stream Example**

```xml
<processing type="tool_execution" tool="search_arxiv" params="{&quot;query&quot;: &quot;LLM context compression&quot;}">
* Executing search_arxiv...
* Found 15 papers matching query
* Success: search_arxiv (15 sources)
</processing>
Completed with 2847 chars of output

<processing type="artefact_building" title="research_summary.md" art_type="document">
* Creating new artefact 'research_summary.md'
* Applying content...
* Finalizing artefact (type: document, language: none)
* Artefact saved as version 1
</processing>

<processing type="widget_building" title="Publication Trends" widget_type="html">
* Building html widget 'Publication Trends'
* Validating HTML/CSS/JS content...
* Validation passed
</processing>
<lollms_widget id="uuid-1234" />
<lollms_inline type="html" title="Publication Trends">
<!DOCTYPE html>
...
</lollms_inline>
```

### 2.5 Secondary Content Streams — CHUNK and DONE Events (DEPRECATED)

> **DEPRECATED**: This section describes the legacy secondary stream protocol.
> New applications should use the unified `<processing>` tag protocol (§2.4.1).
> These events continue to fire for backward compatibility but may be removed
> in a future major version.

The framework uses an internal state machine (`_StreamState`) to intercept XML tags.
When the LLM generates an artefact, note, skill, or widget, the raw content is routed
to a **separate, parallel stream** using dedicated event types rather than appearing in
the main chat bubble stream (`MSG_TYPE_CHUNK`).

#### 2.5.1 Lifecycle of a Tagged Block (Legacy)
1.  **Announcement**: A `MSG_TYPE_CHUNK` fires with `text=""` and `meta` containing the tag title.
2.  **Streaming**: Zero or more `*_CHUNK` events fire (e.g., `MSG_TYPE_ARTEFACT_CHUNK`).
3.  **Completion**: A `*_DONE` event fires. At this exact moment, the framework:
    -   Validates the content (for widgets).
    -   Persists the data to the database (for artefacts/notes/skills).
    -   Fires `MSG_TYPE_ARTEFACTS_STATE_CHANGED`.
4.  **Anchor Placement**: For Widgets and Forms, an anchor tag is appended to the main message content (see §20).

### 2.6 `MSG_TYPE_ARTEFACTS_STATE_CHANGED` — Two Distinct Modes
=======

### 2.6 `MSG_TYPE_ARTEFACTS_STATE_CHANGED` — Two Distinct Modes

**Per-artefact real-time event** (fired as each XML tag is processed):

```python
# text — JSON with operation details  |  meta — {artefact: dict, is_new: bool}
{"type": "artefact_created", "title": "app.py", "version": 1, "art_type": "code"}
```

**Batch summary event** (fired once at end of turn):

```python
# text — JSON list of affected titles  |  meta — {artefacts: [...], action?: str}
'["app.py", "README.md", "Transavia Note"]'
```

### 2.7 Step Events

`STEP_START` and `STEP_END` always arrive in matched pairs linked by a UUID in
`meta["id"]`. See §9 for phases emitted during tool-use.

### 2.8 A Complete Callback Implementation

```python
from lollms_client.lollms_types import MSG_TYPE

def on_event(text, msg_type, meta):
    match msg_type:
        case MSG_TYPE.MSG_TYPE_CHUNK:
            t = meta.get("type")
            if not text and t == "artefact_update":
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
                print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_WIDGET_DONE:
            label = "[patch]" if meta["is_patch"] else f"[{meta['art_type']}]"
            print(f"\n  Artefact ready: {meta['title']} {label}")

        case MSG_TYPE.MSG_TYPE_FORM_READY:
            print(f"\n  [Form ready] {meta['form']['title']}")
            print(f"    Fields: {len(meta['form']['fields'])}")

        case MSG_TYPE.MSG_TYPE_FORM_SUBMITTED:
            print(f"\n  [Form submitted] {meta['form_id'][:8]}")
            print(f"    Answers: {list(meta['answers'].keys())}")

        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            if "artefact" in meta:
                op  = "created" if meta["is_new"] else "updated"
                art = meta["artefact"]
                print(f"\n  [Artefact {op}] {art['title']} v{art['version']}")
            else:
                action = meta.get("action", "updated")
                print(f"\n  [Artefacts {action}] {text}")

        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n  > {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"  < {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n  [Tool] {meta['tool']}({meta['params']})")

        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  [Result] {meta['tool']} -> {text[:120]}")

        case MSG_TYPE.MSG_TYPE_WARNING:
            print(f"\n  [Warning] {text}")
```

---

## 3. The Branching System

### 3.1 Concept

Every message forms a node in a directed acyclic graph. Each node has a single optional
`parent_id`. A **branch** is the linear path from the root (a message with no parent) to
any **leaf** (a message with no children). At any moment one branch is **active** — its
leaf ID is stored in `discussion.active_branch_id` — and this is the sequence of messages
the LLM sees as "the conversation."

```
[root: user "Hello"]
        │
        ├─[assistant reply A]        ← branch A (leaf)
        │
        └─[assistant reply B]
                │
                ├─[user "Follow up 1"]
                │       └─[assistant reply C]   ← branch B (leaf, active)
                │
                └─[user "Follow up 2"]
                        └─[assistant reply D]   ← branch C (leaf)
```

The three branches above share the root and "reply B" node. Each branch is an
independent conversation trajectory.

### 3.2 What Creates a Branch

| Trigger | How |
|---|---|
| **Regeneration** | `chat(add_user_message=False)` after a user message creates a new assistant response with the same `parent_id` as the existing one — producing a sibling |
| **Explicit fork** | `discussion.fork_from(msg_id, ...)` or `add_message(parent_id=<any earlier node id>)` |
| **Agentic tool loop** | Does **not** create permanent branches — temporary messages are deleted when the loop exits |
| **Branch merge** | `discussion.merge_branches(source_leaf, target_leaf)` copies unique source messages onto the target, extending it |

### 3.3 Data Structures — BranchInfo and MessageNode

`BranchMixin` exposes two lightweight dataclasses. They are available at the top-level
import:

```python
from lollms_client.lollms_discussion import BranchInfo, MessageNode
```

#### `BranchInfo`

A snapshot describing one complete root-to-leaf path.

| Field | Type | Description |
|---|---|---|
| `leaf_id` | `str` | The ID of the leaf message |
| `message_ids` | `list[str]` | Ordered IDs `[root_id, …, leaf_id]` |
| `depth` | `int` | Number of messages in the path |
| `label` | `str` | Human-readable label (auto or custom via `set_branch_label`) |
| `is_active` | `bool` | `True` if this is the current active branch |
| `created_at` | `datetime \| None` | Creation time of the leaf message |
| `last_sender` | `str` | Sender of the leaf message |
| `last_content_preview` | `str` | First 80 characters of the leaf's content |

```python
bi = discussion.get_branch_info(leaf_id)
print(bi.label, bi.depth, bi.is_active)
d = bi.to_dict()   # JSON-serialisable
```

#### `MessageNode`

One node in the full tree, with recursive children.

| Field | Type | Description |
|---|---|---|
| `message_id` | `str` | The message's ID |
| `parent_id` | `str \| None` | Parent ID, or `None` for root nodes |
| `sender` | `str` | Sender name |
| `sender_type` | `str` | `"user"` / `"assistant"` / `"system"` |
| `content_preview` | `str` | First 120 characters of content |
| `created_at` | `datetime \| None` | Creation time |
| `children` | `list[MessageNode]` | Direct child nodes (recursive) |
| `branch_count` | `int` | Number of leaf nodes reachable from this node — i.e. how many branches pass through it |
| `is_active_path` | `bool` | `True` if this node is on the current active branch |

```python
roots = discussion.get_tree()           # list of root MessageNode objects
for root in roots:
    print(root.message_id, root.branch_count)
    for child in root.children:         # recursive
        print("  ", child.message_id, child.is_active_path)

# Serialise to JSON for a UI tree component
import json
tree_json = [r.to_dict() for r in roots]
```

### 3.4 Discovery API

All discovery methods are read-only and safe to call at any time.

#### `list_branches() → list[BranchInfo]`

Returns a `BranchInfo` for every leaf in the discussion, sorted by leaf creation time
(oldest first). This is the primary method for showing a branch-picker UI.

```python
branches = discussion.list_branches()
for b in branches:
    marker = "★" if b.is_active else " "
    print(f"{marker} [{b.leaf_id[:8]}] depth={b.depth}  {b.label}")
    print(f"    Last: {b.last_sender}: {b.last_content_preview}")
```

#### `get_branch_info(leaf_id) → BranchInfo | None`

Returns the `BranchInfo` for any specific leaf. Returns `None` if the ID is not found.

```python
bi = discussion.get_branch_info(some_leaf_id)
if bi:
    print(bi.message_ids)   # [root_id, ..., leaf_id]
```

#### `get_tree() → list[MessageNode]`

Returns the full message forest as a list of root `MessageNode` objects. Each node's
`children` attribute is recursively populated. Use this to render an interactive tree
view in a UI.

```python
roots = discussion.get_tree()
# Serialise for a frontend tree component (e.g. Vue / React)
tree_data = [r.to_dict() for r in roots]
```

**`branch_count` is the key field** for tree rendering — it tells the UI how many
independent paths lead through each node. A node with `branch_count == 1` is on a
linear chain (no divergence). A node with `branch_count > 1` is a **fork point** and
should be rendered with a visual branch indicator.

#### `get_children(message_id) → list[LollmsMessage]`

Returns the direct children of a given message. An empty list means the message is a
leaf (no branches continue from it).

```python
children = discussion.get_children(msg.id)
if len(children) > 1:
    print(f"Branch point: {len(children)} alternate continuations")
elif len(children) == 0:
    print("Leaf — end of branch")
```

#### `get_siblings(message_id) → list[LollmsMessage]`

Returns all messages that share the same parent as the given message, including the
message itself. Sorted by creation time. This is the set of alternate replies to the
same user message — the primary use case for a "previous/next reply" UI control.

```python
siblings = discussion.get_siblings(ai_message_id)
print(f"{len(siblings)} alternate replies to this user message")
current_idx = next(i for i, s in enumerate(siblings) if s.id == ai_message_id)
print(f"Currently viewing reply {current_idx + 1} of {len(siblings)}")
```

#### `get_message_branches(message_id) → list[BranchInfo]`

Returns all branches (as `BranchInfo`) that **pass through** the given message. Useful
for a tooltip or popover showing "N branches from here" when a user hovers over a message.

```python
through = discussion.get_message_branches(some_message_id)
print(f"{len(through)} branches pass through this message")
for b in through:
    print(f"  → {b.label} ({'active' if b.is_active else 'inactive'})")
```

### 3.5 Navigation API

#### `switch_branch(leaf_id) → bool`

Change the active branch to the one whose tip is `leaf_id`. Returns `True` on success,
`False` if the ID is not found. This is the main way to let users switch between branches.

```python
branches = discussion.list_branches()
# Let user pick branch 2
ok = discussion.switch_branch(branches[2].leaf_id)
if ok:
    # Now chat() will see this branch as the conversation history
    result = discussion.chat("Continue from here...")
```

> **Equivalence** — `switch_branch(leaf_id)` is equivalent to the existing
> `switch_to_branch(leaf_id)` method inherited from `UtilsMixin`. The new version
> returns a `bool` instead of raising silently.

#### `switch_to_sibling(direction=1) → LollmsMessage | None`

Move to the next (`+1`) or previous (`-1`) sibling of the current active leaf's parent
reply. This is the natural "← previous reply / next reply →" UX for cycling through
alternate AI responses to the same user message.

Returns the new active `LollmsMessage` (the new leaf), or `None` if there is no sibling
in that direction.

```python
# "Previous reply" button
prev = discussion.switch_to_sibling(direction=-1)
if prev:
    print("Switched to:", prev.content[:80])
else:
    print("Already at the first reply")

# "Next reply" button
nxt = discussion.switch_to_sibling(direction=+1)
```

`switch_to_sibling` finds the deepest leaf under the new sibling automatically, so if
the sibling has follow-up messages the active branch becomes the deepest one rather than
stopping at the sibling itself.

#### `get_branch(branch_tip_id) → list[LollmsMessage]`

Pre-existing method. Returns the ordered list of messages on a branch from root to tip.

```python
messages = discussion.get_branch(discussion.active_branch_id)
for m in messages:
    print(f"{m.sender}: {m.content[:60]}")
```

### 3.6 Forking

#### `fork_from(message_id, label=None, initial_content="", initial_sender="user", initial_sender_type="user", **kwargs) → LollmsMessage`

Start a new branch from any existing message by adding a child message to it. The new
message becomes the active branch tip. Subsequent `chat()` calls will continue from here.

```python
# Fork from an earlier user message to explore an alternative direction
branch_msg = discussion.fork_from(
    message_id=some_user_msg_id,
    label="Redis approach",
    initial_content="What if we used Redis instead of Postgres for the session store?",
    initial_sender="user",
)
# branch_msg is now the active branch tip
result = discussion.chat("", add_user_message=False)  # get AI reply on this branch
```

| Parameter | Default | Description |
|---|---|---|
| `message_id` | required | The message to fork from. Its ID becomes the `parent_id` of the new message. |
| `label` | `None` | Human-readable label stored in the new message's metadata as `branch_label` |
| `initial_content` | `""` | Content for the fork message |
| `initial_sender` | `"user"` | Sender name |
| `initial_sender_type` | `"user"` | Sender type |
| `**kwargs` | — | Forwarded to `add_message` (e.g. `images`, `metadata`) |

**Fork without an initial message** — to fork and immediately let the LLM speak, fork
from an existing assistant message, leave `initial_content` empty, and call
`chat("", add_user_message=False)`:

```python
# Go back in history and generate an alternate AI reply to the same user message
user_msg = discussion.get_message(user_msg_id)
fork_tip = discussion.fork_from(user_msg.parent_id)   # fork from grandparent
discussion.active_branch_id = fork_tip.id
result = discussion.chat("", add_user_message=False,
                         branch_tip_id=user_msg.parent_id)
```

A cleaner idiom for regeneration is the existing `regenerate_branch()` method:

```python
result = discussion.regenerate_branch(branch_tip_id=assistant_msg_id)
```

This is equivalent to a fork but handles the parent-resolution logic automatically.

### 3.7 Deletion

Two distinct semantics are provided — choose based on what you want to keep.

#### `delete_branch(leaf_id, keep_ancestors=True) → int`

Remove the branch tip and, optionally, walk back up the tree removing now-childless
ancestors until one with other children is reached.

| `keep_ancestors` | What is removed |
|---|---|
| `True` (default) | The leaf, plus any ancestors that have no other children (the "stub" is fully trimmed) |
| `False` | The entire path from root to leaf, regardless of shared ancestors |

```python
# Remove a dead-end branch completely (trim back to the nearest fork)
removed = discussion.delete_branch(branch_leaf_id)
print(f"Removed {removed} message(s)")

# Remove only the leaf, leave ancestors alone
removed = discussion.delete_branch(branch_leaf_id, keep_ancestors=False)
```

> **Safety constraint** — `delete_branch` requires `leaf_id` to be a genuine leaf
> (no children). If the message has children, use `prune_branch` instead.

#### `prune_branch(message_id) → int`

Remove a message **and all of its descendants** — the entire subtree rooted at
`message_id`. The message's parent is preserved.

```python
# Remove a bad AI reply and all follow-up messages
removed = discussion.prune_branch(bad_ai_reply_id)
print(f"Pruned {removed} messages (the reply and all follow-ups)")
```

Use `prune_branch` when you want to cut off an entire direction of the conversation.
Use `delete_branch` when you want to trim back a dead-end leaf.

**Active branch repair** — both methods automatically repair `active_branch_id` if the
deleted messages included the current tip. The active branch is reset to the nearest
surviving leaf.

### 3.8 Merging

#### `merge_branches(source_leaf_id, target_leaf_id=None, separator_content="--- merged from another branch ---", separator_sender="system") → LollmsMessage`

Copy the messages from the *source* branch that are **not** already on the *target*
branch, and append them to the target. A separator message is inserted between the two
branches to mark the join point.

Returns the new active leaf (the last appended message).

```python
# Bring a parallel research branch back into the main branch
new_tip = discussion.merge_branches(
    source_leaf_id   = research_branch_leaf,
    target_leaf_id   = main_branch_leaf,   # None → use active branch
    separator_content= "--- Research findings merged ---",
)
print(f"Merged. New tip: {new_tip.id}")
result = discussion.chat("Summarise everything we know so far.")
```

Messages are copied with their original sender, sender_type, content, and images.
Metadata is preserved and augmented with `"merged_from": original_message_id`.

To skip the separator, pass `separator_content=""`:

```python
new_tip = discussion.merge_branches(research_leaf, separator_content="")
```

### 3.9 Labelling and Diffing

#### `set_branch_label(leaf_id, label) → bool`

Attach a human-readable label to a branch. Stored in the leaf message's metadata and
returned by `get_branch_info()` / `list_branches()`. Returns `True` on success.

```python
discussion.set_branch_label(branch_leaf_id, "Redis approach — rejected")
discussion.set_branch_label(main_leaf_id,   "Final implementation")
```

#### `branch_diff(leaf_a, leaf_b) → dict`

Compare two branches and identify what they share and what is unique to each.

```python
diff = discussion.branch_diff(leaf_a=branch_a_id, leaf_b=branch_b_id)
# Returns:
# {
#   "common_ancestor_id": str | None,  ← last shared message
#   "only_in_a":          [msg_id, …], ← messages unique to branch A
#   "only_in_b":          [msg_id, …], ← messages unique to branch B
#   "shared":             [msg_id, …], ← messages on both paths
# }

print(f"Diverged after: {diff['common_ancestor_id']}")
print(f"Branch A has {len(diff['only_in_a'])} unique messages")
print(f"Branch B has {len(diff['only_in_b'])} unique messages")
```

### 3.10 Temporary Tool-History Messages

During each round of the agentic loop, two **temporary** messages are inserted into the
branch so the model can see its own prior tool calls and their results:

1. `assistant` — the raw LLM output including the `<tool_call>` tag.
2. `user/system` — a `<tool_result name="...">...</tool_result>` block.

These are deleted from the branch immediately after the loop exits. The persisted
conversation contains only the clean final assistant message. Branch navigation and
discovery APIs never see these temporary messages because they are cleaned up before
`commit()` runs.

### 3.11 Complete Branch Management Example

The following example shows a full branch lifecycle: create, explore, fork, navigate,
delete, and merge.

```python
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, BranchInfo

db = LollmsDataManager("sqlite:///branches.db")
disc = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db, autosave=True,
                                   system_prompt="You are a technical advisor.")

# ── Build an initial conversation ─────────────────────────────────────────────
r1 = disc.chat("What database should I use for a high-traffic read-heavy API?")
r2 = disc.chat("Tell me more about PostgreSQL for this use case.")

main_leaf = disc.active_branch_id        # save the tip of the main branch

# ── Fork: explore a Redis alternative ────────────────────────────────────────
# Fork from the first user message to explore a totally different path
first_user_msg = disc.get_branch(main_leaf)[0]   # first message in history
fork_tip = disc.fork_from(
    message_id   = first_user_msg.id,
    label        = "Redis exploration",
    initial_content = "What if we used Redis as the primary store instead?",
)
r3 = disc.chat("", add_user_message=False)        # AI replies on Redis branch
redis_leaf = disc.active_branch_id

# ── Discover all branches ─────────────────────────────────────────────────────
branches = disc.list_branches()
print(f"\n{len(branches)} branches:")
for b in branches:
    print(f"  {'★' if b.is_active else ' '} [{b.leaf_id[:8]}] "
          f"depth={b.depth}  {b.label}")
    print(f"    {b.last_sender}: {b.last_content_preview}")

# ── Inspect the tree ──────────────────────────────────────────────────────────
roots = disc.get_tree()
def print_tree(node, indent=0):
    marker = "►" if node.is_active_path else " "
    fork   = f" [FORK ×{node.branch_count}]" if node.branch_count > 1 else ""
    print("  " * indent + f"{marker} [{node.message_id[:6]}] "
          f"{node.sender}: {node.content_preview[:40]}{fork}")
    for child in node.children:
        print_tree(child, indent + 1)

for root in roots:
    print_tree(root)

# ── Navigate between branches ────────────────────────────────────────────────
# Switch back to the main branch
disc.switch_branch(main_leaf)
print("\nActive branch:", disc.active_branch_id[:8])

# Cycle through alternate AI replies to the first user message
siblings = disc.get_siblings(r1["ai_message"].id)
print(f"\n{len(siblings)} alternate AI replies to the first question:")
for s in siblings:
    print(f"  [{s.id[:8]}] {s.content[:60]}")

# Move to the next sibling
next_reply = disc.switch_to_sibling(direction=+1)
if next_reply:
    print("Switched to:", next_reply.content[:80])

# ── Check branches through a specific message ────────────────────────────────
branches_through = disc.get_message_branches(first_user_msg.id)
print(f"\n{len(branches_through)} branches pass through the first user message")

# ── Branch diff ──────────────────────────────────────────────────────────────
diff = disc.branch_diff(main_leaf, redis_leaf)
print(f"\nCommon ancestor: {diff['common_ancestor_id'][:8] if diff['common_ancestor_id'] else 'none'}")
print(f"Only in main:    {[m[:8] for m in diff['only_in_a']]}")
print(f"Only in redis:   {[m[:8] for m in diff['only_in_b']]}")

# ── Label branches ────────────────────────────────────────────────────────────
disc.set_branch_label(main_leaf,  "PostgreSQL path — final")
disc.set_branch_label(redis_leaf, "Redis path — prototype")

# ── Merge the Redis branch back into main ────────────────────────────────────
disc.switch_branch(main_leaf)
new_tip = disc.merge_branches(
    source_leaf_id   = redis_leaf,
    target_leaf_id   = main_leaf,
    separator_content= "--- Redis exploration summary ---",
)
r4 = disc.chat("Compare the two approaches and give a final recommendation.")

# ── Delete the old Redis branch leaf (now merged, no longer needed) ──────────
removed = disc.delete_branch(redis_leaf)
print(f"\nRemoved {removed} message(s) from Redis branch stub")

# ── Prune an entire subtree if needed ────────────────────────────────────────
# Suppose r2's AI reply was completely wrong — remove it and all follow-ups
# disc.prune_branch(r2["ai_message"].id)

# ── Final state ───────────────────────────────────────────────────────────────
print("\nFinal branches:")
for b in disc.list_branches():
    print(f"  {'★' if b.is_active else ' '} {b.label}  (depth={b.depth})")
```

#### Quick-Reference: Branch API at a Glance

```python
# ── Discovery ──────────────────────────────────────────────────────────────
disc.list_branches()                    # → [BranchInfo, ...]  all leaves
disc.get_branch_info(leaf_id)           # → BranchInfo | None
disc.get_tree()                         # → [MessageNode, ...]  full forest
disc.get_children(msg_id)              # → [LollmsMessage, ...]  direct children
disc.get_siblings(msg_id)              # → [LollmsMessage, ...]  same-parent peers
disc.get_message_branches(msg_id)      # → [BranchInfo, ...]  branches via msg

# ── Navigation ────────────────────────────────────────────────────────────
disc.switch_branch(leaf_id)            # → bool
disc.switch_to_sibling(direction=±1)   # → LollmsMessage | None
disc.get_branch(leaf_id)               # → [LollmsMessage, ...]  path to root

# ── Forking ───────────────────────────────────────────────────────────────
disc.fork_from(msg_id, label=..., initial_content=...)  # → LollmsMessage

# ── Deletion ──────────────────────────────────────────────────────────────
disc.delete_branch(leaf_id)            # → int  (leaf + childless ancestors)
disc.delete_branch(leaf_id, keep_ancestors=False)       # → int  (whole path)
disc.prune_branch(msg_id)              # → int  (msg + all descendants)

# ── Merging ───────────────────────────────────────────────────────────────
disc.merge_branches(source_leaf, target_leaf=None)      # → LollmsMessage

# ── Labelling & Diff ──────────────────────────────────────────────────────
disc.set_branch_label(leaf_id, label)  # → bool
disc.branch_diff(leaf_a, leaf_b)       # → {common_ancestor_id, only_in_a, only_in_b, shared}
```

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

#### 4.2.2 Fuzzy Title Matching

The framework uses **fuzzy title matching** (bigram overlap + substring containment +
normalised casing) to locate the right artefact when the LLM does not reproduce the
stored title exactly. The matching threshold is **0.60**. A log line is emitted when
fuzzy matching fires:

```
[INFO] Fuzzy title match: 'tag_name' → 'stored_title' (updating in place)
```

#### 4.2.3 Renaming an Artefact

```xml
<artefact name="old_flight_note" rename="Réservation Vol Transavia - Tunis → Lyon">
<<<<<<< SEARCH
Airline: unknown
=======
Airline: Transavia
>>>>>>> REPLACE
</artefact>
```

def greet(name: str = 'world'):
    return f'hello {name}'
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

Fault tolerance: missing `>>>>>>> REPLACE` appended automatically; variant chevron
counts normalised; CRLF normalised to LF. When a patch fails, the existing artefact
is left untouched.

#### 4.2.5 Revert to a Previous Version

```xml
<revert_artefact name="app.py" version="2" />
```

#### 4.2.6 Image Generation and Editing (TTI)

```xml
<generate_image width="1024" height="1024">A photorealistic sunset over a mountain lake</generate_image>
<edit_image name="hero_image.png">Add a small wooden boat in the foreground</edit_image>
```

Auto-disabled when `lollmsClient.tti is None`.

### 4.3 The ArtefactManager Python API

```python
mgr = discussion.artefacts

mgr.list(active_only=False)                              # all artefacts
mgr.list(artefact_type="note")                          # filter by type
mgr.get("app.py")                                        # get by title → dict|None
mgr.add(title="readme.md", artefact_type="document",
        content="# My Project", active=True)             # create
mgr.update("readme.md", new_content="# v2",
           new_title="README.md")                        # update / rename
mgr.activate("readme.md")
mgr.deactivate("old_draft.md")
mgr.revert("app.py", target_version=1)
mgr.remove("scratch.txt")
```

### 4.4 Real-Time Artefact Events

`MSG_TYPE_ARTEFACTS_STATE_CHANGED` fires once per `<artefact>` / `<revert_artefact>`
tag immediately as it is processed, before the full response is assembled. See §2.6 for
event payload shapes.

### 4.5 Artefact Images — Embedding Visual Content in Documents

Artefacts can carry images alongside their text content. This is essential for:
- **PDF documents** converted to text+images (each page rendered as PNG/JPEG)
- **Screenshots** embedded in technical documentation
- **Diagrams** referenced from code explanations
- **Multi-modal reports** combining prose and visual evidence

#### 4.5.1 The Image Anchor System

Images are stored as base64 strings in `artefact["images"]` and referenced **inline**
in the text content using self-closing anchor tags:

```xml
<artefact_image id="TITLE::N" />
```

Where:
- `TITLE` = the artefact's `title` field
- `N` = 0-based index into `artefact["images"]`

**Example artefact with embedded images:**
```xml
<artefact name="annual_report_2024.pdf" type="document">
# Annual Report 2024

## Executive Summary
<artefact_image id="annual_report_2024.pdf::0" />

Revenue increased 23% year-over-year. Key drivers shown in the chart above.

## Regional Breakdown
<artefact_image id="annual_report_2024.pdf::1" />

The Asia-Pacific region now represents our largest market segment.

## Product Portfolio
<artefact_image id="annual_report_2024.pdf::2" />
<artefact_image id="annual_report_2024.pdf::3" />

New product lines launched in Q3 and Q4 respectively.
</artefact>
```

#### 4.5.2 Image Storage and MIME Types

| Field | Type | Description |
|-------|------|-------------|
| `images` | `list[str]` | Base64-encoded image data |
| `image_media_types` | `list[str]` | Parallel list of MIME types (`"image/png"`, `"image/jpeg"`, etc.) |

When `image_media_types` is shorter than `images`, missing entries default to `"image/jpeg"`.

#### 4.5.3 Context Assembly — How the LLM Receives Images

The framework automatically:

1. **Collects** all images from active artefacts via `artefacts.get_context_images()`
2. **Appends** them to the vision input after any user-supplied images
3. **Injects** a mapping note into the scratchpad showing which slot corresponds to which anchor

Image ordering in the LLM call:
```
[user images] → [artefact images in activation order, by index]
```

The system prompt includes instructions (see `_build_artefact_instructions()`) telling the LLM how to interpret `<artefact_image id="..." />` anchors and correlate them with the numbered image slots.

#### 4.5.4 API Methods for Image-Enabled Artefacts

```python
# Create an artefact with images
artefact = discussion.artefacts.add(
    title="my_document",
    artefact_type=ArtefactType.DOCUMENT,
    content="See figure: <artefact_image id='my_document::0' />",
    images=[base64_page_1, base64_page_2],  # list of base64 strings
    image_media_types=["image/png", "image/png"],
    active=True
)

# Retrieve images for rendering
context_images = discussion.artefacts.get_context_images()
# Returns: [{"id": "my_document::0", "data": "...", "media_type": "image/png",
#            "title": "my_document", "index": 0, "active": True}, ...]

# Get only IMAGE-type artefacts (legacy/UI helper)
image_artefacts = discussion.artefacts.get_active_images()
```

#### 4.5.5 Complete Working Example — PDF Import with Page Images

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion import ArtefactType
import base64

# Setup
lc = LollmsClient("ollama", {"model_name": "llava", "host_address": "http://localhost:11434"})
db = LollmsDataManager("sqlite:///multimodal.db")
discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db, autosave=True)

# Simulate PDF processing: extract text and render pages to images
def import_pdf_as_artefact(discussion, pdf_path: str, title: str):
    """
    Example: Convert a PDF to a text+images artefact.
    In production, use pdf2image + pytesseract or similar.
    """
    # Simulated: pages as (text, image_base64) tuples
    pages = [
        ("## Page 1\n\nExecutive summary content here...", "iVBORw0KGgo..."),  # PNG base64
        ("## Page 2\n\nFinancial data table...", "iVBORw0KGgo..."),           # PNG base64
        ("## Page 3\n\nChart showing trends...", "iVBORw0KGgo..."),          # PNG base64
    ]
    
    # Build content with image anchors
    content_parts = []
    images_b64 = []
    media_types = []
    
    for i, (page_text, img_b64) in enumerate(pages):
        # Add image anchor before the page text
        content_parts.append(f"<artefact_image id=\"{title}::{i}\" />\n\n{page_text}")
        images_b64.append(img_b64)
        media_types.append("image/png")
    
    full_content = "\n\n---\n\n".join(content_parts)
    
    # Create the artefact
    artefact = discussion.artefacts.add(
        title=title,
        artefact_type=ArtefactType.DOCUMENT,
        content=full_content,
        images=images_b64,
        image_media_types=media_types,
        active=True,
        description=f"Imported from {pdf_path} — {len(pages)} pages with images"
    )
    
    return artefact

# Import a document
artefact = import_pdf_as_artefact(discussion, "report.pdf", "Q4_Earnings_Report")

# --- Rendering in a UI ---

def render_artefact_with_images(discussion, title: str):
    """
    Render an artefact's text content, replacing image anchors with actual images.
    """
    artefact = discussion.artefacts.get(title)
    if not artefact:
        return None
    
    content = artefact["content"]
    images = artefact.get("images", [])
    media_types = artefact.get("image_media_types", ["image/jpeg"] * len(images))
    
    # Build an HTML representation
    html_parts = []
    
    # Split by image anchors and interleave actual images
    import re
    pattern = r'<artefact_image\s+id=["\']([^"\']+)["\']\s*/?>'
    
    last_end = 0
    for match in re.finditer(pattern, content):
        # Text before this anchor
        html_parts.append(content[last_end:match.start()])
        
        # Parse the image ID
        img_id = match.group(1)
        parsed = discussion.artefacts._artefacts.parse_image_id(img_id)
        if parsed:
            art_title, idx = parsed
            if art_title == title and 0 <= idx < len(images):
                # Embed as data URI
                mime = media_types[idx] if idx < len(media_types) else "image/jpeg"
                data_uri = f"data:{mime};base64,{images[idx]}"
                html_parts.append(f'<img src="{data_uri}" alt="Page {idx+1}" style="max-width:100%;" />')
        
        last_end = match.end()
    
    # Remaining text
    html_parts.append(content[last_end:])
    
    return "".join(html_parts)

# Or use the framework's built-in context image collection
context_images = discussion.artefacts.get_context_images()
print(f"Total images in context: {len(context_images)}")
for img in context_images:
    print(f"  {img['id']}: {img['media_type']} ({len(img['data'])} chars base64)")

# Chat with the document — the LLM sees both text and images
result = discussion.chat(
    "Summarize the key findings from the earnings report, "
    "referencing specific charts and figures by name.",
    streaming_callback=lambda t, mt, m: print(t, end="", flush=True) if mt.value == 0 else None
)
```

#### 4.5.6 Rules for LLM-Generated Artefacts with Images

When the LLM patches or creates artefacts:

| Rule | Rationale |
|------|-----------|
| **Preserve anchors unchanged** | When patching text containing `<artefact_image id="..." />`, do not modify or remove the tags |
| **Cannot create images via XML** | Images must be supplied by the application layer; the LLM cannot generate base64 image data |
| **Can reference anchors in replies** | The LLM may write "As shown in `<artefact_image id='doc::2' />`, revenue..." |
| **Image slots are append-only** | New images are always added to the end; existing indices remain stable |

The `build_artefacts_context_zone()` method preserves anchors verbatim in the text it returns, while `get_context_images()` provides the parallel image data for the LLM API call.

---

## 5. Notes

### 5.1 Concept

Notes are **lightweight, named, persistent documents** saved as `ArtefactType.NOTE`
artefacts. Enabled by `enable_notes=True` (default). Active immediately.

```xml
<note title="Transavia — Price Analysis">
| Route      | Base | +Baggage | Total |
|------------|------|----------|-------|
| TUN -> LYS | 89   | 35       | 124   |
</note>
```

Multiple `<note>` tags per response are supported. A note with the same title as an
existing one creates a new version. The tag is stripped from the visible message text.

```python
# List all notes
notes = discussion.artefacts.list(artefact_type="note")
```

### 5.2 Disabling Notes

```python
result = discussion.chat("Compare these flights.", enable_notes=False)
```

---

## 6. Skills

### 6.1 Concept

Skills are **reusable knowledge capsules** saved as `ArtefactType.SKILL` artefacts.
Enabled by `enable_skills=False` (default — opt-in). Extra metadata: `description` and
`category`.

```xml
<skill title="Python Async HTTP Requests"
       description="Using aiohttp for concurrent HTTP calls"
       category="programming/python/async">
# content here
</skill>
```

Category convention: `domain/subdomain/topic` (e.g. `programming/python/async`,
`cooking/baking/sourdough`).

```python
skills = discussion.artefacts.list(artefact_type="skill")
python = [s for s in skills if s.get("category","").startswith("programming/python")]
```

---

## 7. Inline Interactive Teaching Widgets

### 7.1 Concept

Widgets embed a **live, self-contained interactive learning element** inside a chat
message. Unlike artefacts (side panel), widgets render *in-place* in the chat bubble.
Enabled by `enable_inline_widgets=True` (default).

```xml
<lollms_inline type="html" title="Circle Circumference Explorer">
<!DOCTYPE html>...
</lollms_inline>
```

Three types: `html` (default, most flexible), `react` (JSX component), `svg` (animated
diagrams). The LLM is instructed that a plain-text explanation must accompany every widget.

### 7.2 Post-Processing

The `<lollms_inline>` tag is replaced with a lightweight anchor:
```html
<lollms_widget id="<uuid>" />
```

Widget source is stored in `ai_message.metadata["inline_widgets"]`:
```python
widgets = result["ai_message"].metadata.get("inline_widgets", [])
# [{id, type, title, source}, ...]
```

### 7.3 Frontend Rendering Guide

| Type | Recommended approach |
|---|---|
| `html` | Sandboxed `<iframe srcdoc="...">` with `sandbox="allow-scripts"` |
| `react` | Babel standalone transpile + `ReactDOM.createRoot` in error-bounded `<div>` |
| `svg` | Inline `<svg>` injection or Blob URL `<img>` for isolation |

### 7.4 Disabling

```python
result = discussion.chat("Explain quicksort.", enable_inline_widgets=False)
```

---

## 8. Interactive Forms

### 8.1 Concept

Forms allow the LLM to pause generation and ask the user for structured, multi-field input. When the LLM emits a `<lollms_form>` block, the framework fires `MSG_TYPE_FORM_READY`, generation pauses, and the application renders an interactive form UI. After the user submits their answers, `submit_form_response()` injects the results back into the conversation and generation resumes.

Forms are ideal for:
- Collecting multiple pieces of information before starting a complex task
- Quizzes and interactive evaluations
- Guided workflows where early choices affect later steps
- Any scenario where structured input is more efficient than free-text chat

### 8.2 LLM Interaction — XML Tags

The LLM creates forms by emitting a `<lollms_form>` block with field definitions inside.

#### 8.2.1 Basic XML Syntax

```xml
<lollms_form title="Project Configuration" description="Tell us about your requirements">
  <field name="project_name" label="Project Name" type="text" required="true"/>
  <field name="language" label="Programming Language" type="select" options="Python,JavaScript,Rust,Go" required="true"/>
  <field name="needs_auth" label="Require Authentication?" type="checkbox"/>
  <field name="description" label="Project Description" type="textarea" rows="4"/>
</lollms_form>
```

#### 8.2.2 Alternative JSON Syntax

Forms can also be defined as JSON inside the tag body:

```xml
<lollms_form title="Quiz: Python Basics">
{
  "fields": [
    {"name": "q1", "label": "What does list.append() do?", "type": "radio", "options": ["Adds to start", "Adds to end", "Removes item", "Sorts list"], "required": true},
    {"name": "q2", "label": "Explain your answer", "type": "textarea", "rows": 3, "required": true},
    {"name": "q3", "label": "Confidence (1-5)", "type": "range", "min": 1, "max": 5, "required": true}
  ]
}
</lollms_form>
```

#### 8.2.3 Field Types Reference

| Type | Description | Attributes |
|------|-------------|------------|
| `text` | Single-line text input | `placeholder`, `default` |
| `textarea` | Multi-line text input | `rows` (default 4), `placeholder` |
| `number` | Numeric input | `min`, `max`, `step`, `default` |
| `range` | Slider input | `min`*, `max`*, `step` |
| `select` | Dropdown selection | `options`* (comma-separated), `multiple` |
| `radio` | Radio button group | `options`* (comma-separated) |
| `checkbox` | Single boolean checkbox | `default` |
| `checkbox_group` | Multiple checkboxes | `options`* (comma-separated) |
| `date` | Date picker | `min`, `max` |
| `time` | Time picker | — |
| `color` | Color picker (returns #RRGGBB) | `default` |
| `rating` | Star rating widget | `min` (default 1), `max` (default 5) |
| `file` | File upload | `accept` (MIME type or extension), `multiple` |
| `code` | Code editor with syntax highlighting | `language` (e.g., "python"), `rows` |
| `section` | Visual divider/heading (no input) | `label` (shown as subheading) |
| `hidden` | Hidden field | `default` (value sent to LLM) |

\* Required attribute

### 8.3 Form Events

| Event | When Fired | `meta` Contents |
|-------|-----------|---------------|
| `MSG_TYPE_FORM_READY` (46) | `</lollms_form>` parsed and validated | `{"form": {...}, "form_id": "uuid"}` |
| `MSG_TYPE_FORM_SUBMITTED` (47) | `submit_form_response()` called | `{"form_id": "uuid", "answers": {...}, "form": {...}}` |

### 8.4 Application Integration

The application layer must:
1. Listen for `MSG_TYPE_FORM_READY` events
2. Render the form described in `meta["form"]`
3. Collect user answers (matching field `name` attributes)
4. Call `discussion.submit_form_response(form_id, answers_dict)` to resume

```python
# Example: handling form submission in application code
def on_form_ready(form_id, form_descriptor):
    # Render UI with form_descriptor["fields"]
    answers = show_form_ui_and_wait(form_descriptor)  # blocking or async
    
    # Submit answers back to resume generation
    discussion.submit_form_response(form_id, answers)
```

### 8.5 Complete Working Example

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE

lc = LollmsClient("ollama", {"model_name": "llama3.2", "host_address": "http://localhost:11434"})
db = LollmsDataManager("sqlite:///forms_demo.db")
discussion = LollmsDiscussion.create_new(
    lollms_client=lc,
    db_manager=db,
    autosave=True,
    system_prompt="You are a helpful assistant that creates project plans based on user requirements."
)

# Simulated form submission handler (in a real app, this would come from UI)
pending_forms = {}

def streaming_callback(text, msg_type, meta):
    if msg_type == MSG_TYPE.MSG_TYPE_FORM_READY:
        form = meta["form"]
        print(f"\n📝 FORM READY: {form['title']}")
        print(f"Description: {form.get('description', 'None')}")
        print("\nFields:")
        for field in form["fields"]:
            req = " [required]" if field.get("required") else ""
            print(f"  • {field['name']} ({field['type']}): {field['label']}{req}")
        
        # Simulate user filling the form
        simulated_answers = {
            "project_name": "My API Server",
            "language": "Python",
            "framework": "FastAPI",
            "needs_auth": True,
            "database": "PostgreSQL",
            "description": "A REST API for user management with JWT authentication"
        }
        
        # Store for later submission (or submit immediately)
        pending_forms[meta["form_id"]] = (form, simulated_answers)
        
        # In real app, you'd submit after user interaction:
        # discussion.submit_form_response(meta["form_id"], user_answers)
        
        return False  # Pause streaming (form blocks generation)
    
    elif msg_type == MSG_TYPE.MSG_TYPE_FORM_SUBMITTED:
        print(f"\n✅ Form {meta['form_id'][:8]} submitted with answers:")
        for k, v in meta["answers"].items():
            print(f"  {k}: {v}")
    
    elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(text, end="", flush=True)
    
    return True

# Example 1: Simple configuration form
print("=" * 50)
print("Example 1: Project Configuration Form")
print("=" * 50)

result = discussion.chat(
    "I want to build a web API. Please ask me about my requirements "
    "using a form so you can suggest the best architecture.",
    streaming_callback=streaming_callback,
    enable_forms=True
)

# Submit the pending form if one was created
if pending_forms:
    form_id, (form_desc, answers) = pending_forms.popitem()
    print(f"\n\n[Simulating user submission...]")
    discussion.submit_form_response(form_id, answers)
    
    # Continue the conversation with the form answers injected
    result2 = discussion.chat(
        "Based on my requirements above, what architecture do you recommend?",
        streaming_callback=streaming_callback
    )

# Example 2: Quiz form with scoring
print("\n" + "=" * 50)
print("Example 2: Interactive Quiz")
print("=" * 50)

discussion2 = LollmsDiscussion.create_new(
    lollms_client=lc,
    db_manager=db,
    autosave=True
)

quiz_prompt = """
Create a 3-question quiz about Python list comprehensions.
Use a form with:
- Question 1: Multiple choice about syntax
- Question 2: Code field where user writes a comprehension
- Question 3: Rating of their confidence

After submission, grade their answers and explain any mistakes.
"""

result = discussion2.chat(
    quiz_prompt,
    streaming_callback=streaming_callback,
    enable_forms=True
)

if pending_forms:
    form_id, (form_desc, answers) = pending_forms.popitem()
    # Simulate different answers for quiz
    quiz_answers = {
        "q1": "[x*2 for x in items]",  # correct
        "q2": "[x for x in range(10) if x % 2 == 0]",  # user writes code
        "q3": 4  # confidence 1-5
    }
    print(f"\n\n[Simulating quiz submission...]")
    discussion2.submit_form_response(form_id, quiz_answers)
    
    result2 = discussion2.chat(
        "Please grade my quiz and provide feedback.",
        streaming_callback=streaming_callback
    )

print("\n" + "=" * 50)
print("Forms demonstration complete!")
```

### 8.6 Form Descriptor Schema

The complete form descriptor available in `MSG_TYPE_FORM_READY`:

```python
{
    "id": "uuid-string",           # Unique form instance ID
    "title": "Form Title",         # Display heading
    "description": "...",          # Optional instructions
    "submit_label": "Submit",      # Button text (default "Submit")
    "fields": [                    # List of field descriptors
        {
            "name": "field_id",    # Machine key for answers dict
            "label": "Human label", # Display text
            "type": "text",        # One of the field types above
            "required": True,      # Whether field must be filled
            "default": "...",      # Pre-filled value (optional)
            # ... type-specific attributes
        }
    ]
}
```

### 8.7 Disabling Forms

```python
result = discussion.chat("Just a regular message", enable_forms=False)
```

---

## 9. The Silent Artefact Guard

When the LLM response consists **entirely** of XML tags, the post-processor
auto-generates a human-readable confirmation of what was produced, so the chat bubble
is never empty. Controlled by `enable_silent_artefact_explanation=True` (default).

Format:
- Artefacts: `📄 Created **title** (language) [type]: description.`
- Notes: `📝 Saved note **title** — [first line of content…]`
- Skills: `🎓 Skill saved **title** [category] — description.`
- Widgets: `🎛️ Interactive widget ready: **title** (type) — use the controls below…`
- Forms: `📋 Form ready: **title** — please fill in the fields above…`

---

## 10. The Tooling System

### 10.1 Mechanics

The LLM calls tools by emitting `<tool_call>` JSON tags mid-stream. The framework
detects these, pauses generation, dispatches the tool, appends the result to
`self.scratchpad`, and resumes. Loop runs up to `max_reasoning_steps` times.

### 10.2 Fast Path — No External Tools

When no external tools are registered, the framework skips all agentic scaffolding and
makes a single LLM call. XML tags (`<artefact>`, `<note>`, etc.) are still post-processed.

### 10.3 Anti-Duplication Guards

Every tool call's signature is tracked. On the second identical call a `DUPLICATE CALL
BLOCKED` result is injected; on the third the loop breaks. Different parameters on the
same tool are always allowed.

### 10.4 Registering External Tools

```python
tools = {
    "list_dir": {
        "name":        "list_directory",
        "description": "List files in a directory",
        "parameters": [{"name": "path", "type": "str", "optional": False}],
        "output":     [{"name": "entries", "type": "list"}],
        "callable":   lambda path: {"success": True, "entries": os.listdir(path)},
    }
}
result = discussion.chat("What files are in /tmp?", tools=tools)
```

Any tool with `name="sources"` in its output spec is treated as a RAG tool.

### 10.5 Built-in Tools

| Tool | Flag | Purpose |
|---|---|---|
| `show_tools` | `enable_show_tools=True` | Fire `MSG_TYPE_TOOLS_LIST` with full catalogue |
| `extract_artefact_text` | `enable_extract_artefact=True` | Extract a line range from an artefact by text anchors |
| `final_answer` | `enable_final_answer=True` | Signal the loop to exit |
| `request_clarification` | `enable_request_clarification=True` | Pause and ask the user |
| `deactivate_artefacts` | Context-pressure only | Deactivate artefacts to reduce token count |

---

## 11. The Streaming State Machine

The `_StreamState` machine is the "firewall" of the conversation. Its primary job is to prevent raw XML tags and tool calls from leaking into the user's chat bubble while routing secondary content through the unified `<processing>` stream.

### 11.1 Bracket Buffering
The moment a `<` character appears, all streaming is diverted into a `bracket_buf`.
- If the buffer grows to match a known tag (e.g., `<artifact`), the machine enters `STATE_SECONDARY` and immediately emits `<processing type="...">` (§2.4.1).
- If the buffer can no longer possibly match any known tag (e.g., `< Looking at the...`), the entire buffer is flushed to the chat bubble and the machine returns to `STATE_NORMAL`.
- **Max Buffer**: A safety cap of 4096 characters exists; if exceeded without a match, the buffer is flushed as text.

### 11.2 Unified Processing Stream
When a secondary tag is detected:
1. **Opening**: `<processing type="..." title="...">` is emitted immediately with appropriate attributes.
2. **Status Streaming**: As content arrives, status lines (`* ...`) are streamed inside the processing tag.
3. **Closing**: On `</artifact>` (or other close tag), `</processing>` is emitted.
4. **Final Content**: For widgets/forms, the complete validated content follows in bulk.

### 11.3 Real-time Persistence
DB operations occur **the moment the closing tag is seen**, before `</processing>` completes. This ensures that even if the connection drops or a tool-call causes an error later in the turn, the artefacts created earlier are already saved.

### 11.4 State Diagram

```
STATE_NORMAL ──<"─<"──> STATE_BUFFERING ──match──> STATE_SECONDARY/PROCESSING
     │                      │                           │
     │                      │ no match                  │ content arrives
     │                      ▼                           ▼
     │<───────────────── flush                    emit status lines
     │                                               │
     │<──────────────────────────────────────────── close tag
     │                                               │
     │                                          emit </processing>
     │                                          + final content
     ▼                                               │
STATE_NORMAL <──────────────────────────────────────┘
```

## 12. REPL Text Tools

In-session named buffers (`enable_repl_tools=True`, default) for navigating large tool
outputs without re-injecting them into the context window.

| Tool | Signature | Purpose |
|---|---|---|
| `text_store` | `(handle, content)` | Ingest; returns format + compact summary |
| `text_search` | `(handle, query, max_results, field)` | Keyword/regex search |
| `text_get_range` | `(handle, start, end)` | Records `[start..end]` inclusive |
| `text_get_record` | `(handle, index)` | One full record |
| `text_list_records` | `(handle, page, page_size)` | Paginated listing |
| `text_filter` | `(handle, field, op, value, new_handle)` | Filter: `eq ne gt lt gte lte contains startswith regex` |
| `text_aggregate` | `(handle, operation, field)` | `count sum min max avg unique unique_count` |
| `text_to_artefact` | `(handle, title, artefact_type, language)` | Persist buffer as artefact |
| `text_list_buffers` | `()` | List all active buffers |

Formats auto-detected: `json_array`, `jsonl`, `csv`, `md_table`, `numbered_list`,
`sections`, `lines`.

---

## 12. Context Compression

Runs automatically when `max_context_size` is set and token count exceeds
`max_context_size × 0.80`.

Two strategies:
- **History compression** — keeps newest `max(4, len(branch)//4)` turns; LLM summarises the rest; result stacked in `pruning_summary`.
- **Artefact pressure** — when artefacts dominate, `deactivate_artefacts` tool is injected so the LLM can choose what to offload.

Cache key: `SHA-1(branch_tip_id + "|" + sorted_active_artefact_ids)`. Cache hit skips the LLM summarisation call.

`MSG_TYPE_CONTEXT_COMPRESSION` meta fields: `tokens_before`, `tokens_after`, `budget`,
`cache_hit`, `summary_generated`, `artefact_pressure`, `messages_pruned`.

---

## 13. Scratchpad Placement Model

`export()` injects the scratchpad as a `role: system` message immediately **after the
last user message** (not in the system-prompt header). An empty scratchpad produces no
output at all — no blank system messages.

```
[system: instructions + data zones + artefacts]
...history...
[user:   current question]
[system: == TOOL OUTPUT SCRATCHPAD ==]   ← injected only when non-empty
```

The scratchpad is never written to the database and is cleared at the end of every
`chat()` call.

---

## 14. Source Title Extraction

When RAG chunks arrive without explicit titles, `_extract_content_title()` derives a
label via a priority chain (no LLM call):

1. `chunk["title"]` / `chunk["metadata"]["title"]`
2. `chunk["metadata"]["filename"]` / `chunk["metadata"]["name"]`
3. Source path basename
4. Content analysis: Markdown heading → RST underline → YAML/JSON `title:` → HTML `<title>` → bold/italic opener → first-line heuristic
5. `"Source N"` fallback

---

## 15. `chat()` Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `user_message` | — | The user's input text |
| `personality` | `None` | `LollmsPersonality`; defaults to `NullPersonality` |
| `branch_tip_id` | `None` | Explicit branch tip; defaults to `active_branch_id` |
| `tools` | `None` | Dict of external tool specs |
| `swarm` | `None` | List of `Agent` objects — delegates to `SwarmOrchestrator` |
| `swarm_config` | `None` | `SwarmConfig`; defaults to `SwarmConfig()` when swarm is set |
| `add_user_message` | `True` | If `False`, re-uses current branch tip as user message |
| `max_reasoning_steps` | `20` | Maximum agentic loop iterations |
| `images` | `None` | List of base64 image strings |
| `remove_thinking_blocks` | `True` | Strip `<think>...</think>` from the final response |
| `enable_image_generation` | `True` | Auto-disabled if `lollmsClient.tti is None` |
| `enable_image_editing` | `True` | Auto-disabled if `lollmsClient.tti is None` |
| `auto_activate_artefacts` | `True` | New artefacts / notes / skills are `active=True` |
| `enable_show_tools` | `True` | Register `show_tools` built-in |
| `enable_extract_artefact` | `True` | Register `extract_artefact_text` built-in |
| `enable_final_answer` | `True` | Register `final_answer` signal built-in |
| `enable_request_clarification` | `True` | Register `request_clarification` built-in |
| `enable_repl_tools` | `True` | Register the nine REPL text tools |
| `enable_inline_widgets` | `True` | Process `<lollms_inline>` tags |
| `enable_notes` | `True` | Process `<note>` tags |
| `enable_skills` | `False` | Process `<skill>` tags |
| `enable_forms` | `True` | Process `<lollms_form>` tags |
| `enable_silent_artefact_explanation` | `True` | Auto-generate summary when response is all XML |
| `streaming_callback` | `None` | Callback function (see §2.1) |
| `decision_temperature` | `0.3` | Temperature for intent / structured calls |
| `final_answer_temperature` | `0.7` | Temperature for the final generation pass |
| `rag_top_k` | `5` | Maximum RAG chunks per query |
| `rag_min_similarity_percent` | `0.5` | Minimum similarity score for RAG results |
| `preflight_rag` | `True` | Run preflight RAG when personality has a data source |

---

## 16. `chat()` Return Value Reference

```python
result = discussion.chat(...)
```

| Key | Type | Description |
|---|---|---|
| `user_message` | `LollmsMessage` | The user turn message |
| `ai_message` | `LollmsMessage` | The assistant's final message |
| `sources` | `list[dict]` | All RAG sources collected this turn |
| `scratchpad` | `dict \| None` | Scratchpad state if agentic, else `None` |
| `self_corrections` | `list \| None` | Self-correction log |
| `artefacts` | `list[dict]` | Every artefact created or modified this turn |
| `agent_messages` | `list[LollmsMessage]` | **Swarm only** — one per agent per round |
| `hlf_log` | `list[dict]` | **Swarm only** — full HLF message log |
| `swarm_meta` | `dict` | **Swarm only** — run stats |

`result["ai_message"].metadata` key fields: `mode`, `duration_seconds`, `token_count`,
`tokens_per_second`, `tool_calls`, `events`, `sources`, `artefacts_modified`,
`inline_widgets`, `swarm_mode`, `swarm_synthesis`, `swarm_agents`, `swarm_rounds`.

---

## 17. The Swarm System

### 17.1 Concept

A swarm run turns one `chat()` call into a **multi-agent deliberation session**. A team
of `Agent` instances each contribute their perspective across multiple rounds, collaborate
on shared artefacts, and converge on a synthesised answer from a designated moderator.

```python
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_swarm import SwarmConfig

agents = [
    Agent(lc=lc, personality=proposer_personality, role=AgentRole.PROPOSER),
    Agent(lc=lc, personality=critic_personality,   role=AgentRole.CRITIC),
    Agent(lc=lc, personality=synth_personality,    role=AgentRole.SYNTHESIZER),
]
result = discussion.chat(
    "Design a fault-tolerant distributed cache.",
    swarm=agents,
    swarm_config=SwarmConfig(mode="quality", max_rounds=3),
    streaming_callback=cb,
)
```

### 17.2 The Agent Dataclass

| Field | Default | Description |
|---|---|---|
| `lc` | required | `LollmsClient` binding |
| `personality` | required | `LollmsPersonality` — provides system prompt and name |
| `name` | `None` | Display name override |
| `role` | `"freeform"` | Semantic role (`AgentRole.*`) |
| `model_params` | `{}` | Extra kwargs for every generation call |
| `max_tokens_per_turn` | `1024` | Soft cap on tokens per round |

### 17.3 AgentRole Reference

`PROPOSER` · `CRITIC` · `DEVIL_ADVOCATE` · `DOMAIN_EXPERT` · `SYNTHESIZER` ·
`MODERATOR` · `IMPLEMENTER` · `TESTER` · `NARRATOR` · `PLAYER` · `FREEFORM`

### 17.4 The HLF Protocol

After each agent's NLP turn, a small structured call produces a `HLFMessage`:

| Field | Description |
|---|---|
| `from` | Agent display name |
| `to` | Target agent or `"all"` |
| `type` | `proposal` / `critique` / `question` / `answer` / `vote` / `game_action` / `narration` / `directive` / `steer` |
| `confidence` | Self-reported 0.0–1.0 — drives convergence detection |
| `summary` | One-sentence summary for the log |
| `artefact_ref` | Artefact title if relevant |

Convergence exits early when mean confidence ≥ `convergence_threshold` AND no
outstanding `critique` messages (checked from round 2 onward).

### 17.5 SwarmConfig Reference

| Field | Default | Description |
|---|---|---|
| `mode` | `"quality"` | `"quality"` / `"debate"` / `"simulation"` / `"game"` / `"freeform"` |
| `max_rounds` | `3` | Maximum deliberation rounds (1–20) |
| `convergence_threshold` | `0.85` | Mean confidence for early stopping; `1.0` disables |
| `show_deliberation` | `True` | Stream each agent's NLP to the user |
| `moderator_index` | `0` | Index of the synthesising agent |
| `allow_artefact_collaboration` | `True` | Agents share the discussion's artefact store |
| `anti_sycophancy_strength` | `"medium"` | `"light"` / `"medium"` / `"strong"` |
| `synthesis_prompt_suffix` | `""` | Extra instructions for the final synthesis pass |
| `game_rules` | `""` | For `"game"` mode — injected into every agent's system prompt |
| `world_state` | `""` | For `"simulation"` mode — initial world description |
| `max_nlp_tokens_per_agent` | `512` | Soft cap on visible NLP per agent per round |

### 17.6 Anti-Sycophancy

Rules injected from round 2 onward prevent empty agreement:
- **`"light"`** — add new value; brief acknowledgement allowed.
- **`"medium"`** (default) — no simple confirmation; must add angle / counter-example / risk.
- **`"strong"`** — adversarial: find a flaw before agreeing; confidence ≤ 0.6 on first pass.

### 17.7 Execution Flow

```
chat(swarm=[...]) → add user message → SwarmOrchestrator.run()
  FOR round in 1..max_rounds:
    <processing type="synthesis" round="N">  ← opens for each round
    FOR each agent:
      build system_prompt + hlf_context + nlp_prompt
      agent.generate(streaming) → save as LollmsMessage
      _post_process_llm_response → artefacts, notes, widgets
      generate_structured → HLFMessage
    check convergence → maybe break early
    </processing>  ← closes at round end
  moderator.generate(synthesis_prompt) → final message
  return {user_message, ai_message, agent_messages, hlf_log, artefacts, swarm_meta}
```

**Synthesis Processing Stream**

During swarm execution, each round is wrapped in a `<processing type="synthesis">` tag:

```xml
<processing type="synthesis" round="1" agents="3">
* Proposer: Introducing distributed cache architecture...
* Critic: Identifying potential consistency issues...
* Synthesizer: Integrating perspectives...
</processing>
```

The synthesis processing tag helps UIs show:
- Which round is currently executing
- How many agents are participating
- Per-agent status updates (when streamed with `show_deliberation=True`)

### 16.8 Shared Artefact Collaboration

All agents share `discussion.artefacts`. Any agent may create, patch, revert, or save
notes via XML tags. Fuzzy matching (§4.2.2) ensures patches target the right artefact
even when the tester spells the name differently from the implementer.

### 17.9 User Steering

```python
orchestrator.steer("Focus only on the security implications from here on.")
```

Delivered to all agents at the start of the next round, prefixed with
`"⚡ USER DIRECTIVE:"`, and logged in the HLF as `type="steer"`.

### 17.10 Swarm Result Keys

Standard: `user_message`, `ai_message`, `artefacts`, `sources`.
Swarm-only: `agent_messages`, `hlf_log`, `swarm_meta`
(`{mode, rounds_run, agents, touched_artefacts, duration_seconds}`).

### 17.11 Worked Example — Collaborative Coding

```python
dev_agents = [
    Agent(lc=lc,
          personality=LollmsPersonality(name="Dev",
              system_prompt="Write clean Python. Use <artefact> tags for code files."),
          role=AgentRole.IMPLEMENTER),
    Agent(lc=lc,
          personality=LollmsPersonality(name="Reviewer",
              system_prompt="Find bugs and security issues. Patch with SEARCH/REPLACE."),
          role=AgentRole.TESTER),
]

result = discussion.chat(
    "Implement a thread-safe LRU cache with TTL per key.",
    swarm=dev_agents,
    swarm_config=SwarmConfig(
        mode="quality", max_rounds=3,
        anti_sycophancy_strength="strong",
        moderator_index=0,
        synthesis_prompt_suffix="Produce the final corrected implementation.",
    ),
    streaming_callback=cb,
    enable_notes=True,
)

impl = discussion.artefacts.get("lru_cache.py")
print(impl["content"])
print(f"Version: {impl['version']}")   # > 1 after tester's patches
```

---

## 20. UI Implementation Guide

### 20.1 Unified Streaming Protocol (Recommended Approach)

The new unified streaming protocol (§2.4.1) simplifies UI implementation by delivering
all secondary content through a single channel with predictable XML tags.

**Recommended Parsing Strategy**

```python
import re
from dataclasses import dataclass
from typing import Optional, List, Callable

@dataclass
class BuildOperation:
    type: str           # artefact_building, widget_building, etc.
    title: str
    attrs: dict
    status_lines: List[str]
    final_content: str = ""
    is_complete: bool = False

class UnifiedStreamParser:
    def __init__(self):
        self.current: Optional[BuildOperation] = None
        self.completed: List[BuildOperation] = []
        
    def on_chunk(self, text: str, msg_type, meta: dict):
        """Call this for every MSG_TYPE_CHUNK event."""
        
        # Handle processing meta events
        if meta.get("type") == "processing_open":
            self.current = BuildOperation(
                type=meta["processing_type"],
                title=meta["title"],
                attrs=meta.get("attrs", {}),
                status_lines=[]
            )
            self.on_build_start(self.current)
            
        elif meta.get("type") == "processing_status":
            if self.current:
                self.current.status_lines.append(meta["status"])
                self.on_status_update(self.current, meta["status"])
                
        elif meta.get("type") == "processing_close":
            if self.current:
                self.current.is_complete = True
                self.completed.append(self.current)
                self.on_build_complete(self.current)
                self.current = None
                
        elif meta.get("type") == "processing_final_content":
            if self.current:
                self.current.final_content = text  # or from meta
                self.on_final_content(self.current)
        
        # Also parse raw text for direct XML detection (fallback)
        self._parse_raw_text(text)
    
    def _parse_raw_text(self, text: str):
        """Fallback: detect processing tags directly in stream."""
        # Detect opening tag
        proc_open = re.search(
            r'<processing\s+type="([^"]+)"\s+title="([^"]+)"([^>]*)>',
            text
        )
        if proc_open and not self.current:
            attrs = {}
            # Parse additional attributes
            attr_str = proc_open.group(3)
            for match in re.finditer(r'(\w+)="([^"]*)"', attr_str):
                attrs[match.group(1)] = match.group(2)
            
            self.current = BuildOperation(
                type=proc_open.group(1),
                title=proc_open.group(2),
                attrs=attrs,
                status_lines=[]
            )
        
        # Detect status lines
        if self.current and "* " in text:
            for line in text.split("\n"):
                if line.strip().startswith("* "):
                    status = line.strip()[2:]
                    self.current.status_lines.append(status)
        
        # Detect closing tag
        if "</processing>" in text and self.current:
            self.current.is_complete = True
            self.completed.append(self.current)
            self.on_build_complete(self.current)
            self.current = None
    
    # Override these in your UI
    def on_build_start(self, op: BuildOperation): pass
    def on_status_update(self, op: BuildOperation, status: str): pass
    def on_build_complete(self, op: BuildOperation): pass
    def on_final_content(self, op: BuildOperation): pass
```

**Rendering by Operation Type**

```python
def render_build_operation(op: BuildOperation):
    """Render a completed build operation in the UI."""
    
    if op.type == "artefact_building":
        # Artefacts live in discussion space - show summary
        return {
            "component": "ArtefactCard",
            "props": {
                "title": op.title,
                "type": op.attrs.get("art_type", "document"),
                "language": op.attrs.get("language"),
                "status_lines": op.status_lines,
            }
        }
    
    elif op.type == "widget_building":
        # Widgets have final_content with the actual HTML
        # Extract from <lollms_inline> tag
        import re
        inline_match = re.search(
            r'<lollms_inline[^>]*>(.*?)</lollms_inline>',
            op.final_content,
            re.DOTALL
        )
        if inline_match:
            return {
                "component": "InteractiveWidget",
                "props": {
                    "title": op.title,
                    "widget_type": op.attrs.get("widget_type", "html"),
                    "source": inline_match.group(1),
                    "sandbox": True,
                }
            }
    
    elif op.type == "form_building":
        # Forms have final_content with the form definition
        import re
        form_match = re.search(
            r'<lollms_form[^>]*>(.*?)</lollms_form>',
            op.final_content,
            re.DOTALL
        )
        if form_match:
            # Parse form descriptor
            return {
                "component": "InteractiveForm",
                "props": {
                    "title": op.title,
                    "form_data": parse_form_xml(form_match.group(0)),
                }
            }
    
    elif op.type == "synthesis":
        # Synthesis shows status then final text
        return {
            "component": "SynthesisPanel",
            "props": {
                "round": op.attrs.get("round"),
                "status_lines": op.status_lines,
            }
        }
```

### 20.2 Legacy: Mounting Interactive Components (Deprecated)

> **DEPRECATED**: The anchor-tag approach is maintained for backward compatibility.
> New applications should use the unified `<processing>` protocol (§20.1).

For `lollms_inline` and `lollms_form`, the framework inserts "anchor tags" into the `ai_message.content`. Your UI should scan for these and replace them with live components.

| Tag | Purpose | Source Data Location |
|---|---|---|
| `<lollms_widget id="UUID" />` | Interactive Lab/Demo | `message.metadata['inline_widgets']` |
| `<lollms_form_anchor id="UUID" />` | Structured Input Form | `message.metadata['forms']` |

### 20.3 Handling "The clean bubble"
Because the framework removes the XML tags during streaming, `ai_message.content` only contains the text the LLM intended for the user. 
- **DO NOT** try to regex-extract tags from the final content; they aren't there.
- **DO** use the `affected_artefacts` list returned by `chat()` to know what changed.
- **DO** listen for `MSG_TYPE_ARTEFACTS_STATE_CHANGED` for real-time sidebar updates.
- **DO** parse the `<processing>` stream for real-time build status (new approach).

---

---

## 21. Migration Guide: Legacy to Unified Streaming

### 21.1 What Changed

| Before (Legacy) | After (Unified) |
|---|---|
| Separate `MSG_TYPE_ARTEFACT_CHUNK` stream | `<processing type="artefact_building">` in main stream |
| `MSG_TYPE_WIDGET_DONE` with content | `<processing>` status + bulk `<lollms_inline>` after |
| Anchor tags in final message | Same, but now preceded by processing status |
| Multiple event types to handle | Single `MSG_TYPE_CHUNK` with `meta["type"]` variants |

### 21.2 Minimal Migration

**Before (legacy handler):**
```python
def callback(text, msg_type, meta):
    if msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK:
        update_sidebar(meta["title"], text)
    elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_DONE:
        finalize_sidebar(meta["title"])
```

**After (unified handler):**
```python
def callback(text, msg_type, meta):
    # New: handle processing stream
    if meta.get("type") == "processing_open":
        if meta["processing_type"] == "artefact_building":
            update_sidebar(meta["title"], "")  # placeholder
    elif meta.get("type") == "processing_status":
        if meta["processing_type"] == "artefact_building":
            update_sidebar(meta["title"], meta["status"])
    elif meta.get("type") == "processing_close":
        if meta["processing_type"] == "artefact_building":
            finalize_sidebar(meta["title"])
    
    # Keep legacy for backward compatibility
    elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK:
        pass  # now handled above, or keep for old code
```

### 21.3 Full Migration Checklist

- [ ] Update callback to parse `meta["type"]` for processing events
- [ ] Add UI component for showing `<processing>` status inline
- [ ] Handle bulk final content for widgets/forms after `</processing>`
- [ ] Test with `enable_inline_widgets=True` and `enable_forms=True`
- [ ] Remove legacy handlers once confident (optional)

---

## 22. Worked Example (single-agent)

Full turn: external tool, REPL text tools, notes, skills, artefact, inline widget.

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
    lollms_client=lc, db_manager=db, autosave=True,
    system_prompt="You are a research analyst.", max_context_size=8192,
)

def search_arxiv(query: str, max_results: int = 20) -> dict:
    papers = arxiv_api.search(query, max_results=max_results)
    return {"success": True, "content": json.dumps(papers), "count": len(papers)}

tools = {
    "search_arxiv": {
        "name": "search_arxiv",
        "description": "Search arXiv for academic papers by keyword",
        "parameters": [
            {"name": "query",       "type": "str", "optional": False},
            {"name": "max_results", "type": "int", "optional": True, "default": 20},
        ],
        "output":   [{"name": "content", "type": "str"}],
        "callable": search_arxiv,
    }
}

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
                print(f"\n  [Widget building] {c['title']} ({c['widget_type']})")
            elif not text and t == "form_start":
                print(f"\n  [Form building] {meta['content']['title']}")
            elif not text and t == "artefact_update":
                print(f"\n  [Artefact streaming] {meta['content']['title']}")
            else:
                print(text, end="", flush=True)
        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n> {text}")
        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n[Tool] {meta['tool']}")
        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            if "artefact" in meta:
                op = "created" if meta["is_new"] else "updated"
                print(f"\n  [Artefact {op}] {meta['artefact']['title']} v{meta['artefact']['version']}")

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
    enable_inline_widgets=True,
    enable_notes=True,
    enable_skills=True,
    enable_forms=True,
    enable_silent_artefact_explanation=True,
    max_reasoning_steps=15,
)

artefacts = result["artefacts"]
notes  = [a for a in artefacts if a["type"] == "note"]
skills = [a for a in artefacts if a["type"] == "skill"]
docs   = [a for a in artefacts if a["type"] not in ("note", "skill", "image")]
widgets = result["ai_message"].metadata.get("inline_widgets", [])
forms = result["ai_message"].metadata.get("forms", [])

print(f"\nDocuments : {[a['title'] for a in docs]}")
print(f"Notes     : {[a['title'] for a in notes]}")
print(f"Skills    : {[(a['title'], a.get('category')) for a in skills]}")
print(f"Widgets   : {[w['title'] for w in widgets]}")
print(f"Forms     : {[f['title'] for f in forms]}")
print(f"Sources   : {[s['title'] for s in result['sources']]}")
print(f"Mode      : {result['ai_message'].metadata['mode']}")
print(f"Duration  : {result['ai_message'].metadata['duration_seconds']}s")
```

During this turn the LLM will typically:

1. Call `search_arxiv` — UI shows `<processing type="tool_execution" tool="search_arxiv">` with live status updates, then returns large JSON.
2. Call `text_store(handle="papers", content=<json>)` — compact summary enters context.
3. Call `text_filter(...)` to isolate papers after 2023.
4. Call `text_list_records` / `text_get_record` to read abstracts.
5. Call `text_to_artefact(...)` to persist the filtered dataset.
6. Emit `<note title="Top 5 Summary">...</note>` — saved, stripped, real-time event fires.
7. Emit `<skill title="arXiv Research Workflow" category="research/workflow">...</skill>` — saved, real-time event fires.
8. Emit `<artefact name="context_compression_report.md" type="document">...</artefact>` — saved, real-time event fires.
9. Emit `<lollms_inline type="html" title="Publication Counts">...</lollms_inline>` — bar chart rendered in-place.
10. Emit `<lollms_form title="Citation Preferences">...</lollms_form>` — interactive form pauses generation.
11. If only XML tags were produced, the silent artefact guard generates a summary line for each item.