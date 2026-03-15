# lollms_discussion — Developer Reference

> **Scope** — This document covers every subsystem of `lollms_discussion` in detail:
> the layered context model, the event/callback system, conversation branching,
> artefacts (and how the LLM interacts with them), notes, skills, inline widgets,
> the full tool catalogue (external and built-in), the REPL text tools, context
> compression, the scratchpad placement model, and the aider patch system.

---

## Table of Contents

1. [Core Architecture](#1-core-architecture)
2. [The Event System](#2-the-event-system)
3. [The Branching System](#3-the-branching-system)
4. [The Artefact System](#4-the-artefact-system)
5. [Notes](#5-notes)
6. [Skills](#6-skills)
7. [Inline Interactive Widgets](#7-inline-interactive-widgets)
8. [The Tooling System](#8-the-tooling-system)
9. [REPL Text Tools](#9-repl-text-tools)
10. [Context Compression](#10-context-compression)
11. [Scratchpad Placement Model](#11-scratchpad-placement-model)
12. [Source Title Extraction](#12-source-title-extraction)
13. [`chat()` Parameter Reference](#13-chat-parameter-reference)
14. [`chat()` Return Value Reference](#14-chat-return-value-reference)
15. [Worked Example](#15-worked-example)

---

## 1. Core Architecture

### 1.1 What Is a `LollmsDiscussion`?

A `LollmsDiscussion` is not a simple list of messages. It is a **stateful, database-backed,
branch-aware object** that:

- assembles a complete LLM context from multiple independent layers every turn,
- manages versioned artefacts (code, documents) that the LLM can create and patch inline,
- saves structured notes and reusable skills as first-class typed artefacts,
- embeds live interactive widgets directly inside chat messages,
- routes RAG queries through preflight retrieval and agentic tool calls,
- executes inline `<tool_call>` tags in a streaming loop with anti-duplication guards,
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

### 2.2 Complete MSG_TYPE Reference

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
| 29 | `MSG_TYPE_ARTEFACTS_STATE_CHANGED` | JSON list of affected titles | `{artefacts: [...], action?: str}` |
| 30 | `MSG_TYPE_TOOLS_LIST` | JSON tool catalogue | `{tools: [{name, description, parameters, source}, ...]}` |
| 31 | `MSG_TYPE_CONTEXT_COMPRESSION` | JSON compression report | `{tokens_before, tokens_after, budget, cache_hit, summary_generated, artefact_pressure}` |

### 2.3 Inline Streaming Events (MSG_TYPE_CHUNK with meta)

Some events arrive as `MSG_TYPE_CHUNK` with `text=""` but a non-empty `meta` dict.
These fire in real-time as the LLM streams, before `_post_process_llm_response` runs,
allowing the UI to show placeholders immediately.

| `meta["type"]` | When fired | `meta["content"]` fields |
|---|---|---|
| `"artefact_update"` | Opening `<artefact name="...">` tag detected | `{title: str}` |
| `"note_start"` | Opening `<note title="...">` tag detected | `{title: str}` |
| `"skill_start"` | Opening `<skill title="..." category="...">` tag detected | `{title: str, category: str}` |
| `"inline_widget_start"` | Opening `<lollms_inline ...>` tag detected | `{title: str, widget_type: "html"|"react"|"svg"}` |

### 2.4 Step Events — Named Phases

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

### 2.5 A Complete Callback Implementation

```python
from lollms_client.lollms_types import MSG_TYPE

def on_event(text, msg_type, meta):
    match msg_type:
        case MSG_TYPE.MSG_TYPE_CHUNK:
            t = meta.get("type")
            if not text and t == "artefact_update":
                print(f"\n  [Artefact] {meta['content']['title']}")
            elif not text and t == "note_start":
                print(f"\n  [Note] {meta['content']['title']}")
            elif not text and t == "skill_start":
                c = meta["content"]
                print(f"\n  [Skill] {c['title']}  [{c['category']}]")
            elif not text and t == "inline_widget_start":
                c = meta["content"]
                print(f"\n  [Widget] {c['title']} ({c['widget_type']})")
            else:
                print(text, end="", flush=True)

        case MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n  > {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"  < {text}  [{meta['id'][:8]}]")

        case MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"\n  [Tool] {meta['tool']}({meta['params']})")

        case MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"  [Result] {meta['tool']} -> {text[:120]}")

        case MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
            action = meta.get("action", "updated")
            print(f"\n  [Artefacts {action}] {text}")

        case MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION:
            if meta["cache_hit"]:
                print(f"\n  [Compression cache hit] "
                      f"{meta['tokens_before']:,} -> {meta['tokens_after']:,}")
            elif meta["summary_generated"]:
                print(f"\n  [Compressed {meta['messages_pruned']} messages] "
                      f"{meta['tokens_before']:,} -> {meta['tokens_after']:,}")

        case MSG_TYPE.MSG_TYPE_SOURCES_LIST:
            for src in (meta if isinstance(meta, list) else []):
                score = src.get("relevance_score", src.get("score", 0))
                print(f"  [Source] [{src['title']}]  score={score:.2f}")

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

#### 4.2.2 Patch an Existing Artefact — SEARCH/REPLACE (Aider Format)

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

#### 4.2.3 Revert to a Previous Version

```xml
<revert_artefact name="app.py" version="2" />
```

```python
# Equivalent Python API
discussion.artefacts.revert("app.py", target_version=2)
```

#### 4.2.4 Image Generation and Editing (TTI)

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

# Activate / deactivate
mgr.activate("readme.md")
mgr.deactivate("old_draft.md")

# Revert
mgr.revert("app.py", target_version=1)

# Delete permanently
mgr.remove("scratch.txt")
```

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

---

## 7. Inline Interactive Widgets

### 7.1 Concept

Inline widgets let the LLM embed a **live, self-contained interactive element** directly
inside a chat message — a formula explorer with sliders, a mini chart, an SVG animation,
a colour picker, a physics simulation. Unlike artefacts (which go to a side panel),
widgets render *in-place* inside the chat bubble.

Enabled by `enable_inline_widgets=True` (the default). No TTI or external tool required.

### 7.2 The `<lollms_inline>` Tag

```xml
<lollms_inline type="html" title="Circle Circumference Explorer">
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>
</head>
<body>
  <div id="formula"></div>
  <label>r <input id="r" type="range" min="0.5" max="10" step="0.01" value="4.5">
    <span id="rv">4.50</span></label>
  <p id="circ"></p>
  <canvas id="c" width="300" height="300"></canvas>
  <script>
    const slider = document.getElementById('r');
    katex.render('C = 2\\pi r', document.getElementById('formula'), {throwOnError: false});
    function draw() {
      const r = parseFloat(slider.value);
      document.getElementById('rv').textContent = r.toFixed(2);
      document.getElementById('circ').textContent = 'C = 2pir ~= ' + (2*Math.PI*r).toFixed(2);
      const ctx = document.getElementById('c').getContext('2d');
      ctx.clearRect(0,0,300,300);
      ctx.beginPath();
      ctx.arc(150,150,Math.min(r*20,130),0,2*Math.PI);
      ctx.strokeStyle='#4A9EFF'; ctx.lineWidth=2; ctx.stroke();
    }
    slider.oninput = draw; draw();
  </script>
</body>
</html>
</lollms_inline>
```

| Attribute | Values | Default | Description |
|---|---|---|---|
| `type` | `html`, `react`, `svg` | `html` | Widget source format |
| `title` | any string | `"Interactive Widget"` | Displayed above the widget in the UI |

### 7.3 What Happens During Post-Processing

1. Source extracted and stored in `ai_message.metadata["inline_widgets"]`:
   ```python
   {"id": "<uuid>", "type": "html", "title": "Circle...", "source": "<!DOCTYPE...>"}
   ```
2. Tag replaced in stored message text with `<lollms_widget id="<uuid>" />` anchor.
3. Frontend uses the anchor `id` to render the widget in-place (iframe, shadow DOM, etc.).

### 7.4 Streaming Notification

```python
# MSG_TYPE_CHUNK with text="" and meta:
{"type": "inline_widget_start", "content": {"title": "Circle...", "widget_type": "html"}}
```

### 7.5 Rules and Constraints

- **Fully self-contained** — CDN links to well-known libraries are the only allowed external reference.
- **Compact** — aim for <= 420 px height, full container width.
- **No `alert()` / `confirm()` / `prompt()`.**
- **KaTeX for math** — `https://cdn.jsdelivr.net/npm/katex/`.
- **Not for large applications** — use `<artefact>` for anything needing persistence or export.

### 7.6 Disabling Inline Widgets

```python
result = discussion.chat("Explain quicksort.", enable_inline_widgets=False)
```

---

## 8. The Tooling System

### 8.1 Mechanics

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

### 8.2 Fast Path — No External Tools

When no external tools, personality tools, or RAG sources are registered, the framework
**skips all agentic scaffolding** and goes directly to a single LLM call. The model may
still emit `<artefact>`, `<note>`, `<skill>`, and `<lollms_inline>` XML tags, which are
post-processed as normal.

This eliminates the runaway-loop problem for straightforward requests (e.g. "analyse
these screenshots and create a note") where no external data is needed.

### 8.3 Anti-Duplication Guards

#### 8.3.1 Agent State Header

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

#### 8.3.2 Hard Deduplication Gate

Before any tool executes, `_identical_call_counts[signature]` is checked:

| Count | Behaviour |
|---|---|
| 1 (first call) | Executes normally |
| 2 (first repeat) | Blocked; synthetic `DUPLICATE CALL BLOCKED` result injected |
| 3+ | Loop breaks unconditionally |

**The same tool with different parameters is always allowed.** Multi-step research
calling the same search tool with six different queries will execute all six.

### 8.4 Registering External Tools

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

#### 8.4.1 RAG Tools

Any tool whose `output` spec contains a field named `"sources"` is automatically
registered as a RAG tool. Its results are tracked in `result["sources"]`, emitted
as `MSG_TYPE_SOURCES_LIST` events, and subject to `rag_top_k` / `rag_min_similarity_percent`.

### 8.5 Built-in Framework Tools

All built-in tools default to `True`. The personality tools step event is only emitted
when at least one personality tool is found.

#### 8.5.1 `show_tools` (`enable_show_tools=True`)

Assembles a complete tool catalogue and fires `MSG_TYPE_TOOLS_LIST`.

#### 8.5.2 `extract_artefact_text` (`enable_extract_artefact=True`)

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

#### 8.5.3 `final_answer` (`enable_final_answer=True`)

Signal tool for composable-answer workflows.

#### 8.5.4 `request_clarification` (`enable_request_clarification=True`)

Lets the LLM pause and ask the user a question mid-turn.

#### 8.5.5 `deactivate_artefacts` (context-pressure only)

Conditionally registered when `artefact_tokens > history_tokens AND total > budget`.

```json
{"name": "deactivate_artefacts", "parameters": {"titles": ["old_draft.md"]}}
```

### 8.6 Preflight RAG (Personality)

When a personality is provided, `chat()` runs a preflight RAG pass before the first
generation step. Results are injected into `self.scratchpad` and placed after the last
user message by `export()`. Sources appear in `result["sources"]` with `phase="preflight"`.

### 8.7 `simplified_chat()` — Lighter RAG Path

```python
result = discussion.simplified_chat(
    user_message="What are the best indexing strategies for our schema?",
    rag_data_stores={
        "db_docs": lambda q: vector_db.search(q, top_k=5),
        "wiki":    lambda q: wiki_search(q),
    },
    streaming_callback=on_event,
)
```

Three code paths: **fast path** (trivial message), **memory hit**, **full path**
(intent detection -> optional RAG -> stream final answer).

---

## 9. REPL Text Tools

### 9.1 Purpose

MCP tools and RAG pipelines can return very large payloads. The REPL text tools solve
this with a **named in-session buffer** — the model stores a large result, gets back
only a compact summary, then navigates it with targeted calls.

**Toggle:** `enable_repl_tools=True` (default). All buffers are ephemeral — they exist
only for the duration of a single `chat()` call. Use `text_to_artefact` to persist.

### 9.2 Format Auto-Detection

| Format | Detection rule |
|---|---|
| `json_array` | Top-level JSON `[{...}, ...]` |
| `jsonl` | >= 3 of the first 10 lines parse as standalone JSON objects |
| `csv` | `csv.Sniffer` detects a consistent delimiter; `DictReader` yields >= 2 fields |
| `md_table` | Pipe-delimited lines with a `|---|` separator row |
| `numbered_list` | >= 5 lines matching `1. ...` / `- ...` / `* ...` patterns |
| `sections` | >= 3 `## Heading` blocks |
| `lines` | Fallback — every non-blank line is one record |

### 9.3 The Nine REPL Tools

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

## 10. Context Compression

### 10.1 Trigger

Compression runs automatically when `max_context_size` is set and the current token
count exceeds `max_context_size x 0.80`.

### 10.2 Strategy

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

### 10.3 `MSG_TYPE_CONTEXT_COMPRESSION` Meta Fields

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

## 11. Scratchpad Placement Model

### 11.1 Why Position Matters

The scratchpad holds the full text of every tool result accumulated during an agentic
turn. Placing it in the system-prompt header (far from the user's question) makes the
model less likely to attend to it. Placing it immediately before the model's continuation
point maximises its impact.

### 11.2 Empty Scratchpad Suppression

An empty scratchpad produces **no output at all** — no blank system message, no empty
block. `export()` checks `_scratchpad.strip()` before injecting anything.

### 11.3 Injection Per Format

| `export()` format | Injection mechanism |
|---|---|
| `lollms_text` | `!@>system:\n== TOOL OUTPUT SCRATCHPAD ==\n...` after the last `!@>user:` block |
| `openai_chat` | `{"role": "system", "content": "== TOOL OUTPUT SCRATCHPAD ==\n..."}` after the last user dict |
| `ollama_chat` | Same structure as `openai_chat` |
| `markdown` | `**system**: == TOOL OUTPUT SCRATCHPAD ==\n...` after the last user line |

### 11.4 Lifecycle Within a Single `chat()` Call

```
chat() starts
  -> self.scratchpad = ""                    # reset

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
```

The scratchpad is **never written to the database**. It is purely a per-turn
communication channel between the tool executor and the LLM.

---

## 12. Source Title Extraction

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

## 13. `chat()` Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `user_message` | — | The user's input text |
| `personality` | `None` | `LollmsPersonality` instance; defaults to `NullPersonality` |
| `branch_tip_id` | `None` | Explicit branch tip; defaults to `active_branch_id` |
| `tools` | `None` | Dict of external tool specs (see §8.4) |
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
| `enable_inline_widgets` | `True` | Process `<lollms_inline>` tags; inject widget instructions |
| `enable_notes` | `True` | Process `<note>` tags; save as NOTE artefacts |
| `enable_skills` | `False` | Process `<skill>` tags; save as SKILL artefacts |
| `streaming_callback` | `None` | Callback function (see §2.1) |
| `decision_temperature` | `0.3` | Temperature for intent-detection / structured calls |
| `final_answer_temperature` | `0.7` | Temperature for the final generation pass |
| `rag_top_k` | `5` | Maximum RAG chunks per query |
| `rag_min_similarity_percent` | `0.5` | Minimum similarity score for RAG results |
| `preflight_rag` | `True` | Run preflight RAG pass when personality has a data source |

---

## 14. `chat()` Return Value Reference

```python
result = discussion.chat(...)
```

| Key | Type | Description |
|---|---|---|
| `user_message` | `LollmsMessage` | The user turn message object |
| `ai_message` | `LollmsMessage` | The assistant turn message object |
| `sources` | `list[dict]` | All RAG sources collected during this turn |
| `scratchpad` | `dict or None` | Scratchpad state if the turn was agentic, else `None` |
| `self_corrections` | `list or None` | Log of self-correction events |
| `artefacts` | `list[dict]` | Every artefact created or modified this turn, including notes and skills |

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

---

## 15. Worked Example

Full turn: external tool, REPL text tools, notes, skills, artefact, inline widget,
and event callback.

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
                print(f"\n  [Artefact] {meta['content']['title']}")
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
            print(f"\n[Artefacts] {text}")
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
6. Emit `<note title="Top 5 Summary">...</note>` — saved as a NOTE artefact, stripped from text.
7. Emit `<skill title="arXiv Research Workflow" category="research/workflow" description="...">...</skill>` — saved as a SKILL artefact.
8. Emit `<artefact name="context_compression_report.md" type="document">...</artefact>` — saved as a DOCUMENT artefact.
9. Emit `<lollms_inline type="html" title="Publication Counts">...</lollms_inline>` — bar chart rendered live in the chat bubble.