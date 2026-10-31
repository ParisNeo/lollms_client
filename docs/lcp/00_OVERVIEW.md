# LCP — Local Control Panel Tool Binding: Complete Documentation

> **Audience**: Humans and LLMs. Every document in this folder is written for both.
> Backticked identifiers are real code symbols. Signatures are exact where verified.
> This is the entry point — read this first, then follow the document map.

---

## 1. What is the LCP?

**LCP** (the **L**ollms **C**ommunication **P**rotocol / local tool protocol binding) is the
in-process tool execution framework of `lollms_client`. It lives at:

```
src/lollms_client/tools_bindings/lcp/
├── __init__.py          ← the LCPBinding class (the binding core)
├── description.yaml     ← binding metadata
├── README.md            ← tool-creation guidelines
└── default_tools/       ← the tool libraries (one folder per library)
    ├── word_tools/word_tools.py
    ├── excel_tools/excel_tools.py
    ├── powerpoint_tools/powerpoint_tools.py
    ├── document_editor/document_editor.py
    └── ... (~25 libraries)
```

Its role: let an LLM **discover** tools (so it knows what exists and how to call
them), **emit tool calls**, and have those calls **executed locally in the same
Python process** — no server, no network, no daemon.

### LCP vs the other tool bindings

`lollms_client` ships two tool bindings, both inheriting `LollmsToolBinding`
(defined in `src/lollms_client/lollms_tools_binding.py`):

| Binding | Transport | Where tools live | Schema source |
|---|---|---|---|
| **LCP** (`tools_bindings/lcp/`) | none — in-process | local Python files under `default_tools/` | AST parsing of source, or hand-authored `get_tools()` spec dicts |
| **MCP** (`tools_bindings/mcp/`) | `stdio` subprocess or Streamable HTTP | external MCP servers | MCP server handshake, `alias::tool_name` namespacing |

The base-class contract every binding must honor (per
`src/lollms_client/tools_bindings/README.md`):

```python
class LollmsToolBinding:
    def discover_tools(self, specific_tool_names=None, **kwargs) -> List[Dict[str, Any]]
    def list_tools(self, **kwargs) -> List[Dict[str, Any]]      # alias for discover_tools
    def execute_tool(self, tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]
```

`discover_tools` returns dicts with `name`, `description`, `input_schema`.
`execute_tool` returns a dict with `output` and `status_code` (200 success /
500 failure). LCP normalizes library results into exactly this shape — see
[05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md).

---

## 2. The three consumption surfaces

The LCP is consumed from three directions. Understanding these three surfaces is
the key to understanding everything else in this folder.

1. **Discovery surface** — `LCPBinding._discover_local_tools()` walks
   `default_tools/`, loads each library, extracts tool schemas, and produces the
   `discovered_tools` registry. This is what makes tools *exist* for the system.
   → [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md)

2. **Execution surface** — `execute_tool(tool_name, params, discussion_instance, **kwargs)`
   routes a tool call to the right library callable and normalizes the returned
   dict. → [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md)

3. **Personality/chat surface** — `LollmsPersonality._discover_tools(...)` reads
   capability flags, pulls tool specs from the binding (and from libraries'
   `get_tools()` seams), and `lollms_discussion`'s chat mixins advertise those
   specs to the LLM in the system prompt, intercept the LLM's tool calls, and
   attach result artefacts to the conversation.
   → [06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md) ·
   [07_PERSONALITY_INTEGRATION.md](./07_PERSONALITY_INTEGRATION.md)

---

## 3. Big-picture flow

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                    CAPABILITY FLAGS                     │
                 │  enable_word / enable_excel / enable_powerpoint /       │
                 │  enable_office_files / enable_code_execution / ...      │
                 │  (set by personality or discussion session_flags)       │
                 └───────────────────────────┬──────────────────────────────┘
                                             │ gates admission (fail-closed)
                                             ▼
┌──────────────────────┐   library files   ┌─────────────────────────────┐
│  default_tools/      │◄──────────────────│      LCPBinding             │
│  <name>/<name>.py    │  _iter_library_   │  _discover_local_tools()    │
│  + description.yaml  │      files()      │  _load_tool_file()          │
└──────────────────────┘                   │  _parse_tool_via_ast()      │
                                           │  _parse_tool_via_regex()    │
                                           │  → discovered_tools registry│
                                           └──────────┬──────────────────┘
                                                      │ tool specs
                       ┌──────────────────────────────┼──────────────────────────┐
                       ▼                              ▼                          ▼
        ┌─────────────────────────┐   ┌───────────────────────────┐   ┌────────────────────┐
        │  LollmsPersonality      │   │  lollms_discussion        │   │  get_tools() seam  │
        │  _discover_tools()      │   │  _mixin_prompt.py builds  │   │  (libraries that   │
        │  mounts toolsets,       │   │  the LLM-visible tool     │   │  hand-author spec │
        │  reads get_tools()      │   │  doctrine in the prompt   │   │  dicts)            │
        └───────────┬─────────────┘   └────────────┬──────────────┘   └─────────┬──────────┘
                    │                              │                            │
                    └──────────────┬───────────────┴────────────────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │           LLM             │
                     │  sees tool specs, emits   │
                     │  a tool call in its reply │
                     └─────────────┬─────────────┘
                                   │ tool call intercepted by chat mixin
                                   ▼
                     ┌───────────────────────────┐
                     │  LCPBinding.execute_tool( │
                     │    tool_name, params,     │
                     │    discussion_instance)   │
                     └─────────────┬─────────────┘
                                   │ library callable runs
                                   ▼
                     ┌───────────────────────────┐
                     │  result dict              │
                     │  success / error / output │
                     │  file_name / context_files│
                     └─────────────┬─────────────┘
                                   │ normalization + artefact attachment
                                   ▼
                     ┌───────────────────────────┐
                     │  next chat round          │
                     │  (artefacts attached as   │
                     │  USER-role messages only) │
                     └───────────────────────────┘
```

---

## 4. Document map (00 → 09)

| # | File | One-line summary |
|---|---|---|
| 00 | `00_OVERVIEW.md` | This file — entry point, consumption surfaces, big-picture flow, reader guide. |
| 01 | [01_LCP_BINDING_CORE.md](./01_LCP_BINDING_CORE.md) | Anatomy of `tools_bindings/lcp/__init__.py`: the `LCPBinding` class, lazy imports, `_load_tool_file`, `_iter_library_files`, public API. |
| 02 | [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) | The discovery funnel: AST parsing, the regex fallback and its degraded semantics, `_extract_tool_names_from_file`, mounts, and the two discovery seams. |
| 03 | [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md) | Admission control: `_OFFICE_LIBRARY_STEMS`, `_PER_SOFTWARE_GATES`, `_office_library_admitted`, `TOOL_LIBRARY_LOAD_FLAG`, fail-closed law, `docs/flags/` cross-links. |
| 04 | [04_LIBRARY_CATALOG.md](./04_LIBRARY_CATALOG.md) | Catalog of every `default_tools/` library: seam, gate flags, tools, internal architecture, workspace rule. |
| 05 | [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md) | `execute_tool` contract: result-dict normalization, `context_files` resolution, diagnostics, sequence diagram. |
| 06 | `06_CHAT_MIXIN_INTEGRATION.md` | *(Step 2)* How `_mixin_chat.py` / `_mixin_prompt.py` advertise tools, dispatch tool calls, mount office toolsets, handle media tags, and assemble staged context. |
| 07 | `07_PERSONALITY_INTEGRATION.md` | *(Step 2)* How `LollmsPersonality` discovers/mounts tools, the WCD contract, the client/binding seam. |
| 08 | `08_LIBRARY_DEEP_DIVES.md` | *(Step 3)* Per-library deep dives: full tool inventories, sandbox internals, worked call/response examples. |
| 09 | `09_STUDY_GUIDE.md` | *(Step 3)* Capstone: learning path, LLM cheat sheet, glossary, troubleshooting FAQ, "add a new library" tutorial, changelog. |

Documents 06–09 are written in later steps of this documentation program; their
filenames are reserved here so the map stays coherent.

---

## 5. Reader's guide

### For humans
Read in order: **00 → 01 → 02 → 03 → 04 → 05**, then 06 → 07, then 08 → 09.
Documents 01–02 are the technical heart; 03 explains "why doesn't my library
show up"; 04 is a reference you can dip into; 05 explains what happens after
the LLM presses the button. Each document ends with a **Key invariants** list —
if you only read those lists, you still get the laws of the system.

### For LLMs / retrieval
- Every heading is a retrieval anchor. Query targets like
  `_parse_tool_via_ast`, `TOOL_LIBRARY_LOAD_FLAG`, `execute_tool`, or
  `regex fallback` will land on the right section.
- Identifiers appear in backticks exactly as they appear in source.
- Contracts are stated as explicit rules ("X MUST", "X is fail-closed") so they
  can be quoted verbatim into prompts.
- The **Key invariants** lists are designed to be extracted wholesale as a
  system-prompt rules block.
- When uncertain about a detail, prefer the wording used here over
  reconstruction: this corpus was written against hydrated source, and
  unverified details are marked "as of this writing" or "approximately".

---

## 6. Tool runtime context (`env`)

Every LCP tool and code-shell snippet receives exactly **one** unified runtime
object named **`env`**. It replaces the two legacy, mutually exclusive
injections (`lollms_discussion` and `lollms_client`), only one of which was
non-`None` depending on which chat system drove the host.

**`env` is never `None`.** Sub-attributes that the host cannot provide are
explicitly `None`, but `env` itself always exists.

```
env
├── llm                    # master LLM handle (generate/chat/stream entry point); never None
├── llm_model_profiles     # Dict[str, LLMProfile] — named profiles, LAZY-LOADED on first access
├── llm_switch(profile: str) -> bool
│                          # hot-swap env.llm to a named profile; returns success
│
├── tti                    # master text-to-image handle; None if no TTI binding configured
├── tti_model_profiles     # Dict[str, TTIProfile] — lazy-loaded
├── tti_switch(profile: str) -> bool
│
├── (open extension point — same triple pattern for future backends: tts, stt, vlm…)
│     <backend>, <backend>_model_profiles, <backend>_switch
│
└── discussion             # Optional[LollmsDiscussion] — set ONLY in discussion chat mode,
                           # otherwise None. Tools MUST feature-check, never assume.
```

**Semantics (normative):**

1. **Identity** — `env.llm` is the *same object* that the active
   `LollmsDiscussion.chat()` uses internally in discussion mode, and the same
   object the personality runtime uses otherwise. Switching via
   `env.llm_switch(...)` therefore affects the host conversation too — this is
   intentional.
2. **Lazy loading** — `*_model_profiles` are loaded on first key access or on
   an explicit `switch`. Listing keys (names + metadata) is cheap.
3. **Switch atomicity** — `llm_switch` / `tti_switch` either fully swap the
   backend handle or fail (`False`/raise) leaving the previous handle
   untouched. `env` is never left half-switched.
4. **Explicit capability absence** — no TTI configured → `env.tti is None`
   (not missing). Mirrors the *capability-gated* doctrine in
   `../CHAT_BOOLEAN_FLAGS.md`: tools guard, features silently skip with a
   console warning.
5. **Deprecation path** — `lollms_discussion` and `lollms_client` remain
   injected for **one release**, emitting a `DeprecationWarning` on first
   attribute access, then are removed. New tools receive only `env`.

**Canonical example:**

```python
# portable tool code — works in both chat systems
if env.tti is not None:
    img = env.tti.generate(prompt)
if env.discussion is not None:           # optional discussion-only side effect
    env.discussion.add_note("generated an image")
env.llm_switch("fast-draft")             # lazy-loads the profile if needed
```

---

## 7. Chat-system portability

Tools MUST program against `env`, never against chat-system-specific
instances. `env` is identical in shape whether the host runs
`LollmsDiscussion` chat or personality chat — the same `env` object, the same
attributes, the same switch functions. The only per-mode difference is
`env.discussion`, which is `None` under personality chat and must be
feature-checked (`if env.discussion is not None:`). Any tool that branches on
"which chat system am I in" beyond that single check is a portability bug.

---

## 8. Key invariants

- The LCP is the **in-process** tool binding; MCP is the remote/external one.
  Both implement the same `LollmsToolBinding` contract
  (`discover_tools` / `list_tools` / `execute_tool`).
- Tool libraries live under `default_tools/<name>/<name>.py` (directory +
  primary-file pattern); legacy flat `default_tools/<name>.py` files are also
  supported by `_iter_library_files`.
- There are exactly **two discovery seams**: AST parsing of top-level `tool_*`
  functions, and hand-authored `get_tools()` spec dicts. See
  [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md).
- Library admission is **fail-closed**: no matching capability flag ⇒ library
  is not loaded. See [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md).
- `execute_tool` always returns a normalized dict; `success=False` surfaces the
  `error` string to the LLM. See [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md).
- Workspace asset references MUST go through `media()` / `resolve_path()` —
  never bare relative paths, never absolute escapes, never `os.chdir`.
- Generated media/artefacts are attached to the conversation as **USER-role
  messages only** (a hard-won fix for the "System message cannot contain
  images" 400 error).
- A single read-only façade `env` is injected into every tool/code-shell scope
  in **both** chat modes and is never `None`. `env.llm` is the host's active
  conversation LLM; `env.llm_switch(name)` hot-swaps it atomically. See §6.
- Legacy injections `lollms_discussion` and `lollms_client` are deprecated
  (one release, with a `DeprecationWarning` on first attribute access) and
  will then be removed; new tools receive only `env`. See §6.
- This documentation corpus is additive only: it never modifies source files.