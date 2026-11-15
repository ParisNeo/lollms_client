# Agent Introspection — Listing Tools, Skills, and RAG Databases

This document explains how to programmatically inspect a `LollmsPersonality`
(agent) instance to discover **what it can do** — its tools, its skills, and
its RAG knowledge databases — both for **internal** (handbag-provided) and
**extra** (caller-supplied) sources.

All APIs below are methods/properties on `LollmsPersonality` (imported as
`from lollms_client.lollms_personality import LollmsPersonality`).

---

## 1. RAG Databases

### The recommended API: `get_rag_databases_info()`

Returns a **structured, serialisable report** of *every* RAG database wired to
the personality — internal (handbag-declared / auto-promoted) *and* extra
(caller-supplied) — with stable names that match exactly what the LLM is told
in the system prompt and `tool_query_rag` description.

```python
info = agent.get_rag_databases_info()
# {
#   "has_any_data": True,
#   "total": 2,
#   "databases": [
#     {"name": "ltspice", "description": "...", "kind": "safe_store",
#      "db_path": "/…/rag/ltspice/Ltspice.db"},
#     {"name": "project_rules", "description": "...", "kind": "kb_path",
#      "db_path": "/…/rag/project_rules.md"},
#   ]
# }
```

Each database entry contains:

| Field         | Meaning                                                         |
|---------------|-----------------------------------------------------------------|
| `name`        | Exact `datasource_name` to pass to `tool_query_rag(...)`       |
| `description` | Human-readable description surfaced to the LLM                 |
| `kind`        | `"data_source"`, `"safe_store"`, `"kb_path"`, or `"client_rag"` |
| `db_path`     | Resolved physical `.db` path when applicable, else `None`      |

### Quick probe: `has_rag_databases`

```python
if agent.has_rag_databases:
    print("This agent has at least one usable RAG database.")
```

Unlike `has_data` (which also counts `data_files`, `query_rag_callback`, and a
client-level RAG binding), `has_rag_databases` reflects **only queryable
databases** exposed via `get_rag_databases_info()`.

### Underlying catalogue: `_build_rag_catalog()`

Internal method used by both the system-prompt block and the tool description.
You normally don't call it directly; `get_rag_databases_info()` is the public
wrapper and is the **single source of truth** for what the agent sees.

### What ends up in the catalogue?

The catalogue merges four layers (deduplicated by resolved `.db` path):

1. **Declarative `data_sources`** — passed via `data_source=` / `data_sources=`.
2. **Handbag `extra_safe_stores`** — `.db` files auto-discovered under
   `<handbag>/rag/` or declared in `handbag.yaml` (see
   `docs/PERSONALITIES_AND_HANDBAGS.md`).
3. **Plain `extra_knowledge_bases`** — text files/dirs and, as a safety net,
   any leftover `.db`/`.sqlite`/`.sqlite3` path.
4. **Client-level RAG binding** — `agent.lollms_client.rag` when present.

> ⚠️ **Pitfall**: If an agent says *"I don't have a RAG database"*, first check
> `[Handbag-SCAN]` in the logs at agent-load time. It prints the exact `rag/`
> folder scanned. If `rag/` is empty but the `.db` on screen lives in a
> **different** agent's handbag, you are chatting with the wrong agent.

---

## 2. Tools

### `list_tools_structured()` (a.k.a. `list_tools()` / `get_tools_structured()`)

Returns every tool available to the personality, annotated with provenance:

```python
tools = agent.list_tools_structured()
for t in tools:
    print(f"[{t['source']}/{t['category']}] {t['name']}")
```

Each entry contains:

| Field         | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `name`        | Tool name (as the LLM sees it)                                          |
| `description` | Tool description                                                         |
| `parameters`  | Parameter schema list                                                    |
| `source`      | `"handbag"`, `"workspace_tools"`, `"multimodal_binding"`, `"rag"`, …    |
| `category`    | Logical category (e.g. `"handbag_tools"`, `"execution"`, `"knowledge_retrieval"`) |
| `is_handbag`  | `True` when the tool originates from the agent's own handbag            |
| `source_file` | Absolute path to the source file, when known                            |

The list is sorted with handbag tools first, then by category and name.

### `list_builtin_tools(category=None)`

Enumerates the **packaged LCP native tools** without mutating client state.
Optionally filter by a substring of the library name or tool name:

```python
native = agent.list_builtin_tools(category="workspace")
```

Returns `[]` on any import/discovery failure (never raises).

### Direct handbag tool binding

The handbag's own tools are attached as a `LCPBinding` and are also discoverable
directly:

```python
binding = agent.tools               # LCPBinding or _NullToolBinding
if hasattr(binding, "discovered_tools"):
    for spec in binding.discovered_tools:
        print(spec.get("name"))
```

### Refresh after hot-swapping files on disk

```python
agent.refresh_tools_and_skills()
```

Re-scans `<handbag>/tools/` and/or `<handbag>/skills/` plus any registered
`_extra_tool_files` / `_extra_skills_dirs`, rebuilding the tool binding and
reloading the skills manager.

---

## 3. Skills

### `list_skills_structured(include_content=False)` (a.k.a. `list_skills()` / `get_skills_structured()`)

Returns every skill registered with the personality, annotated with provenance:

```python
skills = agent.list_skills_structured()
for s in skills:
    print(f"[{s['source']}] {s['title']}  (handbag={s['is_handbag']})")
```

Each entry contains:

| Field             | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `title`           | Skill title (the exact name `tool_load_skill` expects)         |
| `description`     | Short description                                              |
| `category`        | Logical category                                               |
| `tags`            | Skill tags                                                     |
| `visibility`      | `"always_visible"`, `"loadable"`, or `"mixed"`                 |
| `modifiable`      | Whether the skill can be edited                                |
| `source`          | `"handbag"`, `"workspace"`, `"bundled"`, `"global"`, `"custom"` |
| `is_handbag`      | `True` when the skill originates from the agent's own handbag |
| `handbag_name`    | Handbag folder name when `is_handbag`                          |
| `file_path`       | Absolute path to the `SKILL.md` (or `None` for in-memory)      |
| `content_preview` | First 200 chars of the skill body                              |
| `content`         | Full body — only present when `include_content=True`           |

### `get_skills_by_source()`

Partitions the inventory into `{"builtin": {...}, "extra": {...}}`:

- `"builtin"` — skills shipped natively with the handbag (under
  `<handbag>/skills/` or or a dir registered via
  `Handbag.register_skills_dir`).
- `"extra"` — caller-supplied via `skills_dir=` / `extra_skills_dirs=` at
  `from_handbag(…)` call time, or globally registered skills.

```python
inv = agent.get_skills_by_source()
print("Builtin:", list(inv["builtin"].keys()))
print("Extra:  ", list(inv["extra"].keys()))
```

The values are `SKILL.md` file paths, or the sentinel `"internal"` for
pure in-memory skills.

### Quick accessors: `builtin_skills` / `extra_skills`

```python
print(agent.builtin_skills)   # {title: path} — handbag-native only
print(agent.extra_skills)     # {title: path} — caller-supplied only
```

Both return `{}` when the personality has no handbag / no extras respectively.

---

## 4. Complete inventory example

```python
from lollms_client.lollms_personality import LollmsPersonality

agent = LollmsPersonality.from_handbag(
    "/path/to/my_agent",
    lollms_client=client,
    extra_knowledge_bases=["/data/extra_notes.md"],
)

print(f"\n=== Agent: {agent.name} ===")

# RAG
rag = agent.get_rag_databases_info()
print(f"\nRAG databases: {rag['total']} (has_any_data={rag['has_any_data']})")
for db in rag["databases"]:
    print(f"  [{db['kind']}] {db['name']} → {db['db_path'] or '(client binding)'}")

# Tools
tools = agent.list_tools_structured()
handbag_tools = [t for t in tools if t["is_handbag"]]
extra_tools = [t for t in tools if not t["is_handbag"]]
print(f"\nTools: {len(tools)} total ({len(handbag_tools)} handbag, {len(extra_tools)} extra)")
for t in tools:
    origin = "handbag" if t["is_handbag"] else t["source"]
    print(f"  [{origin}] {t['name']}")

# Skills
inv = agent.get_skills_by_source()
print(f"\nSkills: {len(inv['builtin'])} builtin, {len(inv['extra'])} extra")
for title, path in {**inv["builtin"], **inv["extra"]}.items():
    suffix = " (builtin)" if title in inv["builtin"] else " (extra)"
    print(f"  {title}{suffix}")
```

---

## 5. Reference — quick cheat sheet

| Need                                   | API                                                        |
|----------------------------------------|------------------------------------------------------------|
| All RAG databases (structured)         | `agent.get_rag_databases_info()`                           |
| Boolean: has any RAG?                  | `agent.has_rag_databases`                                  |
| All tools (structured + provenance)    | `agent.list_tools_structured()`                            |
| Packaged LCP native tools only         | `agent.list_builtin_tools(category=None)`                  |
| Handbag binding's raw discovered tools | `agent.tools.discovered_tools` (when not null binding)     |
| All skills (structured + provenance)   | `agent.list_skills_structured(include_content=False)`      |
| Skills partitioned builtin/extra       | `agent.get_skills_by_source()`                             |
| Builtin skill map `{title: path}`      | `agent.builtin_skills`                                     |
| Extra skill map `{title: path}`        | `agent.extra_skills`                                       |
| Refresh after on-disk changes          | `agent.refresh_tools_and_skills()`                         |

All listing APIs are safe to call on a freshly constructed agent and on
`NullPersonality` instances (`NullPersonality` returns empty collections).