# Lollms Personalities & Handbags

This document is the canonical reference for the two-layer model that powers every LoLLMS agent: the **`LollmsPersonality`** (the runtime brain) and the **`Handbag`** (the self-contained resource folder). It explains what each layer is, how they interlock, and — most importantly — how to **build a Handbag** from scratch and wire it into a personality.

---

## 1. Overview

LoLLMS agents are built from two cooperating layers:

```
┌──────────────────────────────────────────────┐
│  LollmsPersonality  (runtime brain)          │
│  • name, system_prompt, tools                │
│  • tool_specs(), query_data(), has_data      │
│  • SkillsManager, DocNavigator, FailureMemory│
└──────────────────────┬───────────────────────┘
                       │ consumes
                       ▼
┌──────────────────────────────────────────────┐
│  Handbag  (resource bundle — a folder)       │
│  • SOUL.md, handbag.yaml                     │
│  • tools/, skills/, rag/, docs/, assets/     │
│  • memory/, workspace/, coworkers/           │
└──────────────────────────────────────────────┘
```

- The **personality** is the object the chat loop talks to. It exposes a stable contract (`name`, `system_prompt`, `tools`, `tool_specs()`, `query_data()`, `has_data`, `__bool__`).
- The **Handbag** is a plain directory on disk. It carries every resource the personality needs. The `Handbag` class auto-discovers those resources so the personality can be assembled without boilerplate.

The two layers are decoupled: you can use a `Handbag` standalone (to inspect or manipulate resources), or you can hand it to `LollmsPersonality.from_handbag()` to produce a fully-armed agent.

---

## 2. The LollmsPersonality contract

Every `LollmsPersonality` — whether created in code or loaded from a Handbag — guarantees the following contract. The chat loop relies on it and never needs defensive `if personality:` guards.

| Member | Type | Guarantee |
| :--- | :--- | :--- |
| `name` | `str` | Never empty. |
| `system_prompt` | `str` | Never `None`. |
| `tools` | `_NullToolBinding` \| `LollmsToolBinding` | Never `None`. |
| `tool_specs()` | `Dict[str, Dict[str, Any]]` | Never raises. |
| `query_data(q)` | `Dict[str, Any]` | Normalized RAG dict; never raises. |
| `has_data` | `bool` | `True` iff any RAG source is attached. |
| `bool(personality)` | `bool` | `False` only for `NullPersonality`. |

### Supporting components

- **`SkillsManager`** — manages `SKILL.md` libraries with tiered visibility (`visible`, `loadable`, `searchable`). Injects skill context into the system prompt and mounts skill tools (`tool_load_skill`, `tool_search_skills`, etc.).
- **`DocNavigator`** — index-based navigation over the Handbag's `docs/` directory. Exposes `build_doc_tools()` and `build_docs_scope_block()` so the LLM can query documents without loading them wholesale.
- **`FailureMemory`** — signature-based loop interceptor that prevents the agent from retrying the same failing tool call indefinitely.
- **RAG** — one or more `RAGDataSource` objects, each with a `name`, `description`, `query_fn`, optional `store`, and `auto_query` flag. Pre-hydrated on turn start when `auto_query=True`, otherwise queried on demand via `tool_query_rag`.

---

## 3. The Handbag — anatomy

A `Handbag` is instantiated from a path:

```python
from lollms_client.lollms_personality.handbag import Handbag

hb = Handbag("./my_agent_handbag")
```

### Constructor validation

- The path is resolved to an absolute `Path`.
- If the path is **not an existing directory**, a `ValueError` is raised.
- All directory attributes are set (whether or not the subdirectory exists).
- The manifest, tool files, skills directories, RAG files, extra knowledge bases, memory DB path, and assets are then **auto-discovered**.

### Full attribute reference

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `path` | `Path` | The resolved Handbag root. |
| `soul_path` | `Path` | Path to `SOUL.md`. |
| `coworkers_dir` | `Path` | Path to `coworkers/`. |
| `tools_dir` | `Path` | Path to `tools/`. |
| `_skills_dir` | `Path` | Canonical skills directory (backing field). |
| `skills_dir` | `Path` (property) | Public accessor for `_skills_dir`. |
| `assets_dir` | `Path` | Path to `assets/`. |
| `memory_dir` | `Path` | Path to `memory/`. |
| `rag_dir` | `Path` | Path to `rag/`. |
| `docs_dir` | `Path` | Path to `docs/`. |
| `workspace_dir` | `Path` | Path to `workspace/`. |
| `manifest` | `Dict[str, Any]` | Parsed `handbag.yaml` (safe load; `{}` on error). |
| `tool_files` | `List[Path]` | Discovered toolset **folders** (canonical `TOOLS.py` standard). Flat single-file libraries are returned as `.py` files for backward compatibility. Both shapes are handled by `lcp.tool_layout.resolve_library`. |
| `skills_dirs` | `List[Path]` | Registered skill libraries. |
| `rag_files` | `List[Path]` | Discovered RAG source files. |
| `extra_knowledge_bases` | `List[Path]` | Plain-path knowledge bases from `handbag.yaml`. |
| `extra_safe_stores` | `List[Any]` | Opened safe-store datastores (`.db` entries). |
| `memory_db_path` | `Optional[str]` | `sqlite:///<memory/memory.db>` if `memory/` exists, else `None`. |
| `global_memory_db_path` | `Optional[str]` (property) | Alias for `memory_db_path`. |
| `assets` | `Dict[str, str]` | Loaded asset name → content map. |

### Public methods

| Method | Description |
| :--- | :--- |
| `register_skills_dir(skills_dir) -> Path` | Resolves the directory, creates it if missing, and **idempotently** appends it to `skills_dirs`. Returns the resolved path. |
| `normalize_extra_knowledge_bases(declared)` | Normalizes the `extra_knowledge_bases` list from the manifest (used internally by `_load_extra_knowledge_bases`). |

---

## 4. Handbag directory layout (authoritative reference)

```
my_agent_handbag/
├── handbag.yaml
├── SOUL.md
├── coworkers/
├── tools/
│   └── my_toolset/           # folder-per-toolset (canonical)
│       ├── TOOLS.py          # entry point
│       ├── helper.py         # optional
│       ├── README.md         # optional
│       ├── LICENSE.md        # optional
│       └── assets/icon.png   # optional
├── skills/
│   └── SKILL.md
├── assets/
├── memory/            # optional → memory/memory.db
├── rag/
├── docs/
└── workspace/
```

### Per-directory discovery rules

| Directory | Discovery rule |
| :--- | :--- |
| `tools/` | Returns the **toolset folder** for each library (canonical `TOOLS.py` standard). The entry point (`TOOLS.py` → `<folder>.py` → first `.py`, resolved by `lcp.tool_layout.resolve_library_entry`) is picked at load time by `LCPBinding`. Flat single-file `.py` libraries sitting directly in `tools/` are returned as files (legacy compat). Sub-folders named `assets/`, `references/`, `__pycache__/` and hidden folders are skipped. |
| `rag/` | Recursive `rglob("*")`, **files only**, skipping names that start with `.`. |
| `skills/` | The canonical skills directory. Additional libraries can be appended via `register_skills_dir()`. |
| `assets/` | Loaded into `assets: Dict[str, str]`. |
| `memory/` | If present, `memory_db_path = sqlite:///<memory/memory.db>`. If absent, `memory_db_path is None`. |
| `docs/` | Documents for the `DocNavigator`. |
| `workspace/` | The agent's writable sandbox root. |
| `coworkers/` | Co-worker definitions (sub-agents / crewmates). |
| `handbag.yaml:extra_knowledge_bases` | Entries ending in `.db` (case-insensitive) are opened as safe stores (`extra_safe_stores`); everything else is a plain KB path (`extra_knowledge_bases`). |

---

## 5. Building a Handbag programmatically (FULL runnable example)

The following script creates a complete Handbag from scratch, then constructs a `Handbag` object and prints everything it discovered. Copy-paste it and run it — it has no external dependencies beyond `lollms_client`.

```python
from pathlib import Path
from lollms_client.lollms_personality.handbag import Handbag

# ── 1. Create the directory tree ─────────────────────────────────────────────
root = Path("./my_agent_handbag")
for sub in ("tools", "rag", "assets", "skills", "docs", "workspace", "memory"):
    (root / sub).mkdir(parents=True, exist_ok=True)

# ── 2. Write the soul (the agent's identity) ─────────────────────────────────
(root / "SOUL.md").write_text(
    "# Research Assistant\n\n"
    "You are a meticulous research assistant. Cite sources, verify claims, "
    "and never fabricate references.\n",
    encoding="utf-8",
)

# ── 3. Write the manifest (declares extra knowledge bases) ───────────────────
(root / "handbag.yaml").write_text(
    "extra_knowledge_bases:\n"
    "  - ./shared/company_handbook.md\n"
    "  - ./shared/archive.db\n",
    encoding="utf-8",
)

# ── 4. Drop a toolset (canonical TOOLS.py layout) ────────────────────────────
toolset_dir = root / "tools" / "my_toolset"
toolset_dir.mkdir(parents=True, exist_ok=True)
(toolset_dir / "TOOLS.py").write_text(
    "def tool_echo(text: str) -> dict:\n"
    "    '''Echoes the input text back.'''\n"
    "    return {'success': True, 'output': text}\n",
    encoding="utf-8",
)
# Optional siblings (README.md, LICENSE.md, assets/icon.png, helper .py files)
(toolset_dir / "assets").mkdir(exist_ok=True)
(toolset_dir / "README.md").write_text("# my_toolset\n", encoding="utf-8")

# ── 5. Drop a RAG source ─────────────────────────────────────────────────────
(root / "rag" / "notes.md").write_text(
    "# Notes\n\nKey fact: the sky is blue.\n", encoding="utf-8"
)

# ── 6. Drop an asset ─────────────────────────────────────────────────────────
(root / "assets" / "greeting.txt").write_text(
    "Hello from the Handbag!", encoding="utf-8"
)

# ── 7. Drop a skill ──────────────────────────────────────────────────────────
(root / "skills" / "SKILL.md").write_text(
    "---\ntitle: Citation Style\ndescription: How to cite sources.\n---\n\n"
    "Always cite with author, year, and title.\n",
    encoding="utf-8",
)

# ── 8. Construct the Handbag and inspect discovery ───────────────────────────
hb = Handbag(root)

print("path:            ", hb.path)
print("soul_path:       ", hb.soul_path)
print("tool_files:      ", [p.name for p in hb.tool_files])   # -> ['my_toolset'] (folder)
print("rag_files:       ", [p.name for p in hb.rag_files])
print("skills_dirs:     ", [p.name for p in hb.skills_dirs])
print("manifest:        ", hb.manifest)
print("extra_kbs:       ", [str(p) for p in hb.extra_knowledge_bases])
print("extra_safe_stores:", hb.extra_safe_stores)
print("memory_db_path:  ", hb.memory_db_path)
print("assets:          ", list(hb.assets.keys()))
```

Expected output (abridged):

```
path:             /abs/path/my_agent_handbag
soul_path:        /abs/path/my_agent_handbag/SOUL.md
tool_files:       ['my_tool.py'] # the folder, not a file
rag_files:        ['notes.md']
skills_dirs:      ['skills']
manifest:         {'extra_knowledge_bases': ['./shared/company_handbook.md', './shared/archive.db']}
extra_kbs:        ['/abs/path/shared/company_handbook.md']
extra_safe_stores: [<opened safe store for archive.db>]
memory_db_path:   sqlite:////abs/path/my_agent_handbag/memory/memory.db
assets:           ['greeting.txt']
```

---

## 6. Wiring the Handbag into a personality

Once the Handbag exists, `LollmsPersonality.from_handbag()` consumes it and wires every discovered resource into the runtime:

```python
from lollms_client.lollms_personality import LollmsPersonality

personality = LollmsPersonality.from_handbag("./my_agent_handbag")
```

The wiring is:

| Handbag resource | Personality component |
| :--- | :--- |
| `hb.tool_files` | The personality's tool binding. Each entry is a **toolset folder** (canonical `TOOLS.py` standard). The binding resolves `TOOLS.py` → `<folder>.py` → first `.py` via `lcp.tool_layout.resolve_library`, so renaming to `TOOLS.py` is transparent. |
| `hb.rag_files` + `hb.extra_safe_stores` | The RAG source(s). |
| `hb.skills_dirs` | The `SkillsManager`. |
| `hb.docs_dir` | The `DocNavigator`. |
| `hb.soul_path` | The system prompt seed. |
| `hb.memory_db_path` | The `FailureMemory` / `LollmsMemoryManager`. |

### App-shipped skills

If your application ships its own curated skill library, register it on the Handbag before (or after) loading the personality:

```python
personality = LollmsPersonality.from_handbag("./my_agent_handbag")
personality.handbag.register_skills_dir("./app_assets/skills")
```

`register_skills_dir()` is idempotent — calling it twice with the same path is a no-op. It creates the directory if missing and appends it to `skills_dirs`.

### Skill Provenance Inventory (`builtin_skills` / `extra_skills`)

After `from_handbag()` returns, the app can recover **which skill belongs to which source** without re-parsing directories. Three read-only handles are exposed:

| Member | Type | Meaning |
| :--- | :--- | :--- |
| `Handbag.builtin_skills` | `Dict[str, str]` | `{title: absolute SKILL.md path}` for every skill whose `SKILL.md` lives under one of `handbag.skills_dirs` (the handbag's own `skills/` folder, plus any directory added via `register_skills_dir()`). |
| `Handbag.get_skills_by_source(builtin_dirs, extra_dirs)` (via `SkillsManager`) | `Dict[str, Dict[str, str]]` | Two-bucket partition of the *entire* registry. |
| `LollmsPersonality.get_skills_by_source()` | `Dict[str, Dict[str, str]]` | Same two-bucket partition, but resolved against the personality's handbag + `extra_skills_dirs`. |

**The two-bucket contract** (identical for `Handbag` and `LollmsPersonality`):

```python
{
    "builtin": {title: "/abs/path/to/SKILL.md" | "internal"},
    "extra":   {title: "/abs/path/to/SKILL.md" | "internal"},
}
```

- **`"builtin"`** — the skill's `SKILL.md` lives under a handbag-native skills directory.
- **`"extra"`** — everything else: the app-supplied `skills_dir=` given to `from_handbag`, any `extra_skills_dirs=` entry, and any custom/global skill that falls outside both roots.
- **`"internal"`** — the sentinel string used as the *path value* when a skill has **no physical backing file** (e.g. a skill created purely in memory without a handbag, or a skill whose `file_path` is `None` / unresolvable). The skill still appears in its bucket — only the path is replaced by the `"internal"` marker.

Both keys are **always present** so callers can iterate unconditionally:

```python
from lollms_client.lollms_personality import LollmsPersonality

personality = LollmsPersonality.from_handbag(
    "./my_agent_handbag",
    skills_dir="./app_extras/skills",   # app-owned output / override folder
)

inventory = personality.get_skills_by_source()

for title, path in inventory["builtin"].items():
    print(f"[handbag]    {title:32s} -> {path}")

for title, path in inventory["extra"].items():
    print(f"[app-extra]  {title:32s} -> {path}")

# sentinel example — in-memory skills carry no file
assert inventory["extra"].get("ephemeral_note") in (None, "internal")
```

#### Propagated `skills` and `tools` folders from the host application

`from_handbag()` accepts two optional privileged output folders so a host app can force every agent-generated artefact into a folder it controls:

```python
personality = LollmsPersonality.from_handbag(
    "./my_agent_handbag",
    skills_dir="./app_skills_runtime",   # where the agent CREATES / UPDATES skills
    tools_dir="./app_tools_runtime",     # where the agent CREATES / UPDATES tools
)
```

**Resolution order for both folders** (identical for `skills_dir` and `tools_dir`):

1. **Explicit caller argument** (`skills_dir=` / `tools_dir=`) — highest priority.
2. **`handbag.yaml`** manifest keys `skills_output_path` / `tools_output_path`.
3. **The handbag's own `skills/` / `tools/` folder** — legacy fallback when no override is given.

The resolved absolute paths are exposed on both objects so tool-generation code paths (e.g. `tool_maker`) can target them without re-parsing the manifest:

```python
personality.generated_skills_dir   # Path — absolute, created on demand
personality.generated_tools_dir    # Path — absolute, created on demand

personality.handbag.generated_skills_dir  # same, handbag-scoped
personality.handbag.generated_tools_dir
```

When `from_handbag()` is called **without** `skills_dir=`/`tools_dir=` and the manifest declares neither, these properties resolve to `handbag/skills/` and `handbag/tools/` — i.e. the historical behaviour is preserved verbatim.

#### Skill read/write contract (`modifiable:`)

A `SKILL.md` may declare in its YAML frontmatter whether the file is safe to overwrite in place:

| Value | Behaviour |
| :--- | :--- |
| `modifiable: true` | Read/write — `tool_update_skill` / `tool_append_to_skill` / `tool_remove_skill` may rewrite the **actual** backing file in place. |
| `modifiable: false` | Read-only — every mutation tool refuses. |
| *(key absent)* | Treated as **read-only** (safe default), and the loader performs a one-shot **metadata normalisation** on load: it injects a literal `modifiable: false` line into the frontmatter without ever touching the skill body. On load failure of the write, the in-memory state remains read-only. |

Newly **generated** skills are emitted with `modifiable: true` by `tool_create_skill` so they remain overwritable across round-trips. Skill `update_skill` rewrites the exact `skill.file_path` — whether it lives in the handbag, an `extra_skills_dirs` entry, or the app-supplied `skills_dir` — never a copy in `skills_output_path`.

### Using the personality in a discussion

```python
from lollms_client import LollmsClient, LollmsDiscussion

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llama3"})
discussion = LollmsDiscussion.create_new(lollms_client=client)

response = discussion.chat(
    user_message="Summarize my notes and cite the source.",
    personality=personality,
)
print(response["ai_message"].content)
```

For a fully working end-to-end example, see [`examples/handbag_agent_example.py`](../examples/handbag_agent_example.py).

---

## 7. Declaring extra knowledge bases (`handbag.yaml`)

The `handbag.yaml` manifest can declare additional knowledge bases that live **outside** the Handbag folder. The `Handbag` class splits them into two categories:

```yaml
extra_knowledge_bases:
  - ./shared/company_handbook.md   # plain path → extra_knowledge_bases
  - ./shared/archive.db            # .db file → opened as a safe store
  - /abs/path/to/legal_docs/       # plain path → extra_knowledge_bases
```

### Splitting rules

- An entry whose suffix is `.db` (case-insensitive) is treated as a **safe-store datastore**. It is opened via `_open_safe_store()` and appended to `extra_safe_stores`.
- Every other entry is treated as a **plain knowledge-base path** and appended to `extra_knowledge_bases`.

Both lists are then consumed by the personality's RAG layer: plain paths become file-based RAG sources, and safe stores become queryable datastores.

---

## 8. Best-practices / gotchas

- **Tools must not be named `__init__.py`.** The discovery rule explicitly excludes them. A tool file named `__init__.py` will be silently ignored.
- **Hidden RAG files are skipped.** Any file in `rag/` whose name starts with `.` is excluded from `rag_files`. This is intentional — use it to keep scratch files out of the RAG index.
- **The memory DB must live inside `memory/`.** `memory_db_path` is only set when the `memory/` directory exists. If you want a persistent memory DB, create `memory/` and let the Handbag wire `sqlite:///<memory/memory.db>` automatically.
- **`workspace/` is the sandbox CWD.** When the personality runs Python execution tools, the current working directory is set to the Handbag's `workspace/` directory. Tools should resolve files with relative paths.
- **`register_skills_dir()` is idempotent.** Calling it repeatedly with the same path is safe. It creates the directory if missing.
- **The manifest is loaded safely.** If `handbag.yaml` is malformed, `manifest` is `{}` — the Handbag still loads, but `extra_knowledge_bases` will be empty.
- **`tool_files` returns folders, not files.** Each entry corresponds to a toolset **folder** under `tools/` (canonical `TOOLS.py` standard), except flat single-file `.py` libraries which are returned as files for backward compatibility. The entry point inside the folder is chosen by `lcp.tool_layout.resolve_library_entry` — `TOOLS.py` first, then `<folder>.py`, then any non-`__init__` `.py`.
- **Prefer `TOOLS.py`.** New toolsets should ship `TOOLS.py` as the entry point. Legacy `<folder>.py` and flat `.py` layouts are still supported, so migration is optional and can be done incrementally.
- **Extra tools handed to `from_handbag` are folders too.** Pass the toolset **folder** path (not the individual `.py` file), e.g. `extra_tools=[Path("./my_toolset")]`. Passing a single-file `.py` still works.
- **`rag_files` recurses fully.** `rglob("*")` walks the entire `rag/` tree.

---

## 9. See also

### Working examples

- [`examples/handbag_agent_example.py`](../examples/handbag_agent_example.py) — minimal agent from a Handbag.
- [`examples/handbag_rag_skills_example.py`](../examples/handbag_rag_skills_example.py) — RAG + skills from a Handbag.
- [`examples/handbag_docs_example.py`](../examples/handbag_docs_example.py) — documents + `DocNavigator`.
- [`examples/agentic_personality_tools_example.py`](../examples/agentic_personality_tools_example.py) — agentic tool use.

### LCP deep-dives

- [`docs/lcp/07_PERSONALITY_INTEGRATION.md`](lcp/07_PERSONALITY_INTEGRATION.md) — how the personality integrates with the LCP tool binding.
- [`docs/lcp/10_TOOLSET_AUTHORING_GUIDE.md`](lcp/10_TOOLSET_AUTHORING_GUIDE.md) — how to author toolsets that a Handbag can carry.

### Package references

- [`src/lollms_client/lollms_personality/README.md`](../src/lollms_client/lollms_personality/README.md) — the `LollmsPersonality` API reference.
- [`src/lollms_client/lollms_discussion/README.md`](../src/lollms_client/lollms_discussion/README.md) — how the chat loop consumes a personality.