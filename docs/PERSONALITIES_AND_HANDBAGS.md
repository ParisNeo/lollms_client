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
| `tool_files` | `List[Path]` | Discovered `*.py` tool files. |
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
│   └── my_tools/
│       └── search.py
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
| `tools/` | Only `*.py` files whose stem is not `__init__`. Recurses **one level** into immediate subdirectories (`item.glob("*.py")`). |
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

# ── 4. Drop a tool ───────────────────────────────────────────────────────────
(root / "tools" / "my_tool.py").write_text(
    "def tool_echo(text: str) -> dict:\n"
    "    '''Echoes the input text back.'''\n"
    "    return {'success': True, 'output': text}\n",
    encoding="utf-8",
)

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
print("tool_files:      ", [p.name for p in hb.tool_files])
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
tool_files:       ['my_tool.py']
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
| `hb.tool_files` | The personality's tool binding (each `*.py` file is parsed for `tool_*` functions). |
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
- **`tool_files` recurses one level.** Only immediate subdirectories of `tools/` are scanned (`item.glob("*.py")`). Deeper nesting is not discovered.
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