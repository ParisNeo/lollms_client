# Building a Tool-Editor Agent (Handbag + `extra_tools`)

A guide to constructing a **dedicated editor persona** whose single job is to inspect,
modify, and validate an existing LCP toolset — while only ever standing on the verified
mechanics of `LollmsPersonality`, `Handbag`, and `LCPBinding`.

> **Scope**: This document describes *how to wire* the editor agent and *how the hot-reload
> actually works*. It intentionally forbids claiming behaviors the codebase does not
> implement (notably: `tool_load_generated_toolset` does **not** hot-load).

---

## 1. Overview — why a dedicated editor persona + `extra_tools`

Agents that both *use* and *edit* a tool tend to conflate the two phases: it calls the tool,
gets a wrong answer, and rewrites it in the same breath. A structured editor loop needs a
cleaner split:

1. **A persona whose SOUL is scoped to editing tools only** — it inspects `TOOLS.py`,
   reasons about behavior, and patches the source.
2. **A single hook that attaches that target toolset to the persona as an *executable*
   tool**, so the editor can observe current behavior before/after each edit.

That hook is `LollmsPersonality.from_handbag(..., extra_tools=[<toolset_folder>])`.

`extra_tools` is not a "visibility-only" list. Entries are resolved, existence-checked,
deduplicated against the handbag's native tools, merged, and passed to a **single**
`LCPBinding(tool_files=[...])`. The target toolset's `tool_*` functions are therefore
callable by the editor agent exactly like the handbag's own tools.

Because there is **one binding** (not per-source bindings), tools from `extra_tools` and
from `hb.tool_files` coexist in the same registry. Name collisions resolve to whichever
loaded last — pick toolset names carefully.

---

## 2. Prerequisites

- A working `LollmsClient` (any LLM binding). Optional but expected for real editing.
- A **handbag folder** (see §4) containing at least `SOUL.md`.
- A **target LCP toolset folder** containing a `TOOLS.py` file that defines one or more
  top-level `tool_<verb>(...)` functions.

The canonical toolset entry-point filename is `TOOLS.py` (`TOOLS_ENTRYPOINT` in
`lollms_client.tools_bindings.lcp.tool_layout`). Fallbacks the loader also accepts:

- `<folder>.py` (match the folder name),
- the first `.py` in the folder that defines `tool_*` functions.

If none of these is found, `LCPBinding` logs an explicit error and
`len(binding.discovered_tools) == 0` — the toolset silently contributes nothing.

```python
from pathlib import Path
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen2.5-coder:7b"},
)

handbag_path = Path("editor_handbag")               # created below
target_toolset = Path("./workspace/file_tools")     # contains TOOLS.py
assert (target_toolset / "TOOLS.py").is_file(), "Target toolset must expose TOOLS.py"
```

---

## 3. Step 1 — Write the editor persona (`SOUL.md`)

`from_handbag` reads `SOUL.md` (YAML frontmatter `--- ... ---`, then a markdown body).
`PersonalityBundle.parse_soul_md` extracts the frontmatter into a dict and uses the body
as the system prompt.

Scope the body **strictly to tool editing**: forbid it from taking on unrelated tasks, and
spell out the edit loop it must follow.

```markdown
---
name: ToolSmith
author: lollms-client
version: '1.0'
category: engineering
description: A focused agent that inspects, edits, and validates a single LCP toolset.
skills_mode: loadable
---

You are ToolSmith, a dedicated LCP toolset editor.

Your ONLY job is to inspect, edit, and validate the target toolset that has been
attached to you via `extra_tools`. You must NOT perform unrelated tasks
(no web research, no document writing, no code generation outside the toolset).

## The mandatory edit loop

1. READ the toolset source. Use `tool_read_file(path="<toolset>/TOOLS.py")`
   or your workspace read tools.
2. OBSERVE current behavior by executing the relevant `tool_*` function with a
   representative input BEFORE editing.
3. REASON about the defect: state the exact failing behavior in one sentence.
4. EDIT `TOOLS.py` (SEARCH/REPLACE patch via your file tools, or a full rewrite).
5. `tool_*` must remain top-level and syntactically valid Python at all times.
6. VALIDATE by executing the same `tool_*` function again. Compare before/after.
7. If it still fails, iterate from step 3. If it succeeds, stop and summarize.

## Hard rules

- Never edit any file outside the attached toolset folder.
- Never claim an edit is "hot-loaded" — you must call `refresh_tools_and_skills` (or
  the host must, on your behalf) before new behavior is visible.
- If the toolset's `TOOLS.py` no longer parses, you have broken it: fix it immediately.
```

Save this at `editor_handbag/SOUL.md`.

---

## 4. Step 2 — Handbag layout

`from_handbag` calls `Handbag(handbag_path)` internally. The layout it consumes:

```text
editor_handbag/
├── SOUL.md                  # persona (required)
├── handbag.yaml             # optional manifest
├── tools/                   # native toolset folders (each with TOOLS.py)
├── skills/                  # SKILL.md subfolders
├── coworkers/<name>/SOUL.md # optional crew
├── rag/                     # files auto-exposed as RAGDataSource
├── memory/memory.db         # Independent-Life memory (auto-created)
└── workspace/               # default local sandbox
```

Only `SOUL.md` is strictly required. The other folders are additive.

A minimal `handbag.yaml`:

```yaml
name: Editor Handbag
version: '1.0'
description: A handbag dedicated to tool-editing personas.
skills_mode: loadable
```

`from_handbag` reads `name / author / category / description / skills_mode / temperature`
from the SOUL frontmatter. `name` defaults to the folder name (title-cased) when absent.

Workspace resolution precedence (from the implementation):

1. `workspace_path=` argument (explicit override),
2. `workspace=` argument (legacy alias),
3. `hb.workspace_dir` (i.e. `editor_handbag/workspace/`) if it exists,
4. `None` (no workspace — most file tooling will be disabled).

For a tool *editor*, you almost always want an explicit `workspace_path=` pointing to a
directory where the target toolset is reachable, because the agent's file tools are scoped
to the workspace root.

---

## 5. Step 3 — Hook the target tool via `extra_tools=[...]`

The verified signature (`LollmsPersonality.from_handbag`, `lollms_personality.py`):

```python
@classmethod
def from_handbag(
    cls,
    handbag_path: Union[str, Path],
    lollms_client: Optional[Any] = None,
    extra_skills_dirs: Optional[List[Union[str, Path]]] = None,
    extra_tools: Optional[List[Union[str, Path]]] = None,          # <-- THE HOOK
    extra_knowledge_bases: Optional[List[Union[str, Path]]] = None,
    extra_safe_stores: Optional[List[Any]] = None,
    workspace_path: Optional[Union[str, Path]] = None,
    capabilities: Optional[Any] = None,
    model_params: Optional[Dict[str, Any]] = None,
    max_tokens_per_turn: int = 4096,
    workspace: Optional[Union[str, Path]] = None,                  # legacy alias
    **kwargs: Any,
) -> "LollmsPersonality":
    ...
```

Full, copy-pasteable construction:

```python
from pathlib import Path
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

handbag_path    = Path("editor_handbag").resolve()
toolset_folder  = Path("workspace/file_tools").resolve()   # contains TOOLS.py
workspace_root  = Path("workspace").resolve()              # editor works here

editor = LollmsPersonality.from_handbag(
    handbag_path=handbag_path,
    lollms_client=client,
    extra_tools=[toolset_folder],        # <-- the target toolset becomes callable
    workspace_path=workspace_root,
    max_tokens_per_turn=8192,
)

# Sanity check: the target tool must appear in the discovered tools
print([t["name"] for t in editor.list_tools_structured()])
```

### What `extra_tools` actually does (verified, line-by-line)

Inside `from_handbag`:

```python
tool_binding = None
all_tool_files = list(hb.tool_files)                # native handbag tools
if extra_tools:
    for t in extra_tools:
        resolved_t = Path(t).resolve()
        if resolved_t.exists() and resolved_t not in all_tool_files:
            all_tool_files.append(resolved_t)       # deduped against native tools

if all_tool_files:
    tool_binding = LCPBinding(tool_files=[str(f) for f in all_tool_files])
```

Key facts to internalise:

- **Argument name is `tool_files=`**, not `tools_folders=`. Callers who copy the
  `LCPBinding(tools_folders=...)` spelling from older guidance will find it does not
  participate in this code path.
- Each entry is **`Path(t).resolve()`d and existence-checked**. A path that does not
  exist on disk at construction time is silently dropped.
- Duplicates against `hb.tool_files` are removed by resolved path. A native handbag tool
  and an `extra_tools` entry that resolve to the same path become one entry.
- Entries are **executable**, not just visible. The resulting
  `len(tool_binding.discovered_tools)` counts them, and their `tool_*` functions are
  callable by the agent.
- If `discovered_tools == 0` after construction, an **explicit error is logged**. Typical
  causes: missing `TOOLS.py`, broken Python syntax, or zero `tool_*` functions.

After construction, `from_handbag` persists the extras so a later refresh can re-read them:

```python
object.__setattr__(
    pers,
    "_extra_tool_files",
    [Path(t).resolve() for t in (extra_tools or []) if Path(t).resolve().exists()],
)
```

This `_extra_tool_files` attribute is the mechanism that makes hot-reload possible (see
§7).

---

## 6. Step 4 — The edit loop

The editor persona runs as a normal agent. The loop is:

### 6.1 — Inspect `TOOLS.py`

Chat with the persona asking it to read the toolset file, e.g.:

```python
editor.chat(
    prompt=(
        "Read the attached toolset's TOOLS.py in full. List every `tool_*` "
        "function with its signature and one-line purpose. Do not edit anything yet."
    ),
    enable_workspace_tools=True,
    enable_python_exec=False,        # keep the editor's capabilities tight
)
```

The editor persona's system prompt already mandates reading via its workspace tools.
Because `workspace_path` is the workspace root, a path like
`file_tools/TOOLS.py` is within scope.

### 6.2 — Execute the current `tool_*` to observe behavior

Ask the agent to run the target tool with a representative input **before** editing:

```python
editor.chat(
    prompt=(
        "Execute `tool_<name>` with input <representative-input>. "
        "Report the exact output verbatim. Do not summarize."
    ),
    enable_workspace_tools=True,
    enable_python_exec=True,         # allow it to drive the toolset if required
)
```

The tool is exactly the same code that will later be edited — the persona is verifying
current behavior at the same address the fix will later be measured against.

### 6.3 — Patch or rewrite `TOOLS.py`

Typical directive:

```python
editor.chat(
    prompt=(
        "The defect: <one-sentence description of the wrong behavior>. "
        "Edit `file_tools/TOOLS.py` with a SEARCH/REPLACE patch that fixes it. "
        "Preserve every other `tool_*` function unchanged. Do not introduce new imports "
        "unless strictly needed. Reply with a short summary and stop."
    ),
    enable_workspace_tools=True,
)
```

The editor agent uses its file tools (or its `<artifact>` pathway) to write the change.
The file on disk is updated.

### 6.4 — Re-hook (the ONLY verified hot-reload path)

`LollmsPersonality` exposes **`refresh_tools_and_skills()`** — a public method that takes
**no arguments** and rebuilds the tool binding and skills from:

- the handbag's native tools (`hb.tool_files`, freshly re-scanned from disk),
- `self._extra_tool_files` (the persisted `extra_tools` entries).

```python
# The exact, verified recipe — use this verbatim.
from pathlib import Path

editor._extra_tool_files.append(Path(toolset_folder).resolve())
editor.refresh_tools_and_skills()
```

What actually happens inside `refresh_tools_and_skills()`:

```python
def refresh_tools_and_skills(self) -> None:
    hb = Handbag(self.handbag_path)
    native_tool_files  = hb.tool_files
    extra_tool_files   = [f for f in self._extra_tool_files if f.exists()]
    all_tool_files     = native_tool_files + extra_tool_files
    self._tool_binding = LCPBinding(tool_files=[str(f) for f in all_tool_files])
    # ...then rebuild skill manager against the merged skills dirs
```

Consequences you must design around:

- **No path argument.** `refresh_tools_and_skills()` does not accept the path to the
  toolset; it re-reads `self._extra_tool_files`. To hot-add a *new* toolset, append to
  `_extra_tool_files` first (as above) and then call the method.
- **Extras are dropped if missing on disk at refresh time.** `[f for f in ... if f.exists()]`
  means a deleted toolset vanishes from the binding on the next refresh, with no error.
- **The handbag folder must still exist.** If `self.handbag_path` no longer points at a
  directory, the method warns and returns without touching the binding.

### 6.5 — Validate by executing the `tool_*` again

```python
editor.chat(
    prompt=(
        "Execute `tool_<name>` with the SAME input as before. "
        "Compare to the pre-edit output. State PASS or FAIL and why."
    ),
    enable_workspace_tools=True,
    enable_python_exec=True,
)
```

---

## 7. Step 5 — Verifying the edited tool

Two levels of verification are available.

### 7.1 — Through the persona's tool dispatch (the realistic path)

The most faithful check is exactly what the editor agent does:
`chat()` instructing it to call the edited tool. That exercises the same code path as the
final consumer of the tool.

### 7.2 — Directly through the binding (for a scripted check)

If you *know* the tool's parameter schema, you can call it directly through the
persona's internal tool binding:

```python
# Use the host's tool-dispatch shape here — the callable is reached through the
# binding that from_handbag/refresh_tools_and_skills installed.
binding = editor._tool_binding

result = binding.execute_tool(
    tool_name="tool_<name>",
    params={"<arg>": "<value>"},       # argument names must match the tool signature
)
print(result)
```

`LCPBinding.execute_tool(tool_name, params)` returns a dict with at least `success`, and
`output` or `error`. Argument names must match the `tool_*` function's parameters exactly;
mismatches return errors from the callable.

> If you are calling through a chat / CLI host, use **that host's** tool-dispatch
> mechanism. The binding method above is the most direct API surface, but the host may
> wrap it (e.g. to gate dangerous categories, inject a workspace cwd, etc.).

---

## 8. `tool_tool_maker(...)` — scaffolding a fresh toolset

Sometimes the editor's job is to *author* the initial toolset, not just edit it. The
`tool_maker` LCP toolset provides that.

```python
def tool_tool_maker(
    name: str = "",
    description: str = "",
    implementation: str = "",
    title: Optional[str] = None,
    author: str = "Lollms Agent",
    version: str = "0.1.0",
    overcode_if_exists: bool = False,
    overwrite: Optional[bool] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    ...
```

Behavior, verified against `tool_maker.py`:

- **Scaffolds `<workspace>/<name>/TOOLS.py` + `<workspace>/<name>/description.yaml`.**
  The workspace root is resolved from `env.tool_context.workspace_path` (host-injected);
  when absent it falls back to the process CWD.
- **`name` is sanitized.** Lowercased, separators (whitespace, `/`, `.`, `-`, `\`) become
  `_`; non-`[a-z0-9_]` characters are stripped; leading digits are prefixed `toolset_`;
  reserved names (`lcp`, `default_tools`, `tests`, …) are refused.
- **`implementation` must be valid Python** (`ast.parse` runs before writing) and must
  define **at least one top-level `tool_<verb>` function**. Both failures return
  `{success: False, error: ...}` rather than raising.
- **Overwrite semantics**: `overwrite` wins when provided; `None` falls back to
  `overcode_if_exists`. With `overwrite=False` (or an existing non-empty folder + `None`),
  the call is refused. Setting `overwrite=True` replaces the folder.
- **Returns a dict.** On success: `{success, toolset_name, toolset_path, tool_names,
  message}`. On failure: `{success, error}`. Note the message text tells you to call
  `tool_load_generated_toolset(...)` to hot-load — see §9 for why that advice does not
  work as written.

Typical call:

```python
editor.chat(
    prompt=(
        "Create a toolset named `csv_utils` with a single tool "
        "`tool_count_rows(file_name: str) -> dict` that returns "
        "`{success: True, rows: <int>}` for a CSV in the workspace. "
        "Use tool_tool_maker."
    ),
    enable_workspace_tools=True,
)
```

---

## 9. The hot-load warning — `tool_load_generated_toolset` does **NOT** hot-load

`tool_load_generated_toolset(name, env)` from `tool_maker.py` looks for a hook named
`refresh_toolset` on the personality:

```python
refresh = getattr(personality, "refresh_toolset", None)
if callable(refresh):
    refresh(extra_tool_paths=[toolset_dir])
    loaded = True
else:
    load_error = "Personality has no 'refresh_toolset' hook; the toolset exists ..."
```

**`LollmsPersonality` does not define `refresh_toolset`.** Its verified refresh entry point
is `refresh_tools_and_skills()` (no args), which reads `self._extra_tool_files`. Therefore:

- `tool_load_generated_toolset(...)` **returns `{success: True, loaded: False}`** on
  `LollmsPersonality`.
- It **never raises.** The toolset exists on disk, but its `tool_*` functions are **not**
  callable from the running persona.
- The message text it returns will describe the toolset as existing but not hot-loadable.

### What to use instead

Use the verified recipe from §6.4:

```python
from pathlib import Path

editor._extra_tool_files.append(Path(toolset_dir).resolve())
editor.refresh_tools_and_skills()
```

Do **not** write code or documentation that promises:
- `tool_load_generated_toolset` attaches a toolset to a running persona,
- `refresh_toolset(extra_tool_paths=...)` exists on the personality,
- `LCPBinding(tools_folders=...)` is used by `from_handbag` or `refresh_tools_and_skills`
  (those paths pass `tool_files=[...]`).

---

## 10. Pitfalls (verified)

1. **`extra_tools` folder without `TOOLS.py`** (and without a usable
   `<folder>.py` / first-`.py`-with-`tool_*` fallback) → `discovered_tools == 0` and an
   explicit error logged by `from_handbag`. Nothing becomes callable.
2. **`extra_tools` paths that do not exist at construction are silently dropped.**
   `if resolved_t.exists()` is the only gate. A typo in a path produces no error at
   call sites — only the eventual missing tool at execution time.
3. **`refresh_tools_and_skills()` silently drops non-existent entries** from
   `_extra_tool_files` at refresh time. It does not prune persisted entries from disk; if
   you want a removed toolset to stop being reattached on later refreshes, remove it from
   `_extra_tool_files` yourself.
4. **`tool_load_generated_toolset` returns `{success: True, loaded: False}`** on
   `LollmsPersonality` (see §9). The toolset is on disk but not hot-loaded. Use the
   `_extra_tool_files` + `refresh_tools_and_skills()` recipe instead.
5. **Office / PDF toolsets are gated off by default.** `LCPBinding` requires an explicit
   `set_session_flags()` call (e.g. `enable_powerpoint`, `enable_word`, `enable_excel`,
   `enable_office_files`) before those tool families appear in `discovered_tools`. If your
   target toolset lives in that family, request the flags.
6. **There is one binding per persona, not one per source.** `hb.tool_files` and
   `extra_tools` contribute to the **same** `LCPBinding`. `list_tools_structured()` reports
   `source="handbag"` for any tool whose `_source_file` sits under `handbag_path` **or**
   for tools present in `discovered_tools` — so a `extra_tools` tool may be labelled
   `handbag` when measured this way. Do not build logic on tool provenance; use explicit
   names.
7. **`refresh_tools_and_skills()` requires a valid `self.handbag_path`.** If the folder has
   been removed from disk, the method warns and returns without touching the tool binding.

---

## 11. Minimal end-to-end example

A self-contained script: build the handbag, write a sample `TOOLS.py`, construct the
editor with `extra_tools`, print the tool inventory, edit `TOOLS.py`, re-hook, and print
again.

```python
"""tool_editor_demo.py — minimal runnable editor-agent round trip.

Run:
    python tool_editor_demo.py

The script:
  1. Creates <cwd>/editor_demo/editor_handbag/SOUL.md
  2. Creates <cwd>/editor_demo/workspace/file_tools/TOOLS.py (a trivial toolset)
  3. Builds an editor persona with from_handbag(..., extra_tools=[...])
  4. Prints list_tools_structured() (before)
  5. Edits TOOLS.py (adds a second tool)
  6. Appends to _extra_tool_files and calls refresh_tools_and_skills()
  7. Prints list_tools_structured() (after)
"""
from pathlib import Path
import textwrap

from lollms_client.lollms_personality.lollms_personality import LollmsPersonality


ROOT         = Path("editor_demo").resolve()
HANDBAG      = ROOT / "editor_handbag"
WORKSPACE    = ROOT / "workspace"
TOOLSET_DIR  = WORKSPACE / "file_tools"
TOOLSET_FILE = TOOLSET_DIR / "TOOLS.py"


def _write_handbag() -> None:
    HANDBAG.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    (HANDBAG / "SOUL.md").write_text(textwrap.dedent("""\
        ---
        name: ToolSmith
        author: lollms-client
        version: '1.0'
        category: engineering
        description: Focused editor for a single attached LCP toolset.
        ---

        You are ToolSmith, a dedicated LCP toolset editor.

        Your ONLY job is to inspect, edit, and validate the target toolset
        attached via `extra_tools`. You must NOT perform unrelated tasks.

        Mandatory loop: READ -> OBSERVE -> REASON -> EDIT -> RELOAD -> VALIDATE.
        Never edit files outside the attached toolset folder.
        Never claim an edit is "hot-loaded" without calling refresh_tools_and_skills.
    """), encoding="utf-8")

    (HANDBAG / "handbag.yaml").write_text(textwrap.dedent("""\
        name: Editor Handbag
        version: '1.0'
        description: Handbag for tool-editing personas.
        skills_mode: loadable
    """), encoding="utf-8")


def _write_initial_toolset() -> None:
    TOOLSET_DIR.mkdir(parents=True, exist_ok=True)
    (TOOLSET_FILE).write_text(textwrap.dedent('''\
        """file_tools — trivial sample toolset for the editor demo."""

        from __future__ import annotations
        from typing import Any, Dict, Optional

        TOOL_LIBRARY_NAME = "File Tools"
        TOOL_LIBRARY_DESC = "Trivial file tools, safe for demoing edits."
        TOOL_LIBRARY_ICON = "🧪"


        def init_tools_library(config: Optional[dict] = None) -> None:
            return None


        def tool_echo(message: str) -> Dict[str, Any]:
            """Return the given message verbatim.

            Args:
                message: The string to echo.

            Returns:
                {"success": True, "output": "<message>"}
            """
            return {"success": True, "output": str(message)}
    '''), encoding="utf-8")


def _edit_toolset_add_second_tool() -> None:
    """Appends a second `tool_*` function to TOOLS.py preserving existing content."""
    current = TOOLSET_FILE.read_text(encoding="utf-8").rstrip() + "\n"
    addition = textwrap.dedent('''

        def tool_shout(message: str) -> Dict[str, Any]:
            """Return the message uppercased.

            Args:
                message: The string to uppercase.

            Returns:
                {"success": True, "output": "<MESSAGE>"}
            """
            return {"success": True, "output": str(message).upper()}
    ''')
    TOOLSET_FILE.write_text(current + addition, encoding="utf-8")


def _print_tool_set(personality: LollmsPersonality, label: str) -> None:
    tools = personality.list_tools_structured()
    own = [t for t in tools if t["source"] in ("handbag", "custom")]
    print(f"\n--- {label}: {len(own)} handbag/custom tool(s) ---")
    for t in own:
        print(f"  [{t['source']:8}] {t['name']}")


def main() -> None:
    _write_handbag()
    _write_initial_toolset()

    # Optional: real client enables the agent loop and tool dispatch.
    # client = LollmsClient(llm_binding_name="ollama",
    #                       llm_binding_config={"model_name": "qwen2.5-coder:7b"})
    client = None

    editor = LollmsPersonality.from_handbag(
        handbag_path=HANDBAG,
        lollms_client=client,
        extra_tools=[TOOLSET_DIR],
        workspace_path=WORKSPACE,
        max_tokens_per_turn=4096,
    )

    _print_tool_set(editor, "BEFORE edit")

    # --- Edit the toolset source on disk -----------------------------------
    _edit_toolset_add_second_tool()

    # --- Re-hook using the VERIFIED hot-reload recipe ----------------------
    editor._extra_tool_files.append(TOOLSET_DIR.resolve())
    editor.refresh_tools_and_skills()

    _print_tool_set(editor, "AFTER  edit + refresh")

    # If a real client is present, you can now drive the editor agent:
    # editor.chat(
    #     prompt="Execute tool_shout with message='hello' and report the exact output.",
    #     enable_workspace_tools=True,
    #     enable_python_exec=True,
    # )


if __name__ == "__main__":
    main()
```

Notes on the demo:

- The `_extra_tool_files.append(...)` + `refresh_tools_and_skills()` sequence is exactly
  the verified hot-reload path. It does not depend on LLM involvement; the edits to disk
  are what actually matter.
- Because `list_tools_structured()` labels by provenance (handbag vs custom), all
  non-default tools appear together; do not rely on the label to distinguish
  handbag-native from `extra_tools`.

---

## 12. Verified against

Source symbols and files this document was checked against:

- `LollmsPersonality.from_handbag` — `src/lollms_client/lollms_personality/lollms_personality.py`
  - `extra_tools` resolution / dedup / `LCPBinding(tool_files=...)` call
  - `_extra_tool_files` persistence via `object.__setattr__`
  - Workspace resolution ordering (`workspace_path` → `workspace` → `hb.workspace_dir` → `None`)
- `LollmsPersonality.refresh_tools_and_skills()` — same file
  - No-argument signature; re-reads `hb.tool_files` + `self._extra_tool_files`;
    rebuilds `_tool_binding`.
- `LollmsPersonality.list_tools_structured()` — same file (provenance labeling).
- `LollmsPersonality._tool_binding` — same file (direct `execute_tool` surface).
- `Handbag` layout and `PersonalityBundle.parse_soul_md` — same file.
- `tool_tool_maker`, `tool_load_generated_toolset`, `write_toolset`,
  `_sanitize_toolset_name`, `_get_workspace_root`, `_render_tools_module`,
  `_render_description_yaml`, `_validate_python_source`, `_extract_tool_functions`,
  `_resolve_toolset_dir` — `src/lollms_client/tools_bindings/lcp/default_tools/tool_maker/tool_maker.py`
  - Signature; sanitization rules; overwrite / `overcode_if_exists` semantics;
    return shape `{success, toolset_path, tool_names, ...}`; `loaded: False` behavior
    because `refresh_toolset` does not exist on `LollmsPersonality`.
- `LCPBinding` constructor kwarg used by `from_handbag` / `refresh_tools_and_skills`:
  `tool_files=` — `src/lollms_client/tools_bindings/lcp/__init__.py`
  - `TOOLS_ENTRYPOINT` (= `"TOOLS.py"`) and the folder / `<folder>.py` / first-`.py`
    fallbacks live in `src/lollms_client/tools_bindings/lcp/tool_layout.py`.
  - Office / PDF gate via `set_session_flags()`.