# 01 — The LCP Binding Core (`tools_bindings/lcp/__init__.py`)

> **Purpose**: The anatomy of the LCP binding module itself — the `LCPBinding`
> class, its lazy-import machinery, its file-loading semantics, and its public
> API. This is the "engine room" document; the discovery *pipeline* is covered
> in [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) and the
> admission gates in [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md).

---

## 1. Module layout

```
src/lollms_client/tools_bindings/lcp/__init__.py
```

This single module contains the entire binding core:

- the `LCPBinding` class (inherits `LollmsToolBinding` from
  `src/lollms_client/lollms_tools_binding.py`),
- the lazy-import helper `_lazy_import_tools_file(py_file_path)`,
- the per-library loader `_load_tool_file(...)`,
- the library-file iterator `_iter_library_files(...)`,
- the AST/regex schema extractors (`_parse_tool_via_ast`,
  `_parse_tool_via_regex`, `_extract_tool_names_from_file`),
- the office-admission gate constants and `_office_library_admitted(...)`,
- the mount helpers `is_library_mounted(...)` / `mount_tool_library(...)`,
- and `execute_tool(...)` — the execution seam.

Sibling files: `description.yaml` (binding metadata) and `README.md`
(tool-creation guidelines for library authors).

---

## 2. `LCPBinding` — the class

`LCPBinding` is constructed with the standard binding kwargs pattern and
registers itself under the binding name `"lcp"`. It is selected by passing
`tools_binding_name="lcp"` (plus a `tools_binding_config` dict) to
`LollmsClient`.

Its responsibilities, in one line each:

| Responsibility | Mechanism |
|---|---|
| Find candidate library files | `_iter_library_files(...)` over the `default_tools` folder |
| Import each library once, safely | `_lazy_import_tools_file(...)` + `_load_tool_file(...)` |
| Initialize each library | calls `module.init_tools_library(host_config)` if present |
| Decide which libraries are admitted | `_office_library_admitted(...)` + `TOOL_LIBRARY_LOAD_FLAG` (fail-closed) |
| Extract tool schemas | `_parse_tool_via_ast(...)` with `_parse_tool_via_regex(...)` fallback |
| Track what is mounted | `is_library_mounted(...)` / `mount_tool_library(...)` via `_extract_tool_names_from_file` |
| Execute tool calls | `execute_tool(tool_name, params, discussion_instance, **kwargs)` |

The binding holds a `discovered_tools` registry — the accumulated result of
discovery — which downstream consumers (personality `_discover_tools`, chat
prompt builders) read to build LLM-visible tool specs.

---

## 3. Lazy import machinery: `_lazy_import_tools_file`

```python
_lazy_import_tools_file(py_file_path)
```

**Contract**: import a library file under a **persistent, deterministic module
name**:

```
lollms_client.tools_bindings.lcp.persistent_<library_key>
```

where `<library_key>` is derived from the library file (e.g. `word_tools`).
Key properties:

- **Persistent naming**: the module is registered in `sys.modules` under the
  persistent name, so re-discovery does not re-execute the library file and
  module-level state (caches, imported heavy dependencies) survives across
  discovery rounds.
- **Isolation**: each library gets its own module identity; two libraries with
  the same basename cannot collide.
- **Failure semantics**: any exception raised during import propagates to the
  caller (`_load_tool_file`), which attributes it to library initialization —
  see §4.

---

## 4. `_load_tool_file` — loading and initializing one library

```python
_load_tool_file(...)
```

**Flow** for one candidate library file:

1. Call `_lazy_import_tools_file(py_file_path)` to import the module.
2. If the module defines `init_tools_library(host_config)`, call it with the
   host configuration.
3. On success, the library's tools become available for schema extraction and
   execution.

**Error attribution law**: ANY raise during import OR during
`init_tools_library(...)` is caught and reported as:

```
Tool initialization failed for '{library_key}': {e}
```

(error site at approximately line 562 of `__init__.py`, as of this writing).
This single message therefore covers two distinct root causes:

- the file cannot be **imported** (syntax error, missing dependency, exception
  at module level), or
- the file imports fine but `init_tools_library(...)` **raises**.

**Historical note (verified incident)**: the `word_tools.py` truncation saga —
a corrupted `word_tools.py` (dangling `{"` at EOF, orphaned tokens, and a
frankensteined spec whose `name` and `callable` were inverted) produced exactly
this diagnostic, twice, until a full-file rewrite fixed it. When you see
`Tool initialization failed for 'word_tools'`, suspect file corruption first,
then dependencies, then `init_tools_library` logic.

**Library-side contract** (exemplified by `word_tools.py`):

```python
def init_tools_library(config=None):
    """Never raises: dependency installation is soft-failed via pipmaster."""
```

Libraries SHOULD make `init_tools_library` non-raising — dependency problems
are reported through tool results, not through loader crashes. `word_tools.py`
follows this doctrine: its `init_tools_library(config=None)` attempts pipmaster
installs and swallows failures so that discovery never dies on a missing
optional dependency.

---

## 5. `_iter_library_files` — finding candidate libraries

```python
_iter_library_files(...)
```

**Contract**: yield the Python files of every library under `default_tools/`,
using the **directory + primary-file pattern**:

```
default_tools/<name>/<name>.py        ← canonical layout (dir + primary file)
default_tools/<name>.py               ← legacy flat layout (also supported)
```

So for a library named `word_tools`, the canonical file is
`default_tools/word_tools/word_tools.py`. A library folder may also contain a
`description.yaml` declaring metadata (including its load flag — see
[03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md)).

The iterator yields candidates; it does not decide admission (that is the gate
layer's job) and does not import anything (that is `_load_tool_file`'s job).
This separation keeps discovery cheap to enumerate and safe to retry.

---

## 6. `description.yaml` and host config

Each library folder may carry a `description.yaml` next to its primary file.
The binding reads it for library metadata — most importantly the **load flag**
declaration consumed by the `TOOL_LIBRARY_LOAD_FLAG` gate (a library that
declares no load flag passes that gate unconditionally; a library that declares
one is admitted only when the corresponding host-config/session flag is
truthy). The full admission decision tree is documented in
[03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md).

The **host config** is the configuration object passed down from
`LollmsClient` / the personality / the discussion. It is the source of truth
for flag values and is forwarded to `init_tools_library(host_config)` so
libraries can adapt (e.g. resolve the workspace root).

---

## 7. Public API of the binding

The binding exposes the standard `LollmsToolBinding` surface plus LCP-specific
helpers:

```python
# Standard contract (per tools_bindings/README.md)
discover_tools(specific_tool_names=None, **kwargs) -> List[Dict[str, Any]]
list_tools(**kwargs) -> List[Dict[str, Any]]          # alias of discover_tools
execute_tool(tool_name: str, params: Dict[str, Any], discussion_instance=None, **kwargs) -> Dict[str, Any]

# LCP-specific surface
_lazy_import_tools_file(py_file_path)                  # persistent module import
_load_tool_file(...)                                  # import + init one library
_iter_library_files(...)                              # enumerate candidate files
_parse_tool_via_ast(py_file_path)                     # primary schema extraction
_parse_tool_via_regex(...)                            # degraded fallback
_extract_tool_names_from_file(py_file)                # tool-name set for mounts
is_library_mounted(...) / mount_tool_library(...)     # mount state
_office_library_admitted(...)                         # admission gate
```

**Return shapes**:

- `discover_tools` → list of tool dicts with `name`, `description`,
  `input_schema` (JSON-Schema-style parameter definitions).
- `execute_tool` → normalized result dict: `success`, `error`, `output`,
  `file_name`, `context_files`, `status_code` — full contract in
  [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md).

---

## 8. Key invariants

- `_lazy_import_tools_file` imports every library under the persistent module
  name `lollms_client.tools_bindings.lcp.persistent_<library_key>` — module
  state survives re-discovery, and re-imports are avoided.
- ANY import or `init_tools_library` raise is attributed to
  `Tool initialization failed for '{library_key}': {e}` (~L562) — one message,
  two root causes (import failure vs init failure).
- Libraries SHOULD implement `init_tools_library(config=None)` so it never
  raises; `word_tools.py` is the reference implementation (pipmaster
  soft-fail).
- `_iter_library_files` yields `default_tools/<name>/<name>.py` (canonical) and
  legacy flat `default_tools/<name>.py`; it does not gate or import.
- `description.yaml` per library declares metadata including the load flag for
  the `TOOL_LIBRARY_LOAD_FLAG` gate; no declared flag ⇒ that gate passes.
- The binding's `discovered_tools` registry is the single source of truth that
  downstream consumers (personality, chat prompt builders) read.
- The binding implements the same `discover_tools` / `list_tools` /
  `execute_tool` contract as every other `LollmsToolBinding` — the LLM-facing
  interface is transport-agnostic.