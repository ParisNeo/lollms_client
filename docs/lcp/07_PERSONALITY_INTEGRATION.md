# 07 — Personality Integration (`LollmsPersonality`)

> **Purpose**: How `LollmsPersonality`
> (`src/lollms_client/lollms_personality/lollms_personality.py`) integrates
> the LCP — capability flags, the `get_tools()` spec consumption contract, the
> WCD (working-context directory) law, mounting vs discovery, and the
> client/binding seam. Prerequisites:
> [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md),
> [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md),
> [06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md).

---

## 1. The personality's role in the LCP

`LollmsPersonality` is the **capability owner**. It decides which tool
families are enabled (via flags), discovers tools from the LCP binding and
from libraries' `get_tools()` seams, mounts toolsets for the chat, and holds
the client/binding seam through which execution flows. The chat mixin
([06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md)) consumes what
the personality prepares.

---

## 2. `_discover_tools(...)` — the personality-side discovery

The personality's discovery entry point (the `get_tools()` spec consumption
site, around ~L917 of `lollms_personality.py`, per verified findings):

- **Capability flags** consulted: `enable_word`, `enable_excel`,
  `enable_powerpoint`, `enable_office_files`, `enable_workspace_tools`,
  `allow_computer_use`, and the wider flag family (see
  [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md)).
- **Spec consumption contract**: for each tool spec, the personality reads
  `callable`, `parameters`, `description`, `_source_file`; the `output`
  documentation rides along with the spec.
- **Spec shape requirements** (identical to the Seam-B contract in
  [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) §3):

```python
{
    "name": "tool_build_word",          # MUST match the callable's tool name
    "description": "...",
    "parameters": [
        {"name": "title", "type": "string", "optional": False,
         "description": "Document title."},
        ...
    ],
    "output": [
        {"name": "success", "type": "boolean", "description": "..."},
        ...
    ],
    "callable": tool_build_word,
    "_source": "word_tools",
    "_source_file": "<absolute path>",
}
```

---

## 3. The WCD contract (working-context directory)

The personality owns the **WCD** — the directory all workspace-relative asset
references resolve against:

```
personality.wcd  ──►  workspace_data_path seam  ──►  library path resolution
```

- Office tools resolve asset paths **workspace-only**:
  `_candidate_working_dirs` uses the workspace as **sole root** (a `Path(".")`
  fallback exists only when no workspace is set), and
  `resolve_workspace_path` / `resolve_path` are the sanctioned resolvers.
- **NO `os.chdir` doctrine**: the process CWD is never permanently changed to
  the workspace. Instead, tools run with the workspace as CWD **for the
  duration of a call** and the CWD is transactionally restored afterward —
  this is exactly what `test/test_lcp_cwd_management.py` locks in (see
  [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md) §5).

---

## 4. Mounting vs discovery — and how they coexist

Two registration paths coexist, and knowing which one fired is a diagnostic
art:

| Path | Log signature | Meaning |
|---|---|---|
| **Per-family mount** (`_mount_word_tools`-style) | `Mounted 'word_tools' toolset (4 tools)` | The personality explicitly mounted a family's toolset; the count is the mounted tool count. |
| **`discovered_tools` fallback** | `Registered tool_grep_document via direct discovered_tools fallback` | A tool was not registered via the mount path, so it was registered straight from the binding's `discovered_tools` registry. |

**The `Personality binding 'null' yielded 0 chat tool specs` diagnostic**:
binding name is `null` AND `discovered_tools` count is 0 → the defensive
fallback engaged. Root causes: the tools binding was not configured
(`tools_binding_name="lcp"` missing), or discovery never ran / was fully
gated out. Fix: configure the binding and check the admission gates
([03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md)).

**Coexistence law**: mounted toolsets and LCP discovery are complementary —
mounts give explicit, family-scoped control (with expected tool counts),
while `discovered_tools` guarantees nothing admitted by discovery is lost.
The fallback log is informational but signals the mount path missed the tool;
verify seam consistency when it appears.

---

## 5. The client/binding seam

The personality holds a `lollms_client` (a `LollmsClient` instance) and binds
tool execution through it:

- The client's tools binding (`tools_binding_name="lcp"` → `LCPBinding`) is
  the execution engine.
- `lollms_tools_binding.py` (around line 159, per verified findings) consumes
  the **same spec shapes** the personality reads — `name`, `description`,
  `parameters`, `output`, `callable`, `_source`, `_source_file` — so a spec
  authored once serves both discovery and execution.
- Execution flows: chat mixin intercepts a tool call → personality/client seam
  → `LCPBinding.execute_tool` → library callable → normalized result dict.

---

## 6. End-to-end diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ LollmsPersonality construction                                      │
│   • holds lollms_client (LollmsClient with tools_binding_name=lcp)  │
│   • reads capability flags (enable_word / enable_excel / ...)       │
│   • owns the WCD → workspace_data_path seam                         │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Discovery                                                            │
│   • LCP binding discovery (AST seam + regex fallback)                │
│   • get_tools() seam specs (callable/parameters/description/         │
│     _source_file; output rides along)  ← consumed ~L917              │
│   • admission gates (fail-closed) apply to both seams                │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Registration (two coexisting paths)                                  │
│   • per-family mounts → "Mounted 'word_tools' toolset (4 tools)"     │
│   • discovered_tools fallback → "Registered tool_X via direct        │
│     discovered_tools fallback"                                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Chat-side mount + advertisement                                      │
│   • _build_tool_prompt + per-capability builders                     │
│   • office-mount seam (admission sets + set_session_flags)           │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Execution                                                            │
│   • LLM tool call → chat mixin → LCPBinding.execute_tool             │
│   • workspace CWD for the call, transactionally restored             │
│   • result dict → normalization → USER-role artefact attachment      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Key invariants

- The personality is the **capability owner**: flags, discovery, mounts, and
  the WCD all live here; the chat mixin consumes what it prepares.
- `_discover_tools(...)` consumes `get_tools()` specs by reading `callable`,
  `parameters`, `description`, `_source_file` (output documentation rides
  along) — consumption site ~L917 of `lollms_personality.py` (per verified
  findings).
- Spec shape: `name` / `description` /
  `parameters[{name,type,optional,description}]` /
  `output[{name,type,description}]` / `callable` / `_source` / `_source_file`;
  `name` MUST match the callable.
- **WCD law**: `personality.wcd` → `workspace_data_path` seam; asset
  resolution is workspace-only (`_candidate_working_dirs` workspace-sole-root,
  `resolve_workspace_path`/`resolve_path`); **NO permanent `os.chdir`** —
  CWD changes are call-scoped and transactionally restored.
- Two registration paths coexist: per-family mounts (`Mounted 'X' toolset
  (N tools)` log family) and the `discovered_tools` fallback (`Registered
  tool_X via direct discovered_tools fallback`).
- `Personality binding 'null' yielded 0 chat tool specs (discovered_tools
  count: 0)` = binding name null + empty registry → defensive fallback;
  fix by configuring `tools_binding_name="lcp"` and checking gates.
- `lollms_tools_binding.py` (~L159) consumes the same spec shapes as the
  personality — one spec shape serves discovery and execution.
- The personality holds `lollms_client`; execution flows chat mixin →
  personality/client seam → `LCPBinding.execute_tool` → library callable.