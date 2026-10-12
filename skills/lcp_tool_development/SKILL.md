---
title: "LCP Tool Architecture and Human Intervention Mastery"
description: "Complete guide for authoring LCP tool sets: binding discovery, AST schema generation, execution contract, host configuration injection, human-in-the-loop validation, dynamic registration, and mounting APIs for CLI and GUI environments."
category: "tool_engineering"
tags: [lcp, tools, human_in_the_loop, security, ast, safe_mode, execution]
visibility: loadable
modifiable: true
---

# LCP Tool Architecture & Human-in-the-Loop Engineering

The **LollmsCommunicationProtocol (LCP)** is the lightweight, zero-dependency in-process tool execution engine of `lollms_client`, implemented by the `LCPBinding` class (`src/lollms_client/tools_bindings/lcp/__init__.py`, module attribute `BindingName = "LCPBinding"`). It discovers, parses, and executes Python tools in a sandboxed workspace environment.

---

## 1. Core LCP Tool Design Doctrine

1. **Agnostic & Decoupled**: LCP tools are plain Python functions. They operate strictly on primitive arguments (`str`, `int`, `float`, `bool`, `list`, `dict`) and return standard dictionaries. Tools do NOT know about discussions, artefacts, or clients.
2. **Current Working Directory (CWD) Sandbox**: The orchestrator changes the process CWD to the isolated workspace folder before invoking a tool. Tools resolve workspace files using relative paths (`Path("data.csv")`).
3. **AST Schema Generation — no `description.yaml`**: The binding does NOT read any per-tool description file. Schemas are generated purely from Python source: function names (`tool_*`), type annotations, defaults, and docstrings, parsed with the `ast` module at discovery time.
4. **Tool names = function names**: The LLM calls `tool_execute_query`, not the file stem. There is no `execute` entry-point fallback. Every function whose name starts with `tool_` is registered as a tool (nested functions included); helpers without the prefix are invisible to the LLM.

---

## 2. Tool Discovery Contract (How the Binding Finds Your Tools)

`LCPBinding` locates tools from its constructor configuration:

| Parameter | Behavior |
| :--- | :--- |
| `tools_folders` (or `tools_folder_path`) | A path or list of paths to scan. Accepts `str`, `Path`, or `list`. |
| `tool_files` | An explicit path or list of `.py` files to load. **Providing `tool_files` implies an empty folder scan** — folder discovery is skipped. |
| *(neither given)* | Defaults to the built-in `tools_bindings/lcp/default_tools` folder. |

### Folder scanning rules (`_discover_local_tools`)

For each configured folder, for each item on disk:

- **Sub-directory `X/`**: the primary module is `X/X.py` (directory name must match the file name). If it exists, it is loaded, then **all sibling `*.py` files** in `X/` are loaded as well. If `X/X.py` does not exist, any top-level `.py` files inside `X/` are used as fallback modules.
- **Loose `.py` file** directly in the folder: loaded directly.
- **`__init__.py` is always skipped.**
- **Dedup by tool name**: if the same `tool_*` function name appears in multiple files, the first discovery wins; later duplicates are silently ignored.

> **Important**: discovery is *lazy registration*. At construction time the binding only parses files with AST and records tool specs; module code is NOT executed until first use (see §7). This keeps startup fast and side-effect-free.

---

## 3. Anatomy of a Multi-Tool LCP Library

Create a `.py` file inside your tools directory (e.g. `my_tools/database_ops.py`):

```python
"""
database_ops.py — Multi-tool library for SQLite querying and maintenance.
"""
import sqlite3
from typing import Any, Dict, Optional
from pathlib import Path

TOOL_LIBRARY_NAME = "Database Operations"
TOOL_LIBRARY_DESC = "Executes queries and schema inspections on SQLite databases in the workspace."
TOOL_LIBRARY_ICON = "🗄️"

# Host configuration state
AUTONOMY_LEVEL = "safe"
_CONFIRM_HANDLER = None

def init_tools_library(config: Optional[dict] = None) -> None:
    """Invoked lazily when the tool module is first loaded or configured by the host."""
    global AUTONOMY_LEVEL, _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        AUTONOMY_LEVEL = config.get("autonomy_level", "safe")
        _CONFIRM_HANDLER = config.get("confirm_handler")

def tool_database_ops_prompt() -> str:
    """Dynamic prompt provider — NOT registered as a tool."""
    db_files = [f.name for f in Path(".").glob("*.db")]
    return f"Active workspace SQLite databases available: {db_files or 'None'}"

def tool_execute_query(database_name: str, query: str) -> Dict[str, Any]:
    """
    Executes a SELECT query on a workspace SQLite database.

    Args:
        database_name (str): Name of the .db file in the workspace.
        query (str): The SQL SELECT query to execute.
    """
    db_path = Path(database_name)
    if not db_path.exists():
        return {"success": False, "error": f"Database '{database_name}' not found."}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        return {
            "success": True,
            "output": {
                "columns": columns,
                "rows": rows[:50],
                "total_rows": len(rows),
            },
        }
    except Exception as ex:
        return {"success": False, "error": f"SQL Error: {ex}"}
```

### Optional module-level hooks

| Hook | When it runs | Contract |
| :--- | :--- | :--- |
| `TOOL_LIBRARY_NAME` / `TOOL_LIBRARY_DESC` / `TOOL_LIBRARY_ICON` | Never executed; metadata for host UIs and mounting UIs. | Module-level string constants. |
| `init_tools_library(config)` | Once per process, at lazy module load (first tool execution or dynamic prompt call). | Receives the host config dict **positionally if the signature accepts positional args, otherwise called with no arguments** (detected via `inspect.signature`). |
| `set_confirm_handler(handler)` | At lazy load, and re-invoked whenever the host registers a new handler. | Receives the host confirmation callback. |
| `tool_<name>_prompt()` | At spec-build time (`to_chat_tool_specs` / `discover_tools`). | The name must be exactly `tool_` + the base tool name + `_prompt` (e.g. for `tool_execute_query`, the prompt fn is **`tool_execute_query_prompt`** — note the full tool name including prefix). Its return value **replaces** the tool's LLM-facing description. The function itself is never registered as a tool. |

---

## 4. Schema Generation Rules (Exact Semantics)

For each `tool_*` function, the binding builds `{name, description, input_schema}` via AST:

1. **Description** = the **first paragraph** of the docstring (text before the first blank line). If no docstring: `"No description provided."`.
2. **Parameter descriptions** come from docstring lines matching Google/`Args:`-style patterns:
   `param_name (type): description`, `param_name: description`, or `param_name - description`.
3. **Type mapping** from annotations (substring match, lowercased):

   | Python annotation | JSON schema type |
   | :--- | :--- |
   | `int` | `integer` |
   | `float`, `number` | `number` |
   | `bool` | `boolean` |
   | `list`, `array` | `array` |
   | `dict`, `object` | `object` |
   | anything else (or none) | `string` |

4. **Optionality**: a parameter with a default value becomes optional and its `default` is embedded in the schema (via `ast.literal_eval`, falling back to source text). A parameter without a default is added to `required`.
5. **Hidden host-injected parameters** — declaring any of these in a tool signature makes the runtime inject the corresponding object **without exposing it to the LLM**:
   - `tool_context` — host-provided execution context dict.
   - `lollms_client_instance` — the live `LollmsClient`.
   - `discussion_instance` — the live `LollmsDiscussion`.
   - `args` / `kwargs` (`*args`/`**kwargs`) — skipped from the schema; declaring `**kwargs` additionally disables parameter filtering (see §8).
6. **Regex fallback**: if a file fails `ast.parse` entirely (partially invalid source), the binding falls back to a regex-based extractor that never raises — it pulls `def tool_*` names, docstrings, and simple annotated params, marking tools with `_parsed_via_regex_fallback`. Keep your tool files syntactically valid; the fallback is lossy.

---

## 5. Human-in-the-Loop Interrogation & Confirmation Protocol

When a tool performs sensitive, irreversible, or privileged actions, it MUST query the host application for operator approval before execution.

### The Universal Confirmation Contract
The confirmation handler function receives the execution request and returns a tuple `(decision, reason)`:
- `decision`: `"allow"` (run once), `"always"` (auto-approve session), or `"reject"` (deny execution).
- `reason`: Explanation or feedback returned to the LLM upon rejection.

```python
def confirm_handler(source: str, script_label: str, argv: Optional[list] = None) -> tuple[str, str]:
    # Returns ("allow", "") | ("always", "") | ("reject", "User feedback...")
    ...
```

### A. Terminal CLI Confirmation Implementation

```python
import sys

def cli_confirm_handler(source: str, script_label: str, argv: list = None) -> tuple[str, str]:
    """Interactive confirmation prompt for terminal environments."""
    if not sys.stdin or not sys.stdin.isatty():
        return "allow", ""

    print("\n" + "=" * 60)
    print(f"⚠️  AUTHORIZATION REQUIRED: {script_label}")
    print("=" * 60)
    print(source[:500] + ("..." if len(source) > 500 else ""))
    print("=" * 60)

    while True:
        try:
            choice = input("Authorize action? [y]es / [n]o / [a]lways: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "reject", "Operation cancelled by user."

        if choice in ("y", "yes", ""):
            return "allow", ""
        elif choice in ("a", "always"):
            return "always", ""
        elif choice in ("n", "no"):
            reason = input("Optional feedback for the agent: ").strip()
            return "reject", reason or "User declined authorization."
```

### B. Asynchronous GUI / WebUI Confirmation Implementation (e.g. NiceGUI, WebUI)

In event-driven UI apps, the background agent thread MUST NOT block the UI loop. It dispatches a request event to a queue and awaits the user's response:

```python
import queue

def make_gui_confirm_handler(event_queue: queue.Queue):
    def _handler(source: str, script_label: str, argv: list = None) -> tuple[str, str]:
        response_queue = queue.Queue(maxsize=1)

        event_queue.put({
            "type": "approval_request",
            "source": source,
            "script_label": script_label,
            "argv": argv,
            "response_queue": response_queue,
        })

        try:
            decision, reason = response_queue.get(timeout=300)
            return decision, reason
        except Exception:
            return "reject", "Approval request timed out."

    return _handler
```

### Autonomy Levels Matrix

LCP tools support three distinct autonomy modes configured via `host_tool_configs`. Enforcement lives **inside the tool modules themselves** (e.g. `execute_python`, `system_shell`), not in the binding — the binding only propagates the config and handler:

| Level | Behavior | Prompt Trigger |
| :--- | :--- | :--- |
| **`strict`** | Maximum oversight. Prompts the operator for authorization on **every** Python execution and shell command. | Every execution |
| **`safe`** (Default) | Intelligent balance. Automatically executes benign code, math, data science (pandas, numpy), document generation (docx, pptx, pdf, xlsx), plotting (matplotlib, seaborn), and workspace I/O. | Prompts ONLY when risky operations are detected (process spawning via `subprocess`/`multiprocessing`, `os.system`/`os.popen`, shell scripting escapes) |
| **`full_access`** | Unrestricted autonomy. Runs all scripts and shell commands without confirmation prompts or whitelist checks. | Never prompts |

---

## 6. Host Configuration Injection (`host_tool_configs`)

`host_tool_configs` is a dict **keyed by library stem** (the module file name without `.py`):

```python
LCPBinding(host_tool_configs={
    "workspace_tools": {"root_path": "/data"},
    "execute_python": {"autonomy_level": "safe"},
    "system_shell": {"autonomy_level": "strict"},
})
```

At **lazy module load** (`_get_or_load_tool_module`), the binding:
1. Registers the module in `sys.modules` as `lollms_client.tools_bindings.lcp.persistent_<stem>` (a persistent per-process cache — module state survives across executions).
2. Looks up `host_tool_configs[<stem>]` (empty dict if absent) and injects the binding's `confirm_handler` into the config if the entry lacks one.
3. Calls the module's `set_confirm_handler(...)` if present.
4. Calls `init_tools_library(host_config)` if present — positionally if the signature allows it, else with no arguments.

`LCPBinding.set_confirm_handler(handler)` additionally patches **already-imported** `execute_python` / `system_shell` modules (via their module-level `set_confirm_handler` function or `_CONFIRM_HANDLER` attribute), so handlers can be swapped mid-session.

---

## 7. Lazy Loading & Runtime Execution Contract

`execute_tool(tool_name, params, discussion_instance=None, **kwargs)` runs this pipeline:

1. **Lookup**: the tool must be in `discovered_tools`; missing → `{"error": ..., "status_code": 404}`.
2. **Default filling**: any schema parameter the LLM omitted is filled from its `default`.
3. **Lazy module load**: the module is imported on first use and cached (see §6). Initialization failure → `status_code 500`.
4. **Parameter filtering**: unknown parameter names are **dropped** unless the function declares `**kwargs` (in which case everything passes through).
5. **Host-context injection**: `tool_context`, `lollms_client_instance`, and `discussion_instance` are injected only if the function declares them (or has `**kwargs`).
6. **Execution**: a `BaseException` in the tool returns `{"output": ..., "error": ..., "traceback": ..., "success": False, "status_code": 500}` — tools should still catch their own exceptions to give the LLM actionable messages.
7. **Unified result normalization** (flexible to different tool conventions):
   - Non-dict returns (str, list, etc.) are wrapped as `{"success": True, "output": result}`.
   - A dict result is treated as **failed** if ANY of: `success is False`; `status` is `"error"/"failure"/"failed"`; `return_code` is present and non-zero; a truthy `error` key exists (unless `success is explicitly True`).
   - `output` is **guaranteed** to exist for the LLM: if missing, the binding falls back through payload keys `result` → `data` → `response` → `papers` → `content` → `text`, then to the whole dict minus meta keys.
   - Final wrapper: success → `{"output", "status_code": 200, "success": True}`; failure → `{"output", "error", "success": False, "status_code": 500}`.

**Recommended return contract** (cleanest, though the normalizer tolerates other shapes):

```python
{"success": True, "output": "Human/LLM readable summary", ...optional payload keys...}
{"success": False, "error": "Actionable error message"}
```

---

## 8. Mounting & Registration APIs

### Mounting via `LollmsClient`
```python
client = LollmsClient(
    tools_binding_name="lcp",
    tools_binding_config={
        "tools_folders": ["./my_tools"],
        "confirm_handler": cli_confirm_handler,
        "host_tool_configs": {
            "system_shell": {"autonomy_level": "safe"},
            "execute_python": {"autonomy_level": "safe"},
        },
    },
)
```

### Mounting at Discussion / Chat Time
```python
response = personality.chat(
    prompt="Generate a report and chart in presentation.pptx",
    lollms_client=client,
    confirm_handler=cli_confirm_handler,
    shell_autonomy_level="safe",
)
```

### Programmatic mounting on the binding instance

| Method | Purpose |
| :--- | :--- |
| `mount_tool_library(lib_name)` | Mount a library folder (searched across all `tools_folders` plus the built-in `default_tools`). **Idempotent** — returns `True` immediately if fully mounted. |
| `mount_tool_library_if_absent(lib_name)` | Same, but skips the mount-warning path for repeated calls. |
| `is_library_mounted(lib_name)` | True when every expected `tool_*` function of the library is registered. |
| `find_library_for_tool(tool_name)` | Resolves which library (folder or file) provides a tool, scanning all folders and file contents; strips a leading `tool_` for directory lookup. |
| `is_tool_available(tool_name)` | True if discovered or dynamically mountable. |
| `load_tool_by_name(tool_name)` | Mounts the owning library if needed and returns the tool spec dict. This is what the `skill` mechanism invokes. |

### Dynamic (in-memory) registration

| Method | Purpose |
| :--- | :--- |
| `register_tool_from_code(prefix, code)` | `exec`s a source string into an ephemeral module, AST-parses it, and registers every `tool_*` function found (replacing same-named tools). Use for tools generated at runtime as artefacts. |
| `unregister_tools_by_prefix(prefix)` | Removes all tools whose name is `prefix` or starts with `tool_<prefix>`. |

```python
client.tools.register_tool_from_code(
    "fibonacci",
    "def tool_fibonacci(n: int) -> dict:\n"
    "    '''Computes the n-th Fibonacci number.'''\n"
    "    a, b = 0, 1\n"
    "    for _ in range(int(n)): a, b = b, a + b\n"
    "    return {'success': True, 'output': str(a)}\n",
)
```

---

## 9. Output Sanitization & Prompt Injection Control

LCP results can guide the LLM's next conversational response using the `prompt_injection` payload key:

```python
def tool_generate_chart(data_csv: str) -> dict:
    # ... render chart to plot.png ...
    return {
        "success": True,
        "output": "Chart generated as 'plot.png'.",
        "plot_filename": "plot.png",
        "prompt_injection": (
            "\n\n✅ **Chart Rendered:** `plot.png`\n"
            "Reference the chart using `<artefact_image id=\"plot.png::0\" />` "
            "and explain the upward trend to the user."
        ),
    }
```

---

## 10. Best Practices (from the canonical `workspace_tools` library)

1. **CWD-sandbox all paths** with a containment guard — resolve against the workspace and reject escapes:
   ```python
   def _resolve_safe_path(relative_name: str) -> Path:
       candidate = Path(relative_name).resolve()
       root = Path.cwd().resolve()
       if not candidate.is_relative_to(root):
           raise PermissionError(f"Access outside the workspace is forbidden: {relative_name}")
       return candidate
   ```
2. **Coerce LLM-provided values defensively** — LLMs often pass strings for numeric/boolean params:
   ```python
   def tool_find_files(pattern: str, max_results: int = 50):
       if isinstance(max_results, str):
           try: max_results = int(max_results)
           except ValueError: max_results = 50
   ```
3. **Always return the canonical shape**: `{"success": True, "output": ...}` / `{"success": False, "error": ...}` (optionally `traceback` on failure, plus extra payload keys). The binding normalizes anything, but explicit shapes keep LLM behavior predictable.
4. **Write schema-quality docstrings**: the first paragraph is the LLM-facing description; Google-style `Args:` blocks with `name (type): desc` lines give the LLM parameter semantics. This is your only schema — there is no YAML.
5. **Name helpers without the `tool_` prefix** (e.g. `_resolve_safe_path`, `_IGNORED_DIRS`) so they are never exposed to the LLM.
6. **Declare host-injected params explicitly** (`lollms_client_instance=None`, `discussion_instance=None`, `tool_context=None`) only when you need them — they stay hidden from the schema.
7. **Gate risky operations behind the confirm handler** and honor the autonomy level from `init_tools_library` — never assume `full_access`.
8. **Keep libraries cohesive**: one folder per domain, primary module `X/X.py` with the same name as the folder, auxiliary modules as siblings. This maximizes compatibility with `mount_tool_library` and `is_library_mounted`.

---

## 11. Bridging MCP Servers Inside an LCP Tool (LCP → MCP Bridge)

An LCP tool library can embed a full **MCPBinding** (`src/lollms_client/tools_bindings/mcp/__init__.py`, `BindingName = "MCPBinding"`) and proxy calls to any Model Context Protocol server — local (stdio subprocess) or remote (streamable HTTP). MCP tools are namespaced as `server_alias::tool_name` (module constant `TOOL_NAME_SEPARATOR = "::"`), and the binding exposes `execute_tool(tool_name, params, **kwargs) -> Dict` exactly like the LCP contract, so results flow straight through the LCP normalizer.

The MCP server list is host configuration (invisible to the LLM), delivered via `host_tool_configs` and captured in `init_tools_library`:

```python
"""
mcp_bridge.py — LCP library that proxies calls to configured MCP servers.
"""
import json
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "MCP Bridge"
TOOL_LIBRARY_DESC = "Calls tools hosted on configured MCP servers (stdio or HTTP)."
TOOL_LIBRARY_ICON = "🔗"

_mcp_binding = None

def init_tools_library(config: Optional[dict] = None) -> None:
    """Lazily builds the embedded MCPBinding from host-provided server configs."""
    global _mcp_binding
    if _mcp_binding is None and config and isinstance(config, dict):
        from lollms_client.tools_bindings.mcp import MCPBinding
        _mcp_binding = MCPBinding(servers=config.get("servers", []))

def _require_mcp():
    if _mcp_binding is None:
        raise RuntimeError(
            "MCP bridge is not configured. Provide host_tool_configs['mcp_bridge'] "
            "with a 'servers' list."
        )
    return _mcp_binding

def tool_call_mcp_tool(server_alias: str, mcp_tool_name: str, arguments: str = "{}") -> Dict[str, Any]:
    """
    Calls a tool hosted on a configured MCP server and returns its result.

    Args:
        server_alias (str): The alias of the configured MCP server (e.g. 'filesystem').
        mcp_tool_name (str): The name of the tool on that server (e.g. 'read_file').
        arguments (str): A JSON object string with the tool's arguments.
    """
    try:
        payload = json.loads(arguments) if arguments and arguments.strip() else {}
        if not isinstance(payload, dict):
            return {"success": False, "error": "'arguments' must decode to a JSON object."}
    except json.JSONDecodeError as ex:
        return {"success": False, "error": f"Invalid JSON in 'arguments': {ex}"}

    try:
        binding = _require_mcp()
        result = binding.execute_tool(
            f"{server_alias}::{'tool_'}{mcp_tool_name}".replace("tool_tool_", "tool_"),
            payload,
        )
        return {
            "success": bool(result.get("success", not result.get("error"))),
            "output": result.get("output", result),
            "error": result.get("error"),
        }
    except Exception as ex:
        return {"success": False, "error": f"MCP call failed: {ex}"}
```

Host mounting:

```python
client = LollmsClient(
    tools_binding_name="lcp",
    tools_binding_config={
        "tools_folders": ["./my_tools"],
        "host_tool_configs": {
            "mcp_bridge": {
                "servers": [
                    {"alias": "filesystem", "type": "stdio", "command": "npx",
                     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
                    {"alias": "websearch", "type": "http", "url": "http://localhost:8000/mcp"},
                ]
            }
        },
    },
)
```

**Two integration patterns:**
1. **Embedded bridge** (above): the LCP library owns its own `MCPBinding`, built once in `init_tools_library`. Best when the LCP tool set must reach dedicated MCP servers regardless of the host's tool binding.
2. **Reuse the host's binding**: if the host already mounted MCP as its tools binding (`tools_binding_name="mcp"`), an injected `lollms_client_instance` exposes it directly — `client.tools.execute_tool("filesystem::read_file", {"path": "..."})` — letting the LCP tool orchestrate MCP calls with zero extra configuration.

> **Note**: `MCPBinding` requires the `mcp`/`httpx` packages and runs its asyncio event loop on a daemon thread; construction is therefore deferred to `init_tools_library` so the import cost is only paid when the bridge is actually configured.

---

## 12. Full-Capability Tools via Host Injection (LCP Omni-Tools)

A tool that declares `lollms_client_instance=None` in its signature receives the live **`LollmsClient`** (hidden from the LLM schema — see §4.5). The host passes it through `execute_tool(..., lollms_client_instance=client)`. This unlocks the **entire Lollms capability surface inside the tool's internal flow**, before any result is returned to the calling app.

### Verified `LollmsClient` capability surface (each raises `RuntimeError` if the modality is `None` — always guard)

| Client call | Backing attribute | Capability |
| :--- | :--- | :--- |
| `generate_text(prompt=..., n_predict=...)` / `generate(...)` | `client.llm` | Text generation |
| `generate_image(prompt=..., width=..., height=...)` | `client.tti` | Image generation (returns bytes) |
| `edit_image(images=..., prompt=...)` / `generate_omni(...)` | `client.tti` | Image editing / unified omni generation |
| `generate_audio(...)` | `client.tts` | Text-to-speech |
| `transcribe_audio(...)` | `client.stt` | Speech-to-text |
| `generate_video(...)` | `client.ttv` | Text-to-video |
| `generate_music(...)` / `generate_song(...)` / `generate_song_from_lyrics(...)` | `client.ttm` | Music & song generation |
| `query_rag(query, top_k=...)` / `add_document_to_rag(path)` | `client.rag` | Retrieval & knowledge graphs |
| `switch_model(alias)` / `switch_tti(alias)` / `switch_ttm(alias)` / ... | `*_model_profiles_registry` | Hot-swap the active model per modality |

### Example: a multi-modal creative pipeline tool

```python
"""
creative_studio.py — LCP omni-tool that composes image + text + music
using the host's LollmsClient before returning a single unified result.
"""
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "Creative Studio"
TOOL_LIBRARY_DESC = "Generates an illustrated, captioned audio story in one pipeline."
TOOL_LIBRARY_ICON = "🎨"

def tool_create_illustrated_audio_story(
    subject: str,
    style: str = "watercolor",
    with_music: bool = True,
    lollms_client_instance=None,
) -> Dict[str, Any]:
    """
    Creates an illustrated audio story: generates an illustration, writes a poetic
    caption for it, and optionally composes a background music theme.

    Args:
        subject (str): The topic of the story (e.g. 'a lighthouse in a storm').
        style (str): The art style for the illustration (e.g. 'watercolor', 'pixel art').
        with_music (bool): Whether to also generate a background music theme.
    """
    client = lollms_client_instance
    if client is None:
        return {"success": False,
                "error": "This tool requires a host-injected LollmsClient instance."}

    missing = [name for name, attr in
               (("text", client.llm), ("image", client.tti))] if client else []
    if client.llm is None or client.tti is None:
        return {"success": False,
                "error": f"Missing required modality bindings: text (LLM) and image (TTI) must be configured."}

    try:
        # 1. Image generation (TTI) — returns raw image bytes
        image_bytes = client.generate_image(
            prompt=f"{style} illustration of {subject}, highly detailed, dramatic lighting",
            width=1024,
            height=1024,
        )
        image_name = f"story_{uuid.uuid4().hex[:8]}.png"
        Path(image_name).write_bytes(image_bytes)

        # 2. Text generation (LLM) — caption the artwork we just produced
        caption = str(client.generate_text(
            prompt=(
                f"Write a two-sentence poetic caption for a {style} illustration "
                f"of {subject}. Return only the caption text."
            ),
            n_predict=128,
        )).strip()

        # 3. Music generation (TTM) — optional, guarded on binding availability
        music_path = None
        if with_music and client.ttm is not None:
            music_result = client.generate_music(
                prompt=f"Gentle ambient orchestral theme evoking {subject}"
            )
            if isinstance(music_result, (bytes, bytearray)):
                music_path = f"theme_{uuid.uuid4().hex[:8]}.wav"
                Path(music_path).write_bytes(music_result)

        return {
            "success": True,
            "output": (
                f"Created illustrated audio story about '{subject}': "
                f"image '{image_name}'" + (f", music '{music_path}'" if music_path else "")
                + f". Caption: \"{caption}\""
            ),
            "image_filename": image_name,
            "caption": caption,
            "music_filename": music_path,
            "prompt_injection": (
                f"\n\n✅ **Illustrated audio story ready for '{subject}'.** "
                f"Show the artwork using `<artefact_image id=\"{image_name}::0\" />`, "
                f"read the caption \"{caption}\" to the user"
                + (f", and mention the soundtrack `{music_path}`." if music_path else ".")
            ),
        }
    except Exception as ex:
        return {"success": False, "error": f"Creative pipeline failed: {ex}"}
```

### Key rules for omni-tools

1. **Guard every modality**: each `client.generate_*` raises `RuntimeError` when its binding is `None`. Check `client.llm`, `client.tti`, `client.ttm`, ... first and return an actionable `{"success": False, "error": ...}` instead of crashing.
2. **The injection is free-form**: `lollms_client_instance`, `discussion_instance`, and `tool_context` never appear in the LLM-facing schema (§4.5), so the LLM only sees `subject`, `style`, and `with_music`.
3. **Chain modalities internally**: intermediate results (the generated image) can feed the next step (captioning) entirely inside the tool — the calling app only receives the final unified result.
4. **Save binary outputs to the workspace** (CWD-relative paths) and reference them by filename in `output` / `prompt_injection`, so the host UI can mount them as artefacts.
5. **Hot-swapping**: if several models are configured, `client.switch_model(alias)` / `client.switch_tti(alias)` / `client.switch_ttm(alias)` let a tool pick a specialized model (e.g. a fast captioner) mid-flow.

---

## 13. User Interaction Protocol (Asking the User for Input)

Tools can request information directly from the human operator — API keys, free text, or a choice among options — via a host-registered `user_interaction_handler`. The binding propagates it exactly like `confirm_handler`: as a constructor kwarg (`user_interaction_handler=` in `tools_binding_config`), via `LCPBinding.set_user_interaction_handler(handler)`, by injecting the `"user_interaction_handler"` key into every `host_tool_configs` entry, and by calling a module-level `set_user_interaction_handler(handler)` hook at lazy load and on every cached-module execution path.

Request contract (dict in, string or `None` out):

```python
def user_interaction_handler(request: dict) -> Optional[str]:
    # request: {"type": "text" | "secret" | "select",
    #           "title": str, "prompt": str,
    #           "options": List[str], "default": Optional[str]}
    # Returns the user's answer (str), or None when the user cancels.
```

The canonical example library is `default_tools/user_interactive_tools/user_interactive_tools.py`, and the full practical walkthrough lives in the companion skill `lcp_tool_creation`.