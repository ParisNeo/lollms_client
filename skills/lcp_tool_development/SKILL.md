---
title: "LCP Tool Architecture and Human Intervention Mastery"
description: "Complete guide for authoring LCP tools, configuring AST schema extraction, host configuration injection, and implementing human-in-the-loop validation for CLI and GUI environments."
category: "tool_engineering"
tags: [lcp, tools, human_in_the_loop, security, ast, safe_mode, execution]
visibility: visible
modifiable: true
---

# LCP Tool Architecture & Human-in-the-Loop Engineering

The **LollmsCommunicationProtocol (LCP)** is the lightweight, zero-dependency in-process tool execution engine of `lollms_client`. It discovers, parses, and executes Python tools in a sandboxed workspace environment.

---

## 1. Core LCP Tool Design Doctrine

1. **Agnostic & Decoupled**: LCP tools are plain Python functions. They operate strictly on primitive arguments (`str`, `int`, `float`, `bool`, `list`, `dict`) and return standard dictionaries.
2. **Current Working Directory (CWD) Sandbox**: The orchestrator changes the process CWD to the isolated workspace folder before invoking a tool. Tools resolve workspace files using relative paths (`Path("data.csv")`).
3. **AST Schema Generation**: No JSON schema files are required. LCP parses function names (`tool_*`), type hints, and docstring parameter blocks using Python's `ast` module at startup.

---

## 2. Anatomy of a Multi-Tool LCP Library

Create a `.py` file inside your tools directory (e.g. `tools/database_ops.py`):

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
    """Invoked lazily when the tool is mounted or configured by the host."""
    global AUTONOMY_LEVEL, _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        AUTONOMY_LEVEL = config.get("autonomy_level", "safe")
        _CONFIRM_HANDLER = config.get("confirm_handler")

def tool_database_ops_prompt() -> str:
    """Dynamic prompt injected into system instructions."""
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
            "columns": columns,
            "rows": rows[:50],
            "total_rows": len(rows),
        }
    except Exception as ex:
        return {"success": False, "error": f"SQL Error: {ex}"}
```

---

## 3. Human-in-the-Loop Interrogation & Confirmation Protocol

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
from ascii_colors import ASCIIColors

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

        # 1. Dispatch modal dialog event to UI thread
        event_queue.put({
            "type": "approval_request",
            "source": source,
            "script_label": script_label,
            "argv": argv,
            "response_queue": response_queue,
        })

        # 2. Block worker thread until operator clicks Allow/Reject in the modal
        try:
            decision, reason = response_queue.get(timeout=300)
            return decision, reason
        except Exception:
            return "reject", "Approval request timed out."

    return _handler
```

---

## 4. Mounting and Registering LCP Tools

### Autonomy Levels Matrix
LCP tools support three distinct autonomy modes configured via `host_tool_configs`:

| Level | Behavior | Prompt Trigger |
| :--- | :--- | :--- |
| **`strict`** | Maximum oversight. Prompts the operator for authorization on **every** Python execution and shell command. | Every execution |
| **`safe`** (Default) | Intelligent balance. Automatically executes benign code, math, data science (pandas, numpy), document generation (docx, pptx, pdf, xlsx), plotting (matplotlib, seaborn), and workspace I/O. | Prompts ONLY when risky operations are detected (process spawning via `subprocess`/`multiprocessing`, `os.system`/`os.popen`, shell scripting escapes) |
| **`full_access`** | Unrestricted autonomy. Runs all scripts and shell commands without confirmation prompts or whitelist checks. | Never prompts |

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
    shell_autonomy_level="safe",  # Runs PPTX generation and matplotlib plots autonomously
)
```

---

## 5. Output Sanitization & Prompt Injection Control

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