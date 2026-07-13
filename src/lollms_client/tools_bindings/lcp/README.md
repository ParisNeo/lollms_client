# LCP: LollmsCommunicationProtocol Tool Binding

LCP is a lightweight, zero-dependency local tool execution framework for Lollms. It allows the LLM to discover and execute custom Python scripts directly in-process, without needing to run or maintain external servers.

## 🚀 Core Features

### 1. Multi-Tool File Architecture
**One File, Many Tools.** You no longer need one file per tool. LCP now scans every function starting with `tool_` inside a Python file and registers each as an independent, callable tool with its own signature.
*   **Example:** A single `semantic_data_engineer.py` file can expose 15+ distinct macros (`tool_get_schema`, `tool_filter_data`, `tool_plot`, etc.).
*   **Benefit:** Keeps related tool libraries cohesive and organized in a single module.

### 2. Dynamic Library Mounting
**Context-Aware Tool Loading.** Tools are no longer static. The system can dynamically mount specialized tool libraries at runtime based on workspace context.
*   **Auto-Discovery:** If a discussion contains data files (`.csv`, `.db`), the system automatically mounts the `semantic_data_engineer` library without user intervention.
*   **Zero Config:** No need to manually select a "Data Personality." The tools appear automatically when needed.

### 3. Discussion-Isolated Workspaces
**Secure, Sandbox-Per-Chat.** Every discussion gets its own isolated workspace folder (`data_workspace/discussions/{discussion_id}/`).
*   **Automatic Sync:** When a tool creates a file, it is instantly synced as an Artifact in that specific discussion.
*   **No Path Conflicts:** Tools operate on simple relative paths (e.g., `data.csv`) because the CWD is automatically set to the discussion's folder before execution.
*   **Binary Safety:** The system intelligently handles binary files (images, databases) by creating placeholder artifacts with download links, preventing context bloat.

### 4. AST-Based Schema Ingestion
**No Duplicate JSON Schemas.** LCP uses Python's `ast` module to automatically extract tool names, parameters, type annotations, defaults, and descriptions directly from your Python function and its docstring on-the-fly.

### 5. Multi-Source Discovery
*   **`tools_folders`**: Scan multiple local directories simultaneously.
*   **`tool_files`**: Import standalone Python tool files directly from anywhere on disk.

### 6. Context Awareness
Tools can optionally receive the active `LollmsClient` instance and `LollmsDiscussion` session state directly via keyword arguments (`lollms_client_instance`, `discussion_instance`).

### 7. Dynamic Tool Generation from Artefacts
**LLM-Authored Tools.** LCP integrates seamlessly with the Artefact system. If the LLM generates a `type="tool"` artefact, LCP can dynamically compile and register it in memory.
*   **`register_tool_from_code(tool_name_prefix, code)`**: Executes raw Python code in an isolated module namespace, extracts `tool_*` functions via AST, and registers them as active tools.
*   **`unregister_tools_by_prefix(tool_name_prefix)`**: Cleanly removes dynamically generated tools when the artefact is updated or deleted.
*   **Security Gate**: This feature is disabled by default. The host application must explicitly pass `allow_dynamic_tools=True` to `discussion.chat()` to permit the LLM to execute its own code.

---

## 🛠️ How to Write an LCP Tool

To create a tool, simply write a Python file and place it inside your LCP tools directory (e.g., `src/lollms_client/tools_bindings/lcp/default_tools/`).

### Example 1: Multi-Tool Library (Recommended)
Group related functions in one file. LCP will register **both** `tool_calculate_bmi` and `tool_calculate_tdee` as separate tools.

```python
def tool_calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """
    Calculates Body Mass Index (BMI).

    Args:
        weight_kg (float): Weight in kilograms.
        height_m (float): Height in meters.
    """
    bmi = weight_kg / (height_m ** 2)
    category = "Normal"
    if bmi < 18.5: category = "Underweight"
    elif bmi >= 25: category = "Overweight"
    
    return {"bmi": round(bmi, 2), "category": category}

def tool_calculate_tdee(bmr: float, activity_level: str = "sedentary") -> dict:
    """
    Calculates Total Daily Energy Expenditure (TDEE).

    Args:
        bmr (float): Basal Metabolic Rate.
        activity_level (str, optional): Activity level. Defaults to 'sedentary'.
    """
    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    mult = multipliers.get(activity_level, 1.2)
    return {"tdee": round(bmr * mult, 2)}
```

### 🛑 CRITICAL: LCP Tool Agnosticism Doctrine (Relaxed)

**LCP tools are strictly agnostic to Lollms by default.** They must **NEVER** accept `lollms_client_instance`, `discussion_instance`, or any other internal Lollms state object as input parameters unless explicitly required for advanced agentic patterns.

Tools are designed to be standalone Unix-style utilities. They must operate purely on parameters that an LLM can naturally generate (strings, numbers, lists) and interact with the filesystem using relative paths.

1.  **Optional Context Injection**: By default, do not add `discussion_instance` or `lollms_client_instance` to your tool's function signature. However, if a tool requires access to the orchestrator (e.g., for spawning child agents), it may declare these parameters. The ChatMixin orchestrator will inject them, and the LCP AST parser will filter them out when building the JSON schema for the LLM.
2.  **CWD Reliance**: The Lollms orchestrator (ChatMixin) guarantees that the Current Working Directory (CWD) is set to the isolated discussion workspace *before* the tool executes. Tools should resolve files using simple relative paths: `Path(file_name)`.
3.  **Output-Only Communication**: Tools communicate results back by returning a dictionary. They should never attempt to directly manipulate the discussion database, commit artifacts, or access the client (unless injected for advanced patterns).

### 🛡️ Error Tracking & Tracebacks

The `LCPBinding` implements a comprehensive error tracking system to aid debugging. When a tool fails, the binding captures the full Python stack trace and includes it in the returned dictionary.

1.  **Explicit Tool Failures**: If a tool returns `{"success": False, "error": "..."}`, the binding captures the current stack trace and injects it as a `"traceback"` key in the returned dictionary.
2.  **Unexpected Crashes**: If the tool function raises an unhandled `Exception`, the binding catches it, formats the full traceback via `traceback.format_exc()`, and returns:
    ```python
    {
        "error": "Error executing 'tool_name': <str(exception)>",
        "traceback": "<full_stack_trace>",
        "status_code": 500
    }
    ```
3.  **Orchestrator Visibility**: The caller (ChatMixin) receives this dictionary and can log the traceback or display it to the user, ensuring that silent failures never occur.

### Example 2: Standard Agnostic Tool
Tools operate on files in the current working directory. The orchestrator handles artifact registration and database commits.

```python
from pathlib import Path

def tool_file_analyzer(file_name: str) -> dict:
    """
    Analyzes a file and returns its statistics.

    Args:
        file_name (str): Name of the file to inspect in the workspace.
    """
    # The binding automatically sets CWD to the discussion workspace.
    # No discussion_instance is needed or allowed.
    path = Path(file_name)
    
    if not path.exists():
        return {"error": f"File '{file_name}' not found."}
    
    content = path.read_text(errors="ignore")
        
    return {
        "status": "success",
        "output": f"File contains {len(content)} characters."
    }
```

### Example 3: Prompt Injection (Controlling the LLM)
To prevent the LLM from hallucinating next steps after a tool runs, return a `prompt_injection` key. This text is fed directly to the LLM as the tool's "voice".

```python
def tool_generate_plot(data_file: str) -> dict:
    """Generates a plot and saves it."""
    # ... plotting logic ...
    plot_url = "/api/workspace_files/plot_123.png"
    
    return {
        "success": True,
        "output": "Plot generated.",
        "prompt_injection": f"\n\n✅ **Plot Generated Successfully!**\nHere is your chart: \n\n![Plot]({plot_url})\n\nThe plot shows a clear upward trend. You should now explain this trend to the user."
    }
```

---

## 🔧 Configuration

Configure the LCP binding inside your client parameters:

```python
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gemma4:e2b"},
    tools_binding_name="lcp",
    tools_binding_config={
        # Scan multiple local folders
        "tools_folders": [
            "./my_custom_tools_directory",
            "C:/shared_network_tools/lcp_library"
        ],
        # Or map standalone files directly from anywhere on disk
        "tool_files": [
            "C:/projects/utilities/matter_lock_controller.py"
        ]
    }
)
```

---

## 📂 File Organization

### Flat Structure (Legacy/Simple)
```text
default_tools/
├── weather.py          # Exposes: tool_get_weather
├── calculator.py       # Exposes: tool_add, tool_subtract
```

### Library Structure (Recommended)
```text
default_tools/
├── semantic_data_engineer/
│   └── semantic_data_engineer.py  # Exposes: tool_get_schema, tool_filter, tool_plot, etc.
├── matter_controller/
│   └── matter_controller.py       # Exposes: tool_discover, tool_commission, tool_control
```

---

## 🛡️ Execution Lifecycle

1.  **Context Detection:** The chat layer detects data files in the workspace.
2.  **Dynamic Mounting:** `LCPBinding.mount_tool_library("semantic_data_engineer")` is called automatically.
3.  **AST Parsing:** LCP scans the file, finds all `tool_*` functions, and registers them.
4.  **Invocation:** User asks "Filter the data". LLM calls `tool_filter_and_slice_data`.
5.  **Workspace Sync:**
    *   LCP changes CWD to `data_workspace/discussions/{id}/`.
    *   Tool executes and creates `filtered_data.csv`.
    *   LCP detects the new file and syncs it as an Artifact.
6.  **Response:** The tool returns `prompt_injection`, instructing the LLM: "Here is the filtered CSV. Reference it in your answer."
7.  **Final Answer:** LLM presents the result to the user seamlessly.

### Dynamic Tool Lifecycle (LLM-Authored Tools)
1.  **Generation:** LLM outputs `<artifact type="tool" name="my_tool">def tool_run(): ...</artifact>`.
2.  **Security Gate:** `ArtefactManager` checks `discussion.allow_dynamic_tools`. If `False`, the process stops here (file is saved but not executed).
3.  **Registration:** If `True`, `LCPBinding.register_tool_from_code("my_tool", code)` is called. The code is executed in an isolated module namespace.
4.  **Invocation:** The LLM can immediately call `<tool>{"name": "tool_my_tool", "parameters": {...}}</tool>`.
5.  **Cleanup:** If the artefact is updated or deleted, `LCPBinding.unregister_tools_by_prefix("my_tool")` is called to remove the old executable function.

---

## 🔄 Cross-Discussion Tool Portability (.laa / .lab)

Because LCP tools are stored as standard Python artefacts inside the discussion workspace, they fully benefit from the **Decoupled Artefact Protocol**. You can export a tool created in one discussion and import it into another, or share it via the global library.

### Exporting a Tool
If an LLM creates a specialized tool (e.g., `my_analyzer.py`) in Discussion A, you can export it:

```python
# Export as a standalone .laa file (preserves version history)
discussion_a.artefacts.export_artefact_to_archive(
    title="my_analyzer.py", 
    output_path="my_analyzer.laa"
)

# Or save directly to the global library
discussion_a.save_artefact_to_global_archive("my_analyzer.py")
```

### Importing a Tool
In Discussion B, you can import the tool. Once the physical file is reconstructed on disk, the LCP binding can discover and register it.

```python
# Import from a .laa file
discussion_b.artefacts.import_artefact_from_archive("my_analyzer.laa")

# Or load from the global library
discussion_b.load_artefact_from_global_archive("my_analyzer.py")

# The tool is now physically present in Discussion B's workspace.
# If the LCP binding is configured to scan this workspace, it will auto-discover the tool.
```

### Bundling Tool Ecosystems
If a tool requires multiple files (e.g., a main script and a helper module), you can bundle them using the `.lab` format, which preserves the relative folder structure.

```python
# Export a bundle of tool files
discussion_a.artefacts.export_artefact_bundle(
    paths=["workspace_data/my_tool.py", "workspace_data/utils.py"],
    output_path="my_tool_ecosystem.lab"
)

# Import the bundle into a new discussion
discussion_b.artefacts.import_artefact_bundle("my_tool_ecosystem.lab")
```
### 8. Context Visibility & Physical Paths

**Path Sovereignty**: The LLM does not see flat filenames; it sees the exact relative path of the artifact from the workspace root (e.g., `path/to/artefact.py`). This ensures that when the LLM writes Python code to read a CSV or import a module, it uses the correct relative path, preventing `FileNotFoundError`.

**Multi-Tier Visibility**: To prevent context bloat, the LLM only sees files that are explicitly unlocked. The system presents a directory tree index to the LLM:
```text
## artefacts list
path/to/artefact.py[F]
path/to/artefact2.py[L]
path/to/file.md[F]
path/to/file.csv[M]

## Full artefacts content
```python:path/to/artefact.py
# here is the content
```

```markdown:path/to/file.md
here is the content
``` 
```markdown:path/to/file.csv
here is metadata infos about the file
``` 
```
The LLM can dynamically request to load (`[U]` -> `[F]`) or lock (`[F]` -> `[L]`) files using `<unlock_file>` and `<lock_file>` tags.
