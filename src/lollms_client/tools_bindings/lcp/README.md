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

### Example 2: Scoped Context Injection
If your tool needs to access active session details or write artifacts:

```python
from typing import Optional, Any
from pathlib import Path

def tool_file_analyzer(
    file_name: str,
    lollms_client_instance: Optional[Any] = None,
    discussion_instance: Optional[Any] = None
) -> dict:
    """
    Analyzes a file and logs details back to the active conversation.

    Args:
        file_name (str): Path or name of the file to inspect.
    """
    # The binding automatically sets CWD to the discussion workspace
    path = Path(file_name)
    
    if not path.exists():
        return {"error": f"File '{file_name}' not found in workspace."}
    
    # Example: Create a new artifact in the current discussion
    if discussion_instance:
        content = path.read_text(errors="ignore")
        discussion_instance.artefacts.add(
            title=f"analysis_{file_name}",
            artefact_type="document",
            content=content[:2000] # Preview only
        )
        discussion_instance.commit()
        
    return {"status": f"Analysis complete for {file_name}."}
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
