# 🧠 LollmsDiscussion: Cognitive Session & Artefact Architecture

This module implements the **Sovereign Discussion Session**, a stateful, thread-safe conversational engine that bridges the gap between transient LLM tokens and permanent, versioned knowledge storage.

It is composed of five orthogonal mixins:
1.  **`CoreMixin`**: Lifecycle, ORM proxy, message CRUD, and thread-safe DB commits.
2.  **`ChatMixin`**: The agentic reasoning loop, tool execution orchestration, and stream parsing.
3.  **`UtilsMixin`**: Branch management, export normalization, and context token auditing.
4.  **`PromptMixin`**: System prompt construction and XML tag post-processing.
5.  **`FileImportMixin`**: Multi-modal ingestion (PDF, DOCX, Data) and Dual-Stream storage.

---

## 🏛️ 1. The Dual-Stream Artefact System (.lam Protocol)

The core innovation of this architecture is **Dual-Stream Storage**. We solve the "Context vs. Tool" paradox by splitting every artefact into two distinct physical and logical streams.

### The Problem
*   **LLMs** need high-level schemas, stats, and descriptions (Logical) to reason effectively without wasting context window on raw binary data.
*   **Tools** (Python, SQL, Executors) need the exact, raw binary or text file (Physical) on disk to execute against.

### The Solution: `.lam` (Logical Artefact Metadata)
When an artefact is created (especially Data or Binary files), the system writes **two** distinct entities:

1.  **Physical Twin (`versions/{title}_v{N}.{ext}` & `workspace/{title}.{ext}`)**
    *   **Content**: Raw bytes (CSV rows, SQLite binary, PNG pixels, Python source).
    *   **Purpose**: Executable by tools. Accessible via simple relative paths.
    *   **Visibility**: Visible in the workspace tree.

2.  **Logical Twin (`versions/{title}_v{N}.lam`)**
    *   **Content**: Markdown schema, statistics, column types, row counts, descriptions.
    *   **Purpose**: Injected into the LLM context window.
    *   **Visibility**: **Hidden** from the workspace tree (stored only in `versions/`).

### Architecture Diagram
```text
data_workspace/
└── discussions/
    └── {discussion_id}/
        ├── versions/
        │   ├── dataset_v1.csv       <-- Physical Twin (Raw Data)
        │   ├── dataset_v1.lam       <-- Logical Twin (Schema/Stats for LLM)
        │   ├── dataset_v2.csv       <-- Physical Twin (Updated)
        │   └── dataset_v2.lam       <-- Logical Twin (Updated Schema)
        └── dataset.csv              <-- Active Workspace Copy (Symlink/Copy of Physical)
```

---

## 🛑 2. Cancellation & Interrupt Protocol

The `ChatMixin` implements a **Thread-Safe Cancellation Protocol** using `threading.Event`. This ensures that long-running agentic loops, heavy tool executions, or streaming generations can be interrupted instantly without leaving the database or workspace in an inconsistent state.

### How It Works
1.  **Signal**: The user (or UI) calls `discussion.cancel_generation()`.
2.  **Propagation**: This sets a internal `_cancel_event` flag.
3.  **Observation**: The agentic loop checks this flag at **four critical safe points**:
    *   **Start of Reasoning Round**: Before sending a new prompt to the LLM.
    *   **During Streaming**: Inside the token streaming callback (every chunk).
    *   **Post-Generation**: Immediately after the LLM finishes but before tool execution starts.
    *   **Tool Cleanup**: During the `finally` block of tool execution (restoring CWD, closing DB connections).
4.  **Graceful Exit**:
    *   The loop breaks immediately.
    *   The current message content is appended with `"[Generation cancelled by user]"`.
    *   Metadata is flagged: `{"cancelled": True, "mode": "cancelled"}`.
    *   The session is committed to DB safely.
    *   The cancellation flag is reset, allowing the next turn to proceed normally.

### API Usage

```python
# 1. Start a long-running generation in a background thread
def run_chat():
    response = discussion.chat(user_message="Analyze this 1GB CSV file...")
    print(response["was_cancelled"]) # Will be True if interrupted

thread = threading.Thread(target=run_chat)
thread.start()

# 2. User clicks "Stop" button
discussion.cancel_generation() 
# Returns True if signal was sent, False if no generation was active

# 3. Check status programmatically
if discussion.is_generation_cancelled():
    print("Generation is stopping...")

# 4. Automatic Reset
# After the chat() method returns, the cancel state is automatically cleared.
# You do NOT need to manually reset it for the next turn.
discussion.chat(user_message="New question...") # Works normally
```

### ⚠️ Critical Safety Guarantees
*   **No Orphaned Tools**: If cancelled *during* tool execution, the system waits for the current tool to finish its `finally` block (to restore CWD and close files) before breaking the loop. It does not kill the process abruptly (which could corrupt files).
*   **DB Integrity**: The partial message is saved to the database with a cancellation marker. You do not lose the conversation history up to that point.
*   **Context Cleanliness**: The system ensures no partial XML tags or broken JSON structures are left in the context window for the next turn.

---

## 🛠️ 3. Developer Guide: Accessing Artefact Data

### A. How to Recover the `.lam` (Logical) Content
The `.lam` content is what the LLM "sees" in its context. It is stored in the `content` field of the artefact dictionary **IF** the artefact was created with `logical_content` or is a Data type.

```python
# Get the artefact record
art = discussion.artefacts.get("my_dataset")

# Access Logical Content (The .lam schema)
logical_schema = art["content"] 
print(logical_schema) 
# Output: "# Data Interface: my_dataset\nFormat: CSV\nColumns: id, name, value..."
```

### B. How to Recover the Physical File (Raw Bytes)
To get the actual file for processing (e.g., to load into Pandas directly without re-parsing), you must access the **Discussion-Isolated Workspace**.

**CRITICAL**: Do not use absolute paths from the artefact dict. Use the discussion's workspace resolver.

```python
from pathlib import Path

# 1. Resolve the Discussion-Specific Workspace
base_ws = Path("./data_workspace")
# If running inside server, APP_WORKSPACE_DIR overrides this
try:
    from lollms_client.app.server import APP_WORKSPACE_DIR
    if APP_WORKSPACE_DIR: base_ws = APP_WORKSPACE_DIR
except ImportError: pass

disc_ws = base_ws / "discussions" / discussion.id

# 2. Construct Path to Physical Twin
file_ext = art.get("file_ext", ".txt")
physical_path = disc_ws / f"{art['title']}{file_ext}"

# 3. Read Raw Bytes
if physical_path.exists():
    raw_bytes = physical_path.read_bytes()
    # OR for text files
    raw_text = physical_path.read_text(encoding="utf-8")
```

### C. The Self-Healing Protocol
The `semantic_data_engineer` and `FileImportMixin` implement **Self-Healing**.
If a tool requests a file (e.g., `data.csv`) that is missing from the `workspace/` folder but exists in the `versions/` folder (or database), the system **automatically restores it** before execution.

*   **Trigger**: `FileNotFoundError` during tool prep.
*   **Action**: Copy `versions/{title}_v{latest}.{ext}` → `workspace/{title}.{ext}`.
*   **Result**: Tools never fail due to missing workspace files if the artefact exists in memory.

---

## 🧬 5. The Chat Loop & Tool Orchestration (`ChatMixin`)

The `chat()` method is not a simple API call; it is an **Agentic State Machine**.

### Execution Flow
1.  **Pre-Hydration**:
    *   Memory Decay & Associative Pull (SQLite).
    *   RAG Injection (if personality has data).
    *   **Dynamic Tool Mounting**: If data files exist in workspace, `semantic_data_engineer` is auto-mounted.
2.  **Context Assembly**:
    *   System Prompt + Rules.
    *   **Active Artefacts**: Injects `.lam` content (Logical Twins) for all active files.
    *   Memory Handles.
3.  **Reasoning Loop** (Max 20 steps):
    *   **LLM Generation**: Streams tokens to `_StreamState`.
    *   **Stream Parsing**: Intercepts closed XML tags (`<artifact>`, `<tool>`) instantly.
    *   **Tool Execution**:
        *   **CWD Switch**: Changes OS Current Working Directory to `data_workspace/discussions/{id}/`.
        *   **Sync**: Ensures all active artifacts exist on disk.
        *   **Run**: Executes Python function.
        *   **Post-Scan**: Detects NEW files created by tool → Auto-registers as Artefacts.
    *   **Feedback**: Sanitizes tool output (strips base64 blobs) and feeds back to LLM.
4.  **Termination**: Commits DB, resets cancellation flags.

---

## 🌿 5. Branching & Versioning

### Message Branching
Messages are not a linear list. They are a **Directed Acyclic Graph (DAG)**.
*   **`active_branch_id`**: Points to the leaf node of the current conversation path.
*   **`get_branch(leaf_id)`**: Recursively walks parents to root, returning a chronological list.
*   **`regenerate_branch`**: Deletes the current AI leaf and restarts the loop from the user parent.

### Artefact Versioning
Every update to an artefact creates a new version (Git-like).
*   **`artefacts.update(..., bump_version=True)`**: Creates v2, v3, etc.
*   **`artefacts.revert(title, target_version=2)`**: Restores old content as a new highest version.
*   **`artefacts.squash_versions(...)`**: Deletes intermediate versions to save space, preserving history in commit messages.
*   **Tags**: Bind semantic labels (e.g., "stable", "gold") to specific versions.

---

## 📥 7. File Import Modes (`FileImportMixin`)

The `import_file` method supports sophisticated ingestion strategies:

| Mode | Behavior | Use Case |
| :--- | :--- | :--- |
| **`text`** | Extracts raw text only. | Code, Logs, Markdown. |
| **`text_images`** | Extracts text + renders pages/images. Anchors images in text via `<artefact_image id="..." />`. | PDFs, DOCX with diagrams. |
| **`images_only`** | Rasterizes everything to images. No text extraction. | Scanned books, complex layouts. |
| **`ocr`** | Renders pages → Sends to Vision LLM → Transcribes text. | Handwritten notes, non-selectable PDFs. |
| **`data`** | **Dual-Stream**. Parses schema (.lam) + saves raw binary (.csv/.db). | Datasets, Spreadsheets. |
| **`data_bundle`** | **Schema Fusion**. Scans a folder, groups files by column signature, fuses them into a single SQLite DB. | Merging 100s of daily CSV reports into one DB. |

### Data Bundle Fusion Logic
When importing a folder as `data_bundle`:
1.  **Scan**: Reads headers of all CSV/XLSX files.
2.  **Fingerprint**: Normalizes column names (lowercase, underscore) and types.
3.  **Group**: Files with identical schemas are merged.
4.  **LLM Naming**: Sends schema sample to LLM to generate a meaningful table name (e.g., `sales_q1_q2_merged`).
5.  **Consolidate**: Writes a single `.db` file with multiple tables.

---

## 🛡️ 7. Security & Integrity Rules

1.  **Path Sovereignty**: Tools **cannot** access files outside `data_workspace/discussions/{id}/`. The CWD switch enforces this at the OS level.
2.  **No Blind Edits**: The Aider patch engine requires verbatim `SEARCH` blocks. It uses 6-pass fuzzy matching (Exact → Whitespace → Indent → Comments → Blanks → Core Delta) to ensure safe edits.
3.  **Binary Stripping**: Tool results containing base64 blobs (>500 chars) are automatically stripped and replaced with `[base64 blob stripped: 24.3KB]` to prevent context explosion and tool loops.
4.  **Prompt Injection**: Tools can return a `prompt_injection` key. This overrides the standard JSON dump and tells the LLM exactly what to say next (e.g., "Here is your plot: ![img](url)").

---

## 🚀 Quick Start: Creating a Custom Tool

Place a Python file in `tools_bindings/lcp/default_tools/`:

```python
# my_tool.py
def tool_analyze_my_data(file_name: str, discussion_instance=None) -> dict:
    """Analyzes a file and logs to memory."""
    # File is available at simple relative path due to CWD switch
    path = Path(file_name) 
    if not path.exists():
        return {"error": "File missing"}
    
    # Do work...
    
    # Auto-sync result as artifact
    if discussion_instance:
        discussion_instance.artefacts.add(
            title="result", 
            artefact_type="document", 
            content="Analysis complete..."
        )
        
    return {"success": True, "prompt_injection": "Analysis done. Tell the user the result."}
```

The LCP Binding will automatically:
1.  Parse the function signature via AST.
2.  Register it as a callable tool.
3.  Set CWD to the discussion workspace before execution.
4.  Sync any new files created back to the artefact system.
