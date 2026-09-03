# 📦 lollms_artefact: Dynamic Artefact & Context Subsystem

The `lollms_artefact` package implements the core versioning, lifecycle management, and file-tracking layers for Lollms. It introduces a **Git-like Filesystem-as-Source-of-Truth Architecture** designed to bridge the gap between large physical data files on disk and the context window limitations of Large Language Models.

---

## 🏛️ 1. The Git-like Storage Philosophy

Conversational AI agents often experience a conflict between model attention and tool execution:
- **Language Models** require high-level summaries, database schemas, and text content to plan queries and reason logically without wasting thousands of tokens on raw binary data.
- **Local Tools** (Python scripts, SQL queries, executors) require the exact, unmodified physical bytes to perform computations and write outputs.

To solve this, the subsystem uses a **Git-like Filesystem-as-Source-of-Truth** approach:

### A. Workspace Root & Confinement (Sandbox)
All artefacts are strictly confined to a workspace root directory (e.g., `data_workspace/{discussion_id}/workspace_data/`). The `_resolve_confined_path()` method guarantees that no path traversal (`..`) or absolute paths can escape this sandbox. The LLM and tools operate entirely within this box using relative paths.

### B. Active Files & Version Snapshots
* **Active File**: The current, working version of a file is written directly to the root of the workspace folder (e.g., `workspace_data/main.py`).
* **`.versions/` Directory**: A hidden `.versions/` folder inside the workspace root stores historical snapshots. When an artefact is updated, the previous version is archived as `workspace_data/.versions/{uuid}/main_v1.py`. This provides a robust, Git-like local history without relying solely on database blobs.

### C. Single-Stream vs Dual-Stream (`.lam` Protocol)
* **Single-Stream (Text, Code, Documents)**: For text-based files (`.py`, `.md`, `.pdf`, `.docx`, `.txt`), the system reads directly from the active file in the workspace root when building the LLM context. The database acts as a lightweight index (`content_source: "disk"`) and does not store the raw text to prevent memory bloat.
* **Dual-Stream (`.lam` Protocol for Binary & Structured Data)**: For structured data files (`.csv`, `.db`, `.sqlite`, `.xlsx`), the system uses a Dual-Stream approach:
  * **Physical Twin**: Saved at `workspace_data/{title}.{ext}`. Contains the raw bytes (e.g., raw CSV rows, SQLite binary). Consumed by local tools.
  * **Logical Twin (`.lam`)**: Saved inside `workspace_data/.versions/{id}/{name}.lam`. Contains a high-density, text-based abstraction of the file's structure (column names, inferred data types, sample values). Consumed by the LLM context zone.

### D. Filesystem Synchronization
If a file is manually deleted from the workspace folder, the `_sync_index_with_disk()` method detects the orphaned database record during the next synchronization cycle and purges it. The workspace folder is the single source of truth.

---

## 👁️ 2. Multi-Tier Visibility Control

To maintain clean and token-efficient context budgets, every registered artifact is assigned a visibility tier that determines how it is represented in the prompt:

| Visibility Tier | Symbol | Prompt Context Behavior |
| :--- | :--- | :--- |
| **`FULL`** | `[C]` | The content (for text) or `.lam` schema (for data) is fully injected verbatim into the active context zone. |
| **`METADATA`** | `[M]` | Only the basic metadata (such as filename, size, and type) is injected, withholding the full schema description. |
| **`TREE_UNLOCKABLE`**| `[U]` | The file is listed only in the directory index. It is excluded from the active context but can be loaded dynamically. |
| **`LOCKED`** | `[L]` | The file is completely excluded from the conversation context and cannot be loaded. |
| **`HIDDEN`** | — | The artifact is completely excluded from both the directory index and the context. |

The LLM can dynamically promote any `[U]` file into its working memory by outputting the file-loading tag:
```xml
<add_files_to_context>
filename.ext
</add_files_to_context>
```

---

## 🧬 3. Integration with LollmsDiscussion

The `ArtefactManager` interacts directly with `LollmsDiscussion` to orchestrate state updates:

```
        LollmsDiscussion (Session State)
              │
              ├──> ArtefactManager
              │         │
              │         ├──> [SQLite Metadata Record] (Maintains lightweight index & version logs)
              │         │
              │         └──> [Physical Workspace] (Writes active files & archives snapshots in .versions/)
              │
              └──> ChatMixin (Orchestrates tool execution & scans CWD)
```

### A. Automated File-Tracking and Ingestion

**Default Visibility Doctrine**: To prevent context window bloat, all newly discovered or tool-generated files are registered with `TREE_UNLOCKABLE` visibility by default. The LLM must explicitly unlock a file to load its content into the active context.

During local tool execution, the active directory is snapshotted immediately before and after the run. If a tool writes a new file (such as a Matplotlib chart PNG) or modifies an existing dataset:
1. The new file is automatically detected on disk.
2. Its file type is classified, and the raw bytes are saved as a physical twin.
3. A logical twin (`.lam`) or image reference is compiled (if applicable).
4. The artifact is committed to the database, incrementing its version.
5. The corresponding reference tags are appended to the conversational message stream.

### B. Self-Healing and Recovery
If a tool or script requests a physical file that is missing from the active `workspace_data/` folder, the manager intercepts the failure, queries the database version log, and restores the exact versioned physical bytes back to the disk folder automatically before the execution begins.

### C. Live Rendering Tags
The chat interface interprets custom tags inserted into the message history. The parser uses a flexible regex that supports both `<artifact>` and `<artefact>` spellings:
* `<artifact type="atype" name="title" version="N" />`: Renders an interactive file card in the chat bubble allowing the user to view or download the file.
* `<artefact_image id="title::N" />`: Directs the chat bubble to render the decoded base64 image pixels inline (e.g., showing a generated plot directly in the conversation).
* `<revert_artifact name="title" version="N" />`: Reverts the specified artifact to the requested version and updates the UI.

---

## ✏️ 4. Updating Artefact Content

You can modify the content of an existing artefact using the `update()` method. By default, this creates a new version snapshot in the `.versions/` directory and updates the active file in the workspace root.

```python
art = discussion.artefacts.update(
    title="analysis_script.py",
    new_content="import pandas as pd\nprint('new code')",
    commit_message="Refactored import logic"
)
```

### Overwriting the Current Version

If you want to update the content without creating a new version history entry, you can set `bump_version=False`. This will overwrite the content of the current active file without incrementing the version number.

```python
art = discussion.artefacts.update(
    title="temp_notes.md",
    new_content="Updated temporary notes without version bump.",
    bump_version=False
)
```

### Disabling Versioning Globally (Git-Style Mode)

For autonomous agent workflows operating on deep folders where version control is handled externally (e.g., via Git), you can disable versioning entirely. Set the `disable_artefact_versioning` attribute on the discussion object (or proxy) to `True`. When this flag is active, the `add()` method will **overwrite the existing version in-place** instead of appending a new row and deactivating the old ones. The version number will remain at `1`.

```python
# On a LollmsDiscussion instance:
discussion.disable_artefact_versioning = True

# Or on an Agent's internal proxy:
agent = Agent(
    lc=client,
    personality=personality,
    workspace_path="./workspace",
    enable_artefact_system=True,
    disable_artefact_versioning=True
)
```

### The Disk-Source Strategy (Non-Versioned Mode)

When `disable_artefact_versioning=True` is active, the `ArtefactManager` employs a **Disk-Source Strategy** to prevent memory duplication and bloat:

1. **Lightweight DB Index**: The `add()` method sets `content_source: "disk"` on the artefact record and **stops storing content in the DB metadata** (`art["content"] = ""`).
2. **Filesystem as Source of Truth**: The physical file on disk becomes the single source of truth for the artefact's content.
3. **On-the-Fly Hydration**: When `build_artefacts_context_zone()` is called, it detects the `content_source: "disk"` flag and reads content directly from the physical file via a `_read_content_from_disk()` helper.

This architecture transforms the database into a lightweight relational index (tracking visibility, status, and physical paths) while delegating blob storage entirely to the filesystem, significantly reducing memory usage for long-running autonomous agents.

### The Content-First Update Doctrine (DATA Artifacts)
When updating a `DATA` artifact (like a CSV or Excel file) where the raw bytes (`physical_data`) are not explicitly provided in the function call, the `ArtefactManager` enforces a strict **Content-First Update Doctrine**:
1. If the new string content differs from the logical schema, the manager assumes the raw data itself is being updated. It encodes the new string to UTF-8 bytes and writes it as the new physical twin.
2. If the artifact is a binary database (`.db`, `.sqlite`) and raw bytes are missing, the manager **refuses** to encode the string schema as binary data (which would corrupt the database header). Instead, it automatically rehydrates the physical bytes from the existing file on disk before applying the schema update.

---

## ⚠️ 5. Import Conflict Resolution

When importing files into the artefact system, there is a possibility of title collisions (e.g., importing `README.md` from two different sources). The `import_file` method provides an `on_conflict` parameter to define the resolution strategy.

### Strategies

1. **`suffix` (Default)**
   - **Behavior**: If an artifact with the target title already exists, the new file is renamed with an incrementing suffix (e.g., `README_1.md`, `README_2.md`).
   - **Use Case**: Preserving all imported files without losing any data or altering the original artifact.
   - **Result**: Creates a new artifact with the suffixed title. The original artifact remains untouched.

2. **`version`**
   - **Behavior**: The existing artifact is updated, and its version number is incremented. The physical file is overwritten with the new content, but the previous version is preserved in the database history.
   - **Use Case**: Importing an updated version of a file where you want to maintain a clear audit trail of changes.
   - **Result**: Updates the existing artifact and bumps the version (e.g., v1 → v2).

3. **`overwrite`**
   - **Behavior**: The existing artifact's content is replaced with the new content, but the version number is **not** incremented. Previous version history is preserved, but the active version is silently replaced.
   - **Use Case**: Correcting or silently updating a file without polluting the version history.
   - **Result**: Updates the existing artifact. The version number remains the same.

4. **`replace`**
   - **Behavior**: Completely purges all existing versions and history of the artifact, then creates a fresh `v1` baseline with the new content.
   - **Use Case**: Starting over cleanly when the previous iterations are no longer relevant or were imported in error.
   - **Result**: Deletes all previous database records and physical metadata, then creates a new `v1` artifact.

### Example

```python
# Import a file, replacing any existing artifact with the same name completely
discussion.import_file(
    path="path/to/new/README.md",
    mode="text",
    title="README.md",
    on_conflict="replace"
)
```

---

## 🛠️ 6. Dynamic Tool Artefacts (`type="tool"`)

The artefact system natively supports the LLM generating its own executable tools. When the LLM creates an artefact with `type="tool"`, the `ArtefactManager` attempts to register it dynamically.

### Security Gate
To prevent untrusted LLMs from executing arbitrary code, dynamic tool registration is gated by the `allow_dynamic_tools` flag on the active `LollmsDiscussion` instance.
*   If `allow_dynamic_tools` is `False` (the default), the tool artefact is saved as a standard code file but is **NOT** parsed or executed.
*   If `allow_dynamic_tools` is `True`, the manager extracts the Python code and passes it to the active `LCPBinding` for immediate AST parsing and module execution.

### Lifecycle
1.  **Create**: LLM outputs `<artifact type="tool" name="my_tool">def tool_run(): ...</artifact>`.
2.  **Gate Check**: `ArtefactManager._register_tool_artefact()` checks `discussion.allow_dynamic_tools`.
3.  **Register**: If allowed, `LCPBinding.register_tool_from_code("my_tool", code)` is called.
4.  **Execute**: The tool `tool_my_tool` is now available in the active session registry.

---

## 🧩 7. Artefact Properties Reference & Handling Guide

Every artifact in the system is represented as a dictionary (record) with a specific set of keys. Understanding the distinction between these properties is critical for correctly creating, updating, and referencing artifacts, especially when dealing with the Dual-Stream storage architecture.

### Core Properties

| Property | Type | Description & Handling Rules |
| :--- | :--- | :--- |
| `title` | `str` | **The primary logical key.** This is the high-level metadata name used by the LLM and the database to reference the artifact. It may contain subfolder paths (e.g., `My_subfolder/SKILL.md`). It is sanitized via `_sanitize_path_segments` to ensure cross-platform safety. |
| `physical_path` | `str` | **The unique disk location & retrieval key.** This stores the exact relative path (including subfolders and extension) where the physical twin resides in `workspace_data/`. If not explicitly provided during `add()` or `update()`, it defaults to the `title`. **CRITICAL**: The `get()` and `remove()` methods prioritize matching by `physical_path` over `title` to prevent ambiguity when multiple artifacts share similar titles. File-reading tools should be passed the `physical_path` to ensure they open the correct file on disk. |
| `type` | `str` | The category of the artifact (e.g., `ArtefactType.CODE`, `ArtefactType.DATA`). Determines how the artifact is rendered in the context zone and which tools can operate on it. |
| `content` | `str` | The logical text content. For text/code files, this is the verbatim source code. For `DATA` artifacts, this holds the `.lam` schema description, **NOT** the raw binary bytes. |
| `version` | `int` | The version number. Incremented automatically on `update()` if `bump_version=True`. The `get()` method returns the highest version by default. |
| `visibility` | `str` | The context tier (`FULL`, `TREE_UNLOCKABLE`, `METADATA`, `TREE_LOCKED`, `HIDDEN`). Controls how the artifact appears in the LLM's prompt. See [Section 2: Multi-Tier Visibility Control](#-2-multi-tier-visibility-control). |
| `active` | `bool` | A legacy boolean flag that mirrors `visibility == FULL`. It is `True` if the artifact is fully loaded in context, `False` otherwise. |
| `language` | `str` | The programming or markup language (e.g., `python`, `html`). Used for syntax highlighting in the context zone and to infer file extensions. |
| `file_ext` | `str` | The explicit file extension (e.g., `.csv`, `.db`). **CRITICAL for DATA artifacts**: This determines how the physical file is written to disk and prevents binary corruption. |
| `logical_content` | `str` | Explicit storage for the `.lam` schema text of `DATA` artifacts. While usually mirrored in `content`, this field is the authoritative source for the logical twin during Dual-Stream sync operations. |
| `physical_data` | `bytes` | The raw binary bytes of a `DATA` or `IMAGE` artifact. **CRITICAL**: This field is stripped from the database record by `_get_all_raw()` to prevent JSON serialization crashes. It is only present in the dictionary returned directly by `add()` or `update()`. Never assume `art.get("physical_data")` will return bytes from a database query; rehydrate from disk if needed. |
| `token_count` | `int` | The estimated token count of the `content`. Used by the Context Budget Guard to prevent context overflow. |
| `content_source` | `str` | Indicates where the content is stored. `disk` means the database is acting as an index and the filesystem is the source of truth. `db` means the content is stored in the database metadata. |

### Handling Guidelines

#### 1. Title vs. Physical Path Decoupling & Mutually Exclusive Retrieval
The architecture decouples the logical database key (`title`) from the unique disk location (`physical_path`). To enforce explicit intent and prevent ambiguity, the `ArtefactManager.get()` method accepts `title` and `physical_path` as **mutually exclusive** parameters.
*   **Creation**: If you create an artifact with `title="My_subfolder/script.py"`, the `physical_path` automatically mirrors this. The physical file is written to `workspace_data/My_subfolder/script.py`.
*   **Context Injection**: `build_artefacts_context_zone()` displays the `physical_path` to the LLM. When the LLM decides to read a file, it should use this exact string.
*   **Updating**: If you change the `title` during an update (`new_title`), the `physical_path` is updated to match, and the old physical file is deleted from disk.
*   **Retrieval (`get`)**: You may query an artifact by passing either `title="README.md"` OR `physical_path="My_subfolder/README.md"`, but not both. Passing both raises a `ValueError`.
*   **Deletion (`remove`)**: The `remove()` method uses the `physical_path` to calculate the exact metadata directory UUID and purge all physical files (active, versioned, and `.lam` twins) from disk. This guarantees that deleting an artifact removes all traces, preventing the workspace heal scan from resurrecting orphaned files as inactive artifacts.

#### 2. Data Artifact Safety (Binary Corruption Prevention)
`DATA` artifacts (like SQLite databases or CSVs) use the Dual-Stream protocol.
*   **Never write string `content` to a binary file**: The `_sync_to_disk_workspace` method explicitly refuses to write string `content` to `.db`/`.sqlite` files if `physical_data` is missing. This prevents the database header from being overwritten with `.lam` schema text.
*   **Rehydration**: When updating a `DATA` artifact's schema, the `update()` method automatically rehydrates `physical_data` by reading the existing bytes from disk *before* calling `add()`. This ensures the raw binary data is preserved across schema updates.

#### 3. Visibility and Context Budget
*   **Tool-Generated Files**: By default, tool-generated files >100KB are registered with `visibility=TREE_UNLOCKABLE` and `active=False` to prevent context bloat.
*   **Unlocking**: The LLM can use `<add_files_to_context>` to promote a file to `FULL` visibility. However, the Context Budget Guard blocks unlocking files >50,000 tokens, instructing the LLM to use tools (SQL, grep) instead.

#### 4. Image Artifacts
*   Image artifacts store base64 encoded strings in the `images` list and their MIME types in `image_media_types`.
*   They are an exception to the visibility doctrine: they are always registered with `visibility=FULL` and `active=True` when generated by tools, so they can be hydrated into the LLM's vision context immediately.

---

## 🗄️ 8. Artefact Archiving (Export & Import)

The `ArtefactManager` provides robust utilities to export and import artifacts as portable zip archives. This is essential for transferring complex multi-file applications or version histories between different discussions or projects.

### A. Single Artefact Archives (`.laa`)
The `.laa` (Lollms Artefact Archive) format bundles all versions, content, physical bytes, and metadata of a *single* artifact title into a zip file.

*   **Export**: `export_artefact_to_archive(title, output_path)` creates a `.laa` file containing a `manifest.json` and separate files for each version's text content (`vN_content.txt`) and raw binary data (`vN_physical.bin`).
*   **Import**: `import_artefact_from_archive(laa_path, activate)` extracts the archive, purges any existing artifact with the same title, and restores the full version history. If `activate=True`, the latest imported version is activated immediately.

### B. Artefact Bundles (`.lab`)
The `.lab` (Lollms Artefact Bundle) format allows exporting and importing *multiple* files or entire directories at once.

*   **Export**: `export_artefact_bundle(paths, output_path, include_versions)` takes a list of file/directory paths from the `workspace_data` directory and zips them together. If `include_versions=True`, it also bundles the historical version files inside a `_versions/` directory.
*   **Import**: `import_artefact_bundle(lab_path, activate)` extracts the `.lab` archive directly into the active `workspace_data` directory. It automatically classifies each file by extension (e.g., `.py` -> `CODE`, `.csv` -> `DATA`), reads text or physical bytes accordingly, and registers them as new artifacts. Binary files (like `.db` or `.png`) are read as raw bytes, while text files are injected directly into the artifact's `content`.

### C. JSON Export/Import
For lightweight, text-only integrations (such as passing an artifact via an API payload), the manager also supports standard JSON serialization:
*   `export_artefact(title)`: Returns a JSON-serializable dictionary containing all versions and companion image versions.
*   `import_artefact(artefact_data, activate)`: Reconstructs the artifact and its companion images from a JSON dictionary.
*   `export_artefact_bundle(title)`: A legacy JSON-based single-artifact export that includes companion image artifacts.

---

## 🗑️ 10. Configurable Versioning Strategy for Deleted Artifacts

By default, the `ArtefactManager` enforces a strict **Disk-Source-of-Truth** invariant. When an artifact is removed via `remove()`, both the active file in the workspace root and its entire version history in the `.versions/` directory are permanently purged. This ensures the filesystem never contains orphaned files that lack database backing.

However, for workflows requiring data recovery or audit trails, this behavior can be softened using the `keep_deleted_versions` flag.

### Configuration
To enable version preservation, set the `keep_deleted_versions` attribute to `True` on your `LollmsDiscussion` instance before deleting artifacts:

```python
# Enable preservation of version history on deletion
discussion.keep_deleted_versions = True

# Deleting the artifact removes the active file but preserves .versions/
discussion.artefacts.remove("critical_script.py")
```

### Restoration Tools
When `keep_deleted_versions=True`, the `.versions/{uuid}/` directory for the deleted artifact is preserved. You can discover and restore these orphaned snapshots using the following tools:

*   **`list_deleted_artifacts()`**: Scans the `.versions/` directory for orphaned snapshots. Returns a list of dictionaries containing the `title`, `version`, and `snapshot_path` of recoverable files.
*   **`restore_from_version(title, version, activate=True)`**: Re-registers a deleted artifact from its preserved historical snapshot. It reads the physical bytes and logical schema (`.lam`) from the `.versions/` directory and re-introduces it into the active workspace as a fresh `v1`.

```python
# 1. List recoverable deleted artifacts
deleted_items = discussion.artefacts.list_deleted_artifacts()
if deleted_items:
    # 2. Restore the first found snapshot
    target = deleted_items[0]
    restored_art = discussion.artefacts.restore_from_version(
        title=target["title"],
        version=target["version"]
    )
```

### Defensive Synchronization (`sync_all_active_to_disk`)
The `sync_all_active_to_disk()` method contains a **defense-in-depth reconciliation pass**. After syncing all active DB artifacts to disk, it scans the workspace root one final time. If it discovers any file on disk that is NOT backed by an active DB record, it immediately unlinks (deletes) it. This guarantees that even if a stale write or a race condition attempts to re-materialize a deleted file, the filesystem is corrected to perfectly match the database state.

---

## 🛠️ 9. Class Reference

*   **`ArtefactType`**: Registry defining the supported categories (`DATA`, `CODE`, `DOCUMENT`, `IMAGE`, `PRESENTATION`, `NOTE`, `SKILL`, `TOOL`, `SCRATCHPAD`).
    *   `SKILL`: Represents persistent knowledge or behavior capsules. When created in a discussion with a Handbag personality, skills are routed to `handbag/skills/<name>/SKILL.md`. For manual/stateless personalities, they are versioned and stored directly as discussion artefacts.
*   **`ArtefactVisibility`**: Enum-like class defining the context tiers (`FULL`, `TREE_UNLOCKABLE`, `METADATA`, `TREE_LOCKED`, `HIDDEN`).
*   **`ArtefactStatus`**: Enum-like class defining the lifecycle states (`DRAFTING`, `STABLE`, `REVISING`, `ERROR`).
*   **`ArtefactManager`**: Orchestrates database CRUD operations, applies search-and-replace patches, manages version history squashing, and gates dynamic tool registration.
*   **`FileImportMixin`**: Contains multi-modal parser subroutines for importing PDFs, Word documents, PowerPoint presentations, and audio files.
*   **`InternetImportMixin`**: Provides native web scraping and semantic search operations (Arxiv, GitHub, StackOverflow, Wikipedia, etc.).
*   **`ExportMixin`**: Handles exporting artifacts to various formats (PDF, DOCX, PPTX, HTML, ZIP).