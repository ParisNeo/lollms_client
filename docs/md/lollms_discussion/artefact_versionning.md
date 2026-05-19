# Artefact Versioning — Documentation

The artefact system uses a **Linear Versioning Model**. Every time an artefact is modified (either by the LLM via XML tags or programmatically via the API), a new immutable record is created in the database with an incremented version number.

---

## Table of Contents

1. [How the LLM Sees Versions](#1-how-the-llm-sees-versions)
2. [Version History API](#2-version-history-api)
   - 2.1 [`get_version_history(title)`](#21-get_version_historytitle)
   - 2.2 [`diff_versions(title, version_a, version_b)`](#22-diff_versionstitle-version_a-version_b)
3. [Version Squashing API](#3-version-squashing-api)
   - 3.1 [`squash_versions(title, ...)`](#31-squash_versionstitle-)
   - 3.2 [`cleanup_old_versions(title, keep_count, min_age_hours)`](#32-cleanup_old_versionstitle-keep_count-min_age_hours)
4. [Recovering History (Application Side)](#4-recovering-history-application-side)
5. [Versioning Workflow](#5-versioning-workflow)
6. [Safety Rules](#6-safety-rules)

---

## 1. How the LLM Sees Versions

The system automatically manages the LLM's context to prevent token waste and confusion:
*   **Latest Version Priority**: Only the version marked as `active` is injected into the context. By default, this is always the most recently created version.
*   **Version Headers**: The LLM receives a header identifying the version it is reading:
    `###[Code] main.py (v3 | 5 total versions exist)`
*   **Reference Awareness**: Because the header includes the total count, the LLM knows it can "look back" by requesting a revert if the current version is broken.

---

## 2. Version History API

### 2.1 `get_version_history(title)`

Returns the complete version history for an artefact, sorted by version number.

**Signature:**
```python
def get_version_history(self, title: str) -> List[Dict[str, Any]]
```

**Returns:** A list of dicts, one per version, each containing:
| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Version number |
| `created_at` | `str` | ISO datetime when created |
| `updated_at` | `str` | ISO datetime of last update |
| `content_preview` | `str` | First 200 chars of content (with `…` if truncated) |
| `size_chars` | `int` | Total character count of content |
| `image_count` | `int` | Number of embedded images |
| `is_active` | `bool` | Whether this is the currently active version |

**Example:**
```python
history = discussion.artefacts.get_version_history("app.py")
for entry in history:
    marker = "★" if entry["is_active"] else " "
    print(f"{marker} v{entry['version']} ({entry['size_chars']} chars, {entry['image_count']} images)")
    print(f"    {entry['content_preview'][:80]}")
```

**Legacy alias:** `get_version_history_artefact(title)`

---

### 2.2 `diff_versions(title, version_a, version_b)`

Compute a line-based diff between two artefact versions. Useful for showing changes in a UI or auditing what changed between revisions.

**Signature:**
```python
def diff_versions(self, title: str, version_a: int, version_b: int) -> Dict[str, Any]
```

**Returns:** A dict with:
| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Artefact title |
| `version_a` | `int` | First version compared |
| `version_b` | `int` | Second version compared |
| `unified_diff` | `str` | Standard unified diff (GNU diff format) |
| `added_lines` | `int` | Count of lines only in `version_b` |
| `removed_lines` | `int` | Count of lines only in `version_a` |
| `common_lines` | `int` | Count of lines present in both |
| `added_content` | `str` | Newline-joined added lines |
| `removed_content` | `str` | Newline-joined removed lines |

**Example:**
```python
diff = discussion.artefacts.diff_versions("app.py", version_a=2, version_b=5)
print(f"Changes from v{diff['version_a']} → v{diff['version_b']}:")
print(f"  +{diff['added_lines']} / -{diff['removed_lines']} lines")
print(f"  {diff['common_lines']} unchanged lines")

# Show the actual unified diff
print(diff["unified_diff"])
```

**Legacy alias:** `diff_versions_artefact(title, version_a, version_b)`

---

## 3. Version Squashing API

Over time, artefacts can accumulate many versions, consuming storage space and inflating context size. The squashing API allows you to compact version history while preserving important revisions.

### 3.1 `squash_versions(title, ...)`

The main squashing primitive. Supports three mutually exclusive modes controlled by keyword arguments.

**Signature:**
```python
def squash_versions(
    self,
    title: str,
    keep_versions: Optional[List[int]] = None,
    keep_last_n: Optional[int] = None,
    target_version: Optional[int] = None,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `title` | The artefact to squash |
| `keep_versions` | Explicit whitelist of version numbers to preserve. All others deleted. |
| `keep_last_n` | Retain only the N most recent versions (by version number). Older ones deleted. |
| `target_version` | Collapse everything to this single version, which is renumbered to v1. All other versions deleted. |

**Exactly one** of `keep_versions`, `keep_last_n`, or `target_version` must be provided.

**Returns:** A result dict:
```python
{
    "success": True,
    "deleted": [1, 2, 3],           # version numbers removed
    "preserved": [4, 5],            # version numbers kept
    "new_baseline": 1,              # set if target_version was renumbered
    "space_reclaimed_estimate": 4500,  # approximate chars freed
}
```

**Examples:**

```python
# Mode A: Collapse to a single baseline version
result = discussion.artefacts.squash_versions(
    "app.py",
    target_version=5,   # keep v5, delete everything else, renumber to v1
)
print(f"Freed ~{result['space_reclaimed_estimate']} chars")

# Mode B: Keep only specific versions
result = discussion.artefacts.squash_versions(
    "app.py",
    keep_versions=[3, 5, 7],  # preserve v3, v5, v7; delete all others
)

# Mode C: Keep only the last N versions
result = discussion.artefacts.squash_versions(
    "app.py",
    keep_last_n=3,  # preserve the 3 most recent versions
)
```

**Legacy alias:** `squash_versions_artefact(title, keep_versions, keep_last_n, target_version)`

---

### 3.2 `cleanup_old_versions(title, keep_count, min_age_hours)`

Convenience wrapper around `squash_versions` for routine maintenance. Keeps only the N most recent versions, with an optional age guard.

**Signature:**
```python
def cleanup_old_versions(
    self,
    title: str,
    keep_count: int = 5,
    min_age_hours: Optional[float] = None,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `title` | — | Artefact to clean up |
| `keep_count` | `5` | Number of recent versions to retain |
| `min_age_hours` | `None` | If set, only versions older than this many hours are eligible for deletion |

**Example:**
```python
# Keep only 5 most recent versions
result = discussion.artefacts.cleanup_old_versions("app.py", keep_count=5)

# Only delete versions older than 24 hours, keeping 5 recent
result = discussion.artefacts.cleanup_old_versions(
    "app.py",
    keep_count=5,
    min_age_hours=24.0,
)
```

**Legacy alias:** `cleanup_old_versions_artefact(title, keep_count, min_age_hours)`

---

## 4. Recovering History (Application Side)

The application can retrieve the full evolution of an artefact using the `get_version_history` method. This is useful for building "Timeline" or "Undo" UI components.

```python
history = discussion.artefacts.get_version_history("app.py")
for entry in history:
    marker = "★" if entry["is_active"] else " "
    print(f"{marker} v{entry['version']} — {entry['size_chars']} chars")
```

---

## 5. Versioning Workflow

1.  **Creation**: `discussion.chat()` results in `<artefact name="test.py">...</artefact>`. **(v1, Active)**
2.  **Update**: The LLM patches the file using SEARCH/REPLACE. **(v2, Active)**. *Note: v1 is automatically marked Inactive.*
3.  **Validation**: The App checks `get_version_history("test.py")` and sees two entries.
4.  **Token Count**: `get_context_status()` counts only the tokens for **v2**.
5.  **Diff Audit**: App calls `diff_versions("test.py", 1, 2)` to show the user what changed.
6.  **Squash**: After many iterations, App calls `cleanup_old_versions("test.py", keep_count=3)` to reclaim space.
7.  **Recovery**: If the user wants to go back, the App calls `revert(title, 1)`. **(v3, Active)**. *v3 content matches v1.*

---

## 6. Safety Rules

The squashing system enforces two hard safety constraints:

| Rule | Behavior |
|------|----------|
| **Active version protection** | The currently active version is **never** deleted, even if it would be removed by the squashing criteria. It is automatically added to the preserved set. |
| **At-least-one guarantee** | A squash that would delete **all** versions is rejected with a `ValueError`. At least one version must remain after any squash operation. |

These rules ensure that:
- The LLM never loses its current working context unexpectedly.
- Accidental misconfiguration (e.g., `keep_last_n=0`) cannot destroy all version history.