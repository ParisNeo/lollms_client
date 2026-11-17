# `enable_office_files` (DEPRECATED)

```python
enable_office_files: bool = False
```

**Category**: Tools & Security
**Status**: **DEPRECATED** — prefer the four per-software flags
(`enable_powerpoint` / `enable_word` / `enable_excel` / `enable_pdf`).

## What it does

Legacy umbrella flag. Setting `enable_office_files=True`:

1. Enables all four per-software office flags (`enable_powerpoint`,
   `enable_word`, `enable_excel`, `enable_pdf`) — each mounts its own LCP
   library.
2. ALSO mounts the monolithic `office_files_builder` library
   (12 tools: `tool_office_build` / `tool_office_edit` / `tool_office_read`
   plus per-format aliases).

A one-line deprecation notice is logged on every turn that uses the flag.

## What to expect

- Behaves exactly like passing the four per-software flags **plus** an
  additional `office_files_builder` mount.
- Identical tool names between the monolithic library and the per-format
  libraries are deduplicated by the LCP registry — the later mount wins.
- All office-toolset behavior (vision gate, integrity-verified saves,
  sandboxed recipe code, first-use installs) is inherited from the per-format
  libraries.

## Usage

```python
# DEPRECATED — use per-software flags instead
result = discussion.chat(
    "Build a deck and a docx summary",
    enable_office_files=True,   # ← deprecation notice logged
)

# PREFERRED
result = discussion.chat(
    "Build a deck and a docx summary",
    enable_powerpoint=True,
    enable_word=True,
)
```

## Related flags

- `enable_powerpoint` — PowerPoint (.pptx / .pptm) toolset.
- `enable_word` — Word (.docx / .docm) toolset.
- `enable_excel` — Excel (.xlsx / .xlsm) toolset.
- `enable_pdf` — PDF read/inspect/annotate toolset.

## Migration guide

Replace a single `enable_office_files=True` with the intersection of the
per-format flags you actually need. For example:

| Old | New (preferred) |
|---|---|
| `enable_office_files=True` | `enable_powerpoint=True, enable_word=True, enable_excel=True, enable_pdf=True` |
| `enable_office_files=True` (deck only) | `enable_powerpoint=True` |
| `enable_office_files=True` (reports only) | `enable_word=True, enable_excel=True` |

If you relied on the monolithic library's `tool_office_build` /
`tool_office_edit` / `tool_office_read` API, note that the per-format libraries
expose `tool_build_<format>` / `tool_edit_<format>` / `tool_read_<format>` —
functionally equivalent, format-scoped.