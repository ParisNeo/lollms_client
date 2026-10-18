# `enable_data_tools`

```python
enable_data_tools: bool = True
```

**Category**: Tools & Security

## What it does

Enables data tools **contextually**:

- SQL/pandas query tools (`tool_execute_python_data_query`,
  `tool_get_table_schema`, `tool_query_database_sql`) mount only when **data
  files exist** in the workspace (`.csv`, `.db`, `.sqlite`, `.xlsx`, `.parquet`,
  ...).
- Document tools (`tool_inspect_document`, `tool_read_document_content`,
  `tool_annotate_document`, `tool_edit_document_text`, ...docx/excel/pptx
  modifiers) mount only when **document files exist** (`.pdf`, `.docx`, `.pptx`,
  `.md`, ...).

## What to expect

- No matching files → no data tools in the registry, even with the flag on.
- The workspace scan is cached per write-revision (`_workspace_write_revision`)
  for performance.

## Usage

```python
# Ensure a data file exists in the discussion workspace first
discussion.chat("Analyze sales.csv", enable_data_tools=True)
```

## Related flags

- `enable_artefacts` — data files in the workspace come from artifacts.