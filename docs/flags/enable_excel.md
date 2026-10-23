# `enable_excel: bool = False`

**Category**: Tools & Security

## What it does

Mounts the `excel_tools` LCP toolset — build, edit and read Excel workbooks
(`.xlsx` / `.xlsm`) via sandboxed openpyxl recipe code. Three tools:

- `tool_build_excel(file, recipe)` — create a brand-new workbook from a
  sandboxed recipe.
- `tool_edit_excel(file, recipe)` — modify an existing workbook; the pre-edit
  bytes are restored if validation fails.
- `tool_read_excel(file)` — validate, then return a markdown inventory
  (sheets, headers, structure).

## What to expect — FORMULAS ONLY (critical doctrine)

**NEVER write calculation results manually — ALWAYS use formulas** for any
computed value. Excel computes them when the user opens the file:

```python
ws["F2"] = "=SUM(B2:E2)"
ws["G2"] = "=AVERAGE(B2:E2)"
ws["H2"] = '=IF(F2>100,"High","Low")'
ws["I2"] = "=VLOOKUP(A2, Sheet2!A:C, 3, FALSE)"
ws["J2"] = "=INDEX(C:C, MATCH(A2, A:A, 0))"
```

- **Literals ONLY for input data**: raw values belong only in the source data
  cells the user provided. Every derived cell must be a formula.
- **Chart-recalc caveat**: charts referencing formula cells may appear empty
  until Excel recalculates on open — formulas are still the correct choice.
  The validator re-opens the file structurally, so formula cells PASS
  validation (no computed-value requirement).

Other behavior:

- **NO vision gate**: the mount is unconditional when the flag is set
  (text-only workflow: recipe code in → markdown inventory out).
- **Sandbox**: imports allowed ONLY from the `openpyxl.*` whitelist plus safe
  stdlib modules; the roots `wb` and `ws` (active sheet) are pre-bound.
  Forbidden tokens are rejected before execution. Recipes capped at 60,000
  characters. `.xlsm` files keep their VBA macros (`keep_vba`).
- **No `wb.save()`**: the harness saves and integrity-verifies automatically.
  Every save passes 3-layer validation (ZIP + required OOXML parts, strict XML
  parse, openpyxl re-open). Corruption → automatic rollback + failure code
  `office_file_failed_validation`.
- **First call is slower**: pipmaster auto-installs `openpyxl`.
- **Fail-closed LCP gate**: the `excel_tools` library is office-family and
  never auto-discovers — it appears only when this flag (or the legacy kwarg)
  is set.

## Recovering generated stuff

- Tools write into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file`, `file_name` (sanitized).
- Files are auto-registered as artifacts by `_sync_tool_artifacts` (workspace
  snapshot diffing) — recover via `discussion.artefacts.get("kpi.xlsx")` or
  open `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.

## Usage

```python
result = discussion.chat(
    "Build an .xlsx KPI workbook from the sales data, with computed totals",
    enable_excel=True,
)
# The model writes ws["F2"] = "=SUM(B2:E2)" — NOT the literal result.
# Recover: discussion.artefacts.get("kpi_workbook.xlsx")
```

## Related flags

- [`enable_powerpoint`](enable_powerpoint.md) / [`enable_word`](enable_word.md)
  — the sibling per-software toolsets.
- [`enable_office_files`](enable_office_files.md) — deprecated legacy alias.