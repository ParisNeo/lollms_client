"""
semantic_data_engineer.py
=========================
LOLLMS Unified Semantic Data Engineering & Safe Macro Library.
Provides a comprehensive suite of pre-compiled, secure data macros
to analyze, filter, aggregate, and visualize datasets without requiring 
any LLM-generated code execution.
"""

import os
import sys
import re
import json
import sqlite3
import base64
import io
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

TOOL_LIBRARY_NAME = "SEMANTIC_DATA_ENGINEER"
TOOL_LIBRARY_DESC = "A highly specialized data engineering library providing safe pre-compiled data macros (filtering, aggregation, schemas, plotting, ABox conversion) without code execution."
TOOL_LIBRARY_ICON = "📊"

def init_tool_library() -> None:
    """Ensure required libraries are installed."""
    import pipmaster as pm
    pm.ensure_packages({
        "pandas": ">=1.3.0",
        "matplotlib": ">=3.4.0",
        "numpy": ">=1.20.0",
        "sqlalchemy": ">=1.4.0",
        "openpyxl": ">=3.0.0"
    })


def _get_workspace_dir() -> Path:
    """Returns the primary active workspace directory dynamically from the running server process."""
    from ascii_colors import ASCIIColors
    import sys
    workspace_dir = Path("./data_workspace")
    try:
        # Query sys.modules dynamically to avoid stale direct-import reference copies
        if "lollms_client.app.server" in sys.modules:
            server_mod = sys.modules["lollms_client.app.server"]
            active_ws = getattr(server_mod, "APP_WORKSPACE_DIR", None)
            if active_ws is not None:
                workspace_dir = active_ws
                ASCIIColors.cyan(f"[SemanticDataEngineer] Successfully resolved active workspace: '{workspace_dir.resolve()}'")
            else:
                ASCIIColors.warning(f"[SemanticDataEngineer] server.APP_WORKSPACE_DIR is None. Defaulting to: '{workspace_dir.resolve()}'")
        else:
            ASCIIColors.warning(f"[SemanticDataEngineer] lollms_client.app.server is not loaded in sys.modules. Defaulting to: '{workspace_dir.resolve()}'")
    except Exception as e:
        ASCIIColors.warning(f"[SemanticDataEngineer] Dynamic workspace resolution failed: {e}. Defaulting to: '{workspace_dir.resolve()}'")
    return workspace_dir


def _load_data_source(file_path: Path, table_name: Optional[str] = None, discussion_instance: Optional[Any] = None) -> Tuple[Any, str]:
    """Load a CSV, Excel, or SQLite file into a Pandas DataFrame with automatic BOM, delimiter, and self-healing restoration."""
    import pandas as pd
    import numpy as np
    from ascii_colors import ASCIIColors

    resolved_path = file_path.resolve()

    # ── 🛡️ SELF-HEALING FILE RESTORATION PROTOCOL ──
    # If the file is missing from the physical workspace disk folder, but exists
    # as a record inside the SQLite discussion artifacts, restore/write it back instantly!
    if not resolved_path.exists() and discussion_instance:
        try:
            # Strip extension to find the clean artifact title key
            art_title = file_path.name
            ext = file_path.suffix.lower()
            if art_title.lower().endswith(ext):
                art_title = art_title[:-len(ext)]

            art = discussion_instance.artefacts.get(art_title)
            if not art:
                # Fuzzy fallback matching
                for item in discussion_instance.artefacts.list():
                    if art_title in item["title"]:
                        art = item
                        break

            if art and art.get("content"):
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_path.write_text(art["content"], encoding="utf-8")

                # Also write to active unversioned path for general consistency
                active_path = resolved_path.parent.parent / file_path.name
                try:
                    active_path.parent.mkdir(parents=True, exist_ok=True)
                    active_path.write_text(art["content"], encoding="utf-8")
                except Exception:
                    pass

                ASCIIColors.success(f"✓ [Self-Healing] Restored missing workspace file '{file_path.name}' directly from the database record!")
        except Exception as restore_err:
            ASCIIColors.warning(f"Self-healing file restoration failed: {restore_err}")

    ASCIIColors.cyan(f"[SemanticDataEngineer] Loading data source from: '{resolved_path}'")
    ASCIIColors.cyan(f"  - File Exists: {resolved_path.exists()}")
    if resolved_path.exists():
        ASCIIColors.cyan(f"  - File Size  : {resolved_path.stat().st_size:,} bytes")
    else:
        ASCIIColors.error(f"  - File Missing! Crucial Error: Database path does not exist on disk.")

    ext = file_path.suffix.lower()

    if ext in (".db", ".sqlite", ".sqlite3"):
        conn = sqlite3.connect(str(file_path))
        if not table_name:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
            if not tables:
                conn.close()
                raise ValueError("No tables found in SQLite database.")
            table_name = tables[0]
        ASCIIColors.cyan(f"  - Querying SQLite Table: '{table_name}'")
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()
        ASCIIColors.cyan(f"  - Parsed Columns: {list(df.columns)}")
        return df, table_name
    elif ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(str(file_path))
        sheet = table_name or xl.sheet_names[0]
        ASCIIColors.cyan(f"  - Querying Excel Sheet: '{sheet}' (Available Sheets: {xl.sheet_names})")
        df = pd.read_excel(str(file_path), sheet_name=sheet)
        ASCIIColors.cyan(f"  - Parsed Columns: {list(df.columns)}")
        return df, sheet
    else:
        # Read the first line using utf-8-sig to safely detect the delimiter
        try:
            first_line = file_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
            ASCIIColors.cyan(f"  - Raw Header Line Preview: {repr(first_line)}")
        except Exception as ex:
            first_line = ""
            ASCIIColors.warning(f"  - Delimiter reader warning: {ex}")

        # Robust multi-character/multi-delimiter heuristic detection
        sep = ","
        if ";" in first_line:
            # Count separator occurrences to find the dominant delimiter
            semicolon_count = first_line.count(";")
            comma_count = first_line.count(",")
            if semicolon_count > comma_count:
                sep = ";"
            else:
                sep = ","
        elif "\t" in first_line:
            sep = "\t"

        ASCIIColors.cyan(f"  - Resolved Delimiter Separator: {repr(sep)}")
        df = pd.read_csv(str(file_path), sep=sep, encoding="utf-8-sig")

        # Clean column headers of BOM sequences or stray quotation marks
        df.columns = [c.strip().strip("'\"").replace('\ufeff', '') for c in df.columns]

        ASCIIColors.cyan(f"  - Parsed Columns after normalization: {list(df.columns)}")
        return df, file_path.stem


def _save_data_source(df: Any, file_path: Path, table_name: str) -> None:
    """Write modified DataFrame back to disk, preserving format."""
    ext = file_path.suffix.lower()
    if ext in (".db", ".sqlite", ".sqlite3"):
        conn = sqlite3.connect(str(file_path))
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
    elif ext in (".xlsx", ".xls"):
        import pandas as pd
        # Write to single sheet
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=table_name[:31], index=False)
    else:
        sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
        df.to_csv(file_path, sep=sep, index=False)


# ── 1. SCHEMA DETECTOR MACRO ────────────────────────────────────────────────

def tool_get_table_schema(
    file_name: str,
    table_name: Optional[str] = None
) -> dict:
    """
    Retrieves the exact column names, data types, row counts, and null counts of a dataset.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file in the workspace.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
    """
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
        
        schema = {}
        for col in df.columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": [str(x) for x in df[col].dropna().head(3).tolist()]
            }

        return {
            "success": True,
            "file_name": file_name,
            "table_name": resolved_table,
            "total_rows": len(df),
            "columns_count": len(df.columns),
            "schema": schema,
            "output": f"Dataset '{file_name}' (table '{resolved_table}') has {len(df):,} rows and {len(df.columns)} columns."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to retrieve schema: {e}"}


# ── 2. FILTER & SLICE DATA MACRO ────────────────────────────────────────────

def tool_filter_and_slice_data(
    file_name: str,
    table_name: Optional[str] = None,
    filter_column: Optional[str] = None,
    operator: Optional[str] = "==",
    filter_value: Optional[str] = None,
    columns_to_keep: Optional[List[str]] = None,
    limit: int = 50,
    save_as_new_artifact: bool = False,
    output_artifact_title: Optional[str] = None
) -> dict:
    """
    Filters and slices a dataset without writing Python code, optionally saving the output as a new version or artifact.

    Args:
        file_name (str): Filename of the CSV, Excel, or SQLite file in the workspace.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
        filter_column (str, optional): The column name to filter by.
        operator (str, optional): Comparison operator: '==', '!=', '>', '<', '>=', '<=', 'contains'. Defaults to '=='.
        filter_value (str, optional): Value to match against.
        columns_to_keep (list, optional): List of columns to keep (slices table).
        limit (integer, optional): Maximum rows to return in the preview. Defaults to 50.
        save_as_new_artifact (boolean, optional): If True, writes the filtered result back to a new workspace file and registers it. Defaults to False.
        output_artifact_title (str, optional): Title of the new artifact if save_as_new_artifact is True.
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
        
        # Apply columns slice
        if columns_to_keep:
            valid_cols = [c for c in columns_to_keep if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # Apply filter
        if filter_column and filter_column in df.columns and filter_value is not None:
            # Type-coerced comparisons
            if operator == "==":
                df = df[df[filter_column].astype(str) == str(filter_value)]
            elif operator == "!=":
                df = df[df[filter_column].astype(str) != str(filter_value)]
            elif operator == "contains":
                df = df[df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)]
            else:
                # Numeric operators
                try:
                    num_val = float(filter_value)
                    df[filter_column] = pd.to_numeric(df[filter_column])
                    if operator == ">":
                        df = df[df[filter_column] > num_val]
                    elif operator == "<":
                        df = df[df[filter_column] < num_val]
                    elif operator == ">=":
                        df = df[df[filter_column] >= num_val]
                    elif operator == "<=":
                        df = df[df[filter_column] <= num_val]
                except Exception as num_err:
                    return {"success": False, "error": f"Numerical operator '{operator}' failed on column '{filter_column}': {num_err}"}

        total_matching_rows = len(df)
        preview_df = df.head(limit)
        
        from lollms_client.lollms_discussion._data_files import _dataframe_to_markdown
        markdown_table = _dataframe_to_markdown(preview_df)

        # Handle file persistence & new artifact registration
        if save_as_new_artifact and discussion_instance:
            out_title = output_artifact_title or f"{Path(file_name).stem}_filtered"
            ext = file_path.suffix.lower()
            out_filename = f"{out_title}{ext}"
            out_path = workspace_dir / out_filename
            
            _save_data_source(df, out_path, resolved_table)
            
            # Setup new interactive data schema
            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
            new_schema, _ = _parse_data_file(out_path, out_title, version=1, progress_cb=None)
            
            art = discussion_instance.artefacts.add(
                title=out_title,
                artefact_type=ArtefactType.DATA,
                content=new_schema,
                file_ext=ext,
                active=True,
                read_only=True
            )
            discussion_instance.commit()
            
            return {
                "success": True,
                "total_rows": total_matching_rows,
                "artifact_created": out_title,
                "output": f"Filtered dataset saved successfully as new active artifact '{out_title}'.\n\n### Preview (First {limit} rows):\n\n{markdown_table}"
            }

        return {
            "success": True,
            "total_rows": total_matching_rows,
            "output": f"Query found {total_matching_rows:,} matching row(s).\n\n### Preview (First {limit} rows):\n\n{markdown_table}"
        }

    except Exception as e:
        return {"success": False, "error": f"Data filtering failed: {e}"}


# ── 3. GET UNIQUE VALUES MACRO ──────────────────────────────────────────────

def tool_get_unique_values(
    file_name: str,
    column_name: str,
    table_name: Optional[str] = None,
    limit: int = 100
) -> dict:
    """
    Returns unique elements and category frequency counts from a column.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file in the workspace.
        column_name (str): The column to analyze.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
        limit (integer, optional): Maximum unique items to list. Defaults to 100.
    """
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, _ = _load_data_source(file_path, table_name)
        if column_name not in df.columns:
            return {"success": False, "error": f"Column '{column_name}' not found in dataset."}

        counts = df[column_name].value_counts().head(limit)
        unique_list = []
        for val, count in counts.items():
            unique_list.append({"value": str(val), "count": int(count)})

        from lollms_client.lollms_discussion._data_files import _dataframe_to_markdown
        import pandas as pd
        counts_df = pd.DataFrame(unique_list)
        md_table = _dataframe_to_markdown(counts_df)

        return {
            "success": True,
            "column": column_name,
            "total_unique": int(df[column_name].nunique()),
            "unique_values": unique_list,
            "output": f"Column '{column_name}' has {df[column_name].nunique():,} unique values.\n\n### Frequency Counts (Top {limit}):\n\n{md_table}"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get unique values: {e}"}


# ── 4. AGGREGATOR MACRO ─────────────────────────────────────────────────────

def tool_compute_column_aggregations(
    file_name: str,
    metric_column: str,
    group_by_column: Optional[str] = None,
    table_name: Optional[str] = None,
    operation: str = "mean"
) -> dict:
    """
    Computes mathematical aggregations on a numerical column (sum, mean, min, max, count), optionally grouping by another column.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file in the workspace.
        metric_column (str): The numerical column to aggregate.
        group_by_column (str, optional): Optional categorical column to group by.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
        operation (str, optional): Aggregation operation: 'sum', 'mean', 'min', 'max', 'count'. Defaults to 'mean'.
    """
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        import pandas as pd
        df, _ = _load_data_source(file_path, table_name)
        
        if metric_column not in df.columns:
            return {"success": False, "error": f"Metric column '{metric_column}' not found."}

        df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
        op = operation.lower().strip()

        if group_by_column:
            if group_by_column not in df.columns:
                return {"success": False, "error": f"Group By column '{group_by_column}' not found."}
            
            # Apply group by operation
            grouped = df.groupby(group_by_column)[metric_column]
            if op == "sum": res_df = grouped.sum()
            elif op == "min": res_df = grouped.min()
            elif op == "max": res_df = grouped.max()
            elif op == "count": res_df = grouped.count()
            else: res_df = grouped.mean() # default mean

            res_df = res_df.reset_index()
            res_df.columns = [group_by_column, f"{op}_{metric_column}"]
        else:
            # Single value aggregation
            if op == "sum": val = df[metric_column].sum()
            elif op == "min": val = df[metric_column].min()
            elif op == "max": val = df[metric_column].max()
            elif op == "count": val = df[metric_column].count()
            else: val = df[metric_column].mean()

            res_df = pd.DataFrame([{"operation": op, "column": metric_column, "result": float(val)}])

        from lollms_client.lollms_discussion._data_files import _dataframe_to_markdown
        md_table = _dataframe_to_markdown(res_df)

        return {
            "success": True,
            "operation": op,
            "metric_column": metric_column,
            "group_by_column": group_by_column,
            "output": f"Successfully completed {op} aggregation on column '{metric_column}':\n\n{md_table}"
        }
    except Exception as e:
        return {"success": False, "error": f"Aggregation failed: {e}"}


# ── 5. SAFE INTERACTIVE DATA EDITING TOOLS ──────────────────────────────────

def tool_update_cell_value(
    file_name: str,
    table_name: Optional[str] = None,
    row_match_column: str = "",
    row_match_value: str = "",
    column_to_update: str = "",
    new_value: str = ""
) -> dict:
    """
    Surgically updates a cell value in a spreadsheet or database row without code execution.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
        row_match_column (str): Column name used to locate the target row (e.g., 'id').
        row_match_value (str): Value to match in that column to locate the row.
        column_to_update (str): Column name where the value needs to be modified.
        new_value (str): The new value to set.
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
        
        if row_match_column not in df.columns:
            return {"success": False, "error": f"Row matching column '{row_match_column}' not found."}
        if column_to_update not in df.columns:
            return {"success": False, "error": f"Target update column '{column_to_update}' not found."}

        # Find row and apply update
        mask = df[row_match_column].astype(str) == str(row_match_value)
        match_count = int(mask.sum())
        
        if match_count == 0:
            return {"success": False, "error": f"No rows found matching '{row_match_column} == {row_match_value}'."}

        # Apply type-coerced update
        orig_dtype = df[column_to_update].dtype
        try:
            if "int" in str(orig_dtype):
                coerced_val = int(new_value)
            elif "float" in str(orig_dtype):
                coerced_val = float(new_value)
            elif "bool" in str(orig_dtype):
                coerced_val = new_value.lower() in ("true", "1", "yes")
            else:
                coerced_val = str(new_value)
        except Exception:
            coerced_val = str(new_value)

        df.loc[mask, column_to_update] = coerced_val
        
        # Save updated data
        _save_data_source(df, file_path, resolved_table)

        # Trigger version increment in SQLite and file workspace sync
        if discussion_instance:
            existing = discussion_instance.artefacts.get(file_name)
            if existing:
                from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                new_schema, _ = _parse_data_file(file_path, file_name, version=existing["version"] + 1, progress_cb=None)
                discussion_instance.artefacts.update(
                    title=file_name,
                    new_content=new_schema,
                    new_type="data",
                    active=True,
                    file_ext=file_path.suffix.lower()
                )
                discussion_instance.commit()

        return {
            "success": True,
            "rows_updated": match_count,
            "output": f"Successfully updated '{column_to_update}' to '{new_value}' in {match_count} row(s) matching '{row_match_column} == {row_match_value}'."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to edit cell value: {e}"}


def tool_insert_new_row(
    file_name: str,
    row_data: Dict[str, Any],
    table_name: Optional[str] = None
) -> dict:
    """
    Inserts a new row/record into a spreadsheet or SQLite table.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file.
        row_data (dict): Dictionary representing column names and corresponding values for the new row.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
        
        # Build new row aligning with existing columns
        new_row_dict = {}
        for col in df.columns:
            if col in row_data:
                # Type-coerce value to match column data type
                orig_dtype = df[col].dtype
                val = row_data[col]
                try:
                    if "int" in str(orig_dtype): new_row_dict[col] = int(val)
                    elif "float" in str(orig_dtype): new_row_dict[col] = float(val)
                    elif "bool" in str(orig_dtype): new_row_dict[col] = val in (True, "true", "True", 1, "1")
                    else: new_row_dict[col] = str(val)
                except Exception:
                    new_row_dict[col] = val
            else:
                new_row_dict[col] = None

        new_row_df = pd.DataFrame([new_row_dict])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Save
        _save_data_source(df, file_path, resolved_table)

        # Trigger version increment in SQLite and file workspace sync
        if discussion_instance:
            existing = discussion_instance.artefacts.get(file_name)
            if existing:
                from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                new_schema, _ = _parse_data_file(file_path, file_name, version=existing["version"] + 1, progress_cb=None)
                discussion_instance.artefacts.update(
                    title=file_name,
                    new_content=new_schema,
                    new_type="data",
                    active=True,
                    file_ext=file_path.suffix.lower()
                )
                discussion_instance.commit()

        return {
            "success": True,
            "output": f"Successfully inserted new record into '{file_name}' (Table: '{resolved_table}'). Row content: {json.dumps(new_row_dict)}"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to insert row: {e}"}


def tool_delete_rows_by_criteria(
    file_name: str,
    match_column: str,
    match_value: str,
    table_name: Optional[str] = None
) -> dict:
    """
    Deletes all rows matching a specific column value.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file.
        match_column (str): Column name used to identify rows for deletion.
        match_value (str): Value to match in that column.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
    """
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
        
        if match_column not in df.columns:
            return {"success": False, "error": f"Matching column '{match_column}' not found."}

        mask = df[match_column].astype(str) == str(match_value)
        match_count = int(mask.sum())
        
        if match_count == 0:
            return {"success": False, "error": f"No rows found matching '{match_column} == {match_value}'."}

        # Keep everything except the matched rows
        df = df[~mask]
        
        # Save
        _save_data_source(df, file_path, resolved_table)

        # Trigger version increment in SQLite and file workspace sync
        if discussion_instance:
            existing = discussion_instance.artefacts.get(file_name)
            if existing:
                from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                new_schema, _ = _parse_data_file(file_path, file_name, version=existing["version"] + 1, progress_cb=None)
                discussion_instance.artefacts.update(
                    title=file_name,
                    new_content=new_schema,
                    new_type="data",
                    active=True,
                    file_ext=file_path.suffix.lower()
                )
                discussion_instance.commit()

        return {
            "success": True,
            "rows_deleted": match_count,
            "output": f"Successfully deleted {match_count} row(s) from '{file_name}' matching '{match_column} == {match_value}'."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to delete rows: {e}"}


# ── 6. SECURE RELATIONAL QUERY TOOL (SQL) ───────────────────────────────────

def tool_query_database_sql(
    file_name: str,
    sql_query: str
) -> dict:
    """
    Executes standard SQL queries directly against SQLite database files or local CSV/Excel table models.

    Args:
        file_name (str): Filename of the .db, .sqlite, .csv, or .xlsx file.
        sql_query (str): Valid SQLite standard SQL query to execute.
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    ext = file_path.suffix.lower()
    
    # Establish connection
    try:
        conn = sqlite3.connect(":memory:")
        if ext in (".db", ".sqlite", ".sqlite3"):
            disk_conn = sqlite3.connect(str(file_path))
            disk_conn.backup(conn)
            disk_conn.close()
        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(str(file_path))
            for sheet_name in xl.sheet_names:
                table_name = sheet_name.replace(" ", "_")
                df = pd.read_excel(str(file_path), sheet_name=sheet_name)
                df.to_sql(table_name, conn, index=False, if_exists="replace")
        else:
            sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
            df = pd.read_csv(str(file_path), sep=sep)
            df.to_sql(file_path.stem.replace(" ", "_"), conn, index=False, if_exists="replace")
    except Exception as conn_err:
        return {"success": False, "error": f"Failed to assemble SQL connection: {conn_err}"}

    # Execute SQL
    try:
        # Check if write query (updates/deletes/inserts)
        clean_query = sql_query.strip()
        clean_query = re.sub(r'--.*$', '', clean_query, flags=re.MULTILINE).strip()
        clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL).strip()
        is_select = clean_query.lower().startswith("select")

        if is_select:
            df_res = pd.read_sql_query(sql_query, conn)
            conn.close()

            from lollms_client.lollms_discussion._data_files import _dataframe_to_markdown
            md_table = _dataframe_to_markdown(df_res)
            return {
                "success": True,
                "rows_count": len(df_res),
                "output": f"SQL Query returned {len(df_res)} row(s):\n\n{md_table}"
            }
        else:
            # Writable query
            if is_read_only:
                conn.close()
                return {"success": False, "error": "Database is read-only. Writable SQL queries (INSERT/UPDATE/DELETE) are blocked."}

            cursor = conn.cursor()
            cursor.execute(sql_query)
            conn.commit()

            # Backup modified tables from memory DB back to disk file
            if ext in (".db", ".sqlite", ".sqlite3"):
                disk_conn = sqlite3.connect(str(file_path))
                conn.backup(disk_conn)
                disk_conn.close()
            elif ext in (".xlsx", ".xls"):
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                    for t in tables:
                        df_write = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                        df_write.to_excel(writer, sheet_name=t.replace("_", " "), index=False)
            else:
                sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
                df_write = pd.read_sql_query(f"SELECT * FROM {file_path.stem.replace(' ', '_')}", conn)
                df_write.to_csv(file_path, sep=sep, index=False)

            conn.close()

            # Trigger version increment in SQLite and file workspace sync
            if discussion_instance:
                existing = discussion_instance.artefacts.get(file_name)
                if existing:
                    from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                    new_schema, _ = _parse_data_file(file_path, file_name, version=existing["version"] + 1, progress_cb=None)
                    discussion_instance.artefacts.update(
                        title=file_name,
                        new_content=new_schema,
                        new_type="data",
                        active=True,
                        file_ext=file_path.suffix.lower()
                    )
                    discussion_instance.commit()

            return {
                "success": True,
                "rows_affected": cursor.rowcount,
                "output": f"SQL Write Query completed successfully. Affected rows: {cursor.rowcount}"
            }

    except Exception as query_err:
        conn.close()
        return {"success": False, "error": f"SQL query failed: {query_err}"}


# ── 7. ADVANCED MULTI-SERIES CHARTING & PLOT MACRO ──────────────────────────

def tool_generate_advanced_visualization(
    file_name: str,
    x_column: str,
    y_columns: List[str],
    table_name: Optional[str] = None,
    plot_type: str = "line",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colors: Optional[List[str]] = None
) -> dict:
    """
    Generates advanced multi-series charts (multi-line, stacked bar, scatter, pie) in high-quality dark mode.

    Args:
        file_name (str): Filename of the target CSV, Excel, or SQLite file.
        x_column (str): The column used as the X-axis (or category label).
        y_columns (list): List of column names to plot as independent series on the Y-axis.
        table_name (str, optional): Sheet name (Excel) or Table name (SQLite).
        plot_type (str, optional): The chart type ('line', 'bar', 'stacked_bar', 'scatter', 'pie'). Defaults to 'line'.
        title (str, optional): Plot title.
        x_label (str, optional): Independent variable axis label.
        y_label (str, optional): Dependent variable axis label.
        colors (list, optional): Hex codes representing the color palette.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
    except Exception as e:
        return {"success": False, "error": f"Failed to load data: {e}"}

    # Verify columns exist
    if x_column not in df.columns:
        return {"success": False, "error": f"X column '{x_column}' not found."}
    if isinstance(y_columns,list):
        for col in y_columns:
            if col not in df.columns:
                return {"success": False, "error": f"Y column '{col}' not found."}
    else:
        if y_columns not in df.columns:
            return {"success": False, "error": f"Y column '{col}' not found."}
        else:
            y_columns= [y_columns]

    if isinstance(colors, str): 
        colors=[colors]
    palette = colors or ["#4f46e5", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#06b6d4"]
    plot_filename = f"adv_plot_{uuid.uuid4().hex[:6]}.png"
    plot_path = workspace_dir / plot_filename
    plot_b64 = None

    try:
        plt.clf()
        plt.close('all')
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.tick_params(colors='#94a3b8', labelsize=10)
        ax.grid(True, linestyle='--')

        chart_title = title or f"Analysis: {resolved_table}"
        ax.set_title(chart_title, color='#f59e0b', fontsize=13, fontweight='bold', pad=12)

        # Plot Series based on type
        ptype = plot_type.lower().strip()
        
        if ptype == "line":
            for idx, col in enumerate(y_columns):
                c = palette[idx % len(palette)]
                ax.plot(df[x_column].astype(str), df[col], label=col, color=c, linewidth=2, marker='o', markersize=4)
        elif ptype == "scatter":
            for idx, col in enumerate(y_columns):
                c = palette[idx % len(palette)]
                ax.scatter(df[x_column], df[col], label=col, color=c, alpha=0.8)
        elif ptype == "bar":
            # Multi-bar offset rendering
            x = np.arange(len(df))
            width = 0.8 / len(y_columns)
            for idx, col in enumerate(y_columns):
                c = palette[idx % len(palette)]
                ax.bar(x + idx * width - 0.4 + width/2, df[col], width, label=col, color=c)
            ax.set_xticks(x)
            ax.set_xticklabels(df[x_column].astype(str))
        elif ptype == "stacked_bar":
            bottoms = np.zeros(len(df))
            for idx, col in enumerate(y_columns):
                c = palette[idx % len(palette)]
                ax.bar(df[x_column].astype(str), df[col], bottom=bottoms, label=col, color=c)
                bottoms += df[col].fillna(0).values
        elif ptype == "pie":
            # Pie takes only the first Y series
            col = y_columns[0]
            ax.pie(df[col], labels=df[x_column].astype(str), colors=palette, autopct='%1.1f%%', startangle=90, textprops={'color': '#cbd5e1'})
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.grid(False)

        if ptype != "pie":
            ax.set_xlabel(x_label or x_column, color='#94a3b8')
            ax.set_ylabel(y_label or ", ".join(y_columns[:2]), color='#94a3b8')
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        plt.savefig(str(plot_path), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    except Exception as plot_err:
        return {"success": False, "error": f"Advanced plot generation failed: {plot_err}"}

    prompt_injection = (
        f"\n\n=== 📊 ADVANCED VISUALIZATION READY ===\n"
        f"• Chart Type: `{ptype.upper()}` | Target: `{file_name}`\n"
        f"• Active Series: {', '.join([f'`{col}`' for col in y_columns])}\n"
        f"• Rendered Plot URL: [View Plot Image](/api/workspace_files/{plot_filename})\n\n"
        f"Reference in response using: `<img src=\"/api/workspace_files/{plot_filename}\" />`"
    )

    return {
        "success": True,
        "plot_filename": plot_filename,
        "plot_url": f"/api/workspace_files/{plot_filename}",
        "plot_b64": plot_b64,
        "prompt_injection": prompt_injection
    }


# ── 8. STATS & PLOT MACRO ───────────────────────────────────────────────────

def tool_compute_statistics_and_plot(
    file_name: str,
    table_name: Optional[str] = None,
    plot_type: str = "bar",
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    color: str = "#4f46e5"
) -> dict:
    """
    Computes numerical statistics (mean, variance, standard deviation, null counts)
    and generates a polished matplotlib plot saved in the workspace.

    Args:
        file_name (str): The filename of the CSV, Excel, or SQLite file in the workspace.
        table_name (str, optional): The sheet name (Excel) or table name (SQLite).
        plot_type (str): The plot format to generate ("bar", "line", "scatter", "histogram").
        x_column (str, optional): The independent variable column name.
        y_column (str, optional): The dependent variable column name.
        title (str, optional): The plot title.
        color (str): Hex color code for the plot elements.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    try:
        df, resolved_table = _load_data_source(file_path, table_name)
    except Exception as e:
        return {"success": False, "error": f"Failed to load data: {e}"}

    # 1. Compute Numerical Statistics
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std_dev": float(df[col].std()),
            "variance": float(df[col].var()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "missing_values": int(df[col].isnull().sum())
        }

    # 2. Generate Matplotlib Plot
    plot_filename = f"plot_{uuid.uuid4().hex[:6]}.png"
    plot_path = workspace_dir / plot_filename
    plot_b64 = None

    try:
        plt.clf()
        plt.close('all')
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.tick_params(colors='#94a3b8', labelsize=10)
        ax.grid(True, color='rgba(255,255,255,0.05)', linestyle='--')

        plot_title = title or f"{plot_type.capitalize()} Plot: {resolved_table}"
        ax.set_title(plot_title, color='#f59e0b', fontsize=12, fontweight='bold', pad=10)

        # Handle columns
        x_col = x_column or (df.columns[0] if len(df.columns) > 0 else "")
        y_col = y_column or (df.columns[1] if len(df.columns) > 1 else "")

        if x_col not in df.columns:
            x_col = df.columns[0]
        if y_col not in df.columns:
            y_col = df.columns[-1]

        if plot_type == "bar":
            plt.bar(df[x_col].astype(str), df[y_col], color=color)
        elif plot_type == "line":
            plt.plot(df[x_col], df[y_col], marker='o', color=color)
        elif plot_type == "scatter":
            plt.scatter(df[x_col], df[y_col], color=color)
        elif plot_type == "histogram":
            plt.hist(df[x_col].dropna(), bins=15, color=color, edgecolor='black')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        plt.savefig(str(plot_path), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    except Exception as plot_err:
        return {"success": False, "error": f"Plot generation failed: {plot_err}", "statistics": stats}

    prompt_injection = (
        f"\n\n=== 📊 DATA STATISTICS & PLOT READY ===\n"
        f"• File Analyzed : `{file_name}` (Table: `{resolved_table}`)\n"
        f"• Plot Generated: [View Plot Image](/api/workspace_files/{plot_filename})\n"
        f"• Numeric Stats :\n"
    )
    for col, s in stats.items():
        prompt_injection += f"  - **{col}** → Mean: {s['mean']:.2f} | StdDev: {s['std_dev']:.2f} | Max: {s['max']:.2f}\n"
    prompt_injection += f"\nYou can reference the plot inline using `<img src=\"/api/workspace_files/{plot_filename}\" />`!"

    return {
        "success": True,
        "statistics": stats,
        "plot_filename": plot_filename,
        "plot_url": f"/api/workspace_files/{plot_filename}",
        "plot_b64": plot_b64,
        "prompt_injection": prompt_injection
    }


# ── 6. BOOTSTRAP TBOX MACRO ─────────────────────────────────────────────────

def tool_bootstrap_tbox_from_database(
    file_name: str
) -> dict:
    """
    Scans a database file (SQLite, Excel, CSV) and bootstraps a clean ontological schema (TBox).
    Maps physical tables to classes and columns to properties/data-relations.

    Args:
        file_name (str): The filename of the DB, CSV, or Excel file in the workspace.
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    ext = file_path.suffix.lower()
    tbox = {
        "ontology_name": f"{file_path.stem}_tbox",
        "classes": {},
        "properties": {}
    }

    try:
        if ext in (".db", ".sqlite", ".sqlite3"):
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
            
            for table in tables:
                class_name = f"s:{table.capitalize().rstrip('s')}"
                tbox["classes"][class_name] = {
                    "physical_table": table,
                    "description": f"Class representing entities inside table {table}"
                }
                
                cursor.execute(f'PRAGMA table_info("{table}")')
                columns = cursor.fetchall()
                for col in columns:
                    col_name = col[1]
                    col_type = col[2] or "TEXT"
                    prop_name = f"s:has_{col_name}"
                    tbox["properties"][prop_name] = {
                        "domain": class_name,
                        "range": col_type,
                        "physical_column": col_name
                    }
            conn.close()
        else:
            df, resolved_table = _load_data_source(file_path)
            class_name = f"s:{resolved_table.capitalize().rstrip('s')}"
            tbox["classes"][class_name] = {
                "physical_table": resolved_table,
                "description": f"Class representing entities inside dataset {resolved_table}"
            }
            for col in df.columns:
                prop_name = f"s:has_{col.replace(' ', '_').lower()}"
                tbox["properties"][prop_name] = {
                    "domain": class_name,
                    "range": str(df[col].dtype),
                    "physical_column": col
                }

        # Write bootstrapped TBox to a schema file on disk
        tbox_file = workspace_dir / f"{file_path.stem}_tbox.json"
        with open(tbox_file, "w", encoding="utf-8") as f:
            json.dump(tbox, f, indent=2)

        return {
            "success": True,
            "tbox": tbox,
            "tbox_file": tbox_file.name,
            "message": f"Successfully bootstrapped TBox ontology from {file_name}. Saved as {tbox_file.name}."
        }

    except Exception as e:
        return {"success": False, "error": f"TBox bootstrapping failed: {e}"}


# ── 7. CONVERT TO ABOX MACRO ────────────────────────────────────────────────

def tool_convert_to_abox(
    file_name: str,
    tbox_file_name: str
) -> dict:
    """
    Reads a database, parses rows based on a TBox schema, and compiles them into
    logical engram assertions (ABox) in the discussion's long-term memory.

    Args:
        file_name (str): Filename of the source DB, CSV, or Excel file.
        tbox_file_name (str): Filename of the TBox schema file (bootstrapped or custom).
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    db_file_path = (workspace_dir / file_name).resolve()
    tbox_path = (workspace_dir / tbox_file_name).resolve()

    if not db_file_path.exists() or not tbox_path.exists():
        return {"success": False, "error": "Database or TBox file does not exist in workspace."}

    if not discussion_instance or not discussion_instance.memory_manager:
        return {"success": False, "error": "No active discussion or memory manager found."}

    try:
        with open(tbox_path, "r", encoding="utf-8") as f:
            tbox = json.load(f)

        mm = discussion_instance.memory_manager
        nodes_created = 0
        relationships_created = 0

        # Process each class defined in the TBox
        for class_name, class_meta in tbox.get("classes", {}).items():
            table = class_meta.get("physical_table")
            if not table:
                continue

            # Load table rows
            try:
                df, _ = _load_data_source(db_file_path, table)
            except Exception:
                continue

            # Filter properties belonging to this class
            class_props = {
                k: v for k, v in tbox.get("properties", {}).items()
                if v.get("domain") == class_name
            }

            # Map rows to ABox engrams
            for idx, row in df.head(50).iterrows(): # Limit ABox inserts to first 50 rows for safety
                pk_val = str(row.iloc[0]) if len(row) > 0 else str(idx)
                subject_id = f"{class_name.split(':')[-1]}_{pk_val}".lower()

                # Build descriptive content block
                content_parts = [f"Entity type: {class_name} (ID: {subject_id})"]
                for prop_name, prop_meta in class_props.items():
                    col = prop_meta.get("physical_column")
                    if col in df.columns and pd.notnull(row[col]):
                        content_parts.append(f"  • {prop_name.split(':')[-1]}: {row[col]}")

                content_str = "\n".join(content_parts)

                # Add to persistent memory database
                mem = mm.add(
                    content=content_str,
                    importance=0.75,
                    tags=["abox_assertion", class_name.split(':')[-1]],
                    subject_group=class_name.split(':')[-1].lower(),
                    level=2,
                    subject=subject_id,
                    predicate="RELATED_TO",
                    obj=class_name.split(':')[-1].lower()
                )
                nodes_created += 1

                # Relate to general class node
                if mem:
                    mm.add_relationship(
                        source_id=mem["id"],
                        target_id=mem["id"],
                        relationship_type="RELATED_TO",
                        weight=1.0
                    )
                    relationships_created += 1

        discussion_instance.commit()

        return {
            "success": True,
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "message": f"ABox Translation Complete. Created {nodes_created} semantic engrams and {relationships_created} relationship edges."
        }

    except Exception as e:
        return {"success": False, "error": f"ABox Conversion Failed: {e}"}
