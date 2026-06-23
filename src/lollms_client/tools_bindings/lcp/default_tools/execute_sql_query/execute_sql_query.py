import sqlite3
import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute SQL Query"
TOOL_LIBRARY_DESC = "Executes standard SQL queries on dataset files in the workspace."
TOOL_LIBRARY_ICON = "🗄️"

def init_tool_library() -> None:
    import pipmaster as pm
    pm.ensure_packages(["pandas", "openpyxl"])

def tool_execute_sql_query(
    sql_query: str = "",
    file_name: str = ""
) -> Dict[str, Any]:
    """
    Execute standard SQL queries on dataset files in the workspace.

    Files are accessed using simple relative paths since CWD is set to workspace.

    Args:
        sql_query (str, optional): The standard SQL query (SQLite syntax) to run.
        file_name (str, optional): The filename of the database/CSV/Excel file. Defaults to auto-detect.
    """
    sql_query = str(sql_query).strip()

    # Auto-detect file if not specified
    if not file_name:
        db_files = list(Path(".").glob("*.db")) + list(Path(".").glob("*.sqlite")) + list(Path(".").glob("*.csv"))
        if db_files:
            file_name = db_files[0].name
        else:
            return {"success": False, "error": "No database file specified and none found in workspace."}

    file_path = Path(file_name)
    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found in workspace."}

    # Load into in-memory SQLite
    conn = sqlite3.connect(":memory:")
    ext = file_path.suffix.lower()

    try:
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
    except Exception as e:
        conn.close()
        return {"success": False, "error": f"Failed to load dataset: {e}"}

    try:
        clean_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE).strip()
        clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL).strip()
        is_select = clean_query.lower().startswith("select")

        if is_select:
            df_res = pd.read_sql_query(sql_query, conn)
            output_md = df_res.to_markdown(index=False)
            conn.close()
            return {"success": True, "output": output_md}
        else:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            conn.commit()
            conn.close()
            return {"success": True, "output": f"Query executed. Affected rows: {cursor.rowcount}"}
    except Exception as e:
        conn.close()
        return {"success": False, "error": f"SQL execution error: {e}"}
