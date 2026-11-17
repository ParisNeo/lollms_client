import sqlite3
from pathlib import Path
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute SQL Query"
TOOL_LIBRARY_DESC = "Executes standard SQL queries on dataset files in the workspace."
TOOL_LIBRARY_ICON = "🗄️"

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages(["pandas", "openpyxl", "sqlalchemy"])
    global pd
    import pandas as pd

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
    import re
    import pandas as pd

    sql_query = str(sql_query).strip()

    if not file_name:
        db_files = list(Path(".").glob("*.db")) + list(Path(".").glob("*.sqlite")) + list(Path(".").glob("*.csv"))
        if db_files:
            file_name = db_files[0].name
        else:
            return {"success": False, "error": "No database file specified and none found in workspace."}

    file_path = Path(file_name)
    if not file_path.exists():
        return {"success": False, "error": f"File '{file_name}' not found in workspace."}

    conn = sqlite3.connect(":memory:")
    ext = file_path.suffix.lower()

    try:
        if ext == ".sqlconn":
            import json
            from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect
            with open(file_path, "r", encoding="utf-8") as f:
                conn_info = json.load(f)

            dialect = conn_info.get("dialect", "sqlite").lower()
            connection_url = conn_info.get("url", "")

            if not connection_url:
                if dialect == "sqlite":
                    db_path = conn_info.get("database", "")
                    db_path = db_path.replace("\\", "/")
                    connection_url = f"sqlite:///{db_path}"
                else:
                    host = conn_info.get("host", "localhost")
                    port = conn_info.get("port", "")
                    username = conn_info.get("username", "")
                    password = conn_info.get("password", "")
                    database = conn_info.get("database", "")
                    port_str = f":{port}" if port else ""
                    if dialect == "mysql":
                        connection_url = f"mysql+pymysql://{username}:{password}@{host}{port_str}/{database}"
                    elif dialect == "postgresql":
                        connection_url = f"postgresql+psycopg2://{username}:{password}@{host}{port_str}/{database}"
                    else:
                        return {"success": False, "error": f"Unsupported dialect: {dialect}"}

            engine = create_engine(connection_url)
            tables = sqlalchemy_inspect(engine).get_table_names()

            with engine.connect() as connection:
                for table in tables:
                    query_str = f'SELECT * FROM "{table}"' if dialect != "mysql" else f'SELECT * FROM `{table}`'
                    res = connection.execute(text(query_str))
                    cols = list(res.keys())
                    rows = res.fetchall()
                    df = pd.DataFrame(rows, columns=cols)
                    df.to_sql(table, conn, index=False, if_exists="replace")
            engine.dispose()
            
        elif ext in (".db", ".sqlite", ".sqlite3"):
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
            try:
                output_md = df_res.to_markdown(index=False)
            except (ImportError, ModuleNotFoundError):
                from lollms_client.lollms_artefact.data_files import _dataframe_to_markdown
                output_md = _dataframe_to_markdown(df_res)
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