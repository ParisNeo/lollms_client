from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

# ── pipmaster integration for optional dependencies ─────────────────────────
try:
    import pipmaster as pm
    _PM_AVAILABLE = True
except ImportError:
    _PM_AVAILABLE = False


def _ensure_installed(
    package_name: str,
    import_name: Optional[str] = None,
    package_version: Optional[str] = None,
) -> None:
    """
    Ensure an optional package is installed using pipmaster.
    Falls back to a plain import check if pipmaster is not available.
    """
    if _PM_AVAILABLE:
        if package_version:
            pm.ensure_packages({package_name: package_version})
        else:
            pm.ensure_packages(package_name)
    else:
        # Try a plain import; let the caller handle ImportError
        __import__(import_name or package_name)


def _get_sqlalchemy_engine_from_file(file_path: Path) -> Tuple[Any, str]:
    """
    Parses a '.sqlconn' connection metadata file and creates a SQLAlchemy engine.
    Installs required driver packages dynamically on demand via pipmaster.
    """
    import json
    _ensure_installed("sqlalchemy")
    from sqlalchemy import create_engine

    with open(file_path, "r", encoding="utf-8") as f:
        conn_info = json.load(f)

    dialect = conn_info.get("dialect", "mysql").lower().strip()
    host = conn_info.get("host", "localhost").strip()
    port = conn_info.get("port")
    username = conn_info.get("username", "").strip()
    password = conn_info.get("password", "").strip()
    database = conn_info.get("database", "").strip()
    connection_url = conn_info.get("url", "").strip()

    if dialect == "mysql":
        _ensure_installed("pymysql")
        if not connection_url:
            port_str = f":{port}" if port else ""
            connection_url = f"mysql+pymysql://{username}:{password}@{host}{port_str}/{database}"
    elif dialect == "postgresql":
        _ensure_installed("psycopg2-binary", "psycopg2")
        if not connection_url:
            port_str = f":{port}" if port else ""
            connection_url = f"postgresql+psycopg2://{username}:{password}@{host}{port_str}/{database}"
    elif dialect == "oracle":
        _ensure_installed("oracledb")
        if not connection_url:
            port_str = f":{port}" if port else ""
            connection_url = f"oracle+oracledb://{username}:{password}@{host}{port_str}/{database}"
    elif dialect in ("mssql", "sqlserver"):
        _ensure_installed("pyodbc")
        if not connection_url:
            port_str = f":{port}" if port else ""
            connection_url = f"mssql+pyodbc://{username}:{password}@{host}{port_str}/{database}"
    elif dialect == "sqlite":
        if not connection_url:
            connection_url = f"sqlite:///{database}"

    # Handle connection timeouts to avoid hanging the main thread
    connect_args = {}
    if dialect not in ("oracle", "sqlite"):
        connect_args["connect_timeout"] = 5

    engine = create_engine(connection_url, connect_args=connect_args)
    return engine, dialect


def _dataframe_to_markdown(df: Any) -> str:
    """Convert a pandas DataFrame to a Markdown table."""
    try:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return ""
        # Replace newlines in cells with spaces to avoid breaking table format
        clean_df = df.copy()
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.replace("\n", " ").str.replace("\r", " ")
        return clean_df.to_markdown(index=False)
    except Exception:
        # Minimal fallback if pandas to_markdown fails
        lines = []
        # Header
        headers = [str(c) for c in df.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def _parse_data_file(path: Path, art_title: str, version: int = 1, progress_cb: Optional[Callable[[str], None]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    _ensure_installed("sqlalchemy")
    _ensure_installed("pandas")
    _ensure_installed("openpyxl")
    import pandas as pd

    ext = path.suffix.lower()
    schema_parts = [f"# Data Interface: {art_title}\n"]

    try:
        if ext == ".sqlconn":
            if progress_cb: progress_cb("Reading SQL connection details...")
            engine, dialect = _get_sqlalchemy_engine_from_file(path)
            
            from sqlalchemy import inspect, text
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            schema_parts.append(f"Format: Remote Relational Database ({dialect}) | Total Tables: {len(tables)}\n")
            
            for idx, table in enumerate(tables):
                if progress_cb: progress_cb(f"Inspecting table '{table}' ({idx+1}/{len(tables)})...")
                columns = inspector.get_columns(table)
                pk_cols = inspector.get_pk_constraint(table).get("constrained_columns", [])
                
                # Fetch row count
                row_count = 0
                try:
                    with engine.connect() as connection:
                        res = connection.execute(text(f'SELECT COUNT(*) FROM "{table}"' if dialect != "mysql" else f'SELECT COUNT(*) FROM `{table}`'))
                        row_count = res.scalar()
                except Exception:
                    pass

                schema_parts.append(f"## Table: {table}")
                schema_parts.append(f"- Total Rows: {row_count:,} | Columns: {len(columns)}")
                schema_parts.append("### Columns & Schema:")
                for col in columns:
                    pk_marker = " — PRIMARY KEY" if col["name"] in pk_cols else ""
                    nullable_marker = " (NULLABLE)" if col.get("nullable", True) else ""
                    schema_parts.append(f"  • {col['name']} ({col['type']}){pk_marker}{nullable_marker}")

                # Fetch a preview
                try:
                    query_str = f'SELECT * FROM "{table}" LIMIT 3' if dialect != "mysql" else f'SELECT * FROM `{table}` LIMIT 3'
                    raw_conn = engine.raw_connection()
                    try:
                        df = pd.read_sql_query(query_str, raw_conn)
                    finally:
                        raw_conn.close()
                    schema_parts.append("### Preview (First 3 Rows):")
                    schema_parts.append(df.to_markdown(index=False))
                except Exception as ex:
                    schema_parts.append(f"  (Failed to read table preview: {ex})")

                schema_parts.append("\n---\n")
            
            engine.dispose()
        elif ext in (".db", ".sqlite", ".sqlite3"):
            if progress_cb: progress_cb("Connecting to SQLite database...")
            import sqlite3
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # List all tables in the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            schema_parts.append(f"Format: SQLite Relational Database (.db) | Total Tables: {len(tables)}\n")

            for idx, table in enumerate(tables):
                if progress_cb: progress_cb(f"Analyzing table '{table}' ({idx+1}/{len(tables)})...")
                # Get table schema (columns and types) - quote table name to handle special chars
                cursor.execute(f'PRAGMA table_info("{table}");')
                columns_info = cursor.fetchall()
                # Get exact row count - quote table name
                cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                row_count = cursor.fetchone()[0]

                schema_parts.append(f"## Table: {table}")
                schema_parts.append(f"- Total Rows: {row_count:,} | Columns: {len(columns_info)}")
                schema_parts.append("### Columns & Schema:")
                for col in columns_info:
                    pk_marker = " — PRIMARY KEY" if col[5] else ""
                    schema_parts.append(f"  • {col[1]} ({col[2] or 'ANY'}){pk_marker}")

                # Fetch a quick markdown preview using pandas - quote table name
                try:
                    df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 3;', conn)
                    schema_parts.append("### Preview (First 3 Rows):")
                    schema_parts.append(df.to_markdown(index=False))
                except Exception as ex:
                    schema_parts.append(f"  (Failed to read table preview: {ex})")

                schema_parts.append("\n---\n")
            conn.close()

        elif ext in (".xlsx", ".xls"):
            if progress_cb: progress_cb("Reading Excel sheets...")
            xl = pd.ExcelFile(str(path))
            sheets = xl.sheet_names
            schema_parts.append(f"Format: Excel (.xlsx) | Total Sheets: {len(sheets)}\n")

            for idx, sheet in enumerate(sheets):
                if progress_cb: progress_cb(f"Analyzing sheet '{sheet}' ({idx+1}/{len(sheets)})...")
                df = pd.read_excel(str(path), sheet_name=sheet, nrows=5)
                full_df = pd.read_excel(str(path), sheet_name=sheet)
                row_count = len(full_df)

                schema_parts.append(f"## Sheet: {sheet}")
                schema_parts.append(f"- Total Rows: {row_count:,} | Columns: {len(df.columns)}")
                schema_parts.append("### Columns & Types:")
                for col in full_df.columns:
                    dtype = str(full_df[col].dtype)
                    nulls = int(full_df[col].isnull().sum())
                    schema_parts.append(f"  • {col} ({dtype}) — {nulls} missing values")

                numeric_cols = full_df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    schema_parts.append("### Numeric Column Statistics:")
                    stats_df = full_df[numeric_cols].describe().loc[["min", "max", "mean"]]
                    schema_parts.append(stats_df.to_markdown())

                schema_parts.append("### Preview (First 3 Rows):")
                schema_parts.append(df.head(3).to_markdown(index=False))
                schema_parts.append("\n---\n")

        else:
            if progress_cb: progress_cb("Reading CSV headers...")
            sep = ","
            if ext in (".tsv", ".tab"):
                sep = "\t"
            else:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        line = f.readline()
                        if ";" in line: sep = ";"
                        elif "\t" in line: sep = "\t"
                except Exception:
                    pass

            df = pd.read_csv(str(path), sep=sep, nrows=5)
            full_df = pd.read_csv(str(path), sep=sep)
            row_count = len(full_df)

            schema_parts.append(f"Format: CSV (.csv) | Separator: {repr(sep)}\n")
            schema_parts.append(f"- Total Rows: {row_count:,} | Columns: {len(df.columns)}")
            schema_parts.append("### Columns & Types:")
            for col in full_df.columns:
                dtype = str(full_df[col].dtype)
                nulls = int(full_df[col].isnull().sum())
                schema_parts.append(f"  • {col} ({dtype}) — {nulls} missing values")

            numeric_cols = full_df.select_dtypes(include=["number"]).columns
            if not numeric_cols.empty:
                schema_parts.append("### Numeric Column Statistics:")
                stats_df = full_df[numeric_cols].describe().loc[["min", "max", "mean"]]
                schema_parts.append(stats_df.to_markdown())

            schema_parts.append("### Preview (First 3 Rows):")
            schema_parts.append(df.head(3).to_markdown(index=False))

    except Exception as e:
        ASCIIColors.error(f"Failed to parse structured data file: {e}")
        schema_parts.append(f"⚠️ Failed to extract full structure: {e}")

    # Copy the raw file to the data workspace so our execution tools can find it
    workspace_dir = Path("./data_workspace")
    try:
        from lollms_client.app.server import APP_WORKSPACE_DIR as awd
        if awd is not None:
            workspace_dir = awd
    except ImportError:
        pass

    workspace_dir.mkdir(exist_ok=True)
    # Save with unique title and version suffix
    shutil_dest = workspace_dir / f"{art_title}_v{version}{ext}"
    shutil_active = workspace_dir / f"{art_title}{ext}"
    import shutil
    if path.resolve() != shutil_dest.resolve():
        shutil.copy(str(path), str(shutil_dest))
    try:
        shutil.copy(str(path), str(shutil_active))
    except Exception as e:
        ASCIIColors.warning(f"Failed to copy active unversioned file: {e}")
    ASCIIColors.info(f"Raw data file saved to workspace: {shutil_dest}")

    return "\n\n".join(schema_parts), []
