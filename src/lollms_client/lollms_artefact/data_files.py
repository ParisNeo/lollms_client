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
    """
    Parses a data file (CSV, Excel, SQLite, etc.) and returns a rich Markdown schema (.lam content)
    along with the raw physical bytes for tool execution.
    """
    # 🛑 CRITICAL: Force install dependencies BEFORE any import attempts
    try:
        _ensure_installed("pandas")
        _ensure_installed("openpyxl")
        _ensure_installed("sqlalchemy")
        import pandas as pd
        ASCIIColors.success(f"[DataFiles] ✅ Pandas/OpenPyXL successfully loaded for schema extraction.")
    except Exception as install_err:
        ASCIIColors.error(f"[DataFiles] ❌ CRITICAL: Failed to install/load pandas: {install_err}")
        # Fallback to minimal schema if pandas completely fails
        return f"# Data Interface: {art_title}\n\n⚠️ **Critical Error**: Pandas library unavailable. Cannot extract schema.", [], None

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

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            schema_parts.append(f"Format: SQLite Relational Database (.db) | Total Tables: {len(tables)}\n")

            for idx, table in enumerate(tables):
                if progress_cb: progress_cb(f"Analyzing table '{table}' ({idx+1}/{len(tables)})...")
                cursor.execute(f'PRAGMA table_info("{table}");')
                columns_info = cursor.fetchall()
                cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                row_count = cursor.fetchone()[0]

                schema_parts.append(f"## Table: {table}")
                schema_parts.append(f"- Total Rows: {row_count:,} | Columns: {len(columns_info)}")
                schema_parts.append("### Columns & Schema:")
                for col in columns_info:
                    pk_marker = " — PRIMARY KEY" if col[5] else ""
                    schema_parts.append(f"  • {col[1]} ({col[2] or 'ANY'}){pk_marker}")

                try:
                    df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 3;', conn)
                    schema_parts.append("### Preview (First 3 Rows):")
                    schema_parts.append(df.to_markdown(index=False))
                except Exception as ex:
                    schema_parts.append(f"  (Failed to read table preview: {ex})")

                schema_parts.append("\n---\n")
            conn.close()

        elif ext in (".ttl", ".rdf", ".xml"):
            if progress_cb: progress_cb("Parsing Turtle RDF graph...")
            _ensure_installed("rdflib")
            import rdflib
            g = rdflib.Graph()
            g.parse(str(path), format="turtle" if ext == ".ttl" else "xml")

            subjects = set(g.subjects())
            predicates = set(g.predicates())
            objects = set(g.objects())

            schema_parts.append(f"Format: Semantic Web RDF Graph ({ext.upper()}) | Total Triples: {len(g):,}\n")
            schema_parts.append(f"## Graph Summary")
            schema_parts.append(f"- Unique Subjects: {len(subjects):,}")
            schema_parts.append(f"- Unique Predicates (Properties): {len(predicates):,}")
            schema_parts.append(f"- Unique Objects: {len(objects):,}\n")

            schema_parts.append("### Active Namespace Bindings:")
            for prefix, ns in g.namespaces():
                if prefix:
                    schema_parts.append(f"  • PREFIX {prefix}: &lt;{ns}&gt;")

            schema_parts.append("\n### Unique Predicates / Relations List:")
            for pred in sorted(predicates):
                schema_parts.append(f"  • {pred}")

            schema_parts.append("\n### Sample Triples Preview (First 5):")
            sample_rows = []
            for s, p, o in list(g)[:5]:
                sample_rows.append(f"  • &lt;{s}&gt; &lt;{p}&gt; &lt;{o}&gt; .")
            schema_parts.append("\n".join(sample_rows))

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
            # DEFAULT CASE: CSV, TSV, or unknown text-based data
            if progress_cb: progress_cb(f"Reading CSV/Text data (Separator detection)...")
            sep = ","
            if ext in (".tsv", ".tab"):
                sep = "\t"
            else:
                try:
                    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                        first_line = f.readline()
                        if ";" in first_line and first_line.count(";") > first_line.count(","):
                            sep = ";"
                        elif "\t" in first_line:
                            sep = "\t"
                        else:
                            sep = ","
                    ASCIIColors.info(f"[DataFiles] Detected separator: {repr(sep)}")
                except Exception as det_err:
                    ASCIIColors.warning(f"[DataFiles] Separator detection failed: {det_err}. Defaulting to comma.")

            # Read with pandas
            try:
                df = pd.read_csv(str(path), sep=sep, nrows=5, encoding="utf-8-sig")
                full_df = pd.read_csv(str(path), sep=sep, encoding="utf-8-sig")
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
            except Exception as csv_err:
                ASCIIColors.error(f"[DataFiles] CSV parsing failed: {csv_err}")
                schema_parts.append(f"\n⚠️ **Parsing Error**: Could not read as CSV.\nError: {csv_err}")

    except Exception as e:
        ASCIIColors.error(f"Failed to parse structured data file: {e}")
        schema_parts.append(f"\n⚠️ **Critical Failure**: {e}")

    # ── DUAL-STREAM PROTOCOL FOR DATA FILES ───────────────────────────────────
    # 1. Logical Content (for LLM Context): The schema/stats generated above.
    # 2. Physical Content (for Tools): The raw binary file copied from source.

    # Read the raw binary data from the source file
    try:
        raw_physical_data = path.read_bytes()
        ASCIIColors.info(f"[DataFiles] Read {len(raw_physical_data):,} bytes of raw physical data from {path.name}")
    except Exception as e:
        ASCIIColors.error(f"Failed to read raw binary data from {path}: {e}")
        raw_physical_data = None

    # Return both: Schema for context, Raw Bytes for disk storage
    final_schema = "\n\n".join(schema_parts)
    ASCIIColors.success(f"[DataFiles] ✅ Schema generated successfully ({len(final_schema)} chars).")
    return final_schema, [], raw_physical_data
