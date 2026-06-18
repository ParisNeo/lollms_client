"""
semantic_data_engineer.py
=========================
A comprehensive data engineering, statistics, plotting, and semantic translation tool
for LOLLMS. Converts relational databases (SQL, CSV, Excel) into Ontological ABox Graphs.
"""

import os
import sys
import re
import json
import sqlite3
import base64
import io
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

TOOL_LIBRARY_NAME = "SEMANTIC_DATA_ENGINEER"
TOOL_LIBRARY_DESC = "Performs mathematical statistics, generates plots, bootstraps TBox schemas from databases, and compiles physical relational data into semantic ABox graph engrams."
TOOL_LIBRARY_ICON = "📊"

def init_tool_library() -> None:
    """Ensure required packages are available."""
    import pipmaster as pm
    pm.ensure_packages({
        "pandas": ">=1.3.0",
        "matplotlib": ">=3.4.0",
        "numpy": ">=1.20.0",
        "sqlalchemy": ">=1.4.0",
        "openpyxl": ">=3.0.0"
    })


def _get_workspace_dir() -> Path:
    workspace_dir = Path("./data_workspace")
    try:
        from lollms_client.app.server import APP_WORKSPACE_DIR
        if APP_WORKSPACE_DIR is not None:
            workspace_dir = APP_WORKSPACE_DIR
    except ImportError:
        pass
    return workspace_dir


def _load_data_source(file_path: Path, table_name: Optional[str] = None) -> Any:
    """Load a CSV, Excel, or SQLite file into a Pandas DataFrame."""
    import pandas as pd
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
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()
        return df, table_name
    elif ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(str(file_path))
        sheet = table_name or xl.sheet_names[0]
        df = pd.read_excel(str(file_path), sheet_name=sheet)
        return df, sheet
    else:
        sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
        df = pd.read_csv(str(file_path), sep=sep)
        return df, file_path.stem


def tool_compute_statistics_and_plot(
    file_name: str,
    table_name: Optional[str] = None,
    plot_type: str = "bar",  # "bar", "line", "scatter", "histogram"
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    color: str = "#4f46e5"  # Indigo default
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
    matplotlib.use('Agg')  # Prevents thread/GUI loop errors
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
        
        # Apply dark background theme to match app "vibes"
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

        # Handle plot mapping
        x_val = df[x_column] if x_column and x_column in df.columns else df.index
        y_val = df[y_column] if y_column and y_column in df.columns else (df[numeric_cols[0]] if numeric_cols else None)

        if y_val is None:
            return {"success": False, "error": "No numeric columns available to plot."}

        if plot_type == "bar":
            ax.bar(x_val, y_val, color=color, alpha=0.85, edgecolor='rgba(255,255,255,0.1)')
        elif plot_type == "line":
            ax.plot(x_val, y_val, color=color, linewidth=2, marker='o', markersize=4)
        elif plot_type == "scatter":
            ax.scatter(x_val, y_val, color=color, alpha=0.8, edgecolors='none', s=30)
        elif plot_type == "histogram":
            ax.hist(y_val, bins=15, color=color, alpha=0.8, edgecolor='rgba(255,255,255,0.1)')
            ax.set_xlabel(y_column or numeric_cols[0], color='#94a3b8')
            ax.set_ylabel("Frequency", color='#94a3b8')

        if x_column:
            ax.set_xlabel(x_column, color='#94a3b8')
        if y_column and plot_type != "histogram":
            ax.set_ylabel(y_column, color='#94a3b8')

        plt.tight_layout()

        # Save to buffer and disk
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
        f"• Plot Generated: [View Plot Image](/api/workspace_files/{plot_filename}) ({plot_type})\n"
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


def tool_convert_to_abox(
    file_name: str,
    tbox_file_name: str,
    discussion_id: str = "viewer_session"
) -> dict:
    """
    Reads a database, parses rows based on a TBox schema, and compiles them into
    logical engram assertions (ABox) in the discussion's long-term memory.

    Args:
        file_name (str): Filename of the source DB, CSV, or Excel file.
        tbox_file_name (str): Filename of the TBox schema file (bootstrapped or custom).
        discussion_id (str): Active discussion ID (default "viewer_session").
    """
    import pandas as pd
    workspace_dir = _get_workspace_dir()
    db_file_path = (workspace_dir / file_name).resolve()
    tbox_path = (workspace_dir / tbox_file_name).resolve()

    if not db_file_path.exists() or not tbox_path.exists():
        return {"success": False, "error": "Database or TBox file does not exist in workspace."}

    try:
        with open(tbox_path, "r", encoding="utf-8") as f:
            tbox = json.load(f)

        # Retrieve the discussion's persistent memory manager
        from lollms_client.app.server import discussion as active_discussion
        if not active_discussion or not active_discussion.memory_manager:
            return {"success": False, "error": "No active discussion or memory manager found."}

        mm = active_discussion.memory_manager
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
            for idx, row in df.iterrows():
                # Formulate unique Subject ID for this entity (TBox Class + Row Index / PK)
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
                    level=2,  # Store in Deep Memory (Level 2) as default ABox latent graph
                    subject=subject_id,
                    predicate="RELATED_TO",
                    obj=class_name.split(':')[-1].lower()
                )
                nodes_created += 1

                # If primary key / target exists, relate them to the general class
                if mem:
                    mm.add_relationship(
                        source_id=mem["id"],
                        target_id=mem["id"], # self loop or class link
                        relationship_type="RELATED_TO",
                        weight=1.0
                    )
                    relationships_created += 1

        active_discussion.commit()

        return {
            "success": True,
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "message": f"ABox Translation Complete. Created {nodes_created} semantic engrams and {relationships_created} relationship edges."
        }

    except Exception as e:
        return {"success": False, "error": f"ABox Conversion Failed: {e}"}
