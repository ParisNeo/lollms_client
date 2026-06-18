"""
execute_sparql_query.py
=======================
An LCP tool designed to parse Turtle (.ttl) files and execute fully compliant SPARQL 1.1 queries
using rdflib, returning clean JSON results and formatted Markdown tables.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

TOOL_LIBRARY_NAME = "SPARQL_QUERY_RUNNER"
TOOL_LIBRARY_DESC = "Parses local Turtle (.ttl) RDF graphs and executes fully compliant SPARQL 1.1 queries, returning structured bindings."
TOOL_LIBRARY_ICON = "🕸️"

def init_tool_library() -> None:
    """Ensure rdflib is installed."""
    import pipmaster as pm
    pm.ensure_packages("rdflib")


def _get_workspace_dir() -> Path:
    workspace_dir = Path("./data_workspace")
    try:
        from lollms_client.app.server import APP_WORKSPACE_DIR
        if APP_WORKSPACE_DIR is not None:
            workspace_dir = APP_WORKSPACE_DIR
    except ImportError:
        pass
    return workspace_dir


def tool_execute_sparql_query(
    file_name: str,
    sparql_query: str
) -> dict:
    """
    Parses a Turtle (.ttl) file from the workspace and executes a SPARQL 1.1 query.

    Args:
        file_name (str): The filename of the .ttl file in the workspace.
        sparql_query (str): The valid SPARQL 1.1 query to execute (SELECT, ASK, CONSTRUCT).
    """
    import rdflib
    from lollms_client.lollms_discussion._data_files import _dataframe_to_markdown
    import pandas as pd

    workspace_dir = _get_workspace_dir()
    file_path = (workspace_dir / file_name).resolve()

    if not file_path.exists():
        return {
            "success": False,
            "error": f"Turtle file '{file_name}' not found in workspace."
        }

    # 1. Parse RDF Graph
    g = rdflib.Graph()
    try:
        # Determine format based on extension
        ext = file_path.suffix.lower()
        rdf_format = "turtle" if ext == ".ttl" else ("xml" if ext in (".rdf", ".xml") else "turtle")
        g.parse(str(file_path), format=rdf_format)
    except Exception as parse_err:
        return {
            "success": False,
            "error": f"Failed to parse RDF graph: {parse_err}"
        }

    # 2. Execute SPARQL Query
    try:
        query_res = g.query(sparql_query)
        
        # 3. Format results based on query type
        if query_res.type == "SELECT":
            variables = [str(var) for var in query_res.vars]
            rows = []
            for row in query_res:
                row_dict = {}
                for idx, var in enumerate(query_res.vars):
                    val = row[idx]
                    # Convert rdflib terms to clean strings or primitives
                    row_dict[str(var)] = str(val) if val is not None else None
                rows.append(row_dict)

            # Build a polished Markdown table for the chat bubble
            df = pd.DataFrame(rows, columns=variables)
            md_table = _dataframe_to_markdown(df)

            prompt_injection = (
                f"\n\n=== 🕸️ SPARQL QUERY RESULTS ===\n"
                f"• Query Executed on `{file_name}`\n"
                f"• Matching Triples / Bindings Found: {len(rows):,}\n\n"
                f"{md_table}\n"
                f"=== END RESULTS ==="
            )

            return {
                "success": True,
                "type": "SELECT",
                "variables": variables,
                "rows": rows,
                "output": md_table,
                "prompt_injection": prompt_injection
            }

        elif query_res.type == "ASK":
            ans = bool(query_res.askAnswer)
            res_str = f"**ASK Result**: `{ans}`"
            return {
                "success": True,
                "type": "ASK",
                "answer": ans,
                "output": res_str,
                "prompt_injection": f"\n\n=== 🕸️ SPARQL ASK RESULT ===\n{res_str}\n"
            }

        elif query_res.type in ("CONSTRUCT", "DESCRIBE"):
            # Serialize returned graph back to Turtle
            triples = len(query_res)
            serialized = query_res.serialize(format="turtle")
            res_str = f"```turtle\n{serialized}\n```"
            return {
                "success": True,
                "type": str(query_res.type),
                "triples": triples,
                "output": res_str,
                "prompt_injection": f"\n\n=== 🕸️ SPARQL CONSTRUCT RESULT ({triples} triples) ===\n{res_str}\n"
            }

        else:
            return {
                "success": True,
                "type": "UNKNOWN",
                "output": "Query executed successfully (no bindings returned)."
            }

    except Exception as query_err:
        return {
            "success": False,
            "error": f"SPARQL Query Compilation/Execution Failed: {query_err}"
        }
