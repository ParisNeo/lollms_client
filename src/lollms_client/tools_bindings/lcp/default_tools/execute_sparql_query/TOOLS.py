"""
execute_sparql_query.py
=======================
LCP toolset for RDF graph management: parses Turtle (.ttl)/OWL files and executes
fully compliant SPARQL 1.1 read queries (SELECT, ASK, CONSTRUCT, DESCRIBE) and
SPARQL 1.1 Update operations (INSERT/DELETE DATA, INSERT/DELETE WHERE), persisting
mutations back to disk.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

TOOL_LIBRARY_NAME = "SPARQL_QUERY_RUNNER"
TOOL_LIBRARY_DESC = "Parses local Turtle (.ttl)/OWL RDF graphs, executes fully compliant SPARQL 1.1 queries and SPARQL 1.1 Update operations, and persists graph mutations to disk."
TOOL_LIBRARY_ICON = "🕸️"

def init_tools_library() -> None:
    """Ensure rdflib is installed and expose it at module scope."""
    import pipmaster as pm
    pm.ensure_packages("rdflib")
    global rdflib
    import rdflib


def _get_workspace_dir() -> Path:
    """
    🛑 TOOLS ARE AGNOSTIC: Rely on CWD set by orchestrator.
    Fallback to ./data_workspace for standalone execution.
    """
    cwd = Path.cwd()
    if (cwd / "data_workspace").exists() or cwd.name == "data_workspace":
        return cwd
    return Path("./data_workspace").resolve()


def _resolve_graph_path(file_name: str) -> Optional[Path]:
    """Safely resolves an RDF file path inside the workspace sandbox, blocking path traversal."""
    workspace_dir = _get_workspace_dir()
    if not file_name:
        return None
    clean = file_name.replace("\\", "/").lstrip("/")
    if ".." in Path(clean).parts:
        return None
    candidate = (workspace_dir / clean).resolve()
    try:
        candidate.relative_to(workspace_dir.resolve())
    except ValueError:
        return None
    return candidate


def _detect_update(sparql_text: str) -> bool:
    """Detects whether the given SPARQL text is a SPARQL 1.1 Update operation."""
    normalized = " ".join((sparql_text or "").split()).upper()
    update_keywords = (
        "INSERT DATA", "DELETE DATA", "INSERT WHERE",
        "DELETE WHERE", "DELETE {", "INSERT {", "LOAD ", "CLEAR ", "LOAD<SILENT>",
        "CREATE ", "DROP ", "COPY ", "MOVE ", "ADD ",
    )
    return any(normalized.startswith(kw) or f" {kw}" in normalized for kw in update_keywords)


def _load_graph(file_path: Path) -> "rdflib.Graph":
    """Loads (or initializes) the RDF graph from the given path."""
    g = rdflib.Graph()
    if file_path.exists():
        ext = file_path.suffix.lower()
        rdf_format = "turtle" if ext == ".ttl" else ("xml" if ext in (".rdf", ".xml", ".owl") else "turtle")
        g.parse(str(file_path), format=rdf_format)
    return g


def _serialize_graph(g: "rdflib.Graph", output_path: Path) -> None:
    """Serializes the graph to disk in Turtle format, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_path), format="turtle")


def tool_execute_sparql_query(
    file_name: str = "",
    sparql_query: str = "",
    update_query: str = ""
) -> dict:
    """
    Parses a Turtle (.ttl) or OWL (.owl) file from the workspace and executes a SPARQL 1.1 read query.

    Args:
        file_name (str, optional): The filename of the .ttl or .owl file in the workspace. Auto-discovers if omitted.
        sparql_query (str, optional): The valid SPARQL 1.1 read query to execute (SELECT, ASK, CONSTRUCT, DESCRIBE).
        update_query (str, optional): Alias for sparql_query. If the text is a SPARQL Update operation, it is automatically routed to the update executor.
    """
    from lollms_client.lollms_artefact.data_files import _dataframe_to_markdown
    import pandas as pd

    query_text = sparql_query or update_query
    if not query_text:
        return {
            "success": False,
            "error": "No SPARQL query provided. Use 'sparql_query' (or 'update_query') with a valid SPARQL 1.1 expression."
        }

    if _detect_update(query_text):
        return tool_execute_sparql_update(
            file_name=file_name,
            update_query=query_text
        )

    workspace_dir = _get_workspace_dir()

    if not file_name:
        ontology_files = list(workspace_dir.glob("*.ttl")) + list(workspace_dir.glob("*.owl"))
        if ontology_files:
            file_name = ontology_files[0].name
        else:
            return {
                "success": False,
                "error": "No file_name provided and no .ttl or .owl files found in workspace."
            }

    file_path = _resolve_graph_path(file_name)
    if file_path is None:
        return {
            "success": False,
            "error": f"Invalid or unsafe file path: '{file_name}'."
        }

    if not file_path.exists():
        return {
            "success": False,
            "error": f"Ontology file '{file_name}' not found in workspace."
        }

    try:
        g = _load_graph(file_path)
    except Exception as parse_err:
        return {
            "success": False,
            "error": f"Failed to parse RDF graph: {parse_err}"
        }

    try:
        query_res = g.query(query_text)

        if query_res.type == "SELECT":
            variables = [str(var) for var in query_res.vars]
            rows = []
            for row in query_res:
                row_dict = {}
                for idx, var in enumerate(query_res.vars):
                    val = row[idx]
                    row_dict[str(var)] = str(val) if val is not None else None
                rows.append(row_dict)

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
            triples = len(query_res)
            serialized = query_res.serialize(format="turtle")
            if isinstance(serialized, bytes):
                serialized = serialized.decode("utf-8", errors="ignore")
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


def tool_execute_sparql_update(
    file_name: str = "",
    update_query: str = "",
    sparql_query: str = "",
    create_file_if_missing: bool = True
) -> dict:
    """
    Executes a SPARQL 1.1 Update operation (INSERT DATA, DELETE DATA, INSERT/DELETE WHERE) against a local RDF graph, persisting the mutation to disk.

    Args:
        file_name (str, optional): The filename of the .ttl or .owl file in the workspace. Auto-discovers or auto-creates if omitted.
        update_query (str): The valid SPARQL 1.1 Update operation to execute.
        sparql_query (str, optional): Alias for update_query (tolerated for name-drifting LLMs).
        create_file_if_missing (bool, optional): If True (default), creates and initializes the ontology file when it does not exist yet.
    """
    query_text = update_query or sparql_query
    if not query_text:
        return {
            "success": False,
            "error": "No SPARQL update provided. Use 'update_query' with a valid SPARQL 1.1 Update operation."
        }

    if not _detect_update(query_text):
        return {
            "success": False,
            "error": "The provided text is not a SPARQL 1.1 Update operation. Use tool_execute_sparql_query for SELECT/ASK/CONSTRUCT queries."
        }

    workspace_dir = _get_workspace_dir()

    file_created = False
    if not file_name:
        ontology_files = list(workspace_dir.glob("*.ttl"))
        if ontology_files:
            file_name = ontology_files[0].name
        else:
            file_name = "ontology.ttl"
            file_created = True

    file_path = _resolve_graph_path(file_name)
    if file_path is None:
        return {
            "success": False,
            "error": f"Invalid or unsafe file path: '{file_name}'."
        }

    if not file_path.exists():
        if not create_file_if_missing:
            return {
                "success": False,
                "error": f"Ontology file '{file_name}' not found in workspace."
            }
        file_created = True

    try:
        g = _load_graph(file_path)
    except Exception as parse_err:
        return {
            "success": False,
            "error": f"Failed to parse RDF graph: {parse_err}"
        }

    triples_before = len(g)

    try:
        g.update(query_text)
    except Exception as update_err:
        return {
            "success": False,
            "error": f"SPARQL Update Compilation/Execution Failed: {update_err}"
        }

    triples_after = len(g)

    try:
        _serialize_graph(g, file_path)
    except Exception as persist_err:
        return {
            "success": False,
            "error": f"Update executed but persisting the graph to disk failed: {persist_err}"
        }

    delta = triples_after - triples_before
    status_note = "created and initialized" if file_created else "updated"
    summary = (
        f"Ontology '{file_name}' {status_note}. "
        f"Triples: {triples_before:,} → {triples_after:,} ({'+' if delta >= 0 else ''}{delta:,})."
    )

    prompt_injection = (
        f"\n\n=== 🕸️ SPARQL UPDATE APPLIED ===\n"
        f"• File: `{file_name}`\n"
        f"• {summary}\n"
        f"• The graph has been persisted to disk. You can now query it with tool_execute_sparql_query.\n"
        f"=== END UPDATE RESULT ==="
    )

    return {
        "success": True,
        "type": "UPDATE",
        "file_name": file_name,
        "triples_before": triples_before,
        "triples_after": triples_after,
        "triples_delta": delta,
        "output": summary,
        "prompt_injection": prompt_injection
    }