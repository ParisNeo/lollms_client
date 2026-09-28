"""
file_organizer.py — LCP tool for automated workspace file reorganization and migration.
Reads structured migration plans in YAML or JSON (with Markdown table fallback)
and executes directory creation and file/folder moving or copying safely.
"""
from __future__ import annotations

import os
import re
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "File Organizer"
TOOL_LIBRARY_DESC = "Executes workspace file reorganization and directory restructuring from YAML or JSON migration plans."
TOOL_LIBRARY_ICON = "📁"


def init_tools_library() -> None:
    pass


def _parse_plan_content(content: str, file_path: Path) -> List[Tuple[str, str, str]]:
    """
    Parses migration mappings from YAML, JSON, or Markdown tables.
    Returns a list of tuples: (source_path, target_path, description).
    """
    mappings: List[Tuple[str, str, str]] = []
    ext = file_path.suffix.lower()

    # 1. Try YAML / JSON structured parsing first
    parsed_data = None
    if ext in (".yaml", ".yml", ".json"):
        try:
            import yaml
            parsed_data = yaml.safe_load(content)
        except Exception:
            try:
                parsed_data = json.loads(content)
            except Exception:
                pass

    if parsed_data is not None:
        # Format A: {"mappings": [{"source": ..., "target": ...}, ...]}
        # Format B: {"files": [{"source": ..., "target": ...}, ...]}
        # Format C: [{"source": ..., "target": ...}, ...]
        raw_list = []
        if isinstance(parsed_data, dict):
            raw_list = parsed_data.get("mappings") or parsed_data.get("files") or parsed_data.get("plan")
            if raw_list is None:
                # Format D: Key-value dictionary {"source_path": "target_path"}
                for k, v in parsed_data.items():
                    if isinstance(v, str):
                        mappings.append((str(k).strip(), str(v).strip(), ""))
                    elif isinstance(v, dict):
                        tgt = v.get("target") or v.get("destination") or v.get("dest")
                        desc = v.get("description") or v.get("notes") or ""
                        if tgt:
                            mappings.append((str(k).strip(), str(tgt).strip(), str(desc).strip()))
                if mappings:
                    return mappings
        elif isinstance(parsed_data, list):
            raw_list = parsed_data

        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict):
                    src = item.get("source") or item.get("src") or item.get("from") or item.get("source_path")
                    dst = item.get("target") or item.get("dest") or item.get("to") or item.get("target_path") or item.get("destination")
                    desc = item.get("description") or item.get("notes") or item.get("note") or ""
                    if src and dst:
                        mappings.append((str(src).strip(), str(dst).strip(), str(desc).strip()))

    if mappings:
        return mappings

    # 2. Fallback: Parse Markdown tables (e.g. from mapping.md or YAML frontmatter)
    table_row_pattern = re.compile(r'^[ \t]*\|\s*`?([^`|\n]+)`?\s*\|\s*[^|\n]*\|\s*`?([^`|\n]+)`?\s*\|(?:\s*([^|\n]*)\|)?', re.MULTILINE)
    for m in table_row_pattern.finditer(content):
        src = m.group(1).strip().strip("`")
        dst = m.group(2).strip().strip("`")
        desc = (m.group(3) or "").strip().strip("`")

        # Skip table headers and separator rows
        if not src or not dst or src.lower() in ("source", "source path", "file", "---", ":---", ":---:"):
            continue
        if dst.lower() in ("target", "target path", "target class path", "destination", "---", ":---", ":---:"):
            continue
        if re.match(r'^[-: ]+$', src) or re.match(r'^[-: ]+$', dst):
            continue

        mappings.append((src, dst, desc))

    return mappings


def tool_organize_files_from_plan(
    plan_file: str = "mapping.yaml",
    move_files: bool = True
) -> Dict[str, Any]:
    """
    Executes a batch file and directory reorganization according to a structured plan file (YAML, JSON, or Markdown).
    Creates target directories automatically, safely moves or copies items, and handles atomic directory trees.

    Args:
        plan_file (str): Path to the migration plan file (e.g. 'mapping.yaml', 'mapping.json', 'mapping.md'). Defaults to 'mapping.yaml'.
        move_files (bool, optional): If True (default), moves the items. If False, copies items preserving the originals.
    """
    plan_path = Path(plan_file).resolve()
    if not plan_path.exists():
        cwd_candidate = (Path.cwd() / plan_file).resolve()
        if cwd_candidate.exists():
            plan_path = cwd_candidate
        else:
            return {
                "success": False,
                "error": f"Migration plan file '{plan_file}' was not found in the workspace."
            }

    try:
        content = plan_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as ex:
        return {"success": False, "error": f"Failed to read plan file '{plan_file}': {ex}"}

    mappings = _parse_plan_content(content, plan_path)
    if not mappings:
        return {
            "success": False,
            "error": f"No valid file migration mappings found in '{plan_file}'. Ensure the plan contains YAML/JSON mappings or a Markdown table."
        }

    ws_root = Path.cwd().resolve()
    action_verb = "Moved" if move_files else "Copied"

    executed_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    skipped: List[str] = []

    for src_raw, dst_raw, desc in mappings:
        src_clean = src_raw.replace("\\", "/").strip().lstrip("./")
        dst_clean = dst_raw.replace("\\", "/").strip().lstrip("./")

        if not src_clean or not dst_clean:
            continue

        src_path = (ws_root / src_clean).resolve()
        dst_path = (ws_root / dst_clean).resolve()

        # Path traversal guard
        if not str(src_path).startswith(str(ws_root)) or not str(dst_path).startswith(str(ws_root)):
            errors.append(f"Blocked path traversal attempt: '{src_raw}' -> '{dst_raw}'")
            continue

        if not src_path.exists():
            skipped.append(f"Source not found: '{src_clean}'")
            continue

        if src_path == dst_path:
            skipped.append(f"Source and destination are identical: '{src_clean}'")
            continue

        try:
            # Create target parent directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if move_files:
                # If target directory already exists for a directory move, merge contents
                if src_path.is_dir() and dst_path.exists():
                    for root, dirs, files in os.walk(src_path):
                        rel_root = Path(root).relative_to(src_path)
                        sub_dst = dst_path / rel_root
                        sub_dst.mkdir(parents=True, exist_ok=True)
                        for f in files:
                            shutil.move(os.path.join(root, f), str(sub_dst / f))
                    shutil.rmtree(src_path)
                else:
                    shutil.move(str(src_path), str(dst_path))
            else:
                if src_path.is_dir():
                    shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(src_path), str(dst_path))

            executed_items.append({
                "source": src_clean,
                "target": dst_clean,
                "action": action_verb.lower(),
                "description": desc
            })
            ASCIIColors.info(f"[FileOrganizer] {action_verb}: '{src_clean}' -> '{dst_clean}'")

        except Exception as op_err:
            err_msg = f"Failed to {action_verb.lower()} '{src_clean}' -> '{dst_clean}': {op_err}"
            errors.append(err_msg)
            ASCIIColors.warning(f"[FileOrganizer] {err_msg}")

    success = len(executed_items) > 0 and len(errors) == 0

    report_lines = [
        f"### 📁 Workspace Migration Report",
        f"- **Plan File**: `{plan_file}`",
        f"- **Action**: {action_verb}",
        f"- **Total Planned**: {len(mappings)}",
        f"- **Successfully {action_verb}**: {len(executed_items)}",
        f"- **Skipped (missing/same)**: {len(skipped)}",
        f"- **Errors**: {len(errors)}",
        ""
    ]

    if executed_items:
        report_lines.append(f"#### {action_verb} Items:")
        for item in executed_items[:50]:
            report_lines.append(f"- `{item['source']}` ➔ `{item['target']}`")
        if len(executed_items) > 50:
            report_lines.append(f"... (+{len(executed_items) - 50} more items)")

    if errors:
        report_lines.append("\n#### ❌ Errors Encountered:")
        for err in errors:
            report_lines.append(f"- {err}")

    if skipped and not executed_items:
        report_lines.append("\n#### ⚠️ Skipped Items:")
        for sk in skipped[:20]:
            report_lines.append(f"- {sk}")

    output_text = "\n".join(report_lines)

    return {
        "success": success or (len(executed_items) > 0),
        "total_planned": len(mappings),
        "executed_count": len(executed_items),
        "errors_count": len(errors),
        "skipped_count": len(skipped),
        "output": output_text,
        "executed_items": executed_items,
        "errors": errors,
        "skipped": skipped
    }