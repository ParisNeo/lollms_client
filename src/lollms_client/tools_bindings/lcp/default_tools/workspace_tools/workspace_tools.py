import os
import re
import fnmatch
from pathlib import Path
from typing import Dict, Any, List, Optional

TOOL_LIBRARY_NAME = "Workspace Tools"
TOOL_LIBRARY_DESC = "Tools for reading, writing, listing, finding, and grepping files in the agent's workspace."
TOOL_LIBRARY_ICON = "📁"

_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp",
    ".pdf", ".docx", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3",
    ".zip", ".tar", ".gz", ".mp3", ".wav", ".mp4", ".avi", ".pyc", ".pyo", ".so", ".dll",
}


def init_tools_library(config: dict = None) -> None:
    pass


def _resolve_safe_path(file_name: str) -> Path:
    """
    Sanitizes the file name and resolves it safely within the current working directory.
    Prevents path traversal attacks (e.g., ../../etc/passwd).
    """
    clean_name = file_name.replace("\\", "/")
    
    if len(clean_name) > 1 and clean_name[1] == ":":
        clean_name = clean_name[2:]
    clean_name = clean_name.lstrip("/")
    
    base_path = Path.cwd()
    target_path = (base_path / clean_name).resolve()
    
    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise PermissionError(f"Path traversal detected: '{file_name}' attempts to escape workspace.")
        
    return target_path


def tool_write_file(file_name: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file in the workspace. Creates parent directories if they do not exist.

    Args:
        file_name (str): The path of the file to write.
        content (str): The content to write into the file.
    """
    try:
        target_path = _resolve_safe_path(file_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return {"success": True, "output": f"File '{file_name}' written successfully."}
    except PermissionError as pe:
        return {"success": False, "error": str(pe)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Failed to write file '{file_name}': {str(e)}", "traceback": traceback.format_exc()}


def tool_read_file(file_name: str) -> Dict[str, Any]:
    """
    Read content from a file in the workspace.

    Args:
        file_name (str): The path of the file to read.
    """
    try:
        target_path = _resolve_safe_path(file_name)
        if not target_path.exists():
            return {"success": False, "error": f"File '{file_name}' not found."}
        if not target_path.is_file():
            return {"success": False, "error": f"Path '{file_name}' is not a file."}
            
        content = target_path.read_text(encoding="utf-8", errors="ignore")
        return {"success": True, "output": content}
    except PermissionError as pe:
        return {"success": False, "error": str(pe)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Failed to read file '{file_name}': {str(e)}", "traceback": traceback.format_exc()}


def tool_list_files(directory: str = ".") -> Dict[str, Any]:
    """
    List all files in a directory within the workspace.

    Args:
        directory (str, optional): Directory to list. Defaults to current directory ('.').
    """
    try:
        target_path = _resolve_safe_path(directory)
        if not target_path.exists():
            return {"success": False, "error": f"Directory '{directory}' not found."}
        if not target_path.is_dir():
            return {"success": False, "error": f"Path '{directory}' is not a directory."}
            
        files = []
        for p in target_path.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in _BINARY_EXTS:
                    continue
                rel_path = p.relative_to(Path.cwd())
                files.append(str(rel_path).replace("\\", "/"))
                
        files.sort()
        output_str = "\n".join(files) if files else "Directory is empty."
        return {"success": True, "files": files, "output": output_str}
    except PermissionError as pe:
        return {"success": False, "error": str(pe)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Failed to list files in '{directory}': {str(e)}", "traceback": traceback.format_exc()}


def tool_find_files(pattern: str, path: str = ".", max_results: int = 50) -> Dict[str, Any]:
    """
    Recursively searches for files matching a name pattern within a given directory.
    Uses shell wildcards (e.g., '*.py', 'config.*').

    Args:
        pattern (str): The file name pattern to match (supports * and ?).
        path (str): The directory to search in. Defaults to current directory.
        max_results (int): Maximum number of file paths to return. Defaults to 50.
    """
    try:
        search_dir = _resolve_safe_path(path)
        if not search_dir.is_dir():
            return {"success": False, "error": f"Directory '{path}' not found."}

        matches: List[str] = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    try:
                        rel_path = os.path.relpath(os.path.join(root, filename), Path.cwd())
                        matches.append(rel_path.replace("\\", "/"))
                    except ValueError:
                        pass
                        
            if len(matches) >= max_results:
                matches = matches[:max_results]
                break

        output_str = f"Found {len(matches)} file(s) matching '{pattern}':\n" + "\n".join(matches) if matches else f"No files found matching '{pattern}'."
        return {
            "success": True,
            "output": output_str,
            "files": matches
        }
    except Exception as e:
        return {"success": False, "error": f"File search failed: {str(e)}"}


def tool_grep_files(pattern: str, file_extension: Optional[str] = None, max_results: int = 50, case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Searches for a regex pattern across files in the current working directory.
    Useful for extracting specific data from large files without loading them fully into context.

    Args:
        pattern (str): The regular expression pattern to search for.
        file_extension (str, optional): Filter search to specific file extensions (e.g., '.csv', '.json'). If None, searches all text files.
        max_results (int): Maximum number of matching lines to return. Defaults to 50.
        case_sensitive (bool): Whether the search is case-sensitive. Defaults to False.
    """
    try:
        cwd = Path.cwd()
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        results: List[Dict[str, Any]] = []
        files_scanned = 0
        
        for file_path in cwd.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in _BINARY_EXTS:
                continue

            if file_extension:
                if not file_path.name.lower().endswith(file_extension.lower()):
                    continue

            files_scanned += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                "file": str(file_path.relative_to(cwd)).replace("\\", "/"),
                                "line_number": line_num,
                                "text": line.strip()[:500]
                            })
                            if len(results) >= max_results:
                                break
            except Exception:
                pass
                
            if len(results) >= max_results:
                break
                
        output_text = f"Found {len(results)} match(es) for pattern '{pattern}' across {files_scanned} file(s).\n\n"
        for res in results:
            output_text += f"[{res['file']}:{res['line_number']}] {res['text']}\n"
            
        return {
            "success": True,
            "matches_count": len(results),
            "files_scanned": files_scanned,
            "results": results,
            "output": output_text
        }
        
    except re.error as re_err:
        return {"success": False, "error": f"Invalid regex pattern: {re_err}"}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred during grep: {str(e)}"}