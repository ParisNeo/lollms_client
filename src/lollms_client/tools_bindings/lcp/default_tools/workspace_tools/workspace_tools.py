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

_IGNORED_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv",
    "venv", ".idea", ".vscode", ".lollms", "build", "dist", ".next",
    "env", ".env", ".lollms_code", ".lollms_metadata", "egg-info",
    "dist-info", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "htmlcov", "site-packages", "artefacts_metadata",
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
    if isinstance(max_results, str):
        try:
            max_results = int(max_results)
        except ValueError:
            max_results = 50

    try:
        search_dir = _resolve_safe_path(path)
        if not search_dir.is_dir():
            return {"success": False, "error": f"Directory '{path}' not found."}

        matches: List[str] = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS and not d.startswith('.')]

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


def _calculate_fuzzy_score(line: str, keywords: List[str]) -> float:
    """
    Calculates a Jaccard-like similarity score between the line and the keywords.
    Returns a score between 0.0 and 1.0.
    """
    if not keywords:
        return 0.0

    line_words = set(re.findall(r'\b\w+\b', line.lower()))
    keyword_set = set(kw.lower() for kw in keywords)

    intersection = line_words.intersection(keyword_set)
    union = line_words.union(keyword_set)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def tool_grep_files(
    pattern: str, 
    file_extension: Optional[str] = None, 
    max_results: int = 50, 
    search_mode: str = "case_insensitive",
    sort_by_relevance: bool = True
) -> Dict[str, Any]:
    """
    Searches for a pattern across files in the current working directory.
    Supports exact regex, case-sensitive, case-insensitive, fuzzy (approximate), and words (subset) matching.

    Args:
        pattern (str): The search pattern. Can be a regex (if search_mode='exact') or plain text/keywords.
        file_extension (str, optional): Filter search to specific file extensions (e.g., '.csv', '.json'). If '*', searches all files. If None, searches text files.
        max_results (int): Maximum number of matching lines to return. Defaults to 50.
        search_mode (str): The search strategy. Options: 
            - 'exact': Treats pattern as a strict regular expression.
            - 'case_sensitive': Literal string match, case-sensitive.
            - 'case_insensitive': Literal string match, case-insensitive.
            - 'fuzzy': Finds lines containing approximations of the words. Sorts by relevance.
            - 'words': Finds lines containing ALL specified words (subset match).
        sort_by_relevance (bool): If True and mode is 'fuzzy' or 'words', sorts results from best to worst match.
    """
    if isinstance(max_results, str):
        try:
            max_results = int(max_results)
        except ValueError:
            max_results = 50

    try:
        cwd = Path.cwd()
        results: List[Dict[str, Any]] = []
        files_scanned = 0

        # Prepare pattern based on mode
        mode = search_mode.lower()
        use_fuzzy_scoring = False
        regex = None
        keywords = []

        if mode == "exact":
            # User provided a regex
            regex = re.compile(pattern)
        elif mode == "case_sensitive":
            # Literal string match, escape regex chars
            regex = re.compile(re.escape(pattern))
        elif mode == "case_insensitive":
            # Literal string match, escape regex chars, ignore case
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        elif mode == "fuzzy":
            # Split into keywords, we will score lines that contain any of them
            keywords = re.findall(r'\b\w+\b', pattern)
            if not keywords:
                return {"success": False, "error": "Fuzzy search requires at least one keyword."}
            # We still need a loose regex to filter lines before scoring to maintain performance
            # Match any of the keywords (case insensitive)
            loose_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            regex = re.compile(loose_pattern, re.IGNORECASE)
            use_fuzzy_scoring = True
        elif mode == "words":
            # All words must be present
            keywords = re.findall(r'\b\w+\b', pattern)
            if not keywords:
                return {"success": False, "error": "Words search requires at least one keyword."}
            # Build a regex that ensures all words are present in any order
            # (?=.*\bword1\b)(?=.*\bword2\b)
            lookaheads = "".join(f"(?=.*\\b{re.escape(k)}\\b)" for k in keywords)
            regex = re.compile(f"^{lookaheads}.*$", re.IGNORECASE)
            use_fuzzy_scoring = True
        else:
            return {"success": False, "error": f"Invalid search_mode: '{search_mode}'. Must be 'exact', 'case_sensitive', 'case_insensitive', 'fuzzy', or 'words'."}

        for file_path in cwd.rglob("*"):
            if not file_path.is_file():
                continue

            if any(part in _IGNORED_DIRS for part in file_path.relative_to(cwd).parts[:-1]):
                continue

            if file_path.suffix.lower() in _BINARY_EXTS:
                continue

            if file_extension and file_extension != "*":
                if not file_path.name.lower().endswith(file_extension.lower()):
                    continue

            files_scanned += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        match = regex.search(line)
                        if match:
                            score = 1.0
                            if use_fuzzy_scoring:
                                score = _calculate_fuzzy_score(line, keywords)
                                # In 'words' mode, score is technically 1.0 if all match, 
                                # but we calculate anyway in case of partial matches being desired.
                                # If using 'words' mode, we only want exact subset matches (score >= 1.0 conceptually)
                                if mode == "words" and score < 1.0:
                                    continue # Skip if not all words found

                            results.append({
                                "file": str(file_path.relative_to(cwd)).replace("\\", "/"),
                                "line_number": line_num,
                                "text": line.strip()[:500],
                                "score": round(score, 4)
                            })
                            if len(results) >= max_results * 2: # Fetch a bit more if we need to sort
                                break
            except Exception:
                pass

            if len(results) >= max_results * 2:
                break

        # Sort by relevance if requested and applicable
        if sort_by_relevance and use_fuzzy_scoring:
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Trim to max_results
        final_results = results[:max_results]

        output_text = f"Found {len(final_results)} match(es) for pattern '{pattern}' (mode: {mode}) across {files_scanned} file(s).\n\n"
        for res in final_results:
            score_str = f" (Score: {res['score']:.2f})" if use_fuzzy_scoring else ""
            output_text += f"[{res['file']}:{res['line_number']}]{score_str} {res['text']}\n"

        return {
            "success": True,
            "matches_count": len(final_results),
            "files_scanned": files_scanned,
            "results": final_results,
            "output": output_text
        }

    except re.error as re_err:
        return {"success": False, "error": f"Invalid regex pattern: {re_err}"}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred during grep: {str(e)}"}