import os
import re
import fnmatch
from pathlib import Path
from typing import Dict, Any, List, Optional

# Windows reserved device names that can trigger special I/O behavior.
_WIN_DEVICE_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

TOOL_LIBRARY_NAME = "Workspace Tools"
TOOL_LIBRARY_DESC = "Tools for reading, writing, listing, finding, and grepping files in the agent's workspace."
TOOL_LIBRARY_ICON = "📁"

_COMPILED_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".class",
}

_GREP_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp",
    ".pdf", ".docx", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".mp3", ".wav", ".m4a",
    ".mp4", ".avi", ".mov", ".mkv", ".pyc", ".pyo", ".so", ".dll",
}

_IGNORED_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv",
    "venv", ".idea", ".vscode", ".lollms", "build", "dist", ".next",
    "env", ".env", ".lollms_code", ".lollms_metadata", "egg-info",
    "dist-info", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "htmlcov", "site-packages", "artefacts_metadata",
}


# ── Module-level cache state (declared eagerly so that _get_workspace_root
# can be probed BEFORE init_tools_library ever runs, without NameError). ──
_CACHED_WORKSPACE_ROOT: Optional[Path] = None
_TOOL_BINDING = None


def init_tools_library(tool_binding=None, **kwargs):
    """Initializes the workspace tools library and caches the workspace root.

    The LCP runtime has historically called this hook with a single positional
    argument that is EITHER a ``host_config`` dict (``{"workspace": ...,
    "workspace_data_path": ..., "binding": <LCPBinding>}``) OR the binding
    object itself. Older/other callers may pass ``workspace=...`` as a keyword.

    We accept all three shapes and cache the first valid workspace path so that
    read/grep/find/list tools resolve against the REAL discussion workspace
    instead of silently falling back to ``cwd`` (the project root) — which made
    genuinely-present workspace files report as 'not found'.
    """
    global _CACHED_WORKSPACE_ROOT, _TOOL_BINDING

    candidate: Optional[str] = kwargs.get("workspace")
    binding_obj = None

    # Shape A: first positional arg is a host_config dict (current LCP contract).
    if isinstance(tool_binding, dict):
        candidate = candidate or tool_binding.get("workspace") or tool_binding.get("workspace_data_path")
        binding_obj = tool_binding.get("binding")
    # Shape B: first positional arg is the binding object itself.
    elif tool_binding is not None:
        binding_obj = tool_binding
        if not candidate:
            candidate = getattr(tool_binding, "workspace_data_path", None)

    # Keep a reference to the binding (NOT a raw dict) so late-binding lookups
    # in ``_get_workspace_root`` can consult ``.workspace_data_path``.
    _TOOL_BINDING = binding_obj

    if not candidate and binding_obj is not None:
        candidate = getattr(binding_obj, "workspace_data_path", None)
    if not candidate:
        for _env in ("LOLLMS_WORKSPACE_DIR", "LOLLMS_DISCUSSION_WORKSPACE"):
            if os.environ.get(_env):
                candidate = os.environ[_env]
                break

    if candidate:
        try:
            _CACHED_WORKSPACE_ROOT = Path(candidate).expanduser().resolve()
        except Exception:
            _CACHED_WORKSPACE_ROOT = None
def _is_within_base(candidate: Path, base: Path) -> bool:
    """Case-aware containment check that matches the host filesystem semantics."""
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        # On case-insensitive filesystems (Windows, macOS default) perform a
        # normalized comparison so that differing casing of the workspace root
        # is not incorrectly rejected.
        if os.name == "nt" or (hasattr(os, "uname") and os.uname().sysname == "Darwin"):
            cand_parts = [p.lower() for p in candidate.parts]
            base_parts = [p.lower() for p in base.parts]
            if len(cand_parts) >= len(base_parts):
                return cand_parts[: len(base_parts)] == base_parts
        return False


def _assert_within_base(candidate: Path, base: Path, original: str) -> None:
    """Raises PermissionError when candidate escapes base. Used as defense in depth."""
    if not _is_within_base(candidate, base):
        raise PermissionError(
            f"Path traversal detected: '{original}' attempts to escape workspace."
        )


def _reject_dangerous_names(file_name: str) -> None:
    """Rejects embedded NULs, control characters, and Windows device names."""
    if "\x00" in file_name:
        raise PermissionError("Rejected file name: null byte.")
    stripped = file_name.strip()
    if not stripped:
        return
    # Reject pure Windows device names (e.g. "CON", "NUL", "COM1") at the leaf level.
    leaf = stripped.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    leaf_stem = leaf.split(".")[0].lower() if leaf else ""
    if leaf_stem in _WIN_DEVICE_NAMES:
        raise PermissionError(f"Rejected file name: reserved device name '{leaf}'.")


def _get_workspace_root() -> Path:
    """Resolve the workspace root reliably across all execution contexts.

    Prefers the workspace injected by the runtime via
    ``init_tools_library(workspace=...)`` (cached in ``_CACHED_WORKSPACE_ROOT``),
    then the live ``tool_binding.workspace_data_path``, then the ``LOLLMS_*``
    env vars, and only finally falls back to ``cwd``. Without this the tools
    resolved against the project root and reported existing workspace files as
    'not found'.
    """
    if _CACHED_WORKSPACE_ROOT is not None:
        return _CACHED_WORKSPACE_ROOT
    if _TOOL_BINDING is not None:
        bound = getattr(_TOOL_BINDING, "workspace_data_path", None)
        if bound:
            try:
                return Path(bound).expanduser().resolve()
            except Exception:
                pass
    for env_name in ("LOLLMS_WORKSPACE_DIR", "LOLLMS_DISCUSSION_WORKSPACE"):
        env_val = os.environ.get(env_name)
        if env_val:
            try:
                return Path(env_val).expanduser().resolve()
            except Exception:
                continue
    return Path.cwd().resolve()


def _get_workspace_root() -> Path:
    """Resolve the workspace root reliably across all execution contexts.

    Prefers the workspace injected by the runtime via
    ``init_tools_library(workspace=...)`` (cached in ``_CACHED_WORKSPACE_ROOT``),
    then the live ``tool_binding.workspace_data_path``, then the ``LOLLMS_*``
    env vars, and only finally falls back to ``cwd``. Without a stable root the
    walkers relativized against ``cwd`` and reported existing workspace files as
    'not found'.
    """
    if _CACHED_WORKSPACE_ROOT is not None:
        return _CACHED_WORKSPACE_ROOT
    if _TOOL_BINDING is not None:
        bound = getattr(_TOOL_BINDING, "workspace_data_path", None)
        if bound:
            try:
                return Path(bound).expanduser().resolve()
            except Exception:
                pass
    for env_name in ("LOLLMS_WORKSPACE_DIR", "LOLLMS_DISCUSSION_WORKSPACE"):
        env_val = os.environ.get(env_name)
        if env_val:
            try:
                return Path(env_val).expanduser().resolve()
            except Exception:
                continue
    return Path.cwd().resolve()


def _resolve_safe_path(file_name: str) -> Path:
    """
    Sanitizes the file name and resolves it safely within the workspace root.
    Blocks path traversal attacks (e.g., ../../etc/passwd, symlinks escaping the workspace,
    Windows device names, embedded NUL bytes).
    """
    base_path = _get_workspace_root()

    if file_name is None:
        raise PermissionError("Rejected file name: None.")
    if not isinstance(file_name, str):
        raise PermissionError(f"Rejected file name: expected str, got {type(file_name).__name__}.")

    _reject_dangerous_names(file_name)

    if not file_name or file_name.strip() in (".", "./", ".\\"):
        return base_path

    raw_path = Path(file_name.strip())
    if raw_path.is_absolute():
        try:
            target_path = raw_path.resolve()
        except (OSError, RuntimeError) as exc:
            raise PermissionError(f"Could not resolve path '{file_name}': {exc}")
        _assert_within_base(target_path, base_path, file_name)
        return target_path

    # Reject home-directory expansion attempts at the leaf: agents should pass
    # workspace-relative paths, never `~/...`.
    if raw_path.parts and raw_path.parts[0] == "~":
        raise PermissionError(f"Path traversal detected: '{file_name}' uses home expansion.")

    clean_name = file_name.replace("\\", "/")
    # Strip Windows drive prefix like "C:file" (relative drive form).
    if len(clean_name) > 1 and clean_name[1] == ":":
        clean_name = clean_name[2:]
    clean_name = clean_name.lstrip("/")

    target_path = (base_path / clean_name).resolve()
    _assert_within_base(target_path, base_path, file_name)
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
        # Re-verify containment after parent creation (defense in depth).
        _assert_within_base(target_path.resolve(), Path.cwd().resolve(), file_name)
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

        # Re-verify containment after resolution (defense in depth against TOCTOU
        # and symlink swaps between _resolve_safe_path and this read).
        _assert_within_base(target_path.resolve(), Path.cwd().resolve(), file_name)

        ext = target_path.suffix.lower()
        if ext in _GREP_BINARY_EXTS:
            size = target_path.stat().st_size
            return {
                "success": True,
                "output": f"[Binary/media file: {target_path.name} ({size:,} bytes). Content cannot be displayed as plain text. Use appropriate tools to inspect or move this file.]",
                "is_binary": True,
                "size": size
            }

        content = target_path.read_text(encoding="utf-8", errors="ignore")
        return {"success": True, "output": content}
    except PermissionError as pe:
        return {"success": False, "error": str(pe)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Failed to read file '{file_name}': {str(e)}", "traceback": traceback.format_exc()}


def tool_list_files(directory: str = ".", recursive: bool = False) -> Dict[str, Any]:
    """
    List the immediate contents of a directory within the workspace (non-recursive by default).
    Lists all files (including images, audio, video, and documents) so agents can see full directory contents.

    Args:
        directory (str, optional): Directory to list. Defaults to current directory ('.').
        recursive (bool, optional): If True, performs a deep recursive listing while skipping
            hidden and ignored folders (e.g. .git). Defaults to False.
    """
    try:
        target_path = _resolve_safe_path(directory)
        if not target_path.exists():
            return {"success": False, "error": f"Directory '{directory}' not found."}
        if not target_path.is_dir():
            return {"success": False, "error": f"Path '{directory}' is not a directory."}

        entries: List[str] = []

        if not recursive:
            for item in target_path.iterdir():
                name = item.name
                if name in _IGNORED_DIRS or name.startswith("."):
                    continue
                if item.is_dir():
                    entries.append(f"{name}/ (dir)")
                elif item.is_file() and item.suffix.lower() not in _COMPILED_EXTS:
                    entries.append(name)

            entries.sort(key=lambda e: (not e.endswith("/ (dir)"), e.lower()))
            output_str = "\n".join(entries) if entries else "Directory is empty."
            return {"success": True, "files": entries, "output": output_str}

        base_path = _get_workspace_root()
        for root, dirs, files in os.walk(target_path, followlinks=False):
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS and not d.startswith(".")]
            for filename in files:
                if Path(filename).suffix.lower() in _COMPILED_EXTS or filename.startswith("."):
                    continue
                try:
                    resolved = Path(root, filename).resolve()
                    if not _is_within_base(resolved, base_path):
                        continue
                    rel_path = resolved.relative_to(base_path)
                    entries.append(str(rel_path).replace("\\", "/"))
                except (ValueError, OSError):
                    continue

        entries.sort()
        output_str = "\n".join(entries) if entries else "Directory is empty."
        return {"success": True, "files": entries, "output": output_str}
    except PermissionError as pe:
        return {"success": False, "error": str(pe)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Failed to list files in '{directory}': {str(e)}", "traceback": traceback.format_exc()}


def tool_find_files(pattern: str, path: str = ".", max_results: int = 50) -> Dict[str, Any]:
    """
    Recursively searches for files matching a name pattern within a given directory.
    Uses shell wildcards (e.g., '*.py', 'config.*', '*.png').

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

        base_path = _get_workspace_root()
        matches: List[str] = []
        for root, dirs, files in os.walk(search_dir, followlinks=False):
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS and not d.startswith('.')]

            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    try:
                        resolved = Path(root, filename).resolve()
                        if not _is_within_base(resolved, base_path):
                            continue
                        rel_path = os.path.relpath(str(resolved), str(base_path))
                        matches.append(rel_path.replace("\\", "/"))
                    except (ValueError, OSError):
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


def _normalize_search_mode(raw: str) -> str:
    """Normalizes search mode strings: lowercases and replaces hyphens/spaces with underscores."""
    if not raw:
        return ""
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


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
        cwd = _get_workspace_root()
        results: List[Dict[str, Any]] = []
        files_scanned = 0

        mode = _normalize_search_mode(search_mode)
        use_fuzzy_scoring = False
        regex = None
        keywords = []

        if mode == "exact":
            regex = re.compile(pattern)
        elif mode == "case_sensitive":
            regex = re.compile(re.escape(pattern))
        elif mode == "case_insensitive":
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        elif mode == "fuzzy":
            keywords = re.findall(r'\b\w+\b', pattern)
            if not keywords:
                return {"success": False, "error": "Fuzzy search requires at least one keyword."}
            loose_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            regex = re.compile(loose_pattern, re.IGNORECASE)
            use_fuzzy_scoring = True
        elif mode == "words":
            keywords = re.findall(r'\b\w+\b', pattern)
            if not keywords:
                return {"success": False, "error": "Words search requires at least one keyword."}
            lookaheads = "".join(f"(?=.*\\b{re.escape(k)}\\b)" for k in keywords)
            regex = re.compile(f"^{lookaheads}.*$", re.IGNORECASE)
            use_fuzzy_scoring = True
        else:
            return {"success": False, "error": f"Invalid search_mode: '{search_mode}'. Must be 'exact', 'case_sensitive', 'case_insensitive', 'fuzzy', or 'words'."}

        for root_str, dirs, filenames in os.walk(cwd, followlinks=False):
            # Prune ignored / hidden directories at the top of each walk.
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS and not d.startswith(".")]

            root_path = Path(root_str)
            for filename in filenames:
                file_path = root_path / filename

                if file_path.suffix.lower() in _GREP_BINARY_EXTS:
                    continue

                if file_extension and file_extension != "*":
                    if not file_path.name.lower().endswith(file_extension.lower()):
                        continue

                # Defense in depth: verify the resolved path stays inside the
                # workspace (blocks symlink-planted escape attempts).
                try:
                    resolved = file_path.resolve()
                except (OSError, RuntimeError):
                    continue
                if not _is_within_base(resolved, cwd):
                    continue
                if not resolved.is_file():
                    continue

                files_scanned += 1
                try:
                    with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            match = regex.search(line)
                            if match:
                                score = 1.0
                                if use_fuzzy_scoring:
                                    score = _calculate_fuzzy_score(line, keywords)
                                    if mode == "words" and score < 1.0:
                                        continue

                                try:
                                    rel = resolved.relative_to(cwd)
                                except ValueError:
                                    continue
                                results.append({
                                    "file": str(rel).replace("\\", "/"),
                                    "line_number": line_num,
                                    "text": line.strip()[:500],
                                    "score": round(score, 4)
                                })
                                if len(results) >= max_results * 2:
                                    break
                except Exception:
                    continue

                if len(results) >= max_results * 2:
                    break

            if len(results) >= max_results * 2:
                break

        if sort_by_relevance and use_fuzzy_scoring:
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

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


def tool_grep_file(
    pattern: str,
    path: Optional[str] = None,
    file_extension: Optional[str] = None,
    max_results: int = 50,
    search_mode: str = "case_insensitive",
    sort_by_relevance: bool = True,
) -> Dict[str, Any]:
    """
    Search for a pattern across files in the workspace (singular alias of tool_grep_files).

    This alias exists because models frequently call the singular name `tool_grep_file`.
    The `path` argument is accepted for backward compatibility but IGNORED for scoping:
    searching always happens from the workspace root, because LCP already mounts the
    agent in the correct working directory. Passing a file name (e.g. 'train_surrogate.py')
    as `pattern` will search for its occurrences inside file *contents*, not for a filename.
    Use `tool_find_files` to locate a file by name.

    Args:
        pattern (str): The search pattern (regex for 'exact', literal/keywords otherwise).
        path (str, optional): Ignored. Kept only for model-argument compatibility.
        file_extension (str, optional): Filter by extension (e.g. '.py'). '*' = all files.
        max_results (int): Maximum number of matching lines to return. Defaults to 50.
        search_mode (str): 'exact' | 'case_sensitive' | 'case_insensitive' | 'fuzzy' | 'words'.
        sort_by_relevance (bool): Sort fuzzy/words results by score descending.
    """
    return tool_grep_files(
        pattern=pattern,
        file_extension=file_extension,
        max_results=max_results,
        search_mode=search_mode,
        sort_by_relevance=sort_by_relevance,
    )