import os
from pathlib import Path
from typing import List, Dict, Any

def tool_find_files(
    pattern: str, 
    path: str = ".", 
    max_results: int = 50
) -> Dict[str, Any]:
    """
    Recursively searches for files matching a name pattern within a given directory.
    Uses shell wildcards (e.g., '*.py', 'config.*').

    Args:
        pattern (str): The file name pattern to match (supports * and ?).
        path (str): The directory to search in. Defaults to current directory.
        max_results (int): Maximum number of file paths to return. Defaults to 50.
    """
    import fnmatch
    
    try:
        search_dir = Path(path).resolve()
        if not search_dir.is_dir():
            return {
                "success": False,
                "error": f"Directory '{path}' not found."
            }

        matches: List[str] = []
        for root, _, files in os.walk(search_dir):
            # Skip hidden directories like .git, .venv, __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in Path(root).parts):
                continue
                
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    try:
                        rel_path = os.path.relpath(os.path.join(root, filename), search_dir)
                        matches.append(rel_path.replace("\\", "/"))
                    except ValueError:
                        pass
                        
            if len(matches) >= max_results:
                matches = matches[:max_results]
                break

        return {
            "success": True,
            "output": f"Found {len(matches)} file(s) matching '{pattern}':",
            "files": matches
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }