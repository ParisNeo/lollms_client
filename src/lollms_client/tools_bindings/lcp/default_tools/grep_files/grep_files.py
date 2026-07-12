import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def tool_grep_files(
    pattern: str,
    file_extension: Optional[str] = None,
    max_results: int = 50,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Searches for a regex pattern across files in the current working directory.
    Useful for extracting specific data from large files without loading them fully into context.

    Args:
        pattern (str): The regular expression pattern to search for.
        file_extension (str, optional): Filter search to specific file extensions (e.g., '.csv', '.json'). If None, searches all text files.
        max_results (int): Maximum number of matching lines to return. Defaults to 50.
        case_sensitive (bool): Whether the search is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the search results and a prompt injection for the LLM.
    """
    try:
        cwd = Path.cwd()
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        results: List[Dict[str, Any]] = []
        files_scanned = 0
        
        binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp',
            '.pdf', '.docx', '.xlsx', '.xls', '.db', '.sqlite', '.sqlite3',
            '.zip', '.tar', '.gz', '.mp3', '.wav', '.mp4', '.avi'
        }
        
        for file_path in cwd.rglob("*"):
            if not file_path.is_file():
                continue
                
            if file_extension:
                if not file_path.name.lower().endswith(file_extension.lower()):
                    continue
            else:
                if file_path.suffix.lower() in binary_extensions:
                    continue

            files_scanned += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                "file": str(file_path.relative_to(cwd)),
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
            "output": output_text,
            "prompt_injection": f"\n\n✅ **Grep Search Complete.** Found {len(results)} matches. Use the specific lines above to answer the user's question. Do NOT attempt to load the full files into context."
        }
        
    except re.error as re_err:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {re_err}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"An unexpected error occurred during grep: {str(e)}"
        }
