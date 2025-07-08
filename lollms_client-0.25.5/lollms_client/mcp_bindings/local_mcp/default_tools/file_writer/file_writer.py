from pathlib import Path
from typing import Dict, Any

def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes or appends text content to a specified file.
    """
    file_path_str = params.get("file_path")
    content = params.get("content")
    mode = params.get("mode", "overwrite") # Default to overwrite

    if not file_path_str:
        return {"status": "error", "message": "file_path parameter is required."}
    if content is None: # Allow empty string, but not None
        return {"status": "error", "message": "content parameter is required."}
    if mode not in ["overwrite", "append"]:
        return {"status": "error", "message": "Invalid mode. Must be 'overwrite' or 'append'."}

    try:
        # Security consideration: Restrict file paths if necessary in a real application.
        # For this local tool, we'll assume the user/AI provides paths responsibly.
        # However, avoid absolute paths starting from root unless explicitly intended.
        # For simplicity here, we allow relative and 'safe' absolute paths.
        
        # Make path relative to a 'workspace' if not absolute, to prevent writing anywhere.
        # For this example, let's assume a 'tool_workspace' directory next to the tools_folder.
        # This needs to be configurable or defined by the LocalMCPBinding.
        # For now, we'll just resolve the path. A real implementation would need sandboxing.
        
        target_file = Path(file_path_str)

        # Create parent directories if they don't exist
        target_file.parent.mkdir(parents=True, exist_ok=True)

        write_mode = "w" if mode == "overwrite" else "a"

        with open(target_file, write_mode, encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"Content successfully {'written to' if mode == 'overwrite' else 'appended to'} file.",
            "file_path": str(target_file.resolve())
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to write to file: {str(e)}",
            "file_path": str(Path(file_path_str).resolve() if file_path_str else "N/A")
        }

if __name__ == '__main__':
    # Example test
    test_params_overwrite = {
        "file_path": "test_output/example.txt",
        "content": "Hello from the file_writer tool!\nThis is the first line.",
        "mode": "overwrite"
    }
    result_overwrite = execute(test_params_overwrite)
    print(f"Overwrite Test Result: {result_overwrite}")

    test_params_append = {
        "file_path": "test_output/example.txt",
        "content": "\nThis is an appended line.",
        "mode": "append"
    }
    result_append = execute(test_params_append)
    print(f"Append Test Result: {result_append}")

    test_params_error = {
        "content": "Missing file path"
    }
    result_error = execute(test_params_error)
    print(f"Error Test Result: {result_error}")