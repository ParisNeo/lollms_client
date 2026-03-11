import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
import sys
import json
import pipmaster as pm
pm.ensure_packages(["RestrictedPython"])

import RestrictedPython
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.PrintCollector import PrintCollector


def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a snippet of Python code in a restricted environment.
    """
    code = params.get("code")
    timeout_seconds = params.get("timeout_seconds", 10)

    if not code:
        return {
            "stdout": "",
            "stderr": "Error: 'code' parameter is required.",
            "returned_value": None,
            "execution_status": "error"
        }

    # For this local tool, we'll use RestrictedPython for a degree of safety.
    # A more robust solution for untrusted code would involve containers (e.g., Docker)
    # or a more heavily sandboxed environment. RestrictedPython is not foolproof.

    restricted_globals = dict(safe_globals) # type: ignore
    restricted_globals['_print_'] = PrintCollector # Capture print statements
    restricted_globals['_getattr_'] = RestrictedPython.Guards.safer_getattr # Allow safer attribute access
    # Add more globals if needed, e.g., math module, but be cautious.
    # restricted_globals['math'] = math

    stdout_capture = ""
    stderr_capture = ""
    returned_value_str = None
    status = "error" # Default to error

    try:
        # Compile the code in restricted mode
        # Adding "<string>" as the filename for tracebacks
        byte_code = compile_restricted(code, filename='<inline_script>', mode='exec')
        
        # Prepare a dictionary for local variables, including the print collector
        local_vars = {} 
        
        # Execute the compiled code
        # Note: RestrictedPython's exec doesn't directly support timeout.
        # A more complex setup with threading or multiprocessing would be needed for true timeout.
        # For simplicity, this example doesn't implement a hard timeout for RestrictedPython's exec.
        # The timeout_seconds param is more of a hint for now or for alternative execution methods.

        exec(byte_code, restricted_globals, local_vars)
        
        stdout_capture = restricted_globals['_print']() # Get captured prints
        # RestrictedPython itself doesn't easily separate stdout/stderr from exec.
        # Errors during compilation or guarded execution will raise exceptions.

        # Try to get a returned value if the script used 'return' (not typical for 'exec' mode)
        # or if a specific variable was set to indicate a return.
        # For this example, we won't try to capture implicit returns from 'exec'.
        # If the executed code sets a variable like `result = ...`, it would be in `local_vars`.
        # We'll consider `returned_value` as None for simplicity with 'exec'.

        status = "success"

    except SyntaxError as se:
        stderr_capture = f"SyntaxError: {se}"
        status = "error"
    except RestrictedPython.Guards. इलाकोंOnlyĐiPythonGuardViolation as rpe: # Common RestrictedPython error
        stderr_capture = f"RestrictedPythonError: {rpe}"
        status = "error"
    except NameError as ne: # Undeclared variable
        stderr_capture = f"NameError: {ne}"
        status = "error"
    except TypeError as te:
        stderr_capture = f"TypeError: {te}"
        status = "error"
    except Exception as e:
        stderr_capture = f"Execution Error: {type(e).__name__}: {e}"
        status = "error"
        trace_exception(e) # For more detailed logging if needed

    return {
        "stdout": stdout_capture,
        "stderr": stderr_capture,
        "returned_value": returned_value_str, # Typically None with exec
        "execution_status": status
    }


if __name__ == '__main__':
    # Example Tests
    print("--- Python Interpreter Tool Test ---")

    # Test 1: Simple print
    params1 = {"code": "print('Hello from restricted Python!')\nx = 10 + 5\nprint(f'Result is {x}')"}
    result1 = execute(params1)
    print(f"\nTest 1 Result:\n{json.dumps(result1, indent=2)}")
    assert result1["execution_status"] == "success"
    assert "Hello from restricted Python!" in result1["stdout"]
    assert "Result is 15" in result1["stdout"]

    # Test 2: Code with an error (e.g., trying to access restricted features)
    params2 = {"code": "import os\nprint(os.getcwd())"} # os is usually restricted
    result2 = execute(params2)
    print(f"\nTest 2 Result (expected error):\n{json.dumps(result2, indent=2)}")
    assert result2["execution_status"] == "error"
    assert "RestrictedPythonError" in result2["stderr"] or "NameError" in result2["stderr"] # Depending on how os import fails

    # Test 3: Syntax error
    params3 = {"code": "print('Hello without closing quote"}
    result3 = execute(params3)
    print(f"\nTest 3 Result (syntax error):\n{json.dumps(result3, indent=2)}")
    assert result3["execution_status"] == "error"
    assert "SyntaxError" in result3["stderr"]
    
    # Test 4: No code
    params4 = {"code": ""}
    result4 = execute(params4) # Should succeed with empty output or be handled if empty code is invalid.
                               # Current RestrictedPython might raise error on empty.
    print(f"\nTest 4 Result (empty code):\n{json.dumps(result4, indent=2)}")    
    # Depending on compile_restricted behavior for empty string, this might be success or error.
    # It seems compile_restricted can handle empty strings.
    assert result4["execution_status"] == "success" 
    assert result4["stdout"] == ""

    # Test 5: Attempting a disallowed attribute
    params5 = {"code": "x = ().__class__"} # Example of trying to access something disallowed
    result5 = execute(params5)
    print(f"\nTest 5 Result (disallowed attribute):\n{json.dumps(result5, indent=2)}")
    assert result5["execution_status"] == "error"
    assert "RestrictedPythonError" in result5["stderr"]
    
    print("\n--- Tests Finished ---")