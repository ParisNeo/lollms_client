from pathlib import Path
import yaml
import importlib
from typing import List, Dict, Optional, Union
import sys

# Define known binding types and their corresponding folder names
BINDING_TYPES = {
    "llm": "llm_bindings",
    "tti": "tti_bindings",
    "stt": "stt_bindings",
    "tts": "tts_bindings",
    "ttm": "ttm_bindings",
    "ttv": "ttv_bindings",
    "mcp": "mcp_bindings",
}

def list_bindings(binding_type: str = "llm", custom_bindings_dir: Union[Path, str, None] = None) -> List[str]:
    """
    Lists available bindings of a specific type.
    
    Args:
        binding_type (str): The type of binding (llm, tti, stt, tts, ttm, ttv, mcp).
        custom_bindings_dir (Path | str | None): Path to a custom bindings directory.
        
    Returns:
        List[str]: A list of binding names.
    """
    if custom_bindings_dir:
        bindings_dir = Path(custom_bindings_dir)
    else:
        root_dir = Path(__file__).parent
        folder_name = BINDING_TYPES.get(binding_type, f"{binding_type}_bindings")
        bindings_dir = root_dir / folder_name

    if not bindings_dir.exists():
        return []

    bindings = []
    for entry in bindings_dir.iterdir():
        if entry.is_dir() and (entry / "__init__.py").exists():
            bindings.append(entry.name)
    
    return sorted(bindings)

def get_binding_desc(binding_name: str, binding_type: str) -> Dict:
    """
    Retrieves the description of a binding.
    
    Checks for a get_binding_desc() function in the binding's __init__.py first.
    If not found, falls back to reading description.yaml in the binding directory.
    
    Args:
        binding_name (str): The name of the binding.
        binding_type (str, optional): The type of binding to narrow search (llm, tti, etc.). 
                                      If None, searches all known types.
    
    Returns:
        Dict: The binding description.
    """
    
    found_path = None
    found_type = None
    
    root_dir = Path(__file__).parent
    
    # Locate the binding
    if not binding_type:
        raise Exception("Please specify the binding type")
    folder_name = BINDING_TYPES.get(binding_type, f"{binding_type}_bindings")
    candidate_path = root_dir / folder_name / binding_name
    if candidate_path.exists() and (candidate_path / "__init__.py").exists():
        found_path = candidate_path
        found_type = binding_type

    
    if not found_path:
        return {"error": f"Binding {binding_name} not found."}

    # Try to import and call get_binding_desc
    try:
        folder_name = BINDING_TYPES.get(found_type, found_type+'_bindings')
        module_name = f"lollms_client.{folder_name}.{binding_name}.config"
        module = importlib.import_module(module_name)
        
        if hasattr(module, "get_binding_desc") and callable(module.get_binding_desc):
            return module.get_binding_desc()
    except Exception as e:
        # If import fails no config.py file is found, ignore and use yaml
        pass

    # Fallback to description.yaml
    desc_file = found_path / "description.yaml"
    if desc_file.exists():
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            return {"error": f"Failed to load description.yaml: {e}"}
            
    return {"error": "No description found."}
