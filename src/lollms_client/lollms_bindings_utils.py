from pathlib import Path
import yaml
from typing import List, Dict, Optional, Union
import importlib.util
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
    
    This function first attempts to load a custom description loader from
    'binding_config.py' if it exists in the binding directory. This allows
    bindings to customize their description without loading the full binding.
    
    If 'binding_config.py' does not exist or fails to load, it falls back to
    reading 'description.yaml' directly.
    
    Args:
        binding_name (str): The name of the binding.
        binding_type (str): The type of binding (llm, tti, etc.).
    
    Returns:
        Dict: The binding description.
    """
    
    if not binding_type:
        raise Exception("Please specify the binding type")
    
    root_dir = Path(__file__).parent
    folder_name = BINDING_TYPES.get(binding_type, f"{binding_type}_bindings")
    binding_path = root_dir / folder_name / binding_name
    
    if not binding_path.exists() or not binding_path.is_dir():
        return {"error": f"Binding {binding_name} not found in {folder_name}."}

    # Step 1: Try to use custom description loader from binding_config.py
    # This is a lightweight module that should not have heavy dependencies
    config_file = binding_path / "binding_config.py"
    if config_file.exists():
        try:
            # Use spec-based import to avoid polluting sys.modules heavily
            module_name = f"lollms_client.{folder_name}.{binding_name}.binding_config"
            
            # Check if already loaded
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, str(config_file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = None
            
            if module and hasattr(module, "get_binding_desc") and callable(module.get_binding_desc):
                return module.get_binding_desc()
        except Exception as e:
            # Custom loader failed, fall through to YAML fallback
            pass
    
    # Step 2: Fall back to reading description.yaml directly
    # This path has zero side effects - no code execution
    desc_file = binding_path / "description.yaml"
    if desc_file.exists():
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            return {"error": f"Failed to load description.yaml: {e}"}
            
    return {"error": "No description found (neither binding_config.py with get_binding_desc() nor description.yaml exists)."}
