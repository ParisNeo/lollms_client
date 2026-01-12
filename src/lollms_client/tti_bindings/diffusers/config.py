from pathlib import Path
import json

def get_binding_desc():
    """
    Returns the binding description as a dictionary by reading the description file
    located in the same folder as this config.py file.
    """
    # Get the directory of the current file using pathlib
    current_dir = Path(__file__).parent
    
    # Construct the path to the description file
    desc_file_path = current_dir / "description.json"
    
    # Read and return the content as a dictionary
    with open(desc_file_path, 'r', encoding='utf-8') as f:
        desc = json.load(f)

    # Dynamically detect available devices
    try:
        import torch
        available_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                available_devices.append(f"cuda:{i}")
        available_devices.append("cpu")
    except ImportError:
        # If torch is not available, fallback to just "cpu"
        available_devices = ["cpu"]
        # Optional: Try to detect CUDA drivers without torch
        try:
            import subprocess
            # Check if nvidia-smi is available (indicates CUDA drivers are installed)
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                # CUDA drivers are available, but we can't enumerate devices without torch
                # So we add a generic "cuda" device as a hint
                available_devices.append("cuda")
        except (ImportError, FileNotFoundError):
            pass  # No nvidia-smi or subprocess issue, stick with "cpu"

    desc["available_devices"] = available_devices
    return desc
