import os
import sys
import requests
import subprocess
import time
from pathlib import Path
from typing import Optional, List

# Ensure filelock is available for process-safe server startup.
try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "XTTSClientBinding"

class XTTSClientBinding(LollmsTTSBinding):
    """
    Client binding for a dedicated, managed XTTS server.
    This architecture prevents the heavy XTTS model from being loaded into memory
    by multiple worker processes, solving potential OOM errors and speeding up TTS generation.
    """
    def __init__(self, 
                 **kwargs):
        
        binding_name = "xtts"
        super().__init__(binding_name=binding_name, **kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9633)
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path("./venv/tts_xtts_venv")

        if self.auto_start_server:
            self.ensure_server_is_running()

    def is_server_running(self) -> bool:
        """Checks if the server is already running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def ensure_server_is_running(self):
        """
        Ensures the XTTS server is running. If not, it attempts to start it
        in a process-safe manner using a file lock.
        """
        self.server_dir.mkdir(exist_ok=True)
        lock_path = self.server_dir / "xtts_server.lock"
        lock = FileLock(lock_path)

        ASCIIColors.info("Attempting to start or connect to the XTTS server...")
        
        if self.is_server_running():
            ASCIIColors.green("XTTS Server is already running and responsive.")
            return

        try:
            with lock.acquire(timeout=10):
                if not self.is_server_running():
                    ASCIIColors.yellow("Lock acquired. Starting dedicated XTTS server...")
                    self.start_server()
                    self._wait_for_server()
                else:
                    ASCIIColors.green("Server was started by another process while we waited. Connected successfully.")
        except Timeout:
            ASCIIColors.yellow("Could not acquire lock, another process is starting the server. Waiting...")
            self._wait_for_server(timeout=60)

        if not self.is_server_running():
            raise RuntimeError("Failed to start or connect to the XTTS server after all attempts.")


    def install_server_dependencies(self):
        """
        Installs the server's dependencies into a dedicated virtual environment
        using pipmaster, which handles complex packages like PyTorch.
        """
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        # Ensure pipmaster is available.
        try:
            import pipmaster as pm
        except ImportError:
            print("FATAL: pipmaster is not installed. Please install it using: pip install pipmaster")
            raise Exception("pipmaster not found")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir))
        
        requirements_file = self.server_dir / "requirements.txt"
        
        ASCIIColors.info("Installing server dependencies from requirements.txt...")
        success = pm_v.ensure_requirements(str(requirements_file), verbose=True)

        if not success:
            ASCIIColors.error("Failed to install server dependencies. Please check the console output for errors.")
            raise RuntimeError("XTTS server dependency installation failed.")

        ASCIIColors.green("Server dependencies are satisfied.")


    def start_server(self):
        """
        Installs dependencies and launches the FastAPI server as a background subprocess.
        This method should only be called from within a file lock.
        """
        server_script = self.server_dir / "main.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found at {server_script}.")

        if not self.venv_dir.exists():
            self.install_server_dependencies()

        if sys.platform == "win32":
            python_executable = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_dir / "bin" / "python"

        command = [
            str(python_executable),
            str(server_script),
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        # Use DETACHED_PROCESS on Windows to allow the server to run independently.
        creationflags = subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        
        self.server_process = subprocess.Popen(command, creationflags=creationflags)
        ASCIIColors.info("XTTS server process launched in the background.")

    def _wait_for_server(self, timeout=10):
        """Waits for the server to become responsive."""
        ASCIIColors.info("Waiting for XTTS server to become available...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                ASCIIColors.green("XTTS Server is up and running.")
                return
            time.sleep(2)
        raise RuntimeError("Failed to connect to the XTTS server within the specified timeout.")

    def __del__(self):
        # The client destructor does not stop the server,
        # as it is a shared resource for other processes.
        pass

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Generate audio by calling the server's API"""
        payload = {"text": text, "voice": voice}
        # Pass other kwargs from the description file (language, split_sentences)
        payload.update(kwargs)
        
        try:
            response = requests.post(f"{self.base_url}/generate_audio", json=payload, timeout=300)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with XTTS server at {self.base_url}.")
            ASCIIColors.error(f"Error details: {e}")
            raise RuntimeError("Communication with the XTTS server failed.") from e


    def list_voices(self, **kwargs) -> List[str]:
        """Get available voices from the server"""
        try:
            response = requests.get(f"{self.base_url}/list_voices")
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to get voices from XTTS server: {e}")
            return []


    def list_models(self, **kwargs) -> list:
        """Lists models supported by the server"""
        try:
            response = requests.get(f"{self.base_url}/list_models")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to get models from XTTS server: {e}")
            return []