from lollms_client.lollms_tts_binding import LollmsTTSBinding
from typing import Optional, List
from pathlib import Path
import requests
import subprocess
import sys
import time
import pipmaster as pm

# New import for process-safe file locking
try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)


BindingName = "XTTSClientBinding"

class XTTSClientBinding(LollmsTTSBinding):
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 8081, 
                 auto_start_server: bool = True,
                 **kwargs):
        
        binding_name = "xtts"
        super().__init__(binding_name=binding_name, **kwargs)
        self.host = host
        self.port = port
        self.auto_start_server = auto_start_server
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        if self.auto_start_server:
            self.ensure_server_is_running()

    def is_server_running(self) -> bool:
        """Checks if the server is already running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=1)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.ConnectionError:
            return False
        return False

    def ensure_server_is_running(self):
        """
        Ensures the XTTS server is running. If not, it attempts to start it
        in a process-safe manner using a file lock.
        """
        if self.is_server_running():
            print("XTTS Server is already running.")
            return

        lock_path = self.server_dir / "xtts_server.lock"
        lock = FileLock(lock_path, timeout=10) # Wait a maximum of 10 seconds for the lock

        print("Attempting to start or wait for the XTTS server...")
        try:
            with lock:
                # Double-check after acquiring the lock to handle race conditions
                if not self.is_server_running():
                    print("Lock acquired. Starting dedicated XTTS server...")
                    self.start_server()
                else:
                    print("Server was started by another process while waiting for the lock.")
        except Timeout:
            print("Could not acquire lock. Another process is likely starting the server. Waiting...")

        # All workers (the one that started the server and those that waited) will verify the server is ready
        self._wait_for_server()

    def install(self, venv_path, requirements_file):
        print(f"Ensuring virtual environment and dependencies in: {venv_path}")
        pm_v = pm.PackageManager(venv_path=str(venv_path))

        success = pm_v.ensure_requirements(
            str(requirements_file),
            verbose=True
        )

        if not success:
            print("FATAL: Failed to install server dependencies. Aborting launch.")
            return

        print("Dependencies are satisfied. Proceeding to launch server...")

    def start_server(self):
        """
        Installs dependencies and launches the server as a background subprocess.
        This method should only be called from within a file lock.
        """
        requirements_file = self.server_dir / "requirements.txt"
        server_script = self.server_dir / "main.py"

        # 1. Ensure a virtual environment and dependencies
        venv_path = self.server_dir / "venv"

        if not venv_path.exists():
            self.install(venv_path, requirements_file)
            
        # 2. Get the python executable from the venv
        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"

        # 3. Launch the server as a detached subprocess
        command = [
            str(python_executable),
            str(server_script),
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        # The server is started as a background process and is not tied to this specific worker's lifecycle
        subprocess.Popen(command)
        print("XTTS Server process launched in the background.")


    def _wait_for_server(self, timeout=60):
        print("Waiting for XTTS server to become available...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                print("XTTS Server is up and running.")
                return
            time.sleep(1)
        
        raise RuntimeError("Failed to connect to the XTTS server within the specified timeout.")

    def stop_server(self):
        """
        In a multi-worker setup, a single client instance should not stop the shared server.
        The server will continue running until the main application is terminated.
        """
        if self.server_process:
            print("XTTS Client: An instance is shutting down, but the shared server will remain active for other workers.")
            self.server_process = None
    
    def __del__(self):
        """
        The destructor does not stop the server to prevent disrupting other workers.
        """
        pass

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Generate audio by calling the server's API"""
        payload = {"text": text, "voice": voice, **kwargs}
        response = requests.post(f"{self.base_url}/generate_audio", json=payload)
        response.raise_for_status()
        return response.content

    def list_voices(self, **kwargs) -> List[str]:
        """Get available voices from the server"""
        response = requests.get(f"{self.base_url}/list_voices")
        response.raise_for_status()
        return response.json().get("voices", [])


    def list_models(self) -> list:
        """Lists models"""
        response = requests.get(f"{self.base_url}/list_models")
        response.raise_for_status()
        return response.json().get("models", [])
        