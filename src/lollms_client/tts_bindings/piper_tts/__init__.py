# File: lollms_client/tts_bindings/piper/__init__.py
from lollms_client.lollms_tts_binding import LollmsTTSBinding
from typing import Optional, List
from pathlib import Path
import requests
import subprocess
import sys
import time
import pipmaster as pm

BindingName = "PiperClientBinding"

class PiperClientBinding(LollmsTTSBinding):
    def __init__(self,
                 **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        self.host = self.config.get("host", "http://localhost")
        self.port = self.config.get("port", 9632)
        self.auto_start_server = self.config.get("auto_start_server", True)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"

        if self.auto_start_server:
            self.start_server()

    def start_server(self):
        print("Piper Client: Starting dedicated server...")
        binding_root = Path(__file__).parent
        server_dir = binding_root / "server"
        requirements_file = server_dir / "requirements.txt"
        server_script = server_dir / "main.py"

        # 1. Ensure a virtual environment and dependencies
        venv_path = server_dir / "venv"
        pm_v = pm.PackageManager(venv_path=venv_path)
        pm_v.ensure_requirements(str(requirements_file), verbose=True)

        # 2. Get the python executable from the venv
        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"

        # 3. Launch the server as a subprocess with stdout/stderr forwarded to console
        command = [
            str(python_executable),
            str(server_script),
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        # Forward stdout and stderr to the parent process console
        self.server_process = subprocess.Popen(
            command,
            stdout=None,  # Inherit parent's stdout (shows in console)
            stderr=None,  # Inherit parent's stderr (shows in console)
        )
        
        # 4. Wait for the server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout=60):  # Piper is fast to load
        start_time = time.time()
        print("Piper Client: Waiting for server to initialize...")
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/status")
                if response.status_code == 200 and response.json().get("status") == "running":
                    print("Piper Server is up and running.")
                    return
            except requests.ConnectionError:
                time.sleep(1)
        
        self.stop_server()
        raise RuntimeError("Failed to start the Piper server in the specified timeout.")

    def stop_server(self):
        if self.server_process:
            print("Piper Client: Stopping dedicated server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            print("Server stopped.")
    
    def __del__(self):
        # Ensure the server is stopped when the object is destroyed
        self.stop_server()

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Generate audio by calling the server's API"""
        payload = {"text": text, "voice": voice, **kwargs}
        response = requests.post(f"{self.base_url}/generate_audio", json=payload, timeout=30)
        response.raise_for_status()
        return response.content

    def list_voices(self, **kwargs) -> List[str]:
        """Get available voices from the server"""
        response = requests.get(f"{self.base_url}/list_voices")
        response.raise_for_status()
        return response.json().get("voices", [])

    def list_models(self) -> List[str]:
        """Get available models from the server"""
        response = requests.get(f"{self.base_url}/list_models")
        response.raise_for_status()
        return response.json().get("models", [])

    def download_voice(self, voice_name: str):
        """Download a specific voice model"""
        response = requests.post(f"{self.base_url}/download_voice", json={"voice": voice_name})
        response.raise_for_status()
        return response.json()

    def set_voice(self, voice: str):
        """Set the default voice for future generations"""
        response = requests.post(f"{self.base_url}/set_voice", json={"voice": voice})
        response.raise_for_status()
        return response.json()
