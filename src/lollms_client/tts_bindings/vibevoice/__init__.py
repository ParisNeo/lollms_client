import os
import sys
import requests
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure filelock is available for process-safe server startup.
try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "VibeVoiceBinding"

class VibeVoiceBinding(LollmsTTSBinding):
    """
    Client binding for the VibeVoice real-time TTS server.
    """
    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9634)
        self.auto_start_server = kwargs.get("auto_start_server", True)
        
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path("./venv/tts_vibevoice_venv")
        self.server_process = None

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
        """Ensures the VibeVoice server is running using a file lock."""
        self.server_dir.mkdir(exist_ok=True)
        lock_path = self.server_dir / "vibevoice_server.lock"
        lock = FileLock(lock_path)

        ASCIIColors.info("Checking VibeVoice server status...")
        
        if self.is_server_running():
            ASCIIColors.green("VibeVoice Server is already running.")
            return

        try:
            with lock.acquire(timeout=10):
                if not self.is_server_running():
                    ASCIIColors.yellow("Lock acquired. Starting VibeVoice server...")
                    self.start_server()
                    self._wait_for_server()
                else:
                    ASCIIColors.green("Server started by another process. Connected.")
        except Timeout:
            ASCIIColors.yellow("Waiting for another process to start the server...")
            self._wait_for_server(timeout=60)

        if not self.is_server_running():
            raise RuntimeError("Failed to start or connect to the VibeVoice server.")

    def install_server_dependencies(self):
        """Installs server dependencies into a dedicated venv."""
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        try:
            import pipmaster as pm
        except ImportError:
            print("FATAL: pipmaster is not installed. Run: pip install pipmaster")
            raise Exception("pipmaster not found")
        
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir))
        requirements_file = self.server_dir / "requirements.txt"
        
        ASCIIColors.info("Installing VibeVoice dependencies...")
        success = pm_v.ensure_requirements(str(requirements_file), verbose=True)

        if not success:
            raise RuntimeError("VibeVoice dependency installation failed.")
        ASCIIColors.green("Dependencies installed.")

    def start_server(self):
        """Launches the FastAPI server."""
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
        
        creationflags = subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        self.server_process = subprocess.Popen(command, creationflags=creationflags)
        ASCIIColors.info("VibeVoice server process launched.")

    def _wait_for_server(self, timeout=10):
        """Waits for server API to respond."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                return
            time.sleep(1)
        raise RuntimeError("VibeVoice server connection timed out.")

    def __del__(self):
        pass

    # --- Standard TTS Methods ---

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """Standard TTS generation."""
        payload = {"text": text, "voice": voice}
        payload.update(kwargs)
        
        try:
            response = requests.post(f"{self.base_url}/generate_audio", json=payload, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"VibeVoice generation error: {e}")
            raise RuntimeError("VibeVoice generation failed.") from e

    def list_voices(self, **kwargs) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/list_voices")
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.exceptions.RequestException:
            return []

    def list_models(self, **kwargs) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/list_models")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException:
            return []

    # --- Action Methods (Defined in description.yaml) ---

    def start_conversation(self, session_id: str) -> Dict[str, Any]:
        """Action: Initializes a conversation session."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/actions/start_conversation", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ASCIIColors.error(f"Action start_conversation failed: {e}")
            return {"status": False, "message": str(e)}

    def stop_conversation(self) -> Dict[str, Any]:
        """Action: Stops the current conversation."""
        try:
            response = requests.post(f"{self.base_url}/actions/stop_conversation", json={})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ASCIIColors.error(f"Action stop_conversation failed: {e}")
            return {"status": False}

    def change_voice_tone(self, emotion: str) -> Dict[str, Any]:
        """Action: Changes the voice emotion in real-time."""
        try:
            payload = {"emotion": emotion}
            response = requests.post(f"{self.base_url}/actions/change_voice_tone", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ASCIIColors.error(f"Action change_voice_tone failed: {e}")
            return {"success": False}
