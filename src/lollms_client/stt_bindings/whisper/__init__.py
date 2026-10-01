import os
import sys
import base64
import signal
import argparse
import subprocess
import threading
import time
import json
import secrets
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors

try:
    import pipmaster as pm
except ImportError:
    print("FATAL: pipmaster is not installed. Please install it using: pip install pipmaster")
    sys.exit(1)

try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from lollms_client.lollms_stt_binding import LollmsSTTBinding

BindingName = "WhisperSTTBinding"

_PID_FILE_NAME = "whisper_server.pid"
_GRACEFUL_SHUTDOWN_TIMEOUT_S = 10
_FORCE_KILL_GRACE_S = 5


class WhisperSTTBinding(LollmsSTTBinding):
    """
    Speech-To-Text binding for OpenAI Whisper using a resilient, multi-process
    shared model server architecture with event-driven dynamic micro-batching.
    Multiple lollms_client instances and worker processes attach to the same running daemon.
    """

    def __init__(self, **kwargs):
        super().__init__(binding_name="whisper")
        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9633))
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.batch_window = float(kwargs.get("batch_window", 0.02))
        self.max_batch_size = int(kwargs.get("max_batch_size", 8))
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        system_root = self.get_system_dir()
        if kwargs.get("venv_path"):
            self.venv_dir = self.resolve_system_path(kwargs["venv_path"])
        else:
            self.venv_dir = (system_root / "venv" / "stt_whisper_venv").resolve()

        if kwargs.get("cache_dir"):
            self.cache_dir = self.resolve_system_path(kwargs["cache_dir"])
        else:
            self.cache_dir = (system_root / "data" / "stt_models" / "whisper").resolve()

        self.venv_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.token_file = self.cache_dir / "whisper_server.token"
        self.pid_file = self.cache_dir / _PID_FILE_NAME
        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        """Probes the server on loopback with a fast timeout (<= 0.5s)."""
        try:
            resp = requests.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=0.5
            )
            if resp.status_code == 200:
                data = resp.json() if callable(getattr(resp, "json", None)) else {}
                return data.get("status") == "running" if isinstance(data, dict) else True
            elif resp.status_code == 401:
                return True
        except Exception:
            return False
        return False

    def ensure_server_is_running(self, wait: bool = True, timeout_s: int = 120):
        """
        Ensures the shared Whisper server daemon is running without port collisions.
        Uses cross-process FileLock with double-checked probing to guarantee that
        only the first worker process spawns the daemon while all others attach to it.
        """
        if self.is_server_running():
            return

        lock_path = self.cache_dir / "whisper_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"Whisper shared daemon detected on {self.base_url}. Attached successfully.")
                    return

                ASCIIColors.info(f"Spawning shared Whisper server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for Whisper shared daemon to start after {timeout_s}s.")

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up Whisper virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        ASCIIColors.info("Installing Whisper server dependencies...")
        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors>=0.11.10", "pipmaster", "tqdm", "numpy", "pillow", "pydantic"])

        torch_index_url = None
        if sys.platform == "win32":
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu126"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("No GPU detected or nvidia-smi failed. Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchaudio"], index_url=torch_index_url)
        pm_v.ensure_packages(["openai-whisper"])
        pm_v.ensure_packages(["psutil"])
        ASCIIColors.green("Whisper server dependencies are satisfied.")

    def start_server(self, wait: bool = True, timeout_s: int = 120):
        server_script = self.server_dir / "main.py"
        venv_cfg = self.venv_dir / "pyvenv.cfg"

        if not venv_cfg.exists():
            self.install_server_dependencies()

        if sys.platform == "win32":
            python_executable = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_dir / "bin" / "python"

        if not python_executable.exists():
            raise RuntimeError(f"Python executable not found in venv: {python_executable}.")

        if not self.service_key:
            if self.token_file.exists():
                try:
                    self.service_key = self.token_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            if not self.service_key:
                self.service_key = secrets.token_hex(16)
                try:
                    fd = os.open(str(self.token_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(self.service_key)
                except Exception:
                    self.token_file.write_text(self.service_key, encoding="utf-8")

        command = [
            str(python_executable),
            str(server_script),
            "--host", str(self.host),
            "--port", str(self.port),
            "--cache-dir", str(self.cache_dir.resolve()),
            "--token", str(self.service_key),
            "--batch-window", str(self.batch_window),
            "--max-batch-size", str(self.max_batch_size),
        ]

        log_file_path = self.cache_dir / "whisper_server.log"
        log_f = open(log_file_path, "w", encoding="utf-8")

        try:
            popen_kwargs: Dict[str, Any] = {
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                popen_kwargs["start_new_session"] = True

            self.server_process = subprocess.Popen(command, **popen_kwargs)
        finally:
            log_f.close()

        self._write_pid_file(self.server_process.pid)
        ASCIIColors.info(f"Whisper server process launched on http://{self.host}:{self.port} (PID: {self.server_process.pid})")

        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                if self.server_process.poll() is not None:
                    error_tail = "Log file is empty."
                    try:
                        if log_file_path.exists():
                            lines = log_file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                            error_tail = "\n".join(lines[-30:])
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Whisper server process terminated unexpectedly with code {self.server_process.returncode}.\n"
                        f"Log tail:\n{error_tail}"
                    )

                if self.is_server_running():
                    ASCIIColors.success(f"Whisper shared daemon is operational on {self.base_url}.")
                    return

                time.sleep(0.5)

            raise TimeoutError(f"Whisper server failed to become responsive within {timeout_s} seconds.")

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(
                url,
                json=data,
                headers=self._get_headers(),
                timeout=3600
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Whisper server at {url}. Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_detail = e.response.json().get('detail', e.response.text)
                except Exception:
                    err_detail = e.response.text
                raise RuntimeError(f"Whisper server error: {err_detail}") from e
            raise RuntimeError(f"Communication with Whisper server failed: {e}") from e

    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=15
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Communication with Whisper server failed: {e}") from e

    def transcribe_audio(self, audio_source: Union[str, Path, bytes], model: Optional[str] = None, **kwargs) -> str:
        self.ensure_server_is_running(wait=True)

        if isinstance(audio_source, (str, Path)):
            audio_file = Path(audio_source)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found at: {audio_source}")
            audio_bytes = audio_file.read_bytes()
            filename_hint = audio_file.name
        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
            filename_hint = kwargs.get("filename")
        else:
            raise ValueError("audio_source must be str, Path, or bytes")

        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        payload = {
            "audio_b64": audio_b64,
            "model_name": model or self.config.get("model_name", "base"),
            "language": kwargs.get("language"),
            "task": kwargs.get("task", "transcribe"),
            "fp16": kwargs.get("fp16"),
            "device": kwargs.get("device"),
            "filename": filename_hint
        }

        response = self._post_json_request("/transcribe", data=payload)
        return response.json().get("text", "")

    def transcribe_audio_with_diarization(
        self,
        audio_source: Union[str, Path, bytes],
        participants: Optional[List[str]] = None,
        voice_samples: Optional[Dict[str, Union[str, Path, bytes]]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribes audio with speaker diarization and voice matching.

        Segments speech, extracts acoustic embeddings using Whisper's audio encoder,
        and clusters them into distinct speakers. If voice_samples are provided,
        clusters are matched to reference voices. Unmatched clusters are mapped
        to participants in order of appearance, or 'Speaker 1', 'Speaker 2', etc.

        Returns:
            List[Dict[str, Any]]: List of dialogue turns with 'speaker', 'start', 'end', and 'text'.
        """
        self.ensure_server_is_running(wait=True)

        if isinstance(audio_source, (str, Path)):
            audio_file = Path(audio_source)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found at: {audio_source}")
            audio_bytes = audio_file.read_bytes()
            filename_hint = audio_file.name
        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
            filename_hint = kwargs.get("filename")
        else:
            raise ValueError("audio_source must be str, Path, or bytes")

        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Encode reference voice samples to base64 if provided
        encoded_samples: Optional[Dict[str, str]] = None
        if voice_samples:
            encoded_samples = {}
            for spk_name, sample_src in voice_samples.items():
                if isinstance(sample_src, (str, Path)):
                    sp = Path(sample_src)
                    if sp.exists() and sp.is_file():
                        encoded_samples[spk_name] = base64.b64encode(sp.read_bytes()).decode('utf-8')
                elif isinstance(sample_src, bytes):
                    encoded_samples[spk_name] = base64.b64encode(sample_src).decode('utf-8')

        payload = {
            "audio_b64": audio_b64,
            "participants": participants,
            "voice_samples": encoded_samples,
            "model_name": model or self.config.get("model_name", "base"),
            "language": kwargs.get("language"),
            "task": kwargs.get("task", "transcribe"),
            "fp16": kwargs.get("fp16"),
            "device": kwargs.get("device"),
            "filename": filename_hint
        }

        response = self._post_json_request("/transcribe_diarize", data=payload)
        res_json = response.json()
        turns = res_json.get("turns", [])
        return turns

    def shutdown_server(self) -> bool:
        """Sends an authenticated shutdown command to terminate the background server daemon."""
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(
                f"{self.base_url}/shutdown",
                headers=self._get_headers(),
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _write_pid_file(self, pid: int) -> None:
        try:
            self.pid_file.write_text(str(pid), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"Could not write PID file {self.pid_file}: {e}")

    def _read_pid_file(self) -> Optional[int]:
        try:
            if self.pid_file.exists():
                return int(self.pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
        return None

    def _delete_pid_file(self) -> None:
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _is_our_server_process(self, pid: int) -> bool:
        """Verifies the PID belongs to our Whisper server before any kill attempt."""
        try:
            import psutil
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).lower()
            return "whisper" in cmdline and "server" in cmdline and "main.py" in cmdline
        except ImportError:
            return True
        except Exception:
            return False

    def _terminate_pid(self, pid: int) -> bool:
        """Cross-platform process termination with escalation to force kill."""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=_FORCE_KILL_GRACE_S)
                return True
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_FORCE_KILL_GRACE_S)
                return True
        except ImportError:
            pass
        except Exception:
            return False

        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, check=True
                )
            else:
                os.kill(pid, signal.SIGKILL)
            return True
        except Exception:
            return False

    def kill(self, force: bool = False) -> dict:
        """
        Terminates the Whisper server daemon.

        Graceful mode sends the authenticated HTTP /shutdown command and waits
        for the port to be released. Force mode (or an unresponsive daemon)
        escalates to OS-level process termination using the persisted PID file.

        Returns:
            dict: {"status": bool, "message": str}
        """
        was_running = self.is_server_running()
        pid = self._read_pid_file()

        if was_running and not force:
            if self.shutdown_server():
                start_time = time.time()
                while time.time() - start_time < _GRACEFUL_SHUTDOWN_TIMEOUT_S:
                    if not self.is_server_running():
                        self._delete_pid_file()
                        ASCIIColors.success("Whisper server shut down gracefully.")
                        return {"status": True, "message": "Whisper server shut down gracefully."}
                    time.sleep(0.3)
                ASCIIColors.warning("Graceful shutdown timed out. Escalating to force kill.")
            else:
                ASCIIColors.warning("HTTP shutdown failed. Escalating to force kill.")

        if pid is not None and self._is_our_server_process(pid):
            if self._terminate_pid(pid):
                self._delete_pid_file()
                ASCIIColors.success(f"Whisper server process (PID {pid}) terminated.")
                return {"status": True, "message": f"Whisper server process (PID {pid}) terminated."}
            ASCIIColors.error(f"Failed to terminate Whisper server process (PID {pid}).")
            return {"status": False, "message": f"Failed to terminate Whisper server process (PID {pid})."}

        if was_running:
            msg = (
                "Server is running but no valid PID file was found. "
                "Kill it manually or restart with a fresh cache directory."
            )
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}

        self._delete_pid_file()
        return {"status": True, "message": "Whisper server was not running."}

    def restart(self) -> dict:
        """
        Force-kills the daemon and starts a fresh instance with current code.

        Returns:
            dict: {"status": bool, "message": str}
        """
        kill_result = self.kill(force=True)
        if not kill_result.get("status", False):
            return kill_result

        try:
            self.ensure_server_is_running(wait=True)
            if self.is_server_running():
                return {"status": True, "message": "Whisper server restarted and operational."}
            return {"status": False, "message": "Whisper server failed to become responsive after restart."}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Whisper server restart failed: {e}"}

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]

    def status(self) -> dict:
        """
        Returns the daemon status and connection information.

        Returns:
            dict: {"status": bool, "message": str, "data": dict}
        """
        if self.is_server_running():
            return {
                "status": True,
                "message": f"Whisper server is running on {self.base_url}.",
                "data": {"running": True, "base_url": self.base_url},
            }
        return {
            "status": False,
            "message": "Whisper server is not running.",
            "data": {"running": False, "base_url": self.base_url},
        }


def _build_cli_binding(args: argparse.Namespace) -> WhisperSTTBinding:
    return WhisperSTTBinding(
        host=args.host,
        port=args.port,
        auto_start_server=False,
        wait_for_server=False,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lollms-whisper",
        description="Management CLI for the shared Whisper STT daemon.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=9633, help="Server port")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override the models cache directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show daemon status")
    subparsers.add_parser("ps", help="List loaded models and queue state")
    subparsers.add_parser("kill", help="Terminate the daemon (graceful, then force)")
    subparsers.add_parser("restart", help="Force-kill and relaunch the daemon")

    args = parser.parse_args()

    binding = _build_cli_binding(args)

    if args.command == "status":
        result = binding.status()
        print(json.dumps(result.get("data", result), indent=2))
        sys.exit(0 if result.get("status") else 1)
    elif args.command == "ps":
        print(json.dumps(binding.ps(), indent=2))
    elif args.command == "kill":
        result = binding.kill(force=False)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") else 1)
    elif args.command == "restart":
        result = binding.restart()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") else 1)


if __name__ == "__main__":
    main()