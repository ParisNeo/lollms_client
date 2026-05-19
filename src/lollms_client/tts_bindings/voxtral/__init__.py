"""
LoLLMs TTS Binding — Voxtral Local (vLLM-Omni)
================================================
A zero-friction, self-installing binding that runs Mistral's
Voxtral-4B-TTS-2603 model locally via vLLM-Omni.

Philosophy
----------
• The user never has to install anything manually.
• First call auto-installs vLLM + vLLM-Omni into a dedicated venv via pipmaster.
• The vLLM-Omni server is started once (process-safe via file lock) and shared
  across all workers — no duplicate model loading, no OOM.
• On Windows: vLLM-Omni requires Linux. The binding detects Windows and gives a
  clear, friendly message with WSL2 / Docker instructions instead of crashing.
• On Linux / WSL2 / macOS: fully automatic.
"""

import os
import sys
import time
import shutil
import platform
import subprocess
import requests
from pathlib import Path
from typing import Optional, List

# ── filelock is our only hard dependency outside stdlib ─────────────────────
try:
    from filelock import FileLock, Timeout
except ImportError:
    # pipmaster may not be installed yet either — bootstrap it minimally
    subprocess.check_call([sys.executable, "-m", "pip", "install", "filelock", "-q"])
    from filelock import FileLock, Timeout

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "VoxtralLocalBinding"

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "mistralai/Voxtral-4B-TTS-2603"
DEFAULT_PORT    = 8030
HEALTH_ENDPOINT = "/health"
SPEECH_ENDPOINT = "/v1/audio/speech"
VOICES_ENDPOINT = "/v1/audio/voices"

# Preset voices shipped with Voxtral-4B-TTS-2603
PRESET_VOICES: List[str] = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam",  "am_michael",
    "bf_emma",  "bf_isabella",
    "bm_george", "bm_lewis",
]

# Required packages for the dedicated venv
VLLM_PACKAGES = [
    "vllm>=0.18.0",
    "vllm-omni>=0.18.0",
    "mistral_common",   # required by vLLM-Omni for the Voxtral protocol
    "httpx>=0.27.0",
    "soundfile>=0.12.1",
    "numpy>=1.24.0",
]


# ── Windows guard ────────────────────────────────────────────────────────────
def _check_platform():
    """
    vLLM-Omni requires Linux.  On Windows we check if we're inside WSL2 first.
    If not, we print a helpful message and raise a clear RuntimeError.
    """
    if platform.system() != "Windows":
        return  # Linux or macOS — fine

    # On Windows, check if we're actually running inside WSL2
    uname = platform.uname()
    if "microsoft" in uname.release.lower() or "wsl" in uname.release.lower():
        return  # Inside WSL2 — good to go

    msg = (
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║           VoxtralLocal — Windows is not supported            ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  vLLM-Omni (the inference backend for Voxtral) requires      ║\n"
        "║  Linux.  You have two easy options:                          ║\n"
        "║                                                              ║\n"
        "║  Option A — WSL2 (recommended, free)                        ║\n"
        "║    1. Open PowerShell as Admin and run:                      ║\n"
        "║         wsl --install -d Ubuntu-22.04                       ║\n"
        "║    2. Restart your PC, then open the Ubuntu terminal.        ║\n"
        "║    3. Run LoLLMs from inside that Ubuntu shell.              ║\n"
        "║    CUDA works transparently through WSL2 on NVIDIA GPUs.     ║\n"
        "║                                                              ║\n"
        "║  Option B — Docker with NVIDIA Container Toolkit             ║\n"
        "║    docker run --gpus all -p 8030:8030 \\                     ║\n"
        "║      vllm/vllm-omni:latest \\                                ║\n"
        "║      vllm serve mistralai/Voxtral-4B-TTS-2603 --omni \\      ║\n"
        "║      --port 8030 --host 0.0.0.0                              ║\n"
        "║    Then set auto_start_server=false and point the binding    ║\n"
        "║    at localhost:8030.                                        ║\n"
        "║                                                              ║\n"
        "║  Alternatively, use the 'mistral_tts' binding for the cloud  ║\n"
        "║  Mistral API — it works on any OS with no GPU required.     ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )
    ASCIIColors.red(msg)
    raise RuntimeError(
        "VoxtralLocal binding requires Linux or WSL2.  "
        "See the message above for setup instructions."
    )


# ── Main binding class ───────────────────────────────────────────────────────
class VoxtralLocalBinding(LollmsTTSBinding):
    """
    LoLLMs TTS binding for locally-served Voxtral-4B-TTS-2603.

    The first time this binding is used it:
      1. Creates a dedicated Python venv (Python 3.10).
      2. Installs vLLM, vLLM-Omni, and mistral_common into it.
      3. Starts a vLLM-Omni server process in the background.

    All subsequent workers/processes skip steps 1-3 and reuse the running
    server directly — vLLM-Omni handles concurrency and continuous batching
    internally.
    """

    def __init__(self, **kwargs):
        _check_platform()

        # Accept "model" as an alias for "model_name"
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        self.config       = kwargs
        self.host         = kwargs.get("host",         "localhost")
        self.port         = int(kwargs.get("port",     DEFAULT_PORT))
        self.model_name   = kwargs.get("model_name",   DEFAULT_MODEL)
        self.device       = kwargs.get("device",       "auto")
        self.hf_token     = kwargs.get("hf_token",     os.environ.get("HF_TOKEN", ""))
        self.gpu_mem_util = float(kwargs.get("gpu_memory_utilization", 0.6))
        self.default_voice = kwargs.get("default_voice", "af_bella")
        self.auto_start   = kwargs.get("auto_start_server", False)

        self.base_url     = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.venv_dir     = Path(kwargs.get("venv_dir",
                                  self.binding_root / ".venv_voxtral"))

        # Python version vLLM-Omni works with
        self.target_python = kwargs.get("python_version", "3.10")

        self._python_exe: Optional[str] = None
        self._server_proc: Optional[subprocess.Popen] = None

        if self.auto_start:
            self.ensure_server_is_running()

    # ── Health ────────────────────────────────────────────────────────────────

    def is_server_running(self) -> bool:
        """Returns True if the vLLM-Omni server is up and healthy."""
        try:
            r = requests.get(f"{self.base_url}{HEALTH_ENDPOINT}", timeout=2)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def ensure_server_is_running(self):
        """
        Idempotent: if the server is already up, return immediately.
        Otherwise acquire a file lock (so parallel workers don't race),
        install dependencies if needed, and start the server.
        """
        if self.is_server_running():
            ASCIIColors.green("VoxtralLocal: Server already running ✓")
            return

        lock_path = self.venv_dir.parent / "voxtral_server.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))

        ASCIIColors.info("VoxtralLocal: Acquiring startup lock…")
        try:
            with lock.acquire(timeout=15):
                # Double-check inside the lock (another worker may have started it)
                if self.is_server_running():
                    ASCIIColors.green("VoxtralLocal: Server started by another worker ✓")
                    return

                ASCIIColors.yellow("VoxtralLocal: Lock acquired — starting server…")
                self._ensure_dependencies()
                self._start_vllm_server()
                self._wait_for_server(timeout=180)
        except Timeout:
            ASCIIColors.yellow(
                "VoxtralLocal: Another process is starting the server, waiting…"
            )
            self._wait_for_server(timeout=180)

        if not self.is_server_running():
            raise RuntimeError(
                "VoxtralLocal: Server did not come up within the timeout.\n"
                "Check the server logs for details."
            )

    # ── Dependency installation ───────────────────────────────────────────────

    def _ensure_dependencies(self):
        """
        Uses pipmaster to create a dedicated Python venv and install
        vLLM-Omni + its dependencies if they aren't present yet.
        Skips installation if the venv already contains vllm-omni.
        """
        try:
            import pipmaster as pm
        except ImportError:
            ASCIIColors.yellow("VoxtralLocal: Installing pipmaster…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pipmaster", "-q"]
            )
            import pipmaster as pm

        ASCIIColors.info(
            f"VoxtralLocal: Setting up Python {self.target_python} venv "
            f"at {self.venv_dir} …"
        )

        try:
            pm_instance = pm.get_pip_manager_for_version(
                self.target_python, str(self.venv_dir)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"VoxtralLocal: Could not bootstrap Python {self.target_python}: {e}\n"
                "Install it via your system package manager:\n"
                "  sudo apt install python3.10 python3.10-venv  (Debian/Ubuntu)\n"
                "  sudo dnf install python3.10                   (Fedora)\n"
                "  brew install python@3.10                      (macOS)\n"
            ) from e

        self._python_exe = pm_instance.target_python_executable

        # Check if vllm-omni is already installed in the venv
        probe = subprocess.run(
            [self._python_exe, "-c", "import vllm; import vllm_omni"],
            capture_output=True,
        )
        if probe.returncode == 0:
            ASCIIColors.green("VoxtralLocal: vLLM-Omni already installed ✓")
            return

        ASCIIColors.yellow(
            "VoxtralLocal: Installing vLLM + vLLM-Omni (this may take a few minutes "
            "on the first run — subsequent starts are instant)…"
        )

        for pkg in VLLM_PACKAGES:
            ASCIIColors.info(f"  → {pkg}")
            ok = pm_instance.install_if_missing(pkg, verbose=True)
            if not ok:
                raise RuntimeError(
                    f"VoxtralLocal: Failed to install '{pkg}'.\n"
                    "Please check your internet connection and CUDA installation."
                )

        ASCIIColors.green("VoxtralLocal: All dependencies installed ✓")

    # ── Start vLLM-Omni ──────────────────────────────────────────────────────

    def _start_vllm_server(self):
        """
        Launches `vllm serve <model> --omni` as a detached background process
        using the dedicated venv's Python interpreter.
        """
        if self._python_exe is None:
            # Venv already existed — resolve executable path
            if sys.platform == "win32":
                self._python_exe = str(self.venv_dir / "Scripts" / "python.exe")
            else:
                self._python_exe = str(self.venv_dir / "bin" / "python")

        # Build the vllm serve command
        cmd = [
            self._python_exe, "-m", "vllm.entrypoints.openai.api_server",
            "--model",  self.model_name,
            "--host",   self.host,
            "--port",   str(self.port),
            "--omni",
            "--gpu-memory-utilization", str(self.gpu_mem_util),
        ]

        if self.device == "cpu":
            cmd += ["--device", "cpu"]
        else:
            cmd += ["--dtype", "bfloat16"]

        if self.hf_token:
            cmd += ["--hf-token", self.hf_token]

        # Pass HF_TOKEN via environment too (some versions need it there)
        env = os.environ.copy()
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

        ASCIIColors.info(f"VoxtralLocal: Launching vLLM-Omni server…")
        ASCIIColors.info(f"  Command: {' '.join(cmd)}")

        # On Linux we use start_new_session so the server outlives the parent
        kwargs = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        self._server_proc = subprocess.Popen(cmd, env=env, **kwargs)
        ASCIIColors.info(
            f"VoxtralLocal: Server process started (PID {self._server_proc.pid})"
        )

    def _wait_for_server(self, timeout: int = 180):
        """Poll /health until the server responds or timeout is reached."""
        ASCIIColors.info(f"VoxtralLocal: Waiting up to {timeout}s for server…")
        deadline = time.time() + timeout
        last_log = time.time()
        while time.time() < deadline:
            if self.is_server_running():
                ASCIIColors.green("VoxtralLocal: Server is ready ✓")
                return
            if time.time() - last_log > 15:
                elapsed = int(time.time() - (deadline - timeout))
                ASCIIColors.info(
                    f"VoxtralLocal: Still waiting… ({elapsed}s elapsed, "
                    "model download may be in progress)"
                )
                last_log = time.time()
            time.sleep(3)
        raise RuntimeError(
            f"VoxtralLocal: Server did not respond within {timeout}s.\n"
            "The model may still be downloading — try increasing the timeout via "
            "the 'startup_timeout' config key."
        )

    def __del__(self):
        # Intentionally do NOT stop the server — it is a shared resource.
        pass

    # ── Core TTS interface ────────────────────────────────────────────────────

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs,
    ) -> bytes:
        """
        Synthesise *text* and return raw audio bytes (WAV by default).

        Args:
            text:             The text to synthesise.
            voice:            Preset voice name (e.g. 'bm_george') or the name
                              of a previously uploaded custom voice.
            language:         Language code: en, fr, de, es, nl, pt, it, hi, ar.
            **kwargs:         Extra parameters forwarded to vLLM-Omni:
                              response_format, speed, stream, instructions, …
        Returns:
            Raw audio bytes in the requested format (default: WAV).
        """
        self.ensure_server_is_running()

        selected_voice = voice or self.default_voice
        payload = {
            "model":           self.model_name,
            "input":           text,
            "voice":           selected_voice,
            "language":        language,
            "response_format": kwargs.pop("response_format", "wav"),
            "speed":           kwargs.pop("speed", 1.0),
        }
        # Forward any extra kwargs (instructions, stream, max_new_tokens, …)
        payload.update(kwargs)

        ASCIIColors.info(
            f"VoxtralLocal: Synthesising | voice={selected_voice} "
            f"lang={language} chars={len(text)}"
        )

        try:
            response = requests.post(
                f"{self.base_url}{SPEECH_ENDPOINT}",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            detail = ""
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text if e.response is not None else ""
            ASCIIColors.error(f"VoxtralLocal: HTTP {status} — {detail}")
            raise RuntimeError(f"VoxtralLocal server error ({status}): {detail}") from e
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"VoxtralLocal: Network error — {e}")
            raise RuntimeError(
                "VoxtralLocal: Could not reach the server. "
                "Is it still running?"
            ) from e

        audio_bytes = response.content
        ASCIIColors.green(f"VoxtralLocal: Got {len(audio_bytes):,} bytes ✓")
        return audio_bytes

    # ── Voice management ──────────────────────────────────────────────────────

    def list_voices(self, **kwargs) -> List[str]:
        """
        Returns all available voices.
        Queries the live vLLM-Omni server when running (includes uploaded
        custom voices); falls back to the static preset list otherwise.
        """
        if self.is_server_running():
            try:
                r = requests.get(
                    f"{self.base_url}{VOICES_ENDPOINT}", timeout=5
                )
                r.raise_for_status()
                data   = r.json()
                voices = data.get("voices", [])
                if voices:
                    return voices
            except Exception as e:
                ASCIIColors.warning(f"VoxtralLocal: Could not fetch voices from server: {e}")
        return list(PRESET_VOICES)

    def list_models(self, **kwargs) -> List[str]:
        return [DEFAULT_MODEL]

    def upload_voice(
        self,
        voice_path: str,
        voice_name: Optional[str] = None,
        ref_text:   Optional[str] = None,
    ) -> dict:
        """
        Upload a custom reference audio file to vLLM-Omni for voice cloning.

        Voxtral-4B-TTS-2603 supports voice cloning via gated upstream access.
        If ref_text is provided the server uses higher-quality in-context
        cloning; without it only speaker embedding is extracted.

        Args:
            voice_path:  Path to WAV / MP3 file (3–25 s recommended).
            voice_name:  Name to register the voice under.
            ref_text:    Optional transcript of the reference audio
                         (improves clone quality significantly).

        Returns:
            dict with keys: success (bool), voice_name (str), message (str).
        """
        self.ensure_server_is_running()

        vf = Path(voice_path)
        if not vf.exists():
            return {
                "success": False,
                "voice_name": None,
                "message": f"File not found: {voice_path}",
            }

        allowed = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".mp4"}
        if vf.suffix.lower() not in allowed:
            return {
                "success":    False,
                "voice_name": None,
                "message":    f"Unsupported format '{vf.suffix}'. Use one of {allowed}.",
            }

        name = voice_name or vf.stem

        try:
            with open(vf, "rb") as f:
                files = {"audio_sample": (vf.name, f, f"audio/{vf.suffix.lstrip('.')}")}
                data  = {
                    "name":    name,
                    "consent": "user_consent",   # required by vLLM-Omni API
                }
                if ref_text:
                    data["ref_text"] = ref_text

                r = requests.post(
                    f"{self.base_url}{VOICES_ENDPOINT}",
                    files=files,
                    data=data,
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()

            voice_data = result.get("voice", {})
            return {
                "success":    result.get("success", True),
                "voice_name": voice_data.get("name", name),
                "message": (
                    f"Voice '{voice_data.get('name', name)}' uploaded. "
                    f"Use voice='{voice_data.get('name', name)}' in generate_audio()."
                ),
            }

        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"VoxtralLocal: Upload failed — {e}")
            return {
                "success":    False,
                "voice_name": None,
                "message":    f"Upload failed: {e}",
            }

    def delete_voice(self, voice_name: str) -> dict:
        """
        Delete a previously uploaded custom voice from the server.

        Args:
            voice_name: The name used when uploading the voice.

        Returns:
            dict with keys: success (bool), message (str).
        """
        self.ensure_server_is_running()
        try:
            r = requests.delete(
                f"{self.base_url}{VOICES_ENDPOINT}/{voice_name}",
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "success": data.get("success", True),
                "message": data.get("message", f"Voice '{voice_name}' deleted."),
            }
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"VoxtralLocal: Delete voice failed — {e}")
            return {"success": False, "message": str(e)}
