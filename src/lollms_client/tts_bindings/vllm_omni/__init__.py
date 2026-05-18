"""
LoLLMs TTS Binding — vLLM-Omni
================================
A zero-friction, self-installing binding that serves any vLLM-Omni-compatible
TTS model locally (Voxtral, Qwen3-TTS, Fish Speech, CosyVoice3, OmniVoice, …).

Philosophy
----------
• The user never has to install anything manually.
• First call auto-installs vLLM + vLLM-Omni into a dedicated venv via pipmaster.
• The vLLM-Omni server is started once (process-safe via file lock) and shared
  across all workers — no duplicate model loading, no OOM.
• On Windows: vLLM-Omni requires Linux. The binding detects Windows and prints
  clear WSL2 / Docker setup instructions instead of crashing.
• On Linux / WSL2 / macOS: fully automatic.

Supported models (as of vLLM-Omni 0.18+)
------------------------------------------
  mistralai/Voxtral-4B-TTS-2603            (~5 GB VRAM, 11 preset voices)
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice    (~3 GB VRAM, multilingual presets)
  Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign    (~3 GB VRAM, NL style instructions)
  Qwen/Qwen3-TTS-12Hz-1.7B-Base           (~3 GB VRAM, voice cloning)
  fishaudio/s2-pro                         (~8 GB VRAM, 44.1 kHz, voice cloning)
  FunAudioLLM/Fun-CosyVoice3-0.5B-2512    (~2 GB VRAM, voice cloning)
  k2-fsa/OmniVoice                         (~4 GB VRAM, voice cloning)
  openbmb/VoxCPM2                          (~5 GB VRAM, presets + cloning)
  OpenMOSS-Team/MOSS-TTS-Nano              (~1 GB VRAM, voice cloning)
"""

import os
import sys
import time
import platform
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Bootstrap filelock without requiring a pre-existing environment ──────────
try:
    from filelock import FileLock, Timeout
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "filelock", "-q"])
    from filelock import FileLock, Timeout

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "VllmOmniTTSBinding"

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "mistralai/Voxtral-4B-TTS-2603"
DEFAULT_PORT    = 8030
HEALTH_ENDPOINT = "/health"
SPEECH_ENDPOINT = "/v1/audio/speech"
VOICES_ENDPOINT = "/v1/audio/voices"

# Known preset voices per model HF ID (used as fallback when server is down)
MODEL_PRESET_VOICES: Dict[str, List[str]] = {
    "mistralai/Voxtral-4B-TTS-2603": [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam",  "am_michael",
        "bf_emma",  "bf_isabella",
        "bm_george", "bm_lewis",
    ],
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": [
        "Ethan",  "Isabella",  "Liam",   "Mia",      "Noah",
        "Olivia", "Sophia",    "Aiden",  "Charlotte", "Harper",
    ],
    "openbmb/VoxCPM2": [
        "male_01", "female_01", "male_02", "female_02",
    ],
}

# Packages installed into the dedicated venv
VLLM_PACKAGES = [
    "vllm>=0.18.0",
    "vllm-omni>=0.18.0",
    "mistral_common",
    "httpx>=0.27.0",
    "soundfile>=0.12.1",
    "numpy>=1.24.0",
]


# ── Windows guard ────────────────────────────────────────────────────────────
def _check_platform() -> None:
    """
    vLLM-Omni requires Linux.  On Windows we check for WSL2 first.
    If running on native Windows, print a friendly help box and raise.
    """
    if platform.system() != "Windows":
        return  # Linux / macOS — fine

    uname = platform.uname()
    if "microsoft" in uname.release.lower() or "wsl" in uname.release.lower():
        return  # Inside WSL2 — fine

    msg = (
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║         vLLM-Omni TTS — Windows is not supported            ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  vLLM-Omni requires Linux.  Two easy options:               ║\n"
        "║                                                              ║\n"
        "║  Option A — WSL2 (free, recommended)                        ║\n"
        "║    1. Open PowerShell as Administrator:                      ║\n"
        "║         wsl --install -d Ubuntu-22.04                       ║\n"
        "║    2. Restart your PC, then open the Ubuntu app.            ║\n"
        "║    3. Run LoLLMs from inside that Ubuntu shell.              ║\n"
        "║    Your NVIDIA GPU is fully accessible from WSL2.           ║\n"
        "║                                                              ║\n"
        "║  Option B — Docker + NVIDIA Container Toolkit               ║\n"
        "║    docker run --gpus all -p 8030:8030 \\                     ║\n"
        "║      vllm/vllm-omni:latest \\                                ║\n"
        "║      vllm serve mistralai/Voxtral-4B-TTS-2603 \\             ║\n"
        "║      --omni --port 8030 --host 0.0.0.0                      ║\n"
        "║    Then set auto_start_server: false in your config and      ║\n"
        "║    point the binding at localhost:8030.                      ║\n"
        "║                                                              ║\n"
        "║  Want cloud TTS with no GPU?  Use the 'mistral_tts' binding  ║\n"
        "║  — it works on any OS with just an API key.                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )
    ASCIIColors.red(msg)
    raise RuntimeError(
        "vLLM-Omni TTS binding requires Linux or WSL2.  "
        "See the message above for setup instructions."
    )


# ── Main binding class ───────────────────────────────────────────────────────
class VllmOmniTTSBinding(LollmsTTSBinding):
    """
    LoLLMs TTS binding for any vLLM-Omni-compatible TTS model.

    On first use the binding:
      1. Bootstraps pipmaster if absent.
      2. Creates a dedicated Python 3.10 venv.
      3. Installs vLLM + vLLM-Omni into it.
      4. Starts a vLLM-Omni server process in the background.

    All workers / processes share that one server — vLLM-Omni handles
    concurrency, request queuing and continuous batching internally.

    Example configs
    ---------------
    # Voxtral (default)
    VllmOmniTTSBinding(model_name="mistralai/Voxtral-4B-TTS-2603",
                       gpu_memory_utilization=0.6)

    # Qwen3-TTS — lighter, multilingual presets
    VllmOmniTTSBinding(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                       extra_args=["--trust-remote-code", "--enforce-eager",
                                   "--deploy-config",
                                   "vllm_omni/deploy/qwen3_tts.yaml"])

    # Fish Speech S2 Pro — 44.1 kHz, voice cloning
    VllmOmniTTSBinding(model_name="fishaudio/s2-pro",
                       gpu_memory_utilization=0.85)

    # MOSS-TTS-Nano — tiny, fits in 1 GB VRAM
    VllmOmniTTSBinding(model_name="OpenMOSS-Team/MOSS-TTS-Nano",
                       gpu_memory_utilization=0.25)
    """

    def __init__(self, **kwargs):
        _check_platform()

        # Accept "model" as alias for "model_name"
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        self.config         = kwargs
        self.host           = kwargs.get("host",          "localhost")
        self.port           = int(kwargs.get("port",      DEFAULT_PORT))
        self.model_name     = kwargs.get("model_name",    DEFAULT_MODEL)
        self.device         = kwargs.get("device",        "auto")
        self.hf_token       = kwargs.get("hf_token",      os.environ.get("HF_TOKEN", ""))
        self.gpu_mem_util   = float(kwargs.get("gpu_memory_utilization", 0.6))
        self.default_voice  = kwargs.get("default_voice", "")
        self.auto_start     = kwargs.get("auto_start_server", False)

        # Model-specific extra CLI args for vllm serve
        # e.g. ["--trust-remote-code", "--deploy-config", "path/to/config.yaml"]
        self.extra_args: List[str] = kwargs.get("extra_args", [])

        self.base_url       = f"http://{self.host}:{self.port}"
        self.binding_root   = Path(__file__).parent
        self.venv_dir       = Path(kwargs.get(
            "venv_dir", self.binding_root / ".venv_vllm_omni"
        ))
        self.target_python  = kwargs.get("python_version", "3.10")

        self._python_exe: Optional[str]           = None
        self._server_proc: Optional[subprocess.Popen] = None

        if self.auto_start:
            self.ensure_server_is_running()

    # ── Health ────────────────────────────────────────────────────────────────

    def is_server_running(self) -> bool:
        """Returns True if the vLLM-Omni server responds to /health."""
        try:
            r = requests.get(f"{self.base_url}{HEALTH_ENDPOINT}", timeout=2)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def ensure_server_is_running(self) -> None:
        """
        Idempotent entry point called before every API operation.

        Fast path: server already up → return immediately (one HTTP call).
        Slow path: acquire file lock → install deps if needed → start server
                   → wait for readiness.
        If the lock is held by another worker, wait for it to finish starting
        the server instead of competing.
        """
        if self.is_server_running():
            return

        lock_path = self.venv_dir.parent / ".vllm_omni_server.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))

        ASCIIColors.info("vLLM-Omni: Acquiring startup lock…")
        try:
            with lock.acquire(timeout=15):
                # Another worker may have started it while we waited for the lock
                if self.is_server_running():
                    ASCIIColors.green("vLLM-Omni: Server started by another worker ✓")
                    return
                ASCIIColors.yellow("vLLM-Omni: Lock acquired — starting server…")
                self._ensure_dependencies()
                self._start_vllm_server()
                self._wait_for_server(timeout=180)
        except Timeout:
            ASCIIColors.yellow(
                "vLLM-Omni: Another process is starting the server — waiting…"
            )
            self._wait_for_server(timeout=180)

        if not self.is_server_running():
            raise RuntimeError(
                "vLLM-Omni: Server did not come up within the timeout.\n"
                "Tip: increase startup_timeout, check VRAM, or run manually:\n"
                f"  {self._python_exe or 'python'} -m vllm.entrypoints.openai.api_server "
                f"--model {self.model_name} --omni --port {self.port}"
            )

    # ── Dependency installation ───────────────────────────────────────────────

    def _ensure_dependencies(self) -> None:
        """
        Creates the dedicated venv and installs vLLM-Omni if not already present.
        Uses pipmaster so Python version management is handled automatically.
        All output is shown to the user with clear progress messages.
        """
        try:
            import pipmaster as pm
        except ImportError:
            ASCIIColors.yellow("vLLM-Omni: Installing pipmaster…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pipmaster", "-q"]
            )
            import pipmaster as pm

        ASCIIColors.info(
            f"vLLM-Omni: Setting up Python {self.target_python} venv "
            f"at {self.venv_dir} …"
        )

        try:
            pm_instance = pm.get_pip_manager_for_version(
                self.target_python, str(self.venv_dir)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"vLLM-Omni: Could not bootstrap Python {self.target_python}: {e}\n"
                "Install it with your package manager:\n"
                "  Ubuntu/Debian : sudo apt install python3.10 python3.10-venv\n"
                "  Fedora/RHEL   : sudo dnf install python3.10\n"
                "  macOS         : brew install python@3.10\n"
            ) from e

        self._python_exe = pm_instance.target_python_executable

        # Fast check: are vllm and vllm_omni importable in the venv?
        probe = subprocess.run(
            [self._python_exe, "-c", "import vllm; import vllm_omni"],
            capture_output=True,
        )
        if probe.returncode == 0:
            ASCIIColors.green("vLLM-Omni: Dependencies already installed ✓")
            return

        ASCIIColors.yellow(
            "vLLM-Omni: Installing vLLM + vLLM-Omni into the dedicated venv.\n"
            "           This only happens once and may take a few minutes…"
        )
        for pkg in VLLM_PACKAGES:
            ASCIIColors.info(f"  → {pkg}")
            ok = pm_instance.install_if_missing(pkg, verbose=True)
            if not ok:
                raise RuntimeError(
                    f"vLLM-Omni: Failed to install '{pkg}'.\n"
                    "Check your internet connection and CUDA installation."
                )

        ASCIIColors.green("vLLM-Omni: All dependencies installed ✓")

    # ── Start vLLM-Omni ──────────────────────────────────────────────────────

    def _start_vllm_server(self) -> None:
        """
        Launches `vllm serve <model> --omni` as a detached background process
        using the dedicated venv interpreter.
        """
        if self._python_exe is None:
            # Venv existed already — derive executable path conventionally
            if sys.platform == "win32":
                self._python_exe = str(self.venv_dir / "Scripts" / "python.exe")
            else:
                self._python_exe = str(self.venv_dir / "bin" / "python")

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

        # Append any model-specific extra args (trust-remote-code, deploy-config, …)
        if self.extra_args:
            cmd += self.extra_args

        env = os.environ.copy()
        if self.hf_token:
            env["HF_TOKEN"]               = self.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

        popen_kwargs: Dict[str, Any] = {"env": env}
        # start_new_session detaches the child from our process group on Linux
        # so the server keeps running even if the parent LoLLMs worker exits
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True

        ASCIIColors.info(f"vLLM-Omni: Launching server…")
        ASCIIColors.info(f"  {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(cmd, **popen_kwargs)
        ASCIIColors.info(f"vLLM-Omni: Server process PID {self._server_proc.pid}")

    def _wait_for_server(self, timeout: int = 180) -> None:
        """Poll /health with progress messages until ready or timeout."""
        ASCIIColors.info(f"vLLM-Omni: Waiting up to {timeout}s (model may be downloading)…")
        deadline   = time.time() + timeout
        last_log   = time.time()
        start_time = time.time()
        while time.time() < deadline:
            if self.is_server_running():
                elapsed = int(time.time() - start_time)
                ASCIIColors.green(f"vLLM-Omni: Server is ready ✓  ({elapsed}s)")
                return
            if time.time() - last_log > 15:
                elapsed = int(time.time() - start_time)
                ASCIIColors.info(f"vLLM-Omni: Still waiting… ({elapsed}s elapsed)")
                last_log = time.time()
            time.sleep(3)
        raise RuntimeError(
            f"vLLM-Omni: Server did not respond within {timeout}s.\n"
            "The model weights may still be downloading from HuggingFace.\n"
            "Re-run LoLLMs after the download completes, or increase the "
            "startup timeout via the 'startup_timeout' config key."
        )

    def __del__(self):
        # The server is a shared resource — individual client instances never stop it.
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
        Synthesise *text* via the local vLLM-Omni server.

        Args:
            text:             Text to synthesise.
            voice:            Voice ID.  Meaning is model-dependent:
                              • Preset name for Voxtral / Qwen3-TTS-CustomVoice
                              • Free-form NL description for Qwen3-TTS-VoiceDesign
                                (e.g. "A warm, deep British male voice, slow paced")
                              • Name of an uploaded custom voice for cloning models
                              Falls back to self.default_voice when empty.
            language:         BCP-47 code: en, fr, de, es, nl, pt, it, hi, ar.
            **kwargs:         Extra parameters forwarded verbatim to vLLM-Omni:
                              response_format  – wav (default), mp3, flac, opus, pcm
                              speed            – 0.25–4.0 (default 1.0)
                              stream           – True for streaming PCM
                              instructions     – style/emotion prompt (Voxtral, Qwen)

        Returns:
            Raw audio bytes in the requested format.
        """
        self.ensure_server_is_running()

        selected_voice = voice or self.default_voice
        payload: Dict[str, Any] = {
            "model":           self.model_name,
            "input":           text,
            "language":        language,
            "response_format": kwargs.pop("response_format", "wav"),
            "speed":           kwargs.pop("speed", 1.0),
        }
        if selected_voice:
            payload["voice"] = selected_voice
        payload.update(kwargs)

        ASCIIColors.info(
            f"vLLM-Omni: Synthesising | model={self.model_name.split('/')[-1]} "
            f"voice={selected_voice or '(default)'} lang={language} chars={len(text)}"
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
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text if e.response is not None else ""
            ASCIIColors.error(f"vLLM-Omni: HTTP {status} — {detail}")
            raise RuntimeError(f"vLLM-Omni server error ({status}): {detail}") from e
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"vLLM-Omni: Network error — {e}")
            raise RuntimeError(
                "vLLM-Omni: Could not reach the server.  Is it still running?"
            ) from e

        audio_bytes = response.content
        ASCIIColors.green(f"vLLM-Omni: Received {len(audio_bytes):,} bytes ✓")
        return audio_bytes

    # ── Voice management ──────────────────────────────────────────────────────

    def list_voices(self, **kwargs) -> List[str]:
        """
        Returns available voices.
        Queries the live server when running (includes uploaded custom voices);
        falls back to the known preset list for the configured model.
        """
        if self.is_server_running():
            try:
                r = requests.get(f"{self.base_url}{VOICES_ENDPOINT}", timeout=5)
                r.raise_for_status()
                voices = r.json().get("voices", [])
                if voices:
                    return voices
            except Exception as e:
                ASCIIColors.warning(f"vLLM-Omni: Could not fetch voices: {e}")
        # Fallback: return known presets for this model, or empty list
        return list(MODEL_PRESET_VOICES.get(self.model_name, []))

    def list_models(self, **kwargs) -> List[str]:
        """Returns the model currently loaded by the server."""
        return [self.model_name]

    def upload_voice(
        self,
        voice_path:  str,
        voice_name:  Optional[str] = None,
        ref_text:    Optional[str] = None,
    ) -> dict:
        """
        Upload a reference audio file for zero-shot voice cloning.

        Supported by: Qwen3-TTS-Base, Fish Speech S2 Pro, CosyVoice3,
                      OmniVoice, VoxCPM2, MOSS-TTS-Nano.
        Not supported by: Voxtral-4B-TTS-2603 (open-weight release omits
                          the voice encoder — use preset voices only).

        Args:
            voice_path:  Local path to WAV/MP3/FLAC file (3–25 s recommended).
            voice_name:  Name to register.  Defaults to the filename stem.
            ref_text:    Exact transcript of the reference audio.
                         Providing this significantly improves clone quality
                         via in-context learning.

        Returns:
            dict: { success: bool, voice_name: str, message: str }
        """
        self.ensure_server_is_running()

        vf = Path(voice_path)
        if not vf.exists():
            return {"success": False, "voice_name": None,
                    "message": f"File not found: {voice_path}"}

        allowed = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".m4a"}
        if vf.suffix.lower() not in allowed:
            return {"success": False, "voice_name": None,
                    "message": f"Unsupported format '{vf.suffix}'. Use one of {allowed}."}

        name = voice_name or vf.stem

        try:
            with open(vf, "rb") as f:
                files = {"audio_sample": (vf.name, f, f"audio/{vf.suffix.lstrip('.')}")}
                data: Dict[str, str] = {
                    "name":    name,
                    "consent": "user_consent",  # required by vLLM-Omni voice API
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

            registered_name = result.get("voice", {}).get("name", name)
            return {
                "success":    result.get("success", True),
                "voice_name": registered_name,
                "message": (
                    f"Voice '{registered_name}' uploaded successfully. "
                    f"Use voice='{registered_name}' in generate_audio()."
                ),
            }
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"vLLM-Omni: Upload failed — {e}")
            return {"success": False, "voice_name": None,
                    "message": f"Upload failed: {e}"}

    def delete_voice(self, voice_name: str) -> dict:
        """
        Delete a previously uploaded custom voice from the server.

        Args:
            voice_name: The registered name of the voice to remove.

        Returns:
            dict: { success: bool, message: str }
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
            ASCIIColors.error(f"vLLM-Omni: Delete voice failed — {e}")
            return {"success": False, "message": str(e)}
