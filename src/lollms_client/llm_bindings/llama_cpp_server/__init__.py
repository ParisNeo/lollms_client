import subprocess
import sys
import os
import time
import threading
import base64
import shutil
import requests
import socket
import re
import platform
import zipfile
import tarfile
import json
import yaml
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion

# Ensure dependencies
pm.ensure_packages(["openai", "huggingface_hub", "filelock", "requests", "tqdm", "psutil", "pyyaml"])
import openai
from huggingface_hub import hf_hub_download
from filelock import FileLock
from tqdm import tqdm
import psutil

BindingName = "LlamaCppServerBinding"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_free_port(start_port: int = 9624, max_port: int = 10000) -> int:
    """
    Finds a free port on localhost.
    Race-condition safe-ish: We bind to it momentarily to verify availability.
    Real safety comes from the FileLock around the caller.
    """
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available in range.")


def _encode_image(image_path_or_url: str) -> Optional[str]:
    """
    Encodes an image as a base64 data-URI for use in OpenAI-style vision payloads.
    Accepts local file paths OR http(s) URLs.
    Returns None on failure.
    """
    try:
        if image_path_or_url.startswith(("http://", "https://")):
            resp = requests.get(image_path_or_url, timeout=15)
            resp.raise_for_status()
            data = resp.content
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        else:
            p = Path(image_path_or_url)
            if not p.exists():
                ASCIIColors.warning(f"Image not found: {image_path_or_url}")
                return None
            data = p.read_bytes()
            ext = p.suffix.lower().lstrip(".")
            content_type = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")
        b64 = base64.b64encode(data).decode()
        return f"data:{content_type};base64,{b64}"
    except Exception as e:
        ASCIIColors.warning(f"Failed to encode image '{image_path_or_url}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Binding
# ─────────────────────────────────────────────────────────────────────────────

class LlamaCppServerBinding(LollmsLLMBinding):
    """
    A fully-featured, process-safe lollms binding for llama.cpp server.

    Enhancements over the original:
      • Full support for all LollmsLLMBinding parameters (images, system_prompt,
        think / reasoning_effort / reasoning_summary flags, streaming callbacks …)
      • generate_from_messages() implemented.
      • Image encoding + multimodal chat payloads.
      • Idle-timeout enforced via a background watchdog thread.
      • Live context-size query from the running server (falls back to hardcoded list).
      • Improved tokenize / detokenize with path-normalisation and retry.
      • Richer ps() output sourced from live registry.
      • list_models() marks which models are currently running.
      • Atomic, crash-safe registry writes.
      • Comprehensive docstrings throughout.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        self.config = kwargs

        # ── Network / server config ───────────────────────────────────────────
        self.host = kwargs.get("host", "localhost")

        # ── Model config ──────────────────────────────────────────────────────
        self.model_name: Optional[str] = kwargs.get("model_name", "") or None
        self.n_ctx: int = int(kwargs.get("ctx_size", 4096))
        self.n_gpu_layers: int = int(kwargs.get("n_gpu_layers", -1))
        self.n_threads: Optional[int] = kwargs.get("n_threads", None)
        self.n_parallel: int = int(kwargs.get("n_parallel", 1))
        self.batch_size: int = int(kwargs.get("batch_size", 512))
        self.flash_attn: bool = bool(kwargs.get("flash_attn", False))
        self.mmap: bool = bool(kwargs.get("mmap", True))
        self.mlock: bool = bool(kwargs.get("mlock", False))
        self.rope_scale: Optional[float] = kwargs.get("rope_scale", None)
        self.rope_freq_base: Optional[float] = kwargs.get("rope_freq_base", None)
        self.rope_freq_scale: Optional[float] = kwargs.get("rope_freq_scale", None)
        self.tensor_split: Optional[str] = kwargs.get("tensor_split", None)  # "0.5,0.5" …
        self.main_gpu: Optional[int] = kwargs.get("main_gpu", None)
        self.cache_type_k: Optional[str] = kwargs.get("cache_type_k", None)  # "f16", "q8_0" …
        self.cache_type_v: Optional[str] = kwargs.get("cache_type_v", None)

        # ── Capacity / lifecycle ──────────────────────────────────────────────
        self.max_active_models: int = int(kwargs.get("max_active_models", 1))
        self.idle_timeout: float = float(kwargs.get("idle_timeout", -1))  # seconds; -1 = disabled

        # ── Paths ─────────────────────────────────────────────────────────────
        self.binding_dir = Path(__file__).parent
        # Use binaries_path from config if provided, otherwise default to relative 'bin'
        binaries_path = kwargs.get("binaries_path")
        if binaries_path:
            self.bin_dir = Path(binaries_path).resolve()
        else:
            self.bin_dir = self.binding_dir / "bin"
        
        self.models_dir = Path(
            kwargs.get("models_path", "data/models/llama_cpp_models")
        ).resolve()
        self.mm_registry_path = self.models_dir / "multimodal_bindings.yaml"

        # Registry directory – one JSON per running model
        self.servers_dir = self.models_dir / "servers"
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(exist_ok=True)

        # Global file-lock (prevents port races across processes)
        self.global_lock_path = self.models_dir / "global_server_manager.lock"

        # ── Background watchdog (idle timeout) ───────────────────────────────
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        if self.idle_timeout > 0:
            self._start_watchdog()

        # ── Auto-install llama.cpp if missing ────────────────────────────────
        if not self._get_server_executable().exists():
            ASCIIColors.warning("llama.cpp binary not found. Attempting auto-install …")
            self.install_llama_cpp()

        # ── Auto-load model if provided at construction time ──────────────────
        if self.model_name:
            self.load_model(self.model_name)

        atexit.register(self.cleanup_orphans_if_needed)

    # ──────────────────────────────────────────────────────────────────────────
    # Binary / install helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_server_executable(self) -> Path:
        if platform.system() == "Windows":
            return self.bin_dir / "llama-server.exe"
        return self.bin_dir / "llama-server"

    def detect_hardware(self) -> str:
        """Returns 'cuda', 'macos', or 'cpu'."""
        if platform.system() == "Darwin":
            return "macos"
        try:
            subprocess.check_call(
                ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return "cuda"
        except Exception:
            pass
        return "cpu"

    def install_llama_cpp(self, force: bool = False) -> dict:
        """Downloads and extracts the latest llama.cpp release binary.

        Args:
            force: If True, forces re-download even if binary exists.

        Returns:
            dict: {"status": bool, "message": str}
        """
        try:
            # Ensure the target binary directory exists
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            ASCIIColors.info(f"llama.cpp binary directory set to: {self.bin_dir}")

            # Check if binary exists and force is False
            exe = self._get_server_executable()
            if not force and exe.exists():
                # Check if we want to update anyway or just return success
                # For an "update" command, we usually want to force
                pass

            ASCIIColors.info("Fetching latest llama.cpp release …")
            resp = requests.get(
                "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest",
                timeout=30,
            )
            resp.raise_for_status()
            assets = resp.json().get("assets", [])

            hardware = self.detect_hardware()
            sys_plat = platform.system()
            target_asset = None
            search_terms: List[str] = []

            if sys_plat == "Windows":
                search_terms = ["win", "cuda" if hardware == "cuda" else "avx2", "x64"]
            elif sys_plat == "Linux":
                search_terms = ["ubuntu", "x64"]
            elif sys_plat == "Darwin":
                search_terms = [
                    "macos",
                    "arm64" if platform.machine() == "arm64" else "x64",
                ]

            for asset in assets:
                name = asset["name"].lower()
                if "cudart" in name:
                    continue
                if all(t in name for t in search_terms):
                    if "cuda" in name and "cu11" in name and hardware == "cuda":
                        continue
                    target_asset = asset
                    break

            # Windows CPU fallback
            if not target_asset and sys_plat == "Windows" and hardware == "cpu":
                for asset in assets:
                    if "cudart" in asset["name"].lower():
                        continue
                    n = asset["name"].lower()
                    if "win" in n and "x64" in n and "cuda" not in n:
                        target_asset = asset
                        break

            if not target_asset:
                raise RuntimeError(
                    f"No suitable llama.cpp binary found for {sys_plat} / {hardware}"
                )

            filename = target_asset["name"]
            dest_file = self.bin_dir / filename
            ASCIIColors.info(f"Downloading {filename} …")
            with requests.get(
                target_asset["browser_download_url"], stream=True, timeout=120
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest_file, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=filename
                ) as bar:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        bar.update(len(chunk))

            ASCIIColors.info("Extracting …")
            if filename.endswith(".zip"):
                with zipfile.ZipFile(dest_file, "r") as z:
                    z.extractall(self.bin_dir)
            elif filename.endswith(".tar.gz"):
                with tarfile.open(dest_file, "r:gz") as t:
                    t.extractall(self.bin_dir)
                    # List top-level files/dirs to find binaries
                    # For simplicity, we'll scan the bin_dir after extraction
            dest_file.unlink(missing_ok=True)

            # Normalize executable name
            exe = self._get_server_executable()
            legacy = self.bin_dir / (
                "server.exe" if platform.system() == "Windows" else "server"
            )
            
            found_binary = None
            
            # Strategy 1: Check for standard names first
            if exe.exists():
                found_binary = exe
            elif legacy.exists():
                found_binary = legacy
                if not exe.exists():
                    shutil.move(str(legacy), str(exe))
            
            # Strategy 2: If still not found, scan for any executable starting with 'llama' or ending in 'server'
            if not found_binary or not exe.exists():
                candidates = []
                for f in self.bin_dir.iterdir():
                    if f.is_file():
                        name_lower = f.name.lower()
                        # Common patterns: llama-server, llama-cli, server, server.exe, llama-b8287...
                        if "llama" in name_lower and ("server" in name_lower or name_lower.startswith("server")):
                            candidates.append(f)
                        # Fallback: any file that looks like an executable on Linux/Mac
                        elif platform.system() != "Windows" and (name_lower.endswith("-rocm") or name_lower.endswith("-cuda") or name_lower.startswith("llama-")):
                             # Check if it's executable
                             try:
                                 import stat
                                 if f.stat().st_mode & stat.S_IEXEC:
                                     candidates.append(f)
                             except Exception:
                                 pass

                if candidates:
                    # Prefer 'llama-server' if exact match, otherwise take the first candidate
                    preferred = next((c for c in candidates if "llama-server" in c.name.lower()), candidates[0])
                    if preferred != exe:
                        shutil.move(str(preferred), str(exe))
                    found_binary = exe
                else:
                    # Last ditch: look for 'server' in any subdir if extraction created folders
                    for subdir in self.bin_dir.iterdir():
                        if subdir.is_dir():
                            for f in subdir.iterdir():
                                if f.is_file() and (f.name == "server" or f.name == "server.exe"):
                                    shutil.move(str(f), str(exe))
                                    found_binary = exe
                                    break
                        if found_binary: break

            if not exe.exists():
                raise RuntimeError(f"Could not locate 'llama-server' binary after extraction in {self.bin_dir}.")

            if platform.system() != "Windows" and exe.exists():
                os.chmod(exe, 0o755)

            ASCIIColors.success("llama.cpp installed successfully.")
            return {"status": True, "message": "llama.cpp updated successfully."}
        except Exception as e:
            trace_exception(e)
            error_msg = f"Failed to install llama.cpp: {e}"
            ASCIIColors.error(error_msg)
            return {"status": False, "message": error_msg}

    def update_llama_cpp(self) -> dict:
        """Updates or reinstalls the llama.cpp binaries.

        This is the command entry point for the 'update' command.
        It forces a re-download of the latest binaries.

        Returns:
        dict: {"status": bool, "message": str}
        """
        return self.install_llama_cpp(force=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Registry helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_registry_file(self, model_name: str) -> Path:
        safe = "".join(c for c in model_name if c.isalnum() or c in ("-", "_", "."))
        return self.servers_dir / f"{safe}.json"

    def _write_registry_atomic(self, model_name: str, data: dict):
        """Writes registry JSON atomically (temp file + rename)."""
        reg_file = self._get_registry_file(model_name)
        tmp = reg_file.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            tmp.replace(reg_file)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _get_server_info(self, model_name: str) -> Optional[Dict]:
        """
        Reads the registry file for *model_name*.
        Returns the info dict if the server process is alive, else None (and cleans up).
        """
        reg_file = self._get_registry_file(model_name)
        if not reg_file.exists():
            return None
        try:
            with open(reg_file, "r") as f:
                info = json.load(f)
            if psutil.pid_exists(info["pid"]):
                try:
                    p = psutil.Process(info["pid"])
                    pname = p.name().lower()
                    if "llama" in pname or "server" in pname:
                        return info
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            ASCIIColors.warning(
                f"Stale registry for '{model_name}' (PID {info.get('pid')}). Removing."
            )
            reg_file.unlink(missing_ok=True)
        except Exception:
            reg_file.unlink(missing_ok=True)
        return None

    def _kill_server(self, model_name: str, info: Dict):
        """Terminates the server process and removes its registry entry."""
        pid = info.get("pid")
        ASCIIColors.info(f"Stopping server '{model_name}' (PID {pid}) …")
        try:
            p = psutil.Process(pid)
            p.terminate()
            p.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
        except Exception as e:
            ASCIIColors.error(f"Error killing PID {pid}: {e}")
        self._get_registry_file(model_name).unlink(missing_ok=True)

    def _ensure_capacity_locked(self):
        """
        Called while holding the global lock.
        Evicts the Least-Recently-Used model if the active-model cap is reached.
        """
        registry_files = list(self.servers_dir.glob("*.json"))
        valid: List[tuple] = []
        for rf in registry_files:
            try:
                with open(rf, "r") as f:
                    data = json.load(f)
                if psutil.pid_exists(data["pid"]):
                    valid.append((rf, data))
                else:
                    rf.unlink(missing_ok=True)
            except Exception:
                rf.unlink(missing_ok=True)

        if len(valid) >= self.max_active_models:
            valid.sort(key=lambda x: x[0].stat().st_mtime)
            oldest_file, oldest_info = valid[0]
            lru_name = oldest_info.get("model_name", "unknown")
            ASCIIColors.warning(
                f"Capacity ({self.max_active_models}) reached. Evicting LRU: '{lru_name}'"
            )
            self._kill_server(lru_name, oldest_info)

    # ──────────────────────────────────────────────────────────────────────────
    # Multimodal registry
    # ──────────────────────────────────────────────────────────────────────────

    def _load_mm_registry(self) -> Dict[str, str]:
        if not self.mm_registry_path.exists():
            return {}
        try:
            with open(self.mm_registry_path, "r") as f:
                registry: Dict[str, str] = yaml.safe_load(f) or {}
            # Self-heal: remove entries whose files no longer exist
            stale = [
                m
                for m, p in registry.items()
                if not (self.models_dir / m).exists()
                or not (self.models_dir / p).exists()
            ]
            if stale:
                for m in stale:
                    del registry[m]
                self._save_mm_registry(registry)
            return registry
        except Exception as e:
            ASCIIColors.error(f"Failed to load multimodal registry: {e}")
            return {}

    def _save_mm_registry(self, registry: Dict[str, str]):
        try:
            tmp = self.mm_registry_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                yaml.dump(registry, f)
            tmp.replace(self.mm_registry_path)
        except Exception as e:
            ASCIIColors.error(f"Failed to save multimodal registry: {e}")

    def bind_multimodal_model(self, model_name: str, mmproj_name: str) -> dict:
        """Explicitly associates a GGUF model with a vision projector (.mmproj).
        
        If the model is currently running, the server will be automatically 
        restarted to apply the new projector configuration.
        """
        # Validate files
        model_path = self.models_dir / model_name
        mmproj_path = self.models_dir / mmproj_name
        
        if not model_path.exists():
            return {"status": False, "error": f"Model '{model_name}' not found."}
        if not mmproj_path.exists():
            return {"status": False, "error": f"Projector '{mmproj_name}' not found."}
            
        registry = self._load_mm_registry()
        registry[model_name] = mmproj_name
        self._save_mm_registry(registry)
        ASCIIColors.success(f"Bound '{model_name}' ↔ '{mmproj_name}'")

        # Check if the model is currently running and restart it
        is_running = self._get_server_info(model_name) is not None
        was_active = (self.model_name == model_name)
        
        if is_running:
            ASCIIColors.info(f"Server for '{model_name}' is running. Restarting to apply projector...")
            
            # Kill the current server
            info = self._get_server_info(model_name)
            if info:
                self._kill_server(model_name, info)
                # Small delay to ensure port is released
                import time
                time.sleep(1)
            
            # Reload the model (this will spawn a new server with the --mmproj flag)
            try:
                self.load_model(model_name)
                msg = f"Server for '{model_name}' has been restarted successfully with projector '{mmproj_name}'."
                ASCIIColors.success(msg)
                return {"status": True, "message": msg}
            except Exception as e:
                ASCIIColors.error(f"Failed to restart server for '{model_name}': {e}")
                return {"status": False, "error": f"Binding successful, but restart failed: {e}"}
        
        return {"status": True, "message": f"Bound '{model_name}' with '{mmproj_name}'. Restart the server manually or load the model to apply changes."}

    def _find_mmproj(self, model_path: Path) -> Optional[Path]:
        """
        Locates the vision projector for *model_path* via (in order):
        1. Explicit registry entry.
        2. Naming-convention patterns next to the model.
        3. Any file containing 'mmproj' in the same directory.
        """
        registry = self._load_mm_registry()
        if model_path.name in registry:
            proj = self.models_dir / registry[model_path.name]
            if proj.exists():
                return proj

        stem = model_path.stem
        clean = re.sub(r"\.(Q\d_.*|f16|f32)$", "", stem, flags=re.IGNORECASE)
        candidates = [
            f"{stem}.mmproj",
            f"{stem}-mmproj.gguf",
            f"{stem}.mmproj.gguf",
            f"{clean}.mmproj",
            f"{clean}-mmproj.gguf",
            f"mmproj-{stem}.gguf",
            "mmproj.gguf",
        ]
        for c in candidates:
            pot = model_path.parent / c
            if pot.exists():
                return pot

        try:
            for f in model_path.parent.iterdir():
                if (
                    f.is_file()
                    and "mmproj" in f.name.lower()
                    and f != model_path
                    and f.suffix in {".gguf", ".mmproj", ".bin"}
                ):
                    return f
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Server lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _build_server_cmd(self, model_path: Path, port: int) -> List[str]:
        """Constructs the full llama-server command line."""
        exe = self._get_server_executable()
        cmd: List[str] = [
            str(exe),
            "--model", str(model_path),
            "--host", self.host,
            "--port", str(port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--parallel", str(self.n_parallel),
            "--batch-size", str(self.batch_size),
            "--embedding",          # always enable the /embedding endpoint
        ]
        if self.n_threads:
            cmd += ["--threads", str(self.n_threads)]
        if self.flash_attn:
            cmd.append("--flash-attn")
        if not self.mmap:
            cmd.append("--no-mmap")
        if self.mlock:
            cmd.append("--mlock")
        if self.rope_scale is not None:
            cmd += ["--rope-scale", str(self.rope_scale)]
        if self.rope_freq_base is not None:
            cmd += ["--rope-freq-base", str(self.rope_freq_base)]
        if self.rope_freq_scale is not None:
            cmd += ["--rope-freq-scale", str(self.rope_freq_scale)]
        if self.tensor_split is not None:
            cmd += ["--tensor-split", self.tensor_split]
        if self.main_gpu is not None:
            cmd += ["--main-gpu", str(self.main_gpu)]
        if self.cache_type_k is not None:
            cmd += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v is not None:
            cmd += ["--cache-type-v", self.cache_type_v]

        mmproj = self._find_mmproj(model_path)
        if mmproj:
            ASCIIColors.info(f"Vision projector detected: {mmproj.name}")
            cmd += ["--mmproj", str(mmproj)]

        return cmd

    def _spawn_server_detached(self, model_name: str) -> tuple:
        """
        Spawns a detached llama-server process for *model_name*.
        Returns (pid, port, base_url).
        """
        exe = self._get_server_executable()
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        port = get_free_port()
        cmd = self._build_server_cmd(model_path, port)
        base_url = f"http://{self.host}:{port}/v1"

        ASCIIColors.info(f"Spawning '{model_name}' on port {port} …")

        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)

        # Wait up to 120 s for the health endpoint to respond
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{self.host}:{port}/health", timeout=1)
                if r.status_code in (200, 503):   # 503 = loading, still alive
                    if r.status_code == 200:
                        return proc.pid, port, base_url
            except Exception:
                pass
            # Also accept /v1/models becoming available (older builds)
            try:
                r2 = requests.get(f"{base_url}/models", timeout=1)
                if r2.status_code == 200:
                    return proc.pid, port, base_url
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process for '{model_name}' exited early "
                    f"(code {proc.returncode})."
                )
            time.sleep(0.5)

        proc.terminate()
        raise TimeoutError(f"Server for '{model_name}' failed to become ready within 120 s.")

    def load_model(self, model_name: str) -> bool:
        """
        Thread- and process-safe model loader.
        If the model is already running (in any process), updates the LRU timestamp
        and returns immediately.  Otherwise evicts the LRU model if needed, then
        spawns a new server.
        """
        self.global_lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.global_lock_path))
        try:
            with lock.acquire(timeout=60):
                info = self._get_server_info(model_name)
                if info:
                    # Already running – just refresh LRU timestamp
                    try:
                        self._get_registry_file(model_name).touch()
                    except Exception:
                        pass
                    self.model_name = model_name
                    return True

                self._ensure_capacity_locked()
                pid, port, url = self._spawn_server_detached(model_name)
                self._write_registry_atomic(
                    model_name,
                    {
                        "model_name": model_name,
                        "pid": pid,
                        "port": port,
                        "url": url,
                        "started_at": time.time(),
                        "last_used": time.time(),
                    },
                )
                self.model_name = model_name
                ASCIIColors.success(f"Model '{model_name}' loaded on port {port}.")
                return True
        except Exception as e:
            ASCIIColors.error(f"Failed to load model '{model_name}': {e}")
            trace_exception(e)
            return False

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        """Stops the server for *model_name* (defaults to the current model)."""
        target = model_name or self.model_name
        if not target:
            return False
        info = self._get_server_info(target)
        if info:
            self._kill_server(target, info)
            if target == self.model_name:
                self.model_name = None
            return True
        ASCIIColors.warning(f"No running server found for '{target}'.")
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Idle-timeout watchdog
    # ──────────────────────────────────────────────────────────────────────────

    def _start_watchdog(self):
        """Starts a daemon thread that unloads idle servers."""
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="llama-idle-watchdog"
        )
        self._watchdog_thread.start()
        ASCIIColors.info(
            f"Idle-timeout watchdog started (timeout={self.idle_timeout}s)."
        )

    def _watchdog_loop(self):
        check_interval = max(10.0, self.idle_timeout / 4)
        while not self._watchdog_stop.wait(timeout=check_interval):
            self._evict_idle_servers()

    def _evict_idle_servers(self):
        """Terminates any server whose registry mtime is older than idle_timeout."""
        now = time.time()
        for rf in list(self.servers_dir.glob("*.json")):
            try:
                mtime = rf.stat().st_mtime
                if now - mtime > self.idle_timeout:
                    with open(rf, "r") as f:
                        info = json.load(f)
                    mname = info.get("model_name", rf.stem)
                    ASCIIColors.info(
                        f"Idle timeout: evicting '{mname}' "
                        f"(idle {now - mtime:.0f}s > {self.idle_timeout}s)."
                    )
                    self._kill_server(mname, info)
                    if self.model_name == mname:
                        self.model_name = None
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Client factory
    # ──────────────────────────────────────────────────────────────────────────

    def _get_client(self, model_name: Optional[str] = None) -> openai.OpenAI:
        """
        Returns an OpenAI client pointed at the running llama-server for
        *model_name*.  Auto-loads the model if not already running.
        Also refreshes the LRU timestamp so the idle-watchdog doesn't evict it.
        """
        target = model_name or self.model_name
        if not target:
            raise ValueError("No model specified. Call load_model() first.")

        info = self._get_server_info(target)
        if not info:
            if not self.load_model(target):
                raise RuntimeError(f"Could not load model '{target}'.")
            info = self._get_server_info(target)

        # Refresh LRU timestamp
        try:
            self._get_registry_file(target).touch()
        except Exception:
            pass

        if not info:
            raise RuntimeError(f"Model '{target}' failed to start.")
        return openai.OpenAI(base_url=info["url"], api_key="sk-no-key-required")

    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """
        Retries *func* up to 60 times when the server returns 503 (model still
        loading) or a connection error occurs.
        """
        retries = 60
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except openai.InternalServerError as e:
                if getattr(e, "status_code", None) == 503:
                    if i % 10 == 0:
                        ASCIIColors.warning(
                            f"Server loading (503) – retrying … ({i + 1}/{retries})"
                        )
                    time.sleep(2)
                    continue
                raise
            except openai.APIConnectionError:
                if i % 10 == 0:
                    ASCIIColors.warning(
                        f"Connection error – retrying … ({i + 1}/{retries})"
                    )
                time.sleep(2)
                continue
        return func(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Core generation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve(self, value, default):
        """Returns *value* if not None, else *default*."""
        return value if value is not None else default

    def _build_extra_body(self, **kw) -> dict:
        """Assembles llama.cpp-specific extra_body params."""
        body: Dict[str, Any] = {}
        if kw.get("top_k") is not None:
            body["top_k"] = kw["top_k"]
        if kw.get("repeat_penalty") is not None:
            body["repeat_penalty"] = kw["repeat_penalty"]
        if kw.get("repeat_last_n") is not None:
            body["repeat_last_n"] = kw["repeat_last_n"]
        if kw.get("n_predict") is not None:
            body["n_predict"] = kw["n_predict"]
        # thinking / reasoning flags (supported by some llama.cpp builds)
        if kw.get("think"):
            body["thinking"] = True
        if kw.get("reasoning_effort") is not None:
            body["reasoning_effort"] = kw["reasoning_effort"]
        if kw.get("reasoning_summary") is not None:
            body["reasoning_summary"] = kw["reasoning_summary"]
        return body

    def _stream_completion(self, completion, callback: Optional[Callable]) -> str:
        """Drains a streaming completion and feeds chunks to *callback*."""
        full = []
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full.append(content)
            if callback:
                if not callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                    break
        return "".join(full)

    def _stream_raw_completion(self, completion, callback: Optional[Callable]) -> str:
        """Drains a streaming raw (completions) response."""
        full = []
        for chunk in completion:
            content = chunk.choices[0].text or ""
            full.append(content)
            if callback:
                if not callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                    break
        return "".join(full)

    # ──────────────────────────────────────────────────────────────────────────
    # generate_text  (raw completion endpoint)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: str = "",
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        streaming_callback: Optional[Callable] = None,
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        """
        Generates text via the /v1/completions endpoint.

        When *images* are supplied OR when *system_prompt* is non-empty the call
        is automatically promoted to the /v1/chat/completions endpoint so that
        the richer message format (including vision payloads) is used.
        """
        # Resolve defaults
        n_predict    = self._resolve(n_predict,    self.default_n_predict or 1024)
        temperature  = self._resolve(temperature,  self.default_temperature or 0.7)
        top_k        = self._resolve(top_k,        self.default_top_k or 40)
        top_p        = self._resolve(top_p,        self.default_top_p or 0.9)
        repeat_penalty = self._resolve(repeat_penalty, self.default_repeat_penalty or 1.1)
        repeat_last_n  = self._resolve(repeat_last_n,  self.default_repeat_last_n or 64)
        seed         = self._resolve(seed,         self.default_seed)
        cb           = streaming_callback or self.default_streaming_callback
        do_stream    = self._resolve(stream, True if cb else (self.default_stream or False))

        try:
            client = self._get_client()

            # If images or a system_prompt are present, delegate to chat endpoint
            if images or system_prompt:
                messages: List[Dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                user_content: Any
                if images:
                    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                    for img in images:
                        encoded = _encode_image(img)
                        if encoded:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": encoded},
                                }
                            )
                    user_content = parts
                else:
                    user_content = prompt
                messages.append({"role": "user", "content": user_content})
                return self._run_chat_messages(
                    messages,
                    n_predict=n_predict, stream=do_stream, temperature=temperature,
                    top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
                    repeat_last_n=repeat_last_n, seed=seed, cb=cb,
                    think=think, reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                )

            extra_body = self._build_extra_body(
                top_k=top_k, repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n, n_predict=n_predict,
                think=think, reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            )
            extra_kw: Dict[str, Any] = {}
            if seed is not None:
                extra_kw["seed"] = seed

            def _do():
                return client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=n_predict,
                    temperature=temperature,
                    top_p=top_p,
                    stream=do_stream,
                    extra_body=extra_body,
                    **extra_kw,
                )

            completion = self._execute_with_retry(_do)

            if do_stream:
                return self._stream_raw_completion(completion, cb)
            return completion.choices[0].text
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # _chat  (discussion-based, chat/completions endpoint)
    # ──────────────────────────────────────────────────────────────────────────

    def _chat(
        self,
        discussion: LollmsDiscussion,
        branch_tip_id: Optional[str] = None,
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        streaming_callback: Optional[Callable] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        """
        Conducts a chat turn using a LollmsDiscussion object.
        Exports the discussion to OpenAI-style messages and calls
        /v1/chat/completions.
        """
        n_predict   = self._resolve(n_predict,   self.default_n_predict or 1024)
        temperature = self._resolve(temperature, self.default_temperature or 0.7)
        top_k       = self._resolve(top_k,       self.default_top_k or 40)
        top_p       = self._resolve(top_p,       self.default_top_p or 0.9)
        repeat_penalty = self._resolve(repeat_penalty, self.default_repeat_penalty or 1.1)
        repeat_last_n  = self._resolve(repeat_last_n,  self.default_repeat_last_n or 64)
        seed        = self._resolve(seed, self.default_seed)
        cb          = streaming_callback or self.default_streaming_callback
        do_stream   = self._resolve(stream, True if cb else (self.default_stream or False))

        try:
            messages = discussion.export("openai_chat")
            return self._run_chat_messages(
                messages,
                n_predict=n_predict, stream=do_stream, temperature=temperature,
                top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n, seed=seed, cb=cb,
                think=think, reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            )
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # generate_from_messages  (raw messages list)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_from_messages(
        self,
        messages: List[Dict],
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        streaming_callback: Optional[Callable] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        """
        Generates a response directly from an OpenAI-style messages list.
        This is the low-level counterpart to *_chat* for callers that already
        hold a formatted message list.
        """
        n_predict   = self._resolve(n_predict,   self.default_n_predict or 16000)
        temperature = self._resolve(temperature, self.default_temperature or 0.7)
        top_k       = self._resolve(top_k,       self.default_top_k or 40)
        top_p       = self._resolve(top_p,       self.default_top_p or 0.9)
        repeat_penalty = self._resolve(repeat_penalty, self.default_repeat_penalty or 1.1)
        repeat_last_n  = self._resolve(repeat_last_n,  self.default_repeat_last_n or 64)
        seed        = self._resolve(seed, self.default_seed)
        cb          = streaming_callback or self.default_streaming_callback
        do_stream   = self._resolve(stream, True if cb else (self.default_stream or False))

        try:
            return self._run_chat_messages(
                messages,
                n_predict=n_predict, stream=do_stream, temperature=temperature,
                top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n, seed=seed, cb=cb,
                think=think, reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            )
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # Shared chat-completions inner method
    # ──────────────────────────────────────────────────────────────────────────

    def _run_chat_messages(
        self,
        messages: List[Dict],
        n_predict: int,
        stream: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        repeat_last_n: int,
        seed: Optional[int],
        cb: Optional[Callable],
        think: bool = False,
        reasoning_effort: str = "low",
        reasoning_summary: str = "auto",
    ) -> Union[str, dict]:
        """
        Internal helper – all public generation methods funnel through here
        when targeting /v1/chat/completions.
        """
        client = self._get_client()
        extra_body = self._build_extra_body(
            top_k=top_k, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            think=think, reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )
        extra_kw: Dict[str, Any] = {}
        if seed is not None:
            extra_kw["seed"] = seed

        def _do():
            return client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=n_predict,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                extra_body=extra_body,
                **extra_kw,
            )

        response = self._execute_with_retry(_do)
        if stream:
            return self._stream_completion(response, cb)
        content = response.choices[0].message.content
        return content or ""

    # ──────────────────────────────────────────────────────────────────────────
    # Tokenize / detokenize / count_tokens
    # ──────────────────────────────────────────────────────────────────────────

    def _server_base_url(self, client: openai.OpenAI) -> str:
        """Returns the raw base URL string, stripping any '/v1' suffix."""
        url = str(client.base_url).rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    def tokenize(self, text: str) -> list:
        """
        Tokenises *text* via the llama-server's /tokenize endpoint.
        Falls back to character-level split on failure.
        """
        try:
            client = self._get_client()
            base = self._server_base_url(client)

            def _do():
                r = requests.post(
                    f"{base}/tokenize", json={"content": text}, timeout=30
                )
                if r.status_code == 503:
                    raise openai.InternalServerError(
                        "Model loading", response=r, body=None
                    )
                r.raise_for_status()
                return r

            r = self._execute_with_retry(_do)
            return r.json().get("tokens", [])
        except Exception as e:
            ASCIIColors.warning(f"tokenize() fell back to char-split: {e}")
            return list(text)

    def detokenize(self, tokens: list) -> str:
        """
        Detokenises *tokens* via /detokenize.
        Falls back to str-join on failure.
        """
        try:
            client = self._get_client()
            base = self._server_base_url(client)

            def _do():
                r = requests.post(
                    f"{base}/detokenize", json={"tokens": tokens}, timeout=30
                )
                if r.status_code == 503:
                    raise openai.InternalServerError(
                        "Model loading", response=r, body=None
                    )
                r.raise_for_status()
                return r

            r = self._execute_with_retry(_do)
            return r.json().get("content", "")
        except Exception as e:
            ASCIIColors.warning(f"detokenize() fell back to str-join: {e}")
            return "".join(map(str, tokens))

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens in *text* (via tokenize endpoint)."""
        toks = self.tokenize(text)
        return len(toks)

    # ──────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ──────────────────────────────────────────────────────────────────────────

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Returns the embedding vector for *text* via /v1/embeddings.
        The server must have been started with --embedding (which we always do).
        """
        client = self._get_client()

        def _do():
            return client.embeddings.create(input=text, model=self.model_name)

        result = self._execute_with_retry(_do)
        return result.data[0].embedding

    # ──────────────────────────────────────────────────────────────────────────
    # Context size
    # ──────────────────────────────────────────────────────────────────────────

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Returns the context size for *model_name* (defaults to current model).
        Query order:
          1. Live server /props endpoint (exact for the running instance).
          2. Hardcoded lookup table in the base class.
          3. Constructor default.
        """
        target = model_name or self.model_name
        if not target:
            return self.default_ctx_size or self.n_ctx

        info = self._get_server_info(target)
        if info:
            try:
                base = info["url"].rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                r = requests.get(f"{base}/props", timeout=5)
                if r.status_code == 200:
                    props = r.json()
                    # Different llama.cpp versions use different key names
                    for key in ("total_slots", "n_ctx", "ctx_size", "context_size"):
                        if key in props:
                            if key == "total_slots":
                                # total_slots = n_ctx / n_parallel
                                return int(props[key]) * self.n_parallel
                            return int(props[key])
            except Exception as e:
                ASCIIColors.warning(f"Could not query /props for ctx size: {e}")

        # Fall back to base-class hardcoded lookup
        return super().get_ctx_size(target)

    # ──────────────────────────────────────────────────────────────────────────
    # Model info / listing
    # ──────────────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Returns metadata about the current binding and active model."""
        info: Dict[str, Any] = {
            "binding_name": BindingName,
            "version": "enhanced",
            "active_model": self.model_name,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_parallel": self.n_parallel,
        }
        reg = self._get_server_info(self.model_name) if self.model_name else None
        if reg:
            info["server_url"] = reg["url"]
            info["server_pid"] = reg["pid"]
            info["server_port"] = reg["port"]
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists all GGUF model files found in *models_dir*, enriched with
        running-status information from the live registry.
        """
        models = []
        if not self.models_dir.exists():
            return models

        # Build a quick lookup of running models
        running: Dict[str, Dict] = {}
        for rf in self.servers_dir.glob("*.json"):
            try:
                with open(rf, "r") as f:
                    data = json.load(f)
                if psutil.pid_exists(data["pid"]):
                    running[data["model_name"]] = data
            except Exception:
                pass

        for f in sorted(self.models_dir.glob("*.gguf")):
            if "mmproj" in f.name.lower():
                continue
            # Skip split-GGUF parts that are not the first shard
            if re.search(r"-\d{5}-of-\d{5}\.gguf$", f.name):
                if "00001-of-" not in f.name:
                    continue

            entry: Dict[str, Any] = {
                "model_name": f.name,
                "owned_by": "local",
                "created": time.ctime(f.stat().st_ctime),
                "size": f.stat().st_size,
                "running": f.name in running,
            }
            if f.name in running:
                entry["server_port"] = running[f.name].get("port")
                entry["server_pid"] = running[f.name].get("pid")

            # Multimodal?
            mm_reg = self._load_mm_registry()
            if f.name in mm_reg:
                entry["mmproj"] = mm_reg[f.name]
            elif self._find_mmproj(f) is not None:
                entry["mmproj"] = self._find_mmproj(f).name  # type: ignore

            models.append(entry)
        return models

    def ps(self) -> List[Dict[str, Any]]:
        """
        Returns a list of *currently running* model servers enriched with
        process-level resource stats (CPU, memory).
        """
        result = []
        for rf in list(self.servers_dir.glob("*.json")):
            try:
                with open(rf, "r") as f:
                    data = json.load(f)
                pid = data.get("pid")
                if not psutil.pid_exists(pid):
                    rf.unlink(missing_ok=True)
                    continue
                proc = psutil.Process(pid)
                mem = proc.memory_info()
                cpu = proc.cpu_percent(interval=0.1)
                result.append(
                    {
                        "model_name": data.get("model_name"),
                        "pid": pid,
                        "port": data.get("port"),
                        "url": data.get("url"),
                        "started_at": data.get("started_at"),
                        "cpu_percent": cpu,
                        "rss_mb": round(mem.rss / 1024 / 1024, 1),
                        "vms_mb": round(mem.vms / 1024 / 1024, 1),
                        "context_size": self.n_ctx,
                    }
                )
            except Exception as e:
                ASCIIColors.warning(f"ps() skipped entry {rf.name}: {e}")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Model zoo
    # ──────────────────────────────────────────────────────────────────────────

    def get_zoo(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Magistral-Small-2509-GGUF",
                "description": "Mistral AI Magistral 2509 – Efficient Thinking Model",
                "size": "14.3 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Magistral-Small-2509-GGUF",
                "filename": "mistralai_Magistral-Small-2509-Q4_K_M.gguf",
            },
            {
                "name": "Ministral-3-3B-Instruct-2512-GGUF",
                "description": "Mistral AI Ministral 3 3B Instruct – Efficient Edge Model",
                "size": "2.2 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
                "filename": "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
            },
            {
                "name": "Devstral-Small-2-24B-Instruct-GGUF",
                "description": "Mistral AI Devstral Small 2 24B Instruct – Coding Specialist",
                "size": "14.8 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
                "filename": "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
            },
            {
                "name": "Llama-4-Scout-17B-Instruct-GGUF",
                "description": "Meta Llama 4 Scout 17B Instruct – 16-Expert MoE",
                "size": "11.2 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/meta-llama_Llama-4-Scout-17B-16E-Instruct-old-GGUF",
                "filename": "meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf",
            },
            {
                "name": "Qwen3-VL-32B-Thinking-GGUF",
                "description": "Qwen 3 VL 32B Thinking – Vision + CoT Reasoning",
                "size": "19.5 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/Qwen_Qwen3-VL-32B-Thinking-GGUF",
                "filename": "Qwen_Qwen3-VL-32B-Thinking-Q4_K_M.gguf",
            },
            {
                "name": "Qwen3-72B-Embiggened-GGUF",
                "description": "Qwen 3 72B Embiggened – Enhanced Reasoning Dense Model",
                "size": "43.1 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/cognitivecomputations_Qwen3-72B-Embiggened-GGUF",
                "filename": "Qwen3-72B-Embiggened-Q4_K_M.gguf",
            },
            {
                "name": "Devstral-2-123B-Instruct-GGUF",
                "description": "Mistral AI Devstral 2 123B Instruct – Heavy-Duty Coding",
                "size": "71.4 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Devstral-2-123B-Instruct-2512-GGUF",
                "filename": "Devstral-2-123B-Instruct-2512-Q4_K_M.gguf",
            },
            {
                "name": "ChatGPT-OSS-120B-GGUF",
                "description": "OpenAI GPT-OSS 120B – Open-Weight Research Model",
                "size": "69.8 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/openai_gpt-oss-120b-GGUF",
                "filename": "gpt-oss-120b-Q4_K_M.gguf",
            },
            {
                "name": "DeepSeek-V3-0324-GGUF",
                "description": "DeepSeek V3 0324 – 671B MoE Giant",
                "size": "365 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF",
                "filename": "DeepSeek-V3-0324-Q4_K_M.gguf",
            },
        ]

    def download_from_zoo(
        self,
        index: int,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        zoo = self.get_zoo()
        if not (0 <= index < len(zoo)):
            return {"status": False, "message": "Index out of bounds."}
        item = zoo[index]
        return self.pull_model(
            item["link"],
            item.get("filename"),
            progress_callback=progress_callback,
        )

    def pull_model(
        self,
        repo_id: str,
        filename: str,
        mmproj_repo_id: Optional[str] = None,
        mmproj_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Downloads a GGUF model (and optional mmproj) from Hugging Face.
        Handles split-GGUF archives transparently.
        """
        try:
            match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", filename)
            files: List[str] = []
            if match:
                base, total = match.group(1), int(match.group(3))
                ASCIIColors.info(f"Split GGUF detected – {total} parts.")
                for i in range(1, total + 1):
                    files.append(f"{base}-{i:05d}-of-{total:05d}.gguf")
            else:
                files.append(filename)

            downloaded_paths: List[Path] = []
            for part in files:
                ASCIIColors.info(f"Downloading '{part}' from '{repo_id}' …")
                if progress_callback:
                    progress_callback(
                        {"status": "downloading", "message": f"Downloading {part}", "completed": 0, "total": 100}
                    )
                p = hf_hub_download(
                    repo_id=repo_id,
                    filename=part,
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                downloaded_paths.append(Path(p))
                ASCIIColors.success(f"Downloaded '{part}'.")

            if mmproj_filename:
                proj_repo = mmproj_repo_id or repo_id
                ASCIIColors.info(f"Downloading mmproj '{mmproj_filename}' …")
                hf_hub_download(
                    repo_id=proj_repo,
                    filename=mmproj_filename,
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                ASCIIColors.success(f"Downloaded mmproj '{mmproj_filename}'.")
                self.bind_multimodal_model(filename, mmproj_filename)

            msg = f"Successfully downloaded '{filename}'"
            if mmproj_filename:
                msg += f" and bound with projector '{mmproj_filename}'"
            if progress_callback:
                progress_callback({"status": "success", "message": msg, "completed": 100, "total": 100})

            return {"status": True, "message": msg, "path": str(downloaded_paths[0])}
        except Exception as e:
            trace_exception(e)
            err = str(e)
            if progress_callback:
                progress_callback({"status": "error", "message": err})
            return {"status": False, "error": err}

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup_orphans_if_needed(self):
        """
        Called at interpreter exit.  Performs a lightweight registry scan and
        removes stale entries left by crashed processes.
        """
        for rf in list(self.servers_dir.glob("*.json")):
            try:
                with open(rf, "r") as f:
                    data = json.load(f)
                if not psutil.pid_exists(data.get("pid", -1)):
                    rf.unlink(missing_ok=True)
            except Exception:
                rf.unlink(missing_ok=True)

    def __del__(self):
        if hasattr(self, "_watchdog_stop"):
            self._watchdog_stop.set()
