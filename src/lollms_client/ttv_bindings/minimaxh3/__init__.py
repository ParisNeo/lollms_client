import subprocess, sys, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from filelock import FileLock, Timeout
from ascii_colors import ASCIIColors
import json, base64

from lollms_client.lollms_ttv_binding import LollmsTTVBinding
from .server import tiers as _tiers

BindingName = "MinimaxH3TTVBinding"


class MinimaxH3TTVBinding(LollmsTTVBinding):
    """Shared-server TTV binding for MiniMax-H3. The first instance starts the server,
    every other instance (any process) just talks to it. VRAM tier is picked server-side."""

    def __init__(self, **kwargs):
        super().__init__(binding_name="minimax_h3", debug=kwargs.get("debug", False), **kwargs)
        c = self.config
        self.host, self.port = c.get("host", "localhost"), int(c.get("port", 9635))
        self.base_url = f"http://{self.host}:{self.port}"
        self.auto_start_server = c.get("auto_start_server", True)
        root = self.get_system_dir()
        self.venv_dir = self.resolve_system_path(c["venv_path"]) if c.get("venv_path") else root / "venv" / "ttv_minimax_h3_venv"
        self.models_path = self.resolve_system_path(c["models_path"]) if c.get("models_path") else root / "data" / "ttv_models" / "minimax_h3"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.server_dir = Path(__file__).parent / "server"
        self.comfy_root = root / "minimax_h3_comfy"
        self.tier = _tiers.select_tier(_tiers.detect_vram_gb(), c.get("tier"), max_frames=c.get("max_frames"), max_side=c.get("max_side"))
        self.server_process = None
        self._server_cfg = {k: c[k] for k in ("tier", "max_frames", "max_side", "comfy_url", "comfy_port", "workflow_path", "timeout_s", "torch_index_url", "gguf_node_repo", "unet_repo", "te_repo", "vae_repo", "comfy_args") if c.get(k)}
        if self.auto_start_server:
            self.ensure_server_is_running(c.get("wait_for_server", False))

    # ── resource contract ────────────────────────────────────────────────
    def is_local(self) -> bool: return True

    def _status(self) -> Optional[dict]:
        try:
            r = requests.get(f"{self.base_url}/status", timeout=1)
            return r.json() if r.ok else None
        except Exception:
            return None

    def is_server_running(self) -> bool: return self._status() is not None

    def is_model_loaded(self, model_name=None) -> bool:
        s = self._status(); return bool(s and s.get("model_loaded"))

    def get_loaded_models(self) -> List[str]:
        s = self._status(); return [s["artifact"]] if s and s.get("model_loaded") else []

    def unload_model(self, model_name=None) -> bool:
        try: return requests.post(f"{self.base_url}/unload", timeout=30).ok
        except Exception as e: self._last_error = str(e); return False

    def ps(self) -> List[Dict[str, Any]]:
        s = self._status() or {}
        return [{"model_name": s.get("artifact", "n/a"), "is_loaded": bool(s.get("model_loaded")), "device": f"cuda ({s.get('tier')}, {s.get('backend')})",
                 "vram_size": s.get("vram_gb", 0), "gpu_usage_percent": 0.0, "cpu_usage_percent": 0.0, "ref_count": 1,
                 "status": "busy" if s.get("busy") else "active"}]

    # ── server lifecycle ─────────────────────────────────────────────────
    def install_server_dependencies(self):
        import pipmaster as pm
        v = pm.PackageManager(venv_path=str(self.venv_dir))
        v.ensure_packages(["requests", "uvicorn", "fastapi", "filelock", "ascii_colors", "pydantic", "huggingface_hub"])
        if self.tier.backend == "diffusers":   # heavy stack only where it is actually used
            v.ensure_packages(["torch", "torchvision"])
            v.ensure_packages(["transformers", "accelerate", "safetensors", "sentencepiece", "imageio[ffmpeg]"])
            v.ensure_packages(["diffusers @ git+https://github.com/huggingface/diffusers.git"])  # H3 modular pipeline lives on main

    def ensure_server_is_running(self, wait=False):
        if self.is_server_running():
            return
        t = threading.Thread(target=self._start_server, daemon=True); t.start()
        if wait: t.join()

    def _start_server(self):
        lock = FileLock(self.server_dir / "minimax_h3_server.lock")
        try:
            with lock.acquire(timeout=0):
                if self.is_server_running(): return
                if not (self.venv_dir / "pyvenv.cfg").exists(): self.install_server_dependencies()
                py = self.venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
                cmd = [str(py), str(self.server_dir / "main.py"), "--host", self.host, "--port", str(self.port),
                       "--models-path", str(self.models_path), "--comfy-root", str(self.comfy_root), "--config", json.dumps(self._server_cfg)]
                if self.config.get("hf_token"): cmd += ["--hf-token", self.config["hf_token"]]
                self.server_process = subprocess.Popen(cmd, cwd=str(self.server_dir),
                                                       creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0)
                for _ in range(120):
                    if self.is_server_running(): ASCIIColors.green("MiniMax-H3 server up."); return
                    time.sleep(0.5)
        except Timeout:
            # another instance/process is starting it: just wait for it
            for _ in range(240):
                if self.is_server_running(): return
                time.sleep(0.5)

    def get_server_logs(self) -> str: return ""

    def _wait_ready(self, timeout: float):
        """First call blocks while the server installs/downloads; logs phase changes."""
        end, last = time.time() + timeout, None
        while time.time() < end:
            s = self._status()
            if s is None:
                self.ensure_server_is_running(False)
            elif s.get("phase") == "ready":
                return
            elif s.get("phase") == "failed":
                self._last_error = s.get("error"); raise RuntimeError(f"minimax_h3 setup failed: {s.get('error')}")
            elif s and (s.get("message") != last):
                last = s.get("message"); ASCIIColors.info(f"[minimax_h3] {int(s.get('progress', 0) * 100)}% {last}")
            time.sleep(3)
        raise TimeoutError("minimax_h3 server not ready in time")

    # ── generation ───────────────────────────────────────────────────────
    def generate_video(self, prompt: str, **kwargs) -> bytes:
        self._wait_ready(float(self.config.get("install_timeout_s", 7200)))
        neg = kwargs.pop("negative_prompt", "")
        r = requests.post(f"{self.base_url}/generate_video", json={"prompt": prompt, "negative_prompt": neg, "params": kwargs}, timeout=3600)
        if not r.ok:
            self._last_error = r.text; raise RuntimeError(f"minimax_h3 generation failed: {r.text}")
        if r.headers.get("X-Clamped"): ASCIIColors.warning(f"[minimax_h3] {r.headers['X-Clamped']} (tier {r.headers.get('X-Tier')})")
        return r.content

    @staticmethod
    def _media(x, default_ext: str):
        if isinstance(x, (bytes, bytearray)):
            return bytes(x), default_ext
        pth = Path(x).expanduser()
        return pth.read_bytes(), (pth.suffix.lower() or default_ext)

    def supports_lipsync(self) -> bool:
        return self.tier.backend == "comfyui"   # diffusers tier: ref2va wiring unverified

    def generate_lipsync_video(self, image, audio, prompt: str = "", **kwargs) -> bytes:
        if not self.supports_lipsync():
            raise NotImplementedError(f"minimax_h3 lip-sync is not available on tier {self.tier.name} ({self.tier.backend})")
        self._wait_ready(float(self.config.get("install_timeout_s", 7200)))
        img, iext = self._media(image, kwargs.pop("image_ext", ".png"))
        aud, aext = self._media(audio, kwargs.pop("audio_ext", ".wav"))
        neg = kwargs.pop("negative_prompt", "")
        r = requests.post(f"{self.base_url}/lipsync", timeout=7200, json={
            "prompt": prompt, "negative_prompt": neg, "image_b64": base64.b64encode(img).decode(),
            "audio_b64": base64.b64encode(aud).decode(), "image_ext": iext, "audio_ext": aext, "params": kwargs})
        if not r.ok:
            self._last_error = r.text; raise RuntimeError(f"minimax_h3 lip-sync failed: {r.text}")
        if r.headers.get("X-Clamped"): ASCIIColors.warning(f"[minimax_h3] {r.headers['X-Clamped']} (tier {r.headers.get('X-Tier')})")
        return r.content

    def list_models(self, **kwargs) -> List[str]:
        s = self._status()
        return [s["artifact"]] if s else ["MiniMaxAI/MiniMax-H3"]

    def get_zoo(self) -> List[Dict[str, Any]]:
        return [
            {"name": "H3 FL2VA pruned Q3_K_M (12G tier)", "description": "Abiray GGUF, ComfyUI only.", "size": "8.9GB", "type": "gguf", "link": "Abiray/MiniMax-H3-Pruned-GGUF"},
            {"name": "H3 FL2VA pruned Q4_K_M (16G tier)", "description": "Abiray GGUF, ComfyUI only.", "size": "11.6GB", "type": "gguf", "link": "Abiray/MiniMax-H3-Pruned-GGUF"},
            {"name": "H3 FP8 torchao (42G tier)", "description": "Diffusers modular pipeline.", "size": "~32GB", "type": "safetensors", "link": "abhishekchohan/minimax-h3-fp8"},
        ]
