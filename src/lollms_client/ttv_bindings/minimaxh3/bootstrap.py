"""First-run installer for the ComfyUI backend. Idempotent: every step checks before acting.

Installs, under <system_dir>/minimax_h3_comfy/:
  ComfyUI (git) + its own venv (torch + requirements), the ComfyUI-GGUF custom node,
  the tier's UNet GGUF, text-encoder GGUF, video VAE, audio VAE (Hugging Face), then starts ComfyUI.

Filenames inside HF repos are RESOLVED BY LISTING the repo, never hardcoded paths, and the choice is
logged. If nothing matches, the error lists the candidates so you can set an override in the config.
"""
import shutil, subprocess, sys, time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

Progress = Callable[[float, str], None]

DEFAULTS = {
    "comfy_repo": "https://github.com/comfyanonymous/ComfyUI.git",
    "gguf_node_repo": "https://github.com/molbal/ComfyUI-GGUF.git",   # fork documenting H3 + CLIPLoader (GGUF); try city96/ComfyUI-GGUF if it misbehaves
    "torch_index_url": "https://download.pytorch.org/whl/cu128",
    "unet_repo": "Abiray/MiniMax-H3-Pruned-GGUF",
    "te_repo": "joeygambino/MiniMax-H3-encoder-GGUF",
    "vae_repo": "Comfy-Org/MiniMax-H3",
    "vae_video_file": "minimax_h3_video_vae_fp16.safetensors",
    "vae_audio_file": "minimax_h3_audio_vae_fp32.safetensors",
    "te_prefer": {"t12": ["IQ3", "Q3", "Q4"], "t16": ["Q4_K_M", "Q4", "Q5"], "t24": ["Q5", "Q4_K_M", "Q8"]},
    "comfy_args": {"t12": ["--lowvram"], "t16": [], "t24": []},   # tune on hardware
}


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _run(cmd: List[str], cwd: Optional[Path] = None):
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def pick_file(files: List[str], prefer: List[str], suffix: str) -> Optional[str]:
    cands = [f for f in files if f.lower().endswith(suffix)]
    for pat in prefer:
        m = [f for f in cands if pat.lower() in f.lower()]
        if m:
            return sorted(m, key=len)[0]
    return None


def ensure_comfy(root: Path, cfg: dict, progress: Progress) -> Path:
    if not shutil.which("git"):
        raise RuntimeError("git is required for the first-time ComfyUI install (not found in PATH)")
    root.mkdir(parents=True, exist_ok=True)
    comfy = root / "ComfyUI"
    venv = root / "venv"
    if not (comfy / "main.py").exists():
        progress(0.01, "cloning ComfyUI")
        _run(["git", "clone", "--depth", "1", cfg["comfy_repo"], comfy])
    if not (root / ".env_ok").exists():
        progress(0.05, "creating ComfyUI venv")
        _run([sys.executable, "-m", "venv", venv])
        py = venv_python(venv)
        progress(0.08, "installing torch (large download)")
        _run([py, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", cfg["torch_index_url"]])
        progress(0.15, "installing ComfyUI requirements")
        _run([py, "-m", "pip", "install", "-r", comfy / "requirements.txt"])
        (root / ".env_ok").write_text("ok")
    node = comfy / "custom_nodes" / "ComfyUI-GGUF"
    if not node.exists():
        progress(0.18, "installing ComfyUI-GGUF node")
        _run(["git", "clone", "--depth", "1", cfg["gguf_node_repo"], node])
        req = node / "requirements.txt"
        if req.exists():
            _run([venv_python(venv), "-m", "pip", "install", "-r", req])
    return comfy


def _fetch(repo: str, dest: Path, token: Optional[str], progress: Progress, frac: float, label: str,
           exact: Optional[str] = None, prefer: Optional[List[str]] = None, suffix: str = "") -> str:
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo, token=token or None)
    if exact:
        hit = next((f for f in files if Path(f).name == exact), None)
    else:
        hit = pick_file(files, prefer or [], suffix)
    if not hit:
        cands = [f for f in files if f.lower().endswith(suffix or (".gguf", ".safetensors"))][:40]
        raise RuntimeError(f"[{label}] nothing matching in {repo}. Candidates: {cands}. Override in config.")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(hit).name
    if not target.exists():
        progress(frac, f"downloading {label}: {hit}")
        got = hf_hub_download(repo, hit, token=token or None, local_dir=str(dest / ".dl"))
        shutil.move(got, target)
    return target.name


def ensure_models(comfy: Path, tier, cfg: dict, token: Optional[str], progress: Progress) -> Dict[str, str]:
    m = comfy / "models"
    return {
        "unet_name": _fetch(cfg["unet_repo"], m / "diffusion_models", token, progress, 0.25, "UNet GGUF", exact=tier.artifact),
        "clip_name": _fetch(cfg["te_repo"], m / "text_encoders", token, progress, 0.45, "text encoder",
                            prefer=cfg["te_prefer"].get(tier.name, ["Q4"]), suffix=".gguf"),
        "vae_video_name": _fetch(cfg["vae_repo"], m / "vae", token, progress, 0.75, "video VAE", exact=cfg["vae_video_file"]),
        "vae_audio_name": _fetch(cfg["vae_repo"], m / "vae", token, progress, 0.85, "audio VAE", exact=cfg["vae_audio_file"]),
    }


def ensure_ref_unet(comfy: Path, tier, cfg: dict, token: Optional[str], progress: Progress) -> str:
    """Ref2VA UNet is only fetched when lip-sync is first used (saves ~9-17 GB otherwise)."""
    return _fetch(cfg["unet_repo"], comfy / "models" / "diffusion_models", token, progress, 0.05,
                  "Ref2VA UNet GGUF", exact=tier.artifact_ref)


def comfy_alive(url: str) -> bool:
    try:
        return requests.get(f"{url}/system_stats", timeout=2).ok
    except Exception:
        return False


def start_comfy(root: Path, comfy: Path, url: str, port: int, extra_args: List[str], progress: Progress):
    if comfy_alive(url):
        return None
    progress(0.95, "starting ComfyUI")
    proc = subprocess.Popen([str(venv_python(root / "venv")), "main.py", "--listen", "127.0.0.1", "--port", str(port), *extra_args],
                            cwd=str(comfy))
    for _ in range(240):
        if comfy_alive(url):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {proc.returncode} during startup (see console)")
        time.sleep(1)
    raise TimeoutError("ComfyUI did not come up within 4 minutes")


def find_official_template(root: Path) -> Optional[Path]:
    """The official H3 template ships UI-format (with subgraphs). Open it in ComfyUI, then Export (API)."""
    for p in (root / "venv").rglob("*.json"):
        n = p.name.lower()
        if "minimax" in n and "h3" in n:
            return p
    return None
