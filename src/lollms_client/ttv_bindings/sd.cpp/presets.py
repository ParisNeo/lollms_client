"""MiniMax-H3 presets for stable-diffusion.cpp. Pure logic, no heavy imports.

Quant names/sizes come from the unsloth/MiniMax-H3-GGUF card (FL2VA pruned, GiB):
Q2_K 6.26, Q3_K 8.16, Q4_K 10.64, Q5_0 12.97, Q6_K 15.45, Q8_0 19.97; text encoder (Qwen3-VL-32B)
Q2_K_M 12.2, Q4_K_M 17. Flags come from sd.cpp docs/minimax_h3.md and the model cards.
max_frames / max_side are UNCALIBRATED placeholders: measure on hardware, override in config.
"""
import re
import subprocess
from dataclasses import dataclass, replace
from typing import List, Optional

GGUF_REPO = "unsloth/MiniMax-H3-GGUF"            # diffusion + text-encoder GGUFs
VAE_REPO = "Comfy-Org/MiniMax-H3"                # video/audio VAEs (safetensors)
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
H3_NAMES = ("minimax_h3", "minimax-h3", "h3")


@dataclass(frozen=True)
class H3Tier:
    name: str
    min_vram_gb: float
    unet_quants: tuple      # in order of preference
    te_quants: tuple
    te_cpu: bool            # --backend te=cpu
    vae_on_cpu: bool        # --vae-on-cpu
    offload: bool           # --offload-to-cpu
    max_frames: int
    max_side: int


TIERS = [
    H3Tier("t42", 40, ("Q8_0",), ("Q4_K_M",), False, False, False, 243, 1280),
    H3Tier("t24", 22, ("Q6_K", "Q5_0"), ("Q4_K_M",), True, False, True, 124, 960),
    H3Tier("t16", 15, ("Q4_K", "Q4_K_M"), ("Q2_K_M",), True, False, True, 56, 864),
    H3Tier("t12", 11, ("Q3_K", "Q2_K"), ("Q2_K_M",), True, True, True, 56, 640),
]
BY_NAME = {t.name: t for t in TIERS}


def detect_vram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return 0.0


def select_tier(vram_gb: float, override: Optional[str] = None, max_frames: int = 0, max_side: int = 0) -> H3Tier:
    if override and override != "auto":
        if override not in BY_NAME:
            raise ValueError(f"unknown tier {override!r}; valid: {list(BY_NAME)}")
        t = BY_NAME[override]
    else:   # below the smallest tier (or no NVIDIA GPU detected) -> smallest tier, caller warns
        t = next((x for x in TIERS if vram_gb >= x.min_vram_gb), TIERS[-1])
    kw = {k: int(v) for k, v in (("max_frames", max_frames), ("max_side", max_side)) if v}
    return replace(t, **kw) if kw else t


def _quant_re(quants) -> "re.Pattern":
    return re.compile(r"[-_.](" + "|".join(re.escape(q) for q in quants) + r")\.gguf$", re.I)


def pick_diffusion(files: List[str], workflow: str, quants) -> Optional[str]:
    """workflow: 'fl2va' | 'ref2va'. Prefers pruned files, honors quant order."""
    gg = [f for f in files if f.lower().endswith(".gguf") and workflow in f.lower()]
    for q in quants:
        rx = _quant_re([q])
        hits = sorted((f for f in gg if rx.search(f)), key=lambda f: ("pruned" not in f.lower(), len(f)))
        if hits:
            return hits[0]
    return None


def pick_text_encoder(files: List[str], quants) -> Optional[str]:
    gg = [f for f in files if f.lower().endswith(".gguf") and "qwen3vl" in f.lower()]
    for q in quants:
        hits = sorted((f for f in gg if _quant_re([q]).search(f)), key=len)
        if hits:
            return hits[0]
    return None
