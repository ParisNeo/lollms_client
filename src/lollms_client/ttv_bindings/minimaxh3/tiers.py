"""VRAM tier selection. Pure logic, no heavy imports: unit-testable anywhere.

Artifact names/sizes come from the public model cards (Abiray/MiniMax-H3-Pruned-GGUF,
abhishekchohan/minimax-h3-fp8). The caps (max_frames / max_side) are UNCALIBRATED
defaults: measure them on the 12G card and edit here. Everything is overridable
from the binding config (tier=..., max_frames=..., max_side=...).
"""
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Tier:
    name: str
    min_vram_gb: float      # total VRAM (GiB) needed to qualify
    backend: str            # "comfyui" (GGUF) | "diffusers" (ModularPipeline)
    artifact: str           # FL2VA GGUF filename (comfyui) or HF repo id (diffusers)
    artifact_ref: str       # Ref2VA GGUF filename (comfyui; used by lip-sync) or same HF repo
    max_frames: int
    max_side: int           # max of width/height
    offload: str            # "text_encoder_cpu" | "group" | "none"


# Ordered from biggest to smallest.
TIERS = [
    Tier("t42", 40, "diffusers", "abhishekchohan/minimax-h3-fp8", "abhishekchohan/minimax-h3-fp8", 121, 1024, "group"),
    Tier("t24", 22, "comfyui", "MiniMax-H3-FL2VA-Pruned-Q6_K.gguf", "MiniMax-H3-Ref2VA-Pruned-Q6_K.gguf", 97, 768, "text_encoder_cpu"),
    Tier("t16", 15, "comfyui", "MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf", "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf", 49, 768, "text_encoder_cpu"),
    Tier("t12", 11, "comfyui", "MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf", "MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf", 25, 512, "text_encoder_cpu"),
]
BY_NAME: Dict[str, Tier] = {t.name: t for t in TIERS}


def detect_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return 0.0


def select_tier(vram_gb: float, override: Optional[str] = None, **caps: Any) -> Tier:
    """override: tier name ('t12'...) or None/'auto'. caps: max_frames/max_side overrides."""
    if override and override != "auto":
        if override not in BY_NAME:
            raise ValueError(f"unknown tier {override!r}; valid: {list(BY_NAME)}")
        tier = BY_NAME[override]
    else:
        tier = next((t for t in TIERS if vram_gb >= t.min_vram_gb), None)
        if tier is None:
            raise RuntimeError(f"{vram_gb:.1f} GiB VRAM is below the smallest tier "
                               f"({TIERS[-1].min_vram_gb} GiB). Refusing to guess.")
    clean = {k: int(v) for k, v in caps.items() if k in ("max_frames", "max_side") and v}
    return replace(tier, **clean) if clean else tier


def clamp_params(tier: Tier, width: int, height: int, num_frames: int):
    """Returns (width, height, num_frames, notes). Keeps aspect ratio, multiples of 32."""
    notes = []
    side = max(width, height)
    if side > tier.max_side:
        s = tier.max_side / side
        width, height = int(width * s) // 32 * 32, int(height * s) // 32 * 32
        notes.append(f"resolution clamped to {width}x{height}")
    if num_frames > tier.max_frames:
        num_frames = tier.max_frames
        notes.append(f"num_frames clamped to {num_frames}")
    return width, height, num_frames, notes
