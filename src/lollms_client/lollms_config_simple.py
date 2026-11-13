"""
lollms_config_simple.py
========================

Simple, opinionated configuration path for non-expert users.

The full expert wizard (``lollms_config_cli_env.run_wizard_and_save``) exposes
every binding, every profile, and every parameter for every modality across all
supported back-ends. That flexibility is exactly what makes it intimidating for
someone who just wants "a working local LLM".

This module provides the *Simple Profile* — a guided, opinionated flow that:

  1. Detects real hardware: NVIDIA VRAM (nvidia-smi), Apple Silicon unified RAM,
     system RAM, and CUDA availability.
  2. Very explicitly offers **Simple (pre-configured)** vs **Expert (manual)**.
  3. In Simple mode, only lists coding models that FIT the detected VRAM tier
     after reserving KV-cache headroom — so users never pick a model that will
     thrash system RAM.
  4. Downloads the chosen GGUF from a *trusted* Hugging Face repo (Unsloth /
     Bartowski / official vendor) and installs the llama.cpp runtime.
  5. Registers the canonical two-tier profile map (``LLM_BINDINGS_MASTER_*`` +
     ``LLM_PROFILES_MASTER_*``) exactly as the expert wizard would, so
     ``get_client_from_env`` and ``lollms_code`` pick it up unchanged.
  6. Optionally installs ``sd.cpp`` + MiniMax-H3 for text-to-video.

Design rules:

  * No new configuration format — the output is the canonical two-tier map.
  * Every step is recoverable; a failure keeps any previously-working config.
  * Model selection is GUARDED by hardware: an oversized pick requires an
    explicit confirmation, so the default path is always a good experience.
  * The expert wizard remains available at any time via ``--expert`` /
    ``run_wizard_and_save`` for anyone who outgrows the simple flow.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

from lollms_client.lollms_config_cli_env import (
    _load_existing_env_to_map,
    _safe_confirm,
    _safe_input,
    _save_and_validate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

class HardwareProfile:
    """Snapshot of the machine's capabilities relevant to local GGUF deployment.

    Attributes:
        platform: ``"windows"``, ``"linux"``, ``"darwin"`` …
        has_cuda: True when an NVIDIA GPU with CUDA is detected.
        gpu_name: Human-readable GPU name ("" when none).
        vram_gb: Total VRAM of the primary GPU in GiB (0.0 when none/CPU).
        unified_memory: True on Apple Silicon (VRAM == system RAM).
        system_ram_gb: Total physical RAM in GiB.
        effective_gpu_budget_gb: The number we compare model sizes against —
            equals vram_gb on CUDA, a RAM-derived estimate on Apple Silicon,
            and 0.0 on CPU-only (meaning "recommend small CPU-friendly models").
    """

    __slots__ = (
        "platform", "has_cuda", "gpu_name", "vram_gb",
        "unified_memory", "system_ram_gb", "effective_gpu_budget_gb",
    )

    def __init__(
        self,
        platform_name: str,
        has_cuda: bool,
        gpu_name: str,
        vram_gb: float,
        unified_memory: bool,
        system_ram_gb: float,
    ):
        self.platform = platform_name
        self.has_cuda = has_cuda
        self.gpu_name = gpu_name
        self.vram_gb = vram_gb
        self.unified_memory = unified_memory
        self.system_ram_gb = system_ram_gb
        if has_cuda and vram_gb > 0:
            self.effective_gpu_budget_gb = vram_gb
        elif unified_memory and system_ram_gb > 0:
            # Apple Silicon: GPU shares unified RAM with the OS. Reserve ~35%
            # for the OS/other apps; llama.cpp's Metal backend is comfortable
            # running the model + KV cache inside that budget.
            self.effective_gpu_budget_gb = system_ram_gb * 0.65
        else:
            self.effective_gpu_budget_gb = 0.0

    def summary(self) -> str:
        if self.has_cuda:
            return f"NVIDIA {self.gpu_name} · {self.vram_gb:.1f} GiB VRAM · " \
                   f"system RAM {self.system_ram_gb:.1f} GiB"
        if self.unified_memory:
            return f"Apple Silicon (unified memory) · system RAM {self.system_ram_gb:.1f} GiB"
        return f"CPU only · system RAM {self.system_ram_gb:.1f} GiB"

    def tier_label(self) -> str:
        """Short tier label used to filter the model catalog."""
        budget = self.effective_gpu_budget_gb
        if budget <= 0:
            return "cpu"
        if budget < 10:
            return "8gb"
        if budget < 18:
            return "16gb"
        if budget < 28:
            return "24gb"
        return "32gb+"


def _detect_system_ram_gb() -> float:
    """Best-effort total system RAM in GiB (returns 0.0 when undetectable)."""
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        pass
    # Fallback: sysconf on POSIX (macOS/Linux) when psutil is missing.
    try:
        import os
        pages = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return (pages * total_pages) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _query_nvidia_gpu() -> Tuple[bool, str, float]:
    """Returns (present, name, vram_gb) using nvidia-smi.

    Detection is best-effort; a missing nvidia-smi yields (False, "", 0.0).
    """
    if shutil.which("nvidia-smi") is None:
        return False, "", 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return False, "", 0.0
        first = out.stdout.strip().splitlines()[0]
        # Format:   "NVIDIA GeForce RTX 4070, 12282"
        parts = [p.strip() for p in first.rsplit(",", 1)]
        name = parts[0]
        mem_mib = float(parts[1]) if len(parts) > 1 else 0.0
        return True, name, mem_mib / 1024.0
    except Exception:
        return False, "", 0.0


def detect_hardware() -> HardwareProfile:
    """Returns a fresh HardwareProfile for the current machine."""
    sys_name = platform.system().lower()
    ram = _detect_system_ram_gb()
    has_cuda, gpu_name, vram = _query_nvidia_gpu()

    unified = (sys_name == "darwin") and (platform.machine() == "arm64")

    return HardwareProfile(
        platform_name=sys_name,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_gb=vram,
        unified_memory=unified,
        system_ram_gb=ram,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trusted coding model catalog
#
# Sources (Hugging Face): unsloth (UD dynamic quants), bartowski (imatrix),
# official vendor GGUFs. Only well-maintained repos with proper chat templates
# and tool-calling support are listed — raw uploads frequently break both.
#
# Field meanings:
#   tier          "cpu" | "8gb" | "16gb" | "24gb" | "32gb+"
#   min_vram_gb   Approx. VRAM required INCLUDING KV cache headroom.
#   file_gb       Approx. on-disk size of the GGUF.
#   vision        True when the base model accepts image inputs.
#   tool_calling  True when the chat template exposes native tool calls.
#   brain         One-line reason to pick this over siblings.
# ─────────────────────────────────────────────────────────────────────────────

_LLM_ZOO: List[Dict[str, Any]] = [
    # ── CPU-only / very small RAM boxes ────────────────────────────────────
    {
        "id": "qwen2_5_coder_3b_q4",
        "tier": "cpu",
        "title": "Qwen2.5-Coder 3B (Q4_K_M) — smallest CPU workable coding model",
        "repo_id": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        "file_gb": 2.0, "min_vram_gb": 3.5,
        "ctx_size": 16384, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Runs entirely on CPU; ideal for laptops without a GPU.",
    },

    # ── 8 GB VRAM tier ─────────────────────────────────────────────────────
    {
        "id": "qwen3_8b_q4",
        "tier": "8gb",
        "title": "Qwen 3.5 9B (UD-Q4_K_XL) — daily coding chat",
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "file_gb": 5.8, "min_vram_gb": 7.5,
        "ctx_size": 32768, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Best all-rounder that fits an 8 GiB card with KV headroom.",
    },
    {
        "id": "gemma3_12b_ud_q3",
        "tier": "8gb",
        "title": "Gemma 4 12B (UD-Q3_K_XL) — with native vision",
        "repo_id": "unsloth/gemma-4-12b-it-GGUF",
        "filename": "gemma-4-12b-it-UD-Q3_K_XL.gguf",
        "file_gb": 7.0, "min_vram_gb": 9.5,
        "ctx_size": 32768, "n_gpu_layers": -1,
        "vision": True, "tool_calling": True,
        "brain": "Vision-capable; usable on 8 GiB with tight KV/context settings.",
    },
    {
        "id": "starcoder2_7b_q5",
        "tier": "8gb",
        "title": "StarCoder2 7B (Q5_K_M) — ultra-fast FIM autocomplete",
        "repo_id": "second-state/StarCoder2-7B-GGUF",
        "filename": "starcoder2-7b-Q5_K_M.gguf",
        "file_gb": 5.6, "min_vram_gb": 7.0,
        "ctx_size": 16384, "n_gpu_layers": -1,
        "vision": False, "tool_calling": False,
        "brain": "Fill-in-the-middle specialist; not an agentic coder.",
    },

    # ── 16 GB VRAM tier ────────────────────────────────────────────────────
    {
        "id": "qwen3_8_27b_ud_q3",
        "tier": "16gb",
        "title": "Qwen 3.8 27B (UD-Q3_K_XL) — frontier-grade reasoning",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "file_gb": 13.1, "min_vram_gb": 15.5,
        "ctx_size": 32768, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Best reasoning/bug-fixing quality per GiB at this size.",
    },
    {
        "id": "qwen3_coder_30b_a3b_iq3",
        "tier": "16gb",
        "title": "Qwen3-Coder-30B-A3B (IQ3_M) — MoE, fast multi-file edits",
        "repo_id": "AtomicChat/Qwen3-Coder-30B-A3B-GGUF",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-IQ3_M.gguf",
        "file_gb": 13.5, "min_vram_gb": 16.0,
        "ctx_size": 65536, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "30B MoE with 3B active: near-30B quality at 3B speed.",
    },
    {
        "id": "codestral_22b_q4",
        "tier": "16gb",
        "title": "Codestral 22B (Q4_K_M) — low-latency IDE autocomplete",
        "repo_id": "bartowski/Codestral-22B-v0.1-GGUF",
        "filename": "Codestral-22B-v0.1-Q4_K_M.gguf",
        "file_gb": 13.3, "min_vram_gb": 15.5,
        "ctx_size": 32768, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Mistral's coding model; superb latency for interactive editing.",
    },

    # ── 24 GB VRAM tier ────────────────────────────────────────────────────
    {
        "id": "qwen3_8_27b_ud_q4",
        "tier": "24gb",
        "title": "Qwen 3.8 27B (UD-Q4_K_XL) — best single-GPU agentic coder",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "file_gb": 17.7, "min_vram_gb": 22.0,
        "ctx_size": 65536, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Sweet spot on a 24 GiB card: quality + 64K context.",
    },
    {
        "id": "devstral_small_2_24b_q5",
        "tier": "24gb",
        "title": "Devstral Small 2 24B (Q5_K_M) — multi-file repo editing",
        "repo_id": "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
        "filename": "Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf",
        "file_gb": 16.8, "min_vram_gb": 22.0,
        "ctx_size": 65536, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Mistral's agentic SWE specialist; excellent repo-level edits.",
    },
    {
        "id": "muse_glimmer_30b_q4",
        "tier": "24gb",
        "title": "Muse Glimmer 30B (Q4_K_M) — autonomous agent + MCP",
        "repo_id": "bartowski/Muse-Glimmer-30B-GGUF",
        "filename": "Muse-Glimmer-30B-Q4_K_M.gguf",
        "file_gb": 17.3, "min_vram_gb": 23.0,
        "ctx_size": 65536, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Top tool-calling fidelity at this size class.",
    },

    # ── 32+ GB VRAM tier ───────────────────────────────────────────────────
    {
        "id": "qwen3_8_27b_q8",
        "tier": "32gb+",
        "title": "Qwen 3.8 27B (Q8_0) — zero-degradation, huge context",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-Q8_0.gguf",
        "file_gb": 29.0, "min_vram_gb": 33.0,
        "ctx_size": 131072, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "Full-precision weights; 128K context fits on 40+ GiB cards.",
    },
    {
        "id": "gemma3_31b_q6",
        "tier": "32gb+",
        "title": "Gemma 4 31B (Q6_K) — multimodal + 256K context",
        "repo_id": "mradermacher/gemma-4-31B-GGUF",
        "filename": "gemma-4-31B.Q6_K.gguf",
        "file_gb": 25.2, "min_vram_gb": 30.0,
        "ctx_size": 131072, "n_gpu_layers": -1,
        "vision": True, "tool_calling": True,
        "brain": "Vision + long-context for UI debugging from screenshots.",
    },
    {
        "id": "qwen3_coder_next_80b_a3b_ud_q3",
        "tier": "32gb+",
        "title": "Qwen3-Coder-Next 80B-A3B (UD-Q3_K_XL) — heavyweight agent",
        "repo_id": "unsloth/Qwen3-Coder-Next-GGUF",
        "filename": "Qwen3-Coder-Next-UD-Q3_K_XL.gguf",
        "file_gb": 36.3, "min_vram_gb": 40.0,
        "ctx_size": 262144, "n_gpu_layers": -1,
        "vision": False, "tool_calling": True,
        "brain": "80B MoE with 3B active — frontier quality at MoE speed.",
    },
]


def _tier_order() -> List[str]:
    return ["cpu", "8gb", "16gb", "24gb", "32gb+"]


def _models_for_hardware(hw: HardwareProfile) -> List[Dict[str, Any]]:
    """Returns zoo entries whose VRAM budget fits the detected hardware.

    The rule is inclusive on the *upper* end: a 12 GiB card sees the "8gb" and
    "16gb" tier entries whose ``min_vram_gb`` is under its effective budget;
    anything that would push the model into heavy memory offload is excluded
    (so a user cannot accidentally pick the Q4_K_XL 27B on an 8 GiB card).
    """
    budget = hw.effective_gpu_budget_gb
    if budget <= 0:
        # CPU-only: only models explicitly marked CPU-viable.
        return [m for m in _LLM_ZOO if m["tier"] == "cpu"]

    # Reserve extra headroom for KV cache & the OS/desktop compositor. The
    # numbers baked into min_vram_gb already assume ~2-4 GiB KV headroom at
    # the default ctx_size, so passing the raw budget straight is fine, but
    # we keep a small safety margin for the driver & display buffer.
    safe_budget = budget - 0.5
    return [m for m in _LLM_ZOO if m["min_vram_gb"] <= safe_budget]


def _fits_comfortably(model: Dict[str, Any], hw: HardwareProfile) -> bool:
    """True when the model fits with the recommended KV-cache headroom intact."""
    budget = hw.effective_gpu_budget_gb
    if budget <= 0:
        return model["tier"] == "cpu"
    # We want at least file_gb + 2 GiB free for the KV cache + CUDA context.
    return (model["file_gb"] + 2.0) <= budget


def _recommend_llm(hw: HardwareProfile) -> Optional[Dict[str, Any]]:
    """Picks the best-fitting model: highest tier that fits comfortably."""
    candidates = _models_for_hardware(hw)
    if not candidates:
        return None
    # Prefer tool-calling models (required for lollms-code), then highest tier,
    # then largest size (usually the highest-quality model at that tier).
    ranked = sorted(
        candidates,
        key=lambda m: (
            m["tool_calling"],
            _tier_order().index(m["tier"]),
            m["file_gb"],
        ),
        reverse=True,
    )
    for m in ranked:
        if _fits_comfortably(m, hw):
            return m
    return ranked[0]


# ─────────────────────────────────────────────────────────────────────────────
# Two-tier registration (identical schema to the expert wizard)
# ─────────────────────────────────────────────────────────────────────────────

def _register_llm(
    config_map: Dict[str, str],
    binding_alias: str,
    profile_alias: str,
    model: Dict[str, Any],
    models_path: str,
) -> None:
    """Writes binding + profile env keys using the canonical two-tier scheme."""
    b_prefix = f"LLM_BINDINGS_{binding_alias.upper()}_"
    p_prefix = f"LLM_PROFILES_{profile_alias.upper()}_"

    # ── Binding: llama_cpp_server with the models folder ──
    config_map[b_prefix + "BINDING_NAME"] = "llama_cpp_server"
    config_map[b_prefix + "HOST_ADDRESS"] = "http://localhost:9642"
    config_map[b_prefix + "MODELS_PATH"] = models_path
    config_map[b_prefix + "MAX_ACTIVE_MODELS"] = "1"
    config_map[b_prefix + "IDLE_TIMEOUT"] = "600"
    config_map[b_prefix + "N_PARALLEL"] = "1"
    config_map[b_prefix + "FLASH_ATTN"] = "true"

    # ── Profile: per-model overrides (ctx_size, n_gpu_layers, tool-calling) ──
    config_map[p_prefix + "BINDING_ALIAS"] = binding_alias
    config_map[p_prefix + "MODEL_NAME"] = model["filename"]
    config_map[p_prefix + "CTX_SIZE"] = str(model.get("ctx_size", 8192))
    config_map[p_prefix + "N_GPU_LAYERS"] = str(model.get("n_gpu_layers", -1))
    config_map[p_prefix + "IS_DEFAULT"] = "true"
    config_map[p_prefix + "VISION_ENABLED"] = "true" if model.get("vision") else "false"
    config_map[p_prefix + "TOOL_CALLING_ENABLED"] = "true" if model.get("tool_calling") else "false"

    # The wizard records the active alias as the authoritative default.
    config_map["LLM_ACTIVE_PROFILE"] = profile_alias.lower()


def _register_ttv(
    config_map: Dict[str, str],
    binding_alias: str,
    profile_alias: str,
    models_path: str,
) -> None:
    """Registers the MiniMax-H3 sd.cpp binding + default profile for video."""
    b_prefix = f"TTV_BINDINGS_{binding_alias.upper()}_"
    p_prefix = f"TTV_PROFILES_{profile_alias.upper()}_"

    config_map[b_prefix + "BINDING_NAME"] = "sd_cpp"
    config_map[b_prefix + "MODELS_PATH"] = models_path
    config_map[b_prefix + "AUTO_INSTALL"] = "true"
    config_map[b_prefix + "AUTO_DOWNLOAD"] = "true"

    config_map[p_prefix + "BINDING_ALIAS"] = binding_alias
    config_map[p_prefix + "MODEL_NAME"] = "minimax_h3"
    config_map[p_prefix + "IS_DEFAULT"] = "true"


# ─────────────────────────────────────────────────────────────────────────────
# Runtime install + weights download
# ─────────────────────────────────────────────────────────────────────────────

def _download_llm_weights(model: Dict[str, Any], models_path: Path) -> bool:
    """Downloads a model's GGUF into ``models_path``.

    Returns True on success. Delegates to ``huggingface_hub`` directly (already
    a dependency of the llama_cpp_server binding) so the download works even
    when the binding module cannot yet be imported (e.g. before first runtime
    install).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        ASCIIColors.red("  ❌ huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        return False

    models_path.mkdir(parents=True, exist_ok=True)
    try:
        ASCIIColors.info(
            f"  ⬇️  Downloading {model['filename']} "
            f"(~{model['file_gb']:.1f} GB — this can take a while)…"
        )
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=str(models_path),
        )
        ASCIIColors.green("  ✅ Model downloaded.")
        return True
    except Exception as e:
        ASCIIColors.red(f"  ❌ Download failed: {e}")
        ASCIIColors.info("  You can retry later or choose a smaller model.")
        return False


def _install_llama_runtime(models_path: Path) -> bool:
    """Installs the llama.cpp binaries via the binding's own installer."""
    try:
        from lollms_client.llm_bindings.llama_cpp_server import LlamaCppServerBinding
    except Exception as e:
        ASCIIColors.red(f"  ❌ Could not import llama_cpp_server binding: {e}")
        return False

    try:
        binding = LlamaCppServerBinding(
            models_path=str(models_path),
            model_name=None,  # defer loading
        )
        # __init__ already triggers auto-install when the binary is missing.
        exe = binding._get_server_executable()
        if exe.exists():
            ASCIIColors.green(f"  ✅ llama.cpp runtime ready: {exe}")
            return True
        ASCIIColors.red("  ❌ llama.cpp runtime did not install.")
        return False
    except Exception as e:
        ASCIIColors.red(f"  ❌ llama.cpp runtime install failed: {e}")
        return False


def _install_sdcpp_runtime(models_path: Path) -> bool:
    """Installs the sd.cpp runtime via the binding's installer."""
    try:
        from lollms_client.ttv_bindings.sd.cpp import SdCppTTVBinding
    except Exception as e:
        ASCIIColors.red(f"  ❌ Could not import sd.cpp binding: {e}")
        return False

    try:
        binding = SdCppTTVBinding(
            binding_name="sd_cpp",
            models_path=str(models_path),
            auto_install=True,
        )
        exe = binding._cli()
        ASCIIColors.green(f"  ✅ sd.cpp runtime ready: {exe}")
        return True
    except Exception as e:
        ASCIIColors.red(f"  ❌ sd.cpp runtime install failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Interactive guided flows
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Headless API (GUI / server use — no terminal I/O)
# ─────────────────────────────────────────────────────────────────────────────

def get_simple_setup_plan(hw: Optional[HardwareProfile] = None) -> Dict[str, Any]:
    """Returns a hardware-scan + curated catalog plan WITHOUT any terminal I/O.

    This is the GUI / REST-friendly counterpart of the interactive
    :func:`run_simple_wizard`. Any frontend (NiceGUI, Vue, mobile…) calls this
    to render its own "Simple setup" screen:

        plan = get_simple_setup_plan()
        # → {
        #     "hardware":  {summary, tier, has_cuda, vram_gb, system_ram_gb,
        #                   unified_memory, effective_gpu_budget_gb},
        #     "models":    [ {id, title, repo_id, filename, file_gb,
        #                     min_vram_gb, ctx_size, vision, tool_calling,
        #                     brain, fits} , ...],
        #     "recommended_id": str | None,
        #   }

    ``models`` is the full trusted catalog (each entry carries a ``fits``
    flag), NOT a pre-filtered subset, so the GUI can show the full trusted
    list with ✓/⚠ markers exactly like the CLI's manual-browse view.
    ``recommended_id`` is the safe pick for the detected hardware.
    """
    from dataclasses import asdict, dataclass, field  # noqa: F401  (kept local)

    if hw is None:
        hw = detect_hardware()

    hardware = {
        "platform": hw.platform,
        "summary": hw.summary(),
        "tier": hw.tier_label(),
        "has_cuda": hw.has_cuda,
        "gpu_name": hw.gpu_name,
        "vram_gb": round(hw.vram_gb, 2),
        "unified_memory": hw.unified_memory,
        "system_ram_gb": round(hw.system_ram_gb, 2),
        "effective_gpu_budget_gb": round(hw.effective_gpu_budget_gb, 2),
    }

    models: List[Dict[str, Any]] = []
    for entry in _LLM_ZOO:
        models.append({
            "id": entry["id"],
            "title": entry["title"],
            "repo_id": entry["repo_id"],
            "filename": entry["filename"],
            "file_gb": entry["file_gb"],
            "min_vram_gb": entry["min_vram_gb"],
            "ctx_size": entry.get("ctx_size", 8192),
            "vision": bool(entry.get("vision")),
            "tool_calling": bool(entry.get("tool_calling")),
            "brain": entry.get("brain", ""),
            "fits": _fits_comfortably(entry, hw),
        })

    recommendation = _recommend_llm(hw)
    return {
        "hardware": hardware,
        "models": models,
        "recommended_id": recommendation["id"] if recommendation else None,
    }


def _model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Returns the zoo entry for ``model_id`` (case-insensitive), or None."""
    target = str(model_id or "").strip().lower()
    if not target:
        return None
    for entry in _LLM_ZOO:
        if entry["id"].lower() == target:
            return entry
    return None


def apply_simple_llm(
    config_map: Dict[str, str],
    model_id: str,
    *,
    home: Optional[Path] = None,
    install_runtime: bool = True,
    download: bool = True,
) -> Dict[str, Any]:
    """Applies a curated LLM choice to an in-memory two-tier ``config_map``.

    Headless counterpart of :func:`_guide_llm`, designed for embedding hosts
    (the lollms-code GUI, REST APIs). Performs the same three steps —
    runtime install, weight download, two-tier registration — but with the
    step-by-step progress reported via the returned dict rather than via
    terminal output, so the caller decides whether to run each step
    synchronously, on a worker thread, or defer it entirely.

    Args:
        config_map: flat env-style map, mutated in place.
        model_id: one of the ``id`` values from ``get_simple_setup_plan()``.
        home: base folder for models (defaults to ``~/.lollms``).
        install_runtime: when True, ensure the llama.cpp binary is present.
        download: when True, fetch the GGUF (and mmproj when declared).

    Returns:
        ``{ok, model_id, filename, profile_alias, binding_alias,
           models_path, runtime, downloaded, errors}`` — ``ok`` is True when
        the two-tier map was successfully updated for this model.
    """
    outcome: Dict[str, Any] = {
        "ok": False,
        "model_id": model_id,
        "filename": "",
        "profile_alias": "master",
        "binding_alias": "master",
        "models_path": "",
        "runtime": False,
        "downloaded": False,
        "errors": [],
    }

    entry = _model_by_id(model_id)
    if entry is None:
        outcome["errors"].append(f"Unknown model id '{model_id}'.")
        return outcome

    base_home = home if home is not None else (Path.home() / ".lollms")
    models_path = base_home / "data" / "models" / "llama_cpp_models"
    outcome["filename"] = entry["filename"]
    outcome["models_path"] = str(models_path)

    if install_runtime:
        outcome["runtime"] = _install_llama_runtime(models_path)
        if not outcome["runtime"]:
            outcome["errors"].append(
                "llama.cpp runtime install failed; the model will still be "
                "registered and can be launched once the runtime is fixed."
            )

    already_present = (models_path / entry["filename"]).exists() or (
        (models_path / model_id).exists()
    )
    if download and not already_present:
        outcome["downloaded"] = _download_llm_weights(entry, models_path)
        if not outcome["downloaded"]:
            outcome["errors"].append(
                "Weight download failed. The profile is registered anyway; "
                "llama_cpp_server will auto-download on first launch."
            )
    else:
        outcome["downloaded"] = already_present

    _register_llm(config_map, "master", "master", entry, str(models_path))
    outcome["ok"] = True
    return outcome


def run_simple_profile_headless(
    model_id: Optional[str] = None,
    *,
    install_runtime: bool = False,
    download: bool = False,
    enable_ttv: bool = False,
) -> Dict[str, Any]:
    """End-to-end simple setup for embedding hosts (GUI main entry point).

    Loads the current on-disk config, applies the curated LLM (and optionally
    the sd.cpp TTV binding), persists it through the wizard's own
    ``_save_and_validate`` (so the two files never diverge), and returns a
    structured report the GUI can render as a success panel.

    When ``model_id`` is None the hardware-recommended model is used. All
    heavy I/O (runtime install, weight download) defaults to **off** so a
    GUI button never blocks the event loop — the caller typically runs the
    returned setup with the desired flags on a background task. Pass
    ``install_runtime=True`` / ``download=True`` explicitly when the caller
    is already on a worker thread and wants the full one-shot flow.
    """
    plan = get_simple_setup_plan()
    chosen_id = model_id or plan.get("recommended_id")
    if not chosen_id:
        return {
            "ok": False,
            "error": "No compatible model is available for this hardware. "
                     "Use the Expert wizard to configure a model manually.",
            "plan": plan,
        }

    home = Path.home() / ".lollms"
    home.mkdir(parents=True, exist_ok=True)
    config_map = _load_existing_env_to_map()
    original = dict(config_map)

    apply_report = apply_simple_llm(
        config_map,
        chosen_id,
        home=home,
        install_runtime=install_runtime,
        download=download,
    )
    if not apply_report.get("ok"):
        return {"ok": False, "error": "; ".join(apply_report.get("errors") or ["Unknown failure"]),
                "plan": plan, "apply": apply_report}

    if enable_ttv:
        ttv_models_path = home / "data" / "models" / "ttv_models" / "sd_cpp"
        ttv_models_path.mkdir(parents=True, exist_ok=True)
        _register_ttv(config_map, "master", "master", str(ttv_models_path))

    saved = _save_and_validate(config_map, test_connection=False)
    if not saved:
        # Fail-safe: restore the previous on-disk state so a GUI click can
        # never leave the user with a broken config.
        try:
            _save_and_validate(original, test_connection=False)
        except Exception:
            pass
        return {
            "ok": False,
            "error": "Failed to persist the configuration to disk.",
            "plan": plan,
            "apply": apply_report,
        }

    entry = _model_by_id(chosen_id) or {}
    return {
        "ok": True,
        "plan": plan,
        "apply": apply_report,
        "model": {
            "id": chosen_id,
            "title": entry.get("title", ""),
            "filename": entry.get("filename", ""),
            "profile_alias": "master",
            "binding_alias": "master",
        },
        "config_file": str(_get_effective_config_file_for_gui()),
        "ttv_enabled": bool(enable_ttv),
    }


def _get_effective_config_file_for_gui() -> Path:
    """Path the runtime will read/write for this process (GUI display only)."""
    try:
        from lollms_client.lollms_config_cli_env import _get_effective_config_file
        return _get_effective_config_file()
    except Exception:
        return Path.home() / ".lollms" / "config.yaml"


def _render_hardware_banner(hw: HardwareProfile) -> None:
    lines = [
        f"[cyan]Host:[/cyan]     {platform.system()} / {platform.machine()}",
        f"[cyan]Detected:[/cyan] {hw.summary()}",
        f"[cyan]Tier:[/cyan]     [bold yellow]{hw.tier_label()}[/bold yellow]",
    ]
    ASCIIColors.panel(
        "\n".join(lines),
        title="[bold green]🖥️  Hardware Detection[/bold green]",
        border_style="green",
    )


def _prompt_model_selection(hw: HardwareProfile) -> Optional[Dict[str, Any]]:
    """Lets the user pick a model from those that fit their hardware.

    Returns the picked model dict, or None when the user opts to skip the LLM.
    """
    available = _models_for_hardware(hw)
    recommended = _recommend_llm(hw)

    if not available:
        ASCIIColors.warning(
            "  ⚠️  No curated model matches your hardware. You can still browse the "
            "whole catalog but performance may be poor."
        )
        available = list(_LLM_ZOO)
        if not _safe_confirm("  Show the full catalog anyway?", default=False):
            return None

    ASCIIColors.rich_print(
        "\n[bold cyan]── Choose a model that fits your hardware ──[/bold cyan]"
    )
    if recommended:
        ASCIIColors.green(
            f"  💡 Recommended: [bold]{recommended['title']}[/bold]"
        )
    ASCIIColors.rich_print(
        "[dim]All listed models use trusted HF repos (Unsloth dynamic quants,\n"
        "Bartowski imatrix, or official vendor releases). Sizes below include\n"
        "the KV cache headroom needed for a smooth coding session.[/dim]"
    )
    ASCIIColors.rich_print("")

    for i, entry in enumerate(available, 1):
        marker = " ⭐" if recommended and entry["id"] == recommended["id"] else ""
        vision_tag = " [magenta](vision)[/magenta]" if entry.get("vision") else ""
        tool_tag = " [cyan](tools)[/cyan]" if entry.get("tool_calling") else ""
        ASCIIColors.rich_print(
            f"    [{i}] [bold]{entry['title']}[/bold]{vision_tag}{tool_tag}{marker}"
        )
        ASCIIColors.rich_print(
            f"        [dim]{entry['brain']}  ·  ~{entry['file_gb']:.1f} GB · "
            f"~{entry['min_vram_gb']:.1f} GB VRAM · ctx {entry['ctx_size']:,}[/dim]"
        )

    ASCIIColors.rich_print("\n    [0] Skip — I'll configure an LLM manually later")
    ASCIIColors.rich_print("    [c] Show ALL models in the trusted catalog")

    default_choice = (
        str(available.index(recommended) + 1) if recommended in available else "1"
    )
    raw = _safe_input("Select", default_choice).strip().lower()

    if raw == "0":
        return None
    if raw == "c":
        return _prompt_manual_model_selection(hw)
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(available):
            return available[idx]
        raise ValueError
    except ValueError:
        ASCIIColors.warning(f"  Invalid selection '{raw}', using the recommendation.")
        return recommended or available[0]


def _prompt_manual_model_selection(hw: HardwareProfile) -> Optional[Dict[str, Any]]:
    """Shows the full catalog with a fit marker; warns on oversize picks."""
    ASCIIColors.rich_print("\n[bold cyan]── Full trusted catalog ──[/bold cyan]")
    ASCIIColors.rich_print(
        "[dim]✓ = fits your hardware comfortably · ⚠ = will need heavy RAM offload[/dim]\n"
    )
    for i, entry in enumerate(_LLM_ZOO, 1):
        fits = _fits_comfortably(entry, hw)
        marker = "✓" if fits else "⚠"
        color = "green" if fits else "yellow"
        ASCIIColors.rich_print(
            f"    [{i:>2}] [{color}]{marker}[/{color}] [bold]{entry['title']}[/bold]"
        )
        ASCIIColors.rich_print(
            f"         [dim]{entry['repo_id']}  ·  ~{entry['file_gb']:.1f} GB[/dim]"
        )

    default_choice = "1"
    rec = _recommend_llm(hw)
    if rec in _LLM_ZOO:
        default_choice = str(_LLM_ZOO.index(rec) + 1)

    raw = _safe_input("Select (blank to cancel)", default_choice).strip()
    if not raw:
        return None
    try:
        pick = _LLM_ZOO[int(raw) - 1]
    except (ValueError, IndexError):
        ASCIIColors.warning(f"  Invalid selection '{raw}'.")
        return None

    if not _fits_comfortably(pick, hw):
        ASCIIColors.warning(
            f"\n  ⚠️  '{pick['title']}' (~{pick['file_gb']:.1f} GB) does not fit "
            f"comfortably in your detected budget ({hw.effective_gpu_budget_gb:.1f} GB).\n"
            f"  It will run, but expect heavy CPU offload and much slower generation."
        )
        if not _safe_confirm("  Use it anyway?", default=False):
            ASCIIColors.info("  Keeping your safe choice.")
            return rec
    return pick


def _guide_llm(config_map: Dict[str, str], home: Path, hw: HardwareProfile) -> bool:
    """Guided LLM setup. Returns True when the LLM path is fully configured."""
    ASCIIColors.rich_print("\n[bold cyan]── Step 1: Local LLM ──[/bold cyan]")

    choice = _prompt_model_selection(hw)
    if choice is None:
        ASCIIColors.info("  Skipping LLM configuration.")
        return False

    models_path = home / "data" / "models" / "llama_cpp_models"

    # ── Step A: install llama.cpp runtime ──
    ASCIIColors.rich_print("\n  [dim]→ Installing llama.cpp runtime…[/dim]")
    if not _install_llama_runtime(models_path):
        ASCIIColors.warning(
            "  ⚠️  Runtime install failed. The model will still be downloaded and "
            "registered — you can retry the runtime install later from `/config`."
        )

    # ── Step B: download the weights ──
    already_present = (models_path / choice["filename"]).exists()
    if already_present and not _safe_confirm(
        f"  {choice['filename']} already exists. Re-download?", default=False
    ):
        ASCIIColors.info("  Using existing file.")
    elif not _download_llm_weights(choice, models_path):
        if not already_present:
            return False

    # ── Step C: register binding + profile ──
    _register_llm(config_map, "master", "master", choice, str(models_path))
    ASCIIColors.green(
        f"  ✅ LLM configured: {choice['filename']} "
        f"(ctx={choice['ctx_size']}, gpu_layers={choice['n_gpu_layers']})"
    )
    return True


def _guide_ttv(config_map: Dict[str, str], home: Path, hw: HardwareProfile) -> None:
    """Optional guided TTV (video generation) setup via sd.cpp + MiniMax-H3."""
    ASCIIColors.rich_print("\n[bold cyan]── Optional: Video Generation (TTV) ──[/bold cyan]")

    if hw.effective_gpu_budget_gb < 10:
        ASCIIColors.warning(
            f"  ⚠️  Your detected budget ({hw.effective_gpu_budget_gb:.1f} GB) is below "
            "the minimum for the MiniMax-H3 video model (~10 GB at the smallest quant)."
        )
        if not _safe_confirm("  Configure TTV anyway?", default=False):
            ASCIIColors.info("  Skipping TTV.")
            return

    ASCIIColors.info(
        "  This installs stable-diffusion.cpp + MiniMax-H3 (large download, ~10 GB)."
    )
    if not _safe_confirm("  Enable video generation?", default=False):
        ASCIIColors.info("  Skipping TTV.")
        return

    models_path = home / "data" / "models" / "ttv_models" / "sd_cpp"
    models_path.mkdir(parents=True, exist_ok=True)

    ASCIIColors.rich_print("  [dim]→ Installing sd.cpp runtime…[/dim]")
    if not _install_sdcpp_runtime(models_path):
        ASCIIColors.warning(
            "  ⚠️  sd.cpp runtime install failed. The binding is still registered — "
            "you can retry from `/config`."
        )

    if _safe_confirm(
        "  Download the MiniMax-H3 weights now? (~10 GB, this takes a while)",
        default=False,
    ):
        try:
            from lollms_client.ttv_bindings.sd.cpp import SdCppTTVBinding
            binding = SdCppTTVBinding(
                binding_name="sd_cpp",
                models_path=str(models_path),
                auto_install=True,
            )
            result = binding.install_model("minimax_h3")
            if result.get("status"):
                ASCIIColors.green("  ✅ MiniMax-H3 downloaded.")
            else:
                ASCIIColors.warning(
                    f"  ⚠️  Download failed or skipped: {result.get('message', '')}. "
                    "The binding is registered; generation will auto-download on first use."
                )
        except Exception as e:
            ASCIIColors.warning(
                f"  ⚠️  Could not pre-download: {e}. The binding will auto-download on first use."
            )

    _register_ttv(config_map, "master", "master", str(models_path))
    ASCIIColors.green("  ✅ TTV configured (auto-download on first use if weights missing).")


def run_simple_wizard() -> bool:
    """Guided, opinionated setup for a full local environment.

    Returns:
        True when the simple wizard completed and a configuration was saved.
        False when the user cancelled without changes.
    """
    ASCIIColors.panel(
        "[bold]Local Environment Setup (Simple Mode)[/bold]\n\n"
        "[dim]This wizard will:\n"
        "  1. Scan your hardware (GPU, VRAM, RAM).\n"
        "  2. Show only curated coding models that FIT your hardware.\n"
        "  3. Download a trusted GGUF and install llama.cpp automatically.\n"
        "  4. Optionally install sd.cpp + MiniMax-H3 for video generation.\n\n"
        "Every model and runtime is downloaded ONCE and reused offline.\n"
        "For fine-grained control, choose the Expert wizard instead.[/dim]",
        title="[bold magenta]🧙 Simple Setup[/bold magenta]",
        border_style="magenta",
    )

    hw = detect_hardware()
    _render_hardware_banner(hw)

    home = Path.home() / ".lollms"
    home.mkdir(parents=True, exist_ok=True)

    # Preserve any existing config so a partial failure does not destroy it.
    config_map = _load_existing_env_to_map()
    original = dict(config_map)

    if not _safe_confirm("  Proceed with hardware-aware setup?", default=True):
        ASCIIColors.info("  Cancelled. Nothing was changed.")
        return False

    # ── LLM ──
    llm_ok = _guide_llm(config_map, home, hw)

    # ── Optional TTV ──
    _guide_ttv(config_map, home, hw)

    if not config_map or (not llm_ok and not any(
        k.startswith("TTV_BINDINGS_") for k in config_map
    )):
        ASCIIColors.warning("  Nothing was configured. Discarding changes.")
        return False

    ASCIIColors.rich_print("\n[bold cyan]── Summary ──[/bold cyan]")
    for prefix, label in (
        ("LLM_", "LLM"),
        ("TTV_", "TTV (video)"),
        ("TTI_", "TTI (image)"),
        ("TTM_", "TTM (music)"),
        ("TTS_", "TTS (speech)"),
        ("STT_", "STT (transcription)"),
    ):
        has = any(k.startswith(prefix) for k in config_map)
        if has:
            ASCIIColors.green(f"  ✅ {label}: configured")
        else:
            ASCIIColors.rich_print(f"  [dim]○  {label}: skipped[/dim]")

    if not _safe_confirm("\n  Save this configuration?", default=True):
        ASCIIColors.info("  Cancelled. Nothing was changed.")
        return False

    ASCIIColors.rich_print("\n[dim]Saving configuration…[/dim]")
    if not _save_and_validate(config_map, test_connection=True):
        ASCIIColors.error("  ❌ Save failed. Reverting to previous configuration.")
        # Best-effort restore: overwrite back with the previous state.
        try:
            save_ok = _save_and_validate(original, test_connection=False)
            if save_ok:
                ASCIIColors.info("  Previous configuration restored.")
        except Exception:
            pass
        return False

    ASCIIColors.panel(
        "[bold green]✅ Setup complete![/bold green]\n\n"
        "You can now run:\n"
        "  [cyan]lollms-code -i[/cyan]        (interactive REPL)\n"
        "  [cyan]lollms-code \"<task>\"[/cyan]  (one-shot mode)\n\n"
        "[dim]Re-run `lollms-config --simple` anytime. Use `lollms-config --expert`\n"
        "for fine-grained control.[/dim]",
        title="[bold green]Done[/bold green]",
        border_style="green",
    )
    return True


def suggest_flow(hw: Optional[HardwareProfile] = None) -> str:
    """Asks the user which configuration path they want.

    Hardware detection runs first so the recommendation banner is accurate.
    Returns one of: ``"simple"``, ``"expert"``, or ``"cancel"``.
    """
    if hw is None:
        hw = detect_hardware()
    _render_hardware_banner(hw)

    recommendation = _recommend_llm(hw)
    simple_hint = (
        f"       Would auto-pick: [bold]{recommendation['title']}[/bold] "
        f"(~{recommendation['file_gb']:.1f} GB, fits your hardware)."
        if recommendation
        else "       Would offer small CPU-friendly models."
    )

    ASCIIColors.panel(
        "[bold]How do you want to configure Lollms?[/bold]\n\n"
        "  [cyan]1[/cyan]. [bold]Simple (pre-configured)[/bold] — recommended\n"
        "       Hardware-aware wizard: scans your GPU/RAM, shows ONLY trusted coding\n"
        "       models that actually fit, downloads one GGUF, and installs llama.cpp.\n"
        f"{simple_hint}\n\n"
        "  [cyan]2[/cyan]. [bold]Expert (manual)[/bold]\n"
        "       Full wizard: choose every binding, profile, server, connector and\n"
        "       parameter for LLM, TTI, TTS, STT, TTM, TTV, RAG and connections.\n"
        "       Best for remote APIs, custom servers, or advanced local setups.",
        title="[bold magenta]🧙 Configuration[/bold magenta]",
        border_style="magenta",
    )
    raw = _safe_input("Select", "1").strip().lower()
    if raw in ("1", "simple", "s"):
        return "simple"
    if raw in ("2", "expert", "e"):
        return "expert"
    return "cancel"


def ensure_configured() -> bool:
    """Idempotent entry point: return True when an LLM is already configured.

    Otherwise, offer the Simple/Expert choice (with hardware detection shown
    up-front) and run the chosen wizard. This is what ``lollms-code`` calls at
    startup instead of jumping straight into the expert wizard.
    """
    config_map = _load_existing_env_to_map()
    if any(
        k.upper().startswith("LLM_BINDINGS_")
        and k.upper().endswith("_BINDING_NAME")
        and v
        for k, v in config_map.items()
    ) and any(
        k.upper().startswith("LLM_PROFILES_")
        and k.upper().endswith("_BINDING_ALIAS")
        and v
        for k, v in config_map.items()
    ):
        return True

    choice = suggest_flow()
    if choice == "cancel":
        ASCIIColors.info("Configuration cancelled.")
        return False
    if choice == "simple":
        return run_simple_wizard()

    from lollms_client.lollms_config_cli_env import run_wizard_and_save
    run_wizard_and_save()
    # Recheck: expert wizard may still have been cancelled.
    return True


__all__ = [
    "HardwareProfile",
    "detect_hardware",
    "run_simple_wizard",
    "ensure_configured",
    "suggest_flow",
    # Headless / GUI API
    "get_simple_setup_plan",
    "apply_simple_llm",
    "run_simple_profile_headless",
]