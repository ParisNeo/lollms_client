# lollms_client/tti_bindings/diffusers/__init__.py
import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import base64
import pipmaster as pm
import threading
import queue
from concurrent.futures import Future
import time
import hashlib
import requests
from tqdm import tqdm
import json
import shutil
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors

pm.ensure_packages(["torch","torchvision"],index_url="https://download.pytorch.org/whl/cu126")
pm.ensure_packages(["pillow","transformers","safetensors","requests","tqdm"])
pm.ensure_packages([
    {
        "name": "diffusers",
        "vcs": "git+https://github.com/huggingface/diffusers.git",
        "condition": ">=0.35.1"
    }
])
try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        DiffusionPipeline,
        StableDiffusionPipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline
    )
    from diffusers.utils import load_image
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoPipelineForText2Image = None
    AutoPipelineForImage2Image = None
    AutoPipelineForInpainting = None
    DiffusionPipeline = None
    StableDiffusionPipeline = None
    QwenImageEditPipeline = None
    QwenImageEditPlusPipeline = None
    Image = None
    load_image = None
    DIFFUSERS_AVAILABLE = False

BindingName = "DiffusersTTIBinding_Impl"

CIVITAI_MODELS = {
    "realistic-vision-v6": {
        "display_name": "Realistic Vision V6.0",
        "url": "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "realisticVisionV60_v60B1.safetensors",
        "description": "Photorealistic SD1.5 checkpoint.",
        "owned_by": "civitai"
    },
    "absolute-reality": {
        "display_name": "Absolute Reality",
        "url": "https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "absolutereality_v181.safetensors",
        "description": "General realistic SD1.5.",
        "owned_by": "civitai"
    },
    "dreamshaper-8": {
        "display_name": "DreamShaper 8",
        "url": "https://civitai.com/api/download/models/128713",
        "filename": "dreamshaper_8.safetensors",
        "description": "Versatile SD1.5 style model.",
        "owned_by": "civitai"
    },
    "juggernaut-xl": {
        "display_name": "Juggernaut XL",
        "url": "https://civitai.com/api/download/models/133005",
        "filename": "juggernautXL_version6Rundiffusion.safetensors",
        "description": "Artistic SDXL.",
        "owned_by": "civitai"
    },
    "lyriel-v1.6": {
        "display_name": "Lyriel v1.6",
        "url": "https://civitai.com/api/download/models/72396?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "lyriel_v16.safetensors",
        "description": "Fantasy/stylized SD1.5.",
        "owned_by": "civitai"
    },
    "ui_icons": {
        "display_name": "UI Icons",
        "url": "https://civitai.com/api/download/models/367044?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "uiIcons_v10.safetensors",
        "description": "A model for generating UI icons.",
        "owned_by": "civitai"
    },
    "meinamix": {
        "display_name": "MeinaMix",
        "url": "https://civitai.com/api/download/models/948574?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "meinamix_meinaV11.safetensors",
        "description": "Anime/illustration SD1.5.",
        "owned_by": "civitai"
    },
    "rpg-v5": {
        "display_name": "RPG v5",
        "url": "https://civitai.com/api/download/models/124626?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "rpg_v5.safetensors",
        "description": "RPG assets SD1.5.",
        "owned_by": "civitai"
    },
    "pixel-art-xl": {
        "display_name": "Pixel Art XL",
        "url": "https://civitai.com/api/download/models/135931?type=Model&format=SafeTensor",
        "filename": "pixelartxl_v11.safetensors",
        "description": "Pixel art SDXL.",
        "owned_by": "civitai"
    },
    "lowpoly-world": {
        "display_name": "Lowpoly World",
        "url": "https://civitai.com/api/download/models/146502?type=Model&format=SafeTensor",
        "filename": "LowpolySDXL.safetensors",
        "description": "Lowpoly style SD1.5.",
        "owned_by": "civitai"
    },
    "toonyou": {
        "display_name": "ToonYou",
        "url": "https://civitai.com/api/download/models/125771?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "toonyou_beta6.safetensors",
        "description": "Cartoon/Disney SD1.5.",
        "owned_by": "civitai"
    },
    "papercut": {
        "display_name": "Papercut",
        "url": "https://civitai.com/api/download/models/133503?type=Model&format=SafeTensor",
        "filename": "papercut.safetensors",
        "description": "Paper cutout SD1.5.",
        "owned_by": "civitai"
    },
    "fantassifiedIcons": {
        "display_name": "Fantassified Icons",
        "url": "https://civitai.com/api/download/models/67584?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "fantassifiedIcons_fantassifiedIconsV20.safetensors",
        "description": "Flat, modern Icons.",
        "owned_by": "civitai"
    },
    "game_icon_institute": {
        "display_name": "Game icon institute",
        "url": "https://civitai.com/api/download/models/158776?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "gameIconInstituteV10_v10.safetensors",
        "description": "Flat, modern game Icons.",
        "owned_by": "civitai"
    },
    "M4RV3LS_DUNGEONS": {
        "display_name": "M4RV3LS & DUNGEONS",
        "url": "https://civitai.com/api/download/models/139417?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "M4RV3LSDUNGEONSNEWV40COMICS_mD40.safetensors",
        "description": "comics.",
        "owned_by": "civitai"
    },
}

TORCH_DTYPE_MAP_STR_TO_OBJ = {
    "float16": getattr(torch, 'float16', 'float16'),
    "bfloat16": getattr(torch, 'bfloat16', 'bfloat16'),
    "float32": getattr(torch, 'float32', 'float32'),
    "auto": "auto"
}
TORCH_DTYPE_MAP_OBJ_TO_STR = {v: k for k, v in TORCH_DTYPE_MAP_STR_TO_OBJ.items()}
if torch:
    TORCH_DTYPE_MAP_OBJ_TO_STR[None] = "None"

SCHEDULER_MAPPING = {
    "default": None,
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "deis_multistep": "DEISMultistepScheduler",
    "dpm_multistep": "DPMSolverMultistepScheduler",
    "dpm_multistep_karras": "DPMSolverMultistepScheduler",
    "dpm_single": "DPMSolverSinglestepScheduler",
    "dpm_adaptive": "DPMSolverPlusPlusScheduler",
    "dpm++_2m": "DPMSolverMultistepScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "dpm++_2s_ancestral": "DPMSolverAncestralDiscreteScheduler",
    "dpm++_2s_ancestral_karras": "DPMSolverAncestralDiscreteScheduler",
    "dpm++_sde": "DPMSolverSDEScheduler",
    "dpm++_sde_karras": "DPMSolverSDEScheduler",
    "euler_ancestral_discrete": "EulerAncestralDiscreteScheduler",
    "euler_discrete": "EulerDiscreteScheduler",
    "heun_discrete": "HeunDiscreteScheduler",
    "heun_karras": "HeunDiscreteScheduler",
    "lms_discrete": "LMSDiscreteScheduler",
    "lms_karras": "LMSDiscreteScheduler",
    "pndm": "PNDMScheduler",
    "unipc_multistep": "UniPCMultistepScheduler",
    "dpm++_2m_sde": "DPMSolverMultistepScheduler",
    "dpm++_2m_sde_karras": "DPMSolverMultistepScheduler",
    "dpm2": "KDPM2DiscreteScheduler",
    "dpm2_karras": "KDPM2DiscreteScheduler",
    "dpm2_a": "KDPM2AncestralDiscreteScheduler",
    "dpm2_a_karras": "KDPM2AncestralDiscreteScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "lms": "LMSDiscreteScheduler"
}
SCHEDULER_USES_KARRAS_SIGMAS = [
    "dpm_multistep_karras","dpm++_2m_karras","dpm++_2s_ancestral_karras",
    "dpm++_sde_karras","heun_karras","lms_karras",
    "dpm++_2m_sde_karras","dpm2_karras","dpm2_a_karras"
]

class ModelManager:
    def __init__(self, config: Dict[str, Any], models_path: Path):
        self.config = config
        self.models_path = models_path
        self.pipeline: Optional[DiffusionPipeline] = None
        self.current_task: Optional[str] = None
        self.ref_count = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.is_loaded = False
        self.last_used_time = time.time()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.worker_thread.start()
        self._stop_monitor_event = threading.Event()
        self._unload_monitor_thread = None
        self._start_unload_monitor()

    def acquire(self):
        with self.lock:
            self.ref_count += 1
            return self

    def release(self):
        with self.lock:
            self.ref_count -= 1
            return self.ref_count

    def stop(self):
        self._stop_event.set()
        if self._unload_monitor_thread:
            self._stop_monitor_event.set()
            self._unload_monitor_thread.join(timeout=2)
        self.queue.put(None)
        self.worker_thread.join(timeout=5)

    def _start_unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after > 0 and self._unload_monitor_thread is None:
            self._stop_monitor_event.clear()
            self._unload_monitor_thread = threading.Thread(target=self._unload_monitor, daemon=True)
            self._unload_monitor_thread.start()

    def _unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after <= 0:
            return
        ASCIIColors.info(f"Starting inactivity monitor for '{self.config['model_name']}' (timeout: {unload_after}s).")
        while not self._stop_monitor_event.wait(timeout=5.0):
            with self.lock:
                if not self.is_loaded:
                    continue
                if time.time() - self.last_used_time > unload_after:
                    ASCIIColors.info(f"Model '{self.config['model_name']}' has been inactive. Unloading.")
                    self._unload_pipeline()

    def _resolve_model_path(self, model_name: str) -> Union[str, Path]:
        path_obj = Path(model_name)
        if path_obj.is_absolute() and path_obj.exists():
            return model_name
        if model_name in CIVITAI_MODELS:
            filename = CIVITAI_MODELS[model_name]["filename"]
            local_path = self.models_path / filename
            if not local_path.exists():
                self._download_civitai_model(model_name)
            return local_path
        local_path = self.models_path / model_name
        if local_path.exists():
            return local_path
        return model_name

    def _download_civitai_model(self, model_key: str):
        model_info = CIVITAI_MODELS[model_key]
        url = model_info["url"]
        filename = model_info["filename"]
        dest_path = self.models_path / filename
        temp_path = dest_path.with_suffix(".temp")
        ASCIIColors.cyan(f"Downloading '{filename}' from Civitai...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(temp_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {filename}") as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            shutil.move(temp_path, dest_path)
            ASCIIColors.green(f"Model '{filename}' downloaded successfully.")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to download model {filename}: {e}") from e

    def _set_scheduler(self):
        if not self.pipeline:
            return
        scheduler_name_key = self.config["scheduler_name"].lower()
        if scheduler_name_key == "default":
            return
        scheduler_class_name = SCHEDULER_MAPPING.get(scheduler_name_key)
        if scheduler_class_name:
            try:
                SchedulerClass = getattr(importlib.import_module("diffusers.schedulers"), scheduler_class_name)
                scheduler_config = self.pipeline.scheduler.config
                scheduler_config["use_karras_sigmas"] = scheduler_name_key in SCHEDULER_USES_KARRAS_SIGMAS
                self.pipeline.scheduler = SchedulerClass.from_config(scheduler_config)
                ASCIIColors.info(f"Switched scheduler to {scheduler_class_name}")
            except Exception as e:
                ASCIIColors.warning(f"Could not switch scheduler to {scheduler_name_key}: {e}. Using current default.")

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task:
            return
        if self.pipeline:
            self._unload_pipeline()
        model_name = self.config.get("model_name", "")
        if not model_name:
            raise ValueError("Model name cannot be empty for loading.")
        ASCIIColors.info(f"Loading Diffusers model: {model_name} for task: {task}")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        try:
            load_args = {}
            if self.config.get("hf_cache_path"):
                load_args["cache_dir"] = str(self.config["hf_cache_path"])
            if str(model_path).endswith(".safetensors"):
                if task == "text2image":
                    try:
                        self.pipeline = AutoPipelineForText2Image.from_single_file(model_path, torch_dtype=torch_dtype, cache_dir=load_args.get("cache_dir"))
                    except AttributeError:
                        self.pipeline = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, cache_dir=load_args.get("cache_dir"))
                elif task == "image2image":
                    self.pipeline = AutoPipelineForImage2Image.from_single_file(model_path, torch_dtype=torch_dtype, cache_dir=load_args.get("cache_dir"))
                elif task == "inpainting":
                    self.pipeline = AutoPipelineForInpainting.from_single_file(model_path, torch_dtype=torch_dtype, cache_dir=load_args.get("cache_dir"))
            else:
                common_args = {
                    "torch_dtype": torch_dtype,
                    "use_safetensors": self.config["use_safetensors"],
                    "token": self.config["hf_token"],
                    "local_files_only": self.config["local_files_only"]
                }
                if self.config["hf_variant"]:
                    common_args["variant"] = self.config["hf_variant"]
                if not self.config["safety_checker_on"]:
                    common_args["safety_checker"] = None
                if self.config.get("hf_cache_path"):
                    common_args["cache_dir"] = str(self.config["hf_cache_path"])

                if "Qwen-Image-Edit-2509" in str(model_path):
                    common_args.pop('size', None)
                    self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, **common_args)
                elif "Qwen-Image-Edit" in str(model_path):
                    common_args.pop('size', None)
                    self.pipeline = QwenImageEditPipeline.from_pretrained(model_path, **common_args)
                elif task == "text2image":
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **common_args)
                elif task == "image2image":
                    self.pipeline = AutoPipelineForImage2Image.from_pretrained(model_path, **common_args)
                elif task == "inpainting":
                    self.pipeline = AutoPipelineForInpainting.from_pretrained(model_path, **common_args)
        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "gated" in error_str or "authorization" in error_str:
                msg = (
                    f"AUTHENTICATION FAILED for model '{model_name}'. "
                    "Please ensure you accepted the model license and provided a valid HF token."
                )
                raise RuntimeError(msg) from e
            raise e
        self._set_scheduler()
        self.pipeline.to(self.config["device"])
        if self.config["enable_xformers"]:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                ASCIIColors.warning(f"Could not enable xFormers: {e}.")
        if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
            self.pipeline.enable_model_cpu_offload()
        elif self.config["enable_sequential_cpu_offload"] and self.config["device"] != "cpu":
            self.pipeline.enable_sequential_cpu_offload()
        self.is_loaded = True
        self.current_task = task
        self.last_used_time = time.time()
        ASCIIColors.green(f"Model '{model_name}' loaded successfully on '{self.config['device']}' for task '{task}'.")

    def _unload_pipeline(self):
        if self.pipeline:
            model_name = self.config.get('model_name', 'Unknown')
            del self.pipeline
            self.pipeline = None
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            self.current_task = None
            ASCIIColors.info(f"Model '{model_name}' unloaded and VRAM cleared.")

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None:
                    break
                future, task, pipeline_args = job
                try:
                    with self.lock:
                        self.last_used_time = time.time()
                        if not self.is_loaded or self.current_task != task:
                            self._load_pipeline_for_task(task)
                    with torch.no_grad():
                        output = self.pipeline(**pipeline_args)
                    pil = output.images[0]
                    buf = BytesIO()
                    pil.save(buf, format="PNG")
                    future.set_result(buf.getvalue())
                except Exception as e:
                    trace_exception(e)
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

class PipelineRegistry:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._managers = {}
                cls._instance._registry_lock = threading.Lock()
        return cls._instance
    @staticmethod
    def _get_critical_keys():
        return [
            "model_name","device","torch_dtype_str","use_safetensors",
            "safety_checker_on","hf_variant","enable_cpu_offload",
            "enable_sequential_cpu_offload","enable_xformers",
            "local_files_only","hf_cache_path","unload_inactive_model_after"
        ]
    def _get_config_key(self, config: Dict[str, Any]) -> str:
        key_data = tuple(sorted((k, config.get(k)) for k in self._get_critical_keys()))
        return hashlib.sha256(str(key_data).encode('utf-8')).hexdigest()
    def get_manager(self, config: Dict[str, Any], models_path: Path) -> ModelManager:
        key = self._get_config_key(config)
        with self._registry_lock:
            if key not in self._managers:
                self._managers[key] = ModelManager(config.copy(), models_path)
            return self._managers[key].acquire()
    def release_manager(self, config: Dict[str, Any]):
        key = self._get_config_key(config)
        with self._registry_lock:
            if key in self._managers:
                manager = self._managers[key]
                ref_count = manager.release()
                if ref_count == 0:
                    ASCIIColors.info(f"Reference count for model '{config.get('model_name')}' is zero. Cleaning up manager.")
                    manager.stop()
                    with manager.lock:
                        manager._unload_pipeline()
                    del self._managers[key]
    def get_active_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return [m for m in self._managers.values() if m.is_loaded]

class DiffusersTTIBinding_Impl(LollmsTTIBinding):
    DEFAULT_CONFIG = {
        "model_name": "",
        "device": "auto",
        "torch_dtype_str": "auto",
        "use_safetensors": True,
        "scheduler_name": "default",
        "safety_checker_on": True,
        "num_inference_steps": 25,
        "guidance_scale": 7.0,
        "width": 512,
        "height": 512,
        "seed": -1,
        "enable_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "enable_xformers": False,
        "hf_variant": None,
        "hf_token": None,
        "hf_cache_path": None,
        "local_files_only": False,
        "unload_inactive_model_after": 0
    }
    HF_DEFAULT_MODELS = [
        {"family": "SDXL", "model_name": "stabilityai/stable-diffusion-xl-base-1.0", "display_name": "SDXL Base 1.0", "desc": "Text2Image 1024 native."},
        {"family": "SDXL", "model_name": "stabilityai/stable-diffusion-xl-refiner-1.0", "display_name": "SDXL Refiner 1.0", "desc": "Refiner for SDXL."},
        {"family": "SD 1.x", "model_name": "runwayml/stable-diffusion-v1-5", "display_name": "Stable Diffusion 1.5", "desc": "Classic SD1.5."},
        {"family": "SD 2.x", "model_name": "stabilityai/stable-diffusion-2-1", "display_name": "Stable Diffusion 2.1", "desc": "SD2.1 base."},
        {"family": "SD3", "model_name": "stabilityai/stable-diffusion-3-medium-diffusers", "display_name": "Stable Diffusion 3 Medium", "desc": "SD3 medium."},
        {"family": "Qwen", "model_name": "Qwen/Qwen-Image", "display_name": "Qwen Image", "desc": "Dedicated image generation."},
        {"family": "Specialized", "model_name": "playgroundai/playground-v2.5-1024px-aesthetic", "display_name": "Playground v2.5", "desc": "High aesthetic 1024."},
        {"family": "Editors", "model_name": "Qwen/Qwen-Image-Edit", "display_name": "Qwen Image Edit", "desc": "Dedicated image editing."},
        {"family": "Editors", "model_name": "Qwen/Qwen-Image-Edit-2509", "display_name": "Qwen Image Edit Plus (Multi-Image)", "desc": "Advanced multi-image editing, fusion, and pose transfer."}
    ]

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName)
        self.manager: Optional[ModelManager] = None
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers not available. Please install required packages.")
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.model_name = self.config.get("model_name", "")
        
        models_path_str = kwargs.get("models_path", str(Path(__file__).parent / "models"))
        self.models_path = Path(models_path_str)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.registry = PipelineRegistry()
        self._resolve_device_and_dtype()
        if self.model_name:
            self._acquire_manager()

    def ps(self) -> List[dict]:
        if not self.registry:
            return []
        try:
            active = self.registry.get_active_managers()
            out = []
            for m in active:
                with m.lock:
                    cfg = m.config
                    pipe = m.pipeline
                    vram_usage_bytes = 0
                    if torch.cuda.is_available() and cfg.get("device") == "cuda" and pipe:
                        for comp in pipe.components.values():
                            if hasattr(comp, 'parameters'):
                                mem_params = sum(p.nelement() * p.element_size() for p in comp.parameters())
                                mem_bufs = sum(b.nelement() * b.element_size() for b in comp.buffers())
                                vram_usage_bytes += (mem_params + mem_bufs)
                    out.append({
                        "model_name": cfg.get("model_name"),
                        "vram_size": vram_usage_bytes,
                        "device": cfg.get("device"),
                        "torch_dtype": str(pipe.dtype) if pipe else cfg.get("torch_dtype_str"),
                        "pipeline_type": pipe.__class__.__name__ if pipe else "N/A",
                        "scheduler_class": pipe.scheduler.__class__.__name__ if pipe and hasattr(pipe, 'scheduler') else "N/A",
                        "status": "Active" if m.is_loaded else "Idle",
                        "queue_size": m.queue.qsize(),
                        "task": m.current_task or "N/A"
                    })
            return out
        except Exception as e:
            ASCIIColors.error(f"Failed to list running models: {e}")
            return []

    def _acquire_manager(self):
        if self.manager:
            self.registry.release_manager(self.manager.config)
        self.manager = self.registry.get_manager(self.config, self.models_path)
        ASCIIColors.info(f"Binding instance acquired manager for '{self.config['model_name']}'.")

    def _resolve_device_and_dtype(self):
        if self.config["device"].lower() == "auto":
            self.config["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.config["torch_dtype_str"].lower() == "auto":
            self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

    def _decode_image_input(self, item: str) -> Image.Image:
        s = item.strip()
        if s.startswith("data:image/") and ";base64," in s:
            b64 = s.split(";base64,")[-1]
            raw = base64.b64decode(b64)
            return Image.open(BytesIO(raw)).convert("RGB")
        if re_b64 := (s[:30].replace("\n","")):
            try:
                raw = base64.b64decode(s, validate=True)
                return Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                pass
        try:
            return load_image(s).convert("RGB")
        except Exception:
            return Image.open(s).convert("RGB")

    def _prepare_seed(self, kwargs: Dict[str, Any]) -> Optional[torch.Generator]:
        seed = kwargs.pop("seed", self.config["seed"])
        if seed == -1:
            return None
        return torch.Generator(device=self.config["device"]).manual_seed(seed)

    def list_safetensor_models(self) -> List[str]:
        if not self.models_path.exists():
            return []
        return sorted([f.name for f in self.models_path.iterdir() if f.is_file() and f.suffix == ".safetensors"])

    def list_models(self) -> list:
        civitai_list = [
            {'model_name': key, 'display_name': info['display_name'], 'description': info['description'], 'owned_by': info['owned_by']}
            for key, info in CIVITAI_MODELS.items()
        ]
        hf_list = [
            {'model_name': m["model_name"], 'display_name': m["display_name"], 'description': m["desc"], 'owned_by': 'HuggingFace', 'family': m["family"]}
            for m in self.HF_DEFAULT_MODELS
        ]
        custom_local = []
        civitai_filenames = {info['filename'] for info in CIVITAI_MODELS.values()}
        for filename in self.list_safetensor_models():
            if filename not in civitai_filenames:
                custom_local.append({'model_name': filename, 'display_name': filename, 'description': 'Local safetensors file.', 'owned_by': 'local_user'})
        return hf_list + civitai_list + custom_local

    def load_model(self):
        ASCIIColors.info("load_model() called. Loading is automatic on first use.")
        if self.model_name and not self.manager:
            self._acquire_manager()

    def unload_model(self):
        if self.manager:
            ASCIIColors.info(f"Binding instance releasing manager for '{self.manager.config['model_name']}'.")
            self.registry.release_manager(self.manager.config)
            self.manager = None

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int|None = None, height: int|None = None, **kwargs) -> bytes:
        if not self.model_name:
            raise RuntimeError("No model_name configured. Please select a model in settings.")
        if not self.manager:
            self._acquire_manager()
        generator = self._prepare_seed(kwargs)
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or self.config.get("negative_prompt", ""),
            "width": width if width is not None else self.config.get("width", 512),
            "height": height if height is not None else self.config.get("height", 512),
            "num_inference_steps": kwargs.pop("num_inference_steps", self.config.get("num_inference_steps",25)),
            "guidance_scale": kwargs.pop("guidance_scale", self.config.get("guidance_scale",6.5)),
            "generator": generator
        }
        pipeline_args.update(kwargs)
        future = Future()
        self.manager.queue.put((future, "text2image", pipeline_args))
        ASCIIColors.info(f"Job (t2i) '{prompt[:50]}...' queued.")
        try:
            return future.result()
        except Exception as e:
            raise Exception(f"Image generation failed: {e}") from e

    def _encode_image_to_latents(self, pil: Image.Image, width: int, height: int) -> Tuple[torch.Tensor, Tuple[int,int]]:
        pil = pil.convert("RGB").resize((width, height))
        with self.manager.lock:
            self.manager._load_pipeline_for_task("text2image")
            vae = self.manager.pipeline.vae
        img = torch.from_numpy(torch.ByteTensor(bytearray(pil.tobytes())).numpy()).float()  # not efficient but avoids np dep
        img = img.view(pil.height, pil.width, 3).permute(2,0,1).unsqueeze(0) / 255.0
        img = (img * 2.0) - 1.0
        img = img.to(self.config["device"], dtype=getattr(torch, self.config["torch_dtype_str"]))
        with torch.no_grad():
            posterior = vae.encode(img)
            latents = posterior.latent_dist.sample()
            sf = getattr(vae.config, "scaling_factor", 0.18215)
            latents = latents * sf
        return latents, (pil.width, pil.height)

    def edit_image(self,
                   images: Union[str, List[str]],
                   prompt: str,
                   negative_prompt: Optional[str] = "",
                   mask: Optional[str] = None,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   **kwargs) -> bytes:
        if not self.model_name:
            raise RuntimeError("No model_name configured. Please select a model in settings.")
        if not self.manager:
            self._acquire_manager()
        imgs = [images] if isinstance(images, str) else list(images)
        pil_images = [self._decode_image_input(s) for s in imgs]
        out_w = width if width is not None else self.config["width"]
        out_h = height if height is not None else self.config["height"]
        generator = self._prepare_seed(kwargs)
        steps = kwargs.pop("num_inference_steps", self.config["num_inference_steps"])
        guidance = kwargs.pop("guidance_scale", self.config["guidance_scale"])

        # Handle multi-image fusion for Qwen-Image-Edit-2509
        if "Qwen-Image-Edit-2509" in self.model_name and len(pil_images) > 1:
            pipeline_args = {
                "image": pil_images,
                "prompt": prompt,
                "negative_prompt": negative_prompt or " ",
                "width": out_w, "height": out_h,
                "num_inference_steps": steps,
                "true_cfg_scale": guidance,
                "generator": generator
            }
            pipeline_args.update(kwargs)
            future = Future()
            self.manager.queue.put((future, "image2image", pipeline_args))
            ASCIIColors.info(f"Job (multi-image fusion with {len(pil_images)} images) queued.")
            return future.result()

        # Handle inpainting (single image with mask)
        if mask is not None and len(pil_images) == 1:
            try:
                mask_img = self._decode_image_input(mask).convert("L")
            except Exception as e:
                raise ValueError(f"Failed to decode mask image: {e}") from e
            pipeline_args = {
                "image": pil_images[0],
                "mask_image": mask_img,
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "width": out_w, "height": out_h,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator
            }
            pipeline_args.update(kwargs)
            if "Qwen-Image-Edit" in self.model_name:
                pipeline_args["true_cfg_scale"] = pipeline_args.pop("guidance_scale", 7.0)
                if not pipeline_args.get("negative_prompt"): pipeline_args["negative_prompt"] = " "

            future = Future()
            self.manager.queue.put((future, "inpainting", pipeline_args))
            ASCIIColors.info("Job (inpaint) queued.")
            return future.result()

        # Handle standard image-to-image (single image)
        try:
            pipeline_args = {
                "image": pil_images[0],
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "strength": kwargs.pop("strength", 0.6),
                "width": out_w, "height": out_h,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator
            }
            pipeline_args.update(kwargs)
            if "Qwen-Image-Edit" in self.model_name:
                pipeline_args["true_cfg_scale"] = pipeline_args.pop("guidance_scale", 7.0)
                if not pipeline_args.get("negative_prompt"): pipeline_args["negative_prompt"] = " "

            future = Future()
            self.manager.queue.put((future, "image2image", pipeline_args))
            ASCIIColors.info("Job (i2i) queued.")
            return future.result()
        except Exception:
            pass

        # Fallback to latent-based generation if i2i fails for some reason
        try:
            base = pil_images[0]
            latents, _ = self._encode_image_to_latents(base, out_w, out_h)
            pipeline_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "latents": latents,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
                "width": out_w, "height": out_h
            }
            pipeline_args.update(kwargs)
            future = Future()
            self.manager.queue.put((future, "text2image", pipeline_args))
            ASCIIColors.info("Job (t2i with init latents) queued.")
            return future.result()
        except Exception as e:
            raise Exception(f"Image edit failed: {e}") from e


    def list_local_models(self) -> List[str]:
        if not self.models_path.exists():
            return []
        folders = [
            d.name for d in self.models_path.iterdir()
            if d.is_dir() and ((d / "model_index.json").exists() or (d / "unet" / "config.json").exists())
        ]
        safetensors = self.list_safetensor_models()
        return sorted(folders + safetensors)

    def list_available_models(self) -> List[str]:
        discoverable = [m['model_name'] for m in self.list_models()]
        local_models = self.list_local_models()
        return sorted(list(set(local_models + discoverable)))

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        models = self.list_available_models()
        local_models = self.list_local_models()
        if not models:
            return [{"name": "diffusers_no_models", "caption": "No models found", "help": f"Place models in '{self.models_path.resolve()}'."}]
        services = []
        for m in models:
            help_text = "Hugging Face model ID"
            if m in local_models:
                help_text = f"Local model from: {self.models_path.resolve()}"
            elif m in CIVITAI_MODELS:
                help_text = f"Civitai model (downloads as {CIVITAI_MODELS[m]['filename']})"
            services.append({"name": m, "caption": f"Diffusers: {m}", "help": help_text})
        return services

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        available_models = self.list_available_models()
        return [
            {"name": "model_name", "type": "str", "value": self.model_name, "description": "Local, Civitai, or Hugging Face model.", "options": available_models},
            {"name": "unload_inactive_model_after", "type": "int", "value": self.config["unload_inactive_model_after"], "description": "Unload model after X seconds of inactivity (0 to disable)."},
            {"name": "device", "type": "str", "value": self.config["device"], "description": f"Inference device. Resolved: {self.config['device']}", "options": ["auto","cuda","mps","cpu"]},
            {"name": "torch_dtype_str", "type": "str", "value": self.config["torch_dtype_str"], "description": f"Torch dtype. Resolved: {self.config['torch_dtype_str']}", "options": ["auto","float16","bfloat16","float32"]},
            {"name": "hf_variant", "type": "str", "value": self.config["hf_variant"], "description": "HF model variant (e.g., 'fp16')."},
            {"name": "use_safetensors", "type": "bool", "value": self.config["use_safetensors"], "description": "Prefer .safetensors when loading from Hugging Face."},
            {"name": "scheduler_name", "type": "str", "value": self.config["scheduler_name"], "description": "Scheduler for diffusion.", "options": list(SCHEDULER_MAPPING.keys())},
            {"name": "safety_checker_on", "type": "bool", "value": self.config["safety_checker_on"], "description": "Enable the safety checker."},
            {"name": "enable_cpu_offload", "type": "bool", "value": self.config["enable_cpu_offload"], "description": "Enable model CPU offload (saves VRAM, slower)."},
            {"name": "enable_sequential_cpu_offload", "type": "bool", "value": self.config["enable_sequential_cpu_offload"], "description": "Enable sequential CPU offload."},
            {"name": "enable_xformers", "type": "bool", "value": self.config["enable_xformers"], "description": "Enable xFormers memory efficient attention."},
            {"name": "width", "type": "int", "value": self.config["width"], "description": "Default image width."},
            {"name": "height", "type": "int", "value": self.config["height"], "description": "Default image height."},
            {"name": "num_inference_steps", "type": "int", "value": self.config["num_inference_steps"], "description": "Default inference steps."},
            {"name": "guidance_scale", "type": "float", "value": self.config["guidance_scale"], "description": "Default guidance scale (CFG)."},
            {"name": "seed", "type": "int", "value": self.config["seed"], "description": "Default seed (-1 for random)."},
            {"name": "hf_token", "type": "str", "value": self.config["hf_token"], "description": "HF API token (for private/gated models).", "is_secret": True},
            {"name": "hf_cache_path", "type": "str", "value": self.config["hf_cache_path"], "description": "Path to HF cache."},
            {"name": "local_files_only", "type": "bool", "value": self.config["local_files_only"], "description": "Do not download from Hugging Face."}
        ]

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        parsed = settings if isinstance(settings, dict) else {i["name"]: i["value"] for i in settings if "name" in i and "value" in i}
        critical_keys = self.registry._get_critical_keys()
        needs_swap = False
        for key, value in parsed.items():
            if self.config.get(key) != value:
                ASCIIColors.info(f"Setting '{key}' changed to: {value}")
                self.config[key] = value
                if key == "model_name":
                    self.model_name = value
                if key in critical_keys:
                    needs_swap = True
        if needs_swap and self.model_name:
            ASCIIColors.info("Critical settings changed. Swapping model manager...")
            self._resolve_device_and_dtype()
            self._acquire_manager()
        if not needs_swap and self.manager:
            self.manager.config.update(parsed)
            if 'scheduler_name' in parsed and self.manager.pipeline:
                with self.manager.lock:
                    self.manager._set_scheduler()
        return True

    def __del__(self):
        self.unload_model()

if __name__ == '__main__':
    ASCIIColors.magenta("--- Diffusers TTI Binding Test ---")
    if not DIFFUSERS_AVAILABLE:
        ASCIIColors.error("Diffusers not available. Cannot run test.")
        exit(1)
    temp_paths_dir = Path(__file__).parent / "tmp"
    temp_models_path = temp_paths_dir / "models"
    if temp_paths_dir.exists():
        shutil.rmtree(temp_paths_dir)
    temp_models_path.mkdir(parents=True, exist_ok=True)
    try:
        ASCIIColors.cyan("\n--- Test: Loading a small HF model ---")
        cfg = {"models_path": str(temp_models_path), "model_name": "hf-internal-testing/tiny-stable-diffusion-torch"}
        binding = DiffusersTTIBinding_Impl(**cfg)
        img_bytes = binding.generate_image("a tiny robot", width=64, height=64, num_inference_steps=2)
        assert len(img_bytes) > 1000
        ASCIIColors.green("HF t2i generation OK.")
        del binding
        time.sleep(0.1)
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Diffusers binding test failed: {e}")
    finally:
        ASCIIColors.cyan("\nCleaning up temporary directories...")
        if temp_paths_dir.exists():
            shutil.rmtree(temp_paths_dir)
        ASCIIColors.magenta("--- Diffusers TTI Binding Test Finished ---")
