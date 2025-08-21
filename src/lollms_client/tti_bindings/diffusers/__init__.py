# lollms_client/tti_bindings/diffusers/__init__.py
import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pipmaster as pm
# --- Concurrency Imports ---
import threading
import queue
from concurrent.futures import Future
import time
import hashlib
import re
# -------------------------
# --- Download Imports ---
import requests
from tqdm import tqdm
# --------------------

pm.ensure_packages(["torch","torchvision"],index_url="https://download.pytorch.org/whl/cu126")
pm.ensure_packages(["diffusers","pillow","transformers","safetensors", "requests", "tqdm"])

# Attempt to import core dependencies and set availability flag
try:
    import torch
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline, StableDiffusionPipeline
    from diffusers.utils import load_image
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoPipelineForText2Image = None
    DiffusionPipeline = None
    StableDiffusionPipeline = None
    Image = None
    load_image = None
    DIFFUSERS_AVAILABLE = False

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import json
import shutil

# Defines the binding name for the manager
BindingName = "DiffusersTTIBinding_Impl"

# --- START: Civitai Model Definitions ---
# Expanded list of popular Civitai models (as single .safetensors files)
CIVITAI_MODELS = {
    # --- Photorealistic ---
    "realistic-vision-v6": {
        "display_name": "Realistic Vision V6.0",
        "url": "https://civitai.com/api/download/models/130072",
        "filename": "realisticVisionV60_v60B1.safetensors",
        "description": "One of the most popular photorealistic models.",
        "owned_by": "civitai"
    },
    "absolute-reality": {
        "display_name": "Absolute Reality",
        "url": "https://civitai.com/api/download/models/132760",
        "filename": "absolutereality_v181.safetensors",
        "description": "A top-tier model for generating realistic images.",
        "owned_by": "civitai"
    },
    # --- General / Artistic ---
    "dreamshaper-8": {
        "display_name": "DreamShaper 8",
        "url": "https://civitai.com/api/download/models/128713",
        "filename": "dreamshaper_8.safetensors",
        "description": "A very popular and versatile general-purpose model.",
        "owned_by": "civitai"
    },
    "juggernaut-xl": {
        "display_name": "Juggernaut XL",
        "url": "https://civitai.com/api/download/models/133005",
        "filename": "juggernautXL_version6Rundiffusion.safetensors",
        "description": "High-quality artistic model, great for cinematic styles (SDXL-based).",
        "owned_by": "civitai"
    },
    "lyriel-v1.6": {
        "display_name": "Lyriel v1.6",
        "url": "https://civitai.com/api/download/models/92407",
        "filename": "lyriel_v16.safetensors",
        "description": "A popular artistic model for fantasy and stylized images.",
        "owned_by": "civitai"
    },
    # --- Anime / Illustration ---
    "anything-v5": {
        "display_name": "Anything V5",
        "url": "https://civitai.com/api/download/models/9409",
        "filename": "anythingV5_PrtRE.safetensors",
        "description": "A classic and highly popular model for anime-style generation.",
        "owned_by": "civitai"
    },
    "meinamix": {
        "display_name": "MeinaMix",
        "url": "https://civitai.com/api/download/models/119057",
        "filename": "meinamix_meinaV11.safetensors",
        "description": "A highly popular model for generating illustrative and vibrant anime-style images.",
        "owned_by": "civitai"
    },
    # --- Game Assets & Specialized Styles ---
    "rpg-v5": {
        "display_name": "RPG v5",
        "url": "https://civitai.com/api/download/models/137379",
        "filename": "rpg_v5.safetensors",
        "description": "Specialized in generating fantasy characters and assets in the style of classic RPGs.",
        "owned_by": "civitai"
    },
    "pixel-art-xl": {
        "display_name": "Pixel Art XL",
        "url": "https://civitai.com/api/download/models/252919",
        "filename": "pixelartxl_v11.safetensors",
        "description": "A dedicated SDXL model for generating high-quality pixel art sprites and scenes.",
        "owned_by": "civitai"
    },
    "lowpoly-world": {
        "display_name": "Lowpoly World",
        "url": "https://civitai.com/api/download/models/90299",
        "filename": "lowpoly_world_v10.safetensors",
        "description": "Generates assets and scenes with a stylized low-polygon, 3D render aesthetic.",
        "owned_by": "civitai"
    },
    "toonyou": {
        "display_name": "ToonYou",
        "url": "https://civitai.com/api/download/models/152361",
        "filename": "toonyou_beta6.safetensors",
        "description": "Excellent for creating expressive, high-quality cartoon and Disney-style characters.",
        "owned_by": "civitai"
    },
    "papercut": {
        "display_name": "Papercut",
        "url": "https://civitai.com/api/download/models/45579",
        "filename": "papercut_v1.safetensors",
        "description": "Creates unique images with a distinct paper cutout and layered diorama style.",
        "owned_by": "civitai"
    }
}
# --- END: Civitai Model Definitions ---

# Helper for torch.dtype string conversion
TORCH_DTYPE_MAP_STR_TO_OBJ = {
    "float16": getattr(torch, 'float16', 'float16'),
    "bfloat16": getattr(torch, 'bfloat16', 'bfloat16'),
    "float32": getattr(torch, 'float32', 'float32'),
    "auto": "auto"
}
TORCH_DTYPE_MAP_OBJ_TO_STR = {v: k for k, v in TORCH_DTYPE_MAP_STR_TO_OBJ.items()}
if torch:
    TORCH_DTYPE_MAP_OBJ_TO_STR[None] = "None"

# Common Schedulers mapping
SCHEDULER_MAPPING = {
    "default": None,
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "deis_multistep": "DEISMultistepScheduler",
    "dpm_multistep": "DPMSolverMultistepScheduler",
    "dpm_multistep_karras": "DPMSolverMultistepScheduler",
    "dpm_single": "DPMSolverSinglestepScheduler",
    "dpm_adaptive": "DPMSolverPlusPlusScheduler",  # Retained; no direct Diffusers equivalent confirmed, may require custom config
    "dpm++_2m": "DPMSolverMultistepScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "dpm++_2s_ancestral": "DPMSolverAncestralDiscreteScheduler",  # Retained; consider "KDPM2AncestralDiscreteScheduler" as alternative if class unavailable
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
    # Additions
    "dpm++_2m_sde": "DPMSolverMultistepScheduler",
    "dpm++_2m_sde_karras": "DPMSolverMultistepScheduler",
    "dpm2": "KDPM2DiscreteScheduler",
    "dpm2_karras": "KDPM2DiscreteScheduler",
    "dpm2_a": "KDPM2AncestralDiscreteScheduler",
    "dpm2_a_karras": "KDPM2AncestralDiscreteScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "lms": "LMSDiscreteScheduler",
}
SCHEDULER_USES_KARRAS_SIGMAS = [
    "dpm_multistep_karras", "dpm++_2m_karras", "dpm++_2s_ancestral_karras",
    "dpm++_sde_karras", "heun_karras", "lms_karras",
    # Additions
    "dpm++_2m_sde_karras", "dpm2_karras", "dpm2_a_karras",
]

# --- START: Concurrency and Singleton Management ---

class ModelManager:
    """
    Manages a single pipeline instance, its generation queue, a worker thread,
    and an optional auto-unload timer.
    """
    def __init__(self, config: Dict[str, Any], models_path: Path):
        self.config = config
        self.models_path = models_path
        self.pipeline: Optional[DiffusionPipeline] = None
        self.ref_count = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.is_loaded = False
        self.last_used_time = time.time()
        
        # --- Worker and Monitor Threads ---
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
        self.queue.put(None) # Sentinel to unblock queue.get()
        self.worker_thread.join(timeout=5)

    def _start_unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after > 0 and self._unload_monitor_thread is None:
            self._stop_monitor_event.clear()
            self._unload_monitor_thread = threading.Thread(target=self._unload_monitor, daemon=True)
            self._unload_monitor_thread.start()

    def _unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after <= 0: return

        ASCIIColors.info(f"Starting inactivity monitor for '{self.config['model_name']}' (timeout: {unload_after}s).")
        while not self._stop_monitor_event.wait(timeout=5.0): # Check every 5 seconds
            with self.lock:
                if not self.is_loaded:
                    continue
                
                if time.time() - self.last_used_time > unload_after:
                    ASCIIColors.info(f"Model '{self.config['model_name']}' has been inactive. Unloading.")
                    self._unload_pipeline()

    def _load_pipeline(self):
        # This method assumes a lock is already held
        if self.pipeline:
            return

        model_name = self.config.get("model_name", "")
        if not model_name:
            raise ValueError("Model name cannot be empty for loading.")
            
        ASCIIColors.info(f"Loading Diffusers model: {model_name}")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        
        try:
            if str(model_path).endswith(".safetensors"):
                ASCIIColors.info(f"Loading from single safetensors file: {model_path}")
                try:
                    # Modern, preferred method for newer diffusers versions
                    self.pipeline = AutoPipelineForText2Image.from_single_file(
                        model_path,
                        torch_dtype=torch_dtype,
                        cache_dir=self.config.get("hf_cache_path")
                    )
                except AttributeError:
                    # Fallback for older diffusers versions
                    ASCIIColors.warning("AutoPipelineForText2Image.from_single_file not found. Falling back to StableDiffusionPipeline.")
                    ASCIIColors.warning("Consider updating diffusers for better compatibility: pip install --upgrade diffusers")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch_dtype,
                        cache_dir=self.config.get("hf_cache_path")
                    )
            else:
                ASCIIColors.info(f"Loading from pretrained folder/repo: {model_path}")
                load_args = {
                    "torch_dtype": torch_dtype, "use_safetensors": self.config["use_safetensors"],
                    "token": self.config["hf_token"], "local_files_only": self.config["local_files_only"],
                }
                if self.config["hf_variant"]: load_args["variant"] = self.config["hf_variant"]
                if not self.config["safety_checker_on"]: load_args["safety_checker"] = None
                if self.config.get("hf_cache_path"): load_args["cache_dir"] = str(self.config["hf_cache_path"])
                self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **load_args)

        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "gated" in error_str or "authorization" in error_str:
                auth_error_msg = (
                    f"AUTHENTICATION FAILED for model '{model_name}'. This is likely a 'gated' model on Hugging Face.\n"
                    "Please ensure you have accepted its license and provided a valid HF Access Token in the settings."
                )
                raise RuntimeError(auth_error_msg) from e
            else:
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
        self.last_used_time = time.time()
        ASCIIColors.green(f"Model '{model_name}' loaded successfully on '{self.config['device']}'.")

    def _unload_pipeline(self):
        # This method assumes a lock is already held
        if self.pipeline:
            model_name = self.config.get('model_name', 'Unknown')
            del self.pipeline
            self.pipeline = None
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            ASCIIColors.info(f"Model '{model_name}' unloaded and VRAM cleared.")

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None:
                    break
                future, pipeline_args = job
                try:
                    with self.lock:
                        self.last_used_time = time.time()
                        if not self.is_loaded:
                            self._load_pipeline()
                    
                    with torch.no_grad():
                        pipeline_output = self.pipeline(**pipeline_args)
                    pil_image: Image.Image = pipeline_output.images[0]
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format="PNG")
                    future.set_result(img_byte_arr.getvalue())
                except Exception as e:
                    trace_exception(e)
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

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
                with open(temp_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {filename}"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            
            shutil.move(temp_path, dest_path)
            ASCIIColors.green(f"Model '{filename}' downloaded successfully.")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to download model {filename}: {e}") from e

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

    def _set_scheduler(self):
        if not self.pipeline: return
        scheduler_name_key = self.config["scheduler_name"].lower()
        if scheduler_name_key == "default": return

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
            "model_name", "device", "torch_dtype_str", "use_safetensors", 
            "safety_checker_on", "hf_variant", "enable_cpu_offload", 
            "enable_sequential_cpu_offload", "enable_xformers",
            "local_files_only", "hf_cache_path", "unload_inactive_model_after"
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
        "model_name": "", "device": "auto", "torch_dtype_str": "auto", "use_safetensors": True,
        "scheduler_name": "default", "safety_checker_on": True, "num_inference_steps": 25,
        "guidance_scale": 7.0, "default_width": 512, "default_height": 512, "seed": -1,
        "enable_cpu_offload": False, "enable_sequential_cpu_offload": False, "enable_xformers": False,
        "hf_variant": None, "hf_token": None, "hf_cache_path": None, "local_files_only": False,
        "unload_inactive_model_after": 0,
    }

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName)

        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers or its dependencies not installed. "
                "Please run: pip install torch torchvision diffusers Pillow transformers safetensors requests tqdm"
            )

        # Initialize config with defaults, then override with user kwargs
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        self.model_name = self.config.get("model_name", "")
        models_path_str = kwargs.get("models_path", str(Path(__file__).parent / "models"))
        self.models_path = Path(models_path_str)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.registry = PipelineRegistry()
        self.manager: Optional[ModelManager] = None
        
        self._resolve_device_and_dtype()
        if self.model_name:
            self._acquire_manager()

    def ps(self) -> List[dict]:
        """
        Lists running models in a standardized, flat format.
        """
        if not self.registry:
            ASCIIColors.warning("Diffusers PipelineRegistry not available.")
            return []

        try:
            active_managers = self.registry.get_active_managers()
            standardized_models = []

            for manager in active_managers:
                with manager.lock:
                    config = manager.config
                    pipeline = manager.pipeline
                    
                    vram_usage_bytes = 0
                    if torch.cuda.is_available() and config.get("device") == "cuda" and pipeline:
                        for component in pipeline.components.values():
                            if hasattr(component, 'parameters'):
                                mem_params = sum(p.nelement() * p.element_size() for p in component.parameters())
                                mem_bufs = sum(b.nelement() * b.element_size() for b in component.buffers())
                                vram_usage_bytes += (mem_params + mem_bufs)

                    flat_model_info = {
                        "model_name": config.get("model_name"),
                        "vram_size": vram_usage_bytes,
                        "device": config.get("device"),
                        "torch_dtype": str(pipeline.dtype) if pipeline else config.get("torch_dtype_str"),
                        "pipeline_type": pipeline.__class__.__name__ if pipeline else "N/A",
                        "scheduler_class": pipeline.scheduler.__class__.__name__ if pipeline and hasattr(pipeline, 'scheduler') else "N/A",
                        "status": "Active" if manager.is_loaded else "Idle",
                        "queue_size": manager.queue.qsize(),
                    }
                    standardized_models.append(flat_model_info)
            
            return standardized_models

        except Exception as e:
            ASCIIColors.error(f"Failed to list running models from Diffusers registry: {e}")
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

    def list_safetensor_models(self) -> List[str]:
        if not self.models_path.exists(): return []
        return sorted([f.name for f in self.models_path.iterdir() if f.is_file() and f.suffix == ".safetensors"])

    def listModels(self) -> list:
        # Implementation is unchanged...
        civitai_list = [
            {'model_name': key, 'display_name': info['display_name'], 'description': info['description'], 'owned_by': info['owned_by']}
            for key, info in CIVITAI_MODELS.items()
        ]
        hf_default_list = [
            {'model_name': "stabilityai/stable-diffusion-xl-base-1.0", 'display_name': "Stable Diffusion XL 1.0", 'description': "Official SDXL base model from Stability AI. Native resolution is 1024x1024.", 'owned_by': 'HuggingFace'},
            {'model_name': "playgroundai/playground-v2.5-1024px-aesthetic", 'display_name': "Playground v2.5", 'description': "Known for high aesthetic quality. Native resolution is 1024x1024.", 'owned_by': 'HuggingFace'},
            {'model_name': "runwayml/stable-diffusion-v1-5", 'display_name': "Stable Diffusion 1.5", 'description': "A popular and versatile open-access text-to-image model.", 'owned_by': 'HuggingFace'},
        ]
        custom_local_models = []
        civitai_filenames = {info['filename'] for info in CIVITAI_MODELS.values()}
        local_safetensors = self.list_safetensor_models()
        for filename in local_safetensors:
            if filename not in civitai_filenames:
                custom_local_models.append({
                    'model_name': filename, 'display_name': filename,
                    'description': 'Local safetensors file from your models folder.', 'owned_by': 'local_user'
                })
        return civitai_list + hf_default_list + custom_local_models

    def load_model(self):
        ASCIIColors.info("load_model() called. Loading is now automatic on first use.")
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

        # Build pipeline arguments, prioritizing kwargs over config defaults
        seed = kwargs.pop("seed", self.config["seed"])
        generator = torch.Generator(device=self.config["device"]).manual_seed(seed) if seed != -1 else None

        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": width if width is not None else self.config["default_width"],
            "height": height if height is not None else self.config["default_height"],
            "num_inference_steps": self.config["num_inference_steps"],
            "guidance_scale": self.config["guidance_scale"],
            "generator": generator,
        }
        # Allow any other valid pipeline kwargs to be passed through
        pipeline_args.update(kwargs)
        
        future = Future()
        self.manager.queue.put((future, pipeline_args))
        ASCIIColors.info(f"Job for prompt '{prompt[:50]}...' queued. Waiting...")
        
        try:
            image_bytes = future.result()
            ASCIIColors.green("Image generated successfully.")
            return image_bytes
        except Exception as e:
            raise Exception(f"Image generation failed: {e}") from e

    def list_local_models(self) -> List[str]:
        # Implementation is unchanged...
        if not self.models_path.exists(): return []
        folders = [
            d.name for d in self.models_path.iterdir()
            if d.is_dir() and ((d / "model_index.json").exists() or (d / "unet" / "config.json").exists())
        ]
        safetensors = self.list_safetensor_models()
        return sorted(folders + safetensors)
    
    def list_available_models(self) -> List[str]:
        # Implementation is unchanged...
        discoverable_models = [m['model_name'] for m in self.listModels()]
        local_models = self.list_local_models()
        return sorted(list(set(local_models + discoverable_models)))

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        # Implementation is unchanged...
        models = self.list_available_models()
        local_models = self.list_local_models()
        if not models:
            return [{"name": "diffusers_no_models", "caption": "No models found", "help": f"Place models in '{self.models_path.resolve()}'."}]
        services = []
        for m in models:
            help_text = "Hugging Face model ID"
            if m in local_models: help_text = f"Local model from: {self.models_path.resolve()}"
            elif m in CIVITAI_MODELS: help_text = f"Civitai model (downloads as {CIVITAI_MODELS[m]['filename']})"
            services.append({"name": m, "caption": f"Diffusers: {m}", "help": help_text})
        return services

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        available_models = self.list_available_models()
        return [
            {"name": "model_name", "type": "str", "value": self.model_name, "description": "Local, Civitai, or Hugging Face model.", "options": available_models},
            {"name": "unload_inactive_model_after", "type": "int", "value": self.config["unload_inactive_model_after"], "description": "Unload model after X seconds of inactivity (0 to disable)."},
            {"name": "device", "type": "str", "value": self.config["device"], "description": f"Inference device. Resolved: {self.config['device']}", "options": ["auto", "cuda", "mps", "cpu"]},
            {"name": "torch_dtype_str", "type": "str", "value": self.config["torch_dtype_str"], "description": f"Torch dtype. Resolved: {self.config['torch_dtype_str']}", "options": ["auto", "float16", "bfloat16", "float32"]},
            {"name": "hf_variant", "type": "str", "value": self.config["hf_variant"], "description": "HF model variant (e.g., 'fp16')."},
            {"name": "use_safetensors", "type": "bool", "value": self.config["use_safetensors"], "description": "Prefer .safetensors when loading from Hugging Face."},
            {"name": "scheduler_name", "type": "str", "value": self.config["scheduler_name"], "description": "Scheduler for diffusion.", "options": list(SCHEDULER_MAPPING.keys())},
            {"name": "safety_checker_on", "type": "bool", "value": self.config["safety_checker_on"], "description": "Enable the safety checker."},
            {"name": "enable_cpu_offload", "type": "bool", "value": self.config["enable_cpu_offload"], "description": "Enable model CPU offload (saves VRAM, slower)."},
            {"name": "enable_sequential_cpu_offload", "type": "bool", "value": self.config["enable_sequential_cpu_offload"], "description": "Enable sequential CPU offload (more VRAM savings, much slower)."},
            {"name": "enable_xformers", "type": "bool", "value": self.config["enable_xformers"], "description": "Enable xFormers memory efficient attention."},
            {"name": "default_width", "type": "int", "value": self.config["default_width"], "description": "Default image width. Note: SDXL models prefer 1024."},
            {"name": "default_height", "type": "int", "value": self.config["default_height"], "description": "Default image height. Note: SDXL models prefer 1024."},
            {"name": "num_inference_steps", "type": "int", "value": self.config["num_inference_steps"], "description": "Default inference steps."},
            {"name": "guidance_scale", "type": "float", "value": self.config["guidance_scale"], "description": "Default guidance scale (CFG)."},
            {"name": "seed", "type": "int", "value": self.config["seed"], "description": "Default seed (-1 for random)."},
            {"name": "hf_token", "type": "str", "value": self.config["hf_token"], "description": "HF API token (for private/gated models).", "is_secret": True},
            {"name": "hf_cache_path", "type": "str", "value": self.config["hf_cache_path"], "description": "Path to HF cache."},
            {"name": "local_files_only", "type": "bool", "value": self.config["local_files_only"], "description": "Do not download from Hugging Face."},
        ]

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        parsed_settings = settings if isinstance(settings, dict) else \
                          {item["name"]: item["value"] for item in settings if "name" in item and "value" in item}

        critical_keys = self.registry._get_critical_keys()
        needs_manager_swap = False

        for key, value in parsed_settings.items():
            if self.config.get(key) != value:
                ASCIIColors.info(f"Setting '{key}' changed to: {value}")
                self.config[key] = value
                if key == "model_name": self.model_name = value
                if key in critical_keys: needs_manager_swap = True
                
        if needs_manager_swap and self.model_name:
            ASCIIColors.info("Critical settings changed. Swapping model manager...")
            self._resolve_device_and_dtype()
            self._acquire_manager()
        
        if not needs_manager_swap and self.manager:
            # Update non-critical settings on the existing manager
            self.manager.config.update(parsed_settings)
            if 'scheduler_name' in parsed_settings and self.manager.pipeline:
                 with self.manager.lock:
                    self.manager._set_scheduler()

        return True

    def __del__(self):
        self.unload_model()

# Example Usage
if __name__ == '__main__':
    ASCIIColors.magenta("--- Diffusers TTI Binding Test ---")
    
    if not DIFFUSERS_AVAILABLE:
        ASCIIColors.error("Diffusers not available. Cannot run test.")
        exit(1)

    temp_paths_dir = Path(__file__).parent / "temp_lollms_paths_diffusers"
    temp_models_path = temp_paths_dir / "models"
    
    if temp_paths_dir.exists(): shutil.rmtree(temp_paths_dir)
    temp_models_path.mkdir(parents=True, exist_ok=True)
        
    try:
        ASCIIColors.cyan("\n--- Test: Loading a Hugging Face model ---")
        # Using a very small model for fast testing
        binding_config = {"models_path": str(temp_models_path), "model_name": "hf-internal-testing/tiny-stable-diffusion-torch"}
        binding = DiffusersTTIBinding_Impl(**binding_config)
        
        img_bytes = binding.generate_image("a tiny robot", width=64, height=64, num_inference_steps=2)
        assert len(img_bytes) > 1000, "Image generation from HF model should succeed."
        ASCIIColors.green("HF model loading and generation successful.")

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