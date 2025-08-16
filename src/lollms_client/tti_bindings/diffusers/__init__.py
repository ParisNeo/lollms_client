# lollms_client/tti_bindings/diffusers/__init__.py
import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Attempt to import core dependencies and set availability flag
try:
    import torch
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    from diffusers.utils import load_image # Potentially for future img2img etc.
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoPipelineForText2Image = None
    DiffusionPipeline = None
    Image = None
    load_image = None
    DIFFUSERS_AVAILABLE = False
    # A detailed error will be raised in __init__ if the user tries to use the binding.

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import json # For potential JSONDecodeError and settings
import shutil

# Defines the binding name for the manager
BindingName = "DiffusersTTIBinding_Impl"

# Helper for torch.dtype string conversion, handles case where torch is not installed
TORCH_DTYPE_MAP_STR_TO_OBJ = {
    "float16": getattr(torch, 'float16', 'float16'),
    "bfloat16": getattr(torch, 'bfloat16', 'bfloat16'),
    "float32": getattr(torch, 'float32', 'float32'),
    "auto": "auto"
}
TORCH_DTYPE_MAP_OBJ_TO_STR = {v: k for k, v in TORCH_DTYPE_MAP_STR_TO_OBJ.items()}
if torch: # Add None mapping if torch is loaded
    TORCH_DTYPE_MAP_OBJ_TO_STR[None] = "None"


# Common Schedulers mapping (User-friendly name to Class name)
SCHEDULER_MAPPING = {
    "default": None,  # Use model's default
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
}
SCHEDULER_USES_KARRAS_SIGMAS = [
    "dpm_multistep_karras", "dpm++_2m_karras", "dpm++_2s_ancestral_karras",
    "dpm++_sde_karras", "heun_karras", "lms_karras"
]


class DiffusersTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for Hugging Face Diffusers library.
    Allows running various text-to-image models locally.
    """
    DEFAULT_CONFIG = {
        "model_name": "",
        "device": "auto",
        "torch_dtype_str": "auto",
        "use_safetensors": True,
        "scheduler_name": "default",
        "safety_checker_on": True,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "default_width": 768,
        "default_height": 768,
        "seed": -1,
        "enable_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "enable_xformers": False,
        "hf_variant": None,
        "hf_token": None,
        "hf_cache_path": None,
        "local_files_only": False,
    }

    def __init__(self, **kwargs):
        """
        Initialize the Diffusers TTI binding.

        Args:
            **kwargs: A dictionary of configuration parameters.
                Expected keys:
                - model_name (str): The name of the model to use. Can be a Hugging Face Hub ID
                  (e.g., 'stabilityai/stable-diffusion-xl-base-1.0') or the name of a local
                  model directory located in `models_path`.
                - models_path (str or Path): The path to the directory where local models are stored.
                  Defaults to a 'models' folder next to this file.
                - hf_cache_path (str or Path, optional): Path to a directory for Hugging Face
                  to cache downloaded models and files.
                - Other settings from the DEFAULT_CONFIG can be overridden here.
        """
        super().__init__(binding_name=BindingName)

        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library or its dependencies (torch, Pillow, transformers) are not installed. "
                "Please install them using: pip install torch diffusers Pillow transformers safetensors"
            )

        # Merge default config with user-provided kwargs
        self.config = {**self.DEFAULT_CONFIG, **kwargs}

        # model_name is crucial, get it from the merged config
        self.model_name = self.config.get("model_name", "")
        
        # models_path is also special, handle it with its default logic
        self.models_path = Path(kwargs.get("models_path", Path(__file__).parent / "models"))
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.pipeline: Optional[DiffusionPipeline] = None
        self.current_model_id_or_path = None

        self._resolve_device_and_dtype()
        
        if self.model_name:
            self.load_model()
        else:
            ASCIIColors.warning("No model_name provided during initialization. The binding is idle.")


    def _resolve_device_and_dtype(self):
        """Resolves auto settings for device and dtype from config."""
        if self.config["device"].lower() == "auto":
            if torch.cuda.is_available():
                self.config["device"] = "cuda"
            elif torch.backends.mps.is_available():
                self.config["device"] = "mps"
            else:
                self.config["device"] = "cpu"

        if self.config["torch_dtype_str"].lower() == "auto":
            self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

        self.torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower(), torch.float32)
        if self.torch_dtype == "auto": # Final fallback
            self.torch_dtype = torch.float16 if self.config["device"] != "cpu" else torch.float32
            self.config["torch_dtype_str"] = TORCH_DTYPE_MAP_OBJ_TO_STR.get(self.torch_dtype, "float32")

    def _resolve_model_path(self, model_name: str) -> str:
        """
        Resolves a model name to a full path if it's a local model,
        otherwise returns it as is (assuming it's a Hugging Face Hub ID).
        """
        if not model_name:
            raise ValueError("Model name cannot be empty.")
            
        if Path(model_name).is_absolute() and Path(model_name).is_dir():
            ASCIIColors.info(f"Using absolute path for model: {model_name}")
            return model_name
        
        local_model_path = self.models_path / model_name
        if local_model_path.exists() and local_model_path.is_dir():
            ASCIIColors.info(f"Found local model in '{self.models_path}': {local_model_path}")
            return str(local_model_path)
        
        ASCIIColors.info(f"'{model_name}' not found locally. Assuming it is a Hugging Face Hub ID.")
        return model_name

    def load_model(self):
        """Loads the Diffusers pipeline based on current configuration."""
        ASCIIColors.info("Loading Diffusers model...")
        if self.pipeline is not None:
            self.unload_model()

        try:
            model_path = self._resolve_model_path(self.model_name)
            self.current_model_id_or_path = model_path

            load_args = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": self.config["use_safetensors"],
                "token": self.config["hf_token"],
                "local_files_only": self.config["local_files_only"],
            }
            if self.config["hf_variant"]:
                load_args["variant"] = self.config["hf_variant"]
            
            if not self.config["safety_checker_on"]:
                load_args["safety_checker"] = None
            
            if self.config.get("hf_cache_path"):
                load_args["cache_dir"] = str(self.config["hf_cache_path"])

            self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **load_args)
            
            self._set_scheduler()

            self.pipeline.to(self.config["device"])

            if self.config["enable_xformers"]:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    ASCIIColors.info("xFormers memory efficient attention enabled.")
                except Exception as e:
                    ASCIIColors.warning(f"Could not enable xFormers: {e}. Proceeding without it.")
            
            if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_model_cpu_offload()
                ASCIIColors.info("Model CPU offload enabled.")
            elif self.config["enable_sequential_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_sequential_cpu_offload()
                ASCIIColors.info("Sequential CPU offload enabled.")

            ASCIIColors.green(f"Diffusers model '{model_path}' loaded on device '{self.config['device']}'.")

        except Exception as e:
            trace_exception(e)
            self.pipeline = None
            raise RuntimeError(f"Failed to load Diffusers model '{self.model_name}': {e}") from e

    def _set_scheduler(self):
        """Sets the scheduler for the pipeline based on config."""
        if not self.pipeline: return
        
        scheduler_name_key = self.config["scheduler_name"].lower()
        if scheduler_name_key == "default":
            ASCIIColors.info(f"Using model's default scheduler: {self.pipeline.scheduler.__class__.__name__}")
            return

        scheduler_class_name = SCHEDULER_MAPPING.get(scheduler_name_key)
        if scheduler_class_name:
            try:
                SchedulerClass = getattr(importlib.import_module("diffusers.schedulers"), scheduler_class_name)
                scheduler_config = self.pipeline.scheduler.config
                scheduler_config["use_karras_sigmas"] = scheduler_name_key in SCHEDULER_USES_KARRAS_SIGMAS
                self.pipeline.scheduler = SchedulerClass.from_config(scheduler_config)
                ASCIIColors.info(f"Switched scheduler to {scheduler_name_key} ({scheduler_class_name}).")
            except Exception as e:
                ASCIIColors.warning(f"Could not switch scheduler to {scheduler_name_key}: {e}. Using current default.")
        else:
            ASCIIColors.warning(f"Unknown scheduler: '{self.config['scheduler_name']}'. Using model default.")

    def unload_model(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            ASCIIColors.info("Diffusers pipeline unloaded.")

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int = None, height: int = None, **kwargs) -> bytes:
        """Generates an image using the loaded Diffusers pipeline."""
        if not self.pipeline:
            raise RuntimeError("Diffusers pipeline is not loaded. Cannot generate image.")

        _width = width or self.config["default_width"]
        _height = height or self.config["default_height"]
        _num_inference_steps = kwargs.get("num_inference_steps", self.config["num_inference_steps"])
        _guidance_scale = kwargs.get("guidance_scale", self.config["guidance_scale"])
        _seed = kwargs.get("seed", self.config["seed"])

        generator = torch.Generator(device=self.config["device"]).manual_seed(_seed) if _seed != -1 else None
        
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": _width,
            "height": _height,
            "num_inference_steps": _num_inference_steps,
            "guidance_scale": _guidance_scale,
            "generator": generator,
        }
        ASCIIColors.info(f"Generating image with prompt: '{prompt[:100]}...'")

        try:
            with torch.no_grad():
                 pipeline_output = self.pipeline(**pipeline_args)
            
            pil_image: Image.Image = pipeline_output.images[0]
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            
            ASCIIColors.green("Image generated successfully.")
            return img_byte_arr.getvalue()

        except Exception as e:
            trace_exception(e)
            raise Exception(f"Diffusers image generation failed: {e}") from e

    def list_models(self) -> List[str]:
        """Lists available local models from the models_path."""
        if not self.models_path.exists():
            return []
        
        models = []
        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir():
                # Check for key files indicating a valid diffusers model directory
                if (model_dir / "model_index.json").exists() or (model_dir / "unet" / "config.json").exists():
                    models.append(model_dir.name)
        return sorted(models)

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """Lists available local models from the models_path."""
        models = self.list_models()
        if not models:
            return [{
                "name": "diffusers_no_local_models", 
                "caption": "No local Diffusers models found", 
                "help": f"Place Diffusers model folders inside '{self.models_path.resolve()}' or specify a Hugging Face model ID in settings to download one."
            }]
        
        return [{
            "name": model_name,
            "caption": f"Diffusers: {model_name}",
            "help": f"Local Diffusers model from: {self.models_path.resolve()}"
        } for model_name in models]

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieves the current configurable settings for the binding."""
        local_models = self.list_models()
        return [
            {"name": "model_name", "type": "str", "value": self.model_name, "description": "Hugging Face model ID or a local model name from the models folder.", "options": local_models},
            {"name": "device", "type": "str", "value": self.config["device"], "description": f"Device for inference. Current resolved: {self.config['device']}", "options": ["auto", "cuda", "mps", "cpu"]},
            {"name": "torch_dtype_str", "type": "str", "value": self.config["torch_dtype_str"], "description": f"Torch dtype. Current resolved: {self.config['torch_dtype_str']}", "options": ["auto", "float16", "bfloat16", "float32"]},
            {"name": "hf_variant", "type": "str", "value": self.config["hf_variant"], "description": "Model variant from HF (e.g., 'fp16', 'bf16'). Optional."},
            {"name": "use_safetensors", "type": "bool", "value": self.config["use_safetensors"], "description": "Prefer loading models from .safetensors files."},
            {"name": "scheduler_name", "type": "str", "value": self.config["scheduler_name"], "description": "Scheduler for diffusion.", "options": list(SCHEDULER_MAPPING.keys())},
            {"name": "safety_checker_on", "type": "bool", "value": self.config["safety_checker_on"], "description": "Enable the safety checker (if model has one)."},
            {"name": "enable_cpu_offload", "type": "bool", "value": self.config["enable_cpu_offload"], "description": "Enable model CPU offload (saves VRAM, slower)."},
            {"name": "enable_sequential_cpu_offload", "type": "bool", "value": self.config["enable_sequential_cpu_offload"], "description": "Enable sequential CPU offload (more VRAM savings, much slower)."},
            {"name": "enable_xformers", "type": "bool", "value": self.config["enable_xformers"], "description": "Enable xFormers memory efficient attention."},
            {"name": "default_width", "type": "int", "value": self.config["default_width"], "description": "Default width for generated images."},
            {"name": "default_height", "type": "int", "value": self.config["default_height"], "description": "Default height for generated images."},
            {"name": "num_inference_steps", "type": "int", "value": self.config["num_inference_steps"], "description": "Default number of inference steps."},
            {"name": "guidance_scale", "type": "float", "value": self.config["guidance_scale"], "description": "Default guidance scale (CFG)."},
            {"name": "seed", "type": "int", "value": self.config["seed"], "description": "Default seed for generation (-1 for random)."},
            {"name": "hf_token", "type": "str", "value": self.config["hf_token"], "description": "Hugging Face API token (for private models).", "is_secret": True},
            {"name": "hf_cache_path", "type": "str", "value": self.config["hf_cache_path"], "description": "Path to Hugging Face cache. Defaults to ~/.cache/huggingface."},
            {"name": "local_files_only", "type": "bool", "value": self.config["local_files_only"], "description": "Only use local files, do not download."},
        ]

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        """Applies new settings to the binding. Some may trigger a model reload."""
        parsed_settings = settings if isinstance(settings, dict) else \
                          {item["name"]: item["value"] for item in settings if "name" in item and "value" in item}

        needs_reload = False
        critical_keys = ["model_name", "device", "torch_dtype_str", "use_safetensors", 
                         "safety_checker_on", "hf_variant", "enable_cpu_offload", 
                         "enable_sequential_cpu_offload", "enable_xformers", "hf_token", 
                         "local_files_only", "hf_cache_path"]

        for key, value in parsed_settings.items():
            current_value = getattr(self, key, self.config.get(key))
            if current_value != value:
                ASCIIColors.info(f"Setting '{key}' changed to: {value}")
                if key == "model_name":
                    self.model_name = value
                self.config[key] = value
                if key in critical_keys:
                    needs_reload = True
                elif key == "scheduler_name" and self.pipeline:
                    self._set_scheduler()

        if needs_reload and self.model_name:
            ASCIIColors.info("Reloading model due to settings changes...")
            try:
                self._resolve_device_and_dtype()
                self.load_model()
                ASCIIColors.green("Model reloaded successfully.")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"Failed to reload model with new settings: {e}. Binding may be unstable.")
                return False
        return True

    def __del__(self):
        self.unload_model()

# Example Usage (for testing within this file)
if __name__ == '__main__':
    ASCIIColors.magenta("--- Diffusers TTI Binding Test ---")
    
    if not DIFFUSERS_AVAILABLE:
        ASCIIColors.error("Diffusers or its dependencies are not available. Cannot run test.")
        exit(1)

    temp_paths_dir = Path(__file__).parent / "temp_lollms_paths_diffusers"
    temp_models_path = temp_paths_dir / "models"
    temp_cache_path = temp_paths_dir / "shared_cache"
    
    # Clean up previous runs
    if temp_paths_dir.exists():
        shutil.rmtree(temp_paths_dir)
    temp_models_path.mkdir(parents=True, exist_ok=True)
    temp_cache_path.mkdir(parents=True, exist_ok=True)
        
    # A very small, fast model for testing from Hugging Face.
    test_model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
    
    try:
        ASCIIColors.cyan("\n1. Initializing binding without a model...")
        binding = DiffusersTTIBinding_Impl(
            models_path=str(temp_models_path),
            hf_cache_path=str(temp_cache_path)
        )
        assert binding.pipeline is None, "Pipeline should not be loaded initially."
        ASCIIColors.green("Initialization successful.")

        ASCIIColors.cyan("\n2. Listing services (should be empty)...")
        services = binding.list_services()
        ASCIIColors.info(json.dumps(services, indent=2))
        assert services[0]["name"] == "diffusers_no_local_models"

        ASCIIColors.cyan(f"\n3. Setting model_name to '{test_model_id}' to trigger load...")
        binding.set_settings({"model_name": test_model_id})
        assert binding.model_name == test_model_id
        assert binding.pipeline is not None, "Pipeline should be loaded after setting model_name."
        ASCIIColors.green("Model loaded successfully.")

        ASCIIColors.cyan("\n4. Generating an image...")
        image_bytes = binding.generate_image(
            prompt="A tiny robot",
            width=64, height=64,
            num_inference_steps=2
        )
        assert image_bytes and isinstance(image_bytes, bytes)
        ASCIIColors.green(f"Image generated (size: {len(image_bytes)} bytes).")
        test_image_path = Path(__file__).parent / "test_diffusers_image.png"
        with open(test_image_path, "wb") as f:
            f.write(image_bytes)
        ASCIIColors.info(f"Test image saved to: {test_image_path.resolve()}")

        ASCIIColors.cyan("\n5. Unloading model...")
        binding.unload_model()
        assert binding.pipeline is None, "Pipeline should be None after unload."

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Diffusers binding test failed: {e}")
    finally:
        ASCIIColors.cyan("\nCleaning up temporary directories...")
        if temp_paths_dir.exists():
            shutil.rmtree(temp_paths_dir)
        ASCIIColors.magenta("--- Diffusers TTI Binding Test Finished ---")