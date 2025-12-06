# bindings/huggingface/__init__.py
import json
import os
import threading
import time
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any

import psutil 
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

# --- Pipmaster: Ensure dependencies ---
pm.ensure_packages([
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers", 
    "pillow",
    "scipy",
    "huggingface_hub",
    "psutil",
    "peft" # Added for LoRA
])

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TextIteratorStreamer,
        BitsAndBytesConfig, 
        AutoConfig, 
        AutoProcessor, 
        LlavaForConditionalGeneration, 
        LlavaNextForConditionalGeneration,
        StoppingCriteria
    )
    from peft import PeftModel # Logic for LoRA
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download, scan_cache_dir
    from PIL import Image
except ImportError as e:
    ASCIIColors.error(f"Failed to import core libraries: {e}")
    torch = None 
    transformers = None

# --- Helper Classes ---
class ModelContainer:
    """Holds a loaded model, its tokenizer, and metadata."""
    def __init__(self, model_id, model, tokenizer, processor=None, device="cpu", quant=None):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.quantization = quant
        self.last_used = time.time()
        self.supports_vision = processor is not None

    def update_usage(self):
        self.last_used = time.time()

BindingName = "HuggingFace"

class HuggingFace(LollmsLLMBinding):
    DEFAULT_CONFIG_ARGS = {
        "device": "auto", 
        "quantize": False, 
        "torch_dtype": "auto", 
        "max_new_tokens": 4096, 
        "temperature": 0.7,
        "trust_remote_code": False, 
        "use_flash_attention_2": False, 
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2", 
        "max_active_models": 1,
        # Path where we save merged models. 
        # By default, we try to use the binding's local path provided by Lollms
        "local_models_path": "models/huggingface_merged" 
    }

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        
        if torch is None or transformers is None:
            raise ImportError("Core libraries not available.")

        self.config = {**self.DEFAULT_CONFIG_ARGS, **kwargs.get("config", {}), **kwargs}
        
        # Determine local storage path
        # If lollms_paths is passed (standard in Lollms), use it
        lollms_paths = kwargs.get("lollms_paths")
        if lollms_paths:
            # Create a dedicated folder for merged models
            self.local_models_path = Path(lollms_paths.personal_models_path) / "huggingface"
        else:
            self.local_models_path = Path(self.config["local_models_path"])
        
        self.local_models_path.mkdir(parents=True, exist_ok=True)

        # Smart Management
        self.loaded_models: Dict[str, ModelContainer] = {} 
        self.active_model_id: Optional[str] = None
        self.inference_lock = threading.Lock()
        
        self.embedding_model = None
        self.load_embedding_model()

        model_name = kwargs.get("model_name")
        if model_name:
            self.load_model(model_name)

    def load_embedding_model(self):
        name = self.config.get("embedding_model_name")
        if name:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedding_model = SentenceTransformer(name, device=device)
            except Exception as e:
                ASCIIColors.warning(f"Failed to load embedding model: {e}")

    def _get_device_and_dtype(self):
        device_pref = self.config.get("device", "auto")
        if device_pref == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = device_pref
        
        dtype_pref = self.config.get("torch_dtype", "auto")
        if dtype_pref == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        elif dtype_pref == "bfloat16": dtype = torch.bfloat16
        else: dtype = torch.float32
        return device, dtype

    def _manage_memory(self):
        max_models = int(self.config.get("max_active_models", 1))
        while len(self.loaded_models) >= max_models:
            lru_id = min(self.loaded_models, key=lambda k: self.loaded_models[k].last_used)
            if lru_id == self.active_model_id and len(self.loaded_models) == 1:
                pass
            ASCIIColors.info(f"Smart Manager: Unloading {lru_id} to free memory.")
            self.unload_model_by_id(lru_id)

    def load_model(self, model_name_or_id: str) -> bool:
        # Check if it is a local path first
        possible_local_path = self.local_models_path / model_name_or_id
        if possible_local_path.exists():
            model_name_or_id = str(possible_local_path)
            ASCIIColors.info(f"Found local merged model: {model_name_or_id}")

        if model_name_or_id in self.loaded_models:
            self.active_model_id = model_name_or_id
            self.loaded_models[model_name_or_id].update_usage()
            ASCIIColors.success(f"Switched to loaded model: {model_name_or_id}")
            return True

        self._manage_memory()

        ASCIIColors.info(f"Loading {model_name_or_id}...")
        device, dtype = self._get_device_and_dtype()
        
        quant_mode = self.config.get("quantize", False)
        bnb_config = None
        load_in_8bit = str(quant_mode) == "8bit"
        load_in_4bit = str(quant_mode) == "4bit"

        if device == "cuda" and load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
            )

        model_args = {
            "trust_remote_code": self.config.get("trust_remote_code", False),
            "torch_dtype": dtype if not (load_in_8bit or load_in_4bit) else None,
        }
        
        if self.config.get("use_flash_attention_2") and device == "cuda":
            try:
                import flash_attn
                model_args["attn_implementation"] = "flash_attention_2"
            except: pass

        if load_in_8bit: model_args["load_in_8bit"] = True
        if load_in_4bit: model_args["quantization_config"] = bnb_config
        if device == "cuda" and (load_in_8bit or load_in_4bit or torch.cuda.device_count() > 1):
            model_args["device_map"] = "auto"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=model_args["trust_remote_code"])
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            config = AutoConfig.from_pretrained(model_name_or_id, trust_remote_code=model_args["trust_remote_code"])
            
            processor = None
            if "llava" in config.model_type or "Llava" in str(getattr(config, "architectures", [])):
                processor = AutoProcessor.from_pretrained(model_name_or_id, trust_remote_code=model_args["trust_remote_code"])
                ModelClass = LlavaNextForConditionalGeneration if "LlavaNext" in str(getattr(config, "architectures", [])) else LlavaForConditionalGeneration
                model = ModelClass.from_pretrained(model_name_or_id, **model_args)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_id, **model_args)

            if "device_map" not in model_args and device != "cpu":
                model.to(device)
            
            model.eval()
            
            container = ModelContainer(model_name_or_id, model, tokenizer, processor, device, quant_mode)
            self.loaded_models[model_name_or_id] = container
            self.active_model_id = model_name_or_id
            
            ASCIIColors.success(f"Model {model_name_or_id} loaded successfully.")
            return True

        except Exception as e:
            ASCIIColors.error(f"Failed to load model: {e}")
            trace_exception(e)
            return False

    def unload_model_by_id(self, model_id: str):
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    def get_container(self) -> Optional[ModelContainer]:
        if self.active_model_id and self.active_model_id in self.loaded_models:
            return self.loaded_models[self.active_model_id]
        return None

    def generate_text(self, prompt: str, images: List[str] = None, system_prompt: str = "", 
                      stream: bool = False, streaming_callback=None, split=False, 
                      n_predict=None, **kwargs) -> Union[str, Dict]:
        
        container = self.get_container()
        if not container:
            return {"status": False, "error": "No model loaded."}
        
        container.update_usage()

        with self.inference_lock:
            inputs = {}
            if container.supports_vision and images:
                pil_images = [Image.open(p).convert("RGB") for p in images]
                inputs = container.processor(text=prompt, images=pil_images, return_tensors="pt").to(container.model.device)
            else:
                if hasattr(container.tokenizer, 'apply_chat_template') and not split:
                    messages = []
                    if system_prompt: messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    try:
                        text = container.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = container.tokenizer(text, return_tensors="pt").to(container.model.device)
                    except:
                        inputs = container.tokenizer(prompt, return_tensors="pt").to(container.model.device)
                else:
                    inputs = container.tokenizer(prompt, return_tensors="pt").to(container.model.device)

            gen_kwargs = {
                "max_new_tokens": n_predict or self.config.get("max_new_tokens"),
                "temperature": kwargs.get("temperature", self.config.get("temperature")),
                "do_sample": kwargs.get("temperature", 0.7) > 0,
                "pad_token_id": container.tokenizer.eos_token_id
            }
            
            try:
                if stream and streaming_callback:
                    streamer = TextIteratorStreamer(container.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    gen_kwargs["streamer"] = streamer
                    
                    t = threading.Thread(target=container.model.generate, kwargs={**inputs, **gen_kwargs})
                    t.start()
                    
                    full_text = ""
                    for chunk in streamer:
                        full_text += chunk
                        if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                            break 
                    t.join()
                    return full_text
                else:
                    outputs = container.model.generate(**inputs, **gen_kwargs)
                    text = container.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    return text
            except Exception as e:
                trace_exception(e)
                return {"status": False, "error": str(e)}

    # --- Management Commands ---

    def list_models(self) -> List[Dict[str, str]]:
        """Lists models from HF Cache AND Local Merged Storage."""
        models = []
        
        # 1. Scan Local Merged Models
        if self.local_models_path.exists():
            for item in self.local_models_path.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    models.append({
                        "model_name": item.name,
                        "source": "Local Merged",
                        "size": "N/A",
                        "path": str(item)
                    })

        # 2. Scan HF Cache
        try:
            hf_cache_info = scan_cache_dir()
            for repo in hf_cache_info.repos:
                size_gb = repo.size_on_disk / (1024**3)
                models.append({
                    "model_name": repo.repo_id,
                    "source": "HF Cache",
                    "size": f"{size_gb:.2f} GB",
                    "path": str(repo.repo_path)
                })
        except Exception:
            pass 

        return models

    def pull_model(self, model_name: str) -> dict:
        try:
            ASCIIColors.info(f"Downloading {model_name}...")
            path = snapshot_download(repo_id=model_name)
            msg = f"Model {model_name} downloaded."
            ASCIIColors.success(msg)
            return {"status": True, "message": msg, "path": path}
        except Exception as e:
            return {"status": False, "message": str(e)}

    def merge_lora(self, base_model_name: str, lora_model_name: str, new_model_name: str) -> dict:
        """
        Loads a base model and a LoRA adapter, merges them, and saves to disk.
        """
        try:
            ASCIIColors.info(f"Starting merge: {base_model_name} + {lora_model_name} -> {new_model_name}")
            
            # Destination path
            save_path = self.local_models_path / new_model_name
            if save_path.exists():
                return {"status": False, "message": f"Model '{new_model_name}' already exists locally."}

            # 1. Load Base Model (Low CPU Memory usage to prevent OOM)
            # We usually merge on CPU or CUDA but strictly without quantization for correct saving
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ASCIIColors.info(f"Loading base model {base_model_name} on {device} (no quantization)...")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                return_dict=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True,
                trust_remote_code=self.config.get("trust_remote_code", False)
            )
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=self.config.get("trust_remote_code", False))

            # 2. Load LoRA
            ASCIIColors.info(f"Loading LoRA adapter {lora_model_name}...")
            model_to_merge = PeftModel.from_pretrained(base_model, lora_model_name)

            # 3. Merge
            ASCIIColors.info("Merging weights...")
            model_to_merge = model_to_merge.merge_and_unload()
            
            # 4. Save
            ASCIIColors.info(f"Saving to {save_path}...")
            model_to_merge.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Cleanup
            del model_to_merge
            del base_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            msg = f"Successfully created '{new_model_name}'. It is now available in your model list."
            ASCIIColors.success(msg)
            return {"status": True, "message": msg, "path": str(save_path)}

        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Merge failed: {str(e)}"}

    def ps(self) -> List[Dict]:
        status_list = []
        system_ram = psutil.virtual_memory()
        
        gpu_info = "N/A"
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_res = torch.cuda.memory_reserved() / (1024**3)
            gpu_info = f"{gpu_mem:.2f}GB / {gpu_res:.2f}GB"

        for mid, container in self.loaded_models.items():
            status_list.append({
                "model_name": mid,
                "active": mid == self.active_model_id,
                "device": container.device,
                "quantization": container.quantization,
                "system_ram_usage": f"{system_ram.percent}%",
                "gpu_memory": gpu_info,
            })
        
        return status_list or [{"model_name": "No models loaded.", "active": False}]