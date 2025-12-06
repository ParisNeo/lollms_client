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

# --- Dependencies ---
pm.ensure_packages([
    "torch", "transformers", "accelerate", "bitsandbytes",
    "sentence_transformers", "pillow", "scipy", "huggingface_hub",
    "psutil", "peft", "trl", "datasets"
])

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer,
        BitsAndBytesConfig, AutoConfig, AutoProcessor, 
        LlavaForConditionalGeneration, LlavaNextForConditionalGeneration,
        TrainingArguments
    )
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    from PIL import Image
except ImportError as e:
    ASCIIColors.error(f"Failed to import core libraries: {e}")
    torch = None 
    transformers = None

# --- Container ---
class ModelContainer:
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
        "local_models_path": "" # If empty, dynamic default is used
    }

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        
        if torch is None or transformers is None:
            raise ImportError("Core libraries not available.")

        self.config = {**self.DEFAULT_CONFIG_ARGS, **kwargs.get("config", {}), **kwargs}
        
        # --- 1. Setup Local Models Path ---
        # Priority: Config Override -> Lollms Personal Path -> Default relative path
        if self.config["local_models_path"]:
             self.local_models_path = Path(self.config["local_models_path"])
        elif kwargs.get("lollms_paths"):
            self.local_models_path = Path(kwargs["lollms_paths"].personal_models_path) / "huggingface"
        else:
            self.local_models_path = Path("models/huggingface")
        
        self.local_models_path.mkdir(parents=True, exist_ok=True)
        ASCIIColors.info(f"HuggingFace Local Storage: {self.local_models_path}")

        # State
        self.loaded_models: Dict[str, ModelContainer] = {} 
        self.active_model_id: Optional[str] = None
        self.inference_lock = threading.Lock()
        self.is_training = False
        
        # Load Embeddings
        self.embedding_model = None
        self.load_embedding_model()

        # Initial Load
        model_name = kwargs.get("model_name")
        if model_name:
            self.load_model(model_name)

    def load_embedding_model(self):
        name = self.config.get("embedding_model_name")
        if name:
            try:
                ASCIIColors.info(f"Loading embedding model: {name}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedding_model = SentenceTransformer(name, device=device)
            except Exception as e:
                ASCIIColors.warning(f"Failed to load embedding model: {e}")

    def _manage_memory(self):
        max_models = int(self.config.get("max_active_models", 1))
        while len(self.loaded_models) >= max_models:
            lru_id = min(self.loaded_models, key=lambda k: self.loaded_models[k].last_used)
            # Avoid unloading the active one if possible, unless it's the only one and we need a swap
            if lru_id == self.active_model_id and len(self.loaded_models) == 1:
                pass 
            ASCIIColors.info(f"Unloading {lru_id} to free memory.")
            self.unload_model_by_id(lru_id)

    def load_model(self, model_name_or_id: str) -> bool:
        """
        Loads a model. Priorities:
        1. Local folder (self.local_models_path / model_name_or_id)
        2. Hugging Face Hub (download/cache automatically)
        """
        # --- Resolve Path ---
        # Clean naming for folder lookup
        folder_name = model_name_or_id.replace("/", "_") # Sanitize potential subdirs if user types "meta-llama/Llama-2"
        
        # Check standard path mapping
        possible_paths = [
            self.local_models_path / model_name_or_id, # Exact match (subfolders)
            self.local_models_path / folder_name,      # Flattened match
            Path(model_name_or_id)                     # Absolute path provided by user
        ]
        
        model_path_to_use = model_name_or_id # Default to ID for HF Hub
        
        for p in possible_paths:
            if p.exists() and p.is_dir() and (p / "config.json").exists():
                ASCIIColors.info(f"Found local model at: {p}")
                model_path_to_use = str(p)
                break
        
        # Check if already loaded
        if model_name_or_id in self.loaded_models:
            self.active_model_id = model_name_or_id
            self.loaded_models[model_name_or_id].update_usage()
            return True

        self._manage_memory()
        if self.is_training:
            ASCIIColors.error("Training in progress. Cannot load new model.")
            return False

        ASCIIColors.info(f"Loading {model_name_or_id} (Path/ID: {model_path_to_use})...")
        
        # --- Config & Device ---
        device = "cuda" if torch.cuda.is_available() and self.config["device"]=="auto" else "cpu"
        if self.config["device"] != "auto": device = self.config["device"]

        dtype_map = {"auto": torch.float16 if device=="cuda" else torch.float32, 
                     "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.config["torch_dtype"], torch.float32)

        quant_mode = self.config.get("quantize", False)
        load_in_4bit = str(quant_mode) == "4bit"
        load_in_8bit = str(quant_mode) == "8bit"
        
        bnb_config = None
        if device == "cuda" and load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

        model_args = {
            "trust_remote_code": self.config.get("trust_remote_code", False),
            "torch_dtype": dtype if not (load_in_4bit or load_in_8bit) else None,
            "device_map": "auto" if device == "cuda" else None
        }

        if self.config.get("use_flash_attention_2") and device == "cuda":
            try:
                import flash_attn
                model_args["attn_implementation"] = "flash_attention_2"
            except ImportError:
                ASCIIColors.warning("Flash Attention 2 enabled but not installed.")

        if load_in_4bit: model_args["quantization_config"] = bnb_config
        if load_in_8bit: model_args["load_in_8bit"] = True

        try:
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path_to_use, trust_remote_code=model_args["trust_remote_code"])
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

            # Architecture Detection
            config = AutoConfig.from_pretrained(model_path_to_use, trust_remote_code=model_args["trust_remote_code"])
            processor = None
            
            # LLaVA Check
            if "llava" in config.model_type.lower() or "Llava" in str(getattr(config, "architectures", [])):
                processor = AutoProcessor.from_pretrained(model_path_to_use, trust_remote_code=model_args["trust_remote_code"])
                ModelClass = LlavaNextForConditionalGeneration if "next" in config.model_type.lower() else LlavaForConditionalGeneration
                model = ModelClass.from_pretrained(model_path_to_use, **model_args)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path_to_use, **model_args)

            # Fallback for device placement
            if not model_args.get("device_map") and device != "cpu" and not (load_in_4bit or load_in_8bit):
                model.to(device)
            
            model.eval()
            
            container = ModelContainer(model_name_or_id, model, tokenizer, processor, device, quant_mode)
            self.loaded_models[model_name_or_id] = container
            self.active_model_id = model_name_or_id
            ASCIIColors.success(f"Loaded {model_name_or_id}")
            return True
            
        except Exception as e:
            ASCIIColors.error(f"Load failed: {e}")
            trace_exception(e)
            return False

    def unload_model_by_id(self, model_id: str):
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()

    def get_container(self):
        return self.loaded_models.get(self.active_model_id)

    # --- Generation ---
    def generate_text(self, prompt, images=None, system_prompt="", stream=False, streaming_callback=None, split=False, n_predict=None, **kwargs):
        if self.is_training: return {"status": False, "error": "Training in progress."}
        
        container = self.get_container()
        if not container: return {"status": False, "error": "No model loaded."}
        
        container.update_usage()

        with self.inference_lock:
            inputs = {}
            # Vision
            if container.supports_vision and images:
                pil_images = [Image.open(p).convert("RGB") for p in images]
                inputs = container.processor(text=prompt, images=pil_images, return_tensors="pt").to(container.model.device)
            # Text / Chat
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
                    
                    full = ""
                    for chunk in streamer:
                        full += chunk
                        if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK): break
                    t.join()
                    return full
                else:
                    outputs = container.model.generate(**inputs, **gen_kwargs)
                    text = container.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    return text
            except Exception as e:
                trace_exception(e)
                return {"status": False, "error": str(e)}

    # --- Commands ---

    def list_models(self) -> List[Dict[str, str]]:
        """Scans the designated local_models_path."""
        models = []
        if self.local_models_path.exists():
            for item in self.local_models_path.iterdir():
                if item.is_dir():
                    # Simple heuristic to check if it's a valid HF model folder
                    if (item / "config.json").exists() or (item / "adapter_config.json").exists():
                        try:
                            size_gb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**3)
                        except: size_gb = 0
                        
                        models.append({
                            "model_name": item.name,
                            "path": str(item),
                            "size": f"{size_gb:.2f} GB",
                            "source": "Local Storage"
                        })
        return models

    def pull_model(self, model_name: str) -> dict:
        """Downloads model files directly to self.local_models_path."""
        try:
            ASCIIColors.info(f"Downloading {model_name} to {self.local_models_path}...")
            
            # We preserve the folder structure simply using the last part of the repo name
            # e.g. 'meta-llama/Llama-2-7b' -> 'Llama-2-7b' folder in local path.
            # OR use the full 'meta-llama_Llama-2-7b' to avoid name collisions.
            folder_name = model_name.replace("/", "_")
            target_dir = self.local_models_path / folder_name
            
            # local_dir ensures actual files are downloaded, not just cache pointers
            path = snapshot_download(repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False)
            
            msg = f"Model downloaded successfully to {path}"
            ASCIIColors.success(msg)
            return {"status": True, "message": msg, "path": str(path)}
        except Exception as e:
            return {"status": False, "message": str(e)}

    def train(self, base_model_name: str, dataset_path: str, new_model_name: str, num_epochs=1, batch_size=1, learning_rate=2e-4) -> dict:
        if self.is_training: return {"status": False, "message": "Busy."}
        
        # Output to local path
        output_dir = self.local_models_path / new_model_name
        if output_dir.exists(): return {"status": False, "message": "Model exists."}
        
        # Resolve base model path (is it local or remote?)
        # Reuse logic from load_model's resolution if strictly needed, or let HF handle it.
        # But for QLoRA, we usually want the base model weights. 
        # We pass 'base_model_name' directly; if it matches a local folder in `load_model`, 
        # the user should probably pass that full path or we resolve it here.
        # Let's resolve it against local path:
        possible_local = self.local_models_path / base_model_name
        if possible_local.exists():
            base_model_path = str(possible_local)
        else:
            base_model_path = base_model_name

        t = threading.Thread(target=self._run_training_job, args=(base_model_path, dataset_path, str(output_dir), num_epochs, batch_size, learning_rate))
        t.start()
        return {"status": True, "message": f"Training started. Output: {output_dir}"}

    def _run_training_job(self, base_model, dataset_path, output_dir, epochs, batch_size, lr):
        self.is_training = True
        self.inference_lock.acquire()
        try:
            ASCIIColors.info(f"Training Base: {base_model}")
            
            # Dataset
            ext = "json" if dataset_path.endswith("json") else "text"
            dataset = load_dataset(ext, data_files=dataset_path, split="train")

            # QLoRA Setup
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb_config, device_map="auto",
                trust_remote_code=self.config.get("trust_remote_code", False)
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)
            
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            peft_config = LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none", lora_dropout=0.1)
            model = get_peft_model(model, peft_config)

            # Formatting
            def format_prompts(examples):
                texts = []
                for i in range(len(examples.get("instruction", []))):
                    ins = examples["instruction"][i]
                    inp = examples.get("input", [""])[i]
                    out = examples.get("output", [""])[i]
                    if inp: text = f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}<|endoftext|>"
                    else: text = f"### Instruction:\n{ins}\n\n### Response:\n{out}<|endoftext|>"
                    texts.append(text)
                return texts if texts else examples.get("text", [])

            trainer = SFTTrainer(
                model=model, train_dataset=dataset, peft_config=peft_config,
                formatting_func=format_prompts, tokenizer=tokenizer,
                args=TrainingArguments(
                    output_dir=output_dir, num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size, gradient_accumulation_steps=4,
                    learning_rate=lr, fp16=True, logging_steps=10, save_strategy="epoch", optim="paged_adamw_32bit"
                )
            )
            trainer.train()
            trainer.save_model(output_dir)
            ASCIIColors.success("Training Finished.")
        except Exception as e:
            ASCIIColors.error(f"Training error: {e}")
            trace_exception(e)
        finally:
            self.inference_lock.release()
            self.is_training = False

    def merge_lora(self, base_model_name, lora_model_name, new_model_name):
        # Resolve Base
        possible_base = self.local_models_path / base_model_name
        base_path = str(possible_base) if possible_base.exists() else base_model_name
        
        # Resolve LoRA (Usually local if trained here)
        possible_lora = self.local_models_path / lora_model_name
        lora_path = str(possible_lora) if possible_lora.exists() else lora_model_name

        save_path = self.local_models_path / new_model_name
        
        try:
            ASCIIColors.info(f"Merging {base_path} + {lora_path} -> {save_path}")
            base = AutoModelForCausalLM.from_pretrained(base_path, return_dict=True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=self.config.get("trust_remote_code"))
            tokenizer = AutoTokenizer.from_pretrained(base_path)
            
            merged = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
            merged.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            return {"status": True, "message": "Merged."}
        except Exception as e:
            return {"status": False, "message": str(e)}

def ps(self) -> Dict[str, List[Dict]]:
        """
        Returns the process status of loaded models, including memory usage.
        """
        models_status = []
        
        # Get global GPU info once
        gpu_total_mem = 0
        if torch.cuda.is_available():
            try:
                gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            except:
                gpu_total_mem = 0

        system_mem = psutil.virtual_memory()

        for mid, container in self.loaded_models.items():
            # 1. Calculate Model Size (Bytes)
            try:
                # Hugging Face models track their own footprint
                size_bytes = container.model.get_memory_footprint()
            except Exception:
                size_bytes = 0
            
            # 2. Split into VRAM/RAM based on device
            size_vram = 0
            size_ram = 0
            
            if container.device == "cuda":
                size_vram = size_bytes
            else:
                size_ram = size_bytes
            
            # 3. Calculate Percentages
            gpu_usage_percent = 0
            if gpu_total_mem > 0:
                gpu_usage_percent = (size_vram / gpu_total_mem) * 100
            
            # For CPU, we compare against total system RAM
            cpu_usage_percent = 0
            if system_mem.total > 0:
                cpu_usage_percent = (size_ram / system_mem.total) * 100

            models_status.append({
                "model_name": mid,            # UI Standard: 'model_name'
                "active": mid == self.active_model_id,
                "size": size_bytes,           # Total size in bytes
                "size_vram": size_vram,       # GPU memory usage in bytes
                "size_ram": size_ram,         # RAM usage in bytes
                "device": container.device,
                "gpu_usage_percent": round(gpu_usage_percent, 2),
                "cpu_usage_percent": round(cpu_usage_percent, 2),
                "loader": "HuggingFace"
            })
            
        # Return a dictionary matching the YAML output definition
        return {"models": models_status}