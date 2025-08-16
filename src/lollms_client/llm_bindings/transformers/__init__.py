# bindings/huggingface_hub/binding.py
import json
import os
import pprint
import re
import socket # Not used directly for server, but good to keep for consistency if needed elsewhere
import subprocess # Not used for server
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any, Set
import base64 # For potential image data handling, though PIL.Image is primary
import requests # Not used for server, but for consistency

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

# --- Pipmaster: Ensure dependencies ---
pm.ensure_packages([
    "torch",
    "transformers",
    "accelerate",       # For device_map="auto" and advanced model loading
    "bitsandbytes",     # For 4-bit/8-bit quantization (works best on CUDA)
    "sentence_transformers", # For robust embedding generation
    "pillow"            # For image handling (vision models)
])

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer,
        BitsAndBytesConfig, AutoConfig, GenerationConfig,
        AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, # Common LLaVA models
        StoppingCriteria, StoppingCriteriaList
    )
    from sentence_transformers import SentenceTransformer
    from PIL import Image
except ImportError as e:
    ASCIIColors.error(f"Failed to import core libraries: {e}")
    ASCIIColors.error("Please ensure torch, transformers, accelerate, bitsandbytes, sentence_transformers, and pillow are installed.")
    trace_exception(e)
    # Set them to None so the binding can report failure cleanly if __init__ is still called.
    torch = None 
    transformers = None
    sentence_transformers = None
    Image = None


# --- Custom Stopping Criteria for Hugging Face generate ---
class StopOnWords(StoppingCriteria):
    def __init__(self, tokenizer, stop_words: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequences_token_ids = []
        for word in stop_words:
            # Encode stop words without adding special tokens to get their raw token IDs
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if token_ids:
                self.stop_sequences_token_ids.append(torch.tensor(token_ids))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_seq_ids in self.stop_sequences_token_ids:
            if input_ids.shape[1] >= stop_seq_ids.shape[0]:
                # Check if the end of input_ids matches the stop sequence
                if torch.equal(input_ids[0, -stop_seq_ids.shape[0]:], stop_seq_ids.to(input_ids.device)):
                    return True
        return False


BindingName = "HuggingFaceHubBinding"

class HuggingFaceHubBinding(LollmsLLMBinding):
    DEFAULT_CONFIG_ARGS = {
        "device": "auto",  # "auto", "cuda", "mps", "cpu"
        "quantize": False, # False, "8bit", "4bit" (8bit/4bit require CUDA and bitsandbytes)
        "torch_dtype": "auto", # "auto", "float16", "bfloat16", "float32"
        "max_new_tokens": 2048, # Default for generation
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "trust_remote_code": False, # Set to True for models like Phi, some LLaVA, etc.
        "use_flash_attention_2": False, # If supported by hardware/model & transformers version
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2", # Default for embed()
        "generation_timeout": 300, # Timeout for non-streaming generation
        "stop_words": [], # List of strings to stop generation
    }

    def __init__(self,
                 **kwargs # Overrides for config_args
                 ):
        """
        Initializes the Hugging Face Hub binding.
        Args:
            model_name (str): Hugging Face Hub model ID or local folder name.
            models_path (str or Path): Path to the directory containing local models.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary to override defaults.
            default_completion_format (ELF_COMPLETION_FORMAT): Default format for text generation.
        """
        super().__init__(BindingName, **kwargs)

        model_name_or_id = kwargs.get("model_name")
        models_path = kwargs.get("models_path")
        config = kwargs.get("config")
        default_completion_format = kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)

        if torch is None or transformers is None: # Check if core imports failed
            raise ImportError("Core libraries (torch, transformers) not available. Binding cannot function.")

        self.models_path = Path(models_path)
        self.config = {**self.DEFAULT_CONFIG_ARGS, **(config or {}), **kwargs}
        self.default_completion_format = default_completion_format
        
        self.model_identifier: Optional[str] = None
        self.model_name: Optional[str] = None # User-friendly name (folder name or hub id)
        self.model: Optional[Union[AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration]] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[AutoProcessor] = None # For vision models
        self.embedding_model: Optional[SentenceTransformer] = None
        self.device: Optional[str] = None
        self.torch_dtype: Optional[torch.dtype] = None
        self.supports_vision: bool = False

        # Attempt to load the model during initialization
        if not self.load_model(model_name_or_id):
             # load_model will print errors. Here we can raise if critical.
             ASCIIColors.error(f"Initial model load failed for {model_name_or_id}. Binding may not be functional.")
             # Depending on Lollms behavior, this might be acceptable if user can select another model later.

    def _resolve_model_path_or_id(self, model_name_or_id: str) -> str:
        # 1. Check if it's an absolute path to a model directory
        abs_path = Path(model_name_or_id)
        if abs_path.is_absolute() and abs_path.is_dir() and (abs_path / "config.json").exists():
            ASCIIColors.info(f"Using absolute model path: {abs_path}")
            return str(abs_path)

        # 2. Check if it's a name relative to self.models_path
        local_model_path = self.models_path / model_name_or_id
        if local_model_path.is_dir() and (local_model_path / "config.json").exists():
            ASCIIColors.info(f"Found local model in models_path: {local_model_path}")
            return str(local_model_path)
        
        # 3. Assume it's a Hugging Face Hub ID
        ASCIIColors.info(f"Assuming '{model_name_or_id}' is a Hugging Face Hub ID.")
        return model_name_or_id

    def load_model(self, model_name_or_id: str) -> bool:
        if self.model is not None:
            self.unload_model()

        self.model_identifier = self._resolve_model_path_or_id(model_name_or_id)
        self.model_name = Path(self.model_identifier).name # User-friendly name

        # --- Device Selection ---
        device_pref = self.config.get("device", "auto")
        if device_pref == "auto":
            if torch.cuda.is_available(): self.device = "cuda"
            elif torch.backends.mps.is_available(): self.device = "mps" # For Apple Silicon
            else: self.device = "cpu"
        else:
            self.device = device_pref
        ASCIIColors.info(f"Using device: {self.device}")

        # --- Dtype Selection ---
        dtype_pref = self.config.get("torch_dtype", "auto")
        if dtype_pref == "auto":
            if self.device == "cuda": self.torch_dtype = torch.float16 # bfloat16 is better for Ampere+
            else: self.torch_dtype = torch.float32 # MPS and CPU generally use float32
        elif dtype_pref == "float16": self.torch_dtype = torch.float16
        elif dtype_pref == "bfloat16": self.torch_dtype = torch.bfloat16
        else: self.torch_dtype = torch.float32
        ASCIIColors.info(f"Using DType: {self.torch_dtype}")

        # --- Quantization ---
        quantize_mode = self.config.get("quantize", False)
        load_in_8bit = False
        load_in_4bit = False
        bnb_config = None

        if self.device == "cuda": # bitsandbytes primarily for CUDA
            if quantize_mode == "8bit":
                load_in_8bit = True
                ASCIIColors.info("Quantizing model to 8-bit.")
            elif quantize_mode == "4bit":
                load_in_4bit = True
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.torch_dtype # e.g., torch.bfloat16 for computation
                )
                ASCIIColors.info("Quantizing model to 4-bit.")
        elif quantize_mode and self.device != "cuda":
            ASCIIColors.warning(f"Quantization ('{quantize_mode}') is selected but device is '{self.device}'. bitsandbytes works best on CUDA. Proceeding without quantization.")
            quantize_mode = False


        # --- Model Loading Arguments ---
        model_load_args = {
            "trust_remote_code": self.config.get("trust_remote_code", False),
            # torch_dtype is handled by BitsAndBytesConfig if quantizing, otherwise set directly
            "torch_dtype": self.torch_dtype if not (load_in_8bit or load_in_4bit) else None,
        }
        if self.config.get("use_flash_attention_2", False) and self.device == "cuda":
            if hasattr(transformers, " আসছেAttention"): # Check for Flash Attention support in transformers version
                model_load_args["attn_implementation"] = "flash_attention_2"
                ASCIIColors.info("Attempting to use Flash Attention 2.")
            else:
                ASCIIColors.warning("Flash Attention 2 requested but not found in this transformers version. Using default.")


        if load_in_8bit: model_load_args["load_in_8bit"] = True
        if load_in_4bit: model_load_args["quantization_config"] = bnb_config
        
        # device_map="auto" for multi-GPU or when quantizing on CUDA
        if self.device == "cuda" and (load_in_8bit or load_in_4bit or torch.cuda.device_count() > 1):
            model_load_args["device_map"] = "auto"
            ASCIIColors.info("Using device_map='auto'.")
        
        try:
            ASCIIColors.info(f"Loading tokenizer for '{self.model_identifier}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                trust_remote_code=model_load_args["trust_remote_code"]
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                ASCIIColors.info("Tokenizer `pad_token` was None, set to `eos_token`.")

            # --- Determine if it's a LLaVA-like vision model ---
            model_config_hf = AutoConfig.from_pretrained(self.model_identifier, trust_remote_code=model_load_args["trust_remote_code"])
            self.supports_vision = "llava" in model_config_hf.model_type.lower() or \
                                   any("Llava" in arch for arch in getattr(model_config_hf, "architectures", [])) or \
                                   "vision_tower" in model_config_hf.to_dict() # Common LLaVA config key
            
            if self.supports_vision:
                ASCIIColors.info(f"Detected LLaVA-like vision model: '{self.model_identifier}'.")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_identifier,
                    trust_remote_code=model_load_args["trust_remote_code"]
                )
                # Choose appropriate LLaVA model class
                if "llava-next" in self.model_identifier.lower() or any("LlavaNext" in arch for arch in getattr(model_config_hf, "architectures", [])):
                     ModelClass = LlavaNextForConditionalGeneration
                elif "llava" in self.model_identifier.lower() or any("LlavaForConditionalGeneration" in arch for arch in getattr(model_config_hf, "architectures", [])):
                     ModelClass = LlavaForConditionalGeneration
                else: # Fallback if specific Llava class not matched by name
                    ASCIIColors.warning("Could not determine specific LLaVA class, using AutoModelForCausalLM. Vision capabilities might be limited.")
                    ModelClass = AutoModelForCausalLM # This might not fully work for all LLaVAs
                
                self.model = ModelClass.from_pretrained(self.model_identifier, **model_load_args)
            else:
                ASCIIColors.info(f"Loading text model '{self.model_identifier}'...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier, **model_load_args)

            # If not using device_map, move model to the selected device
            if "device_map" not in model_load_args and self.device != "cpu":
                self.model.to(self.device)
            
            self.model.eval() # Set to evaluation mode

            # --- Load Embedding Model ---
            emb_model_name = self.config.get("embedding_model_name")
            if emb_model_name:
                try:
                    ASCIIColors.info(f"Loading embedding model: {emb_model_name} on device: {self.device}")
                    self.embedding_model = SentenceTransformer(emb_model_name, device=self.device)
                except Exception as e_emb:
                    ASCIIColors.warning(f"Failed to load embedding model '{emb_model_name}': {e_emb}. Embeddings will not be available.")
                    self.embedding_model = None
            else:
                ASCIIColors.info("No embedding_model_name configured. Skipping embedding model load.")
                self.embedding_model = None

            ASCIIColors.green(f"Model '{self.model_identifier}' loaded successfully.")
            return True

        except Exception as e:
            ASCIIColors.error(f"Failed to load model '{self.model_identifier}': {e}")
            trace_exception(e)
            self.unload_model() # Ensure partial loads are cleaned up
            return False

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.model_identifier = None
        self.model_name = None
        self.supports_vision = False
        ASCIIColors.info("Hugging Face model unloaded.")

    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None, 
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: bool = False,
                     temperature: float = None,
                     top_k: int = None,
                     top_p: float = None,
                     repeat_penalty: float = None,
                     seed: Optional[int] = None,
                     stop_words: Optional[List[str]] = None, # Added custom stop_words
                     streaming_callback: Optional[Callable[[str, int], bool]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     use_chat_format_override: Optional[bool] = None,
                     **generation_kwargs 
                     ) -> Union[str, Dict[str, Any]]:

        if self.model is None or self.tokenizer is None:
             return {"status": False, "error": "Model not loaded."}

        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda": torch.cuda.manual_seed_all(seed)

        _use_chat_format = use_chat_format_override if use_chat_format_override is not None \
                           else (self.default_completion_format == ELF_COMPLETION_FORMAT.Chat)

        # --- Prepare Inputs ---
        inputs_dict = {}
        processed_images = []
        if self.supports_vision and self.processor and images:
            try:
                for img_path in images:
                    processed_images.append(Image.open(img_path).convert("RGB"))
                # LLaVA processor typically takes text and images, returns combined inputs
                inputs_dict = self.processor(text=prompt, images=processed_images, return_tensors="pt").to(self.model.device)
                ASCIIColors.debug("Processed inputs with LLaVA processor.")
            except Exception as e_img:
                ASCIIColors.error(f"Error processing images for LLaVA: {e_img}")
                return {"status": False, "error": f"Image processing error: {e_img}"}
        
        elif _use_chat_format and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt: messages.append({"role": "system", "content": system_prompt})
            
            # Newer chat templates can handle images directly in content if tokenizer supports it
            # Example: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}]
            # For now, this example keeps LLaVA processor separate.
            messages.append({"role": "user", "content": prompt})
            
            try:
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs_dict = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                ASCIIColors.debug("Applied chat template.")
            except Exception as e_tmpl: # Some tokenizers might fail if template is complex or not well-defined
                 ASCIIColors.warning(f"Failed to apply chat template ({e_tmpl}). Falling back to raw prompt.")
                 _use_chat_format = False # Fallback

        if not _use_chat_format or not inputs_dict: # Raw prompt or fallback
            full_prompt_text = ""
            if system_prompt: full_prompt_text += system_prompt + "\n\n"
            full_prompt_text += prompt
            inputs_dict = self.tokenizer(full_prompt_text, return_tensors="pt").to(self.model.device)
            ASCIIColors.debug("Using raw prompt format.")

        input_ids = inputs_dict.get("input_ids")
        if input_ids is None: return {"status": False, "error": "Failed to tokenize prompt."}
        
        current_input_length = input_ids.shape[1]

        # --- Generation Parameters ---
        gen_conf = GenerationConfig.from_model_config(self.model.config) # Start with model's default
        
        gen_conf.max_new_tokens = n_predict if n_predict is not None else self.config.get("max_new_tokens")
        gen_conf.temperature = temperature if temperature is not None else self.config.get("temperature")
        gen_conf.top_k = top_k if top_k is not None else self.config.get("top_k")
        gen_conf.top_p = top_p if top_p is not None else self.config.get("top_p")
        gen_conf.repetition_penalty = repeat_penalty if repeat_penalty is not None else self.config.get("repetition_penalty")
        gen_conf.pad_token_id = self.tokenizer.eos_token_id # Crucial for stopping
        gen_conf.eos_token_id = self.tokenizer.eos_token_id

        # Apply any other valid GenerationConfig parameters from generation_kwargs
        for key, value in generation_kwargs.items():
            if hasattr(gen_conf, key): setattr(gen_conf, key, value)

        # --- Stopping Criteria ---
        stopping_criteria_list = StoppingCriteriaList()
        effective_stop_words = stop_words if stop_words is not None else self.config.get("stop_words", [])
        if effective_stop_words:
            stopping_criteria_list.append(StopOnWords(self.tokenizer, effective_stop_words))

        # --- Generation ---
        try:
            if stream and streaming_callback:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_thread_kwargs = {
                    **inputs_dict, # input_ids, attention_mask, pixel_values (if vision)
                    "generation_config": gen_conf,
                    "streamer": streamer,
                    "stopping_criteria": stopping_criteria_list if effective_stop_words else None
                }
                
                thread = threading.Thread(target=self.model.generate, kwargs=generation_thread_kwargs)
                thread.start()

                full_response_text = ""
                for new_text_chunk in streamer:
                    if streaming_callback(new_text_chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        full_response_text += new_text_chunk
                    else: # Callback requested stop
                        ASCIIColors.info("Streaming callback requested stop.")
                        # Note: stopping the model.generate thread externally is complex.
                        # The thread will complete its current generation.
                        break 
                thread.join(timeout=self.config.get("generation_timeout", 300))
                if thread.is_alive():
                    ASCIIColors.warning("Generation thread did not finish in time after streaming.")
                return full_response_text
            else: # Non-streaming
                outputs = self.model.generate(
                    **inputs_dict,
                    generation_config=gen_conf,
                    stopping_criteria=stopping_criteria_list if effective_stop_words else None
                )
                # outputs contains the full sequence (prompt + new tokens)
                generated_tokens = outputs[0][current_input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return generated_text.strip()

        except Exception as e:
            ASCIIColors.error(f"Error during text generation: {e}")
            trace_exception(e)
            return {"status": False, "error": str(e)}

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None: raise RuntimeError("Tokenizer not loaded.")
        return self.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        if self.tokenizer is None: raise RuntimeError("Tokenizer not loaded.")
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None: raise RuntimeError("Tokenizer not loaded.")
        return len(self.tokenizer.encode(text))

    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Configure 'embedding_model_name'.")
        try:
            # SentenceTransformer's encode can take a string or list of strings
            embeddings_np = self.embedding_model.encode(text, **kwargs)
            if isinstance(text, str): # Single text input
                return embeddings_np.tolist()
            else: # List of texts input
                return [emb.tolist() for emb in embeddings_np]
        except Exception as e:
            ASCIIColors.error(f"Embedding generation failed: {e}")
            trace_exception(e)
            raise

    def get_model_info(self) -> dict:
        info = {
            "binding_name": self.binding_name,
            "model_name": self.model_name,
            "model_identifier": self.model_identifier,
            "loaded": self.model is not None,
            "config": self.config, # Binding's own config
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "supports_vision": self.supports_vision,
            "embedding_model_name": self.config.get("embedding_model_name") if self.embedding_model else None,
        }
        if self.model and hasattr(self.model, 'config'):
            model_hf_config = self.model.config.to_dict()
            info["model_hf_config"] = {k: str(v)[:200] for k,v in model_hf_config.items()} # Truncate long values
            info["max_model_len"] = getattr(self.model.config, "max_position_embeddings", "N/A")
        
        info["supports_structured_output"] = False # HF models don't inherently support grammar like llama.cpp server
                                                # (unless using external libraries like outlines)
        return info

    def listModels(self) -> List[Dict[str, str]]:
        models_found = []
        unique_model_names = set()

        if self.models_path.exists() and self.models_path.is_dir():
            for item in self.models_path.iterdir():
                if item.is_dir(): # HF models are directories
                    # Basic check for a config file to qualify as a model dir
                    if (item / "config.json").exists():
                        model_name = item.name
                        if model_name not in unique_model_names:
                            try:
                                # Calculating size can be slow for large model repos
                                # total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                # size_gb_str = f"{total_size / (1024**3):.2f} GB"
                                size_gb_str = "N/A (size calculation disabled for speed)"
                            except Exception:
                                size_gb_str = "N/A"

                            models_found.append({
                                "model_name": model_name, # This is the folder name
                                "path_hint": str(item.relative_to(self.models_path.parent) if item.is_relative_to(self.models_path.parent) else item),
                                "size_gb": size_gb_str
                            })
                            unique_model_names.add(model_name)
        
        ASCIIColors.info("Tip: You can also use any Hugging Face Hub model ID directly (e.g., 'mistralai/Mistral-7B-Instruct-v0.1').")
        return models_found
    
    def __del__(self):
        self.unload_model()


if __name__ == '__main__':
    global full_streamed_text
    ASCIIColors.yellow("Testing HuggingFaceHubBinding...")

    # --- Configuration ---
    # For testing, you might need to download a model first or use a small Hub ID.
    # Option 1: Use a small model from Hugging Face Hub
    # test_model_name = "gpt2" # Small, good for quick tests
    test_model_name = "microsoft/phi-2" # Small, good quality, requires trust_remote_code=True
    # test_model_name = "HuggingFaceH4/zephyr-7b-beta" # Larger, powerful

    # Option 2: Path to a local model folder (if you have one)
    # Replace 'path/to/your/models' with the PARENT directory of your HF model folders.
    # And 'your-local-model-folder' with the actual folder name.
    # Example:
    # test_models_parent_path = Path.home() / "lollms_models" # Example path
    # test_model_name = "phi-2" # if "phi-2" folder is inside test_models_parent_path

    # For local testing, models_path should be where your HF model *folders* are.
    # If using a Hub ID like "gpt2", models_path is less critical unless you expect
    # the binding to *only* look there (which it doesn't, it prioritizes Hub IDs).
    # Let's use a dummy path for models_path for Hub ID testing.
    
    # Adjust current_directory for local model testing if needed
    # For this test, we'll assume a Hub ID. `models_path` is where `listModels` would scan.
    test_models_parent_path = Path("./test_hf_models_dir") # Create a dummy for listModels scan
    test_models_parent_path.mkdir(exist_ok=True)

    binding_config = {
        "device": "auto",          # "cuda", "mps", "cpu"
        "quantize": False,         # False, "4bit", "8bit" (requires CUDA & bitsandbytes for 4/8 bit)
        "torch_dtype": "auto",     # "float16" or "bfloat16" on CUDA for speed
        "max_new_tokens": 100,     # Limit generation length for tests
        "trust_remote_code": True, # Needed for models like Phi-2
        "stop_words": ["\nHuman:", "\nUSER:"], # Example stop words
        # "embedding_model_name": "sentence-transformers/paraphrase-MiniLM-L3-v2" # Smaller embedding model
    }

    active_binding = None
    try:
        ASCIIColors.cyan("\n--- Initializing HuggingFaceHubBinding ---")
        active_binding = HuggingFaceHubBinding(
            model_name_or_id=test_model_name,
            models_path=test_models_parent_path,
            config=binding_config
        )
        if not active_binding.model:
            raise RuntimeError(f"Model '{test_model_name}' failed to load.")

        ASCIIColors.green(f"Binding initialized. Model '{active_binding.model_name}' loaded on {active_binding.device}.")
        ASCIIColors.info(f"Model Info: {json.dumps(active_binding.get_model_info(), indent=2, default=str)}")

        # --- List Models (scans configured models_path) ---
        ASCIIColors.cyan("\n--- Listing Models (from models_path) ---")
        # To make this test useful, you could manually place a model folder in `test_hf_models_dir`
        # e.g., download "gpt2" and put it in `test_hf_models_dir/gpt2`
        # For now, it will likely be empty unless you do that.
        listed_models = active_binding.listModels()
        if listed_models:
            ASCIIColors.green(f"Found {len(listed_models)} potential model folders. First 5:")
            for m in listed_models[:5]: print(m)
        else: ASCIIColors.warning(f"No model folders found in '{test_models_parent_path}'. This is normal if it's empty.")

        # --- Tokenize/Detokenize ---
        ASCIIColors.cyan("\n--- Tokenize/Detokenize ---")
        sample_text = "Hello, Hugging Face world!"
        tokens = active_binding.tokenize(sample_text)
        ASCIIColors.green(f"Tokens for '{sample_text}': {tokens[:10]}...")
        token_count = active_binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count: {token_count}")
        if tokens:
            detokenized_text = active_binding.detokenize(tokens)
            ASCIIColors.green(f"Detokenized text: {detokenized_text}")
        else: ASCIIColors.warning("Tokenization returned empty list.")

        # --- Text Generation (Non-Streaming, Chat Format if supported) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "What is the capital of France?"
        # For Phi-2, system prompt might need specific formatting if not using apply_chat_template strictly
        # For models like Zephyr, system_prompt is part of chat template
        system_prompt_text = "You are a helpful AI assistant." 
        generated_text = active_binding.generate_text(
            prompt_text, system_prompt=system_prompt_text, stream=False,
            n_predict=30 # Override default max_new_tokens for this call
        )
        if isinstance(generated_text, str): ASCIIColors.green(f"Generated text: {generated_text}")
        else: ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            global full_streamed_text
            ASCIIColors.green(f"{chunk}", end="", flush=True)
            full_streamed_text += chunk
            return True # Continue streaming
        
        result = active_binding.generate_text(
            "Tell me a short story about a brave robot.", 
            stream=True, 
            streaming_callback=stream_callback,
            n_predict=70
        )
        print("\n--- End of Stream ---")
        if isinstance(result, str): ASCIIColors.green(f"Full streamed text collected: {result}")
        else: ASCIIColors.error(f"Streaming generation failed: {result}")

        # --- Embeddings ---
        if active_binding.embedding_model:
            ASCIIColors.cyan("\n--- Embeddings ---")
            embedding_text = "This is a test sentence for Hugging Face embeddings."
            try:
                embedding_vector = active_binding.embed(embedding_text)
                ASCIIColors.green(f"Embedding for '{embedding_text}' (first 3 dims): {embedding_vector[:3]}...")
                ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")

                # Test batch embedding
                batch_texts = ["First sentence.", "Second sentence, quite different."]
                batch_embeddings = active_binding.embed(batch_texts)
                ASCIIColors.green(f"Batch embeddings generated for {len(batch_texts)} texts.")
                ASCIIColors.info(f"First batch embedding (first 3 dims): {batch_embeddings[0][:3]}...")

            except Exception as e_emb: ASCIIColors.warning(f"Could not get embedding: {e_emb}")
        else: ASCIIColors.yellow("\n--- Embeddings Skipped (no embedding model loaded) ---")

        # --- LLaVA Vision Test (Conceptual - requires a LLaVA model and an image) ---
        # To test LLaVA properly:
        # 1. Set `test_model_name` to a LLaVA model, e.g., "llava-hf/llava-1.5-7b-hf" (very large!)
        #    or a smaller one like "unum-cloud/uform-gen2-qwen-500m" (check its specific prompting style).
        # 2. Ensure `trust_remote_code=True` might be needed.
        # 3. Provide a real image path.
        if active_binding.supports_vision:
            ASCIIColors.cyan("\n--- LLaVA Vision Test ---")
            dummy_image_path = Path("test_dummy_image.png")
            try:
                # Create a dummy image for testing
                img = Image.new('RGB', (200, 100), color = ('skyblue'))
                from PIL import ImageDraw
                d = ImageDraw.Draw(img)
                d.text((10,10), "Hello LLaVA from HF!", fill=('black'))
                img.save(dummy_image_path)
                ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

                llava_prompt = "Describe this image." # LLaVA models often use "<image>\nUSER: <prompt>\nASSISTANT:"
                                                   # or just the prompt if processor handles template.
                                                   # For AutoProcessor, often just the text part of the prompt.
                llava_response = active_binding.generate_text(
                    prompt=llava_prompt, 
                    images=[str(dummy_image_path)], 
                    n_predict=50, 
                    stream=False
                )
                if isinstance(llava_response, str): ASCIIColors.green(f"LLaVA response: {llava_response}")
                else: ASCIIColors.error(f"LLaVA generation failed: {llava_response}")

            except ImportError: ASCIIColors.warning("Pillow's ImageDraw not found for dummy image text.")
            except Exception as e_llava: ASCIIColors.error(f"LLaVA test error: {e_llava}"); trace_exception(e_llava)
            finally:
                if dummy_image_path.exists(): dummy_image_path.unlink()
        else:
            ASCIIColors.yellow("\n--- LLaVA Vision Test Skipped (model does not support vision or not configured for it) ---")

    except ImportError as e_imp:
        ASCIIColors.error(f"Import error: {e_imp}. Check installations.")
    except RuntimeError as e_rt:
        ASCIIColors.error(f"Runtime error: {e_rt}")
    except Exception as e_main:
        ASCIIColors.error(f"An unexpected error occurred: {e_main}")
        trace_exception(e_main)
    finally:
        if active_binding:
            ASCIIColors.cyan("\n--- Unloading Model ---")
            active_binding.unload_model()
            ASCIIColors.green("Model unloaded.")
        if test_models_parent_path.exists() and not any(test_models_parent_path.iterdir()): # cleanup dummy dir if empty
            try: os.rmdir(test_models_parent_path)
            except: pass


    ASCIIColors.yellow("\nHuggingFaceHubBinding test finished.")