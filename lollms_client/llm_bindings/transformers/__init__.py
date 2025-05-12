# bindings/ollama/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors

import pipmaster as pm
if not pm.is_installed("torch"):
    ASCIIColors.yellow("Diffusers: Torch not found. Installing it")
    pm.install_multiple(["torch", "torchvision", "torchaudio"], "https://download.pytorch.org/whl/cu121", force_reinstall=True)

import torch
if not torch.cuda.is_available():
    ASCIIColors.yellow("Diffusers: Torch not using cuda. Reinstalling it")
    pm.install_multiple(["torch", "torchvision", "torchaudio"], "https://download.pytorch.org/whl/cu121", force_reinstall=True)
    import torch

if not pm.is_installed("transformers"):
    pm.install_or_update("transformers")

BindingName = "TransformersBinding"

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from packaging import version
import transformers

class TransformersBinding(LollmsLLMBinding):
    """Transformers-specific binding implementation"""
    
    def __init__(self,
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat,
                 prompt_template: Optional[str] = None):
        """
        Initialize the Transformers binding.

        Args:
            host_address (str): Host address for the service. Defaults to None.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            default_completion_format (ELF_COMPLETION_FORMAT): Default format for completions.
            prompt_template (Optional[str]): Custom prompt template. If None, inferred from model.
        """
        super().__init__(
            binding_name = "transformers",
            host_address=host_address,
            model_name=model_name,
            service_key=service_key,
            verify_ssl_certificate=verify_ssl_certificate,
            default_completion_format=default_completion_format
        )
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_name),
            trust_remote_code=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )
        
        self.generation_config = GenerationConfig.from_pretrained(str(model_name))
        
        # Infer or set prompt template
        self.prompt_template = prompt_template if prompt_template else self._infer_prompt_template(model_name)
        
        # Display device information
        device = next(self.model.parameters()).device
        device_type = "CPU" if device.type == "cpu" else "GPU"
        device_str = f"Running on {device}"
        
        ASCIIColors.multicolor(
            ["Model loaded - ", device_str],
            [ASCIIColors.color_green, ASCIIColors.color_blue if device_type == "GPU" else ASCIIColors.color_red]
        )

    def _infer_prompt_template(self, model_name: str) -> str:
        """
        Infer the prompt template based on the model name.

        Args:
            model_name (str): Name of the model.

        Returns:
            str: The inferred prompt template format string.
        """
        model_name = model_name.lower()
        if "llama-2" in model_name or "llama" in model_name:
            return "[INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt} [/INST]"
        elif "gpt" in model_name:
            return "{system_prompt}\n{user_prompt}"  # Simple concatenation for GPT-style models
        else:
            # Default to a basic chat format
            ASCIIColors.yellow(f"Warning: No specific template found for {model_name}. Using default chat format.")
            return "[INST] {system_prompt}\n{user_prompt} [/INST]"

    def generate_text(self, 
                      prompt: str,
                      images: Optional[List[str]] = None,
                      n_predict: Optional[int] = None,
                      stream: bool = False,
                      temperature: float = 0.1,
                      top_k: int = 50,
                      top_p: float = 0.95,
                      repeat_penalty: float = 0.8,
                      repeat_last_n: int = 40,
                      seed: Optional[int] = None,
                      n_threads: int = 8,
                      ctx_size: int | None = None,
                      streaming_callback: Optional[Callable[[str, str], None]] = None,
                      return_legacy_cache: bool = False,
                      system_prompt: str = "You are a helpful assistant.") -> Union[str, dict]:
        """
        Generate text using the Transformers model, with optional image support.

        Args:
            prompt (str): The input prompt for text generation (user prompt).
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.1.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p sampling parameter. Defaults to 0.95.
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40.
            seed (Optional[int]): Random seed for generation.
            n_threads (int): Number of threads to use. Defaults to 8.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback for streaming output.
            return_legacy_cache (bool): Whether to use legacy cache format (pre-v4.47). Defaults to False.
            system_prompt (str): System prompt to set model behavior. Defaults to "You are a helpful assistant."

        Returns:
            Union[str, dict]: Generated text if successful, or a dictionary with status and error if failed.
        """
        try:
            if not self.model or not self.tokenizer:
                return {"status": "error", "error": "Model or tokenizer not loaded"}

            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # Apply the prompt template
            formatted_prompt = self.prompt_template.format(
                system_prompt=system_prompt,
                user_prompt=prompt
            )

            # Prepare generation config
            self.generation_config.max_new_tokens = n_predict if n_predict else 2048
            self.generation_config.temperature = temperature
            self.generation_config.top_k = top_k
            self.generation_config.top_p = top_p
            self.generation_config.repetition_penalty = repeat_penalty
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            # Tokenize input with attention mask
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)

            # Handle image input if provided (basic implementation)
            if images and len(images) > 0:
                ASCIIColors.yellow("Warning: Image processing not fully implemented in this binding")
                formatted_prompt += "\n[Image content not processed]"

            # Check transformers version for cache handling
            use_legacy_cache = return_legacy_cache or version.parse(transformers.__version__) < version.parse("4.47.0")

            if stream:
                # Streaming case
                if not streaming_callback:
                    return {"status": "error", "error": "Streaming callback required for stream mode"}

                generated_text = ""
                # Generate with streaming
                for output in self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    return_legacy_cache=use_legacy_cache
                ):
                    # Handle different output formats based on version/cache setting
                    if use_legacy_cache:
                        sequences = output[0]
                    else:
                        sequences = output.sequences
                    
                    # Decode the new tokens
                    new_tokens = sequences[:, -1:]  # Get the last generated token
                    chunk = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                    generated_text += chunk
                    
                    # Send chunk through callback
                    streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK)
                
                return generated_text

            else:
                # Non-streaming case
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    return_legacy_cache=use_legacy_cache
                )
                
                # Handle different output formats
                sequences = outputs[0] if use_legacy_cache else outputs.sequences
                
                # Decode the full sequence, removing the input prompt
                generated_text = self.tokenizer.decode(
                    sequences[0][input_ids.shape[-1]:],
                    skip_special_tokens=True
                )
                
                return generated_text

        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            ASCIIColors.red(error_msg)
            return {"status": "error", "error": error_msg}

    def tokenize(self, text: str) -> list:
        """Tokenize the input text into a list of characters."""
        return list(text)
    
    def detokenize(self, tokens: list) -> str:
        """Convert a list of tokens back to text."""
        return "".join(tokens)
    
    def embed(self, text: str, **kwargs) -> list:
        """Get embeddings for the input text (placeholder)."""
        pass
    
    def get_model_info(self) -> dict:
        """Return information about the current model."""
        return {
            "name": "transformers",
            "version": transformers.__version__,
            "host_address": self.host_address,
            "model_name": self.model_name
        }
    
    def listModels(self):
        """Lists available models (placeholder)."""
        pass
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model into the binding."""
        self.model = model_name
        self.model_name = model_name
        return True
