# bindings/llamacpp/binding.py
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
# from lollms_client.lollms_utilities import encode_image # Used for LLaVA if needed to prepare image data.

from typing import Optional, Callable, List, Union, Dict, Any
import os
import sys
import base64 # For LLaVA image encoding

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

# Ensure llama-cpp-python is installed
# Installation can be tricky due to C++ compilation.
# Users might need to install it with specific flags for their hardware (e.g., CUDA, Metal).
pm.ensure_packages(["llama-cpp-python", "pillow", "tiktoken"]) # tiktoken as a very last resort if llama_cpp fails

try:
    from llama_cpp import Llama, LlamaGrammar, LogStdErrToPython
    # For LLaVA (multimodal) support
    from llama_cpp.llama_chat_format import LlamaChatCompletionRequestMessageImageContentPart, LlamaChatCompletionRequestMessageTextContentPart
except ImportError as e:
    ASCIIColors.error(f"Failed to import llama_cpp: {e}. Please ensure it is installed correctly.")
    ASCIIColors.error("Try: pip install llama-cpp-python")
    ASCIIColors.error("For GPU support, you might need to compile it with specific flags, e.g.:")
    ASCIIColors.error("  CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install llama-cpp-python (for NVIDIA)")
    ASCIIColors.error("  CMAKE_ARGS=\"-DLLAMA_METAL=on\" FORCE_CMAKE=1 pip install llama-cpp-python (for Apple Metal)")
    Llama = None 
    LlamaGrammar = None
    LogStdErrToPython = None
    LlamaChatCompletionRequestMessageImageContentPart = None
    LlamaChatCompletionRequestMessageTextContentPart = None
    # It's critical that the script can run even if llama_cpp is not installed,
    # so LoLLMs can still list it as an available binding and guide user for installation.
    # The __init__ will raise an error if Llama is None and an attempt is made to use the binding.


BindingName = "PythonLlamaCppBinding"

class PythonLlamaCppBinding(LollmsLLMBinding):
    """
    Llama.cpp binding implementation using the llama-cpp-python library.
    This binding loads and runs GGUF models locally.
    """
    
    DEFAULT_CONFIG = {
        "n_gpu_layers": 0,
        "main_gpu": 0,
        "tensor_split": None,
        "vocab_only": False,
        "use_mmap": True,
        "use_mlock": False,
        "seed": -1, # -1 for random
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": None,
        "n_threads_batch": None,
        "rope_scaling_type": None,
        "rope_freq_base": 0.0,
        "rope_freq_scale": 0.0,
        "yarn_ext_factor": -1.0,
        "yarn_attn_factor": 1.0,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_orig_ctx": 0,
        "logits_all": False,
        "embedding": False, # Enable for model.embed()
        "chat_format": "chatml", # Default chat format, LLaVA needs specific e.g. "llava-1-5"
        "clip_model_path": None, # For LLaVA: path to the mmproj GGUF file
        "verbose": True,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "mirostat_mode": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "grammar_file": None,
    }

    def __init__(self,
                 model_path: str, 
                 config: Optional[Dict[str, Any]] = None, 
                 lollms_paths: Optional[Dict[str, str]] = None, 
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat,
                 **kwargs 
                 ):
        
        super().__init__(binding_name=BindingName)
        
        if Llama is None: # Check if import failed
            raise ImportError("Llama-cpp-python library is not available. Please install it.")

        self.model_path = model_path
        self.default_completion_format = default_completion_format
        self.lollms_paths = lollms_paths if lollms_paths else {}

        self.llama_config = {**self.DEFAULT_CONFIG, **(config or {}), **kwargs}

        self.model: Optional[Llama] = None
        self.grammar: Optional[LlamaGrammar] = None

        # Resolve and load grammar if specified
        self._load_grammar_from_config()

        # Attempt to load the model
        self.load_model(self.model_path)

    def _load_grammar_from_config(self):
        grammar_file_path = self.llama_config.get("grammar_file")
        if grammar_file_path:
            full_grammar_path = grammar_file_path
            if self.lollms_paths.get('grammars_path') and not os.path.isabs(grammar_file_path):
                full_grammar_path = os.path.join(self.lollms_paths['grammars_path'], grammar_file_path)
            
            if os.path.exists(full_grammar_path):
                try:
                    self.grammar = LlamaGrammar.from_file(full_grammar_path)
                    ASCIIColors.info(f"Loaded GBNF grammar from: {full_grammar_path}")
                except Exception as e:
                    ASCIIColors.warning(f"Failed to load GBNF grammar from {full_grammar_path}: {e}")
            else:
                ASCIIColors.warning(f"Grammar file not found: {full_grammar_path}")

    def load_model(self, model_path: str) -> bool:
        self.model_path = model_path
        resolved_model_path = self.model_path
        if not os.path.exists(resolved_model_path):
            models_base_path = self.lollms_paths.get('personal_models_path', self.lollms_paths.get('models_zoo_path'))
            if models_base_path:
                # Assuming model_path might be relative to a binding-specific folder within models_base_path
                # e.g. models_zoo_path/llamacpp/model_name.gguf
                # Or it could be directly models_zoo_path/model_name.gguf
                potential_path_direct = os.path.join(models_base_path, self.model_path)
                potential_path_binding_specific = os.path.join(models_base_path, self.binding_name.lower(), self.model_path)

                if os.path.exists(potential_path_direct):
                    resolved_model_path = potential_path_direct
                elif os.path.exists(potential_path_binding_specific):
                    resolved_model_path = potential_path_binding_specific
                else:
                    raise FileNotFoundError(f"Model file '{self.model_path}' not found directly or in model paths: '{potential_path_direct}', '{potential_path_binding_specific}'")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        ASCIIColors.info(f"Attempting to load GGUF model from: {resolved_model_path}")
        self.model_path = resolved_model_path # Update to resolved path

        llama_constructor_keys = [
            "n_gpu_layers", "main_gpu", "tensor_split", "vocab_only", "use_mmap", "use_mlock",
            "seed", "n_ctx", "n_batch", "n_threads", "n_threads_batch",
            "rope_scaling_type", "rope_freq_base", "rope_freq_scale",
            "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
            "logits_all", "embedding", "verbose", "chat_format", "clip_model_path"
        ]
        constructor_params = {k: self.llama_config[k] for k in llama_constructor_keys if k in self.llama_config and self.llama_config[k] is not None}

        # Ensure seed is int
        if "seed" in constructor_params and not isinstance(constructor_params["seed"], int):
            constructor_params["seed"] = int(self.llama_config.get("seed", self.DEFAULT_CONFIG["seed"]))
        
        if "n_ctx" in constructor_params: constructor_params["n_ctx"] = int(constructor_params["n_ctx"])
        
        if "verbose" in constructor_params and isinstance(constructor_params["verbose"], str):
            constructor_params["verbose"] = constructor_params["verbose"].lower() in ["true", "1", "yes"]

        # Resolve clip_model_path for LLaVA if relative
        if constructor_params.get("clip_model_path") and self.lollms_paths.get('personal_models_path'):
            clip_path = constructor_params["clip_model_path"]
            if not os.path.isabs(clip_path):
                # Try resolving relative to where main model was found or standard models path
                model_dir = os.path.dirname(self.model_path)
                potential_clip_path1 = os.path.join(model_dir, clip_path)
                potential_clip_path2 = os.path.join(self.lollms_paths['personal_models_path'], clip_path)
                potential_clip_path3 = os.path.join(self.lollms_paths.get('models_zoo_path', ''), clip_path)

                if os.path.exists(potential_clip_path1):
                    constructor_params["clip_model_path"] = potential_clip_path1
                elif os.path.exists(potential_clip_path2):
                    constructor_params["clip_model_path"] = potential_clip_path2
                elif self.lollms_paths.get('models_zoo_path') and os.path.exists(potential_clip_path3):
                    constructor_params["clip_model_path"] = potential_clip_path3
                else:
                    ASCIIColors.warning(f"LLaVA clip_model_path '{clip_path}' not found at various potential locations.")


        ASCIIColors.info(f"Llama.cpp constructor parameters: {constructor_params}")
        try:
            if constructor_params.get("verbose", False) and LogStdErrToPython:
                LogStdErrToPython()
            self.model = Llama(model_path=self.model_path, **constructor_params)
            ASCIIColors.green("GGUF Model loaded successfully.")
            self.llama_config["n_ctx"] = self.model.context_params.n_ctx # Update n_ctx from loaded model
            return True
        except Exception as e:
            ASCIIColors.error(f"Failed to load GGUF model {self.model_path}: {e}")
            trace_exception(e)
            self.model = None
            raise RuntimeError(f"Failed to load GGUF model {self.model_path}") from e

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
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     streaming_callback: Optional[Callable[[str, int], bool]] = None,
                     use_chat_format: bool = True, 
                     grammar: Optional[Union[str, LlamaGrammar]] = None,
                     **generation_kwargs 
                     ) -> Union[str, Dict[str, any]]:
        
        if not self.model:
             return {"status": False, "error": "Llama.cpp model not loaded."}

        gen_params_from_config = {
            k: self.llama_config.get(k) for k in [
                "temperature", "top_k", "top_p", "repeat_penalty", 
                "mirostat_mode", "mirostat_tau", "mirostat_eta"
            ]
        }
        # repeat_last_n is penalty_last_n for Llama.generate, repeat_penalty_last_n for create_completion/chat_completion
        _repeat_last_n_cfg = self.llama_config.get("repeat_last_n")

        # Override with call-specific parameters
        gen_params = {
            "temperature": temperature if temperature is not None else gen_params_from_config["temperature"],
            "top_k": top_k if top_k is not None else gen_params_from_config["top_k"],
            "top_p": top_p if top_p is not None else gen_params_from_config["top_p"],
            "repeat_penalty": repeat_penalty if repeat_penalty is not None else gen_params_from_config["repeat_penalty"],
            "mirostat_mode": gen_params_from_config["mirostat_mode"],
            "mirostat_tau": gen_params_from_config["mirostat_tau"],
            "mirostat_eta": gen_params_from_config["mirostat_eta"],
        }
        _repeat_last_n = repeat_last_n if repeat_last_n is not None else _repeat_last_n_cfg
        if _repeat_last_n is not None:
            gen_params["penalty_last_n"] = _repeat_last_n # For Llama.generate (legacy, less used)
            gen_params["repeat_penalty_last_n"] = _repeat_last_n # For create_completion / create_chat_completion

        if n_predict is not None: gen_params['max_tokens'] = n_predict
        if seed is not None: gen_params['seed'] = seed
        
        gen_params = {k: v for k, v in gen_params.items() if v is not None} # Filter None
        gen_params.update(generation_kwargs) # Add any extra kwargs

        # Handle grammar for this call
        active_grammar = self.grammar # Model's default grammar
        if grammar:
            if isinstance(grammar, LlamaGrammar):
                active_grammar = grammar
            elif isinstance(grammar, str): # Path to grammar file
                g_path = grammar
                if self.lollms_paths.get('grammars_path') and not os.path.isabs(g_path):
                    g_path = os.path.join(self.lollms_paths['grammars_path'], g_path)
                if os.path.exists(g_path):
                    try:
                        active_grammar = LlamaGrammar.from_file(g_path)
                    except Exception as e_g: ASCIIColors.warning(f"Failed to load dynamic GBNF grammar from {g_path}: {e_g}")
                else: ASCIIColors.warning(f"Dynamic grammar file not found: {g_path}")
        if active_grammar: gen_params["grammar"] = active_grammar

        full_response_text = ""
        try:
            if use_chat_format:
                messages = []
                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                
                user_message_content = prompt
                if images and LlamaChatCompletionRequestMessageImageContentPart and LlamaChatCompletionRequestMessageTextContentPart:
                    # LLaVA format: content can be a list of text and image parts
                    content_parts = [{"type": "text", "text": prompt}]
                    for img_path in images:
                        try:
                            with open(img_path, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                            image_type = os.path.splitext(img_path)[1][1:].lower() or "png"
                            if image_type == "jpg": image_type = "jpeg"
                            image_uri = f"data:image/{image_type};base64,{encoded_string}"
                            content_parts.append({"type": "image_url", "image_url": {"url": image_uri}})
                        except Exception as e_img:
                            ASCIIColors.error(f"Failed to process image {img_path} for LLaVA: {e_img}")
                    user_message_content = content_parts # type: ignore
                
                messages.append({"role": "user", "content": user_message_content})

                response_iter = self.model.create_chat_completion(messages=messages, stream=stream, **gen_params)
                
                if stream:
                    for chunk in response_iter:
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        chunk_content = delta.get('content', '')
                        if chunk_content:
                            full_response_text += chunk_content
                            if streaming_callback and not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break 
                    return full_response_text
                else: # Not streaming (response_iter is a single dict)
                    return response_iter.get('choices', [{}])[0].get('message', {}).get('content', '')
            else: # Raw completion
                full_raw_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
                response_iter = self.model.create_completion(prompt=full_raw_prompt, stream=stream, **gen_params)
                if stream:
                    for chunk in response_iter:
                        chunk_content = chunk.get('choices', [{}])[0].get('text', '')
                        if chunk_content:
                            full_response_text += chunk_content
                            if streaming_callback and not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                    return full_response_text
                else:
                    return response_iter.get('choices', [{}])[0].get('text', '')

        except Exception as ex:
            error_message = f"Llama.cpp generation error: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}
    
    def tokenize(self, text: str) -> List[int]:
        if not self.model:
            ASCIIColors.warning("Llama.cpp model not loaded. Tokenization fallback to tiktoken.")
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text) 
        return self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False) 
            
    def detokenize(self, tokens: List[int]) -> str:
        if not self.model:
            ASCIIColors.warning("Llama.cpp model not loaded. Detokenization fallback to tiktoken.")
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)
        try:
            return self.model.detokenize(tokens).decode("utf-8", errors="ignore")
        except Exception: # Fallback if detokenize gives non-utf8 bytes
            return self.model.detokenize(tokens).decode("latin-1", errors="ignore")

    def count_tokens(self, text: str) -> int:
        if not self.model:
            ASCIIColors.warning("Llama.cpp model not loaded. Token count fallback to tiktoken.")
            return len(tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text))
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        if not self.model:
             raise Exception("Llama.cpp model not loaded.")
        if not self.llama_config.get("embedding"): # or not self.model.params.embedding:
            raise Exception("Embedding support was not enabled when loading the model (set 'embedding: true' in config).")
        try:
            return self.model.embed(text)
        except Exception as ex:
            trace_exception(ex); raise Exception(f"Llama.cpp embedding failed: {str(ex)}") from ex
        
    def get_model_info(self) -> dict:
        if not self.model:
            return {
                "name": self.binding_name, "model_path": self.model_path, "loaded": False,
                "error": "Model not loaded or failed to load."
            }
        
        is_llava_model = "llava" in self.model_path.lower() or \
                         (self.llama_config.get("chat_format", "").startswith("llava") and \
                          self.llama_config.get("clip_model_path") is not None)

        return {
            "name": self.binding_name, "model_path": self.model_path, "loaded": True,
            "n_ctx": self.model.context_params.n_ctx,
            "n_gpu_layers": self.llama_config.get("n_gpu_layers"),
            "seed": self.llama_config.get("seed"),
            "supports_structured_output": self.grammar is not None or self.llama_config.get("grammar_file") is not None,
            "supports_vision": is_llava_model and LlamaChatCompletionRequestMessageImageContentPart is not None,
            "config": self.llama_config 
        }

    def listModels(self) -> List[Dict[str, str]]: # type: ignore
        # This method is more for server-based bindings. For LlamaCpp, it describes the loaded model.
        # It could be extended to scan lollms_paths for GGUF files.
        if self.model:
            return [{
                'model_name': os.path.basename(self.model_path), 'path': self.model_path, 'loaded': True,
                'n_ctx': str(self.model.context_params.n_ctx), 
                'n_gpu_layers': str(self.llama_config.get("n_gpu_layers","N/A")),
            }]
        return [{'model_name': os.path.basename(self.model_path) if self.model_path else "Not specified", 
                 'path': self.model_path, 'loaded': False, 'error': "Model not loaded."}]
    
    def unload_model(self):
        if self.model:
            del self.model 
            self.model = None
            ASCIIColors.info("Llama.cpp model unloaded.")
            # In Python, explicit memory freeing for C extensions can be tricky.
            # `del self.model` removes the Python reference. If llama.cpp's Llama class
            # has a proper __del__ method (it does), it should free its C resources.
            # Forcing GC might help, but not guaranteed immediate effect.
            # import gc; gc.collect() 

    def __del__(self):
        self.unload_model()


if __name__ == '__main__':
    global full_streamed_text
    ASCIIColors.yellow("Testing PythonLlamaCppBinding...")

    # --- IMPORTANT: Configure model path ---
    # Replace with the ACTUAL PATH to your GGUF model file.
    # e.g., gguf_model_path = "C:/Models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"
    # If this path is not found, a dummy GGUF will be created for basic tests.
    gguf_model_path = "model.gguf" # <<< REPLACE THIS OR ENSURE 'model.gguf' EXISTS

    # --- LLaVA Test Configuration (Optional) ---
    # To test LLaVA, set this to your LLaVA GGUF model path
    llava_test_model_path = None # e.g., "path/to/your/llava-v1.6-mistral-7b.Q4_K_M.gguf"
    # And the corresponding mmproj (clip model) GGUF path
    llava_test_clip_model_path = None # e.g., "path/to/your/mmproj-mistral7b-f16.gguf"
    # And set the chat format for LLaVA
    llava_chat_format = "llava-1-6" # or "llava-1-5" depending on your model

    # Attempt to create a dummy GGUF if specified path doesn't exist (for placeholder testing)
    is_dummy_model = False
    if not os.path.exists(gguf_model_path):
        ASCIIColors.warning(f"Model path '{gguf_model_path}' not found.")
        ASCIIColors.warning("Creating a tiny dummy GGUF file ('dummy_model.gguf') for placeholder testing.")
        ASCIIColors.warning("This dummy file WILL NOT WORK for actual inference.")
        try:
            with open("dummy_model.gguf", "wb") as f: # Minimal valid GGUF structure
                f.write(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00")
                key, val = "tokenizer.ggml.model", "llama"
                f.write(len(key).to_bytes(8,'little')+key.encode()+b"\x02\x00\x00\x00\x00\x00\x00\x00"+len(val).to_bytes(8,'little')+val.encode())
            gguf_model_path = "dummy_model.gguf"
            is_dummy_model = True
            ASCIIColors.info(f"Using dummy GGUF: {gguf_model_path}. Real inference tests will fail or be skipped.")
        except Exception as e_dummy:
            ASCIIColors.error(f"Could not create dummy GGUF: {e_dummy}. Please set a valid GGUF model path.")
            sys.exit(1)

    binding_config = {
        "n_gpu_layers": 0, # 0 for CPU, -1 for all possible layers to GPU, or specific number
        "n_ctx": 1024,     # Short context for testing
        "seed": 1234,
        "embedding": True, # Enable embedding generation for the test
        "verbose": False,  # Set to True for detailed llama.cpp logs
        # "grammar_file": "json.gbnf" # Example for grammar test
    }
    
    mock_lollms_paths = { "personal_models_path": ".", "grammars_path": "grammars_test" }
    if not os.path.exists(mock_lollms_paths["grammars_path"]):
        os.makedirs(mock_lollms_paths["grammars_path"], exist_ok=True)
    with open(os.path.join(mock_lollms_paths["grammars_path"], "test_grammar.gbnf"), "w") as f:
        f.write('root ::= ("hello" | "world")')

    active_binding = None
    try:
        ASCIIColors.cyan("\n--- Initializing PythonLlamaCppBinding ---")
        active_binding = PythonLlamaCppBinding(model_path=gguf_model_path, config=binding_config, lollms_paths=mock_lollms_paths)
        ASCIIColors.green(f"Binding initialized. Model: {active_binding.model_path}")
        ASCIIColors.info(f"Model Info: {json.dumps(active_binding.get_model_info(), indent=2)}")

        if is_dummy_model:
            ASCIIColors.warning("\nRUNNING WITH DUMMY MODEL. MOST FUNCTIONALITY TESTS WILL BE SKIPPED.")
        else:
            # --- List Models ---
            ASCIIColors.cyan("\n--- Listing Models ---")
            print(json.dumps(active_binding.listModels(), indent=2))

            # --- Tokenize/Detokenize ---
            ASCIIColors.cyan("\n--- Tokenize/Detokenize ---")
            sample_text = "Hello, Llama.cpp world! This is a test sentence."
            tokens = active_binding.tokenize(sample_text)
            ASCIIColors.green(f"Tokens for '{sample_text}': {tokens[:15]}...")
            token_count = active_binding.count_tokens(sample_text)
            ASCIIColors.green(f"Token count: {token_count}")
            detokenized_text = active_binding.detokenize(tokens)
            ASCIIColors.green(f"Detokenized text: {detokenized_text}")
            assert detokenized_text.strip() == sample_text.strip(), "Tokenization/Detokenization mismatch!"

            # --- Text Generation (Non-Streaming, Chat Format) ---
            ASCIIColors.cyan("\n--- Text Generation (Non-Streaming, Chat) ---")
            prompt_text = "What is the capital of France?"
            system_prompt_text = "You are a helpful geography expert."
            generated_text = active_binding.generate_text(
                prompt_text, system_prompt=system_prompt_text, n_predict=30, stream=False, use_chat_format=True
            )
            if isinstance(generated_text, str): ASCIIColors.green(f"Generated text: {generated_text}")
            else: ASCIIColors.error(f"Generation failed: {generated_text}")

            # --- Text Generation (Streaming, Chat Format) ---
            ASCIIColors.cyan("\n--- Text Generation (Streaming, Chat) ---")
            full_streamed_text = ""
            def stream_callback(chunk: str, msg_type: int):
                global full_streamed_text; print(f"{ASCIIColors.GREEN}{chunk}{ASCIIColors.RESET}", end="", flush=True)
                full_streamed_text += chunk; return True
            
            result = active_binding.generate_text(
                prompt_text, system_prompt=system_prompt_text, n_predict=50, stream=True, 
                streaming_callback=stream_callback, use_chat_format=True
            )
            print("\n--- End of Stream ---")
            if isinstance(result, str): ASCIIColors.green(f"Full streamed text: {result}")
            else: ASCIIColors.error(f"Streaming generation failed: {result}")
            
            # --- Text Generation with Grammar ---
            ASCIIColors.cyan("\n--- Text Generation with Grammar ---")
            generated_grammar_text = active_binding.generate_text(
                "Output a greeting:", n_predict=5, stream=False, use_chat_format=False, # Grammar often better with raw completion
                grammar=os.path.join(mock_lollms_paths["grammars_path"], "test_grammar.gbnf")
            )
            if isinstance(generated_grammar_text, str):
                ASCIIColors.green(f"Generated text with grammar: '{generated_grammar_text.strip()}'")
                assert generated_grammar_text.strip().lower() in ["hello", "world"], "Grammar constraint failed!"
            else: ASCIIColors.error(f"Grammar generation failed: {generated_grammar_text}")

            # --- Embeddings ---
            if binding_config.get("embedding"):
                ASCIIColors.cyan("\n--- Embeddings ---")
                embedding_text = "This is a test for embeddings."
                try:
                    embedding_vector = active_binding.embed(embedding_text)
                    ASCIIColors.green(f"Embedding for '{embedding_text}' (first 3 dims): {embedding_vector[:3]}...")
                    ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
                except Exception as e_emb: ASCIIColors.warning(f"Could not get embedding: {e_emb}")
            else: ASCIIColors.yellow("\n--- Embeddings Skipped (embedding: false in config) ---")

        # --- LLaVA Test (if configured and real model is LLaVA) ---
        if not is_dummy_model and llava_test_model_path and os.path.exists(llava_test_model_path) and \
           llava_test_clip_model_path and os.path.exists(llava_test_clip_model_path) and \
           active_binding and active_binding.model_path.lower() == llava_test_model_path.lower():
            
            ASCIIColors.cyan("\n--- LLaVA Vision Test ---")
            # This assumes the 'active_binding' was ALREADY loaded with the LLaVA model
            # and its specific config (clip_model_path, chat_format="llava-1-x").
            # If not, you'd need to unload and reload/reinitialize the binding for LLaVA.
            if not (active_binding.llama_config.get("chat_format","").startswith("llava") and \
                    active_binding.llama_config.get("clip_model_path")):
                ASCIIColors.warning("Current binding not configured for LLaVA. Skipping LLaVA test.")
                ASCIIColors.warning("To test LLaVA, ensure gguf_model_path points to LLaVA model and config includes 'chat_format' and 'clip_model_path'.")
            else:
                dummy_image_path = "dummy_llava_image.png"
                try:
                    from PIL import Image, ImageDraw
                    img = Image.new('RGB', (200, 80), color = ('cyan'))
                    d = ImageDraw.Draw(img); d.text((10,20), "LLaVA Test", fill=('black'))
                    img.save(dummy_image_path)
                    ASCIIColors.info(f"Created dummy image for LLaVA: {dummy_image_path}")

                    llava_prompt = "What do you see in this image?"
                    llava_response = active_binding.generate_text(
                        prompt=llava_prompt, images=[dummy_image_path], n_predict=50, stream=False, use_chat_format=True
                    )
                    if isinstance(llava_response, str): ASCIIColors.green(f"LLaVA response: {llava_response}")
                    else: ASCIIColors.error(f"LLaVA generation failed: {llava_response}")
                except ImportError: ASCIIColors.warning("Pillow not found. Cannot create dummy image for LLaVA.")
                except Exception as e_llava: ASCIIColors.error(f"LLaVA test error: {e_llava}"); trace_exception(e_llava)
                finally:
                    if os.path.exists(dummy_image_path): os.remove(dummy_image_path)
        elif not is_dummy_model and llava_test_model_path: # If LLaVA test paths are set but model isn't LLaVA
             ASCIIColors.yellow(f"LLaVA test paths are set, but current model '{active_binding.model_path if active_binding else 'N/A'}' is not '{llava_test_model_path}'.")
             ASCIIColors.yellow("Skipping LLaVA-specific test section. To run, set main gguf_model_path to LLaVA model and configure LLaVA params.")


    except ImportError as e_imp:
        ASCIIColors.error(f"Import error: {e_imp}. Llama-cpp-python might not be installed/configured correctly.")
    except FileNotFoundError as e_fnf:
        ASCIIColors.error(f"Model file error: {e_fnf}. Ensure GGUF model path is correct.")
    except RuntimeError as e_rt: 
        ASCIIColors.error(f"Runtime error (often model load failure or llama.cpp issue): {e_rt}")
        trace_exception(e_rt)
    except Exception as e_main:
        ASCIIColors.error(f"An unexpected error occurred: {e_main}")
        trace_exception(e_main)
    finally:
        if active_binding:
            ASCIIColors.cyan("\n--- Unloading Model ---")
            active_binding.unload_model()
            ASCIIColors.green("Model unloaded.")
        
        if is_dummy_model and os.path.exists("dummy_model.gguf"):
            os.remove("dummy_model.gguf")
        
        test_grammar_file = os.path.join(mock_lollms_paths["grammars_path"], "test_grammar.gbnf")
        if os.path.exists(test_grammar_file): os.remove(test_grammar_file)
        if os.path.exists(mock_lollms_paths["grammars_path"]) and not os.listdir(mock_lollms_paths["grammars_path"]):
            os.rmdir(mock_lollms_paths["grammars_path"])

    ASCIIColors.yellow("\nPythonLlamaCppBinding test finished.")