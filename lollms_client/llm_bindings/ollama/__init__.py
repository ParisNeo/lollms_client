# bindings/ollama/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
# encode_image is not strictly needed if ollama-python handles paths, but kept for consistency if ever needed.
# from lollms_client.lollms_utilities import encode_image 
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from typing import Optional, Callable, List, Union, Dict

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm
from lollms_client.lollms_utilities import ImageTokenizer
pm.ensure_packages(["ollama","pillow","tiktoken"])


import ollama
import tiktoken
BindingName = "OllamaBinding"


def count_tokens_ollama(
    text_to_tokenize: str,
    model_name: str,
    ollama_client: ollama.Client,
) -> int:
    """
    Counts the number of tokens in a given text for a specified Ollama model
    by making a minimal request to the /api/generate endpoint and extracting
    the 'prompt_eval_count' from the response.

    This method is generally more accurate for the specific Ollama model instance
    than using an external tokenizer, but it incurs the overhead of an API call
    and model processing for the prompt.

    Args:
        text_to_tokenize: The string to tokenize.
        model_name: The name of the Ollama model (e.g., "llama3:8b", "mistral").
        ollama_host: The URL of the Ollama API host.
        timeout: Timeout for the request to Ollama.
        verify_ssl_certificate: Whether to verify SSL certificates for the Ollama host.
        headers: Optional custom headers for the request to Ollama.
        num_predict_for_eval: How many tokens to ask the model to "predict" to get
                              the prompt evaluation count. 0 is usually sufficient and most efficient.
                              If 0 doesn't consistently yield `prompt_eval_count`, try 1.

    Returns:
        The number of tokens as reported by 'prompt_eval_count'.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        KeyError: If 'prompt_eval_count' is not found in the response.
        json.JSONDecodeError: If the response is not valid JSON.
        RuntimeError: For other operational errors.
    """
    res = ollama_client.chat(
                        model=model_name,
                        messages=[{"role":"system","content":""},{"role":"user", "content":text_to_tokenize}],
                        stream=False,options={"num_predict":1}                        
                    )
    
    return res.prompt_eval_count-5
class OllamaBinding(LollmsLLMBinding):
    """Ollama-specific binding implementation using the ollama-python library."""
    
    DEFAULT_HOST_ADDRESS = "http://localhost:11434"
    
    def __init__(self,
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 **kwargs
                 ):
        """
        Initialize the Ollama binding.

        Args:
            host_address (str): Host address for the Ollama service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service (used in Authorization header). Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            default_completion_format (ELF_COMPLETION_FORMAT): Default completion format.
        """
        _host_address = host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS
        super().__init__(
            binding_name=BindingName, # Use the module-level BindingName
        )
        self.host_address=_host_address
        self.model_name=model_name
        self.service_key=service_key
        self.verify_ssl_certificate=verify_ssl_certificate
        self.default_completion_format=kwargs.get("default_completion_format",ELF_COMPLETION_FORMAT.Chat) 

        if ollama is None:
            raise ImportError("Ollama library is not installed. Please run 'pip install ollama'.")

        self.ollama_client_headers = {}
        if self.service_key:
            self.ollama_client_headers['Authorization'] = f'Bearer {self.service_key}'

        try:
            self.ollama_client = ollama.Client(
                host=self.host_address,
                headers=self.ollama_client_headers if self.ollama_client_headers else None,
                verify=self.verify_ssl_certificate # Passed to httpx.Client
            )
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize Ollama client: {e}")
            self.ollama_client = None # Ensure it's None if initialization fails
            # Optionally re-raise or handle so the binding is clearly unusable
            raise ConnectionError(f"Could not connect or initialize Ollama client at {self.host_address}: {e}") from e

    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: float = 0.7, # Ollama default is 0.8, common default 0.7
                     top_k: int = 40,          # Ollama default is 40
                     top_p: float = 0.9,       # Ollama default is 0.9
                     repeat_penalty: float = 1.1, # Ollama default is 1.1
                     repeat_last_n: int = 64,  # Ollama default is 64
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """

        if not self.ollama_client:
             return {"status": False, "error": "Ollama client not initialized."}

        options = {}
        if n_predict is not None: options['num_predict'] = n_predict
        if temperature is not None: options['temperature'] = float(temperature)
        if top_k is not None: options['top_k'] = top_k
        if top_p is not None: options['top_p'] = top_p
        if repeat_penalty is not None: options['repeat_penalty'] = repeat_penalty
        if repeat_last_n is not None: options['repeat_last_n'] = repeat_last_n
        if seed is not None: options['seed'] = seed
        if n_threads is not None: options['num_thread'] = n_threads
        if ctx_size is not None: options['num_ctx'] = ctx_size
        
        full_response_text = ""

        try:
            if images: # Multimodal
                # ollama-python expects paths or bytes for images
                processed_images = []
                for img_path in images:
                    # Assuming img_path is a file path. ollama-python will read and encode it.
                    # If images were base64 strings, they would need decoding to bytes first.
                    processed_images.append(img_path)

                messages = [
                            {'role': 'system', 'content':system_prompt},
                        ]
                if split:
                    messages += self.split_discussion(prompt,user_keyword=user_keyword, ai_keyword=ai_keyword)
                    if processed_images:
                        messages[-1]["images"]=processed_images
                else:
                    messages.append({'role': 'user', 'content': prompt, 'images': processed_images if processed_images else None})
                if stream:
                    response_stream = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=True,
                        options=options if options else None
                    )
                    for chunk_dict in response_stream:
                        chunk_content = chunk_dict.get('message', {}).get('content', '')
                        if chunk_content: # Ensure there is content to process
                            full_response_text += chunk_content
                            if streaming_callback:
                                if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break # Callback requested stop
                    return full_response_text
                else: # Not streaming
                    response_dict = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        options=options if options else None
                    )
                    return response_dict.get('message', {}).get('content', '')
            else: # Text-only
                messages = [
                            {'role': 'system', 'content':system_prompt},
                        ]
                if split:
                    messages += self.split_discussion(prompt,user_keyword=user_keyword, ai_keyword=ai_keyword)
                else:
                    messages.append({'role': 'user', 'content': prompt})

                if stream:
                    response_stream = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=True,
                        options=options if options else None
                    )
                    for chunk_dict in response_stream:
                        chunk_content = chunk_dict.message.content
                        if chunk_content:
                            full_response_text += chunk_content
                            if streaming_callback:
                                if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break
                    return full_response_text
                else: # Not streaming
                    response_dict = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        options=options if options else None
                    )
                    return response_dict.message.content
        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message, "status_code": e.status_code}
        except ollama.RequestError as e: # Covers connection errors, timeouts during request
            error_message = f"Ollama API RequestError: {str(e)}"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}

    def generate_from_messages(self,
                     messages: List[Dict],
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repeat_penalty: Optional[float] = None,
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     **kwargs
                     ) -> Union[str, dict]:
        if not self.ollama_client:
             return {"status": False, "error": "Ollama client not initialized."}

        options = {}
        if n_predict is not None: options['num_predict'] = n_predict
        if temperature is not None: options['temperature'] = float(temperature)
        if top_k is not None: options['top_k'] = top_k
        if top_p is not None: options['top_p'] = top_p
        if repeat_penalty is not None: options['repeat_penalty'] = repeat_penalty
        if repeat_last_n is not None: options['repeat_last_n'] = repeat_last_n
        if seed is not None: options['seed'] = seed
        if n_threads is not None: options['num_thread'] = n_threads
        if ctx_size is not None: options['num_ctx'] = ctx_size
        
        full_response_text = ""

        try:
            if stream:
                response_stream = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options=options if options else None
                )
                for chunk_dict in response_stream:
                    chunk_content = chunk_dict.get('message', {}).get('content', '')
                    if chunk_content: # Ensure there is content to process
                        full_response_text += chunk_content
                        if streaming_callback:
                            if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break # Callback requested stop
                return full_response_text
            else: # Not streaming
                response_dict = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options=options if options else None
                )
                return response_dict.get('message', {}).get('content', '')

        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message, "status_code": e.status_code}
        except ollama.RequestError as e: # Covers connection errors, timeouts during request
            error_message = f"Ollama API RequestError: {str(e)}"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}
    

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: float = 0.7,
             top_k: int = 40,
             top_p: float = 0.9,
             repeat_penalty: float = 1.1,
             repeat_last_n: int = 64,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Ollama model using a LollmsDiscussion object.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p sampling parameter.
            repeat_penalty (float): Penalty for repeated tokens.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty.
            seed (Optional[int]): Random seed for generation.
            n_threads (Optional[int]): Number of threads to use.
            ctx_size (Optional[int]): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        if not self.ollama_client:
             return {"status": "error", "message": "Ollama client not initialized."}

        # 1. Export the discussion to the Ollama chat format
        # This handles system prompts, user/assistant roles, and base64-encoded images.
        messages = discussion.export("ollama_chat", branch_tip_id)

        # 2. Build the generation options dictionary
        options = {
            'num_predict': n_predict,
            'temperature': float(temperature),
            'top_k': top_k,
            'top_p': top_p,
            'repeat_penalty': repeat_penalty,
            'repeat_last_n': repeat_last_n,
            'seed': seed,
            'num_thread': n_threads,
            'num_ctx': ctx_size,
        }
        # Remove None values, as ollama-python expects them to be absent
        options = {k: v for k, v in options.items() if v is not None}

        full_response_text = ""

        try:
            # 3. Call the Ollama API
            if stream:
                response_stream = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options=options if options else None
                )
                for chunk in response_stream:
                    chunk_content = chunk.get('message', {}).get('content', '')
                    if chunk_content:
                        full_response_text += chunk_content
                        if streaming_callback:
                            if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else: # Not streaming
                response_dict = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options=options if options else None
                )
                return response_dict.get('message', {}).get('content', '')

        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            return {"status": "error", "message": error_message}
        except ollama.RequestError as e:
            error_message = f"Ollama API RequestError: {str(e)}"
            ASCIIColors.error(error_message)
            return {"status": "error", "message": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of individual characters.
        """
        ## Since ollama has no endpoints to tokenize the text, we use tiktoken to have a rough estimate
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens (characters) to detokenize.

        Returns:
            str: Detokenized text.
        """
        ## Since ollama has no endpoints to tokenize the text, we use tiktoken to have a rough estimate
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text using the Ollama server's /api/tokenize endpoint.

        Args:
            text (str): Text to count tokens from.

        Returns:
            int: Number of tokens in text. Returns -1 on error.
        """        
        if not self.model_name:
            ASCIIColors.warning("Cannot count tokens, model_name is not set.")
            return -1
        #return count_tokens_ollama(text, self.model_name, self.ollama_client)
        return len(self.tokenize(text))

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.

        Args:
            image (str): Image to count tokens from. Either base64 string, path to image file, or URL.

        Returns:
            int: Estimated number of tokens for the image. Returns -1 on error.
        """
        try:
            # Delegate token counting to ImageTokenizer
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e}")
            return -1

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text using Ollama API.
        
        Args:
            text (str): Input text to embed.
            **kwargs: Optional arguments. Can include 'model' to override self.model_name,
                      and 'options' dictionary for Ollama embedding options.
        
        Returns:
            List[float]: The embedding vector.
        
        Raises:
            Exception: if embedding fails or Ollama client is not available.
        """
        if not self.ollama_client:
             raise Exception("Ollama client not initialized.")

        model_to_use = kwargs.get("model", "bge-m3")
        if not model_to_use:
            raise ValueError("Model name for embedding must be specified either in init or via kwargs.")
            
        ollama_options = kwargs.get("options", None)
        try:
            response = self.ollama_client.embeddings(
                model=model_to_use, 
                prompt=text,
                options=ollama_options
            )
            return response['embedding']
        except ollama.ResponseError as e:
            error_message = f"Ollama API Embeddings ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            raise Exception(error_message) from e
        except ollama.RequestError as e:
            error_message = f"Ollama API Embeddings RequestError: {str(e)}"
            ASCIIColors.error(error_message)
            raise Exception(error_message) from e
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Embedding failed: {str(ex)}") from ex
        
    def get_model_info(self) -> dict:
        """
        Return information about the current Ollama model setup.

        Returns:
            dict: Dictionary containing binding name, version, host address, and model name.
        """
        return {
            "name": self.binding_name, # from super class
            "version": pm.get_installed_version("ollama") if ollama else "unknown", # Ollama library version
            "host_address": self.host_address,
            "model_name": self.model_name,
            "supports_structured_output": False, # Ollama primarily supports text/chat
            "supports_vision": True # Many Ollama models (e.g. llava, bakllava) support vision
        }

    def listModels(self) -> List[Dict[str, str]]:
        """
        Lists available models from the Ollama service using the ollama-python library.
        The returned list of dictionaries matches the format of the original template.
        
        Returns:
            List[Dict[str, str]]: A list of model information dictionaries.
                                  Each dict has 'model_name', 'owned_by', 'created_datetime'.
        """
        if not self.ollama_client:
            ASCIIColors.error("Ollama client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug(f"Listing ollama models from {self.host_address}")
            response_data = self.ollama_client.list() # This returns {'models': [{'name':..., 'modified_at':..., ...}]}
            
            model_info_list = []
            if 'models' in response_data:
                for model_entry in response_data['models']:
                    model_info_list.append({
                        'model_name': model_entry.get('model'),
                        'owned_by': "", # Ollama API doesn't provide a direct "owned_by" field.
                        'created_datetime': model_entry.get('modified_at') 
                    })
            return model_info_list
        except ollama.ResponseError as e:
            ASCIIColors.error(f"Ollama API listModels ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code}) from {self.host_address}")
            return []
        except ollama.RequestError as e: # Covers connection errors, timeouts during request
            ASCIIColors.error(f"Ollama API listModels RequestError: {str(e)} from {self.host_address}")
            return []
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """
        Set the model name for subsequent operations. Ollama loads models on demand.
        This method can be used to verify if a model exists by attempting a small operation,
        but for now, it just sets the name.

        Args:
            model_name (str): Name of the model to set.

        Returns:
            bool: True if model name is set.
        """
        self.model_name = model_name
        # Optionally, you could try a quick self.ollama_client.show(model_name) to verify existence.
        # For simplicity, we just set it.
        ASCIIColors.info(f"Ollama model set to: {model_name}. It will be loaded by the server on first use.")
        return True

if __name__ == '__main__':
    global full_streamed_text
    # Example Usage (requires an Ollama server running)
    ASCIIColors.yellow("Testing OllamaBinding...")

    # --- Configuration ---
    # Replace with your Ollama server details if not localhost
    ollama_host = "http://localhost:11434" 
    # Common model, pull it first: `ollama pull llama3` or `ollama pull llava` for vision
    test_model_name = "llama3" 
    test_vision_model_name = "llava" # or another vision model you have

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = OllamaBinding(host_address=ollama_host, model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using Ollama client version: {ollama.__version__ if ollama else 'N/A'}")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.listModels()
        if models:
            ASCIIColors.green(f"Found {len(models)} models. First 5:")
            for m in models[:5]:
                print(m)
        else:
            ASCIIColors.warning("No models found or failed to list models. Ensure Ollama is running and has models.")

        # --- Load Model (sets active model) ---
        ASCIIColors.cyan(f"\n--- Setting model to: {test_model_name} ---")
        binding.load_model(test_model_name)

        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        # --- Tokenize/Detokenize (using server for tokenize) ---
        ASCIIColors.cyan("\n--- Tokenize/Detokenize ---")
        tokens = binding.tokenize(sample_text)
        ASCIIColors.green(f"Tokens for '{sample_text}': {tokens[:10]}...") # Print first 10
        detokenized_text = binding.detokenize(tokens)
        # Note: detokenize might not be perfect if tokens are IDs and not chars
        ASCIIColors.green(f"Detokenized text (may vary based on tokenization type): {detokenized_text}")


        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Why is the sky blue?"
        ASCIIColors.info(f"Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, n_predict=50, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text: {generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            global full_streamed_text
            print(f"{ASCIIColors.GREEN}Stream chunk: {chunk}{ASCIIColors.RESET}", end="", flush=True)
            full_streamed_text += chunk
            if len(full_streamed_text) > 100: # Example: stop after 100 chars for test
                # print("\nStopping stream early for test.")
                # return False # uncomment to test early stop
                pass
            return True
        
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, n_predict=100, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        if isinstance(result, str):
             ASCIIColors.green(f"Full streamed text: {result}") # 'result' is the full_streamed_text
        else:
            ASCIIColors.error(f"Streaming generation failed: {result}")


        # --- Embeddings ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        # Ensure you have an embedding model like 'mxbai-embed-large' or 'nomic-embed-text'
        # Or use a general model if it supports embedding (some do implicitly)
        # For this test, we'll try with the current test_model_name, 
        # but ideally use a dedicated embedding model.
        # binding.load_model("mxbai-embed-large") # if you have it
        try:
            embedding_text = "Lollms is a cool project."
            embedding_vector = binding.embed(embedding_text) # Uses current self.model_name
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.warning(f"Could not get embedding with '{binding.model_name}': {e}. Some models don't support /api/embeddings or may need to be specified.")
            ASCIIColors.warning("Try `ollama pull mxbai-embed-large` and set it as model for embedding.")


        # --- Vision Model Test (if llava or similar is available) ---
        # Create a dummy image file for testing
        dummy_image_path = "dummy_test_image.png"
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 30), color = ('red'))
            d = ImageDraw.Draw(img)
            d.text((10,10), "Hello", fill=('white'))
            img.save(dummy_image_path)
            ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

            ASCIIColors.cyan(f"\n--- Vision Generation (using {test_vision_model_name}) ---")
            # Check if vision model exists
            vision_model_exists = any(m['model_name'].startswith(test_vision_model_name) for m in models)
            if not vision_model_exists:
                ASCIIColors.warning(f"Vision model '{test_vision_model_name}' not found in pulled models. Skipping vision test.")
                ASCIIColors.warning(f"Try: `ollama pull {test_vision_model_name}`")
            else:
                binding.load_model(test_vision_model_name) # Switch to vision model
                vision_prompt = "What is written in this image?"
                ASCIIColors.info(f"Vision Prompt: {vision_prompt} with image {dummy_image_path}")
                
                vision_response = binding.generate_text(
                    prompt=vision_prompt,
                    images=[dummy_image_path],
                    n_predict=50,
                    stream=False
                )
                if isinstance(vision_response, str):
                    ASCIIColors.green(f"Vision model response: {vision_response}")
                else:
                    ASCIIColors.error(f"Vision generation failed: {vision_response}")
        except ImportError:
            ASCIIColors.warning("Pillow library not found. Cannot create dummy image for vision test. `pip install Pillow`")
        except Exception as e:
            ASCIIColors.error(f"Error during vision test: {e}")
        finally:
            import os
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)


    except ConnectionRefusedError:
        ASCIIColors.error("Connection to Ollama server refused. Is Ollama running?")
    except ImportError as e:
        ASCIIColors.error(f"Import error: {e}. Make sure 'ollama' library is installed ('pip install ollama').")
    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nOllamaBinding test finished.")
