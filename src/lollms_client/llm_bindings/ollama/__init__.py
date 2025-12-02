# bindings/ollama/__init__.py
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
import re

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
                        stream=False,
                        think=False,
                        options={"num_predict":1}                        
                    )
    
    return res.prompt_eval_count-5
class OllamaBinding(LollmsLLMBinding):
    """Ollama-specific binding implementation using the ollama-python library."""
    
    DEFAULT_HOST_ADDRESS = "http://localhost:11434"
    
    def __init__(self,
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
        host_address = kwargs.get("host_address")
        _host_address = host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS
        super().__init__(BindingName, **kwargs)
        self.host_address=_host_address
        self.model_name=kwargs.get("model_name")
        self.service_key=kwargs.get("service_key")
        self.verify_ssl_certificate=kwargs.get("verify_ssl_certificate", True)
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
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[bool] = "low", # low, medium, high
                    reasoning_summary: Optional[bool] = "auto", # auto
                    **kwargs
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
        think = think if "gpt-oss" not in self.model_name else reasoning_effort
        ASCIIColors.magenta(f"Generation with think: {think}")

        try:
            if images: # Multimodal
                # ollama-python expects paths or bytes for images
                processed_images = []
                for img_path in images:
                    # Assuming img_path is a file path. ollama-python will read and encode it.
                    # If images were base64 strings, they would need decoding to bytes first.
                    if img_path.startswith("data:image/png;base64,"):
                        img_path = img_path[len("data:image/png;base64,"):]
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
                        think=think,
                        options=options if options else None
                    )
                    in_thinking = False
                    for chunk in response_stream:
                        if chunk.message.thinking and not in_thinking:
                            full_response_text += "<think>\n"
                            in_thinking = True
                            
                        if chunk.message.content:# Ensure there is content to process
                            chunk_content = chunk.message.content
                            if in_thinking:
                                full_response_text += "\n</think>\n"                            
                                in_thinking = False
                            full_response_text += chunk_content
                            if streaming_callback:
                                if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break # Callback requested stop
                    return full_response_text
                else: # Not streaming
                    response = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        think=think,
                        options=options if options else None
                    )
                    full_response_text = response.message.content
                    if think:
                        full_response_text = "<think>\n"+response.message.thinking+"\n</think>\n"+full_response_text
                    return full_response_text
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
                        think=think,
                        options=options if options else None
                    )
                    in_thinking = False
                    for chunk in response_stream:
                        if chunk.message.thinking and not in_thinking:
                            full_response_text += "<think>\n"
                            in_thinking = True
                            
                        if chunk.message.content:# Ensure there is content to process
                            chunk_content = chunk.message.content
                            if in_thinking:
                                full_response_text += "\n</think>\n"                            
                                in_thinking = False
                            full_response_text += chunk_content
                            if streaming_callback:
                                if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break # Callback requested stop
                    return full_response_text
                else: # Not streaming
                    response = self.ollama_client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        think=think,
                        options=options if options else None
                    )
                    full_response_text = response.message.content
                    if think:
                        full_response_text = "<think>\n"+response.message.thinking+"\n</think>\n"+full_response_text
                    return full_response_text
                    
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
                        think: Optional[bool] = False,
                        reasoning_effort: Optional[bool] = "low", # low, medium, high
                        reasoning_summary: Optional[bool] = "auto", # auto
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

        def normalize_message(msg: Dict) -> Dict:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts = []
            images = []

            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "input_image" or  item.get("type") == "image_url":
                        base64_data = item.get("image_url")
                        if base64_data:
                            if isinstance(base64_data, str):
                                # ⚠️ remove prefix "data:image/...;base64,"
                                cleaned = re.sub(r"^data:image/[^;]+;base64,", "", base64_data)
                                images.append(cleaned)
                            elif base64_data and isinstance(base64_data, dict) :
                                if "base64" in base64_data:
                                    cleaned = re.sub(r"^data:image/[^;]+;base64,", "", base64_data["base64"])
                                    images.append(cleaned)
                                elif "url" in base64_data :
                                    if "http" in base64_data["url"]:
                                        images.append(base64_data["url"])
                                    else:
                                        cleaned = re.sub(r"^data:image/[^;]+;base64,", "", base64_data["url"])
                                        images.append(cleaned)


            return {
                "role": role,
                "content": "\n".join([p for p in text_parts if p.strip()]),
                "images": images if images else None
            }

        ollama_messages = []
        for m in messages:
            nm = normalize_message(m)
            if nm["images"]:
                ollama_messages.append({
                    "role": nm["role"],
                    "content": nm["content"],
                    "images": nm["images"]
                })
            else:
                ollama_messages.append({
                    "role": nm["role"],
                    "content": nm["content"]
                })

        full_response_text = ""

        try:
            if stream:
                response_stream = self.ollama_client.chat(
                    model=self.model_name,
                    messages=ollama_messages,
                    stream=True,
                    think = think,
                    options=options if options else None
                )
                for chunk_dict in response_stream:
                    chunk_content = chunk_dict.get('message', {}).get('content', '')
                    if chunk_content:
                        full_response_text += chunk_content
                        if streaming_callback:
                            if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=ollama_messages,
                    stream=False,
                    think=think if "gpt-oss" not in self.model_name else reasoning_effort,
                    options=options if options else None
                )
                full_response_text = response.message.content
                if think:
                    full_response_text = "<think>\n"+response.message.thinking+"\n</think>\n"+full_response_text
                return full_response_text

        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message, "status_code": e.status_code}
        except ollama.RequestError as e:
            error_message = f"Ollama API RequestError: {str(e)}"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}


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
            streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
            think: Optional[bool] = False,
            reasoning_effort: Optional[bool] = "low", # low, medium, high
            reasoning_summary: Optional[bool] = "auto", # auto
            **kwargs
            
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
            'temperature': float(temperature) if temperature else None,
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
        think = think if "gpt-oss" not in self.model_name else reasoning_effort
        ASCIIColors.magenta(f"Generation with think: {think}")

        try:
            # 3. Call the Ollama API
            if stream:
                response_stream = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    think=think,
                    options=options if options else None
                )
                in_thinking = False
                for chunk in response_stream:
                    if chunk.message.thinking and not in_thinking:
                        full_response_text += "<think>\n"
                        in_thinking = True
                        
                    if chunk.message.content:# Ensure there is content to process
                        chunk_content = chunk.message.content
                        if in_thinking:
                            full_response_text += "\n</think>\n"                            
                            in_thinking = False
                        full_response_text += chunk_content
                        if streaming_callback:
                            if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break # Callback requested stop

                return full_response_text
            else: # Not streaming
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    think=think,
                    options=options if options else None
                )
                full_response_text = response.message.content
                if think:
                    full_response_text = "<think>\n"+response.message.thinking+"\n</think>\n"+full_response_text
                return full_response_text

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
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text, disallowed_special=())
            
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

    def pull_model(self, model_name: str, progress_callback: Callable[[dict], None] = None, **kwargs) -> bool:
        """
        Pulls a model from the Ollama library.

        Args:
            model_name (str): The name of the model to pull.
            progress_callback (Callable[[dict], None], optional): A callback function that receives progress updates. 
                                                                  The dict typically contains 'status', 'completed', 'total'.

        Returns:
            bool: True if the model was pulled successfully, False otherwise.
        """
        if not self.ollama_client:
             ASCIIColors.error("Ollama client not initialized. Cannot pull model.")
             return False

        try:
            ASCIIColors.info(f"Pulling model {model_name}...")
            # Stream the pull progress
            for progress in self.ollama_client.pull(model_name, stream=True):
                # Send raw progress to callback if provided
                if progress_callback:
                    progress_callback(progress)
                
                # Default console logging
                status = progress.get('status', '')
                completed = progress.get('completed')
                total = progress.get('total')
                
                if completed and total:
                    percent = (completed / total) * 100
                    print(f"\r{status}: {percent:.2f}%", end="", flush=True)
                else:
                     print(f"\r{status}", end="", flush=True)
            
            print() # Clear line
            ASCIIColors.success(f"Model {model_name} pulled successfully.")
            return True

        except ollama.ResponseError as e:
            ASCIIColors.error(f"Ollama API Pull Error: {e.error or 'Unknown error'} (status code: {e.status_code})")
            return False
        except ollama.RequestError as e:
            ASCIIColors.error(f"Ollama API Request Error: {str(e)}")
            return False
        except Exception as ex:
            ASCIIColors.error(f"An unexpected error occurred while pulling model: {str(ex)}")
            trace_exception(ex)
            return False

    def list_models(self) -> List[Dict[str, str]]:
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
            ASCIIColors.error(f"Ollama API list_models ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code}) from {self.host_address}")
            return []
        except ollama.RequestError as e: # Covers connection errors, timeouts during request
            ASCIIColors.error(f"Ollama API list_models RequestError: {str(e)} from {self.host_address}")
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

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Retrieves the context size for an Ollama model.
        
        The effective context size is the `num_ctx` parameter if overridden in the Modelfile,
        otherwise it falls back to the model's default context length from its architecture details.
        As a final failsafe, uses a hardcoded list of known popular models' context lengths.
        """
        if model_name is None:
            model_name = self.model_name
            if not model_name:
                ASCIIColors.warning("Model name not specified and no default model set.")
                return None
            
        try:
            info = ollama.show(model_name)
            
            # Parse num_ctx from the 'parameters' string (e.g., "PARAMETER num_ctx 4096")
            parameters = info.get('parameters', '')
            num_ctx = None
            for param in parameters.split('\n'):
                if param.strip().startswith('num_ctx'):
                    num_ctx = int(param.split()[1])
                    break
            
            if num_ctx is not None:
                return num_ctx
            
            # Fall back to model_info context_length (e.g., 'llama.context_length')
            model_info = info.get('model_info', {})
            arch = model_info.get('general.architecture', '')
            context_key = f'{arch}.context_length' if arch else 'general.context_length'
            context_length = model_info.get(context_key)
            
            if context_length is not None:
                return int(context_length)
            
        except Exception as e:
            ASCIIColors.warning(f"Error fetching model info: {str(e)}")
        
        # Failsafe: Hardcoded context sizes for popular Ollama models
        known_contexts = {
            'llama2': 4096,       # Llama 2 default
            'llama3': 8192,       # Llama 3 default
            'llama3.1': 131072,   # Llama 3.1 extended context
            'llama3.2': 131072,   # Llama 3.2 extended context
            'llama3.3': 131072,   # Assuming similar to 3.1/3.2
            'gpt-oss:20b': 16000,     # GPT-OSS extended
            'gpt-oss:120b': 128000,     # GPT-OSS extended
            'codestral': 256000,  # Codestral
            'mistralai-medium': 128000,  # Mistral medium
            'mistralai-mini':   128000,  # Mistral medium
            'mistral': 32768,     # Mistral 7B v0.2+ default
            'mixtral': 32768,     # Mixtral 8x7B default
            'mixtral8x22b': 65536, # Mixtral 8x22B default
            'gemma': 8192,        # Gemma default
            'gemma2': 8192,       # Gemma 2 default
            'gemma3': 131072,     # Gemma 3 with 128K context
            'phi': 2048,          # Phi default (older)
            'phi2': 2048,         # Phi-2 default
            'phi3': 131072,       # Phi-3 variants often use 128K (mini/medium extended)
            'qwen': 8192,         # Qwen default
            'qwen2': 32768,       # Qwen2 default for 7B
            'qwen2.5': 131072,    # Qwen2.5 with 128K
            'qwen3': 128000,       # Qwen3 with 128k
            'qwen3-vl': 128000,       # Qwen3-vl with 128k
            'qwen3-coder': 256000, # Qwen3 with 256k
            'codellama': 16384,   # CodeLlama extended
            'codegemma': 8192,    # CodeGemma default
            'deepseek-coder': 16384,  # DeepSeek-Coder V1 default
            'deepseek-coder-v2': 131072,  # DeepSeek-Coder V2 with 128K
            'deepseek-llm': 4096,     # DeepSeek-LLM default
            'deepseek-v2': 131072,    # DeepSeek-V2 with 128K
            'yi': 4096,           # Yi base default
            'yi1.5': 32768,       # Yi-1.5 with 32K
            'command-r': 131072,  # Command-R with 128K
            'vicuna': 2048,       # Vicuna default (up to 16K in some variants)
            'wizardlm': 16384,    # WizardLM default
            'wizardlm2': 32768,   # WizardLM2 (Mistral-based)
            'zephyr': 65536,      # Zephyr beta (Mistral-based extended)
            'falcon': 2048,       # Falcon default
            'starcoder': 8192,    # StarCoder default
            'stablelm': 4096,     # StableLM default
            'orca': 4096,         # Orca default
            'orca2': 4096,        # Orca 2 default
            'dolphin': 32768,     # Dolphin (often Mistral-based)
            'openhermes': 8192,   # OpenHermes default
        }
        
        # Extract base model name (e.g., 'llama3' from 'llama3:8b-instruct')
        base_name = model_name.split(':')[0].lower().strip()
        
        if base_name in known_contexts:
            ASCIIColors.warning(f"Using hardcoded context size for model '{model_name}': {known_contexts[base_name]}")
            return known_contexts[base_name]
        
        ASCIIColors.warning(f"Context size not found for model '{model_name}'")
        return None

    def ps(self):
        """
        Lists running models in a standardized, flat format.

        This method corresponds to the /api/ps endpoint in the Ollama API. It retrieves
        the models currently loaded into memory and transforms the data into a simplified,
        flat list of dictionaries.

        Returns:
            list[dict]: A list of dictionaries, each representing a running model with a standardized set of keys.
                        Returns an empty list if the client is not initialized or if an error occurs.
        
        Example of a returned model dictionary:
        {
            "model_name": "gemma3:12b",
            "size": 13861175232,
            "vram_size": 10961479680,
            "parameters_size": "12.2B",
            "quantization_level": "Q4_K_M",
            "context_size": 32000,
            "parent_model": "",
            "expires_at": "2025-08-20T22:28:18.6708784+02:00"
        }
        """
        if not self.ollama_client:
            ASCIIColors.warning("Ollama client not initialized. Cannot list running models.")
            return []

        try:
            running_models_response = self.ollama_client.ps()
            
            models_list = running_models_response.get('models', [])
            standardized_models = []

            for model_data in models_list:
                details = model_data.get('details', {})
                
                flat_model_info = {
                    "model_name": model_data.get("name"),
                    "size": model_data.get("size"),
                    "vram_size": model_data.get("size_vram"),
                    "expires_at": model_data.get("expires_at"),
                    "parameters_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                    "parent_model": details.get("parent_model"),
                    # Add context_size if it exists in the details
                    "context_size": details.get("context_length") 
                }
                standardized_models.append(flat_model_info)
            
            return standardized_models

        except Exception as e:
            ASCIIColors.error(f"Failed to list running models from Ollama at {self.host_address}: {e}")
            return []

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
        models = binding.list_models()
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
        generated_text = binding.generate_text(prompt_text, n_predict=50, stream=False, think=False)
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