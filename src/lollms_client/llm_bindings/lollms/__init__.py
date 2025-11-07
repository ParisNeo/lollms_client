# bindings/Lollms_chat/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Dict
import httpx
import pipmaster as pm
import mimetypes
import base64

pm.ensure_packages(["openai","tiktoken"])

import openai
import tiktoken
import os

BindingName = "LollmsBinding"

def _read_file_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _extract_markdown_path(s):
    s = s.strip()
    if s.startswith("[") and s.endswith(")"):
        lb, rb = s.find("["), s.find("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb+1:rb].strip()
    return s

def _guess_mime_from_name(name, default="image/jpeg"):
    mime, _ = mimetypes.guess_type(name)
    return mime or default

def _to_data_url(b64_str, mime):
    return f"data:{mime};base64,{b64_str}"

def normalize_image_input(img, default_mime="image/jpeg"):
    """
    Returns a Responses API-ready content block:
      { "type": "input_image", "image_url": "data:<mime>;base64,<...>" }
    Accepts:
      - dict {'data': '<base64>', 'mime': 'image/png'}
      - dict {'path': 'E:\\images\\x.png'}
      - string raw base64
      - string local path (Windows/POSIX), including markdown-like "[E:\\path\\img.png]()"
    URLs are intentionally not supported (base64 only).
    """
    if isinstance(img, dict):
        if "data" in img and isinstance(img["data"], str):
            mime = img.get("mime", default_mime)
            return {"type": "input_image", "image_url": _to_data_url(img["data"], mime)}
        if "path" in img and isinstance(img["path"], str):
            p = _extract_markdown_path(img["path"])
            b64 = _read_file_as_base64(p)
            mime = _guess_mime_from_name(p, default_mime)
            return {"type": "input_image", "image_url": _to_data_url(b64, mime)}
        if "url" in img:
            raise ValueError("URL inputs not allowed here; provide base64 or local path")
        raise ValueError("Unsupported dict format for image input")

    if isinstance(img, str):
        s = _extract_markdown_path(img)
        # Accept already-correct data URLs as-is
        if s.startswith("data:"):
            return {"type": "input_image", "image_url": s}
        # Local path heuristics: exists on disk or looks like a path
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith("/") or s.startswith("."):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "input_image", "image_url": _to_data_url(b64, mime)}
        # Otherwise, treat as raw base64 payload
        return {"type": "input_image", "image_url": _to_data_url(s, default_mime)}

    raise ValueError("Unsupported image input type")
class LollmsBinding(LollmsLLMBinding):
    """Lollms-specific binding implementation (open ai compatible with some extra parameters)"""
    
    
    def __init__(self,
                 **kwargs):
        """
        Initialize the OpenAI binding.

        Args:
            host_address (str): Host address for the OpenAI service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None. This is a key generated 
                               on the lollms interface (it is advised to use LOLLMS_API_KEY environment variable instead)
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Ignored parameter for compatibility with LollmsLLMBinding.
        """
        super().__init__(BindingName, **kwargs)
        self.host_address=kwargs.get("host_address","http://localhost:9642/v1").rstrip("/")
        if not self.host_address.endswith("v1"):
            self.host_address += "/v1"  
        self.model_name=kwargs.get("model_name")
        self.service_key=kwargs.get("service_key")
        self.verify_ssl_certificate=kwargs.get("verify_ssl_certificate", True)
        self.default_completion_format=kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)

        if not self.service_key:
            self.service_key = os.getenv("LOLLMS_API_KEY", self.service_key)
        self.client = openai.OpenAI(api_key=self.service_key, base_url=None if self.host_address is None else self.host_address if len(self.host_address)>0 else None, http_client=httpx.Client(verify=self.verify_ssl_certificate))
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    def lollms_listMountedPersonalities(self, host_address:str|None=None):
        host_address = host_address if host_address else self.host_address
        url = f"{host_address}/personalities"

        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            try:
                text = json.loads(response.content.decode("utf-8"))
                return text
            except Exception as ex:
                return {"status": False, "error": str(ex)}
        else:
            return {"status": False, "error": response.text}

  
    def _build_openai_params(self, messages: list, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = [
            "gpt-5",
            "gpt-4o",
            "o1",
            "o3",
            "o4"
        ]

        allowed_params = {
            "model", "messages", "temperature", "top_p", "n",
            "stop", "max_tokens", "presence_penalty", "frequency_penalty",
            "logit_bias", "stream", "user", "max_completion_tokens"
        }
        if kwargs.get("think", False):
            allowed_params.append("reasoning")
            kwargs["reasoning"]={
                "effort": allowed_params.append("reasoning_effort", "low"),
                "summary": allowed_params.append("reasoning_summary", "auto")
            }

        params = {
            "model": model,
            "messages": messages,
        }

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v
            else:
                if v is not None:
                    ASCIIColors.warning(f"Removed unsupported OpenAI param '{k}'")

        model_lower = model.lower()
        if any(fam in model_lower for fam in restricted_families):
            if "temperature" in params and params["temperature"] != 1:
                ASCIIColors.warning(f"{model} does not support temperature != 1. Overriding to 1.")
                params["temperature"] = 1
            if "top_p" in params:
                ASCIIColors.warning(f"{model} does not support top_p. Removing it.")
                params.pop("top_p")

        return params
  
    def generate_text(self,
                    prompt: str,
                    images: Optional[List[str]] = None,
                    system_prompt: str = "",
                    n_predict: Optional[int] = None,
                    stream: Optional[bool] = None,
                    temperature: float = 0.7, 
                    top_k: int = 40,          
                    top_p: float = 0.9,       
                    repeat_penalty: float = 1.1, 
                    repeat_last_n: int = 64, 
                    seed: Optional[int] = None,
                    n_threads: Optional[int] = None,
                    ctx_size: int | None = None,
                    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                    split: Optional[bool] = False,
                    user_keyword: Optional[str] = "!@>user:",
                    ai_keyword: Optional[str] = "!@>assistant:",
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[bool] = "low", # low, medium, high
                    reasoning_summary: Optional[bool] = "auto", # auto
                    **kwargs
                    ) -> Union[str, dict]:

        count = 0
        output = ""
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]

        if images:
            if split:
                # Original call to split message roles
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
                # Convert the last message content to the structured content array
                last = messages[-1]
                text_block = {"type": "text", "text": last["content"]}
                image_blocks = [normalize_image_input(img) for img in images]
                last["content"] = [text_block] + image_blocks
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + [
                        normalize_image_input(img) for img in images
                    ]
                })
        else:
            if split:
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
            else:
                messages.append({'role': 'user', 'content': [{"type": "text", "text": prompt}]})

        try:
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
                params = self._build_openai_params(messages=messages,
                                                n_predict=n_predict,
                                                stream=stream,
                                                temperature=temperature,
                                                top_p=top_p,
                                                repeat_penalty=repeat_penalty,
                                                seed=seed,
                                                think = think,
                                                reasoning_effort=reasoning_effort,
                                                reasoning_summary=reasoning_summary
                                                )
                try:
                    chat_completion = self.client.chat.completions.create(**params)
                except Exception as ex:
                    # exception for new openai models
                    params["max_completion_tokens"]=params.get("max_tokens") or params.get("max_completion_tokens") or self.default_ctx_size
                    params["temperature"]=1
                    try: del params["max_tokens"] 
                    except Exception: pass
                    try: del params["top_p"]
                    except Exception: pass
                    try: del params["frequency_penalty"]
                    except Exception: pass
                    
                    chat_completion = self.client.chat.completions.create(**params)                

                if stream:
                    for resp in chat_completion:
                        if count >= (n_predict or float('inf')):
                            break
                        word = getattr(resp.choices[0].delta, "content", "") or ""
                        if streaming_callback and not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                        if word:
                            output += word
                            count += 1
                else:
                    output = chat_completion.choices[0].message.content

            else:
                params = self._build_openai_params(prompt=prompt,
                                                n_predict=n_predict,
                                                stream=stream,
                                                temperature=temperature,
                                                top_p=top_p,
                                                repeat_penalty=repeat_penalty,
                                                seed=seed,
                                                think = think,
                                                reasoning_effort=reasoning_effort,
                                                reasoning_summary=reasoning_summary)
                try:
                    completion =  self.client.completions.create(**params)
                except Exception as ex:
                    # exception for new openai models
                    params["max_completion_tokens"]=params["max_tokens"]
                    params["temperature"]=1
                    try: del params["max_tokens"] 
                    except Exception: pass
                    try: del params["top_p"]
                    except Exception: pass
                    try: del params["frequency_penalty"]
                    except Exception: pass

                    
                    completion =  self.client.completions.create(**params)                

                if stream:
                    for resp in completion:
                        if count >= (n_predict or float('inf')):
                            break
                        word = getattr(resp.choices[0], "text", "") or ""
                        if streaming_callback and not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                        if word:
                            output += word
                            count += 1
                else:
                    output = completion.choices[0].text

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    
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
        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream
        }
        # Add seed if available, as it's supported by newer OpenAI models
        if seed is not None:
            params["seed"] = seed

        # Remove None values, as the API expects them to be absent
        params = {k: v for k, v in params.items() if v is not None}
        
        output = ""
        # 2. Call the API
        try:
            completion = self.client.chat.completions.create(**params)

            if stream:
                for chunk in completion:
                    # The streaming response for chat has a different structure
                    delta = chunk.choices[0].delta
                    if delta.content:
                        word = delta.content
                        if streaming_callback is not None:
                            if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                        output += word
            else:
                output = completion.choices[0].message.content
        
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(error_message, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": error_message}
            
        return output
    
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
            **kwargs
            ) -> Union[str, dict]:
        """
        Conduct a chat session with the OpenAI model using a LollmsDiscussion object.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter (Note: not all OpenAI models use this).
            top_p (float): Top-p sampling parameter.
            repeat_penalty (float): Frequency penalty for repeated tokens.
            seed (Optional[int]): Random seed for generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        # 1. Export the discussion to the OpenAI chat format
        # This handles system prompts, user/assistant roles, and multi-modal content automatically.
        messages = discussion.export("openai_chat", branch_tip_id)

        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream
        }
        # Add seed if available, as it's supported by newer OpenAI models
        if seed is not None:
            params["seed"] = seed

        # Remove None values, as the API expects them to be absent
        params = {k: v for k, v in params.items() if v is not None}
        
        output = ""
        # 2. Call the API
        try:
            # Check if we should use the chat completions or legacy completions endpoint
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
                completion = self.client.chat.completions.create(**params)

                if stream:
                    for chunk in completion:
                        # The streaming response for chat has a different structure
                        delta = chunk.choices[0].delta
                        if delta.content:
                            word = delta.content
                            if streaming_callback is not None:
                                if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break
                            output += word
                else:
                    output = completion.choices[0].message.content

            else: # Fallback to legacy completion format (not recommended for chat)
                # We need to format the messages list into a single string prompt
                legacy_prompt = discussion.export("openai_completion", branch_tip_id)
                legacy_params = {
                    "model": self.model_name,
                    "prompt": legacy_prompt,
                    "max_tokens": n_predict,
                    "n": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": repeat_penalty,
                    "stream": stream
                }
                completion = self.client.completions.create(**legacy_params)

                if stream:
                    for chunk in completion:
                        word = chunk.choices[0].text
                        if streaming_callback is not None:
                            if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                        output += word
                else:
                    output = completion.choices[0].text
        
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(error_message, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": error_message}
            
        return output    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of individual characters.
        """
        try:
            return tiktoken.model.encoding_for_model(self.model_name).encode(text)
        except:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens (characters) to detokenize.

        Returns:
            str: Detokenized text.
        """
        try:
            return tiktoken.model.encoding_for_model(self.model_name).decode(tokens)
        except:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """        
        return len(self.tokenize(text))

        
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using OpenAI API.

        Args:
            text (str): Input text to embed.
            **kwargs: Additional arguments. The 'model' argument can be used 
                      to specify the embedding model (e.g., "text-embedding-3-small").
                      Defaults to "text-embedding-ada-002".

        Returns:
            list: The embedding vector as a list of floats, or an empty list on failure.
        """
        # Determine the embedding model, prioritizing kwargs, with a default
        embedding_model = kwargs.get("model", self.model_name)
        
        try:
            # The OpenAI API expects the input to be a list of strings
            response = self.client.embeddings.create(
                model=embedding_model,
                input=[text]  # Wrap the single text string in a list
            )
            
            # Extract the embedding from the response
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                ASCIIColors.warning("OpenAI API returned no data for the embedding request.")
                return []
                
        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using OpenAI API: {e}")
            trace_exception(e)
            return []
        

    def get_model_info(self) -> dict:
        """
        Return information about the current OpenAI model.

        Returns:
            dict: Dictionary containing model name, version, and host address.
        """
        return {
            "name": "OpenAI",
            "version": "2.0",
            "host_address": self.host_address,
            "model_name": self.model_name
        }

    def list_models(self) -> List[Dict]:
        # Known context lengths
        known_context_lengths = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-1106-preview": 128000,
            "gpt-4-0125-preview": 128000,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16000,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "davinci": 2049,
            "curie": 2049,
            "babbage": 2049,
            "ada": 2049,
        }

        generation_prefixes = (
            "gpt-",
            "text-davinci",
            "davinci",
            "curie",
            "babbage",
            "ada"
        )

        models_info = []
        prompt_buffer = 500

        try:
            models = self.client.models.list()
            for model in models.data:
                model_id = model.id
                if model_id.startswith(generation_prefixes):
                    context_length = known_context_lengths.get(model_id, "unknown")
                    max_generation = (
                        context_length - prompt_buffer
                        if isinstance(context_length, int)
                        else "unknown"
                    )
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": context_length,
                        "max_generation": max_generation,
                    })
                else:
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": None,
                        "max_generation": None,
                    })
                    
        except Exception as e:
            print(f"Failed to list models: {e}")

        return models_info


    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into the OpenAI binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully.
        """
        self.model = model_name
        self.model_name = model_name
        return True
