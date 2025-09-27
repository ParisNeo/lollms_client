import requests
import json
import base64
import os
import mimetypes
import math
from typing import Optional, Callable, List, Union, Dict

import httpx
import tiktoken
import pipmaster as pm

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import encode_image
from ascii_colors import ASCIIColors, trace_exception

# Ensure required packages are installed
pm.ensure_packages(["httpx", "tiktoken"])

BindingName = "OpenWebUIBinding"


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
    Returns an OpenAI API-ready content block for an image.
    Accepts various input formats and converts them to a data URL.
    """
    if isinstance(img, str):
        # Handle path-like strings or raw base64
        s = _extract_markdown_path(img)
        if os.path.exists(s):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            url = _to_data_url(b64, mime)
        else: # Assume it's a base64 string
            url = _to_data_url(s, default_mime)
        return {"type": "image_url", "image_url": {"url": url}}

    raise ValueError("Unsupported image input type for OpenWebUI")


class OpenWebUIBinding(LollmsLLMBinding):
    """OpenWebUI-specific binding implementation"""
    
    def __init__(self, **kwargs):
        """
        Initialize the OpenWebUI binding.

        Args:
            host_address (str): The URL of the OpenWebUI server (e.g., "http://localhost:8080").
            model_name (str): Name of the model to use.
            service_key (str): Authentication token for the service.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        super().__init__(BindingName, **kwargs)
        self.host_address = kwargs.get("host_address")
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key", os.getenv("OPENWEBUI_API_KEY"))
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        
        if not self.host_address:
            raise ValueError("OpenWebUI host address is required.")
        if not self.service_key:
            ASCIIColors.warning("No service key provided for OpenWebUI. Requests may fail.")

        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json"
        }
        
        self.client = httpx.Client(
            base_url=self.host_address,
            headers=headers,
            verify=self.verify_ssl_certificate,
            timeout=None 
        )

    def _build_request_params(self, messages: list, **kwargs) -> dict:
        """Builds the request parameters for the OpenWebUI API."""
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": kwargs.get("stream", True),
        }
        
        # Map Lollms parameters to OpenAI-compatible parameters
        if "n_predict" in kwargs and kwargs["n_predict"] is not None:
            params["max_tokens"] = kwargs["n_predict"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs and kwargs["top_k"] is not None:
             # Note: top_k is not standard in OpenAI API, but some backends might support it.
             # We include it here for potential compatibility.
            params["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs and kwargs["repeat_penalty"] is not None:
            params["frequency_penalty"] = kwargs["repeat_penalty"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            params["seed"] = kwargs["seed"]
        
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
                      streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                      **kwargs
                      ) -> Union[str, dict]:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                user_content.append(normalize_image_input(img))
        
        messages.append({"role": "user", "content": user_content})

        params = self._build_request_params(
            messages=messages,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs
        )
        
        return self._process_request(params, stream, streaming_callback)


    def generate_from_messages(self,
                               messages: List[Dict],
                               n_predict: Optional[int] = None,
                               stream: Optional[bool] = None,
                               temperature: Optional[float] = None,
                               top_k: Optional[int] = None,
                               top_p: Optional[float] = None,
                               repeat_penalty: Optional[float] = None,
                               streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                               **kwargs
                               ) -> Union[str, dict]:
        
        params = self._build_request_params(
            messages=messages,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs
        )
        
        return self._process_request(params, stream, streaming_callback)

    def _process_request(self, params, stream, streaming_callback):
        """Helper to process streaming or non-streaming API calls."""
        output = ""
        try:
            if stream:
                with self.client.stream("POST", "/api/chat/completions", json=params) as response:
                    if response.status_code != 200:
                        error_content = response.read().decode('utf-8')
                        raise Exception(f"API Error: {response.status_code} - {error_content}")

                    for line in response.iter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    word = delta.get("content", "")
                                    if word:
                                        if streaming_callback:
                                            if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                                break
                                        output += word
                            except json.JSONDecodeError:
                                continue # Ignore malformed SSE lines
            else:
                response = self.client.post("/api/chat/completions", json=params)
                if response.status_code != 200:
                    raise Exception(f"API Error: {response.status_code} - {response.text}")
                
                data = response.json()
                output = data["choices"][0]["message"]["content"]
                if streaming_callback:
                    streaming_callback(output, MSG_TYPE.MSG_TYPE_CHUNK)

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenWebUI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def list_models(self) -> List[Dict]:
        models_info = []
        try:
            response = self.client.get("/api/models")
            response.raise_for_status()
            models_data = response.json().get("data", [])
            
            for model in models_data:
                models_info.append({
                    "model_name": model.get("id", "N/A"),
                    "owned_by": model.get("details", {}).get("family", "N/A"),
                    "created": model.get("modified_at", "N/A"),
                    # Assuming context length might be in details, though not guaranteed
                    "context_length": model.get("details", {}).get("parameter_size", "unknown"), 
                })
        except Exception as e:
            ASCIIColors.error(f"Failed to list models from OpenWebUI: {e}")
        return models_info
    
    def _get_encoding(self, model_name: str | None = None):
        """Uses tiktoken as a general-purpose tokenizer."""
        try:
            return tiktoken.encoding_for_model(model_name or self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text: str) -> list[int]:
        encoding = self._get_encoding()
        return encoding.encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        encoding = self._get_encoding()
        return encoding.decode(tokens)
        
    def count_tokens(self, text: str) -> int:      
        return len(self.tokenize(text))
    
    def get_input_tokens_price(self, model_name: str | None = None) -> float:
        return 0.0

    def get_output_tokens_price(self, model_name: str | None = None) -> float:
        return 0.0

    def embed(self, text: str | list[str], **kwargs) -> list:
        """Get embeddings using Ollama's passthrough endpoint."""
        embedding_model = kwargs.get("model", self.model_name)
        is_single_input = isinstance(text, str)
        input_texts = [text] if is_single_input else text
        embeddings = []

        try:
            for t in input_texts:
                payload = {"model": embedding_model, "prompt": t}
                response = self.client.post("/ollama/api/embeddings", json=payload)
                response.raise_for_status()
                embedding_data = response.json().get("embedding")
                if embedding_data:
                    embeddings.append(embedding_data)

            return embeddings[0] if is_single_input and embeddings else embeddings

        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using model '{embedding_model}': {e}")
            trace_exception(e)
            return []

    def load_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True