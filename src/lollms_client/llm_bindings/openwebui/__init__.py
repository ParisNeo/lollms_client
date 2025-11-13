import json
import base64
import os
import mimetypes
import io
from typing import Optional, Callable, List, Union, Dict

import httpx
import tiktoken
import pipmaster as pm
from PIL import Image

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from ascii_colors import ASCIIColors, trace_exception

# Ensure required packages are installed
pm.ensure_packages(["httpx", "tiktoken", "Pillow"])

BindingName = "OpenWebUIBinding"


def _read_file_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_markdown_path(s):
    s = s.strip()
    if s.startswith("[") and s.endswith(")"):
        lb, rb = s.find("["), s.find("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb + 1 : rb].strip()
    return s


def _guess_mime_from_name(name, default="image/jpeg"):
    mime, _ = mimetypes.guess_type(name)
    return mime or default


def _to_data_url(b64_str, mime):
    return f"data:{mime};base64,{b64_str}"


def normalize_image_input(
    img_path: str,
    cap_size: bool = False,
    max_dim: int = 2048,
    default_mime="image/jpeg"
) -> dict:
    if not isinstance(img_path, str):
        raise ValueError("Unsupported image input type for OpenWebUI")

    s = _extract_markdown_path(img_path)
    if not os.path.exists(s):
        url = _to_data_url(s, default_mime)
        return {"type": "image_url", "image_url": {"url": url}}

    if cap_size:
        with Image.open(s) as img_obj:
            width, height = img_obj.size
            if width > max_dim or height > max_dim:
                ratio = max_dim / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                ASCIIColors.info(f"Downsizing image from {width}x{height} to {new_width}x{new_height}")
                resized_img = img_obj.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                if resized_img.mode in ('RGBA', 'P'):
                    resized_img = resized_img.convert('RGB')
                resized_img.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()
                
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                mime = "image/jpeg"
            else:
                b64 = _read_file_as_base64(s)
                mime = _guess_mime_from_name(s, default_mime)
    else:
        b64 = _read_file_as_base64(s)
        mime = _guess_mime_from_name(s, default_mime)
    
    url = _to_data_url(b64, mime)
    return {"type": "image_url", "image_url": {"url": url}}


class OpenWebUIBinding(LollmsLLMBinding):
    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        self.host_address = kwargs.get("host_address")
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key", os.getenv("OPENWEBUI_API_KEY"))
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.allow_non_standard_parameters = kwargs.get("allow_non_standard_parameters", False)
        self.cap_image_size = kwargs.get("cap_image_size", True)
        self.image_downsizing_max_dimension = kwargs.get("image_downsizing_max_dimension", 2048)

        if not self.host_address:
            raise ValueError("OpenWebUI host address is required.")

        headers = {"Content-Type": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
        
        self.client = httpx.Client(
            base_url=self.host_address,
            headers=headers,
            verify=self.verify_ssl_certificate,
            timeout=None,
        )

    def _build_request_params(self, messages: list, **kwargs) -> dict:
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": kwargs.get("stream", True),
        }

        if "n_predict" in kwargs and kwargs["n_predict"] is not None:
            params["max_tokens"] = kwargs["n_predict"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            params["top_p"] = kwargs["top_p"]
        
        if self.allow_non_standard_parameters:
            if "top_k" in kwargs and kwargs["top_k"] is not None:
                params["top_k"] = kwargs["top_k"]

        if "repeat_penalty" in kwargs and kwargs["repeat_penalty"] is not None:
            params["frequency_penalty"] = kwargs["repeat_penalty"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            params["seed"] = kwargs["seed"]

        return params

    def _process_request(
        self,
        params: dict,
        stream: Optional[bool],
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
    ) -> Union[str, dict]:
        output = ""
        try:
            if stream:
                with self.client.stream("POST", "/api/chat/completions", json=params) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        data_str = None
                        if isinstance(line, bytes):
                            if line.startswith(b"data:"):
                                data_str = line[len(b"data:"):].strip().decode("utf-8")
                        elif isinstance(line, str):
                            if line.startswith("data:"):
                                data_str = line[len("data:"):].strip()
                        
                        if data_str is None:
                            continue

                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                word = delta.get("content", "")
                                if word and streaming_callback:
                                    if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                        break
                                output += word
                        except json.JSONDecodeError:
                            continue
            else:
                response = self.client.post("/api/chat/completions", json=params)
                response.raise_for_status()
                data = response.json()
                output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if streaming_callback:
                    streaming_callback(output, MSG_TYPE.MSG_TYPE_CHUNK)

        except httpx.HTTPStatusError as e:
            try:
                e.response.read()
                response_text = e.response.text
            except Exception:
                response_text = "(Could not read error response body)"
            err_msg = f"API Error: {e.response.status_code} - {response_text}"
            trace_exception(e)
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}
        except Exception as e:
            err_msg = f"An unexpected error occurred with the OpenWebUI API: {e}"
            trace_exception(e)
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def generate_text(
        self,
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
        **kwargs,
    ) -> Union[str, dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": prompt}
        if images:
            b64_images = []
            for img_path in images:
                normalized = normalize_image_input(
                    img_path,
                    cap_size=self.cap_image_size,
                    max_dim=self.image_downsizing_max_dimension
                )
                data_url = normalized["image_url"]["url"]
                if "base64," in data_url:
                    b64_images.append(data_url.split("base64,")[1])
            if b64_images:
                user_message["images"] = b64_images
        
        messages.append(user_message)

        params = self._build_request_params(
            messages=messages, n_predict=n_predict, stream=stream,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repeat_penalty=repeat_penalty, **kwargs,
        )
        return self._process_request(params, stream, streaming_callback)

    def chat(
        self,
        discussion: LollmsDiscussion,
        branch_tip_id: Optional[str] = None,
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        **kwargs,
    ) -> Union[str, dict]:
        messages = discussion.export("ollama_chat", branch_tip_id)
        params = self._build_request_params(
            messages=messages, n_predict=n_predict, stream=stream,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repeat_penalty=repeat_penalty, **kwargs,
        )
        return self._process_request(params, stream, streaming_callback)

    def generate_from_messages(
        self,
        messages: List[Dict],
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        **kwargs,
    ) -> Union[str, dict]:
        # Convert from OpenAI vision format to Ollama vision format
        ollama_messages = []
        for msg in messages:
            content = msg.get("content")
            role = msg.get("role")
            
            if isinstance(content, list):
                text_parts = []
                image_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if "base64," in url:
                            image_parts.append(url.split("base64,")[1])
                
                new_msg = {"role": role, "content": "\n".join(text_parts)}
                if image_parts:
                    new_msg["images"] = image_parts
                ollama_messages.append(new_msg)
            else:
                ollama_messages.append(msg)

        params = self._build_request_params(
            messages=ollama_messages, n_predict=n_predict, stream=stream,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repeat_penalty=repeat_penalty, **kwargs,
        )
        return self._process_request(params, stream, streaming_callback)


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
       
        # Failsafe: Hardcoded context sizes for popular Ollama models
        known_contexts = {
            'llama2': 4096,       # Llama 2 default
            'llama3': 8192,       # Llama 3 default
            'llama3.1': 131072,   # Llama 3.1 extended context
            'llama3.2': 131072,   # Llama 3.2 extended context
            'llama3.3': 131072,   # Assuming similar to 3.1/3.2
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
    
    def list_models(self) -> List[Dict]:
        models_info = []
        try:
            response = self.client.get("/api/v1/models")
            if response.status_code == 403 and "API key is not enabled" in response.text:
                temp_client = httpx.Client(
                    base_url=self.host_address,
                    headers={"Content-Type": "application/json"},
                    verify=self.verify_ssl_certificate, timeout=None,
                )
                response = temp_client.get("/api/v1/models")
                temp_client.close()

            response.raise_for_status()
            models_data = response.json().get("data", [])
            for model in models_data:
                models_info.append({
                    "model_name": model.get("id", "N/A"),
                    "owned_by": model.get("details", {}).get("family", "N/A"),
                    "created": model.get("modified_at", "N/A"),
                    "context_length": model.get("details", {}).get("parameter_size", "unknown"),
                })
        except Exception as e:
            ASCIIColors.error(f"Failed to list models from OpenWebUI: {e}")
            trace_exception(e)
        return models_info

    def _get_encoding(self, model_name: str = None):
        try:
            return tiktoken.encoding_for_model(model_name or self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text: str) -> list[int]:
        return self._get_encoding().encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        return self._get_encoding().decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def embed(self, text: Union[str, List[str]], **kwargs) -> List:
        embedding_model = kwargs.get("model", self.model_name)
        single_input = isinstance(text, str)
        inputs = [text] if single_input else list(text)
        embeddings = []
        try:
            for t in inputs:
                payload = {"model": embedding_model, "prompt": t}
                response = self.client.post("/ollama/api/embeddings", json=payload)
                response.raise_for_status()
                data = response.json()
                vec = data.get("embedding")
                if vec is not None:
                    embeddings.append(vec)
            return embeddings[0] if single_input and embeddings else embeddings
        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using model '{embedding_model}': {e}")
            trace_exception(e)
            return []
        
    def get_model_info(self) -> dict:
        return {
            "name": self.binding_name,
            "version": "1.5",
            "host_address": self.host_address,
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": True,
        }

    def load_model(self, model_name: str) -> bool:
        self.model_name = model_name
        ASCIIColors.info(f"OpenWebUI model set to: {model_name}")
        return True

    def ps(self):
        return []


OpenWebUIBinding.__abstractmethods__ = set()
