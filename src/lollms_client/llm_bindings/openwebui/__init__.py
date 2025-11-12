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
            return s[lb + 1 : rb].strip()
    return s


def _guess_mime_from_name(name, default="image/jpeg"):
    mime, _ = mimetypes.guess_type(name)
    return mime or default


def _to_data_url(b64_str, mime):
    return f"data:{mime};base64,{b64_str}"


def normalize_image_input(img, default_mime="image/jpeg"):
    """
    Returns an OpenAI API‑compatible content block for an image.
    Accepts various input formats and converts them to a data URL.
    """
    if isinstance(img, str):
        # Handle path‑like strings or raw base64
        s = _extract_markdown_path(img)
        if os.path.exists(s):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            url = _to_data_url(b64, mime)
        else:  # Assume it's a raw base64 string
            url = _to_data_url(s, default_mime)
        return {"type": "image_url", "image_url": {"url": url}}

    raise ValueError("Unsupported image input type for OpenWebUI")


class OpenWebUIBinding(LollmsLLMBinding):
    """OpenWebUI‑specific binding implementation"""

    def __init__(self, **kwargs):
        """
        Initialize the OpenWebUI binding.

        Args:
            host_address (str): URL of the OpenWebUI server (e.g. ``http://localhost:8080``).
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
            ASCIIColors.warning(
                "No service key provided for OpenWebUI. Requests may fail."
            )

        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
        }

        self.client = httpx.Client(
            base_url=self.host_address,
            headers=headers,
            verify=self.verify_ssl_certificate,
            timeout=None,
        )

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _build_request_params(self, messages: list, **kwargs) -> dict:
        """Construct the JSON payload expected by the OpenWebUI /chat/completions endpoint."""
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": kwargs.get("stream", True),
        }

        # Map Lollms parameters to OpenAI‑compatible fields
        if "n_predict" in kwargs and kwargs["n_predict"] is not None:
            params["max_tokens"] = kwargs["n_predict"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            params["top_p"] = kwargs["top_p"]
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
        """Execute the request – handling both streaming and non‑streaming modes."""
        output = ""
        try:
            if stream:
                with self.client.stream(
                    "POST", "/api/chat/completions", json=params
                ) as response:
                    if response.status_code != 200:
                        err = response.read().decode("utf-8")
                        raise Exception(
                            f"API Error: {response.status_code} - {err}"
                        )

                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith(b"data:"):
                            data_str = line[len(b"data:") :].strip().decode("utf-8")
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if chunk.get("choices"):
                                    delta = chunk["choices"][0].get("delta", {})
                                    word = delta.get("content", "")
                                    if word:
                                        if streaming_callback:
                                            if not streaming_callback(
                                                word, MSG_TYPE.MSG_TYPE_CHUNK
                                            ):
                                                break
                                        output += word
                            except json.JSONDecodeError:
                                continue
            else:
                response = self.client.post("/api/chat/completions", json=params)
                if response.status_code != 200:
                    raise Exception(
                        f"API Error: {response.status_code} - {response.text}"
                    )
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

    # --------------------------------------------------------------------- #
    # Public API required by LollmsLLMBinding
    # --------------------------------------------------------------------- #
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
        """Generate text (or multimodal output) via OpenWebUI."""
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
            **kwargs,
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
        """Generate from a pre‑formatted list of OpenAI‑compatible messages."""
        params = self._build_request_params(
            messages=messages,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs,
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
        repeat_last_n: int = 64,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[bool] = "low",
        reasoning_summary: Optional[bool] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        """
        Conduct a chat session using a :class:`LollmsDiscussion` object.
        The discussion is exported in an OpenAI‑compatible format and then
        passed to :meth:`_process_request`.
        """
        messages = discussion.export("openai_chat", branch_tip_id)

        params = self._build_request_params(
            messages=messages,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            **kwargs,
        )
        return self._process_request(params, stream, streaming_callback)

    def list_models(self) -> List[Dict]:
        """Return a list of models known to the OpenWebUI server."""
        models_info = []
        try:
            response = self.client.get("/api/models")
            response.raise_for_status()
            models_data = response.json().get("data", [])
            for model in models_data:
                models_info.append(
                    {
                        "model_name": model.get("id", "N/A"),
                        "owned_by": model.get("details", {}).get("family", "N/A"),
                        "created": model.get("modified_at", "N/A"),
                        "context_length": model.get("details", {}).get(
                            "parameter_size", "unknown"
                        ),
                    }
                )
        except Exception as e:
            ASCIIColors.error(f"Failed to list models from OpenWebUI: {e}")
        return models_info

    def _get_encoding(self, model_name: str | None = None):
        """Fallback to tiktoken for generic tokenisation."""
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

    def embed(self, text: str | List[str], **kwargs) -> List:
        """
        Obtain embeddings via the OpenWebUI ``/embeddings`` endpoint.
        If a single string is supplied, a single embedding vector is returned;
        otherwise a list of vectors is returned.
        """
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
            ASCIIColors.error(
                f"Failed to generate embeddings using model '{embedding_model}': {e}"
            )
            trace_exception(e)
            return []

    def get_model_info(self) -> dict:
        """Return basic information about the current binding configuration."""
        return {
            "name": self.binding_name,
            "version": pm.get_installed_version("openwebui")
            if "openwebui" in globals()
            else "unknown",
            "host_address": self.host_address,
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": True,
        }

    def load_model(self, model_name: str) -> bool:
        """Select a model for subsequent calls."""
        self.model_name = model_name
        ASCIIColors.info(f"OpenWebUI model set to: {model_name}")
        return True

    def ps(self):
        """Placeholder – OpenWebUI does not expose a process‑list endpoint."""
        return []


# Ensure the class is treated as concrete (no remaining abstract methods)
OpenWebUIBinding.__abstractmethods__ = set()
