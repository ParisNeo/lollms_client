# bindings/sglang/__init__.py
from __future__ import annotations
import requests
import json
import httpx
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT

# Ensure required libraries are installed
pm.ensure_packages(["openai", "tiktoken"])
import openai
import tiktoken
import os
import base64
import mimetypes

BindingName = "SGLangBinding"


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
    Returns a standard OpenAI compatible image content block:
      { "type": "image_url", "image_url": { "url": "data:<mime>;base64,<...>" } }
    """
    if isinstance(img, dict):
        if "data" in img and isinstance(img["data"], str):
            mime = img.get("mime", default_mime)
            return {"type": "image_url", "image_url": {"url": _to_data_url(img["data"], mime)}}
        if "path" in img and isinstance(img["path"], str):
            p = _extract_markdown_path(img["path"])
            b64 = _read_file_as_base64(p)
            mime = _guess_mime_from_name(p, default_mime)
            return {"type": "image_url", "image_url": {"url": _to_data_url(b64, mime)}}
        if "url" in img:
            return {"type": "image_url", "image_url": {"url": img["url"]}}
        raise ValueError("Unsupported dict format for image input")

    if isinstance(img, str):
        s = _extract_markdown_path(img)
        if s.startswith("data:") or s.startswith("http:") or s.startswith("https:"):
            return {"type": "image_url", "image_url": {"url": s}}
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith("/") or s.startswith("."):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "image_url", "image_url": {"url": _to_data_url(b64, mime)}}
        return {"type": "image_url", "image_url": {"url": _to_data_url(s, default_mime)}}

    raise ValueError("Unsupported image input type")


class SGLangBinding(LollmsLLMBinding):
    """SGLang-specific binding implementation"""

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        self.host_address = kwargs.get("host_address", "http://localhost:30000/v1")
        self.model_name = kwargs.get("model_name", "default")
        self.service_key = kwargs.get("service_key", "EMPTY")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path", None)
        self.default_completion_format = kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)

        # Basic validation/normalization of host_address to ensure v1 endpoint
        if self.host_address and not self.host_address.endswith("/v1") and not self.host_address.endswith("/v1/"):
            if self.host_address.endswith("/"):
                self.host_address = self.host_address + "v1"
            else:
                self.host_address = self.host_address + "/v1"

        verify = False if not self.verify_ssl_certificate else self.certificate_file_path if self.certificate_file_path else True
        
        self.client = openai.OpenAI(
            api_key=self.service_key,
            base_url=self.host_address,
            http_client=httpx.Client(verify=verify)
        )
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    def _build_sglang_params(self, messages: list, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        allowed_params = {
            "model", "messages", "temperature", "top_p", "n",
            "stop", "max_tokens", "presence_penalty", "frequency_penalty",
            "stream", "user"
        }

        params = {
            "model": model,
            "messages": messages,
        }

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v

        return params

    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: str = "",
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
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs
    ) -> Union[str, dict]:

        count = 0
        output = ""

        # Build message list
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        if images:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword
                )
                last = messages[-1]
                last["content"] = (
                    [{"type": "text", "text": last["content"]}]
                    + [normalize_image_input(img) for img in images]
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [normalize_image_input(img) for img in images]
                        )
                    }
                )
        else:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                )

        try:
            params = self._build_sglang_params(
                messages=messages,
                n_predict=n_predict,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed
            )

            if think:
                params["reasoning"] = {
                    "effort": reasoning_effort or "low",
                    "summary": reasoning_summary or "auto"
                }
                params.pop("temperature", None)
                params.pop("top_p", None)

            chat_completion = self.client.chat.completions.create(**params)

            if stream:
                for resp in chat_completion:
                    if self.is_cancelled():
                        break

                    if not resp.choices:
                        continue

                    delta = resp.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        output += content
                        if streaming_callback:
                            if not streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
            else:
                output = chat_completion.choices[0].message.content or ""

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the SGLang API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def generate_from_messages(
        self,
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
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs
    ) -> Union[str, dict]:

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
                    elif item.get("type") in ("input_image", "image_url"):
                        val = item.get("image_url")
                        if isinstance(val, dict):
                            val = val.get("url") or val.get("base64")
                        if isinstance(val, str) and val:
                            images.append(val)

            text_content = "\n".join(p for p in text_parts if p.strip())

            if not images:
                return {"role": role, "content": text_content}

            openai_content = []
            if text_content:
                openai_content.append({"type": "text", "text": text_content})
            for img in images:
                img_url = img
                if not img.startswith("http") and not img.startswith("data:"):
                    img_url = f"data:image/jpeg;base64,{img}"
                openai_content.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            return {"role": role, "content": openai_content}

        openai_messages = [normalize_message(m) for m in messages]

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream,
        }
        if seed is not None:
            params["seed"] = seed

        params = {k: v for k, v in params.items() if v is not None}

        if think:
            params["reasoning"] = {
                "effort": reasoning_effort or "low",
                "summary": reasoning_summary or "auto",
            }
            params.pop("temperature", None)
            params.pop("top_p", None)

        output = ""

        try:
            completion = self.client.chat.completions.create(**params)

            if stream:
                for chunk in completion:
                    if self.is_cancelled():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        output += content
                        if streaming_callback:
                            if not streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
            else:
                output = completion.choices[0].message.content or ""

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with SGLang API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def _get_encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text: str) -> list[int]:
        if text is None:
            return []
        encoding = self._get_encoding()
        return encoding.encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        encoding = self._get_encoding()
        return encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def embed(self, text: str | list[str], normalize: bool = False, **kwargs) -> list:
        embedding_model = kwargs.get("model", self.model_name)
        is_single_input = isinstance(text, str)
        input_texts = [text] if is_single_input else text

        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=input_texts
            )

            if not response.data:
                ASCIIColors.warning("SGLang API returned no data for the embedding request.")
                return []

            embeddings = [item.embedding for item in response.data]

            if normalize:
                import math
                embeddings = [
                    [v / math.sqrt(sum(x*x for x in emb)) for v in emb]
                    for emb in embeddings
                ]

            return embeddings[0] if is_single_input else embeddings

        except Exception as e:
            ASCIIColors.error(f"Failed to generate SGLang embeddings: {e}")
            trace_exception(e)
            return []

    def get_ctx_size(self, model_name: str | None = None) -> int:
        if model_name is None:
            model_name = self.model_name
        context_map = {
            "llama-3.1": 131072,
            "llama-3": 8192,
            "llama-2": 4096,
            "qwen": 32768,
            "gemma": 8192,
            "mistral": 32768,
            "phi-3": 128000,
        }
        model_name_lower = model_name.lower()
        for key, size in context_map.items():
            if key in model_name_lower:
                return size
        return super().get_ctx_size(model_name=model_name) or 4096

    def get_model_info(self) -> dict:
        return {
            "name": "SGLang",
            "version": "1.0",
            "host_address": self.host_address,
            "model_name": self.model_name
        }

    def list_models(self) -> List[Dict]:
        models_info = []
        try:
            models = self.client.models.list()
            for model in models.data:
                models_info.append({
                    "model_name": model.id,
                    "owned_by": getattr(model, "owned_by", "N/A"),
                    "created": getattr(model, "created", "N/A"),
                })
        except Exception as e:
            ASCIIColors.error(f"Failed to list SGLang models: {e}")
        return models_info

    def load_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True
