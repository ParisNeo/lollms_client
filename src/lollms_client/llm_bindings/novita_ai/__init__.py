from __future__ import annotations

import os
import json
import requests
import base64
import re
from typing import Optional, Callable, List, Union, Dict, Any

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

pm.ensure_packages(["requests", "tiktoken"])
import tiktoken

BindingName = "NovitaAIBinding"
DEFAULT_HOST_ADDRESS = "https://api.novita.ai/v3/openai"

_FALLBACK_MODELS = [
    {
        "model_name": "meta-llama/llama-3.3-70b-instruct",
        "display_name": "Llama 3.3 70B Instruct",
        "description": "Meta's flagship Llama 3.3 70B multilingual instruction-tuned model.",
        "owned_by": "Meta",
        "context_length": 131072,
        "context_size": 131072,
    },
    {
        "model_name": "deepseek/deepseek-r1",
        "display_name": "DeepSeek R1",
        "description": "DeepSeek R1 frontier reasoning model.",
        "owned_by": "DeepSeek",
        "context_length": 65536,
        "context_size": 65536,
    },
    {
        "model_name": "deepseek/deepseek-v3",
        "display_name": "DeepSeek V3",
        "description": "DeepSeek V3 671B mixture-of-experts model.",
        "owned_by": "DeepSeek",
        "context_length": 65536,
        "context_size": 65536,
    },
    {
        "model_name": "qwen/qwen-2.5-coder-32b-instruct",
        "display_name": "Qwen 2.5 Coder 32B Instruct",
        "description": "Alibaba's specialized high-performance code generation model.",
        "owned_by": "Alibaba",
        "context_length": 32768,
        "context_size": 32768,
    },
    {
        "model_name": "qwen/qwen-2.5-72b-instruct",
        "display_name": "Qwen 2.5 72B Instruct",
        "description": "Alibaba's general-purpose flagship model.",
        "owned_by": "Alibaba",
        "context_length": 32768,
        "context_size": 32768,
    },
    {
        "model_name": "mistralai/mistral-large-instruct-2407",
        "display_name": "Mistral Large Instruct 2407",
        "description": "Mistral AI's flagship frontier model.",
        "owned_by": "Mistral AI",
        "context_length": 131072,
        "context_size": 131072,
    },
    {
        "model_name": "meta-llama/llama-3.1-8b-instruct",
        "display_name": "Llama 3.1 8B Instruct",
        "description": "Meta's efficient lightweight 8B model.",
        "owned_by": "Meta",
        "context_length": 131072,
        "context_size": 131072,
    },
    {
        "model_name": "meta-llama/llama-3.2-11b-vision-instruct",
        "display_name": "Llama 3.2 11B Vision Instruct",
        "description": "Meta's multimodal vision and document understanding model.",
        "owned_by": "Meta",
        "context_length": 131072,
        "context_size": 131072,
    },
]


class NovitaAIBinding(LollmsLLMBinding):
    """Novita AI LLM binding using the OpenAI-compatible v3 endpoint."""

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.model_name = kwargs.get("model_name") or "meta-llama/llama-3.3-70b-instruct"
        self.service_key = (
            kwargs.get("service_key")
            or kwargs.get("api_key")
            or os.getenv("NOVITA_API_KEY")
            or ""
        )

        raw_host = kwargs.get("host_address") or DEFAULT_HOST_ADDRESS
        raw_host = raw_host.rstrip("/")
        if raw_host in ("https://api.novita.ai", "http://api.novita.ai"):
            raw_host = f"{raw_host}/v3/openai"
        self.host_address = raw_host

        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self._model_context_sizes: Dict[str, int] = {}

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
        return headers

    def _require_api_key(self) -> None:
        if not self.service_key:
            raise ValueError(
                "Novita AI API key is required. Please set it via 'service_key' "
                "or the NOVITA_API_KEY environment variable."
            )

    @property
    def supports_vision(self) -> bool:
        if getattr(self, "vision_enabled", False):
            return True
        name = (self.model_name or "").lower()
        return any(v in name for v in ("vision", "vl", "omni", "pixtral", "llama-3.2-11b", "qwen-vl", "qwen2-vl"))

    def _get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        target_model = model_name or self.model_name
        if not target_model:
            return None
        if target_model in self._model_context_sizes:
            return self._model_context_sizes[target_model]
        self.list_models()
        return self._model_context_sizes.get(target_model)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists available models from Novita AI dynamically with fallback to curated models.
        """
        if self.service_key:
            url = f"{self.host_address.rstrip('/')}/models"
            try:
                response = requests.get(
                    url,
                    headers=self._build_headers(),
                    verify=self.verify_ssl_certificate,
                    timeout=15,
                )
                if response.status_code == 200:
                    data = response.json()
                    models_raw = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(models_raw, list) and models_raw:
                        formatted = []
                        for m in models_raw:
                            if not isinstance(m, dict):
                                continue
                            m_id = m.get("id") or m.get("name") or m.get("model_name")
                            if not m_id:
                                continue
                            ctx_size = m.get("context_size") or m.get("context_length")
                            if ctx_size:
                                try:
                                    self._model_context_sizes[m_id] = int(ctx_size)
                                except (ValueError, TypeError):
                                    pass
                            formatted.append({
                                "model_name": m_id,
                                "display_name": m.get("title") or m_id,
                                "description": m.get("description", ""),
                                "owned_by": m.get("owned_by") or (m_id.split("/")[0] if "/" in m_id else "Novita AI"),
                                "created": m.get("created"),
                                "created_datetime": m.get("created"),
                                "context_length": ctx_size,
                                "context_size": ctx_size,
                            })
                        if formatted:
                            return sorted(formatted, key=lambda x: x.get("display_name", x["model_name"]))
            except Exception as e:
                ASCIIColors.warning(f"[{self.binding_name}] Failed to fetch online models: {e}. Using fallback list.")

        for m in _FALLBACK_MODELS:
            if m.get("context_size"):
                self._model_context_sizes[m["model_name"]] = m["context_size"]
        return sorted(_FALLBACK_MODELS, key=lambda x: x.get("display_name", x["model_name"]))

    def _format_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            images = msg.get("images") or []
            active_images = msg.get("active_images") or []

            all_imgs = []
            if images:
                all_imgs.extend(images)
            if active_images:
                all_imgs.extend(active_images)

            if all_imgs and self.supports_vision:
                content_parts = []
                if isinstance(content, str) and content:
                    content_parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    content_parts.extend(content)

                for img in all_imgs:
                    if isinstance(img, str):
                        if img.startswith("http://") or img.startswith("https://") or img.startswith("data:"):
                            url_val = img
                        else:
                            cleaned = re.sub(r"^data:image/[^;]+;base64,", "", img)
                            url_val = f"data:image/jpeg;base64,{cleaned}"
                        content_parts.append({"type": "image_url", "image_url": {"url": url_val}})
                    elif isinstance(img, bytes):
                        b64_str = base64.b64encode(img).decode("utf-8")
                        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})

                formatted.append({"role": role, "content": content_parts})
            else:
                if isinstance(content, list):
                    text_only = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                    formatted.append({"role": role, "content": text_only})
                else:
                    formatted.append({"role": role, "content": content})
        return formatted

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
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs
    ) -> Union[str, dict]:
        self._require_api_key()

        if streaming_callback is not None:
            stream = True
        elif stream is None:
            stream = bool(self.default_stream)

        alternated = self.clean_and_alternate_messages(messages)
        formatted_messages = self._format_messages_for_api(alternated)

        temp_val = temperature if temperature is not None else self.default_temperature
        top_p_val = top_p if top_p is not None else self.default_top_p
        seed_val = seed if seed is not None else self.default_seed
        max_tokens_val = n_predict if n_predict is not None else self.default_n_predict

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": stream,
        }

        if temp_val is not None:
            payload["temperature"] = float(temp_val)
        if top_p_val is not None:
            payload["top_p"] = float(top_p_val)
        if max_tokens_val is not None:
            payload["max_tokens"] = int(max_tokens_val)
        if seed_val is not None:
            payload["seed"] = int(seed_val)
        if repeat_penalty is not None:
            payload["frequency_penalty"] = float(repeat_penalty)

        effective_effort = self.normalize_reasoning_effort(think, reasoning_effort)
        if effective_effort is not None:
            payload["reasoning_effort"] = effective_effort

        url = f"{self.host_address.rstrip('/')}/chat/completions"
        full_response_text = ""
        full_reasoning_text = ""

        try:
            if stream:
                with requests.post(
                    url,
                    headers=self._build_headers(),
                    json=payload,
                    stream=True,
                    verify=self.verify_ssl_certificate,
                    timeout=kwargs.get("timeout", 180),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if self.is_cancelled():
                            break
                        if not line:
                            continue

                        decoded = line.decode("utf-8")
                        if decoded.startswith("data:"):
                            raw_chunk = decoded[5:].strip()
                            if raw_chunk == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(raw_chunk)
                                choices = chunk_json.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})

                                reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning") or ""
                                if reasoning_chunk:
                                    full_reasoning_text += reasoning_chunk
                                    if streaming_callback:
                                        streaming_callback(reasoning_chunk, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

                                text_chunk = delta.get("content") or ""
                                if text_chunk:
                                    full_response_text += text_chunk
                                    if streaming_callback:
                                        if not streaming_callback(text_chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                            break
                            except json.JSONDecodeError:
                                continue

                if effective_effort is not None and full_reasoning_text and not full_response_text.startswith("<think>"):
                    return f"<think>\n{full_reasoning_text}\n</think>\n{full_response_text}"
                return full_response_text

            else:
                response = requests.post(
                    url,
                    headers=self._build_headers(),
                    json=payload,
                    verify=self.verify_ssl_certificate,
                    timeout=kwargs.get("timeout", 180),
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                reasoning = choice.get("reasoning_content") or choice.get("reasoning") or ""

                if effective_effort is not None and reasoning:
                    return f"<think>\n{reasoning}\n</think>\n{content}"
                return content

        except requests.exceptions.RequestException as e:
            err_msg = f"Novita AI API request failed: {e}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    err_msg += f" (Status {e.response.status_code}: {e.response.text})"
                except Exception:
                    pass
            ASCIIColors.error(err_msg)
            raise RuntimeError(err_msg) from e
        except Exception as ex:
            trace_exception(ex)
            raise RuntimeError(f"Unexpected error in Novita AI generate: {ex}") from ex

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
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs
    ) -> Union[str, dict]:
        messages: List[Dict] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        if split:
            messages.extend(self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword))
            if images and messages:
                messages[-1]["images"] = images
        else:
            user_msg = {"role": "user", "content": prompt}
            if images:
                user_msg["images"] = images
            messages.append(user_msg)

        return self.generate_from_messages(
            messages=messages,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            seed=seed,
            streaming_callback=streaming_callback,
            think=think,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            **kwargs
        )

    def tokenize(self, text: str) -> list:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text or "")
        except Exception:
            return list((text or "").encode("utf-8"))

    def detokenize(self, tokens: list) -> str:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except Exception:
            return bytes(tokens).decode("utf-8", errors="ignore")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        self._require_api_key()
        model_to_use = kwargs.get("model") or "BAAI/bge-m3"
        url = f"{self.host_address.rstrip('/')}/embeddings"
        payload = {
            "model": model_to_use,
            "input": text,
        }
        try:
            response = requests.post(
                url,
                headers=self._build_headers(),
                json=payload,
                verify=self.verify_ssl_certificate,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as ex:
            trace_exception(ex)
            raise RuntimeError(f"Novita AI embedding failed: {ex}") from ex

    def get_model_info(self) -> dict:
        return {
            "name": self.binding_name,
            "host_address": self.host_address,
            "model_name": self.model_name,
            "supports_vision": self.supports_vision,
            "supports_structured_output": True,
        }

    def load_model(self, model_name: str) -> bool:
        self.model_name = model_name
        ASCIIColors.info(f"[{self.binding_name}] Active model set to: {model_name}")
        return True

    def get_user_balance(self) -> Dict[str, Any]:
        """
        Queries the user account balance from Novita AI.
        """
        if not self.service_key:
            return {"status": False, "message": "No API key configured."}

        urls = [
            "https://api.novita.ai/openapi/v1/billing/balance/detail",
            "https://api.novita.ai/v3/user/balance",
        ]
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=10, verify=self.verify_ssl_certificate)
                if resp.status_code == 200:
                    data = resp.json()
                    avail = data.get("availableBalance") or data.get("balance") or data.get("data", {}).get("balance")
                    if avail is not None:
                        try:
                            avail_float = float(avail)
                            usd = avail_float / 10000.0 if avail_float > 1000 else avail_float
                            return {
                                "status": True,
                                "raw_balance": avail,
                                "balance_usd": f"${usd:.2f}",
                                "data": data,
                            }
                        except Exception:
                            return {"status": True, "balance": str(avail), "data": data}
                    return {"status": True, "data": data}
            except Exception:
                continue

        return {"status": False, "message": "Failed to fetch user balance from Novita AI."}

    def validate_key(self) -> Dict[str, Any]:
        """
        Validates the Novita AI API key and checks connectivity.
        """
        if not self.service_key:
            return {"status": False, "message": "API key is missing."}
        try:
            models = self.list_models()
            if models and len(models) > 0:
                return {
                    "status": True,
                    "message": f"API key is valid. Retrieved {len(models)} available models.",
                    "models_count": len(models),
                }
        except Exception as e:
            return {"status": False, "message": f"API key validation failed: {e}"}
        return {"status": False, "message": "Could not validate API key with Novita AI."}