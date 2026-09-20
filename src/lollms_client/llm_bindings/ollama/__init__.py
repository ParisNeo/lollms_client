from __future__ import annotations

import base64
import json
import os
import platform
import re
import subprocess
import threading
import urllib.request
import zipfile
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import pipmaster as pm
import tiktoken
from ascii_colors import ASCIIColors, trace_exception

pm.ensure_packages(["ollama>=0.6.1", "pillow", "tiktoken"])

import ollama

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE
from lollms_client.lollms_utilities import ImageTokenizer

BindingName = "OllamaBinding"

_THINK_OPEN = "\n<think>\n"

_THINK_CLOSE = "\n</think>\n"

class _ThinkingStreamTracker:
    """
    Single-purpose state machine that guarantees balanced 
    delimiters across an Ollama streaming response.

    Ollama interleaves `message.thinking` and `message.content` chunks without
    any explicit boundary marker. This tracker detects the thinking↔content
    transitions and synthesizes the closing tag exactly once, either at the
    transition point or at end-of-stream (flush).
    """

    __slots__ = ("in_thinking", "_opened")

    def __init__(self) -> None:
        self.in_thinking = False
        self._opened = False

    def open_think(self) -> str:
        """Enters thinking mode. Returns the opening delimiter (idempotent)."""
        if self.in_thinking:
            return ""
        self.in_thinking = True
        self._opened = True
        return _THINK_OPEN

    def close_think(self) -> str:
        """Exits thinking mode. Returns the closing delimiter (idempotent)."""
        if not self.in_thinking:
            return ""
        self.in_thinking = False
        return _THINK_CLOSE

    def feed_thinking(self, text: str) -> str:
        """Processes a thinking chunk. Returns text to emit (opener + chunk)."""
        emitted = self.open_think()
        if text:
            emitted += text
        return emitted

    def feed_content(self, text: str) -> str:
        """Processes a content chunk. Returns text to emit (closer + chunk)."""
        emitted = self.close_think()
        if text:
            emitted += text
        return emitted

    def flush(self) -> str:
        """Closes any dangling think block at end-of-stream. Idempotent."""
        return self.close_think()


def count_tokens_ollama(
    text_to_tokenize: str,
    model_name: str,
    ollama_client: "ollama.Client",
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
        ollama_client: An initialized ollama.Client used to perform the request.

    Returns:
        The number of tokens as reported by 'prompt_eval_count'.

    Raises:
        ollama.ResponseError: If the API request fails.
        RuntimeError: For other operational errors.
    """
    res = ollama_client.chat(
        model=model_name,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": text_to_tokenize}],
        stream=False,
        think=False,
        options={"num_predict": 1},
    )

    return res.prompt_eval_count - 5


class OllamaBinding(LollmsLLMBinding):
    """Ollama-specific binding implementation using the ollama-python library."""

    DEFAULT_HOST_ADDRESS = "http://localhost:11434"

    def __init__(
        self,
        **kwargs,
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
        self.debug = kwargs.get("debug", False)
        self.host_address = _host_address
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.default_completion_format = kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)
        self.n_threads = kwargs.get("n_threads", -1)

        if ollama is None:
            raise ImportError("Ollama library is not installed. Please run 'pip install ollama'.")

        self.ollama_client_headers: dict[str, str] = {}
        if self.service_key:
            self.ollama_client_headers["Authorization"] = f"Bearer {self.service_key}"

        self._active_client: "ollama.Client | None" = None
        self._client_lock = threading.Lock()

    def clean_message_images(self, messages: list[dict]) -> list[dict]:
        """
        Ensures all base64-encoded images in the messages list are clean,
        decoded bytes objects (stripping any 'data:image/...;base64,' prefix).
        """
        cleaned_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            images = msg.get("images") or []

            text_parts = []
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") in ("input_image", "image_url"):
                        base64_data = item.get("image_url")
                        if isinstance(base64_data, str):
                            cleaned = re.sub(r"^data:image/[^;]+;base64,", "", base64_data)
                            images.append(cleaned)
                        elif isinstance(base64_data, dict):
                            url_val = base64_data.get("url") or base64_data.get("base64") or ""
                            cleaned = re.sub(r"^data:image/[^;]+;base64,", "", url_val)
                            images.append(cleaned)
                content = "\n".join([p for p in text_parts if p.strip()])

            cleaned_images = []
            for img in images:
                if isinstance(img, str):
                    cleaned = re.sub(r"^data:image/[^;]+;base64,", "", img)
                    try:
                        missing_padding = len(cleaned) % 4
                        if missing_padding:
                            cleaned += "=" * (4 - missing_padding)
                        decoded = base64.b64decode(cleaned)
                        cleaned_images.append(decoded)
                    except (ValueError, TypeError) as e:
                        ASCIIColors.warning(f"Failed to decode base64 image data: {e!s}")
                        cleaned_images.append(img)
                else:
                    cleaned_images.append(img)

            cleaned_msg = {
                "role": role,
                "content": content,
            }
            if cleaned_images:
                cleaned_msg["images"] = cleaned_images

            cleaned_messages.append(cleaned_msg)

        return cleaned_messages

    @contextmanager
    def _client(self):
        """
        Context manager that yields a shared ollama.Client instance
        to leverage connection pooling and prevent Windows socket exhaustion (WinError 10053).
        """
        if not hasattr(self, "_shared_client") or self._shared_client is None:
            self._shared_client = ollama.Client(
                host=self.host_address,
                headers=self.ollama_client_headers if self.ollama_client_headers else None,
                verify=self.verify_ssl_certificate,
            )
        with self._client_lock:
            self._active_client = self._shared_client
        try:
            yield self._shared_client
        finally:
            with self._client_lock:
                self._active_client = None

    def cancel(self) -> None:
        """
        Signal the binding to stop the current generation as soon as possible.
        """
        super().cancel()

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        """
        Unloads the current model from Ollama's memory/VRAM.
        """
        target_model = model_name or self.model_name
        if not target_model:
            ASCIIColors.warning(f"[{self.binding_name}] No active model to unload.")
            return False
        try:
            with self._client() as client:
                client.generate(model=target_model, keep_alive=0)
            ASCIIColors.success(f"[{self.binding_name}] Successfully requested Ollama to unload model '{target_model}'.")
            return True
        except Exception as e:
            ASCIIColors.warning(f"[{self.binding_name}] Failed to unload Ollama model '{target_model}': {e!s}")
            return False

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
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
        **kwargs,
    ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[list[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split (Optional[bool]): Put to true if the prompt is a discussion.
            user_keyword (Optional[str]): When splitting we use this to extract user prompt.
            ai_keyword (Optional[str]): When splitting we use this to extract ai prompt.

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        if streaming_callback:
            stream = True

        options = {
            "num_predict": n_predict,
            "temperature": float(temperature) if temperature is not None else None,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "seed": seed,
            "num_ctx": self.forced_ctx_size if self.forced_ctx_size else self.default_ctx_size,
        }
        if self.n_threads > 0:
            options["num_thread"] = self.n_threads

        options = {k: v for k, v in options.items() if v is not None}

        full_response_text = ""
        think = self.normalize_reasoning_effort(think, reasoning_effort)
        think = False if think is None else (think != "low")

        try:
            with self._client() as client:
                if images:
                    processed_images = []
                    for img_path in images:
                        if isinstance(img_path, str):
                            cleaned = re.sub(r"^data:image/[^;]+;base64,", "", img_path)
                            try:
                                missing_padding = len(cleaned) % 4
                                if missing_padding:
                                    cleaned += "=" * (4 - missing_padding)
                                decoded = base64.b64decode(cleaned)
                                processed_images.append(decoded)
                            except (ValueError, TypeError) as e:
                                ASCIIColors.warning(f"Failed to decode base64 image data: {e!s}")
                                processed_images.append(img_path)
                        else:
                            processed_images.append(img_path)

                    messages = [
                        {"role": "system", "content": system_prompt},
                    ]
                    if split:
                        messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
                        if processed_images:
                            messages[-1]["images"] = processed_images
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": prompt,
                                "images": processed_images if processed_images else None,
                            }
                        )
                    alternated_messages = self.clean_and_alternate_messages(messages)
                    chat_kwargs = {
                        "model": self.model_name,
                        "messages": alternated_messages,
                        "stream": True,
                        "options": options if options else None,
                    }
                    if think is not None:
                        chat_kwargs["think"] = think

                    if stream:
                        response_stream = client.chat(**chat_kwargs)
                        tracker = _ThinkingStreamTracker()
                        for chunk in response_stream:
                            if self.is_cancelled():
                                break

                            if hasattr(chunk, "message"):
                                msg_obj = chunk.message
                                chunk_thinking = getattr(msg_obj, "thinking", None)
                                chunk_content = getattr(msg_obj, "content", None)
                            elif isinstance(chunk, dict):
                                msg_dict = chunk.get("message", {})
                                chunk_thinking = msg_dict.get("thinking")
                                chunk_content = msg_dict.get("content")
                            else:
                                chunk_thinking = None
                                chunk_content = None

                            if chunk_thinking:
                                emitted = tracker.feed_thinking(chunk_thinking)
                                full_response_text += emitted
                                if streaming_callback:
                                    streaming_callback(emitted, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                                continue

                            if chunk_content:
                                emitted = tracker.feed_content(chunk_content)
                                full_response_text += emitted
                                if streaming_callback:
                                    if not streaming_callback(emitted, MSG_TYPE.MSG_TYPE_CHUNK):
                                        break
                        closing = tracker.flush()
                        full_response_text += closing
                        if closing and streaming_callback:
                            streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                        return full_response_text
                    else:
                        chat_kwargs = {
                            "model": self.model_name,
                            "messages": alternated_messages,
                            "stream": False,
                            "options": options if options else None,
                        }
                        if think is not None:
                            chat_kwargs["think"] = think

                        if self.debug:
                            ASCIIColors.cyan(f"[{self.binding_name}] Sending non-streaming chat request to Ollama...")

                        response = client.chat(**chat_kwargs)
                        full_response_text = response.message.content

                        if self.debug:
                            ASCIIColors.cyan(f"[{self.binding_name}] Received response: {full_response_text[:200]}...")
                        if think:
                            full_response_text = "\n" + response.message.thinking + "\n</think>\n" + full_response_text
                        return full_response_text
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                    ]
                    if split:
                        messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
                    else:
                        messages.append({"role": "user", "content": prompt})

                    alternated_messages = self.clean_and_alternate_messages(messages)
                    chat_kwargs = {
                        "model": self.model_name,
                        "messages": alternated_messages,
                        "stream": stream,
                        "options": options if options else None,
                    }
                    if think is not None:
                        chat_kwargs["think"] = think

                    if self.debug:
                        ASCIIColors.cyan(f"[{self.binding_name}] Sending chat request to Ollama:")
                        ASCIIColors.cyan(f"  • Model: {self.model_name}")
                        if chat_kwargs.get("options"):
                            ASCIIColors.cyan(f"  • Options: {json.dumps(chat_kwargs['options'], indent=2)}")

                    if stream:
                        response_stream = client.chat(**chat_kwargs)
                        tracker = _ThinkingStreamTracker()
                        for chunk in response_stream:
                            if self.is_cancelled():
                                break

                            if hasattr(chunk, "message"):
                                msg_obj = chunk.message
                                chunk_thinking = getattr(msg_obj, "thinking", None)
                                chunk_content = getattr(msg_obj, "content", None)
                            elif isinstance(chunk, dict):
                                msg_dict = chunk.get("message", {})
                                chunk_thinking = msg_dict.get("thinking")
                                chunk_content = msg_dict.get("content")
                            else:
                                chunk_thinking = None
                                chunk_content = None

                            if chunk_thinking:
                                full_response_text += tracker.feed_thinking(chunk_thinking)
                                continue

                            if chunk_content:
                                if self.debug:
                                    ASCIIColors.rich_print(f"[cyan]{chunk_content}[/cyan]", end="", flush=True)
                                emitted = tracker.feed_content(chunk_content)
                                full_response_text += emitted
                                if streaming_callback:
                                    if not streaming_callback(emitted, MSG_TYPE.MSG_TYPE_CHUNK):
                                        break
                        closing = tracker.flush()
                        full_response_text += closing
                        if closing and streaming_callback:
                            streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                        return full_response_text
                    else:
                        response = client.chat(
                            model=self.model_name,
                            messages=alternated_messages,
                            stream=False,
                            think=think,
                            options=options if options else None,
                        )
                        full_response_text = response.message.content
                        if think and response.message.thinking:
                            full_response_text = "WebResponse\n" + response.message.thinking + "\n</think>\n" + full_response_text
                        return full_response_text

        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except ollama.RequestError as e:
            error_message = f"Ollama API RequestError: {e!s}"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as ex:
            error_message = f"An unexpected error occurred: {ex!s}"
            trace_exception(ex)
            raise RuntimeError(error_message) from ex

    def generate_from_messages(
        self,
        messages: list[dict],
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
        **kwargs,
    ) -> Union[str, dict]:
        options = {}
        if n_predict is not None:
            options["num_predict"] = n_predict
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_k is not None:
            options["top_k"] = top_k
        if top_p is not None:
            options["top_p"] = top_p
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if repeat_last_n is not None:
            options["repeat_last_n"] = repeat_last_n
        if seed is not None:
            options["seed"] = seed
        if self.n_threads > 0:
            options["num_thread"] = self.n_threads
        if self.forced_ctx_size is not None:
            options["num_ctx"] = self.forced_ctx_size
        elif self.default_ctx_size:
            options["num_ctx"] = self.default_ctx_size

        alternated_messages = self.clean_and_alternate_messages(messages)
        ollama_messages = self.clean_message_images(alternated_messages)
        full_response_text = ""
        think = self.normalize_reasoning_effort(think, reasoning_effort)
        think = False if think is None else (think != "low")

        try:
            with self._client() as client:
                chat_kwargs = {
                    "model": self.model_name,
                    "messages": ollama_messages,
                    "options": options if options else None,
                }

                if stream:
                    chat_kwargs["stream"] = True
                    if think is not None:
                        chat_kwargs["think"] = think
                    response_stream = client.chat(**chat_kwargs)
                    tracker = _ThinkingStreamTracker()
                    for chunk in response_stream:
                        if self.is_cancelled():
                            break

                        if hasattr(chunk, "message"):
                            msg_obj = chunk.message
                            chunk_thinking = getattr(msg_obj, "thinking", None)
                            chunk_content = getattr(msg_obj, "content", None)
                        elif isinstance(chunk, dict):
                            msg_dict = chunk.get("message", {})
                            chunk_thinking = msg_dict.get("thinking")
                            chunk_content = msg_dict.get("content")
                        else:
                            chunk_thinking = None
                            chunk_content = None

                        if chunk_thinking:
                            emitted = tracker.feed_thinking(chunk_thinking)
                            full_response_text += emitted
                            if streaming_callback:
                                streaming_callback(emitted, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                            continue

                        if chunk_content:
                            emitted = tracker.feed_content(chunk_content)
                            full_response_text += emitted
                            if streaming_callback:
                                if not streaming_callback(emitted, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break
                    closing = tracker.flush()
                    full_response_text += closing
                    if closing and streaming_callback:
                        streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                    return full_response_text
                else:
                    chat_kwargs["stream"] = False
                    eff_think = think if "gpt-oss" not in self.model_name else reasoning_effort
                    if eff_think is not None:
                        chat_kwargs["think"] = eff_think
                    response = client.chat(**chat_kwargs)
                    full_response_text = response.message.content
                    if think:
                        full_response_text = "WebResponse\n" + response.message.thinking + "\n</think>\n" + full_response_text
                    return full_response_text

        except ollama.ResponseError as e:
            error_message = f"Ollama API ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except ollama.RequestError as e:
            error_message = f"Ollama API RequestError: {e!s}"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as ex:
            error_message = f"An unexpected error occurred: {ex!s}"
            trace_exception(ex)
            raise RuntimeError(error_message) from ex

    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of individual characters.
        """
        if text is None:
            return []
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text, disallowed_special=())

    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens (characters) to detokenize.

        Returns:
            str: Detokenized text.
        """
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
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e!s}")
            return -1

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Get embeddings for the input text using Ollama API.

        Args:
            text (str): Input text to embed.
            **kwargs: Optional arguments. Can include 'model' to override self.model_name,
                      and 'options' dictionary for Ollama embedding options.

        Returns:
            list[float]: The embedding vector.

        Raises:
            RuntimeError: if embedding fails or Ollama client is not available.
        """
        model_to_use = kwargs.get("model", "bge-m3")
        if not model_to_use:
            raise ValueError("Model name for embedding must be specified either in init or via kwargs.")

        ollama_options = kwargs.get("options", None)
        try:
            with self._client() as client:
                response = client.embeddings(
                    model=model_to_use,
                    prompt=text,
                    options=ollama_options,
                )
                return response["embedding"]
        except ollama.ResponseError as e:
            error_message = f"Ollama API Embeddings ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except ollama.RequestError as e:
            error_message = f"Ollama API Embeddings RequestError: {e!s}"
            ASCIIColors.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as ex:
            trace_exception(ex)
            raise RuntimeError(f"Embedding failed: {ex!s}") from ex

    @property
    def supports_vision(self) -> bool:
        """
        Dynamically determine if the active model supports vision.
        """
        if not self.model_name:
            return False
        try:
            with self._client() as client:
                info = client.show(self.model_name)

                details = info.get("details", {})
                families = details.get("families", []) or [details.get("family", "")]
                families = [f.lower() for f in families if f]

                if any(f in families for f in ("llava", "mllama", "clip", "vision")):
                    return True

                model_info = info.get("model_info", {})
                for key in model_info:
                    key_lower = key.lower()
                    if "vision" in key_lower or "clip" in key_lower or "projector" in key_lower:
                        return True

            return False
        except Exception as e:
            ASCIIColors.warning(f"Failed to determine vision support for '{self.model_name}': {e!s}")
            return True

    def get_model_info(self) -> dict:
        """
        Return information about the current Ollama model setup.

        Returns:
            dict: Dictionary containing binding name, version, host address, and model name.
        """
        return {
            "name": self.binding_name,
            "version": pm.get_installed_version("ollama") if ollama else "unknown",
            "host_address": self.host_address,
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": self.supports_vision,
        }

    def pull_model(self, model_name: str, progress_callback: Callable[[dict], None] | None = None, **kwargs) -> dict:
        """
        Pulls a model from the Ollama library.

        Args:
            model_name (str): The name of the model to pull.
            progress_callback (Callable[[dict], None] | None): A callback function that receives progress updates.
                The dict typically contains 'status', 'completed', 'total'.

        Returns:
            dict: Dictionary with status (bool) and message (str).
        """
        try:
            with self._client() as client:
                ASCIIColors.info(f"Pulling model {model_name}...")
                for progress in client.pull(model_name, stream=True):
                    if progress_callback:
                        progress_callback(progress)

                    status = progress.get("status", "")
                    completed = progress.get("completed")
                    total = progress.get("total")

                    if completed and total:
                        percent = (completed / total) * 100
                        print(f"\r{status}: {percent:.2f}%", end="", flush=True)
                    else:
                        print(f"\r{status}", end="", flush=True)

                print()
                msg = f"Model {model_name} pulled successfully."
                ASCIIColors.success(msg)
                return {"status": True, "message": msg}

        except ollama.ResponseError as e:
            msg = f"Ollama API Pull Error: {e.error or 'Unknown error'} (status code: {e.status_code})"
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}
        except ollama.RequestError as e:
            msg = f"Ollama API Request Error: {e!s}"
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}
        except Exception as ex:
            msg = f"An unexpected error occurred while pulling model: {ex!s}"
            ASCIIColors.error(msg)
            trace_exception(ex)
            return {"status": False, "message": msg}

    def get_zoo(self) -> list[dict[str, Any]]:
        """
        Returns a list of models available for download.
        each entry is a dict with:
        name, description, size, type, link
        """
        return [
            {"name": "Llama3 8B", "description": "Meta's Llama 3 8B model. Good for general purpose chat.", "size": "4.7GB", "type": "model", "link": "llama3"},
            {"name": "Llama3 70B", "description": "Meta's Llama 3 70B model. High capability.", "size": "40GB", "type": "model", "link": "llama3:70b"},
            {"name": "Phi-3 Mini", "description": "Microsoft's Phi-3 Mini 3.8B model. Lightweight and capable.", "size": "2.3GB", "type": "model", "link": "phi3"},
            {"name": "Phi-3 Medium", "description": "Microsoft's Phi-3 Medium 14B model.", "size": "7.9GB", "type": "model", "link": "phi3:medium"},
            {"name": "Mistral 7B", "description": "Mistral AI's 7B model v0.3.", "size": "4.1GB", "type": "model", "link": "mistral"},
            {"name": "Mixtral 8x7B", "description": "Mistral AI's Mixture of Experts model.", "size": "26GB", "type": "model", "link": "mixtral"},
            {"name": "Gemma 2 9B", "description": "Google's Gemma 2 9B model.", "size": "5.4GB", "type": "model", "link": "gemma2"},
            {"name": "Gemma 2 27B", "description": "Google's Gemma 2 27B model.", "size": "16GB", "type": "model", "link": "gemma2:27b"},
            {"name": "Qwen 2.5 7B", "description": "Alibaba Cloud's Qwen2.5 7B model.", "size": "4.5GB", "type": "model", "link": "qwen2.5"},
            {"name": "Qwen 2.5 Coder 7B", "description": "Alibaba Cloud's Qwen2.5 Coder 7B model.", "size": "4.5GB", "type": "model", "link": "qwen2.5-coder"},
            {"name": "CodeLlama 7B", "description": "Meta's CodeLlama 7B model.", "size": "3.8GB", "type": "model", "link": "codellama"},
            {"name": "LLaVA 7B", "description": "Visual instruction tuning model (Vision).", "size": "4.5GB", "type": "model", "link": "llava"},
            {"name": "Nomic Embed Text", "description": "A high-performing open embedding model.", "size": "274MB", "type": "embedding", "link": "nomic-embed-text"},
            {"name": "DeepSeek Coder V2", "description": "DeepSeek Coder V2 model.", "size": "8.9GB", "type": "model", "link": "deepseek-coder-v2"},
            {"name": "OpenHermes 2.5 Mistral", "description": "High quality finetune of Mistral 7B.", "size": "4.1GB", "type": "model", "link": "openhermes"},
            {"name": "Dolphin Phi", "description": "Uncensored Dolphin fine-tune of Phi-2.", "size": "1.6GB", "type": "model", "link": "dolphin-phi"},
            {"name": "TinyLlama", "description": "A compact 1.1B model.", "size": "637MB", "type": "model", "link": "tinyllama"},
        ]

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] | None = None) -> dict:
        """
        Downloads a model from the zoo using its index.
        """
        zoo = self.get_zoo()
        if index < 0 or index >= len(zoo):
            msg = "Index out of bounds"
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}
        item = zoo[index]
        return self.pull_model(item["link"], progress_callback=progress_callback)

    def install_ollama(self, callback: Callable[[dict], None] | None = None, **kwargs) -> dict:
        """
        Installs Ollama based on the operating system.
        """
        system = platform.system()

        def report_progress(status, message, completed=0, total=100):
            if callback:
                callback({"status": status, "message": message, "completed": completed, "total": total})
            else:
                print(f"{status}: {message}")

        try:
            if system == "Linux":
                report_progress("working", "Detected Linux. Running installation script...", 10, 100)
                cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                _, stderr = process.communicate()

                if process.returncode == 0:
                    report_progress("success", "Ollama installed successfully on Linux.", 100, 100)
                    return {"status": True, "message": "Ollama installed successfully."}
                else:
                    msg = f"Installation failed: {stderr!s}"
                    report_progress("error", msg, 0, 0)
                    return {"status": False, "error": msg}

            elif system == "Windows":
                report_progress("working", "Detected Windows. Downloading OllamaSetup.exe...", 10, 100)
                url = "https://ollama.com/download/OllamaSetup.exe"
                filename = "OllamaSetup.exe"

                try:
                    def dl_callback(count, block_size, total_size):
                        percent = int(count * block_size * 100 / total_size)
                        report_progress("working", f"Downloading... {percent}%", percent, 100)

                    urllib.request.urlretrieve(url, filename, dl_callback)
                except Exception as e:
                    return {"status": False, "error": f"Failed to download installer: {e!s}"}

                report_progress("working", "Running installer...", 90, 100)
                try:
                    subprocess.run([filename], check=True)
                    report_progress("success", "Installer launched. Please complete the installation.", 100, 100)
                    return {"status": True, "message": "Installer launched."}
                except Exception as e:
                    return {"status": False, "error": f"Failed to launch installer: {e!s}"}

            elif system == "Darwin":
                report_progress("working", "Detected macOS. Downloading Ollama...", 10, 100)
                url = "https://ollama.com/download/Ollama-darwin.zip"
                filename = "Ollama-darwin.zip"

                try:
                    def dl_callback(count, block_size, total_size):
                        percent = int(count * block_size * 100 / total_size)
                        report_progress("working", f"Downloading... {percent}%", percent, 100)

                    urllib.request.urlretrieve(url, filename, dl_callback)
                except Exception as e:
                    return {"status": False, "error": f"Failed to download: {e!s}"}

                report_progress("working", "Unzipping...", 80, 100)
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall("Ollama_Install")

                report_progress("success", "Ollama downloaded and extracted to 'Ollama_Install'. Please move 'Ollama.app' to Applications.", 100, 100)
                return {"status": True, "message": "Downloaded and extracted. Please install Ollama.app manually."}

            else:
                return {"status": False, "error": f"Unsupported OS: {system}"}

        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    def list_models(self) -> list[dict[str, str]]:
        """
        Lists available models from the Ollama service using the ollama-python library.
        The returned list of dictionaries matches the format of the original template.

        Returns:
            list[dict[str, str]]: A list of model information dictionaries.
                Each dict has 'model_name', 'owned_by', 'created_datetime'.
        """
        try:
            with self._client() as client:
                ASCIIColors.debug(f"Listing ollama models from {self.host_address}")
                response_data = client.list()

                model_info_list = []
                if "models" in response_data:
                    for model_entry in response_data["models"]:
                        model_info_list.append(
                            {
                                "model_name": model_entry.get("model"),
                                "owned_by": "",
                                "created_datetime": model_entry.get("modified_at"),
                            }
                        )
                return model_info_list
        except ollama.ResponseError as e:
            ASCIIColors.error(f"Ollama API list_models ResponseError: {e.error or 'Unknown error'} (status code: {e.status_code}) from {self.host_address}")
            return []
        except ollama.RequestError as e:
            ASCIIColors.error(f"Ollama API list_models RequestError: {e!s} from {self.host_address}")
            return []
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """
        Set the model name for subsequent operations. Ollama loads models on demand.

        Args:
            model_name (str): Name of the model to set.

        Returns:
            bool: True if model name is set.
        """
        self.model_name = model_name
        ASCIIColors.info(f"Ollama model set to: {model_name}. It will be loaded by the server on first use.")
        return True

    def _get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Retrieves the context size for an Ollama model.

        The effective context size is the ``num_ctx`` parameter if overridden in the
        Modelfile, otherwise it falls back to the model's default context length from
        its architecture details. As a final failsafe, the base-class heuristic is used.
        """
        if model_name is None:
            model_name = self.model_name
            if not model_name:
                ASCIIColors.warning("Model name not specified and no default model set.")
                return None

        try:
            import requests

            url = f"{self.host_address}/api/show"
            headers = {}
            if self.service_key:
                headers["Authorization"] = f"Bearer {self.service_key}"

            response = requests.post(
                url,
                json={"name": model_name},
                headers=headers,
                verify=self.verify_ssl_certificate,
                timeout=10,
            )
            response.raise_for_status()
            info = response.json()

            parameters = info.get("parameters", "") or ""
            num_ctx = None
            for param in str(parameters).split("\n"):
                stripped = param.strip()
                if stripped.startswith("num_ctx"):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            num_ctx = int(parts[1])
                        except (ValueError, IndexError):
                            pass
                    break

            if num_ctx is not None:
                return num_ctx

            model_info = info.get("model_info", {})

            if isinstance(model_info, dict):
                arch = model_info.get("general.architecture", "")

                if arch:
                    context_key = f"{arch}.context_length"
                    context_length = model_info.get(context_key)
                    if context_length is not None:
                        try:
                            return int(context_length)
                        except (ValueError, TypeError):
                            pass

                context_length = model_info.get("general.context_length")
                if context_length is not None:
                    try:
                        return int(context_length)
                    except (ValueError, TypeError):
                        pass

                for key, value in model_info.items():
                    if "context_length" in str(key).lower() and value is not None:
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            continue

            elif isinstance(model_info, list):
                for item in model_info:
                    if not isinstance(item, dict):
                        continue
                    item_key = item.get("key")
                    item_value = item.get("value")

                    if item_key and "context_length" in str(item_key).lower() and item_value is not None:
                        try:
                            return int(item_value)
                        except (ValueError, TypeError):
                            continue

        except requests.exceptions.RequestException as e:
            ASCIIColors.warning(f"HTTP error fetching model info for '{model_name}': {e!s}")
        except Exception as e:
            ASCIIColors.warning(f"Error fetching model info for '{model_name}': {e!s}")

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
        """
        try:
            with self._client() as client:
                running_models_response = client.ps()

                models_list = running_models_response.get("models", [])
                standardized_models = []

                for model_data in models_list:
                    details = model_data.get("details", {})

                    size = model_data.get("size", 0)
                    size_vram = model_data.get("size_vram", 0)

                    gpu_usage = 0
                    cpu_usage = 0
                    if size > 0:
                        gpu_usage = min(100, (size_vram / size) * 100)
                        cpu_usage = max(0, 100 - gpu_usage)

                    flat_model_info = {
                        "model_name": model_data.get("name"),
                        "size": size,
                        "vram_size": size_vram,
                        "gpu_usage_percent": round(gpu_usage, 2),
                        "cpu_usage_percent": round(cpu_usage, 2),
                        "expires_at": model_data.get("expires_at"),
                        "parameters_size": details.get("parameter_size"),
                        "quantization_level": details.get("quantization_level"),
                        "parent_model": details.get("parent_model"),
                        "context_size": details.get("context_length"),
                    }
                    standardized_models.append(flat_model_info)

                return standardized_models

        except Exception as e:
            ASCIIColors.error(f"Failed to list running models from Ollama at {self.host_address}: {e!s}")
            return []


if __name__ == "__main__":
    full_streamed_text = ""

    def stream_callback(chunk: str, msg_type: int):
        global full_streamed_text
        full_streamed_text += chunk
        if len(full_streamed_text) > 100:
            print("\nStopping stream early for test.")
        return True

    ASCIIColors.yellow("Testing OllamaBinding...")

    ollama_host = "http://localhost:11434"
    test_model_name = "llama3"
    test_vision_model_name = "llava"

    try:
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = OllamaBinding(host_address=ollama_host, model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using Ollama client version: {ollama.__version__ if ollama else 'N/A'}")

        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Found {len(models)} models. First 5:")
            for m in models[:5]:
                print(m)
        else:
            ASCIIColors.warning("No models found or failed to list models. Ensure Ollama is running and has models.")

        ASCIIColors.cyan(f"\n--- Setting model to: {test_model_name} ---")
        binding.load_model(test_model_name)

        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        ASCIIColors.cyan("\n--- Tokenize/Detokenize ---")
        tokens = binding.tokenize(sample_text)
        ASCIIColors.green(f"Tokens for '{sample_text}': {tokens[:10]}...")
        detokenized_text = binding.detokenize(tokens)
        ASCIIColors.green(f"Detokenized text (may vary based on tokenization type): {detokenized_text}")

        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Why is the sky blue?"
        ASCIIColors.info(f"Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, n_predict=50, stream=False, think=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text: {generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        full_streamed_text = ""
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, n_predict=100, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        if isinstance(result, str):
            ASCIIColors.green(f"Full streamed text: {result}")
        else:
            ASCIIColors.error(f"Streaming generation failed: {result}")

        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            embedding_text = "Lollms is a cool project."
            embedding_vector = binding.embed(embedding_text)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.warning(f"Could not get embedding with '{binding.model_name}': {e!s}. Some models don't support /api/embeddings or may need to be specified.")
            ASCIIColors.warning("Try `ollama pull mxbai-embed-large` and set it as model for embedding.")

        dummy_image_path = "dummy_test_image.png"
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (100, 30), color=("red"))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Hello", fill=("white"))
            img.save(dummy_image_path)
            ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

            ASCIIColors.cyan(f"\n--- Vision Generation (using {test_vision_model_name}) ---")
            vision_model_exists = any(m["model_name"].startswith(test_vision_model_name) for m in models)
            if not vision_model_exists:
                ASCIIColors.warning(f"Vision model '{test_vision_model_name}' not found in pulled models. Skipping vision test.")
                ASCIIColors.warning(f"Try: `ollama pull {test_vision_model_name}`")
            else:
                binding.load_model(test_vision_model_name)
                vision_prompt = "What is written in this image?"
                ASCIIColors.info(f"Vision Prompt: {vision_prompt} with image {dummy_image_path}")

                vision_response = binding.generate_text(
                    prompt=vision_prompt,
                    images=[dummy_image_path],
                    n_predict=50,
                    stream=False,
                )
                if isinstance(vision_response, str):
                    ASCIIColors.green(f"Vision model response: {vision_response}")
                else:
                    ASCIIColors.error(f"Vision generation failed: {vision_response}")
        except ImportError:
            ASCIIColors.warning("Pillow library not found. Cannot create dummy image for vision test. `pip install Pillow`")
        except Exception as e:
            ASCIIColors.error(f"Error during vision test: {e!s}")
        finally:
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

    except ConnectionRefusedError:
        ASCIIColors.error("Connection to Ollama server refused. Is Ollama running?")
    except ImportError as e:
        ASCIIColors.error(f"Import error: {e!s}. Make sure 'ollama' library is installed ('pip install ollama').")
    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e!s}")
        trace_exception(e)

    ASCIIColors.yellow("\nOllamaBinding test finished.")