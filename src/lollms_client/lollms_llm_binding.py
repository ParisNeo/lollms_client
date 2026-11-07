

# lollms_llm_binding.py
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ascii_colors import trace_exception, ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import ImageTokenizer, robust_json_parser
from lollms_client.lollms_base_binding import LollmsBaseBinding
from lollms_client.lollms_text_processing import LollmsTextProcessor
from functools import lru_cache
import re
import yaml
import json


def _coerce_positive_int(value, fallback: int) -> int:
    """Coerces a value to a positive int, falling back on bad input."""
    try:
        n = int(value)
        return n if n > 0 else fallback
    except (TypeError, ValueError):
        return fallback


@dataclass
class CommandParameter:
    """Schema for a single executable-command parameter.

    The configuration UI renders one editable field per entry. There is NO
    separate "enable command" gate: presence of a CommandSpec in a binding's
    ``get_commands()`` means the command is runnable.
    """

    name: str
    type: str = "str"          # str | int | float | bool | path | json
    description: str = ""
    required: bool = False
    default: Any = None
    choices: Optional[List[Any]] = None


@dataclass
class CommandSpec:
    """Backend-agnostic description of a binding operation exposed to config UIs.

    ``dangerous`` is display-only (UX warning); it never blocks execution.
    ``method_name`` overrides the dispatched method when the command name and
    the underlying method name differ.
    """

    name: str
    description: str = ""
    parameters: List[CommandParameter] = field(default_factory=list)
    dangerous: bool = False
    method_name: Optional[str] = None


def _normalize_effort_list(value):
    """Accepts None, a comma-separated string, or a list. Returns a clean list.

    Delegates to the shared normalizer in ``lollms_core`` when available
    (lazy import to avoid a circular import at module load time), and falls
    back to an inline copy otherwise.
    """
    try:
        from lollms_client.lollms_core import _normalize_effort_list as _shared
        return _shared(value)
    except Exception:
        pass
    if value is None:
        return ["low", "medium", "high"]
    if isinstance(value, str):
        value = value.split(",")
    out = []
    for item in value:
        s = str(item).strip().lower()
        if s and s not in {"", "none", "off", "disabled", "false", "0", "no"} and s not in out:
            out.append(s)
    return out or ["low", "medium", "high"]


@lru_cache(maxsize=1)
def load_known_contexts() -> dict:
    """
    Loads the known_contexts data from a JSON file.

    Returns:
        dict: A dictionary containing the known_contexts data with 'exact', 'aliases', 
              and 'prefix' keys. Returns empty dict structure on error.
    """
    file_path = Path(__file__).parent / "assets" / "models_ctx_sizes.json"
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level dict in {file_path}")

        # Ensure all required keys exist
        data.setdefault("exact", {})
        data.setdefault("aliases", {})
        data.setdefault("prefix", {})

        return data

    except FileNotFoundError:
        ASCIIColors.warning(f"Context sizes file not found at {file_path}, using empty defaults")
    except json.JSONDecodeError as e:
        ASCIIColors.warning(f"Could not decode JSON from {file_path}: {e}, using empty defaults")
    except Exception as e:
        ASCIIColors.warning(f"Unexpected error loading context sizes: {e}, using empty defaults")

    # Always return a proper dict structure, never a list
    return {"exact": {}, "aliases": {}, "prefix": {}}

class LollmsLLMBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS LLM bindings"""

    def __init__(self, 
                 binding_name: str ="unknown",
                 debug:Optional[bool] = False,
                 **kwargs
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            binding_name (Optional[str]): The name of the bindingto be used
        """
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)
        self.model_name = None #Must be set by the instance

        _ctx_size = kwargs.get("ctx_size")
        if _ctx_size is not None:
            try:
                self.default_ctx_size = int(_ctx_size)
            except (ValueError, TypeError):
                self.default_ctx_size = None
        else:
            self.default_ctx_size = None

        _forced_ctx_size = kwargs.get("forced_ctx_size")
        if _forced_ctx_size is not None:
            try:
                self.forced_ctx_size = int(_forced_ctx_size)
            except (ValueError, TypeError):
                self.forced_ctx_size = None
        else:
            self.forced_ctx_size = None

        self.default_n_predict = kwargs.get("n_predict")
        self.default_stream = kwargs.get("stream")
        self.default_temperature = kwargs.get("temperature")
        self.default_top_k = kwargs.get("top_k")
        self.default_top_p = kwargs.get("top_p")
        self.default_repeat_penalty = kwargs.get("repeat_penalty")
        self.default_repeat_last_n = kwargs.get("repeat_last_n")
        self.default_seed = kwargs.get("seed")
        self.default_n_threads = kwargs.get("n_threads")
        self.default_streaming_callback = kwargs.get("streaming_callback")
        self.glm_image_embedding = kwargs.get("glm_image_embedding", False)
        # ── Capability carriers (mirror LollmsModelProfile / LollmsBindingProfile) ──
        # The model profile is authoritative; these are the fallback values that
        # every binding instance carries so profile-less guests behave identically.
        self.image_token_cost: int = _coerce_positive_int(kwargs.get("image_token_cost"), 256)
        self.video_enabled: bool = bool(kwargs.get("video_enabled", False))
        self.thinking_enabled: bool = bool(kwargs.get("thinking_enabled", False))
        self.supported_reasoning_efforts: List[str] = _normalize_effort_list(
            kwargs.get("supported_reasoning_efforts")
        )
        _default_effort = kwargs.get("default_reasoning_effort")
        self.default_reasoning_effort: Optional[str] = (
            str(_default_effort).strip().lower() if _default_effort else None
        )
        if not self.thinking_enabled:
            self.default_reasoning_effort = None

        # Prompt Formatting defaults
        self.user_name = kwargs.get("user_name", "user")
        self.ai_name = kwargs.get("ai_name", "assistant")
        self.start_header_id_template = "!@>"
        self.end_header_id_template = ": "
        self.system_message_template = "system"


        # ── Cancellation state ────────────────────────────────────────────────
        # A threading.Event so it is safe to set() from any thread (including
        # the async event loop via call_soon_threadsafe) and check from the
        # blocking generation thread.
        import threading
        self._cancel_event = threading.Event()

        # build text processor
        self.tp = LollmsTextProcessor(self)

    @staticmethod
    def normalize_reasoning_effort(
        think: Optional[bool],
        reasoning_effort: Optional[Union[str, bool]],
    ) -> Optional[str]:
        """
        Resolve the effective reasoning effort.

        Rule:
          - If think is not True: thought is completely deactivated (returns None).
          - If think is True: inspects the effort parameter. If effort is None or
            in ("none", "off", "disabled", "false", "0", ""), thought is completely
            deactivated (returns None). Otherwise returns the valid normalized effort string.
        """
        if think is not True:
            return None

        if reasoning_effort is None:
            return None

        if isinstance(reasoning_effort, bool):
            return "high" if reasoning_effort else None

        effort = str(reasoning_effort).strip().lower()
        if effort in ("none", "off", "disabled", "false", "0", ""):
            return None

        valid = {"low", "medium", "high", "max"}
        if effort in valid:
            return effort

        ASCIIColors.warning(
            f"Unknown reasoning_effort '{reasoning_effort}'. Falling back to 'low'."
        )
        return "low"

    @staticmethod
    def translate_reasoning_effort(
        reasoning_effort: Optional[Union[str, bool]] = None,
        supported_efforts: Optional[List[str]] = None,
        think: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Translates an incoming reasoning effort setting to a level supported by the model.

        Rule:
          - If think is explicitly False: thought is completely deactivated (returns None).
          - If think is False or reasoning_effort is disabled: returns None.
          - Otherwise projects to supported_efforts.
        """
        if think is False:
            return None

        if reasoning_effort is None or reasoning_effort is False:
            if think is True:
                raw_str = "high"
            else:
                return None
        elif reasoning_effort is True:
            raw_str = "high"
        else:
            raw_str = str(reasoning_effort).strip().lower()

        disabled_literals = {"none", "off", "disabled", "false", "0", ""}
        if raw_str in disabled_literals:
            if supported_efforts:
                for cand in supported_efforts:
                    if cand.strip().lower() in disabled_literals:
                        return cand
            return None

        if not supported_efforts:
            return LollmsLLMBinding.normalize_reasoning_effort(think, raw_str)

        supported_lower_map = {s.strip().lower(): s for s in supported_efforts if s}
        if not supported_lower_map:
            return LollmsLLMBinding.normalize_reasoning_effort(think, raw_str)

        if raw_str in supported_lower_map:
            return supported_lower_map[raw_str]

        anchor_scores = {
            "none": 0.0,
            "off": 0.0,
            "false": 0.0,
            "disabled": 0.0,
            "minimal": 0.15,
            "min": 0.15,
            "low": 0.30,
            "medium": 0.60,
            "med": 0.60,
            "default": 0.60,
            "high": 0.85,
            "on": 0.85,
            "true": 0.85,
            "max": 1.00,
            "maximum": 1.00,
            "extreme": 1.00,
        }

        target_score = anchor_scores.get(raw_str, 0.60)

        active_candidates = [
            s for s in supported_efforts
            if s and s.strip().lower() not in disabled_literals
        ]
        if not active_candidates:
            active_candidates = list(supported_efforts)

        def score_candidate(cand: str, idx: int, total: int) -> float:
            lowered = cand.strip().lower()
            if lowered in anchor_scores:
                return anchor_scores[lowered]
            return idx / max(1, total - 1)

        best_cand = active_candidates[0]
        min_distance = float("inf")
        best_cand_score = -1.0

        total_cands = len(active_candidates)
        for idx, cand in enumerate(active_candidates):
            c_score = score_candidate(cand, idx, total_cands)
            dist = abs(c_score - target_score)
            if dist < min_distance or (abs(dist - min_distance) < 1e-5 and c_score > best_cand_score):
                min_distance = dist
                best_cand = cand
                best_cand_score = c_score

        return best_cand

    def get_effective_reasoning_effort(
        self,
        think: Optional[bool] = None,
        reasoning_effort: Optional[Union[str, bool]] = None,
    ) -> Optional[str]:
        return self.translate_reasoning_effort(
            reasoning_effort=reasoning_effort,
            supported_efforts=getattr(self, "supported_reasoning_efforts", None),
            think=think,
        )

    @staticmethod
    def get_deactivate_thinking_payload(model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns parameters for OpenAI, vLLM, Ollama, Z.AI, and SGLang backends to strictly
        deactivate thinking when think is False and reasoning_effort is None.
        """
        m_lower = (model_name or "").lower()
        is_glm = "glm" in m_lower
        is_qwen = "qwen" in m_lower

        extra: Dict[str, Any] = {
            "thinking": False,
            "chat_template_kwargs": {
                "thinking": False,
                "enable_thinking": False,
            },
        }
        if is_glm:
            extra["thinking"] = {"type": "disabled"}
            extra["chat_template_kwargs"]["enable_thinking"] = False
            extra["chat_template_kwargs"]["thinking"] = False
        elif is_qwen:
            extra["chat_template_kwargs"]["enable_thinking"] = False

        return extra

    # ── Cancellation API ─────────────────────────────────────────────────────

    def cancel(self) -> None:
        """
        Signal the binding to stop the current generation as soon as possible.

        The base implementation sets a threading.Event that:
          - is checked by the default streaming callback guard (see
            _is_cancelled() below), and
          - should be checked inside any subclass generate_text() /
            generate_from_messages() loop.

        Subclasses that hold a live HTTP session (e.g. httpx, requests) should
        override this method, close/cancel that session, and then call
        super().cancel() so the event is set too.

        Example override for an httpx-based binding::

            def cancel(self) -> None:
                if self._http_client is not None:
                    self._http_client.close()   # aborts in-flight request
                super().cancel()
        """
        self._cancel_event.set()

    def reset_cancel(self) -> None:
        """
        Clear the cancellation flag so the binding is ready for a new generation.
        Must be called before each new generate_text() / generate_from_messages()
        call (the server layer is responsible for doing this).
        """
        self._cancel_event.clear()

    def is_cancelled(self) -> bool:
        """
        Returns True if cancel() has been called and the flag has not yet been
        reset via reset_cancel().  Subclasses can call this inside tight
        generation loops as a cooperative cancellation check.
        """
        return self._cancel_event.is_set()

    @property
    def system_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.system_message_template}{self.end_header_id_template}"

    @property
    def user_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.user_name}{self.end_header_id_template}"

    @property
    def ai_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.ai_name}{self.end_header_id_template}"

    @abstractmethod
    def generate_text(self,
                    prompt: str,
                    images: Optional[List[str]] = None,
                    videos: Optional[List[str]] = None,
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
                    split:Optional[bool]=False, # put to true if the prompt is a discussion
                    user_keyword:Optional[str]="!@>user:",
                    ai_keyword:Optional[str]="!@>assistant:",
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[str] = None,
                    reasoning_summary: Optional[str] = None,
                    **kwargs
                    ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.
        """
        pass

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
                    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[str] = None,
                    reasoning_summary: Optional[str] = None,
                    tools: Optional[List[Dict]] = None,
                    tool_choice: Optional[Union[str, Dict]] = None,
                    **kwargs
                    ) -> Union[str, dict]:
        """
        Generate text from an OpenAI-style message list.

        The ``tools`` / ``tool_choice`` parameters follow the OpenAI Chat
        Completions contract: bindings that support them forward them to the
        provider without alteration. Bindings that do not MUST at minimum
        preserve the wire state of any assistant/tool messages (``tool_calls``,
        ``tool_call_id``, ``name``) so a caller-driven multi-round loop can
        continue.

        The return value is the assistant's text response. Any tool call
        produced by the model — whether native ``tool_calls`` or leaked
        textual DSML — is translated into Lollms ``<tool>{"name":...,
        "parameters":{...}}</tool>`` XML and appended to the response.
        """
        binding_name = getattr(self, "binding_name", "unknown")
        message = (
            f"Binding '{binding_name}' does not implement generate_from_messages(). "
            "Switch the model profile to a binding that supports tool calling "
            "(lollms, openai, vllm, ollama, or llama_cpp_server)."
        )
        ASCIIColors.error(f"[{binding_name}] {message}")
        return {"status": "error", "message": message}


    @staticmethod
    def _has_tool_dialogue(messages: List[Dict]) -> bool:
        """True if any message carries OpenAI-compatible tool wire state."""
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            if m.get("tool_calls"):
                return True
            if str(m.get("role", "")).lower() in ("tool", "function"):
                return True
        return False

    def _normalize_tool_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Preserve OpenAI tool wire state (``tool_calls``, ``tool_call_id``,
        ``name``) while merging leading system prompts.

        Tool-role messages and assistant messages that carry ``tool_calls``
        are deliberately NOT merged and NOT forced into user/assistant
        alternation — the OpenAI wire protocol already guarantees a valid
        order once tools participate. This keeps a caller-driven loop
        round-trippable through the binding.
        """
        system_parts: List[str] = []
        start = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    system_parts.append(c.strip())
                start = i + 1
            else:
                break

        normalized: List[Dict] = []
        if system_parts:
            normalized.append({"role": "system", "content": "\n\n".join(system_parts)})

        for msg in messages[start:]:
            role = str(msg.get("role", "user") or "user").lower()
            content = msg.get("content", "")

            entry: Dict[str, Any] = {"role": role, "content": content}
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
                if entry["content"] in (None, ""):
                    entry["content"] = None  # OpenAI permits null with tool_calls
            if msg.get("tool_call_id"):
                entry["tool_call_id"] = msg["tool_call_id"]
            if msg.get("name"):
                entry["name"] = msg["name"]
            if msg.get("images"):
                entry["images"] = list(msg["images"])

            normalized.append(entry)

        return normalized

    def clean_and_alternate_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Ensures that:
          1. An optional system prompt only appears at the very beginning (index 0).
             All initial consecutive system prompts are merged into a single system prompt.
          2. Any system prompts appearing later in the conversation are converted to user messages.
          3. Consecutive messages of the same role (e.g., user/user or assistant/assistant) are merged
             into a single message of that role with a double newline separator.
          4. The conversation must strictly alternate: system (optional), user, assistant, user, assistant...

        When the message list contains OpenAI-style tool wire state
        (``tool_calls`` / ``role="tool"``), normalisation is bypassed in favour
        of ``_normalize_tool_messages`` — the wire protocol is preserved
        verbatim and no alternation is enforced.
        """
        if not messages:
            return []

        if self._has_tool_dialogue(messages):
            return self._normalize_tool_messages(messages)

        # 1. Extract and merge all initial system messages
        system_contents = []
        first_non_system_index = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    if content.strip():
                        system_contents.append(content.strip())
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "").strip()
                            if text:
                                system_contents.append(text)
                first_non_system_index = i + 1
            else:
                break

        # 2. Build the normalized list starting with the merged system prompt if present
        normalized = []
        if system_contents:
            normalized.append({
                "role": "system",
                "content": "\n\n".join(system_contents)
            })

        # 3. Process the remaining messages
        for msg in messages[first_non_system_index:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Strip special tool-calling template tokens that cause llama-server parser crashes
            if isinstance(content, str):
                content = re.sub(r'\[TOOL_CALLS\][^\n]*\n?', '', content, flags=re.IGNORECASE)
                content = re.sub(r'\[TOOL_CALLS[^\n]*\n?', '', content, flags=re.IGNORECASE)
                content = re.sub(r'=== END (?:ACTIVE )?SKILLS ===\s*', '', content, flags=re.IGNORECASE).strip()

            # Convert any system message in the middle of the chat to user role
            if role == "system":
                role = "user"

            msg_entry = {
                "role": role,
                "content": content
            }
            if "images" in msg:
                msg_entry["images"] = list(msg["images"])
            if "active_images" in msg:
                msg_entry["active_images"] = list(msg["active_images"])

            normalized.append(msg_entry)

        # 4. Merge consecutive messages with the same role
        merged = []
        if normalized and normalized[0]["role"] == "system":
            merged.append(normalized[0])
            remaining = normalized[1:]
        else:
            remaining = normalized

        for msg in remaining:
            if not merged:
                merged.append(msg)
                continue

            prev = merged[-1]
            if prev["role"] == msg["role"]:
                prev_content = prev["content"]
                curr_content = msg["content"]

                def _to_list_of_blocks(content_val):
                    if isinstance(content_val, str):
                        return [{"type": "text", "text": content_val}] if content_val else []
                    elif isinstance(content_val, list):
                        return content_val
                    return []

                p1 = _to_list_of_blocks(prev_content)
                p2 = _to_list_of_blocks(curr_content)

                combined_parts = []
                for part in p1 + p2:
                    if part.get("type") == "text":
                        text_val = part.get("text", "").strip()
                        if text_val:
                            if combined_parts and combined_parts[-1].get("type") == "text":
                                combined_parts[-1]["text"] += "\n\n" + text_val
                            else:
                                combined_parts.append({"type": "text", "text": text_val})
                    else:
                        combined_parts.append(part)

                if len(combined_parts) == 1 and combined_parts[0]["type"] == "text":
                    prev["content"] = combined_parts[0]["text"]
                else:
                    prev["content"] = combined_parts

                # Merge image lists when combining messages of the same role
                if "images" in msg:
                    prev_images = prev.setdefault("images", [])
                    for img in msg["images"]:
                        if img not in prev_images:
                            prev_images.append(img)
                if "active_images" in msg:
                    prev_active = prev.setdefault("active_images", [])
                    prev_active.extend(msg["active_images"])
            else:
                merged.append(msg)

        # 5. Ensure strict alternation
        alternating = []
        if merged and merged[0]["role"] == "system":
            alternating.append(merged[0])
            remaining_merged = merged[1:]
        else:
            remaining_merged = merged

        for msg in remaining_merged:
            if not alternating or alternating[-1]["role"] == "system":
                if msg["role"] == "assistant":
                    alternating.append({"role": "user", "content": "Proceed."})
                alternating.append(msg)
            else:
                prev_role = alternating[-1]["role"]
                if prev_role == msg["role"]:
                    pass
                else:
                    alternating.append(msg)

        return alternating

    def normalize_model_name(self, model_name: str) -> str:
        return model_name.strip().lower()


    def set_forced_ctx_size(self, ctx_size: Optional[int]) -> None:
        """Dynamically updates the forced context size."""
        if ctx_size is not None:
            try:
                self.forced_ctx_size = int(ctx_size)
            except (ValueError, TypeError):
                self.forced_ctx_size = None
        else:
            self.forced_ctx_size = None

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        # Layer 1: Forced Context Size
        if self.forced_ctx_size is not None:
            return self.forced_ctx_size

        # Layer 2: Binding-specific implementation
        if hasattr(self, "_get_ctx_size") and callable(self._get_ctx_size):
            try:
                binding_ctx = self._get_ctx_size(model_name=model_name)
                if binding_ctx is not None:
                    return binding_ctx
            except Exception as e:
                ASCIIColors.warning(f"[{self.binding_name}] _get_ctx_size failed: {e}")

        # Layer 3: Hardcoded List
        if model_name is None:
            model_name = self.model_name

        normalized = self.normalize_model_name(model_name)
        known_contexts = load_known_contexts()

        exact = {
            self.normalize_model_name(k): v
            for k, v in known_contexts.get("exact", {}).items()
        }
        aliases = {
            self.normalize_model_name(k): self.normalize_model_name(v)
            for k, v in known_contexts.get("aliases", {}).items()
        }
        prefix = {
            self.normalize_model_name(k): v
            for k, v in known_contexts.get("prefix", {}).items()
        }

        if normalized in exact:
            context_size = exact[normalized]
            ASCIIColors.warning(
                f"Using exact context size for model '{model_name}': {context_size}"
            )
            return context_size

        if normalized in aliases:
            aliased_name = aliases[normalized]
            context_size = exact.get(aliased_name)
            if context_size is not None:
                ASCIIColors.warning(
                    f"Using aliased context size for model '{model_name}' "
                    f"via '{aliased_name}': {context_size}"
                )
                return context_size

        for prefix_name in sorted(prefix.keys(), key=len, reverse=True):
            if normalized.startswith(prefix_name):
                context_size = prefix[prefix_name]
                ASCIIColors.warning(
                    f"Using prefix context size for model '{model_name}' "
                    f"via '{prefix_name}': {context_size}"
                )
                return context_size

        # Layer 4: Default Fallback
        ASCIIColors.warning(
            f"Context size not found for model '{model_name}', "
            f"falling back to default: {self.default_ctx_size}"
        )
        return self.default_ctx_size


    @abstractmethod
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens.
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.
        """        
        pass

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.
        """
        try:
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e}")
            return 256
    @abstractmethod
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        """
        pass    
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Return information about the current model.
        """
        pass

    def get_zoo(self) -> List[Dict[str, Any]]:
        """
        Returns a list of models available for download.
        each entry is a dict with:
        name, description, size, type, link
        """
        return []

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Downloads a model from the zoo using its index.
        """
        return {"status": False, "message": "Not implemented"}

    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.
        """
        pass

    # ── OpenAI-compatible tool helpers (shared by all bindings) ─────────
    # These thin wrappers de-duplicate the DSML / native tool-call plumbing
    # that every provider binding must perform. They are pure functions
    # (no instance state) and safe to call from any binding subclass.

    # Known OpenAI-compatible server misconfiguration signatures, mapped to
    # actionable remediation hints. Populated by ``_classify_tool_server_error``
    # so every binding surfaces the SAME diagnostic for the SAME root cause.
    _TOOL_SERVER_ERROR_SIGNATURES: List[Tuple[str, str]] = [
        (
            '"auto" tool choice requires --enable-auto-tool-choice',
            (
                "vLLM was launched WITHOUT tool-calling support. "
                "Relaunch the vLLM server with:\n"
                "    --enable-auto-tool-choice --tool-call-parser <family>\n"
                "where <family> matches the model (e.g. `deepseek_v3` for "
                "DeepSeek, `hermes` for Qwen/Hermes-style, `mistral` for "
                "Mistral, `llama3_json` for Llama-3). Without these flags "
                "vLLM rejects any request that sets tool_choice=\"auto\"."
            ),
        ),
        (
            "requires --enable-auto-tool-choice and --tool-call-parser",
            (
                "vLLM was launched WITHOUT tool-calling support. "
                "Relaunch the vLLM server with:\n"
                "    --enable-auto-tool-choice --tool-call-parser <family>\n"
                "where <family> matches the model (e.g. `deepseek_v3`, "
                "`hermes`, `mistral`, `llama3_json`)."
            ),
        ),
        (
            "tool-call-parser",
            (
                "The OpenAI-compatible server rejected the request because "
                "no tool-call parser is configured. For vLLM add "
                "`--enable-auto-tool-choice --tool-call-parser <family>`; "
                "for llama.cpp / Ollama load a tool-capable chat template."
            ),
        ),
        (
            "Model returned an empty response with tools enabled",
            (
                "The server returned an EMPTY response when tools were "
                "requested. Two likely causes:\n"
                "  1. vLLM launched without "
                "`--enable-auto-tool-choice --tool-call-parser <family>`.\n"
                "  2. The model's chat template does not expose the tools "
                "array (llama.cpp / Ollama need a tool-capable template).\n"
                "Check the server log for the actual tool-parsing messages."
            ),
        ),
    ]

    @staticmethod
    def _classify_tool_server_error(error_or_body: Any) -> Optional[str]:
        """Return an actionable remediation string for known server-side
        tool-calling misconfigurations, or ``None`` if unrecognised.

        Accepts an exception, an error dict (``{"error": {"message": ...}}``),
        a plain string, or the ``server_body`` attribute of an
        ``openai.BadRequestError`` / ``openai.APIError``.
        """
        if error_or_body is None:
            return None

        # Flatten the error into a searchable string.
        text_parts: List[str] = []
        if isinstance(error_or_body, str):
            text_parts.append(error_or_body)
        elif isinstance(error_or_body, dict):
            err = error_or_body.get("error")
            if isinstance(err, dict):
                text_parts.append(str(err.get("message", "")))
                text_parts.append(str(err.get("code", "")))
            text_parts.append(str(error_or_body.get("message", "")))
        else:
            text_parts.append(str(error_or_body))
            body = getattr(error_or_body, "body", None)
            if body is not None:
                text_parts.append(str(body))
            text_parts.append(str(getattr(error_or_body, "message", "")))

        haystack = " ".join(t for t in text_parts if t).lower()
        if not haystack.strip():
            return None

        for needle, hint in LollmsLLMBinding._TOOL_SERVER_ERROR_SIGNATURES:
            if needle.lower() in haystack:
                return hint
        return None

    @staticmethod
    def tool_calls_to_lollms_xml(tool_calls: List[Any]) -> str:
        """Convert OpenAI-native ``tool_calls`` objects to Lollms ``<tool>`` XML."""
        if not tool_calls:
            return ""
        parts: List[str] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name", "") or ""
                raw_args = fn.get("arguments", "") or ""
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn is not None else ""
                raw_args = getattr(fn, "arguments", "") if fn is not None else ""
                name = name or ""
                raw_args = raw_args or ""
            if not name:
                continue
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {"raw": raw_args}
            parts.append(f"<tool>{json.dumps({'name': name, 'parameters': tool_args})}</tool>")
        return "\n".join(parts)

    @staticmethod
    def accumulated_tool_calls_to_lollms_xml(accumulated: Dict[int, Dict]) -> str:
        """Convert accumulated streamed tool-call deltas to Lollms ``<tool>`` XML."""
        if not accumulated:
            return ""
        parts: List[str] = []
        for idx in sorted(accumulated.keys()):
            tc = accumulated[idx] or {}
            name = (tc.get("name") or "").strip()
            if not name:
                continue
            raw_args = tc.get("arguments") or ""
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {"raw": raw_args}
            parts.append(f"<tool>{json.dumps({'name': name, 'parameters': tool_args})}</tool>")
        return "\n".join(parts)

    @staticmethod
    def collect_streamed_tool_deltas(accumulated: Dict[int, Dict], delta_tool_calls: Any) -> None:
        """Accumulate OpenAI-compatible streamed tool_call deltas by index."""
        if not delta_tool_calls:
            return
        for delta_tc in delta_tool_calls:
            if isinstance(delta_tc, dict):
                idx = delta_tc.get("index", 0) or 0
                tc_id = delta_tc.get("id")
                fn = delta_tc.get("function") or {}
                name_frag = fn.get("name") or ""
                args_frag = fn.get("arguments") or ""
            else:
                idx = getattr(delta_tc, "index", 0) or 0
                tc_id = getattr(delta_tc, "id", None)
                fn = getattr(delta_tc, "function", None)
                name_frag = getattr(fn, "name", "") if fn is not None else ""
                args_frag = getattr(fn, "arguments", "") if fn is not None else ""
                name_frag = name_frag or ""
                args_frag = args_frag or ""

            slot = accumulated.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            if tc_id and not slot.get("id"):
                slot["id"] = tc_id
            if name_frag:
                slot["name"] = (slot.get("name") or "") + name_frag
            if args_frag:
                slot["arguments"] = (slot.get("arguments") or "") + args_frag

    @staticmethod
    def extract_native_tool_calls(message_obj: Any) -> List[Any]:
        """Extract native ``tool_calls`` from a non-streaming message object or dict."""
        if message_obj is None:
            return []
        if isinstance(message_obj, dict):
            return list(message_obj.get("tool_calls") or [])
        return list(getattr(message_obj, "tool_calls", None) or [])

    def split_discussion(self, lollms_prompt_string: str, system_keyword="!@>system:", user_keyword="!@>user:", ai_keyword="!@>assistant:") -> list:
        """
        Splits a LoLLMs prompt into a list of OpenAI-style messages.
        """
        pattern = r"(?={}|{}|{})".format(
            re.escape(system_keyword),
            re.escape(user_keyword),
            re.escape(ai_keyword)
        )
        parts = re.split(pattern, lollms_prompt_string)
        messages = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.startswith(system_keyword):
                role = "system"
                content = part[len(system_keyword):].strip()
            elif part.startswith(user_keyword):
                role = "user"
                content = part[len(user_keyword):].strip()
            elif part.startswith(ai_keyword):
                role = "assistant"
                content = part[len(ai_keyword):].strip()
            else:
                if not messages:
                    role = "system"
                    content = part
                else:
                    continue

            messages.append({"role": role, "content": content})
            if messages[-1]["content"]=="":
                del messages[-1]
        return messages
    
    def ps(self):
        """
        List models (simulating a process status command).
        Since Lollms/OpenAI API doesn't have a specific 'ps' endpoint for running models with memory stats,
        we list available models and populate structure with available info, leaving hardware stats empty.
        """
        # Since there is no dedicated ps endpoint to see *running* models in the standard OpenAI API,
        # we list available models and try to map relevant info.
        models = self.list_models()
        standardized_models = []
        for m in models:
            standardized_models.append({
                "model_name": m.get("model_name"),
                "size": None,
                "vram_size": None,
                "gpu_usage_percent": None,
                "cpu_usage_percent": None,
                "expires_at": None,
                "parameters_size": None,
                "quantization_level": None,
                "parent_model": None,
                "context_size": m.get("context_length"),
                "owned_by": m.get("owned_by"),
                "created": m.get("created")
            })
        return standardized_models
        
    def get_context_size(self) -> Optional[int]:
        """
        Returns the default context size for the binding.
        """
        return self.default_ctx_size


class LollmsLLMBindingManager:
    """Manages binding discovery and instantiation"""

    def __init__(self, llm_bindings_dir: Union[str, Path] = Path(__file__).parent / "llm_bindings"):
        self.llm_bindings_dir = Path(llm_bindings_dir)
        self.available_bindings = {}
        self.load_errors: Dict[str, str] = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific binding implementation from the llm bindings directory."""
        binding_dir = self.llm_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.llm_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
                self.load_errors.pop(binding_name, None)
            except Exception as e:
                trace_exception(e)
                self.load_errors[binding_name] = f"{type(e).__name__}: {e}"
                print(f"Failed to load binding {binding_name}: {self.load_errors[binding_name]}")

    def create_binding(self, 
                      binding_name: str,
                      **kwargs) -> Optional[LollmsLLMBinding]:
        """
        Create an instance of a specific binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        if binding_name in self.load_errors:
            raise RuntimeError(
                f"Failed to load LLM binding '{binding_name}': {self.load_errors[binding_name]}. "
                "Ensure the binding's optional Python dependencies are installed."
            )
        return None
    @staticmethod
    def _get_fallback_description(binding_name: str) -> Dict:
        """
        Generates a default description dictionary for a binding without a description.yaml file.
        """
        return {
            "binding_name": binding_name,
            "title": binding_name.replace("_", " ").title(),
            "author": "Unknown",
            "creation_date": "N/A",
            "last_update_date": "N/A",
            "description": f"A binding for {binding_name}. No description.yaml file was found, so common parameters are shown as a fallback.",
            "input_parameters": [
                {
                    "name": "model_name",
                    "type": "str",
                    "description": "The model name, ID, or filename to be used.",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "host_address",
                    "type": "str",
                    "description": "The host address of the service (for API-based bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "models_path",
                    "type": "str",
                    "description": "The path to the models directory (for local bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "service_key",
                    "type": "str",
                    "description": "The API key or service key for authentication (if applicable).",
                    "mandatory": False,
                    "default": ""
                }
            ]
        }

    @staticmethod
    def get_bindings_list(llm_bindings_dir: Union[str, Path]=None) -> List[Dict]:
        """
        Lists all available LLM bindings by scanning a directory.
        """
        if not llm_bindings_dir:
            llm_bindings_dir = Path(__file__).parent/"llm_bindings"

        bindings_dir = Path(llm_bindings_dir)
        if not bindings_dir.is_dir():
            return []

        bindings_list = []
        for binding_folder in bindings_dir.iterdir():
            if binding_folder.is_dir() and (binding_folder / "__init__.py").exists():
                binding_name = binding_folder.name
                description_file = binding_folder / "description.yaml"
                
                binding_info = {}
                if description_file.exists():
                    try:
                        with open(description_file, 'r', encoding='utf-8') as f:
                            binding_info = yaml.safe_load(f)
                        binding_info['binding_name'] = binding_name
                    except Exception as e:
                        print(f"Error loading description.yaml for {binding_name}: {e}")
                        binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))
    
    def get_available_bindings(self) -> List[Dict]:
        """
        Retrieves a list of all available LLM bindings with their full descriptions.
        """
        return LollmsLLMBindingManager.get_bindings_list(self.llm_bindings_dir)

def get_available_bindings(llm_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    """
    Lists all available LLM bindings with their detailed descriptions.
    """
    if llm_bindings_dir is None:
        llm_bindings_dir = Path(__file__).parent / "llm_bindings"
    return LollmsLLMBindingManager.get_bindings_list(llm_bindings_dir)

def list_binding_models(llm_binding_name: str, llm_binding_config: Optional[Dict[str, any]]|None = None, llm_bindings_dir: str|Path = Path(__file__).parent / "llm_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsLLMBindingManager(llm_bindings_dir).create_binding(
        binding_name=llm_binding_name,
        **{
            k: v
            for k, v in (llm_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
