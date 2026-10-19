"""
lollms_connection_binding.py
Base binding for communication services (Discord, Telegram, Slack, WhatsApp, etc.).

Follows the same pattern as LollmsTTIBinding / LollmsSTTBinding:
- Abstract base class `LollmsConnectionBinding` (extends LollmsBaseBinding)
- Manager `LollmsConnectionBindingManager` with dynamic discovery
- Each binding has a `description.yaml` with global_input_parameters and
  model_input_parameters (renamed to instance_input_parameters conceptually,
  but keeping the same YAML schema for UI compatibility)

A connection binding maps to ONE service endpoint with ONE active "instance"
(e.g., a specific bot on Discord, or a specific chat group on Telegram).
Profiles layer on top: LOLLMS_CONNECTION_PROFILES_<alias>_<param> with
instance_name specifying which chat/channel/conversation to target.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import mimetypes
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_base_binding import LollmsBaseBinding

# ── Result Containers ──────────────────────────────────────────────────────

class ConnectionSendResult(dict):
    """
    Unified result from sending a message via a connection binding.

    Keys:
        sent: bool                  -> True if message was delivered
        message_id: Optional[str]   -> Platform-assigned message ID
        channel: Optional[str]      -> Resolved channel/conversation name
        raw: Optional[Any]          -> Raw platform response
        error: Optional[str]        -> Error message if failed
    """
    def __init__(self, sent=None, message_id=None, channel=None, raw=None, error=None, **extra):
        if sent is None:
            sent = bool(extra.pop("success", False))
        super().__init__(
            sent=sent,
            message_id=message_id,
            channel=channel,
            raw=raw,
            error=error,
            **extra,
        )

    @property
    def sent(self) -> bool:
        return self.get("sent", False)

    @property
    def error(self) -> Optional[str]:
        return self.get("error")


class ConnectionReceiveResult(dict):
    """
    Container for a message received via a connection binding.
    Keys: content, sender, channel, timestamp, raw
    """
    def __init__(self, content=None, sender=None, channel=None, timestamp=None, raw=None):
        super().__init__(
            content=content or "",
            sender=sender,
            channel=channel,
            timestamp=timestamp,
            raw=raw,
        )

    @property
    def content(self) -> str:
        return self.get("content", "")


# ── Base Class ─────────────────────────────────────────────────────────────

class LollmsConnectionBinding(LollmsBaseBinding):
    """
    Abstract base class for all LoLLMS connection bindings.

    A connection binding represents a link to a communication platform.
    It can send messages to a configured channel/conversation and optionally
    listen for incoming messages.

    Multi-instance support: The `instance_name` kwarg identifies the specific
    conversation/channel within the platform (e.g., "#general" on Slack,
    chat_id on Telegram). Profiles map instance_name to distinct targets.
    """

    def __init__(
        self,
        binding_name: str = "unknown",
        debug: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)
        self.host_address = kwargs.get("host_address", "")
        self.service_key = kwargs.get("service_key", "")
        self.instance_name = kwargs.get("instance_name", "")
        self._connected = False
        self._message_handlers: List[Callable] = []
        self.artefacts_enabled = bool(kwargs.get("artefacts_enabled", True))
        self.max_artefact_size = int(kwargs.get("max_artefact_size", 10 * 1024 * 1024))
        self.artefacts_enabled = bool(kwargs.get("artefacts_enabled", True))
        self.max_artefact_size = int(kwargs.get("max_artefact_size", 10 * 1024 * 1024))

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Establish the connection to the platform. Returns True on success.
        Subclasses that require an active handshake should override this."""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Gracefully close the connection. Default implementation."""
        self._connected = False
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Core Operations ────────────────────────────────────────────────────

    @abstractmethod
    def send_message(
        self,
        content: str,
        sender_name: Optional[str] = None,
        artefacts: Optional[List[dict]] = None,
        **kwargs,
    ) -> ConnectionSendResult:
        """
        Send a message to the configured channel/conversation.

        Args:
            content: The message text.
            sender_name: Optional display name override.
            artefacts: Optional list of artefact dicts to attach. Schema:
                {
                    "title": str,             # display name / filename
                    "type": str,              # "document" | "code" | "image" | ...
                    "path": Optional[str],    # local file path (used when content is absent)
                    "content": Optional[str], # text content, or base64 for binary mimes
                    "mime": str,              # e.g. "text/markdown", "image/png"
                }
            **kwargs: Platform-specific parameters (e.g., attachments, embeds).

        Returns:
            ConnectionSendResult with sent status and metadata.
        """
        pass

    _ARTEFACT_BLOCK_RE = re.compile(
        r"<art(?:ef)?act\b([^>]*)>(.*?)</art(?:ef)?act>",
        re.IGNORECASE | re.DOTALL,
    )
    _ARTEFACT_ATTR_RE = re.compile(r"(\w+)\s*=\s*[\"']([^\"']*)[\"']")

    def _extract_artefacts_from_text(self, text: str) -> Tuple[str, List[dict]]:
        """Splits LLM output into clean text and a list of artefact dicts.

        Recognizes <artefact type="..." title="..." path="...">content</artefact>
        blocks (the "artifact" spelling is tolerated too). Blocks are removed
        from the visible text and returned as artefact dicts. Returns the
        original text unchanged when artefacts are disabled or nothing matches.
        """
        if not text or not getattr(self, "artefacts_enabled", True):
            return text, []
        if not self._ARTEFACT_BLOCK_RE.search(text):
            return text, []

        artefacts: List[dict] = []

        def _capture(match) -> str:
            attrs = dict(self._ARTEFACT_ATTR_RE.findall(match.group(1) or ""))
            content = (match.group(2) or "").strip()
            artefact_type = str(attrs.get("type", "document")).strip().lower() or "document"
            title = str(attrs.get("title") or attrs.get("name") or "").strip() or f"artefact_{len(artefacts) + 1}"
            path = str(attrs.get("path", "")).strip() or None
            mime = "text/markdown" if artefact_type == "document" else "text/plain"
            if path:
                guessed, _ = mimetypes.guess_type(path)
                mime = guessed or mime
            artefacts.append(
                {
                    "title": title,
                    "type": artefact_type,
                    "path": path,
                    "content": content,
                    "mime": mime,
                }
            )
            return ""

        clean_text = self._ARTEFACT_BLOCK_RE.sub(_capture, text)
        return clean_text.strip(), artefacts

    def _load_artefact_payload(self, artefact: dict) -> Tuple[Optional[bytes], str, str]:
        """Normalizes an artefact dict into (raw_bytes, mime, title) for delivery.

        Reads `content` (raw text for text/* mimes, base64 otherwise) or the
        local file at `path`. Enforces `max_artefact_size`. Never raises; the
        bytes component is None when the artefact must be skipped.
        """
        title = str(artefact.get("title") or "artefact")
        mime = str(artefact.get("mime") or "text/plain")
        max_size = getattr(self, "max_artefact_size", 0) or 0
        try:
            data: Optional[bytes] = None
            content = artefact.get("content")
            path = artefact.get("path")
            if content is not None and str(content).strip():
                raw = str(content)
                if mime.startswith("text/"):
                    data = raw.encode("utf-8")
                else:
                    data = base64.b64decode(raw, validate=False)
            elif path:
                file_path = Path(str(path))
                if not file_path.is_file():
                    ASCIIColors.warning(f"[{self.binding_name}] Artefact file not found: {path}")
                    return None, mime, title
                data = file_path.read_bytes()

            if data is None:
                return None, mime, title
            if max_size and len(data) > max_size:
                ASCIIColors.warning(
                    f"[{self.binding_name}] Artefact '{title}' exceeds max_artefact_size "
                    f"({len(data)} > {max_size} bytes). Skipped."
                )
                return None, mime, title
            return data, mime, title
        except Exception as e:
            ASCIIColors.warning(f"[{self.binding_name}] Failed to load artefact '{title}': {e}")
            return None, mime, title


    def receive_message(
        self,
        timeout: int = 0,
        **kwargs,
    ) -> Optional[ConnectionReceiveResult]:
        """
        Listen for a single incoming message. Non-blocking if timeout=0.

        Default implementation returns None (bindable bindings that don't
        support listening override this).

        Args:
            timeout: Seconds to wait. 0 = return immediately.
            **kwargs: Platform-specific filters.

        Returns:
            ConnectionReceiveResult or None.
        """
        return None

    def on_message(self, callback: Callable) -> None:
        """Register a callback for incoming messages."""
        self._message_handlers.append(callback)

    def off_message(self, callback: Callable) -> None:
        """Unregister a message callback."""
        if callback in self._message_handlers:
            self._message_handlers.remove(callback)

    def _dispatch_message(self, result: ConnectionReceiveResult) -> None:
        """Internal: dispatch a received message to all registered handlers."""
        for handler in self._message_handlers:
            try:
                handler(result)
            except Exception as e:
                ASCIIColors.warning(
                    f"[{self.binding_name}] Message handler error: {e}"
                )

    # ── Discovery / Listing ────────────────────────────────────────────────

    @abstractmethod
    def list_channels(self) -> List[Dict[str, Any]]:
        """
        List available channels, conversations, or rooms on this connection.
        Returns a list of dicts with at least {"id", "name", "type"}.
        """
        pass

    def list_models(self) -> List[Any]:
        """List available model profiles (required by LollmsBaseBinding).
        Defaults to the platform's channels."""
        return self.list_channels()

    def get_zoo(self) -> List[Dict[str, Any]]:
        return []

    def download_from_zoo(self, index: int, progress_callback=None) -> dict:
        return {"status": False, "message": "Not implemented"}

    # ── Settings ──────────────────────────────────────────────────────────

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.config.update(settings)
        return True


# ── Manager ────────────────────────────────────────────────────────────────

class LollmsConnectionBindingManager:
    """
    Manages connection binding discovery and instantiation.
    Scans connection_bindings/ for packages with __init__.py and loads
    the class named ConnectionBindingName.
    """

    def __init__(
        self,
        connection_bindings_dir: Union[str, Path] = Path(__file__).parent / "connection_bindings",
    ):
        self.connection_bindings_dir = Path(connection_bindings_dir)
        self.available_bindings: Dict[str, type] = {}

    def _load_binding(self, binding_name: str) -> None:
        """Dynamically load a specific connection binding implementation."""
        binding_dir = self.connection_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(
                    f"lollms_client.connection_bindings.{binding_name}"
                )
                binding_class = self._resolve_concrete_class(module, binding_name)
                if binding_class is not None:
                    self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                ASCIIColors.warning(
                    f"Failed to load Connection binding {binding_name}: {e}"
                )

    def _resolve_concrete_class(self, module: Any, binding_name: str) -> Optional[type]:
        preferred_name = getattr(module, "BindingName", None)
        preferred_cls = getattr(module, preferred_name, None) if preferred_name else None
        if self._is_concrete_binding_class(preferred_cls):
            return preferred_cls

        for attr_name in dir(module):
            member_cls = getattr(module, attr_name)
            if self._is_concrete_binding_class(member_cls):
                return member_cls

        ASCIIColors.warning(
            f"Connection binding '{binding_name}' exposes no concrete "
            "LollmsConnectionBinding subclass."
        )
        return None

    @staticmethod
    def _is_concrete_binding_class(cls: Any) -> bool:
        return (
            inspect.isclass(cls)
            and issubclass(cls, LollmsConnectionBinding)
            and cls is not LollmsConnectionBinding
            and not inspect.isabstract(cls)
        )

    def create_binding(
        self,
        binding_name: str,
        **kwargs,
    ) -> Optional[LollmsConnectionBinding]:
        """
        Create an instance of a specific connection binding.
        Automatically calls connect() on the created binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                binding = binding_class(**kwargs)
                binding.connect()
                return binding
            except Exception as e:
                trace_exception(e)
                ASCIIColors.warning(
                    f"Failed to instantiate Connection binding {binding_name}: {e}"
                )
                return None
        return None

    def get_available_bindings(self) -> List[str]:
        """Return list of available connection binding names."""
        return [
            binding_dir.name
            for binding_dir in self.connection_bindings_dir.iterdir()
            if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()
        ]

    @staticmethod
    def get_bindings_list(
        connection_bindings_dir: Union[str, Path],
    ) -> List[Dict]:
        """List all available connection bindings with their descriptions."""
        import yaml

        bindings_dir = Path(connection_bindings_dir)
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
                        with open(description_file, "r", encoding="utf-8") as f:
                            binding_info = yaml.safe_load(f)
                        binding_info["binding_name"] = binding_name
                    except Exception as e:
                        ASCIIColors.warning(
                            f"Error loading description.yaml for {binding_name}: {e}"
                        )
                        binding_info = _get_fallback_description(binding_name)
                else:
                    binding_info = _get_fallback_description(binding_name)

                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get("title", b["binding_name"]))


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_fallback_description(binding_name: str) -> Dict:
    """Fallback description when description.yaml is missing."""
    return {
        "binding_name": binding_name,
        "title": binding_name.replace("_", " ").title(),
        "author": "Unknown",
        "version": "N/A",
        "description": f"A connection binding for {binding_name}. No description.yaml found.",
        "global_input_parameters": [
            {
                "name": "host_address",
                "type": "str",
                "description": "Platform API base URL or bot endpoint.",
                "mandatory": True,
                "default": "",
            },
            {
                "name": "service_key",
                "type": "str",
                "description": "API key, bot token, or webhook URL.",
                "mandatory": False,
                "default": "",
            },
        ],
        "model_input_parameters": [
            {
                "name": "instance_name",
                "type": "str",
                "description": "Target channel, room, or conversation identifier.",
                "mandatory": False,
                "default": "",
            },
        ],
    }


def get_available_bindings(
    connection_bindings_dir: Union[str, Path] = None,
) -> List[Dict]:
    """Module-level helper to list all available connection bindings."""
    if connection_bindings_dir is None:
        connection_bindings_dir = Path(__file__).parent / "connection_bindings"
    return LollmsConnectionBindingManager.get_bindings_list(connection_bindings_dir)


def list_binding_channels(
    connection_binding_name: str,
    connection_binding_config: Optional[Dict[str, Any]] = None,
    connection_bindings_dir: Union[str, Path] = Path(__file__).parent / "connection_bindings",
) -> List[Dict]:
    """
    Lists all available channels for a specific connection binding.
    Creates a temporary binding instance to list channels.
    """
    binding = LollmsConnectionBindingManager(connection_bindings_dir).create_binding(
        binding_name=connection_binding_name,
        **{
            k: v
            for k, v in (connection_binding_config or {}).items()
            if k != "binding_name"
        },
    )
    return binding.list_channels() if binding else []