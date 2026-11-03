# lollms_client/ttv_bindings/lollms/__init__.py
"""
Placeholder TTV binding for the standard LOLLMS server.

The LOLLMS server does not yet expose a text-to-video endpoint. This binding
exists so that the TTV binding manager can resolve the name "lollms" without
crashing, and so that users can see a clear NotImplementedError when they try
to use it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.lollms_ttv_binding import LollmsTTVBinding

BindingName = "LollmsTTVBinding_Impl"


class LollmsTTVBinding_Impl(LollmsTTVBinding):
    """Placeholder TTV binding for the standard LOLLMS server."""

    def __init__(self, **kwargs):
        super().__init__(binding_name="lollms", **kwargs)
        self.host_address = kwargs.get("host_address", "http://localhost:9600")
        self.model_name = kwargs.get("model_name", "main_video_model")
        self.service_key = kwargs.get("service_key", None)
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        ASCIIColors.warning(
            "LOLLMS TTV binding is a placeholder. The LOLLMS server does not yet "
            "expose a text-to-video endpoint."
        )

    def generate_video(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError(
            "LOLLMS TTV generate_video is not implemented yet. "
            "Use the 'diffusers' or 'minimax_h3' binding for local video generation."
        )

    def list_models(self, **kwargs) -> List[str]:
        return [self.model_name] if self.model_name else []

    # ── Resource management contract ────────────────────────────────────────
    def is_local(self) -> bool:
        return False

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        return True

    def has_active_resources(self) -> bool:
        return False

    def get_loaded_models(self) -> List[str]:
        return [self.model_name] if self.model_name else []

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        return True

    def install_model(self, model_name: str, **kwargs) -> dict:
        return {"status": False, "message": "Remote LoLLMS server; models are managed server-side."}

    def get_zoo(self) -> List[Dict[str, Any]]:
        return []

    def download_from_zoo(self, index: int, progress_callback=None) -> dict:
        return {"status": False, "message": "Remote LoLLMS server; models are managed server-side."}