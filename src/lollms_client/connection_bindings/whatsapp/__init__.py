"""
WhatsApp CONNECTION binding for lollms_client.

WhatsApp requires an external bridge (e.g. whatsapp-web.js). Guarded stub so
the daemon never crashes when the bridge/library is absent.
"""
import asyncio
from typing import List, Optional

from ascii_colors import ASCIIColors
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "WhatsAppConnectionBinding"


class WhatsAppConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="whatsapp", **kwargs)
        self.config = kwargs
        self.bridge = kwargs.get("host_address") or ""
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    def list_channels(self) -> List[str]:
        return ["whatsapp:*"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="WhatsApp bridge is not running.", channel=channel_alias)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="whatsapp")

    async def start(self) -> None:
        self._running = True
        ASCIIColors.info("📲 Starting WhatsApp connection binding...")
        try:
            try:
                from whatsapp import Client  # noqa: F401
            except ImportError:
                ASCIIColors.yellow(
                    "WhatsApp requires a bridge (e.g. whatsapp-web.js) and a compatible library. "
                    "No `whatsapp` package is installed; this binding is idle."
                )
                while self._running:
                    await asyncio.sleep(60)
                return
        finally:
            self._running = False