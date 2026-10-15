"""
Slack CONNECTION binding for lollms_client.
"""
from typing import List, Optional

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "SlackConnectionBinding"


class SlackConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="slack", **kwargs)
        self.config = kwargs
        self.token = kwargs.get("service_key") or ""
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    def list_channels(self) -> List[str]:
        return ["slack:*"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="Slack bot is not running.", channel=channel_alias)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="slack")

    async def start(self) -> None:
        if not self.token:
            ASCIIColors.red("Slack binding: missing 'service_key' (bot token). Skipping.")
            return
        try:
            from slack_bolt import App
        except ImportError:
            ASCIIColors.red("slack-bolt not installed. Run: pip install slack-bolt")
            return

        self._running = True
        ASCIIColors.info("💬 Starting Slack connection binding...")
        try:
            app = App(token=self.token)

            @app.message
            async def handle_message(message, say):
                await say(self._reply(message.text or ""))

            app.start()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"Slack binding error: {e}")
        finally:
            self._running = False

    def _reply(self, text: str) -> str:
        client = self.lollms_client
        if client is not None and getattr(client, "llm", None) is not None:
            try:
                out = client.llm.generate_text(prompt=text)
                if isinstance(out, dict):
                    return out.get("message", text)
                return out or text
            except Exception:
                return text
        return text