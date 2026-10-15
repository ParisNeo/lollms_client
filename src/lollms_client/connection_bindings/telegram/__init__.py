"""
Telegram CONNECTION binding for lollms_client.

First-class connection modality: runs a Telegram bot, routes incoming
messages to the attached LollmsClient's LLM, and replies on the same chat.
Credentials come from the binding config (service_key = bot token).
"""
import asyncio
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "TelegramConnectionBinding"


class TelegramConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="telegram", **kwargs)
        self.config = kwargs
        self.token = kwargs.get("service_key") or ""
        self.allowed_user_ids = self._parse_ids(kwargs.get("allowed_user_ids"))
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    @staticmethod
    def _parse_ids(raw) -> List[int]:
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw if str(x).strip()]
        return [int(x) for x in str(raw).split(",") if x.strip()]

    def list_channels(self) -> List[str]:
        return [f"telegram:{uid}" for uid in self.allowed_user_ids] or ["telegram:*"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="Telegram bot is not running.", channel=channel_alias)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="telegram")

    async def start(self) -> None:
        if not self.token:
            ASCIIColors.red("Telegram binding: missing 'service_key' (bot token). Skipping.")
            return
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
        except ImportError:
            ASCIIColors.red("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return

        self._running = True
        ASCIIColors.info("📱 Starting Telegram connection binding...")
        try:
            app = Application.builder().token(self.token).build()

            async def start_cmd(update, context):
                await update.message.reply_text("Connected to Lollms.")

            async def handle_message(update, context):
                if self.allowed_user_ids and update.effective_user.id not in self.allowed_user_ids:
                    return
                reply = self._reply(update.message.text or "")
                await update.message.reply_text(reply)

            app.add_handler(CommandHandler("start", start_cmd))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            try:
                while self._running:
                    await asyncio.sleep(1)
            finally:
                await app.stop()
                await app.shutdown()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"Telegram binding error: {e}")
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