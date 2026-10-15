import asyncio
from typing import Dict, Any
from lollms_client.apps.lollms_bot.connections.base import Connection
from ascii_colors import ASCIIColors

class SlackConnection(Connection):
    async def run(self):
        ASCIIColors.info("💬 Starting Slack Bot...")
        try:
            from slack_bolt import App
            
            token = self.config.get("token")
            if not token:
                ASCIIColors.red("Slack token missing.")
                return

            app = App(token=token)
            
            @app.message("hello")
            async def handle_hello(message, say):
                await say("Hello from Lollms-Bot!")
                
            app.start()
        except ImportError:
            ASCIIColors.red("slack-bolt not installed. Run: pip install slack-bolt")
        except Exception as e:
            ASCIIColors.red(f"Slack error: {e}")