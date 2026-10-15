import asyncio
import logging
from typing import List
from lollms_client.apps.lollms_bot.config import BotConfig
from lollms_client.apps.lollms_bot.connections.telegram import TelegramConnection
from lollms_client.apps.lollms_bot.connections.discord import DiscordConnection
from lollms_client.apps.lollms_bot.connections.slack import SlackConnection
from lollms_client.apps.lollms_bot.connections.whatsapp import WhatsAppConnection
from ascii_colors import ASCIIColors

class BotDaemon:
    def __init__(self, config: BotConfig):
        self.config = config
        self.connections = []
        self.llm_config = config.config.get("llm", {})
        
    async def start(self):
        ASCIIColors.info("🚀 Starting Lollms-Bot Daemon...")
        
        # Initialize LLM Client (Placeholder for actual client creation)
        # In a real scenario, we'd create a LollmsClient here using the config
        ASCIIColors.info(f"Using LLM Profile: {self.llm_config.get('profile')}")
        
        conn_config = self.config.config.get("connections", {})
        
        if conn_config.get("telegram", {}).get("enabled"):
            self.connections.append(TelegramConnection(conn_config["telegram"], self.llm_config))
        if conn_config.get("discord", {}).get("enabled"):
            self.connections.append(DiscordConnection(conn_config["discord"], self.llm_config))
        if conn_config.get("slack", {}).get("enabled"):
            self.connections.append(SlackConnection(conn_config["slack"], self.llm_config))
        if conn_config.get("whatsapp", {}).get("enabled"):
            self.connections.append(WhatsAppConnection(conn_config["whatsapp"], self.llm_config))
        
        if not self.connections:
            ASCIIColors.yellow("No connections enabled. Check your configuration.")
            return
        
        ASCIIColors.info(f"Starting {len(self.connections)} connection(s)...")
        
        tasks = [conn.run() for conn in self.connections]
        await asyncio.gather(*tasks)