import asyncio
from typing import Dict, Any
from lollms_client.apps.lollms_bot.connections.base import Connection
from ascii_colors import ASCIIColors

class DiscordConnection(Connection):
    async def run(self):
        ASCIIColors.info("🎮 Starting Discord Bot...")
        try:
            import discord
            from discord.ext import commands
            
            token = self.config.get("token")
            if not token:
                ASCIIColors.red("Discord token missing.")
                return

            client = discord.Client(intents=discord.Intents.default())
            
            @client.event
            async def on_ready():
                ASCIIColors.info(f"Discord Bot logged in as {client.user}")
                
            client.run(token)
        except ImportError:
            ASCIIColors.red("discord.py not installed. Run: pip install discord.py")
        except Exception as e:
            ASCIIColors.red(f"Discord error: {e}")