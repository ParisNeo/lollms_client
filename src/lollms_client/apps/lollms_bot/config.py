import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from ascii_colors import ASCIIColors

class BotConfig:
    """
    Manages the lollms-bot configuration file.
    Defaults to global config if no specific path is provided.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".lollms_client" / "bot_config.yaml"
        
        self.config: Dict[str, Any] = self._load()
    
    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return self._default_config()
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or self._default_config()
        except Exception as e:
            ASCIIColors.red(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "profile": "default",
                "model": "lollms",
                "context_size": 4096
            },
            "connections": {
                "telegram": {
                    "enabled": False,
                    "token": ""
                },
                "discord": {
                    "enabled": False,
                    "token": ""
                },
                "slack": {
                    "enabled": False,
                    "token": ""
                },
                "whatsapp": {
                    "enabled": False,
                    "config_path": ""
                }
            }
        }
    
    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
        ASCIIColors.green(f"Configuration saved to: {self.config_path}")
    
    def run_wizard(self):
        ASCIIColors.info("🛠️  Lollms-Bot Configuration Wizard")
        
        # LLM Section
        ASCIIColors.info("\n--- LLM Configuration ---")
        self.config["llm"]["profile"] = input("LLM Profile (default): ").strip() or "default"
        self.config["llm"]["model"] = input("LLM Model (lollms): ").strip() or "lollms"
        ctx_input = input("Context Size (4096): ").strip()
        self.config["llm"]["context_size"] = int(ctx_input) if ctx_input.isdigit() else 4096
        
        # Connections Section
        ASCIIColors.info("\n--- Connections ---")
        
        # Telegram
        tg_enabled = input("Enable Telegram? [y/N]: ").strip().lower() == 'y'
        self.config["connections"]["telegram"]["enabled"] = tg_enabled
        if tg_enabled:
            self.config["connections"]["telegram"]["token"] = input("Telegram Bot Token: ").strip()
        
        # Discord
        dc_enabled = input("Enable Discord? [y/N]: ").strip().lower() == 'y'
        self.config["connections"]["discord"]["enabled"] = dc_enabled
        if dc_enabled:
            self.config["connections"]["discord"]["token"] = input("Discord Bot Token: ").strip()
            
        # Slack
        sl_enabled = input("Enable Slack? [y/N]: ").strip().lower() == 'y'
        self.config["connections"]["slack"]["enabled"] = sl_enabled
        if sl_enabled:
            self.config["connections"]["slack"]["token"] = input("Slack Bot Token: ").strip()
            
        # WhatsApp
        wa_enabled = input("Enable WhatsApp? [y/N]: ").strip().lower() == 'y'
        self.config["connections"]["whatsapp"]["enabled"] = wa_enabled
        if wa_enabled:
            self.config["connections"]["whatsapp"]["config_path"] = input("WhatsApp Config Path (optional): ").strip()
        
        self.save()
        ASCIIColors.green("✅ Configuration Complete.")