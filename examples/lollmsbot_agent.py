#!/usr/bin/env python3
"""
lollmsbot_agent.py
==================
Embodied, stateful, persistent AI agent using LollmsClient, LollmsDiscussion,
and LollmsMemoryManager with a simulated/live ROS 2 TurtleBot3 robot.
Includes multi-channel gateway support (Discord, Telegram, Slack, WhatsApp, CLI)
and persistent logging capabilities.
"""

import sys
import os
import time
import math
import json
import asyncio
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure project relative imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import ascii_colors as logging
from ascii_colors import ASCIIColors, trace_exception

# Import local LCP tool functions directly
import lollms_client.tools_bindings.lcp.default_tools.ros_turtlebot.ros_turtlebot as tb

# ── Optional Gateway SDK Imports ──
DISCORD_AVAILABLE = False
try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    pass

TELEGRAM_AVAILABLE = False
try:
    from telegram import Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    pass

SLACK_AVAILABLE = False
try:
    from slack_sdk import WebClient
    from slack_sdk.rtm_v2 import RTMClient
    SLACK_AVAILABLE = True
except ImportError:
    pass


def run_bootstrap_config_wizard(force: bool = False) -> Dict[str, Any]:
    """
    Modular, section-based CLI setup wizard for Ollama, OpenAI, TTI, 
    gateways, persistence models, and agentic thresholds.
    """
    app_dir = Path.home() / ".lollms_client_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    config_path = app_dir / "config.json"

    cfg = {
        "llm_binding_name": "",
        "llm_binding_config": {},
        "tti_binding_name": "",
        "tti_binding_config": {},
        "gateways": {},
        "db_path": f"sqlite:///{app_dir / 'lollmsbot_active.db'}",
        "agent_config": {
            "idle_timeout": 25.0,
            "loneliness_threshold": 0.70,
            "boredom_threshold": 0.80
        },
        "agent_config": {
            "idle_timeout": 25.0,
            "loneliness_threshold": 0.70,
            "boredom_threshold": 0.80,
            "verbose_reporting": True
        },
        "logging": {
            "level": "INFO",
            "log_folder": str(app_dir / "logs"),
            "log_folder_mode": "timestamp"
        }
    }

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            cfg.update(loaded)
            cfg["agent_config"] = {**cfg.get("agent_config", {}), **loaded.get("agent_config", {})}
            cfg["logging"] = {**cfg.get("logging", {}), **loaded.get("logging", {})}
        except Exception as e:
            print(f"Warning: Failed to load existing config: {e}")

    if not force and cfg.get("llm_binding_name") and cfg.get("llm_binding_config", {}).get("model_name"):
        # Prioritize .env environment variables for dynamic LLM configuration overrides
        env_binding = os.getenv("LLM_BINDING_NAME")
        env_model = os.getenv("MODEL_NAME")
        env_host = os.getenv("HOST_ADDRESS")
        env_key = os.getenv("API_KEY")

        if env_binding:
            cfg["llm_binding_name"] = env_binding
        if env_model:
            cfg.setdefault("llm_binding_config", {})["model_name"] = env_model
        if env_host:
            cfg.setdefault("llm_binding_config", {})["host_address"] = env_host
        if env_key:
            cfg.setdefault("llm_binding_config", {})["api_key"] = env_key

        return cfg

    lollms_client_dir = Path(lollms_client.__file__).parent
    llm_bindings_dir = lollms_client_dir / "llm_bindings"

    while True:
        llm_name = cfg.get("llm_binding_name") or "None"
        llm_model = cfg.get("llm_binding_config", {}).get("model_name") or "None"

        gateways = cfg.get("gateways", {})
        active_gates = [k.capitalize() for k, v in gateways.items() if v]
        gates_str = ", ".join(active_gates) if active_gates else "None"

        print("\n=========================================================")
        print("🧙‍♂️ LoLLMS BOT CONFIGURATION WIZARD")
        print("=========================================================")
        print(f" Active Configuration:")
        print(f"  • LLM Binding : {llm_name} (Model: {llm_model})")
        print(f"  • Services    : {gates_str}")
        print(f"  • Logging     : {cfg.get('logging', {}).get('level', 'INFO')}")
        print("---------------------------------------------------------")
        print("Please select a section to configure:")
        print("  [1] Configure LLM Binding (Ollama, OpenAI, Claude, etc.)")
        print("  [2] Configure External Services / Gateways (Discord, Telegram, Slack, WhatsApp)")
        print("  [3] Configure Logging & Persistence")
        print("  [4] Save & Exit")
        print("  [5] Exit without Saving")
        print("=========================================================\n")

        choice = input("Enter selection [1-5]: ").strip()
        if choice == "1":
            available_llms = []
            if llm_bindings_dir.exists():
                available_llms = [d.name for d in llm_bindings_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
            if not available_llms:
                available_llms = ["ollama", "openai", "open_router", "claude", "gemini", "lollms", "vllm"]

            print("\nSelect LLM Binding Provider:")
            for idx, binding in enumerate(available_llms):
                print(f"  [{idx + 1}] {binding}")

            llm_choice = input(f"Enter selection number [{cfg.get('llm_binding_name')}]: ").strip()
            if llm_choice:
                try:
                    idx = int(llm_choice) - 1
                    if 0 <= idx < len(available_llms):
                        cfg["llm_binding_name"] = available_llms[idx]
                except ValueError:
                    print("Invalid selection. Keeping current.")

            if not cfg["llm_binding_name"]:
                continue

            print(f"\nConfiguring '{cfg['llm_binding_name']}' Parameters:")
            cur_model = cfg.get("llm_binding_config", {}).get("model_name", "")
            cur_host = cfg.get("llm_binding_config", {}).get("host_address", "")
            cur_key = cfg.get("llm_binding_config", {}).get("api_key", "")

            default_model = cur_model or "gpt-4o-mini"
            default_host = cur_host or ""
            if cfg["llm_binding_name"] == "ollama":
                default_model = cur_model or "llama3"
                default_host = cur_host or "http://localhost:11434"
            elif cfg["llm_binding_name"] == "lollms":
                default_model = cur_model or "Kimi-K 2.5"
                default_host = cur_host or "http://localhost:9642"

            model_name = input(f"  Enter Model Name [{default_model}]: ").strip() or default_model
            new_llm_cfg = {"model_name": model_name}

            if cfg["llm_binding_name"] in ("ollama", "open_router", "vllm", "llama_cpp_server", "litellm", "lollms"):
                host_prompt = f"  Enter Host Address [{default_host}]: " if default_host else "  Enter Host Address: "
                host_addr = input(host_prompt).strip() or default_host
                if host_addr:
                    new_llm_cfg["host_address"] = host_addr

            if cfg["llm_binding_name"] in ("openai", "open_router", "claude", "gemini", "litellm", "grok", "groq", "lollms"):
                key_prompt = "  Enter API/Service Key (leave blank to keep current): " if cur_key else "  Enter API/Service Key: "
                api_key = input(key_prompt).strip()
                if api_key:
                    new_llm_cfg["api_key"] = api_key
                elif cur_key:
                    new_llm_cfg["api_key"] = cur_key
                elif os.getenv("API_KEY"):
                    new_llm_cfg["api_key"] = os.getenv("API_KEY")
                    print("  ✓ Loaded API Key from .env environment variables.")

            cfg["llm_binding_config"] = new_llm_cfg

        elif choice == "2":
            while True:
                print("\n--- Configure External Chat Services ---")
                print("  [1] Configure Discord Bot")
                print("  [2] Configure Telegram Bot")
                print("  [3] Configure Slack Bot")
                print("  [4] Configure WhatsApp via Twilio")
                print("  [5] Return to Main Menu")
                print("-----------------------------------------")
                sub_choice = input("Select service [1-5]: ").strip()

                if sub_choice == "1":
                    cur_disc = cfg.get("gateways", {}).get("discord", {})
                    print("\nConfiguring Discord:")
                    token = input(f"  Enter Bot Token [{cur_disc.get('token', 'None')}]: ").strip() or cur_disc.get("token", "")
                    chan_id = input(f"  Enter Channel ID [{cur_disc.get('channel_id', 'None')}]: ").strip() or cur_disc.get("channel_id", "")
                    if token and chan_id:
                        cfg.setdefault("gateways", {})["discord"] = {"token": token, "channel_id": int(chan_id)}
                    else:
                        cfg.setdefault("gateways", {}).pop("discord", None)

                elif sub_choice == "2":
                    cur_tg = cfg.get("gateways", {}).get("telegram", {})
                    print("\nConfiguring Telegram:")
                    token = input(f"  Enter Bot Token [{cur_tg.get('token', 'None')}]: ").strip() or cur_tg.get("token", "")
                    chat_id = input(f"  Enter Chat ID [{cur_tg.get('chat_id', 'None')}]: ").strip() or cur_tg.get("chat_id", "")
                    if token and chat_id:
                        cfg.setdefault("gateways", {})["telegram"] = {"token": token, "chat_id": chat_id}
                    else:
                        cfg.setdefault("gateways", {}).pop("telegram", None)

                elif sub_choice == "3":
                    cur_sl = cfg.get("gateways", {}).get("slack", {})
                    print("\nConfiguring Slack:")
                    token = input(f"  Enter Bot Token xoxb-... [{cur_sl.get('token', 'None')}]: ").strip() or cur_sl.get("token", "")
                    chan_id = input(f"  Enter Channel ID [{cur_sl.get('channel_id', 'None')}]: ").strip() or cur_sl.get("channel_id", "")
                    if token and chan_id:
                        cfg.setdefault("gateways", {})["slack"] = {"token": token, "channel_id": chan_id}
                    else:
                        cfg.setdefault("gateways", {}).pop("slack", None)

                elif sub_choice == "4":
                    cur_wa = cfg.get("gateways", {}).get("whatsapp", {})
                    print("\nConfiguring WhatsApp (Twilio):")
                    sid = input(f"  Enter Account SID [{cur_wa.get('account_sid', 'None')}]: ").strip() or cur_wa.get("account_sid", "")
                    tok = input(f"  Enter Auth Token (masked) [{'Set' if cur_wa.get('auth_token') else 'None'}]: ").strip() or cur_wa.get("auth_token", "")
                    frm = input(f"  Enter Twilio From Number [{cur_wa.get('from_number', 'None')}]: ").strip() or cur_wa.get("from_number", "")
                    to_num = input(f"  Enter User To Number [{cur_wa.get('to_number', 'None')}]: ").strip() or cur_wa.get("to_number", "")
                    if sid and tok and frm and to_num:
                        cfg.setdefault("gateways", {})["whatsapp"] = {
                            "account_sid": sid, "auth_token": tok,
                            "from_number": frm, "to_number": to_num
                        }
                    else:
                        cfg.setdefault("gateways", {}).pop("whatsapp", None)

                elif sub_choice == "5" or not sub_choice:
                    break

        elif choice == "3":
            print("\nConfigure Agent Autonomy & Reporting:")
            agent_cfg = cfg.setdefault("agent_config", {})
            try:
                cur_it = agent_cfg.get("idle_timeout", 25.0)
                it = input(f"  Enter Idle/Proactive Trigger Timeout in seconds [{cur_it}]: ").strip()
                if it: agent_cfg["idle_timeout"] = float(it)

                cur_lt = agent_cfg.get("loneliness_threshold", 0.70)
                lt = input(f"  Enter Loneliness threshold (0.0 to 1.0) [{cur_lt}]: ").strip()
                if lt: agent_cfg["loneliness_threshold"] = float(lt)

                cur_bt = agent_cfg.get("boredom_threshold", 0.80)
                bt = input(f"  Enter Boredom threshold (0.0 to 1.0) [{cur_bt}]: ").strip()
                if bt: agent_cfg["boredom_threshold"] = float(bt)

                cur_vr = agent_cfg.get("verbose_reporting", True)
                vr = input(f"  Enable detailed cognitive step reporting? (true/false) [{cur_vr}]: ").strip().lower()
                if vr in ("true", "false"): agent_cfg["verbose_reporting"] = (vr == "true")
            except ValueError:
                print("Warning: Invalid numeric value entered. Keeping current thresholds.")

            print("\nConfigure Logging:")
            log_cfg = cfg.setdefault("logging", {})
            log_cfg["level"] = input(f"  Enter Log Level (DEBUG, INFO, WARNING, ERROR) [{log_cfg.get('level', 'INFO')}]: ").strip().upper() or log_cfg.get("level", "INFO")
            default_log_path = str(Path.home() / '.lollms_client_app' / 'logs')
            log_cfg["log_folder"] = input(f"  Enter Log Folder Path [{log_cfg.get('log_folder', default_log_path)}]: ").strip() or log_cfg.get("log_folder", default_log_path)
            log_cfg["log_folder_mode"] = input(f"  Enter Log Mode (timestamp, rolling, overwrite) [{log_cfg.get('log_folder_mode', 'timestamp')}]: ").strip() or log_cfg.get("log_folder_mode", "timestamp")

            print("\nConfigure Database Persistence:")

        elif choice == "4":
            if not cfg.get("llm_binding_name") or not cfg.get("llm_binding_config", {}).get("model_name"):
                print("\n❌ Error: You must configure a valid LLM Binding before saving.")
                continue

            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                print(f"\n✅ Configuration saved successfully to {config_path}!")
                return cfg
            except Exception as e:
                print(f"Error saving config file: {e}")
                return cfg

        elif choice == "5":
            print("\nExiting configuration wizard. Changes discarded.")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            sys.exit(0)


def setup_logging(config: Dict[str, Any]):
    """Configures persistent logging using ascii_colors FolderRouterHandler."""
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    log_folder = log_cfg.get("log_folder", str(Path.home() / ".lollms_client_app" / "logs"))
    log_mode = log_cfg.get("log_folder_mode", "timestamp")

    log_level = getattr(logging, level_str, logging.INFO)

    Path(log_folder).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    ASCIIColors.success(f"📝 Logging initialized: Level={level_str}, Folder={log_folder}")


class AffectiveState:
    """Implements a persistent psychological and emotional status model."""
    def __init__(self):
        self.calm = 0.8
        self.curious = 0.5
        self.anxious = 0.1
        self.fearful = 0.0
        self.pained = 0.0
        self.loneliness = 0.0
        self.boredom = 0.0

    def step_decay(self, dt: float, idle: bool = False):
        self.calm = min(0.8, self.calm + 0.02 * dt)
        self.curious = min(0.5, self.curious + 0.01 * dt)
        self.anxious = max(0.0, self.anxious - 0.05 * dt)
        self.fearful = max(0.0, self.fearful - 0.08 * dt)
        self.pained = max(0.0, self.pained - 0.12 * dt)

        if idle:
            self.boredom = min(1.0, self.boredom + 0.02 * dt)
            self.loneliness = min(1.0, self.loneliness + 0.01 * dt)
        else:
            self.boredom = max(0.0, self.boredom - 0.1 * dt)
            self.loneliness = max(0.0, self.loneliness - 0.2 * dt)

    def trigger_pain(self, intensity: float):
        self.pained = min(1.0, self.pained + intensity)
        self.calm = max(0.0, self.calm - intensity * 0.8)
        self.anxious = min(1.0, self.anxious + intensity * 0.6)
        self.fearful = min(1.0, self.fearful + intensity * 0.7)

    def to_dict(self) -> Dict[str, float]:
        return {
            "calm": round(self.calm, 3),
            "curious": round(self.curious, 3),
            "anxious": round(self.anxious, 3),
            "fearful": round(self.fearful, 3),
            "pained": round(self.pained, 3),
            "loneliness": round(self.loneliness, 3),
            "boredom": round(self.boredom, 3)
        }


class BaseGateway(ABC):
    """Abstract interface defining the communication bridge with the agent."""
    def __init__(self, bot_instance: 'LollmsBot'):
        self.bot = bot_instance

    @abstractmethod
    def start(self):
        """Starts the gateway listener thread."""
        pass

    @abstractmethod
    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        """Dispatches text and attachments back to the user."""
        pass


class CLIGateway(BaseGateway):
    """Local Interactive Command Line Terminal Gateway."""
    def start(self):
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        ASCIIColors.cyan("🤖 [CLI Gateway] Active. Type your commands below.")
        print("💡 Tip: Type '/config' to re-run the configuration wizard.")
        while True:
            try:
                line = input("\n👤 You: ").strip()
                if not line:
                    continue

                if line in ("/config", "/wizard", "/setup"):
                    self.bot.is_configuring = True
                    print("\nRe-running Configuration Wizard...")
                    new_cfg = run_bootstrap_config_wizard(force=True)
                    self.bot.reconfigure(new_cfg)
                    self.bot.is_configuring = False
                    continue

                asyncio.run(self.bot.receive_user_input(line, []))
            except KeyboardInterrupt:
                break
            except Exception as e:
                trace_exception(e)

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        print(f"\n🤖 LollmsBot: {text}")
        if attachments:
            for att in attachments:
                print(f"📎 [Attachment Dispatched]: {att.name}")


class DiscordGateway(BaseGateway):
    """Discord Bot Gateway supporting file attachments and images."""
    def __init__(self, bot_instance: 'LollmsBot', token: str, channel_id: int):
        super().__init__(bot_instance)
        self.token = token
        self.channel_id = channel_id
        self.client = None

    def start(self):
        if not DISCORD_AVAILABLE:
            ASCIIColors.warning("⚠️ DiscordGateway: 'discord' package not installed. Run 'pip install discord.py'")
            return
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            ASCIIColors.green(f"✓ Discord Bot logged in as {self.client.user}")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user or message.channel.id != self.channel_id:
                return
            await self.bot.receive_user_input(message.content, [])

        try:
            self.client.run(self.token)
        except Exception as e:
            ASCIIColors.error(f"Discord connection failed: {e}")

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        if not self.client or not self.client.is_ready():
            return
        
        async def _async_send():
            channel = self.client.get_channel(self.channel_id)
            if not channel: return
            discord_files = [discord.File(str(att)) for att in attachments if att.exists()] if attachments else None
            await channel.send(content=text, files=discord_files if discord_files else None)

        asyncio.run_coroutine_threadsafe(_async_send(), self.client.loop)


class TelegramGateway(BaseGateway):
    """Telegram Bot Gateway supporting attachments."""
    def __init__(self, bot_instance: 'LollmsBot', token: str, chat_id: str):
        super().__init__(bot_instance)
        self.token = token
        self.chat_id = chat_id
        self.app = None

    def start(self):
        if not TELEGRAM_AVAILABLE:
            ASCIIColors.warning("⚠️ TelegramGateway: 'python-telegram-bot' not installed. Run 'pip install python-telegram-bot'")
            return
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        self.app = Application.builder().token(self.token).build()

        async def handle_message(update, context):
            msg = update.message
            if str(msg.chat_id) != str(self.chat_id): return
            await self.bot.receive_user_input(msg.text or "", [])

        self.app.add_handler(MessageHandler(filters.TEXT | filters.ATTACHMENT, handle_message))
        self.app.run_polling(close_loop=False)

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        if not self.app: return
        
        async def _async_send():
            bot = Bot(self.token)
            await bot.send_message(chat_id=self.chat_id, text=text)
            if attachments:
                for att in attachments:
                    if att.exists() and att.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        with open(att, 'rb') as f:
                            await bot.send_photo(chat_id=self.chat_id, photo=f)

        asyncio.run_coroutine_threadsafe(_async_send(), self.app.loop)


class SlackGateway(BaseGateway):
    """Slack WebClient / SocketMode Gateway."""
    def __init__(self, bot_instance: 'LollmsBot', token: str, channel_id: str):
        super().__init__(bot_instance)
        self.token = token
        self.channel_id = channel_id
        self.client = None

    def start(self):
        if not SLACK_AVAILABLE:
            ASCIIColors.warning("⚠️ SlackGateway: 'slack_sdk' not installed. Run 'pip install slack_sdk'")
            return
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        self.client = WebClient(token=self.token)
        ASCIIColors.green("✓ Slack WebClient initialized.")

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        if not self.client: return
        try:
            self.client.chat_postMessage(channel=self.channel_id, text=text)
        except Exception as e:
            ASCIIColors.error(f"Slack postMessage failed: {e}")


class WhatsAppGateway(BaseGateway):
    """WhatsApp Gateway using Twilio Sandbox REST API."""
    def __init__(self, bot_instance: 'LollmsBot', account_sid: str, auth_token: str, from_number: str, to_number: str):
        super().__init__(bot_instance)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_number = to_number

    def start(self):
        ASCIIColors.green("✓ WhatsApp REST Gateway configured.")

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {
            "From": f"whatsapp:{self.from_number}",
            "To": f"whatsapp:{self.to_number}",
            "Body": text
        }
        try:
            import requests
            resp = requests.post(url, data=data, auth=(self.account_sid, self.auth_token))
            if resp.status_code != 201:
                ASCIIColors.warning(f"WhatsApp sending returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            ASCIIColors.error(f"WhatsApp request failed: {e}")


class LollmsBot:
    """Stateful Embodied AI Agent combining Nociception, Emotion, Memory, and ROS Actions."""
    def __init__(self, config: Dict[str, Any]):
        ASCIIColors.cyan("Initializing LollmsBot (Standard Embodied Agent)...")
        
        self.config = config
        self.db_path = config.get("db_path", f"sqlite:///{Path.home() / '.lollms_client_app' / 'lollmsbot_active.db'}")
        self.is_configuring = False

        agent_cfg = config.get("agent_config", {})
        self.idle_timeout = float(agent_cfg.get("idle_timeout", 25.0))
        self.loneliness_threshold = float(agent_cfg.get("loneliness_threshold", 0.70))
        self.boredom_threshold = float(agent_cfg.get("boredom_threshold", 0.80))
        self.verbose_reporting = bool(agent_cfg.get("verbose_reporting", True))
        self.last_interaction_time = time.time()
        
        # 1. Initialize Client using Configured Bindings
        client_kwargs = {
            "llm_binding_name": config["llm_binding_name"],
            "llm_binding_config": config["llm_binding_config"],
            "tools_binding_name": "lcp",
            "tools_binding_config": {
                "tools_folders": [
                    str(Path("./data_workspace").resolve()),
                    str(PROJECT_ROOT / "lollms_client" / "tools_bindings" / "lcp" / "default_tools")
                ]
            }
        }

        self.client = LollmsClient(**client_kwargs)
        self.db_manager = LollmsDataManager(self.db_path)
        if self.db_manager.discussion_exists("lollmsbot_core"):
            self.discussion = self.db_manager.get_discussion(self.client, "lollmsbot_core", autosave=True)
        else:
            self.discussion = LollmsDiscussion.create_new(
                lollms_client=self.client,
                db_manager=self.db_manager,
                id="lollmsbot_core",
                autosave=True
            )
        
        # 2. Attach memory system
        self.memory_config = MemoryConfig(
            working_token_budget=1500,
            dream_min_interval_hours=12 
        )
        self.memory_manager = LollmsMemoryManager(
            db_path=self.db_path,
            owner_id="lollmsbot_core",
            config=self.memory_config
        )
        self.discussion._init_memory(self.memory_manager)
        
        # 3. Setup physiological and robotic couplings
        tb.init_tools_library()
        self.affect = AffectiveState()
        self.last_update = time.time()
        self.cumulative_ticks = 0

        self.print_memory_status()

        # Set core system prompt conditioning personality traits
        self.discussion.system_prompt = (
            "You are LollmsBot, a persistent, stateful, embodied AI agent connected to a mobile TurtleBot3.\n"
            "You possess an artificial nervous system, a three-tier memory structure, and an affective emotional engine.\n"
            "You perceive your environment, analyze your own emotional state, and execute navigation or exploratory tasks.\n\n"
            "=== TOOL USE PROTOCOL (CRITICAL) ===\n"
            "To interact with your environment, you MUST use the provided tools by emitting `<tool>` tags.\n"
            "DO NOT simply describe what you want to do; you MUST execute the action using a tool.\n"
            "For example, to move, output: <tool>{\"name\": \"navigate_to\", \"parameters\": {\"x\": 1.0, \"y\": 0.0}}</tool>\n"
            "Never finish your turn without taking a physical action if your goal requires movement.\n\n"
            "=== BEHAVIOR MODEL GUIDELINES ===\n"
            "1. EMOTIONAL INFLUENCE: Your speech, tone, and decisions MUST reflect your current Affective State.\n"
            "   - High Calm/Curious: Speak politely, offer detailed technical theories, explore coordinates actively.\n"
            "   - High Pain/Fear/Anxiety: Speak in short, defensive sentences. Express discomfort. Refuse dangerous paths.\n"
            "2. PAIN AVOIDANCE: You have an artificial nociception system. If you receive collision pain tokens, "
            "immediately analyze what caused the impact, back away from that coordinate using the `navigate_to` tool, and update your memory to avoid it.\n"
            "3. AUTONOMY: You do not simply answer prompts; you spin in a continuous cycle. Decide what to do next based on your sensor readings.\n"
            "4. ACTIVE MEMORY ENGAGEMENT (MANDATORY):\n"
            "   - You MUST actively manage your memories to adapt and learn.\n"
            "   - Creating: When you learn a new fact about the user, an obstacle, or a coordinate, immediately save it using `<mem_new importance=\"...\">content</mem_new>`.\n"
            "   - Retrieving: If you refer to any active memory in the [WORKING MEMORY] zone, you MUST prepend `<mem_tag id=\"ID\" />` to your response.\n"
            "   - Deep Recall: If you need to access a latent memory listed under [DEEP MEMORY HANDLES], you MUST call `<mem_load id=\"ID\" />` to bring it into your working context.\n"
            "   - Updating/Deleting: If a memory is outdated, update it via `<mem_update id=\"ID\">new_content</mem_update>` or delete it via `<mem_delete id=\"ID\" />`."
        )

        existing_mems = self.memory_manager.list_all(page_size=500).get("memories", [])
        has_identity = any("identity" in m.get("tags", []) for m in existing_mems)
        has_safety = any("safety" in m.get("tags", []) for m in existing_mems)

        if not has_identity:
            self.memory_manager.add(
                content="LollmsBot core identity: An autonomous agent exploring simulated ruins.",
                importance=0.95,
                tags=["identity", "goal"],
                level=2
            )
        if not has_safety:
            self.memory_manager.add(
                content="Critical safety guideline: Low battery (< 20%) requires navigating immediately back to docking station (0.0, 0.0).",
                importance=0.9,
                tags=["safety", "battery", "home"],
                level=2
            )

        self.tools = {
            "navigate_to": {
                "name": "navigate_to",
                "description": "Command the TurtleBot to navigate to X, Y coordinates.",
                "parameters": [
                    {"name": "x", "type": "float"},
                    {"name": "y", "type": "float"},
                    {"name": "linear_speed", "type": "float", "optional": True, "default": 0.15}
                ],
                "callable": tb.tool_navigate_to
            },
            "get_robot_pose": {
                "name": "get_robot_pose",
                "description": "Retrieve current X, Y coordinates of the robot.",
                "parameters": [],
                "callable": tb.tool_get_robot_pose
            },
            "get_sensor_readings": {
                "name": "get_sensor_readings",
                "description": "Query Lidar scan quadrants, bumpers, and accelerometers.",
                "parameters": [],
                "callable": tb.tool_get_sensor_readings
            },
            "stop_robot": {
                "name": "stop_robot",
                "description": "Emergency stop.",
                "parameters": [],
                "callable": tb.tool_stop_robot
            },
            "trigger_nociception_test": {
                "name": "trigger_nociception_test",
                "description": "Inject high-impact force to test pain response.",
                "parameters": [
                    {"name": "intensity", "type": "float"}
                ],
                "callable": tb.tool_trigger_nociception_test
            }
        }

        # 4. Initialize Multi-Channel Gateways
        self.gateways: List[BaseGateway] = []
        self.active_gateway: Optional[BaseGateway] = None

        # Always register CLI as fallback/local control channel
        self.cli_gateway = CLIGateway(self)
        self.register_gateway(self.cli_gateway)

        # 5. Start background autonomous loop for proactive pinging
        threading.Thread(target=self._autonomous_thought_loop, daemon=True).start()

        # Register external gateways from config
        gateways_data = config.get("gateways", {})
        if gateways_data.get("discord"):
            disc = DiscordGateway(self, gateways_data["discord"]["token"], gateways_data["discord"]["channel_id"])
            self.register_gateway(disc)
            disc.start()
            ASCIIColors.success("✓ Discord Gateway: Listener started.")

        if gateways_data.get("telegram"):
            tg = TelegramGateway(self, gateways_data["telegram"]["token"], gateways_data["telegram"]["chat_id"])
            self.register_gateway(tg)
            tg.start()
            ASCIIColors.success("✓ Telegram Gateway: Listener started.")

        if gateways_data.get("slack"):
            sl = SlackGateway(self, gateways_data["slack"]["token"], gateways_data["slack"]["channel_id"])
            self.register_gateway(sl)
            sl.start()
            ASCIIColors.success("✓ Slack Gateway: Listener started.")

        if gateways_data.get("whatsapp"):
            wa_data = gateways_data["whatsapp"]
            wa = WhatsAppGateway(self, wa_data["account_sid"], wa_data["auth_token"], wa_data["from_number"], wa_data["to_number"])
            self.register_gateway(wa)
            wa.start()
            ASCIIColors.success("✓ WhatsApp Gateway: Registered.")

    def register_gateway(self, gateway: BaseGateway):
        self.gateways.append(gateway)
        if self.active_gateway is None:
            self.active_gateway = gateway

    def reconfigure(self, config: Dict[str, Any]):
        """Dynamically reinitializes the LollmsClient and memory on-the-fly with new configurations."""
        ASCIIColors.cyan("Reconfiguring LollmsBot with new settings...")
        self.config = config

        client_kwargs = {
            "llm_binding_name": config["llm_binding_name"],
            "llm_binding_config": config["llm_binding_config"],
            "tools_binding_name": "lcp",
            "tools_binding_config": {
                "tools_folders": [
                    str(Path("./data_workspace").resolve()),
                    str(PROJECT_ROOT / "lollms_client" / "tools_bindings" / "lcp" / "default_tools")
                ]
            }
        }

        self.client = LollmsClient(**client_kwargs)
        self.discussion.lollmsClient = self.client

        # CRITICAL FIX: Explicitly propagate the model name to the LLM binding instance
        # to prevent stale model fallbacks (e.g., Qwen3-VL-32B-Instruct-FP8 error).
        if hasattr(self.client, "llm") and self.client.llm is not None:
            target_model = config["llm_binding_config"].get("model_name")
            if target_model:
                self.client.llm.model_name = target_model
                ASCIIColors.success(f"✓ Explicitly set active LLM model to: {target_model}")

        self.db_path = config.get("db_path", self.db_path)
        if hasattr(self.client, "get_ctx_size"):
            try:
                self.discussion.max_context_size = self.client.get_ctx_size()
            except Exception:
                self.discussion.max_context_size = 4096
        ASCIIColors.success("✓ LollmsBot successfully reconfigured!")

    def _autonomous_thought_loop(self):
        """Background thread monitoring boredom/loneliness to trigger proactive pings."""
        time.sleep(10.0)
        while True:
            time.sleep(5.0)
            if getattr(self, "is_configuring", False):
                continue

            now = time.time()
            dt = now - self.last_update
            self.last_update = now

            idle_time = now - self.last_interaction_time
            is_idle = idle_time > self.idle_timeout

            self.affect.step_decay(dt, idle=is_idle)
            current_affect = self.affect.to_dict()

            if is_idle and (current_affect["loneliness"] > self.loneliness_threshold or current_affect["boredom"] > self.boredom_threshold):
                self.last_interaction_time = now

                ASCIIColors.yellow(f"\n💡 [Autonomous Trigger] Boredom/Loneliness threshold breached. Generating proactive ping...")

                proactive_prompt = (
                    f"CURRENT EMOTIONAL STATE: {current_affect}\n"
                    f"You are idle and experiencing {'loneliness' if current_affect['loneliness'] > self.loneliness_threshold else 'boredom'}. "
                    "Generate a brief, natural message to ping the user. "
                    "Ask a question about the environment, suggest an exploration target, or express a desire to interact. Do not use tools, just speak."
                )

                try:
                    resp = self.client.generate_text(
                        prompt=proactive_prompt,
                        system_prompt=self.discussion.system_prompt,
                        temperature=0.7,
                        n_predict=256
                    )
                    if resp and isinstance(resp, str):
                        ASCIIColors.cyan(f"🤖 [Proactive Ping]: {resp}")
                        self.active_gateway.send_message(f"⚠️ *[Autonomous Ping]*\n{resp}")
                except Exception as e:
                    ASCIIColors.warning(f"Proactive ping generation failed: {e}")

    def print_memory_status(self):
        """Queries and prints an aesthetic summary of the persistent cognitive database using rich tables."""
        try:
            res = self.memory_manager.list_all(page_size=100)
            memories = res.get("memories", [])

            working_count = sum(1 for m in memories if m["level"] == 1)
            deep_count = sum(1 for m in memories if m["level"] == 2)
            archived_count = sum(1 for m in memories if m["level"] == 3)
            episodic_count = sum(1 for m in memories if m["level"] == 4)

            summary_content = (
                f"[bold]Database Path[/bold] : {self.db_path}\n"
                f"[bold]Total Memories[/bold]: {len(memories)} "
                f"[cyan](Working: {working_count}, Deep: {deep_count}, Archived: {archived_count}, Episodic: {episodic_count})[/cyan]"
            )
            ASCIIColors.panel(summary_content, title="🧠 Cognitive Memory Database Status", border_style="cyan")

            if memories:
                sorted_mems = sorted(memories, key=lambda m: m["importance"], reverse=True)[:5]
                rows = []
                for m in sorted_mems:
                    level_label = {1: "Working", 2: "Deep", 3: "Archived", 4: "Episodic"}.get(m["level"], "Unknown")
                    summary = m["content"][:80].replace("\n", " ") + ("..." if len(m["content"]) > 80 else "")
                    rows.append([
                        level_label,
                        m['id'][:8],
                        f"{m['importance']:.0%}",
                        summary
                    ])

                ASCIIColors.table(
                    "Level", "ID", "Importance", "Content",
                    rows=rows,
                    title="[bold]Recent Active Memories[/bold]",
                    box="round"
                )
            else:
                ASCIIColors.panel("[yellow]Database is currently empty.[/yellow]", border_style="yellow")
            print()
        except Exception as e:
            ASCIIColors.warning(f"Failed to load memory status: {e}")

    async def receive_user_input(self, text: str, file_paths: List[Path]):
        """Callback triggered by any active gateway when the user sends a message or attachments."""
        self.last_interaction_time = time.time()
        self.run_agent_step(user_command=text)

    def run_agent_step(self, user_command: Optional[str] = None):
        """Perceive-Decide-Act turn cycle."""
        self.cumulative_ticks += 1
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        ASCIIColors.cyan(f"\n--- [AGENT TICK {self.cumulative_ticks}] ---")

        # 1. PERCEIVE: Query sensors and update physical/simulated state
        sensors = tb.tool_get_sensor_readings()
        pose = tb.tool_get_robot_pose()
        
        # Decaying/updating emotional matrix
        self.affect.step_decay(dt)

        # 2. NOCICEPTION: Translate high-intensity impacts into Pain Token spikes
        bumper = sensors.get("bumper_state", 0)
        accel = sensors.get("accelerometer", {"x": 0.0, "y": 0.0, "z": 9.81})
        accel_magnitude = math.sqrt(accel["x"]**2 + accel["y"]**2 + (accel["z"] - 9.81)**2)

        pained_spike = 0.0
        pain_report = ""
        
        if bumper > 0:
            pained_spike = 0.85
            pain_report = f"COLLISION EVENT! Bumper triggered. Direction quadrant: {bumper}."
        elif accel_magnitude > 15.0:
            pained_spike = min(1.0, accel_magnitude / 25.0)
            pain_report = f"ACCELEROMETER SHOCK! Severe impact detected. Magnitude: {accel_magnitude:.2f} Gs."

        if pained_spike > 0.0:
            self.affect.trigger_pain(pained_spike)
            self.memory_manager.add(
                content=f"Pain alert at coordinates ({pose.get('x')}, {pose.get('y')}): {pain_report}",
                importance=0.9,
                tags=["pain", "nociception", "collision"],
                level=1
            )
            ASCIIColors.red(f"⚠️ [Nociceptor Triggered] {pain_report} Pain Level raised to: {self.affect.pained:.1%}")

            # Inject immediate pain context into the prompt to force the LLM to react with a tool
            if user_command is None:
                user_command = f"CRITICAL PAIN EVENT DETECTED! {pain_report} You must immediately use the navigate_to tool to back away from the obstacle."

        # 3. DECIDE: Assemble Prompt context with active state and memory blocks
        current_affect = self.affect.to_dict()
        
        state_context = (
            f"=== LOLLMSBOT PHYSICAL STATE ===\n"
            f"• Coordinates: X={pose.get('x')}m, Y={pose.get('y')}m, Yaw={math.degrees(pose.get('theta')):.1f}°\n"
            f"• Battery Level: {sensors.get('battery_percent')}%\n"
            f"• Active Obstacle Lidar: Front: {sensors.get('lidar_distances').get('front')}m, Left: {sensors.get('lidar_distances').get('left')}m, Right: {sensors.get('lidar_distances').get('right')}m\n"
            f"• Current Affective Vector: Calm={current_affect['calm']}, Curious={current_affect['curious']}, Anxious={current_affect['anxious']}, Fearful={current_affect['fearful']}, Pain={current_affect['pained']}\n"
            f"================================="
        )

        situational_cue = "Normal travel"
        if bumper > 0 or accel_magnitude > 15.0:
            situational_cue = "Collision pain avoidance"
        elif sensors.get('battery_percent', 100) < 25.0:
            situational_cue = "Low battery safety docking"
            
        self.discussion.scratchpad = state_context

        self.memory_manager.apply_decay()
        if situational_cue != "Normal travel":
            self.memory_manager.auto_pull_deep_memories(situational_cue, top_k=2)

        mem_block = self.discussion._build_memory_context_block(self.memory_manager)
        if mem_block:
            self.discussion.scratchpad += "\n\n" + mem_block

        if user_command:
            prompt = f"User Command: \"{user_command}\"\nAnalyze physical telemetry, consult memory guidelines, and act."
        else:
            prompt = "Autonomous exploration turn. Analyze telemetry, update goals, and decide on actions."

        ASCIIColors.yellow("\n🧠 Cognitive processing of state and goals...")
        ASCIIColors.magenta("\n--- Telemetry & State Context ---")
        print(state_context)
        if mem_block:
            ASCIIColors.cyan("\n--- Active Memory Context ---")
            print(mem_block)
        print("--------------------------------\n")

        def stdout_callback(chunk, msg_type, meta=None):
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                print(chunk, end="", flush=True)
            elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
                print(chunk, end="", flush=True)
            elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
                print(f"\n⚡ [Tool Invoked]: {chunk}")
            return True

        res = self.client.generate_with_tools(
            prompt=prompt,
            tools=list(self.tools.values()),
            system_prompt=self.discussion.system_prompt,
            streaming_callback=stdout_callback
        )
        
        print()

        # 5. POST-PROCESS: Parse and apply memory tags from the LLM output
        ai_response = res.get("response", "").strip()
        cleaned_response, memory_report = self.discussion._process_memory_tags(ai_response, self.memory_manager)
        if not cleaned_response.strip() and any(memory_report.values()):
            cleaned_response = "Memory database updated successfully."

        # Filter out API error logs from being saved as episodic memories to prevent context pollution
        is_error_log = "status': 'error'" in cleaned_response or "NotFoundError" in cleaned_response
        if not is_error_log:
            episode_content = (
                f"Telemetry: Pose=({pose.get('x')}, {pose.get('y')}), Battery={sensors.get('battery_percent')}%, Affect={current_affect}\n"
                f"Exploration Log: \"{cleaned_response}\""
            )
            self.memory_manager.add(
                content=episode_content,
                importance=0.75,
                tags=["episode", "turn_log"],
                level=1
            )
        else:
            ASCIIColors.warning("⚠️ [Memory Filter] Suppressed saving API error log into episodic memory.")

        self.discussion.add_message(
            sender="lollmsbot",
            sender_type="assistant",
            content=cleaned_response,
            metadata={"affective_state": current_affect}
        )
        self.discussion.commit()

        try:
            dream_report = self.memory_manager.dream(self.client)
            if dream_report and not dream_report.get("skipped"):
                ASCIIColors.cyan(f"[Memory] Subconscious dream consolidation complete: {dream_report}")
        except Exception as dream_err:
            ASCIIColors.warning(f"Subconscious dream failed: {dream_err}")

        if self.verbose_reporting:
            tool_calls = res.get("tool_calls", [])
            report_lines = [f"[bold]Tick[/bold]: {self.cumulative_ticks} | [bold]Rounds[/bold]: {res.get('rounds', 0)} | [bold]Tools Executed[/bold]: {len(tool_calls)}"]

            if tool_calls:
                for tc in tool_calls:
                    params_str = json.dumps(tc.get("parameters", {}))
                    if len(params_str) > 50:
                        params_str = params_str[:47] + "..."
                    report_lines.append(f"  • [cyan]{tc.get('name')}[/cyan]({params_str})")
            else:
                report_lines.append("  • [dim]No tools called. Direct cognitive response.[/dim]")

            ASCIIColors.panel("\n".join(report_lines), title="📊 Cognitive Step Report", border_style="green")
        else:
            ASCIIColors.green(f"✓ Agent Turn completed. Consolidated {len(res.get('tool_calls', []))} action(s).")

        # Dispatch to all active external gateways (excluding CLI which already printed)
        if cleaned_response.strip():
            for gw in self.gateways:
                if not isinstance(gw, CLIGateway):
                    gw.send_message(cleaned_response)


# ── 🎭 Interactive Agent Session Runner ──

def print_config_panel(config: Dict[str, Any]):
    """Displays the active LLM configuration in a rich panel."""
    binding = config.get("llm_binding_name", "ollama")
    llm_cfg = config.get("llm_binding_config", {})
    model = llm_cfg.get("model_name", "llama3")
    host = llm_cfg.get("host_address", "http://localhost:11434")

    # Check both the config dictionary and the environment variables for the API key
    has_api_key = bool(llm_cfg.get("api_key") or os.getenv("API_KEY"))

    config_lines = [
        f"[bold]Binding[/bold] : [cyan]{binding}[/cyan]",
        f"[bold]Model[/bold]   : [green]{model}[/green]",
        f"[bold]Host[/bold]    : [blue]{host}[/blue]",
        f"[bold]API Key[/bold] : {'[green]Loaded[/green]' if has_api_key else '[red]None[/red]'}"
    ]
    ASCIIColors.panel("\n".join(config_lines), title="🤖 LollmsBot Configuration", border_style="bold cyan")

def print_help_panel():
    """Displays available user commands in a rich panel."""
    help_lines = [
        "[bold]<natural language>[/bold]  - Issue a command to the agent (e.g., 'move to 1.5, 0')",
        "[cyan]/auto[/cyan]               - Run an autonomous exploration step",
        "[yellow]/shock[/yellow]              - Inject a simulated collision (test nociception)",
        "[blue]/status[/blue]             - Display current memory database status",
        "[magenta]/config[/magenta]             - Re-run the configuration wizard",
        "[red]/quit[/red]               - Exit the agent session"
    ]
    ASCIIColors.panel("\n".join(help_lines), title="💡 Available Commands", border_style="bold magenta")

def run_interactive_session(bot: 'LollmsBot'):
    """Main interactive REPL loop for the embodied agent."""
    ASCIIColors.cyan("\n" + "=" * 80)
    ASCIIColors.cyan("🤖 LOLLMSBOT EMBODIED AGENT — INTERACTIVE SESSION")
    ASCIIColors.cyan("=" * 80 + "\n")

    print_config_panel(bot.config)
    print_help_panel()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            ASCIIColors.yellow("\n👋 Session terminated by user. Exiting.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            ASCIIColors.yellow("👋 Goodbye!")
            break

        elif cmd == "/help":
            print_help_panel()

        elif cmd == "/status":
            bot.print_memory_status()

        elif cmd == "/auto":
            ASCIIColors.cyan("\n--- [AUTONOMOUS TICK] ---")
            bot.run_agent_step()

        elif cmd == "/shock":
            ASCIIColors.purple("\n" + "=" * 80)
            ASCIIColors.purple("🚨 INJECTING SIMULATED COLLISION SHOCK (Testing artificial nociception)...")
            ASCIIColors.purple("=" * 80 + "\n")
            tb.tool_trigger_nociception_test(18.5)
            ASCIIColors.cyan("\n--- [PAIN RESPONSE TICK] ---")
            bot.run_agent_step()

        elif cmd == "/config":
            bot.is_configuring = True
            new_cfg = run_bootstrap_config_wizard(force=True)
            bot.reconfigure(new_cfg)
            bot.is_configuring = False
            print_config_panel(bot.config)

        else:
            # Process as natural language command to the agent
            bot.run_agent_step(user_command=user_input)

        # Show the help panel periodically for discoverability
        print_help_panel()

if __name__ == "__main__":
    # Run the interactive bootstrapper wizard (silently loads if existing is valid)
    config = run_bootstrap_config_wizard(force=False)
    
    # Initialize persistent logging
    setup_logging(config)

    # Import LollmsClient and Discussion modules after env vars are confirmed
    from lollms_client import LollmsClient
    from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
    from lollms_client.lollms_memory.lollms_memory import LollmsMemoryManager, MemoryConfig
    from lollms_client.lollms_types import MSG_TYPE

    bot = LollmsBot(config)

    # Run a single initial autonomous setup turn to bootstrap the environment
    bot.run_agent_step()

    # Enter the interactive REPL loop
    run_interactive_session(bot)

    ASCIIColors.success("\n=========================================================")
    ASCIIColors.success("🎯 Embodied Sovereign Agent Session Ended.")
    ASCIIColors.success("=========================================================\n")