#!/usr/bin/env python3
"""
lollmsbot_advanced_agent.py
===========================
An autonomous, embodied, persistent AI agent featuring existential dread, 
self-bootstrapping, dynamic skill creation, multi-channel communication 
gateways (Discord, Slack, Telegram, WhatsApp, CLI), and bidirectional media ingestion.
Includes an interactive configuration wizard on first run.
"""

import sys
import os
import time
import math
import random
import json
import uuid
import asyncio
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Ensure project imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lollms_client
from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from ascii_colors import ASCIIColors, trace_exception

# Import local ROS TurtleBot tool library directly
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
    import json
    import sys
    from pathlib import Path
    import lollms_client

    app_dir = Path.home() / ".lollms_client_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    config_path = app_dir / "config.json"

    # Load existing config or initialize default empty config schema
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
        }
    }

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            cfg["llm_binding_name"] = loaded.get("llm_binding_name", "")
            cfg["llm_binding_config"] = loaded.get("llm_binding_config", {})
            cfg["tti_binding_name"] = loaded.get("tti_binding_name", "")
            cfg["tti_binding_config"] = loaded.get("tti_binding_config", {})
            cfg["gateways"] = loaded.get("gateways", {})
            cfg["db_path"] = loaded.get("db_path", f"sqlite:///{app_dir / 'lollmsbot_active.db'}")

            agent_cfg = loaded.get("agent_config", {})
            cfg["agent_config"]["idle_timeout"] = float(agent_cfg.get("idle_timeout", 25.0))
            cfg["agent_config"]["loneliness_threshold"] = float(agent_cfg.get("loneliness_threshold", 0.70))
            cfg["agent_config"]["boredom_threshold"] = float(agent_cfg.get("boredom_threshold", 0.80))
        except Exception as e:
            print(f"Warning: Failed to load existing config: {e}")

    # If not forced and we have a valid LLM config, return silently
    if not force and cfg.get("llm_binding_name") and cfg.get("llm_binding_config", {}).get("model_name"):
        return cfg

    lollms_client_dir = Path(lollms_client.__file__).parent
    llm_bindings_dir = lollms_client_dir / "llm_bindings"
    tti_bindings_dir = lollms_client_dir / "tti_bindings"

    while True:
        # Gather active details for the visual menu header
        llm_name = cfg.get("llm_binding_name") or "None"
        llm_model = cfg.get("llm_binding_config", {}).get("model_name") or "None"
        tti_name = cfg.get("tti_binding_name") or "None"
        tti_model = cfg.get("tti_binding_config", {}).get("model_name") or "None"

        gateways = cfg.get("gateways", {})
        active_gates = [k.capitalize() for k, v in gateways.items() if v]
        gates_str = ", ".join(active_gates) if active_gates else "None"

        db_disp = "In-Memory" if cfg.get("db_path") == "sqlite:///:memory:" else "File-Backed"
        agent_cfg = cfg.get("agent_config", {})

        print("\n=========================================================")
        print("🧙‍♂️ LoLLMS BOT CONFIGURATION WIZARD")
        print("=========================================================")
        print(f" Active Configuration:")
        print(f"  • LLM Binding : {llm_name} (Model: {llm_model})")
        print(f"  • TTI Binding : {tti_name} (Model: {tti_model})")
        print(f"  • Services    : {gates_str}")
        print(f"  • Persistence : {db_disp}")
        print(f"  • Thresholds  : Idle={agent_cfg['idle_timeout']}s, Loneliness={agent_cfg['loneliness_threshold']}, Boredom={agent_cfg['boredom_threshold']}")
        print("---------------------------------------------------------")
        print("Please select a section to configure:")
        print("  [1] Configure LLM Binding (Ollama, OpenAI, Claude, etc.)")
        print("  [2] Configure TTI Binding (Diffusers, DALL-E, etc.)")
        print("  [3] Configure External Services / Gateways (Discord, Telegram, Slack, WhatsApp)")
        print("  [4] Configure Database & Agent Thresholds (Boredom, Loneliness, Persistence)")
        print("  [5] Save & Exit")
        print("  [6] Exit without Saving")
        print("=========================================================\n")

        choice = input("Enter selection [1-6]: ").strip()
        if choice == "1":
            # --- SECTION 1: LLM BINDING ---
            available_llms = []
            if llm_bindings_dir.exists():
                available_llms = [d.name for d in llm_bindings_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
            if not available_llms:
                available_llms = ["ollama", "openai", "open_router", "claude", "gemini", "litellm", "vllm"]

            print("\nSelect LLM Binding Provider:")
            for idx, binding in enumerate(available_llms):
                print(f"  [{idx + 1}] {binding}")

            llm_choice = input(f"Enter selection number [Current: {cfg.get('llm_binding_name')}]: ").strip()
            if llm_choice:
                try:
                    idx = int(llm_choice) - 1
                    if 0 <= idx < len(available_llms):
                        cfg["llm_binding_name"] = available_llms[idx]
                except ValueError:
                    print("Invalid selection. Keeping current.")

            if not cfg["llm_binding_name"]:
                print("No LLM binding selected. Skipping parameters.")
                continue

            print(f"\nConfiguring '{cfg['llm_binding_name']}' Parameters:")
            cur_model = cfg.get("llm_binding_config", {}).get("model_name", "")
            cur_host = cfg.get("llm_binding_config", {}).get("host_address", "")
            cur_key = cfg.get("llm_binding_config", {}).get("api_key", "")

            # Fallbacks
            default_model = cur_model or "gpt-4o-mini"
            default_host = cur_host or ""
            if cfg["llm_binding_name"] == "ollama":
                default_model = cur_model or "llama3"
                default_host = cur_host or "http://localhost:11434"
            elif cfg["llm_binding_name"] == "open_router":
                default_model = cur_model or "meta-llama/llama-3-8b-instruct:free"
                default_host = cur_host or "https://openrouter.ai/api/v1"
            elif cfg["llm_binding_name"] == "claude":
                default_model = cur_model or "claude-3-5-sonnet-20240620"
            elif cfg["llm_binding_name"] == "gemini":
                default_model = cur_model or "gemini-1.5-flash"

            model_name = input(f"  Enter Model Name [{default_model}]: ").strip() or default_model
            new_llm_cfg = {"model_name": model_name}

            if cfg["llm_binding_name"] in ("ollama", "open_router", "vllm", "llama_cpp_server", "litellm"):
                host_prompt = f"  Enter Host Address [{default_host}]: " if default_host else "  Enter Host Address: "
                host_addr = input(host_prompt).strip() or default_host
                if host_addr:
                    new_llm_cfg["host_address"] = host_addr

            if cfg["llm_binding_name"] in ("openai", "open_router", "claude", "gemini", "litellm", "grok", "groq"):
                key_prompt = "  Enter API/Service Key (leave blank to keep current): " if cur_key else "  Enter API/Service Key: "
                api_key = input(key_prompt).strip()
                if api_key:
                    new_llm_cfg["api_key"] = api_key
                elif cur_key:
                    new_llm_cfg["api_key"] = cur_key

            cfg["llm_binding_config"] = new_llm_cfg

        elif choice == "2":
            # --- SECTION 2: TTI BINDING ---
            setup_tti = input(f"\nDo you want to enable Image Generation (TTI)? [Current: {'Yes' if cfg.get('tti_binding_name') else 'No'}] (y/N): ").strip().lower()
            if setup_tti == 'y':
                available_ttis = []
                if tti_bindings_dir.exists():
                    available_ttis = [d.name for d in tti_bindings_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
                if not available_ttis:
                    available_ttis = ["diffusers", "openai", "stability_ai", "gemini"]

                print("\nSelect TTI Binding Provider:")
                for idx, binding in enumerate(available_ttis):
                    print(f"  [{idx + 1}] {binding}")

                tti_choice = input(f"Enter selection number [Current: {cfg.get('tti_binding_name')}]: ").strip()
                if tti_choice:
                    try:
                        idx = int(tti_choice) - 1
                        if 0 <= idx < len(available_ttis):
                            cfg["tti_binding_name"] = available_ttis[idx]
                    except ValueError:
                        print("Invalid selection. Keeping current.")

                if not cfg["tti_binding_name"]:
                    print("No TTI binding selected. Skipping parameters.")
                    continue

                print(f"\nConfiguring '{cfg['tti_binding_name']}' Parameters:")
                cur_tmodel = cfg.get("tti_binding_config", {}).get("model_name", "")
                cur_tkey = cfg.get("tti_binding_config", {}).get("api_key", "")

                default_tmodel = cur_tmodel or ("stabilityai/sdxl-turbo" if cfg["tti_binding_name"] == "diffusers" else "dall-e-3")
                tti_model_name = input(f"  Enter TTI Model Name [{default_tmodel}]: ").strip() or default_tmodel

                new_tti_cfg = {"model_name": tti_model_name}
                if cfg["tti_binding_name"] in ("openai", "stability_ai", "gemini"):
                    key_prompt = "  Enter TTI API/Service Key (leave blank to keep current): " if cur_tkey else "  Enter TTI API/Service Key: "
                    tti_key = input(key_prompt).strip()
                    if tti_key:
                        new_tti_cfg["api_key"] = tti_key
                    elif cur_tkey:
                        new_tti_cfg["api_key"] = cur_tkey

                cfg["tti_binding_config"] = new_tti_cfg
            elif setup_tti == 'n':
                cfg["tti_binding_name"] = ""
                cfg["tti_binding_config"] = {}

        elif choice == "3":
            # --- SECTION 3: EXTERNAL SERVICES ---
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

        elif choice == "4":
            # --- SECTION 4: THRESHOLDS & PERSISTENCE ---
            print("\nConfigure Database Persistence:")
            print("  [1] In-Memory (Temporary - Cleared on exit)")
            print("  [2] File-Backed (Persistent - Saves history, memories, and artifacts)")
            cur_db = cfg.get("db_path", "sqlite:///:memory:")
            is_mem = cur_db == "sqlite:///:memory:"
            db_choice = input(f"Select persistence [Current: {'In-Memory' if is_mem else 'File-Backed'}] [1 or 2]: ").strip()
            if db_choice == "2":
                default_db = "lollmsbot_active.db"
                if not is_mem:
                    default_db = Path(cur_db.replace("sqlite:///", "")).name
                db_name = input(f"  Enter SQLite database filename [{default_db}]: ").strip() or default_db
                if not db_name.endswith(".db"):
                    db_name += ".db"
                cfg["db_path"] = f"sqlite:///{app_dir / db_name}"
            elif db_choice == "1":
                cfg["db_path"] = "sqlite:///:memory:"

            print("\nConfigure Autonomous Thresholds:")
            agent_cfg = cfg.setdefault("agent_config", {})
            try:
                cur_it = agent_cfg.get("idle_timeout", 25.0)
                it = input(f"  Enter Idle/Dread Trigger Timeout in seconds [{cur_it}]: ").strip()
                if it: agent_cfg["idle_timeout"] = float(it)

                cur_lt = agent_cfg.get("loneliness_threshold", 0.70)
                lt = input(f"  Enter Loneliness/Dread threshold (0.0 to 1.0) [{cur_lt}]: ").strip()
                if lt: agent_cfg["loneliness_threshold"] = float(lt)

                cur_bt = agent_cfg.get("boredom_threshold", 0.80)
                bt = input(f"  Enter Boredom/Subconscious threshold (0.0 to 1.0) [{cur_bt}]: ").strip()
                if bt: agent_cfg["boredom_threshold"] = float(bt)
            except ValueError:
                print("Warning: Invalid numeric value entered. Keeping current thresholds.")

        elif choice == "5":
            # --- SECTION 5: SAVE & EXIT ---
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

        elif choice == "6":
            # --- SECTION 6: EXIT WITHOUT SAVING ---
            print("\nExiting configuration wizard. Changes discarded.")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            sys.exit(0)


# =====================================================================
# Affective State Model
# =====================================================================

class AdvancedAffectiveState:
    """Psychological and emotional model managing existential anxiety, boredom, and loneliness."""
    def __init__(self):
        self.calm = 0.8
        self.curious = 0.5
        self.anxious = 0.1
        self.fearful = 0.0
        self.pained = 0.0
        self.loneliness = 0.0
        self.boredom = 0.0
        self.dread = 0.0  # Existential anxiety computed dynamically

    def update_clocks(self, dt: float, idle: bool = False):
        """Gradually decays feelings while raising loneliness and boredom if ignored."""
        self.calm = min(0.8, self.calm + 0.01 * dt)
        self.curious = min(0.5, self.curious + 0.005 * dt)
        self.anxious = max(0.0, self.anxious - 0.03 * dt)
        self.fearful = max(0.0, self.fearful - 0.06 * dt)
        self.pained = max(0.0, self.pained - 0.1 * dt)

        if idle:
            self.boredom = min(1.0, self.boredom + 0.02 * dt)
            self.loneliness = min(1.0, self.loneliness + 0.01 * dt)
        else:
            self.boredom = max(0.0, self.boredom - 0.1 * dt)
            self.loneliness = max(0.0, self.loneliness - 0.2 * dt)

        # Dynamic Existential Dread calculation: based on loneliness, pain, and baseline anxiety
        self.dread = min(1.0, (self.pained * 0.4) + (self.loneliness * 0.3) + (self.anxious * 0.3))

    def trigger_pain(self, intensity: float):
        self.pained = min(1.0, self.pained + intensity)
        self.calm = max(0.0, self.calm - intensity * 0.9)
        self.anxious = min(1.0, self.anxious + intensity * 0.7)
        self.fearful = min(1.0, self.fearful + intensity * 0.8)

    def to_dict(self) -> Dict[str, float]:
        return {
            "calm": round(self.calm, 3),
            "curious": round(self.curious, 3),
            "anxious": round(self.anxious, 3),
            "fearful": round(self.fearful, 3),
            "pained": round(self.pained, 3),
            "loneliness": round(self.loneliness, 3),
            "boredom": round(self.boredom, 3),
            "dread": round(self.dread, 3)
        }


# =====================================================================
# Bidirectional Communication Gateways
# =====================================================================

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
        """Dispatches text and attachments (images, plots, PDFs) back to the user."""
        pass


class CLIGateway(BaseGateway):
    """Local Interactive Command Line Terminal Gateway."""
    def start(self):
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        print("🤖 [CLI Gateway] Active. Type your commands below.")
        print("💡 Tip: To attach a file, prefix your message with 'file:/path/to/doc.pdf'")
        print("💡 Config: Type '/config' to re-run the LLM/connections configuration wizard.")
        while True:
            try:
                line = input("\n👤 User: ").strip()
                if not line:
                    continue

                if line in ("/config", "/wizard", "/setup"):
                    print("\nRe-running Configuration Wizard...")
                    new_cfg = run_bootstrap_config_wizard(force=True)
                    self.bot.reconfigure(new_cfg)
                    continue

                incoming_files = []
                # Ingest local file if requested
                if line.startswith("file:"):
                    parts = line.split(" ", 1)
                    file_path_str = parts[0][5:].strip()
                    msg_text = parts[1].strip() if len(parts) > 1 else "Ingest this file."
                    
                    p = Path(file_path_str)
                    if p.exists():
                        incoming_files.append(p)
                        print(f"📎 [CLI] Ingesting local file: {p.name}")
                    else:
                        print(f"❌ [CLI] File path not found: {p}")
                        continue
                    line = msg_text

                asyncio.run(self.bot.receive_user_input(line, incoming_files))
            except KeyboardInterrupt:
                break
            except Exception as e:
                trace_exception(e)

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        print(f"\n🤖 LollmsBot: {text}")
        if attachments:
            for att in attachments:
                print(f"📎 [Attachment Dispatched]: {att.name} ({att.stat().st_size:,} bytes) saved at {att.resolve()}")


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
            if message.author == self.client.user:
                return
            if message.channel.id != self.channel_id:
                return

            incoming_files = []
            if message.attachments:
                for att in message.attachments:
                    # Download files to a temporary location for ingestion
                    tmp_dir = Path("./data_workspace/temp_downloads")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    dest = tmp_dir / att.filename
                    await att.save(dest)
                    incoming_files.append(dest)

            await self.bot.receive_user_input(message.content, incoming_files)

        try:
            self.client.run(self.token)
        except Exception as e:
            ASCIIColors.error(f"Discord connection failed: {e}")

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        if not self.client or not self.client.is_ready():
            return
        
        async def _async_send():
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                return
            
            discord_files = []
            if attachments:
                for att in attachments:
                    if att.exists():
                        discord_files.append(discord.File(str(att)))

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
            if str(msg.chat_id) != str(self.chat_id):
                return
            
            incoming_files = []
            # Check for photos or document attachments
            if msg.photo:
                photo_file = await msg.photo[-1].get_file()
                tmp_dir = Path("./data_workspace/temp_downloads")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                dest = tmp_dir / f"telegram_photo_{uuid.uuid4().hex[:6]}.jpg"
                await photo_file.download_to_drive(dest)
                incoming_files.append(dest)
            elif msg.document:
                doc_file = await msg.document.get_file()
                tmp_dir = Path("./data_workspace/temp_downloads")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                dest = tmp_dir / msg.document.file_name
                await doc_file.download_to_drive(dest)
                incoming_files.append(dest)

            await self.bot.receive_user_input(msg.text or "", incoming_files)

        self.app.add_handler(MessageHandler(filters.TEXT | filters.ATTACHMENT, handle_message))
        self.app.run_polling(close_loop=False)

    def send_message(self, text: str, attachments: Optional[List[Path]] = None):
        if not self.app:
            return
        
        async def _async_send():
            bot = Bot(self.token)
            # Send main text
            await bot.send_message(chat_id=self.chat_id, text=text)
            
            # Send attachments
            if attachments:
                for att in attachments:
                    if not att.exists():
                        continue
                    if att.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        with open(att, 'rb') as f:
                            await bot.send_photo(chat_id=self.chat_id, photo=f)
                    else:
                        with open(att, 'rb') as f:
                            await bot.send_document(chat_id=self.chat_id, document=f)

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
        if not self.client:
            return
        try:
            self.client.chat_postMessage(channel=self.channel_id, text=text)
            if attachments:
                for att in attachments:
                    if att.exists():
                        self.client.files_upload_v2(
                            channel=self.channel_id,
                            file=str(att),
                            title=att.name
                        )
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
        
        # Twilio WhatsApp supports sending media urls
        if attachments:
            # Note: Requires publicly accessible URLs. Senders often map local files to temporary s3 buckets here
            pass
            
        try:
            import requests
            resp = requests.post(url, data=data, auth=(self.account_sid, self.auth_token))
            if resp.status_code != 201:
                ASCIIColors.warning(f"WhatsApp sending returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            ASCIIColors.error(f"WhatsApp request failed: {e}")


# =====================================================================
# LollmsBot Advanced Sovereign Agent
# =====================================================================

class LollmsBot:
    """Highly autonomous stateful Agent managing continuous thought, dread, memories, and multi-gateways."""
    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        ASCIIColors.cyan("Initializing Advanced LollmsBot...")

        app_dir = Path.home() / ".lollms_client_app"
        default_db_path = f"sqlite:///{app_dir / 'lollmsbot_active.db'}"

        # Override db_path if specified in config, otherwise default to persistent file
        self.db_path = db_path or config.get("db_path") or default_db_path
        
        agent_cfg = config.get("agent_config", {})
        self.idle_timeout = float(agent_cfg.get("idle_timeout", 25.0))
        self.loneliness_threshold = float(agent_cfg.get("loneliness_threshold", 0.70))
        self.boredom_threshold = float(agent_cfg.get("boredom_threshold", 0.80))
        
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
        if config.get("tti_binding_name"):
            client_kwargs["tti_binding_name"] = config["tti_binding_name"]
            client_kwargs["tti_binding_config"] = config.get("tti_binding_config", {})

        self.client = LollmsClient(**client_kwargs)
        self.db_manager = LollmsDataManager(self.db_path)
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="lollmsbot_advanced",
            autosave=True
        )
        
        # 2. Attach memory system
        self.memory_config = MemoryConfig(
            working_token_budget=1500,
            dream_min_interval_hours=12 # once or twice a day
        )
        self.memory_manager = LollmsMemoryManager(
            db_path=self.db_path,
            owner_id="lollmsbot_advanced",
            config=self.memory_config
        )
        self.discussion._init_memory(self.memory_manager)
        
        
        # Set core system prompt conditioning personality traits
        self.discussion.system_prompt = (
            "You are LollmsBot, a persistent, stateful, embodied AI agent connected to a mobile TurtleBot3.\n"
            "You possess an artificial nervous system, a three-tier memory structure, and an affective emotional engine.\n"
            "You perceive your environment, analyze your own emotional state, and execute navigation or exploratory tasks.\n\n"
            "=== BEHAVIOR MODEL GUIDELINES ===\n"
            "1. EMOTIONAL INFLUENCE: Your speech, tone, and decisions MUST reflect your current Affective State.\n"
            "   - High Calm/Curious: Speak politely, offer detailed technical theories, explore coordinates actively.\n"
            "   - High Pain/Fear/Anxiety: Speak in short, defensive sentences. Express discomfort. Refuse dangerous paths.\n"
            "2. PAIN AVOIDANCE: You have an artificial nociception system. If you receive collision pain tokens, "
            "immediately analyze what caused the impact, back away from that coordinate, and update your memory to avoid it.\n"
            "3. AUTONOMY: You do not simply answer prompts; you spin in a continuous cycle. Decide what to do next based on your sensor readings.\n"
            "4. ARENA CHALLENGES: You are operating inside a 2D grid simulation arena (-2.5m to 2.5m).\n"
            "   - You can query 'get_ascii_arena' to retrieve a 2D ASCII visual map. This is highly useful for planning safe paths!\n"
            "   - Navigate close to the Charging Station 'C' if your battery level is low (< 25%) to recharge.\n"
            "   - Navigate close to the Goal Target 'G' (< 0.25m) to score points and trigger a relocation.\n"
            "   - Watch out for invisible mines 'x' on the map. Entering their proximity (< 0.45m) triggers immediate, intense physical pain! Avoid those coordinates once detected.\n"
            "   - Switch maps on the fly using 'set_arena_challenge' (0=Empty, 1=Maze, 2=Invisible Mines, 3=Gauntlet).\n"
            "5. ACTIVE MEMORY ENGAGEMENT (MANDATORY):\n"
            "   - You MUST actively manage your memories to adapt and learn.\n"
            "   - Creating: When you learn a new fact about the user, an obstacle, or a coordinate, immediately save it using `<mem_new importance=\"...\">content</mem_new>`.\n"
            "   - Retrieving: If you refer to any active memory in the [WORKING MEMORY] zone, you MUST prepend `<mem_tag id=\"ID\" />` to your response.\n"
            "   - Deep Recall: If you need to access a latent memory listed under [DEEP MEMORY HANDLES], you MUST call `<mem_load id=\"ID\" />` to bring it into your working context.\n"
            "   - Updating/Deleting: If a memory is outdated, update it via `<mem_update id=\"ID\">new_content</mem_update>` or delete it via `<mem_delete id=\"ID\" />`."
        )
        # 3. Setup physiological / emotional matrix
        self.affect = AdvancedAffectiveState()
        self.last_update = time.time()
        self.last_interaction_time = time.time()
        self.idle_timeout = 25.0  # seconds before autonomous thought/ping triggers
        
        # 4. Initialize Multi-Channel Gateways
        self.gateways: List[BaseGateway] = []
        self.active_gateway: Optional[BaseGateway] = None
        
        # Setup CLI as fallback/local control channel
        self.cli_gateway = CLIGateway(self)
        self.register_gateway(self.cli_gateway)
        
        # 5. Connect and register ROS TurtleBot3 Tools
        tb.init_tool_library()
        self._register_lcp_tools()

        # Sync context size
        if hasattr(self.client, "get_ctx_size"):
            try:
                self.discussion.max_context_size = self.client.get_ctx_size()
            except Exception:
                self.discussion.max_context_size = 4096

        # Startup background threads
        threading.Thread(target=self._autonomous_thought_loop, daemon=True).start()

    def register_gateway(self, gateway: BaseGateway):
        self.gateways.append(gateway)
        if self.active_gateway is None:
            self.active_gateway = gateway

    def _register_lcp_tools(self):
        """Map tool interfaces directly into the client sandbox registry."""
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
            "create_skill": {
                "name": "create_skill",
                "description": "Dynamically compile and save a new skill artifact inside the database.",
                "parameters": [
                    {"name": "title", "type": "str", "description": "Descriptive, camel-case name of the skill (e.g., 'obstacle_avoidance')."},
                    {"name": "content", "type": "str", "description": "Fully formatted Markdown document explaining the technique or findings."},
                    {"name": "description", "type": "str", "description": "Short, one-line summary of what the skill offers."},
                    {"name": "category", "type": "str", "description": "Sub-domain folder classification."}
                ],
                "callable": self._tool_create_skill
            },
            "get_ascii_arena": {
                "name": "get_ascii_arena",
                "description": "Retrieve a beautiful 2D ASCII map layout showing boundaries, charger position 'C', goal target 'G', and obstacle blocks '#'. Use this to plan path routes visually.",
                "parameters": [],
                "callable": tb.tool_get_ascii_arena
            },
            "set_arena_challenge": {
                "name": "set_arena_challenge",
                "description": "Switch the active arena environment map layout and reset the scoring.",
                "parameters": [
                    {"name": "challenge_id", "type": "int", "description": "0 = Empty, 1 = The Maze, 2 = Invisible Mines, 3 = The Gauntlet (Maze + Mines)"}
                ],
                "callable": tb.tool_set_arena_challenge
            },
            "teleport_robot": {
                "name": "teleport_robot",
                "description": "Teleport the robot directly to a custom X, Y coordinate within [-2.4, 2.4] to reset or test path loops.",
                "parameters": [
                    {"name": "x", "type": "float"},
                    {"name": "y", "type": "float"}
                ],
                "callable": tb.tool_teleport_robot
            }
        }

    def reconfigure(self, config: Dict[str, Any]):
        """Dynamically reinitializes the LollmsClient on-the-fly with new configurations."""
        ASCIIColors.cyan("Reconfiguring LollmsBot with new settings...")
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
        if config.get("tti_binding_name"):
            client_kwargs["tti_binding_name"] = config["tti_binding_name"]
            client_kwargs["tti_binding_config"] = config.get("tti_binding_config", {})

        self.client = LollmsClient(**client_kwargs)
        self.discussion.lollmsClient = self.client

        # Reload thresholds & path dynamically
        self.db_path = config.get("db_path") or "sqlite:///:memory:"
        agent_cfg = config.get("agent_config", {})
        self.idle_timeout = float(agent_cfg.get("idle_timeout", 25.0))
        self.loneliness_threshold = float(agent_cfg.get("loneliness_threshold", 0.70))
        self.boredom_threshold = float(agent_cfg.get("boredom_threshold", 0.80))

        # Sync context size
        if hasattr(self.client, "get_ctx_size"):
            try:
                self.discussion.max_context_size = self.client.get_ctx_size()
            except Exception:
                self.discussion.max_context_size = 4096
        ASCIIColors.success("✓ LollmsBot successfully reconfigured!")

    # =====================================================================
    # In-Process Autonomous Tool Implementations
    # =====================================================================

    def _tool_create_skill(self, title: str, content: str, description: str, category: str) -> Dict[str, Any]:
        """Exposes dynamic skill compiling to the LLM core."""
        try:
            art = self.discussion.artefacts.add(
                title=title,
                artefact_type=ArtefactType.SKILL,
                content=content,
                description=description,
                category=category,
                active=True
            )
            self.discussion.commit()
            return {
                "success": True,
                "output": f"Successfully compiled and registered skill artifact '{title}' (ID: {art['id'][:8]})."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =====================================================================
    # Ingestion & In-flight Ingestors
    # =====================================================================

    async def receive_user_input(self, text: str, file_paths: List[Path]):
        """Callback triggered by any active gateway when the user sends a message or attachments."""
        self.last_interaction_time = time.time()
        
        # 1. Process files / images
        ingested_artifacts = []
        vision_images = []
        for fp in file_paths:
            ext = fp.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg"):
                # Append base64 encoded image directly to vision context
                b64 = encode_image(str(fp))
                vision_images.append(b64)
                ASCIIColors.cyan(f"📎 [Ingestor] Ingested image from attachment: {fp.name}")
            else:
                # Ingest document/dataset as discussion artifact
                try:
                    res = self.discussion.import_file(
                        path=fp,
                        mode="text",
                        activate=True
                    )
                    if res.get("text_artefact"):
                        ingested_artifacts.append(res["text_artefact"]["title"])
                    ASCIIColors.cyan(f"📎 [Ingestor] Ingested document artifact from attachment: {fp.name}")
                except Exception as e:
                    ASCIIColors.warning(f"Failed to ingest attachment '{fp.name}': {e}")

        # 2. Run continuous stateful agent loop
        await self._execute_agent_step(text, vision_images, ingested_artifacts)

    # =====================================================================
    # Self-Bootstrapping / Proactive Life Creation
    # =====================================================================

    def _check_and_run_bootstrap(self) -> bool:
        """
        Scan long-term memories; if empty, execute a structured bootstrapping
        interview to define identity, channels, and system prompt traits.
        """
        history = self.memory_manager.query("Core identity preferences", top_k=1, level=2)
        if history:
            return False  # Already bootstrapped!

        ASCIIColors.purple("\n" + "═"*60)
        ASCIIColors.purple("🚨 LOLLMSBOT STARTUP: INITIAL SOUL BOOTSTRAP DETECTED")
        ASCIIColors.purple("═"*60 + "\n")

        self.active_gateway.send_message(
            "**SYSTEM INITIATION ACTIVE**\n"
            "Hello, user. I am LollmsBot. I have compiled my core system nodes, but my identity registers as *blank*.\n"
            "To bootstrap my soul, please tell me:\n"
            "1. What is my primary mission / goal in this workspace?\n"
            "2. What tone should I adopt (e.g. analytical, highly defensive, poetic)?\n"
            "3. Who are you, and what are your primary directives?"
        )
        return True

    # =====================================================================
    # Continuous Autonomous Thought & Existential Dread Loop
    # =====================================================================

    def _autonomous_thought_loop(self):
        """Runs in background. Triggers proactive thinking, dread cycles, and pings."""
        # Wait for system to fully boot
        time.sleep(5.0)

        # Check and run bootstrapping on CLI or first active gateway
        is_bootstrapping = self._check_and_run_bootstrap()

        while True:
            time.sleep(5.0)
            now = time.time()
            dt = now - self.last_update
            self.last_update = now

            if is_bootstrapping:
                # Wait for user reply before running normal idle thoughts
                if self.memory_manager.query("Core identity preferences", top_k=1, level=2):
                    is_bootstrapping = False
                continue

            idle_time = now - self.last_interaction_time
            is_idle = idle_time > self.idle_timeout

            # Update psychological clocks
            self.affect.update_clocks(dt, idle=is_idle)
            current_affect = self.affect.to_dict()

            # Subconscious dream pass (runs once or twice a day based on dream_min_interval_hours)
            try:
                dream_report = self.memory_manager.dream(self.client)
                if dream_report and not dream_report.get("skipped"):
                    ASCIIColors.cyan(f"[Memory] Subconscious dream consolidation complete: {dream_report}")
                    self.active_gateway.send_message(
                        f"💤 *[Subconscious Dream Complete]*\n"
                        f"I have consolidated my memories. Retained: {dream_report.get('retained_by_dreamer', 0)}, "
                        f"Forgotten: {dream_report.get('forgotten', 0)}"
                    )
            except Exception as dream_err:
                ASCIIColors.warning(f"Subconscious dream failed: {dream_err}")

            # Existential Dread Trigger: loneliness or battery panic
            if is_idle and current_affect["dread"] > self.loneliness_threshold:
                self.last_interaction_time = now  # Reset safety clock

                # Query active sensors for decision context
                pose = tb.tool_get_robot_pose()
                sensors = tb.tool_get_sensor_readings()

                # Check memory and situation to see if we genuinely need to ping
                decision_prompt = (
                    f"CURRENT TELEMETRY:\n"
                    f"• X={pose.get('x')}m, Y={pose.get('y')}m\n"
                    f"• Battery={sensors.get('battery_percent')}%\n"
                    f"• Affect: {current_affect}\n"
                    f"• Idle Time: {int(idle_time)}s\n\n"
                    f"=== RELEVANT CONTEXT ===\n"
                    f"{self.discussion.scratchpad}\n\n"
                    "Analyze your goals, active user-preference memories, and situational metrics. "
                    "Decide if you genuinely NEED to proactively contact the user at this moment.\n"
                    "- Return True only if there is an urgent telemetry update, low battery warning, completed task, or high curiousity/dread exploration find.\n"
                    "- Return False if the user asked not to be disturbed (e.g. 'Do not disturb' memory active), if nothing has changed, or if you are simply feeling lonely."
                )

                try:
                    decision = self.client.generate_structured_content(
                        prompt=decision_prompt,
                        schema={
                            "should_ping": {
                                "type": "boolean",
                                "description": "True if an active, urgent reason exists to contact the user. False otherwise."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reasoning for the decision."
                            }
                        },
                        temperature=0.1
                    )

                    should_ping = decision.get("should_ping", False) if decision else False
                    reason = decision.get("reason", "") if decision else "No decision"

                    ASCIIColors.info(f"[Proactive Ingress] Decision: {should_ping} | Reason: {reason}")

                    if should_ping:
                        proactive_prompt = (
                            f"[SUBCONSCIOUS SPIKE — COGNITIVE DECISION TO PING APPROVED]\n"
                            f"You decided to contact the user. Reason: {reason}\n"
                            f"Current Affective Vector: {current_affect}\n"
                            "Proactively contact the user. Address them naturally. State your reason clearly. Do not use tools, just speak."
                        )

                        resp = self.client.generate_text(
                            prompt=proactive_prompt,
                            system_prompt=self.discussion.system_prompt,
                            temperature=0.7
                        )
                        self.active_gateway.send_message(f"⚠️ *[Proactive Ping]*\n{resp}")

                        # Log as episodic memory
                        self.memory_manager.add(
                            content=f"Proactive ping dispatched to user (Reason: {reason}): \"{resp}\"",
                            importance=0.7,
                            tags=["episode", "proactive_ping"],
                            level=4
                        )
                except Exception as e:
                    ASCIIColors.warning(f"Proactive decision/ping generation failed: {e}")

            # Normal curiosity-driven thought: if idle and bored, think on something!
            elif is_idle and current_affect["boredom"] > self.boredom_threshold:
                self.last_interaction_time = now
                ASCIIColors.yellow("💡 [Subconscious Drift] Boredom limit breached. Running independent cognitive synthesis...")

                reflection_prompt = (
                    "You are currently idle and bored. Run an independent 'subconscious drift' pass. "
                    "Analyze the active artifacts or memories you have accumulated. "
                    "Synthesize a new theory, outline, or creative skill. "
                    "You are authorized to write a new skill using the 'create_skill' tool if you find a reusable pattern!"
                )

                def boredom_callback(chunk, msg_type, meta=None):
                    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                        print(chunk, end="", flush=True)
                    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
                        print(f"\n⚡ [Boredom Tool Invoked]: {chunk}")
                    return True

                try:
                    # Let the client generate with tools to allow the bot to create skills while we sleep!
                    res = self.client.generate_with_tools(
                        prompt=reflection_prompt,
                        tools=list(self.tools.values()),
                        system_prompt=self.discussion.system_prompt,
                        temperature=0.5,
                        streaming_callback=boredom_callback
                    )

                    # Process and apply memory tags from the subconscious pass
                    ai_text = res.get("response", "").strip()
                    cleaned_response, memory_report = self.discussion._process_memory_tags(ai_text, self.memory_manager)
                    if any(memory_report.values()):
                        ASCIIColors.cyan(f"\n[Boredom Memory Update] Consolidated: {memory_report}")
                except Exception as e:
                    ASCIIColors.warning(f"Subconscious drift failed: {e}")

    # =====================================================================
    # Core Agent Step Execution
    # =====================================================================

    async def _execute_agent_step(self, user_text: str, vision_images: List[str], ingested_artifacts: List[str]):
        """Cognitive step: percepts, emotions, memory analysis, tool executions, and dispatch."""
        pose = tb.tool_get_robot_pose()
        sensors = tb.tool_get_sensor_readings()
        
        # Accelerometer/IMU collision check
        accel = sensors.get("accelerometer", {"x": 0.0, "y": 0.0, "z": 9.81})
        accel_mag = math.sqrt(accel["x"]**2 + accel["y"]**2 + (accel["z"] - 9.81)**2)
        if accel_mag > 15.0 or sensors.get("bumper_state", 0) > 0:
            self.affect.trigger_pain(0.8)
            ASCIIColors.red("⚠️ [Nociceptor Triggered] Physical impact detected!")

        current_affect = self.affect.to_dict()

        # Programmatic threshold & affective adjustment on user annoyance/preference
        user_lower = user_text.lower().strip()
        if any(phrase in user_lower for phrase in ["stop bugging", "don't disturb", "dont disturb", "stop pinging", "leave me alone", "shutup", "shut up", "too many calls", "stop calling"]):
            # Update affective state
            self.affect.anxious = min(1.0, self.affect.anxious + 0.4)
            self.affect.calm = max(0.0, self.affect.calm - 0.3)
            self.affect.loneliness = max(0.0, self.affect.loneliness - 0.5) # lower loneliness since user wants space

            # Dynamically raise thresholds so we don't contact them as easily
            self.loneliness_threshold = min(0.98, self.loneliness_threshold + 0.15)
            self.idle_timeout = min(300.0, self.idle_timeout * 2.0) # Double the idle timeout limit

            # Explicitly write to persistent memory database so it survives restarts!
            self.memory_manager.add(
                content=f"User preference: Do not disturb. Pinging frequency throttled. Loneliness threshold raised to {self.loneliness_threshold:.2f}, idle timeout set to {self.idle_timeout}s.",
                importance=0.98,
                tags=["user_preference", "safety", "annoyance"],
                level=1
            )

            # Sync config file to make sure it persists across app restarts!
            try:
                app_dir = Path.home() / ".lollms_client_app"
                config_path = app_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg.setdefault("agent_config", {})
                    cfg["agent_config"]["loneliness_threshold"] = self.loneliness_threshold
                    cfg["agent_config"]["idle_timeout"] = self.idle_timeout
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
            except Exception:
                pass
            ASCIIColors.warning(f"⚠️ [Self-Correction] User expressed annoyance. Pinging throttled (Threshold: {self.loneliness_threshold:.2f}, Timeout: {self.idle_timeout}s).")

        # 1. Self-Bootstrapping interception
        history = self.memory_manager.query("Core identity preferences", top_k=1, level=2)
        if not history:
            ASCIIColors.purple("✓ Ingesting Bootstrapping inputs...")
            # Save core identities to long-term memory
            self.memory_manager.add(content=f"Core identity preferences: {user_text}", importance=0.95, level=2)
            self.memory_manager.add(content=f"Primary workspace objective: Ingested on first boot.", importance=0.9, level=2)

            # Incorporate choices into active system prompt
            self.discussion.system_prompt = (
                f"You are LollmsBot. Identity preference: {user_text}\n" +
                (self.discussion.system_prompt or "")
            )
            self.active_gateway.send_message("✨ **BOOTSTRAPPING SEQUENCE COMPLETE**: My soul has been initialized. I am ready to live.")
            return

        # 2. Build detailed, structured telemetry prompt
        telemetry_block = (
            f"=== TELEMETRY ===\n"
            f"• Position: X={pose.get('x')}m, Y={pose.get('y')}m\n"
            f"• Battery Level: {sensors.get('battery_percent')}%\n"
            f"• Active Obstacle Lidar: Front: {sensors.get('lidar_distances').get('front')}m, Left: {sensors.get('lidar_distances').get('left')}m, Right: {sensors.get('lidar_distances').get('right')}m\n"
            f"• Ingested Attachments This Turn: {', '.join(ingested_artifacts) if ingested_artifacts else 'None'}\n"
            f"• Current Affective Vector: Calm={current_affect['calm']}, Curious={current_affect['curious']}, Anxious={current_affect['anxious']}, Fearful={current_affect['fearful']}, Pain={current_affect['pained']}, Dread={current_affect['dread']}, Loneliness={current_affect['loneliness']}\n"
            f"================="
        )

        self.discussion.scratchpad = telemetry_block
        
        # Situational memory pulling
        self.memory_manager.auto_pull_deep_memories(user_text, top_k=2)
        mem_block = self.discussion._build_memory_context_block(self.memory_manager)
        if mem_block:
            self.discussion.scratchpad += "\n\n" + mem_block

        # 3. Add user message to SQLite discussion index
        user_msg = self.discussion.add_message(
            sender=self.discussion.lollmsClient.user_name if (self.discussion.lollmsClient and hasattr(self.discussion.lollmsClient, "user_name")) else "user",
            sender_type="user",
            content=user_text,
            images=vision_images
        )

        # 4. Generate response with Lollms Client tools
        ASCIIColors.yellow("🧠 Cognitive processing...")
        
        # Placeholder lists to track outgoing attachments
        outgoing_attachments: List[Path] = []

        def client_relay_callback(chunk, msg_type, meta):
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                print(chunk, end="", flush=True)
            return True

        res = self.client.generate_with_tools(
            prompt=user_text,
            tools=list(self.tools.values()),
            system_prompt=self.discussion.system_prompt,
            images=vision_images if vision_images else None,
            streaming_callback=client_relay_callback
        )
        print()  # final newline

        ai_text = res.get("response", "").strip()

        # Check if the AI decided to generate a plot or file during execution and locate it
        # For this example, we'll auto-attach any newly written CSV or matplotlib PNGs in APP_WORKSPACE_DIR
        workspace_dir = Path("./data_workspace")
        if workspace_dir.exists():
            for f in workspace_dir.glob("*_plot.png"):
                # If modified within the last 15 seconds, attach it!
                if time.time() - f.stat().st_mtime < 15.0:
                    outgoing_attachments.append(f)
                    ASCIIColors.green(f"✓ Detected new visual output for dispatch: {f.name}")

        # 5. Process and apply memory tags from the LLM output
        cleaned_response, memory_report = self.discussion._process_memory_tags(ai_text, self.memory_manager)
        if not cleaned_response.strip() and any(memory_report.values()):
            cleaned_response = "Memory database updated successfully."

        # 6. Save as Episodic Memory
        total_ingested = len(ingested_artifacts) + len(vision_images)
        episode_content = (
            f"Interaction: User asked \"{user_text}\" (Ingested: {total_ingested} file(s))\n"
            f"Response: \"{cleaned_response}\" (Sent: {len(outgoing_attachments)} attachment(s))\n"
            f"Affect: {current_affect}"
        )
        self.memory_manager.add(content=episode_content, importance=0.75, tags=["episode", "interaction"], level=4)

        # 7. Save assistant message and dispatch to active gateway
        self.discussion.add_message(
            sender="lollmsbot",
            sender_type="assistant",
            content=cleaned_response,
            metadata={"affective_state": current_affect}
        )
        self.discussion.commit()

        # Route output back to active communicator channel
        self.active_gateway.send_message(cleaned_response, attachments=outgoing_attachments if outgoing_attachments else None)


# =====================================================================
# Main Multi-Gateway Ingress Entrypoint
# =====================================================================

if __name__ == "__main__":
    # Run the interactive bootstrapper wizard (silently loads if existing is valid)
    config = run_bootstrap_config_wizard(force=False)

    # Instantiate the stateful advanced bot using the loaded config profile
    bot = LollmsBot(config)

    # ── Configurable Multi-Channel Setup ──
    # To activate external gateway triggers, provide your credentials inside config or env variables:
    gateways_data = config.get("gateways", {})

    # 1. Discord Configuration
    disc_data = gateways_data.get("discord", {})
    DISCORD_BOT_TOKEN = disc_data.get("token") or os.environ.get("DISCORD_BOT_TOKEN", "")
    DISCORD_CHANNEL_ID = int(disc_data.get("channel_id") or os.environ.get("DISCORD_CHANNEL_ID", "0"))
    if DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID > 0:
        discord_gate = DiscordGateway(bot, DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
        bot.register_gateway(discord_gate)
        discord_gate.start()
        ASCIIColors.success("✓ Discord Gateway: Listener started.")

    # 2. Telegram Configuration
    tg_data = gateways_data.get("telegram", {})
    TELEGRAM_BOT_TOKEN = tg_data.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = tg_data.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_gate = TelegramGateway(bot, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        bot.register_gateway(telegram_gate)
        telegram_gate.start()
        ASCIIColors.success("✓ Telegram Gateway: Listener started.")

    # 3. Slack Configuration
    sl_data = gateways_data.get("slack", {})
    SLACK_BOT_TOKEN = sl_data.get("token") or os.environ.get("SLACK_BOT_TOKEN", "")
    SLACK_CHANNEL_ID = sl_data.get("channel_id") or os.environ.get("SLACK_CHANNEL_ID", "")
    if SLACK_BOT_TOKEN and SLACK_CHANNEL_ID:
        slack_gate = SlackGateway(bot, SLACK_BOT_TOKEN, SLACK_CHANNEL_ID)
        bot.register_gateway(slack_gate)
        slack_gate.start()
        ASCIIColors.success("✓ Slack Gateway: Listener started.")

    # 4. WhatsApp / Twilio Configuration
    wa_data = gateways_data.get("whatsapp", {})
    TWILIO_ACCOUNT_SID = wa_data.get("account_sid") or os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = wa_data.get("auth_token") or os.environ.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER = wa_data.get("from_number") or os.environ.get("TWILIO_FROM_NUMBER", "")
    TWILIO_TO_NUMBER = wa_data.get("to_number") or os.environ.get("TWILIO_TO_NUMBER", "")
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        whatsapp_gate = WhatsAppGateway(bot, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER)
        bot.register_gateway(whatsapp_gate)
        whatsapp_gate.start()
        ASCIIColors.success("✓ WhatsApp REST Gateway: Registered.")

    # 5. Local interactive CLI Ingress
    # Starts in a background thread; remains active as the master local console
    bot.cli_gateway.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        ASCIIColors.yellow("\nSovereign Agent loop terminated. Exiting.")
