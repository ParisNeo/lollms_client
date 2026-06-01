#!/usr/bin/env python3
"""
lollmsbot_agent.py
==================
Embodied, stateful, persistent AI agent using LollmsClient, LollmsDiscussion,
and LollmsMemoryManager with a simulated/live ROS 2 TurtleBot3 robot.
Includes an interactive configuration wizard on first run.
"""

import sys
import os
import time
import math
import random
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project relative imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lollms_client
from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

# Import local LCP tool functions directly
import lollms_client.tools_bindings.lcp.default_tools.ros_turtlebot.ros_turtlebot as tb


def run_bootstrap_config_wizard() -> Dict[str, Any]:
    """
    Console wizard that automatically discovers and configures 
    available LLM and TTI bindings, saving results to the user's config file.
    """
    app_dir = Path.home() / ".lollms_client_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    config_path = app_dir / "config.json"
    
    # 1. Attempt to load existing config
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("llm_binding_name") and cfg.get("llm_binding_config", {}).get("model_name"):
                print(f"👁️ Found existing configuration: {cfg['llm_binding_name']} (Model: {cfg['llm_binding_config']['model_name']})")
                use_existing = input("Use this configuration? (Y/n): ").strip().lower()
                if use_existing != 'n':
                    return cfg
        except Exception as e:
            print(f"Warning: Failed to load existing config: {e}")
            
    print("\n=========================================================")
    print("🧙‍♂️ LoLLMS BOT CONFIGURATION WIZARD")
    print("=========================================================\n")
    
    # 2. Discover local LLM bindings
    lollms_client_dir = Path(lollms_client.__file__).parent
    llm_bindings_dir = lollms_client_dir / "llm_bindings"
    
    available_llms = []
    if llm_bindings_dir.exists():
        available_llms = [d.name for d in llm_bindings_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    if not available_llms:
        available_llms = ["ollama", "openai", "open_router", "claude", "gemini", "litellm", "vllm"]
        
    print("Please select an LLM Binding Provider:")
    for idx, binding in enumerate(available_llms):
        print(f"  [{idx + 1}] {binding}")
        
    while True:
        try:
            choice = int(input("\nEnter selection number: ").strip())
            if 1 <= choice <= len(available_llms):
                selected_llm = available_llms[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

    print(f"\nConfiguring '{selected_llm}' LLM Binding:")
    
    # Apply sensible model and host address defaults based on the chosen provider
    default_model = "gpt-4o-mini"
    default_host = ""
    
    if selected_llm == "ollama":
        default_model = "gemma2"
        default_host = "http://localhost:11434"
    elif selected_llm == "open_router":
        default_model = "meta-llama/llama-3-8b-instruct:free"
        default_host = "https://openrouter.ai/api/v1"
    elif selected_llm == "claude":
        default_model = "claude-3-5-sonnet-20240620"
    elif selected_llm == "gemini":
        default_model = "gemini-1.5-flash"
    
    model_name = input(f"Enter Model Name [{default_model}]: ").strip() or default_model
    llm_config = {"model_name": model_name}
    
    # Ask for host address if necessary
    if selected_llm in ("ollama", "open_router", "vllm", "llama_cpp_server", "litellm"):
        host_prompt = f"Enter Host Address [{default_host}]: " if default_host else "Enter Host Address: "
        host_addr = input(host_prompt).strip() or default_host
        if host_addr:
            llm_config["host_address"] = host_addr
            
    # Ask for API Key if necessary
    if selected_llm in ("openai", "open_router", "claude", "gemini", "litellm", "grok", "groq"):
        api_key = input("Enter API/Service Key (will be saved in clear text): ").strip()
        if api_key:
            llm_config["api_key"] = api_key
            
    cfg = {
        "llm_binding_name": selected_llm,
        "llm_binding_config": llm_config
    }
    
    # 3. Optional TTI (Image Generation) Setup
    setup_tti = input("\nDo you want to configure Image Generation (TTI)? (y/N): ").strip().lower()
    if setup_tti == 'y':
        tti_bindings_dir = lollms_client_dir / "tti_bindings"
        available_ttis = []
        if tti_bindings_dir.exists():
            available_ttis = [d.name for d in tti_bindings_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        if not available_ttis:
            available_ttis = ["diffusers", "openai", "stability_ai", "gemini"]
            
        print("\nPlease select a TTI Binding Provider:")
        for idx, binding in enumerate(available_ttis):
            print(f"  [{idx + 1}] {binding}")
            
        while True:
            try:
                choice = int(input("\nEnter selection number: ").strip())
                if 1 <= choice <= len(available_ttis):
                    selected_tti = available_ttis[choice - 1]
                    break
            except ValueError:
                pass
            print("Invalid selection. Please try again.")
            
        print(f"\nConfiguring '{selected_tti}' TTI Binding:")
        tti_model = "stabilityai/sdxl-turbo" if selected_tti == "diffusers" else "dall-e-3"
        tti_model_name = input(f"Enter TTI Model Name [{tti_model}]: ").strip() or tti_model
        
        tti_config = {"model_name": tti_model_name}
        if selected_tti in ("openai", "stability_ai", "gemini"):
            tti_key = input("Enter TTI API/Service Key: ").strip()
            if tti_key:
                tti_config["api_key"] = tti_key
                
        cfg["tti_binding_name"] = selected_tti
        cfg["tti_binding_config"] = tti_config

    # Save to user home app config folder
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"\n✅ Configuration saved successfully to {config_path}!")
    except Exception as e:
        print(f"Error saving config file: {e}")
        
    return cfg


class AffectiveState:
    """Implements a persistent psychological and emotional status model."""
    def __init__(self):
        # Emotional matrix vectors (0.0 to 1.0)
        self.calm = 0.8
        self.curious = 0.5
        self.anxious = 0.1
        self.fearful = 0.0
        self.pained = 0.0
        
    def step_decay(self, dt: float):
        """Gradually decay emotions back toward baseline over time."""
        self.calm = min(0.8, self.calm + 0.02 * dt)
        self.curious = min(0.5, self.curious + 0.01 * dt)
        self.anxious = max(0.0, self.anxious - 0.05 * dt)
        self.fearful = max(0.0, self.fearful - 0.08 * dt)
        self.pained = max(0.0, self.pained - 0.12 * dt)

    def trigger_pain(self, intensity: float):
        """Physical pain spikes nervous arousal and fear instantly."""
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
            "pained": round(self.pained, 3)
        }


class LollmsBot:
    """Stateful Embodied AI Agent combining Nociception, Emotion, Memory, and ROS Actions."""
    def __init__(self, config: Dict[str, Any], db_path: str = "sqlite:///:memory:"):
        ASCIIColors.cyan("Initializing LollmsBot (Standard Embodied Agent)...")
        
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
        self.db_manager = LollmsDataManager(db_path)
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="lollmsbot_core",
            autosave=True
        )
        
        # 2. Attach memory system
        self.memory_config = MemoryConfig(working_token_budget=1500)
        self.memory_manager = LollmsMemoryManager(
            db_path=db_path,
            owner_id="lollmsbot_core",
            config=self.memory_config
        )
        self.discussion._init_memory(self.memory_manager)
        
        # 3. Setup physiological and robotic couplings
        tb.init_tool_library()
        self.affect = AffectiveState()
        self.last_update = time.time()
        self.cumulative_ticks = 0
        
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
            "3. AUTONOMY: You do not simply answer prompts; you spin in a continuous cycle. Decide what to do next based on your sensor readings."
        )

        # Pre-populate Long-Term "Deep" Memory with stable goals
        self.memory_manager.add(
            content="LollmsBot core identity: An autonomous agent exploring simulated ruins.",
            importance=0.95,
            tags=["identity", "goal"],
            level=2
        )
        self.memory_manager.add(
            content="Critical safety guideline: Low battery (< 20%) requires navigating immediately back to docking station (0.0, 0.0).",
            importance=0.9,
            tags=["safety", "battery", "home"],
            level=2
        )

        # In-process tool registry mapping
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

        # Sync context budget limit
        if hasattr(self.client, "get_ctx_size"):
            try:
                self.discussion.max_context_size = self.client.get_ctx_size()
            except Exception:
                self.discussion.max_context_size = 4096

        ASCIIColors.success("✓ LollmsBot initialization complete. Nervous system active.")

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
            # Create a localized memory trace
            self.memory_manager.add(
                content=f"Pain alert at coordinates ({pose.get('x')}, {pose.get('y')}): {pain_report}",
                importance=0.9,
                tags=["pain", "nociception", "collision"],
                level=1
            )
            ASCIIColors.red(f"⚠️ [Nociceptor Triggered] {pain_report} Pain Level raised to: {self.affect.pained:.1%}")

        # 3. DECIDE: Assemble Prompt context with active state and memory blocks
        current_affect = self.affect.to_dict()
        
        # Build state context block for LLM
        state_context = (
            f"=== LOLLMSBOT PHYSICAL STATE ===\n"
            f"• Coordinates: X={pose.get('x')}m, Y={pose.get('y')}m, Yaw={math.degrees(pose.get('theta')):.1f}°\n"
            f"• Battery Level: {sensors.get('battery_percent')}%\n"
            f"• Active Obstacle Lidar: Front: {sensors.get('lidar_distances').get('front')}m, Left: {sensors.get('lidar_distances').get('left')}m, Right: {sensors.get('lidar_distances').get('right')}m\n"
            f"• Current Affective Vector: Calm={current_affect['calm']}, Curious={current_affect['curious']}, Anxious={current_affect['anxious']}, Fearful={current_affect['fearful']}, Pain={current_affect['pained']}\n"
            f"================================="
        )

        # Query and pull relevant memories based on sensory situation
        situational_cue = "Normal travel"
        if bumper > 0 or accel_magnitude > 15.0:
            situational_cue = "Collision pain avoidance"
        elif sensors.get('battery_percent', 100) < 25.0:
            situational_cue = "Low battery safety docking"
            
        self.discussion.scratchpad = state_context
        
        # Perform memory pre-turn decay and situational deep pulling
        self.memory_manager.apply_decay()
        if situational_cue != "Normal travel":
            self.memory_manager.auto_pull_deep_memories(situational_cue, top_k=2)
            
        mem_block = self.discussion._build_memory_context_block(self.memory_manager)
        if mem_block:
            self.discussion.scratchpad += "\n\n" + mem_block

        # Construct final directive prompt
        if user_command:
            prompt = f"User Command: \"{user_command}\"\nAnalyze physical telemetry, consult memory guidelines, and act."
        else:
            prompt = "Autonomous exploration turn. Analyze telemetry, update goals, and decide on actions."

        # 4. ACT: Run the decision through the LLM core using generate_with_tools
        ASCIIColors.yellow("🧠 Cognitive processing of state and goals...")
        
        # Callback to echo streaming to stdout
        def stdout_callback(chunk, msg_type, meta):
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
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
        
        print()  # final newline

        # 5. POST-PROCESS: Log as episodic memory turn to build autobiographical continuity
        ai_response = res.get("response", "").strip()
        episode_content = (
            f"Telemetry: Pose=({pose.get('x')}, {pose.get('y')}), Battery={sensors.get('battery_percent')}%, Affect={current_affect}\n"
            f"Exploration Log: \"{ai_response}\""
        )
        self.memory_manager.add(
            content=episode_content,
            importance=0.75,
            tags=["episode", "turn_log"],
            level=4
        )
        
        # Sync the discussion message node
        self.discussion.add_message(
            sender="lollmsbot",
            sender_type="assistant",
            content=ai_response,
            metadata={"affective_state": current_affect}
        )
        self.discussion.commit()

        # Print some summary info
        ASCIIColors.green(f"✓ Agent Turn completed. Consolidated {len(res.get('tool_calls', []))} action(s).")


# ── 🎭 Interactive Agent Session Runner ──
if __name__ == "__main__":
    # Run the interactive bootstrapper wizard first before starting any processes!
    config = run_bootstrap_config_wizard()
    
    bot = LollmsBot(config)
    
    # Run some initial autonomous setup turns
    bot.run_agent_step()
    
    # Inject a simulated collision shock to trigger pain nociceptors and watch LollmsBot adapt
    ASCIIColors.purple("\n" + "="*80)
    ASCIIColors.purple("🚨 INJECTING SIMULATED COLLISION SHOCK (Testing artificial nociception)...")
    ASCIIColors.purple("="*80 + "\n")
    
    tb.tool_trigger_nociception_test(18.5)
    
    # Run next step under pain spike
    bot.run_agent_step()
    
    # Run a third step to watch physical backing away and emotional recovery
    bot.run_agent_step()
    
    ASCIIColors.success("\n=========================================================")
    ASCIIColors.success("🎯 Embodied Sovereign Agent Simulation Turn Complete.")
    ASCIIColors.success("=========================================================\n")
