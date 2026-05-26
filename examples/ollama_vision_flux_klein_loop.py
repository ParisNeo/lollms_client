#!/usr/bin/env python3
"""
llm_vision_flux_klein_loop.py
============================
An interactive creative assistant that combines:
  • Vision-capable LLM (via Ollama) for prompt enhancement
    and image critique/understanding
  • Enhanced Diffusers Model Zoo supporting standard pipelines, pre-quantized 8-bit,
    and ultra-low memory GGUF quantized 4B, 9B, and 12B checkpoints.

Workflow:
  1. Allows choosing the root data folder (defaults to e:/data) [1.14.1].
  2. Displays the complete enhanced model zoo [1.12.1].
  3. Prompt the user to choose an optimized model from the zoo [1.12.1].
  4. If a GGUF model is selected, prompt for the preferred quantization level.
  5. Client fetches and applies optimized defaults dynamically in one line.
  6. Runs the interactive image generation and surgical manual editing loop.

Cooperative VRAM Management:
  When `cooperative_vram_management=True` is enabled, the client orchestrator
  automatically commands Ollama to unload the LLM from active memory right before
  running diffusion, and commands the Diffusers server to unload right before
  running LLM text/vision generation. This ensures smooth operations on consumer GPUs.

Requirements:
  pip install lollms_client ascii_colors pillow

Downloads:
  • A vision-capable model on Ollama (e.g. gemma4:e2b, llava, or mistral)
  • Your chosen model from the Diffusers TTI Zoo (auto-pulled on startup)
"""

import sys
import base64
import time
import json
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any, List

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Allow setting the base data directory (defaulting to e:/data)
try:
    data_dir_input = input("📂 Enter root data folder [default e:/data]: ").strip()
    DATA_DIR = Path(data_dir_input if data_dir_input else "e:/data")
except (EOFError, KeyboardInterrupt):
    DATA_DIR = Path("e:/data")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Output directory for saved images
OUTPUT_DIR = DATA_DIR / "generated_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# LLM: Vision-capable model configured in Ollama
LLM_MODEL_NAME = "gemma4:e2b"

# Maximum refinement iterations
MAX_ITERATIONS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def progress_callback(payload: dict):
    """Called during model download."""
    status = payload.get("status", "unknown")
    message = payload.get("message", "")
    completed = payload.get("completed", 0)
    total = payload.get("total", 100)

    if status == "downloading":
        pct = (completed / total * 100) if total else 0
        print(f"⬇️  [{pct:5.1f}%] {message}")
    elif status == "success":
        print(f"✅ {message}")
    elif status == "error":
        print(f"❌ ERROR: {message}")


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 data URI for vision input."""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def encode_pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to base64 data URI."""
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def save_image(image_bytes: bytes, prefix: str = "generated") -> Path:
    """Save image bytes to disk with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    path = OUTPUT_DIR / filename
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────

class CreativeAssistant:
    def __init__(self):
        self.llm_client: Optional[LollmsClient] = None
        self.tti_client: Optional[LollmsClient] = None
        self.llm = None
        self.tti = None
        self.selected_model_name = ""

    def setup(self):
        """Initialize both LLM and TTI bindings with cooperative VRAM enabled."""
        print("=" * 70)
        print("🎨 Creative Assistant: Enhanced Zoo & Collaborative VRAM")
        print("=" * 70)

        # ── Setup LLM (Ollama with vision) ─────────────────────────
        print("\n🧠 Initializing LLM (Ollama with vision)...")
        self.llm_client = LollmsClient(
            llm_binding_name="ollama",
            llm_binding_config={
                "model_name": LLM_MODEL_NAME,
                "host_address": "http://localhost:11434"
            },
            user_name="ParisNeo",
            ai_name="Lollms",
            cooperative_vram_management=True,
            debug=True
        )        
        self.llm = self.llm_client.llm
        print(f"✅ LLM loaded.")

        # ── Setup temporary TTI client to fetch Zoo options ───────
        print("\n🎨 Loading Diffusers Zoo Catalog...")
        temp_tti = LollmsClient(
            tti_binding_name="diffusers",
            tti_binding_config={
                "host": "localhost",
                "port": 9632,
                "auto_start_server": True,
                "wait_for_server": True,
                "venv_path": str(DATA_DIR / "tti_venv"),
                "models_path": str(DATA_DIR / "tti_models" / "diffusers"),
            }
        )
        zoo = temp_tti.tti.get_zoo()

        # Display the Zoo to the user
        print("\n=== AVAILABLE MODELS IN THE DIFFUSERS ZOO ===")
        for idx, item in enumerate(zoo):
            print(f"  [{idx:2d}] {item['name']} — {item['description']} ({item['size']})")
        print("=============================================")

        # Ask user for their choice
        try:
            choice_str = input(f"\n👉 Select model index (0-{len(zoo)-1}) [default 0 - SD 1.5]: ").strip()
            choice_idx = int(choice_str) if choice_str.isdigit() else 0
            if choice_idx < 0 or choice_idx >= len(zoo):
                choice_idx = 0
        except (EOFError, KeyboardInterrupt):
            choice_idx = 0

        # Retrieve optimized configuration using the new fast Zoo helper
        selected_item = zoo[choice_idx]
        self.selected_model_name = selected_item["name"]
        print(f"\n🎯 Resolving optimized config for: {self.selected_model_name}...")
        
        optimized_config = temp_tti.tti.get_zoo_model_config(choice_idx)
        
        # ── GGUF Quantization Selection ──
        if optimized_config.get("quant_backend") == "gguf":
            print("\n🛡️  [GGUF DETECTED] Select Quantization Level:")
            print("  [1] Q8_0   (High quality,   ~8.5 GB VRAM)")
            print("  [2] Q5_K_M (Balanced,       ~5.5 GB VRAM)")
            print("  [3] Q4_K_M (Optimal/Recom,   ~4.5 GB VRAM)")
            print("  [4] Q3_K_M (Low VRAM,       ~3.5 GB VRAM)")
            print("  [5] Q2_K   (Minimum VRAM,   ~2.5 GB VRAM)")
            
            try:
                q_choice = input("\n👉 Choice (1-5) [default 3]: ").strip()
                q_level = {"1": "Q8_0", "2": "Q5_K_M", "3": "Q4_K_M", "4": "Q3_K_M", "5": "Q2_K"}.get(q_choice, "Q4_K_M")
            except (EOFError, KeyboardInterrupt):
                q_level = "Q4_K_M"
                
            optimized_config["quant_level"] = q_level
            # Restrict downloading to only the chosen GGUF file
            optimized_config["allow_patterns"] = [f"*{q_level}.gguf", "*.json", "*.txt"]
            print(f"✅ Quantization level set to: {q_level}")
        
        # Merge custom path parameters
        full_tti_config = {
            "host": "localhost",
            "port": 9632,
            "auto_start_server": True,
            "wait_for_server": True,
            "venv_path": str(DATA_DIR / "tti_venv"),
            "models_path": str(DATA_DIR / "tti_models" / "diffusers"),
            "device": "cuda",
            **optimized_config
        }

        # Re-initialize the active TTI client with full settings
        self.tti_client = LollmsClient(
            tti_binding_name="diffusers",
            tti_binding_config=full_tti_config,
            cooperative_vram_management=True,
        )
        self.tti = self.tti_client.tti

        # Pull model if not present on disk
        print(f"\n⬇️  Ensuring selected model '{full_tti_config['model_name']}' is available...")
        pull_result = self.tti.pull_model(
            full_tti_config["model_name"], 
            allow_patterns=full_tti_config.get("allow_patterns"),
            progress_callback=progress_callback
        )
        if not pull_result.get("status"):
            print(f"⚠️  Pull message: {pull_result.get('message', pull_result.get('error'))}")

        print("\n✅ All systems ready!")
        print(f"   LLM : {LLM_MODEL_NAME}")
        print(f"   TTI : {self.selected_model_name}")
        print(f"   Cooperative VRAM Management: ENABLED")
        print(f"   Output Folder: {OUTPUT_DIR}")

    def print_vram_status(self):
        """Query and print the VRAM / active model status for both bindings."""
        print("\n📊 [VRAM STATUS MONITOR]")
        
        # 1. Query LLM (Ollama / Remote fallback)
        try:
            llm_ps = self.llm_client.llm.ps()
            if llm_ps:
                print("  🧠 LLM Status:")
                for m in llm_ps:
                    vram_gb = m.get("vram_size", 0) / (1024**3) if m.get("vram_size") else 0
                    device = m.get("device", "unknown")
                    if vram_gb > 0:
                        print(f"    - {m['model_name']} | Device: {device} | VRAM: {vram_gb:.2f} GB | GPU: {m.get('gpu_usage_percent', 0)}%")
                    else:
                        print(f"    - {m['model_name']} | Device: {device} | Status: {m.get('status', 'active')}")
            else:
                print("  🧠 LLM Status: Idle / Unloaded")
        except Exception as e:
            print(f"  🧠 LLM Status Query Failed: {e}")

        # 2. Query TTI (Diffusers / Remote fallback)
        try:
            tti_ps = self.tti_client.tti.ps()
            if tti_ps:
                print("  🎨 TTI Status:")
                for m in tti_ps:
                    is_loaded = m.get("is_loaded", False) or m.get("status") == "active (remote simulation)"
                    if is_loaded:
                        print(f"    - {m['model_name']} | Device: {m.get('device', 'unknown')} | Status: {m.get('status', 'loaded')}")
                    else:
                        print(f"    - {m['model_name']} | Status: Idle/Unloaded")
            else:
                print("  🎨 TTI Status: Idle / Unloaded")
        except Exception as e:
            print(f"  🎨 TTI Status Query Failed: {e}")
        print("-" * 30)

    def enhance_prompt(self, user_concept: str) -> str:
        """Use the LLM to enhance a simple user concept into a detailed image prompt."""
        print("\n" + "-" * 70)
        print("✨ Enhancing prompt with LLM...")
        print("-" * 70)

        self.print_vram_status()

        system_prompt = (
            "You are an expert prompt engineer for AI image generation. "
            "Take the user's concept and expand it into a detailed, vivid prompt "
            "optimized for Diffusers image generation. "
            "Include style, lighting, composition, mood, and artistic direction. "
            "Output ONLY the enhanced prompt text, no explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Concept: {user_concept}\n\nWrite the enhanced image generation prompt:"},
        ]

        response = self.llm_client.generate_from_messages(
            messages,
            n_predict=512,
            temperature=0.8,
            top_p=0.9,
            streaming_callback=streaming_callback,
        )
        print()  # newline after streaming

        if isinstance(response, dict) and not response.get("status", True):
            print(f"❌ Prompt enhancement failed: {response.get('error')}")
            return user_concept  # fallback

        enhanced = response.strip() if isinstance(response, str) else user_concept
        return enhanced

    def generate_image(self, prompt: str, seed: int = -1) -> bytes:
        """Generate image using the active Zoo model (will trigger automatic LLM unload)."""
        print("\n" + "-" * 70)
        print(f"🎨 Generating image with {self.selected_model_name}...")
        print("-" * 70)

        self.print_vram_status()

        params = {}
        if seed != -1:
            params["seed"] = seed

        t0 = time.time()
        image_bytes = self.tti_client.generate_image(
            prompt=prompt,
            negative_prompt="",
            **params,
        )
        elapsed = time.time() - t0
        print(f"✅ Image generated in {elapsed:.1f}s ({len(image_bytes)} bytes)")

        return image_bytes

    def critique_image(self, image_path: Path, original_prompt: str) -> Dict[str, Any]:
        """Use vision-capable LLM to critique the generated image (will trigger TTI unload)."""
        print("\n" + "-" * 70)
        print("👁️  LLM is reviewing the generated image...")
        print("-" * 70)

        self.print_vram_status()

        image_b64 = encode_image_to_base64(image_path)

        system_prompt = (
            "You are an expert art director and image critic. "
            "Review the generated image carefully. Assess:\n"
            "1. Prompt adherence (does it match the request?)\n"
            "2. Visual quality (composition, lighting, detail)\n"
            "3. Any issues or artifacts\n"
            "4. Specific suggestions for improvement if needed\n\n"
            "Respond in JSON format with keys: approved (bool), feedback (str), "
            "suggested_changes (str, empty if approved), score (int 1-10)."
        )

        user_content = [
            {"type": "text", "text": f"Original prompt: {original_prompt}\n\nPlease review this generated image:"},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm_client.generate_from_messages(
            messages,
            n_predict=1024,
            temperature=0.3,
            top_p=0.9,
            streaming_callback=streaming_callback,
        )
        print()  # newline after streaming

        if isinstance(response, dict) and not response.get("status", True):
            print(f"❌ Critique failed: {response.get('error')}")
            return {"approved": True, "feedback": "Critique failed, auto-approving.", "suggested_changes": "", "score": 7}

        text = response.strip() if isinstance(response, str) else ""

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            critique = json.loads(text)
            critique.setdefault("approved", False)
            critique.setdefault("feedback", "")
            critique.setdefault("suggested_changes", "")
            critique.setdefault("score", 5)
            return critique
        except json.JSONDecodeError:
            approved = "approve" in text.lower() and "not approve" not in text.lower()
            score = 7 if approved else 5
            return {
                "approved": approved,
                "feedback": text,
                "suggested_changes": "" if approved else "Please refine based on feedback.",
                "score": score,
            }

    def edit_image(self, image_path: Path, original_prompt: str, changes: str) -> bytes:
        """Use the active Zoo model's image editing capability (will trigger LLM unload)."""
        print("\n" + "-" * 70)
        print(f"🔧 Editing image with {self.selected_model_name}...")
        print("-" * 70)

        self.print_vram_status()

        pil_image = Image.open(image_path).convert("RGB")
        edit_prompt = f"{original_prompt}. {changes}"

        t0 = time.time()
        image_bytes = self.tti_client.edit_image(
            images=pil_image,
            prompt=edit_prompt,
            strength=0.6,
        )
        elapsed = time.time() - t0
        print(f"✅ Image edited in {elapsed:.1f}s ({len(image_bytes)} bytes)")

        return image_bytes

    def run(self):
        """Main interactive loop."""
        print("\n" + "=" * 70)
        print("🎨 CREATIVE ASSISTANT READY")
        print("=" * 70)
        print("Describe an image you'd like to create (or 'quit' to exit):")

        while True:
            try:
                user_input = input("\n📝 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            # ── Step 1: Enhance prompt ────────────────────────────────
            enhanced_prompt = self.enhance_prompt(user_input)
            print(f"\n🎯 Enhanced prompt:\n{enhanced_prompt}")

            # ── Step 2: Generate initial image ─────────────────────────
            current_image_bytes = self.generate_image(enhanced_prompt)
            current_path = save_image(current_image_bytes, prefix="v1")
            print(f"💾 Saved initial image: {current_path}")

            # ── Step 3: Refinement loop ───────────────────────────────
            iteration = 0
            current_prompt = enhanced_prompt

            while True:
                print("\nWhat would you like to do next with the generated image?")
                print("  [1] Approve and save this version")
                print("  [2] Let the LLM automatically critique and refine it")
                print("  [3] Manually edit this image with a custom prompt")
                
                try:
                    choice = input("\n👉 Choice (1/2/3): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                
                if choice == "1":
                    print(f"\n✅ Image approved! Final saved to: {current_path}")
                    break
                
                elif choice == "2":
                    if iteration >= MAX_ITERATIONS:
                        print("\n⏹️  Max auto-critique iterations reached! Choose another option.")
                        continue
                    
                    critique = self.critique_image(current_path, current_prompt)
                    print(f"\n📊 Critique Score: {critique['score']}/10")
                    print(f"📝 Feedback: {critique['feedback']}")

                    if critique["approved"] or critique["score"] >= 8:
                        print(f"\n✅ Image approved by LLM! Final saved to: {current_path}")
                        break

                    changes = critique.get("suggested_changes", critique.get("feedback", "Improve quality and adherence to prompt."))
                    print(f"\n🔧 Applying automatic changes: {changes}")

                    new_bytes = self.edit_image(current_path, current_prompt, changes)
                    iteration += 1
                    current_path = save_image(new_bytes, prefix=f"v{iteration + 1}")
                    print(f"💾 Saved revision {iteration + 1}: {current_path}")

                    current_prompt = f"{current_prompt}. Refined: {changes}"
                
                elif choice == "3":
                    try:
                        manual_changes = input("\n🔧 Enter manual edit instructions: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not manual_changes:
                        continue
                    
                    # Run image-to-image edit
                    new_bytes = self.edit_image(current_path, current_prompt, manual_changes)
                    iteration += 1
                    current_path = save_image(new_bytes, prefix=f"v{iteration + 1}")
                    print(f"💾 Saved revision {iteration + 1}: {current_path}")

                    current_prompt = f"{current_prompt}. Manual Edit: {manual_changes}"
                
                else:
                    print("Invalid choice. Please select 1, 2, or 3.")

            print("\n" + "=" * 70)
            print("Describe another image (or 'quit' to exit):")

        # Cleanup
        print("\n🧹 Cleaning up...")
        self.llm_client.llm.unload_model()
        self.tti_client.tti.unload_model()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    assistant = CreativeAssistant()
    assistant.setup()
    assistant.run()


if __name__ == "__main__":
    main()
