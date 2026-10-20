#!/usr/bin/env python3
"""
llm_vision_flux_klein_loop.py
============================
An interactive creative assistant that combines:
  • Vision-capable LLM (Mistral 3B + mmproj via llama.cpp) for prompt enhancement
    and image critique/understanding
  • FLUX.2 Klein 4B (smallest/fastest) for high-quality image generation and editing

Workflow:
  1. User describes an image concept
  2. LLM enhances the prompt with artistic direction
  3. FLUX.2 Klein generates the image
  4. LLM (with vision) reviews the generated image
  5. If approved → save to disk
  6. If needs changes → edit with feedback, loop back to review

Requirements:
  pip install lollms_client ascii_colors pillow

Downloads:
  • ~2.2 GB: mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
  • ~500 MB: matching mmproj file for vision
  • ~4 GB: FLUX.2-klein-4B (distilled, Apache 2.0, fully commercial)
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
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# LLM: Mistral 3B Instruct (small, fast, vision-capable with mmproj)
LLM_MODEL_NAME = "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
LLM_MMPROJ_NAME = "mmproj-mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"  # adjust to your actual filename

LLM_BINDING_CONFIG = {
    "models_path": "e:/data/models/llama_cpp_models",
    "binaries_path": "e:/data/bin/llm/llama_cpp_server",
    "ctx_size": 8192,
    "n_gpu_layers": -1,      # offload all to GPU if available
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "multimodal": True,     # enable vision projector detection
    "idle_timeout": 600,    # keep model loaded for 10 min
}

# TTI: FLUX.2 Klein 4B distilled (smallest, fastest, Apache 2.0)
FLUX_MODEL_NAME = "diffusers/FLUX.1-dev-bnb-4bit"
FLUX_BINDING_CONFIG = {
    "host": "localhost",
    "port": 9632,
    "model_name": FLUX_MODEL_NAME,
    "auto_start_server": True,
    "wait_for_server": True,
    "venv_path": "e:/data/tti_venv",
    "models_path": "e:/data/tti_models/diffusers",
    "device": "cuda",
    "torch_dtype_str": "bfloat16",
    "num_inference_steps": 4,      # Klein distilled: 4 steps is enough
    "guidance_scale": 1.0,       # distilled: CFG-free
    "width": 1024,
    "height": 1024,
    "seed": -1,
}

# Output directory for saved images
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    def setup(self):
        """Initialize both LLM and TTI bindings."""
        print("=" * 70)
        print("🎨 Creative Assistant: LLM Vision + FLUX.2 Klein")
        print("=" * 70)

        # ── Setup LLM (llama.cpp with vision) ─────────────────────────
        print("\n🧠 Initializing LLM (Mistral 3B + vision)...")
        self.llm_client = LollmsClient(
            llm_binding_name="llama_cpp_server",
            llm_binding_config=LLM_BINDING_CONFIG,
            user_name="user",
            ai_name="assistant",
        )
        self.llm = self.llm_client.llm

        # Download LLM model if missing
        model_path = Path(LLM_BINDING_CONFIG["models_path"]) / LLM_MODEL_NAME
        if not model_path.exists():
            print(f"\n⬇️  Downloading LLM model: {LLM_MODEL_NAME}")
            # Find in zoo or pull directly
            zoo = self.llm.get_zoo()
            found_idx = None
            for i, item in enumerate(zoo):
                if LLM_MODEL_NAME.replace(".gguf", "") in item.get("filename", ""):
                    found_idx = i
                    break
            
            if found_idx is not None:
                result = self.llm.download_from_zoo(found_idx, progress_callback=progress_callback)
            else:
                # Direct pull from bartowski's repo
                result = self.llm.pull_model(
                    "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
                    LLM_MODEL_NAME,
                    progress_callback=progress_callback,
                )
            
            if not result.get("status"):
                print(f"❌ Failed to download LLM: {result.get('error')}")
                sys.exit(1)

        # Check for mmproj and bind if found
        mmproj_path = Path(LLM_BINDING_CONFIG["models_path"]) / LLM_MMPROJ_NAME
        if mmproj_path.exists():
            print(f"🔗 Binding vision projector: {LLM_MMPROJ_NAME}")
            bind_result = self.llm.bind_multimodal_model(LLM_MODEL_NAME, LLM_MMPROJ_NAME)
            if not bind_result.get("status"):
                print(f"⚠️  Could not bind mmproj: {bind_result.get('error')}")
                print("   Vision capabilities may be limited.")
        else:
            print(f"⚠️  mmproj not found at {mmproj_path}")
            print("   Please download the matching mmproj file for vision support.")
            print("   Continuing with text-only mode...")

        # Load the LLM model
        print(f"\n🔌 Loading LLM model: {LLM_MODEL_NAME}")
        t0 = time.time()
        success = self.llm.load_model(LLM_MODEL_NAME)
        if not success:
            print("❌ Failed to load LLM model.")
            sys.exit(1)
        print(f"✅ LLM loaded in {time.time() - t0:.1f}s")

        # ── Setup TTI (FLUX.2 Klein via Diffusers) ──────────────────
        print("\n🎨 Initializing TTI (FLUX.2 Klein 4B)...")
        self.tti_client = LollmsClient(
            tti_binding_name="diffusers",
            tti_binding_config=FLUX_BINDING_CONFIG,
        )
        self.tti = self.tti_client.tti

        # Pull FLUX.2 Klein if not present
        print(f"\n⬇️  Ensuring FLUX.2 Klein is available...")
        pull_result = self.tti.pull_model(FLUX_MODEL_NAME, progress_callback=progress_callback)
        if not pull_result.get("status"):
            print(f"⚠️  Pull message: {pull_result.get('message', pull_result.get('error'))}")
            # May already exist, continue

        print("\n✅ All systems ready!")
        print(f"   LLM: {LLM_MODEL_NAME} (vision: {mmproj_path.exists()})")
        print(f"   TTI: {FLUX_MODEL_NAME}")
        print(f"   Output: {OUTPUT_DIR}")

    def enhance_prompt(self, user_concept: str) -> str:
        """Use the LLM to enhance a simple user concept into a detailed image prompt."""
        print("\n" + "-" * 70)
        print("✨ Enhancing prompt with LLM...")
        print("-" * 70)

        system_prompt = (
            "You are an expert prompt engineer for AI image generation. "
            "Take the user's concept and expand it into a detailed, vivid prompt "
            "optimized for FLUX.2 Klein image generation. "
            "Include style, lighting, composition, mood, and artistic direction. "
            "Output ONLY the enhanced prompt text, no explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Concept: {user_concept}\n\nWrite the enhanced image generation prompt:"},
        ]

        response = self.llm.generate_from_messages(
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
        """Generate image using FLUX.2 Klein."""
        print("\n" + "-" * 70)
        print("🎨 Generating image with FLUX.2 Klein...")
        print("-" * 70)

        # Update seed if specified
        params = {}
        if seed != -1:
            params["seed"] = seed

        t0 = time.time()
        image_bytes = self.tti.generate_image(
            prompt=prompt,
            negative_prompt="",
            **params,
        )
        elapsed = time.time() - t0
        print(f"✅ Image generated in {elapsed:.1f}s ({len(image_bytes)} bytes)")

        return image_bytes

    def critique_image(self, image_path: Path, original_prompt: str) -> Dict[str, Any]:
        """Use vision-capable LLM to critique the generated image."""
        print("\n" + "-" * 70)
        print("👁️  LLM is reviewing the generated image...")
        print("-" * 70)

        # Encode image for vision input
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

        # Build multimodal message
        user_content = [
            {"type": "text", "text": f"Original prompt: {original_prompt}\n\nPlease review this generated image:"},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.generate_from_messages(
            messages,
            n_predict=1024,
            temperature=0.3,  # lower temp for structured output
            top_p=0.9,
            streaming_callback=streaming_callback,
        )
        print()  # newline after streaming

        # Try to parse JSON from response
        if isinstance(response, dict) and not response.get("status", True):
            print(f"❌ Critique failed: {response.get('error')}")
            return {"approved": True, "feedback": "Critique failed, auto-approving.", "suggested_changes": "", "score": 7}

        text = response.strip() if isinstance(response, str) else ""

        # Extract JSON block if wrapped in markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            critique = json.loads(text)
            # Ensure required keys
            critique.setdefault("approved", False)
            critique.setdefault("feedback", "")
            critique.setdefault("suggested_changes", "")
            critique.setdefault("score", 5)
            return critique
        except json.JSONDecodeError:
            # Fallback: heuristic parsing
            approved = "approve" in text.lower() and "not approve" not in text.lower()
            score = 7 if approved else 5
            return {
                "approved": approved,
                "feedback": text,
                "suggested_changes": "" if approved else "Please refine based on feedback.",
                "score": score,
            }

    def edit_image(self, image_path: Path, original_prompt: str, changes: str) -> bytes:
        """Use FLUX.2 Klein's image-to-image/editing capability to refine."""
        print("\n" + "-" * 70)
        print("🔧 Editing image with FLUX.2 Klein...")
        print("-" * 70)

        # Load the image as PIL for the edit endpoint
        pil_image = Image.open(image_path).convert("RGB")

        # Build edit prompt combining original with changes
        edit_prompt = f"{original_prompt}. {changes}"

        t0 = time.time()
        image_bytes = self.tti.edit_image(
            images=pil_image,
            prompt=edit_prompt,
            strength=0.7,  # significant but not total redraw
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

            # ── Step 3: Review loop ───────────────────────────────────
            iteration = 0
            current_prompt = enhanced_prompt

            while iteration < MAX_ITERATIONS:
                critique = self.critique_image(current_path, current_prompt)

                print(f"\n📊 Critique Score: {critique['score']}/10")
                print(f"📝 Feedback: {critique['feedback']}")

                if critique["approved"] or critique["score"] >= 8:
                    print(f"\n✅ Image approved! Final saved to: {current_path}")
                    break

                if iteration >= MAX_ITERATIONS - 1:
                    print(f"\n⏹️  Max iterations reached. Saving last version: {current_path}")
                    break

                # Need changes
                changes = critique.get("suggested_changes", critique.get("feedback", "Improve quality and adherence to prompt."))
                print(f"\n🔧 Applying changes: {changes}")

                # Edit the image
                new_bytes = self.edit_image(current_path, current_prompt, changes)
                iteration += 1
                current_path = save_image(new_bytes, prefix=f"v{iteration + 1}")
                print(f"💾 Saved revision {iteration + 1}: {current_path}")

                # Update prompt for next critique context
                current_prompt = f"{current_prompt}. Refined: {changes}"

            print("\n" + "=" * 70)
            print("Describe another image (or 'quit' to exit):")

        # Cleanup
        print("\n🧹 Cleaning up...")
        self.llm.unload_model()
        self.tti.unload_model()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    assistant = CreativeAssistant()
    assistant.setup()
    assistant.run()


if __name__ == "__main__":
    main()
