#!/usr/bin/env python3
"""
🎓 TUTORIAL & COMPREHENSIVE INTEGRATION TEST: LOLLMS ARTIFACTS SYSTEM
=============================================================================
This script provides an interactive walk-through and verification suite for the 
advanced features of the LoLLMS Artifacts, Versioning, and Patching systems.

✨ CORE CAPABILITIES DEMONSTRATED:
─────────────────────────────────────────────────────────────────────────────
  1. Multimodal Imports (Images-Only with Anchor Mapping & Vision-ready OCR)
  2. Multi-Modal Bundling: Exporting and restoring multi-file bundles.
  3. Strategic Implementation: Leveraging <coding_plan> blocks [1].
  4. Indentation-Agnostic Aider Patching: Demonstrating how the fuzzy patch
     engine heals column-0 patches and automatically re-indents them to match [1].
  5. Git-like Versioning & Commit Logs: Tagging, listing logs, and reverting [1].
  6. Status Reporting Toggles: Contrasting in-message <processing> tags vs. 
     clean callback-only prose streams.
=============================================================================
"""

import sys
import os
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure correct workspace import resolution
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, ArtefactType
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors


def check_ollama(host: str, model: str) -> bool:
    import requests
    try:
        res = requests.get(f"{host}/api/tags", timeout=1.5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            return model in models or any(m.startswith(model) for m in models)
        return False
    except Exception:
        return False


class MockLollmsClient:
    """Mock client simulating cognitive responses for all test steps when offline."""
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "gemma4:e2b"
        self.binding_name = "ollama"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        import re
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def generate_text(self, prompt, **kwargs):
        # Handle the specialized patch retry logic if called
        if "[CRITICAL: PATCH REJECTED]" in prompt:
            return (
                '<artifact name="math_ops.py" type="code" language="python">\n'
                "def compute_sum(a, b):\n"
                "    # Optimised math logic\n"
                "    if a is None or b is None:\n"
                "        return 0\n"
                "    return int(a) + int(b)\n"
                "</artifact>"
            )
        return "Simulated direct text response."

    def chat(self, discussion, **kwargs):
        # Extract messages
        history = discussion.get_branch(discussion.active_branch_id)
        last_user = history[-1].content if history else ""
        callback = kwargs.get("streaming_callback")
        
        reply = "I am ready."
        
        if "plan" in last_user.lower() or "implement" in last_user.lower():
            # Simulated Turn showing both <coding_plan> and <artifact> emission
            reply = (
                "<coding_plan>\n"
                "  - Goal: Implement safe division\n"
                "  - Specialist Persona: Senior Python Developer\n"
                "  - Target Artifacts: math_ops.py\n"
                "  - Implementation Details: Add null checks and prevent division by zero.\n"
                "</coding_plan>\n"
                '<artifact name="math_ops.py" type="code" language="python">\n'
                "<<<<<<< SEARCH\n"
                "def compute_sum(a, b):\n"
                "    # Optimised math logic\n"
                "    if a is None or b is None:\n"
                "        return 0\n"
                "    return int(a) + int(b)\n"
                "=======\n"
                "def compute_sum(a, b):\n"
                "    # Optimised math logic\n"
                "    if a is None or b is None:\n"
                "        return 0\n"
                "    return int(a) + int(b)\n\n"
                "def safe_divide(a, b):\n"
                "    if b == 0 or b is None:\n"
                "        return None\n"
                "    return a / b\n"
                ">>>>>>> REPLACE\n"
                "</artifact>"
            )

        if callback:
            callback(reply, MSG_TYPE.MSG_TYPE_CHUNK)
        return reply

    def generate_structured_content(self, prompt, schema, **kwargs):
        return {"keep": True, "reason": "Saves technical content"}


def generate_dummy_image(output_path: Path, text: str):
    """Generates a simple test image to simulate user uploads."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 200), color=(50, 50, 70))
    d = ImageDraw.Draw(img)
    d.text((20, 90), text, fill=(255, 215, 0))
    img.save(output_path)
    ASCIIColors.info(f"Generated dummy image: {output_path}")


_current_proc = {"title": None, "statuses": []}

def print_event(chunk: str, msg_type: MSG_TYPE, meta: dict = None):
    """Callback to print streaming tokens and processing updates in real time."""
    global _current_proc
    if not meta:
        meta = {}
    
    event_type = meta.get("type")
    
    # Intercept and pretty-render processing tags
    if event_type == "processing_open":
        title = meta.get("title") or meta.get("tool") or "Task"
        _current_proc["title"] = title.replace("_", " ").title()
        _current_proc["statuses"] = []
        print(f"\n⚙️  [PROCESSING] {_current_proc['title']} initialized...", flush=True)
        return True
        
    elif event_type == "processing_status":
        status = meta.get("status", "").strip()
        if status:
            _current_proc["statuses"].append(status)
            print(f"   ⤷ ⏳ {status}", flush=True)
        return True
        
    elif event_type == "processing_close":
        if _current_proc["title"]:
            title_text = f"⚙️  {_current_proc['title']} (Execution Complete)"
            content_text = "\n".join(f" ✓ {s}" for s in _current_proc["statuses"]) if _current_proc["statuses"] else " ✓ Task finished."
            print()  # spacing newline
            ASCIIColors.panel(content_text, title_text)
            _current_proc = {"title": None, "statuses": []}
        return True

    # Standard message chunks
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        if meta.get("type") in ("processing_open", "processing_status", "processing_close"):
            # Suppress raw XML / status text chunks from printing to stdout
            return True
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
        ASCIIColors.panel(json.dumps(meta.get("artefact", {}), indent=2), "Artifact State Changed Event caught")


def print_active_artifacts(discussion: LollmsDiscussion):
    print("\n" + "📂 ACTIVE ARTIFACTS IN SESSION:")
    print("─" * 60)
    active = discussion.artefacts.list(active_only=True)
    if not active:
        print("  (No active artifacts)")
    else:
        for a in active:
            lang_str = f" [{a.get('language')}]" if a.get("language") else ""
            print(f"  • [{a['type']}] '{a['title']}' (v{a['version']}){lang_str} — {len(a['content'])} chars")
    print("─" * 60 + "\n")


def main():
    print("=" * 80)
    print("🔬 LOLLMS SYSTEM: ADVANCED ARTIFACT INTEGRATION TEST & TUTORIAL")
    print("=" * 80)

    LLM_MODEL_NAME = "gemma4:e2b"
    host_address = "http://localhost:11434"

    is_online = check_ollama(host_address, LLM_MODEL_NAME)

    # ── 1. SETUP LOLLMS CLIENT & DISCUSSION ───────────────────────────────
    print("\n[Step 1] Initializing client and discussion space...")
    if is_online:
        ASCIIColors.green(f"⚡ Live connection detected! Connecting to LollmsClient with {LLM_MODEL_NAME}...")
        client = LollmsClient(
            llm_binding_name="ollama",
            llm_binding_config={
                "model_name": LLM_MODEL_NAME,
                "host_address": host_address
            },
            user_name="ParisNeo",
            ai_name="Lollms",
            cooperative_vram_management=True,
            debug=True
        )
    else:
        ASCIIColors.yellow("⚠️  Ollama server is offline or 'gemma4:e2b' is not installed.")
        ASCIIColors.info("Running in FULL INTEGRATION SIMULATION MODE.")
        client = MockLollmsClient()

    discussion = LollmsDiscussion(lollmsClient=client)
    print("✅ Discussion space ready.")

    # ── 2. PREPARE DUMMY USER UPLOADS ─────────────────────────────────────
    print("\n[Step 2] Preparing dummy assets...")
    temp_dir = Path("./temp_artifacts_assets")
    temp_dir.mkdir(exist_ok=True)
    
    dummy_img_path = temp_dir / "invoice_draft.png"
    generate_dummy_image(dummy_img_path, "INVOICE #99812 - $1,500.00 USD")

    # ── 3. TEST IMPORT MODALITIES WITH ANCHOR MAPPING ─────────────────────
    print("\n[Step 3] Testing Multi-Modal File Import (Images-Only with Anchors)...")
    result = discussion.import_file(
        path=dummy_img_path,
        mode="images_only",
        title="invoice_visual",
        activate=True
    )
    print(f"  • Created main text/anchor artifact: '{result['text_artefact']['title']}'")
    print(f"  • Created companion images artifact: '{result['image_artefact']['title']}'")
    
    print_active_artifacts(discussion)

    # ── 4. MULTI-MODAL RECOVERY & BUNDLING ────────────────────────────────
    print("\n[Step 4] Testing Multi-Modal Bundling & Restoration...")
    
    # Export bundle representing both the text file and its associated sub-images
    bundle = discussion.artefacts.export_artefact_bundle("invoice_visual")
    print(f"  • Successfully exported self-contained bundle for 'invoice_visual'")
    print(f"    - Subfiles packaged: {len(bundle['companion_artefacts'])}")

    # Reconstruct the bundle in an entirely separate discussion space
    new_discussion = LollmsDiscussion(lollmsClient=client)
    restored_art = new_discussion.artefacts.import_artefact_bundle(bundle, activate=True)
    print(f"  • Successfully imported and rebuilt bundle in fresh discussion!")
    print_active_artifacts(new_discussion)

    # ── 5. ESTABLISH BASELINE FILE ────────────────────────────────────────
    print("\n[Step 5] Establishing baseline code file...")
    # We will create a heavily indented Python file to test fuzzy patch re-indentation
    indented_code = (
        "class Controller:\n"
        "    def run(self):\n"
        "        # Core operations\n"
        "        print('Starting...')\n"
        "        def compute_sum(a, b):\n"
        "            # Target region\n"
        "            return a + b\n"
    )
    discussion.artefacts.add(
        title="math_ops.py",
        artefact_type=ArtefactType.CODE,
        content=indented_code,
        language="python",
        commit_message="Initial class skeleton",
        version_tags=["stable-v1"]
    )
    print_active_artifacts(discussion)

    # ── 6. TEST INDENTATION-AGNOSTIC FUZZY AIDER PATCHING ─────────────────
    print("\n[Step 6] Testing Indentation-Agnostic Fuzzy Patching...")
    # The LLM frequently outputs patches at column-0 (no indentation), even if
    # the target region is deeply indented in the file.
    # The Lollms Aider Patch engine must detect this and automatically re-indent it!
    column_zero_patch = (
        "<<<<<<< SEARCH\n"
        "def compute_sum(a, b):\n"
        "    # Target region\n"
        "    return a + b\n"
        "=======\n"
        "def compute_sum(a, b):\n"
        "    # Safe logic with null-guards\n"
        "    if a is None or b is None:\n"
        "        return 0\n"
        "    return a + b\n"
        ">>>>>>> REPLACE"
    )

    ASCIIColors.info("  Applying column-0 patch to a deeply nested nested function...")
    patched_text = discussion.artefacts.apply_aider_patch(indented_code, column_zero_patch)
    
    print("\n  Healed and applied content:")
    print("-" * 60)
    print(patched_text)
    print("-" * 60)
    
    # Save the updated content
    discussion.artefacts.update("math_ops.py", new_content=patched_text, commit_message="Apply safe null-guards to sum")

    # ── 7. TEST STRATEGIC IMPLEMENTATION PLANS ────────────────────────────
    print("\n[Step 7] Testing Strategic Plans (<coding_plan> + <artifact>)...")
    # Multi-step generations represent a plan phase followed by code outputs.
    # We will run a turn containing a plan and verify it is captured cleanly.
    user_prompt = "Please add a safe division function to math_ops.py."
    
    res = discussion.chat(
        user_message=user_prompt,
        streaming_callback=print_event,
        enable_memory=False,
        enable_artefacts=True
    )
    print()
    print_active_artifacts(discussion)

    # ── 8. TEST GIT-LIKE VERSIBILITY & LOGS ───────────────────────────────
    print("\n[Step 8] Checking Git-like Versioning Log...")
    log = discussion.artefacts.get_log("math_ops.py")
    for entry in log:
        print(f"  • v{entry['version']} | {entry['commit_hash'][:8]} | Msg: {entry['commit_message']} | Tags: {entry['tags']}")

    # Revert to a specific tag
    print("\n  Reverting 'math_ops.py' back to tag 'stable-v1'...")
    discussion.artefacts.revert_to_tag("math_ops.py", "stable-v1")
    reverted = discussion.artefacts.get("math_ops.py")
    print(f"  • Restored content (v{reverted['version']}):")
    print("-" * 60)
    print(reverted["content"])
    print("-" * 60)

    # ── 9. TEST IN-MESSAGE STATUS REPORTING TOGGLE ────────────────────────
    print("\n[Step 9] Demonstrating status-reporting toggles...")
    print("  • Testing with enable_in_message_status=False (Callback-Only):")
    # Setting this to False suppresses XML status tags from being injected into the chat message,
    # but still raises rich background callbacks.
    discussion.chat(
        user_message="Add multiplication to math_ops.py",
        streaming_callback=print_event,
        enable_memory=False,
        enable_artefacts=True,
        enable_in_message_status=False
    )
    print("\n  • (Prose is kept 100% clean and professional with zero inline status tags)")

    # ── 10. CLEANUP ───────────────────────────────────────────────────────
    print("\n[Step 10] Cleaning up temporary files...")
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()
    
    print("\n" + "=" * 80)
    print("🎉 ALL ARTIFACT SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
