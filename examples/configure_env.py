#!/usr/bin/env python3
"""
configure_env.py
================
A one-shot configuration wizard that scans the lollms_client bindings
and writes a .env file for the examples.

Usage:
    python examples/configure_env.py
"""

import os
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"

def main():
    print("=" * 60)
    print("🧙‍♂️ LoLLMS Examples Configuration Wizard")
    print("=" * 60)
    
    config = {}
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        k, v = line.strip().split("=", 1)
                        config[k] = v
        except Exception:
            pass

    # --- LLM Configuration ---
    print("\n--- LLM Configuration ---")
    llm_binding = input(f"Enter LLM Binding Name [{config.get('LLM_BINDING_NAME', 'ollama')}]: ").strip() or config.get("LLM_BINDING_NAME", "ollama")
    config["LLM_BINDING_NAME"] = llm_binding

    default_model = "llama3"
    if llm_binding == "lollms":
        default_model = "Kimi-K 2.5"
    elif llm_binding == "openai":
        default_model = "gpt-4o-mini"

    model_name = input(f"Enter Model Name [{config.get('MODEL_NAME', default_model)}]: ").strip() or config.get("MODEL_NAME", default_model)
    config["MODEL_NAME"] = model_name

    default_host = "http://localhost:11434"
    if llm_binding in ["lollms", "open_router", "vllm"]:
        default_host = "http://localhost:9642" if llm_binding == "lollms" else "https://openrouter.ai/api/v1"

    host_address = input(f"Enter Host Address [{config.get('HOST_ADDRESS', default_host)}]: ").strip() or config.get("HOST_ADDRESS", default_host)
    config["HOST_ADDRESS"] = host_address

    verify_ssl = input(f"Verify SSL? (true/false) [{config.get('VERIFY_SSL', 'false')}]: ").strip().lower() or config.get("VERIFY_SSL", "false")
    config["VERIFY_SSL"] = verify_ssl

    api_key = input(f"Enter API Key (leave blank if none) [{'Set' if config.get('API_KEY') else 'None'}]: ").strip()
    if api_key:
        config["API_KEY"] = api_key
    elif "API_KEY" not in config:
        config["API_KEY"] = ""

    # --- TTI Configuration (Optional) ---
    print("\n--- TTI Configuration (Optional) ---")
    setup_tti = input("Do you want to configure Text-to-Image (TTI)? (y/N): ").strip().lower()
    if setup_tti == 'y':
        tti_binding = input(f"Enter TTI Binding Name [{config.get('TTI_BINDING_NAME', 'diffusers')}]: ").strip() or config.get("TTI_BINDING_NAME", "diffusers")
        config["TTI_BINDING_NAME"] = tti_binding

        default_tti_model = "stabilityai/sdxl-turbo" if tti_binding == "diffusers" else "dall-e-3"
        tti_model = input(f"Enter TTI Model Name [{config.get('TTI_MODEL_NAME', default_tti_model)}]: ").strip() or config.get("TTI_MODEL_NAME", default_tti_model)
        config["TTI_MODEL_NAME"] = tti_model
    else:
        config["TTI_BINDING_NAME"] = ""
        config["TTI_MODEL_NAME"] = ""

    # --- Write to .env ---
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write("# LoLLMS Examples Configuration\n")
            for k, v in config.items():
                f.write(f"{k}={v}\n")
        print(f"\n✅ Configuration saved successfully to {ENV_FILE}!")
    except Exception as e:
        print(f"\n❌ Error saving .env file: {e}")

if __name__ == "__main__":
    main()