"""
Automated tests for the LoLLMS Examples Configuration Wizard.

`main()` is interactive (driven by `input()`), so every test scripts its answers
via `unittest.mock.patch("builtins.input", side_effect=[...])` and points
`configure_env.ENV_FILE` at a temp file so the real `examples/.env` is never
touched.

Prompt sequence in `main()` (verified against examples/configure_env.py):
  1. Enter LLM Binding Name
  2. Enter Model Name
  3. Enter Host Address
  4. Verify SSL?
  5. Enter API Key
  6. Do you want to configure Text-to-Image (TTI)? (y/N)
  7-8. [if TTI == 'y'] TTI Binding Name, TTI Model Name
  9. Do you want to configure vision/video support? (y/N)
  10-12. [if Vision == 'y'] image input?, video input?, Tokens per image

Answer-count matrix:
  TTI=n, Vision=n -> 7
  TTI=n, Vision=y -> 10
  TTI=y, Vision=n -> 9
  TTI=y, Vision=y -> 12
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import configure_env  # noqa: E402


def _read_env(path: Path) -> dict:
    """Parse a KEY=VALUE .env file into a dict (skipping comments/blank lines)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
    return out


def test_configure_env_writes_all_keys_baseline():
    """Declining TTI and Vision still writes all 7 base + 3 capability keys."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        # LLM(5) + TTI one-liner(1) + Vision one-liner(1) = 7 prompts
        answers = ["ollama", "llama3", "http://localhost:11434", "false", "", "n", "n"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=answers):
            configure_env.main()

        data = _read_env(env_path)
        assert data["LLM_BINDING_NAME"] == "ollama"
        assert data["MODEL_NAME"] == "llama3"
        assert data["HOST_ADDRESS"] == "http://localhost:11434"
        assert data["VERIFY_SSL"] == "false"
        assert data["API_KEY"] == ""
        assert data["TTI_BINDING_NAME"] == ""
        assert data["TTI_MODEL_NAME"] == ""
        assert data["VISION_ENABLED"] == "false"
        assert data["VIDEO_ENABLED"] == "false"
        assert data["IMAGE_TOKEN_COST"] == "256"


def test_configure_env_vision_optin_writes_custom_values():
    """Opting into vision writes user-provided true/true/512."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        # LLM(5) + TTI="n"(1) + Vision y/N="y"(1) + "true" + "true" + "512"(3) = 10
        answers = ["ollama", "llama3", "http://localhost:11434", "false", "",
                   "n", "y", "true", "true", "512"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=answers):
            configure_env.main()

        data = _read_env(env_path)
        assert data["VISION_ENABLED"] == "true"
        assert data["VIDEO_ENABLED"] == "true"
        assert data["IMAGE_TOKEN_COST"] == "512"


def test_configure_env_invalid_image_cost_falls_back_to_256():
    """Non-integer image cost triggers the wizard's fallback to 256."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        answers = ["ollama", "llama3", "http://localhost:11434", "false", "",
                   "n", "y", "true", "true", "not_an_int"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=answers):
            configure_env.main()

        data = _read_env(env_path)
        assert data["IMAGE_TOKEN_COST"] == "256"


def test_configure_env_tti_optin_writes_tti_keys():
    """Opting into TTI writes binding + model, Vision declined keeps defaults."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        # LLM(5) + TTI="y"(1) + TTI binding + model(2) + Vision one-liner(1) = 9
        answers = ["ollama", "llama3", "http://localhost:11434", "false", "",
                   "y", "diffusers", "sdxl-turbo", "n"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=answers):
            configure_env.main()

        data = _read_env(env_path)
        assert data["TTI_BINDING_NAME"] == "diffusers"
        assert data["TTI_MODEL_NAME"] == "sdxl-turbo"
        assert data["VISION_ENABLED"] == "false"
        assert data["VIDEO_ENABLED"] == "false"
        assert data["IMAGE_TOKEN_COST"] == "256"


def test_configure_env_is_idempotent_preserves_prior_answers():
    """Rerunning against an existing .env preserves prior values via defaults."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        # First run — user opts into vision with cost 1024
        first_answers = ["ollama", "llama3", "http://localhost:11434", "false", "",
                         "n", "y", "true", "true", "1024"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=first_answers):
            configure_env.main()

        assert _read_env(env_path)["IMAGE_TOKEN_COST"] == "1024"

        # Second run — all blank answers → defaults come from existing .env
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", return_value=""):
            configure_env.main()

        data = _read_env(env_path)
        # LLM values preserved from prior run
        assert data["LLM_BINDING_NAME"] == "ollama"
        assert data["MODEL_NAME"] == "llama3"
        # Vision capability preserved (else-branch uses config.get)
        assert data["VISION_ENABLED"] == "true"
        assert data["VIDEO_ENABLED"] == "true"
        assert data["IMAGE_TOKEN_COST"] == "1024"


def test_configure_env_fresh_dir_creates_env_file():
    """Fresh temp dir (no prior .env) still produces a valid file."""
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / "subdir" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        answers = ["ollama", "llama3", "http://localhost:11434", "false", "", "n", "n"]
        with patch.object(configure_env, "ENV_FILE", env_path), \
             patch("builtins.input", side_effect=answers):
            configure_env.main()

        assert env_path.exists()
        assert "# LoLLMS Examples Configuration" in env_path.read_text(encoding="utf-8")