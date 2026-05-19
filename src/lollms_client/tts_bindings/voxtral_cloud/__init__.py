import os
import sys
import requests
import time
from pathlib import Path
from typing import Optional, List

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "MistralTTSClientBinding"

MISTRAL_TTS_ENDPOINT = "https://api.mistral.ai/v1/audio/speech"

# Voices supported by Voxtral TTS (as of March 2026)
MISTRAL_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam",  "am_michael",
    "bf_emma",  "bf_isabella",
    "bm_george", "bm_lewis",
]

MISTRAL_TTS_MODELS = [
    "voxtral-tts-2603",
]


class MistralTTSClientBinding(LollmsTTSBinding):
    """
    LoLLMs TTS binding for Mistral's Voxtral TTS cloud API.

    Unlike XTTS this is a *thin* client: there is no local server to manage.
    Every synthesis request hits https://api.mistral.ai/v1/audio/speech directly.
    Voice cloning is not supported; voices are chosen from the fixed Mistral catalogue.
    """

    def __init__(self, **kwargs):
        # Accept "model" as alias for "model_name"
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        self.config = kwargs
        self.api_key: str = kwargs.get("api_key", os.environ.get("MISTRAL_API_KEY", ""))
        self.model_name: str = kwargs.get("model_name", MISTRAL_TTS_MODELS[0])
        self.default_voice: str = kwargs.get("default_voice", "af_bella")
        self.response_format: str = kwargs.get("response_format", "wav")
        self.base_url: str = kwargs.get("base_url", MISTRAL_TTS_ENDPOINT)

        if not self.api_key:
            ASCIIColors.warning(
                "MistralTTSBinding: No API key provided. "
                "Set 'api_key' in config or the MISTRAL_API_KEY environment variable."
            )

    # ------------------------------------------------------------------ #
    #  Core interface                                                      #
    # ------------------------------------------------------------------ #

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs,
    ) -> bytes:
        """
        Generate speech from *text* and return raw audio bytes (WAV by default).

        Args:
            text:     The text to synthesise.
            voice:    One of the Mistral voice IDs (e.g. "af_bella").
                      Falls back to self.default_voice when not specified.
            language: Language hint – forwarded to the API where supported.
            **kwargs: Additional parameters forwarded verbatim to the API
                      (e.g. speed, stream).

        Returns:
            Raw audio bytes in the format specified by self.response_format.
        """
        if not self.api_key:
            raise RuntimeError(
                "MistralTTSBinding: No API key configured. "
                "Pass api_key= in kwargs or set MISTRAL_API_KEY."
            )

        selected_voice = voice or self.default_voice
        if selected_voice not in MISTRAL_VOICES:
            ASCIIColors.warning(
                f"Voice '{selected_voice}' is not in the known Mistral voice list. "
                f"Falling back to '{self.default_voice}'."
            )
            selected_voice = self.default_voice

        payload = {
            "model":           self.model_name,
            "input":           text,
            "voice":           selected_voice,
            "response_format": self.response_format,
        }
        # Allow callers to override / extend (e.g. speed=1.2)
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        ASCIIColors.info(
            f"MistralTTS: Generating audio | model={self.model_name} "
            f"voice={selected_voice} chars={len(text)}"
        )

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            detail = ""
            try:
                detail = e.response.json()
            except Exception:
                pass
            ASCIIColors.error(f"MistralTTS: HTTP {status} — {detail}")
            raise RuntimeError(f"Mistral TTS API error ({status}): {detail}") from e
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"MistralTTS: Network error — {e}")
            raise RuntimeError("Mistral TTS network error.") from e

        audio_bytes = response.content
        ASCIIColors.green(f"MistralTTS: Received {len(audio_bytes):,} bytes of audio.")
        return audio_bytes

    def list_voices(self, **kwargs) -> List[str]:
        """Return the static list of Mistral Voxtral voices."""
        return list(MISTRAL_VOICES)

    def list_models(self, **kwargs) -> List[str]:
        """Return the supported Mistral TTS model identifiers."""
        return list(MISTRAL_TTS_MODELS)
