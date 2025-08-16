# lollms_client/stt_bindings/lollms/__init__.py
import requests
import base64
from pathlib import Path
from lollms_client.lollms_stt_binding import LollmsSTTBinding
from typing import Optional, List, Union
from ascii_colors import trace_exception, ASCIIColors
import json # Added for potential error parsing

# Defines the binding name for the manager
BindingName = "LollmsSTTBinding_Impl"

class LollmsSTTBinding_Impl(LollmsSTTBinding):
    """Concrete implementation of the LollmsSTTBinding for the standard LOLLMS server."""

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the LOLLMS STT binding.

        Args:
            host_address (Optional[str]): Host address for the LOLLMS service.
            model_name (Optional[str]): Default STT model identifier.
            service_key (Optional[str]): Authentication key (currently unused by default LOLLMS STT).
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        super().__init__("lollms")
        self.host_address=kwargs.get("host_address")
        self.model_name=kwargs.get("model_name")
        self.service_key=kwargs.get("service_key")
        self.verify_ssl_certificate=kwargs.get("verify_ssl_certificate")

    def transcribe_audio(self, audio_path: Union[str, Path], model: Optional[str] = None, **kwargs) -> str:
        """
        Transcribes audio using an assumed LOLLMS /audio2text endpoint.
        Sends audio data as base64 encoded string.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            model (Optional[str]): Specific STT model to use. If None, uses default from init.
            **kwargs: Additional parameters (e.g., language hint - passed if provided).

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: If the request fails or transcription fails on the server.
        """
        endpoint = f"{self.host_address}/audio2text" # Assumed endpoint
        model_to_use = model if model else self.model_name

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        try:
            # Read audio file and encode as base64
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            request_data = {
                "audio_data": audio_base64, # Sending data instead of path
                "model": model_to_use,
                "file_name": audio_file.name # Send filename as metadata if server supports it
            }
            # Add language hint if provided in kwargs
            if "language" in kwargs:
                 request_data["language"] = kwargs["language"]

            # Filter out None values if server requires it
            request_data = {k: v for k, v in request_data.items() if v is not None}

            headers = {'Content-Type': 'application/json'}
            if self.service_key:
                headers['Authorization'] = f'Bearer {self.service_key}'

            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()

            response_json = response.json()
            transcribed_text = response_json.get("text")

            if transcribed_text is None:
                 # Check for error message from server
                 error_msg = response_json.get("error", "Server did not return transcribed text.")
                 raise Exception(error_msg)

            return transcribed_text

        except FileNotFoundError as e:
            raise e # Re-raise file not found
        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Audio transcription failed: {e}") from e


    def list_models(self, **kwargs) -> List[str]:
        """
        Lists available STT models using an assumed LOLLMS /list_stt_models endpoint.

        Args:
            **kwargs: Additional parameters (currently unused).

        Returns:
            List[str]: List of STT model identifiers. Returns empty list on failure.
        """
        endpoint = f"{self.host_address}/list_stt_models" # Assumed endpoint
        headers = {'Content-Type': 'application/json'}
        if self.service_key:
            headers['Authorization'] = f'Bearer {self.service_key}'

        try:
            response = requests.get(endpoint, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            models_data = response.json()
            if "error" in models_data:
                 ASCIIColors.error(f"Error listing STT models from server: {models_data['error']}")
                 return [] # Fallback
            return models_data.get("models", []) # Default if key missing
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Couldn't list STT models due to connection error: {e}")
            trace_exception(e)
            return [] # Fallback
        except json.JSONDecodeError as e:
            ASCIIColors.error(f"Couldn't parse STT models response from server: {e}")
            trace_exception(e)
            return [] # Fallback
        except Exception as e:
            ASCIIColors.error(f"An unexpected error occurred while listing STT models: {e}")
            trace_exception(e)
            return [] # Fallback