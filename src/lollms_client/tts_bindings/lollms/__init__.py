# lollms_client/tts_bindings/lollms/__init__.py
import requests
from lollms_client.lollms_tts_binding import LollmsTTSBinding
from typing import Optional, List
from ascii_colors import trace_exception, ASCIIColors
import json

# Defines the binding name for the manager
BindingName = "LollmsTTSBinding_Impl"

class LollmsTTSBinding_Impl(LollmsTTSBinding):
    """Concrete implementation of the LollmsTTSBinding for the standard LOLLMS server."""

    def __init__(self,
                 host_address: Optional[str] = "http://localhost:9600",
                 model_name: Optional[str] = "main_voice",
                 service_key: Optional[str] = None, # This will be used as client_id
                 verify_ssl_certificate: bool = True):
        """
        Initialize the LOLLMS TTS binding.

        Args:
            host_address (Optional[str]): Host address for the LOLLMS service.
            model_name (Optional[str]): Default voice/model name to use.
            service_key (Optional[str]): Authentication key, used as client_id for LOLLMS server.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        super().__init__(host_address=host_address,
                         model_name=model_name,
                         service_key=service_key, # Stored in the parent class
                         verify_ssl_certificate=verify_ssl_certificate)
        self.host_address = host_address
        # self.client_id = service_key # Can access via self.service_key from parent

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """
        Generates audio data from text using the LOLLMS /text2Audio endpoint.

        Args:
            text (str): The text content to synthesize.
            voice (Optional[str]): The specific voice to use. If None, uses the default from init.
            **kwargs: Accepts 'fn' (filename) for server-side saving,
                      and 'client_id' to override the instance's default client_id.

        Returns:
            bytes: Returns an empty bytes string as the server saves the file and returns status JSON.

        Raises:
            Exception: If the request fails or the server returns an error status.
            ValueError: If client_id is not available.
        """
        endpoint = f"{self.host_address}/text2Audio"
        voice_to_use = voice if voice else self.model_name
        filename_on_server = kwargs.get("fn")
        
        # Determine client_id: use from kwargs if provided, otherwise from instance's service_key
        client_id_to_use = kwargs.get("client_id", self.service_key)
        if not client_id_to_use:
            # Fallback or raise error if client_id is strictly required by the server
            # For /text2Audio, it is strictly required.
            raise ValueError("client_id is required for text2Audio but was not provided during LollmsClient initialization (as service_key) or in this call.")


        request_data = {
            "client_id": client_id_to_use, # ADDED client_id
            "text": text,
            "voice": voice_to_use,
            "fn": filename_on_server
        }
        # Filter out None values for 'voice' and 'fn' if the server doesn't expect them or handles defaults
        # client_id and text are mandatory for the server model LollmsText2AudioRequest
        if voice_to_use is None:
            del request_data["voice"]
        if filename_on_server is None:
            del request_data["fn"]


        headers = {'Content-Type': 'application/json'}
        # service_key (if different from client_id for auth header) isn't used by this endpoint's auth
        # The check_access on server side uses the client_id from the payload.

        try:
            ASCIIColors.debug(f"Sending TTS request to {endpoint} with payload: {request_data}")
            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            
            response_json = response.json()
            if response_json.get("status") is False or "error" in response_json: # Check for explicit error
                 raise Exception(f"Server returned error: {response_json.get('error', 'Unknown error')}")

            ASCIIColors.info(f"Audio generation requested. Server response: {response_json}")
            return b""

        except requests.exceptions.HTTPError as e:
            # Log the response content for 422 errors specifically
            if e.response is not None and e.response.status_code == 422:
                try:
                    error_details = e.response.json()
                    ASCIIColors.error(f"Unprocessable Entity. Server details: {error_details}")
                except json.JSONDecodeError:
                    ASCIIColors.error(f"Unprocessable Entity. Server response: {e.response.text}")
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e # Re-raise the original HTTPError
        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request connection failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Audio generation failed: {e}") from e


    def list_voices(self, **kwargs) -> List[str]:
        """
        Lists the available voices using the LOLLMS /list_voices endpoint.
        This endpoint does not require client_id in the request body based on server code.

        Args:
            **kwargs: Additional parameters (currently unused).

        Returns:
            List[str]: A list of available voice identifiers. Returns ["main_voice"] on failure.
        """
        endpoint = f"{self.host_address}/list_voices"
        headers = {'Content-Type': 'application/json'}
        # No client_id needed in payload for this specific GET endpoint on the server

        try:
            response = requests.get(endpoint, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            voices_data = response.json()
            if "error" in voices_data or voices_data.get("status") is False:
                 ASCIIColors.error(f"Error listing voices from server: {voices_data.get('error', 'Unknown server error')}")
                 return ["main_voice"]
            return voices_data.get("voices", ["main_voice"])
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Couldn't list voices due to connection error: {e}")
            trace_exception(e)
            return ["main_voice"]
        except json.JSONDecodeError as e:
            ASCIIColors.error(f"Couldn't parse voices response from server: {e}")
            trace_exception(e)
            return ["main_voice"]
        except Exception as e:
            ASCIIColors.error(f"An unexpected error occurred while listing voices: {e}")
            trace_exception(e)
            return ["main_voice"]

    def list_models(self) -> list:
        """Lists models"""
        return  ["lollms"]