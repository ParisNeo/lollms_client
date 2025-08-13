# lollms_client/tti_bindings/lollms/__init__.py
import requests
import base64
import io
from PIL import Image
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from typing import Optional, List, Dict, Any
from ascii_colors import trace_exception, ASCIIColors
import json # Added for potential error parsing

# Defines the binding name for the manager
BindingName = "LollmsWebuiTTIBinding_Impl"
class LollmsWebuiTTIBinding_Impl(LollmsTTIBinding):
    """Concrete implementation of the LollmsTTIBinding for the standard LOLLMS server."""

    def __init__(self,
                 **kwargs):
        """
        Initialize the LOLLMS TTI binding.

        Args:
            host_address (Optional[str]): Host address for the LOLLMS service.
            service_key (Optional[str]): Authentication key (used for client_id verification).
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        super().__init__(binding_name="lollms")

        # Extract parameters from kwargs, providing defaults
        self.host_address = kwargs.get("host_address", "http://localhost:9600")  # Default LOLLMS host
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)

        # The 'service_key' here will act as the 'client_id' for TTI requests if provided.
        # This assumes the client library user provides their LOLLMS client_id here.
        self.client_id = kwargs.get("service_key", None) # Use service_key or None

    def _get_client_id(self, **kwargs) -> str:
        """Helper to get client_id, prioritizing kwargs then instance default."""
        client_id = kwargs.get("client_id", self.client_id)
        if not client_id:
             # Allowing anonymous access for generation, but other endpoints might fail
             # raise ValueError("client_id is required for this TTI operation but was not provided.")
             ASCIIColors.warning("client_id not provided for TTI operation. Some features might require it.")
             return "lollms_client_user" # Default anonymous ID
        return client_id

    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 512,
                       height: int = 512,
                       **kwargs) -> bytes:
        """
        Generates image data using the LOLLMS /generate_image endpoint.

        Args:
            prompt (str): The positive text prompt.
            negative_prompt (Optional[str]): The negative prompt.
            width (int): Image width.
            height (int): Image height.
            **kwargs: Additional parameters (passed to server if needed, currently unused by endpoint).

        Returns:
            bytes: The generated image data (JPEG format).

        Raises:
            Exception: If the request fails or image generation fails on the server.
        """
        endpoint = f"{self.host_address}/generate_image"
        request_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height
        }
        # Include extra kwargs if the server API might use them in the future
        request_data.update(kwargs)

        headers = {'Content-Type': 'application/json'}
        # Note: /generate_image endpoint doesn't currently require client_id based on provided code

        try:
            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()

            response_json = response.json()
            img_base64 = response_json.get("image")

            if not img_base64:
                raise Exception("Server did not return image data.")

            # Decode the base64 string to bytes
            img_bytes = base64.b64decode(img_base64)
            return img_bytes

        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            # Attempt to get error detail from response if possible
            error_detail = "Unknown server error"
            try:
                error_detail = response.json().get("detail", error_detail)
            except: pass
            raise Exception(f"Image generation failed: {e} - Detail: {error_detail}") from e


    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """
        Lists available TTI services using the LOLLMS /list_tti_services endpoint.

        Args:
            **kwargs: Must include 'client_id' for server verification.

        Returns:
            List[Dict[str, str]]: List of service dictionaries.

        Raises:
            ValueError: If client_id is not provided.
            Exception: If the request fails.
        """
        endpoint = f"{self.host_address}/list_tti_services"
        client_id = self._get_client_id(**kwargs) # Raises ValueError if missing

        request_data = {"client_id": client_id}
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            return response.json() # Returns the list directly
        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to list TTI services: {e}") from e

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Retrieves TTI settings using the LOLLMS /get_active_tti_settings endpoint.

        Args:
            **kwargs: Must include 'client_id' for server verification.

        Returns:
            Optional[Dict[str, Any]]: Settings dictionary or empty dict if not available/failed.

        Raises:
            ValueError: If client_id is not provided.
            Exception: If the request fails.
        """
        endpoint = f"{self.host_address}/get_active_tti_settings"
        client_id = self._get_client_id(**kwargs) # Raises ValueError if missing

        request_data = {"client_id": client_id}
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            settings = response.json()
            # The endpoint returns the template directly if successful, or {} if no TTI/settings
            return settings if isinstance(settings, list) else {} # Ensure correct format or empty
        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to get TTI settings: {e}") from e


    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        """
        Applies TTI settings using the LOLLMS /set_active_tti_settings endpoint.

        Args:
            settings (Dict[str, Any]): The new settings (matching server's expected format).
            **kwargs: Must include 'client_id' for server verification.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            ValueError: If client_id is not provided.
            Exception: If the request fails.
        """
        endpoint = f"{self.host_address}/set_active_tti_settings"
        client_id = self._get_client_id(**kwargs) # Raises ValueError if missing

        request_data = {
            "client_id": client_id,
            "settings": settings
            }
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(endpoint, json=request_data, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("status", False)
        except requests.exceptions.RequestException as e:
            trace_exception(e)
            raise Exception(f"HTTP request failed: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to set TTI settings: {e}") from e