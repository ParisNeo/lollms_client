# bindings/lollms/binding.py
import requests
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from ascii_colors import ASCIIColors, trace_exception
from typing import Optional, Callable, List, Union
import json

BindingName = "LollmsLLMBinding"


class LollmsLLMBinding(LollmsLLMBinding):
    """LOLLMS-specific binding implementation"""
    
    DEFAULT_HOST_ADDRESS = "http://localhost:9600"
    
    def __init__(self, 
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 personality: Optional[int] = None, 
                 **kwargs
                 ):
        """
        Initialize the LOLLMS binding.

        Args:
            host_address (str): Host address for the LOLLMS service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Personality ID for generation. Defaults to None.
        """
        super().__init__(
            binding_name = "lollms"
        )
        
        self.host_address=host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS
        self.model_name=model_name
        self.service_key=service_key
        self.verify_ssl_certificate=verify_ssl_certificate
        self.default_completion_format=kwargs.get("default_completion_format",ELF_COMPLETION_FORMAT.Chat) 
        self.personality = personality
        self.model = None
    
    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: bool = False,
                     temperature: float = 0.1,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     repeat_penalty: float = 0.8,
                     repeat_last_n: int = 40,
                     seed: Optional[int] = None,
                     n_threads: int = 8,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, str], None]] = None) -> Union[str, dict]:
        """
        Generate text using the LOLLMS service, with optional image support.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
                If provided, uses the /lollms_generate_with_images endpoint.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.1.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p sampling parameter. Defaults to 0.95.
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40.
            seed (Optional[int]): Random seed for generation.
            n_threads (int): Number of threads to use. Defaults to 8.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback for streaming output.
                - First parameter (str): The chunk of text received from the stream.
                - Second parameter (str): The message type (typically MSG_TYPE.MSG_TYPE_CHUNK).

        Returns:
            Union[str, dict]: Generated text if successful, or a dictionary with status and error if failed.
        """
        # Determine endpoint based on presence of images
        endpoint = "/lollms_generate_with_images" if images else "/lollms_generate"
        url = f"{self.host_address}{endpoint}"
        
        # Set headers
        headers = {
            'Content-Type': 'application/json',
        }
        if self.service_key:
            headers['Authorization'] = f'Bearer {self.service_key}'

        # Handle images if provided
        image_data = []
        if images:
            for image_path in images:
                try:
                    encoded_image = encode_image(image_path)
                    image_data.append(encoded_image)
                except Exception as e:
                    return {"status": False, "error": f"Failed to process image {image_path}: {str(e)}"}
        
        # Prepare request data
        data = {
            "prompt":"!@>system: "+system_prompt+"\n"+"!@>user: "+prompt if system_prompt else prompt,
            "model_name": self.model_name,
            "personality": self.personality,
            "n_predict": n_predict,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "seed": seed,
            "n_threads": n_threads
        }
        
        if image_data:
            data["images"] = image_data

        # Make the request
        response = requests.post(
            url, 
            json=data, 
            headers=headers, 
            stream=stream,
            verify=self.verify_ssl_certificate
        )
        
        if not stream:
            if response.status_code == 200:
                try:
                    text = response.text.strip()
                    return text
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            else:
                return {"status": False, "error": response.text}
        else:
            text = ""
            if response.status_code == 200:
                try:
                    for line in response.iter_lines():
                        chunk = line.decode("utf-8")
                        text += chunk
                        if streaming_callback:
                            streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK)
                    # Handle potential quotes from streaming response
                    if text and text[0] == '"':
                        text = text[1:]
                    if text and text[-1] == '"':
                        text = text[:-1]
                    return text.rstrip('!')
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            else:
                return {"status": False, "error": response.text}
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens using the /lollms_tokenize endpoint.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        try:
            # Prepare the request payload
            payload = {
                "prompt": text,
                "return_named": False  # Set to True if you want named tokens
            }
            
            # Make the POST request to the /lollms_tokenize endpoint
            response = requests.post(f"{self.host_address}/lollms_tokenize", json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to tokenize text: {response.text}")
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Failed to tokenize text: {response.text}")
          
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text using the /lollms_detokenize endpoint.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        try:
            # Prepare the request payload
            payload = {
                "tokens": tokens,
                "return_named": False  # Set to True if you want named tokens
            }

            # Make the POST request to the /lollms_detokenize endpoint
            response = requests.post(f"{self.host_address}/lollms_detokenize", json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to detokenize tokens: {response.text}")
        except Exception as ex:
            return {"status": False, "error": str(ex)}

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """        
        return len(self.tokenize(text))
    
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        
        Args:
            text (str or List[str]): Input text to embed
            **kwargs: Additional arguments like model, truncate, options, keep_alive
        
        Returns:
            dict: Response containing embeddings
        """
        api_key = kwargs.pop("api_key", None)
        headers = (
            {"Content-Type": "application/json", "Authorization": api_key}
            if api_key
            else {"Content-Type": "application/json"}
        )
        embeddings = []
        request_data = {"text": text}
        response = requests.post(f"{self.host_address}/lollms_embed", json=request_data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["vector"]

    def get_model_info(self) -> dict:
        """
        Return information about the current LOLLMS model.

        Returns:
            dict: Dictionary containing model name, version, host address, and personality.
        """
        return {
            "name": "lollms",
            "version": "1.0",
            "host_address": self.host_address,
            "model_name": self.model_name,
            "personality": self.personality
        }


    def listModels(self) -> dict:
        """Lists models"""
        url = f"{self.host_address}/list_models"

        response = requests.get(url)

        if response.status_code == 200:
            try:
                models = json.loads(response.content.decode("utf-8"))
                return [{"model_name":m} for m in models]
            except Exception as ex:
                return {"status": False, "error": str(ex)}
        else:
            return {"status": False, "error": response.text}


    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into the LOLLMS binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully.
        """
        self.model = model_name
        self.model_name = model_name
        return True

    # Lollms specific methods
    def lollms_listMountedPersonalities(self, host_address:str=None):
        host_address = host_address if host_address else self.host_address
        url = f"{host_address}/list_mounted_personalities"

        response = requests.get(url)

        if response.status_code == 200:
            try:
                text = json.loads(response.content.decode("utf-8"))
                return text
            except Exception as ex:
                return {"status": False, "error": str(ex)}
        else:
            return {"status": False, "error": response.text}
