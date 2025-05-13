# bindings/ollama/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors, trace_exception

BindingName = "OllamaBinding"


def count_tokens_ollama(
    text_to_tokenize: str,
    model_name: str,
    ollama_host: str = "http://localhost:11434",
    timeout: int = 30
) -> int:
    """
    Counts the number of tokens in a given text using a specified Ollama model
    by calling the Ollama server's /api/tokenize endpoint.

    Args:
        text_to_tokenize (str): The text to be tokenized.
        model_name (str): The name of the Ollama model to use (e.g., "llama3", "mistral").
        ollama_host (str): The base URL of the Ollama server (default: "http://localhost:11434").
        timeout (int): Timeout for the request in seconds (default: 30).

    Returns:
        int: The number of tokens. Returns -1 if an error occurs.

    Raises:
        requests.exceptions.RequestException: For network issues or timeouts.
        requests.exceptions.HTTPError: For HTTP error responses (4xx or 5xx).
        ValueError: If the response from Ollama is not as expected (e.g., missing 'tokens' key).
    """
    api_url = f"{ollama_host}/api/tokenize"
    payload = {
        "model": model_name,
        "prompt": text_to_tokenize
        # You can add "options" here if needed, but for tokenization, it's usually not required.
        # "options": {"num_ctx": 4096} # Example, might influence tokenizer for specific context length
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()

        if "tokens" in response_data and isinstance(response_data["tokens"], list):
            return len(response_data["tokens"])
        else:
            raise ValueError(
                f"Ollama response did not contain a 'tokens' list. Response: {response_data}"
            )

    except requests.exceptions.HTTPError as http_err:
        # You might want to inspect http_err.response.text for more details from Ollama
        print(f"HTTP error occurred: {http_err} - {http_err.response.text}")
        raise  # Re-raise the exception
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        raise
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
        raise
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred with the request: {req_err}")
        raise
    except json.JSONDecodeError as json_err:
        # This can happen if the server returns non-JSON, e.g., an HTML error page
        raise ValueError(
            f"Failed to decode JSON response from Ollama: {json_err}. Response text: {response.text}"
        ) from json_err
    except ValueError as val_err: # Catches the ValueError raised above for missing 'tokens'
        print(f"Value error: {val_err}")
        raise
class OllamaBinding(LollmsLLMBinding):
    """Ollama-specific binding implementation"""
    
    DEFAULT_HOST_ADDRESS = "http://localhost:11434"
    
    def __init__(self,
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat
                 ):
        """
        Initialize the Ollama binding.

        Args:
            host_address (str): Host address for the Ollama service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Ignored parameter for compatibility with LollmsLLMBinding.
        """
        super().__init__(
            binding_name="ollama",
            host_address=host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS,
            model_name=model_name,
            service_key=service_key,
            verify_ssl_certificate=verify_ssl_certificate,
            default_completion_format=default_completion_format
        )
        self.model = None
    
    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None,
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
        Generate text using the Ollama service, with optional image support.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
                If provided, uses the /api endpoint with message format.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.1.
            top_k (int): Top-k sampling parameter. Defaults to 50 (not used in Ollama API directly).
            top_p (float): Top-p sampling parameter. Defaults to 0.95 (not used in Ollama API directly).
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8 (not used in Ollama API directly).
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40 (not used).
            seed (Optional[int]): Random seed for generation.
            n_threads (int): Number of threads to use. Defaults to 8 (not used in Ollama API directly).
            streaming_callback (Optional[Callable[[str, str], None]]): Callback for streaming output.
                - First parameter (str): The chunk of text received from the stream.
                - Second parameter (str): The message type (typically MSG_TYPE.MSG_TYPE_CHUNK).

        Returns:
            Union[str, dict]: Generated text if successful, or a dictionary with status and error if failed.

        Note:
            Some parameters (top_k, top_p, repeat_penalty, repeat_last_n, n_threads) are included for interface
            consistency but are not directly used in the Ollama API implementation.
        """
        # Set headers
        headers = {
            'Content-Type': 'application/json',
        }
        if self.service_key:
            headers['Authorization'] = f'Bearer {self.service_key}'

        # Clean host address
        host_address = self.host_address.rstrip('/')

        # Prepare data based on whether images are provided
        if images:
            # Multimodal generation using /api endpoint
            images_list = [encode_image(image_path) for image_path in images]
            data = {
                'model': self.model_name,
                'messages': [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                        } for img in images_list
                    ]
                }],
                "stream": stream,
                "temperature": float(temperature),
                "max_tokens": n_predict,
            }
            if ctx_size is not None:
                data["num_ctx"] = ctx_size            
            url = f'{host_address}/api/chat'
        else:
            # Text-only generation using /api/generate endpoint
            data = {
                'model': self.model_name,
                'prompt': prompt,
                "stream": stream,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
            url = f'{host_address}/api/generate'

        # Make the request
        response = requests.post(url, json=data, headers=headers, stream=stream)

        # Handle response
        if not stream:
            if response.status_code == 200:
                try:
                    if images:
                        # For multimodal, response is in chat format
                        return response.json()["message"]["content"]
                    else:
                        # For text-only
                        return response.json()["response"]
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            elif response.status_code == 404:
                ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
                return {"status": False, "error": "404 Not Found"}
            else:
                return {"status": False, "error": response.text}
        else:
            text = ""
            if response.status_code == 200:
                try:
                    for line in response.iter_lines():
                        decoded = line.decode("utf-8")
                        if images:
                            # Streaming with images (chat format)
                            if decoded.startswith("data: "):
                                json_data = json.loads(decoded[5:].strip())
                                chunk = json_data["message"]["content"] if "message" in json_data else ""
                            else:
                                continue
                        else:
                            # Streaming without images (generate format)
                            json_data = json.loads(decoded)
                            chunk = json_data["response"]
                        
                        text += chunk
                        if streaming_callback:
                            if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                    return text
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            elif response.status_code == 404:
                ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
                return {"status": False, "error": "404 Not Found"}
            elif response.status_code == 400:
                try:
                    content = json.loads(response.content.decode("utf8"))
                    return {"status": False, "error": content.get("error", {}).get("message", content.get("message", "Unknown error"))}
                except:
                    return {"status": False, "error": response.content.decode("utf8")}
            else:
                return {"status": False, "error": response.text}
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of individual characters.
        """
        return list(text)
    
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens (characters) to detokenize.

        Returns:
            str: Detokenized text.
        """
        return "".join(tokens)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """        
        return count_tokens_ollama(text, self.model_name, self.host_address)
    
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        
        Args:
            text (str or List[str]): Input text to embed
            **kwargs: Additional arguments like model, truncate, options, keep_alive
        
        Returns:
            dict: Response containing embeddings
        """
        import requests
        
        url = f"{self.base_url}/api/embed"
        
        # Prepare the request payload
        payload = {
            "input": text,
            "model": kwargs.get("model", "llama2")  # default model
        }
        
        # Add optional parameters if provided
        if "truncate" in kwargs:
            payload["truncate"] = kwargs["truncate"]
        if "options" in kwargs:
            payload["options"] = kwargs["options"]
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Embedding request failed: {str(e)}")  
        

    def get_model_info(self) -> dict:
        """
        Return information about the current Ollama model.

        Returns:
            dict: Dictionary containing model name, version, and host address.
        """
        return {
            "name": "ollama",
            "version": "2.0",
            "host_address": self.host_address,
            "model_name": self.model_name
        }
    def listModels(self):
        """ Lists available models """
        url = f'{self.host_address}/api/tags'
        headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {self.service_key}'
                }
        response = requests.get(url, headers=headers, verify= self.verify_ssl_certificate)
        try:
            ASCIIColors.debug("Listing ollama models")
            data = response.json()
            model_info = []

            for model in data['models']:
                model_name = model['name']
                owned_by = ""
                created_datetime = model["modified_at"]
                model_info.append({'model_name': model_name, 'owned_by': owned_by, 'created_datetime': created_datetime})

            return model_info
        except Exception as ex:
            trace_exception(ex)
            return []
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into the Ollama binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully.
        """
        self.model = model_name
        self.model_name = model_name
        return True
