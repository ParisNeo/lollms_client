# bindings/lollms/binding.py
import requests
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
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
                     stream: Optional[bool] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repeat_penalty: Optional[float] = None,
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
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
    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: Optional[float] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             repeat_penalty: Optional[float] = None,
             repeat_last_n: Optional[int] = None,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: int | None = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with a lollms-webui server using a LollmsDiscussion object.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            ... (other parameters) ...

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        # 1. Export the discussion to the lollms-native text format
        prompt_text = discussion.export("lollms_text", branch_tip_id)
        
        # 2. Extract images from the LAST message of the branch
        # lollms-webui's endpoint associates images with the final prompt
        active_branch_id = branch_tip_id or discussion.active_branch_id
        branch = discussion.get_branch(active_branch_id)
        last_message = branch[-1] if branch else None
        
        image_data = []
        if last_message and last_message.images:
            # The endpoint expects a list of base64 strings.
            # We will only process images of type 'base64'. URL types are not supported by this endpoint.
            for img in last_message.images:
                if img['type'] == 'base64':
                    image_data.append(img['data'])
                # Note: 'url' type images are ignored for this binding.

        # 3. Determine endpoint and build payload
        endpoint = "/lollms_generate_with_images" if image_data else "/lollms_generate"
        url = f"{self.host_address}{endpoint}"
        
        headers = {'Content-Type': 'application/json'}
        if self.service_key:
            headers['Authorization'] = f'Bearer {self.service_key}'

        data = {
            "prompt": prompt_text,
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

        # 4. Make the request (logic copied and adapted from generate_text)
        try:
            response = requests.post(
                url, 
                json=data, 
                headers=headers, 
                stream=stream,
                verify=self.verify_ssl_certificate
            )
            response.raise_for_status() # Raise an exception for bad status codes

            if not stream:
                return response.text.strip()
            else:
                full_response_text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        full_response_text += chunk
                        if streaming_callback:
                            if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                # Clean up potential quotes from some streaming formats
                if full_response_text.startswith('"') and full_response_text.endswith('"'):
                    full_response_text = full_response_text[1:-1]
                return full_response_text.rstrip('!')

        except requests.exceptions.RequestException as e:
            error_message = f"lollms-webui request error: {e}"
            return {"status": "error", "message": error_message}
        except Exception as ex:
            error_message = f"lollms-webui generation error: {str(ex)}"
            return {"status": "error", "message": error_message}    
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
