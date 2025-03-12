# bindings/OpenAI/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm
if not pm.is_installed("openai"):
    pm.install("openai")
if not pm.is_installed("tiktoken"):
    pm.install("tiktoken")
import openai
import tiktoken
import os

BindingName = "OpenAIBinding"


class OpenAIBinding(LollmsLLMBinding):
    """OpenAI-specific binding implementation"""
    
    
    def __init__(self,
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat):
        """
        Initialize the OpenAI binding.

        Args:
            host_address (str): Host address for the OpenAI service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Ignored parameter for compatibility with LollmsLLMBinding.
        """
        super().__init__(
            host_address=host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS,
            model_name=model_name,
            service_key=service_key,
            verify_ssl_certificate=verify_ssl_certificate,
            default_completion_format=default_completion_format
        )
        self.service_key = os.getenv("OPENAI_API_KEY","")
        self.client = openai.OpenAI(base_url=host_address)

    
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
                    streaming_callback: Optional[Callable[[str, str], None]] = None) -> str:
        """
        Generate text based on the provided prompt and parameters.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.1.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p sampling parameter. Defaults to 0.95.
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40.
            seed (Optional[int]): Random seed for generation.
            n_threads (int): Number of threads to use. Defaults to 8.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).

        Returns:
            str: Generated text or error dictionary if failed.
        """
        count = 0
        output = ""

        # Prepare messages based on whether images are provided
        if images:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                            }
                        }
                        for image_path in images
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Generate text using the OpenAI API
        if completion_format == ELF_COMPLETION_FORMAT.Chat:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,  # Choose the engine according to your OpenAI plan
                messages=messages,
                max_tokens=n_predict,  # Adjust the desired length of the generated response
                n=1,  # Specify the number of responses you want
                temperature=temperature,  # Adjust the temperature for more or less randomness in the output
                stream=stream
            )

            if stream:
                for resp in chat_completion:
                    if count >= n_predict:
                        break
                    try:
                        word = resp.choices[0].delta.content
                    except Exception as ex:
                        word = ""
                    if streaming_callback is not None:
                        if not streaming_callback(word, "MSG_TYPE_CHUNK"):
                            break
                    if word:
                        output += word
                        count += 1
            else:
                output = chat_completion.choices[0].message.content
        else:
            completion = self.client.completions.create(
                model=self.model_name,  # Choose the engine according to your OpenAI plan
                prompt=prompt,
                max_tokens=n_predict,  # Adjust the desired length of the generated response
                n=1,  # Specify the number of responses you want
                temperature=temperature,  # Adjust the temperature for more or less randomness in the output
                stream=stream
            )

            if stream:
                for resp in completion:
                    if count >= n_predict:
                        break
                    try:
                        word = resp.choices[0].text
                    except Exception as ex:
                        word = ""
                    if streaming_callback is not None:
                        if not streaming_callback(word, "MSG_TYPE_CHUNK"):
                            break
                    if word:
                        output += word
                        count += 1
            else:
                output = completion.choices[0].text

        return output
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of individual characters.
        """
        try:
            return tiktoken.model.encoding_for_model(self.model_name).encode(text)
        except:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens (characters) to detokenize.

        Returns:
            str: Detokenized text.
        """
        try:
            return tiktoken.model.encoding_for_model(self.model_name).decode(tokens)
        except:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)
        
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using OpenAI API
        
        Args:
            text (str or List[str]): Input text to embed
            **kwargs: Additional arguments like model, truncate, options, keep_alive
        
        Returns:
            dict: Response containing embeddings
        """
        pass    
    def get_model_info(self) -> dict:
        """
        Return information about the current OpenAI model.

        Returns:
            dict: Dictionary containing model name, version, and host address.
        """
        return {
            "name": "OpenAI",
            "version": "2.0",
            "host_address": self.host_address,
            "model_name": self.model_name
        }
    def listModels(self):
        """ Lists available models """
        url = f'{self.host_address}/v1/models'
        headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {self.service_key}'
                }
        response = requests.get(url, headers=headers, verify= self.verify_ssl_certificate)
        try:
            data = response.json()
            model_info = []

            for model in data["data"]:
                model_name = model['id']
                owned_by = model['owned_by']
                created_datetime = model["created"]
                model_info.append({'model_name': model_name, 'owned_by': owned_by, 'created_datetime': created_datetime})

            return model_info
        except Exception as ex:
            trace_exception(ex)
            return []        
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into the OpenAI binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully.
        """
        self.model = model_name
        self.model_name = model_name
        return True