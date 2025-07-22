# bindings/OpenAI/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Dict

import pipmaster as pm

pm.ensure_packages(["openai","tiktoken"])

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
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat,
                 **kwargs):
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
            binding_name = "openai",
        )
        self.host_address=host_address
        self.model_name=model_name
        self.service_key=service_key
        self.verify_ssl_certificate=verify_ssl_certificate
        self.default_completion_format=default_completion_format

        if not self.service_key:
            self.service_key = os.getenv("OPENAI_API_KEY", self.service_key)
        self.client = openai.OpenAI(api_key=self.service_key, base_url=None if host_address is None else host_address if len(host_address)>0 else None)
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    
    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: float = 0.7, 
                     top_k: int = 40,          
                     top_p: float = 0.9,       
                     repeat_penalty: float = 1.1, 
                     repeat_last_n: int = 64, 
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
        count = 0
        output = ""
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            }
        ]

        # Prepare messages based on whether images are provided
        if images:
            if split:
                messages += self.split_discussion(prompt,user_keyword=user_keyword, ai_keyword=ai_keyword)
                if images:
                    messages[-1]["content"] = [
                        {
                            "type": "text",
                            "text": messages[-1]["content"]
                        }
                    ]+[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                            }
                        }
                        for image_path in images
                    ]
            else:
                messages.append({
                        'role': 'user', 
                        'content': [
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
                )
            
        else:
            
            if split:
                messages += self.split_discussion(prompt,user_keyword=user_keyword, ai_keyword=ai_keyword)
                if images:
                    messages[-1]["content"] = [
                        {
                            "type": "text",
                            "text": messages[-1]["content"]
                        }
                    ]
            else:
                messages.append({
                        'role': 'user', 
                        'content': [
                                        {
                                            "type": "text",
                                            "text": prompt
                                        }
                                    ]
                    }
                )

        # Generate text using the OpenAI API
        if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,  # Choose the engine according to your OpenAI plan
                    messages=messages,
                    max_tokens=n_predict,  # Adjust the desired length of the generated response
                    n=1,  # Specify the number of responses you want
                    temperature=temperature,  # Adjust the temperature for more or less randomness in the output
                    stream=stream
                )
            except Exception as ex:
                # exception for new openai models
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,  # Choose the engine according to your OpenAI plan
                    messages=messages,
                    max_completion_tokens=n_predict,  # Adjust the desired length of the generated response
                    n=1,  # Specify the number of responses you want
                    temperature=1,  # Adjust the temperature for more or less randomness in the output
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
                        if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
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
    
    def generate_from_messages(self,
                     messages: List[Dict],
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
                     **kwargs
                     ) -> Union[str, dict]:
        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream
        }
        # Add seed if available, as it's supported by newer OpenAI models
        if seed is not None:
            params["seed"] = seed

        # Remove None values, as the API expects them to be absent
        params = {k: v for k, v in params.items() if v is not None}
        
        output = ""
        # 2. Call the API
        try:
            try:
                completion = self.client.chat.completions.create(**params)
            except Exception as ex:
                # exception for new openai models
                params["max_completion_tokens"]=params["max_tokens"]
                params["temperature"]=1
                del params["max_tokens"]
                del params["top_p"]
                del params["frequency_penalty"]
                
                completion = self.client.chat.completions.create(**params)
            if stream:
                for chunk in completion:
                    # The streaming response for chat has a different structure
                    delta = chunk.choices[0].delta
                    if delta.content:
                        word = delta.content
                        if streaming_callback is not None:
                            if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                        output += word
            else:
                output = completion.choices[0].message.content
        
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(error_message, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": error_message}
            
        return output
    
    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: float = 0.7,
             top_k: int = 40,
             top_p: float = 0.9,
             repeat_penalty: float = 1.1,
             repeat_last_n: int = 64,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the OpenAI model using a LollmsDiscussion object.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter (Note: not all OpenAI models use this).
            top_p (float): Top-p sampling parameter.
            repeat_penalty (float): Frequency penalty for repeated tokens.
            seed (Optional[int]): Random seed for generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        # 1. Export the discussion to the OpenAI chat format
        # This handles system prompts, user/assistant roles, and multi-modal content automatically.
        messages = discussion.export("openai_chat", branch_tip_id)

        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream
        }
        # Add seed if available, as it's supported by newer OpenAI models
        if seed is not None:
            params["seed"] = seed

        # Remove None values, as the API expects them to be absent
        params = {k: v for k, v in params.items() if v is not None}
        
        output = ""
        # 2. Call the API
        try:
            # Check if we should use the chat completions or legacy completions endpoint
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
                try:
                    completion = self.client.chat.completions.create(**params)
                except Exception as ex:
                    # exception for new openai models
                    params["max_completion_tokens"]=params["max_tokens"]
                    params["temperature"]=1
                    del params["max_tokens"]
                    del params["top_p"]
                    del params["frequency_penalty"]
                    
                    completion = self.client.chat.completions.create(**params)
                    
                if stream:
                    for chunk in completion:
                        # The streaming response for chat has a different structure
                        delta = chunk.choices[0].delta
                        if delta.content:
                            word = delta.content
                            if streaming_callback is not None:
                                if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                    break
                            output += word
                else:
                    output = completion.choices[0].message.content

            else: # Fallback to legacy completion format (not recommended for chat)
                # We need to format the messages list into a single string prompt
                legacy_prompt = discussion.export("openai_completion", branch_tip_id)
                legacy_params = {
                    "model": self.model_name,
                    "prompt": legacy_prompt,
                    "max_tokens": n_predict,
                    "n": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": repeat_penalty,
                    "stream": stream
                }
                completion = self.client.completions.create(**legacy_params)

                if stream:
                    for chunk in completion:
                        word = chunk.choices[0].text
                        if streaming_callback is not None:
                            if not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                        output += word
                else:
                    output = completion.choices[0].text
        
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(error_message, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": error_message}
            
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
        Get embeddings for the input text using OpenAI API.

        Args:
            text (str): Input text to embed.
            **kwargs: Additional arguments. The 'model' argument can be used 
                      to specify the embedding model (e.g., "text-embedding-3-small").
                      Defaults to "text-embedding-ada-002".

        Returns:
            list: The embedding vector as a list of floats, or an empty list on failure.
        """
        # Determine the embedding model, prioritizing kwargs, with a default
        embedding_model = kwargs.get("model", "text-embedding-ada-002")
        
        try:
            # The OpenAI API expects the input to be a list of strings
            response = self.client.embeddings.create(
                model=embedding_model,
                input=[text]  # Wrap the single text string in a list
            )
            
            # Extract the embedding from the response
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                ASCIIColors.warning("OpenAI API returned no data for the embedding request.")
                return []
                
        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using OpenAI API: {e}")
            trace_exception(e)
            return []

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

    def listModels(self) -> List[Dict]:
        # Known context lengths
        known_context_lengths = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-1106-preview": 128000,
            "gpt-4-0125-preview": 128000,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16000,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "davinci": 2049,
            "curie": 2049,
            "babbage": 2049,
            "ada": 2049,
        }

        generation_prefixes = (
            "gpt-",
            "text-davinci",
            "davinci",
            "curie",
            "babbage",
            "ada"
        )

        models_info = []
        prompt_buffer = 500

        try:
            models = self.client.models.list()
            for model in models.data:
                model_id = model.id
                if model_id.startswith(generation_prefixes):
                    context_length = known_context_lengths.get(model_id, "unknown")
                    max_generation = (
                        context_length - prompt_buffer
                        if isinstance(context_length, int)
                        else "unknown"
                    )
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": context_length,
                        "max_generation": max_generation,
                    })
                else:
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": None,
                        "max_generation": None,
                    })
                    
        except Exception as e:
            print(f"Failed to list models: {e}")

        return models_info


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