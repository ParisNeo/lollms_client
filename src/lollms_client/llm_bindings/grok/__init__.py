import base64
import os
import json
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["requests", "pillow", "tiktoken"])

from PIL import Image, ImageDraw
import tiktoken

BindingName = "GrokBinding"

# API Endpoint
GROK_API_BASE_URL = "https://api.x.ai/v1"

# A hardcoded list to be used as a fallback if the API call fails
_FALLBACK_MODELS = [
    {'model_name': 'grok-2-latest', 'display_name': 'Grok 2 Latest', 'description': 'The latest conversational model from xAI.', 'owned_by': 'xAI'},
    {'model_name': 'grok-2', 'display_name': 'Grok 2', 'description': 'Grok 2 model.', 'owned_by': 'xAI'},
    {'model_name': 'grok-2-vision-latest', 'display_name': 'Grok 2 Vision Latest', 'description': 'Latest multimodal model from xAI.', 'owned_by': 'xAI'},
    {'model_name': 'grok-beta', 'display_name': 'Grok Beta', 'description': 'Beta model.', 'owned_by': 'xAI'},
    {'model_name': 'grok-vision-beta', 'display_name': 'Grok Vision Beta', 'description': 'Beta vision model.', 'owned_by': 'xAI'},
]

# Helper to check if a string is a valid path to an image
def is_image_path(path_str: str) -> bool:
    try:
        p = Path(path_str)
        return p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    except Exception:
        return False

# Helper to get image media type for base64 URI
def get_media_type_for_uri(image_path: Union[str, Path]) -> str:
    path = Path(image_path)
    ext = path.suffix.lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        # Default to PNG as it's lossless and widely supported
        return "image/png"


class GrokBinding(LollmsLLMBinding):
    """xAI Grok-specific binding implementation."""

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the Grok binding.

        Args:
            model_name (str): Name of the Grok model to use.
            service_key (str): xAI API key.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name", "grok-2-latest")
        self.service_key = kwargs.get("service_key")
        self.base_url = kwargs.get("base_url", GROK_API_BASE_URL)
        self._cached_models: Optional[List[Dict[str, str]]] = None

        if not self.service_key:
            self.service_key = os.getenv("XAI_API_KEY")
        
        if not self.service_key:
            raise ValueError("xAI API key is required. Please set it via the 'service_key' parameter or the XAI_API_KEY environment variable.")

        self.headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json"
        }

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int) -> Dict[str, any]:
        """Builds a parameters dictionary for the Grok API."""
        params = {"stream": True} # Always stream from the API
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        # Grok has a model-specific max_tokens, but we can request less.
        if n_predict is not None: params['max_tokens'] = n_predict
        return params

    def _process_and_handle_stream(self,
                                  response: requests.Response,
                                  stream: bool,
                                  streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
                                  think: bool = False
                                  ) -> Union[str, dict]:
        """Helper to process streaming responses from the API."""
        full_response_text = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                # Check for reasoning content (DeepSeek-style) if Grok adopts it or if proxied
                                reasoning = delta.get('reasoning_content', '')

                                if reasoning:
                                    # If thinking is requested and we get reasoning tokens
                                    if think:
                                        if streaming_callback:
                                            # We just stream the reasoning as is, user UI typically handles tagging or we could inject <think>
                                            # Here we assume just passing the text is safer unless we track state
                                            streaming_callback(reasoning, MSG_TYPE.MSG_TYPE_CHUNK)
                                    # We don't append reasoning to full_response_text usually if it's separate, 
                                    # unless we want to return it in the final string wrapped.
                                    # Let's wrap it for the final return string.
                                    full_response_text += f"<think>{reasoning}</think>" # Naive wrapping for stream accumulation

                                if content:
                                    full_response_text += content
                                    if stream and streaming_callback:
                                        if not streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                                            # Stop streaming if the callback returns False
                                            return full_response_text
                        except json.JSONDecodeError:
                            ASCIIColors.warning(f"Could not decode JSON chunk: {json_str}")
                            continue

            # This handles both cases:
            # - If stream=True, we have already sent chunks. We return the full string.
            # - If stream=False, we have buffered the whole response and now return it.
            return full_response_text
        
        except Exception as ex:
            error_message = f"An unexpected error occurred while processing the Grok stream: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}


    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = 2048,
                     stream: Optional[bool] = False,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1, # Not supported
                     repeat_last_n: int = 64,   # Not supported
                     seed: Optional[int] = None,      # Not supported
                     n_threads: Optional[int] = None, # Not applicable
                     ctx_size: int | None = None,     # Determined by model
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     think: Optional[bool] = False,
                     reasoning_effort: Optional[str] = "low", # low, medium, high
                     reasoning_summary: Optional[bool] = False, # auto
                     **kwargs
                     ) -> Union[str, dict]:
        """
        Generate text using the Grok model.
        """
        if not self.service_key:
            return {"status": False, "error": "xAI API key not configured."}

        api_params = self._construct_parameters(temperature, top_p, n_predict)

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
            
        user_content = []
        if prompt and prompt.strip():
            user_content.append({"type": "text", "text": prompt})

        if images:
            for image_data in images:
                try:
                    if is_image_path(image_data):
                        media_type = get_media_type_for_uri(image_data)
                        with open(image_data, "rb") as image_file:
                            b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                    else: # Assume it's a base64 string
                        b64_data = image_data
                        if b64_data.startswith("data:image"):
                             b64_data = b64_data.split(",")[1]
                        media_type = "image/png" # Default assumption

                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64_data}"}
                    })
                except Exception as e:
                    error_msg = f"Failed to process image: {e}"
                    ASCIIColors.error(error_msg)
                    return {"status": False, "error": error_msg}

        if not user_content:
            if stream and streaming_callback:
                streaming_callback("", MSG_TYPE.MSG_TYPE_FINISHED_MESSAGE)
            return ""

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            **api_params
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True  # We always use the streaming endpoint
            )
            response.raise_for_status()
            
            return self._process_and_handle_stream(response, stream, streaming_callback, think=think)

        except requests.exceptions.RequestException as ex:
            error_message = f"Grok API request failed: {str(ex)}"
            try: # Try to get more info from the response body
                 error_message += f"\nResponse: {ex.response.text}"
            except:
                pass
            trace_exception(ex)
            return {"status": False, "error": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred with Grok API: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}


    def _chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = 2048,
             stream: Optional[bool] = False,
             temperature: float = 0.7,
             top_p: float = 0.9,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
             think: Optional[bool] = False,
             reasoning_effort: Optional[str] = "low", # low, medium, high
             reasoning_summary: Optional[bool] = False, # auto
             **kwargs
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Grok model using a LollmsDiscussion object.
        """
        if not self.service_key:
             return {"status": "error", "message": "xAI API key not configured."}

        system_prompt = discussion.system_prompt
        discussion_messages = discussion.get_messages(branch_tip_id)
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
            
        for msg in discussion_messages:
            role = 'assistant' if msg.sender_type == "assistant" else 'user'
            
            content_parts = []
            if msg.content and msg.content.strip():
                content_parts.append({"type": "text", "text": msg.content})
            
            if msg.images:
                for file_path in msg.images:
                    if is_image_path(file_path):
                        try:
                            media_type = get_media_type_for_uri(file_path)
                            with open(file_path, "rb") as image_file:
                                b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{b64_data}"}
                            })
                        except Exception as e:
                            ASCIIColors.warning(f"Could not load image {file_path}: {e}")
                    else:
                        # Attempt to handle base64
                        try:
                            b64_data = file_path
                            if b64_data.startswith("data:image"):
                                b64_data = b64_data.split(",")[1]
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
                            })
                        except:
                            pass
            
            # Grok API expects content to be a string for assistant, or list for user.
            if role == 'user':
                messages.append({'role': role, 'content': content_parts})
            else: # assistant
                # Assistants can't send images, so we just extract the text.
                text_content = next((part['text'] for part in content_parts if part['type'] == 'text'), "")
                if text_content:
                    messages.append({'role': role, 'content': text_content})
        
        if not messages or messages[-1]['role'] != 'user':
            return {"status": "error", "message": "Cannot start chat without a user message."}

        api_params = self._construct_parameters(temperature, top_p, n_predict)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **api_params
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            return self._process_and_handle_stream(response, stream, streaming_callback, think=think)

        except requests.exceptions.RequestException as ex:
            error_message = f"Grok API request failed: {str(ex)}"
            try:
                 error_message += f"\nResponse: {ex.response.text}"
            except:
                pass
            trace_exception(ex)
            return {"status": "error", "message": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred with Grok API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}
            
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text.
        Note: Grok doesn't expose a public tokenizer API.
        Using tiktoken's cl100k_base for a reasonable estimate.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except:
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize a list of tokens.
        Note: Based on the placeholder tokenizer.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except:
            return bytes(tokens).decode('utf-8', errors='ignore')

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text using the fallback tokenizer.
        """
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text.
        Note: xAI does not provide a dedicated embedding model API.
        """
        ASCIIColors.warning("xAI does not offer a public embedding API. This method is not implemented.")
        raise NotImplementedError("Grok binding does not support embeddings.")

    def get_model_info(self) -> dict:
        """Return information about the current Grok model setup."""
        return {
            "name": self.binding_name,
            "host_address": self.base_url,
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": "vision" in self.model_name or "grok-1.5" in self.model_name or "grok-2" in self.model_name,
        }

    def list_models(self) -> List[Dict[str, str]]:
        """
        Lists available models from the xAI API.
        Caches the result to avoid repeated API calls.
        Falls back to a static list if the API call fails.
        """
        if self._cached_models is not None:
            return self._cached_models

        if not self.service_key:
            ASCIIColors.warning("Cannot fetch models without an API key. Using fallback list.")
            self._cached_models = _FALLBACK_MODELS
            return self._cached_models

        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "data" in data and isinstance(data["data"], list):
                models_data = data["data"]
                formatted_models = []
                for model in models_data:
                    model_id = model.get("id")
                    if not model_id: continue
                    
                    display_name = model_id.replace("-", " ").title()
                    description = f"Context: {model.get('context_window', 'N/A')} tokens."
                    
                    formatted_models.append({
                        'model_name': model_id,
                        'display_name': display_name,
                        'description': description,
                        'owned_by': model.get('owned_by', 'xAI')
                    })

                self._cached_models = formatted_models
                ASCIIColors.green(f"Successfully fetched {len(self._cached_models)} models.")
                return self._cached_models
            else:
                raise ValueError("API response is malformed.")

        except Exception as e:
            ASCIIColors.error(f"Failed to fetch models from xAI API: {e}")
            ASCIIColors.warning("Using hardcoded fallback list of models.")
            trace_exception(e)
            self._cached_models = _FALLBACK_MODELS
            return self._cached_models

    def load_model(self, model_name: str) -> bool:
        """Set the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Grok model set to: {model_name}. It will be used on the next API call.")
        return True


if __name__ == '__main__':
    # Example Usage (requires XAI_API_KEY environment variable)
    if 'XAI_API_KEY' not in os.environ:
        ASCIIColors.red("Error: XAI_API_KEY environment variable not set.")
        print("Please get your key from xAI and set it as an environment variable.")
        exit(1)

    ASCIIColors.yellow("--- Testing GrokBinding ---")

    # --- Configuration ---
    test_model_name = "grok-2-latest"
    test_vision_model_name = "grok-2-vision-latest"

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = GrokBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models (dynamic) ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Found {len(models)} models.")
            for m in models:
                print(f"- {m['model_name']} ({m['display_name']})")
        else:
            ASCIIColors.error("Failed to list models.")

        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, world! This is a test from the Grok binding."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count} (using tiktoken)")

        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Explain who Elon Musk is in one sentence."
        ASCIIColors.info(f"Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, n_predict=100, stream=False, system_prompt="Be very concise.", think=True)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            ASCIIColors.green(chunk, end="", flush=True)
            full_streamed_text += chunk
            return True
        
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, n_predict=150, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")
        assert result == full_streamed_text

        # --- Embeddings ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            binding.embed("This should fail.")
        except NotImplementedError as e:
            ASCIIColors.green(f"Successfully caught expected error for embeddings: {e}")

        # --- Vision Model Test ---
        dummy_image_path = "grok_dummy_test_image.png"
        try:
            available_model_names = [m['model_name'] for m in models]
            if test_vision_model_name not in available_model_names:
                 ASCIIColors.warning(f"Vision test model '{test_vision_model_name}' not available. Skipping vision test.")
            else:
                img = Image.new('RGB', (250, 60), color=('red'))
                d = ImageDraw.Draw(img)
                d.text((10, 10), "This is a test image for Grok", fill=('white'))
                img.save(dummy_image_path)
                ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

                ASCIIColors.cyan(f"\n--- Vision Generation (using {test_vision_model_name}) ---")
                binding.load_model(test_vision_model_name)
                vision_prompt = "Describe this image. What does the text say?"
                ASCIIColors.info(f"Vision Prompt: {vision_prompt} with image {dummy_image_path}")
                
                vision_response = binding.generate_text(
                    prompt=vision_prompt,
                    images=[dummy_image_path],
                    n_predict=100,
                    stream=False
                )
                if isinstance(vision_response, str):
                    ASCIIColors.green(f"Vision model response: {vision_response}")
                else:
                    ASCIIColors.error(f"Vision generation failed: {vision_response}")
        except Exception as e:
            ASCIIColors.error(f"Error during vision test: {e}")
            trace_exception(e)
        finally:
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nGrokBinding test finished.")