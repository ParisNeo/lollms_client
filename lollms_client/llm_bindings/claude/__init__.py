import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict
import json
import requests

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
# Added 'requests' for dynamic model listing
pm.ensure_packages(["anthropic", "pillow", "tiktoken", "requests"])

import anthropic
from PIL import Image, ImageDraw
import tiktoken

BindingName = "ClaudeBinding"

# API Endpoint for model listing
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com/v1"

# A hardcoded list to be used as a fallback if the API call fails
_FALLBACK_MODELS = [
    {'model_name': 'claude-3-opus-20240229', 'display_name': 'Claude 3 Opus', 'description': 'Most powerful model for highly complex tasks.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-3-5-sonnet-20240620', 'display_name': 'Claude 3.5 Sonnet', 'description': 'Our most intelligent model, a new industry standard.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-3-sonnet-20240229', 'display_name': 'Claude 3 Sonnet', 'description': 'Ideal balance of intelligence and speed for enterprise workloads.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-3-haiku-20240307', 'display_name': 'Claude 3 Haiku', 'description': 'Fastest and most compact model for near-instant responsiveness.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-2.1', 'display_name': 'Claude 2.1', 'description': 'Legacy model with a 200K token context window.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-2.0', 'display_name': 'Claude 2.0', 'description': 'Legacy model.', 'owned_by': 'Anthropic'},
    {'model_name': 'claude-instant-1.2', 'display_name': 'Claude Instant 1.2', 'description': 'Legacy fast and light-weight model.', 'owned_by': 'Anthropic'},
]


# Helper to check if a string is a valid path to an image
def is_image_path(path_str: str) -> bool:
    try:
        p = Path(path_str)
        return p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    except Exception:
        return False

# Helper to get image media type
def get_media_type(image_path: Union[str, Path]) -> str:
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
        # A default or raise an error
        return "image/jpeg"

class ClaudeBinding(LollmsLLMBinding):
    """Anthropic Claude-specific binding implementation."""

    def __init__(self,
                 host_address: str = None,  # Ignored, for compatibility
                 model_name: str = "claude-3-sonnet-20240229",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True, # Ignored, for compatibility
                 **kwargs
                 ):
        """
        Initialize the Claude binding.

        Args:
            model_name (str): Name of the Claude model to use.
            service_key (str): Anthropic API key.
        """
        super().__init__(binding_name=BindingName)
        self.model_name = model_name
        self.service_key = service_key
        self._cached_models: Optional[List[Dict[str, str]]] = None

        if not self.service_key:
            self.service_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.service_key:
            raise ValueError("Anthropic API key is required. Please set it via the 'service_key' parameter or the ANTHROPIC_API_KEY environment variable.")

        try:
            self.client = anthropic.Anthropic(api_key=self.service_key)
        except Exception as e:
            ASCIIColors.error(f"Failed to configure Anthropic client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure Anthropic client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              top_k: int,
                              n_predict: int) -> Dict[str, any]:
        """Builds a parameters dictionary for the Claude API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if top_k is not None: params['top_k'] = top_k
        if n_predict is not None: params['max_tokens'] = n_predict
        return params

    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = 2048,
                     stream: Optional[bool] = False,
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1, # Not supported
                     repeat_last_n: int = 64,   # Not supported
                     seed: Optional[int] = None,      # Not supported
                     n_threads: Optional[int] = None, # Not applicable
                     ctx_size: int | None = None,     # Determined by model
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, # Not used in this direct method
                     user_keyword:Optional[str]="!@>user:", # Not used
                     ai_keyword:Optional[str]="!@>assistant:", # Not used
                     ) -> Union[str, dict]:
        """
        Generate text using the Claude model.
        """
        if not self.client:
            return {"status": False, "error": "Anthropic client not initialized."}

        api_params = self._construct_parameters(temperature, top_p, top_k, n_predict)
        
        message_content = []
        if prompt and prompt.strip():
            message_content.append({"type": "text", "text": prompt})

        if images:
            for image_data in images:
                try:
                    # ... (image processing code is unchanged)
                    if is_image_path(image_data):
                        with open(image_data, "rb") as image_file:
                            b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                        media_type = get_media_type(image_data)
                    else:
                        b64_data = image_data
                        media_type = "image/jpeg"
                    
                    message_content.append({
                        "type": "image",
                        "source": { "type": "base64", "media_type": media_type, "data": b64_data }
                    })
                except Exception as e:
                    error_msg = f"Failed to process image: {e}"
                    ASCIIColors.error(error_msg)
                    return {"status": False, "error": error_msg}

        if not message_content:
            if stream and streaming_callback:
                streaming_callback("", MSG_TYPE.MSG_TYPE_FINISHED_MESSAGE)
            return ""

        messages = [{"role": "user", "content": message_content}]
        full_response_text = ""

        # ---- CHANGE START ----
        # Conditionally build the request arguments to avoid sending an empty `system` parameter.
        request_args = {
            "model": self.model_name,
            "messages": messages,
            **api_params
        }
        if system_prompt and system_prompt.strip():
            request_args["system"] = system_prompt
        # ---- CHANGE END ----

        try:
            if stream:
                with self.client.messages.stream(**request_args) as stream_response:
                    for chunk in stream_response.text_stream:
                        full_response_text += chunk
                        if streaming_callback:
                            if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                response = self.client.messages.create(**request_args)
                if response.stop_reason == "error":
                     return {"status": False, "error": f"API returned an error: {response.stop_reason}"}
                return response.content[0].text

        except Exception as ex:
            error_message = f"An unexpected error occurred with Claude API: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = 2048,
             stream: Optional[bool] = False,
             temperature: float = 0.7,
             top_k: int = 40,
             top_p: float = 0.9,
             repeat_penalty: float = 1.1, # Not supported
             repeat_last_n: int = 64, # Not supported
             seed: Optional[int] = None, # Not supported
             n_threads: Optional[int] = None, # Not supported
             ctx_size: Optional[int] = None, # Not supported
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Claude model using a LollmsDiscussion object.
        """
        if not self.client:
             return {"status": "error", "message": "Anthropic client not initialized."}

        system_prompt = discussion.system_prompt
        messages = discussion.get_messages(branch_tip_id)
        
        history = []
        # ... (history building code is unchanged)
        for msg in messages:
            role = 'user' if msg.sender_type == "user" else 'assistant'
            content_parts = []
            if msg.content and msg.content.strip():
                content_parts.append({"type": "text", "text": msg.content})
            
            if msg.images:
                for file_path in msg.images:
                    if is_image_path(file_path):
                        try:
                            with open(file_path, "rb") as image_file:
                                b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                            media_type = get_media_type(file_path)
                            content_parts.append({
                                "type": "image",
                                "source": { "type": "base64", "media_type": media_type, "data": b64_data }
                            })
                        except Exception as e:
                            ASCIIColors.warning(f"Could not load image {file_path}: {e}")
            
            if content_parts:
                if history and history[-1]['role'] == role:
                    history[-1]['content'].extend(content_parts)
                else:
                    history.append({'role': role, 'content': content_parts})
        
        if not history:
            return {"status": "error", "message": "Cannot start chat with an empty discussion."}

        api_params = self._construct_parameters(temperature, top_p, top_k, n_predict)
        full_response_text = ""

        # ---- CHANGE START ----
        # Conditionally build the request arguments to avoid sending an empty `system` parameter.
        request_args = {
            "model": self.model_name,
            "messages": history,
            **api_params
        }
        if system_prompt and system_prompt.strip():
            request_args["system"] = system_prompt
        # ---- CHANGE END ----

        try:
            if stream:
                with self.client.messages.stream(**request_args) as stream_response:
                    for chunk in stream_response.text_stream:
                        full_response_text += chunk
                        if streaming_callback:
                            if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                response = self.client.messages.create(**request_args)
                if response.stop_reason == "error":
                     return {"status": "error", "message": f"API returned an error: {response.stop_reason}"}
                return response.content[0].text

        except Exception as ex:
            error_message = f"An unexpected error occurred with Claude API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}
            
    # ... (Rest of the file is unchanged) ...
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text.
        Note: Claude doesn't expose a public tokenizer API.
        Using tiktoken for a rough estimate, NOT accurate for Claude.
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
        Count tokens from a text using the Anthropic API.
        This provides a more accurate count than a fallback tokenizer.
        """
        if not text or not text.strip():
            return 0

        if not self.client:
            ASCIIColors.warning("Cannot count tokens, Anthropic client not initialized.")
            return len(self.tokenize(text))
        try:
            # Note: count_tokens doesn't use a system prompt, so it's safe.
            # However, for consistency, we could add one if needed by the logic.
            # For now, this is fine as it only counts user content tokens.
            response = self.client.count_tokens( # Changed from messages.count_tokens to top-level client method
                model=self.model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.token_count # Updated to correct response attribute
        except Exception as e:
            ASCIIColors.error(f"Failed to count tokens with Claude API: {e}")
            return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text.
        Note: Anthropic does not provide a dedicated embedding model API.
        """
        ASCIIColors.warning("Anthropic does not offer a public embedding API. This method is not implemented.")
        raise NotImplementedError("Claude binding does not support embeddings.")

    def get_model_info(self) -> dict:
        """Return information about the current Claude model setup."""
        return {
            "name": self.binding_name,
            "version": anthropic.__version__,
            "host_address": "https://api.anthropic.com",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": "claude-3" in self.model_name,
        }

    def listModels(self) -> List[Dict[str, str]]:
        """
        Lists available models from the Anthropic API.
        Caches the result to avoid repeated API calls.
        Falls back to a static list if the API call fails.
        """
        if self._cached_models is not None:
            return self._cached_models

        if not self.service_key:
            ASCIIColors.warning("Cannot fetch models without an API key. Using fallback list.")
            self._cached_models = _FALLBACK_MODELS
            return self._cached_models

        # This part is complex and likely correct, leaving as is.
        # It's good practice.
        headers = {
            "x-api-key": self.service_key,
            "anthropic-version": "2023-06-01", 
            "accept": "application/json"
        }
        url = f"{ANTHROPIC_API_BASE_URL}/models"
        
        try:
            ASCIIColors.info("Fetching available models from Anthropic API...")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "data" in data and isinstance(data["data"], list):
                models_data = data["data"]
                formatted_models = []
                for model in models_data:
                    model_id = model.get("id")
                    if not model_id: continue
                    
                    display_name = model.get("name", model_id.replace("-", " ").title())
                    
                    desc_parts = []
                    if model.get('context_length'):
                        desc_parts.append(f"Context: {model['context_length']:,} tokens.")
                    if model.get('max_output_tokens'):
                         desc_parts.append(f"Max Output: {model['max_output_tokens']:,} tokens.")
                    description = " ".join(desc_parts) or f"Anthropic model: {model_id}"

                    formatted_models.append({
                        'model_name': model_id,
                        'display_name': display_name,
                        'description': description,
                        'owned_by': 'Anthropic'
                    })

                formatted_models.sort(key=lambda x: x['model_name'])
                self._cached_models = formatted_models
                ASCIIColors.green(f"Successfully fetched and parsed {len(self._cached_models)} models.")
                return self._cached_models
            else:
                raise ValueError("API response is malformed. 'data' field missing or not a list.")

        except Exception as e:
            ASCIIColors.error(f"Failed to fetch models from Anthropic API: {e}")
            ASCIIColors.warning("Using hardcoded fallback list of models.")
            trace_exception(e)
            self._cached_models = _FALLBACK_MODELS
            return self._cached_models

    def load_model(self, model_name: str) -> bool:
        """Set the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Claude model set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Example Usage (requires ANTHROPIC_API_KEY environment variable)
    if 'ANTHROPIC_API_KEY' not in os.environ:
        ASCIIColors.red("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please get your key from Anthropic and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing ClaudeBinding ---")

    # --- Configuration ---
    test_model_name = "claude-3-haiku-20240307" # Use Haiku for speed in testing
    test_vision_model_name = "claude-3-sonnet-20240229"
    
    full_streamed_text = ""

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = ClaudeBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using anthropic version: {anthropic.__version__}")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models (dynamic) ---")
        models = binding.listModels()
        if models:
            ASCIIColors.green(f"Found {len(models)} models.")
            for m in models:
                print(f"- {m['model_name']} ({m['display_name']})")
        else:
            ASCIIColors.error("Failed to list models.")

        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens (with valid and empty text) ---")
        sample_text = "Hello, world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count} (via API)")
        empty_token_count = binding.count_tokens("  ")
        ASCIIColors.green(f"Token count for empty string: {empty_token_count} (handled locally)")
        assert empty_token_count == 0

        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Explain the importance of bees in one paragraph."
        ASCIIColors.info(f"Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, n_predict=100, stream=False, system_prompt=" ")
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        
        captured_chunks = []
        def stream_callback(chunk: str, msg_type: int):
            ASCIIColors.green(chunk, end="", flush=True)
            captured_chunks.append(chunk)
            return True
        
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, n_predict=150, stream=True, streaming_callback=stream_callback)
        full_streamed_text = "".join(captured_chunks)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")
        assert result == full_streamed_text

        # --- Embeddings ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            embedding_text = "Lollms is a cool project."
            embedding_vector = binding.embed(embedding_text)
        except NotImplementedError as e:
            ASCIIColors.green(f"Successfully caught expected error for embeddings: {e}")
        except Exception as e:
            ASCIIColors.error(f"Caught an unexpected error for embeddings: {e}")

        # --- Vision Model Test ---
        dummy_image_path = "claude_dummy_test_image.png"
        try:
            available_model_names = [m['model_name'] for m in models]
            if test_vision_model_name not in available_model_names:
                 ASCIIColors.warning(f"Vision test model '{test_vision_model_name}' not available. Skipping vision test.")
            else:
                img = Image.new('RGB', (200, 50), color = ('blue'))
                d = ImageDraw.Draw(img)
                d.text((10,10), "Test Image", fill=('yellow'))
                img.save(dummy_image_path)
                ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

                ASCIIColors.cyan(f"\n--- Vision Generation (using {test_vision_model_name}) ---")
                binding.load_model(test_vision_model_name)
                vision_prompt = "What color is the text and what does it say?"
                ASCIIColors.info(f"Vision Prompt: {vision_prompt} with image {dummy_image_path}")
                
                vision_response = binding.generate_text(
                    prompt=vision_prompt,
                    images=[dummy_image_path],
                    n_predict=50,
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

    ASCIIColors.yellow("\nClaudeBinding test finished.")