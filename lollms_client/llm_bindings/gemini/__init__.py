# bindings/gemini/binding.py
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["google-generativeai", "pillow", "tiktoken", "protobuf"])

import google.generativeai as genai
from PIL import Image, ImageDraw # ImageDraw is used in the test script below
import tiktoken

BindingName = "GeminiBinding"

# Helper to check if a string is a valid path to an image
def is_image_path(path_str: str) -> bool:
    try:
        p = Path(path_str)
        return p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    except Exception:
        return False

class GeminiBinding(LollmsLLMBinding):
    """Google Gemini-specific binding implementation."""

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the Gemini binding.

        Args:
            model_name (str): Name of the Gemini model to use.
            service_key (str): Google AI Studio API key.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name", None)
        self.service_key = kwargs.get("service_key", None)

        if not self.service_key:
            self.service_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.service_key:
            raise ValueError("Google API key is required. Please set it via the 'service_key' parameter or the GOOGLE_API_KEY environment variable.")

        try:
            genai.configure(api_key=self.service_key)
            self.client = genai # Alias for consistency
        except Exception as e:
            ASCIIColors.error(f"Failed to configure Gemini client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure Gemini client: {e}") from e

    def get_generation_config(self, 
                              temperature: float, 
                              top_p: float, 
                              top_k: int, 
                              n_predict: int) -> genai.types.GenerationConfig:
        """Builds a GenerationConfig object from parameters."""
        config = {}
        if temperature is not None: config['temperature'] = float(temperature)
        if top_p is not None: config['top_p'] = top_p
        if top_k is not None: config['top_k'] = top_k
        if n_predict is not None: config['max_output_tokens'] = n_predict
        return genai.types.GenerationConfig(**config)

    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = 2048,
                     stream: Optional[bool] = False,
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1, # Not directly supported by Gemini API
                     repeat_last_n: int = 64,   # Not directly supported
                     seed: Optional[int] = None,      # Not directly supported
                     n_threads: Optional[int] = None, # Not applicable
                     ctx_size: int | None = None,     # Determined by model, not settable per-call
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, 
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using the Gemini model.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths or base64 strings.
            system_prompt (str): The system prompt to guide the model.
            ... other LollmsLLMBinding parameters ...

        Returns:
            Union[str, dict]: Generated text or error dictionary.
        """
        if not self.client:
            return {"status": False, "error": "Gemini client not initialized."}

        # Gemini uses 'system_instruction' for GenerativeModel, not part of the regular message list.
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt if system_prompt else None
        )

        generation_config = self.get_generation_config(temperature, top_p, top_k, n_predict)

        # Prepare content for the API call
        content_parts = []
        if split:
            # Note: The 'split' logic for Gemini should ideally build a multi-turn history,
            # but for `generate_text`, we'll treat the last user part as the main prompt.
            discussion_messages = self.split_discussion(prompt, user_keyword, ai_keyword)
            if discussion_messages:
                last_message = discussion_messages[-1]['content']
                content_parts.append(last_message)
            else:
                content_parts.append(prompt)
        else:
            content_parts.append(prompt)

        if images:
            for image_data in images:
                try:
                    if is_image_path(image_data):
                        img = Image.open(image_data)
                    else: # Assume base64
                        img = Image.open(BytesIO(base64.b64decode(image_data)))
                    content_parts.append(img)
                except Exception as e:
                    error_msg = f"Failed to process image: {e}"
                    ASCIIColors.error(error_msg)
                    return {"status": False, "error": error_msg}

        full_response_text = ""
        try:
            response = model.generate_content(
                contents=content_parts,
                generation_config=generation_config,
                stream=stream
            )

            if stream:
                for chunk in response:
                    try:
                        chunk_text = chunk.text
                    except ValueError:
                        # Handle potential empty parts in the stream
                        chunk_text = ""

                    if chunk_text:
                        full_response_text += chunk_text
                        if streaming_callback:
                            if not streaming_callback(chunk_text, MSG_TYPE.MSG_TYPE_CHUNK):
                                break # Callback requested stop
                return full_response_text
            else:
                # Check for safety blocks
                if response.prompt_feedback.block_reason:
                    error_msg = f"Content blocked due to: {response.prompt_feedback.block_reason.name}"
                    ASCIIColors.warning(error_msg)
                    return {"status": False, "error": error_msg}
                return response.text

        except Exception as ex:
            error_message = f"An unexpected error occurred with Gemini API: {str(ex)}"
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
             repeat_penalty: float = 1.1,
             repeat_last_n: int = 64,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Gemini model using a LollmsDiscussion object.
        """
        if not self.client:
             return {"status": "error", "message": "Gemini client not initialized."}

        # 1. Manually export discussion to Gemini's format.
        # Gemini uses 'user' and 'model' roles.
        # The system prompt is handled separately at model initialization.
        system_prompt = discussion.system_prompt
        messages = discussion.get_messages(branch_tip_id)
        
        history = []
        for msg in messages:
            role = 'user' if msg.sender_type == "user" else 'assistant'
            
            # Handle multimodal content in the message
            content_parts = []
            if msg.content:
                content_parts.append(msg.content)
            
            # Check for images associated with this message
            if msg.images:
                for file_path in msg.images:
                     if is_image_path(file_path):
                         try:
                            content_parts.append(Image.open(file_path))
                         except Exception as e:
                            ASCIIColors.warning(f"Could not load image {file_path}: {e}")

            if content_parts:
                history.append({'role': role, 'parts': content_parts})
        
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt
        )
        
        # History must not be empty and should not contain consecutive roles of the same type.
        # We also need to separate the final prompt from the history.
        if not history:
            return {"status": "error", "message": "Cannot start chat with an empty discussion."}

        chat_history = history[:-1] if len(history) > 1 else []
        last_prompt_parts = history[-1]['parts']

        # Ensure history is valid (no consecutive same roles)
        valid_history = []
        if chat_history:
            valid_history.append(chat_history[0])
            for i in range(1, len(chat_history)):
                if chat_history[i]['role'] != chat_history[i-1]['role']:
                    valid_history.append(chat_history[i])
        
        chat_session = model.start_chat(history=valid_history)
        
        generation_config = self.get_generation_config(temperature, top_p, top_k, n_predict)
        
        full_response_text = ""
        try:
            response = chat_session.send_message(
                content=last_prompt_parts,
                generation_config=generation_config,
                stream=stream
            )
            
            if stream:
                for chunk in response:
                    try:
                        chunk_text = chunk.text
                    except ValueError:
                        chunk_text = ""
                        
                    if chunk_text:
                        full_response_text += chunk_text
                        if streaming_callback:
                            if not streaming_callback(chunk_text, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                if response.prompt_feedback.block_reason:
                    error_msg = f"Content blocked due to: {response.prompt_feedback.block_reason.name}"
                    ASCIIColors.warning(error_msg)
                    return {"status": "error", "message": error_msg}
                return response.text

        except Exception as ex:
            error_message = f"An unexpected error occurred with Gemini API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}
            
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text.
        Note: Gemini doesn't expose a public tokenizer API.
        Using tiktoken for a rough estimate, NOT accurate for Gemini.
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
        Count tokens from a text using the Gemini API.
        """
        if not self.client or not self.model_name:
            ASCIIColors.warning("Cannot count tokens, Gemini client or model_name not set.")
            return -1
        try:
            model = self.client.GenerativeModel(self.model_name)
            return model.count_tokens(text).total_tokens
        except Exception as e:
            ASCIIColors.error(f"Failed to count tokens with Gemini API: {e}")
            # Fallback to tiktoken for a rough estimate
            return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text using Gemini API.
        """
        if not self.client:
             raise Exception("Gemini client not initialized.")
        
        # Default to a known Gemini embedding model
        model_to_use = kwargs.get("model", "models/embedding-001")
        
        try:
            response = self.client.embed_content(
                model=model_to_use,
                content=text,
                task_type="retrieval_document" # or "semantic_similarity", etc.
            )
            return response['embedding']
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Gemini embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        """Return information about the current Gemini model setup."""
        return {
            "name": self.binding_name,
            "version": genai.__version__,
            "host_address": "https://generativelanguage.googleapis.com",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": "vision" in self.model_name or "gemini-1.5" in self.model_name,
        }

    def listModels(self) -> List[Dict[str, str]]:
        """Lists available generative models from the Gemini service."""
        if not self.client:
            ASCIIColors.error("Gemini client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing Gemini models...")
            model_info_list = []
            for m in self.client.list_models():
                # We are interested in models that can generate content.
                if 'generateContent' in m.supported_generation_methods:
                    model_info_list.append({
                        'model_name': m.name,
                        'display_name': m.display_name,
                        'description': m.description,
                        'owned_by': 'Google' 
                    })
            return model_info_list
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """Set the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Gemini model set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Example Usage (requires GOOGLE_API_KEY environment variable)
    if 'GOOGLE_API_KEY' not in os.environ:
        ASCIIColors.red("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please get your key from Google AI Studio and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing GeminiBinding ---")

    # --- Configuration ---
    test_model_name = "gemini-1.5-pro-latest"
    test_vision_model_name = "gemini-1.5-pro-latest" # or gemini-pro-vision
    test_embedding_model = "models/embedding-001"
    
    # This variable is global to the script's execution
    full_streamed_text = ""

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = GeminiBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using google-generativeai version: {genai.__version__}")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.listModels()
        if models:
            ASCIIColors.green(f"Found {len(models)} generative models. First 5:")
            for m in models[:5]:
                print(m['model_name'])
        else:
            ASCIIColors.warning("No models found or failed to list models.")

        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Explain the importance of bees in one paragraph."
        ASCIIColors.info(f"Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, n_predict=100, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        
        def stream_callback(chunk: str, msg_type: int):
            # FIX: Use 'global' to modify the variable in the module's scope
            global full_streamed_text
            ASCIIColors.green(chunk, end="", flush=True)
            full_streamed_text += chunk
            return True
        
        # Reset for this test
        full_streamed_text = ""
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, n_predict=150, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        # 'result' is the full text after streaming, which should match our captured text.
        ASCIIColors.green(f"Full streamed text (for verification): {result}")

        # --- Embeddings ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            embedding_text = "Lollms is a cool project."
            embedding_vector = binding.embed(embedding_text, model=test_embedding_model)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.warning(f"Could not get embedding: {e}")

        # --- Vision Model Test ---
        dummy_image_path = "gemini_dummy_test_image.png"
        try:
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

    ASCIIColors.yellow("\nGeminiBinding test finished.")