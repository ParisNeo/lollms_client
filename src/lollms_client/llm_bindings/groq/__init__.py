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
pm.ensure_packages(["groq", "pillow", "tiktoken"])

import groq
from PIL import Image, ImageDraw
import tiktoken

BindingName = "GroqBinding"

class GroqBinding(LollmsLLMBinding):
    """
    Groq API binding implementation.

    This binding allows communication with Groq's LPU-powered inference service,
    known for its high-speed generation. It uses an OpenAI-compatible API structure.
    """

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the GroqBinding.

        Args:
            model_name (str): The name of the Groq model to use.
            service_key (str): The API key for the Groq service.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name", "llama3-8b-8192")
        self.groq_api_key = kwargs.get("service_key") or os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set it via 'groq_api_key' or GROQ_API_KEY env var.")

        try:
            self.client = groq.Groq(api_key=self.groq_api_key)
        except Exception as e:
            ASCIIColors.error(f"Failed to configure Groq client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure Groq client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              seed: Optional[int]) -> Dict[str, any]:
        """Builds a parameters dictionary for the Groq API."""
        params = {}
        # Groq API mirrors OpenAI's parameters
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_tokens'] = n_predict
        if seed is not None: params['seed'] = seed
        return params

    def _prepare_messages(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Prepares the message list for the Groq API from a LollmsDiscussion."""
        history = []
        if discussion.system_prompt:
            history.append({"role": "system", "content": discussion.system_prompt})

        for msg in discussion.get_messages(branch_tip_id):
            role = 'user' if msg.sender_type == "user" else 'assistant'
            # Note: Groq models currently do not support image inputs.
            # We only process the text content.
            if msg.content:
                history.append({'role': role, 'content': msg.content})
        return history

    def generate_text(self, prompt: str, **kwargs) -> Union[str, dict]:
        """
        Generate text using Groq. This is a wrapper around the chat method.
        """
        # Create a temporary discussion to leverage the `chat` method's logic
        temp_discussion = LollmsDiscussion.from_messages([
            LollmsMessage.new_message(sender_type="user", content=prompt)
        ])
        if kwargs.get("system_prompt"):
            temp_discussion.system_prompt = kwargs.get("system_prompt")
        
        return self.chat(temp_discussion, **kwargs)

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = 2048,
             stream: Optional[bool] = False,
             temperature: float = 0.7,
             top_p: float = 0.9,
             seed: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
             **kwargs
             ) -> Union[str, dict]:
        """
        Conduct a chat session with a Groq model.
        """
        if not self.client:
            return {"status": "error", "message": "Groq client not initialized."}

        messages = self._prepare_messages(discussion, branch_tip_id)
        api_params = self._construct_parameters(temperature, top_p, n_predict, seed)
        full_response_text = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                **api_params
            )

            if stream:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response_text += delta
                        if streaming_callback:
                            if not streaming_callback(delta, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                return response.choices[0].message.content

        except Exception as ex:
            error_message = f"An unexpected error occurred with Groq API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def tokenize(self, text: str) -> list:
        """Tokenize text using tiktoken for a rough estimate."""
        try:
            # Most models on Groq (like Llama) use tokenizers that are
            # reasonably approximated by cl100k_base.
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except Exception:
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        """Detokenize tokens using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except Exception:
            return bytes(tokens).decode('utf-8', errors='ignore')

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text using the fallback tokenizer."""
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Groq does not provide an embedding API. This method is not implemented.
        """
        ASCIIColors.warning("Groq does not offer a public embedding API. This method is not implemented.")
        raise NotImplementedError("Groq binding does not support embeddings.")

    def get_model_info(self) -> dict:
        """Return information about the current Groq setup."""
        return {
            "name": self.binding_name,
            "version": groq.__version__,
            "host_address": "https://api.groq.com/openai/v1",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": False, # Groq models do not currently support vision
        }

    def list_models(self) -> List[Dict[str, str]]:
        """Lists available models from the Groq service."""
        if not self.client:
            ASCIIColors.error("Groq client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing Groq models...")
            models = self.client.models.list()
            model_info_list = []
            for m in models.data:
                model_info_list.append({
                    'model_name': m.id,
                    'display_name': m.id.replace('-', ' ').title(),
                    'description': f"Context window: {m.context_window}, Active: {m.active}",
                    'owned_by': m.owned_by
                })
            return model_info_list
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """Sets the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Groq model set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Environment variable to set for testing:
    # GROQ_API_KEY: Your Groq API key

    if "GROQ_API_KEY" not in os.environ:
        ASCIIColors.red("Error: GROQ_API_KEY environment variable not set.")
        print("Please get your key from https://console.groq.com/keys and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing GroqBinding ---")

    # Use a fast and common model for testing
    test_model_name = "llama3-8b-8192"

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = GroqBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using groq library version: {groq.__version__}")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Found {len(models)} models on Groq. Available models:")
            for m in models:
                print(f"- {m['model_name']} (owned by {m['owned_by']})")
        else:
            ASCIIColors.warning("No models found or failed to list models.")
        
        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "The quick brown fox jumps over the lazy dog."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "What is the capital of France? Be concise."
        generated_text = binding.generate_text(prompt_text, n_predict=20, stream=False)
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
        
        stream_prompt = "Write a very short, 3-line poem about speed."
        result = binding.generate_text(stream_prompt, n_predict=50, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")

        # --- Embeddings Test ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            binding.embed("This should fail.")
        except NotImplementedError as e:
            ASCIIColors.green(f"Successfully caught expected error for embeddings: {e}")
        except Exception as e:
            ASCIIColors.error(f"Caught an unexpected error for embeddings: {e}")

        # --- Vision Test (should be unsupported) ---
        ASCIIColors.cyan("\n--- Vision Test (Expecting No Support) ---")
        model_info = binding.get_model_info()
        if not model_info.get("supports_vision"):
            ASCIIColors.green("Binding correctly reports no support for vision.")
        else:
            ASCIIColors.warning("Binding reports support for vision, which is unexpected for Groq.")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nGroqBinding test finished.")