import os
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["mistralai", "pillow", "tiktoken"])

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from PIL import Image, ImageDraw
import tiktoken

BindingName = "MistralBinding"

class MistralBinding(LollmsLLMBinding):
    """
    Mistral AI API binding implementation.

    This binding allows communication with Mistral's API for both their
    open-weight and proprietary models.
    """

    def __init__(self,
                 model_name: str = "mistral-large-latest",
                 mistral_api_key: str = None,
                 **kwargs
                 ):
        """
        Initialize the MistralBinding.

        Args:
            model_name (str): The name of the Mistral model to use.
            mistral_api_key (str): The API key for the Mistral service.
        """
        super().__init__(binding_name=BindingName)
        self.model_name = model_name
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")

        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required. Set it via 'mistral_api_key' or MISTRAL_API_KEY env var.")

        try:
            self.client = MistralClient(api_key=self.mistral_api_key)
        except Exception as e:
            ASCIIColors.error(f"Failed to configure Mistral client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure Mistral client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              seed: Optional[int]) -> Dict[str, any]:
        """Builds a parameters dictionary for the Mistral API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_tokens'] = n_predict
        if seed is not None: params['random_seed'] = seed # Mistral uses 'random_seed'
        return params

    def _prepare_messages(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None) -> List[ChatMessage]:
        """Prepares the message list for the Mistral API from a LollmsDiscussion."""
        history = []
        if discussion.system_prompt:
            # Mistral prefers the system prompt as the first message with a user/assistant turn.
            # A lone system message is not ideal. We will prepend it to the first user message.
            # However, for API consistency, we will treat it as a separate message if it exists.
            # The official client will likely handle this.
            history.append(ChatMessage(role="system", content=discussion.system_prompt))

        for msg in discussion.get_messages(branch_tip_id):
            role = 'user' if msg.sender_type == "user" else 'assistant'
            # Note: Mistral API currently does not support image inputs via the chat endpoint.
            if msg.content:
                history.append(ChatMessage(role=role, content=msg.content))
        return history

    def generate_text(self, prompt: str, **kwargs) -> Union[str, dict]:
        """
        Generate text using Mistral. This is a wrapper around the chat method.
        """
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
        Conduct a chat session with a Mistral model.
        """
        if not self.client:
            return {"status": "error", "message": "Mistral client not initialized."}

        messages = self._prepare_messages(discussion, branch_tip_id)
        api_params = self._construct_parameters(temperature, top_p, n_predict, seed)
        full_response_text = ""

        try:
            if stream:
                response = self.client.chat_stream(
                    model=self.model_name,
                    messages=messages,
                    **api_params
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response_text += delta
                        if streaming_callback:
                            if not streaming_callback(delta, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    **api_params
                )
                return response.choices[0].message.content

        except Exception as ex:
            error_message = f"An unexpected error occurred with Mistral API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def tokenize(self, text: str) -> list:
        """Tokenize text using tiktoken as a fallback."""
        try:
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
        Get embeddings for the input text using the Mistral embedding API.
        """
        if not self.client:
            raise Exception("Mistral client not initialized.")
        
        # Default to the recommended embedding model
        model_to_use = kwargs.get("model", "mistral-embed")

        try:
            response = self.client.embeddings(
                model=model_to_use,
                input=[text] # API expects a list of strings
            )
            return response.data[0].embedding
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Mistral embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        """Return information about the current Mistral setup."""
        return {
            "name": self.binding_name,
            "version": "unknown", # mistralai library doesn't expose a version attribute easily
            "host_address": "https://api.mistral.ai",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": False, # Mistral API does not currently support vision
        }

    def listModels(self) -> List[Dict[str, str]]:
        """Lists available models from the Mistral service."""
        if not self.client:
            ASCIIColors.error("Mistral client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing Mistral models...")
            models = self.client.list_models()
            model_info_list = []
            for m in models.data:
                model_info_list.append({
                    'model_name': m.id,
                    'display_name': m.id.replace('-', ' ').title(),
                    'description': f"Owned by: {m.owned_by}",
                    'owned_by': m.owned_by
                })
            return model_info_list
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """Sets the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Mistral model set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Environment variable to set for testing:
    # MISTRAL_API_KEY: Your Mistral API key

    if "MISTRAL_API_KEY" not in os.environ:
        ASCIIColors.red("Error: MISTRAL_API_KEY environment variable not set.")
        print("Please get your key from https://console.mistral.ai/api-keys/ and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing MistralBinding ---")

    test_model_name = "mistral-small-latest" # Use a smaller, faster model for testing
    test_embedding_model = "mistral-embed"
    
    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = MistralBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.listModels()
        if models:
            ASCIIColors.green(f"Found {len(models)} models on Mistral. Available models:")
            for m in models:
                print(f"- {m['model_name']}")
        else:
            ASCIIColors.warning("No models found or failed to list models.")
        
        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Who developed the transformer architecture and in what paper?"
        generated_text = binding.generate_text(prompt_text, n_predict=100, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            nonlocal full_streamed_text
            ASCIIColors.green(chunk, end="", flush=True)
            full_streamed_text += chunk
            return True
        
        result = binding.generate_text(prompt_text, n_predict=150, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")

        # --- Embeddings Test ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            embedding_text = "Mistral AI is based in Paris."
            embedding_vector = binding.embed(embedding_text, model=test_embedding_model)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.error(f"Embedding test failed: {e}")

        # --- Vision Test (should be unsupported) ---
        ASCIIColors.cyan("\n--- Vision Test (Expecting No Support) ---")
        model_info = binding.get_model_info()
        if not model_info.get("supports_vision"):
            ASCIIColors.green("Binding correctly reports no support for vision.")
        else:
            ASCIIColors.warning("Binding reports support for vision, which is unexpected for Mistral.")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nMistralBinding test finished.")