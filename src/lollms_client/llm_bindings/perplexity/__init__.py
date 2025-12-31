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
pm.ensure_packages(["requests", "tiktoken"])

import tiktoken

BindingName = "PerplexityBinding"
API_BASE_URL = "https://api.perplexity.ai"

# A hardcoded list of models based on Perplexity's documentation
# The API does not provide a models listing endpoint.
# Sourced from: https://docs.perplexity.ai/docs/models
_FALLBACK_MODELS = [
    # Sonar Models
    {'model_name': 'llama-3.1-sonar-small-128k-chat', 'display_name': 'Llama 3.1 Sonar Small Chat (128k)', 'description': 'Fast and cost-effective conversational model.', 'owned_by': 'Perplexity'},
    {'model_name': 'llama-3.1-sonar-small-128k-online', 'display_name': 'Llama 3.1 Sonar Small Online (128k)', 'description': 'Fast and cost-effective conversational model with web access.', 'owned_by': 'Perplexity'},
    {'model_name': 'llama-3.1-sonar-large-128k-chat', 'display_name': 'Llama 3.1 Sonar Large Chat (128k)', 'description': 'State-of-the-art conversational model.', 'owned_by': 'Perplexity'},
    {'model_name': 'llama-3.1-sonar-large-128k-online', 'display_name': 'Llama 3.1 Sonar Large Online (128k)', 'description': 'State-of-the-art conversational model with web access.', 'owned_by': 'Perplexity'},
    # Llama 3 Instruct Models
    {'model_name': 'llama-3-8b-instruct', 'display_name': 'Llama 3 8B Instruct', 'description': 'Meta\'s Llama 3 8B instruction-tuned model.', 'owned_by': 'Meta'},
    {'model_name': 'llama-3-70b-instruct', 'display_name': 'Llama 3 70B Instruct', 'description': 'Meta\'s Llama 3 70B instruction-tuned model.', 'owned_by': 'Meta'},
    # Mixtral Model
    {'model_name': 'mixtral-8x7b-instruct', 'display_name': 'Mixtral 8x7B Instruct', 'description': 'Mistral AI\'s Mixtral 8x7B instruction-tuned model.', 'owned_by': 'Mistral AI'},
    # Legacy Sonar Models
    {'model_name': 'sonar-small-32k-chat', 'display_name': 'Sonar Small Chat (32k)', 'description': 'Legacy small conversational model.', 'owned_by': 'Perplexity'},
    {'model_name': 'sonar-small-32k-online', 'display_name': 'Sonar Small Online (32k)', 'description': 'Legacy small conversational model with web access.', 'owned_by': 'Perplexity'},
    {'model_name': 'sonar-medium-32k-chat', 'display_name': 'Sonar Medium Chat (32k)', 'description': 'Legacy medium conversational model.', 'owned_by': 'Perplexity'},
    {'model_name': 'sonar-medium-32k-online', 'display_name': 'Sonar Medium Online (32k)', 'description': 'Legacy medium conversational model with web access.', 'owned_by': 'Perplexity'},
]

class PerplexityBinding(LollmsLLMBinding):
    """Perplexity AI-specific binding implementation."""

    def __init__(self, **kwargs):
        """
        Initialize the Perplexity binding.

        Args:
            model_name (str): Name of the Perplexity model to use.
            service_key (str): Perplexity API key.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")

        if not self.service_key:
            self.service_key = os.getenv("PERPLEXITY_API_KEY")

        if not self.service_key:
            raise ValueError("Perplexity API key is required. Please set it via the 'service_key' parameter or the PERPLEXITY_API_KEY environment variable.")

        self.headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              top_k: int,
                              n_predict: int,
                              presence_penalty: float,
                              frequency_penalty: float) -> Dict[str, any]:
        """Builds a parameters dictionary for the Perplexity API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if top_k is not None: params['top_k'] = top_k
        if n_predict is not None: params['max_tokens'] = n_predict
        if presence_penalty is not None: params['presence_penalty'] = presence_penalty
        if frequency_penalty is not None: params['frequency_penalty'] = frequency_penalty
        return params

    def _chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = 2048,
             stream: Optional[bool] = False,
             temperature: float = 0.7,
             top_k: int = 50,
             top_p: float = 0.9,
             repeat_penalty: float = 1.1, # maps to frequency_penalty
             presence_penalty: Optional[float] = 0.0,
             seed: Optional[int] = None, # Not supported
             n_threads: Optional[int] = None, # Not applicable
             ctx_size: Optional[int] = None, # Determined by model
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Perplexity model using a LollmsDiscussion object.
        """
        system_prompt = discussion.system_prompt
        messages = discussion.get_messages(branch_tip_id)

        history = []
        if system_prompt and system_prompt.strip():
            history.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.sender_type == "user":
                role = "user"
            else:
                role = "assistant"
            
            if msg.images:
                ASCIIColors.warning("Perplexity API does not support images. They will be ignored.")

            if msg.content and msg.content.strip():
                history.append({"role": role, "content": msg.content})

        if not history:
            return {"status": "error", "message": "Cannot start chat with an empty discussion."}

        api_params = self._construct_parameters(
            temperature, top_p, top_k, n_predict, presence_penalty, repeat_penalty
        )
        
        payload = {
            "model": self.model_name,
            "messages": history,
            "stream": stream,
            **api_params
        }
        
        url = f"{API_BASE_URL}/chat/completions"
        full_response_text = ""

        try:
            if stream:
                with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data:"):
                                content = decoded_line[len("data: "):].strip()
                                if content == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(content)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    text_chunk = delta.get("content", "")
                                    if text_chunk:
                                        full_response_text += text_chunk
                                        if streaming_callback:
                                            if not streaming_callback(text_chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                                                break
                                except json.JSONDecodeError:
                                    ASCIIColors.error(f"Failed to decode JSON chunk: {content}")
                                    continue
                return full_response_text
            else:
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred with the Perplexity API: {e}"
            trace_exception(e)
            return {"status": "error", "message": str(e)}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text. Perplexity uses the same tokenizer as GPT-4.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except Exception as e:
            ASCIIColors.error(f"Could not use tiktoken, falling back to simple encoding: {e}")
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize a list of tokens.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except Exception as e:
            ASCIIColors.error(f"Could not use tiktoken, falling back to simple decoding: {e}")
            return bytes(tokens).decode('utf-8', errors='ignore')

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.
        """
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text.
        """
        ASCIIColors.warning("Perplexity does not offer a public embedding API. This method is not implemented.")
        raise NotImplementedError("Perplexity binding does not support embeddings.")

    def get_model_info(self) -> dict:
        """Return information about the current model setup."""
        return {
            "name": self.binding_name,
            "version": "1.0",
            "host_address": API_BASE_URL,
            "model_name": self.model_name,
            "supports_vision": False,
            "supports_structured_output": False
        }

    def list_models(self) -> List[Dict[str, str]]:
        """
        Lists available models. Perplexity API does not have a models endpoint,
        so a hardcoded list is returned.
        """
        return sorted(_FALLBACK_MODELS, key=lambda x: x['display_name'])

    def load_model(self, model_name: str) -> bool:
        """Set the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Perplexity model set to: {model_name}.")
        return True

if __name__ == '__main__':
    if 'PERPLEXITY_API_KEY' not in os.environ:
        ASCIIColors.red("Error: PERPLEXITY_API_KEY environment variable not set.")
        print("Please get your key from Perplexity AI and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing PerplexityBinding ---")
    
    test_model_name = "llama-3.1-sonar-small-128k-online"

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = PerplexityBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models (static list) ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Found {len(models)} models.")
            for m in models:
                print(f"- {m['model_name']} ({m['display_name']})")
        else:
            ASCIIColors.error("Failed to list models.")

        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        # --- Chat (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Chat (Non-Streaming) ---")
        discussion_non_stream = LollmsDiscussion.from_messages(
            messages=[
                {"sender":"user", "content": "What is the capital of France?"}
            ],
            system_prompt="You are a helpful and concise assistant."
        )
        ASCIIColors.info(f"Prompt: What is the capital of France?")
        generated_text = binding.chat(discussion_non_stream, n_predict=50, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Chat (Streaming) ---
        ASCIIColors.cyan("\n--- Chat (Streaming) ---")
        
        captured_chunks = []
        def stream_callback(chunk: str, msg_type: int):
            ASCIIColors.green(chunk, end="", flush=True)
            captured_chunks.append(chunk)
            return True
        
        discussion_stream = LollmsDiscussion.from_messages(
            messages=[
                {"sender":"user", "content": "Explain the importance of bees in one short paragraph."}
            ],
            system_prompt="You are a helpful assistant."
        )
        ASCIIColors.info(f"Prompt: Explain the importance of bees in one short paragraph.")
        result = binding.chat(
            discussion_stream, 
            n_predict=150, 
            stream=True, 
            streaming_callback=stream_callback
        )
        print("\n--- End of Stream ---")
        full_streamed_text = "".join(captured_chunks)
        assert result == full_streamed_text

        # --- Embeddings (Expected to fail) ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            binding.embed("This should not work.")
        except NotImplementedError as e:
            ASCIIColors.green(f"Successfully caught expected error for embeddings: {e}")
        except Exception as e:
            ASCIIColors.error(f"Caught an unexpected error for embeddings: {e}")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nPerplexityBinding test finished.")