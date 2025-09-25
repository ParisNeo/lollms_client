import os
import json
import requests
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["requests", "tiktoken"])

import tiktoken

BindingName = "NovitaAIBinding"
API_BASE_URL = "https://api.novita.ai"

# A hardcoded list of models based on Novita AI's documentation.
# The API is OpenAI-compatible but does not provide a models listing endpoint.
# Sourced from: https://docs.novita.ai/language-model/models
_FALLBACK_MODELS = [
    {'model_name': 'meta-llama/Llama-3-8B-Instruct', 'display_name': 'Llama 3 8B Instruct', 'description': 'Meta\'s Llama 3 8B instruction-tuned model.', 'owned_by': 'Meta'},
    {'model_name': 'meta-llama/Llama-3-70B-Instruct', 'display_name': 'Llama 3 70B Instruct', 'description': 'Meta\'s Llama 3 70B instruction-tuned model.', 'owned_by': 'Meta'},
    {'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'display_name': 'Mixtral 8x7B Instruct', 'description': 'Mistral AI\'s Mixtral 8x7B instruction-tuned model.', 'owned_by': 'Mistral AI'},
    {'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'display_name': 'Mistral 7B Instruct v0.2', 'description': 'Mistral AI\'s 7B instruction-tuned model.', 'owned_by': 'Mistral AI'},
    {'model_name': 'google/gemma-7b-it', 'display_name': 'Gemma 7B IT', 'description': 'Google\'s Gemma 7B instruction-tuned model.', 'owned_by': 'Google'},
    {'model_name': 'google/gemma-2-9b-it', 'display_name': 'Gemma 2 9B IT', 'description': 'Google\'s next-generation Gemma 2 9B instruction-tuned model.', 'owned_by': 'Google'},
    {'model_name': 'deepseek-ai/deepseek-coder-33b-instruct', 'display_name': 'Deepseek Coder 33B Instruct', 'description': 'A powerful coding model from Deepseek AI.', 'owned_by': 'Deepseek AI'},
]

class NovitaAIBinding(LollmsLLMBinding):
    """Novita AI-specific binding implementation using their OpenAI-compatible API."""

    def __init__(self, **kwargs):
        """
        Initialize the Novita AI binding.

        Args:
            model_name (str): Name of the Novita AI model to use.
            service_key (str): Novita AI API key.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")

        if not self.service_key:
            self.service_key = os.getenv("NOVITA_API_KEY")

        if not self.service_key:
            raise ValueError("Novita AI API key is required. Please set it via the 'service_key' parameter or the NOVITA_API_KEY environment variable.")

        self.headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              presence_penalty: float,
                              frequency_penalty: float) -> Dict[str, any]:
        """Builds a parameters dictionary for the Novita AI API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_tokens'] = n_predict
        if presence_penalty is not None: params['presence_penalty'] = presence_penalty
        if frequency_penalty is not None: params['frequency_penalty'] = frequency_penalty
        return params

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = 2048,
             stream: Optional[bool] = False,
             temperature: float = 0.7,
             top_k: int = 50, # Not supported by Novita API
             top_p: float = 0.9,
             repeat_penalty: float = 1.1, # maps to frequency_penalty
             presence_penalty: Optional[float] = 0.0,
             seed: Optional[int] = None, # Not supported
             n_threads: Optional[int] = None, # Not applicable
             ctx_size: Optional[int] = None, # Determined by model
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session with a Novita AI model using a LollmsDiscussion object.
        """
        system_prompt = discussion.system_prompt
        messages = discussion.get_messages(branch_tip_id)

        history = []
        if system_prompt and system_prompt.strip():
            history.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = 'user' if msg.sender_type == "user" else 'assistant'
            
            if msg.images:
                ASCIIColors.warning("Novita AI API does not support images. They will be ignored.")

            if msg.content and msg.content.strip():
                history.append({"role": role, "content": msg.content})

        if not history:
            return {"status": "error", "message": "Cannot start chat with an empty discussion."}

        api_params = self._construct_parameters(
            temperature, top_p, n_predict, presence_penalty, repeat_penalty
        )
        
        payload = {
            "model": self.model_name,
            "messages": history,
            "stream": stream,
            **api_params
        }
        
        url = f"{API_BASE_URL}/v1/chat/completions"
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
        except requests.exceptions.HTTPError as e:
            try:
                error_details = e.response.json()
                error_message = error_details.get("error", {}).get("message", e.response.text)
            except json.JSONDecodeError:
                error_message = e.response.text
            ASCIIColors.error(f"HTTP Error received from Novita AI API: {e.response.status_code} - {error_message}")
            return {"status": "error", "message": f"HTTP Error: {e.response.status_code} - {error_message}"}
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred with the Novita AI API: {e}"
            trace_exception(e)
            return {"status": "error", "message": str(e)}

    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text. Novita uses an OpenAI-compatible API,
        so we use the same tokenizer as GPT-4.
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
        ASCIIColors.warning("Novita AI does not offer a public embedding API via this binding. This method is not implemented.")
        raise NotImplementedError("Novita AI binding does not support embeddings.")

    def get_model_info(self) -> dict:
        """Return information about the current model setup."""
        return {
            "name": self.binding_name,
            "host_address": API_BASE_URL,
            "model_name": self.model_name,
            "supports_vision": False
        }

    def listModels(self) -> List[Dict[str, str]]:
        """
        Lists available models. Novita AI API does not have a models endpoint,
        so a hardcoded list from their documentation is returned.
        """
        return sorted(_FALLBACK_MODELS, key=lambda x: x['display_name'])

    def load_model(self, model_name: str) -> bool:
        """Set the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Novita AI model set to: {model_name}.")
        return True

if __name__ == '__main__':
    if 'NOVITA_API_KEY' not in os.environ:
        ASCIIColors.red("Error: NOVITA_API_KEY environment variable not set.")
        print("Please get your key from novita.ai and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing NovitaAIBinding ---")
    
    test_model_name = "meta-llama/Llama-3-8B-Instruct"

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = NovitaAIBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models (static list) ---")
        models = binding.listModels()
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
                {"sender":"user", "content": "What is the largest planet in our solar system?"}
            ],
            system_prompt="You are a helpful and concise astronomical assistant."
        )
        ASCIIColors.info(f"Prompt: What is the largest planet in our solar system?")
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
                {"sender":"user", "content": "Explain the concept of photosynthesis in one short paragraph."}
            ]
        )
        ASCIIColors.info(f"Prompt: Explain the concept of photosynthesis in one short paragraph.")
        result = binding.chat(
            discussion_stream, 
            n_predict=150, 
            stream=True, 
            streaming_callback=stream_callback
        )
        print("\n--- End of Stream ---")
        full_streamed_text = "".join(captured_chunks)
        assert result == full_streamed_text

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nNovitaAIBinding test finished.")