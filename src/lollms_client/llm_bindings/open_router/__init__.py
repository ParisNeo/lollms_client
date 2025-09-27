import os
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["openai", "pillow", "tiktoken"])

import openai
from PIL import Image, ImageDraw
import tiktoken

BindingName = "OpenRouterBinding"

class OpenRouterBinding(LollmsLLMBinding):
    """
    OpenRouter API binding implementation.

    This binding allows communication with the OpenRouter service, which acts as a
    aggregator for a vast number of AI models from different providers. It uses
    an OpenAI-compatible API structure.
    """
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the OpenRouterBinding.

        Args:
            model_name (str): The name of the model to use from OpenRouter (e.g., 'anthropic/claude-3-haiku-20240307').
            service_key (str): The API key for the OpenRouter service.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name","google/gemini-flash-1.5")
        self.api_key = kwargs.get("service_key") or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set it via 'open_router_api_key' or OPENROUTER_API_KEY env var.")

        try:
            self.client = openai.OpenAI(
                base_url=self.BASE_URL,
                api_key=self.api_key,
            )
        except Exception as e:
            ASCIIColors.error(f"Failed to configure OpenRouter client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure OpenRouter client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              seed: Optional[int]) -> Dict[str, any]:
        """Builds a parameters dictionary for the API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_tokens'] = n_predict
        if seed is not None: params['seed'] = seed
        return params

    def _prepare_messages(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Prepares the message list for the API from a LollmsDiscussion."""
        history = []
        if discussion.system_prompt:
            history.append({"role": "system", "content": discussion.system_prompt})

        for msg in discussion.get_messages(branch_tip_id):
            role = 'user' if msg.sender_type == "user" else 'assistant'
            # Note: Vision support depends on the specific model being called via OpenRouter.
            # We will not implement it in this generic binding to avoid complexity,
            # as different models might expect different formats.
            if msg.content:
                history.append({'role': role, 'content': msg.content})
        return history

    def generate_text(self,
                    prompt: str,
                    images: Optional[List[str]] = None,
                    system_prompt: str = "",
                    n_predict: Optional[int] = None,
                    stream: Optional[bool] = None,
                    temperature: float = 0.7,  # Ollama default is 0.8, common default 0.7
                    top_k: int = 40,          # Ollama default is 40
                    top_p: float = 0.9,       # Ollama default is 0.9
                    repeat_penalty: float = 1.1,  # Ollama default is 1.1
                    repeat_last_n: int = 64,  # Ollama default is 64
                    seed: Optional[int] = None,
                    n_threads: Optional[int] = None,
                    ctx_size: int | None = None,
                    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                    split: Optional[bool] = False,  # put to true if the prompt is a discussion
                    user_keyword: Optional[str] = "!@>user:",
                    ai_keyword: Optional[str] = "!@>assistant:",
                    **kwargs
                    ) -> Union[str, dict]:
        """
        Generate text using OpenRouter. This is a wrapper around the chat method.
        """
        temp_discussion = LollmsDiscussion(None)
        temp_discussion.add_message(sender="user", content=prompt, images=images or [])
        if system_prompt:
            temp_discussion.system_prompt = system_prompt
        
        return self.chat(temp_discussion, 
                        n_predict=n_predict,
                        stream=stream,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        repeat_last_n=repeat_last_n,
                        seed=seed,
                        n_threads=n_threads,
                        ctx_size=ctx_size,
                        streaming_callback=streaming_callback,
                        split=split,
                        user_keyword=user_keyword,
                        ai_keyword=ai_keyword,
                        **kwargs)

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
        Conduct a chat session with a model via OpenRouter.
        """
        if not self.client:
            return {"status": "error", "message": "OpenRouter client not initialized."}

        messages = self._prepare_messages(discussion, branch_tip_id)
        api_params = self._construct_parameters(temperature, top_p, n_predict, seed)
        full_response_text = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream if stream else False,
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
            error_message = f"An unexpected error occurred with OpenRouter API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def tokenize(self, text: str) -> list:
        """Tokenize text using tiktoken as a general-purpose fallback."""
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
        Get embeddings for the input text using an OpenRouter embedding model.
        """
        if not self.client:
             raise Exception("OpenRouter client not initialized.")
        
        # User must specify an embedding model, e.g., 'text-embedding-ada-002'
        embedding_model = kwargs.get("model")
        if not embedding_model:
            raise ValueError("An embedding model name must be provided via the 'model' kwarg for the embed method.")
        
        try:
            # The client is already configured for OpenRouter's base URL
            response = self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"OpenRouter embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        """Return information about the current OpenRouter setup."""
        return {
            "name": self.binding_name,
            "version": openai.__version__,
            "host_address": self.BASE_URL,
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": "Depends on the specific model selected. This generic binding does not support vision.",
        }

    def list_models(self) -> List[Dict[str, str]]:
        """Lists available models from the OpenRouter service."""
        if not self.client:
            ASCIIColors.error("OpenRouter client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing OpenRouter models...")
            models = self.client.models.list()
            model_info_list = []
            for m in models.data:
                model_info_list.append({
                    'model_name': m.id,
                    'display_name': m.name if hasattr(m, 'name') else m.id,
                    'description': m.description if hasattr(m, 'description') else "No description available.",
                    'owned_by': m.id.split('/')[0] # Heuristic to get the provider
                })
            return model_info_list
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """Sets the model name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"OpenRouter model set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Environment variable to set for testing:
    # OPENROUTER_API_KEY: Your OpenRouter API key (starts with sk-or-...)

    if "OPENROUTER_API_KEY" not in os.environ:
        ASCIIColors.red("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please get your key from https://openrouter.ai/keys and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing OpenRouterBinding ---")

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        # Initialize with a fast, cheap, and well-known model
        binding = OpenRouterBinding(model_name="mistralai/mistral-7b-instruct")
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Successfully fetched {len(models)} models from OpenRouter.")
            ASCIIColors.info("Sample of available models:")
            # Print a few examples from different providers
            providers_seen = set()
            count = 0
            for m in models:
                provider = m['owned_by']
                if provider not in providers_seen:
                    print(f"- {m['model_name']}")
                    providers_seen.add(provider)
                    count += 1
                if count >= 5:
                    break
        else:
            ASCIIColors.warning("No models found or failed to list models.")
        
        # --- Text Generation (Testing with a Claude model) ---
        ASCIIColors.cyan("\n--- Text Generation (Claude via OpenRouter) ---")
        binding.load_model("anthropic/claude-3-haiku-20240307")
        prompt_text = "Why is Claude Haiku a good choice for fast-paced chat applications?"
        generated_text = binding.generate_text(prompt_text, n_predict=100, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Text Generation (Streaming with a Groq model) ---
        ASCIIColors.cyan("\n--- Text Generation (Llama3 on Groq via OpenRouter) ---")
        binding.load_model("meta-llama/llama-3-8b-instruct:free") # Use the free tier on OpenRouter
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            ASCIIColors.green(chunk, end="", flush=True)
            full_streamed_text += chunk
            return True
        
        stream_prompt = "Write a very short, 3-line poem about the speed of Groq."
        result = binding.generate_text(stream_prompt, n_predict=50, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")

        # --- Embeddings Test ---
        ASCIIColors.cyan("\n--- Embeddings (OpenAI model via OpenRouter) ---")
        try:
            embedding_model = "openai/text-embedding-ada-002"
            embedding_text = "OpenRouter simplifies everything."
            embedding_vector = binding.embed(embedding_text, model=embedding_model)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.error(f"Embedding test failed: {e}")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nOpenRouterBinding test finished.")
