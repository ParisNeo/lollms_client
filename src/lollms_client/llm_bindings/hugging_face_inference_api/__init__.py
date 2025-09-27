import os
from typing import Optional, Callable, List, Union, Dict

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

import pipmaster as pm

# Ensure the required packages are installed
pm.ensure_packages(["huggingface_hub", "tiktoken"])

from huggingface_hub import HfApi, InferenceClient
import tiktoken

BindingName = "HuggingFaceInferenceAPIBinding"

class HuggingFaceInferenceAPIBinding(LollmsLLMBinding):
    """
    Hugging Face Inference API binding implementation.

    This binding communicates with the Hugging Face serverless Inference API,
    allowing access to thousands of models hosted on the Hub.
    """

    def __init__(self,
                 **kwargs
                 ):
        """
        Initialize the HuggingFaceInferenceAPIBinding.

        Args:
            model_name (str): The repository ID of the model on the Hugging Face Hub.
            service_key (str): The Hugging Face API key.
        """
        super().__init__(BindingName, **kwargs)
        self.model_name = kwargs.get("model_name")
        self.hf_api_key = kwargs.get("service_key") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        if not self.hf_api_key:
            raise ValueError("Hugging Face API key is required. Set it via 'hf_api_key' or HUGGING_FACE_HUB_TOKEN env var.")

        try:
            self.client = InferenceClient(model=self.model_name, token=self.hf_api_key)
            self.hf_api = HfApi(token=self.hf_api_key)
        except Exception as e:
            ASCIIColors.error(f"Failed to configure Hugging Face client: {e}")
            self.client = None
            self.hf_api = None
            raise ConnectionError(f"Could not configure Hugging Face client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              repeat_penalty: float,
                              seed: Optional[int]) -> Dict[str, any]:
        """Builds a parameters dictionary for the HF Inference API."""
        params = {"details": False, "do_sample": True}
        if temperature is not None and temperature > 0:
            params['temperature'] = float(temperature)
        else:
            # A temperature of 0 can cause issues, a small epsilon is better
            params['temperature'] = 0.001
            params['do_sample'] = False


        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_new_tokens'] = n_predict
        if repeat_penalty is not None: params['repetition_penalty'] = repeat_penalty
        if seed is not None: params['seed'] = seed
        return params

    def _format_chat_prompt(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None) -> str:
        """
        Formats a discussion into a single prompt string, attempting to use the model's chat template.
        """
        messages = []
        if discussion.system_prompt:
             messages.append({"role": "system", "content": discussion.system_prompt})

        for msg in discussion.get_messages(branch_tip_id):
            role = 'user' if msg.sender_type == "user" else 'assistant'
            messages.append({"role": role, "content": msg.content})

        try:
            # This is the preferred way, as it respects the model's specific formatting.
            return self.client.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for models without a chat template or if the client fails to fetch it
            ASCIIColors.warning("Could not apply chat template. Using generic formatting.")
            full_prompt = ""
            if discussion.system_prompt:
                full_prompt += f"<|system|>\n{discussion.system_prompt}\n"
            for msg in messages:
                if msg['role'] == 'user':
                    full_prompt += f"<|user|>\n{msg['content']}\n"
                else:
                    full_prompt += f"<|assistant|>\n{msg['content']}\n"
            full_prompt += "<|assistant|>\n"
            return full_prompt

    def generate_text(self,
                     prompt: str,
                     n_predict: Optional[int] = 1024,
                     stream: Optional[bool] = False,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1,
                     seed: Optional[int] = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     **kwargs
                     ) -> Union[str, dict]:
        """
        Generate text using the Hugging Face Inference API.
        """
        if not self.client:
            return {"status": "error", "message": "HF Inference client not initialized."}

        api_params = self._construct_parameters(temperature, top_p, n_predict, repeat_penalty, seed)
        full_response_text = ""

        try:
            if stream:
                for chunk in self.client.text_generation(prompt, stream=True, **api_params):
                    full_response_text += chunk
                    if streaming_callback:
                        if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                return full_response_text
            else:
                return self.client.text_generation(prompt, **api_params)

        except Exception as ex:
            error_message = f"An unexpected error occurred with HF Inference API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def chat(self, discussion: LollmsDiscussion, **kwargs) -> Union[str, dict]:
        """
        Conduct a chat session using the Inference API by formatting the discussion into a single prompt.
        """
        prompt = self._format_chat_prompt(discussion)
        return self.generate_text(prompt, **kwargs)

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
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings using a dedicated sentence-transformer model from the Inference API.
        """
        if not self.client:
            raise Exception("HF Inference client not initialized.")
        
        # User should specify a sentence-transformer model
        embedding_model = kwargs.get("model")
        if not embedding_model:
            raise ValueError("A sentence-transformer model ID must be provided via the 'model' kwarg.")

        try:
            # This is a different endpoint on the InferenceClient
            response = self.client.feature_extraction(text, model=embedding_model)
            # The output for many models is a nested list, we need the first element.
            if isinstance(response, list) and isinstance(response[0], list):
                return response[0]
            return response
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"HF Inference API embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        return {
            "name": self.binding_name,
            "version": "unknown",
            "host_address": "https://api-inference.huggingface.co",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": False, # Vision models use a different API call
        }

    def list_models(self) -> List[Dict[str, str]]:
        """Lists text-generation models from the Hugging Face Hub."""
        if not self.hf_api:
            ASCIIColors.error("HF API client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing Hugging Face text-generation models...")
            # We filter for the 'text-generation' pipeline tag
            models = self.hf_api.list_models(filter="text-generation", sort="downloads", direction=-1, limit=100)
            model_info_list = []
            for m in models:
                model_info_list.append({
                    'model_name': m.modelId,
                    'display_name': m.modelId,
                    'description': f"Downloads: {m.downloads}, Likes: {m.likes}",
                    'owned_by': m.author or "Hugging Face Community"
                })
            return model_info_list
        except Exception as ex:
            trace_exception(ex)
            return []

    def load_model(self, model_name: str) -> bool:
        """Sets the model for subsequent operations and re-initializes the client."""
        self.model_name = model_name
        try:
            self.client = InferenceClient(model=self.model_name, token=self.hf_api_key)
            ASCIIColors.info(f"Hugging Face model set to: {model_name}. It will be used on the next API call.")
            return True
        except Exception as e:
            ASCIIColors.error(f"Failed to re-initialize client for model {model_name}: {e}")
            self.client = None
            return False

if __name__ == '__main__':
    # Environment variable to set for testing:
    # HUGGING_FACE_HUB_TOKEN: Your Hugging Face API key with read access.

    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        ASCIIColors.red("Error: HUGGING_FACE_HUB_TOKEN environment variable not set.")
        print("Please get your token from https://huggingface.co/settings/tokens and set it.")
        exit(1)

    ASCIIColors.yellow("--- Testing HuggingFaceInferenceAPIBinding ---")

    test_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    test_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = HuggingFaceInferenceAPIBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.list_models()
        if models:
            ASCIIColors.green(f"Successfully fetched {len(models)} text-generation models.")
            ASCIIColors.info("Top 5 most downloaded models:")
            for m in models[:5]:
                print(f"- {m['model_name']} ({m['description']})")
        else:
            ASCIIColors.warning("No models found or failed to list models.")
        
        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "In a world where AI companions are common, a detective is assigned a new AI partner. Write the first paragraph of their first meeting."
        ASCIIColors.info("Waiting for model to load (this might take a moment for cold starts)...")
        generated_text = binding.generate_text(prompt_text, n_predict=150, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text:\n{generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        # --- Chat (Streaming) ---
        ASCIIColors.cyan("\n--- Chat (Streaming) ---")
        chat_discussion = LollmsDiscussion.from_messages([
            LollmsMessage.new_message(sender_type="system", content="You are a helpful and pirate-themed assistant named Captain Coder."),
            LollmsMessage.new_message(sender_type="user", content="Ahoy there! Tell me, what be the best language for a scallywag to learn for data science?"),
        ])
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            nonlocal full_streamed_text
            ASCIIColors.green(chunk, end="", flush=True)
            full_streamed_text += chunk
            return True
        
        result = binding.chat(chat_discussion, n_predict=100, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        ASCIIColors.green(f"Full streamed text (for verification): {result}")

        # --- Embeddings Test ---
        ASCIIColors.cyan("\n--- Embeddings ---")
        try:
            embedding_text = "Hugging Face is the home of open-source AI."
            embedding_vector = binding.embed(embedding_text, model=test_embedding_model)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.error(f"Embedding test failed: {e}")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nHuggingFaceInferenceAPIBinding test finished.")