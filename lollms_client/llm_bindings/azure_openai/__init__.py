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
pm.ensure_packages(["openai", "pillow", "tiktoken"])

import openai
from PIL import Image, ImageDraw
import tiktoken

BindingName = "AzureOpenAIBinding"

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
    # While OpenAI supports various types, it's often safest to send common ones.
    # We don't need to be as specific as Claude's API.
    return "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

class AzureOpenAIBinding(LollmsLLMBinding):
    """
    Microsoft Azure OpenAI-specific binding implementation.

    This binding connects to an Azure OpenAI deployment. It requires
    the Azure endpoint, API key, and the specific deployment name.
    """

    def __init__(self,
                 model_name: str, # In Azure, this is the DEPLOYMENT NAME
                 azure_api_key: str = None,
                 azure_endpoint: str = None,
                 azure_api_version: str = "2024-02-01",
                 **kwargs
                 ):
        """
        Initialize the AzureOpenAIBinding.

        Args:
            model_name (str): The name of the Azure OpenAI DEPLOYMENT to use.
            azure_api_key (str): The API key for the Azure OpenAI service.
            azure_endpoint (str): The endpoint URL for the Azure OpenAI service.
            azure_api_version (str): The API version to use.
        """
        super().__init__(binding_name=BindingName)
        self.model_name = model_name  # Here, it's the deployment name
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not self.model_name:
            raise ValueError("Azure deployment name ('model_name') is required.")
        if not self.azure_api_key:
            raise ValueError("Azure API key is required. Set it via 'azure_api_key' or AZURE_OPENAI_API_KEY env var.")
        if not self.azure_endpoint:
            raise ValueError("Azure endpoint is required. Set it via 'azure_endpoint' or AZURE_OPENAI_ENDPOINT env var.")

        try:
            self.client = openai.AzureOpenAI(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        except Exception as e:
            ASCIIColors.error(f"Failed to configure AzureOpenAI client: {e}")
            self.client = None
            raise ConnectionError(f"Could not configure AzureOpenAI client: {e}") from e

    def _construct_parameters(self,
                              temperature: float,
                              top_p: float,
                              n_predict: int,
                              seed: Optional[int]) -> Dict[str, any]:
        """Builds a parameters dictionary for the OpenAI API."""
        params = {}
        if temperature is not None: params['temperature'] = float(temperature)
        if top_p is not None: params['top_p'] = top_p
        if n_predict is not None: params['max_tokens'] = n_predict
        if seed is not None: params['seed'] = seed
        return params

    def _prepare_messages(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Prepares the message list for the OpenAI API from a LollmsDiscussion."""
        history = []
        if discussion.system_prompt:
            history.append({"role": "system", "content": discussion.system_prompt})

        for msg in discussion.get_messages(branch_tip_id):
            role = 'user' if msg.sender_type == "user" else 'assistant'
            
            content_parts = []
            if msg.content:
                content_parts.append({"type": "text", "text": msg.content})
            
            if msg.images:
                for file_path in msg.images:
                    if is_image_path(file_path):
                        try:
                            with open(file_path, "rb") as image_file:
                                b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{get_media_type(file_path)};base64,{b64_data}"}
                            })
                        except Exception as e:
                            ASCIIColors.warning(f"Could not load image {file_path}: {e}")
            
            if content_parts:
                history.append({'role': role, 'content': content_parts})
        return history

    def generate_text(self, prompt: str, **kwargs) -> Union[str, dict]:
        """
        Generate text using an Azure OpenAI deployment. This is a wrapper around the chat method.
        Note: The 'chat' method is preferred for multi-turn conversations.
        """
        # Create a temporary discussion to leverage the `chat` method's logic
        temp_discussion = LollmsDiscussion.from_messages([
            LollmsMessage.new_message(sender_type="user", content=prompt, images=kwargs.get("images"))
        ])
        if kwargs.get("system_prompt"):
            temp_discussion.system_prompt = kwargs.get("system_prompt")
        
        # Pass all relevant kwargs to the chat method
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
        Conduct a chat session with the Azure OpenAI deployment.
        """
        if not self.client:
            return {"status": "error", "message": "AzureOpenAI client not initialized."}

        messages = self._prepare_messages(discussion, branch_tip_id)
        api_params = self._construct_parameters(temperature, top_p, n_predict, seed)
        full_response_text = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # This must be the DEPLOYMENT NAME
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
            error_message = f"An unexpected error occurred with Azure OpenAI API: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}

    def tokenize(self, text: str) -> list:
        """Tokenize text using tiktoken, the tokenizer used by OpenAI models."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except Exception:
            # Fallback for when tiktoken is not available
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        """Detokenize tokens using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except Exception:
            return bytes(tokens).decode('utf-8', errors='ignore')
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text using tiktoken."""
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Get embeddings for the input text using an Azure OpenAI embedding deployment.
        """
        if not self.client:
             raise Exception("AzureOpenAI client not initialized.")
        
        # The embedding deployment name must be passed via kwargs
        embedding_deployment = kwargs.get("model")
        if not embedding_deployment:
            raise ValueError("An embedding deployment name must be provided via the 'model' kwarg for the embed method.")
        
        try:
            response = self.client.embeddings.create(
                model=embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Azure OpenAI embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        """Return information about the current Azure OpenAI setup."""
        return {
            "name": self.binding_name,
            "version": openai.__version__,
            "host_address": self.azure_endpoint,
            "model_name": self.model_name, # This is the deployment name
            "supports_structured_output": False,
            "supports_vision": True, # Assume modern deployments support vision
        }

    def listModels(self) -> List[Dict[str, str]]:
        """
        List Models is not supported via the Azure OpenAI API.
        Deployments are managed in the Azure Portal. This method returns an empty list.
        """
        ASCIIColors.warning("Listing models is not supported for Azure OpenAI. Manage deployments in the Azure Portal.")
        return []

    def load_model(self, model_name: str) -> bool:
        """Sets the deployment name for subsequent operations."""
        self.model_name = model_name
        ASCIIColors.info(f"Azure OpenAI deployment set to: {model_name}. It will be used on the next API call.")
        return True

if __name__ == '__main__':
    # Environment variables to set for testing:
    # AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    # AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)
    # AZURE_DEPLOYMENT_NAME: The name of your chat deployment (e.g., gpt-4o)
    # AZURE_VISION_DEPLOYMENT_NAME: The name of a vision-capable deployment (can be the same as above)
    # AZURE_EMBEDDING_DEPLOYMENT_NAME: The name of your embedding deployment (e.g., text-embedding-ada-002)

    if not all(k in os.environ for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_DEPLOYMENT_NAME"]):
        ASCIIColors.red("Error: Required environment variables not set.")
        print("Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_DEPLOYMENT_NAME.")
        exit(1)

    ASCIIColors.yellow("--- Testing AzureOpenAIBinding ---")

    test_deployment_name = os.environ["AZURE_DEPLOYMENT_NAME"]
    
    try:
        # --- Initialization ---
        ASCIIColors.cyan("\n--- Initializing Binding ---")
        binding = AzureOpenAIBinding(model_name=test_deployment_name)
        ASCIIColors.green("Binding initialized successfully.")
        ASCIIColors.info(f"Using openai library version: {openai.__version__}")
        ASCIIColors.info(f"Endpoint: {binding.azure_endpoint}")
        ASCIIColors.info(f"Deployment: {binding.model_name}")

        # --- List Models ---
        ASCIIColors.cyan("\n--- Listing Models ---")
        models = binding.listModels()
        if not models:
            ASCIIColors.green("Correctly returned an empty list for models, as expected for Azure.")
        
        # --- Count Tokens ---
        ASCIIColors.cyan("\n--- Counting Tokens ---")
        sample_text = "Hello, Azure! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count}")

        # --- Text Generation (Non-Streaming) ---
        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "What is the Azure cloud platform known for? Answer in one sentence."
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

        # --- Embeddings ---
        if "AZURE_EMBEDDING_DEPLOYMENT_NAME" in os.environ:
            ASCIIColors.cyan("\n--- Embeddings ---")
            embedding_deployment = os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
            embedding_text = "LoLLMs and Azure make a great team."
            embedding_vector = binding.embed(embedding_text, model=embedding_deployment)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        else:
            ASCIIColors.yellow("\nSkipping Embeddings test: AZURE_EMBEDDING_DEPLOYMENT_NAME not set.")

        # --- Vision Model Test ---
        if "AZURE_VISION_DEPLOYMENT_NAME" in os.environ:
            vision_deployment = os.environ["AZURE_VISION_DEPLOYMENT_NAME"]
            dummy_image_path = "azure_dummy_test_image.png"
            try:
                img = Image.new('RGB', (250, 60), color = ('#0078D4')) # Azure blue
                d = ImageDraw.Draw(img)
                d.text((10,10), "Azure Test Image", fill=('white'))
                img.save(dummy_image_path)

                ASCIIColors.cyan(f"\n--- Vision Generation (deployment: {vision_deployment}) ---")
                binding.load_model(vision_deployment)
                vision_prompt = "Describe the image. What color is the background and what does the text say?"
                
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
            finally:
                if os.path.exists(dummy_image_path):
                    os.remove(dummy_image_path)
        else:
            ASCIIColors.yellow("\nSkipping Vision test: AZURE_VISION_DEPLOYMENT_NAME not set.")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nAzureOpenAIBinding test finished.")