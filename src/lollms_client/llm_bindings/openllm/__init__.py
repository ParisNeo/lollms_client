# bindings/openllm/binding.py
import requests # May not be strictly needed if openllm client handles all
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import encode_image # Keep for potential image handling
from typing import Optional, Callable, List, Union, Dict

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

# Ensure openllm, pillow (for dummy image), and tiktoken are installed
pm.ensure_packages(["openllm", "pillow", "tiktoken"])

import openllm
import tiktoken # For fallback tokenization

BindingName = "OpenLLMBinding"

# Helper function to count tokens by making a minimal API call
# This is more accurate for the specific model than a generic tokenizer
def count_tokens_openllm(
    text_to_tokenize: str,
    openllm_client: openllm.client.HTTPClient,
    timeout: int = 60,
) -> int:
    """
    Counts the number of tokens in a given text for the connected OpenLLM model
    by making a minimal request to the /v1/generate endpoint and extracting
    the length of 'prompt_token_ids' from the response.
    """
    try:
        # Make a generation request asking for 0 or 1 new token
        # Some models might require at least 1 max_new_tokens
        llm_config = openllm.LLMConfig(max_new_tokens=1).model_dump(flatten=True, omit_default=True)
        response = openllm_client.generate(prompt=text_to_tokenize, llm_config=llm_config, timeout=timeout)
        
        if response.prompt_token_ids is not None and len(response.prompt_token_ids) > 0:
            # The prompt_token_ids from OpenLLM often include special tokens (e.g., BOS)
            # depending on the model's tokenizer configuration.
            # For consistency with typical "user text token count", we might need to adjust.
            # However, for now, let's return the raw count from the model.
            # A simple heuristic might be to subtract 1 for a BOS token if always present.
            # This needs model-specific knowledge or further investigation.
            # For llama3 with ollama, it was prompt_eval_count - 5 (system, user, content etc)
            # For OpenLLM, it's harder to generalize the "overhead".
            # Let's assume prompt_token_ids is the count of tokens for the user's text.
            return len(response.prompt_token_ids)
        else:
            # Fallback if prompt_token_ids is not available or empty
            ASCIIColors.warning("prompt_token_ids not found in OpenLLM response, using tiktoken for count_tokens.")
            return len(tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text_to_tokenize))
    except Exception as e:
        ASCIIColors.warning(f"Failed to count tokens via OpenLLM API, using tiktoken fallback: {e}")
        return len(tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text_to_tokenize))


class OpenLLMBinding(LollmsLLMBinding):
    """OpenLLM-specific binding implementation using the openllm-python client."""
    
    DEFAULT_HOST_ADDRESS = "http://localhost:3000" # Default OpenLLM server address
    
    def __init__(self,
                 **kwargs
                 ):
        """        Initialize the OpenLLM binding.
        Args:
            host_address (str): The address of the OpenLLM server (default: http://localhost:3000).
            model_name (str): The name of the model to connect to. This is primarily for informational purposes.
            service_key (Optional[str]): Optional service key for authentication, not used by openllm client.
            verify_ssl_certificate (bool): Whether to verify SSL certificates (default: True).
            timeout (int): Timeout for client requests in seconds (default: 120).
        """
        host_address = kwargs.get("host_address")
        _host_address = host_address if host_address is not None else self.DEFAULT_HOST_ADDRESS
        super().__init__(BindingName, **kwargs)
        self.host_address = _host_address
        self.model_name = kwargs.get("model_name") # Can be set by load_model or from config
        self.default_completion_format=kwargs.get("default_completion_format",ELF_COMPLETION_FORMAT.Chat) 
        self.timeout = kwargs.get("timeout")

        if openllm is None or openllm.client is None:
            raise ImportError("OpenLLM library is not installed or client module not found. Please run 'pip install openllm'.")

        try:
            self.openllm_client = openllm.client.HTTPClient(
                address=self.host_address,
                timeout=self.timeout
            )
            # Perform a quick health check or metadata fetch to confirm connection
            if not self._verify_connection():
                raise ConnectionError(f"Failed to connect or verify OpenLLM server at {self.host_address}")
            
            # Try to fetch model_name if not provided
            if not self.model_name:
                metadata = self._get_model_metadata_from_server()
                if metadata and 'model_id' in metadata:
                    self.model_name = metadata['model_id']
                else:
                    ASCIIColors.warning("Could not automatically determine model name from OpenLLM server.")

        except Exception as e:
            ASCIIColors.error(f"Failed to initialize OpenLLM client: {e}")
            self.openllm_client = None
            raise ConnectionError(f"Could not connect or initialize OpenLLM client at {self.host_address}: {e}") from e

    def _verify_connection(self) -> bool:
        if not self.openllm_client:
            return False
        try:
            return self.openllm_client.health() # health() returns True if healthy, raises error otherwise
        except Exception as e:
            ASCIIColors.warning(f"OpenLLM server health check failed for {self.host_address}: {e}")
            return False

    def _get_model_metadata_from_server(self) -> Optional[Dict]:
        if not self.openllm_client:
            return None
        try:
            # metadata() returns a GenerationOutput object which contains model_name, backend etc.
            meta_output = self.openllm_client.metadata()
            # The actual LLMConfig and model details are in meta_output.configuration (a string JSON)
            # and meta_output.model_name, meta_output.backend etc.
            # For simplicity, let's try to parse configuration or use model_name
            config_dict = {}
            if meta_output.configuration:
                try:
                    config_dict = json.loads(meta_output.configuration)
                except json.JSONDecodeError:
                    ASCIIColors.warning("Failed to parse model configuration from OpenLLM metadata.")
            
            return {
                "model_id": config_dict.get("model_id", meta_output.model_name), # model_id from config is better
                "model_name": meta_output.model_name, # As reported by client.metadata()
                "backend": meta_output.backend,
                "timeout": meta_output.timeout,
                "configuration": config_dict
            }
        except Exception as e:
            ASCIIColors.warning(f"Could not fetch metadata from OpenLLM server: {e}")
            return None

    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None, # List of image file paths
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: bool = False,
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1,
                     # repeat_last_n: int = 64, # OpenLLM's LLMConfig doesn't have direct repeat_last_n
                     seed: Optional[int] = None,
                     # n_threads: Optional[int] = None, # Server-side config for OpenLLM
                     # ctx_size: Optional[int] = None,  # Server-side config, though some models might allow via llm_config
                     streaming_callback: Optional[Callable[[str, int], bool]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",                     
                     ) -> Union[str, Dict[str, any]]:
        
        if not self.openllm_client:
             return {"status": False, "error": "OpenLLM client not initialized."}

        # Construct LLMConfig
        # Note: Not all Lollms params map directly to OpenLLM's LLMConfig.
        # We map what's available.
        config_params = {
            "temperature": float(temperature),
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
        }
        if n_predict is not None: config_params['max_new_tokens'] = n_predict
        if seed is not None: config_params['seed'] = seed # seed might not be supported by all backends/models
        
        llm_config = openllm.LLMConfig(**config_params).model_dump(flatten=True, omit_default=True)

        # Prepend system prompt if provided
        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:" # Common instruct format

        # Handle images: This is highly model-dependent for OpenLLM.
        # For LLaVA-like models, images are base64 encoded and put in the prompt.
        # This is a simplified approach. A robust solution needs model-specific prompt templating.
        if images:
            ASCIIColors.warning("Image support in OpenLLMBinding is basic and assumes a LLaVA-like model "
                                "that accepts base64 image data in the prompt.")
            image_parts = []
            for img_path in images:
                try:
                    # encode_image from lollms_utilities returns base64 string
                    base64_image = encode_image(img_path) 
                    # Basic assumption: image can be prepended or appended.
                    # For LLaVA, it's often "<image>\nUSER: What is this? ASSISTANT:"
                    # or the raw base64 data might be directly in the prompt.
                    # This is a placeholder for where more complex prompt construction would go.
                    # For now, let's just put the base64 string.
                    image_parts.append(f"[Image data: {base64_image}]") # Simplistic
                except Exception as e:
                    ASCIIColors.error(f"Could not encode image {img_path}: {e}")
            
            if image_parts:
                full_prompt = "\n".join(image_parts) + "\n" + full_prompt
        
        full_response_text = ""
        try:
            if stream:
                response_stream = self.openllm_client.generate_stream(
                    prompt=full_prompt,
                    llm_config=llm_config,
                    timeout=self.timeout
                )
                for chunk in response_stream:
                    # chunk is openllm.GenerationChunk
                    chunk_content = chunk.text
                    if chunk_content:
                        full_response_text += chunk_content
                        if streaming_callback:
                            if not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break # Callback requested stop
                return full_response_text
            else: # Not streaming
                response_output = self.openllm_client.generate(
                    prompt=full_prompt,
                    llm_config=llm_config,
                    timeout=self.timeout
                )
                # response_output is openllm.GenerationOutput
                # It can contain multiple responses if n > 1 (not used here)
                if response_output.responses:
                    return response_output.responses[0].text
                else:
                    return {"status": False, "error": "OpenLLM returned no response."}
        except openllm.exceptions.OpenLLMException as e:
            error_message = f"OpenLLM API Error: {str(e)}"
            ASCIIColors.error(error_message)
            # Attempt to get more details if it's an HTTPError from httpx
            if hasattr(e, '__cause__') and isinstance(e.__cause__, requests.exceptions.HTTPError):
                 error_message += f" - HTTP Status: {e.__cause__.response.status_code}, Response: {e.__cause__.response.text}"
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'): # For httpx.HTTPStatusError
                 error_message += f" - HTTP Status: {e.response.status_code}, Response: {e.response.text}"

            return {"status": False, "error": error_message}
        except Exception as ex:
            error_message = f"An unexpected error occurred: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}

    def tokenize(self, text: str) -> list:
        """Tokenize text using tiktoken as a fallback."""
        # OpenLLM client doesn't provide a direct tokenization API.
        # For accurate tokenization, it would depend on the specific model served.
        # Using tiktoken as a general approximation.
        try:
            # Try to use a tokenizer related to the model if known, else default
            if "llama" in self.model_name.lower(): # Crude check
                enc = tiktoken.encoding_for_model("text-davinci-003") # Llama tokenizers are different but this is a proxy
            elif "gpt" in self.model_name.lower(): # e.g. gpt2 served by OpenLLM
                enc = tiktoken.get_encoding("gpt2")
            else:
                enc = tiktoken.model.encoding_for_model("gpt-3.5-turbo") # Fallback
            return enc.encode(text)
        except Exception:
            # Further fallback
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            
    def detokenize(self, tokens: list) -> str:
        """Detokenize tokens using tiktoken as a fallback."""
        try:
            if "llama" in self.model_name.lower():
                enc = tiktoken.encoding_for_model("text-davinci-003")
            elif "gpt" in self.model_name.lower():
                enc = tiktoken.get_encoding("gpt2")
            else:
                enc = tiktoken.model.encoding_for_model("gpt-3.5-turbo")
            return enc.decode(tokens)
        except Exception:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the OpenLLM server if possible, else tiktoken."""
        if not self.openllm_client:
            ASCIIColors.warning("OpenLLM client not initialized. Using tiktoken for count_tokens.")
            return len(self.tokenize(text)) # Fallback to tiktoken via self.tokenize

        # Try the API call method for better accuracy for the specific model
        # return count_tokens_openllm(text, self.openllm_client, self.timeout)
        # The API call above can be slow. For faster, but less model-specific count:
        return len(self.tokenize(text))


    def embed(self, text: str, **kwargs) -> List[float]:
        """Get embeddings for the input text using OpenLLM API."""
        if not self.openllm_client:
             raise Exception("OpenLLM client not initialized.")

        # model_to_use kwarg is less relevant here as client is tied to one model server.
        # If that server is an embedding model, it will work.
        # llm_config can be passed via kwargs if needed for embeddings.
        llm_config_dict = kwargs.get("llm_config", {})
        llm_config = openllm.LLMConfig(**llm_config_dict).model_dump(flatten=True, omit_default=True) if llm_config_dict else None
        
        try:
            # openllm_client.embeddings expects a list of prompts
            response = self.openllm_client.embeddings(
                prompts=[text], 
                llm_config=llm_config,
                timeout=self.timeout
            )
            # response is a list of embeddings (list of lists of floats)
            if response and len(response) > 0:
                return response[0] 
            else:
                raise Exception("OpenLLM returned no embeddings.")
        except openllm.exceptions.OpenLLMException as e:
            error_message = f"OpenLLM API Embeddings Error: {str(e)}"
            ASCIIColors.error(error_message)
            raise Exception(error_message) from e
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Embedding failed: {str(ex)}") from ex
        
    def get_model_info(self) -> dict:
        """Return information about the current OpenLLM model setup."""
        server_metadata = self._get_model_metadata_from_server()
        model_id_from_server = "unknown"
        if server_metadata and 'model_id' in server_metadata:
            model_id_from_server = server_metadata['model_id']
        
        # Try to determine vision support based on model name (very basic)
        supports_vision = False
        if self.model_name and any(vm_name in self.model_name.lower() for vm_name in ["llava", "bakllava", "vision"]):
            supports_vision = True

        return {
            "name": self.binding_name,
            "version": openllm.__version__ if openllm else "unknown",
            "host_address": self.host_address,
            "model_name": self.model_name or model_id_from_server, # Use self.model_name if set, else from server
            "supports_structured_output": False, # Generic OpenLLM text generation doesn't guarantee this
            "supports_vision": supports_vision # Highly dependent on the specific model served
        }

    def listModels(self) -> List[Dict[str, str]]:
        """
        Lists the model currently served by the connected OpenLLM instance.
        OpenLLM client connects to one model server at a time.
        """
        if not self.openllm_client:
            ASCIIColors.error("OpenLLM client not initialized. Cannot list models.")
            return []
        
        metadata = self._get_model_metadata_from_server()
        if metadata:
            return [{
                'model_name': metadata.get('model_id', metadata.get('model_name', 'Unknown Model')), # Prefer model_id
                'owned_by': metadata.get('backend', 'OpenLLM'), # Using backend as a proxy for owner/type
                # OpenLLM metadata doesn't typically include a creation/modification date for the model files themselves.
                'created_datetime': None 
            }]
        return []

    def load_model(self, model_name: str) -> bool:
        """
        For OpenLLM, this primarily sets the model_name for reference, as the
        model is already loaded by the server the client connects to.
        Optionally, it could re-initialize the client if host_address also changes,
        or verify the existing connection serves this model.
        Args:
            model_name (str): Name of the model (e.g., 'mistralai/Mistral-7B-Instruct-v0.1').
                              This should match what the server at self.host_address is running.
        Returns:
            bool: True if model name is set and connection seems okay.
        """
        self.model_name = model_name
        ASCIIColors.info(f"OpenLLM binding model_name set to: {model_name}.")
        ASCIIColors.info(f"Ensure OpenLLM server at {self.host_address} is running this model.")
        
        # Optionally, verify the connected server's model matches
        server_meta = self._get_model_metadata_from_server()
        if server_meta:
            current_server_model_id = server_meta.get('model_id', server_meta.get('model_name'))
            if current_server_model_id and model_name not in current_server_model_id : # Check if model_name is substring of actual ID
                ASCIIColors.warning(f"Warning: Requested model '{model_name}' may not match model '{current_server_model_id}' served at {self.host_address}.")
            else:
                ASCIIColors.green(f"Connected OpenLLM server model appears to be '{current_server_model_id}'.")
        
        return self._verify_connection()


if __name__ == '__main__':
    global full_streamed_text
    ASCIIColors.yellow("Testing OpenLLMBinding...")

    # --- Configuration ---
    # Ensure an OpenLLM server is running. Example:
    # `openllm start mistralai/Mistral-7B-Instruct-v0.1`
    # or for embeddings: `openllm start baai/bge-small-en-v1.5`
    # or for vision (if you have a LLaVA model compatible with OpenLLM):
    # `openllm start llava-hf/llava-1.5-7b-hf` (You might need to convert/setup some vision models for OpenLLM)

    openllm_host = "http://localhost:3000" 
    # This should match the model_id you started OpenLLM with
    test_model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Example, change if your server runs a different model
    # test_model_name = "facebook/opt-125m" # A smaller model for quicker tests if available

    # For embedding test, you'd point to an OpenLLM server running an embedding model
    # openllm_embedding_host = "http://localhost:3001" # If running embedding model on different port
    # test_embedding_model_name = "baai/bge-small-en-v1.5"

    # For vision, if you have a LLaVA model running with OpenLLM
    # openllm_vision_host = "http://localhost:3002"
    # test_vision_model_name = "llava-hf/llava-1.5-7b-hf" # Example

    try:
        ASCIIColors.cyan("\n--- Initializing Binding for Text Generation ---")
        # Initialize with the host where your text generation model is running
        binding = OpenLLMBinding(host_address=openllm_host, model_name=test_model_name)
        ASCIIColors.green(f"Binding initialized successfully. Connected to model: {binding.model_name}")
        ASCIIColors.info(f"Using OpenLLM client version: {openllm.__version__ if openllm else 'N/A'}")

        ASCIIColors.cyan("\n--- Listing Model (should be the one connected) ---")
        models = binding.listModels()
        if models:
            ASCIIColors.green(f"Connected model info:")
            for m in models:
                print(m)
        else:
            ASCIIColors.warning("Failed to list model from server. Ensure OpenLLM server is running.")

        ASCIIColors.cyan(f"\n--- Setting model to (for info): {test_model_name} ---")
        binding.load_model(test_model_name) # This confirms the model name and checks connection

        ASCIIColors.cyan("\n--- Counting Tokens (using tiktoken fallback or API) ---")
        sample_text = "Hello, OpenLLM world! This is a test."
        token_count = binding.count_tokens(sample_text)
        ASCIIColors.green(f"Token count for '{sample_text}': {token_count} (may use tiktoken approximation)")

        ASCIIColors.cyan("\n--- Tokenize/Detokenize (using tiktoken fallback) ---")
        tokens = binding.tokenize(sample_text)
        ASCIIColors.green(f"Tokens (tiktoken): {tokens[:10]}...")
        detokenized_text = binding.detokenize(tokens)
        ASCIIColors.green(f"Detokenized text (tiktoken): {detokenized_text}")

        ASCIIColors.cyan("\n--- Text Generation (Non-Streaming) ---")
        prompt_text = "Why is the sky blue?"
        system_prompt_text = "You are a helpful AI assistant providing concise answers."
        ASCIIColors.info(f"System Prompt: {system_prompt_text}")
        ASCIIColors.info(f"User Prompt: {prompt_text}")
        generated_text = binding.generate_text(prompt_text, system_prompt=system_prompt_text, n_predict=50, stream=False)
        if isinstance(generated_text, str):
            ASCIIColors.green(f"Generated text: {generated_text}")
        else:
            ASCIIColors.error(f"Generation failed: {generated_text}")

        ASCIIColors.cyan("\n--- Text Generation (Streaming) ---")
        full_streamed_text = ""
        def stream_callback(chunk: str, msg_type: int):
            global full_streamed_text
            print(f"{ASCIIColors.GREEN}{chunk}{ASCIIColors.RESET}", end="", flush=True)
            full_streamed_text += chunk
            return True
        
        ASCIIColors.info(f"Prompt: {prompt_text}")
        result = binding.generate_text(prompt_text, system_prompt=system_prompt_text, n_predict=100, stream=True, streaming_callback=stream_callback)
        print("\n--- End of Stream ---")
        if isinstance(result, str):
             ASCIIColors.green(f"Full streamed text: {result}")
        else:
            ASCIIColors.error(f"Streaming generation failed: {result}")

        # --- Embeddings Test ---
        # You need to run an OpenLLM server with an embedding model for this.
        # Example: `openllm start baai/bge-small-en-v1.5 --port 3001`
        # Then change openllm_host to "http://localhost:3001" for this section.
        ASCIIColors.cyan("\n--- Embeddings Test ---")
        ASCIIColors.magenta("INFO: This test requires an OpenLLM server running an EMBEDDING model (e.g., bge, E5).")
        ASCIIColors.magenta(f"      If your server at {openllm_host} is a text generation model, this might fail.")
        embedding_text = "Lollms is a cool project using OpenLLM."
        try:
            # If your main binding is for text-gen, you might need a separate binding instance
            # for an embedding model if it's on a different host/port.
            # For this example, we'll try with the current binding.
            # If it fails, it means the model at openllm_host doesn't support /v1/embeddings
            embedding_vector = binding.embed(embedding_text)
            ASCIIColors.green(f"Embedding for '{embedding_text}' (first 5 dims): {embedding_vector[:5]}...")
            ASCIIColors.info(f"Embedding vector dimension: {len(embedding_vector)}")
        except Exception as e:
            ASCIIColors.warning(f"Could not get embedding with model '{binding.model_name}' at '{binding.host_address}': {e}")
            ASCIIColors.warning("Ensure the OpenLLM server is running an embedding-capable model and supports the /v1/embeddings endpoint.")

        # --- Vision Model Test ---
        ASCIIColors.cyan("\n--- Vision Model Test (Conceptual) ---")
        ASCIIColors.magenta("INFO: This test requires an OpenLLM server running a VISION model (e.g., LLaVA).")
        ASCIIColors.magenta(f"      And the model needs to accept images as base64 in prompt. This is a basic test.")
        
        dummy_image_path = "dummy_test_image_openllm.png"
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 50), color = ('blue'))
            d = ImageDraw.Draw(img)
            d.text((10,10), "OpenLLM Test", fill=('white'))
            img.save(dummy_image_path)
            ASCIIColors.info(f"Created dummy image: {dummy_image_path}")

            # Assuming your 'binding' is connected to a vision model server.
            # If not, you'd initialize a new binding pointing to your vision model server.
            # e.g., vision_binding = OpenLLMBinding(host_address=openllm_vision_host, model_name=test_vision_model_name)
            
            # Check if current model_name hints at vision
            if "llava" not in binding.model_name.lower() and "vision" not in binding.model_name.lower() :
                 ASCIIColors.warning(f"Current model '{binding.model_name}' might not be a vision model. Vision test may not be meaningful.")

            vision_prompt = "What is written in the image and what color is the background?"
            ASCIIColors.info(f"Vision Prompt: {vision_prompt} with image {dummy_image_path}")
            
            vision_response = binding.generate_text(
                prompt=vision_prompt,
                images=[dummy_image_path], # The binding will attempt to base64 encode this
                n_predict=50,
                stream=False
            )
            if isinstance(vision_response, str):
                ASCIIColors.green(f"Vision model response: {vision_response}")
            else:
                ASCIIColors.error(f"Vision generation failed: {vision_response}")
        except ImportError:
            ASCIIColors.warning("Pillow library not found. Cannot create dummy image for vision test. `pip install Pillow`")
        except Exception as e:
            ASCIIColors.error(f"Error during vision test: {e}")
            trace_exception(e)
        finally:
            import os
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

    except ConnectionRefusedError:
        ASCIIColors.error(f"Connection to OpenLLM server at {openllm_host} refused. Is OpenLLM server running?")
        ASCIIColors.error("Example: `openllm start mistralai/Mistral-7B-Instruct-v0.1`")
    except openllm.exceptions.OpenLLMException as e:
        ASCIIColors.error(f"OpenLLM specific error: {e}")
        trace_exception(e)
    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nOpenLLMBinding test finished.")