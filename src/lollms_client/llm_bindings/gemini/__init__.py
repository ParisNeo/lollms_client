# bindings/gemini/__init__.py
import base64
import os
import re
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
from PIL import Image, ImageDraw 
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
        self.model_name = kwargs.get("model_name", "gemini-1.5-pro-latest")
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
                     think: Optional[bool] = False,
                     reasoning_effort: Optional[str] = "low", # low, medium, high
                     reasoning_summary: Optional[bool] = False, # auto
                     ) -> Union[str, dict]:
        """
        Generate text using the Gemini model.
        """
        if not self.client:
            return {"status": False, "error": "Gemini client not initialized."}

        # Handle 'think' parameter logging
        if think and "thinking" not in str(self.model_name).lower():
             ASCIIColors.info(f"Thinking requested but model '{self.model_name}' may not be a thinking model. Proceeding.")

        # Gemini uses 'system_instruction' for GenerativeModel
        try:
            model = self.client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None
            )
        except Exception as e:
            return {"status": False, "error": f"Failed to initialize GenerativeModel: {e}"}

        generation_config = self.get_generation_config(temperature, top_p, top_k, n_predict)

        # Prepare content for the API call
        content_parts = []
        if split:
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
                        if image_data.startswith("data:image"):
                             # Remove prefix if present
                             image_data = image_data.split(",")[1]
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
                    return {"status": False, "error": error_msg}
                return response.text

        except Exception as ex:
            error_message = f"An unexpected error occurred with Gemini API: {str(ex)}"
            trace_exception(ex)
            return {"status": False, "error": error_message}

    def generate_from_messages(self,
                        messages: List[Dict],
                        n_predict: Optional[int] = None,
                        stream: Optional[bool] = None,
                        temperature: Optional[float] = None,
                        top_k: Optional[int] = None,
                        top_p: Optional[float] = None,
                        repeat_penalty: Optional[float] = None,
                        repeat_last_n: Optional[int] = None,
                        seed: Optional[int] = None,
                        n_threads: Optional[int] = None,
                        ctx_size: int | None = None,
                        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                        think: Optional[bool] = False,
                        reasoning_effort: Optional[bool] = "low", 
                        reasoning_summary: Optional[bool] = "auto",
                        **kwargs
                        ) -> Union[str, dict]:
        """
        Generate content using a list of messages. This is the low-level method 
        that handles chat history structure for Gemini.
        """
        if not self.client:
            return {"status": False, "error": "Gemini client not initialized."}

        gen_config = self.get_generation_config(temperature, top_p, top_k, n_predict)

        system_instruction = None
        gemini_contents = []

        # Parse messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini specific role mapping
            if role == "system":
                # Gemini takes system prompt at initialization, not in the content list
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                     # Extract text from list if system prompt is complex (rare)
                     text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                     system_instruction = "\n".join(text_parts)
                continue
            
            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            # Parse content (Text or List of multimodal)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(item.get("text", ""))
                    elif item_type in ["input_image", "image_url"]:
                        # Extract Base64
                        base64_data = None
                        url_data = item.get("image_url", item.get("input_image"))
                        
                        if isinstance(url_data, dict):
                            # Format: {"url": "data:image/..."} or {"base64": "..."}
                            if "base64" in url_data:
                                base64_data = url_data["base64"]
                            elif "url" in url_data:
                                base64_data = url_data["url"]
                        elif isinstance(url_data, str):
                            base64_data = url_data

                        if base64_data:
                            # Clean "data:image/png;base64," prefix
                            if "base64," in base64_data:
                                base64_data = base64_data.split("base64,")[1]
                            try:
                                img = Image.open(BytesIO(base64.b64decode(base64_data)))
                                parts.append(img)
                            except Exception as e:
                                ASCIIColors.warning(f"Failed to decode image in message: {e}")

            if parts:
                gemini_contents.append({"role": gemini_role, "parts": parts})

        # Initialize Model with system instruction
        try:
            model = self.client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
        except Exception as e:
            return {"status": False, "error": f"Failed to initialize GenerativeModel: {e}"}

        full_response_text = ""

        try:
            # Generate content based on the full history
            response = model.generate_content(
                contents=gemini_contents,
                generation_config=gen_config,
                stream=stream
            )

            if stream:
                for chunk in response:
                    try:
                        chunk_text = chunk.text
                    except ValueError:
                        chunk_text = "" # Safety filter or empty chunk
                    
                    if chunk_text:
                        full_response_text += chunk_text
                        if streaming_callback:
                            if not streaming_callback(chunk_text, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                return full_response_text
            else:
                if response.prompt_feedback.block_reason:
                    error_msg = f"Content blocked due to: {response.prompt_feedback.block_reason.name}"
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
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
             think: Optional[bool] = False,
             reasoning_effort: Optional[str] = "low", # low, medium, high
             reasoning_summary: Optional[bool] = False, # auto
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the Gemini model using a LollmsDiscussion object.
        """
        if not self.client:
             return {"status": "error", "message": "Gemini client not initialized."}

        system_prompt = discussion.system_prompt
        messages = discussion.get_messages(branch_tip_id)
        
        history = []
        for msg in messages:
            role = 'user' if msg.sender_type == "user" else 'model' 
            
            content_parts = []
            if msg.content:
                content_parts.append(msg.content)
            
            if msg.images:
                for file_path in msg.images:
                     if is_image_path(file_path):
                         try:
                            content_parts.append(Image.open(file_path))
                         except Exception as e:
                            ASCIIColors.warning(f"Could not load image {file_path}: {e}")
                     else:
                        try:
                            b64_data = file_path
                            if b64_data.startswith("data:image"):
                                b64_data = b64_data.split(",")[1]
                            content_parts.append(Image.open(BytesIO(base64.b64decode(b64_data))))
                        except:
                            pass

            if content_parts:
                history.append({'role': role, 'parts': content_parts})
        
        try:
            model = self.client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None
            )
        except Exception as e:
            return {"status": "error", "message": f"Failed to initialize GenerativeModel: {e}"}
        
        # Organize history for chat session
        # ChatSession expects history excluding the very last message if we use send_message
        if not history:
            return {"status": "error", "message": "Cannot start chat with an empty discussion."}

        # Filter and consolidate consecutive roles
        consolidated_history = []
        current_role = None
        current_parts = []

        for msg in history:
            if msg['role'] == current_role:
                current_parts.extend(msg['parts'])
                consolidated_history[-1]['parts'] = current_parts
            else:
                current_role = msg['role']
                current_parts = list(msg['parts']) 
                consolidated_history.append({'role': current_role, 'parts': current_parts})

        # Separate the last message for the send_message call
        last_message = consolidated_history[-1]
        chat_history = consolidated_history[:-1]

        # Only 'user' can send_message in Gemini chat. 
        # If the last message in history is from 'model', the flow is slightly broken for standard chat,
        # but we assume the standard Lollms loop (User -> AI).
        
        chat_session = model.start_chat(history=chat_history)
        
        generation_config = self.get_generation_config(temperature, top_p, top_k, n_predict)
        
        full_response_text = ""
        try:
            response = chat_session.send_message(
                content=last_message['parts'],
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
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except:
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except:
            return bytes(tokens).decode('utf-8', errors='ignore')

    def count_tokens(self, text: str) -> int:
        if not self.client or not self.model_name:
            ASCIIColors.warning("Cannot count tokens, Gemini client or model_name not set.")
            return -1
        try:
            model = self.client.GenerativeModel(self.model_name)
            return model.count_tokens(text).total_tokens
        except Exception as e:
            ASCIIColors.error(f"Failed to count tokens with Gemini API: {e}")
            return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        if not self.client:
             raise Exception("Gemini client not initialized.")
        
        model_to_use = kwargs.get("model", "models/embedding-001")
        
        try:
            response = self.client.embed_content(
                model=model_to_use,
                content=text,
                task_type="retrieval_document" 
            )
            return response['embedding']
        except Exception as ex:
            trace_exception(ex)
            raise Exception(f"Gemini embedding failed: {str(ex)}") from ex

    def get_model_info(self) -> dict:
        return {
            "name": self.binding_name,
            "version": genai.__version__,
            "host_address": "https://generativelanguage.googleapis.com",
            "model_name": self.model_name,
            "supports_structured_output": False,
            "supports_vision": True,
        }

    def list_models(self) -> List[Dict[str, str]]:
        if not self.client:
            ASCIIColors.error("Gemini client not initialized. Cannot list models.")
            return []
        try:
            ASCIIColors.debug("Listing Gemini models...")
            model_info_list = []
            for m in self.client.list_models():
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
    full_streamed_text = ""

    try:
        # --- Initialization ---
        binding = GeminiBinding(model_name=test_model_name)
        ASCIIColors.green("Binding initialized successfully.")

        # --- Test generate_from_messages ---
        ASCIIColors.cyan("\n--- Testing generate_from_messages ---")
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "How are you today?"}
        ]
        response = binding.generate_from_messages(messages, n_predict=50)
        ASCIIColors.green(f"Pirate Response: {response}")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during testing: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\nGeminiBinding test finished.")

    def test_connection(self) -> dict:
        """
        Tests the connection to the Gemini API using the provided key.
        """
        if not self.client:
             return {"status": False, "message": "Client not configured. Check API Key."}
        try:
            # Attempt to list 1 model to verify auth
            list(self.client.list_models(page_size=1))
            return {"status": True, "message": "Connection successful! API Key is valid."}
        except Exception as e:
            return {"status": False, "message": f"Connection failed: {str(e)}"}

    def list_tuned_models(self) -> dict:
        """
        Lists tuned models available for this API key.
        """
        if not self.client:
             return {"status": False, "message": "Client not configured."}
        try:
            tuned_models = []
            # iterate over the tuned models generator
            for model in self.client.list_tuned_models():
                tuned_models.append(model.name)
            
            if not tuned_models:
                 return {"status": True, "message": "No tuned models found on this account.", "models": []}
            
            # Join them for a pretty message, but return the list for the UI
            msg = f"Found: {', '.join(tuned_models)}"
            return {"status": True, "message": msg, "models": tuned_models}
        except Exception as e:
            return {"status": False, "message": f"Could not list tuned models: {str(e)}", "models": []}

    def get_active_model_info(self) -> dict:
        """
        Retrieves details about the currently selected model name.
        """
        if not self.client or not self.model_name:
            return {"status": False, "message": "Client or model name not set."}
        
        try:
            model_info = self.client.get_model(self.model_name)
            
            info = {
                "display_name": model_info.display_name,
                "description": model_info.description,
                "input_token_limit": model_info.input_token_limit,
                "output_token_limit": model_info.output_token_limit,
                "supported_methods": model_info.supported_generation_methods
            }
            
            msg = f"Input Limit: {info['input_token_limit']}, Output Limit: {info['output_token_limit']}"
            return {"status": True, "message": msg, "info": info}
        except Exception as e:
            return {"status": False, "message": f"Failed to get info for {self.model_name}: {str(e)}"}