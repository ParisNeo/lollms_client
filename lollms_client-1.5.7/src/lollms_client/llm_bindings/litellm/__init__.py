# bindings/LiteLLM/binding.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import encode_image
from typing import Optional, Callable, List, Union, Dict
from ascii_colors import ASCIIColors, trace_exception

# Use pipmaster to ensure required packages are installed
try:
    import pipmaster as pm
except ImportError:
    print("Pipmaster not found. Please install it using 'pip install pipmaster'")
    raise

# Ensure requests and tiktoken are installed
pm.ensure_packages(["requests", "tiktoken"])

import tiktoken

BindingName = "LiteLLMBinding"

def get_icon_path(model_name: str) -> str:
    model_name = model_name.lower()
    if 'gpt' in model_name: return '/bindings/openai/logo.png'
    if 'mistral' in model_name or 'mixtral' in model_name: return '/bindings/mistral/logo.png'
    if 'claude' in model_name: return '/bindings/anthropic/logo.png'
    return '/bindings/litellm/logo.png'

class LiteLLMBinding(LollmsLLMBinding):
    """
    A binding for the LiteLLM proxy using direct HTTP requests.
    This version includes detailed logging, a fallback for listing models,
    and correct payload formatting for both streaming and non-streaming modes.
    """
    
    def __init__(self, 
                 **kwargs):
        """ Initializes the LiteLLM binding with the provided parameters.
        Args:
            host_address (str): The base URL of the LiteLLM server.
            model_name (str): The name of the model to use.
            service_key (str): The API key for authentication.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        super().__init__(BindingName, **kwargs)
        self.host_address = kwargs.get("host_address")
        if self.host_address: self.host_address = self.host_address.rstrip('/')
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate")

    def _perform_generation(self, messages: List[Dict], n_predict: Optional[int], stream: bool, temperature: float, top_p: float, repeat_penalty: float, seed: Optional[int], streaming_callback: Optional[Callable[[str, MSG_TYPE], None]]) -> Union[str, dict]:
        url = f'{self.host_address}/v1/chat/completions'
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.service_key}'}
        payload = {
            "model": self.model_name, "messages": messages, "max_tokens": n_predict,
            "temperature": temperature, "top_p": top_p, "frequency_penalty": repeat_penalty,
            "stream": stream
        }
        if seed is not None: payload["seed"] = seed
        
        payload = {k: v for k, v in payload.items() if v is not None}
        output = ""
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), stream=stream, verify=self.verify_ssl_certificate)
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            if '[DONE]' in decoded_line: break
                            json_data_string = decoded_line[6:]
                            try:
                                chunk_data = json.loads(json_data_string)
                                delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                                if 'content' in delta and delta['content'] is not None:
                                    word = delta['content']
                                    if streaming_callback and not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                        return output
                                    output += word
                            except json.JSONDecodeError: continue
            else:
                full_response = response.json()
                output = full_response['choices'][0]['message']['content']
                if streaming_callback:
                    streaming_callback(output, MSG_TYPE.MSG_TYPE_CHUNK)
        except Exception as e:
            error_message = f"An error occurred: {e}\nResponse: {response.text if 'response' in locals() else 'No response'}"
            trace_exception(e)
            if streaming_callback: streaming_callback(error_message, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": error_message}
        return output

    def generate_text(self, prompt: str, images: Optional[List[str]] = None, system_prompt: str = "", n_predict: Optional[int] = None, stream: Optional[bool] = None, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.1, seed: Optional[int] = None, streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None, **kwargs) -> Union[str, dict]:
        """Generates text from a prompt, correctly formatting for text-only and multi-modal cases."""
        is_streaming = stream if stream is not None else (streaming_callback is not None)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # --- THIS IS THE CRITICAL FIX ---
        if images:
            # If images are present, use the multi-modal list format for content
            user_content = [{"type": "text", "text": prompt}]
            for image_path in images:
                base64_image = encode_image(image_path)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            messages.append({"role": "user", "content": user_content})
        else:
            # If no images, use a simple string for content to avoid the API error
            messages.append({"role": "user", "content": prompt})
        # --- END OF FIX ---

        return self._perform_generation(messages, n_predict, is_streaming, temperature, top_p, repeat_penalty, seed, streaming_callback)

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
                     **kwargs
                     ) -> Union[str, dict]:
        is_streaming = stream if stream is not None else (streaming_callback is not None)
        return self._perform_generation(messages, n_predict, is_streaming, temperature, top_p, repeat_penalty, seed, streaming_callback)
        

    def chat(self, discussion: LollmsDiscussion, branch_tip_id: Optional[str] = None, n_predict: Optional[int] = None, stream: Optional[bool] = None, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.1, seed: Optional[int] = None, streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None, **kwargs) -> Union[str, dict]:
        is_streaming = stream if stream is not None else (streaming_callback is not None)
        messages = discussion.export("openai_chat", branch_tip_id)
        return self._perform_generation(messages, n_predict, is_streaming, temperature, top_p, repeat_penalty, seed, streaming_callback)

    def embed(self, text: str, **kwargs) -> List[float]:
        url = f'{self.host_address}/v1/embeddings'
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.service_key}'}
        payload = {"model": self.model_name, "input": text}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), verify=self.verify_ssl_certificate)
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            trace_exception(e)
            return []

    def tokenize(self, text: str) -> list:
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            
    def detokenize(self, tokens: list) -> str:
        return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))
    
    def _list_models_openai_fallback(self) -> List[Dict]:
        ASCIIColors.warning("--- [LiteLLM Binding] Falling back to /v1/models endpoint. Rich metadata will be unavailable.")
        url = f'{self.host_address}/v1/models'
        headers = {'Authorization': f'Bearer {self.service_key}'}
        entries = []
        try:
            response = requests.get(url, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status()
            models_data = response.json().get('data', [])
            for model in models_data:
                model_name = model.get('id')
                entries.append({
                    "category": "api", "datasets": "unknown", "icon": get_icon_path(model_name),
                    "license": "unknown", "model_creator": model.get('owned_by', 'unknown'),
                    "name": model_name, "provider": "litellm", "rank": "1.0", "type": "api",
                    "variants": [{"name": model_name, "size": -1}]
                })
        except Exception as e:
            ASCIIColors.error(f"--- [LiteLLM Binding] Fallback method failed: {e}")
        return entries

    def listModels(self) -> List[Dict]:
        url = f'{self.host_address}/model/info'
        headers = {'Authorization': f'Bearer {self.service_key}'}
        entries = []
        ASCIIColors.yellow(f"--- [LiteLLM Binding] Attempting to list models from: {url}")
        try:
            response = requests.get(url, headers=headers, verify=self.verify_ssl_certificate)
            if response.status_code == 404:
                ASCIIColors.warning("--- [LiteLLM Binding] /model/info endpoint not found (404).")
                return self._list_models_openai_fallback()
            response.raise_for_status()
            models_data = response.json().get('data', [])
            ASCIIColors.info(f"--- [LiteLLM Binding] Successfully parsed {len(models_data)} models from primary endpoint.")
            for model in models_data:
                model_name = model.get('model_name')
                if not model_name: continue
                model_info = model.get('model_info', {})
                context_size = model_info.get('max_tokens', model_info.get('max_input_tokens', 4096))
                entries.append({
                    "category": "api", "datasets": "unknown", "icon": get_icon_path(model_name),
                    "license": "unknown", "model_creator": model_info.get('owned_by', 'unknown'),
                    "model_name": model_name, "provider": "litellm", "rank": "1.0", "type": "api",
                    "variants": [{
                        "model_name": model_name, "size": context_size,
                        "input_cost_per_token": model_info.get('input_cost_per_token', 0),
                        "output_cost_per_token": model_info.get('output_cost_per_token', 0),
                        "max_output_tokens": model_info.get('max_output_tokens', 0),
                    }]
                })
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"--- [LiteLLM Binding] Network error when trying to list models: {e}")
            if "404" in str(e): return self._list_models_openai_fallback()
        except Exception as e:
            ASCIIColors.error(f"--- [LiteLLM Binding] An unexpected error occurred while listing models: {e}")
        return entries

    def get_model_info(self) -> dict:
        return {"name": "LiteLLM", "host_address": self.host_address, "model_name": self.model_name}

    def load_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True
