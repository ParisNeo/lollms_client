# lollms_client/llm_bindings/lmstudio/__init__.py
import requests
import json
import time
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from typing import Optional, Callable, List, Union, Dict, Any
from ascii_colors import ASCIIColors, trace_exception

BindingName = "LMStudioBinding"

class LMStudioBinding(LollmsLLMBinding):
    """
    LM Studio-specific binding implementation with enhanced GPU optimization features.
    
    Supports both OpenAI-compatible API and native LM Studio REST API for better performance monitoring.
    Optimized for RTX 3060 12GB and similar GPU setups.
    """
    
    DEFAULT_HOST_ADDRESS = "http://localhost:1234"
    
    def __init__(self,
                 host_address: str = None,
                 model_name: str = "",
                 service_key: str = None,  # Not required for LM Studio
                 verify_ssl_certificate: bool = True,
                 use_native_api: bool = True,  # Use LM Studio's native REST API for better stats
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat):
        """
        Initialize the LM Studio binding.

        Args:
            host_address (str): Host address for LM Studio. Defaults to localhost:1234.
            model_name (str): Name of the model to use. Auto-detected if empty.
            service_key (str): Not required for LM Studio local server.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
            use_native_api (bool): Use LM Studio's native REST API for enhanced features.
            default_completion_format: Chat or Instruct format preference.
        """
        super().__init__("lmstudio")
        
        self.host_address = host_address or self.DEFAULT_HOST_ADDRESS
        self.model_name = model_name
        self.service_key = service_key or "lm-studio-local"  # Dummy key for compatibility
        self.verify_ssl_certificate = verify_ssl_certificate
        self.use_native_api = use_native_api
        self.default_completion_format = default_completion_format
        
        # API endpoints
        self.openai_base_url = f"{self.host_address}/v1"
        self.native_base_url = f"{self.host_address}/api/v0"
        
        # Performance tracking
        self.last_generation_stats = {}
        
        # Auto-detect available models if none specified
        if not self.model_name:
            self._auto_detect_model()
    
    def _auto_detect_model(self):
        """Auto-detect the best available model for the current setup."""
        try:
            models = self.listModels()
            if models:
                # Prefer loaded models first
                loaded_models = [m for m in models if isinstance(m, dict) and m.get('state') == 'loaded']
                if loaded_models:
                    self.model_name = loaded_models[0]['id']
                    ASCIIColors.info(f"Auto-detected loaded model: {self.model_name}")
                else:
                    # Fall back to first available model
                    first_model = models[0]
                    self.model_name = first_model['id'] if isinstance(first_model, dict) else str(first_model)
                    ASCIIColors.info(f"Auto-detected model: {self.model_name}")
        except Exception as e:
            ASCIIColors.warning(f"Could not auto-detect model: {e}")
    
    def _make_request(self, endpoint: str, data: dict, stream: bool = False, use_native: bool = None) -> requests.Response:
        """Make HTTP request to LM Studio API with proper error handling."""
        use_native = use_native if use_native is not None else self.use_native_api
        base_url = self.native_base_url if use_native else self.openai_base_url
        url = f"{base_url}/{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        if not use_native:
            headers["Authorization"] = f"Bearer {self.service_key}"
        
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                stream=stream,
                verify=self.verify_ssl_certificate,
                timeout=300  # 5 minute timeout for long generations
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"LM Studio API request failed: {e}")
            raise
    
    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1,
                     repeat_last_n: int = 64,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: Optional[int] = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split: Optional[bool] = False,
                     user_keyword: Optional[str] = "!@>user:",
                     ai_keyword: Optional[str] = "!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using LM Studio with enhanced performance monitoring.
        
        Returns generation stats including tokens/second, TTFT, and GPU utilization info.
        """
        try:
            # Prepare messages for chat completion
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Handle multimodal input
            if images:
                content = [{"type": "text", "text": prompt}]
                for image_path in images:
                    try:
                        encoded_image = encode_image(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        })
                    except Exception as e:
                        ASCIIColors.warning(f"Failed to encode image {image_path}: {e}")
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # Prepare request data
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": n_predict or -1,  # -1 means no limit in LM Studio
                "stream": stream or False,
                "seed": seed
            }
            
            # Add LM Studio specific parameters
            if repeat_penalty != 1.1:
                data["frequency_penalty"] = (repeat_penalty - 1.0) * 2.0  # Convert to OpenAI format
            
            start_time = time.time()
            
            if stream and streaming_callback:
                return self._handle_streaming_response(data, streaming_callback, start_time)
            else:
                return self._handle_non_streaming_response(data, start_time)
                
        except Exception as e:
            ASCIIColors.error(f"Text generation failed: {e}")
            trace_exception(e)
            return {"error": str(e)}
    
    def _handle_streaming_response(self, data: dict, callback: Callable, start_time: float) -> str:
        """Handle streaming response with performance monitoring."""
        data["stream"] = True
        full_response = ""
        first_token_time = None
        
        try:
            response = self._make_request("chat/completions", data, stream=True)
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        json_str = line_str[6:]
                        if json_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(json_str)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    
                                    full_response += content
                                    if callback:
                                        callback(content, MSG_TYPE.MSG_TYPE_CHUNK)
                        except json.JSONDecodeError:
                            continue
            
            # Calculate and store performance stats
            end_time = time.time()
            self._calculate_performance_stats(full_response, start_time, first_token_time, end_time)
            
            return full_response
            
        except Exception as e:
            if callback:
                callback(f"Streaming error: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"error": str(e)}
    
    def _handle_non_streaming_response(self, data: dict, start_time: float) -> str:
        """Handle non-streaming response with performance monitoring."""
        try:
            response = self._make_request("chat/completions", data)
            response_data = response.json()
            
            if 'choices' in response_data and response_data['choices']:
                content = response_data['choices'][0]['message']['content']
                
                # Extract LM Studio performance stats if available
                if 'stats' in response_data:
                    self.last_generation_stats = response_data['stats']
                    ASCIIColors.info(f"🚀 Performance: {self.last_generation_stats.get('tokens_per_second', 0):.1f} tok/s, "
                                   f"TTFT: {self.last_generation_stats.get('time_to_first_token', 0):.3f}s")
                
                return content
            else:
                return {"error": "No response content"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_performance_stats(self, text: str, start_time: float, first_token_time: float, end_time: float):
        """Calculate and store performance statistics."""
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        # Rough token estimation (more accurate would require tokenizer)
        estimated_tokens = len(text.split()) * 1.3  # Rough approximation
        tokens_per_second = estimated_tokens / total_time if total_time > 0 else 0
        
        self.last_generation_stats = {
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": ttft,
            "generation_time": total_time,
            "estimated_tokens": estimated_tokens
        }
        
        ASCIIColors.info(f"🚀 Performance: {tokens_per_second:.1f} tok/s, TTFT: {ttft:.3f}s")

    def listModels(self) -> List[Dict[str, Any]]:
        """
        List all available models in LM Studio with detailed information.

        Returns enhanced model info including state, quantization, context length.
        """
        try:
            if self.use_native_api:
                # Use native LM Studio API for richer model information
                response = requests.get(f"{self.native_base_url}/models",
                                      verify=self.verify_ssl_certificate)
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
            else:
                # Fall back to OpenAI-compatible endpoint
                response = requests.get(f"{self.openai_base_url}/models",
                                      headers={"Authorization": f"Bearer {self.service_key}"},
                                      verify=self.verify_ssl_certificate)
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            ASCIIColors.error(f"Failed to list models: {e}")
            return []

    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model ID to get info for. Uses current model if None.

        Returns:
            Dict with model information including GPU optimization details.
        """
        model_id = model_id or self.model_name
        if not model_id:
            return {"error": "No model specified"}

        try:
            if self.use_native_api:
                response = requests.get(f"{self.native_base_url}/models/{model_id}",
                                      verify=self.verify_ssl_certificate)
                response.raise_for_status()
                return response.json()
            else:
                # For OpenAI API, return basic info
                return {
                    "id": model_id,
                    "object": "model",
                    "binding": "lmstudio"
                }
        except Exception as e:
            ASCIIColors.error(f"Failed to get model info: {e}")
            return {"error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get the last generation's performance statistics."""
        return self.last_generation_stats.copy()

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using LM Studio.

        Args:
            text: Text to embed

        Returns:
            List of embedding values or empty list if failed
        """
        try:
            # Check if an embedding model is available
            models = self.listModels()
            embedding_models = [m for m in models if isinstance(m, dict) and m.get('type') == 'embeddings']

            if not embedding_models:
                ASCIIColors.warning("No embedding models available in LM Studio")
                return []

            embedding_model = embedding_models[0]['id']

            data = {
                "model": embedding_model,
                "input": text
            }

            endpoint = "embeddings"
            response = self._make_request(endpoint, data, use_native=self.use_native_api)
            response_data = response.json()

            if 'data' in response_data and response_data['data']:
                return response_data['data'][0]['embedding']
            else:
                return []

        except Exception as e:
            ASCIIColors.error(f"Embedding generation failed: {e}")
            return []

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text. Note: LM Studio doesn't expose tokenization directly.
        This is a placeholder implementation.
        """
        ASCIIColors.warning("LM Studio doesn't expose tokenization. Using rough estimation.")
        # Rough estimation: ~1.3 tokens per word
        words = text.split()
        return list(range(len(words)))  # Dummy token IDs

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize tokens. Placeholder implementation.
        """
        ASCIIColors.warning("LM Studio doesn't expose detokenization.")
        return f"[{len(tokens)} tokens]"

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using rough estimation.
        """
        # Rough estimation: ~1.3 tokens per word for most models
        return int(len(text.split()) * 1.3)

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: Optional[float] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             repeat_penalty: Optional[float] = None,
             repeat_last_n: Optional[int] = None,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        Conduct a chat session using LM Studio with a LollmsDiscussion object.
        """
        try:
            # Convert discussion to messages format
            messages = []

            # Get messages from discussion
            discussion_messages = discussion.get_messages()

            for msg in discussion_messages:
                role = "user" if msg.sender == discussion.user_name else "assistant"
                messages.append({
                    "role": role,
                    "content": msg.content
                })

            # Prepare request data
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature or 0.7,
                "top_p": top_p or 0.9,
                "max_tokens": n_predict or -1,
                "stream": stream or False,
                "seed": seed
            }

            start_time = time.time()

            if stream and streaming_callback:
                return self._handle_streaming_response(data, streaming_callback, start_time)
            else:
                return self._handle_non_streaming_response(data, start_time)

        except Exception as e:
            ASCIIColors.error(f"Chat failed: {e}")
            return {"error": str(e)}

    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model in LM Studio.

        Note: LM Studio loads models on-demand, so this just sets the model name.
        The actual loading happens when the first request is made.
        """
        try:
            # Check if model exists
            models = self.listModels()
            available_model_ids = [m['id'] if isinstance(m, dict) else str(m) for m in models]

            if model_name not in available_model_ids:
                ASCIIColors.error(f"Model '{model_name}' not found in LM Studio")
                return False

            self.model_name = model_name
            ASCIIColors.info(f"LM Studio model set to: {model_name}")
            return True

        except Exception as e:
            ASCIIColors.error(f"Failed to load model: {e}")
            return False

    def get_gpu_utilization(self) -> Dict[str, Any]:
        """Get real-time GPU utilization stats for performance monitoring."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                return {
                    "gpu_utilization": int(gpu_util),
                    "memory_used_mb": int(mem_used),
                    "memory_total_mb": int(mem_total),
                    "temperature_c": int(temp),
                    "memory_usage_percent": round(int(mem_used) / int(mem_total) * 100, 1)
                }
        except:
            pass
        return {}

    def auto_optimize_for_hardware(self) -> Dict[str, Any]:
        """
        Automatically optimize LM Studio settings for the current hardware.
        Returns recommended settings for RTX 3060 12GB and similar setups.
        """
        gpu_info = self.get_gpu_utilization()

        # RTX 3060 12GB optimized settings
        optimizations = {
            "recommended_context_length": 8192,  # Sweet spot for 12GB VRAM
            "recommended_batch_size": 512,
            "gpu_layers": -1,  # All layers on GPU
            "flash_attention": True,
            "rope_scaling": "linear",
            "recommended_models": [
                "gemma-3-12b-it",  # Multimodal
                "deepcoder-7b-aurora-v2",  # Programming
                "qwen2.5-coder-14b-instruct",  # Advanced coding
                "phi-3.5-vision-instruct",  # Vision tasks
                "llama-3.2-3b-instruct"  # Fast general purpose
            ]
        }

        if gpu_info:
            vram_gb = gpu_info["memory_total_mb"] / 1024
            if vram_gb >= 12:
                optimizations["max_model_size"] = "14B parameters"
                optimizations["recommended_quantization"] = "Q4_K_M or Q5_K_M"
            elif vram_gb >= 8:
                optimizations["max_model_size"] = "7B parameters"
                optimizations["recommended_quantization"] = "Q4_K_M"
            else:
                optimizations["max_model_size"] = "3B parameters"
                optimizations["recommended_quantization"] = "Q4_0"

        return optimizations

    def get_model_recommendations(self, task_type: str = "general") -> List[str]:
        """
        Get model recommendations based on task type and hardware.

        Args:
            task_type: Type of task ("programming", "vision", "reasoning", "multimodal", "general")
        """
        available_models = [m['id'] for m in self.listModels() if isinstance(m, dict)]

        recommendations = {
            "programming": [
                "deepcoder-7b-aurora-v2", "qwen2.5-coder-14b-instruct",
                "deepcoder-14b-preview.gguf", "deepcoder-1.5b-preview"
            ],
            "vision": [
                "phi-3.5-vision-instruct", "llama-3.2-vision", "qwen2-vl-7b-instruct",
                "granite-vision-3.2-2b"
            ],
            "reasoning": [
                "deepseek-r1-distill-qwen-14b", "cogito-v1-preview-qwen-14b",
                "l3.1-dark-reasoning-lewdplay-evo-hermes-r1"
            ],
            "multimodal": [
                "gemma-3-12b-it", "gemma-3-27b-abliterated-dpo", "phi-3.5-vision-instruct"
            ],
            "general": [
                "llama-3.2-3b-instruct", "gemma-3-4b-it-qat", "phi-3.5-vision-instruct"
            ]
        }

        task_models = recommendations.get(task_type, recommendations["general"])
        return [model for model in task_models if model in available_models]

    def auto_select_best_model(self, task_type: str = "general", prefer_loaded: bool = True) -> str:
        """
        Automatically select the best model for a given task type.

        Args:
            task_type: Type of task to optimize for
            prefer_loaded: Prefer already loaded models for faster response
        """
        models = self.listModels()

        if prefer_loaded:
            loaded_models = [m for m in models if isinstance(m, dict) and m.get('state') == 'loaded']
            if loaded_models:
                # Return first loaded model that matches task type
                recommendations = self.get_model_recommendations(task_type)
                for rec in recommendations:
                    for model in loaded_models:
                        if rec in model['id']:
                            return model['id']
                # Fall back to first loaded model
                return loaded_models[0]['id']

        # Get best available model for task
        recommendations = self.get_model_recommendations(task_type)
        if recommendations:
            return recommendations[0]

        # Ultimate fallback
        if models:
            first_model = models[0]
            return first_model['id'] if isinstance(first_model, dict) else str(first_model)

        return ""

    def get_system_prompt_for_task(self, task_type: str) -> str:
        """Get optimized system prompts for different task types."""
        prompts = {
            "programming": "You are an expert programmer and software engineer. Provide clean, efficient, well-documented code with proper error handling. Explain your reasoning and suggest best practices.",

            "vision": "You are an expert computer vision AI. Analyze images carefully and provide detailed, accurate descriptions. Focus on relevant details and spatial relationships.",

            "reasoning": "You are an expert reasoning AI. Think step by step, show your work, and explain your logical process. Break down complex problems into manageable parts.",

            "multimodal": "You are a versatile AI assistant capable of handling text, images, and complex reasoning tasks. Adapt your response style to the specific needs of each request.",

            "general": "You are a helpful, knowledgeable AI assistant. Provide accurate, clear, and useful responses while being concise and well-structured."
        }

        return prompts.get(task_type, prompts["general"])

    def create_optimized_client_config(self, task_type: str = "general") -> Dict[str, Any]:
        """
        Create an optimized LollmsClient configuration for LM Studio.

        Returns a complete config dict that can be passed to LollmsClient(**config)
        """
        best_model = self.auto_select_best_model(task_type)
        optimizations = self.auto_optimize_for_hardware()

        config = {
            "binding_name": "lmstudio",
            "host_address": self.host_address,
            "model_name": best_model,
            "llm_binding_config": {
                "use_native_api": True,
                "default_completion_format": ELF_COMPLETION_FORMAT.Chat
            },

            # Optimized generation parameters
            "temperature": 0.7 if task_type == "general" else 0.3,  # Lower temp for technical tasks
            "top_p": 0.9,
            "ctx_size": optimizations.get("recommended_context_length", 8192),
            "n_predict": 1024,
            "stream": True,  # Always use streaming for better UX

            # System prompt optimization
            "system_prompt": self.get_system_prompt_for_task(task_type)
        }

        return config

    def benchmark_model_performance(self, model_name: str = None, test_prompts: List[str] = None) -> Dict[str, Any]:
        """
        Benchmark a model's performance with various test prompts.

        Returns detailed performance metrics including tokens/second, TTFT, etc.
        """
        model_name = model_name or self.model_name
        if not model_name:
            return {"error": "No model specified"}

        if not test_prompts:
            test_prompts = [
                "Write a Python function to sort a list.",
                "Explain quantum computing in simple terms.",
                "What are the benefits of renewable energy?",
                "Create a simple HTML webpage structure.",
                "Describe the process of photosynthesis."
            ]

        results = {
            "model": model_name,
            "test_results": [],
            "average_tokens_per_second": 0,
            "average_ttft": 0,
            "total_test_time": 0
        }

        total_start = time.time()

        for i, prompt in enumerate(test_prompts):
            ASCIIColors.info(f"Running benchmark {i+1}/{len(test_prompts)}")

            start_time = time.time()
            response = self.generate_text(
                prompt=prompt,
                n_predict=100,  # Short responses for benchmarking
                stream=False,
                temperature=0.7
            )
            end_time = time.time()

            if isinstance(response, str):
                test_time = end_time - start_time
                estimated_tokens = len(response.split()) * 1.3
                tokens_per_second = estimated_tokens / test_time if test_time > 0 else 0

                test_result = {
                    "prompt": prompt[:50] + "...",
                    "response_length": len(response),
                    "estimated_tokens": estimated_tokens,
                    "generation_time": test_time,
                    "tokens_per_second": tokens_per_second
                }

                results["test_results"].append(test_result)

        total_time = time.time() - total_start
        results["total_test_time"] = total_time

        if results["test_results"]:
            avg_tps = sum(r["tokens_per_second"] for r in results["test_results"]) / len(results["test_results"])
            results["average_tokens_per_second"] = avg_tps

        return results


# Test function for the LM Studio binding
if __name__ == '__main__':
    ASCIIColors.yellow("Testing LM Studio Binding...")

    try:
        # Initialize binding
        binding = LMStudioBinding()

        # Test model listing
        ASCIIColors.cyan("Available models:")
        models = binding.listModels()
        for model in models[:5]:  # Show first 5 models
            if isinstance(model, dict):
                state = model.get('state', 'unknown')
                model_type = model.get('type', 'unknown')
                quant = model.get('quantization', 'unknown')
                print(f"  - {model['id']} ({model_type}, {quant}, {state})")

        if not models:
            ASCIIColors.error("No models found. Please load a model in LM Studio.")
            exit(1)

        # Test text generation
        ASCIIColors.cyan("\nTesting text generation...")
        prompt = "Write a Python function to calculate the factorial of a number."

        def test_callback(chunk, msg_type):
            if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                print(chunk, end="", flush=True)
            return True

        response = binding.generate_text(
            prompt=prompt,
            stream=True,
            streaming_callback=test_callback,
            n_predict=200
        )

        print(f"\n\nGeneration completed!")
        stats = binding.get_performance_stats()
        if stats:
            print(f"Performance: {stats}")

        ASCIIColors.green("✅ LM Studio binding test completed successfully!")

    except Exception as e:
        ASCIIColors.error(f"❌ Test failed: {e}")
        trace_exception(e)
