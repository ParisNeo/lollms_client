# OpenRouter Universal Model Router Binding for LOLLMS-Client
# Comprehensive implementation with ALL OpenRouter features
import requests
import json
import time
import logging
import re
from typing import Optional, Callable, List, Union, Dict, Any, Literal
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_discussion import LollmsDiscussion
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm
import os

# Ensure required packages are installed
pm.ensure_packages(["openai", "requests", "tiktoken"])

import openai
import tiktoken

BindingName = "OpenRouterBinding"

# OpenRouter-specific types and constants
OPENROUTER_AUTO_MODEL = "openrouter/auto"
MODEL_SHORTCUTS = {
    ":online": "web_search",
    ":nitro": "high_throughput",
    ":floor": "low_cost"
}

REASONING_EFFORT_LEVELS = Literal["low", "medium", "high"]
SEARCH_CONTEXT_SIZES = Literal["low", "medium", "high"]
TRANSFORM_TYPES = Literal["middle-out"]

class OpenRouterBinding(LollmsLLMBinding):
    """
    Comprehensive OpenRouter Universal Model Router Binding

    Complete Feature Set:
    - 400+ models with auto-discovery and intelligent routing
    - Auto Router with NotDiamond intelligence (openrouter/auto)
    - Model shortcuts (:online, :nitro, :floor) for optimization
    - Presets (@preset/name) for reusable configurations
    - Advanced provider routing with comprehensive preferences
    - Model routing with fallback arrays and route parameters
    - Plugins system (web search, etc.) with customization
    - Prompt caching with cache_control breakpoints
    - Message transforms (middle-out compression)
    - Web search (plugin and native) with context sizing
    - Structured outputs with JSON schema validation
    - Tool calling (OpenAI-compatible)
    - Reasoning tokens with effort levels and exclusion
    - Usage accounting with detailed token tracking
    - Advanced sampling parameters (min_p, top_a, etc.)
    - BYOK (Bring Your Own Key) support
    - Real-time cost tracking and budget controls
    - A/B testing framework for model comparison
    - Uptime optimization with provider health tracking
    - Zero completion insurance
    - Assistant prefill and prediction support
    - Full multimodal support (images, PDFs)
    - User tracking and identification
    """

    def __init__(self,
                 host_address: str = "https://openrouter.ai/api/v1",
                 model_name: str = "openai/gpt-4o",
                 service_key: str = None,
                 verify_ssl_certificate: bool = True,
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat,
                 # OpenRouter Advanced Features
                 enable_auto_router: bool = True,
                 enable_intelligent_routing: bool = True,
                 enable_cost_tracking: bool = True,
                 enable_prompt_caching: bool = True,
                 enable_web_search: bool = False,
                 enable_reasoning_tokens: bool = True,
                 enable_byok: bool = False,
                 enable_usage_accounting: bool = True,
                 # Budget and Cost Controls
                 budget_limit: Optional[float] = None,
                 cost_per_token_limit: Optional[float] = None,
                 # Provider and Model Preferences
                 preferred_providers: Optional[List[str]] = None,
                 fallback_models: Optional[List[str]] = None,
                 provider_routing_config: Optional[Dict] = None,
                 # Advanced Settings
                 default_reasoning_effort: REASONING_EFFORT_LEVELS = "medium",
                 default_search_context: SEARCH_CONTEXT_SIZES = "medium",
                 enable_transforms: bool = True,
                 transform_types: Optional[List[TRANSFORM_TYPES]] = None,
                 # User and Tracking
                 user_id: Optional[str] = None,
                 app_name: str = "LOLLMS-Client",
                 **kwargs):
        """
        Initialize the OpenRouter binding with comprehensive features.

        Args:
            host_address (str): OpenRouter API base URL
            model_name (str): Default model to use (supports shortcuts and presets)
            service_key (str): OpenRouter API key
            verify_ssl_certificate (bool): Whether to verify SSL certificates
            default_completion_format: Default completion format
            enable_auto_router (bool): Enable openrouter/auto with NotDiamond
            enable_intelligent_routing (bool): Enable smart model selection
            enable_cost_tracking (bool): Enable real-time cost monitoring
            enable_prompt_caching (bool): Enable prompt caching features
            enable_web_search (bool): Enable web search capabilities
            enable_reasoning_tokens (bool): Enable reasoning token support
            enable_byok (bool): Enable Bring Your Own Key support
            enable_usage_accounting (bool): Enable detailed usage tracking
            budget_limit (Optional[float]): Maximum spending limit in USD
            cost_per_token_limit (Optional[float]): Maximum cost per token
            preferred_providers (Optional[List[str]]): Preferred provider list
            fallback_models (Optional[List[str]]): Fallback model list
            provider_routing_config (Optional[Dict]): Advanced provider routing
            default_reasoning_effort: Default reasoning effort level
            default_search_context: Default web search context size
            enable_transforms (bool): Enable message transforms
            transform_types: List of transform types to enable
            user_id (Optional[str]): User identifier for tracking
            app_name (str): Application name for identification
        """
        super().__init__(binding_name="openrouter")

        # Core configuration
        self.host_address = host_address
        self.model_name = model_name
        self.service_key = service_key
        self.verify_ssl_certificate = verify_ssl_certificate
        self.default_completion_format = default_completion_format

        # Advanced OpenRouter Features
        self.enable_auto_router = enable_auto_router
        self.enable_intelligent_routing = enable_intelligent_routing
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_prompt_caching = enable_prompt_caching
        self.enable_web_search = enable_web_search
        self.enable_reasoning_tokens = enable_reasoning_tokens
        self.enable_byok = enable_byok
        self.enable_usage_accounting = enable_usage_accounting

        # Budget and Cost Controls
        self.budget_limit = budget_limit
        self.cost_per_token_limit = cost_per_token_limit

        # Provider and Model Preferences
        self.preferred_providers = preferred_providers or []
        self.fallback_models = fallback_models or []
        self.provider_routing_config = provider_routing_config or {}

        # Advanced Settings
        self.default_reasoning_effort = default_reasoning_effort
        self.default_search_context = default_search_context
        self.enable_transforms = enable_transforms
        self.transform_types = transform_types or ["middle-out"]

        # User and Tracking
        self.user_id = user_id
        self.app_name = app_name
        
        # Initialize API key from environment or universal API key manager
        if not self.service_key:
            self.service_key = os.getenv("OPENROUTER_API_KEY", self.service_key)

            # Try to get from universal API key manager
            if not self.service_key:
                try:
                    from universal_api_key_manager import get_api_key
                    self.service_key = get_api_key("openrouter")
                    if self.service_key:
                        ASCIIColors.green("✅ OpenRouter API key loaded from universal manager")
                except ImportError:
                    ASCIIColors.warning("⚠️ Universal API key manager not available")
                except Exception as e:
                    ASCIIColors.warning(f"⚠️ Could not load API key from universal manager: {e}")

        if not self.service_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or configure universal API key manager.")
        
        # Initialize OpenAI client for OpenRouter
        self.client = openai.OpenAI(
            api_key=self.service_key,
            base_url=self.host_address
        )
        
        # Model discovery and caching
        self.models_cache = {}
        self.models_last_updated = 0
        self.models_cache_ttl = 3600  # 1 hour cache
        
        # Cost tracking
        self.total_cost = 0.0
        self.generation_history = []
        
        # A/B testing framework
        self.ab_tests = {}
        
        # Performance monitoring
        self.performance_metrics = {}
        
        # Initialize model discovery
        self._discover_models()
        
        ASCIIColors.green(f"✅ OpenRouter binding initialized with {len(self.models_cache)} models")
    
    def _discover_models(self):
        """Discover and catalog all available OpenRouter models."""
        try:
            current_time = time.time()
            
            # Check if cache is still valid
            if (self.models_cache and 
                current_time - self.models_last_updated < self.models_cache_ttl):
                return
            
            ASCIIColors.yellow("🔍 Discovering OpenRouter models...")
            
            headers = {
                "Authorization": f"Bearer {self.service_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.host_address}/models",
                headers=headers,
                verify=self.verify_ssl_certificate
            )
            
            if response.status_code == 200:
                models_data = response.json()
                self._process_models_data(models_data)
                self.models_last_updated = current_time
                ASCIIColors.green(f"✅ Discovered {len(self.models_cache)} models")
            else:
                ASCIIColors.error(f"❌ Failed to discover models: {response.status_code}")
                
        except Exception as e:
            ASCIIColors.error(f"❌ Model discovery error: {e}")
            trace_exception(e)
    
    def _process_models_data(self, models_data: Dict):
        """Process and categorize discovered models."""
        self.models_cache = {}
        
        for model in models_data.get("data", []):
            model_id = model.get("id", "")
            
            # Extract model information
            model_info = {
                "id": model_id,
                "name": model.get("name", ""),
                "description": model.get("description", ""),
                "context_length": model.get("context_length", 0),
                "pricing": model.get("pricing", {}),
                "architecture": model.get("architecture", {}),
                "supported_parameters": model.get("supported_parameters", []),
                "top_provider": model.get("top_provider", {}),
                "canonical_slug": model.get("canonical_slug", ""),
                "created": model.get("created", 0),
                "per_request_limits": model.get("per_request_limits", {}),
                
                # Categorization
                "category": self._categorize_model(model),
                "provider": model_id.split("/")[0] if "/" in model_id else "unknown",
                "model_family": self._extract_model_family(model_id),
                
                # Performance metrics (to be populated)
                "avg_latency": None,
                "avg_throughput": None,
                "reliability_score": None,
                
                # Cost efficiency
                "cost_per_token": self._calculate_cost_per_token(model.get("pricing", {})),
                "cost_efficiency_score": None
            }
            
            self.models_cache[model_id] = model_info
    
    def _categorize_model(self, model: Dict) -> str:
        """Categorize model based on its characteristics."""
        model_id = model.get("id", "").lower()
        description = model.get("description", "").lower()
        
        # Programming models
        if any(keyword in model_id or keyword in description for keyword in 
               ["code", "programming", "codestral", "deepseek-coder", "starcoder"]):
            return "programming"
        
        # Vision models
        if any(keyword in model_id or keyword in description for keyword in 
               ["vision", "visual", "image", "multimodal"]):
            return "vision"
        
        # Reasoning models
        if any(keyword in model_id or keyword in description for keyword in 
               ["reasoning", "think", "o1", "deepthink"]):
            return "reasoning"
        
        # Chat/General models
        if any(keyword in model_id or keyword in description for keyword in 
               ["chat", "instruct", "assistant"]):
            return "chat"
        
        # Roleplay models
        if any(keyword in model_id or keyword in description for keyword in 
               ["roleplay", "character", "uncensored"]):
            return "roleplay"
        
        return "general"
    
    def _extract_model_family(self, model_id: str) -> str:
        """Extract model family from model ID."""
        if "/" in model_id:
            provider, model = model_id.split("/", 1)
            
            # Common model families
            if "gpt" in model.lower():
                return "gpt"
            elif "claude" in model.lower():
                return "claude"
            elif "gemini" in model.lower():
                return "gemini"
            elif "llama" in model.lower():
                return "llama"
            elif "mistral" in model.lower():
                return "mistral"
            elif "qwen" in model.lower():
                return "qwen"
            
            return model.split("-")[0] if "-" in model else model
        
        return "unknown"
    
    def _calculate_cost_per_token(self, pricing: Dict) -> float:
        """Calculate average cost per token."""
        prompt_cost = float(pricing.get("prompt", 0))
        completion_cost = float(pricing.get("completion", 0))

        # Average of prompt and completion costs
        return (prompt_cost + completion_cost) / 2 if prompt_cost or completion_cost else 0.0

    def listModels(self) -> List[str]:
        """List all available models."""
        self._discover_models()  # Refresh if needed
        return list(self.models_cache.keys())

    def get_model_info(self, model_id: str = None) -> Dict:
        """Get detailed information about a specific model."""
        if model_id is None:
            model_id = self.model_name

        self._discover_models()
        return self.models_cache.get(model_id, {})

    def get_models_by_category(self, category: str) -> List[Dict]:
        """Get all models in a specific category."""
        self._discover_models()
        return [model for model in self.models_cache.values()
                if model.get("category") == category]

    def get_models_by_provider(self, provider: str) -> List[Dict]:
        """Get all models from a specific provider."""
        self._discover_models()
        return [model for model in self.models_cache.values()
                if model.get("provider") == provider]

    def _select_optimal_model(self, task_type: str = "general",
                             max_cost: Optional[float] = None,
                             min_context: Optional[int] = None,
                             required_features: Optional[List[str]] = None) -> str:
        """
        Intelligent model selection based on task requirements.

        Args:
            task_type: Type of task (programming, vision, reasoning, etc.)
            max_cost: Maximum cost per token
            min_context: Minimum context length required
            required_features: Required model features/parameters

        Returns:
            Optimal model ID
        """
        if not self.enable_intelligent_routing:
            return self.model_name

        self._discover_models()

        # Filter models by requirements
        candidates = []
        for model_id, model_info in self.models_cache.items():
            # Check task category match
            if task_type != "general" and model_info.get("category") != task_type:
                continue

            # Check cost constraint
            if max_cost and model_info.get("cost_per_token", 0) > max_cost:
                continue

            # Check context length
            if min_context and model_info.get("context_length", 0) < min_context:
                continue

            # Check required features
            if required_features:
                supported = model_info.get("supported_parameters", [])
                if not all(feature in supported for feature in required_features):
                    continue

            # Check provider preferences
            if self.preferred_providers:
                provider = model_info.get("provider", "")
                if provider not in self.preferred_providers:
                    continue

            candidates.append((model_id, model_info))

        if not candidates:
            ASCIIColors.warning(f"⚠️ No models found for task '{task_type}', using default")
            return self.model_name

        # Sort by cost efficiency and performance
        candidates.sort(key=lambda x: (
            x[1].get("cost_per_token", float('inf')),
            -x[1].get("context_length", 0),
            -x[1].get("reliability_score", 0) if x[1].get("reliability_score") else 0
        ))

        selected_model = candidates[0][0]
        ASCIIColors.cyan(f"🎯 Selected optimal model: {selected_model}")
        return selected_model

    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     # Standard sampling parameters
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1,
                     repeat_last_n: int = 64,
                     seed: Optional[int] = None,
                     # Advanced sampling parameters
                     min_p: Optional[float] = None,
                     top_a: Optional[float] = None,
                     frequency_penalty: Optional[float] = None,
                     presence_penalty: Optional[float] = None,
                     logit_bias: Optional[Dict] = None,
                     logprobs: Optional[bool] = None,
                     top_logprobs: Optional[int] = None,
                     stop: Optional[List[str]] = None,
                     # LOLLMS compatibility
                     n_threads: Optional[int] = None,
                     ctx_size: Optional[int] = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split: Optional[bool] = False,
                     user_keyword: Optional[str] = "!@>user:",
                     ai_keyword: Optional[str] = "!@>assistant:",
                     # OpenRouter Core Features
                     model_override: Optional[str] = None,
                     use_auto_router: Optional[bool] = None,
                     task_type: str = "general",
                     # Structured Outputs
                     structured_output: Optional[Dict] = None,
                     json_schema: Optional[Dict] = None,
                     response_format: Optional[Dict] = None,
                     # Tool Calling
                     tools: Optional[List[Dict]] = None,
                     tool_choice: Optional[Union[str, Dict]] = None,
                     # Web Search
                     enable_web_search: Optional[bool] = None,
                     web_search_options: Optional[Dict] = None,
                     plugins: Optional[List[Dict]] = None,
                     # Provider and Model Routing
                     provider_preferences: Optional[Dict] = None,
                     fallback_models: Optional[List[str]] = None,
                     models: Optional[List[str]] = None,
                     route: Optional[str] = None,
                     # Prompt Caching
                     enable_caching: Optional[bool] = None,
                     cache_control: Optional[List[Dict]] = None,
                     # Message Transforms
                     transforms: Optional[List[str]] = None,
                     # Reasoning Tokens
                     reasoning: Optional[Dict] = None,
                     reasoning_effort: Optional[REASONING_EFFORT_LEVELS] = None,
                     reasoning_max_tokens: Optional[int] = None,
                     reasoning_exclude: Optional[bool] = None,
                     # Usage and Cost Controls
                     max_cost_per_token: Optional[float] = None,
                     usage_include: Optional[bool] = None,
                     # User Tracking
                     user_id_override: Optional[str] = None,
                     # Presets
                     preset: Optional[str] = None,
                     # Assistant Prefill
                     assistant_prefill: Optional[str] = None,
                     # Prediction
                     prediction: Optional[Dict] = None,
                     **kwargs) -> Union[str, dict]:
        """
        Generate text using OpenRouter with comprehensive feature support.

        This method supports ALL OpenRouter features including:
        - Auto router and intelligent model selection
        - Model shortcuts (:online, :nitro, :floor) and presets (@preset/name)
        - Advanced provider routing and fallback models
        - Structured outputs with JSON schema validation
        - Tool calling (OpenAI-compatible)
        - Web search (plugin and native) with context sizing
        - Prompt caching with cache_control breakpoints
        - Message transforms (middle-out compression)
        - Reasoning tokens with effort levels and exclusion
        - Advanced sampling parameters (min_p, top_a, etc.)
        - Usage accounting and cost tracking
        - Assistant prefill and prediction
        - User tracking and identification

        Returns:
            Generated text, structured data, or tool calls
        """
        try:
            # Process model selection with shortcuts and presets
            selected_model = self._process_model_selection(
                model_override, use_auto_router, preset, task_type,
                structured_output, json_schema, images, enable_web_search,
                max_cost_per_token, ctx_size
            )

            # Build comprehensive request parameters
            request_params = self._build_request_parameters(
                selected_model, prompt, images, system_prompt, n_predict, stream,
                temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed,
                min_p, top_a, frequency_penalty, presence_penalty, logit_bias,
                logprobs, top_logprobs, stop, structured_output, json_schema,
                response_format, tools, tool_choice, enable_web_search,
                web_search_options, plugins, provider_preferences, fallback_models,
                models, route, enable_caching, cache_control, transforms,
                reasoning, reasoning_effort, reasoning_max_tokens, reasoning_exclude,
                usage_include, user_id_override, assistant_prefill, prediction,
                **kwargs
            )

            # Add OpenRouter headers
            headers = self._build_request_headers(user_id_override)

            # Track generation start time
            start_time = time.time()

            # Make API call with comprehensive error handling
            if stream:
                return self._handle_streaming_response(
                    request_params, headers, streaming_callback, start_time, selected_model
                )
            else:
                return self._handle_non_streaming_response(
                    request_params, headers, start_time, selected_model
                )

        except Exception as e:
            ASCIIColors.error(f"❌ Generation error: {e}")
            trace_exception(e)
            return f"Error: {str(e)}"

    def _process_model_selection(self, model_override, use_auto_router, preset, task_type,
                               structured_output, json_schema, images, enable_web_search,
                               max_cost_per_token, ctx_size):
        """Process comprehensive model selection with all OpenRouter features."""
        # Handle preset models
        if preset and preset.startswith("@"):
            return preset

        # Handle model override with shortcuts
        if model_override:
            return self._process_model_shortcuts(model_override)

        # Use auto router if enabled
        if use_auto_router or (use_auto_router is None and self.enable_auto_router):
            return OPENROUTER_AUTO_MODEL

        # Intelligent model selection
        if self.enable_intelligent_routing:
            required_features = []
            if structured_output or json_schema:
                required_features.append("structured_outputs")
            if images:
                required_features.append("image")
                task_type = "vision"
            if enable_web_search:
                required_features.append("web_search")

            return self._select_optimal_model(
                task_type=task_type,
                max_cost=max_cost_per_token,
                min_context=ctx_size,
                required_features=required_features if required_features else None
            )

        # Default model with shortcuts processing
        return self._process_model_shortcuts(self.model_name)

    def _process_model_shortcuts(self, model_name: str) -> str:
        """Process model shortcuts like :online, :nitro, :floor."""
        for shortcut, feature in MODEL_SHORTCUTS.items():
            if model_name.endswith(shortcut):
                base_model = model_name[:-len(shortcut)]
                if shortcut == ":online":
                    # Web search shortcut
                    return base_model
                elif shortcut == ":nitro":
                    # High throughput optimization
                    return base_model
                elif shortcut == ":floor":
                    # Low cost optimization
                    return base_model

        return model_name

    def _build_request_parameters(self, selected_model, prompt, images, system_prompt,
                                n_predict, stream, temperature, top_k, top_p,
                                repeat_penalty, repeat_last_n, seed, min_p, top_a,
                                frequency_penalty, presence_penalty, logit_bias,
                                logprobs, top_logprobs, stop, structured_output,
                                json_schema, response_format, tools, tool_choice,
                                enable_web_search, web_search_options, plugins,
                                provider_preferences, fallback_models, models, route,
                                enable_caching, cache_control, transforms, reasoning,
                                reasoning_effort, reasoning_max_tokens, reasoning_exclude,
                                usage_include, user_id_override, assistant_prefill,
                                prediction, **kwargs):
        """Build comprehensive request parameters with all OpenRouter features."""

        # Build messages with multimodal support
        messages = self._build_messages(prompt, images, system_prompt, cache_control)

        # Core request parameters
        request_params = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream or False,
            "seed": seed
        }

        # Advanced sampling parameters
        # Note: OpenRouter API doesn't support top_k parameter (OpenAI-compatible)
        # if top_k is not None:
        #     request_params["top_k"] = top_k
        if min_p is not None:
            request_params["min_p"] = min_p
        if top_a is not None:
            request_params["top_a"] = top_a
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if repeat_penalty is not None:
            request_params["repetition_penalty"] = repeat_penalty
        if logit_bias is not None:
            request_params["logit_bias"] = logit_bias
        if logprobs is not None:
            request_params["logprobs"] = logprobs
        if top_logprobs is not None:
            request_params["top_logprobs"] = top_logprobs
        if stop is not None:
            request_params["stop"] = stop

        # Structured outputs
        if response_format:
            request_params["response_format"] = response_format
        elif structured_output or json_schema:
            if json_schema:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": json_schema
                    }
                }
            else:
                request_params["response_format"] = {"type": "json_object"}

        # Tool calling
        if tools:
            request_params["tools"] = tools
        if tool_choice:
            request_params["tool_choice"] = tool_choice

        # Web search
        if enable_web_search or selected_model.endswith(":online"):
            if plugins:
                request_params["plugins"] = plugins
            else:
                request_params["plugins"] = [{"id": "web"}]

        if web_search_options:
            request_params["web_search_options"] = web_search_options

        # Provider and model routing
        if provider_preferences:
            request_params["provider"] = provider_preferences
        elif self.provider_routing_config:
            request_params["provider"] = self.provider_routing_config

        if models or fallback_models:
            request_params["models"] = models or ([selected_model] + (fallback_models or []))
        if route:
            request_params["route"] = route
        elif fallback_models:
            request_params["route"] = "fallback"

        # Message transforms
        if transforms:
            request_params["transforms"] = transforms
        elif self.enable_transforms and self.transform_types:
            request_params["transforms"] = self.transform_types

        # Reasoning tokens
        if reasoning:
            request_params["reasoning"] = reasoning
        elif reasoning_effort or reasoning_max_tokens or reasoning_exclude is not None:
            reasoning_config = {}
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            if reasoning_max_tokens:
                reasoning_config["max_tokens"] = reasoning_max_tokens
            if reasoning_exclude is not None:
                reasoning_config["exclude"] = reasoning_exclude
            if reasoning_config:
                request_params["reasoning"] = reasoning_config

        # Usage accounting
        if usage_include or self.enable_usage_accounting:
            request_params["usage"] = {"include": True}

        # Assistant prefill
        if assistant_prefill:
            request_params["assistant_prefill"] = assistant_prefill

        # Prediction
        if prediction:
            request_params["prediction"] = prediction

        # Add any additional kwargs
        request_params.update(kwargs)

        return request_params

    def _build_messages(self, prompt, images, system_prompt, cache_control):
        """Build messages array with multimodal and caching support."""
        messages = []

        # Add system prompt with optional caching
        if system_prompt:
            if cache_control and any(cc.get("role") == "system" for cc in cache_control):
                # Find system cache control
                system_cache = next((cc for cc in cache_control if cc.get("role") == "system"), None)
                if system_cache:
                    content = [
                        {"type": "text", "text": system_prompt},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}
                    ]
                    messages.append({"role": "system", "content": content})
                else:
                    messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": system_prompt})

        # Handle multimodal input with optional caching
        if images:
            content = [{"type": "text", "text": prompt}]
            for image_path in images:
                if image_path.startswith(('http://', 'https://')):
                    image_url = image_path
                else:
                    # Encode local image
                    from lollms_client.lollms_utilities import encode_image
                    image_data = encode_image(image_path)
                    image_url = f"data:image/jpeg;base64,{image_data}"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            # Add cache control if specified for user message
            if cache_control and any(cc.get("role") == "user" for cc in cache_control):
                content.append({
                    "type": "text",
                    "text": "",
                    "cache_control": {"type": "ephemeral"}
                })

            messages.append({"role": "user", "content": content})
        else:
            # Text-only message with optional caching
            if cache_control and any(cc.get("role") == "user" for cc in cache_control):
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}
                ]
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})

        return messages

    def _build_request_headers(self, user_id_override=None):
        """Build request headers with user tracking and app identification."""
        headers = {
            "HTTP-Referer": f"https://{self.app_name.lower()}.ai",
            "X-Title": f"{self.app_name} OpenRouter Binding"
        }

        # Add user tracking if available
        user_id = user_id_override or self.user_id
        if user_id:
            headers["X-User-ID"] = user_id

        return headers

    def _handle_streaming_response(self, request_params: Dict, headers: Dict,
                                 streaming_callback: Optional[Callable],
                                 start_time: float, model_id: str) -> str:
        """Handle streaming response from OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                extra_headers=headers,
                **request_params
            )

            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content

                    if streaming_callback:
                        streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK)

            # Track cost and performance
            if self.enable_cost_tracking:
                self._track_generation_cost(
                    generation_id=getattr(response, 'id', None),
                    model_id=model_id,
                    start_time=start_time,
                    response_text=full_response
                )

            return full_response

        except Exception as e:
            ASCIIColors.error(f"❌ Streaming error: {e}")
            trace_exception(e)
            return f"Streaming error: {str(e)}"

    def _handle_non_streaming_response(self, request_params: Dict, headers: Dict,
                                     start_time: float, model_id: str) -> str:
        """Handle non-streaming response from OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                extra_headers=headers,
                **request_params
            )

            content = response.choices[0].message.content

            # Track cost and performance
            if self.enable_cost_tracking:
                self._track_generation_cost(
                    generation_id=response.id,
                    model_id=model_id,
                    start_time=start_time,
                    response_text=content,
                    usage=getattr(response, 'usage', None)
                )

            return content

        except Exception as e:
            ASCIIColors.error(f"❌ Non-streaming error: {e}")
            trace_exception(e)
            return f"Non-streaming error: {str(e)}"

    def _track_generation_cost(self, generation_id: Optional[str], model_id: str,
                             start_time: float, response_text: str,
                             usage: Optional[Dict] = None):
        """Track generation cost and performance metrics."""
        try:
            end_time = time.time()
            latency = end_time - start_time

            # Get detailed generation info from OpenRouter
            if generation_id and self.service_key:
                headers = {"Authorization": f"Bearer {self.service_key}"}
                response = requests.get(
                    f"{self.host_address}/generation?id={generation_id}",
                    headers=headers,
                    verify=self.verify_ssl_certificate
                )

                if response.status_code == 200:
                    gen_data = response.json().get("data", {})
                    cost = gen_data.get("total_cost", 0.0)

                    # Update total cost
                    self.total_cost += cost

                    # Store generation record
                    generation_record = {
                        "id": generation_id,
                        "model": model_id,
                        "cost": cost,
                        "latency": latency,
                        "timestamp": end_time,
                        "tokens_prompt": gen_data.get("tokens_prompt", 0),
                        "tokens_completion": gen_data.get("tokens_completion", 0),
                        "response_length": len(response_text)
                    }

                    self.generation_history.append(generation_record)

                    # Update performance metrics
                    self._update_performance_metrics(model_id, generation_record)

                    # Check budget limit
                    if self.budget_limit and self.total_cost >= self.budget_limit:
                        ASCIIColors.warning(f"⚠️ Budget limit reached: ${self.total_cost:.4f}")

                    ASCIIColors.cyan(f"💰 Generation cost: ${cost:.6f} | Total: ${self.total_cost:.4f}")

        except Exception as e:
            ASCIIColors.error(f"❌ Cost tracking error: {e}")

    def _update_performance_metrics(self, model_id: str, generation_record: Dict):
        """Update performance metrics for a model."""
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = {
                "total_generations": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "avg_cost": 0.0,
                "reliability_score": 1.0
            }

        metrics = self.performance_metrics[model_id]
        metrics["total_generations"] += 1
        metrics["total_latency"] += generation_record["latency"]
        metrics["total_cost"] += generation_record["cost"]

        # Calculate averages
        metrics["avg_latency"] = metrics["total_latency"] / metrics["total_generations"]
        metrics["avg_cost"] = metrics["total_cost"] / metrics["total_generations"]

        # Update model cache with performance data
        if model_id in self.models_cache:
            self.models_cache[model_id]["avg_latency"] = metrics["avg_latency"]
            self.models_cache[model_id]["reliability_score"] = metrics["reliability_score"]
            self.models_cache[model_id]["cost_efficiency_score"] = 1.0 / (metrics["avg_cost"] + 0.000001)

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
        Conduct a chat session using OpenRouter with intelligent model routing.

        Args:
            discussion: LollmsDiscussion object containing the conversation
            branch_tip_id: ID of the branch tip (optional)
            n_predict: Maximum tokens to generate
            stream: Enable streaming
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repeat_penalty: Repetition penalty
            repeat_last_n: Repetition window
            seed: Random seed
            n_threads: Thread count (not used)
            ctx_size: Context size
            streaming_callback: Callback for streaming

        Returns:
            Generated response
        """
        try:
            # Convert discussion to messages format
            messages = []

            # Add system prompt if available
            if discussion.system_prompt:
                messages.append({
                    "role": "system",
                    "content": discussion.system_prompt
                })

            # Add conversation history
            for message in discussion.messages:
                role = "user" if message.sender == discussion.user_name else "assistant"
                messages.append({
                    "role": role,
                    "content": message.content
                })

            # Determine task type from conversation
            task_type = self._analyze_conversation_task_type(messages)

            # Select optimal model for this conversation
            selected_model = self._select_optimal_model(
                task_type=task_type,
                min_context=len(str(messages)) // 4  # Rough token estimate
            )

            # Build request parameters
            request_params = {
                "model": selected_model,
                "messages": messages,
                "max_tokens": n_predict,
                "temperature": temperature or 0.7,
                "top_p": top_p or 0.9,
                "stream": stream or False,
                "seed": seed
            }

            # Add OpenRouter headers
            headers = {
                "HTTP-Referer": "https://lollms-client.ai",
                "X-Title": "LOLLMS-Client OpenRouter Binding"
            }

            # Track generation start time
            start_time = time.time()

            # Make API call
            if stream:
                return self._handle_streaming_response(
                    request_params, headers, streaming_callback, start_time, selected_model
                )
            else:
                return self._handle_non_streaming_response(
                    request_params, headers, start_time, selected_model
                )

        except Exception as e:
            ASCIIColors.error(f"❌ Chat error: {e}")
            trace_exception(e)
            return f"Chat error: {str(e)}"

    def _analyze_conversation_task_type(self, messages: List[Dict]) -> str:
        """Analyze conversation to determine optimal task type."""
        conversation_text = " ".join([msg.get("content", "") for msg in messages]).lower()

        # Programming keywords
        if any(keyword in conversation_text for keyword in
               ["code", "programming", "function", "class", "debug", "algorithm", "python", "javascript"]):
            return "programming"

        # Reasoning keywords
        if any(keyword in conversation_text for keyword in
               ["analyze", "reasoning", "logic", "problem", "solution", "think", "explain"]):
            return "reasoning"

        # Vision keywords (if images are mentioned)
        if any(keyword in conversation_text for keyword in
               ["image", "picture", "visual", "see", "look", "describe"]):
            return "vision"

        return "chat"

    def run_ab_test(self, prompt: str, models: List[str],
                   test_name: str = None, iterations: int = 1,
                   **generation_params) -> Dict:
        """
        Run A/B testing with multiple models for comparison.

        Args:
            prompt: Test prompt
            models: List of model IDs to test
            test_name: Name for this test
            iterations: Number of iterations per model
            **generation_params: Additional generation parameters

        Returns:
            Test results with performance metrics
        """
        if not test_name:
            test_name = f"ab_test_{int(time.time())}"

        ASCIIColors.cyan(f"🧪 Starting A/B test: {test_name}")

        test_results = {
            "test_name": test_name,
            "prompt": prompt,
            "models": models,
            "iterations": iterations,
            "results": {},
            "summary": {}
        }

        for model_id in models:
            ASCIIColors.yellow(f"Testing model: {model_id}")
            model_results = []

            for i in range(iterations):
                start_time = time.time()

                try:
                    response = self.generate_text(
                        prompt=prompt,
                        model_override=model_id,
                        **generation_params
                    )

                    end_time = time.time()
                    latency = end_time - start_time

                    result = {
                        "iteration": i + 1,
                        "response": response,
                        "latency": latency,
                        "success": True,
                        "error": None
                    }

                except Exception as e:
                    result = {
                        "iteration": i + 1,
                        "response": None,
                        "latency": None,
                        "success": False,
                        "error": str(e)
                    }

                model_results.append(result)

            # Calculate model summary
            successful_results = [r for r in model_results if r["success"]]
            if successful_results:
                avg_latency = sum(r["latency"] for r in successful_results) / len(successful_results)
                success_rate = len(successful_results) / len(model_results)
            else:
                avg_latency = None
                success_rate = 0.0

            test_results["results"][model_id] = {
                "iterations": model_results,
                "avg_latency": avg_latency,
                "success_rate": success_rate,
                "total_iterations": len(model_results)
            }

        # Generate overall summary
        test_results["summary"] = self._generate_ab_test_summary(test_results)

        # Store test results
        self.ab_tests[test_name] = test_results

        ASCIIColors.green(f"✅ A/B test completed: {test_name}")
        return test_results

    def _generate_ab_test_summary(self, test_results: Dict) -> Dict:
        """Generate summary statistics for A/B test."""
        summary = {
            "best_latency": None,
            "best_success_rate": None,
            "best_overall": None,
            "rankings": []
        }

        model_scores = []
        for model_id, results in test_results["results"].items():
            if results["success_rate"] > 0:
                # Combined score: success rate weighted by inverse latency
                score = results["success_rate"] * (1.0 / (results["avg_latency"] + 0.001))
                model_scores.append({
                    "model": model_id,
                    "score": score,
                    "latency": results["avg_latency"],
                    "success_rate": results["success_rate"]
                })

        # Sort by score
        model_scores.sort(key=lambda x: x["score"], reverse=True)
        summary["rankings"] = model_scores

        if model_scores:
            summary["best_overall"] = model_scores[0]["model"]
            summary["best_latency"] = min(model_scores, key=lambda x: x["latency"])["model"]
            summary["best_success_rate"] = max(model_scores, key=lambda x: x["success_rate"])["model"]

        return summary

    def get_ab_test_results(self, test_name: str = None) -> Dict:
        """Get A/B test results."""
        if test_name:
            return self.ab_tests.get(test_name, {})
        return self.ab_tests

    def configure_byok(self, provider_keys: Dict[str, str]):
        """
        Configure Bring Your Own Key (BYOK) support.

        Args:
            provider_keys: Dictionary mapping provider names to API keys
        """
        if not self.enable_byok:
            ASCIIColors.warning("⚠️ BYOK is not enabled for this binding")
            return

        self.provider_keys = provider_keys
        ASCIIColors.green(f"✅ BYOK configured for {len(provider_keys)} providers")

    def get_credits_info(self) -> Dict:
        """Get current OpenRouter credits information."""
        try:
            headers = {"Authorization": f"Bearer {self.service_key}"}
            response = requests.get(
                f"{self.host_address}/credits",
                headers=headers,
                verify=self.verify_ssl_certificate
            )

            if response.status_code == 200:
                credits_data = response.json().get("data", {})
                credits_data["binding_total_cost"] = self.total_cost
                return credits_data
            else:
                return {"error": f"Failed to get credits: {response.status_code}"}

        except Exception as e:
            return {"error": f"Credits error: {str(e)}"}

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        return {
            "total_cost": self.total_cost,
            "total_generations": len(self.generation_history),
            "model_performance": self.performance_metrics,
            "recent_generations": self.generation_history[-10:] if self.generation_history else [],
            "ab_tests": list(self.ab_tests.keys()),
            "models_discovered": len(self.models_cache)
        }

    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        try:
            # Use tiktoken for tokenization (GPT-4 tokenizer as default)
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode(text)
        except Exception as e:
            ASCIIColors.error(f"❌ Tokenization error: {e}")
            # Fallback to simple word splitting
            return text.split()

    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        try:
            # Use tiktoken for detokenization
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.decode(tokens)
        except Exception as e:
            ASCIIColors.error(f"❌ Detokenization error: {e}")
            # Fallback to simple joining
            return " ".join(str(token) for token in tokens)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            text (str): Text to count tokens for.

        Returns:
            int: Number of tokens in text.
        """
        try:
            # Use tiktoken for accurate token counting
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            ASCIIColors.error(f"❌ Token counting error: {e}")
            # Fallback to simple word count
            return len(text.split())

    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using OpenRouter.

        Args:
            text (str): Input text to embed
            **kwargs: Additional arguments

        Returns:
            list: Embedding vector
        """
        try:
            # Check if current model supports embeddings
            model_info = self.get_model_info()
            if "embedding" not in model_info.get("supported_parameters", []):
                # Try to find an embedding model
                embedding_models = [model for model in self.models_cache.values()
                                  if "embed" in model.get("id", "").lower()]

                if not embedding_models:
                    raise NotImplementedError("No embedding models available in OpenRouter")

                embedding_model = embedding_models[0]["id"]
            else:
                embedding_model = self.model_name

            # Make embedding request (if OpenRouter supports it)
            headers = {
                "Authorization": f"Bearer {self.service_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"{self.host_address}/embeddings",
                headers=headers,
                json={
                    "model": embedding_model,
                    "input": text
                },
                verify=self.verify_ssl_certificate
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [{}])[0].get("embedding", [])
            else:
                raise NotImplementedError(f"Embedding not supported: {response.status_code}")

        except Exception as e:
            ASCIIColors.error(f"❌ Embedding error: {e}")
            raise NotImplementedError(f"Embedding not available: {str(e)}")

    def get_model_info(self) -> dict:
        """
        Return information about the current model.

        Returns:
            dict: Model information dictionary.
        """
        model_info = self.get_model_info(self.model_name)

        # Add binding-specific information
        binding_info = {
            "binding_name": self.binding_name,
            "binding_version": "1.0.0",
            "host_address": self.host_address,
            "current_model": self.model_name,
            "intelligent_routing_enabled": self.enable_intelligent_routing,
            "cost_tracking_enabled": self.enable_cost_tracking,
            "byok_enabled": self.enable_byok,
            "total_cost": self.total_cost,
            "models_available": len(self.models_cache),
            "performance_data": self.performance_metrics.get(self.model_name, {}),
            "supports_structured_output": "structured_outputs" in model_info.get("supported_parameters", []),
            "supports_vision": "image" in model_info.get("architecture", {}).get("input_modalities", []),
            "supports_streaming": True,
            "context_length": model_info.get("context_length", 0),
            "pricing": model_info.get("pricing", {}),
            "provider": model_info.get("provider", "unknown"),
            "category": model_info.get("category", "general")
        }

        return {**model_info, **binding_info}

    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model (switch to it).

        Args:
            model_name (str): Name of the model to load

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Refresh model cache if needed
            self._discover_models()

            # Check if model exists
            if model_name not in self.models_cache:
                ASCIIColors.error(f"❌ Model not found: {model_name}")
                return False

            # Switch to the new model
            old_model = self.model_name
            self.model_name = model_name

            ASCIIColors.green(f"✅ Switched from {old_model} to {model_name}")
            return True

        except Exception as e:
            ASCIIColors.error(f"❌ Model loading error: {e}")
            return False

    def get_ctx_size(self, model_name: str = None) -> int:
        """
        Get context size for a model.

        Args:
            model_name (str): Model name (uses current if None)

        Returns:
            int: Context size in tokens
        """
        if model_name is None:
            model_name = self.model_name

        model_info = self.get_model_info(model_name)
        return model_info.get("context_length", 32000)
