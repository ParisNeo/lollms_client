# lollms_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, Callable, List, Union
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
import importlib
from pathlib import Path
from typing import Optional, Dict, List
from ascii_colors import trace_exception, ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import ImageTokenizer
import re
class LollmsLLMBinding(ABC):
    """Abstract base class for all LOLLMS LLM bindings"""
    
    def __init__(self, 
                 binding_name: Optional[str] ="unknown"
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            binding_name (Optional[str]): The name of the bindingto be used
        """
        self.binding_name=binding_name
        self.model_name = None #Must be set by the instance
    
    @abstractmethod
    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
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
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        pass

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
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            messages (List[Dict]): A openai compatible list of messages
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        ASCIIColors.red("This binding does not support generate_from_messages")


    @abstractmethod
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
        A method to conduct a chat session with the model using a LollmsDiscussion object.
        This method is responsible for formatting the discussion into the specific
        format required by the model's API and then calling the generation endpoint.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (Optional[float]): Sampling temperature.
            top_k (Optional[int]): Top-k sampling parameter.
            top_p (Optional[float]): Top-p sampling parameter.
            repeat_penalty (Optional[float]): Penalty for repeated tokens.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty.
            seed (Optional[int]): Random seed for generation.
            n_threads (Optional[int]): Number of threads to use.
            ctx_size (Optional[int]): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        pass

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Retrieves context size for a model from a hardcoded list.

        This method checks if the model name contains a known base model identifier
        (e.g., 'llama3.1', 'gemma2') to determine its context length. It's intended
        as a failsafe when the context size cannot be retrieved directly from the
        Ollama API.
        """
        if model_name is None:
            model_name = self.model_name

        # Hardcoded context sizes for popular models. More specific names (e.g., 'llama3.1')
        # should appear, as they will be checked first due to the sorting logic below.
        known_contexts = {
            'agentica-org/deepcoder-14b-preview': 8192,
            'agentica-org/deepcoder-14b-preview:free': 8192,
            'ai21/jamba-large-1.7': 256000,
            'ai21/jamba-mini-1.7': 256000,
            'aion-labs/aion-1.0': 8192,
            'aion-labs/aion-1.0-mini': 8192,
            'aion-labs/aion-rp-llama-3.1-8b': 131072,
            'alfredpros/codellama-7b-instruct-solidity': 16384,
            'alpindale/goliath-120b': 4096,
            'amazon/nova-lite-v1': 32768,
            'amazon/nova-micro-v1': 32768,
            'amazon/nova-pro-v1': 32768,
            'anthracite-org/magnum-v2-72b': 131072,
            'anthracite-org/magnum-v4-72b': 131072,
            'anthropic/claude-3-haiku': 200000,
            'anthropic/claude-3-haiku:beta': 200000,
            'anthropic/claude-3-opus': 200000,
            'anthropic/claude-3-opus:beta': 200000,
            'anthropic/claude-3.5-haiku': 200000,
            'anthropic/claude-3.5-haiku-20241022': 200000,
            'anthropic/claude-3.5-haiku:beta': 200000,
            'anthropic/claude-3.5-sonnet': 200000,
            'anthropic/claude-3.5-sonnet-20240620': 200000,
            'anthropic/claude-3.5-sonnet-20240620:beta': 200000,
            'anthropic/claude-3.5-sonnet:beta': 200000,
            'anthropic/claude-3.7-sonnet': 200000,
            'anthropic/claude-3.7-sonnet:beta': 200000,
            'anthropic/claude-3.7-sonnet:thinking': 200000,
            'anthropic/claude-opus-4': 200000,
            'anthropic/claude-opus-4.1': 200000,
            'anthropic/claude-sonnet-4': 200000,
            'arcee-ai/coder-large': 32768,
            'arcee-ai/maestro-reasoning': 32768,
            'arcee-ai/spotlight': 32768,
            'arcee-ai/virtuoso-large': 32768,
            'arliai/qwq-32b-arliai-rpr-v1': 8192,
            'arliai/qwq-32b-arliai-rpr-v1:free': 8192,
            'baidu/ernie-4.5-300b-a47b': 128000,
            'bytedance/ui-tars-1.5-7b': 8192,
            'cognitivecomputations/dolphin-mistral-24b-venice-edition:free': 32768,
            'cognitivecomputations/dolphin-mixtral-8x22b': 65536,
            'cognitivecomputations/dolphin3.0-mistral-24b': 32768,
            'cognitivecomputations/dolphin3.0-mistral-24b:free': 32768,
            'cognitivecomputations/dolphin3.0-r1-mistral-24b': 32768,
            'cognitivecomputations/dolphin3.0-r1-mistral-24b:free': 32768,
            'cohere/command': 8192,
            'cohere/command-a': 8192,
            'cohere/command-r': 128000,
            'cohere/command-r-03-2024': 128000,
            'cohere/command-r-08-2024': 128000,
            'cohere/command-r-plus': 128000,
            'cohere/command-r-plus-04-2024': 128000,
            'cohere/command-r-plus-08-2024': 128000,
            'cohere/command-r7b-12-2024': 128000,
            'deepseek/deepseek-chat': 32768,
            'deepseek/deepseek-chat-v3-0324': 32768,
            'deepseek/deepseek-chat-v3-0324:free': 32768,
            'deepseek/deepseek-prover-v2': 131072,
            'deepseek/deepseek-r1': 32768,
            'deepseek/deepseek-r1-0528': 32768,
            'deepseek/deepseek-r1-0528-qwen3-8b': 32768,
            'deepseek/deepseek-r1-0528-qwen3-8b:free': 32768,
            'deepseek/deepseek-r1-0528:free': 32768,
            'deepseek/deepseek-r1-distill-llama-70b': 131072,
            'deepseek/deepseek-r1-distill-llama-70b:free': 131072,
            'deepseek/deepseek-r1-distill-llama-8b': 131072,
            'deepseek/deepseek-r1-distill-qwen-1.5b': 32768,
            'deepseek/deepseek-r1-distill-qwen-14b': 32768,
            'deepseek/deepseek-r1-distill-qwen-14b:free': 32768,
            'deepseek/deepseek-r1-distill-qwen-32b': 32768,
            'deepseek/deepseek-r1-distill-qwen-7b': 32768,
            'deepseek/deepseek-r1:free': 32768,
            'deepseek/deepseek-v3-base': 32768,
            'eleutherai/llemma_7b': 8192,
            'featherless/qwerky-72b:free': 8192,
            'google/gemini-2.0-flash-001': 1000000,
            'google/gemini-2.0-flash-exp:free': 1000000,
            'google/gemini-2.0-flash-lite-001': 1000000,
            'google/gemini-2.5-flash': 1000000,
            'google/gemini-2.5-flash-lite': 1000000,
            'google/gemini-2.5-flash-lite-preview-06-17': 1000000,
            'google/gemini-2.5-pro': 2000000,
            'google/gemini-2.5-pro-exp-03-25': 2000000,
            'google/gemini-2.5-pro-preview': 2000000,
            'google/gemini-2.5-pro-preview-05-06': 2000000,
            'google/gemini-flash-1.5': 1000000,
            'google/gemini-flash-1.5-8b': 1000000,
            'google/gemini-pro-1.5': 2000000,
            'google/gemma-2-27b-it': 8192,
            'google/gemma-2-9b-it': 8192,
            'google/gemma-2-9b-it:free': 8192,
            'google/gemma-3-12b-it': 131072,
            'google/gemma-3-12b-it:free': 131072,
            'google/gemma-3-27b-it': 131072,
            'google/gemma-3-27b-it:free': 131072,
            'google/gemma-3-4b-it': 131072,
            'google/gemma-3-4b-it:free': 131072,
            'google/gemma-3n-e2b-it:free': 131072,
            'google/gemma-3n-e4b-it': 131072,
            'google/gemma-3n-e4b-it:free': 131072,
            'gryphe/mythomax-l2-13b': 4096,
            'inception/mercury': 32768,
            'inception/mercury-coder': 32768,
            'infermatic/mn-inferor-12b': 8192,
            'inflection/inflection-3-pi': 128000,
            'inflection/inflection-3-productivity': 128000,
            'liquid/lfm-3b': 8192,
            'liquid/lfm-40b': 8192,
            'liquid/lfm-7b': 8192,
            'mancer/weaver': 8192,
            'meta-llama/llama-3-70b-instruct': 8192,
            'meta-llama/llama-3-8b-instruct': 8192,
            'meta-llama/llama-3.1-405b': 131072,
            'meta-llama/llama-3.1-405b-instruct': 131072,
            'meta-llama/llama-3.1-405b-instruct:free': 131072,
            'meta-llama/llama-3.1-70b-instruct': 131072,
            'meta-llama/llama-3.1-8b-instruct': 131072,
            'meta-llama/llama-3.2-11b-vision-instruct': 131072,
            'meta-llama/llama-3.2-11b-vision-instruct:free': 131072,
            'meta-llama/llama-3.2-1b-instruct': 131072,
            'meta-llama/llama-3.2-3b-instruct': 131072,
            'meta-llama/llama-3.2-3b-instruct:free': 131072,
            'meta-llama/llama-3.2-90b-vision-instruct': 131072,
            'meta-llama/llama-3.3-70b-instruct': 131072,
            'meta-llama/llama-3.3-70b-instruct:free': 131072,
            'meta-llama/llama-4-maverick': 131072,
            'meta-llama/llama-4-scout': 131072,
            'meta-llama/llama-guard-2-8b': 8192,
            'meta-llama/llama-guard-3-8b': 131072,
            'meta-llama/llama-guard-4-12b': 131072,
            'microsoft/mai-ds-r1': 32768,
            'microsoft/mai-ds-r1:free': 32768,
            'microsoft/phi-3-medium-128k-instruct': 131072,
            'microsoft/phi-3-mini-128k-instruct': 131072,
            'microsoft/phi-3.5-mini-128k-instruct': 131072,
            'microsoft/phi-4': 131072,
            'microsoft/phi-4-multimodal-instruct': 131072,
            'microsoft/phi-4-reasoning-plus': 131072,
            'microsoft/wizardlm-2-8x22b': 65536,
            'minimax/minimax-01': 200000,
            'minimax/minimax-m1': 200000,
            'mistralai/codestral-2501': 32768,
            'mistralai/codestral-2508': 32768,
            'mistralai/devstral-medium': 32768,
            'mistralai/devstral-small': 32768,
            'mistralai/devstral-small-2505': 32768,
            'mistralai/devstral-small-2505:free': 32768,
            'mistralai/magistral-medium-2506': 32768,
            'mistralai/magistral-medium-2506:thinking': 32768,
            'mistralai/magistral-small-2506': 32768,
            'mistralai/ministral-3b': 32768,
            'mistralai/ministral-8b': 32768,
            'mistralai/mistral-7b-instruct': 32768,
            'mistralai/mistral-7b-instruct-v0.1': 8192,
            'mistralai/mistral-7b-instruct-v0.2': 32768,
            'mistralai/mistral-7b-instruct-v0.3': 32768,
            'mistralai/mistral-7b-instruct:free': 32768,
            'mistralai/mistral-large': 32768,
            'mistralai/mistral-large-2407': 128000,
            'mistralai/mistral-large-2411': 128000,
            'mistralai/mistral-medium-3': 32768,
            'mistralai/mistral-nemo': 128000,
            'mistralai/mistral-nemo:free': 128000,
            'mistralai/mistral-saba': 32768,
            'mistralai/mistral-small': 32768,
            'mistralai/mistral-small-24b-instruct-2501': 32768,
            'mistralai/mistral-small-24b-instruct-2501:free': 32768,
            'mistralai/mistral-small-3.1-24b-instruct': 32768,
            'mistralai/mistral-small-3.1-24b-instruct:free': 32768,
            'mistralai/mistral-small-3.2-24b-instruct': 32768,
            'mistralai/mistral-small-3.2-24b-instruct:free': 32768,
            'mistralai/mistral-tiny': 32768,
            'mistralai/mixtral-8x22b-instruct': 65536,
            'mistralai/mixtral-8x7b-instruct': 32768,
            'mistralai/pixtral-12b': 128000,
            'mistralai/pixtral-large-2411': 128000,
            'moonshotai/kimi-dev-72b:free': 200000,
            'moonshotai/kimi-k2': 200000,
            'moonshotai/kimi-k2:free': 200000,
            'moonshotai/kimi-vl-a3b-thinking': 200000,
            'moonshotai/kimi-vl-a3b-thinking:free': 200000,
            'morph/morph-v3-fast': 8192,
            'morph/morph-v3-large': 8192,
            'neversleep/llama-3-lumimaid-70b': 8192,
            'neversleep/llama-3.1-lumimaid-8b': 131072,
            'neversleep/noromaid-20b': 32768,
            'nousresearch/deephermes-3-llama-3-8b-preview:free': 8192,
            'nousresearch/deephermes-3-mistral-24b-preview': 32768,
            'nousresearch/hermes-2-pro-llama-3-8b': 8192,
            'nousresearch/hermes-3-llama-3.1-405b': 131072,
            'nousresearch/hermes-3-llama-3.1-70b': 131072,
            'nousresearch/nous-hermes-2-mixtral-8x7b-dpo': 32768,
            'nvidia/llama-3.1-nemotron-70b-instruct': 131072,
            'nvidia/llama-3.1-nemotron-ultra-253b-v1': 131072,
            'nvidia/llama-3.1-nemotron-ultra-253b-v1:free': 131072,
            'nvidia/llama-3.3-nemotron-super-49b-v1': 131072,
            'openai/chatgpt-4o-latest': 128000,
            'openai/codex-mini': 2048,
            'openai/gpt-3.5-turbo': 4096,
            'openai/gpt-3.5-turbo-0613': 4096,
            'openai/gpt-3.5-turbo-16k': 16384,
            'openai/gpt-3.5-turbo-instruct': 4096,
            'openai/gpt-4': 8192,
            'openai/gpt-4-0314': 8192,
            'openai/gpt-4-1106-preview': 128000,
            'openai/gpt-4-turbo': 128000,
            'openai/gpt-4-turbo-preview': 128000,
            'openai/gpt-4.1': 128000,
            'openai/gpt-4.1-mini': 128000,
            'openai/gpt-4.1-nano': 128000,
            'openai/gpt-4o': 128000,
            'openai/gpt-4o-2024-05-13': 128000,
            'openai/gpt-4o-2024-08-06': 128000,
            'openai/gpt-4o-2024-11-20': 128000,
            'openai/gpt-4o-mini': 128000,
            'openai/gpt-4o-mini-2024-07-18': 128000,
            'openai/gpt-4o-mini-search-preview': 128000,
            'openai/gpt-4o-search-preview': 128000,
            'openai/gpt-4o:extended': 128000,
            'openai/gpt-5': 200000,
            'openai/gpt-5-chat': 200000,
            'openai/gpt-5-mini': 200000,
            'openai/gpt-5-nano': 200000,
            'openai/gpt-oss-120b': 128000,
            'openai/gpt-oss-20b': 128000,
            'openai/gpt-oss-20b:free': 128000,
            'openai/o1': 128000,
            'openai/o1-mini': 128000,
            'openai/o1-mini-2024-09-12': 128000,
            'openai/o1-pro': 128000,
            'openai/o3': 200000,
            'openai/o3-mini': 200000,
            'openai/o3-mini-high': 200000,
            'openai/o3-pro': 200000,
            'openai/o4-mini': 128000,
            'openai/o4-mini-high': 128000,
            'opengvlab/internvl3-14b': 8192,
            'openrouter/auto': 8192,
            'perplexity/r1-1776': 32768,
            'perplexity/sonar': 32768,
            'perplexity/sonar-deep-research': 32768,
            'perplexity/sonar-pro': 32768,
            'perplexity/sonar-reasoning': 32768,
            'perplexity/sonar-reasoning-pro': 32768,
            'pygmalionai/mythalion-13b': 4096,
            'qwen/qwen-2-72b-instruct': 32768,
            'qwen/qwen-2.5-72b-instruct': 131072,
            'qwen/qwen-2.5-72b-instruct:free': 131072,
            'qwen/qwen-2.5-7b-instruct': 131072,
            'qwen/qwen-2.5-coder-32b-instruct': 131072,
            'qwen/qwen-2.5-coder-32b-instruct:free': 131072,
            'qwen/qwen-2.5-vl-7b-instruct': 131072,
            'qwen/qwen-max': 32768,
            'qwen/qwen-plus': 32768,
            'qwen/qwen-turbo': 8192,
            'qwen/qwen-vl-max': 32768,
            'qwen/qwen-vl-plus': 32768,
            'qwen/qwen2.5-vl-32b-instruct': 131072,
            'qwen/qwen2.5-vl-32b-instruct:free': 131072,
            'qwen/qwen2.5-vl-72b-instruct': 131072,
            'qwen/qwen2.5-vl-72b-instruct:free': 131072,
            'qwen/qwen3-14b': 32768,
            'qwen/qwen3-14b:free': 32768,
            'qwen/qwen3-235b-a22b': 32768,
            'qwen/qwen3-235b-a22b-2507': 32768,
            'qwen/qwen3-235b-a22b-thinking-2507': 32768,
            'qwen/qwen3-235b-a22b:free': 32768,
            'qwen/qwen3-30b-a3b': 32768,
            'qwen/qwen3-30b-a3b-instruct-2507': 32768,
            'qwen/qwen3-30b-a3b:free': 32768,
            'qwen/qwen3-32b': 32768,
            'qwen/qwen3-4b:free': 32768,
            'qwen/qwen3-8b': 32768,
            'qwen/qwen3-8b:free': 32768,
            'qwen/qwen3-coder': 32768,
            'qwen/qwen3-coder:free': 32768,
            'qwen/qwq-32b': 32768,
            'qwen/qwq-32b-preview': 32768,
            'qwen/qwq-32b:free': 32768,
            'raifle/sorcererlm-8x22b': 65536,
            'rekaai/reka-flash-3:free': 128000,
            'sao10k/l3-euryale-70b': 8192,
            'sao10k/l3-lunaris-8b': 8192,
            'sao10k/l3.1-euryale-70b': 131072,
            'sao10k/l3.3-euryale-70b': 131072,
            'sarvamai/sarvam-m:free': 8192,
            'scb10x/llama3.1-typhoon2-70b-instruct': 131072,
            'shisa-ai/shisa-v2-llama3.3-70b': 131072,
            'shisa-ai/shisa-v2-llama3.3-70b:free': 131072,
            'sophosympatheia/midnight-rose-70b': 4096,
            'switchpoint/router': 8192,
            'tencent/hunyuan-a13b-instruct': 8192,
            'tencent/hunyuan-a13b-instruct:free': 8192,
            'thedrummer/anubis-70b-v1.1': 8192,
            'thedrummer/anubis-pro-105b-v1': 8192,
            'thedrummer/rocinante-12b': 8192,
            'thedrummer/skyfall-36b-v2': 8192,
            'thedrummer/unslopnemo-12b': 128000,
            'thedrummer/valkyrie-49b-v1': 8192,
            'thudm/glm-4-32b': 2000000,
            'thudm/glm-4.1v-9b-thinking': 2000000,
            'thudm/glm-z1-32b:free': 2000000,
            'tngtech/deepseek-r1t-chimera': 32768,
            'tngtech/deepseek-r1t-chimera:free': 32768,
            'tngtech/deepseek-r1t2-chimera:free': 32768,
            'undi95/remm-slerp-l2-13b': 4096,
            'x-ai/grok-2-1212': 128000,
            'x-ai/grok-2-vision-1212': 128000,
            'x-ai/grok-3': 128000,
            'x-ai/grok-3-beta': 128000,
            'x-ai/grok-3-mini': 128000,
            'x-ai/grok-3-mini-beta': 128000,
            'x-ai/grok-4': 128000,
            'x-ai/grok-vision-beta': 128000,
            'z-ai/glm-4-32b': 2000000,
            'z-ai/glm-4.5': 2000000,
            'z-ai/glm-4.5-air': 2000000,
            'z-ai/glm-4.5-air:free': 2000000,
            'llama3.1': 131072,   # Llama 3.1 extended context
            'llama3.2': 131072,   # Llama 3.2 extended context
            'llama3.3': 131072,   # Assuming similar to 3.1/3.2
            'llama3': 8192,       # Llama 3 default
            'llama2': 4096,       # Llama 2 default
            'mixtral8x22b': 65536, # Mixtral 8x22B default
            'mixtral': 32768,     # Mixtral 8x7B default
            'mistral': 32768,     # Mistral 7B v0.2+ default
            'gemma3': 131072,     # Gemma 3 with 128K context
            'gemma2': 8192,       # Gemma 2 default
            'gemma': 8192,        # Gemma default
            'phi3': 131072,       # Phi-3 variants often use 128K (mini/medium extended)
            'phi2': 2048,         # Phi-2 default
            'phi': 2048,          # Phi default (older)
            'qwen2.5': 131072,    # Qwen2.5 with 128K
            'qwen2': 32768,       # Qwen2 default for 7B
            'qwen': 8192,         # Qwen default
            'codellama': 16384,   # CodeLlama extended
            'codegemma': 8192,    # CodeGemma default
            'deepseek-coder-v2': 131072,  # DeepSeek-Coder V2 with 128K
            'deepseek-coder': 16384,  # DeepSeek-Coder V1 default
            'deepseek-v2': 131072,    # DeepSeek-V2 with 128K
            'deepseek-llm': 4096,     # DeepSeek-LLM default
            'yi1.5': 32768,       # Yi-1.5 with 32K
            'yi': 4096,           # Yi base default
            'command-r': 131072,  # Command-R with 128K
            'wizardlm2': 32768,   # WizardLM2 (Mistral-based)
            'wizardlm': 16384,    # WizardLM default
            'zephyr': 65536,      # Zephyr beta (Mistral-based extended)
            'vicuna': 2048,       # Vicuna default (up to 16K in some variants)
            'falcon': 2048,       # Falcon default
            'starcoder': 8192,    # StarCoder default
            'stablelm': 4096,     # StableLM default
            'orca2': 4096,        # Orca 2 default
            'orca': 4096,         # Orca default
            'dolphin': 32768,     # Dolphin (often Mistral-based)
            'openhermes': 8192,   # OpenHermes default
            'gpt-oss': 128000,  # GPT-OSS with 128K context
            'gpt-3.5-turbo': 4096, # GPT-3.5 Turbo default
            'gpt-4': 8192,        # GPT-4 default
            'grok-2': 128000,
            'grok-2-1212': 128000,
            'grok-2-vision-1212': 128000,
            'grok-3': 128000,
            'grok-3-fast': 128000,
            'grok-3-beta': 128000,
            'grok-3-mini': 128000,
            'grok-3-mini-beta': 128000,
            'grok-3-mini-fast': 128000,
            'grok-4-0709': 128000,
            'grok-4': 128000,
            'grok-vision-beta': 128000,
        }

        normalized_model_name = model_name.lower().strip()

        # Sort keys by length in descending order. This ensures that a more specific
        # name like 'llama3.1' is checked before a less specific name like 'llama3'.
        sorted_base_models = sorted(known_contexts.keys(), key=len, reverse=True)

        for base_name in sorted_base_models:
            if base_name in normalized_model_name:
                context_size = known_contexts[base_name]
                ASCIIColors.warning(
                    f"Using hardcoded context size for model '{model_name}' "
                    f"based on base name '{base_name}': {context_size}"
                )
                return context_size

        ASCIIColors.warning(f"Context size not found for model '{model_name}' in the hardcoded list.")
        return None


    @abstractmethod
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """        
        pass

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.

        Args:
            image (str): Image to count tokens from. Either base64 string, path to image file, or URL.

        Returns:
            int: Estimated number of tokens for the image. Returns -1 on error.
        """
        try:
            # Delegate token counting to ImageTokenizer
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e}")
            return -1
    @abstractmethod
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        
        Args:
            text (str or List[str]): Input text to embed
            **kwargs: Additional arguments like model, truncate, options, keep_alive
        
        Returns:
            dict: Response containing embeddings
        """
        pass    
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Return information about the current model.

        Returns:
            dict: Model information dictionary.
        """
        pass

    @abstractmethod
    def listModels(self) -> list:
        """Lists models"""
        pass
    
    
    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        pass


    def split_discussion(self, lollms_prompt_string: str, system_keyword="!@>system:", user_keyword="!@>user:", ai_keyword="!@>assistant:") -> list:
        """
        Splits a LoLLMs prompt into a list of OpenAI-style messages.
        If the very first chunk has no prefix, it's assigned to "system".
        """
        # Regex to split on any of the three prefixes (lookahead)
        pattern = r"(?={}|{}|{})".format(
            re.escape(system_keyword),
            re.escape(user_keyword),
            re.escape(ai_keyword)
        )
        parts = re.split(pattern, lollms_prompt_string)
        messages = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Determine role and strip prefix if present
            if part.startswith(system_keyword):
                role = "system"
                content = part[len(system_keyword):].strip()
            elif part.startswith(user_keyword):
                role = "user"
                content = part[len(user_keyword):].strip()
            elif part.startswith(ai_keyword):
                role = "assistant"
                content = part[len(ai_keyword):].strip()
            else:
                # No prefix: if it's the first valid chunk, treat as system
                if not messages:
                    role = "system"
                    content = part
                else:
                    # otherwise skip unrecognized segments
                    continue

            messages.append({"role": role, "content": content})
            if messages[-1]["content"]=="":
                del messages[-1]
        return messages




class LollmsLLMBindingManager:
    """Manages binding discovery and instantiation"""

    def __init__(self, llm_bindings_dir: str = "llm_bindings"):
        """
        Initialize the LollmsLLMBindingManager.

        Args:
            llm_bindings_dir (str): Directory containing binding implementations. Defaults to "llm_bindings".
        """
        self.llm_bindings_dir = Path(llm_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific binding implementation from the llm bindings directory."""
        binding_dir = self.llm_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.llm_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load binding {binding_name}: {str(e)}")

    def create_binding(self, 
                      binding_name: str,
                      **kwargs) -> Optional[LollmsLLMBinding]:
        """
        Create an instance of a specific binding.

        Args:
            binding_name (str): Name of the binding to create.
            kwargs: binding specific arguments

        Returns:
            Optional[LollmsLLMBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available binding names.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.llm_bindings_dir.iterdir() if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]

def get_available_bindings():
    bindings_dir = Path(__file__).parent/"llm_bindings"
    return [binding_dir.name for binding_dir in bindings_dir.iterdir() if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]
