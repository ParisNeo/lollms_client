## DOCS FOR: `lollms_client/lollms_core.py` (LollmsClient Class)

### `LollmsClient`

**Purpose:**
The `LollmsClient` class is the primary interface for interacting with various Large Language Model (LLM) backends and modality services. It orchestrates text generation, multimodal operations, and function calling through a unified API, managing different "bindings" for specific backends or services.

**Key Attributes/Properties:**

*   **`binding` (`LollmsLLMBinding`)**: The currently active LLM binding instance responsible for text generation and LLM-specific operations.
*   **`tts` (`Optional[LollmsTTSBinding]`)**: The active Text-to-Speech binding instance, if configured.
*   **`tti` (`Optional[LollmsTTIBinding]`)**: The active Text-to-Image binding instance, if configured.
*   **`stt` (`Optional[LollmsSTTBinding]`)**: The active Speech-to-Text binding instance, if configured.
*   **`ttv` (`Optional[LollmsTTVBinding]`)**: The active Text-to-Video binding instance, if configured.
*   **`ttm` (`Optional[LollmsTTMBinding]`)**: The active Text-to-Music binding instance, if configured.
*   **`mcp` (`Optional[LollmsMCPBinding]`)**: The active Model Context Protocol (MCP) binding instance for function calling, if configured.
*   **`binding_manager` (`LollmsLLMBindingManager`)**: Manages the discovery and creation of LLM bindings.
*   **`tts_binding_manager`, `tti_binding_manager`, etc.**: Managers for respective modality bindings.
*   **`default_ctx_size`, `default_n_predict`, etc.**: Default parameters for LLM text generation.
*   **`user_name`, `ai_name`**: Default role names for formatting prompts.
*   **Prompt Formatting Attributes**: (`system_full_header`, `user_full_header`, `ai_full_header`, etc.) Properties for constructing standardized prompt headers.

**Methods:**

*   **`__init__(...)`**:
    *   **Purpose**: Initializes the `LollmsClient`, sets up the primary LLM binding, and optionally configures modality and MCP bindings.
    *   **Parameters**:
        *   `binding_name` (str): Name of the LLM binding (e.g., "lollms", "ollama").
        *   `host_address` (Optional[str]): Default host for service-based bindings.
        *   `models_path` (Optional[str]): Default path for local model files.
        *   `model_name` (str): Default LLM model name.
        *   `llm_bindings_dir` (Path): Directory for LLM binding implementations.
        *   `llm_binding_config` (Optional[Dict]): Configuration for the LLM binding.
        *   `tts_binding_name`, `tti_binding_name`, etc. (Optional[str]): Names for modality/MCP bindings.
        *   `tts_bindings_dir`, `tti_bindings_dir`, etc. (Path): Directories for modality/MCP bindings.
        *   `tts_binding_config`, `tti_binding_config`, etc. (Optional[Dict]): Configurations for modality/MCP bindings.
        *   `service_key` (Optional[str]): Shared authentication key or client ID.
        *   `verify_ssl_certificate` (bool): SSL verification flag.
        *   `ctx_size`, `n_predict`, `stream`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads` (Optional): Default LLM generation parameters.
        *   `streaming_callback` (Optional[Callable]): Default callback for streaming LLM output.
        *   `user_name`, `ai_name` (str): Default role names.
    *   **Returns**: `LollmsClient` instance.

*   **`tokenize(text: str) -> list`**:
    *   **Purpose**: Tokenizes text using the active LLM binding.
    *   **Parameters**: `text` (str).
    *   **Returns**: List of tokens.

*   **`detokenize(tokens: list) -> str`**:
    *   **Purpose**: Converts tokens back to text using the active LLM binding.
    *   **Parameters**: `tokens` (list).
    *   **Returns**: Detokenized string.

*   **`count_tokens(text: str) -> int`**:
    *   **Purpose**: Counts tokens in a given text using the active LLM binding.
    *   **Parameters**: `text` (str).
    *   **Returns**: Number of tokens.

*   **`get_model_details() -> dict`**:
    *   **Purpose**: Retrieves information about the currently loaded LLM model from the active binding.
    *   **Returns**: Dictionary containing model details.

*   **`switch_model(model_name: str) -> bool`**:
    *   **Purpose**: Loads a new model in the active LLM binding.
    *   **Parameters**: `model_name` (str).
    *   **Returns**: `True` if successful, `False` otherwise.

*   **`get_available_llm_bindings() -> List[str]`**:
    *   **Purpose**: Lists the names of available LLM bindings.
    *   **Returns**: List of strings.

*   **`generate_text(...) -> Union[str, dict]`**:
    *   **Purpose**: Generates text using the active LLM binding. Can handle text-only or multimodal (text + images) input.
    *   **Parameters**:
        *   `prompt` (str): The main text prompt.
        *   `images` (Optional[List[str]]): List of image file paths for multimodal input.
        *   `system_prompt` (str): An optional system-level instruction.
        *   `n_predict`, `stream`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads`, `ctx_size` (Optional): Generation parameters, overriding client defaults if provided.
        *   `streaming_callback` (Optional[Callable]): Callback for handling streamed output.
        *   `split` (Optional[bool]): If `True`, treats the prompt as a structured discussion for splitting.
        *   `user_keyword`, `ai_keyword` (Optional[str]): Keywords for splitting discussion prompts.
    *   **Returns**: Generated text as a string, or a dictionary with an "error" key if generation failed.

*   **`embed(text, **kwargs) -> list`**:
    *   **Purpose**: Generates embeddings for the input text using the active LLM binding.
    *   **Parameters**: `text` (str or List[str]), `**kwargs` for binding-specific embedding options.
    *   **Returns**: List of floats (embedding vector) or list of lists of floats (for batch input).

*   **`listModels() -> list`**: (Note: Original `lollms_core_classes.md` shows `listModels(self, host_address: str = None) -> None`. This should probably return a list and delegate to the binding.)
    *   **Purpose**: Lists models available to the current LLM binding.
    *   **Returns**: List of model information dictionaries.

*   **`listMountedPersonalities() -> Union[List[Dict], Dict]`**:
    *   **Purpose**: Lists mounted personalities if the active LLM binding is "lollms".
    *   **Returns**: List of personality dictionaries or an error dictionary.

*   **`generate_with_mcp(...) -> Dict[str, Any]`**:
    *   **Purpose**: Generates a response that may involve calling one or more tools via the configured MCP binding.
    *   **Parameters**:
        *   `prompt` (str): User's initial prompt.
        *   `discussion_history` (Optional[List[Dict]]): Previous conversation turns.
        *   `images` (Optional[List[str]]): Images for the current prompt.
        *   `tools` (Optional[List[Dict]]): Specific tools to use; if `None`, discovers from MCP binding.
        *   `max_tool_calls` (int): Max distinct tool calls per turn.
        *   `max_llm_iterations` (int): Max LLM decisions to call tools before forcing a final answer.
        *   `tool_call_decision_temperature` (float): Temperature for LLM tool decision.
        *   `final_answer_temperature` (float): Temperature for LLM final answer generation.
        *   `streaming_callback` (Optional[Callable]): Callback for streaming LLM thoughts, tool calls, and final answer. Signature: `(chunk_str, msg_type, metadata_dict, history_list_for_turn) -> bool`.
        *   `interactive_tool_execution` (bool): If `True`, prompt user before executing a tool.
        *   `**llm_generation_kwargs`: Additional keyword arguments for the underlying LLM generation calls.
    *   **Returns**: Dictionary with `"final_answer"`, `"tool_calls"`, and optionally `"error"`.

*   **`generate_text_with_rag(...) -> Dict[str, Any]`**:
    *   **Purpose**: Generates text using Retrieval Augmented Generation (RAG) with multiple hops and optional objective extraction.
    *   **Parameters**:
        *   `prompt` (str): The user's query.
        *   `rag_query_function` (Callable): Function to retrieve relevant chunks. Signature: `(query_text, vectorizer_name, top_k, min_similarity_percent) -> List[Dict]`.
        *   `rag_query_text` (Optional[str]): Initial query for RAG; if `None`, derived from `prompt`.
        *   `rag_vectorizer_name` (Optional[str]): Name of vectorizer for `rag_query_function`.
        *   `rag_top_k` (int): Number of chunks to retrieve per RAG hop.
        *   `rag_min_similarity_percent` (float): Minimum similarity for retrieved chunks.
        *   `max_rag_hops` (int): Maximum number of RAG refinement hops.
        *   `images`, `system_prompt`, `n_predict`, etc.: Standard LLM generation parameters.
        *   `extract_objectives` (bool): If `True`, LLM first extracts objectives from the prompt.
        *   `streaming_callback` (Optional[Callable]): Callback for streaming RAG steps and final answer.
        *   `max_rag_context_characters` (int): Max characters for accumulated RAG context before summarization.
        *   `**llm_generation_kwargs`: Additional keyword arguments for LLM calls.
    *   **Returns**: Dictionary with `"final_answer"`, `"rag_hops_history"`, `"all_retrieved_sources"`, and optionally `"error"`.

*   **`generate_codes(...) -> List[dict]`**:
    *   **Purpose**: Generates multiple code blocks based on a prompt, optionally using a template and specifying language.
    *   **Parameters**: Similar to `generate_text`, plus `template` (str), `language` (str), `code_tag_format` (str).
    *   **Returns**: List of extracted code block dictionaries.

*   **`generate_code(...) -> Optional[str]`**:
    *   **Purpose**: Generates a single code block, handling potential continuations.
    *   **Parameters**: Similar to `generate_codes`.
    *   **Returns**: The content of the generated code block as a string, or `None` if failed.

*   **`extract_code_blocks(text: str, format: str = "markdown") -> List[dict]`**:
    *   **Purpose**: Extracts code blocks from text (Markdown or HTML).
    *   **Parameters**: `text` (str), `format` (str: "markdown" or "html").
    *   **Returns**: List of dictionaries, each representing a code block with "index", "file_name", "content", "type", "is_complete".

*   **`extract_thinking_blocks(text: str) -> List[str]`**:
    *   **Purpose**: Extracts content from `<thinking>` or `<think>` tags.
    *   **Parameters**: `text` (str).
    *   **Returns**: List of strings, each being the content of a thinking block.

*   **`remove_thinking_blocks(text: str) -> str`**:
    *   **Purpose**: Removes `<thinking>` or `<think>` blocks (including tags) from text.
    *   **Parameters**: `text` (str).
    *   **Returns**: Cleaned string.

*   **`yes_no(...) -> Union[bool, dict]`**:
    *   **Purpose**: Answers a yes/no question based on context, using LLM JSON generation.
    *   **Parameters**: `question` (str), `context` (str), `max_answer_length` (int), `conditionning` (str), `return_explanation` (bool), `callback` (Optional[Callable]).
    *   **Returns**: Boolean answer, or a dictionary with "answer" and "explanation" if `return_explanation` is `True`.

*   **`multichoice_question(...) -> Union[int, dict]`**:
    *   **Purpose**: Interprets a multiple-choice question using LLM JSON generation.
    *   **Parameters**: `question` (str), `possible_answers` (list), `context` (str), etc. (similar to `yes_no`).
    *   **Returns**: Integer index of the selected answer, or a dictionary with "index" and "explanation".

*   **`multichoice_ranking(...) -> dict`**:
    *   **Purpose**: Ranks answers for a question from best to worst using LLM JSON generation.
    *   **Parameters**: `question` (str), `possible_answers` (list), `context` (str), etc.
    *   **Returns**: Dictionary with "ranking" (list of indices) and optionally "explanations".

*   **`sequential_summarize(...) -> str`**:
    *   **Purpose**: Processes text in chunks sequentially, updating a memory at each step to produce a final summary.
    *   **Parameters**:
        *   `text` (str): The text to summarize.
        *   `chunk_processing_prompt` (str): Prompt for processing each chunk and updating memory.
        *   `chunk_processing_output_format` (str): Expected format of the memory (e.g., "markdown").
        *   `final_memory_processing_prompt` (str): Prompt for generating the final summary from the memory.
        *   `final_output_format` (str): Format for the final summary.
        *   `ctx_size`, `chunk_size`, `overlap`, `bootstrap_chunk_size`, `bootstrap_steps` (Optional[int]): Parameters for chunking.
        *   `callback` (Optional[Callable]): Callback for progress updates.
        *   `debug` (bool): If `True`, prints intermediate steps.
    *   **Returns**: The final summary string.

*   **`deep_analyze(...) -> str`**:
    *   **Purpose**: Searches for information related to a query in long text or multiple files, processing chunk by chunk.
    *   **Parameters**:
        *   `query` (str): The query to search for.
        *   `text` (Optional[str]): Text to analyze (if not using files).
        *   `files` (Optional[List[Union[str, Path]]]): List of file paths to analyze.
        *   `aggregation_prompt` (str): Prompt for aggregating findings.
        *   `output_format` (str): Desired output format.
        *   `ctx_size`, `chunk_size`, `overlap`, `bootstrap_chunk_size`, `bootstrap_steps` (Optional[int]): Chunking parameters.
        *   `callback` (Optional[Callable]): Callback for progress.
        *   `debug` (bool): If `True`, prints intermediate steps.
    *   **Returns**: The aggregated answer string.

**Usage Examples:**

```python
# Initialize LollmsClient (example with Ollama)
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

def my_callback(chunk, msg_type, params=None, metadata=None):
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    # Handle other msg_types if needed
    return True

try:
    lc = LollmsClient(
        binding_name="ollama",
        model_name="mistral:latest", # Ensure this model is pulled in Ollama
        streaming_callback=my_callback # Set a default callback
    )

    # Basic text generation
    print("\n--- Basic Text Generation ---")
    response = lc.generate_text("Explain quantum computing in one sentence.", stream=True)
    print("\n")

    # Text generation with image (requires a vision-capable model in Ollama, e.g., llava)
    # Assuming 'ollama_vision_client' is configured with a model like 'llava:latest'
    # and 'example.jpg' exists.
    # print("\n--- Text Generation with Image (using llava) ---")
    # try:
    #     lc_vision = LollmsClient(binding_name="ollama", model_name="llava:latest", streaming_callback=my_callback)
    #     # Create a dummy image file for the example if it doesn't exist
    #     from PIL import Image
    #     if not Path("example.jpg").exists():
    #         img = Image.new('RGB', (60, 30), color = 'red')
    #         img.save("example.jpg")
    #     vision_response = lc_vision.generate_text("Describe this image:", images=["example.jpg"], stream=True)
    #     print("\n")
    # except Exception as e_vision:
    #     ASCIIColors.warning(f"Skipping vision test: {e_vision}")


    # Function Calling with MCP (using local_mcp and default tools)
    print("\n--- Function Calling with MCP (Internet Search) ---")
    lc_with_mcp = LollmsClient(
        binding_name="ollama", model_name="mistral:latest", # LLM for decisions
        mcp_binding_name="local_mcp", # Enable local tools
        streaming_callback=my_callback # Callback for MCP steps and final answer
    )
    mcp_response = lc_with_mcp.generate_with_mcp(
        prompt="What's the current weather in Paris?"
    )
    print(f"\nFinal Answer (Weather): {mcp_response.get('final_answer')}")
    print(f"Tool Calls: {mcp_response.get('tool_calls')}")


    # Sequential Summarization
    print("\n--- Sequential Summarization ---")
    long_text = "This is the first part of a very long document. It discusses apples. " * 50
    long_text += "The second part is about bananas. It explains their nutritional value. " * 50
    long_text += "Finally, the document concludes with oranges and their citrusy benefits. " * 50

    # A simple extraction prompt for summarization
    extraction_template = """Important topics found:
- {{topics_list}}

Key entities mentioned:
- {{entities_list}}

Overall sentiment: {{sentiment}}
"""
    summary = lc.sequential_summarize(
        text=long_text,
        chunk_processing_prompt=f"Extract topics, entities, and sentiment from the chunk and update the memory based on this template:\n{extraction_template}",
        chunk_processing_output_format="text", # Or "json" if template is JSON
        final_memory_processing_prompt="Generate a concise final summary from the accumulated memory.",
        final_output_format="markdown",
        ctx_size=lc.default_ctx_size, # Use client's default context size
        chunk_size=512, # Smaller chunks for faster processing in example
        debug=False # Set to True to see intermediate steps
    )
    print(f"\nSequential Summary:\n{summary}")


except Exception as e:
    ASCIIColors.error(f"An error occurred: {e}")
    from ascii_colors import trace_exception
    trace_exception(e)
```

**Dependencies:**
*   `requests`
*   `ascii_colors`
*   `lollms_client.lollms_types`
*   `lollms_client.lollms_utilities`
*   `lollms_client.lollms_llm_binding` (and its managers for TTS, TTI, etc.)
*   `lollms_client.lollms_mcp_binding`
*   `json`, `re`, `enum`, `base64`, `numpy`, `pathlib`, `os` (standard libraries)

**Configuration Options:**
The `LollmsClient` is primarily configured during instantiation via its `__init__` parameters. Key configuration areas include:
*   Selection and configuration of the primary LLM binding.
*   Selection and configuration of optional modality (TTS, TTI, etc.) and MCP bindings.
*   Default generation parameters for the LLM.
*   Host addresses and model paths for bindings.
