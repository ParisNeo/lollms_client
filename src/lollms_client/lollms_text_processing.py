import json
import re
from typing import Any, Callable, Optional, Union, List
from ascii_colors import ASCIIColors, trace_exception

class LollmsTextProcessor:
    """
    A comprehensive text and code processing layer that sits on top of any LLM.
    
    This class handles:
    - Long context processing with chunking and summarization
    - Code generation (single and multi-file) with XML tags
    - Structured JSON generation with schema validation
    - Automatic handling of truncated responses
    - Efficient code editing with structured operations
    - Pydantic model support
    - Helper functions (yes/no, multiple choice, ranking, etc.)
    - Text parsing utilities
    
    Usage:
        llm = YourLLMClass()
        processor = LollmsTextProcessor(llm)
        result = processor.generate_structured_content(prompt, schema)
    """
    
    def __init__(self, llm):
        """
        Initialize the text processor.
        
        Args:
            llm: An LLM instance that has generate_text(), get_context_size(), and count_tokens() methods
            ascii_colors: Optional ASCIIColors instance for logging
        """
        self.llm = llm
        
        # Verify the LLM has the required methods
        if not hasattr(llm, 'generate_text'):
            raise AttributeError(
                "LLM must have a 'generate_text' method. "
                "Expected signature: generate_text(prompt, images=[], system_prompt=None, ...)"
            )
        
        if not hasattr(llm, 'count_tokens'):
            self._log_warning("LLM doesn't have 'count_tokens' method. Long context processing will use character-based estimation.")
        
        if not hasattr(llm, 'get_context_size'):
            self._log_warning("LLM doesn't have 'get_context_size' method. Will default to 8192 tokens.")
    
    def _log_info(self, message: str):
        """Log info message."""
        ASCIIColors.info(f"[INFO] {message}")
    
    def _log_warning(self, message: str):
        """Log warning message."""
        ASCIIColors.warning(message)
    
    def _log_error(self, message: str):
        """Log error message."""
        ASCIIColors.error(message)
    
    def _log_success(self, message: str):
        """Log success message."""
        ASCIIColors.success(message)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count if LLM doesn't provide count_tokens method."""
        if hasattr(self.llm, 'count_tokens'):
            return self.llm.count_tokens(text)
        else:
            # Rough estimation: ~4 characters per token on average
            return len(text) // 4
    
    def _get_context_size(self) -> int:
        """Get context size from LLM or use default."""
        if hasattr(self.llm, 'get_context_size'):
            return self.llm.get_context_size() or 8192
        return 8192
    
    # ========================
    # LONG CONTEXT PROCESSING
    # ========================
    
    def long_context_processing(
        self,
        text_to_process: str,
        contextual_prompt: str,
        processing_type: str = "text",
        chunk_size_ratio: float = 0.5,
        overlap_ratio: float = 0.1,
        synthesis_prompt: str = None,
        images: list = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Process long context that exceeds LLM's context window by chunking and synthesizing.
        
        This method:
        1. Checks if text fits in context window
        2. If too long: splits into chunks, processes each, then synthesizes results
        3. If fits: processes directly
        
        Args:
            text_to_process: The long text/context to process
            contextual_prompt: The task/question to perform on the text
            processing_type: Type of processing - "text", "code", "structured", "yes_no", "multichoice"
            chunk_size_ratio: Ratio of context window to use per chunk (0.5 = 50%)
            overlap_ratio: Ratio of overlap between chunks (0.1 = 10%)
            synthesis_prompt: Custom prompt for synthesizing chunk results (optional)
            images: Optional images for multimodal models
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (schema for structured, possible_answers for multichoice, etc.)
        
        Returns:
            Processed result - type depends on processing_type
        """
        images = images or []
        ctx_size = self._get_context_size()
        
        # Estimate tokens in text
        text_tokens = self._estimate_tokens(text_to_process)
        prompt_tokens = self._estimate_tokens(contextual_prompt)
        
        # Use 70% of context for safety margin (and for compression threshold)
        available_tokens = int(ctx_size * 0.7)
        
        self._log_info(f"Context size: {ctx_size}, Text tokens: {text_tokens}, Available: {available_tokens}")
        
        # If text fits in context, process directly
        if text_tokens + prompt_tokens < available_tokens:
            self._log_info("Text fits in context window. Processing directly...")
            return self._process_with_context(
                text_to_process, contextual_prompt, processing_type,
                images, system_prompt, **kwargs
            )
        
        # Text is too long - use chunking strategy
        self._log_warning(f"Text exceeds context window ({text_tokens} > {available_tokens}). Using chunking strategy...")
        
        # Calculate chunk size in tokens
        chunk_size = int(available_tokens * chunk_size_ratio)
        overlap_size = int(chunk_size * overlap_ratio)
        
        self._log_info(f"Chunk size: {chunk_size} tokens, Overlap: {overlap_size} tokens")
        
        # Split text into chunks
        chunks = self._split_text_into_chunks(text_to_process, chunk_size, overlap_size)
        
        self._log_info(f"Split text into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_results = []
        current_text_buffer = "" # For text mode accumulation and compression
        
        for i, chunk in enumerate(chunks):
            self._log_info(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Explicitly instruct extraction vs generation to avoid premature task completion
            chunk_prompt = f"""You are analyzing part {i+1} of {len(chunks)} of a long document.
Your goal is to extract information that will be useful for this task:
"{contextual_prompt}"

INSTRUCTIONS:
1. Do NOT perform the final task yet.
2. Do NOT generate the final output (e.g. do not create slides, essays, etc.).
3. Extract relevant facts, statistics, quotes, narratives, and logical steps from this chunk that support the task.
4. If the chunk has no relevant info, return "NO_INFO".
5. Keep the extraction concise but detailed enough to be useful.

Document excerpt follows."""
            
            result = self._process_with_context(
                chunk, chunk_prompt, processing_type,
                images, system_prompt, **kwargs
            )
            
            if result and (isinstance(result, str) and "NO_INFO" not in result or isinstance(result, (dict, list))):
                if processing_type == "text":
                    # Accumulate text results
                    new_entry = f"\n\n--- Extracted from chunk {i+1} ---\n{result}"
                    current_text_buffer += new_entry
                    
                    # Check for context overflow
                    current_tokens = self._estimate_tokens(current_text_buffer)
                    if current_tokens > available_tokens:
                        self._log_info(f"Accumulated information ({current_tokens} tokens) exceeds limit. Compressing...")
                        
                        compression_prompt = f"""I am gathering information for this task: "{contextual_prompt}".
The collected notes are getting too long. Please consolidate and compress them.
Retain all key facts, statistics, and details necessary for the task.
Remove redundancies.

Current Notes:
{current_text_buffer}

Consolidated Notes:"""
                        
                        compressed_result = self.llm.generate_text(
                            compression_prompt,
                            images=images,
                            system_prompt=system_prompt,
                            **kwargs
                        )
                        
                        if isinstance(compressed_result, dict) and not compressed_result.get("status", True):
                             self._log_error("Compression failed. Keeping original buffer.")
                        else:
                             # Update buffer with compressed version
                             current_text_buffer = f"--- Consolidated Info (chunks 1-{i+1}) ---\n{compressed_result}"
                             self._log_success(f"Context compressed to {self._estimate_tokens(current_text_buffer)} tokens.")
                else:
                    # For non-text modes, just append to list as before
                    chunk_results.append({
                        'chunk_index': i,
                        'result': result
                    })
        
        if processing_type == "text":
            if not current_text_buffer:
                self._log_error("No relevant information found in any chunk.")
                return None
            # Wrap buffer into the format expected by synthesize
            # Use index -1 to indicate it's a consolidated buffer
            chunk_results = [{'chunk_index': -1, 'result': current_text_buffer}]
        
        if not chunk_results:
            self._log_error("Failed to process any chunks")
            return None
        
        self._log_success(f"Successfully processed chunks. Synthesizing results...")
        
        # Synthesize results from all chunks
        return self._synthesize_chunk_results(
            chunk_results, contextual_prompt, processing_type,
            synthesis_prompt, images, system_prompt, **kwargs
        )
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        Split text into overlapping chunks based on token estimates.
        
        Uses paragraph boundaries when possible for more coherent chunks.
        """
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # If single paragraph exceeds chunk size, split it by sentences
            if para_tokens > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self._estimate_tokens(sent)
                    
                    if current_tokens + sent_tokens > chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append('\n\n'.join(current_chunk))
                        
                        # Start new chunk with overlap
                        overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 1 else current_chunk[-1]
                        current_chunk = [overlap_text, sent]
                        current_tokens = self._estimate_tokens('\n\n'.join(current_chunk))
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
            else:
                # Regular paragraph handling
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append('\n\n'.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 1 else current_chunk[-1]
                    current_chunk = [overlap_text, para]
                    current_tokens = self._estimate_tokens('\n\n'.join(current_chunk))
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _process_with_context(
        self,
        context: str,
        prompt: str,
        processing_type: str,
        images: list,
        system_prompt: Optional[str],
        **kwargs
    ) -> Any:
        """Route to appropriate processing method based on type."""
        
        full_prompt = f"{prompt}\n\nContext:\n{context}"
        
        if processing_type == "text":
            return self.llm.generate_text(
                full_prompt,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        elif processing_type == "code":
            language = kwargs.get('language', 'python')
            return self.generate_code(
                prompt=full_prompt,
                images=images,
                system_prompt=system_prompt,
                language=language,
                **kwargs
            )
        
        elif processing_type == "structured":
            schema = kwargs.get('schema')
            if not schema:
                raise ValueError("schema required for structured processing")
            return self.generate_structured_content(
                prompt=full_prompt,
                schema=schema,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        elif processing_type == "yes_no":
            return self.yes_no(
                question=prompt,
                context=context,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        elif processing_type == "multichoice":
            possible_answers = kwargs.get('possible_answers')
            if not possible_answers:
                raise ValueError("possible_answers required for multichoice processing")
            return self.multichoice_question(
                question=prompt,
                possible_answers=possible_answers,
                context=context,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown processing_type: {processing_type}")
    
    def _synthesize_chunk_results(
        self,
        chunk_results: List[dict],
        original_prompt: str,
        processing_type: str,
        synthesis_prompt: Optional[str],
        images: list,
        system_prompt: Optional[str],
        **kwargs
    ) -> Any:
        """Synthesize results from multiple chunks into final answer."""
        
        if processing_type == "text":
            # Concatenate text results
            # Handle the case where we passed a consolidated buffer (chunk_index -1)
            combined_text = "\n\n---\n\n".join([
                f"From chunk {r['chunk_index']+1}:\n{r['result']}" if r.get('chunk_index', 0) >= 0 else r['result']
                for r in chunk_results
            ])
            
            if not synthesis_prompt:
                synthesis_prompt = f"""The following are results from processing different parts of a document.
Synthesize these into a coherent, comprehensive answer.

Original task: {original_prompt}

Results from chunks:
{combined_text}

Provide the final synthesized result:"""
            
            return self.llm.generate_text(
                synthesis_prompt,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        elif processing_type == "structured":
            # Merge structured results
            schema = kwargs.get('schema')
            combined_data = {
                "chunk_results": [r['result'] for r in chunk_results]
            }
            
            if not synthesis_prompt:
                synthesis_prompt = f"""The following are structured results from processing different parts of a document.
Merge and synthesize these into a single comprehensive result.

Original task: {original_prompt}

Chunk results:
{json.dumps(combined_data, indent=2)}

Provide the final synthesized result following the same schema:"""
            
            return self.generate_structured_content(
                prompt=synthesis_prompt,
                schema=schema,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
        
        elif processing_type == "yes_no":
            # Majority vote for yes/no
            yes_count = sum(1 for r in chunk_results if r['result'] is True)
            no_count = len(chunk_results) - yes_count
            
            return yes_count > no_count
        
        elif processing_type == "multichoice":
            # Majority vote for multichoice
            from collections import Counter
            votes = [r['result'] for r in chunk_results if isinstance(r['result'], int)]
            if not votes:
                return -1
            counter = Counter(votes)
            return counter.most_common(1)[0][0]
        
        elif processing_type == "code":
            # Combine code segments
            code_segments = [r['result'] for r in chunk_results if r['result']]
            
            if not synthesis_prompt:
                segments_text = "\n".join(
                    [f"# Segment {i+1}:\n{seg}" for i, seg in enumerate(code_segments)]
                )
                synthesis_prompt = f"""The following are code segments generated from different parts of a document.
Combine and synthesize these into complete, working code.

Original task: {original_prompt}

Code segments:
{segments_text}

Provide the final, complete code:"""
            
            return self.generate_code(
                prompt=synthesis_prompt,
                images=images,
                system_prompt=system_prompt,
                language=kwargs.get('language', 'python'),
                **kwargs
            )
        
        return None
    
    # ========================
    # CODE GENERATION
    # ========================
    
    def generate_code(
        self,
        prompt: str,
        images: list = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        language: str = "python",
        n_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        callback: Optional[Callable] = None,
        debug: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Generate a single code block based on a prompt using XML tags.
        
        Args:
            prompt: The user prompt requesting code
            images: List of images to include in the request
            system_prompt: Optional system prompt (will be extended with instructions)
            template: Optional code template to guide generation
            language: Programming language (python, javascript, java, etc.)
            n_predict: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repeat_penalty: Repetition penalty
            repeat_last_n: Last n tokens to consider for repetition penalty
            callback: Streaming callback function
            debug: Enable debug logging
            **kwargs: Additional parameters to pass to LLM
        
        Returns:
            The generated code as a string, or None if generation failed
        """
        images = images or []
        
        # Build system prompt
        if not system_prompt:
            system_prompt = "Act as a code generation assistant that generates code from user prompt."
        
        if template and template != "{}":
            system_prompt += f"\n\nHere is a template of the answer:\n<code language='{language}'>\n{template}\n</code>"
        
        system_prompt += f"""

You must wrap your code response in XML tags like this:
<code language="{language}">
your code here
</code>

Important rules:
- Return only ONE <code> tag
- Do not split the code into multiple tags
- All code must be inside the <code></code> tags
- Do not use markdown code fences (```)"""
        
        # Generate response
        response = self.llm.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback,
            **kwargs
        )
        
        if isinstance(response, dict) and not response.get("status", True):
            self._log_error(f"Code generation failed: {response.get('error')}")
            return None
        
        # Extract code using XML parsing
        code_content = self._extract_code_from_xml(response, language)
        
        if not code_content:
            self._log_warning("No code block found in response")
            return None
        
        # Check if code seems incomplete and try to continue if needed
        if self._is_code_incomplete(code_content):
            code_content = self._continue_incomplete_code(
                prompt, code_content, language, images,
                n_predict, temperature, top_k, top_p,
                repeat_penalty, repeat_last_n, callback, **kwargs
            )
        
        return code_content
    
    def generate_codes(
        self,
        prompt: str,
        images: list = None,
        system_prompt: Optional[str] = None,
        n_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        callback: Optional[Callable] = None,
        debug: bool = False,
        **kwargs
    ) -> List[dict]:
        """
        Generate multiple code files based on a prompt using XML tags.
        
        This is useful for generating entire projects or multi-file applications.
        
        Args:
            prompt: The user prompt requesting code
            images: List of images to include in the request
            system_prompt: Optional system prompt (will be extended with instructions)
            n_predict: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repeat_penalty: Repetition penalty
            repeat_last_n: Last n tokens to consider for repetition penalty
            callback: Streaming callback function
            debug: Enable debug logging
            **kwargs: Additional parameters to pass to LLM
        
        Returns:
            List of dicts with keys:
                - file_name: Name of the file
                - content: Code content
                - language: Programming language
                - is_complete: Whether the code block was properly closed
        """
        images = images or []
        
        # Build system prompt
        if not system_prompt:
            system_prompt = "Act as a code generation assistant that generates multiple code files from user prompt."
        
        system_prompt += """

You must wrap each code file in XML tags with a filename attribute like this:
<code language="python" filename="main.py">
your code here
</code>

<code language="javascript" filename="app.js">
your code here
</code>

Important rules:
- Return MULTIPLE <code> tags, one for each file
- Each tag must have a 'filename' attribute
- Each tag must have a 'language' attribute
- All code must be inside the <code></code> tags
- Do not use markdown code fences (```)
- Close each tag properly before starting the next one"""
        
        # Generate response
        response = self.llm.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback,
            **kwargs
        )
        
        if isinstance(response, dict) and not response.get("status", True):
            self._log_error(f"Multi-file code generation failed: {response.get('error')}")
            return []
        
        # Extract multiple code blocks
        code_files = self._extract_multiple_code_from_xml(response)
        
        if not code_files:
            self._log_warning("No code blocks found in response")
            return []
        
        self._log_success(f"Successfully extracted {len(code_files)} code file(s)")
        
        # Check for incomplete files and try to continue if needed
        for i, code_file in enumerate(code_files):
            if not code_file['is_complete'] or self._is_code_incomplete(code_file['content']):
                self._log_info(f"File '{code_file['file_name']}' appears incomplete, attempting continuation...")
                
                # Try to continue this specific file
                continued_content = self._continue_incomplete_code(
                    prompt=f"Continue the file {code_file['file_name']}:\n{prompt}",
                    partial_code=code_file['content'],
                    language=code_file['language'],
                    images=images,
                    n_predict=n_predict,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    repeat_last_n=repeat_last_n,
                    callback=callback,
                    **kwargs
                )
                
                code_files[i]['content'] = continued_content
                code_files[i]['is_complete'] = True
        
        return code_files
    
    def edit_code(
        self,
        original_code: str,
        edit_instruction: str,
        language: str = "python",
        file_name: str = "code",
        images: list = None,
        system_prompt: Optional[str] = None,
        max_chunk_size: int = 50,
        context_lines: int = 3,
        temperature: Optional[float] = None,
        fallback_to_full_rewrite: bool = True,
        **kwargs
    ) -> dict:
        """
        Efficiently edit code by having the LLM specify precise edit operations.
        
        This method uses a hybrid approach:
        1. Shows the LLM line-numbered code
        2. Asks for structured edit instructions (replace/insert/delete operations)
        3. Applies edits programmatically for reliability
        4. Falls back to full rewrite if edits fail
        
        Args:
            original_code: The original code to edit
            edit_instruction: What changes to make
            language: Programming language
            file_name: Name of the file being edited
            images: Optional images for context
            system_prompt: Optional system prompt
            max_chunk_size: Maximum lines to show in each chunk (for very long files)
            context_lines: Lines of context around changes
            temperature: Sampling temperature
            fallback_to_full_rewrite: If True, regenerate full code if edits fail
            **kwargs: Additional LLM parameters
        
        Returns:
            dict with keys:
                - success: bool - Whether edit succeeded
                - content: str - The edited code
                - method: str - 'structured_edit' or 'full_rewrite'
                - edits_applied: int - Number of edits applied
                - error: str - Error message if failed
        """
        images = images or []
        
        # Number the lines for reference
        lines = original_code.split('\n')
        numbered_code = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])
        
        # Build system prompt
        base_system = f"""You are a precise code editor. You will receive code with line numbers and an edit instruction.

Your task is to specify EXACT edits using structured JSON. Each edit operation should:
- Reference specific line numbers
- Be as minimal as possible
- Maintain code correctness

Available operations:
1. REPLACE: Replace specific lines with new content
2. INSERT: Insert new lines after a specific line
3. DELETE: Delete specific line range

Be very careful with line numbers - they must be exact."""

        if system_prompt:
            base_system = f"{system_prompt}\n\n{base_system}"
        
        user_prompt = f"""File: {file_name} ({language})

Current code with line numbers:
{numbered_code}

Edit instruction: {edit_instruction}

Analyze the code and specify the minimal edits needed. Return ONLY the edits, not the full code."""
        
        # Define schema for edit operations
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["replace", "insert", "delete"]
                            },
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "new_content": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["operation", "start_line"]
                    }
                },
                "summary": {"type": "string"}
            },
            "required": ["edits", "summary"]
        }
        
        # Get structured edits from LLM
        self._log_info(f"Analyzing code and generating edit instructions...")
        result = self.generate_structured_content(
            prompt=user_prompt,
            schema=schema,
            images=images,
            system_prompt=base_system,
            temperature=temperature or 0.3,  # Lower temperature for precision
            max_retries=2,
            **kwargs
        )
        
        if not result or not result.get('edits'):
            self._log_warning("Failed to generate structured edits")
            
            if fallback_to_full_rewrite:
                self._log_info("Falling back to full code rewrite...")
                return self._full_code_rewrite(
                    original_code, edit_instruction, language, 
                    file_name, images, system_prompt, **kwargs
                )
            
            return {
                "success": False,
                "content": original_code,
                "method": "none",
                "edits_applied": 0,
                "error": "Failed to generate edits and fallback disabled"
            }
        
        # Apply edits
        try:
            edited_code = self._apply_code_edits(lines, result['edits'])
            
            self._log_success(f"Successfully applied {len(result['edits'])} edit(s)")
            self._log_info(f"Summary: {result.get('summary', 'No summary provided')}")
            
            return {
                "success": True,
                "content": edited_code,
                "method": "structured_edit",
                "edits_applied": len(result['edits']),
                "summary": result.get('summary', ''),
                "edits": result['edits']
            }
            
        except Exception as e:
            self._log_error(f"Failed to apply edits: {e}")
            
            if fallback_to_full_rewrite:
                self._log_info("Falling back to full code rewrite...")
                return self._full_code_rewrite(
                    original_code, edit_instruction, language,
                    file_name, images, system_prompt, **kwargs
                )
            
            return {
                "success": False,
                "content": original_code,
                "method": "none",
                "edits_applied": 0,
                "error": str(e)
            }
    
    def _apply_code_edits(self, lines: List[str], edits: List[dict]) -> str:
        """
        Apply a list of edit operations to code lines.
        
        Operations are applied in reverse order (from bottom to top) to maintain
        line number validity.
        """
        # Sort edits by start_line in descending order
        sorted_edits = sorted(edits, key=lambda e: e['start_line'], reverse=True)
        
        # Make a copy of lines to edit
        result_lines = lines.copy()
        
        for edit in sorted_edits:
            operation = edit['operation'].lower()
            start_line = edit['start_line'] - 1  # Convert to 0-indexed
            
            if operation == 'delete':
                end_line = edit.get('end_line', start_line + 1) - 1
                
                # Validate line numbers
                if start_line < 0 or end_line >= len(result_lines):
                    self._log_warning(f"Invalid delete range: {start_line+1}-{end_line+1}")
                    continue
                
                # Delete lines
                del result_lines[start_line:end_line + 1]
                self._log_info(f"Deleted lines {start_line+1}-{end_line+1}")
            
            elif operation == 'replace':
                end_line = edit.get('end_line', start_line + 1) - 1
                new_content = edit.get('new_content', '')
                
                # Validate line numbers
                if start_line < 0 or end_line >= len(result_lines):
                    self._log_warning(f"Invalid replace range: {start_line+1}-{end_line+1}")
                    continue
                
                # Replace lines
                new_lines = new_content.split('\n')
                result_lines[start_line:end_line + 1] = new_lines
                self._log_info(f"Replaced lines {start_line+1}-{end_line+1} with {len(new_lines)} line(s)")
            
            elif operation == 'insert':
                new_content = edit.get('new_content', '')
                
                # Validate line number
                if start_line < 0 or start_line > len(result_lines):
                    self._log_warning(f"Invalid insert position: {start_line+1}")
                    continue
                
                # Insert after start_line
                new_lines = new_content.split('\n')
                result_lines[start_line + 1:start_line + 1] = new_lines
                self._log_info(f"Inserted {len(new_lines)} line(s) after line {start_line+1}")
        
        return '\n'.join(result_lines)
    
    def _full_code_rewrite(
        self,
        original_code: str,
        edit_instruction: str,
        language: str,
        file_name: str,
        images: list,
        system_prompt: Optional[str],
        **kwargs
    ) -> dict:
        """Fallback method: regenerate the entire code with modifications."""
        
        base_system = f"You are a code editor. Rewrite the provided code with the requested changes."
        if system_prompt:
            base_system = f"{system_prompt}\n\n{base_system}"
        
        prompt = f"""File: {file_name} ({language})

Original code:
{original_code}

Instructions: {edit_instruction}

Provide the complete modified code."""
        
        new_code = self.generate_code(
            prompt=prompt,
            images=images,
            system_prompt=base_system,
            language=language,
            **kwargs
        )
        
        if new_code:
            return {
                "success": True,
                "content": new_code,
                "method": "full_rewrite",
                "edits_applied": 1
            }
        else:
            return {
                "success": False,
                "content": original_code,
                "method": "none",
                "edits_applied": 0,
                "error": "Full rewrite failed"
            }
    
    def _extract_code_from_xml(self, response: str, language: str) -> Optional[str]:
        """Extract code from XML tags using regex."""
        # Try to find code with specific language
        pattern = rf'<code\s+language=["\']?{re.escape(language)}["\']?>(.*?)</code>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Try to find any code tag
        pattern = r'<code(?:\s+language=["\']?[^"\']*["\']?)?>(.*?)</code>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_multiple_code_from_xml(self, response: str) -> List[dict]:
        """
        Extract multiple code blocks from XML tags.
        
        Returns list of dicts with file_name, content, language, and is_complete.
        """
        code_files = []
        
        # Pattern to match code tags with attributes and content
        pattern = r'<code\s+([^>]*)>(.*?)(?:</code>|$)'
        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            attributes_str = match.group(1)
            content = match.group(2).strip()
            
            # Check if the tag was properly closed
            is_complete = match.group(0).endswith('</code>')
            
            # Extract language attribute
            language_match = re.search(r'language=["\']?([^"\'>\s]+)["\']?', attributes_str, re.IGNORECASE)
            language = language_match.group(1) if language_match else "text"
            
            # Extract filename attribute
            filename_match = re.search(r'filename=["\']?([^"\'>\s]+)["\']?', attributes_str, re.IGNORECASE)
            file_name = filename_match.group(1) if filename_match else f"file_{len(code_files)}.{language}"
            
            code_files.append({
                'file_name': file_name,
                'content': content,
                'language': language,
                'is_complete': is_complete
            })
        
        return code_files
    
    def _is_code_incomplete(self, code: str) -> bool:
        """Heuristic check if code appears incomplete."""
        if not code:
            return False
        
        # Check for common incomplete patterns
        incomplete_indicators = [
            code.rstrip().endswith(('...', 'TODO', 'FIXME')),
            code.count('{') != code.count('}'),
            code.count('(') != code.count(')'),
            code.count('[') != code.count(']'),
            code.rstrip().endswith((',', '\\', '&', '|', '+', '-', '*', '/')),
        ]
        
        return any(incomplete_indicators)
    
    def _continue_incomplete_code(
        self,
        original_prompt: str,
        partial_code: str,
        language: str,
        images: list,
        n_predict: Optional[int],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repeat_penalty: Optional[float],
        repeat_last_n: Optional[int],
        callback: Optional[Callable],
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Attempt to continue incomplete code generation."""
        full_code = partial_code
        
        for retry in range(max_retries):
            self._log_info(f"Code appears incomplete. Continuation attempt {retry + 1}/{max_retries}...")
            
            continuation_prompt = f"""The following code generation was incomplete:

<code language="{language}">
{full_code}
</code>

Continue the code exactly from where it left off. Do not repeat any part of the previous code. Only provide the continuation inside a <code> tag."""
            
            continuation_response = self.llm.generate_text(
                continuation_prompt,
                images=images,
                n_predict=n_predict,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,
                streaming_callback=callback,
                **kwargs
            )
            
            if isinstance(continuation_response, dict) and not continuation_response.get("status", True):
                break
            
            continued_part = self._extract_code_from_xml(continuation_response, language)
            
            if not continued_part:
                break
            
            full_code += "\n" + continued_part
            
            if not self._is_code_incomplete(full_code):
                self._log_success("Code completion successful!")
                break
        
        return full_code


    def _extract_json_from_xml(self, response: str) -> Optional[str]:
        """Extract JSON content from XML tags, avoiding nested occurrences in other tags."""
        
        # First, try to remove common wrapper tags that might contain nested <json> references
        # This handles cases where LLM uses <think>, <reasoning>, etc. tags
        cleaned_response = response
        for wrapper_tag in ['think', 'reasoning', 'analysis', 'scratchpad', 'internal']:
            # Remove content within these wrapper tags to avoid false matches
            cleaned_response = re.sub(
                f'<{wrapper_tag}>.*?</{wrapper_tag}>', 
                '', 
                cleaned_response, 
                flags=re.DOTALL | re.IGNORECASE
            )
        
        # Now search for <json> tags in the cleaned response
        pattern = r'<json>(.*?)</json>'
        matches = list(re.finditer(pattern, cleaned_response, re.DOTALL | re.IGNORECASE))
        
        if matches:
            # Return the last complete match (most likely to be the actual output)
            return matches[-1].group(1).strip()
        
        # Fallback: try original response for closed tags if cleaning removed everything
        matches = list(re.finditer(pattern, response, re.DOTALL | re.IGNORECASE))
        if matches:
            # Return the last match from original response
            return matches[-1].group(1).strip()
        
        # Last resort: try to find unclosed json tag (for truncated responses)
        pattern = r'<json>(.*?)$'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            self._log_warning("Found unclosed <json> tag - response may be truncated")
            return match.group(1).strip()
        
        return None
    
    def _create_schema_example(self, schema: dict) -> any:
        """Create an example instance from a JSON schema."""
        if not isinstance(schema, dict):
            return {}
        
        if "const" in schema:
            return schema["const"]
        
        if "default" in schema:
            return schema["default"]
        
        schema_type = schema.get("type")
        
        if schema_type == "string":
            return schema.get("examples", [""])[0] if "examples" in schema else ""
        elif schema_type == "integer":
            return schema.get("examples", [0])[0] if "examples" in schema else 0
        elif schema_type == "number":
            return schema.get("examples", [0.0])[0] if "examples" in schema else 0.0
        elif schema_type == "boolean":
            return schema.get("examples", [False])[0] if "examples" in schema else False
        elif schema_type == "array":
            item_schema = schema.get("items", {})
            return [self._create_schema_example(item_schema)] if item_schema else []
        elif schema_type == "object":
            result = {}
            for key, value_schema in schema.get("properties", {}).items():
                result[key] = self._create_schema_example(value_schema)
            return result
        
        return {}

    def _try_parse_json_with_truncation_detection(self, json_string: str) -> dict:
        """
        Attempt to parse JSON and detect if it was truncated.
        
        Returns:
            dict with keys:
                - success: bool
                - truncated: bool
                - data: parsed JSON if successful
                - error: error message if failed
        """
        try:
            parsed = json.loads(json_string)
            return {"success": True, "truncated": False, "data": parsed}
        except json.JSONDecodeError as e:
            # Analyze the error to detect truncation
            error_msg = str(e)
            
            # Common truncation patterns
            truncation_indicators = [
                "Expecting ',' delimiter",
                "Expecting ':' delimiter",
                "Expecting property name",
                "Unterminated string",
                "Expecting value",
                "Unterminated",
            ]
            
            is_truncated = any(indicator in error_msg for indicator in truncation_indicators)
            
            # Additional check: does the string end abruptly?
            stripped = json_string.rstrip()
            ends_abruptly = not stripped.endswith(('}', ']', '"', 'null', 'true', 'false')) or \
                           stripped.count('{') != stripped.count('}') or \
                           stripped.count('[') != stripped.count(']')
            
            is_truncated = is_truncated or ends_abruptly
            
            return {
                "success": False,
                "truncated": is_truncated,
                "error": error_msg,
                "parse_position": e.pos if hasattr(e, 'pos') else None
            }    

    def _continue_truncated_json(
        self,
        original_prompt: str,
        partial_json: str,
        images: list,
        system_prompt: str,
        max_continuations: int = 3,
        **kwargs
    ) -> Optional[str]:
        """
        Continue a truncated JSON response.
        
        Returns the complete JSON string or None if continuation failed.
        """
        full_json = partial_json
        
        for continuation_num in range(max_continuations):
            self._log_info(f"JSON continuation {continuation_num + 1}/{max_continuations}...")
            
            # Create continuation prompt
            continuation_prompt = f"""The following JSON generation was incomplete (truncated):

<json>
{full_json}
</json>

Continue the JSON exactly from where it left off. Do not repeat any previous content. Only provide the continuation wrapped in <json> tags. Make sure to properly close all arrays and objects."""
            
            continuation_response = self.llm.generate_text(
                continuation_prompt,
                images=images,
                system_prompt=system_prompt,
                **kwargs
            )
            
            if isinstance(continuation_response, dict) and not continuation_response.get("status", True):
                self._log_error("Continuation generation failed")
                break
            
            # Extract the continuation
            continued_part = self._extract_json_from_xml(continuation_response)
            
            if not continued_part:
                self._log_warning("No JSON found in continuation response")
                break
            
            # Merge the continuation
            # Remove trailing incomplete tokens from partial_json
            full_json = full_json.rstrip()
            if full_json.endswith(','):
                full_json += "\n" + continued_part
            else:
                # Try to intelligently merge
                full_json += continued_part
            
            # Check if it's now valid
            parse_result = self._try_parse_json_with_truncation_detection(full_json)
            
            if parse_result["success"]:
                self._log_success(f"JSON continuation successful after {continuation_num + 1} attempt(s)")
                return full_json
            elif not parse_result["truncated"]:
                # Not truncated but still invalid - this is a different error
                self._log_error(f"JSON continuation produced invalid JSON: {parse_result.get('error')}")
                break
        
        self._log_error("Failed to complete truncated JSON after all continuation attempts")
        return None
    # ========================
    # STRUCTURED JSON GENERATION
    # ========================
    
    def generate_structured_content(
        self,
        prompt: str,
        schema: Union[dict, str],
        images: list = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        auto_continue_on_truncation: bool = True,
        **kwargs
    ) -> Optional[dict]:
        """Generate structured JSON content conforming to a schema."""
        images = images or []
        
        try:
            from jsonschema import validate
            has_validator = True
        except ImportError:
            has_validator = False
            self._log_warning("jsonschema not available, skipping validation")
        
        # Parse schema
        if isinstance(schema, dict):
            schema_obj = schema
        elif isinstance(schema, str):
            try:
                schema_obj = json.loads(schema)
            except json.JSONDecodeError:
                raise ValueError("The provided schema string is not valid JSON")
        else:
            raise TypeError("schema must be a dict or a JSON string")
        
        # Normalize schema
        if "type" not in schema_obj and "properties" not in schema_obj:
            if all(isinstance(v, dict) for v in schema_obj.values()):
                schema_obj = {
                    "type": "object",
                    "properties": schema_obj,
                    "required": list(schema_obj.keys())
                }
        
        # Create example from schema
        example_instance = self._create_schema_example(schema_obj)
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
        example_str = json.dumps(example_instance, indent=2, ensure_ascii=False)
        
        # Build system prompt
        base_system = f"""Your objective is to generate a JSON object that satisfies the user's request and conforms to the provided schema.

Schema:
{schema_str}

Example structure:
{example_str}

You **MUST** wrap your JSON response in XML tags:
<json>
your json here
</json>

Important:
- Return valid JSON only
- Follow the schema exactly
- Use proper JSON syntax with double quotes
- Do not include any text outside the <json> tags
- Ensure the JSON is COMPLETE - close all arrays and objects properly"""
        
        final_system_prompt = f"{system_prompt}\n\n{base_system}" if system_prompt else base_system
        kwargs["ctx_size"] = kwargs.get("ctx_size", self.llm.default_ctx_size)
        
        # Attempt generation with retries
        for attempt in range(max_retries):
            response = self.llm.generate_text(
                prompt,
                images=images,
                system_prompt=final_system_prompt,
                **kwargs
            )
            response = self.remove_thinking_blocks(response)
            if isinstance(response, dict) and not response.get("status", True):
                continue
            
            json_string = self._extract_json_from_xml(response)
            
            if not json_string:
                self._log_warning(f"Attempt {attempt + 1}: No JSON found in response")
                continue
            
            parse_result = self._try_parse_json_with_truncation_detection(json_string)
            
            if parse_result["success"]:
                parsed_json = parse_result["data"]
                
                if has_validator:
                    try:
                        validate(instance=parsed_json, schema=schema_obj)
                        self._log_success("JSON validated successfully against schema")
                        return parsed_json
                    except Exception as e:
                        self._log_warning(f"Attempt {attempt + 1}: Validation failed - {e}")
                        if attempt == max_retries - 1:
                            return parsed_json
                        continue
                else:
                    return parsed_json
            
            elif parse_result["truncated"] and auto_continue_on_truncation:
                self._log_warning(f"Attempt {attempt + 1}: JSON appears truncated. Attempting continuation...")
                
                continued_json = self._continue_truncated_json(
                    original_prompt=prompt,
                    partial_json=json_string,
                    images=images,
                    system_prompt=final_system_prompt,
                    **kwargs
                )
                
                if continued_json:
                    final_parse = self._try_parse_json_with_truncation_detection(continued_json)
                    if final_parse["success"]:
                        parsed_json = final_parse["data"]
                        
                        if has_validator:
                            try:
                                validate(instance=parsed_json, schema=schema_obj)
                                self._log_success("Continued JSON validated successfully")
                                return parsed_json
                            except:
                                if attempt == max_retries - 1:
                                    return parsed_json
                        else:
                            return parsed_json
            else:
                self._log_error(f"Attempt {attempt + 1}: Failed to parse JSON - {parse_result.get('error')}")
        
        self._log_error("Failed to generate valid structured content after all retries")
        return None
    
    # Helper methods for structured content (truncation detection, etc.) would go here
    # See part 1 for _try_parse_json_with_truncation_detection, _continue_truncated_json, etc.
    
    # ========================
    # HELPER METHODS FOR COMMON TASKS
    # ========================
    
    def yes_no(self, question: str, context: str = "", system_prompt: str = "", 
               return_explanation: bool = False, images: list = None, **kwargs) -> Union[bool, dict]:
        """Ask the LLM a yes/no question and get a boolean answer."""
        images = images or []
        
        base_system = "You are a helpful assistant that answers yes/no questions accurately and concisely."
        if system_prompt:
            base_system = f"{system_prompt}\n\n{base_system}"
        
        user_prompt = "Answer the following question with only 'true' or 'false' and provide a brief explanation."
        if context:
            user_prompt += f"\n\nContext:\n{context}"
        user_prompt += f"\n\nQuestion: {question}"
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "boolean"},
                "explanation": {"type": "string"}
            },
            "required": ["answer", "explanation"]
        }
        
        result = self.generate_structured_content(
            prompt=user_prompt, schema=schema, images=images,
            system_prompt=base_system, max_retries=2, **kwargs
        )
        
        if not result:
            return {"answer": False, "explanation": "Failed"} if return_explanation else False
        
        answer = result.get("answer", False)
        if isinstance(answer, str):
            answer = answer.lower() == 'true'
        
        return {"answer": answer, "explanation": result.get("explanation", "")} if return_explanation else answer
    
    def multichoice_question(self, question: str, possible_answers: list, context: str = "",
                            return_explanation: bool = False, **kwargs) -> Union[int, dict]:
        """Ask the LLM a multiple choice question."""
        choices = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])
        
        user_prompt = "Answer the multiple-choice question by selecting the index of the best answer."
        if context:
            user_prompt += f"\n\nContext:\n{context}"
        user_prompt += f"\n\nQuestion:\n{question}\n\nPossible Answers:\n{choices}"
        
        schema = {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0, "maximum": len(possible_answers) - 1},
                "explanation": {"type": "string"}
            },
            "required": ["index", "explanation"]
        }
        
        result = self.generate_structured_content(prompt=user_prompt, schema=schema, max_retries=2, **kwargs)
        
        if not result:
            return {"index": -1, "explanation": "Failed"} if return_explanation else -1
        
        index = result.get("index", -1)
        if not (0 <= index < len(possible_answers)):
            index = -1
        
        return {"index": index, "explanation": result.get("explanation", "")} if return_explanation else index
    
    def multichoice_ranking(
        self,
        question: str,
        possible_answers: list,
        context: str = "",
        return_explanation: bool = False,
        **kwargs
    ) -> Union[list[int], dict]:
        """Ask the LLM to rank multiple choice answers from best to worst."""
        choices = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])

        user_prompt = (
            "Rank the possible answers from best to worst by returning a list of indices."
        )
        if context:
            user_prompt += f"\n\nContext:\n{context}"
        user_prompt += f"\n\nQuestion:\n{question}\n\nPossible Answers:\n{choices}"

        schema = {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": len(possible_answers) - 1
                    },
                    "minItems": len(possible_answers),
                    "maxItems": len(possible_answers),
                    "uniqueItems": True
                },
                "explanation": {"type": "string"}
            },
            "required": ["ranking", "explanation"]
        }

        result = self.generate_structured_content(
            prompt=user_prompt,
            schema=schema,
            max_retries=2,
            **kwargs
        )

        if not result:
            empty = [] if not return_explanation else {"ranking": [], "explanation": "Failed"}
            return empty

        ranking = result.get("ranking", [])
        if (
            not isinstance(ranking, list)
            or len(ranking) != len(possible_answers)
            or any(not isinstance(i, int) or i < 0 or i >= len(possible_answers) for i in ranking)
            or len(set(ranking)) != len(ranking)
        ):
            ranking = []

        if return_explanation:
            return {
                "ranking": ranking,
                "explanation": result.get("explanation", "")
            }

        return ranking

    def summerize_text(self, text: str, summary_instruction: str = "Summarize concisely", **kwargs) -> str:
        """Summarize a given text."""
        prompt = f"{summary_instruction}\n\nText to summarize:\n{text}"
        response = self.llm.generate_text(prompt, **kwargs)
        
        if isinstance(response, dict) and not response.get("status", True):
            return ""
        return response.strip()
    
    def extract_keywords(self, text: str, num_keywords: int = 5, **kwargs) -> list:
        """Extract key keywords/phrases from text."""
        prompt = f"Extract the {num_keywords} most important keywords from:\n\n{text}"
        
        schema = {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        result = self.generate_structured_content(prompt=prompt, schema=schema, **kwargs)
        return result.get("keywords", []) if result else []
    
    # ========================
    # TEXT PARSING UTILITIES
    # ========================
    
    def extract_thinking_blocks(self, text: str) -> List[str]:
        """Extract content between <thinking> or <think> tags."""
        pattern = r'<(thinking|think)>(.*?)</\1>'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        return [match.group(2).strip() for match in matches]
    
    def remove_thinking_blocks(self, text: str) -> str:
        """Remove thinking blocks from text."""
        pattern = r'<(thinking|think)>.*?</\1>\s*'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    
    def extract_code_blocks(self, text: str, format: str = "markdown") -> List[dict]:
        """
        Extracts code blocks from text in Markdown or HTML format.
        
        This is the legacy method for extracting markdown/HTML code blocks.
        For new code, prefer using generate_code() with XML tags.
        
        Args:
            text: The text containing code blocks
            format: Either "markdown" (```) or "html" (<code>)
        
        Returns:
            List of dicts with keys:
                - index: Block index
                - file_name: Extracted filename if present
                - content: Code content
                - type: Programming language
                - is_complete: Whether block is properly closed
        """
        code_blocks = []
        remaining = text
        first_index = 0
        indices = []

        if format.lower() == "markdown":
            while remaining:
                try:
                    index = remaining.index("```")
                    indices.append(index + first_index)
                    remaining = remaining[index + 3:]
                    first_index += index + 3
                except ValueError:
                    if len(indices) % 2 == 1:
                        indices.append(first_index + len(remaining))
                    break

        elif format.lower() == "html":
            cursor = 0
            while cursor < len(text):
                try:
                    start_index = text.index("<code", cursor)
                    try:
                        end_of_opening = text.index(">", start_index)
                    except ValueError:
                        break

                    indices.append(start_index)
                    opening_tag_end = end_of_opening + 1
                    cursor = opening_tag_end

                    nest_level = 0
                    temp_cursor = cursor
                    found_closing = False
                    while temp_cursor < len(text):
                        if text[temp_cursor:].startswith("<code"):
                            nest_level += 1
                            try:
                                temp_cursor = text.index(">", temp_cursor) + 1
                            except ValueError:
                                break 
                        elif text[temp_cursor:].startswith("</code>"):
                            if nest_level == 0:
                                indices.append(temp_cursor)
                                cursor = temp_cursor + len("</code>")
                                found_closing = True
                                break
                            nest_level -= 1
                            temp_cursor += len("</code>")
                        else:
                            temp_cursor += 1

                    if not found_closing:
                        indices.append(len(text))
                        break

                except ValueError:
                    break
        else:
            raise ValueError("Format must be 'markdown' or 'html'")

        for i in range(0, len(indices), 2):
            block_infos = {
                'index': i // 2,
                'file_name': "",
                'content': "",
                'type': 'language-specific',
                'is_complete': False
            }

            start_pos = indices[i]
            search_area_start = max(0, start_pos - 200)
            preceding_text_segment = text[search_area_start:start_pos]
            lines = preceding_text_segment.strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line.startswith("<file_name>") and last_line.endswith("</file_name>"):
                    block_infos['file_name'] = last_line[len("<file_name>"):-len("</file_name>")].strip()
                elif last_line.lower().startswith("file:") or last_line.lower().startswith("filename:"):
                    block_infos['file_name'] = last_line.split(":", 1)[1].strip()

            if format.lower() == "markdown":
                content_start = start_pos + 3
                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content_raw = text[content_start:end_pos]
                    block_infos['is_complete'] = True
                else:
                    content_raw = text[content_start:]
                    block_infos['is_complete'] = False

                first_line_end = content_raw.find('\n')
                if first_line_end != -1:
                    first_line = content_raw[:first_line_end].strip()
                    if first_line and not first_line.isspace() and ' ' not in first_line:
                        block_infos['type'] = first_line
                        content = content_raw[first_line_end + 1:].strip()
                    else:
                        content = content_raw.strip()
                else:
                    content = content_raw.strip()
                    if content and not content.isspace() and ' ' not in content and len(content)<20:
                         block_infos['type'] = content
                         content = ""

            elif format.lower() == "html":
                try:
                    opening_tag_end = text.index(">", start_pos) + 1
                except ValueError:
                    continue

                opening_tag = text[start_pos:opening_tag_end]

                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content = text[opening_tag_end:end_pos].strip()
                    block_infos['is_complete'] = True
                else:
                    content = text[opening_tag_end:].strip()
                    block_infos['is_complete'] = False

                match = re.search(r'class\s*=\s*["\']([^"\']*)["\']', opening_tag)
                if match:
                    classes = match.group(1).split()
                    for cls in classes:
                        if cls.startswith("language-"):
                            block_infos['type'] = cls[len("language-"):]
                            break

            block_infos['content'] = content
            if block_infos['content'] or block_infos['is_complete']:
                code_blocks.append(block_infos)

        return code_blocks
