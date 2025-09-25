import importlib.util
import json
from pathlib import Path
from typing import Callable, List, Optional, Union

class LollmsPersonality:
    """
    A class that encapsulates the full personality of an AI agent.

    This includes its identity, system prompts, specialized knowledge (via RAG),
    and custom execution logic (via a Python script). It is designed to be a
    portable and self-contained unit that can be loaded and used by any
    application using the lollms-client.
    """
    def __init__(
        self,
        # Metadata for identification and display
        name: str,
        author: str,
        category: str,
        description: str,
        # Core behavioral instruction
        system_prompt: str,        
        icon: Optional[str] = None,  # Base64 encoded image string
        active_mcps: Optional[List[str]] = None, # The list of MCPs to activate with this personality
        data_source: Optional[Union[str, Callable[[str], str]]] = None, # Static string data or a callable for dynamic data retrieval


        # RAG - Data Files and Application-provided Callbacks
        data_files: Optional[List[Union[str, Path]]] = None,
        vectorize_chunk_callback: Optional[Callable[[str, str], None]] = None,  # (chunk_text, chunk_id) -> None
        is_vectorized_callback: Optional[Callable[[str], bool]] = None,        # (chunk_id) -> bool
        query_rag_callback: Optional[Callable[[str], str]] = None,              # (user_query) -> rag_context_str

        # Custom Logic Override
        script: Optional[str] = None,  # The Python script as a raw string

        # Internal state
        personality_id: Optional[str] = None
    ):
        """
        Initializes a LollmsPersonality instance.

        Args:
            name: The display name of the personality.
            author: The author of the personality.
            category: A category for organization (e.g., 'Code', 'Writing', 'Fun').
            description: A brief description of what the personality does.
            icon: An optional base64 encoded string for a display icon.
            system_prompt: The core system prompt that defines the AI's behavior.
            active_mcps: An optional list of MCP (tool) names to be automatically activated with this personality.
            data_source: A source of knowledge. Can be a static string or a callable function that takes a query and returns a string.
            data_files: A list of file paths to be used as a knowledge base for RAG.
            vectorize_chunk_callback: A function provided by the host app to vectorize and store a text chunk.
            is_vectorized_callback: A function provided by the host app to check if a chunk is already vectorized.
            query_rag_callback: A function provided by the host app to query the vector store for relevant context.
            script: A string containing a Python script to override default chat behavior.
            personality_id: An optional unique identifier. If not provided, it's generated from the author and name.
        """
        self.name = name
        self.author = author
        self.category = category
        self.description = description
        self.icon = icon
        self.system_prompt = system_prompt
        self.active_mcps = active_mcps or []
        self.data_source = data_source
        self.data_files = [Path(f) for f in data_files] if data_files else []
        
        # RAG Callbacks provided by the host application
        self.vectorize_chunk_callback = vectorize_chunk_callback
        self.is_vectorized_callback = is_vectorized_callback
        self.query_rag_callback = query_rag_callback
        
        self.script = script
        self.script_module = None
        self.personality_id = personality_id or self._generate_id()
        
        # Prepare custom logic and data upon initialization
        self._prepare_script()
        self.ensure_data_vectorized()

    def _generate_id(self) -> str:
        """
        Creates a filesystem-safe, unique ID based on the author and name.
        """
        if self.author:
            safe_author = "".join(c if c.isalnum() else '_' for c in self.author)
        else:
            safe_author = "".join(c if c.isalnum() else '_' for c in "ParisNeo")
        safe_name = "".join(c if c.isalnum() else '_' for c in self.name)
        return f"{safe_author}_{safe_name}"

    def _prepare_script(self):
        """
        Dynamically loads the personality's script as an in-memory Python module.

        This allows the script to be executed without being saved as a .py file on disk,
        making the system more secure and self-contained.
        """
        if not self.script:
            return
        try:
            module_name = f"lollms_personality_script_{self.personality_id}"
            
            # Create a module specification and a module object from it
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            self.script_module = importlib.util.module_from_spec(spec)
            
            # Execute the script code within the new module's namespace
            exec(self.script, self.script_module.__dict__)
            print(f"[{self.name}] Custom script loaded successfully.")
        except Exception as e:
            print(f"[{self.name}] Failed to load custom script: {e}")
            self.script_module = None

    def ensure_data_vectorized(self, chunk_size: int = 1024):
        """
        Checks if the personality's data files are vectorized using the host callbacks.
        
        It iterates through each data file, splits it into chunks, and for each chunk,
        it checks if it's already processed. If not, it calls the vectorization callback
        provided by the host application.

        Args:
            chunk_size: The size of each text chunk to process for vectorization.
        """
        if not self.data_files or not self.vectorize_chunk_callback or not self.is_vectorized_callback:
            return
            
        print(f"[{self.name}] Checking RAG data vectorization...")
        all_vectorized = True
        for file_path in self.data_files:
            if not file_path.exists():
                print(f"  - Warning: Data file not found, skipping: {file_path}")
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    # Generate a unique and deterministic ID for each chunk
                    chunk_id = f"{self.personality_id}_{file_path.name}_chunk_{i}"
                    
                    if not self.is_vectorized_callback(chunk_id):
                        all_vectorized = False
                        print(f"  - Vectorizing '{file_path.name}' chunk {i+1}/{len(chunks)}...")
                        self.vectorize_chunk_callback(chunk, chunk_id)

            except Exception as e:
                print(f"  - Error processing file {file_path.name}: {e}")
                continue
        
        if all_vectorized:
            print(f"[{self.name}] All RAG data is already vectorized.")
        else:
            print(f"[{self.name}] RAG data vectorization complete.")
        
    def get_rag_context(self, query: str) -> Optional[str]:
        """
        Queries the vectorized data to get relevant context for a given query.

        This method relies on the `query_rag_callback` provided by the host application
        to perform the actual search in the vector store.

        Args:
            query: The user's query string.

        Returns:
            A string containing the relevant context, or None if no callback is available.
        """
        if not self.query_rag_callback:
            return None
        return self.query_rag_callback(query)

    def to_dict(self) -> dict:
        """
        Serializes the personality's metadata to a dictionary.
        Note: Callbacks and the script module are not serialized.
        """
        return {
            "personality_id": self.personality_id,
            "name": self.name,
            "author": self.author,
            "category": self.category,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "active_mcps": self.active_mcps,
            "has_data_source": self.data_source is not None,
            "data_files": [str(p) for p in self.data_files],
            "has_script": self.script is not None
        }
