from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Dict, Any, Optional, Callable
import json
from pathlib import Path

# --- Mock RAG Implementation ---
# In a real application, this would interact with your vector database (Pinecone, ChromaDB, FAISS, etc.)
# and use a real sentence transformer for vectorization.

MOCK_KNOWLEDGE_BASE = {
    "python_basics.md": [
        {"chunk_id": 1, "text": "Python is a high-level, interpreted programming language known for its readability and versatility. It was created by Guido van Rossum and first released in 1991."},
        {"chunk_id": 2, "text": "Key features of Python include dynamic typing, automatic memory management (garbage collection), and a large standard library. It supports multiple programming paradigms, such as procedural, object-oriented, and functional programming."},
        {"chunk_id": 3, "text": "Common applications of Python include web development (e.g., Django, Flask), data science (e.g., Pandas, NumPy, Scikit-learn), machine learning, artificial intelligence, automation, and scripting."},
    ],
    "javascript_info.js": [
        {"chunk_id": 1, "text": "JavaScript is a scripting language primarily used for front-end web development to create interactive effects within web browsers. It is also used in back-end development (Node.js), mobile app development, and game development."},
        {"chunk_id": 2, "text": "JavaScript is dynamically typed, prototype-based, and multi-paradigm. Along with HTML and CSS, it is one of the core technologies of the World Wide Web."},
        {"chunk_id": 3, "text": "Popular JavaScript frameworks and libraries include React, Angular, Vue.js for front-end, and Express.js for Node.js back-end applications."},
    ],
    "ai_concepts.txt": [
        {"chunk_id": 1, "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving."},
        {"chunk_id": 2, "text": "Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep Learning (DL) is a further subset of ML based on artificial neural networks with representation learning."},
        {"chunk_id": 3, "text": "Retrieval Augmented Generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information."},
    ]
}

def mock_rag_query_function(
    query_text: str,
    vectorizer_name: Optional[str] = None, # Ignored in mock
    top_k: int = 3,
    min_similarity_percent: float = 0.0 # Ignored in mock, simple keyword match
) -> List[Dict[str, Any]]:
    """
    A mock RAG query function.
    Performs a simple keyword search in the MOCK_KNOWLEDGE_BASE.
    """
    ASCIIColors.magenta(f"  [MOCK RAG] Querying with: '{query_text}', top_k={top_k}")
    results = []
    query_lower = query_text.lower()
    
    all_chunks = []
    for file_path, chunks_in_file in MOCK_KNOWLEDGE_BASE.items():
        for chunk_data in chunks_in_file:
            all_chunks.append({"file_path": file_path, **chunk_data})

    # Simple keyword matching and scoring (very basic)
    scored_chunks = []
    for chunk_info in all_chunks:
        score = 0
        for keyword in query_lower.split():
            if keyword in chunk_info["text"].lower() and len(keyword)>2: # Basic relevance
                score += 1
        if "python" in query_lower and "python" in chunk_info["file_path"].lower(): score+=5
        if "javascript" in query_lower and "javascript" in chunk_info["file_path"].lower(): score+=5
        if "ai" in query_lower and "ai" in chunk_info["file_path"].lower(): score+=3


        if score > 0 : # Only include if some keywords match
            # Simulate similarity percentage (higher score = higher similarity)
            similarity = min(100.0, score * 20.0 + 40.0) # Arbitrary scaling
            if similarity >= min_similarity_percent:
                scored_chunks.append({
                    "file_path": chunk_info["file_path"],
                    "chunk_text": chunk_info["text"],
                    "similarity_percent": similarity,
                    "_score_for_ranking": score # Internal score for sorting
                })
    
    # Sort by internal score (descending) and take top_k
    scored_chunks.sort(key=lambda x: x["_score_for_ranking"], reverse=True)
    results = [
        {"file_path": c["file_path"], "chunk_text": c["chunk_text"], "similarity_percent": c["similarity_percent"]}
        for c in scored_chunks[:top_k]
    ]
    ASCIIColors.magenta(f"  [MOCK RAG] Found {len(results)} relevant chunks.")
    return results

# --- Streaming Callback for RAG and LLM ---
def rag_streaming_callback(
    chunk: str, 
    msg_type: MSG_TYPE, 
    metadata: Optional[Dict] = None, 
    turn_history: Optional[List] = None # history of this specific RAG turn
) -> bool:
    """
    Handles various stages of RAG and final LLM generation.
    """
    metadata = metadata or {}
    turn_history = turn_history or [] # Should be populated by LollmsClient

    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: # Final answer chunks
        ASCIIColors.success(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        step_type = metadata.get("type", "step")
        hop = metadata.get("hop", "")
        info = metadata.get("query", chunk) if step_type == "rag_query_generation" or step_type == "rag_retrieval" else chunk
        ASCIIColors.yellow(f"\n>> RAG Step Start (Hop {hop}): {step_type} - Info: {str(info)[:100]}...", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        step_type = metadata.get("type", "step")
        hop = metadata.get("hop", "")
        num_chunks = metadata.get("num_chunks", "")
        query = metadata.get("query", "")
        decision = metadata.get("decision", "")
        
        info_str = ""
        if step_type == "rag_query_generation" and query: info_str = f"Generated Query: {query}"
        elif step_type == "rag_retrieval": info_str = f"Retrieved {num_chunks} chunks"
        elif step_type == "rag_llm_decision": info_str = f"LLM Decision: {json.dumps(decision)}"
        elif step_type == "final_answer_generation": info_str = "Final answer generation complete."
        else: info_str = chunk

        ASCIIColors.green(f"\n<< RAG Step End (Hop {hop}): {step_type} - {info_str}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nError in RAG stream: {chunk}", flush=True)
    
    # You can inspect turn_history here if needed:
    # ASCIIColors.debug(f"Current RAG Turn History: {turn_history}")
    return True

# --- Main Example ---
if __name__ == "__main__":
    ASCIIColors.red("--- Multi-Hop RAG Example with LollmsClient ---")

    # LLM Configuration (use a model good at instruction following and JSON)
    # Ensure your Ollama server is running and has this model pulled.
    LLM_BINDING_NAME = "ollama"
    LLM_MODEL_NAME = "qwen3:4b" # or llama3, phi3 etc.
    # LLM_MODEL_NAME = "qwen2:1.5b" # Smaller model for quicker tests, but might struggle with complex JSON

    try:
        lc = LollmsClient(
            binding_name=LLM_BINDING_NAME,
            model_name=LLM_MODEL_NAME,
            temperature=0.1, # Default temp for final answer if not overridden
            # Other LollmsClient params as needed
        )
        ASCIIColors.green(f"LollmsClient initialized with LLM: {LLM_BINDING_NAME}/{LLM_MODEL_NAME}")

        # --- Test Case 1: Classic RAG (max_rag_hops = 0) ---
        ASCIIColors.cyan("\n\n--- Test Case 1: Classic RAG (max_rag_hops = 0) ---")
        classic_rag_prompt = "What are the key features of Python?"
        ASCIIColors.blue(f"User Prompt: {classic_rag_prompt}")

        classic_rag_result = lc.generate_text_with_rag(
            prompt=classic_rag_prompt,
            rag_query_function=mock_rag_query_function,
            # rag_query_text=None, # Will use `prompt` for query
            max_rag_hops=0,
            rag_top_k=2, # Get 2 best chunks
            rag_min_similarity_percent=50.0,
            streaming_callback=rag_streaming_callback,
            n_predict=1024 # Max tokens for final answer
        )
        print("\n--- End of Classic RAG ---")
        ASCIIColors.magenta("\nClassic RAG Final Output:")
        print(json.dumps(classic_rag_result, indent=2))


        # --- Test Case 2: Multi-Hop RAG (max_rag_hops = 1) ---
        ASCIIColors.cyan("\n\n--- Test Case 2: Multi-Hop RAG (max_rag_hops = 1) ---")
        multihop_prompt_1 = "Compare Python and JavaScript for web development based on their common applications and core technologies."
        ASCIIColors.blue(f"User Prompt: {multihop_prompt_1}")

        multihop_rag_result_1 = lc.generate_text_with_rag(
            prompt=multihop_prompt_1,
            rag_query_function=mock_rag_query_function,
            # rag_query_text="Python web development applications", # Optional: provide an initial query
            max_rag_hops=1, # Allow one hop for LLM to refine search or decide
            rag_top_k=2,
            rag_min_similarity_percent=60.0,
            streaming_callback=rag_streaming_callback,
            n_predict=1024,
            rag_hop_query_generation_temperature=0.1, # Focused query gen
        )
        print("\n--- End of Multi-Hop RAG (1 hop) ---")
        ASCIIColors.magenta("\nMulti-Hop RAG (1 hop) Final Output:")
        print(json.dumps(multihop_rag_result_1, indent=2))
        

        # --- Test Case 3: Multi-Hop RAG (max_rag_hops = 2) - LLM might decide it has enough earlier ---
        ASCIIColors.cyan("\n\n--- Test Case 3: Multi-Hop RAG (max_rag_hops = 2) ---")
        multihop_prompt_2 = "Explain Retrieval Augmented Generation (RAG) and its relation to Machine Learning."
        ASCIIColors.blue(f"User Prompt: {multihop_prompt_2}")

        multihop_rag_result_2 = lc.generate_text_with_rag(
            prompt=multihop_prompt_2,
            rag_query_function=mock_rag_query_function,
            max_rag_hops=2, # Allow up to two refinement hops
            rag_top_k=1, # Get only the best chunk per hop to force more specific queries
            rag_min_similarity_percent=50.0,
            streaming_callback=rag_streaming_callback,
            n_predict=300
        )
        print("\n--- End of Multi-Hop RAG (up to 2 hops) ---")
        ASCIIColors.magenta("\nMulti-Hop RAG (up to 2 hops) Final Output:")
        print(json.dumps(multihop_rag_result_2, indent=2))


    except ValueError as ve:
        ASCIIColors.error(f"Initialization or RAG parameter error: {ve}")
        trace_exception(ve)
    except ConnectionRefusedError:
        ASCIIColors.error(f"Connection refused. Is the Ollama server ({LLM_BINDING_NAME}) running?")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)

    ASCIIColors.red("\n--- Multi-Hop RAG Example Finished ---")