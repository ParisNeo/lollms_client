from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Dict, Any, Optional, Callable
import json
from pathlib import Path

# --- Internet Search RAG Implementation ---
_duckduckgo_search_installed = False
_search_installation_error_message = ""
try:
    import pipmaster as pm
    # ensure_packages should be called by the binding init ideally,
    # but we call it here for the example's standalone execution.
    pm.ensure_packages(["duckduckgo_search"]) 
    from duckduckgo_search import DDGS
    _duckduckgo_search_installed = True
except Exception as e:
    _search_installation_error_message = str(e)
    DDGS = None
    ASCIIColors.error(f"Failed to import duckduckgo_search: {_search_installation_error_message}")
    ASCIIColors.info("Please install it: pip install duckduckgo-search")


def perform_internet_search_rag(
    query_text: str,
    vectorizer_name: Optional[str] = None, # Not used for search
    top_k: int = 5,
    min_similarity_percent: float = 0.0 # Not used directly for search filter, but can influence result quality/rank
) -> List[Dict[str, Any]]:
    """
    Performs an internet search using DuckDuckGo and formats results for RAG.
    Similarity is simulated based on rank.
    """
    if not _duckduckgo_search_installed or DDGS is None:
         ASCIIColors.error("duckduckgo_search is not available. Cannot perform internet search.")
         return []

    ASCIIColors.magenta(f"  [INTERNET SEARCH] Querying DuckDuckGo for: '{query_text}', max_results={top_k}")
    search_results_raw = []
    try:
        # DDGS().text returns a generator, max_results limits it.
        # Note: The DDGS library might sometimes return fewer results than max_results.
        with DDGS() as ddgs:
            search_results_raw = list(ddgs.text(keywords=query_text, max_results=top_k))
            
    except Exception as e:
        ASCIIColors.error(f"  [INTERNET SEARCH] Search failed: {e}")
        trace_exception(e)
        return []

    formatted_results: List[Dict[str, Any]] = []
    if search_results_raw:
        for i, r in enumerate(search_results_raw):
            # Simulate similarity based on rank (rank 1 is highest sim)
            # Max similarity is 100% for rank 1, decreases linearly or non-linearly.
            # Simple linear decrease: 100 - (rank * (100 / top_k+1))
            # Let's use rank-based score: 100% for rank 1, 90% for rank 2, ... 50% for rank 5 etc.
            # Ensure similarity is above min_similarity_percent if that param was intended as a filter here
            
            simulated_similarity = max(0.0, 100.0 - i * (100.0 / (top_k + 1))) # Higher rank = lower sim
            simulated_similarity = round(simulated_similarity, 2)

            if simulated_similarity >= min_similarity_percent:
                formatted_results.append({
                    "file_path": r.get("href", "# Unknown URL"), # Use URL as document identifier
                    "chunk_text": f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}", # Combine title and snippet
                    "similarity_percent": simulated_similarity,
                })
            else:
                 ASCIIColors.debug(f"  [INTERNET SEARCH] Skipping result {i+1} due to low simulated similarity ({simulated_similarity}%)")

    ASCIIColors.magenta(f"  [INTERNET SEARCH] Formatted {len(formatted_results)} results for RAG.")
    if not formatted_results: ASCIIColors.yellow(f"  [INTERNET SEARCH] No results found for query: '{query_text}' or none met min_similarity_percent.")
    return formatted_results

# --- Streaming Callback for RAG and LLM ---
def rag_streaming_callback(
    chunk: str, 
    msg_type: MSG_TYPE, 
    metadata: Optional[Dict] = None, 
    turn_history: Optional[List] = None
) -> bool:
    metadata = metadata or {}
    hop = metadata.get("hop", "")
    type_info = metadata.get("type", "N/A")

    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: # Final answer chunks
        ASCIIColors.success(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        info = metadata.get("query", chunk) if type_info in ["rag_query_generation", "rag_retrieval"] else chunk
        ASCIIColors.yellow(f"\n>> RAG Hop {hop} | START | {type_info.upper()} | Info: {str(info)[:100]}...", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        num_chunks = metadata.get("num_chunks")
        query = metadata.get("query")
        decision = metadata.get("decision")
        
        end_info = []
        if query: end_info.append(f"Query: '{str(query)[:50]}...'")
        if num_chunks is not None: end_info.append(f"Results: {num_chunks}")
        if decision: end_info.append(f"LLM Decision: NeedMore={decision.get('need_more_data')}, Summary: '{str(decision.get('new_information_summary'))[:50]}...'")
        
        ASCIIColors.green(f"\n<< RAG Hop {hop} | END   | {type_info.upper()} | {' | '.join(end_info) if end_info else chunk}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nError in RAG stream: {chunk}", flush=True)
    
    return True

# --- Main Example ---
if __name__ == "__main__":
    ASCIIColors.red("--- Multi-Hop Internet Search Example with LollmsClient ---")

    # LLM Configuration (use a model good at instruction following and JSON)
    # Ensure your Ollama server is running and has this model pulled.
    LLM_BINDING_NAME = "ollama"
    LLM_MODEL_NAME = "mistral:latest" # or llama3, phi3 etc.

    # You could also enable the internet_search tool via MCP,
    # but this example specifically uses it directly via generate_text_with_rag.
    # For MCP example, see examples/local_mcp.py

    try:
        lc = LollmsClient(
            binding_name=LLM_BINDING_NAME,
            model_name=LLM_MODEL_NAME,
            temperature=0.1,
            ctx_size=4096 
        )
        ASCIIColors.green(f"LollmsClient initialized with LLM: {LLM_BINDING_NAME}/{LLM_MODEL_NAME}")

        if not _duckduckgo_search_installed or DDGS is None:
             ASCIIColors.error("duckduckgo_search is not installed. Cannot run search examples.")
             exit()


        # --- Test Case 1: Classic Search RAG (max_rag_hops = 0) ---
        ASCIIColors.cyan("\n\n--- Test Case 1: Classic Internet Search RAG (max_rag_hops = 0) ---")
        classic_search_prompt = "What is the current population of Japan?"
        ASCIIColors.blue(f"User Prompt: {classic_search_prompt}")

        classic_rag_result = lc.generate_text_with_rag(
            prompt=classic_search_prompt,
            rag_query_function=perform_internet_search_rag, # Use the search function
            max_rag_hops=0,
            rag_top_k=3, # Get 3 search results
            rag_min_similarity_percent=50.0, # Only use results with simulated sim >= 50%
            streaming_callback=rag_streaming_callback,
            n_predict=250
        )
        print("\n--- End of Classic Search RAG ---")
        ASCIIColors.magenta("\nClassic Search RAG Final Output Structure:")
        print(f"  Final Answer (first 100 chars): {classic_rag_result.get('final_answer', '')}...")
        print(f"  Error: {classic_rag_result.get('error')}")
        print(f"  Number of Hops: {len(classic_rag_result.get('rag_hops_history', []))}")
        print(f"  Total Unique Sources Retrieved: {len(classic_rag_result.get('all_retrieved_sources', []))}")
        if classic_rag_result.get('all_retrieved_sources'):
            print("  Example Retrieved Source:")
            source_ex = classic_rag_result['all_retrieved_sources'][0]
            print(f"    Document (URL): {source_ex.get('document')}")
            print(f"    Similarity: {source_ex.get('similarity')}%")
            print(f"    Content (Snippet, first 50 chars): {source_ex.get('content', '')}...")


        # --- Test Case 2: Multi-Hop Search RAG (max_rag_hops = 1) ---
        ASCIIColors.cyan("\n\n--- Test Case 2: Multi-Hop Internet Search RAG (max_rag_hops = 1) ---")
        multihop_search_prompt_1 = "Tell me about the latest developments in fusion energy, including any recent news."
        ASCIIColors.blue(f"User Prompt: {multihop_search_prompt_1}")

        multihop_rag_result_1 = lc.generate_text_with_rag(
            prompt=multihop_search_prompt_1,
            rag_query_function=perform_internet_search_rag,
            rag_query_text=None, # LLM will generate first query
            max_rag_hops=1, # Allow one refinement hop
            rag_top_k=2, # Get 2 search results per query
            rag_min_similarity_percent=50.0,
            streaming_callback=rag_streaming_callback,
            n_predict=400,
            rag_hop_query_generation_temperature=0.1
        )
        print("\n--- End of Multi-Hop Search RAG (1 hop max) ---")
        ASCIIColors.magenta("\nMulti-Hop Search RAG (1 hop max) Final Output Structure:")
        print(f"  Final Answer (first 100 chars): {multihop_rag_result_1.get('final_answer', '')}...")
        print(f"  Error: {multihop_rag_result_1.get('error')}")
        print(f"  Number of Hops Made: {len(multihop_rag_result_1.get('rag_hops_history', []))}")
        for i, hop_info in enumerate(multihop_rag_result_1.get('rag_hops_history', [])):
            print(f"    Hop {i+1} Query: '{hop_info.get('query')}'")
            print(f"    Hop {i+1} Results Count: {len(hop_info.get('retrieved_chunks_details',[]))}")
            print(f"    Hop {i+1} Summary (first 50): '{str(hop_info.get('new_information_summary'))[:50]}...'")
            print(f"    Hop {i+1} LLM Decision: NeedMoreData={hop_info.get('llm_decision_json',{}).get('need_more_data')}")
        print(f"  Total Unique Sources Retrieved: {len(multihop_rag_result_1.get('all_retrieved_sources', []))}")


        # --- Test Case 3: More complex multi-hop (max_rag_hops = 2) ---
        ASCIIColors.cyan("\n\n--- Test Case 3: More Complex Multi-Hop Internet Search RAG (max_rag_hops = 2) ---")
        multihop_search_prompt_2 = "What are the requirements and steps to install the lollms_client python library, and what are some of its key features?"
        ASCIIColors.blue(f"User Prompt: {multihop_search_prompt_2}")

        multihop_rag_result_2 = lc.generate_text_with_rag(
            prompt=multihop_search_prompt_2,
            rag_query_function=perform_internet_search_rag,
            max_rag_hops=2, # Allow up to two refinement hops
            rag_top_k=2, # Get 2 results per query
            rag_min_similarity_percent=40.0, # Lower similarity to maybe get broader initial results
            streaming_callback=rag_streaming_callback,
            n_predict=500 # Allow more for the installation steps and features
        )
        print("\n--- End of More Complex Multi-Hop Search RAG (up to 2 hops) ---")
        ASCIIColors.magenta("\nMore Complex Multi-Hop Search RAG (up to 2 hops) Final Output Structure:")
        print(f"  Final Answer (first 100 chars): {multihop_rag_result_2.get('final_answer', '')[:100]}...")
        print(f"  Error: {multihop_rag_result_2.get('error')}")
        print(f"  Number of Hops Made: {len(multihop_rag_result_2.get('rag_hops_history', []))}")
        for i, hop_info in enumerate(multihop_rag_result_2.get('rag_hops_history', [])):
            print(f"    Hop {i+1} Query: '{hop_info.get('query')}'")
            print(f"    Hop {i+1} Results Count: {len(hop_info.get('retrieved_chunks_details',[]))}")
            print(f"    Hop {i+1} Summary (first 50): '{str(hop_info.get('new_information_summary'))[:50]}...'")
        print(f"  Total Unique Sources Retrieved: {len(multihop_rag_result_2.get('all_retrieved_sources', []))}")


    except ValueError as ve:
        ASCIIColors.error(f"Initialization or RAG parameter error: {ve}")
        trace_exception(ve)
    except ConnectionRefusedError:
        ASCIIColors.error(f"Connection refused. Is the Ollama server ({LLM_BINDING_NAME}) running?")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)

    ASCIIColors.red("\n--- Multi-Hop Internet Search Example Finished ---")