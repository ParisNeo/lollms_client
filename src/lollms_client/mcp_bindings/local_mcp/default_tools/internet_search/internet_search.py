from typing import Dict, Any, List

# --- Package Management ---
_duckduckgo_search_installed = False
_installation_error_message = ""
try:
    import pipmaster as pm
    pm.ensure_packages(["duckduckgo_search"])
    from duckduckgo_search import DDGS
    _duckduckgo_search_installed = True
except Exception as e:
    _installation_error_message = str(e)
    DDGS = None # Ensure DDGS is None if import fails
# --- End Package Management ---

def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs an internet search using DuckDuckGo.
    """
    if not _duckduckgo_search_installed:
        return {
            "search_results": [],
            "error": f"Required library 'duckduckgo_search' is not installed or import failed: {_installation_error_message}"
        }

    query = params.get("query")
    num_results = params.get("num_results", 5)

    if not query:
        return {"search_results": [], "error": "'query' parameter is required."}
    if not isinstance(num_results, int) or num_results <= 0:
        num_results = 5 # Default to 5 if invalid

    try:
        search_results_formatted: List[Dict[str, str]] = []
        # DDGS().text returns a generator, max_results limits it.
        # Note: The DDGS library might sometimes return fewer results than max_results.
        with DDGS() as ddgs:
            results = ddgs.text(keywords=query, max_results=num_results)
            if results:
                for r in results:
                    search_results_formatted.append({
                        "title": r.get("title", "N/A"),
                        "link": r.get("href", "#"),
                        "snippet": r.get("body", "N/A")
                    })
            
        if not search_results_formatted and results is None: # Check if ddgs.text itself returned None
             return {"search_results": [], "error": "Search returned no results or failed to connect."}


        return {"search_results": search_results_formatted, "error": None}

    except Exception as e:
        # Log the exception for server-side debugging if possible
        # For now, just return it in the error field.
        print(f"Error during internet search: {str(e)}") # Basic logging
        # from ascii_colors import trace_exception (if you want to import it here)
        # trace_exception(e)
        return {"search_results": [], "error": f"An unexpected error occurred during search: {str(e)}"}

if __name__ == '__main__':
    import json
    print("--- Internet Search Tool Test ---")

    if not _duckduckgo_search_installed:
        print(f"Cannot run test: duckduckgo_search not installed. Error: {_installation_error_message}")
    else:
        # Test 1: Simple search
        params1 = {"query": "What is the capital of France?", "num_results": 3}
        result1 = execute(params1)
        print(f"\nTest 1 Result (Query: '{params1['query']}'):\n{json.dumps(result1, indent=2)}")
        if result1.get("search_results"):
            assert len(result1["search_results"]) <= 3
            assert "error" not in result1 or result1["error"] is None
        else:
            print(f"Warning: Test 1 might have failed to retrieve results. Error: {result1.get('error')}")


        # Test 2: Search with more results
        params2 = {"query": "Latest AI breakthroughs", "num_results": 6}
        result2 = execute(params2)
        print(f"\nTest 2 Result (Query: '{params2['query']}'):\n{json.dumps(result2, indent=2)}")
        if result2.get("search_results"):
            assert len(result2["search_results"]) <= 6
        else:
            print(f"Warning: Test 2 might have failed to retrieve results. Error: {result2.get('error')}")


        # Test 3: No query
        params3 = {}
        result3 = execute(params3)
        print(f"\nTest 3 Result (No query):\n{json.dumps(result3, indent=2)}")
        assert result3["error"] is not None
        assert "query' parameter is required" in result3["error"]

        # Test 4: Invalid num_results
        params4 = {"query": "python programming", "num_results": -1}
        result4 = execute(params4) # Should default to 5
        print(f"\nTest 4 Result (Invalid num_results):\n{json.dumps(result4, indent=2)}")
        if result4.get("search_results"):
             assert len(result4["search_results"]) <= 5
        else:
            print(f"Warning: Test 4 might have failed to retrieve results. Error: {result4.get('error')}")


    print("\n--- Tests Finished ---")