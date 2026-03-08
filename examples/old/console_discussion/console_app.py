import os
import shutil
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Correctly import all necessary classes from the lollms_client package
from lollms_client import LollmsClient, LollmsDataManager, LollmsDiscussion, LollmsPersonality
from ascii_colors import ASCIIColors, trace_exception

# --- Configuration ---
MAX_CONTEXT_SIZE_FOR_TEST = 2048 # Increased for agentic turns

# Database and workspace configuration
WORKSPACE_DIR = Path("./test_workspace_agentic")
DATABASE_PATH = f"sqlite:///{WORKSPACE_DIR / 'test_discussion_agentic.db'}"
DISCUSSION_ID = "console-agentic-test-1" # Use a fixed ID for easy resumption

# --- MOCK KNOWLEDGE BASE for RAG ---
MOCK_KNOWLEDGE_BASE = {
    "python_basics.md": [
        {"chunk_id": 1, "text": "Python is a high-level, interpreted programming language known for its readability. It was created by Guido van Rossum and released in 1991."},
        {"chunk_id": 2, "text": "Key features of Python include dynamic typing, garbage collection, and a large standard library. It supports procedural, object-oriented, and functional programming."},
    ],
    "javascript_info.js": [
        {"chunk_id": 1, "text": "JavaScript is a scripting language for front-end web development. It is also used in back-end development (Node.js)."},
        {"chunk_id": 2, "text": "Popular JavaScript frameworks include React, Angular, and Vue.js."},
    ],
    "ai_concepts.txt": [
        {"chunk_id": 1, "text": "Retrieval Augmented Generation (RAG) is an AI framework for improving LLM responses by grounding the model on external knowledge sources."},
    ]
}

# --- Dummy MCP Server Scripts ---
TIME_SERVER_PY = """
import asyncio
from datetime import datetime
from mcp.server.fastmcp import FastMCP
mcp_server = FastMCP("TimeMCP", description="A server that provides the current time.", host="localhost",
            port=9624,
            log_level="DEBUG")
@mcp_server.tool()
def get_current_time(user_id: str = "unknown"):
    return {"time": datetime.now().isoformat(), "user_id": user_id}
if __name__ == "__main__": mcp_server.run(transport="streamable-http")
"""
CALCULATOR_SERVER_PY = """
import asyncio
from typing import List, Union
from mcp.server.fastmcp import FastMCP
mcp_server = FastMCP("TimeMCP", description="A server that provides the current time.", host="localhost",
            port=9625,
            log_level="DEBUG")
@mcp_server.tool()
def add_numbers(numbers: List[Union[int, float]]):
    if not isinstance(numbers, list): return {"error": "Input must be a list"}
    return {"sum": sum(numbers)}
if __name__ == "__main__": mcp_server.run(transport="streamable-http")
"""

# --- RAG Mock Function ---
def mock_rag_query_function(query_text: str, top_k: int = 3, **kwargs) -> List[Dict[str, Any]]:
    ASCIIColors.magenta(f"\n  [MOCK RAG] Querying knowledge base for: '{query_text}'")
    results = []
    query_lower = query_text.lower()
    for file_path, chunks in MOCK_KNOWLEDGE_BASE.items():
        for chunk in chunks:
            if any(word in chunk["text"].lower() for word in query_lower.split() if len(word) > 2):
                results.append({"file_path": file_path, "chunk_text": chunk["text"]})
    ASCIIColors.magenta(f"  [MOCK RAG] Found {len(results[:top_k])} relevant chunks.")
    return results[:top_k]

def start_mcp_servers():
    """Starts the dummy MCP servers in the background."""
    ASCIIColors.yellow("--- Starting background MCP servers ---")
    server_dir = WORKSPACE_DIR / "mcp_servers"
    server_dir.mkdir(exist_ok=True, parents=True)
    
    (server_dir / "time_server.py").write_text(TIME_SERVER_PY)
    (server_dir / "calculator_server.py").write_text(CALCULATOR_SERVER_PY)

    procs = []
    procs.append(subprocess.Popen([sys.executable, str(server_dir / "time_server.py")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    procs.append(subprocess.Popen([sys.executable, str(server_dir / "calculator_server.py")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    ASCIIColors.yellow("--- MCP servers launched ---")
    return procs

def setup_client_and_discussion() -> LollmsDiscussion:
    """Sets up the LollmsClient with MCP, the DB manager, and the discussion."""
    print("--- Setting up Lollms Environment ---")
    WORKSPACE_DIR.mkdir(exist_ok=True)

    mcp_config = {
        "servers_infos": {
            "time_machine": {"server_url": "http://localhost:9624/mcp"},
            "calc_unit": {"server_url": "http://localhost:9625/mcp"},
        }
    }

    try:
        client = LollmsClient(
            "ollama",
            model_name="mistral-nemo:latest",
            mcp_binding_name="remote_mcp",
            mcp_binding_config=mcp_config
        )
    except Exception as e:
        trace_exception(e)
        print("\n---FATAL ERROR---")
        print("Could not initialize LollmsClient. Ensure Ollama is running and mcp is installed.")
        exit()

    print(f"-> Using model: {client.binding.model_name}")
    print(f"-> Using MCP binding: {client.mcp.binding_name}")

    db_manager = LollmsDataManager(db_path=DATABASE_PATH)
    discussion = db_manager.get_discussion(client, DISCUSSION_ID)
    
    if discussion:
        print(f"-> Resuming discussion (ID: {DISCUSSION_ID})")
        discussion.max_context_size = MAX_CONTEXT_SIZE_FOR_TEST
    else:
        print(f"-> Creating new discussion (ID: {DISCUSSION_ID})")
        discussion = LollmsDiscussion.create_new(
            lollms_client=client,
            db_manager=db_manager,
            id=DISCUSSION_ID,
            title="Console Agentic Test",
            max_context_size=MAX_CONTEXT_SIZE_FOR_TEST
        )

    print("--- Setup Complete. Ready to chat! ---\n")
    return discussion

def print_help():
    print("\n--- Commands ---")
    print("!agent <prompt>  - Run a prompt using all available tools (MCP).")
    print("!rag <prompt>    - Run a prompt using the mock knowledge base (RAG).")
    print("!both <prompt>   - Run a prompt using both MCP tools and RAG.")
    print("!status          - Show current discussion state (pruning, message count).")
    print("!regen           - Regenerate the last AI response.")
    print("!exit            - Exit the application.")
    print("----------------\n")

def print_agentic_results(response_dict):
    """Renders a beautiful report of the agent's turn."""
    ai_message = response_dict.get('ai_message')
    if not ai_message:
        return

    ASCIIColors.cyan("\n" + "="*22 + " Agentic Turn Report " + "="*22)

    # --- Final Answer ---
    ASCIIColors.blue("\nFinal Answer:")
    ASCIIColors.green(f"  {ai_message.content}")

    # --- Agent's Internal Monologue (The Scratchpad) ---
    if ai_message.scratchpad:
        ASCIIColors.blue("\nAgent's Reasoning Log (Scratchpad):")
        # Print scratchpad line by line for better color coding
        for line in ai_message.scratchpad.split('\n'):
            if line.startswith("### Step"):
                ASCIIColors.yellow(line)
            elif line.startswith("- **Action**:") or line.startswith("- **Result**:") or line.startswith("- **Error**:") :
                ASCIIColors.magenta(line)
            else:
                print(line)

    # --- Sources Used (from metadata) ---
    if ai_message.metadata and "sources" in ai_message.metadata:
        sources = ai_message.metadata.get("sources", [])
        if sources:
            ASCIIColors.blue("\nSources Consulted (RAG):")
            for i, source in enumerate(sources):
                print(f"  [{i+1}] Path: {source.get('file_path', 'N/A')}")
                # Indent the content for readability
                content = source.get('chunk_text', 'N/A').replace('\n', '\n      ')
                print(f"      Content: \"{content}\"")
    
    ASCIIColors.cyan("\n" + "="*61 + "\n")

def run_chat_console(discussion: LollmsDiscussion):
    print_help()
    while True:
        user_input = input("You: ")
        if not user_input:
            continue

        use_mcps_flag = False
        use_data_store_flag = False
        prompt = user_input

        # --- Command Handling ---
        if user_input.lower().startswith("!exit"):
            break
        elif user_input.lower().startswith("!help"):
            print_help()
            continue
        elif user_input.lower().startswith("!status"):
            # Assuming a print_status function exists
            # print_status(discussion)
            continue
        elif user_input.lower().startswith("!regen"):
            # Assuming a regenerate_branch method exists
            # discussion.regenerate_branch(...)
            continue
        elif user_input.lower().startswith("!agent "):
            use_mcps_flag = True
            prompt = user_input[7:].strip()
            ASCIIColors.yellow(f"Agentic MCP turn initiated for: '{prompt}'")
        elif user_input.lower().startswith("!rag "):
            use_data_store_flag = True
            prompt = user_input[5:].strip()
            ASCIIColors.yellow(f"Agentic RAG turn initiated for: '{prompt}'")
        elif user_input.lower().startswith("!both "):
            use_mcps_flag = True
            use_data_store_flag = True
            prompt = user_input[6:].strip()
            ASCIIColors.yellow(f"Agentic MCP+RAG turn initiated for: '{prompt}'")
        
        # --- Streaming Callback ---
        def stream_callback(chunk, msg_type, metadata={}, **kwargs):
            # Render steps and thoughts in real-time
            if msg_type == 12: # MSG_TYPE.MSG_TYPE_STEP_START
                 ASCIIColors.cyan(f"\n> Starting: {chunk}")
            elif msg_type == 13: # MSG_TYPE.MSG_TYPE_STEP_END
                 ASCIIColors.cyan(f"> Finished: {chunk}")
            elif msg_type == 2: # MSG_TYPE.MSG_TYPE_INFO (for thoughts)
                 ASCIIColors.yellow(f"\n  (Thought): {chunk}")
            else: # Final answer chunks are printed by the main loop
                pass # The final answer is printed after the full report
            return True

        # --- Main Chat Logic ---
        try:
            #print("\nAI: ", end="", flush=True)
            
            response_dict = discussion.chat(
                user_message=prompt,
                use_mcps=use_mcps_flag,
                use_data_store={"coding_store": mock_rag_query_function} if use_data_store_flag else None,
                streaming_callback=stream_callback
            )
            if use_mcps_flag or use_data_store_flag:
                print_agentic_results(response_dict)
            print("\nAI: ", end="")
            ASCIIColors.green(response_dict['ai_message'].content)            

        except Exception as e:
            trace_exception(e)
            print(f"\nAn error occurred during generation.")

if __name__ == "__main__":
    mcp_procs = start_mcp_servers()
    try:
        discussion_session = setup_client_and_discussion()
        run_chat_console(discussion_session)
    finally:
        ASCIIColors.red("\n--- Shutting down MCP servers ---")
        for proc in mcp_procs:
            proc.terminate()
            proc.wait()
        shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
        print("Cleanup complete. Goodbye!")