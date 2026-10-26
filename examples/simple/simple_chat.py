import asyncio
from pathlib import Path
from ascii_colors import ASCIIColors

from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client.lollms_core import LollmsBindingProfile, LollmsModelProfile

def main():
    # --- Profile-based client with Tools Modality ---
    client = LollmsClient(
        # LLM (text generation)
        llm_binding_profiles={
            "local_ollama": LollmsBindingProfile(
                name="local_ollama",
                binding_name="ollama",
                binding_config={"host_address": "http://localhost:11434"}
            )
        },
        llm_model_profiles={
            "fast-chat": LollmsModelProfile(
                name="fast-chat",
                binding_profile_name="local_ollama",
                model_name="llama3.1:8b",
                is_default=True
            )
        },
        
        # Tools (MCP/LCP) — the 9th modality
        tools_binding_profiles={
            "mcp_main": LollmsBindingProfile(
                name="mcp_main",
                binding_name="mcp",
                binding_config={"host": "127.0.0.1", "port": 12345}
            ),
            "lcp_local": LollmsBindingProfile(
                name="lcp_local",
                binding_name="lcp",
                binding_config={"workspace": str(Path.cwd())}
            )
        },
        tools_model_profiles={
            "web-tools": LollmsModelProfile(
                name="web-tools",
                binding_profile_name="mcp_main",
                model_name="web-search",
                is_default=True
            ),
            "fs-tools": LollmsModelProfile(
                name="fs-tools",
                binding_profile_name="lcp_local",
                model_name="filesystem"
            )
        }
    )
    
    # Switch to a specific toolset (lazy instantiation)
    print("\n--- Switching to filesystem tools ---")
    client.switch_tools("fs-tools")
    
    # List available tools from the active toolset
    if client.tools:
        tools = client.tools.list_tools()
        print(f"Available tools: {[t.get('name') for t in tools]}")
    
    # Generate with tools (uses the active toolset)
    print("\n--- Chat with tools ---")
    discussion = LollmsDiscussion(client)
    response = discussion.chat(
        "List the files in the current directory using your tools.",
        enable_tools=True
    )
    print(response)
    
    # Switch back to web tools
    print("\n--- Switching back to web tools ---")
    client.switch_tools("web-tools")
    
    # Direct tool execution example
    result = client.generate_with_tools(
        prompt="Search for 'lollms' on the web and summarize the first result.",
        tools=[]  # Uses client.tools.to_chat_tool_specs() when empty
    )
    print(result["response"])

if __name__ == "__main__":
    main()