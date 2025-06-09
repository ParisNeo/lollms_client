# Conceptual: lollms_client/mcp_bindings/remote_mcp/__init__.py

import asyncio
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any, Tuple
from lollms_client.lollms_mcp_binding import LollmsMCPBinding
from ascii_colors import ASCIIColors, trace_exception
import threading
try:
    from mcp import ClientSession, types
    # Import the specific network client from MCP SDK
    from mcp.client.streamable_http import streamablehttp_client
    # If supporting OAuth, you'd import auth components:
    # from mcp.client.auth import OAuthClientProvider, TokenStorage
    # from mcp.shared.auth import OAuthClientMetadata, OAuthToken
    MCP_LIBRARY_AVAILABLE = True
except ImportError:
    # ... (error handling as in StandardMCPBinding) ...
    MCP_LIBRARY_AVAILABLE = False
    ClientSession = None # etc.
    streamablehttp_client = None


BindingName = "RemoteMCPBinding"
# No TOOL_NAME_SEPARATOR needed if connecting to one remote server per instance,
# or if server aliases are handled differently (e.g. part of URL or config)
TOOL_NAME_SEPARATOR  = "::"

class RemoteMCPBinding(LollmsMCPBinding):
    def __init__(self,
                 server_url: str, # e.g., "http://localhost:8000/mcp"
                 alias: str = "remote_server", # An alias for this connection
                 auth_config: Optional[Dict[str, Any]] = None, # For API keys, OAuth, etc.
                 **other_config_params: Any):
        super().__init__(binding_name="remote_mcp")
        
        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: MCP library not available.")
            return

        if not server_url:
            ASCIIColors.error(f"{self.binding_name}: server_url is required.")
            # Or raise ValueError
            return

        self.server_url = server_url
        self.alias = alias # Could be used to prefix tool names if managing multiple remotes
        self.auth_config = auth_config or {}
        self.config = {
            "server_url": server_url,
            "alias": alias,
            "auth_config": self.auth_config
        }
        self.config.update(other_config_params)

        self._mcp_session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._discovered_tools_cache: List[Dict[str, Any]] = []
        self._is_initialized = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._start_event_loop_thread() # Similar to StandardMCPBinding

    def _start_event_loop_thread(self): # Simplified from StandardMCPBinding
        if self._loop and self._loop.is_running(): return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop_forever, daemon=True)
        self._thread.start()

    def _run_loop_forever(self):
        if not self._loop: return
        asyncio.set_event_loop(self._loop)
        try: self._loop.run_forever()
        finally:
            # ... (loop cleanup as in StandardMCPBinding) ...
            if not self._loop.is_closed(): self._loop.close()

    def _run_async(self, coro, timeout=None): # Simplified
        if not self._loop or not self._loop.is_running(): raise RuntimeError("Event loop not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    async def _initialize_connection_async(self) -> bool:
        if self._is_initialized: return True
        ASCIIColors.info(f"{self.binding_name}: Initializing connection to {self.server_url}...")
        try:
            self._exit_stack = AsyncExitStack()
            
            # --- Authentication Setup (Conceptual) ---
            # oauth_provider = None
            # if self.auth_config.get("type") == "oauth":
            #     # oauth_provider = OAuthClientProvider(...) # Setup based on auth_config
            #     pass
            # http_headers = {}
            # if self.auth_config.get("type") == "api_key":
            #     key = self.auth_config.get("key")
            #     header_name = self.auth_config.get("header_name", "X-API-Key")
            #     if key: http_headers[header_name] = key
            
            # Use streamablehttp_client from MCP SDK
            # The `auth` parameter of streamablehttp_client takes an OAuthClientProvider
            # For simple API key headers, you might need to use `httpx` directly
            # or see if streamablehttp_client allows passing custom headers.
            # The MCP client example for streamable HTTP doesn't show custom headers directly,
            # it focuses on OAuth.
            # If `streamablehttp_client` takes `**kwargs` that are passed to `httpx.AsyncClient`,
            # then `headers=http_headers` might work.
            
            # Assuming streamablehttp_client can take headers if needed, or auth provider
            # For now, let's assume no auth for simplicity or that it's handled by underlying httpx if passed via kwargs
            client_streams = await self._exit_stack.enter_async_context(
                streamablehttp_client(self.server_url) # Add auth=oauth_provider or headers=http_headers if supported
            )
            read_stream, write_stream, _http_client_instance = client_streams # http_client_instance might be useful

            self._mcp_session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._mcp_session.initialize()
            self._is_initialized = True
            ASCIIColors.green(f"{self.binding_name}: Connected to {self.server_url}")
            await self._refresh_tools_cache_async()
            return True
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Failed to connect to {self.server_url}: {e}")
            if self._exit_stack: await self._exit_stack.aclose() # Cleanup on failure
            self._exit_stack = None
            self._mcp_session = None
            self._is_initialized = False
            return False

    def _ensure_initialized_sync(self, timeout=30.0):
        if not self._is_initialized:
            success = self._run_async(self._initialize_connection_async(), timeout=timeout)
            if not success: raise ConnectionError(f"Failed to initialize remote MCP connection to {self.server_url}")
        if not self._mcp_session: # Double check
             raise ConnectionError(f"MCP Session not valid after init attempt for {self.server_url}")


    async def _refresh_tools_cache_async(self):
        if not self._is_initialized or not self._mcp_session: return
        ASCIIColors.info(f"{self.binding_name}: Refreshing tools from {self.server_url}...")
        try:
            list_tools_result = await self._mcp_session.list_tools()
            current_tools = []
            # ... (tool parsing logic similar to StandardMCPBinding, but no server alias prefix needed if one server per binding instance)
            for tool_obj in list_tools_result.tools:
                # ...
                input_schema_dict = {}
                tool_input_schema = getattr(tool_obj, 'inputSchema', getattr(tool_obj, 'input_schema', None))
                if tool_input_schema:
                    if hasattr(tool_input_schema, 'model_dump'):
                        input_schema_dict = tool_input_schema.model_dump(mode='json', exclude_none=True)
                    elif isinstance(tool_input_schema, dict):
                        input_schema_dict = tool_input_schema
                
                tool_name_for_client = f"{self.alias}{TOOL_NAME_SEPARATOR}{tool_obj.name}" if TOOL_NAME_SEPARATOR else tool_obj.name

                current_tools.append({
                    "name": tool_name_for_client, # Use self.alias to prefix
                    "description": tool_obj.description or "",
                    "input_schema": input_schema_dict
                })
            self._discovered_tools_cache = current_tools
            ASCIIColors.green(f"{self.binding_name}: Tools refreshed for {self.server_url}. Found {len(current_tools)} tools.")
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Error refreshing tools from {self.server_url}: {e}")

    def discover_tools(self, force_refresh: bool = False, timeout_per_server: float = 10.0, **kwargs) -> List[Dict[str, Any]]:
        # This binding instance connects to ONE server, so timeout_per_server is just 'timeout'
        try:
            self._ensure_initialized_sync(timeout=timeout_per_server)
            if force_refresh or not self._discovered_tools_cache:
                self._run_async(self._refresh_tools_cache_async(), timeout=timeout_per_server)
            return self._discovered_tools_cache
        except Exception as e:
            ASCIIColors.error(f"{self.binding_name}: Problem during tool discovery for {self.server_url}: {e}")
            return []
            
    async def _execute_tool_async(self, actual_tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_initialized or not self._mcp_session:
            return {"error": f"Not connected to {self.server_url}", "status_code": 503}
        
        ASCIIColors.info(f"{self.binding_name}: Executing remote tool '{actual_tool_name}' on {self.server_url} with params: {json.dumps(params)}")
        try:
            mcp_call_result = await self._mcp_session.call_tool(name=actual_tool_name, arguments=params)
            # ... (result parsing as in StandardMCPBinding) ...
            output_parts = [p.text for p in mcp_call_result.content if isinstance(p, types.TextContent) and p.text is not None] if mcp_call_result.content else []
            if not output_parts: return {"output": {"message": "Tool executed but returned no textual content."}, "status_code": 200}
            combined_output_str = "\n".join(output_parts)
            try: return {"output": json.loads(combined_output_str), "status_code": 200}
            except json.JSONDecodeError: return {"output": combined_output_str, "status_code": 200}
        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing remote tool '{actual_tool_name}': {str(e)}", "status_code": 500}


    def execute_tool(self, tool_name_with_alias: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        timeout = float(kwargs.get('timeout', 60.0))
        
        # If using alias prefixing (self.alias + TOOL_NAME_SEPARATOR + actual_name)
        expected_prefix = f"{self.alias}{TOOL_NAME_SEPARATOR}"
        if TOOL_NAME_SEPARATOR and tool_name_with_alias.startswith(expected_prefix):
            actual_tool_name = tool_name_with_alias[len(expected_prefix):]
        elif not TOOL_NAME_SEPARATOR and tool_name_with_alias: # No prefixing, tool_name is actual_tool_name
             actual_tool_name = tool_name_with_alias
        else:
            return {"error": f"Tool name '{tool_name_with_alias}' does not match expected alias '{self.alias}'.", "status_code": 400}

        try:
            self._ensure_initialized_sync(timeout=min(timeout, 30.0))
            return self._run_async(self._execute_tool_async(actual_tool_name, params), timeout=timeout)
        # ... (error handling as in StandardMCPBinding) ...
        except ConnectionError as e: return {"error": f"{self.binding_name}: Connection issue for '{self.server_url}': {e}", "status_code": 503}
        except TimeoutError: return {"error": f"{self.binding_name}: Remote tool '{actual_tool_name}' on '{self.server_url}' timed out.", "status_code": 504}
        except Exception as e:
            trace_exception(e)
            return {"error": f"{self.binding_name}: Failed to run remote MCP tool '{actual_tool_name}': {e}", "status_code": 500}

    def close(self):
        ASCIIColors.info(f"{self.binding_name}: Closing connection to {self.server_url}...")
        if self._exit_stack:
            try:
                # The anyio task error might also occur here if not careful
                self._run_async(self._exit_stack.aclose(), timeout=10.0)
            except Exception as e:
                ASCIIColors.error(f"{self.binding_name}: Error during async close for {self.server_url}: {e}")
        self._exit_stack = None
        self._mcp_session = None
        self._is_initialized = False
        
        # Stop event loop thread
        if self._loop and self._loop.is_running(): self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=5.0)
        ASCIIColors.info(f"{self.binding_name}: Remote connection binding closed.")

    def get_binding_config(self) -> Dict[str, Any]: # LollmsMCPBinding might expect this
        return self.config