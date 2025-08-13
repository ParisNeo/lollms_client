import asyncio
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any, Tuple
from lollms_client.lollms_mcp_binding import LollmsMCPBinding
from ascii_colors import ASCIIColors, trace_exception
import threading
import json
import pipmaster as pm

try:
    pm.ensure_packages(["mcp"])
    from mcp import ClientSession, types
    from mcp.client.streamable_http import streamablehttp_client
    MCP_LIBRARY_AVAILABLE = True
except ImportError:
    MCP_LIBRARY_AVAILABLE = False
    ClientSession = None
    streamablehttp_client = None


BindingName = "RemoteMCPBinding"
TOOL_NAME_SEPARATOR  = "::"

class RemoteMCPBinding(LollmsMCPBinding):
    """
    This binding allows the connection to one or more remote MCP servers.
    Tools from all connected servers are aggregated and prefixed with the server's alias.
    """
    def __init__(self,
                 servers_infos: Dict[str, Dict[str, Any]],
                 **kwargs: Any):
        """
        Initializes the binding to connect to multiple MCP servers.

        Args:
            servers_infos (Dict[str, Dict[str, Any]]): A dictionary where each key is a unique
                alias for a server, and the value is another dictionary containing connection
                details for that server.
                Example:
                {
                    "main_server": {"server_url": "http://localhost:8787", "auth_config": {}},
                    "experimental_server": {"server_url": "http://test.server:9000"}
                }
            **kwargs (Any): Additional configuration parameters.
        """
        super().__init__(binding_name="remote_mcp")
        # initialization in case no servers are present
        self.servers = None
        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: MCP library not available. This binding will be disabled.")
            return

        if not servers_infos or not isinstance(servers_infos, dict):
            ASCIIColors.error(f"{self.binding_name}: `servers_infos` dictionary is required and cannot be empty.")
            return

        ### NEW: Store the overall configuration
        self.config = {
            "servers_infos": kwargs.get("servers_infos"),
            **kwargs
        }

        ### NEW: State management for multiple servers.
        # The key is the server alias. The value is a dictionary holding the state for that server.
        self.servers: Dict[str, Dict[str, Any]] = {}
        for alias, info in servers_infos.items():
            if "server_url" not in info:
                ASCIIColors.warning(f"{self.binding_name}: Skipping server '{alias}' due to missing 'server_url'.")
                continue
            
            self.servers[alias] = {
                "url": info["server_url"],
                "auth_config": info.get("auth_config", {}),
                "session": None,        # Will hold the ClientSession
                "exit_stack": None,     # Will hold the AsyncExitStack
                "initialized": False,
                "initializing_lock": threading.Lock() # Prevents race conditions on initialization
            }

        self._discovered_tools_cache: List[Dict[str, Any]] = []

        ### MODIFIED: These are now shared across all connections
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_started_event = threading.Event()

        if self.servers:
            self._start_event_loop_thread()
        else:
            ASCIIColors.warning(f"{self.binding_name}: No valid servers configured.")

    # _start_event_loop_thread, _run_loop_forever, _wait_for_loop, _run_async
    # are utility methods for the shared event loop and do not need to be changed.
    # They manage the async infrastructure for the entire binding instance.
    def _start_event_loop_thread(self):
        if self._loop and self._loop.is_running(): return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop_forever, daemon=True)
        self._thread.start()

    def _run_loop_forever(self):
        if not self._loop: return
        asyncio.set_event_loop(self._loop)
        try:
            self._loop_started_event.set()
            self._loop.run_forever()
        finally:
            if not self._loop.is_closed(): self._loop.close()

    def _wait_for_loop(self, timeout=5.0):
        if not self._loop_started_event.wait(timeout=timeout):
            raise RuntimeError(f"{self.binding_name}: Event loop thread failed to start in time.")
        if not self._loop or not self._loop.is_running():
            raise RuntimeError(f"{self.binding_name}: Event loop is not running after start signal.")

    def _run_async(self, coro, timeout=None):
        if not self._loop or not self._loop.is_running(): 
            raise RuntimeError("Event loop not running. This should have been caught earlier.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def _prepare_headers(self, alias: str) -> Dict[str, str]:
        """Prepares the headers dictionary from the server's auth_config."""
        server_info = self.servers[alias]
        auth_config = server_info.get("auth_config", {})
        headers = {}
        auth_type = auth_config.get("type")
        if auth_type == "api_key":
            api_key = auth_config.get("key")
            header_name = auth_config.get("header_name", "X-API-Key") # Default to X-API-Key
            if api_key:
                headers[header_name] = api_key
                ASCIIColors.info(f"{self.binding_name}: Using API Key authentication for server '{alias}'.")

        elif auth_type == "bearer": # <-- NEW BLOCK
            token = auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
        return headers

    async def _initialize_connection_async(self, alias: str) -> bool:
        server_info = self.servers[alias]
        if server_info["initialized"]:
            return True
            
        server_url = server_info["url"]
        ASCIIColors.info(f"{self.binding_name}: Initializing connection to '{alias}' ({server_url})...")
        try:
            # Prepare authentication headers
            auth_headers = self._prepare_headers(alias)

            exit_stack = AsyncExitStack()
            
            client_streams = await exit_stack.enter_async_context(
                streamablehttp_client(url=server_url, headers=auth_headers) # Pass the headers here
            )
            read_stream, write_stream, _ = client_streams

            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            # Update the state for this specific server
            server_info["session"] = session
            server_info["exit_stack"] = exit_stack
            server_info["initialized"] = True

            ASCIIColors.green(f"{self.binding_name}: Connected to '{alias}' ({server_url})")
            return True
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Failed to connect to '{alias}' ({server_url}): {e}")
            if 'exit_stack' in locals() and exit_stack:
                await exit_stack.aclose()
            
            # Reset state for this server on failure
            server_info["session"] = None
            server_info["exit_stack"] = None
            server_info["initialized"] = False
            return False

    ### MODIFIED: Ensures a specific server is initialized
    def _ensure_initialized_sync(self, alias: str, timeout=30.0):
        self._wait_for_loop() 
        
        server_info = self.servers.get(alias)
        if not server_info:
            raise ValueError(f"Unknown server alias: '{alias}'")

        # Use a lock to prevent multiple threads trying to initialize the same server
        with server_info["initializing_lock"]:
            if not server_info["initialized"]:
                success = self._run_async(self._initialize_connection_async(alias), timeout=timeout)
                if not success:
                    raise ConnectionError(f"Failed to initialize remote MCP connection to '{alias}' ({server_info['url']})")
        
        if not server_info.get("session"):
             raise ConnectionError(f"MCP Session not valid after init attempt for '{alias}' ({server_info['url']})")

    ### MODIFIED: Refreshes tools from ALL connected servers and aggregates them
    async def _refresh_all_tools_cache_async(self):
        ASCIIColors.info(f"{self.binding_name}: Refreshing tools from all servers...")
        all_tools = []
        # Create a list of tasks to run concurrently
        refresh_tasks = [
            self._fetch_tools_from_server_async(alias) for alias in self.servers.keys()
        ]
        
        # Gather results from all tasks
        results = await asyncio.gather(*refresh_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                # Error already logged inside the fetch function
                continue
            if result:
                all_tools.extend(result)
        
        self._discovered_tools_cache = all_tools
        ASCIIColors.green(f"{self.binding_name}: Tool refresh complete. Found {len(all_tools)} tools across all servers.")

    ### NEW: Helper async function to fetch tools from a single server
    async def _fetch_tools_from_server_async(self, alias: str) -> List[Dict[str, Any]]:
        server_info = self.servers[alias]
        if not server_info["initialized"] or not server_info["session"]:
            ASCIIColors.debug(f"{self.binding_name}: Skipping tool refresh for non-initialized server '{alias}'.")
            return []
        
        try:
            list_tools_result = await server_info["session"].list_tools()
            server_tools = []
            for tool_obj in list_tools_result.tools:
                input_schema_dict = {}
                tool_input_schema = getattr(tool_obj, 'inputSchema', getattr(tool_obj, 'input_schema', None))
                if tool_input_schema:
                    if hasattr(tool_input_schema, 'model_dump'):
                        input_schema_dict = tool_input_schema.model_dump(mode='json', exclude_none=True)
                    elif isinstance(tool_input_schema, dict):
                        input_schema_dict = tool_input_schema
                
                tool_name_for_client = f"{alias}{TOOL_NAME_SEPARATOR}{tool_obj.name}"

                server_tools.append({
                    "name": tool_name_for_client,
                    "description": tool_obj.description or "",
                    "input_schema": input_schema_dict
                })
            ASCIIColors.info(f"{self.binding_name}: Found {len(server_tools)} tools on server '{alias}'.")
            return server_tools
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Error refreshing tools from '{alias}': {e}")
            return []


    ### MODIFIED: Discovers tools from all configured servers
    def discover_tools(self, force_refresh: bool = False, timeout_per_server: float = 30.0, **kwargs) -> List[Dict[str, Any]]:
        if not self.servers:
            return []

        # Initialize all servers that are not yet initialized.
        for alias in self.servers.keys():
            try:
                # _ensure_initialized_sync is internally locked and idempotent
                self._ensure_initialized_sync(alias, timeout=timeout_per_server)
            except Exception as e:
                # One server failing to connect shouldn't stop discovery on others.
                ASCIIColors.warning(f"{self.binding_name}: Could not ensure connection to '{alias}' for discovery: {e}")
        
        try:
            if force_refresh or not self._discovered_tools_cache:
                # The timeout for refreshing all tools should be longer
                self._run_async(self._refresh_all_tools_cache_async(), timeout=timeout_per_server * len(self.servers))
            return self._discovered_tools_cache
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Problem during tool discovery: {e}")
            return []
            
    ### MODIFIED: Now operates on a specific server identified by alias
    async def _execute_tool_async(self, alias: str, actual_tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        server_info = self.servers[alias]
        server_url = server_info["url"]
        
        if not server_info["initialized"] or not server_info["session"]:
            return {"error": f"Not connected to server '{alias}' ({server_url})", "status_code": 503}
        
        ASCIIColors.info(f"{self.binding_name}: Executing remote tool '{actual_tool_name}' on '{alias}' ({server_url}) with params: {json.dumps(params)}")
        try:
            mcp_call_result = await server_info["session"].call_tool(name=actual_tool_name, arguments=params)
            output_parts = [p.text for p in mcp_call_result.content if isinstance(p, types.TextContent) and p.text is not None] if mcp_call_result.content else []
            if not output_parts: return {"output": {"message": "Tool executed but returned no textual content."}, "status_code": 200}
            
            combined_output_str = "\n".join(output_parts)
            try:
                return {"output": json.loads(combined_output_str), "status_code": 200}
            except json.JSONDecodeError:
                return {"output": combined_output_str, "status_code": 200}
        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing remote tool '{actual_tool_name}' on '{alias}': {str(e)}", "status_code": 500}

    ### MODIFIED: Parses alias from tool name and routes the call
    def execute_tool(self, tool_name_with_alias: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        timeout = float(kwargs.get('timeout', 60.0))
        
        if TOOL_NAME_SEPARATOR not in tool_name_with_alias:
            return {"error": f"Invalid tool name format. Expected 'alias{TOOL_NAME_SEPARATOR}tool_name', but got '{tool_name_with_alias}'.", "status_code": 400}

        alias, actual_tool_name = tool_name_with_alias.split(TOOL_NAME_SEPARATOR, 1)

        if alias not in self.servers:
            return {"error": f"Tool name '{tool_name_with_alias}' has an unknown server alias '{alias}'.", "status_code": 400}

        try:
            # Ensure this specific server is connected before executing
            self._ensure_initialized_sync(alias, timeout=timeout)
            return self._run_async(self._execute_tool_async(alias, actual_tool_name, params), timeout=timeout)
        except (ConnectionError, RuntimeError) as e:
            return {"error": f"{self.binding_name}: Connection issue for server '{alias}': {e}", "status_code": 503}
        except TimeoutError:
            return {"error": f"{self.binding_name}: Remote tool '{actual_tool_name}' on '{alias}' timed out.", "status_code": 504}
        except Exception as e:
            trace_exception(e)
            return {"error": f"{self.binding_name}: Failed to run remote MCP tool '{actual_tool_name}' on '{alias}': {e}", "status_code": 500}

    ### MODIFIED: Closes all connections
    def close(self):
        ASCIIColors.info(f"{self.binding_name}: Closing all remote connections...")
        
        async def _close_all_connections():
            close_tasks = []
            for alias, server_info in self.servers.items():
                if server_info.get("exit_stack"):
                    ASCIIColors.info(f"{self.binding_name}: Closing connection to '{alias}'...")
                    close_tasks.append(server_info["exit_stack"].aclose())
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

        # Check if loop is running before trying to schedule work on it
        if self._loop and self._loop.is_running():
            try:
                self._run_async(_close_all_connections(), timeout=10.0)
            except Exception as e:
                ASCIIColors.error(f"{self.binding_name}: Error during async close: {e}")
        
        # Reset all server states
        for alias in self.servers:
            self.servers[alias].update({
                "exit_stack": None,
                "session": None,
                "initialized": False
            })
        
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        ASCIIColors.info(f"{self.binding_name}: Remote connection binding closed.")

    def get_binding_config(self) -> Dict[str, Any]:
        return self.config
    
    
    def set_auth_config(self, alias: str, auth_config: Dict[str, Any]):
        """
        Dynamically updates the authentication configuration for a specific server.

        If a connection was already active for this server, it will be closed to force
        a new connection with the new authentication details on the next call.

        Args:
            alias (str): The alias of the server to update (the key in servers_infos).
            auth_config (Dict[str, Any]): The new authentication configuration dictionary.
                Example: {"type": "bearer", "token": "new-token-here"}
        """
        ASCIIColors.info(f"{self.binding_name}: Updating auth_config for server '{alias}'.")
        
        server_info = self.servers.get(alias)
        if not server_info:
            raise ValueError(f"Server alias '{alias}' does not exist in the configuration.")

        # Update the configuration in the binding's internal state
        server_info["config"]["auth_config"] = auth_config

        # If the server was already initialized, its connection is now obsolete.
        # We must close it and mark it as uninitialized.
        if server_info["initialized"]:
            ASCIIColors.warning(f"{self.binding_name}: Existing connection for '{alias}' is outdated due to new authentication. It will be reset.")
            try:
                # Execute the close operation asynchronously on the event loop thread
                self._run_async(self._close_connection_async(alias), timeout=10.0)
            except Exception as e:
                ASCIIColors.error(f"{self.binding_name}: Error while closing the outdated connection for '{alias}': {e}")
                # Even on error, reset the state to force a new connection attempt
                server_info.update({"session": None, "exit_stack": None, "initialized": False})


    # --- NEW INTERNAL HELPER METHOD ---
    async def _close_connection_async(self, alias: str):
        """Cleanly closes the connection for a specific server alias."""
        server_info = self.servers.get(alias)
        if not server_info or not server_info.get("exit_stack"):
            return # Nothing to do.

        ASCIIColors.info(f"{self.binding_name}: Closing connection for '{alias}'...")
        try:
            await server_info["exit_stack"].aclose()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Exception while closing the exit_stack for '{alias}': {e}")
        finally:
            # Reset the state for this alias, no matter what.
            server_info.update({
                "session": None,
                "exit_stack": None,
                "initialized": False
            })
