# File: lollms_client/mcp_bindings/standard_mcp/__init__.py

import pipmaster as pm

# Ensure critical dependencies for this binding are present.
# If pipmaster itself is missing, lollms_client is not correctly installed.
pm.ensure_packages(["mcp", "ascii-colors"])

import asyncio
import json
import threading
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# These imports should now succeed if pipmaster did its job.
from lollms_client.lollms_mcp_binding import LollmsMCPBinding # Assuming this base class exists
from ascii_colors import ASCIIColors, trace_exception

# Attempt to import MCP library components.
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp import types # Use mcp.types for data structures
    MCP_LIBRARY_AVAILABLE = True
    ASCIIColors.green("Successfully imported MCP library components for StandardMCPBinding.")
except ImportError as e:
    ASCIIColors.error(f"StandardMCPBinding: Critical MCP library components could not be imported even after pipmaster attempt: {e}")
    ASCIIColors.error("Please check your Python environment, internet connection, and pip installation.")
    ASCIIColors.error("StandardMCPBinding will be non-functional.")
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    types = None # MCP types module unavailable
    MCP_LIBRARY_AVAILABLE = False

# This variable is used by LollmsMCPBindingManager to identify the binding class.
BindingName = "StandardMCPBinding" # Must match the class name below
TOOL_NAME_SEPARATOR = "::"

class StandardMCPBinding(LollmsMCPBinding):
    """
    A LollmsMCPBinding to connect to multiple standard Model Context Protocol (MCP) servers.
    This binding acts as an MCP client to these servers.
    Each server is launched via a command, communicates over stdio, and is identified by a unique alias.
    Tool names are prefixed with 'server_alias::' for disambiguation.
    """

    def __init__(self,
                 **kwargs: Any):
        super().__init__(binding_name="standard_mcp")
        self.config = kwargs
        initial_servers = kwargs.get("initial_servers", {})

        self._server_configs: Dict[str, Dict[str, Any]] = {}
        # Type hint with ClientSession, actual obj if MCP_LIBRARY_AVAILABLE
        self._mcp_sessions: Dict[str, ClientSession] = {} # type: ignore
        self._exit_stacks: Dict[str, AsyncExitStack] = {}
        self._discovered_tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._server_locks: Dict[str, threading.Lock] = {}
        self._initialization_status: Dict[str, bool] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: Cannot initialize; MCP library components are missing.")
            return # Binding remains in a non-functional state

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_event_loop, daemon=True,
                                        name=f"{self.binding_name}EventLoopThread")
        self._thread.start()
        ASCIIColors.info(f"{self.binding_name}: Event loop thread started.")

        if initial_servers:
            for alias, config_data in initial_servers.items():
                if isinstance(config_data, dict):
                    # Ensure command is a list
                    command = config_data.get("command")
                    if isinstance(command, str): # if command is a single string, convert to list
                        command = command.split()

                    self.add_server(
                        alias=alias,
                        command=command, # type: ignore
                        cwd=config_data.get("cwd"),
                        env=config_data.get("env")
                    )
                else:
                    ASCIIColors.warning(f"{self.binding_name}: Invalid configuration for server alias '{alias}' in 'initial_servers'. Expected a dictionary.")

    def _start_event_loop(self):
        if not self._loop: return
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Cleanup tasks before closing the loop
            if hasattr(asyncio, 'all_tasks'): # Python 3.7+
                pending = asyncio.all_tasks(self._loop)
            else: # Python 3.6
                pending = asyncio.Task.all_tasks(self._loop) # type: ignore

            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            if self._loop.is_running():
                self._loop.stop()

            if not self._loop.is_closed():
                if sys.platform == "win32" and isinstance(self._loop, asyncio.ProactorEventLoop): # type: ignore
                    self._loop.call_soon(self._loop.stop)
                    try:
                        # This run_until_complete might be problematic if called from non-loop thread after stop
                        # but often necessary for proactor loop cleanup on Windows
                        self._loop.run_until_complete(asyncio.sleep(0.1))
                    except RuntimeError as e:
                        if "cannot be called from a different thread" not in str(e):
                            ASCIIColors.warning(f"{self.binding_name}: Minor issue during proactor loop sleep: {e}")
                self._loop.close()
            ASCIIColors.info(f"{self.binding_name}: Asyncio event loop has stopped and closed.")


    def _run_async_task(self, coro, timeout: Optional[float] = None) -> Any:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not self._loop.is_running() or not self._thread or not self._thread.is_alive():
            raise RuntimeError(f"{self.binding_name}'s event loop is not operational or MCP library is missing.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel() # Attempt to cancel the coroutine
            raise
        except Exception:
            raise

    def add_server(self, alias: str, command: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> bool:
        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: Cannot add server '{alias}', MCP library is not available.")
            return False

        if not alias or not isinstance(alias, str):
            ASCIIColors.error(f"{self.binding_name}: Server alias must be a non-empty string.")
            return False
        if not command or not isinstance(command, list) or not all(isinstance(c, str) for c in command) or not command[0]:
            ASCIIColors.error(f"{self.binding_name}: Server command for '{alias}' must be a non-empty list of strings (e.g., ['python', 'server.py']).")
            return False

        if alias in self._server_configs:
            ASCIIColors.warning(f"{self.binding_name}: Reconfiguring server '{alias}'. Existing connection (if any) will be closed.")
            self.remove_server(alias, silent=True)

        self._server_configs[alias] = {"command": command, "cwd": cwd, "env": env}
        self._server_locks[alias] = threading.Lock()
        self._initialization_status[alias] = False
        self._discovered_tools_cache[alias] = [] # Initialize cache for the new server
        ASCIIColors.info(f"{self.binding_name}: Server '{alias}' configured with command: {command}")

        if "initial_servers" not in self.config:
            self.config["initial_servers"] = {}
        if isinstance(self.config["initial_servers"], dict): # Ensure it's a dict
            self.config["initial_servers"][alias] = self._server_configs[alias]
        return True

    async def _close_server_connection_async(self, alias: str):
        exit_stack_to_close = self._exit_stacks.pop(alias, None)
        # Pop session and status immediately to reflect desired state
        self._mcp_sessions.pop(alias, None)
        self._initialization_status[alias] = False

        if exit_stack_to_close:
            ASCIIColors.info(f"{self.binding_name}: Attempting to close MCP connection for server '{alias}'...")
            try:
                await exit_stack_to_close.aclose()
                ASCIIColors.info(f"{self.binding_name}: MCP connection for '{alias}' resources released via aclose.")
            except RuntimeError as e:
                if "Attempted to exit cancel scope in a different task" in str(e):
                    ASCIIColors.warning(f"{self.binding_name}: Known anyio task ownership issue during close for '{alias}': {e}.")
                    ASCIIColors.warning(f"{self.binding_name}: Underlying MCP client resources for '{alias}' may not have been fully cleaned up due to this anyio constraint.")
                    # At this point, the stdio process might still be running.
                    # Further action (like trying to kill the process) is outside the scope of AsyncExitStack.
                else:
                    # Reraise other RuntimeErrors or handle them
                    trace_exception(e)
                    ASCIIColors.error(f"{self.binding_name}: Unexpected RuntimeError closing MCP connection for '{alias}': {e}")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"{self.binding_name}: General error closing MCP connection for '{alias}': {e}")
        # else:
            # ASCIIColors.debug(f"{self.binding_name}: No active exit stack found for server '{alias}' to close (already closed or never fully initialized).")

    def remove_server(self, alias: str, silent: bool = False):
        if not MCP_LIBRARY_AVAILABLE:
            if not silent: ASCIIColors.error(f"{self.binding_name}: Cannot remove server '{alias}', MCP library issues persist."); return

        if alias not in self._server_configs:
            if not silent: ASCIIColors.warning(f"{self.binding_name}: Server '{alias}' not found for removal.")
            return

        if not silent: ASCIIColors.info(f"{self.binding_name}: Removing server '{alias}'.")

        if self._initialization_status.get(alias) or alias in self._exit_stacks or alias in self._mcp_sessions:
             try:
                self._run_async_task(self._close_server_connection_async(alias), timeout=10.0)
             except RuntimeError as e:
                if not silent: ASCIIColors.warning(f"{self.binding_name}: Could not run async close for '{alias}' (event loop issue?): {e}")
             except Exception as e:
                if not silent: ASCIIColors.error(f"{self.binding_name}: Exception during async close for '{alias}': {e}")

        self._server_configs.pop(alias, None)
        self._server_locks.pop(alias, None)
        self._initialization_status.pop(alias, None)
        self._discovered_tools_cache.pop(alias, None)
        if "initial_servers" in self.config and isinstance(self.config["initial_servers"], dict) and alias in self.config["initial_servers"]:
            self.config["initial_servers"].pop(alias)
        if not silent: ASCIIColors.info(f"{self.binding_name}: Server '{alias}' removed.")

    async def _initialize_connection_async(self, alias: str) -> bool:
        if not MCP_LIBRARY_AVAILABLE or not types or not ClientSession or not StdioServerParameters or not stdio_client:
            ASCIIColors.error(f"{self.binding_name}: MCP library components (types, ClientSession, etc.) not available. Cannot initialize '{alias}'.")
            return False
        if self._initialization_status.get(alias): return True
        if alias not in self._server_configs:
            ASCIIColors.error(f"{self.binding_name}: No configuration for server alias '{alias}'. Cannot initialize.")
            return False

        config = self._server_configs[alias]
        ASCIIColors.info(f"{self.binding_name}: Initializing MCP connection for server '{alias}'...")
        try:
            if alias in self._exit_stacks: # Should ideally be cleaned up if a previous attempt failed
                old_stack = self._exit_stacks.pop(alias)
                await old_stack.aclose()

            exit_stack = AsyncExitStack()
            self._exit_stacks[alias] = exit_stack

            server_params = StdioServerParameters(
                command=config["command"][0],
                args=config["command"][1:],
                cwd=Path(config["cwd"]) if config["cwd"] else None,
                env=config["env"]
            )
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))

            # CORRECTED: Removed client_name from ClientSession constructor
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

            await session.initialize() # This is where client capabilities/info might be exchanged
            self._mcp_sessions[alias] = session
            self._initialization_status[alias] = True
            ASCIIColors.green(f"{self.binding_name}: Successfully initialized MCP session for server '{alias}'.")
            await self._refresh_tools_cache_async(alias)
            return True
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Failed to initialize MCP connection for '{alias}': {e}")
            if alias in self._exit_stacks:
                current_stack = self._exit_stacks.pop(alias)
                try:
                    await current_stack.aclose()
                except Exception as e_close:
                    ASCIIColors.error(f"{self.binding_name}: Error during cleanup after failed init for '{alias}': {e_close}")
            self._initialization_status[alias] = False
            self._mcp_sessions.pop(alias, None)
            return False

    def _ensure_server_initialized_sync(self, alias: str, timeout: float = 30.0):
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
             raise ConnectionError(f"{self.binding_name}: MCP library/event loop/types module not available. Cannot initialize server '{alias}'.")

        if alias not in self._server_configs:
            raise ValueError(f"{self.binding_name}: Server alias '{alias}' is not configured.")

        lock = self._server_locks.get(alias)
        if not lock:
            ASCIIColors.error(f"{self.binding_name}: Internal error - No lock for server '{alias}'. Creating one now.")
            self._server_locks[alias] = threading.Lock()
            lock = self._server_locks[alias]


        with lock:
            if not self._initialization_status.get(alias):
                ASCIIColors.info(f"{self.binding_name}: Connection for '{alias}' not initialized. Attempting initialization...")
                try:
                    success = self._run_async_task(self._initialize_connection_async(alias), timeout=timeout)
                    if not success: 
                        # If init itself reports failure (e.g. returns False from _initialize_connection_async)
                        self._discovered_tools_cache[alias] = [] # CLEAR CACHE ON FAILURE
                        raise ConnectionError(f"MCP init for '{alias}' reported failure.")
                except TimeoutError:
                    self._discovered_tools_cache[alias] = [] # CLEAR CACHE ON FAILURE
                    raise ConnectionError(f"MCP init for '{alias}' timed out.")
                except Exception as e: # Other exceptions during run_async_task
                    self._discovered_tools_cache[alias] = [] # CLEAR CACHE ON FAILURE
                    raise ConnectionError(f"MCP init for '{alias}' failed: {e}")
        
        if not self._initialization_status.get(alias) or alias not in self._mcp_sessions:
            # This means init was thought to be successful by the lock block, but status is bad
            # This case might indicate a race or an issue if _initialize_connection_async doesn't set status correctly on all paths
            self._discovered_tools_cache[alias] = [] # Also clear here as a safeguard
            raise ConnectionError(f"MCP Session for '{alias}' not valid post-init attempt, despite no immediate error.")

    async def _refresh_tools_cache_async(self, alias: str):
        if not MCP_LIBRARY_AVAILABLE or not types:
            ASCIIColors.error(f"{self.binding_name}: MCP library or types module not available. Cannot refresh tools for '{alias}'.")
            return
        if not self._initialization_status.get(alias) or alias not in self._mcp_sessions:
            ASCIIColors.warning(f"{self.binding_name}: Server '{alias}' not initialized or no session. Cannot refresh tools.")
            return

        session = self._mcp_sessions[alias]
        ASCIIColors.info(f"{self.binding_name}: Refreshing tools cache for server '{alias}'...")
        try:
            list_tools_result = await session.list_tools() # Expected to be types.ListToolsResult
            current_server_tools = []
            if list_tools_result and list_tools_result.tools:
                for tool_obj in list_tools_result.tools: # tool_obj is expected to be types.Tool
                    # --- DEBUGGING ---
                    # print(f"DEBUG: tool_obj type: {type(tool_obj)}")
                    # print(f"DEBUG: tool_obj dir: {dir(tool_obj)}")
                    # if hasattr(tool_obj, 'model_fields'): print(f"DEBUG: tool_obj fields: {tool_obj.model_fields.keys()}") # Pydantic v2
                    # elif hasattr(tool_obj, '__fields__'): print(f"DEBUG: tool_obj fields: {tool_obj.__fields__.keys()}") # Pydantic v1
                    # if hasattr(tool_obj, 'model_dump_json'): print(f"DEBUG: tool_obj JSON: {tool_obj.model_dump_json(indent=2)}")
                    # elif hasattr(tool_obj, 'json'): print(f"DEBUG: tool_obj JSON: {tool_obj.json(indent=2)}")
                    # --- END DEBUGGING ---

                    input_schema_dict = {}
                    # Try accessing with 'inputSchema' (camelCase) or check other potential names based on debug output
                    tool_input_schema = None
                    if hasattr(tool_obj, 'inputSchema'): # Common JSON convention
                        tool_input_schema = tool_obj.inputSchema
                    elif hasattr(tool_obj, 'input_schema'): # Python convention
                        tool_input_schema = tool_obj.input_schema
                    # Add more elif for other possibilities if revealed by debugging

                    if tool_input_schema: # Check if the schema object itself exists and is not None
                        # tool_input_schema is expected to be types.InputSchema | None
                        # or a Pydantic model that has model_dump
                        if hasattr(tool_input_schema, 'model_dump'):
                             input_schema_dict = tool_input_schema.model_dump(mode='json', exclude_none=True)
                        else:
                            # If it's not a Pydantic model but some other dict-like structure
                            # This part might need adjustment based on what tool_input_schema actually is
                            ASCIIColors.warning(f"{self.binding_name}: input schema for tool '{tool_obj.name}' on '{alias}' is not a Pydantic model with model_dump. Type: {type(tool_input_schema)}")
                            if isinstance(tool_input_schema, dict):
                                input_schema_dict = tool_input_schema
                            # else: leave it as empty dict

                    tool_dict = {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                        "input_schema": input_schema_dict
                    }
                    current_server_tools.append(tool_dict)
            self._discovered_tools_cache[alias] = current_server_tools
            ASCIIColors.green(f"{self.binding_name}: Tools cache for '{alias}' refreshed. Found {len(current_server_tools)} tools.")
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"{self.binding_name}: Error refreshing tools cache for '{alias}': {e}")
            self._discovered_tools_cache[alias] = [] # Clear cache on error

    def discover_tools(self, specific_tool_names: Optional[List[str]]=None, force_refresh: bool=False, timeout_per_server: float=10.0, **kwargs) -> List[Dict[str, Any]]:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
             ASCIIColors.warning(f"{self.binding_name}: Cannot discover tools, MCP library, event loop, or types module not available.")
             return []

        stn = kwargs.get('specific_tool_names', specific_tool_names)
        fr = kwargs.get('force_refresh', force_refresh)
        tps = kwargs.get('timeout_per_server', timeout_per_server)

        all_tools: List[Dict[str, Any]] = []
        active_aliases = list(self._server_configs.keys())

        for alias in active_aliases:
            try:
                if force_refresh: # Explicitly clear before ensuring init if forcing
                    ASCIIColors.yellow(f"{self.binding_name}: Force refresh - clearing cache for '{alias}' before init.")
                    self._discovered_tools_cache[alias] = [] 
                
                self._ensure_server_initialized_sync(alias, timeout=tps)

                # If force_refresh OR if server is initialized but cache is empty/stale
                if force_refresh or (self._initialization_status.get(alias) and not self._discovered_tools_cache.get(alias)):
                    ASCIIColors.info(f"{self.binding_name}: Refreshing tools for '{alias}' (force_refresh={force_refresh}, cache_empty={not self._discovered_tools_cache.get(alias)}).")
                    self._run_async_task(self._refresh_tools_cache_async(alias), timeout=tps)

                if fr or (self._initialization_status.get(alias) and not self._discovered_tools_cache.get(alias)):
                    ASCIIColors.info(f"{self.binding_name}: Force refreshing tools for '{alias}' or cache is empty.")
                    self._run_async_task(self._refresh_tools_cache_async(alias), timeout=tps)

                for tool_data in self._discovered_tools_cache.get(alias, []):
                    prefixed_tool_data = tool_data.copy()
                    prefixed_tool_data["name"] = f"{alias}{TOOL_NAME_SEPARATOR}{tool_data['name']}"
                    all_tools.append(prefixed_tool_data)
            except ConnectionError as e:
                ASCIIColors.error(f"{self.binding_name}: Connection problem with server '{alias}' during tool discovery: {e}")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"{self.binding_name}: Unexpected problem with server '{alias}' during tool discovery: {e}")

        if stn:
            return [t for t in all_tools if t.get("name") in stn]
        return all_tools

    def _parse_tool_name(self, prefixed_tool_name: str) -> Optional[Tuple[str, str]]:
        parts = prefixed_tool_name.split(TOOL_NAME_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        ASCIIColors.warning(f"{self.binding_name}: Tool name '{prefixed_tool_name}' is not in the expected 'alias{TOOL_NAME_SEPARATOR}tool' format.")
        return None

    async def _execute_tool_async(self, server_alias: str, actual_tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not MCP_LIBRARY_AVAILABLE or not types:
            error_msg = f"{self.binding_name}: MCP library or types module not available. Cannot execute tool '{actual_tool_name}' on '{server_alias}'."
            ASCIIColors.error(error_msg)
            return {"error": error_msg, "status_code": 503}

        if not self._initialization_status.get(server_alias) or server_alias not in self._mcp_sessions:
            error_msg = f"Server '{server_alias}' not initialized or session lost. Cannot execute tool '{actual_tool_name}'."
            ASCIIColors.error(f"{self.binding_name}: {error_msg}")
            return {"error": error_msg, "status_code": 503}

        session = self._mcp_sessions[server_alias]
        # Use a more careful way to log params if they can be very large or sensitive
        params_log = {k: (v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) for k,v in params.items()}
        ASCIIColors.info(f"{self.binding_name}: Executing MCP tool '{actual_tool_name}' on server '{server_alias}' with params: {json.dumps(params_log)}")
        try:
            # call_tool returns types.CallToolResult
            mcp_call_result = await session.call_tool(name=actual_tool_name, arguments=params)

            output_parts = []
            if mcp_call_result and mcp_call_result.content: # content is List[types.ContentPart]
                for content_part in mcp_call_result.content:
                    if isinstance(content_part, types.TextContent) and hasattr(content_part, 'text') and content_part.text is not None:
                        output_parts.append(content_part.text)

            if not output_parts:
                ASCIIColors.info(f"{self.binding_name}: Tool '{actual_tool_name}' on '{server_alias}' executed but returned no textual content.")
                return {"output": {"message": "Tool executed successfully but returned no textual content."}, "status_code": 200}

            combined_output_str = "\n".join(output_parts)
            ASCIIColors.success(f"{self.binding_name}: Tool '{actual_tool_name}' on '{server_alias}' executed. Raw output (first 200 chars): '{combined_output_str[:200]}'")

            try:
                parsed_output = json.loads(combined_output_str)
                return {"output": parsed_output, "status_code": 200}
            except json.JSONDecodeError:
                return {"output": combined_output_str, "status_code": 200}

        except Exception as e:
            trace_exception(e)
            error_msg = f"Error executing tool '{actual_tool_name}' on server '{server_alias}': {str(e)}"
            ASCIIColors.error(f"{self.binding_name}: {error_msg}")
            return {"error": error_msg, "status_code": 500}

    def execute_tool(self, tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
             error_msg = f"{self.binding_name}: MCP support (library, event loop, or types module) not available. Cannot execute tool '{tool_name}'."
             ASCIIColors.warning(error_msg)
             return {"error": error_msg, "status_code": 503}

        timeout = float(kwargs.get('timeout', 60.0))

        parsed_name = self._parse_tool_name(tool_name)
        if not parsed_name:
            return {"error": f"Invalid tool name format for {self.binding_name}: '{tool_name}'. Expected 'alias{TOOL_NAME_SEPARATOR}toolname'.", "status_code": 400}

        server_alias, actual_tool_name = parsed_name

        if server_alias not in self._server_configs:
            return {"error": f"Server alias '{server_alias}' (from tool name '{tool_name}') is not configured.", "status_code": 404}

        try:
            init_timeout = min(timeout, 30.0)
            self._ensure_server_initialized_sync(server_alias, timeout=init_timeout)
        except ConnectionError as e:
            return {"error": f"{self.binding_name}: Connection or configuration issue for server '{server_alias}': {e}", "status_code": 503}
        except Exception as e:
            trace_exception(e)
            return {"error": f"{self.binding_name}: Failed to ensure server '{server_alias}' is initialized: {e}", "status_code": 500}

        try:
            return self._run_async_task(self._execute_tool_async(server_alias, actual_tool_name, params), timeout=timeout)
        except TimeoutError:
            return {"error": f"{self.binding_name}: Tool '{actual_tool_name}' on server '{server_alias}' timed out after {timeout} seconds.", "status_code": 504}
        except RuntimeError as e:
             return {"error": f"{self.binding_name}: Runtime error executing tool '{actual_tool_name}' on '{server_alias}': {e}", "status_code": 500}
        except Exception as e:
            trace_exception(e)
            return {"error": f"{self.binding_name}: An unexpected error occurred while running MCP tool '{actual_tool_name}' on server '{server_alias}': {e}", "status_code": 500}

    def close(self):
        ASCIIColors.info(f"{self.binding_name}: Initiating shutdown process...")

        if hasattr(self, '_server_configs') and self._server_configs:
            active_aliases = list(self._server_configs.keys())
            if active_aliases:
                ASCIIColors.info(f"{self.binding_name}: Closing connections for servers: {active_aliases}")
                for alias in active_aliases:
                    self.remove_server(alias, silent=True)

        if hasattr(self, '_loop') and self._loop:
            if self._loop.is_running():
                ASCIIColors.info(f"{self.binding_name}: Requesting event loop to stop.")
                self._loop.call_soon_threadsafe(self._loop.stop)

        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            ASCIIColors.info(f"{self.binding_name}: Waiting for event loop thread to join...")
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                ASCIIColors.warning(f"{self.binding_name}: Event loop thread did not terminate cleanly after 10 seconds.")
            else:
                ASCIIColors.info(f"{self.binding_name}: Event loop thread joined successfully.")

        ASCIIColors.info(f"{self.binding_name}: Binding closed.")

    def __del__(self):
        # Check if attributes relevant to closing exist to prevent errors if __init__ failed early
        needs_close = False
        if hasattr(self, '_loop') and self._loop and (self._loop.is_running() or not self._loop.is_closed()):
            needs_close = True
        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            needs_close = True
        if hasattr(self, '_server_configs') and self._server_configs: # Check if there are any servers to close
             needs_close = True

        if needs_close:
            ASCIIColors.warning(f"{self.binding_name}: __del__ called; attempting to close resources. Explicit .close() is recommended for reliability.")
            try:
                self.close()
            except Exception as e:
                # __del__ should not raise exceptions
                ASCIIColors.error(f"{self.binding_name}: Error during __del__ cleanup: {e}")