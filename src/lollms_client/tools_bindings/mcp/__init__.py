import asyncio
import json
import threading
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

import pipmaster as pm
from lollms_client.lollms_tools_binding import LollmsToolBinding
from ascii_colors import ASCIIColors, trace_exception

try:
    pm.ensure_packages(["mcp", "httpx"])
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    import httpx
    MCP_LIBRARY_AVAILABLE = True
except ImportError as e:
    ASCIIColors.error(f"MCPBinding: Critical MCP library components could not be imported: {e}")
    MCP_LIBRARY_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    types = None
    httpx = None

BindingName = "MCPBinding"
TOOL_NAME_SEPARATOR = "::"

class MCPBinding(LollmsToolBinding):
    """
    A unified LollmsToolBinding to connect to multiple Model Context Protocol (MCP) servers.
    Supports both local subprocess servers (stdio) and remote servers (streamable HTTP).
    Tools are namespaced with 'server_alias::tool_name'.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(binding_name="mcp")
        self.config = kwargs
        servers_config = kwargs.get("servers", [])

        self._server_configs: Dict[str, Dict[str, Any]] = {}
        self._mcp_sessions: Dict[str, ClientSession] = {} if MCP_LIBRARY_AVAILABLE else {}
        self._exit_stacks: Dict[str, AsyncExitStack] = {}
        self._discovered_tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._server_locks: Dict[str, threading.Lock] = {}
        self._initialization_status: Dict[str, bool] = {}
        
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: Cannot initialize; MCP library components are missing.")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_event_loop, daemon=True, name=f"{self.binding_name}EventLoopThread")
        self._thread.start()
        ASCIIColors.info(f"{self.binding_name}: Event loop thread started.")

        if servers_config:
            for server_data in servers_config:
                if isinstance(server_data, dict):
                    alias = server_data.get("alias")
                    s_type = server_data.get("type", "stdio").lower()
                    if not alias:
                        ASCIIColors.warning(f"{self.binding_name}: Server config missing alias. Skipping.")
                        continue

                    if s_type == "stdio":
                        command = server_data.get("command")
                        if isinstance(command, str):
                            command = command.split()
                        self.add_server(
                            alias=alias,
                            server_type="stdio",
                            command=command,
                            cwd=server_data.get("cwd"),
                            env=server_data.get("env")
                        )
                    elif s_type == "http":
                        self.add_server(
                            alias=alias,
                            server_type="http",
                            url=server_data.get("url"),
                            auth_config=server_data.get("auth_config", {}),
                            timeout=server_data.get("timeout", 2.0)
                        )
                    else:
                        ASCIIColors.warning(f"{self.binding_name}: Unknown server type '{s_type}' for alias '{alias}'.")

    def _start_event_loop(self):
        if not self._loop: return
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            if hasattr(asyncio, 'all_tasks'):
                pending = asyncio.all_tasks(self._loop)
            else:
                pending = asyncio.Task.all_tasks(self._loop) # type: ignore

            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            if self._loop.is_running():
                self._loop.stop()

            if not self._loop.is_closed():
                self._loop.close()
            ASCIIColors.info(f"{self.binding_name}: Asyncio event loop has stopped and closed.")

    def _run_async_task(self, coro, timeout: Optional[float] = None) -> Any:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not self._loop.is_running() or not self._thread or not self._thread.is_alive():
            raise RuntimeError(f"{self.binding_name}'s event loop is not operational.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise
        except Exception:
            raise

    def add_server(self, alias: str, server_type: str, **kwargs) -> bool:
        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: Cannot add server '{alias}', MCP library not available.")
            return False

        if not alias or not isinstance(alias, str):
            ASCIIColors.error(f"{self.binding_name}: Server alias must be a non-empty string.")
            return False

        if alias in self._server_configs:
            ASCIIColors.warning(f"{self.binding_name}: Reconfiguring server '{alias}'. Closing existing connection.")
            self.remove_server(alias, silent=True)

        config = {"type": server_type}
        
        if server_type == "stdio":
            command = kwargs.get("command")
            if not command or not isinstance(command, list) or not all(isinstance(c, str) for c in command) or not command[0]:
                ASCIIColors.error(f"{self.binding_name}: Server command for '{alias}' must be a non-empty list of strings.")
                return False
            config["command"] = command
            config["cwd"] = kwargs.get("cwd")
            config["env"] = kwargs.get("env")
        elif server_type == "http":
            url = kwargs.get("url")
            if not url or not isinstance(url, str):
                ASCIIColors.error(f"{self.binding_name}: Server URL for '{alias}' must be a non-empty string.")
                return False
            config["url"] = url
            config["auth_config"] = kwargs.get("auth_config", {})
            config["timeout"] = float(kwargs.get("timeout", 2.0))
        else:
            ASCIIColors.error(f"{self.binding_name}: Invalid server type '{server_type}' for '{alias}'.")
            return False

        self._server_configs[alias] = config
        self._server_locks[alias] = threading.Lock()
        self._initialization_status[alias] = False
        self._discovered_tools_cache[alias] = []
        ASCIIColors.info(f"{self.binding_name}: Server '{alias}' configured ({server_type}).")
        return True

    async def _close_server_connection_async(self, alias: str):
        exit_stack_to_close = self._exit_stacks.pop(alias, None)
        self._mcp_sessions.pop(alias, None)
        self._initialization_status[alias] = False

        if exit_stack_to_close:
            ASCIIColors.info(f"{self.binding_name}: Closing MCP connection for '{alias}'...")
            try:
                await exit_stack_to_close.aclose()
            except RuntimeError as e:
                if "Attempted to exit cancel scope in a different task" in str(e):
                    ASCIIColors.warning(f"{self.binding_name}: Known anyio task ownership issue during close for '{alias}'.")
                else:
                    trace_exception(e)
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"{self.binding_name}: General error closing MCP connection for '{alias}': {e}")

    def remove_server(self, alias: str, silent: bool = False):
        if not MCP_LIBRARY_AVAILABLE:
            if not silent: ASCIIColors.error(f"{self.binding_name}: Cannot remove server '{alias}', MCP library issues persist.")
            return

        if alias not in self._server_configs:
            if not silent: ASCIIColors.warning(f"{self.binding_name}: Server '{alias}' not found for removal.")
            return

        if not silent: ASCIIColors.info(f"{self.binding_name}: Removing server '{alias}'.")

        if self._initialization_status.get(alias) or alias in self._exit_stacks or alias in self._mcp_sessions:
            try:
                self._run_async_task(self._close_server_connection_async(alias), timeout=10.0)
            except Exception as e:
                if not silent: ASCIIColors.warning(f"{self.binding_name}: Exception during async close for '{alias}': {e}")

        self._server_configs.pop(alias, None)
        self._server_locks.pop(alias, None)
        self._initialization_status.pop(alias, None)
        self._discovered_tools_cache.pop(alias, None)

    def _prepare_http_headers(self, alias: str) -> Dict[str, str]:
        server_info = self._server_configs[alias]
        auth_config = server_info.get("auth_config", {})
        server_url = server_info.get("url", "")
        
        origin = ""
        try:
            parsed = urlparse(server_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
        except:
            pass

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        
        if origin and server_url.startswith("https"):
            headers["Origin"] = origin
            headers["Referer"] = origin + "/"
        
        auth_type = auth_config.get("type")
        if auth_type == "api_key":
            api_key = auth_config.get("key")
            header_name = auth_config.get("header_name", "X-API-Key")
            if api_key:
                headers[header_name] = api_key
        elif auth_type == "bearer":
            token = auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        return headers

    async def _resolve_redirects(self, url: str, headers: Dict[str, str]) -> str:
        try:
            async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=5.0) as client:
                resp = await client.head(url, headers=headers)
                final_url = str(resp.url)
                if final_url != url:
                    ASCIIColors.yellow(f"Resolved redirect: {url} -> {final_url}")
                return final_url
        except Exception:
            return url

    async def _initialize_connection_async(self, alias: str) -> bool:
        if not MCP_LIBRARY_AVAILABLE or not types or not ClientSession:
            return False
        if self._initialization_status.get(alias): return True
        if alias not in self._server_configs:
            return False

        config = self._server_configs[alias]
        ASCIIColors.info(f"{self.binding_name}: Initializing MCP connection for server '{alias}' ({config['type']})...")
        try:
            if alias in self._exit_stacks:
                old_stack = self._exit_stacks.pop(alias)
                await old_stack.aclose()

            exit_stack = AsyncExitStack()
            self._exit_stacks[alias] = exit_stack

            if config["type"] == "stdio":
                server_params = StdioServerParameters(
                    command=config["command"][0],
                    args=config["command"][1:],
                    cwd=Path(config["cwd"]) if config["cwd"] else None,
                    env=config["env"]
                )
                read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))
            elif config["type"] == "http":
                auth_headers = self._prepare_http_headers(alias)
                final_url = await self._resolve_redirects(config["url"], auth_headers)
                client_streams = await exit_stack.enter_async_context(
                    streamablehttp_client(url=final_url, headers=auth_headers)
                )
                read_stream, write_stream, _ = client_streams
            else:
                raise ValueError(f"Unsupported server type: {config['type']}")

            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            
            if config["type"] == "http":
                handshake_timeout = config.get("timeout", 2.0)
                await asyncio.wait_for(session.initialize(), timeout=handshake_timeout)
            else:
                await session.initialize()

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
                except Exception:
                    pass
            self._initialization_status[alias] = False
            self._mcp_sessions.pop(alias, None)
            return False

    def _ensure_server_initialized_sync(self, alias: str, timeout: float = 30.0):
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
            raise ConnectionError(f"{self.binding_name}: MCP library/event loop not available.")

        if alias not in self._server_configs:
            raise ValueError(f"{self.binding_name}: Server alias '{alias}' is not configured.")

        lock = self._server_locks.get(alias)
        if not lock:
            self._server_locks[alias] = threading.Lock()
            lock = self._server_locks[alias]

        with lock:
            if not self._initialization_status.get(alias):
                try:
                    success = self._run_async_task(self._initialize_connection_async(alias), timeout=timeout)
                    if not success: 
                        self._discovered_tools_cache[alias] = []
                        raise ConnectionError(f"MCP init for '{alias}' reported failure.")
                except TimeoutError:
                    self._discovered_tools_cache[alias] = []
                    raise ConnectionError(f"MCP init for '{alias}' timed out.")
                except Exception as e:
                    self._discovered_tools_cache[alias] = []
                    raise ConnectionError(f"MCP init for '{alias}' failed: {e}")
        
        if not self._initialization_status.get(alias) or alias not in self._mcp_sessions:
            self._discovered_tools_cache[alias] = []
            raise ConnectionError(f"MCP Session for '{alias}' not valid post-init attempt.")

    async def _refresh_tools_cache_async(self, alias: str):
        if not MCP_LIBRARY_AVAILABLE or not types:
            return
        if not self._initialization_status.get(alias) or alias not in self._mcp_sessions:
            return

        session = self._mcp_sessions[alias]
        try:
            list_tools_result = await session.list_tools()
            current_server_tools = []
            if list_tools_result and list_tools_result.tools:
                for tool_obj in list_tools_result.tools:
                    input_schema_dict = {}
                    tool_input_schema = getattr(tool_obj, 'inputSchema', getattr(tool_obj, 'input_schema', None))
                    if tool_input_schema:
                        if hasattr(tool_input_schema, 'model_dump'):
                            input_schema_dict = tool_input_schema.model_dump(mode='json', exclude_none=True)
                        elif isinstance(tool_input_schema, dict):
                            input_schema_dict = tool_input_schema

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
            self._discovered_tools_cache[alias] = []

    def discover_tools(self, specific_tool_names: Optional[List[str]]=None, force_refresh: bool=False, timeout_per_server: float=10.0, **kwargs) -> List[Dict[str, Any]]:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
            return []

        all_tools: List[Dict[str, Any]] = []
        active_aliases = list(self._server_configs.keys())

        for alias in active_aliases:
            try:
                if force_refresh:
                    self._discovered_tools_cache[alias] = [] 
                
                self._ensure_server_initialized_sync(alias, timeout=timeout_per_server)

                if force_refresh or (self._initialization_status.get(alias) and not self._discovered_tools_cache.get(alias)):
                    self._run_async_task(self._refresh_tools_cache_async(alias), timeout=timeout_per_server)

                for tool_data in self._discovered_tools_cache.get(alias, []):
                    prefixed_tool_data = tool_data.copy()
                    prefixed_tool_data["name"] = f"{alias}{TOOL_NAME_SEPARATOR}{tool_data['name']}"
                    all_tools.append(prefixed_tool_data)
            except ConnectionError as e:
                ASCIIColors.error(f"{self.binding_name}: Connection problem with server '{alias}' during discovery: {e}")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"{self.binding_name}: Unexpected problem with server '{alias}' during discovery: {e}")

        if specific_tool_names:
            return [t for t in all_tools if t.get("name") in specific_tool_names]
        return all_tools

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        return self.discover_tools(**kwargs)
    
    def _parse_tool_name(self, prefixed_tool_name: str) -> Optional[Tuple[str, str]]:
        parts = prefixed_tool_name.split(TOOL_NAME_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        ASCIIColors.warning(f"{self.binding_name}: Tool name '{prefixed_tool_name}' is not in 'alias{TOOL_NAME_SEPARATOR}tool' format.")
        return None

    async def _execute_tool_async(self, server_alias: str, actual_tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not MCP_LIBRARY_AVAILABLE or not types:
            return {"error": "MCP library not available.", "status_code": 503}

        if not self._initialization_status.get(server_alias) or server_alias not in self._mcp_sessions:
            return {"error": f"Server '{server_alias}' not initialized.", "status_code": 503}

        session = self._mcp_sessions[server_alias]
        try:
            mcp_call_result = await session.call_tool(name=actual_tool_name, arguments=params)

            output_parts = []
            if mcp_call_result and mcp_call_result.content:
                for content_part in mcp_call_result.content:
                    if isinstance(content_part, types.TextContent) and hasattr(content_part, 'text') and content_part.text is not None:
                        output_parts.append(content_part.text)

            if not output_parts:
                return {"output": {"message": "Tool executed but returned no textual content."}, "status_code": 200}

            combined_output_str = "\n".join(output_parts)
            try:
                parsed_output = json.loads(combined_output_str)
                return {"output": parsed_output, "status_code": 200}
            except json.JSONDecodeError:
                return {"output": combined_output_str, "status_code": 200}

        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing tool '{actual_tool_name}' on '{server_alias}': {str(e)}", "status_code": 500}

    def execute_tool(self, tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not MCP_LIBRARY_AVAILABLE or not self._loop or not types:
            return {"error": "MCP support not available.", "status_code": 503}

        timeout = float(kwargs.get('timeout', 60.0))
        parsed_name = self._parse_tool_name(tool_name)
        if not parsed_name:
            return {"error": f"Invalid tool name format: '{tool_name}'.", "status_code": 400}

        server_alias, actual_tool_name = parsed_name

        if server_alias not in self._server_configs:
            return {"error": f"Server alias '{server_alias}' is not configured.", "status_code": 404}

        try:
            init_timeout = min(timeout, 30.0)
            self._ensure_server_initialized_sync(server_alias, timeout=init_timeout)
        except ConnectionError as e:
            return {"error": f"Connection issue for server '{server_alias}': {e}", "status_code": 503}
        except Exception as e:
            trace_exception(e)
            return {"error": f"Failed to ensure server '{server_alias}' is initialized: {e}", "status_code": 500}

        try:
            return self._run_async_task(self._execute_tool_async(server_alias, actual_tool_name, params), timeout=timeout)
        except TimeoutError:
            return {"error": f"Tool '{actual_tool_name}' on server '{server_alias}' timed out.", "status_code": 504}
        except Exception as e:
            trace_exception(e)
            return {"error": f"Unexpected error running MCP tool '{actual_tool_name}': {e}", "status_code": 500}

    def close(self):
        ASCIIColors.info(f"{self.binding_name}: Initiating shutdown process...")
        if hasattr(self, '_server_configs') and self._server_configs:
            active_aliases = list(self._server_configs.keys())
            for alias in active_aliases:
                self.remove_server(alias, silent=True)

        if hasattr(self, '_loop') and self._loop:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)

        ASCIIColors.info(f"{self.binding_name}: Binding closed.")

    def __del__(self):
        needs_close = False
        if hasattr(self, '_loop') and self._loop and (self._loop.is_running() or not self._loop.is_closed()):
            needs_close = True
        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            needs_close = True
        if hasattr(self, '_server_configs') and self._server_configs:
            needs_close = True

        if needs_close:
            try:
                self.close()
            except Exception:
                pass