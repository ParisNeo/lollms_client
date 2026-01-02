import asyncio
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from lollms_client.lollms_mcp_binding import LollmsMCPBinding
from ascii_colors import ASCIIColors, trace_exception
import threading
import json
import pipmaster as pm

# Ensure required packages
try:
    pm.ensure_packages(["mcp", "httpx"])
    from mcp import ClientSession, types
    from mcp.client.streamable_http import streamablehttp_client
    import httpx
    MCP_LIBRARY_AVAILABLE = True
except ImportError:
    MCP_LIBRARY_AVAILABLE = False
    ClientSession = None
    streamablehttp_client = None
    httpx = None


BindingName = "RemoteMCPBinding"
TOOL_NAME_SEPARATOR  = "::"

class RemoteMCPBinding(LollmsMCPBinding):
    """
    This binding allows the connection to one or more remote MCP servers.
    Tools from all connected servers are aggregated and prefixed with the server's alias.
    """
    def __init__(self,
                 **kwargs: Any
                 ):
        """
        Initializes the binding to connect to multiple MCP servers.
        """
        super().__init__(binding_name="remote_mcp")
        ASCIIColors.info(f"[{self.binding_name}] Initializing RemoteMCPBinding...")
        
        servers_infos: Dict[str, Dict[str, Any]] = kwargs.get("servers_infos", {})
        self.servers = None
        if not MCP_LIBRARY_AVAILABLE:
            ASCIIColors.error(f"{self.binding_name}: MCP library or httpx not available. This binding will be disabled.")
            return

        if not servers_infos or not isinstance(servers_infos, dict):
            self.servers = {}
        else:
            self.config = {
                "servers_infos": kwargs.get("servers_infos"),
                **kwargs
            }

            self.servers: Dict[str, Dict[str, Any]] = {}
            for alias, info in servers_infos.items():
                if "server_url" not in info:
                    ASCIIColors.warning(f"{self.binding_name}: Skipping server '{alias}' due to missing 'server_url'.")
                    continue
                
                self.servers[alias] = {
                    "url": info["server_url"],
                    "auth_config": info.get("auth_config", {}),
                    "timeout": float(info.get("timeout", 2.0)), # NEW: Read configured timeout
                    "session": None,
                    "exit_stack": None,
                    "initialized": False,
                    "initializing_lock": threading.Lock()
                }

        self._discovered_tools_cache: List[Dict[str, Any]] = []

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_started_event = threading.Event()

        if self.servers:
            self._start_event_loop_thread()

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
             import time
             time.sleep(0.1)
             if not self._loop or not self._loop.is_running():
                raise RuntimeError(f"{self.binding_name}: Event loop is not running after start signal.")

    def _run_async(self, coro, timeout=None):
        if not self._loop: 
            self._start_event_loop_thread()
            self._wait_for_loop()

        if not self._loop or not self._loop.is_running(): 
            raise RuntimeError("Event loop not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def _prepare_headers(self, alias: str) -> Dict[str, str]:
        """Prepares the headers dictionary from the server's auth_config."""
        server_info = self.servers[alias]
        auth_config = server_info.get("auth_config", {})
        server_url = server_info.get("url", "")
        
        origin = ""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(server_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
        except:
            pass

        # Standard browser headers
        # NOTE: We DO NOT set 'Accept' here manually to avoid conflicting with mcp library's default
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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
                ASCIIColors.info(f"[{alias}] Auth: API Key ({header_name})")

        elif auth_type == "bearer":
            token = auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                ASCIIColors.info(f"[{alias}] Auth: Bearer Token")
        
        # Debug Log: Print headers (masking sensitive values)
        debug_headers = headers.copy()
        for k, v in debug_headers.items():
            if "key" in k.lower() or "token" in k.lower() or "auth" in k.lower():
                debug_headers[k] = f"{v[:4]}...{v[-4:]}" if len(v) > 8 else "***"
        
        # ASCIIColors.yellow(f"[{alias}] DEBUG: Headers: {json.dumps(debug_headers)}")
        return headers

    async def _resolve_redirects(self, url: str, headers: Dict[str, str]) -> str:
        """Resolves any HTTP redirects to find the final SSE endpoint."""
        try:
            # We use a stream request to follow redirects without downloading body
            async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=5.0) as client:
                # We use GET because SSE connects via GET
                resp = await client.head(url, headers=headers)
                if resp.status_code in [301, 302, 307, 308]:
                    # Should be handled by follow_redirects=True, but checking url just in case
                    pass
                final_url = str(resp.url)
                if final_url != url:
                    ASCIIColors.yellow(f"Resolved redirect: {url} -> {final_url}")
                return final_url
        except Exception as e:
            ASCIIColors.warning(f"Redirect resolution failed for {url}: {e}. Using original URL.")
            return url

    async def _initialize_connection_async(self, alias: str) -> bool:
        server_info = self.servers[alias]
        if server_info["initialized"]:
            return True
            
        initial_url = server_info["url"]
        handshake_timeout = server_info["timeout"]
        ASCIIColors.cyan(f"[{alias}] Connecting to {initial_url}...")
        
        try:
            auth_headers = self._prepare_headers(alias)

            # Resolve Redirects (Critical for some hosts)
            final_url = await self._resolve_redirects(initial_url, auth_headers)

            exit_stack = AsyncExitStack()
            
            # Open the real connection using final URL
            client_streams = await exit_stack.enter_async_context(
                streamablehttp_client(url=final_url, headers=auth_headers) 
            )

            read_stream, write_stream, _ = client_streams
            
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Enforce strict timeout on handshake
            try:
                await asyncio.wait_for(session.initialize(), timeout=handshake_timeout)
            except asyncio.TimeoutError:
                ASCIIColors.red(f"[{alias}] Handshake timed out after {handshake_timeout}s.")
                raise TimeoutError("MCP Handshake Timeout")

            server_info["session"] = session
            server_info["exit_stack"] = exit_stack
            server_info["initialized"] = True

            ASCIIColors.success(f"[{alias}] Ready.")
            return True
            
        except Exception as e:
            ASCIIColors.red(f"[{alias}] CONNECTION FAILED: {e}")
            
            if 'exit_stack' in locals() and exit_stack:
                await exit_stack.aclose()
            
            server_info["session"] = None
            server_info["exit_stack"] = None
            server_info["initialized"] = False
            return False

    def _ensure_initialized_sync(self, alias: str, timeout=30.0):
        self._wait_for_loop() 
        
        server_info = self.servers.get(alias)
        if not server_info:
            raise ValueError(f"Unknown server alias: '{alias}'")

        with server_info["initializing_lock"]:
            if not server_info["initialized"]:
                success = self._run_async(self._initialize_connection_async(alias), timeout=timeout + 5)
                if not success:
                    raise ConnectionError(f"Failed to initialize remote MCP connection to '{alias}'")
        
        if not server_info.get("session"):
             raise ConnectionError(f"MCP Session not valid after init attempt for '{alias}'")

    async def _refresh_all_tools_cache_async(self):
        ASCIIColors.info(f"{self.binding_name}: Refreshing tools...")
        all_tools = []
        
        for alias in self.servers.keys():
            try:
                tools = await self._fetch_tools_from_server_async(alias)
                all_tools.extend(tools)
            except Exception as e:
                ASCIIColors.error(f"Failed to refresh {alias}: {e}")

        self._discovered_tools_cache = all_tools
        ASCIIColors.green(f"{self.binding_name}: Found {len(all_tools)} tools total.")

    async def _fetch_tools_from_server_async(self, alias: str) -> List[Dict[str, Any]]:
        server_info = self.servers[alias]
        if not server_info["initialized"] or not server_info["session"]:
            return []
        
        try:
            list_tools_result = await asyncio.wait_for(server_info["session"].list_tools(), timeout=30.0)
            
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
            ASCIIColors.info(f"[{alias}] Found {len(server_tools)} tools.")
            return server_tools
        except asyncio.TimeoutError:
            ASCIIColors.red(f"[{alias}] Tool listing timed out.")
            return []
        except Exception as e:
            ASCIIColors.red(f"[{alias}] Error listing tools: {e}")
            return []


    def discover_tools(self, force_refresh: bool = False, timeout_per_server: float = 30.0, **kwargs) -> List[Dict[str, Any]]:
        if not self.servers:
            return []

        for alias in self.servers.keys():
            try:
                self._ensure_initialized_sync(alias, timeout=timeout_per_server)
            except Exception as e:
                ASCIIColors.warning(f"[{alias}] Discovery skipped (connection error).")
        
        try:
            if force_refresh or not self._discovered_tools_cache:
                self._run_async(self._refresh_all_tools_cache_async(), timeout=timeout_per_server * len(self.servers))
            return self._discovered_tools_cache
        except Exception as e:
            trace_exception(e)
            return []
            
    async def _execute_tool_async(self, alias: str, actual_tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        server_info = self.servers[alias]
        
        if not server_info["initialized"] or not server_info["session"]:
            return {"error": f"Not connected to server '{alias}'", "status_code": 503}
        
        ASCIIColors.info(f"[{alias}] Executing '{actual_tool_name}' with params: {json.dumps(params)}")
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

    def execute_tool(self, tool_name_with_alias: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        timeout = float(kwargs.get('timeout', 60.0))
        
        if TOOL_NAME_SEPARATOR not in tool_name_with_alias:
            return {"error": f"Invalid tool name format. Expected 'alias{TOOL_NAME_SEPARATOR}tool_name'", "status_code": 400}

        alias, actual_tool_name = tool_name_with_alias.split(TOOL_NAME_SEPARATOR, 1)

        if alias not in self.servers:
            return {"error": f"Unknown server alias '{alias}'.", "status_code": 400}

        try:
            self._ensure_initialized_sync(alias, timeout=timeout)
            return self._run_async(self._execute_tool_async(alias, actual_tool_name, params), timeout=timeout)
        except (ConnectionError, RuntimeError) as e:
            return {"error": f"{self.binding_name}: Connection issue for server '{alias}': {e}", "status_code": 503}
        except TimeoutError:
            return {"error": f"{self.binding_name}: Remote tool '{actual_tool_name}' on '{alias}' timed out.", "status_code": 504}
        except Exception as e:
            trace_exception(e)
            return {"error": f"{self.binding_name}: Failed to run remote MCP tool: {e}", "status_code": 500}

    def close(self):
        ASCIIColors.info(f"{self.binding_name}: Closing all remote connections...")
        
        async def _close_all_connections():
            close_tasks = []
            for alias, server_info in self.servers.items():
                if server_info.get("exit_stack"):
                    close_tasks.append(server_info["exit_stack"].aclose())
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

        if self._loop and self._loop.is_running():
            try:
                self._run_async(_close_all_connections(), timeout=10.0)
            except Exception as e:
                ASCIIColors.error(f"{self.binding_name}: Error during async close: {e}")
        
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
        ASCIIColors.info(f"{self.binding_name}: Updating auth_config for server '{alias}'.")
        
        server_info = self.servers.get(alias)
        if not server_info:
            raise ValueError(f"Server alias '{alias}' does not exist in the configuration.")

        server_info["config"]["auth_config"] = auth_config

        if server_info["initialized"]:
            ASCIIColors.warning(f"{self.binding_name}: Resetting connection for '{alias}' due to auth update.")
            try:
                self._run_async(self._close_connection_async(alias), timeout=10.0)
            except Exception as e:
                ASCIIColors.error(f"{self.binding_name}: Error closing connection: {e}")
                server_info.update({"session": None, "exit_stack": None, "initialized": False})

    async def _close_connection_async(self, alias: str):
        server_info = self.servers.get(alias)
        if not server_info or not server_info.get("exit_stack"):
            return

        ASCIIColors.info(f"{self.binding_name}: Closing connection for '{alias}'...")
        try:
            await server_info["exit_stack"].aclose()
        except Exception as e:
            trace_exception(e)
        finally:
            server_info.update({
                "session": None,
                "exit_stack": None,
                "initialized": False
            })