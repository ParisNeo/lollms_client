"""
tools_bindings — LOLLMS Tool Execution Bindings.

Hosts concrete tool binding implementations. Each sub-package exposes a
`LollmsToolBinding` subclass that the `LollmsToolsBindingManager` can
instantiate by name (e.g. `"lcp"`, `"mcp"`).

Available bindings:
    - lcp : LoLLMS Code Protocol — native tool discovery, gating, and execution
            (default_tools/ ships dozens of built-in toolsets: websearch,
            execute_python, file_organizer, git_manager, computer_use, etc.).
    - mcp : Model Context Protocol — external MCP server bridge with security
            policy enforcement.

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the tools binding manager.
"""