---
name: Lollms Communication Protocol (LCP) and Local Smart Tools
description: Teaches how to build, discover, and execute local, zero-configuration Python tools in Lollms using AST schema parsing and context injection. Includes step-by-step tool file construction guide and code examples.
author: ParisNeo
version: 1.1.0
category: lollms_client/lcp_tools
created: 2026-05-25
---

# Lollms Communication Protocol (LCP) and Local Smart Tools

This skill explains how to write, configure, and dynamically execute local Python-based smart tools using the Lollms Communication Protocol (LCP) binding.

## 1. The Core Philosophy of LCP
Unlike conventional tool-calling systems (such as standard OpenAI function calling or remote MCP servers) that require duplicate, hard-to-maintain JSON schema descriptors, LCP is a **zero-configuration** local execution framework. 

LCP uses Python's standard `ast` (Abstract Syntax Tree) module to automatically extract:
- **Tool Name**: Taken directly from the Python file stem (e.g., `get_weather.py` defines the tool `get_weather`).
- **Tool Description**: Extracted from the main function's docstring.
- **Input Parameters & Types**: Parsed directly from Python type annotations (e.g. `count: int`).
- **Default Values**: Captured from the function arguments (e.g. `unit: str = "celsius"`).
- **Mandatory Fields**: Derived by identifying parameters that lack default values.

This means you only write standard, clean Python code. The LCP engine builds the compliant LLM function schema on-the-fly at runtime!

## 2. Structural Conventions of an LCP Tool
An LCP tool is defined by a standalone Python file (`.py`). The file can contain multiple auxiliary functions, but must expose one main entry point function.

### Entry Point Naming Convention
LCP scans the file's AST and binds the first function matching any of the following patterns (in order of priority):
1. `tool_[tool_name]` (e.g. `tool_file_compressor` inside `file_compressor.py`).
2. `execute` (e.g. `execute` inside `file_compressor.py`).
3. Any callable starting with `tool_` (e.g. `tool_compress` inside `file_compressor.py`).

### Supported Type Annotations
LCP maps Python type-hints to standard JSON-schema data types:
- `int` / `integer` → `"integer"`
- `float` / `number` → `"number"`
- `bool` / `boolean` → `"boolean"`
- `list` / `array` → `"array"`
- `dict` / `object` → `"object"`
- `str` / `string` (or unannotated) → `"string"`

---

## 3. How to Build an LCP Tool File (Anatomy)

Every standalone LCP tool file follows a simple, 3-part layout:

### Part 1: Metadata Declarations (Header)
Define the UI-facing identity of your tool. These variables are read on startup by the LCP manager:
* `TOOL_LIBRARY_NAME`: The user-friendly title of the tool.
* `TOOL_LIBRARY_DESC`: A brief summary of the tool library.
* `TOOL_LIBRARY_ICON`: A single emoji acting as the avatar.

```python
TOOL_LIBRARY_NAME = "Network Scanner"
TOOL_LIBRARY_DESC = "Inspect local subnets for open ports and services."
TOOL_LIBRARY_ICON = "🌐"
```

### Part 2: Installation Hook (`init_tools_library`)
This function is invoked by the client during initialization. Use it to check and auto-install any third-party dependencies required by the script using `pipmaster`. This keeps the script fully portable.

```python
def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({"scapy": ">=2.5.0"})
```

### Part 3: Main Execution Function (`tool_<name>` or `execute`)
The core function that runs the tool. It must have:
1. Clear type annotations for parameters and return values.
2. A detailed **docstring** structured with parameter descriptions. The LCP engine extracts this block directly as the prompt description for the LLM!

```python
def tool_network_scanner(ip_range: str, timeout: float = 2.0) -> dict:
    """
    Scans a local IP range for active hosts and open ports.

    Args:
        ip_range (str): The target subnet in CIDR notation (e.g., '192.168.1.0/24').
        timeout (float, optional): Connection timeout in seconds. Defaults to 2.0.
    """
    # Logic goes here...
```

---

## 4. Complete, Functional Code Example: `web_scraper.py`

Below is a complete, production-ready example of a web scraping tool. It handles downloading its own dependencies (`beautifulsoup4`), parses a target URL, and returns a clean, structured JSON response.

```python
# data_workspace/web_scraper.py
# Standalone LCP Smart Tool

import urllib.request
from typing import Optional

# ── Part 1: Metadata ──
TOOL_LIBRARY_NAME = "HTML Web Scraper"
TOOL_LIBRARY_DESC = "Fetches a URL and extracts readable plain text, headers, and metadata."
TOOL_LIBRARY_ICON = "🕸️"

# ── Part 2: Dependency Verification ──
def init_tools_library() -> None:
    """Ensure beautifulsoup4 is present in the active virtual environment."""
    import pipmaster as pm
    pm.ensure_packages("beautifulsoup4")

# ── Part 3: Execution Logic ──
def tool_web_scraper(
    url: str,
    max_paragraphs: int = 10,
    include_headings: bool = True
) -> dict:
    """
    Scrapes and extracts semantic readable text content from any public webpage.

    Args:
        url (str): The complete target URL to scrape (must start with http:// or https://).
        max_paragraphs (int, optional): Maximum paragraphs of text to extract. Defaults to 10.
        include_headings (bool, optional): Whether to also extract h1, h2, h3 headings. Defaults to True.
    """
    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "Invalid URL format. Protocol (http/https) is required."}

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"success": False, "error": "BeautifulSoup4 was not installed correctly by init_tools_library."}

    try:
        # Fetch raw HTML using python's standard urllib
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()

        soup = BeautifulSoup(html, 'html.parser')

        # Strip script and style blocks
        for element in soup(["script", "style", "noscript", "iframe"]):
            element.decompose()

        result = {
            "success": True,
            "url": url,
            "title": soup.title.string.strip() if soup.title else "Untitled Page",
            "headings": [],
            "paragraphs": []
        }

        # Extract headings if requested
        if include_headings:
            for h in soup.find_all(["h1", "h2", "h3"])[:10]:
                text = h.get_text().strip()
                if text:
                    result["headings"].append({"tag": h.name, "text": text})

        # Extract paragraphs
        paragraph_count = 0
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if len(text) > 15:  # skip empty or short elements
                result["paragraphs"].append(text)
                paragraph_count += 1
                if paragraph_count >= max_paragraphs:
                    break

        return result

    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"Scraping request failed: {str(e)}"
        }
```

---

## 5. Context Injection (With LollmsClient & Discussion)
If your tool needs to access active session details, query other models, or write user artifacts, declare the context parameters in the function signature:

```python
from typing import Optional, Any

def tool_file_analyzer(
    file_name: str,
    lollms_client_instance: Optional[Any] = None,
    discussion_instance: Optional[Any] = None
) -> dict:
    """
    Analyzes a file and logs details back to the active conversation.

    Args:
        file_name (str): Path or name of the file to inspect.
    """
    # lollms_client_instance and discussion_instance are automatically injected on execution
    if discussion_instance:
        discussion_instance.add_message(
            sender="system",
            content=f"Starting analysis on file: {file_name}"
        )
    return {"status": "Analysis logged."}
```

## 6. Binding Configuration and Discovery
LCP scans directories and files defined in the LollmsClient configuration. You can scan multiple folders simultaneously or load standalone script files.

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gemma4:e2b"},
    tools_binding_name="lcp",
    tools_binding_config={
        "tools_folders": [
            "./data_workspace",                      # Ingest custom personality tools
            "./lollms_client/tools_bindings/lcp/default_tools" # Ingest default system tools
        ],
        "tool_files": [
            "C:/shared_libs/network_scanners.py"    # Ingest direct standalone files
        ]
    }
)

# List all dynamically parsed schemas
discovered_tools = client.tools.list_tools()
for tool in discovered_tools:
    print(f"Tool Name: {tool['name']}")
    print(f"Properties: {list(tool['input_schema']['properties'].keys())}")
```

## 7. Multimodal Tool Orchestration (TTI + VLM Verification)

When a task requires **generating an image and then verifying it**, the agent MUST emit tool calls immediately — never narrate intent first. The correct pattern is a two-step tool chain in a single turn:

### Step 1 — Generate (TTI)
Call the image generation tool exposed by the active TTI binding (e.g. `generate_image`). The tool returns a file path or base64 payload.

### Step 2 — Verify (VLM)
Immediately call `vlm_query` (or the equivalent vision tool) with the generated image path and a compliance prompt. This closes the loop and proves the image matches the request.

### Canonical Example
```python
# Step 1: generate
result = tool_generate_image(
    prompt="Cyber Monalisa, neon circuitry, holographic smile, dark background",
    width=1024,
    height=1024,
)
image_path = result["image_path"]

# Step 2: verify (same turn, no preamble)
verification = tool_vlm_query(
    image_path=image_path,
    question="Does this image depict a cyberpunk Mona Lisa? Describe the subject, style, and any compliance issues.",
)
```

### Anti-Preamble Rule
- **DO NOT** write "I will now generate an image..." — emit the tool call directly.
- **DO NOT** wait for user confirmation between generate and verify.
- **ALWAYS** terminate the turn with `<done/>` after the verification tool returns.

---

## 8. llama_cpp_server Binding Pitfalls

The `llama_cpp_server` binding spawns `llama-server.exe` with a set of flags. Two flags are **mutually exclusive** with chat completions and will cause the agent to receive empty responses (triggering the stall detector):

### `--embedding` (DO NOT use for chat)
When `--embedding` is passed, `llama-server` runs in **embedding-only mode**. Chat completion requests return an embedding vector (or empty content), and the agent sees no `<done/>` → stall loop.

**Symptom**: `No <done/> or <end/> detected` on every round, even for `hi`.

**Fix**: Remove `--embedding` from the server launch flags when the model is used for chat. Only enable it for a dedicated embedding server instance.

### `--reasoning off` + `think=False`
If the model is a reasoning model (e.g. Qwen3), passing `--reasoning off` while the binding sends `think=False` can produce empty content. Prefer letting the binding auto-detect reasoning support, or explicitly set `reasoning_effort` to a valid value.

### Diagnostic Checklist
1. Check the server launch log for `--embedding` — remove it for chat.
2. Verify `curl http://localhost:<port>/v1/chat/completions` returns a non-empty `choices[0].message.content`.
3. If empty, check the server's `_error.log` for context-length or template errors.

---

## 9. Summary of Best Practices
- **Never nest `<artifact>` inside markdown code blocks** when writing tools, let the XML render directly.
- **Always provide a detailed docstring** for your main tool function, as the LCP engine extracts it directly as the LLM-facing description.
- **Use `init_tools_library()`** inside your tool scripts to run a pipmaster check ensuring all required third-party libraries (e.g. `websockets`, `pandas`, `beautifulsoup4`) are installed automatically on startup.
- **For multimodal tasks, chain tools in a single turn** (generate → verify) and emit `<done/>` immediately after the last tool result.
