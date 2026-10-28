"""Web Search & Fetch Toolset for LCP.

Provides web search and URL content fetching capabilities conforming to the 
LCP authoring guide (Seam A, workspace-anchored, never-raising init).

Requirements: standard library only (urllib, html.parser, json).
"""

from pathlib import Path
import json
import urllib.request
import urllib.parse
import html.parser
import re

TOOL_LIBRARY_LOAD_FLAG = "enable_websearch"

_WORKSPACE = Path(".")


def init_tools_library(config=None):
    """Never raises: dependency checks or configuration loading soft-fail."""
    pass


def _resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a workspace-relative reference; reject absolute paths and '..'."""
    cleaned = Path(str(relative_path).strip())
    if cleaned.is_absolute() or ".." in cleaned.parts:
        raise ValueError(
            "Absolute paths and '..' are forbidden. Use a workspace-relative path."
        )
    return (_WORKSPACE / cleaned).resolve()


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Simple HTML parser to strip tags and extract plain text."""
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text.append(text)

    def get_text(self):
        return " ".join(self.text)


def tool_web_search(query: str, max_results: int = 5, save_to_file: str = ""):
    """Perform a web search and return relevant search snippets and sources.

    Args:
        query (str): The search query string.
        max_results (int, optional): Maximum number of results to return (default 5).
        save_to_file (str, optional): Workspace-relative filename to save results as JSON.
    """
    try:
        query = query.strip()
        if not query:
            return {"success": False, "error": "Search query cannot be empty."}

        # Using DuckDuckGo HTML endpoint via standard library as a zero-dependency fallback,
        # or checking for configured API providers (Tavily/Exa) if present in environment.
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Compatible; LCP-OpenCodeWebSearch/1.0)"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode("utf-8", errors="ignore")

        # Basic regex extraction for DuckDuckGo HTML results
        # Pattern matches result snippets and links
        result_blocks = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html_content, re.DOTALL)
        title_blocks = re.findall(r'<a class="result__url"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        link_blocks = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"', html_content, re.DOTALL)

        results = []
        parser = _HTMLTextExtractor()

        for i in range(min(max_results, len(result_blocks))):
            parser.feed(result_blocks[i])
            snippet = parser.get_text()
            parser.text.clear()

            title = title_blocks[i] if i < len(title_blocks) else "No Title"
            link = link_blocks[i] if i < len(link_blocks) else ""

            results.append({
                "index": i + 1,
                "title": title.strip(),
                "snippet": snippet,
                "url": link
            })

        if not results:
            return {
                "success": True,
                "output": f"Web search executed for '{query}', but no structured HTML results were parsed.",
                "results": []
            }

        output_text = f"Found {len(results)} results for query: '{query}'\n\n"
        for r in results:
            output_text += f"[{r['index']}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n\n"

        context_files = []
        if save_to_file:
            out_path = _resolve_workspace_path(save_to_file)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            context_files.append(out_path.name)
            output_text += f"Saved search results to workspace file: {out_path.name}"

        return {
            "success": True,
            "output": output_text.strip(),
            "context_files": context_files
        }

    except Exception as ex:
        return {
            "success": False,
            "error": f"Web search failed: {str(ex)}"
        }


def tool_web_fetch(url: str, save_to_file: str = ""):
    """Retrieve and read text content from a specific web URL (webfetch).

    Args:
        url (str): The full http/https URL to fetch.
        save_to_file (str, optional): Workspace-relative filename to save fetched page content.
    """
    try:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return {"success": False, "error": "Invalid URL. Must start with http:// or https://."}

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Compatible; LCP-OpenCodeWebFetch/1.0)"}
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            html_content = response.read().decode("utf-8", errors="ignore")

        # Strip HTML tags to get clean readable text for the LLM
        parser = _HTMLTextExtractor()
        parser.feed(html_content)
        page_text = parser.get_text()

        # Truncate if extremely long to respect context windows (~15,000 chars max)
        if len(page_text) > 15000:
            page_text = page_text[:15000] + "\n[Content truncated due to length...]"

        context_files = []
        if save_to_file:
            out_path = _resolve_workspace_path(save_to_file)
            out_path.write_text(page_text, encoding="utf-8")
            context_files.append(out_path.name)

        return {
            "success": True,
            "output": f"Successfully fetched content from {url}:\n\n{page_text}",
            "file_name": save_to_file if save_to_file else None,
            "context_files": context_files
        }

    except Exception as ex:
        return {
            "success": False,
            "error": f"Web fetch failed for {url}: {str(ex)}"
        }