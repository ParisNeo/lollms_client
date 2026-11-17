"""Web Search & Fetch Toolset for LCP.

Provides web search and URL content fetching capabilities conforming to the 
LCP authoring guide (Seam A, workspace-anchored, never-raising init).

Requirements: standard library only (urllib, html.parser, json).
"""

from pathlib import Path
import json
import time
import urllib.request
import urllib.parse
import html.parser
import re

__lcp_tool_binding__ = True  # Explicit marker for binding detection
TOOL_LIBRARY_LOAD_FLAG = "enable_websearch"

_WORKSPACE = Path(".")

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Class presence checked via lookahead so attribute order (class↔href) never matters.
_RESULT_TITLE_HREF = re.compile(
    r'<a(?=[^>]*\bclass="result__a")[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_RESULT_SNIPPET = re.compile(
    r'<a(?=[^>]*\bclass="result__snippet")[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_LITE_LINK = re.compile(
    r'<a(?=[^>]*\bclass=["\']result-link["\'])[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_LITE_SNIPPET = re.compile(
    r'<td(?=[^>]*\bclass=["\']result-snippet["\'])[^>]*>(.*?)</td>',
    re.DOTALL | re.IGNORECASE,
)


def init_tools_library(config=None):
    """Never raises: dependency checks or configuration loading soft-fail."""
    try:
        # Any future config loading here
        pass
    except Exception:
        pass  # Soft-fail per LCP contract


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


def _clean_fragment(fragment: str) -> str:
    """Strip tags from an HTML fragment using a FRESH parser instance."""
    parser = _HTMLTextExtractor()
    parser.feed(fragment)
    parser.close()
    return html.parser.unescape(parser.get_text()).strip()


def _unwrap_ddg_link(href: str) -> str:
    """Unwrap DuckDuckGo redirect links to the real target URL."""
    href = html.parser.unescape(href)
    if "uddg=" in href:
        try:
            parsed = urllib.parse.urlparse(href)
            query = urllib.parse.parse_qs(parsed.query)
            if query.get("uddg"):
                return urllib.parse.unquote(query["uddg"][0])
        except Exception:
            pass
    if href.startswith("//"):
        return "https:" + href
    return href


def _ddg_fetch(url: str, data=None) -> str:
    """Fetch a DuckDuckGo endpoint with browser-like headers. Raises on HTTP errors."""
    req = urllib.request.Request(url, data=data, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_html_results(page: str, max_results: int):
    """Extract results from html.duckduckgo.com output, pairing each title
    anchor with the first snippet occurring after it."""
    results = []
    seen_urls = set()
    snippets = list(_RESULT_SNIPPET.finditer(page))
    for match in _RESULT_TITLE_HREF.finditer(page):
        url = _unwrap_ddg_link(match.group(1))
        if url in seen_urls:
            continue
        snippet = ""
        for snip_match in snippets:
            if snip_match.start() > match.start():
                snippet = _clean_fragment(snip_match.group(1))
                break
        results.append({
            "title": _clean_fragment(match.group(2)),
            "url": url,
            "snippet": snippet,
        })
        seen_urls.add(url)
        if len(results) >= max_results:
            break
    return results


def _extract_lite_results(page: str, max_results: int):
    """Extract results from lite.duckduckgo.com output, pairing each link
    with the first snippet occurring after it."""
    results = []
    seen_urls = set()
    snippets = list(_LITE_SNIPPET.finditer(page))
    for match in _LITE_LINK.finditer(page):
        url = _unwrap_ddg_link(match.group(1))
        if url in seen_urls:
            continue
        snippet = ""
        for snip_match in snippets:
            if snip_match.start() > match.start():
                snippet = _clean_fragment(snip_match.group(1))
                break
        results.append({
            "title": _clean_fragment(match.group(2)),
            "url": url,
            "snippet": snippet,
        })
        seen_urls.add(url)
        if len(results) >= max_results:
            break
    return results


def tool_web_search(query: str, max_results: int = 5, save_to_file: str = ""):
    """Perform a web search (DuckDuckGo with fallback endpoints) and return
    relevant search snippets and sources.

    Args:
        query (str): The search query string.
        max_results (int, optional): Maximum number of results to return (default 5).
        save_to_file (str, optional): Workspace-relative filename to save results as JSON.
    """
    try:
        query = query.strip()
        if not query:
            return {"success": False, "error": "Search query cannot be empty."}

        try:
            max_results = max(1, int(max_results))
        except (TypeError, ValueError):
            max_results = 5

        encoded_query = urllib.parse.quote_plus(query)
        post_data = urllib.parse.urlencode({"q": query}).encode("utf-8")

        # 3-attempt chain: GET html → POST html → GET lite.
        results = []
        diagnostics = []

        attempts = [
            ("GET-html", lambda: _ddg_fetch(f"https://html.duckduckgo.com/html/?q={encoded_query}"), _extract_html_results),
            ("POST-html", lambda: _ddg_fetch("https://html.duckduckgo.com/html/", data=post_data), _extract_html_results),
            ("GET-lite", lambda: _ddg_fetch(f"https://lite.duckduckgo.com/lite/?q={encoded_query}"), _extract_lite_results),
        ]

        for name, fetch, extractor in attempts:
            try:
                page = fetch()
                results = extractor(page, max_results)
                if results:
                    break
                diagnostics.append(f"{name}: 0 results parsed")
            except Exception as ex:
                diagnostics.append(f"{name}: {ex}")

        if not results:
            return {
                "success": False,
                "error": (
                    "Web search failed: DuckDuckGo returned no parseable results "
                    "(possible bot-block or markup change). Attempts: " + "; ".join(diagnostics)
                ),
                "results": [],
            }

        for i, r in enumerate(results, start=1):
            r["index"] = i

        date_stamp = time.strftime("%Y-%m-%d")
        output_text = f"Found {len(results)} results for query: '{query}' (searched on {date_stamp})\n\n"
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
            "results": results,
            "context_files": context_files,
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

        req = urllib.request.Request(url, headers=_BROWSER_HEADERS)

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