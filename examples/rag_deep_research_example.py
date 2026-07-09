#!/usr/bin/env python3
"""
rag_deep_research_example.py
============================
A deep research agent demonstrating the full RAG pipeline with intelligent caching:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Web Search │────→│   Download  │────→│  Data Lake  │
    │   (Tool 1)  │     │  (Tool 2)   │     │  (Tool 3)   │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   ▲                   │
           │            ┌──────┘                   │
           │            │                          │
           └────────────┤    ┌─────────────┐      │
                        └───→│  Cache Hit  │──────┘
                             │  (data/web) │
                             └─────────────┘

Workflow
--------
1. User poses a complex research question
2. Agent searches the web for relevant sources
3. Agent checks cache in data/web/ — if fresh content exists, reuse it
4. Agent downloads only uncached or stale content
5. Agent chunks and indexes content into a local vector data lake
6. Agent performs semantic RAG queries against the data lake
7. Agent synthesizes a comprehensive, cited answer

Caching Strategy
----------------
- Cache directory: data/web/ (organized by domain subdirectories)
- Cache key: MD5 hash of normalized URL
- Cache file: JSON with metadata + content
- Cache expiration: 24 hours (configurable)
- Cache hit: Skip download, load from disk
- Cache miss: Download, process, store in cache

Requirements
------------
pip install lollms_client ascii_colors requests beautifulsoup4 sentence-transformers numpy

Downloads: ~2.2 GB (Ministral 3B Q4_K_M) + ~500 MB (sentence-transformers model)
"""

import sys
import json
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Ensure the source is importable when running from the repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO_INDEX = 1   # Ministral-3-3B-Instruct-2512

BINDING_CONFIG = {
    "models_path": "data/models/llama_cpp_models",
    "binaries_path": "data/bin/llm/llama_cpp_server",
    "ctx_size": 8192,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "idle_timeout": 300,
}

TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"

# Cache configuration
CACHE_DIR = PROJECT_ROOT / "data" / "web"
CACHE_EXPIRATION_HOURS = 24  # How long cached content remains fresh


# ─────────────────────────────────────────────────────────────────────────────
# Cache Helpers (used by main script and tools)
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_path_for_url(url: str) -> Path:
    """Compute the cache file path for a given URL."""
    # Normalize URL
    normalized = url.strip().rstrip("/").lower()
    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    # Extract domain for subdirectory organization
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or "unknown"
        # Clean domain for filesystem safety
        domain = re.sub(r'[^\w\-\.]', '_', domain)
    except Exception:
        domain = "unknown"
    
    domain_dir = CACHE_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    return domain_dir / f"{url_hash}.json"


def is_cache_fresh(cache_path: Path, max_age_hours: int = CACHE_EXPIRATION_HOURS) -> bool:
    """Check if a cached file exists and is within the freshness window."""
    if not cache_path.exists():
        return False
    
    try:
        mtime = cache_path.stat().st_mtime
        age = datetime.now() - datetime.fromtimestamp(mtime)
        return age < timedelta(hours=max_age_hours)
    except Exception:
        return False


def load_from_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load cached content from disk."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_to_cache(cache_path: Path, data: Dict[str, Any]) -> bool:
    """Save content to cache on disk."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Return statistics about the cache directory."""
    if not CACHE_DIR.exists():
        return {"total_files": 0, "total_size_mb": 0, "domains": []}
    
    total_files = 0
    total_size = 0
    domains = []
    
    for domain_dir in CACHE_DIR.iterdir():
        if domain_dir.is_dir():
            files = list(domain_dir.glob("*.json"))
            domain_size = sum(f.stat().st_size for f in files)
            total_files += len(files)
            total_size += domain_size
            domains.append({
                "domain": domain_dir.name,
                "files": len(files),
                "size_mb": round(domain_size / (1024 * 1024), 2),
            })
    
    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "domains": sorted(domains, key=lambda x: x["size_mb"], reverse=True),
    }


def prune_cache(max_age_days: int = 7) -> Dict[str, Any]:
    """Remove cache entries older than max_age_days."""
    if not CACHE_DIR.exists():
        return {"removed": 0, "freed_mb": 0}
    
    cutoff = datetime.now() - timedelta(days=max_age_days)
    removed = 0
    freed = 0
    
    for domain_dir in CACHE_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        for cache_file in domain_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime < cutoff:
                    size = cache_file.stat().st_size
                    cache_file.unlink()
                    freed += size
                    removed += 1
            except Exception:
                pass
        # Remove empty domain directories
        try:
            if domain_dir.exists() and not any(domain_dir.iterdir()):
                domain_dir.rmdir()
        except Exception:
            pass
    
    return {
        "removed": removed,
        "freed_mb": round(freed / (1024 * 1024), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Lake Implementation (In-Memory Vector Store)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata."""
    id: str
    text: str
    source_url: str
    source_title: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None


class DataLake:
    """
    Simple in-memory vector data lake for RAG.
    Uses sentence-transformers for embeddings and cosine similarity for retrieval.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks: List[DocumentChunk] = []
        self._embedder = None
        self._model_name = model_name
        self._initialized = False
    
    def _ensure_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"📥 Loading embedding model: {self._model_name}")
                self._embedder = SentenceTransformer(self._model_name)
                self._initialized = True
                print(f"✅ Embedding model loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and add documents to the data lake.
        
        Args:
            documents: List of dicts with 'url', 'title', 'content' keys
        
        Returns:
            Dict with status and chunking statistics
        """
        self._ensure_embedder()
        
        total_chunks = 0
        for doc in documents:
            url = doc.get("url", "unknown")
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            
            if not content or len(content) < 50:
                continue
            
            # Chunk the content (simple sentence-based chunking)
            chunks = self._chunk_text(content, max_chunk_size=512, overlap=50)
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{url}:{i}:{chunk_text[:50]}".encode()
                ).hexdigest()[:12]
                
                # Compute embedding
                embedding = self._embedder.encode(
                    chunk_text, 
                    convert_to_tensor=False
                ).tolist()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_url=url,
                    source_title=title,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    embedding=embedding,
                )
                self.chunks.append(chunk)
                total_chunks += 1
        
        return {
            "success": True,
            "documents_added": len(documents),
            "total_chunks": total_chunks,
            "total_chunks_in_lake": len(self.chunks),
        }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Semantic search against the data lake.
        
        Args:
            query_text: The search query
            top_k: Number of top results to return
        
        Returns:
            Dict with matches and relevance scores
        """
        self._ensure_embedder()
        
        if not self.chunks:
            return {
                "success": False,
                "error": "Data lake is empty. Add documents first.",
                "matches": [],
            }
        
        # Encode query
        query_embedding = self._embedder.encode(
            query_text,
            convert_to_tensor=False
        )
        
        # Compute cosine similarities
        import numpy as np
        query_norm = np.linalg.norm(query_embedding)
        
        scored_chunks = []
        for chunk in self.chunks:
            emb = np.array(chunk.embedding)
            chunk_norm = np.linalg.norm(emb)
            if chunk_norm == 0:
                continue
            
            cosine_sim = np.dot(query_embedding, emb) / (query_norm * chunk_norm)
            scored_chunks.append((chunk, float(cosine_sim)))
        
        # Sort by similarity descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_chunks[:top_k]
        
        matches = []
        for chunk, score in top_matches:
            matches.append({
                "chunk_id": chunk.id,
                "text": chunk.text,
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "relevance_score": round(score, 4),
            })
        
        return {
            "success": True,
            "query": query_text,
            "matches_found": len(matches),
            "matches": matches,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Return data lake statistics."""
        sources = set(c.source_url for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "unique_sources": len(sources),
            "embedding_model": self._model_name,
            "sources": list(sources),
        }
    
    @staticmethod
    def _chunk_text(text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Chunk text into overlapping segments.
        Uses sentence boundaries when possible.
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Store current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk)[-overlap:] if overlap > 0 else ''
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length if overlap_text else sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# Global data lake instance (shared across tool calls)
_GLOBAL_DATA_LAKE: Optional[DataLake] = None


def get_or_create_data_lake() -> DataLake:
    """Get or create the global data lake instance."""
    global _GLOBAL_DATA_LAKE
    if _GLOBAL_DATA_LAKE is None:
        _GLOBAL_DATA_LAKE = DataLake()
    return _GLOBAL_DATA_LAKE


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (lollms format)
# ─────────────────────────────────────────────────────────────────────────────

WEB_SEARCH_TOOL = '''TOOL_LIBRARY_NAME = 'Web Search'
TOOL_LIBRARY_DESC = 'Search the internet for relevant URLs and snippets.'
TOOL_LIBRARY_ICON = '🌐'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'requests': '>=2.28.0'})

def tool_web_search(args: dict):
    """
    Search the web for relevant content.

    Args:
        args: dict with keys:
            - query (str): Search query string
            - num_results (int, optional): Number of results to return (default: 5)
    """
    import requests
    import json
    import urllib.parse
    
    try:
        query = args.get('query', '')
        num_results = args.get('num_results', 5)
        
        if not query:
            return "Error: Query is required."
        
        # Use DuckDuckGo HTML scraping (no API key needed)
        # This is a simple implementation for demonstration
        encoded_query = urllib.parse.quote(query)
        
        # Try multiple search approaches
        results = []
        
        # Approach 1: DuckDuckGo Lite
        try:
            url = f"https://duckduckgo.com/html/?q={encoded_query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0"
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Extract results from HTML
                import re
                # Find result links and titles
                result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>'
                matches = re.findall(result_pattern, response.text)
                
                for link, title in matches[:num_results]:
                    # Clean up the URL (DuckDuckGo redirects)
                    clean_url = link
                    if clean_url.startswith("//"):
                        clean_url = "https:" + clean_url
                    
                    # Extract snippet
                    snippet_pattern = r'<a[^>]*href="' + re.escape(link) + r'"[^>]*>.*?</a>.*?<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
                    snippet_match = re.search(snippet_pattern, response.text, re.DOTALL)
                    snippet = snippet_match.group(1) if snippet_match else "No snippet available"
                    
                    results.append({
                        "title": title.strip(),
                        "url": clean_url,
                        "snippet": snippet.strip(),
                    })
        except Exception as e:
            pass
        
        # Fallback: return simulated results if scraping fails
        if not results:
            # For demo purposes, return structured placeholder results
            # In production, you'd use a proper search API (SerpAPI, Bing, etc.)
            return {
                "warning": "Web scraping returned no results. This demo uses a simplified search.",
                "query": query,
                "results": [
                    {
                        "title": f"Result for: {query}",
                        "url": f"https://example.com/search?q={encoded_query}",
                        "snippet": "Please configure a proper search API for production use.",
                    }
                ],
                "note": "For production, integrate with SerpAPI, Google Custom Search, or Bing API."
            }
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
        }
        
    except Exception as e:
        return f"Error during web search: {str(e)}"
'''

CONTENT_DOWNLOAD_TOOL = '''TOOL_LIBRARY_NAME = 'Content Downloader'
TOOL_LIBRARY_DESC = 'Download and extract clean text from web URLs with intelligent caching.'
TOOL_LIBRARY_ICON = '📥'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'requests': '>=2.28.0', 'beautifulsoup4': '>=4.11.0'})

def _get_cache_path(url: str, cache_dir: str) -> str:
    """Compute cache file path for a URL."""
    import hashlib
    import os
    from urllib.parse import urlparse
    
    normalized = url.strip().rstrip("/").lower()
    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    try:
        domain = urlparse(url).netloc or "unknown"
        domain = "".join(c if c.isalnum() or c in "-._" else "_" for c in domain)
    except Exception:
        domain = "unknown"
    
    domain_dir = os.path.join(cache_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    return os.path.join(domain_dir, f"{url_hash}.json")


def _is_cache_fresh(cache_path: str, max_age_hours: int = 24) -> bool:
    """Check if cached file exists and is fresh."""
    import os
    from datetime import datetime, timedelta
    
    if not os.path.exists(cache_path):
        return False
    
    try:
        mtime = os.path.getmtime(cache_path)
        age = datetime.now() - datetime.fromtimestamp(mtime)
        return age < timedelta(hours=max_age_hours)
    except Exception:
        return False


def _load_from_cache(cache_path: str):
    """Load cached content from disk."""
    import json
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_to_cache(cache_path: str, data: dict) -> bool:
    """Save content to cache."""
    import json
    import os
    
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def tool_download_content(args: dict):
    """
    Download content from a URL with intelligent caching.
    Checks data/web/ cache first; only downloads if cache miss or stale.

    Args:
        args: dict with keys:
            - url (str): The URL to download
            - max_length (int, optional): Max characters to return (default: 10000)
            - use_cache (bool, optional): Whether to use caching (default: True)
            - cache_dir (str, optional): Cache directory (default: "data/web")
            - cache_max_age_hours (int, optional): Cache freshness in hours (default: 24)
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    import os
    import sys
    
    try:
        url = args.get('url', '')
        max_length = args.get('max_length', 10000)
        use_cache = args.get('use_cache', True)
        cache_dir = args.get('cache_dir', 'data/web')
        cache_max_age_hours = args.get('cache_max_age_hours', 24)
        
        if not url:
            return "Error: URL is required."
        
        # ── CHECK CACHE FIRST ─────────────────────────────────────────
        if use_cache:
            cache_path = _get_cache_path(url, cache_dir)
            
            if _is_cache_fresh(cache_path, cache_max_age_hours):
                cached = _load_from_cache(cache_path)
                if cached:
                    # Apply max_length truncation to cached content
                    content = cached.get('content', '')
                    original_length = cached.get('original_length', len(content))
                    
                    if len(content) > max_length:
                        content = content[:max_length] + f"... [truncated from {original_length} chars]"
                    
                    return {
                        "success": True,
                        "url": url,
                        "title": cached.get('title', 'Cached Content'),
                        "content": content,
                        "content_length": len(content),
                        "original_length": original_length,
                        "cached": True,
                        "cache_path": cache_path,
                        "downloaded_at": cached.get('downloaded_at', 'unknown'),
                    }
        
        # ── CACHE MISS: DOWNLOAD FRESH ────────────────────────────────
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main'))
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        
        # Clean up whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Truncate if needed
        original_length = len(text)
        if len(text) > max_length:
            text = text[:max_length] + f"... [truncated from {original_length} chars]"
        
        # ── SAVE TO CACHE ─────────────────────────────────────────────
        if use_cache:
            cache_path = _get_cache_path(url, cache_dir)
            cache_data = {
                "url": url,
                "title": title,
                "content": text,
                "original_length": original_length,
                "downloaded_at": __import__('datetime').datetime.now().isoformat(),
                "status_code": response.status_code,
            }
            _save_to_cache(cache_path, cache_data)
        
        return {
            "success": True,
            "url": url,
            "title": title,
            "content": text,
            "content_length": len(text),
            "original_length": original_length,
            "cached": False,
            "status_code": response.status_code,
        }
        
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network error: {str(e)}", "url": url}
    except Exception as e:
        return {"success": False, "error": f"Processing error: {str(e)}", "url": url}


def tool_check_cache(args: dict):
    """
    Check cache statistics or inspect a specific cached URL.

    Args:
        args: dict with keys:
            - action (str): "stats" or "inspect"
            - url (str, optional): URL to inspect (required for action="inspect")
            - cache_dir (str, optional): Cache directory (default: "data/web")
    """
    import os
    import json
    from datetime import datetime, timedelta
    
    try:
        action = args.get('action', 'stats')
        cache_dir = args.get('cache_dir', 'data/web')
        
        if action == 'stats':
            if not os.path.exists(cache_dir):
                return {"success": True, "total_files": 0, "total_size_mb": 0, "domains": []}
            
            total_files = 0
            total_size = 0
            domains = []
            
            for domain_dir in os.listdir(cache_dir):
                domain_path = os.path.join(cache_dir, domain_dir)
                if not os.path.isdir(domain_path):
                    continue
                
                files = [f for f in os.listdir(domain_path) if f.endswith('.json')]
                domain_size = sum(os.path.getsize(os.path.join(domain_path, f)) for f in files)
                total_files += len(files)
                total_size += domain_size
                
                domains.append({
                    "domain": domain_dir,
                    "files": len(files),
                    "size_mb": round(domain_size / (1024 * 1024), 2),
                })
            
            return {
                "success": True,
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "domains": sorted(domains, key=lambda x: x["size_mb"], reverse=True),
            }
        
        elif action == 'inspect':
            url = args.get('url', '')
            if not url:
                return {"success": False, "error": "URL required for inspect action"}
            
            cache_path = _get_cache_path(url, cache_dir)
            if not os.path.exists(cache_path):
                return {"success": True, "cached": False, "url": url}
            
            cached = _load_from_cache(cache_path)
            if cached:
                mtime = os.path.getmtime(cache_path)
                age = datetime.now() - datetime.fromtimestamp(mtime)
                
                return {
                    "success": True,
                    "cached": True,
                    "url": url,
                    "title": cached.get('title', 'Unknown'),
                    "content_length": len(cached.get('content', '')),
                    "age_hours": round(age.total_seconds() / 3600, 1),
                    "downloaded_at": cached.get('downloaded_at', 'unknown'),
                }
            
            return {"success": True, "cached": False, "url": url}
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
            
    except Exception as e:
        return {"success": False, "error": f"Cache check error: {str(e)}"}


def tool_clear_cache(args: dict):
    """
    Clear cached content for a specific URL or entire domain.

    Args:
        args: dict with keys:
            - target (str): "all", a domain name, or a specific URL
            - cache_dir (str, optional): Cache directory (default: "data/web")
    """
    import os
    import shutil
    
    try:
        target = args.get('target', '')
        cache_dir = args.get('cache_dir', 'data/web')
        
        if not target:
            return {"success": False, "error": "Target required: 'all', domain name, or URL"}
        
        removed = 0
        
        if target == 'all':
            if os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    removed += 1
            return {"success": True, "removed": removed, "message": "All cache cleared"}
        
        # Check if target is a URL
        if target.startswith('http'):
            cache_path = _get_cache_path(target, cache_dir)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                removed = 1
            return {"success": True, "removed": removed, "message": f"Cleared cache for {target}"}
        
        # Treat as domain
        domain_dir = os.path.join(cache_dir, target)
        if os.path.exists(domain_dir) and os.path.isdir(domain_dir):
            files = [f for f in os.listdir(domain_dir) if f.endswith('.json')]
            for f in files:
                os.remove(os.path.join(domain_dir, f))
                removed += 1
            os.rmdir(domain_dir)
            return {"success": True, "removed": removed, "message": f"Cleared domain {target}"}
        
        return {"success": False, "error": f"Target not found: {target}"}
        
    except Exception as e:
        return {"success": False, "error": f"Clear cache error: {str(e)}"}
'''

DATA_LAKE_TOOL = '''TOOL_LIBRARY_NAME = 'Data Lake Builder'
TOOL_LIBRARY_DESC = 'Build and query a vector data lake from downloaded content.'
TOOL_LIBRARY_ICON = '🗄️'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'numpy': '>=1.21.0'})

def tool_build_data_lake(args: dict):
    """
    Add documents to the data lake for RAG querying.
    Can load from cache directly if documents have cache paths.

    Args:
        args: dict with keys:
            - documents (list): List of document dicts with 'url', 'title', 'content'
            - prefer_cache (bool, optional): Load content from cache if available (default: True)
            - cache_dir (str, optional): Cache directory to read from (default: "data/web")
    """
    try:
        import sys
        import os
        import json
        
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, 'get_or_create_data_lake'):
            data_lake = main_module.get_or_create_data_lake()
        else:
            return {"success": False, "error": "Data lake not available. Run from main script."}
        
        documents = args.get('documents', [])
        prefer_cache = args.get('prefer_cache', True)
        cache_dir = args.get('cache_dir', 'data/web')
        
        if not documents:
            return {"success": False, "error": "No documents provided."}
        
        # If prefer_cache is True, try to enrich documents from cache
        if prefer_cache:
            for doc in documents:
                url = doc.get('url', '')
                if not url:
                    continue
                
                # Try to load from cache
                try:
                    import hashlib
                    from urllib.parse import urlparse
                    
                    normalized = url.strip().rstrip("/").lower()
                    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
                    
                    try:
                        domain = urlparse(url).netloc or "unknown"
                        domain = "".join(c if c.isalnum() or c in "-._" else "_" for c in domain)
                    except Exception:
                        domain = "unknown"
                    
                    cache_path = os.path.join(cache_dir, domain, f"{url_hash}.json")
                    
                    if os.path.exists(cache_path):
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        
                        # Update document with cached content if missing or short
                        current_content = doc.get('content', '')
                        if not current_content or len(current_content) < 100:
                            doc['content'] = cached.get('content', current_content)
                            doc['title'] = cached.get('title', doc.get('title', 'Untitled'))
                            doc['from_cache'] = True
                except Exception:
                    pass
        
        result = data_lake.add_documents(documents)
        return result
        
    except Exception as e:
        return {"success": False, "error": f"Data lake build error: {str(e)}"}

def tool_query_data_lake(args: dict):
    """
    Query the data lake with semantic search.

    Args:
        args: dict with keys:
            - query (str): The search query
            - top_k (int, optional): Number of results (default: 5)
    """
    try:
        import sys
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, 'get_or_create_data_lake'):
            data_lake = main_module.get_or_create_data_lake()
        else:
            return {"success": False, "error": "Data lake not available."}
        
        query = args.get('query', '')
        top_k = args.get('top_k', 5)
        
        if not query:
            return {"success": False, "error": "Query is required."}
        
        result = data_lake.query(query, top_k=top_k)
        return result
        
    except Exception as e:
        return {"success": False, "error": f"Data lake query error: {str(e)}"}

def tool_data_lake_stats(args: dict):
    """
    Get statistics about the data lake.

    Args:
        args: dict with keys:
            - (none required)
    """
    try:
        import sys
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, 'get_or_create_data_lake'):
            data_lake = main_module.get_or_create_data_lake()
        else:
            return {"success": False, "error": "Data lake not available."}
        
        return data_lake.get_stats()
        
    except Exception as e:
        return {"success": False, "error": f"Stats error: {str(e)}"}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_tools() -> Tuple[str, str, str]:
    """Create tool files if they don't exist."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    web_search_path = TOOLS_DIR / "web_search.py"
    download_path = TOOLS_DIR / "content_downloader.py"
    data_lake_path = TOOLS_DIR / "data_lake_tools.py"
    
    if not web_search_path.exists():
        print(f"📝 Creating web search tool: {web_search_path}")
        web_search_path.write_text(WEB_SEARCH_TOOL, encoding="utf-8")
    
    if not download_path.exists():
        print(f"📝 Creating content downloader tool: {download_path}")
        download_path.write_text(CONTENT_DOWNLOAD_TOOL, encoding="utf-8")
    
    if not data_lake_path.exists():
        print(f"📝 Creating data lake tool: {data_lake_path}")
        data_lake_path.write_text(DATA_LAKE_TOOL, encoding="utf-8")
    
    return str(web_search_path), str(download_path), str(data_lake_path)


def progress_callback(payload: dict):
    """Called during model download."""
    status = payload.get("status", "unknown")
    message = payload.get("message", "")
    completed = payload.get("completed", 0)
    total = payload.get("total", 100)

    if status == "downloading":
        pct = (completed / total * 100) if total else 0
        print(f"⬇️  [{pct:5.1f}%] {message}")
    elif status == "success":
        print(f"✅ {message}")
    elif status == "error":
        print(f"❌ ERROR: {message}")


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    """Stream tokens to console."""
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


def print_research_summary(result: Dict[str, Any]):
    """Pretty-print the research execution metadata."""
    print("\n" + "=" * 70)
    print("📊 RESEARCH EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total agentic rounds:  {result['rounds']}")
    print(f"Tool calls executed:   {len(result['tool_calls'])}")
    
    if not result['tool_calls']:
        print("  (No tools were called — model answered directly)")
        return
    
    for tc in result['tool_calls']:
        print(f"\n  🔹 Round {tc['round']}: {tc['name']}")
        print(f"     Parameters: {json.dumps(tc['parameters'], indent=2, ensure_ascii=False)}")
        
        # Find matching result
        tr = next((r for r in result['tool_results'] if r['round'] == tc['round']), None)
        if tr:
            res = tr['result']
            status = "✅ SUCCESS" if res.get('success') else "❌ FAILED"
            
            # Format output based on tool type
            if tc['name'] == 'tool_web_search' and res.get('success'):
                results = res.get('results', [])
                print(f"     Result: {status} — Found {len(results)} sources")
                for i, r in enumerate(results[:3], 1):
                    print(f"       [{i}] {r.get('title', 'Untitled')}")
                    print(f"           URL: {r.get('url', 'N/A')}")
            
            elif tc['name'] == 'tool_download_content' and res.get('success'):
                title = res.get('title', 'Untitled')
                length = res.get('content_length', 0)
                cached = " [CACHED]" if res.get('cached') else " [FRESH]"
                print(f"     Result: {status}{cached} — Downloaded '{title}' ({length} chars)")
            
            elif tc['name'] == 'tool_build_data_lake' and res.get('success'):
                chunks = res.get('total_chunks', 0)
                print(f"     Result: {status} — Indexed {chunks} chunks")
            
            elif tc['name'] == 'tool_query_data_lake' and res.get('success'):
                matches = res.get('matches', [])
                print(f"     Result: {status} — Found {len(matches)} relevant chunks")
                for i, m in enumerate(matches[:3], 1):
                    score = m.get('relevance_score', 0)
                    source = m.get('source_title', 'Unknown')
                    print(f"       [{i}] Score: {score} | Source: {source}")
            
            else:
                output = str(res.get('output', res.get('error', 'No output')))[:200]
                print(f"     Result: {status}")
                print(f"     Output: {output}...")
    
    if result.get('pending_tool'):
        print(f"\n  ⏸️  PENDING (manual execution):")
        pt = result['pending_tool']
        print(f"     {pt['name']}({json.dumps(pt['parameters'])})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🔬 Deep Research Agent — RAG Pipeline with Smart Caching")
    print("=" * 70)
    print("This agent performs deep research by:")
    print("  1. Searching the web for relevant sources")
    print("  2. Checking data/web/ cache — reusing fresh content, downloading only stale/missing")
    print("  3. Building a vector data lake from downloaded/cached content")
    print("  4. Performing semantic RAG queries against the data lake")
    print("  5. Synthesizing a comprehensive, cited answer")
    print()

    # ── 0. Show cache status ────────────────────────────────────────────
    print("📁 Cache Status:")
    cache_stats = get_cache_stats()
    if cache_stats['total_files'] > 0:
        print(f"   {cache_stats['total_files']} cached files ({cache_stats['total_size_mb']} MB)")
        for domain in cache_stats['domains'][:3]:
            print(f"     • {domain['domain']}: {domain['files']} files ({domain['size_mb']} MB)")
    else:
        print("   Cache is empty — all content will be downloaded fresh")

    # ── 1. Ensure tools exist ─────────────────────────────────────────
    web_search_path, download_path, data_lake_path = ensure_tools()
    print(f"\n📁 Tools ready:")
    print(f"   • {web_search_path}")
    print(f"   • {download_path}")
    print(f"   • {data_lake_path}")

    # ── 2. Create LollmsClient ────────────────────────────────────────
    print("\n🚀 Creating LollmsClient with llama_cpp_server binding...")
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config=BINDING_CONFIG,
        user_name="user",
        ai_name="assistant",
    )

    # ── 3. Download model if missing ──────────────────────────────────
    zoo = client.llm.get_zoo()
    chosen = zoo[MODEL_ZOO_INDEX]
    model_filename = chosen["filename"]

    model_path = Path(BINDING_CONFIG["models_path"]) / model_filename
    if not model_path.exists():
        print(f"\n⬇️  Downloading {chosen['name']} ({chosen['size']}) ...")
        result = client.llm.download_from_zoo(MODEL_ZOO_INDEX, progress_callback=progress_callback)
        if not result.get("status"):
            print(f"❌ Download failed: {result.get('error')}")
            sys.exit(1)
        print("✅ Download complete.")
    else:
        print(f"\n📁 Model already exists: {model_filename}")

    # ── 4. Load the model ─────────────────────────────────────────────
    print(f"\n🔌 Loading model '{model_filename}' ...")
    t0 = time.time()
    success = client.llm.load_model(model_filename)
    if not success:
        print("❌ Failed to load model.")
        sys.exit(1)
    load_time = time.time() - t0
    print(f"✅ Model loaded in {load_time:.1f}s")

    for srv in client.llm.ps():
        print(f"   Server: PID {srv['pid']} | Port {srv['port']} | RSS {srv['rss_mb']} MB")

    # ── 5. Create Deep Research Personality ───────────────────────────
    print("\n🎭 Creating DeepResearchAgent personality...")
    personality = LollmsPersonality(
        name="DeepResearchAgent",
        author="lollms-client",
        category="Research",
        description=(
            "An expert research assistant that performs deep, multi-source "
            "investigation using web search, content analysis, and RAG."
        ),
        system_prompt=(
            "You are DeepResearchAgent, an expert research assistant with access "
            "to the internet and a powerful RAG data lake system.\n\n"
            "Your workflow for deep research:\n"
            "1. SEARCH the web for relevant sources using tool_web_search\n"
            "2. DOWNLOAD full content from the most promising URLs using tool_download_content\n"
            "   (The downloader automatically caches content in data/web/ — reuse fresh cache!)\n"
            "3. BUILD a data lake by feeding downloaded documents into tool_build_data_lake\n"
            "4. QUERY the data lake with specific questions using tool_query_data_lake\n"
            "5. SYNTHESIZE findings into a comprehensive, well-cited report\n\n"
            "CRITICAL RULES:\n"
            "• Always perform at least 2-3 web searches with different queries\n"
            "• Download at least 3-5 full articles for comprehensive coverage\n"
            "• The content downloader caches automatically — prefer cached content when fresh\n"
            "• Build the data lake BEFORE querying it\n"
            "• Use specific RAG queries to extract detailed information\n"
            "• Cite sources using [Source: URL] format\n"
            "• Be thorough but concise in your final synthesis"
        ),
    )

    # ── 6. Create Agent ───────────────────────────────────────────────
    print("🤖 Creating Agent with deep research capabilities...")
    agent = Agent(
        lc=client,
        personality=personality,
        name="DeepResearchAgent",
        role=AgentRole.DOMAIN_EXPERT,
        model_params={"temperature": 0.7},
        max_tokens_per_turn=4096,
        metadata={"specialization": "Deep web research with RAG"},
    )
    print(f"   Agent: {agent.display_name} | Role: {agent.role} | ID: {agent._agent_id[:8]}")

    # ── 7. Define deep research query ─────────────────────────────────
    research_query = (
        "What are the latest developments in quantum computing for 2024-2025? "
        "I want to understand:"
        "1. Recent breakthroughs and milestones"
        "2. Key players and their contributions"
        "3. Practical applications emerging"
        "4. Challenges and limitations"
        "Please perform thorough research using multiple sources and provide citations."
    )

    print("\n" + "-" * 70)
    print("📝 RESEARCH QUERY:")
    print("-" * 70)
    print(research_query)
    print("-" * 70)

    # ── 8. Execute deep research ──────────────────────────────────────
    print("\n🔍 Starting deep research pipeline (this may take several minutes)...\n")
    print("=" * 70)
    print("🤖 AGENT RESPONSE (streaming):")
    print("=" * 70)

    overall_t0 = time.time()
    
    result = agent.generate_with_tools(
        prompt=research_query,
        tools=[web_search_path, download_path, data_lake_path],
        system_prompt=personality.system_prompt,
        temperature=0.7,
        n_predict=4096,
        max_tool_rounds=15,  # Allow many rounds for deep research
        streaming_callback=streaming_callback,
        auto_execute=True,
    )

    overall_elapsed = time.time() - overall_t0

    print("\n")  # Newline after streaming

    # ── 9. Display execution metadata ─────────────────────────────────
    print_research_summary(result)

    # ── 10. Display final report ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📝 FINAL RESEARCH REPORT")
    print("=" * 70)
    print(result["response"])

    # ── 11. Data lake statistics ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("🗄️  DATA LAKE STATISTICS")
    print("=" * 70)
    lake = get_or_create_data_lake()
    stats = lake.get_stats()
    print(f"Total chunks indexed:    {stats['total_chunks']}")
    print(f"Unique sources:            {stats['unique_sources']}")
    print(f"Embedding model:           {stats['embedding_model']}")
    if stats['sources']:
        print("Sources:")
        for src in stats['sources'][:5]:
            print(f"  • {src}")

    # ── 12. Cache statistics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 CACHE STATISTICS")
    print("=" * 70)
    final_cache_stats = get_cache_stats()
    print(f"Total cached files:      {final_cache_stats['total_files']}")
    print(f"Total cache size:          {final_cache_stats['total_size_mb']} MB")
    if final_cache_stats['domains']:
        print("Domains:")
        for domain in final_cache_stats['domains'][:5]:
            print(f"  • {domain['domain']}: {domain['files']} files ({domain['size_mb']} MB)")

    # ── 13. Performance summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("⏱️  PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Model load time:       {load_time:.1f}s")
    print(f"Total research time:   {overall_elapsed:.1f}s")
    print(f"Agentic rounds:        {result['rounds']}")
    print(f"Tools utilized:        {len(result['tool_calls'])}")
    print(f"Final response length: {len(result['response'])} chars")

    # ── 14. Cleanup ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🧹 Cleanup")
    print("=" * 70)
    print("Unloading model...")
    client.llm.unload_model()
    print("👋 Done!")


if __name__ == "__main__":
    main()
