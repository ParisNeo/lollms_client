# lollms_discussion/_mixin_internet_import.py
# ─────────────────────────────────────────────────────────────────────────────
# InternetImportMixin — Native web scraping and semantic search operations

from __future__ import annotations

import re
import os
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ascii_colors import ASCIIColors, trace_exception

try:
    import requests
except ImportError:
    import pipmaster as pm
    pm.ensure_packages("requests")
    import requests


class InternetImportMixin:
    """Mixin for LollmsDiscussion providing native web-scraping and internet imports."""

    def search_web(
        self,
        query: str,
        provider: str = "duckduckgo",
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform a web search using DuckDuckGo or Google CSE and return structured results."""
        results = []
        try:
            if provider == "duckduckgo":
                import pipmaster as pm
                pm.ensure_packages("duckduckgo-search")
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    from ddgs import DDGS
                with DDGS() as ddgs:
                    raw_results = [r for r in ddgs.text(query, max_results=max_results)]
                    results = [{'title': r.get('title'), 'url': r.get('href'), 'snippet': r.get('body')} for r in raw_results]
            elif provider == "google":
                if not google_api_key or not google_cse_id:
                    raise ValueError("Google Search API is not configured: missing key or cse id.")
                import pipmaster as pm
                pm.ensure_packages("google-api-python-client")
                from googleapiclient.discovery import build as google_build

                service = google_build("customsearch", "v1", developerKey=google_api_key)
                res = service.cse().list(q=query, cx=google_cse_id, num=max_results).execute()
                items = res.get('items', [])
                results = [{'title': item.get('title'), 'url': item.get('link'), 'snippet': item.get('snippet')} for item in items]
        except Exception as e:
            trace_exception(e)
            ASCIIColors.warning(f"[InternetImport] Web search failed: {e}")
        return results

    def search_wikipedia(self, query: str, lang: str = "en") -> List[Dict[str, str]]:
        """Search Wikipedia and return structured article title and snippet results."""
        try:
            query = query.strip()
            if query.startswith("http://") or query.startswith("https://"):
                parsed = urllib.parse.urlparse(query)
                if "wikipedia.org" in parsed.netloc:
                    title = urllib.parse.unquote(parsed.path.split('/')[-1]).replace('_', ' ')
                    return [{"title": title, "url": query, "snippet": "Direct URL import."}]

            api_url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 10
            }
            headers = {
                "User-Agent": "LoLLMs/1.0 (https://github.com/ParisNeo/lollms; parisneo@gmail.com)"
            }
            resp = requests.get(api_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                results.append({
                    "title": item["title"],
                    "url": f"https://{lang}.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                    "snippet": item["snippet"].replace('<span class="searchmatch">', '').replace('</span>', '')
                })
            return results
        except Exception as e:
            trace_exception(e)
            return []

    def search_scopus(
        self,
        query: str,
        api_key: Optional[str] = None,
        inst_token: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Scopus database for scholarly papers, abstracts, and citations."""
        config = getattr(self, "internet_config", {}) or {}
        api_key = api_key or config.get("scopus_api_key") or os.environ.get("SCOPUS_API_KEY")
        inst_token = inst_token or config.get("scopus_inst_token") or os.environ.get("SCOPUS_INST_TOKEN")

        if not api_key:
            ASCIIColors.warning("Scopus API key is missing. Skipping Scopus search.")
            return []

        url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token

        params = {
            "query": query,
            "count": max_results
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            entries = data.get("search-results", {}).get("entry", [])
            for entry in entries:
                title = entry.get("dc:title", "Untitled")
                creator = entry.get("dc:creator", "Unknown")
                publication = entry.get("prj:publicationName", entry.get("pubName", "Unknown Publication"))
                cover_date = entry.get("prism:coverDate", "Unknown Date")
                scopus_id = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")

                links = entry.get("link", [])
                scopus_url = ""
                for link in links:
                    if link.get("@ref") == "scopus":
                        scopus_url = link.get("@href")
                        break
                if not scopus_url and scopus_id:
                    scopus_url = f"https://www.scopus.com/record/display.uri?eid=2-s2.0-{scopus_id}&origin=resultslist"

                snippet = f"Creator: {creator} | Publication: {publication} | Date: {cover_date}"
                results.append({
                    "title": title,
                    "url": scopus_url,
                    "snippet": snippet
                })
            return results
        except Exception as e:
            trace_exception(e)
            ASCIIColors.warning(f"[InternetImport] Scopus search failed: {e}")
            return []

    def import_wikipedia(self, title: str, url: str, auto_load: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch a Wikipedia article by title, format as Markdown, and save as a discussion artifact."""
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "format": "json"
            }
            headers = {
                "User-Agent": "LoLLMs/1.0 (https://github.com/ParisNeo/lollms; parisneo_ai@gmail.com)"
            }
            resp = requests.get(api_url, params=params, headers=headers)
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            page_id = list(pages.keys())[0]
            if page_id != "-1":
                content = pages[page_id].get("extract", "")
                full_md = f"# {title}\nSource: {url}\n\n{content}"
                filename = f"{title}.md"
                existing = self.artefacts.get(filename)
                if existing is None:
                    art = self.artefacts.add(
                        title=filename,
                        content=full_md,
                        active=auto_load
                    )
                else:
                    art = self.artefacts.update(
                        title=filename,
                        new_content=full_md,
                        active=auto_load
                    )
                self.commit()
                return art
        except Exception as e:
            trace_exception(e)
        return None

    def search_arxiv(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        year: Optional[int] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Arxiv for scholarly papers matching query and author filters."""
        import pipmaster as pm
        pm.ensure_packages("arxiv")
        import arxiv
        try:
            query_parts = []
            if query: query_parts.append(query)
            if author: query_parts.append(f"au:{author}")

            query_str = " AND ".join(query_parts) if query_parts else "all:*"

            search = arxiv.Search(
                query=query_str,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            client = arxiv.Client()
            results = []
            for r in client.results(search):
                if year and r.published.year != year:
                    continue
                results.append({
                    "id": r.entry_id.split('/')[-1],
                    "title": r.title,
                    "authors": [a.name for a in r.authors],
                    "year": r.published.year,
                    "abstract": r.summary,
                    "pdf_url": r.pdf_url
                })
            return results
        except Exception as e:
            trace_exception(e)
            return []

    def import_arxiv(self, arxiv_id: str, mode: str = "abstract", auto_load: bool = True) -> Optional[Dict[str, Any]]:
        """Imports an Arxiv paper's abstract, or renders and extracts the full PDF, saving it as an artifact."""
        import pipmaster as pm
        pm.ensure_packages(["arxiv", "pymupdf"])
        import arxiv
        import fitz
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))
            
            if mode == "full":
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    paper.download_pdf(filename=tf.name)
                    pdf_path = tf.name
                
                pdf_doc = fitz.open(pdf_path)
                text = "\n".join([page.get_text() for page in pdf_doc])
                pdf_doc.close()
                os.unlink(pdf_path)
                
                content = f"# {paper.title}\nAuthors: {', '.join([a.name for a in paper.authors])}\nSource: {paper.entry_id}\n\n{text}"
            else:
                content = f"# {paper.title} (Abstract)\nAuthors: {', '.join([a.name for a in paper.authors])}\nSource: {paper.entry_id}\n\n{paper.summary}"

            filename = f"Arxiv_{arxiv_id}.md"
            existing = self.artefacts.get(filename)
            if existing is None:
                art = self.artefacts.add(
                    title=filename,
                    content=content,
                    active=auto_load
                )
            else:
                art = self.artefacts.update(
                    title=filename,
                    new_content=content,
                    active=auto_load
                )
            self.commit()
            return art
        except Exception as e:
            trace_exception(e)
        return None

    def search_github(self, query: str) -> List[Dict[str, str]]:
        """Search GitHub issues/pulls or return a direct URL import block."""
        try:
            query = query.strip()
            if query.startswith("http://") or query.startswith("https://"):
                return [{"title": "Direct GitHub URL", "url": query, "snippet": "Direct import."}]
                
            api_url = f"https://api.github.com/search/issues?q={query}&per_page=10"
            r = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
            r.raise_for_status()
            items = r.json().get('items', [])
            return [{
                "title": i.get('title', 'Unknown'), 
                "url": i.get('html_url', ''), 
                "snippet": f"[{i.get('state', 'unknown').upper()}] {i.get('repository_url', '').split('/')[-1]} | Comments: {i.get('comments', 0)}"
            } for i in items]
        except Exception as e:
            trace_exception(e)
            return []

    def import_github(self, url: str, auto_load: bool = True) -> Optional[Dict[str, Any]]:
        """Imports a GitHub file blob, raw content, or full issue conversation thread with comments."""
        url = url.strip()
        content = ""
        title = "Github_Import"
        
        try:
            if "github.com" in url:
                blob_match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)', url)
                issue_match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)', url)
                
                if blob_match:
                    user, repo, branch, filepath = blob_match.groups()
                    raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"
                    r = requests.get(raw_url)
                    r.raise_for_status()
                    
                    ext = filepath.split('.')[-1] if '.' in filepath else 'txt'
                    content = f"# File: {filepath}\nSource: {url}\n\n```{ext}\n{r.text}\n```"
                    title = f"GH_{filepath.split('/')[-1]}"
                    
                elif issue_match:
                    user, repo, type_, num = issue_match.groups()
                    api_url = f"https://api.github.com/repos/{user}/{repo}/issues/{num}"
                    r = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                    r.raise_for_status()
                    data = r.json()
                    
                    title_str = data.get('title', f'{type_.capitalize()} #{num}')
                    body = data.get('body', '')
                    state = data.get('state', 'unknown')
                    content = f"# [{state.upper()}] {title_str}\nSource: {url}\n\n{body}"
                    
                    comments_url = data.get('comments_url')
                    if comments_url:
                        rc = requests.get(comments_url, headers={"Accept": "application/vnd.github.v3+json"})
                        if rc.status_code == 200:
                            comments = rc.json()
                            for c in comments:
                                content += f"\n\n---\n**{c.get('user',{}).get('login','User')}** commented:\n{c.get('body','')}"
                    title = f"GH_{type_}_{num}"
                    
                else:
                    raise ValueError("Only GitHub file blobs, issues, or pull requests are currently supported.")
                    
            elif "raw.githubusercontent.com" in url:
                r = requests.get(url)
                r.raise_for_status()
                filepath = url.split('/')[-1]
                ext = filepath.split('.')[-1] if '.' in filepath else 'txt'
                content = f"# File: {filepath}\nSource: {url}\n\n```{ext}\n{r.text}\n```"
                title = f"GH_Raw_{filepath}"
            else:
                 raise ValueError("Not a valid GitHub URL.")

            safe_title = re.sub(r'[^A-Za-z0-9_.-]', '_', title) + ".md"

            existing = self.artefacts.get(safe_title)
            if existing is None:
                art = self.artefacts.add(
                    title=safe_title,
                    content=content,
                    active=auto_load
                )
            else:
                art = self.artefacts.update(
                    title=safe_title,
                    new_content=content,
                    active=auto_load
                )
            self.commit()
            return art
        except Exception as e:
            trace_exception(e)
        return None

    def search_stackoverflow(self, query: str) -> List[Dict[str, str]]:
        """Search StackOverflow for questions matching relevant terms."""
        try:
            query = query.strip()
            if query.startswith("http://") or query.startswith("https://"):
                return [{"title": "Direct StackOverflow URL", "url": query, "snippet": "Direct import."}]

            api_url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={query}&site=stackoverflow&pagesize=10"
            r = requests.get(api_url)
            r.raise_for_status()
            items = r.json().get('items', [])
            return [{
                "title": i.get('title', 'Unknown'), 
                "url": i.get('link', ''), 
                "snippet": f"Score: {i.get('score', 0)} | Answers: {i.get('answer_count', 0)} | Tags: {', '.join(i.get('tags', []))}"
            } for i in items]
        except Exception as e:
            trace_exception(e)
            return []

    def import_stackoverflow(self, url: str, auto_load: bool = True) -> Optional[Dict[str, Any]]:
        """Imports a StackOverflow question and its top three answers, formatted into clean Markdown."""
        import pipmaster as pm
        pm.ensure_packages("markdownify")
        from markdownify import markdownify as md
        
        url = url.strip()
        q_match = re.search(r'stackoverflow\.com/questions/(\d+)', url)
        if not q_match:
             raise ValueError("Invalid StackOverflow question URL. Must contain /questions/ID")
        
        q_id = q_match.group(1)
        
        try:
            api_q = f"https://api.stackexchange.com/2.3/questions/{q_id}?site=stackoverflow&filter=withbody"
            r_q = requests.get(api_q)
            r_q.raise_for_status()
            q_data = r_q.json().get('items',[])
            if not q_data:
                raise ValueError("Question not found on StackOverflow.")
            
            question = q_data[0]
            q_title = question.get('title', 'StackOverflow Question')
            q_body_html = question.get('body', '')
            q_body_md = md(q_body_html).strip()

            content = f"# {q_title}\nSource: {url}\n\n**Question (Score: {question.get('score', 0)}):**\n\n{q_body_md}\n\n---\n"

            api_a = f"https://api.stackexchange.com/2.3/questions/{q_id}/answers?site=stackoverflow&order=desc&sort=votes&filter=withbody"
            r_a = requests.get(api_a)
            if r_a.status_code == 200:
                a_data = r_a.json().get('items',[])
                for i, ans in enumerate(a_data[:3]):
                    is_accepted = "✅ ACCEPTED " if ans.get('is_accepted') else ""
                    a_body_md = md(ans.get('body', '')).strip()
                    content += f"\n### {is_accepted}Answer {i+1} (Score: {ans.get('score', 0)})\n\n{a_body_md}\n\n---\n"
                
            title = f"SO_{q_id}.md"

            existing = self.artefacts.get(title)
            if existing is None:
                art = self.artefacts.add(
                    title=title,
                    content=content,
                    active=auto_load
                )
            else:
                art = self.artefacts.update(
                    title=title,
                    new_content=content,
                    active=auto_load
                )
            self.commit()
            return art
        except Exception as e:
            trace_exception(e)
        return None

    def import_youtube_transcript(self, video_url: str, language: str = "en", auto_load: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch and format a YouTube video transcript with minute-based timestamp headers."""
        import pipmaster as pm
        pm.ensure_packages("youtube-transcript-api")
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise RuntimeError("youtube_transcript_api is not installed.")

        video_id = None
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})'
        ]

        for p in patterns:
             match = re.search(p, video_url)
             if match:
                 video_id = match.group(1)
                 break

        if not video_id:
             if len(video_url) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', video_url):
                 video_id = video_url

        if not video_id:
            raise ValueError("Could not extract a valid YouTube Video ID from the provided URL.")

        try:
            # Instantiate the API and list transcripts using the instance-based API
            try:
                yvt = YouTubeTranscriptApi()
                transcript_list_obj = yvt.list(video_id)
            except Exception as e:
                raise ValueError(f"Failed to retrieve transcript list: {e}")

            target_transcript = None
            requested_lang = language.lower().strip() if language else None

            if requested_lang:
                try:
                    target_transcript = transcript_list_obj.find_transcript([requested_lang])
                except:
                    try:
                        first_available = next(iter(transcript_list_obj))
                        if first_available.is_translatable:
                            target_transcript = first_available.translate(requested_lang)
                    except:
                        pass

            if not target_transcript:
                try:
                    target_transcript = transcript_list_obj.find_generated_transcript(['en'])
                except:
                    pass

                if not target_transcript:
                    try:
                        target_transcript = transcript_list_obj.find_manually_created_transcript(['en'])
                    except:
                        pass

                if not target_transcript:
                    try:
                        target_transcript = next(iter(transcript_list_obj))
                    except:
                        pass

            if not target_transcript:
                raise ValueError("No suitable transcript found.")

            # Fetch transcript data
            transcript_data = target_transcript.fetch()

            # Format snippets supporting both legacy list-of-dicts and newer FetchedTranscript classes
            lines = []
            snippets = []
            if hasattr(transcript_data, "snippets"):
                snippets = transcript_data.snippets
            elif isinstance(transcript_data, list):
                snippets = transcript_data
            else:
                snippets = list(transcript_data)

            for entry in snippets:
                if isinstance(entry, dict):
                    start = int(entry.get('start', 0))
                    text = entry.get('text', '')
                else:
                    start = int(getattr(entry, 'start', 0))
                    text = getattr(entry, 'text', '')

                minutes = start // 60
                seconds = start % 60
                lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

            lang_label = target_transcript.language if hasattr(target_transcript, 'language') else (requested_lang or 'unknown')
            full_content = f"# YouTube Transcript ({lang_label})\nSource: {video_url}\n\n" + "\n".join(lines)

            artefact_name = f"Youtube_Transcript_{video_id}.md"
            existing = self.artefacts.get(artefact_name)
            if existing is None:
                art = self.artefacts.add(
                    title=artefact_name,
                    content=full_content,
                    active=auto_load
                )
            else:
                art = self.artefacts.update(
                    title=artefact_name,
                    new_content=full_content,
                    active=auto_load
                )
            self.commit()
            return art
        except Exception as e:
            trace_exception(e)
        return None
