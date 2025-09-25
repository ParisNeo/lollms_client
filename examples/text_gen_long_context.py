from lollms_client import LollmsClient, MSG_TYPE
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time

def load_long_text(
    source: str,
    source_type: str = "auto",
    max_retries: int = 3,
    timeout: int = 30,
    encoding: Optional[str] = None,
    clean_text: bool = True,
    min_length: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Load long text from various internet sources with intelligent content extraction.
    
    Supports:
    - Project Gutenberg books
    - Wikipedia articles  
    - Research papers (HTML)
    - News articles
    - Plain text files
    - General web pages
    """
    
    def detect_source_type(url: str) -> str:
        """Auto-detect source type based on URL patterns."""
        url_lower = url.lower()
        if "gutenberg.org" in url_lower:
            return "gutenberg"
        elif "wikipedia.org" in url_lower:
            return "wikipedia"
        elif "arxiv.org" in url_lower:
            return "arxiv"
        elif url_lower.endswith('.txt'):
            return "txt"
        else:
            return "html"
    
    def make_request(url: str) -> requests.Response:
        """Make HTTP request with retries and proper headers."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(f"Failed to retrieve content after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    def clean_gutenberg_text(text: str) -> tuple:
        """Clean Project Gutenberg text by removing headers/footers."""
        lines = text.split('\n')
        
        # Find start of actual content
        start_idx = 0
        for i, line in enumerate(lines):
            if any(marker in line.upper() for marker in [
                'START OF THIS PROJECT GUTENBERG',
                '*** START OF',
                'CHAPTER 1',
                'CHAPTER I'
            ]):
                start_idx = i + 1
                break
        
        # Find end of content
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if any(marker in lines[i].upper() for marker in [
                'END OF THIS PROJECT GUTENBERG',
                '*** END OF'
            ]):
                end_idx = i
                break
        
        # Extract title
        title = "Unknown Title"
        for line in lines[:50]:
            line_clean = line.strip()
            if (line_clean and len(line_clean) > 10 and 
                not line_clean.startswith('The Project Gutenberg') and
                len(line_clean) < 100):
                title = line_clean
                break
        
        content_lines = lines[start_idx:end_idx]
        cleaned_text = '\n'.join(content_lines)
        return cleaned_text.strip(), title
    
    def extract_wikipedia_content(soup: BeautifulSoup) -> tuple:
        """Extract main content from Wikipedia page."""
        # Get title
        title_elem = soup.find('h1', {'class': 'firstHeading'})
        title = title_elem.get_text().strip() if title_elem else "Wikipedia Article"
        
        # Find main content
        content_div = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            raise ValueError("Could not find Wikipedia article content")
        
        # Extract paragraphs and headers
        content_elements = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        extracted_text = []
        for elem in content_elements:
            text = elem.get_text().strip()
            if text and len(text) > 20:
                if elem.name.startswith('h'):
                    extracted_text.append(f"\n## {text}\n")
                else:
                    extracted_text.append(text)
        
        return '\n\n'.join(extracted_text), title
    
    def extract_html_content(soup: BeautifulSoup, url: str) -> tuple:
        """Extract main content from general HTML pages."""
        # Get title
        title_elem = soup.find('title')
        title = title_elem.get_text().strip() if title_elem else urlparse(url).netloc
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # Try content selectors
        content_selectors = ['article', 'main', '.content', '.post-content', '.entry-content', '#content']
        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                break
        
        if not content:
            content = soup.find('body')
        
        if not content:
            raise ValueError("Could not extract content from HTML")
        
        # Extract text
        text_elements = content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        extracted_text = []
        
        for elem in text_elements:
            text = elem.get_text().strip()
            if text and len(text) > 30:
                if elem.name.startswith('h'):
                    extracted_text.append(f"\n## {text}\n")
                else:
                    extracted_text.append(text)
        
        return '\n\n'.join(extracted_text), title
    
    def clean_extracted_text(text: str) -> str:
        """Clean and normalize extracted text."""
        if not clean_text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\[edit\]', '', text)  # Wikipedia edit links
        text = re.sub(r'\[\d+\]', '', text)  # Reference numbers
        
        # Fix encoding issues
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes
        text = text.replace('–', '-').replace('—', '-')  # Dashes
        text = text.replace('…', '...')  # Ellipsis
        
        return text.strip()
    
    # Main execution
    if not source or not isinstance(source, str):
        raise ValueError("Source must be a valid URL string")
    
    if source_type == "auto":
        source_type = detect_source_type(source)
    
    print(f"🔍 Loading text from: {source}")
    print(f"📝 Detected source type: {source_type}")
    
    try:
        response = make_request(source)
        
        # Handle different source types
        if source_type in ["txt", "gutenberg"]:
            text_content = response.text
            if encoding:
                text_content = response.content.decode(encoding)
            
            title = urlparse(source).path.split('/')[-1]
            
            if source_type == "gutenberg" or "gutenberg" in source.lower():
                text_content, title = clean_gutenberg_text(text_content)
        else:
            # HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if source_type == "wikipedia":
                text_content, title = extract_wikipedia_content(soup)
            else:
                text_content, title = extract_html_content(soup, source)
        
        # Clean the text
        text_content = clean_extracted_text(text_content)
        
        # Validate length
        if len(text_content) < min_length:
            raise ValueError(f"Text too short ({len(text_content)} chars, minimum {min_length})")
        
        # Calculate metrics
        word_count = len(text_content.split())
        char_count = len(text_content)
        
        print(f"✅ Successfully loaded: {char_count:,} characters, ~{word_count:,} words")
        
        return {
            'text': text_content,
            'title': title,
            'source_url': source,
            'word_count': word_count,
            'char_count': char_count,
            'source_type': source_type,
            'metadata': {
                'encoding': response.encoding,
                'content_type': response.headers.get('content-type', 'unknown'),
                'status_code': response.status_code
            }
        }
        
    except Exception as e:
        print(f"❌ Error loading text: {e}")
        raise

# Predefined sample sources for easy testing
SAMPLE_SOURCES = {
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "pride_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "alice_wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "ai_wikipedia": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "climate_change_wikipedia": "https://en.wikipedia.org/wiki/Climate_change",
}

def load_sample_text(sample_name: str, **kwargs) -> Dict[str, Any]:
    """Load a predefined sample text."""
    if sample_name not in SAMPLE_SOURCES:
        available = ", ".join(SAMPLE_SOURCES.keys())
        raise ValueError(f"Unknown sample '{sample_name}'. Available: {available}")
    
    url = SAMPLE_SOURCES[sample_name]
    print(f"📚 Loading sample: {sample_name}")
    return load_long_text(url, **kwargs)

# Initialize your LollmsClient (already configured)
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={
    "model_name": "ollama/gemma3:12b",
    "service_key": "lollms_m1fOU6eS_HXTc09wA9CtCl-yyJBGpaqcvPtOvMqANKPZL9_PEn18",
    "ctx_size": 32768,
})
# Load text from internet
text_data = load_sample_text("moby_dick")  # Loads Moby Dick from Project Gutenberg
print(f"Loaded: {text_data['title']} ({text_data['word_count']:,} words)")

# Process with enhanced method
def callback(message, msg_type, metadata=None):
    
    if msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        print(f"{message}")
        print(f"metadata: {metadata}")
    if msg_type == MSG_TYPE.MSG_TYPE_SCRATCHPAD:
        print(f"{message}")
    if msg_type == MSG_TYPE.MSG_TYPE_STEP:
        print(f"🔄 {message}")
    return True

result = lc.long_context_processing(
    text_to_process=text_data['text'],  # Your 1.2M character file
    contextual_prompt="Create comprehensive literary analysis of Moby-Dick focusing on themes, character development, narrative techniques, and symbolism",
    context_fill_percentage=0.70,  # Conservative for stability
    expected_generation_tokens=400,
    debug=True,  # See what happens at each step!
    temperature=0.7,
    streaming_callback=callback
)

print("Analysis Result:")
print(result)