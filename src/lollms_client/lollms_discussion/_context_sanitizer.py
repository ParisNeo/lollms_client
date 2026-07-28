import re
from typing import List, Optional

_ARTEFACT_RE = re.compile(
    r'<(?:artifact|artefact)\s+([^>]*)>(.*?)</(?:artifact|artefact)>',
    re.DOTALL | re.IGNORECASE
)
_PROCESSING_RE = re.compile(
    r'<processing[^>]*>.*?(?:</processing>|$)',
    re.DOTALL | re.IGNORECASE
)
_STATUS_COMMENT_RE = re.compile(r'<!-- status:[^>]*-->', re.IGNORECASE)
_ARTIFACT_IMAGE_RE = re.compile(r'<artefact_image\s+id=["\'][^"\']+["\']\s*/?>', re.IGNORECASE)
_LOLLMS_ARTIFACT_RE = re.compile(r'<lollms_artifact\s+[^/]*/?>', re.IGNORECASE)


def _extract_artifact_metadata(attrs_str: str) -> dict:
    """Safely extracts attributes from an artifact tag's attribute string."""
    attrs = {}
    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
        attrs[m.group(1).lower()] = m.group(2)
    return attrs


def compress_artifacts_to_anchors(text: str) -> str:
    """
    Replaces full <artifact>...</artifact> blocks with compact, read-only anchors.
    This is the core of the Context Diet Protocol.
    """
    def _replace_match(match: re.Match) -> str:
        attrs = _extract_artifact_metadata(match.group(1))
        title = attrs.get("name") or attrs.get("title") or "unknown"
        atype = attrs.get("type", "code")
        version = attrs.get("version", "1")
        return f"[🔒SYSTEM_ARTIFACT_CREATED:{title}|{atype}|v{version}]"

    return _ARTEFACT_RE.sub(_replace_match, text)


def scrub_processing_blocks(text: str) -> str:
    """
    Removes <processing>...</processing> blocks and orphaned status comments.
    These are execution logs that bloat context and cause LLMs to mimic log generation.
    """
    text = _PROCESSING_RE.sub('', text)
    text = _STATUS_COMMENT_RE.sub('', text)
    return text


def sanitize_context_for_llm(text: str) -> str:
    """
    Full sanitization pipeline for historical context.
    1. Compress artifacts to anchors.
    2. Scrub processing logs.
    3. Normalize whitespace.
    """
    if not text:
        return ""
    
    text = compress_artifacts_to_anchors(text)
    text = scrub_processing_blocks(text)
    
    # Remove leftover artifact image/lollms_artifact tags if they survived
    text = _ARTIFACT_IMAGE_RE.sub('', text)
    text = _LOLLMS_ARTIFACT_RE.sub('', text)
    
    # Normalize excessive newlines caused by scrubbing
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def build_anti_mimicry_directives() -> str:
    """
    Generates the system prompt directives that prevent the LLM from
    mimicking system anchors and instructs it to use real XML tags.
    """
    return (
        "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
        "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating text patterns like `[🔒SYSTEM_ARTIFACT_CREATED:...`, `[SYSTEM:`, or `[content stripped...`. These are **INFRASTRUCTURE-ONLY** markers used in history to save space. If you output them, NO ACTION will occur.\n"
        "2. **USE REAL TAGS**: To create artifacts, you MUST use the actual `<artifact name=\"...\">` XML tags. To call tools, use `<tool>`. Do NOT mimic the placeholder markers from past messages.\n"
        "3. **TAG ISOLATION**: Functional tags (`<artifact>`, `<tool>`, `<tool_result>`) MUST NEVER appear inside `</thinking>` blocks. They must ONLY appear in the final response body AFTER the closing `</thinking>` tag.\n"
        "=== END ANTI-MIMICRY PROTOCOL ===\n"
    )
