# lollms_discussion/_context_sanitizer.py
# Pure-functional execution layer for the Context Diet Protocol.

import re
from typing import Dict, List, Optional, Any

_ARTIFACT_PATTERN = re.compile(
    r'<artifact\s+([^>]*)>(.*?)</artifact>',
    re.DOTALL | re.IGNORECASE
)

_TOOL_PATTERN = re.compile(
    r'<tool>(.*?)</tool>',
    re.DOTALL | re.IGNORECASE
)

_PROCESSING_PATTERN = re.compile(
    r'<processing[^>]*>.*?(?:</processing>|$)',
    re.DOTALL | re.IGNORECASE
)

_ORPHAN_PROCESSING_PATTERN = re.compile(
    r'<processing[^>]*>.*$',
    re.DOTALL | re.IGNORECASE
)

_ARTEFACT_IMAGE_PATTERN = re.compile(
    r'<artefact_image\s+[^/]*/>',
    re.IGNORECASE
)

def _extract_attrs(attr_str: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

def compress_artifacts_to_anchors(text: str) -> str:
    """
    Replaces full <artifact> XML blocks with compact system anchors.
    """
    def _replace_artifact(match: re.Match) -> str:
        attrs = _extract_attrs(match.group(1))
        title = attrs.get('name', 'unknown')
        return f"[🔒SYSTEM_ARTIFACT_ANCHOR:{title}]"
    
    return _ARTIFACT_PATTERN.sub(_replace_artifact, text)

def compress_tool_calls_to_anchors(text: str) -> str:
    """
    Replaces full <tool> JSON blocks with compact system anchors.
    """
    def _replace_tool(match: re.Match) -> str:
        body = match.group(1).strip()
        try:
            data = json.loads(body)
            tool_name = data.get("name", "unknown")
        except Exception:
            tool_name = "unknown"
        return f"[🔒SYSTEM_TOOL_EXECUTED:{tool_name}]"
    
    return _TOOL_PATTERN.sub(_replace_tool, text)

def scrub_processing_blocks(text: str) -> str:
    """
    Removes <processing> execution logs completely.
    """
    text = _PROCESSING_PATTERN.sub('', text)
    text = _ORPHAN_PROCESSING_PATTERN.sub('', text)
    return text

def sanitize_context_for_llm(text: str) -> str:
    """
    Applies the full sanitization pipeline: 
    compress artifacts & tools, scrub processing logs, and remove image anchors.
    """
    text = compress_artifacts_to_anchors(text)
    text = compress_tool_calls_to_anchors(text)
    text = scrub_processing_blocks(text)
    text = _ARTEFACT_IMAGE_PATTERN.sub('', text)
    return text.strip()

def build_anti_mimicry_directives() -> str:
    """
    Returns the strict anti-mimicry directives to be injected into the system prompt.
    """
    return (
        "=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
        "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating text patterns like `[🔒SYSTEM_ARTIFACT_ANCHOR:...`, `[🔒SYSTEM_TOOL_EXECUTED:...`, `[SYSTEM:`, or `[content stripped...`. These are **INFRASTRUCTURE-ONLY** markers used in history to save space. If you output them, NO ACTION will occur.\n"
        "2. **USE REAL TAGS**: To create artifacts, you MUST use the actual `<artifact name=\"...\">` XML tags. To call tools, use `<tool>`. Do NOT mimic the placeholder markers from past messages.\n"
        "=== END ANTI-MIMICRY PROTOCOL ==="
    )