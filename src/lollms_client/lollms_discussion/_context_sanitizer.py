# lollms_discussion/_context_sanitizer.py
# Pure-functional execution layer for Context Sanitization and Tag Hygiene.

import re
from typing import Dict, List, Optional, Any

_ARTIFACT_PATTERN = re.compile(
    r'<art(?:ifact|efact)\s+([^>]*)>(.*?)</art(?:ifact|efact)>',
    re.DOTALL | re.IGNORECASE
)

_SKILL_PATTERN = re.compile(
    r'<skill\s+([^>]*)>(.*?)</skill>',
    re.DOTALL | re.IGNORECASE
)

_NOTE_PATTERN = re.compile(
    r'<note\s+([^>]*)>(.*?)</note>',
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

_ACTION_RESULT_PATTERN = re.compile(
    r'<action_result[^>]*>.*?</action_result>',
    re.DOTALL | re.IGNORECASE
)


def _extract_attrs(attr_str: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}


def scrub_processing_and_status_blocks(text: str) -> str:
    """
    Removes system-generated execution logs (<processing> blocks, status comments)
    without touching the LLM's conversational text or functional action tags.
    """
    if not text:
        return ""
    text = _PROCESSING_PATTERN.sub('', text)
    text = _ORPHAN_PROCESSING_PATTERN.sub('', text)
    text = re.sub(r'<!--\s*status:[^>]*-->', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</processing>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<lollms_artifact[^/]*/>', '', text, flags=re.IGNORECASE)
    text = _ARTEFACT_IMAGE_PATTERN.sub('', text)
    return text.strip()


def sanitize_context_for_llm(text: str) -> str:
    """
    Sanitizes older messages by stripping processing logs while preserving functional tags.
    """
    if not text:
        return ""
    return scrub_processing_and_status_blocks(text)


def build_anti_mimicry_directives() -> str:
    """
    Returns concise, positive directives for tool and artifact emission.
    """
    return (
        "=== ACTION & OUTPUT INTEGRITY PROTOCOL (MANDATORY) ===\n"
        "1. **EMIT REAL FUNCTIONAL TAGS**: To create or edit files use `<artifact name=\"...\">` with complete content or SEARCH/REPLACE blocks. To invoke tools use `<tool>`. To create skills use `<skill title=\"...\">`.\n"
        "2. **PROSE IS NOT ACTION**: Simply saying 'I will write the code' or 'I created the file' in natural text DOES NOT create or modify files. You MUST output the actual XML tags.\n To create artifacts use `<artifact name=\"...\">`, to create skills use `<skill title=\"...\">`, to save notes use `<note title=\"...\">`, and to invoke tools use `<tool>`. Do NOT simulate results or mimic past system messages.\n"
        "3. **DIRECT ENGAGEMENT**: Respond naturally and directly to the user's request. Do not provide meta-commentary about turns or system statuses.\n"
        "=== END ACTION PROTOCOL ==="
    )