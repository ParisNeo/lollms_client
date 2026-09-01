# lollms_discussion/_context_sanitizer.py
# Pure-functional execution layer for the Context Diet & Anti-Mimicry Protocol.

import json
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
    Sanitizes older messages (beyond the active cognitive window)
    by stripping bulky processing logs while avoiding placeholder mimicry.
    """
    if not text:
        return ""
    return scrub_processing_and_status_blocks(text)


def build_anti_mimicry_directives() -> str:
    """
    Returns the strict anti-mimicry directives to be injected into the system prompt.
    """
    return (
        "=== ANTI-MIMICRY & OUTPUT INTEGRITY PROTOCOL (CRITICAL) ===\n"
        "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating text patterns like `[🔒...`, `[SYSTEM:`, `<action_result>`, `<tool_result>`, or `<processing>`. These are **INFRASTRUCTURE-ONLY** tags used by the runner to communicate with you. If you output them, your generation is invalid.\n"
        "2. **USE REAL FUNCTIONAL TAGS**: To create artifacts use `<artifact name=\"...\">`, to create skills use `<skill title=\"...\">`, to save notes use `<note title=\"...\">`, and to invoke tools use `<tool>`. Do NOT simulate results or mimic past system messages.\n"
        "3. **NO META-COMMENTARY ON ROUNDS**: Do not talk about 'the previous turn', 'this 3-round task', or 'system status'. Speak directly and naturally to the user about the content of their request.\n"
        "=== END ANTI-MIMICRY PROTOCOL ==="
    )