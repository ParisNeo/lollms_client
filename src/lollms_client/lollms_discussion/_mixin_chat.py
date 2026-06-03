# lollms_discussion/_mixin_chat.py
# ChatMixin: simplified_chat() and chat() — the two high-level conversation methods.
#
# ARTIFACT STREAMING CONTRACT
# ---------------------------
# Content inside <artifact>, <note>, <skill> tags is
# NEVER forwarded to the main chat bubble via MSG_TYPE_CHUNK.
#
# ARTEFACT IMAGE CONTRACT
# -----------------------
# When active artefacts carry images (e.g. PDF pages), those images are collected by
# _collect_artefact_images() and merged with any user-supplied images before the LLM
# call.  The system prompt (built by PromptMixin._build_artefact_instructions) already
# informs the LLM about <artefact_image id="TITLE::N" /> anchors.
#
# Image ordering sent to the LLM:
#   [discussion-level images] + [user message images] + [artefact images in order]
#
# Each artefact image is keyed by its id ("TITLE::N") which matches the anchor tag
# in the artefact text so the model can correlate text and pixel data.
#
# STREAMING STATE MACHINE
# -----------------------
# The streaming interceptor uses an explicit _StreamState object.
# States: NORMAL | BUFFERING_TAG | TOOL_CALL | SECONDARY
#
# BRACKET BUFFERING CONTRACT
# --------------------------
# '<' always starts buffering. Buffer flushed as text only when conclusively not a
# known tag. Whitespace or length alone NEVER trigger flush.
#
# DOUBLE-PROCESSING FIX
# ---------------------
# _emit_processing_close() must NOT call _cb() with final_content (widget HTML /
# form tag) because the relay wraps _cb and calls ss.feed() re-entrantly. This
# causes <lollms_form>/<lollms_inline> to be seen as a new secondary tag and fires
# a second <processing> block.
#
# Fix: _emit_processing_close() stores final_content in self.pending_final_content.
# _feed_secondary() flushes it AFTER state has been fully reset to STATE_NORMAL,
# so no re-entrant feed() call can match it as a tag opener.

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

from ._artefacts import _find_best_title_match, ArtefactType, make_image_id, parse_image_id
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_personality import NullPersonality
from ._message import LollmsMessage

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Hard cap on bracket buffer size
# ---------------------------------------------------------------------------
_MAX_BRACKET_BUF = 4096
# Marker injected into virtual history to replace processed <processing> blocks.
# Uses Unicode chars unlikely to appear in any LLM output naturally.
# The streaming scrubber strips this if the model reproduces it anyway.
_EXEC_MARKER = "\u00b7\u1d3d\u0427\u00d8\u0441\u00b7"  # ·ᴽЧØс·
_EXEC_MARKER_RE = re.compile(re.escape(_EXEC_MARKER) + r'|(\[BLIND_ACTION_EXECUTED\])')
# ---------------------------------------------------------------------------
# All tag prefixes that must NEVER leak to the chat bubble
# ---------------------------------------------------------------------------

# At module level — add alongside other constants
_MAX_PATCH_RETRIES = 3

_PATCH_RETRY_PROMPT_TEMPLATE = """\
[CRITICAL: PATCH REJECTED]
Your SEARCH/REPLACE block for '{title}' failed to match.

DIAGNOSTIC:
- You expected to find: '{expected_first_line}'
- The closest match in the file is: '{closest_line}'
- Reason: Your search block does not match the actual file content character-for-character.

ACTUAL CURRENT CONTENT OF '{title}' (Use this as your source of truth):
---
{current_content}
---

INSTRUCTIONS:
1. Compare your previous SEARCH block against the ACTUAL content above.
2. Notice the differences in casing, punctuation, or spacing.
3. Your SEARCH block must match the ACTUAL content CHARACTER-FOR-CHARACTER.
4. If you cannot find a stable context anchor, provide the FULL file content in your tag.

Reissue your artifact tag now.
"""

_TAG_STARTS = [
    "<tool_call>",
    "<think>", "<think ",
    "<artifact", "<artefact",
    "<coding_plan",
    "<generate_image", "<edit_image",
    "<revert_artifact", "<revert_artefact",
    "<generate_slides", "<street_view", "<schedule_task",
    "<note", "<skill",
    "<lollms_inline",  # Widget tags
    "<lollms_form",    # Form tags
    "<lollms_event",
    "<use_handle",
    "<processing",     # Unified processing indicator
    "<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete"
]
_SUPPRESS_TOKENS = [
    "[BLIND_ACTION_EXECUTED]",
    # Add future markers here
]
_MAX_SUPPRESS_BUF = 64
def _is_fast_message(msg: str) -> bool:
    """Heuristic to detect short greetings or single-word confirmations."""
    m = msg.lower().strip()
    if not m:
        return True
    if len(m) < 20 and any(x in m for x in ["bonjour", "salut", "hello", "hi", "hey", "test"]):
        return True
    return m in ["ok", "merci", "thanks", "cool", "yes", "no", "oui", "non"]


_SECONDARY_TAG_MAP = {
    "<artifact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK,
                       MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artifact>"),
    "<artefact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK,
                       MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artefact>"),
    "<coding_plan":   ("coding_plan_start",   MSG_TYPE.MSG_TYPE_CODING_PLAN_CHUNK,
                       MSG_TYPE.MSG_TYPE_CODING_PLAN_DONE,  "</coding_plan>"),
    "<note":          ("note_start",          MSG_TYPE.MSG_TYPE_NOTE_CHUNK,
                       MSG_TYPE.MSG_TYPE_NOTE_DONE,         "</note>"),
    "<skill":         ("skill_start",         MSG_TYPE.MSG_TYPE_SKILL_CHUNK,
                       MSG_TYPE.MSG_TYPE_SKILL_DONE,        "</skill>"),
    "<lollms_inline": ("inline_widget_start", MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,
                       MSG_TYPE.MSG_TYPE_WIDGET_DONE,       "</lollms_inline>"),
    "<lollms_form":   ("form_start",          MSG_TYPE.MSG_TYPE_FORM_READY,
                       MSG_TYPE.MSG_TYPE_FORM_READY,        "</lollms_form>"),
    "<mem_new":       ("memory_new",          MSG_TYPE.MSG_TYPE_INFO,
                       MSG_TYPE.MSG_TYPE_INFO,              "</mem_new>"),
    "<mem_update":    ("memory_update",       MSG_TYPE.MSG_TYPE_INFO,
                       MSG_TYPE.MSG_TYPE_INFO,              "</mem_update>"),
}



# ---------------------------------------------------------------------------
# Internal callback helpers
# ---------------------------------------------------------------------------

def _cb(callback, text: str, msg_type: MSG_TYPE, meta: Optional[Dict] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        if result is False:
            return False
    except Exception as e:
        trace_exception(e)
    return True


def _step_start(callback, text: str, meta: Optional[Dict] = None) -> Optional[str]:
    event_id = str(uuid.uuid4())
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_START, {"id": event_id, **(meta or {})})
    return event_id


def _step_end(callback, text: str, event_id: Optional[str] = None,
              meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_END, {"id": event_id, **(meta or {})})


def _info(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_INFO, meta)


def _warning(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_WARNING, meta)


# ── Web content extraction & quality scoring ─────────────────────────────

# HTML tags to strip completely (boilerplate / navigation)
_HTML_BOILERPLATE_TAGS = {
    'nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript',
    'iframe', 'form', 'button', 'input', 'select', 'textarea', 'label',
    'svg', 'canvas', 'video', 'audio', 'source', 'track', 'embed', 'object',
    'advertisement', 'ad', 'banner', 'sidebar', 'widget', 'popup', 'modal',
    'cookie', 'consent', 'gdpr', 'newsletter', 'subscribe', 'social', 'share',
}

# Tags that usually contain main content
_CONTENT_TAGS = {
    'article', 'main', 'section', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'td', 'th', 'pre', 'code', 'blockquote', 'summary', 'details',
}

# Stop words that don't add information value
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'now', 'also', 'get', 'go', 'make',
    'see', 'know', 'take', 'use', 'want', 'come', 'look', 'find', 'give',
    'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep',
    'let', 'begin', 'seem', 'help', 'show', 'hear', 'play', 'run', 'move',
    'live', 'believe', 'bring', 'happen', 'stand', 'lose', 'pay', 'meet',
    'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
    'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add',
    'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
    'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
    'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest',
    'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull', 'and',
    'but', 'or', 'yet', 'so', 'if', 'because', 'although', 'though', 'while',
    'whereas', 'unless', 'whether', 'either', 'neither', 'both', 'all',
    'any', 'some', 'many', 'much', 'more', 'most', 'other', 'another',
    'such', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
    'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
    'yourselves', 'themselves', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'must', 'ought', 'need', 'dare', 'used', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'now', 'also', 'get', 'go', 'make', 'see', 'know', 'take',
    'use', 'want', 'come', 'look', 'find', 'give', 'tell', 'ask', 'work',
    'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin', 'help',
    'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'bring',
    'happen', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set',
    'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop',
    'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open',
    'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear',
    'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay',
    'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass',
    'sell', 'require', 'report', 'decide', 'pull', 'and', 'but', 'or',
    'yet', 'so', 'if', 'because', 'although', 'though', 'while', 'whereas',
    'unless', 'whether', 'either', 'neither', 'both', 'any', 'many', 'much',
    'more', 'most', 'another', 'what', 'which', 'who', 'whom', 'whose',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
    'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'myself',
    'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves',
    'themselves', 'page', 'home', 'menu', 'search', 'login', 'sign',
    'register', 'account', 'cart', 'checkout', 'contact', 'about', 'help',
    'faq', 'terms', 'privacy', 'policy', 'sitemap', 'rss', 'feed', 'follow',
    'like', 'tweet', 'share', 'comment', 'reply', 'post', 'blog', 'news',
    'latest', 'recent', 'popular', 'trending', 'featured', 'recommended',
    'related', 'similar', 'more', 'less', 'prev', 'next', 'first', 'last',
    'click', 'tap', 'swipe', 'scroll', 'hover', 'select', 'choose', 'pick',
    'option', 'submit', 'apply', 'save', 'cancel', 'close', 'open', 'back',
    'forward', 'refresh', 'reload', 'loading', 'wait', 'please', 'thank',
    'thanks', 'welcome', 'hello', 'hi', 'hey', 'goodbye', 'bye', 'visit',
    'browse', 'explore', 'discover', 'find', 'view', 'see', 'watch', 'read',
    'learn', 'start', 'begin', 'continue', 'proceed', 'next', 'step',
    'stage', 'phase', 'part', 'section', 'chapter', 'item', 'entry', 'record',
    'detail', 'info', 'information', 'data', 'content', 'text', 'copy',
    'rights', 'reserved', 'copyright', 'trademark', 'patent', 'legal',
    'disclaimer', 'notice', 'warning', 'caution', 'alert', 'important',
    'note', 'tip', 'hint', 'suggestion', 'advice', 'guideline', 'rule',
    'regulation', 'requirement', 'mandatory', 'required', 'optional',
    'recommended', 'suggested', 'advised', 'preferred', 'default',
    'standard', 'normal', 'usual', 'common', 'general', 'typical',
    'regular', 'basic', 'simple', 'easy', 'quick', 'fast', 'instant',
    'immediate', 'direct', 'clear', 'obvious', 'evident', 'apparent',
    'visible', 'noticeable', 'significant', 'important', 'major', 'main',
    'primary', 'chief', 'key', 'central', 'core', 'essential', 'vital',
    'critical', 'crucial', 'necessary', 'needed', 'required', 'mandatory',
    'compulsory', 'obligatory', 'forced', 'involuntary', 'unwilling',
    'reluctant', 'hesitant', 'doubtful', 'uncertain', 'unsure', 'ambiguous',
    'vague', 'unclear', 'obscure', 'hidden', 'concealed', 'secret',
    'private', 'personal', 'confidential', 'restricted', 'limited',
    'exclusive', 'select', 'special', 'particular', 'specific', 'exact',
    'precise', 'accurate', 'correct', 'right', 'true', 'valid', 'legitimate',
    'authentic', 'genuine', 'real', 'actual', 'factual', 'literal', 'verbatim',
    'word', 'word', 'for', 'word', 'exactly', 'precisely', 'strictly',
    'rigidly', 'firmly', 'solidly', 'strongly', 'powerfully', 'forcefully',
    'intensely', 'extremely', 'highly', 'greatly', 'significantly',
    'substantially', 'considerably', 'markedly', 'noticeably', 'clearly',
    'plainly', 'obviously', 'evidently', 'apparently', 'seemingly',
    'presumably', 'probably', 'likely', 'possibly', 'perhaps', 'maybe',
    'perchance', 'conceivably', 'potentially', 'theoretically',
    'hypothetically', 'supposedly', 'allegedly', 'reportedly', 'ostensibly',
    'purportedly', 'nominally', 'officially', 'formally', 'technically',
    'strictly', 'literally', 'actually', 'really', 'truly', 'genuinely',
    'authentically', 'legitimately', 'validly', 'correctly', 'accurately',
    'precisely', 'exactly', 'specifically', 'particularly', 'especially',
    'notably', 'remarkably', 'strikingly', 'surprisingly', 'astonishingly',
    'amazingly', 'incredibly', 'extraordinarily', 'exceptionally',
    'unusually', 'uncommonly', 'rarely', 'seldom', 'hardly', 'scarcely',
    'barely', 'only', 'just', 'merely', 'simply', 'purely', 'solely',
    'exclusively', 'entirely', 'completely', 'totally', 'wholly', 'fully',
    'utterly', 'absolutely', 'altogether', 'overall', 'generally',
    'broadly', 'widely', 'extensively', 'comprehensively', 'thoroughly',
    'deeply', 'profoundly', 'intensively', 'extensively', 'broadly',
    'widely', 'generally', 'usually', 'normally', 'typically', 'commonly',
    'ordinarily', 'regularly', 'routinely', 'habitually', 'consistently',
    'constantly', 'continually', 'continuously', 'persistently', 'repeatedly',
    'frequently', 'often', 'many', 'times', 'much', 'lot', 'lots', 'plenty',
    'abundance', 'wealth', 'profusion', 'myriad', 'multitude', 'host',
    'array', 'assortment', 'variety', 'range', 'selection', 'choice',
    'option', 'alternative', 'substitute', 'replacement', 'equivalent',
    'counterpart', 'parallel', 'analogue', 'match', 'peer', 'equal',
    'equivalent', 'same', 'identical', 'alike', 'similar', 'comparable',
    'analogous', 'corresponding', 'equivalent', 'parallel', 'matching',
    'twin', 'double', 'duplicate', 'copy', 'replica', 'reproduction',
    'imitation', 'facsimile', 'clone', 'mirror', 'reflection', 'image',
    'likeness', 'resemblance', 'semblance', 'appearance', 'guise', 'form',
    'shape', 'figure', 'outline', 'profile', 'silhouette', 'contour',
    'configuration', 'structure', 'construction', 'composition', 'makeup',
    'constitution', 'formation', 'organization', 'arrangement', 'ordering',
    'pattern', 'design', 'layout', 'format', 'scheme', 'plan', 'blueprint',
    'map', 'diagram', 'chart', 'graph', 'table', 'list', 'catalog',
    'directory', 'index', 'guide', 'manual', 'handbook', 'compendium',
    'compilation', 'collection', 'anthology', 'treasury', 'repository',
    'store', 'stock', 'supply', 'reserve', 'fund', 'pool', 'bank', 'cache',
    'hoard', 'stash', 'accumulation', 'aggregation', 'concentration',
    'cluster', 'clump', 'bunch', 'group', 'batch', 'set', 'series',
    'sequence', 'succession', 'chain', 'string', 'train', 'procession',
    'progression', 'advancement', 'development', 'evolution', 'growth',
    'expansion', 'extension', 'enlargement', 'increase', 'gain', 'rise',
    'upturn', 'upsurge', 'upswing', 'boom', 'spurt', 'surge', 'wave',
    'tide', 'flood', 'deluge', 'torrent', 'avalanche', 'landslide',
    'cascade', 'waterfall', 'cataract', 'geyser', 'fountain', 'spring',
    'well', 'source', 'origin', 'root', 'basis', 'foundation', 'ground',
    'bedrock', 'cornerstone', 'keystone', 'linchpin', 'mainstay', 'pillar',
    'support', 'prop', 'stay', 'brace', 'buttress', 'strut', 'truss',
    'beam', 'girder', 'joist', 'rafter', 'stud', 'post', 'pole', 'column',
    'pier', 'pile', 'stilt', 'leg', 'foot', 'base', 'pedestal', 'plinth',
    'platform', 'stage', 'level', 'tier', 'layer', 'stratum', 'sheet',
    'film', 'coat', 'coating', 'covering', 'blanket', 'cloak', 'veil',
    'shroud', 'mantle', 'cape', 'wrap', 'cover', 'lid', 'top', 'cap',
    'hat', 'head', 'crown', 'crest', 'peak', 'summit', 'apex', 'vertex',
    'zenith', 'acme', 'pinnacle', 'climax', 'culmination', 'consummation',
    'completion', 'finish', 'end', 'close', 'conclusion', 'termination',
    'cessation', 'stop', 'halt', 'standstill', 'dead', 'lock', 'impasse',
    'stalemate', 'checkmate', 'gridlock', 'standoff', 'draw', 'tie',
    'dead', 'heat', 'photo', 'finish', 'neck', 'neck', 'close', 'call',
    'near', 'miss', 'bare', 'escape', 'lucky', 'break', 'fortune', 'chance',
    'luck', 'fate', 'destiny', 'kismet', 'karma', 'providence', 'divine',
    'intervention', 'miracle', 'wonder', 'marvel', 'phenomenon', 'spectacle',
    'sight', 'scene', 'view', 'vista', 'panorama', 'prospect', 'outlook',
    'perspective', 'angle', 'slant', 'spin', 'twist', 'turn', 'bend',
    'curve', 'arc', 'arch', 'bow', 'loop', 'coil', 'spiral', 'helix',
    'corkscrew', 'whorl', 'volute', 'scroll', 'roll', 'reel', 'spool',
    'bobbin', 'drum', 'barrel', 'cask', 'keg', 'vat', 'tub', 'tank',
    'reservoir', 'basin', 'bowl', 'dish', 'plate', 'platter', 'tray',
    'salver', 'charger', 'coaster', 'mat', 'pad', 'cushion', 'pillow',
    'bolster', 'headrest', 'armrest', 'backrest', 'footrest', 'legrest',
    'seat', 'chair', 'stool', 'bench', 'sofa', 'couch', 'settee', 'divan',
    'ottoman', 'pouf', 'hassock', 'tabouret', 'barstool', 'folding',
    'chair', 'rocking', 'chair', 'swivel', 'chair', 'recliner', 'lounger',
    'chaise', 'longue', 'daybed', 'futon', 'hammock', 'swing', 'cradle',
    'crib', 'bassinet', 'cot', 'bed', 'bunk', 'bed', 'loft', 'bed', 'trundle',
    'bed', 'murphy', 'bed', 'waterbed', 'airbed', 'mattress', 'pad',
    'topper', 'protector', 'sheet', 'blanket', 'quilt', 'comforter',
    'duvet', 'coverlet', 'bedspread', 'throw', 'afghan', 'shawl', 'wrap',
    'stole', 'scarf', 'muffler', 'necklace', 'chain', 'choker', 'collar',
    'torque', 'pendant', 'locket', 'medallion', 'amulet', 'talisman',
    'charm', 'fetish', 'totem', 'idol', 'icon', 'image', 'effigy', 'statue',
    'sculpture', 'figure', 'figurine', 'statuette', 'bust', 'head', 'torso',
    'trophy', 'prize', 'award', 'medal', 'ribbon', 'badge', 'pin', 'button',
    'brooch', 'clasp', 'clip', 'snap', 'hook', 'eye', 'latch', 'lock', 'bolt',
    'catch', 'hasp', 'hinge', 'pivot', 'swivel', 'joint', 'knuckle', 'elbow',
    'knee', 'ankle', 'wrist', 'shoulder', 'hip', 'socket', 'cavity', 'hole',
    'opening', 'aperture', 'orifice', 'vent', 'outlet', 'inlet', 'entry',
    'entrance', 'access', 'approach', 'way', 'path', 'track', 'trail', 'route',
    'course', 'line', 'lane', 'alley', 'avenue', 'boulevard', 'street', 'road',
    'highway', 'freeway', 'motorway', 'turnpike', 'tollway', 'parkway',
    'driveway', 'pathway', 'walkway', 'sidewalk', 'pavement', 'footpath',
    'bridle', 'path', 'cycle', 'path', 'hiking', 'trail', 'nature', 'trail',
    'walking', 'track', 'running', 'track', 'race', 'track', 'speedway',
    'raceway', 'circuit', 'lap', 'course', 'round', 'turn', 'bend', 'curve',
    'hairpin', 'switchback', 's', 'curve', 'chicane', ' esses', 'straight',
    'stretch', 'run', 'home', 'stretch', 'final', 'lap', 'last', 'leg',
    'anchor', 'leg', 'relay', 'race', 'team', 'event', 'individual', 'event',
    'field', 'event', 'track', 'event', 'combined', 'event', 'decathlon',
    'heptathlon', 'pentathlon', 'triathlon', 'duathlon', 'aquathlon',
    'biathlon', 'modern', 'pentathlon', 'equestrian', 'eventing', 'dressage',
    'show', 'jumping', 'cross', 'country', 'jumping', 'steeplechase', 'hurdles',
    'hurdle', 'race', 'sprint', 'dash', '100m', '200m', '400m', '800m', '1500m',
    'mile', '5000m', '10000m', 'marathon', 'half', 'marathon', 'ultramarathon',
    'ekiden', 'relay', 'sprint', 'relay', 'medley', 'relay', 'freestyle',
    'relay', 'swimming', 'relay', 'rowing', 'relay', 'sailing', 'relay',
    'cycling', 'relay', 'team', 'time', 'trial', 'individual', 'time', 'trial',
    'prologue', 'time', 'trial', 'stage', 'race', 'grand', 'tour', 'tour',
    'giro', 'vuelta', 'dauphine', 'suisse', 'romandie', 'basque', 'country',
    'catalonia', 'california', 'colorado', 'utah', 'alberta', 'quebec',
    'british', 'columbia', 'alps', 'pyrenees', 'dolomites', 'hindu', 'kush',
    'karakoram', 'pamir', 'tian', 'shan', 'altai', 'say', 'khentii', 'khangai',
    'yablonoi', 'stanovoi', 'dzhugdzhur', 'sikhote', 'alin', 'kolyma',
    'chukotka', 'kamchatka', 'kuril', 'sakhalin', 'hokkaido', 'honshu',
    'shikoku', 'kyushu', 'okinawa', 'taiwan', 'luzon', 'mindanao', 'palawan',
    'borneo', 'sumatra', 'java', 'sulawesi', 'new', 'guinea', 'australia',
    'tasmania', 'zealand', 'caledonia', 'fiji', 'samoa', 'tonga', 'vanuatu',
    'solomon', 'islands', 'papua', 'guinea', 'indonesia', 'malaysia',
    'singapore', 'thailand', 'vietnam', 'cambodia', 'laos', 'myanmar',
    'bangladesh', 'india', 'pakistan', 'nepal', 'bhutan', 'sri', 'lanka',
    'maldives', 'afghanistan', 'iran', 'iraq', 'syria', 'lebanon', 'jordan',
    'israel', 'palestine', 'saudi', 'arabia', 'yemen', 'oman', 'uae', 'qatar',
    'bahrain', 'kuwait', 'turkey', 'cyprus', 'armenia', 'azerbaijan',
    'georgia', 'kazakhstan', 'uzbekistan', 'turkmenistan', 'tajikistan',
    'kyrgyzstan', 'mongolia', 'china', 'korea', 'japan', 'russia', 'belarus',
    'ukraine', 'moldova', 'romania', 'bulgaria', 'serbia', 'montenegro',
    'bosnia', 'croatia', 'slovenia', 'hungary', 'slovakia', 'czech', 'republic',
    'poland', 'lithuania', 'latvia', 'estonia', 'finland', 'sweden', 'norway',
    'denmark', 'iceland', 'ireland', 'kingdom', 'france', 'germany',
    'netherlands', 'belgium', 'luxembourg', 'switzerland', 'austria', 'italy',
    'spain', 'portugal', 'greece', 'albania', 'macedonia', 'kosovo', 'malta',
    'andorra', 'monaco', 'liechtenstein', 'san', 'marino', 'vatican',
    'slovenia', 'croatia', 'montenegro', 'albania', 'macedonia', 'serbia',
    'bosnia', 'herzegovina', 'kosovo', 'slovenia', 'croatia', 'montenegro',
    'albania', 'macedonia', 'serbia', 'bosnia', 'herzegovina', 'kosovo',
}


def _extract_main_content(html_text: str, query_words: set) -> List[Dict[str, Any]]:
    """
    Extract information-dense passages from raw HTML or web text.
    Returns list of passages with quality scores.
    """
    if not html_text or not html_text.strip():
        return []

    # Strip common HTML tags
    text = html_text
    # Remove script/style content entirely
    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', ' ', text, flags=re.S | re.I)
    # Remove other tags but keep their text content
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<')
    text = text.replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return []

    # Split into candidate passages (paragraphs or sentence groups)
    # Use multiple sentence boundaries to create coherent passages
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    passages = []
    current_passage = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        if current_len + len(sent) > 800:
            # Flush current passage
            if current_passage:
                passages.append(' '.join(current_passage))
            current_passage = [sent]
            current_len = len(sent)
        else:
            current_passage.append(sent)
            current_len += len(sent)

    if current_passage:
        passages.append(' '.join(current_passage))

    # Score each passage
    scored_passages = []
    for p in passages:
        p_lower = p.lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', p_lower)
        if not words:
            continue

        # 1. Information density score (non-stop-words ratio)
        meaningful_words = [w for w in words if w not in _STOP_WORDS]
        density = len(meaningful_words) / max(len(words), 1)

        # 2. Query relevance score (how many query words appear)
        query_hits = sum(1 for qw in query_words if qw in p_lower)
        query_relevance = query_hits / max(len(query_words), 1)

        # 3. Length score (prefer medium-length passages, penalize too short or too long)
        word_count = len(words)
        if word_count < 20:
            length_score = word_count / 20  # Penalize very short
        elif word_count > 300:
            length_score = max(0, 1 - (word_count - 300) / 500)  # Penalize very long
        else:
            length_score = 1.0

        # 4. Structural quality (penalize lists of links, repetitive patterns)
        link_like = len(re.findall(r'https?://|www\.|\.com|\.org|\.net', p))
        list_like = len(re.findall(r'^\s*[-*•]\s', p, re.M))
        structure_penalty = min(1.0, (link_like * 0.1) + (list_like * 0.05))

        # 5. Composite score
        composite = (
            density * 0.35 +
            query_relevance * 0.35 +
            length_score * 0.20 +
            (1 - structure_penalty) * 0.10
        )

        scored_passages.append({
            "text": p,
            "score": round(composite, 4),
            "density": round(density, 4),
            "query_relevance": round(query_relevance, 4),
            "word_count": word_count,
        })

    # Sort by composite score descending
    scored_passages.sort(key=lambda x: x["score"], reverse=True)
    return scored_passages


def _format_web_search_for_llm(processed: Dict[str, Any], max_chars: int = 2000) -> str:
    """
    Format processed web search results into clean, readable text for the LLM.
    """
    lines = [
        f"Web Search Results for: '{processed.get('query', '')}'",
        f"Found {processed.get('source_count', 0)} sources with {processed.get('total_passages', 0)} quality passages",
        "",
    ]

    total_used = 0
    for src in processed.get("sources", [])[:5]:  # Top 5 sources
        src_lines = [
            f"--- Source: {src.get('title', 'Untitled')} ---",
            f"URL: {src.get('url', '')}",
            f"Quality Score: {src.get('quality_score', 0)} | Passages: {src.get('passage_count', 0)}",
            "",
        ]

        # Add top passages
        for p in src.get("passages", [])[:3]:  # Top 3 passages per source
            p_text = p.get("text", '')
            if total_used + len(p_text) > max_chars * 0.8:
                # Truncate if we're running out of budget
                remaining = max_chars * 0.8 - total_used
                if remaining > 100:
                    p_text = p_text[:int(remaining)] + "..."
                else:
                    break
            src_lines.append(f"  [Passage — score {p.get('score', 0):.2f}]")
            src_lines.append(f"  {p_text}")
            src_lines.append("")
            total_used += len(p_text) + 50  # overhead for formatting

        # Add key facts
        facts = src.get("key_facts", [])
        if facts:
            src_lines.append("  Key Facts:")
            for f in facts[:3]:
                src_lines.append(f"    • {f.get('text', '')}")
            src_lines.append("")

        lines.extend(src_lines)

        if total_used >= max_chars * 0.9:
            break

    return "\n".join(lines)


def _process_web_search_result(
    result: Dict[str, Any],
    query: str,
    max_passages: int = 5,
    min_score: float = 0.15,
) -> Dict[str, Any]:
    """
    Process a web search tool result to extract high-quality content.
    Returns structured result with ranked passages.
    """
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))

    # Handle different result shapes
    sources = []
    raw_items = []

    if isinstance(result, dict):
        # Try common result keys
        for key in ['results', 'sources', 'items', 'data', 'pages', 'hits', 'documents']:
            if key in result and isinstance(result[key], list):
                raw_items = result[key]
                break
        if not raw_items and 'content' in result:
            raw_items = [result]
    elif isinstance(result, list):
        raw_items = result

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        title = (
            item.get('title') or item.get('name') or item.get('heading') or
            item.get('subject') or 'Untitled'
        )
        url = (
            item.get('url') or item.get('link') or item.get('href') or
            item.get('source') or ''
        )

        # Get raw content from various possible fields
        raw_content = (
            item.get('content') or item.get('text') or item.get('body') or
            item.get('snippet') or item.get('summary') or item.get('description') or
            item.get('html', '') or ''
        )

        # Extract quality passages
        passages = _extract_main_content(raw_content, query_words)

        # Filter by minimum score and take top passages
        good_passages = [p for p in passages if p["score"] >= min_score][:max_passages]

        if good_passages:
            # Compute overall source quality
            avg_score = sum(p["score"] for p in good_passages) / len(good_passages)
            total_words = sum(p["word_count"] for p in good_passages)

            # Extract key facts (highest-scoring sentences)
            key_facts = []
            for p in good_passages[:3]:
                # Split passage into sentences and score each
                sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', p["text"])
                for s in sents:
                    s = s.strip()
                    if len(s) < 20 or len(s) > 200:
                        continue
                    s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
                    fact_score = len(s_words & query_words) / max(len(query_words), 1)
                    if fact_score > 0.3:
                        key_facts.append({
                            "text": s,
                            "relevance": round(fact_score, 3),
                        })

            sources.append({
                "title": title,
                "url": url,
                "quality_score": round(avg_score, 4),
                "total_words": total_words,
                "passage_count": len(good_passages),
                "passages": good_passages,
                "key_facts": key_facts[:5],  # Top 5 key facts
            })

    # Sort sources by quality score
    sources.sort(key=lambda x: x["quality_score"], reverse=True)

    return {
        "success": True,
        "query": query,
        "source_count": len(sources),
        "sources": sources,
        "total_passages": sum(s["passage_count"] for s in sources),
        "total_words": sum(s["total_words"] for s in sources),
    }


def _smart_truncate_result(data: Any, max_total_chars: int = 2000, max_string_len: int = 300) -> Any:
    """
    Intelligently truncate a tool result for LLM consumption.
    Preserves structure but caps string lengths and list sizes.
    """
    if isinstance(data, dict):
        result = {}
        total_estimate = 0
        for k, v in data.items():
            if total_estimate >= max_total_chars:
                result[k] = "... (truncated)"
                continue
            truncated_v = _smart_truncate_result(v, max_total_chars // 2, max_string_len)
            result[k] = truncated_v
            total_estimate += len(str(truncated_v))
        return result
    elif isinstance(data, list):
        if len(data) == 0:
            return data
        # Keep first few items, summarize the rest
        keep_count = min(5, len(data))
        result = [_smart_truncate_result(item, max_total_chars // keep_count, max_string_len) 
                  for item in data[:keep_count]]
        if len(data) > keep_count:
            result.append(f"... ({len(data) - keep_count} more items)")
        return result
    elif isinstance(data, str):
        if len(data) > max_string_len:
            return data[:max_string_len] + f"... [{len(data) - max_string_len} chars truncated]"
        return data
    else:
        return data


def _extract_content_title(content: str, max_len: int = 80) -> Optional[str]:
    if not content or not content.strip():
        return None
    lines     = content.splitlines()
    non_blank = [l for l in lines if l.strip()]
    if not non_blank:
        return None

    def _clean(s: str) -> str:
        s = re.sub(r"[*_`#~]+", "", s).strip(" \t\r\n|>")
        if len(s) > max_len:
            s = s[:max_len].rsplit(" ", 1)[0] + "…"
        return s or ""

    for line in non_blank[:30]:
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    for i, line in enumerate(non_blank[:-1]):
        nxt = non_blank[i + 1]
        if re.match(r"^[=\-~^\"'`#*+]{4,}$", nxt.strip()) and line.strip():
            t = _clean(line)
            if t:
                return t

    fm = re.search(
        r'(?:^|[\n,{])\s*["\']?title["\']?\s*[:=]\s*["\']?([^\n"\'},]+)',
        content[:1000], re.IGNORECASE,
    )
    if fm:
        t = _clean(fm.group(1))
        if t:
            return t

    for pat in [r"<title[^>]*>([^<]+)</title>", r"<h1[^>]*>([^<]+)</h1>"]:
        m = re.search(pat, content[:2000], re.IGNORECASE)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    m = re.match(r"^\s*[*_]{1,2}([^*_\n]{4,60})[*_]{1,2}", content.lstrip())
    if m:
        t = _clean(m.group(1))
        if t:
            return t

    candidate = non_blank[0].strip()
    stripped  = _clean(candidate).rstrip(".")
    if stripped and len(stripped) <= max_len and not re.search(r"[.!?;]", stripped):
        return stripped

    return _clean(non_blank[0])


# ---------------------------------------------------------------------------
# Handle system helpers
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str) -> List[Dict[str, str]]:
    blocks = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    for m in pattern.finditer(text):
        blocks.append({
            "language": m.group(1).strip(),
            "content":  m.group(2),
            "raw":      m.group(0),
        })
    return blocks


def _resolve_handle(ref: str, branch_messages: List) -> Optional[Dict[str, str]]:
    parts = ref.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        msg_idx   = int(parts[0])
        block_idx = int(parts[1])
    except ValueError:
        return None

    if msg_idx < 0 or msg_idx >= len(branch_messages):
        return None

    msg    = branch_messages[msg_idx]
    blocks = _extract_code_blocks(getattr(msg, "content", "") or "")

    if block_idx < 0 or block_idx >= len(blocks):
        return None

    return blocks[block_idx]


def _build_handle_instructions(branch_messages: List) -> str:
    entries = []
    for msg_idx, msg in enumerate(branch_messages):
        blocks = _extract_code_blocks(getattr(msg, "content", "") or "")
        for block_idx, blk in enumerate(blocks):
            lang    = blk["language"] or "text"
            preview = blk["content"].strip().splitlines()[0][:60] if blk["content"].strip() else ""
            entries.append(f"  {msg_idx}:{block_idx}  [{lang}]  {preview}")

    if not entries:
        return ""

    lines = [
        "",
        "=== AVAILABLE HANDLES ===",
        "Instead of rewriting a code block that already exists in the conversation,",
        "you can reference it by handle to create or update an artefact directly.",
        "",
        "Syntax (self-closing tag):",
        '  <use_handle ref="<msg_idx>:<block_idx>" name="filename.ext"',
        '              type="code" language="python"/>',
        "",
        "Available handles in this conversation:",
    ] + entries + [
        "",
        "Example — convert the Python block at position 1:0 into an artefact:",
        '  <use_handle ref="1:0" name="main.py" type="code" language="python"/>',
        "=== END HANDLES ===",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool-catalogue helpers
# ---------------------------------------------------------------------------
# ── Large output condensation ──────────────────────────────────────────────
# When a tool returns a very large payload, we store it in a TextBuffer,
# extract relevant snippets into the scratchpad, and feed the LLM a
# condensed summary.  This keeps the context window healthy while still
# allowing the model to make FOLLOW-UP tool calls.
_LARGE_OUTPUT_THRESHOLD_CHARS = 2000

_TOOL_CALL_HEADER = """\

╔══════════════════════════════════════════════════════════════════╗
║  TOOL USE — READ CAREFULLY BEFORE GENERATING                     ║
╠══════════════════════════════════════════════════════════════════╣
║  You have external tools. To use one you MUST emit a tool_call   ║
║  tag containing a JSON object — NO markdown, NO prose calls.     ║
║                                                                  ║
║  EXACT FORMAT (copy this pattern):                               ║
║    <tool_call>{"name": "tool_name",                              ║
║                "parameters": {"key": "value"}}</tool_call>       ║
║                                                                  ║
║  Rules:                                                          ║
║    • One tool call per response turn.                            ║
║    • Do NOT call a tool you already called this turn (see STATE) ║
║    • After calling ALL needed tools, write your final answer.    ║
║    • If the user explicitly asks you to use a tool, USE IT.      ║
╚══════════════════════════════════════════════════════════════════╝

TOOLS AVAILABLE:
"""

_TOOL_CALL_REMINDER = """\
[AGENT REMINDER — YOU HAVE TOOLS]
Call syntax:  <tool_call>{"name": "NAME", "parameters": {…}}</tool_call>
One call per turn. Check AGENT STATE above before calling.
If the user asked you to use a specific tool, call it NOW.
"""

_TOOL_CALL_CORRECTION = """\
You were expected to call a tool but did not emit a <tool_call> tag.
Please re-read the available tools and call the most appropriate one now.
Do not explain why you didn't call it — just call it.
"""

_SURGICAL_UPDATE_GUIDANCE = """\

╔══════════════════════════════════════════════════════════════════╗
║  ARTEFACT UPDATE POLICY — SURGICAL EDITS PREFERRED               ║
╠══════════════════════════════════════════════════════════════════╣
║  When modifying an EXISTING artefact, you MUST use SEARCH/REPLACE║
║  patch blocks unless the change affects >60% of the content.     ║
║                                                                  ║
║  PATCH FORMAT (Aider-style — copy exactly):                      ║
║    <artifact name="filename.ext" type="code" language="python">  ║
║    <<<<<<< SEARCH                                                 ║
║    exact lines to find (verbatim, incl. indentation)             ║
║    =======                                                        ║
║    replacement lines                                             ║
║    >>>>>>> REPLACE                                                ║
║    </artifact>                                                   ║
║                                                                  ║
║  Rules:                                                          ║
║    • SEARCH block must match the current artefact EXACTLY.       ║
║    • Multiple SEARCH/REPLACE blocks allowed in one tag.          ║
║    • Only rewrite the full content when creating NEW artefacts   ║
║      or when >60% of lines change.                               ║
║    • Never add commentary inside SEARCH/REPLACE blocks.          ║
║    • If a previous patch was REJECTED, widen the SEARCH context  ║
║      by ±3 lines and try again.                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

def _build_tool_system_prompt(
    base_system_prompt: str,
    tool_descriptions: List[str],
) -> str:
    cleaned = re.split(
        r'\n*╔══+╗.*?╚══+╝.*?\nTOOLS AVAILABLE:',
        base_system_prompt,
        flags=re.DOTALL,
    )[0].rstrip()
    cleaned = re.split(
        r'\n*## Available Tools.*',
        cleaned,
        flags=re.DOTALL,
    )[0].rstrip()
    cleaned = re.split(
        r'\n*### FUNCTION CALLING INSTRUCTIONS.*',
        cleaned,
        flags=re.DOTALL,
    )[0].rstrip()

    # Strip any previous surgical-update guidance so we don't duplicate it
    cleaned = re.split(
        r'\n*╔══+╗.*?ARTEFACT UPDATE POLICY.*?╚══+╝',
        cleaned,
        flags=re.DOTALL,
    )[0].rstrip()

    tool_block = _TOOL_CALL_HEADER + "\n".join(tool_descriptions)
    return cleaned + _SURGICAL_UPDATE_GUIDANCE + "\n\n" + tool_block


# ---------------------------------------------------------------------------
# Widget content validation
# ---------------------------------------------------------------------------

_NON_WEB_FENCE_RE = re.compile(
    r'```(?:python|mermaid|java|c\+\+|cpp|c#|csharp|rust|go|ruby|php|r|'
    r'swift|kotlin|scala|haskell|erlang|elixir|clojure|lua|perl|bash|sh|'
    r'zsh|powershell|sql|graphql|yaml|toml|ini|json|xml|latex|tex|'
    r'dockerfile|makefile|cmake)[\s\S]*?```',
    re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r'<(?:html|head|body|div|span|script|style|p|h[1-6]|'
                           r'canvas|svg|button|input|form|table|ul|ol|li|a|'
                           r'section|article|main|header|footer|nav)[^>]*>',
                           re.IGNORECASE)


def _validate_widget_content(raw: str, title: str) -> Optional[str]:
    # Leniently strip any leading/trailing markdown code block fences if present
    cleaned = raw.strip()
    # Strip opening ```html, ```xml, ```, etc.
    cleaned = re.sub(r'^```[a-zA-Z0-9]*\s*\n', '', cleaned, flags=re.IGNORECASE)
    # Strip closing ```
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        ASCIIColors.warning(f"[Widget '{title}'] Content is empty after cleaning. Widget discarded.")
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Form parsing helpers
# ---------------------------------------------------------------------------

def _parse_form_xml(tag_attrs_str: str, body: str) -> Optional[Dict[str, Any]]:
    def _parse_attrs(s: str) -> Dict[str, str]:
        return {m.group(1): m.group(2)
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', s)}

    top_attrs = _parse_attrs(tag_attrs_str)

    form: Dict[str, Any] = {
        "id":           str(uuid.uuid4()),
        "title":        top_attrs.get("title", "Please fill in the form"),
        "description":  top_attrs.get("description", ""),
        "submit_label": top_attrs.get("submit_label", "Submit"),
        "fields":       [],
    }

    body_stripped = body.strip()

    if body_stripped.startswith("{") or body_stripped.startswith("["):
        try:
            parsed = json.loads(body_stripped)
            if isinstance(parsed, dict):
                form.update({k: v for k, v in parsed.items() if k != "id"})
                if "fields" not in form:
                    form["fields"] = []
                return form
        except json.JSONDecodeError:
            pass

    field_pattern = re.compile(
        r'<(?:field|section)\s([^/]*?)(?:/\s*>|>.*?</(?:field|section)>)',
        re.DOTALL | re.IGNORECASE,
    )
    fields_found = []
    for m in field_pattern.finditer(body_stripped):
        attrs = _parse_attrs(m.group(1))
        field: Dict[str, Any] = {
            "name":    attrs.get("name", f"field_{len(fields_found)}"),
            "label":   attrs.get("label", attrs.get("name", f"Field {len(fields_found)+1}")),
            "type":    attrs.get("type", "text"),
            "required": attrs.get("required", "true").lower() not in ("false", "0", "no"),
        }
        for num_key in ("min", "max", "step", "rows", "min_rating", "max_rating"):
            if num_key in attrs:
                try:
                    field[num_key] = float(attrs[num_key]) if '.' in attrs[num_key] \
                                     else int(attrs[num_key])
                except ValueError:
                    field[num_key] = attrs[num_key]
        for str_key in ("default", "placeholder", "hint", "accept", "language",
                        "category", "options"):
            if str_key in attrs:
                field[str_key] = attrs[str_key]
        if "options" in field and isinstance(field["options"], str):
            field["options"] = [o.strip() for o in field["options"].split(",") if o.strip()]
        if "multiple" in attrs:
            field["multiple"] = attrs["multiple"].lower() not in ("false", "0", "no")
        fields_found.append(field)

    if fields_found:
        form["fields"] = fields_found
        return form

    question_re = re.compile(r'^[-*\d.]+\s+(.+)', re.MULTILINE)
    questions = question_re.findall(body_stripped)
    if questions:
        form["fields"] = [
            {
                "name":     f"q{i+1}",
                "label":    q.strip().rstrip("?:"),
                "type":     "textarea",
                "required": True,
                "rows":     3,
            }
            for i, q in enumerate(questions)
        ]
        return form

    ASCIIColors.warning(f"[Form] Could not parse form body. Returning empty form.")
    return form

def _format_form_answers_for_llm(form_descriptor: Dict, answers: Dict[str, Any]) -> str:
    lines = [
        f"### 📋 Form Submission: {form_descriptor.get('title', 'User Form')}",
        "",
    ]
    fields = form_descriptor.get("fields", [])
    field_map = {f["name"]: f for f in fields if f.get("type") != "section"}

    for name, value in answers.items():
        label = field_map.get(name, {}).get("label", name)
        if isinstance(value, str) and len(value) > 2000:
            display = value[:2000] + f"… [+{len(value)-2000} chars truncated]"
        else:
            display = value
        lines.append(f"* **{label}**: {display}")

    lines.append("\n*Form submitted successfully.*")
    return "\n".join(lines)


def _parse_tag_attrs(attr_str: str) -> Dict[str, str]:
    return {m.group(1): m.group(2)
            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}


# ---------------------------------------------------------------------------
# Artefact image collection
# ---------------------------------------------------------------------------

def _collect_artefact_images(discussion: Any) -> List[str]:
    context_imgs = discussion.artefacts.get_context_images()
    return [img["data"] for img in context_imgs]


def _build_artefact_image_index(discussion: Any) -> Dict[str, int]:
    context_imgs = discussion.artefacts.get_context_images()
    return {img["id"]: idx for idx, img in enumerate(context_imgs)}


def _build_artefact_image_map_note(
    discussion: Any,
    user_image_count: int,
) -> str:
    context_imgs = discussion.artefacts.get_context_images()
    if not context_imgs:
        return ""
    lines = [
        "[Artefact image map — images are appended after user images in the vision input]"
    ]
    for idx, img in enumerate(context_imgs):
        slot = user_image_count + idx
        lines.append(
            f'  <artefact_image id="{img["id"]}" /> → vision input image #{slot}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# StreamState — explicit state machine for the inline relay
# ---------------------------------------------------------------------------

class _StreamState:
    """
    Encapsulates all mutable state for the streaming interceptor.

    UNIFIED STREAMING PROTOCOL:
    All secondary content (artefacts, widgets, forms, notes, skills) streams
    through <processing> tags. Status messages stream inside these tags; final
    bulk content (widget HTML, form tag) is stored in pending_final_content and
    flushed AFTER the state machine resets to STATE_NORMAL — never emitted via
    _cb() from inside _fire_secondary_done() or _emit_processing_close().

    WHY: The relay callbacks (e.g. _fast_relay, _inline_relay) intercept all
    MSG_TYPE_CHUNK calls and feed them back into ss.feed(). If final_content is
    emitted via _cb() while still inside _feed_secondary(), the re-entrant
    ss.feed() call sees <lollms_form>/<lollms_inline> as a new secondary tag
    and fires a duplicate <processing> block. By flushing pending_final_content
    only after state reset, we guarantee no re-entrant tag matching occurs.
    """

    STATE_NORMAL    = "normal"
    STATE_BUFFERING = "buffering"
    STATE_TOOL_CALL = "tool_call"
    STATE_SECONDARY = "secondary"

    def __init__(
        self,
        discussion: 'LollmsDiscussion',
        callback,
        ai_message,
        enable_notes: bool,
        enable_skills: bool,
        enable_inline_widgets: bool,
        enable_forms: bool,
        auto_activate_artefacts: bool = True,
        enable_artefacts: bool = True,
        enable_in_message_status: bool = True,
    ):
        self.discussion            = discussion
        self.callback              = callback
        self.ai_message            = ai_message
        self.enable_artefacts      = enable_artefacts
        self.enable_in_message_status = enable_in_message_status

        if not enable_artefacts:
            self.enable_notes          = False
            self.enable_skills         = False
            self.enable_inline_widgets = False
            self.enable_forms          = False
        else:
            self.enable_notes          = enable_notes
            self.enable_skills         = enable_skills
            self.enable_inline_widgets = enable_inline_widgets
            self.enable_forms          = enable_forms

        self.auto_activate         = auto_activate_artefacts

        self.state: str             = self.STATE_NORMAL
        self.bracket_buf: List[str] = []

        self.tool_buf: List[str]    = []
        self.tool_trigger: bool     = False

        # Secondary stream state
        self.sec_prefix: str        = ""
        self.sec_chunk_mt: Any      = None
        self.sec_done_mt: Any       = None
        self.sec_close_tag: str     = ""
        self.sec_open_attrs: Dict   = {}
        self.sec_content: List[str] = []
        self.sec_close_scan: str    = ""

        # Unified processing state
        self.proc_type: str          = "general_processing"
        self.proc_title: str         = "Processing"
        self.proc_attrs: Dict        = {}
        self.proc_content: List[str] = []
        self.proc_has_opened: bool   = False

        # ---------------------------------------------------------------
        # KEY FIX: final bulk content (widget HTML / form tag) is stored
        # here and flushed by _feed_secondary() AFTER state has been fully
        # reset to STATE_NORMAL — never inside _emit_processing_close().
        # This prevents re-entrant ss.feed() from matching the content as
        # a new secondary tag and emitting a duplicate <processing> block.
        # ---------------------------------------------------------------
        self.pending_final_content: str = ""

        self.affected_artefacts: List[Dict] = []
        self.patch_error_occurred: bool = False
        self.stream_buf: List[str]  = []
        self.clean_prose: List[str] = []   # only actual MSG_TYPE_CHUNK prose, no processing XML
        self.suppress_buf: List[str] = []
        self._in_suppress: bool = False
        self.reasoning_chunks_count = 0

    # ---------------------------------------------------------------- public entry point

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str):
            return True

        # ── STATUS MIMICRY TRAP ──
        # If the LLM starts generating framework-only tags, we trigger an abort.
        if "<processing" in chunk.lower():
            ASCIIColors.error(f"[Mimicry Trap] LLM attempted to generate internal status: {chunk!r}")
            self.patch_error_occurred = True # Trigger the execution failure logic
            self.proc_title = "Log Mimicry Detected"
            return False

        self.stream_buf.append(chunk)

        if self.proc_has_opened and self.proc_type == "agent_reasoning":
            self.reasoning_chunks_count += 1

            # Every 12-15 chunks, emit a comforting milestone status
            _REASONING_STATUSES = [
                (15,  "Analyzing conversation context & constraints..."),
                (35,  "Mapping technical goals & file structures..."),
                (60,  "Querying active local memories & learned guidelines..."),
                (90,  "Scanning workspace folder for relevant source files..."),
                (130, "Structuring coding plan and verifying logic..."),
                (180, "Formulating final specialized response strategy..."),
            ]
            for threshold, status_msg in _REASONING_STATUSES:
                if self.reasoning_chunks_count == threshold:
                    self._emit_processing_status(status_msg)
                    break

        pos = 0
        while pos < len(chunk):
            if self.state == self.STATE_NORMAL:
                pos = self._feed_normal(chunk, pos)

            elif self.state == self.STATE_BUFFERING:
                pos = self._feed_buffering(chunk, pos)
                if self.tool_trigger:
                    return False

            elif self.state == self.STATE_TOOL_CALL:
                pos = self._feed_tool_call(chunk, pos)
                if self.tool_trigger:
                    return False

            elif self.state == self.STATE_SECONDARY:
                pos = self._feed_secondary(chunk, pos)

            else:
                pos += 1

        return True

    # ---------------------------------------------------------------- passthrough relay

    def passthrough(self, chunk, msg_type=None, meta=None) -> bool:
        if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
            if msg_type in (MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, MSG_TYPE.MSG_TYPE_REASONING):
                self.ai_message.thoughts = (self.ai_message.thoughts or "") + (chunk or "")
            return _cb(self.callback, chunk, msg_type, meta)
        return True

    # ---------------------------------------------------------------- STATE_NORMAL

    def _feed_normal(self, chunk: str, pos: int) -> int:
        # Check for both '<' (XML tags) and '[' (suppress tokens)
        next_special = -1
        for c in ('<', '['):
            idx = chunk.find(c, pos)
            if idx != -1 and (next_special == -1 or idx < next_special):
                next_special = idx
                next_char = c

        if next_special == -1:
            text = chunk[pos:]
            if text:
                if self.proc_has_opened and self.proc_type == "agent_reasoning":
                    self._emit_processing_close()
                self.ai_message.content += text
                self.clean_prose.append(text)
                _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)
            return len(chunk)

        if next_special > pos:
            text = chunk[pos:next_special]
            if text:
                if self.proc_has_opened and self.proc_type == "agent_reasoning":
                    self._emit_processing_close()
                self.ai_message.content += text
                self.clean_prose.append(text)
                _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)

        if next_char == '<':
            self.state = self.STATE_BUFFERING
            self.bracket_buf = ["<"]
        else:  # '['
            self.state = self.STATE_BUFFERING
            self.bracket_buf = ["["]
            self._suppress_mode = True  # flag that we're in a [...] buffer
        return next_special + 1

    # ---------------------------------------------------------------- STATE_BUFFERING

    def _feed_buffering(self, chunk: str, pos: int) -> int:
        gt_idx = chunk.find(">", pos)

        if gt_idx != -1:
            self.bracket_buf.append(chunk[pos:gt_idx + 1])
            b_str = "".join(self.bracket_buf)
            new_pos = gt_idx + 1

            if "<tool_call>" in b_str:
                if self.proc_has_opened and self.proc_type == "agent_reasoning":
                    self._emit_processing_close()
                self.state = self.STATE_TOOL_CALL
                self.tool_buf = [b_str]
                return new_pos

            b_str_lower = b_str.lower()
            # Intercept self-closing memory tags instantly inside buffering
            if any(b_str_lower.startswith(prefix) for prefix in ("<mem_tag", "<mem_load", "<mem_delete")) or (
                b_str_lower.startswith("<mem_new") and b_str_lower.endswith("/>")
            ):
                if self.proc_has_opened and self.proc_type == "agent_reasoning":
                    self._emit_processing_close()
                self._handle_memory_tag_stream(b_str)
                self.bracket_buf.clear()
                self.state = self.STATE_NORMAL
                return new_pos

            matched_prefix = self._match_secondary_prefix(b_str)
            if matched_prefix:
                self.bracket_buf.clear()
                self.state = self.STATE_SECONDARY
                self._enter_secondary(b_str, matched_prefix)
                return new_pos

            self._flush_bracket_buf_as_text()
            self.state = self.STATE_NORMAL
            return new_pos

        else:
            self.bracket_buf.append(chunk[pos:])
            b_str = "".join(self.bracket_buf)

            if len(b_str) >= _MAX_BRACKET_BUF:
                ASCIIColors.warning(
                    f"[StreamState] Bracket buffer hard cap ({_MAX_BRACKET_BUF}) reached. "
                    "Flushing as plain text."
                )
                self._flush_bracket_buf_as_text()
                self.state = self.STATE_NORMAL
                return len(chunk)

            b_str_lower = b_str.lower()
            can_still_match = any(
                s.lower().startswith(b_str_lower) or b_str_lower.startswith(s[:len(b_str_lower)].lower())
                for s in _TAG_STARTS
            )
            if not can_still_match and len(b_str) > 1:
                self._flush_bracket_buf_as_text()
                self.state = self.STATE_NORMAL

            return len(chunk)

    def _flush_bracket_buf_as_text(self):
        text = "".join(self.bracket_buf)
        self.bracket_buf.clear()
        if text:
            if self.proc_has_opened and self.proc_type == "agent_reasoning":
                self._emit_processing_close()
            self.ai_message.content += text
            self.clean_prose.append(text)
            _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)

    def _handle_memory_tag_stream(self, tag_str: str) -> str:
        """Process a memory tag in real-time and output inline <processing> statuses."""
        mm = self.discussion.memory_manager
        if not mm:
            return ""

        tag_str = tag_str.strip()
        tag_lower = tag_str.lower()
        op_type = ""
        details = ""

        if tag_lower.startswith("<mem_tag"):
            m = mm._PAT_TAG.match(tag_str)
            if m:
                full_id = mm._resolve_id(m.group(1))
                if full_id:
                    res = mm.tag(full_id)
                    if res:
                        op_type = "memory_update"
                        details = f"Acknowledged/reinforced memory: '{res['content'][:40]}...'"
        elif tag_lower.startswith("<mem_load"):
            m = mm._PAT_LOAD.match(tag_str)
            if m:
                full_id = mm._resolve_id(m.group(1))
                if full_id:
                    res = mm.load_to_working(full_id)
                    if res:
                        op_type = "memory_update"
                        details = f"Loaded deep memory: '{res['content'][:40]}...'"
        elif tag_lower.startswith("<mem_delete"):
            m = mm._PAT_DELETE.match(tag_str)
            if m:
                full_id = mm._resolve_id(m.group(1))
                if full_id and mm.delete(full_id):
                    op_type = "memory_update"
                    details = f"Deleted memory ID: {m.group(1)}"
        elif tag_lower.startswith("<mem_new") and tag_str.endswith("/>"):
            m = mm._PAT_NEW.match(tag_str)
            if m:
                attrs = {attr_m.group(1).lower(): attr_m.group(2)
                         for attr_m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', m.group(1))}
                imp_str = attrs.get("importance")
                try:
                    imp = float(imp_str) if imp_str else mm.config.default_importance
                except ValueError:
                    imp = mm.config.default_importance
                content = attrs.get("content", "")
                if content:
                    res = mm.add(content.strip(), importance=max(0.0, min(1.0, imp)))
                    if res:
                        op_type = "memory_update"
                        details = f"Created new memory (Imp: {imp:.1%}): '{content[:40]}...'"

        if op_type and details:
            if self.enable_in_message_status:
                self.proc_type = op_type
                self.proc_title = "Memory System Update"
                self._emit_processing_open()
                self._emit_processing_status(details)
                self._emit_processing_close()
            # Send informational updates to callbacks
            _cb(self.callback, details, MSG_TYPE.MSG_TYPE_INFO, {"type": "memory_update", "report": {}})
            return tag_str
        return ""

    def _auto_pull_deep_memory_cues(self, user_message: str):
        """Auto-grep deep memories matching keywords in user_message."""
        # Clean punctuation to allow robust matches on words like 'language?'
        cleaned_text = re.sub(r'[^\w\s]', ' ', user_message)
        words = set(cleaned_text.lower().split())
        if not words:
            return

        mm = self.discussion.memory_manager
        if not mm:
            return

        matching_deep = mm.query(user_message, top_k=3, level=2)
        for m in matching_deep:
            mm.load_to_working(m["id"])
            ASCIIColors.cyan(
                f"[Memory] Detected deep memory cue — Auto-pulled '{m['id'][:8]}' "
                f"('{m['content'][:30]}...') into Working Memory."
            )

    def _match_secondary_prefix(self, b_str: str) -> Optional[str]:
        b_str_lower = b_str.lower()
        for prefix in _SECONDARY_TAG_MAP:
            if b_str_lower.startswith(prefix.lower()):
                if prefix in ("<artifact", "<artefact") and not self.enable_artefacts: continue
                if prefix in ("<note",)  and not self.enable_notes:           continue
                if prefix in ("<skill",) and not self.enable_skills:          continue
                if prefix in ("<lollms_inline",) and not self.enable_inline_widgets: continue
                if prefix in ("<lollms_form",)   and not self.enable_forms:   continue
                return prefix
        return None

    # ---------------------------------------------------------------- STATE_TOOL_CALL

    def _feed_tool_call(self, chunk: str, pos: int) -> int:
        self.tool_buf.append(chunk[pos:])
        accumulated = "".join(self.tool_buf)
        if "</tool_call>" in accumulated:
            self.tool_trigger = True
            return len(chunk)
        return len(chunk)

    # ---------------------------------------------------------------- Tool processing helpers

    def _emit_tool_processing_open(self, tool_name: str, params: Dict[str, Any]):
        """Emit opening <processing> tag for tool call execution with full types and details."""
        if not self.enable_in_message_status:
            return
        self.proc_type = "tool_execution"
        self.proc_title = f"Executing {tool_name.replace('_', ' ').title()}"
        params_str = json.dumps(params, ensure_ascii=False)[:200]
        params_escaped = params_str.replace('"', '&quot;')
        tag = f'<processing type="tool_execution" title="{self.proc_title}" tool="{tool_name}" params="{params_escaped}">\n'
        self.ai_message.content += tag
        _cb(self.callback, tag, MSG_TYPE.MSG_TYPE_CHUNK, {
            "type": "processing_open",
            "processing_type": "tool_execution",
            "title": self.proc_title,
            "tool": tool_name,
            "params": params,
            "was_processed": True,
        })
        self.proc_has_opened = True

    def _emit_tool_processing_status(self, status_text: str):
        """Stream a status update for tool execution."""
        line = f"* {status_text}\n"
        self.proc_content.append(line)
        if self.enable_in_message_status:
            if not self.proc_has_opened:
                return
            self.ai_message.content += line
            _cb(self.callback, line, MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_status",
                "processing_type": "tool_execution",
                "status": status_text,
                "was_processed": True,
            })

    def _emit_tool_processing_close(self, result_summary: str = ""):
        """Close the tool execution processing tag.

        Note: result_summary is plain text (no XML tags) so it is safe to emit
        directly via _cb without risk of re-entrant tag matching.
        """
        if not self.enable_in_message_status:
            return

        if not self.proc_has_opened:
            return

        close_tag = "</processing>\n\n"
        self.ai_message.content += close_tag
        _cb(self.callback, close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {
            "type": "processing_close",
            "processing_type": "tool_execution",
            "was_processed": True,
        })

        if result_summary:
            self.ai_message.content += result_summary
            _cb(self.callback, result_summary, MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_final_content",
                "processing_type": "tool_execution",
                "summary": result_summary,
            })

        self.proc_has_opened = False
        self.proc_type = ""

    # ---------------------------------------------------------------- STATE_SECONDARY

    def _enter_secondary(self, opening_tag: str, prefix: str):
        if self.proc_has_opened and self.proc_type == "agent_reasoning":
            self._emit_processing_close()

        ann_type, chunk_mt, done_mt, close_tag = _SECONDARY_TAG_MAP[prefix]

        self.sec_prefix     = prefix
        self.sec_chunk_mt   = chunk_mt
        self.sec_done_mt    = done_mt
        self.sec_close_tag  = close_tag
        self.sec_open_attrs = _parse_tag_attrs(opening_tag)
        self.sec_content    = []
        self.sec_close_scan = ""

        # Line-based status buffering (reset per secondary tag)
        self._status_line_buffer = ""
        self._status_lines_processed = 0
        self._status_context_lines = []

        # Map prefix → processing type for the status reporting
        proc_type_map = {
            "<artifact":      "artefact_building",
            "<artefact":      "artefact_building",
            "<coding_plan":   "coding_planning",
            "<note":          "note_building",
            "<skill":         "skill_building",
            "<lollms_inline": "widget_building",
            "<lollms_form":   "form_building",
            "<mem_new":       "memory_update",
            "<mem_update":    "memory_update",
        }

        # ── Determine Context: Building vs Editing ──
        raw_title = self.sec_open_attrs.get('name') or self.sec_open_attrs.get('title')

        # Enforce titles for specialized tags
        if not raw_title:
            if prefix == "<coding_plan":
                raw_title = "Strategic Planning"
            elif prefix in ("<mem_new", "<mem_update"):
                raw_title = "Memory System Update"
            else:
                raw_title = "Untitled Task"
        existing_titles = self.discussion.artefacts._all_latest_titles()
        is_update = raw_title in existing_titles or _find_best_title_match(raw_title, existing_titles) is not None

        if prefix in ("<artifact", "<artefact"):
            self.proc_type = "artefact_patching" if is_update else "artefact_building"
            label = "🔧 EDITING" if is_update else "🏗️ BUILDING"
            new_title = f"{label} ARTEFACT: {raw_title}"
        elif prefix in ("<mem_new", "<mem_update"):
            self.proc_type = "memory_update"
            new_title = "Memory System Update"
        else:
            self.proc_type = proc_type_map.get(prefix, "building")
            new_title = raw_title

        # If we are already in a processing block, just update the title/status instead of re-opening
        if self.proc_has_opened:
            if self.proc_title != new_title:
                self.proc_title = new_title
                self._emit_processing_status(f"Switching to: {new_title}")
        else:
            self.proc_title = new_title
            self._emit_processing_open()

        self.proc_attrs = {}
        if self.sec_open_attrs.get('id'):
            self.proc_attrs['id'] = self.sec_open_attrs.get('id')

        self.pending_final_content = ""

        # Build type-specific attrs for metadata
        if prefix in ("<artifact", "<artefact"):
            self.proc_attrs['art_type'] = self.sec_open_attrs.get('type', 'document')
            if self.sec_open_attrs.get('language'):
                self.proc_attrs['language'] = self.sec_open_attrs.get('language')

    def _feed_secondary(self, chunk: str, pos: int) -> int:
        close_tag = self.sec_close_tag
        close_len = len(close_tag)
        incoming  = chunk[pos:]

        self.sec_close_scan += incoming

        close_idx = self.sec_close_scan.find(close_tag)

        if close_idx != -1:
            pre_close  = self.sec_close_scan[:close_idx]
            post_close = self.sec_close_scan[close_idx + close_len:]

            if pre_close:
                self.sec_content.append(pre_close)
                self._fire_secondary_chunk(pre_close)

            # _fire_secondary_done() processes content, emits <processing> status lines,
            # calls _emit_processing_close() which stores final bulk content in
            # self.pending_final_content rather than emitting it via _cb.
            self._fire_secondary_done()

            # ── Full state reset ────────────────────────────────────────
            saved_proc_type  = self.proc_type   # preserve for pending_final_content meta
            self.sec_prefix     = ""
            self.sec_chunk_mt   = None
            self.sec_done_mt    = None
            self.sec_close_tag  = ""
            self.sec_open_attrs = {}
            self.sec_close_scan = ""
            self.state          = self.STATE_NORMAL

            # ── Flush pending final content (widget HTML / form tag / specialist tags) ──
            # KEY FIX: We must ensure this content DOES NOT re-trigger the state machine.
            # ── Flush pending final content (widget HTML / form tag / specialist tags) ──
            if self.pending_final_content:
                content = self.pending_final_content
                self.pending_final_content = ""

                # We append to content so it's saved in the message
                self.ai_message.content += content

                # If it is a specialist artifact or plan result, do NOT emit it as a raw MSG_TYPE_CHUNK chunk to the UI,
                # to prevent the WebUI's client-side parser from opening redundant/empty processing blocks.
                # Instead, we emit it on MSG_TYPE_ARTEFACT_DONE.
                if saved_proc_type in ("artefact_building", "artefact_patching", "coding_planning"):
                    _cb(self.callback, content, MSG_TYPE.MSG_TYPE_ARTEFACT_DONE, {
                        "type": "processing_final_content",
                        "processing_type": saved_proc_type,
                        "content_length": len(content),
                        "was_processed": True
                    })
                else:
                    # For other widget/form types, emit as normal chunk
                    _cb(self.callback, content, MSG_TYPE.MSG_TYPE_CHUNK, {
                        "type": "processing_final_content",
                        "processing_type": saved_proc_type,
                        "content_length": len(content),
                        "was_processed": True  # Sentinel to prevent double-execution
                    })

            # ── Continue with any text after the closing tag ────────────
            if post_close:
                self._feed_post_close(post_close)

        else:
            safe_len = max(0, len(self.sec_close_scan) - close_len + 1)
            safe_content = self.sec_close_scan[:safe_len]
            if safe_content:
                self.sec_content.append(safe_content)
                self._fire_secondary_chunk(safe_content)
                self.sec_close_scan = self.sec_close_scan[safe_len:]

        return len(chunk)

    def _feed_post_close(self, text: str):
        pos = 0
        while pos < len(text):
            if self.state == self.STATE_NORMAL:
                pos = self._feed_normal(text, pos)
            elif self.state == self.STATE_BUFFERING:
                pos = self._feed_buffering(text, pos)
                if self.tool_trigger:
                    break
            elif self.state == self.STATE_SECONDARY:
                pos = self._feed_secondary(text, pos)
            else:
                pos += 1

    # ---------------------------------------------------------------- Processing helpers

    def _emit_processing_open(self):
        """Emit the opening <processing> tag with attributes."""
        if not self.enable_in_message_status:
            return
        if self.proc_has_opened:
            return
        attrs_str = f' type="{self.proc_type}" title="{self.proc_title}"'
        for k, v in self.proc_attrs.items():
            if v:
                attrs_str += f' {k}="{v}"'
        tag = f"<processing{attrs_str}>\n"
        self.ai_message.content += tag
        _cb(self.callback, tag, MSG_TYPE.MSG_TYPE_CHUNK, {
            "type": "processing_open",
            "processing_type": self.proc_type,
            "title": self.proc_title,
            "attrs": self.proc_attrs,
            "was_processed": True,
        })
        self.proc_has_opened = True

    def _emit_processing_status(self, status_text: str):
        """Stream a status line inside the processing tag."""
        line = f"* {status_text}\n"
        self.proc_content.append(line)
        if self.enable_in_message_status:
            if not self.proc_has_opened:
                self._emit_processing_open()
            self.ai_message.content += line
            _cb(self.callback, line, MSG_TYPE.MSG_TYPE_CHUNK, {
                "type": "processing_status",
                "processing_type": self.proc_type,
                "status": status_text,
                "was_processed": True,
            })

    def _emit_processing_close(self, final_content: str = ""):
        """Close the <processing> tag only if it's open.

        NOTE: In the current 'sticky' architecture, this is usually called
        at the very end of the stream by flush_remaining_buffer() or if 
        we transition back to conversational text.
        """
        if not self.enable_in_message_status:
            if final_content:
                self.pending_final_content = final_content
            return

        if not self.proc_has_opened:
            if final_content:
                self.pending_final_content = final_content
            return

        close_tag = "</processing>\n\n"
        self.ai_message.content += close_tag
        _cb(self.callback, close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {
            "type": "processing_close",
            "processing_type": self.proc_type,
            "title": self.proc_title,
            "was_processed": True,
        })

        if final_content:
            self.pending_final_content = final_content

        self.proc_has_opened = False

    # ---------------------------------------------------------------- Secondary event helpers
    def _fire_secondary_chunk(self, content: str):
        """
        Buffer arriving secondary content and emit progress status messages.

        Strategy
        --------
        • Widgets / forms  : silent — no chunk events, no status spam.
        • Artefacts / notes: scan the accumulated buffer for NEW structural
          features (Markdown headers, class definitions, function definitions)
          and fire ONE status message per chunk at most — for the *first* new
          feature found in document order.  Falls back to size milestones only
          when no new structural feature exists in this chunk.

        Milestone tracking (`_fired_milestones`) is stored on the instance so
        it persists across chunks of the same stream, but is reset at the start
        of each new secondary tag (see _reset_secondary_state).
        """
        if not self.sec_chunk_mt or not content:
            return

        attrs  = self.sec_open_attrs
        prefix = self.sec_prefix

        # Widgets and forms: buffer silently
        if prefix in ("<lollms_inline", "<lollms_form"):
            return

        # Lazy-init milestone tracker (also reset by _reset_secondary_state)
        if not hasattr(self, '_fired_milestones'):
            self._fired_milestones = set()

        # ── LINE-BASED STATUS EMIT ───────────────────────────────────────────
        # Only emit status updates when complete lines have been received.
        # Buffer partial lines across chunks; process only on newline.
        if not hasattr(self, '_status_line_buffer'):
            self._status_line_buffer = ""
            self._status_lines_processed = 0

        self._status_line_buffer += content

        # Extract only newly-completed lines (those ending in \n)
        lines_to_process = []
        while '\n' in self._status_line_buffer:
            line, self._status_line_buffer = self._status_line_buffer.split('\n', 1)
            lines_to_process.append(line)

        # If no complete lines yet, skip status emission entirely
        if not lines_to_process:
            pass  # Will fall through to legacy chunk events below
        else:
            # Rebuild a pseudo-buffer from only the lines we're scanning now,
            # plus enough context from prior lines for regex anchoring.
            # We keep a sliding window of the last 5 lines for context.
            if not hasattr(self, '_status_context_lines'):
                self._status_context_lines = []
            self._status_context_lines.extend(lines_to_process)
            # Trim context window to last 50 lines to prevent unbounded growth
            self._status_context_lines = self._status_context_lines[-50:]

            scan_buffer = '\n'.join(self._status_context_lines)
            total_len = len("".join(self.sec_content)) + len(self._status_line_buffer)

            # ── 1. Structural feature detection ──────────────────────────────
            # Scan only newly-completed lines for features not yet reported.
            # Markdown headers
            for m in re.finditer(r'^(#{1,4})\s+(.+)$', scan_buffer, re.MULTILINE):
                key = f"feat:hdr:{m.group(2).strip()}"
                if key not in self._fired_milestones:
                    self._fired_milestones.add(key)
                    self._emit_processing_status(f"📖 Section: {m.group(2).strip()}")
                    break

            else:
                # Classes (only if no header was newly found above)
                for m in re.finditer(r'class\s+([a-zA-Z0-9_]+)', scan_buffer):
                    key = f"feat:cls:{m.group(1)}"
                    if key not in self._fired_milestones:
                        self._fired_milestones.add(key)
                        self._emit_processing_status(f"🏗️ Class: {m.group(1)}")
                        break
                else:
                    # Functions
                    for m in re.finditer(
                        r'(?:def|function)\s+([a-zA-Z0-9_]+)\s*\(', scan_buffer
                    ):
                        key = f"feat:fn:{m.group(1)}"
                        if key not in self._fired_milestones:
                            self._fired_milestones.add(key)
                            self._emit_processing_status(
                                f"⚙️ Implementing: {m.group(1)}()"
                            )
                            break
                    else:
                        # ── 2. Fallback size milestones ───────────────────────
                        # Only reached when no new structural feature exists.
                        _MILESTONES = (
                            (1_000,  "🔨 Refining content…"),
                            (5_000,  "🚀 Adding depth…"),
                            (15_000, "🏗️ Building substantial block…"),
                            (40_000, "🌊 Working on a massive document…"),
                        )
                        for threshold, message in _MILESTONES:
                            key = f"ms:{threshold}"
                            if total_len >= threshold and key not in self._fired_milestones:
                                self._fired_milestones.add(key)
                                self._emit_processing_status(message)
                                break

        # ── Legacy chunk events for artefacts, notes, skills ─────────────────
        if prefix in ("<artifact", "<artefact"):
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title":    attrs.get('name') or attrs.get('title', ''),
                "chunk":    content,
                "art_type": attrs.get('type', 'document'),
                "language": attrs.get('language'),
            })
        elif prefix == "<coding_plan":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "chunk": content,
            })
        elif prefix == "<note":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title": attrs.get('title') or attrs.get('name', 'Note'),
                "chunk": content,
            })
        elif prefix == "<skill":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title":       attrs.get('title') or attrs.get('name', 'Skill'),
                "chunk":       content,
                "category":    attrs.get('category', ''),
                "description": attrs.get('description', ''),
            })

    def _fire_secondary_done(self):
        """
        Process completed secondary content; emit processing status + close event.

        Artefact patch path
        -------------------
        When the content contains SEARCH/REPLACE markers, apply_aider_patch() is
        called against the current artefact content.  If the result is identical
        to the original (patch failed to match), a correction prompt is sent to
        the model (non-streaming, temperature=0.1) and the patched content is
        retried.  This loop runs up to _MAX_PATCH_RETRIES times.

        On the penultimate attempt the correction prompt explicitly allows a full
        rewrite.  After all retries are exhausted the system falls back to a full
        overwrite if the last model response contained no SEARCH markers, or
        preserves the original and sets self.patch_error_occurred = True if it did.

        Widget / form path
        ------------------
        The final bulk content is passed to _emit_processing_close() which stores
        it in self.pending_final_content.  _feed_secondary() flushes it after
        state reset — see class docstring.
        """
        if not self.sec_done_mt:
            return

        full_content = "".join(self.sec_content)
        attrs        = dict(self.sec_open_attrs)
        prefix       = self.sec_prefix

        # Reset milestone tracker so the next artefact starts fresh
        self._fired_milestones = set()
        self._status_line_buffer = ""
        self._status_lines_processed = 0
        self._status_context_lines = []

        def _fire_state_change(art, is_new):
            if not self.callback:
                return
            ev_type = "artifact_created" if is_new else "artifact_updated"
            _cb(
                self.callback,
                json.dumps({
                    "type":    ev_type,
                    "title":   art.get("title"),
                    "version": art.get("version"),
                    "art_type": art.get("type"),
                }),
                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                {"artefact": art, "is_new": is_new},
            )

        # ── Coding Plans ──────────────────────────────────────────────────────
        if prefix == "<coding_plan":
            self.proc_title = "Strategic Planning"
            self._emit_processing_status("Plan formulated. Invoking Artifact Specialist...")

            plan_text = full_content.strip()

            try:
                spec_result = self._execute_hyper_focused_artifact_update(plan_text)

                if spec_result:
                    self._emit_processing_status("Specialist implemented the plan successfully.")

                    # CRITICAL FIX: Store result AND emit it as final content
                    self.pending_final_content = spec_result

                    # LOG TO TURN SCRATCHPAD
                    if hasattr(self, "_turn_action_history"):
                        # Extract artifact name if possible for the summary
                        name_match = re.search(r'name=["\']([^"\']+)["\']', spec_result)
                        art_name = name_match.group(1) if name_match else "unnamed artifact"
                        self._turn_action_history.append(f"✓ SUCCESSFULLY implemented plan for: {art_name}")
                else:
                    self._emit_processing_status("⚠️ Specialist failed to generate valid tags.")
                    if hasattr(self, "_turn_action_history"):
                        self._turn_action_history.append("✗ FAILED to implement plan (No valid XML returned)")
            except Exception as e:
                self._emit_processing_status(f"❌ Specialist Error: {str(e)}")

            # CRITICAL FIX: Emit the specialist result as a tool output event so it's visible
            if hasattr(self, "_turn_action_history"):
                current_meta = dict(getattr(self.ai_message, 'metadata', {}) or {})
                events = current_meta.get("events", [])
                events.append({
                    "type": "tool_output", 
                    "content": spec_result if spec_result else "No result from specialist",
                    "id": str(uuid.uuid4()), 
                    "tool": "artifact_specialist",
                    "result": {"success": bool(spec_result)},
                    "offset": len(self.ai_message.content)
                })
                current_meta["events"] = events
                self.ai_message.metadata = current_meta

            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt, {"plan": plan_text})

        # ── Artefacts ─────────────────────────────────────────────────────────
        elif prefix in ("<artifact", "<artefact"):
            tag_title = attrs.pop('name', attrs.pop('title', 'untitled'))
            new_name  = attrs.pop('rename', None)
            atype     = attrs.pop('type', 'code')
            lang      = attrs.pop('language', None)
            attrs.pop('images', None)
            attrs.pop('image_media_types', None)

            commit_message = attrs.pop('commit_message', None)
            version_tags_raw = attrs.pop('version_tags', None)
            version_tags = [t.strip().lower() for t in version_tags_raw.split(',') if t.strip()] if version_tags_raw else None

            existing_titles  = self.discussion.artefacts._all_latest_titles()
            resolved_title   = tag_title if tag_title in existing_titles else (
                _find_best_title_match(tag_title, existing_titles) or tag_title
            )
            is_new   = resolved_title not in existing_titles
            is_patch = bool(re.search(r'<{6,8}\s*SEARCH', full_content, re.I))

            existing_before_update = self.discussion.artefacts.get(resolved_title)
            if existing_before_update and existing_before_update.get("type") == "data":
                self._emit_processing_status(
                    f"❌ REJECTED: Standard text/markdown editing is forbidden on Data Interface '{resolved_title}'. "
                    "You must use the 'execute_python_data_query' tool to modify this dataset."
                )
                self.patch_error_occurred = True
                self._emit_processing_close()
                return

            self._emit_processing_status(
                f"{'Creating new' if is_new else 'Updating'} artefact '{resolved_title}'"
            )

            # ── CAPTURE STATE BEFORE UPDATE ──
            original_content_snapshot = existing_before_update.get('content', '') if existing_before_update else None
            existing_before_update = self.discussion.artefacts.get(resolved_title)

            result_art = None

            # ── Surgical patch path ───────────────────────────────────────────
            if is_patch and not is_new:
                existing = existing_before_update
                if existing is None:
                    self._emit_processing_status(
                        f"WARNING: '{resolved_title}' not found for patch; creating new"
                    )
                    # Clean leaked sentinels from raw content if creating new from a failed patch
                    clean_content = re.sub(r'<{5,8}\s*SEARCH.*?={5,8}.*?>{5,8}\s*REPLACE', '', full_content, flags=re.DOTALL).strip()
                    result_art = self.discussion.artefacts.add(
                        resolved_title, atype, clean_content,
                        language=lang, active=self.auto_activate,
                        commit_message=commit_message, version_tags=version_tags,
                        **attrs,
                    )
                else:
                    patch_content  = full_content
                    original_text  = existing.get('content', '')
                    patch_accepted = False

                    for attempt in range(1, _MAX_PATCH_RETRIES + 1):
                        self._emit_processing_status(
                            f"Applying patch (attempt {attempt}/{_MAX_PATCH_RETRIES})…"
                        )
                        err_text = None
                        try:
                            patched = self.discussion.artefacts.apply_aider_patch(
                                original_text, patch_content
                            )
                        except ValueError as exc:
                            patched  = original_text
                            err_text = str(exc)
                            first_line = err_text.splitlines()[0]
                            self._emit_processing_status(
                                f"❌ Match Failed: {first_line}"
                            )

                        if patched != original_text:
                            old_lines = original_text.splitlines()
                            new_lines = patched.splitlines()
                            added   = sum(1 for l in new_lines if l not in old_lines)
                            removed = sum(1 for l in old_lines if l not in new_lines)
                            self._emit_processing_status(
                                f"✅ Patch accepted — +{added} / -{removed} lines"
                            )
                            result_art = self.discussion.artefacts.update(
                                resolved_title,
                                new_content=patched,
                                new_title=new_name,
                                language=lang,
                                active=self.auto_activate,
                                commit_message=commit_message,
                                version_tags=version_tags,
                                **attrs,
                            )
                            patch_accepted = True
                            break

                        # ── Patch failed — request correction ─────────────────
                        self._emit_processing_status(
                            f"Patch REJECTED (attempt {attempt}): "
                            "SEARCH block did not match current content"
                        )

                        if attempt >= _MAX_PATCH_RETRIES:
                            break

                        # On the last retry, explicitly allow a full rewrite
                        current_file_map = original_text
                        if attempt == _MAX_PATCH_RETRIES - 1:
                            current_file_map += (
                                "\n\n[!!! FINAL ATTEMPT !!!]\n"
                                "Your patches are failing to match the anchors.\n"
                                "FOR THIS TURN ONLY: Provide the COMPLETE and FINAL version of the file.\n"
                                "Wrap the entire file content in the <artifact> tag.\n"
                                "Do NOT use SEARCH/REPLACE blocks now."
                            )

                        # Extract diagnostic hints from the patch error
                        expected_hint = "Unknown"
                        closest_hint  = "Unknown"
                        if err_text:
                            for line in err_text.splitlines():
                                if "Expected first line :" in line:
                                    expected_hint = line.split(":", 1)[1].strip().strip("'")
                                elif "Closest line found  :" in line:
                                    closest_hint = line.split(":", 1)[1].strip().strip("'")

                        correction_prompt = _PATCH_RETRY_PROMPT_TEMPLATE.format(
                            title=resolved_title,
                            expected_first_line=expected_hint,
                            closest_line=closest_hint,
                            current_content=current_file_map,
                        )
                        self._emit_processing_status(
                            f"Requesting corrected patch from model…"
                        )
                        try:
                            retry_raw = self.discussion.lollmsClient.generate_text(
                                correction_prompt,
                                n_predict=min(2048, len(original_text) + 512),
                                temperature=0.1,
                            )
                            art_match = re.search(
                                r'<art[ei]fact[^>]*>(.*?)</art[ei]fact>',
                                retry_raw, re.DOTALL | re.IGNORECASE,
                            )
                            if art_match:
                                patch_content = art_match.group(1)
                                self._emit_processing_status(
                                    "Corrected patch received — retrying…"
                                )
                            else:
                                self._emit_processing_status(
                                    "Model did not return a valid patch tag — aborting retry"
                                )
                                break
                        except Exception as retry_exc:
                            self._emit_processing_status(
                                f"Retry generation failed: {str(retry_exc)[:80]}"
                            )
                            break

                    if not patch_accepted:
                        self._emit_processing_status(
                            f"All {_MAX_PATCH_RETRIES} patch attempts failed. "
                            "Falling back to full content overwrite."
                        )
                        has_search     = bool(re.search(r'<{6,8}\s*SEARCH', patch_content, re.I))
                        fallback_content = original_text if has_search else patch_content

                        if fallback_content.strip() == original_text.strip():
                            result_art = None # Force it to be None so it's not added to affected_artefacts
                            self.patch_error_occurred = True
                            self._emit_processing_status(
                                "❌ No changes detected: Content remains identical to original."
                            )
                            if hasattr(self, "_turn_action_history"):
                                self._turn_action_history.append(f"✗ FAILED: Specialist returned content identical to the original for '{resolved_title}'. Mission NOT accomplished.")
                        else:
                            result_art = self.discussion.artefacts.update(
                                resolved_title,
                                new_content=fallback_content,
                                new_title=new_name,
                                language=lang,
                                active=self.auto_activate,
                                commit_message=commit_message,
                                version_tags=version_tags,
                                **attrs,
                            )
                            self._emit_processing_status(
                                "✅ Full content overwrite applied"
                            )
                            if hasattr(self, "_turn_action_history"):
                                self._turn_action_history.append(f"✓ SUCCESSFULLY updated {resolved_title} via full rewrite")

            # ── Patch on a new artefact (shouldn't happen, but handle it) ─────
            elif is_patch and is_new:
                self._emit_processing_status(
                    f"NOTE: Patch requested for new artefact '{resolved_title}'; "
                    "creating with raw content"
                )
                # Clean leaked sentinels from raw content if creating new from a failed patch
                clean_content = re.sub(r'<{5,8}\s*SEARCH.*?={5,8}.*?>{5,8}\s*REPLACE', '', full_content, flags=re.DOTALL).strip()
                result_art = self.discussion.artefacts.add(
                    resolved_title, atype, clean_content,
                    language=lang, active=self.auto_activate,
                    commit_message=commit_message, version_tags=version_tags,
                    **attrs,
                )

            # ── Full content path ─────────────────────────────────────────────
            else:
                if is_new:
                    result_art = self.discussion.artefacts.add(
                        resolved_title, atype, full_content.strip(),
                        language=lang, active=self.auto_activate,
                        commit_message=commit_message, version_tags=version_tags,
                        **attrs,
                    )
                else:
                    self._emit_processing_status(
                        f"Full rewrite of '{resolved_title}' "
                        f"({len(full_content.splitlines())} lines)"
                    )
                    result_art = self.discussion.artefacts.update(
                        resolved_title,
                        new_content=full_content.strip(),
                        new_title=new_name,
                        new_type=atype,
                        language=lang,
                        active=self.auto_activate,
                        commit_message=commit_message,
                        version_tags=version_tags,
                        **attrs,
                    )

            # ── INTEGRITY VERIFICATION ──
            has_real_changes = False
            if result_art:
                def _clean_compare(s): return (s or "").replace('\r\n', '\n').strip()
                new_c = _clean_compare(result_art.get("content", ""))
                old_c = _clean_compare(original_content_snapshot)

                if new_c != old_c or result_art.get("title") != resolved_title:
                    has_real_changes = True
                    diff = len(new_c) - len(old_c)
                    ASCIIColors.green(f"  [Integrity] Change confirmed: {diff:+} chars.")
                else:
                    # If we had a patch accepted but content is same, it was a null-op
                    ASCIIColors.red("  [Integrity] FAILED: Content is binary-identical to previous version.")

            # Process success if we have a new artifact or confirmed changes
            # Process success if we have a new artifact, confirmed changes, or it is a full overwrite (not a patch)
            if result_art and (is_new or has_real_changes or not is_patch):
                self.affected_artefacts.append(result_art)
                current_meta = dict(self.ai_message.metadata or {})
                mod_list = current_meta.get("artefacts_modified", [])
                if result_art.get("title") not in mod_list:
                    mod_list.append(result_art.get("title"))
                current_meta["artefacts_modified"] = mod_list
                self.ai_message.metadata = current_meta

                _fire_state_change(result_art, is_new)
                self._emit_processing_status(
                    f"Artefact saved as version {result_art.get('version', '?')}"
                )
                if hasattr(self, "_turn_action_history"):
                    self._turn_action_history.append(f"✓ SUCCESSFULLY updated '{resolved_title}' to v{result_art.get('version')}")
            else:
                # Fallback path: No real changes detected on a patch
                self.patch_error_occurred = True
                self._emit_processing_status(
                    "❌ No changes detected: Content remains identical to original."
                )
                if hasattr(self, "_turn_action_history"):
                    error_msg = f"Flux FAILED: Specialist returned content for '{resolved_title}' that is binary-identical to the current version. No changes were saved."
                    self._turn_action_history.append(error_msg)
                    ASCIIColors.error(f"[Master Feedback] {error_msg}")

                    
            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt, {
                "title":    resolved_title,
                "content":  full_content,
                "art_type": atype,
                "language": lang,
                "is_patch": is_patch,
                "attrs":    attrs,
            })

        # ── Memory (Standard Tags with inner content) ─────────────────────────
        elif prefix == "<mem_new":
            tag_title = attrs.get('importance')
            try:
                imp = float(tag_title) if tag_title else self.discussion.memory_manager.config.default_importance
            except ValueError:
                imp = self.discussion.memory_manager.config.default_importance

            res = self.discussion.memory_manager.add(full_content.strip(), importance=max(0.0, min(1.0, imp)))
            if res:
                self.affected_artefacts.append(res)
                self._emit_processing_status(f"Created new memory (Imp: {imp:.1%}): '{full_content[:40]}...'")
            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt, {"importance": imp, "content": full_content})

        elif prefix == "<mem_update":
            tag_id = attrs.get('id')
            if tag_id:
                full_id = self.discussion.memory_manager._resolve_id(tag_id)
                if full_id:
                    res = self.discussion.memory_manager.update(full_id, full_content.strip())
                    if res:
                        self.affected_artefacts.append(res)
                        self._emit_processing_status(f"Updated memory [{tag_id[:8]}]: '{full_content[:40]}...'")
            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt, {"id": tag_id, "content": full_content})

        # ── Notes ─────────────────────────────────────────────────────────────
        elif prefix == "<note":
            title = attrs.get('title') or attrs.get('name', f'note_{uuid.uuid4().hex[:8]}')
            self._emit_processing_status(f"Creating note '{title}'")
            art = self.discussion.artefacts.add(
                title=title, artefact_type=ArtefactType.NOTE,
                content=full_content.strip(), active=self.auto_activate,
            )
            self.affected_artefacts.append(art)
            _fire_state_change(art, True)
            self._emit_processing_status("Note saved successfully")
            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt,
                {"title": title, "content": full_content})

        # ── Skills ────────────────────────────────────────────────────────────
        elif prefix == "<skill":
            title = attrs.get('title') or attrs.get('name', f'skill_{uuid.uuid4().hex[:8]}')
            desc  = attrs.get('description', '')
            cat   = attrs.get('category', '')
            self._emit_processing_status(f"Creating skill '{title}'")
            if cat:
                self._emit_processing_status(f"Category: {cat}")
            art = self.discussion.artefacts.add(
                title=title, artefact_type=ArtefactType.SKILL,
                content=full_content.strip(), active=self.auto_activate,
                description=desc, category=cat,
            )
            self.affected_artefacts.append(art)
            _fire_state_change(art, True)
            self._emit_processing_status("Skill saved successfully")
            self._emit_processing_close()
            _cb(self.callback, full_content, self.sec_done_mt,
                {"title": title, "content": full_content, "category": cat, "description": desc})

        # ── Inline widgets ────────────────────────────────────────────────────
        elif prefix == "<lollms_inline":
            title       = attrs.get('title') or attrs.get('name', f'widget_{uuid.uuid4().hex[:8]}')
            widget_type = attrs.get('type', 'html')
            self._emit_processing_status(f"Building {widget_type} widget '{title}'")
            self._emit_processing_status("Validating HTML/CSS/JS content…")
            validated = _validate_widget_content(full_content, title)
            if validated is None:
                self._emit_processing_status("Validation failed — widget discarded")
                self._emit_processing_close()
                _cb(self.callback, "", self.sec_done_mt, {
                    "title": title, "content": "", "widget_type": widget_type,
                    "error": "Validation failed",
                })
            else:
                self._emit_processing_status("Validation passed")
                wrapped = (
                    f'\n\n<lollms_inline title="{title}" type="{widget_type}">\n'
                    f'{validated}\n</lollms_inline>\n\n'
                )
                self._emit_processing_close(wrapped)
                _cb(self.callback, validated, self.sec_done_mt,
                    {"title": title, "content": validated, "widget_type": widget_type})

        # ── Forms ─────────────────────────────────────────────────────────────
        elif prefix == "<lollms_form":
            title = attrs.get('title') or attrs.get('name', f'form_{uuid.uuid4().hex[:8]}')
            self._emit_processing_status(f"Building form '{title}'")
            self._emit_processing_status("Parsing form fields…")

            form_descriptor = _parse_form_xml(
                ' '.join(f'{k}="{v}"' for k, v in attrs.items()),
                full_content,
            )

            if form_descriptor and form_descriptor.get('fields'):
                form_id = form_descriptor.get("id") or str(uuid.uuid4())
                form_descriptor["id"] = form_id
                self.discussion._get_pending_forms()[form_id] = form_descriptor

                n = len(form_descriptor['fields'])
                self._emit_processing_status(f"Found {n} field(s)")

                form_attrs_parts = []
                form_attrs_parts.append(f'id="{form_id}"')
                if title:
                    form_attrs_parts.append(f'title="{title}"')
                if attrs.get('description'):
                    form_attrs_parts.append(f'description="{attrs["description"]}"')
                if attrs.get('submit_label'):
                    form_attrs_parts.append(f'submit_label="{attrs["submit_label"]}"')

                field_lines = []
                for f in form_descriptor['fields']:
                    attrs_parts = [
                        f'name="{f["name"]}"',
                        f'label="{f["label"]}"',
                        f'type="{f["type"]}"'
                    ]
                    if f.get('required'):
                        attrs_parts.append('required="true"')
                    if f.get('options'):
                        opts_str = ",".join(f['options']) if isinstance(f['options'], list) else str(f['options'])
                        attrs_parts.append(f'options="{opts_str}"')
                    for k in ('default', 'placeholder', 'min', 'max', 'step'):
                        if f.get(k) is not None:
                            attrs_parts.append(f'{k}="{f[k]}"')
                    field_lines.append('  <field ' + ' '.join(attrs_parts) + ' />')
                sep = " " if form_attrs_parts else ""
                full_form_tag = (
                    f'<lollms_form{sep}{" ".join(form_attrs_parts)}>\n'
                    + ('\n'.join(field_lines) + '\n' if field_lines else '')
                    + '</lollms_form>'
                )
                self._emit_processing_status("Form ready")
                self._emit_processing_close(full_form_tag)
                _cb(self.callback, full_content, self.sec_done_mt,
                    {"title": title, "content": full_content, "form": form_descriptor})
            else:
                self._emit_processing_status("Form parsing failed")
                self._emit_processing_close()
                _cb(self.callback, "", self.sec_done_mt,
                    {"title": title, "content": "", "error": "Form parsing failed"})   # ---------------------------------------------------------------- accessors

    def get_accumulated_stream(self) -> str:
        return "".join(self.stream_buf)

    def get_tool_call_json(self) -> Optional[str]:
        if not self.tool_trigger:
            return None
        raw = "".join(self.tool_buf)
        open_idx  = raw.find("<tool_call>")
        close_idx = raw.find("</tool_call>")
        if open_idx == -1 or close_idx == -1:
            return None
        return raw[open_idx + len("<tool_call>"):close_idx].strip()

    def get_clean_text_so_far(self) -> str:
        return self.ai_message.content

    def _execute_hyper_focused_artifact_update(self, plan: str) -> str:
        """
        Invokes a specialized LLM call with a precision persona built on the fly.
        Handles Code, Markdown, and Text with domain-specific accuracy.
        """
        # 1. Parse requested persona and task parameters from plan or derive from context
        persona_match = re.search(r"Specialist Persona:\s*([^\n\r]+)", plan, re.I)
        requested_persona = persona_match.group(1).strip() if persona_match else "Surgical Artifact Specialist"

        goal_match = re.search(r"Goal:\s*([^\n\r]+)", plan, re.I)
        goal = goal_match.group(1).strip() if goal_match else "Implement requested modifications."

        details_match = re.search(r"Implementation Details:\s*([^\n\r]+)", plan, re.I)
        details = details_match.group(1).strip() if details_match else "Surgically apply plan logic."

        # 2. Extract parent custom personality traits to inject into spinoff persona
        parent_prompt = self.discussion.system_prompt or ""
        # Strip system system-instructions to isolate the pure custom personality traits (style, role, core guidelines)
        custom_traits = parent_prompt.split("=== ARTIFACT SYSTEM ===")[0].strip()
        is_generic = not custom_traits or "You are Lollms" in custom_traits or "You are a helpful" in custom_traits or len(custom_traits) < 50

        inherited_traits_block = ""
        if not is_generic:
            inherited_traits_block = f"=== INHERITED PERSONALITY TRAITS ===\n{custom_traits}\n"

        # 3. Gather Selective Memory Context
        memory_block = ""
        selected_mem_list = []
        mm = getattr(self.discussion, 'memory_manager', None)
        if mm:
            # Parse requested memory IDs from the plan
            mem_ids_match = re.search(r"Relevant Memories:\s*([^\n\r]+)", plan, re.I)
            if mem_ids_match:
                requested_ids = [idx.strip() for idx in mem_ids_match.group(1).split(",") if idx.strip()]
                for rid in requested_ids:
                    full_id = mm._resolve_id(rid)
                    if full_id:
                        m_data = mm.get(full_id)
                        if m_data:
                            selected_mem_list.append(f"[{m_data['id'][:8]}] {m_data['content']}")

                if selected_mem_list:
                    memory_block = "=== SELECTIVE PROJECT MEMORIES ===\n" + "\n".join(selected_mem_list) + "\n"

        # 4. Parse selected artifacts, supporting context, and skills from plan to construct focused data context
        target_match = re.search(r"Target Artifacts:\s*([^\n\r]+)", plan, re.I)
        supporting_match = re.search(r"Supporting Context:\s*([^\n\r]+)", plan, re.I)
        skills_match = re.search(r"Relevant Skills:\s*([^\n\r]+)", plan, re.I)

        target_names = [name.strip() for name in target_match.group(1).split(",") if name.strip()] if target_match else []
        supporting_names = [name.strip() for name in supporting_match.group(1).split(",") if name.strip()] if supporting_match else []
        skill_names = [name.strip() for name in skills_match.group(1).split(",") if name.strip()] if skills_match else []

        existing_titles = self.discussion.artefacts._all_latest_titles()

        resolved_targets = []
        for name in target_names:
            resolved = name if name in existing_titles else _find_best_title_match(name, existing_titles)
            if resolved:
                resolved_targets.append(resolved)
            else:
                resolved_targets.append(name)

        resolved_supporting = []
        for name in supporting_names:
            resolved = name if name in existing_titles else _find_best_title_match(name, existing_titles)
            if resolved:
                resolved_supporting.append(resolved)

        resolved_skills = []
        for name in skill_names:
            resolved = name if name in existing_titles else _find_best_title_match(name, existing_titles)
            if resolved:
                resolved_skills.append(resolved)

        # Build selective context (only including assessed relevant content to reduce context noise)
        parts = []
        udz = (self.discussion.user_data_zone or "").strip()
        ddz = (self.discussion.discussion_data_zone or "").strip()
        pdz = (self.discussion.personality_data_zone or "").strip()

        if udz:
            parts.append(f"-- User Data Zone --\n{udz}")
        if ddz:
            parts.append(f"-- Discussion Data Zone --\n{ddz}")
        if pdz:
            parts.append(f"-- Personality Data Zone --\n{pdz}")

        all_selected_titles = set(resolved_targets + resolved_supporting + resolved_skills)
        if all_selected_titles:
            parts.append("## Selected Artifacts & Skills")
            for title in all_selected_titles:
                art = self.discussion.artefacts.get(title)
                if art:
                    atype = art.get('type', 'document')
                    lang = art.get('language') or ''
                    fence = f"```{lang}" if lang else "```"
                    content_text = art.get('content', '').strip()
                    label = ArtefactType.LABELS.get(atype, atype.capitalize())
                    header = f"###[{label}] {art['title']} (v{art['version']})"
                    if atype == ArtefactType.CODE or (lang and "<artefact_image" not in content_text):
                        parts.append(f"{header}\n{fence}\n{content_text}\n```")
                    else:
                        parts.append(f"{header}\n{content_text}")

        full_context = "\n\n".join(parts)

        # 5. Build the On-The-Fly Spinoff System Prompt
        target_example = resolved_targets[0] if resolved_targets else "example.py"
        surgical_prompt = (
            f"You are the {requested_persona}.\n"
            "You are a specialized spinoff spinoff persona spawned by the main system architect specifically to complete this implementation.\n"
            "You operate in a hyper-focused sandbox environment isolated from the main conversation's noise.\n\n"
            "Your ONLY task is to implement the provided PLAN by outputting updated <artifact> tags.\n"
            "DO NOT create widgets, notes, or forms. DO NOT use conversational prose.\n\n"
            f"{inherited_traits_block}"
            "=== TASK-SPECIFIC DEFINITION ===\n"
            f"• Goal: {goal}\n"
            f"• Details: {details}\n\n"
            "CORE PROTOCOL:\n"
            "- MEMORY: If a [SELECTIVE PROJECT MEMORIES] block is provided, you MUST adhere to those constraints.\n"
            "- TARGET ARTEFACTS: These are the files you must edit or create.\n"
            "- SUPPORTING CONTEXT: These artifacts contain reference data (API docs, specs). Do NOT edit these; use them as the 'Source of Truth' to inform your changes to the Targets.\n"
            "- If the plan concerns Code: Ensure idiomatic accuracy and architectural consistency.\n"
            "- If the plan concerns Markdown/Text: Preserve tone, formatting, and structural depth.\n\n"
            "STRICT CONSTRAINTS:\n"
            f"1. Output ONLY <artifact> tags. You MUST specify the name attribute matching the target file exactly, e.g. <artifact name=\"{target_example}\" type=\"code\">\n"
            "2. For modifying existing artifacts, you MUST use SEARCH/REPLACE blocks inside the tag.\n"
            "3. No conversation, no explanations, no markdown fences OUTSIDE the tags.\n"
            "4. Provide exactly 2 lines of context in SEARCH blocks to ensure match uniqueness.\n"
            "5. For new artifacts, provide 100% of the content.\n"
            "6. Match indentation, punctuation, and blank lines character-for-character.\n"
            "7. VISION CHECK: If images are provided, they have been pre-filtered by the main agent as relevant. \n"
            "   However, you MUST still verify their relevance to the specific code task. \n"
            "   If an image is clearly unrelated (e.g. a bunny when fixing code), IGNORE IT completely.\n"
            "   Do not let irrelevant images distract you from the code architecture.\n"
        )

        # Ingest active conversation images (e.g. error screenshots) to forward to the specialist
        active_images = self.discussion.get_active_images()
        filtered_images = []

        # ── VISION FILTERING PROTOCOL ──
        # Before forwarding images to the specialist, the main agent must evaluate
        # their relevance to the current goal. This prevents "Vision Pollution" where
        # irrelevant images (like a bunny rabbit) distract the specialist from code fixes.
        if active_images:
            self._emit_processing_status(f"👁️ Evaluating {len(active_images)} active image(s) for relevance...")

            # Build a concise summary of the current task for the filter
            filter_prompt = (
                f"TASK CONTEXT: {goal}\n"
                f"PLAN DETAILS: {details}\n"
                f"TARGET ARTIFACTS: {', '.join(resolved_targets) if resolved_targets else 'None'}\n"
                "\n"
                "INSTRUCTION: \n"
                "You have {n} image(s) attached to the current conversation context.\n"
                "Determine if ANY of these images are CRITICALLY relevant to the task above.\n"
                "- Relevant: Error screenshots, diagrams of the target architecture, UI bugs in the target file.\n"
                "- Irrelevant: Decorative images, previous unrelated generations, generic assets.\n"
                "\n"
                "Output format: JSON object with key 'relevant_image_indices' (list of 0-based indices).\n"
                "If no images are relevant, return an empty list.\n"
                "Example: {\"relevant_image_indices\": [0, 2]}"
            )

            try:
                # Ask the main client (which has vision) to filter
                filter_result = self.discussion.lollmsClient.generate_structured_content(
                    prompt=filter_prompt,
                    schema={
                        "relevant_image_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of 0-based indices of images relevant to the current task."
                        }
                    },
                    temperature=0.0, # Deterministic decision
                    images=active_images # Pass all images for evaluation
                )

                if filter_result and isinstance(filter_result, dict):
                    indices = filter_result.get("relevant_image_indices", [])
                    if isinstance(indices, list):
                        filtered_images = [active_images[i] for i in indices if 0 <= i < len(active_images)]
                        if filtered_images:
                            self._emit_processing_status(f"✅ {len(filtered_images)} image(s) selected as relevant.")
                        else:
                            self._emit_processing_status("🚫 No images deemed relevant to current task.")
                    else:
                        # Fallback: If schema parsing fails but images exist, keep them to be safe
                        filtered_images = active_images
                        self._emit_processing_status("⚠️ Filter parsing failed, keeping all images.")
                else:
                    # Fallback: Keep all images if no result
                    filtered_images = active_images
                    self._emit_processing_status("⚠️ Filter returned empty, keeping all images.")
            except Exception as filter_err:
                # Fallback on error: keep all images to avoid losing critical info
                filtered_images = active_images
                self._emit_processing_status(f"⚠️ Image filter failed: {str(filter_err)[:50]}. Keeping all images.")
        else:
            self._emit_processing_status("👁️ No active images to evaluate.")

        # Report the agent details and selective choices via the processing block if active or callback
        self._emit_processing_status(f"Spawning specialized spinoff agent: '{requested_persona}'")
        self._emit_processing_status(f"User Intent: {goal}")
        self._emit_processing_status(f"Target Artifacts to edit: {', '.join(resolved_targets) if resolved_targets else 'None'}")
        self._emit_processing_status(f"Supporting Context provided: {', '.join(resolved_supporting) if resolved_supporting else 'None'}")
        self._emit_processing_status(f"Skills provided: {', '.join(resolved_skills) if resolved_skills else 'None'}")
        self._emit_processing_status(f"Memories provided: {', '.join(selected_mem_list) if selected_mem_list else 'None'}")
        self._emit_processing_status(f"Images forwarded to agent: {len(active_images) if active_images else 0}")
        self._emit_processing_status(f"Parent Scratchpad size: {len(getattr(self.discussion, 'scratchpad', '') or '')} characters")

        # 6. Build the payload
        user_payload = []
        if memory_block:
            user_payload.append(memory_block)

        # Inject the scratchpad containing sequential reading summaries if populated
        if getattr(self.discussion, "scratchpad", "").strip():
            user_payload.append(f"=== RECENT SEARCH & SCAN RESULTS ===\n{self.discussion.scratchpad.strip()}")

        user_payload.append(f"=== CONTEXTUAL ARTEFACTS ===\n{full_context}")
        user_payload.append(f"=== PLAN TO EXECUTE ===\n{plan}")
        user_payload.append("Execute the plan now. Output only the required XML tags.")

        final_payload = "\n\n".join(user_payload)

        # ── DEBUG LOG: Show exactly what the implementation specialist sees ──
        ASCIIColors.info("--- [DEBUG] HYPER-FOCUSED SPECIALIST INPUT ---")
        ASCIIColors.cyan(f"  • Persona             : {requested_persona}")
        cond_preview = custom_traits[:120].replace('\n', ' ') + "..." if len(custom_traits) > 120 else custom_traits.replace('\n', ' ')
        ASCIIColors.cyan(f"  • Conditioning Snippet: {cond_preview}")
        ASCIIColors.cyan(f"  • Skills Selected     : {', '.join(resolved_skills) if resolved_skills else 'None'}")
        ASCIIColors.cyan(f"  • Target Artefacts    : {', '.join(resolved_targets) if resolved_targets else 'None'}")
        ASCIIColors.cyan(f"  • Support Artefacts   : {', '.join(resolved_supporting) if resolved_supporting else 'None'}")
        ASCIIColors.cyan(f"  • Images Forwarded    : {len(filtered_images) if filtered_images else 0}")
        ASCIIColors.yellow(f"  • Payload Size        : {len(final_payload):,} chars")
        if self.discussion.lollmsClient.debug:
            ASCIIColors.white("\n--- FULL PAYLOAD ---")
            print(final_payload)
            ASCIIColors.white("--------------------\n")
        ASCIIColors.info("----------------------------------------------")

        # 6. Call LLM at low temperature for maximum reliability
        # Use filtered_images instead of active_images
        effective_images = filtered_images if filtered_images else None
        if effective_images:
            self._emit_processing_status(f"📤 Forwarding {len(effective_images)} relevant image(s) to specialist.")

        try:
            raw_output = self.discussion.lollmsClient.generate_text(
                prompt=final_payload,
                system_prompt=surgical_prompt,
                images=effective_images,
                temperature=None,
                n_predict=None
            )

            # ── DEBUG LOG: Show Raw Specialist Response ──
            ASCIIColors.info("--- [DEBUG] SPECIALIST RAW OUTPUT ---")
            ASCIIColors.yellow(f"Output Length: {len(raw_output or '')} chars")
            if self.discussion.lollmsClient.debug:
                ASCIIColors.white(raw_output)
            ASCIIColors.info("-------------------------------------")

            # 7. ANTI-HALLUCINATION FILTER
            # Extract expected targets from the plan
            targets_match = re.search(r"Target Artifacts:\s*([^\n\r]+)", plan, re.I)
            if targets_match:
                # Get list of valid names
                allowed_names = [t.strip().lower() for t in targets_match.group(1).split(",")]

                # Regex to find all artifact tags in output
                tag_pattern = re.compile(r'(<art[ei]fact\s+[^>]*>.*?</art[ei]fact>)', re.DOTALL | re.IGNORECASE)
                valid_tags = []

                for tag_match in tag_pattern.finditer(raw_output):
                    full_tag = tag_match.group(1)
                    name_attr = re.search(r'(?:name|title)=["\']([^"\']+)["\']', full_tag, re.I)
                    if name_attr:
                        tag_name = name_attr.group(1).strip().lower()
                        # STRICT MATCH: Only allow if explicitly in targets
                        is_allowed = any((a in tag_name or tag_name in a) for a in allowed_names if a)

                        if is_allowed:
                            valid_tags.append(full_tag)
                        else:
                            # Log discarded but don't save
                            ASCIIColors.warning(f"Discarding intermediate/unplanned artifact: {tag_name}")
                    elif len(allowed_names) == 1:
                        valid_tags.append(full_tag)

                # Reconstruct output with only allowed tags
                if valid_tags:
                    sanitized_output = "\n\n".join(valid_tags)
                else:
                    ASCIIColors.error("[Specialist] No tags survived the whitelist. Discarding output.")
                    return "" # Return empty to trigger the "Specialist failed" path

                # Keep memory tags if present
                mem_tags = re.findall(r'(<mem_[^>]*>.*?</mem_[^>]*>|<mem_[^>]*/>)', raw_output, re.DOTALL)
                if mem_tags:
                    sanitized_output += "\n" + "\n".join(mem_tags)

                raw_output = sanitized_output

            # Post-Generation: If the specialist used/created memories, process them immediately
            if mm and ("<mem_" in raw_output):
                _, mem_report = mm.process_llm_output(raw_output)
                if any(mem_report.values()):
                    self._turn_action_history.append(f"✓ Memory System updated by Specialist.")

            return raw_output.strip()
        except Exception as e:
            ASCIIColors.error(f"Hyper-focused coding call failed: {e}")
            return ""

    def flush_remaining_buffer(self):
        if self.state == self.STATE_BUFFERING and self.bracket_buf:
            self._flush_bracket_buf_as_text()
            self.state = self.STATE_NORMAL

        # Final safety close for sticky processing block
        if self.proc_has_opened:
            self._emit_processing_close()


# ---------------------------------------------------------------------------
# ChatMixin
# ---------------------------------------------------------------------------

class ChatMixin:
    """
    Provides simplified_chat() and chat().

    Artefact images
    ---------------
    When active artefacts carry images (e.g. PDF pages rendered to PNG/JPEG),
    those images are automatically collected and appended to the LLM call
    alongside any user-supplied images.  The artefact text content uses
    <artefact_image id="TITLE::N" /> anchors so the model can correlate
    each image to its position in the document.

    Image ordering sent to the LLM:
        [discussion-level images] + [user message images] + [artefact images]
    """

    # ------------------------------------------------------------------ pending forms

    def _get_pending_forms(self) -> Dict[str, Dict]:
        if not hasattr(self, '_pending_forms_store'):
            object.__setattr__(self, '_pending_forms_store', {})
        return self._pending_forms_store  # type: ignore[attr-defined]

    def submit_form_response(self, form_id: str, answers: Dict[str, Any]) -> bool:
        pending = self._get_pending_forms()
        form_descriptor = pending.pop(form_id, None)
        if form_descriptor is None:
            ASCIIColors.warning(f"[Form] submit_form_response: form_id '{form_id}' not found.")
            return False

        answer_text = _format_form_answers_for_llm(form_descriptor, answers)
        self.add_message(
            sender="user",
            sender_type="user",
            content=answer_text,
            metadata={"form_id": form_id, "form_answers": answers},
        )

        cb = getattr(self, '_active_callback', None)
        _cb(cb, json.dumps({"form_id": form_id, "answers": answers}),
            MSG_TYPE.MSG_TYPE_FORM_SUBMITTED,
            {"form_id": form_id, "answers": answers, "form": form_descriptor})

        ASCIIColors.success(f"[Form] '{form_descriptor.get('title')}' answers injected.")
        return True

    # ------------------------------------------------------------------ helpers

    def _merge_artefact_images(self, user_images: Optional[List[str]]) -> List[str]:
        base = list(user_images or [])
        art_images = _collect_artefact_images(self)

        if not art_images:
            return base

        combined = base + art_images

        map_note = _build_artefact_image_map_note(self, len(base))
        if map_note:
            existing_scratch = getattr(self, 'scratchpad', '') or ''
            object.__setattr__(
                self, 'scratchpad',
                (existing_scratch + "\n\n" + map_note).strip()
            )

        return combined

    def _stream_final_answer(self, callback, images, branch_tip_id, temperature, **kwargs):
        caller_stream = kwargs.pop("stream", None)
        kwargs.pop("callback", None)
        kwargs.pop("streaming_callback", None)

        do_stream = (callback is not None) and (caller_stream is not False)
        collected = []

        merged_images = self._merge_artefact_images(images)

        ss = _StreamState(
            discussion            = self,
            callback              = callback,
            ai_message            = LollmsMessage(self, SimpleNamespace(id=None, content="", metadata={})),
            enable_notes          = kwargs.get("enable_notes", True),
            enable_skills         = kwargs.get("enable_skills", False),
            enable_inline_widgets = kwargs.get("enable_inline_widgets", True),
            enable_forms          = kwargs.get("enable_forms", True),
            auto_activate_artefacts = kwargs.get("auto_activate_artefacts", True),
            enable_artefacts      = kwargs.get("enable_artefacts", True),
            enable_in_message_status = kwargs.get("enable_in_message_status", True),
        )

        def _streaming_relay(chunk, msg_type=None, meta=None):
            if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                return ss.passthrough(chunk, msg_type, meta)
            if isinstance(chunk, str):
                # If the meta flag 'was_processed' is present, it means this chunk
                # is a result of a specialist or form and should NOT be re-analyzed
                # by the state machine (preventing double-execution).
                if meta and meta.get("was_processed"):
                    return True

                collected.append(chunk)
                return ss.feed(chunk)
            return True

        result = self.lollmsClient.chat(
            self,
            images=merged_images,
            branch_tip_id=branch_tip_id,
            stream=do_stream,
            streaming_callback=_streaming_relay if do_stream else None,
            temperature=temperature,
            **kwargs,
        )

        if do_stream:
            if isinstance(result, str) and result and not collected:
                _cb(callback, result, MSG_TYPE.MSG_TYPE_CHUNK)
                return result
            return "".join(collected) if collected else (result or "")
        return result if isinstance(result, str) else (result or "")

    # ------------------------------------------------------------------ context compression

    def _compress_context(
        self,
        callback,
        max_context_size: int,
        answer_reserve_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        import hashlib

        budget        = int(max_context_size * (1.0 - answer_reserve_ratio))
        status        = self.get_context_status()
        tokens_before = status.get("current_tokens", 0)

        if tokens_before <= budget:
            return {
                "needed": False, "artefact_pressure": False,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "budget": budget, "cache_hit": False, "summary_generated": False,
            }

        zones           = status.get("zones", {})
        history_tokens  = zones.get("message_history", {}).get("tokens", 0)
        artefact_tokens = (
            zones.get("system_context", {})
                 .get("breakdown", {})
                 .get("artefacts_zone", {})
                 .get("tokens", 0)
        )
        artefact_pressure = (
            artefact_tokens > 0
            and artefact_tokens > history_tokens
            and tokens_before > budget
        )

        _info(callback,
              f"Context at {tokens_before:,} / {budget:,} tokens "
              f"({'artefact-heavy' if artefact_pressure else 'history-heavy'}) -- compressing...")

        active_ids = sorted(a.get("id", "") for a in self.artefacts.list(active_only=True))
        key_src    = (self.active_branch_id or "") + "|" + ",".join(active_ids)
        cache_key  = hashlib.sha1(key_src.encode()).hexdigest()

        meta   = dict(self.metadata or {})
        cache  = meta.get("_compression_cache", {})
        cached = cache.get(cache_key)

        if cached:
            self.pruning_summary  = cached.get("summary", "")
            self.pruning_point_id = cached.get("pruning_point_id", "")
            self.touch()
            tokens_after = self.lollmsClient.count_tokens(
                self.export("lollms_text", self.active_branch_id, 999999)
            )
            _cb(callback,
                json.dumps({"type": "cache_hit", "tokens_before": tokens_before,
                            "tokens_after": tokens_after, "budget": budget,
                            "cache_key": cache_key,
                            "artefact_pressure": artefact_pressure}),
                MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
                {"tokens_before": tokens_before, "tokens_after": tokens_after,
                 "budget": budget, "cache_hit": True,
                 "artefact_pressure": artefact_pressure})
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_after,
                    "budget": budget, "cache_hit": True, "summary_generated": False}

        branch          = self.get_branch(self.active_branch_id)
        preserve_last_n = max(4, len(branch) // 4)

        if len(branch) <= preserve_last_n:
            _info(callback, "  History too short to prune -- reporting artefact pressure")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        to_prune      = branch[:-preserve_last_n]
        pruning_point = branch[-preserve_last_n]
        text_to_sum   = "\n\n".join(f"{m.sender}: {m.content}" for m in to_prune)

        try:
            summary = self.lollmsClient.generate_text(
                "Produce a concise but complete summary of the following conversation "
                "segment, preserving all technical decisions, code snippets, file names, "
                "variable names, and key facts. The summary will replace the original "
                "messages in the context window.\n\n"
                f"--- BEGIN SEGMENT ---\n{text_to_sum}\n--- END SEGMENT ---\n\nSUMMARY:",
                n_predict=1024, temperature=0.1,
            )
        except Exception as e:
            _warning(callback, f"  Compression failed: {e}")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        self.pruning_summary = (
            ((self.pruning_summary or "").rstrip()
             + "\n\n--- Summary ---\n" + summary.strip()).strip()
        )
        self.pruning_point_id = pruning_point.id
        self.touch()

        cache[cache_key] = {
            "summary":          self.pruning_summary,
            "pruning_point_id": self.pruning_point_id,
            "tokens_before":    tokens_before,
            "created_at":       datetime.utcnow().isoformat(),
        }
        if len(cache) > 10:
            del cache[next(iter(cache))]
        meta["_compression_cache"] = cache
        self.metadata = meta
        self.commit()

        tokens_after = self.lollmsClient.count_tokens(
            self.export("lollms_text", self.active_branch_id, 999999)
        )
        _cb(callback,
            json.dumps({"type": "summary_generated", "messages_pruned": len(to_prune),
                        "tokens_before": tokens_before, "tokens_after": tokens_after,
                        "budget": budget, "cache_key": cache_key,
                        "artefact_pressure": artefact_pressure}),
            MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
            {"tokens_before": tokens_before, "tokens_after": tokens_after,
             "budget": budget, "cache_hit": False, "summary_generated": True,
             "messages_pruned": len(to_prune), "artefact_pressure": artefact_pressure})
        _info(callback,
              f"  Compressed {len(to_prune)} messages: "
              f"{tokens_before:,} -> {tokens_after:,} tokens")

        return {"needed": True, "artefact_pressure": artefact_pressure,
                "tokens_before": tokens_before, "tokens_after": tokens_after,
                "budget": budget, "cache_hit": False, "summary_generated": True}

    # ------------------------------------------------------------------ simplified_chat

    def simplified_chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        mcps=None,
        rag_data_stores=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        use_rlm: bool = False,
        decision_temperature: float = 0.2,
        final_answer_temperature: float = 0.7,
        rag_top_k: int = 5,
        rag_min_similarity_percent: float = 0.5,
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:   bool = True,
        enable_notes:            bool = True,
        enable_skills:           bool = True,
        enable_forms:            bool = True,
        enable_silent_artefact_explanation: bool = True,
        memory_manager=None,
        enable_artefacts:        bool = True,
        enable_memory:           bool = True,
        enable_auto_dream:       bool = True,
        enable_deep_memory_pulling: bool = True,
        enable_in_message_status: bool = True,
        **kwargs
        ) -> Dict[str, Any]:
        self.scratchpad = ""
        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        # ── Memory ────────────────────────────────────────────────────────
        _mm = self._get_memory_manager(memory_manager) if enable_memory else None
        _counter = self.lollmsClient.count_tokens if self.lollmsClient else None
        if _mm:
            self._memory_pre_turn(_mm, user_message=user_message, enable_deep_memory_pulling=enable_deep_memory_pulling, token_counter=_counter)
            _mem_instructions = self._build_memory_system_instructions(_mm)
        else:
            _mem_instructions = ""

        object.__setattr__(self, '_active_callback', callback)

        self.scratchpad = ""
        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        def is_fast(msg):
            m = msg.lower().strip()
            if len(m) < 20 and any(x in m for x in ["bonjour", "salut", "hello", "hi", "hey"]):
                return True
            return m in ["ok", "merci", "thanks", "cool", "yes", "no", "oui", "non"]

        extra_instructions = ""
        if enable_artefacts:
            extra_instructions += self._build_artefact_instructions()
            if enable_inline_widgets:
                extra_instructions += self._build_inline_widget_instructions()
            if enable_notes:
                extra_instructions += self._build_note_instructions()
            if enable_skills:
                extra_instructions += self._build_skill_instructions()
            if enable_forms:
                extra_instructions += self._build_form_instructions()

            branch_msgs = self.get_branch(branch_tip_id or self.active_branch_id)
            handle_instructions = _build_handle_instructions(branch_msgs)
            if handle_instructions:
                extra_instructions += handle_instructions

        if _mem_instructions:
            extra_instructions += _mem_instructions
        if enable_image_generation or enable_image_editing:
            extra_instructions += self._build_image_generation_instructions()

        if debug or self.lollmsClient.debug:
            ASCIIColors.cyan("=== [DEBUG] SIMPLIFIED CHAT STATE ===")
            ASCIIColors.yellow(f"  • Active Artifacts: {[a['title'] for a in self.artefacts.list(active_only=True)]}")
            ASCIIColors.yellow(f"  • Scratchpad Summary: {self.scratchpad[:500]}...")
            ASCIIColors.info("=====================================")

        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        user_msg = None
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=user_message,
                images=images,
                **kwargs,
            )
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images   = user_msg.get_active_images()
            user_message = user_msg.content

        ss = _StreamState(
            discussion            = self,
            callback              = callback,
            ai_message            = LollmsMessage(self, SimpleNamespace(id=None, content="", metadata={})),
            enable_notes          = enable_notes,
            enable_skills         = enable_skills,
            enable_inline_widgets = enable_inline_widgets,
            enable_forms          = enable_forms,
            auto_activate_artefacts = auto_activate_artefacts,
            enable_artefacts      = enable_artefacts,
            enable_in_message_status = enable_in_message_status,
        )

        def _finish(text):
            ai = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content=text,
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
            )
            branch_msgs_updated = self.get_branch(ai.id)
            text_after_handles, handle_artefacts = _apply_handles(
                text, branch_msgs_updated, self.artefacts
            )
            if text_after_handles != text:
                ai.content = text_after_handles

            cleaned, affected_pp = self._post_process_llm_response(
                text_after_handles, ai,
                enable_image_generation, enable_image_editing,
                auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets if enable_artefacts else False,
                enable_notes=enable_notes if enable_artefacts else False,
                enable_skills=enable_skills if enable_artefacts else False,
                enable_forms=enable_forms if enable_artefacts else False,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation if enable_artefacts else False,
            )
            affected = handle_artefacts + ss.affected_artefacts + affected_pp
            if cleaned != text_after_handles:
                ai.content = cleaned

            # Memory tag processing
            mem_cleaned, mem_report = self._process_memory_tags(
                ai.content, _mm, callback)
            if mem_cleaned != ai.content:
                ai.content = mem_cleaned

            if _mm:
                self._save_episodic_memory_turn(user_message, ai.content, _mm)

            if affected and callback:
                _cb(callback, json.dumps([a.get("title") for a in affected]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})

            # Auto-dream pass
            dream_report = None
            if enable_auto_dream and _mm is not None:
                try:
                    dream_report = _mm.dream(self.lollmsClient)
                    if dream_report and not dream_report.get("skipped"):
                        ASCIIColors.cyan(f"[Memory] Auto-Dream complete: {dream_report}")
                        if callback:
                            try:
                                callback(
                                    json.dumps(dream_report, default=str), 
                                    MSG_TYPE.MSG_TYPE_INFO, 
                                    {"type": "memory_dream", "report": dream_report}
                                )
                            except Exception:
                                pass
                except Exception as dream_err:
                    ASCIIColors.warning(f"[Memory] Auto-dream execution failed: {dream_err}")

            object.__setattr__(self, '_active_callback', None)
            return {"user_message": user_msg, "ai_message": ai,
                    "sources": [], "artefacts": affected,
                    "memory_report": mem_report, "dream_report": dream_report}

        if _is_fast_message(user_message):
            _info(callback, "Simple response path")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, 0.1, **kwargs))

        if self.memory and user_message.lower() in self.memory.lower():
            _info(callback, "Answering from memory")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, final_answer_temperature, **kwargs))

        intent_id = _step_start(callback, "Analyzing intent...")
        intent = self.lollmsClient.generate_structured_content(
            prompt=user_message,
            schema={
                "needs_internal_knowledge": "boolean",
                "needs_full_documents":     "boolean",
                "needs_external_search":    "boolean",
                "reasoning":                "string",
            },
            temperature=decision_temperature,
        )
        _step_end(callback, "Intent analysis complete", intent_id, {"intent": intent})

        scratchpad = ""
        sources: List[str] = []

        if intent and intent.get("needs_full_documents"):
            docs_id = _step_start(callback, "Loading context documents...")
            for zone_name, zone_content in [
                ("user_data",        self.user_data_zone),
                ("discussion_data",  self.discussion_data_zone),
                ("personality_data", self.personality_data_zone),
            ]:
                if zone_content:
                    scratchpad += f"\n--- {zone_name} ---\n{zone_content}\n"
                    sources.append(zone_name)
            _step_end(callback, f"Loaded {len(sources)} zone(s)", docs_id)

        if intent and intent.get("needs_external_search") and rag_data_stores:
            rag_id = _step_start(callback, "Searching external knowledge...")
            for name, fn in rag_data_stores.items():
                if callable(fn):
                    try:
                        res = fn(user_message)
                        if res:
                            scratchpad += f"\n--- {name} ---\n{str(res)}\n"
                            sources.append(name)
                            _info(callback, f"  Retrieved results from `{name}`")
                    except Exception as e:
                        _warning(callback, f"  `{name}` search error: {e}")
            if sources:
                _cb(callback, sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            _step_end(callback, "External search complete", rag_id, {"sources": sources})

        if scratchpad:
            self.scratchpad = scratchpad.strip()

        # Inject active working/deep memory into the scratchpad so the LLM sees it
        if _mm:
            mem_block = self._build_memory_context_block(_mm, token_counter=_counter)
            if mem_block:
                self.scratchpad = (self.scratchpad or "") + "\n\n" + mem_block

        answer_id  = _step_start(callback, "Generating answer...")
        final_text = self._stream_final_answer(
            callback, images, branch_tip_id, final_answer_temperature, **kwargs)
        _step_end(callback, "Answer generation complete", answer_id)

        # ── Ensure answer was actually produced ─────────────────────────────
        if not final_text or not final_text.strip():
            _forced_id = _step_start(callback, "Forcing final answer...")
            _forced_prompt = (
                "[SYSTEM INSTRUCTION] You must now provide a direct answer to "
                "the user's question. Be concise and helpful. Do NOT use tools."
            )
            self.scratchpad = (self.scratchpad or "") + "\n" + _forced_prompt
            final_text = self._stream_final_answer(
                callback, images, branch_tip_id, final_answer_temperature, **kwargs)
            self.scratchpad = ""  # Clear after use
            _step_end(callback, "Forced answer generated", _forced_id)

        if remove_thinking_blocks:
            final_text = self.lollmsClient.remove_thinking_blocks(final_text)

        ai = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content=final_text,
            parent_id=user_msg.id if user_msg else None,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"sources": sources} if sources else {},
        )

        branch_msgs_updated = self.get_branch(ai.id)
        final_text_after_handles, handle_artefacts = _apply_handles(
            final_text, branch_msgs_updated, self.artefacts
        )
        if final_text_after_handles != final_text:
            ai.content = final_text_after_handles

        cleaned, affected = self._post_process_llm_response(
            final_text_after_handles, ai,
            enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets if enable_artefacts else False,
            enable_notes=enable_notes if enable_artefacts else False,
            enable_skills=enable_skills if enable_artefacts else False,
            enable_forms=enable_forms if enable_artefacts else False,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation if enable_artefacts else False,
        )
        affected = handle_artefacts + affected
        if cleaned != final_text_after_handles:
            ai.content = cleaned
        if affected and callback:
            _cb(callback, json.dumps([a.get("title") for a in affected]),
                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        return {"user_message": user_msg, "ai_message": ai,
                "sources": sources, "artefacts": affected}

    # ------------------------------------------------------------------ chat
    def _should_compress_agentic_history(self, branch: List) -> bool:
        """
        Return True when the branch contains ≥2 turns that performed artifact
        updates.  Those turns carry build-mode noise (status lines, XML
        fragments, retry loops) that degrades the model's language quality
        on subsequent creative or analytical turns.
        """
        artifact_turns = sum(
            1 for m in branch
            if isinstance(m.metadata, dict) and (
                m.metadata.get("artefacts_modified") or
                m.metadata.get("mode") in ("agentic", "rlm_agentic")
            )
        )
        return artifact_turns >= 2
 
    def _build_compressed_history(self, branch: List) -> List:
        """
        Collapse all past agentic / artifact turns into a single synthetic
        system message: a terse PROJECT STATE summary built from live artefact
        data plus a brief change log from message metadata.
 
        Only the last clean (non-agentic) user→assistant exchange is preserved
        verbatim to give the model an immediate conversational anchor.
 
        Returns a new virtual-history list ready for generate_from_messages().
        """
        import re as _re
 
        # 1. Live artifact inventory ─────────────────────────────────────────
        active_arts = self.artefacts.list(active_only=True)
        art_lines = []
        if active_arts:
            art_lines.append("=== ACTIVE ARTIFACTS (must update these, do not recreate) ===")
            for a in active_arts:
                lang  = f" [{a.get('language')}]" if a.get('language') else ""
                preview = (a.get('content') or '')[:60].replace('\n', ' ')
                art_lines.append(
                    f"  • [{a['type']}] \"{a['title']}\" "
                    f"({a.get('version', 1)} version(s), {len(a.get('content',''))} chars) "
                    f"— lang: {lang}"
                    f"— starts: {preview!r}"
                )
            art_lines.append("=== END ACTIVE ARTIFACTS ===")
        art_summary = "\n".join(art_lines) if art_lines else "  (none)"
 
        # 2. Change log from message metadata (last 10 changes) ──────────────
        decisions: List[str] = []
        for m in branch:
            meta     = m.metadata if isinstance(m.metadata, dict) else {}
            modified = meta.get("artefacts_modified", [])
            for title in modified:
                art = self.artefacts.get(title)
                if art:
                    decisions.append(
                        f"  • '{title}' → v{art.get('version', 1)}"
                    )
        decision_block = (
            "\n".join(decisions[-10:]) if decisions else "  (none recorded)"
        )
 
        synopsis = (
            "╔══ PROJECT STATE SYNOPSIS ══════════════════════════════════════════╗\n"
            "║  Build history suppressed — only current state shown below.        ║\n"
            "╠════════════════════════════════════════════════════════════════════╣\n"
            "║  ACTIVE ARTIFACTS                                                  ║\n"
            f"{art_summary}\n"
            "║                                                                    ║\n"
            "║  RECENT CHANGES                                                    ║\n"
            f"{decision_block}\n"
            "╠════════════════════════════════════════════════════════════════════╣\n"
            "║  Respond naturally. Do NOT reference build logs or status output.  ║\n"
            "╚══════════════════════════════════════════════════════════════════╝"
        )
 
        # 3. Find last clean (non-agentic) user→assistant exchange ───────────
        last_user_msg: Any = None
        last_ai_msg:   Any = None
        for m in reversed(branch):
            meta = m.metadata if isinstance(m.metadata, dict) else {}
            is_agentic = (
                meta.get("mode") in ("agentic", "rlm_agentic") or
                bool(meta.get("artefacts_modified"))
            )
            if m.sender_type == "assistant" and last_ai_msg is None and not is_agentic:
                # Strip residual processing markup
                clean = _re.sub(
                    r'<processing.*?>.*?</processing>', '',
                    m.content or '', flags=_re.DOTALL
                ).strip()
                last_ai_msg = SimpleNamespace(
                    sender_type="assistant", content=clean, metadata={}
                )
            elif m.sender_type == "user" and last_user_msg is None:
                last_user_msg = m
            if last_user_msg and last_ai_msg:
                break
 
        # 4. Assemble compressed virtual history ─────────────────────────────
        compressed: List = []
        if self.system_prompt:
            compressed.append(SimpleNamespace(
                sender_type="system", content=self.system_prompt
            ))
        compressed.append(SimpleNamespace(
            sender_type="system", content=synopsis
        ))
        if last_user_msg:
            compressed.append(last_user_msg)
        if last_ai_msg:
            compressed.append(last_ai_msg)
        return compressed
    
    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        swarm=None,
        swarm_config=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = True,
        enable_image_editing:    bool = True,
        auto_activate_artefacts: bool = True,
        enable_show_tools:            bool = True,
        enable_extract_artefact:      bool = True,
        enable_final_answer:          bool = True,
        enable_repl_tools:            bool = True,
        enable_inline_widgets:        bool = True,
        enable_notes:                 bool = True,
        enable_skills:                bool = False,
        enable_forms:                 bool = True,
        enable_books:                 bool = False,
        enable_presentations:         bool = False,
        enable_silent_artefact_explanation: bool = True,
        memory_manager=None,
        enable_artefacts:             bool = True,
        enable_memory:                bool = True,
        enable_auto_dream:            bool = True,
        enable_deep_memory_pulling:   bool = True,
        **kwargs
        ) -> Dict[str, Any]:
        self.scratchpad = ""

        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        # ── Memory ────────────────────────────────────────────────────────────
        _mm      = self._get_memory_manager(memory_manager) if enable_memory else None
        _counter = self.lollmsClient.count_tokens if self.lollmsClient else None
        if _mm:
            self._memory_pre_turn(_mm, user_message=user_message, enable_deep_memory_pulling=enable_deep_memory_pulling, token_counter=_counter)
            _mem_instructions = self._build_memory_system_instructions(_mm)
            # Inject active working/deep memory content so the LLM actually sees it
            mem_block = self._build_memory_context_block(_mm, token_counter=_counter)
            if mem_block:
                self.scratchpad = (self.scratchpad or "") + "\n\n" + mem_block
        else:
            _mem_instructions = ""

        if "temperature" in kwargs:
            final_answer_temperature = kwargs.pop("temperature")
        else:
            final_answer_temperature = None
 
        object.__setattr__(self, '_active_callback', None)
 
        # ====================================================================
        #  SWARM DISPATCH
        # ====================================================================
        if swarm:
            from lollms_client.lollms_swarm import SwarmOrchestrator, SwarmConfig as _SC
            _swarm_config = swarm_config if swarm_config is not None else _SC()
            if add_user_message:
                user_msg = self.add_message(
                    sender=kwargs.get("user_name", "user"),
                    sender_type="user",
                    content=user_message,
                    images=images,
                    **kwargs,
                )
            else:
                user_msg = None
            orchestrator = SwarmOrchestrator(
                discussion  = self,
                agents      = swarm,
                config      = _swarm_config,
                callback    = callback,
                user_msg_id = user_msg.id if user_msg else self.active_branch_id,
            )
            result = orchestrator.run(user_message, **kwargs)
            result["user_message"] = user_msg
            object.__setattr__(self, '_active_callback', None)
            self.scratchpad = ""
            if self._is_db_backed and self.autosave:
                self.commit()
            return result
 
        # ── Effective image flags ─────────────────────────────────────────────
        _tti_available = getattr(self.lollmsClient, 'tti', None) is not None
        _eff_img_gen   = enable_image_generation and _tti_available
        _eff_img_edit  = enable_image_editing     and _tti_available
 
        # ── System-prompt instructions ────────────────────────────────────────
        extra_instructions = ""
        if enable_artefacts:
            extra_instructions += self._build_artefact_instructions()
            if enable_inline_widgets:
                extra_instructions += self._build_inline_widget_instructions()
            if enable_notes:
                extra_instructions += self._build_note_instructions()
            if enable_skills:
                extra_instructions += self._build_skill_instructions()
            if enable_forms:
                extra_instructions += self._build_form_instructions()
            if enable_books:
                extra_instructions += self._build_book_instructions()
            if enable_presentations:
                extra_instructions += self._build_presentation_instructions()

            branch_msgs_now = self.get_branch(branch_tip_id or self.active_branch_id)
            handle_instructions = _build_handle_instructions(branch_msgs_now)
            if handle_instructions:
                extra_instructions += handle_instructions

        if _mem_instructions:
            extra_instructions += _mem_instructions
        if _eff_img_gen or _eff_img_edit:
            extra_instructions += self._build_image_generation_instructions()

        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)
 
        # ── Generation parameters ─────────────────────────────────────────────
        kwargs.pop("temperature", None)
        decision_temperature       = kwargs.get("decision_temperature",       0.3)
        final_answer_temperature   = kwargs.get("final_answer_temperature",   final_answer_temperature)
        rag_top_k                  = kwargs.get("rag_top_k",                  5)
        rag_min_similarity_percent = kwargs.get("rag_min_similarity_percent", 0.5)
        preflight_rag_enabled      = kwargs.get("preflight_rag",              True)
 
        # ── RLM detection ─────────────────────────────────────────────────────
        rlm_enabled          = False
        rlm_context_var_name = "USER_INPUT_CONTEXT"
        actual_user_content  = user_message
 
        if tools:
            rlm_enabled = (
                any(t.get("name") == "python_exec" for t in tools.values() if isinstance(t, dict))
                and
                any(t.get("name") == "llm_query"   for t in tools.values() if isinstance(t, dict))
            )
 
        if rlm_enabled and len(user_message) > 10000:
            actual_user_content = "\n".join([
                "<RLM_STUB>",
                f"Large input ({len(user_message):,} chars) stored in `{rlm_context_var_name}`.",
                "PREVIEW:", user_message[:500], "...",
                "Use python_exec() to access the full content.",
                "</RLM_STUB>",
            ])
 
        # ── Add user message ──────────────────────────────────────────────────
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=actual_user_content,
                images=images,
                **kwargs,
            )
            if rlm_enabled and len(user_message) > 10000:
                user_msg.metadata["rlm_full_content"] = user_message
                user_msg.metadata["rlm_var_name"]     = rlm_context_var_name
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images   = user_msg.get_active_images()
            user_message = user_msg.content

        # ── Fast Path Bypass ──────────────────────────────────────────────────
        if _is_fast_message(user_message):
            ai_message = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content="",
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
                metadata={"mode": "direct_fast_path"},
            )

            ss_fast = _StreamState(
                discussion            = self,
                callback              = callback,
                ai_message            = ai_message,
                enable_notes          = False,
                enable_skills         = False,
                enable_inline_widgets = False,
                enable_forms          = False,
                auto_activate_artefacts = False,
                enable_artefacts      = False,
            )

            def _fast_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss_fast.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    return ss_fast.feed(chunk)
                return True

            _cb(callback, ai_message.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_message.id})

            raw_text = self._stream_final_answer(
                _fast_relay, images,
                branch_tip_id or self.active_branch_id,
                None,  # Use the default temperature
                **kwargs,
            )
            ss_fast.flush_remaining_buffer()

            if remove_thinking_blocks:
                ai_message.content = self.lollmsClient.remove_thinking_blocks(ai_message.content)

            if self._is_db_backed and self.autosave:
                self.commit()

            self.scratchpad = ""
            object.__setattr__(self, '_active_callback', None)
            return {
                "user_message":     user_msg,
                "ai_message":       ai_message,
                "sources":          [],
                "scratchpad":       None,
                "self_corrections": None,
                "artefacts":        [],
                "memory_report":    {}
            }

        # ── Source inference helper ───────────────────────────────────────────
        def _infer_sources_from_json(data: Any, tool_name: str) -> List[Dict]:
            found_sources = []
            def looks_like_source(d: dict) -> bool:
                title_keys = {'title', 'name', 'label', 'header', 'id', 'filename'}
                return any(k in d for k in title_keys)
            def scan(obj: Any):
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and looks_like_source(item):
                            found_sources.append(item)
                        else:
                            scan(item)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in ['results', 'sources', 'data', 'items', 'content'] \
                                and isinstance(v, list):
                            scan(v)
                        else:
                            scan(v)
            scan(data)
            if not found_sources and isinstance(data, dict) and looks_like_source(data):
                found_sources.append(data)
            normalized = []
            for item in found_sources:
                title = (item.get('title') or item.get('name') or item.get('label') or
                         item.get('id') or item.get('filename') or f"{tool_name} Result")
                content = (item.get('content') or item.get('summary') or item.get('snippet') or
                           item.get('text') or item.get('description') or "")
                link = (item.get('url') or item.get('link') or item.get('href') or
                        item.get('source') or item.get('pdf_url') or "")
                raw_score = item.get('score') or item.get('relevance', 100 if content else 0)
                try:
                    score = float(raw_score) * 100 if 0 < float(raw_score) <= 1 \
                        else float(raw_score)
                except (TypeError, ValueError):
                    score = 0.0
                metadata = {k: v for k, v in item.items()
                            if k not in ['title', 'content', 'url', 'link', 'score']}
                normalized.append({
                    "title": str(title), "content": str(content), "source": str(link),
                    "relevance_score": score, "metadata": metadata, "tool": tool_name
                })
            return normalized
 
        # ── Document zone extraction helper ───────────────────────────────────
        def _extract_docs(zone_content):
            if not zone_content:
                return []
            return [
                {
                    "name":        m[0].strip(),
                    "content":     m[1].strip(),
                    "size":        len(m[1].strip()),
                    "token_count": self.lollmsClient.count_tokens(m[1].strip()),
                }
                for m in re.findall(
                    r"--- Document: (.+?) ---\n(.*?)\n--- End Document: \1 ---",
                    zone_content, re.DOTALL,
                )
            ]
 
        all_documents: List[Dict] = []
        for zone_content, zone_label in [
            (self.discussion_data_zone,  "discussion"),
            (self.user_data_zone,        "user"),
            (None if personality.has_data else self.personality_data_zone, "personality"),
        ]:
            if zone_content:
                for d in _extract_docs(zone_content):
                    d["zone"] = zone_label
                    all_documents.append(d)
 
        # ====================================================================
        #  Tool registry
        # ====================================================================
        tool_registry:     Dict[str, Any] = {}
        tool_descriptions: List[str]      = []
        rag_registry:      Dict[str, Any] = {}
        rag_tool_specs:    Dict[str, Any] = {}
 
        composable_answer  = {"sections": [], "complete": False, "last_updated": None}
        scratchpad_state   = {"notes": {}, "history": [], "assumptions": {}, "corrections": []}
        collected_sources: List[Dict] = []
        queries_performed: List[Dict] = []
        self_corrections:  List[Dict] = []
 
        def get_current_answer():
            active = [s for s in composable_answer["sections"] if s.get("status") == "active"]
            full_text = "\n\n".join(s["content"] for s in active)
            return {"success": True, "full_text": full_text, "sections": active,
                    "total_sections": len(active), "total_length": len(full_text),
                    "last_updated": composable_answer.get("last_updated")}
 
        def _make_wrapper(fn: Any, params_spec: List[Dict]) -> Any:
            def _wrapped(**kw):
                try:
                    call_args: Dict[str, Any] = {}
                    for p in params_spec:
                        pn = p.get("name")
                        pt = str(p.get("type", "string")).lower()
                        if pn in kw:
                            val = kw[pn]
                            # ── 📄 Dynamic Artifact & Image Address Resolution ──
                            # Instead of pasting massive code/images inside the JSON tool_call block,
                            # the LLM can create an ephemeral artifact and simply pass its title/address!
                            if pt == "artifact" and isinstance(val, str):
                                art = self.artefacts.get(val)
                                if art:
                                    call_args[pn] = art
                                else:
                                    return {"error": f"Artifact address '{val}' not found in active session context.", "success": False}
                            elif pt == "image" and isinstance(val, str):
                                art = self.artefacts.get(val)
                                if art and art.get("images"):
                                    # Inject the last active base64 image from the resolved artifact
                                    call_args[pn] = art["images"][-1]
                                else:
                                    # Fallback: treat as raw base64 or file path
                                    call_args[pn] = val
                            else:
                                call_args[pn] = val
                        elif not p.get("optional", False):
                            return {"error": f"Missing required parameter: {pn}", "success": False}
                        elif "default" in p:
                            call_args[pn] = p["default"]
                    result = fn(**call_args)
                    from lollms_client.lollms_types import LCPResult
                    if isinstance(result, LCPResult):
                        result = result.to_dict()
                    elif not isinstance(result, dict):
                        result = {"output": result}
                    if "success" not in result:
                        result["success"] = "error" not in result or not result.get("error")
                    return result
                except Exception as exc:
                    return {"error": str(exc), "success": False}
            return _wrapped
 
        def _sig(params: List[Dict]) -> str:
            parts = []
            for p in params:
                pn, pt = p.get("name", "arg"), p.get("type", "any")
                opt, dv = p.get("optional", False), p.get("default")
                if opt and dv is not None:
                    parts.append(f"{pn}: {pt} = {dv}")
                elif opt:
                    parts.append(f"{pn}: {pt} (optional)")
                else:
                    parts.append(f"{pn}: {pt}")
            return ", ".join(parts)
 
        def _register(name, fn, params, description, output=None):
            tool_registry[name] = _make_wrapper(fn, params)
            tool_descriptions.append(f"- {name}({_sig(params)}): {description}")
            if any(o.get("name") == "sources" for o in (output or [])):
                rag_registry[name]   = tool_registry[name]
                rag_tool_specs[name] = {
                    "default_top_k":   rag_top_k,
                    "default_min_sim": rag_min_similarity_percent,
                }
 
        # ── Layer 1: caller-supplied tools ────────────────────────────────────
        for tool_name, tool_spec in (tools or {}).items():
            if not isinstance(tool_spec, dict):
                continue
            fn = tool_spec.get("callable")
            if not callable(fn):
                continue
            _register(
                name        = tool_spec.get("name", tool_name),
                fn          = fn,
                params      = tool_spec.get("parameters", []),
                description = tool_spec.get("description", f"Execute {tool_name}"),
                output      = tool_spec.get("output", []),
            )
 
        # ── Layer 2: personality.tool_specs() ─────────────────────────────────
        _pt_specs = {}
        try:
            _pt_specs = personality.tool_specs(
                client_binding=getattr(self.lollmsClient, "tools", None)
            )
        except Exception as _pte:
            _warning(callback, f"  Personality tool discovery failed: {_pte}")
            trace_exception(_pte)
 
        if _pt_specs:
            _pt_step_id = _step_start(callback, f"Loading {len(_pt_specs)} personality tool(s)...")
            for pt_name, pt_spec in _pt_specs.items():
                fn = pt_spec.get("callable")
                if not callable(fn):
                    continue
                _register(
                    name        = pt_name,
                    fn          = fn,
                    params      = pt_spec.get("parameters", []),
                    description = pt_spec.get("description", f"Execute {pt_name}"),
                    output      = pt_spec.get("output", []),
                )
            _step_end(callback, f"{len(_pt_specs)} personality tool(s) ready", _pt_step_id,
                      {"tool_count": len(_pt_specs)})
 
        # ── Layer 2.5: Autonomous Python Data Query Tool ──────────────────────
        # Automatically registered whenever an active data/database artifact is present in the discussion.
        active_data = [a for a in self.artefacts.list(active_only=True) if a.get("type") == "data"]
        if active_data:
            def _execute_python_data_query_tool_impl(code: str) -> Dict[str, Any]:
                # Dynamically retrieve the latest version of the data artifact to support multi-round updates
                latest_data = [a for a in self.artefacts.list(active_only=True) if a.get("type") == "data"]
                if not latest_data:
                    err_msg = "No active data artifact found in this session."
                    ASCIIColors.error(f"❌ {err_msg}")
                    return {"success": False, "error": err_msg}

                title = latest_data[0]["title"]
                ext = latest_data[0].get("file_ext", ".csv")
                current_version = latest_data[0].get("version", 1)
                is_read_only = latest_data[0].get("read_only", False)
                plot_b64 = None

                workspace_dir = Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass

                # If read-only, we execute directly on the active alias without making a versioned copy
                if is_read_only:
                    new_version = current_version
                    source_file_path = workspace_dir / f"{title}{ext}"
                    new_file_path = source_file_path
                else:
                    new_version = current_version + 1
                    source_file_path = workspace_dir / f"{title}_v{current_version}{ext}"
                    new_file_path = workspace_dir / f"{title}_v{new_version}{ext}"

                ASCIIColors.info(f"--- [execute_python_data_query] Ingestion started for: '{title}' (v{current_version}{ext}) ---")

                if not source_file_path.exists():
                    # Fallback scan: check other workspaces and the global data_workspace
                    found_path = None
                    scan_dirs = [Path("./data_workspace")]
                    try:
                        from lollms_client.app.server import APP_DIR
                        if APP_DIR and APP_DIR.exists():
                            ws_dir = APP_DIR / "workspaces"
                            if ws_dir.exists():
                                for d in ws_dir.iterdir():
                                    if d.is_dir():
                                        scan_dirs.append(d / "data_workspace")
                    except Exception:
                        pass

                    for sd in scan_dirs:
                        cand = sd / f"{title}_v{current_version}{ext}"
                        if cand.exists():
                            found_path = cand
                            break

                    if found_path:
                        try:
                            workspace_dir.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy(str(found_path), str(source_file_path))
                            unversioned_dest = workspace_dir / f"{title}{ext}"
                            try:
                                shutil.copy(str(found_path), str(unversioned_dest))
                            except Exception:
                                pass
                            ASCIIColors.success(f"✓ Recovered missing data file from '{found_path.parent.parent.name or 'global'}' workspace!")
                        except Exception as copy_err:
                            ASCIIColors.error(f"Failed to copy recovered file: {copy_err}")

                if not source_file_path.exists():
                    err_msg = f"Raw data file '{title}_v{current_version}{ext}' is missing from workspace."
                    ASCIIColors.error(f"❌ {err_msg}")
                    return {"success": False, "error": err_msg}

                import pandas as pd
                import numpy as np
                import base64
                import io
                import sys
                import shutil
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                local_vars = {
                    "pd": pd,
                    "plt": plt,
                    "np": np,
                    "Path": Path
                }

                # Pre-populate local vars with dataset for immediate accessibility
                sep = ","
                try:
                    if ext in (".db", ".sqlite", ".sqlite3"):
                        shutil.copy(str(source_file_path), str(new_file_path))
                        import sqlite3
                        conn = sqlite3.connect(str(new_file_path))
                        local_vars["conn"] = conn
                        local_vars["cursor"] = conn.cursor()
                    elif ext in (".xlsx", ".xls"):
                        xl = pd.ExcelFile(str(source_file_path))
                        dfs = {sheet: pd.read_excel(str(source_file_path), sheet_name=sheet) for sheet in xl.sheet_names}
                        local_vars["dfs"] = dfs
                        if len(dfs) == 1:
                            local_vars["df"] = list(dfs.values())[0]
                    else:
                        sep = ";" if ext == ".csv" and ";" in source_file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
                        local_vars["df"] = pd.read_csv(str(source_file_path), sep=sep)
                    ASCIIColors.success(f"✓ Dataset loaded into memory successfully (Variables available: pd, plt, np, Path, and {'conn, cursor' if ext in ('.db', '.sqlite', '.sqlite3') else 'dfs' if ext in ('.xlsx', '.xls') else 'df'})")
                except Exception as e:
                    ASCIIColors.error(f"❌ Failed to load dataset: {e}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": f"Failed to load dataset: {e}"}

                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output

                import os
                old_cwd = os.getcwd()
                # List files before execution to detect changes
                files_before = set(os.listdir(str(workspace_dir))) if workspace_dir.exists() else set()

                # --- 1. Execute Code Sandbox ---
                active_file = workspace_dir / f"{title}{ext}"
                active_file_mtime_before = active_file.stat().st_mtime if active_file.exists() else 0

                try:
                    if workspace_dir and workspace_dir.exists():
                        os.chdir(str(workspace_dir))

                    plt.clf()
                    plt.close('all')
                    ASCIIColors.info(f"⚡ Executing Sandboxed Python code:\n{code}")
                    exec(code, local_vars)
                except Exception as e:
                    sys.stdout = old_stdout
                    err_msg = f"Generated code execution error: {e}"
                    ASCIIColors.error(f"❌ {err_msg}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": err_msg, "output": redirected_output.getvalue()}

                # --- 2. Update and Commit Workspace State ---
                try:
                    # Detect new or modified CSV/Excel files in the workspace to register as UI Data Views
                    files_after = set(os.listdir(str(workspace_dir))) if workspace_dir.exists() else set()
                    new_or_modified_files = []
                    for f in files_after:
                        # Exclude versioned files, internal files, and target only unversioned output datasets
                        if f.endswith((".csv", ".xlsx", ".xls")) and not "_v" in f:
                            new_or_modified_files.append(f)

                    if new_or_modified_files:
                        current_meta = dict(ai_message.metadata or {})
                        ui_views = current_meta.get("ui_data_views", [])
                        for f in new_or_modified_files:
                            if f not in ui_views:
                                ui_views.append(f)
                        current_meta["ui_data_views"] = ui_views
                        ai_message.metadata = current_meta

                    fig_nums = plt.get_fignums()
                    if fig_nums:
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        ASCIIColors.success("✓ Matplotlib visualization generated inside sandbox.")

                        # Automatically save/update as a persistent image artifact in the active workspace
                        plot_title = f"{title}_plot"
                        existing_plot = self.artefacts.get(plot_title)
                        if existing_plot is None:
                            self.artefacts.add(
                                title=plot_title,
                                artefact_type="image",
                                content=f"### Matplotlib Visualization: {plot_title}\n\n<artefact_image id=\"{plot_title}::0\" />",
                                images=[plot_b64],
                                image_media_types=["image/png"],
                                active=True
                            )
                        else:
                            # Update in-place to append a new version of the generated plot
                            self.artefacts.update(
                                title=plot_title,
                                new_content=f"### Matplotlib Visualization (Version {existing_plot.get('version', 1) + 1}): {plot_title}\n\n<artefact_image id=\"{plot_title}::{existing_plot.get('version', 1)}\" />",
                                new_images=existing_plot.get("images", []) + [plot_b64],
                                new_image_media_types=existing_plot.get("image_media_types", []) + ["image/png"],
                                bump_version=True,
                                active=True
                            )
                        self.commit()
                        ASCIIColors.success(f"✓ Registered/updated Matplotlib plot as a persistent image artifact: '{plot_title}'")

                    if not is_read_only:
                        active_file_mtime_after = active_file.stat().st_mtime if active_file.exists() else 0
                        file_written_by_script = (active_file_mtime_after > active_file_mtime_before)

                        if file_written_by_script:
                            import shutil
                            shutil.copy(str(active_file), str(new_file_path))
                        else:
                            if ext in (".db", ".sqlite", ".sqlite3") and "conn" in local_vars:
                                local_vars["conn"].commit()
                                local_vars["conn"].close()
                            elif ext in (".xlsx", ".xls") and "dfs" in local_vars:
                                with pd.ExcelWriter(new_file_path, engine="openpyxl") as writer:
                                    for sheet_name, sheet_df in local_vars["dfs"].items():
                                        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            elif "df" in local_vars:
                                local_vars["df"].to_csv(new_file_path, index=False, sep=sep)

                        from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                        new_schema, _ = _parse_data_file(new_file_path, title, version=new_version, progress_cb=None)
                        self.artefacts.update(
                            title=title,
                            new_content=new_schema,
                            new_type="data",
                            active=True,
                            file_ext=ext
                        )
                        ASCIIColors.success(f"✓ Code executed and data version incremented to v{new_version} successfully.")
                    else:
                        if ext in (".db", ".sqlite", ".sqlite3") and "conn" in local_vars:
                            local_vars["conn"].close()
                        ASCIIColors.success("✓ Code executed successfully (Read-Only mode: no files written).")
                except Exception as e:
                    sys.stdout = old_stdout
                    err_msg = f"System Error in database updater: {e}"
                    ASCIIColors.error(f"❌ {err_msg}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    if new_file_path.exists() and ext not in (".db", ".sqlite", ".sqlite3"):
                        new_file_path.unlink()
                    return {"success": False, "error": err_msg, "output": redirected_output.getvalue()}
                finally:
                    sys.stdout = old_stdout
                    try:
                        os.chdir(old_cwd)
                    except Exception:
                        pass

                out_str = redirected_output.getvalue()
                result = {
                    "success": True,
                    "output": out_str or "Code executed successfully (no stdout prints)."
                }
                if plot_b64:
                    result["plot_b64"] = plot_b64
                    result["images"] = [plot_b64]
                    result["output"] += "\n\n[SYSTEM: A matplotlib visualization was generated. The user can view it in their UI.]"
                return result

            _register(
                name="execute_python_data_query",
                fn=_execute_python_data_query_tool_impl,
                params=[{"name": "code", "type": "str", "description": "The Python code to execute. To load active datasets, open them directly by their clean name (e.g., 'sales_database.xlsx' or 'sales_database.csv'). Do NOT use '/api/workspace_files/...' prefixes."}],
                description="MANDATORY FOR DATA ANALYSIS/MODIFICATION: Execute sandboxed Python code using pandas or sqlite3 to analyze active datasets. To read active datasets (e.g., 'sales_database.xlsx'), simply open them directly by their clean filename (do NOT use '/api/workspace_files/...' prefixes, as those are strictly for client-side JavaScript).",
                output=[{"name": "output", "type": "str"}]
            )

            def _execute_sql_query_impl(sql_query: str) -> Dict[str, Any]:
                """
                Executes a standard SQLite SQL query on the active datasets.
                If the dataset is an Excel spreadsheet or CSV, they are automatically
                loaded as tables in an in-memory SQLite database.
                """
                import sqlite3
                import pandas as pd
                from pathlib import Path
                from lollms_client.lollms_types import LCPResult
                from ascii_colors import ASCIIColors, trace_exception

                title = active_data[0]["title"]
                ext = active_data[0].get("file_ext", ".csv")
                current_version = active_data[0].get("version", 1)
                is_read_only = active_data[0].get("read_only", False)

                workspace_dir = Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass

                # Check for files
                if is_read_only:
                    file_path = workspace_dir / f"{title}{ext}"
                else:
                    file_path = workspace_dir / f"{title}_v{current_version}{ext}"

                if not file_path.exists():
                    file_path = workspace_dir / f"{title}{ext}"
                    if not file_path.exists():
                        # Run self-healing scanner as fallback
                        found_path = None
                        scan_dirs = [Path("./data_workspace")]
                        try:
                            from lollms_client.app.server import APP_DIR
                            if APP_DIR and APP_DIR.exists():
                                ws_dir = APP_DIR / "workspaces"
                                if ws_dir.exists():
                                    for d in ws_dir.iterdir():
                                        if d.is_dir():
                                            scan_dirs.append(d / "data_workspace")
                        except Exception:
                            pass

                        for sd in scan_dirs:
                            cand = sd / f"{title}_v{current_version}{ext}"
                            if cand.exists():
                                found_path = cand
                                break
                            cand = sd / f"{title}{ext}"
                            if cand.exists():
                                found_path = cand
                                break

                        if found_path:
                            try:
                                workspace_dir.mkdir(parents=True, exist_ok=True)
                                import shutil
                                shutil.copy(str(found_path), str(file_path))
                                unversioned_dest = workspace_dir / f"{title}{ext}"
                                try:
                                    shutil.copy(str(found_path), str(unversioned_dest))
                                except Exception:
                                    pass
                                ASCIIColors.success(f"✓ Recovered missing data file from '{found_path.parent.parent.name or 'global'}' workspace!")
                            except Exception as copy_err:
                                ASCIIColors.error(f"Failed to copy recovered file: {copy_err}")

                if not file_path.exists():
                    return {"success": False, "error": f"Raw data file '{title}' is missing from workspace."}

                ASCIIColors.info(f"--- [execute_sql_query] Compiling SQL query inside sandbox... ---")

                # Setup in-memory SQLite database
                try:
                    conn = sqlite3.connect(":memory:")

                    if ext in (".db", ".sqlite", ".sqlite3"):
                        # If SQLite database on disk, backup/copy tables to our in-memory DB
                        disk_conn = sqlite3.connect(str(file_path))
                        disk_conn.backup(conn)
                        disk_conn.close()
                    elif ext in (".xlsx", ".xls"):
                        # If Excel, load sheets as tables
                        xl = pd.ExcelFile(str(file_path))
                        for sheet_name in xl.sheet_names:
                            # Standardize table name (no spaces)
                            table_name = sheet_name.replace(" ", "_")
                            df = pd.read_excel(str(file_path), sheet_name=sheet_name)
                            df.to_sql(table_name, conn, index=False, if_exists="replace")
                    else:
                        # CSV
                        sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
                        df = pd.read_csv(str(file_path), sep=sep)
                        df.to_sql(title.replace(" ", "_"), conn, index=False, if_exists="replace")
                    ASCIIColors.success("✓ Relational dataset compiled into in-memory SQLite successfully.")
                except Exception as e:
                    ASCIIColors.error(f"❌ Failed to load dataset tables into memory SQLite: {e}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": f"Failed to load dataset: {e}"}

                # Execute SQL query on our in-memory DB
                try:
                    # Strip SQL comments and whitespace to reliably detect SELECT queries
                    clean_query = sql_query.strip()
                    clean_query = re.sub(r'--.*$', '', clean_query, flags=re.MULTILINE).strip()
                    clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL).strip()

                    is_select = clean_query.lower().startswith("select")

                    if is_select:
                        df_res = pd.read_sql_query(sql_query, conn)
                        output_md = df_res.to_markdown(index=False)
                        sources = [{
                            "title": f"SQL Query Result: {sql_query}",
                            "content": output_md,
                            "source": f"sql_query"
                        }]
                        ASCIIColors.success(f"✓ SQL select query executed successfully: found {len(df_res)} rows.")
                    else:
                        if is_read_only:
                            conn.close()
                            err_msg = "Database is read-only. Writable SQL queries (INSERT/UPDATE/DELETE) are blocked."
                            ASCIIColors.error(f"❌ {err_msg}")
                            return {"success": False, "error": err_msg}

                        # Run write query
                        cursor = conn.cursor()
                        cursor.execute(sql_query)
                        conn.commit()

                        # Write back modified tables from memory DB back to Excel/CSV file on disk
                        if ext in (".db", ".sqlite", ".sqlite3"):
                            disk_conn = sqlite3.connect(str(file_path))
                            conn.backup(disk_conn)
                            disk_conn.close()
                        elif ext in (".xlsx", ".xls"):
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = [row[0] for row in cursor.fetchall()]

                            new_file_path = workspace_dir / f"{title}_v{current_version + 1}{ext}"
                            with pd.ExcelWriter(new_file_path, engine="openpyxl") as writer:
                                for t in tables:
                                    df_write = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                                    df_write.to_excel(writer, sheet_name=t.replace("_", " "), index=False)

                            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                            new_schema, _ = _parse_data_file(new_file_path, title, version=current_version + 1, progress_cb=None)
                            self.artefacts.update(
                                title=title,
                                new_content=new_schema,
                                new_type="data",
                                active=True,
                                file_ext=ext
                            )
                        else:
                            new_file_path = workspace_dir / f"{title}_v{current_version + 1}{ext}"
                            df_write = pd.read_sql_query(f"SELECT * FROM {title.replace(' ', '_')}", conn)
                            df_write.to_csv(new_file_path, index=False, sep=sep)

                            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                            new_schema, _ = _parse_data_file(new_file_path, title, version=current_version + 1, progress_cb=None)
                            self.artefacts.update(
                                title=title,
                                new_content=new_schema,
                                new_type="data",
                                active=True,
                                file_ext=ext
                            )

                        output_md = f"Query executed successfully: `{sql_query}`. Affected rows: {cursor.rowcount}"
                        sources = []
                        ASCIIColors.success(f"✓ SQL write query executed successfully, workspace state updated.")

                    conn.close()

                    return LCPResult(
                        success=True,
                        output=output_md,
                        sources=sources
                    )
                except Exception as e:
                    conn.close()
                    ASCIIColors.error(f"❌ SQL execution failed: {e}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": f"SQL execution error: {e}"}

            _register(
                name="execute_sql_query",
                fn=_execute_sql_query_impl,
                params=[{"name": "sql_query", "type": "str", "description": "The standard SQL query (SQLite syntax) to run on the database. Available tables match the Sheet names (e.g. 'Customers', 'Products', 'Orders', 'Order_Details' with spaces replaced by underscores, e.g., 'Order_Details')."}],
                description="MANDATORY FOR SQL DATA ANALYSIS/QUERIES: Execute standard SQL queries on the active dataset tables. Available tables are the sheet names with spaces replaced by underscores. For write queries (INSERT/UPDATE/DELETE), ensure the dataset is in WRITABLE mode.",
                output=[{"name": "output", "type": "str"}]
            )

            def _execute_sql_query_impl(sql_query: str) -> Dict[str, Any]:
                """
                Executes a standard SQLite SQL query on the active datasets.
                If the dataset is an Excel spreadsheet or CSV, they are automatically
                loaded as tables in an in-memory SQLite database.
                """
                import sqlite3
                import pandas as pd
                from pathlib import Path
                from lollms_client.lollms_types import LCPResult
                from ascii_colors import ASCIIColors, trace_exception

                title = active_data[0]["title"]
                ext = active_data[0].get("file_ext", ".csv")
                current_version = active_data[0].get("version", 1)
                is_read_only = active_data[0].get("read_only", False)

                workspace_dir = Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass

                # Check for files
                if is_read_only:
                    file_path = workspace_dir / f"{title}{ext}"
                else:
                    file_path = workspace_dir / f"{title}_v{current_version}{ext}"

                if not file_path.exists():
                    file_path = workspace_dir / f"{title}{ext}"
                    if not file_path.exists():
                        # Run self-healing scanner as fallback
                        found_path = None
                        scan_dirs = [Path("./data_workspace")]
                        try:
                            from lollms_client.app.server import APP_DIR
                            if APP_DIR and APP_DIR.exists():
                                ws_dir = APP_DIR / "workspaces"
                                if ws_dir.exists():
                                    for d in ws_dir.iterdir():
                                        if d.is_dir():
                                            scan_dirs.append(d / "data_workspace")
                        except Exception:
                            pass

                        for sd in scan_dirs:
                            cand = sd / f"{title}_v{current_version}{ext}"
                            if cand.exists():
                                found_path = cand
                                break
                            cand = sd / f"{title}{ext}"
                            if cand.exists():
                                found_path = cand
                                break

                        if found_path:
                            try:
                                workspace_dir.mkdir(parents=True, exist_ok=True)
                                import shutil
                                shutil.copy(str(found_path), str(file_path))
                                unversioned_dest = workspace_dir / f"{title}{ext}"
                                try:
                                    shutil.copy(str(found_path), str(unversioned_dest))
                                except Exception:
                                    pass
                                ASCIIColors.success(f"✓ Recovered missing data file from '{found_path.parent.parent.name or 'global'}' workspace!")
                            except Exception as copy_err:
                                ASCIIColors.error(f"Failed to copy recovered file: {copy_err}")

                if not file_path.exists():
                    return {"success": False, "error": f"Raw data file '{title}' is missing from workspace."}

                ASCIIColors.info(f"--- [execute_sql_query] Compiling SQL query inside sandbox... ---")

                # Setup in-memory SQLite database
                try:
                    conn = sqlite3.connect(":memory:")

                    if ext in (".db", ".sqlite", ".sqlite3"):
                        # If SQLite database on disk, backup/copy tables to our in-memory DB
                        disk_conn = sqlite3.connect(str(file_path))
                        disk_conn.backup(conn)
                        disk_conn.close()
                    elif ext in (".xlsx", ".xls"):
                        # If Excel, load sheets as tables
                        xl = pd.ExcelFile(str(file_path))
                        for sheet_name in xl.sheet_names:
                            # Standardize table name (no spaces)
                            table_name = sheet_name.replace(" ", "_")
                            df = pd.read_excel(str(file_path), sheet_name=sheet_name)
                            df.to_sql(table_name, conn, index=False, if_exists="replace")
                    else:
                        # CSV
                        sep = ";" if ext == ".csv" and ";" in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
                        df = pd.read_csv(str(file_path), sep=sep)
                        df.to_sql(title.replace(" ", "_"), conn, index=False, if_exists="replace")
                    ASCIIColors.success("✓ Relational dataset compiled into in-memory SQLite successfully.")
                except Exception as e:
                    ASCIIColors.error(f"❌ Failed to load dataset tables into memory SQLite: {e}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": f"Failed to load dataset: {e}"}

                # Execute SQL query on our in-memory DB
                try:
                    is_select = sql_query.strip().lower().startswith("select")

                    if is_select:
                        df_res = pd.read_sql_query(sql_query, conn)
                        output_md = df_res.to_markdown(index=False)
                        sources = [{
                            "title": f"SQL Query Result: {sql_query}",
                            "content": output_md,
                            "source": f"sql_query"
                        }]
                        ASCIIColors.success(f"✓ SQL select query executed successfully: found {len(df_res)} rows.")
                    else:
                        if is_read_only:
                            conn.close()
                            err_msg = "Database is read-only. Writable SQL queries (INSERT/UPDATE/DELETE) are blocked."
                            ASCIIColors.error(f"❌ {err_msg}")
                            return {"success": False, "error": err_msg}

                        # Run write query
                        cursor = conn.cursor()
                        cursor.execute(sql_query)
                        conn.commit()

                        # Write back modified tables from memory DB back to Excel/CSV file on disk
                        if ext in (".db", ".sqlite", ".sqlite3"):
                            disk_conn = sqlite3.connect(str(file_path))
                            conn.backup(disk_conn)
                            disk_conn.close()
                        elif ext in (".xlsx", ".xls"):
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = [row[0] for row in cursor.fetchall()]

                            new_file_path = workspace_dir / f"{title}_v{current_version + 1}{ext}"
                            with pd.ExcelWriter(new_file_path, engine="openpyxl") as writer:
                                for t in tables:
                                    df_write = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                                    df_write.to_excel(writer, sheet_name=t.replace("_", " "), index=False)

                            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                            new_schema, _ = _parse_data_file(new_file_path, title, version=current_version + 1, progress_cb=None)
                            self.artefacts.update(
                                title=title,
                                new_content=new_schema,
                                new_type="data",
                                active=True,
                                file_ext=ext
                            )
                        else:
                            new_file_path = workspace_dir / f"{title}_v{current_version + 1}{ext}"
                            df_write = pd.read_sql_query(f"SELECT * FROM {title.replace(' ', '_')}", conn)
                            df_write.to_csv(new_file_path, index=False, sep=sep)

                            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                            new_schema, _ = _parse_data_file(new_file_path, title, version=current_version + 1, progress_cb=None)
                            self.artefacts.update(
                                title=title,
                                new_content=new_schema,
                                new_type="data",
                                active=True,
                                file_ext=ext
                            )

                        output_md = f"Query executed successfully: `{sql_query}`. Affected rows: {cursor.rowcount}"
                        sources = []
                        ASCIIColors.success(f"✓ SQL write query executed successfully, workspace state updated.")

                    conn.close()

                    return LCPResult(
                        success=True,
                        output=output_md,
                        sources=sources
                    )
                except Exception as e:
                    conn.close()
                    ASCIIColors.error(f"❌ SQL execution failed: {e}")
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    return {"success": False, "error": f"SQL execution error: {e}"}

            _register(
                name="execute_sql_query",
                fn=_execute_sql_query_impl,
                params=[{"name": "sql_query", "type": "str", "description": "The standard SQL query (SQLite syntax) to run on the database. Available tables match the Sheet names (e.g. 'Customers', 'Products', 'Orders', 'Order_Details' with spaces replaced by underscores, e.g., 'Order_Details')."}],
                description="MANDATORY FOR SQL DATA ANALYSIS/QUERIES: Execute standard SQL queries on the active dataset tables. Available tables are the sheet names with spaces replaced by underscores. For write queries (INSERT/UPDATE/DELETE), ensure the dataset is in WRITABLE mode.",
                output=[{"name": "output", "type": "str"}]
            )

        # ── Layer 3: personality RAG tool ─────────────────────────────────────
        if personality.has_data:
            def _personality_rag(query: str) -> Dict[str, Any]:
                result = personality.query_data(query)
                sources_filtered = sorted(
                    (s for s in result.get("sources", [])),
                    key=lambda x: x.get("score", 0), reverse=True
                )
                for s in sources_filtered:
                    score = float(s.get("score", 1.0))
                    if score > 1.0:
                        score = 1.0 if score > 100.0 else score / 100.0
                    s["score"] = score
                sources_filtered = [
                    s for s in sources_filtered[:rag_top_k]
                    if s.get("score", 0) >= rag_min_similarity_percent
                ]
                result["sources"] = sources_filtered
                result["count"]   = len(sources_filtered)
                return result
 
            _register(
                name        = "search_personality_knowledge",
                fn          = _personality_rag,
                params      = [{"name": "query", "type": "str", "description": "Search query"}],
                description = "Search the personality's knowledge base",
                output      = [{"name": "sources", "type": "list"}],
            )
 
        # ── Personality system prompt ──────────────────────────────────────────
        if personality.system_prompt:
            veracity = (
                "\n=== VERACITY & ATTRIBUTION REQUIREMENTS ===\n"
                "Cite retrieved sources as [1],[2]... "
                "Use 'From my understanding...' for general knowledge.\n"
                "Never fabricate facts. Say 'I don't know' when uncertain.\n"
                "=== END ===\n"
            )
            object.__setattr__(
                self, "_system_prompt",
                personality.system_prompt + veracity + extra_instructions,
            )
 
        # ── Pre-flight RAG ────────────────────────────────────────────────────
        if preflight_rag_enabled and personality.has_data:
            preflight_id = _step_start(callback, "Pre-flight knowledge retrieval...")
            ctx = self.export("markdown", suppress_system_prompt=True)
            try:
                query_json = self.lollmsClient.generate_structured_content(
                    prompt=ctx[-2000:] + "\nGenerate a concise search query (JSON).",
                    schema={"query": "Your concise search query string"},
                    system_prompt="Output only JSON.",
                    temperature=0.1,
                )
                if query_json and "query" in query_json:
                    rag_result = personality.query_data(query_json["query"])
                    if rag_result.get("success"):
                        fmt = ""
                        for idx, chunk in enumerate(rag_result.get("sources", [])):
                            src  = chunk.get("source", "")
                            meta = chunk.get("metadata", {})
                            title = (
                                chunk.get("title") or meta.get("title") or
                                meta.get("filename") or meta.get("name") or
                                (src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if src else "") or
                                f"Source {idx + 1}"
                            )
                            fmt += (
                                f"[Source {idx+1}] ({src}, "
                                f"{chunk.get('score', 0):.2f})\n"
                                f"{chunk.get('content', '')}\n\n"
                            )
                            collected_sources.append({
                                "title":           title,
                                "content":         chunk.get("content", ""),
                                "source":          src,
                                "query":           query_json["query"],
                                "relevance_score": chunk.get("score", 0),
                                "index":           idx + 1,
                                "phase":           "preflight",
                                "metadata":        meta,
                            })
                        if fmt:
                            self.scratchpad = (
                                fmt.strip() + "\n\nIMPORTANT: Cite sources as [1],[2],..."
                            )
                        if collected_sources:
                            _cb(callback, collected_sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            except Exception as e:
                trace_exception(e)
            _step_end(callback, "Pre-flight retrieval complete", preflight_id,
                      {"source_count": len(collected_sources)})
 
        # Quick intent check to see if the user's message actually requires tools or agentic actions
        _needs_tools = False

        # CRITICAL FIX: Check for active data artifacts first - ANY analytical question should trigger tool use
        has_active_data_artifacts = any(
            a.get("type") == "data" 
            for a in self.artefacts.list(active_only=True)
        )

        if has_active_data_artifacts and tool_registry:
            # Check if the question is analytical/factual (could relate to data)
            # Keywords that suggest data analysis/querying even without explicit "database" mention
            _data_intent_keywords = {
                'question_words': ['what', 'which', 'how many', 'how much', 'where', 'when', 'why', 'who'],
                'analytical_verbs': ['find', 'get', 'query', 'check', 'see', 'look', 'analyze', 'calculate', 'compute', 'count', 'list', 'show', 'retrieve', 'extract'],
                'comparative_words': ['most', 'least', 'more', 'less', 'better', 'worse', 'highest', 'lowest', 'top', 'bottom', 'average', 'total', 'sum', 'frequency', 'often', 'rarely'],
                'data_terms': ['metal', 'equipment', 'contamination', 'value', 'measure', 'record', 'entry', 'row', 'column', 'table', 'dataset']
            }

            user_msg_lower = user_message.lower()
            data_intent_score = 0

            # Count matching keywords
            for word_list in _data_intent_keywords.values():
                for keyword in word_list:
                    if keyword in user_msg_lower:
                        data_intent_score += 1

            # If question has multiple analytical indicators AND there's a data artifact, force tool use
        if has_active_data_artifacts and tool_registry:
            try:
                intent_prompt = (
                    f"User message: \"{user_message}\"\n\n"
                    "IMPORTANT CONTEXT: There is an active DATA artifact in this session (a database or dataset file).\n\n"
                    "Analyze if this message requires:\n"
                    "1. Executing a tool to query/analyze the data artifact\n"
                    "2. Modifying/creating files (artifacts)\n"
                    "3. Generating/editing images\n\n"
                    "CRITICAL: If the question is analytical, factual, or asks about specific values/frequencies/count/comparisons,\n"
                    "it likely requires querying the data artifact even if it doesn't explicitly mention 'database' or 'query'.\n\n"
                    "Return false ONLY if:\n"
                    "- The user is greeting you (hello, hi, etc.)\n"
                    "- The question has NO relation to data analysis (e.g., philosophical, abstract)\n"
                    "- It's purely conversational chit-chat\n\n"
                    f"When in doubt with analytical questions and an active data artifact, return TRUE.\n"
                )
                intent_res = self.lollmsClient.generate_structured_content(
                    prompt=intent_prompt,
                    schema={
                        "requires_tools_or_actions": {
                            "type": "boolean",
                            "description": "True if the message requires tool execution (especially data queries), file modification/creation, or image generation/edit. False only for pure conversation."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning explaining why tools are or aren't needed"
                        }
                    },
                    temperature=0.0
                )
                if intent_res and isinstance(intent_res, dict):
                    _needs_tools = intent_res.get("requires_tools_or_actions", False)

                    # Additional safety: If data artifact exists AND question has analytical keywords, override to True
                    if has_active_data_artifacts and data_intent_score >= 2:
                        _needs_tools = True
                        ASCIIColors.info(f"[Intent Classifier] Overrode to TRUE due to active data artifact + {data_intent_score} analytical keywords")

                    ASCIIColors.info(f"[Intent Classifier] Requires tools/actions: {_needs_tools} | Reasoning: {intent_res.get('reasoning')}")
            except Exception as e:
                # If intent classification fails AND there's a data artifact, default to True (safer)
                if has_active_data_artifacts:
                    _needs_tools = True
                    ASCIIColors.warning(f"[Intent Classifier] Failed: {e}. Defaulting to TRUE due to active data artifact.")
                else:
                    _needs_tools = False
        elif tool_registry or enable_image_generation or enable_image_editing or enable_artefacts:
            # Standard path for non-data-artifact scenarios
            try:
                intent_prompt = (
                    f"User message: \"{user_message}\"\n\n"
                    "Analyze if this message requires executing an external tool, modifying/creating files (artifacts), or generating/editing images.\n"
                    "Return false if the user is just having a normal conversation, greeting you, asking a general question, or talking casually."
                )
                intent_res = self.lollmsClient.generate_structured_content(
                    prompt=intent_prompt,
                    schema={
                        "requires_tools_or_actions": {
                            "type": "boolean",
                            "description": "True if the message requests a tool execution, file modification/creation, or image generation/edit. False for normal conversation."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the decision."
                        }
                    },
                    temperature=0.0
                )
                if intent_res and isinstance(intent_res, dict):
                    _needs_tools = intent_res.get("requires_tools_or_actions", False)
                    ASCIIColors.info(f"[Intent Classifier] Requires tools/actions: {_needs_tools} | Reasoning: {intent_res.get('reasoning')}")
            except Exception as e:
                _needs_tools = True
                ASCIIColors.warning(f"[Intent Classifier] Failed: {e}. Defaulting to True.")
        else:
            _needs_tools = False

        # ====================================================================
        #  FAST PATH — no external tools registered
        # ====================================================================
        _has_external_tools = bool(tool_registry) and _needs_tools
 
        if not _has_external_tools:
            ss = _StreamState(
                discussion            = self,
                callback              = callback,
                ai_message            = None,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = enable_inline_widgets,
                enable_forms          = enable_forms,
                auto_activate_artefacts = auto_activate_artefacts,
                enable_artefacts      = enable_artefacts,
            )
 
            ai_message = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content="",
                parent_id=user_msg.id,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
                metadata={"mode": "direct"},
            )
            ss.ai_message = ai_message
 
            def _fast_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True
 
            raw_text = self._stream_final_answer(
                _fast_relay, images,
                branch_tip_id or self.active_branch_id,
                final_answer_temperature, **kwargs,
            )
 
            ss.flush_remaining_buffer()
 
            if raw_text and not ai_message.content:
                for ch in raw_text:
                    ss.feed(ch)
                ss.flush_remaining_buffer()
 
            raw_text = ai_message.content
 
            # Scrub any leaked internal markers
            if _EXEC_MARKER_RE.search(raw_text or ""):
                raw_text = _EXEC_MARKER_RE.sub('', raw_text).strip()
                ai_message.content = raw_text
 
            if not raw_text or not raw_text.strip():
                ASCIIColors.warning("[chat] Fast path produced no output — forcing retry")
                _retry_prompt = (
                    "[SYSTEM INSTRUCTION] Please provide a direct answer to the user's question. "
                    "Be concise and helpful."
                )
                self.scratchpad = (self.scratchpad or "") + "\n" + _retry_prompt
                raw_text = self._stream_final_answer(
                    _fast_relay, images,
                    branch_tip_id or self.active_branch_id,
                    final_answer_temperature, **kwargs,
                )
                ss.flush_remaining_buffer()
                raw_text = ai_message.content
                self.scratchpad = ""

            # ADD: detect "described but didn't act" on fast path
            _intent_keywords = ["update", "add", "change", "fix", "create", "build", "write", "make"]
            _user_intends_to_act = any(kw in user_message.lower() for kw in _intent_keywords)
            if (_user_intends_to_act
                    and not ss.affected_artefacts
                    and len(ai_message.content.strip()) > 30
                    and bool(self.artefacts.list(active_only=True))):
                # Model wrote prose about what it would do but didn't emit an <artifact> tag
                # while active artifacts exist. Re-run with correction injected.
                self.scratchpad = (
                    "\n⚠️ CORRECTION: You wrote a description but did not emit any <artifact> tag. "
                    "Active artifacts exist. You MUST emit the <artifact> tag with your changes NOW. "
                    "Do not write prose — emit the XML directly.\n"
                    + (self.scratchpad or "")
                )
                # clear and re-run (similar to existing retry_prompt pattern)
                ai_message.content = ""
                raw_text = self._stream_final_answer(
                    _fast_relay, images,
                    branch_tip_id or self.active_branch_id,
                    final_answer_temperature, **kwargs,
                )
                ss.flush_remaining_buffer()                
 
            if remove_thinking_blocks:
                raw_text = self.lollmsClient.remove_thinking_blocks(raw_text)
 
            branch_for_handles = self.get_branch(ai_message.id)
            raw_after_handles, handle_arts = _apply_handles(
                raw_text, branch_for_handles, self.artefacts
            )
            ai_message.content = raw_after_handles
 
            cleaned, affected_pp = self._post_process_llm_response(
                raw_after_handles, ai_message, _eff_img_gen, _eff_img_edit,
                auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets if enable_artefacts else False,
                enable_notes=enable_notes if enable_artefacts else False,
                enable_skills=enable_skills if enable_artefacts else False,
                enable_forms=enable_forms if enable_artefacts else False,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation if enable_artefacts else False,
            )
            affected = handle_arts + ss.affected_artefacts + affected_pp
            if cleaned != raw_after_handles:
                ai_message.content = cleaned
 
            _mem_cleaned, _mem_report = self._process_memory_tags(
                ai_message.content, _mm, callback)
            if _mem_cleaned != ai_message.content:
                ai_message.content = _mem_cleaned

            if _mm:
                self._save_episodic_memory_turn(user_message, ai_message.content, _mm)

            # Auto-dream pass
            dream_report = None
            if enable_auto_dream and _mm is not None:
                try:
                    dream_report = _mm.dream(self.lollmsClient)
                    if dream_report and not dream_report.get("skipped"):
                        ASCIIColors.cyan(f"[Memory] Auto-Dream complete: {dream_report}")
                        if callback:
                            try:
                                callback(
                                    json.dumps(dream_report, default=str), 
                                    MSG_TYPE.MSG_TYPE_INFO, 
                                    {"type": "memory_dream", "report": dream_report}
                                )
                            except Exception:
                                pass
                except Exception as dream_err:
                    ASCIIColors.warning(f"[Memory] Auto-dream execution failed: {dream_err}")

            if self._is_db_backed and self.autosave:
                self.commit()
            self.scratchpad = ""
            object.__setattr__(self, '_active_callback', None)
            return {
                "user_message":     user_msg,
                "ai_message":       ai_message,
                "sources":          collected_sources,
                "scratchpad":       None,
                "self_corrections": None,
                "artefacts":        affected,
                "memory_report":    _mem_report,
                "dream_report":     dream_report,
            }
 
        # ====================================================================
        #  Built-in tools (only when external tools exist)
        # ====================================================================
 
        if enable_show_tools:
            def _show_tools_impl():
                catalogue: List[Dict[str, Any]] = []
                for _tname, _tspec in (tools or {}).items():
                    if not isinstance(_tspec, dict):
                        continue
                    catalogue.append({
                        "name":        _tspec.get("name", _tname),
                        "description": _tspec.get("description", ""),
                        "parameters":  _tspec.get("parameters", []),
                        "output":      _tspec.get("output", []),
                        "source":      "user",
                    })
                for pt_name, pt_spec in _pt_specs.items():
                    catalogue.append({
                        "name":        pt_name,
                        "description": pt_spec.get("description", ""),
                        "parameters":  pt_spec.get("parameters", []),
                        "output":      pt_spec.get("output", []),
                        "source":      "personality",
                        "binding":     pt_spec.get("_binding", ""),
                    })
                _cb(callback, json.dumps(catalogue, indent=2),
                    MSG_TYPE.MSG_TYPE_TOOLS_LIST, {"tools": catalogue})
                return {"success": True, "tool_count": len(catalogue), "tools": catalogue}

            _register(
                name="show_tools",
                fn=_show_tools_impl,
                params=[],
                description="Display the full list of available tools"
            )

        if enable_extract_artefact:
            def _extract_artefact_text_impl(
                source_title: str, new_title: str,
                start_line_hint: str, end_line_hint: str,
                occurrence: int = 1,
                artefact_type: str = "document",
                language: str = "",
            ) -> Dict[str, Any]:
                source = self.artefacts.get(source_title)
                if source is None:
                    return {"success": False, "error": f"Artifact '{source_title}' not found."}
                all_lines = source.get("content", "").splitlines()
                total = len(all_lines)
                if total == 0:
                    return {"success": False, "error": f"Artifact '{source_title}' is empty."}
                sh = start_line_hint.strip().lower()
                eh = end_line_hint.strip().lower()
                if not sh:
                    return {"success": False, "error": "start_line_hint must not be empty."}
                if not eh:
                    return {"success": False, "error": "end_line_hint must not be empty."}
                start_idx, hit = None, 0
                for i, ln in enumerate(all_lines):
                    if ln.strip().lower().startswith(sh):
                        hit += 1
                        if hit == occurrence:
                            start_idx = i
                            break
                if start_idx is None:
                    return {"success": False,
                            "error": (f"start_line_hint {start_line_hint!r} not found "
                                      f"(occurrence {occurrence} of {hit} found)."),
                            "total_lines": total}
                end_idx = None
                for i in range(start_idx, total):
                    if all_lines[i].strip().lower().startswith(eh):
                        end_idx = i
                        break
                if end_idx is None:
                    return {"success": False,
                            "error": (f"end_line_hint {end_line_hint!r} not found "
                                      f"after line {start_idx + 1}."),
                            "start_line_no": start_idx + 1, "total_lines": total}
                from ._artefacts import ArtefactType as _AT
                resolved_type = (artefact_type if artefact_type in _AT.ALL
                                 else source.get("type", _AT.DOCUMENT))
                extracted = "\n".join(all_lines[start_idx:end_idx + 1])
                new_art   = self.artefacts.add(
                    title=new_title, artefact_type=resolved_type,
                    content=extracted,
                    language=language or source.get("language") or None,
                    active=True,
                )
                return {
                    "success": True, "source_title": source_title, "new_title": new_title,
                    "start_line_no": start_idx + 1, "end_line_no": end_idx + 1,
                    "total_lines": total, "lines_extracted": end_idx - start_idx + 1,
                    "artefact_id": new_art.get("id"),
                }

            _register(
                name="extract_artifact_text",
                fn=_extract_artefact_text_impl,
                params=[
                    {"name": "source_title", "type": "str", "description": "Title of the source artifact"},
                    {"name": "new_title", "type": "str", "description": "Title of the new artifact to create"},
                    {"name": "start_line_hint", "type": "str", "description": "Line content / prefix of the starting block (case insensitive)"},
                    {"name": "end_line_hint", "type": "str", "description": "Line content / prefix of the ending block (case insensitive)"},
                    {"name": "occurrence", "type": "int", "description": "Occurrence count if multiple matching lines exist", "optional": True, "default": 1},
                    {"name": "artefact_type", "type": "str", "description": "The target artifact type to assign (e.g., 'code', 'document')", "optional": True, "default": "document"},
                    {"name": "language", "type": "str", "description": "Optional programming language if type is code", "optional": True, "default": ""},
                ],
                description="Extract a range from an artifact by line-prefix anchors"
            )

        # ── 5. Promote Artifact Tool ──
        def _promote_artifact_impl(title: str) -> Dict[str, Any]:
            art = self.artefacts.get(title)
            if not art:
                return {"success": False, "error": f"Artifact '{title}' not found."}
            # Remove ephemeral/hidden flag to make it persistent and visible in the sidebar list
            self.artefacts.update(
                title=title,
                active=True,
                ephemeral=False,
                bump_version=False # update in place
            )
            self.commit()
            return {
                "success": True,
                "message": f"Successfully promoted artifact '{title}' to persistent workspace sidebar list."
            }
        
        _register(
            name="promote_artifact",
            fn=_promote_artifact_impl,
            params=[{"name": "title", "type": "str", "description": "The exact title of the ephemeral artifact to promote to the visible workspace sidebar list."}],
            description="Promote an ephemeral/temporary artifact to a permanent, visible workspace file."
        )

        if enable_repl_tools:
            try:
                from ._repl_tools import TextBuffer, register_repl_tools as _reg_repl
                _reg_repl(tool_registry, tool_descriptions, TextBuffer(), self.artefacts)
            except ImportError as _e:
                _warning(callback, f"REPL text tools unavailable: {_e}")

        if enable_final_answer:
            _register(
                name="final_answer",
                fn=lambda: {
                    "status":  "final",
                    "answer":  get_current_answer()["full_text"]
                               if composable_answer["sections"] else None,
                    "success": True,
                },
                params=[],
                description="Signal that the answer is ready"
            )
  
        object.__setattr__(
            self, "_system_prompt",
            _build_tool_system_prompt(self._system_prompt or "", tool_descriptions)
        )
 
        # ── Context compression ───────────────────────────────────────────────
        if self.max_context_size is not None:
            _cr = self._compress_context(callback, self.max_context_size)
            if _cr["needed"] and _cr["artefact_pressure"]:
                def _deactivate_artefacts_impl(titles: List[str]) -> Dict[str, Any]:
                    deactivated, not_found = [], []
                    for t in titles:
                        if self.artefacts.get(t) is None:
                            not_found.append(t)
                        else:
                            self.artefacts.deactivate(t)
                            deactivated.append(t)
                    if deactivated:
                        if callback:
                            _cb(callback, json.dumps(deactivated),
                                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                                {"artefacts": deactivated, "action": "deactivated_for_compression"})
                        tokens_after = self._compress_context(
                            callback, self.max_context_size
                        ).get("tokens_after", _cr["tokens_before"])
                    else:
                        tokens_after = _cr["tokens_before"]
                    return {
                        "success": bool(deactivated), "deactivated": deactivated,
                        "not_found": not_found,
                        "tokens_freed_estimate": _cr["tokens_before"] - tokens_after,
                    }
                tool_registry["deactivate_artefacts"] = _deactivate_artefacts_impl
                tool_descriptions.insert(0,
                    "- deactivate_artefacts(titles: list[str]): "
                    "CONTEXT PRESSURE -- deactivate unneeded artifacts first")
                object.__setattr__(
                    self, "_system_prompt",
                    _build_tool_system_prompt(self._system_prompt or "", tool_descriptions)
                )
 
        # ====================================================================
        #  Agentic loop setup
        # ====================================================================
        start_time            = datetime.now()
        is_agentic_turn       = False
        tool_calls_this_turn: List[Dict] = []
        all_events:           List[Dict] = []
        _accumulated_full     = ""
        _clean_text_so_far    = ""
        _round                = 0
        _temp_msg_ids:        List[str] = []
        _current_branch_tip   = branch_tip_id or self.active_branch_id
 
        _completed_tool_calls:    List[str] = []
        _created_artefact_titles: List[str] = []
 
        _MAX_IDENTICAL_REPEATS              = 1
        _identical_call_counts: Dict[str, int] = {}
        _recent_queries: Dict[str, set]        = {}
 
        _round1_no_tool_call = False
 
        ai_message = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"mode": "agentic", "events": []}
        )
        if self._is_db_backed:
            self.commit()
        _cb(callback, ai_message.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE,
            {"message_id": ai_message.id})
 
        # ── Virtual history: use compression when ≥2 artifact turns exist ────
        branch = self.get_branch(_current_branch_tip)
 
        if self._should_compress_agentic_history(branch):
            ASCIIColors.info(
                "[chat] ≥2 artifact-update turns in history — "
                "compressing to PROJECT STATE synopsis"
            )
            _virtual_history = self._build_compressed_history(branch)
            is_tenacious_mode = False  # compression already handles this
        else:
            # Standard path: tenacious mode kicks in at 2+ agentic turns
            agentic_turns = sum(
                1 for m in branch
                if m.metadata.get("mode") in ("agentic", "rlm_agentic")
            )
            is_tenacious_mode = agentic_turns >= 2
 
            if is_tenacious_mode:
                ASCIIColors.warning(
                    "[Tenacious Mode] Collapsing history to PROJECT STATE "
                    "to prevent mimicry poisoning."
                )
                self.summarize_and_prune(force_technical=True, preserve_last_n=1)
                branch = self.get_branch(_current_branch_tip)
 
            _virtual_history = []
            if self.system_prompt:
                _virtual_history.append(SimpleNamespace(
                    sender_type="system", content=self.system_prompt
                ))
 
            if is_tenacious_mode and self.pruning_summary:
                active_art_list = [a.get('title') for a in self.artefacts.list(active_only=True)]
                target_str = (
                    ", ".join(active_art_list) if active_art_list
                    else "None (Create new if needed)"
                )
                synopsis_content = (
                    "╔══════════════════════════════════════════════════════════════════╗\n"
                    "║  📍 PROJECT STATE & TECHNICAL SYNOPSIS                           ║\n"
                    "╠══════════════════════════════════════════════════════════════════╣\n"
                    f"{self.pruning_summary}\n"
                    "║                                                                  ║\n"
                    f"║  PRIMARY WORK TARGETS: {target_str[:40]}...                      ║\n"
                    "╚══════════════════════════════════════════════════════════════════╝"
                )
                _virtual_history.append(SimpleNamespace(
                    sender_type="system", content=synopsis_content
                ))
                if active_art_list:
                    hint = (
                        f"[ARCHITECT NOTICE: Use the established architecture. "
                        f"Prioritize updates to: {', '.join(active_art_list)}]"
                    )
                    _virtual_history.append(SimpleNamespace(
                        sender_type="user", content=hint
                    ))
                if branch:
                    _virtual_history.append(branch[-1])
            else:
                for m in branch:
                    _virtual_history.append(m)
 
        # ── Helper: replace processing blocks with opaque exec marker ─────────
        def _compact_repl(m):
            """Replace <processing>…</processing> with an opaque non-prose marker."""
            return f"\n{_EXEC_MARKER}\n"

        # ── Helper: build synthetic tool_result for history injection ──────────────
        def _build_synthetic_result(action_type: str, title: str, details: str = ""):
            """Create a synthetic tool_result marker that the LLM can learn from."""
            if action_type == "artifact_created":
                return (
                    f'\n<tool_result name="create_artifact">'
                    f'{{"success": true, "title": "{title}", "action": "created"}}'
                    f'</tool_result>\n'
                )
            elif action_type == "artifact_updated":
                return (
                    f'\n<tool_result name="update_artifact">'
                    f'{{"success": true, "title": "{title}", "action": "updated", "details": "{details}"}}'
                    f'</tool_result>\n'
                )
            elif action_type == "tool_execution":
                return (
                    f'\n<tool_result name="{title}">'
                    f'{{"success": true, "result": "executed"}}'
                    f'</tool_result>\n'
                )
            return ""
 
        # ── Per-turn action scratchpad ─────────────────────────────────────────
        _turn_action_history: List[str] = []
 
        # ====================================================================
        #  Main agentic loop
        # ====================================================================
        while _round < max_reasoning_steps:
            _round += 1
 
            # ── Mission-complete fast exit ────────────────────────────────────
            _mission_complete = any(
                "✓ SUCCESSFULLY" in entry for entry in _turn_action_history
            )
 
            if _mission_complete:
                ASCIIColors.success(
                    "[Master] Success detected. Revoking tool access and forcing summary."
                )
                _round = max_reasoning_steps
                active_tool_registry = {
                    "final_answer": tool_registry.get("final_answer"),
                }
                self.scratchpad = (
                    "╔══════════════════════════════════════════════════════════════════╗\n"
                    "║ 🛑 STOP: MISSION ACCOMPLISHED                                    ║\n"
                    "╠══════════════════════════════════════════════════════════════════╣\n"
                    "║ The artifacts are already updated. Tool access has been REVOKED. ║\n"
                    "║ 1. DO NOT call any more tools.                                   ║\n"
                    "║ 2. Summarize your work to the user now.                          ║\n"
                    "╚══════════════════════════════════════════════════════════════════╝"
                )
            else:
                active_tool_registry = tool_registry
 
            _recent_queries.clear()
            _active_temp    = final_answer_temperature
            _saved_scratchpad = self.scratchpad
 
            # ── Build per-round state block ───────────────────────────────────
            state_lines: List[str] = []
 
            if _turn_action_history:
                if any(
                    "✓ SUCCESSFULLY implemented" in e or "✓ SUCCESSFULLY updated" in e
                    for e in _turn_action_history
                ):
                    state_lines += [
                        "╔══════════════════════════════════════════════════════════════════╗",
                        "║ 🛑 STOP: MISSION ACCOMPLISHED                                    ║",
                        "╠══════════════════════════════════════════════════════════════════╣",
                        "║ SUCCESS: The requested changes have been applied and saved.      ║",
                        "║                                                                  ║",
                        "║ 1. DO NOT emit any more <artifact> or <tool_call> tags.          ║",
                        "║ 2. DO NOT describe internal 'processing' or 'updating' steps.    ║",
                        "║ 3. PROVIDE your final summary to the user immediately.           ║",
                        "╚══════════════════════════════════════════════════════════════════╝",
                    ]
                state_lines.append("=== TURN ACTION SCRATCHPAD (Completed so far) ===")
                state_lines.extend(_turn_action_history)
                state_lines.append("================================================")
 
            if _completed_tool_calls or _created_artefact_titles:
                state_lines.append(
                    "=== AGENT STATE (already completed this turn — DO NOT repeat) ==="
                )
                if _completed_tool_calls:
                    state_lines.append("Tool calls already made:")
                    state_lines.extend(f"  ✓ {c}" for c in _completed_tool_calls)
                if _created_artefact_titles:
                    state_lines.append("Artifacts / notes already created:")
                    state_lines.extend(f"  ✓ {t}" for t in _created_artefact_titles)
                state_lines.append("=== END AGENT STATE ===")
 
            state_lines.append(_TOOL_CALL_REMINDER)
 
            if _round == 1 and not _completed_tool_calls:
                state_lines.append(
                    "AVAILABLE TOOLS (quick list): " + ", ".join(tool_registry.keys())
                )
 
            if any(
                "internet_search" in call or "google" in call
                for call in _completed_tool_calls
            ):
                state_lines += [
                    "╔══════════════════════════════════════════════════════════════════╗",
                    "║ 🛠️ ARCHITECT DIRECTIVE: SEARCH COMPLETE                          ║",
                    "╠══════════════════════════════════════════════════════════════════╣",
                    "║ You have retrieved external information. Your ONLY goal now is   ║",
                    "║ to apply this knowledge to the project artifacts.                ║",
                    "║                                                                  ║",
                    "║ 1. DO NOT call 'show_tools' or other diagnostic tools.           ║",
                    "║ 2. DO NOT build forms to ask the user for more choices.          ║",
                    "║ 3. PROCEED IMMEDIATELY to <coding_plan> and <artifact> update.   ║",
                    "╚══════════════════════════════════════════════════════════════════╝",
                ]
 
            self.scratchpad = (
                "\n".join(state_lines) +
                ("\n\n" + (self.scratchpad or "") if self.scratchpad else "")
            )

            if debug or self.lollmsClient.debug:
                ASCIIColors.cyan("=== [DEBUG] MAIN CHAT LOOP STATE ===")
                ASCIIColors.yellow(f"  • Round: {_round}")
                ASCIIColors.yellow(f"  • Active Artifacts: {[a['title'] for a in self.artefacts.list(active_only=True)]}")
                ASCIIColors.yellow(f"  • Scratchpad Summary: {self.scratchpad[:500]}...")
                ASCIIColors.info("====================================")

            round_content_start = len(ai_message.content)
 
            ss = _StreamState(
                discussion            = self,
                callback              = callback,
                ai_message            = ai_message,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = inline_widgets if 'inline_widgets' in locals() else enable_inline_widgets,
                enable_forms          = enable_forms,
                auto_activate_artefacts = auto_activate_artefacts,
                enable_artefacts      = enable_artefacts,
            )

            # Report reasoning loop status to the active processing tag and callback with filled types and details
            if _round == 1:
                ss.proc_type = "agent_reasoning"
                ss.proc_title = "Agent Reasoning"
                ss._emit_processing_status("Formulating thought process and planning action...")
            else:
                ss.proc_type = "agent_reasoning"
                ss.proc_title = f"Agent Reasoning (Round {_round})"
                ss._emit_processing_status(f"Executing follow-up reasoning (Round {_round}/{max_reasoning_steps}). Refining plan...")

            def _inline_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    result = ss.feed(chunk)
                    if ss.tool_trigger:
                        return False
                    return result
                return True
 
            merged_images = self._merge_artefact_images(images)
 
            _gen_kwargs = kwargs.copy()
            _gen_kwargs.pop("stream",              None)
            _gen_kwargs.pop("temperature",         None)
            _gen_kwargs.pop("streaming_callback",  None)
 
            # ── Build sanitized message list for this round ───────────────────
            # The system context (instructions + active artifacts) is always
            # injected fresh as the first message so the model has a clean view.
            # All prior assistant turns are stripped of processing XML and the
            # opaque exec marker is used in place of those blocks.
            # This prevents the model from treating framework markers as prose
            # it should reproduce.
            _formatted_messages: List[Dict[str, str]] = []
 
            full_system_context = (
                (self.system_prompt or "") + "\n\n" + self.get_full_data_zone()
            )
            if self.scratchpad:
                full_system_context += "\n\n" + f"=== SCRATCHPAD / TEMPORARY CONTEXT ===\n{self.scratchpad}\n=== END SCRATCHPAD ==="
            _formatted_messages.append({
                "role":    "system",
                "content": full_system_context.strip(),
            })
 
            for m in _virtual_history:
                if m.sender_type == "system":
                    # Already injected above; skip to avoid duplication
                    continue
                role    = m.sender_type
                content = m.content or ""
                metadata = getattr(m, 'metadata', {}) or {}

                if role == "assistant":
                    # Replace all processing blocks with the opaque marker
                    content = re.sub(
                        r'<processing.*?>.*?</processing>',
                        f'\n{_EXEC_MARKER}\n',
                        content, flags=re.DOTALL
                    )
                    content = re.sub(r'<lollms_event.*?>', '', content)
                    # Strip any leaked status / log lines
                    content = re.sub(
                        r'^[ \t]*[*✓🏗️🔧✅❌·ᴽЧØс].*$', '',
                        content, flags=re.MULTILINE
                    )
                    # Strip the marker itself — the model must never see it in
                    # an assistant turn it can learn to reproduce
                    content = _EXEC_MARKER_RE.sub('', content)

                msg_role = (
                    "assistant" if role == "assistant"
                    else ("user" if role == "user" else "system")
                )
                content = content.strip()
                if content:
                    _formatted_messages.append({"role": msg_role, "content": content})

                # ── INJECT SYNTHETIC TOOL RESULTS FOR ACTION LEARNING ──────────────
                # If this assistant message has action metadata, inject a synthetic
                # tool_result that shows what happened after the action
                if role == "assistant" and metadata:
                    artifacts_modified = metadata.get("artefacts_modified", [])
                    events = metadata.get("events", [])

                    # Detect if this was an action-less reasoning turn
                    has_action = bool(artifacts_modified or events)

                    if has_action:
                        # Inject artifact action results
                        for art_title in artifacts_modified:
                            # Check if it was a creation or update based on content presence
                            art = self.artefacts.get(art_title)
                            if art:
                                action_type = "artifact_created" if art.get("version", 1) == 1 else "artifact_updated"
                                synth_result = _build_synthetic_result(
                                    action_type,
                                    art_title,
                                    f"version {art.get('version', 1)}"
                                )
                                if synth_result:
                                    _formatted_messages.append({
                                        "role": "system",
                                        "content": synth_result
                                    })

                        # Inject tool execution results
                        for evt in events:
                            if evt.get("type") == "tool_call" and evt.get("tool"):
                                tool_name = evt.get("tool")
                                synth_result = _build_synthetic_result(
                                    "tool_execution",
                                    tool_name
                                )
                                if synth_result:
                                    _formatted_messages.append({
                                        "role": "system",
                                        "content": synth_result
                                    })
                    else:
                        # CRITICAL FIX: Inject synthetic FAILURE marker for action-less turns
                        # This teaches the LLM that reasoning without action = failure
                        synth_failure = (
                            f'\n<tool_result name="action_failure">'
                            f'{{"success": false, "error": "NO ACTION TAKEN: You produced reasoning but no tools were called and no artifacts were modified. The system requires you to call tools or emit <artifact> tags to make progress."}}'
                            f'</tool_result>\n'
                        )
                        _formatted_messages.append({
                            "role": "system",
                            "content": synth_failure
                        })
 
            # Images
            _binding = getattr(self.lollmsClient, 'llm', None)
            _vision_ok = getattr(_binding, 'supports_vision', True)  # default True for compat
            if merged_images and not _vision_ok:
                # Build a text description fallback
                _img_note = (
                    "\n[SYSTEM: The user attached an image but this model has no vision capability. "
                    "Tell the user you cannot see images and ask them to describe it.]\n"
                )
                self.scratchpad = (self.scratchpad or "") + _img_note
                merged_images = []  # don't pass images that will be silently dropped

            elif merged_images:
                # Force weak models to acknowledge the image first
                _img_preamble = (
                    f"\n[VISION INPUT: {len(merged_images)} image(s) attached. "
                    "You MUST look at the image(s) and base your answer on what you actually see. "
                    "Do NOT answer from memory or conversation history when image content is relevant.]\n"
                )
                self.scratchpad = (self.scratchpad or "") + _img_preamble

            self.lollmsClient.generate_from_messages(
                messages           = _formatted_messages,
                images             = merged_images,
                stream             = True,
                temperature        = _active_temp,
                streaming_callback = _inline_relay,
                **_gen_kwargs
            )
 
            ss.flush_remaining_buffer()
            self.scratchpad = _saved_scratchpad
 
            # ── Post-round scrub: remove any leaked internal markers ───────────
            if _EXEC_MARKER_RE.search(ai_message.content):
                ASCIIColors.warning(
                    "[chat] Scrubbing leaked internal markers from ai_message.content"
                )
                ai_message.content = _EXEC_MARKER_RE.sub('', ai_message.content).strip()
 
            _so_far = "".join(ss.stream_buf)
            _accumulated_full += _so_far
 
            # Clean up raw tool_call block from visible content if it leaked
            if ss.tool_trigger:
                match = re.search(
                    r"<tool_call>.*?</tool_call>",
                    ai_message.content[round_content_start:], re.DOTALL
                )
                if match:
                    ai_message.content = (
                        ai_message.content[:round_content_start + match.start()]
                    )
 
            _round_clean       = "".join(ss.clean_prose)
            _clean_text_so_far += _round_clean
 
            # ── Artifact-update failure: re-loop with correction ───────────────
            if ss.patch_error_occurred and not ss.affected_artefacts:
                _error_breadcrumb = (
                    "\n"
                    "⚠️ [EXECUTION FAILURE] ⚠️\n"
                    f"The update to '{ss.proc_title}' failed to match any content in the file.\n"
                    "ACTION REQUIRED: Do not repeat the same SEARCH block. Re-read the file "
                    "content provided in the system context and perform a FULL REWRITE or a "
                    "more surgical patch.\n"
                )
                _clean_so_far_for_llm = re.sub(
                    r'<processing.*?>.*?</processing>',
                    _compact_repl, _so_far, flags=re.DOTALL
                )
                _virtual_history.append(SimpleNamespace(
                    sender_type="assistant", content=_clean_so_far_for_llm.strip()
                ))
                _virtual_history.append(SimpleNamespace(
                    sender_type="user", content=_error_breadcrumb
                ))
                # Preserve accumulated content; only reset the clean prose tracker
                # so the next round starts fresh but history is kept.
                _clean_text_so_far = ai_message.content[:round_content_start]
                ss.clean_prose.clear()
                ASCIIColors.error(
                    "[chat] Artifact update failed — re-looping to inform LLM."
                )
                continue
 
            _tool_trigger  = ss.tool_trigger
            _tool_json_str = ss.get_tool_call_json()

            _artefacts_built = len(ss.affected_artefacts) > 0
            _has_action_tags = any(
                tag in _so_far.lower()
                for tag in ["<generate_image", "<edit_image", "<lollms_inline", "<lollms_form", "<revert_artifact", "<revert_artefact"]
            )
            # Check if specialist executed successfully (coding_plan with successful implementation)
            _specialist_executed = hasattr(ss, 'pending_final_content') and bool(ss.pending_final_content)

            _did_something   = _tool_trigger or _artefacts_built or _has_action_tags or _specialist_executed

            # ── Round-1 correction: model produced prose instead of action ────
            if _round == 1 and not _did_something:
                _output_clean = _round_clean.strip()
                _intent_to_act = any(
                    kw in user_message.lower()
                    for kw in [
                        "update", "add", "change", "fix", "create", "make",
                        "music", "color", "search", "write",
                    ]
                )
                _needs_correction = False
                _correction_msg   = ""

                if _has_external_tools and (
                    len(_output_clean) < 50 or
                    re.search(
                        r"i (cannot|can't|don't|am unable|have no access|cannot access)",
                        _output_clean, re.IGNORECASE
                    )
                ):
                    _needs_correction = True
                    _correction_msg   = _TOOL_CALL_CORRECTION

                elif _intent_to_act and not _artefacts_built:
                    _needs_correction = True
                    _correction_msg   = (
                        "\n"
                        "╔══════════════════════════════════════════════════════════════════╗\n"
                        "║  🧠 MENTAL RESET — HALLUCINATION DETECTED                        ║\n"
                        "╠══════════════════════════════════════════════════════════════════╣\n"
                        "║  You claimed to be doing work but you ONLY produced prose.       ║\n"
                        "║  The system DID NOT save any changes because you omitted the tags.║\n"
                        "║                                                                  ║\n"
                        "║  1. STOP typing tables or lists describing what you 'did'.       ║\n"
                        "║  2. STOP acting like the work is finished. It is NOT.            ║\n"
                        "║  3. EMIT the functional <artifact> or <tool_call> tags NOW.      ║\n"
                        "║  4. If you do not emit a tag, this conversation cannot proceed.  ║\n"
                        "╚══════════════════════════════════════════════════════════════════╝\n"
                    )

                if _needs_correction:
                    _round1_no_tool_call = True
                    ASCIIColors.yellow(
                        f"[chat] Round 1 failed to act — injecting correction."
                    )
                    _warning(callback, "Action missing; forcing model to emit tags/calls.")

                    # Notify user of intent correction and loop restart
                    ss._emit_processing_status("⚠️ Diagnostic: Action/tag was missing in response. Injecting correction and retrying...")
                    ss._emit_processing_close()

                    _clean_so_far_for_llm = re.sub(
                        r'<processing.*?>.*?</processing>',
                        _compact_repl, _so_far, flags=re.DOTALL
                    )
                    _virtual_history.append(SimpleNamespace(
                        sender_type="assistant", content=_clean_so_far_for_llm.strip()
                    ))
                    _virtual_history.append(SimpleNamespace(
                        sender_type="user", content=_TOOL_CALL_CORRECTION
                    ))
                    # Preserve prior accumulated content; reset trackers for next round
                    _clean_text_so_far = ai_message.content[:round_content_start]
                    _accumulated_full  = _accumulated_full[:len("".join(ss.stream_buf[:0]))] or ""
                    ai_message.content = ai_message.content[:round_content_start]
                    continue
 
            # CRITICAL FIX: Enhanced loop exit detection with rescue protocol
            if not _tool_trigger and not _artefacts_built and not _has_action_tags and not _specialist_executed:
                # Check if we're in Round 2+ and still no action - force final answer now!
                if _round >= 2:
                    ASCIIColors.warning(
                        f"[chat] Round {_round} also failed to act after correction. "
                        "Forcing final answer pass with all gathered context."
                    )
                    # Set flag to trigger immediate forced answer after loop
                    is_agentic_turn = True
                    _turn_action_history.append(
                        "⚠️ CORRECTION: Model failed to call tools in Round 2. Forcing summary."
                    )
                break

        # ====================================================================
            #  Tool execution
            # ====================================================================
            def _parse_tool_call_lenient(raw: str) -> dict:
                """Try json.loads first, then regex extraction as fallback."""
                raw = raw.strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
                # Fallback: extract name and parameters by regex
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                params_match = re.search(r'"parameters"\s*:\s*(\{[^}]*\})', raw, re.DOTALL)
                if name_match:
                    name = name_match.group(1)
                    try:
                        params = json.loads(params_match.group(1)) if params_match else {}
                    except Exception:
                        params = {}
                    return {"name": name, "parameters": params}
                return {}
            is_agentic_turn = True
            try:
                _call_data   = _parse_tool_call_lenient(_tool_json_str or "{}")
                _tool_name   = _call_data.get("name", "")
                _tool_params = dict(_call_data.get("parameters", {}) or {})
                # Merge top-level keys as fallbacks for flat parameters
                for k, v in _call_data.items():
                    if k not in ("name", "parameters"):
                        _tool_params.setdefault(k, v)
            except Exception as e:
                _warning(callback, f"Failed to parse tool call: {e}")
                break
 
            _params_summary = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in _tool_params.items()
            )
            _call_signature = f"{_tool_name}({_params_summary})"
            _call_tag       = f"round {_round}: {_call_signature}"
 
            # Duplicate / semantic-loop detection
            _identical_call_counts[_call_signature] = (
                _identical_call_counts.get(_call_signature, 0) + 1
            )
            _sig_count = _identical_call_counts[_call_signature]
 
            _query_key    = str(
                _tool_params.get("query", _tool_params.get("prompt", ""))
            ).strip().lower()
            _is_semantic_dup = False
            if _query_key and len(_query_key) > 3:
                _prev_queries = _recent_queries.setdefault(_tool_name, set())
                if _query_key in _prev_queries:
                    _is_semantic_dup = True
                _prev_queries.add(_query_key)
 
            if _sig_count > _MAX_IDENTICAL_REPEATS or _is_semantic_dup:
                _dup_type = (
                    "IDENTICAL" if _sig_count > _MAX_IDENTICAL_REPEATS
                    else "SEMANTIC DUPLICATE"
                )
                _warning(callback,
                         f"[RUNAWAY] {_dup_type} call — injecting pattern-break.")
                _shake_prompt = (
                    "\n"
                    "╔══════════════════════════════════════════════════════════════════╗\n"
                    "║  🧠 MENTAL RESET — PATTERN DISRUPTION ACTIVE                     ║\n"
                    "╠══════════════════════════════════════════════════════════════════╣\n"
                    "║  You have fallen into a repetitive logic loop.                   ║\n"
                    "║  1. STOP following your previous reasoning path.                 ║\n"
                    "║  2. DISREGARD your previous failed tool attempts.                ║\n"
                    "║  3. RE-EVALUATE the user's need from a fresh perspective.        ║\n"
                    "║  4. ACT NOW: Change your approach or provide the final answer.   ║\n"
                    "╚══════════════════════════════════════════════════════════════════╝\n"
                )
                _clean_so_far_for_llm = re.sub(
                    r'<processing.*?>.*?</processing>',
                    _compact_repl, _so_far, flags=re.DOTALL
                )
                _virtual_history.append(SimpleNamespace(
                    sender_type="assistant", content=_clean_so_far_for_llm.strip()
                ))
                _virtual_history.append(SimpleNamespace(
                    sender_type="user", content=_shake_prompt
                ))
                _clean_text_so_far = ""
                ss.clean_prose.clear()
                ai_message.content = ai_message.content[:round_content_start]
                continue
 
            _current_offset = len(_clean_text_so_far)
            _call_id        = str(uuid.uuid4())
            _call_evt = {
                "type": "tool_call", "content": f"Calling {_tool_name}",
                "id": _call_id, "tool": _tool_name,
                "params": _tool_params, "offset": _current_offset,
            }
            _cb(callback, _call_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_CALL, _call_evt)
            all_events.append(_call_evt)
 
            _marker    = f"\n<lollms_event id=\"{_call_id}\" />\n"
            _clean_text_so_far    += _marker
            ai_message.content     = _clean_text_so_far
            ai_message.metadata["events"] = list(all_events)
 
            _step_lbl = f"Running: {_tool_name.replace('_', ' ').title()}"
            _step_id  = _step_start(callback, _step_lbl,
                                    {"tool": _tool_name, "offset": _current_offset})
            all_events.append({"type": "step_start", "content": _step_lbl,
                               "id": _step_id, "offset": _current_offset})
 
            tool_title = _tool_name.replace('_', ' ').title()
            ss.proc_title = tool_title
            ss._emit_tool_processing_open(_tool_name, _tool_params)
 
            _already_done = any(
                "✓ SUCCESSFULLY implemented" in entry
                for entry in _turn_action_history
            )
 
            if _already_done and _tool_name not in ["final_answer"]:
                ss._emit_tool_processing_status(
                    "Tool Access Revoked: Mission already complete."
                )
                _result     = {"success": False,
                               "error": "MISSION COMPLETE. Do not call more tools. Summarize now."}
                _result_str = json.dumps(_result)
                ss._emit_tool_processing_close("Blocked")
 
            elif _tool_name not in active_tool_registry:
                _err_msg = (
                    f"Tool '{_tool_name}' is currently DISABLED (Mission Complete)."
                    if _mission_complete else
                    f"Tool '{_tool_name}' not found."
                )
                ss._emit_tool_processing_status(_err_msg)
                _warning(callback, _err_msg)
                _result_str = f"Error: {_err_msg}"
                _result     = {"error": _result_str}
                _err_evt = {
                    "type": "tool_output", "content": _result_str,
                    "id": str(uuid.uuid4()), "tool": _tool_name,
                    "result": _result, "offset": _current_offset,
                }
                _cb(callback, _err_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _err_evt)
                _step_end(callback, f"Unknown tool '{_tool_name}'",
                          _step_id, {"status": "failed"})
                all_events.extend([
                    _err_evt,
                    {"type": "step_end", "content": f"Unknown tool '{_tool_name}'",
                     "id": _step_id, "offset": _current_offset, "status": "failed"},
                ])
                ss._emit_tool_processing_close("Failed: tool not found")
 
            else:
                try:
                    ss._emit_tool_processing_status(f"Executing {_tool_name}...")
                    _result = active_tool_registry[_tool_name](**_tool_params)

                    # Normalize multi-structured tool output into a robust LCPResult instance
                    from lollms_client.lollms_types import LCPResult
                    if isinstance(_result, LCPResult):
                        _result_obj = _result
                    elif isinstance(_result, dict) and ("success" in _result or "prompt_injection" in _result):
                        _result_obj = LCPResult.from_dict(_result)
                    else:
                        _success = True
                        _err = None
                        if isinstance(_result, dict):
                            _success = _result.get("success", "error" not in _result)
                            _err = _result.get("error")
                            _out = _result.get("output", json.dumps(_result, indent=2))
                        else:
                            _out = str(_result)
                        _result_obj = LCPResult(success=_success, output=_out, error=_err)

                    # ── 📸 Multi-Modal Image Routing ──
                    if _result_obj.images:
                        # Append to the active turn's images list so subsequent reasoning rounds can "see" them
                        if images is None:
                            images = []
                        for img_b64 in _result_obj.images:
                            if img_b64 not in images:
                                images.append(img_b64)
                        # Append to the assistant's message image pack so they are fully rendered in the UI
                        ai_message.add_image_pack(
                            images=_result_obj.images,
                            group_type="generated",
                            active_by_default=True,
                            title=f"{_tool_name}_img"
                        )

                    # ── 💻 Extra Structured Code Blocks Formatting ──
                    if _result_obj.code_blocks:
                        code_text_blocks = []
                        for blk in _result_obj.code_blocks:
                            lang = blk.get("language") or "python"
                            code_text_blocks.append(f"\n```{lang}\n{blk.get('content')}\n```")
                        _result_obj.output += "\n" + "\n".join(code_text_blocks)

                    # ── 📂 Monospace Paths Ingestion ──
                    if _result_obj.paths:
                        path_texts = [f"  • `{p}`" for p in _result_obj.paths]
                        _result_obj.output += "\n\nGenerated files:\n" + "\n".join(path_texts)

                    # ── 🌐 Structured Sources Ingestion ──
                    # If the tool returned structured sources (RAG, databases, search),
                    # ingest them directly so they are beautifully cited and formatted
                    if _result_obj.sources:
                        for s in _result_obj.sources:
                            s.setdefault("index", len(collected_sources) + 1)
                            s.setdefault("tool", _tool_name)
                            collected_sources.append(s)

                    # ── 🧠 Context Ingestion ──
                    inferred_srcs = _infer_sources_from_json(_result_obj.to_dict(), _tool_name)
                    res_label     = _tool_name.replace('_', ' ').title()
                    llm_block     = f"### [Source List: {res_label}]\n"
                    for s in inferred_srcs:
                        if s not in collected_sources:
                            s["index"] = len(collected_sources) + 1
                            collected_sources.append(s)
                        llm_block += f"[[{s['index']}]] {s['title']}\n"
                        if s['content']:
                            llm_block += f"Content: {s['content']}\n"
                        if s['metadata']:
                            llm_block += f"Metadata: {json.dumps(s['metadata'])}\n"
                        llm_block += "---\n"
                    if not inferred_srcs:
                        llm_block += _result_obj.output

                    self.scratchpad = (self.scratchpad or "") + (
                        f"\n--- Tool: {res_label} (round {_round}) ---\n"
                        f"{llm_block}\n"
                        f"--- End {res_label} ---\n"
                    )

                    _completed_tool_calls.append(_call_tag)
                    _created_title = (
                        _result_obj.metadata.get("title") or _result_obj.metadata.get("name") or
                        _tool_params.get("title") or _tool_params.get("name")
                    )
                    if _created_title and str(_created_title) not in _created_artefact_titles:
                        _created_artefact_titles.append(str(_created_title))

                    # ── 🧠 Custom Prompt Injection Override ──
                    # If prompt_injection is provided, we completely bypass JSON serialization and inject the custom text directly!
                    if _result_obj.prompt_injection:
                        _result_str = _result_obj.prompt_injection
                    else:
                        _raw_result_json = json.dumps(_result_obj.to_dict(), indent=2, ensure_ascii=False)

                        _is_web_search = (
                            ("search" in _tool_name.lower() or "query" in _tool_name.lower())
                            and isinstance(_result, dict) and "sources" in _result
                        )
                        if _is_web_search:
                            _formatted = _format_web_search_for_llm(_result, max_chars=2000)
                            _result_str = (
                                _formatted[:2500] + "\n... [additional results truncated]"
                                if len(_formatted) > 2500 else _formatted
                            )
                        else:
                            if len(_raw_result_json) <= 2000:
                                _result_str = _raw_result_json
                            else:
                                _truncated  = _smart_truncate_result(_result_obj.to_dict(), max_total_chars=2000)
                                _result_str = json.dumps(_truncated, indent=2, ensure_ascii=False)

                    tool_calls_this_turn.append({
                        "name": _tool_name, "params": _tool_params, "result": _result_obj.to_dict(),
                    })
 
                    _source_count = len(inferred_srcs)
                    _result_keys  = list(_result_obj.to_dict().keys()) if isinstance(_result_obj.to_dict(), dict) else []
                    _is_failed = not _result_obj.success

                    if _is_failed:
                        result_summary = f"❌ Failed: {res_label}"
                    else:
                        result_summary = f"✓ Success: {res_label}"

                    if _source_count:
                        result_summary += f" — found {_source_count} source(s)"
                    if _result_keys:
                        result_summary += f" — result keys: {', '.join(_result_keys[:5])}"
                    ss._emit_tool_processing_status(result_summary)

                    if _is_failed and _result_obj.error:
                        # Render tool execution failure details as an elegant collapsible block inside the processing drawer
                        error_details = f"<details class='proc-error-details'><summary>⚠️ Error Details</summary><pre style='color:#ef4444; white-space:pre-wrap;'>{_result_obj.error}</pre></details>"
                        ss._emit_tool_processing_status(error_details)
                    elif _result_obj.success and _result_obj.output.strip():
                        # Render successful output / stdout as an elegant collapsible block inside the processing drawer
                        clean_out = _result_obj.output.strip()
                        escaped_out = clean_out.replace("<", "&lt;").replace(">", "&gt;")
                        if len(escaped_out) > 3000:
                            escaped_out = escaped_out[:3000] + "\n... [additional output truncated]"
                        status_out = f"<details class='proc-success-details'><summary>💻 Execution Output / Return Value</summary><pre style='color:#10b981; white-space:pre-wrap;'>{escaped_out}</pre></details>"
                        ss._emit_tool_processing_status(status_out)
 
                    ss._emit_tool_processing_close(
                        f"Completed — output: {len(_raw_result_json):,} chars"
                    )
 
                    _out_evt = {
                        "type": "tool_output", "content": _result_str,
                        "id": str(uuid.uuid4()), "tool": _tool_name,
                        "result": _result, "offset": _current_offset,
                    }
                    _cb(callback, _out_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _out_evt)
                    _step_end(callback, f"Done: {_tool_name}",
                              _step_id, {"status": "success"})
                    all_events.extend([
                        _out_evt,
                        {"type": "step_end", "content": f"Done: {_tool_name}",
                         "id": _step_id, "offset": _current_offset, "status": "success"},
                    ])
 
                    current_meta = dict(ai_message.metadata or {})
                    current_meta["events"]  = list(all_events)
                    current_meta["sources"] = list(collected_sources)
                    ai_message.metadata = current_meta
 
                    if self._is_db_backed:
                        self.commit()
 
                    _q       = _tool_params.get("query", _tool_params.get("prompt", ""))
                    raw_srcs = _result.get("sources", [])
                    if not raw_srcs:
                        cand = _result.get(
                            "results", _result.get(
                                "content",
                                _result if isinstance(_result, list) else []
                            )
                        )
                        if isinstance(cand, list):
                            raw_srcs = cand
                    if raw_srcs and isinstance(raw_srcs, list):
                        queries_performed.append({
                            "step": _round, "tool": _tool_name, "query": _q,
                            "result_count": len(raw_srcs),
                        })
                        for _doc in raw_srcs:
                            if not isinstance(_doc, dict):
                                continue
                            _doc_title   = (
                                _doc.get("title") or _doc.get("name") or
                                _doc.get("source") or f"{_tool_name} Result"
                            )
                            _doc_content = (
                                _doc.get("content") or _doc.get("summary") or
                                _doc.get("snippet") or ""
                            )
                            _doc_link = (
                                _doc.get("link") or _doc.get("url") or
                                _doc.get("source") or ""
                            )
                            collected_sources.append({
                                "title":           _doc_title,
                                "content":         _doc_content,
                                "source":          _doc_link,
                                "query":           _q,
                                "relevance_score": _doc.get("score",
                                    _doc.get("relevance", 100 if _doc_content else 0)),
                                "index":           len(collected_sources) + 1,
                                "tool":            _tool_name,
                            })
 
                except Exception as e:
                    if self.lollmsClient.debug:
                        trace_exception(e)
                    ss._emit_tool_processing_status(f"Error during execution: {str(e)[:100]}")
                    _warning(callback, f"Tool error ({_tool_name}): {e}")
                    ss._emit_tool_processing_close(f"Failed: {str(e)[:200]}")
                    _step_end(callback, f"Error: {e}", _step_id, {"status": "error"})
                    all_events.append({
                        "type": "step_end", "content": f"Error: {e}",
                        "id": _step_id, "offset": _current_offset, "status": "error",
                    })
                    _result_str = f"Error: {e}"
                    _result     = {"error": _result_str}
 
            # ── Large output condensation ─────────────────────────────────────
            _result_len = len(_result_str)
            if _result_len > _LARGE_OUTPUT_THRESHOLD_CHARS and _tool_name != "final_answer":
                _reading_id = _step_start(
                    callback, f"Condensing large output ({_result_len:,} chars)..."
                )
                from ._repl_tools import TextBuffer
                _read_buf   = TextBuffer()
                _buf_handle = f"tool_result_{_tool_name}_{_round}"
                _read_buf.store(_buf_handle, _result_str)
 
                _query = _tool_params.get("query", _tool_params.get("prompt", ""))
                if _query:
                    _search_res       = _read_buf.search(_buf_handle, _query, max_results=5)
                    _relevant_indices = [h["index"] for h in _search_res.get("hits", [])]
                else:
                    _list_res         = _read_buf.list_records(_buf_handle, page=1, page_size=5)
                    _relevant_indices = [i["index"] for i in _list_res.get("items", [])]
 
                _condensed_parts = []
                for _idx in _relevant_indices[:5]:
                    _rec_res = _read_buf.get_record(_buf_handle, _idx)
                    if _rec_res.get("success"):
                        _condensed_parts.append(f"[Record {_idx}] {_rec_res['record']}")
 
                _condensed_summary = (
                    "\n".join(_condensed_parts) if _condensed_parts
                    else _result_str[:1500]
                )
                _reading_note = (
                    f"\n--- Tool: {_tool_name} (large output, {_result_len:,} chars) ---\n"
                    f"Query: {_query or '(none)'}\n"
                    f"Key findings ({len(_relevant_indices)} relevant records):\n"
                    f"{_condensed_summary}\n"
                    f"--- End {_tool_name} ---\n"
                )
                self.scratchpad = (self.scratchpad or "") + _reading_note
                _step_end(callback,
                          f"Condensed to {len(_condensed_summary):,} chars", _reading_id)
                _result_str_for_llm = (
                    f"[LARGE OUTPUT CONDENSED — original was {_result_len:,} chars]\n"
                    f"{_condensed_summary}\n"
                    f"[Full data available in scratchpad if more detail needed]"
                )
            else:
                _result_str_for_llm = _result_str
 
            # ── Update virtual history for the next round ─────────────────────
            # Strip processing blocks and status markers; use the opaque marker.
            _clean_so_far_for_llm = re.sub(
                r'<processing.*?>.*?</processing>',
                _compact_repl, _so_far, flags=re.DOTALL
            )
            _clean_so_far_for_llm = re.sub(
                r'^[ \t]*[*✓🏗️🔧✅❌].*$', '',
                _clean_so_far_for_llm, flags=re.MULTILINE
            )
            # Final safety: remove the marker itself so the model can't see it
            _clean_so_far_for_llm = _EXEC_MARKER_RE.sub('', _clean_so_far_for_llm)

            _virtual_history.append(SimpleNamespace(
                sender_type="assistant",
                content=_clean_so_far_for_llm.strip()
            ))
            # Keep the real tool result for tool calls
            _virtual_history.append(SimpleNamespace(
                sender_type="user",
                content=f'<tool_result name="{_tool_name}">{_result_str_for_llm}</tool_result>'
            ))

            # ── INJECT SYNTHETIC ARTIFACT RESULTS OR FAILURE MARKERS ───────────
            # If artifacts were created/updated, inject synthetic results so the
            # model learns the pattern: emit tag → see result
            if ss.affected_artefacts:
                for art in ss.affected_artefacts:
                    art_title = art.get("title", "unknown")
                    action_type = "artifact_created" if art.get("version", 1) == 1 else "artifact_updated"
                    synth_result = _build_synthetic_result(
                        action_type,
                        art_title,
                        f"version {art.get('version', 1)}"
                    )
                    if synth_result:
                        _virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=synth_result
                        ))
            else:
                # CRITICAL FIX: Inject synthetic FAILURE marker when no action was taken
                # This teaches the LLM that reasoning without action = failure
                synth_failure = (
                    f'\n<tool_result name="action_failure">'
                    f'{{"success": false, "error": "NO ACTION TAKEN: You produced reasoning but no tools were called and no artifacts were modified. The system requires you to call tools or emit <artifact> tags to make progress."}}'
                    f'</tool_result>\n'
                )
                _virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=synth_failure
                ))
            # CRITICAL FIX: Preserve the full accumulated content in ai_message.content,
            # not just the clean prose. This ensures tool results and all text survive.
            ai_message.content = _accumulated_full

            # CRITICAL FIX: If there's pending_final_content from specialist/coding_plan, add it!
            if hasattr(ss, 'pending_final_content') and ss.pending_final_content:
                ai_message.content += "\n\n" + ss.pending_final_content
                # Don't append to clean_prose here - that's for streaming chunks only

            ss.clean_prose.clear()
 
        # ====================================================================
        #  Forced final-answer pass
        # ====================================================================
        _tool_call_count   = len(tool_calls_this_turn)
        # Use ai_message.content as the source of truth for text produced,
        # since it accumulates everything including between tool calls.
        _has_produced_text = (
            bool(ai_message.content.strip()) and
            len(ai_message.content.strip()) > 100
        )
        _seems_stuck = _tool_call_count >= 3 and not _has_produced_text

        # Check if specialist successfully executed but no final answer was shown
        _specialist_succeeded = hasattr(ss, 'pending_final_content') and bool(ss.pending_final_content)

        # Detect if we exited the loop after multiple failed reasoning rounds
        _loop_exit_after_correction_failure = (
            _round >= 2 and 
            not _tool_trigger and 
            not _artefacts_built and
            any("failed to act" in entry or "CORRECTION" in entry for entry in _turn_action_history)
        )

        # CRITICAL FIX: Force answer if multiple rounds failed to act OR specialist succeeded but no output
        _needs_forced_answer = (
            is_agentic_turn and (
                not _has_produced_text or  # No text at all
                _loop_exit_after_correction_failure or  # Failed to act in Round 2+ after correction
                (_specialist_succeeded and not ai_message.content.strip()) or  # Specialist worked but no output shown
                (_round >= max_reasoning_steps) or  # Max rounds reached
                (_seems_stuck) or  # Too many tool calls with no text
                (tool_calls_this_turn and not any(
                    c.get("name") == "final_answer"
                    for c in tool_calls_this_turn
                ))  # Tools called but no final answer signal
            )
        )

        if _needs_forced_answer:
            _final_id = _step_start(callback, "Generating final answer...")

            # Preserve all content generated so far; final answer appends to it
            _content_before_final = ai_message.content

            ss_final = _StreamState(
                callback              = callback,
                ai_message            = ai_message,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = enable_inline_widgets,
                enable_forms          = enable_forms,
                discussion            = self,
            )
            # Do NOT clear ai_message.content — the final answer should append
 
            def _final_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss_final.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    return ss_final.feed(chunk)
                return True
 
            _scratch_before_final = self.scratchpad
 
            _forced_prompt_parts = [
                "[SYSTEM INSTRUCTION — MANDATORY — FINAL ANSWER REQUIRED]",
                "",
                "You have completed all necessary research and tool calls.",
                "Your task NOW is to write a comprehensive final answer to the user's",
                "original question using the information gathered below.",
                "",
                "⚠️  ABSOLUTE PROHIBITIONS:",
                "• Do NOT call any more tools — there are no more tool calls allowed",
                "• Do NOT emit <tool_call> tags — they will be ignored",
                "• Do NOT ask follow-up questions — answer what you have",
                "• Do NOT say 'I need more information' — use what you have gathered",
                "",
                "✅  REQUIREMENTS:",
                "• Synthesize ALL gathered information into a coherent response",
                "• Cite sources using [1], [2] format when applicable",
                "• Answer directly, completely, and concisely",
                "• If information is incomplete, say so briefly then answer with what you have",
            ]
 
            if _scratch_before_final and _scratch_before_final.strip():
                _forced_prompt_parts.extend([
                    "",
                    "=== GATHERED INFORMATION (scratchpad) ===",
                    _scratch_before_final.strip()[:3000],
                    "=== END GATHERED INFORMATION ===",
                ])
            else:
                _forced_prompt_parts.extend([
                    "",
                    "[NO GATHERED INFORMATION — answer from your general knowledge]",
                ])
 
            _forced_prompt_parts.extend([
                "",
                "[END SYSTEM INSTRUCTION — WRITE YOUR FINAL ANSWER NOW]",
            ])
 
            self.scratchpad = "\n".join(_forced_prompt_parts)
 
            merged_images = self._merge_artefact_images(images)
            kwargs['streaming_callback'] = _final_relay
            self.lollmsClient.chat(
                self,
                images             = merged_images,
                branch_tip_id      = _current_branch_tip,
                stream             = True,
                temperature        = final_answer_temperature,
                **kwargs,
            )
 
            ss_final.flush_remaining_buffer()
            self.scratchpad = _scratch_before_final

            # The final answer has been streaming into ai_message.content;
            # ensure we capture the complete result including any prior content.
            _accumulated_full  += "".join(ss_final.stream_buf)
            # ai_message.content already contains _content_before_final + new final text
            _clean_text_so_far  = ai_message.content
            _step_end(callback, "Final answer generated", _final_id)
 
        # ── Clean up temporary tool-history messages ──────────────────────────
        for mid in reversed(_temp_msg_ids):
            if mid == user_msg.id:
                continue
            if hasattr(self, 'remove_message'):
                self.remove_message(mid)
            elif hasattr(self, 'delete_message'):
                self.delete_message(mid)
            else:
                self.db_manager.delete_message(mid)
 
        # ── Final content cleanup ─────────────────────────────────────────────
        import re as _re
        # Use ai_message.content as the primary source — it contains the
        # complete accumulated text including all rounds and final answer.
        _raw_final = ai_message.content or _clean_text_so_far or ""
        _clean = _re.sub(
            r"<tool_call>.*?(?:</tool_call>|$)", "",
            _raw_final, flags=_re.DOTALL
        ).strip()
        # Remove any residual internal markers
        _clean = _EXEC_MARKER_RE.sub('', _clean).strip()
        if remove_thinking_blocks:
            _clean = self.lollmsClient.remove_thinking_blocks(_clean)
 
        end_time    = datetime.now()
        duration    = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(_clean)
        tok_per_sec = (token_count / duration) if duration > 0 else 0
 
        message_meta: Dict[str, Any] = {
            "mode": (
                "rlm_agentic" if rlm_enabled
                else ("agentic" if is_agentic_turn else "direct")
            ),
            "duration_seconds":  duration,
            "token_count":       token_count,
            "tokens_per_second": tok_per_sec,
        }
        if tool_calls_this_turn:
            message_meta["tool_calls"] = tool_calls_this_turn
        if all_events:
            existing_events = message_meta.get("events", [])
            for evt in all_events:
                if evt not in existing_events:
                    existing_events.append(evt)
            message_meta["events"] = existing_events
        if collected_sources:
            message_meta["sources"]       = collected_sources
        if queries_performed:
            message_meta["query_history"] = queries_performed
        if is_agentic_turn:
            message_meta["scratchpad"]    = scratchpad_state
        if self_corrections:
            message_meta["self_corrections"] = self_corrections
        if _pt_specs:
            message_meta["personality_tools_used"] = [
                n for n in _pt_specs
                if any(tc["name"] == n for tc in tool_calls_this_turn)
            ]
        if _round1_no_tool_call:
            message_meta["round1_correction_applied"] = True
 
        self.scratchpad = ""
 
        # CRITICAL: Ensure we save the complete content, not just the cleaned version.
        # The raw_content preserves everything; content is the user-visible version.
        ai_message.content          = _clean
        ai_message.raw_content      = _accumulated_full or ai_message.content
        ai_message.tokens           = token_count
        ai_message.generation_speed = tok_per_sec
        ai_message.metadata         = message_meta
 
        branch_for_final_handles = self.get_branch(ai_message.id)
        _clean_after_handles, handle_arts = _apply_handles(
            _clean, branch_for_final_handles, self.artefacts
        )
        if _clean_after_handles != _clean:
            ai_message.content = _clean_after_handles
 
        cleaned_content, affected_pp = self._post_process_llm_response(
            _clean_after_handles, ai_message, _eff_img_gen, _eff_img_edit,
            auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets,
            enable_notes=enable_notes,
            enable_skills=enable_skills,
            enable_forms=enable_forms,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation,
        )
        affected_artefacts = handle_arts + affected_pp
        if cleaned_content != _clean_after_handles:
            ai_message.content = cleaned_content
 
        if affected_artefacts:
            message_meta["artefacts_modified"] = [a.get("title") for a in affected_artefacts]
            ai_message.metadata = message_meta
            if callback:
                _cb(callback, json.dumps(message_meta["artefacts_modified"]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                    {"artefacts": affected_artefacts})
 
        _mem_cleaned, _mem_report = self._process_memory_tags(
            ai_message.content, _mm, callback
        )
        if _mem_cleaned != ai_message.content:
            ai_message.content = _mem_cleaned

        if _mm:
            self._save_episodic_memory_turn(user_message, ai_message.content, _mm)

        if self._is_db_backed and self.autosave:
            self.commit()

        object.__setattr__(self, '_active_callback', None)

        return {
            "user_message":     user_msg,
            "ai_message":       ai_message,
            "sources":          collected_sources,
            "scratchpad":       scratchpad_state if is_agentic_turn else None,
            "self_corrections": self_corrections or None,
            "artefacts":        affected_artefacts,
            "memory_report":    _mem_report,
        }

# ---------------------------------------------------------------------------
# Handle resolution — called after generation completes
# ---------------------------------------------------------------------------

def _apply_handles(
    text: str,
    branch_messages: List,
    artefacts_manager: Any,
) -> tuple:
    from ._artefacts import ArtefactType

    handle_pattern = re.compile(
        r'<use_handle\s+([^/]*)/>', re.DOTALL | re.IGNORECASE
    )
    affected: List[Dict] = []
    cleaned  = text

    def _parse_attrs_local(attr_str: str) -> Dict[str, str]:
        return {m.group(1): m.group(2)
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

    def _handle_match(match: re.Match) -> str:
        attrs = _parse_attrs_local(match.group(1))

        ref      = attrs.get("ref", "").strip()
        name     = attrs.get("name", "").strip()
        atype    = attrs.get("type",     ArtefactType.CODE)
        language = attrs.get("language", "").strip()

        if not ref or not name:
            ASCIIColors.warning(
                f"<use_handle> missing ref or name attribute: {match.group(0)}")
            return match.group(0)

        block = _resolve_handle(ref, branch_messages)
        if block is None:
            ASCIIColors.warning(
                f"<use_handle ref='{ref}'> — handle not found in branch; "
                "check msg_idx and block_idx.")
            return f"[handle {ref} not found]"

        content  = block["content"]
        eff_lang = language or block.get("language") or ""
        eff_type = atype if atype in ArtefactType.ALL else ArtefactType.CODE

        existing = artefacts_manager.get(name)
        if existing is None:
            art = artefacts_manager.add(
                title=name, artefact_type=eff_type,
                content=content, language=eff_lang or None, active=True,
            )
            ASCIIColors.success(
                f"[use_handle] Created artefact '{name}' from handle {ref}")
        else:
            art = artefacts_manager.update(
                title=name, new_content=content,
                language=eff_lang or None, bump_version=True, active=True,
            )
            ASCIIColors.success(
                f"[use_handle] Updated artefact '{name}' from handle {ref} "
                f"→ v{art.get('version','?')}")

        affected.append(art)
        return ""

    cleaned = handle_pattern.sub(_handle_match, cleaned)
    return cleaned.strip(), affected


