
import json
import re
import uuid
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

# ── Web content extraction & quality scoring ─────────────────────────────

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


def _infer_sources_from_json(result_dict: Dict[str, Any], tool_name: str) -> List[Dict[str, Any]]:
    """Infers and extracts structured sources from a tool result dictionary."""
    sources = []
    if not isinstance(result_dict, dict):
        return sources

    raw_sources = result_dict.get("sources", [])
    if isinstance(raw_sources, list):
        for s in raw_sources:
            if isinstance(s, dict):
                s.setdefault("title", s.get("name") or s.get("title") or f"{tool_name} Source")
                s.setdefault("content", s.get("content") or s.get("snippet") or "")
                s.setdefault("source", s.get("source") or s.get("url") or "")
                s.setdefault("metadata", s.get("metadata") or {})
                sources.append(s)

    output = result_dict.get("output")
    if isinstance(output, dict):
        for key in ["results", "sources", "items"]:
            val = output.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        sources.append({
                            "title": item.get("title") or item.get("name") or f"{tool_name} Result",
                            "content": item.get("content") or item.get("snippet") or item.get("text") or "",
                            "source": item.get("source") or item.get("url") or item.get("link") or "",
                            "metadata": item.get("metadata") or {}
                        })
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                sources.append({
                    "title": item.get("title") or item.get("name") or f"{tool_name} Result",
                    "content": item.get("content") or item.get("snippet") or item.get("text") or "",
                    "source": item.get("source") or item.get("url") or item.get("link") or "",
                    "metadata": item.get("metadata") or {}
                })
    return sources


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

