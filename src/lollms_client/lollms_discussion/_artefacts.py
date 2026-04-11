# lollms_discussion/_artefacts.py
# ArtefactType constants and ArtefactManager: full typed artefact subsystem.
#
# NOTE ON SPELLING
# ----------------
# The Python API keeps the British spelling "artefact" (ArtefactType, ArtefactManager,
# artefacts.*) for backwards compatibility.
# The XML tag used in LLM prompts and response parsing uses the American spelling
# "artifact" (<artifact>, </artifact>, <revert_artifact />) because it is far more
# common in LLM training corpora and avoids model confusion.
#
# ARTEFACT IMAGE SYSTEM
# ---------------------
# Artefacts can carry images (e.g. pages of a PDF converted to PNG/JPEG).
# Each image is a base64 string stored in artefact["images"][N].
# In the artefact *text content*, images are referenced with anchor tags:
#
#   <artefact_image id="TITLE::N" />
#
# where TITLE is the artefact title and N is the 0-based image index.
# build_artefacts_context_zone() preserves these anchors verbatim in the
# text it returns, and also returns a parallel list of image dicts:
#
#   [{"id": "TITLE::0", "data": "<base64>", "media_type": "image/png"}, ...]
#
# The chat layer collects that list and passes it alongside the messages to the
# LLM so the model receives the actual pixel data keyed to the in-text anchors.
#
# Application side (PDF import example)
# --------------------------------------
#   pages_text = []
#   images_b64 = []
#   for i, page in enumerate(pdf_pages):
#       images_b64.append(page.render_to_base64())
#       pages_text.append(f"## Page {i+1}\n\n<artefact_image id=\"my_doc::{i}\" />\n\n{page.text}")
#   combined_text = "\n\n".join(pages_text)
#   discussion.artefacts.add(
#       title="my_doc", artefact_type=ArtefactType.DOCUMENT,
#       content=combined_text, images=images_b64, active=True
#   )

import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

if TYPE_CHECKING:
    from lollms_discussion import LollmsDiscussion


# ---------------------------------------------------------------------------
# ArtefactType
# ---------------------------------------------------------------------------

class ArtefactType:
    """
    Registry for supported artefact types.

    Standard types: file, search_result, note, skill, code, document, image.
    Use register_custom_type() to add domain-specific types.
    """
    FILE          = "file"
    SEARCH_RESULT = "search_result"
    NOTE          = "note"
    SKILL         = "skill"
    CODE          = "code"
    DOCUMENT      = "document"
    IMAGE         = "image"

    ALL = {FILE, SEARCH_RESULT, NOTE, SKILL, CODE, DOCUMENT, IMAGE}

    LABELS = {
        FILE:          "File",
        SEARCH_RESULT: "Search Result",
        NOTE:          "Note",
        SKILL:         "Skill",
        CODE:          "Code",
        DOCUMENT:      "Document",
        IMAGE:         "Image",
    }

    @classmethod
    def register_custom_type(cls, type_name: str, label: Optional[str] = None):
        name = type_name.lower().strip()
        cls.ALL.add(name)
        if label:
            cls.LABELS[name] = label
        elif name not in cls.LABELS:
            cls.LABELS[name] = name.capitalize()


# ---------------------------------------------------------------------------
# Image ID helpers
# ---------------------------------------------------------------------------

# Separator used in image IDs: "artefact_title::image_index"
_IMAGE_ID_SEP = "::"

def make_image_id(artefact_title: str, index: int) -> str:
    """Return the canonical image ID for artefact_title, image index."""
    return f"{artefact_title}{_IMAGE_ID_SEP}{index}"


def parse_image_id(image_id: str) -> Optional[Tuple[str, int]]:
    """Parse an image ID back to (artefact_title, index), or None on failure."""
    if _IMAGE_ID_SEP not in image_id:
        return None
    title, _, idx_str = image_id.rpartition(_IMAGE_ID_SEP)
    try:
        return title, int(idx_str)
    except ValueError:
        return None


# Regex matching <artefact_image id="..." /> (self-closing, optional space before /)
_ARTEFACT_IMAGE_TAG_RE = re.compile(
    r'<artefact_image\s+id=["\']([^"\']+)["\'](?:\s*/?>|>)',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Levenshtein-1 heuristic helper (no external deps)
# ---------------------------------------------------------------------------

def _find_closest_line(needle: str, haystack: str) -> str:
    needle_s = needle.strip()
    if not needle_s:
        return ''
    best_line  = ''
    best_score = -1.0
    for line in haystack.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        common = sum(a == b for a, b in zip(needle_s, line_s))
        score  = common / max(len(needle_s), len(line_s))
        if score > best_score:
            best_score = score
            best_line  = line
    return best_line


# ---------------------------------------------------------------------------
# Fuzzy title matching
# ---------------------------------------------------------------------------

def _title_similarity(a: str, b: str) -> float:
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'\.[a-z0-9]{1,6}$', '', s)
        s = re.sub(r'[\s\-_/\\]+', ' ', s)
        return s

    na, nb = _norm(a), _norm(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 0.85

    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)}

    bg_a, bg_b = bigrams(na), bigrams(nb)
    if not bg_a or not bg_b:
        common = sum(x == y for x, y in zip(na, nb))
        return common / max(len(na), len(nb))

    overlap = len(bg_a & bg_b)
    return (2.0 * overlap) / (len(bg_a) + len(bg_b))


def _find_best_title_match(
    candidate: str,
    existing_titles: List[str],
    threshold: float = 0.60,
) -> Optional[str]:
    best_title: Optional[str] = None
    best_score: float         = -1.0
    for title in existing_titles:
        score = _title_similarity(candidate, title)
        if score > best_score:
            best_score = score
            best_title = title
    if best_score >= threshold:
        return best_title
    return None


# ---------------------------------------------------------------------------
# ArtefactManager
# ---------------------------------------------------------------------------

class ArtefactManager:
    """
    Manages the lifecycle of typed artefacts inside a LollmsDiscussion.

    Artefact schema (stored in discussion_metadata["_artefacts"]):
    {
        "id":           str   – stable UUID
        "title":        str   – human-readable name / identifier
        "type":         str   – one of ArtefactType.ALL
        "version":      int   – incremented on each update
        "content":      str   – text content; may contain <artefact_image id="TITLE::N" /> anchors
        "images":       list  – list of base64 strings referenced by the anchors above
        "image_media_types": list  – parallel list of MIME types ("image/png", "image/jpeg", …)
                                     Defaults to "image/jpeg" when not supplied.
        "audios":       list  – list of base64 strings  (reserved)
        "videos":       list  – list of base64 strings  (reserved)
        "zip":          str | None  – base64 zip blob   (reserved)
        "language":     str | None  – programming language hint  (CODE type)
        "url":          str | None  – source URL
        "tags":         list  – free-form tags
        "active":       bool  – injected into context when True
        "created_at":   str   – ISO datetime
        "updated_at":   str   – ISO datetime
    }

    IMAGE ANCHOR CONVENTION
    -----------------------
    When an artefact holds both text and images (e.g. a PDF converted to
    text+images), the text content may contain self-closing tags:

        <artefact_image id="my_doc::0" />
        <artefact_image id="my_doc::1" />

    The id attribute is  "<artefact_title>::<0-based-image-index>".

    build_artefacts_context_zone() returns the text verbatim (anchors
    preserved).  get_context_images() returns the list of images that
    are referenced by active artefacts, formatted for the LLM API.
    """

    def __init__(self, discussion: 'LollmsDiscussion'):
        object.__setattr__(self, '_discussion', discussion)

    # --------------------------------------------------------- internal helpers

    def _get_all_raw(self) -> List[Dict]:
        metadata = self._discussion.metadata or {}
        raw = metadata.get("_artefacts", [])
        now = datetime.utcnow().isoformat()
        dirty = False
        migrated = []
        for a in raw:
            fixed = a.copy()
            if "id"               not in fixed: fixed["id"]               = str(uuid.uuid4()); dirty = True
            if "type"             not in fixed: fixed["type"]             = ArtefactType.DOCUMENT; dirty = True
            if "title"            not in fixed: fixed["title"]            = "untitled"; dirty = True
            if "content"          not in fixed: fixed["content"]          = ""; dirty = True
            if "images"           not in fixed: fixed["images"]           = []; dirty = True
            if "image_media_types" not in fixed: fixed["image_media_types"] = []; dirty = True
            if "audios"           not in fixed: fixed["audios"]           = []; dirty = True
            if "videos"           not in fixed: fixed["videos"]           = []; dirty = True
            if "zip"              not in fixed: fixed["zip"]              = None; dirty = True
            if "language"         not in fixed: fixed["language"]         = None; dirty = True
            if "url"              not in fixed: fixed["url"]              = None; dirty = True
            if "tags"             not in fixed: fixed["tags"]             = []; dirty = True
            if "active"           not in fixed: fixed["active"]           = False; dirty = True
            if "version"          not in fixed: fixed["version"]          = 1; dirty = True
            if "created_at"       not in fixed: fixed["created_at"]       = now; dirty = True
            if "updated_at"       not in fixed: fixed["updated_at"]       = now; dirty = True
            migrated.append(fixed)
        if dirty:
            new_meta = (self._discussion.metadata or {}).copy()
            new_meta["_artefacts"] = migrated
            self._discussion.metadata = new_meta
            self._discussion.commit()
        return migrated

    def _save_all(self, artefacts: List[Dict]):
        new_meta = (self._discussion.metadata or {}).copy()
        new_meta["_artefacts"] = artefacts
        self._discussion.metadata = new_meta
        self._discussion.commit()

    def _all_latest_titles(self) -> List[str]:
        seen: Dict[str, int] = {}
        for a in self._get_all_raw():
            t, v = a.get('title', ''), a.get('version', 1)
            if t not in seen or v > seen[t]:
                seen[t] = v
        return list(seen.keys())

    # ----------------------------------------------------------------- CRUD

    def add(
        self,
        title:             str,
        artefact_type:     str = ArtefactType.DOCUMENT,
        content:           str = "",
        images:            Optional[List[str]] = None,
        image_media_types: Optional[List[str]] = None,
        audios:            Optional[List[str]] = None,
        videos:            Optional[List[str]] = None,
        zip_content:       Optional[str] = None,
        language:          Optional[str] = None,
        url:               Optional[str] = None,
        tags:              Optional[List[str]] = None,
        version:           int = 1,
        active:            bool = True,
        **extra_data
    ) -> Dict[str, Any]:
        if artefact_type not in ArtefactType.ALL:
            raise ValueError(
                f"Unknown artefact type '{artefact_type}'. "
                f"Use one of: {sorted(ArtefactType.ALL)}"
            )
        artefacts = self._get_all_raw()
        artefacts = [a for a in artefacts
                     if not (a.get('title') == title and a.get('version') == version)]
        for a in artefacts:
            if a.get('title') == title:
                a['active'] = False

        imgs  = images or []
        mtypes = image_media_types or []
        # Pad / extend media_types to match image count
        if len(mtypes) < len(imgs):
            mtypes = mtypes + ["image/jpeg"] * (len(imgs) - len(mtypes))

        now = datetime.utcnow().isoformat()
        new_artefact: Dict[str, Any] = {
            "id":               str(uuid.uuid4()),
            "title":            title,
            "type":             artefact_type,
            "version":          version,
            "content":          content,
            "images":           imgs,
            "image_media_types": mtypes,
            "audios":           audios or [],
            "videos":           videos or [],
            "zip":              zip_content,
            "language":         language,
            "url":              url,
            "tags":             tags or [],
            "active":           active,
            "created_at":       now,
            "updated_at":       now,
            **extra_data,
        }
        artefacts.append(new_artefact)
        self._save_all(artefacts)
        return new_artefact

    def get(self, title: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        candidates = [a for a in self._get_all_raw() if a.get('title') == title]
        if not candidates:
            return None
        if version is not None:
            return next((a for a in candidates if a.get('version') == version), None)
        return max(candidates, key=lambda a: a.get('version', 0))

    def get_by_id(self, artefact_id: str) -> Optional[Dict[str, Any]]:
        return next((a for a in self._get_all_raw() if a.get('id') == artefact_id), None)

    def list(
        self,
        artefact_type: Optional[str] = None,
        active_only:   bool = False,
        tags:          Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        result = self._get_all_raw()
        if artefact_type:
            result = [a for a in result if a.get('type') == artefact_type]
        if active_only:
            result = [a for a in result if a.get('active', False)]
        if tags:
            tag_set = set(tags)
            result = [a for a in result if tag_set.issubset(set(a.get('tags', [])))]
        return result

    def update(
        self,
        title:             str,
        new_content:       Optional[str] = None,
        new_type:          Optional[str] = None,
        new_images:        Optional[List[str]] = None,
        new_image_media_types: Optional[List[str]] = None,
        new_tags:          Optional[List[str]] = None,
        language:          Optional[str] = None,
        url:               Optional[str] = None,
        new_title:         Optional[str] = None,
        bump_version:      bool = True,
        active:            Optional[bool] = None,
        **extra_data
    ) -> Dict[str, Any]:
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Cannot update non-existent artefact '{title}'.")

        if new_type is None and "artefact_type" in extra_data:
            new_type = extra_data.pop("artefact_type")
        else:
            extra_data.pop("artefact_type", None)

        new_version  = (latest.get('version', 1) + 1) if bump_version else latest.get('version', 1)
        new_active   = active if active is not None else latest.get('active', True)
        target_title = new_title if new_title else title

        internal_keys = {
            "id", "title", "type", "version", "content", "images", "image_media_types",
            "audios", "videos", "zip", "language", "url", "tags", "active",
            "created_at", "updated_at", "artefact_type"
        }
        merged_extra = {k: v for k, v in latest.items() if k not in internal_keys}
        merged_extra.update(extra_data)

        if new_title and new_title != title:
            artefacts = self._get_all_raw()
            for a in artefacts:
                if a.get('title') == title:
                    a['active'] = False
            self._save_all(artefacts)
            ASCIIColors.info(f"Renaming artefact '{title}' → '{new_title}'")

        # When renaming, rewrite any image-anchor IDs in the content
        use_content = new_content if new_content is not None else latest.get('content', '')
        if new_title and new_title != title:
            use_content = use_content.replace(
                f'id="{title}{_IMAGE_ID_SEP}',
                f'id="{new_title}{_IMAGE_ID_SEP}'
            ).replace(
                f"id='{title}{_IMAGE_ID_SEP}",
                f"id='{new_title}{_IMAGE_ID_SEP}"
            )

        use_images = new_images if new_images is not None else latest.get('images', [])
        use_mtypes = (
            new_image_media_types if new_image_media_types is not None
            else latest.get('image_media_types', [])
        )

        return self.add(
            title             = target_title,
            artefact_type     = new_type if new_type is not None else latest.get('type', ArtefactType.DOCUMENT),
            content           = use_content,
            images            = use_images,
            image_media_types = use_mtypes,
            audios            = latest.get('audios', []),
            videos            = latest.get('videos', []),
            zip_content       = latest.get('zip'),
            language          = language if language is not None else latest.get('language'),
            url               = url if url is not None else latest.get('url'),
            tags              = new_tags if new_tags is not None else latest.get('tags', []),
            version           = new_version,
            active            = new_active,
            **merged_extra,
        )

    def revert(self, title: str, target_version: int) -> Dict[str, Any]:
        target = self.get(title, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} of artefact '{title}' not found.")
        latest = self.get(title)
        new_version = (latest.get('version', 1) + 1) if latest else 1
        extra_keys = {k: v for k, v in target.items() if k not in [
            "id", "title", "type", "version", "content", "images", "image_media_types",
            "audios", "videos", "zip", "language", "url", "tags", "active",
            "created_at", "updated_at"
        ]}
        ASCIIColors.info(f"Reverting '{title}' to version {target_version} (now saved as v{new_version})")
        return self.add(
            title             = title,
            artefact_type     = target.get('type', ArtefactType.DOCUMENT),
            content           = target.get('content', ''),
            images            = target.get('images', []),
            image_media_types = target.get('image_media_types', []),
            audios            = target.get('audios', []),
            videos            = target.get('videos', []),
            zip_content       = target.get('zip'),
            language          = target.get('language'),
            url               = target.get('url'),
            tags              = target.get('tags', []),
            version           = new_version,
            active            = True,
            **extra_keys,
        )

    def remove(self, title: str, version: Optional[int] = None) -> int:
        artefacts = self._get_all_raw()
        initial = len(artefacts)
        if version is None:
            artefacts = [a for a in artefacts if a.get('title') != title]
        else:
            artefacts = [a for a in artefacts
                         if not (a.get('title') == title and a.get('version') == version)]
        removed = initial - len(artefacts)
        if removed:
            self._save_all(artefacts)
            ASCIIColors.info(f"Removed {removed} artefact(s) titled '{title}'.")
        return removed

    def remove_by_id(self, artefact_id: str) -> bool:
        artefacts = self._get_all_raw()
        new_list = [a for a in artefacts if a.get('id') != artefact_id]
        if len(new_list) < len(artefacts):
            self._save_all(new_list)
            return True
        return False

    # --------------------------------------------------------- activation

    def activate(self, title: str, version: Optional[int] = None):
        self._set_active(title, version, True)

    def deactivate(self, title: str, version: Optional[int] = None):
        self._set_active(title, version, False)

    def toggle(self, title: str, version: Optional[int] = None) -> bool:
        artefact = self.get(title, version)
        if artefact is None:
            raise ValueError(f"Artefact '{title}' not found.")
        new_state = not artefact.get('active', False)
        self._set_active(title, version or artefact['version'], new_state)
        return new_state

    def _set_active(self, title: str, version: Optional[int], state: bool):
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title and (version is None or a.get('version') == version):
                a['active'] = state
                changed = True
        if changed:
            self._save_all(artefacts)

    # --------------------------------------------------------- context zone

    def build_artefacts_context_zone(self) -> str:
        """
        Assembles a text block describing all active artefacts.

        <artefact_image id="TITLE::N" /> anchors in artefact content are
        preserved verbatim so the LLM can correlate text with images.
        The companion image list is obtained via get_context_images().
        """
        active = [a for a in self._get_all_raw() if a.get('active', False)]
        with_content = [a for a in active if a.get('content', '').strip() or a.get('url')]
        if not with_content:
            return ""
        parts = []
        for item in with_content:
            atype    = item.get('type', ArtefactType.DOCUMENT)
            lang     = item.get('language') or ''
            fence    = f"```{lang}" if lang else "```"
            url_line = f"\nSource: {item['url']}" if item.get('url') else ""
            versions = [a.get('version', 1) for a in self._get_all_raw() if a.get('title') == item['title']]
            total_versions = len(versions)
            version_str = f"v{item['version']}"
            if total_versions > 1:
                version_str += f" | {total_versions} total versions exist"
            meta_str = ""
            if item.get('author'):      meta_str += f" | Author: {item['author']}"
            if item.get('description'): meta_str += f"\nDescription: {item['description']}"
            label  = ArtefactType.LABELS.get(atype, atype.capitalize())
            header = f"###[{label}] {item['title']} ({version_str}){meta_str}{url_line}"

            # Image count note (only when images present)
            img_note = ""
            imgs = item.get('images') or []
            if imgs:
                img_note = (
                    f"\n<!-- This artefact has {len(imgs)} image(s). "
                    f"They are referenced inline via <artefact_image id=\"{item['title']}::N\" /> tags. "
                    f"The actual image data has been appended to the conversation context. -->"
                )

            if item.get('content', '').strip():
                # For code artefacts that have no image anchors, keep the fence.
                # For document artefacts that embed image anchors, use plain text
                # so the anchors are not hidden inside a code block.
                content_text = item['content'].strip()
                if atype == ArtefactType.CODE or (
                    lang and not _ARTEFACT_IMAGE_TAG_RE.search(content_text)
                ):
                    parts.append(f"{header}{img_note}\n{fence}\n{content_text}\n```")
                else:
                    parts.append(f"{header}{img_note}\n{content_text}")
            else:
                parts.append(header + img_note)

        return "## Active Artifacts\n\n" + "\n\n".join(parts)

    def get_context_images(self) -> List[Dict[str, Any]]:
        """
        Returns ALL images from active artefacts that have images.

        Each entry:
            {
                "id":         str  – "<title>::<index>"
                "data":       str  – base64 encoded image data
                "media_type": str  – e.g. "image/jpeg", "image/png"
                "title":      str  – artefact title
                "index":      int  – 0-based index within the artefact
                "active":     bool – True
            }

        The list is ordered: artefacts in activation order, images in index order.
        The chat layer merges these with message-level images before calling the LLM.
        """
        result: List[Dict[str, Any]] = []
        for a in self._get_all_raw():
            if not a.get('active', False):
                continue
            imgs   = a.get('images') or []
            mtypes = a.get('image_media_types') or []
            for idx, img_b64 in enumerate(imgs):
                if not img_b64:
                    continue
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                result.append({
                    "id":         make_image_id(a['title'], idx),
                    "data":       img_b64,
                    "media_type": mtype,
                    "title":      a['title'],
                    "index":      idx,
                    "active":     True,
                })
        return result

    def get_active_images(self) -> List[Dict[str, Any]]:
        """
        Legacy / UI helper: returns images from active IMAGE-type artefacts only.
        For full artefact image context, use get_context_images().
        """
        result = []
        for a in self._get_all_raw():
            if a.get('active', False) and a.get('type') == ArtefactType.IMAGE:
                source_id = f"artifact:{a['title']} v{a['version']}"
                for img_b64 in (a.get('images') or []):
                    result.append({
                        "data":       img_b64,
                        "source":     source_id,
                        "active":     True,
                        "created_at": a.get('created_at', ''),
                    })
        return result

    # --------------------------------------------------- aider-style patching

    @staticmethod
    def apply_aider_patch(original: str, patch_block: str) -> str:
        """
        Applies one or more aider SEARCH/REPLACE blocks to *original* text.
        (Full docstring preserved from original — see source.)
        """
        import re as _re

        SEARCH_RE  = _re.compile(r'^<{6,8}\s*SEARCH\s*$',  _re.IGNORECASE)
        SEP_RE     = _re.compile(r'^={5,}\s*$')
        REPLACE_RE = _re.compile(r'^>{6,8}\s*REPLACE\s*$', _re.IGNORECASE)

        patch_block = patch_block.replace('\r\n', '\n').replace('\r', '\n')

        lines = patch_block.split('\n')
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and not REPLACE_RE.match(lines[-1].rstrip()):
            lines.append('>>>>>>> REPLACE')
        patch_block = '\n'.join(lines)

        raw_lines = patch_block.split('\n')
        segments: List[Tuple[str, str]] = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i].rstrip()
            if SEARCH_RE.match(line):
                i += 1
                search_lines: List[str] = []
                while i < len(raw_lines):
                    l = raw_lines[i].rstrip()
                    if SEP_RE.match(l) or REPLACE_RE.match(l):
                        break
                    search_lines.append(raw_lines[i])
                    i += 1

                if i >= len(raw_lines):
                    raise ValueError(
                        "Malformed aider patch: SEARCH block has no ======= separator.\n"
                        f"Search text was:\n{''.join(search_lines)}"
                    )

                separator_line = raw_lines[i].rstrip()

                if REPLACE_RE.match(separator_line):
                    segments.append(('\n'.join(search_lines), ''))
                    i += 1
                    continue

                i += 1
                replace_lines: List[str] = []
                while i < len(raw_lines):
                    l = raw_lines[i].rstrip()
                    if REPLACE_RE.match(l):
                        break
                    replace_lines.append(raw_lines[i])
                    i += 1
                if i < len(raw_lines) and REPLACE_RE.match(raw_lines[i].rstrip()):
                    i += 1

                segments.append(('\n'.join(search_lines), '\n'.join(replace_lines)))
            else:
                i += 1

        if not segments:
            raise ValueError(
                "Malformed aider patch: no valid <<<<<<< SEARCH … >>>>>>> REPLACE "
                "block found.\nPatch preview:\n" + patch_block[:500]
            )

        result = original
        for search_text, replace_text in segments:
            if search_text.startswith('\n'):
                search_text = search_text[1:]
            if replace_text.startswith('\n'):
                replace_text = replace_text[1:]
            if search_text.endswith('\n'):
                search_text = search_text[:-1]
            if replace_text.endswith('\n'):
                replace_text = replace_text[:-1]

            if search_text not in result:
                first_line = search_text.split('\n')[0]
                hint = _find_closest_line(first_line, result)
                raise ValueError(
                    f"SEARCH text not found in artefact content.\n"
                    f"Expected first line : {first_line!r}\n"
                    f"Closest line found  : {hint!r}\n"
                    f"Tip: check indentation, trailing spaces, and line endings."
                )
            result = result.replace(search_text, replace_text, 1)

        return result

    # -------------------------------------------- LLM artifact XML parser

    def _apply_artefact_xml(
        self,
        text:          str,
        default_type:  str = ArtefactType.CODE,
        auto_activate: bool = True,
        replacements:  Optional[Dict[str, str]] = None,
        event_callback: Optional[Any] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scans *text* for ``<artifact …>…</artifact>`` blocks (American spelling)
        and also accepts the legacy ``<artefact …>…</artefact>`` spelling.

        NOTE: The LLM cannot create artefact images via XML — images are supplied
        by the application layer when calling add() / update() directly.
        The parser ignores any image-related attributes it encounters.
        """
        affected: List[Dict] = []
        cleaned  = text

        artefact_pattern = re.compile(
            r'<art[ei]fact\s([^>]*)>(.*?)</art[ei]fact>',
            re.DOTALL | re.IGNORECASE
        )

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        existing_titles = self._all_latest_titles()

        def handle_artefact(match: re.Match) -> str:
            nonlocal existing_titles

            attrs   = _parse_attrs(match.group(1))
            content = match.group(2)

            if replacements:
                for placeholder, original in replacements.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

            tag_title   = attrs.pop('name', attrs.pop('title', f'artifact_{uuid.uuid4().hex[:8]}'))
            new_name    = attrs.pop('rename', None)
            atype       = attrs.pop('type', default_type)
            language    = attrs.pop('language', None)
            version_str = attrs.pop('version', '1')
            version     = int(version_str) if version_str.isdigit() else 1
            # Strip any image-related attrs the LLM might have hallucinated
            attrs.pop('images', None)
            attrs.pop('image_media_types', None)

            if atype not in ArtefactType.ALL:
                atype = default_type

            resolved_title: Optional[str] = None
            is_new = False

            if tag_title in existing_titles:
                resolved_title = tag_title
            else:
                fuzzy = _find_best_title_match(tag_title, existing_titles)
                if fuzzy:
                    ASCIIColors.info(
                        f"Fuzzy title match: '{tag_title}' → '{fuzzy}' (updating in place)"
                    )
                    resolved_title = fuzzy
                else:
                    resolved_title = tag_title
                    is_new = True

            _has_search = bool(
                re.search(r'<{6,8}\s*SEARCH', content, re.IGNORECASE)
            )

            result_artefact: Optional[Dict] = None

            if _has_search:
                existing = self.get(resolved_title)
                if existing is None:
                    ASCIIColors.warning(
                        f"Aider patch for '{resolved_title}' but artefact does not exist yet — "
                        "saving raw patch content as initial version."
                    )
                    result_artefact = self.add(
                        title=resolved_title, artefact_type=atype, content=content,
                        language=language, version=version, active=auto_activate,
                        **attrs
                    )
                    is_new = True
                else:
                    try:
                        patched = ArtefactManager.apply_aider_patch(
                            existing.get('content', ''), content
                        )
                        result_artefact = self.update(
                            title=resolved_title,
                            new_content=patched,
                            new_title=new_name,
                            language=language,
                            bump_version=True,
                            active=auto_activate,
                            **attrs
                        )
                        final_title = result_artefact.get('title', resolved_title)
                        ASCIIColors.success(
                            f"Aider patch applied to '{resolved_title}'"
                            + (f" → renamed '{final_title}'" if new_name else "")
                            + f" → v{result_artefact.get('version', '?')}"
                        )
                    except ValueError as e:
                        ASCIIColors.warning(
                            f"⚠ Aider patch FAILED for '{resolved_title}':\n  {e}\n"
                            "  Existing artefact is unchanged."
                        )
                        result_artefact = existing
            else:
                if is_new:
                    result_artefact = self.add(
                        title=resolved_title, artefact_type=atype,
                        content=content.strip(),
                        language=language, version=version, active=auto_activate,
                        **attrs
                    )
                else:
                    result_artefact = self.update(
                        title=resolved_title,
                        new_content=content.strip(),
                        new_title=new_name,
                        new_type=atype,
                        language=language,
                        bump_version=True,
                        active=auto_activate,
                        **attrs
                    )
                    final_title = result_artefact.get('title', resolved_title)
                    if new_name:
                        ASCIIColors.info(
                            f"Artefact '{resolved_title}' renamed to '{final_title}'")

            existing_titles = self._all_latest_titles()

            affected.append(result_artefact)

            if event_callback and result_artefact:
                try:
                    event_callback(result_artefact, is_new)
                except Exception as _ecb_err:
                    ASCIIColors.warning(f"Artefact event callback error: {_ecb_err}")

            return ''

        cleaned = artefact_pattern.sub(handle_artefact, cleaned)

        revert_pattern = re.compile(r'<revert_art[ei]fact\s([^>]+)/?>', re.IGNORECASE)

        def handle_revert(match: re.Match) -> str:
            attrs   = _parse_attrs(match.group(1))
            title   = attrs.get('name') or attrs.get('title')
            version = attrs.get('version')
            if title and version and version.isdigit():
                resolved = title if title in existing_titles else (
                    _find_best_title_match(title, existing_titles) or title
                )
                try:
                    res_artefact = self.revert(resolved, int(version))
                    affected.append(res_artefact)
                    if event_callback:
                        try:
                            event_callback(res_artefact, False)
                        except Exception:
                            pass
                except ValueError as e:
                    ASCIIColors.warning(str(e))
            return ''

        cleaned = revert_pattern.sub(handle_revert, cleaned)

        return cleaned.strip(), affected

    # ------------------------------------------------- legacy compat shims

    def add_artefact(self, title, content="", images=None, audios=None, videos=None,
                     zip_content=None, version=1, **extra_data) -> Dict:
        atype = extra_data.pop('artefact_type', ArtefactType.DOCUMENT)
        return self.add(
            title=title, artefact_type=atype,
            content=content, images=images, audios=audios, videos=videos,
            zip_content=zip_content, version=version, **extra_data
        )

    def get_artefact(self, title, version=None) -> Optional[Dict]:
        return self.get(title, version)

    def update_artefact(self, title, new_content, new_images=None, **extra_data) -> Dict:
        return self.update(title=title, new_content=new_content,
                           new_images=new_images, **extra_data)

    def list_artefacts(self) -> List[Dict]:
        items = self.list()
        disc  = self._discussion
        for a in items:
            section_start  = f"--- Document: {a['title']} v{a['version']} ---"
            content_loaded = section_start in (disc.discussion_data_zone or "")
            source_id      = f"artifact:{a['title']} v{a['version']}"
            image_loaded   = any(
                img.get("source") == source_id for img in disc.get_discussion_images()
            )
            a["is_loaded"] = content_loaded or image_loaded
        return items

    def remove_artefact(self, title, version=None) -> int:
        return self.remove(title, version)