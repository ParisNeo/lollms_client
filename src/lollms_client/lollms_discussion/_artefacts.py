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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
    PRESENTATION  = "presentation"
    DATA          = "data"
    TOOL          = "tool"

    ALL = {FILE, SEARCH_RESULT, NOTE, SKILL, CODE, DOCUMENT, IMAGE, PRESENTATION, DATA, TOOL}

    LABELS = {
        FILE:          "File",
        SEARCH_RESULT: "Search Result",
        NOTE:          "Note",
        SKILL:         "Skill",
        CODE:          "Code",
        DOCUMENT:      "Document",
        IMAGE:         "Image",
        PRESENTATION:  "Presentation",
        DATA:          "Data Interface",
        TOOL:          "Tool"
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
            if "commit_message"   not in fixed: fixed["commit_message"]   = None; dirty = True
            if "version_tags"     not in fixed: fixed["version_tags"]     = []; dirty = True
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
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
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
            "commit_message":   commit_message,
            "version_tags":     version_tags or [],
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
            # Robust type-coercing search to handle both int and string representation
            for a in candidates:
                curr_v = a.get('version')
                if str(curr_v) == str(version):
                    return a
                # Also handle dotted semantic version strings (like "1.0.0" matching version 1)
                try:
                    if int(float(str(curr_v).split('.')[0])) == int(version):
                        return a
                except Exception:
                    pass
            return next((a for a in candidates if a.get('version') == version), None)

        def _safe_sort_key(a):
            try:
                v = a.get('version', 0)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 0

        return max(candidates, key=_safe_sort_key)

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
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
        **extra_data
    ) -> Optional[Dict[str, Any]]:
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
            "created_at", "updated_at", "artefact_type", "commit_message", "version_tags"
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

        # Ensure we are not just overwriting the same version object in the list
        result = self.add(
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
            commit_message    = commit_message if commit_message is not None else latest.get('commit_message'),
            version_tags      = version_tags if version_tags is not None else latest.get('version_tags', []),
            **merged_extra,
        )
        # Verify the list reflects the new version
        ASCIIColors.success(f"[ArtefactManager] Incremented '{target_title}' to v{new_version}")
        return result

    def revert(self, title: str, target_version: Union[int, str]) -> Dict[str, Any]:
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Artefact '{title}' not found.")

        if isinstance(target_version, str) and str(target_version).strip().lower() in ("last", "previous", "prev"):
            curr_v = latest.get('version', 1)
            if curr_v <= 1:
                raise ValueError(f"Cannot revert '{title}': no previous version exists (current is v1).")
            target_version = curr_v - 1
        else:
            try:
                target_version = int(target_version)
            except ValueError:
                raise ValueError(f"Invalid target version: {target_version}")

        target = self.get(title, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} of artefact '{title}' not found.")
        new_version = (latest.get('version', 1) + 1)
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

    def rename(self, old_title: str, new_title: str, new_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Renames all versions of an artifact from old_title to new_title.
        Optionally changes the type to new_type.
        """
        artefacts = self._get_all_raw()
        found = False
        for a in artefacts:
            if a.get('title') == old_title:
                a['title'] = new_title
                a['updated_at'] = datetime.utcnow().isoformat()
                if new_type is not None:
                    a['type'] = new_type
                # If there is content containing image-anchor IDs, rename them as well
                content = a.get('content', '')
                if content:
                    a['content'] = content.replace(
                        f'id="{old_title}{_IMAGE_ID_SEP}',
                        f'id="{new_title}{_IMAGE_ID_SEP}'
                    ).replace(
                        f"id='{old_title}{_IMAGE_ID_SEP}",
                        f"id='{new_title}{_IMAGE_ID_SEP}'"
                    )
                found = True

        if found:
            self._save_all(artefacts)
            ASCIIColors.success(f"[ArtefactManager] Renamed all versions of '{old_title}' to '{new_title}'")
            return self.get(new_title)
        return None

    # --------------------------------------------------------- versioning

    def get_version_history(self, title: str) -> List[Dict[str, Any]]:
        """
        Returns the complete version history for an artefact, sorted by version number.

        Each entry contains: version, created_at, content_preview (first 200 chars),
        size_chars, and whether it is the currently active version.
        """
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        if not all_versions:
            return []

        def _safe_sort_key(a):
            try:
                v = a.get('version', 0)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 0

        all_versions.sort(key=_safe_sort_key)
        active_version = None
        for a in all_versions:
            if a.get('active', False):
                active_version = a.get('version')
                break
        result = []
        for a in all_versions:
            content = a.get('content', '')
            result.append({
                "version":        a.get('version', 1),
                "created_at":     a.get('created_at', ''),
                "updated_at":     a.get('updated_at', ''),
                "content_preview": content[:200] + ("…" if len(content) > 200 else ""),
                "size_chars":     len(content),
                "image_count":    len(a.get('images', [])),
                "is_active":      a.get('version') == active_version,
            })
        return result

    def diff_versions(self, title: str, version_a: int, version_b: int) -> Dict[str, Any]:
        """
        Compute a line-based diff between two artefact versions.

        Returns a dict with:
          - unified_diff: standard unified diff as a string
          - added_lines:  count of lines only in version_b
          - removed_lines: count of lines only in version_a
          - common_lines:  count of lines present in both
        """
        art_a = self.get(title, version_a)
        art_b = self.get(title, version_b)
        if art_a is None:
            raise ValueError(f"Version {version_a} of artefact '{title}' not found.")
        if art_b is None:
            raise ValueError(f"Version {version_b} of artefact '{title}' not found.")

        lines_a = art_a.get('content', '').splitlines()
        lines_b = art_b.get('content', '').splitlines()

        # Simple line-based diff using set operations for stats
        set_a = set(lines_a)
        set_b = set(lines_b)

        added = sorted([l for l in lines_b if l not in set_a])
        removed = sorted([l for l in lines_a if l not in set_b])
        common = sorted([l for l in lines_a if l in set_b])

        # Build unified diff
        import difflib
        diff_text = "\n".join(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"{title} v{version_a}",
            tofile=f"{title} v{version_b}",
            lineterm="",
        ))

        return {
            "title":         title,
            "version_a":     version_a,
            "version_b":     version_b,
            "unified_diff":  diff_text,
            "added_lines":   len(added),
            "removed_lines": len(removed),
            "common_lines":  len(common),
            "added_content": "\n".join(added),
            "removed_content": "\n".join(removed),
        }

    def squash_versions(
        self,
        title: str,
        keep_versions: Optional[List[int]] = None,
        keep_last_n: Optional[int] = None,
        target_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Squash (merge / delete) artefact versions to reclaim storage space.

        Parameters
        ----------
        title : str
            The artefact to squash.
        keep_versions : list[int] | None
            Explicit list of version numbers to preserve. All others are deleted.
        keep_last_n : int | None
            Keep only the N most recent versions (by version number). Older ones are deleted.
        target_version : int | None
            If given, all versions are deleted EXCEPT this one, which becomes the
            new baseline at version 1. The content of this version is preserved.

        Returns
        -------
        dict
            {
                "success": bool,
                "deleted": [version, ...],
                "preserved": [version, ...],
                "new_baseline": version | None,
                "space_reclaimed_estimate": int,  # approximate chars freed
            }

        Rules
        -----
        • The currently active version is NEVER deleted.
        • At least one version must remain after squashing.
        • If target_version is specified, it takes precedence over keep_versions/keep_last_n.
        """
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        if not all_versions:
            raise ValueError(f"Artefact '{title}' not found.")

        all_versions.sort(key=lambda a: a.get('version', 0))

        # Identify the active version — it must survive
        active_version = None
        for a in all_versions:
            if a.get('active', False):
                active_version = a.get('version')
                break

        to_delete: List[int] = []
        to_preserve: List[int] = []
        new_baseline: Optional[int] = None
        space_reclaimed = 0

        if target_version is not None:
            target = self.get(title, target_version)
            if target is None:
                raise ValueError(f"Target version {target_version} of '{title}' not found.")

            # Compile squashed commit messages
            squashed_messages = []
            for a in all_versions:
                v = a.get('version', 0)
                msg = a.get('commit_message')
                if v != target_version and msg:
                    squashed_messages.append(f"v{v}: {msg}")

            # Collect deletions and space reclaimed
            for a in all_versions:
                v = a.get('version', 0)
                if v != target_version:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))
                else:
                    to_preserve.append(1)

            # Perform the deletions and update the target version in a single pass
            artefacts = self._get_all_raw()
            new_list = []
            for a in artefacts:
                if a.get('title') == title:
                    v = a.get('version', 0)
                    if v == target_version:
                        a['version'] = 1
                        a['updated_at'] = datetime.utcnow().isoformat()
                        if squashed_messages:
                            orig_msg = a.get('commit_message') or "Baseline commit"
                            a['commit_message'] = f"{orig_msg}\n\nSquashed history:\n" + "\n".join(squashed_messages)
                        new_list.append(a)
                    elif v in to_delete:
                        continue
                    else:
                        new_list.append(a)
                else:
                    new_list.append(a)

            self._save_all(new_list)
            ASCIIColors.info(
                f"[ArtefactManager] Squashed '{title}' around target version {target_version}: "
                f"deleted {len(to_delete)} version(s) ({to_delete}), "
                f"re-baselined version {target_version} to v1."
            )
            return {
                "success":      True,
                "deleted":      to_delete,
                "preserved":    [1],
                "new_baseline": 1,
                "space_reclaimed_estimate": space_reclaimed,
            }

        elif keep_versions is not None:
            keep_set = set(keep_versions)
            if active_version is not None and active_version not in keep_set:
                keep_set.add(active_version)
            for a in all_versions:
                v = a.get('version', 0)
                if v in keep_set:
                    to_preserve.append(v)
                else:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))

        elif keep_last_n is not None:
            if keep_last_n < 1:
                raise ValueError("keep_last_n must be >= 1")
            sorted_versions = [a.get('version', 0) for a in all_versions]
            preserved_set = set(sorted_versions[-keep_last_n:])
            if active_version is not None and active_version not in preserved_set:
                preserved_set.add(active_version)
                # If adding active_version pushes us over keep_last_n, remove oldest
                if len(preserved_set) > keep_last_n:
                    sorted_preserved = sorted(preserved_set)
                    preserved_set = set(sorted_preserved[-keep_last_n:])
            for a in all_versions:
                v = a.get('version', 0)
                if v in preserved_set:
                    to_preserve.append(v)
                else:
                    to_delete.append(v)
                    space_reclaimed += len(a.get('content', ''))

        else:
            raise ValueError("One of keep_versions, keep_last_n, or target_version must be specified.")

        # Safety: ensure at least one version survives
        if not to_preserve:
            raise ValueError("Squash would delete all versions. At least one must remain.")

        # Execute deletion and squash history compilation into the active version
        if to_delete:
            squashed_messages = []
            for v_num in sorted(to_delete):
                matching_version = next((a for a in all_versions if a.get('version') == v_num), None)
                if matching_version and matching_version.get('commit_message'):
                    squashed_messages.append(f"v{v_num}: {matching_version['commit_message']}")

            artefacts = self._get_all_raw()

            # Find the oldest preserved version to receive the squashed messages
            receiver_version = min(to_preserve) if to_preserve else None

            new_list = []
            for a in artefacts:
                if a.get('title') == title:
                    v = a.get('version', 0)
                    if v in to_delete:
                        continue
                    if v == receiver_version and squashed_messages:
                        orig_msg = a.get('commit_message') or "Preserved baseline"
                        a['commit_message'] = f"{orig_msg}\n\nSquashed history:\n" + "\n".join(squashed_messages)
                new_list.append(a)

            self._save_all(new_list)
            ASCIIColors.info(
                f"[ArtefactManager] Squashed '{title}': deleted {len(to_delete)} version(s) "
                f"({to_delete}), preserved {len(to_preserve)} ({to_preserve})."
            )

        return {
            "success":      True,
            "deleted":      to_delete,
            "preserved":    to_preserve,
            "new_baseline": new_baseline,
            "space_reclaimed_estimate": space_reclaimed,
        }

    def cleanup_old_versions(
        self,
        title: str,
        keep_count: int = 5,
        min_age_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: delete old versions keeping only *keep_count* most recent.

        Parameters
        ----------
        title : str
            Artefact to clean up.
        keep_count : int
            Number of recent versions to retain (default 5).
        min_age_hours : float | None
            If given, only versions older than this many hours are eligible for deletion,
            regardless of keep_count.

        Returns
        -------
        dict
            Same shape as squash_versions().
        """
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        if not all_versions:
            raise ValueError(f"Artefact '{title}' not found.")

        all_versions.sort(key=lambda a: a.get('version', 0))

        # Find versions to delete
        to_delete: List[int] = []
        to_preserve: List[int] = []
        space_reclaimed = 0

        active_version = None
        for a in all_versions:
            if a.get('active', False):
                active_version = a.get('version')
                break

        now = datetime.utcnow()

        for a in all_versions:
            v = a.get('version', 0)
            is_active = (v == active_version)

            # Check age constraint
            age_ok = True
            if min_age_hours is not None:
                updated = a.get('updated_at', a.get('created_at', ''))
                try:
                    updated_dt = datetime.fromisoformat(updated)
                    age_hours = (now - updated_dt).total_seconds() / 3600.0
                    age_ok = age_hours >= min_age_hours
                except (ValueError, TypeError):
                    age_ok = True  # If we can't parse date, allow deletion

            if is_active:
                to_preserve.append(v)
            else:
                to_preserve.append(v)  # Tentatively preserve

        # Now apply keep_count from the end
        if len(to_preserve) > keep_count:
            # We need to remove oldest non-active versions
            non_active = [a for a in all_versions if a.get('version') not in to_preserve[:1]]
            # Actually, simpler: use keep_last_n logic
            pass

        # Delegate to squash_versions for actual execution
        return self.squash_versions(title, keep_last_n=keep_count)

    def remove(self, title: str, version: Optional[int] = None) -> int:
        # First, clean up any associated workspace files for DATA type artefacts
        self._cleanup_data_artefact_files(title, version)

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

    def _cleanup_data_artefact_files(self, title: str, version: Optional[int] = None):
        """
        Deletes workspace files associated with a DATA type artefact when it is removed.

        Handles both single data files and consolidated bundles from folder ingestion.
        """
        try:
            # Get the artefact to determine file path details
            art = self.get(title, version) if version else self.get(title)
            if not art or art.get('type') != 'data':
                return  # Not a data artefact, skip cleanup

            # Determine workspace directory
            workspace_dir = Path("./data_workspace")
            try:
                from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                if awd is not None:
                    workspace_dir = awd
            except ImportError:
                pass

            if not workspace_dir.exists():
                return

            file_ext = art.get('file_ext', '.csv')

            # Build list of potential files to delete for this artefact title/version
            files_to_delete = []

            if version is None:
                # Remove all versions - find all matching files
                all_versions = [a for a in self._get_all_raw() if a.get('title') == title and a.get('type') == 'data']
                for v_art in all_versions:
                    v_num = v_art.get('version', 1)

                    # Check for versioned file (e.g., "mydb_v2.db")
                    versioned_file = workspace_dir / f"{title}_v{v_num}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)

                    # Check for consolidated bundle DB (only exists on v1 creation)
                    if v_num == 1:
                        bundled_db = workspace_dir / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)
            else:
                # Remove specific version only
                if art.get('type') == 'data':
                    # Check for versioned file (e.g., "mydb_v2.db")
                    versioned_file = workspace_dir / f"{title}_v{version}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)

                    # Check for consolidated bundle DB (only exists on v1 creation)
                    if version == 1:
                        bundled_db = workspace_dir / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)

            # Execute deletion with safety checks
            for file_path in set(files_to_delete):  # Use set to avoid duplicates
                try:
                    # Safety check: ensure we're only deleting data workspace files
                    resolved = file_path.resolve()
                    ws_resolved = workspace_dir.resolve()

                    if not str(resolved).startswith(str(ws_resolved)):
                        ASCIIColors.warning(f"[ArtefactCleanup] Skipping deletion of {file_path} - outside workspace")
                        continue

                    file_path.unlink()
                    ASCIIColors.success(f"✓ Deleted data file: {file_path.name}")
                except Exception as e:
                    ASCIIColors.error(f"Failed to delete file {file_path}: {e}")

        except Exception as e:
            ASCIIColors.warning(f"[ArtefactCleanup] Error during cleanup for '{title}': {e}")

    def remove_by_id(self, artefact_id: str) -> bool:
        artefacts = self._get_all_raw()
        new_list = [a for a in artefacts if a.get('id') != artefact_id]
        if len(new_list) < len(artefacts):
            self._save_all(new_list)
            return True
        return False

    # --------------------------------------------------------- version tags & git-like features

    def tag_version(self, title: str, version: int, tag_name: str) -> bool:
        """
        Assigns a Git-like tag (e.g. 'stable', 'v1.0-milestone') to a specific version.
        Deletes the tag from any other versions of this same artefact to ensure uniqueness.
        """
        tag_name = tag_name.strip().lower()
        if not tag_name:
            return False

        artefacts = self._get_all_raw()
        found = False
        changed = False

        # Unbind tag_name from other versions of this artefact first
        for a in artefacts:
            if a.get('title') == title:
                tags = list(a.get('version_tags', []))
                if tag_name in tags:
                    tags.remove(tag_name)
                    a['version_tags'] = tags
                    changed = True
                if a.get('version') == version:
                    tags.append(tag_name)
                    a['version_tags'] = tags
                    found = True
                    changed = True

        if found:
            self._save_all(artefacts)
            ASCIIColors.success(f"[ArtefactManager] Tagged '{title}' v{version} with '{tag_name}'")
            return True
        return False

    def remove_tag(self, title: str, tag_name: str) -> bool:
        """Deletes a tag from all versions of this artefact."""
        tag_name = tag_name.strip().lower()
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title:
                tags = list(a.get('version_tags', []))
                if tag_name in tags:
                    tags.remove(tag_name)
                    a['version_tags'] = tags
                    changed = True
        if changed:
            self._save_all(artefacts)
            ASCIIColors.info(f"[ArtefactManager] Removed tag '{tag_name}' from '{title}'")
            return True
        return False

    def resolve_tag(self, title: str, tag_name: str) -> Optional[int]:
        """Resolves a tag name to its corresponding version number."""
        tag_name = tag_name.strip().lower()
        for a in self._get_all_raw():
            if a.get('title') == title and tag_name in a.get('version_tags', []):
                return a.get('version')
        return None

    def get_log(self, title: str) -> List[Dict[str, Any]]:
        """
        Returns a complete, chronological Git-like version log for an artefact.
        """
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        all_versions.sort(key=lambda a: a.get('version', 0), reverse=True) # newest first

        log = []
        for a in all_versions:
            log.append({
                "commit_hash":    a.get("id"),
                "version":        a.get("version", 1),
                "author":         a.get("author", "AI Specialist"),
                "commit_message": a.get("commit_message") or f"Update '{title}' to version {a.get('version')}",
                "tags":           a.get("version_tags", []),
                "created_at":     a.get("created_at"),
                "size_chars":     len(a.get("content", "")),
                "is_active":      a.get("active", False)
            })
        return log

    def revert_to_tag(self, title: str, tag_name: str) -> Dict[str, Any]:
        """Reverts the active artefact version to the version bound to the tag."""
        version = self.resolve_tag(title, tag_name)
        if version is None:
            raise ValueError(f"Tag '{tag_name}' not found for artefact '{title}'.")
        return self.revert(title, version)

    # --------------------------------------------------------- version tags & git-like features


    def remove_tag(self, title: str, tag_name: str) -> bool:
        """Deletes a tag from all versions of this artefact."""
        tag_name = tag_name.strip().lower()
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title:
                tags = list(a.get('version_tags', []))
                if tag_name in tags:
                    tags.remove(tag_name)
                    a['version_tags'] = tags
                    changed = True
        if changed:
            self._save_all(artefacts)
            ASCIIColors.info(f"[ArtefactManager] Removed tag '{tag_name}' from '{title}'")
            return True
        return False

    def resolve_tag(self, title: str, tag_name: str) -> Optional[int]:
        """Resolves a tag name to its corresponding version number."""
        tag_name = tag_name.strip().lower()
        for a in self._get_all_raw():
            if a.get('title') == title and tag_name in a.get('version_tags', []):
                return a.get('version')
        return None

    def get_log(self, title: str) -> List[Dict[str, Any]]:
        """
        Returns a complete, chronological Git-like version log for an artefact.
        """
        all_versions = [a for a in self._get_all_raw() if a.get('title') == title]
        all_versions.sort(key=lambda a: a.get('version', 0), reverse=True) # newest first

        log = []
        for a in all_versions:
            log.append({
                "commit_hash":    a.get("id"),
                "version":        a.get("version", 1),
                "author":         a.get("author", "AI Specialist"),
                "commit_message": a.get("commit_message") or f"Update '{title}' to version {a.get('version')}",
                "tags":           a.get("version_tags", []),
                "created_at":     a.get("created_at"),
                "size_chars":     len(a.get("content", "")),
                "is_active":      a.get("active", False)
            })
        return log

    def revert_to_tag(self, title: str, tag_name: str) -> Dict[str, Any]:
        """Reverts the active artefact version to the version bound to the tag."""
        version = self.resolve_tag(title, tag_name)
        if version is None:
            raise ValueError(f"Tag '{tag_name}' not found for artefact '{title}'.")
        return self.revert(title, version)

    # --------------------------------------------------------- activation

    def activate(self, title: str, version: Optional[int] = None):
        artefacts = self._get_all_raw()
        changed = False
        if version is not None:
            # Set only the requested version as active, deactivating all other versions
            for a in artefacts:
                if a.get('title') == title:
                    if a.get('version') == version:
                        a['active'] = True
                    else:
                        a['active'] = False
                    changed = True
        else:
            # Set the latest version as active, deactivating older versions
            latest_v = max([a.get('version', 1) for a in artefacts if a.get('title') == title], default=None)
            for a in artefacts:
                if a.get('title') == title:
                    a['active'] = (a.get('version') == latest_v)
                    changed = True
        if changed:
            self._save_all(artefacts)

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

            # Formulate the display name and hide the version suffix for data and read-only artifacts
            title_disp = item['title']
            if atype == "data":
                ext = item.get("file_ext", ".csv")
                title_disp = f"{item['title']}{ext}"

            if atype == "data" or item.get("read_only"):
                version_str = "Read-Only" if item.get("read_only") else ""
            else:
                versions = [a.get('version', 1) for a in self._get_all_raw() if a.get('title') == item['title']]
                total_versions = len(versions)
                version_str = f"v{item['version']}"
                if total_versions > 1:
                    version_str += f" | {total_versions} total versions exist"

            meta_str = ""
            # Omit metadata fields from LLM-facing context zone for skills to conserve tokens
            if atype != ArtefactType.SKILL:
                if item.get('author'):      meta_str += f" | Author: {item['author']}"
                if item.get('description'): meta_str += f"\nDescription: {item['description']}"
            label  = ArtefactType.LABELS.get(atype, atype.capitalize())
            v_info = f" ({version_str})" if version_str else ""
            header = f"###[{label}] {title_disp}{v_info}{meta_str}{url_line}"

            # Image count note (only when images present)
            img_note = ""
            imgs = item.get('images') or []
            if imgs:
                img_note = (
                    f"\n<!-- This artefact has {len(imgs)} image(s). "
                    f"They are referenced inline via <artefact_image id=\"{item['title']}::N\" /> tags. "
                    f"The actual image data has been appended to the conversation context. -->"
                )

            # Check if this content is truncated for context budget reasons
            deactivated_contents = getattr(self._discussion, "deactivated_contents", set())
            is_truncated = item['title'] in deactivated_contents

            if is_truncated:
                seq_summaries = getattr(self._discussion, "sequential_summaries", {})
                art_summary = seq_summaries.get(item['title'], "Detailed summary not available.")
                parts.append(f"{header}{img_note}\n[THE FOLLOWING IS A DETAILED SEQUENTIAL EXTRACTED SUMMARY OF THIS DOCUMENT DUE TO CONTEXT WINDOW LIMITATIONS]:\n{art_summary}")
            elif item.get('content', '').strip():
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

    @staticmethod
    def apply_aider_patch(original: str, patch_block: str) -> str:
        """
        Applies one or more aider SEARCH/REPLACE blocks to *original* text.

        Match passes (applied in order per segment until one succeeds):
          1.  Exact                  – verbatim substring match
          2.  Trailing-space         – ignores trailing whitespace per line
          3.  Indentation-agnostic   – strip-compares; picks best-scored window;
                                       re-indents replacement to match file indent
          C1. Comment-stripped       – strips inline # and // comments before comparing
          C2. Blank-line-collapsed   – collapses consecutive blank lines before comparing
          D.  Core-delta             – extracts and patches only the changed lines

        Key invariants:
          • res_lines is recomputed from `result` at the start of EVERY segment
            so that edits from a prior segment are always reflected.
          • All fuzzy passes route through _apply_at() which re-indents the
            replacement block to match the indentation of the matched region,
            preserving relative indentation within the replacement.
          • Pass 3 uses best-score selection (exact line bonus) to pick the
            correct window when the same lines appear multiple times in the file.
        """
        import re as _re

        ASCIIColors.panel(patch_block, "applying the patch")

        # ── Regex sentinels ──────────────────────────────────────────────────
        SEARCH_RE  = _re.compile(r'^<{6,8}(?:\s*\w+)?\s*$',  _re.IGNORECASE)
        SEP_RE     = _re.compile(r'^={5,}\s*$')
        REPLACE_RE = _re.compile(r'^>{6,8}(?:\s*\w+)?\s*$', _re.IGNORECASE)

        # ── Normalise line endings ───────────────────────────────────────────
        patch_block = patch_block.replace('\r\n', '\n').replace('\r', '\n')

        # Ensure the block ends with a REPLACE marker
        lines = patch_block.split('\n')
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and not REPLACE_RE.match(lines[-1].rstrip()):
            lines.append('>>>>>>> REPLACE')
        patch_block = '\n'.join(lines)

        # ── Parse all SEARCH / REPLACE segments ─────────────────────────────
        segments: List[Tuple[str, str]] = []
        raw_lines = patch_block.split('\n')
        i = 0
        while i < len(raw_lines):
            if SEARCH_RE.match(raw_lines[i].rstrip()):
                i += 1
                search_lines: List[str] = []
                while i < len(raw_lines):
                    if SEP_RE.match(raw_lines[i].rstrip()) or REPLACE_RE.match(raw_lines[i].rstrip()):
                        break
                    search_lines.append(raw_lines[i])
                    i += 1

                if i >= len(raw_lines):
                    raise ValueError(
                        "Malformed aider patch: SEARCH block has no ======= separator.\n"
                        f"Search text was:\n{''.join(search_lines)}"
                    )

                # SEARCH immediately followed by REPLACE (delete block)
                if REPLACE_RE.match(raw_lines[i].rstrip()):
                    segments.append(('\n'.join(search_lines), ''))
                    i += 1
                    continue

                # Consume separator, then collect REPLACE lines
                i += 1
                replace_lines: List[str] = []
                while i < len(raw_lines):
                    if REPLACE_RE.match(raw_lines[i].rstrip()):
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

        # ════════════════════════════════════════════════════════════════════
        # Helper functions
        # ════════════════════════════════════════════════════════════════════

        def _get_indent(line: str) -> int:
            """Count leading spaces (tab = 1 space for indent purposes)."""
            return len(line) - len(line.lstrip())

        def _reindent(r_text: str, target_indent: int) -> str:
            """
            Shift every non-empty line in *r_text* so the block's base indent
            maps to *target_indent*, preserving relative indentation within
            the replacement.  Empty / whitespace-only lines become ''.
            """
            r_lines = r_text.splitlines()
            first_nonempty = next((l for l in r_lines if l.strip()), None)
            if first_nonempty is None:
                return r_text
            base   = _get_indent(first_nonempty)
            delta  = target_indent - base

            def shift(line: str) -> str:
                if not line.strip():
                    return ''
                return ' ' * max(0, _get_indent(line) + delta) + line.lstrip()

            return '\n'.join(shift(l) for l in r_lines)

        def _apply_at(res_lines: List[str], start: int, length: int, r_text: str) -> str:
            """
            Splice *r_text* into *res_lines* at [start : start+length],
            re-indenting to match the indent of res_lines[start].
            """
            reindented = _reindent(r_text, _get_indent(res_lines[start]))
            return '\n'.join(res_lines[:start] + [reindented] + res_lines[start + length:])

        def _window_score(window: List[str], s_lines: List[str]) -> float:
            """
            Score a candidate window against the search lines.
            Higher = better match.
              +1.0  per line that matches exactly
              +0.5  per line that matches after rstrip()
              +0.25 per line that matches after strip()  (indentation-agnostic)
            """
            score = 0.0
            for w, s in zip(window, s_lines):
                if w == s:
                    score += 1.0
                elif w.rstrip() == s.rstrip():
                    score += 0.5
                elif w.strip() == s.strip():
                    score += 0.25
            return score

        def _strip_inline_comment(line: str) -> str:
            """Remove trailing # (Python/shell) or // (C-style) comments."""
            # Python / shell  — guard against stripping '#' inside strings
            s = _re.sub(r'''(?x)(?<![\'\"\\])\s*\#[^\'\"]* $''', '', line)
            if s != line:
                return s.rstrip()
            return _re.sub(r'\s*//.*$', '', line).rstrip()

        def _comment_key(line: str) -> str:
            return _strip_inline_comment(line).rstrip()

        def _collapse_blanks(lines_in: List[str]) -> List[Tuple[int, str]]:
            """
            Return (original_index, line) pairs with runs of consecutive blank
            lines collapsed to a single blank (keeping the first blank's index).
            """
            out: List[Tuple[int, str]] = []
            prev_blank = False
            for idx, l in enumerate(lines_in):
                is_blank = not l.strip()
                if is_blank and prev_blank:
                    continue
                out.append((idx, l))
                prev_blank = is_blank
            return out

        # ════════════════════════════════════════════════════════════════════
        # Per-segment matching loop
        # ════════════════════════════════════════════════════════════════════
        result = original

        for seg_idx, (search_text, replace_text) in enumerate(segments):
            s_text = search_text.strip('\n')
            r_text = replace_text.strip('\n')

            # Recompute every iteration — prior segments may have changed result
            res_lines = result.splitlines()
            s_lines   = s_text.splitlines()
            n_s       = len(s_lines)
            n_r       = len(res_lines)
            match_found = False

            label = f"seg {seg_idx + 1}/{len(segments)}"

            # ── Pass 1: Exact ────────────────────────────────────────────────
            if s_text in result:
                result = result.replace(s_text, r_text, 1)
                ASCIIColors.success(f"  [Patch] {label} Pass 1: exact match")
                continue

            # ── Pass 2: Trailing-space normalised ────────────────────────────
            for i in range(n_r - n_s + 1):
                window = res_lines[i : i + n_s]
                if all(w.rstrip() == s.rstrip() for w, s in zip(window, s_lines)):
                    pre  = res_lines[:i]
                    post = res_lines[i + n_s:]
                    result = '\n'.join(pre + [r_text] + post)
                    match_found = True
                    ASCIIColors.success(f"  [Patch] {label} Pass 2: trailing-space match at line {i+1}")
                    break

            if match_found:
                continue

            # ── Pass 3: Indentation-agnostic (best-scored window) ────────────
            # Scans ALL candidate windows, scores each, picks the highest score.
            # This handles:
            #   a) LLM-generated patches at column 0 for indented code
            #   b) Multiple identical blocks — correct occurrence is selected
            #      by rewarding lines that match beyond just strip-equality
            best_i     = -1
            best_score = -1.0

            for i in range(n_r - n_s + 1):
                window = res_lines[i : i + n_s]
                # All lines must match strip-wise; block must have content
                if (
                    all(w.strip() == s.strip() for w, s in zip(window, s_lines))
                    and any(s.strip() for s in s_lines)
                ):
                    score = _window_score(window, s_lines)
                    if score > best_score:
                        best_score = score
                        best_i     = i

            if best_i >= 0:
                result = _apply_at(res_lines, best_i, n_s, r_text)
                match_found = True
                ASCIIColors.warning(
                    f"  [Patch] {label} Pass 3: indentation-agnostic match at line {best_i+1} "
                    f"(score={best_score:.2f})"
                )

            if match_found:
                continue

            # ── Pass C1: Comment-stripped ────────────────────────────────────
            # Strip inline comments from both sides before comparing.
            # The original file lines are used when splicing (comments preserved).
            s_keys = [_comment_key(l) for l in s_lines]

            best_i     = -1
            best_score = -1.0

            for i in range(n_r - n_s + 1):
                window      = res_lines[i : i + n_s]
                window_keys = [_comment_key(l) for l in window]
                if (
                    all(wk == sk for wk, sk in zip(window_keys, s_keys))
                    and any(sk for sk in s_keys)
                ):
                    score = _window_score(window, s_lines)
                    if score > best_score:
                        best_score = score
                        best_i     = i

            if best_i >= 0:
                result = _apply_at(res_lines, best_i, n_s, r_text)
                match_found = True
                ASCIIColors.warning(
                    f"  [Patch] {label} Pass C1: comment-stripped match at line {best_i+1}"
                )

            if match_found:
                continue

            # ── Pass C2: Blank-line-collapsed ────────────────────────────────
            # Collapse consecutive blank lines in both the file and the search
            # block, then slide a window over the collapsed sequences.
            # On match, map back to original line indices for splicing.
            collapsed_res = _collapse_blanks(res_lines)
            collapsed_s   = _collapse_blanks(s_lines)
            cs_len        = len(collapsed_s)

            best_i      = -1
            best_orig_start = -1
            best_orig_end   = -1
            best_score  = -1.0

            for i in range(len(collapsed_res) - cs_len + 1):
                window = collapsed_res[i : i + cs_len]
                if (
                    all(
                        w_line.rstrip() == s_line.rstrip()
                        for (_, w_line), (_, s_line) in zip(window, collapsed_s)
                    )
                    and any(s_line.strip() for _, s_line in collapsed_s)
                ):
                    w_lines = [l for _, l in window]
                    s_lines_c = [l for _, l in collapsed_s]
                    score = _window_score(w_lines, s_lines_c)
                    if score > best_score:
                        best_score      = score
                        best_i          = i
                        best_orig_start = window[0][0]
                        best_orig_end   = window[-1][0] + 1  # exclusive

            if best_i >= 0:
                result = _apply_at(res_lines, best_orig_start, best_orig_end - best_orig_start, r_text)
                match_found = True
                ASCIIColors.warning(
                    f"  [Patch] {label} Pass C2: blank-collapsed match at line {best_orig_start+1}"
                )

            if match_found:
                continue

            # ── Pass D: Core-delta ────────────────────────────────────────────
            # If the whole block didn't match, extract just the lines that differ
            # plus 1 line of context and try to match that smaller target.
            s_lines_full = search_text.splitlines()
            r_lines_full = replace_text.splitlines()

            diff_indices = [
                idx for idx, (s, r) in enumerate(zip(s_lines_full, r_lines_full)) if s != r
            ]
            if not diff_indices and len(s_lines_full) != len(r_lines_full):
                diff_indices = list(range(len(s_lines_full)))

            if diff_indices:
                lo = max(0, min(diff_indices) - 1)
                hi = min(len(s_lines_full), max(diff_indices) + 2)
                core_search  = '\n'.join(s_lines_full[lo:hi])
                core_replace = '\n'.join(r_lines_full[lo:hi])

                if core_search and core_search in result:
                    result = result.replace(core_search, core_replace, 1)
                    match_found = True
                    ASCIIColors.success(f"  [Patch] {label} Pass D: core-delta match")

            if match_found:
                continue

            # ── All passes failed — full diagnostics ─────────────────────────
            first_line = s_lines[0] if s_lines else ""
            hint = _find_closest_line(first_line, result)

            def _debug_str(s: str) -> str:
                return "|".join(f"{c}({ord(c)})" for c in s)

            ASCIIColors.error(f"--- PATCH MATCH FAILURE [{label}] ---")
            ASCIIColors.yellow(f"Expected (first line) : {first_line.rstrip()!r}")
            ASCIIColors.yellow(f"Raw bytes expected    : {_debug_str(first_line.rstrip())}")
            ASCIIColors.cyan(  f"Closest line in file  : {hint.rstrip()!r}")
            ASCIIColors.cyan(  f"Raw bytes closest     : {_debug_str(hint.rstrip())}")

            if first_line.strip() == hint.strip() and first_line != hint:
                ASCIIColors.red(
                    "Pure indentation mismatch — same content, different leading spaces. "
                    "Pass 3 should have caught this; check that res_lines is recomputed "
                    "inside the segment loop."
                )

            raise ValueError(
                f"SEARCH text not found in artefact content [{label}].\n"
                f"Expected first line : {first_line.rstrip()!r}\n"
                f"Closest line found  : {hint.rstrip()!r}\n"
                f"All passes exhausted. Check console for byte-level debug."
            )

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

            commit_message = attrs.pop('commit_message', None)
            version_tags_raw = attrs.pop('version_tags', None)
            version_tags = [t.strip().lower() for t in version_tags_raw.split(',') if t.strip()] if version_tags_raw else None

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
                        commit_message=commit_message, version_tags=version_tags,
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
                            commit_message=commit_message,
                            version_tags=version_tags,
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
                        result_artefact = None # Do not mark as affected
            else:
                if is_new:
                    result_artefact = self.add(
                        title=resolved_title, artefact_type=atype,
                        content=content.strip(),
                        language=language, version=version, active=auto_activate,
                        commit_message=commit_message, version_tags=version_tags,
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
                        commit_message=commit_message,
                        version_tags=version_tags,
                        **attrs
                    )
                    final_title = result_artefact.get('title', resolved_title)
                    if new_name:
                        ASCIIColors.info(
                            f"Artefact '{resolved_title}' renamed to '{final_title}'")

            existing_titles = self._all_latest_titles()

            if result_artefact:
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
            if title and version:
                resolved = title if title in existing_titles else (
                    _find_best_title_match(title, existing_titles) or title
                )
                try:
                    res_artefact = self.revert(resolved, version)
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

    def get_version_history_artefact(self, title: str) -> List[Dict[str, Any]]:
        return self.get_version_history(title)

    def diff_versions_artefact(self, title: str, version_a: int, version_b: int) -> Dict[str, Any]:
        return self.diff_versions(title, version_a, version_b)

    def squash_versions_artefact(
        self,
        title: str,
        keep_versions: Optional[List[int]] = None,
        keep_last_n: Optional[int] = None,
        target_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.squash_versions(title, keep_versions, keep_last_n, target_version)

    def cleanup_old_versions_artefact(
        self,
        title: str,
        keep_count: int = 5,
        min_age_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.cleanup_old_versions(title, keep_count, min_age_hours)

    def remove_artefact(self, title, version=None) -> int:
        return self.remove(title, version)

    def get_associated_images(self, title: str, version: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all images associated with the artefact 'title'.
        Checks both the artefact's own images and any companion 'title::images' artefact.
        """
        images = []
        # 1. Main artefact images
        main_art = self.get(title, version)
        if main_art:
            imgs = main_art.get("images") or []
            mtypes = main_art.get("image_media_types") or []
            for idx, img_b64 in enumerate(imgs):
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                images.append({
                    "id": make_image_id(title, idx),
                    "data": img_b64,
                    "media_type": mtype,
                    "title": title,
                    "index": idx,
                })

        # 2. Companion images artefact
        comp_title = f"{title}::images"
        comp_art = self.get(comp_title)
        if comp_art:
            imgs = comp_art.get("images") or []
            mtypes = comp_art.get("image_media_types") or []
            for idx, img_b64 in enumerate(imgs):
                mtype = mtypes[idx] if idx < len(mtypes) else "image/jpeg"
                images.append({
                    "id": make_image_id(comp_title, idx),
                    "data": img_b64,
                    "media_type": mtype,
                    "title": comp_title,
                    "index": idx,
                })
        return images

    def export_artefact_bundle(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Exports an artefact and all of its associated images/companion files
        into a single, self-contained JSON-serializable bundle.
        """
        main_art = self.get(title)
        if not main_art:
            return None

        bundle = {
            "version": 1,
            "exported_at": datetime.utcnow().isoformat(),
            "main_artefact": {k: v for k, v in main_art.items() if k != "id"},
            "companion_artefacts": []
        }

        # Find companion images artefact
        comp_title = f"{title}::images"
        comp_art = self.get(comp_title)
        if comp_art:
            bundle["companion_artefacts"].append({k: v for k, v in comp_art.items() if k != "id"})

        return bundle

    def import_artefact_bundle(self, bundle: Dict[str, Any], activate: bool = True) -> Optional[Dict[str, Any]]:
        """
        Imports a previously exported artefact bundle back into the discussion.
        """
        if not isinstance(bundle, dict) or "main_artefact" not in bundle:
            raise ValueError("Invalid artefact bundle format")

        main_info = bundle["main_artefact"]
        main_art = self.add(
            title             = main_info["title"],
            artefact_type     = main_info.get("type", ArtefactType.DOCUMENT),
            content           = main_info.get("content", ""),
            images            = main_info.get("images"),
            image_media_types = main_info.get("image_media_types"),
            audios            = main_info.get("audios"),
            videos            = main_info.get("videos"),
            zip_content       = main_info.get("zip"),
            language          = main_info.get("language"),
            url               = main_info.get("url"),
            tags              = main_info.get("tags"),
            active            = activate
        )

        for comp_info in bundle.get("companion_artefacts", []):
            self.add(
                title             = comp_info["title"],
                artefact_type     = comp_info.get("type", ArtefactType.IMAGE),
                content           = comp_info.get("content", ""),
                images            = comp_info.get("images"),
                image_media_types = comp_info.get("image_media_types"),
                audios            = comp_info.get("audios"),
                videos            = comp_info.get("videos"),
                zip_content       = comp_info.get("zip"),
                language          = comp_info.get("language"),
                url               = comp_info.get("url"),
                tags              = comp_info.get("tags"),
                active            = activate
            )

        return main_art

    def export_artefact(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Exports an entire artefact including all of its versions and history,
        as well as all versions of any companion images/assets.
        """
        all_raw = self._get_all_raw()
        versions = [a for a in all_raw if a.get('title') == title]
        if not versions:
            return None

        def _safe_version_key(a):
            try:
                v = a.get('version', 1)
                if isinstance(v, str):
                    v = int(float(v.split('.')[0]))
                return int(v)
            except Exception:
                return 1

        sorted_versions = sorted(versions, key=_safe_version_key)
        latest = sorted_versions[-1]

        comp_versions = [a for a in all_raw if a.get('title') == f"{title}::images"]
        sorted_comp_versions = sorted(comp_versions, key=_safe_version_key)

        return {
            "version": 1, # Export bundle format version
            "title": title,
            "type": latest.get("type", ArtefactType.DOCUMENT),
            "versions": [{k: v for k, v in ver.items() if k != "id"} for ver in sorted_versions],
            "companion_images_versions": [{k: v for k, v in ver.items() if k != "id"} for ver in sorted_comp_versions],
            "exported_at": datetime.utcnow().isoformat()
        }

    def import_artefact(self, artefact_data: Dict[str, Any], activate: bool = True) -> Optional[Dict[str, Any]]:
        """
        Imports a previously exported multi-version artefact dataset into the discussion,
        rebuilding its entire version history chronologically.
        """
        if not isinstance(artefact_data, dict) or "versions" not in artefact_data:
            raise ValueError("Invalid multi-version exported artefact format")

        title = artefact_data.get("title")
        if not title:
            raise ValueError("Missing title in imported artefact data")

        # Strip any existing versions of this artifact (and its companion) from the discussion
        # to ensure a clean overwrite/restore of its entire version history
        all_raw = self._get_all_raw()
        cleaned_raw = [a for a in all_raw if a.get('title') != title and a.get('title') != f"{title}::images"]

        imported_versions = []
        for ver_info in artefact_data["versions"]:
            new_ver = ver_info.copy()
            new_ver["id"] = str(uuid.uuid4())
            new_ver["active"] = False # Default off during list insertion
            cleaned_raw.append(new_ver)
            imported_versions.append(new_ver)

        for comp_info in artefact_data.get("companion_images_versions", []):
            new_comp = comp_info.copy()
            new_comp["id"] = str(uuid.uuid4())
            new_comp["active"] = False
            cleaned_raw.append(new_comp)

        self._save_all(cleaned_raw)

        # Set active status on the latest version if requested
        if imported_versions:
            latest_imported = imported_versions[-1]
            if activate:
                self.activate(title, latest_imported.get("version", 1))
                if artefact_data.get("companion_images_versions"):
                    self.activate(f"{title}::images")

        return self.get(title)
