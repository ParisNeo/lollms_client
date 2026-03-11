# lollms_discussion/_artefacts.py
# ArtefactType constants and ArtefactManager: full typed artefact subsystem.

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
    Constants for the supported artefact types.

    All types share the same flat schema (title, type, version, content,
    images, language, url, tags, active, …extra_data).  The ``type`` field
    is purely a label — there are no distinct per-type sub-schemas.
    Extra fields meaningful to a specific type (e.g. ``language`` for CODE,
    ``url`` for SEARCH_RESULT) are stored in the same dict via **extra_data.
    """
    FILE          = "file"           # Generic uploaded file (text / binary as b64)
    SEARCH_RESULT = "search_result"  # Results from a web / RAG search
    NOTE          = "note"           # Free-form user note
    SKILL         = "skill"          # SKILL.md-compatible capability descriptor
    CODE          = "code"           # LLM-generated or user-supplied code
    DOCUMENT      = "document"       # Generated or imported text document
    IMAGE         = "image"          # Image stored as base64 in the ``images`` list

    ALL = {FILE, SEARCH_RESULT, NOTE, SKILL, CODE, DOCUMENT, IMAGE}

    # Human-readable display labels (for UI / logging)
    LABELS = {
        FILE:          "File",
        SEARCH_RESULT: "Search Result",
        NOTE:          "Note",
        SKILL:         "Skill",
        CODE:          "Code",
        DOCUMENT:      "Document",
        IMAGE:         "Image",
    }


# ---------------------------------------------------------------------------
# ArtefactManager
# ---------------------------------------------------------------------------

class ArtefactManager:
    """
    Manages the lifecycle of typed artefacts inside a LollmsDiscussion.

    Artefact schema (stored in discussion_metadata["_artefacts"]):
    {
        "id":           str   – stable UUID (survives renames / version bumps)
        "title":        str   – human-readable name / identifier
        "type":         str   – one of ArtefactType.ALL
        "version":      int   – incremented on each update
        "content":      str   – text content (code, markdown, skill text, …)
        "images":       list  – list of base64 strings  (for IMAGE type)
        "audios":       list  – list of base64 strings  (reserved)
        "videos":       list  – list of base64 strings  (reserved)
        "zip":          str | None  – base64 zip blob   (reserved)
        "language":     str | None  – programming language hint  (CODE type)
        "url":          str | None  – source URL  (SEARCH_RESULT / FILE)
        "tags":         list  – free-form tags
        "active":       bool  – whether it is injected into the context
        "created_at":   str   – ISO datetime
        "updated_at":   str   – ISO datetime
        # …any extra_data kwargs are also stored here
    }
    """

    def __init__(self, discussion: 'LollmsDiscussion'):
        object.__setattr__(self, '_discussion', discussion)

    # --------------------------------------------------------- internal helpers

    def _get_all_raw(self) -> List[Dict]:
        """Returns the raw list from metadata, migrating legacy entries."""
        metadata = self._discussion.metadata or {}
        raw = metadata.get("_artefacts", [])
        now = datetime.utcnow().isoformat()
        dirty = False
        migrated = []
        for a in raw:
            fixed = a.copy()
            if "id"         not in fixed: fixed["id"]         = str(uuid.uuid4()); dirty = True
            if "type"       not in fixed: fixed["type"]        = ArtefactType.DOCUMENT; dirty = True
            if "title"      not in fixed: fixed["title"]       = "untitled"; dirty = True
            if "content"    not in fixed: fixed["content"]     = ""; dirty = True
            if "images"     not in fixed: fixed["images"]      = []; dirty = True
            if "audios"     not in fixed: fixed["audios"]      = []; dirty = True
            if "videos"     not in fixed: fixed["videos"]      = []; dirty = True
            if "zip"        not in fixed: fixed["zip"]         = None; dirty = True
            if "language"   not in fixed: fixed["language"]    = None; dirty = True
            if "url"        not in fixed: fixed["url"]         = None; dirty = True
            if "tags"       not in fixed: fixed["tags"]        = []; dirty = True
            if "active"     not in fixed: fixed["active"]      = False; dirty = True
            if "version"    not in fixed: fixed["version"]     = 1; dirty = True
            if "created_at" not in fixed: fixed["created_at"]  = now; dirty = True
            if "updated_at" not in fixed: fixed["updated_at"]  = now; dirty = True
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

    # ----------------------------------------------------------------- CRUD

    def add(
        self,
        title:         str,
        artefact_type: str = ArtefactType.DOCUMENT,
        content:       str = "",
        images:        Optional[List[str]] = None,
        audios:        Optional[List[str]] = None,
        videos:        Optional[List[str]] = None,
        zip_content:   Optional[str] = None,
        language:      Optional[str] = None,
        url:           Optional[str] = None,
        tags:          Optional[List[str]] = None,
        version:       int = 1,
        active:        bool = True,
        **extra_data
    ) -> Dict[str, Any]:
        """
        Adds a new artefact or a specific version of an existing one.

        Args:
            title: The unique identifier/filename for the artefact.
            artefact_type: The category (e.g., code, document, skill). See ArtefactType.
            content: The text/markdown content.
            images: Optional list of base64 strings.
            audios: Optional list of base64 strings (reserved).
            videos: Optional list of base64 strings (reserved).
            zip_content: Optional base64 zip blob.
            language: Programming language hint for syntax highlighting.
            url: Source URL if applicable.
            tags: List of organizational tags.
            version: The version number. Defaults to 1.
            active: If True, the content is injected into the LLM system prompt.
            **extra_data: Arbitrary metadata (author, source, description, etc.)

        Returns:
            The created artefact dictionary.
        """
        if artefact_type not in ArtefactType.ALL:
            raise ValueError(
                f"Unknown artefact type '{artefact_type}'. "
                f"Use one of: {sorted(ArtefactType.ALL)}"
            )
        artefacts = self._get_all_raw()
        artefacts = [a for a in artefacts
                     if not (a.get('title') == title and a.get('version') == version)]
        now = datetime.utcnow().isoformat()
        new_artefact: Dict[str, Any] = {
            "id":         str(uuid.uuid4()),
            "title":      title,
            "type":       artefact_type,
            "version":    version,
            "content":    content,
            "images":     images or [],
            "audios":     audios or [],
            "videos":     videos or [],
            "zip":        zip_content,
            "language":   language,
            "url":        url,
            "tags":       tags or [],
            "active":     active,
            "created_at": now,
            "updated_at": now,
            **extra_data,
        }
        artefacts.append(new_artefact)
        self._save_all(artefacts)
        return new_artefact

    def get(self, title: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Returns an artefact by title. If version is None, returns the latest."""
        candidates = [a for a in self._get_all_raw() if a.get('title') == title]
        if not candidates:
            return None
        if version is not None:
            return next((a for a in candidates if a.get('version') == version), None)
        return max(candidates, key=lambda a: a.get('version', 0))

    def get_by_id(self, artefact_id: str) -> Optional[Dict[str, Any]]:
        """Returns an artefact by its stable UUID."""
        return next((a for a in self._get_all_raw() if a.get('id') == artefact_id), None)

    def list(
        self,
        artefact_type: Optional[str] = None,
        active_only:   bool = False,
        tags:          Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Lists artefacts with optional filters (type, active_only, tags)."""
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
        title:         str,
        new_content:   Optional[str] = None,
        new_type:      Optional[str] = None,
        new_images:    Optional[List[str]] = None,
        new_tags:      Optional[List[str]] = None,
        language:      Optional[str] = None,
        url:           Optional[str] = None,
        bump_version:  bool = True,
        active:        Optional[bool] = None,
        **extra_data
    ) -> Dict[str, Any]:
        """
        Creates a new version of an existing artefact with updated fields.

        Args:
            title: Title of the artefact to update.
            new_content: New text content. If None, current content is preserved.
            new_type: If provided, changes the ArtefactType (e.g., from 'document' to 'code').
            new_images: New images list. If None, current images are preserved.
            new_tags: New tags list.
            language: Programming language hint.
            url: Updated source URL.
            bump_version: If True (default), increments version. If False, overwrites current.
            active: Sets or preserves the activation state.
            **extra_data: New or updated metadata fields.

        Returns:
            The updated artefact dictionary.
        """
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Cannot update non-existent artefact '{title}'.")
        new_version = (latest.get('version', 1) + 1) if bump_version else latest.get('version', 1)
        
        # Determine active state: priority to argument, fallback to previous state
        new_active = active if active is not None else latest.get('active', True)

        # Merge old extra_data with new ones so we don't lose existing metadata (author, etc.)
        merged_extra = {k: v for k, v in latest.items() if k not in ["id", "title", "type", "version", "content", "images", "audios", "videos", "zip", "language", "url", "tags", "active", "created_at", "updated_at"]}
        merged_extra.update(extra_data)

        return self.add(
            title         = title,
            artefact_type = new_type if new_type is not None else latest.get('type', ArtefactType.DOCUMENT),
            content       = new_content if new_content is not None else latest.get('content', ''),
            images        = new_images  if new_images  is not None else latest.get('images', []),
            audios        = latest.get('audios', []),
            videos        = latest.get('videos', []),
            zip_content   = latest.get('zip'),
            language      = language if language is not None else latest.get('language'),
            url           = url if url is not None else latest.get('url'),
            tags          = new_tags if new_tags is not None else latest.get('tags', []),
            version       = new_version,
            active        = new_active,
            **merged_extra,
        )

    def revert(self, title: str, target_version: int) -> Dict[str, Any]:
        """
        Reverts an artefact to a previous version by creating a new version 
        that copies the content and metadata of the target version.
        """
        target = self.get(title, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} of artefact '{title}' not found.")
        
        # Create a new bump version
        latest = self.get(title)
        new_version = (latest.get('version', 1) + 1) if latest else 1
        
        # Extract extra metadata from target
        extra_keys = {k: v for k, v in target.items() if k not in["id", "title", "type", "version", "content", "images", "audios", "videos", "zip", "language", "url", "tags", "active", "created_at", "updated_at"]}
        
        ASCIIColors.info(f"Reverting '{title}' to version {target_version} (now saved as v{new_version})")
        
        return self.add(
            title         = title,
            artefact_type = target.get('type', ArtefactType.DOCUMENT),
            content       = target.get('content', ''),
            images        = target.get('images', []),
            audios        = target.get('audios',[]),
            videos        = target.get('videos',[]),
            zip_content   = target.get('zip'),
            language      = target.get('language'),
            url           = target.get('url'),
            tags          = target.get('tags',[]),
            version       = new_version,
            active        = True,
            **extra_keys,
        )

    def remove(self, title: str, version: Optional[int] = None) -> int:
        """Removes artefact(s) by title. Removes all versions if version is None."""
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
        """Removes an artefact by its stable UUID. Returns True if found and removed."""
        artefacts = self._get_all_raw()
        new_list = [a for a in artefacts if a.get('id') != artefact_id]
        if len(new_list) < len(artefacts):
            self._save_all(new_list)
            return True
        return False

    # --------------------------------------------------------- activation

    def activate(self, title: str, version: Optional[int] = None):
        """Marks an artefact as active (it will be injected into the system prompt)."""
        self._set_active(title, version, True)

    def deactivate(self, title: str, version: Optional[int] = None):
        """Marks an artefact as inactive (excluded from system prompt)."""
        self._set_active(title, version, False)

    def toggle(self, title: str, version: Optional[int] = None) -> bool:
        """Toggles the active state. Returns the new state."""
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
        Assembles a text block injected into the system prompt for every active
        artefact that has non-empty content.  IMAGE-only artefacts with no text
        are skipped (their images surface via message.images).
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
            
            # Version tracking context
            versions =[a.get('version', 1) for a in self._get_all_raw() if a.get('title') == item['title']]
            total_versions = len(versions)
            version_str = f"v{item['version']}"
            if total_versions > 1:
                version_str += f" | {total_versions} total versions exist"

            # Metadata formatting
            meta_str = ""
            if item.get('author'): meta_str += f" | Author: {item['author']}"
            if item.get('description'): meta_str += f"\nDescription: {item['description']}"

            label    = ArtefactType.LABELS.get(atype, atype.capitalize())
            header   = f"###[{label}] {item['title']} ({version_str}){meta_str}{url_line}"
            
            if item.get('content', '').strip():
                parts.append(f"{header}\n{fence}\n{item['content'].strip()}\n```")
            else:
                parts.append(header)

        return "## Active Artefacts\n\n" + "\n\n".join(parts)

    def get_active_images(self) -> List[Dict[str, Any]]:
        """
        Returns images from active IMAGE-type artefacts for UI/export use.
        These are NOT injected into the chat context — images go through message.images.
        """
        result = []
        for a in self._get_all_raw():
            if a.get('active', False) and a.get('type') == ArtefactType.IMAGE:
                source_id = f"artefact:{a['title']} v{a['version']}"
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

        patch_block format:
            <<<<<<< SEARCH
            [exact lines to find]
            =======
            [replacement lines]
            >>>>>>> REPLACE

        Raises ValueError if a SEARCH section is not found verbatim.
        """
        SEARCH_MARKER  = "<<<<<<< SEARCH"
        SEP_MARKER     = "======="
        REPLACE_MARKER = ">>>>>>> REPLACE"

        result = original
        segments = patch_block.split(SEARCH_MARKER)
        for seg in segments[1:]:
            if SEP_MARKER not in seg or REPLACE_MARKER not in seg:
                raise ValueError(f"Malformed aider patch block:\n{seg}")
            search_part, rest  = seg.split(SEP_MARKER, 1)
            replace_part, _    = rest.split(REPLACE_MARKER, 1)
            search_text  = search_part.lstrip('\n').rstrip('\n')
            replace_text = replace_part.lstrip('\n').rstrip('\n')
            if search_text not in result:
                raise ValueError(
                    f"SEARCH text not found in artefact content:\n---\n{search_text}\n---"
                )
            result = result.replace(search_text, replace_text, 1)
        return result

    # -------------------------------------------- LLM artefact XML parser (internal)

    def _apply_artefact_xml(
        self,
        text:         str,
        default_type: str = ArtefactType.CODE,
        auto_activate: bool = True,
        replacements:  Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Internal helper called exclusively from LollmsDiscussion._post_process_llm_response.

        Scans *text* for ``<artefact …>…</artefact>`` blocks and:
        * Creates a new artefact when content has no aider markers.
        * Applies an aider SEARCH/REPLACE patch when markers are present.

        Returns (cleaned_text, affected_artefacts_list).
        """
        affected: List[Dict] = []
        cleaned  = text

        artefact_pattern = re.compile(
            r'<artefact\s([^>]*)>(.*?)</artefact>',
            re.DOTALL | re.IGNORECASE
        )

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        def handle_artefact(match: re.Match) -> str:
            attrs    = _parse_attrs(match.group(1))
            content  = match.group(2)

            # Restore masked content (code blocks) before saving to the database
            if replacements:
                for placeholder, original in replacements.items():
                    content = content.replace(placeholder, original)
                    # Also check attributes just in case
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

            title    = attrs.pop('name', attrs.pop('title', f'artefact_{uuid.uuid4().hex[:8]}'))
            atype    = attrs.pop('type', default_type)
            language = attrs.pop('language', None)
            version_str = attrs.pop('version', '1')
            version  = int(version_str) if version_str.isdigit() else 1

            if atype not in ArtefactType.ALL:
                atype = default_type

            if "<<<<<<< SEARCH" in content:
                existing = self.get(title)
                if existing is None:
                    result_artefact = self.add(
                        title=title, artefact_type=atype, content=content,
                        language=language, version=version, active=auto_activate,
                        **attrs
                    )
                else:
                    try:
                        patched = self.apply_aider_patch(existing.get('content', ''), content)
                        result_artefact = self.update(
                            title=title, new_content=patched,
                            language=language, bump_version=True,
                            active=auto_activate,
                            **attrs
                        )
                    except ValueError as e:
                        ASCIIColors.warning(f"Aider patch failed for '{title}': {e}")
                        result_artefact = existing
            else:
                result_artefact = self.add(
                    title=title, artefact_type=atype, content=content.strip(),
                    language=language, version=version, active=auto_activate,
                    **attrs
                )

            affected.append(result_artefact)
            return ''  # strip tag from cleaned output

        cleaned = artefact_pattern.sub(handle_artefact, cleaned)
        
        # Also parse <revert_artefact name="..." version="..." /> tags
        revert_pattern = re.compile(r'<revert_artefact\s([^>]+)/?>', re.IGNORECASE)
        def handle_revert(match: re.Match) -> str:
            attrs = _parse_attrs(match.group(1))
            title = attrs.get('name') or attrs.get('title')
            version = attrs.get('version')
            if title and version and version.isdigit():
                try:
                    res_artefact = self.revert(title, int(version))
                    affected.append(res_artefact)
                except ValueError as e:
                    ASCIIColors.warning(str(e))
            return ''
            
        cleaned = revert_pattern.sub(handle_revert, cleaned)

        return cleaned.strip(), affected

    # ------------------------------------------------- legacy compat shims

    def add_artefact(self, title, content="", images=None, audios=None, videos=None,
                     zip_content=None, version=1, **extra_data) -> Dict:
        """Legacy shim → self.add(). Allows overriding artefact_type via extra_data."""
        # Extract artefact_type if user provided it in kwargs, otherwise default to DOCUMENT
        atype = extra_data.pop('artefact_type', ArtefactType.DOCUMENT)
        return self.add(
            title=title, artefact_type=atype,
            content=content, images=images, audios=audios, videos=videos,
            zip_content=zip_content, version=version, **extra_data
        )

    def get_artefact(self, title, version=None) -> Optional[Dict]:
        """Legacy shim → self.get()"""
        return self.get(title, version)

    def update_artefact(self, title, new_content, new_images=None, **extra_data) -> Dict:
        """Legacy shim → self.update()"""
        return self.update(title=title, new_content=new_content,
                           new_images=new_images, **extra_data)

    def list_artefacts(self) -> List[Dict]:
        """Legacy shim → self.list()  (adds is_loaded field for compat)"""
        items = self.list()
        disc = self._discussion
        for a in items:
            section_start  = f"--- Document: {a['title']} v{a['version']} ---"
            content_loaded = section_start in (disc.discussion_data_zone or "")
            source_id      = f"artefact:{a['title']} v{a['version']}"
            image_loaded   = any(
                img.get("source") == source_id for img in disc.get_discussion_images()
            )
            a["is_loaded"] = content_loaded or image_loaded
        return items

    def remove_artefact(self, title, version=None) -> int:
        """Legacy shim → self.remove()"""
        return self.remove(title, version)
