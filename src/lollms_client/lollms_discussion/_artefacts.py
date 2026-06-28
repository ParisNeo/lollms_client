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
import os
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors

if TYPE_CHECKING:
    from lollms_discussion import LollmsDiscussion


# ---------------------------------------------------------------------------
# ArtefactType
# ---------------------------------------------------------------------------

class ArtefactVisibility:
    """Visibility tiers for context optimization."""
    HIDDEN          = "hidden"           # Level 0: Invisible to LLM
    TREE_LOCKED     = "tree_locked"      # Level 1: In tree, cannot be unlocked
    TREE_UNLOCKABLE = "tree_unlockable"  # Level 2: In tree, can be unlocked by LLM
    METADATA        = "metadata"         # Level 3: Visible with schema/signatures only
    FULL            = "full"             # Level 4: Fully loaded in context


class ArtefactType:
    """
    Registry for supported artefact types.
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
# Code Signature & Metadata Extraction Helpers
# ---------------------------------------------------------------------------

def _extract_code_signatures(content: str, language: Optional[str]) -> str:
    """Extracts function, class, and method definitions to create a signature stub."""
    lang = (language or "").lower()

    # 1. Python Signature Extractor
    if lang == "python" or "def " in content or "class " in content:
        import ast
        try:
            tree = ast.parse(content)
            sigs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            args = [arg.arg for arg in child.args.args]
                            methods.append(f"    def {child.name}({', '.join(args)}): ...")
                    base_classes = [ast.unparse(base) for base in node.bases]
                    base_str = f"({', '.join(base_classes)})" if base_classes else ""
                    sigs.append(f"class {node.name}{base_str}:\n" + "\n".join(methods))
                elif isinstance(node, ast.FunctionDef) and not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if node in p.body):
                    args = [arg.arg for arg in node.args.args]
                    sigs.append(f"def {node.name}({', '.join(args)}): ...")
            if sigs:
                return "\n\n".join(sigs)
        except Exception:
            pass

    # 2. Regular Expression Based Definition Scanner (C, C++, JS, TS, Rust, Go)
    signatures = []
    lines = content.splitlines()

    # Regex patterns matching standard function, class, or interface declarations
    patterns = [
        # JS/TS/Go/Rust: function name(args), fn name(args), func name(args)
        r'^\s*(async\s+)?(function|fn|func|class|interface|struct)\s+([a-zA-Z0-9_]+)\b',
        # C/C++: type name(args) {
        r'^\s*(void|int|float|double|char|bool|auto|std::\w+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{?',
        # JS ES6 Arrow function assignments: const name = (args) =>
        r'^\s*(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(async\s*)?\([^)]*\)\s*=>'
    ]
    compiled = [re.compile(p) for p in patterns]

    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith(("//", "#", "/*", "*")):
            continue
        for rx in compiled:
            if rx.match(line_strip):
                # Clean up braces and print definition stub
                clean_line = re.sub(r'\s*\{\s*$', '', line_strip)
                signatures.append(clean_line + " { ... }")
                break

    if signatures:
        return "\n".join(signatures)

    # 3. Fallback: return line/character metadata only
    return f"// [Code Signature Extraction Unavailable: File contains {len(lines)} lines]"


def _extract_file_metadata_and_signature(art: Dict[str, Any]) -> str:
    """Compiles a compact signature metadata descriptor for partial visibility tiers."""
    content = art.get("content", "")
    atype = art.get("type", "document")
    title = art["title"]
    lang = art.get("language")

    size_chars = len(content)
    line_count = len(content.splitlines())

    summary = [
        f"### [Partial Metadata] {title}",
        f"- Format Type: {atype.upper()} {f'({lang})' if lang else ''}",
        f"- Size: {size_chars:,} characters | {line_count:,} lines"
    ]

    # For data structures, we already have a compact schema in the content
    if atype == ArtefactType.DATA:
        # Extract headers and first few rows only
        schema_lines = []
        for line in content.splitlines()[:20]:
            schema_lines.append(line)
        if len(content.splitlines()) > 20:
            schema_lines.append("\n  ... [remaining schema lines truncated]")
        summary.append("\n".join(schema_lines))
        return "\n".join(summary)

    # For HTML/Presentation layouts, extract outline
    if lang == "html" or atype == "presentation":
        soup_headings = re.findall(r'<h[1-3][^>]*>(.*?)</h[1-3]>|<section\s+class="slide[^"]*"[^>]*data-notes="([^"]*)"', content, re.I)
        if soup_headings:
            summary.append("#### Structure Outline:")
            for h in soup_headings[:15]:
                h_text = h[0] or f"Slide with notes: {h[1][:40]}..."
                summary.append(f"  • {re.sub(r'<[^>]+>', '', h_text).strip()}")
            return "\n".join(summary)

    # For code, extract signatures
    if atype in (ArtefactType.CODE, ArtefactType.TOOL) or lang in _CODE_EXTENSIONS:
        summary.append("#### Abstract Class & Function Signatures:")
        summary.append("```" + (lang or "python"))
        summary.append(_extract_code_signatures(content, lang))
        summary.append("```")
        return "\n".join(summary)

    # General plain text fallback
    preview = []
    for line in content.splitlines()[:6]:
        preview.append(line)
    summary.append("#### Context Preview:")
    summary.append("\n".join(preview) + "\n  ... [remaining document content truncated]")
    return "\n".join(summary)


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

            # Migrate legacy binary active state to the 5-tier visibility string
            if "visibility" not in fixed:
                legacy_active = fixed.get("active", False)
                fixed["visibility"] = ArtefactVisibility.FULL if legacy_active else ArtefactVisibility.HIDDEN
                dirty = True

            if "active"           not in fixed: fixed["active"]           = (fixed["visibility"] == ArtefactVisibility.FULL); dirty = True
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

    def _get_filename_with_ext(self, title: str, atype: str, language: Optional[str] = None, file_ext: Optional[str] = None) -> str:
        if "." in title and len(title.split(".")[-1]) <= 5:
            return title
        ext = file_ext or ""
        if not ext:
            if language:
                lang_map = {
                    "python": ".py", "javascript": ".js", "typescript": ".ts",
                    "html": ".html", "css": ".css", "markdown": ".md",
                    "latex": ".tex", "tex": ".tex", "sparql": ".rq", "sql": ".sql"
                }
                ext = lang_map.get(language.lower(), "")
            if not ext:
                type_map = {
                    "code": ".py", "document": ".md", "note": ".txt",
                    "skill": ".md", "data": ".csv", "presentation": ".html",
                    "tool": ".py"
                }
                ext = type_map.get(atype, ".txt")
        return f"{title}{ext}"

    def _sync_to_disk_workspace(self, title: str, content: str, version: int, atype: str, language: Optional[str] = None, file_ext: Optional[str] = None, physical_data: Optional[bytes] = None, logical_content: Optional[str] = None):
        """
        Saves artifact to the discussion's ISOLATED workspace folder on disk using Dual-Stream Storage.

        ARCHITECTURAL RULE: Dual-Stream Storage (.lam Protocol)
        - physical_data (bytes): The raw binary/text file for tools to execute against (CSV, DB, PNG, etc.).
          Saved to: workspace_data/{title}.{ext}
        - logical_content (str): The .lam metadata (Schema, Stats, Description) for the LLM context.
          Saved to: artefacts_metadata/{uuid}/{title}.lam (NEVER copied to workspace_data)
        """
        try:
            # Resolve directories dynamically from the structured core discussion attributes
            if getattr(self._discussion, "workspace_data_path", None):
                ws_data_dir = Path(self._discussion.workspace_data_path)
                meta_dir = Path(self._discussion.artefacts_metadata_path)
            else:
                # Standalone fallback if called outside server context
                base_workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")
                disc_id = self._discussion.id
                disc_ws_dir = base_workspace_dir / disc_id
                ws_data_dir = disc_ws_dir / "workspace_data"
                meta_dir = disc_ws_dir / "artefacts_metadata"

            ws_data_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            clean_title = title.replace("workspace/", "").replace("data_workspace/", "").replace("/", "_").replace("\\", "_")
            filename = self._get_filename_with_ext(clean_title, atype, language, file_ext)
            name_part, ext_part = os.path.splitext(filename) if '.' in filename else (filename, "")

            # Recover or generate a stable UUID for this specific artifact to prevent version collisions
            art_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_title))
            art_meta_dir = meta_dir / art_id
            art_meta_dir.mkdir(parents=True, exist_ok=True)

            primary_path = ws_data_dir / filename
            versioned_path = art_meta_dir / f"{name_part}_v{version}{ext_part}"

            # .lam paths (Logical Twin)
            lam_filename = f"{name_part}.lam"
            lam_path = art_meta_dir / lam_filename

            # ── 1. WRITE PHYSICAL TWIN (For Tools) ─────────────────────────────
            wrote_physical = False
            if physical_data is not None:
                try:
                    primary_path.write_bytes(physical_data)
                    versioned_path.write_bytes(physical_data)
                    wrote_physical = True
                    ASCIIColors.success(f"[Dual-Stream] ✈️ Synced PHYSICAL '{filename}' (v{version}) to workspace_data & versions.")
                except Exception as bin_err:
                    ASCIIColors.warning(f"Failed to write physical data for '{filename}': {bin_err}")

            # Fallback: If no physical_data but content is raw text (Code/Doc), treat content as physical
            # 🛡️ SOVEREIGN PHYSICAL GAURD: Image/Data placeholder descriptions must NEVER be written as physical text files
            if not wrote_physical and isinstance(content, str) and atype not in (ArtefactType.IMAGE, ArtefactType.DATA):
                try:
                    primary_path.write_text(content, encoding="utf-8", errors="ignore")
                    versioned_path.write_text(content, encoding="utf-8", errors="ignore")
                    wrote_physical = True
                    ASCIIColors.success(f"[Dual-Stream] ✈️ Synced TEXT '{filename}' (v{version}) to workspace_data & versions.")
                except Exception as txt_err:
                    ASCIIColors.warning(f"Failed to write text content for '{filename}': {txt_err}")

            # ── 2. WRITE LOGICAL TWIN (.lam For LLM) ───────────────────────────
            # We ALWAYS write the .lam if provided, regardless of physical success
            if logical_content:
                try:
                    lam_path.write_text(logical_content, encoding="utf-8", errors="ignore")
                    ASCIIColors.cyan(f"[Dual-Stream] 🧠 Saved LOGICAL '{lam_filename}' to artefacts_metadata (Hidden from tools).")
                except Exception as lam_err:
                    ASCIIColors.warning(f"Failed to write logical .lam for '{title}': {lam_err}")
            elif wrote_physical and atype in (ArtefactType.DATA, ArtefactType.IMAGE, ArtefactType.DOCUMENT):
                # Auto-generate a minimal .lam if missing for binary types
                minimal_lam = f"# Artefact Metadata: {filename}\n- **Type**: {atype}\n- **Version**: {version}\n- **Physical Path**: workspace_data/{filename}\n\n> **Note**: No logical summary generated. This is a raw binary/text file."
                try:
                    lam_path.write_text(minimal_lam, encoding="utf-8", errors="ignore")
                    ASCIIColors.info(f"[Dual-Stream] 🧠 Generated minimal LOGICAL '{lam_filename}'.")
                except Exception:
                    pass

            # ── 3. VERIFICATION ────────────────────────────────────────────────
            if primary_path.exists():
                ASCIIColors.success(f"[Dual-Stream] ✓ Verified: PHYSICAL '{filename}' exists at {primary_path.resolve()}")
            elif not wrote_physical and not logical_content:
                ASCIIColors.error(f"[Dual-Stream] ✗ FAILED: No physical or logical content written for '{title}'!")

        except Exception as e:
            ASCIIColors.warning(f"Failed to sync artifact '{title}' to disk workspace: {e}")

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
        active:            Optional[bool] = None,
        visibility:        Optional[str] = None,
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
        physical_data:     Optional[bytes] = None,  # Raw binary/text for disk (Physical Twin)
        logical_content:   Optional[str] = None,    # .lam metadata for LLM context (Logical Twin)
        **extra_data
    ) -> Dict[str, Any]:
        """
        Adds a new artifact to the discussion using Dual-Stream Storage.

        ARCHITECTURAL RULE: Dual-Stream Storage (.lam Protocol)
        - content (str): Fallback for simple text/code files.
        - physical_data (bytes): The raw binary/text file for tools (CSV, DB, PNG).
        - logical_content (str): The .lam metadata (Schema, Stats) for the LLM.
          If provided, this is injected into context INSTEAD of 'content'.
        """
        # 🛡️ SOVEREIGN TYPE-SAFETY OVERRIDE: Prevent LLM from misclassifying text/code files as structured binary datasets
        file_ext = extra_data.get("file_ext") or Path(title).suffix
        file_ext_clean = (file_ext or "").lower()
        if file_ext_clean == ".sql" or title.lower().endswith(".sql"):
            artefact_type = ArtefactType.CODE
            language = "sql"
        elif file_ext_clean in (".py", ".js", ".ts", ".html", ".css", ".sh", ".bash") or any(title.lower().endswith(ext) for ext in (".py", ".js", ".ts", ".html", ".css", ".sh", ".bash")):
            artefact_type = ArtefactType.CODE
            if not language:
                language = file_ext_clean.replace(".", "")

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
                a['visibility'] = ArtefactVisibility.HIDDEN

        # Determine visibility based on active flag / fallback parameters
        resolved_visibility = visibility
        if not resolved_visibility:
            if active is not None:
                resolved_visibility = ArtefactVisibility.FULL if active else ArtefactVisibility.HIDDEN
            else:
                resolved_visibility = ArtefactVisibility.FULL  # default to active/full

        imgs  = images or []
        mtypes = image_media_types or []
        # Pad / extend media_types to match image count
        if len(mtypes) < len(imgs):
            mtypes = mtypes + ["image/jpeg"] * (len(imgs) - len(mtypes))

        now = datetime.utcnow().isoformat()
        token_count = self._discussion.lollmsClient.count_tokens(content) if (self._discussion and self._discussion.lollmsClient) else len(content) // 4
        new_artefact: Dict[str, Any] = {
            "id":               str(uuid.uuid4()),
            "title":            title,
            "type":             artefact_type,
            "version":          version,
            "content":          content,
            "token_count":      token_count,  # Cache token length
            "images":           imgs,
            "image_media_types": mtypes,
            "audios":           audios or [],
            "videos":           videos or [],
            "zip":              zip_content,
            "language":         language,
            "url":              url,
            "tags":             tags or [],
            "visibility":       resolved_visibility,
            "active":           (resolved_visibility == ArtefactVisibility.FULL),
            "created_at":       now,
            "updated_at":       now,
            "commit_message":   commit_message,
            "version_tags":     version_tags or [],
            **extra_data,
        }
        artefacts.append(new_artefact)
        self._save_all(artefacts)

        # ── DUAL-STREAM SYNC PROTOCOL ─────────────────────────────────────────
        # 1. Determine Physical Data to Write
        final_physical_data = physical_data

        # If no physical data provided, but content exists and is not a schema description
        if final_physical_data is None and content:
            # For Code/Text files, Physical == Logical
            if artefact_type not in (ArtefactType.DATA, ArtefactType.IMAGE, ArtefactType.DOCUMENT):
                final_physical_data = content.encode('utf-8', errors='ignore')
            # For Data files, if content looks like raw data (not schema), encode it
            elif artefact_type == ArtefactType.DATA and not content.strip().startswith("# Data Interface:"):
                final_physical_data = content.encode('utf-8', errors='ignore')

        # 2. Sync to disk workspace IMMEDIATELY so tools can access the file
        self._sync_to_disk_workspace(
            title, 
            content, 
            version, 
            artefact_type, 
            language, 
            extra_data.get("file_ext"),
            physical_data=final_physical_data,
            logical_content=logical_content  # Pass the .lam content
        )
        # Use custom workspace_path if provided, otherwise default to ./data_workspace
        workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")

        # Try to override with server's APP_WORKSPACE_DIR only if no custom workspace_path was set
        if not self._discussion.workspace_path:
            try:
                from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                if awd is not None:
                    workspace_dir = awd
            except ImportError:
                pass

        filename = self._get_filename_with_ext(title, artefact_type, language, extra_data.get("file_ext"))
        expected_path = workspace_dir / self._discussion.id / filename

        if expected_path.exists():
            ASCIIColors.success(f"[ArtefactManager.add] Verified artifact file exists on disk: {expected_path}")
        else:
            if artefact_type != ArtefactType.DATA or not content.strip().startswith("# Data Interface:"):
                ASCIIColors.error(f"[ArtefactManager.add] WARNING: Artifact file was NOT written to disk! Expected: {expected_path}")
            else:
                ASCIIColors.info(f"[ArtefactManager.add] Skipped disk write for schema-only data artifact '{title}'.")

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
        visibility:        Optional[str] = None,
        commit_message:    Optional[str] = None,
        version_tags:      Optional[List[str]] = None,
        version:           Optional[int] = None,
        **extra_data
    ) -> Optional[Dict[str, Any]]:
        latest = self.get(title)
        if latest is None:
            raise ValueError(f"Cannot update non-existent artefact '{title}'.")

        if new_type is None and "artefact_type" in extra_data:
            new_type = extra_data.pop("artefact_type")
        else:
            extra_data.pop("artefact_type", None)

        if version is not None:
            new_version = version
        else:
            new_version  = (latest.get('version', 1) + 1) if bump_version else latest.get('version', 1)

        # Map visibility parameters
        resolved_visibility = visibility or latest.get("visibility")
        if not resolved_visibility:
            if active is not None:
                resolved_visibility = ArtefactVisibility.FULL if active else ArtefactVisibility.HIDDEN
            else:
                resolved_visibility = latest.get("visibility") or ArtefactVisibility.FULL

        if active is not None:
            resolved_visibility = ArtefactVisibility.FULL if active else ArtefactVisibility.HIDDEN

        target_title = new_title if new_title else title

        internal_keys = {
            "id", "title", "type", "version", "content", "images", "image_media_types",
            "audios", "videos", "zip", "language", "url", "tags", "active", "visibility",
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
            try:
                # Remove unversioned copy of old file from working directory
                workspace_dir = Path("./data_workspace")
                try:
                    from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                    if awd is not None:
                        workspace_dir = awd
                except ImportError:
                    pass
                old_filename = self._get_filename_with_ext(title, latest.get('type'), latest.get('language'), latest.get('file_ext'))
                old_path = workspace_dir / "discussions" / self._discussion.id / old_filename
                if old_path.exists():
                    old_path.unlink()
            except Exception as e:
                ASCIIColors.warning(f"Failed to clear old renamed file from workspace: {e}")

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
            visibility        = resolved_visibility,
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

        art = self.get(title, version)
        if art:
            try:
                # Use custom workspace_path if provided, otherwise default to ./data_workspace
                workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")

                # Try to override with server's APP_WORKSPACE_DIR only if no custom workspace_path was set
                if not self._discussion.workspace_path:
                    try:
                        from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                        if awd is not None:
                            workspace_dir = awd
                    except ImportError:
                        pass
                del_filename = self._get_filename_with_ext(title, art.get('type'), art.get('language'), art.get('file_ext'))
                del_path = workspace_dir / self._discussion.id / del_filename
                if del_path.exists():
                    del_path.unlink()
            except Exception as e:
                ASCIIColors.warning(f"Failed to delete working file on remove: {e}")

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
            # Use custom workspace_path if provided, otherwise default to ./data_workspace
            workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")

            # Try to override with server's APP_WORKSPACE_DIR only if no custom workspace_path was set
            if not self._discussion.workspace_path:
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
                    versioned_file = workspace_dir / self._discussion.id / f"{title}_v{v_num}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)

                    # Check for consolidated bundle DB (only exists on v1 creation)
                    if v_num == 1:
                        bundled_db = workspace_dir / self._discussion.id / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)
            else:
                # Remove specific version only
                if art.get('type') == 'data':
                    # Check for versioned file (e.g., "mydb_v2.db")
                    versioned_file = workspace_dir / self._discussion.id / f"{title}_v{version}{file_ext}"
                    if versioned_file.exists():
                        files_to_delete.append(versioned_file)

                    # Check for consolidated bundle DB (only exists on v1 creation)
                    if version == 1:
                        bundled_db = workspace_dir / self._discussion.id / f"{title}_consolidated.db"
                        if bundled_db.exists():
                            files_to_delete.append(bundled_db)

            # Execute deletion with safety checks
            for file_path in set(files_to_delete):  # Use set to avoid duplicates
                try:
                    # Safety check: ensure we're only deleting data workspace files
                    resolved = file_path.resolve()
                    ws_resolved = workspace_dir.resolve()

                    # Explicit path verification guard
                    if resolved.parts[:len(ws_resolved.parts)] != ws_resolved.parts:
                        ASCIIColors.warning(f"[ArtefactCleanup] Skipping deletion of {file_path} - outside workspace boundary")
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
            active_art = self.get(title, version)
            if active_art:
                self._sync_to_disk_workspace(
                    title=active_art["title"],
                    content=active_art["content"],
                    version=active_art["version"],
                    atype=active_art["type"],
                    language=active_art.get("language"),
                    file_ext=active_art.get("file_ext")
                )

    def sync_all_active_to_disk(self):
        """Ensures a physical copy of all active/loaded artifacts exists in the DISCUSSION-SOLATED workspace."""
        active_arts = self.list(active_only=True)
        if not active_arts:
            ASCIIColors.info("[ArtefactManager] No active artifacts to sync")
            return

        # CRITICAL FIX: Resolve the DISCUSSION-SPECIFIC workspace directory
        # Use custom workspace_path if provided, otherwise default to ./data_workspace
        base_workspace_dir = Path(self._discussion.workspace_path) if self._discussion.workspace_path else Path("./data_workspace")

        # Try to override with server's APP_WORKSPACE_DIR only if no custom workspace_path was set
        if not self._discussion.workspace_path:
            try:
                from lollms_client.app.server import APP_WORKSPACE_DIR as awd
                if awd is not None:
                    base_workspace_dir = awd
            except ImportError:
                pass

        disc_id = self._discussion.id
        workspace_dir = base_workspace_dir / disc_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        workspace_dir = workspace_dir.resolve()

        ASCIIColors.info(f"[ArtefactManager] ════════════════════════════════════════════════════")
        ASCIIColors.info(f"[ArtefactManager] SYNCING ARTIFACTS TO DISK (Discussion: {disc_id})")
        ASCIIColors.info(f"[ArtefactManager] ════════════════════════════════════════════════════")
        ASCIIColors.info(f"[ArtefactManager] Workspace directory: {workspace_dir}")
        ASCIIColors.info(f"[ArtefactManager] Artifacts to sync ({len(active_arts)}): {[a['title'] for a in active_arts]}")

        synced_files = []
        for art in active_arts:
            ASCIIColors.info(f"[ArtefactManager] ── Syncing: {art['title']} (type={art.get('type')}, ext={art.get('file_ext', 'N/A')})")

            # For DATA type artifacts, sync the actual file from versions folder within discussion workspace
            if art.get("type") == "data":
                file_ext = art.get("file_ext", "")
                version = art.get("version", 1)
                title = art["title"]
                art_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, title.replace("workspace/", "").replace("data_workspace/", "").replace("/", "_").replace("\\", "_")))

                # Check for versioned data file in discussion versions folder
                versioned_data = workspace_dir / "artefacts_metadata" / art_id / f"{title.replace('.csv','')}_v{version}{file_ext}"
                if versioned_data.exists():
                    dest = workspace_dir / "workspace_data" / f"{title}{file_ext}"
                    import shutil
                    shutil.copy(str(versioned_data), str(dest))
                    ASCIIColors.success(f"[ArtefactManager] ✓ Synced data file: {title}{file_ext}")
                    synced_files.append(str(dest.resolve()))
                else:
                    # Try unversioned copy in discussion workspace_data
                    unversioned = workspace_dir / "workspace_data" / f"{title}{file_ext}"
                    if unversioned.exists():
                        ASCIIColors.info(f"[ArtefactManager] ✓ Found unversioned data file: {title}{file_ext}")
                        synced_files.append(str(unversioned.resolve()))
                    else:
                        ASCIIColors.warning(f"[ArtefactManager] ✗ Data file not found for {title}{file_ext}")
                        if art.get("content"):
                            ASCIIColors.info(f"[ArtefactManager]   → Artifact has content, will sync as text file")

            # Sync the artifact content (for code, documents, etc.) to discussion workspace
            self._sync_to_disk_workspace(
                title=art["title"],
                content=art["content"],
                version=art["version"],
                atype=art["type"],
                language=art.get("language"),
                file_ext=art.get("file_ext")
            )

            # Track synced file (now in discussion folder)
            filename = self._get_filename_with_ext(art["title"], art["type"], art.get("language"), art.get("file_ext"))
            expected_path = workspace_dir / filename
            if expected_path.exists():
                synced_files.append(str(expected_path.resolve()))

        # Final verification: list all files in DISCUSSION workspace
        ASCIIColors.info(f"[ArtefactManager] ────────────────────────────────────────────────────")
        if workspace_dir.exists():
            workspace_files = list(workspace_dir.iterdir())
            subdirs = {}
            for item in workspace_dir.iterdir():
                if item.is_dir():
                    subfiles = list(item.iterdir())
                    if subfiles:
                        subdirs[item.name] = [f.name for f in subfiles]

            ASCIIColors.success(f"[ArtefactManager] ✓ Workspace directory: {workspace_dir}")
            ASCIIColors.success(f"[ArtefactManager] ✓ Workspace ROOT files ({len(workspace_files)}): {[f.name for f in workspace_files]}")
            if subdirs:
                for subdir, files in subdirs.items():
                    ASCIIColors.success(f"[ArtefactManager]   Subdir '{subdir}/' ({len(files)} files): {files}")

            ASCIIColors.success(f"[ArtefactManager] ✓ SYNCED FILES ({len(synced_files)}):")
            for f in synced_files:
                ASCIIColors.success(f"[ArtefactManager]     → {f}")
        ASCIIColors.info(f"[ArtefactManager] ════════════════════════════════════════════════════")

        return workspace_dir, synced_files

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
                a['visibility'] = ArtefactVisibility.FULL if state else ArtefactVisibility.HIDDEN
                a['active'] = state
                changed = True
        if changed:
            self._save_all(artefacts)
            if state:
                active_art = self.get(title, version)
                if active_art:
                    self._sync_to_disk_workspace(
                        title=active_art["title"],
                        content=active_art["content"],
                        version=active_art["version"],
                        atype=active_art["type"],
                        language=active_art.get("language"),
                        file_ext=active_art.get("file_ext")
                    )

    def set_visibility(self, title: str, visibility: str, version: Optional[int] = None) -> bool:
        """Sets the exact visibility tier of an artifact."""
        if visibility not in (ArtefactVisibility.HIDDEN, ArtefactVisibility.TREE_LOCKED, ArtefactVisibility.TREE_UNLOCKABLE, ArtefactVisibility.METADATA, ArtefactVisibility.FULL):
            raise ValueError(f"Invalid visibility tier: '{visibility}'")
        artefacts = self._get_all_raw()
        changed = False
        for a in artefacts:
            if a.get('title') == title and (version is None or a.get('version') == version):
                a['visibility'] = visibility
                a['active'] = (visibility == ArtefactVisibility.FULL)
                changed = True
        if changed:
            self._save_all(artefacts)
            return True
        return False

    # --------------------------------------------------------- context zone

    def build_artefacts_context_zone(self) -> str:
        """
        Assembles the comprehensive multi-tier workspace context zone using Dual-Stream (.lam) logic.

        LOGICAL TWIN PRIORITY:
        - For Data/Binary files, we load the .lam content from the artifact's 'content' field (which holds the schema).
        - For Code/Text files, we load the raw content.
        """
        all_raw = self._get_all_raw()

        visible_artifacts = [
            a for a in all_raw 
            if a.get("visibility", ArtefactVisibility.HIDDEN) != ArtefactVisibility.HIDDEN 
            and not a.get("title", "").endswith("::images")
        ]

        if not visible_artifacts:
            return ""

        # 🛑 DEBUG DUMP: Log exactly what the LLM sees for Data artifacts
        for a in visible_artifacts:
            if a.get("type") == "data":
                title = a.get("title", "UNNAMED")
                file_ext = a.get("file_ext", "")
                # Construct the expected filename
                expected_filename = title if title.endswith(file_ext) else f"{title}{file_ext}"
                ASCIIColors.cyan(f"[ContextDebug] Artifact '{title}' (ext: {file_ext}) → LLM sees filename: '{expected_filename}'")
                ASCIIColors.cyan(f"[ContextDebug]   Raw DB Record: title='{title}', file_ext='{file_ext}'")

        # ── 1. COMPILE WORKSPACE DIRECTORY TREE INDEX ─────────────────────────
        tree_lines = ["  workspace/"]
        root_node = {"files": [], "folders": {}}
        for a in visible_artifacts:
            # 🛡️ ENSURE FILENAME CLARITY: Use title directly (which should now include extension)
            # If title is missing extension, we append it here for display ONLY
            display_title = a["title"]
            if a.get("type") == "data" and a.get("file_ext"):
                if not display_title.lower().endswith(a["file_ext"].lower()):
                    display_title = f"{display_title}{a['file_ext']}"

            parts = display_title.split("/")
            curr = root_node
            for i in range(len(parts) - 1):
                folder = parts[i]
                curr = curr["folders"].setdefault(folder, {"files": [], "folders": {}})
            curr["files"].append({**a, "display_title": display_title})

        def _traverse_tree_prompt(node, depth=1):
            lines = []
            indent = "  " * depth
            for f_name, f_node in sorted(node["folders"].items()):
                lines.append(f"{indent}├── {f_name}/")
                lines.extend(_traverse_tree_prompt(f_node, depth + 1))
            for a in sorted(node["files"], key=lambda x: x["display_title"]):
                v_tier = a.get("visibility", ArtefactVisibility.FULL)
                marker = "[L]"
                if v_tier == ArtefactVisibility.FULL: marker = "[C]"
                elif v_tier == ArtefactVisibility.METADATA: marker = "[M]"
                elif v_tier == ArtefactVisibility.TREE_UNLOCKABLE: marker = "[U]"
                f_name = a["display_title"].split("/")[-1]
                lines.append(f"{indent}├── {f_name}  {marker}")
            return lines

        tree_lines.extend(_traverse_tree_prompt(root_node))
        tree_text = "\n".join(tree_lines)

        # ── 2. COMPILE FULL CONTENT & SIGNATURES SECTIONS ─────────────────────
        full_visible_parts = []
        partial_visible_parts = []

        for item in visible_artifacts:
            v_tier = item.get("visibility", ArtefactVisibility.FULL)
            atype = item.get("type", ArtefactType.DOCUMENT)
            lang = item.get("language") or ""
            fence = f"```{lang}" if lang else "```"

            # 🛡️ ENSURE FILENAME CLARITY IN HEADERS
            # Construct the filename that the LLM should use in tool calls
            display_title = item["title"]
            if atype == ArtefactType.DATA and item.get("file_ext"):
                if not display_title.lower().endswith(item["file_ext"].lower()):
                    display_title = f"{display_title}{item['file_ext']}"

            if v_tier == ArtefactVisibility.FULL:
                content_text = item.get("content", "").strip()
                if not content_text:
                    continue

                header = f"### [Full File: '{display_title}']"
                deactivated_contents = getattr(self._discussion, "deactivated_contents", set())

                if display_title in deactivated_contents:
                    seq_summaries = getattr(self._discussion, "sequential_summaries", {})
                    summary_text = seq_summaries.get(display_title, "Detailed summary not available.")
                    full_visible_parts.append(f"{header}\n[SEQUENTIAL COMPRESSED DATA DUE TO BUDGET CONSTRAINTS]:\n{summary_text}")
                else:
                    # ── DUAL-STREAM LOGIC: Prioritize Logical Twin (.lam) for Data ──
                    if atype == ArtefactType.DATA:
                        # 'content' field ALREADY holds the .lam schema/stats (Logical Twin)
                        # We do NOT want to show raw CSV rows here.
                        schema_desc = content_text
                        # Prepend explicit filename instruction for Data types
                        schema_desc = f"**FILENAME FOR TOOLS:** `{display_title}`\n\n{schema_desc}"
                        full_visible_parts.append(f"{header}\n{schema_desc}")
                    elif atype == ArtefactType.CODE or (lang and not _ARTEFACT_IMAGE_TAG_RE.search(content_text)):
                        full_visible_parts.append(f"{header}\n{fence}\n{content_text}\n```")
                    else:
                        full_visible_parts.append(f"{header}\n{content_text}")

            elif v_tier == ArtefactVisibility.METADATA:
                partial_desc = _extract_file_metadata_and_signature(item)
                if partial_desc:
                    partial_visible_parts.append(partial_desc)

        # ── 3. ASSEMBLE SYSTEM CONTEXT ZONE ───────────────────────────────────
        context_parts = [
            "## Workspace Directory Tree Index",
            "This index displays all files in your workspace directory tree with their context state markers:",
            "  - [C] Fully Loaded in Context (Verbatim text/code is read-ready below)",
            "  - [M] Signature / Metadata Only (Exposes schemas, layouts, or code signatures below)",
            "  - [U] Inactive/Unlockable (Excluded from context, but you can unlock/load it to [C] by calling <add_files_to_context>)",
            "  - [L] Locked in Tree (Excluded from context and cannot be unlocked)\n",
            tree_text
        ]

        if partial_visible_parts:
            context_parts.append("\n## Partial File Signatures & Metadata [M]\n" + "\n\n---\n\n".join(partial_visible_parts))

        if full_visible_parts:
            context_parts.append("\n## Fully Loaded File Contents [C]\n" + "\n\n---\n\n".join(full_visible_parts))

        return "\n\n".join(context_parts)

    def get_context_images(self) -> List[Dict[str, Any]]:
        """
        Returns ALL images from active/fully visible artefacts that have images,
        including companion images artefacts.
        """
        result: List[Dict[str, Any]] = []
        raw_list = self._get_all_raw()
        by_title = {a.get('title'): a for a in raw_list}

        for a in raw_list:
            # For image-type artifacts, we only load pixels into the vision context when visibility is set to FULL
            if a.get('visibility', ArtefactVisibility.FULL) != ArtefactVisibility.FULL:
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

            # Retrieve from companion images artefact (e.g., "title::images") if present
            comp_title = f"{a.get('title')}::images"
            if comp_title in by_title:
                comp_a = by_title[comp_title]
                if comp_a.get('visibility', ArtefactVisibility.FULL) == ArtefactVisibility.FULL:
                    comp_imgs = comp_a.get('images') or []
                    comp_mtypes = comp_a.get('image_media_types') or []
                    for idx, img_b64 in enumerate(comp_imgs):
                        if not img_b64:
                            continue
                        mtype = comp_mtypes[idx] if idx < len(comp_mtypes) else "image/jpeg"
                        result.append({
                            "id":         make_image_id(comp_title, idx),
                            "data":       img_b64,
                            "media_type": mtype,
                            "title":      comp_title,
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
        already_processed_artifacts: Optional[List[str]] = None,
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
            r'^[ \t]*<art[ei]fact\s([^>]*)>(.*?)</art[ei]fact>',
            re.DOTALL | re.IGNORECASE | re.MULTILINE
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
            is_ephemeral = attrs.pop('ephemeral', 'false').lower() in ('true', '1', 'yes')

            commit_message = attrs.pop('commit_message', None)
            version_tags_raw = attrs.pop('version_tags', None)
            version_tags = [t.strip().lower() for t in version_tags_raw.split(',') if t.strip()] if version_tags_raw else None

            # Strip any image-related attrs the LLM might have hallucinated
            attrs.pop('images', None)
            attrs.pop('image_media_types', None)

            if atype not in ArtefactType.ALL:
                if atype in ("csv", "tsv", "excel", "xlsx", "xls", "db", "sqlite"):
                    atype = ArtefactType.DATA
                elif atype in ("text", "txt", "markdown", "md"):
                    atype = ArtefactType.DOCUMENT
                else:
                    atype = default_type

            resolved_title = tag_title if tag_title in existing_titles else (
                _find_best_title_match(tag_title, existing_titles) or tag_title
            )

            # Skip if already processed during streaming
            if already_processed_artifacts and resolved_title in already_processed_artifacts:
                ASCIIColors.info(f"Skipping post-processing for already applied artifact: '{resolved_title}'")
                art = self.get(resolved_title)
                if art:
                    affected.append(art)
                tag_name = "artifact" if "artifact" in match.group(0).lower() else "artefact"
                return f'<{tag_name} {match.group(1)}>\n[content stripped, refer to the artefact for details]\n</{tag_name}>'
            
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
                        ephemeral=is_ephemeral,
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
                            ephemeral=is_ephemeral,
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
                        ephemeral=is_ephemeral,
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
                        ephemeral=is_ephemeral,
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

            tag_name = "artifact" if "artifact" in match.group(0).lower() else "artefact"
            return f'<{tag_name} {match.group(1)}>\n[content stripped, refer to the artefact for details]\n</{tag_name}>'

        cleaned = artefact_pattern.sub(handle_artefact, cleaned)

        revert_pattern = re.compile(r'^[ \t]*<revert_art[ei]fact\s([^>]+)/?>', re.IGNORECASE | re.MULTILINE)

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

    def import_file(self, file_path: Union[str, Path], title: Optional[str] = None, active: bool = True) -> Dict[str, Any]:
        """
        HIGH-LEVEL API: Imports a file from disk into the discussion as a typed artifact.

        ARCHITECTURAL RULE: The library handles ALL dual-stream complexity.
        The caller simply provides a path. The library:
        1. Detects type (Code, Data, Image, Document).
        2. Reads Physical Bytes (for tools/disk).
        3. Generates Logical Schema/Content (for LLM context).
        4. Registers the artifact with both streams.

        Args:
            file_path: Path to the file on disk.
            title: Optional custom title. Defaults to filename.
            active: Whether to activate the artifact immediately.

        Returns:
            The created artifact dictionary.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if title is None:
            title = path.stem

        ext = path.suffix.lower()

        # 1. Determine Artifact Type
        artefact_type = ArtefactType.DOCUMENT
        if ext in (".csv", ".tsv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3", ".sqlconn", ".ttl", ".rdf", ".parquet"):
            artefact_type = ArtefactType.DATA
        elif ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
            artefact_type = ArtefactType.CODE
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
            artefact_type = ArtefactType.IMAGE
        elif ext in (".md", ".txt", ".log", ".json", ".yaml", ".yml", ".xml"):
            artefact_type = ArtefactType.DOCUMENT

        # 2. Prepare Physical Data (Bytes)
        try:
            physical_data = path.read_bytes()
        except Exception as e:
            ASCIIColors.error(f"Failed to read physical bytes for {title}: {e}")
            physical_data = None

        # 3. Prepare Logical Content (Schema/Text)
        content = ""
        if artefact_type == ArtefactType.DATA:
            # Use internal parser to generate schema/stats
            try:
                from ._data_files import _parse_data_file
                schema_result = _parse_data_file(path, title, version=1)
                # Handle both 2-tuple (old) and 3-tuple (new) returns
                if len(schema_result) == 3:
                    content, _, _ = schema_result
                else:
                    content, _ = schema_result
            except Exception as e:
                content = f"# Data Interface: {title}\n⚠️ Failed to parse schema: {e}"
        elif artefact_type == ArtefactType.IMAGE:
            content = f"### Image: `{title}{ext}`\n\n<artefact_image id=\"{title}{ext}::0\" />"
        elif artefact_type in (ArtefactType.CODE, ArtefactType.DOCUMENT):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                content = f"⚠️ Failed to read text content: {e}"

        # 4. Register with Dual-Stream
        art = self.add(
            title=title + ext,
            artefact_type=artefact_type,
            content=content,
            language=ext.replace(".", "") if artefact_type == ArtefactType.CODE else None,
            active=active,
            file_ext=ext,
            physical_data=physical_data
        )

        # Commit discussion if attached
        if self._discussion:
            self._discussion.commit()

        return art

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
