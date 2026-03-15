# lollms_discussion/_mixin_prompt.py
# PromptMixin: system-prompt instruction builders and LLM response post-processor.

import re
import uuid
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ascii_colors import ASCIIColors
from ._artefacts import ArtefactType

if TYPE_CHECKING:
    from ._message import LollmsMessage


class PromptMixin:
    """
    Builds artefact / image-generation / inline-widget instructions for the
    system prompt and post-processes the LLM response to apply any XML action
    tags it contains.
    """

    # ─────────────────────────────────────── instruction builders ────────────

    def _build_artefact_instructions(self) -> str:
        """Returns prompt instructions for creating/updating artefacts via XML tags."""
        lines = [
            "",
            "=== ARTEFACT SYSTEM ===",
            "You can create, modify, and version control artefacts (persistent code, documents, notes, skills).",
            "Always use XML tags for artefact operations. The system keeps full version history automatically.",
            "",
            "To CREATE a new artefact (supports metadata attributes):",
            '<artefact name="unique_name.py" type="code" language="python" author="Your Name" description="What it does">',
            "[full content here]",
            "</artefact>",
            "",
            "To UPDATE an existing artefact, use SEARCH/REPLACE blocks inside the tag.",
            "CRITICAL RULES FOR SEARCH/REPLACE:",
            "  1. Keep each SEARCH block as SHORT as possible — 1 to 5 lines is ideal.",
            "     The longer the SEARCH block, the higher the chance of a whitespace or",
            "     punctuation mismatch that causes the patch to fail.",
            "  2. You may include MULTIPLE SEARCH/REPLACE blocks inside a single <artefact>",
            "     tag to make several independent edits in one turn.",
            "  3. Each block MUST use this exact structure — do not skip or reorder markers:",
            "       <<<<<<< SEARCH",
            "       [exact lines to find — copy verbatim from the artefact]",
            "       =======",
            "       [replacement lines]",
            "       >>>>>>> REPLACE",
            "  4. The SEARCH text must match the artefact content CHARACTER FOR CHARACTER",
            "     (spaces, indentation, blank lines). When in doubt, use fewer lines.",
            "  5. Do NOT place the replacement lines where ======= should be.",
            "     Do NOT omit the >>>>>>> REPLACE marker at the end.",
            "  6. If you need to change many scattered parts, prefer many small blocks",
            "     over one large block.",
            "",
            "Example — two independent edits in one tag:",
            '<artefact name="app.py">',
            "<<<<<<< SEARCH",
            "def greet():",
            "    return 'hello'",
            "=======",
            "def greet(name: str = 'world'):",
            "    return f'hello {name}'",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "PORT = 8000",
            "=======",
            "PORT = int(os.getenv('PORT', 8000))",
            ">>>>>>> REPLACE",
            "</artefact>",
            "",
            "To REVERT to a previous known version:",
            '<revert_artefact name="existing_name.py" version="1" />',
            "",
            f"Supported types: {', '.join(sorted(list(ArtefactType.ALL)))}",
            "=== END ARTEFACT INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_image_generation_instructions(self) -> str:
        """Returns prompt instructions for image generation / editing."""
        tti = getattr(self.lollmsClient, 'tti', None)
        if tti is None:
            return ""
        lines = [
            "",
            "=== IMAGE GENERATION / EDITING ===",
            "You can generate or edit images using the following XML tags.",
            "",
            "To generate a new image:",
            '<generate_image width="512" height="512">',
            "  A detailed description of the image you want to generate",
            "</generate_image>",
            "",
            "To edit an existing image artefact:",
            '<edit_image name="artefact_name">',
            "  Description of how to edit / modify the image",
            "</edit_image>",
            "=== END IMAGE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_inline_widget_instructions(self) -> str:
        """Returns prompt instructions for generating inline interactive widgets."""
        lines = [
            "",
            "=== INLINE INTERACTIVE WIDGETS ===",
            "You can embed a live, interactive widget directly inside your reply.",
            "Use this whenever an interactive visual would explain your answer better than text.",
            "",
            "Tag syntax:",
            '<lollms_inline type="html" title="Descriptive title">',
            "  <!-- self-contained HTML/JS/CSS — no external files, CDN links are OK -->",
            "</lollms_inline>",
            "",
            "Supported types: html (default), react, svg",
            "",
            "Good uses: formula explorers with sliders, mini charts, SVG animations,",
            "           physics / math simulations, colour pickers, interactive diagrams.",
            "Bad uses: full applications (use <artefact> instead), anything needing a server.",
            "",
            "Rules:",
            "  • Fully self-contained — the widget must work with zero network requests",
            "    (CDN links to well-known libraries are the only exception).",
            "  • Keep it compact: aim for ≤ 420 px height, full container width.",
            "  • Use KaTeX (https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js) for math.",
            "  • No alert() / confirm() / prompt().",
            "=== END INLINE WIDGET INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_note_instructions(self) -> str:
        """
        Returns prompt instructions for creating persistent notes via the <note> tag.

        Notes are lightweight named documents saved as ArtefactType.NOTE artefacts.
        They are ideal for structured summaries, analysis results, comparison tables,
        research findings, or any content the user explicitly wants saved and retrievable
        later — as opposed to artefacts (code, full documents) or inline widgets (live UI).

        Tag anatomy
        -----------
        <note title="Human-readable title">
          [note content — plain text or Markdown]
        </note>

        Rules
        -----
        1. Use a clear, specific title that describes the content (not "Note 1").
        2. A single response may contain multiple <note> tags with different titles.
        3. If a note with the same title already exists it is replaced (new version created).
        4. Notes are immediately visible in the discussion artefact panel.
        5. Do NOT use <note> for code — use <artefact type="code"> instead.
        6. Do NOT use <note> for large structured documents — use <artefact type="document">.
        """
        lines = [
            "",
            "=== NOTE SYSTEM ===",
            "You can save structured notes directly from your response.",
            "Notes are persistent, named documents stored alongside the discussion.",
            "",
            "Use <note> for: summaries, analysis results, comparison tables, key findings,",
            "  action item lists, or any content the user would want to save and reference.",
            "Do NOT use <note> for code (use <artefact type=\"code\">) or large documents.",
            "",
            "Tag syntax:",
            '<note title="Clear descriptive title">',
            "  [note content — plain text or Markdown]",
            "</note>",
            "",
            "You may emit multiple notes in a single response:",
            '<note title="Transavia - Price Analysis">',
            "  | Route | Base | +Baggage | Total |",
            "  |-------|------|----------|-------|",
            "  | TUN→LYS | €89 | €35 | €124 |",
            "</note>",
            '<note title="easyJet - Price Analysis">',
            "  | Route | Base | +Baggage | Total |",
            "  |-------|------|----------|-------|",
            "  | TUN→LYS | €95 | €25 | €120 |",
            "</note>",
            "=== END NOTE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_skill_instructions(self) -> str:
        """
        Returns prompt instructions for saving reusable skills via the <skill> tag.

        Skills are reusable knowledge capsules — code patterns, workflows, techniques,
        or domain recipes — that the user or the system can retrieve in future sessions.
        They are saved as ArtefactType.SKILL artefacts with rich metadata.

        Tag anatomy
        -----------
        <skill title="Skill name"
               description="One-sentence description of what this teaches"
               category="domain/subdomain/topic">
          [skill content — Markdown with explanations and code examples]
        </skill>

        Category convention
        -------------------
        Use forward-slash-separated hierarchical labels:
          programming/python/async
          language/french/subjunctive
          cooking/baking/bread
          devops/docker/networking

        Rules
        -----
        1. Only emit a <skill> tag when the user explicitly asks to save a skill,
           or when the response encapsulates a reusable, teachable pattern.
        2. Skills should be self-contained — someone reading only the skill content
           should be able to apply the technique without additional context.
        3. Include concrete examples; avoid vague advice.
        4. Do NOT use <skill> for one-off answers — use it for genuinely reusable knowledge.
        """
        lines = [
            "",
            "=== SKILL SYSTEM ===",
            "You can save reusable knowledge capsules as skills.",
            "Skills are retrieved automatically in future sessions when relevant.",
            "",
            "Emit a <skill> tag ONLY when:",
            "  • The user explicitly asks to save this as a skill / learn this pattern.",
            "  • Your response encapsulates a reusable, teachable technique.",
            "",
            "Tag syntax:",
            '<skill title="Skill Name"',
            '       description="One-sentence description of what this teaches"',
            '       category="domain/subdomain/topic">',
            "  [skill content — Markdown with explanation and concrete examples]",
            "</skill>",
            "",
            "Category examples: programming/python/async  |  language/french/grammar",
            "                   cooking/baking/sourdough  |  devops/kubernetes/networking",
            "=== END SKILL INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────── LLM response post-processor ─────────

    def _post_process_llm_response(
        self,
        text: str,
        ai_message: 'LollmsMessage',
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:   bool = True,
        enable_notes:            bool = True,
        enable_skills:           bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Scans the raw LLM response for XML action tags and applies them.

        This is the *single* place where all LLM-output XML is intercepted.
        Called at the end of ``chat()`` / ``simplified_chat()``, after
        ``remove_thinking_blocks`` has run, and *after* the ai_message object
        has been created (so images / inline widgets can be attached directly).

        Handled tags
        ------------
        ``<artefact name="…" type="…" language="…">…</artefact>``
            Create or patch (aider SEARCH/REPLACE) a named artefact.

        ``<generate_image width="…" height="…">prompt</generate_image>``
            Calls ``lollmsClient.tti.generate()`` and attaches the result
            to *ai_message* via ``add_image_pack``.
            Only when ``enable_image_generation=True`` and tti is set.

        ``<edit_image name="…">edit prompt</edit_image>``
            Calls ``lollmsClient.tti.edit()`` on a source image.
            Only when ``enable_image_editing=True`` and tti is set.

        ``<lollms_inline type="html|react|svg" title="…">…</lollms_inline>``
            Extracts a self-contained widget and stores it in
            ``ai_message.metadata["inline_widgets"]``.
            Only when ``enable_inline_widgets=True``.

        ``<note title="…">…</note>``
            Saves a persistent named note as an ``ArtefactType.NOTE`` artefact.
            Active immediately; visible in the artefact panel.
            Only when ``enable_notes=True``.

        ``<skill title="…" description="…" category="…">…</skill>``
            Saves a reusable knowledge capsule as an ``ArtefactType.SKILL``
            artefact with category and description metadata.
            Only when ``enable_skills=True``.

        Returns
        -------
        (cleaned_text, affected_artefacts)
        ``cleaned_text`` has all processed XML tags stripped.
        """
        # ── Mask code blocks so XML inside documentation isn't processed ─────
        code_blocks: Dict[str, str] = {}

        def mask_code_block(match):
            placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
            code_blocks[placeholder] = match.group(0)
            return placeholder

        masked_text = re.sub(r'(`{3,})[\s\S]*?\1', mask_code_block, text)
        masked_text = re.sub(r'`[^`]+`',           mask_code_block, masked_text)

        has_artefact = bool(re.search(r'<(?:revert_)?artefact[\s>]', masked_text, re.IGNORECASE))
        has_gen      = enable_image_generation and bool(
            re.search(r'<generate_image[\s>]', masked_text, re.IGNORECASE))
        has_edit     = enable_image_editing and bool(
            re.search(r'<edit_image[\s>]', masked_text, re.IGNORECASE))
        has_inline   = enable_inline_widgets and bool(
            re.search(r'<lollms_inline[\s>]', masked_text, re.IGNORECASE))
        has_note     = enable_notes and bool(
            re.search(r'<note[\s>]', masked_text, re.IGNORECASE))
        has_skill    = enable_skills and bool(
            re.search(r'<skill[\s>]', masked_text, re.IGNORECASE))

        if not (has_artefact or has_gen or has_edit or has_inline or has_note or has_skill):
            return text, []

        cleaned             = masked_text
        affected_artefacts: List[Dict] = []

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        # ── 1. Artefact create / patch ────────────────────────────────────────
        if has_artefact:
            cleaned, affected_artefacts = self.artefacts._apply_artefact_xml(
                cleaned, auto_activate=auto_activate_artefacts,
                replacements=code_blocks,
            )

        # ── 2. Image generation → message.images ─────────────────────────────
        if has_gen:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<generate_image> found but lollmsClient.tti is None — skipping.")
            else:
                gen_pattern = re.compile(
                    r'<generate_image\s*([^>]*)>(.*?)</generate_image>',
                    re.DOTALL | re.IGNORECASE,
                )

                def handle_generate(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()
                    for placeholder, original in code_blocks.items():
                        prompt = prompt.replace(placeholder, original)
                        for k, v in attrs.items():
                            if placeholder in v:
                                attrs[k] = v.replace(placeholder, original)
                    width  = int(attrs.get('width',  1024))
                    height = int(attrs.get('height', 1024))
                    try:
                        img_bytes = tti.generate_image(
                            prompt=prompt, width=width, height=height)
                        if img_bytes:
                            import base64
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            ai_message.add_image_pack(
                                images=[img_b64],
                                group_type="generated",
                                active_by_default=True,
                                title=attrs.get('name', f'gen_{uuid.uuid4().hex[:6]}'),
                                prompt=prompt,
                            )
                            ASCIIColors.success(
                                f"Generated image ({width}×{height}) added to message.")
                    except Exception as e:
                        ASCIIColors.warning(f"Image generation failed: {e}")
                    return ''

                cleaned = gen_pattern.sub(handle_generate, cleaned)

        # ── 3. Image editing → message.images ────────────────────────────────
        if has_edit:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<edit_image> found but lollmsClient.tti is None — skipping.")
            else:
                edit_pattern = re.compile(
                    r'<edit_image\s*([^>]*)>(.*?)</edit_image>',
                    re.DOTALL | re.IGNORECASE,
                )

                def handle_edit(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()
                    for placeholder, original in code_blocks.items():
                        prompt = prompt.replace(placeholder, original)
                        for k, v in attrs.items():
                            if placeholder in v:
                                attrs[k] = v.replace(placeholder, original)

                    source_b64: Optional[str] = None
                    artefact_name = attrs.get('name', '')
                    if artefact_name:
                        a = self.artefacts.get(artefact_name)
                        if a and a.get('images'):
                            source_b64 = a['images'][-1]
                        else:
                            ASCIIColors.warning(
                                f"<edit_image name='{artefact_name}'> — "
                                "artefact not found or has no images; "
                                "falling back to last message image.")
                    if source_b64 is None:
                        active_imgs = ai_message.get_active_images()
                        if active_imgs:
                            source_b64 = active_imgs[-1]
                    if source_b64 is None:
                        ASCIIColors.warning(
                            "<edit_image> — no source image available; skipping.")
                        return match.group(0)
                    try:
                        img_bytes = tti.edit_image(image=source_b64, prompt=prompt)
                        if img_bytes:
                            import base64
                            edited_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            ai_message.add_image_pack(
                                images=[edited_b64],
                                group_type="edited",
                                active_by_default=True,
                                title=f"edit_{uuid.uuid4().hex[:6]}",
                                prompt=prompt,
                            )
                            ASCIIColors.success("Edited image added to message.")
                    except Exception as e:
                        ASCIIColors.warning(f"Image edit failed: {e}")
                    return ''

                cleaned = edit_pattern.sub(handle_edit, cleaned)

        # ── 4. Inline widgets → message.metadata["inline_widgets"] ───────────
        if has_inline:
            inline_pattern = re.compile(
                r'<lollms_inline\s*([^>]*)>(.*?)</lollms_inline>',
                re.DOTALL | re.IGNORECASE,
            )

            # Ensure the metadata list exists
            meta = dict(ai_message.metadata or {})
            if "inline_widgets" not in meta:
                meta["inline_widgets"] = []

            def handle_inline(match: re.Match) -> str:
                attrs  = _parse_attrs(match.group(1))
                source = match.group(2)

                # Restore any masked code inside the widget source
                for placeholder, original in code_blocks.items():
                    source = source.replace(placeholder, original)

                widget_id   = str(uuid.uuid4())
                widget_type = attrs.get('type', 'html').lower().strip()
                widget_title = attrs.get('title', 'Interactive Widget')

                # Normalise type to one of the three supported values
                if widget_type not in ('html', 'react', 'svg'):
                    widget_type = 'html'

                widget_entry = {
                    "id":     widget_id,
                    "type":   widget_type,
                    "title":  widget_title,
                    "source": source.strip(),
                }
                meta["inline_widgets"].append(widget_entry)

                ASCIIColors.success(
                    f"Inline widget '{widget_title}' ({widget_type}) registered.")

                # Replace the tag in the message text with a lightweight
                # anchor element that the frontend can use to locate and
                # render the widget in-place.
                return f'<lollms_widget id="{widget_id}" />'

            cleaned = inline_pattern.sub(handle_inline, cleaned)

            # Only write back metadata if at least one widget was found
            if meta["inline_widgets"]:
                ai_message.metadata = meta

        # ── 5. Notes → ArtefactType.NOTE artefacts ───────────────────────────
        if has_note:
            note_pattern = re.compile(
                r'<note\s*([^>]*)>(.*?)</note>',
                re.DOTALL | re.IGNORECASE,
            )

            def handle_note(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

                # Restore any masked code blocks inside the note body
                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title = (attrs.get('title') or attrs.get('name') or
                         f'note_{uuid.uuid4().hex[:8]}')

                # Notes are always saved as NOTE type, active by default so
                # they appear immediately in the artefact panel.
                note_artefact = self.artefacts.add(
                    title         = title,
                    artefact_type = ArtefactType.NOTE,
                    content       = content.strip(),
                    active        = auto_activate_artefacts,
                )
                affected_artefacts.append(note_artefact)
                ASCIIColors.success(f"Note '{title}' saved.")
                return ''   # strip tag from visible text

            cleaned = note_pattern.sub(handle_note, cleaned)

        # ── 6. Skills → ArtefactType.SKILL artefacts ─────────────────────────
        if has_skill:
            skill_pattern = re.compile(
                r'<skill\s*([^>]*)>(.*?)</skill>',
                re.DOTALL | re.IGNORECASE,
            )

            def handle_skill(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

                # Restore any masked code blocks inside the skill body
                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title       = (attrs.get('title') or attrs.get('name') or
                               f'skill_{uuid.uuid4().hex[:8]}')
                description = attrs.get('description', '')
                category    = attrs.get('category', '')

                # Skills are saved with description and category as extra metadata
                # so the application layer can index / search them by category.
                skill_artefact = self.artefacts.add(
                    title         = title,
                    artefact_type = ArtefactType.SKILL,
                    content       = content.strip(),
                    active        = auto_activate_artefacts,
                    description   = description,
                    category      = category,
                )
                affected_artefacts.append(skill_artefact)
                ASCIIColors.success(
                    f"Skill '{title}' saved"
                    + (f" [{category}]" if category else "") + "."
                )
                return ''   # strip tag from visible text

            cleaned = skill_pattern.sub(handle_skill, cleaned)

        # ── Unmask code blocks ────────────────────────────────────────────────
        for placeholder, original in code_blocks.items():
            cleaned = cleaned.replace(placeholder, original)

        return cleaned.strip(), affected_artefacts