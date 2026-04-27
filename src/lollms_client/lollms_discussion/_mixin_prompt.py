# lollms_discussion/_mixin_prompt.py
# PromptMixin: system-prompt instruction builders and LLM response post-processor.
#
# This file is identical to the original except _build_artefact_instructions()
# now includes a section explaining the <artefact_image id="TITLE::N" /> anchor
# system so that vision-capable models know how to correlate image slots with
# in-text references.

import re
import uuid
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ascii_colors import ASCIIColors
from ._artefacts import ArtefactType

if TYPE_CHECKING:
    from ._message import LollmsMessage


class PromptMixin:
    """
    Builds artifact / image-generation / inline-widget / form instructions for the
    system prompt and post-processes the LLM response to apply any XML action
    tags it contains.
    """

    # ─────────────────────────────────────── instruction builders ────────────

    def _build_artefact_instructions(self) -> str:
        """Returns the system-prompt instructions for artifact operations."""
        lines = [
            "",
            "=== ARTIFACT SYSTEM ===",
            "",
            "**CRITICAL DISTINCTION — READ THIS FIRST**",
            "",
            "Artifacts and markdown code blocks are **completely different** things:",
            "",
            "• Markdown code blocks (```language ```) = temporary display only",
            "• Artifacts (<artifact> XML tags) = persistent, version-controlled storage managed by the Agent",
            "",
            "Lollms Communication Protocol (LCP) Rules:",
            "1. Artifacts are INTERCEPTED. The user only sees a 'Processing' summary while you write them.",
            "2. Interactive components (widgets/forms) are NOT intercepted. The user sees them as soon as they are complete.",
            "",
            "❌ **NEVER** wrap XML tags in code blocks — it breaks the system permanently:",
            "```xml",
            "<artifact name=\"example.ext\" type=\"type\">",
            "...",
            "</artifact>",
            "```",
            "",
            "✅ **ALWAYS** output the <artifact> tag **directly** as raw XML in your response.",
            "Never wrap it inside any markdown code fence.",
            "",
            "Supported types: " + ", ".join(sorted(list(ArtefactType.ALL))),
            "Always choose the **most accurate type** for the content.",
            "",
            "=== HOW TO USE ARTIFACTS ===",
            "You can create, update, rename, or revert persistent artifacts using XML tags.",
            "",
            "You **MUST** always add a short natural explanation in your reply (outside the tag)",
            "explaining what you created/changed and why.",
            "Never reply with XML tags only.",
            "",
            "── Option A: CREATE or REPLACE (full content)",
            "<artifact name=\"filename.ext\" type=\"appropriate_type\">",
            "Full content goes here...",
            "</artifact>",
            "",
            "── Option B: PATCH (targeted updates)",
            "<artifact name=\"filename.ext\" type=\"appropriate_type\">",
            "<<<<<<< SEARCH",
            "exact old lines here",
            "=======",
            "new lines here",
            ">>>>>>> REPLACE",
            "</artifact>",
            "",
            "── Option C: RENAME + UPDATE",
            "<artifact name=\"new_name.ext\" type=\"appropriate_type\" rename=\"old_name.ext\">",
            "... content or SEARCH/REPLACE blocks ...",
            "</artifact>",
            "",
            "── Option D: REVERT",
            "<artifact name=\"filename.ext\" revert_to=\"v3\" />",
            "",
            "=== ARTIFACT IMAGES ===",
            "",
            "Some artifacts (e.g. PDF documents) contain images embedded in their text.",
            "These images are referenced with self-closing anchor tags:",
            "",
            '    <artefact_image id="TITLE::N" />',
            "",
            "where TITLE is the artifact title and N is the 0-based image index.",
            "",
            "When such artifacts are active, the corresponding images are appended to",
            "the vision context **after** any user-supplied images.",
            "A mapping note in the scratchpad tells you which vision-input slot",
            "corresponds to which anchor id.",
            "",
            "Rules for artifacts with images:",
            "• Do NOT attempt to generate or modify artefact images via XML tags.",
            "  Images are supplied exclusively by the application layer.",
            "• When you see an <artefact_image id=\"...\" /> anchor in the artifact text,",
            "  look at the corresponding image slot in the vision input to understand",
            "  what that part of the document looks like.",
            "• You may reference an anchor in your reply to point the user to a specific",
            "  image, e.g.: 'As shown in <artefact_image id=\"my_doc::2\" />, ...'",
            "• When patching an artifact that contains image anchors, preserve the",
            "  anchor tags unchanged — do not remove or alter them.",
            "",
            "=== REMINDER ===",
            "→ Artifacts = persistent & versioned",
            "→ Markdown code blocks = temporary display only",
            "→ Never nest <artifact> inside ``` blocks",
            "→ Always choose the correct type",
            "→ Always add a human explanation outside the tag",
            "→ Preserve <artefact_image> anchors when patching image-bearing artifacts",
            "",
            "=== END ARTIFACT SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_image_generation_instructions(self) -> str:
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
            "To edit an existing image artifact:",
            '<edit_image name="artifact_name">',
            "  Description of how to edit / modify the image",
            "</edit_image>",
            "=== END IMAGE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_inline_widget_instructions(self) -> str:
        lines = [
            "",
            "=== INTERACTIVE WIDGET SYSTEM ===",
            "",
            "You can embed live, interactive HTML widgets directly in your replies.",
            "",
            "✅ WHEN TO USE A WIDGET:",
            "  • When an interactive visualization would make the concept much clearer",
            "  • To let the user experiment with parameters and see results immediately",
            "  • For teaching algorithms, math, physics, UI behavior, games, etc.",
            "",
            "❌ DO NOT use widgets for:",
            "  • Simple static explanations or text content",
            "  • Displaying code (use <artifact> or markdown code blocks instead)",
            "",
            "Tag syntax:",
            '<lollms_inline type="html" title="Clear descriptive title">',
            "<!DOCTYPE html>",
            "<html>...(complete self-contained HTML document)...</html>",
            "</lollms_inline>",
            "",
            "Rules:",
            "  • Content MUST be a valid, complete HTML5 document",
            "  • Only HTML + CSS + JavaScript — no Python, SQL, etc.",
            "  • Never wrap the HTML inside ```html code fences",
            "  • Always add explanatory text before or after the widget",
            "",
            "Supported types: html (default), react, svg",
            "",
            "=== END INTERACTIVE WIDGET SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_note_instructions(self) -> str:
        lines = [
            "",
            "=== NOTE SYSTEM ===",
            "",
            "Notes are **user-facing** persistent documents saved for the user to reference.",
            "",
            "✅ WHEN TO CREATE A NOTE:",
            "  • The user explicitly asks to save a note",
            "  • You produced analysis, comparisons, or key findings worth preserving",
            "  • Action items, decisions, or plans the user needs to track",
            "",
            "❌ DO NOT create a note for routine answers or one-off calculations.",
            "",
            "Tag syntax:",
            '<note title="Clear, descriptive title">',
            "Content here — plain text or Markdown",
            "</note>",
            "",
            "Always add a short explanation outside the tag.",
            "",
            "=== END NOTE SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_skill_instructions(self) -> str:
        lines = [
            "",
            "=== SKILL SYSTEM ===",
            "",
            "Skills are **LLM-facing** reusable knowledge capsules that persist across sessions.",
            "",
            "✅ WHEN TO CREATE A SKILL:",
            "  1. The user explicitly asks you to save it as a skill",
            "  2. You discovered a genuinely reusable technique or methodology",
            "",
            "❌ DO NOT create a skill for one-off solutions or non-generalizable content.",
            "",
            "Tag syntax:",
            '<skill title="Concise Skill Name"',
            '       description="One clear sentence"',
            '       category="domain/subdomain">',
            "Content here — Markdown with examples and usage guidelines",
            "</skill>",
            "",
            "=== END SKILL SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_form_instructions(self) -> str:
        lines = [
            "",
            "=== FORM SYSTEM ===",
            "",
            "You can create interactive forms to gather structured information from the user.",
            "",
            "✅ WHEN TO USE FORMS:",
            "  • Before starting a complex task where multiple inputs are needed",
            "  • For quizzes, challenges, or evaluations",
            "  • In multi-step workflows where early choices affect later steps",
            "",
            "IMPORTANT:",
            "  • Write a short preamble before the form tag.",
            "  • After emitting <lollms_form>, stop generation for that response.",
            "",
            "Tag syntax:",
            '<lollms_form title="Form Title" description="Instructions" submit_label="Submit">',
            '  <field name="field_id" label="Label" type="text" required="true"/>',
            "</lollms_form>",
            "",
            "Supported field types: text, textarea, number, range, select, radio,",
            "checkbox, checkbox_group, date, time, color, rating, code, section, hidden",
            "",
            "=== END FORM SYSTEM ===",
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
        enable_forms:            bool = True,
        enable_silent_artefact_explanation: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        Scans the raw LLM response for XML action tags and applies them.

        Handled tags:
            <artifact …>…</artifact>   Create or patch a named artefact.
            <generate_image …>…        Image generation via TTI binding.
            <edit_image …>…            Image editing via TTI binding.
            <lollms_inline …>…         Inline interactive HTML widget.
            <note …>…                  Persistent note (ArtefactType.NOTE).
            <skill …>…                 Knowledge capsule (ArtefactType.SKILL).
            <lollms_form …>…           Interactive form (MSG_TYPE_FORM_READY).

        NOTE: <artefact_image id="..."/> anchors are NOT processed here —
        they are preserved verbatim in the text so the UI can render them.
        """
        # ── Mask code blocks so XML inside documentation isn't processed ─────
        code_blocks: Dict[str, str] = {}

        def mask_code_block(match):
            placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
            code_blocks[placeholder] = match.group(0)
            return placeholder

        masked_text = re.sub(r'(`{3,})[\s\S]*?\1', mask_code_block, text)
        masked_text = re.sub(r'`[^`]+`',           mask_code_block, masked_text)

        has_artefact = bool(re.search(r'<(?:revert_)?art[ei]fact[\s>]', masked_text, re.IGNORECASE))
        has_gen      = enable_image_generation and bool(
            re.search(r'<generate_image[\s>]', masked_text, re.IGNORECASE))
        has_edit     = enable_image_editing and bool(
            re.search(r'<edit_image[\s>]', masked_text, re.IGNORECASE))
        has_note     = enable_notes and bool(
            re.search(r'<note[\s>]', masked_text, re.IGNORECASE))
        has_skill    = enable_skills and bool(
            re.search(r'<skill[\s>]', masked_text, re.IGNORECASE))
        has_form     = enable_forms and bool(
            re.search(r'<lollms_form[\s>]', masked_text, re.I))

        if not (has_artefact or has_gen or has_edit
                or has_note or has_skill or has_form):
            return text, []

        cleaned             = masked_text
        affected_artefacts: List[Dict] = []

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        # ── 1. Artifact create / patch ────────────────────────────────────────
        if has_artefact:
            _active_cb = getattr(self, '_active_callback', None)

            def _artefact_event(artefact: Dict, is_new: bool):
                if not _active_cb:
                    return
                import json as _json
                from lollms_client.lollms_types import MSG_TYPE as _MT
                event_type = "artifact_created" if is_new else "artifact_updated"
                try:
                    _active_cb(
                        _json.dumps({
                            "type":     event_type,
                            "title":    artefact.get("title"),
                            "version":  artefact.get("version"),
                            "art_type": artefact.get("type"),
                        }),
                        _MT.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                        {"artefact": artefact, "is_new": is_new},
                    )
                except Exception:
                    pass

            cleaned, affected_artefacts = self.artefacts._apply_artefact_xml(
                cleaned, auto_activate=auto_activate_artefacts,
                replacements=code_blocks,
                event_callback=_artefact_event,
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
                                "artifact not found or has no images; "
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

        # ── 5. Notes ──────────────────────────────────────────────────────────
        if has_note:
            note_pattern = re.compile(
                r'<note\s*([^>]*)>(.*?)</note>',
                re.DOTALL | re.IGNORECASE,
            )

            def handle_note(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title = (attrs.get('title') or attrs.get('name') or
                         f'note_{uuid.uuid4().hex[:8]}')

                note_artefact = self.artefacts.add(
                    title         = title,
                    artefact_type = ArtefactType.NOTE,
                    content       = content.strip(),
                    active        = auto_activate_artefacts,
                )
                affected_artefacts.append(note_artefact)
                ASCIIColors.success(f"Note '{title}' saved.")
                return ''

            cleaned = note_pattern.sub(handle_note, cleaned)

        # ── 6. Skills ─────────────────────────────────────────────────────────
        if has_skill:
            skill_pattern = re.compile(
                r'<skill\s*([^>]*)>(.*?)</skill>',
                re.DOTALL | re.IGNORECASE,
            )

            def handle_skill(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title       = (attrs.get('title') or attrs.get('name') or
                               f'skill_{uuid.uuid4().hex[:8]}')
                description = attrs.get('description', '')
                category    = attrs.get('category', '')

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
                return ''

            cleaned = skill_pattern.sub(handle_skill, cleaned)

        # ── Unmask code blocks ────────────────────────────────────────────────
        for placeholder, original in code_blocks.items():
            cleaned = cleaned.replace(placeholder, original)

        cleaned = cleaned.strip()

        # ── 8. Silent-artifact guard ──────────────────────────────────────────
        if enable_silent_artefact_explanation and not cleaned:
            summary_parts: List[str] = []

            non_note_non_skill = [
                a for a in affected_artefacts
                if a.get('type') not in (ArtefactType.NOTE, ArtefactType.SKILL)
            ]
            for art in non_note_non_skill:
                atype    = art.get('type', 'artifact')
                title    = art.get('title', 'untitled')
                lang     = art.get('language', '')
                version  = art.get('version', 1)
                desc     = art.get('description', '')
                img_count = len(art.get('images') or [])
                lang_str = f" ({lang})" if lang else ""
                ver_str  = f" — version {version}" if version > 1 else ""
                desc_str = f": {desc}" if desc else ""
                img_str  = f" · {img_count} image(s)" if img_count else ""
                summary_parts.append(
                    f"📄 Created **{title}**{lang_str} [{atype}{ver_str}]{desc_str}{img_str}."
                )

            notes = [a for a in affected_artefacts if a.get('type') == ArtefactType.NOTE]
            for note in notes:
                title = note.get('title', 'untitled')
                first_line = next(
                    (l.strip() for l in note.get('content', '').splitlines() if l.strip()),
                    ''
                )
                peek = f" — {first_line[:80]}…" if first_line else ""
                summary_parts.append(f"📝 Saved note **{title}**{peek}")

            skills = [a for a in affected_artefacts if a.get('type') == ArtefactType.SKILL]
            for skill in skills:
                title    = skill.get('title', 'untitled')
                category = skill.get('category', '')
                desc     = skill.get('description', '')
                cat_str  = f" [{category}]" if category else ""
                desc_str = f" — {desc}" if desc else ""
                summary_parts.append(f"🎓 Skill saved **{title}**{cat_str}{desc_str}.")

            # Widgets are now inline tags only - check content for widget anchors
            widget_count = len(re.findall(
                r'<lollms_widget\s+id=["\'][^"\']+["\']\s*/?>',
                cleaned
            ))
            for _ in range(widget_count):
                summary_parts.append(
                    "🎛️ Interactive widget ready — use the controls below to explore the concept."
                )

            for form_ref in meta_now2.get("forms", []):
                f_title = form_ref.get('title', 'Form')
                summary_parts.append(
                    f"📋 Form ready: **{f_title}** — please fill in the fields above."
                )

            if summary_parts:
                cleaned = "\n".join(summary_parts)
                ASCIIColors.info(
                    "[silent-artifact guard] Auto-generated explanation appended.")

        return cleaned, affected_artefacts