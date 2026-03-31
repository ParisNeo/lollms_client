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
    Builds artifact / image-generation / inline-widget / form instructions for the
    system prompt and post-processes the LLM response to apply any XML action
    tags it contains.

    NOTE ON SPELLING
    ----------------
    The XML tags shown to the LLM use the American spelling "artifact"
    (<artifact>, </artifact>, <revert_artifact />) because it appears far
    more frequently in LLM training corpora.  The Python API (method names,
    variable names, class names) keeps the British spelling "artefact" for
    backwards compatibility.  The parser accepts BOTH spellings from the LLM.

    FORM SYSTEM
    -----------
    <lollms_form> lets the LLM pause and ask the user structured questions.
    It renders as an interactive form in the UI.  The complete form descriptor
    fires as MSG_TYPE_FORM_READY.  The application must call
    discussion.submit_form_response(form_id, answers) to resume generation.

    WIDGET VALIDATION
    -----------------
    <lollms_inline> content is validated: non-web code fences (python, mermaid,
    etc.) are stripped.  If no HTML tag survives, the widget is replaced with
    an error placeholder.  Only HTML/CSS/JS is accepted.

    HANDLE SYSTEM
    -------------
    <use_handle ref="N:M" name="filename" type="code" language="python"/>
    Converts an already-generated code block (msg N, block M) into an artefact
    without rewriting it.  See _mixin_chat._apply_handles for resolution logic.
    """

    # ─────────────────────────────────────── instruction builders ────────────
    def _build_artefact_instructions(self) -> str:
        """Returns the latest 2026-best-practice prompt for artifact operations."""
        lines = [
            "",
            "=== ARTIFACT SYSTEM ===",
            "",
            "**CRITICAL DISTINCTION — READ THIS FIRST**",
            "",
            "Artifacts and markdown code blocks are **completely different** things:",
            "",
            "• Markdown code blocks (```language ```)",
            "• Artifacts (<artifact> XML tags) = persistent, version-controlled storage",
            "",
            "❌ **NEVER** do this — it breaks the system permanently:",
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
            "Always choose the **most accurate type** for the content. Do not default to any single language.",
            "",
            "=== HOW TO USE ARTIFACTS ===",
            "You can create, update, rename, or revert persistent artifacts using XML tags.",
            "The system automatically maintains full version history.",
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
            "── Option B: HANDLE (reference previous code block — no duplication)",
            "<artifact name=\"filename.ext\" type=\"appropriate_type\" handle=\"msg_idx:block_idx\" />",
            "",
            "── Option C: PATCH (targeted updates)",
            "<artifact name=\"filename.ext\" type=\"appropriate_type\">",
            "<<<<<<< SEARCH",
            "exact old lines here",
            "=======",
            "new lines here",
            ">>>>>>> REPLACE",
            "",
            "<<<<<<< SEARCH",
            "another exact section",
            "=======",
            "replacement",
            ">>>>>>> REPLACE",
            "</artifact>",
            "",
            "── Option D: RENAME + UPDATE",
            "<artifact name=\"new_name.ext\" type=\"appropriate_type\" rename=\"old_name.ext\">",
            "... content or SEARCH/REPLACE blocks ...",
            "</artifact>",
            "",
            "── Option E: REVERT",
            "<artifact name=\"filename.ext\" revert_to=\"v3\" />",
            "",
            "=== REMINDER (most important rules again) ===",
            "→ Artifacts = persistent & versioned",
            "→ Markdown code blocks = temporary display only",
            "→ Never nest <artifact> inside ``` blocks",
            "→ Always choose the correct type — never default",
            "→ Always add a human explanation outside the tag",
            "",
            "=== END ARTIFACT SYSTEM ===",
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
            "**Purpose**",
            "You can embed live, interactive HTML widgets directly in your replies to help users *learn by doing*. ",
            "Widgets turn abstract concepts into mini interactive labs or demos.",
            "",
            "**CRITICAL DISTINCTION**",
            "• Normal code blocks or explanations = static text",
            "• Widgets (`<lollms_inline>`) = **live, interactive experiences** the user can play with",
            "",
            "✅ WHEN TO USE A WIDGET:",
            "  • When an interactive visualization would make the concept much clearer than text alone",
            "  • To let the user experiment with parameters and immediately see results",
            "  • For teaching algorithms, math, physics, UI behavior, data transformations, games, etc.",
            "  • Whenever 'learning by doing' is more effective than 'learning by reading'",
            "",
            "❌ DO NOT use widgets for:",
            "  • Simple static explanations or text content",
            "  • Displaying code (use <artifact> or markdown code blocks instead)",
            "  • Non-interactive diagrams (use Mermaid or normal images)",
            "  • Anything that doesn't meaningfully benefit from user interaction",
            "",
            "=== HOW TO CREATE A WIDGET ===",
            "",
            "Always include a short, natural explanation **outside** the tag that tells the user:",
            "  • What the widget demonstrates",
            "  • Which controls they can interact with",
            "  • What key insight they should discover",
            "",
            "Never let the widget be your entire response.",
            "",
            "Tag syntax — Content must be a **complete, standalone HTML document**:",
            '<lollms_inline type="html" title="Clear descriptive title">',
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset=\"UTF-8\">",
            "  <title>Widget Title</title>",
            "  <style>",
            "    /* All CSS here */",
            "  </style>",
            "</head>",
            "<body>",
            "  <!-- All HTML content here -->",
            "  <script>",
            "    /* All JavaScript here */",
            "  </script>",
            "</body>",
            "</html>",
            "</lollms_inline>",
            "",
            "=== WIDGET CONTENT RULES (Strict) ===",
            "  • The entire content MUST be a valid, complete HTML5 document",
            "  • Only HTML + CSS + JavaScript allowed — nothing else",
            "  • **Never** wrap the HTML inside ```html ... ``` code fences",
            "  • Do NOT include Python, Mermaid, SQL, or any non-web languages",
            "  • No alert(), confirm(), or prompt() dialogs",
            "  • Keep height reasonable (ideally ≤ 460px) and responsive to width",
            "  • Make every button, slider, or control clearly labeled",
            "  • Fully self-contained (you may use public CDNs for common libraries like Chart.js, Three.js, etc.)",
            "  • The widget must work immediately when rendered",
            "",
            "Supported types: html (default), react, svg",
            "",
            "=== REMINDER ===",
            "→ Widgets are for **interactive learning**, not static content",
            "→ Always add explanatory text before or after the widget",
            "→ Never output the widget alone",
            "→ Do not wrap widget content in markdown code blocks",
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
            "**CRITICAL DISTINCTION**",
            "",
            "Notes are **user-facing** persistent documents.",
            "They are saved for the **user** to read and reference later in this discussion.",
            "",
            "Use the <note> tag **only** when it provides real value to the user.",
            "",
            "✅ WHEN TO CREATE A NOTE:",
            "  • The user explicitly asks to save a note for himself",
            "  • You have produced analysis, comparisons, tables, key findings, or summaries",
            "    that the user would benefit from referencing later",
            "  • You are creating action items, task lists, decisions, or plans the user needs to track",
            "  • You believe the user would find it genuinely useful to have this information persisted",
            "",
            "❌ DO NOT create a note for:",
            "  • Routine explanations or answers to simple questions",
            "  • One-off calculations or solutions (unless the user asks to save them)",
            "  • Basic facts or concepts that don't need persistence",
            "  • Code or reusable techniques → use <artifact> instead",
            "",
            "When in doubt: Ask the user 'Would you like me to save this as a note for future reference?'",
            "",
            "Tag syntax:",
            '<note title="Clear, descriptive title">',
            "Content here — use plain text or Markdown (headings, lists, tables, etc.)",
            "</note>",
            "",
            "Always add a short natural explanation **outside** the tag, e.g.:",
            '"I\'ve saved the comparison table as a note titled \"Model Options\" for easy reference."',
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
            "**CRITICAL DISTINCTION**",
            "",
            "Skills are **LLM-facing** reusable knowledge capsules.",
            "They persist across sessions and are automatically retrieved when relevant to future queries.",
            "Skills are for the **LLM** to become better at solving similar problems in the future.",
            "",
            "Create a <skill> **only** when it has clear long-term reusable value.",
            "",
            "✅ WHEN TO CREATE A SKILL:",
            "  1. The user **explicitly** asks you to save it as a skill ('save this as a skill', 'learn this pattern', 'remember this technique', etc.)",
            "  2. OR you have discovered or synthesized a genuinely reusable technique, design pattern, methodology, prompt template, or heuristic during this session that would help you perform better on unrelated future tasks.",
            "",
            "❌ DO NOT create a skill for:",
            "  • One-off solutions, calculations, or specific problem instances",
            "  • Basic explanations or easily searchable facts",
            "  • Non-generalizable code snippets (use <artifact> instead)",
            "  • Anything without clear reusable value across sessions",
            "",
            "When in doubt: Ask the user 'Should I save this as a reusable skill for future conversations?'",
            "",
            "Tag syntax:",
            '<skill title="Concise Skill Name"',
            '       description="One clear sentence describing what this skill teaches"',
            '       category="domain/subdomain">',
            "Content here — use Markdown with explanation + concrete examples + usage guidelines",
            "</skill>",
            "",
            "Category examples: programming/python/async, writing/prompting/advanced, analysis/comparison/tables, etc.",
            "",
            "Always add a short natural explanation **outside** the tag explaining why this skill is valuable.",
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
            "The application renders the form as a nice UI; the user fills it in and submits.",
            "Their answers are sent back to you as a new system message so you can continue accurately.",
            "",
            "✅ WHEN TO USE FORMS:",
            "  • Before starting a complex task where user preferences or multiple inputs are needed",
            "  • When collecting several pieces of information at once is more efficient than many messages",
            "  • For quizzes, challenges, or evaluations",
            "  • In multi-step workflows where early choices affect later steps",
            "  • To make the interaction more guided, enjoyable, and structured",
            "",
            "❌ DO NOT overuse forms for simple or single-question requests.",
            "",
            "IMPORTANT RULES:",
            "  • Always write a short, friendly preamble explaining why you need the information and what you'll do with it.",
            "  • After emitting the <lollms_form> tag, **stop generation** — do not add any more text in that response.",
            "  • The conversation continues after the user submits the form.",
            "",
            "── Basic XML syntax ─────────────────────────────────────────────────────",
            '<lollms_form title="Form Title" description="Optional instructions for the user"',
            '             submit_label="Submit">',
            '  <field name="field_id" label="User visible label" type="text" required="true"/>',
            "</lollms_form>",
            "",
            "── Field types reference (exact supported types) ────────────────────────",
            "  text            — single-line text input",
            "  textarea        — multi-line text input (use rows='N' to control height)",
            "  number          — numeric input (supports min=, max=, step=)",
            "  range           — slider input (requires min= and max= attributes)",
            "  select          — dropdown list (use options='Value1,Value2,Value3')",
            "  radio           — radio buttons (single choice, use options='A,B,C')",
            "  checkbox        — single yes/no checkbox",
            "  checkbox_group  — multiple checkboxes (use options='A,B,C')",
            "  date            — date picker",
            "  time            — time picker",
            "  color           — color picker (returns hexadecimal #RRGGBB)",
            "  rating          — star rating widget (use min= and max= to set number of stars)",
            "  code            — code editor with syntax highlighting (use language='python' or similar)",
            "  section         — visual divider or sub-heading (no user input, use label= for title)",
            "  hidden          — hidden field whose value is set via default= attribute",
            "",
            "── Quiz / challenge example ─────────────────────────────────────────────",
            '<lollms_form title="Python Quiz — Lists" submit_label="Check my answers">',
            '  <field name="q1" label="What method adds an element to the END of a list?"',
            '         type="text" placeholder="method name only" required="true"/>',
            '  <field name="q2" label="What does list[1:3] return for [10,20,30,40]?"',
            '         type="radio" options="[10,20],[20,30],[30,40],[20,30,40]" required="true"/>',
            '  <field name="q3" label="Write a list comprehension for squares 1–5"',
            '         type="code" language="python" rows="3" required="true"/>',
            "</lollms_form>",
            "",
            "── Alternative: JSON body ───────────────────────────────────────────────",
            "You may also provide the form definition as a JSON object containing a 'fields' array.",
            "The JSON structure mirrors the XML field attributes shown above.",
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

        Handled tags
        ------------
        ``<artifact …>…</artifact>``  (both spellings)
            Create or patch a named artefact.

        ``<generate_image …>…</generate_image>``
            Image generation via TTI binding.

        ``<edit_image …>…</edit_image>``
            Image editing via TTI binding.

        ``<lollms_inline …>…</lollms_inline>``
            Inline interactive HTML/CSS/JS widget.
            Content is validated — non-web code is stripped.

        ``<note …>…</note>``
            Persistent named note (ArtefactType.NOTE).

        ``<skill …>…</skill>``
            Reusable knowledge capsule (ArtefactType.SKILL).

        ``<lollms_form …>…</lollms_form>``
            Interactive form rendered by the application.
            Fires MSG_TYPE_FORM_READY and waits for
            discussion.submit_form_response().

        Silent artefact guard
        ---------------------
        When enable_silent_artefact_explanation=True and the cleaned text is
        blank, a concise auto-generated explanation is appended.
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
        has_inline   = enable_inline_widgets and bool(
            re.search(r'<lollms_inline[\s>]', masked_text, re.IGNORECASE))
        has_note     = enable_notes and bool(
            re.search(r'<note[\s>]', masked_text, re.IGNORECASE))
        has_skill    = enable_skills and bool(
            re.search(r'<skill[\s>]', masked_text, re.IGNORECASE))
        has_form     = enable_forms and bool(
            re.search(r'<lollms_form[\s>]', masked_text, re.I))

        if not (has_artefact or has_gen or has_edit or has_inline
                or has_note or has_skill or has_form):
            return text, []

        cleaned             = masked_text
        affected_artefacts: List[Dict] = []

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        # ── 1. Artifact create / patch (both spellings) ───────────────────────
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

        # ── 4. Inline widgets → message.metadata["inline_widgets"] ───────────
        if has_inline:
            from ._mixin_chat import _validate_widget_content

            inline_pattern = re.compile(
                r'<lollms_inline\s*([^>]*)>(.*?)</lollms_inline>',
                re.DOTALL | re.IGNORECASE,
            )

            meta = dict(ai_message.metadata or {})
            if "inline_widgets" not in meta:
                meta["inline_widgets"] = []

            def handle_inline(match: re.Match) -> str:
                attrs  = _parse_attrs(match.group(1))
                source = match.group(2)

                for placeholder, original in code_blocks.items():
                    source = source.replace(placeholder, original)

                widget_id    = str(uuid.uuid4())
                widget_type  = attrs.get('type', 'html').lower().strip()
                widget_title = attrs.get('title', 'Interactive Widget')

                if widget_type not in ('html', 'react', 'svg'):
                    widget_type = 'html'

                # ── Validate content — only HTML/CSS/JS allowed ───────────────
                validated = _validate_widget_content(source, widget_title)
                if validated is None:
                    ASCIIColors.warning(
                        f"[post-process] Widget '{widget_title}' discarded — no valid HTML.")
                    return (
                        f"\n\n*[Widget '{widget_title}' could not be rendered: "
                        "the content did not contain valid HTML/CSS/JS.]*\n\n"
                    )

                widget_entry = {
                    "id":     widget_id,
                    "type":   widget_type,
                    "title":  widget_title,
                    "source": validated,
                }
                meta["inline_widgets"].append(widget_entry)
                ASCIIColors.success(
                    f"Inline widget '{widget_title}' ({widget_type}) registered.")
                return f'<lollms_widget id="{widget_id}" />'

            cleaned = inline_pattern.sub(handle_inline, cleaned)

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

        # ── 6. Skills → ArtefactType.SKILL artefacts ─────────────────────────
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

        # ── 7. Forms → MSG_TYPE_FORM_READY  ──────────────────────────────────
        if has_form:
            from ._mixin_chat import _parse_form_xml, _format_form_answers_for_llm
            from lollms_client.lollms_types import MSG_TYPE as _MT

            form_pattern = re.compile(
                r'<lollms_form\s*([^>]*)>(.*?)</lollms_form>',
                re.DOTALL | re.IGNORECASE,
            )

            meta_now = dict(ai_message.metadata or {})
            if "forms" not in meta_now:
                meta_now["forms"] = []

            def handle_form(match: re.Match) -> str:
                import json as _json
                attrs_str = match.group(1)
                body      = match.group(2)

                # Unmask code blocks in body
                for placeholder, original in code_blocks.items():
                    body = body.replace(placeholder, original)

                form_descriptor = _parse_form_xml(attrs_str, body)
                if not form_descriptor:
                    ASCIIColors.warning("[post-process] Form parsing failed; skipping.")
                    return ''

                # Register in pending forms store
                self._get_pending_forms()[form_descriptor["id"]] = form_descriptor

                # Store in message metadata
                meta_now["forms"].append({
                    "id":    form_descriptor["id"],
                    "title": form_descriptor["title"],
                })

                # Fire FORM_READY event
                _active_cb = getattr(self, '_active_callback', None)
                if _active_cb:
                    try:
                        _active_cb(
                            _json.dumps(form_descriptor),
                            _MT.MSG_TYPE_FORM_READY,
                            {"form": form_descriptor, "form_id": form_descriptor["id"]},
                        )
                    except Exception:
                        pass

                ASCIIColors.cyan(
                    f"[Form] '{form_descriptor['title']}' ready "
                    f"(id={form_descriptor['id'][:8]}). "
                    "Awaiting submit_form_response()."
                )
                # Return the anchor so it replaces the XML tag in the text
                return f'\n<lollms_form_anchor id="{form_descriptor["id"]}" />\n'

            cleaned = form_pattern.sub(handle_form, cleaned)

            if meta_now["forms"]:
                ai_message.metadata = meta_now

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
                lang_str = f" ({lang})" if lang else ""
                ver_str  = f" — version {version}" if version > 1 else ""
                desc_str = f": {desc}" if desc else ""
                summary_parts.append(
                    f"📄 Created **{title}**{lang_str} [{atype}{ver_str}]{desc_str}."
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

            meta_now2 = dict(ai_message.metadata or {})
            for widget in meta_now2.get("inline_widgets", []):
                w_title = widget.get('title', 'Interactive Widget')
                w_type  = widget.get('type', 'html')
                summary_parts.append(
                    f"🎛️ Interactive widget ready: **{w_title}** ({w_type}) — "
                    "use the controls below to explore the concept."
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