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
    Builds artifact / image-generation / inline-widget instructions for the
    system prompt and post-processes the LLM response to apply any XML action
    tags it contains.

    NOTE ON SPELLING
    ----------------
    The XML tags shown to the LLM use the American spelling "artifact"
    (<artifact>, </artifact>, <revert_artifact />) because it appears far
    more frequently in LLM training corpora.  The Python API (method names,
    variable names, class names) keeps the British spelling "artefact" for
    backwards compatibility.  The parser accepts BOTH spellings from the LLM.

    HANDLE SYSTEM
    -------------
    <use_handle ref="N:M" name="filename" type="code" language="python"/>
    Converts an already-generated code block (msg N, block M) into an artefact
    without rewriting it.  See _mixin_chat._apply_handles for resolution logic.
    """

    # ─────────────────────────────────────── instruction builders ────────────

    def _build_artefact_instructions(self) -> str:
        """Returns prompt instructions for creating/updating artifacts via XML tags."""
        lines = [
            "",
            "=== ARTIFACT SYSTEM ===",
            "You can create, modify, and version control artifacts (persistent code, documents, notes, skills).",
            "Always use XML tags for artifact operations. The system keeps full version history automatically.",
            "",
            "IMPORTANT: Whenever you create or update an artifact, you MUST also write a",
            "short explanation in your reply (outside the tag) telling the user what you",
            "created/changed and why. Never let your entire response consist of only XML tags.",
            "Example: 'I've created `hello.py` with a basic Flask server — here's what it does: …'",
            "",
            "── Option A: CREATE or REPLACE (write full content) ──────────────────────",
            '<artifact name="unique_name.py" type="code" language="python"',
            '          author="Your Name" description="What it does">',
            "[full content here]",
            "</artifact>",
            "",
            "── Option B: HANDLE (reference an existing code block — no rewriting) ─────",
            "If you already wrote a code block earlier in this conversation, you can",
            "convert it directly into an artefact WITHOUT copying it again:",
            "",
            '<use_handle ref="<msg_idx>:<block_idx>" name="filename.py"',
            '            type="code" language="python"/>',
            "",
            "  msg_idx   — 0-based index of the message in this conversation",
            "  block_idx — 0-based index of the fenced code block in that message",
            "",
            "Available handles are listed under '=== AVAILABLE HANDLES ===' if any exist.",
            "Use handles whenever you want to save code you already wrote — avoid",
            "duplicating content unless the code actually needs to change.",
            "",
            "── Option C: PATCH (SEARCH/REPLACE — update an existing artefact) ──────────",
            "To UPDATE an existing artifact, use SEARCH/REPLACE blocks inside the tag.",
            "CRITICAL RULES FOR SEARCH/REPLACE:",
            "  1. Keep each SEARCH block as SHORT as possible — 1 to 5 lines is ideal.",
            "  2. You may include MULTIPLE SEARCH/REPLACE blocks inside a single <artifact>",
            "     tag to make several independent edits in one turn.",
            "  3. Each block MUST use this exact structure:",
            "       <<<<<<< SEARCH",
            "       [exact lines to find — copy verbatim from the artifact]",
            "       =======",
            "       [replacement lines]",
            "       >>>>>>> REPLACE",
            "  4. The SEARCH text must match the artifact content CHARACTER FOR CHARACTER.",
            "  5. Do NOT omit the >>>>>>> REPLACE marker at the end.",
            "",
            "Example — two independent edits in one tag:",
            '<artifact name="app.py">',
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
            "</artifact>",
            "",
            "── Option D: RENAME ────────────────────────────────────────────────────────",
            "To rename an artifact while updating it, add a rename attribute:",
            '<artifact name="old_title" rename="new_title">',
            "  [full new content OR SEARCH/REPLACE blocks]",
            "</artifact>",
            "",
            "── Option E: REVERT ────────────────────────────────────────────────────────",
            "To revert to a previous known version:",
            '<revert_artifact name="existing_name.py" version="1" />',
            "",
            f"Supported types: {', '.join(sorted(list(ArtefactType.ALL)))}",
            "=== END ARTIFACT INSTRUCTIONS ===",
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
            "=== INTERACTIVE TEACHING WIDGETS ===",
            "You can embed a live, interactive widget directly inside your reply to help",
            "the user *learn by doing*.  Use a widget whenever an interactive visual would",
            "make a concept clearer than text alone — think of it as a mini lab or demo.",
            "",
            "IMPORTANT: Always write a short explanation in your reply alongside the widget.",
            "Never let the widget be your entire response.  Tell the user:",
            "  • what the widget demonstrates,",
            "  • which controls to interact with and what to look for,",
            "  • the key insight they should walk away with.",
            "",
            "Tag syntax:",
            '<lollms_inline type="html" title="Descriptive title">',
            "  <!-- self-contained HTML/JS/CSS — no external files; CDN links are OK -->",
            "</lollms_inline>",
            "",
            "Supported types:",
            "  html  — default; full HTML/JS/CSS in one file (most flexible)",
            "  react — JSX component rendered client-side",
            "  svg   — animated or interactive SVG",
            "",
            "GOOD uses (teaching moments):",
            "  • Physics / math simulations with parameter sliders",
            "  • Algorithm step-through visualisers",
            "  • Signal / wave / Fourier explorers",
            "  • Probability / statistics sandboxes",
            "  • Mini quizzes or flashcard drills",
            "",
            "BAD uses (do NOT use a widget for these):",
            "  • Full applications → use <artifact> instead",
            "  • Static charts with no interactivity → just describe or use a note",
            "  • Anything requiring a server or file I/O",
            "",
            "Technical rules:",
            "  • Fully self-contained — zero network requests at runtime",
            "    (CDN links to well-known libraries are fine).",
            "  • Target ≤ 460 px height, full container width.",
            "  • Label every control clearly so the user knows what to do.",
            "  • No alert() / confirm() / prompt().",
            "=== END INTERACTIVE TEACHING WIDGET INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_note_instructions(self) -> str:
        lines = [
            "",
            "=== NOTE SYSTEM ===",
            "Notes are persistent, named documents stored alongside the discussion.",
            "",
            "Use <note> ONLY when:",
            "  • The user explicitly asks you to save or remember information",
            "  • You have analysis results, comparison tables, or key findings the user needs",
            "    to reference later in THIS discussion",
            "  • You are creating action items or task lists the user needs to track",
            "",
            "DO NOT use <note> for:",
            "  • Routine explanations or answers to single questions",
            "  • Calculations or solutions to specific problems (unless user asks to save)",
            "  • Basic facts or concepts that don't need persistence",
            '  • Code (use <artifact type="code">)',
            "",
            "Tag syntax:",
            '<note title="Clear descriptive title">',
            "  [note content — plain text or Markdown]",
            "</note>",
            "",
            "When in doubt, ask the user if they want to save something as a note.",
            "=== END NOTE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_skill_instructions(self) -> str:
        lines = [
            "",
            "=== SKILL SYSTEM ===",
            "Skills are reusable knowledge capsules that persist across sessions.",
            "They are retrieved automatically when relevant to future questions.",
            "",
            "Emit a <skill> tag ONLY when ALL these conditions are met:",
            "  1. The user EXPLICITLY asks you to save something as a skill ('save this as a skill',",
            "     'remember this pattern', 'learn this trick', etc.) — OR —",
            "  2. You are teaching a genuinely reusable technique, design pattern, or methodology",
            "     that the user would explicitly want to reuse in unrelated future tasks.",
            "",
            "DO NOT save as a skill:",
            "  • Single calculations, formulas, or one-off solutions to specific problems",
            "  • Explanations of basic concepts that are easily searchable",
            "  • Code snippets that are not generalizable patterns or frameworks",
            "  • Anything where the user has not explicitly indicated this should be remembered",
            "",
            "When in doubt, ask the user: 'Should I save this as a reusable skill?'",
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
        enable_silent_artefact_explanation: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        Scans the raw LLM response for XML action tags and applies them.

        Accepts both spellings from the LLM:
          <artifact …>  or  <artefact …>   (American / British)
          <revert_artifact …>  or  <revert_artefact …>
        All other tags use a single canonical spelling.
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
            Extracts a self-contained interactive teaching widget and stores it
            in ``ai_message.metadata["inline_widgets"]``.
            Only when ``enable_inline_widgets=True``.

        ``<note title="…">…</note>``
            Saves a persistent named note as an ``ArtefactType.NOTE`` artefact.
            Active immediately; visible in the artefact panel.
            Only when ``enable_notes=True``.

        ``<skill title="…" description="…" category="…">…</skill>``
            Saves a reusable knowledge capsule as an ``ArtefactType.SKILL``
            artefact with category and description metadata.
            Only when ``enable_skills=True``.

        Silent artefact guard
        ---------------------
        When ``enable_silent_artefact_explanation=True`` (the default), if the
        cleaned text after stripping all XML tags is blank or only whitespace,
        a concise auto-generated explanation is appended so the user always
        receives a human-readable confirmation of what was done.

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

        if not (has_artefact or has_gen or has_edit or has_inline or has_note or has_skill):
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

                widget_entry = {
                    "id":     widget_id,
                    "type":   widget_type,
                    "title":  widget_title,
                    "source": source.strip(),
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

        # ── Unmask code blocks ────────────────────────────────────────────────
        for placeholder, original in code_blocks.items():
            cleaned = cleaned.replace(placeholder, original)

        cleaned = cleaned.strip()

        # ── 7. Silent-artifact guard ──────────────────────────────────────────
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

            meta_now = dict(ai_message.metadata or {})
            for widget in meta_now.get("inline_widgets", []):
                w_title = widget.get('title', 'Interactive Widget')
                w_type  = widget.get('type', 'html')
                summary_parts.append(
                    f"🎛️ Interactive widget ready: **{w_title}** ({w_type}) — "
                    "use the controls below to explore the concept."
                )

            if summary_parts:
                cleaned = "\n".join(summary_parts)
                ASCIIColors.info(
                    "[silent-artifact guard] Auto-generated explanation appended.")

        return cleaned, affected_artefacts