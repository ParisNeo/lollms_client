# lollms_discussion/_mixin_prompt.py
# PromptMixin: system-prompt instruction builders and LLM response post-processor.

import re
import uuid
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ascii_colors import ASCIIColors

if TYPE_CHECKING:
    from ._message import LollmsMessage


class PromptMixin:
    """
    Builds artefact / image-generation instructions for the system prompt and
    post-processes the LLM response to apply any XML action tags it contains.
    """

    # ------------------------------------------------ instruction builders

    def _build_artefact_instructions(self) -> str:
        """Returns prompt instructions for creating/updating artefacts via XML tags."""
        lines = [
            "",
            "=== ARTEFACT SYSTEM ===",
            "You can create, modify, and version control artefacts (persistent code, documents, notes, skills).",
            "Always use XML tags for artefact operations. The system keeps full version history automatically.",
            "",
            "To CREATE an empty structure or new artefact (Supports metadata like author and description):",
            '<artefact name="unique_name.py" type="code" language="python" author="Your Name" description="Initial boilerplate">',
            "[full content here]",
            "</artefact>",
            "",
            "To UPDATE an existing artefact efficiently, use the SEARCH/REPLACE aider format.",
            "This preserves context limits. Provide exactly the lines to find and exactly the lines to replace.",
            '<artefact name="existing_name.py">',
            "<<<<<<< SEARCH",
            "[exact lines to find]",
            "=======",
            "[replacement lines]",
            ">>>>>>> REPLACE",
            "</artefact>",
            "",
            "To REVERT to a previous known version:",
            '<revert_artefact name="existing_name.py" version="1" />',
            "",
            "Supported types: code, document, note, skill, file, search_result, image",
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

    # ------------------------------------------------ LLM response post-processor

    def _post_process_llm_response(
        self,
        text: str,
        ai_message: 'LollmsMessage',
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        Scans the raw LLM response for XML action tags and applies them.

        This is the *single* place where all LLM-output XML is intercepted.
        Called at the end of ``chat()`` / ``simplified_chat()``, after
        ``remove_thinking_blocks`` has run, and *after* the ai_message object
        has been created (so images can be attached to it directly).

        Handled tags
        ------------
        ``<artefact name="…" type="…" language="…">…</artefact>``
            Create or patch (aider SEARCH/REPLACE) a named artefact.
            Always processed when present — no flag needed.

        ``<generate_image width="…" height="…">prompt</generate_image>``
            Calls ``lollmsClient.tti.generate()`` and attaches the result
            to *ai_message* via ``add_image_pack``.  Only processed when
            ``enable_image_generation=True`` and ``lollmsClient.tti`` is set.

        ``<edit_image name="…">edit prompt</edit_image>``
            Calls ``lollmsClient.tti.edit()`` on the last available source
            image (from a named artefact, or falling back to the last image
            already on *ai_message*).  Result attached via ``add_image_pack``.
            Only processed when ``enable_image_editing=True`` and
            ``lollmsClient.tti`` is set.

        Images go to ``message.images`` only — they are NOT stored as artefacts.

        Returns:
            (cleaned_text, affected_artefacts)
            ``cleaned_text`` has all processed XML tags stripped.
        """
        # Mask code blocks to prevent processing XML tags inside documentation/code
        code_blocks = {}
        def mask_code_block(match):
            placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
            code_blocks[placeholder] = match.group(0)
            return placeholder

        # Mask markdown block code (triple backticks or more)
        masked_text = re.sub(r'(`{3,})[\s\S]*?\1', mask_code_block, text)
        # Mask inline code
        masked_text = re.sub(r'`[^`]+`', mask_code_block, masked_text)

        has_artefact = bool(re.search(r'<(?:revert_)?artefact[\s>]', masked_text, re.IGNORECASE))
        has_gen      = enable_image_generation and bool(
            re.search(r'<generate_image[\s>]', masked_text, re.IGNORECASE))
        has_edit     = enable_image_editing and bool(
            re.search(r'<edit_image[\s>]', masked_text, re.IGNORECASE))

        if not (has_artefact or has_gen or has_edit):
            return text, []

        cleaned             = masked_text
        affected_artefacts: List[Dict] = []

        # ── 1. Artefact create / patch ────────────────────────────────────────
        if has_artefact:
            cleaned, affected_artefacts = self.artefacts._apply_artefact_xml(
                cleaned, auto_activate=auto_activate_artefacts,
                replacements=code_blocks
            )

        # Extract type from patch tag if present for re-typing support
        def _extract_type_from_patch(text: str) -> Optional[str]:
            patch_match = re.search(r'<artefact\s+[^>]*type=["\']([^"\']+)["\'][^>]*>', text, re.IGNORECASE)
            return patch_match.group(1) if patch_match else None

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        # ── 2. Image generation → message.images ─────────────────────────────
        if has_gen:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<generate_image> found but lollmsClient.tti is None — skipping.")
            else:
                gen_pattern = re.compile(
                    r'<generate_image\s*([^>]*)>(.*?)</generate_image>',
                    re.DOTALL | re.IGNORECASE
                )

                def handle_generate(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()

                    # Restore masked content in prompt or attributes
                    for placeholder, original in code_blocks.items():
                        prompt = prompt.replace(placeholder, original)
                        for k, v in attrs.items():
                            if placeholder in v:
                                attrs[k] = v.replace(placeholder, original)

                    width  = int(attrs.get('width',  1024))
                    height = int(attrs.get('height', 1024))
                    try:
                        # Use generate_image method from TTI binding
                        img_bytes = tti.generate_image(prompt=prompt, width=width, height=height)
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
                    re.DOTALL | re.IGNORECASE
                )

                def handle_edit(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()

                    # Restore masked content in prompt or attributes
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
                        # Use edit_image method from TTI binding
                        # Note: bindings usually take base64 or paths for edit_image
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

        # Unmask code blocks
        for placeholder, original in code_blocks.items():
            cleaned = cleaned.replace(placeholder, original)

        return cleaned.strip(), affected_artefacts
