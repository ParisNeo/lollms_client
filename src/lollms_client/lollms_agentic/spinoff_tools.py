# lollms_agentic/spinoff_tools.py
# Generic spinoff sub-agent factory: exposes ONE tool that lets the LLM
# condition a fully bespoke, specialized sub-agent at runtime.
#
# DOCTRINE:
#   * Spinoffs are worker-tier execution capabilities. When orchestrator_mode
#     is True the registry returns an EMPTY dict — the Orchestrator persona
#     must never hold executable tool syntax. Workers inherit this factory
#     through the tools registry instead.
#   * The caller (LLM) describes WHO the specialist is (persona), WHAT it must
#     do (task), WHAT it may use (tools whitelist, skills, files). The runtime
#     resolves everything under sandbox containment and never fabricates
#     capabilities.
#
# Security invariants:
#   * Tool whitelisting is a strict SUBSET of the parent registry — a spinoff
#     can never invent or escalate privileges.
#   * Context files are resolved against the workspace root with traversal
#     rejection and containment checks before reading.
#   * All failures are structured data, never raised exceptions.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ascii_colors import ASCIIColors

_MAX_TASK_CHARS = 8000
_MAX_PERSONA_CHARS = 4000
_MAX_INLINE_SKILL_CHARS = 8000
_MAX_CONTEXT_FILES = 12
_MAX_SKILL_TITLES = 6


@dataclass(frozen=True)
class SpinoffConfig:
    """Validated, bounded configuration for one spinoff agent spawn."""
    name: str
    task: str
    persona_directives: str = ""
    tools_whitelist: tuple = ()
    skill_titles: tuple = ()
    inline_skill: str = ""
    context_files: tuple = ()
    max_rounds: int = 6
    temperature: float = 0.3


def build_spinoff_agent_tools(
    discussion,
    images: list,
    orchestrator_mode: bool = False,
    tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    skills_manager: Any = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Registers the generic spinoff agent factory as a single executable tool.

    Enables the LLM to delegate heavy cognitive, formatting, or parsing work to
    a fully self-conditioned specialist without polluting the primary
    conversation context or breaking KV-cache alignment.

    Args:
        discussion: The active LollmsDiscussion instance.
        images: Active image payloads forwarded to the spinoff generation.
        orchestrator_mode: When True (Orchestrator persona), returns an EMPTY
            dict — the Orchestrator must never be handed executable tool syntax.
            Workers spawned by the Orchestrator receive the spinoffs through the
            tools registry instead.
        tools_registry: The parent turn's live tool registry. Used as the
            privilege ceiling for the spinoff's tools whitelist (strict subset).
        skills_manager: Optional SkillsManager instance used to resolve skill
            titles referenced by the spinoff configuration.
    """
    spinoffs: Dict[str, Dict[str, Any]] = {}

    if orchestrator_mode:
        ASCIIColors.info(
            "[SpinoffTools] Orchestrator persona active — spinoff sub-agent "
            "tools withheld from its registry (Workers inherit them)."
        )
        return spinoffs

    parent_registry: Dict[str, Dict[str, Any]] = dict(tools_registry or {})

    # ────────────────────────────── sandbox helpers ──────────────────────────

    def _workspace_root() -> Path:
        artefacts = getattr(discussion, "artefacts", None)
        if artefacts and hasattr(artefacts, "_get_workspace_root"):
            try:
                return Path(artefacts._get_workspace_root())
            except Exception:
                pass
        return Path(
            getattr(discussion, "workspace_data_path", None)
            or getattr(discussion, "_resolved_workspace", None)
            or getattr(discussion, "workspace_path", None)
            or "."
        ).resolve()

    def _artifact_content(art: Dict[str, Any]) -> str:
        artefacts = getattr(discussion, "artefacts", None)
        content = art.get("content") or ""
        if artefacts is not None:
            if art.get("type") == "data" and hasattr(artefacts, "_get_lam_content"):
                content = artefacts._get_lam_content(art) or content
            elif not content and hasattr(artefacts, "_read_content_from_disk"):
                content = artefacts._read_content_from_disk(art)
        return content or ""

    def _resolve_context_files(
        names: List[str]
    ) -> tuple:
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []

        artefacts = getattr(discussion, "artefacts", None)
        ws_root = _workspace_root()

        for name in names:
            if artefacts:
                art = artefacts.get(name)
                if art is not None:
                    resolved.append(art)
                    continue

            rel_path = name.replace("\\", "/").lstrip("/")
            if rel_path.startswith("workspace_data/"):
                rel_path = rel_path[len("workspace_data/"):]
            if not rel_path or ".." in Path(rel_path).parts:
                missing.append(name)
                continue
            try:
                candidate = (ws_root / rel_path).resolve()
                candidate.relative_to(ws_root)
                if candidate.is_file():
                    content = candidate.read_text(encoding="utf-8", errors="ignore")
                    resolved.append({
                        "title": rel_path,
                        "type": "document",
                        "content": content,
                        "language": Path(rel_path).suffix.lstrip(".") or None,
                    })
                    continue
            except (PermissionError, ValueError, OSError) as ex:
                ASCIIColors.warning(
                    f"[SpinoffTools] Context file '{name}' rejected: {ex}"
                )
            missing.append(name)

        return resolved, missing

    def _resolve_skills(titles: List[str]) -> tuple:
        resolved: List[Dict[str, str]] = []
        missing: List[str] = []

        if skills_manager is None or not titles:
            return resolved, missing

        for title in titles:
            try:
                content = skills_manager.load_skill(title)
            except Exception:
                content = None
            if content:
                resolved.append({"title": title, "content": content})
            else:
                missing.append(title)

        return resolved, missing

    def _build_persona_system_prompt(
        config: SpinoffConfig,
        skills: List[Dict[str, str]],
        files: List[Dict[str, Any]],
        missing_files: List[str],
        missing_skills: List[str],
    ) -> str:
        from lollms_client.lollms_agentic.prompts import WORKER_SYSTEM_PROMPT

        sections: List[str] = []
        sections.append(WORKER_SYSTEM_PROMPT)

        if config.persona_directives:
            sections.append(
                "=== SPECIALIST PERSONA (conditioned by the orchestrator) ===\n"
                f"{config.persona_directives}\n"
                "=== END SPECIALIST PERSONA ==="
            )

        if skills:
            lines = ["=== CONDITIONED SKILLS (READ-ONLY — never update, patch, or append) ==="]
            for skill in skills:
                lines.append(f"\n--- Skill: {skill['title']} ---\n")
                lines.append(skill["content"])
                lines.append(f"\n--- End Skill: {skill['title']} ---")
            lines.append("=== END CONDITIONED SKILLS ===")
            sections.append("\n".join(lines))

        if config.inline_skill:
            sections.append(
                "=== INLINE SKILL DOCTRINE (READ-ONLY — never update, patch, or append) ===\n"
                f"{config.inline_skill}\n"
                "=== END INLINE SKILL DOCTRINE ==="
            )

        if files:
            lines = ["=== CONDITIONED CONTEXT FILES (name ONLY files from this list) ==="]
            for art in files:
                title = art.get("title", "untitled")
                content = _artifact_content(art)
                lines.append(f'<file path="{title}">\n{content}\n</file>')
            lines.append("=== END CONTEXT FILES ===")
            sections.append("\n".join(lines))

        if missing_files:
            listed = ", ".join(f"`{m}`" for m in missing_files)
            sections.append(
                f"NOTE: The following requested files were NOT found in the "
                f"workspace and are unavailable: {listed}."
            )
        if missing_skills:
            listed = ", ".join(f"`{m}`" for m in missing_skills)
            sections.append(
                f"NOTE: The following requested skills were NOT found in the "
                f"library and are unavailable: {listed}."
            )

        sections.append(
            "=== COMPLETION CONTRACT ===\n"
            "When your task is complete, write your final result wrapped exactly as:\n"
            "<report>\n...what was done, results, files created/modified...\n</report>\n"
            "Then emit `<done/>` on a new line.\n"
            "=== END COMPLETION CONTRACT ==="
        )

        return "\n\n".join(sections)

    def _filter_tools(whitelist: tuple) -> Dict[str, Dict[str, Any]]:
        if not whitelist:
            return {}
        return {
            name: spec
            for name, spec in parent_registry.items()
            if name in whitelist
        }

    # ────────────────────────────────── tool ─────────────────────────────────

    def tool_spinoff_agent(
        agent_name: str = "",
        task: str = "",
        persona: str = "",
        tools: str = "",
        skills: str = "",
        inline_skill: str = "",
        context_files: str = "",
        max_rounds: int = 6,
        temperature: float = 0.3,
    ) -> dict:
        """
        Spawns a fully conditioned, specialized sub-agent in an isolated, focused sandbox.
        Use this to delegate heavy cognitive work (code, analysis, parsing, design, research)
        to a bespoke specialist you define, without polluting the main conversation.

        Args:
            agent_name (str): A short name for the specialist (e.g. "SQLOptimizer"). Defaults to "".
            task (str): Precise, self-contained instructions the specialist must execute.
            persona (str, optional): Custom persona conditioning: expertise, tone, methodology,
                strict rules. This defines WHO the specialist is. Defaults to "".
            tools (str, optional): Comma-separated tool names to grant the specialist.
                Must be a subset of currently available tools. Defaults to "" (no tools).
            skills (str, optional): Comma-separated skill titles to load from the library
                into the specialist's system prompt. Defaults to "".
            inline_skill (str, optional): A complete skill/methodology written inline,
                injected as read-only doctrine for the specialist. Defaults to "".
            context_files (str, optional): Comma-separated workspace file names the
                specialist may read. Defaults to "" (workspace tree only).
            max_rounds (int, optional): Bounded reasoning budget for the specialist. Defaults to 6.
            temperature (float, optional): Sampling temperature for the specialist. Defaults to 0.3.

        Returns:
            dict: {"success": bool, "output": str (the specialist's report), "files": list}
        """
        try:
            # ── Input validation & bounding ──
            if not task or not task.strip():
                return {"success": False, "error": "Parameter 'task' is required and must be non-empty."}

            clean_name = (agent_name or "").strip()[:64] or "Specialist"
            bounded_task = task.strip()[:_MAX_TASK_CHARS]
            bounded_persona = (persona or "").strip()[:_MAX_PERSONA_CHARS]
            bounded_inline_skill = (inline_skill or "").strip()[:_MAX_INLINE_SKILL_CHARS]

            tool_names = [
                t.strip() for t in (tools or "").split(",") if t.strip()
            ][:_MAX_CONTEXT_FILES]
            skill_titles = [
                s.strip() for s in (skills or "").split(",") if s.strip()
            ][:_MAX_SKILL_TITLES]
            file_names = [
                f.strip() for f in (context_files or "").split(",") if f.strip()
            ][:_MAX_CONTEXT_FILES]

            try:
                bounded_rounds = max(2, min(int(max_rounds), 12))
            except (TypeError, ValueError):
                bounded_rounds = 6
            try:
                bounded_temp = max(0.0, min(float(temperature), 2.0))
            except (TypeError, ValueError):
                bounded_temp = 0.3

            unknown_tools = [t for t in tool_names if t not in parent_registry]
            granted_tools = _filter_tools(tuple(tool_names))

            resolved_files, missing_files = _resolve_context_files(file_names)
            resolved_skills, missing_skills = _resolve_skills(skill_titles)

            config = SpinoffConfig(
                name=clean_name,
                task=bounded_task,
                persona_directives=bounded_persona,
                tools_whitelist=tuple(tool_names),
                skill_titles=tuple(skill_titles),
                inline_skill=bounded_inline_skill,
                context_files=tuple(file_names),
                max_rounds=bounded_rounds,
                temperature=bounded_temp,
            )

            system_prompt = _build_persona_system_prompt(
                config, resolved_skills, resolved_files,
                missing_files, missing_skills,
            )

            ASCIIColors.info(
                f"[SpinoffTools] Spinning off '{clean_name}' — task: {bounded_task[:80]}..., "
                f"tools: {len(granted_tools)}, skills: {len(resolved_skills)}, "
                f"files: {len(resolved_files)}, rounds: {bounded_rounds}"
            )

            from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
            worker_persona = LollmsPersonality(
                name=clean_name,
                system_prompt=system_prompt,
                lollms_client=discussion.lollmsClient,
            )

            pre_run_tip = getattr(discussion, "active_branch_id", None)
            result: Dict[str, Any] = {}
            spinoff_error: Optional[str] = None

            def spinoff_stream_relay(chunk: str, msg_type=None, meta=None) -> bool:
                if callback is None:
                    return True
                try:
                    mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
                    return callback(chunk, mt, meta or {})
                except Exception:
                    return True

            try:
                is_discussion = hasattr(discussion, "add_message") and hasattr(discussion, "active_branch_id")
                if is_discussion:
                    result = discussion.chat(
                        user_message=bounded_task,
                        personality=worker_persona,
                        add_user_message=True,
                        tools=granted_tools or None,
                        enable_artefacts=True,
                        enable_memory=False,
                        enable_episodic_memory=False,
                        enable_auto_dream=False,
                        enable_deep_memory_pulling=False,
                        prehydrate_rag=False,
                        max_nb_rounds=bounded_rounds,
                        enable_notes=False,
                        enable_skills=False,
                        enable_forms=False,
                        orchestrator_mode=False,
                        event_mode=EventMode.SILENT_MODE,
                        streaming_callback=spinoff_stream_relay,
                    )
                else:
                    result = discussion.chat(
                        prompt=bounded_task,
                        tools=granted_tools or None,
                        enable_artefacts=True,
                        use_internal_history=False,
                        max_nb_rounds=bounded_rounds,
                        orchestrator_mode=False,
                        event_mode=EventMode.SILENT_MODE,
                        streaming_callback=spinoff_stream_relay,
                    )
            except Exception as ex:
                trace_exception(ex)
                spinoff_error = f"Spinoff runtime crashed: {ex}"
            finally:
                if hasattr(discussion, "active_branch_id"):
                    try:
                        current_tip = getattr(discussion, "active_branch_id", None)
                        if current_tip and current_tip != pre_run_tip:
                            discussion.remove_message(current_tip)
                        if pre_run_tip and getattr(discussion, "active_branch_id", None) != pre_run_tip:
                            discussion.active_branch_id = pre_run_tip
                    except Exception as detach_ex:
                        ASCIIColors.warning(f"[SpinoffTools] Branch detach failed: {detach_ex}")

            # Guarantee physical disk synchronization of spinoff-created files.
            try:
                artefacts = getattr(discussion, "artefacts", None)
                if artefacts:
                    artefacts.sync_all_active_to_disk()
            except Exception as sync_err:
                ASCIIColors.warning(f"[SpinoffTools] Post-run disk sync: {sync_err}")

            ai_message = (result or {}).get("ai_message")
            full_text = getattr(ai_message, "content", "") or (result or {}).get("response", "") or ""
            from lollms_client.lollms_discussion._mixin_chat import _scrub_for_llm_context
            raw_report = extract_worker_report(_scrub_for_llm_context(full_text))

            spawned_files: List[str] = [
                a.get("title", "")
                for a in (result or {}).get("artefacts", [])
                if isinstance(a, dict)
            ]

            if spinoff_error and not raw_report:
                raw_report = f"Spinoff agent failed before producing a report. {spinoff_error}"
            elif spinoff_error:
                raw_report = f"{raw_report}\n\n[Spinoff runtime error: {spinoff_error}]"

            if len(raw_report) > _MAX_REPORT_CHARS:
                raw_report = (
                    raw_report[:_MAX_REPORT_CHARS]
                    + f"\n... [report truncated, {len(raw_report) - _MAX_REPORT_CHARS} more chars]"
                )

            notes: List[str] = []
            if unknown_tools:
                notes.append(f"unknown tools ignored: {', '.join(unknown_tools)}")
            if missing_files:
                notes.append(f"missing files: {', '.join(missing_files)}")
            if missing_skills:
                notes.append(f"missing skills: {', '.join(missing_skills)}")
            if notes:
                raw_report += "\n\n[Spinoff notes: " + "; ".join(notes) + "]"

            success = spinoff_error is None and bool(raw_report.strip())

            return {
                "success": success,
                "output": raw_report,
                "files": spawned_files,
            }
        except Exception as ex:
            trace_exception(ex)
            return {"success": False, "error": f"Spinoff agent failed: {ex}"}

    spinoffs["tool_spinoff_agent"] = {
        "name": "tool_spinoff_agent",
        "description": (
            "Spawn a fully conditioned, specialized sub-agent in an isolated sandbox. "
            "You define its persona (expertise, rules, methodology), grant it a subset of "
            "your tools, load skills from the library or inline, and give it workspace "
            "files to read. Ideal for heavy or focused cognitive work: code surgery, "
            "data analysis, document drafting, design, research synthesis."
        ),
        "parameters": [
            {"name": "agent_name", "type": "str", "description": "A short name for the specialist (e.g. 'SQLOptimizer')."},
            {"name": "task", "type": "str", "description": "Precise, self-contained instructions the specialist must execute."},
            {"name": "persona", "type": "str", "description": "Custom persona conditioning: expertise, tone, methodology, strict rules. This defines WHO the specialist is.", "optional": True},
            {"name": "tools", "type": "str", "description": "Comma-separated tool names to grant the specialist. Must be a subset of currently available tools.", "optional": True},
            {"name": "skills", "type": "str", "description": "Comma-separated skill titles to load from the library into the specialist's system prompt.", "optional": True},
            {"name": "inline_skill", "type": "str", "description": "A complete skill or methodology written inline, injected as read-only doctrine for the specialist.", "optional": True},
            {"name": "context_files", "type": "str", "description": "Comma-separated workspace file names the specialist may read.", "optional": True},
            {"name": "max_rounds", "type": "int", "description": "Bounded reasoning budget for the specialist (2-12).", "optional": True},
            {"name": "temperature", "type": "float", "description": "Sampling temperature for the specialist (0.0-2.0).", "optional": True},
        ],
        "callable": tool_spinoff_agent,
    }

    ASCIIColors.info(
        f"[SpinoffTools] Registered generic spinoff sub-agent factory "
        f"'tool_spinoff_agent' for the worker persona."
    )

    return spinoffs