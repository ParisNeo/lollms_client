"""
_isolated_skill_runner — sandbox runner for ``tool_run_skill_test_agent``.

Runs INSIDE a throw-away subprocess. Its only job is to run a hard-capped
reasoning loop with the LLM equipped with EXACTLY the tools declared in the
skill's YAML frontmatter (``tools:`` list). Anything else is rejected.

Invocation contract (stdin JSON):
    {
        "skill_file":       "/abs/path/to/SKILL.md",
        "declared_tools":   ["tool_load_skill", "tool_list_skills", ...],
        "candidate_skills_dirs": ["/abs/skills/dir/1", ...],
        "test_task":        "natural-language directive",
        "max_steps":        6,
        "timeout_sec":      120,
        "llm_binding_profiles": {...},
        "llm_model_profiles":   {...},
        "system_dir":       "/abs/path/to/.lollms_client",
    }

stdout: a single JSON line (result envelope). Everything else → stderr.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from lollms_client import LollmsClient
except Exception as _imp_err:
    print(json.dumps({
        "ok": False,
        "error": f"Child runner cannot import lollms_client: {_imp_err}",
    }))
    sys.exit(2)


_HARD_MAX_STEPS = 12
_CHILD_CLIENT: Any = None


def _guarded_tool_execute(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool_name, str) or not tool_name.startswith("tool_"):
        return {"success": False, "error": f"Refused: '{tool_name}' is not a tool_* function."}
    try:
        result = _CHILD_CLIENT.tools.execute_tool(tool_name, dict(params or {}))
        if not isinstance(result, dict):
            result = {"success": True, "output": result}
        return result
    except BaseException as be:
        return {"success": False, "error": f"{type(be).__name__}: {be}", "traceback": traceback.format_exc()}


def _extract_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    import re
    out: List[Dict[str, Any]] = []
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"(tool_[A-Za-z0-9_]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\s*\}',
        text or "", re.DOTALL,
    ):
        try:
            params = json.loads(m.group(2))
        except Exception:
            params = {}
        out.append({"name": m.group(1), "parameters": params})
    return out


def _run_loop(
    skill_file: Path,
    declared_tools: List[str],
    test_task: str,
    max_steps: int,
    available: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tool_names = [t.get("name") for t in available if t.get("name", "").startswith("tool_")]
    if not tool_names:
        return {
            "ok": False,
            "error": "The skill's declared tools are not available. Mount them first.",
            "transcript": [],
        }

    tools_brief = "\n".join(
        f"- {t.get('name')}: {(t.get('description') or '')[:120]}"
        for t in available if t.get("name", "").startswith("tool_")
    )
    system_prompt = (
        "You are a CONSTRAINED test agent exercising a candidate SKILL.md. "
        "You may call ONLY the tools declared by that skill (listed below). "
        "You have NO shell, NO Python, NO workspace, NO sub-agents, NO skills "
        "management. Keep your reasoning short. When the skill definition has "
        "been vetted (either successfully or as broken), respond with a single "
        "line: DONE: <one-sentence verdict>.\n\n"
        f"Skill file: {skill_file}\nAvailable tools:\n{tools_brief}\n"
    )

    transcript: List[Dict[str, Any]] = []
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_task},
    ]
    steps_taken = 0

    for step in range(max_steps):
        steps_taken = step + 1
        try:
            response = _CHILD_CLIENT.generate_from_messages(
                conversation, n_predict=512, temperature=0.2
            )
        except BaseException as ge:
            return {
                "ok": False,
                "error": f"Child LLM call failed: {type(ge).__name__}: {ge}",
                "transcript": transcript,
                "steps": steps_taken,
            }

        text = response.strip() if isinstance(response, str) else ""
        transcript.append({"role": "assistant", "content": text})

        if text.startswith("DONE:"):
            return {
                "ok": True,
                "verdict": text[5:].strip(),
                "transcript": transcript,
                "steps": steps_taken,
            }

        calls = _extract_json_tool_calls(text)
        if not calls:
            conversation.append({"role": "assistant", "content": text})
            conversation.append({
                "role": "user",
                "content": "Emit a tool call as JSON on one line, or respond 'DONE: <verdict>'.",
            })
            continue

        tool_results: List[str] = []
        for call in calls:
            name = call["name"]
            if name not in tool_names:
                tool_results.append(
                    f"BLOCKED: '{name}' is not in the skill's declared tools."
                )
                continue
            result = _guarded_tool_execute(name, call.get("parameters") or {})
            transcript.append({"role": "tool_call", "tool": name, "result": result})
            tool_results.append(f"{name} -> {json.dumps(result, default=str)[:2000]}")

        conversation.append({"role": "assistant", "content": text})
        conversation.append({"role": "user", "content": "Tool results:\n" + "\n".join(tool_results)})

    return {
        "ok": False,
        "error": f"Child did not finish within {max_steps} steps.",
        "transcript": transcript,
        "steps": steps_taken,
    }


def _handle_sigterm(signum, frame):
    print(json.dumps({
        "ok": False,
        "error": f"Child received signal {signum} (timeout or cancel).",
    }))
    sys.exit(3)


def main() -> None:
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        pass

    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"ok": False, "error": "Empty stdin."}))
        sys.exit(2)

    try:
        payload = json.loads(raw)
    except Exception as je:
        print(json.dumps({"ok": False, "error": f"Bad stdin payload: {je}"}))
        sys.exit(2)

    skill_file = Path(payload["skill_file"]).resolve()
    declared_tools = list(payload.get("declared_tools") or [])
    test_task = str(payload.get("test_task") or "").strip()
    max_steps = min(int(payload.get("max_steps", 6) or 6), _HARD_MAX_STEPS)
    skills_dirs = [Path(p).resolve() for p in (payload.get("candidate_skills_dirs") or [])]

    if not declared_tools:
        print(json.dumps({
            "ok": False,
            "error": "Skill declares no tools — nothing to test or nothing was parsed.",
        }))
        sys.exit(0)

    global _CHILD_CLIENT
    try:
        # tools_folders pinned to the default LCP folder only — the child
        # cannot mount anything the operator has not already installed.
        from lollms_client import LollmsClient as _LC
        _pkg_root = Path(_LC.__module__ and __import__("lollms_client").__file__).resolve().parent
        from lollms_client.tools_bindings.lcp.default_tools import __name__ as _dt_name  # noqa: F401
    except Exception:
        _pkg_root = None

    try:
        _CHILD_CLIENT = LollmsClient(
            system_dir=payload.get("system_dir"),
            llm_binding_profiles=payload.get("llm_binding_profiles") or {},
            llm_model_profiles=payload.get("llm_model_profiles") or {},
            tools_binding_name="lcp",
            tools_binding_config={
                "tools_folders": [str(Path(__import__("lollms_client").__file__).resolve().parent / "tools_bindings" / "lcp" / "default_tools")],
                "host_tool_configs": {},
                "system_dir": payload.get("system_dir"),
                "cwd": str(skill_file.parent),
            },
            debug=False,
        )
    except BaseException as ce:
        print(json.dumps({"ok": False, "error": f"Child LollmsClient failed: {ce}"}))
        sys.exit(2)

    # Filter to ONLY the tools the skill declared.
    declared_set = {n for n in declared_tools if isinstance(n, str) and n.startswith("tool_")}
    available: List[Dict[str, Any]] = []
    try:
        for t in _CHILD_CLIENT.tools.discover_tools():
            if t.get("name") in declared_set:
                available.append(t)
    except BaseException as de:
        print(json.dumps({"ok": False, "error": f"Tool discovery failed: {de}"}))
        sys.exit(2)

    if not available:
        print(json.dumps({
            "ok": False,
            "error": f"None of the declared tools {sorted(declared_set)} are available in this environment.",
        }))
        sys.exit(0)

    result = _run_loop(
        skill_file=skill_file,
        declared_tools=declared_tools,
        test_task=test_task or "Confirm the candidate SKILL.md is coherent and its declared tools are callable.",
        max_steps=max_steps,
        available=available,
    )
    print(json.dumps({**result, "available_tools": [t.get("name") for t in available]}, default=str))


if __name__ == "__main__":
    try:
        main()
    except BaseException as top_exc:
        print(json.dumps({
            "ok": False,
            "error": f"Child runner crashed top-level: {type(top_exc).__name__}: {top_exc}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)