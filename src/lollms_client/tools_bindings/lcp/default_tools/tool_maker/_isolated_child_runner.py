"""
_isolated_child_runner — Sandbox-side runner for the RSI test agent.

This module runs INSIDE a throw-away subprocess spawned by
``tool_maker.tool_run_test_agent``. It only uses:

  * The toolset folder placed on ``sys.path``.
  * The minimal LCP binding (discovery + execute_tool) — no workspace,
    no shell, no python, no filesystem helpers.

The child agent is the model itself, driven by a hard-capped reasoning
loop that can ONLY call tools originating from the candidate toolset file.

Invocation contract (stdin JSON):
    {
        "toolset_dir":    "/abs/path/to/toolset_folder",
        "toolset_module": "/abs/path/to/toolset_folder/TOOLS.py",
        "test_task":      "natural-language test directive",
        "max_steps":      6,
        "timeout_sec":    120,
        "llm_binding_profiles": {...},   # forwarded so the child can talk to the model
        "llm_model_profiles":   {...},
        "system_dir":     "/abs/path/to/.lollms_client",
        "workspace_dir":  "/abs/path/to/workspace",  # informational only; not used to grant access
    }

stdout: a single JSON line (result envelope). Everything else goes to stderr.
"""
from __future__ import annotations

import json
import os
import sys
import signal
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# Defensive imports — these MUST be present; a missing binding means the
# orchestrator misconfigured the subprocess and we abort loudly.
try:
    from lollms_client import LollmsClient
    from lollms_client.lollms_types import MSG_TYPE, EventMode
except Exception as _imp_err:  # pragma: no cover - configuration error
    print(json.dumps({
        "ok": False,
        "error": f"Child runner cannot import lollms_client: {_imp_err}",
    }))
    sys.exit(2)


# Hard limits — the orchestrator already enforces these, but we re-enforce
# inside the subprocess so a hostile candidate toolset cannot escalate.
_HARD_MAX_STEPS = 12
_HARD_TOOL_NAME_ALLOWLIST_PREFIX = "tool_"


def _guarded_tool_execute(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run one tool from the candidate toolset with deep defensive guards."""
    if not isinstance(tool_name, str) or not tool_name.startswith(_HARD_TOOL_NAME_ALLOWLIST_PREFIX):
        return {"success": False, "error": f"Refused: '{tool_name}' is not a tool_* function."}
    try:
        result = _CHILD_CLIENT.tools.execute_tool(tool_name, dict(params or {}))
        if not isinstance(result, dict):
            result = {"success": True, "output": result}
        return result
    except BaseException as be:
        return {
            "success": False,
            "error": f"{type(be).__name__}: {be}",
            "traceback": traceback.format_exc(),
        }


def _extract_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Very permissive extraction of {"name": "tool_*", "parameters": {...}} blobs."""
    import re
    out: List[Dict[str, Any]] = []
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"(tool_[A-Za-z0-9_]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\s*\}',
        text or "",
        re.DOTALL,
    ):
        try:
            params = json.loads(m.group(2))
        except Exception:
            params = {}
        out.append({"name": m.group(1), "parameters": params})
    return out


def _run_loop(
    test_task: str,
    max_steps: int,
    available_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Hard-capped reasoning loop. Children CANNOT recurse into sub-agents."""
    tool_names = [t.get("name") for t in available_tools if t.get("name", "").startswith("tool_")]
    if not tool_names:
        return {
            "ok": False,
            "error": "Candidate toolset exposes zero tool_* functions — nothing to test.",
            "transcript": [],
        }

    tools_brief = "\n".join(
        f"- {t.get('name')}: {(t.get('description') or '')[:120]}"
        for t in available_tools
        if t.get("name", "").startswith("tool_")
    )
    system_prompt = (
        "You are a CONSTRAINED test agent for a candidate LCP toolset. "
        "You may call ONLY the tools listed below. "
        "You have NO shell, NO Python execution, NO filesystem helpers, "
        "NO sub-agents, NO workspace tools. "
        "Keep your reasoning short. When the test directive is satisfied "
        "(or you have proven the tool is broken), respond with a single "
        "line: DONE: <one-sentence verdict>.\n\n"
        f"Available tools:\n{tools_brief}\n"
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
            response = _CHILD_CLIENT.generate_from_messages(conversation, n_predict=512, temperature=0.2)
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
            # No tool call, no DONE → prompt the child to be explicit and move on.
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
                tool_results.append(f"BLOCKED: '{name}' is not in the candidate toolset.")
                continue
            result = _guarded_tool_execute(name, call.get("parameters") or {})
            transcript.append({"role": "tool_call", "tool": name, "result": result})
            truncated = json.dumps(result, default=str)[:2000]
            tool_results.append(f"{name} -> {truncated}")

        conversation.append({"role": "assistant", "content": text})
        conversation.append({
            "role": "user",
            "content": "Tool results:\n" + "\n".join(tool_results),
        })

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
        print(json.dumps({"ok": False, "error": "Child runner received empty stdin."}))
        sys.exit(2)

    try:
        payload = json.loads(raw)
    except Exception as je:
        print(json.dumps({"ok": False, "error": f"Bad stdin payload: {je}"}))
        sys.exit(2)

    toolset_dir = Path(payload["toolset_dir"]).resolve()
    toolset_module = Path(payload["toolset_module"]).resolve()
    test_task = str(payload.get("test_task") or "").strip()
    max_steps = min(int(payload.get("max_steps", 6) or 6), _HARD_MAX_STEPS)

    if not toolset_module.exists():
        print(json.dumps({"ok": False, "error": f"Candidate toolset file missing: {toolset_module}"}))
        sys.exit(2)

    # Constrain sys.path to ONLY the candidate toolset folder — no orchestrator,
    # no host tools, no workspace. lollms_client itself is importable from the
    # subprocess's pre-set PYTHONPATH.
    sys.path[:] = [str(toolset_dir)] + [p for p in sys.path if "site-packages" in p or "python" in p.lower()]

    # Build a MINIMAL LollmsClient with only the candidate toolset mounted and
    # only the child-relevant modalities — nothing else gets wired.
    global _CHILD_CLIENT
    try:
        _CHILD_CLIENT = LollmsClient(
            system_dir=payload.get("system_dir"),
            llm_binding_profiles=payload.get("llm_binding_profiles") or {},
            llm_model_profiles=payload.get("llm_model_profiles") or {},
            tools_binding_name="lcp",
            tools_binding_config={
                "tools_folders": [str(toolset_dir)],
                "host_tool_configs": {},
                "system_dir": payload.get("system_dir"),
                "cwd": str(toolset_dir),
            },
            debug=False,
        )
    except BaseException as ce:
        print(json.dumps({"ok": False, "error": f"Failed to build child LollmsClient: {ce}"}))
        sys.exit(2)

    # Discover only tools defined in the candidate file.
    available = []
    try:
        for t in _CHILD_CLIENT.tools.discover_tools():
            py_path = Path(t.get("_python_file_path") or "").resolve()
            if py_path == toolset_module:
                available.append(t)
    except BaseException as de:
        print(json.dumps({"ok": False, "error": f"Tool discovery failed: {de}"}))
        sys.exit(2)

    if not available:
        print(json.dumps({"ok": False, "error": "No tools discovered in the candidate file."}))
        sys.exit(0)

    result = _run_loop(test_task or "Run every tool in this toolset once with a minimal valid argument set.", max_steps, available)
    print(json.dumps({**result, "available_tools": [t.get("name") for t in available]}, default=str))


_CHILD_CLIENT: Any = None

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