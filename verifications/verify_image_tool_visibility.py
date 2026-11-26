
"""Diagnostic: does the LLM see a `generate_image` tool in a discussion?

The reported symptom is:
    "Tool call blocked: 'generate_image' is not registered in this session.
     I have activated generate_image when calling the chat!"

The codebase's documented contract (`docs/CHAT_BOOLEAN_FLAGS.md`
§`enable_image_generation` and `_mixin_prompt._build_image_generation_instructions`)
is unambiguous:

    Image generation is a TAG (`<generate_image>`), NOT an LCP tool.
    There is deliberately NO `generate_image` tool in the tool registry.

This probe verifies that contract WITHOUT a live LLM round-trip (the network
turn is optional and tightly guarded, because a live call can fail for
reasons unrelated to tool visibility — e.g. SQLAlchemy session binding).

It builds a client from the CLI env via `get_client_from_env(create_tti=True)`,
then reports:
  1. TTI capability (instance / callable / profile registry);
  2. whether the tool registry contains any `generate_image`-like name
     (it MUST NOT);
  3. whether the system-prompt doctrine advertises the `<generate_image>`
     tag and explicitly states "there is NO tool named generate_image".

Read-only: nothing is written to a project workspace.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _try_import_client():
    """Resolve `get_client_from_env` from any known import layout."""
    errors = []
    for modname in ("lollms_client.lollms_config_cli_env", "lollms_client"):
        try:
            mod = __import__(modname, fromlist=["get_client_from_env"])
            fn = getattr(mod, "get_client_from_env", None)
            if callable(fn):
                return fn, None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{modname}: {exc}")
    return None, "; ".join(errors)


def _probe_tti_capability(client) -> dict:
    tti = getattr(client, "tti", None)
    registry = getattr(client, "tti_model_profiles_registry", None)
    return {
        "tti_instance_present": tti is not None,
        "tti_callable_generate_image": callable(
            getattr(tti, "generate_image", None)
        )
        if tti is not None
        else False,
        "tti_model_profiles_registry_nonempty": bool(registry),
    }


def _image_like_tool_names(registry) -> list[str]:
    needles = ("generate_image", "edit_image", "image_gen", "generate_pic")
    names = registry.keys() if isinstance(registry, dict) else []
    return sorted(
        name for name in names if any(n in name.lower() for n in needles)
    )


def main() -> int:
    fn, import_err = _try_import_client()
    report: dict = {"import_error": import_err}
    if fn is None:
        report["verdict"] = "IMPORT_FAILED"
        print(json.dumps(report, indent=2))
        return 2

    try:
        client = fn(create_tti=True, create_llm=True, run_wizard_if_fail=False)
    except Exception as exc:  # noqa: BLE001
        report["client_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["verdict"] = "CLIENT_BUILD_FAILED"
        print(json.dumps(report, indent=2))
        return 3

    if client is None:
        report["verdict"] = "CLIENT_NONE"
        print(json.dumps(report, indent=2))
        return 3

    report["client_class"] = type(client).__name__
    report["tti_capability"] = _probe_tti_capability(client)

    # ── Registry inspection (no chat needed) ──
    tools_binding = getattr(client, "tools", None)
    discovered = (
        getattr(tools_binding, "discovered_tools", None)
        if tools_binding is not None
        else None
    )
    if not isinstance(discovered, dict):
        discovered = {}
    report["tool_registry_size"] = len(discovered)
    report["image_like_tools_in_registry"] = _image_like_tool_names(discovered)

    # ── Doctrine probe (the authoritative "does the model learn the tag?" test) ──
    try:
        from lollms_client.lollms_discussion._mixin_prompt import PromptMixin

        probe = PromptMixin.__new__(PromptMixin)  # bypass __init__
        probe.lollmsClient = client  # type: ignore[attr-defined]

        image_doctrine = PromptMixin._build_image_generation_instructions(
            probe, enabled=True
        )
        report["image_doctrine_chars"] = len(image_doctrine)
        report["image_doctrine_advertises_tag"] = "<generate_image" in image_doctrine
        report["image_doctrine_says_no_tool"] = (
            "There is NO tool named 'edit_image', 'generate_image'"
            in image_doctrine
        )
        report["image_doctrine_warns_phantom"] = (
            "blocked as a phantom" in image_doctrine
        )

        # Sanity: with TTI absent, the doctrine must be empty (self-gating).
        probe_no_tti = PromptMixin.__new__(PromptMixin)
        class _BlindClient:
            tti = None
            tti_model_profiles_registry = {}
        probe_no_tti.lollmsClient = _BlindClient()  # type: ignore[attr-defined]
        empty_doctrine = PromptMixin._build_image_generation_instructions(
            probe_no_tti, enabled=True
        )
        report["doctrine_empty_without_tti"] = empty_doctrine == ""
    except Exception as exc:  # noqa: BLE001
        report["doctrine_probe_error"] = f"{type(exc).__name__}: {exc}"
        report["doctrine_traceback"] = traceback.format_exc()

    # ── Verdict ──
    img_tools = report.get("image_like_tools_in_registry") or []
    if img_tools:
        report["verdict"] = (
            "UNEXPECTED: image-generation-like TOOL(s) registered "
            f"{img_tools}. This contradicts the tag-based design."
        )
    elif not report.get("image_doctrine_advertises_tag"):
        report["verdict"] = (
            "DOCTRINE_MISSING: <generate_image> not advertised. The TTI probe "
            "likely returned False, so the model never learns the tag, then "
            "invents a phantom tool."
        )
    elif not report.get("image_doctrine_says_no_tool"):
        report["verdict"] = (
            "DOCTRINE_INCOMPLETE: tag advertised but the 'no such tool' warning "
            "is missing."
        )
    else:
        report["verdict"] = (
            "OK: image generation is tag-based (no image tool in registry) AND "
            "the <generate_image> doctrine is advertised with an explicit "
            "'there is NO generate_image tool' warning. A block therefore means "
            "the MODEL emitted a phantom <tool> call against the doctrine — "
            "a model-compliance issue, not a registry/visibility issue."
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
