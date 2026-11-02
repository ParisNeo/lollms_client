"""
models_tools.py — LCP toolset for model introspection, switching, and child-agent spawning.

Lets the LLM discover every configured LLM model profile (with autorouting
settings), hot-swap the active model, and run tasks on other models as
parallel background child agents.
"""
import threading
import time
import uuid
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "Models Switcher"
TOOL_LIBRARY_DESC = (
    "Switches the active LLM model on demand: lists every configured model "
    "profile with its capabilities and autorouting settings, hot-swaps the "
    "active model mid-session, and spawns child agents running on other models."
)
TOOL_LIBRARY_ICON = "🧠"
TOOL_LIBRARY_LOAD_FLAG = "enable_auto_model_selection"
TOOL_LIBRARY_ALIASES = [
    "model_switcher",
    "models",
    "model_tools",
    "model_selection",
    "model_selector",
]

_MAX_CONCURRENT_AGENTS = 8
_MAX_RESULT_CHARS = 8000

_CHILD_AGENTS: Dict[str, Dict[str, Any]] = {}
_REGISTRY_LOCK = threading.Lock()

# ── Authoritative editable schema of an LLM model profile ───────────────────
# One entry per field on LollmsModelProfile. Host UIs render one editor input
# per entry, pre-populated with models[*][field["name"]] (or field["default"]).
# Adding a new profile field requires updating ONLY this tuple.
_MODEL_PROFILE_FIELD_SCHEMA = (
    {"name": "name", "type": "str", "default": "", "editable": True,
     "description": "Profile alias (unique key in the registry)."},
    {"name": "binding_profile_name", "type": "str", "default": "", "editable": True,
     "description": "Binding profile providing the actual connection (host, API key, binding library)."},
    {"name": "model_name", "type": "str|null", "default": None, "editable": True,
     "description": "Underlying model identifier passed to the binding (e.g. 'gpt-4o', 'llama3')."},
    {"name": "is_default", "type": "bool", "default": False, "editable": True,
     "description": "Mark this profile as the client's default model."},
    {"name": "vision_enabled", "type": "bool", "default": False, "editable": True,
     "description": "Model accepts image input."},
    {"name": "video_enabled", "type": "bool", "default": False, "editable": True,
     "description": "Model accepts video input."},
    {"name": "glm_image_embedding", "type": "bool", "default": False, "editable": True,
     "description": "Use GLM-style image embeddings for retrieval (advanced)."},
    {"name": "image_token_cost", "type": "int", "default": 256, "editable": True,
     "description": "Approximate tokens charged per image when the binding gives no count_image_tokens."},
    {"name": "thinking_enabled", "type": "bool", "default": False, "editable": True,
     "description": "Model supports explicit thinking / reasoning mode."},
    {"name": "supported_reasoning_efforts", "type": "list[str]",
     "default": ["low", "medium", "high"], "editable": True,
     "description": "Ordered effort levels the model accepts (cheap to expensive)."},
    {"name": "default_reasoning_effort", "type": "str|null", "default": None, "editable": True,
     "description": "Effort used when the caller says 'think' with no level (null = middle of supported list)."},
    {"name": "forced_context_size", "type": "int|null", "default": None, "editable": True,
     "description": "Override the binding's context window in tokens."},
    {"name": "routing_config", "type": "dict", "default": {}, "editable": True,
     "description": "Autorouting settings (smart-router binding; usually empty)."},
)


def _require_client(lollms_client_instance) -> Any:
    if lollms_client_instance is None:
        raise RuntimeError(
            "This tool requires a host-injected LollmsClient instance, "
            "which is not available in this environment."
        )
    return lollms_client_instance


def _resolve_model_profile(client: Any, model_alias: str) -> Any:
    registry = getattr(client, "llm_model_profiles_registry", None) or {}
    profile = registry.get(model_alias)
    if profile is None:
        available = ", ".join(sorted(registry.keys())) or "none"
        raise ValueError(
            f"Unknown model alias '{model_alias}'. Available profiles: {available}. "
            "Use tool_list_models to discover configured models."
        )
    return profile


def _resolve_child_binding(client: Any, model_alias: str, profile: Any) -> Any:
    instance_cache = getattr(client, "llms", None)
    if isinstance(instance_cache, dict) and model_alias in instance_cache:
        return instance_cache[model_alias]

    binding = client._instantiate_binding_from_profile(
        model_alias,
        profile,
        client.llm_binding_manager,
        "llm",
    )
    if binding is None:
        raise RuntimeError(
            f"Failed to instantiate model '{model_alias}' "
            f"(binding profile '{profile.binding_profile_name}')."
        )
    if isinstance(instance_cache, dict):
        instance_cache[model_alias] = binding
    return binding


def _purge_finished_agents() -> int:
    finished = [
        agent_id
        for agent_id, record in _CHILD_AGENTS.items()
        if record.get("status") in ("done", "error")
    ]
    for agent_id in finished:
        _CHILD_AGENTS.pop(agent_id, None)
    return len(finished)


def _run_child_agent(agent_id: str, client: Any, model_alias: str, binding: Any,
                     task: str, system_prompt: str, n_predict: Optional[int]) -> None:
    started = time.time()
    try:
        effective_system_prompt = system_prompt or "You are a helpful assistant."
        n_predict_kwargs = {"n_predict": n_predict} if n_predict else {}
        try:
            result = binding.generate_text(
                prompt=task,
                system_prompt=effective_system_prompt,
                **n_predict_kwargs,
            )
        except TypeError:
            fused_prompt = (
                f"{effective_system_prompt}\n\n{task}"
                if effective_system_prompt else task
            )
            try:
                result = binding.generate_text(fused_prompt, **n_predict_kwargs)
            except TypeError:
                result = binding.generate_text(fused_prompt)
        if isinstance(result, dict) and result.get("status") == "error":
            error_msg = result.get("message", str(result))
            with _REGISTRY_LOCK:
                _CHILD_AGENTS[agent_id].update(
                    status="error",
                    error=str(error_msg),
                    elapsed_seconds=round(time.time() - started, 2),
                )
            return

        output = result if isinstance(result, str) else str(result)
        if len(output) > _MAX_RESULT_CHARS:
            output = output[:_MAX_RESULT_CHARS] + f"\n... [{len(output) - _MAX_RESULT_CHARS} chars truncated]"

        with _REGISTRY_LOCK:
            _CHILD_AGENTS[agent_id].update(
                status="done",
                output=output,
                elapsed_seconds=round(time.time() - started, 2),
            )
    except BaseException as ex:
        with _REGISTRY_LOCK:
            _CHILD_AGENTS[agent_id].update(
                status="error",
                error=f"{type(ex).__name__}: {ex}",
                elapsed_seconds=round(time.time() - started, 2),
            )


def tool_list_models(lollms_client_instance=None) -> Dict[str, Any]:
    """
    Lists every configured LLM model profile with its model name, binding profile,
    default/active/instantiated flags, capabilities (vision, video, thinking),
    reasoning efforts (supported + default), per-image token cost, forced context
    size, and autorouting settings. Use this first to discover which model aliases
    exist before switching or spawning child agents.

    Args:
        None
    """
    try:
        client = _require_client(lollms_client_instance)
    except RuntimeError as ex:
        return {"success": False, "error": str(ex)}

    registry = getattr(client, "llm_model_profiles_registry", None) or {}
    instance_cache = getattr(client, "llms", None) or {}
    active_alias = getattr(client, "_active_llm_alias", None)

    if not registry:
        return {"success": True,
                "output": (
                    "No LLM model profiles are configured on this client. "
                    "See `field_schema` for the fields a new profile may declare."
                ),
                "models": [],
                "field_schema": list(_MODEL_PROFILE_FIELD_SCHEMA)}

    models = []
    lines = []
    for alias, profile in registry.items():
        entry = {
            "alias": alias,
            "model_name": getattr(profile, "model_name", None),
            "binding_profile": getattr(profile, "binding_profile_name", None),
            "is_default": bool(getattr(profile, "is_default", False)),
            "is_active": alias == active_alias,
            "is_instantiated": alias in instance_cache,
            "vision_enabled": bool(getattr(profile, "vision_enabled", False)),
            "video_enabled": bool(getattr(profile, "video_enabled", False)),
            "glm_image_embedding": bool(getattr(profile, "glm_image_embedding", False)),
            "image_token_cost": int(getattr(profile, "image_token_cost", 256)),
            "thinking_enabled": bool(getattr(profile, "thinking_enabled", False)),
            "supported_reasoning_efforts": list(
                getattr(profile, "supported_reasoning_efforts", None)
                or ["low", "medium", "high"]
            ),
            "default_reasoning_effort": getattr(profile, "default_reasoning_effort", None),
            "forced_context_size": getattr(profile, "forced_context_size", None),
            "routing_config": getattr(profile, "routing_config", None) or {},
        }
        models.append(entry)

        flags = []
        if entry["is_default"]:
            flags.append("default")
        if entry["is_active"]:
            flags.append("ACTIVE")
        if entry["is_instantiated"]:
            flags.append("loaded")
        if entry["vision_enabled"]:
            flags.append("vision")
        if entry["video_enabled"]:
            flags.append("video")
        if entry["thinking_enabled"]:
            flags.append("thinking")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        lines.append(
            f"- {alias} → {entry['binding_profile']} / "
            f"'{entry['model_name'] or 'server default'}'{flag_str}"
        )
        if entry["routing_config"]:
            lines.append(f"    autorouting: {entry['routing_config']}")
        if entry["glm_image_embedding"]:
            lines.append("    image embedding mode: glm")
        if entry["forced_context_size"] is not None:
            lines.append(f"    forced context size: {entry['forced_context_size']}")
        if entry["image_token_cost"] != 256:
            lines.append(f"    image token cost: {entry['image_token_cost']}")
        if entry["thinking_enabled"]:
            efforts = ", ".join(entry["supported_reasoning_efforts"]) or "low, medium, high"
            default_effort = entry["default_reasoning_effort"] or "(middle of list)"
            lines.append(
                f"    thinking: enabled — efforts: {efforts}; default: {default_effort}"
            )
        else:
            lines.append(
                f"    thinking: disabled (declared efforts: "
                f"{', '.join(entry['supported_reasoning_efforts'])})"
            )

    return {
        "success": True,
        "output": "Configured LLM models:\n" + "\n".join(lines),
        "models": models,
        "field_schema": list(_MODEL_PROFILE_FIELD_SCHEMA),
    }


def tool_get_model_profile_schema() -> Dict[str, Any]:
    """
    Returns the complete editable schema of an LLM model profile: every field
    name, type, default value, and description. Use this to render a profile
    editor that exposes *all* fields — including those currently at their
    default value (i.e. "not yet set"). This is the authoritative source of
    truth for profile fields; do not hardcode them in the UI.

    Args:
        None
    """
    return {
        "success": True,
        "output": (
            "Model profile editable fields:\n"
            + "\n".join(
                f"- {f['name']} ({f['type']}, default={f['default']!r}): {f['description']}"
                for f in _MODEL_PROFILE_FIELD_SCHEMA
            )
        ),
        "fields": list(_MODEL_PROFILE_FIELD_SCHEMA),
    }


def tool_switch_model(model_alias: str, lollms_client_instance=None) -> Dict[str, Any]:
    """
    Switches the active LLM to a configured model profile. All subsequent
    generation in this session uses the new model. Returns the new active model
    on success or an actionable error if the alias is unknown or instantiation fails.

    Args:
        model_alias (str): The profile alias to activate (discover aliases with tool_list_models).
    """
    try:
        client = _require_client(lollms_client_instance)
        profile = _resolve_model_profile(client, str(model_alias).strip())
    except (RuntimeError, ValueError) as ex:
        return {"success": False, "error": str(ex)}

    try:
        if not client.switch_model(str(model_alias).strip()):
            return {
                "success": False,
                "error": f"Failed to switch to '{model_alias}' "
                         f"(binding profile '{profile.binding_profile_name}' could not be instantiated).",
            }
    except Exception as ex:
        return {"success": False, "error": f"Model switch raised an error: {ex}"}

    return {
        "success": True,
        "output": (
            f"Active model switched to '{model_alias}' "
            f"(model: '{profile.model_name or 'server default'}')."
        ),
        "active_alias": str(model_alias).strip(),
    }


def tool_spawn_child_agent(task: str, model_alias: str, system_prompt: str = "",
                           agent_name: str = "", n_predict: Optional[int] = None,
                           lollms_client_instance=None) -> Dict[str, Any]:
    """
    Spawns a background child agent that runs a task on a different model without
    changing the active model. Returns an agent_id immediately; poll the result with
    tool_get_child_agent_result. Multiple children can run in parallel on different
    models. Note: local models may contend for VRAM — prefer remote models for parallel children.

    Args:
        task (str): The instruction the child agent must complete.
        model_alias (str): The profile alias of the model the child runs on.
        system_prompt (str): Optional system prompt for the child agent.
        agent_name (str): Optional human-readable label for the child agent.
        n_predict (int): Optional maximum number of tokens the child may generate.
    """
    try:
        client = _require_client(lollms_client_instance)
        model_alias = str(model_alias).strip()
        profile = _resolve_model_profile(client, model_alias)
        binding = _resolve_child_binding(client, model_alias, profile)
    except (RuntimeError, ValueError) as ex:
        return {"success": False, "error": str(ex)}

    max_tokens = None
    if n_predict is not None:
        try:
            max_tokens = int(n_predict)
        except (TypeError, ValueError):
            max_tokens = None

    with _REGISTRY_LOCK:
        if len(_CHILD_AGENTS) >= _MAX_CONCURRENT_AGENTS:
            purged = _purge_finished_agents()
            if len(_CHILD_AGENTS) >= _MAX_CONCURRENT_AGENTS:
                return {
                    "success": False,
                    "error": (
                        f"Maximum of {_MAX_CONCURRENT_AGENTS} concurrent child agents reached. "
                        "Poll and collect finished results before spawning more."
                    ),
                }
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        _CHILD_AGENTS[agent_id] = {
            "status": "running",
            "agent_name": agent_name or agent_id,
            "model_alias": model_alias,
            "model_name": profile.model_name,
            "task": task,
            "output": None,
            "error": None,
            "elapsed_seconds": None,
        }

    thread = threading.Thread(
        target=_run_child_agent,
        args=(agent_id, client, model_alias, binding, task,
              str(system_prompt or ""), max_tokens),
        name=f"models_tools_{agent_id}",
        daemon=True,
    )
    thread.start()

    return {
        "success": True,
        "output": (
            f"Child agent '{agent_name or agent_id}' spawned on model '{model_alias}' "
            f"({agent_id}). Poll tool_get_child_agent_result(agent_id='{agent_id}') "
            "to retrieve its output."
        ),
        "agent_id": agent_id,
    }


def tool_get_child_agent_result(agent_id: str, remove_if_done: bool = True) -> Dict[str, Any]:
    """
    Retrieves the status and output of a spawned child agent. Returns 'running' while
    the child is still generating, or the final output/error once finished. Finished
    entries are removed from the registry by default after being collected.

    Args:
        agent_id (str): The agent_id returned by tool_spawn_child_agent.
        remove_if_done (bool): Remove the agent record once its result is collected.
    """
    key = str(agent_id).strip()
    with _REGISTRY_LOCK:
        record = _CHILD_AGENTS.get(key)
        if record is None:
            return {
                "success": False,
                "error": f"Unknown agent_id '{key}'. It may have already been collected.",
                "status": "not_found",
            }
        snapshot = dict(record)
        if remove_if_done and record.get("status") in ("done", "error"):
            _CHILD_AGENTS.pop(key, None)

    status = snapshot.get("status")
    if status == "done":
        return {
            "success": True,
            "output": (
                f"Agent '{snapshot.get('agent_name', key)}' (model "
                f"'{snapshot.get('model_alias')}') finished in "
                f"{snapshot.get('elapsed_seconds')}s:\n\n{snapshot.get('output', '')}"
            ),
            "status": "done",
            "result": snapshot.get("output"),
        }
    if status == "error":
        return {
            "success": False,
            "error": f"Agent '{snapshot.get('agent_name', key)}' failed: {snapshot.get('error')}",
            "status": "error",
        }
    return {
        "success": True,
        "output": (
            f"Agent '{snapshot.get('agent_name', key)}' is still running on "
            f"'{snapshot.get('model_alias')}'. Poll again later."
        ),
        "status": "running",
    }