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
    default/active/instantiated flags, capabilities (vision, video), reasoning efforts,
    forced context size, and autorouting settings. Use this first to discover which
    model aliases exist before switching or spawning child agents.

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
                "output": "No LLM model profiles are configured on this client.",
                "models": []}

    models = []
    lines = []
    for alias, profile in registry.items():
        entry = {
            "alias": alias,
            "model_name": profile.model_name,
            "binding_profile": profile.binding_profile_name,
            "is_default": bool(profile.is_default),
            "is_active": alias == active_alias,
            "is_instantiated": alias in instance_cache,
            "vision_enabled": bool(profile.vision_enabled),
            "video_enabled": bool(profile.video_enabled),
            "glm_image_embedding": bool(profile.glm_image_embedding),
            "supported_reasoning_efforts": profile.supported_reasoning_efforts or [],
            "forced_context_size": profile.forced_context_size,
            "routing_config": profile.routing_config or {},
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
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        lines.append(
            f"- {alias} → {profile.binding_profile_name} / "
            f"'{profile.model_name or 'server default'}'{flag_str}"
        )
        if entry["routing_config"]:
            lines.append(f"    autorouting: {entry['routing_config']}")
        if entry["supported_reasoning_efforts"]:
            lines.append(
                f"    reasoning efforts: {', '.join(entry['supported_reasoning_efforts'])}"
            )

    return {
        "success": True,
        "output": "Configured LLM models:\n" + "\n".join(lines),
        "models": models,
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