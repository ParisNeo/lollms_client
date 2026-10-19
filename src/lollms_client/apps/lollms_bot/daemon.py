import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ascii_colors import ASCIIColors
from lollms_client.apps.lollms_bot.config import BotConfig, get_bot_home, get_or_create_handbag

if TYPE_CHECKING:
    from lollms_client.lollms_client import LollmsClient
    from lollms_client.lollms_personality import LollmsPersonality


def _build_client() -> Optional["LollmsClient"]:
    """
    Builds the shared LollmsClient from the global configuration, reusing the
    exact same resolution path as lollms-code (config.yaml + .env + two-tier
    profiles). Requests both the LLM and the CONNECTION profiles so the client
    populates its connection bindings. Returns None when no LLM is configured.
    """
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        return get_client_from_env(
            create_llm=True,
            create_connection=True,
            run_wizard_if_fail=False,
        )
    except Exception as e:
        ASCIIColors.red(f"Could not build the LollmsClient: {e}")
        ASCIIColors.info("Configure one first: python -m lollms_client.lollms_config_cli_env")
        return None


def _read_persona_display_name(folder: Path) -> str:
    """Reads a persona folder's display name from its manifest or SOUL.md."""
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            name = manifest.get("name")
            if name:
                return str(name)
        except Exception:
            pass

    soul_path = folder / "SOUL.md"
    if soul_path.exists():
        try:
            for line in soul_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("name:"):
                    return stripped.split(":", 1)[1].strip()
        except Exception:
            pass

    return folder.name


def _scan_personas(config: BotConfig) -> Dict[str, Dict[str, Any]]:
    """
    Discovers handbag-shaped persona folders. A persona is any folder that
    contains a SOUL.md or manifest.json file. Scan locations:
    ~/.lollms-bot/personas/, ~/.lollms-bot/handbag/coworkers/, plus any extra
    directories listed in config['personas']['dirs'].
    """
    personas_cfg = (config.config.get("personas") or {})
    if not bool(personas_cfg.get("enabled", True)):
        return {}

    home = get_bot_home()
    search_roots: List[Path] = [home / "personas", home / "handbag" / "coworkers"]
    for extra in personas_cfg.get("dirs") or []:
        try:
            search_roots.append(Path(str(extra)).expanduser())
        except Exception:
            continue

    registry: Dict[str, Dict[str, Any]] = {}
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            if not ((folder / "SOUL.md").exists() or (folder / "manifest.json").exists()):
                continue
            registry[folder.name.lower()] = {
                "path": str(folder),
                "display_name": _read_persona_display_name(folder),
            }
    return registry


def build_personality(
    client: "LollmsClient",
    config: BotConfig,
    persona_name: Optional[str] = None,
) -> Optional["LollmsPersonality"]:
    """
    Builds the bot personality from the configured handbag, wired with
    the full personality option surface: handbag (SOUL.md, tools, skills,
    RAG, memory), workspace, capabilities, and the per-turn token budget.
    When persona_name is given and present in the persona registry, that
    persona's handbag folder is loaded instead of the main handbag.
    """
    from lollms_client.lollms_personality import LollmsPersonality, CapabilityFlags
    from lollms_client.lollms_personality.handbag import Handbag

    registry = _scan_personas(config)
    p_cfg = config.config.get("personality", {}) or {}

    if persona_name:
        match = next((key for key in registry if key.lower() == persona_name.lower()), None)
        if match is None:
            ASCIIColors.warning(f"Persona '{persona_name}' not found in the registry. Using the main handbag.")
            persona_name = None
        else:
            persona_handbag_path = Path(registry[match]["path"])
            ASCIIColors.info(f"Loading persona '{registry[match]['display_name']}' from {persona_handbag_path}")
            try:
                handbag = Handbag(persona_handbag_path)
            except Exception as e:
                ASCIIColors.red(f"Failed to load persona handbag '{persona_handbag_path}': {e}")
                return None
            personality = _finish_personality(LollmsPersonality.from_handbag(
                str(handbag.path),
                lollms_client=client,
            ), p_cfg)
            if personality is None:
                return None
            ASCIIColors.success(
                f"Persona '{personality.name}' loaded "
                f"(memory: {personality.memory_manager is not None}, "
                f"skills: {len(personality.skills_manager.skills) if personality.skills_manager else 0}, "
                f"rag sources: {len(personality.data_sources)})"
            )
            return personality

    if persona_name is None:
        startup_persona = p_cfg.get("persona_name")
        if startup_persona:
            return build_personality(client, config, persona_name=str(startup_persona))

    handbag_path = config.config.get("handbag_path")
    try:
        handbag = get_or_create_handbag(handbag_path)
    except Exception as e:
        ASCIIColors.red(f"Failed to prepare the handbag: {e}")
        return None

    personality = _finish_personality(LollmsPersonality.from_handbag(
        str(handbag.path),
        lollms_client=client,
    ), p_cfg)
    if personality is None:
        return None

    ASCIIColors.success(
        f"Personality '{personality.name}' loaded from handbag "
        f"(memory: {personality.memory_manager is not None}, "
        f"skills: {len(personality.skills_manager.skills) if personality.skills_manager else 0}, "
        f"rag sources: {len(personality.data_sources)})"
    )
    return personality


def _finish_personality(personality: "LollmsPersonality", p_cfg: Dict[str, Any]) -> Optional["LollmsPersonality"]:
    """Applies the configured role, capabilities, and token budget to a personality."""
    from lollms_client.lollms_personality import CapabilityFlags

    if personality is None:
        return None

    personality.role = str(p_cfg.get("role", "assistant"))
    personality.category = str(p_cfg.get("category", "assistant"))

    personality.capabilities = CapabilityFlags(
        enable_code_execution=bool(p_cfg.get("enable_code_execution", False)),
        enable_workspace_tools=bool(p_cfg.get("enable_workspace_tools", True)),
        enable_skill_loading=bool(p_cfg.get("enable_skill_loading", True)),
        enable_skill_creation=bool(p_cfg.get("enable_skill_creation", True)),
        skills_mode=str(p_cfg.get("skills_mode", "loadable")),
        enable_sub_agents=bool(p_cfg.get("enable_sub_agents", True)),
        enable_model_switching=bool(p_cfg.get("enable_model_switching", True)),
        enable_image_generation=bool(p_cfg.get("enable_image_generation", False)),
        enable_image_editing=bool(p_cfg.get("enable_image_editing", False)),
        enable_tts=bool(p_cfg.get("enable_tts", False)),
        enable_stt=bool(p_cfg.get("enable_stt", False)),
        enable_ttm=bool(p_cfg.get("enable_ttm", False)),
        enable_ttv=bool(p_cfg.get("enable_ttv", False)),
        enable_computer_use=bool(p_cfg.get("enable_computer_use", False)),
        max_sub_agent_depth=int(p_cfg.get("max_sub_agent_depth", 3)),
        max_sub_agents_per_turn=int(p_cfg.get("max_sub_agents_per_turn", 5)),
    )

    personality.max_tokens_per_turn = int(p_cfg.get("max_tokens_per_turn", 4096))
    return personality

    try:
        personality = LollmsPersonality.from_handbag(
            str(handbag.path),
            lollms_client=client,
        )
    except Exception as e:
        ASCIIColors.red(f"Failed to load the personality from the handbag: {e}")
        return None

    p_cfg = config.config.get("personality", {}) or {}

    personality.role = str(p_cfg.get("role", "assistant"))
    personality.category = str(p_cfg.get("category", "assistant"))

    personality.capabilities = CapabilityFlags(
        enable_code_execution=bool(p_cfg.get("enable_code_execution", False)),
        enable_workspace_tools=bool(p_cfg.get("enable_workspace_tools", True)),
        enable_skill_loading=bool(p_cfg.get("enable_skill_loading", True)),
        enable_skill_creation=bool(p_cfg.get("enable_skill_creation", True)),
        skills_mode=str(p_cfg.get("skills_mode", "loadable")),
        enable_sub_agents=bool(p_cfg.get("enable_sub_agents", True)),
        enable_model_switching=bool(p_cfg.get("enable_model_switching", True)),
        enable_image_generation=bool(p_cfg.get("enable_image_generation", False)),
        enable_image_editing=bool(p_cfg.get("enable_image_editing", False)),
        enable_tts=bool(p_cfg.get("enable_tts", False)),
        enable_stt=bool(p_cfg.get("enable_stt", False)),
        enable_ttm=bool(p_cfg.get("enable_ttm", False)),
        enable_ttv=bool(p_cfg.get("enable_ttv", False)),
        enable_computer_use=bool(p_cfg.get("enable_computer_use", False)),
        max_sub_agent_depth=int(p_cfg.get("max_sub_agent_depth", 3)),
        max_sub_agents_per_turn=int(p_cfg.get("max_sub_agents_per_turn", 5)),
    )

    personality.max_tokens_per_turn = int(p_cfg.get("max_tokens_per_turn", 4096))

    ASCIIColors.success(
        f"Personality '{personality.name}' loaded from handbag "
        f"(memory: {personality.memory_manager is not None}, "
        f"skills: {len(personality.skills_manager.skills) if personality.skills_manager else 0}, "
        f"rag sources: {len(personality.data_sources)})"
    )
    return personality


def _active_connection_bindings(client) -> List:
    registry = getattr(client, "connections", None)
    if isinstance(registry, dict) and registry:
        return list(registry.values())
    active = getattr(client, "connection", None)
    if active is not None:
        return [active]
    return []


def _attach_personality(
    bindings: List,
    personality,
    personality_factory=None,
    persona_registry=None,
    on_persona_selected=None,
) -> None:
    """
    Attaches the shared personality to each binding so they generate replies
    through the personality's full chat pipeline (tools, skills, memory),
    plus the persona-switching surface used by slash commands.
    """
    for binding in bindings:
        try:
            binding.lollms_personality = personality
            binding.lollms_client = personality.lollms_client
            binding.personality_factory = personality_factory
            binding.persona_registry = persona_registry
            binding.on_persona_selected = on_persona_selected
        except Exception:
            pass


def _attach_client(bindings: List, client) -> None:
    """Fallback attachment of the shared client (legacy reply path)."""
    for binding in bindings:
        try:
            binding.lollms_client = client
        except Exception:
            pass


async def _start_connections(bindings: List) -> None:
    tasks = [b.start() for b in bindings if hasattr(b, "start")]
    if not tasks:
        ASCIIColors.yellow("No connection bindings to start.")
        return
    ASCIIColors.info(f"Starting {len(tasks)} connection binding(s)...")
    await asyncio.gather(*tasks)


class BotDaemon:
    """
    Thin host that loads the handbag-based personality, builds the shared
    LollmsClient, attaches the personality to the configured connection
    bindings, and starts them until interrupted.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.client = _build_client()
        self.persona_registry = _scan_personas(config)
        startup_persona = (config.config.get("personality", {}) or {}).get("persona_name")
        self.personality: Optional["LollmsPersonality"] = build_personality(
            self.client, config, persona_name=startup_persona
        )
        self.personality_factory = (
            lambda name: build_personality(self.client, self.config, persona_name=name)
            if self.client is not None
            else None
        )

    def _persist_persona(self, name: str) -> None:
        personality_cfg = self.config.config.setdefault("personality", {})
        personality_cfg["persona_name"] = name
        try:
            self.config.save()
        except Exception as e:
            ASCIIColors.warning(f"Could not persist the persona selection: {e}")

    async def start(self):
        ASCIIColors.info("🚀 Starting Lollms-Bot Daemon...")

        if self.client is None:
            ASCIIColors.red("No LLM client available. The bot cannot respond.")
            return

        llm = self.client.llm
        ASCIIColors.info(
            f"Using LLM → binding: {getattr(llm, 'binding_name', None)}, "
            f"model: {getattr(llm, 'model_name', None)}"
        )

        if self.personality is None:
            ASCIIColors.red("Failed to load the bot personality from the handbag.")
            ASCIIColors.info("Re-run the wizard (lollms-bot --config) to recreate it.")
            return

        if self.persona_registry:
            ASCIIColors.info(
                f"Persona registry: {len(self.persona_registry)} persona(s) available "
                f"({', '.join(sorted(self.persona_registry))})."
            )

        bindings = _active_connection_bindings(self.client)
        if not bindings:
            ASCIIColors.yellow(
                "No connection bindings configured. Add one in the wizard:\n"
                "   python -m lollms_client.lollms_config_cli_env   (CONNECTION section)"
            )
            return

        _attach_personality(
            bindings,
            self.personality,
            personality_factory=self.personality_factory,
            persona_registry=self.persona_registry,
            on_persona_selected=self._persist_persona,
        )
        await _start_connections(bindings)