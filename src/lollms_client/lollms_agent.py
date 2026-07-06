# lollms_agent.py
# Agent: a named, role-bearing wrapper around a LollmsClient + LollmsPersonality.
# Used as the unit of participation in a SwarmOrchestrator run.

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from lollms_client.lollms_core import LollmsClient
    from lollms_client.lollms_personality import LollmsPersonality


# ---------------------------------------------------------------------------
# AgentRole — semantic label for an agent's function inside the swarm
# ---------------------------------------------------------------------------

class AgentRole:
    """
    Well-known role labels.  Custom strings are also accepted; these are
    just constants to avoid typos and to let the orchestrator apply
    role-specific prompt rules automatically.
    """
    PROPOSER        = "proposer"        # introduces initial ideas / drafts
    CRITIC          = "critic"          # challenges, finds flaws, stress-tests
    DEVIL_ADVOCATE  = "devil_advocate"  # argues the unpopular / opposite position
    DOMAIN_EXPERT   = "domain_expert"   # deep specialist contribution
    SYNTHESIZER     = "synthesizer"     # integrates all viewpoints into a conclusion
    MODERATOR       = "moderator"       # drives the session, asks clarifying questions
    IMPLEMENTER     = "implementer"     # turns plans into code / artefacts
    TESTER          = "tester"          # reviews artefacts for correctness / edge cases
    NARRATOR        = "narrator"        # describes events in simulation / game modes
    PLAYER          = "player"          # participates in a game or simulation as a character
    FREEFORM        = "freeform"        # no prescribed role; acts on its own judgment


# ---------------------------------------------------------------------------
# ToolsManager — load and execute lollms-format tool scripts
# ---------------------------------------------------------------------------

class ToolsManager:
    """
    Discovers, parses, and executes lollms-format tool libraries.

    A lollms-format tool script is a Python file containing:
      • Global metadata: TOOL_LIBRARY_NAME, TOOL_LIBRARY_DESC, TOOL_LIBRARY_ICON
      • An optional ``init_tools_library()`` function for dependency setup
      • One or more ``tool_*`` functions with docstring-described arguments

    The manager extracts JSON-schema-style parameter definitions from
    docstrings and can execute tools in isolated namespace contexts.
    """

    # Default search paths
    SYSTEM_TOOLS_DIR = Path("app/tools")
    USER_TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"

    def __init__(self, extra_dirs: Optional[List[Union[str, Path]]] = None):
        self._extra_dirs: List[Path] = [Path(d) for d in (extra_dirs or [])]
        self._loaded_modules: Dict[str, ModuleType] = {}
        self._tool_cache: Dict[str, Dict[str, Any]] = {}

    # ── Discovery ─────────────────────────────────────────────────────────

    @classmethod
    def ensure_dirs(cls):
        """Create default tool directories if missing."""
        cls.SYSTEM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        cls.USER_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    def _scan_paths(self) -> List[Path]:
        """Return all directories to scan for ``*.py`` tools."""
        dirs = [self.SYSTEM_TOOLS_DIR, self.USER_TOOLS_DIR] + self._extra_dirs
        return [d for d in dirs if d.exists()]

    def list_available_files(self) -> List[Path]:
        """Return sorted list of all ``*.py`` tool file paths."""
        files: set = set()
        for directory in self._scan_paths():
            for fp in directory.glob("*.py"):
                if fp.name == "__init__.py":
                    continue
                files.add(fp.resolve())
        return sorted(files, key=lambda p: p.name.lower())

    # ── Parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_metadata(content: str) -> Dict[str, str]:
        """Extract global metadata variables from Python source via AST."""
        meta = {
            "name": "Unnamed Tool Library",
            "description": "No description provided.",
            "icon": "🔧",
        }
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "TOOL_LIBRARY_NAME":
                                meta["name"] = ast.literal_eval(node.value)
                            elif target.id == "TOOL_LIBRARY_DESC":
                                meta["description"] = ast.literal_eval(node.value)
                            elif target.id == "TOOL_LIBRARY_ICON":
                                meta["icon"] = ast.literal_eval(node.value)
        except Exception:
            pass
        return meta

    @staticmethod
    def get_tool_definitions(content: str) -> List[Dict[str, Any]]:
        """
        Parse ``tool_*`` functions and their docstrings into OpenAI-style
        function definitions.
        """
        tools: List[Dict[str, Any]] = []
        titles: Dict[str, str] = {}
        try:
            tree = ast.parse(content)

            # First pass: collect TOOL_TITLES if present
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "TOOL_TITLES":
                            titles = ast.literal_eval(node.value)

            # Second pass: process tool_* functions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_"):
                    docstring = ast.get_docstring(node) or "No description provided."

                    params: Dict[str, Any] = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }

                    # Parse docstring for Args section
                    # Matches: - name (type): description
                    #          - name (type, optional): description
                    arg_pattern = re.compile(
                        r'^\s*-\s+([\w_]+)\s*\(([\w_]+)(?:,\s*optional)?\):\s*(.*)',
                        re.MULTILINE | re.IGNORECASE,
                    )
                    for m in arg_pattern.finditer(docstring):
                        name, p_type, desc = m.groups()
                        p_type_map = {
                            "str": "string",
                            "int": "integer",
                            "float": "number",
                            "bool": "boolean",
                            "dict": "object",
                            "list": "array",
                        }
                        params["properties"][name] = {
                            "type": p_type_map.get(p_type.lower(), "string"),
                            "description": desc.strip(),
                        }
                        if "optional" not in m.group(0).lower():
                            params["required"].append(name)

                    # Fallback: if no args parsed but function takes 'args', use generic
                    if not params["properties"]:
                        # Check if function signature has 'args' parameter
                        has_args = any(
                            (isinstance(arg, ast.arg) and arg.arg == "args")
                            for arg in node.args.args
                        )
                        if has_args:
                            params["properties"]["args"] = {
                                "type": "object",
                                "description": "Arguments for the tool",
                            }

                    tools.append({
                        "type": "function",
                        "pretty_name": titles.get(node.name),
                        "function": {
                            "name": node.name,
                            "description": docstring.split('\n\n')[0].strip(),
                            "parameters": params,
                        },
                    })
        except Exception:
            pass
        return tools

    # ── Loading & Execution ───────────────────────────────────────────

    def load_file(self, file_path: Union[str, Path]) -> ModuleType:
        """Load a tool script into an isolated module namespace."""
        fp = Path(file_path).resolve()
        key = str(fp)

        if key in self._loaded_modules:
            return self._loaded_modules[key]

        content = fp.read_text(encoding="utf-8")
        module_name = f"lollms_tools_{fp.stem}_{uuid.uuid4().hex[:8]}"

        # Create module and execute in its namespace
        module = ModuleType(module_name)
        module.__file__ = str(fp)

        # Execute the module content
        exec(compile(content, str(fp), "exec"), module.__dict__)

        # Run init if present
        if hasattr(module, "init_tools_library"):
            try:
                module.init_tools_library()
            except Exception as e:
                from ascii_colors import ASCIIColors
                ASCIIColors.warning(f"Tool init failed for {fp.name}: {e}")

        self._loaded_modules[key] = module
        return module

    def get_callable_tools(self, file_path: Union[str, Path]) -> Dict[str, Callable]:
        """
        Return a dict of ``{tool_name: callable}`` for all ``tool_*``
        functions found in *file_path*.
        """
        module = self.load_file(file_path)
        return {
            name: getattr(module, name)
            for name in dir(module)
            if name.startswith("tool_") and callable(getattr(module, name))
        }

    def execute_tool(
        self,
        file_path: Union[str, Path],
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Execute a single tool from *file_path* with *args*."""
        callables = self.get_callable_tools(file_path)
        if tool_name not in callables:
            raise ValueError(f"Tool '{tool_name}' not found in {file_path}")
        return callables[tool_name](args)

    def resolve_tool_file(self, tool_name: str) -> Optional[Path]:
        """
        Find which file contains *tool_name* by scanning all tool directories.
        """
        for fp in self.list_available_files():
            defs = self.get_tool_definitions(fp.read_text(encoding="utf-8"))
            for d in defs:
                if d["function"]["name"] == tool_name:
                    return fp
        return None

    # ── Unified tool spec builder ───────────────────────────────────────

    def build_tool_specs(self, sources: List[Union[str, Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert mixed *sources* (file paths or inline dicts) into a unified
        list of OpenAI-style function definitions.

        File paths are parsed for metadata + tool definitions.
        Inline dicts are passed through unchanged.
        """
        specs: List[Dict[str, Any]] = []
        for src in sources:
            if isinstance(src, dict):
                # Inline tool spec — pass through
                specs.append(src)
                continue

            # File path
            fp = Path(src)
            if not fp.exists():
                raise FileNotFoundError(f"Tool file not found: {fp}")
            content = fp.read_text(encoding="utf-8")
            file_specs = self.get_tool_definitions(content)
            # Enrich with source file info for later execution
            for s in file_specs:
                s["_source_file"] = str(fp.resolve())
            specs.extend(file_specs)
        return specs

    def build_inline_tools_dict(
        self,
        sources: List[Union[str, Path, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build the ``{tool_name: {"callable": fn, "parameters": [...], ...}}``
        dict expected by ``LollmsDiscussion.chat(tools=...)``.
        """
        tools_dict: Dict[str, Dict[str, Any]] = {}
        for src in sources:
            if isinstance(src, dict):
                # Already an inline tool dict with "callable"
                name = src.get("name", src.get("function", {}).get("name", "unknown"))
                tools_dict[name] = src
                continue

            # File path
            fp = Path(src)
            if not fp.exists():
                raise FileNotFoundError(f"Tool file not found: {fp}")

            module = self.load_file(fp)
            callables = self.get_callable_tools(fp)

            for tool_name, fn in callables.items():
                # Parse docstring for parameter info
                doc = (fn.__doc__ or "").strip()
                params: List[Dict[str, Any]] = []

                arg_pattern = re.compile(
                    r'^\s*-\s+([\w_]+)\s*\(([\w_]+)(?:,\s*optional)?\):\s*(.*)',
                    re.MULTILINE | re.IGNORECASE,
                )
                for m in arg_pattern.finditer(doc):
                    pname, ptype, pdesc = m.groups()
                    is_optional = "optional" in m.group(0).lower()
                    p_entry: Dict[str, Any] = {
                        "name": pname,
                        "type": ptype.lower(),
                        "description": pdesc.strip(),
                    }
                    if is_optional:
                        p_entry["optional"] = True
                    params.append(p_entry)

                tools_dict[tool_name] = {
                    "name": tool_name,
                    "callable": fn,
                    "parameters": params,
                    "description": doc.split('\n\n')[0].strip() if doc else f"Execute {tool_name}",
                    "_source_file": str(fp.resolve()),
                }

        return tools_dict


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """
    A single participant in a swarm run.

    Parameters
    ----------
    lc : LollmsClient
        The LLM binding this agent uses.  Multiple agents may share the same
        binding or use completely different ones (e.g. one local, one cloud).
    personality : LollmsPersonality
        The personality that defines the agent's system prompt, knowledge base,
        and any personality-level tools.  The personality's ``name`` attribute
        is used as the agent's display name in the discussion unless ``name``
        is explicitly provided.
    name : str | None
        Override the display name.  Defaults to ``personality.name``.
    role : str
        Semantic role label (see ``AgentRole``).  Influences the anti-sycophancy
        prompt injected by the orchestrator.
    tools : dict | None
        External tool specs in the same format accepted by ``chat(tools=...)``.
    model_params : dict
        Extra kwargs forwarded to every LLM call this agent makes
        (e.g. ``{"temperature": 0.8, "n_predict": 2048}``).
    max_tokens_per_turn : int
        Soft cap on how many tokens this agent generates per round.
        Passed as ``n_predict`` if the binding honours it.
    metadata : dict
        Arbitrary application-layer metadata (avatar URL, colour, etc.).
    """
    lc:                  Any                   # LollmsClient
    personality:         Any                   # LollmsPersonality
    name:                Optional[str]         = None
    role:                str                   = AgentRole.FREEFORM
    tools:               Optional[Dict]        = None
    model_params:        Dict[str, Any]        = field(default_factory=dict)
    max_tokens_per_turn: int                   = 1024
    metadata:            Dict[str, Any]        = field(default_factory=dict)

    # set by the orchestrator at the start of a swarm run
    _agent_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)

    # ---------------------------------------------------------------- derived

    @property
    def display_name(self) -> str:
        return self.name or getattr(self.personality, "name", "Agent")

    @property
    def system_prompt(self) -> str:
        return getattr(self.personality, "system_prompt", "") or ""

    def has_knowledge(self) -> bool:
        return getattr(self.personality, "has_data", False)

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return (f"<Agent name={self.display_name!r} role={self.role!r} "
                f"id={self._agent_id[:8]}>")

    # ---------------------------------------------------------------- helpers

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 1024,
        streaming_callback: Optional[Callable] = None,
        **extra,
    ) -> str:
        """
        Direct (non-discussion) generation from this agent's binding.
        Used by the orchestrator for HLF self-assessment calls and for
        agents whose turn does not produce a persistent discussion message.
        """
        kwargs: Dict[str, Any] = {
            **self.model_params,
            "temperature":        temperature,
            "n_predict":          min(n_predict, self.max_tokens_per_turn),
            **extra,
        }
        if streaming_callback:
            kwargs["streaming_callback"] = streaming_callback

        full_prompt = ""
        if system_prompt:
            full_prompt = f"!@>system:\n{system_prompt}\n!@>user:\n{prompt}\n!@>assistant:\n"
        else:
            full_prompt = prompt

        try:
            return self.lc.generate_text(full_prompt, **kwargs) or ""
        except Exception as e:
            from ascii_colors import trace_exception
            trace_exception(e)
            return f"[{self.display_name} generation error: {e}]"

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, str],
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Optional[Dict]:
        """Structured JSON generation for HLF messages."""
        try:
            return self.lc.generate_structured_content(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt or self.system_prompt,
                temperature=temperature,
            )
        except Exception as e:
            from ascii_colors import trace_exception
            trace_exception(e)
            return None

    # ---------------------------------------------------------------- tool-enabled generation

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 4096,
        max_tool_rounds: int = 10,
        streaming_callback: Optional[Callable] = None,
        auto_execute: bool = True,
        **extra,
    ) -> Dict[str, Any]:
        """
        Generate a response with access to tools.

        Parameters
        ----------
        prompt : str
            The user prompt / task description.
        tools : list
            Mixed list of:
              • ``str`` or ``Path`` — file path to a lollms-format tool script
              • ``dict`` — inline tool spec with ``{"name": ..., "callable": ..., ...}``
        system_prompt : str
            Optional system prompt override.
        temperature : float
            Sampling temperature.
        n_predict : int
            Max tokens per generation.
        max_tool_rounds : int
            Maximum agentic tool-call loops before forcing final answer.
        streaming_callback : callable
            Optional streaming callback ``(chunk, msg_type, meta) -> bool``.
        auto_execute : bool
            If True, automatically execute tool calls and feed results back
            to the model. If False, return the tool call request for manual
            execution.

        Returns
        -------
        dict
            {
                "response": str,           # Final text response
                "tool_calls": list,        # All tool calls made
                "tool_results": list,      # All tool execution results
                "rounds": int,             # Number of agentic rounds
            }
        """
        from ascii_colors import ASCIIColors

        # ── 1. Build unified tool registry ──────────────────────────────
        tools_mgr = ToolsManager()
        inline_tools = tools_mgr.build_inline_tools_dict(tools)

        if not inline_tools:
            # No valid tools — fall back to plain generation
            return {
                "response": self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    n_predict=n_predict,
                    streaming_callback=streaming_callback,
                    **extra,
                ),
                "tool_calls": [],
                "tool_results": [],
                "rounds": 0,
            }

        # ── 2. Build tool descriptions for the system prompt ──────────────
        tool_descriptions: List[str] = []
        for name, spec in inline_tools.items():
            params = spec.get("parameters", [])
            param_str = ", ".join(
                f"{p['name']}: {p['type']}" + (" (optional)" if p.get("optional") else "")
                for p in params
            )
            desc = spec.get("description", f"Execute {name}")
            tool_descriptions.append(f"- {name}({param_str}): {desc}")

        tool_header = (
            "=== TOOL USE — MANDATORY FORMAT ===\n"
            "You have external tools. To use one you MUST use EXACTLY this format:\n"
            "<tool>{\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}</tool>\n\n"
            "CRITICAL RULES:\n"
            "1. The ENTIRE tool call must be wrapped in <tool> tags.\n"
            "2. NO markdown code fences (no ```json).\n"
            "3. NO raw JSON without the XML wrapper.\n"
            "4. NO explanations before or after the tool call.\n"
            "5. ONLY output the <tool> line when calling a tool.\n"
            "6. One tool call per response turn.\n"
            "7. After calling ALL needed tools, write your final answer.\n"
            "8. If the user explicitly asks you to use a tool, USE IT.\n"
            "=== END TOOL USE RULES ===\n\n"
            "TOOLS AVAILABLE:\n"
        )

        tool_block = tool_header + "\n".join(tool_descriptions)

        # ── 3. Prepare conversation state ─────────────────────────────────
        full_system = (system_prompt or self.system_prompt).rstrip()
        if full_system:
            full_system += "\n\n"
        full_system += tool_block

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        rounds = 0

        # ── 4. Agentic loop ───────────────────────────────────────────────
        while rounds < max_tool_rounds:
            rounds += 1

            # Generate response
            kwargs: Dict[str, Any] = {
                **self.model_params,
                "temperature": temperature,
                "n_predict": min(n_predict, self.max_tokens_per_turn),
                **extra,
            }
            if streaming_callback:
                kwargs["streaming_callback"] = streaming_callback

            try:
                raw_response = self.lc.generate_from_messages(
                    messages=conversation,
                    **kwargs,
                )
            except Exception as e:
                ASCIIColors.error(f"generate_with_tools: generation failed: {e}")
                return {
                    "response": f"[Error during generation: {e}]",
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "rounds": rounds,
                }

            if not isinstance(raw_response, str):
                raw_response = str(raw_response) if raw_response is not None else ""

            # ── 5. Parse tool calls ─────────────────────────────────────────
            # Primary: XML-wrapped tool calls <tool>...</tool>
            tool_call_pattern = re.compile(
                r'<tool>(.*?)</tool>',
                re.DOTALL | re.IGNORECASE,
            )
            matches = list(tool_call_pattern.finditer(raw_response))

            # Fallback: detect raw JSON tool calls (models sometimes omit XML tags)
            tool_json_str = None
            visible_response = raw_response.strip()

            if matches:
                # Extract the first tool call (one per turn)
                match = matches[0]
                tool_json_str = match.group(1).strip()
                visible_response = raw_response[:match.start()].strip()
            else:
                # Try to detect raw JSON that looks like a tool call
                # Pattern: {"name": "tool_...", "parameters": {...}}
                raw_json_pattern = re.compile(
                    r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*\{',
                    re.MULTILINE,
                )
                # Also try to find a complete JSON object with name+parameters
                json_obj_pattern = re.compile(
                    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}',
                    re.DOTALL,
                )

                json_match = json_obj_pattern.search(raw_response)
                if json_match:
                    tool_json_str = json_match.group(0).strip()
                    # Determine visible response (text before the JSON object)
                    json_start = json_match.start()
                    visible_response = raw_response[:json_start].strip()
                    ASCIIColors.warning(
                        f"Model emitted raw JSON tool call (missing <tool> tags). "
                        f"Tool: {json_match.group(1)}"
                    )

            if not tool_json_str:
                # No tool call — this is the final answer
                cleaned = tool_call_pattern.sub('', raw_response).strip()
                return {
                    "response": cleaned,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "rounds": rounds,
                }

            # ALWAYS add assistant message to maintain strict user/assistant
            # alternation required by llama.cpp Jinja chat templates.
            # Even if visible_response is empty, the assistant "spoke" (the tool call).
            conversation.append({"role": "assistant", "content": visible_response})

            # Parse tool call JSON
            try:
                call_data = json.loads(tool_json_str)
            except json.JSONDecodeError as e:
                ASCIIColors.warning(f"Failed to parse tool call JSON: {e}")
                conversation.append({
                    "role": "user",
                    "content": f"Error: Invalid tool call JSON. {e}",
                })
                continue

            tool_name = call_data.get("name", "")
            tool_params = call_data.get("parameters", {})

            call_record = {
                "round": rounds,
                "name": tool_name,
                "parameters": tool_params,
                "raw": tool_json_str,
            }
            all_tool_calls.append(call_record)

            if not auto_execute:
                # Manual mode: return the tool call for external handling
                return {
                    "response": visible_response,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "pending_tool": call_record,
                    "rounds": rounds,
                }

            # ── 6. Execute tool ─────────────────────────────────────────────
            if tool_name not in inline_tools:
                error_msg = f"Error: Tool '{tool_name}' not found in registry."
                ASCIIColors.warning(error_msg)
                result = {"error": error_msg, "success": False}
            else:
                tool_spec = inline_tools[tool_name]
                fn = tool_spec.get("callable")
                if not callable(fn):
                    error_msg = f"Error: Tool '{tool_name}' has no callable."
                    ASCIIColors.warning(error_msg)
                    result = {"error": error_msg, "success": False}
                else:
                    try:
                        # Normalize parameters: lollms-format tools use `args: dict`
                        # but some inline tools may use kwargs. Try kwargs first,
                        # fall back to single dict arg if signature mismatch.
                        try:
                            result = fn(**tool_params)
                        except TypeError as te:
                            if "unexpected keyword argument" in str(te):
                                result = fn(tool_params)
                            else:
                                raise

                        # Normalize result to dict if it's a plain string
                        if isinstance(result, str):
                            result = {"output": result, "success": True}
                        elif not isinstance(result, dict):
                            result = {"output": str(result), "success": True}

                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {e}"
                        ASCIIColors.warning(error_msg)
                        result = {"error": error_msg, "success": False}

            result_record = {
                "round": rounds,
                "name": tool_name,
                "result": result,
            }
            all_tool_results.append(result_record)

            # Format result for LLM context
            if isinstance(result, dict) and result.get("success"):
                result_text = result.get("output", json.dumps(result, indent=2))
            else:
                result_text = json.dumps(result, indent=2, ensure_ascii=False)

            # Truncate very large results
            max_result_len = 4000
            if len(result_text) > max_result_len:
                result_text = result_text[:max_result_len] + f"\n... [{len(result_text) - max_result_len} chars truncated]"

            # Add tool result to conversation
            conversation.append({
                "role": "user",
                "content": (
                    f'<tool_result name="{tool_name}">\n'
                    f"{result_text}\n"
                    f"</tool_result>"
                ),
            })

        # ── 7. Max rounds exceeded — force final answer ───────────────────
        ASCIIColors.warning(f"generate_with_tools: max rounds ({max_tool_rounds}) exceeded")
        conversation.append({
            "role": "user",
            "content": (
                "[SYSTEM] Maximum tool rounds reached. "
                "Provide your final answer now without calling any more tools."
            ),
        })

        try:
            final_response = self.lc.generate_from_messages(
                messages=conversation,
                temperature=temperature,
                n_predict=min(n_predict, self.max_tokens_per_turn),
                **{k: v for k, v in extra.items() if k not in ("temperature", "n_predict")},
            )
        except Exception as e:
            final_response = f"[Error generating final answer: {e}]"

        cleaned = tool_call_pattern.sub('', str(final_response)).strip()
        return {
            "response": cleaned,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "rounds": rounds,
        }

    def generate_with_tools_sync(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        **kwargs,
    ) -> str:
        """
        Synchronous convenience wrapper that returns only the final text
        response (drops metadata). Useful for simple fire-and-forget usage.
        """
        result = self.generate_with_tools(prompt, tools, **kwargs)
        return result.get("response", "")
