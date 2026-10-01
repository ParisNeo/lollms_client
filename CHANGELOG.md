```markdown
# CHANGELOG

All notable changes to this project will be documented in this file.
#
- fix(gui:realtime_tool_streaming): preserve `tool_progress` metadata in `QueueStreamingCallback`, expand active tool panels with live spinner, and stream execution lines into the GUI code box in real-time
- fix(execute_python:realtime_streaming): eliminate stdout redirection recursion loop in `_ProgressStringIO`, stream unbuffered tool lines directly to the OS terminal in real-time during `time.sleep()`, and sanitize Rich markup
- feat(tools:live_streaming): stream stdout/stderr lines in real-time during tool execution (`execute_python` and `system_shell`) to CLI terminal and GUI code boxes via `ToolContext.emit_progress`
- fix(execute_python:workspace_resolution): remove accidental directory diversion to `data_workspace` in `_get_workspace_root()`, ensure `_resolve_workspace_path()` checks both root and CWD, and fix `tool_context` forwarding across `VAR_KEYWORD` callable signatures
- fix(lollms_code:client_lifecycle): recreate LollmsClient and personality instances immediately after configuration changes and saves across GUI and CLI, preserving conversation history
- fix(personality:done_termination): ensure `<done/>` with completed artifacts terminates immediately in round 1 when no tool calls are pending
- fix(agent_state:code_fence): restore code fence protection in `_AgentStreamState` by removing `has_action_tag_in_pending` so tags inside markdown code blocks are not intercepted as live tools
- fix(core:vision_capability): harden `has_vision_capability()` with strict type and mock checks on `models_dir`, `model_name`, and `_find_mmproj` to prevent mocks from falsely reporting vision capability
- fix(gui:workflow_studio): separate SVG markup and client-side JavaScript execution in Workflow Studio 2D graph canvas to eliminate NiceGUI `ValueError: HTML elements must not contain <script> tags`
- fix(workflow:types): remove duplicate `compute_auto_layout()` method definition in `workflow_types.py`
- fix(gui:linear_history): ensure `replay_transcript_from_log()` clears all transcript DOM elements, panel references, and active buffers so `resend_from_point` and `open_edit_dialog` purge all subsequent messages on screen
- fix(workspace_tools): remove binary file exclusion from `tool_list_files` so images, audio, video, documents, and archives are visible to agents during file organization tasks
- fix(workspace_tools): improve `_resolve_safe_path` to handle root and absolute workspace path references without nested path corruption
- fix(gui:workflow_studio): pre-seed `ui.select` options on initialization to prevent NiceGUI `ValueError: Invalid value` crashes in Workflow Studio
- feat(gui:workflow_studio): implement interactive 2D graph canvas with draggable elements, ports, curved bezier links, auto-layout DAG alignment, and visual loopback indicators
- feat(gui:workflow_studio): implement interactive 2D graph canvas with draggable elements, ports, curved bezier links, auto-layout DAG alignment, and visual loopback indicators
- feat(gui:workflow_studio): overhaul Workflow Studio into a full project IDE with visual graph canvas, per-step node configuration dialogs, subtask model routing, interactive gate decision resumption, and project `.lollms_code/workflows/*.yaml` persistence
- fix(workflow:imports): add explicit `__all__` exports across `workflow_types.py` and `workflow_engine.py`, synchronize `lollms_workflow/__init__.py`, and provide robust submodule fallback in `open_workflow_studio_dialog()`
- fix(workflow:bootstrap): eliminate package circular import on `Workflow` and `NodeType` during `lollms-code` launch by converting workflow exports in `src/lollms_client/__init__.py` to PEP 562 lazy resolution
- fix(agent:tool_parser): add AST literal evaluation and inline argument normalization to parse single-quoted Python dicts and function calls (`tool_name(args=...)`) into valid tools
- fix(personality:loop_continuation): allow the reasoning loop to continue to the next round after tool execution rather than breaking prematurely
- fix(workflow:import): resolve partially initialized package `ImportError` on `Workflow` by converting `lollms_workflow` to relative imports
- feat(workflow:engine): introduce deterministic graph-based execution harness (`lollms_workflow`) supporting hard constraint walls, conditional if-this-then-that branching, and variable template piping
- feat(subtask:model_selection): allow selecting a different model or profile alias for subtasks in `SubAgentSpawner.spawn()`, `tool_spawn_sub_agent`, and `tool_spinoff_agent`, automatically restoring the parent model upon turn completion
- feat(gui:workflow_studio): add interactive NiceGUI Workflow Studio dialog with visual graph inspection, per-step model routing, and hard constraint guard verification
- refactor(agent:done_handling): remove fragile regex-based phantom `<done/>` interception to prevent false-positive summary loops, restoring sovereign `<done/>` termination
- feat(prompt:slm_grounding): integrate few-shot tool-calling exemplars and positive action-first syntax steering into system prompt based on SLM hallucination research
- refactor(skills:file_organization): streamline procedural headers in `SKILL.md` to prevent 8B models from parroting prompt headers into conversational output
- docs(enhancements): add research analysis on SLM tool-bypass hallucinations in `incremental_enhancements/`
- feat(agent:approval_hydration): automatically hydrate user approval replies ("yes", "proceed") with explicit directives to execute pending `mapping.yaml` plans via `tool_organize_files_from_plan`
- fix(gui:resend): resolve `NameError: name 'effective_prompt' is not defined` in `resend_from_point()`
- fix(llama_cpp:vision): dynamically detect vision projector binding and set `vision_enabled = True` on `LlamaCppServerBinding` for VL models
- fix(personality:prompt): resolve `NameError: name 'rules' is not defined` in `_build_system_prompt()` by integrating `operating_protocol` into safe prompt component joining
- fix(llama_cpp:bind_multimodal): prevent NoneType path crashes in `bind_multimodal_model` and implement `list_mmproj_models()` so vision projectors appear in UI dropdowns
- fix(gui:settings): resolve half-light/half-dark contrast issue in settings page by eliminating Quasar `bg-white` class collision, binding `:dark` props on all cards, and applying strict CSS dark overrides
- feat(prompt:optimization): streamline system prompt, eliminate multi-block instruction redundancy, and implement progressive tool disclosure cutting baseline prompt tokens by ~60%
- feat(tools:lean_defaults): restrict default toolset to essential workspace and execution primitives, keeping Git and rich document tools on-demand via `tool_load_tool`
- docs(enhancements): add comprehensive documentation of prompt optimization and lean tooling in `incremental_enhancements/`
- fix(gui:inspector): resolve empty context and 0 token count in Context Inspector by rendering system prompt and prompt placeholder when conversation is empty
- fix(history:export): allow `HistoryManager.export()` to format and return system prompt even when branch message list is empty
- fix(prompt:dedup): eliminate duplicate system prompt rendering in `agent_bridge.get_context_preview()` and deduplicate core mandates in `LollmsPersonality._build_system_prompt()`
- fix(agent:stream): isolate post-stream recovery sweep to content outside markdown fences and ensure artifact body is removed from `self.content`
- fix(core:profiles): register `"master"` model profile when `extra_llms` are supplied alongside legacy master binding
- fix(llm:reasoning): allow `translate_reasoning_effort` to project disabled literals (e.g. `"none"`) to matching supported tiers (e.g. `"off"`)
- fix(core:profiles): fix `UnboundLocalError` on `new_binding` in `_switch_modality` by maintaining reference to `current_binding` across cache hits and misses
- fix(llm:reasoning): fix `translate_reasoning_effort` and `get_effective_reasoning_effort` to project effort when `think` is None or True
- fix(tools:vlm_query): fix VLM resolution on mocks and align error messages with test assertions
- fix(memory:working): ensure `build_working_zone` contains standard `=== WORKING MEMORY ===` marker
- fix(tools:opt_in): honor `enable_workspace_tools=False` in `ChatMixin._resolve_active_tools` to uphold Sovereign Tool Opt-In doctrine
- fix(agent:stream): prevent artifact body leakage into `self.content` in `_AgentStreamState` and protect inline code tags during text scrubbing
- fix(personality:discovery): ensure multimodal bindings, sub-agents, and model switcher tools mount properly in `_discover_tools`
- perf(test:diffusers): optimize `test/test_diffusers_install_model.py` by setting `auto_start_server=False` on TTV, eliminating multi-gigabyte pip dependency downloads and infinite thread polling
- fix(gui:disconnect): prevent NiceGUI event-loop disconnects on workspace load by replacing retrying HTTP status checks with fast 0.5s probes and excluding TTM/TTV media daemons from `lollms_code` client startup
- fix(bindings:paths): anchor `venv_dir` and `cache_dir` across `diffusers` TTM, `diffusers` TTV, and `whisper` STT to `~/.lollms_client/venv/` and `~/.lollms_client/data/`
- fix(discussion:chat): repair missing `if new_symbols:` indentation block in `_mixin_chat.py` resolving collection-time `IndentationError`
- feat(computer_use): implement conditional desktop automation toolset mounting gated strictly on `allow_computer_use=True` and active vision model verification
- feat(computer_use): expand primitive toolset with `tool_computer_mouse_down`, `tool_computer_mouse_up`, `tool_computer_drag`, `tool_computer_wait`, and `tool_computer_cursor_position`
- feat(computer_use:gating): integrate `allow_computer_use` across `CapabilityFlags`, `LollmsPersonality`, `LollmsDiscussion`, `lollms_code` CLI (`--allow-computer-use`), and GUI settings
- test(computer_use): add comprehensive unit test suite in `test/test_computer_use_tools.py` verifying vision gating, rejection without vision, and safe execution with mocked backend
- docs(dynamic_mode): document Dynamic Mode operation across README.md and DOC_DEV.md, covering autonomous reasoning effort escalation via `<effort level="..."/>`, task-adapted temperature regulation, context-fitting token budgets, sub-agent delegation, and CLI/GUI controls
- fix(artefact:patching): resolve Aider patch conflict marker leakage into source code by stopping replace block absorption on subsequent search headers, scrubbing stray sentinel lines from replacements, and decontaminating final result strings
- fix(artefact:guard): block raw search/replace conflict blocks from being written directly to disk as full files when targets are missing, preventing disk corruption with `=======`
- fix(artefact:streaming): dynamically detect `<<<<<<< SEARCH` during chunk streaming to switch UI state from full rewrite to patch, and stream whole generated content into the processing block and UI code box
- feat(openai:vllm): add vLLM server support to OpenAI binding, injecting `chat_template_kwargs` (`enable_thinking`, `thinking`, `reasoning_effort`) via `extra_body`, forwarding `top_k`, `repetition_penalty`, and `min_tokens`, preserving `temperature` and `top_p` sampling controls on vLLM, and adding auto-detection via `/version` probe
- fix(bindings:reasoning): enforce thought suppression rule across `lollms`, `openai`, and `ollama` bindings: if `think is True`, inspect `reasoning_effort` and send to server, or deactivate thought completely if effort is none/None; for `ollama`, automatically deactivate thinking whenever `think` is not True independently of the effort parameter
- fix(lollms:reasoning): fix `reasoning_effort` parameter defaults in `LollmsBinding` from `"low"` to `None`, implement `_apply_thinking_params`, add `suppress_thinking` support to `_StreamThinkingHandler`, and strip reasoning tags on thought deactivation
- fix(ollama:reasoning): project `'max'` and extreme reasoning efforts to `'high'` to conform to Ollama's `ChatRequest` schema (`Union[bool, Literal['low', 'medium', 'high']]`), resolving Pydantic validation crashes
- fix(discussion:chat): initialize `was_cancelled` before `_persist_round_state()` definition in `_mixin_chat.py` to prevent enclosing scope `UnboundLocalError` warnings
- feat(ollama:reasoning): upgrade requirement to `ollama>=0.6.2`, pass `think=False` & `options["think"]=False` when disabled to instruct engine not to think, and pass exact effort level strings (`low`, `medium`, `high`, `max`) when enabled
- fix(client:profiles): resolve duplicate default profiles across all modalities on startup by keeping the first default, demoting extras, and persisting the single-default invariant to `config.yaml` and `.env`; eliminate phantom `'master'` profile injection with missing `model_name` when real model profiles exist
- fix(openai:reasoning): patch `OpenAIBinding` (`generate_text`, `generate_from_messages`, `_StreamThinkingHandler`) to apply `_apply_thinking_params`, instruct engines not to think (`reasoning_effort="none"`, `chat_template_kwargs={"enable_thinking": False, "thinking": False}`, `thinking={"type": "disabled"}` for GLM), and silence residual thoughts when deactivated
- feat(openai:reasoning): explicitly instruct engines not to think (`reasoning_effort="none"`, `chat_template_kwargs={"enable_thinking": False}`, `thinking={"type": "disabled"}` for GLM) when `think is False` and `reasoning_effort is None`
- fix(ollama:reasoning): strictly deactivate thinking in Ollama binding (`think=False`, `options["think"]=False`) when `think is False` and `reasoning_effort is None`, suppress thinking stream chunks when disabled, and filter out raw `<think>` content
- fix(discussion:db): resolve SQLite `database or disk is full` error by setting `PRAGMA temp_store=MEMORY` on all pooled connections, auto-checkpointing WAL, adding safe rollback to `commit()`, and stripping bulky raw content from `sources` metadata
- fix(discussion:chat): add explicit `web_search` and `internet_search` parameters to `ChatMixin.chat()` signature and guard pre-hydration evaluation to eliminate `NameError: name 'web_search' is not defined`
- fix(discussion:sources): persist RAG and web search sources in `ai_message.metadata["sources"]` across round checkpoints and turn completions, number prompt source headers as `[1]`, `[2]`, `[3]`, and expose `LollmsMessage.sources` to guarantee references remain active after page reloads
- fix(llm:reasoning): ensure GLM/vLLM backend thinking deactivation (`chat_template_kwargs={"enable_thinking": False}`, `thinking={"type": "disabled"}`) and fix multi-chunk `_in_think_block` state tracking in `_mixin_chat.py` so thoughts never leak into conversational chat bubbles
- fix(llm:reasoning): ensure all LLM bindings and core client generation pipelines explicitly deactivate thinking (`thinking=False`, `chat_template_kwargs={"thinking": False}`) when `think is False` and `reasoning_effort is None`
- fix(rag:import): ensure `LollmsRAGBinding` and `LollmsRAGBindingManager` are completely defined and exported in `lollms_rag_binding.py`, with graceful fallback in `lollms_core.py` and top-level export in `__init__.py`
- feat(client:profile_management): add first-class runtime profile management methods (`list_binding_profiles`, `list_model_profiles`, `add_binding_profile`, `add_model_profile`, `remove_binding_profile`, `remove_model_profile`, `get_active_profile`, `switch_profile`) to `LollmsClient`
- feat(client:from_config): add `LollmsClient.from_config()` classmethod factory to instantiate clients directly from `.yaml`, `.json`, `.env` files, dictionaries, or the active environment
- feat(config:universal_modalities): add `BindingType.RAG` and `BindingType.CONNECTION` to `lollms_config.py` and register `"rag"` across `lollms_config_api.py` and profile builder workflows
- feat(cli:rag_profiles): wire two-tier RAG binding/model profiles and automated client creation into `lollms_code` CLI (`CodeAgentConfig`) and GUI (`agent_bridge.py`)
- feat(rag:modality): introduce new `rag_bindings` modality category with `LollmsRAGBinding` base class and `LollmsRAGBindingManager`
- feat(rag:safe_store): implement first-class `safe_store` RAG binding supporting dense vectors, BM25 FTS5 sparse search, RRF fusion, chunk reconstruction, W3C SPARQL 1.1 graphs, AES-128 encryption, and automated LLM generator bridging
- feat(rag:stores_architecture): enforce two-tier profile architecture for RAG where connection layer defines engines and execution layer defines persistent data stores
- feat(tools:rag): implement automated `tool_query_rag`, `tool_sparql_query`, `tool_add_document_to_rag`, and `tool_get_rag_info` in `BindingToolsBuilder`
- feat(config:rag): add RAG modality support to CLI/GUI configuration menus, environment serializers, and client resolvers
- feat(skills:tool_dependency): enforce automatic toolset loading when a skill requires specific tools, refusing to load the skill if required tools cannot be found or mounted
- feat(zoo:skill_tools): automatically search and install missing required tools from the `lollms_tools_zoo` repository when installing a skill from the skills zoo
- fix(personality:tool_loader): wire dynamic tool availability checking and tool loader on `skills_manager` in `lollms_personality`, enabling instant tool activation when skills load
- fix(lcp:multi_folder): expand `mount_tool_library` and `find_library_for_tool` to search across all configured tool folders (project, global, default)
- feat(gui:linear_history_edit): add linear history Edit and Resend actions to user speech bubbles, cleanly discarding subsequent turns from memory, transcript, and disk to continue execution from that point
- feat(gui:dynamic_mode): add 1-click Dynamic Mode quick toggle (`⚡ Dynamic: ON/OFF`) and `/dynamic` slash command activating autonomous effort scaling, task-adapted temperature, and auto max tokens
- feat(gui:linear_history_edit): add linear history Edit and Resend actions to user speech bubbles, cleanly discarding subsequent turns from memory, transcript, and disk to continue execution from that point
- feat(gui:dynamic_mode): add 1-click Dynamic Mode quick toggle (`⚡ Dynamic: ON/OFF`) and `/dynamic` slash command activating autonomous effort scaling, task-adapted temperature, and auto max tokens
- feat(turn:resumption): implement round-end state checkpointing and turn resumption across both `lollms_discussion` and `lollms_personality`, allowing interrupted turns to resume seamlessly
- feat(gui:resume_button): add dedicated "Resume Turn" top-bar button and `/resume` slash command to resume cut turns from their exact round checkpoint
- fix(gui:live_turn_persistence): implement live continuous auto-save during generation (on tool completion, artifact creation, sub-agent completion, and every 2.5s) to guarantee zero loss of transcript or work when interrupted midway
- fix(personality:cancelled_turns): preserve partial turn history and assistant actions in `self._conversation` upon cancellation instead of wiping the turn, enabling seamless resume
- feat(gui:session_persistence): resolve session loss when navigating between Settings and Chat by caching active sessions per workspace and replaying transcript history
- feat(gui:sessions_manager): add persistent Sessions Manager (`/sessions` and top navigation bar button) allowing users to save, resume, browse, and delete previous discussions from disk
- feat(gui:smart_scroll): implement intelligent stick-to-bottom auto-scroll allowing users to freely scroll up and inspect earlier history during generation without viewport snapping
- feat(gui:collapsibles): prevent live stream updates from overriding user-toggled expansion states on collapsible panels during generation
- feat(agent_state:realtime_flush): immediately flush completed artifacts to disk upon `</artifact>` tag closure, ensuring files are written to disk even during long multi-minute generation rounds
- fix(chat_page:code_viewer): replace `ui.code` with styled `ui.element("pre")` to eliminate black/empty code boxes and render streaming artifact tokens smoothly
- fix(agent_state:stream): eliminate empty string artifact chunk bug by capturing `incoming_chunk = self._pending_buffer` before clearing, enabling live content streaming into artifact viewports
- feat(agent_state:stream_complete): immediately emit `MSG_TYPE_ARTEFACT_BUILD_END` with parsed body content when `</artifact>` closing tag arrives
- fix(chat_page:live_artifacts): resolve empty collapsible artifact views by retaining `ui.expansion` object structure, adding `find_active_artefact_item()` and `update_code_box()` for real-time text rendering
- feat(gui:live_artifacts): stream live code and text directly into collapsible artifact expansion panels in real-time with animated spinner and dynamic header displaying the latest written line
- feat(config:auto_tuning): add Auto Temperature (task-adapted 0.15 for code/patches vs 0.7 for chat) and Auto Max Generation Tokens (auto-calculated from remaining context window)
- feat(subagent:ui): add detailed, user-facing collapsible cards in GUI and Rich panels in CLI for sub-agent spawning and completion with task directives, depth, effort, and reports
- feat(context:inspector): add dedicated 'Assembled Context (LLM View)' tab in Context Inspector displaying verbatim system prompt and turn history exactly as transmitted to the LLM backend
- fix(subagent:control): resolve loss of cancellation control by propagating `cancel_generation()` to active child agents via `SubAgentSpawner.cancel_active_child()`
- fix(subagent:doctrine): enforce headless worker doctrine forbidding sub-agents from asking human questions or awaiting confirmation, mandating immediate autonomous execution and structured `<report>` output
- feat(workspace:tree): redesign workspace tree context generation with compact sizes (`11.7 MB`), sequence clustering (`img_dalle__1..16.png`), relative child paths, and ```` ```text ```` encapsulation, shrinking token consumption by >60% and guaranteeing vertical line returns
- fix(chat:inspector): render Context Inspector system prompt and message tabs in preformatted `<pre>` containers to prevent markdown headers from inflating into 36px H1 headings
- feat(workspace:tree): redesign workspace tree context generation with compact sizes (`11.7 MB`), sequence clustering (`img_dalle__1..16.png`), relative child paths, and ```` ```text ```` encapsulation, shrinking token consumption by >60% and guaranteeing vertical line returns
- fix(chat:inspector): render Context Inspector system prompt and message tabs in preformatted `<pre>` containers to prevent markdown headers from inflating into 36px H1 headings
- feat(memory:readability): format working memory and deep memory handles with bold backticked IDs (`• **`[ID]`**`), dates, importance percentages, clean indented content, and double line returns to eliminate unreadable run-on blocks
- fix(agent_bridge:preview): eliminate redundant duplicate `=== ACTIVE MEMORIES ===` wrapper in context inspector
- fix(workspace:tree): wrap directory and file paths in backticks in `_build_workspace_tree_r()` to prevent Markdown from turning underscored filenames into collapsed italics
- feat(prompt:readability): format `=== TOOLS AVAILABLE ===` with bold backticked headers (`#### 🛠️ **`tool_name`**`), parameter lists, and clean double line returns to prevent markdown italic distortion and wall-of-text collapse
- feat(vlm:tool): implement self-contained `tool_inspect_image` and `tool_vlm_query` LCP tool library to allow visual inspection and categorization of workspace images via active VLM
- feat(skills:file_organization): mandate exhaustive file mapping and granular taxonomies, incorporating `tool_inspect_image` visual inspection for ambiguous images
- feat(subagent:telemetry): stream sub-agent execution live to GUI via `child_stream_relay` and render dedicated `worker_spawn_start`/`worker_spawn_end` panels in `chat_page.py`, eliminating silent 200s+ background stalls
- feat(memory:auditability): emit live `memory_consolidated` events with content and tags during autonomous consolidation passes, rendering dedicated audit cards in the chat transcript
- fix(patch:recovery): inject authoritative full-file rewrite instruction upon Aider SEARCH/REPLACE failure to break infinite search guessing loops
- fix(subagent:args): accept `effort`, `dynamic_effort`, and `**kwargs` in `SubAgentSpawner.spawn()` to resolve `TypeError: unexpected keyword argument 'effort'` crash
- fix(chat:artefacts): render actual content preview in `artefact_end` cards when no code symbols are present instead of `(no output)`
- docs(skills:file_organization): enforce invariant prohibiting `tool_spawn_sub_agent` invocation during Phase 3 confirmation request
- fix(stream:delimiters): include leading backtick-blockquote sequences (`` `> ``, `` >` ``) in tool tag matching and scrub trailing delimiters from `text_before`, completely eliminating leaked `` `> `` artifacts in speech bubbles
- feat(skills:resolver): add category (`workspace_management`) and tag matching to `SkillsManager.get_skill()`, preventing `Skill 'workspace_management' not found` errors
- fix(chat:delimiters): scrub stray trailing braces and backtick delimiters (e.g. `` `} `` or `}`) from speech bubbles and stream buffers
- fix(prompt:dedup): eliminate duplicate skills block injection where `=== AVAILABLE SKILLS ===` was rendered twice in the system prompt
- fix(skills:args): accept `title`, `name`, and `skill_name` flexibly in `tool_load_skill` and `tool_unload_skill` to prevent unexpected keyword argument `TypeError` crashes
- feat(agent:parser): intercept and parse curly-brace pseudo-tags (`{tool}{...}`, `{artifact}{...}`) into actionable executions to eliminate 16-round stall loops
- fix(chat:render): scrub unclosed broken backtick fences and pseudo-tags in `chat_page.py` so headings do not inflate into massive H1 elements
- fix(skills:matching): implement fuzzy and normalized title/slug resolution in `SkillsManager.get_skill()` so `"file_organization"`, `"file_and_folder_organization"`, and title variants all match
- fix(agent:stream): intercept bare and blockquoted tool JSON calls (`>{"name": ...}`) to execute tools cleanly without leaking raw JSON into chat bubbles
- fix(skills:lifecycle): return instant confirmation directive when a skill is already active in context to prevent repetitive loading loops across rounds
- fix(skills:discovery): resolve project root crawling and fix key deduplication bug so all 25 skills are restored and visible in the Sub-WS panel
- feat(skills:gui): render Global and Bundled skills alongside Handbag and Project skills in `chat_page.py` with individual `[U]`/`[C]` toggle controls
- feat(skills): enforce unloaded-at-start doctrine (`[U]`) across all skills so no skills are loaded into active context (`[C]`) on startup without explicit request
- feat(skills): add `tool_unload_skill` allowing LLM to dynamically unload skills from context to free tokens
- fix(skills:sidebar): eliminate duplicate skills display in sidebar by deduplicating dual-indexed title and slug keys in `SkillsManager.get_unique_skills()` and `agent_bridge`
- fix(chat_page): resolve NiceGUI RuntimeError in `delete_message()` by firing `ui.notify()` before deleting parent element row
- fix(llama_cpp_server, history): scrub `[TOOL_CALLS]` and template tokens across message normalization and context export to prevent llama-server `Failed to parse input at pos 0` API errors
- fix(agent_state): allow functional action tags (`<tool>`, `<artifact>`, `<unlock_file>`) to bypass code fence swallowing and add post-stream recovery sweep for missed action tags
- feat(agent_state): support tool calls with attributes on the `<tool>` tag itself (e.g. `<tool name="..." parameters="...">`)
- fix(skills:file_organization): eliminate markdown code fences wrapping `<artifact>` and `<tool>` examples in `SKILL.md` to prevent models from generating fenced XML and empty mapping tables
- fix(history): eliminate duplicate scratchpad and memory injection in user messages by setting `scratchpad=""` and `memory_manager=None` in `_HistoryContextAdapter`
- fix(memory): format working memory entries as clean newline-separated bullet points (`• [ID] (Date) Content [tags: ...]`) and purge corrupted boundary tokens and backticks
- fix(llama_cpp_server): correct context size detection in `/props` to query `n_ctx` rather than concurrency `total_slots` which reported 1 token
- refactor(scratchpad): redefine scratchpad as strictly ephemeral per-session working buffer (maintained across rounds and turns, purged between sessions)
- fix(scratchpad): prevent injection of empty or boilerplate scratchpads into system prompts
- fix(chat_page, cli): automatically wipe `.lollms_code/scratchpad.md` on new session launch, `/clear-history`, and `clear_conversation()`
- fix(agent): implement Greeting Immunity Shield to programmatically intercept and discard unprompted file writes and tool calls on conversational greetings like 'HI THERE'
- fix(agent): suppress stale `CURRENT.md` roadmaps and historical `scratchpad.md` injections on greeting turns
- fix(chat_page): automatically reset `.lollms_code/CURRENT.md` on `new_session()` and `/clear-history` to prevent previous session plans from haunting new sessions
- fix(skills:gui): parse YAML frontmatter cleanly in `_view_skill_content()` to display structured metadata chips (Author, Version, Category, Date, Tags) and strip raw frontmatter from the markdown viewer
- fix(agent_state): enable functional tag execution (`<unlock_file>`, `<tool>`, `<artifact>`) even when enclosed in markdown backticks or code fences
- fix(agent_state): add post-stream sweep to intercept and execute missed action tags and prevent raw XML leakage into speech bubbles
- feat(skills): bundle default skills inside `lollms_code/skills/` and auto-sync on launch to `~/.lollms_client/skills/` to guarantee global availability regardless of active workspace
- feat(agent_state): parse and execute direct XML tool calls (e.g. `<tool_load_skill title="..." />`) preventing unhandled tag loops
- feat(skills): support dual-key indexing by both human-readable title and directory slug in `SkillsManager`
- fix(skills): add round-aware skill context injection to transition from Round 1 loading mandate to Round 2+ active execution directive, eliminating repetitive `tool_load_skill` apology loops
- fix(agent_state): support blockquoted tool calls (`(?:>\s*)?<\s*tool`) to intercept tools emitted inside markdown blockquotes
- fix(personality): inject explicit directive after `tool_load_skill` confirming doctrine receipt and commanding immediate execution of Phase 1
- fix(personality): resolve false greeting detection where words containing 'hi' (e.g. 'this' in 'organize this folder') falsely triggered greeting mode
- fix(agent_state, chat_page): strip model `[TOOL_CALLS]` tokens and leaked `=== END ACTIVE SKILLS ===` lines from visible speech bubbles
- fix(skills): add critical execution mandate to `file_organization` to execute Phase 1 and Phase 2 immediately without asking permission to start
- feat(skills): add Phase 3.5 Iterative Refinement protocol to `file_organization` so user inquiries and folder adjustments update `classes.md`/`mapping.md` without triggering migration prematurely
- fix(personality): de-escalate error recovery prompts to calm guidance to prevent 8B model panic loops and backtick cascades
- fix(agent_state): support divider-prefixed functional tags (`---<tool>`, `---<artifact>`)
- feat(skills): add Skill-First Dispatch Mandate requiring the agent to call `tool_load_skill` in Round 1 before taking action when a specialized skill exists
- feat(skills): add proactive recommendation engine in `SkillsManager` highlighting matching skills for the user's prompt (e.g. `file_organization` on "organize this folder")
- fix(skills:paths): implement dynamic root crawler `_find_project_root()` in `cli.py` and `agent_bridge.py` ensuring `skills/` is discovered reliably
- feat(spinoff): activate spinoff sub-agent factory (`tool_spinoff_agent`) by default across agent and discussion sessions
- feat(skills): add comprehensive `file_organization` skill implementing 4-phase taxonomy generation (`classes.md` & `mapping.md`), sub-agent ambiguity inspection, mandatory user confirmation gate, and delegated migration execution
- perf(skills): enforce loadable-by-default visibility across all skills, shrinking prompt overhead from ~11,000 tokens to under ~900 tokens (91% context reduction)
- fix(skill): prevent unflagged skills from defaulting to 'visible' in mixed mode
- refactor(personality, lollms_code): massively streamline monolithic system prompt from 63k chars (~11k tokens) down to ~1.2k tokens
- feat(skills): modularize specialized agent instructions into on-demand loadable skills (`file_organization`, `document_analysis_and_extraction`, `bibliography_and_research`, `deep_websearch_and_extraction`, `fullstack_development`, `git_workflow_mastery`, `desktop_automation`)
- feat(handbag): auto-seed modular skills into the default coder handbag on initialization
- fix(personality): resolve empty response stall on greetings like 'hi' by synthesizing conversational greeting fallback and reprompting on empty Round 1 <done/> emissions
- fix(personality): clarify Rule 5 mandate to write conversational response text before emitting <done/>, strictly forbidding solitary <done/> emissions
- fix(memory): prevent memory task contamination by adding strict MEMORY DOCTRINE asserting memories are passive background facts, not current task instructions
- feat(memory): add automatic startup deduplication (`deduplicate_all()`) and task backlog demotion (`clean_task_backlog_memories()`) to eliminate redundant memories and merged notes
- fix(memory): guard `auto_pull_deep_memories` against firing on greetings and trivial messages like 'Hi' or 'Hello'
- fix(memory): filter ephemeral task requests from `_autonomous_memory_consolidation()`
- fix(lollms_code:gui): resolve argument mismatch in `get_context_preview()` allowing flexible 2, 3, or 4 argument invocation for Context Inspector
- feat(llama_cpp_server): add `port` parameter to `description.yaml` and support custom port assignment for concurrent instances with separate models folders
- feat(bindings): establish local resource management contract (`is_local()`, `is_model_loaded()`, `has_active_resources()`, `get_loaded_models()`, `unload_model()`) in `LollmsBaseBinding`
- feat(core): add `free_local_binding_resources()` and `ensure_model_loaded()` to coordinate VRAM/RAM reclamation across bindings on OOM before retrying
- refactor(bindings): retain current working directory (`Path(".")`) as the universal default path for standalone apps while supporting `system_dir` / `cwd` override
- feat(lollms_code): configure CLI and GUI to explicitly pass `~/.lollms_client` as the `system_dir`/`cwd` override to prevent binary downloads and venv creation inside project workspaces
- feat(settings): add dynamic binding commands interface driven by `description.yaml` with live progress bars and model download/update support
- feat(settings): warn on duplicate binding/profile aliases with choice to return and edit name or auto-save with incrementing suffix
- fix(config): eliminate phantom binding generation by discovering binding aliases strictly through `_BINDING_NAME` declarations
- fix(env_config): auto-purge orphaned ghost binding keys on environment reload
- fix(settings_page): eliminate stacked dialog bug when adding server bindings by replacing multi-listener registration with a tab-aware dispatcher
- feat(lollms_code:gui): replace static model label in header bar with interactive dropdown to select and persist default LLM binding/model profiles
- feat(lollms_code): add "Inspect Context" dialog and `/inspect` command to visualize exact prompt messages, memory blocks, and runtime parameters sent to the LLM
- fix(memory): ensure `=== ACTIVE MEMORIES ===` and `=== DEEP MEMORY HANDLES ===` zones are always rendered in prompt context even when empty to prevent model amnesia
- fix(sub_workspace): delegate directory inputs in `import_file` to `import_folder` instead of raising FileNotFoundError
- fix(folder_picker): strictly enforce `p.is_file()` validation across `pick_file` tiers
- fix(chat_page): wrap all reference import operations in exception handlers and support directory fallback
- fix(lollms_code:gui): use `pick_file` instead of `pick_folder` for reference file import buttons
- feat(lollms_code:gui): launch application as a maximized desktop window with native titlebar rather than borderless fullscreen
- feat(memory): mount callable memory tools (`tool_save_memory`, `tool_search_memory`, `tool_load_memory`) in LollmsPersonality
- feat(memory): hydrate both Working Memory and Deep Memory Handles zones into context in LollmsPersonality
- fix(memory): add `<mem_load>`, `<mem_search>`, `<mem_delete>`, and `<mem_tag>` interception to `_AgentStreamState`
- feat(lollms_code): add `/memory on|off|toggle` slash commands and quick memory status toggle in GUI header
- docs(memory): enforce strict same-response memory saving mandate when users instruct the agent to remember facts/rules
- feat(lollms_code:gui): refresh workspace tree on every round end using lazy loading and preserving expanded directory state
- fix(chat): define `had_prior_actions` in `_mixin_chat.py` and eliminate intent heuristics when `enforce_end_tag=True`
- fix(test_high_grade_agent): ensure tests requiring `<done/>` include `<done/>` in scripted completions and test multi-round continuation until `<done/>`
- fix(lollms_code): ensure `enforce_end_tag=True` strictly requires `<done/>` to stop and intercepts action intent statements like `"I'll copy..."` to force execution
- fix(chat): resolve UnboundLocalError for is_inside_thoughts in _StreamState.feed() by positioning <effort> interception after thought bounds calculation
- fix(chat): correct delta text extraction and round tag handling in empty response guard to terminate on round 1
- fix(gui): resolve main-thread hang in NiceGUI by preventing secondary modality daemons from synchronously blocking client startup
- fix(ttm, tti): default `wait_for_server` to `False` and spawn daemons in background threads to avoid freezing caller event loops
- fix(lollms_code): restrict GUI client modality profiles to TTI, TTS, and STT, eliminating unwanted TTM/TTV daemon launches
- feat(chat): support infinite reasoning rounds (`max_steps=0` / `max_nb_rounds=0`) with safety warnings across CLI, GUI, and core engines
- docs(effort): document dynamic effort scaling, sub-agent effort delegation, and infinite rounds across all guides
- feat(lollms_code): add reasoning effort and dynamic effort configuration to CLI, GUI settings, and fast effort top-bar selector
- feat(chat): add `dynamic_effort` support to discussion and personality chat loops via `<effort level="..."/>` tags
- feat(agentic): add known effort assignments and dynamic effort delegation to sub-agents and spinoff tools
- feat(events): emit `<round id="N"/>` tag in chunk stream on new round start when in PROCESSING_TAG_MODE
- fix(diffusers): implement get_settings and list_services in DiffusersTTIBinding to fulfill abstract base class
- fix(diffusers): resolve host_address attribute error and pip requirement format in DiffusersTTVBinding
- fix(events): ensure non-chunk artifact state events are not emitted in PROCESSING_TAG_MODE
- fix(personality): ensure evicted unindexed artifacts are on-demand imported and unlocked to [C] during rolling compaction
- feat(diffusers): add install_model and pull_model commands across TTI, TTM, and TTV diffusers bindings to download Hugging Face models into searchable local directories# [Unreleased]

- feat(events): enforce EventMode doctrine across discussion and agent streams (PROCESSING_TAG_MODE, FULL_CALLBACK_MODE, MIXED_MODE, SILENT_MODE)# [Unreleased]

- refactor(lollms_client): update pathlib imports# [Unreleased]

- refactor(bindings): update diffusers and xtts bindings

#
- chore(lollms_client): update bindings and server initialization# [Unreleased]

- feat(ollama): add model zoo with download helper

#
- feat(lollms_discussion): add image activation toggle support# [Unreleased]

- chore(lollms_client): bump version to 1.8.1 and remove debug output

#
- refactor(lollms_client): clean up imports and extend discussion handling# [Unreleased]

- feat(lollms_discussion): add new discussion features and update changelog

#
- feat(bindings): update multiple LLM bindings and context size definitions# [Unreleased]

- feat(discussion, tti-bindings): enhance image handling and expand Diffusers/OpenRouter support

#
- feat(tti-bindings): enhance image handling and add Diffusers/OpenRouter support# [Unreleased]

- feat(tti): refactor and clean up TTI bindings

#
- feat(xtts): ensure server is running before making API calls# [Unreleased]

- feat: bump package version to 1.9.2

#
- feat(diffusers): add TTI binding implementation# [Unreleased]

- refactor(vibevoice): remove deprecated VibeVoice TTS binding and cleanup


## [2026-10-01 14:24]

- feat(artefact): append .md suffix to audio transcript import titles

## [2026-10-01 13:57]

- feat(stt): add whisper binding with local transcription support

## [2026-10-01 12:37]

- docs(discussion): document real-time tool execution streaming pipeline

## [2026-10-01 00:02]

- fix(gui: deck_page): fix infinite loop in tool stack handling

## [2026-09-30 19:53]

- feat(tools): enhance execute_python tool and tool binding capabilities

## [2026-09-30 09:26]

- refactor(stt): update whisper batching tests and discussion core mixin

## [2026-09-29 09:21]

- fix(lollms): remove unsafe final disk reconciliation cleanup

## [2026-09-28 22:07]

- feat(lollms): update chat_page UI and workflow_engine/process execution logic for dynamic user interactions

## [2026-09-28 18:25]

- Fix(diffusers-bindings): resolve race condition in connection pool for `diffuser_instant_model` and enhance path handling

## [2026-09-28 09:23]

- fix(llm-bindings): update safe flag checks and add dynamic mode in chat UI

## [2026-09-28 01:43]

- changelog: bump version to 1.20.5 and add lcp_binding attribute in ChatMixin

## [2026-09-28 01:31]

- feat(agent): add support for per-call `think` and `reasoning_effort` parameters in OpenAI reasoning

## [2026-09-27 23:51]

- feat(llm): add GLM/vLLM backend thinking deactivation support via chat_template_kwargs

## [2026-09-25 14:20]

- fix(diffusers): update round ID emission logic for chunk stream in event processing

## [2026-09-25 11:09]

- feat(lollms-client): add install_model and pull_model commands for diffusers bindings in TTI, TTM, and TTV

## [2026-09-25 13:10]

- fix(events): eliminate triplication of <think> and </think> tags in PROCESSING_TAG_MODE by deduping thought stream wrappers across bindings and stream parser states
- fix(openai, lollms): stop emitting literal <think> boundary strings as thought chunk payloads to prevent outer handler re-wrapping

## [2026-09-25 07:49]

- fix(events): enforce EventMode enforcement in discussion and agent streams with PROCESSING_TAG_MODE compatibility

## [2026-09-25 06:10]

- `fix(doc: version bump) => Update lollms_client/__init__.py from "1.20.1" to "1.20.2"`

## [2026-09-25 05:56]

- fix(lollms_client): increment version in client initialization for new patch release

## [2026-09-25 05:55]

- fix(lollms_client): cleanup deprecated TTS bindings and update CHATLOG.md entry

## [2026-09-25 05:55]

- a commit message for:

## [2026-09-25 01:36]

- `fix(lollms_client): update minor version for v1.20.0`

## [2026-09-25 07:55]

- fix(lollms_code): resolve startup freeze by eliminating eager recursive workspace crawling and importing
- perf(lollms_personality): replace unpruned `Path.rglob('*')` with directory-pruned `os.walk` across workspace scans
- perf(lollms_chat_core): optimize `take_workspace_snapshot` to prune `.git`, `venv`, and `node_modules` at directory entry
- test(lollms_code): add unit tests ensuring sub-second startup and on-demand file loading on large workspaces

## [2026-09-25 00:14]

- fix(docs): update documentation for music and song generation with TTM

## [2026-09-25 02:25]

- docs(phenix): establish Project Phenix documentation suite in `phenix_docs/` covering shared daemon IPC, continuous micro-batching, and unified API reference
- docs(ttm): document Diffusers TTM binding and MiniMax Music 3 full-song generation across root README, library README, and user/developer guides
- docs(ports): establish canonical loopback port registry (9632-9637) eliminating port drift across all modalities

## [2026-09-25 02:15]

- feat(ttm:diffusers): introduce new multi-user Diffusers TTM binding on dedicated port 9637 with self-spawning shared singleton daemon, FileLock synchronization, and continuous job queueing
- feat(ttm:minimax): add MiniMax-Music3 to model zoo, supporting full 5-minute song generation with expressive vocals and structured progression
- feat(ttm:core): extend LollmsTTMBinding and LollmsClient with first-class `generate_song` and `generate_song_from_lyrics` APIs
- feat(personality): add `tool_generate_song` to BindingToolsBuilder to enable autonomous agentic full-song generation

## [2026-09-25 02:05]

- fix(test): wire proxy delegation in _mk_discussion fixture so ChatMixin tests access discussion mock methods seamlessly
- fix(chat_mixin): guard add_message and lollmsClient across ChatMixin.chat() for isolated execution safety

## [2026-09-25 01:45]

- fix(chat_mixin): guard lollmsClient attribute access to support bare ChatMixin test instances
- fix(chat_mixin): resolve AttributeError on completed_actions in _StreamState empty response guard
- fix(personality): enable round 1 conversational short-circuit for pure conversational answers without tools or intent announcements

## [2026-09-25 01:10]

- fix(personality): resolve PEP 572 syntax error by replacing unparenthesized assignment expression in round 1 preamble check with standard boolean evaluation

## [2026-09-25 01:05]

- fix(vllm): guard load_model in __init__ with auto_start_server flag to allow isolated test inspection without spawning server
- fix(chat_core): improve sanitize_host_paths regex to handle quoted and space-containing host paths
- fix(chat_mixin): add hasattr guard for _get_memory_manager and reinstate duplicate artifact warning break
- fix(chat_mixin): add zero-token empty response guard to terminate pathological loops immediately
- fix(artefact): preserve file_ext explicitly for data and rich text imports while omitting it for plain text
- fix(artefact): remove readme from extensionless files so README documents receive .md extension
- fix(artefact): preserve db_content in add() to allow healing unlinked files in agentic mode
- fix(core_mixin): register artifacts with title=f_path.name and physical_path=rel_str during sync
- fix(agent_state): halt generation by returning False when <processing> tag mimicry is detected
- fix(personality): allow single-turn conversational answers to finish without forcing continuation

## [2026-09-25 00:45]

- fix(whisper): replace Unicode box characters with ASCII hyphens in server banner to prevent CP1252 charmap encoding crash on Windows
- fix(whisper): use direct requests calls to ensure test mock compatibility
- fix(vllm): implement missing abstract method `get_model_info` in VLLMBinding
- fix(openai): fallback to 'EMPTY' api_key when service_key is omitted to support local inference engines
- fix(agent): expose `tool_spawn_sub_agent` in `_discover_tools()` when `enable_sub_agents` is active
- fix(artefact): ensure SVG image artifacts are written to disk by inspecting file suffix
- fix(memory): remove premature soft-delete purge in `dream()` to allow Dreamer LLM evaluation pass
- fix(discussion): wire client `chat()` callable override in `regenerate_branch()` and add mimicry interception guard

## [2026-09-25 00:34]

- feat(tts:piper): migrate to self-spawning shared daemon on dedicated port 9635, fix destructive __del__ termination bug, add FileLock and HMAC token auth
- feat(tts:bark): migrate to shared model daemon on port 9636 with dynamic micro-batch queue, fix __del__ server kill bug, add token auth and shutdown RPC
- feat(tts:xtts): isolate to non-conflicting port 9634, harden multi-process FileLock with double-checked probe, add queue worker and token auth
- feat(tti:diffusers): eliminate port-drifting pathology on port 9632 to enforce true single-instance VRAM mutualization, add token auth and shutdown RPC

## [2026-09-25 00:27]

- feat(stt:whisper): implement self-spawning shared daemon architecture with multi-process mutualization, eliminating port drift and race conditions
- feat(stt:whisper): add event-driven continuous dynamic micro-batching (`batch_window`, `max_batch_size`) with batched mel decoding for short audio queries
- feat(stt:whisper): add constant-time HMAC token authentication (`whisper_server.token` with 0o600 permissions) and `/shutdown` lifecycle RPC
- fix(stt:whisper): replace timeout=0 filelock with double-checked probe pattern to allow seamless multi-client attachment

## [2026-09-24 22:14]

- fix(vllm): update description and binding class to reflect vllm registry refactoring

## [2026-09-24 22:10]

- feat(lollms-client): update server mutualization and recipe presets documentation

## [2026-09-24 21:15]

- git commit -m "fix(openai): add OpenAI API description updates and minor GUI refinements"

## [2026-09-24 22:30]

- feat(reasoning): add cross-model reasoning effort translation (`supported_reasoning_efforts`) with topological anchor projection
- feat(multimodal): add native video comprehension input (`videos`, `video_enabled`, `normalize_video_input`, `has_video_capability`)
- feat(multimodal): add GLM-5.3-Flash sequential image embedding support (`glm_image_embedding`)
- fix(openai): preserve visible `<think>` tags in output and stream thoughts via `MSG_TYPE_THOUGHT_CHUNK` for clean reprompt stripping

## [2026-09-24 19:45]

- feat(openai:description) add OpenAI description configuration updates

## [2026-09-24 18:19]

- feat(openai): add OpenAI description and core API updates

## [2026-09-24 16:40]

- fix(openai): align OpenAI client initialization imports for better readability

## [2026-09-24 12:17]

- `fix(lollms_client): update version from 1.19.12 → 1.19.13 in __init__.py`

## [2026-09-24 11:58]

- fix(pymupdf): suppress MuPDF stdout/stderr with "fd:2" instead of "0"

## [2026-09-24 11:00]

- `fix(client): resolve import warnings and refactor connection module binding into lollms_bindings_utils`

## [2026-09-24 09:42]

- fix(lc): correct default tool import paths and resolve circular dependencies in the document editor module

## [2026-09-24 09:35]

- fix(lollms): minor refactoring and API adjustment for data validation and personality interface handling

## [2026-09-24 08:54]

- fix(gui): consolidate GUI page error checks for consistent CLI validation alignment

## [2026-09-23 00:01]

- fix(sandbox): resolve race condition in connection pool logic for sandboxed LLM execution

## [2026-09-22 23:35]

- fix(lollms_core): optimize token estimation logic for performance

## [2026-09-22 23:30]

- ---

## [2026-09-22 22:42]

- `feat(lollms_client): refactor CLI and GUI interfaces with chat_page and env_config improvements`

## [2026-09-22 22:26]

- ---

## [2026-09-22 11:36]

- docs(chat): update README examples to reflect new streaming callback parameter

## [2026-09-22 07:18]

- `fix(binding): resolve LCP binding initialization issues in lcp/__init__.py and system_shell`

## [2026-09-22 07:07]

- fix(lcp): Fix race condition in default_tools shell components and ensure proper CLI integration

## [2026-09-22 01:58]

- fix(doc): update description.yaml and __init__.py for Novita AI client bindings

## [2026-09-21 19:17]

- fix(editor:): restructured CLI and GUI initialization to enforce strict task macro-level requirements and minor refactoring in chat_page

## [2026-09-21 08:10]

- `fix: update core and client library imports to resolve circular dependency warnings in lollms_core and client API modules`

## [2026-09-20 21:02]

- fix(spinoff): add `trace_exception` import to handle unhandled exceptions in async operations

## [2026-09-20 20:46]

- git commit -m "Fix(lllms_client): Improve CLI and GUI error handling"

## [2026-09-20 15:11]

- fix(lollms-binding): update minor imports and syntax fixes in lollms/__init__.py

## [2026-09-19 17:13]

- feat(llm-bindings): add model listing support for Ollama and OpenAI bindings

## [2026-09-17 12:31]

- fix(cli): remove deprecated chat_core import from __init__.py & update lollms_discussion

## [2026-09-17 10:15]

- feat(bindings): enhance llm bindings and personality skills handling

## [2026-09-16 11:32]

- feat(llm-bindings): add new capabilities across claude, gemini, lollms, ollama, and openai bindings

## [2026-09-15 22:36]

- delete(helloworld): remove unused Hello World script

## [2026-09-15 22:36]

- feat(agentic): add spinoff tools and sub-agent spawner enhancements

## [2026-09-14 21:32]

- feat(personality): add personality tools support to chat mixin and update documentation

## [2026-09-14 09:11]

- build(pyproject): update build config and dependency list

## [2026-09-14 01:54]

- refactor(artefact): consolidate artefact module and remove duplicate root file

## [2026-09-13 15:58]

- Remove outdated chat discussion mixin and refactor agentic documentation.

## [2026-09-13 00:26]

- feat: update LollmsCode app and core discussion mixins

## [2026-09-10 14:03]

- feat(artefact): add export functionality and enhance artefact documentation

## [2026-09-10 13:19]

- fix(core): harden artefact syncing and context handling across client and chat mixin

## [2026-09-09 23:21]

- docs(readme): update discussion and personality documentation and refine execute_python tool

## [2026-09-09 21:06]

- feat(artefact): add artefact symbol detection handling in chat pipeline

## [2026-09-09 14:35]

- feat(tools): update execute_python tool and chat mixin

## [2026-09-09 13:51]

- fix(chat): sync active artefacts before LCP workspace snapshot and correct workspace root detection

## [2026-09-09 13:40]

- feat(personality): enhance agent state and personality handling with workspace tools updates

## [2026-09-09 12:09]

- feat(client): update LCP tool bindings, chat mixin, and personality handling

## [2026-09-08 23:04]

- fix(lollms_client): minor imports and dependency adjustments in core and CLI files

## [2026-09-08 14:31]

- feat(tools): add SPARQL query tool and enhance Python code execution

## [2026-09-08 11:30]

- feat(discussion): add sovereign discussion module enhancements to core mixins

## [2026-09-07 14:33]

- fix(chat): preserve verbatim functional tags in latest assistant turn to prevent phantom completions

## [2026-09-07 08:48]

- fix(stt): remove merge conflict artifact and fix lock file cleanup in whisper server

## [2026-09-07 08:43]

- fix(stt): update whisper binding and server transcription logic

## [2026-09-07 08:31]

- feat(stt): add filename passthrough and harden Whisper transcription endpoint

## [2026-09-06 23:53]

- feat(code): add agent config loading to CLI and enhance Lollms binding settings

## [2026-09-06 20:51]

- fix(doc: update docstring to reflect new `n_ctx` being nullable and adjust command-line arguments)

## [2026-09-06 16:50]

- fix(artefact): safely handle non-string logical_content when stripping

## [2026-09-04 20:08]

- fix(agent): resolve context tag stall loop by resetting stream state on context actions

## [2026-09-04 18:28]

- fix(personality): correct parameter name in edit_image call from image to images

## [2026-09-04 16:41]

- fix(tti): simplify SSL verification and restrict forwarded kwargs

## [2026-09-04 13:07]

- fix(chat): resolve issue in chat mixin

## [2026-09-04 12:37]

- fix(chat): preserve existing tools binding and auto-provision LCPBinding for code execution

## [2026-09-04 12:17]

- refactor(client): update lollms bindings, personality state, and cli configuration

## [2026-09-04 09:05]

- refactor(client): update artefact handling and discussion mixins

## [2026-09-04 07:29]

- feat(apps): implement lollms_loops and remove lollms_discussions

## [2026-09-03 14:09]

- fix: improve document editor and code CLI modality resolution

## [2026-09-03 06:32]

- feat(lollms_code): add TTI capability prompt and refine SSL verification

## [2026-09-03 00:51]

- feat(client): implement artefact management and as-is document processing

## [2026-09-03 13:36]

- refactor(client): reorganize source files into src directory and update modules

## [2026-09-02 16:03]

- fix(document_editor): resolve PyMuPDF flags safely to prevent version crashes

## [2026-09-02 16:02]

- refactor(personality): update agent state and document tool bindings

## [2026-09-02 15:46]

- fix(lollms_code): improve agent state handling and tool bindings across CLI and GUI

## [2026-09-01 15:27]

- feat(lollms_code): add agent CLI config and update GUI chat and personality modules

## [2026-09-01 01:13]

- refactor(client): update core components and improve documentation

## [2026-08-31 22:43]

- feat(smart_router): improve binding resolution and fix manager path discovery

## [2026-08-31 22:22]

- feat(client): update history, personality, and LCP tool bindings

## [2026-08-31 00:21]

- feat: enhance lollms discussion and personality modules

## [2026-08-28 16:29]

- feat(tools): enhance default tool bindings and integrate skills manager

## [2026-08-28 01:32]

- feat(personality): conditionally mount LCP document and data tools based on workspace file types

## [2026-08-27 23:31]

- docs(lollms_client): update README with expanded architecture and routing details

## [2026-08-27 23:30]
- fix(personality): pre-inject skills context into system prompt and refine skill loading

## [2026-08-27 09:31]

- fix: resolve issues in chat mixin, agent state, personality, and default tools

## [2026-08-26 15:38]

- fix: resolve issues in stream rendering, file import, and personality agent state

## [2026-08-25 10:16]

- feat(personality): update agent state and document tools

## [2026-08-24 23:34]

- refactor(client): update personality management and document tools bindings

## [2026-08-24 22:17]

- refactor(personality): update skill and skills manager implementations

## [2026-08-24 21:22]

- feat(client): enhance agentic capabilities, personality management and artefact handling

## [2026-08-24 13:37]

- feat(chat): enhance agent bridge and chat page functionality

## [2026-08-24 08:34]

- feat(config): hydrate config from YAML fallback before forcing setup wizard

## [2026-08-24 08:18]

- feat(cli): customize menu exit text and safely initialize LCP tool libraries

## [2026-08-24 07:18]

- feat: enhance universal profiles and multi-model routing with expanded documentation and examples


## [2026-08-20 23:30]

- docs: cleanup documentation and update core client components

## [2026-08-20 08:45]

- fix: resolve bugs in cli, artefact, chat, and personality modules

## [2026-08-19 14:59]

- fix(personality): update agent state and python code execution tool

## [2026-08-19 07:50]

- fix(personality): update personality and skills manager

## [2026-08-19 07:27]

- refactor(core): update lollms_code CLI pipeline, personality, and system_shell tool

## [2026-08-18 23:36]

- feat(tools): sync generated plots to discussion artefacts system

## [2026-08-18 09:18]

- feat(personality): update agent state and personality handling in lollms code CLI

## [2026-08-18 00:26]

- feat(gui): add file visibility commands and update agent environment rules

## [2026-08-18 00:00]

- feat(core): enhance personality, artefact, and tool bindings across lollms client

## [2026-08-15 00:41]

- refactor(lollms_code): update CLI pipeline and GUI components

## [2026-08-14 23:42]

- feat(agent): handle think blocks in stream parsing

## [2026-08-14 12:45]

- feat(cli): enhance autonomous workflow with state/memory segregation and stream buffering

## [2026-08-14 12:30]

- fix(lollms_code): use current working directory as default workspace

## [2026-08-14 12:02]

- refactor(core): improve CLI rendering, artefact sanitization, and personality state management

## [2026-08-12 10:57]

- fix(lollms_personality): update agent state and personality handling

## [2026-08-09 09:42]

- refactor: update core modules and bindings across lollms_client

## [2026-08-05 18:53]

- fix(personality): resolve issue in lollms_personality module

## [2026-08-05 18:46]

- refactor: unify LollmsPersonality and Handbag systems across agent and discussion modules

## [2026-08-05 13:48]

- docs(readme): update unified configuration description

## [2026-08-04 16:15]

- feat(agent): integrate artefact system with agent and discussion modules

## [2026-08-03 16:35]

- refactor(agent): remove config_wizard and update lollms_agent examples

## [2026-08-02 14:52]

- docs: update READMEs and enhance MCP tool bindings security

## [2026-08-01 16:42]

- docs(lollms_discussion): update README and chat mixin documentation

## [2026-08-01 16:10]

- refactor(lollms_discussion): improve context sanitizer and diet protocol with updated cognitive decision tests

## [2026-07-31 18:31]

- fix(llm-bindings): correct line number references in multiple binding modules

## [2026-07-31 17:46]

- feat: add agentic tools and update examples

## [2026-07-16 06:15]

- feat(discussion): add option to suppress images for non-vision LLMs

## [2026-07-15 12:19]

- feat(code-agent): add configuration wizard and enhanced LLM binding config

## [2026-07-15 01:28]

- feat(app): refactor application structure and remove legacy server implementation

## [2026-07-15 00:34]

- refactor(lollms_agent): update _parse_skill_md implementation

## [2026-07-15 00:12]

- refactor(client): remove lollms_agent and update core client dependencies

## [2026-07-14 19:59]

- docs(examples): update agentic personality tools example and env config

## [2026-07-13 16:15]

- fix(openai): sanitize tools for NVIDIA NIM compatibility and fix tool injection

## [2026-07-13 15:58]

- fix(openai): sanitize tools for NVIDIA NIM compatibility and fix tool injection

## [2026-07-07 22:06]

- fix(openai): set completion format to chat and harden debug logging for responses

## [2026-07-07 17:38]

- fix(openai): handle base address suffix and null values for host address

## [2026-07-07 12:58]

- feat(openai): add base_address and open_ai_host_address to OpenAIBinding

## [2026-07-06 10:48]

- feat(ttm): add generate_song and generate_song_from_lyrics abstract methods to LollmsTTMBinding

## [2026-07-03 04:44]

- refactor(artefact): update artefact handling and import logic

## [2026-07-02 08:07]

- fix(openai): use flat reasoning_effort instead of nested reasoning dict

## [2026-07-01 20:34]

- feat(openai): update binding implementation and expand description metadata

## [2026-06-30 11:26]

- refactor: minor updates to file import and chat mixin

## [2026-06-30 00:32]

- feat(llm_bindings): add sglang support

## [2026-06-29 01:16]

- chore: bump version to 1.15.6 and update prompt rules and memory exports

## [2026-06-29 00:47]

- chore: bump version to 1.15.5

## [2026-06-29 00:28]

- feat(discussion): enhance chat settings and UI integration

## [2026-06-29 00:05]

- refactor(chat): update workspace state initialization and chat mixin logic

## [2026-06-28 23:39]

- refactor(discussion): reorganize discussion mixins and cleanup artefacts logic

## [2026-06-28 22:30]

- feat(discussion): refine artefact visibility defaults and clean up tool call blocks

## [2026-06-28 21:46]

- fix(ui, artefacts, chat): stop thinking timer on close and improve vision hydration

## [2026-06-28 20:59]

- feat(memory): enhance cognitive memory implementation and update related examples and tests

## [2026-06-26 12:48]

- refactor: update chat mixin and semantic data engineer tools

## [2026-06-24 08:15]

- refactor(client): remove loggers from public api exports

## [2026-06-22 22:22]

- chore: bump version to 1.15.3

## [2026-06-22 01:27]

- refactor: clean up formatting in app.js and _mixin_chat.py

## [2026-06-22 00:56]

- chore(version): bump version to 1.15.2

## [2026-06-22 00:50]

- refactor(client): update discussion mixins and artefact management logic

## [2026-06-21 23:15]

- feat(discussion): enhance chat memory and tool integration for semantic data engineering

## [2026-06-20 23:56]

- refactor(core): restructure lollms client and update llm bindings

## [2026-06-19 09:33]

- chore: remove unnecessary None file containing patent data

## [2026-06-17 22:59]

- refactor(client): update web interface and enhance core discussion logic

## [2026-06-15 22:20]

- feat(app): update frontend UI and refine LLM binding integrations

## [2026-06-15 01:05]

- fix(discussion): add missing newlines to tool category logging output

## [2026-06-15 01:02]

- refactor(core): update chat logic and ollama bindings

## [2026-06-12 16:31]

- chore(gitignore): ignore all log files

## [2026-06-12 01:59]

- feat(memory): add debug flag to suppress verbose init logs

## [2026-06-12 01:18]

- fix(chat): robustly detect and prevent duplicate `<processing>` tag emissions

## [2026-06-11 05:16]

- feat(memory): add memories endpoint and management UI

## [2026-06-08 02:25]

- feat(lollms_client): Improve state tracking for markdown code blocks

## [2026-06-08 01:40]

- feat(lollms_client): Integrate LLM bindings, memory management, and tool execution

## [2026-06-05 12:04]

- feat: Add new cognitive decisions test file

## [2026-06-05 09:11]

- feat(memory): Implement Episodic Memory examples and agent framework updates

## [2026-06-04 07:49]

- Refactor: Improve tool execution flow, artifact type classification, and LLM output handling

## [2026-06-04 07:36]

- feat(lollms_client): Integrate document chat functionality and update UI

## [2026-06-04 06:11]

- Refactor path handling in server and improve model loading UX in client

## [2026-06-04 00:34]

- feat(lollms): Introduce mixins for chat and memory functionality

## [2026-06-03 22:30]

- feat(lollms_discussion): Add critical constraints for <coding\_plan> and <artifact> tags

## [2026-06-03 22:29]

- feat(lollms_discussion): Add critical constraint on using <coding_plan> and <artifact> tags for data artifacts

## [2026-06-03 09:09]

- feat(lollmsbot): Update agent examples, client setup, and LLM bindings

## [2026-06-02 07:10]

- feat(agents): enhance agent examples and update core imports

## [2026-06-01 23:21]

- feat: add Lollms Text Processor documentation and update dependencies

## [2026-05-31 23:00]

- feat(artefacts): add rename method and enhanced revert with version aliases

## [2026-05-31 00:35]

- feat: add AI data query endpoint and update client bindings

## [2026-05-31 00:13]

- feat(workspace,ui): add dynamic file serving and enhance tools panel

## [2026-05-29 06:54]

- refactor: remove src/app and add lollms-client-app entry point

## [2026-05-27 19:31]

- fix(diffusers): apply bitsandbytes compatibility patch and guard null device

## [2026-05-27 17:30]

- fix(chat): initialize reasoning_chunks_count in _StreamState

## [2026-05-27 15:27]

- feat: add toast notifications, optional app deps, and update bindings

## [2026-05-26 17:29]

- feat: enhance TTI parameter handling and art display

## [2026-05-22 15:20]

- feat: update structured generation and synchronize core bindings

## [2026-05-21 01:36]

- fix(diffusers): improve server startup reliability and bump version to 1.13.21

## [2026-05-21 01:24]

- chore(deps): bump pipmaster to >=1.1.13 across project files

## [2026-05-18 21:19]

- fix(llama_cpp_server): ensure shared libs are found on Linux/Unix

## [2026-05-18 21:00]

- chore(deps): bump pipmaster to >=1.1.12 and update bindings compatibility

## [2026-05-18 00:25]

- docs: consolidate and clean up discussion documentation

## [2026-05-11 01:37]

- feat(bindings): extend LLM and TTI binding implementations

## [2026-05-07 10:51]

- feat(discussion): add MSG_TYPE_FORM_READY and expand prompt/chat artefact handling

## [2026-05-07 07:43]

- I need to analyze these changes to generate a proper conventional commit message.

## [2026-05-07 06:19]

- fix(artefacts): ensure apply_patch always returns result on success

## [2026-05-06 09:38]

- feat(lollms_discussion): enhance artefacts and chat prompt handling

## [2026-04-26 23:13]

- feat: add OpenAI TTI binding and refactor imports

## [2026-04-23 23:33]

- feat(lollms_client): update version and import bindings utils

## [2026-04-23 01:57]

- feat(lollms_client): Update LLM binding configurations and modernize Python syntax

## [2026-04-21 02:58]

- feat(discussion): add new functionality to chat and core mixins

## [2026-04-12 11:04]

- docs: update discussion docs and refactor swarm artefact handling

## [2026-04-11 23:31]

- refactor(discussion): improve artefact handling and chat mixin structure

## [2026-03-31 01:03]

- docs(chat): update discussion documentation and prompt handling

## [2026-03-30 23:49]

- refactor(_mixin_chat): reduce post-processing logic

## [2026-03-30 01:30]

- chore: fix whitespace and formatting inconsistencies

## [2026-03-26 02:54]

- <type>[optional scope]: <description>

## [2026-03-26 02:11]

- refactor: update discussion mixins for improved chat/core/prompt/utils handling

## [2026-03-24 01:21]

- feat(discussion): add artefacts, chat, and prompt mixins to lollms_discussion

## [2026-03-22 23:07]

- feat(discussion): refactor lollms_discussion module and remove deprecated lollms_agentic

## [2026-03-22 22:47]

- Based on the diff snippets provided, I can see the following changes:

## [2026-03-18 21:06]

- Fix: Move error check before `remove_thinking_blocks` processing

## [2026-03-15 23:28]

- Bump version to 1.12.7

## [2026-03-15 23:26]

- refactor(discussion): improve context extraction and type definitions

## [2026-03-15 02:06]

- chore(deps): bump ascii-colors to >=0.11.21 and update discussion docs

## [2026-03-12 09:41]

- refactor: modularize LLM bindings and discussion mixins

## [2026-03-12 07:14]

- fix: minor adjustments across bindings, artefacts, and personality modules

## [2026-03-11 16:06]

- feat(mcp): restructure MCP bindings architecture with local, remote, and standard variants

## [2026-03-09 02:21]

- chore(deps): bump ascii-colors to >=0.11.20 and update discussion imports

## [2026-03-08 23:35]

- refactor: remove deprecated examples and streamline documentation

## [2026-03-01 23:49]

- chore(release): bump version to 1.11.13

## [2026-03-01 23:04]

- feat(llm): add qwen3.5 model support and improve response handling

## [2026-02-27 13:00]

- fix: correct import statement and JSON continuation in text processing

## [2026-02-26 01:52]

- fix(ssl): correct certificate verification logic in Lollms and OpenAI bindings

## [2026-02-23 02:29]

- chore(version): bump to 1.11.8 and add glm-5 context window

## [2026-02-19 23:14]

- feat: add XTTS server functionality and update binding imports

## [2026-02-16 03:38]

- feat(xtts): enhance XTTS binding with extended configuration options

## [2026-02-11 00:56]

- refactor(lollms_client): clean up unused code and fix imports

## [2026-02-06 01:59]

- docs(readme): uncomment service_key examples to improve API key setup visibility

## [2026-02-04 01:56]

- feat(bindings): add safe module loading for binding descriptions

## [2026-01-26 00:08]

- **Commit Title:**

## [2026-01-25 22:56]

- Enhance RAG reasoning loop with safety limits and richer feedback

## [2026-01-25 22:36]

- **feat: improve RAG query handling and enforce attribution**

## [2026-01-25 22:09]

- **feat: improve discussion handling and text processing utilities**

## [2026-01-24 01:13]

- **fix(lollms_discussion): remove premature save of user message during RL‑mode handling**

## [2026-01-20 08:13]

- chore(version): bump package version to 1.11.1

## [2026-01-18 15:19]

- refactor(lollms): update core and types enums

## [2026-01-13 01:38]

- chore(release): update changelog and bump pipmaster

## [2026-01-13 00:30]

- feat(xtts): add Python 3.10 support and portable env

## [2026-01-12 23:16]

- feat(xtts): add Python 3.10 support and portable env

## [2026-01-12 23:15]

- feat(diffusers): add reinstall command and extra deps

## [2026-01-12 03:03]

- feat(lollms): update LLM/STT/TTI/TTI bindings metadata

## v0.31.0  (2025-08-03)
*   **Bug Fix:** Resolved issue causing discussion level images to be difficult to use by apps.
*   **Refactor:** Reorganized LollmsDiscussion class.
*   **Documentation:** Expanded documentation for `LollmsDiscussion` and `summarize` methods (to fit new updates).


## v0.30.0 (2025-07-19)

*   **Feature:** Introduced advanced memory management with `LollmsDiscussion` allowing for multi-session context and long-term knowledge storage.
*   **Feature:** Implemented `summarize` method for handling texts exceeding model context windows, enabling sequential summarization.
*   **Feature:** Added `toggle_image_activation` for precise control over image usage in multimodal prompts.
*   **Bug Fix:** Resolved issue causing incorrect token counts in `get_context_status`.
*   **Refactor:** Reorganized binding initialization for improved clarity and consistency.
*   **Documentation:** Expanded documentation for `LollmsDiscussion` and `summarize` methods.

## v0.29.0 (2025-07-03)

*   **Feature:** Added support for Anthropic Claude bindings.
*   **Feature:** Introduced `LollmsPersonality` for defining agent roles and knowledge bases.
*   **Refactor:** Improved error handling and reporting across all bindings.
*   **Documentation:** Added comprehensive examples for `LollmsPersonality` and agent creation.

## v0.28.0 (2025-06-03)

*   **Feature:** Added support for OpenRouter API aggregator.
*   **Refactor:** Improved code modularity and reduced dependencies.
*   **Bug Fix:** Resolved issue with incorrect streaming behavior in some bindings.
*   **Documentation:** Updated documentation for streaming callbacks.

## v0.27.0 (2025-05-03)

*   **Feature:** Added support for Google Gemini bindings.
*   **Refactor:** Improved binding configuration system.
*   **Bug Fix:** Resolved issue with long prompt handling in some models.
*   **Documentation:** Added example for using OpenRouter.

## v0.26.0 (2025-04-03)

*   **Feature:** Introduced `get_context_status` method for detailed context analysis.
*   **Refactor:** Improved code structure and readability.
*   **Bug Fix:** Resolved issue with incorrect model name handling.

## v0.25.0 (2025-03-03)

*   **Feature:** Added support for Hugging Face Inference API.
*   **Refactor:** Improved error reporting and logging.
*   **Bug Fix:** Resolved issue with tokenization discrepancies across bindings.

## v0.24.0 (2025-02-03)

*   **Feature:** Introduced `LollmsClient.generate_with_mcp` for function calling and tool use.
*   **Refactor:** Improved code organization and maintainability.
*   **Bug Fix:** Resolved issue with long text processing.

## v0.23.0 (2025-01-03)

*   **Feature:** Added support for Groq bindings.
*   **Refactor:** Improved code structure for better extensibility.
*   **Bug Fix:** Resolved issue with incorrect API key handling.

## v0.22.0 (2024-12-03)

*   **Feature:** Added support for OpenAI bindings.
*   **Refactor:** Improved code organization and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.21.0 (2024-11-03)

*   **Feature:** Added support for PythonLlamaCpp binding.
*   **Refactor:** Improved code modularity and testability.
*   **Bug Fix:** Resolved issue with incorrect token count reporting.

## v0.20.0 (2024-10-03)

*   **Feature:** Added support for Ollama bindings.
*   **Refactor:** Improved code structure and error handling.
*   **Bug Fix:** Resolved issue with streaming behavior.

## v0.19.0 (2024-09-03)

*   **Feature:** Introduced streaming callbacks for real-time response handling.
*   **Refactor:** Improved code organization and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.18.0 (2024-08-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.17.0 (2024-07-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.16.0 (2024-06-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.15.0 (2024-05-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.14.0 (2024-04-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.13.0 (2024-03-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.12.0 (2024-02-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.11.0 (2024-01-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.10.0 (2023-12-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.09.0 (2023-11-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.08.0 (2023-10-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.07.0 (2023-09-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.06.0 (2023-08-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.05.0 (2023-07-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.04.0 (2023-06-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.03.0 (2023-05-03)

*   **Refactor:** Improved code structure and documentation.
*   **Bug Fix:** Resolved issue with long prompt handling.

## v0.02.0 (2023-04-03)

*   **Initial Release:** Basic text generation functionality.
*   Added initial code structure.
```