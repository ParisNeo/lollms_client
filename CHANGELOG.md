```markdown
# CHANGELOG

All notable changes to this project will be documented in this file.
#
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