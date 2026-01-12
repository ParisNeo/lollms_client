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