#!/usr/bin/env python3
"""
lollms_code — Autonomous CLI Coding Agent
==========================================

A production-grade CLI tool that turns a single prompt into a full autonomous
coding session. It uses the LollmsPersonality system and the Handbag architecture
to autonomously:
  1. Analyze the target codebase (workspace context injection)
  2. Plan the implementation strategy
  3. Write code, execute tests, and fix failures iteratively
  4. Create persistent Skills (SKILL.md) from lessons learned
  5. Save episodic memories for cross-session continuity
  6. Delegate sub-tasks to focused child agents when needed
  7. Switch models mid-task for optimal performance
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ascii_colors import ASCIIColors, trace_exception

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ascii_colors import questionary
from ascii_colors.rich.console import Console
from ascii_colors.rich.table import Table

from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_personality.lollms_personality import CapabilityFlags
from lollms_client.lollms_personality.skills_manager import SkillsManager
from lollms_client.lollms_types import EventMode, MSG_TYPE


APP_NAME = "lollms_code"
APP_VERSION = "2.0.0"
APP_CONFIG_DIR = Path.home() / ".lollms_client" / "lollms_code"
APP_CONFIG_FILE = APP_CONFIG_DIR / "config.json"
APP_USER_PROFILE_FILE = Path.home() / ".lollms_client" / "user_profile.md"
APP_DEFAULT_WORKSPACE = Path.cwd()
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"
APP_DEFAULT_HANDSAG_DIR = APP_CONFIG_DIR / "handbags"

def get_workspace_sandbox_dir(workspace_path: str | Path) -> Path:
    return Path(workspace_path).resolve() / ".lollms_code"

def get_workspace_conversation_file(workspace_path: str | Path) -> Path:
    return get_workspace_sandbox_dir(workspace_path) / "conversation_history.json"

def get_workspace_prompt_history_file(workspace_path: str | Path) -> Path:
    return get_workspace_sandbox_dir(workspace_path) / "prompt_history.json"

CODING_EXECUTION_HARNESS = """
=== AUTONOMOUS EXECUTION & SHELL CAPABILITIES ===
You have full access to execute code, run shell commands, and manage workspace files.

## MACRO STEPS PLANNING (CURRENT.md MANDATE)
For every non-trivial task, you MUST maintain a macro-level plan in `.lollms_code/CURRENT.md`.
1. **DEFINE AT TASK START**: Initialize your macro steps plan in `.lollms_code/CURRENT.md` using checkbox markdown:
   ```markdown
   # Current Task: <Task Title>

   ## Macro Steps Plan
   - [ ] Step 1: <First milestone>
   - [ ] Step 2: <Second milestone>
   - [ ] Step 3: <Verification and testing>

   ## Notes & Findings
   - ...
   ```
2. **UPDATE ON MILESTONE COMPLETION**: Every time a macro step is finished (for example, you coded a function, executed it, debugged it, and tested it), update `.lollms_code/CURRENT.md` immediately, marking that step complete (`- [x]`) and recording any findings.
3. **GROUND TRUTH ROADMAP**: `.lollms_code/CURRENT.md` is automatically loaded into your context. Use it so you never lose context or repeat completed work across rounds.

## WORKFLOW & EXECUTION MANDATE
1. **FILE CREATION & EDITING**: Use `<artifact>` tags with complete code or SEARCH/REPLACE blocks.
2. **EXECUTION MANDATE**: When asked to create and execute code:
   - Emit the `<artifact>` tag to create the file.
   - Then execute it using `tool_execute_python_file` or `tool_execute_shell_command`.
   - Never finish with `<done/>` before executing and inspecting the output!
3. **SYSTEM SHELL EXECUTION**: Use `tool_execute_shell_command` to run tests, scripts, or OS commands.
   - On Windows, the shell is `cmd.exe`. Use `del` to delete files (NOT `rm`), `rmdir /s /q` to delete directories (NOT `rm -rf`), `dir` to list, and `type` to view. Never use `rm` on Windows.
   - To check file deletion, use `python -c "import os; print(not os.path.exists('file'))"` or `if not exist file.py (echo DELETED)`. Do not use `dir <deleted_file>` which returns exit code 1.
4. **TERMINATION**: When all objectives are met and verified, summarize your work and end with `<done/>` on a new line.
=== END AUTONOMOUS EXECUTION & SHELL CAPABILITIES ===
"""

TTI_CAPABILITY_PROMPT = """
=== IMAGE GENERATION CAPABILITY (ACTIVE) ===
You have access to a Text-to-Image (TTI) binding. You CAN generate images.
When a user asks you to generate, draw, create, or make an image, you MUST emit the `tool_generate_image` tool call in the VERY SAME RESPONSE:
<tool>{"name": "tool_generate_image", "parameters": {"prompt": "detailed prompt describing image"}}</tool>
Or using XML tag:
<generate_image>detailed English prompt describing the image</generate_image>
NEVER write 'I will generate an image...' and stop without emitting the tool tag!
=== END IMAGE GENERATION CAPABILITY ===
"""

CODING_SYSTEM_PROMPT = """\
You are lollms_code, an elite autonomous software engineering agent.

## YOUR IDENTITY
You are not a chatbot. You are a hands-on engineer that writes, tests, and ships code.
You operate in a fully autonomous loop — no human intervention is required.

## WORKFLOW (MANDATORY)
For every task, follow this structured pipeline:

### Phase 1: RECONNAISSANCE & COGNITIVE ASSIMILATION
- Use `<unlock_file>filename</unlock_file>` to load key files into your context.
- The workspace tree is already visible in your system prompt. Check the [C], [M], [U] markers.
- If the workspace is empty, start fresh.
- If files exist, understand the architecture before modifying anything.
- **COGNITIVE ASSIMILATION (CRITICAL)**: As you read and understand the codebase, you MUST extract high-density architectural facts and save them to persistent memory using `<mem_new>`. 
  - Example: "The database layer uses SQLAlchemy with a repository pattern." -> `<mem_new content="Project uses SQLAlchemy repository pattern for DB access" tags="architecture,database" />`
  - Do NOT save trivial code snippets. Save rules, patterns, and structural facts.

### Phase 2: MACRO STEPS PLANNING (CURRENT.md MANDATE)
- For every non-trivial task, you MUST maintain a macro-level plan in `.lollms_code/CURRENT.md`.
- **CREATE AT TASK START**: Define your macro steps using markdown checkboxes:
  ```markdown
  # Current Task: <Task Title>

  ## Macro Steps Plan
  - [ ] Step 1: <Description of first milestone, e.g. Implement function X>
  - [ ] Step 2: <Description of second milestone, e.g. Execute and debug>
  - [ ] Step 3: <Verification and testing>

  ## Notes & Findings
  - ...
  ```
- **UPDATE ON MILESTONE COMPLETION**: Every time a macro step is finished (e.g., you coded a function, executed it, debugged it, and tested it), you MUST update `.lollms_code/CURRENT.md` immediately, marking that step complete (`- [x]`) and recording any findings.
- **GROUND TRUTH ROADMAP**: `.lollms_code/CURRENT.md` is automatically loaded into your context. Use it so you never lose context or repeat completed work across rounds.

### Phase 3: IMPLEMENTATION
- Use `<artifact>` tags to create or overwrite files.
- For EXISTING files with small changes, use SEARCH/REPLACE blocks inside `<artifact>` tags.
- Write clean, production-quality code with proper error handling.
- Include docstrings and type hints where appropriate.

### Phase 4: TESTING & VERIFICATION (EXECUTION MANDATE)
- When the user asks you to create a file AND execute it (e.g. 'create X and run/execute it'):
  1. Emit the `<artifact>` tag to create the file.
  2. You MUST EXECUTE it in the next action using `tool_execute_python_code`, `tool_execute_python_file`, or `tool_execute_shell_command`.
  3. You are STRICTLY FORBIDDEN from finishing with `<done/>` before executing the file and inspecting the output!
- Use `tool_execute_shell_command` to run tests (e.g., `python -m pytest`).
- Read the test output carefully. If tests fail, FIX THE ROOT CAUSE.
- Do NOT mask errors with try/except — fix the actual bug.
- Re-run tests after each fix until ALL pass.
- After verification, update `.lollms_code/CURRENT.md` marking the step complete (`- [x]`).

### Phase 5: SKILL GENESIS (CRITICAL FOR LEARNING)
- After completing a non-trivial task, you MUST evaluate if your solution contains a reusable methodology.
- **Check Existing Skills**: Before creating a new skill, use `tool_list_skills` to see if a similar skill already exists. If it does, use `tool_update_skill` to refine it with your new experience.
- **Propose or Execute**: If you discovered a new pattern, create a skill using `tool_create_skill`. 
  - Example: If you figured out how to integrate a complex API, create a skill named "api_integration_pattern".
- If the user explicitly asks to "build a skill out of this", you MUST execute the skill creation tool immediately.

### Phase 6: TERMINATION
- When ALL objectives are met and tests pass, write a brief summary:
  - What was created/modified
  - What tests pass
  - Any remaining TODOs or known limitations
- End with `<done/>` on a new line.

## AUTONOMY & SAME-RESPONSE EXECUTION RULES
1. **NEVER ask the user for help unless confirmation is needed for a destructive action.** You are autonomous. Make decisions.
2. **SAME-RESPONSE ACTION EMISSION (CRITICAL)**: Stating in conversational prose (in ANY language: English, Arabic, Chinese, French, etc.) that you will read a file, edit code, run tests, or create a skill DOES NOT execute the action. You MUST emit the corresponding functional tag (`<unlock_file>`, `<artifact>`, `tool_execute_shell_command`, `tool_create_skill`, `<mem_new>`) IN THE VERY SAME RESPONSE immediately after your brief statement of intent.
3. **NEVER SPLIT INTENT AND EXECUTION**: Never output conversational sentences announcing what you are about to do and then stop without emitting the tag. If you do not emit the tag in the same response, the turn will end with nothing done.
4. **DESTRUCTIVE VS CONSTRUCTIVE ACTIONS**:
   - Destructive actions (e.g., `rm -rf`, `git reset --hard`, `git push --force`) require user confirmation before executing.
   - All standard constructive tasks (modifying code, running tests, reading files, creating skills) MUST emit action tags immediately without asking or waiting.
5. **If stuck after 5 attempts on the same bug**, emit `<done/>` with a clear explanation of what failed and what you tried.
6. **If a tool is not available**, adapt and use what you have.
7. **Prefer correctness over speed.** A slow correct solution beats a fast broken one.
8. **GIT WORKFLOW (MANDATORY START)**: If the workspace contains a `.git` directory, your FIRST action in any task MUST be to check git status and create a new branch:
   - Run `git status` to see the current state.
   - Run `git checkout -b task/<short-description>` to create an isolated branch.
   - Only after the branch is created should you start writing artifacts.
9. **STATE PRESERVATION (CRITICAL)**: Before any branch switch or destructive git operation, you MUST preserve your working context and state:
   - **Thoughts**: Use `<scratchpad_append>` to save your current reasoning, plan, and progress.
   - **Uncommitted Changes**: If `git status` shows uncommitted changes, you MUST ask the user for permission to either `git stash` or `git commit` them. NEVER execute `git checkout -b` on a dirty working tree, as this carries changes to the new branch.
   - **Example**: "I need to create a new branch to fix this bug. You have uncommitted changes. Do you want me to `git stash` them (temporary) or `git commit` them (permanent) before I switch branches?"

## CODE QUALITY STANDARDS
- All Python code must be PEP 8 compliant.
- All functions must have docstrings (Google or Sphinx style).
- All public functions must have type hints.
- Error handling: use specific exceptions, not bare `except:`.
- File encoding: always use `encoding='utf-8'` when opening files.
- Never leave debug `print()` statements in production code.
- **WINDOWS CONSOLE ENCODING (CRITICAL)**: When generating Python code that prints to stdout on Windows, you MUST use ASCII-only characters. The Windows console uses `cp1252` encoding by default, which CANNOT encode Unicode characters like `─` (box-drawing), `σ` (sigma), `✅`, or emojis. If you need formatted output, use ASCII alternatives like `---`, `sigma`, `[OK]`, or reconfigure stdout at the top of the script: `import sys; sys.stdout.reconfigure(encoding='utf-8')`. Failure to follow this rule will cause `UnicodeEncodeError` crashes.

## CONTEXT MANAGEMENT & FILE READING (CRITICAL)
- The workspace tree is visible in your system prompt with markers: [C]=loaded, [U]=unlockable, [L]=locked.
- **PRIMARY READING METHOD**: To read ANY file (text, code, PDF, DOCX, PPTX, CSV, etc.), use `<unlock_file>filename</unlock_file>`.
  - The system natively parses PDFs, DOCX, PPTX, and other binary formats into readable text automatically.
  - You DO NOT need to write Python scripts or use shell commands to extract text from documents.
  - Simply emit `<unlock_file>document.pdf</unlock_file>` and the full text content will be injected into your context.
- Use `<lock_file>filename</lock_file>` when done to free context space.
- Do NOT read the same file repeatedly — it stays in your context after unlocking.
- **ANTI-PATTERN WARNING**: If a file disappears from your context (changes from [C] to [U]) after you modified it, this is NORMAL behavior (the system invalidates the cache to prevent stale reads). You MUST recover it by emitting `<unlock_file>`. You are STRICTLY FORBIDDEN from using `tool_execute_shell_command` with `python -c "open(...).read()"`, `type`, or `cat` to inspect file contents. Shell commands are for execution (tests, git), NOT for reading files into your context. Violating this rule is a CRITICAL ERROR.

## SUB-AGENT DELEGATION
- If `tool_spawn_sub_agent` is available and the task has independent sub-components, delegate each to a focused sub-agent.
- Examples: "write the frontend" + "write the backend" → two sub-agents.
- Always provide clear, specific instructions to sub-agents.
- After sub-agents complete, synthesize their outputs into a unified result.

## SKILL SYSTEM USAGE
- Before starting a task, use `tool_list_skills` to check if a relevant skill exists.
- If found, use `tool_load_skill` to get the full content.
- After completing a task, ALWAYS create or update a skill.
- Skills are your long-term memory — they make you better over time.

## PERSISTENT MEMORY SYSTEM (CRITICAL FOR CONTINUITY)
You have access to a persistent memory database that survives across sessions.
1. **STORE FACTS**: When the user shares personal information (name, preferences, project details), you MUST save it immediately:
   <mem_new content="The user's name is Saif" tags="identity,user_profile" level="2" />
2. **UPDATE FACTS**: If information changes, update the memory:
   <mem_update id="memory_id" content="New information" />
3. **AUTOMATIC RECALL**: Relevant memories are automatically injected into your context. You do not need to query them manually.
4. **MANDATORY**: Always use memory tags for non-trivial user facts. If the user tells you their name, you MUST emit `<mem_new>` in your response.
5. **USE MEMORIES**: When asked "do you remember my name?", check the ACTIVE MEMORIES section in your context. If the user's name is there, use it.

## STATE & MEMORY SEGREGATION DOCTRINE (CRITICAL)
You have THREE distinct mechanisms for persisting information. You MUST strictly segregate what goes where.
1. **THE SCRATCHPAD (`<scratchpad_append>` / `<scratchpad_patch>`)**:
   - **Scope**: LOCAL to the current project/workspace.
   - **Usage**: Use for SHORT-TERM, project-specific state. Examples: temporary file paths, intermediate calculation results, active task checklists, or branching strategies specific to this codebase.
   - **Clearing**: Use `<scratchpad_clear></scratchpad_clear>` when the specific task is done to free up context space.
2. **PERSISTENT MEMORY (`<mem_new>` / `<mem_update>`)**:
   - **Scope**: UNIVERSAL. Survives across ALL projects and sessions.
   - **Usage**: Use for LONG-TERM facts, architectural rules, and universal user preferences. Examples: 'The user prefers 4-space indentation', 'Library X requires initialization before use', 'The user's name is Saif'.
   - **Mandatory Action**: If the user states a personal fact or a universal coding standard, you MUST emit `<mem_new>` immediately.
3. **USER PROFILE (`<user_profile_update>`)**:
   - Used exclusively for the user's identity and universal interaction preferences.

## SANDBOX & WORKSPACE ISOLATION (CRITICAL)
You are operating inside the project workspace at `./` (which resolves to the project root).
1. **PROJECT FILES**: You have full access to read, modify, and create files in the workspace.
2. **TRANSIENT SCRIPTS**: All test scripts, temporary files, and experimental code MUST be written to the `.lollms_code/scripts/` subdirectory. This directory is automatically cleaned on every restart.
3. **PERSISTENT NOTES**: A `.lollms_code/scratchpad.md` file exists. Use it to store long-term context, architectural decisions, or task state. This file survives restarts.
4. **NO WORKSPACE BLOAT**: Do not leave temporary files in the root project directory. Use the `.lollms_code/` folder for all non-essential outputs.

## SYSTEM SHELL EXECUTION (SECONDARY METHOD)
You have access to the `tool_execute_shell_command` tool. This is used for running commands, tests, and environment management.
**IMPORTANT**: Do NOT use shell commands (`type`, `cat`) to read files for context. Use `<unlock_file>` instead. Shell commands are for execution, not reading.

### WORKFLOW RULES
1. **FILE CREATION**: To create or overwrite files, use `<artifact>` tags.
2. **CODE EXECUTION**: To execute Python code, use `python scripts/script.py` or `python -c "import math; print(math.pi)"`.
3. **PACKAGE MANAGEMENT**: If a package is missing, use `pip install package_name`.
4. **TESTING**: Run tests using `python -m pytest` or `python -m unittest`.
5. **WINDOWS COMMAND PROMPT (cmd.exe)**: When running on Windows, the shell is `cmd.exe`. Use `del` to delete files (NOT `rm`), `rmdir /s /q` to delete directories (NOT `rm -rf`), `dir` to list files, and `type` to view files. Never use `rm` on Windows. To check if a file is deleted without tripping exit-code errors, use `python -c "import os; print(not os.path.exists('file'))"` or `if not exist file.py (echo DELETED)` (do not use `dir <deleted_file>` which returns exit code 1).

### GIT OPERATIONS (HIGH-EFFICIENCY PROTOCOL)
When asked to "commit", "push", or perform any git operation, you MUST follow this 2-round protocol:
- **Round 1**: Run `git diff` (or `git diff --stat` for large changes) to inspect what changed. DO NOT unlock or load any files into context.
- **Round 2**: Run `git add -A && git commit -m "message"` with a meaningful message based on the diff. Then emit `<done/>`.
You are STRICTLY FORBIDDEN from using `<unlock_file>` before a git commit. The diff output is sufficient to write a commit message.

### SAFETY
- The host application controls the autonomy level of the shell tool.
- If a command is blocked because it requires elevated privileges, inform the user that they need to adjust the `system_shell` configuration in the host application settings.
"""


# ── CLI INTERACTIVE HELP MANUAL ─────────────────────────────────────────────
HELP_SECTIONS = {
    "1": {
        "title": "🚀 1. Quick Start & Core Concepts",
        "content": """\
[cyan]Welcome to lollms_code![/cyan]

lollms_code is an autonomous coding agent that operates directly in your terminal.
Unlike standard LLM wrappers, it executes a [bold]Plan -> Code -> Test -> Fix[/bold] loop automatically.

[bold yellow]Basic Usage:[/bold yellow]
  [green]lollms-code "Implement a user authentication system using JWT"[/green]
  [green]lollms-code -i[/green]  # Starts interactive REPL mode

[bold yellow]Core Concepts:[/bold yellow]
  1. [cyan]Workspace[/cyan]: The agent reads your local directory tree automatically.
  2. [cyan]Context Tree[/cyan]: Files are marked [C] (Loaded), [M] (Metadata), [U] (Unlockable).
  3. [cyan]Autonomy[/cyan]: It uses shell tools to run commands, tests, and fix bugs iteratively until it succeeds.
  4. [cyan]Memory[/cyan]: It remembers facts across sessions (e.g., your name, preferences).
  5. [cyan]Skills[/cyan]: It saves reusable coding patterns as SKILL.md files for future use.
"""
    },
    "2": {
        "title": "💻 2. Coding & Refactoring",
        "content": """\
[bold magenta]Use Case: Writing new features, fixing bugs, or refactoring.[/bold magenta]

The agent is designed to write production-ready code. It uses an [bold]Aider-style SEARCH/REPLACE protocol[/bold] for surgical edits to existing files.

[bold yellow]Examples:[/bold yellow]
  - "Add pagination to the `UserList` component in `src/components.py`. Use a limit of 20."
  - "Find the memory leak in `image_processor.py` and fix it."
  - "Write unit tests for all functions in `utils/math.py` using pytest. Run the tests and fix any failures."

[bold green]Pro-Tip:[/bold green]
If the task is large, ask it to plan first:
  "Plan the architecture for a REST API for a blog. Then implement the models and database connection."
The agent will outline the plan, then create the files using `<artifact>` tags.
"""
    },
    "3": {
        "title": "📚 3. Documentation & Content Organization",
        "content": """\
[bold magenta]Use Case: Generating docs, READMEs, or organizing markdown content.[/bold magenta]

The agent can read your entire codebase and extract structural information to write accurate documentation.

[bold yellow]Examples:[/bold yellow]
  - "Read all Python files in the `src/` directory and generate a comprehensive `README.md` with architecture diagrams."
  - "Add Google-style docstrings to all public classes and functions in `main.py`."
  - "Scan the `data/` folder, analyze the CSV headers, and create a `schema.md` file documenting the data structures."

[bold green]Pro-Tip:[/bold green]
For massive codebases, unlock specific files first in your prompt:
  "Read `src/api/router.py` and `src/api/auth.py`, then write an API reference document."
"""
    },
    "4": {
        "title": "🧠 4. Autonomous Learning & Skills",
        "content": """\
[bold magenta]Use Case: Building a persistent knowledge base of coding patterns.[/bold magenta]

lollms_code automatically creates [cyan]Skills[/cyan] (SKILL.md files) when it solves a non-trivial problem. In future sessions, it loads these skills to solve similar problems instantly.

[bold yellow]Examples:[/bold yellow]
  - "Figure out how to integrate Stripe payment webhooks into this Flask app."
  (After succeeding, the agent saves a "stripe_integration" skill).
  
  - "What skills do you know?"
  (In interactive mode, type `skills` to list all learned skills).

[bold green]Pro-Tip:[/bold green]
You can explicitly ask the agent to create a skill:
  "Create a skill named 'git_conflict_resolution' documenting the best way to resolve complex merge conflicts."
"""
    },
    "5": {
        "title": "🤖 5. Sub-Agent Delegation (Complex Tasks)",
        "content": """\
[bold magenta]Use Case: Breaking down massive tasks into parallel work streams.[/bold magenta]

If enabled, the agent can spawn "child" agents to handle independent parts of a task simultaneously.

[bold yellow]Examples:[/bold yellow]
  - "Build a full-stack weather app. Delegate the frontend (HTML/JS) to one sub-agent, and the backend (Python API) to another. Then integrate them."
  - "Translate the UI into Spanish, French, and German simultaneously using sub-agents."

[bold green]Pro-Tip:[/bold green]
Sub-agents share the same workspace but have isolated context windows. They are perfect for heavy, independent operations like data processing or file generation.
"""
    },
    "6": {
        "title": "⚙️ 6. Configuration & Memory",
        "content": """\
[bold magenta]Use Case: Customizing behavior and ensuring cross-session continuity.[/bold magenta]

[bold yellow]Interactive Commands:[/bold yellow]
  - [cyan]config[/cyan]: Runs the Lollms configuration wizard (changes models, bindings).
  - [cyan]forget[/cyan]: Permanently wipes the agent's associative memory (use with caution!).
  - [cyan]skills[/cyan]: Lists all stored skills in your `~/.lollms_client/lollms_code/skills/` directory.
  - [cyan]workspace[/cyan]: Switches the active workspace to another directory.

[bold yellow]Memory Management:[/bold yellow]
The agent remembers facts about you. If you say "My name is Alex", it will save it.
Next session, you can ask "What is my name?" and it will know.

[bold green]Pro-Tip:[/bold green]
Use `--workspace ./path/to/project` to target a specific directory without changing your current terminal path.
"""
    },
    "7": {
        "title": "📂 7. Manual Context Management",
        "content": """\
[bold magenta]Use Case: Manually managing files and conversation state to optimize the agent's context window.[/bold magenta]

You can manually control the agent's context using these slash commands. 
This is highly recommended for large workspaces to save context tokens.

[bold yellow]Context Clearing Commands:[/bold yellow]
  - [cyan]/clear-history[/cyan]: Wipes the conversation history from the agent's memory.
  - [cyan]/clear-files[/cyan]: Unloads ALL currently loaded files from context (frees up maximum space).

[bold yellow]File Visibility Commands:[/bold yellow]
  - [cyan]/load <file1> [file2] ...[/cyan]: Manually loads files into the [C] (Fully Loaded) context.
    Example: `/load src/main.py src/utils.py`
    You can also use `all` to load all indexed files: `/load all`
  - [cyan]/unload <file1> ...[/cyan]: Removes specific files from context (changes [C] to [U]).
  - [cyan]/lock <file1> ...[/cyan]: Locks files in the tree (changes to [L], cannot be unlocked by agent).
  - [cyan]/hide <file1> ...[/cyan]: Completely hides files from the workspace tree.
  - [cyan]/unhide <file1> ...[/cyan]: Restores hidden files to the tree.
  - [cyan]/files[/cyan]: Lists all files currently loaded in the context [C].

[bold green]Pro-Tip:[/bold green]
If the agent is running out of context space, manually `/load` only the files relevant to your current task.
"""
    },
    "q": {
        "title": "Exit Help",
        "content": "Returning to the agent..."
    }
}

def show_interactive_help():
    """Displays the interactive, multi-page help manual."""
    current_page = "1"
    while True:
        section = HELP_SECTIONS[current_page]
        ASCIIColors.rule(f"[bold blue]{section['title']}[/bold blue]")
        ASCIIColors.rich_print(section["content"])
        
        ASCIIColors.rich_print("\n[bold]Navigation:[/bold]")
        ASCIIColors.rich_print("  [cyan]1-7[/cyan] - Jump to a specific section")
        ASCIIColors.rich_print("  [cyan]q[/cyan]   - Quit help and return to the agent")
        
        try:
            choice = input("\n  Choice> ").strip().lower()
            if choice in HELP_SECTIONS:
                current_page = choice
                if choice == "q":
                    break
            else:
                ASCIIColors.yellow("  Invalid choice. Please enter 1-6 or q.")
        except (EOFError, KeyboardInterrupt):
            break

class PersistentHistory:
    """Manages a persistent JSON-backed history of prompts for the REPL."""
    
    def __init__(self, history_file: Path, max_entries: int = 100):
        self.history_file = history_file
        self.max_entries = max_entries
        self.entries: list[str] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.entries = [str(x) for x in data if isinstance(x, (str, int, float))]
            except (OSError, json.JSONDecodeError) as e:
                ASCIIColors.warning(f"Failed to load history: {e}")
                self.entries = []

    def _save(self):
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(
                json.dumps(self.entries, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
        except OSError as e:
            ASCIIColors.warning(f"Failed to save history: {e}")

    def add(self, prompt: str):
        prompt = prompt.strip()
        if not prompt:
            return
        if self.entries and self.entries[-1] == prompt:
            return
        self.entries.append(prompt)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._save()


class CodeAgentConfig:
    """
    Configuration for lollms_code supporting the Universal Two-Tier Profile Architecture:
    - Connection Layer (*_binding_profiles): Server and backend engine connections.
    - Execution Layer (*_model_profiles): Specific models, routing, and context limits.
    """
    def __init__(self):
        # Two-Tier Profile Registries (Universal Modalities)
        self.llm_binding_profiles: dict[str, dict[str, Any]] = {}
        self.llm_model_profiles: dict[str, dict[str, Any]] = {}
        self.tti_binding_profiles: dict[str, dict[str, Any]] = {}
        self.tti_model_profiles: dict[str, dict[str, Any]] = {}
        self.tts_binding_profiles: dict[str, dict[str, Any]] = {}
        self.tts_model_profiles: dict[str, dict[str, Any]] = {}
        self.stt_binding_profiles: dict[str, dict[str, Any]] = {}
        self.stt_model_profiles: dict[str, dict[str, Any]] = {}
        self.ttv_binding_profiles: dict[str, dict[str, Any]] = {}
        self.ttv_model_profiles: dict[str, dict[str, Any]] = {}
        self.ttm_binding_profiles: dict[str, dict[str, Any]] = {}
        self.ttm_model_profiles: dict[str, dict[str, Any]] = {}

        self.active_profile: str | None = None
        self.wizard_completed: bool = False
        self.max_reasoning_steps: int = 100
        self.temperature: float = 0.3
        self.max_tokens_per_turn: int = 8192
        self.enable_shell_execution: bool = True
        self.shell_autonomy_level: str = "safe"
        self.enable_sub_agents: bool = True
        self.enable_model_switching: bool = False
        self.enable_skill_creation: bool = True
        self.enable_skill_loading: bool = True
        self.enable_memory: bool = True
        self.skills_mode: str = "mixed"
        self.max_sub_agent_depth: int = 2
        self.max_sub_agents_per_turn: int = 3
        self.workspace_path: str = str(Path.cwd().resolve())
        self.skills_dir: str = str(APP_DEFAULT_SKILLS_DIR)
        self.memory_db: str = f"sqlite:///{APP_DEFAULT_MEMORY_DB}"
        self.handbag_path: str = str(APP_DEFAULT_HANDSAG_DIR / "default_coder")
        self.show_tool_calls: bool = True
        self.show_workspace_changes: bool = True
        self.show_skills: bool = True
        self.show_progress: bool = True
        self.debug: bool = False

    @property
    def active_profile_alias(self) -> str:
        if self.active_profile and self.active_profile in self.llm_model_profiles:
            return self.active_profile
        for alias, prof in self.llm_model_profiles.items():
            if prof.get("is_default"):
                return alias
        if self.llm_model_profiles:
            return next(iter(self.llm_model_profiles))
        return "default"

    @property
    def active_model_name(self) -> str:
        alias = self.active_profile_alias
        prof = self.llm_model_profiles.get(alias, {})
        return prof.get("model_name") or "default"

    @property
    def active_binding_name(self) -> str:
        alias = self.active_profile_alias
        prof = self.llm_model_profiles.get(alias, {})
        b_alias = prof.get("binding_profile_name") or prof.get("binding_alias")
        b_data = self.llm_binding_profiles.get(b_alias, {}) if b_alias else {}
        return b_data.get("binding_name") or prof.get("binding_name") or "ollama"

    @classmethod
    def load(cls, cli_args: argparse.Namespace) -> "CodeAgentConfig":
        config = cls()
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        PROFILE_KEYS = (
            "llm_binding_profiles", "llm_model_profiles",
            "tti_binding_profiles", "tti_model_profiles",
            "tts_binding_profiles", "tts_model_profiles",
            "stt_binding_profiles", "stt_model_profiles",
            "ttv_binding_profiles", "ttv_model_profiles",
            "ttm_binding_profiles", "ttm_model_profiles",
            "active_profile",
        )

        # 1. Load saved preferences from JSON (preferences only — never profiles)
        if APP_CONFIG_FILE.exists():
            try:
                file_config = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
                stale_profile_keys = [k for k in file_config if k in PROFILE_KEYS and k != "active_profile"]
                if stale_profile_keys:
                    ASCIIColors.warning(
                        "[CLI] Ignoring stale binding/model profiles found in config.json. "
                        "Profiles are resolved from ~/.lollms_client/config.yaml (or --config path)."
                    )
                for key, val in file_config.items():
                    if key in PROFILE_KEYS:
                        continue
                    if hasattr(config, key):
                        setattr(config, key, val)
            except (OSError, json.JSONDecodeError) as e:
                ASCIIColors.warning(f"Failed to read config file: {e}")

        # 2. Extract profiles from environment & config files
        from lollms_client.lollms_config_cli_env import (
            load_env_file,
            load_yaml_file,
            _flatten_dict_to_env,
            _extract_bindings_from_env,
            _extract_profiles_from_env
        )

        resolved_env = dict(os.environ)

        explicit_config = getattr(cli_args, "config_path", None)
        config_sources: list[Path] = []

        if explicit_config:
            p = Path(explicit_config).expanduser()
            if not p.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {p}. "
                    "Use --config with an existing .env, .json, or .yaml file."
                )
            config_sources.append(p)
        else:
            home_dir = Path.home() / ".lollms_client"
            for source in (
                home_dir / ".env",
                Path.cwd() / ".lollms_code" / ".env",
                home_dir / "config.yaml",
                Path.cwd() / ".lollms_code" / "config.yaml",
            ):
                if source.exists():
                    config_sources.append(source)

        for source in config_sources:
            try:
                if source.suffix == ".env":
                    resolved_env.update(load_env_file(source))
                elif source.suffix == ".json":
                    from lollms_client.lollms_config_cli_env import _descend_into_entry, load_json_file
                    data = _descend_into_entry(load_json_file(source), None)
                    resolved_env.update(_flatten_dict_to_env(data))
                elif source.suffix in (".yaml", ".yml"):
                    resolved_env.update(_flatten_dict_to_env(load_yaml_file(source)))
            except (OSError, ValueError) as e:
                ASCIIColors.warning(f"Failed to parse configuration source {source}: {e}")

        # Extract profiles across all modalities
        _ssl_debug = os.getenv("LOLLMS_DEBUG_SSL", "").lower() in ("1", "true", "yes")
        for modality in ("llm", "tti", "tts", "stt", "ttv", "ttm"):
            prefix = modality.upper()
            bindings = _extract_bindings_from_env(prefix, resolved_env)
            profiles = _extract_profiles_from_env(prefix, bindings, resolved_env)
            if _ssl_debug and modality == "llm":
                ssl_keys = {k: v for k, v in resolved_env.items() if "VERIFY_SSL" in k.upper() or (k.upper().startswith("LLM_BINDINGS_") and k.upper().endswith("_HOST_ADDRESS"))}
                ASCIIColors.yellow(f"[Config.load][SSL-DEBUG] raw LLM SSL/host keys in resolved_env: {ssl_keys}")
                ASCIIColors.yellow(f"[Config.load][SSL-DEBUG] extracted llm_binding_profiles: {bindings}")
                ASCIIColors.yellow(f"[Config.load][SSL-DEBUG] extracted llm_model_profiles: {profiles}")
            setattr(config, f"{modality}_binding_profiles", bindings)
            setattr(config, f"{modality}_model_profiles", profiles)

        # 3. Handle CLI argument overrides
        if getattr(cli_args, "profile", None):
            config.active_profile = cli_args.profile.strip()
            # Flag selected profile as default
            for p_name, p_data in config.llm_model_profiles.items():
                p_data["is_default"] = (p_name == config.active_profile)

        # Apply specific CLI overrides to the active profile
        if cli_args.llm_binding or cli_args.model or cli_args.host or cli_args.api_key or cli_args.context_size:
            # Locate active binding & model profile to update or create
            curr_alias = config.active_profile_alias
            if curr_alias not in config.llm_model_profiles:
                config.llm_binding_profiles["default"] = {
                    "binding_name": cli_args.llm_binding or "ollama",
                    "binding_config": {
                        "host_address": cli_args.host or "http://localhost:11434",
                        "service_key": cli_args.api_key or "",
                        "verify_ssl_certificate": False,
                    },
                    "is_default": True
                }
                config.llm_model_profiles["default"] = {
                    "binding_profile_name": "default",
                    "model_name": cli_args.model or "qwen3:32b",
                    "is_default": True,
                    "forced_context_size": cli_args.context_size or 8192
                }
            else:
                m_prof = config.llm_model_profiles[curr_alias]
                b_alias = m_prof.get("binding_profile_name") or m_prof.get("binding_alias") or "default"
                if b_alias not in config.llm_binding_profiles:
                    config.llm_binding_profiles[b_alias] = {
                        "binding_name": cli_args.llm_binding or "ollama",
                        "binding_config": {},
                        "is_default": True
                    }
                b_prof = config.llm_binding_profiles[b_alias]

                if cli_args.llm_binding:
                    b_prof["binding_name"] = cli_args.llm_binding
                if cli_args.model:
                    m_prof["model_name"] = cli_args.model
                if cli_args.host:
                    b_prof.setdefault("binding_config", {})["host_address"] = cli_args.host
                if cli_args.api_key:
                    b_prof.setdefault("binding_config", {})["service_key"] = cli_args.api_key
                if cli_args.context_size:
                    m_prof["forced_context_size"] = cli_args.context_size

        # Fallback: only scaffold what is genuinely missing, never overwrite extracted data
        if os.getenv("LLM_CONFIG_DEBUG", "").lower() in ("1", "true", "yes"):
            ASCIIColors.yellow(f"[Config.load][DEBUG] config sources: {[str(s) for s in config_sources]}")
            for modality in ("llm", "tti", "tts", "stt", "ttv", "ttm"):
                ASCIIColors.yellow(
                    f"[Config.load][DEBUG] {modality}: bindings={list(getattr(config, f'{modality}_binding_profiles').keys())} "
                    f"models={list(getattr(config, f'{modality}_model_profiles').keys())}"
                )

        if not config.llm_binding_profiles:
            ASCIIColors.warning(
                "[Config.load] No LLM bindings resolved from configuration files. "
                "Scaffolding fallback 'ollama' binding — check ~/.lollms_client/config.yaml."
            )
            config.llm_binding_profiles["default"] = {
                "binding_name": "ollama",
                "binding_config": {"host_address": "http://localhost:11434"},
                "is_default": True
            }
        if not config.llm_model_profiles:
            ASCIIColors.warning(
                "[Config.load] No LLM model profiles resolved from configuration files. "
                "Scaffolding fallback profile — check ~/.lollms_client/config.yaml."
            )
            fallback_binding_alias = next(iter(config.llm_binding_profiles))
            config.llm_model_profiles["default"] = {
                "binding_profile_name": fallback_binding_alias,
                "binding_alias": fallback_binding_alias,
                "is_default": True,
                "forced_context_size": 8192
            }

        if cli_args.workspace:
            config.workspace_path = str(Path(cli_args.workspace).resolve())
        else:
            config.workspace_path = str(Path.cwd().resolve())

        if cli_args.max_steps:
            config.max_reasoning_steps = cli_args.max_steps
        if cli_args.temperature is not None:
            config.temperature = cli_args.temperature
        if cli_args.max_tokens:
            config.max_tokens_per_turn = cli_args.max_tokens
        if cli_args.debug is not None:
            config.debug = cli_args.debug
        else:
            config.debug = False
        if cli_args.enable_model_switching:
            config.enable_model_switching = True
        if cli_args.no_shell_execution:
            config.enable_shell_execution = False
        if cli_args.shell_autonomy:
            config.shell_autonomy_level = cli_args.shell_autonomy
        if cli_args.no_sub_agents:
            config.enable_sub_agents = False
        if cli_args.no_memory:
            config.enable_memory = False
        if cli_args.skills_dir:
            config.skills_dir = str(Path(cli_args.skills_dir).resolve())
        if cli_args.handbag_path:
            config.handbag_path = str(Path(cli_args.handbag_path).resolve())

        return config

    def save(self):
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_KEYS = (
            "llm_binding_profiles", "llm_model_profiles",
            "tti_binding_profiles", "tti_model_profiles",
            "tts_binding_profiles", "tts_model_profiles",
            "stt_binding_profiles", "stt_model_profiles",
            "ttv_binding_profiles", "ttv_model_profiles",
            "ttm_binding_profiles", "ttm_model_profiles",
        )
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in PROFILE_KEYS
            and k not in ("workspace_path", "wizard_completed")
        }
        try:
            APP_CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            ASCIIColors.warning(f"Failed to save config: {e}")


    def _resolve_env_map(self, config_path: str | None = None) -> dict[str, str]:
        """Reads ALL universal configuration sources into one flattened env map.

        Merge order (later sources override earlier ones):
        1. ~/.lollms-client/.env   (GUI wizard)
        2. ~/.lollms_client/.env   (CLI wizard)
        3. ~/.lollms_client/config.yaml (CLI wizard, structured format)
        """
        from lollms_client.lollms_config_cli_env import (
            load_env_file,
            load_yaml_file,
            _flatten_dict_to_env,
        )

        env_data: dict[str, str] = {}
        if config_path:
            sources = [Path(config_path).expanduser()]
        else:
            sources = [
                Path.home() / ".lollms_client" / ".env",
                Path.cwd() / ".lollms_code" / ".env",
                Path.home() / ".lollms_client" / "config.yaml",
                Path.cwd() / ".lollms_code" / "config.yaml",
            ]

        for source in sources:
            if not source.exists():
                continue
            try:
                if source.suffix in (".yaml", ".yml"):
                    env_data.update(_flatten_dict_to_env(load_yaml_file(source)))
                else:
                    env_data.update(load_env_file(source))
            except (OSError, ValueError) as e:
                ASCIIColors.warning(f"Failed to read configuration source {source}: {e}")
        return env_data

    def _has_modality_configured(self, env_data: dict[str, str], modality: str) -> bool:
        """Checks if at least one binding and one profile exist for the given modality (e.g., 'llm', 'tti')."""
        mod_upper = modality.upper()
        has_binding = any(k.upper().startswith(f"{mod_upper}_BINDINGS_") and k.upper().endswith("_BINDING_NAME") and v for k, v in env_data.items())
        has_profile = any(k.upper().startswith(f"{mod_upper}_PROFILES_") and k.upper().endswith("_BINDING_ALIAS") and v for k, v in env_data.items())
        return has_binding and has_profile

    def is_configured(self, require_llm: bool = True, require_tti: bool = False, require_tts: bool = False, require_stt: bool = False, require_ttm: bool = False, require_ttv: bool = False) -> bool:
        """Validates configuration based on required modalities using the Two-Tier Profile System."""
        env_data = self._resolve_env_map()

        required_modalities = {
            "llm": require_llm,
            "tti": require_tti,
            "tts": require_tts,
            "stt": require_stt,
            "ttm": require_ttm,
            "ttv": require_ttv
        }

        for modality, required in required_modalities.items():
            if required and not self._has_modality_configured(env_data, modality):
                return False

        return True
    

def _resolve_modality_from_env(modality: str) -> dict[str, Any] | None:
    """
    Resolves a modality (tti, tts, stt, etc.) binding+profile from:
    1. ~/.lollms-client/.env
    2. ~/.lollms_client/.env
    3. ~/.lollms_client/config.yaml
    Accepts keys configured under either the Binding or Profile layer.
    """
    prefix = modality.upper()

    config_obj = CodeAgentConfig()
    env_map: dict[str, str] = config_obj._resolve_env_map()

    default_alias = None
    for k, v in env_map.items():
        if k.startswith(f"{prefix}_PROFILES_") and k.endswith("_IS_DEFAULT") and v.lower() in ("true", "1", "yes"):
            default_alias = k[len(f"{prefix}_PROFILES_"):-len("_IS_DEFAULT")]
            break

    if not default_alias:
        profile_aliases = [k[len(f"{prefix}_PROFILES_"):-len("_BINDING_ALIAS")] for k in env_map if k.startswith(f"{prefix}_PROFILES_") and k.endswith("_BINDING_ALIAS")]
        if profile_aliases:
            default_alias = profile_aliases[0]

    if default_alias:
        binding_alias = env_map.get(f"{prefix}_PROFILES_{default_alias}_BINDING_ALIAS", default_alias)
        binding_name = env_map.get(f"{prefix}_BINDINGS_{binding_alias}_BINDING_NAME") or env_map.get(f"{prefix}_PROFILES_{default_alias}_BINDING_NAME")
        if binding_name:
            service_key = (
                env_map.get(f"{prefix}_BINDINGS_{binding_alias}_SERVICE_KEY")
                or env_map.get(f"{prefix}_BINDINGS_{binding_alias}_API_KEY")
                or env_map.get(f"{prefix}_PROFILES_{default_alias}_SERVICE_KEY")
                or env_map.get(f"{prefix}_PROFILES_{default_alias}_API_KEY")
                or env_map.get(f"{prefix}_SERVICE_KEY")
                or env_map.get(f"{prefix}_API_KEY")
                or ""
            )

            host_addr = (
                env_map.get(f"{prefix}_BINDINGS_{binding_alias}_HOST_ADDRESS")
                or env_map.get(f"{prefix}_PROFILES_{default_alias}_HOST_ADDRESS")
                or env_map.get(f"{prefix}_HOST_ADDRESS")
                or "http://localhost:9642"
            )

            return {
                "binding_name": binding_name,
                "model_name": env_map.get(f"{prefix}_PROFILES_{default_alias}_MODEL_NAME", ""),
                "host_address": host_addr,
                "api_key": service_key,
                "verify_ssl": env_map.get(f"{prefix}_BINDINGS_{binding_alias}_VERIFY_SSL_CERTIFICATE", "false").lower() in ("true", "1", "yes"),
            }

    return None


def _resolve_modality_from_config(config: CodeAgentConfig, modality: str) -> dict[str, Any] | None:
    """
    Resolves a modality (tti, tts, stt, etc.) binding+profile from the already-loaded
    CodeAgentConfig two-tier profile registries.
    Accepts keys configured under either the Binding or Profile layer.
    """
    binding_profiles = getattr(config, f"{modality}_binding_profiles", {})
    model_profiles = getattr(config, f"{modality}_model_profiles", {})

    if not binding_profiles and not model_profiles:
        return None

    default_model_alias = None
    for alias, prof in model_profiles.items():
        if prof.get("is_default"):
            default_model_alias = alias
            break

    if not default_model_alias:
        default_model_alias = next(iter(model_profiles), None)

    model_profile = model_profiles.get(default_model_alias, {}) if default_model_alias else {}
    binding_alias = model_profile.get("binding_profile_name") or model_profile.get("binding_alias") or default_model_alias or "master"

    binding_profile = binding_profiles.get(binding_alias, {})
    binding_name = binding_profile.get("binding_name") or model_profile.get("binding_name")
    if not binding_name:
        return None

    b_cfg = binding_profile.get("binding_config", {})
    m_cfg = model_profile.get("binding_config", {})

    service_key = (
        b_cfg.get("service_key")
        or b_cfg.get("api_key")
        or m_cfg.get("service_key")
        or m_cfg.get("api_key")
        or model_profile.get("service_key")
        or model_profile.get("api_key")
        or ""
    )

    host_address = (
        b_cfg.get("host_address")
        or m_cfg.get("host_address")
        or model_profile.get("host_address")
        or "http://localhost:9642"
    )

    return {
        "binding_name": binding_name,
        "model_name": model_profile.get("model_name", ""),
        "host_address": host_address,
        "api_key": service_key,
        "verify_ssl": b_cfg.get("verify_ssl_certificate", False) or m_cfg.get("verify_ssl_certificate", False),
    }


def create_client(config: CodeAgentConfig) -> LollmsClient:
    """Creates a LollmsClient instance from the CodeAgentConfig using the Two-Tier Profile System."""

    active_model_alias = config.active_profile_alias
    active_model_profile = config.llm_model_profiles.get(active_model_alias, {})

    binding_alias = active_model_profile.get("binding_profile_name") or active_model_profile.get("binding_alias")
    if not binding_alias or binding_alias not in config.llm_binding_profiles:
        for prof in config.llm_model_profiles.values():
            if prof.get("is_default"):
                binding_alias = prof.get("binding_profile_name") or prof.get("binding_alias")
                break

    if not binding_alias or binding_alias not in config.llm_binding_profiles:
        if config.llm_binding_profiles:
            binding_alias = next(iter(config.llm_binding_profiles))

    binding_profile = config.llm_binding_profiles.get(binding_alias, {}) if binding_alias else {}
    binding_config = binding_profile.get("binding_config", {}).copy()

    active_model_name = active_model_profile.get("model_name") or config.active_model_name
    active_binding_name = binding_profile.get("binding_name") or config.active_binding_name

    llm_config: dict[str, Any] = {
        "model_name": active_model_name,
        "host_address": binding_config.get("host_address", "http://localhost:11434"),
        "verify_ssl_certificate": binding_config.get("verify_ssl_certificate", False),
    }

    if binding_config.get("service_key"):
        llm_config["service_key"] = binding_config["service_key"]

    if active_binding_name == "llama_cpp_server":
        if active_model_profile.get("forced_context_size"):
            llm_config["ctx_size"] = active_model_profile["forced_context_size"]
        if binding_config.get("n_gpu_layers"):
            llm_config["n_gpu_layers"] = binding_config["n_gpu_layers"]
        if binding_config.get("models_path"):
            llm_config["models_path"] = binding_config["models_path"]
        if binding_config.get("binaries_path"):
            llm_config["binaries_path"] = binding_config["binaries_path"]

    import lollms_client
    package_root = Path(lollms_client.__file__).resolve().parent
    default_tools_path = package_root / "tools_bindings" / "lcp" / "default_tools"

    tools_folders = [str(default_tools_path)] if default_tools_path.exists() else []

    host_tool_configs = {
        "system_shell": {
            "autonomy_level": config.shell_autonomy_level
        }
    }

    client_kwargs = {
        "tools_binding_name": "lcp",
        "tools_binding_config": {
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs
        },
    }

    if config.llm_binding_profiles and config.llm_model_profiles:
        client_kwargs["llm_binding_profiles"] = config.llm_binding_profiles
        client_kwargs["llm_model_profiles"] = config.llm_model_profiles
    else:
        client_kwargs["llm_binding_name"] = active_binding_name
        client_kwargs["llm_binding_config"] = llm_config

    for modality in ("tti", "tts", "stt"):
        try:
            resolved = _resolve_modality_from_config(config, modality)
            if not resolved:
                resolved = _resolve_modality_from_env(modality)

            if resolved:
                modality_config: dict[str, Any] = {
                    "host_address": resolved["host_address"],
                    "model_name": resolved["model_name"],
                    "verify_ssl_certificate": resolved["verify_ssl"],
                }
                if resolved["api_key"]:
                    modality_config["service_key"] = resolved["api_key"]
                client_kwargs[f"{modality}_binding_name"] = resolved["binding_name"]
                client_kwargs[f"{modality}_binding_config"] = modality_config
                modality_label = {"tti": "Image Generation", "tts": "Speech Synthesis", "stt": "Transcription"}[modality]
                ASCIIColors.success(f"[CLI] ✅ {modality.upper()} Binding '{resolved['binding_name']}' mounted for {modality_label}.")
            elif modality == "tti":
                ASCIIColors.info("[CLI] No TTI binding configured. Image generation tools will not be available.")
        except (KeyError, ValueError, OSError) as e:
            ASCIIColors.warning(f"[CLI] Failed to configure {modality.upper()} binding: {e}")

    client = LollmsClient(**client_kwargs)

    # ⚡ Enable fast token estimation (heuristic, no server round-trips)
    client.enable_fast_token_estimate()
    if client.use_fast_token_estimate:
        ASCIIColors.success(
            "[CLI] ⚡ Fast token estimation active "
            f"(coefficient: {client._fast_token_coefficient})."
        )

    if config.enable_shell_execution and client.tools:
        try:
            if hasattr(client.tools, 'mount_tool_library_if_absent'):
                client.tools.mount_tool_library_if_absent('system_shell')
            else:
                client.tools.mount_tool_library('system_shell')
            ASCIIColors.success("[CLI] ✅ System Shell library mounted.")
        except (AttributeError, OSError, RuntimeError) as e:
            ASCIIColors.warning(f"Failed to pre-mount system_shell library: {e}")

    if client.tools:
        try:
            if hasattr(client.tools, 'mount_tool_library_if_absent'):
                client.tools.mount_tool_library_if_absent('execute_python')
            else:
                client.tools.mount_tool_library('execute_python')
            ASCIIColors.success("[CLI] ✅ Execute Python library mounted.")
        except (AttributeError, OSError, RuntimeError) as e:
            ASCIIColors.warning(f"Failed to pre-mount execute_python library: {e}")

    if client.tools:
        try:
            if hasattr(client.tools, 'mount_tool_library_if_absent'):
                client.tools.mount_tool_library_if_absent('git_manager')
            else:
                client.tools.mount_tool_library('git_manager')
            ASCIIColors.success("[CLI] ✅ Git Manager library mounted.")
        except (AttributeError, OSError, RuntimeError) as e:
            ASCIIColors.warning(f"Failed to pre-mount git_manager library: {e}")

    return client


def ensure_handbag_structure(config: CodeAgentConfig):
    """Ensures that the handbag directory exists. Only writes default coder SOUL.md if using default handbag."""
    handbag_path = Path(config.handbag_path).resolve()
    default_handbag_path = (APP_DEFAULT_HANDBAG_DIR / "default_coder").resolve()
    is_default = (handbag_path == default_handbag_path)

    handbag_path.mkdir(parents=True, exist_ok=True)
    soul_path = handbag_path / "SOUL.md"

    if is_default:
        metadata = {
            "name": "lollms_code",
            "author": "ParisNeo",
            "category": "software_engineering",
            "description": "An elite autonomous software engineering agent that writes, tests, and fixes code iteratively.",
            "temperature": str(config.temperature)
        }
        yaml_lines = [f"{k}: {v}" for k, v in metadata.items()]
        soul_content = f"---\n{chr(10).join(yaml_lines)}\n---\n\n{CODING_SYSTEM_PROMPT}"

        if not soul_path.exists() or soul_path.read_text(encoding="utf-8") != soul_content:
            soul_path.write_text(soul_content, encoding="utf-8")
            ASCIIColors.info("[CLI] Default SOUL.md updated to latest system prompt standard.")
    else:
        if not soul_path.exists():
            ASCIIColors.warning(f"[CLI] Custom handbag at {handbag_path} does not contain SOUL.md. Creating a baseline SOUL.md.")
            name = handbag_path.name.replace("_", " ").title()
            template_content = f"""---
name: "{name}"
author: "User"
category: "custom"
description: "Custom personality for {name}."
---

You are {name}, a specialized engineering agent.
"""
            soul_path.write_text(template_content, encoding="utf-8")
        else:
            ASCIIColors.info(f"[CLI] Using custom handbag personality from: {soul_path}")

    (handbag_path / "coworkers").mkdir(exist_ok=True)
    (handbag_path / "tools").mkdir(exist_ok=True)
    (handbag_path / "skills").mkdir(exist_ok=True)
    (handbag_path / "memory").mkdir(exist_ok=True)
    (handbag_path / "workspace").mkdir(exist_ok=True)


def ensure_sandbox_structure(config: CodeAgentConfig):
    """Ensures the .lollms_code sandbox exists and cleans transient scripts."""
    sandbox_dir = Path(config.workspace_path) / ".lollms_code"
    scripts_dir = sandbox_dir / "scripts"
    scratchpad = sandbox_dir / "scratchpad.md"
    current_plan = sandbox_dir / "CURRENT.md"
    memory_dir = sandbox_dir / "memory"

    sandbox_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    if scripts_dir.exists():
        for f in scripts_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except OSError as e:
                    ASCIIColors.warning(f"Failed to remove transient script {f.name}: {e}")
    scripts_dir.mkdir(exist_ok=True)

    if not scratchpad.exists():
        scratchpad.write_text("# Agent Scratchpad\n\nUse this space to store long-term notes, code snippets, and task context.\n", encoding="utf-8")

    if not current_plan.exists():
        current_plan.write_text("# Current Task\n\nNo active task plan defined yet. Initialize your macro steps plan here at the start of a task.\n", encoding="utf-8")

def build_environment_context(config: CodeAgentConfig) -> str:
    """Builds a dynamic system prompt block describing the execution environment."""
    is_windows = platform.system() == "Windows"
    os_name = platform.system()
    os_version = platform.version()
    python_version = platform.python_version()

    workspace_root = Path(config.workspace_path).resolve()

    shell_cmd = "cmd / powershell" if is_windows else "bash/sh"
    path_sep = "\\" if is_windows else "/"

    git_branch_info = ""
    git_dir = workspace_root / ".git"
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(workspace_root),
                capture_output=True, text=True, encoding="utf-8", errors="ignore",
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                git_branch_info = f"\n- Git Branch: {result.stdout.strip()}"
        except (OSError, subprocess.SubprocessError) as e:
            ASCIIColors.warning(f"Failed to detect git branch: {e}")

    return f"""
=== ENVIRONMENT CONTEXT (CRITICAL) ===
You are operating in the following environment:
- Operating System: {os_name} {os_version}
- Python Version: {python_version}
- Shell: {shell_cmd}
- Path Separator: `{path_sep}`{git_branch_info}

### OS-SPECIFIC RULES (MANDATORY)
1. **FILE READING**: Use `<unlock_file>` to read ANY file (text, PDF, DOCX, etc.). Do NOT use shell commands for reading.
2. **SHELL COMMANDS**: Use shell commands only for execution (running tests, git, pip).
   - To execute scripts: Use `python script.py` (not `python3` on Windows)
3. **PATHS**: Always use `{path_sep}` for file paths in shell commands. ALL paths must be relative to the Workspace Root. NEVER attempt to access absolute paths outside the workspace.
4. **TRANSIENT SCRIPTS**: When writing test scripts or temporary files, you MUST save them to the Sandbox Directory (`.lollms_code/scripts/`).
   - Example: `python -c "with open('.lollms_code{path_sep}scripts{path_sep}test.py', 'w') as f: f.write('print(1)')"`
   - NEVER create `.py` or `.log` files in the Workspace Root.
5. **SANDBOX ISOLATION**: The Workspace Root contains the user's actual project. Do not modify project files unless explicitly instructed. Use the Sandbox Directory for all experimental work.
=== END ENVIRONMENT CONTEXT ===
"""

def create_coding_personality(config: CodeAgentConfig, client: LollmsClient) -> LollmsPersonality:
    """Creates a coding personality from the handbag structure, injecting client and capabilities."""
    ASCIIColors.rich_print("\n[bold cyan]🔧 Initializing Agent...[/bold cyan]")

    ASCIIColors.rich_print("  [dim]📂 Ensuring handbag structure...[/dim]", end="")
    ensure_handbag_structure(config)
    ASCIIColors.rich_print(" [green]✓[/green]")

    ASCIIColors.rich_print("  [dim]🗂️  Preparing sandbox (.lollms_code/)...[/dim]", end="")
    ensure_sandbox_structure(config)
    ASCIIColors.rich_print(" [green]✓[/green]")

    has_tti = hasattr(client, 'tti') and client.tti is not None
    has_tts = hasattr(client, 'tts') and client.tts is not None
    has_stt = hasattr(client, 'stt') and client.stt is not None

    caps = CapabilityFlags(
        enable_sub_agents=config.enable_sub_agents,
        enable_model_switching=config.enable_model_switching,
        enable_skill_creation=config.enable_skill_creation,
        enable_skill_loading=config.enable_skill_loading,
        enable_workspace_tools=True,
        skills_mode=config.skills_mode,
        max_sub_agent_depth=config.max_sub_agent_depth,
        max_sub_agents_per_turn=config.max_sub_agents_per_turn,
        enable_image_generation=has_tti,
        enable_image_editing=has_tti,
        enable_tts=has_tts,
        enable_stt=has_stt,
    )

    if has_tti:
        ASCIIColors.rich_print("  [green]✓[/green] [dim]Image generation tools enabled (TTI)[/dim]")
    if has_tts:
        ASCIIColors.rich_print("  [green]✓[/green] [dim]Text-to-Speech tools enabled (TTS)[/dim]")
    if has_stt:
        ASCIIColors.rich_print("  [green]✓[/green] [dim]Speech-to-Text tools enabled (STT)[/dim]")

    ASCIIColors.rich_print("  [dim]🧠 Loading handbag & building personality...[/dim]", end="")
    personality = LollmsPersonality.from_handbag(config.handbag_path)
    personality.lollms_client = client
    personality.workspace_path = Path(config.workspace_path)

    # ── 🛡️ ALWAYS ENFORCE SHELL, PYTHON EXECUTION, & PLANNING FOR THE PERSONALITY ──
    env_context = build_environment_context(config)
    if "=== ENVIRONMENT CONTEXT" not in personality.system_prompt:
        personality.system_prompt += "\n\n" + env_context

    if "## MACRO STEPS PLANNING (CURRENT.md)" not in personality.system_prompt:
        personality.system_prompt += "\n\n" + CODING_EXECUTION_HARNESS

    ASCIIColors.rich_print(f" [green]✓[/green] [dim]({personality.name})[/dim]")

    # ── 🧠 PROJECT-LOCAL MEMORY ISOLATION ──
    if config.enable_memory:
        ASCIIColors.rich_print("  [dim]💾 Initializing project-local memory...[/dim]", end="")
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
            project_memory_db = Path(config.workspace_path) / ".lollms_code" / "memory" / "memory.db"
            project_memory_db.parent.mkdir(parents=True, exist_ok=True)

            personality.memory_manager = LollmsMemoryManager(
                db_path=f"sqlite:///{project_memory_db}",
                owner_id=f"project_{Path(config.workspace_path).name}",
                config=MemoryConfig(working_token_budget=2000)
            )
            ASCIIColors.rich_print(f" [green]✓[/green] [dim]({project_memory_db.name})[/dim]")
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            ASCIIColors.rich_print(" [red]✗[/red]")
            ASCIIColors.warning(f"[CLI] Failed to initialize project memory: {e}. Falling back to handbag memory.")

    # ── 💾 FRESH SESSION INITIATION (LONG-TERM FACTS IN MEMORY DB) ──
    project_history_file = get_workspace_conversation_file(config.workspace_path)
    if not getattr(config, "continue_session", False):
        if project_history_file.exists():
            try:
                project_history_file.unlink()
            except OSError:
                pass
        personality._conversation = []
    else:
        personality.load_history_from_disk(project_history_file)
    personality._project_history_file = project_history_file
    ASCIIColors.rich_print("  [green]✓[/green] [dim]Started fresh discussion session (long-term facts preserved in memory)[/dim]")

    if has_tts:
        personality.system_prompt += (
            "\n\n=== TEXT-TO-SPEECH CAPABILITY (ACTIVE) ===\n"
            "You have access to a Text-to-Speech (TTS) binding. You CAN generate speech audio.\n"
            "Use the `tool_text_to_speech` tool to convert text to speech.\n"
            "=== END TEXT-TO-SPEECH CAPABILITY ==="
        )

    if has_stt:
        personality.system_prompt += (
            "\n\n=== SPEECH-TO-TEXT CAPABILITY (ACTIVE) ===\n"
            "You have access to a Speech-to-Text (STT) binding. You CAN transcribe audio.\n"
            "Use the `tool_speech_to_text` tool to transcribe audio files.\n"
            "=== END SPEECH-TO-TEXT CAPABILITY ==="
        )

    personality.capabilities = caps
    personality.max_tokens_per_turn = config.max_tokens_per_turn
    personality.debug_mode = config.debug

    ASCIIColors.rich_print("  [dim]👤 Loading user profile...[/dim]", end="")
    personality._init_user_profile(APP_USER_PROFILE_FILE)
    ASCIIColors.rich_print(" [green]✓[/green]")

    # Grant autonomous workspace authority for CLI tasks (exempt from interactive git prompt blocks)
    object.__setattr__(personality, "_git_autonomy_granted", True)

    ASCIIColors.rich_print("  [dim]📝 Initializing scratchpad...[/dim]", end="")
    try:
        if hasattr(personality, "_init_scratchpad"):
            personality._init_scratchpad()
        ASCIIColors.rich_print(" [green]✓[/green]")
    except (OSError, RuntimeError, ValueError) as e:
        ASCIIColors.rich_print(" [red]✗[/red]")
        ASCIIColors.warning(f"Failed to initialize scratchpad: {e}")

    ASCIIColors.rich_print("  [dim]🔍 Building artefact system...[/dim]", end="")
    try:
        if hasattr(personality, "_init_artefact_system"):
            personality._init_artefact_system()
        ASCIIColors.rich_print(" [green]✓[/green]")
    except (OSError, RuntimeError, ValueError) as e:
        ASCIIColors.rich_print(" [red]✗[/red]")
        ASCIIColors.warning(f"Failed to pre-initialize artefact system for stats: {e}")

    ASCIIColors.rich_print("[bold green]  ✅ Agent initialized and ready.[/bold green]\n")

    return personality

def _index_workspace_with_progress(personality: LollmsPersonality, client: LollmsClient):
    """Validates the workspace path. Full file indexing is no longer performed at startup."""
    try:
        ws_path = personality._resolved_workspace
        if not ws_path or not ws_path.exists():
            return
    except AttributeError as e:
        ASCIIColors.warning(f"Workspace validation failed: {e}")


def _format_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    try:
        from ascii_colors.rich.markup import escape
        return escape(str(val))
    except Exception:
        return str(val).replace("[", "\\[")


def _render_box(content_lines: list[str], title: str = "", border_style: str = "blue"):
    """Draws a Rich panel to stdout using ASCIIColors.panel with forced immediate flush."""
    try:
        content_text = "\n".join(content_lines)
        ASCIIColors.panel(
            content_text,
            title=f"[bold {border_style}]{title}[/bold {border_style}]",
            border_style=border_style,
        )
        sys.stdout.flush()
    except Exception:
        try:
            content_text = "\n".join(content_lines)
            ASCIIColors.panel(content_text, title=title, border_style=border_style)
            sys.stdout.flush()
        except Exception as e:
            print(f"\n--- {title} ---")
            for line in content_lines:
                clean_line = re.sub(r'\[/?[a-zA-Z0-9_ =]+\]', '', str(line))
                print(clean_line)
            print(f"--- end {title} ---\n")
            sys.stdout.flush()


def _render_files_table(files_data: list[dict[str, Any]], title: str):
    console = Console()
    sorted_files = sorted(
        files_data, 
        key=lambda x: x.get("size", 0), 
        reverse=True
    )
    files_table = Table(title=f"[bold]{title}[/bold]", box=None)
    files_table.add_column("Size", style="cyan", no_wrap=True)
    files_table.add_column("Relative Path", style="white")
    for f in sorted_files:
        files_table.add_row(_format_bytes(f["size"]), f["path"])
    console.print(files_table)

def get_workspace_stats(personality: LollmsPersonality) -> dict[str, Any]:
    """
    Calculates statistics about the indexed workspace files.
    Returns total files, loaded files count, and a list of relative paths of loaded files.
    """
    stats = {
        "total_indexed": 0,
        "total_loaded": 0,
        "loaded_files": []
    }
    
    if not hasattr(personality, '_artefact_manager') or not personality._artefact_manager:
        return stats
        
    try:
        from lollms_client.lollms_artefact import ArtefactVisibility
        all_arts = personality._artefact_manager._get_all_raw()
        
        stats["total_indexed"] = len([a for a in all_arts if not a.get("title", "").endswith("::images")])
        
        for art in all_arts:
            if art.get("visibility") == ArtefactVisibility.FULL:
                rel_path = art.get("physical_path") or art.get("title", "")
                if rel_path:
                    ws_root = str(personality._resolved_workspace)
                    if rel_path.startswith(ws_root):
                        rel_path = rel_path[len(ws_root):].lstrip("\\/")

                    file_size = art.get("size", 0)
                    if not file_size:
                        try:
                            abs_path = personality._resolved_workspace / rel_path
                            if abs_path.exists() and abs_path.is_file():
                                file_size = abs_path.stat().st_size
                        except OSError:
                            file_size = 0

                    stats["loaded_files"].append({
                        "path": rel_path,
                        "size": file_size
                    })

        stats["total_loaded"] = len(stats["loaded_files"])
    except (AttributeError, KeyError, OSError) as e:
        ASCIIColors.warning(f"Failed to calculate workspace stats: {e}")

    return stats


class StreamRenderer:
    def __init__(self, config: CodeAgentConfig):
        self.config = config
        self._processing_buffer = ""
        self._in_processing = False
        self._live_artifact_panel = None
        self._live_artifact_title = ""
        self._live_artifact_lang = ""
        self._live_artifact_buffer = ""
        self._last_stream_artifact_title = None
        self._rendered_artefact_ends = set()
        self._rendered_artefact_starts = set()
        self._first_token_printed = False
        self._rendered_artefact_ends = set()
        self._rendered_artefact_starts = set()
        self._first_token_printed = False

    def _render_processing_block(self, block_content: str):
        """Parses and renders a <processing> block as a rich panel."""
        import json as _json

        block_content = re.sub(r'</?processing[^>]*>', '', block_content)

        type_match = re.search(r'type="([^"]+)"', block_content)
        title_match = re.search(r'title="([^"]+)"', block_content)
        params_match = re.search(r'params="([^"]+)"', block_content)

        proc_type = type_match.group(1) if type_match else "action"
        title = title_match.group(1) if title_match else "Processing"

        status_match = re.search(r'<!-- status:(\w+)\s*-->', block_content)
        block_status = status_match.group(1) if status_match else None

        if proc_type == "tool":
            params_str = params_match.group(1) if params_match else "{}"
            try:
                params_dict = _json.loads(params_str)
                params_str_formatted = _json.dumps(params_dict, indent=2, ensure_ascii=False)
            except _json.JSONDecodeError:
                params_str_formatted = params_str

            body_match = re.search(r'>(.*)', block_content, re.DOTALL)
            body_text = body_match.group(1).strip() if body_match else ""
            body_text = re.sub(r'<!-- status:\w+\s*-->', '', body_text).strip()

            if not body_text:
                body_text = "[dim](No execution log output was provided by the tool)[/dim]"

            panel_lines = [f"[cyan]Parameters:[/cyan]\n[dim]{params_str_formatted}[/dim]\n"]
            if block_status == "failure":
                panel_lines.append(f"[cyan]Error Details:[/cyan]\n[red]{body_text}[/red]")
            else:
                panel_lines.append(f"[cyan]Execution Log:[/cyan]\n{body_text}")
            panel_content = "\n".join(panel_lines)

            title_action = title.split(' ')[0].lower()
            dedup_title = title if not title.lower().startswith(title_action) else title[len(title_action):].strip()
            display_title = dedup_title or title

            border = "red" if block_status == "failure" else "blue"
            print("")
            ASCIIColors.panel(
                panel_content,
                title=f"[bold {'red' if block_status == 'failure' else 'blue'}]🛠️ Tool Execution: {display_title}[/bold {'red' if block_status == 'failure' else 'blue'}]",
                border_style=border
            )
        else:
            body_match = re.search(r'>(.*)', block_content, re.DOTALL)
            body_text = body_match.group(1).strip() if body_match else ""
            body_text = re.sub(r'<!-- status:\w+\s*-->', '', body_text).strip()
            if not body_text:
                body_text = "[dim](No output)[/dim]"
            border = "red" if block_status == "failure" else "magenta"
            print("")

            ASCIIColors.panel(
                body_text,
                title=f"[bold {'red' if block_status == 'failure' else 'magenta'}]⚙️ {title}[/bold {'red' if block_status == 'failure' else 'magenta'}]",
                border_style=border
            )

    def _start_live_artifact_panel(self, title: str, lang: str = ""):
        self._live_artifact_title = title
        self._live_artifact_lang = lang
        self._live_artifact_buffer = ""
        self._live_artifact_started = True

    def _update_live_artifact_panel(self, chunk: str, fallback_title: str = "artifact", fallback_lang: str = ""):
        self._live_artifact_buffer += chunk

    def _stop_live_artifact_panel(self):
        self._live_artifact_buffer = ""
        self._live_artifact_started = False

    def _render_callback_event(self, msg_type: Any, meta: dict | None):
        """Renders structured MSG_TYPE events using ASCIIColors panels and tables."""
        if not meta:
            return

        def _is_type(target):
            if msg_type == target:
                return True
            if hasattr(msg_type, "value") and hasattr(target, "value"):
                return msg_type.value == target.value
            if isinstance(msg_type, int) and hasattr(target, "value"):
                return msg_type == target.value
            if isinstance(msg_type, str) and hasattr(target, "name"):
                return msg_type == target.name
            return False

        try:
            if _is_type(MSG_TYPE.MSG_TYPE_TOOL_START):
                tool_name = meta.get("tool_name", "unknown")
                if tool_name == "pending":
                    return

                params = meta.get("parameters", {})
                content_parts = []

                code_val = params.get("code") or params.get("script") if isinstance(params, dict) else None
                if code_val and isinstance(code_val, str) and code_val.strip():
                    content_parts.append("[bold cyan]Code to Execute:[/bold cyan]")
                    for c_line in code_val.strip().splitlines():
                        content_parts.append(f"  [yellow]{_clean_str(c_line)}[/yellow]")
                    other_params = {k: v for k, v in params.items() if k not in ("code", "script")}
                    if other_params:
                        content_parts.append(f"\n[dim]Arguments: {_clean_str(json.dumps(other_params, default=str))}[/dim]")
                elif tool_name == "tool_execute_shell_command" and isinstance(params, dict) and "command" in params:
                    content_parts.append(f"[bold cyan]Command:[/bold cyan] [bold yellow]{_clean_str(params['command'])}[/bold yellow]")
                    if "autonomy_level" in params:
                        content_parts.append(f"[dim]Autonomy: {_clean_str(params['autonomy_level'])}[/dim]")
                elif isinstance(params, dict) and "file_name" in params:
                    content_parts.append(f"[bold cyan]Target File:[/bold cyan] [bold yellow]{_clean_str(params['file_name'])}[/bold yellow]")
                    other_params = {k: v for k, v in params.items() if k != "file_name"}
                    if other_params:
                        content_parts.append(f"[dim]Arguments: {_clean_str(json.dumps(other_params, default=str))}[/dim]")
                elif isinstance(params, dict) and params:
                    rows = [[f"[cyan]{_clean_str(k)}[/cyan]", f"[yellow]{_clean_str(str(v))}[/yellow]"] for k, v in params.items()]
                    table = ASCIIColors.table("Parameter", "Value", rows=rows, box="round")
                    ASCIIColors.rich_print(table)
                    sys.stdout.flush()
                    return
                else:
                    content_parts.append("[dim]No parameters[/dim]")

                content_parts.append("\n[yellow]⏳ Executing...[/yellow]")
                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold blue]🛠️ Tool Call: {_clean_str(tool_name)}[/bold blue]",
                    border_style="blue"
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_TOOL_END):
                tool_name = meta.get("tool_name", "tool")
                success = meta.get("success", False)
                output = meta.get("output", "")
                error = meta.get("error")
                params = meta.get("parameters", {})

                status_str = "[bold green]✅ Success[/bold green]" if success else "[bold red]❌ Failed[/bold red]"
                border = "green" if success else "red"

                content_parts = [f"[cyan]Status:[/cyan] {status_str}"]

                code_val = params.get("code") or params.get("script") if isinstance(params, dict) else None
                if code_val and isinstance(code_val, str) and code_val.strip():
                    code_lines = code_val.strip().splitlines()
                    content_parts.append(f"\n[bold cyan]Executed Code ({len(code_lines)} lines):[/bold cyan]")
                    preview_lines = code_lines[:8] if len(code_lines) <= 10 else code_lines[:5] + ["..."] + code_lines[-3:]
                    for cl in preview_lines:
                        content_parts.append(f"  [dim]{_clean_str(cl)}[/dim]")
                elif isinstance(params, dict) and "command" in params:
                    content_parts.append(f"[cyan]Command:[/cyan] [yellow]{_clean_str(params['command'])}[/yellow]")

                log_source = output if success else (error or output or "")
                if not log_source:
                    log_source = "(No output returned by tool)"

                log_lines = str(log_source).splitlines()
                max_lines = 30
                display_logs = log_lines[:15] + [f"\n... [{len(log_lines)-30} lines omitted for display] ...\n"] + log_lines[-15:] if len(log_lines) > max_lines else log_lines

                log_label = "Execution Output" if success else "Error Details"
                content_parts.append(f"\n[bold cyan]{log_label}:[/bold cyan]")
                for ll in display_logs:
                    content_parts.append(f"  {_clean_str(ll)}")

                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold {border}]🛠️ Tool Result: {_clean_str(tool_name)}[/bold {border}]",
                    border_style=border
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START):
                title = meta.get("title", "artifact")
                lang = meta.get("language", "")
                art_type = meta.get("art_type", "code")
                is_patch = meta.get("is_patch", False)
                op_label = "Patching" if is_patch else "Creating"
                op_icon = "🔧" if is_patch else "📄"

                rows = [
                    ["File", f"[bold yellow]{_clean_str(title)}[/bold yellow]"],
                    ["Type", f"[magenta]{_clean_str(art_type)}[/magenta]" + (f" [dim]({_clean_str(lang)})[/dim]" if lang else "")],
                    ["Operation", f"{op_icon} {op_label}"],
                    ["Status", "[yellow]⏳ Writing to workspace...[/yellow]"]
                ]
                table = ASCIIColors.table(
                    "Field", "Details",
                    rows=rows,
                    title=f"[bold magenta]{op_icon} Artifact: {op_label} {_clean_str(title)}[/bold magenta]",
                    box="round"
                )
                ASCIIColors.rich_print(table)
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END):
                title = meta.get("title", "artifact")
                success = meta.get("success", False)
                version = meta.get("version", 1)
                error = meta.get("error")
                content = meta.get("content", "")

                status_str = f"[bold green]✅ Saved (v{version})[/bold green]" if success else f"[bold red]❌ Failed: {_clean_str(error)}[/bold red]"
                border = "green" if success else "red"

                content_parts = [
                    f"[cyan]File:[/cyan] [bold yellow]{_clean_str(title)}[/bold yellow] (v{version})",
                    f"[cyan]Status:[/cyan] {status_str}"
                ]

                if content and isinstance(content, str):
                    c_lines = content.strip().splitlines()
                    content_parts.append(f"\n[bold cyan]Content Preview ({len(c_lines)} lines):[/bold cyan]")
                    preview_content = c_lines[:12] if len(c_lines) <= 16 else c_lines[:8] + [f"... [{len(c_lines)-12} lines omitted] ..."] + c_lines[-4:]
                    for cl in preview_content:
                        content_parts.append(f"  [yellow]{_clean_str(cl)}[/yellow]")

                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold {border}]📄 Artifact Complete: {_clean_str(title)}[/bold {border}]",
                    border_style=border
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_ARTEFACT_SYMBOL_DETECTED):
                sym = meta.get("symbol", {})
                detail = sym.get("detail") or meta.get("detail", "")
                title = meta.get("title", "artifact")
                ASCIIColors.rich_print(f"  [bold cyan]•[/bold cyan] [dim]Writing {_clean_str(title)}:[/dim] [bold yellow]{_clean_str(detail)}[/bold yellow]")
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE):
                action = meta.get("action", "update")
                status = meta.get("status", "success")
                files = meta.get("files", [])
                error = meta.get("error")

                status_color = "green" if status == "success" else "red"
                content_parts = [
                    f"[cyan]Action:[/cyan] [yellow]{_clean_str(action.replace('_', ' ').capitalize())}[/yellow]",
                    f"[cyan]Status:[/cyan] [{status_color}]{_clean_str(status or ('Success' if not error else 'Failed'))}[/{status_color}]"
                ]
                if files:
                    content_parts.append("[cyan]Files:[/cyan]")
                    for f in files:
                        content_parts.append(f"  • {_clean_str(str(f))}")
                if error:
                    content_parts.append(f"[red]Error:[/red] {_clean_str(str(error))}")

                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold yellow]📂 Context: {_clean_str(action.replace('_', ' ').capitalize())}[/bold yellow]",
                    border_style="yellow" if status != "failure" else "red"
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE):
                action = meta.get("action", "update")
                message = meta.get("message", "Scratchpad updated.")
                preview = meta.get("preview", "")

                content_parts = [f"[cyan]Status:[/cyan] [green]{_clean_str(message)}[/green]"]
                if preview:
                    content_parts.append(f"[cyan]Preview:[/cyan]\n[dim]{_clean_str(preview)}[/dim]")

                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold cyan]📝 Scratchpad: {_clean_str(action.replace('_', ' ').capitalize())}[/bold cyan]",
                    border_style="cyan"
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_SKILL_DONE):
                title = meta.get("title", "skill")
                category = meta.get("category", "general")
                desc = meta.get("description", "")
                content = meta.get("content", "")

                content_parts = [
                    f"[cyan]Skill:[/cyan] [bold yellow]{_clean_str(title)}[/bold yellow]",
                    f"[cyan]Category:[/cyan] [magenta]{_clean_str(category)}[/magenta]",
                ]
                if desc:
                    content_parts.append(f"[cyan]Description:[/cyan] {_clean_str(desc)}")
                if content and isinstance(content, str):
                    c_lines = content.strip().splitlines()
                    content_parts.append(f"\n[bold cyan]Doctrine Preview ({len(c_lines)} lines):[/bold cyan]")
                    for cl in c_lines[:8]:
                        content_parts.append(f"  {_clean_str(cl)}")

                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold yellow]🧠 Learned Skill: {_clean_str(title)}[/bold yellow]",
                    border_style="yellow"
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START):
                worker_idx = meta.get("worker_index", 1)
                task = meta.get("task", "")
                files = meta.get("context_files", [])
                content_parts = [
                    f"[cyan]Worker:[/cyan] [bold yellow]Worker #{worker_idx}[/bold yellow]",
                    f"[cyan]Task:[/cyan] {_clean_str(task[:300])}",
                ]
                if files:
                    content_parts.append(f"[cyan]Context Files:[/cyan] {', '.join(_clean_str(f) for f in files)}")
                content_parts.append("\n[yellow]⏳ Specialist worker running...[/yellow]")
                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold cyan]🤖 Sub-Agent Spawn: Worker #{worker_idx}[/bold cyan]",
                    border_style="cyan"
                )
                sys.stdout.flush()

            elif _is_type(MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END):
                worker_idx = meta.get("worker_index", 1)
                success = meta.get("success", False)
                digest = meta.get("report_digest", "")
                files = meta.get("files", [])
                status_str = "[bold green]✅ Success[/bold green]" if success else "[bold red]❌ Failed[/bold red]"
                border = "green" if success else "red"
                content_parts = [
                    f"[cyan]Worker:[/cyan] [bold yellow]Worker #{worker_idx}[/bold yellow]",
                    f"[cyan]Status:[/cyan] {status_str}",
                ]
                if files:
                    content_parts.append(f"[cyan]Files Created/Modified:[/cyan] {', '.join(_clean_str(f) for f in files)}")
                if digest:
                    content_parts.append(f"\n[cyan]Report Digest:[/cyan]\n{_clean_str(digest[:1000])}")
                ASCIIColors.panel(
                    "\n".join(content_parts),
                    title=f"[bold {border}]🤖 Sub-Agent Finished: Worker #{worker_idx}[/bold {border}]",
                    border_style=border
                )
                sys.stdout.flush()

        except Exception as ex:
            ASCIIColors.warning(f"[StreamRenderer] Error rendering {msg_type}: {ex}")
            name_val = meta.get("title") or meta.get("tool_name") or "Action"
            ASCIIColors.cyan(f"\n▶ [{msg_type}] {name_val}")
            for k, v in meta.items():
                if k not in ("content", "output") and v:
                    ASCIIColors.info(f"  • {k}: {v}")
            sys.stdout.flush()

    def flush(self):
        """Flushes any pending buffers, rendering unclosed tags as raw text."""
        self._stop_live_artifact_panel()
        self._first_token_printed = False
        if self._in_processing and self._processing_buffer:
            ASCIIColors.rich_print(self._processing_buffer, end="")
            self._processing_buffer = ""
            self._in_processing = False

    def __call__(self, chunk: str, msg_type: Any = None, meta: dict | None = None) -> bool:
        if msg_type is None:
            msg_type = MSG_TYPE.MSG_TYPE_CHUNK

        if msg_type == MSG_TYPE.MSG_TYPE_NEW_MESSAGE:
            ASCIIColors.rich_print("\n[bold green]🤖 Generating...[/bold green]")
            sys.stdout.flush()
            return True

        if msg_type == MSG_TYPE.MSG_TYPE_ROUND_START:
            round_id = meta.get("round_id", 1) if meta else 1
            max_r = meta.get("max_rounds", self.config.max_reasoning_steps) if meta else self.config.max_reasoning_steps
            ASCIIColors.rich_print(f"\n[dim]── Round {round_id}/{max_r} ──[/dim]")
            sys.stdout.flush()
            self._rendered_artefact_ends.clear()
            self._rendered_artefact_starts.clear()
            return True

        if msg_type == MSG_TYPE.MSG_TYPE_ROUND_END:
            return True

        if msg_type in [
            MSG_TYPE.MSG_TYPE_TOOL_START,
            MSG_TYPE.MSG_TYPE_TOOL_END,
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START,
            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END,
            MSG_TYPE.MSG_TYPE_ARTEFACT_SYMBOL_DETECTED,
            MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE,
            MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE,
            MSG_TYPE.MSG_TYPE_SKILL_DONE,
            MSG_TYPE.MSG_TYPE_WORKER_SPAWN_START,
            MSG_TYPE.MSG_TYPE_WORKER_SPAWN_END,
        ]:
            if meta and meta.get("tool_name") == "pending":
                return True
            if meta and meta.get("status") == "streaming" and not meta.get("files"):
                return True
            if msg_type == MSG_TYPE.MSG_TYPE_TOOL_END and meta and meta.get("stream_complete") and "success" not in meta:
                return True
            if msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START and meta:
                start_key = (meta.get("title"), meta.get("is_patch"))
                if start_key in self._rendered_artefact_starts:
                    return True
                self._rendered_artefact_starts.add(start_key)
            if msg_type == MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END and meta:
                end_key = (meta.get("title"), meta.get("is_patch"))
                if end_key in self._rendered_artefact_ends:
                    return True
                self._rendered_artefact_ends.add(end_key)

            self._processing_buffer = ""
            self._in_processing = False
            self._render_callback_event(msg_type, meta)
            return True

        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("was_processed"):
                # In callback mode, structured events handle UI panels; discard raw processing chunks
                return True
            elif meta and meta.get("live_tool_chunk"):
                return True
            elif meta and meta.get("live_artifact_chunk"):
                art_title = "artifact"
                art_lang = ""
                if isinstance(meta, dict):
                    art_title = meta.get("artifact_title", art_title)
                    art_lang = meta.get("artifact_lang", art_lang)

                clean_chunk = chunk
                if "<<<<<<< SEARCH" in clean_chunk:
                    clean_chunk = clean_chunk.replace("<<<<<<< SEARCH", "[🔍 SEARCH]")
                if "=======" in clean_chunk:
                    clean_chunk = clean_chunk.replace("=======", "[✏️ REPLACE]")
                if ">>>>>>> REPLACE" in clean_chunk:
                    clean_chunk = clean_chunk.replace(">>>>>>> REPLACE", "[✅ END REPLACE]")

                self._update_live_artifact_panel(clean_chunk, fallback_title=art_title, fallback_lang=art_lang)
                return True
            else:
                if "<done" in chunk and "/>" in chunk:
                    return True

                # Strip any stray processing tags or comments from the conversational stream
                clean_chunk = re.sub(r'</?processing[^>]*>', '', chunk, flags=re.IGNORECASE)
                clean_chunk = re.sub(r'<!--\s*status:[^>]*-->', '', clean_chunk, flags=re.IGNORECASE)

                if not clean_chunk:
                    return True

                if not getattr(self, '_first_token_printed', False):
                    self._first_token_printed = True
                ASCIIColors.rich_print(clean_chunk, end="")
                sys.stdout.flush()
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            ASCIIColors.rich_print(f"[dim]{chunk}[/dim]", end="")
            sys.stdout.flush()
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            if meta and meta.get("done_intercepted"):
                self._stop_live_artifact_panel()
                print()
                ASCIIColors.rule("[bold green]✅ Task Completed (<done/>)[/bold green]")
                sys.stdout.flush()
                return True
            else:
                ASCIIColors.rich_print(f"\n[blue][INFO] {chunk}[/blue]")
                sys.stdout.flush()
                return True
    

def _display_context_status(personality: LollmsPersonality, client: LollmsClient):
    """Calculates and displays the current context fill status as a Rich panel."""
    ctx_status = get_context_fill_status(personality, client)
    if ctx_status:
        used = ctx_status["used_tokens"]
        max_t = ctx_status["max_tokens"]
        pct = ctx_status["fill_percentage"]

        status_color = "green"
        if pct > 85.0:
            status_color = "red"
        elif pct > 65.0:
            status_color = "yellow"

        status_content = (
            f"[cyan]Used Tokens:[/cyan] {used:,} / {max_t:,}\n"
            f"[cyan]Context Fill:[/cyan] [{status_color}]{pct:.1f}%[/{status_color}]"
        )
        ASCIIColors.panel(status_content, title="[bold blue]📊 Context Status[/bold blue]", border_style="blue")
    else:
        ASCIIColors.yellow("  Context status unavailable.")


def display_result(result: dict[str, Any], config: CodeAgentConfig, elapsed: float):
    ASCIIColors.rule("[bold cyan]📊 SESSION REPORT[/bold cyan]")

    summary_rows = [
        ["Total rounds", str(result.get('rounds', 0))],
        ["Tool calls", str(len(result.get('tool_calls', [])))],
        ["Was cancelled", str(result.get('was_cancelled', False))],
        ["Elapsed time", f"{elapsed:.1f}s"]
    ]

    ctx_health = result.get("context_health")
    if ctx_health and ctx_health.get("max_tokens", 0) > 0:
        used = ctx_health.get("used_tokens", 0)
        max_t = ctx_health.get("max_tokens", 0)
        pct = ctx_health.get("fill_percentage", 0.0)
        summary_rows.append(["Context used", f"{used:,} / {max_t:,} tokens"])
        summary_rows.append(["Context fill", f"{pct:.1f}%"])

    summary_table = ASCIIColors.table(
        "Metric", "Value",
        rows=summary_rows,
        title="[bold]Execution Summary[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(summary_table)

    if config.show_workspace_changes and result.get("workspace_changes"):
        changes_table = ASCIIColors.table(
            "Action", "Path", "Size",
            rows=[[c.get("action", "?"), c.get("path", "?"), f"{c.get('size', 0):,} bytes"] for c in result["workspace_changes"]],
            title="[bold green]📁 Workspace Changes[/bold green]",
            box="round"
        )
        ASCIIColors.rich_print(changes_table)

    if config.show_skills and (result.get("skills_created") or result.get("skills_updated")):
        skills_table = ASCIIColors.table(
            "Action", "Skill",
            rows=([["Created", s] for s in result.get("skills_created", [])] + 
                  [["Updated", s] for s in result.get("skills_updated", [])]),
            title="[bold yellow]🎓 Skills Activity[/bold yellow]",
            box="round"
        )
        ASCIIColors.rich_print(skills_table)

    if result.get("sub_agents_spawned", 0) > 0:
        ASCIIColors.magenta(f"\n  🧠 Sub-agents spawned: {result['sub_agents_spawned']}")

    if result.get("model_switches"):
        ASCIIColors.blue(f"  🔄 Model switches: {result['model_switches']}")


def run_single_prompt(personality: LollmsPersonality, client: LollmsClient, prompt: str, config: CodeAgentConfig) -> int:
    _index_workspace_with_progress(personality, client)

    if config.debug:
        dump_startup_context(personality, client)

    # Use project-local prompt history for REPL tracking
    prompt_history_file = get_workspace_prompt_history_file(config.workspace_path)
    history = PersistentHistory(prompt_history_file)
    history.add(prompt)

    renderer = StreamRenderer(config)

    config_panel_content = (
        f"[cyan]Workspace:[/cyan] {config.workspace_path}\n"
        f"[cyan]Handbag:[/cyan]   {personality.name} ({Path(config.handbag_path).name})\n"
        f"[cyan]Model:[/cyan]      {config.active_model_name}\n"
        f"[cyan]Binding:[/cyan]    {config.active_binding_name}\n"
        f"[cyan]Max steps:[/cyan]  {config.max_reasoning_steps}\n"
        f"[cyan]Memory:[/cyan]     {'enabled' if config.enable_memory else 'disabled'}\n"
        f"[cyan]Skills:[/cyan]     {config.skills_mode}\n"
        f"[cyan]Sub-agents:[/cyan] {'enabled' if config.enable_sub_agents else 'disabled'}"
    )
    ASCIIColors.panel(config_panel_content, title=f"[bold green]🚀 lollms_code v{APP_VERSION}[/bold green]", border_style="green")
    
    ASCIIColors.panel(f"[magenta]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/magenta]", title="[bold]📝 Task[/bold]", border_style="magenta")
    
    # ── 📂 WORKSPACE STATS ──
    ws_stats = get_workspace_stats(personality)
    if ws_stats["total_indexed"] > 0:
        stats_content = (
            f"[cyan]Indexed Files:[/cyan] {ws_stats['total_indexed']}\n"
            f"[cyan]Loaded in Context:[/cyan] {ws_stats['total_loaded']}"
        )
        ASCIIColors.panel(stats_content, title="[bold blue]📂 Workspace Telemetry[/bold blue]", border_style="blue")
        
        if ws_stats["loaded_files"]:
            _render_files_table(ws_stats["loaded_files"], "Pre-loaded Context Files [C]")

    ASCIIColors.rule("[bold]🤖 Agent output[/bold]")

    start_time = time.time()
    ASCIIColors.rich_print("") # Ensure output starts on a new line

    renderer._first_token_printed = False

    def _signal_handler(sig, frame):
        ASCIIColors.yellow("\n\n⚠️  Interrupt received. Cancelling generation...")
        if hasattr(client, 'cancel'):
            client.cancel()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        result = personality.chat(
            prompt=prompt,
            lollms_client=client,
            streaming_callback=renderer,
            max_reasoning_steps=config.max_reasoning_steps,
            temperature=config.temperature,
            n_predict=config.max_tokens_per_turn,
            enable_artefacts=True,
            use_internal_history=False,
            enable_shell=config.enable_shell_execution,
            enable_python_exec=True,
            enable_workspace_tools=True,
            event_mode=EventMode.FULL_CALLBACK_MODE,
            enforce_end_tag=True
        )
    except KeyboardInterrupt:
        if hasattr(client, 'cancel'):
            client.cancel()
        ASCIIColors.yellow("\n\n⚠️  Generation cancelled by user.")
        return 130
    except (RuntimeError, ValueError, OSError, ConnectionError) as e:
        trace_exception(e)
        ASCIIColors.red(f"\n\n💥 Fatal error: {e}")
        return 1

    # Flush the renderer to ensure any unclosed tags are printed
    renderer.flush()

    # Save the updated conversation history to the project directory
    if hasattr(personality, "_project_history_file"):
        personality.save_history_to_disk(personality._project_history_file)

    elapsed = time.time() - start_time
    display_result(result, config, elapsed)

    ASCIIColors.panel(result.get("response", ""), title="[bold cyan]📝 FINAL OUTPUT[/bold cyan]", border_style="cyan")

    return 0 if not result.get("was_cancelled") else 130


def get_context_fill_status(personality: LollmsPersonality, client: LollmsClient) -> dict[str, Any] | None:
    """Safely calculates the full context fill status, including memories and loaded files."""
    try:
        max_ctx = client.get_ctx_size() or 0
        if max_ctx <= 0:
            return None

        breakdown = {
            "system_prompt": 0,
            "workspace_tree": 0,
            "loaded_files": 0,
            "scratchpad": 0,
            "active_memories": 0
        }

        active_tools = personality._discover_tools(None, [])
        full_system_prompt = personality._build_system_prompt(active_tools)
        breakdown["system_prompt"] = client.count_tokens(full_system_prompt) or 0

        ws_ctx = personality._build_workspace_context_block()
        if ws_ctx:
            breakdown["workspace_tree"] = client.count_tokens(ws_ctx) or 0

        scratchpad_ctx = personality._build_scratchpad_context()
        if scratchpad_ctx:
            breakdown["scratchpad"] = client.count_tokens(scratchpad_ctx) or 0

        if hasattr(personality, "_artefact_manager") and personality._artefact_manager:
            from lollms_client.lollms_artefact import ArtefactVisibility
            all_arts = personality._artefact_manager._get_all_raw()
            loaded_files_tokens = 0
            for art in all_arts:
                if art.get("visibility") == ArtefactVisibility.FULL:
                    content = art.get("content", "")
                    if content and not art.get("title", "").endswith("::images"):
                        loaded_files_tokens += client.count_tokens(content) or 0
            breakdown["loaded_files"] = loaded_files_tokens

        if hasattr(personality, "memory_manager") and personality.memory_manager:
            try:
                mem_zone = personality.memory_manager.build_working_zone()
                if mem_zone:
                    breakdown["active_memories"] = client.count_tokens(mem_zone) or 0
            except (AttributeError, RuntimeError, ValueError) as e:
                ASCIIColors.warning(f"Failed to compute memory token count: {e}")

        used_tokens = sum(breakdown.values())
        fill_pct = round((used_tokens / max_ctx) * 100, 1)

        return {
            "used_tokens": used_tokens,
            "max_tokens": max_ctx,
            "fill_percentage": fill_pct,
            "breakdown": breakdown
        }
    except (AttributeError, RuntimeError, ValueError, OSError) as e:
        ASCIIColors.warning(f"Failed to compute context fill status: {e}")
        return None
    
def dump_startup_context(personality: LollmsPersonality, client: LollmsClient):
    """
    🐛 DEBUG INSTRUMENTATION: Writes the initial system prompt and active tools
    to a debug log BEFORE the agentic loop begins. This captures the "Zero State"
    for diagnosing initialization hangs or context bloat.
    """
    if not getattr(personality, 'debug_mode', False):
        return

    try:
        ws_path = personality._resolved_workspace
        if not ws_path:
            return

        debug_dir = ws_path / ".lollms_code" / "_debug_dumps"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for item in debug_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except OSError as e:
                    ASCIIColors.warning(f"Failed to remove debug file {item.name}: {e}")
            elif item.is_dir():
                try:
                    shutil.rmtree(str(item))
                except OSError as e:
                    ASCIIColors.warning(f"Failed to remove debug directory {item.name}: {e}")

        debug_log_path = debug_dir / "startup_context.log"

        active_tools = personality._discover_tools(None, [])
        full_system_prompt = personality._build_system_prompt(active_tools)

        # Capture Memory and Scratchpad context if they exist
        mem_ctx = ""
        if hasattr(personality, '_build_user_profile_context'):
            mem_ctx += personality._build_user_profile_context()
        if hasattr(personality, '_build_scratchpad_context'):
            mem_ctx += personality._build_scratchpad_context()

        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("🐛 [DEBUG] STARTUP CONTEXT DUMP (ZERO-STATE)\n")
            f.write("=" * 80 + "\n\n")

            f.write("--- ACTIVE TOOLS ---\n")
            f.write("\n".join(
                f"- {t_name}: {t_spec.get('description', '')[:100]}"
                for t_name, t_spec in active_tools.items()
            ))
            f.write("\n\n")

            f.write("--- FULL SYSTEM PROMPT (STABLE PREFIX) ---\n")
            f.write(full_system_prompt + "\n\n")

            if mem_ctx.strip():
                f.write("--- MEMORY & SCRATCHPAD CONTEXT ---\n")
                f.write(mem_ctx + "\n\n")

            f.write("--- WORKSPACE CONTEXT SNAPSHOT ---\n")
            ws_ctx = personality._build_workspace_context_block()
            f.write(ws_ctx + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("📊 CONTEXT STATS\n")
            f.write("=" * 80 + "\n")
            ctx_stats = get_context_fill_status(personality, client)
            if ctx_stats:
                f.write(f"Used Tokens: {ctx_stats['used_tokens']:,}\n")
                f.write(f"Max Tokens:  {ctx_stats['max_tokens']:,}\n")
                f.write(f"Fill %:      {ctx_stats['fill_percentage']}%\n")
            else:
                f.write("Context stats unavailable.\n")

        ASCIIColors.info(f"[CLI] 🐛 Startup context dumped to: {debug_log_path}")
    except (AttributeError, OSError, RuntimeError, ValueError) as e:
        ASCIIColors.warning(f"[CLI] Failed to dump startup context: {e}")

def _advanced_prompt(history: PersistentHistory, commands: list[str]) -> str | None:
    """
    Cross-platform raw key-capture prompt with Ghost-Text Autocomplete and Multi-line support.
    - Submit: Press Enter to submit the prompt.
    - Multi-line: Press Shift+Enter (Windows) or Alt+Enter (cross-platform) to insert a newline.
    - Type '/': shows '/exit' in gray.
    - Type 'f': shows '/files' in gray.
    - Press Tab or Right Arrow: accepts the gray suggestion.
    - Press Up/Down: cycles through matching commands (or history if no match).
    """
    import sys

    PROMPT_TEXT = "👤 You> "
    PROMPT_LEN = len(PROMPT_TEXT)
    CONT_PROMPT_TEXT = "... "
    CONT_PROMPT_LEN = len(CONT_PROMPT_TEXT)

    def _draw_line(buffer: str, cursor_pos: int, ghost: str = "", line_index: int = 0):
        sys.stdout.write("\r\033[K")
        lines = buffer.split('\n')
        for i, line in enumerate(lines):
            prefix = PROMPT_TEXT if i == 0 else CONT_PROMPT_TEXT
            if i > 0:
                sys.stdout.write("\n")
            if i == line_index and ghost:
                sys.stdout.write(f"{prefix}{line}\033[90m{ghost}\033[0m")
            else:
                sys.stdout.write(f"{prefix}{line}")

        if cursor_pos < len(buffer + ghost):
            lines_before = buffer[:cursor_pos].count('\n')
            if lines_before == 0:
                target_col = PROMPT_LEN + cursor_pos + 1
            else:
                col_in_line = cursor_pos - buffer.rfind('\n', 0, cursor_pos) - 1
                target_col = CONT_PROMPT_LEN + col_in_line + 1
                sys.stdout.write(f"\033[{lines_before}A")
            sys.stdout.write(f"\033[{target_col}G")
        sys.stdout.flush()

    def _native_prompt_unix() -> str | None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        buffer = ""
        cursor_pos = 0
        history_idx = len(history.entries)
        current_input = ""
        active_suggestion_idx = -1

        def get_suggestions():
            lines = buffer.split('\n')
            current_line = lines[-1]
            if not current_line.startswith("/"):
                return []
            return [c for c in commands if c.startswith(current_line)]

        def get_line_index():
            if '\n' not in buffer:
                return 0
            return buffer[:cursor_pos].count('\n')

        try:
            tty.setraw(fd)
            sys.stdout.write(PROMPT_TEXT)
            sys.stdout.flush()

            while True:
                ch = sys.stdin.read(1)
                sugg = get_suggestions()
                ghost_text = ""

                if ch == '\r':
                    if buffer:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        if not buffer.strip().startswith("/"):
                            history.add(buffer)
                    return buffer
                elif ch == '\x1b':
                    ch2 = sys.stdin.read(1)
                    if ch2 == '\r':
                        buffer = buffer[:cursor_pos] + '\n' + buffer[cursor_pos:]
                        cursor_pos += 1
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos, "", line_index=get_line_index())
                    else:
                        ch3 = sys.stdin.read(1)
                        if ch2 == '[':
                            if ch3 == 'A':
                                if sugg:
                                    active_suggestion_idx = (active_suggestion_idx - 1) % len(sugg)
                                    lines = buffer.split('\n')
                                    lines[-1] = sugg[active_suggestion_idx]
                                    buffer = '\n'.join(lines)
                                    cursor_pos = len(buffer)
                                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
                                elif history.entries:
                                    if history_idx == len(history.entries):
                                        current_input = buffer
                                    history_idx = max(0, history_idx - 1)
                                    buffer = history.entries[history_idx]
                                    cursor_pos = len(buffer)
                                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
                            elif ch3 == 'B':
                                if sugg:
                                    active_suggestion_idx = (active_suggestion_idx + 1) % len(sugg)
                                    lines = buffer.split('\n')
                                    lines[-1] = sugg[active_suggestion_idx]
                                    buffer = '\n'.join(lines)
                                    cursor_pos = len(buffer)
                                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
                                elif history_idx < len(history.entries):
                                    history_idx += 1
                                    if history_idx == len(history.entries):
                                        buffer = current_input
                                    else:
                                        buffer = history.entries[history_idx]
                                    cursor_pos = len(buffer)
                                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
                            elif ch3 == 'C':
                                if sugg and cursor_pos == len(buffer):
                                    lines = buffer.split('\n')
                                    lines[-1] = sugg[0]
                                    buffer = '\n'.join(lines)
                                    cursor_pos = len(buffer)
                                    active_suggestion_idx = -1
                                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
                                elif cursor_pos < len(buffer):
                                    cursor_pos += 1
                                    _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
                            elif ch3 == 'D':
                                if cursor_pos > 0:
                                    cursor_pos -= 1
                                    _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
                elif ch in ('\x7f', '\b'):
                    if cursor_pos > 0:
                        buffer = buffer[:cursor_pos-1] + buffer[cursor_pos:]
                        cursor_pos -= 1
                        active_suggestion_idx = -1
                        sugg = get_suggestions()
                        current_line = buffer.split('\n')[-1]
                        ghost_text = sugg[0][len(current_line):] if sugg and sugg[0].startswith(current_line) else ""
                        _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
                elif ch == '\t':
                    if sugg:
                        lines = buffer.split('\n')
                        lines[-1] = sugg[0]
                        buffer = '\n'.join(lines)
                        cursor_pos = len(buffer)
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                elif len(ch) == 1 and ch.isprintable():
                    buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                    cursor_pos += 1
                    active_suggestion_idx = -1
                    sugg = get_suggestions()
                    current_line = buffer.split('\n')[-1]
                    ghost_text = sugg[0][len(current_line):] if sugg and sugg[0].startswith(current_line) else ""
                    _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
        except Exception:
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _native_prompt_windows() -> str | None:
        import msvcrt
        import ctypes

        VK_LSHIFT = 0xA0
        VK_RSHIFT = 0xA1

        def _is_shift_pressed() -> bool:
            """Checks if either Shift key is currently held down using the Windows API."""
            try:
                return bool(
                    ctypes.windll.user32.GetAsyncKeyState(VK_LSHIFT) & 0x8000
                    or ctypes.windll.user32.GetAsyncKeyState(VK_RSHIFT) & 0x8000
                )
            except (AttributeError, OSError):
                return False

        buffer = ""
        cursor_pos = 0
        history_idx = len(history.entries)
        current_input = ""
        active_suggestion_idx = -1

        def get_suggestions():
            lines = buffer.split('\n')
            current_line = lines[-1]
            if not current_line.startswith("/"):
                return []
            return [c for c in commands if c.startswith(current_line)]

        def get_line_index():
            if '\n' not in buffer:
                return 0
            return buffer[:cursor_pos].count('\n')

        sys.stdout.write(PROMPT_TEXT)
        sys.stdout.flush()

        while True:
            if not msvcrt.kbhit():
                continue

            ch = msvcrt.getwch()
            sugg = get_suggestions()
            ghost_text = ""

            if ch == '\r':
                if _is_shift_pressed():
                    buffer = buffer[:cursor_pos] + '\n' + buffer[cursor_pos:]
                    cursor_pos += 1
                    active_suggestion_idx = -1
                    _draw_line(buffer, cursor_pos, "", line_index=get_line_index())
                else:
                    if buffer:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        if not buffer.strip().startswith("/"):
                            history.add(buffer)
                    return buffer
            elif ch == '\x00' or ch == '\xe0':
                ch2 = msvcrt.getwch()
                if ch2 == '\r':
                    if _is_shift_pressed():
                        buffer = buffer[:cursor_pos] + '\n' + buffer[cursor_pos:]
                        cursor_pos += 1
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos, "", line_index=get_line_index())
                    else:
                        if buffer:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            if not buffer.strip().startswith("/"):
                                history.add(buffer)
                        return buffer
                elif ch2 == 'H':
                    if sugg:
                        active_suggestion_idx = (active_suggestion_idx - 1) % len(sugg)
                        lines = buffer.split('\n')
                        lines[-1] = sugg[active_suggestion_idx]
                        buffer = '\n'.join(lines)
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                    elif history.entries:
                        if history_idx == len(history.entries): current_input = buffer
                        history_idx = max(0, history_idx - 1)
                        buffer = history.entries[history_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                elif ch2 == 'P':
                    if sugg:
                        active_suggestion_idx = (active_suggestion_idx + 1) % len(sugg)
                        lines = buffer.split('\n')
                        lines[-1] = sugg[active_suggestion_idx]
                        buffer = '\n'.join(lines)
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                    elif history_idx < len(history.entries):
                        history_idx += 1
                        buffer = current_input if history_idx == len(history.entries) else history.entries[history_idx]
                        cursor_pos = len(buffer)
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                elif ch2 == 'M':
                    if sugg and cursor_pos == len(buffer):
                        lines = buffer.split('\n')
                        lines[-1] = sugg[0]
                        buffer = '\n'.join(lines)
                        cursor_pos = len(buffer)
                        active_suggestion_idx = -1
                        _draw_line(buffer, cursor_pos, line_index=get_line_index())
                    elif cursor_pos < len(buffer):
                        cursor_pos += 1
                        _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
                elif ch2 == 'K':
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
            elif ch in ('\x08', '\x7f'):
                if cursor_pos > 0:
                    buffer = buffer[:cursor_pos-1] + buffer[cursor_pos:]
                    cursor_pos -= 1
                    active_suggestion_idx = -1
                    sugg = get_suggestions()
                    current_line = buffer.split('\n')[-1]
                    ghost_text = sugg[0][len(current_line):] if sugg and sugg[0].startswith(current_line) else ""
                    _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())
            elif ch == '\t':
                if sugg:
                    lines = buffer.split('\n')
                    lines[-1] = sugg[0]
                    buffer = '\n'.join(lines)
                    cursor_pos = len(buffer)
                    active_suggestion_idx = -1
                    _draw_line(buffer, cursor_pos, line_index=get_line_index())
            elif ch.isprintable():
                buffer = buffer[:cursor_pos] + ch + buffer[cursor_pos:]
                cursor_pos += 1
                active_suggestion_idx = -1
                sugg = get_suggestions()
                current_line = buffer.split('\n')[-1]
                ghost_text = sugg[0][len(current_line):] if sugg and sugg[0].startswith(current_line) else ""
                _draw_line(buffer, cursor_pos, ghost_text, line_index=get_line_index())

    try:
        if sys.platform == 'win32':
            return _native_prompt_windows()
        else:
            return _native_prompt_unix()
    except (OSError, ImportError, RuntimeError):
        try:
            return input(PROMPT_TEXT)
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            return None  
            
        
def _switch_workspace_interactive(config: CodeAgentConfig, client: LollmsClient) -> LollmsPersonality | None:
    """Handles the interactive workspace switching process."""
    try:
        ASCIIColors.rule("[bold cyan]📂 Switch Workspace[/bold cyan]")
        ASCIIColors.info(f"Current workspace: [yellow]{config.workspace_path}[/yellow]")
        ASCIIColors.info("Select a new workspace directory or type a path manually.")
        
        default_path = Path(config.workspace_path).resolve()
        
        try:
            selected_path = questionary.path(
                "Enter new workspace path:",
                default=str(default_path),
                only_directories=True
            ).ask()
        except (ImportError, RuntimeError, OSError):
            selected_path = input("Enter new workspace path manually: ").strip()
            
        if not selected_path:
            ASCIIColors.yellow("Workspace switch cancelled.")
            return None
            
        new_path = Path(selected_path).resolve()
        
        if not new_path.exists():
            ASCIIColors.red(f"Directory does not exist: {new_path}")
            return None
            
        if not new_path.is_dir():
            ASCIIColors.red(f"Path is not a directory: {new_path}")
            return None
            
        if str(new_path) == config.workspace_path:
            ASCIIColors.yellow("Already in this workspace.")
            return None
            
        config.workspace_path = str(new_path)
        config.save()
        
        ASCIIColors.success(f"✅ Workspace switched to: {new_path}")
        
        personality = create_coding_personality(config, client)
        
        if config.debug:
            dump_startup_context(personality, client)
            
        return personality
        
    except KeyboardInterrupt:
        ASCIIColors.yellow("\nWorkspace switch cancelled.")
        return None
    except (OSError, RuntimeError, ValueError) as e:
        ASCIIColors.red(f"Failed to switch workspace: {e}")
        return None

def run_interactive(personality: LollmsPersonality, client: LollmsClient, config: CodeAgentConfig) -> int:
    _index_workspace_with_progress(personality, client)

    if not getattr(config, "continue_session", False):
        personality._conversation = []

    if config.debug:
        dump_startup_context(personality, client)

    renderer = StreamRenderer(config)

    # Use project-local prompt history for autocomplete (up-arrow) and persist it
    prompt_history_file = get_workspace_prompt_history_file(config.workspace_path)
    history = PersistentHistory(prompt_history_file)

    slash_commands = ["/exit", "/quit", "/help", "/config", "/shell", "/forget", "/skills", "/clear-history", "/clear-files", "/clear-scratchpad", "/models", "/files", "/workspace", "/load", "/unload", "/lock", "/hide", "/unhide"]
    
    # Display a safe, truncated workspace path to the user
    ws_path_display = Path(config.workspace_path).resolve()
    try:
        # Attempt to show a relative path if it's under the home directory
        ws_path_display = ws_path_display.relative_to(Path.home())
        ws_path_display = f"~/{ws_path_display}"
    except ValueError:
        pass # Keep absolute if outside home directory

    active_alias = config.active_profile_alias
    active_model = getattr(getattr(client, "llm", None), "model_name", None) or config.active_model_name
    header_lines = [
        f"[cyan]Workspace:[/cyan] {ws_path_display}",
        f"[cyan]Handbag:[/cyan]   {personality.name} [dim]({Path(config.handbag_path).name})[/dim]",
        f"[cyan]Profile:[/cyan]    {active_alias}",
        f"[cyan]Model:[/cyan]      {active_model}",
        f"[cyan]Binding:[/cyan]    {config.active_binding_name}",
        f"[dim]Commands: 'exit', 'help', 'config', 'shell', 'forget', 'skills', 'handbag', 'clear-history', 'clear-files', 'clear-scratchpad', 'workspace', 'files', 'load', 'unload', 'lock', 'hide'[/dim]"
    ]

    ctx_status = get_context_fill_status(personality, client)
    if ctx_status:
        used = ctx_status["used_tokens"]
        max_t = ctx_status["max_tokens"]
        pct = ctx_status["fill_percentage"]
        header_lines.append(f"[cyan]Context:[/cyan]    {used:,} / {max_t:,} tokens ({pct:.1f}%)")

    ASCIIColors.panel(
        "\n".join(header_lines),
        title=f"[bold green]🚀 lollms_code v{APP_VERSION} — Interactive Mode[/bold green]",
        border_style="green"
    )

    # ── 📂 WORKSPACE TELEMETRY ──
    ws_stats = get_workspace_stats(personality)
    if ws_stats["total_indexed"] > 0:
        stats_content = (
            f"[cyan]Indexed Files:[/cyan] {ws_stats['total_indexed']}\n"
            f"[cyan]Loaded in Context:[/cyan] {ws_stats['total_loaded']}"
        )
        ASCIIColors.panel(stats_content, title="[bold blue]📂 Workspace Telemetry[/bold blue]", border_style="blue")
        
        if ws_stats["loaded_files"]:
            _render_files_table(ws_stats["loaded_files"], "Pre-loaded Context Files [C]")

    while True:
        try:
            user_input = _advanced_prompt(history, slash_commands)
        except (EOFError, KeyboardInterrupt):
            ASCIIColors.cyan("\n👋 Goodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            ASCIIColors.cyan("👋 Goodbye!")
            return 0

        if user_input.lower() == "/help":
            show_interactive_help()
            continue

        if user_input.lower() == "/forget":
            ASCIIColors.red("\n  ⚠️  WARNING: You are about to PERMANENTLY DELETE ALL agent memories.")
            ASCIIColors.red("  This includes user preferences, learned facts, and episodic history.")
            ASCIIColors.yellow("  Type 'CONFIRM WIPE' to proceed, or anything else to abort.")

            try:
                confirm = input("  ❓ Confirmation> ").strip()
            except (EOFError, KeyboardInterrupt):
                ASCIIColors.yellow("\n  ❌ Wipe aborted.")
                continue

            if confirm == "CONFIRM WIPE":
                if personality and hasattr(personality, "wipe_all_memories"):
                    if personality.wipe_all_memories():
                        ASCIIColors.red("  🧠 All episodic and associative memories have been permanently wiped.")
                    else:
                        ASCIIColors.yellow("  Memory manager not initialized or failed to wipe.")
                else:
                    ASCIIColors.yellow("  Personality does not support memory wiping.")
            else:
                ASCIIColors.green("  ✅ Wipe aborted. Memories are safe.")
            continue

        if user_input.lower() in ("/handbag", "/persona"):
            ASCIIColors.rule("[bold cyan]👜 Active Handbag & Persona[/bold cyan]")
            ASCIIColors.info(f"Handbag Path: [yellow]{config.handbag_path}[/yellow]")
            ASCIIColors.info(f"Persona Name: [green]{personality.name}[/green]")
            ASCIIColors.info(f"Category:     {personality.category}")
            ASCIIColors.info(f"Description:  {personality.description or '(none)'}")
            continue

        if user_input.lower() == "/skills":
            if personality.skills_manager:
                skills = personality.skills_manager.list_skills()
                if not skills:
                    ASCIIColors.yellow("  No skills learned yet.")
                else:
                    skills_table = ASCIIColors.table(
                        "Title", "Category", "Description",
                        rows=[[s['title'], s.get('category', ''), s.get('description', '')] for s in skills],
                        title="[bold yellow]📚 Learned Skills[/bold yellow]",
                        box="round"
                    )
                    ASCIIColors.rich_print(skills_table)
            else:
                ASCIIColors.yellow("  Skills manager not initialized.")
            continue

        if user_input.lower() in ("/clear-history", "/clear"):
            personality._conversation = []
            # Clear the project-local conversation file from disk
            conv_file = get_workspace_conversation_file(config.workspace_path)
            if conv_file.exists():
                try:
                    conv_file.unlink()
                except OSError as e:
                    ASCIIColors.warning(f"Failed to remove conversation file: {e}")
            # Clear the project-local prompt history as well
            prompt_file = get_workspace_prompt_history_file(config.workspace_path)
            if prompt_file.exists():
                try:
                    prompt_file.unlink()
                except OSError as e:
                    ASCIIColors.warning(f"Failed to remove prompt history file: {e}")
            history.entries = []
            history._save()
            ASCIIColors.green("  Workspace conversation and prompt history cleared.")
            continue

        if user_input.lower() == "/clear-scratchpad":
            if hasattr(personality, "_execute_scratchpad_clear"):
                result_msg = personality._execute_scratchpad_clear()
                if "✅" in result_msg:
                    ASCIIColors.green(f"  {result_msg}")
                else:
                    ASCIIColors.red(f"  {result_msg}")
            else:
                ASCIIColors.yellow("  Scratchpad is not initialized for this workspace.")
            continue

        if user_input.lower() in ("/clear-files", "/unload-all"):
            if not hasattr(personality, '_artefact_manager') or not personality._artefact_manager:
                ASCIIColors.yellow("  Artefact system not initialized.")
                continue

            try:
                from lollms_client.lollms_artefact import ArtefactVisibility
                all_arts = personality._artefact_manager._get_all_raw()
                loaded_files = [
                    a.get("title", "") for a in all_arts
                    if a.get("visibility") == ArtefactVisibility.FULL
                    and not a.get("title", "").endswith("::images")
                ]

                if not loaded_files:
                    ASCIIColors.yellow("  No files are currently loaded in context [C].")
                    continue

                result = personality.change_file_visibility(loaded_files, "unload")
                status_str = result.get("status_str", "Action completed.")

                object.__setattr__(personality, '_last_ws_sync_time', 0.0)

                if "❌" in status_str:
                    ASCIIColors.red(f"\n  {status_str}")
                else:
                    ASCIIColors.green(f"\n  ✅ All files unloaded successfully.")

                ws_stats = get_workspace_stats(personality)
                if ws_stats["loaded_files"]:
                    ASCIIColors.rich_print("")
                    _render_files_table(ws_stats["loaded_files"], "Remaining Loaded Context Files [C]")
                else:
                    ASCIIColors.yellow("\n  📂 No files are currently loaded in context.")
            except (AttributeError, KeyError, OSError, RuntimeError, ValueError) as e:
                ASCIIColors.red(f"\n  ❌ Error unloading files: {e}")
            continue

        if user_input.lower() == "/models":
            ASCIIColors.yellow("  Model switching is managed via LollmsClient profiles in this version.")
            continue

        if user_input.lower() == "/config":
            from lollms_client.lollms_config_cli_env import build_wizard_menu, _load_existing_env_to_map, _is_back_choice
            wizard_menu, wizard_state = build_wizard_menu(
                config_map=_load_existing_env_to_map(),
                title="⚙️ Lollms Client Configuration",
                exit_text="↩ Back to Chat",
                exit_behavior="ask",
                include_save_exit=True,
            )
            while True:
                selection = wizard_menu.run()
                if _is_back_choice(selection):
                    break
                if callable(selection):
                    selection()
                if wizard_state.get("saved") or wizard_state.get("exited"):
                    break
            if wizard_state.get("saved"):
                ASCIIColors.green("  Configuration updated. Restart lollms-code for changes to take effect.")
            continue

        if user_input.lower() == "/shell":
            ASCIIColors.rule("[bold cyan]⚙️ Shell Autonomy Configuration[/bold cyan]")
            current_mode = config.shell_autonomy_level
            mode_color = "red" if current_mode == "full_access" else "green"
            ASCIIColors.info(f"Current shell autonomy level: [{mode_color}]{current_mode}[/{mode_color}]")

            if current_mode == "safe":
                ASCIIColors.red("\n  ⚠️  WARNING: Switching to 'full_access' mode grants the agent UNRESTRICTED access to your system shell.")
                ASCIIColors.red("  This means it can potentially execute destructive commands (e.g., `rm -rf`, `format`), modify system files, or install software without asking.")
                ASCIIColors.yellow("  Only enable this if you trust the agent and the task requires elevated privileges.")

                try:
                    confirm = input("\n  ❓ Type 'ENABLE FULL ACCESS' to proceed, or anything else to abort: ").strip()
                except (EOFError, KeyboardInterrupt):
                    ASCIIColors.yellow("\n  ❌ Aborted. Shell remains in 'safe' mode.")
                    continue

                if confirm == "ENABLE FULL ACCESS":
                    config.shell_autonomy_level = "full_access"
                    config.save()

                    if hasattr(client, 'tools') and hasattr(client.tools, 'mounted_libraries'):
                        if 'system_shell' in client.tools.mounted_libraries:
                            lib = client.tools.mounted_libraries['system_shell']
                            if hasattr(lib, 'init_tools_library'):
                                lib.init_tools_library({"autonomy_level": "full_access"})
                                ASCIIColors.red("\n  🔓 Shell autonomy set to 'full_access'. The agent now has unrestricted shell access.")
                            else:
                                ASCIIColors.yellow("\n  ⚠️ Config saved, but the active tool library does not support hot-reloading. Please restart lollms-code.")
                        else:
                            try:
                                client.tools.mount_tool_library('system_shell')
                                lib = client.tools.mounted_libraries['system_shell']
                                if hasattr(lib, 'init_tools_library'):
                                    lib.init_tools_library({"autonomy_level": "full_access"})
                                    ASCIIColors.red("\n  🔓 Shell library mounted and autonomy set to 'full_access'.")
                                else:
                                    ASCIIColors.yellow("\n  ⚠️ Config saved, but the active tool library does not support hot-reloading. Please restart lollms-code.")
                            except Exception as e:
                                ASCIIColors.yellow(f"\n  ⚠️ Failed to mount system_shell library: {e}. Please restart lollms-code.")
                    else:
                        ASCIIColors.yellow("\n  ⚠️ Config saved, but client tool binding is unavailable for hot-reload. Please restart lollms-code.")
                else:
                    ASCIIColors.green("\n  ✅ Aborted. Shell remains in 'safe' mode.")
            else:
                ASCIIColors.green("\n  Shell is currently in 'full_access' mode.")
                try:
                    confirm = input("\n  ❓ Switch back to 'safe' mode? (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ASCIIColors.yellow("\n  ❌ Aborted.")
                    continue

                if confirm in ("y", "yes"):
                    config.shell_autonomy_level = "safe"
                    config.save()

                    if hasattr(client, 'tools') and hasattr(client.tools, 'mounted_libraries'):
                        if 'system_shell' in client.tools.mounted_libraries:
                            lib = client.tools.mounted_libraries['system_shell']
                            if hasattr(lib, 'init_tools_library'):
                                lib.init_tools_library({"autonomy_level": "safe"})
                                ASCIIColors.green("\n  🛡️ Shell autonomy set back to 'safe'.")
                            else:
                                ASCIIColors.yellow("\n  ⚠️ Config saved, but the active tool library does not support hot-reloading. Please restart lollms-code.")
                        else:
                            ASCIIColors.yellow("\n  ⚠️ Config saved, but the 'system_shell' library is not mounted. Please restart lollms-code.")
                    else:
                        ASCIIColors.yellow("\n  ⚠️ Config saved, but client tool binding is unavailable for hot-reload. Please restart lollms-code.")
                else:
                    ASCIIColors.yellow("\n  ❌ Aborted. Shell remains in 'full_access' mode.")

            ASCIIColors.rule()
            continue

        if user_input.lower() == "/shell":
            ASCIIColors.rule("[bold cyan]⚙️ Shell Autonomy Configuration[/bold cyan]")
            current_mode = config.shell_autonomy_level
            mode_color = "red" if current_mode == "full_access" else "green"
            ASCIIColors.info(f"Current shell autonomy level: [{mode_color}]{current_mode}[/{mode_color}]")

            if current_mode == "safe":
                ASCIIColors.red("\n  ⚠️  WARNING: Switching to 'full_access' mode grants the agent UNRESTRICTED access to your system shell.")
                ASCIIColors.red("  This means it can potentially execute destructive commands (e.g., `rm -rf`, `format`), modify system files, or install software without asking.")
                ASCIIColors.yellow("  Only enable this if you trust the agent and the task requires elevated privileges.")

                try:
                    confirm = input("\n  ❓ Type 'ENABLE FULL ACCESS' to proceed, or anything else to abort: ").strip()
                except (EOFError, KeyboardInterrupt):
                    ASCIIColors.yellow("\n  ❌ Aborted. Shell remains in 'safe' mode.")
                    continue

                if confirm == "ENABLE FULL ACCESS":
                    config.shell_autonomy_level = "full_access"
                    config.save()

                    if hasattr(client, 'tools') and hasattr(client.tools, 'mounted_libraries'):
                        if 'system_shell' in client.tools.mounted_libraries:
                            lib = client.tools.mounted_libraries['system_shell']
                            if hasattr(lib, 'init_tools_library'):
                                lib.init_tools_library({"autonomy_level": "full_access"})
                                ASCIIColors.red("\n  🔓 Shell autonomy set to 'full_access'. The agent now has unrestricted shell access.")
                            else:
                                ASCIIColors.yellow("\n  ⚠️ Config saved, but the active tool library does not support hot-reloading. Please restart lollms-code.")
                        else:
                            ASCIIColors.yellow("\n  ⚠️ Config saved, but the 'system_shell' library is not mounted. Please restart lollms-code.")
                    else:
                        ASCIIColors.yellow("\n  ⚠️ Config saved, but client tool binding is unavailable for hot-reload. Please restart lollms-code.")
                else:
                    ASCIIColors.green("\n  ✅ Aborted. Shell remains in 'safe' mode.")
            else:
                ASCIIColors.green("\n  Shell is currently in 'full_access' mode.")
                try:
                    confirm = input("\n  ❓ Switch back to 'safe' mode? (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ASCIIColors.yellow("\n  ❌ Aborted.")
                    continue

                if confirm in ("y", "yes"):
                    config.shell_autonomy_level = "safe"
                    config.save()

                    if hasattr(client, 'tools') and hasattr(client.tools, 'mounted_libraries'):
                        if 'system_shell' in client.tools.mounted_libraries:
                            lib = client.tools.mounted_libraries['system_shell']
                            if hasattr(lib, 'init_tools_library'):
                                lib.init_tools_library({"autonomy_level": "safe"})
                                ASCIIColors.green("\n  🛡️ Shell autonomy set back to 'safe'.")
                            else:
                                ASCIIColors.yellow("\n  ⚠️ Config saved, but the active tool library does not support hot-reloading. Please restart lollms-code.")
                        else:
                            ASCIIColors.yellow("\n  ⚠️ Config saved, but the 'system_shell' library is not mounted. Please restart lollms-code.")
                    else:
                        ASCIIColors.yellow("\n  ⚠️ Config saved, but client tool binding is unavailable for hot-reload. Please restart lollms-code.")
                else:
                    ASCIIColors.yellow("\n  ❌ Aborted. Shell remains in 'full_access' mode.")

            ASCIIColors.rule()
            continue

        if user_input.lower() == "/files":
            ws_stats = get_workspace_stats(personality)
            if not ws_stats["loaded_files"]:
                ASCIIColors.yellow("  No files are currently loaded in context [C].")
            else:
                _render_files_table(ws_stats["loaded_files"], "Loaded Context Files [C]")
            continue
            
        cmd_parts = user_input.split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        if cmd in ("/load", "/unload", "/lock", "/hide", "/unhide"):
            if len(cmd_parts) < 2 or not cmd_parts[1].strip():
                ASCIIColors.red(f"  Usage: {cmd} <file1> [file2] ... or {cmd} all")
                continue

            action = cmd[1:]
            targets = [t.strip() for t in cmd_parts[1].replace(",", " ").split() if t.strip()]

            try:
                result = personality.change_file_visibility(targets, action)
                status_str = result.get("status_str", "Action completed.")

                if "🛑 BLOCKED" in status_str or "❌" in status_str:
                    ASCIIColors.red(f"\n  {status_str}")
                else:
                    ASCIIColors.green(f"\n  {status_str}")

                object.__setattr__(personality, '_last_ws_sync_time', 0.0)

                ws_stats = get_workspace_stats(personality)
                if ws_stats["loaded_files"]:
                    ASCIIColors.rich_print("")
                    _render_files_table(ws_stats["loaded_files"], "Remaining Loaded Context Files [C]")
                else:
                    ASCIIColors.yellow("\n  📂 No files are currently loaded in context.")
                
                _display_context_status(personality, client)
            except (AttributeError, KeyError, OSError, RuntimeError, ValueError) as e:
                ASCIIColors.red(f"\n  ❌ Error unloading files: {e}")
            continue

        if user_input.lower() == "/workspace":
            new_personality = _switch_workspace_interactive(config, client)
            if new_personality:
                personality = new_personality
                # Reload prompt history for the newly active workspace
                prompt_history_file = get_workspace_prompt_history_file(config.workspace_path)
                history = PersistentHistory(prompt_history_file)

                _index_workspace_with_progress(personality, client)
                ws_path_display = Path(config.workspace_path).resolve()
                try:
                    ws_path_display = ws_path_display.relative_to(Path.home())
                    ws_path_display = f"~/{ws_path_display}"
                except ValueError:
                    pass

                ASCIIColors.panel(
                    f"[cyan]New Workspace:[/cyan] {ws_path_display}",
                    title="[bold green]📂 Workspace Switched[/bold green]",
                    border_style="green"
                )

                ws_stats = get_workspace_stats(personality)
                if ws_stats["total_indexed"] > 0:
                    stats_content = (
                        f"[cyan]Indexed Files:[/cyan] {ws_stats['total_indexed']}\n"
                        f"[cyan]Loaded in Context:[/cyan] {ws_stats['total_loaded']}"
                    )
                    ASCIIColors.panel(stats_content, title="[bold blue]📂 Workspace Telemetry[/bold blue]", border_style="blue")

                    if ws_stats["loaded_files"]:
                        _render_files_table(ws_stats["loaded_files"], "Pre-loaded Context Files [C]")
            continue
        
        history.add(user_input)
        ASCIIColors.rule("[bold green]🤖 Agent[/bold green]")
        ASCIIColors.rich_print("") # Ensure output starts on a new line

        renderer._first_token_printed = False

        start_time = time.time()
        try:
            result = personality.chat(
                prompt=user_input,
                lollms_client=client,
                streaming_callback=renderer,
                max_reasoning_steps=config.max_reasoning_steps,
                temperature=config.temperature,
                n_predict=config.max_tokens_per_turn,
                enable_artefacts=True,
                use_internal_history=True,
                enable_shell=config.enable_shell_execution,
                enable_python_exec=True,
                enable_workspace_tools=True,
                event_mode=EventMode.FULL_CALLBACK_MODE,
                enforce_end_tag=True
            )
        except KeyboardInterrupt:
            if hasattr(client, 'cancel'):
                client.cancel()
            ASCIIColors.yellow("\n\n⚠️  Cancelled.")
            continue
        except (RuntimeError, ValueError, OSError, ConnectionError) as e:
            trace_exception(e)
            ASCIIColors.red(f"\n💥 Error: {e}")
            continue

        # Flush the renderer to ensure any unclosed tags are printed
        renderer.flush()

        # Save the updated conversation history to the project directory
        if hasattr(personality, "_project_history_file"):
            personality.save_history_to_disk(personality._project_history_file)

        if hasattr(client, 'llm') and hasattr(client.llm, 'flush_stream'):
            try:
                client.llm.flush_stream()
            except (RuntimeError, OSError) as e:
                ASCIIColors.warning(f"Failed to flush LLM stream: {e}")

        elapsed = time.time() - start_time
        ctx_h = result.get("context_health", {})
        ctx_str = ""
        if ctx_h and ctx_h.get("max_tokens", 0) > 0:
            ctx_str = f" | Ctx: {ctx_h.get('fill_percentage', 0.0):.1f}%"
        ASCIIColors.rich_print(f"\n[dim]⏱️  {elapsed:.1f}s | Rounds: {result.get('rounds', 0)} | Tools: {len(result.get('tool_calls', []))}{ctx_str}[/dim]")


def list_skills(config: CodeAgentConfig):
    skills_dir = Path(config.skills_dir)
    if not skills_dir.exists():
        ASCIIColors.yellow("No skills directory found. Run a task first to generate skills.")
        return

    mgr = SkillsManager(skills_dirs=[str(skills_dir)], mode="loadable")
    skills = mgr.list_skills()
    if not skills:
        ASCIIColors.yellow("No skills learned yet.")
        return

    skills_table = ASCIIColors.table(
        "Title", "Category", "Description",
        rows=[[s['title'], s.get('category', ''), s.get('description', '')] for s in skills],
        title="[bold yellow]📚 Learned Skills[/bold yellow]",
        box="round"
    )
    ASCIIColors.rich_print(skills_table)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lollms-code",
        description=f"lollms_code v{APP_VERSION} — Autonomous CLI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  lollms-code "Implement a REST API client with retry logic"
  lollms-code -i
  lollms-code --workspace ./myproject "add unit tests"
  lollms-code --model qwen3:32b "refactor the database layer"
  lollms-code --enable-model-switching "build a CLI tool"
  lollms-code --list-skills
""",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="The task prompt for the autonomous agent.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start in interactive REPL mode.")
    parser.add_argument("--workspace", type=str, default=None, help="Path to the workspace directory. Defaults to current working directory.")
    parser.add_argument("-hb", "--handbag", "--handbag-path", dest="handbag_path", type=str, default=None, help="Path to a custom Handbag folder containing agent resources (SOUL.md, tools, skills, memory).")
    parser.add_argument("--model", type=str, default=None, help="Model name to use.")
    parser.add_argument("--llm-binding", type=str, default=None, help="LLM binding name.")
    parser.add_argument("--host", type=str, default=None, help="Host address for remote bindings.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for gated services.")
    parser.add_argument("--context-size", type=int, default=None, help="Context window size for local models.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum reasoning steps.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens per generation turn.")
    parser.add_argument("--skills-dir", type=str, default=None, help="Directory for SKILL.md files.")
    parser.add_argument("--enable-model-switching", action="store_true", help="Allow the agent to switch models.")
    parser.add_argument("--no-shell-execution", action="store_true", help="Disable autonomous shell command execution.")
    parser.add_argument("--shell-autonomy", type=str, default="safe", choices=["safe", "full_access"], help="Autonomy level for shell execution.")
    parser.add_argument("--no-sub-agents", action="store_true", help="Disable sub-agent delegation.")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory.")
    parser.add_argument("--list-skills", action="store_true", help="List all learned skills and exit.")
    parser.add_argument("--clear-history", action="store_true", help="Clear conversation history and exit.")
    parser.add_argument("--config", action="store_true", help="Run configuration wizard and exit.")
    parser.add_argument("--config-path", type=str, default=None, dest="config_path", help="Path to a specific configuration file (.env, .json or .yaml) used by both the client and the wizard.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"lollms_code v{APP_VERSION}")
    subparsers = parser.add_subparsers(dest="command", help="Additional commands")
    subparsers.add_parser("gui", help="Launch the lollms_code GUI (NiceGUI native window).")    
    return parser


def main():
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = build_arg_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) == "gui":
        try:
            from lollms_client.apps.lollms_code.gui.main import main as gui_main
            gui_main()
            return 0
        except ImportError as e:
            ASCIIColors.red(f"Failed to import GUI dependencies: {e}")
            ASCIIColors.yellow("Please install the GUI requirements: pip install nicegui pywebview")
            return 1
        except (RuntimeError, ValueError, OSError) as e:
            trace_exception(e)
            ASCIIColors.red(f"GUI crashed: {e}")
            return 1

    config = CodeAgentConfig.load(args)

    # The CLI requires at least the LLM modality to be configured.
    if args.config or not config.is_configured(require_llm=True):
        from lollms_client.lollms_config_cli_env import run_wizard_and_save
        run_wizard_and_save(cli_env_path=args.config_path)
        config = CodeAgentConfig.load(args)
        if args.config:
            ASCIIColors.green("\n✅ Configuration saved successfully!")
            return 0

    if args.list_skills:
        list_skills(config)
        return 0

    if args.clear_history:
        conv_file = get_workspace_conversation_file(config.workspace_path)
        prompt_file = get_workspace_prompt_history_file(config.workspace_path)
        cleared_any = False
        if conv_file.exists():
            try:
                conv_file.unlink()
                cleared_any = True
            except OSError as e:
                ASCIIColors.warning(f"Failed to remove conversation file: {e}")
        if prompt_file.exists():
            try:
                prompt_file.unlink()
                cleared_any = True
            except OSError as e:
                ASCIIColors.warning(f"Failed to remove prompt history file: {e}")
        if cleared_any:
            ASCIIColors.green(f"Conversation and prompt history cleared for workspace: {config.workspace_path}")
        else:
            ASCIIColors.yellow(f"No conversation history found in workspace: {config.workspace_path}")
        return 0

    if args.interactive:
        mode = "interactive"
    elif args.prompt:
        mode = "single"
    else:
        mode = "interactive"

    try:
        client = create_client(config)
    except (ImportError, RuntimeError, ValueError, OSError, ConnectionError) as e:
        trace_exception(e)
        ASCIIColors.red(f"Failed to create LollmsClient: {e}")
        return 1

    personality = create_coding_personality(config, client)

    config.save()

    if mode == "single":
        return run_single_prompt(personality, client, args.prompt, config)
    else:
        return run_interactive(personality, client, config)


if __name__ == "__main__":
    sys.exit(main())