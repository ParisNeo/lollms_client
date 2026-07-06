# TOOL_LIBRARY_NAME: Child Agent Spawner
# TOOL_LIBRARY_DESC: Spawns a virtual child agent with a specific instruction and optional personality. The child operates in the same workspace as the parent, builds something, and returns a text output. Useful for delegating complex sub-tasks.
# TOOL_LIBRARY_ICON: 🧠

from typing import Optional, Dict, Any

TOOL_LIBRARY_NAME = "Child Agent Spawner"
TOOL_LIBRARY_DESC = "Spawns a virtual child agent with a specific instruction and optional personality. The child operates in the same workspace as the parent, builds something, and returns a text output. Useful for delegating complex sub-tasks."
TOOL_LIBRARY_ICON = "🧠"

def init_tools_library() -> None:
    """No initialization required for this tool."""
    pass

def tool_spawn_child_agent(
    instruction: str,
    personality_conditioning: Optional[str] = None,
    model_name: Optional[str] = None,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Spawns a virtual child agent to perform a sub-task.
    The child agent shares the same workspace directory as the parent, allowing it to read and write files.
    It runs a non-streaming generation and returns the final text output to the parent.

    Args:
        instruction (str): The specific task or instruction for the child agent to execute.
        personality_conditioning (str, optional): A custom system prompt or personality condition for the child. Defaults to a helpful assistant.
        model_name (str, optional): Specific model name to use for the child. Defaults to the parent's model.
        discussion_instance (Any, optional): Injected by the orchestrator. The parent LollmsDiscussion.
        lollms_client_instance (Any, optional): Injected by the orchestrator. The LollmsClient.
    """
    try:
        if not discussion_instance or not lollms_client_instance:
            return {
                "success": False,
                "error": "Orchestrator instances not injected. This tool requires advanced agentic context."
            }

        from lollms_client.lollms_discussion import LollmsDiscussion
        from ascii_colors import ASCIIColors

        ASCIIColors.info(f"[Child Agent] Spawning child agent for instruction: {instruction[:50]}...")

        # 1. Create a child discussion sharing the parent's workspace
        # We use create_new to ensure a fresh message branch without polluting the parent's DB session.
        child_discussion = LollmsDiscussion.create_new(
            lollms_client=lollms_client_instance,
            db_manager=discussion_instance.db_manager,
            workspace_path=discussion_instance.workspace_path,  # Inherit workspace
            system_prompt=personality_conditioning or "You are a helpful AI assistant specialized in executing specific sub-tasks.",
            autosave=False
        )

        # 2. Execute the child chat non-interactively
        # We disable tools for the child to prevent infinite recursive spawning.
        # The child operates purely as a text/code generator in this context.
        child_response = child_discussion.chat(
            user_message=instruction,
            tools=None,
            add_user_message=True,
            stream=False,
            enable_artefacts=True,  # Allow child to create artifacts in the shared workspace
            enable_show_tools=False,
            enable_extract_artefact=False,
            enable_final_answer=False,
            enable_request_clarification=False,
            enable_repl_tools=False,
            enable_inline_widgets=False,
            enable_notes=False,
            enable_skills=False,
            enable_forms=False,
            enable_books=False,
            enable_presentations=False,
            enable_silent_artifact_explanation=True,
            model_name=model_name,
            max_reasoning_steps=5  # Limit child reasoning to prevent infinite loops
        )

        # 3. Extract the final text output
        if child_response and "ai_message" in child_response:
            final_text = child_response["ai_message"].content
            ASCIIColors.success(f"[Child Agent] Child agent completed task.")
            
            return {
                "success": True,
                "output": final_text,
                "prompt_injection": f"\n\n=== 🧠 CHILD AGENT REPORT ===\nThe child agent completed the task: '{instruction}'.\nHere is its output:\n\n{final_text}\n=== END REPORT ==="
            }
        else:
            return {
                "success": False,
                "error": "Child agent did not produce a valid response."
            }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Child agent spawning failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
