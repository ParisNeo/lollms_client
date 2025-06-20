# research_app_final.py

import os
import json
import shutil
import gradio as gr
from pathlib import Path
# Use the correct, specified import style
from lollms_client import LollmsClient, LollmsDiscussion, MSG_TYPE, LollmsDataManager
from ascii_colors import ASCIIColors
from sqlalchemy import Column, String

# --- 1. Define Application-Specific Schema ---
# This allows applications to store and query their own metadata in the database.
class ResearchDiscussionMixin:
    project_name = Column(String(100), index=True, nullable=False)

class ResearchMessageMixin:
    pass

# --- 2. Global Setup: Client and Database ---
# These are initialized once and used throughout the app's lifecycle.
try:
    lc = LollmsClient("ollama", model_name="mistral-nemo:latest")
    db_manager = LollmsDataManager(
        db_path="sqlite:///research_projects_gradio.db",
        discussion_mixin=ResearchDiscussionMixin,
        message_mixin=ResearchMessageMixin,
        encryption_key="a-super-secret-key-for-the-gradio-app"
    )
    print("✅ Client and Database initialized successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not initialize services. Is Ollama running? Error: {e}")
    lc = None
    db_manager = None

# --- 3. UI Helper Functions ---
# These functions connect the Gradio UI to our discussion library's backend logic.

def _get_discussion_list():
    """Helper to fetch and format the list of discussions for the dropdown."""
    if not db_manager: return []
    discussions = db_manager.list_discussions()
    return [(d.get('project_name', d['id']), d['id']) for d in discussions]

def _format_chatbot_history(discussion: LollmsDiscussion):
    """Converts a discussion's active branch into Gradio's chatbot format."""
    history = []
    if not discussion: return history
    
    branch = discussion.get_branch(discussion.active_branch_id)
    # This robust loop correctly pairs user and AI messages.
    i = 0
    while i < len(branch):
        if branch[i]['sender_type'] == 'user':
            user_msg = branch[i]['content']
            if i + 1 < len(branch) and branch[i+1]['sender_type'] == 'assistant':
                ai_msg = branch[i+1]['content']
                history.append((user_msg, ai_msg))
                i += 2
            else:
                history.append((user_msg, None))
                i += 1
        else:
            ai_msg = branch[i]['content']
            history.append((None, ai_msg))
            i += 1
    return history

# --- 4. Gradio UI Event Handler Functions ---

def handle_new_discussion(name: str):
    """Called when the 'New Project' button is clicked."""
    if not name.strip():
        gr.Warning("Project name cannot be empty.")
        return gr.Dropdown(choices=_get_discussion_list()), None, []
    
    discussion = LollmsDiscussion.create_new(
        lollms_client=lc,
        db_manager=db_manager,
        project_name=name.strip(),
        autosave=True
    )
    discussion.set_system_prompt(f"This is a research project about {name.strip()}. Be helpful and concise, but use <think> tags to outline your process before answering.")
    discussion.set_participants({"user":"user", "assistant":"assistant"})
    
    gr.Info(f"Project '{name.strip()}' created!")
    
    return gr.Dropdown(choices=_get_discussion_list(), value=discussion.id), discussion.id, []

def handle_load_discussion(discussion_id: str):
    """Called when a discussion is selected from the dropdown."""
    if not discussion_id:
        return None, []
    
    discussion = db_manager.get_discussion(lollms_client=lc, discussion_id=discussion_id)
    chatbot_history = _format_chatbot_history(discussion)
    
    return chatbot_history

def handle_delete_discussion(discussion_id: str):
    """Called when the 'Delete' button is clicked."""
    if not discussion_id:
        gr.Warning("No project selected to delete.")
        return gr.Dropdown(choices=_get_discussion_list()), None, []
        
    db_manager.delete_discussion(discussion_id)
    gr.Info("Project deleted.")
    
    return gr.Dropdown(choices=_get_discussion_list(), value=None), None, []

def handle_chat_submit(user_input: str, chatbot_history: list, discussion_id: str, show_thoughts: bool):
    """The main chat handler, called on message submit. Uses a generator for streaming."""
    if not discussion_id:
        gr.Warning("Please select or create a project first.")
        return "", chatbot_history
    if not user_input.strip():
        return "", chatbot_history
        
    discussion = db_manager.get_discussion(lollms_client=lc, discussion_id=discussion_id, autosave=True)
    
    chatbot_history.append((user_input, None))
    yield "", chatbot_history
    
    ai_message_buffer = ""
    
    def stream_to_chatbot(token: str, msg_type: MSG_TYPE):
        nonlocal ai_message_buffer
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            ai_message_buffer += token
            chatbot_history[-1] = (user_input, ai_message_buffer)
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            thought_html = f"<p style='color:magenta;'><i>{token}</i></p>"
            chatbot_history[-1] = (user_input, ai_message_buffer + thought_html)
        return True

    discussion.chat(
        user_message=user_input,
        show_thoughts=show_thoughts,
        streaming_callback=stream_to_chatbot
    )
    
    yield "", chatbot_history

def handle_regenerate(chatbot_history: list, discussion_id: str, show_thoughts: bool):
    """Called to regenerate the last AI response."""
    if not discussion_id:
        gr.Warning("Please select a project first.")
        return chatbot_history
    if not chatbot_history or chatbot_history[-1][1] is None:
        gr.Warning("Nothing to regenerate.")
        return chatbot_history

    discussion = db_manager.get_discussion(lollms_client=lc, discussion_id=discussion_id, autosave=True)

    chatbot_history.pop()
    user_input_for_ui = chatbot_history[-1][0] if chatbot_history else ""
    chatbot_history.append((user_input_for_ui, None))
    yield chatbot_history

    ai_message_buffer = ""

    def stream_to_chatbot(token: str, msg_type: MSG_TYPE):
        nonlocal ai_message_buffer
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            ai_message_buffer += token
            chatbot_history[-1] = (user_input_for_ui, ai_message_buffer)
        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
            thought_html = f"<p style='color:magenta;'><i>{token}</i></p>"
            chatbot_history[-1] = (user_input_for_ui, ai_message_buffer + thought_html)
        return True

    discussion.regenerate_branch(
        show_thoughts=show_thoughts,
        streaming_callback=stream_to_chatbot
    )
    
    yield chatbot_history

# --- 5. Build and Launch the Gradio App ---
with gr.Blocks(theme=gr.themes.Soft(), title="Lollms Discussion Manager") as demo:
    discussion_id_state = gr.State(None)

    gr.Markdown("# Lollms Discussion Manager")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## Projects")
            discussion_dd = gr.Dropdown(
                choices=_get_discussion_list(),
                label="Select Project",
                interactive=True
            )
            with gr.Accordion("Manage Projects", open=False):
                new_discussion_name = gr.Textbox(label="New Project Name", placeholder="Enter name and press button")
                with gr.Row():
                    new_discussion_btn = gr.Button("➕ New")
                    delete_discussion_btn = gr.Button("❌ Delete")
            
            gr.Markdown("---")
            gr.Markdown("## Options")
            show_thoughts_check = gr.Checkbox(label="Show AI Thoughts", value=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, render_markdown=True)
            with gr.Row():
                user_input_tb = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    scale=5,
                    autofocus=True
                )
                send_btn = gr.Button("✉️ Send", variant="primary", scale=1)
                regenerate_btn = gr.Button("🔄 Regenerate", scale=1)

    # --- Event Handling: Wiring the UI to the backend functions ---

    new_discussion_btn.click(
        fn=handle_new_discussion,
        inputs=[new_discussion_name],
        outputs=[discussion_dd, discussion_id_state, chatbot]
    ).then(fn=lambda: "", inputs=None, outputs=[new_discussion_name])

    delete_discussion_btn.click(
        fn=handle_delete_discussion,
        inputs=[discussion_id_state],
        outputs=[discussion_dd, discussion_id_state, chatbot]
    )

    discussion_dd.change(
        fn=handle_load_discussion,
        inputs=[discussion_dd],
        outputs=[chatbot]
    ).then(lambda x: x, inputs=[discussion_dd], outputs=[discussion_id_state])

    user_input_tb.submit(
        fn=handle_chat_submit,
        inputs=[user_input_tb, chatbot, discussion_id_state, show_thoughts_check],
        outputs=[user_input_tb, chatbot]
    )
    send_btn.click(
        fn=handle_chat_submit,
        inputs=[user_input_tb, chatbot, discussion_id_state, show_thoughts_check],
        outputs=[user_input_tb, chatbot]
    )
    regenerate_btn.click(
        fn=handle_regenerate,
        inputs=[chatbot, discussion_id_state, show_thoughts_check],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    if lc is None or db_manager is None:
        print("Could not start Gradio app due to initialization failure.")
    else:
        demo.launch()
        print("\n--- App closed. Cleaning up. ---")
        if os.path.exists("research_projects_gradio.db"):
            os.remove("research_projects_gradio.db")