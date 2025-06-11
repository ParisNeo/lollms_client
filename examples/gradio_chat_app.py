# final_working_chat_app.py

import sys
import os
import json
import gradio as gr
import requests
from typing import List, Dict, Optional, Tuple

# --- Dependency Installation ---
try:
    import pipmaster as pm
    print("Pipmaster found. Ensuring dependencies are installed...")
    pm.ensure_packages(["gradio", "requests", "ascii_colors"])
except ImportError:
    pass

# --- Import Core Components ---
try:
    from lollms_client import LollmsClient
    from lollms_client.lollms_discussion import LollmsDiscussion 
    from ascii_colors import ASCIIColors
except ImportError as e:
    print(f"\nFATAL: A required library is missing.\nPlease ensure lollms-client and ascii_colors are installed.")
    print(f"Error: {e}"); sys.exit(1)

# --- Standalone Helper Functions for LollmsDiscussion ---
def export_for_chatbot(discussion: Optional[LollmsDiscussion]) -> List[Dict[str, str]]:
    if not discussion: return []
    branch = discussion.get_branch(discussion.active_branch_id)
    return [{"role": discussion.participants.get(msg.sender, "user"), "content": msg.content} for msg in branch]

def render_discussion_tree(discussion: Optional[LollmsDiscussion]) -> str:
    if not discussion or not discussion.messages: return "No messages yet."
    tree_markdown = "### Discussion Tree\n\n"; root_ids = [msg.id for msg in discussion.messages if msg.parent_id is None]
    def _render_node(node_id: str, depth: int) -> str:
        node = discussion.message_index.get(node_id)
        if not node: return ""
        is_active = "  <span class='activ'>[ACTIVE]</span>" if node.id == discussion.active_branch_id else ""
        line = f"{'    ' * depth}- **{node.sender}**: _{node.content.replace(chr(10), ' ').strip()[:60]}..._{is_active}\n"
        for child_id in discussion.children_index.get(node.id, []): line += _render_node(child_id, depth + 1)
        return line
    for root_id in root_ids: tree_markdown += _render_node(root_id, 0)
    return tree_markdown

def get_message_choices(discussion: Optional[LollmsDiscussion]) -> List[tuple]:
    if not discussion: return []
    return [(f"{msg.sender}: {msg.content[:40]}...", msg.id) for msg in discussion.messages]

# --- Configuration & File Management ---
CONFIG_FILE = "config.json"; DISCUSSIONS_DIR = "discussions"; os.makedirs(DISCUSSIONS_DIR, exist_ok=True)
DEFAULT_CONFIG = {"binding_name": "ollama", "model_name": "mistral:latest", "host_address": "http://localhost:11434", "openai_api_key": "", "openai_model_name": "gpt-4o"}
def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: ASCIIColors.info(f"Loaded config from {CONFIG_FILE}"); return json.load(f)
        except: ASCIIColors.warning(f"Could not load {CONFIG_FILE}, using defaults."); return DEFAULT_CONFIG
    return DEFAULT_CONFIG
def save_config(config: Dict):
    with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=2); ASCIIColors.green(f"Saved config to {CONFIG_FILE}")

# --- LollmsClient & Discussion Management ---
def create_lollms_client(config: Dict) -> Optional[LollmsClient]:
    try:
        if config["binding_name"] == "ollama": client = LollmsClient(binding_name="ollama", host_address=config["host_address"], model_name=config["model_name"])
        elif config["binding_name"] == "openai":
            if not config.get("openai_api_key"): gr.Warning("OpenAI API key missing."); return None
            client = LollmsClient(binding_name="openai", model_name=config["openai_model_name"], service_key=config["openai_api_key"])
        else: gr.Warning(f"Unsupported binding: {config['binding_name']}"); return None
        ASCIIColors.green("LollmsClient created successfully."); return client
    except Exception as e: gr.Error(f"Failed to create LollmsClient: {e}"); return None
def get_discussions_list() -> List[str]: return sorted([f for f in os.listdir(DISCUSSIONS_DIR) if f.endswith(".yaml")])
def load_discussion(filename: str, client: LollmsClient) -> Optional[LollmsDiscussion]:
    if not client: ASCIIColors.warning("Cannot load discussion: client is not initialized."); return None
    try:
        discussion = LollmsDiscussion(client); discussion.load_from_disk(os.path.join(DISCUSSIONS_DIR, filename))
        ASCIIColors.info(f"Loaded discussion: {filename}"); return discussion
    except Exception as e: gr.Error(f"Failed to load discussion {filename}: {e}"); return None
def list_ollama_models(host: str) -> List[str]:
    try:
        r = requests.get(f"{host}/api/tags"); r.raise_for_status(); return [m["name"] for m in r.json().get("models", [])]
    except: gr.Warning(f"Could not fetch models from {host}."); return []

# --- Gradio UI & Logic ---
with gr.Blocks(theme=gr.themes.Soft(), css=".activ { font-weight: bold; color: #FF4B4B; }") as demo:
    client_state = gr.State()
    discussion_state = gr.State()
    
    gr.Markdown("# 🌿 Multi-Branch Discussion App")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Session & Branch Management")
            discussion_selector = gr.Dropdown(label="Load Discussion", interactive=True)
            new_discussion_name = gr.Textbox(label="New Discussion Name", placeholder="Enter name and press Enter...")
            delete_discussion_button = gr.Button("Delete Current Discussion", variant="stop")
            branch_selector = gr.Dropdown(label="Select Message to Branch From", interactive=True)
            discussion_tree_display = gr.Markdown("No discussion loaded.")
        with gr.Column(scale=2):
            with gr.Accordion("⚙️ Settings & System Prompt", open=False):
                system_prompt_input = gr.Textbox(label="System Prompt", lines=3, interactive=True)
                with gr.Row():
                    binding_selector = gr.Radio(["ollama", "openai"], label="AI Binding")
                    save_settings_button = gr.Button("Save Settings & Re-initialize", variant="primary")
                with gr.Group(visible=True) as ollama_settings_group:
                    ollama_host_input = gr.Textbox(label="Ollama Host Address"); ollama_model_selector = gr.Dropdown(label="Ollama Model", interactive=True); refresh_ollama_button = gr.Button("Refresh Ollama Models")
                with gr.Group(visible=False) as openai_settings_group:
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", type="password"); openai_model_selector = gr.Dropdown(choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], label="OpenAI Model", interactive=True)
            chatbot = gr.Chatbot(label="Conversation", height=600, type="messages")
            user_input = gr.Textbox(show_label=False, placeholder="Type your message here...", lines=3)
            send_button = gr.Button("Send", variant="primary")

    # --- Event Handler Functions ---
    def on_load():
        config = load_config(); client = create_lollms_client(config)
        discussions_list = get_discussions_list(); discussion = load_discussion(discussions_list[0], client) if discussions_list else (LollmsDiscussion(client) if client else None)
        active_discussion_file = discussions_list[0] if discussions_list else None
        
        history = export_for_chatbot(discussion) if discussion else [{"role": "assistant", "content": "Welcome! Configure client in Settings and create a new chat."}]
        tree = render_discussion_tree(discussion); branch_choices = get_message_choices(discussion)
        sys_prompt = discussion.system_prompt if discussion else ""
        active_branch_id = discussion.active_branch_id if discussion else None
        is_ollama = config['binding_name'] == 'ollama'; ollama_models = list_ollama_models(config['host_address']) if is_ollama and client else []

        return (client, discussion, gr.update(choices=discussions_list, value=active_discussion_file), config['binding_name'], 
                gr.update(visible=is_ollama), gr.update(visible=not is_ollama), config['host_address'], 
                gr.update(choices=ollama_models, value=config.get('model_name')), config['openai_api_key'], 
                config.get('openai_model_name'), sys_prompt, history, tree, gr.update(choices=branch_choices, value=active_branch_id))

    def handle_save_settings(binding, host, ollama_model, openai_key, openai_model):
        config = {"binding_name": binding, "host_address": host, "model_name": ollama_model, "openai_api_key": openai_key, "openai_model_name": openai_model}
        save_config(config); gr.Info("Settings saved! Reloading application..."); return on_load()

    def handle_new_discussion(client, name):
        if not client: gr.Error("Client not initialized."); return (gr.skip(),) * 5
        if not name.strip(): gr.Warning("Provide a name."); return (gr.skip(),) * 5
        filename = f"{name.strip().replace(' ', '_')}.yaml"
        if os.path.exists(os.path.join(DISCUSSIONS_DIR, filename)): gr.Warning(f"Discussion '{name}' already exists."); return (gr.skip(),) * 5
        discussion = LollmsDiscussion(client); discussion.set_participants({"user": "user", "assistant": "assistant"})
        discussion.add_message("assistant", f"This is the beginning of '{name}'."); discussion.save_to_disk(os.path.join(DISCUSSIONS_DIR, filename))
        return discussion, gr.update(choices=get_discussions_list(), value=filename), export_for_chatbot(discussion), render_discussion_tree(discussion), gr.update(choices=get_message_choices(discussion), value=discussion.active_branch_id)

    def handle_load_discussion(client, filename):
        if not client: gr.Error("Client not initialized."); return (gr.skip(),) * 5
        if not filename: return (gr.skip(),) * 5
        discussion = load_discussion(filename, client)
        if not discussion: return (gr.skip(),) * 5
        return discussion, discussion.system_prompt or "", export_for_chatbot(discussion), render_discussion_tree(discussion), gr.update(choices=get_message_choices(discussion), value=discussion.active_branch_id)

    def handle_delete_discussion(filename):
        if not filename: gr.Warning("No discussion selected to delete."); return (gr.skip(),) * 14
        try:
            os.remove(os.path.join(DISCUSSIONS_DIR, filename)); ASCIIColors.red(f"Deleted discussion: {filename}"); gr.Info(f"Deleted {filename}.")
            return on_load()
        except Exception as e:
            gr.Error(f"Failed to delete file: {e}"); return (gr.skip(),) * 14

    def handle_chat_submit(client, discussion, user_text, history, filename):
        if not client: gr.Error("Client not initialized."); return
        if not discussion: gr.Error("No discussion loaded."); return
        if not user_text.strip(): return
        if not filename: gr.Error("No active discussion file. Cannot save."); return
        
        parent_id = discussion.active_branch_id
        discussion.add_message(sender="user", content=user_text, parent_id=parent_id)
        history.append({"role": "user", "content": user_text}); history.append({"role": "assistant", "content": ""})
        yield history
        
        full_response = ""
        try:
            # The callback must return True to continue the stream.
            for chunk in client.chat(discussion, stream=True, streaming_callback=lambda c,t: True):
                full_response += chunk; history[-1]["content"] = full_response; yield history
            discussion.add_message(sender="assistant", content=full_response); discussion.save_to_disk(os.path.join(DISCUSSIONS_DIR, filename))
        except Exception as e:
            full_response = f"An error occurred: {e}"; gr.Error(full_response); history[-1]["content"] = full_response
            discussion.add_message(sender="assistant", content=f"ERROR: {full_response}")

    def on_chat_finish(discussion):
        # This function updates non-streaming components after the chat is done
        if not discussion: return gr.skip(), gr.skip()
        return render_discussion_tree(discussion), gr.update(choices=get_message_choices(discussion), value=discussion.active_branch_id)

    def handle_branch_change(discussion, selected_id):
        if not discussion or not selected_id: return gr.skip(), gr.skip()
        discussion.set_active_branch(selected_id)
        return discussion, export_for_chatbot(discussion)

    # --- Wire up Components ---
    outputs_on_load = [client_state, discussion_state, discussion_selector, binding_selector, ollama_settings_group, openai_settings_group, ollama_host_input, ollama_model_selector, openai_api_key_input, openai_model_selector, system_prompt_input, chatbot, discussion_tree_display, branch_selector]
    demo.load(on_load, outputs=outputs_on_load)
    save_settings_button.click(handle_save_settings, [binding_selector, ollama_host_input, ollama_model_selector, openai_api_key_input, openai_model_selector], outputs_on_load)
    binding_selector.change(lambda x: (gr.update(visible=x=='ollama'), gr.update(visible=x=='openai')), binding_selector, [ollama_settings_group, openai_settings_group])
    refresh_ollama_button.click(list_ollama_models, ollama_host_input, ollama_model_selector)
    system_prompt_input.blur(lambda d,t,f: d.set_system_prompt(t) and d.save_to_disk(os.path.join(DISCUSSIONS_DIR,f)) if d and f else None, [discussion_state, system_prompt_input, discussion_selector], [])

    new_discussion_name.submit(handle_new_discussion, [client_state, new_discussion_name], [discussion_state, discussion_selector, chatbot, discussion_tree_display, branch_selector]).then(lambda: "", outputs=[new_discussion_name])
    discussion_selector.change(handle_load_discussion, [client_state, discussion_selector], [discussion_state, system_prompt_input, chatbot, discussion_tree_display, branch_selector])
    delete_discussion_button.click(handle_delete_discussion, [discussion_selector], outputs_on_load)

    # --- CORRECTED WIRING FOR CHAT ---
    chat_stream_event = user_input.submit(
        fn=handle_chat_submit, 
        inputs=[client_state, discussion_state, user_input, chatbot, discussion_selector], 
        outputs=[chatbot],
    )
    # After the stream from handle_chat_submit is done, its input (discussion_state) will be updated.
    # We can then pass that state to on_chat_finish.
    chat_stream_event.then(
        fn=on_chat_finish, 
        inputs=[discussion_state], # The input is the state object that was modified by the previous function
        outputs=[discussion_tree_display, branch_selector]
    ).then(lambda: "", outputs=[user_input])
    
    send_button_stream_event = send_button.click(
        fn=handle_chat_submit, 
        inputs=[client_state, discussion_state, user_input, chatbot, discussion_selector], 
        outputs=[chatbot]
    )
    send_button_stream_event.then(
        fn=on_chat_finish, 
        inputs=[discussion_state],
        outputs=[discussion_tree_display, branch_selector]
    ).then(lambda: "", outputs=[user_input])

    branch_selector.change(handle_branch_change, [discussion_state, branch_selector], [discussion_state, chatbot])

if __name__ == "__main__":
    demo.launch()