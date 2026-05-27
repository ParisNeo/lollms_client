#!/usr/bin/env python3
"""
test_form_generation.py
=============================================================================
A comprehensive example showcasing how the interactive form system works.
Fires a chat turn that requests the model to gather structured user information
using a custom <lollms_form> block, demonstrating:
  1. Form instruction injection.
  2. Interactive form descriptor construction and parsing.
  3. Answer injection via submit_form_response() [1].
=============================================================================
"""

import sys
from pathlib import Path

# Ensure correct workspace imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion


def main():
    print("=" * 80)
    print("📋 LOLLMS INTERACTIVE FORM GENERATION EXAMPLE")
    print("=" * 80)

    # Initialize client for simulation
    client = LollmsClient(user_name="ParisNeo", ai_name="Lollms")
    discussion = LollmsDiscussion(lollmsClient=client)

    # Inject Form Instructions
    discussion.system_prompt = discussion._build_form_instructions()

    user_prompt = (
        "I want to configure a custom API endpoint. "
        "Please create an interactive form to gather my name, the endpoint route, "
        "the HTTP method (GET, POST, PUT, DELETE), and if authentication is required."
    )

    print(f"\nUser Prompt:\n{user_prompt}\n")

    print("Generating Form Turn...")
    print("-" * 60)
    
    # We simulate the AI producing the form
    ai_response = (
        "Certainly! I have generated a form to collect your custom API configuration details. "
        "Please fill it in so I can generate the correct backend code.\n\n"
        '<lollms_form title="Custom API Configuration" description="Specify your endpoint parameters" submit_label="Generate Code">\n'
        '  <field name="author_name" label="Author Name" type="text" placeholder="e.g. ParisNeo" required="true" />\n'
        '  <field name="endpoint_route" label="API Route Path" type="text" placeholder="e.g. /api/users" required="true" />\n'
        '  <field name="http_method" label="HTTP Method" type="select" options="GET,POST,PUT,DELETE" default="GET" />\n'
        '  <field name="auth_required" label="Require Authentication" type="checkbox" default="false" />\n'
        '</lollms_form>'
    )
    
    # Add to discussion tree
    user_msg = discussion.add_message(sender="user", content=user_prompt)
    ai_msg = discussion.add_message(sender="assistant", content=ai_response)
    
    print(ai_response)
    print("-" * 60)

    # Locate the parsed form inside our pending forms store
    pending_forms = discussion._get_pending_forms()
    print(f"\nDiscovered Pending Forms in Session: {list(pending_forms.keys())}")
    
    for form_id, form_desc in pending_forms.items():
        print(f"  • Form ID: {form_id}")
        print(f"    Title: {form_desc['title']}")
        print(f"    Fields collected:")
        for field in form_desc["fields"]:
            print(f"      - {field['name']} ({field['type']}): \"{field['label']}\"")

    # Simulate user filling in and submitting the form
    simulated_answers = {
        "author_name": "ParisNeo",
        "endpoint_route": "/api/v1/auth/login",
        "http_method": "POST",
        "auth_required": True
    }
    
    print(f"\nSimulating User Submission of Form response...")
    form_id = list(pending_forms.keys())[0]
    discussion.submit_form_response(form_id, simulated_answers)

    print("\nUpdated Conversation History (Answers Injected):")
    print("-" * 60)
    for m in discussion.get_branch(discussion.active_branch_id):
        print(f"[{m.sender.upper()}]:\n{m.content}\n")
    print("-" * 60)
    
    print("✅ Form generation and submission simulation completed successfully.")


if __name__ == "__main__":
    main()
