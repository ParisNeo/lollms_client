from lollms_client import LollmsDiscussion, LollmsClient

# --- Create a discussion instance ---
discussion = LollmsDiscussion(LollmsClient())

# --- Add artefacts ---
discussion.add_artefact("spec_doc", "Technical specification text v1")
discussion.add_artefact("design_notes", "Design notes content v2", 2)

# --- Load multiple artefacts into the data zone ---
discussion.load_artefact_into_data_zone("spec_doc")
discussion.load_artefact_into_data_zone("design_notes", version=2)

# --- Check which artefacts are loaded ---
print(discussion.is_artefact_in_data_zone("spec_doc"))      # True
print(discussion.is_artefact_in_data_zone("design_notes"))  # True

# --- View the current discussion data zone ---
print(discussion.discussion_data_zone)
"""
--- Document: spec_doc v1 ---
Technical specification text v1
--- End Document: spec_doc ---

--- Document: design_notes v2 ---
Design notes content v2
--- End Document: design_notes ---
"""

# --- Export the current context as a new artefact ---
exported = discussion.export_as_artefact("full_context_snapshot", version=1)
print("Exported artefact:", exported)

# --- Unload one artefact while keeping the other ---
discussion.unload_artefact_from_data_zone("spec_doc")
print(discussion.is_artefact_in_data_zone("spec_doc"))      # False
print(discussion.is_artefact_in_data_zone("design_notes"))  # True

# --- View the updated discussion data zone ---
print(discussion.discussion_data_zone)
"""
--- Document: design_notes v2 ---
Design notes content v2
--- End Document: design_notes ---
"""
