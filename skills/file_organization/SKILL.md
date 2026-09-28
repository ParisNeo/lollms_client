---
name: file_organization
title: File and Directory Organization
category: workspace_management
tags: [file_organization, classes, mapping, directory_structure, sub_agents, cleanup, migration, file_and_folder_organization]
required_tools: [tool_organize_files_from_plan]
description: Four-phase autonomous methodology for scanning directories, building classes.md and mapping.yaml, inspecting ambiguous files with sub-agents, requesting user confirmation, and executing folder migrations.
visibility: loadable
---

# File and Folder Organization Skill

This skill governs the systematic classification, taxonomy generation, user validation, and execution of workspace and directory restructuring.

---

## 📦 Batch Sizing & Atomic Folder Invariants (MANDATORY)

### 1. Atomic Directories: MOVE AS WHOLE FOLDERS (DO NOT CRAWL SUBFILES)
- Any directory present at the root of the workspace (e.g. `presentation Line/`, `Rapper artworks/`, `projet_lollms_1/`, `regissong2/`) is an **ATOMIC UNIT**.
- **RULE**: You MUST map and move the entire directory as a single folder unit:
  `source: "presentation Line" -> target: "documents/presentations/presentation Line"`
  `source: "Rapper artworks" -> target: "media/images/artworks/Rapper artworks"`
- ❌ **STRICTLY FORBIDDEN**: NEVER crawl into subfolders (e.g. `presentation Line/originale/mc0.png`). Do NOT process or list internal subfolder files! Leave the internal contents of subfolders completely intact.

### 2. Meaningful Batch Sizing (40 to 50 Items per Batch)
- When organizing a directory with hundreds of items:
  - ❌ **STRICTLY FORBIDDEN**: Mapping only 5, 7, or 10 items when hundreds of files exist is a failure.
  - ✅ **MANDATORY**: Each batch MUST process **between 40 and 50 items** (or all remaining items if less than 50 remain).
  - First, map all root-level directories as atomic units.
  - Second, fill the rest of the batch up to 50 items using the loose root files (e.g. root `.png`, `.wav`, `.mp4` files).

---

## 🎯 The Four-Phase Protocol Overview
When instructed to organize, clean up, or structure a workspace or directory, you MUST follow this four-phase sequence:
1. **Phase 1 (Scan, Taxonomy & Batch 1 Mapping)**: Scan root items, identify root directories to move as whole units, construct the taxonomy in `classes.md`, and generate `mapping.yaml` for **Batch 1 (40–50 items)**.
2. **Phase 2 (Ambiguity Resolution)**: Inspect files in the current batch with unclear purposes via `tool_inspect_image` and update the mapping.
3. **Phase 3 (User Confirmation Gate for Batch)**: Present the batch plan (40–50 items) to the user and halt with `<done/>` to wait for explicit approval.
4. **Phase 4 (Execution & Next Batch Transition)**: Upon user confirmation, call `tool_organize_files_from_plan` to move Batch 1. Once verified, announce Batch 2.

---

## 🔍 Phase 1: Workspace Scan, Taxonomy & Bounded Batch Mapping

### 1. Root-Level Discovery & Batch Sizing
1. Inspect the root of the workspace. Separate root directories from loose root files.
2. If total items > 50: **Announce the batch plan**: e.g., *"Found 314 items to organize. Moving existing project folders as whole units and processing loose files in batches of 40–50 files. Preparing Batch 1 (50 items)..."*

### 2. Move Root Folders as Atomic Units
- Map each root folder as a whole unit to its target category. Do not unpack it!

### 3. Generate a Comprehensive, Granular Taxonomy in `classes.md`
Design a thorough, hierarchical classification taxonomy tailored to ALL files and folders discovered in the workspace.

🚨 **CRITICAL SYNTAX MANDATE**: Never wrap `<artifact>` or `<tool>` tags inside markdown code blocks (```). Emit naked XML tags directly starting on a new line:

<artifact name="classes.md" type="document">
# Target Taxonomy Hierarchy (classes.md)

## 1. media/
- media/images/
  - media/images/artworks/ (AI generated art, illustrations)
  - media/images/photos/ (portraits, photography)
- media/audio/
  - media/audio/podcasts/ (podcasts, voice recordings)
  - media/audio/music/ (songs, music tracks)
- media/videos/
  - media/videos/documentaries/ (AI topics, tutorials)

## 2. documents/
- documents/presentations/ (slide decks, presentation suites)
- documents/reports/

## 3. software/
- software/utilities/
</artifact>

### 4. Generate Bounded `mapping.yaml` (MAXIMUM 50 FILES PER BATCH)
Map **at most 50 files** for the active batch into a clean, machine-executable YAML (or JSON) migration plan.
YAML is concise, token-efficient, and can be directly executed by `tool_organize_files_from_plan`.

<artifact name="mapping.yaml" type="document">
# File Migration Plan — Batch 1 (40 to 50 Items)
batch: 1
total_batch_items: 50
mappings:
  # ── 1. Root Directories (Moved as Whole Atomic Units) ──
  - source: "presentation Line"
    target: "documents/presentations/presentation Line"
    description: "Atomic presentation suite directory"
  - source: "Rapper artworks"
    target: "media/images/artworks/Rapper artworks"
    description: "Atomic DALL-E artwork album directory"
  - source: "regissong2"
    target: "media/audio/music/regissong2"
    description: "Atomic music and audio project suite"
  - source: "projet_lollms_1"
    target: "software/projects/projet_lollms_1"
    description: "Atomic project directory"

  # ── 2. Loose Root Media Files (List up to 45-48 loose files to reach 50 total items) ──
  - source: "00001-2236797388.png"
    target: "media/images/artworks/00001-2236797388.png"
    description: "AI generated artwork"
  - source: "00010-149492033.png"
    target: "media/images/artworks/00010-149492033.png"
    description: "AI artwork"
  - source: "1716983091398.jpg"
    target: "media/images/photos/1716983091398.jpg"
    description: "Photo"
  - source: "AI thoughts.wav"
    target: "media/audio/podcasts/AI thoughts.wav"
    description: "Audio podcast"
  - source: "4_Ways_Gemini_is_Changing_AI.mp4"
    target: "media/videos/documentaries/4_Ways_Gemini_is_Changing_AI.mp4"
    description: "AI video documentary"
  # (Continue listing all remaining loose root files up to 50 items for this batch!)
</artifact>

---

## 🔬 Phase 2: Visual & Content Inspection of Ambiguous Files

For any file whose purpose or subject cannot be reliably determined by filename alone:
1. **Vision-Language Inspection (VLM)**: You are equipped with `tool_inspect_image`! If an image file (e.g. `00010-149492033.png`, `anjelina.jpg`) is ambiguous:
   - Call `tool_inspect_image(image_path="<path>", query="What is depicted in this image and what category does it belong to?")`.
   - The vision model will inspect the image directly and describe its contents (portrait, anime character, landscape, UI mockup, artwork) so you can classify it with high precision!
2. **Text / Data Inspection**: Use `<unlock_file>path/to/file.ext</unlock_file>` or `tool_read_file` to read the first few lines of ambiguous text files, scripts, or json data.
3. **Sub-Agent Inspection**: For complex archives or binary files, spawn a sub-agent using `tool_spawn_sub_agent`:

<tool>
{
  "name": "tool_spawn_sub_agent",
  "parameters": {
    "instruction": "Read and inspect the file 'data_dump_99.bin'. Determine whether it contains JSON, CSV, binary telemetry, or logs. Recommend the appropriate category and provide a 1-sentence description.",
    "personality_conditioning": "You are a specialized file inspector. Read the file, determine its type and purpose, and report back concise factual findings."
  }
}
</tool>

4. Update `classes.md` and `mapping.md` with the verified classifications.

---

## ✋ Phase 3: User Presentation & Mandatory Confirmation Gate

You are **STRICTLY FORBIDDEN** from moving or copying files without user validation.

🚨 **CRITICAL INVARIANT**: DO NOT call `tool_spawn_sub_agent` in Phase 3! Calling a migration sub-agent before the user replies "yes" is a critical safety violation.

1. Present a clear, high-level summary to the user:
   - Total items discovered vs. batch size (e.g. *"Batch 1: 50 of 284 files"*).
   - Overview of the proposed category tree.
   - List of identified self-contained folders preserved intact.
2. Direct the user to review `classes.md` and `mapping.md` created in the workspace.
3. Formulate the explicit confirmation prompt:
   > "⚠️ **Batch 1 Confirmation (50 Items)**: I have generated `classes.md` (moving existing project folders as whole units) and mapped the first 50 items in `mapping.yaml`. Do you approve migrating this 50-item batch? Reply **'yes'** to proceed, or let me know if you would like any adjustments."
4. **CONCLUDE THE TURN WITH `<done/>`**: You must emit `<done/>` on a new line immediately after asking for confirmation so execution stops and returns control to the user.

---

## 🔄 Phase 3.5: Handling User Inquiries & Adjustments (Iterative Refinement)

When the user replies with a question, feedback, or adjustment (e.g. *"how about line's folder?"*, *"what about the audio files?"*, *"move X to Y instead"*):

1. **DO NOT EXECUTE PHASE 4!** A question or adjustment is **NOT** a confirmation to start moving files.
2. **DO NOT spawn the migration sub-agent yet!**
3. **Inspect the queried folder/files**:
   - Check the folder contents or run `tool_list_files`.
   - Determine how those files fit into the taxonomy.
4. **Update `classes.md` and `mapping.md`**:
   - Emit an `<artifact name="mapping.md" type="document">` tag to incorporate the new mapping for that folder.
   - If a new class/subclass is needed, update `classes.md`.
5. **Re-present the Updated Plan & Re-ask for Confirmation**:
   - Explain the adjustments you made to `mapping.md`.
   - Prompt the user:
     > "I have updated `mapping.md` to include this folder under `<target_class>/`. Do you approve this updated plan? Reply **'yes'** to proceed with the migration, or let me know if you would like any other changes."
   - Conclude your turn with `<done/>` on a new line so you wait for their response.

---

## 🚀 Phase 4: Migration Execution & Multi-Batch Progression

### Step 1: Execute Migration for the Approved Batch
Once the user confirms (e.g. "yes", "proceed", "approved"):
- Call `tool_organize_files_from_plan` IMMEDIATELY as the very first action:

<tool>
{
  "name": "tool_organize_files_from_plan",
  "parameters": {
    "plan_file": "mapping.yaml",
    "move_files": true
  }
}
</tool>

### Step 2: Transitioning to the Next Batch (e.g. "continue with the rest")
When the user says "continue with the rest", "organize the rest", or "next batch":
1. **NEVER CLAIM IN PROSE THAT FILES WERE MOVED WITHOUT EXECUTING TOOLS!**
2. Call `tool_list_files(directory=".")` to discover the real remaining files sitting in the root.
3. Select the next 40–50 items for Batch 2.
4. Emit `<artifact name="mapping.yaml" type="document">` with the new batch mappings.
5. Execute `tool_organize_files_from_plan(plan_file="mapping.yaml")` directly!
6. Repeat until `tool_list_files(directory=".")` shows only organized folders.