/**
 * app.js
 * ======
 * Client-side script implementing the three-panel layout:
 * - Left: Artifact list and active memories.
 * - Center: Tabbed workspace (Chat Companion, plus dynamic tabs for loaded/active artifacts).
 * - Right: Operational macros.
 * Plus Graphviz rendering with Viz.js, stacked token budget calculations,
 * LaTeX math rendering with KaTeX, and a modal ingestion panel.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── Global State variables ──
    let selectedFiles = [];
    let activeArtifactTitle = null;
    let activeArtifactType = null;
    let selectedVersion = null;
    let activeSearchResults = [];
    let openTabs = new Set(); // Stores titles of currently open tabs
    let codeEditors = {}; // Keeps track of active CodeMirror instances key-mapped by artifact title
    let toolRefDocs = {}; // Reference doc contents mapped to tool titles: { title: [{name, content}] }
    let availableBindings = [];
    let currentBindingConfig = {};
    let availableTtiBindings = [];
    let currentTtiBindingConfig = {};
    let discoveredLlmModels = [];
    let discoveredTtiModels = [];

    // ── Core DOM Elements ──
    const serverStatus = document.getElementById("server-status");
    const artifactsList = document.getElementById("artifacts-list");
    const memoriesList = document.getElementById("memories-list");
    
    // Chat & Workspace Center DOM Elements
    const workspaceTabs = document.getElementById("workspace-tabs");
    const viewport = document.getElementById("workspace-content-viewport");
    const chatHistory = document.getElementById("chat-history");
    const chatInput = document.getElementById("chat-input");
    const btnChatSend = document.getElementById("btn-chat-send");
    
    // Modal Overlay Elements
    const btnOpenImport = document.getElementById("btn-open-import");
    const btnCloseImport = document.getElementById("btn-close-import");
    const importModal = document.getElementById("import-modal");
    
    // Modal Inner Navigation Elements
    const modalTabs = document.querySelectorAll(".modal-tab");
    const modalTabContents = document.querySelectorAll(".modal-tab-content-pane");

    // Local Ingestion Elements inside Modal
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("file-input");
    const importMode = document.getElementById("import-mode");
    const customTitle = document.getElementById("custom-title");
    const btnSubmit = document.getElementById("btn-submit");

    // Internet Ingestion Elements inside Modal
    const internetSourceType = document.getElementById("internet-source-type");
    const btnInternetSearch = document.getElementById("btn-internet-search");
    const searchResultsContainer = document.getElementById("search-results-container");
    const searchResultsList = document.getElementById("search-results-list");
    const btnInternetImportSelected = document.getElementById("btn-internet-import-selected");

    // Skills & Ingestion inside Modal
    const btnScanSkills = document.getElementById("btn-scan-skills");
    const skillsScanPath = document.getElementById("skills-scan-path");
    const btnImportSingleSkill = document.getElementById("btn-import-single-skill");
    const singleSkillInput = document.getElementById("single-skill-input");

    // Bundle Portability inside Modal & Left Sidebar
    const btnExportBundle = document.getElementById("btn-export-bundle");
    const btnImportBundle = document.getElementById("btn-import-bundle");
    const bundleInput = document.getElementById("bundle-input");

    // Discovered Tools Sidebar List
    const toolsList = document.getElementById("tools-list");

    // Contextual Export Modal Elements
    const exportDetailsModal = document.getElementById("export-details-modal");
    const exportModalTitle = document.getElementById("export-modal-title");
    const exportModalBody = document.getElementById("export-modal-body");
    const btnCloseExportModal = document.getElementById("btn-close-export-modal");
    const btnCancelExport = document.getElementById("btn-cancel-export");
    const btnConfirmExport = document.getElementById("btn-confirm-export");

    // Ingestion Progress Indicators
    const progressContainer = document.getElementById("progress-container");
    const progressStatusText = document.getElementById("progress-status-text");
    const progressBarFill = document.getElementById("progress-bar-fill");

    // Stacked Token Budget Bar
    const contextBudgetBar = document.getElementById("context-budget-bar");
    const contextBudgetText = document.getElementById("context-budget-text");
    const contextBudgetLegend = document.getElementById("context-budget-legend");

    // Right Panel Macro Elements
    const macroSummarize = document.getElementById("macro-summarize");
    const macroCompare = document.getElementById("macro-compare");
    const macroGraph = document.getElementById("macro-graph");
    const btnOpenSettings = document.getElementById("btn-open-settings");
    const selBinding = document.getElementById("settings-llm-binding");
    const configForm = document.getElementById("settings-llm-config-form");
    const btnValidate = document.getElementById("btn-validate-binding");
    const selModel = document.getElementById("settings-llm-model");
    const modelGroup = document.getElementById("settings-model-group");
    const applyGroup = document.getElementById("settings-apply-group");
    const btnApply = document.getElementById("btn-apply-settings");
    const valStatus = document.getElementById("settings-validation-status");
    const headerModel = document.getElementById("header-model-select");
    const headerModelSpinner = document.getElementById("header-model-spinner");
    const headerPersonality = document.getElementById("header-personality-select");
    const headerPersonalitySpinner = document.getElementById("header-personality-spinner");

    // Header TTI Selectors
    const headerTtiModel = document.getElementById("header-tti-model-select");
    const headerTtiModelSpinner = document.getElementById("header-tti-model-spinner");

    // Workspace Selection Screen Selectors
    const workspaceSelectionScreen = document.getElementById("workspace-selection-screen");
    const workspaceCardGrid = document.getElementById("workspace-card-grid");
    const btnBackToWorkspaces = document.getElementById("btn-back-to-workspaces");
    const btnChatClear = document.getElementById("btn-chat-clear");
    const btnChatStop = document.getElementById("btn-chat-stop");

    // TTI Configuration Selectors
    const selTtiBinding = document.getElementById("settings-tti-binding");
    const ttiConfigForm = document.getElementById("settings-tti-config-form");
    const btnValidateTti = document.getElementById("btn-validate-tti-binding");
    const selTtiModel = document.getElementById("settings-tti-model");
    const ttiModelSearch = document.getElementById("settings-tti-model-search");
    const ttiModelGroup = document.getElementById("settings-tti-model-group");
    const valTtiStatus = document.getElementById("settings-tti-validation-status");

    // LLM Model Search Selector
    const llmModelSearch = document.getElementById("settings-llm-model-search");

    // Profile Management Selectors
    const profileSaveName = document.getElementById("profile-save-name");
    const btnSaveProfile = document.getElementById("btn-save-profile");
    const settingsProfilesList = document.getElementById("settings-profiles-list");

    // ── Startup Initialization ──
    fetchArtifacts();
    fetchMemories();
    fetchDiscoveredTools();
    updateContextBudget();
    loadSettingsIntoUI();
    fetchMessageHistory();

    const btnAddMemory = document.getElementById("btn-add-memory-part");
    if (btnAddMemory) {
        btnAddMemory.addEventListener("click", async () => {
            const content = prompt("Enter memory content:");
            if (!content) return;
            const impStr = prompt("Importance (0.0 - 1.0):", "0.75");
            const importance = parseFloat(impStr || "0.75");
            const lvlStr = prompt("Level (1=Working, 2=Deep, 3=Archived):", "1");
            const level = parseInt(lvlStr || "1", 10);
            try {
                const res = await fetch("/api/memories/import", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ content, importance, level })
                });
                const data = await res.json();
                if (data.success) fetchMemories();
            } catch (e) { alert("Failed to add memory: " + e); }
        });
    }

    // ── Check Server Status ──
    fetch("/status")
        .then(res => res.json())
        .then(data => {
            if (data.status === "running") {
                serverStatus.textContent = data.is_mock ? "Server Online (Mock Fallback)" : "Server Online";
                serverStatus.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
                serverStatus.style.color = "#10b981";
            }
        })
        .catch(() => {
            serverStatus.textContent = "Offline";
            serverStatus.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
            serverStatus.style.color = "#ef4444";
        });

    // ── Combined Ingestion Modal Display Toggle ──
    btnOpenImport.addEventListener("click", () => {
        importModal.style.display = "flex";
        progressContainer.style.display = "none";
        progressBarFill.style.width = "0%";
    });

    btnCloseImport.addEventListener("click", () => {
        importModal.style.display = "none";
    });

    importModal.addEventListener("click", (e) => {
        if (e.target === importModal) {
            importModal.style.display = "none";
        }
    });

    modalTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            modalTabs.forEach(t => t.classList.remove("active"));
            modalTabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            document.getElementById(tab.dataset.modalTab).classList.add("active");
        });
    });

    // ── 🧠 Context Budget Stacked Render Subroutine ──
    async function updateContextBudget() {
        try {
            const res = await fetch("/api/context_status");
            const data = await res.json();

            const maxTokens = data.max_tokens || 8192;
            const currentTokens = data.current_tokens || 0;
            const percent = ((currentTokens / maxTokens) * 100).toFixed(1);

            contextBudgetText.textContent = `${currentTokens.toLocaleString()} / ${maxTokens.toLocaleString()} tokens (${percent}%)`;

            let sysTokens = 0;
            const sysCtx = data.zones.system_context;
            if (sysCtx) {
                sysTokens = sysCtx.tokens || 0;
            }

            let artTokens = 0;
            if (sysCtx && sysCtx.breakdown && sysCtx.breakdown.artefacts) {
                artTokens = sysCtx.breakdown.artefacts.tokens || 0;
                sysTokens = Math.max(0, sysTokens - artTokens);
            }

            let histTokens = 0;
            const histCtx = data.zones.message_history;
            if (histCtx) {
                histTokens = histCtx.tokens || 0;
            }

            const sysPercent = ((sysTokens / maxTokens) * 100).toFixed(2);
            const artPercent = ((artTokens / maxTokens) * 100).toFixed(2);
            const histPercent = ((histTokens / maxTokens) * 100).toFixed(2);

            contextBudgetBar.innerHTML = `
                <div class="context-budget-segment system" style="width: ${sysPercent}%" title="System / Static Prompt: ${sysTokens} tokens"></div>
                <div class="context-budget-segment artifacts" style="width: ${artPercent}%" title="Active Artifacts: ${artTokens} tokens"></div>
                <div class="context-budget-segment history" style="width: ${histPercent}%" title="Message History: ${histTokens} tokens"></div>
            `;

            contextBudgetLegend.innerHTML = `
                <div class="legend-item"><span class="legend-color-dot system"></span> System (${sysTokens.toLocaleString()})</div>
                <div class="legend-item"><span class="legend-color-dot artifacts"></span> Artifacts (${artTokens.toLocaleString()})</div>
                <div class="legend-item"><span class="legend-color-dot history"></span> History (${histTokens.toLocaleString()})</div>
            `;
        } catch (err) {
            console.error("Failed to update context budget:", err);
        }
    }

    // ── 📐 LaTeX Math Parser Subroutine (using KaTeX) ──
    function renderMath(element) {
        if (typeof renderMathInElement === "function") {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        }
    }

    // ── ⚡ Dynamic Tab Manager Subroutine ──
    
    function makeSafeId(str) {
        return str.replace(/[^a-zA-Z0-9_-]/g, "_");
    }

    function switchCenterTab(targetTabId) {
        document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

        const targetBtn = document.querySelector(`.tab-btn[data-tab="${targetTabId}"]`);
        const targetContent = document.getElementById(targetTabId);

        if (targetBtn && targetContent) {
            targetBtn.classList.add("active");
            targetContent.classList.add("active");
        }
    }

    function createArtifactTab(title, type) {
        const safeId = makeSafeId(title);
        const tabId = `tab-art-${safeId}`;

        if (openTabs.has(title)) {
            switchCenterTab(tabId);
            return;
        }

        // Determine appropriate icon based on artifact type
        let icon = "📄";
        if (type === "skill") icon = "🎓";
        else if (type === "tool") icon = "🛠️";
        else if (type === "data") icon = "📊";
        else if (type === "code") icon = "💻";

        const tabBtn = document.createElement("button");
        tabBtn.className = "tab-btn";
        tabBtn.dataset.tab = tabId;
        tabBtn.dataset.artTitle = title;
        tabBtn.innerHTML = `
            <span>${icon} ${title}</span>
            <span class="tab-close" data-art-title="${title}">×</span>
        `;
        workspaceTabs.appendChild(tabBtn);

        const tabContent = document.createElement("div");
        tabContent.className = "tab-content";
        tabContent.id = tabId;
        tabContent.innerHTML = `
            <div class="artifact-sub-tabs">
                <button class="sub-tab-btn active" data-sub-tab="rendered-${safeId}">👁️ Rendered View</button>
                <button class="sub-tab-btn" data-sub-tab="raw-${safeId}">💻 Raw Source Code</button>
                <div class="version-select-container">
                    <label>Version:</label>
                    <select class="version-select" data-art-title="${title}"></select>
                    <button class="sub-tab-btn download-art-btn" data-art-title="${title}" title="Download / Export this artifact" style="margin-left: 8px; background-color: var(--chat-user-bg); color: white; border-color: var(--chat-user-bg);">💾 Download / Export</button>
                </div>
            </div>
            <div class="sub-tab-content active" id="sub-tab-rendered-${safeId}">
                <div class="rendered-container" id="rendered-view-${safeId}"></div>
            </div>
            <div class="sub-tab-content" id="sub-tab-raw-${safeId}">
                <div class="raw-editor-controls" style="display: flex; justify-content: flex-end; margin-bottom: 10px; flex-shrink: 0;">
                    <button class="btn btn-primary save-raw-btn" id="btn-save-raw-${safeId}" style="width: auto; padding: 6px 14px; font-size: 12px;">💾 Save Changes</button>
                </div>
                <textarea class="raw-textarea" id="raw-view-${safeId}" placeholder="Edit your artifact raw source code here..."></textarea>
            </div>
        `;
        viewport.appendChild(tabContent);

        openTabs.add(title);

        const subTabButtons = tabContent.querySelectorAll(".sub-tab-btn");
        const subTabPanes = tabContent.querySelectorAll(".sub-tab-content");
        subTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                subTabButtons.forEach(b => b.classList.remove("active"));
                subTabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                tabContent.querySelector(`#sub-tab-${btn.dataset.subTab}`).classList.add("active");
                if (btn.dataset.subTab === `rendered-${safeId}` && codeEditors[title]) {
                    setTimeout(() => codeEditors[title].refresh(), 10);
                }
            });
        });

        const vSelect = tabContent.querySelector(".version-select");
        vSelect.addEventListener("change", (e) => {
            loadArtifactVersion(title, parseInt(e.target.value, 10));
        });

        // Bind Context-Aware Download/Export Click
        const downloadBtn = tabContent.querySelector(".download-art-btn");
        if (downloadBtn) {
            downloadBtn.addEventListener("click", () => {
                if (type === "image") {
                    // Direct binary download for images
                    const a = document.createElement("a");
                    a.href = `/api/images/${encodeURIComponent(title)}/0`;
                    a.download = `${title}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                } else {
                    // Open the unified export modal with this artifact pre-selected
                    openExportModal("artifacts");
                    setTimeout(() => {
                        const sel = document.getElementById("export-art-select");
                        if (sel) {
                            sel.value = title;
                            sel.dispatchEvent(new Event("change"));
                        }
                    }, 150);
                }
            });
        }

        tabBtn.addEventListener("click", (e) => {
            if (e.target.classList.contains("tab-close")) {
                return;
            }
            selectArtifact(title);
        });

        const closeBtn = tabBtn.querySelector(".tab-close");
        closeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            closeArtifactTab(title);
        });

        // Bind raw source code "Save Changes" button
        const saveRawBtn = tabContent.querySelector(`.save-raw-btn`);
        const rawTextarea = tabContent.querySelector(`.raw-textarea`);

        saveRawBtn.addEventListener("click", async () => {
            saveRawBtn.disabled = true;
            saveRawBtn.textContent = "Saving...";

            try {
                const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}/update`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ content: rawTextarea.value })
                });
                const data = await res.json();
                if (data.success) {
                    alert(`Artifact successfully saved as version ${data.version}!`);
                    // Re-fetch list and reload selection to refresh the Rendered View!
                    await fetchArtifacts();
                    await selectArtifact(title);
                } else {
                    alert(`Save failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Save request failed: ${err}`);
            } finally {
                saveRawBtn.disabled = false;
                saveRawBtn.textContent = "💾 Save Changes";
            }
        });

        switchCenterTab(tabId);
    }

    function closeArtifactTab(title) {
        const safeId = makeSafeId(title);
        const tabId = `tab-art-${safeId}`;
        const tabBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
        const tabContent = document.getElementById(tabId);

        if (tabBtn) tabBtn.remove();
        if (tabContent) tabContent.remove();

        openTabs.delete(title);
        if (codeEditors[title]) {
            delete codeEditors[title];
        }

        if (activeArtifactTitle === title) {
            activeArtifactTitle = null;
            activeArtifactType = null;
            selectedVersion = null;
            
            document.querySelectorAll(".artifact-card").forEach(card => card.classList.remove("selected"));
            
            switchCenterTab("tab-chat");
            if (btnExportBundle) btnExportBundle.disabled = true;
        }
    }

    document.getElementById("tab-btn-chat").addEventListener("click", () => {
        activeArtifactTitle = null;
        activeArtifactType = null;
        selectedVersion = null;
        document.querySelectorAll(".artifact-card").forEach(card => card.classList.remove("selected"));
        switchCenterTab("tab-chat");
    });
    // ── 🛠️ Create New Tool Action ──
    const btnCreateTool = document.getElementById("btn-create-tool");
    btnCreateTool.addEventListener("click", () => {
        const rawName = prompt("Enter a snake_case name for your new custom LCP tool (e.g. file_zipper):");
        if (!rawName) return;

        const cleanName = rawName.replace(/[^a-zA-Z0-9_]/g, "").trim().toLowerCase();
        if (!cleanName) {
            alert("Invalid tool name. Only alphanumeric characters and underscores are allowed.");
            return;
        }

        const templateCode = `# ${cleanName}.py\n# Lollms local LCP tool\n# -----------------------------------------------------------------------------\n\nTOOL_LIBRARY_NAME = "${cleanName.replace(/_/g, ' ').toUpperCase()}"\nTOOL_LIBRARY_DESC = "Explain what this smart tool library does in this metadata block."\nTOOL_LIBRARY_ICON = "🔧"\n\ndef init_tool_library() -> None:\n    \"\"\"\n    Optional: Initialize any third-party libraries needed by your tool using pipmaster.\n    \"\"\"\n    import pipmaster as pm\n    # pm.ensure_packages({"requests": ">=2.0"})\n    pass\n\ndef tool_${cleanName}(\n    query: str,\n    count: int = 5\n) -> dict:\n    \"\"\"\n    Brief description of what this custom tool does.\n\n    Args:\n        query (str): Input query, path, or payload.\n        count (int, optional): Number of items to return/fetch. Defaults to 5.\n    \"\"\"\n    if not query:\n        return {"success": False, "error": "Query parameter must not be empty"}\n\n    return {\n        "success": True,\n        "result": f"Executed '${cleanName}' on query: '{query}' with count {count}"\n    }\n`;

        // Register the new artifact programmatically
        const artTitle = `${cleanName}_tool`;
        const artContent = `# Smart Tool: ${cleanName}\nGenerated on: ${new Date().toISOString()}\n\n### Python Implementation (\`${cleanName}.py\`)\n\`\`\`python\n${templateCode}\n\`\`\``;

        // Perform Save request directly to persist to local LCP folder and SQLite DB
        fetch("/api/save_tool", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: artTitle, code: templateCode, commit_message: "Initialize tool" })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                activeArtifactTitle = artTitle;
                fetchArtifacts().then(() => {
                    selectArtifact(artTitle);
                    fetchDiscoveredTools();
                });
            } else {
                alert(`Failed to save tool: ${data.detail}`);
            }
        })
        .catch(err => alert(`Error creating tool: ${err}`));
    });
    // Extract utility helper to capture python code blocks inside Markdown
    function extractCodeFromMarkdown(content) {
        const match = content.match(/```python\s*\n([\s\S]*?)\n```/i);
        return match ? match[1] : content;
    }

    // ── 💾 Load Specific Version of an Artifact ──
    async function loadArtifactVersion(title, version) {
        const safeId = makeSafeId(title);
        const renderedView = document.getElementById(`rendered-view-${safeId}`);
        const rawView = document.getElementById(`raw-view-${safeId}`);

        if (!renderedView || !rawView) return;

        try {
            // Retrieve full artifacts structure to determine type
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            const matched = arts.find(a => a.title === title);
            if (!matched) return;

            activeArtifactType = matched.type;

            if (activeArtifactType === "data") {
                renderSpreadsheetGridInTab(title, version, renderedView);

                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;
            } else if (activeArtifactType === "tool") {
                // Renders the Full Interactive LCP Tool Builder Workspace inside Rendered View!
                renderToolEditor(title, version, renderedView);

                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;
            } else if (activeArtifactType === "presentation") {
                // ── Render Presentation Slide Decks inside an Isolated Iframe ──
                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;

                const blob = new Blob([art.content], { type: "text/html" });
                const iframeUrl = URL.createObjectURL(blob);
                renderedView.innerHTML = `
                    <iframe src="${iframeUrl}" style="width: 100%; height: 100%; border: none; min-height: 520px; border-radius: 8px; background: #000;" id="presentation-frame-${safeId}"></iframe>
                `;
            } else {
                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;

                if (matched.type === "code") {
                    const lang = matched.language || "";
                    if (lang === "graphviz" || art.content.includes("digraph") || art.content.includes("graph")) {
                        renderedView.innerHTML = `<div class="empty-viewer-msg"><span class="spinner inline" style="margin-right: 8px;"></span> Compiling Graphviz DOT logic...</div>`;
                        try {
                            const viz = new Viz();
                            viz.renderSVGElement(art.content)
                                .then(element => {
                                    element.style.width = "100%";
                                    element.style.height = "auto";
                                    renderedView.innerHTML = "";
                                    renderedView.appendChild(element);
                                })
                                .catch(err => {
                                    console.error(err);
                                    renderedView.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; font-family: monospace;">Viz.js Compilation Failed:<br>${err.message || err}</div>`;
                                });
                        } catch (err) {
                            renderedView.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Viz.js library initialization failed: ${err.message || err}</div>`;
                        }
                    } else if (lang === "python" || lang === "html") {
                        if (codeEditors[title]) {
                            try { codeEditors[title].toTextArea(); } catch(e) {}
                            delete codeEditors[title];
                        }
                        renderedView.innerHTML = `
                            <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: bold;">${lang} Source</span>
                                <button class="btn btn-primary" id="btn-run-code-${safeId}" style="width:auto; padding: 6px 14px; font-size: 12px;">▶ Run ${lang.toUpperCase()}</button>
                            </div>
                            <textarea id="cm-rendered-${safeId}"></textarea>
                            <div id="code-run-output-${safeId}" style="display:none; margin-top: 12px; flex-direction: column; gap: 6px;"></div>
                        `;
                        const txtArea = renderedView.querySelector(`#cm-rendered-${safeId}`);
                        const cm = CodeMirror.fromTextArea(txtArea, {
                            mode: lang === "html" ? "htmlmixed" : "python",
                            theme: "dracula",
                            lineNumbers: true,
                            readOnly: true,
                            lineWrapping: true,
                            tabSize: 4,
                        });
                        cm.setValue(art.content);
                        codeEditors[title] = cm;

                        const runBtn = renderedView.querySelector(`#btn-run-code-${safeId}`);
                        const outputBox = renderedView.querySelector(`#code-run-output-${safeId}`);

                        runBtn.addEventListener("click", async () => {
                            if (lang === "html") {
                                const blob = new Blob([art.content], { type: "text/html" });
                                const url = URL.createObjectURL(blob);
                                window.open(url, "_blank");
                                setTimeout(() => URL.revokeObjectURL(url), 60000);
                            } else {
                                runBtn.disabled = true;
                                runBtn.textContent = "Running...";
                                outputBox.style.display = "flex";
                                outputBox.innerHTML = `<div class="empty-viewer-msg" style="justify-content: flex-start; padding: 0;"><span class="spinner inline" style="margin-right: 8px;"></span> Executing Python sandbox...</div>`;
                                try {
                                    const res = await fetch("/api/execute_sandbox", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ code: art.content, language: "python" })
                                    });
                                    const data = await res.json();
                                    if (data.success) {
                                        outputBox.innerHTML = `
                                            <div class="query-title">💻 Sandbox Output</div>
                                            <pre class="query-stdout">${data.output || "(No stdout output)"}</pre>
                                        `;
                                    } else {
                                        outputBox.innerHTML = `
                                            <div class="query-title" style="color: #ef4444;">❌ Execution Error</div>
                                            <pre class="query-stdout" style="color: #fca5a5;">${data.error}</pre>
                                            ${data.output ? `<div class="query-title" style="margin-top: 6px;">Stdout prior to error:</div><pre class="query-stdout">${data.output}</pre>` : ""}
                                        `;
                                    }
                                } catch (err) {
                                    outputBox.innerHTML = `<div class="query-title" style="color: #ef4444;">Request Failed</div><pre class="query-stdout">${err}</pre>`;
                                } finally {
                                    runBtn.disabled = false;
                                    runBtn.textContent = "▶ Run PYTHON";
                                }
                            }
                        });
                    } else {
                        const renderContent = `\`\`\`${lang}\n${art.content}\n\`\`\``;
                        renderedView.innerHTML = marked.parse(renderContent);
                        renderMath(renderedView);
                    }
                } else {
                    const parsedMarkdown = marked.parse(art.content);
                    const resolvedHTML = resolveImageAnchors(parsedMarkdown, title);
                    renderedView.innerHTML = resolvedHTML;
                    renderMath(renderedView);
                }
            }
            updateContextBudget();
        } catch (err) {
            console.error("Failed to load artifact version:", err);
        }
    }

    // ── 🛠️ Render Full Interactive LCP Tool Builder Workspace ──
    async function renderToolEditor(title, version, containerElement) {
        containerElement.innerHTML = `<div class="empty-viewer-msg"><span class="spinner inline" style="margin-right: 8px;"></span> Loading Tool Editor...</div>`;
        try {
            const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
            const art = await artRes.json();
            const rawCode = extractCodeFromMarkdown(art.content);

            const safeId = makeSafeId(title);

            containerElement.innerHTML = `
                <div class="tool-editor-layout">
                    <!-- Left Sidebar (Internal Navigation Tabs) -->
                    <div class="tool-internal-tabs">
                        <button class="tool-tab-btn active" data-tool-tab="editor-${safeId}">✏️ Code Editor</button>
                        <button class="tool-tab-btn" data-tool-tab="docs-${safeId}">📚 Reference Docs</button>
                        <button class="tool-tab-btn" data-tool-tab="tests-${safeId}">🧪 Test Suite</button>
                        <button class="tool-tab-btn" data-tool-tab="guide-${safeId}">📘 Developer Guide</button>
                    </div>

                    <!-- Right Column (Active tab pane viewport) -->
                    <div class="tool-tab-viewport">
                        <!-- Pane 1: Code Editor & AI Refiner -->
                        <div class="tool-tab-pane active" id="tool-pane-editor-${safeId}">
                            <div class="tool-code-pane">
                                <textarea id="cm-editor-${safeId}"></textarea>
                                <div class="tool-editor-controls">
                                    <div class="tool-ai-wrapper">
                                        <input type="text" class="tool-ai-input" id="ai-instruction-${safeId}" placeholder="🤖 Ask AI to refine code (e.g. 'Add parameters validation')">
                                        <button class="tool-ai-btn" id="btn-ai-refine-${safeId}">🤖 Refine Code</button>
                                    </div>
                                    <button class="btn btn-primary" id="btn-save-tool-${safeId}">💾 Save & Compile Tool (v${version})</button>
                                </div>
                            </div>
                        </div>

                        <!-- Pane 2: Reference Documentation -->
                        <div class="tool-tab-pane" id="tool-pane-docs-${safeId}">
                            <div class="context-budget-container" style="margin-bottom: 16px;">
                                <div class="context-budget-header">
                                    <span>🧠 Coder Context Size</span>
                                    <span id="tool-context-text-${safeId}">0 / 8,192 tokens (0%)</span>
                                </div>
                                <div class="context-budget-bar-bg">
                                    <div class="progress-bar-fill" id="tool-context-bar-fill-${safeId}" style="width: 0%; background-color: var(--chat-user-bg);"></div>
                                </div>
                            </div>

                            <div style="display: flex; gap: 16px; flex: 1;">
                                <div style="flex: 1; display: flex; flex-direction: column;">
                                    <h3>Attached Reference Documents</h3>
                                    <ul class="artifact-list" id="tool-attached-docs-${safeId}" style="margin-top: 10px; max-height: 250px; overflow-y: auto; flex: 1;">
                                        <li class="empty-msg">No reference documents attached yet.</li>
                                    </ul>
                                </div>
                                <div style="width: 250px; display: flex; flex-direction: column; gap: 10px;">
                                    <h3>Quick Attach</h3>
                                    <div class="input-group">
                                        <label for="tool-doc-select-${safeId}">Attach Active Artifact</label>
                                        <select id="tool-doc-select-${safeId}" style="padding: 6px; font-size: 12px;"></select>
                                    </div>
                                    <button class="btn btn-secondary" id="btn-attach-artifact-${safeId}">🔗 Attach Selection</button>
                                    <button class="btn btn-secondary" id="btn-upload-ref-file-${safeId}">📁 Upload File</button>
                                    <input type="file" id="tool-ref-file-input-${safeId}" hidden>
                                </div>
                            </div>
                        </div>

                        <!-- Pane 3: Test Suite & Execution -->
                        <div class="tool-tab-pane" id="tool-pane-tests-${safeId}">
                            <div style="display: flex; gap: 16px; height: 100%;">
                                <div style="width: 250px; border-right: 1px solid var(--border-color); padding-right: 16px; display: flex; flex-direction: column; gap: 12px;">
                                    <button class="btn btn-primary" id="btn-run-init-${safeId}" style="background-color: var(--success-color); color: #020617;">🧪 Run init_tool_library()</button>
                                    <h3>Select Function to Test</h3>
                                    <div id="tool-tests-fn-list-${safeId}" style="display: flex; flex-direction: column; gap: 6px; overflow-y: auto; flex: 1;">
                                        <li class="empty-msg">Save the tool first to parse callable functions.</li>
                                    </div>
                                </div>
                                <div style="flex: 1; display: flex; flex-direction: column; gap: 12px;">
                                    <div id="tool-test-exec-panel-${safeId}" style="display: none; flex-direction: column; gap: 10px; flex: 1;">
                                        <h3 id="tool-test-fn-name-${safeId}">Function: tool_name</h3>
                                        <p id="tool-test-fn-doc-${safeId}" style="font-size: 11px; color: var(--text-secondary);"></p>
                                        <div id="tool-test-params-container-${safeId}" style="display: flex; flex-direction: column; gap: 8px;"></div>
                                        <button class="btn btn-primary" id="btn-execute-test-${safeId}">⚡ Execute Function</button>
                                        <div class="query-result-box" style="flex: 1; display: flex; flex-direction: column; margin-top: 10px;">
                                            <div class="query-title">💻 Sandbox Output / Result</div>
                                            <pre class="query-stdout" id="tool-test-stdout-${safeId}" style="flex: 1; overflow-y: auto; max-height: 180px; font-family: monospace; font-size: 11.5px;"></pre>
                                        </div>
                                    </div>
                                    <div id="tool-test-empty-panel-${safeId}" class="empty-viewer-msg">
                                        Select a parsed function on the left to configure parameters and run local tests.
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Pane 4: Developer Guide -->
                        <div class="tool-tab-pane" id="tool-pane-guide-${safeId}">
                            <div class="tool-help-pane">
                                <h3>📘 LCP Tool Developer Guide</h3>
                                <p style="margin-top: 8px;">The Lollms Local Tool Binding (LCP) automatically discovers and compiles your standalone Python scripts.</p>
                                <h4 style="margin-top: 12px; font-weight: bold;">🔑 Structural Rules</h4>
                                <ul style="padding-left: 20px; margin-top: 6px;">
                                    <li><strong>Entry Point</strong>: Your script must define a function named <code>tool_[tool_name]</code> (e.g., <code>tool_${title.replace('_tool','')}</code>) or simply <code>execute</code>.</li>
                                    <li><strong>Schema Auto-Discovery</strong>: Parameter names, default values, type-hints, and descriptive docstrings are parsed via AST on the server automatically. No JSON configuration is needed!</li>
                                    <li><strong>Imports</strong>: Import any required library inside your function (or at the top). Use <code>pipmaster</code> inside <code>init_tool_library()</code> if custom libraries need automatic installation.</li>
                                </ul>
                                <h4 style="margin-top: 12px; font-weight: bold;">🚀 Advanced Integration</h4>
                                <p style="margin-top: 6px;">To access active conversational or client variables directly inside your tool, simply declare them in your function parameters:</p>
                                <ul style="padding-left: 20px; margin-top: 6px;">
                                    <li><code>lollms_client_instance</code>: Accesses the core LLM, STT, or TTI bindings.</li>
                                    <li><code>discussion_instance</code>: Read metadata, add direct messages, or update other artifacts.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Initialize CodeMirror instance for this tab
            const txtArea = containerElement.querySelector(`#cm-editor-${safeId}`);
            const editor = CodeMirror.fromTextArea(txtArea, {
                mode: "python",
                theme: "dracula",
                lineNumbers: true,
                indentUnit: 4,
                tabSize: 4,
                indentWithTabs: false,
                lineWrapping: true
            });
            editor.setValue(rawCode);

            // Keep reference to the active editor
            codeEditors[title] = editor;

            // ── Internal Tab Navigation Clicks ──
            const internalTabButtons = containerElement.querySelectorAll(".tool-tab-btn");
            const internalTabPanes = containerElement.querySelectorAll(".tool-tab-pane");
            internalTabButtons.forEach(btn => {
                btn.addEventListener("click", () => {
                    internalTabButtons.forEach(b => b.classList.remove("active"));
                    internalTabPanes.forEach(p => p.classList.remove("active"));
                    btn.classList.add("active");
                    containerElement.querySelector(`#tool-pane-${btn.dataset.toolTab}`).classList.add("active");

                    // Refresh CodeMirror layout on focus
                    if (btn.dataset.toolTab.startsWith("editor")) {
                        editor.refresh();
                    }
                });
            });

            // ── Reference Docs Sub-System ──
            if (!toolRefDocs[title]) {
                toolRefDocs[title] = [];
            }

            function updateRefDocsUI() {
                const attachedDocsList = containerElement.querySelector(`#tool-attached-docs-${safeId}`);
                if (toolRefDocs[title].length === 0) {
                    attachedDocsList.innerHTML = `<li class="empty-msg">No reference documents attached yet.</li>`;
                } else {
                    attachedDocsList.innerHTML = toolRefDocs[title].map((doc, idx) => `
                        <li class="artifact-card" style="padding: 8px 12px;">
                            <div class="artifact-card-header">
                                <span style="font-weight: bold; font-size: 11.5px;">📄 ${doc.name} (${doc.content.length} chars)</span>
                                <button class="artifact-action-btn delete-ref-doc" data-index="${idx}" title="Detach document" style="padding: 2px;">✕</button>
                            </div>
                        </li>
                    `).join("");

                    // Bind Detach Clicks
                    attachedDocsList.querySelectorAll(".delete-ref-doc").forEach(delBtn => {
                        delBtn.addEventListener("click", () => {
                            const idx = parseInt(delBtn.dataset.index, 10);
                            toolRefDocs[title].splice(idx, 1);
                            updateRefDocsUI();
                        });
                    });
                }

                // Update Coder Context Token Counter
                const instruction = containerElement.querySelector(`#ai-instruction-${safeId}`).value.trim();
                const docStrings = toolRefDocs[title].map(d => d.content);

                fetch("/api/count_tool_tokens", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: editor.getValue(), docs: docStrings, instruction: instruction })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        const tokenCount = data.tokens;
                        const maxTokens = 8192; // standard limit
                        const pct = Math.min(100, (tokenCount / maxTokens) * 100).toFixed(1);
                        containerElement.querySelector(`#tool-context-text-${safeId}`).textContent = `${tokenCount.toLocaleString()} / ${maxTokens.toLocaleString()} tokens (${pct}%)`;
                        containerElement.querySelector(`#tool-context-bar-fill-${safeId}`).style.width = `${pct}%`;
                    }
                })
                .catch(err => console.error("Failed to count tool tokens:", err));
            }

            // Populate the Active Artifact Quick Attach selector
            fetch("/api/artifacts")
                .then(res => res.json())
                .then(arts => {
                    const sel = containerElement.querySelector(`#tool-doc-select-${safeId}`);
                    // Filter out this tool itself from selection
                    const validDocs = arts.filter(a => a.title !== title);
                    if (validDocs.length === 0) {
                        sel.innerHTML = `<option value="">No artifacts available</option>`;
                    } else {
                        sel.innerHTML = validDocs.map(a => `<option value="${a.title}">${a.title}</option>`).join("");
                    }
                });

            // Bind Quick Attach Click
            const attachBtn = containerElement.querySelector(`#btn-attach-artifact-${safeId}`);
            attachBtn.addEventListener("click", async () => {
                const targetTitle = containerElement.querySelector(`#tool-doc-select-${safeId}`).value;
                if (!targetTitle) return;

                if (toolRefDocs[title].some(d => d.name === targetTitle)) {
                    alert("This document is already attached as reference.");
                    return;
                }

                try {
                    const res = await fetch(`/api/artifacts/${encodeURIComponent(targetTitle)}`);
                    const artData = await res.json();
                    toolRefDocs[title].push({ name: targetTitle, content: artData.content });
                    updateRefDocsUI();
                } catch (err) {
                    alert(`Failed to fetch artifact: ${err}`);
                }
            });

            // Bind Local Reference File Upload Clicks
            const refFileInput = containerElement.querySelector(`#tool-ref-file-input-${safeId}`);
            const uploadRefBtn = containerElement.querySelector(`#btn-upload-ref-file-${safeId}`);
            uploadRefBtn.addEventListener("click", () => refFileInput.click());

            refFileInput.addEventListener("change", (e) => {
                if (e.target.files.length === 0) return;
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function(evt) {
                    toolRefDocs[title].push({ name: file.name, content: evt.target.result });
                    updateRefDocsUI();
                    refFileInput.value = ""; // reset
                };
                reader.readAsText(file);
            });

            editor.on("change", () => {
                updateRefDocsUI();
            });

            updateRefDocsUI();

            // ── 🧪 Test Suite & Function Executor Sub-System ──
            const runInitBtn = containerElement.querySelector(`#btn-run-init-${safeId}`);
            const testStdout = containerElement.querySelector(`#tool-test-stdout-${safeId}`);
            const testsFnList = containerElement.querySelector(`#tool-tests-fn-list-${safeId}`);
            const execPanel = containerElement.querySelector(`#tool-test-exec-panel-${safeId}`);
            const emptyPanel = containerElement.querySelector(`#tool-test-empty-panel-${safeId}`);

            // Bind init runner
            runInitBtn.addEventListener("click", () => {
                runInitBtn.disabled = true;
                runInitBtn.textContent = "Executing init_tool_library()...";
                testStdout.textContent = "Loading dependencies in Python sandbox...";

                fetch("/api/run_tool_init", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: editor.getValue() })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        testStdout.textContent = data.output;
                    } else {
                        testStdout.textContent = `Error in init_tool_library():\n${data.error}\n\nStdout:\n${data.output}`;
                    }
                })
                .catch(err => {
                    testStdout.textContent = `Sandbox execution request failed: ${err}`;
                })
                .finally(() => {
                    runInitBtn.disabled = false;
                    runInitBtn.textContent = "🧪 Run init_tool_library()";
                });
            });

            // Load and list functions
            function loadAndListFunctions() {
                fetch("/api/parse_tool_functions", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: editor.getValue() })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success && data.functions.length > 0) {
                        testsFnList.innerHTML = data.functions.map(fn => `
                            <button class="sub-tab-btn select-fn-btn" data-fn-name="${fn.name}" style="text-align: left; width: 100%; border-radius: 4px; padding: 6px 10px;">
                                ⚙️ ${fn.name}()
                            </button>
                        `).join("");

                        // Bind function clicks
                        testsFnList.querySelectorAll(".select-fn-btn").forEach(fnBtn => {
                            fnBtn.addEventListener("click", () => {
                                testsFnList.querySelectorAll(".select-fn-btn").forEach(b => b.classList.remove("active"));
                                fnBtn.classList.add("active");
                                const fnName = fnBtn.dataset.fnName;
                                const matchedFn = data.functions.find(f => f.name === fnName);
                                if (matchedFn) {
                                    renderFunctionExecutor(matchedFn);
                                }
                            });
                        });
                    } else {
                        testsFnList.innerHTML = `<li class="empty-msg">No callable functions parsed.</li>`;
                    }
                })
                .catch(err => {
                    testsFnList.innerHTML = `<li class="empty-msg" style="color: #ef4444;">Failed to parse functions: ${err}</li>`;
                });
            }

            function renderFunctionExecutor(fnObj) {
                emptyPanel.style.display = "none";
                execPanel.style.display = "flex";

                containerElement.querySelector(`#tool-test-fn-name-${safeId}`).textContent = `Function: ${fnObj.name}()`;
                containerElement.querySelector(`#tool-test-fn-doc-${safeId}`).textContent = fnObj.docstring;

                const paramsContainer = containerElement.querySelector(`#tool-test-params-container-${safeId}`);
                if (fnObj.parameters.length === 0) {
                    paramsContainer.innerHTML = `<p class="empty-msg" style="text-align: left;">This function takes no parameters.</p>`;
                } else {
                    paramsContainer.innerHTML = fnObj.parameters.map(p => {
                        const requiredLabel = p.required ? `<span style="color: #ef4444;">*</span>` : "";
                        const defaultVal = p.default !== null ? ` (Default: ${p.default})` : "";
                        const val = p.default !== null ? p.default : "";
                        return `
                            <div class="input-group" style="margin-bottom: 8px;">
                                <label style="text-transform: none; font-size: 11.5px; font-weight: bold;">${p.name}${requiredLabel}${defaultVal}</label>
                                <input type="text" class="fn-param-input" data-param-name="${p.name}" data-param-type="${p.type}" value="${val}" placeholder="${p.type} value" style="padding: 6px; font-size: 12px; border-radius: 4px;">
                            </div>
                        `;
                    }).join("");
                }

                // Bind execution button for this function
                const execBtn = containerElement.querySelector(`#btn-execute-test-${safeId}`);
                // Replace any previous listeners
                const newExecBtn = execBtn.cloneNode(true);
                execBtn.parentNode.replaceChild(newExecBtn, execBtn);

                newExecBtn.addEventListener("click", () => {
                    newExecBtn.disabled = true;
                    newExecBtn.textContent = "Executing Sandbox...";
                    testStdout.textContent = "Running code in Python sandbox...";

                    // Read params
                    const paramsPayload = {};
                    let validationFailed = false;

                    containerElement.querySelectorAll(".fn-param-input").forEach(inp => {
                        const pName = inp.dataset.paramName;
                        const pType = inp.dataset.paramType;
                        const val = inp.value.trim();

                        if (!val) {
                            // If required
                            const spec = fnObj.parameters.find(p => p.name === pName);
                            if (spec && spec.required) {
                                alert(`Parameter '${pName}' is required.`);
                                validationFailed = true;
                                return;
                            }
                        }

                        paramsPayload[pName] = val;
                    });

                    if (validationFailed) {
                        newExecBtn.disabled = false;
                        newExecBtn.textContent = "⚡ Execute Function";
                        return;
                    }

                    fetch("/api/execute_tool_function", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ code: editor.getValue(), function_name: fnObj.name, params: paramsPayload })
                    })
                    .then(res => res.json())
                    .then(data => {
                        if (data.success) {
                            let stdout = data.output || "";
                            if (data.result !== undefined) {
                                stdout += `\n\n[RETURN VALUE]:\n${JSON.stringify(data.result, null, 2)}`;
                            }
                            testStdout.textContent = stdout || "Execution completed with no return value/stdout.";
                        } else {
                            testStdout.textContent = `Error during execution:\n${data.error}\n\nStdout:\n${data.output}`;
                        }
                    })
                    .catch(err => {
                        testStdout.textContent = `Execution failed: ${err}`;
                    })
                    .finally(() => {
                        newExecBtn.disabled = false;
                        newExecBtn.textContent = "⚡ Execute Function";
                    });
                });
            }

            loadAndListFunctions();

            // Bind Save & Compile Button
            const saveBtn = containerElement.querySelector(`#btn-save-tool-${safeId}`);
            saveBtn.addEventListener("click", () => {
                const currentCode = editor.getValue();
                saveBtn.disabled = true;
                saveBtn.textContent = "Compiling & Saving...";

                fetch("/api/save_tool", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title: title, code: currentCode, commit_message: `Manual save update` })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        selectArtifact(title);
                        fetchDiscoveredTools();
                        alert(`Successfully compiled and saved v${data.version}!`);
                    } else {
                        alert(`Compilation failed: ${data.detail}`);
                        saveBtn.disabled = false;
                        saveBtn.textContent = "💾 Save & Compile Tool";
                    }
                })
                .catch(err => {
                    alert(`Save request failed: ${err}`);
                    saveBtn.disabled = false;
                    saveBtn.textContent = "💾 Save & Compile Tool";
                });
            });

            // Bind AI-Assisted Refine Button
            const aiBtn = containerElement.querySelector(`#btn-ai-refine-${safeId}`);
            const aiInput = containerElement.querySelector(`#ai-instruction-${safeId}`);
            aiBtn.addEventListener("click", () => {
                const instruction = aiInput.value.trim();
                if (!instruction) {
                    alert("Please enter a refinement instruction first.");
                    return;
                }

                const currentCode = editor.getValue();
                aiBtn.disabled = true;
                aiBtn.textContent = "AI Executing...";

                // Map attached documents to strings payload
                const docStrings = toolRefDocs[title].map(d => d.content);

                fetch("/api/refine_tool", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: currentCode, instruction: instruction, docs: docStrings })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        editor.setValue(data.code);
                        aiInput.value = "";
                        saveBtn.click();
                    } else {
                        alert(`Refinement failed: ${data.detail}`);
                    }
                })
                .catch(err => alert(`AI request failed: ${err}`))
                .finally(() => {
                    aiBtn.disabled = false;
                    aiBtn.textContent = "🤖 Refine Code";
                });
            });

        } catch (err) {
            containerElement.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Failed to render Tool Editor: ${err}</div>`;
        }
    }

    // ── 📊 Render Spreadsheet Table Grid (Multi-Sheet Excel Support) inside Center Tab ──
    async function renderSpreadsheetGridInTab(docTitle, version, containerElement) {
        containerElement.innerHTML = `<div class="empty-viewer-msg"><span class="spinner inline" style="margin-right: 8px;"></span> Loading dataset spreadsheet...</div>`;
        try {
            let url = `/api/data/${encodeURIComponent(docTitle)}`;
            if (version !== undefined) {
                url += `?version=${version}`;
            }
            const res = await fetch(url);
            const data = await res.json();

            if (data.type === "excel" || data.type === "sqlite") {
                const sheetNames = Object.keys(data.sheets);
                
                const tabsHtml = `
                    <div class="sheet-tabs-container">
                        ${sheetNames.map((s, idx) => `
                            <button class="sheet-tab ${idx === 0 ? 'active' : ''}" data-sheet="${s}">${s}</button>
                        `).join("")}
                    </div>
                    <div class="data-grid-wrapper" id="spreadsheet-grid-target-${makeSafeId(docTitle)}"></div>
                `;
                containerElement.innerHTML = tabsHtml;

                containerElement.querySelectorAll(".sheet-tab").forEach(tab => {
                    tab.addEventListener("click", () => {
                        containerElement.querySelectorAll(".sheet-tab").forEach(t => t.classList.remove("active"));
                        tab.classList.add("active");
                        drawTableInTab(data.sheets[tab.dataset.sheet], docTitle);
                    });
                });

                drawTableInTab(data.sheets[sheetNames[0]], docTitle);

            } else {
                containerElement.innerHTML = `<div class="data-grid-wrapper" id="spreadsheet-grid-target-${makeSafeId(docTitle)}"></div>`;
                drawTableInTab(data, docTitle);
            }
        } catch (err) {
            containerElement.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Failed to render spreadsheet: ${err}</div>`;
        }
    }

    function drawTableInTab(sheetData, docTitle) {
        const gridTarget = document.getElementById(`spreadsheet-grid-target-${makeSafeId(docTitle)}`);
        if (!gridTarget) return;

        const columns = sheetData.columns;
        const rows = sheetData.rows;

        const tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${columns.map(c => `<th>${c}</th>`).join("")}
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(r => `
                        <tr>
                            ${columns.map(c => `<td>${r[c] !== null ? r[c] : ''}</td>`).join("")}
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
        gridTarget.innerHTML = tableHtml;
    }

    // ── 💾 Select and Highlight Sidebar Artifact Cards ──
    async function selectArtifact(title) {
        activeArtifactTitle = title;

        document.querySelectorAll(".artifact-card").forEach(card => {
            card.classList.toggle("selected", card.dataset.title === title);
        });

        try {
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            const matched = arts.find(a => a.title === title);
            if (!matched) return;

            activeArtifactType = matched.type;

            createArtifactTab(title, activeArtifactType);

            const historyRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}/history`);
            const history = await historyRes.json();

            const safeId = makeSafeId(title);
            const vSelect = document.querySelector(`#tab-art-${safeId} .version-select`);
            
            if (vSelect) {
                vSelect.innerHTML = history.map(h => `
                    <option value="${h.version}" ${h.is_active ? 'selected' : ''}>v${h.version} (${h.size_chars} chars)</option>
                `).join("");
            }

            const activeVersionObj = history.find(h => h.is_active) || history[history.length - 1];
            selectedVersion = activeVersionObj ? activeVersionObj.version : 1;

            loadArtifactVersion(title, selectedVersion);

            if (btnExportBundle) btnExportBundle.disabled = false;
            
            updateContextBudget();
        } catch (err) {
            console.error("Failed to select artifact:", err);
        }
    }


    // ── ⚡ Macro Dispatches ──

    function triggerMacroPrompt(promptText) {
        switchCenterTab("tab-chat");
        chatInput.value = promptText;
        sendChatMessage();
    }

    macroSummarize.addEventListener("click", () => {
        triggerMacroPrompt("Generate a comprehensive, professional summary of all currently active user-facing documents and reports. Note: You must strictly EXCLUDE any developer-facing 'skill' or 'tool' type artifacts from this summary. Save your response as a new note artifact by wrapping it in the exact XML tag: <note title=\"summary_report\"> ... </note>");
    });

    macroCompare.addEventListener("click", () => {
        triggerMacroPrompt("Perform a detailed comparative analysis across all currently active user-facing documents and reports, highlighting overlapping concepts, differences, and core recommendations. Note: You must strictly EXCLUDE any developer-facing 'skill' or 'tool' type artifacts from this analysis. Save your response as a new document artifact by wrapping it in the exact XML tag: <artifact name=\"comparative_analysis.md\" type=\"document\"> ... </artifact>");
    });

    macroGraph.addEventListener("click", () => {
        triggerMacroPrompt("Analyze the connections, flow of concepts, and relationships between all currently active artifacts. Generate a Graphviz DOT language graph representing this knowledge map. Save your response as a new code artifact by wrapping it in the exact XML tag: <artifact name=\"knowledge_graph.dot\" type=\"code\" language=\"graphviz\"> ... </artifact>");
    });

    const macroReport = document.getElementById("macro-report");
    const macroInfographic = document.getElementById("macro-infographic");
    const macroPresentation = document.getElementById("macro-presentation");

    macroReport.addEventListener("click", () => {
        triggerMacroPrompt("Generate a highly structured, professional, multi-section technical report synthesizing all currently active user-facing documents. Structure it with an executive summary, detailed analysis chapters, comparison tables, and conclusion sections. Save your response as a new document artifact by wrapping it in the exact XML tag: <artifact name=\"comprehensive_technical_report.md\" type=\"document\"> ... </artifact>");
    });

    macroInfographic.addEventListener("click", () => {
        // Query server settings to verify if TTI is actually active
        fetch("/api/settings")
            .then(res => res.json())
            .then(data => {
                if (data.success && data.tti_binding_name) {
                    triggerMacroPrompt("Fuses the most critical data and insights from the active documents into a visually rich infographic design concept. Write a detailed visual prompt and execute it to generate a stunning infographic using the active TTI engine. Save your response as a new image artifact by wrapping it in the exact XML tag: <artifact name=\"visual_infographic.png\" type=\"image\"> ... </artifact>");
                } else {
                    alert("🎨 Infographic Macro requires an active TTI (Text-to-Image) engine. Please open settings (⚙️) and configure a TTI binding provider first.");
                }
            });
    });

    macroPresentation.addEventListener("click", () => {
        triggerMacroPrompt("Convert the active artifacts into a complete, beautiful slide deck presentation in semantic HTML5. Use custom inline CSS styles for layout, transitions, charts, and data figures, and add speakernotes (using 'data-notes'). Save your response as a new presentation artifact by wrapping it in the exact XML tag: <artifact name=\"slideshow_presentation.html\" type=\"presentation\"> ... </artifact>");
    });

    // ── Local Ingestion Handlers (Relocated into Modal) ──
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files);
        }
    });

    function handleFileSelect(files) {
        selectedFiles = Array.from(files);
        if (selectedFiles.length === 1) {
            dropzone.innerHTML = `<p>📄 Selected: <strong>${selectedFiles[0].name}</strong> (${(selectedFiles[0].size / (1024 * 1024)).toFixed(2)} MB)</p>`;
            if (!customTitle.value) {
                customTitle.value = selectedFiles[0].name.split(".")[0].replace(/\s+/g, "_");
            }
        } else {
            dropzone.innerHTML = `<p>📚 Selected: <strong>${selectedFiles.length} files</strong> (supports multiple uploads)</p>`;
            customTitle.value = "";
        }
    }

    btnSubmit.addEventListener("click", async () => {
        if (selectedFiles.length === 0) {
            alert("Please select one or more documents first.");
            return;
        }

        btnSubmit.disabled = true;
        progressContainer.style.display = "flex";

        for (let i = 0; i < selectedFiles.length; i++) {
            const file = selectedFiles[i];
            const currentNum = i + 1;
            const totalNum = selectedFiles.length;

            btnSubmit.textContent = `Processing Ingestion (${currentNum}/${totalNum})...`;
            progressBarFill.style.width = "0%";
            progressStatusText.textContent = `[${currentNum}/${totalNum}] Uploading ${file.name}...`;

            const formData = new FormData();
            formData.append("file", file);
            formData.append("mode", importMode.value);
            if (selectedFiles.length === 1 && customTitle.value) {
                formData.append("title", customTitle.value);
            }

            try {
                const res = await fetch("/api/import", {
                    method: "POST",
                    body: formData
                });

                if (!res.ok) {
                    throw new Error(`Server returned HTTP ${res.status}`);
                }

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\n\n");
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.trim().startsWith("data: ")) {
                            const rawJson = line.trim().slice(6);
                            const event = JSON.parse(rawJson);
                            handleStreamEvent(event, currentNum, totalNum);
                        }
                    }
                }
            } catch (err) {
                alert(`Ingestion for '${file.name}' failed: ${err}`);
            }
        }

        btnSubmit.disabled = false;
        btnSubmit.textContent = "Process Document";
        selectedFiles = [];
        dropzone.innerHTML = `<p>Drag & drop PDF, DOCX, PPTX, MD, CSV, XLSX, DB, or Image file here, or click to upload (supports multiple files)</p>`;
        customTitle.value = "";
    });

    function handleStreamEvent(event, currentNum = 1, totalNum = 1) {
        if (event.type === "progress") {
            const msg = event.message;
            progressStatusText.textContent = `[${currentNum}/${totalNum}] ${msg}`;

            let percent = 10;
            const lowerMsg = msg.toLowerCase();
            const pageMatch = lowerMsg.match(/(?:page|transcribing page)\s+(\d+)\/(\d+)/);

            if (pageMatch) {
                const current = parseInt(pageMatch[1], 10);
                const total = parseInt(pageMatch[2], 10);
                percent = Math.round(25 + (current / total) * 65);
            } else if (lowerMsg.includes("loading")) {
                percent = 15;
            } else if (lowerMsg.includes("rendering") || lowerMsg.includes("extracting")) {
                percent = 25;
            } else if (lowerMsg.includes("complete") || lowerMsg.includes("retrieving")) {
                percent = 95;
            }

            progressBarFill.style.width = `${percent}%`;

        } else if (event.type === "result") {
            progressBarFill.style.width = "100%";
            progressStatusText.textContent = "Import complete! Resolving document...";
            
            chatInput.disabled = false;
            btnChatSend.disabled = false;

            fetchArtifacts().then(() => {
                selectArtifact(event.title);
                fetchMemories();
                importModal.style.display = "none";
            });

            setTimeout(() => {
                progressContainer.style.display = "none";
            }, 1000);

        } else if (event.type === "error") {
            alert(`Error: ${event.message}`);
            progressContainer.style.display = "none";
        }
    }

    // ── Internet Ingestion Handlers (Relocated into Modal) ──
    internetSourceType.addEventListener("change", (e) => {
        const type = e.target.value;
        document.querySelectorAll(".internet-fields").forEach(f => f.style.display = "none");
        
        const fieldBlock = document.getElementById(`fields-${type}`);
        if (fieldBlock) {
            fieldBlock.style.display = "block";
        }

        if (type === "youtube" || type === "github" || type === "stackoverflow") {
            btnInternetSearch.textContent = "Import URL directly";
        } else {
            btnInternetSearch.textContent = "Search / Ingest";
        }

        searchResultsContainer.style.display = "none";
        searchResultsList.innerHTML = "";
    });

    btnInternetSearch.addEventListener("click", async () => {
        const type = internetSourceType.value;
        btnInternetSearch.disabled = true;
        btnInternetSearch.textContent = "Processing...";

        try {
            if (type === "youtube") {
                const url = document.getElementById("youtube-url").value.trim();
                const lang = document.getElementById("youtube-language").value.trim() || "en";
                if (!url) { alert("Please specify a YouTube URL."); return; }
                await importDirectURL("/api/youtube/import", { url: url, language: lang });
            } 
            else if (type === "github" && document.getElementById("github-url").value.trim().startsWith("http")) {
                const url = document.getElementById("github-url").value.trim();
                await importDirectURL("/api/github/import", { url: url });
            } 
            else if (type === "stackoverflow" && document.getElementById("stackoverflow-url").value.trim().startsWith("http")) {
                const url = document.getElementById("stackoverflow-url").value.trim();
                await importDirectURL("/api/stackoverflow/import", { url: url });
            } 
            else {
                await executeSearch(type);
            }
        } catch (err) {
            alert(`Search/Ingestion failed: ${err}`);
        } finally {
            btnInternetSearch.disabled = false;
            btnInternetSearch.textContent = (type === "youtube" || type === "github" || type === "stackoverflow") ? "Import URL directly" : "Search / Ingest";
        }
    });

    async function importDirectURL(endpoint, payload) {
        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                alert("Successfully imported content as discussion artifact!");
                importModal.style.display = "none";
                fetchArtifacts();
            } else {
                alert(`Import failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Direct URL import request failed: ${err}`);
        }
    }

    async function executeSearch(type) {
        let endpoint = "";
        let payload = {};

        if (type === "web") {
            const q = document.getElementById("web-query").value.trim();
            if (!q) { alert("Please specify keywords."); return; }
            endpoint = "/api/web/search";
            payload = { query: q, provider: "duckduckgo" };
        } else if (type === "wikipedia") {
            const q = document.getElementById("wiki-query").value.trim();
            if (!q) { alert("Please specify article title."); return; }
            endpoint = "/api/wikipedia/search";
            payload = { query: q };
        } else if (type === "arxiv") {
            const q = document.getElementById("arxiv-query").value.trim();
            const author = document.getElementById("arxiv-author").value.trim();
            if (!q && !author) { alert("Please specify either keywords or author."); return; }
            endpoint = "/api/arxiv/search";
            payload = { query: q || null, author: author || null, max_results: 5 };
        } else if (type === "github") {
            const q = document.getElementById("github-url").value.trim();
            if (!q) { alert("Please specify GitHub search keywords."); return; }
            endpoint = "/api/github/search";
            payload = { query: q };
        } else if (type === "stackoverflow") {
            const q = document.getElementById("stackoverflow-url").value.trim();
            if (!q) { alert("Please specify StackOverflow search keywords."); return; }
            endpoint = "/api/stackoverflow/search";
            payload = { query: q };
        }

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success && data.results.length > 0) {
                activeSearchResults = data.results;
                renderSearchResults(type, data.results);
            } else {
                searchResultsContainer.style.display = "none";
                alert("No search results found.");
            }
        } catch (err) {
            alert(`Search request failed: ${err}`);
        }
    }

    function renderSearchResults(type, results) {
        searchResultsList.innerHTML = results.map((r, idx) => {
            const url = r.url || r.pdf_url || (r.id ? `https://arxiv.org/abs/${r.id}` : "#");
            const titleHtml = url !== "#" 
                ? `<a href="${url}" target="_blank" class="search-result-link" title="Open source link in new tab">${r.title} 🔗</a>` 
                : `<span class="search-result-title">${r.title}</span>`;

            return `
                <li class="search-result-card">
                    <input type="checkbox" class="result-check" data-index="${idx}">
                    <div class="search-result-content">
                        ${titleHtml}
                        <span class="search-result-snippet">${r.snippet || r.abstract || r.url || ""}</span>
                    </div>
                </li>
            `;
        }).join("");

        searchResultsContainer.style.display = "block";
        btnInternetImportSelected.disabled = false;

        document.querySelectorAll(".result-check").forEach(chk => {
            chk.addEventListener("change", () => {
                const anyChecked = Array.from(document.querySelectorAll(".result-check")).some(c => c.checked);
                btnInternetImportSelected.disabled = !anyChecked;
            });
        });
    }

    btnInternetImportSelected.addEventListener("click", async () => {
        const checkedBoxes = Array.from(document.querySelectorAll(".result-check:checked"));
        if (checkedBoxes.length === 0) return;

        btnInternetImportSelected.disabled = true;
        btnInternetImportSelected.textContent = "Importing...";

        const selectedItems = checkedBoxes.map(chk => activeSearchResults[parseInt(chk.dataset.index, 10)]);
        const type = internetSourceType.value;

        let endpoint = "";
        let payload = {};

        if (type === "wikipedia") {
            endpoint = "/api/wikipedia/import";
            payload = {
                items: selectedItems.map(item => ({ title: item.title, url: item.url })),
                auto_load: true
            };
        } else if (type === "arxiv") {
            endpoint = "/api/arxiv/import";
            const mode = document.getElementById("arxiv-mode").value;
            payload = {
                items: selectedItems.map(item => ({ id: item.id, title: item.title, mode: mode })),
                auto_load: true
            };
        } else {
            for (const item of selectedItems) {
                let directEndpoint = "";
                if (item.url.includes("github.com")) {
                    directEndpoint = "/api/github/import";
                } else if (item.url.includes("stackoverflow.com")) {
                    directEndpoint = "/api/stackoverflow/import";
                } else {
                    directEndpoint = "/api/import_url";
                }
                await importDirectURL(directEndpoint, { url: item.url });
            }
            btnInternetImportSelected.disabled = false;
            btnInternetImportSelected.textContent = "Import Selected";
            searchResultsContainer.style.display = "none";
            importModal.style.display = "none";
            return;
        }

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                alert(`Successfully imported ${selectedItems.length} selected items into session!`);
                searchResultsContainer.style.display = "none";
                importModal.style.display = "none";
                fetchArtifacts();
            } else {
                alert(`Import failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Import selected request failed: ${err}`);
        } finally {
            btnInternetImportSelected.disabled = false;
            btnInternetImportSelected.textContent = "Import Selected";
        }
    });

    // ── Skills & Scanning Handlers (Relocated into Modal) ──
    btnImportSingleSkill.addEventListener("click", () => singleSkillInput.click());

    singleSkillInput.addEventListener("change", async (e) => {
        if (e.target.files.length === 0) return;
        
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", "text");

        btnImportSingleSkill.disabled = true;
        btnImportSingleSkill.textContent = "Loading...";

        try {
            const res = await fetch("/api/import", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error("Server returned HTTP " + res.status);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.trim().startsWith("data: ")) {
                        const rawJson = line.trim().slice(6);
                        const event = JSON.parse(rawJson);
                        if (event.type === "result") {
                            alert("Successfully loaded skill: '" + event.title + "'!");
                            importModal.style.display = "none";
                            fetchArtifacts();
                        } else if (event.type === "error") {
                            alert("Error: " + event.message);
                        }
                    }
                }
            }
        } catch (err) {
            alert("Failed to load skill file: " + err);
        } finally {
            btnImportSingleSkill.disabled = false;
            btnImportSingleSkill.textContent = "Load Single Skill File";
            singleSkillInput.value = "";
        }
    });

    btnScanSkills.addEventListener("click", async () => {
        const path = skillsScanPath.value.trim();
        if (!path) {
            alert("Please specify a directory path to scan.");
            return;
        }

        btnScanSkills.disabled = true;
        btnScanSkills.textContent = "Scanning...";

        try {
            const res = await fetch("/api/scan_skills", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path })
            });
            const data = await res.json();

            if (data.success) {
                alert(`Successfully scanned and loaded ${data.count} skill(s) from folder!`);
                importModal.style.display = "none";
                fetchArtifacts();
            } else {
                alert(`Scanning failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Scan request failed: ${err}`);
        } finally {
            btnScanSkills.disabled = false;
            btnScanSkills.textContent = "Scan & Load Folder";
        }
    });

    // ── Portable Bundle Import (Relocated into Modal) ──
    btnImportBundle.addEventListener("click", () => bundleInput.click());

    bundleInput.addEventListener("change", async (e) => {
        if (e.target.files.length === 0) return;

        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        try {
            btnImportBundle.disabled = true;
            btnImportBundle.textContent = "Restoring...";

            const res = await fetch("/api/import_bundle", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                activeArtifactTitle = data.title;
                btnExportBundle.disabled = false;

                chatInput.disabled = false;
                btnChatSend.disabled = false;

                fetchArtifacts().then(() => {
                    selectArtifact(data.title);
                    fetchMemories();
                    importModal.style.display = "none";
                });
                alert(`Successfully imported bundle '${data.title}'! Workspace companion is active.`);
            } else {
                alert(`Import failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Bundle upload request failed: ${err}`);
        } finally {
            btnImportBundle.disabled = false;
            btnImportBundle.textContent = "Import Bundle File";
            bundleInput.value = "";
        }
    });

    // ── Custom Personality Zip Upload Ingestion ──
    const btnImportPersonality = document.getElementById("btn-import-personality");
    const personalityZipInput = document.getElementById("personality-zip-input");

    btnImportPersonality.addEventListener("click", () => personalityZipInput.click());

    personalityZipInput.addEventListener("change", async (e) => {
        if (e.target.files.length === 0) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        btnImportPersonality.disabled = true;
        btnImportPersonality.textContent = "Importing...";

        try {
            const res = await fetch("/api/personalities/import", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                alert(`Successfully imported custom personality '${data.persona}'!`);
                await fetchPersonalities();
                activatePersonality(data.category, data.persona);
            } else {
                alert(`Import failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Request failed: ${err}`);
        } finally {
            btnImportPersonality.disabled = false;
            btnImportPersonality.textContent = "Import Zipped Personality";
            personalityZipInput.value = "";
        }
    });

    // ── Download Personality from Zoo ──
    const btnDownloadZooPersona = document.getElementById("btn-download-zoo-persona");
    btnDownloadZooPersona.addEventListener("click", async () => {
        const category = document.getElementById("zoo-category").value.trim().toLowerCase();
        const persona = document.getElementById("zoo-persona").value.trim().toLowerCase();

        if (!category || !persona) {
            alert("Please specify both a category and a persona name.");
            return;
        }

        btnDownloadZooPersona.disabled = true;
        btnDownloadZooPersona.textContent = "Downloading from Zoo...";

        try {
            const res = await fetch("/api/personalities/download_zoo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ category, persona })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Successfully downloaded and installed '${persona}' from the Zoo!`);
                document.getElementById("zoo-category").value = "";
                document.getElementById("zoo-persona").value = "";
                await fetchPersonalities();
                activatePersonality(data.category, data.persona);
            } else {
                alert(`Failed to download: ${data.detail || "Make sure the category and persona exist in the repository."}`);
            }
        } catch (err) {
            alert(`Download request failed: ${err}`);
        } finally {
            btnDownloadZooPersona.disabled = false;
            btnDownloadZooPersona.textContent = "Download & Install";
        }
    });

    // ── Header Dropdown Personality Select Event ──
    headerPersonality.addEventListener("change", () => {
        const val = headerPersonality.value;
        if (!val) return;
        const [category, persona] = val.split("/");
        activatePersonality(category, persona);
    });

    // ── Header Dropdown TTI Model Select Event ──
    headerTtiModel.addEventListener("change", async () => {
        const val = headerTtiModel.value;
        if (!val) return;
        headerTtiModelSpinner.style.display = "inline-block";
        try {
            const settingsRes = await fetch("/api/settings");
            const settingsData = await settingsRes.json();
            if (settingsData.success) {
                const config = settingsData.tti_binding_config || {};
                config.model_name = val;

                const applyRes = await fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        llm_binding_name: settingsData.llm_binding_name,
                        llm_binding_config: settingsData.llm_binding_config || {},
                        tti_binding_name: settingsData.tti_binding_name || null,
                        tti_binding_config: config,
                        personality_name: settingsData.personality_name || null
                    })
                });
                const applyData = await applyRes.json();
                if (applyData.success) {
                    ASCIIColors.green(`Successfully switched TTI model to ${val}`);
                } else {
                    alert(`Failed to switch TTI model: ${applyData.error}`);
                }
            }
        } catch (err) {
            alert(`TTI Model switch request failed: ${err}`);
        } finally {
            headerTtiModelSpinner.style.display = "none";
        }
    });

    // ── Header Dropdown Model Select Event ──
    headerModel.addEventListener("change", async () => {
        const val = headerModel.value;
        if (!val) return;
        headerModelSpinner.style.display = "inline-block";
        try {
            // Apply model switch directly to the active client config
            const settingsRes = await fetch("/api/settings");
            const settingsData = await settingsRes.json();
            if (settingsData.success) {
                const config = settingsData.llm_binding_config || {};
                config.model_name = val;

                const applyRes = await fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        llm_binding_name: settingsData.llm_binding_name,
                        llm_binding_config: config,
                        tti_binding_name: settingsData.tti_binding_name || null,
                        tti_binding_config: settingsData.tti_binding_config || null,
                        personality_name: settingsData.personality_name || null
                    })
                });
                const applyData = await applyRes.json();
                if (applyData.success) {
                    ASCIIColors.green(`Successfully switched model to ${val}`);
                } else {
                    alert(`Failed to switch model: ${applyData.error}`);
                }
            }
        } catch (err) {
            alert(`Model switch request failed: ${err}`);
        } finally {
            headerModelSpinner.style.display = "none";
        }
    });

    // ── Export Portable Bundle ──
    if (btnExportBundle) {
        btnExportBundle.addEventListener("click", async () => {
            if (!activeArtifactTitle) return;
            try {
                const res = await fetch(`/api/bundle/${activeArtifactTitle}`);
                const bundle = await res.json();

                const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${activeArtifactTitle}_bundle.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (err) {
                alert(`Failed to package bundle: ${err}`);
            }
        });
    }

    // ── 🧠 Fetch and Render Discovered Local LCP Tools ──
    async function fetchDiscoveredTools() {
        try {
            const res = await fetch("/api/tools");
            const tools = await res.json();

            if (tools.length === 0) {
                toolsList.innerHTML = `<li class="empty-msg">No discovered tools found.</li>`;
                return;
            }

            toolsList.innerHTML = tools.map(t => {
                const isActive = t.active !== false;
                const toggleIcon = isActive ? "👁️" : "💤";
                const toggleTitle = isActive ? "Deactivate (exclude from context)" : "Activate (include in context)";
                const inactiveClass = isActive ? "" : "inactive";
                return `
                    <li class="tool-card ${inactiveClass}" data-tool-name="${t.name}">
                        <div class="artifact-card-header">
                            <span class="tool-title">🔧 ${t.name}</span>
                            <div class="artifact-actions">
                                <button class="artifact-action-btn toggle" data-tool-name="${t.name}" title="${toggleTitle}">${toggleIcon}</button>
                            </div>
                        </div>
                        <p class="tool-desc">${t.description || 'No description provided.'}</p>
                    </li>
                `;
            }).join("");

            toolsList.querySelectorAll(".artifact-action-btn.toggle").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const name = btn.dataset.toolName;
                    try {
                        const res = await fetch(`/api/tools/${encodeURIComponent(name)}/toggle`, { method: "POST" });
                        const data = await res.json();
                        if (data.success) {
                            fetchDiscoveredTools();
                            updateContextBudget();
                        }
                    } catch (err) {
                        console.error("Failed to toggle tool:", err);
                    }
                });
            });
        } catch (err) {
            console.error("Failed to fetch discovered tools:", err);
        }
    }

    // ── 📤 Contextual Export Modal Manager Subroutine ──
    let activeExportType = null; // "artifacts" | "memories" | "tools"

    const btnExportArtifacts = document.getElementById("btn-export-artifacts-part");
    const btnExportMemories = document.getElementById("btn-export-memories-part");
    const btnExportTools = document.getElementById("btn-export-tools-part");

    function openExportModal(type) {
        activeExportType = type;
        exportModalBody.innerHTML = ""; // Clear
        exportDetailsModal.style.display = "flex";

        if (type === "artifacts") {
            exportModalTitle.textContent = "📤 Export Discussion Artifacts";

            // Build artifact dropdown & formatting selection
            fetch("/api/artifacts")
                .then(res => res.json())
                .then(arts => {
                    const latestArts = [];
                    const seen = {};
                    for (const a of arts) {
                        const t = a.title;
                        if (!seen[t] || a.version > seen[t].version) {
                            seen[t] = a;
                        }
                    }
                    Object.values(seen).forEach(a => latestArts.push(a));

                    if (latestArts.length === 0) {
                        exportModalBody.innerHTML = `<p class="empty-msg">No artifacts available to export.</p>`;
                        btnConfirmExport.disabled = true;
                        return;
                    }

                    btnConfirmExport.disabled = false;
                    exportModalBody.innerHTML = `
                        <div class="input-group">
                            <label for="export-art-select">Select Artifact</label>
                            <select id="export-art-select">
                                ${latestArts.map(a => `<option value="${a.title}" data-type="${a.type}" data-ver="${a.version}">${a.title} (v${a.version})</option>`).join("")}
                            </select>
                        </div>
                        <div class="input-group" id="export-art-format-group">
                            <label for="export-art-format">Export Format</label>
                            <select id="export-art-format">
                                <option value="markdown">Markdown (.md)</option>
                                <option value="html">HTML Webpage (.html)</option>
                                <option value="pdf">PDF Document (.pdf)</option>
                                <option value="docx">Microsoft Word (.docx)</option>
                                <option value="pptx">Microsoft PowerPoint (.pptx)</option>
                            </select>
                        </div>
                        <div class="input-group" style="display: flex; flex-direction: row; gap: 8px; align-items: center; margin-top: 10px;">
                            <input type="checkbox" id="export-as-bundle-check" style="width: auto; cursor: pointer; accent-color: var(--accent-color);">
                            <label for="export-as-bundle-check" style="cursor: pointer; text-transform: none;">Package as Portable Bundle (.json)</label>
                        </div>
                    `;

                    // Handle format adapting based on type (disable pptx for non-slides, etc.)
                    const artSelect = document.getElementById("export-art-select");
                    const artFormat = document.getElementById("export-art-format");
                    const bundleCheck = document.getElementById("export-as-bundle-check");

                    function adaptFormats() {
                        const selectedOption = artSelect.options[artSelect.selectedIndex];
                        const atype = selectedOption.dataset.type;
                        const isData = atype === "data";
                        const isLatex = atype === "latex";

                        // Reset
                        bundleCheck.disabled = false;

                        Array.from(artFormat.options).forEach(opt => {
                            if (isData) {
                                opt.disabled = (opt.value !== "csv" && opt.value !== "excel");
                            } else if (isLatex) {
                                opt.disabled = (opt.value !== "pdf" && opt.value !== "markdown" && opt.value !== "tex");
                                if (opt.value === "markdown") {
                                    opt.textContent = "LaTeX Source (.tex)";
                                    opt.value = "tex";
                                }
                            } else {
                                opt.disabled = (opt.value === "csv" || opt.value === "excel" || opt.value === "tex");
                                if (opt.value === "tex") {
                                    opt.textContent = "Markdown (.md)";
                                    opt.value = "markdown";
                                }
                            }
                        });

                        if (isData) {
                            artFormat.innerHTML = `
                                <option value="csv" selected>CSV Spreadsheet (.csv)</option>
                                <option value="excel">Microsoft Excel (.xlsx)</option>
                            `;
                            bundleCheck.disabled = true; // No bundling for pure spreadsheet grids
                            bundleCheck.checked = false;
                        } else {
                            artFormat.innerHTML = `
                                <option value="markdown" selected>Markdown (.md)</option>
                                <option value="html">HTML Webpage (.html)</option>
                                <option value="pdf">PDF Document (.pdf)</option>
                                <option value="docx">Microsoft Word (.docx)</option>
                                <option value="pptx">Microsoft PowerPoint (.pptx)</option>
                                <option value="tex">LaTeX Source (.tex)</option>
                            `;
                            // Disable incompatible options for presentation etc.
                        }
                    }

                    artSelect.addEventListener("change", adaptFormats);
                    adaptFormats();
                });

        } else if (type === "memories") {
            exportModalTitle.textContent = "📤 Export Persistent Memories";
            btnConfirmExport.disabled = false;
            exportModalBody.innerHTML = `
                <div class="input-group">
                    <label for="export-mem-scope">Memory Scope</label>
                    <select id="export-mem-scope">
                        <option value="all" selected>All Memories (Complete Long-Term Database)</option>
                        <option value="working">Active Working Memory (Level 1)</option>
                        <option value="deep">Latent Deep Memory (Level 2)</option>
                        <option value="archived">Archived Memory (Level 3)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="export-mem-format">Format</label>
                    <select id="export-mem-format">
                        <option value="json" selected>JSON Database Extract (.json)</option>
                        <option value="csv">CSV Spreadsheet (.csv)</option>
                    </select>
                </div>
            `;

        } else if (type === "tools") {
            exportModalTitle.textContent = "📤 Export Discovered LCP Tools";

            fetch("/api/tools")
                .then(res => res.json())
                .then(tools => {
                    if (tools.length === 0) {
                        exportModalBody.innerHTML = `<p class="empty-msg">No LCP tools available to export.</p>`;
                        btnConfirmExport.disabled = true;
                        return;
                    }

                    btnConfirmExport.disabled = false;
                    exportModalBody.innerHTML = `
                        <p class="section-desc">Select which active LCP python tool files to package and download.</p>
                        <div class="export-checklist-container">
                            ${tools.map(t => `
                                <label class="export-check-item">
                                    <input type="checkbox" class="tool-export-check" value="${t.name}" checked>
                                    <span>🔧 ${t.name}</span>
                                </label>
                            `).join("")}
                        </div>
                        <div class="input-group" style="margin-top: 14px;">
                            <label for="export-tool-format">Export Format</label>
                            <select id="export-tool-format">
                                <option value="py" selected>Standalone Python Files (.zip Archive)</option>
                                <option value="json">Standard LCP Schema (.json Bundle)</option>
                            </select>
                        </div>
                    `;
                });
        }
    }

    btnExportArtifacts.addEventListener("click", () => openExportModal("artifacts"));
    btnExportMemories.addEventListener("click", () => openExportModal("memories"));
    btnExportTools.addEventListener("click", () => openExportModal("tools"));

    // ── ⚙️ Settings Modal Sub-System ──
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnCancelSettings = document.getElementById("btn-cancel-settings");
    const settingsModal = document.getElementById("settings-modal");
    const settingsTabs = document.querySelectorAll(".modal-tab[data-settings-tab]");
    const settingsPanes = document.querySelectorAll(".settings-tab-pane");

    btnOpenSettings.addEventListener("click", () => {
        settingsModal.style.display = "flex";
        fetchBindings();
    });

    const closeSettingsModal = () => {
        settingsModal.style.display = "none";
        valStatus.className = "validation-status";
        valStatus.textContent = "";
        valTtiStatus.className = "validation-status";
        valTtiStatus.textContent = "";
    };

    btnCloseSettings.addEventListener("click", closeSettingsModal);
    if (btnCancelSettings) {
        btnCancelSettings.addEventListener("click", closeSettingsModal);
    }

    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            settingsModal.style.display = "none";
        }
    });

    settingsTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            if (tab.disabled) return;
            settingsTabs.forEach(t => t.classList.remove("active"));
            settingsPanes.forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(tab.dataset.settingsTab).classList.add("active");
        });
    });

    selModel.addEventListener("change", () => {
        const infoDiv = document.getElementById("settings-model-info");
        const opt = selModel.options[selModel.selectedIndex];
        if (!opt) return;
        const parts = [];
        if (opt.dataset.ownedBy) parts.push(`Owner: ${opt.dataset.ownedBy}`);
        if (opt.dataset.created) {
            try { parts.push(`Created: ${new Date(opt.dataset.created).toLocaleDateString()}`); } catch {}
        }
        if (opt.dataset.size) parts.push(`Size: ${opt.dataset.size}`);
        if (parts.length > 0) {
            infoDiv.textContent = parts.join(" · ");
            infoDiv.style.display = "block";
        } else {
            infoDiv.style.display = "none";
        }
    });

    async function fetchTtiBindings() {
        try {
            const res = await fetch("/api/bindings/tti");
            const data = await res.json();
            if (data.success) {
                availableTtiBindings = data.bindings;
                selTtiBinding.innerHTML = `<option value="">-- Select TTI Binding --</option>` +
                    availableTtiBindings.map(b => `<option value="${b.binding_name}">${b.title || b.binding_name}</option>`).join("");
            }
        } catch (err) {
            console.error("Failed to fetch TTI bindings:", err);
        }
    }

    function renderTtiConfigForm(bindingName) {
        const binding = availableTtiBindings.find(b => b.binding_name === bindingName);
        if (!binding || !binding.input_parameters) {
            ttiConfigForm.innerHTML = `<p class="empty-msg">No configurable parameters for this TTI binding.</p>`;
            return;
        }

        ttiConfigForm.innerHTML = binding.input_parameters.map(p => {
            if (p.name === "model_name") return "";
            const key = p.name;
            const label = p.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
            const type = p.type || "string";
            const defaultVal = p.default !== undefined ? p.default : "";
            const desc = p.description || "";

            if (type === "bool" || type === "boolean") {
                return `
                    <div class="input-group">
                        <label style="flex-direction:row; align-items:center; gap:8px; text-transform:none;">
                            <input type="checkbox" id="tti-cfg-${key}" data-tti-key="${key}" ${defaultVal ? "checked" : ""} />
                            <span>${label}</span>
                        </label>
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (type === "int" || type === "integer" || type === "float" || type === "number") {
                return `
                    <div class="input-group">
                        <label for="tti-cfg-${key}">${label}</label>
                        <input type="number" id="tti-cfg-${key}" data-tti-key="${key}" data-type="${type}" value="${defaultVal}" step="${type === 'float' || type === 'number' ? '0.01' : '1'}" />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else {
                return `
                    <div class="input-group">
                        <label for="tti-cfg-${key}">${label}</label>
                        <input type="text" id="tti-cfg-${key}" data-tti-key="${key}" value="${defaultVal}" placeholder="${desc}" />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            }
        }).join("");
    }

    function collectTtiBindingConfig() {
        const inputs = ttiConfigForm.querySelectorAll("[data-tti-key]");
        const cfg = {};
        inputs.forEach(inp => {
            const key = inp.dataset.ttiKey || inp.getAttribute("data-tti-key");
            const type = inp.dataset.type || "string";
            if (inp.type === "checkbox") {
                cfg[key] = inp.checked;
            } else if (type === "int" || type === "integer") {
                cfg[key] = parseInt(inp.value, 10);
            } else if (type === "float" || type === "number") {
                cfg[key] = parseFloat(inp.value);
            } else {
                cfg[key] = inp.value;
            }
        });
        return cfg;
    }

    function renderTtiModelDropdown(modelsList) {
        selTtiModel.innerHTML = modelsList.map(m => {
            const value = m.model_name || m.name || m.id || m;
            const label = m.display_name || value;
            return `<option value="${value}">${label}</option>`;
        }).join("");
        selTtiModel.dispatchEvent(new Event("change"));
    }

    // Attach real-time filter event for TTI Models search bar
    ttiModelSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().strip ? e.target.value.toLowerCase().strip() : e.target.value.toLowerCase().trim();
        if (!query) {
            renderTtiModelDropdown(discoveredTtiModels);
            return;
        }

        const filtered = discoveredTtiModels.filter(m => {
            const value = m.model_name || m.name || m.id || m;
            const label = m.display_name || value;
            return value.toLowerCase().includes(query) || label.toLowerCase().includes(query);
        });

        renderTtiModelDropdown(filtered);
    });

    selTtiBinding.addEventListener("change", () => {
        ttiModelGroup.style.display = "none";
        valTtiStatus.className = "validation-status";
        valTtiStatus.textContent = "";
        renderTtiConfigForm(selTtiBinding.value);
    });

    btnValidateTti.addEventListener("click", async () => {
        if (!selTtiBinding.value) {
            valTtiStatus.className = "validation-status error";
            valTtiStatus.textContent = "Please select a TTI binding first.";
            return;
        }
        btnValidateTti.disabled = true;
        btnValidateTti.textContent = "Validating...";
        valTtiStatus.className = "validation-status";
        valTtiStatus.textContent = "Connecting to TTI service...";

        const payload = {
            binding_name: selTtiBinding.value,
            config: collectTtiBindingConfig()
        };

        try {
            const res = await fetch("/api/bindings/tti/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success && Array.isArray(data.models)) {
                valTtiStatus.className = "validation-status success";
                valTtiStatus.textContent = `✅ Connection successful. Found ${data.models.length} model(s) / styles.`;

                // Store complete TTI models list in-memory
                discoveredTtiModels = data.models;
                ttiModelSearch.value = ""; // Clear previous search input

                renderTtiModelDropdown(discoveredTtiModels);
                ttiModelGroup.style.display = "flex";
                currentTtiBindingConfig = payload.config;
            } else {
                valTtiStatus.className = "validation-status error";
                valTtiStatus.textContent = `❌ Validation failed: ${data.error || "Could not reach TTI process."}`;
                ttiModelGroup.style.display = "none";
            }
        } catch (err) {
            valTtiStatus.className = "validation-status error";
            valTtiStatus.textContent = `❌ Request failed: ${err}`;
        } finally {
            btnValidateTti.disabled = false;
            btnValidateTti.textContent = "🔌 Validate & Load Models";
        }
    });

    // ── 📂 Profiles Management Sub-System ──
    async function fetchProfiles() {
        try {
            const res = await fetch("/api/profiles");
            const data = await res.json();
            if (data.success && Array.isArray(data.profiles)) {
                if (data.profiles.length === 0) {
                    settingsProfilesList.innerHTML = `<li class="empty-msg">No profiles saved yet.</li>`;
                    return;
                }

                settingsProfilesList.innerHTML = data.profiles.map(p => `
                    <li class="artifact-card" style="padding: 10px 14px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: bold; font-size: 13px;">📁 ${p.name.replace(/_/g, ' ').toUpperCase()}</span>
                            <span class="details" style="display: block; font-size: 10px; color: var(--text-secondary);">Saved: ${p.created_at}</span>
                        </div>
                        <div class="artifact-actions">
                            <button class="artifact-action-btn load-profile-btn" data-name="${p.name}" title="Load profile">⚡ Load</button>
                            <button class="artifact-action-btn delete-profile-btn" data-name="${p.name}" title="Delete profile" style="color: #ef4444;">✕</button>
                        </div>
                    </li>
                `).join("");

                settingsProfilesList.querySelectorAll(".load-profile-btn").forEach(btn => {
                    btn.addEventListener("click", () => loadProfile(btn.dataset.name));
                });
                settingsProfilesList.querySelectorAll(".delete-profile-btn").forEach(btn => {
                    btn.addEventListener("click", () => deleteProfile(btn.dataset.name));
                });
            }
        } catch (err) {
            console.error("Failed to fetch profiles:", err);
        }
    }

    async function loadProfile(name) {
        if (!confirm(`Are you sure you want to load the profile '${name.toUpperCase()}'? This will reinitialize active model servers.`)) {
            return;
        }
        try {
            const res = await fetch("/api/profiles/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Profile '${name.toUpperCase()}' loaded successfully!`);
                settingsModal.style.display = "none";
                location.reload(); // Refresh to clean and initialize UI state
            } else {
                alert(`Failed to load profile: ${data.detail}`);
            }
        } catch (err) {
            alert(`Load profile failed: ${err}`);
        }
    }

    async function deleteProfile(name) {
        if (!confirm(`Delete profile '${name.toUpperCase()}'?`)) return;
        try {
            const res = await fetch(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE" });
            if (res.ok) fetchProfiles();
        } catch (err) {
            console.error("Failed to delete profile:", err);
        }
    }

    btnSaveProfile.addEventListener("click", async () => {
        const name = profileSaveName.value.trim();
        if (!name) {
            alert("Please enter a profile name first.");
            return;
        }

        btnSaveProfile.disabled = true;
        btnSaveProfile.textContent = "Saving...";

        try {
            const res = await fetch("/api/profiles/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Current settings saved under profile '${name.toUpperCase()}'!`);
                profileSaveName.value = "";
                await fetchProfiles();
            } else {
                alert(`Failed to save: ${data.detail}`);
            }
        } catch (err) {
            alert(`Request failed: ${err}`);
        } finally {
            btnSaveProfile.disabled = false;
            btnSaveProfile.textContent = "Save Profile";
        }
    });


    async function fetchBindings() {
        try {
            const res = await fetch("/api/bindings/llm");
            const data = await res.json();
            if (data.success) {
                availableBindings = data.bindings;
                selBinding.innerHTML = `<option value="">-- Select Binding --</option>` +
                    availableBindings.map(b => `<option value="${b.binding_name}">${b.title || b.binding_name}</option>`).join("");
            }
        } catch (err) {
            console.error("Failed to fetch bindings:", err);
        }
    }

    function renderConfigForm(bindingName) {
        const binding = availableBindings.find(b => b.binding_name === bindingName);
        if (!binding || !binding.input_parameters) {
            configForm.innerHTML = `<p class="empty-msg">No configurable parameters for this binding.</p>`;
            return;
        }

        configForm.innerHTML = binding.input_parameters.map(p => {
            if (p.name === "model_name") return ""; // model selection happens after validation
            const key = p.name;
            const label = p.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
            const type = p.type || "string";
            const required = p.mandatory ? "required" : "";
            const defaultVal = p.default !== undefined ? p.default : "";
            const desc = p.description || "";

            const isSecret = key.toLowerCase().includes("key") || key.toLowerCase().includes("token") || p.is_secret;

            if (type === "bool" || type === "boolean") {
                return `
                    <div class="input-group">
                        <label style="flex-direction:row; align-items:center; gap:8px; text-transform:none;">
                            <input type="checkbox" id="cfg-${key}" data-key="${key}" ${defaultVal ? "checked" : ""} />
                            <span>${label} ${p.mandatory ? '<span style="color:#ef4444">*</span>' : ''}</span>
                        </label>
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (type === "list") {
                return `
                    <div class="input-group">
                        <label for="cfg-${key}">${label} ${p.mandatory ? '<span style="color:#ef4444">*</span>' : ''}</label>
                        <input type="text" id="cfg-${key}" data-key="${key}" data-type="list" value="${Array.isArray(defaultVal) ? JSON.stringify(defaultVal) : defaultVal}" placeholder='JSON array, e.g. [&quot;item1&quot;]' ${required} />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (type === "int" || type === "integer" || type === "float" || type === "number") {
                return `
                    <div class="input-group">
                        <label for="cfg-${key}">${label} ${p.mandatory ? '<span style="color:#ef4444">*</span>' : ''}</label>
                        <input type="number" id="cfg-${key}" data-key="${key}" data-type="${type}" value="${defaultVal}" step="${type === 'float' || type === 'number' ? '0.01' : '1'}" ${required} />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (isSecret) {
                return `
                    <div class="input-group">
                        <label for="cfg-${key}">${label} ${p.mandatory ? '<span style="color:#ef4444">*</span>' : ''}</label>
                        <div style="position: relative; display: flex; align-items: center; width: 100%;">
                            <input type="password" id="cfg-${key}" data-key="${key}" value="${defaultVal}" placeholder="${desc}" style="padding-right: 40px;" ${required} />
                            <button type="button" class="btn-toggle-secret" data-target="cfg-${key}" style="position: absolute; right: 8px; background: none; border: none; color: var(--accent-color); cursor: pointer; font-size: 16px; outline: none; padding: 4px;" title="Toggle Visibility">👁️</button>
                        </div>
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else {
                return `
                    <div class="input-group">
                        <label for="cfg-${key}">${label} ${p.mandatory ? '<span style="color:#ef4444">*</span>' : ''}</label>
                        <input type="text" id="cfg-${key}" data-key="${key}" value="${defaultVal}" placeholder="${desc}" ${required} />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            }
        }).join("");

        // Attach event listeners for secret toggles
        configForm.querySelectorAll(".btn-toggle-secret").forEach(btn => {
            btn.addEventListener("click", () => {
                const targetInput = document.getElementById(btn.dataset.target);
                if (targetInput) {
                    const isPassword = targetInput.type === "password";
                    targetInput.type = isPassword ? "text" : "password";
                    btn.textContent = isPassword ? "🔒" : "👁️";
                }
            });
        });
    }

    function collectBindingConfig() {
        const inputs = configForm.querySelectorAll("[data-key]");
        const cfg = {};
        inputs.forEach(inp => {
            const key = inp.dataset.key;
            const type = inp.dataset.type || "string";
            if (inp.type === "checkbox") {
                cfg[key] = inp.checked;
            } else if (type === "list") {
                try { cfg[key] = JSON.parse(inp.value); } catch { cfg[key] = inp.value.split(",").map(s => s.trim()); }
            } else if (type === "int" || type === "integer") {
                cfg[key] = parseInt(inp.value, 10);
            } else if (type === "float" || type === "number") {
                cfg[key] = parseFloat(inp.value);
            } else {
                cfg[key] = inp.value;
            }
        });
        return cfg;
    }

    function renderLlmModelDropdown(modelsList) {
        selModel.innerHTML = modelsList.map(m => {
            let label, value, dataAttrs = "";
            if (typeof m === "string") {
                label = value = m;
            } else {
                value = m.model_name || m.name || m.id || "";
                label = value;
                if (m.owned_by) dataAttrs += ` data-owned-by="${m.owned_by}"`;
                if (m.created_datetime) dataAttrs += ` data-created="${m.created_datetime}"`;
                if (m.size) dataAttrs += ` data-size="${m.size}"`;
            }
            return `<option value="${value}"${dataAttrs}>${label}</option>`;
        }).join("");
        selModel.dispatchEvent(new Event("change"));
    }

    // Attach real-time filter event for LLM Models search bar
    llmModelSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().strip ? e.target.value.toLowerCase().strip() : e.target.value.toLowerCase().trim();
        if (!query) {
            renderLlmModelDropdown(discoveredLlmModels);
            return;
        }

        const filtered = discoveredLlmModels.filter(m => {
            const name = typeof m === "string" ? m : (m.model_name || m.name || m.id || "");
            return name.toLowerCase().includes(query);
        });

        renderLlmModelDropdown(filtered);
    });

    selBinding.addEventListener("change", () => {
        modelGroup.style.display = "none";
        valStatus.className = "validation-status";
        valStatus.textContent = "";
        renderConfigForm(selBinding.value);
    });

    btnValidate.addEventListener("click", async () => {
        if (!selBinding.value) {
            valStatus.className = "validation-status error";
            valStatus.textContent = "Please select a binding provider first.";
            return;
        }
        btnValidate.disabled = true;
        btnValidate.textContent = "Validating...";
        valStatus.className = "validation-status";
        valStatus.textContent = "Connecting to binding and retrieving model list...";

        const payload = {
            binding_name: selBinding.value,
            config: collectBindingConfig()
        };

        try {
            const res = await fetch("/api/bindings/llm/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success && Array.isArray(data.models)) {
                valStatus.className = "validation-status success";
                valStatus.textContent = `✅ Connection successful. Found ${data.models.length} model(s).`;

                // Store complete models list in-memory
                discoveredLlmModels = data.models;
                llmModelSearch.value = ""; // Clear previous search input

                renderLlmModelDropdown(discoveredLlmModels);
                modelGroup.style.display = "flex";
                currentBindingConfig = payload.config;
            } else {
                valStatus.className = "validation-status error";
                valStatus.textContent = `❌ Validation failed: ${data.error || "Unknown error"}`;
                modelGroup.style.display = "none";
            }
        } catch (err) {
            valStatus.className = "validation-status error";
            valStatus.textContent = `❌ Request failed: ${err}`;
        } finally {
            btnValidate.disabled = false;
            btnValidate.textContent = "🔌 Validate & Load Models";
        }
    });

    // Bind the LLM-specific apply button (if it exists separately in the future)
    // Currently, btnApply is bound to the LLM tab button, but we need the global one.
    const btnApplyGlobal = document.getElementById("btn-apply-global-settings");
    if (btnApplyGlobal) {
        btnApplyGlobal.addEventListener("click", async () => {
            if (!selBinding.value || !selModel.value) {
                valStatus.className = "validation-status error";
                valStatus.textContent = "Please validate an LLM binding and select a model before applying.";
                return;
            }
            btnApplyGlobal.disabled = true;
            btnApplyGlobal.textContent = "Applying...";

            // Collect LLM configuration directly from the form inputs
            const payload = {
                llm_binding_name: selBinding.value,
                llm_binding_config: { ...collectBindingConfig(), model_name: selModel.value }
            };

            // Collect optional TTI configuration directly from the form inputs
            if (selTtiBinding.value && selTtiModel.value) {
                payload["tti_binding_name"] = selTtiBinding.value;
                payload["tti_binding_config"] = { ...collectTtiBindingConfig(), model_name: selTtiModel.value };
            }

            try {
                const res = await fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if (data.success) {
                    valStatus.className = "validation-status success";
                    valStatus.textContent = "✅ Configuration applied successfully.";
                    await fetchCurrentModels();
                    setTimeout(() => {
                        settingsModal.style.display = "none";
                        valStatus.className = "validation-status";
                        valStatus.textContent = "";
                    }, 800);
                } else {
                    valStatus.className = "validation-status error";
                    valStatus.textContent = `❌ Apply failed: ${data.error || "Unknown error"}`;
                }
            } catch (err) {
                valStatus.className = "validation-status error";
                valStatus.textContent = `❌ Apply request failed: ${err}`;
            } finally {
                btnApplyGlobal.disabled = false;
                btnApplyGlobal.textContent = "💾 Save Configuration";
            }
        });
    }

    async function fetchPersonalities() {
        try {
            const res = await fetch("/api/personalities");
            const data = await res.json();
            if (data.success && Array.isArray(data.personalities)) {
                // Populate Header Dropdown
                headerPersonality.innerHTML = `<option value="">-- Active Personality --</option>` +
                    data.personalities.map(p => `<option value="${p.category}/${p.name}">${p.title}</option>`).join("");

                // Render Modal Personalities list
                const pList = document.getElementById("modal-personalities-list");
                if (data.personalities.length === 0) {
                    pList.innerHTML = `<li class="empty-msg">No custom personalities installed yet.</li>`;
                    return;
                }

                pList.innerHTML = data.personalities.map(p => {
                    const avatar = p.icon_url ? `<img src="${p.icon_url}" class="persona-avatar" style="width: 32px; height: 32px; border-radius: 50%; border: 1px solid var(--border-color); object-fit: cover;" />` : `🎭`;
                    return `
                        <li class="artifact-card" style="padding: 12px; display: flex; align-items: center; gap: 12px;">
                            ${avatar}
                            <div style="flex: 1;">
                                <div class="artifact-card-header" style="margin-bottom: 2px;">
                                    <span style="font-weight: bold; font-size: 13px;">${p.title}</span>
                                    <div class="artifact-actions">
                                        <button class="artifact-action-btn select-persona-btn" data-cat="${p.category}" data-name="${p.name}" title="Activate Personality">⚡ Load</button>
                                        <button class="artifact-action-btn export-persona-btn" data-cat="${p.category}" data-name="${p.name}" title="Export .zip">📤 Export</button>
                                    </div>
                                </div>
                                <p class="desc-text" style="margin: 0; font-size: 11px; line-height: 1.3;">${p.description || "No description provided."}</p>
                                <span class="details" style="font-size: 10px;">Category: ${p.category} · By: ${p.author} · v${p.version}</span>
                            </div>
                        </li>
                    `;
                }).join("");

                // Bind select and export buttons
                pList.querySelectorAll(".select-persona-btn").forEach(btn => {
                    btn.addEventListener("click", () => activatePersonality(btn.dataset.cat, btn.dataset.name));
                });
                pList.querySelectorAll(".export-persona-btn").forEach(btn => {
                    btn.addEventListener("click", () => exportPersonality(btn.dataset.cat, btn.dataset.name));
                });
            }
        } catch (err) {
            console.error("Failed to fetch personalities:", err);
        }
    }

    // ── Activate Personality Subroutine ──
    async function activatePersonality(category, name) {
        headerPersonalitySpinner.style.display = "inline-block";
        try {
            const res = await fetch("/api/personalities/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ category, persona: name })
            });
            const data = await res.json();
            if (data.success) {
                headerPersonality.value = `${category}/${name}`;
                appendChatBubble("assistant", `🎭 Active Personality shifted to **${name.replace('_', ' ').toUpperCase()}**.\nSystem instructions, tools, and skills have been loaded.`);
                fetchArtifacts();
                fetchDiscoveredTools();
            } else {
                alert(`Failed to activate personality: ${data.detail}`);
            }
        } catch (err) {
            alert(`Activation request failed: ${err}`);
        } finally {
            headerPersonalitySpinner.style.display = "none";
        }
    }

    // ── Export Personality Subroutine ──
    function exportPersonality(category, name) {
        const url = `/api/personalities/${category}/${name}/export`;
        const a = document.createElement("a");
        a.href = url;
        a.download = `${name}_personality.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    async function fetchCurrentTtiModels() {
        try {
            const currentSelection = headerTtiModel.value;
            // Scan for configured TTI local models
            const res = await fetch("/api/bindings/tti");
            const data = await res.json();
            if (data.success && Array.isArray(data.bindings)) {
                // Find current active TTI binding models
                const settingsRes = await fetch("/api/settings");
                const settingsData = await settingsRes.json();
                const activeTtiBinding = settingsData.tti_binding_name;

                if (activeTtiBinding) {
                    const testRes = await fetch("/api/bindings/tti/test", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ binding_name: activeTtiBinding, config: settingsData.tti_binding_config || {} })
                    });
                    const testData = await testRes.json();
                    if (testData.success && Array.isArray(testData.models)) {
                        let html = `<option value="">-- Active TTI Model --</option>`;
                        html += testData.models.map(m => {
                            const value = m.model_name || m.name || m.id || m;
                            const label = m.display_name || value;
                            return `<option value="${value}">${label}</option>`;
                        }).join("");

                        headerTtiModel.innerHTML = html;
                        if (currentSelection) headerTtiModel.value = currentSelection;
                    }
                } else {
                    headerTtiModel.innerHTML = `<option value="">-- TTI Disabled --</option>`;
                }
            }
        } catch (err) {
            console.error("Failed to fetch current TTI models:", err);
        }
    }

    async function fetchCurrentModels() {
        try {
            const currentSelection = headerModel.value;
            const res = await fetch("/api/models");
            const data = await res.json();
            if (data.success && Array.isArray(data.models) && data.models.length > 0) {
                const hasCurrent = data.models.some(m => {
                    const v = typeof m === "string" ? m : (m.model_name || m.name || m.id || "");
                    return v === currentSelection;
                });

                let html = "";
                if (currentSelection && !hasCurrent) {
                    html += `<option value="${currentSelection}">${currentSelection}</option>`;
                }

                html += data.models.map(m => {
                    let label, value;
                    if (typeof m === "string") {
                        label = value = m;
                    } else {
                        value = m.model_name || m.name || m.id || "";
                        label = value;
                    }
                    return `<option value="${value}">${label}</option>`;
                }).join("");

                headerModel.innerHTML = html;
                if (currentSelection) headerModel.value = currentSelection;
            }
        } catch (err) {
            console.error("Failed to fetch current models:", err);
        }
    }

    async function loadSettingsIntoUI() {
        try {
            const res = await fetch("/api/settings");
            const data = await res.json();
            
            await fetchBindings();
            await fetchTtiBindings();
            await fetchProfiles();

            if (data.success && data.llm_binding_name) {
                // Pre-select binding and load its form fields
                selBinding.value = data.llm_binding_name;
                renderConfigForm(data.llm_binding_name);

                // Populate config inputs first
                const config = data.llm_binding_config || {};
                for (const [key, value] of Object.entries(config)) {
                    const input = document.getElementById(`cfg-${key}`);
                    if (input) {
                        if (input.type === "checkbox") {
                            input.checked = !!value;
                        } else {
                            input.value = Array.isArray(value) ? JSON.stringify(value) : value;
                        }
                    }
                }

                // Trigger validation with the loaded config values directly
                const valRes = await fetch("/api/bindings/llm/test", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ binding_name: data.llm_binding_name, config: config })
                });
                const valData = await valRes.json();
                if (valData.success && Array.isArray(valData.models)) {
                    discoveredLlmModels = valData.models;
                    renderLlmModelDropdown(discoveredLlmModels);
                    modelGroup.style.display = "flex";
                    currentBindingConfig = config;
                    
                    if (config.model_name) {
                        selModel.value = config.model_name;
                        selModel.dispatchEvent(new Event("change"));
                    }
                }

                // ── Load TTI Configuration ──
                if (data.tti_binding_name) {
                    selTtiBinding.value = data.tti_binding_name;
                    renderTtiConfigForm(data.tti_binding_name);

                    const ttiConfig = data.tti_binding_config || {};
                    for (const [key, value] of Object.entries(ttiConfig)) {
                        const input = document.getElementById(`tti-cfg-${key}`);
                        if (input) {
                            if (input.type === "checkbox") {
                                input.checked = !!value;
                            } else {
                                input.value = Array.isArray(value) ? JSON.stringify(value) : value;
                            }
                        }
                    }

                    // Trigger validation with TTI config directly
                    const ttiValRes = await fetch("/api/bindings/tti/test", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ binding_name: data.tti_binding_name, config: ttiConfig })
                    });
                    const ttiValData = await ttiValRes.json();
                    if (ttiValData.success && Array.isArray(ttiValData.models)) {
                        discoveredTtiModels = ttiValData.models;
                        renderTtiModelDropdown(discoveredTtiModels);
                        ttiModelGroup.style.display = "flex";
                        currentTtiBindingConfig = ttiConfig;

                        if (ttiConfig.model_name) {
                            selTtiModel.value = ttiConfig.model_name;
                            selTtiModel.dispatchEvent(new Event("change"));
                        }
                    }
                }
                
                if (data.personality_name) {
                    setTimeout(() => {
                        headerPersonality.value = data.personality_name;
                    }, 500);
                }
            } else {
                // ── First Run Guidance ──
                // Pre-populate with 'ollama' local guidance to assist the user on first launch
                selBinding.value = "ollama";
                selBinding.dispatchEvent(new Event("change"));
                
                const hostInput = document.getElementById("cfg-host_address");
                if (hostInput) hostInput.value = "http://localhost:11434";
                
                btnOpenSettings.click();
            }
            
            // Populate models list first before setting values
            await fetchCurrentModels();
            await fetchCurrentTtiModels();
            await fetchPersonalities();
            
            // Explicitly sync the top header dropdown value with the saved setting
            if (data.success && data.llm_binding_config && data.llm_binding_config.model_name) {
                headerModel.value = data.llm_binding_config.model_name;
            }
            if (data.success && data.tti_binding_config && data.tti_binding_config.model_name) {
                headerTtiModel.value = data.tti_binding_config.model_name;
            }
        } catch (err) {
            console.error("Failed to load settings:", err);
        }
    }
    function closeExportModal() {
        exportDetailsModal.style.display = "none";
        activeExportType = null;
    }

    btnCloseExportModal.addEventListener("click", closeExportModal);
    btnCancelExport.addEventListener("click", closeExportModal);

    // ── Settings Sub-Tabs Navigation (Parameters vs Custom Commands) ──
    const llmSubTabButtons = document.querySelectorAll(".sub-tab-btn[data-llm-sub-tab]");
    const llmSubPanes = document.querySelectorAll(".llm-sub-pane");
    llmSubTabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            llmSubTabButtons.forEach(b => b.classList.remove("active"));
            llmSubPanes.forEach(p => p.style.display = "none");
            btn.classList.add("active");
            document.getElementById(`llm-sub-pane-${btn.dataset.llmSubTab}`).style.display = "flex";

            if (btn.dataset.llmSubTab === "commands") {
                populateBindingCommands("llm", selBinding.value);
            }
        });
    });

    const ttiSubTabButtons = document.querySelectorAll(".sub-tab-btn[data-tti-sub-tab]");
    const ttiSubPanes = document.querySelectorAll(".tti-sub-pane");
    ttiSubTabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            ttiSubTabButtons.forEach(b => b.classList.remove("active"));
            ttiSubPanes.forEach(p => p.style.display = "none");
            btn.classList.add("active");
            document.getElementById(`tti-sub-pane-${btn.dataset.tti-sub-tab}`).style.display = "flex";

            if (btn.dataset.tti-sub-tab === "commands") {
                populateBindingCommands("tti", selTtiBinding.value);
            }
        });
    });

    // ── Custom Commands Form Generation & Execution Sub-System ──
    function populateBindingCommands(modality, bindingName) {
        const cmdSelect = document.getElementById(`settings-${modality}-command-select`);
        const cmdForm = document.getElementById(`settings-${modality}-command-form`);
        const execBtn = document.getElementById(`btn-execute-${modality}-command`);
        const statusDiv = document.getElementById(`${modality}-command-status`);

        cmdSelect.innerHTML = `<option value="">-- Select Command --</option>`;
        cmdForm.style.display = "none";
        execBtn.style.display = "none";
        statusDiv.style.display = "none";

        if (!bindingName) {
            return;
        }

        const bindingList = modality === "llm" ? availableBindings : availableTtiBindings;
        const binding = bindingList.find(b => b.binding_name === bindingName);
        if (!binding || !binding.commands || binding.commands.length === 0) {
            cmdSelect.innerHTML = `<option value="">No commands supported by this binding</option>`;
            return;
        }

        cmdSelect.innerHTML += binding.commands.map(c => `<option value="${c.name}">${c.title || c.name}</option>`).join("");

        // Replace change listeners cleanly
        const newCmdSelect = cmdSelect.cloneNode(true);
        cmdSelect.parentNode.replaceChild(newCmdSelect, cmdSelect);

        newCmdSelect.addEventListener("change", () => {
            const cmdName = newCmdSelect.value;
            if (!cmdName) {
                cmdForm.style.display = "none";
                execBtn.style.display = "none";
                return;
            }

            const command = binding.commands.find(c => c.name === cmdName);
            renderCommandForm(modality, command);
        });
    }

    function renderCommandForm(modality, command) {
        const cmdForm = document.getElementById(`settings-${modality}-command-form`);
        const execBtn = document.getElementById(`btn-execute-${modality}-command`);
        const statusDiv = document.getElementById(`${modality}-command-status`);

        cmdForm.innerHTML = "";
        statusDiv.style.display = "none";

        if (!command.parameters || command.parameters.length === 0) {
            cmdForm.innerHTML = `<p class="empty-msg" style="padding: 10px 0;">This command has no parameters. Ready to run.</p>`;
        } else {
            cmdForm.innerHTML = command.parameters.map(p => {
                const required = p.mandatory ? "required" : "";
                const requiredLabel = p.mandatory ? `<span style="color:#ef4444">*</span>` : "";
                return `
                    <div class="input-group">
                        <label for="cmd-param-${modality}-${p.name}">${p.name} ${requiredLabel}</label>
                        <input type="text" class="cmd-param-input" id="cmd-param-${modality}-${p.name}" data-param-name="${p.name}" placeholder="${p.description || ''}" ${required} />
                    </div>
                `;
            }).join("");
        }

        cmdForm.style.display = "flex";
        execBtn.style.display = "inline-block";

        // Bind Execution Click cleanly
        const newExecBtn = execBtn.cloneNode(true);
        execBtn.parentNode.replaceChild(newExecBtn, execBtn);

        newExecBtn.addEventListener("click", async () => {
            newExecBtn.disabled = true;
            newExecBtn.textContent = "Running Command...";
            statusDiv.className = "validation-status";
            statusDiv.style.display = "flex";
            statusDiv.textContent = "Executing custom command inside virtual environment...";

            // Collect parameters
            const params = {};
            let failedValidation = false;
            cmdForm.querySelectorAll(".cmd-param-input").forEach(inp => {
                const name = inp.getAttribute("data-param-name");
                const val = inp.value.trim();
                const spec = command.parameters.find(p => p.name === name);

                if (spec && spec.mandatory && !val) {
                    alert(`Parameter '${name}' is required.`);
                    failedValidation = true;
                    return;
                }
                params[name] = val;
            });

            if (failedValidation) {
                newExecBtn.disabled = false;
                newExecBtn.textContent = "⚡ Run Command";
                return;
            }

            const payload = {
                modality: modality,
                binding_name: modality === "llm" ? selBinding.value : selTtiBinding.value,
                command_name: command.name,
                parameters: params
            };

            try {
                const res = await fetch("/api/bindings/execute_command", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if (data.success) {
                    statusDiv.className = "validation-status success";
                    statusDiv.innerHTML = `
                        <div style="display:flex; flex-direction:column; gap:4px; text-align: left; width: 100%;">
                            <strong>✅ Command Executed Successfully</strong>
                            <span>Result: ${JSON.stringify(data.result, null, 2)}</span>
                        </div>
                    `;
                    // Refresh local models and selectors
                    if (modality === "llm") {
                        await fetchCurrentModels();
                    } else {
                        await fetchCurrentTtiModels();
                    }
                } else {
                    statusDiv.className = "validation-status error";
                    statusDiv.textContent = `❌ Execution failed: ${data.detail || "Unknown error."}`;
                }
            } catch (err) {
                statusDiv.className = "validation-status error";
                statusDiv.textContent = `❌ Request failed: ${err}`;
            } finally {
                newExecBtn.disabled = false;
                newExecBtn.textContent = "⚡ Run Command";
            }
        });
    }

    // Confirm and download selected export
    btnConfirmExport.addEventListener("click", async () => {
        btnConfirmExport.disabled = true;
        btnConfirmExport.textContent = "Exporting...";

        try {
            if (activeExportType === "artifacts") {
                const title = document.getElementById("export-art-select").value;
                const fmt = document.getElementById("export-art-format").value;
                const asBundle = document.getElementById("export-as-bundle-check").checked;

                if (asBundle) {
                    // Export portable bundle
                    const res = await fetch(`/api/bundle/${title}`);
                    const bundle = await res.json();
                    triggerDownload(JSON.stringify(bundle, null, 2), `${title}_bundle.json`, "application/json");
                } else {
                    // Standard format-coerced export
                    const res = await fetch(`/api/export/${encodeURIComponent(title)}?format=${fmt}`);
                    if (!res.ok) throw new Error("Export failed");
                    const blob = await res.blob();
                    triggerBlobDownload(blob, `${title}_export.${fmt === 'excel' ? 'xlsx' : fmt === 'markdown' ? 'md' : fmt}`);
                }

            } else if (activeExportType === "memories") {
                const scope = document.getElementById("export-mem-scope").value;
                const fmt = document.getElementById("export-mem-format").value;

                const res = await fetch("/api/export_memories", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ scope: scope, format: fmt })
                });
                if (!res.ok) throw new Error("Memory export failed");
                const blob = await res.blob();
                triggerBlobDownload(blob, `memories_${scope}.${fmt}`);

            } else if (activeExportType === "tools") {
                const checkedTools = Array.from(document.querySelectorAll(".tool-export-check:checked")).map(c => c.value);
                const fmt = document.getElementById("export-tool-format").value;

                if (checkedTools.length === 0) {
                    alert("Please select at least one tool to export.");
                    btnConfirmExport.disabled = false;
                    btnConfirmExport.textContent = "Export";
                    return;
                }

                const res = await fetch("/api/export_tools", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ tools: checkedTools, format: fmt })
                });
                if (!res.ok) throw new Error("Tools export failed");
                const blob = await res.blob();
                triggerBlobDownload(blob, `tools_export.${fmt === 'py' ? 'zip' : 'json'}`);
            }

            closeExportModal();
        } catch (err) {
            alert(`Export failed: ${err}`);
        } finally {
            btnConfirmExport.disabled = false;
            btnConfirmExport.textContent = "Export";
        }
    });

    function triggerDownload(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        triggerBlobDownload(blob, filename);
    }

    function triggerBlobDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ── Fetch Session Artifacts ──
    async function fetchArtifacts() {
        try {
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            
            if (arts.length === 0) {
                artifactsList.innerHTML = `<li class="empty-msg">No active artifacts in session.</li>`;
                return;
            }

            artifactsList.innerHTML = arts.map(a => {
                const isSelected = a.title === activeArtifactTitle ? "selected" : "";
                const isInactive = !a.active ? "inactive" : "";
                const toggleTitle = a.active ? "Deactivate (exclude from context)" : "Activate (include in context)";
                const toggleIcon = a.active ? "👁️" : "💤";

                let innerContent = "";
                if (a.type === "skill") {
                    const descHtml = a.description ? `<p class="desc-text">${a.description}</p>` : "";
                    const catHtml = a.category ? `<span class="category-tag">${a.category}</span>` : "";
                    const authorHtml = a.author ? ` · By: ${a.author}` : "";
                    innerContent = `
                        <div class="artifact-card-header">
                            <span class="title">🎓 ${a.title}</span>
                            <div class="artifact-actions">
                                <button class="artifact-action-btn toggle" data-title="${a.title}" title="${toggleTitle}">${toggleIcon}</button>
                                <button class="artifact-action-btn delete" data-title="${a.title}" title="Delete completely">🗑️</button>
                            </div>
                        </div>
                        ${descHtml}
                        <span class="details">${catHtml}${authorHtml} · v${a.version} · ${a.size} chars</span>
                    `;
                } else if (a.type === "tool") {
                    const descHtml = a.description ? `<p class="desc-text">${a.description}</p>` : "";
                    const catHtml = a.category ? `<span class="category-tag">${a.category}</span>` : `<span class="category-tag">lcp_tool</span>`;
                    const authorHtml = a.author ? ` · By: ${a.author}` : "";
                    innerContent = `
                        <div class="artifact-card-header">
                            <span class="title">🛠️ ${a.title}</span>
                            <div class="artifact-actions">
                                <button class="artifact-action-btn toggle" data-title="${a.title}" title="${toggleTitle}">${toggleIcon}</button>
                                <button class="artifact-action-btn delete" data-title="${a.title}" title="Delete completely">🗑️</button>
                            </div>
                        </div>
                        ${descHtml}
                        <span class="details">${catHtml}${authorHtml} · v${a.version} · ${a.size} chars</span>
                    `;
                } else {
                    let typeIcon = "📄";
                    if (a.type === "data") typeIcon = "📊";
                    else if (a.type === "code") typeIcon = "💻";
                    innerContent = `
                        <div class="artifact-card-header">
                            <span class="title">${typeIcon} ${a.title}</span>
                            <div class="artifact-actions">
                                <button class="artifact-action-btn toggle" data-title="${a.title}" title="${toggleTitle}">${toggleIcon}</button>
                                <button class="artifact-action-btn delete" data-title="${a.title}" title="Delete completely">🗑️</button>
                            </div>
                        </div>
                        <span class="details">Type: ${a.type} | Version: v${a.version} | Size: ${a.size} chars</span>
                    `;
                }

                return `
                    <li class="artifact-card ${a.type} ${isSelected} ${isInactive}" data-title="${a.title}">
                        ${innerContent}
                    </li>
                `;
            }).join("");

            document.querySelectorAll(".artifact-card").forEach(card => {
                card.addEventListener("click", (e) => {
                    if (e.target.closest(".artifact-action-btn")) {
                        return;
                    }
                    const title = card.dataset.title;
                    selectArtifact(title);
                });
            });

            document.querySelectorAll(".artifact-action-btn.toggle").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const title = btn.dataset.title;
                    try {
                        const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}/toggle`, { method: "POST" });
                        const data = await res.json();
                        if (data.success) {
                            fetchArtifacts();
                            updateContextBudget();
                        }
                    } catch (err) {
                        console.error("Failed to toggle artifact active state:", err);
                    }
                });
            });

            document.querySelectorAll(".artifact-action-btn.delete").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const title = btn.dataset.title;
                    if (!confirm(`Are you sure you want to permanently delete the artifact '${title}'? This will also remove all its version history.`)) {
                        return;
                    }
                    try {
                        const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}`, { method: "DELETE" });
                        const data = await res.json();
                        if (data.success) {
                            closeArtifactTab(title);
                            fetchArtifacts();
                            updateContextBudget();
                        }
                    } catch (err) {
                        console.error("Failed to delete artifact:", err);
                    }
                });
            });
        } catch (err) {
            console.error("Failed to load artifacts:", err);
        }
    }

    // ── 💬 Fetch, Render, and Manage State-rich Message History ──
    async function fetchMessageHistory() {
        try {
            const res = await fetch("/api/discussions/viewer_session/messages");
            const messages = await res.json();

            chatHistory.innerHTML = ""; // Clear
            if (messages.length === 0) {
                chatHistory.innerHTML = `
                    <div class="chat-welcome-msg">
                        Ask Lollms questions. You can chat freely with or without loaded artifacts. 
                        Click on the <strong>+</strong> button below to import local files or internet content!
                    </div>
                `;
                return;
            }

            messages.forEach(msg => {
                const sender = msg.sender_type === "user" ? "user" : "assistant";
                renderMessageBubble(msg.id, sender, msg.content, {
                    model_name: msg.model_name,
                    tokens: msg.tokens,
                    speed: msg.generation_speed,
                    ttft: msg.metadata ? msg.metadata.ttft : null
                });
            });
            chatHistory.scrollTop = chatHistory.scrollHeight;
        } catch (err) {
            console.error("Failed to fetch message history:", err);
        }
    }

    function renderMessageBubble(msgId, sender, text, metrics = {}) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;
        bubble.dataset.msgId = msgId;

        const contentDiv = document.createElement("div");
        contentDiv.className = "chat-assistant-container";

        const proseSpan = document.createElement("span");
        proseSpan.className = "prose-span";

        // Resolve inline anchors and compile markdown
        const parsedMarkdown = marked.parse(text);
        const resolvedHTML = resolveImageAnchors(parsedMarkdown, activeArtifactTitle);
        proseSpan.innerHTML = resolvedHTML;
        contentDiv.appendChild(proseSpan);
        bubble.appendChild(contentDiv);

        // ── 1. Append Metrics Bar (Assistant only) ──
        if (sender === "assistant" && metrics.model_name && metrics.model_name !== "unknown") {
            const metaBar = document.createElement("div");
            metaBar.className = "msg-meta-bar";

            const speedText = metrics.speed ? `${parseFloat(metrics.speed).toFixed(1)} tps` : "N/A";
            const ttftText = metrics.ttft ? `${parseFloat(metrics.ttft).toFixed(2)}s` : "N/A";
            const tokensText = metrics.tokens ? `${metrics.tokens}` : "N/A";

            metaBar.innerHTML = `
                <div class="meta-item" title="Model name">🤖 ${metrics.model_name}</div>
                <div class="meta-item" title="Generation Speed">⚡ ${speedText}</div>
                <div class="meta-item" title="Time to First Token">⏱️ TTFT: ${ttftText}</div>
                <div class="meta-item" title="Token count">🪙 ${tokensText} tokens</div>
            `;
            contentDiv.appendChild(metaBar);
        }

        // ── 2. Append Action Buttons Row (Both User & Assistant) ──
        const actionsRow = document.createElement("div");
        actionsRow.className = "msg-actions-row";

        const actionLabel = sender === "user" ? "🔄 Resend" : "🔄 Regenerate";
        actionsRow.innerHTML = `
            <button class="msg-action-btn edit" data-msg-id="${msgId}">✏️ Edit</button>
            <button class="msg-action-btn delete" data-msg-id="${msgId}">🗑️ Delete</button>
            <button class="msg-action-btn resend" data-msg-id="${msgId}">${actionLabel}</button>
        `;
        contentDiv.appendChild(actionsRow);
        chatHistory.appendChild(bubble);

        // Bind Action Clicks
        bindMessageActions(bubble);
        renderMath(proseSpan);
        attachDataQueryInterceptors(contentDiv);
    }

    function bindMessageActions(bubbleElement) {
        const msgId = bubbleElement.dataset.msgId;
        const proseSpan = bubbleElement.querySelector(".prose-span");
        const actionsRow = bubbleElement.querySelector(".msg-actions-row");

        // A. Edit Message
        bubbleElement.querySelector(".msg-action-btn.edit").addEventListener("click", () => {
            if (bubbleElement.querySelector(".inline-edit-area")) return;

            // Retrieve original raw text of message
            fetch("/api/discussions/viewer_session/messages")
                .then(res => res.json())
                .then(messages => {
                    const matched = messages.find(m => m.id === msgId);
                    const originalRaw = matched ? matched.content : proseSpan.textContent;

                    proseSpan.style.display = "none";
                    actionsRow.style.display = "none";

                    const editArea = document.createElement("textarea");
                    editArea.className = "inline-edit-area";
                    editArea.rows = 8; // Sets a comfortable default height
                    editArea.value = originalRaw;

                    const controls = document.createElement("div");
                    controls.className = "inline-edit-controls";
                    controls.innerHTML = `
                        <button class="btn btn-secondary cancel-edit" style="width:auto; padding:6px 12px; font-size:11.5px;">Cancel</button>
                        <button class="btn btn-primary save-edit" style="width:auto; padding:6px 12px; font-size:11.5px;">Save</button>
                    `;

                    proseSpan.after(controls);
                    proseSpan.after(editArea);

                    controls.querySelector(".cancel-edit").addEventListener("click", () => {
                        editArea.remove();
                        controls.remove();
                        proseSpan.style.display = "block";
                        actionsRow.style.display = "flex";
                    });

                    controls.querySelector(".save-edit").addEventListener("click", async () => {
                        const updatedVal = editArea.value.trim();
                        if (!updatedVal) return;

                        try {
                            const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}/edit`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ content: updatedVal })
                            });
                            const data = await res.json();
                            if (data.success) {
                                editArea.remove();
                                controls.remove();

                                const resolved = resolveImageAnchors(data.content, activeArtifactTitle);
                                proseSpan.innerHTML = marked.parse(resolved);
                                renderMath(proseSpan);

                                proseSpan.style.display = "block";
                                actionsRow.style.display = "flex";
                                fetchMessageHistory();
                            }
                        } catch (err) { alert("Failed to save: " + err); }
                    });
                });
        });

        // B. Delete Message
        bubbleElement.querySelector(".msg-action-btn.delete").addEventListener("click", async () => {
            if (!confirm("Are you sure you want to delete this message?")) return;
            try {
                const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}`, { method: "DELETE" });
                const data = await res.json();
                if (data.success) {
                    bubbleElement.remove();
                    fetchMessageHistory();
                }
            } catch (err) { alert("Delete failed: " + err); }
        });

        // C. Resend / Regenerate
        bubbleElement.querySelector(".msg-action-btn.resend").addEventListener("click", async () => {
            try {
                const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}/regenerate`, { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    // Trigger the duplicate-free assistant regeneration stream directly
                    sendChatMessage(true);
                }
            } catch (err) { alert("Regeneration failed: " + err); }
        });
    }

    // ── 🗃️ Workspaces CRUD & Selection Screen Sub-System ──
    async function fetchWorkspaces() {
        try {
            const res = await fetch("/api/workspaces");
            const data = await res.json();
            if (data.success && Array.isArray(data.workspaces)) {
                renderWorkspaceGrid(data.workspaces);
            }
        } catch (err) {
            console.error("Failed to fetch workspaces:", err);
        }
    }

    function renderWorkspaceGrid(workspaces) {
        workspaceCardGrid.innerHTML = workspaces.map(ws => {
            const activeClass = ws.active ? "active-ws" : "";
            const activeTag = ws.active ? `<span class="level-tag working" style="display:inline-block; margin-bottom:4px;">★ ACTIVE</span>` : "";
            const deleteBtn = ws.name !== "default" ? `<button class="ws-btn delete-ws" data-name="${ws.name}">🗑️ Delete</button>` : "";
            return `
                <div class="workspace-card ${activeClass}" data-name="${ws.name}">
                    ${activeTag}
                    <div class="ws-name">🗃️ ${ws.name.replace(/_/g, ' ').toUpperCase()}</div>
                    <div class="ws-meta">Created: ${ws.created_at}</div>
                    <div class="ws-actions">
                        <button class="ws-btn select-ws" data-name="${ws.name}">⚡ Open</button>
                        ${deleteBtn}
                    </div>
                </div>
            `;
        }).join("") + `
            <div class="workspace-card create-new-card" id="btn-create-workspace-card">
                <span>➕ Create New Workspace</span>
            </div>
        `;

        // Bind clicks
        workspaceCardGrid.querySelectorAll(".workspace-card").forEach(card => {
            card.addEventListener("click", (e) => {
                if (e.target.closest(".ws-btn") || card.id === "btn-create-workspace-card") return;
                selectWorkspace(card.dataset.name);
            });
        });

        workspaceCardGrid.querySelectorAll(".ws-btn.select-ws").forEach(btn => {
            btn.addEventListener("click", () => selectWorkspace(btn.dataset.name));
        });

        workspaceCardGrid.querySelectorAll(".ws-btn.delete-ws").forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteWorkspace(btn.dataset.name);
            });
        });

        const createCard = document.getElementById("btn-create-workspace-card");
        if (createCard) {
            createCard.addEventListener("click", () => {
                const name = prompt("Enter a name for the new isolated workspace:");
                if (name) createWorkspace(name);
            });
        }
    }

    async function createWorkspace(name) {
        try {
            const res = await fetch("/api/workspaces/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (data.success) {
                fetchWorkspaces();
            } else {
                alert(`Creation failed: ${data.detail}`);
            }
        } catch (err) {
            alert(`Request failed: ${err}`);
        }
    }

    async function selectWorkspace(name) {
        workspaceSelectionScreen.style.display = "flex";
        try {
            const res = await fetch("/api/workspaces/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (data.success) {
                sessionStorage.setItem("workspace_selected", "true");
                workspaceSelectionScreen.style.display = "none";
                location.reload(); // Refresh fully to load new workspaces state
            }
        } catch (err) {
            alert(`Failed to load workspace: ${err}`);
        }
    }

    async function deleteWorkspace(name) {
        if (!confirm(`Permanently delete workspace '${name.toUpperCase()}' and ALL of its associated discussions, memories, and artifacts?`)) {
            return;
        }
        try {
            const res = await fetch(`/api/workspaces/${encodeURIComponent(name)}`, { method: "DELETE" });
            if (res.ok) fetchWorkspaces();
        } catch (err) {
            console.error("Failed to delete workspace:", err);
        }
    }

    // ── Workspaces Session Startup Guard ──
    const isWorkspaceSelected = sessionStorage.getItem("workspace_selected");
    if (!isWorkspaceSelected) {
        fetchWorkspaces().then(() => {
            workspaceSelectionScreen.style.display = "flex";
        });
    } else {
        workspaceSelectionScreen.style.display = "none";
    }

    // Bind back-to-workspaces header button
    btnBackToWorkspaces.addEventListener("click", () => {
        sessionStorage.removeItem("workspace_selected");
        fetchWorkspaces().then(() => {
            workspaceSelectionScreen.style.display = "flex";
        });
    });

    // ── 🧠 Fetch and Render Persistent Memories ──
    async function fetchMemories() {
        try {
            const res = await fetch("/api/memories");
            const memories = await res.json();

            if (memories.length === 0) {
                memoriesList.innerHTML = `<li class="empty-msg">No memories stored yet.</li>`;
                return;
            }

            const working = memories.filter(m => m.level === 1);
            const deep = memories.filter(m => m.level === 2);
            const archived = memories.filter(m => m.level === 3);

            function renderZone(items, label, cls) {
                if (items.length === 0) {
                    return `<li class="empty-msg" style="padding:4px 12px;font-size:11px;">No ${label} memories.</li>`;
                }
                return items.map(m => `
                    <li class="memory-card ${cls}" data-mem-id="${m.id}">
                        <div class="memory-header">
                            <span class="level-tag ${cls}">${label} (Level ${m.level})</span>
                            <span class="importance-badge">Imp: ${(m.importance * 100).toFixed(0)}%</span>
                        </div>
                        <p class="desc-text" style="color: var(--text-primary); margin-top: 4px;">${m.content}</p>
                        <div class="details" style="font-size: 10px;">Uses: ${m.use_count} · ID: ${m.id.substring(0, 8)}</div>
                        <div class="memory-actions">
                            <button class="mem-btn" data-action="up" title="Promote (decrease level)">⬆️</button>
                            <button class="mem-btn" data-action="down" title="Demote (increase level)">⬇️</button>
                            <button class="mem-btn" data-action="imp" title="Edit importance">⚡</button>
                            <button class="mem-btn" data-action="del" title="Delete memory">🗑️</button>
                        </div>
                    </li>
                `).join("");
            }

            memoriesList.innerHTML = `
                <li class="memory-zone-header working-zone">⚡ Working Memory</li>
                ${renderZone(working, "Working", "working")}
                <li class="memory-zone-header deep-zone">🔒 Deep Memory</li>
                ${renderZone(deep, "Deep", "deep")}
                <li class="memory-zone-header archived-zone">📦 Archived Memory</li>
                ${renderZone(archived, "Archived", "archived")}
            `;

            memoriesList.querySelectorAll(".memory-actions .mem-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const card = btn.closest(".memory-card");
                    const id = card.dataset.memId;
                    const action = btn.dataset.action;
                    if (action === "del") {
                        if (!confirm("Delete this memory permanently?")) return;
                        try {
                            const res = await fetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
                            if (res.ok) fetchMemories();
                        } catch (err) { console.error(err); }
                    } else if (action === "up" || action === "down") {
                        const currentLevel = parseInt(card.querySelector(".level-tag").textContent.match(/\d+/)[0], 10);
                        const newLevel = action === "up" ? Math.max(1, currentLevel - 1) : Math.min(3, currentLevel + 1);
                        try {
                            const res = await fetch(`/api/memories/${encodeURIComponent(id)}/edit`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ level: newLevel })
                            });
                            if (res.ok) fetchMemories();
                        } catch (err) { console.error(err); }
                    } else if (action === "imp") {
                        const raw = prompt("New importance (0.0 - 1.0):");
                        if (raw === null) return;
                        const importance = parseFloat(raw);
                        if (isNaN(importance)) return;
                        try {
                            const res = await fetch(`/api/memories/${encodeURIComponent(id)}/edit`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ importance })
                            });
                            if (res.ok) fetchMemories();
                        } catch (err) { console.error(err); }
                    }
                });
            });
        } catch (err) {
            console.error("Failed to load memories:", err);
        }
    }

    // ── 💬 Interactive Conversational Chat Companion ──
    btnChatSend.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    btnChatClear.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear the conversation history? This cannot be undone.")) {
            return;
        }
        try {
            const res = await fetch("/api/chat/clear", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                chatHistory.innerHTML = `
                    <div class="chat-welcome-msg">
                        Conversation cleared successfully. Ask Lollms any questions!
                    </div>
                `;
                updateContextBudget();
            }
        } catch (err) {
            alert("Failed to clear chat: " + err);
        }
    });

    // Bind stop button execution
    btnChatStop.addEventListener("click", async () => {
        btnChatStop.disabled = true;
        btnChatStop.textContent = "Stopping...";
        try {
            await fetch("/api/chat/cancel", { method: "POST" });
        } catch (err) {
            console.error("Failed to cancel generation:", err);
        }
    });

    async function sendChatMessage(regenerate = false) {
        let text = "";
        if (!regenerate) {
            text = chatInput.value.trim();
            if (!text) return;
            chatInput.value = "";
        }

        chatInput.disabled = true;

        // Hide Send and show Stop button
        btnChatSend.style.display = "none";
        btnChatStop.style.display = "inline-block";
        btnChatStop.disabled = false;
        btnChatStop.textContent = "🛑 Stop";

        const startTime = Date.now();
        let ttft = null;

        // Clear and reload chronological history
        await fetchMessageHistory();

        let currentProse = "";
        let chatActiveProc = null;
        let chatActiveProcDiv = null;
        const chatAffectedArtifacts = new Set();

        let proseSpan = null;
        let bubbleContentDiv = null;
        let activeMsgId = null;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, regenerate: regenerate })
            });

            if (!res.ok) {
                throw new Error(`Chat request returned HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.trim().startsWith("data: ")) {
                        const rawJson = line.trim().slice(6);
                        const event = JSON.parse(rawJson);

                        if (event.error) {
                            if (proseSpan) proseSpan.innerHTML = `<span style="color: #ef4444;">Error: ${event.error}</span>`;
                            break;
                        }

                        const meta = event.meta || {};
                        const evType = meta.type;

                        if (event.msg_type === "MSG_TYPE_NEW_MESSAGE") {
                            // Backend initialized a new message node. 
                            activeMsgId = event.chunk; // returns message_id

                            // Surgically append the new assistant bubble directly to the DOM
                            // This completely prevents duplicate bubble rendering and race conditions!
                            renderMessageBubble(activeMsgId, "assistant", "Lollms is thinking...", {
                                model_name: headerModel.value || "unknown",
                                tokens: 0,
                                speed: 0,
                                ttft: null
                            });

                            // Bind stream target variables to the newly appended bubble
                            const assistantBubble = chatHistory.querySelector(`.chat-bubble.assistant[data-msg-id="${activeMsgId}"]`);
                            if (assistantBubble) {
                                bubbleContentDiv = assistantBubble.querySelector(".chat-assistant-container");
                                proseSpan = assistantBubble.querySelector(".prose-span");
                            }
                            continue;
                        }

                        if (event.msg_type === "MSG_TYPE_ARTEFACTS_STATE_CHANGED") {
                            if (meta.artefact && meta.artefact.title) {
                                chatAffectedArtifacts.add(meta.artefact.title);
                            }
                        }

                        if (evType === "processing_open") {
                            chatActiveProc = { title: meta.title || "Task", statuses: [] };
                            chatActiveProcDiv = document.createElement("div");
                            chatActiveProcDiv.className = "inline-proc-box";
                            chatActiveProcDiv.innerHTML = `
                                <div class="proc-header">
                                    <span>⚙️ ${chatActiveProc.title}</span>
                                    <span class="spinner inline"></span>
                                </div>
                                <div class="proc-statuses"></div>
                            `;
                            if (bubbleContentDiv) {
                                // Append before the actions row
                                const actionsRow = bubbleContentDiv.querySelector(".msg-actions-row");
                                if (actionsRow) {
                                    bubbleContentDiv.insertBefore(chatActiveProcDiv, actionsRow);
                                } else {
                                    bubbleContentDiv.appendChild(chatActiveProcDiv);
                                }
                            } else {
                                // No assistant bubble yet (e.g. during pre-flight or context scans)
                                // Append directly to chatHistory as a temporary root-level block
                                chatHistory.appendChild(chatActiveProcDiv);
                                const welcome = chatHistory.querySelector(".chat-welcome-msg");
                                if (welcome) welcome.remove();
                            }
                            chatHistory.scrollTop = chatHistory.scrollHeight;

                        } else if (evType === "processing_status") {
                            if (chatActiveProc && chatActiveProcDiv) {
                                const statusText = meta.status || "";
                                chatActiveProc.statuses.push(statusText);
                                const statusesDiv = chatActiveProcDiv.querySelector(".proc-statuses");
                                const item = document.createElement("div");
                                item.className = "proc-item";
                                item.textContent = `⤷ ⏳ ${statusText}`;
                                statusesDiv.appendChild(item);
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }

                        } else if (evType === "processing_close") {
                            if (chatActiveProc && chatActiveProcDiv) {
                                chatActiveProcDiv.innerHTML = `
                                    <div class="proc-header complete">
                                        <span>✅ ${chatActiveProc.title} (Complete)</span>
                                    </div>
                                    <div class="proc-statuses">
                                        ${chatActiveProc.statuses.map(s => `<div class="proc-item complete">✓ ${s}</div>`).join("")}
                                    </div>
                                `;
                                
                                // If this was a pre-flight/guard scan appended directly to chat history,
                                // we can remove it after completion to keep the conversation clean.
                                if (!bubbleContentDiv) {
                                    const tempDiv = chatActiveProcDiv;
                                    setTimeout(() => {
                                        if (tempDiv && tempDiv.parentNode) {
                                            tempDiv.remove();
                                        }
                                    }, 2000);
                                }
                                
                                chatActiveProc = null;
                                chatActiveProcDiv = null;
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }

                        } else if (evType === "processing_close") {
                            if (chatActiveProc && chatActiveProcDiv) {
                                chatActiveProcDiv.innerHTML = `
                                    <div class="proc-header complete">
                                        <span>✅ ${chatActiveProc.title} (Complete)</span>
                                    </div>
                                    <div class="proc-statuses">
                                        ${chatActiveProc.statuses.map(s => `<div class="proc-item complete">✓ ${s}</div>`).join("")}
                                    </div>
                                `;
                                chatActiveProc = null;
                                chatActiveProcDiv = null;
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }

                        } else if (event.msg_type === "MSG_TYPE_CHUNK" && proseSpan) {
                            if (event.chunk) {
                                if (event.meta && event.meta.was_processed) {
                                    // Intentionally drop status indicators
                                } else if (!event.chunk.startsWith("<processing") && !event.chunk.startsWith("</processing>")) {
                                    // ── ⏱️ Measure Time to First Token ──
                                    if (ttft === null && currentProse === "") {
                                        ttft = (Date.now() - startTime) / 1000;
                                    }

                                    if (currentProse === "") {
                                        proseSpan.innerHTML = "";
                                    }

                                    currentProse += event.chunk;
                                    const parsedMarkdown = marked.parse(currentProse);
                                    const resolvedHTML = resolveImageAnchors(parsedMarkdown, activeArtifactTitle);
                                    proseSpan.innerHTML = resolvedHTML;
                                    renderMath(proseSpan);
                                    chatHistory.scrollTop = chatHistory.scrollHeight;
                                }
                            }
                        }
                    }
                }
            }
        } catch (err) {
            if (proseSpan) proseSpan.innerHTML = `<span style="color: #ef4444;">Connection failed: ${err}</span>`;
        } finally {
            chatInput.disabled = false;
            
            // Restore Send button and hide Stop button
            btnChatStop.style.display = "none";
            btnChatSend.style.display = "inline-block";
            btnChatSend.disabled = false;
            chatInput.focus();
            
            // Re-render and append full metrics & actions
            await fetchMessageHistory();
            await fetchArtifacts();
            
            if (ttft !== null && activeMsgId) {
                // Update TTFT into the rendered message metadata bar
                const targetMetaBar = chatHistory.querySelector(`.chat-bubble.assistant[data-msg-id="${activeMsgId}"] .msg-meta-bar`);
                if (targetMetaBar) {
                    const ttftDiv = document.createElement("div");
                    ttftDiv.className = "meta-item";
                    ttftDiv.title = "Time to First Token";
                    ttftDiv.innerHTML = `⏱️ TTFT: ${ttft.toFixed(2)}s`;
                    targetMetaBar.appendChild(ttftDiv);
                    
                    // We also save TTFT to the backend message metadata so it persists across reloads!
                    fetch(`/api/discussions/viewer_session/messages/${activeMsgId}/edit`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ content: currentProse }) // sync prose
                    }).then(() => {
                        // Update metadata with ttft
                        fetch(`/api/discussions/viewer_session/messages`)
                            .then(res => res.json())
                            .then(messages => {
                                const m = messages.find(msg => msg.id === activeMsgId);
                                if (m) {
                                    const metadata = m.metadata || {};
                                    metadata.ttft = ttft;
                                    fetch(`/api/memories/${activeMsgId}/edit`, { // Helper to update metadata
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ metadata: metadata })
                                    }).catch(() => {});
                                }
                            });
                    });
                }
            }

            if (activeArtifactTitle) {
                selectArtifact(activeArtifactTitle);
            } else if (chatAffectedArtifacts.size > 0) {
                const target = Array.from(chatAffectedArtifacts).pop();
                selectArtifact(target);
                activeArtifactTitle = target;
            }
            updateContextBudget();
            fetchMemories();
        }
    }

    function appendChatBubble(sender, text) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;
        bubble.textContent = text;
        chatHistory.appendChild(bubble);
        renderMath(bubble); // Parse any math delimiters inside user chat bubble too
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        const welcome = chatHistory.querySelector(".chat-welcome-msg");
        if (welcome) welcome.remove();
    }

    function appendAssistantBubble() {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";
        
        const bubbleContentDiv = document.createElement("div");
        bubbleContentDiv.className = "chat-assistant-container";
        
        const proseSpan = document.createElement("span");
        proseSpan.className = "prose-span";
        proseSpan.innerHTML = "<em>Lollms is thinking...</em>";
        
        bubbleContentDiv.appendChild(proseSpan);
        bubble.appendChild(bubbleContentDiv);
        chatHistory.appendChild(bubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        const welcome = chatHistory.querySelector(".chat-welcome-msg");
        if (welcome) welcome.remove();

        return { bubbleContentDiv, proseSpan };
    }

    // ── 🚀 Python Data Analysis Sandbox Interceptor ──
    function attachDataQueryInterceptors(bubbleContentElement) {
        const codeBlocks = bubbleContentElement.querySelectorAll("pre code.language-python");
        
        codeBlocks.forEach(block => {
            if (!activeArtifactTitle) return;

            if (block.parentNode.nextSibling && block.parentNode.nextSibling.className === "inline-query-btn") {
                return;
            }

            const btnQuery = document.createElement("button");
            btnQuery.className = "inline-query-btn";
            btnQuery.innerHTML = "🚀 Run Python Data Analysis";
            
            block.parentNode.after(btnQuery);

            btnQuery.addEventListener("click", async () => {
                btnQuery.disabled = true;
                btnQuery.textContent = "Executing Sandbox...";

                const codeText = block.textContent;

                try {
                    const res = await fetch("/api/query_data", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ title: activeArtifactTitle, code: codeText })
                    });
                    const data = await res.json();

                    const prevBox = btnQuery.nextSibling;
                    if (prevBox && prevBox.className === "query-result-box") {
                        prevBox.remove();
                    }

                    const resultBox = document.createElement("div");
                    resultBox.className = "query-result-box";

                    if (data.success) {
                        const stdoutHtml = data.output ? `<pre class="query-stdout">${data.output}</pre>` : "<em>Code executed successfully (no stdout prints).</em>";
                        const plotHtml = data.plot_b64 ? `<img src="data:image/png;base64,${data.plot_b64}" class="query-plot-img" />` : "";
                        
                        resultBox.innerHTML = `
                            <div class="query-title">💻 Sandbox Output</div>
                            ${stdoutHtml}
                            ${plotHtml}
                        `;
                        
                        selectArtifact(activeArtifactTitle);
                    } else {
                        resultBox.innerHTML = `
                            <div class="query-title" style="color: #ef4444;">❌ Sandbox Execution Error</div>
                            <pre class="query-stdout" style="color: #fca5a5;">${data.error}</pre>
                            ${data.output ? `<div class="query-title" style="margin-top: 6px;">Stdout prior to error:</div><pre class="query-stdout">${data.output}</pre>` : ""}
                        `;
                    }

                    btnQuery.after(resultBox);
                    chatHistory.scrollTop = chatHistory.scrollHeight;

                } catch (err) {
                    alert(`Sandbox connection failed: ${err}`);
                } finally {
                    btnQuery.disabled = false;
                    btnQuery.textContent = "🚀 Run Python Data Analysis";
                }
            });
        });
    }

});

/**
 * Parses raw text and resolves any <artefact_image> anchors into HTML <img> elements.
 */
function resolveImageAnchors(content, title) {
    if (!content) return "";
    const pattern = /<artefact_image\s+id=["']([^"']+)["']\s*(?:\/>|>)/gi;
    return content.replace(pattern, (match, imageId) => {
        if (imageId.includes("::")) {
            const parts = imageId.split("::");
            const imgIndex = parts[parts.length - 1];
            const imgTitle = parts.slice(0, -1).join("::");
            return `<div class="rendered-image-container"><img src="/api/images/${encodeURIComponent(imgTitle)}/${imgIndex}" class="rendered-page-img" /><button class="img-delete-btn" onclick="deleteArtifactImage('${encodeURIComponent(imgTitle)}', ${imgIndex})">✕ Remove Image</button></div>`;
        }
        return match;
    });
}

/**
 * Expose delete image function to window/global scope for onclick handler.
 */
window.deleteArtifactImage = async function(title, index) {
    const decodedTitle = decodeURIComponent(title);
    const baseTitle = decodedTitle.endsWith("::images") ? decodedTitle.replace("::images", "") : decodedTitle;

    if (!confirm(`Are you sure you want to remove this image from the artifact?`)) {
        return;
    }

    try {
        const res = await fetch(`/api/artifacts/${encodeURIComponent(baseTitle)}/images/${index}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (data.success) {
            alert("Image removed successfully!");
            location.reload();
        }
    } catch (err) {
        alert(`Failed to delete image: ${err}`);
    }
};
