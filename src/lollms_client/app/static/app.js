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

// ── Global Function States for Chat Settings ──
let funcStates = {};

// Expose funcStates to the window scope for global visibility
window.funcStates = funcStates;

document.addEventListener("DOMContentLoaded", () => {
    // ── 🔔 Modern Toast Notification System ──
    function showNotification(message, type = "info", duration = 4500) {
        // Dynamically inject CSS keyframes if missing
        if (!document.getElementById("toast-animation-styles")) {
            const style = document.createElement("style");
            style.id = "toast-animation-styles";
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateY(-30px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        let container = document.getElementById("notification-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "notification-container";
            container.style.position = "fixed";
            container.style.top = "24px";
            container.style.right = "24px";
            container.style.zIndex = "200000";
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.gap = "12px";
            container.style.maxWidth = "380px";
            container.style.width = "calc(100% - 48px)";
            container.style.pointerEvents = "none";
            document.body.appendChild(container);
        }

        const toast = document.createElement("div");
        toast.className = `toast-notification ${type}`;
        toast.style.padding = "14px 18px";
        toast.style.borderRadius = "8px";
        toast.style.color = "#ffffff";
        toast.style.fontSize = "13px";
        toast.style.fontWeight = "600";
        toast.style.boxShadow = "0 10px 25px rgba(0, 0, 0, 0.4)";
        toast.style.display = "flex";
        toast.style.alignItems = "center";
        toast.style.gap = "10px";
        toast.style.pointerEvents = "auto";
        toast.style.animation = "slideIn 0.25s ease-out";
        toast.style.transition = "opacity 0.25s ease, transform 0.25s ease";

        let bg = "var(--chat-user-bg, #4f46e5)";
        let icon = "ℹ️";
        if (type === "success") {
            bg = "var(--success-color, #10b981)";
            icon = "✅";
        } else if (type === "error") {
            bg = "#ef4444";
            icon = "❌";
        } else if (type === "warning") {
            bg = "var(--accent-color, #f59e0b)";
            icon = "⚠️";
        }
        toast.style.backgroundColor = bg;
        toast.style.borderLeft = "4px solid rgba(0, 0, 0, 0.25)";

        toast.innerHTML = `
            <span style="font-size: 16px;">${icon}</span>
            <span style="flex: 1; line-height: 1.4;">${message}</span>
            <span class="toast-close" style="cursor: pointer; font-size: 18px; font-weight: bold; opacity: 0.6; transition: opacity 0.15s; user-select: none; margin-left: 8px;">×</span>
        `;

        container.appendChild(toast);

        const closeBtn = toast.querySelector(".toast-close");
        closeBtn.addEventListener("mouseover", () => closeBtn.style.opacity = "1");
        closeBtn.addEventListener("mouseout", () => closeBtn.style.opacity = "0.6");
        closeBtn.onclick = () => dismiss();

        let dismissTimer = setTimeout(dismiss, duration);

        function dismiss() {
            clearTimeout(dismissTimer);
            toast.style.opacity = "0";
            toast.style.transform = "translateY(-12px)";
            setTimeout(() => {
                toast.remove();
            }, 250);
        }
    }

    // Overwrite the native alert dialog with our beautiful custom notifications
    window.alert = function(message) {
        let type = "info";
        const msgLower = String(message).toLowerCase();
        if (msgLower.includes("success") || msgLower.includes("complete") || msgLower.includes("ready") || msgLower.includes("imported") || msgLower.includes("restored") || msgLower.includes("saved") || msgLower.includes("active") || msgLower.includes("passed")) {
            type = "success";
        } else if (msgLower.includes("fail") || msgLower.includes("error") || msgLower.includes("exception") || msgLower.includes("rejected") || msgLower.includes("invalid") || msgLower.includes("missing")) {
            type = "error";
        } else if (msgLower.includes("please") || msgLower.includes("select") || msgLower.includes("enter") || msgLower.includes("specify") || msgLower.includes("choose") || msgLower.includes("warn")) {
            type = "warning";
        }
        showNotification(message, type);
    };

    // ── Global State variables ──
    let selectedFiles = [];
    let activeArtifactTitle = null;
    let activeArtifactType = null;
    let selectedVersion = null;
    let activeSearchResults = [];
    let openTabs = new Set(); // Stores titles of currently open tabs
    let codeEditors = {}; // Keeps track of active CodeMirror instances key-mapped by artifact title
    let activeChatAbortController = null;
    let toolRefDocs = {}; // Reference doc contents mapped to tool titles: { title: [{name, content}] }
    let availableBindings = [];
    let currentBindingConfig = {};
    let availableTtiBindings = [];
    let currentTtiBindingConfig = {};
    let availableSttBindings = [];
    let currentSttBindingConfig = {};
    let discoveredLlmModels = [];
    let discoveredTtiModels = [];
    let discoveredSttModels = [];
    let savedLlmConfigs = {};
    let savedTtiConfigs = {};
    let savedSttConfigs = {};
    let isConfigLoading = false;
    let memViewMode = "abox";
    let graphViewMode = "abox";

    // ── 🌿 Custom Asynchronous Modal Prompt (Drop-in replacement for window.prompt) ──
    function showCustomPrompt(title, placeholder = "", defaultValue = "") {
        return new Promise((resolve) => {
            const overlay = document.createElement("div");
            overlay.className = "modal-overlay";
            overlay.style.zIndex = "1000";

            const container = document.createElement("div");
            container.className = "modal-container";
            container.style.width = "400px";

            container.innerHTML = `
                <div class="modal-header">
                    <h2>⚙️ ${title}</h2>
                    <button class="modal-close-btn" id="custom-prompt-close">×</button>
                </div>
                <div class="modal-body" style="padding: 20px;">
                    <div class="input-group" style="margin-bottom: 0;">
                        <input type="text" id="custom-prompt-input" value="${defaultValue}" placeholder="${placeholder}" autofocus style="width: 100%; padding: 10px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 6px; outline: none;" />
                    </div>
                </div>
                <div class="modal-footer" style="padding: 12px 20px; border-top: 1px solid var(--border-color); display: flex; gap: 10px; justify-content: flex-end; background-color: var(--bg-panel); border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;">
                    <button class="btn btn-secondary" id="custom-prompt-cancel" style="width: auto; padding: 6px 14px; font-size: 12.5px;">Cancel</button>
                    <button class="btn btn-primary" id="custom-prompt-ok" style="width: auto; padding: 6px 14px; font-size: 12.5px;">OK</button>
                </div>
            `;

            overlay.appendChild(container);
            document.body.appendChild(overlay);

            const input = container.querySelector("#custom-prompt-input");
            input.focus();
            if (input.value) {
                input.setSelectionRange(input.value.length, input.value.length);
            }

            function cleanup() {
                overlay.remove();
            }

            container.querySelector("#custom-prompt-close").addEventListener("click", () => {
                cleanup();
                resolve(null);
            });

            container.querySelector("#custom-prompt-cancel").addEventListener("click", () => {
                cleanup();
                resolve(null);
            });

            const okBtn = container.querySelector("#custom-prompt-ok");
            okBtn.addEventListener("click", () => {
                const val = input.value.trim();
                cleanup();
                resolve(val);
            });

            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    okBtn.click();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    container.querySelector("#custom-prompt-cancel").click();
                }
            });
        });
    }

    // ── 🗑️ Reusable Slick Confirmation Modal helper ──
    function showSlickConfirm(title, message, confirmText = "Delete", isDanger = true) {
        return new Promise((resolve) => {
            const modal = document.getElementById("confirm-modal");
            const titleEl = document.getElementById("confirm-modal-title");
            const bodyEl = document.getElementById("confirm-modal-body-text");
            const cancelBtn = document.getElementById("btn-cancel-confirm");
            const okBtn = document.getElementById("btn-ok-confirm");
            const closeBtn = document.getElementById("btn-close-confirm-modal");

            titleEl.textContent = title;
            bodyEl.textContent = message;
            okBtn.textContent = confirmText;

            if (isDanger) {
                okBtn.style.backgroundColor = "#ef4444";
                okBtn.style.borderColor = "#ef4444";
                okBtn.style.color = "#ffffff";
            } else {
                okBtn.style.backgroundColor = "var(--accent-color)";
                okBtn.style.borderColor = "var(--accent-color)";
                okBtn.style.color = "#020617";
            }

            modal.style.display = "flex";

            function cleanup() {
                modal.style.display = "none";
                // Remove event listeners by cloning
                const newOk = okBtn.cloneNode(true);
                okBtn.parentNode.replaceChild(newOk, okBtn);
                const newCancel = cancelBtn.cloneNode(true);
                cancelBtn.parentNode.replaceChild(newCancel, cancelBtn);
                const newClose = closeBtn.cloneNode(true);
                closeBtn.parentNode.replaceChild(newClose, closeBtn);
            }

            document.getElementById("btn-ok-confirm").onclick = () => {
                cleanup();
                resolve(true);
            };

            document.getElementById("btn-cancel-confirm").onclick = () => {
                cleanup();
                resolve(false);
            };

            document.getElementById("btn-close-confirm-modal").onclick = () => {
                cleanup();
                resolve(false);
            };
        });
    }

    // ── Core DOM Elements ──
    const serverStatus = document.getElementById("server-status");
    const artifactsList = document.getElementById("artifacts-list");
    const memoriesList = document.getElementById("memories-list");
    const branchSelect = document.getElementById("branch-select");
    
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
    const modalTabs = document.querySelectorAll("#import-modal .modal-tab");
    const modalTabContents = document.querySelectorAll("#import-modal .modal-tab-content-pane");

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

    // STT Configuration Selectors
    const selSttBinding = document.getElementById("settings-stt-binding");
    const sttConfigForm = document.getElementById("settings-stt-config-form");
    const btnValidateStt = document.getElementById("btn-validate-stt-binding");
    const selSttModel = document.getElementById("settings-stt-model");
    const sttModelSearch = document.getElementById("settings-stt-model-search");
    const sttModelGroup = document.getElementById("settings-stt-model-group");
    const valSttStatus = document.getElementById("settings-stt-validation-status");

    // LLM Model Search Selector
    const llmModelSearch = document.getElementById("settings-llm-model-search");

    // Profile Management Selectors
    const profileSaveName = document.getElementById("profile-save-name");
    const btnSaveProfile = document.getElementById("btn-save-profile");
    const settingsProfilesList = document.getElementById("settings-profiles-list");

    // ── Startup Initialization ──
    const btnAddMemory = document.getElementById("btn-add-memory-part");
    if (btnAddMemory) {
        btnAddMemory.addEventListener("click", async () => {
            const content = await showCustomPrompt("Add Memory", "Enter memory content");
            if (!content) return;
            const impStr = await showCustomPrompt("Memory Importance", "Importance (0.0 - 1.0)", "0.75");
            const importance = parseFloat(impStr || "0.75");
            const lvlStr = await showCustomPrompt("Memory Level", "Level (1=Working, 2=Deep, 3=Archived)", "1");
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

    const btnClearMemories = document.getElementById("btn-clear-memories-part");
    if (btnClearMemories) {
        btnClearMemories.addEventListener("click", async () => {
            const labels = { 1: "Working", 2: "Deep / Associative", 3: "Archived", 4: "Episodic" };
            const categoryName = labels[activeMemoryLevel] || "selected";

            const confirmed = await showSlickConfirm(
                "🧹 Clear Category Memories",
                `Are you sure you want to permanently delete ALL ${categoryName} memories? This cannot be undone.`
            );
            if (!confirmed) return;

            btnClearMemories.disabled = true;
            btnClearMemories.textContent = "Clearing...";

            try {
                const res = await fetch("/api/memories/clear", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ level: activeMemoryLevel })
                });
                const data = await res.json();
                if (data.success) {
                    showNotification(`✓ Successfully cleared all ${categoryName} memories!`, "success", 3000);
                    await fetchMemories();
                } else {
                    alert(`Clear failed: ${data.detail || data.error}`);
                }
            } catch (err) {
                alert(`Request failed: ${err}`);
            } finally {
                btnClearMemories.disabled = false;
                btnClearMemories.textContent = "🧹 Clear";
            }
        });
    }

    // ── 🎯 Chat Function Badges Handler ──
    const funcBadges = document.querySelectorAll(".func-badge");

    funcBadges.forEach(badge => {
        const funcKey = badge.dataset.func;
        let defaultVal = "true";
        if (funcKey === "enable_skills" || funcKey === "enable_books" || funcKey === "think") {
            defaultVal = "false";
        }

        const savedState = localStorage.getItem(`chat_func_${funcKey}`) || defaultVal;
        const isActive = savedState === "true";
        funcStates[funcKey] = isActive;

        badge.classList.toggle("active", isActive);

        badge.addEventListener("click", () => {
            const current = funcStates[funcKey];
            const nextState = !current;
            funcStates[funcKey] = nextState;
            localStorage.setItem(`chat_func_${funcKey}`, nextState ? "true" : "false");
            badge.classList.toggle("active", nextState);
            updateContextBudget();
            showNotification(`${badge.textContent.trim()} function is now ${nextState ? 'enabled 🟢' : 'disabled 🔴'}`, "info", 2000);
        });
    });

    // ── 🎯 Chat Functions Dropdown Menu Toggle ──
    const btnFunctionsToggle = document.getElementById("btn-chat-functions-toggle");
    const dropdownFunctions = document.getElementById("chat-functions-dropdown");

    if (btnFunctionsToggle && dropdownFunctions) {
        btnFunctionsToggle.addEventListener("click", (e) => {
            e.stopPropagation();
            const isHidden = dropdownFunctions.style.display === "none";
            dropdownFunctions.style.display = isHidden ? "flex" : "none";
            btnFunctionsToggle.classList.toggle("active", isHidden);
        });

        // Close dropdown when clicking outside
        document.addEventListener("click", (e) => {
            if (!dropdownFunctions.contains(e.target) && e.target !== btnFunctionsToggle) {
                dropdownFunctions.style.display = "none";
                btnFunctionsToggle.classList.remove("active");
            }
        });
    }

    // ── 🌡️ Temperature Override Handler ──
    const tempOverrideEnable = document.getElementById("temp-override-enable");
    const tempOverrideValue = document.getElementById("temp-override-value");

    if (tempOverrideEnable && tempOverrideValue) {
        const savedOverride = localStorage.getItem("chat_temp_override_enable") === "true";
        const savedValue = localStorage.getItem("chat_temp_override_value") || "0.7";

        tempOverrideEnable.checked = savedOverride;
        tempOverrideValue.value = savedValue;
        tempOverrideValue.style.display = savedOverride ? "inline-block" : "none";

        tempOverrideEnable.addEventListener("change", () => {
            const isChecked = tempOverrideEnable.checked;
            localStorage.setItem("chat_temp_override_enable", isChecked ? "true" : "false");
            tempOverrideValue.style.display = isChecked ? "inline-block" : "none";
            showNotification(`Temperature override is now ${isChecked ? 'enabled 🌡️' : 'disabled ❄️'}`, "info", 2000);
        });

        tempOverrideValue.addEventListener("input", () => {
            let val = parseFloat(tempOverrideValue.value);
            if (isNaN(val)) val = 0.7;
            val = Math.max(0.0, Math.min(2.0, val));
            localStorage.setItem("chat_temp_override_value", val.toString());
        });
    }

    // ── 📸 Clipboard Image Paste & Auto-expanding Textarea Handlers ──
    const pastedImages = [];

    chatInput.addEventListener("paste", (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        let imagePasted = false;
        for (const item of items) {
            if (item.type.indexOf("image") !== -1) {
                imagePasted = true;
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = function(event) {
                    const base64Str = event.target.result.split(",")[1];
                    pastedImages.push({
                        data: base64Str,
                        mimeType: item.type
                    });
                    renderPastedImagesPreview();
                    showNotification("Image pasted successfully! 📸", "success", 2000);
                };
                reader.readAsDataURL(blob);
            }
        }
        if (imagePasted) {
            e.preventDefault();
        }
    });

    function renderPastedImagesPreview() {
        const container = document.getElementById("pasted-images-container");
        if (pastedImages.length === 0) {
            container.style.display = "none";
            container.innerHTML = "";
            return;
        }
        container.style.display = "flex";
        container.innerHTML = pastedImages.map((img, idx) => `
            <div class="pasted-image-thumb" style="position: relative; width: 64px; height: 64px; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; background-color: var(--bg-app); flex-shrink: 0; box-shadow: var(--shadow-sm);">
                <img src="data:${img.mimeType};base64,${img.data}" style="width: 100%; height: 100%; object-fit: cover;" />
                <span class="pasted-image-remove" data-index="${idx}" style="position: absolute; top: 2px; right: 2px; width: 16px; height: 16px; background-color: rgba(0,0,0,0.6); color: white; font-size: 11px; font-weight: bold; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; user-select: none;">×</span>
            </div>
        `).join("");

        container.querySelectorAll(".pasted-image-remove").forEach(btn => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.dataset.index, 10);
                pastedImages.splice(idx, 1);
                renderPastedImagesPreview();
            });
        });
    }

    // Auto-expanding text height event listener
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(150, chatInput.scrollHeight) + "px";
    });

    // ── Check Server Status ──
    // Deactivate chat input by default until LLM server state is verified
    chatInput.disabled = true;
    btnChatSend.disabled = true;

    async function verifyServerStatus() {
        try {
            const res = await fetch("/status");
            const data = await res.json();
            const overlay = document.getElementById("connection-overlay");
            if (data.status === "running") {
                serverStatus.textContent = data.is_mock ? "Server Online (Mock Fallback)" : "Server Online";
                serverStatus.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
                serverStatus.style.color = "#10b981";

                if (data.needs_configuration) {
                    if (overlay) overlay.style.display = "none";
                    settingsModal.style.display = "flex";
                    chatInput.disabled = true;
                    btnChatSend.disabled = true;
                    chatInput.placeholder = "⚠️ Please configure an LLM binding to start chatting...";
                } else {
                    if (overlay) overlay.style.display = "none";
                    chatInput.disabled = false;
                    btnChatSend.disabled = false;
                    chatInput.placeholder = "Type your message or run a macro...";
                }
            }
        } catch (err) {
            const overlay = document.getElementById("connection-overlay");
            if (overlay) overlay.style.display = "none";
            serverStatus.textContent = "Offline";
            serverStatus.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
            serverStatus.style.color = "#ef4444";
        }
    }

    async function initializeApp() {
        // 1. Immediately remove the full page spinner on start so the user can interact with the UI instantly
        const globalLoader = document.getElementById("global-page-loader");
        if (globalLoader) {
            globalLoader.style.opacity = "0";
            globalLoader.style.visibility = "hidden";
            setTimeout(() => globalLoader.remove(), 400);
        }

        try {
            // 2. Load all workspace data and parameters asynchronously in the background
            await Promise.all([
                fetchArtifacts(),
                fetchMemories(),
                fetchDiscoveredTools(),
                updateContextBudget(),
                loadSettingsIntoUI(),
                fetchMessageHistory(),
                verifyServerStatus()
            ]);
        } catch (err) {
            console.error("Workspace background initialization failed:", err);
        } finally {
            // 3. Smoothly hide and unmount the localized discussion spinner once background loading is complete
            const startupSpinner = document.getElementById("discussion-startup-spinner");
            if (startupSpinner) {
                startupSpinner.style.opacity = "0";
                startupSpinner.style.transition = "opacity 0.25s ease";
                setTimeout(() => {
                    startupSpinner.style.display = "none";
                    startupSpinner.remove();
                }, 250);
            }
        }
    }

    // ── 📁 Left & Right Sidebars Collapsible Affordance ──
    const btnToggleLeft = document.getElementById("btn-toggle-left");
    const btnToggleRight = document.getElementById("btn-toggle-right");
    const sidebarLeft = document.getElementById("sidebar-left");
    const sidebarRight = document.getElementById("sidebar-right");

    if (btnToggleLeft && sidebarLeft) {
        btnToggleLeft.addEventListener("click", () => {
            const isCollapsed = sidebarLeft.classList.toggle("collapsed");
            btnToggleLeft.textContent = isCollapsed ? "▶" : "◀";
            btnToggleLeft.style.left = isCollapsed ? "0" : "360px";
            // Trigger window resize to recalculate canvas elements
            window.dispatchEvent(new Event("resize"));
        });
    }

    if (btnToggleRight && sidebarRight) {
        btnToggleRight.addEventListener("click", () => {
            const isCollapsed = sidebarRight.classList.toggle("collapsed");
            btnToggleRight.textContent = isCollapsed ? "◀" : "▶";
            btnToggleRight.style.right = isCollapsed ? "0" : "360px";
            // Trigger window resize to recalculate canvas elements
            window.dispatchEvent(new Event("resize"));
        });
    }

    // ── ⚡ Collapsible Macro Operations Section ──
    const macroHeader = document.getElementById("macro-section-header");
    const macroContent = document.getElementById("macro-section-content");

    if (macroHeader && macroContent) {
        macroHeader.addEventListener("click", () => {
            const isCollapsed = macroContent.classList.toggle("collapsed");
            const chevron = macroHeader.querySelector(".chevron");
            if (chevron) {
                chevron.style.transform = isCollapsed ? "rotate(-90deg)" : "rotate(0deg)";
            }
        });
    }

    // ── 🔍 Ctrl+K Command Palette Sub-System ──
    const cmdPaletteModal = document.getElementById("command-palette-modal");
    const cmdPaletteInput = document.getElementById("command-palette-input");
    const cmdPaletteResults = document.getElementById("command-palette-results");
    const btnCloseCmdPalette = document.getElementById("btn-close-command-palette");

    function openCommandPalette() {
        if (!cmdPaletteModal) return;
        cmdPaletteModal.style.display = "flex";
        cmdPaletteInput.value = "";
        cmdPaletteInput.focus();
        renderCommandPaletteResults("");
    }

    function closeCommandPalette() {
        if (cmdPaletteModal) {
            cmdPaletteModal.style.display = "none";
        }
    }

    if (btnCloseCmdPalette) {
        btnCloseCmdPalette.addEventListener("click", closeCommandPalette);
    }

    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            openCommandPalette();
        } else if (e.key === "Escape" && cmdPaletteModal && cmdPaletteModal.style.display === "flex") {
            closeCommandPalette();
        }
    });

    if (cmdPaletteInput) {
        cmdPaletteInput.addEventListener("input", (e) => {
            renderCommandPaletteResults(e.target.value.toLowerCase().trim());
        });
    }

    async function renderCommandPaletteResults(query) {
        if (!cmdPaletteResults) return;
        cmdPaletteResults.innerHTML = "";

        const items = [];

        // 1. Add Active Macros
        const macros = [
            { title: "Summarize Content", type: "macro", id: "macro-summarize", icon: "📝" },
            { title: "Compare Documents", type: "macro", id: "macro-compare", icon: "⚖️" },
            { title: "Build Knowledge Graph", type: "macro", id: "macro-graph", icon: "🕸️" },
            { title: "Generate Full Report", type: "macro", id: "macro-report", icon: "📊" },
            { title: "Build Infographic", type: "macro", id: "macro-infographic", icon: "🖼️" },
            { title: "Build Presentation", type: "macro", id: "macro-presentation", icon: "🎬" },
            { title: "Test Interactive Form", type: "macro", id: "macro-test-form", icon: "📋" }
        ];
        macros.forEach(m => {
            if (!query || m.title.toLowerCase().includes(query)) {
                items.push(m);
            }
        });

        // 2. Add Active Artifacts
        try {
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            arts.forEach(a => {
                if (!query || a.title.toLowerCase().includes(query)) {
                    let icon = "📄";
                    if (a.type === "skill") icon = "🎓";
                    else if (a.type === "tool") icon = "🛠️";
                    else if (a.type === "data") icon = "📊";
                    else if (a.type === "code") icon = "💻";

                    items.push({
                        title: a.title,
                        type: "artifact",
                        id: a.title,
                        icon: icon
                    });
                }
            });
        } catch (e) {
            console.error("Command palette failed to fetch artifacts:", e);
        }

        // 3. Add Cached Memories
        if (Array.isArray(cachedMemories)) {
            cachedMemories.forEach(m => {
                if (!query || m.content.toLowerCase().includes(query)) {
                    items.push({
                        title: m.content.substring(0, 60) + "...",
                        type: "memory",
                        id: m.id,
                        icon: "🧠"
                    });
                }
            });
        }

        if (items.length === 0) {
            cmdPaletteResults.innerHTML = `<li class="empty-msg" style="text-align:center; padding: 12px;">No matching results found.</li>`;
            return;
        }

        // Keep results capped at top 10 for clean look
        cmdPaletteResults.innerHTML = items.slice(0, 10).map((item, idx) => `
            <li class="command-palette-item ${idx === 0 ? 'selected' : ''}" data-type="${item.type}" data-id="${item.id}">
                <span class="item-title">${item.icon} ${item.title}</span>
                <span class="item-type">${item.type}</span>
            </li>
        `).join("");

        // Bind clicks to items
        cmdPaletteResults.querySelectorAll(".command-palette-item").forEach(el => {
            el.addEventListener("click", () => {
                const type = el.dataset.type;
                const id = el.dataset.id;

                closeCommandPalette();

                if (type === "macro") {
                    const macroBtn = document.getElementById(id);
                    if (macroBtn) {
                        macroBtn.click();
                    }
                } else if (type === "artifact") {
                    selectArtifact(id);
                } else if (type === "memory") {
                    const memCard = document.querySelector(`.memory-card[data-mem-id="${id}"]`);
                    if (memCard) {
                        memCard.scrollIntoView({ behavior: "smooth", block: "center" });
                        memCard.style.outline = "2px solid var(--accent-color)";
                        setTimeout(() => {
                            memCard.style.outline = "none";
                        }, 2000);
                    } else {
                        // Switch tab to the memory level first
                        const foundMem = cachedMemories.find(m => m.id === id);
                        if (foundMem) {
                            activeMemoryLevel = foundMem.level;
                            renderMemories();
                            setTimeout(() => {
                                const freshCard = document.querySelector(`.memory-card[data-mem-id="${id}"]`);
                                if (freshCard) {
                                    freshCard.scrollIntoView({ behavior: "smooth", block: "center" });
                                    freshCard.style.outline = "2px solid var(--accent-color)";
                                    setTimeout(() => {
                                        freshCard.style.outline = "none";
                                    }, 2000);
                                }
                            }, 150);
                        }
                    }
                }
            });
        });
    }

    // Stored Memories TBox/ABox Toggle Listener
    const btnToggleTBoxABox = document.getElementById("btn-toggle-tbox-abox");
    if (btnToggleTBoxABox) {
        btnToggleTBoxABox.addEventListener("click", () => {
            memViewMode = memViewMode === "abox" ? "tbox" : "abox";
            btnToggleTBoxABox.classList.toggle("active", memViewMode === "tbox");
            btnToggleTBoxABox.textContent = memViewMode === "tbox" ? "📐 TBox (Schema)" : "📐 ABox (Instances)";
            renderMemories();
        });
    }

    // Left Sidebar Dream Action Listener
    const btnDreamMemories = document.getElementById("btn-dream-memories-part");
    if (btnDreamMemories) {
        btnDreamMemories.addEventListener("click", async () => {
            btnDreamMemories.disabled = true;
            btnDreamMemories.textContent = "Dreaming...";
            try {
                const res = await fetch("/api/memories/dream", { method: "POST" });
                const data = await res.json();
                if (data.success && data.report) {
                    const r = data.report;
                    if (r.skipped) {
                        alert("💤 Dreaming skipped: " + r.reason);
                    } else {
                        alert(`💤 Subconscious Dream Consolidation complete!\n\n• Decayed: ${r.decayed} memory nodes\n• Promoted: ${r.promoted} nodes\n• Reinforced: ${r.reinforced} nodes\n• Forgotten: ${r.forgotten} nodes\n• Retained by Dreamer: ${r.retained_by_dreamer} nodes\n• Fused (Redundant Merges): ${r.fused_nodes || 0} nodes\n• Audited Orphans: ${r.audited_orphans || 0} nodes\n• Time taken: ${r.duration_seconds.toFixed(2)}s`);
                        fetchMemories();
                    }
                } else {
                    alert("Consolidation pass returned no report.");
                }
            } catch (err) {
                alert("Dreaming pass failed: " + err);
            } finally {
                btnDreamMemories.disabled = false;
                btnDreamMemories.textContent = "💤 Dream";
            }
        });
    }

    // Left Sidebar Tab Switcher
    const sidebarTabButtons = document.querySelectorAll(".sidebar-tab-btn");
    const sidebarTabPanes = document.querySelectorAll(".sidebar-tab-pane");

    sidebarTabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            sidebarTabButtons.forEach(b => b.classList.remove("active"));
            sidebarTabPanes.forEach(p => {
                p.classList.remove("active");
                p.style.display = "none";
            });

            btn.classList.add("active");
            const targetPane = document.getElementById(btn.dataset.sidebarTab);
            if (targetPane) {
                targetPane.classList.add("active");
                targetPane.style.display = "flex";
            }
        });
    });

    // Launch workspace bootstrapper
    initializeApp();

    // ── 🧠 Startup Connection Guard & Safety Timer ──
    // If the server is offline, broken, or hanging during startup, this guard ensures the user
    // is never stuck on the splash loader. It forces the loader off and opens settings.
    setTimeout(() => {
        const globalLoader = document.getElementById("global-page-loader");
        if (globalLoader) {
            console.warn("Connection guard triggered: Forcing loader removal and opening settings.");
            globalLoader.style.opacity = "0";
            globalLoader.style.visibility = "hidden";
            setTimeout(() => globalLoader.remove(), 400);

            const settingsModal = document.getElementById("settings-modal");
            if (settingsModal) {
                settingsModal.style.display = "flex";
                showNotification("⚠️ Connection to LLM Server failed. Please verify your binding settings.", "warning", 5000);
            }
        }
    }, 15000); // 15 seconds generous safety limit to prevent premature triggers on slower environments

    // ── Combined Ingestion Modal Display Toggle ──
    btnOpenImport.addEventListener("click", () => {
        importModal.style.display = "flex";
        progressContainer.style.display = "none";
        progressBarFill.style.width = "0%";
    });

    btnCloseImport.addEventListener("click", () => {
        importModal.style.display = "none";
    });


    
    // Bulletproof click-outside closure tracking (prevents accidental close during text selection drags)
    let isMouseDownOnImportOverlay = false;
    importModal.addEventListener("mousedown", (e) => {
        isMouseDownOnImportOverlay = (e.target === importModal);
    });
    importModal.addEventListener("mouseup", (e) => {
        if (isMouseDownOnImportOverlay && e.target === importModal) {
            importModal.style.display = "none";
        }
        isMouseDownOnImportOverlay = false;
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
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();

            const maxTokens = (data && data.max_tokens) ? data.max_tokens : 8192;
            const currentTokens = (data && data.current_tokens) ? data.current_tokens : 0;
            const percent = ((currentTokens / maxTokens) * 100).toFixed(1);

            contextBudgetText.textContent = `${currentTokens.toLocaleString()} / ${maxTokens.toLocaleString()} tokens (${percent}%)`;

            let sysTokens = 0;
            const sysCtx = (data && data.zones) ? data.zones.system_context : null;
            if (sysCtx) {
                sysTokens = sysCtx.tokens || 0;
            }

            let artTokens = 0;
            if (sysCtx && sysCtx.breakdown && sysCtx.breakdown.artefacts) {
                artTokens = sysCtx.breakdown.artefacts.tokens || 0;
            }

            let memTokens = 0;
            if (sysCtx && sysCtx.breakdown) {
                if (sysCtx.breakdown.working_memory) {
                    memTokens += sysCtx.breakdown.working_memory.tokens || 0;
                }
                if (sysCtx.breakdown.deep_memory) {
                    memTokens += sysCtx.breakdown.deep_memory.tokens || 0;
                }
            }

            // Subtract sub-components from raw system context block to avoid double-counting
            sysTokens = Math.max(0, sysTokens - artTokens - memTokens);

            let histTokens = 0;
            const histCtx = data.zones.message_history;
            if (histCtx) {
                histTokens = histCtx.tokens || 0;
            }

            const sysPercent = ((sysTokens / maxTokens) * 100).toFixed(2);
            const artPercent = ((artTokens / maxTokens) * 100).toFixed(2);
            const memPercent = ((memTokens / maxTokens) * 100).toFixed(2);
            const histPercent = ((histTokens / maxTokens) * 100).toFixed(2);

            contextBudgetBar.innerHTML = `
                <div class="context-budget-segment system" style="width: ${sysPercent}%" title="System / Static Prompt: ${sysTokens} tokens"></div>
                <div class="context-budget-segment artifacts" style="width: ${artPercent}%" title="Active Artifacts: ${artTokens} tokens"></div>
                <div class="context-budget-segment memories" style="width: ${memPercent}%" title="Stored Memories: ${memTokens} tokens"></div>
                <div class="context-budget-segment history" style="width: ${histPercent}%" title="Message History: ${histTokens} tokens"></div>
            `;

            contextBudgetLegend.innerHTML = `
                <div class="legend-item"><span class="legend-color-dot system"></span> System (${sysTokens.toLocaleString()})</div>
                <div class="legend-item"><span class="legend-color-dot artifacts"></span> Artifacts (${artTokens.toLocaleString()})</div>
                <div class="legend-item"><span class="legend-color-dot memories" style="background-color: var(--success-color);"></span> Memories (${memTokens.toLocaleString()})</div>
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

        if (targetBtn) {
            targetBtn.classList.add("active");
        }
        if (targetContent) {
            targetContent.classList.add("active");

            // Dispatch a window resize event to force any lazy-rendered elements or iframes to rescale instantly
            window.dispatchEvent(new Event('resize'));

            // If the content contains an iframe, trigger a resize on its window too
            const iframe = targetContent.querySelector('iframe');
            if (iframe && iframe.contentWindow) {
                try {
                    iframe.contentWindow.dispatchEvent(new Event('resize'));
                } catch(e) {}
            }
        }
    }

    function sanitizeTabTitle(title) {
        if (!title) return "";
        // Normalize away trailing type annotations e.g. "myfile.md (type=document)" -> "myfile.md"
        return title.replace(/\s*\(type=\w+\)\.?\s*$/i, "").trim();
    }

    function createArtifactTab(title, type) {
        const cleanTitle = sanitizeTabTitle(title);
        const safeId = makeSafeId(cleanTitle);
        const tabId = `tab-art-${safeId}`;

        if (openTabs.has(cleanTitle)) {
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
        tabBtn.dataset.artTitle = cleanTitle;
        tabBtn.innerHTML = `
            <span>${icon} ${cleanTitle}</span>
            <span class="tab-close" data-art-title="${cleanTitle}">×</span>
        `;
        workspaceTabs.appendChild(tabBtn);

        const tabContent = document.createElement("div");
        tabContent.className = "tab-content";
        tabContent.id = tabId;
        tabContent.innerHTML = `
            <div class="artifact-sub-tabs">
                <button class="sub-tab-btn active" data-sub-tab="rendered-${safeId}">👁️ Rendered View</button>
                <button class="sub-tab-btn" data-sub-tab="raw-${safeId}">💻 Raw Source Code</button>
                <div class="version-select-container" style="display: flex; align-items: center; gap: 6px;">
                    <label>Version:</label>
                    <select class="version-select" data-art-title="${title}"></select>
                    <button class="btn-version-action btn-set-active-version" data-art-title="${title}" title="⭐ Set this version as active/default in the session context" style="background: none; border: 1px solid var(--border-color); color: var(--text-secondary); padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: bold; transition: all 0.15s; height: auto; width: auto; line-height: 1.4;">⭐ Set Active</button>
                    <button class="btn-version-action btn-delete-version" data-art-title="${title}" title="🗑️ Delete this specific version permanently" style="background: none; border: 1px solid var(--border-color); color: #ef4444; border-color: rgba(239, 68, 68, 0.2); padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: bold; transition: all 0.15s; height: auto; width: auto; line-height: 1.4;">🗑️ Delete</button>
                    <button class="btn-version-action btn-squash-versions" data-art-title="${title}" title="🗜️ Compress/Squash version history" style="background: none; border: 1px solid var(--border-color); color: var(--accent-color); border-color: rgba(245, 158, 11, 0.2); padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: bold; transition: all 0.15s; height: auto; width: auto; line-height: 1.4;">🗜️ Squash</button>
                    <button class="sub-tab-btn download-art-btn" data-art-title="${title}" title="Download / Export this artifact" style="margin-left: 8px; background-color: var(--chat-user-bg); color: white; border-color: var(--chat-user-bg); padding: 6px 12px; height: auto;">💾 Download / Export</button>
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

        openTabs.add(cleanTitle);

        const subTabButtons = tabContent.querySelectorAll(".sub-tab-btn");
        const subTabPanes = tabContent.querySelectorAll(".sub-tab-content");
        subTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                subTabButtons.forEach(b => b.classList.remove("active"));
                subTabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                tabContent.querySelector(`#sub-tab-${btn.dataset.subTab}`).classList.add("active");
                if (btn.dataset.subTab === `rendered-${safeId}` && codeEditors[cleanTitle]) {
                    setTimeout(() => codeEditors[cleanTitle].refresh(), 10);
                }
            });
        });

        const vSelect = tabContent.querySelector(".version-select");
        const btnSetActive = tabContent.querySelector(".btn-set-active-version");
        const btnDeleteVer = tabContent.querySelector(".btn-delete-version");
        const btnSquash = tabContent.querySelector(".btn-squash-versions");

        // 1. Browsing a version simply loads it visually (no auto-activation)
        vSelect.addEventListener("change", async (e) => {
            const selectedVer = parseInt(e.target.value, 10);
            await loadArtifactVersion(cleanTitle, selectedVer, type);
        });

        // 2. Set Active Button Click
        btnSetActive.addEventListener("click", async () => {
            const selectedVer = parseInt(vSelect.value, 10);
            btnSetActive.disabled = true;
            btnSetActive.textContent = "Setting...";
            try {
                const res = await fetch(`/api/artifacts/${encodeURIComponent(cleanTitle)}/select_version?version=${selectedVer}`, {
                    method: "POST"
                });
                const data = await res.json();
                if (data.success) {
                    alert(`Version v${selectedVer} is now set as the active/default version in the session context!`);
                    await fetchArtifacts(); // Refreshes active/dormant icons in left sidebar
                    await selectArtifact(cleanTitle); // Reload dropdown listing & state
                }
            } catch (err) {
                alert(`Failed to set active version: ${err}`);
            } finally {
                btnSetActive.disabled = false;
                btnSetActive.textContent = "⭐ Set Active";
            }
        });

        // 3. Delete Version Button Click
        btnDeleteVer.addEventListener("click", async () => {
            const selectedVer = parseInt(vSelect.value, 10);
            const confirmed = await showSlickConfirm(
                "🗑️ Delete Artifact Version",
                `Are you sure you want to permanently delete version v${selectedVer} of the artifact '${cleanTitle}'? This cannot be undone.`
            );
            if (!confirmed) return;

            btnDeleteVer.disabled = true;
            btnDeleteVer.textContent = "Deleting...";
            try {
                const res = await fetch(`/api/artifacts/${encodeURIComponent(cleanTitle)}/versions/${selectedVer}`, {
                    method: "DELETE"
                });
                const data = await res.json();
                if (data.success) {
                    alert(`Version v${selectedVer} successfully deleted.`);
                    await fetchArtifacts();
                    // Reload artifact – this automatically switches the view to the new active version
                    await selectArtifact(cleanTitle);
                } else {
                    alert(`Deletion failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Request failed: ${err}`);
            } finally {
                btnDeleteVer.disabled = false;
                btnDeleteVer.textContent = "🗑️ Delete";
            }
        });

        // 4. Squash Versions Button Click
        btnSquash.addEventListener("click", async () => {
            const numStr = await showCustomPrompt(
                "🗜️ Squash Version History",
                "Number of recent versions to keep (e.g., 3), or type 'active' to squash all other versions.",
                "3"
            );
            if (!numStr) return;

            const payload = {};
            if (numStr.toLowerCase() === "active") {
                const selectedVer = parseInt(vSelect.value, 10);
                payload.target_version = selectedVer;
            } else {
                const keepVal = parseInt(numStr, 10);
                if (isNaN(keepVal) || keepVal < 1) {
                    alert("Please enter a valid positive number.");
                    return;
                }
                payload.keep_last_n = keepVal;
            }

            btnSquash.disabled = true;
            btnSquash.textContent = "Squashing...";
            try {
                const res = await fetch(`/api/artifacts/${encodeURIComponent(cleanTitle)}/squash`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    alert(`Successfully squashed version history! Space reclaimed: ${data.report.space_reclaimed_estimate.toLocaleString()} characters.`);
                    await fetchArtifacts();
                    await selectArtifact(cleanTitle);
                } else {
                    alert(`Squash failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Request failed: ${err}`);
            } finally {
                btnSquash.disabled = false;
                btnSquash.textContent = "🗜️ Squash";
            }
        });

        // Bind Context-Aware Download/Export Click
        const downloadBtn = tabContent.querySelector(".download-art-btn");
        if (downloadBtn) {
            downloadBtn.addEventListener("click", () => {
                if (type === "image") {
                    // Direct binary download for the currently selected version of the image
                    const selectedVer = parseInt(vSelect.value, 10) || 1;
                    const imgIndex = selectedVer - 1; // 0-based index
                    const a = document.createElement("a");
                    a.href = `/api/images/${encodeURIComponent(cleanTitle)}/${imgIndex}`;
                    a.download = `${cleanTitle}_v${selectedVer}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                } else {
                    // Open the unified export modal with this artifact pre-selected
                    openExportModal("artifacts");
                    setTimeout(() => {
                        const sel = document.getElementById("export-art-select");
                        if (sel) {
                            sel.value = cleanTitle;
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
            selectArtifact(cleanTitle);
        });

        const closeBtn = tabBtn.querySelector(".tab-close");
        closeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            closeArtifactTab(cleanTitle);
        });

        // Bind raw source code "Save Changes" button
        const saveRawBtn = tabContent.querySelector(`.save-raw-btn`);
        const rawTextarea = tabContent.querySelector(`.raw-textarea`);

        saveRawBtn.addEventListener("click", async () => {
            saveRawBtn.disabled = true;
            saveRawBtn.textContent = "Saving...";

            // Clear cache for this artifact to ensure fresh data is loaded
            for (const key of datasetCache.keys()) {
                if (key.startsWith(`${cleanTitle}-`)) {
                    datasetCache.delete(key);
                }
            }

            try {
                const res = await fetch(`/api/artifacts/${encodeURIComponent(cleanTitle)}/update`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ content: rawTextarea.value })
                });
                const data = await res.json();
                if (data.success) {
                    alert(`Artifact successfully saved as version ${data.version}!`);
                    // Re-fetch list and reload selection to refresh the Rendered View!
                    await fetchArtifacts();
                    await selectArtifact(cleanTitle);
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
        const cleanTitle = sanitizeTabTitle(title);
        const safeId = makeSafeId(cleanTitle);
        const tabId = `tab-art-${safeId}`;
        const tabBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
        const tabContent = document.getElementById(tabId);

        if (tabBtn) tabBtn.remove();
        if (tabContent) tabContent.remove();

        openTabs.delete(cleanTitle);
        if (codeEditors[cleanTitle]) {
            delete codeEditors[cleanTitle];
        }

        if (activeArtifactTitle === cleanTitle) {
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
    btnCreateTool.addEventListener("click", async () => {
        const rawName = await showCustomPrompt("Create New Tool", "Enter a snake_case name (e.g. file_zipper)");
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

    // ── 🛠️ Import LCP Tool Handler ──
    const btnImportTool = document.getElementById("btn-import-tools-part");
    const toolImportFileInput = document.getElementById("tool-import-file-input");

    if (btnImportTool && toolImportFileInput) {
        btnImportTool.addEventListener("click", () => toolImportFileInput.click());

        toolImportFileInput.addEventListener("change", async (e) => {
            if (e.target.files.length === 0) return;
            const file = e.target.files[0];
            const formData = new FormData();
            formData.append("file", file);

            btnImportTool.disabled = true;
            btnImportTool.textContent = "Importing...";

            try {
                const res = await fetch("/api/tools/import", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    showNotification(`Successfully imported tool: '${file.name}'! 🛠️`, "success", 3000);
                    fetchDiscoveredTools();
                    fetchArtifacts(); // refresh active artifacts list
                } else {
                    alert(`Import failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Request failed: ${err}`);
            } finally {
                btnImportTool.disabled = false;
                btnImportTool.textContent = "📥 Import";
                toolImportFileInput.value = ""; // reset
            }
        });
    }
    // Extract utility helper to capture python code blocks inside Markdown
    function extractCodeFromMarkdown(content) {
        const match = content.match(/```python\s*\n([\s\S]*?)\n```/i);
        return match ? match[1] : content;
    }

    // ── 💾 Load Specific Version of an Artifact ──
    // ── 💾 Load Specific Version of an Artifact ──
    async function loadArtifactVersion(title, version, type = null) {
        const safeId = makeSafeId(title);
        const renderedView = document.getElementById(`rendered-view-${safeId}`);
        const rawView = document.getElementById(`raw-view-${safeId}`);

        if (!renderedView || !rawView) return;

        // Reset inline styles before tab loading to prevent style contamination
        renderedView.removeAttribute("style");
        renderedView.className = "rendered-container";

        // Show loading spinner immediately
        renderedView.innerHTML = `
            <div class="empty-viewer-msg" style="user-select: none;">
                <span class="spinner" style="width: 32px; height: 32px; border-width: 3px; margin-right: 12px;"></span>
                <span style="font-weight: bold; color: var(--text-secondary);">Loading artifact content...</span>
            </div>
        `;

        try {
            // Retrieve type-specific details locally to prevent global state contamination
            let activeType = type;
            let isReadOnly = false;
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            const matched = arts.find(a => a.title === title);
            if (matched) {
                activeType = matched.type;
                isReadOnly = matched.read_only;
            }

            if (activeType === "image") {
                // Direct high-fidelity binary image preview in Rendered View
                const selectedVer = version || 1;
                const imgIndex = selectedVer - 1; // 0-based
                renderedView.style.height = "100%";
                renderedView.style.overflow = "auto";
                renderedView.style.display = "flex";
                renderedView.style.flexDirection = "column";
                renderedView.style.justifyContent = "center";
                renderedView.style.alignItems = "center";
                renderedView.style.padding = "20px";

                renderedView.innerHTML = `
                    <div class="rendered-image-container" style="max-width: 100%; max-height: 100%; display: flex; flex-direction: column; align-items: center; gap: 12px; margin: 0;">
                        <img src="/api/images/${encodeURIComponent(title)}/${imgIndex}?version=${selectedVer}" style="max-width: 100%; max-height: 65vh; object-fit: contain; border-radius: 8px; border: 1px solid var(--border-color); box-shadow: var(--shadow-lg);" />
                        <span style="font-size: 11px; color: var(--text-secondary); font-weight: bold; background-color: rgba(0,0,0,0.4); padding: 4px 12px; border-radius: 12px;">🎨 ${title} (v${selectedVer})</span>
                    </div>
                `;

                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;
            } else if (activeType === "data") {
                await renderSpreadsheetGridInTab(title, version, renderedView);

                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;
            } else if (activeType === "tool") {
                // Renders the Full Interactive LCP Tool Builder Workspace inside Rendered View!
                renderToolEditor(title, version, renderedView);

                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;
            } else if (activeType === "presentation") {
                // ── Render Presentation Slide Decks inside an Isolated Iframe ──
                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}?version=${version}`);
                const art = await artRes.json();
                rawView.value = art.content;

                // Resolve <artefact_image> custom tags into standard browser-supported <img> tags before creating the render Blob
                let resolvedContent = art.content.replace(/<artefact_image\s+id=["']([^"']+)["']\s*(?:\/>|>)/gi, (match, imageId) => {
                    if (imageId.includes("::")) {
                        const parts = imageId.split("::");
                        const imgIndex = parts[parts.length - 1];
                        const imgTitle = parts.slice(0, -1).join("::");
                        return `<img src="/api/images/${encodeURIComponent(imgTitle)}/${imgIndex}" style="width:100%; height:100%; object-fit:cover;" />`;
                    }
                    return match;
                });

                const blob = new Blob([resolvedContent], { type: "text/html" });
                const iframeUrl = URL.createObjectURL(blob);
                renderedView.innerHTML = `
                    <iframe src="${iframeUrl}" style="width: 100%; height: 100%; border: none; min-height: 520px; border-radius: 8px; background: #000;" id="presentation-frame-${safeId}"></iframe>
                `;
            } else {
                const artRes = await fetch(`/api/artifacts/${encodeURIComponent(title)}${version !== undefined ? `?version=${version}` : ""}`);
                const art = await artRes.json();
                rawView.value = art.content;

                const isHtml = title.endsWith(".html") || title.endsWith(".htm") || 
                               (art.language && art.language === "html") || 
                               art.content.trim().toLowerCase().startsWith("<!doctype html>") || 
                               art.content.trim().toLowerCase().startsWith("<html>");

                if (isHtml) {
                    if (codeEditors[title]) {
                        try { codeEditors[title].toTextArea(); } catch(e) {}
                        delete codeEditors[title];
                    }
                    renderedView.style.height = "100%";
                    renderedView.style.overflow = "hidden";
                    renderedView.style.display = "flex";
                    renderedView.style.flexDirection = "column";

                    renderedView.innerHTML = `
                        <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-shrink: 0;">
                            <span style="font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: bold;">HTML Source</span>
                            <button class="btn btn-primary" id="btn-run-code-${safeId}" style="width:auto; padding: 6px 14px; font-size: 12px;">▶ Run HTML</button>
                        </div>
                        <div class="code-editor-scroll-container" style="flex: 1; min-height: 150px; overflow: hidden; display: flex; flex-direction: column;">
                            <textarea id="cm-rendered-${safeId}"></textarea>
                        </div>
                        <div id="code-run-output-${safeId}" class="console-locked-panel" style="display:none; margin-top: 12px; flex-shrink: 0; max-height: 220px; overflow-y: auto; background-color: #020617; border: 1px solid var(--border-color); border-radius: 6px; padding: 12px; flex-direction: column; gap: 6px;"></div>
                    `;
                    const txtArea = renderedView.querySelector(`#cm-rendered-${safeId}`);
                    const cm = CodeMirror.fromTextArea(txtArea, {
                        mode: "htmlmixed",
                        theme: "dracula",
                        lineNumbers: true,
                        readOnly: true,
                        lineWrapping: true,
                        tabSize: 4,
                    });
                    cm.setValue(art.content);
                    codeEditors[title] = cm;

                    const runBtn = renderedView.querySelector(`#btn-run-code-${safeId}`);
                    runBtn.addEventListener("click", () => {
                        const blob = new Blob([art.content], { type: "text/html" });
                        const url = URL.createObjectURL(blob);
                        window.open(url, "_blank");
                        setTimeout(() => URL.revokeObjectURL(url), 60000);
                    });
                } else if (activeType === "code") {
                    const lang = art.language || "";
                    if (lang === "graphviz" || art.content.includes("digraph") || art.content.includes("graph")) {
                        renderedView.style.height = "100%";
                        renderedView.style.overflow = "auto";
                        renderedView.style.display = "block";
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
                        renderedView.style.height = "100%";
                        renderedView.style.overflow = "hidden";
                        renderedView.style.display = "flex";
                        renderedView.style.flexDirection = "column";

                        renderedView.innerHTML = `
                            <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-shrink: 0;">
                                <span style="font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: bold;">${lang} Source</span>
                                <button class="btn btn-primary" id="btn-run-code-${safeId}" style="width:auto; padding: 6px 14px; font-size: 12px;">▶ Run ${lang.toUpperCase()}</button>
                            </div>
                            <div class="code-editor-scroll-container" style="flex: 1; min-height: 150px; overflow: hidden; display: flex; flex-direction: column;">
                                <textarea id="cm-rendered-${safeId}"></textarea>
                            </div>
                            <div id="code-run-output-${safeId}" class="console-locked-panel" style="display:none; margin-top: 12px; flex-shrink: 0; max-height: 220px; overflow-y: auto; background-color: #020617; border: 1px solid var(--border-color); border-radius: 6px; padding: 12px; flex-direction: column; gap: 6px;"></div>
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
                                outputBox.innerHTML = `<div class="empty-viewer-msg" style="user-select: none; justify-content: flex-start; padding: 0;"><span class="spinner inline" style="margin-right: 8px;"></span> Executing Python sandbox...</div>`;
                                try {
                                    const res = await fetch("/api/execute_sandbox", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ code: art.content, language: "python" })
                                    });
                                    const data = await res.json();
                                    if (data.success) {
                                        outputBox.innerHTML = `
                                            <div class="query-title" style="user-select: none;">💻 Sandbox Output</div>
                                            <pre class="query-stdout">${data.output || "(No stdout output)"}</pre>
                                        `;
                                    } else {
                                        outputBox.innerHTML = `
                                            <div class="query-title" style="color: #ef4444; user-select: none;">❌ Execution Error</div>
                                            <pre class="query-stdout" style="color: #fca5a5;">${data.error}</pre>
                                            ${data.output ? `<div class="query-title" style="margin-top: 6px; user-select: none;">Stdout prior to error:</div><pre class="query-stdout">${data.output}</pre>` : ""}
                                        `;
                                    }
                                } catch (err) {
                                    outputBox.innerHTML = `<div class="query-title" style="color: #ef4444; user-select: none;">Request Failed</div><pre class="query-stdout">${err}</pre>`;
                                } finally {
                                    runBtn.disabled = false;
                                    runBtn.textContent = "▶ Run PYTHON";
                                }
                            }
                        });
                    } else {
                        renderedView.style.height = "100%";
                        renderedView.style.overflow = "auto";
                        renderedView.style.display = "block";
                        const renderContent = `\`\`\`${lang}\n${art.content}\n\`\`\``;
                        renderedView.innerHTML = marked.parse(renderContent);
                        renderMath(renderedView);
                    }
                } else {
                    renderedView.style.height = "100%";
                    renderedView.style.overflow = "auto";
                    renderedView.style.display = "block";

                    const placeholders = {};
                    let maskedText = art.content;

                    // Mask <lollms_inline>
                    const inlinePattern = /<lollms_inline\s*([^>]*)>([\s\S]*?)<\/lollms_inline>/gi;
                    maskedText = maskedText.replace(inlinePattern, (match) => {
                        const id = `__WIDGET_PLACEHOLDER_${Math.random().toString(36).substr(2, 9)}__`;
                        placeholders[id] = match;
                        return id;
                    });

                    // Mask <lollms_form>
                    const formPattern = /<lollms_form\s*([^>]*)>([\s\S]*?)<\/lollms_form>/gi;
                    maskedText = maskedText.replace(formPattern, (match) => {
                        const id = `__FORM_PLACEHOLDER_${Math.random().toString(36).substr(2, 9)}__`;
                        placeholders[id] = match;
                        return id;
                    });

                    const parsedMarkdown = marked.parse(maskedText);

                    let restoredHTML = parsedMarkdown;
                    for (const [id, original] of Object.entries(placeholders)) {
                        restoredHTML = restoredHTML.replace(id, original);
                    }

                    const resolvedHTML = resolveImageAnchors(restoredHTML, activeArtifactTitle, null, version);
                    renderedView.innerHTML = restoredHTML;
                    renderMath(renderedView);
                }
            }
            updateContextBudget();

            // After loading, update button highlight states
            const tabId = `tab-art-${safeId}`;
            const tabContent = document.getElementById(tabId);
            if (tabContent) {
                updateVersionActionButtonsState(tabContent, title, version);
            }
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
        const cacheKey = `${docTitle}-${version}`;
        if (datasetCache.has(cacheKey)) {
            const cachedData = datasetCache.get(cacheKey);
            if (cachedData.type === "excel" || cachedData.type === "sqlite") {
                renderTabbedSheets(cachedData, docTitle, containerElement);
            } else {
                renderFlatTable(cachedData, docTitle, containerElement);
            }
            return;
        }

        containerElement.innerHTML = `<div class="empty-viewer-msg"><span class="spinner inline" style="margin-right: 8px;"></span> Loading dataset spreadsheet...</div>`;
        try {
            let url = `/api/data/${encodeURIComponent(docTitle)}`;
            if (version !== undefined) {
                url += `?version=${version}`;
            }
            const res = await fetch(url);
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
            }
            const data = await res.json();

            // Cache the loaded data
            datasetCache.set(cacheKey, data);

            if (data.type === "excel" || data.type === "sqlite" || data.type === "sql_connection") {
                renderTabbedSheets(data, docTitle, containerElement);
            } else {
                renderFlatTable(data, docTitle, containerElement);
            }
        } catch (err) {
            containerElement.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Failed to render spreadsheet: ${err}</div>`;
        }
    }

    function drawTableInTab(sheetData, docTitle) {
        const gridTarget = document.getElementById(`spreadsheet-grid-target-${makeSafeId(docTitle)}`);
        if (!gridTarget) return;

        const columns = sheetData?.columns;
        const rows = sheetData?.rows;

        if (!columns || !rows) {
            gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 20px;">Failed to draw table: Invalid sheet data structure.</div>`;
            return;
        }

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

            if (vSelect) {
                vSelect.dataset.activeVersion = activeVersionObj ? activeVersionObj.version : "";
                vSelect.dataset.readOnly = matched.read_only ? "true" : "false";
            }

            loadArtifactVersion(title, selectedVersion, activeArtifactType);

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
    const macroTestForm = document.getElementById("macro-test-form");

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

    // 🎬 Presentation Builder Dashboard Modal & Trigger
    const presModal = document.getElementById("presentation-dashboard-modal");
    const btnClosePresModal = document.getElementById("btn-close-presentation-modal");
    const btnCancelPres = document.getElementById("btn-cancel-presentation");
    const btnBuildPres = document.getElementById("btn-build-presentation");

    if (btnClosePresModal) {
        btnClosePresModal.addEventListener("click", () => presModal.style.display = "none");
    }
    if (btnCancelPres) {
        btnCancelPres.addEventListener("click", () => presModal.style.display = "none");
    }

    macroPresentation.addEventListener("click", async () => {
        presModal.style.display = "flex";
        const infoDiv = document.getElementById("pres-available-images-info");
        infoDiv.style.display = "none";

        // Query active workspace artifacts to identify if any image assets are available for embedding
        try {
            const res = await fetch("/api/artifacts");
            const arts = await res.json();
            const activeImages = arts.filter(a => a.active && a.type === "image");
            if (activeImages.length > 0) {
                infoDiv.style.display = "block";
                infoDiv.innerHTML = `📸 Found <strong>${activeImages.length} active image(s)</strong> available for embedding:<br>${activeImages.map(a => `• <em>${a.title}</em>`).join("<br>")}`;
            }
        } catch (e) {
            console.error("Failed to fetch active images for presentation builder:", e);
        }
    });

    macroTestForm.addEventListener("click", () => {
        triggerMacroPrompt(
            "Generate an interactive form using the <lollms_form> system to let the user configure a new cloud deployment.\n\n" +
            "The form MUST include exactly these fields:\n" +
            "1. A text field named 'project_name' with label 'Project Name'\n" +
            "2. A select field named 'cloud_provider' with label 'Cloud Provider' and options: AWS, GCP, Azure, DigitalOcean\n" +
            "3. A radio field named 'server_size' with label 'Server Size' and options: Micro, Medium, Large\n" +
            "4. A checkbox field named 'enable_backups' with label 'Enable Backups'\n" +
            "5. A range field named 'scaling_instances' with label 'Scaling Instances' (min 1, max 10, default 2)\n\n" +
            "Include a polite greeting and let the user know they can submit the configuration. Do NOT wrap the tags in markdown fences."
        );
    });

    if (btnBuildPres) {
        btnBuildPres.addEventListener("click", () => {
            const theme = document.getElementById("pres-style").value;
            const slideCount = parseInt(document.getElementById("pres-slides-count").value, 10) || 5;
            const hints = document.getElementById("pres-structure-hints").value.trim();
            const embedImages = document.getElementById("pres-embed-images").checked;

            let imageInstructions = "";
            if (embedImages) {
                imageInstructions = `
                - IMAGE EMBEDDING IS AUTHORIZED: Yes. You can embed the active image artifacts using the following anchor tag format inside your slides:
                  <figure class="slide-image"><artefact_image id="IMAGE_TITLE::0" /></figure>
                  Make sure to match the image title exactly from the active artifacts list. Position these figures nicely inside a grid or two-column slide layout.`;
            } else {
                imageInstructions = `
                - IMAGE EMBEDDING IS AUTHORIZED: No. Do not attempt to embed active images or reference any <artefact_image> tags. Use solid colors, styled text, and custom CSS layouts instead.`;
            }

            const promptText = `Convert the active artifacts into a complete, beautiful slide deck presentation in semantic HTML5.

            DESIGN & STRUCTURE SPECIFICATIONS:
            - Theme / Style: ${theme.toUpperCase()} style. Use customized CSS variables, backgrounds, fonts, and borders aligned with this style.
            - Expected Slide Panels: Exactly ${slideCount} slides.
            - Structure / Outline Hints: ${hints ? hints : "Synthesize the core themes of the active artifacts across the slides."}
            ${imageInstructions}

            Use custom inline CSS styles for layout, transitions, charts, and data figures, and add speakernotes (using 'data-notes' on each <section class="slide">). 
            Save your response as a new presentation artifact by wrapping it in the exact XML tag: 
            <artifact name="slideshow_presentation.html" type="presentation"> ... </artifact>`;

            triggerMacroPrompt(promptText);
            presModal.style.display = "none";
        });
    }

    // ── Local Ingestion Handlers (Relocated into Modal) ──
    dropzone.addEventListener("click", () => fileInput.click());

    // Additional safety check for folder browse button
    setTimeout(() => {
        const btnBrowseFolder = document.getElementById("btn-browse-folder");
        const folderInput = document.getElementById("folder-input");
        if (btnBrowseFolder && folderInput) {
            console.log("✅ Folder browse button initialized");
            // Ensure the button has proper cursor and clickability
            btnBrowseFolder.style.cursor = "pointer";
            btnBrowseFolder.style.opacity = "1";
        }
    }, 100);

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

    // ── Folder Import Handlers ──
    const folderPath = document.getElementById("folder-path");
    const btnValidateFolder = document.getElementById("btn-validate-folder");
    const folderImportSection = document.getElementById("folder-import-section");

    // Toggle folder import section visibility based on mode selection
    if (importMode && folderImportSection) {
        function toggleFolderImportSection() {
            const mode = importMode.value;
            console.log("🔍 Toggle folder section - Current mode:", mode);
            if (mode === "data_bundle") {
                folderImportSection.style.display = "flex";
                folderImportSection.style.flexDirection = "column";
                folderImportSection.style.gap = "8px";
                folderImportSection.style.marginTop = "12px";
                console.log("✅ Folder section SHOWN");
            } else {
                folderImportSection.style.display = "none";
                console.log("❌ Folder section HIDDEN");
            }
        }

        // Initial state - check immediately
        console.log("🚀 Initial mode value:", importMode.value);
        toggleFolderImportSection();

        // Listen for mode changes
        importMode.addEventListener("change", () => {
            console.log("🔄 Mode changed to:", importMode.value);
            toggleFolderImportSection();
        });
    } else {
        console.error("❌ Missing elements:");
        console.log("  - importMode:", importMode);
        console.log("  - folderImportSection:", folderImportSection);
    }

    // Validate folder path when user clicks validate button
    if (btnValidateFolder && folderPath) {
        btnValidateFolder.addEventListener("click", async () => {
            const pathValue = folderPath.value.trim();
            if (!pathValue) {
                alert("Please enter a folder path first.");
                return;
            }

            btnValidateFolder.disabled = true;
            btnValidateFolder.textContent = "Validating...";

            try {
                // Send validation request to backend
                const res = await fetch("/api/validate_folder", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ folder_path: pathValue })
                });

                const data = await res.json();

                if (data.success) {
                    folderPath.style.borderColor = "var(--success-color)";
                    setTimeout(() => {
                        folderPath.style.borderColor = "var(--border-color)";
                    }, 2000);
                    alert(`✅ Valid folder path: ${pathValue}`);
                } else {
                    folderPath.style.borderColor = "#ef4444";
                    setTimeout(() => {
                        folderPath.style.borderColor = "var(--border-color)";
                    }, 2000);
                    alert(`❌ Invalid folder path: ${data.error || "Path does not exist or is not accessible"}`);
                }
            } catch (err) {
                alert(`Validation failed: ${err}`);
            } finally {
                btnValidateFolder.disabled = false;
                btnValidateFolder.textContent = "✓ Validate Path";
            }
        });
    }

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
        const mode = importMode.value;

        // Handle folder bundle import
        if (mode === "data_bundle") {
            if (!folderPath.value || folderPath.value.trim() === "") {
                alert("Please specify a folder path. Use the Validate button first to verify the path exists.");
                return;
            }

            const pathValue = folderPath.value.trim();
            console.log("🚀 Starting folder bundle import for:", pathValue);

            btnSubmit.disabled = true;
            progressContainer.style.display = "flex";
            progressBarFill.style.width = "0%";
            progressStatusText.textContent = "Processing folder bundle ingestion...";

            try {
                const description = document.getElementById("custom-description").value.trim();
                const res = await fetch("/api/import_folder", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        folder_path: pathValue,
                        title: customTitle.value || "folder_bundle",
                        description: description || null
                    })
                });

                const data = await res.json();
                console.log("📊 Folder import response:", data);

                if (data.success) {
                    alert(`✅ Successfully imported ${data.file_count || 'multiple'} files into ${data.groups || 'data'} group(s)!`);
                    importModal.style.display = "none";
                    selectedFiles = [];
                    folderPath.value = "";
                    dropzone.innerHTML = `<p>Drag & drop PDF, DOCX, PPTX, MD, CSV, XLSX, DB, or Image file here, or click to upload (supports multiple files)</p>`;
                    customTitle.value = "";
                    const descField = document.getElementById("custom-description");
                    if (descField) descField.value = "";
                    fetchArtifacts();
                } else {
                    alert(`❌ Folder import failed: ${data.error || "Unknown error"}`);
                }
            } catch (err) {
                console.error("❌ Folder import request failed:", err);
                alert(`❌ Folder import request failed: ${err.message || err}`);
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.textContent = "Process Document";
                progressContainer.style.display = "none";
            }
            return;
        }

        // Handle regular file import
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
            formData.append("mode", mode);
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

                function processImportLine(line) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ")) {
                        try {
                            const rawJson = trimmed.slice(6);
                            const event = JSON.parse(rawJson);
                            handleStreamEvent(event, currentNum, totalNum);
                        } catch (e) {
                            console.error("Failed to parse import event:", e);
                        }
                    }
                }

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) {
                        if (buffer.trim()) {
                            processImportLine(buffer);
                        }
                        break;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split(/\r?\n/);
                    buffer = lines.pop();

                    for (const line of lines) {
                        processImportLine(line);
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

        if (type === "youtube" || type === "github" || type === "stackoverflow" || (type === "scopus" && document.getElementById("scopus-query").value.trim().startsWith("http"))) {
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
            else if (type === "scopus" && document.getElementById("scopus-query").value.trim().startsWith("http")) {
                const url = document.getElementById("scopus-query").value.trim();
                // Extract scopus ID from URL if possible
                const idMatch = url.match(/eid=2-s2.0-(\d+)/) || url.match(/record\/display\.uri\?eid=([^&]+)/);
                const scopusId = idMatch ? idMatch[1] : url.split("/").pop();
                await importDirectURL("/api/scopus/import", { items: [{ scopus_id: scopusId, title: "Scopus Article" }], auto_load: true });
            }
            else {
                await executeSearch(type);
            }
        } catch (err) {
            alert(`Search/Ingestion failed: ${err}`);
        } finally {
            btnInternetSearch.disabled = false;
            btnInternetSearch.textContent = (type === "youtube" || type === "github" || type === "stackoverflow" || (type === "scopus" && document.getElementById("scopus-query").value.trim().startsWith("http"))) ? "Import URL directly" : "Search / Ingest";
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
        } else if (type === "google_scholar") {
            const q = document.getElementById("scholar-query").value.trim();
            if (!q) { alert("Please specify search keywords."); return; }
            endpoint = "/api/google_scholar/search";
            payload = { query: q, max_results: 5 };
        } else if (type === "scopus") {
            const q = document.getElementById("scopus-query").value.trim();
            if (!q) { alert("Please specify Scopus search keywords."); return; }
            endpoint = "/api/scopus/search";
            payload = { query: q, max_results: 5 };
        } else if (type === "orbit") {
            const q = document.getElementById("orbit-query").value.trim();
            if (!q) { alert("Please specify Orbit search keywords."); return; }
            endpoint = "/api/orbit/search";
            payload = { query: q, max_results: 5 };
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
            let url = r.url || r.pdf_url || (r.id ? `https://arxiv.org/abs/${r.id}` : "#");
            // Extract Scopus / Patent IDs safely
            let id_str = r.scopus_id || r.patent_id || r.id || "";
            let id_badge = id_str ? `<span class="level-tag deep" style="margin-left:6px; font-size:9px; padding:1px 4px; display:inline-block;">ID: ${id_str}</span>` : "";

            const titleHtml = url !== "#" 
                ? `<a href="${url}" target="_blank" class="search-result-link" title="Open source link in new tab">${r.title} 🔗</a>${id_badge}` 
                : `<span class="search-result-title">${r.title}</span>${id_badge}`;

            return `
                <li class="search-result-card">
                    <input type="checkbox" class="result-check" data-index="${idx}">
                    <div class="search-result-content">
                        ${titleHtml}
                        <span class="search-result-snippet" style="white-space: pre-wrap; margin-top:4px;">${r.snippet || r.abstract || r.url || ""}</span>
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
        } else if (type === "google_scholar") {
            endpoint = "/api/google_scholar/import";
            payload = {
                items: selectedItems.map(item => {
                    const snippetParts = (item.snippet || "").split("\n");
                    const author = snippetParts[0] || "Unknown";
                    const abstract = snippetParts.slice(1).join("\n") || "No abstract available.";
                    return { title: item.title, author: author, abstract: abstract, url: item.url };
                }),
                auto_load: true
            };
        } else if (type === "scopus") {
            endpoint = "/api/scopus/import";
            payload = {
                items: selectedItems.map(item => {
                    const url = item.url || "";
                    const idMatch = url.match(/eid=2-s2.0-(\d+)/) || url.match(/record\/display\.uri\?eid=([^&]+)/);
                    const scopusId = idMatch ? idMatch[1] : url.split("/").pop();
                    return { scopus_id: scopusId, title: item.title };
                }),
                auto_load: true
            };
        } else if (type === "orbit") {
            endpoint = "/api/orbit/import";
            payload = {
                items: selectedItems.map(item => {
                    const snippet = item.snippet || "";
                    const idMatch = snippet.match(/Patent ID:\s*([A-Za-z0-9]+)/i);
                    const patentId = idMatch ? idMatch[1] : item.title.split(" ").pop();
                    return { patent_id: patentId, title: item.title };
                }),
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

            function processSkillLine(line) {
                const trimmed = line.trim();
                if (trimmed.startsWith("data: ")) {
                    try {
                        const rawJson = trimmed.slice(6);
                        const event = JSON.parse(rawJson);
                        if (event.type === "result") {
                            alert("Successfully loaded skill: '" + event.title + "'!");
                            importModal.style.display = "none";
                            fetchArtifacts();
                        } else if (event.type === "error") {
                            alert("Error: " + event.message);
                        }
                    } catch (e) {
                        console.error("Failed to parse skill import event:", e);
                    }
                }
            }

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    if (buffer.trim()) {
                        processSkillLine(buffer);
                    }
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split(/\r?\n/);
                buffer = lines.pop();

                for (const line of lines) {
                    processSkillLine(line);
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
        // Show loading spinner immediately
        exportModalBody.innerHTML = `
            <div class="empty-viewer-msg" style="user-select: none; padding: 40px 0;">
                <span class="spinner" style="width: 28px; height: 28px; border-width: 2.5px; margin-right: 10px;"></span>
                <span style="font-weight: bold; color: var(--text-secondary); font-size: 13px;">Retrieving available items...</span>
            </div>
        `;
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
                                <option value="zip">Complete Zip Bundle (.zip)</option>
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
        loadSettingsIntoUI();
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

    // Bulletproof click-outside closure tracking for settings
    let isMouseDownOnSettingsOverlay = false;
    settingsModal.addEventListener("mousedown", (e) => {
        isMouseDownOnSettingsOverlay = (e.target === settingsModal);
    });
    settingsModal.addEventListener("mouseup", (e) => {
        if (isMouseDownOnSettingsOverlay && e.target === settingsModal) {
            settingsModal.style.display = "none";
        }
        isMouseDownOnSettingsOverlay = false;
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

    async function fetchSttBindings() {
        try {
            const res = await fetch("/api/bindings/stt");
            const data = await res.json();
            if (data.success) {
                const currentValue = selSttBinding.value;
                availableSttBindings = data.bindings;
                selSttBinding.innerHTML = `<option value="">-- Select STT Binding --</option>` +
                    availableSttBindings.map(b => `<option value="${b.binding_name}">${b.title || b.binding_name}</option>`).join("");
                if (currentValue) {
                    selSttBinding.value = currentValue;
                }
            }
        } catch (err) {
            console.error("Failed to fetch STT bindings:", err);
        }
    }

    async function fetchTtiBindings() {
        try {
            const res = await fetch("/api/bindings/tti");
            const data = await res.json();
            if (data.success) {
                const currentValue = selTtiBinding.value;
                availableTtiBindings = data.bindings;
                selTtiBinding.innerHTML = `<option value="">-- Select TTI Binding --</option>` +
                    availableTtiBindings.map(b => `<option value="${b.binding_name}">${b.title || b.binding_name}</option>`).join("");
                if (currentValue) {
                    selTtiBinding.value = currentValue;
                }
            }
        } catch (err) {
            console.error("Failed to fetch TTI bindings:", err);
        }
    }

    function renderSttConfigForm(bindingName) {
        const binding = availableSttBindings.find(b => b.binding_name === bindingName);
        if (!binding || !binding.input_parameters) {
            sttConfigForm.innerHTML = `<p class="empty-msg">No configurable parameters for this STT binding.</p>`;
            return;
        }

        const savedConfig = savedSttConfigs[bindingName] || {};

        sttConfigForm.innerHTML = binding.input_parameters.map(p => {
            if (p.name === "model_name") return "";
            const key = p.name;
            const label = p.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
            const type = p.type || "string";

            const defaultVal = savedConfig[key] !== undefined ? savedConfig[key] : (p.default !== undefined ? p.default : "");
            const desc = p.description || "";

            const isSecret = key.toLowerCase().includes("key") || key.toLowerCase().includes("token") || p.is_secret;

            if (type === "bool" || type === "boolean") {
                return `
                    <div class="input-group">
                        <label style="flex-direction:row; align-items:center; gap:8px; text-transform:none;">
                            <input type="checkbox" id="stt-cfg-${key}" data-stt-key="${key}" ${defaultVal ? "checked" : ""} />
                            <span>${label}</span>
                        </label>
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (type === "int" || type === "integer" || type === "float" || type === "number") {
                return `
                    <div class="input-group">
                        <label for="stt-cfg-${key}">${label}</label>
                        <input type="number" id="stt-cfg-${key}" data-stt-key="${key}" data-type="${type}" value="${defaultVal}" step="${type === 'float' || type === 'number' ? '0.01' : '1'}" />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else if (isSecret) {
                return `
                    <div class="input-group">
                        <label for="stt-cfg-${key}">${label}</label>
                        <div style="position: relative; display: flex; align-items: center; width: 100%;">
                            <input type="password" id="stt-cfg-${key}" data-stt-key="${key}" value="${defaultVal}" placeholder="${desc}" style="padding-right: 40px;" />
                            <button type="button" class="btn-toggle-secret" data-target="stt-cfg-${key}" style="position: absolute; right: 8px; background: none; border: none; color: var(--accent-color); cursor: pointer; font-size: 16px; outline: none; padding: 4px;" title="Toggle Visibility">👁️</button>
                        </div>
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            } else {
                return `
                    <div class="input-group">
                        <label for="stt-cfg-${key}">${label}</label>
                        <input type="text" id="stt-cfg-${key}" data-stt-key="${key}" value="${defaultVal}" placeholder="${desc}" />
                        <span style="font-size:10px; color:var(--text-secondary); margin-top:2px;">${desc}</span>
                    </div>`;
            }
        }).join("");

        sttConfigForm.querySelectorAll(".btn-toggle-secret").forEach(btn => {
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

    function renderTtiConfigForm(bindingName) {
        const binding = availableTtiBindings.find(b => b.binding_name === bindingName);
        if (!binding || !binding.input_parameters) {
            ttiConfigForm.innerHTML = `<p class="empty-msg">No configurable parameters for this TTI binding.</p>`;
            return;
        }

        const savedConfig = savedTtiConfigs[bindingName] || {};

        ttiConfigForm.innerHTML = binding.input_parameters.map(p => {
            if (p.name === "model_name") return "";
            const key = p.name;
            const label = p.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
            const type = p.type || "string";

            // Use saved value if present, otherwise default
            const defaultVal = savedConfig[key] !== undefined ? savedConfig[key] : (p.default !== undefined ? p.default : "");
            const desc = p.description || "";

            // Detect secret fields (API keys, tokens, etc.)
            const isSecret = key.toLowerCase().includes("key") || key.toLowerCase().includes("token") || p.is_secret;

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
            } else if (isSecret) {
                return `
                    <div class="input-group">
                        <label for="tti-cfg-${key}">${label}</label>
                        <div style="position: relative; display: flex; align-items: center; width: 100%;">
                            <input type="password" id="tti-cfg-${key}" data-tti-key="${key}" value="${defaultVal}" placeholder="${desc}" style="padding-right: 40px;" />
                            <button type="button" class="btn-toggle-secret" data-target="tti-cfg-${key}" style="position: absolute; right: 8px; background: none; border: none; color: var(--accent-color); cursor: pointer; font-size: 16px; outline: none; padding: 4px;" title="Toggle Visibility">👁️</button>
                        </div>
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

        // Attach event listeners for secret toggles
        ttiConfigForm.querySelectorAll(".btn-toggle-secret").forEach(btn => {
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

    function collectSttBindingConfig() {
        const inputs = sttConfigForm.querySelectorAll("[data-stt-key]");
        const cfg = {};
        inputs.forEach(inp => {
            const key = inp.dataset.sttKey || inp.getAttribute("data-stt-key");
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

    function renderSttModelDropdown(modelsList) {
        selSttModel.innerHTML = modelsList.map(m => {
            const value = m.model_name || m.name || m.id || m;
            const label = m.display_name || value;
            return `<option value="${value}">${label}</option>`;
        }).join("");
        selSttModel.dispatchEvent(new Event("change"));
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

    let discoveredTtiZoo = [];

    async function populateTtiZoo(bindingName) {
        const zooSelect = document.getElementById("settings-tti-zoo-select");
        const downloadBtn = document.getElementById("btn-download-tti-zoo");
        const statusDiv = document.getElementById("tti-zoo-status");

        zooSelect.innerHTML = `<option value="">-- Select Zoo Model --</option>`;
        downloadBtn.style.display = "none";
        statusDiv.style.display = "none";

        if (!bindingName) return;

        downloadBtn.style.display = "inline-block";
        statusDiv.className = "validation-status";
        statusDiv.style.display = "flex";
        statusDiv.textContent = "Fetching TTI Zoo list from binding...";

        try {
            const res = await fetch("/api/bindings/tti/zoo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    binding_name: bindingName,
                    config: collectTtiBindingConfig()
                })
            });
            const data = await res.json();
            if (data.success && Array.isArray(data.zoo)) {
                discoveredTtiZoo = data.zoo;
                if (discoveredTtiZoo.length === 0) {
                    statusDiv.className = "validation-status error";
                    statusDiv.textContent = "No models available in this binding's Zoo.";
                    zooSelect.innerHTML = `<option value="">No models available</option>`;
                    downloadBtn.style.display = "none";
                } else {
                    renderTtiZooDropdown(discoveredTtiZoo);
                    statusDiv.style.display = "none";
                }
            } else {
                statusDiv.className = "validation-status error";
                statusDiv.textContent = "Could not fetch Zoo list: " + (data.error || "Unknown error.");
            }
        } catch (err) {
            statusDiv.className = "validation-status error";
            statusDiv.textContent = "Request failed: " + err;
        }
    }

    function renderTtiZooDropdown(zooList) {
        const zooSelect = document.getElementById("settings-tti-zoo-select");
        zooSelect.innerHTML = `<option value="">-- Select Zoo Model --</option>` +
            zooList.map(m => `<option value="${m.link}">${m.name} (${m.size || 'unknown size'})</option>`).join("");
    }

    const ttiZooSearch = document.getElementById("settings-tti-zoo-search");
    if (ttiZooSearch) {
        ttiZooSearch.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderTtiZooDropdown(discoveredTtiZoo);
                return;
            }
            const filtered = discoveredTtiZoo.filter(m => 
                m.name.toLowerCase().includes(query) || 
                (m.description && m.description.toLowerCase().includes(query)) ||
                m.link.toLowerCase().includes(query)
            );
            renderTtiZooDropdown(filtered);
        });
    }

    const btnDownloadTtiZoo = document.getElementById("btn-download-tti-zoo");
    if (btnDownloadTtiZoo) {
        btnDownloadTtiZoo.addEventListener("click", async () => {
            const zooSelect = document.getElementById("settings-tti-zoo-select");
            const statusDiv = document.getElementById("tti-zoo-status");
            const modelLink = zooSelect.value;

            if (!modelLink) {
                alert("Please select a zoo model to download first.");
                return;
            }

            btnDownloadTtiZoo.disabled = true;
            btnDownloadTtiZoo.textContent = "Downloading...";
            statusDiv.className = "validation-status";
            statusDiv.style.display = "flex";
            statusDiv.textContent = "Downloading model from Hub. This may take several minutes...";

            try {
                const res = await fetch("/api/bindings/execute_command", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        modality: "tti",
                        binding_name: selTtiBinding.value,
                        command_name: "pull_model",
                        parameters: {
                            model_name: modelLink
                        }
                    })
                });
                const data = await res.json();
                if (data.success) {
                    statusDiv.className = "validation-status success";
                    statusDiv.textContent = "✅ Model downloaded and installed successfully!";
                    // Refresh available models list
                    await fetchCurrentTtiModels();
                } else {
                    statusDiv.className = "validation-status error";
                    statusDiv.textContent = "❌ Download failed: " + (data.error || "Unknown error.");
                }
            } catch (err) {
                statusDiv.className = "validation-status error";
                statusDiv.textContent = "❌ Request failed: " + err;
            } finally {
                btnDownloadTtiZoo.disabled = false;
                btnDownloadTtiZoo.textContent = "📥 Download Selected Model";
            }
        });
    }

    sttModelSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
            renderSttModelDropdown(discoveredSttModels);
            return;
        }
        const filtered = discoveredSttModels.filter(m => {
            const value = m.model_name || m.name || m.id || m;
            const label = m.display_name || value;
            return value.toLowerCase().includes(query) || label.toLowerCase().includes(query);
        });
        renderSttModelDropdown(filtered);
    });

    selSttBinding.addEventListener("change", () => {
        if (isConfigLoading) return;
        const prevSttBinding = selSttBinding.dataset.prevSttBinding || "";
        if (selSttBinding.value === prevSttBinding) return;

        if (prevSttBinding) {
            savedSttConfigs[prevSttBinding] = collectSttBindingConfig();
        }
        selSttBinding.dataset.prevSttBinding = selSttBinding.value;

        sttModelGroup.style.display = "none";
        valSttStatus.className = "validation-status";
        valSttStatus.textContent = "";
        renderSttConfigForm(selSttBinding.value);
    });

    btnValidateStt.addEventListener("click", async () => {
        if (!selSttBinding.value) {
            valSttStatus.className = "validation-status error";
            valSttStatus.textContent = "Please select an STT binding first.";
            return;
        }
        btnValidateStt.disabled = true;
        btnValidateStt.textContent = "Validating...";
        valSttStatus.className = "validation-status";
        valSttStatus.textContent = "Connecting to STT service...";

        const payload = {
            binding_name: selSttBinding.value,
            config: collectSttBindingConfig()
        };

        try {
            const res = await fetch("/api/bindings/stt/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success && Array.isArray(data.models)) {
                valSttStatus.className = "validation-status success";
                valSttStatus.textContent = `✅ Connection successful. Found ${data.models.length} model(s).`;

                discoveredSttModels = data.models;
                sttModelSearch.value = "";

                renderSttModelDropdown(discoveredSttModels);
                sttModelGroup.style.display = "flex";
                currentSttBindingConfig = payload.config;
            } else {
                valSttStatus.className = "validation-status error";
                valSttStatus.textContent = `❌ Validation failed: ${data.error || "Could not reach STT process."}`;
                sttModelGroup.style.display = "none";
            }
        } catch (err) {
            valSttStatus.className = "validation-status error";
            valSttStatus.textContent = `❌ Request failed: ${err}`;
        } finally {
            btnValidateStt.disabled = false;
            btnValidateStt.textContent = "🔌 Validate & Load Models";
        }
    });

    selTtiBinding.addEventListener("change", () => {
        if (isConfigLoading) return;
        const prevTtiBinding = selTtiBinding.dataset.prevTtiBinding || "";
        if (selTtiBinding.value === prevTtiBinding) return;

        // UX Cache: Save current input values to local map before switching form
        if (prevTtiBinding) {
            savedTtiConfigs[prevTtiBinding] = collectTtiBindingConfig();
        }
        selTtiBinding.dataset.prevTtiBinding = selTtiBinding.value;

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
        const confirmed = await showSlickConfirm(
            "⚡ Load Profile",
            `Are you sure you want to load the profile '${name.toUpperCase()}'? This will reinitialize active model servers.`,
            "Load Profile",
            false
        );
        if (!confirmed) return;
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
        const confirmed = await showSlickConfirm(
            "🗑️ Delete Profile",
            `Are you sure you want to permanently delete the profile '${name.toUpperCase()}'?`
        );
        if (!confirmed) return;
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
                const currentValue = selBinding.value;
                availableBindings = data.bindings;
                selBinding.innerHTML = `<option value="">-- Select Binding --</option>` +
                    availableBindings.map(b => `<option value="${b.binding_name}">${b.title || b.binding_name}</option>`).join("");
                if (currentValue) {
                    selBinding.value = currentValue;
                }
            }
        } catch (err) {
            console.error("Failed to fetch bindings:", err);
        }
    }

    function renderConfigForm(bindingName) {
        const binding = availableBindings.find(b => b.binding_name === bindingName);
        if (!binding || !binding.input_parameters) {
            configForm.innerHTML = `<p class="empty-msg">Select a binding to configure its parameters.</p>`;
            return;
        }

        const savedConfig = savedLlmConfigs[bindingName] || {};

        configForm.innerHTML = binding.input_parameters.map(p => {
            if (p.name === "model_name") return ""; // model selection happens after validation
            const key = p.name;
            const label = p.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
            const type = p.type || "string";
            const required = p.mandatory ? "required" : "";
            
            // Priority: Saved value in historical map -> Default value -> empty string
            const defaultVal = savedConfig[key] !== undefined ? savedConfig[key] : (p.default !== undefined ? p.default : "");
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
        if (isConfigLoading) return;
        const prevBinding = selBinding.dataset.prevBinding || "";
        if (selBinding.value === prevBinding) return;

        // UX Cache: Save current input values to local map before switching form
        if (prevBinding) {
            savedLlmConfigs[prevBinding] = collectBindingConfig();
        }
        selBinding.dataset.prevBinding = selBinding.value;

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

    // Bind open settings folder button
    const btnOpenSettingsFolder = document.getElementById("btn-open-settings-folder");
    if (btnOpenSettingsFolder) {
        btnOpenSettingsFolder.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/settings/open_folder", { method: "POST" });
                const data = await res.json();
                if (!data.success) {
                    alert("Failed to open settings folder: " + data.detail);
                }
            } catch (err) {
                alert("Request failed: " + err);
            }
        });
    }

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

            // Collect optional STT configuration directly from the form inputs
            if (selSttBinding.value && selSttModel.value) {
                payload["stt_binding_name"] = selSttBinding.value;
                payload["stt_binding_config"] = { ...collectSttBindingConfig(), model_name: selSttModel.value };
            }

            // Collect optional Internet Search configuration directly from the form inputs
            const internetConfig = {
                google_api_key: document.getElementById("settings-google-api-key").value.trim(),
                google_cse_id: document.getElementById("settings-google-cse-id").value.trim(),
                scopus_api_key: document.getElementById("settings-scopus-api-key").value.trim(),
                scopus_inst_token: document.getElementById("settings-scopus-inst-token").value.trim(),
                orbit_api_key: document.getElementById("settings-orbit-api-key").value.trim(),
                scholar_api_key: document.getElementById("settings-scholar-api-key").value.trim()
            };
            payload["internet_config"] = internetConfig;

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

                    // Re-enable chat after successful configuration save
                    chatInput.disabled = false;
                    btnChatSend.disabled = false;
                    chatInput.placeholder = "Type your message or run a macro...";

                    setTimeout(() => {
                        settingsModal.style.display = "none";
                        valStatus.className = "validation-status";
                        valStatus.textContent = "";
                    }, 800);
                }                if (data.success) {
                    valStatus.className = "validation-status success";
                    valStatus.textContent = "✅ Configuration applied successfully.";
                    await fetchCurrentModels();

                    // Re-enable chat after successful configuration save
                    chatInput.disabled = false;
                    btnChatSend.disabled = false;
                    chatInput.placeholder = "Type your message or run a macro...";

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

    async function fetchCurrentTtiModels(savedModelName = null) {
        try {
            const currentSelection = savedModelName || headerTtiModel.value;
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
                        const hasCurrent = testData.models.some(m => {
                            const v = m.model_name || m.name || m.id || m;
                            return v === currentSelection;
                        });

                        let html = `<option value="">-- Active TTI Model --</option>`;
                        if (currentSelection && !hasCurrent) {
                            html += `<option value="${currentSelection}">${currentSelection}</option>`;
                        }

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

    async function fetchCurrentModels(savedModelName = null) {
        try {
            const currentSelection = savedModelName || headerModel.value;
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
            isConfigLoading = true;
            console.log("🔍 Loading settings into UI...");
            const res = await fetch("/api/settings");
            const data = await res.json();

            console.log("📦 Settings data:", data);

            savedLlmConfigs = data.llm_bindings_configs || {};
            savedTtiConfigs = data.tti_bindings_configs || {};
            savedSttConfigs = data.stt_bindings_configs || {};

            // Populate Internet config inputs
            const internetConfig = data.internet_config || {};
            document.getElementById("settings-google-api-key").value = internetConfig.google_api_key || "";
            document.getElementById("settings-google-cse-id").value = internetConfig.google_cse_id || "";
            document.getElementById("settings-scopus-api-key").value = internetConfig.scopus_api_key || "";
            document.getElementById("settings-scopus-inst-token").value = internetConfig.scopus_inst_token || "";
            document.getElementById("settings-orbit-api-key").value = internetConfig.orbit_api_key || "";
            document.getElementById("settings-scholar-api-key").value = internetConfig.scholar_api_key || "";

            // First, load all binding options
            await fetchBindings();
            console.log("📋 Available bindings loaded:", availableBindings.map(b => b.binding_name));
            console.log("📋 SelBinding options:", Array.from(selBinding.options).map(o => o.value));

            await fetchTtiBindings();
            await fetchSttBindings();
            await fetchProfiles();

            if (data.success && data.llm_binding_name) {
                console.log("💾 Saved binding name:", data.llm_binding_name);

                // Verify the saved binding exists in the loaded options before selecting it
                const bindingExists = availableBindings.some(b => b.binding_name === data.llm_binding_name);
                console.log("✅ Binding exists check:", bindingExists);

                if (bindingExists) {
                    console.log("⚙️ Pre-selecting binding:", data.llm_binding_name);

                    // Pre-select and render the form synchronously
                    selBinding.value = data.llm_binding_name;
                    selBinding.dataset.prevBinding = data.llm_binding_name;
                    renderConfigForm(data.llm_binding_name);
                    // Wait for DOM to render the selection, then trigger change event
                    setTimeout(() => {
                        console.log("🔁 Verifying binding selection after DOM render...");
                        console.log("📋 selBinding.value after timeout:", selBinding.value);

                        // IMPORTANT: Trigger change event to populate config form
                        const changeEvent = new Event('change', { bubbles: true });
                        selBinding.dispatchEvent(changeEvent);
                        console.log("📤 Change event dispatched for binding selection");

                        // Verify selection persisted
                        setTimeout(() => {
                            console.log("🔍 Final verification - selBinding.value:", selBinding.value);
                            if (selBinding.value !== data.llm_binding_name) {
                                console.error("❌ Binding selection was reset! Re-applying...");
                                selBinding.value = data.llm_binding_name;
                                console.log("✅ Re-applied binding selection:", selBinding.value);
                            }
                        }, 50);
                    }, 100);                    

                    // Populate config inputs directly
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

                    // Populates the configuration UI form without blocking the thread with heavy /test calls
                    renderLlmModelDropdown([config.model_name || "unknown"]);
                    modelGroup.style.display = "flex";
                    currentBindingConfig = config;
                }

                // ── Load TTI Configuration ──
                if (data.tti_binding_name) {
                    const ttiBindingExists = availableTtiBindings.some(b => b.binding_name === data.tti_binding_name);
                    if (ttiBindingExists) {
                        selTtiBinding.value = data.tti_binding_name;
                        selTtiBinding.dataset.prevTtiBinding = data.tti_binding_name;
                        renderTtiConfigForm(data.tti_binding_name);

                        const ttiConfig = data.tti_binding_config || {};
                        for (const [key, value] of Object.entries(ttiConfig)) {
                            const input = document.getElementById("tti-cfg-" + key);
                            if (input) {
                                if (input.type === "checkbox") {
                                    input.checked = !!value;
                                } else {
                                    input.value = Array.isArray(value) ? JSON.stringify(value) : value;
                                }
                            }
                        }
                        renderTtiModelDropdown([ttiConfig.model_name || "unknown"]);
                        ttiModelGroup.style.display = "flex";
                        currentTtiBindingConfig = ttiConfig;
                    }
                }

                if (data.stt_binding_name) {
                    const sttBindingExists = availableSttBindings.some(b => b.binding_name === data.stt_binding_name);
                    if (sttBindingExists) {
                        selSttBinding.value = data.stt_binding_name;
                        selSttBinding.dataset.prevSttBinding = data.stt_binding_name;
                        renderSttConfigForm(data.stt_binding_name);

                        const r_key_map = data.stt_binding_config || {};
                        for (const [key, value] of Object.entries(r_key_map)) {
                            const input = document.getElementById("stt-cfg-" + key);
                            if (input) {
                                if (input.type === "checkbox") {
                                    input.checked = !!value;
                                } else {
                                    input.value = Array.isArray(value) ? JSON.stringify(value) : value;
                                }
                            }
                        }
                        renderSttModelDropdown([r_key_map.model_name || "unknown"]);
                        sttModelGroup.style.display = "flex";
                        currentSttBindingConfig = r_key_map;
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
                const ollamaExists = availableBindings.some(b => b.binding_name === "ollama");
                if (ollamaExists) {
                    selBinding.value = "ollama";
                    selBinding.dispatchEvent(new Event("change"));

                    const hostInput = document.getElementById("cfg-host_address");
                    if (hostInput) hostInput.value = "http://localhost:11434";
                }

                btnOpenSettings.click();
            }

            // Populate models list first before setting values
            const savedLlmModelName = (data.success && data.llm_binding_config) ? data.llm_binding_config.model_name : null;
            const savedTtiModelName = (data.success && data.tti_binding_config) ? data.tti_binding_config.model_name : null;

            await fetchCurrentModels(savedLlmModelName);
            await fetchCurrentTtiModels(savedTtiModelName);
            await fetchPersonalities();

            // Explicitly sync the top header dropdown value with the saved setting
            if (savedLlmModelName) {
                headerModel.value = savedLlmModelName;
            }
            if (savedTtiModelName) {
                headerTtiModel.value = savedTtiModelName;
            }
        } catch (err) {
            console.error("Failed to load settings:", err);
        } finally {
            isConfigLoading = false;
        }
    }
    function closeExportModal() {
        exportDetailsModal.style.display = "none";
        activeExportType = null;
    }

    btnCloseExportModal.addEventListener("click", closeExportModal);
    btnCancelExport.addEventListener("click", closeExportModal);

    // Active Cytoscape Instance Reference
    let activeCyInstance = null;

    // ── Memory Network Graph Modal Handlers ──
    const btnViewGraph = document.getElementById("btn-view-memory-graph");
    const memoryGraphModal = document.getElementById("memory-graph-modal");
    const btnCloseGraph = document.getElementById("btn-close-memory-graph");
    const graphViewport = document.getElementById("memory-graph-viewport");

    if (btnViewGraph) {
        btnViewGraph.addEventListener("click", () => {
            memoryGraphModal.style.display = "flex";
            renderMemoryNetworkGraph();
            // Critical Fix: Trigger Cytoscape redraw/recenter after DOM layout paint completes
            setTimeout(() => {
                if (activeCyInstance) {
                    activeCyInstance.resize();
                    activeCyInstance.invalidateDimensions && activeCyInstance.invalidateDimensions();
                    activeCyInstance.fit();
                    activeCyInstance.center();
                }
            }, 100);
        });
    }

    const btnRecenterGraph = document.getElementById("btn-recenter-memory-graph");
    if (btnRecenterGraph) {
        btnRecenterGraph.addEventListener("click", () => {
            if (activeCyInstance) {
                activeCyInstance.resize();
                activeCyInstance.fit();
                activeCyInstance.center();
                showNotification("🎯 Graph recentered and scaled to viewport.", "success", 2000);
            }
        });
    }

    if (btnCloseGraph) {
        btnCloseGraph.addEventListener("click", () => {
            memoryGraphModal.style.display = "none";
        });
    }

    // Bulletproof click-outside closure tracking for memory graph modal
    let isMouseDownOnGraphOverlay = false;
    if (memoryGraphModal) {
        memoryGraphModal.addEventListener("mousedown", (e) => {
            isMouseDownOnGraphOverlay = (e.target === memoryGraphModal);
        });
        memoryGraphModal.addEventListener("mouseup", (e) => {
            if (isMouseDownOnGraphOverlay && e.target === memoryGraphModal) {
                memoryGraphModal.style.display = "none";
            }
            isMouseDownOnGraphOverlay = false;
        });
    }

    async function renderMemoryNetworkGraph() {
        const viewport = document.getElementById("memory-graph-viewport");
        const inspector = document.getElementById("inspector-content");

        if (!viewport || !inspector) return;

        const graphToggleBtn = document.getElementById("btn-graph-toggle-tbox-abox");
        if (graphToggleBtn && !graphToggleBtn.dataset.listenerAdded) {
            graphToggleBtn.dataset.listenerAdded = "true";
            graphToggleBtn.addEventListener("click", () => {
                graphViewMode = graphViewMode === "abox" ? "tbox" : "abox";
                graphToggleBtn.textContent = graphViewMode === "tbox" ? "📐 Switch to ABox Instances" : "📐 Switch to TBox Schema";
                graphToggleBtn.classList.toggle("active", graphViewMode === "tbox");
                renderMemoryNetworkGraph();
            });
        }

        if (graphViewMode === "tbox") {
            viewport.innerHTML = "";
            inspector.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 12px; width: 100%;">
                    <span class="level-tag deep" style="font-size: 9.5px; font-weight: bold; text-transform: uppercase; color: var(--accent-color);">TBox Ontological Schema</span>
                    <p style="font-size: 12.5px; line-height: 1.5; color: var(--text-primary); margin: 0;">This graph represents the abstract terminological box (TBox) schema of the memory system, defining valid concept classes and their relationships.</p>
                    <div style="border-top: 1px solid var(--border-color); padding-top: 10px; font-size: 11px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 6px;">
                        <span>• Tap any class node to inspect its description.</span>
                        <span>• Hover over edges to see their semantic predicates.</span>
                    </div>
                </div>
            `;

            const tboxElements = [
                { group: "nodes", data: { id: "CONCEPT", label: "LABEL: CONCEPT", color: "#818cf8" } },
                { group: "nodes", data: { id: "PREFERENCE", label: "LABEL: PREFERENCE", color: "#10b981" } },
                { group: "nodes", data: { id: "EVENT", label: "LABEL: EVENT", color: "#f59e0b" } },
                { group: "nodes", data: { id: "DECISION", label: "LABEL: DECISION", color: "#f43f5e" } },

                { group: "edges", data: { id: "e1", source: "PREFERENCE", target: "CONCEPT", label: "PREFERS" } },
                { group: "edges", data: { id: "e2", source: "CONCEPT", target: "DECISION", label: "IMPLEMENTS" } },
                { group: "edges", data: { id: "e3", source: "EVENT", target: "EVENT", label: "TEMPORAL_AFTER" } },
                { group: "edges", data: { id: "e4", source: "CONCEPT", target: "CONCEPT", label: "PART_OF" } },
                { group: "edges", data: { id: "e5", source: "CONCEPT", target: "CONCEPT", label: "CONTRADICTS" } },
                { group: "edges", data: { id: "e6", source: "CONCEPT", target: "CONCEPT", label: "SUPPORTS" } },
                { group: "edges", data: { id: "e7", source: "CONCEPT", target: "CONCEPT", label: "RELATED_TO" } }
            ];

            const cy = cytoscape({
                container: viewport,
                elements: tboxElements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(id)',
                            'color': '#fff',
                            'font-size': '10px',
                            'font-family': 'monospace',
                            'font-weight': 'bold',
                            'background-color': 'data(color)',
                            'border-width': '2px',
                            'border-color': '#fff',
                            'width': '64px',
                            'height': '64px',
                            'text-valign': 'center',
                            'text-halign': 'center'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'label': 'data(label)',
                            'font-size': '8.5px',
                            'color': '#94a3b8',
                            'width': 2,
                            'line-color': '#475569',
                            'target-arrow-color': '#475569',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'text-background-opacity': 0.85,
                            'text-background-color': '#0b0f19',
                            'text-background-padding': '3px',
                            'text-background-shape': 'roundrectangle'
                        }
                    }
                ],
                layout: {
                    name: 'circle',
                    fit: true,
                    padding: 60
                }
            });

            activeCyInstance = cy;

            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                const id = node.data('id');
                const descs = {
                    "CONCEPT": "Abstract ideas, subjects, tools, or entities active in the workspace.",
                    "PREFERENCE": "User settings, operational guidelines, constraints, and custom personality rules.",
                    "EVENT": "Episodes, historical milestones, turn contexts, or tool execution outcomes.",
                    "DECISION": "Architectural choices, code designs, technical lessons, or executed plan milestones."
                };
                inspector.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 12px; width: 100%;">
                        <span class="level-tag deep" style="font-size: 9.5px; font-weight: bold; text-transform: uppercase; color: var(--accent-color);">Class: ${id}</span>
                        <p style="font-size: 13px; line-height: 1.5; color: var(--text-primary); margin: 0;">${descs[id]}</p>
                    </div>
                `;
            });
            return;
        }

        viewport.innerHTML = `<div class="empty-viewer-msg"><span class="spinner inline" style="margin-right: 8px;"></span> Retrieving complete semantic memory network topology...</div>`;

        try {
            const res = await fetch("/api/memories/graph");
            const data = await res.json();

            if (!data.success || !data.nodes || data.nodes.length === 0) {
                viewport.innerHTML = `<div class="empty-viewer-msg">No memories stored in the semantic graph yet.</div>`;
                return;
            }

            viewport.innerHTML = ""; // Clear loader

            // 1. Prepare Cytoscape elements (nodes and edges)
            const elements = [];
            const nodeIds = new Set();

            // Append memory nodes
            data.nodes.forEach(m => {
                const isWorking = m.level === 1;
                
                // Formulate a clean, human-readable label based on subject/object, or fall back to summary/content
                let label = "";
                if (m.subject && m.object && m.subject !== "unknown" && m.object !== "unknown") {
                    label = `${m.subject.toUpperCase()} ➔ ${m.object.toUpperCase()}`;
                } else {
                    const cleanContent = m.content.replace(/Entity type:.*$/i, "").trim();
                    label = (m.summary || cleanContent).substring(0, 24) + "...";
                }

                elements.push({
                    group: "nodes",
                    data: {
                        id: m.id,
                        label: label,
                        raw: m,
                        is_working: isWorking,
                        level: m.level,
                        importance: m.importance,
                        centrality: m.centrality || 0.0
                    }
                });
                nodeIds.add(m.id);
            });

            // Append explicit relationships (edges)
            data.edges.forEach(e => {
                if (nodeIds.has(e.source_id) && nodeIds.has(e.target_id)) {
                    elements.push({
                        group: "edges",
                        data: {
                            id: e.id,
                            source: e.source_id,
                            target: e.target_id,
                            label: e.relationship_type,
                            raw: e
                        }
                    });
                }
            });

            if (typeof cytoscape === "undefined") {
                viewport.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Cytoscape.js library was not loaded properly. Check your internet connection.</div>`;
                return;
            }

            // 2. Instantiate Cytoscape
            const cy = cytoscape({
                container: viewport,
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'color': '#cbd5e1',
                            'font-size': '11px',
                            'font-family': 'sans-serif',
                            'text-wrap': 'wrap',
                            'text-max-width': '140px',
                            'background-color': '#1e293b',
                            'border-width': '2px',
                            'border-color': '#475569',
                            'width': 'mapData(centrality, 0, 1, 30, 50)',
                            'height': 'mapData(centrality, 0, 1, 30, 50)',
                            'text-valign': 'bottom',
                            'text-margin-y': '6px',
                            'transition-property': 'background-color, border-color, box-shadow',
                            'transition-duration': '0.2s'
                        }
                    },
                    {
                        selector: 'node[level = 1]', // Working/Live Memory - Emerald Green Glow!
                        style: {
                            'background-color': 'rgba(16, 185, 129, 0.15)',
                            'border-color': '#10b981',
                            'color': '#a7f3d0',
                            'font-weight': 'bold',
                            'border-width': '3px'
                        }
                    },
                    {
                        selector: 'node[level = 2]', // Deep Memory - Amber!
                        style: {
                            'background-color': 'rgba(245, 158, 11, 0.1)',
                            'border-color': '#f59e0b',
                            'color': '#fde68a'
                        }
                    },
                    {
                        selector: 'node[level = 3]', // Archived Memory - Faded!
                        style: {
                            'background-color': '#0f172a',
                            'border-color': 'rgba(71, 85, 105, 0.4)',
                            'color': '#64748b'
                        }
                    },
                    {
                        selector: 'node:selected', // Highlight selected node
                        style: {
                            'background-color': 'rgba(79, 70, 229, 0.25)',
                            'border-color': '#818cf8',
                            'border-width': '4px',
                            'color': '#fff'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'label': 'data(label)',
                            'font-size': '8px',
                            'color': '#94a3b8',
                            'width': 1.5,
                            'line-color': '#334155',
                            'target-arrow-color': '#334155',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'text-background-opacity': 0.8,
                            'text-background-color': '#0b0f19',
                            'text-background-padding': '3px',
                            'text-background-shape': 'roundrectangle'
                        }
                    },
                    {
                        selector: 'edge[label = "CONTRADICTS"]',
                        style: {
                            'line-color': '#ef4444',
                            'target-arrow-color': '#ef4444',
                            'color': '#fca5a5'
                        }
                    },
                    {
                        selector: 'edge[label = "SUPPORTS"]',
                        style: {
                            'line-color': '#10b981',
                            'target-arrow-color': '#10b981',
                            'color': '#a7f3d0'
                        }
                    },
                    {
                        selector: 'edge[label = "IMPLEMENTS"]',
                        style: {
                            'line-color': '#3b82f6',
                            'target-arrow-color': '#3b82f6',
                            'color': '#93c5fd'
                        }
                    }
                ],
                layout: {
                    name: 'cose',
                    idealEdgeLength: 100,
                    nodeOverlap: 20,
                    refresh: 20,
                    fit: true,
                    padding: 30,
                    randomize: false,
                    componentSpacing: 100,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                }
            });

            activeCyInstance = cy;

            // 3. Bind click events for Node Inspection
            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                const r = node.data('raw');

                const levelLabels = { 1: "Working (Level 1)", 2: "Deep / Associative (Level 2)", 3: "Archived (Level 3)", 4: "Episodic (Level 4)" };
                const levelClasses = { 1: "working", 2: "deep", 3: "archived", 4: "episodic" };

                let tripleHtml = "";
                if (r.subject && r.object) {
                    tripleHtml = `
                        <div style="margin-top: 8px;">
                            <span style="font-size: 10px; text-transform: uppercase; font-weight: bold; color: var(--text-secondary); display: block; margin-bottom: 4px;">Ontological Triplet</span>
                            <div style="display: inline-flex; align-items: center; gap: 4px; font-family: monospace; font-size: 10.5px; background-color: rgba(79, 70, 229, 0.15); color: #818cf8; border: 1px solid rgba(79, 70, 229, 0.3); padding: 4px 8px; border-radius: 4px; width: 100%;">
                                <strong>${r.subject}</strong> -[${r.predicate}]-> <strong>${r.object}</strong>
                            </div>
                        </div>
                    `;
                }

                let tagsHtml = "";
                if (r.tags) {
                    tagsHtml = `
                        <div style="margin-top: 8px;">
                            <span style="font-size: 10px; text-transform: uppercase; font-weight: bold; color: var(--text-secondary); display: block; margin-bottom: 4px;">Tags</span>
                            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                ${r.tags.split(",").map(t => `<span class="category-tag" style="background-color: rgba(245, 158, 11, 0.15); color: var(--accent-color); padding: 2px 6px; border-radius: 4px; font-size: 9px; font-weight: bold;">#${t.strip ? t.strip() : t.trim()}</span>`).join("")}
                            </div>
                        </div>
                    `;
                }

                inspector.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 12px; width: 100%;">
                        <div>
                            <span class="level-tag ${levelClasses[r.level]}" style="font-size: 9px; font-weight: bold; text-transform: uppercase; color: ${r.level === 1 ? 'var(--success-color)' : r.level === 2 ? 'var(--accent-color)' : '#64748b'};">${levelLabels[r.level]}</span>
                            <span style="float: right; font-size: 10px; font-weight: bold; background-color: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">Imp: ${(r.importance * 100).toFixed(0)}%</span>
                        </div>

                        <div style="display: flex; flex-direction: column; gap: 4px; border-top: 1px solid var(--border-color); padding-top: 10px;">
                            <span style="font-size: 10px; text-transform: uppercase; font-weight: bold; color: var(--text-secondary);">Memory Content</span>
                            <p style="font-size: 12.5px; line-height: 1.5; color: var(--text-primary); margin: 0; white-space: pre-wrap;">${r.content}</p>
                        </div>

                        ${tripleHtml}
                        ${tagsHtml}

                        <div style="border-top: 1px solid var(--border-color); padding-top: 10px; font-size: 10px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 4px;">
                            <span>ID: <code style="font-family: monospace;">${r.id}</code></span>
                            <span>Centrality Index: <strong>${(r.centrality * 100).toFixed(1)}%</strong></span>
                            <span>Activation Score: <strong>${r.activation.toFixed(2)}</strong></span>
                            <span>Usage Count: <strong>${r.use_count} times</strong></span>
                            <span>Created: <strong>${r.created_at.replace("T", " ").substring(0, 19)}</strong></span>
                        </div>
                    </div>
                `;
            });

            // Deselect reset inspector
            cy.on('tap', function(evt) {
                if (evt.target === cy) {
                    inspector.innerHTML = `<div class="empty-viewer-msg" style="justify-content: flex-start; text-align: left; font-style: normal; color: var(--text-secondary);">Click on any memory node in the network to inspect its full semantic details.</div>`;
                }
            });

        } catch (err) {
            console.error("Memory graph rendering failed:", err);
            viewport.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444;">Failed to build semantic network: ${err.message || err}</div>`;
        }
    }

    // ── Settings Sub-Tabs Navigation (Parameters vs Custom Commands) ──
    // Settings Sub-Tabs Navigation (Parameters vs Custom Commands)
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
            document.getElementById(`tti-sub-pane-${btn.dataset.ttiSubTab}`).style.display = "flex";

            if (btn.dataset.ttiSubTab === "commands") {
                populateBindingCommands("tti", selTtiBinding.value);
            } else if (btn.dataset.ttiSubTab === "zoo") {
                populateTtiZoo(selTtiBinding.value);
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

                    let ro_badge = "";
                    let ro_action_icon = "🔓 Make Writable";
                    if (a.read_only) {
                        ro_badge = `<span class="level-tag archived" style="display:inline-block; font-size:9px; font-weight:bold; margin-left:6px; background-color:rgba(239, 68, 68, 0.1); color:#ef4444; border:1px solid rgba(239, 68, 68, 0.2);">🔒 READ-ONLY</span>`;
                        ro_action_icon = "🔓 Make Writable";
                    } else {
                        ro_badge = `<span class="level-tag working" style="display:inline-block; font-size:9px; font-weight:bold; margin-left:6px; background-color:rgba(16, 185, 129, 0.1); color:#10b981; border:1px solid rgba(16, 185, 129, 0.2);">✏️ WRITABLE</span>`;
                        ro_action_icon = "🔒 Make Read-Only";
                    }

                    innerContent = `
                        <div class="artifact-card-header">
                            <span class="title">${typeIcon} ${a.title} ${ro_badge}</span>
                            <div class="artifact-actions">
                                <button class="artifact-action-btn toggle-ro" data-title="${a.title}" title="${ro_action_icon}">${a.read_only ? '🔒' : '✏️'}</button>
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

            document.querySelectorAll(".artifact-action-btn.toggle-ro").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const title = btn.dataset.title;
                    try {
                        const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}/toggle_read_only`, { method: "POST" });
                        const data = await res.json();
                        if (data.success) {
                            fetchArtifacts();
                            const safeId = makeSafeId(title);
                            const tabContent = document.getElementById(`tab-art-${safeId}`);
                            if (tabContent) {
                                await selectArtifact(title);
                            }
                        }
                    } catch (err) {
                        console.error("Failed to toggle read-only:", err);
                    }
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

                    // Check if this is a data artifact with physical tables
                    let deletePhysical = false;
                    try {
                        const artRes = await fetch("/api/artifacts");
                        const arts = await artRes.json();
                        const matched = arts.find(a => a.title === title);
                        if (matched && matched.type === "data") {
                            // Prompt user for physical data deletion
                            const wantPhysicalDelete = await showSlickConfirm(
                                "🗑️ Delete Artifact Data",
                                `This is a DATA artifact. Do you also want to delete the underlying database tables from the consolidated file?\n\n⚠️ This action cannot be undone!`,
                                "Delete Physical Tables",
                                true
                            );
                            deletePhysical = wantPhysicalDelete;
                        }

                        const confirmed = await showSlickConfirm(
                            "🗑️ Delete Artifact",
                            `Are you sure you want to permanently delete the artifact '${title}'? This will also remove all its version history.${deletePhysical ? '\n\n✓ Physical database tables will be dropped.' : ''}`
                        );

                        if (!confirmed) return;

                        const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}`, { 
                            method: "DELETE",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ delete_physical_data: deletePhysical })
                        });

                        const data = await res.json();
                        if (data.success) {
                            closeArtifactTab(title);
                            fetchArtifacts();
                            updateContextBudget();

                            if (deletePhysical && data.tables_dropped) {
                                showNotification(`✓ Deleted artifact and dropped ${data.tables_dropped} physical database table(s)`, "success", 4000);
                            }
                        }
                    } catch (err) {
                        console.error("Failed to delete artifact:", err);
                        alert(`Delete failed: ${err}`);
                    }
                });
            });


            // ── Orphaned Table Cleanup Button (added to data artifacts tab) ──
            const originalCreateArtifactTab = window.createArtifactTab;
            if (!window._patchedCreateArtifactTab) {
                window.createArtifactTab = function(title, type) {
                    originalCreateArtifactTab(title, type);

                    // Add cleanup button for data artifact tabs
                    if (type === "data") {
                        const safeId = makeSafeId(title);
                        const tabContent = document.getElementById(`tab-art-${safeId}`);
                        if (tabContent) {
                            const vSelect = tabContent.querySelector(".version-select");
                            if (vSelect) {
                                const cleanupBtn = document.createElement("button");
                                cleanupBtn.className = "btn-version-action btn-cleanup-orphaned";
                                cleanupBtn.dataset.artTitle = title;
                                cleanupBtn.title = "🧹 Remove orphaned database tables from consolidated file";
                                cleanupBtn.textContent = "🧹 Cleanup Orphans";
                                cleanupBtn.style.cssText = "background: none; border: 1px solid var(--border-color); color: #f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: bold; transition: all 0.15s; height: auto; width: auto; line-height: 1.4;";

                                vSelect.parentNode.appendChild(cleanupBtn);

                                cleanupBtn.addEventListener("click", async () => {
                                    const confirmed = await showSlickConfirm(
                                        "🧹 Cleanup Orphaned Tables",
                                        `Scan the consolidated database for tables that no longer have corresponding artifacts and remove them?`,
                                        "Cleanup Now"
                                    );

                                    if (!confirmed) return;

                                    cleanupBtn.disabled = true;
                                    cleanupBtn.textContent = "Scanning...";

                                    try {
                                        const res = await fetch(`/api/artifacts/${encodeURIComponent(title)}/cleanup_orphaned_tables`, {
                                            method: "POST"
                                        });
                                        const data = await res.json();

                                        if (data.success) {
                                            alert(`✓ Cleanup complete!\n\nOrphaned tables found: ${data.orphaned_tables_found}\nTables deleted: ${data.tables_deleted}`);
                                        } else {
                                            alert(`Cleanup failed: ${data.detail || data.error}`);
                                        }
                                    } catch (err) {
                                        alert(`Request failed: ${err}`);
                                    } finally {
                                        cleanupBtn.disabled = false;
                                        cleanupBtn.textContent = "🧹 Cleanup Orphans";
                                    }
                                });
                            }
                        }
                    }
                };

                window._patchedCreateArtifactTab = true;
            }
        } catch (err) {
            console.error("Failed to load artifacts:", err);
        }
    }

    // ── 💬 Fetch, Render, and Manage State-rich Message History ──
    async function fetchMessageHistory() {
        try {
            const res = await fetch("/api/discussions/viewer_session/messages");
            const messages = await res.json();

            // Check and preserve active thinking indicator state prior to innerHTML wipe
            const indicator = document.getElementById("thinking-indicator");
            const wasActive = indicator && indicator.classList.contains("active");
            const activeText = indicator && indicator.querySelector(".thinking-text") ? indicator.querySelector(".thinking-text").textContent : "Lollms is thinking...";

            chatHistory.innerHTML = ""; // Clear
            if (messages.length === 0) {
                // Keep chat completely empty per user requirements
                return;
            }

            messages.forEach(msg => {
                try {
                    const sender = msg.sender_type === "user" ? "user" : "assistant";
                    renderMessageBubble(msg.id, sender, msg.content, {
                        model_name: msg.model_name,
                        tokens: msg.tokens,
                        speed: msg.generation_speed,
                        ttft: msg.metadata ? msg.metadata.ttft : null
                    }, msg);
                } catch (renderErr) {
                    console.error("Failed to render message:", msg.id, renderErr);
                    // Fallback to avoid blank screen
                    appendChatBubble(msg.sender_type, msg.content || "");
                }
            });
            if (wasActive) {
                showThinkingIndicator(activeText);
            }
            chatHistory.scrollTop = chatHistory.scrollHeight;
        } catch (err) {
            console.error("Failed to fetch message history:", err);
        }
    }

    async function cycleSibling(messageId, direction) {
        try {
            const res = await fetch(`/api/discussions/viewer_session/messages/${messageId}/siblings/cycle?direction=${direction}`, {
                method: "POST"
            });
            const data = await res.json();
            if (data.success) {
                await fetchMessageHistory();
            }
        } catch (err) {
            console.error("Failed to cycle sibling:", err);
        }
    }

    function runMarkdownCleanup(txt) {
        if (!txt) return "";
        return txt.replace(/\\`{1,3}/g, m => m.replace("\\", ""))
                  .replace(/\\\*/g, "*")
                  .replace(/\\_/g, "_");
    }

    // ── 📊 Frontend In-Memory Dataset Cache ──
    const datasetCache = new Map(); // Cache key: "title-version" -> JSON data

    function renderTabbedSheets(data, docTitle, containerElement) {
        const sheetNames = Object.keys(data.sheets);
        const safeTitle = makeSafeId(docTitle);

        const tabsHtml = `
            <!-- Data Intelligence Controls Panel -->
            <div class="data-intelligence-panel" style="display: flex; flex-direction: column; gap: 10px; background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px; margin-bottom: 16px; flex-shrink: 0; box-shadow: var(--shadow-sm); user-select: none;">
                <div style="display: flex; gap: 10px; align-items: center; width: 100%;">
                    <!-- 1. Text Search Bar -->
                    <div style="flex: 1; position: relative;">
                        <input type="text" class="data-local-search" id="search-local-${safeTitle}" placeholder="🔍 Live filter visible rows..." style="width: 100%; padding: 8px 12px; font-size: 12px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 6px; outline: none; transition: border-color 0.15s;" />
                    </div>
                    <!-- Toggle buttons -->
                    <button class="btn btn-secondary toggle-sql-btn" id="btn-toggle-sql-${safeTitle}" style="width: auto; padding: 8px 14px; font-size: 11.5px; font-weight: bold; border-radius: 6px; background-color: var(--border-color); color: var(--text-primary);">💻 Raw SQL Sandbox</button>
                    <button class="btn btn-primary toggle-ai-btn" id="btn-toggle-ai-${safeTitle}" style="width: auto; padding: 8px 14px; font-size: 11.5px; font-weight: bold; border-radius: 6px; background-color: var(--chat-user-bg); color: white; border-color: var(--chat-user-bg);">🤖 Ask AI Assistant</button>
                </div>

                <!-- 2. Hidden SQL Sandbox Panel -->
                <div id="sql-panel-${safeTitle}" style="display: none; flex-direction: column; gap: 8px; border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 4px;">
                    <label style="font-size: 11px; font-weight: bold; color: var(--text-secondary); text-transform: uppercase;">SQLite SQL Query</label>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" id="sql-query-${safeTitle}" placeholder="SELECT * FROM Customers WHERE country = 'France' LIMIT 10" style="flex: 1; padding: 8px 12px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 6px; outline: none; font-family: monospace; font-size: 12px;" />
                        <button class="btn btn-primary" id="btn-run-sql-${safeTitle}" style="width: auto; padding: 8px 16px; font-size: 12px; font-weight: bold;">Run SQL</button>
                    </div>
                </div>

                <!-- 3. Hidden AI Query Panel -->
                <div id="ai-panel-${safeTitle}" style="display: none; flex-direction: column; gap: 8px; border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 4px;">
                    <label style="font-size: 11px; font-weight: bold; color: var(--text-secondary); text-transform: uppercase;">🤖 Ask AI Natural Language Question</label>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" id="ai-question-${safeTitle}" placeholder="e.g. How many Gold members signup in 2024?" style="flex: 1; padding: 8px 12px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 6px; outline: none; font-size: 12.5px;" />
                        <input type="submit" class="btn btn-primary" id="btn-run-ai-${safeTitle}" style="width: auto; padding: 8px 16px; font-size: 12px; font-weight: bold; background-color: var(--chat-user-bg); color: white; border-color: var(--chat-user-bg);" value="Ask AI" />
                    </div>
                </div>
            </div>

            <!-- Sheets Tab bar -->
            <div class="sheet-tabs-container" style="margin-bottom: 12px; user-select: none;">
                ${sheetNames.map((s, idx) => `
                    <button class="sheet-tab msg-sheet-tab-${safeTitle} ${idx === 0 ? 'active' : ''}" data-sheet="${s}">${s}</button>
                `).join("")}
            </div>
            <div class="data-grid-wrapper" id="spreadsheet-grid-target-${safeTitle}" style="max-height: 500px; overflow-y: auto; overflow-x: auto; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        `;
        containerElement.innerHTML = tabsHtml;

        // Wire up Local Live Filter
        const localSearch = containerElement.querySelector(`#search-local-${safeTitle}`);
        localSearch.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            const rows = containerElement.querySelectorAll(`#spreadsheet-grid-target-${safeTitle} tbody tr`);
            rows.forEach(tr => {
                const text = tr.textContent.toLowerCase();
                tr.style.display = text.includes(query) ? "" : "none";
            });
        });

        // Wire up SQL Toggle
        const sqlBtn = containerElement.querySelector(`#btn-toggle-sql-${safeTitle}`);
        const sqlPanel = containerElement.querySelector(`#sql-panel-${safeTitle}`);
        const aiBtn = containerElement.querySelector(`#btn-toggle-ai-${safeTitle}`);
        const aiPanel = containerElement.querySelector(`#ai-panel-${safeTitle}`);

        sqlBtn.addEventListener("click", () => {
            const isHidden = sqlPanel.style.display === "none";
            sqlPanel.style.display = isHidden ? "flex" : "none";
            sqlBtn.classList.toggle("active", isHidden);
            if (isHidden) {
                // Close AI panel
                aiPanel.style.display = "none";
                aiBtn.classList.remove("active");
            }
        });

        // Wire up AI Toggle
        aiBtn.addEventListener("click", () => {
            const isHidden = aiPanel.style.display === "none";
            aiPanel.style.display = isHidden ? "flex" : "none";
            aiBtn.classList.toggle("active", isHidden);
            if (isHidden) {
                // Close SQL panel
                sqlPanel.style.display = "none";
                sqlBtn.classList.remove("active");
            }
        });

        // Wire up Run SQL click
        const runSqlBtn = containerElement.querySelector(`#btn-run-sql-${safeTitle}`);
        const sqlInput = containerElement.querySelector(`#sql-query-${safeTitle}`);
        runSqlBtn.addEventListener("click", async () => {
            const query = sqlInput.value.trim();
            if (!query) return;

            runSqlBtn.disabled = true;
            runSqlBtn.textContent = "Querying...";
            const gridTarget = containerElement.querySelector(`#spreadsheet-grid-target-${safeTitle}`);
            gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 20px;"><span class="spinner inline" style="margin-right: 8px;"></span> Compiling SQL query inside SQLite sandbox...</div>`;

            try {
                const res = await fetch(`/api/data/${encodeURIComponent(docTitle)}/raw_query`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sql_query: query })
                });
                const resData = await res.json();
                if (resData.success) {
                    drawQueryResultsInTable(resData, docTitle, gridTarget);
                } else {
                    gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 20px;">❌ SQL Query Error:<br/>${resData.error}</div>`;
                }
            } catch (err) {
                gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 20px;">Request failed: ${err}</div>`;
            } finally {
                runSqlBtn.disabled = false;
                runSqlBtn.textContent = "Run SQL";
            }
        });

        // Wire up Run AI click
        const runAiBtn = containerElement.querySelector(`#btn-run-ai-${safeTitle}`);
        const aiInput = containerElement.querySelector(`#ai-question-${safeTitle}`);
        runAiBtn.addEventListener("click", async () => {
            const question = aiInput.value.trim();
            if (!question) return;

            runAiBtn.disabled = true;
            runAiBtn.textContent = "AI Translating...";
            const gridTarget = containerElement.querySelector(`#spreadsheet-grid-target-${safeTitle}`);
            gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 20px;"><span class="spinner inline" style="margin-right: 8px;"></span> 🤖 AI Ingesting Schema & Formulating SQL...</div>`;

            try {
                const res = await fetch(`/api/data/${encodeURIComponent(docTitle)}/ai_query`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: question })
                });
                const resData = await res.json();
                if (resData.success) {
                    drawQueryResultsInTable(resData, docTitle, gridTarget, resData.sql_query, resData.explanation);
                } else {
                    gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 20px;">❌ AI Query Failed:<br/>${resData.error}</div>`;
                }
            } catch (err) {
                gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 20px;">Request failed: ${err}</div>`;
            } finally {
                runAiBtn.disabled = false;
                runAiBtn.value = "Ask AI";
            }
        });

        // Wire up sheet tabs
        containerElement.querySelectorAll(`.msg-sheet-tab-${safeTitle}`).forEach(tab => {
            tab.addEventListener("click", () => {
                containerElement.querySelectorAll(`.msg-sheet-tab-${safeTitle}`).forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                drawTableInTab(data.sheets[tab.dataset.sheet], docTitle);
            });
        });

        drawTableInTab(data.sheets[sheetNames[0]], docTitle);
    }

    function drawQueryResultsInTable(resData, docTitle, gridTarget, generatedSql = null, explanation = null) {
        const columns = resData.columns;
        const rows = resData.rows;

        if (columns.length === 0 || rows.length === 0) {
            gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 20px;">Query returned 0 results.</div>`;
            return;
        }

        let headerHtml = "";
        if (generatedSql) {
            headerHtml = `
                <div style="background-color: #020617; border-bottom: 1px solid var(--border-color); padding: 12px 14px; font-family: monospace; font-size: 11.5px; color: var(--accent-color);">
                    <strong>Generated SQLite Query:</strong><br/>
                    <pre style="margin-top: 6px; color: #fde68a; font-family: inherit; white-space: pre-wrap;">${generatedSql}</pre>
                    ${explanation ? `<div style="margin-top: 8px; color: var(--text-secondary); font-family: sans-serif; font-size: 11px;"><strong>Insight:</strong> ${explanation}</div>` : ""}
                </div>
            `;
        }

        gridTarget.innerHTML = `
            ${headerHtml}
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
    }

    function renderFlatTable(data, docTitle, containerElement) {
        // Flat CSV can be treated as a single table. To support SQL and AI querying, we can wrap its data
        // inside the exact same tabbed sheets structure (with "Data" as the single sheet name) so the entire
        // Data Intelligence Panel is fully functional for CSV files as well!
        const parsedData = {
            type: "excel",
            sheets: {
                "Data": data
            }
        };
        renderTabbedSheets(parsedData, docTitle, containerElement);
    }

    function updateVersionActionButtonsState(tabContent, title, selectedVer) {
        const vSelect = tabContent.querySelector(".version-select");
        const btnSetActive = tabContent.querySelector(".btn-set-active-version");
        const btnDeleteVer = tabContent.querySelector(".btn-delete-version");
        const btnSquash = tabContent.querySelector(".btn-squash-versions");
        if (!vSelect || !btnSetActive) return;
        const activeVer = parseInt(vSelect.dataset.activeVersion, 10);

        if (btnDeleteVer) btnDeleteVer.style.display = "inline-block";
        if (btnSquash) btnSquash.style.display = "inline-block";
        if (selectedVer === activeVer) {
            btnSetActive.disabled = true;
            btnSetActive.textContent = "⭐ Active";
            btnSetActive.style.opacity = "0.5";
            btnSetActive.style.cursor = "default";
        } else {
            btnSetActive.disabled = false;
            btnSetActive.textContent = "⭐ Set Active";
            btnSetActive.style.opacity = "1.0";
            btnSetActive.style.cursor = "pointer";
        }
    }

    function formatMessageContent(text, msgId, msgObj = null) {
        const cleanedText = runMarkdownCleanup(text);
        const processedText = resolveProcessingTags(cleanedText);

        const placeholders = {};
        let maskedText = processedText;

        const inlinePattern = /<lollms_inline\s*([^>]*)>([\s\S]*?)<\/lollms_inline>/gi;
        maskedText = maskedText.replace(inlinePattern, (match) => {
            const id = `__WIDGET_PLACEHOLDER_${Math.random().toString(36).substr(2, 9)}__`;
            placeholders[id] = match;
            return id;
        });

        const formPattern = /<lollms_form\s*([^>]*)>([\s\S]*?)<\/lollms_form>/gi;
        maskedText = maskedText.replace(formPattern, (match) => {
            const id = `__FORM_PLACEHOLDER_${Math.random().toString(36).substr(2, 9)}__`;
            placeholders[id] = match;
            return id;
        });

        const parsedMarkdown = marked.parse(maskedText);

        let restoredHTML = parsedMarkdown;
        for (const [id, original] of Object.entries(placeholders)) {
            restoredHTML = restoredHTML.split(id).join(original);
        }

        return resolveImageAnchors(restoredHTML, activeArtifactTitle, msgId, null, msgObj);
    }

    function renderMessageBubble(msgId, sender, text, metrics = {}, msgObj = null) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;
        bubble.dataset.msgId = msgId;
        if (msgObj) {
            bubble.dataset.msgObj = JSON.stringify(msgObj);
        }

        const contentDiv = document.createElement("div");
        contentDiv.className = "chat-assistant-container";

        const proseSpan = document.createElement("span");
        proseSpan.className = "prose-span";

        proseSpan.innerHTML = formatMessageContent(text, msgId, msgObj);
        contentDiv.appendChild(proseSpan);

        // Render any active message images (such as sandboxed Matplotlib plots) at the bottom of the bubble
        if (msgObj && msgObj.images && msgObj.images.length > 0) {
            const imagesContainer = document.createElement("div");
            imagesContainer.className = "msg-images-gallery";
            imagesContainer.style = "display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;";

            // Collect all image IDs referenced in the text to avoid duplicate rendering
            const imgPattern = /<artefact_image\s+id=["']([^"']+)["']/gi;
            const referencedIds = [];
            let match;
            while ((match = imgPattern.exec(text)) !== null) {
                referencedIds.push(match[1]);
            }

            const activeFlags = msgObj.active_images || [];
            msgObj.images.forEach((imgB64, idx) => {
                const isActive = idx < activeFlags.length ? activeFlags[idx] : true;
                if (!isActive) return;

                // Check if this image was already rendered inline via artefact_image tag
                const isReferenced = referencedIds.some(refId => {
                    if (refId.includes("::")) {
                        const parts = refId.split("::");
                        const refIdx = parseInt(parts[parts.length - 1], 10);
                        return refIdx === idx;
                    }
                    return false;
                });

                if (isReferenced) return;

                const imgContainer = document.createElement("div");
                imgContainer.className = "rendered-image-container";
                imgContainer.style = "position: relative; max-width: 100%; margin: 10px 0;";
                imgContainer.innerHTML = `
                    <img src="data:image/png;base64,${imgB64}" class="rendered-page-img" style="display: block; max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-color); box-shadow: var(--shadow-lg);" />
                `;
                imagesContainer.appendChild(imgContainer);
            });

            if (imagesContainer.children.length > 0) {
                contentDiv.appendChild(imagesContainer);
            }
        }

        // Render any dynamic workspace dataset views (ui_data_views) at the bottom of the chat bubble
        if (msgObj && msgObj.metadata && msgObj.metadata.ui_data_views && msgObj.metadata.ui_data_views.length > 0) {
            const views = msgObj.metadata.ui_data_views;
            const viewsContainer = document.createElement("div");
            viewsContainer.className = "msg-data-views-container";
            viewsContainer.style = "margin-top: 14px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; background-color: var(--bg-panel); display: flex; flex-direction: column; width: 100%; box-shadow: var(--shadow-md);";

            const safeMsgId = makeSafeId(msgId);

            viewsContainer.innerHTML = `
                <div class="sheet-tabs-container" style="background-color: #151f30; padding: 6px 10px; display: flex; gap: 6px; overflow-x: auto; border-bottom: 1px solid var(--border-color);">
                    ${views.map((f, idx) => `
                        <button class="sheet-tab msg-sheet-tab-${safeMsgId} ${idx === 0 ? 'active' : ''}" data-file="${f}" style="padding: 4px 10px; font-size: 11.5px; border-radius: 4px; font-weight: bold; background: none; border: 1px solid transparent; color: var(--text-secondary); cursor: pointer; transition: all 0.15s; white-space: nowrap;">📊 ${f}</button>
                    `).join("")}
                </div>
                <div class="data-grid-wrapper" id="msg-grid-target-${safeMsgId}" style="max-height: 280px; overflow-y: auto; overflow-x: auto; background-color: var(--bg-panel);"></div>
            `;

            contentDiv.appendChild(viewsContainer);

            async function renderMsgGrid(filename) {
                const gridTarget = viewsContainer.querySelector(`#msg-grid-target-${safeMsgId}`);
                if (!gridTarget) return;

                gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 15px;"><span class="spinner inline" style="margin-right: 8px;"></span> Loading dataset preview...</div>`;
                try {
                    const res = await fetch(`/api/workspace_files/${encodeURIComponent(filename)}`);
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    const csvText = await res.text();

                    if (typeof Papa === "undefined") {
                        gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 15px; color: #ef4444;">Failed to render: PapaParse library missing.</div>`;
                        return;
                    }

                    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
                    const columns = parsed.meta.fields || [];
                    const rows = parsed.data || [];

                    if (columns.length === 0 || rows.length === 0) {
                        gridTarget.innerHTML = `<div class="empty-viewer-msg" style="padding: 15px;">Dataset is empty.</div>`;
                        return;
                    }

                    gridTarget.innerHTML = `
                        <table class="data-table" style="font-size: 11.5px; width: 100%; border-collapse: collapse; text-align: left;">
                            <thead>
                                <tr style="background-color: #020617; color: var(--accent-color); position: sticky; top: 0; z-index: 10;">
                                    ${columns.map(c => `<th style="padding: 8px 12px; border-bottom: 2px solid var(--border-color); font-weight: bold;">${c}</th>`).join("")}
                                </tr>
                            </thead>
                            <tbody>
                                ${rows.map(r => `
                                    <tr style="border-bottom: 1px solid var(--border-color);">
                                        ${columns.map(c => `<td style="padding: 6px 12px;">${r[c] !== undefined ? r[c] : ''}</td>`).join("")}
                                    </tr>
                                `).join("")}
                            </tbody>
                        </table>
                    `;
                } catch(err) {
                    gridTarget.innerHTML = `<div class="empty-viewer-msg" style="color: #ef4444; padding: 15px;">Failed to load dataset: ${err.message || err}</div>`;
                }
            }

            viewsContainer.querySelectorAll(`.msg-sheet-tab-${safeMsgId}`).forEach(tab => {
                tab.addEventListener("click", () => {
                    viewsContainer.querySelectorAll(`.msg-sheet-tab-${safeMsgId}`).forEach(t => t.classList.remove("active"));
                    tab.classList.add("active");
                    renderMsgGrid(tab.dataset.file);
                });
            });

            renderMsgGrid(views[0]);
        }

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

        if (msgObj && msgObj.siblings && msgObj.siblings.length > 1) {
            const siblingPager = document.createElement("div");
            siblingPager.className = "sibling-pager";
            siblingPager.style = "display: inline-flex; align-items: center; gap: 8px; font-size: 11px; font-weight: bold; color: var(--text-secondary); margin-left: auto; user-select: none;";
            siblingPager.innerHTML = `
                <button class="sibling-page-btn prev-sib" style="background: none; border: none; color: var(--accent-color); cursor: pointer; font-size: 12px; padding: 2px 4px;">◀</button>
                <span>${msgObj.active_sibling_index + 1} / ${msgObj.siblings.length}</span>
                <button class="sibling-page-btn next-sib" style="background: none; border: none; color: var(--accent-color); cursor: pointer; font-size: 12px; padding: 2px 4px;">▶</button>
            `;

            siblingPager.querySelector(".prev-sib").addEventListener("click", () => {
                cycleSibling(msgId, -1);
            });
            siblingPager.querySelector(".next-sib").addEventListener("click", () => {
                cycleSibling(msgId, 1);
            });

            actionsRow.appendChild(siblingPager);
        }

        contentDiv.appendChild(actionsRow);
        chatHistory.appendChild(bubble);

        // Bind Action Clicks
        bindMessageActions(bubble);
        renderMath(proseSpan);
        attachDataQueryInterceptors(contentDiv);
    }

    async function bindMessageActions(bubbleElement) {
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

        // B. Delete Message (Recursive Subtree Pruning Protocol)
        bubbleElement.querySelector(".msg-action-btn.delete").addEventListener("click", async () => {
            const isUser = bubbleElement.classList.contains("user");
            const nodeLabel = isUser ? "User Message and subsequent replies" : "Assistant Message and subsequent conversation";
            const confirmed = await showSlickConfirm(
                "✂️ Prune Conversation Branch",
                `Are you sure you want to PRUNE this branch? This permanently deletes this ${nodeLabel} on this path.`
            );
            if (!confirmed) return;
            try {
                const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}?prune=true`, { method: "DELETE" });
                const data = await res.json();
                if (data.success) {
                    await fetchMessageHistory();
                }
            } catch (err) { alert("Pruning failed: " + err); }
        });

        // C. Resend / Regenerate (Branching Resend/Regenerate protocol)
        bubbleElement.querySelector(".msg-action-btn.resend").addEventListener("click", async () => {
            const msgObj = bubbleElement.dataset.msgObj ? JSON.parse(bubbleElement.dataset.msgObj) : null;
            const isUser = bubbleElement.classList.contains("user");

            if (isUser) {
                // User message: Resend forks a new branch instantly without prompting for text input
                const originalText = proseSpan.textContent.trim();
                const parentId = msgObj ? (msgObj.parent_id || "null") : "null";
                try {
                    const res = await fetch(`/api/discussions/viewer_session/messages/${parentId}/fork`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ 
                            initial_content: originalText,
                            images: msgObj && msgObj.images ? msgObj.images : []
                        })
                    });
                    const data = await res.json();
                    if (data.success) {
                        await fetchMessageHistory();
                        sendChatMessage(true); // run completion turn on new branch
                    }
                } catch (err) {
                    alert("Resend failed: " + err);
                }
            } else {
                // Assistant message: Call the dedicated regenerate endpoint to cleanly delete the bad message and rewind the active branch
                try {
                    const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}/regenerate`, {
                        method: "POST"
                    });
                    const data = await res.json();
                    if (data.success) {
                        await fetchMessageHistory();
                        sendChatMessage(true); // run completion turn to generate a new sibling assistant reply
                    } else {
                        alert(`Regeneration failed: ${data.detail || data.error}`);
                    }
                } catch (err) {
                    alert("Regeneration failed: " + err);
                }
            }
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
            createCard.addEventListener("click", async () => {
                const name = await showCustomPrompt("Create Workspace", "Enter workspace name", "Bazinga");
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
        const confirmed = await showSlickConfirm(
            "🗑️ Delete Workspace",
            `Are you sure you want to permanently delete the workspace '${name.toUpperCase()}' and ALL of its associated discussions, memories, and artifacts?`
        );
        if (!confirmed) return;
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

    // ── 🧠 Dynamic Thinking Indicator Helpers ──
    function showThinkingIndicator(statusText = "Lollms is thinking...") {
        let indicator = document.getElementById("thinking-indicator");
        if (!indicator) {
            indicator = document.createElement("div");
            indicator.id = "thinking-indicator";
            indicator.className = "typing-indicator";
            indicator.innerHTML = `
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="thinking-text" style="font-size: 11.5px; font-weight: bold; color: var(--text-secondary); margin-left: 4px;">${statusText}</span>
            `;
        } else {
            const textEl = indicator.querySelector(".thinking-text");
            if (textEl) textEl.textContent = statusText;
        }
        chatHistory.appendChild(indicator);
        indicator.classList.add("active");
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function hideThinkingIndicator() {
        const indicator = document.getElementById("thinking-indicator");
        if (indicator) {
            indicator.remove();
        }
    }

    // ── Local Memories Tabbed Viewport Variables ──
    let activeMemoryLevel = 1;
    let cachedMemories = [];

    // Bind Memories Local Tabs Click Events
    document.querySelectorAll(".mem-tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            activeMemoryLevel = parseInt(btn.dataset.memLevel, 10);
            renderMemories();
        });
    });

    // ── Fetch and Render Active Branches ──
    async function fetchBranches() {
        if (!branchSelect) return;
        try {
            const res = await fetch("/api/discussions/viewer_session/branches");
            const branches = await res.json();

            if (branches.length === 0) {
                branchSelect.innerHTML = `<option value="">No branches found</option>`;
                return;
            }

            branchSelect.innerHTML = branches.map(b => {
                const isSelected = b.is_active ? "selected" : "";
                return `<option value="${b.leaf_id}" ${isSelected}>${b.label}</option>`;
            }).join("");
        } catch (err) {
            console.error("Failed to fetch branches:", err);
        }
    }

    if (branchSelect) {
        branchSelect.addEventListener("change", async () => {
            const leafId = branchSelect.value;
            if (!leafId) return;
            try {
                const res = await fetch("/api/discussions/viewer_session/branches/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ leaf_id: leafId })
                });
                const data = await res.json();
                if (data.success) {
                    await fetchMessageHistory();
                    await fetchBranches();
                }
            } catch (err) {
                alert("Failed to switch branch: " + err);
            }
        });
    }

    // ── 🧠 Fetch and Render Persistent Memories ──
    async function fetchMemories() {
        try {
            const res = await fetch("/api/memories");
            cachedMemories = await res.json();
            renderMemories();
        } catch (err) {
            console.error("Failed to fetch memories:", err);
        }
    }

    function renderMemories() {
        const memTabs = document.querySelector(".memory-tabs");
        const memTabActions = document.querySelector(".memory-tab-actions");

        if (memViewMode === "tbox") {
            if (memTabs) memTabs.style.display = "none";
            if (memTabActions) memTabActions.style.display = "none";

            // Render TBox Schema
            memoriesList.innerHTML = `
                <div class="tbox-schema-container" style="display: flex; flex-direction: column; gap: 12px; padding: 4px; user-select: none;">
                    <div style="font-size: 11px; font-weight: bold; color: var(--accent-color); text-transform: uppercase; border-bottom: 1px solid var(--border-color); padding-bottom: 4px;">📐 Ontological TBox Classes</div>
                    <div class="tbox-class-card" style="background-color: rgba(79, 70, 229, 0.1); border: 1px solid rgba(79, 70, 229, 0.25); padding: 10px; border-radius: 6px;">
                        <span style="font-weight: bold; font-size: 12px; color: #818cf8;">🏷️ CONCEPT</span>
                        <p style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; line-height: 1.3;">Abstract ideas, subjects, tools, or entities active in the workspace.</p>
                    </div>
                    <div class="tbox-class-card" style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); padding: 10px; border-radius: 6px;">
                        <span style="font-weight: bold; font-size: 12px; color: var(--success-color);">⚙️ PREFERENCE</span>
                        <p style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; line-height: 1.3;">User settings, operational guidelines, constraints, and custom personality rules.</p>
                    </div>
                    <div class="tbox-class-card" style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); padding: 10px; border-radius: 6px;">
                        <span style="font-weight: bold; font-size: 12px; color: var(--accent-color);">📅 EVENT</span>
                        <p style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; line-height: 1.3;">Episodes, historical milestones, turn contexts, or tool execution outcomes.</p>
                    </div>
                    <div class="tbox-class-card" style="background-color: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); padding: 10px; border-radius: 6px;">
                        <span style="font-weight: bold; font-size: 12px; color: #f43f5e;">💡 DECISION</span>
                        <p style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; line-height: 1.3;">Architectural choices, code designs, technical lessons, or executed plan milestones.</p>
                    </div>

                    <div style="font-size: 11px; font-weight: bold; color: var(--accent-color); text-transform: uppercase; border-bottom: 1px solid var(--border-color); padding-bottom: 4px; margin-top: 8px;">🔗 Semantic Relationships</div>
                    <ul style="list-style: none; display: flex; flex-direction: column; gap: 6px; padding-left: 4px; font-size: 11px; line-height: 1.4;">
                        <li><strong>RELATED_TO</strong>: Default associative link</li>
                        <li><strong>PREFERS</strong>: Preference binding</li>
                        <li><strong>IMPLEMENTS</strong>: Concrete realization of choices</li>
                        <li><strong>CONTRADICTS</strong>: Semantic conflicts and blocks</li>
                        <li><strong>SUPPORTS</strong>: Validating evidence</li>
                        <li><strong>TEMPORAL_AFTER</strong>: Chronological ordering</li>
                        <li><strong>PART_OF</strong>: Component hierarchy decomposition</li>
                    </ul>
                </div>
            `;
            return;
        }

        // ABox View
        if (memTabs) memTabs.style.display = "flex";
        if (memTabActions) memTabActions.style.display = "flex";

        // Synchronize active local tab button highlight
        document.querySelectorAll(".mem-tab-btn").forEach(btn => {
            const lvl = parseInt(btn.dataset.memLevel, 10);
            btn.classList.toggle("active", lvl === activeMemoryLevel);
        });

        if (cachedMemories.length === 0) {
            memoriesList.innerHTML = `<li class="empty-msg">No memories stored yet.</li>`;
            return;
        }

        const filtered = cachedMemories.filter(m => m.level === activeMemoryLevel);

        if (filtered.length === 0) {
            const labels = { 1: "Working", 2: "Deep / Associative", 3: "Archived", 4: "Episodic" };
            memoriesList.innerHTML = `<li class="empty-msg">No ${labels[activeMemoryLevel]} memories in session.</li>`;
            return;
        }

        const labels = { 1: "Working", 2: "Deep", 3: "Archived", 4: "Episodic" };
        const classes = { 1: "working", 2: "deep", 3: "archived", 4: "episodic" };

        memoriesList.innerHTML = filtered.map(m => {
            let tripleHtml = "";
            if (m.subject && m.object) {
                // High-fidelity structured ABox triple display
                tripleHtml = `
                    <div class="memory-triple-badge" style="display: flex; flex-direction: column; gap: 4px; font-family: monospace; font-size: 10px; background-color: rgba(15, 23, 42, 0.5); border: 1px solid var(--border-color); padding: 6px; border-radius: 6px; margin-top: 8px; width: 100%;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #94a3b8; font-weight: bold; background-color: rgba(255,255,255,0.05); padding: 1px 4px; border-radius: 3px;">Sub: ${m.subject}</span>
                            <span style="color: var(--accent-color); font-weight: 800; font-size: 8.5px;">[${m.predicate || 'RELATED_TO'}]</span>
                        </div>
                        <div style="color: #cbd5e1; font-weight: bold; padding-left: 4px;">Obj: ${m.object}</div>
                    </div>
                `;
            }

            let activationHtml = "";
            if (m.activation !== undefined && m.activation !== null) {
                const actVal = parseFloat(m.activation);
                const actPct = Math.min(100, Math.max(0, (actVal + 5) * 10)); // Map -5..5 range to 0..100%
                let actClass = "neutral";
                if (actVal > 2.0) {
                    actClass = "glowing";
                } else if (actVal < -1.0) {
                    actClass = "faded";
                }
                activationHtml = `
                    <div class="memory-activation-container" style="display: flex; align-items: center; gap: 8px; margin-top: 6px; user-select: none;">
                        <span style="font-size: 9px; font-weight: bold; color: var(--text-secondary); text-transform: uppercase;">Activation (${actVal.toFixed(1)})</span>
                        <div style="flex: 1; height: 4px; background-color: #020617; border-radius: 2px; overflow: hidden; border: 1px solid var(--border-color); display: flex;">
                            <div class="activation-bar ${actClass}" style="width: ${actPct}%; height: 100%; transition: width 0.3s;"></div>
                        </div>
                    </div>
                `;
            }

            return `
                <li class="memory-card ${classes[m.level]}" data-mem-id="${m.id}" style="box-shadow: var(--shadow-sm); border-radius: 8px; padding: 12px; margin-bottom: 8px; transition: all 0.15s ease;">
                    <div class="memory-header" style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px; margin-bottom: 6px;">
                        <span class="level-tag ${classes[m.level]}">${labels[m.level]} (Level ${m.level})</span>
                        <span class="importance-badge">Imp: ${(m.importance * 100).toFixed(0)}%</span>
                    </div>
                    <p class="desc-text" style="color: var(--text-primary); margin-top: 4px; white-space: pre-wrap; line-height: 1.4; font-size: 12.5px;">${m.content}</p>
                    ${tripleHtml}
                    ${activationHtml}
                    <div class="details" style="font-size: 9.5px; margin-top: 6px; color: var(--text-secondary); display: flex; justify-content: space-between;">
                        <span>Uses: ${m.use_count}</span>
                        <span>ID: <code style="font-family: monospace;">${m.id.substring(0, 8)}</code></span>
                    </div>
                    <div class="memory-actions" style="margin-top: 8px;">
                        <button class="mem-btn" data-action="up" title="Promote (decrease level)">⬆️</button>
                        <button class="mem-btn" data-action="down" title="Demote (increase level)">⬇️</button>
                        <button class="mem-btn" data-action="imp" title="Edit importance">⚡</button>
                        <button class="mem-btn" data-action="del" title="Delete memory">🗑️</button>
                    </div>
                </li>
            `;
        }).join("");

        // Bind click actions to the newly rendered cards
        memoriesList.querySelectorAll(".memory-actions .mem-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                e.stopPropagation();
                const card = btn.closest(".memory-card");
                const id = card.dataset.memId;
                const action = btn.dataset.action;

                if (action === "del") {
                    const confirmed = await showSlickConfirm(
                        "🗑️ Delete Memory",
                        "Are you sure you want to permanently delete this memory? This cannot be undone."
                    );
                    if (!confirmed) return;
                    try {
                        const res = await fetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
                        if (res.ok) fetchMemories();
                    } catch (err) { console.error(err); }
                } else if (action === "up" || action === "down") {
                    const currentLevel = activeMemoryLevel;
                    const newLevel = action === "up" ? Math.max(1, currentLevel - 1) : Math.min(4, currentLevel + 1);
                    try {
                        const res = await fetch(`/api/memories/${encodeURIComponent(id)}/edit`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ level: newLevel })
                        });
                        if (res.ok) {
                            activeMemoryLevel = newLevel; // Follow the memory card's level swap for seamless UX!
                            fetchMemories();
                        }
                    } catch (err) { console.error(err); }
                } else if (action === "imp") {
                    const currentImp = parseFloat(card.querySelector(".importance-badge").textContent.match(/[\d.]+/)[0]) / 100;
                    const raw = await showCustomPrompt("Edit Importance", "New importance (0.0 - 1.0)", currentImp.toString());
                    if (raw === null || raw === "") return;
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
    }

    // ── 💬 Interactive Conversational Chat Companion ──
    btnChatSend.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    btnChatClear.addEventListener("click", async () => {
        const confirmed = await showSlickConfirm(
            "🧹 Clear Conversation",
            "Are you sure you want to clear the conversation history? This cannot be undone."
        );
        if (!confirmed) return;
        try {
            const res = await fetch("/api/chat/clear", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                chatHistory.innerHTML = ""; // Complete wipe, removes everything
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
        if (activeChatAbortController) {
            try {
                activeChatAbortController.abort(); // Instantly abort the stream reader
            } catch (err) {
                console.error("Abort failed:", err);
            }
        }
        try {
            await fetch("/api/chat/cancel", { method: "POST" });
        } catch (err) {
            console.error("Failed to cancel generation:", err);
        }
    });

    async function sendChatMessage(regenerate = false) {
        console.log("🚀 [DEBUG] sendChatMessage triggered", { regenerate, pastedImagesCount: pastedImages.length });
        let text = "";
        if (!regenerate) {
            text = chatInput.value.trim();
            if (!text && pastedImages.length === 0) {
                console.log("⚠️ [DEBUG] sendChatMessage aborted: both text and pastedImages are empty");
                return;
            }
            if (!text && pastedImages.length > 0) {
                text = "Analyze the attached image.";
                console.log("ℹ️ [DEBUG] Empty text with images, defaulted prompt to:", text);
            }
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

        // Clear and reload chronological history first
        await fetchMessageHistory();

        if (!regenerate) {
            // Append user bubble immediately
            renderMessageBubble(`temp-user-${Date.now()}`, "user", text);
        }

        // Dynamically show the animated thinking indicator at the absolute bottom
        showThinkingIndicator("Lollms is thinking...");
        chatHistory.scrollTop = chatHistory.scrollHeight;

        let currentProse = "";
        let chatActiveProc = null;
        let chatActiveProcDiv = null;
        const chatAffectedArtifacts = new Set();

        let proseSpan = null;
        let bubbleContentDiv = null;
        let activeMsgId = null;

        try {
            const imagesPayload = pastedImages.map(img => img.data);

            const payload = {
                message: text,
                regenerate: regenerate,
                images: imagesPayload.length > 0 ? imagesPayload : null,
                enable_memory: funcStates["enable_memory"],
                enable_artefacts: funcStates["enable_artefacts"],
                enable_in_message_status: funcStates["enable_in_message_status"],
                enable_presentations: funcStates["enable_presentations"],
                enable_books: funcStates["enable_books"],
                enable_skills: funcStates["enable_skills"],
                enable_image_generation: funcStates["enable_image_generation"],
                enable_image_editing: funcStates["enable_image_editing"],
                enable_forms: funcStates["enable_forms"],
                enable_inline_widgets: funcStates["enable_inline_widgets"],
                temperature: (tempOverrideEnable && tempOverrideEnable.checked) ? parseFloat(tempOverrideValue.value) : null,
                think: funcStates["think"]
            };

            if (tempOverrideEnable && tempOverrideEnable.checked) {
                const tVal = parseFloat(tempOverrideValue.value);
                payload.temperature = isNaN(tVal) ? 0.7 : tVal;
            }

            activeChatAbortController = new AbortController();
            console.log("📡 [DEBUG] Sending POST request to /api/chat with payload:", payload);
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: activeChatAbortController.signal
            });

            if (!res.ok) {
                throw new Error(`Chat request returned HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            function processStreamLine(line) {
                const trimmed = line.trim();
                if (!trimmed.startsWith("data: ")) return;

                try {
                    const rawJson = trimmed.slice(6);
                    const event = JSON.parse(rawJson);

                    if (event.error) {
                        if (proseSpan) proseSpan.innerHTML = `<span style="color: #ef4444;">Error: ${event.error}</span>`;
                        return;
                    }

                    const meta = event.meta || {};
                    const evType = meta.type;

                    if (event.msg_type === "MSG_TYPE_NEW_MESSAGE") {
                        activeMsgId = event.chunk;
                        renderMessageBubble(activeMsgId, "assistant", "Lollms is thinking...", {
                            model_name: headerModel.value || "unknown",
                            tokens: 0,
                            speed: 0,
                            ttft: null
                        });

                        const assistantBubble = chatHistory.querySelector(`.chat-bubble.assistant[data-msg-id="${activeMsgId}"]`);
                        if (assistantBubble) {
                            bubbleContentDiv = assistantBubble.querySelector(".chat-assistant-container");
                            proseSpan = assistantBubble.querySelector(".prose-span");
                        }
                        return;
                    }

                    if (event.msg_type === "MSG_TYPE_ARTEFACTS_STATE_CHANGED") {
                        if (meta.artefact && meta.artefact.title) {
                            chatAffectedArtifacts.add(meta.artefact.title);
                        }
                        return;
                    }

                    if (event.msg_type === "MSG_TYPE_ARTEFACT_CHUNK") {
                        const title = meta.title;
                        const chunk = event.chunk;
                        const cleanTitle = sanitizeTabTitle(title);
                        const safeId = makeSafeId(cleanTitle);

                        if (!openTabs.has(cleanTitle)) {
                            // Create the tab silently in the background — DO NOT switch focus from the Chat Companion!
                            createArtifactTab(cleanTitle, meta.art_type || "code");
                        }

                        const rawView = document.getElementById(`raw-view-${safeId}`);
                        if (rawView) {
                            rawView.value += chunk;
                            if (codeEditors[cleanTitle]) {
                                const editor = codeEditors[cleanTitle];
                                const doc = editor.getDoc();
                                editor.replaceRange(chunk, { line: doc.lineCount() });
                            }
                        }
                        return;
                    }

                    if ((event.msg_type === "MSG_TYPE_CHUNK" || event.msg_type === "MSG_TYPE_CODING_PLAN_CHUNK") && proseSpan) {
                        if (event.chunk) {
                            if (ttft === null && currentProse === "") {
                                ttft = (Date.now() - startTime) / 1000;
                            }

                            if (currentProse === "") {
                                proseSpan.innerHTML = "";
                            }

                            currentProse += event.chunk;
                            proseSpan.innerHTML = formatMessageContent(currentProse, activeMsgId);
                            renderMath(proseSpan);
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    }
                } catch (e) {
                    console.error("Failed to parse stream JSON:", e, trimmed);
                }
            }

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    if (buffer.trim()) {
                        processStreamLine(buffer);
                    }
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split(/\r?\n/);
                buffer = lines.pop();

                for (const line of lines) {
                    processStreamLine(line);
                }
            }
        } catch (err) {
            console.error("❌ [DEBUG] sendChatMessage caught error during execution:", err);
            if (proseSpan) proseSpan.innerHTML = `<span style="color: #ef4444;">Connection failed: ${err}</span>`;
        } finally {
            chatInput.disabled = false;

            // Restore Send button and hide Stop button
            btnChatStop.style.display = "none";
            btnChatSend.style.display = "inline-block";
            btnChatSend.disabled = false;
            chatInput.focus();

            hideThinkingIndicator();

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

                    // We also save TTFT and other metadata directly to the message's metadata so it persists across reloads!
                    fetch(`/api/discussions/viewer_session/messages/${activeMsgId}/edit`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            content: currentProse,
                            metadata: { ttft: ttft }
                        })
                    }).catch(() => {});
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

            // Clear pasted images list and previews upon successful completion
            pastedImages.length = 0;
            renderPastedImagesPreview();
            chatInput.style.height = "36px"; // Reset height
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
 * Resolves closed and unclosed <processing> tags into collapsible timeline panels.
 * Run BEFORE marked.parse to prevent markdown list item corruption on nested lines.
 */
function resolveProcessingTags(content) {
    if (!content) return "";

    // ── Render <think> tags or strip them based on active toggle state ──
    const thinkPattern = /<think>([\s\S]*?)(?:<\/think>|$)/gi;
    if (window.funcStates && window.funcStates["think"]) {
        content = content.replace(thinkPattern, (match, bodyText) => {
            const escaped = bodyText.replace(/</g, "&lt;").replace(/>/g, "&gt;").trim();
            return `<details class="inline-proc-accordion" open style="border-color: rgba(147, 51, 234, 0.15);"><summary class="proc-accordion-header" style="color: #c084fc; background-color: rgba(147, 51, 234, 0.05);"><span class="chevron">▶</span><span>💭 Thought Process / Reasoning</span></summary><div class="proc-accordion-content" style="background-color: #020617; border-color: rgba(147, 51, 234, 0.15);"><pre style="color: #cbd5e1; font-family: inherit; font-size: 11.5px; white-space: pre-wrap; line-height: 1.5; margin: 0; padding: 0;">${escaped}</pre></div></details>`;
        });
    } else {
        content = content.replace(thinkPattern, "");
    }

    const procPattern = /<processing\s*([^>]*)>([\s\S]*?)(?:<\/processing>|$)/gi;
    return content.replace(procPattern, (match, attrsStr, bodyText) => {
        const attrs = {};
        attrsStr.replace(/(\w+)=["']([^"']*)["']/g, (m, k, v) => attrs[k] = v);

        const procTitle = attrs.title || attrs.tool || attrs.type || "Task";
        const titleText = procTitle.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());

        const isClosed = match.toLowerCase().endsWith("</processing>");

        // Extract multiline <details> blocks to placeholders so we don't chop them up with split('\n')
        const detailsMap = new Map();
        let tempBody = bodyText;
        const detailsRegex = /<details[\s\S]*?<\/details>/gi;
        tempBody = tempBody.replace(detailsRegex, (detailsMatch) => {
            const id = `__DETAILS_PLACEHOLDER_${Math.random().toString(36).substr(2, 9)}__`;
            detailsMap.set(id, detailsMatch);
            return id;
        });

        const lines = tempBody.trim().split("\n");

        let paramsHtml = "";
        if (attrs.params) {
            try {
                const unescapedParams = attrs.params.replace(/&quot;/g, '"');
                const parsedParams = JSON.parse(unescapedParams);
                const paramEntries = Object.entries(parsedParams).map(([key, val]) => {
                    if (key === "code" && typeof val === "string") {
                        return `<strong>💻 Python Code:</strong><pre style="background-color:#020617; border:1px solid var(--border-color); border-radius:6px; padding:10px; margin-top:4px; font-family:monospace; white-space:pre-wrap; color:#cbd5e1;">${val}</pre>`;
                    }
                    return `<strong>⚙️ ${key}:</strong> <code style="font-family:monospace; background-color:#020617; padding:2px 4px; border-radius:4px; color:#cbd5e1;">${JSON.stringify(val)}</code>`;
                }).join("<br/>");

                if (paramEntries) {
                    paramsHtml = `<div class="proc-item-no-bullet"><details class="proc-params-details" style="margin-top:6px; outline:none;"><summary style="cursor:pointer; font-weight:bold; outline:none; user-select:none;">🔧 Tool Parameters / Arguments</summary><div style="margin-top:8px; font-size:11.5px; line-height:1.5;">${paramEntries}</div></details></div>`;
                }
            } catch(e) {
                console.error("Failed to parse tool parameters for display:", e);
            }
        }

        if (isClosed) {
            let statusItems = lines.map(line => {
                const clean = line.replace(/^\*\s*/, "").trim();
                if (!clean) return "";

                // Restore complete <details> block if this is a placeholder
                if (detailsMap.has(clean)) {
                    return `<div class="proc-item-no-bullet">${detailsMap.get(clean)}</div>`;
                }

                if (clean.startsWith("<details>")) {
                    return `<div class="proc-item-no-bullet">${clean}</div>`;
                }
                return `<div class="proc-item complete">✓ ${clean}</div>`;
            }).filter(x => x !== "").join("");

            if (paramsHtml) {
                statusItems = paramsHtml + statusItems;
            }

            return `<details class="inline-proc-accordion" open>
<summary class="proc-accordion-header complete">
<span class="chevron">▶</span>
<span>✅ ${titleText} (Complete)</span>
</summary>
<div class="proc-accordion-content">
${statusItems}
</div>
</details>`;
        } else {
            let statusItems = lines.map(line => {
                const clean = line.replace(/^\*\s*/, "").trim();
                if (!clean) return "";

                // Restore complete <details> block if this is a placeholder
                if (detailsMap.has(clean)) {
                    return `<div class="proc-item-no-bullet">${detailsMap.get(clean)}</div>`;
                }

                if (clean.startsWith("<details>")) {
                    return `<div class="proc-item-no-bullet">${clean}</div>`;
                }
                return `<div class="proc-item">⤷ ⏳ ${clean}</div>`;
            }).filter(x => x !== "").join("");

            if (paramsHtml) {
                statusItems = paramsHtml + statusItems;
            }

            return `<details class="inline-proc-accordion" open>
<summary class="proc-accordion-header">
<span class="chevron">▶</span>
<span>⚙️ ${titleText}...</span>
<span class="spinner inline"></span>
</summary>
<div class="proc-accordion-content">
${statusItems}
</div>
</details>`;
        }
    });
}

/**
 * Parses raw text and resolves any <artefact_image> anchors into HTML <img> elements.
 */
function resolveImageAnchors(content, title, msgId = null, version = null, msgObj = null) {
    if (!content) return "";

    // 3. Parse and resolve <lollms_form> tags into a gorgeous, interactive, functional form card
    const formPattern = /<lollms_form\s+([^>]*)>([\s\S]*?)<\/lollms_form>/gi;
    content = content.replace(formPattern, (match, attrsStr, bodyText) => {
        const attrs = {};
        attrsStr.replace(/(\w+)=["']([^"']*)["']/g, (m, k, v) => attrs[k] = v);

        const formId = attrs.id || `form_${Math.random().toString(36).substr(2, 9)}`;
        const fTitle = attrs.title || "User Survey Form";
        const fDesc = attrs.description || "Please fill in the required fields.";
        const fSubmitLabel = attrs.submit_label || "Submit Response";

        // Parse fields
        const fields = [];
        const fieldPattern = /<field\s+([^>]+)\/?>/g;
        let fm;
        while ((fm = fieldPattern.exec(bodyText)) !== null) {
            const f_attrs = {};
            fm[1].replace(/(\w+)=["']([^"']*)["']/g, (m, k, v) => f_attrs[k] = v);
            fields.push(f_attrs);
        }

        const fieldsHtml = fields.map(f => {
            const name = f.name || "field";
            const label = f.label || name;
            const type = f.type || "text";
            const req = f.required === "true" ? `<span style="color:#ef4444">*</span>` : "";
            const placeholder = f.placeholder || "";
            const defaultValue = f.default || "";

            let inputHtml = "";
            if (type === "textarea") {
                inputHtml = `<textarea class="form-field-input" data-field-name="${name}" data-field-type="${type}" placeholder="${placeholder}" rows="${f.rows || 3}">${defaultValue}</textarea>`;
            } else if (type === "select") {
                const opts = (f.options || "").split(",").map(o => o.trim()).filter(o => o);
                inputHtml = `
                    <select class="form-field-input" data-field-name="${name}" data-field-type="${type}">
                        ${opts.map(o => `<option value="${o}" ${o === defaultValue ? 'selected' : ''}>${o}</option>`).join("")}
                    </select>
                `;
            } else if (type === "radio") {
                const opts = (f.options || "").split(",").map(o => o.trim()).filter(o => o);
                inputHtml = `
                    <div class="form-field-radio-group">
                        ${opts.map(o => `
                            <label class="form-field-radio-label">
                                <input type="radio" class="form-field-input" name="${formId}-${name}" data-field-name="${name}" data-field-type="${type}" value="${o}" ${o === defaultValue ? 'checked' : ''} />
                                <span>${o}</span>
                            </label>
                        `).join("")}
                    </div>
                `;
            } else if (type === "checkbox") {
                inputHtml = `
                    <label class="form-field-checkbox-label">
                        <input type="checkbox" class="form-field-input" data-field-name="${name}" data-field-type="${type}" ${defaultValue === "true" ? "checked" : ""} />
                        <span>Enable / Agree</span>
                    </label>
                `;
            } else if (type === "range") {
                const min = f.min || 0;
                const max = f.max || 100;
                const step = f.step || 1;
                inputHtml = `
                    <div style="display:flex; align-items:center; gap:10px; width:100%;">
                        <input type="range" class="form-field-input" data-field-name="${name}" data-field-type="${type}" min="${min}" max="${max}" step="${step}" value="${defaultValue || min}" oninput="this.nextElementSibling.textContent = this.value" />
                        <span style="font-weight:bold; min-width:30px; text-align:right;">${defaultValue || min}</span>
                    </div>
                `;
            } else {
                inputHtml = `<input type="text" class="form-field-input" data-field-name="${name}" data-field-type="${type}" value="${defaultValue}" placeholder="${placeholder}" />`;
            }

            return `
                <div class="form-field-group">
                    <label class="form-field-label">${label} ${req}</label>
                    ${inputHtml}
                </div>
            `;
        }).join("");

        return `
            <div class="interactive-form-card" data-form-id="${formId}">
                <div class="form-card-header">
                    <span>📋 ${fTitle}</span>
                </div>
                <p class="form-card-desc">${fDesc}</p>
                <div class="form-card-fields">
                    ${fieldsHtml}
                </div>
                <button class="btn btn-primary btn-submit-form" onclick="submitWorkspaceForm('${formId}')">${fSubmitLabel}</button>
            </div>
        `;
    });

    // ── 🎛️ Parse and resolve <lollms_inline> tags into an interactive iframe card ──
    const inlinePattern = /<lollms_inline\s+([^>]*)>([\s\S]*?)<\/lollms_inline>/gi;
    content = content.replace(inlinePattern, (match, attrsStr, bodyText) => {
        const attrs = {};
        attrsStr.replace(/(\w+)=["']([^"']*)["']/g, (m, k, v) => attrs[k] = v);

        const title = attrs.title || "Interactive Widget";
        const type = attrs.type || "html";

        // Clean any leading/trailing markdown code block fences if present
        let cleanedBody = bodyText.trim();
        const codeBlockMatch = cleanedBody.match(/^```(?:html)?\s*\n([\s\S]+?)\n```\s*$/i);
        if (codeBlockMatch) {
            cleanedBody = codeBlockMatch[1].trim();
        }

        // Create a secure blob URL for the iframe
        const blob = new Blob([cleanedBody], { type: "text/html" });
        const iframeUrl = URL.createObjectURL(blob);
        const widgetId = `inline-widget-${Math.random().toString(36).substr(2, 9)}`;

        return `
            <div class="interactive-widget-card" id="${widgetId}" style="background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: 8px; margin: 12px 0; overflow: hidden; display: flex; flex-direction: column; width: 100%; max-width: 500px; box-shadow: var(--shadow-md);">
                <div class="form-card-header" style="background-color: #020617; padding: 10px 14px; font-weight: bold; color: var(--accent-color); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); font-size:13px; text-transform:uppercase;">
                    <span>🎛️ ${title}</span>
                    <button class="btn btn-secondary" onclick="window.open('${iframeUrl}', '_blank')" style="width: auto; padding: 4px 10px; font-size: 11px; border-radius: 12px; height:auto;">↗ Open in New Tab</button>
                </div>
                <iframe src="${iframeUrl}" style="width: 100%; height: 320px; border: none; background: #fff;" sandbox="allow-scripts"></iframe>
            </div>
        `;
    });

    // 4. Parse and resolve any failed generate_image / edit_image tags into an interactive retry card
    const genPattern = /<(generate_image|edit_image)\s*([^>]*)>([\s\S]*?)<\/\1>/gi;
    content = content.replace(genPattern, (match, tagName, attrsStr, promptText) => {
        // Strict HTML tag stripping to prevent leaking status containers inside prompt box
        const cleanPrompt = promptText.replace(/<\/?[^>]+>/g, "").trim();

        if (!msgId) {
            return `
                <div class="failed-image-card">
                    <div class="failed-image-header">
                        <span>🎨 Image Prompt (${tagName === 'edit_image' ? 'Edit' : 'Generation'})</span>
                    </div>
                    <textarea class="failed-image-prompt-input" readonly style="width: 100%; min-height: 60px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px; border-radius: 4px; resize: vertical; outline: none; font-family: inherit; font-size: 13px;">${cleanPrompt}</textarea>
                </div>
            `;
        }
        return `
            <div class="failed-image-card" data-msg-id="${msgId}">
                <div class="failed-image-header">
                    <span>🎨 Image Prompt (${tagName === 'edit_image' ? 'Edit' : 'Generation'})</span>
                </div>
                <textarea class="failed-image-prompt-input" style="width: 100%; min-height: 60px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px; border-radius: 4px; resize: vertical; outline: none; font-family: inherit; font-size: 13px;">${cleanPrompt}</textarea>
                <button class="btn btn-primary btn-retry-image" onclick="retryImageGeneration('${msgId}')" style="margin-top: 8px;">
                    ${tagName === 'edit_image' ? '🎨 Edit Image' : '🎨 Generate Image'}
                </button>
            </div>
        `;
    });

    const pattern = /<artefact_image\s+id=["']([^"']+)["']\s*(?:\/>|>)/gi;
    return content.replace(pattern, (match, imageId) => {
        if (imageId.includes("::")) {
            const parts = imageId.split("::");
            const imgIndex = parts[parts.length - 1];
            const imgTitle = parts.slice(0, -1).join("::");
            const versionParam = version ? `?version=${version}` : "";

            let prompt = "";
            if (msgObj && msgObj.metadata && msgObj.metadata.image_groups) {
                const group = msgObj.metadata.image_groups.find(g => g.title === imgTitle);
                if (group && group.prompt) {
                    prompt = group.prompt;
                }
            }
            if (!prompt) {
                prompt = `Vibrant infographic design concept illustrating '${imgTitle.replace(/_/g, ' ')}'`;
            }
            const encodedPrompt = encodeURIComponent(prompt);

            return `
                <div class="rendered-image-container">
                    <img src="/api/images/${encodeURIComponent(imgTitle)}/${imgIndex}${versionParam}" class="rendered-page-img" onerror="this.parentNode.innerHTML='<div class=&quot;failed-image-card&quot; style=&quot;margin:10px 0; width:460px; max-width:100%;&quot;><div class=&quot;failed-image-header&quot;><span>🎨 Image Prompt (Artifact Missing)</span></div><textarea class=&quot;failed-image-prompt-input&quot; style=&quot;width: 100%; min-height: 80px; background-color: var(--bg-app); border: 1px solid var(--border-color); color: var(--text-primary); padding: 10px; border-radius: 6px; resize: vertical; outline: none; font-family: inherit; font-size: 13px;&quot;>'+decodeURIComponent('${encodedPrompt}')+'</textarea><button class=&quot;btn btn-primary btn-retry-image&quot; onclick=&quot;retryImageGeneration(\\'${msgId}\\', \\'${encodeURIComponent(imgTitle)}\\')&quot; style=&quot;margin-top: 8px; width:auto !important; padding: 6px 14px !important;&quot;>🎨 Generate Image</button></div>';" />
                    <div class="img-actions-overlay">
                        <button class="img-action-overlay-btn edit" onclick="openImageEditor('${encodeURIComponent(imgTitle)}', ${imgIndex})">✏️ Edit Image</button>
                        <button class="img-action-overlay-btn delete" onclick="deleteArtifactImage('${encodeURIComponent(imgTitle)}', ${imgIndex})">✕ Remove Image</button>
                    </div>
                </div>`;
        }
        return match;
    });
}

window.openImageEditor = function(title, index) {
    const url = `/editor.html?title=${title}&index=${index}`;
    window.open(url, "_blank");
};

window.submitWorkspaceForm = async function(formId) {
    const card = document.querySelector(`.interactive-form-card[data-form-id="${formId}"]`);
    if (!card) return;

    const btn = card.querySelector(".btn-submit-form");
    if (btn) {
        btn.disabled = true;
        btn.textContent = "Submitting...";
    }

    // Collect field answers
    const answers = {};
    const inputs = card.querySelectorAll(".form-field-input");
    inputs.forEach(inp => {
        const name = inp.getAttribute("data-field-name");
        const type = inp.getAttribute("data-field-type");

        if (type === "radio") {
            if (inp.checked) {
                answers[name] = inp.value;
            }
        } else if (type === "checkbox") {
            answers[name] = inp.checked;
        } else {
            answers[name] = inp.value;
        }
    });

    try {
        const res = await fetch(`/api/discussions/viewer_session/forms/${formId}/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answers })
        });
        const data = await res.json();
        if (data.success) {
            // Dynamically re-render and trigger the LLM completion turn without reload!
            await fetchMessageHistory();
            await fetchBranches();
            sendChatMessage(true);
        } else {
            alert("Submission failed: " + (data.detail || data.error));
            if (btn) {
                btn.disabled = false;
                btn.textContent = "Submit Response";
            }
        }
    } catch (err) {
        alert(`Request failed: ${err}`);
        if (btn) {
            btn.disabled = false;
            btn.textContent = "Submit Response";
        }
    }
};

window.retryImageGeneration = async function(msgId) {
    const btn = document.querySelector(`.failed-image-card[data-msg-id="${msgId}"] .btn-retry-image`);
    let loader = null;
    let textElement = null;
    let timerId = null;

    const messages = [
        { time: 0, text: "Generating your masterpiece... Please hold on, this can take up to 30 seconds depending on your TTI engine." },
        { time: 10000, text: "Still rendering... Diffusion models are carefully arranging the pixels. Thank you for your patience." },
        { time: 25000, text: "Adding fine details and textures... Almost there! This is going to look amazing." },
        { time: 45000, text: "Polishing the artwork... Generating complex high-resolution details takes a bit of extra time." },
        { time: 70000, text: "Taking longer than expected, but we are still actively refining and completing your image..." },
        { time: 100000, text: "Almost done... Handcrafting high-fidelity AI art takes passion and a few extra seconds." }
    ];

    if (btn) {
        btn.style.display = "none";
        loader = document.createElement("div");
        loader.className = "generation-loading-container";
        loader.innerHTML = `
            <span class="spinner"></span>
            <span class="generation-loading-text">Generating your masterpiece... Please hold on, this can take up to 30 seconds depending on your TTI engine.</span>
        `;
        btn.parentNode.appendChild(loader);
        textElement = loader.querySelector(".generation-loading-text");

        const startTime = Date.now();
        timerId = setInterval(() => {
            const elapsed = Date.now() - startTime;
            let activeMsg = messages[0].text;
            for (const msg of messages) {
                if (elapsed >= msg.time) {
                    activeMsg = msg.text;
                }
            }
            if (textElement) {
                textElement.textContent = activeMsg;
            }
        }, 1000);
    }

    const card = document.querySelector(`.failed-image-card[data-msg-id="${msgId}"]`);
    const promptInput = card ? card.querySelector(".failed-image-prompt-input") : null;
    const editedPrompt = promptInput ? promptInput.value.trim() : null;

    try {
        const res = await fetch(`/api/discussions/viewer_session/messages/${msgId}/generate_image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: editedPrompt })
        });
        const data = await res.json();
        if (data.success) {
            location.reload();
        } else {
            alert(`Generation failed: ${data.detail || data.error}`);
            if (btn) {
                btn.style.display = "inline-block";
            }
            if (loader) {
                loader.remove();
            }
        }
    } catch (err) {
        alert(`Request failed: ${err}`);
        if (btn) {
            btn.style.display = "inline-block";
        }
        if (loader) {
            loader.remove();
        }
    } finally {
        if (timerId) {
            clearInterval(timerId);
        }
    }
};

window.deleteArtifactImage = async function(title, index) {
    const decodedTitle = decodeURIComponent(title);
    const baseTitle = decodedTitle.endsWith("::images") ? decodedTitle.replace("::images", "") : decodedTitle;

    const confirmed = await showSlickConfirm(
        "🖼️ Remove Image",
        "Are you sure you want to remove this image from the artifact?"
    );
    if (!confirmed) return;

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