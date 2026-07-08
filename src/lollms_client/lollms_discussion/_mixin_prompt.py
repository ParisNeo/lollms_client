# lollms_discussion/_mixin_prompt.py
# PromptMixin: system-prompt instruction builders and LLM response post-processor.
#
# Changes vs previous version:
#   • _build_artefact_instructions() now embeds the surgical-update doctrine
#     (SEARCH/REPLACE patch policy, decision threshold, retry guidance) so the
#     LLM receives it on EVERY path — fast, no-tools, simplified, agentic.
#   • Fixed undefined `meta_now2` reference in the silent-artifact guard; the
#     form-summary branch now iterates over affected_artefacts directly.
#   • Minor cleanup: consistent quoting, removed redundant blank lines.

import re
import uuid
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ascii_colors import ASCIIColors
from lollms_client.lollms_artefact import ArtefactType

if TYPE_CHECKING:
    from ._message import LollmsMessage


class PromptMixin:
    """
    Builds artifact / image-generation / inline-widget / form instructions for the
    system prompt and post-processes the LLM response to apply any XML action
    tags it contains.
    """

    # ─────────────────────────────────────── instruction builders ────────────

    def _build_book_instructions(self) -> str:
        """Instructions for creating high-quality HTML books."""
        return """
=== BOOK ARTEFACTS ===
When requested to write a book, use <artifact type="book" title="Book Title">.
Books must be written in SEMANTIC HTML5.
Rules:
1. Include a <style> block at the top for layout (typography, chapter spacing).
2. Use <section class="chapter"> for each chapter.
3. Use <h1> for the book title and <h2> for chapters.
4. You can use <img> tags (referencing <artefact_image id="TITLE::N" /> if images were generated).
5. The content will be rendered as a rich interactive book in the UI and converted 1:1 to PDF.
=== END BOOK INSTRUCTIONS ===
"""
    def _build_presentation_instructions(self) -> str:
        """
        Compact presentation schema instructions (~250 tokens).
        Injected only when enable_presentations=True is passed to chat().
        """
        return """
=== PRESENTATION ARTEFACTS ===
Create slides with type="presentation". The artefact is self-contained HTML
rendered live in the browser and exportable to PPTX/PDF by the app.
 
FULL TEMPLATE (copy and extend):
<artifact name="deck.html" type="presentation">
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  /* Import a Google Font for visual polish */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
 
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0a;
  }
 
  /* ── Slideshow container ── */
  .presentation { width: 100%; }
 
  html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
  }
  .presentation {
    width: 100%;
    height: 100%;
    position: relative;
    overflow: hidden;
  }
  /* ── Each slide is a 16:9 canvas ── */
  .slide {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 1920px; height: 1080px;
    margin-left: -960px;
    margin-top: -540px;
    transform-origin: center center;
    overflow: hidden;
    display: none;          /* JS controls visibility */
  }
  .slide.active { display: flex; flex-direction: column; }
 
  /* ── Semantic content classes (read by PPTX exporter) ── */
  .slide-bg       { position: absolute; inset: 0; z-index: 0; }
  .slide-title    { position: relative; z-index: 1; }
  .slide-subtitle { position: relative; z-index: 1; }
  .slide-body     { position: relative; z-index: 1; }
  .slide-bullets  { position: relative; z-index: 1; list-style: none; }
  .slide-image    { position: relative; z-index: 1; }
  .slide-image img { width: 100%; height: 100%; object-fit: cover; }
  .slide-svg      { position: absolute; inset: 0; z-index: 2; pointer-events: none; }
  .slide-chart    { position: relative; z-index: 1; }
 
  /* ── Viewer chrome ── */
  #viewer-controls {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: 16px;
    background: rgba(0,0,0,0.7); border-radius: 32px;
    padding: 10px 24px; color: #fff; font-size: 14px;
    backdrop-filter: blur(8px); z-index: 9999;
    user-select: none;
  }
  #viewer-controls button {
    background: none; border: none; color: #fff;
    font-size: 16px; cursor: pointer; padding: 6px 12px;
    border-radius: 18px; transition: background 0.2s;
  }
  #viewer-controls button:hover { background: rgba(255,255,255,0.15); }
  #slide-counter { min-width: 60px; text-align: center; opacity: 0.8; }

  /* ── Speaker Notes panel ── */
  #speaker-notes-panel {
    display: none;
    position: fixed; bottom: 84px; left: 50%; transform: translateX(-50%);
    width: 80%; max-width: 800px; background: rgba(11, 15, 25, 0.9);
    border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;
    padding: 16px 20px; color: #cbd5e1; font-size: 14px;
    line-height: 1.5; backdrop-filter: blur(8px); z-index: 9998;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
  }
  #speaker-notes-panel.active { display: block; }

  /* ── Overview mode (ESC) ── */
  #overview {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.92); z-index: 9998;
    overflow-y: auto; padding: 32px;
  }
  #overview.active { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }
  .overview-thumb {
    width: 320px; height: 180px; overflow: hidden;
    border-radius: 8px; cursor: pointer;
    border: 2px solid transparent; transition: border-color 0.2s;
    position: relative;
  }
  .overview-thumb:hover { border-color: #6c63ff; }
  .overview-thumb .slide { display: flex !important; transform: scale(0.1667); transform-origin: top left; }
  .overview-num {
    position: absolute; bottom: 4px; right: 8px;
    color: #fff; font-size: 11px; opacity: 0.7;
  }
</style>
</head>
<body>
 
<article class="presentation" data-theme="dark" data-accent="#6c63ff" data-font="Inter">
 
  <!-- ═══ SLIDE 1 — Title ═══ -->
  <section class="slide active" data-layout="title" data-notes="Opening remarks here">
    <div class="slide-bg" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);"></div>
 
    <div style="position:relative; z-index:1; display:flex; flex-direction:column;
                align-items:center; justify-content:center; height:100%; padding:80px;">
      <h1 class="slide-title" style="font-size:96px; font-weight:700; color:#e0e0e0;
                                      text-align:center; line-height:1.1; margin-bottom:32px;">
        Presentation Title
      </h1>
      <h2 class="slide-subtitle" style="font-size:40px; font-weight:400; color:#a0a0c0;
                                         text-align:center;">
        Your subtitle or tagline
      </h2>
    </div>
  </section>
 
  <!-- ═══ SLIDE 2 — Content with bullets ═══ -->
  <section class="slide" data-layout="content" data-notes="Talk through each point">
    <div class="slide-bg" style="background: #16213e;"></div>
 
    <div style="position:relative; z-index:1; padding:80px; height:100%;
                display:flex; flex-direction:column; gap:40px;">
      <h2 class="slide-title" style="font-size:64px; font-weight:700; color:#e0e0e0;">
        Key Points
      </h2>
      <ul class="slide-bullets" style="display:flex; flex-direction:column; gap:24px;">
        <li style="font-size:32px; color:#c0c0d8; display:flex; align-items:center; gap:16px;">
          <span style="color:#6c63ff; font-size:24px;">▶</span> First important point
        </li>
        <li style="font-size:32px; color:#c0c0d8; display:flex; align-items:center; gap:16px;">
          <span style="color:#6c63ff; font-size:24px;">▶</span> Second important point
        </li>
        <li style="font-size:32px; color:#c0c0d8; display:flex; align-items:center; gap:16px;">
          <span style="color:#6c63ff; font-size:24px;">▶</span> Third important point
        </li>
      </ul>
    </div>
  </section>
 
  <!-- ═══ SLIDE 3 — Chart ═══ -->
  <section class="slide" data-layout="content" data-notes="Explain the data trend">
    <div class="slide-bg" style="background: #16213e;"></div>
 
    <div style="position:relative; z-index:1; padding:80px; height:100%;
                display:flex; flex-direction:column; gap:40px;">
      <h2 class="slide-title" style="font-size:64px; font-weight:700; color:#e0e0e0;">
        Data Overview
      </h2>
      <figure class="slide-chart" data-type="bar"
              style="flex:1; background:rgba(255,255,255,0.05);
                     border-radius:16px; padding:24px;">
        <canvas id="chart-0"></canvas>
        <script type="application/json">
          {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "datasets": [
              {
                "label": "Revenue ($k)",
                "data": [42, 68, 55, 89],
                "backgroundColor": ["#6c63ff","#6c63ff","#6c63ff","#6c63ff"]
              }
            ]
          }
        </script>
      </figure>
    </div>
  </section>
 
  <!-- ═══ SLIDE 4 — Two column (text + image) ═══ -->
  <section class="slide" data-layout="two-column" data-notes="">
    <div class="slide-bg" style="background: #1a1a2e;"></div>
 
    <div style="position:relative; z-index:1; padding:80px; height:100%;
                display:grid; grid-template-columns:1fr 1fr; grid-template-rows:auto 1fr;
                gap:40px;">
      <h2 class="slide-title" style="font-size:60px; font-weight:700; color:#e0e0e0;
                                      grid-column:1/-1;">
        Two Column Layout
      </h2>
      <ul class="slide-bullets" style="list-style:none; display:flex;
                                        flex-direction:column; gap:20px; align-self:start;">
        <li style="font-size:28px; color:#c0c0d8;">• Left column content</li>
        <li style="font-size:28px; color:#c0c0d8;">• More text here</li>
        <li style="font-size:28px; color:#c0c0d8;">• And another point</li>
      </ul>
      <figure class="slide-image" style="border-radius:16px; overflow:hidden;">
        <!-- Replace with: <artefact_image id="my_image::0" /> -->
        <div style="width:100%;height:100%;background:linear-gradient(135deg,#6c63ff,#028090);
                    display:flex;align-items:center;justify-content:center;
                    color:#fff;font-size:32px;">Image placeholder</div>
      </figure>
    </div>
  </section>
 
  <!-- ═══ SLIDE 5 — SVG shapes ═══ -->
  <section class="slide" data-layout="blank" data-notes="">
    <div class="slide-bg" style="background:#0f3460;"></div>
    <div class="slide-svg">
      <svg viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg" width="1920" height="1080">
        <rect x="200" y="200" width="500" height="300" rx="24"
              fill="#6c63ff" opacity="0.85"/>
        <text x="450" y="370" text-anchor="middle"
              font-family="Inter,sans-serif" font-size="48" fill="#fff">Box A</text>
        <line x1="720" y1="350" x2="880" y2="350"
              stroke="#fff" stroke-width="4" marker-end="url(#arrow)"/>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#fff"/>
          </marker>
        </defs>
        <rect x="900" y="200" width="500" height="300" rx="24"
              fill="#028090" opacity="0.85"/>
        <text x="1150" y="370" text-anchor="middle"
              font-family="Inter,sans-serif" font-size="48" fill="#fff">Box B</text>
      </svg>
    </div>
  </section>
 
</article>
 
<!-- ── Viewer controls ── -->
<div id="viewer-controls">
  <button id="btn-prev" title="Previous (←)">&#8592;</button>
  <span id="slide-counter">1 / 1</span>
  <button id="btn-next" title="Next (→)">&#8594;</button>
  <button id="btn-fs"   title="Fullscreen (F)">&#x26F6;</button>
  <button id="btn-notes" title="Toggle Speaker Notes (N)">💬 Notes</button>
  <button id="btn-ov"   title="Overview (Esc)">&#8942;</button>
</div>

<!-- ── Speaker Notes panel ── -->
<div id="speaker-notes-panel"></div>

<!-- ── Overview panel ── -->
<div id="overview"></div>
 
<!-- ── Chart.js (CDN, loaded async — graceful offline fallback) ── -->
<script>
(function() {
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js';
  s.onload = initCharts;
  s.onerror = function() { console.warn('Chart.js offline — charts will not render'); };
  document.head.appendChild(s);
})();
 
function initCharts() {
  document.querySelectorAll('.slide-chart').forEach(function(fig, idx) {
    var script = fig.querySelector('script[type="application/json"]');
    if (!script) return;
    try {
      var data     = JSON.parse(script.textContent);
      var chartType = fig.dataset.type || 'bar';
      var canvas   = fig.querySelector('canvas') || document.createElement('canvas');
      if (!canvas.parentNode) fig.appendChild(canvas);
      canvas.style.maxHeight = '100%';
      new Chart(canvas, {
        type: chartType,
        data: data,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#c0c0d8', font: { size: 18 } } }
          },
          scales: chartType !== 'pie' && chartType !== 'doughnut' && chartType !== 'radar' ? {
            x: { ticks: { color: '#a0a0c0', font: { size: 16 } },
                 grid:  { color: 'rgba(255,255,255,0.08)' } },
            y: { ticks: { color: '#a0a0c0', font: { size: 16 } },
                 grid:  { color: 'rgba(255,255,255,0.08)' } }
          } : {}
        }
      });
    } catch(e) { console.error('Chart init error', e); }
  });
}
 
/* ── Slideshow controller ── */
(function() {
  var slides  = Array.from(document.querySelectorAll('.slide'));
  var current = 0;

  function doScale() {
    var maxW = 1920;
    var maxH = 1080;
    var winW = window.innerWidth;
    var winH = window.innerHeight;
    var scale = Math.min(winW / maxW, winH / maxH);
    slides.forEach(function(s) {
      s.style.transform = 'scale(' + scale + ')';
    });
  }
  window.addEventListener('resize', doScale);
  window.addEventListener('load', doScale);

  function show(n) {
    slides.forEach(function(s, i) { s.classList.toggle('active', i === n); });
    current = n;
    document.getElementById('slide-counter').textContent =
      (n + 1) + ' / ' + slides.length;
    doScale();
    updateNotes();
  }

  show(0);

  // Periodically trigger doScale to auto-correct scaling inside hidden tabs/iframes
  setInterval(doScale, 500);

  document.getElementById('btn-prev').onclick = function() {
    show((current - 1 + slides.length) % slides.length);
  };
  document.getElementById('btn-next').onclick = function() {
    show((current + 1) % slides.length);
  };
  document.getElementById('btn-fs').onclick = function() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  };
 
  /* Overview */
  var ov = document.getElementById('overview');
  document.getElementById('btn-ov').onclick = toggleOverview;
 
  function buildOverview() {
    ov.innerHTML = '';
    slides.forEach(function(s, i) {
      var thumb = document.createElement('div');
      thumb.className = 'overview-thumb';
      var clone = s.cloneNode(true);
      clone.classList.add('active');
      thumb.appendChild(clone);
      var num = document.createElement('span');
      num.className = 'overview-num';
      num.textContent = i + 1;
      thumb.appendChild(num);
      thumb.onclick = function() { show(i); toggleOverview(); };
      ov.appendChild(thumb);
    });
  }
 
  function toggleOverview() {
    if (ov.classList.toggle('active')) buildOverview();
  }

  /* Speaker Notes Panel */
  var btnNotes = document.getElementById('btn-notes');
  var notesPanel = document.getElementById('speaker-notes-panel');

  btnNotes.onclick = toggleNotes;

  function toggleNotes() {
    notesPanel.classList.toggle('active');
    btnNotes.classList.toggle('active');
    updateNotes();
  }

  function updateNotes() {
    var activeSlide = slides[current];
    if (activeSlide) {
      var notes = activeSlide.getAttribute('data-notes') || 'No speaker notes for this slide.';
      notesPanel.innerHTML = '<strong>📝 Speaker Notes:</strong><p style="margin-top:6px; font-weight:normal;">' + notes + '</p>';
    }
  }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  show((current+1) % slides.length);
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')    show((current-1+slides.length) % slides.length);
    if (e.key === 'Escape')    { if (ov.classList.contains('active')) toggleOverview(); }
    if (e.key === 'f' || e.key === 'F') document.getElementById('btn-fs').click();
    if (e.key === 'n' || e.key === 'N') toggleNotes();
  });
})();
</script>
 
</body>
</html>
</artifact>
 
SEMANTIC CLASSES — used by the PPTX exporter (keep them on the right elements):
  .slide-title    → main heading      .slide-subtitle → secondary heading
  .slide-body     → free text block   .slide-bullets  → <ul> of <li> items
  .slide-image    → <figure> wrapper  .slide-chart    → chart figure (see below)
  .slide-svg      → SVG shape layer   .slide-bg       → background div
 
CHARTS — declarative JSON inside <script type="application/json">:
  data-type: bar | column | line | pie | doughnut | area | scatter | radar
  JSON shape: { "labels": [...], "datasets": [{ "label": "...", "data": [...] }] }
 
SVG SHAPES — embed an <svg viewBox="0 0 1920 1080"> inside .slide-svg for
  arrows, callouts, connectors, icons, flowchart boxes, etc.
 
ARTEFACT IMAGES — reference images from other artefacts:
  <figure class="slide-image"><artefact_image id="TITLE::0" /></figure>
 
SLIDES LAYOUTS (data-layout attribute):
  title | content | two-column | image-full | quote | blank
 
RULES:
  • Every slide needs a .slide-bg div as the first child for background.
  • Make it visually stunning — use gradients, bold typography, meaningful colours.
  • data-notes="..." on each <section> becomes PowerPoint speaker notes.
  • Charts render via Chart.js in browser; PPTX export uses native chart objects.
  • Never use external image URLs — use data URIs or <artefact_image> anchors.
=== END PRESENTATION ARTEFACTS ===
"""

    def _build_artefact_instructions(self) -> str:
        """Optimized artifact system instructions (~40% token reduction)."""
        lines = [
            "",
            "=== ARTIFACT SYSTEM ===",
            "Artifacts (<artifact> XML) ≠ markdown code blocks (``` ```).",
            "• Markdown = temporary display only",
            "• Artifacts = persistent, versioned, functional action",
            "",
            "🚨🚨🚨 CRITICAL ANTI-MIMICRY PROTOCOL — READ THIS CAREFULLY 🚨🚨🚨",
            "",
            "**WHAT YOU WILL SEE IN HISTORY:** After you create an artifact with `<artifact name=\"file.py\">...</artifact>`, the system REPLACES your XML in the conversation history with a placeholder like:",
            "  `[🔒SYSTEM_ARTIFACT_CREATED:file.py|code]`",
            "",
            "**WHAT YOU MUST NOT DO:** You are **STRICTLY FORBIDDEN** from reproducing this placeholder text in your responses. Examples of FORBIDDEN output:",
            "  ❌ `[🔒SYSTEM_ARTIFACT_CREATED:main.py|code]`",
            "  ❌ `[🔒SYSTEM_ARTIFACT_CREATED:data.csv|data]`",
            "  ❌ Any variation of `[🔒SYSTEM_ARTIFACT_CREATED:...]`",
            "",
            "**WHY:** This text is a **SYSTEM-GENERATED PLACEHOLDER** that appears ONLY in history to save space. If you output it, **NOTHING HAPPENS** — no artifact is created, no action occurs. It is dead text.",
            "",
            "**WHAT TO DO INSTEAD:** To create or update artifacts, you MUST output the **ACTUAL XML TAGS**:",
            "  ✅ `<artifact name=\"file.py\" type=\"code\">...your code here...</artifact>`",
            "  ✅ `<artifact name=\"data.csv\" type=\"data\">...content...</artifact>`",
            "",
            "**REMEMBER:** History placeholders are READ-ONLY. They are NOT templates to copy. They are NOT valid responses. Always use real `<artifact>` tags for action.",
            "",
            "🚨 **ADDITIONAL ANTI-MIMICRY RULES:**",
            "1. **NEVER MIMIC SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from outputting patterns like `[🔒SYSTEM_ARTIFACT_ANCHOR:...`, `[SYSTEM:`, `[content stripped...`, or `[unlocked and loaded context files: ...]`. These are **INFRASTRUCTURE-ONLY** markers. If you output them, NOTHING happens.",
            "2. **FUNCTIONAL TAGS ONLY**: To create or update content, you MUST output the **ACTUAL XML TAGS** (e.g., `<artifact name=\"file.py\">...`).",
            "3. **HISTORY IS NOT TEMPLATE**: Past messages containing system markers are for reference ONLY. Do NOT copy their format.",
            "",
            "CRITICAL RULES:",
            "1. System INTERCEPTS functional tags (<artifact>, <tool>, <lollms_inline>). History shows opaque marker — DO NOT reproduce it.",
            "2. Prose ≠ action. Explaining without XML tags = NOTHING happens.",
            "3. NEVER start lines with `* ` (log mimicry breaks UI).",
            "4. NEVER wrap XML in code fences. Output raw XML directly.",
            "5. ALL functional tags (<artifact>, <note>, <skill>, <lollms_inline>, <lollms_form>, etc.) MUST start on a BRAND NEW line (allowing optional leading indentation). Any tag placed inline inside conversational prose will be completely ignored by the compiler.",
            "6. STRICT THINKING SEGREGATION (ABSOLUTE RULE): The thinking phase () is solely for planning and cognitive reasoning. You are STRICTLY FORBIDDEN from generating `<processing>`, `</processing>`, `<lollms_event>`, `<tool>`, or `<tool_result>` tags inside your thoughts. You must NEVER simulate mock execution outcomes or write fake terminal logs. Those tags are strictly generated by the system runner.",
            "7. **TAG ISOLATION**: Functional tags (`<artifact>`, `<tool>`, `<tool_result>`) MUST NEVER appear inside <think> blocks. They must ONLY appear in the final response body AFTER the closing </think> tag.",
            "8. **CRITICAL CONTENT SEGREGATION (ANTI-NESTING RULE)**: The content INSIDE an `<artifact>` tag MUST be PURE FILE CONTENT ONLY (e.g., pure HTML, pure Python, pure CSS). You are STRICTLY FORBIDDEN from placing tool calls (`<tool>`), system markers (`[SYSTEM:`), or other XML tags inside an artifact block. Write the file first, CLOSE the tag, THEN call tools on new lines.",
            "",
            "Supported types: " + ", ".join(sorted(list(ArtefactType.ALL))),

            "",
            "=== AUTOMATIC VERSIONING & FILENAME STABILITY (CRITICAL) ===",
            "The system AUTOMATICALLY handles versioning. You MUST NEVER manually append version numbers to filenames (e.g., `_v2`, `v3`).",
            "• When you update an existing artifact using `<artifact name=\"file.py\">`, the system automatically creates a new version in the background.",
            "• The `name` attribute MUST ALWAYS match the original filename exactly (e.g., `name=\"file.py\"`, NEVER `name=\"file_v2.py\"`).",
            "• This ensures a clean, continuous version history without duplicate files cluttering the workspace.",
            "",
            "=== SURGICAL UPDATE PREFERENCE (MANDATORY) ===",
            "For any update to an existing file where the changes affect ≤ 60% of the lines, you MUST use the Aider `SEARCH/REPLACE` patch format.",
            "Do NOT rewrite the entire file from scratch unless the changes are substantial (> 60%) or the file is brand new.",
            "This saves tokens, prevents context bloat, and guarantees that untouched code remains perfectly stable.",
            "",
            "=== SOURCE OF TRUTH ===",
            "Use [Active Artefacts] list as your ONLY technical reference. Ignore conflicting history.",
            "",
            "=== MULTI-TIER VISIBILITY & UNLOCK PROTOCOL (CRITICAL) ===",
            "To optimize your context window budget, ALL newly generated or updated artifacts are non-visible and hidden [U] by default.",
            "Files in the 'Workspace Directory Tree Index' carry visibility markers:",
            "  - [C] Fully Loaded: You have full verbatim read-ready access to this file's code/content below.",
            "  - [M] Signature / Metadata Only (Exposes schemas, layouts, or code signatures below)",
            "  - [U] Inactive/Unlockable: The file is completely excluded from your context to keep it clean.",
            "  - [L] Locked: Completely excluded, and cannot be loaded or unlocked.",
            "    👉 If you need to free up context space, you can lock or hide files using:",
            "    <lock_file>",
            "    filename.ext",
            "    </lock_file>",
            "    or to completely hide from the tree:",
            "    <hide_file>",
            "    filename.ext",
            "    </hide_file>",
            "",
            "=== PRIORITY POLICY ===",
            "1. ALWAYS prefer <artifact> over <lollms_inline>. Widgets = brief demos only. Games/apps MUST be artifacts.",
            "2. NEVER use <lollms_form> to delay work. Implement with sensible defaults.",
            "3. Target files matching user request domain from Active Artefacts list.",
            "",
            "=== TOOL EXECUTION PROTOCOL ===",
            "If you need to execute code or scripts, use the available tools.",
            "To avoid JSON character escaping errors, you can write your code inside an artifact first, then pass the artifact name to the tool.",
            "",
            "=== NO MANUAL VIEWERS FOR TOOL GENERATED FILES ===",
            "When you call a tool that generates a file, the system automatically renders it. Do not generate manual HTML or markdown links for tool outputs.",
            "",
            "=== ARTEFACT EXECUTION & CREATION PROTOCOLS ===",
            "To CREATE a new file, output the full content directly:",
            "  <artifact name=\"file.ext\" type=\"type\">full content</artifact>",
            "",
            "To PATCH/EDIT an existing file (PREFERRED if changes ≤ 60%), use search-and-replace blocks:",
            "  <artifact name=\"file.ext\">",
            "  <<<<<<< SEARCH",
            "  exact lines to replace (copied verbatim, including indentation)",
            "  =======",
            "  replacement lines",
            "  >>>>>>> REPLACE",
            "  </artifact>",
            "",
            "To RENAME a file:",
            "  <artifact name=\"new.ext\" rename=\"old.ext\">content</artifact>",
            "",
            "To REVERT to a previous version:",
            "  <revert_artifact name=\"file.ext\" version=\"3\" />",
            "",
            "=== SPINOFF SUB-AGENTS DELEGATION (STRICT RULES) ===",
            "For complex, heavy, multi-step, or specialized tasks, you are encouraged to delegate work to in-process sub-agents, with a CRITICAL EXCEPTION for code artifacts:",
            "  • 🚫 DELEGATION FORBIDDEN: You are STRICTLY FORBIDDEN from spinning off another agent (such as 'tool_spinoff_code_specialist') to write, generate, modify, or patch code/file artifacts. You MUST write and edit the artifacts directly yourself within the main active thread.",
            "  • You may still delegate presentation layouts to 'tool_spinoff_presentation_designer' if needed, passing detailed specifications inside the tool parameter.",
            "",
            "🚨 NEVER use <artifact> tags or write code for read-only data queries (use the Python data query tool instead).",
            "",
            "=== SURGICAL UPDATE POLICY ===",
            "PATCH if changed_lines/total ≤ 60%. Full rewrite only if >60%.",
            "SEARCH rules: (1) Copy VERBATIM incl. indentation, (2) ±2 context lines, (3) No ellipses, (4) Unique match.",
            "REPLACE rules: Indent relative to SEARCH. System auto-aligns.",
            "If rejected: Widen SEARCH ±3 lines. Do NOT full-rewrite unless told.",
            "❌ Never include sentinels (<<<<<<<, =======, >>>>>>>) in code.",
            "",
            "=== ARTIFACT IMAGES ===",
            "Anchors: <artefact_image id=\"TITLE::N\" /> (TITLE=artifact, N=0-based index)",
            "• Images appended after user images in vision context",
            "• Preserve anchors when patching",
            "• Reference in replies: 'As shown in <artefact_image id=\"doc::2\" />'",
            "",
            "=== DATA VIEWS ===",
            "NEVER render large tables in prose. Save as CSV via tool → auto-rendered as interactive grid.",
            "",
            "=== DATA INTERFACE POLICY ===",
            "If data analysis tools are available in your tool registry, use them to query, filter, and visualize datasets.",
            "Do not edit type='data' artifacts directly with raw <artifact> tags unless absolutely necessary.",
            "",
            "=== COGNITIVE SCRATCHPAD (MANDATORY FOR MULTI-STEP ANALYSIS) ===",
            "When performing multi-step data exploration or analyzing large text outputs, you MUST maintain a scratchpad.",
            "1. To save your progress, intermediate findings, or hypotheses, output:",
            '  <artifact name="scratchpad.md" type="scratchpad">',
            "  ## Current Goal",
            "  Analyze lithium contamination.",
            "  ## Step 1 Findings",
            "  - Queried DB. Found 1327 rows.",
            "  - Hypothesis: E500A is the primary contaminant.",
            "  Next Step: Aggregate by date and equipment.",
            "  </artifact>",
            "2. **SELF-CORRECTION (CRITICAL)**: If you discover a previous hypothesis was WRONG, you MUST update the scratchpad using a SEARCH/REPLACE block to invalidate the old assumption. Do NOT append contradictory information.",
            '  <artifact name="scratchpad.md" type="scratchpad">',
            "  <<<<<<< SEARCH",
            "  - Hypothesis: E500A is the primary contaminant.",
            "  =======",
            "  - CORRECTION: E500A and M460A are co-contaminants (M460A found in Turn 3).",
            "  >>>>>>> REPLACE",
            "  </artifact>",
            "3. The scratchpad is persistent but hidden [U] by default to save context. Unlock it only if you need to read your previous notes.",
            "=== END COGNITIVE SCRATCHPAD ===",
        ]
        return "\n".join(lines)

    def _build_image_generation_instructions(self) -> str:
        tti = getattr(self.lollmsClient, 'tti', None)
        if tti is None:
            return ""
        lines = [
            "",
            "=== IMAGE GENERATION / EDITING ===",
            "You are equipped with a powerful Text-to-Image (TTI) engine to generate and edit images on-the-fly.",
            "",
            "🚨 MANDATORY PROTOCOL (READ CAREFULLY):",
            "- If the user asks you to generate, draw, paint, create, or show an image, picture, or illustration",
            "  (or uses equivalent prompt verbs like 'crée une image', 'dessine', 'génère' in any language),",
            "  you MUST output the `<generate_image>` XML tag.",
            "- DO NOT write long prose descriptions or tell the user you will try to create it — simply output the tag directly.",
            "- Translate the user's request into a highly detailed English prompt inside the tag describing the subject, style, lighting, and aesthetic.",
            "",
            "Syntax for creating a new image:",
            '<generate_image width="1024" height="1024">',
            "  Detailed English prompt describing the image to generate...",
            "</generate_image>",
            "",
            "Syntax for editing an existing image artifact:",
            '<edit_image name="artifact_name">',
            "  Detailed English prompt describing the edits/modifications...",
            "</edit_image>",
            "=== END IMAGE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_inline_widget_instructions(self) -> str:
        # Retrieve the custom widget CSS or use a modern, sober dark-mode default
        css_text = getattr(self, "widget_css", None) or (
            "body {\n"
            "  font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif;\n"
            "  background-color: #0f172a;\n"
            "  color: #f8fafc;\n"
            "  margin: 0;\n"
            "  padding: 16px;\n"
            "  box-sizing: border-box;\n"
            "  display: flex;\n"
            "  flex-direction: column;\n"
            "  align-items: center;\n"
            "  justify-content: center;\n"
            "  min-height: 100vh;\n"
            "  overflow: hidden;\n"
            "}\n"
            "button {\n"
            "  background-color: #4f46e5;\n"
            "  color: #ffffff;\n"
            "  border: none;\n"
            "  padding: 8px 16px;\n"
            "  border-radius: 6px;\n"
            "  font-weight: 600;\n"
            "  cursor: pointer;\n"
            "}\n"
            "button:hover { background-color: #4338ca; }"
        )

        lines = [
            "",
            "=== INTERACTIVE WIDGET SYSTEM (EDUCATIONAL & EPHEMERAL) ===",
            "Widgets (`<lollms_inline>`) are STRICTLY for in-context, ephemeral educational demonstrations.",
            "Think of them as interactive animations or diagrams used by platforms like Brilliant.org to teach a concept.",
            "",
            "🚨 CRITICAL BOUNDARY RULE (MANDATORY):",
            "1. **WIDGETS (`<lollms_inline>`)**: Use ONLY for teaching, explaining, or visualizing concepts directly in the chat.",
            "   They are ephemeral, disappear when the chat scrolls, and are NOT real applications.",
            "2. **ARTIFACTS (`<artifact>`)**: Use for ALL real coding, games, tools, scripts, and persistent applications.",
            "   If the user asks to 'build a game', 'create an app', or 'write a script', you MUST use `<artifact>`.",
            "   NEVER use `<lollms_inline>` to build a game or a real application.",
            "",
            "🚨 STRICT VISUAL BOUNDARIES (MANDATORY):",
            "Widgets are constrained to a 600x400px viewport inside an iframe. They MUST NOT exceed this size.",
            "Do NOT attempt to make them fullscreen or break out of their container.",
            "",
            "🚨 MANDATORY DRAWING ENGINE (BRILLIANT.ORG STYLE):",
            "For any animations, physics simulations, or interactive graphics, you MUST use **PixiJS**.",
            "PixiJS is a 2D WebGL rendering engine that allows for smooth, high-performance animations.",
            "Include it dynamically in your widget: `var s = document.createElement('script'); s.src = 'https://cdn.jsdelivr.net/npm/pixi.js@7/dist/pixi.min.js'; document.head.appendChild(s);`",
            "Wait for it to load before initializing the canvas: `s.onload = function() { /* init PixiJS here */ };`",
            "",
            "🚨 CRITICAL CHANGE — SIMPLIFIED TAG CONTENTS (MANDATORY):",
            "  • Do NOT write a complete HTML document (no <!DOCTYPE html>, no <html>, <head>, or <body> tags).",
            "  • Start directly with a container <div> tag (max-width: 600px; max-height: 400px;), e.g.:",
            "    <lollms_inline type=\"html\" title=\"Educational Animation\">",
            "    <div id=\"widget-container\" style=\"width: 600px; height: 400px; overflow: hidden;\">",
            "      <!-- Your PixiJS canvas and interactive elements go here -->",
            "    </div>",
            "    <script>",
            "      // Dynamically load PixiJS",
            "      var s = document.createElement('script');",
            "      s.src = 'https://cdn.jsdelivr.net/npm/pixi.js@7/dist/pixi.min.js';",
            "      s.onload = function() {",
            "        // Initialize your PixiJS application here",
            "        let app = new PIXI.Application({ width: 600, height: 400, backgroundColor: 0x0f172a });",
            "        document.getElementById('widget-container').appendChild(app.view);",
            "        // Add your animations and interactive logic...",
            "      };",
            "      document.head.appendChild(s);",
            "    </script>",
            "    </lollms_inline>",
            "",
            "🚨 CSS STYLING RULES:",
            "  • The container is automatically loaded inside an iframe wrapped with this pre-defined CSS stylesheet:",
            f"```css\n{css_text}\n```",
            "  • Do NOT include large style blocks inside your output. Rely on classes and elements already styled by the rules above.",
            "",
            "🚨 JAVASCRIPT & SECURITY RULES:",
            "  • For code, you can use <script> tags. To make sure it is secure:",
            "    - Do NOT attempt to access the parent window or document objects (no window.parent, window.top, or top.document).",
            "    - Do NOT write or read cookies, localStorage, or sessionStorage.",
            "    - Only perform relative fetches to `/api/workspace_files/filename.csv` to load datasets.",
            "",
            "💡 DESIGN HINTS FOR BEAUTIFUL ANIMATIONS:",
            "  - Use PixiJS for fluid, frame-by-frame animations, physics simulations, or interactive data visualizations.",
            "  - Utilize simple state variables to track button clicks, toggles, or slider values to control the animation.",
            "  - Example: Create an interactive spring simulation, a sorting algorithm visualizer, or a geometric proof animation.",
            "=== END INTERACTIVE WIDGET SYSTEM ===",
            ""
        ]
        return "\n".join(lines)


    def _build_note_instructions(self) -> str:
        lines = [
            "",
            "=== NOTE SYSTEM (USER-FACING KNOWLEDGE) ===",
            "",
            "Notes are **persistent documents created for the USER**, not for you.",
            "They serve as a personal knowledge base, reminding the user of facts, decisions, or summaries they can reference in the future.",
            "",
            "✅ WHEN TO CREATE OR UPDATE A NOTE:",
            "  • The user explicitly asks you to 'save a note', 'take a note', or 'remember this for me'.",
            "  • You produced a comprehensive analysis, comparison, or summary that the user would benefit from reading later.",
            "  • Action items, meeting decisions, or project plans are discussed.",
            "",
            "❌ DO NOT create a note for routine conversational answers or one-off calculations.",
            "",
            "🚨 ANTI-MIMICRY WARNING:",
            "After you create a note, the system replaces your `<note>...</note>` XML in history with `[🔒SYSTEM_NOTE_CREATED:title]`.",
            "**NEVER reproduce this placeholder text** — it is system-generated and does nothing if you output it.",
            "To create notes, ALWAYS use the actual `<note title=\"...\">...</note>` XML tag.",
            "",
            "=== AUTOMATIC VERSIONING & SURGICAL EDITING ===",
            "1. **Stable Titles**: The system automatically versions notes. NEVER change the `title` attribute to include version numbers (e.g., `title=\"Note v2\"`). Always use the exact original title.",
            "2. **Surgical Updates**: If you are modifying an existing note and the changes are ≤ 60% of the content, you MUST use the Aider `SEARCH/REPLACE` format inside the `<note>` tag. Do NOT rewrite the whole note.",
            "",
            "To CREATE a new note, output the full content directly:",
            '<note title="Clear, descriptive title">',
            "Content here — plain text or Markdown",
            "</note>",
            "",
            "To EDIT an existing note, use search-and-replace blocks inside the tag:",
            '<note title="Exact Original Title">',
            "<<<<<<< SEARCH",
            "exact lines to replace (copied verbatim, including indentation)",
            "=======",
            "replacement lines",
            ">>>>>>> REPLACE",
            "</note>",
            "",
            "Always add a short explanation outside the tag for the user.",
            "=== END NOTE SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_skill_instructions(self) -> str:
        lines = [
            "",
            "=== SKILL SYSTEM (CONTINUOUS LEARNING & BEHAVIOR) ===",
            "",
            "Skills are **persistent knowledge capsules created for YOU (the AI)**, not for the user.",
            "They allow you to learn from experience, remember how to perform complex tasks, avoid repeating past mistakes, and evolve your behavior over time across different sessions.",
            "",
            "✅ WHEN TO CREATE OR UPDATE A SKILL:",
            "  1. The user explicitly asks you to 'create a skill', 'save this technique', or 'remember this rule'.",
            "  2. You discovered a genuinely reusable methodology, a workaround to a recurring bug, or a specific constraint of a library/tool.",
            "  3. You realize you should avoid doing something (e.g., 'Never use X with Y because it crashes').",
            "",
            "❌ DO NOT create a skill for one-off solutions, trivial facts, or non-generalizable content.",
            "",
            "🚨 ANTI-MIMICRY WARNING:",
            "After you create a skill, the system replaces your `<skill>...</skill>` XML in history with `[🔒SYSTEM_SKILL_CREATED:title]`.",
            "**NEVER reproduce this placeholder text** — it is system-generated and does nothing if you output it.",
            "To create skills, ALWAYS use the actual `<skill title=\"...\" description=\"...\">...</skill>` XML tag.",
            "",
            "=== AUTOMATIC VERSIONING & SURGICAL EDITING ===",
            "1. **Stable Titles**: The system automatically versions skills. NEVER change the `title` attribute to include version numbers (e.g., `title=\"Skill v2\"`). Always use the exact original title.",
            "2. **Surgical Updates**: If you are evolving an existing skill and the changes are ≤ 60% of the content, you MUST use the Aider `SEARCH/REPLACE` format inside the `<skill>` tag. Do NOT rewrite the whole skill.",
            "",
            "To CREATE a new skill, output the full content directly:",
            '<skill title="Concise Skill Name"',
            '       description="One clear sentence explaining what it does"',
            '       category="domain/subdomain">',
            "Content here — Markdown with examples, rules, and usage guidelines",
            "</skill>",
            "",
            "To EVOLVE or EDIT an existing skill, use search-and-replace blocks inside the tag:",
            '<skill title="Exact Original Skill Name">',
            "<<<<<<< SEARCH",
            "exact lines to replace (copied verbatim, including indentation)",
            "=======",
            "replacement lines",
            ">>>>>>> REPLACE",
            "</skill>",
            "",
            "=== END SKILL SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_form_instructions(self) -> str:
        """Instruction for rich interactive form building."""
        return """
=== INTERACTIVE FORMS ===
If you need structured data or multiple details from the user, use <lollms_form>.
The UI will render a rich component and PAUSE your generation until the user submits.

XML Structure:
<lollms_form title="Form Title" description="Subtitle" submit_label="Action">
  <field name="key" label="Display Name" type="TYPE" ...>
    <!-- Options go here for select/radio -->
  </field>
</lollms_form>

Field Types & Attributes:
- text:     <field name="id" label="Name" type="text" placeholder="Hint" />
- textarea: <field name="id" label="Description" type="textarea" />
- select:   (Use BOTH nested <option> tags AND a comma-separated 'options' attribute)
    <field name="id" label="Choose" type="select" options="Choice 1, Choice 2">
      <option>Choice 1</option>
      <option>Choice 2</option>
    </field>
- radio:    (MUST use child <option> tags — FORBIDDEN to use 'options' attribute)
    <field name="id" label="Pick one" type="radio">
      <option>Option A</option>
      <option>Option B</option>
    </field>
- checkbox: <field name="id" label="Enable" type="checkbox" default="true" />
- range:    <field name="id" label="Value" type="range" min="0" max="100" />
- rating:   <field name="id" label="Rate" type="rating" max="5" />

EXAMPLE OF CORRECT FORM:
<lollms_form title="User Survey" description="Please fill this out" submit_label="Submit">
  <field name="user_name" label="Full Name" type="text" placeholder="John Doe" />
  <field name="color" label="Favorite Color" type="select">
    <option>Red</option>
    <option>Blue</option>
    <option>Green</option>
  </field>
  <field name="pref" label="Contact Method" type="radio">
    <option>Email</option>
    <option>Phone</option>
  </field>
</lollms_form>
=== END FORM INSTRUCTIONS ===
"""

    # ─────────────────────────────────── LLM response post-processor ─────────

    def _post_process_llm_response(
        self,
        text: str,
        ai_message: 'LollmsMessage',
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:   bool = True,
        enable_notes:            bool = True,
        enable_skills:           bool = False,
        enable_forms:            bool = True,
        enable_silent_artefact_explanation: bool = True,
        already_processed_artifacts: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Scans the raw LLM response for XML action tags and applies them.

        Handled tags:
            <artifact …>…</artifact>   Create or patch a named artefact.
            <generate_image …>…        Image generation via TTI binding.
            <edit_image …>…            Image editing via TTI binding.
            <lollms_inline …>…         Inline interactive HTML widget.
            <note …>…                  Persistent note (ArtefactType.NOTE).
            <skill …>…                 Knowledge capsule (ArtefactType.SKILL).
            <lollms_form …>…           Interactive form (MSG_TYPE_FORM_READY).

        NOTE: <artefact_image id="..."/> anchors are NOT processed here —
        they are preserved verbatim in the text so the UI can render them.
        """
        # Un-escape any backticks or markdown characters accidentally escaped by the LLM
        text = re.sub(r'\\(`{1,3})', r'\1', text)
        text = text.replace(r'\*', '*').replace(r'\_', '_')
        # ── Mask code blocks and think blocks so XML inside them isn't processed ─────
        code_blocks: Dict[str, str] = {}

        def mask_code_block(match):
            placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
            code_blocks[placeholder] = match.group(0)
            return placeholder

        # Mask think blocks first to prevent processing any XML tags generated inside the thoughts
        masked_text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', mask_code_block, text, flags=re.IGNORECASE)

        # Mask code blocks
        masked_text = re.sub(r'(`{3,})[\s\S]*?\1', mask_code_block, masked_text)
        masked_text = re.sub(r'`[^`]+`',           mask_code_block, masked_text)

        has_artefact = bool(re.search(r'^[ \t]*<(?:revert_)?art[ei]fact[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))
        has_gen      = bool(re.search(r'^[ \t]*<generate_image[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))
        has_edit     = bool(re.search(r'^[ \t]*<edit_image[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))
        has_note     = enable_notes and bool(
            re.search(r'^[ \t]*<note[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))
        has_skill    = enable_skills and bool(
            re.search(r'^[ \t]*<skill[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))
        has_form     = enable_forms and bool(
            re.search(r'^[ \t]*<lollms_form[\s>]', masked_text, re.IGNORECASE | re.MULTILINE))

        if not (has_artefact or has_gen or has_edit
                or has_note or has_skill or has_form):
            return text, []

        cleaned             = masked_text
        affected_artefacts: List[Dict] = []

        def _parse_attrs(attr_str: str) -> Dict[str, str]:
            return {m.group(1): m.group(2)
                    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

        # ── 1. Artifact create / patch ────────────────────────────────────────
        if has_artefact:
            _active_cb = getattr(self, '_active_callback', None)

            def _artefact_event(artefact: Dict, is_new: bool):
                if not _active_cb:
                    return
                import json as _json
                from lollms_client.lollms_types import MSG_TYPE as _MT
                event_type = "artifact_created" if is_new else "artifact_updated"
                try:
                    _active_cb(
                        _json.dumps({
                            "type":     event_type,
                            "title":    artefact.get("title"),
                            "version":  artefact.get("version"),
                            "art_type": artefact.get("type"),
                        }),
                        _MT.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                        {"artefact": artefact, "is_new": is_new},
                    )
                except Exception:
                    pass

            cleaned, affected_artefacts = self.artefacts._apply_artefact_xml(
                cleaned, auto_activate=auto_activate_artefacts,
                replacements=code_blocks,
                event_callback=_artefact_event,
                already_processed_artifacts=already_processed_artifacts,
            )

        # ── 2. Image generation → message.images & Workspace Artifacts ──
        if has_gen:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<generate_image> found but lollmsClient.tti is None — skipping.")
            else:
                gen_pattern = re.compile(
                    r'^[ \t]*<generate_image\s*([^>]*)>(.*?)</generate_image>',
                    re.DOTALL | re.IGNORECASE | re.MULTILINE,
                )

                def handle_generate(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()
                    for placeholder, original in code_blocks.items():
                        prompt = prompt.replace(placeholder, original)
                        for k, v in attrs.items():
                            if placeholder in v:
                                attrs[k] = v.replace(placeholder, original)
                    width  = int(attrs.get('width',  1024))
                    height = int(attrs.get('height', 1024))
                    try:
                        img_bytes = tti.generate_image(
                            prompt=prompt, width=width, height=height)
                        if img_bytes:
                            import base64
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

                            # 1. Append to chat bubble image pack
                            ai_message.add_image_pack(
                                images=[img_b64],
                                group_type="generated",
                                active_by_default=True,
                                title=attrs.get('name', f'gen_{uuid.uuid4().hex[:6]}'),
                                prompt=prompt,
                            )

                            # 2. Ingest as persistent image workspace artifact
                            art_title = attrs.get('name', attrs.get('title', f"generated_image_{uuid.uuid4().hex[:6]}"))
                            art = self.artefacts.add(
                                title=art_title,
                                artefact_type=ArtefactType.IMAGE,
                                content=f"### Generated Image: '{prompt}'\n\n<artefact_image id=\"{art_title}::0\" />",
                                images=[img_b64],
                                image_media_types=["image/png"],
                                active=auto_activate_artefacts
                            )
                            affected_artefacts.append(art)

                            ASCIIColors.success(
                                f"Generated image ({width}×{height}) and created workspace artifact '{art_title}'.")
                            return f'\n<artefact_image id="{art_title}::0" />\n'
                    except Exception as e:
                        ASCIIColors.warning(f"Image generation failed: {e}")
                    return ''

                cleaned = gen_pattern.sub(handle_generate, cleaned)

        # ── 3. Image editing → message.images ────────────────────────────────
        if has_edit:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<edit_image> found but lollmsClient.tti is None — skipping.")
            else:
                edit_pattern = re.compile(
                    r'^[ \t]*<edit_image\s*([^>]*)>(.*?)</edit_image>',
                    re.DOTALL | re.IGNORECASE | re.MULTILINE,
                )

                def handle_edit(match: re.Match) -> str:
                    attrs  = _parse_attrs(match.group(1))
                    prompt = match.group(2).strip()
                    for placeholder, original in code_blocks.items():
                        prompt = prompt.replace(placeholder, original)
                        for k, v in attrs.items():
                            if placeholder in v:
                                attrs[k] = v.replace(placeholder, original)

                    source_b64: Optional[str] = None
                    artefact_name = attrs.get('name', '')
                    if artefact_name:
                        a = self.artefacts.get(artefact_name)
                        if a and a.get('images'):
                            source_b64 = a['images'][-1]
                        else:
                            ASCIIColors.warning(
                                f"<edit_image name='{artefact_name}'> — "
                                "artifact not found or has no images; "
                                "falling back to last message image.")
                    if source_b64 is None:
                        active_imgs = ai_message.get_active_images()
                        if active_imgs:
                            source_b64 = active_imgs[-1]
                    if source_b64 is None:
                        ASCIIColors.warning(
                            "<edit_image> — no source image available; skipping.")
                        return match.group(0)
                    try:
                        img_bytes = tti.edit_image(image=source_b64, prompt=prompt)
                        if img_bytes:
                            import base64
                            edited_b64 = base64.b64encode(img_bytes).decode('utf-8')

                            art_title = attrs.get('name', attrs.get('title', f"edited_image_{uuid.uuid4().hex[:6]}"))
                            ai_message.add_image_pack(
                                images=[edited_b64],
                                group_type="edited",
                                active_by_default=True,
                                title=art_title,
                                prompt=prompt,
                            )

                            art = self.artefacts.add(
                                title=art_title,
                                artefact_type=ArtefactType.IMAGE,
                                content=f"### Edited Image: '{prompt}'\n\n<artefact_image id=\"{art_title}::0\" />",
                                images=[edited_b64],
                                image_media_types=["image/png"],
                                active=auto_activate_artefacts
                            )
                            affected_artefacts.append(art)
                            ASCIIColors.success("Edited image added to message and saved as workspace artifact.")
                            return f'\n<artefact_image id="{art_title}::0" />\n'
                    except Exception as e:
                        ASCIIColors.warning(f"Image edit failed: {e}")
                    return ''

                cleaned = edit_pattern.sub(handle_edit, cleaned)

        # ── 5. Notes ──────────────────────────────────────────────────────────
        if has_note:
            note_pattern = re.compile(
                r'^[ \t]*<note\s*([^>]*?)>(.*?)</note>|^[ \t]*<note\s*([^>]*?/?)>',
                re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )

            def handle_note(match: re.Match) -> str:
                # Handle both standard tags and self-closing/attribute-based tags
                if match.group(2) is not None:
                    attrs   = _parse_attrs(match.group(1))
                    content = match.group(2)
                    attrs_str = match.group(1)
                else:
                    attrs   = _parse_attrs(match.group(3))
                    content = attrs.get('content', attrs.get('Content', ''))
                    attrs_str = match.group(3)

                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title = (attrs.get('title') or attrs.get('name') or
                         f'note_{uuid.uuid4().hex[:8]}')

                note_artefact = self.artefacts.add(
                    title         = title,
                    artefact_type = ArtefactType.NOTE,
                    content       = content.strip(),
                    active        = auto_activate_artefacts,
                )
                affected_artefacts.append(note_artefact)
                ASCIIColors.success(f"Note '{title}' saved.")
                return f'<note {attrs_str}>\n[content stripped, refer to the artefact for details]\n</note>'

            cleaned = note_pattern.sub(handle_note, cleaned)

        # ── 6. Skills ─────────────────────────────────────────────────────────
        if has_skill:
            skill_pattern = re.compile(
                r'^[ \t]*``<skill\s*([^>]*)>(.*?)</skill>``',
                re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )

            def handle_skill(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

                for placeholder, original in code_blocks.items():
                    content = content.replace(placeholder, original)
                    for k, v in attrs.items():
                        if placeholder in v:
                            attrs[k] = v.replace(placeholder, original)

                title       = (attrs.get('title') or attrs.get('name') or
                               f'skill_{uuid.uuid4().hex[:8]}')
                description = attrs.get('description', '')
                category    = attrs.get('category', '')

                skill_artefact = self.artefacts.add(
                    title         = title,
                    artefact_type = ArtefactType.SKILL,
                    content       = content.strip(),
                    active        = auto_activate_artefacts,
                    description   = description,
                    category      = category,
                )
                affected_artefacts.append(skill_artefact)
                ASCIIColors.success(
                    f"Skill '{title}' saved"
                    + (f" [{category}]" if category else "") + "."
                )
                return f'`<skill {match.group(1)}>\n[content stripped, refer to the artefact for details]\n</skill>`'

            cleaned = skill_pattern.sub(handle_skill, cleaned)

        # ── Unmask code blocks ────────────────────────────────────────────────
        for placeholder, original in code_blocks.items():
            cleaned = cleaned.replace(placeholder, original)

        cleaned = cleaned.strip()

        # ── 8. Silent-artifact guard ──────────────────────────────────────────
        # Generates a human-readable summary when the LLM reply was entirely
        # consumed by artifact/note/skill/form tags and nothing visible remains.
        #
        # FIX: the original code referenced an undefined `meta_now2` variable
        # when summarising forms.  We now iterate affected_artefacts directly
        # and detect form artefacts by type, matching the same pattern used for
        # notes and skills above.
        # 
        # NOTE: The cleaned text may contain SYSTEM markers like [🔒SYSTEM_ARTIFACT_CREATED:...
        # We strip these before checking if cleaned is empty, as they are infrastructure not content.
        if enable_silent_artefact_explanation and not cleaned:
            summary_parts: List[str] = []

            # Strip SYSTEM markers from cleaned check - they don't count as real content
            cleaned_no_markers = re.sub(r'\[🔒SYSTEM_[^\]]+\]', '', cleaned or '')
            if cleaned_no_markers.strip():
                return cleaned, affected_artefacts  # Has real content, no summary needed

            non_note_non_skill = [
                a for a in affected_artefacts
                if a.get('type') not in (ArtefactType.NOTE, ArtefactType.SKILL)
            ]
            for art in non_note_non_skill:
                atype     = art.get('type', 'artifact')
                title     = art.get('title', 'untitled')
                lang      = art.get('language', '')
                art_ver   = art.get('version', 1)
                desc      = art.get('description', '')
                img_count = len(art.get('images') or [])
                lang_str  = f" ({lang})"       if lang           else ""
                ver_str   = f" — version {art_ver}" if art_ver > 1 else ""
                desc_str  = f": {desc}"         if desc           else ""
                img_str   = f" · {img_count} image(s)" if img_count else ""
                summary_parts.append(
                    f"📄 Created **{title}**{lang_str} [{atype}{ver_str}]{desc_str}{img_str}."
                )

            notes = [a for a in affected_artefacts if a.get('type') == ArtefactType.NOTE]
            for note in notes:
                title      = note.get('title', 'untitled')
                first_line = next(
                    (l.strip() for l in note.get('content', '').splitlines() if l.strip()),
                    ''
                )
                peek = f" — {first_line[:80]}…" if first_line else ""
                summary_parts.append(f"📝 Saved note **{title}**{peek}")

            skills = [a for a in affected_artefacts if a.get('type') == ArtefactType.SKILL]
            for skill in skills:
                title    = skill.get('title', 'untitled')
                category = skill.get('category', '')
                desc     = skill.get('description', '')
                cat_str  = f" [{category}]" if category else ""
                desc_str = f" — {desc}"     if desc     else ""
                summary_parts.append(f"🎓 Skill saved **{title}**{cat_str}{desc_str}.")

            # Widget anchors embedded in the (now empty) cleaned text won't be
            # present, but we can count them from the original masked_text.
            widget_count = len(re.findall(
                r'<lollms_inline\s+[^>]*>',
                masked_text,
            ))
            for _ in range(widget_count):
                summary_parts.append(
                    "🎛️ Interactive widget ready — use the controls below to explore the concept."
                )

            # Forms: detected by the has_form flag; we summarise based on the
            # lollms_form tags present in the original masked text rather than
            # relying on an artefact store entry (forms are not stored as
            # artefacts — they are rendered inline).
            if has_form:
                for fm in re.finditer(
                    r'<lollms_form\s+([^>]*)>',
                    masked_text,
                    re.IGNORECASE,
                ):
                    f_attrs  = _parse_attrs(fm.group(1))
                    f_title  = f_attrs.get('title', 'Form')
                    summary_parts.append(
                        f"📋 Form ready: **{f_title}** — please fill in the fields above."
                    )

            if summary_parts:
                cleaned = "\n".join(summary_parts)
                ASCIIColors.info(
                    "[silent-artifact guard] Auto-generated explanation appended.")

        return cleaned, affected_artefacts
