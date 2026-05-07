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
from ._artefacts import ArtefactType

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
 
  /* ── Each slide is a 16:9 canvas ── */
  .slide {
    position: relative;
    width: 1920px; height: 1080px;
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
    font-size: 20px; cursor: pointer; padding: 4px 8px;
    border-radius: 6px; transition: background 0.2s;
  }
  #viewer-controls button:hover { background: rgba(255,255,255,0.15); }
  #slide-counter { min-width: 60px; text-align: center; opacity: 0.8; }
 
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
  <button id="btn-ov"   title="Overview (Esc)">&#8942;</button>
</div>
 
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
 
  function show(n) {
    slides.forEach(function(s, i) { s.classList.toggle('active', i === n); });
    current = n;
    document.getElementById('slide-counter').textContent =
      (n + 1) + ' / ' + slides.length;
  }
 
  show(0);
 
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
 
  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  show((current+1) % slides.length);
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')    show((current-1+slides.length) % slides.length);
    if (e.key === 'Escape')    { if (ov.classList.contains('active')) toggleOverview(); }
    if (e.key === 'f' || e.key === 'F') document.getElementById('btn-fs').click();
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
        """
        Returns the system-prompt instructions for artifact operations.

        Includes:
          • LCP interception rules (what the user sees vs what is stored)
          • Full create / patch / rename / revert syntax
          • SURGICAL UPDATE POLICY — the core doctrine that drives patch preference
          • Artefact-image anchor documentation
        """
        lines = [
            "",
            "=== ARTIFACT SYSTEM ===",
            "",
            "**CRITICAL DISTINCTION — READ THIS FIRST**",
            "",
            "Artifacts and markdown code blocks are **completely different** things:",
            "",
            "• Markdown code blocks (```language ```) = temporary display only",
            "• Artifacts (<artifact> XML tags) = persistent, functional action",
            "",
            "1. **FUNCTIONAL INTERCEPTION & BLIND SPOTS**: When you emit functional tags",
            "   (<artifact>, <tool_call>, <lollms_inline>), the system INTERCEPTS them.",
            "   In your history, these will appear as `[BLIND_ACTION_EXECUTED]`. ",
            "   This means your action was SUCCESSFUL, but you cannot see the raw XML ",
            "   again. DO NOT attempt to rewrite the logs you see in history.",
            "2. **PROSE-ACTION HALLUCINATION GUARD**: Prose is NOT action. Markdown tables",
            "   are NOT action. If you explain a change but fail to emit the XML tag,",
            "   NOTHING happens. The system ignores your natural language instructions.",
            "3. **LOG MIMICRY IS A CRITICAL ERROR**: You are the Architect, not the system",
            "   logger. ",
            "   ❌ NEVER type out `<processing>` tags.",
            "   ❌ NEVER start lines with `* ` to describe internal steps.",
            "   ❌ NEVER mimic the `[BLIND_ACTION_EXECUTED]` marker.",
            "   Generating logs causes immediate generation failure. Only emit the ",
            "   raw functional XML tags required to perform the work.",
            "",
            "❌ NEVER wrap XML tags in code blocks — it breaks the system permanently:",
            "   ```xml",
            '   <artifact name="example.ext" type="type">',
            "   ...",
            "   </artifact>",
            "   ```",
            "",
            "✅ ALWAYS output the <artifact> tag **directly** as raw XML in your response.",
            "   Never wrap it inside any markdown code fence.",
            "",
            "Supported types: " + ", ".join(sorted(list(ArtefactType.ALL))),
            "Always choose the **most accurate type** for the content.",
            "",
            "=== ARTEFACT VISIBILITY: SOURCE OF TRUTH ===",
            "As the Architect, you are provided with a [Technical Synopsis] and a",
            "list of [Active Artefacts] below. ",
            "• If an artefact is listed in the 'Active Artefacts' zone, it is CURRENTLY",
            "  loaded in memory. Use this as your ONLY technical source of truth.",
            "• Ignore older conversational history if it conflicts with the Active Artefacts.",
            "",
            "=== ARTEFACT PRIORITY & ANTI-STALLING POLICY ===",
            "1. **STRICT HIERARCHY**: You must ALWAYS prefer updating existing artifacts",
            "   over creating new ones. Creating a widget or a new file to solve a",
            "   problem that belongs in an active artifact is a CRITICAL FAILURE.",
            "2. **ANTI-STALLING RULE**: Do NOT create a `<lollms_form>` to ask for ",
            "   choices if you have the technical knowledge to implement the feature.",
            "   If the user asks for a feature, use sensible engineering defaults and",
            "   IMPLEMENT IT in the artifact immediately. Forms are for data collection,",
            "   NOT for delaying work.",
            "3. **TARGET ANCHORING**: Look at the 'Active Artefacts' list below.",
            "   If a file name matches the domain of the user request (e.g., 'music',",
            "   'engine', 'code'), you MUST apply your changes to THAT file.",
            "",
            "=== HOW TO USE ARTEFACTS ===",
            "Artifact operations follow a MANDATORY two-step Hyper-Focus Protocol:",
            "",
            "Step 1: FORMULATE PLAN",
            "Before emitting an <artifact> tag, you MUST write a <coding_plan>. This block ",
            "summarizes the changes, identifies relevant sections/files, and explains the approach.",
            "Consult your WORKING MEMORY to ensure consistency with previously learned facts.",
            "This applies to everything: Code, Markdown documents, Notes, and Data.",
            "",
            "Syntax:",
            "<coding_plan>",
            "  - Goal: [Short summary of the update or creation]",
            "  - Specialist Persona: [Define the specific expertise needed, e.g., 'React Senior Developer', 'Legal Analyst', 'SVG Optimization Expert']",
            "  - Target Artifacts: [Comma-separated list of exact artifact names to be modified or created. ONLY these will be saved.]",
            "  - Implementation Details: [Logic steps for code, or section updates for text]",
            "  - Supporting Context: [List of artifacts containing reference info, like API docs or design specs]",
            "  - Relevant Memories: [List 8-char IDs of specific memories required for this task. Leave empty if none apply.]",
            "</coding_plan>",
            "",
            "Step 2: EXECUTION",
            "After the plan, you may emit the <artifact> tag. NOTE: The system will use a ",
            "Specialized Artifact Specialist to execute your plan in a clean environment.",
            "",
            "── Option A: CREATE (new artefact, full content)",
            '<artifact name="filename.ext" type="appropriate_type">',
            "Full content goes here...",
            "</artifact>",
            "",
            "── Option B: SURGICAL PATCH (targeted update — PREFERRED for existing artefacts)",
            '<artifact name="filename.ext" type="appropriate_type">',
            "<<<<<<< SEARCH",
            "exact old lines here — copy verbatim, including indentation",
            "=======",
            "new replacement lines here",
            ">>>>>>> REPLACE",
            "</artifact>",
            "",
            "   Multiple SEARCH/REPLACE blocks are allowed in a single tag.",
            "   Each block is applied independently in document order.",
            "",
            "── Option C: RENAME + UPDATE",
            '<artifact name="new_name.ext" type="appropriate_type" rename="old_name.ext">',
            "... full content or SEARCH/REPLACE blocks ...",
            "</artifact>",
            "",
            "── Option D: REVERT",
            '<artifact name="filename.ext" revert_to="v3" />',
            "",
"╔══════════════════════════════════════════════════════════════════╗",
            "║  SURGICAL UPDATE POLICY — READ BEFORE EVERY ARTEFACT EDIT       ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            "║                                                                  ║",
            "║  When modifying an EXISTING artefact you MUST use SEARCH/REPLACE ║",
            "║  patches UNLESS the change affects more than 60% of all lines.   ║",
            "║                                                                  ║",
            "║  Decision rule (apply in order):                                 ║",
            "║    1. Count lines that need changing.                            ║",
            "║    2. If changed_lines / total_lines > 0.60  → full rewrite OK. ║",
            "║    3. Otherwise                              → patch REQUIRED.   ║",
            "║                                                                  ║",
            "║  SEARCH block rules — HIGHEST PRIORITY:                         ║",
            "║    • Copy lines VERBATIM from the artefact, including ALL        ║",
            "║      leading spaces and indentation.                             ║",
            "║      Example — if the file has:                                  ║",
            "║        '        let audioCtx = null;'   (8 leading spaces)      ║",
            "║      your SEARCH block MUST have those exact 8 leading spaces.  ║",
            "║      Stripping indentation is the #1 cause of patch failures.   ║",
            "║    • Include ±2 unchanged context lines around every edit site   ║",
            "║      to make the match unambiguous.                              ║",
            "║    • NEVER add, remove, or alter inline comments inside a        ║",
            "║      SEARCH block — copy them exactly as-is.                    ║",
            "║    • NEVER add ellipses (...) or placeholder comments like       ║",
            "║      '// ... rest unchanged' inside a SEARCH block.             ║",
            "║    • NEVER collapse or add blank lines inside a SEARCH block.   ║",
            "║    • Each SEARCH block must be unique within the file; if the    ║",
            "║      same lines appear multiple times, widen the context until   ║",
            "║      the block is unique.                                        ║",
            "║                                                                  ║",
            "║  REPLACE block rules:                                            ║",
            "║    • Indentation in the REPLACE block is applied relative to     ║",
            "║      the indentation of the matched SEARCH block.               ║",
            "║    • You may write the REPLACE block at column 0 — the system   ║",
            "║      will re-indent it to match the file automatically.         ║",
            "║                                                                  ║",
            "║  If a patch is REJECTED (SEARCH did not match):                 ║",
            "║    • The system will show you the CURRENT file content.          ║",
            "║    • Widen the SEARCH context by ±3 additional lines and retry. ║",
            "║    • Do NOT switch to a full rewrite unless told to.             ║",
            "║                                                                  ║",
            "║  Verification: after every patch the system reports             ║",
            "║    '+N added / -M removed lines'. If you see '±0 lines' the     ║",
            "║    patch was rejected. You will be asked to reissue it.          ║",
            "║                                                                  ║",
            "║  ❌ NEVER include the sentinels (<<<<<<<, =======, >>>>>>>)     ║",
            "║     inside the code itself. They are for the PATCH FORMAT only.  ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            "=== ARTIFACT IMAGES ===",
            "",
            "Some artifacts (e.g. PDF documents) contain images embedded in their text.",
            "These images are referenced with self-closing anchor tags:",
            "",
            '    <artefact_image id="TITLE::N" />',
            "",
            "where TITLE is the artifact title and N is the 0-based image index.",
            "",
            "When such artifacts are active, the corresponding images are appended to",
            "the vision context **after** any user-supplied images.",
            "A mapping note in the scratchpad tells you which vision-input slot",
            "corresponds to which anchor id.",
            "",
            "Rules for artifacts with images:",
            "• Do NOT attempt to generate or modify artefact images via XML tags.",
            "  Images are supplied exclusively by the application layer.",
            "• When you see an <artefact_image id=\"...\" /> anchor in the artifact text,",
            "  look at the corresponding image slot in the vision input to understand",
            "  what that part of the document looks like.",
            "• You may reference an anchor in your reply to point the user to a specific",
            '  image, e.g.: \'As shown in <artefact_image id="my_doc::2" />, ...\'',
            "• When patching an artifact that contains image anchors, preserve the",
            "  anchor tags unchanged — do not remove or alter them.",
            "",
            "=== REMINDER ===",
            "→ Artifacts = persistent & versioned",
            "→ Markdown code blocks = temporary display only",
            "→ Never nest <artifact> inside ``` blocks",
            "→ PATCH existing artefacts surgically — full rewrites only when >60% changes",
            "→ SEARCH blocks must be verbatim copies of the current content",
            "→ Always add a human explanation outside the tag",
            "→ Preserve <artefact_image> anchors when patching image-bearing artifacts",
            "",
            "=== END ARTIFACT SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_image_generation_instructions(self) -> str:
        tti = getattr(self.lollmsClient, 'tti', None)
        if tti is None:
            return ""
        lines = [
            "",
            "=== IMAGE GENERATION / EDITING ===",
            "You can generate or edit images using the following XML tags.",
            "",
            "To generate a new image:",
            '<generate_image width="512" height="512">',
            "  A detailed description of the image you want to generate",
            "</generate_image>",
            "",
            "To edit an existing image artifact:",
            '<edit_image name="artifact_name">',
            "  Description of how to edit / modify the image",
            "</edit_image>",
            "=== END IMAGE INSTRUCTIONS ===",
            "",
        ]
        return "\n".join(lines)

    def _build_inline_widget_instructions(self) -> str:
        lines = [
            "",
            "=== INTERACTIVE WIDGET SYSTEM ===",
            "",
            "You can embed live, interactive HTML widgets directly in your replies.",
            "",
            "✅ WHEN TO USE A WIDGET:",
            "  • When an interactive visualization would make the concept much clearer",
            "  • To let the user experiment with parameters and see results immediately",
            "  • For teaching algorithms, math, physics, UI behavior, games, etc.",
            "",
            "❌ DO NOT use widgets for:",
            "  • Simple static explanations or text content",
            "  • Displaying code (use <artifact> or markdown code blocks instead)",
            "",
            "Tag syntax:",
            '<lollms_inline type="html" title="Clear descriptive title">',
            "<!DOCTYPE html>",
            "<html>...(complete self-contained HTML document)...</html>",
            "</lollms_inline>",
            "",
            "Rules:",
            "  • Content MUST be a valid, complete HTML5 document",
            "  • Only HTML + CSS + JavaScript — no Python, SQL, etc.",
            "  • Never wrap the HTML inside ```html code fences",
            "  • Always add explanatory text before or after the widget",
            "",
            "Supported types: html (default), react, svg",
            "",
            "=== END INTERACTIVE WIDGET SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_note_instructions(self) -> str:
        lines = [
            "",
            "=== NOTE SYSTEM ===",
            "",
            "Notes are **user-facing** persistent documents saved for the user to reference.",
            "",
            "✅ WHEN TO CREATE A NOTE:",
            "  • The user explicitly asks to save a note",
            "  • You produced analysis, comparisons, or key findings worth preserving",
            "  • Action items, decisions, or plans the user needs to track",
            "",
            "❌ DO NOT create a note for routine answers or one-off calculations.",
            "",
            "Tag syntax:",
            '<note title="Clear, descriptive title">',
            "Content here — plain text or Markdown",
            "</note>",
            "",
            "Always add a short explanation outside the tag.",
            "",
            "=== END NOTE SYSTEM ===",
            "",
        ]
        return "\n".join(lines)

    def _build_skill_instructions(self) -> str:
        lines = [
            "",
            "=== SKILL SYSTEM ===",
            "",
            "Skills are **LLM-facing** reusable knowledge capsules that persist across sessions.",
            "",
            "✅ WHEN TO CREATE A SKILL:",
            "  1. The user explicitly asks you to save it as a skill",
            "  2. You discovered a genuinely reusable technique or methodology",
            "",
            "❌ DO NOT create a skill for one-off solutions or non-generalizable content.",
            "",
            "Tag syntax:",
            '<skill title="Concise Skill Name"',
            '       description="One clear sentence"',
            '       category="domain/subdomain">',
            "Content here — Markdown with examples and usage guidelines",
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
        # ── Mask code blocks so XML inside documentation isn't processed ─────
        code_blocks: Dict[str, str] = {}

        def mask_code_block(match):
            placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
            code_blocks[placeholder] = match.group(0)
            return placeholder

        masked_text = re.sub(r'(`{3,})[\s\S]*?\1', mask_code_block, text)
        masked_text = re.sub(r'`[^`]+`',           mask_code_block, masked_text)

        has_artefact = bool(re.search(r'<(?:revert_)?art[ei]fact[\s>]', masked_text, re.IGNORECASE))
        has_gen      = enable_image_generation and bool(
            re.search(r'<generate_image[\s>]', masked_text, re.IGNORECASE))
        has_edit     = enable_image_editing and bool(
            re.search(r'<edit_image[\s>]', masked_text, re.IGNORECASE))
        has_note     = enable_notes and bool(
            re.search(r'<note[\s>]', masked_text, re.IGNORECASE))
        has_skill    = enable_skills and bool(
            re.search(r'<skill[\s>]', masked_text, re.IGNORECASE))
        has_form     = enable_forms and bool(
            re.search(r'<lollms_form[\s>]', masked_text, re.I))

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
            )

        # ── 2. Image generation → message.images ─────────────────────────────
        if has_gen:
            tti = getattr(self.lollmsClient, 'tti', None)
            if tti is None:
                ASCIIColors.warning(
                    "<generate_image> found but lollmsClient.tti is None — skipping.")
            else:
                gen_pattern = re.compile(
                    r'<generate_image\s*([^>]*)>(.*?)</generate_image>',
                    re.DOTALL | re.IGNORECASE,
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
                            ai_message.add_image_pack(
                                images=[img_b64],
                                group_type="generated",
                                active_by_default=True,
                                title=attrs.get('name', f'gen_{uuid.uuid4().hex[:6]}'),
                                prompt=prompt,
                            )
                            ASCIIColors.success(
                                f"Generated image ({width}×{height}) added to message.")
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
                    r'<edit_image\s*([^>]*)>(.*?)</edit_image>',
                    re.DOTALL | re.IGNORECASE,
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
                            ai_message.add_image_pack(
                                images=[edited_b64],
                                group_type="edited",
                                active_by_default=True,
                                title=f"edit_{uuid.uuid4().hex[:6]}",
                                prompt=prompt,
                            )
                            ASCIIColors.success("Edited image added to message.")
                    except Exception as e:
                        ASCIIColors.warning(f"Image edit failed: {e}")
                    return ''

                cleaned = edit_pattern.sub(handle_edit, cleaned)

        # ── 5. Notes ──────────────────────────────────────────────────────────
        if has_note:
            note_pattern = re.compile(
                r'<note\s*([^>]*)>(.*?)</note>',
                re.DOTALL | re.IGNORECASE,
            )

            def handle_note(match: re.Match) -> str:
                attrs   = _parse_attrs(match.group(1))
                content = match.group(2)

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
                return ''

            cleaned = note_pattern.sub(handle_note, cleaned)

        # ── 6. Skills ─────────────────────────────────────────────────────────
        if has_skill:
            skill_pattern = re.compile(
                r'<skill\s*([^>]*)>(.*?)</skill>',
                re.DOTALL | re.IGNORECASE,
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
                return ''

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
        if enable_silent_artefact_explanation and not cleaned:
            summary_parts: List[str] = []

            non_note_non_skill = [
                a for a in affected_artefacts
                if a.get('type') not in (ArtefactType.NOTE, ArtefactType.SKILL)
            ]
            for art in non_note_non_skill:
                atype     = art.get('type', 'artifact')
                title     = art.get('title', 'untitled')
                lang      = art.get('language', '')
                version   = art.get('version', 1)
                desc      = art.get('description', '')
                img_count = len(art.get('images') or [])
                lang_str  = f" ({lang})"       if lang           else ""
                ver_str   = f" — version {version}" if version > 1 else ""
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
                r'<lollms_widget\s+id=["\'][^"\']+["\']\s*/?>',
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