#!/usr/bin/env python3
"""
handbag_docs_example.py
=======================
End-to-end demonstration of the Handbag Bibliographic Documentation System.

Doctrine under test (NO vector RAG, NO embeddings):
  1. PersonalityStudio crafts a handbag (SOUL.md) from elements.
  2. Handbag.add_document() ingests documents (txt/md/pdf/docx/pptx)
     deterministically (no LLM) into a heading-chunked docs/ tree:
       - 1 document = 1 directory with INDEX.md (title, authors, abstract,
         ordered section map with token estimates).
       - A chunk = one complete heading section (title + full body).
  3. Two documents from UNRELATED domains are ingested on purpose:
     concept A (hybrid search) lives only in document A, concept B
     (replication lag policy) lives only in document B. The agent must
     keep them distinct — the anti-conflation doctrine.
  4. A budget-capped "HANDBAG DOCUMENTATION SCOPE" glimpse is injected into
     the system prompt (titles + authors + abstract digests + token counts).
  5. Five navigation tools are mounted (only because docs/ exists):
     tool_doc_index / tool_doc_load / tool_doc_search / tool_doc_peek /
     tool_doc_send_to_scratchpad.
  6. The agent navigates, sends source-stamped extracts to its scratchpad,
     and answers with [Source: docs/...] references.

Requirements
------------
pip install lollms_client ascii_colors
A configured LLM binding (run: python -m lollms_client.lollms_config_cli_env)
"""

import sys
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ascii_colors import ASCIIColors
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_personality import (
    LollmsPersonality,
    CapabilityFlags,
    Handbag,
    PersonalityStudio,
    DocNavigator,
    build_docs_scope_block,
)
from lollms_client.lollms_types import MSG_TYPE


# ── HANDBAG DIRECTORY ────────────────────────────────────────────────────────
HANDBAG_PATH = PROJECT_ROOT / "data_workspace" / "docs_handbag_demo"


# ── DOCUMENT CONTENTS ────────────────────────────────────────────────────────
# Document A: Markdown paper WITH frontmatter → tests frontmatter metadata
# extraction (title/author) + heading-based chunking (H1/H2/H3 tree).
HYBRID_SEARCH_PAPER = """---
title: "SafeStore Hybrid Search Architecture"
author: "S. Parisot"
---

# SafeStore Hybrid Search Architecture

## Abstract
SafeStore combines sparse BM25 retrieval with dense vector search behind a
single hybrid query interface. The sparse path guarantees exact keyword
recall for identifiers and rare terms, while the dense path captures
semantic similarity. A reciprocal rank fusion layer merges both rankings.

## Sparse Retrieval
The sparse path uses Okapi BM25 with k1=1.2 and b=0.75. Exact identifiers
such as `E500A` or table names are always found by this path, which is why
the sparse stage is never disabled in production.

### Term Weighting
Terms are weighted by inverse document frequency; stop words are retained
in the index but receive near-zero weight.

## Dense Retrieval
The dense path embeds queries with a small local model and searches an
associated vector collection. It captures paraphrases and cross-lingual
matches that lexical search cannot see.

## Rank Fusion
Reciprocal Rank Fusion (RRF) with k=60 merges the two rankings. This makes
the system robust to either path failing: if the vector index is offline,
the sparse ranking alone is still served.
"""

# Document B: plain TXT runbook WITHOUT frontmatter → tests deterministic
# heuristics: first line = title, "by X" = authors, ALL-CAPS lines and
# "CHAPTER N" patterns = headings. Deliberately an UNRELATED domain.
CLUSTER_RUNBOOK_TXT = """PostgreSQL Cluster Operations Runbook
by Ops Team Alpha

CHAPTER 1 Cluster Policy
The maximum allowed replication lag is 3000 milliseconds.
Any latency exceeding 5000ms triggers an automatic P1 alert escalation.
Failover to the standby replica is mandatory during a P0 outage.

CHAPTER 2 Verification Commands
LAG CHECK: SELECT now() - pg_last_xact_replay_timestamp();
PROMOTE: pg_ctl promote -D /var/lib/postgresql/data
Always verify the standby node state before issuing a failover command.

CHAPTER 3 Severity Classification
P0 means total service outage or data corruption. Immediate rollback.
P1 means degraded performance affecting more than 20 percent of users.
P2 means a non-blocking bug or partial redundancy loss.
"""


SOUL_CONTENT = """You are DocLibrarian, a precise research assistant.
You carry a private documentation library in your handbag. When a question
touches your library's topics:
1. Call tool_doc_index("") to see the library, or tool_doc_search to locate
   the right document.
2. Load exactly what you need with tool_doc_load (or tool_doc_peek for big files).
3. Send the relevant extracts to your scratchpad with tool_doc_send_to_scratchpad
   (never rewrite content manually).
4. Answer citing each fact with its [Source: docs/...] path. NEVER merge facts
   from different documents into one continuous narrative — each source stays
   clearly attributed.
End with <done/> when finished.
"""


def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


def verify_ingestion(studio: PersonalityStudio) -> None:
    """Offline doctrine verification — no LLM required."""
    ASCIIColors.rule("[bold blue]🧪 OFFLINE DOCTRINE VERIFICATION (no LLM)[/bold blue]")

    navigator = DocNavigator(HANDBAG_PATH / "docs")

    # 1. Both documents became directories with INDEX.md
    docs = navigator.list_documents()
    titles = {d["title"] for d in docs}
    ASCIIColors.cyan(f"Documents discovered: {len(docs)}")
    for d in docs:
        ASCIIColors.info(
            f'   • "{d["title"]}" — {d["authors"]} '
            f"(~{d['tokens']:,} tokens, {d['sections']} sections) → {d['rel']}/"
        )
    assert len(docs) == 2, f"Expected 2 documents, found {len(docs)}"
    assert "SafeStore Hybrid Search Architecture" in titles, "Markdown paper title not extracted"
    assert "PostgreSQL Cluster Operations Runbook" in titles, "TXT runbook title not extracted"
    ASCIIColors.green("   ✅ Metadata extraction (md frontmatter + txt heuristics)")

    # 2. Abstract digests present in the glimpse
    scope = build_docs_scope_block(navigator)
    assert "HANDBAG DOCUMENTATION SCOPE" in scope
    assert "Abstract digest" in scope
    ASCIIColors.panel(scope, title="[bold]System-Prompt Glimpse (budget-capped)[/bold]", border_style="blue")

    # 3. Heading-chunked tree: document A must have subsection files
    paper_dir = HANDBAG_PATH / "docs" / "safestore_hybrid_search_architecture"
    chunk_files = sorted(p.name for p in paper_dir.rglob("*.md") if p.name != "INDEX.md")
    ASCIIColors.cyan(f"Chunk tree of the paper: {chunk_files}")
    assert any("dense_retrieval" in n for n in chunk_files), "Heading chunking failed"
    assert any("term_weighting" in n for n in chunk_files), "H3 nesting failed"

    # 4. Anti-conflation search: each concept is found ONLY in its own document
    lag_hits = navigator.search("replication lag")
    search_hits = navigator.search("dense vector search")
    assert "cluster_runbook" in lag_hits.lower() or "postgresql" in lag_hits.lower(), "Lag policy not found in runbook"
    assert "safestore" in search_hits.lower(), "Hybrid search not found in paper"
    assert "SOURCE GROUP" in lag_hits and "SOURCE GROUP" in search_hits
    ASCIIColors.green("   ✅ Search results grouped by document (per-source banners)")

    # 5. Over-budget graceful degradation
    tiny_limit = navigator.load("safestore_hybrid_search_architecture", max_chars=200)
    assert "TOO LARGE" in tiny_limit and "index" in tiny_limit.lower()
    ASCIIColors.green("   ✅ Over-budget load degrades to index (no hard failure)")

    # 6. Scratchpad block carries full provenance
    block = navigator.build_scratchpad_block(
        "safestore_hybrid_search_architecture", note="For the report"
    )
    assert "[DOC EXTRACT]" in block and "[Source:" not in block  # banner uses SOURCE:
    assert "=== SOURCE: docs/safestore" in block
    ASCIIColors.green("   ✅ Scratchpad extracts carry full source banners")
    ASCIIColors.success("\n✅ ALL OFFLINE DOCTRINE CHECKS PASSED\n")


def main():
    ASCIIColors.panel(
        "[bold]Handbag Bibliographic Documentation System — End-to-End Test[/bold]\n"
        "[dim]Hierarchical navigation + provenance-native extracts. No vector RAG, no embeddings.[/dim]",
        title="[bold green]🎒📚 HANDBAG DOCS DEMO[/bold green]",
        border_style="green",
    )

    # ── 1. Craft the handbag with the Studio ────────────────────────────────
    if HANDBAG_PATH.exists():
        shutil.rmtree(HANDBAG_PATH, ignore_errors=True)

    ASCIIColors.info("[1/5] Crafting handbag with PersonalityStudio...")
    studio = PersonalityStudio(HANDBAG_PATH)
    studio.set_soul(
        name="DocLibrarian",
        system_prompt=SOUL_CONTENT,
        author="lollms-client",
        category="research",
        description="Demonstrates the handbag documentation navigation system.",
    )
    studio.set_manifest(name="Docs Demo Handbag", skills_mode="mixed")

    # ── 2. Ingest two documents from UNRELATED domains (deterministic) ─────
    ASCIIColors.info("[2/5] Ingesting documents (deterministic, no LLM)...")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="lollms_docs_"))
    try:
        paper_file = tmp / "hybrid_search_paper.md"
        paper_file.write_text(HYBRID_SEARCH_PAPER, encoding="utf-8")
        runbook_file = tmp / "cluster_runbook.txt"
        runbook_file.write_text(CLUSTER_RUNBOOK_TXT, encoding="utf-8")

        doc_a = studio.add_document(paper_file, use_llm=False)
        doc_b = studio.add_document(runbook_file, use_llm=False)
        ASCIIColors.green(f"   • Paper   → {doc_a.relative_to(HANDBAG_PATH)}")
        ASCIIColors.green(f"   • Runbook → {doc_b.relative_to(HANDBAG_PATH)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── 3. Offline doctrine verification (works without a live LLM) ────────
    verify_ingestion(studio)

    # ── 4. Build the personality and check the mounted tools ───────────────
    ASCIIColors.info("[4/5] Building personality and connecting to the LLM...")
    try:
        client = get_client_from_env(create_llm=True)
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration error: {e}")
        ASCIIColors.yellow("Run 'python -m lollms_client.lollms_config_cli_env' to configure your connection.")
        sys.exit(1)

    agent = studio.build(lollms_client=client)
    agent.capabilities = CapabilityFlags(
        enable_code_execution=False,
        enable_workspace_tools=False,
        enable_sub_agents=False,
        enable_skill_loading=True,
        skills_mode="mixed",
    )
    agent.max_tokens_per_turn = 4096

    doc_tool_names = sorted(
        t["name"] for t in agent.list_tools_structured() if t["name"].startswith("tool_doc_")
    )
    ASCIIColors.cyan(f"Mounted documentation tools: {doc_tool_names}")
    assert len(doc_tool_names) == 5, f"Expected 5 doc tools, found {doc_tool_names}"

    # ── 5. Live run: two concepts, two sources, zero conflation ────────────
    task_prompt = (
        "I need two things from your documentation library:\n"
        "1. What is the maximum allowed replication lag in our PostgreSQL cluster policy,\n"
        "   and what happens if latency exceeds 5000ms?\n"
        "2. What are the two retrieval paths of the SafeStore hybrid search architecture,\n"
        "   and what makes the sparse path indispensable?\n"
        "Navigate your library, send the relevant extracts to your scratchpad with their\n"
        "sources, then answer. Attribute every fact to its [Source: docs/...] path and\n"
        "keep the two documents clearly separate. End with <done/>."
    )
    ASCIIColors.panel(task_prompt, title="[bold yellow]📝 User Task[/bold yellow]", border_style="yellow")
    ASCIIColors.rule("[bold green]🤖 Agent Deliberation & Execution Stream[/bold green]")

    result = agent.chat(
        prompt=task_prompt,
        streaming_callback=streaming_callback,
        max_reasoning_steps=12,
        temperature=0.2,
    )

    # ── Verification report ─────────────────────────────────────────────────
    print("\n\n")
    ASCIIColors.rule("[bold cyan]📊 VERIFICATION REPORT[/bold cyan]")

    tool_calls = result.get("tool_calls", [])
    called_names = [tc.get("name", "") for tc in tool_calls]
    ASCIIColors.cyan(f"Tools called: {called_names}")

    checks = {
        "Agent used doc navigation tools": any(n.startswith("tool_doc_") for n in called_names),
        "Agent searched or indexed the library": any(
            n in ("tool_doc_index", "tool_doc_search") for n in called_names
        ),
        "Agent sent an extract to the scratchpad": "tool_doc_send_to_scratchpad" in called_names,
    }

    scratchpad_path = getattr(agent, "_scratchpad_path", None)
    scratchpad_content = ""
    if scratchpad_path and Path(scratchpad_path).exists():
        scratchpad_content = Path(scratchpad_path).read_text(encoding="utf-8", errors="ignore")
        checks["Scratchpad contains [DOC EXTRACT] blocks"] = "[DOC EXTRACT]" in scratchpad_content
        checks["Scratchpad keeps separate source banners"] = scratchpad_content.count("=== SOURCE: docs/") >= 2
        checks["Both documents represented in scratchpad"] = (
            "postgresql" in scratchpad_content.lower()
            and "safestore" in scratchpad_content.lower()
        )

    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            ASCIIColors.green(f"   ✅ {check_name}")
        else:
            ASCIIColors.red(f"   ❌ {check_name}")
            all_passed = False

    if scratchpad_content:
        ASCIIColors.panel(
            scratchpad_content[:1500],
            title="[bold]📝 Scratchpad (source-stamped extracts)[/bold]",
            border_style="magenta",
        )

    summary_table = ASCIIColors.table(
        "Metric", "Value",
        rows=[
            ["Total Reasoning Rounds", str(result.get("rounds", 0))],
            ["Tool Calls Made", str(len(tool_calls))],
            ["Doc Tool Calls", str(sum(1 for n in called_names if n.startswith("tool_doc_")))],
            ["Was Cancelled", str(result.get("was_cancelled", False))],
        ],
        title="[bold]Session Metrics[/bold]",
        box="round",
    )
    ASCIIColors.rich_print(summary_table)

    if all_passed:
        ASCIIColors.success("\n🎉 ALL CHECKS PASSED: hierarchical navigation + provenance-native extraction work end-to-end!")
    else:
        ASCIIColors.error("\n⚠️ SOME CHECKS FAILED — review the tool calls and scratchpad above.")

    if HANDBAG_PATH.exists():
        shutil.rmtree(HANDBAG_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()