"""
memory_explorer.py — Interactive GUI explorer and manager for agent memories.
Provides tiered inspection, database switching, cross-owner filtering,
pagination, sorting, manual curation, hard purge, dream cycle consolidation,
an interactive Tag Graph visualizer with K-Hop neighborhood filtering, and
an automated Redundancy Auditor to detect and merge duplicate memories.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from nicegui import ui

__all__ = ["open_memory_explorer_dialog"]

LEVEL_NAMES = {
    1: "Working (L1)",
    2: "Deep (L2)",
    3: "Archived (L3)",
    4: "Episodic (L4)",
}

LEVEL_COLORS = {
    1: "emerald",
    2: "amber",
    3: "slate",
    4: "purple",
}

LEVEL_HEX_COLORS = {
    1: "#10b981",  # emerald-500
    2: "#f59e0b",  # amber-500
    3: "#64748b",  # slate-500
    4: "#8b5cf6",  # purple-500
}

SORT_OPTIONS = {
    "importance_desc": "Importance (High → Low)",
    "importance_asc": "Importance (Low → High)",
    "date_desc": "Date (Newest first)",
    "date_asc": "Date (Oldest first)",
    "centrality_desc": "Centrality (High → Low)",
    "use_count_desc": "Most Used",
}

PAGE_SIZES = [10, 25, 50, 100, 0]


def discover_available_databases(prefs: Any) -> Dict[str, str]:
    """Finds all available SQLite memory databases in project, global, and handbag folders."""
    databases = {}
    ws_path = getattr(prefs, "workspace_path", None)
    if ws_path:
        project_db = Path(ws_path) / ".lollms_code" / "memory" / "memory.db"
        if project_db.exists():
            databases["Project Memory"] = str(project_db.resolve())

    global_db = Path.home() / ".lollms_client" / "lollms_code" / "memory.db"
    if global_db.exists():
        databases["Global User Memory"] = str(global_db.resolve())

    handbag_path = getattr(prefs, "handbag_path", None)
    if handbag_path:
        hb_db = Path(handbag_path) / "memory" / "memory.db"
        if hb_db.exists():
            databases["Handbag Memory"] = str(hb_db.resolve())

    if not databases and ws_path:
        fallback = Path(ws_path) / ".lollms_code" / "memory" / "memory.db"
        databases["Project Memory"] = str(fallback.resolve())

    return databases


def _get_tags_set(m: Dict[str, Any]) -> Set[str]:
    raw = m.get("tags") or ""
    return set(t.strip().lower() for t in raw.split(",") if t.strip())


def find_redundant_memories(
    memories: List[Dict[str, Any]],
    tag_threshold: float = 0.35,
    text_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Scans a collection of memories and detects suspected duplicates / redundant pairs
    using semantic tag overlap and text sequence similarity.
    """
    candidates: List[Dict[str, Any]] = []
    n = len(memories)

    for i in range(n):
        m1 = memories[i]
        tags1 = _get_tags_set(m1)
        text1 = (m1.get("content") or "").strip().lower()
        if not text1:
            continue
        words1 = set(re.findall(r'\b\w{3,}\b', text1))

        for j in range(i + 1, n):
            m2 = memories[j]
            tags2 = _get_tags_set(m2)
            text2 = (m2.get("content") or "").strip().lower()
            if not text2:
                continue
            words2 = set(re.findall(r'\b\w{3,}\b', text2))

            # 1. Tag overlap and Jaccard
            shared_tags = tags1 & tags2
            union_tags = tags1 | tags2
            tag_jaccard = len(shared_tags) / max(1, len(union_tags)) if union_tags else 0.0

            # 2. Text overlap and sequence matching
            word_intersection = words1 & words2
            word_union = words1 | words2
            word_jaccard = len(word_intersection) / max(1, len(word_union)) if word_union else 0.0

            if word_jaccard >= 0.3 or tag_jaccard >= 0.35 or len(shared_tags) >= 2:
                seq_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
            else:
                seq_ratio = word_jaccard

            # Redundancy criteria
            is_suspect = (
                seq_ratio >= 0.70
                or (seq_ratio >= 0.45 and (len(shared_tags) >= 1 or tag_jaccard >= tag_threshold))
                or (word_jaccard >= 0.50 and len(shared_tags) >= 1)
            )

            if is_suspect:
                confidence = max(seq_ratio, (seq_ratio * 0.6 + tag_jaccard * 0.4))
                candidates.append({
                    "memory_a": m1,
                    "memory_b": m2,
                    "shared_tags": sorted(list(shared_tags)),
                    "text_similarity": round(seq_ratio, 2),
                    "tag_jaccard": round(tag_jaccard, 2),
                    "confidence": round(confidence, 2),
                })

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def get_k_hop_subgraph(
    center_memory_id: str,
    memories: List[Dict[str, Any]],
    max_hops: int = 1,
) -> Set[str]:
    """
    Returns the set of memory IDs within `max_hops` of `center_memory_id`
    based on shared tags connectivity.
    """
    mem_tags = {m.get("id", ""): _get_tags_set(m) for m in memories}
    adj: Dict[str, Set[str]] = {m.get("id", ""): set() for m in memories}
    mem_ids = list(mem_tags.keys())

    for i in range(len(mem_ids)):
        id_a = mem_ids[i]
        tags_a = mem_tags[id_a]
        if not tags_a:
            continue
        for j in range(i + 1, len(mem_ids)):
            id_b = mem_ids[j]
            tags_b = mem_tags[id_b]
            if tags_a & tags_b:
                adj[id_a].add(id_b)
                adj[id_b].add(id_a)

    if center_memory_id not in adj:
        return {center_memory_id}

    visited: Set[str] = {center_memory_id}
    current_level = {center_memory_id}

    for _ in range(max_hops):
        next_level = set()
        for nid in current_level:
            for neighbor in adj.get(nid, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
        current_level = next_level
        if not current_level:
            break

    return visited


def build_memory_graph_options(
    memories: List[Dict[str, Any]],
    mode: str = "hubs",
    repulsion: int = 250,
    search_query: str = "",
    focused_id: Optional[str] = None,
    focused_hop_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Builds an Apache ECharts graph configuration connecting memories via shared tags.
    Supports 'hubs' mode (bipartite tag attractor hubs) and 'direct' mode (edges between memories).
    """
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    categories = [
        {"name": "Working (L1)", "itemStyle": {"color": LEVEL_HEX_COLORS[1]}},
        {"name": "Deep (L2)", "itemStyle": {"color": LEVEL_HEX_COLORS[2]}},
        {"name": "Archived (L3)", "itemStyle": {"color": LEVEL_HEX_COLORS[3]}},
        {"name": "Episodic (L4)", "itemStyle": {"color": LEVEL_HEX_COLORS[4]}},
    ]

    tag_to_memories: Dict[str, List[Dict[str, Any]]] = {}
    memory_parsed_tags: Dict[str, Set[str]] = {}

    search_lower = search_query.lower().strip()

    # Filter memories by focused hop set if active
    visible_memories = memories
    if focused_hop_ids is not None:
        visible_memories = [m for m in memories if m.get("id") in focused_hop_ids]

    for m in visible_memories:
        mid = m.get("id", "")
        tags = _get_tags_set(m)
        memory_parsed_tags[mid] = tags
        for t in tags:
            tag_to_memories.setdefault(t, []).append(m)

        short_id = mid[:8]
        content = m.get("content", "")
        snippet = content[:32] + "…" if len(content) > 32 else content
        level = m.get("level", 1)
        importance = float(m.get("importance", 0.0) or 0.0)
        color = LEVEL_HEX_COLORS.get(level, "#94a3b8")

        node_size = max(20, min(50, int(importance * 28) + 22))
        is_focused = (mid == focused_id)
        if is_focused:
            node_size = max(node_size, 45)

        # Search match highlighting
        is_match = True
        if search_lower:
            is_match = (
                search_lower in content.lower()
                or search_lower in (m.get("tags") or "").lower()
                or search_lower in short_id.lower()
            )

        nodes.append({
            "id": mid,
            "name": f"[{short_id}] {snippet}",
            "value": round(importance, 2),
            "symbolSize": node_size,
            "category": LEVEL_NAMES.get(level, f"Level {level}"),
            "itemStyle": {
                "color": color,
                "opacity": 1.0 if is_match else 0.25,
                "borderColor": "#38bdf8" if is_focused else ("#ffffff" if is_match and search_lower else "#00000000"),
                "borderWidth": 4 if is_focused else (2 if is_match and search_lower else 0),
            },
            "label": {
                "show": is_focused,
                "formatter": f"[{short_id}]",
                "position": "top",
            },
            "is_memory": True,
            "memory_id": mid,
        })

    if mode == "hubs":
        categories.append({"name": "Tag Hubs", "itemStyle": {"color": "#6366f1"}})
        for tag, connected_mems in tag_to_memories.items():
            count = len(connected_mems)
            tag_node_id = f"tag::{tag}"
            hub_size = max(18, min(42, 16 + count * 4))

            is_tag_match = True
            if search_lower:
                is_tag_match = (search_lower in tag or any(search_lower in (m.get("content") or "").lower() for m in connected_mems))

            nodes.append({
                "id": tag_node_id,
                "name": f"#{tag}",
                "value": count,
                "symbol": "diamond",
                "symbolSize": hub_size,
                "category": "Tag Hubs",
                "itemStyle": {
                    "color": "#6366f1",
                    "opacity": 1.0 if is_tag_match else 0.25,
                },
                "label": {
                    "show": True,
                    "position": "bottom",
                    "fontSize": 11,
                },
                "is_memory": False,
                "tag_name": tag,
            })

            for m in connected_mems:
                links.append({
                    "source": m.get("id", ""),
                    "target": tag_node_id,
                    "lineStyle": {
                        "width": 1.5,
                        "color": "#94a3b8",
                        "opacity": 0.45 if is_tag_match else 0.15,
                    },
                })
    else:
        # Direct Memory-to-Memory Links
        mem_ids = list(memory_parsed_tags.keys())
        linked_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(mem_ids)):
            id_a = mem_ids[i]
            tags_a = memory_parsed_tags[id_a]
            if not tags_a:
                continue

            for j in range(i + 1, len(mem_ids)):
                id_b = mem_ids[j]
                tags_b = memory_parsed_tags[id_b]
                common = tags_a & tags_b
                if common:
                    pair_key = (min(id_a, id_b), max(id_a, id_b))
                    if pair_key not in linked_pairs:
                        linked_pairs.add(pair_key)
                        strength = len(common)
                        links.append({
                            "source": id_a,
                            "target": id_b,
                            "value": strength,
                            "lineStyle": {
                                "width": min(6, 1 + strength * 1.5),
                                "color": "#6366f1",
                                "opacity": min(0.85, 0.35 + strength * 0.15),
                                "curveness": 0.08,
                            },
                        })

    return {
        "tooltip": {
            "trigger": "item",
            "confine": True,
        },
        "legend": {
            "data": [c["name"] for c in categories],
            "orient": "horizontal",
            "top": 6,
            "textStyle": {"fontSize": 11},
        },
        "series": [{
            "type": "graph",
            "layout": "force",
            "animation": False,
            "roam": True,
            "draggable": True,
            "categories": categories,
            "data": nodes,
            "links": links,
            "emphasis": {
                "focus": "adjacency",
                "lineStyle": {"width": 4},
            },
            "force": {
                "repulsion": repulsion,
                "edgeLength": [50, 150] if mode == "hubs" else [80, 200],
                "gravity": 0.12,
            },
        }],
    }


def open_memory_explorer_dialog(session: Any, prefs: Any) -> None:
    """Opens the Memory Explorer modal dialog with Card List and Tag Graph views."""
    try:
        session.ensure_ready()
    except Exception as e:
        ui.notify(f"Could not prepare agent session: {e}", type="negative")
        return

    from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig

    personality = getattr(session, "personality", None)
    available_dbs = discover_available_databases(prefs)

    active_db_key = {"key": next(iter(available_dbs.keys())) if available_dbs else "Project Memory"}
    active_manager_holder: Dict[str, Optional[LollmsMemoryManager]] = {
        "manager": getattr(personality, "memory_manager", None)
    }

    if not active_manager_holder["manager"] and available_dbs:
        first_db = next(iter(available_dbs.values()))
        active_manager_holder["manager"] = LollmsMemoryManager(
            db_path=f"sqlite:///{first_db}",
            config=MemoryConfig(working_token_budget=2000)
        )

    dialog = ui.dialog().props("maximized")

    with dialog, ui.card().classes("w-full h-full flex flex-col p-4 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100"):
        # ---- Top Controls Row ----
        with ui.row().classes("w-full items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-800 flex-wrap gap-2"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("psychology", size="30px").classes("text-primary")
                with ui.column().classes("gap-0"):
                    ui.label("Memory Explorer").classes("text-lg font-bold text-slate-900 dark:text-slate-100")
                    db_path_label = ui.label("").classes("text-xs text-slate-500 font-mono truncate max-w-xl")

            with ui.row().classes("items-center gap-2 flex-wrap"):
                # View Mode Switcher
                view_mode_toggle = ui.toggle(
                    {"list": "📋 List View", "graph": "🕸️ Tag Graph"},
                    value="list",
                ).props("dense unelevated size=sm").classes("text-xs")

                db_select = ui.select(
                    list(available_dbs.keys()),
                    value=active_db_key["key"],
                    label="Database"
                ).classes("w-44 text-xs").props("outlined dense options-dense")

                ui.button("Add Memory", icon="add", on_click=lambda: open_add_memory_dialog(active_manager_holder["manager"], reload_memories)).props(
                    "outline dense size=sm no-caps color=primary"
                ).tooltip("Manually add a memory to the selected database")

                ui.button("Audit Redundancies", icon="auto_fix_high", on_click=lambda: open_redundancy_auditor_dialog(active_manager_holder["manager"], reload_memories)).props(
                    "outline dense size=sm no-caps color=amber"
                ).tooltip("Detect and merge suspected redundant memories using shared tags & text similarity")

                ui.button("Purge 0% Faded", icon="delete_sweep", on_click=lambda: confirm_purge_faded(active_manager_holder["manager"], reload_memories)).props(
                    "outline dense size=sm no-caps color=red"
                ).tooltip("Permanently delete all soft-deleted and 0% importance memories")

                ui.button("Run Dream Cycle", icon="bedtime", on_click=lambda: trigger_dream_cycle(active_manager_holder["manager"], session.client, reload_memories)).props(
                    "outline dense size=sm no-caps color=purple"
                ).tooltip("Trigger synaptic consolidation pass (decay recalculation, centrality re-indexing, fusion)")

                ui.button("Refresh", icon="refresh", on_click=lambda: reload_memories()).props(
                    "flat dense size=sm no-caps"
                )
                ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

        # ---- Filters & Parameters Row ----
        filter_state = {
            "level": None,
            "search": "",
            "owner": "__all__",
            "sort": "importance_desc",
            "page": 1,
            "page_size": 25,
            "total_pages": 1,
            "graph_mode": "hubs",
            "graph_repulsion": 260,
            "focused_node_id": None,
            "focused_hops": 0,
            "focused_hop_ids": None,
        }

        with ui.row().classes("w-full items-center justify-between py-2 gap-3 flex-wrap border-b border-slate-200 dark:border-slate-800"):
            # Level Filter Buttons
            with ui.row().classes("items-center gap-1"):
                def set_level(lvl: Optional[int]):
                    filter_state["level"] = lvl
                    filter_state["page"] = 1
                    for btn_lvl, b in level_buttons.items():
                        if btn_lvl == lvl:
                            b.props("color=primary unelevated")
                        else:
                            b.props("flat color=grey")
                    reload_memories()

                level_buttons = {}
                b_all = ui.button("All", on_click=lambda: set_level(None)).props("dense size=sm unelevated color=primary no-caps")
                level_buttons[None] = b_all

                for lvl, name in LEVEL_NAMES.items():
                    btn = ui.button(name, on_click=lambda l=lvl: set_level(l)).props("dense size=sm flat color=grey no-caps")
                    level_buttons[lvl] = btn

            # Search, Owner, Sort controls
            with ui.row().classes("items-center gap-2 flex-wrap"):
                owner_select = ui.select(
                    {"__all__": "All Owners / Projects"},
                    value="__all__",
                    label="Owner Filter"
                ).classes("w-44 text-xs").props("outlined dense options-dense")

                def on_owner_change(e):
                    filter_state["owner"] = e.value
                    filter_state["page"] = 1
                    reload_memories()

                owner_select.on_value_change(on_owner_change)

                sort_select = ui.select(
                    SORT_OPTIONS,
                    value=filter_state["sort"],
                    label="Sort By"
                ).classes("w-44 text-xs").props("outlined dense options-dense")

                def on_sort_change(e):
                    filter_state["sort"] = e.value
                    filter_state["page"] = 1
                    reload_memories()

                sort_select.on_value_change(on_sort_change)

                search_input = ui.input(placeholder="Search content or tags…").classes("w-56 bg-white dark:bg-slate-800 text-xs").props(
                    "dense outlined clearable input-debounce=300"
                )
                search_input.on("clear", lambda: on_search_change(""))
                search_input.on_value_change(lambda e: on_search_change(e.value))

                def on_search_change(val: Optional[str]):
                    filter_state["search"] = (val or "").strip()
                    filter_state["page"] = 1
                    reload_memories()

        # ---- Statistics & Pagination Bar (List View) ----
        list_nav_bar = ui.row().classes("w-full items-center justify-between px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded text-xs font-mono border border-slate-200 dark:border-slate-700")
        with list_nav_bar:
            stats_label = ui.label("Loading stats...").classes("text-slate-700 dark:text-slate-300 font-semibold")

            with ui.row().classes("items-center gap-2"):
                page_info_label = ui.label("").classes("text-slate-600 dark:text-slate-400 font-medium")

                page_size_select = ui.select(
                    {10: "10 / page", 25: "25 / page", 50: "50 / page", 100: "100 / page", 0: "Show All"},
                    value=filter_state["page_size"],
                    label="Page Size"
                ).classes("w-28 text-xs").props("dense options-dense")

                def on_page_size_change(e):
                    filter_state["page_size"] = int(e.value)
                    filter_state["page"] = 1
                    reload_memories()

                page_size_select.on_value_change(on_page_size_change)

                prev_btn = ui.button(icon="chevron_left", on_click=lambda: change_page(-1)).props("flat dense round size=xs")
                next_btn = ui.button(icon="chevron_right", on_click=lambda: change_page(1)).props("flat dense round size=xs")

        def change_page(delta: int):
            new_p = filter_state["page"] + delta
            if 1 <= new_p <= filter_state["total_pages"]:
                filter_state["page"] = new_p
                reload_memories()

        # ---- Graph Controls Bar (Graph View Only) ----
        graph_controls_bar = ui.row().classes("w-full items-center justify-between px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded text-xs font-mono border border-slate-200 dark:border-slate-700 flex-wrap gap-2")
        graph_controls_bar.visible = False
        with graph_controls_bar:
            with ui.row().classes("items-center gap-2"):
                ui.label("Topology:").classes("text-slate-600 dark:text-slate-400 font-bold")
                graph_mode_select = ui.select(
                    {"hubs": "Tag Hub Clusters", "direct": "Direct Memory Links"},
                    value=filter_state["graph_mode"],
                ).classes("w-48 text-xs").props("dense options-dense")

                def on_graph_mode_change(e):
                    filter_state["graph_mode"] = e.value
                    refresh_graph_view()

                graph_mode_select.on_value_change(on_graph_mode_change)

            # Focus / Neighborhood Indicator
            with ui.row().classes("items-center gap-1 text-xs"):
                focus_label = ui.label("Showing All Graph").classes("text-slate-700 dark:text-slate-300 font-semibold")
                clear_focus_btn = ui.button("Clear Focus", icon="cancel", on_click=lambda: clear_node_focus()).props("dense flat size=xs color=red no-caps")
                clear_focus_btn.visible = False

            with ui.row().classes("items-center gap-2"):
                ui.label("Repulsion:").classes("text-slate-600 dark:text-slate-400")
                repulsion_slider = ui.slider(min=100, max=600, step=20, value=filter_state["graph_repulsion"]).classes("w-32").props("dense")

                def on_repulsion_change(e):
                    filter_state["graph_repulsion"] = int(e.value)
                    refresh_graph_view()

                repulsion_slider.on_value_change(on_repulsion_change)

        # ---- Main Body: Split between List View and Graph View ----
        main_content_area = ui.row().classes("w-full flex-1 min-h-0 mt-2 gap-3 overflow-hidden flex-nowrap")

        with main_content_area:
            # 1. Card List Area
            scroll_area = ui.scroll_area().classes("w-full h-full flex-1 pr-2")
            with scroll_area:
                memories_list = ui.column().classes("w-full gap-2")

            # 2. Graph Area
            graph_container = ui.row().classes("w-full h-full flex-1 min-h-[500px] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-white dark:bg-slate-950 flex-nowrap relative")
            graph_container.visible = False

            with graph_container:
                graph_chart_slot = ui.column().classes("flex-1 h-full min-h-[480px] p-0")

                # Graph Side Inspector Drawer (for clicked nodes)
                graph_inspector = ui.column().classes("w-96 h-full border-l border-slate-200 dark:border-slate-800 p-4 bg-slate-50 dark:bg-slate-900 overflow-y-auto gap-3 shrink-0")
                graph_inspector.visible = False

        # Manage view toggle transitions
        def on_view_toggle_change(e):
            is_graph = (e.value == "graph")
            scroll_area.visible = not is_graph
            list_nav_bar.visible = not is_graph
            graph_container.visible = is_graph
            graph_controls_bar.visible = is_graph
            if is_graph:
                refresh_graph_view()
            else:
                reload_memories()

        view_mode_toggle.on_value_change(on_view_toggle_change)

        # Database Switcher
        def on_db_switch(e):
            db_name = e.value
            active_db_key["key"] = db_name
            target_path = available_dbs.get(db_name)
            if target_path:
                active_manager_holder["manager"] = LollmsMemoryManager(
                    db_path=f"sqlite:///{target_path}",
                    config=MemoryConfig(working_token_budget=2000)
                )
                refresh_owners()
                filter_state["page"] = 1
                clear_node_focus()
                if view_mode_toggle.value == "graph":
                    refresh_graph_view()
                else:
                    reload_memories()

        db_select.on_value_change(on_db_switch)

        def refresh_owners():
            mm_inst = active_manager_holder["manager"]
            if not mm_inst:
                return
            try:
                owners = mm_inst.get_all_owner_ids()
                owner_dict = {"__all__": "All Owners / Projects"}
                for o in owners:
                    owner_dict[o] = o
                owner_select.options = owner_dict
                if filter_state["owner"] not in owner_dict:
                    filter_state["owner"] = "__all__"
                    owner_select.value = "__all__"
            except Exception:
                pass

        def update_stats():
            mm_inst = active_manager_holder["manager"]
            if not mm_inst:
                stats_label.set_text("Total: —")
                return

            db_path_label.set_text(getattr(mm_inst, "resolved_disk_path", ""))
            try:
                all_res = mm_inst.list_all(level=None, page=1, page_size=0, ignore_owner=True)
                mems = all_res.get("memories", [])
                counts = {1: 0, 2: 0, 3: 0, 4: 0}
                zero_imp_count = 0
                for m in mems:
                    lvl = m.get("level", 1)
                    counts[lvl] = counts.get(lvl, 0) + 1
                    if float(m.get("importance", 0.0) or 0.0) <= 0.001:
                        zero_imp_count += 1

                faded_str = f" | Faded (0%): {zero_imp_count}" if zero_imp_count > 0 else ""
                stats_label.set_text(
                    f"Total: {len(mems)} | Working (L1): {counts.get(1, 0)} | "
                    f"Deep (L2): {counts.get(2, 0)} | Archived (L3): {counts.get(3, 0)} | "
                    f"Episodic (L4): {counts.get(4, 0)}{faded_str}"
                )
            except Exception as ex:
                stats_label.set_text(f"Total: Error ({ex})")

        def reload_memories():
            memories_list.clear()
            mm_inst = active_manager_holder["manager"]
            if not mm_inst:
                with memories_list:
                    ui.label("Memory manager not available.").classes("text-sm text-red-500")
                return

            update_stats()
            lvl = filter_state["level"]
            q = filter_state["search"]
            owner_choice = filter_state["owner"]
            ignore_owner = (owner_choice == "__all__")
            owner_arg = None if ignore_owner else owner_choice

            try:
                data = mm_inst.list_all(
                    level=lvl,
                    search_query=q or None,
                    page=filter_state["page"],
                    page_size=filter_state["page_size"],
                    owner_id=owner_arg,
                    ignore_owner=ignore_owner,
                    order_by=filter_state["sort"],
                )
                memories = data.get("memories", [])
                total_records = data.get("total", 0)
                filter_state["total_pages"] = data.get("pages", 1)
            except Exception as ex:
                with memories_list:
                    ui.label(f"Failed to fetch memories: {ex}").classes("text-sm text-red-500 italic")
                return

            current_page = filter_state["page"]
            total_p = filter_state["total_pages"]
            page_info_label.set_text(f"Page {current_page} of {total_p} ({total_records} items)")
            prev_btn.set_visibility(current_page > 1)
            next_btn.set_visibility(current_page < total_p)

            if not memories:
                with memories_list:
                    with ui.column().classes("w-full items-center justify-center p-8 gap-1"):
                        ui.icon("sentiment_dissatisfied", size="36px").classes("text-slate-400")
                        ui.label("No memories found matching the current filters.").classes("text-sm text-slate-600 dark:text-slate-400 italic")
                return

            with memories_list:
                for memory in memories:
                    render_memory_card(memory, mm_inst, reload_memories)

        # Graph Renderer & Interactive Node Inspector
        active_chart_holder: Dict[str, Any] = {"chart": None}
        cached_graph_memories: List[Dict[str, Any]] = []

        def clear_node_focus():
            filter_state["focused_node_id"] = None
            filter_state["focused_hops"] = 0
            filter_state["focused_hop_ids"] = None
            focus_label.set_text("Showing All Graph")
            clear_focus_btn.set_visibility(False)
            refresh_graph_view()

        def apply_node_focus(node_id: str, hops: int):
            filter_state["focused_node_id"] = node_id
            filter_state["focused_hops"] = hops
            hop_set = get_k_hop_subgraph(node_id, cached_graph_memories, max_hops=hops)
            filter_state["focused_hop_ids"] = hop_set
            focus_label.set_text(f"🎯 Focused: [{node_id[:8]}] ({hops}-Hop: {len(hop_set)} nodes)")
            clear_focus_btn.set_visibility(True)
            refresh_graph_view()

        def refresh_graph_view():
            graph_chart_slot.clear()
            mm_inst = active_manager_holder["manager"]
            if not mm_inst:
                with graph_chart_slot:
                    ui.label("Memory manager not available.").classes("text-sm text-red-500 p-4")
                return

            update_stats()
            lvl = filter_state["level"]
            q = filter_state["search"]
            owner_choice = filter_state["owner"]
            ignore_owner = (owner_choice == "__all__")
            owner_arg = None if ignore_owner else owner_choice

            try:
                all_data = mm_inst.list_all(
                    level=lvl,
                    search_query=q or None,
                    page=1,
                    page_size=0,
                    owner_id=owner_arg,
                    ignore_owner=ignore_owner,
                    order_by=filter_state["sort"],
                )
                nonlocal cached_graph_memories
                cached_graph_memories = all_data.get("memories", [])
            except Exception as ex:
                with graph_chart_slot:
                    ui.label(f"Failed to load memories for graph: {ex}").classes("text-sm text-red-500 p-4")
                return

            if not cached_graph_memories:
                with graph_chart_slot:
                    with ui.column().classes("w-full h-full items-center justify-center p-8 gap-1"):
                        ui.icon("hub", size="42px").classes("text-slate-400")
                        ui.label("No memories to visualize with current filters.").classes("text-sm text-slate-500 italic")
                return

            options = build_memory_graph_options(
                memories=cached_graph_memories,
                mode=filter_state["graph_mode"],
                repulsion=filter_state["graph_repulsion"],
                search_query=filter_state["search"],
                focused_id=filter_state["focused_node_id"],
                focused_hop_ids=filter_state["focused_hop_ids"],
            )

            with graph_chart_slot:
                try:
                    chart = ui.echart(options).classes("w-full h-full min-h-[480px]")
                    active_chart_holder["chart"] = chart

                    def on_chart_click(e):
                        args = getattr(e, "args", {}) or {}
                        item_data = args.get("data") or {}
                        data_type = args.get("dataType")
                        name_str = args.get("name") or ""

                        # Robust ID extraction from node click
                        target_id = item_data.get("memory_id") or item_data.get("id")
                        if not target_id and name_str:
                            id_match = re.search(r'\[([a-f0-9]{8})\]', name_str)
                            if id_match:
                                pref = id_match.group(1)
                                match_mem = next((m for m in cached_graph_memories if m.get("id", "").startswith(pref)), None)
                                if match_mem:
                                    target_id = match_mem.get("id")

                        if target_id and (item_data.get("is_memory") or not str(target_id).startswith("tag::")):
                            target_memory = next((m for m in cached_graph_memories if m.get("id") == target_id), None)
                            if target_memory:
                                show_graph_node_inspector(target_memory, mm_inst)
                        elif str(target_id).startswith("tag::") or item_data.get("tag_name"):
                            tag_clicked = item_data.get("tag_name") or str(target_id)[5:]
                            if tag_clicked:
                                search_input.value = tag_clicked
                                ui.notify(f"Filtered to #{tag_clicked}", type="info")

                    chart.on("click", on_chart_click)
                except Exception as chart_err:
                    ui.label(f"Could not initialize ECharts graph: {chart_err}").classes("text-sm text-red-500 p-4")

        def show_graph_node_inspector(memory: Dict[str, Any], mm_inst: Any):
            graph_inspector.clear()
            graph_inspector.visible = True

            mid = memory.get("id", "")
            short_id = mid[:8]
            level = memory.get("level", 1)
            importance = float(memory.get("importance", 0.0) or 0.0)
            centrality = float(memory.get("centrality", 0.0) or 0.0)
            content = memory.get("content", "")
            created_at = memory.get("created_at", "")[:19].replace("T", " ")
            tags = memory.get("tags", "")
            subject = memory.get("subject")
            predicate = memory.get("predicate")
            obj = memory.get("object")
            owner_id = memory.get("owner_id")

            with graph_inspector:
                with ui.row().classes("w-full items-center justify-between pb-1 border-b border-slate-200 dark:border-slate-800"):
                    with ui.row().classes("items-center gap-1.5"):
                        ui.icon("memory", size="18px").classes("text-primary")
                        ui.label(f"[{short_id}]").classes("text-xs font-mono font-bold")
                    ui.button(icon="close", on_click=lambda: hide_inspector()).props("flat dense round size=xs")

                # Metrics badges
                with ui.row().classes("items-center gap-1.5 flex-wrap"):
                    ui.badge(LEVEL_NAMES.get(level, f"L{level}"), color=LEVEL_COLORS.get(level, "slate")).props("rounded dense")
                    ui.label(f"Imp: {importance:.0%}").classes("text-xs font-bold text-primary")
                    if centrality > 0.0:
                        ui.label(f"Cent: {centrality:.0%}").classes("text-xs font-mono text-blue-500")

                if subject and predicate and obj:
                    ui.label(f"({subject} --[{predicate}]--> {obj})").classes("text-xs text-purple-600 dark:text-purple-400 font-mono italic")

                if owner_id:
                    ui.label(f"Owner: {owner_id}").classes("text-[10px] text-slate-500 font-mono")
                ui.label(f"Created: {created_at}").classes("text-[10px] text-slate-400 font-mono")

                # Subgraph / Hop Filtering Controls
                with ui.card().classes("w-full p-2 bg-slate-100 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700 gap-1.5"):
                    ui.label("🎯 Hop Neighborhood Isolation:").classes("text-[11px] font-bold text-slate-700 dark:text-slate-300")
                    with ui.row().classes("items-center gap-1 w-full justify-between"):
                        ui.button("1 Hop (Direct)", on_click=lambda: apply_node_focus(mid, 1)).props("dense unelevated size=xs color=primary no-caps")
                        ui.button("2 Hops", on_click=lambda: apply_node_focus(mid, 2)).props("dense outline size=xs color=primary no-caps")
                        ui.button("All", on_click=lambda: clear_node_focus()).props("dense flat size=xs color=grey no-caps")

                # Content area
                ui.separator().classes("my-0.5")
                ui.label("Content:").classes("text-xs font-bold text-slate-700 dark:text-slate-300")
                ui.markdown(content).classes("text-xs text-slate-900 dark:text-slate-100 break-words max-h-52 overflow-y-auto p-2 bg-white dark:bg-slate-950 rounded border border-slate-200 dark:border-slate-800 leading-relaxed")

                # Tags
                if tags:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    with ui.row().classes("items-center gap-1 pt-1 flex-wrap"):
                        for t in tag_list:
                            ui.label(f"#{t}").classes("text-[10px] bg-slate-200 dark:bg-slate-800 px-1.5 py-0.5 rounded font-mono cursor-pointer").on("click", lambda tg=t: filter_by_tag(tg))

                # Check Redundancy for this single memory
                def check_single_memory_redundancies():
                    dups = find_redundant_memories(cached_graph_memories)
                    matched_dups = [d for d in dups if d["memory_a"]["id"] == mid or d["memory_b"]["id"] == mid]
                    if matched_dups:
                        ui.notify(f"Found {len(matched_dups)} suspected redundant link(s) for [{short_id}]. Opening auditor.", type="warning")
                        open_redundancy_auditor_dialog(mm_inst, reload_memories, prefiltered_duplicates=matched_dups)
                    else:
                        ui.notify(f"No redundant memories detected for [{short_id}]. Memory is unique.", type="positive")

                ui.button("Scan Redundancies for this Node", icon="find_replace", on_click=check_single_memory_redundancies).props("dense outline size=xs color=amber no-caps").classes("w-full")

                ui.separator().classes("my-1")

                # Action buttons
                with ui.row().classes("w-full justify-between items-center gap-1 mt-auto pt-1"):
                    if level != 1:
                        ui.button("Promote L1", icon="arrow_upward", on_click=lambda: (promote_to_working(mid, mm_inst, refresh_graph_view))).props("dense outline size=xs color=emerald no-caps")
                    ui.button("Edit", icon="edit", on_click=lambda: open_edit_memory_dialog(memory, mm_inst, refresh_graph_view)).props("dense outline size=xs color=primary no-caps")
                    ui.button("Delete", icon="delete", on_click=lambda: confirm_delete_memory(memory, mm_inst, refresh_graph_view)).props("dense flat size=xs color=red no-caps")

        def filter_by_tag(tag_str: str):
            search_input.value = tag_str
            ui.notify(f"Search filtered to #{tag_str}", type="info")

        def hide_inspector():
            graph_inspector.clear()
            graph_inspector.visible = False

        refresh_owners()
        reload_memories()

    dialog.open()


def render_memory_card(memory: Dict[str, Any], mm: Any, on_change: Any) -> None:
    """Renders a single memory record card with interactive actions and clean contrast."""
    mid = memory.get("id", "")
    short_id = mid[:8] if mid else "unknown"
    level = memory.get("level", 1)
    importance = float(memory.get("importance", 0.0) or 0.0)
    centrality = float(memory.get("centrality", 0.0) or 0.0)
    content = memory.get("content", "")
    created_at = memory.get("created_at", "")[:19].replace("T", " ")
    tags = memory.get("tags", "")
    subject = memory.get("subject")
    predicate = memory.get("predicate")
    obj = memory.get("object")
    owner_id = memory.get("owner_id")

    lvl_name = LEVEL_NAMES.get(level, f"L{level}")
    lvl_color = LEVEL_COLORS.get(level, "slate")

    imp_color = (
        "text-emerald-600 dark:text-emerald-400 font-bold" if importance >= 0.7
        else "text-amber-600 dark:text-amber-400 font-bold" if importance >= 0.3
        else "text-slate-500 font-bold"
    )

    with ui.card().classes("w-full p-3 no-shadow border border-slate-300 dark:border-slate-800 bg-white dark:bg-slate-950 rounded-lg gap-2 shadow-sm"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.badge(lvl_name, color=lvl_color).props("rounded dense")
                ui.label(f"[{short_id}]").classes("text-xs font-mono text-slate-600 dark:text-slate-400 select-all").tooltip(f"Full ID: {mid}")
                ui.label(f"Importance: {importance:.0%}").classes(f"text-xs {imp_color}")
                if centrality > 0.0:
                    ui.label(f"Centrality: {centrality:.0%}").classes("text-xs text-blue-600 dark:text-blue-400 font-mono")
                if subject and predicate and obj:
                    ui.label(f"({subject} --[{predicate}]--> {obj})").classes("text-xs text-purple-700 dark:text-purple-300 font-mono italic")
                if owner_id:
                    ui.label(f"owner: {owner_id}").classes("text-[10px] text-slate-500 font-mono")
                ui.label(created_at).classes("text-[11px] text-slate-500 font-mono")

            with ui.row().classes("items-center gap-1"):
                if level != 1:
                    ui.button(icon="arrow_upward", on_click=lambda m_id=mid: promote_to_working(m_id, mm, on_change)).props(
                        "flat round dense size=xs color=emerald"
                    ).tooltip("Promote to Working Memory (Level 1)")
                ui.button(icon="edit", on_click=lambda m=memory: open_edit_memory_dialog(m, mm, on_change)).props(
                    "flat round dense size=xs color=primary"
                ).tooltip("Edit memory")
                ui.button(icon="delete", on_click=lambda m=memory: confirm_delete_memory(m, mm, on_change)).props(
                    "flat round dense size=xs color=red"
                ).tooltip("Delete memory (Hard Purge or Soft Archive)")

        ui.markdown(content).classes("text-sm text-slate-900 dark:text-slate-100 break-words leading-relaxed")

        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            with ui.row().classes("items-center gap-1.5 pt-1"):
                for t in tag_list:
                    ui.label(f"#{t}").classes("text-[11px] text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded font-mono border border-slate-200 dark:border-slate-700")


def promote_to_working(memory_id: str, mm: Any, on_change: Any):
    try:
        res = mm.load_to_working(memory_id)
        if res:
            ui.notify(f"Memory [{memory_id[:8]}] promoted to Working Memory.", type="positive")
            on_change()
        else:
            ui.notify("Failed to promote memory.", type="warning")
    except Exception as e:
        ui.notify(f"Error promoting memory: {e}", type="negative")


def confirm_delete_memory(memory: Dict[str, Any], mm: Any, on_change: Any):
    memory_id = memory.get("id", "")
    content_preview = memory.get("content", "")[:60]
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[420px] bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 p-4 gap-3"):
        ui.label("Delete Memory Option").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label(f"Target: [{memory_id[:8]}] \"{content_preview}...\"").classes("text-xs text-slate-500 font-mono")
        ui.label("Choose how you want to remove this memory:").classes("text-xs text-slate-600 dark:text-slate-400")

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def do_soft():
                dialog.close()
                try:
                    success = mm.delete(memory_id)
                    if success:
                        ui.notify(f"Memory [{memory_id[:8]}] archived (importance set to 0%).", type="info")
                        on_change()
                except Exception as ex:
                    ui.notify(f"Error archiving memory: {ex}", type="negative")

            def do_hard():
                dialog.close()
                try:
                    success = mm.hard_delete(memory_id)
                    if success:
                        ui.notify(f"Memory [{memory_id[:8]}] permanently purged.", type="positive")
                        on_change()
                    else:
                        ui.notify("Memory not found.", type="warning")
                except Exception as ex:
                    ui.notify(f"Error purging memory: {ex}", type="negative")

            ui.button("Archive (0%)", on_click=do_soft).props("outline dense color=amber").tooltip("Set importance to 0% and move to Level 3")
            ui.button("Permanently Delete", on_click=do_hard).props("unelevated dense color=red").tooltip("Permanently purge from the SQLite table")

    dialog.open()


def confirm_purge_faded(mm: Any, on_change: Any):
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[400px] bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 p-4 gap-3"):
        ui.label("Purge All Faded Memories?").classes("text-base font-bold text-red-500")
        ui.label("Permanently delete all memories with importance <= 0% from this database? This cannot be undone.").classes(
            "text-xs text-slate-600 dark:text-slate-400"
        )
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def do_purge():
                dialog.close()
                try:
                    count = mm.purge_zero_importance()
                    ui.notify(f"Purged {count} faded memory record(s).", type="positive")
                    on_change()
                except Exception as ex:
                    ui.notify(f"Purge failed: {ex}", type="negative")

            ui.button("Purge Now", on_click=do_purge).props("unelevated dense color=red")
    dialog.open()


def open_edit_memory_dialog(memory: Dict[str, Any], mm: Any, on_change: Any):
    mid = memory.get("id", "")
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[560px] p-4 flex flex-col gap-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"):
        ui.label(f"Edit Memory [{mid[:8]}]").classes("text-base font-bold")

        content_input = ui.textarea("Content", value=memory.get("content", "")).classes("w-full bg-white dark:bg-slate-800").props("outlined autogrow dense")

        with ui.row().classes("w-full gap-3 items-center"):
            level_select = ui.select(
                {1: "Working (L1)", 2: "Deep (L2)", 3: "Archived (L3)", 4: "Episodic (L4)"},
                value=memory.get("level", 1),
                label="Level"
            ).classes("flex-1").props("outlined dense")

            importance_input = ui.number(
                "Importance (0.0 - 1.0)",
                value=float(memory.get("importance", 0.75)),
                min=0.0, max=1.0, step=0.05
            ).classes("flex-1").props("outlined dense")

        tags_input = ui.input("Tags (comma-separated)", value=memory.get("tags", "")).classes("w-full").props("outlined dense")

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def save_edit():
                new_text = (content_input.value or "").strip()
                if not new_text:
                    ui.notify("Content cannot be empty.", type="warning")
                    return
                dialog.close()
                try:
                    tag_list = [t.strip() for t in (tags_input.value or "").split(",") if t.strip()]
                    mm.edit_memory(
                        memory_id=mid,
                        content=new_text,
                        importance=float(importance_input.value or 0.75),
                        level=int(level_select.value),
                        tags=tag_list
                    )
                    ui.notify("Memory updated successfully.", type="positive")
                    on_change()
                except Exception as ex:
                    ui.notify(f"Error updating memory: {ex}", type="negative")

            ui.button("Save", on_click=save_edit).props("color=primary dense")
    dialog.open()


def open_add_memory_dialog(mm: Any, on_change: Any):
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[560px] p-4 flex flex-col gap-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"):
        ui.label("Add New Memory").classes("text-base font-bold")

        content_input = ui.textarea("Content", placeholder="Enter the knowledge, rule, or preference to persist…").classes("w-full bg-white dark:bg-slate-800").props("outlined autogrow dense")

        with ui.row().classes("w-full gap-3 items-center"):
            level_select = ui.select(
                {1: "Working (L1)", 2: "Deep (L2)", 3: "Archived (L3)", 4: "Episodic (L4)"},
                value=1,
                label="Tier Level"
            ).classes("flex-1").props("outlined dense")

            importance_input = ui.number(
                "Importance (0.0 - 1.0)",
                value=0.8,
                min=0.0, max=1.0, step=0.05
            ).classes("flex-1").props("outlined dense")

        tags_input = ui.input("Tags", placeholder="e.g. coding_style, architecture, database").classes("w-full").props("outlined dense")

        with ui.expansion("Ontology Triples (Optional)", icon="hub").classes("w-full text-slate-900 dark:text-slate-100").props('header-class="text-slate-900 dark:text-slate-100 text-xs"'):
            with ui.row().classes("w-full gap-2"):
                sub_input = ui.input("Subject", placeholder="e.g. user").classes("flex-1").props("outlined dense")
                pred_input = ui.input("Predicate", placeholder="e.g. PREFERS").classes("flex-1").props("outlined dense")
                obj_input = ui.input("Object", placeholder="e.g. python").classes("flex-1").props("outlined dense")

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def do_create():
                text_val = (content_input.value or "").strip()
                if not text_val:
                    ui.notify("Content is required.", type="warning")
                    return
                dialog.close()
                try:
                    tag_list = [t.strip() for t in (tags_input.value or "").split(",") if t.strip()]
                    mm.add(
                        content=text_val,
                        importance=float(importance_input.value or 0.75),
                        tags=tag_list,
                        level=int(level_select.value),
                        subject=sub_input.value or None,
                        predicate=pred_input.value or None,
                        obj=obj_input.value or None,
                    )
                    ui.notify("Memory added to database.", type="positive")
                    on_change()
                except Exception as ex:
                    ui.notify(f"Failed to add memory: {ex}", type="negative")

            ui.button("Create Memory", on_click=do_create).props("color=primary dense")
    dialog.open()


def open_redundancy_auditor_dialog(
    mm: Any,
    on_change: Any,
    prefiltered_duplicates: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Opens the Redundancy Auditor modal to compare and merge duplicate memory records.
    """
    dialog = ui.dialog().props("maximized")

    with dialog, ui.card().classes("w-full h-full flex flex-col p-4 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100"):
        with ui.row().classes("w-full items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-800"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("auto_fix_high", size="28px").classes("text-amber-500")
                with ui.column().classes("gap-0"):
                    ui.label("Memory Redundancy Auditor").classes("text-lg font-bold")
                    ui.label("Identify and resolve overlapping memories using shared tags and text sequence similarity.").classes("text-xs text-slate-500")
            ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

        scroll_area = ui.scroll_area().classes("w-full flex-1 min-h-0 pr-2")
        with scroll_area:
            clusters_container = ui.column().classes("w-full gap-4 p-2")

        def reload_auditor():
            clusters_container.clear()
            if prefiltered_duplicates is not None:
                duplicates = prefiltered_duplicates
            else:
                try:
                    all_data = mm.list_all(level=None, page=1, page_size=0, ignore_owner=True)
                    mems = all_data.get("memories", [])
                    duplicates = find_redundant_memories(mems)
                except Exception as ex:
                    with clusters_container:
                        ui.label(f"Failed to analyze memories: {ex}").classes("text-sm text-red-500")
                    return

            if not duplicates:
                with clusters_container:
                    with ui.column().classes("w-full items-center justify-center p-12 gap-2 text-center"):
                        ui.icon("verified", size="48px").classes("text-emerald-500")
                        ui.label("No Redundant Memories Detected!").classes("text-base font-bold text-slate-800 dark:text-slate-200")
                        ui.label("All memories in this database have distinct semantic tags and unique contents.").classes("text-xs text-slate-500 max-w-sm")
                return

            with clusters_container:
                ui.label(f"Found {len(duplicates)} suspected redundant memory pair(s):").classes("text-xs font-bold text-slate-600 dark:text-slate-400")

                for pair in duplicates:
                    m1 = pair["memory_a"]
                    m2 = pair["memory_b"]
                    conf = pair["confidence"]
                    shared = pair["shared_tags"]
                    sim = pair["text_similarity"]

                    with ui.card().classes("w-full p-4 border border-amber-300 dark:border-amber-900/60 bg-white dark:bg-slate-950 rounded-xl shadow-sm gap-3"):
                        # Header with confidence
                        with ui.row().classes("w-full items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-2"):
                            with ui.row().classes("items-center gap-2"):
                                ui.badge(f"{conf:.0%} Match", color="amber").props("rounded")
                                ui.label(f"Text similarity: {sim:.0%}").classes("text-xs font-mono text-slate-500")
                                if shared:
                                    ui.label(f"Shared tags: {', '.join(f'#{t}' for t in shared)}").classes("text-xs font-mono text-indigo-500")

                            with ui.row().classes("items-center gap-2"):
                                def merge_pair(mem_a=m1, mem_b=m2):
                                    try:
                                        # Combine tags
                                        tags_a = _get_tags_set(mem_a)
                                        tags_b = _get_tags_set(mem_b)
                                        fused_tags = list(tags_a | tags_b)

                                        # Keep highest importance and lower level (more accessible)
                                        fused_importance = max(mem_a.get("importance", 0.5), mem_b.get("importance", 0.5))
                                        fused_level = min(mem_a.get("level", 1), mem_b.get("level", 1))

                                        # Prefer longer, more descriptive content
                                        primary_mem = mem_a if len(mem_a.get("content", "")) >= len(mem_b.get("content", "")) else mem_b
                                        redundant_mem = mem_b if primary_mem == mem_a else mem_a

                                        mm.edit_memory(
                                            memory_id=primary_mem["id"],
                                            content=primary_mem["content"],
                                            importance=fused_importance,
                                            level=fused_level,
                                            tags=fused_tags,
                                        )
                                        # Hard delete redundant
                                        mm.hard_delete(redundant_mem["id"])
                                        ui.notify(f"Merged memory [{redundant_mem['id'][:8]}] into [{primary_mem['id'][:8]}].", type="positive")
                                        on_change()
                                        reload_auditor()
                                    except Exception as merge_err:
                                        ui.notify(f"Merge failed: {merge_err}", type="negative")

                                ui.button("1-Click Merge", icon="call_merge", on_click=merge_pair).props("unelevated dense size=xs color=primary no-caps").tooltip("Combines tags, preserves higher importance, and purges redundant record")

                        # Side-by-side memory comparisons
                        with ui.row().classes("w-full gap-3 items-stretch"):
                            # Memory A
                            with ui.column().classes("flex-1 p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 gap-1"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.label(f"Memory A [{m1['id'][:8]}]").classes("text-xs font-mono font-bold")
                                    ui.badge(LEVEL_NAMES.get(m1.get("level", 1), ""), color=LEVEL_COLORS.get(m1.get("level", 1), "slate")).props("dense")
                                ui.markdown(m1.get("content", "")).classes("text-xs text-slate-800 dark:text-slate-200")
                                if m1.get("tags"):
                                    ui.label(f"Tags: {m1['tags']}").classes("text-[10px] text-slate-500 font-mono mt-auto pt-1")
                                with ui.row().classes("w-full justify-end pt-1"):
                                    def del_a(mem=m1):
                                        mm.hard_delete(mem["id"])
                                        ui.notify(f"Purged Memory A [{mem['id'][:8]}].", type="info")
                                        on_change()
                                        reload_auditor()
                                    ui.button("Purge A", icon="delete", on_click=del_a).props("flat dense size=xs color=red no-caps")

                            # Memory B
                            with ui.column().classes("flex-1 p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 gap-1"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.label(f"Memory B [{m2['id'][:8]}]").classes("text-xs font-mono font-bold")
                                    ui.badge(LEVEL_NAMES.get(m2.get("level", 1), ""), color=LEVEL_COLORS.get(m2.get("level", 1), "slate")).props("dense")
                                ui.markdown(m2.get("content", "")).classes("text-xs text-slate-800 dark:text-slate-200")
                                if m2.get("tags"):
                                    ui.label(f"Tags: {m2['tags']}").classes("text-[10px] text-slate-500 font-mono mt-auto pt-1")
                                with ui.row().classes("w-full justify-end pt-1"):
                                    def del_b(mem=m2):
                                        mm.hard_delete(mem["id"])
                                        ui.notify(f"Purged Memory B [{mem['id'][:8]}].", type="info")
                                        on_change()
                                        reload_auditor()
                                    ui.button("Purge B", icon="delete", on_click=del_b).props("flat dense size=xs color=red no-caps")

        reload_auditor()

    dialog.open()


def trigger_dream_cycle(mm: Any, client: Any, on_change: Any):
    """Runs a dream consolidation cycle and presents the report."""
    ui.notify("Running Dream consolidation cycle…", type="info")
    try:
        report = mm.dream(client)
        if report.get("skipped"):
            ui.notify(f"Dream cycle skipped: {report.get('reason')}", type="warning")
            return

        decayed = report.get("decayed", 0)
        retained = report.get("retained_by_dreamer", 0)
        fused = report.get("fused_nodes", 0)
        forgotten = report.get("forgotten", 0)
        sec = report.get("duration_seconds", 0.0)

        summary = f"Dream complete in {sec:.2f}s: Decayed {decayed}, Retained {retained}, Fused {fused}, Pruned {forgotten}."
        ui.notify(summary, type="positive", timeout=5000)
        on_change()
    except Exception as e:
        ui.notify(f"Dream cycle failed: {e}", type="negative")