# lollms_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# LollmsMemoryManager — Human-brain-inspired multi-level memory system
#
# ARCHITECTURE OVERVIEW
# ─────────────────────
# Level 0 — Volatile Scratchpad  (in-process only, cleared each turn)
#   Agentic iteration notes, tool output summaries, reasoning traces.
#   Never persisted. The LLM sees it injected right before the last user turn.
#
# Level 1 — Working Memory  (SQLite, always in context)
#   Recent, high-importance facts the LLM can directly read, tag, update,
#   or create. Injected as a compact zone before the last user message.
#   Capped by token budget; overflow → Level 2.
#
# Level 2 — Deep Memory  (SQLite, accessed via handles)
#   Long-term storage for lower-importance or less-recently-used memories.
#   Not injected verbatim — only compact *handles* appear in context so
#   the LLM knows something exists and can request a load.
#
# Level 3 — Archived Memory  (SQLite, never auto-loaded)
#   Very old, near-zero importance memories. The `dream()` pass can either
#   promote them back or mark them for permanent deletion.
#
# IMPORTANCE SCORING
# ──────────────────
# Each memory carries a float [0.0, 1.0] importance score.
# Events that change importance:
#   • LLM tags  a memory    → +10 %  (recognition / retrieval)
#   • LLM updates content   → +25 %  (active refinement)
#   • New memory created    → 75 %   (configurable default)
#   • dream() consolidation → recomputed from recency + use frequency
#   • Aging decay           → −(decay_rate) per day since last_used_at
#   • Score < 25%           → demoted to deep memory
#   • Score < 5%            → archived
#   • Explicit deletion     → hard delete
#
# XML TAGS (emitted by LLM)
# ─────────────────────────
# <mem_tag id="UUID" />                          — recognise / use memory
# <mem_update id="UUID">new content</mem_update> — update content
# <mem_new>content</mem_new>                     — create memory
# <mem_new importance="0.9">content</mem_new>    — create with explicit score
# <mem_delete id="UUID" />                       — hard delete
# <mem_load id="UUID" />                         — promote deep → working
#
# ─────────────────────────────────────────────────────────────────────────────


from __future__ import annotations

import math
import re
import uuid
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from sqlalchemy import (Column, DateTime, Float, Index, Integer, String,
                        Text, create_engine)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from ascii_colors import ASCIIColors

_Base = declarative_base()

class _MemoryRecord(_Base):
    __tablename__ = "memories"
    id              = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id        = Column(String,   nullable=True,  index=True)
    content         = Column(Text,     nullable=False)
    summary         = Column(Text,     nullable=True)
    level           = Column(Integer,  nullable=False, default=1)
    importance      = Column(Float,    nullable=False, default=0.75)
    use_count       = Column(Integer,  nullable=False, default=0)
    tags            = Column(Text,     nullable=True)
    subject_group   = Column(String,   nullable=True)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at    = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_mem_owner_level_importance", "owner_id", "level", "importance"),
    )
# ─────────────────────────────────────────────────────────────────────────────
# MemoryConfig
# ─────────────────────────────────────────────────────────────────────────────

class MemoryConfig:
    """
    Tuneable parameters for the memory system.

    Parameters
    ----------
    working_token_budget : int
        Maximum tokens the working-memory zone may consume in the context
        window. If active memories exceed this, lowest-importance ones are
        demoted to deep memory.
    handles_token_budget : int
        Maximum tokens the handles zone (deep-memory stubs) may consume.
        Groups are merged when the raw handle list would exceed this.
    default_importance : float
        Importance assigned to newly created memories.
    tag_boost : float
        Added to importance when the LLM tags a memory (+10 % default).
    update_boost : float
        Added to importance when the LLM updates a memory (+25 % default).
    decay_rate_per_day : float
        Fractional importance lost per day of non-use.
    demotion_threshold : float
        Memories below this score are moved to deep memory.
    archive_threshold : float
        Memories below this score are moved to the archive.
    dream_min_interval_hours : int
        Minimum wall-clock hours between automatic dream() passes.
        Set to 0 to allow dream() to run on every call.
    max_working_memories : int
        Hard cap on working-memory records (independent of token budget).
    max_handles : int
        Maximum number of individual handle lines before grouping kicks in.
    """

    def __init__(
        self,
        working_token_budget:   int   = 1024,
        handles_token_budget:   int   = 256,
        default_importance:     float = 0.75,
        tag_boost:              float = 0.10,
        update_boost:           float = 0.25,
        decay_rate_per_day:     float = 0.02,
        demotion_threshold:     float = 0.25,
        archive_threshold:      float = 0.05,
        dream_min_interval_hours: int = 1,
        max_working_memories:   int   = 40,
        max_handles:            int   = 30,
    ):
        self.working_token_budget      = working_token_budget
        self.handles_token_budget      = handles_token_budget
        self.default_importance        = default_importance
        self.tag_boost                 = tag_boost
        self.update_boost              = update_boost
        self.decay_rate_per_day        = decay_rate_per_day
        self.demotion_threshold        = demotion_threshold
        self.archive_threshold         = archive_threshold
        self.dream_min_interval_hours  = dream_min_interval_hours
        self.max_working_memories      = max_working_memories
        self.max_handles               = max_handles

# ─────────────────────────────────────────────────────────────────────────────
# LollmsMemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class LollmsMemoryManager:
    """
    Multi-level memory manager for LollmsDiscussion.
    """

    # ── XML tag patterns ─────────────────────────────────────────────────────
 
    _PAT_TAG      = re.compile(r'<mem_tag\s+id=["\']([^"\']+)["\'](?:\s*/)?>',      re.I)
    _PAT_DELETE   = re.compile(r'<mem_delete\s+id=["\']([^"\']+)["\'](?:\s*/)?>',   re.I)
    _PAT_LOAD     = re.compile(r'<mem_load\s+id=["\']([^"\']+)["\'](?:\s*/)?>',     re.I)
    _PAT_UPDATE   = re.compile(r'<mem_update\s+id=["\']([^"\']+)["\']>(.*?)</mem_update>', re.I | re.DOTALL)
    _PAT_NEW      = re.compile(r'<mem_new(?:\s+importance=["\']([^"\']*)["\'])?\s*>(.*?)</mem_new>', re.I | re.DOTALL)

    def __init__(
        self,
        db_path:       str,
        owner_id:      Optional[str]   = None,
        lollms_client: Optional[Any]   = None,
        config:        Optional[MemoryConfig] = None,
    ):
        self.db_path       = db_path
        self.owner_id      = owner_id
        self.lollms_client = lollms_client
        self.config        = config or MemoryConfig()
        self._engine       = create_engine(db_path)
        _Base.metadata.create_all(self._engine)
        self._Session      = sessionmaker(bind=self._engine, autoflush=False)
        self._last_dream   = datetime.utcnow() - timedelta(days=1)
        ASCIIColors.info(f"[MemoryManager] Initialised — db={db_path}, owner={owner_id}")

    # ──────────────────────────────────────────────── session helper

    @contextmanager
    def _session(self):
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def _q(self, session: Session):
        q = session.query(_MemoryRecord)
        if self.owner_id:
            q = q.filter(_MemoryRecord.owner_id == self.owner_id)
        return q

    # ──────────────────────────────────────────────── CRUD

    def add(self, content: str, importance: Optional[float] = None, tags: Optional[List[str]] = None, subject_group: Optional[str] = None, level: int = 1) -> Dict:
        now = datetime.utcnow()
        rec = _MemoryRecord(
            id=str(uuid.uuid4()), owner_id=self.owner_id, content=content.strip(),
            summary=self._auto_summary(content), level=level,
            importance=max(0.0, min(1.0, importance if importance is not None else self.config.default_importance)),
            use_count=0, tags=",".join(tags) if tags else None, subject_group=subject_group,
            created_at=now, updated_at=now, last_used_at=now,
        )
        with self._session() as s:
            s.add(rec)
            s.flush()
            return self._to_dict(rec)

    def get(self, memory_id: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            return self._to_dict(r) if r else None

    def update(self, memory_id: str, new_content: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return None
            r.content = new_content.strip()
            r.summary = self._auto_summary(new_content)
            r.importance = min(1.0, r.importance + self.config.update_boost)
            r.updated_at = datetime.utcnow()
            r.last_used_at = datetime.utcnow()
            r.level = 1
            s.flush()
            return self._to_dict(r)

    def tag(self, memory_id: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return None
            r.importance = min(1.0, r.importance + self.config.tag_boost)
            r.use_count += 1
            r.last_used_at = datetime.utcnow()
            if r.level > 1: r.level = 1
            s.flush()
            return self._to_dict(r)

    def delete(self, memory_id: str) -> bool:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return False
            s.delete(r)
            return True

    def load_to_working(self, memory_id: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return None
            r.level = 1
            r.last_used_at = datetime.utcnow()
            r.importance = max(r.importance, self.config.demotion_threshold + 0.1)
            s.flush()
            return self._to_dict(r)

    def list_working(self) -> List[Dict]:
        with self._session() as s:
            recs = (self._q(s).filter(_MemoryRecord.level == 1).order_by(_MemoryRecord.importance.desc()).all())
            return [self._to_dict(r) for r in recs]

    def query(self, text: str, top_k: int = 5, level: Optional[int] = None) -> List[Dict]:
        """Simple keyword search."""
        words = set(text.lower().split())
        with self._session() as s:
            q = self._q(s)
            if level is not None:
                q = q.filter(_MemoryRecord.level == level)
            records_raw = q.order_by(_MemoryRecord.importance.desc()).limit(200).all()
            dicts = [self._to_dict(r) for r in records_raw]
        scored = []
        for d in dicts:
            c = d['content'].lower()
            hits = sum(1 for w in words if w in c)
            if hits:
                scored.append((hits, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    # ──────────────────────────────────────────────── context assembly    def build_working_zone(self, token_counter=None) -> str:
        records = self.list_working()
        if not records: return ""
        budget, lines, used_tok = self.config.working_token_budget, [], 0
        for r in records:
            entry = f"[{r['id'][:8]}] ({r['importance']:.0%}) {r['content']}"
            if r.get('tags'): entry += f"  #{r['tags'].replace(',', ' #')}"
            tok = token_counter(entry) if token_counter else len(entry) // 4
            if used_tok + tok > budget: break
            lines.append(entry)
            used_tok += tok
        return "=== WORKING MEMORY ===\n" + "\n".join(lines) + "\n=== END WORKING MEMORY ===\n" if lines else ""

    def build_handles_zone(self, token_counter=None) -> str:
        with self._session() as s:
            recs = (self._q(s).filter(_MemoryRecord.level == 2).order_by(_MemoryRecord.importance.desc()).all())
            if not recs: return ""
            dicts = [self._to_dict(r) for r in recs]
        budget, max_h = self.config.handles_token_budget, self.config.max_handles
        if len(dicts) <= max_h:
            lines, used = [], 0
            for r in dicts:
                tag, line = r.get('subject_group') or 'general', f"  [{r['id'][:8]}] [{r.get('subject_group') or 'general'}] {r.get('summary') or r['content'][:60]}"
                tok = token_counter(line) if token_counter else len(line) // 4
                if used + tok > budget: lines.append(f"  … (+{len(dicts) - len(lines)} more)"); break
                lines.append(line); used += tok
            body = "\n".join(lines)
        else:
            groups, lines, used = {}, [], 0
            for r in dicts: groups.setdefault(r.get('subject_group') or 'general', []).append(r)
            for g, members in sorted(groups.items()):
                line = f"  [{g}] ({len(members)} memories) ids: {', '.join(m['id'][:8] for m in members[:5])}{' +'+str(len(members)-5) if len(members)>5 else ''}"
                tok = token_counter(line) if token_counter else len(line) // 4
                if used + tok > budget: break
                lines.append(line); used += tok
            body = "\n".join(lines)
        return "=== DEEP MEMORY HANDLES ===\n(Use <mem_load id=\"ID\"/> to bring into working memory)\n" + body + "\n=== END DEEP MEMORY HANDLES ===\n"

    def build_system_instructions(self) -> str:
        return (
            "\n=== MEMORY SYSTEM ===\n\nYou have multi-level persistent memory. Use it to recall facts.\n"
            "── Working Memory: Memories [ID] (importance%) in WORKING MEMORY zone.\n"
            "── Deep Memory: Stub handles. Load via <mem_load id=\"XXXXXXXX\"/>.\n"
            "── Tags: <mem_tag id=\"ID\"/> (recognition), <mem_new>Content</mem_new>, <mem_update id=\"ID\">New</mem_update>, <mem_delete id=\"ID\"/>.\n"
            "── Rules: IDs are 8-char prefixes. Tags are stripped from reply.\n=== END MEMORY SYSTEM ===\n"
        )

    def process_llm_output(self, text: str) -> Tuple[str, Dict[str, Any]]:
        cleaned, report = text, {"created": [], "updated": [], "tagged": [], "deleted": [], "loaded":  []}
        for m in self._PAT_UPDATE.finditer(text):
            full_id = self._resolve_id(m.group(1))
            if full_id:
                res = self.update(full_id, m.group(2).strip())
                if res: report["updated"].append(res)
            cleaned = cleaned.replace(m.group(0), "", 1)
        for m in self._PAT_NEW.finditer(text):
            imp = float(m.group(1)) if m.group(1) else self.config.default_importance
            res = self.add(m.group(2).strip(), importance=max(0.0, min(1.0, imp)))
            report["created"].append(res)
            cleaned = cleaned.replace(m.group(0), "", 1)
        for pat, key, fn in [(self._PAT_TAG, "tagged", self.tag), (self._PAT_LOAD, "loaded", self.load_to_working)]:
            for m in pat.finditer(text):
                full_id = self._resolve_id(m.group(1))
                if full_id:
                    res = fn(full_id)
                    if res: report[key].append(res)
                cleaned = cleaned.replace(m.group(0), "", 1)
        for m in self._PAT_DELETE.finditer(text):
            full_id = self._resolve_id(m.group(1))
            if full_id and self.delete(full_id): report["deleted"].append(m.group(1))
            cleaned = cleaned.replace(m.group(0), "", 1)
        return cleaned.strip(), report

    def _resolve_id(self, partial_id: str) -> Optional[str]:
        if len(partial_id) == 36: return partial_id
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id.like(f"{partial_id}%")).first()
            return r.id if r else None

    def apply_decay(self) -> int:
        now, count = datetime.utcnow(), 0
        with self._session() as s:
            for r in self._q(s).all():
                delta = (now - r.last_used_at).total_seconds() / 86400.0
                new_imp = max(0.0, r.importance - (self.config.decay_rate_per_day * delta))
                if abs(new_imp - r.importance) > 0.001: r.importance = new_imp; count += 1
                if r.level == 1 and r.importance < self.config.demotion_threshold: r.level = 2
                elif r.level <= 2 and r.importance < self.config.archive_threshold: r.level = 3
        return count

    def enforce_budget(self, token_counter=None) -> int:
        recs = self.list_working()
        if not recs: return 0
        budget, max_w, used, keep = self.config.working_token_budget, self.config.max_working_memories, 0, set()
        for r in recs:
            entry = f"[{r['id'][:8]}] ({r['importance']:.0%}) {r['content']}"
            tok = token_counter(entry) if token_counter else len(entry) // 4
            if used + tok > budget or len(keep) >= max_w: break
            keep.add(r['id']); used += tok
        demoted = 0
        with self._session() as s:
            for r in self._q(s).filter(_MemoryRecord.level == 1).all():
                if r.id not in keep: r.level = 2; demoted += 1
        return demoted


    def dream(self, lollms_client: Optional[Any] = None) -> Dict:
        """Consolidation pass — recompute importance, build clusters."""
        elapsed = (datetime.utcnow() - self._last_dream).total_seconds() / 3600
        if elapsed < self.config.dream_min_interval_hours:
            return {"skipped": True, "reason": "too_soon"}
        start = datetime.utcnow()
        report = {"decayed": self.apply_decay(), "promoted": 0, "duration_seconds": 0.0}
        # Rule-based promotion for recently used archived memories
        with self._session() as s:
            for r in self._q(s).filter(_MemoryRecord.level == 3).all():
                if (datetime.utcnow() - r.last_used_at).total_seconds() < 86400:
                    r.level = 2; report["promoted"] += 1
        self.enforce_budget()
        report["duration_seconds"] = (datetime.utcnow() - start).total_seconds()
        self._last_dream = datetime.utcnow()
        return report
    def _auto_summary(self, content: str) -> str:
        for line in content.splitlines():
            s = line.strip()
            if s: return s[:80] + ("…" if len(s) > 80 else "")
        return content[:80]

    def _to_dict(self, r: _MemoryRecord) -> Dict:
        return {
            "id": r.id, "owner_id": r.owner_id, "content": r.content, "summary": r.summary,
            "level": r.level, "importance": round(r.importance, 4), "use_count": r.use_count,
            "tags": r.tags, "subject_group": r.subject_group, "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(), "last_used_at": r.last_used_at.isoformat(),
        }