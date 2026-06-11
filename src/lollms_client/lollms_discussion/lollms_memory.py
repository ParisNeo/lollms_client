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
# Level 4 — Episodic Memory  (SQLite, persistent records of past events/conversations)
#   Historical context of past interactions and specific user-AI sessions.
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
                        Text, ForeignKey, create_engine)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from ascii_colors import ASCIIColors

_Base = declarative_base()

class MemoryOntology:
    """Lightweight Memory Ontology representing standard classes and predicates."""
    # Node Classes (Sovereign categories)
    class NodeClass:
        CONCEPT = "CONCEPT"          # Abstract ideas, subjects, tools
        PREFERENCE = "PREFERENCE"    # User settings, guidelines, rules
        EVENT = "EVENT"              # Episodes, historical milestones, tool executions
        DECISION = "DECISION"        # Architectural/code choices, plans

    # Semantic Predicates (Relationships)
    class Predicate:
        RELATED_TO = "RELATED_TO"          # Faded / default association
        PREFERS = "PREFERS"                # User/Subject preference association
        IMPLEMENTS = "IMPLEMENTS"          # Code implementation of a concept/goal
        CONTRADICTS = "CONTRADICTS"        # Logic conflicts
        SUPPORTS = "SUPPORTS"              # Logical validation
        TEMPORAL_AFTER = "TEMPORAL_AFTER"  # Chronological order of events
        PART_OF = "PART_OF"                # Parent-child hierarchy decomposition

    @classmethod
    def validate_predicate(cls, predicate: str) -> str:
        p = str(predicate).strip().upper()
        valid = {cls.Predicate.RELATED_TO, cls.Predicate.PREFERS, cls.Predicate.IMPLEMENTS, 
                 cls.Predicate.CONTRADICTS, cls.Predicate.SUPPORTS, cls.Predicate.TEMPORAL_AFTER, 
                 cls.Predicate.PART_OF}
        return p if p in valid else cls.Predicate.RELATED_TO

class _MemoryRecord(_Base):
    __tablename__ = "memories"
    id              = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id        = Column(String,   nullable=True,  index=True)
    content         = Column(Text,     nullable=False)
    summary         = Column(Text,     nullable=True)
    level           = Column(Integer,  nullable=False, default=1)
    importance      = Column(Float,    nullable=False, default=0.75)
    centrality      = Column(Float,    nullable=True, default=0.0)  # Graph centrality score
    use_count       = Column(Integer,  nullable=False, default=0)
    tags            = Column(Text,     nullable=True)
    subject_group   = Column(String,   nullable=True)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at    = Column(DateTime, nullable=False, default=datetime.utcnow)
    subject         = Column(Text,     nullable=True)
    predicate       = Column(Text,     nullable=True)
    object          = Column(Text,     nullable=True)
    activation      = Column(Float,    nullable=True, default=0.0)

    __table_args__ = (
        Index("ix_mem_owner_level_importance", "owner_id", "level", "importance"),
    )


class _MemoryRelationship(_Base):
    """Explicit graph edges between memory nodes."""
    __tablename__ = "memory_relationships"
    id              = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id       = Column(String,   ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True)
    target_id       = Column(String,   ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type = Column(String, nullable=False, default="RELATED_TO")  # RELATED_TO, DERIVED_FROM, CONTRADICTS, SUPPORTS, TEMPORAL_AFTER
    weight          = Column(Float,    nullable=False, default=1.0)  # Edge weight for traversal
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    relationship_metadata = Column(Text, nullable=True)  # JSON metadata

    __table_args__ = (
        Index("ix_rel_source_target", "source_id", "target_id", unique=True),
    )


class _RetrievalLog(_Base):
    __tablename__ = "retrieval_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True)
    retrieved_at = Column(DateTime, nullable=False, default=datetime.utcnow)


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
    forget_threshold : float
        Memories below this score are automatically evaluated for forgetting
        by the dreamer LLM.
    """

    def __init__(
        self,
        working_token_budget:   int   = 1024,
        handles_token_budget:   int   = 256,
        default_importance:     float = 0.75,
        tag_boost:              float = 0.10,
        update_boost:           float = 0.25,
        decay_rate_per_day:     float = 0.5,   # Petroff's power law decay rate parameter d
        spread_probability:     float = 0.9,   # Attenuation multiplier for Spreading Activation
        demotion_threshold:     float = 0.25,
        archive_threshold:      float = 0.05,
        dream_min_interval_hours: int = 1,
        max_working_memories:   int   = 40,
        max_handles:            int   = 30,
        forget_threshold:       float = 0.02,
    ):
        self.working_token_budget      = working_token_budget
        self.handles_token_budget      = handles_token_budget
        self.default_importance        = default_importance
        self.tag_boost                 = tag_boost
        self.update_boost              = update_boost
        self.decay_rate_per_day        = decay_rate_per_day
        self.spread_probability        = spread_probability
        self.demotion_threshold        = demotion_threshold
        self.archive_threshold         = archive_threshold
        self.dream_min_interval_hours  = dream_min_interval_hours
        self.max_working_memories      = max_working_memories
        self.max_handles               = max_handles
        self.forget_threshold          = forget_threshold

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
    _PAT_NEW      = re.compile(r'<mem_new\s+([^>]*?)(?:>(.*?)</mem_new>|/?>)', re.I | re.DOTALL)

    def __init__(
        self,
        db_path:       str,
        owner_id:      Optional[str]   = None,
        lollms_client: Optional[Any]   = None,
        config:        Optional[MemoryConfig] = None,
    ):
        # Normalize Windows backslashes to forward slashes for SQLAlchemy compatibility
        if db_path.startswith("sqlite:///"):
            path_part = db_path[10:]
            clean_path = path_part.replace("\\", "/")
            db_path = f"sqlite:///{clean_path}"
        elif db_path.startswith("sqlite://"):
            path_part = db_path[9:]
            clean_path = path_part.replace("\\", "/")
            db_path = f"sqlite://{clean_path}"

        self.db_path       = db_path
        self.owner_id      = owner_id
        self.lollms_client = lollms_client
        self.config        = config or MemoryConfig()
        # Use StaticPool to ensure all connections share the same in-memory SQLite database instance
        engine_kwargs = {}
        if db_path.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        if db_path == "sqlite:///:memory:":
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool

        self._engine = create_engine(db_path, **engine_kwargs)
        _Base.metadata.create_all(self._engine)

        # Safe in-place column migrations for existing SQLite databases
        try:
            with self._engine.connect() as connection:
                from sqlalchemy import text
                cursor = connection.execute(text("PRAGMA table_info(memories)"))
                columns = {row[1] for row in cursor.fetchall()}
                migrations = [
                    ('subject', "ALTER TABLE memories ADD COLUMN subject TEXT"),
                    ('predicate', "ALTER TABLE memories ADD COLUMN predicate TEXT"),
                    ('object', "ALTER TABLE memories ADD COLUMN object TEXT"),
                    ('activation', "ALTER TABLE memories ADD COLUMN activation REAL"),
                    ('centrality', "ALTER TABLE memories ADD COLUMN centrality REAL"),
                ]
                for col, sql in migrations:
                    if col not in columns:
                        ASCIIColors.info(f"  -> Upgrading 'memories' table: Adding '{col}' column.")
                        connection.execute(text(sql))
                connection.commit()
        except Exception as e:
            ASCIIColors.warning(f"Memory migration warning: {e}")

        self._Session      = sessionmaker(bind=self._engine, autoflush=False)
        self._last_dream   = datetime.utcnow() - timedelta(days=1)

        # In-Memory Cache representing active network state
        self._cache = {}
        self._working_zone_cache = None
        self._handles_zone_cache = None

        ASCIIColors.info(f"[MemoryManager] Initialised — db={db_path}, owner={owner_id}")

    # ──────────────────────────────────────────────── session helper

    def _clear_cache(self):
        self._cache.clear()
        self._working_zone_cache = None
        self._handles_zone_cache = None

    @contextmanager
    def _session(self):
        s = self._Session()
        try:
            yield s
            s.commit()
            # Clear caches on successful commit of any transaction (writes, tags, deletes)
            self._clear_cache()
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

    def add(
        self,
        content: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        subject_group: Optional[str] = None,
        level: int = 1,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> Dict:
        now = datetime.utcnow()
        with self._session() as s:
            # Query maximum created_at in database to enforce monotonicity
            max_created = s.query(_MemoryRecord.created_at).filter(_MemoryRecord.owner_id == self.owner_id).order_by(_MemoryRecord.created_at.desc()).first()
            if max_created and max_created[0]:
                if now <= max_created[0]:
                    now = max_created[0] + timedelta(microseconds=1)

            # Standardise Ontological Triples fallback if omitted by caller
            sub_val = subject or (tags[0] if tags and len(tags) > 0 else "unknown")
            pred_val = predicate or "RELATED_TO"
            obj_val = obj or (tags[1] if tags and len(tags) > 1 else "unknown")

            rec = _MemoryRecord(
                id=str(uuid.uuid4()), owner_id=self.owner_id, content=content.strip(),
                summary=self._auto_summary(content), level=level,
                importance=max(0.0, min(1.0, importance if importance is not None else self.config.default_importance)),
                use_count=0, tags=",".join(tags) if tags else None, subject_group=subject_group,
                created_at=now, updated_at=now, last_used_at=now,
                subject=sub_val.strip().lower(),
                predicate=MemoryOntology.validate_predicate(pred_val),
                object=obj_val.strip().lower(),
                activation=0.0
            )
            s.add(rec)
            s.flush()

            # Record initial retrieval log
            log = _RetrievalLog(node_id=rec.id, retrieved_at=now)
            s.add(log)
            s.flush()

            return self._to_dict(rec)

    def get(self, memory_id: str) -> Optional[Dict]:
        # Check in-memory cache first
        if memory_id in self._cache:
            return self._cache[memory_id]
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r:
                d = self._to_dict(r)
                self._cache[memory_id] = d
                return d
            return None

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

    def auto_pull_associative_memories(self, memory_id: str, top_k: int = 2) -> List[Dict]:
        """
        Locates other deep/archived memories associated with the given memory_id
        (via shared tags, matching subject_group, or keyword similarity)
        and promotes them to Level 2 (Deep Memory) to make their handles visible.
        """
        target = self.get(memory_id)
        if not target:
            return []

        associated_ids = set()
        pulled_memories = []

        target_tags = set(t.strip().lower() for t in (target.get("tags") or "").split(",") if t.strip())
        target_group = target.get("subject_group")

        with self._session() as s:
            candidates = self._q(s).filter(_MemoryRecord.id != memory_id)\
                                   .filter(_MemoryRecord.level.in_([2, 3]))\
                                   .all()

            for c in candidates:
                # Association A: Match by identical Subject Group
                group_match = target_group and c.subject_group == target_group

                # Association B: Match by shared tags
                c_tags = set(t.strip().lower() for t in (c.tags or "").split(",") if t.strip())
                tag_match = bool(target_tags & c_tags)

                if group_match or tag_match:
                    associated_ids.add(c.id)

        for cid in list(associated_ids)[:top_k]:
            with self._session() as s:
                r = s.get(_MemoryRecord, cid)
                if r:
                    if r.level == 3:
                        r.level = 2
                    r.importance = min(1.0, r.importance + 0.05)  # slight associative boost
                    s.flush()
                    pulled_memories.append(self._to_dict(r))

        if pulled_memories:
            ASCIIColors.cyan(
                f"[Memory] Associative Recall — Staged {len(pulled_memories)} related memory handle(s) "
                f"associated with '{target['id'][:8]}' ('{target['content'][:30]}...')."
            )
        return pulled_memories

    def compute_petroff_activation(self, session: Session, node_id: str, t: datetime) -> float:
        """
        Computes the base-level activation using Petroff's power-law decay approximation.
        Bi = ln( Sum( (t - t_j)^(-d) ) )
        """
        import math
        logs = session.query(_RetrievalLog).filter_by(node_id=node_id).all()
        if not logs:
            return -99.0
        
        total = 0.0
        d = self.config.decay_rate_per_day
        for log in logs:
            delta = (t - log.retrieved_at).total_seconds()
            if delta < 0.001:
                delta = 0.001
            total += math.pow(delta, -d)
            
        return math.log(total) if total > 0.0 else -99.0

    def spread_activation(self, session: Session, source_id: str, t: datetime):
        """
        Spreads energy multiplicatively from source_id to linked semantic neighbors.
        Pre-warms associated nodes in deep memory without loading full contents.
        """
        r = session.query(_MemoryRecord).filter_by(id=source_id).first()
        if not r or not r.subject or not r.object:
            return

        src_activation = r.activation if r.activation is not None else -99.0
        if src_activation == -99.0:
            src_activation = self.compute_petroff_activation(session, source_id, t)

        # Find linked nodes sharing Subject, Predicate, or Object
        neighbors = session.query(_MemoryRecord).filter(
            _MemoryRecord.id != source_id,
            (
                (_MemoryRecord.subject == r.subject) | 
                (_MemoryRecord.object == r.subject) | 
                (_MemoryRecord.subject == r.object) | 
                (_MemoryRecord.object == r.object)
            )
        ).all()

        for n in neighbors:
            boost = src_activation * self.config.spread_probability
            n.activation = (n.activation or 0.0) + boost
            if n.level == 3:
                n.level = 2
            n.importance = max(0.0, min(1.0, n.importance + 0.05))  # slight associative boost
            ASCIIColors.info(f"[Spreading Activation] Pre-warmed associated memory '{n.content[:40]}...' (New activation: {n.activation:.2f})")
        session.flush()

    def tag(self, memory_id: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return None

            now = datetime.utcnow()
            log = _RetrievalLog(node_id=memory_id, retrieved_at=now)
            s.add(log)
            s.flush()

            r.importance = min(1.0, r.importance + self.config.tag_boost)
            r.use_count += 1
            r.last_used_at = now
            r.updated_at = now
            if r.level > 1: r.level = 1
            
            # Recalculate Petroff's activation value
            r.activation = self.compute_petroff_activation(s, r.id, now)
            s.flush()

            # Cascade: Spreading Activation to linked semantic neighbors
            try:
                self.spread_activation(s, r.id, now)
            except Exception as e:
                ASCIIColors.warning(f"Spreading activation failed: {e}")

            return self._to_dict(r)

    def delete(self, memory_id: str) -> bool:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return False
            s.delete(r)
            return True

    def clear_level(self, level: int) -> int:
        with self._session() as s:
            q = self._q(s).filter(_MemoryRecord.level == level)
            count = q.count()
            q.delete(synchronize_session=False)
            return count

    def edit_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        level: Optional[int] = None,
        tags: Optional[List[str]] = None,
        subject_group: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Manually edit any aspect of a memory record.
        Enables applications to provide a direct user interface (UI) to manage memories.
        """
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None:
                return None
            
            if content is not None:
                r.content = content.strip()
                r.summary = self._auto_summary(content)
            if importance is not None:
                r.importance = max(0.0, min(1.0, float(importance)))
            if level is not None:
                r.level = int(level)
            if tags is not None:
                r.tags = ",".join(tags) if tags else None
            if subject_group is not None:
                r.subject_group = subject_group
                
            r.updated_at = datetime.utcnow()
            s.flush()
            return self._to_dict(r)

    def list_all(
        self,
        level: Optional[int] = None,
        search_query: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """
        List all memories with optional level filtering, search, and pagination.
        Ideal for building user management control panels.
        """
        with self._session() as s:
            q = self._q(s)
            if level is not None:
                q = q.filter(_MemoryRecord.level == level)
            if search_query:
                # Case-insensitive SQL substring match
                q = q.filter(_MemoryRecord.content.ilike(f"%{search_query}%"))
                
            total = q.count()
            recs = q.order_by(_MemoryRecord.importance.desc())\
                    .offset((page - 1) * page_size)\
                    .limit(page_size)\
                    .all()
            
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "pages": (total + page_size - 1) // page_size,
                "memories": [self._to_dict(r) for r in recs]
            }

    def load_to_working(self, memory_id: str) -> Optional[Dict]:
        with self._session() as s:
            r = self._q(s).filter(_MemoryRecord.id == memory_id).first()
            if r is None: return None
            r.level = 1
            r.last_used_at = datetime.utcnow()
            r.updated_at = datetime.utcnow()
            r.importance = max(r.importance, self.config.demotion_threshold + 0.1)
            s.flush()

            # Cascade and pull associated memories into Level 2
            try:
                self.auto_pull_associative_memories(memory_id)
            except Exception as e:
                ASCIIColors.warning(f"Associative memory propagation failed: {e}")

            return self._to_dict(r)

    def auto_pull_deep_memories(self, user_message: str, top_k: int = 3) -> List[Dict]:
        """
        Auto-grep deep memories matching keywords in user_message and load them into working memory.
        Enables fast proactive cognitive recall without requiring an explicit LLM-emitted load tag.
        """
        matching_deep = self.query(user_message, top_k=top_k, level=2)
        pulled = []
        for m in matching_deep:
            loaded = self.load_to_working(m["id"])
            if loaded:
                pulled.append(loaded)
                ASCIIColors.cyan(
                    f"[Memory] Detected deep memory cue — Auto-pulled '{m['id'][:8]}' "
                    f"('{m['content'][:30]}...') into Working Memory."
                )
        return pulled

    def list_working(self) -> List[Dict]:
        with self._session() as s:
            recs = (self._q(s).filter(_MemoryRecord.level == 1).order_by(_MemoryRecord.importance.desc()).all())
            return [self._to_dict(r) for r in recs]

    def query(self, text: str, top_k: int = 5, level: Optional[int] = None) -> List[Dict]:
        """Fast on-the-fly TF-IDF keyword query matching over database memories."""
        import math
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        query_words = [w for w in cleaned_text.lower().split() if len(w) >= 3 and w not in {
            'the', 'and', 'are', 'was', 'were', 'for', 'you', 'your', 'with', 'this', 'that',
            'they', 'them', 'their', 'our', 'what', 'who', 'how', 'why', 'can', 'not', 'but',
            'some', 'has', 'have', 'had', 'did', 'does', 'get', 'out', 'into', 'from', 'about',
            'way', 'yet', 'all', 'any', 'one', 'two', 'she', 'him', 'her', 'its', 'his'
        }]
        if not query_words:
            query_words = [w for w in cleaned_text.lower().split() if len(w) >= 2]
            if not query_words:
                return []

        with self._session() as s:
            q = self._q(s)
            if level is not None:
                q = q.filter(_MemoryRecord.level == level)
            records_raw = q.order_by(_MemoryRecord.importance.desc()).limit(200).all()
            dicts = [self._to_dict(r) for r in records_raw]

        if not dicts:
            return []

        total_docs = len(dicts)
        df_map = {}
        for w in query_words:
            df_map[w] = sum(1 for d in dicts if w in d['content'].lower())

        idf_map = {}
        for w, df in df_map.items():
            idf_map[w] = math.log(1.0 + (total_docs / (1.0 + df)))

        scored = []
        for d in dicts:
            content_lower = d['content'].lower()
            doc_words = content_lower.split()
            score = 0.0
            for w in query_words:
                tf = doc_words.count(w)
                if tf > 0:
                    tf_scaled = 1.0 + math.log(tf)
                    score += tf_scaled * idf_map[w]
            if score > 0.0:
                scored.append((score + (d['importance'] * 0.1), d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    # ──────────────────────────────────────────────── context assembly
    # ──────────────────────────────────────────────── context assembly
    def build_working_zone(self, token_counter=None) -> str:
        # Check cache
        if self._working_zone_cache is not None:
            return self._working_zone_cache

        records = self.list_working()
        if not records: return ""
        # Sort chronologically by created_at (oldest first, newest last)
        records.sort(key=lambda x: x["created_at"])
        budget, lines, used_tok = self.config.working_token_budget, [], 0
        for r in records:
            ts = r['created_at'][:19].replace('T', ' ')
            entry = f"[{ts}] [{r['id'][:8]}] ({r['importance']:.0%}) [Centrality: {r['centrality']:.0%}] {r['content']}"
            if r.get('tags'): entry += f"  #{r['tags'].replace(',', ' #')}"
            tok = token_counter(entry) if token_counter else len(entry) // 4
            if used_tok + tok > budget: break
            lines.append(entry)
            used_tok += tok

        # Query relationships between these working memory nodes
        rel_lines = []
        if lines:
            active_ids = {r['id'] for r in records[:len(lines)]}
            with self._session() as s:
                rels = s.query(_MemoryRelationship).filter(
                    _MemoryRelationship.source_id.in_(active_ids),
                    _MemoryRelationship.target_id.in_(active_ids)
                ).all()
                for rel in rels:
                    rel_lines.append(f"  [{rel.source_id[:8]}] --({rel.relationship_type})--> [{rel.target_id[:8]}]")

        rel_block = ""
        if rel_lines:
            rel_block = "\n\n=== SEMANTIC CONNECTIONS ===\n" + "\n".join(rel_lines) + "\n=== END CONNECTIONS ===\n"

        result = "=== WORKING MEMORY ===\n" + "\n".join(lines) + rel_block + "\n=== END WORKING MEMORY ===\n" if lines else ""
        self._working_zone_cache = result
        return result

    def build_handles_zone(self, token_counter=None) -> str:
        # Check cache
        if self._handles_zone_cache is not None:
            return self._handles_zone_cache
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
        result = "=== DEEP MEMORY HANDLES ===\n(Use <mem_load id=\"ID\"/> to bring into working memory)\n" + body + "\n=== END DEEP MEMORY HANDLES ===\n"
        self._handles_zone_cache = result
        return result

    def build_system_instructions(self) -> str:
        return (
            "\n=== MEMORY SYSTEM ===\n\n"
            "You are equipped with a multi-level persistent memory system with GRAPH-BASED relationships to recall facts, user preferences, and past events/conversations across turns.\n\n"
            "── Working Memory: Active memories and past conversation episodes appear in the WORKING MEMORY zone below (injected into your context). They are shown in chronological order with their timestamp, [ID], importance, and centrality (graph connectivity).\n"
            "   👉 CRITICAL RULE: When you utilize or refer to an active memory [ID] to answer, you MUST prepend `<mem_tag id=\"ID\" />` to your response so the system can track its usage.\n"
            "── Deep Memory: Stored memories and past episodes that are currently inactive. Only their compact handles appear. "
            "If you see a handle (e.g. [abc123de]) that contains information needed to answer the user's question, you MUST load it first by outputting `<mem_load id=\"ID\" />`.\n"
            "── Memory Graph: Memories are connected via explicit relationships. High-centrality memories are more connected and important.\n"
            "── Tags Available:\n"
            "   • `<mem_new importance=\"...\">content</mem_new>` — Save a new fact, preference, or event (importance is a float, default 0.75).\n"
            "     🚨 CURATION PROTOCOL (ONLY SAVE HIGH-DENSITY, PERSISTENT KNOWLEDGE):\n"
            "     ✅ ALWAYS SAVE: Core user preferences (e.g., language choice, custom guidelines), persistent architectural decisions, structural rules, and primary project goals.\n"
            "     ❌ NEVER SAVE: Casual pleasantries, greetings, small talk, weather, conversational fluff, temporary feelings, or large raw code blocks.\n"
            "   • `<mem_update id=\"ID\">content</mem_update>` — Update an existing memory's verbatim content.\n"
            "   • `<mem_tag id=\"ID\" />` — Tag/acknowledge that a memory was retrieved and used to answer the user.\n"
            "   • `<mem_load id=\"ID\" />` — Load an inactive Deep Memory into active Working Memory.\n"
            "   • `<mem_delete id=\"ID\" />` — Delete a memory that is no longer correct or relevant.\n"
            "   • `<mem_rel source=\"ID\" target=\"ID\" type=\"TYPE\" weight=\"1.0\" />` — Create a graph relationship between memories.\n"
            "     Relationship types: RELATED_TO, DERIVED_FROM, CONTRADICTS, SUPPORTS, TEMPORAL_AFTER\n\n"
            "── Rules: Use exact 8-character ID prefixes. All tags are automatically stripped from your reply before display.\n"
            "── CRITICAL: Memory tags are processed silently by the system. After using any memory tag, you MUST continue with a natural, helpful conversational response to the user. Do not stop after only emitting a memory tag.\n"
            "=== END MEMORY SYSTEM ===\n"
        )

    _PAT_REL = re.compile(r'<mem_rel\s+source=["\']([^"\']+)["\']\s+target=["\']([^"\']+)["\'](?:\s+type=["\']([^"\']+)["\'])?(?:\s+weight=["\']([^"\']+)["\'])?(?:\s*/)?>', re.I)

    def process_llm_output(self, text: str) -> Tuple[str, Dict[str, Any]]:
        cleaned, report = text, {"created": [], "updated": [], "tagged": [], "deleted": [], "loaded": [], "relationships": []}
        for m in self._PAT_UPDATE.finditer(text):
            full_id = self._resolve_id(m.group(1))
            if full_id:
                res = self.update(full_id, m.group(2).strip())
                if res: report["updated"].append(res)
            cleaned = cleaned.replace(m.group(0), "", 1)
        for m in self._PAT_NEW.finditer(text):
            attrs = {attr_m.group(1).lower(): attr_m.group(2)
                     for attr_m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', m.group(1))}

            imp_str = attrs.get("importance")
            try:
                imp = float(imp_str) if imp_str else self.config.default_importance
            except ValueError:
                imp = self.config.default_importance

            content = m.group(2) if m.group(2) is not None else attrs.get("content", "")
            if not content or not content.strip():
                continue

            tags_list = [t.strip().lower() for t in attrs.get("tags", "").split(",") if t.strip()] if attrs.get("tags") else None
            res = self.add(
                content=content.strip(),
                importance=max(0.0, min(1.0, imp)),
                tags=tags_list,
                subject_group=attrs.get("subject_group") or attrs.get("subject"),
                subject=attrs.get("subject"),
                predicate=attrs.get("predicate"),
                obj=attrs.get("object") or attrs.get("obj")
            )
            if res:
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
        # Process relationship tags
        for m in self._PAT_REL.finditer(text):
            source_id = self._resolve_id(m.group(1))
            target_id = self._resolve_id(m.group(2))
            rel_type = m.group(3) or "RELATED_TO"
            try:
                weight = float(m.group(4)) if m.group(4) else 1.0
            except ValueError:
                weight = 1.0
            if source_id and target_id:
                res = self.add_relationship(source_id, target_id, rel_type, weight)
                if res: report["relationships"].append(res)
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
            for r in self._q(s).filter(_MemoryRecord.level <= 3).all():
                delta = (now - r.updated_at).total_seconds() / 86400.0
                new_imp = max(0.0, r.importance - (self.config.decay_rate_per_day * delta))
                if abs(new_imp - r.importance) > 0.001:
                    r.importance = new_imp
                    r.updated_at = now
                    count += 1
                if r.level == 1 and r.importance < self.config.demotion_threshold: r.level = 2
                elif r.level == 2 and r.importance < self.config.archive_threshold: r.level = 3
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
        """Consolidation pass — recompute importance, build clusters, and prune low-value memories."""
        elapsed = (datetime.utcnow() - self._last_dream).total_seconds() / 3600
        if elapsed < self.config.dream_min_interval_hours:
            return {"skipped": True, "reason": "too_soon"}
            
        start = datetime.utcnow()
        report = {
            "decayed": self.apply_decay(),
            "promoted": 0,
            "reinforced": 0,
            "forgotten": 0,
            "retained_by_dreamer": 0,
            "duration_seconds": 0.0
        }
        
        # 1. Usage-based Reinforcement & Decay adjustments
        # Memories with active retrieval (use_count > 0) are boosted logarithmically to reinforce them
        with self._session() as s:
            for r in self._q(s).all():
                if r.use_count > 0:
                    boost = math.log1p(r.use_count) * 0.05
                    r.importance = min(1.0, r.importance + boost)
                    report["reinforced"] += 1
                    # Gradual reduction of use_count so memory can decay again if left unused
                    r.use_count = max(0, r.use_count - 1)
        
        # 2. Rule-based promotion for recently used archived memories
        with self._session() as s:
            for r in self._q(s).filter(_MemoryRecord.level == 3).all():
                if (datetime.utcnow() - r.last_used_at).total_seconds() < 86400:
                    r.level = 2; report["promoted"] += 1
                    
        self.enforce_budget()
        
        # 3. Forgetting Pass: Faded memories (Level 3 with importance < forget_threshold) are purged
        #    unless the dreamer thinks they are still important
        to_evaluate = []
        with self._session() as s:
            candidates = self._q(s).filter(_MemoryRecord.level == 3).filter(_MemoryRecord.importance < self.config.forget_threshold).all()
            for c in candidates:
                to_evaluate.append({
                    "id": c.id,
                    "content": c.content,
                    "importance": c.importance,
                    "tags": c.tags
                })

        for item in to_evaluate:
            keep = False
            reason = "Automatic cleanup (no dreamer available)"
            if lollms_client:
                prompt = (
                    "You are the Dreamer, the subconscious mind of an AI.\n"
                    "You are consolidating your long-term memories. The following memory has faded due to lack of use:\n\n"
                    f"Memory Content: \"{item['content']}\"\n"
                    f"Importance Score: {item['importance']:.3f}\n"
                    f"Tags: {item['tags'] or 'None'}\n\n"
                    "Is this memory still important to preserve for the future?\n"
                    "If it contains critical personal experiences, architectural constraints, key facts, or core user preferences, keep it.\n"
                    "If it is trivial, redundant, or outdated, forget it."
                )
                try:
                    res = lollms_client.generate_structured_content(
                        prompt=prompt,
                        schema={
                            "keep": "boolean",
                            "reason": "string"
                        },
                        temperature=0.1
                    )
                    if res:
                        keep = res.get("keep", False)
                        reason = res.get("reason", "")
                except Exception as ex:
                    ASCIIColors.warning(f"[MemoryManager] Dreamer evaluation failed: {ex}")
            
            if keep:
                # Retain the memory and restore its importance to demotion threshold + small buffer
                with self._session() as s:
                    r = s.get(_MemoryRecord, item["id"])
                    if r:
                        r.importance = self.config.demotion_threshold + 0.05
                        r.last_used_at = datetime.utcnow()
                        r.level = 1  # Promote rescued memory back to active Working Memory (Level 1)
                        report["retained_by_dreamer"] += 1
                        ASCIIColors.info(f"[MemoryManager] Retained memory '{item['id'][:8]}' by dreamer: {reason}")
            else:
                # Forget/delete permanently
                self.delete(item["id"])
                report["forgotten"] += 1
                ASCIIColors.warning(f"[MemoryManager] Forgotten memory '{item['id'][:8]}': {reason}")
                
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
            "level": r.level, "importance": round(r.importance, 4), "centrality": round(r.centrality, 4) if r.centrality else 0.0,
            "use_count": r.use_count, "tags": r.tags, "subject_group": r.subject_group,
            "created_at": r.created_at.isoformat(), "updated_at": r.updated_at.isoformat(),
            "last_used_at": r.last_used_at.isoformat(), "subject": r.subject, "predicate": r.predicate,
            "object": r.object, "activation": round(r.activation, 4) if r.activation is not None else 0.0
        }

    # ──────────────────────────────────────────────── GRAPH OPERATIONS

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str = "RELATED_TO", weight: float = 1.0, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Create an explicit graph edge between two memories."""
        now = datetime.utcnow()
        with self._session() as s:
            # Verify both nodes exist
            source = s.get(_MemoryRecord, source_id)
            target = s.get(_MemoryRecord, target_id)
            if not source or not target:
                return None

            # Check for existing relationship
            existing = s.query(_MemoryRelationship).filter(
                _MemoryRelationship.source_id == source_id,
                _MemoryRelationship.target_id == target_id
            ).first()
            if existing:
                existing.relationship_type = relationship_type
                existing.weight = weight
                existing.relationship_metadata = json.dumps(metadata) if metadata else None
                existing.created_at = now
                s.flush()
                return self._rel_to_dict(existing)

            rel = _MemoryRelationship(
                id=str(uuid.uuid4()), source_id=source_id, target_id=target_id,
                relationship_type=MemoryOntology.validate_predicate(relationship_type), weight=max(0.0, min(10.0, weight)),
                relationship_metadata=json.dumps(metadata) if metadata else None, created_at=now
            )
            s.add(rel)
            s.flush()

            # Recalculate centrality for affected nodes
            self._recalculate_centrality(s, source_id)
            self._recalculate_centrality(s, target_id)

            return self._rel_to_dict(rel)

    def remove_relationship(self, source_id: str, target_id: str) -> bool:
        """Remove a graph edge between two memories."""
        with self._session() as s:
            rel = s.query(_MemoryRelationship).filter(
                _MemoryRelationship.source_id == source_id,
                _MemoryRelationship.target_id == target_id
            ).first()
            if rel:
                s.delete(rel)
                self._recalculate_centrality(s, source_id)
                self._recalculate_centrality(s, target_id)
                return True
            return False

    def get_relationships(self, memory_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Get all relationships for a memory node."""
        with self._session() as s:
            q = s.query(_MemoryRelationship).filter(
                (_MemoryRelationship.source_id == memory_id) | (_MemoryRelationship.target_id == memory_id)
            )
            if relationship_type:
                q = q.filter(_MemoryRelationship.relationship_type == relationship_type.upper())
            return [self._rel_to_dict(r) for r in q.all()]

    def traverse_graph(self, start_id: str, max_depth: int = 3, relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """BFS traversal of memory graph from a starting node."""
        visited = set()
        queue = [(start_id, 0)]  # (node_id, depth)
        results = []

        with self._session() as s:
            while queue:
                node_id, depth = queue.pop(0)
                if node_id in visited or depth > max_depth:
                    continue
                visited.add(node_id)

                # Get node
                node = s.get(_MemoryRecord, node_id)
                if node:
                    node_dict = self._to_dict(node)
                    node_dict['_depth'] = depth
                    results.append(node_dict)

                # Get neighbors
                rels = s.query(_MemoryRelationship).filter(
                    (_MemoryRelationship.source_id == node_id) | (_MemoryRelationship.target_id == node_id)
                )
                if relationship_types:
                    rels = rels.filter(_MemoryRelationship.relationship_type.in_([t.upper() for t in relationship_types]))

                for rel in rels:
                    neighbor_id = rel.target_id if rel.source_id == node_id else rel.source_id
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, depth + 1))

        return results

    def _recalculate_centrality(self, session: Session, memory_id: str):
        """Calculate PageRank-like centrality score for a memory node."""
        # Get all relationships for this node
        rels = session.query(_MemoryRelationship).filter(
            (_MemoryRelationship.source_id == memory_id) | (_MemoryRelationship.target_id == memory_id)
        ).all()

        if not rels:
            # No connections = low centrality
            node = session.get(_MemoryRecord, memory_id)
            if node:
                node.centrality = 0.1
            return

        # Simple centrality: weighted degree / max possible degree
        total_weight = sum(r.weight for r in rels)
        node = session.get(_MemoryRecord, memory_id)
        if node:
            # Normalize: centrality = min(1.0, total_weight / 10.0)
            node.centrality = min(1.0, total_weight / 10.0)

    def recalculate_all_centrality(self):
        """Recalculate centrality for all memories (expensive operation)."""
        with self._session() as s:
            for r in self._q(s).all():
                self._recalculate_centrality(s, r.id)

    def get_high_centrality_memories(self, top_k: int = 10, level: Optional[int] = None) -> List[Dict]:
        """Get memories with highest graph centrality (most connected/important)."""
        with self._session() as s:
            q = self._q(s).filter(_MemoryRecord.centrality > 0.0)
            if level is not None:
                q = q.filter(_MemoryRecord.level == level)
            recs = q.order_by(_MemoryRecord.centrality.desc()).limit(top_k).all()
            return [self._to_dict(r) for r in recs]

    def _rel_to_dict(self, r: _MemoryRelationship) -> Dict:
        return {
            "id": r.id, "source_id": r.source_id, "target_id": r.target_id,
            "relationship_type": r.relationship_type, "weight": r.weight,
            "metadata": json.loads(r.relationship_metadata) if r.relationship_metadata else {},
            "created_at": r.created_at.isoformat()
        }
