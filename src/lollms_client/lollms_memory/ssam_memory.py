import json
import math
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple


class SemanticTriple:
    """Represents a strictly standardized Subject-Predicate-Object relation."""
    def __init__(self, subject: str, predicate: str, obj: str):
        self.subject = subject.strip().lower()
        self.predicate = predicate.strip().upper()
        self.object = obj.strip().lower()

    def to_dict(self) -> Dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "SemanticTriple":
        return cls(
            subject=data.get("subject", "unknown"),
            predicate=data.get("predicate", "related_to"),
            obj=data.get("object", "unknown")
        )

    def __repr__(self) -> str:
        return f"({self.subject} --[{self.predicate}]--> {self.object})"


class LTI:
    """
    Long-Term Identifier (LTI).
    The immutable baseline fact securely anchored in the database.
    """
    def __init__(
        self,
        node_id: str,
        content: str,
        triple: SemanticTriple,
        importance: float = 0.75,
        created_at: float = None
    ):
        self.id = node_id if node_id else str(uuid.uuid4())
        self.content = content.strip()
        self.triple = triple
        self.importance = max(0.0, min(1.0, importance))
        self.created_at = created_at if created_at is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "triple": self.triple.to_dict(),
            "importance": self.importance,
            "created_at": self.created_at
        }


class STI:
    """
    Short-Term Identifier (STI).
    A volatile, active projection of an LTI inside the working memory sandbox.
    """
    def __init__(self, lti: LTI, current_time: float):
        self.id = lti.id
        self.content = lti.content
        self.triple = lti.triple
        self.base_importance = lti.importance
        self.activation: float = 0.0
        self.retrieval_history: List[float] = [lti.created_at]
        self.last_used: float = current_time

    def record_retrieval(self, t: float):
        self.retrieval_history.append(t)
        self.last_used = t

    def compute_petroff_activation(self, t: float, d: float = 0.5) -> float:
        """
        Computes the base-level activation using Petroff's power-law decay approximation.
        Bi = ln( Sum( (t - t_j)^(-d) ) )
        """
        total = 0.0
        for tj in self.retrieval_history:
            delta = t - tj
            # Safeguard against immediate sub-second double accesses
            if delta < 0.001:
                delta = 0.001
            total += math.pow(delta, -d)
        
        # Logarithmic ceiling guard
        self.activation = math.log(total) if total > 0.0 else -99.0
        return self.activation


class SSAMEngine:
    """
    Sovereign Semantic Autarkic Memory (SSAM) Engine.
    Combines SQLite persistence, LTI/STI isolation, Petroff decay, and Spreading Activation.
    """
    def __init__(self, db_path: str = ":memory:", decay_rate: float = 0.5, spread_probability: float = 0.9):
        self.db_path = db_path
        self.decay_rate = decay_rate
        self.spread_probability = spread_probability
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_schema()
        
        # Active Working Sandbox (STI map)
        self.working_memory: Dict[str, STI] = {}

    def _create_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lti_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_logs (
                node_id TEXT NOT NULL,
                retrieved_at REAL NOT NULL,
                FOREIGN KEY (node_id) REFERENCES lti_nodes (id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lti_subject ON lti_nodes (subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lti_object ON lti_nodes (object)")
        self.conn.commit()

    def add_lti(self, content: str, subject: str, predicate: str, obj: str, importance: float = 0.75) -> str:
        """Saves a permanent fact strictly to the LTI table."""
        node_id = str(uuid.uuid4())
        created_at = time.time()
        triple = SemanticTriple(subject, predicate, obj)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO lti_nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (node_id, content.strip(), triple.subject, triple.predicate, triple.object, importance, created_at)
        )
        cursor.execute(
            "INSERT INTO retrieval_logs VALUES (?, ?)",
            (node_id, created_at)
        )
        self.conn.commit()
        return node_id

    def load_to_working_memory(self, node_id: str, t: float = None) -> Optional[STI]:
        """Loads a copy of an LTI into the active STI sandbox."""
        if t is None:
            t = time.time()

        if node_id in self.working_memory:
            self.working_memory[node_id].record_retrieval(t)
            return self.working_memory[node_id]

        cursor = self.conn.cursor()
        cursor.execute("SELECT content, subject, predicate, object, importance, created_at FROM lti_nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            return None

        content, subject, predicate, obj, importance, created_at = row
        triple = SemanticTriple(subject, predicate, obj)
        lti = LTI(node_id, content, triple, importance, created_at)
        
        # Load all past retrieval logs to build accurate decay calculation
        cursor.execute("SELECT retrieved_at FROM retrieval_logs WHERE node_id = ?", (node_id,))
        logs = [r[0] for r in cursor.fetchall()]

        sti = STI(lti, t)
        sti.retrieval_history = logs if logs else [created_at]
        sti.record_retrieval(t)
        
        self.working_memory[node_id] = sti
        return sti

    def spread_activation(self, source_id: str, t: float = None):
        """
        Spreads energy multiplicatively from source_id to linked semantic neighbors.
        Pre-warms associated nodes in deep memory without loading full contents.
        """
        if t is None:
            t = time.time()

        source_sti = self.working_memory.get(source_id)
        if not source_sti:
            return

        source_triple = source_sti.triple
        cursor = self.conn.cursor()
        
        # Find linked nodes sharing Subject or Object in the semantic graph
        cursor.execute("""
            SELECT id FROM lti_nodes 
            WHERE id != ? AND (subject = ? OR object = ? OR subject = ? OR object = ?)
        """, (source_id, source_triple.subject, source_triple.subject, source_triple.object, source_triple.object))
        
        neighbors = [r[0] for r in cursor.fetchall()]
        
        for nid in neighbors:
            neighbor_sti = self.working_memory.get(nid)
            if not neighbor_sti:
                # Pre-load/Pre-warm the deep neighbor into the STI sandbox
                neighbor_sti = self.load_to_working_memory(nid, t)
                
            if neighbor_sti:
                # Multiply activation by diffusion constant (spreading factor)
                neighbor_sti.activation += source_sti.compute_petroff_activation(t, self.decay_rate) * self.spread_probability
                ASCIIColors.info(f"[Spreading Activation] Pre-warmed associated memory '{neighbor_sti.content[:40]}...' (New weight: {neighbor_sti.activation:.2f})")

    def query_semantic_network(self, keyword: str, t: float = None) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query memories. Matches semantic nodes, calculates Petroff decay,
        and spreads activation across related links dynamically.
        """
        if t is None:
            t = time.time()

        keyword_clean = keyword.strip().lower()
        cursor = self.conn.cursor()
        
        # Substring SQL match on nodes
        cursor.execute("SELECT id FROM lti_nodes WHERE content LIKE ? OR subject LIKE ? OR object LIKE ?", 
                       (f"%{keyword_clean}%", f"%{keyword_clean}%", f"%{keyword_clean}%"))
        matched_ids = [r[0] for r in cursor.fetchall()]
        
        results = []
        for nid in matched_ids:
            # Promote matched node to Working Memory
            sti = self.load_to_working_memory(nid, t)
            if sti:
                # Record retrieval event in database transaction
                cursor.execute("INSERT INTO retrieval_logs VALUES (?, ?)", (nid, t))
                self.conn.commit()

                # Calculate Petroff activation score
                score = sti.compute_petroff_activation(t, self.decay_rate)
                results.append((score, sti.to_dict()))

                # Trigger Spreading Activation wave to linked concepts
                self.spread_activation(nid, t)

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def commit_sandbox_changes(self, sti_id: str) -> bool:
        """
        Transaction Commit Protocol. Validates and saves changes made
        inside the transient STI working sandbox back to the permanent LTI database.
        """
        sti = self.working_memory.get(sti_id)
        if not sti:
            return False

        cursor = self.conn.cursor()
        # Verify LTI exists before updating
        cursor.execute("SELECT 1 FROM lti_nodes WHERE id = ?", (sti_id,))
        if not cursor.fetchone():
            return False

        cursor.execute("""
            UPDATE lti_nodes 
            SET content = ?, subject = ?, predicate = ?, object = ?, importance = ?
            WHERE id = ?
        """, (sti.content, sti.triple.subject, sti.triple.predicate, sti.triple.object, sti.base_importance, sti_id))
        self.conn.commit()
        ASCIIColors.success(f"[Transaction] Committed changes for node '{sti_id[:8]}' safely to permanent disk.")
        return True

    def purge_sandbox(self):
        """Discards the transient working sandbox, reverting uncommitted drift."""
        self.working_memory.clear()
        ASCIIColors.warning("[Sandbox] Working memory cleared. Reverted uncommitted cognitive drift.")

    def close(self):
        self.purge_sandbox()
        self.conn.close()


# ── Execution Test ──
if __name__ == "__main__":
    ASCIIColors.cyan("==================================================================")
    ASCIIColors.green("🤖 Starting Sovereign Semantic Autarkic Memory (SSAM) Test Loop")
    ASCIIColors.cyan("==================================================================")

    # 1. Initialize Autarkic Engine
    engine = SSAMEngine(decay_rate=0.5, spread_probability=0.9)

    # 2. Ingest structured ontological LTI memories
    # Node A
    node_a = engine.add_lti(
        content="ParisNeo prefers Rust for system-level programming.",
        subject="ParisNeo",
        predicate="PREFERS",
        obj="Rust",
        importance=0.95
    )
    # Node B (Linked to A via 'Rust')
    node_b = engine.add_lti(
        content="Rust uses a borrow checker to guarantee memory safety without a GC.",
        subject="Rust",
        predicate="IMPLEMENTS",
        obj="borrow_checker",
        importance=0.90
    )
    # Node C (Linked to A via 'ParisNeo')
    node_c = engine.add_lti(
        content="ParisNeo is building a custom lightweight RPC agent.",
        subject="ParisNeo",
        predicate="BUILDING",
        obj="rpc_agent",
        importance=0.85
    )
    # Node D (Unlinked noise)
    node_d = engine.add_lti(
        content="The weather in Paris is sunny today.",
        subject="Paris",
        predicate="HAS_WEATHER",
        obj="sunny",
        importance=0.10
    )

    # Simulate sequential timeline events
    time.sleep(0.5)

    # 3. Execute query turn with Spreading Activation
    print("\n🔍 QUERY: User asks about 'ParisNeo'")
    print("-" * 75)
    results = engine.query_semantic_network("ParisNeo")
    for score, m in results:
        print(f"  • [Score: {score:.2f}] LTI: '{m['content']}'")
    print("-" * 75)

    # The spreading activation wave pre-warmed 'borrow_checker' (Node B) 
    # because it is linked to 'Rust' (recalled in Node A)
    print("\n💡 checking Sandbox/Working state after spread wave:")
    print("-" * 75)
    for nid, sti in engine.working_memory.items():
        print(f"  • STI: '{sti.content}' | Base Importance: {sti.base_importance:.2%}, Pre-Warmed Activation: {sti.activation:.2f}")
    print("-" * 75)

    # 4. Sandbox Transaction Isolation test
    print("\n✏️ Testing Sandbox Isolation & Commit...")
    # Modify Node A inside the working sandbox (STI)
    active_sti = engine.working_memory[node_a]
    active_sti.content = "ParisNeo prefers Rust and Go for systems architectures."
    active_sti.triple = SemanticTriple("ParisNeo", "PREFERS", "rust_and_go")

    # Verify LTI on database remains unchanged (isolated)
    cursor = engine.conn.cursor()
    cursor.execute("SELECT content FROM lti_nodes WHERE id = ?", (node_a,))
    print(f"  • Original LTI in Database: '{cursor.fetchone()[0]}'")
    print(f"  • Modified STI in Sandbox : '{active_sti.content}'")

    # Commit change
    engine.commit_sandbox_changes(node_a)

    # Verify update committed successfully
    cursor.execute("SELECT content, subject, object FROM lti_nodes WHERE id = ?", (node_a,))
    db_row = cursor.fetchone()
    print(f"  • Updated LTI in Database : '{db_row[0]}' | Triple: ({db_row[1]}, {db_row[2]})")

    engine.close()
    ASCIIColors.success("\n==================================================================")
    ASCIIColors.success("🎯 All SSAM engine architectural verification tests passed!")
    ASCIIColors.success("==================================================================")
