# 🔐 Shared Model Server Daemon & IPC Architecture

---

## 1. Architectural Challenge

In multi-process Python deployments (e.g. multiple Uvicorn workers, task queues, or background scripts), loading deep learning models into each process causes:
1. **Linear VRAM Waste**: $N$ processes $\times$ 15 GB weights = Instant GPU Out-of-Memory.
2. **Zero Concurrency Consolidation**: Concurrent requests from different workers execute in isolation without batching.
3. **Port Collisions**: Uncoordinated server spawning causes race conditions and port drift.

---

## 2. The Phenix IPC Lifecycle

```
[ Worker Process A ]                     [ FileLock ]                 [ Shared Daemon Server ]
        │                                     │                                  │
        ├─► is_server_running() (<= 1.5s)? ───┼─────────────────────────────────►│
        │   └─► Returns False                 │                                  │
        │                                     │                                  │
        ├─► Acquire FileLock(timeout=120s) ──►│ (Acquired)                       │
        │                                     │                                  │
        ├─► Double-Check Probe ───────────────┼─────────────────────────────────►│
        │   └─► Still False                   │                                  │
        │                                     │                                  │
        ├─► Spawn Subprocess (main.py) ───────┼─────────────► Spawns ───────────►│
        │   (CREATE_NO_WINDOW / start_new_session)                              │
        │                                     │               Writes Token (0o600)
        │                                     │               Binds Port FIRST   │
        │                                     │               Loads Model Weights│
        │                                     │                                  │
        ├─► Poll is_server_running() ─────────┼─────────────────────────────────►│
        │   └─► Returns True (Ready!)         │                                  │
        │                                     │                                  │
        ├─► Release FileLock ────────────────►│ (Released)                       │
        ▼                                     ▼                                  ▼
[ Worker Process B ]                          │                                  │
        │                                     │                                  │
        ├─► Acquire FileLock                  │                                  │
        ├─► Double-Check Probe ───────────────┼─────────────────────────────────►│
        │   └─► Returns True! (Already Up)    │                                  │
        ├─► Release Lock & Attach Immediately!│                                  │
```

---

## 3. Dynamic Continuous Micro-Batching

Traditional sequential request processing incurs substantial GPU latency overhead. The shared daemon implements **event-driven continuous micro-batching**:

```python
def _batch_worker(self):
    while not self._stop_event.is_set():
        try:
            first_job = self.queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if first_job is None:
            break

        batch = [first_job]

        # 1. Open adaptive micro-batch collection window (e.g. 20ms)
        if self.batch_window > 0:
            time.sleep(self.batch_window)

        # 2. Drain concurrent requests up to max_batch_size
        while len(batch) < self.max_batch_size:
            try:
                job = self.queue.get_nowait()
                if job is None:
                    break
                batch.append(job)
            except queue.Empty:
                break

        # 3. Process batch in a single forward tensor pass
        self._process_batch(batch)
```

### Key Advantages
- **Zero Busy-Waiting**: The worker thread sleeps on OS queue events with 0% CPU consumption during idle periods.
- **Latency vs. Throughput Harmony**: The 20ms micro-window adds imperceptible latency while grouping 2–8 concurrent requests into a single tensor execution pass.
- **Automatic Fallback & Fault Tolerance**: Short audio clips sharing decoding parameters are batched together via stacked log-mel spectrograms; long clips or heterogeneous tasks gracefully fall back to individual execution without stalling the queue.

---

## 4. Hardened Security & Isolation Standards

1. **Constant-Time HMAC Authentication**:
   - Each daemon generates a high-entropy authentication token (`secrets.token_hex(16)`) stored with strict user-only read/write permissions (`0o600`).
   - Every HTTP endpoint verifies request tokens using constant-time `hmac.compare_digest`.
2. **Loopback Binding Isolation (`127.0.0.1`)**:
   - Shared daemons bind strictly to `127.0.0.1` by default, preventing unintended network exposure.
3. **Crash-Safe Process Spawning**:
   - Windows processes launch with `subprocess.CREATE_NO_WINDOW`, closing parent log handles immediately to prevent Windows file-sharing locks (`WinError 32`).
   - POSIX systems launch with `start_new_session=True` to prevent process group signal cascades.