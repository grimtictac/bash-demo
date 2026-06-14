# DECISIONS.md

# MAKE SURE TO READ SUMMARY.MD!

---

## 1. Proven — code, documentation, and measured data

**Concurrency fix:**
The original `async def predict_endpoint` called `service.predict()` synchronously, stalling the event loop and serialising all in-flight requests. Fixed with `await run_in_threadpool(service.predict, parsed)`. Measured with 50 concurrent requests against `SlowModel` (150ms/call): wall clock dropped from 7,784ms (47.4x baseline) to 794ms (4.8x baseline). See `docs/01-serialisation.md` and `demo_slow_parallel.py` / `demo_slow_serialised.py`.

**Semantic output validation:**
Extended `validate_response` to reject `None` predictions and non-finite float values anywhere in the prediction structure. The original only checked count. Demonstrated in `demo_validation.py` — both failure modes return `200 OK` without the fix and `503` with it. See `docs/03-output-validation.md`.

**Rollover — swap does not stall in-flight requests:**
The original held a single `_lock` for the full duration of `load_model()`, blocking all concurrent `predict()` calls for the entire load time. `ModelSlot` and `_load_lock` separate the two concerns: loads serialise against each other, predict reads the slot atomically without any lock. Demonstrated in `demo_rollover_manual.py` (12-second simulated load, requests continue uninterrupted) and `demo_stall_on_swap.py` (reproduces the original freeze). See `docs/04-rollover.md`.

---

## 2. Implemented and documented — no live demo

**Rollback correctness:**
The original `rollback()` swapped `_current_model` and `_previous_model`, leaving the bad model in `_previous_model`. A second rollback would restore it. The fix restores and clears: `_slot = _previous_slot; _previous_slot = None`. A second call returns `False`. Documented in `docs/02-locking-and-rollback.md`. A `demo_rollback.py` script is noted there as a natural addition to exercise this path directly.

**Failed load does not replace a good model:**
`load_model()` wraps the repository call in a try/except. On failure, `model_load_failure` is incremented, a structured error is logged, and `ModelLoadError` is raised. The slot is only updated after a successful load.

**Version reflects the model that actually served the request:**
`ModelSlot` carries `version` alongside `model`. `PredictionResponse.model_version` is set from `slot.version`, not the construction-time `_model_version`. Correct before, during, and after a rollback.

**Thread-safe Metrics:**
`Metrics` was a plain dataclass; `incr()` was an unprotected read-modify-write. Converted to a class with an internal `Lock`. Documented in `docs/02-locking-and-rollback.md`.

**Latency metric:**
`predict_latency_ms_total` accumulates inference time in milliseconds via the `finally` block in `predict()`. Mean latency = `predict_latency_ms_total / predict_success`. No dedicated doc or demo — visible in the `/health` response under `metrics`.

---

## 3. Design decisions and tradeoffs

**What I deliberately left unchanged:**
- No new HTTP endpoints. `/admin/load` would be operationally useful but wasn't in the contract.
- `validate_request` is intact (not replaced with Pydantic) because it is part of the public surface.
- Single previous-model slot. A ring buffer of N versions is over-engineering for a one-step rollback use case.
- No async model loading. The thread-pool model is consistent with how the rest of the service handles blocking I/O.

**Main tradeoffs:**

| Decision | Upside | Downside |
|---|---|---|
| `run_in_threadpool` for inference | Event loop stays free; concurrent requests proceed in parallel | anyio thread pool has a fixed cap; pathological load can saturate it |
| `_load_lock` serialises loads | No race between two simultaneous loads | A slow load blocks any concurrent `load_model()` call for its duration |
| `ModelSlot` atomic reference | No lock needed in `predict()`; no torn model/version read | Relies on CPython reference assignment being atomic; not a language guarantee |
| Rollback clears previous slot | No accidental toggle back to a bad model | Operator must trigger a fresh load to establish a new rollback target |
| NaN/Inf check on every prediction | Catches a common silent failure mode from numeric models | O(predictions × fields) traversal; negligible at typical batch sizes |

---

## 4. Assumptions and open questions

**Assumptions made:**
- The model predict contract (`Sequence[Dict]` → `Sequence[Any]`) is fixed. Shape and numeric finiteness are validated but not domain semantics (score range, required keys) because those aren't specified.
- Concurrent loads should be serialised but must never block in-flight inference.
- Rollback is a one-way operation. A new load is required to establish a new rollback target.
- Counter races under high throughput are acceptable. The lock on `Metrics` is a precaution, not a hard requirement.

**Open questions I'd ask before building for real:**
1. What is the semantic contract for a valid prediction? Without a schema we can only check structural and numeric validity.
2. Should `load_model()` accept a new version parameter? Currently version is fixed at construction time.
3. What is the SLO? Without latency and error-rate targets we can't set useful alert thresholds.
4. Is the model repository safe to retry on failure?
5. Should `/health` return `503` when no model is loaded? Right now it always returns `200`.

---

## 5. Advisory — not implemented

**What I'd do next with more time:**
1. **Unhealthy health check.** Return `503` from `/health` when `_slot is None`.
2. **Request-level tracing.** Propagate `request_id` through every log line.
3. **Configurable output schema validation.** JSON Schema or Pydantic at service construction instead of heuristic NaN/Inf checks.
4. **Background model loading** with a status endpoint (`POST /admin/load → 202`, `GET /admin/load/status`).
5. **Explicit in-flight tracking.** A counter of requests currently holding each slot, so we can log how long after a swap the old model is still referenced.

**Deployment / rollback / monitoring (notes only):**

Deploy: container image per commit, behind a load balancer with `/health` checks. Load the model at startup before registering with the LB.

Rollback: `POST /admin/rollback` on each instance via control plane or script. For automation: wire an error-rate alert to a webhook that calls rollback and pages on-call.

Alert on:
- `model_load_failure` > 0 — any load failure is actionable
- `predict_validation_error` rate > 0 — model returning wrong shape or NaN
- `predict_failure` rate > ~0.1% — unexpected exceptions from the model
- `predict_latency_ms_total / predict_success` breaching the SLO
- Thread pool saturation — load exceeding the anyio default pool size
