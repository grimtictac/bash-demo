# DECISIONS.md

## Assumptions and open questions

**Assumptions made:**
- The model predict contract (`Sequence[Dict]` → `Sequence[Any]`) is fixed. I validate shape and numeric finiteness but not domain semantics (e.g., score range, required keys) because those aren't specified.
- `model_version` in responses should reflect the version that *actually served* the request, not the version the service was configured with. The original always returned the init-time version, which would be wrong after a rollback.
- Concurrent loads should be serialised (one at a time) but must never block in-flight inference.
- Rollback is a one-way operation: rolling back clears the previous-model slot. This prevents an operator accidentally toggling forward to a model that was rolled back because it was bad. A new load is required to establish a new rollback target.
- Counter races under high throughput are acceptable for metrics. I added a lock to `Metrics` to be safe, but losing a handful of increments would not constitute an incident.

**Open questions I'd ask before building for real:**
1. What is the semantic contract for a valid prediction? Are scores always `float` in `[0, 1]`? Can a prediction be a list instead of a dict? Without a schema we can only check structural and numeric validity.
2. Should `load_model()` accept a new version parameter, or is the version fixed at service construction? Currently version is fixed — every reload picks up the same configured version.
3. What is the SLO? Without latency and error-rate targets we can't set useful alert thresholds.
4. Is the model repository safe to retry on failure? Can we auto-retry a failed load, or does a failure require operator intervention?
5. Should `/health` return 503 when no model is loaded? Right now it always returns 200.
6. How long can a model load take? If loads are slow (tens of seconds), the `_load_lock` will block any concurrent load call for that duration — is that acceptable, or do we need async loading with a status endpoint?

---

## What changed and why

**Concurrency fix (critical):**
The original `async def predict_endpoint` called `service.predict()` synchronously. `model.predict()` is a blocking call (demonstrated by `SlowModel.time.sleep(0.05)`). In an `async def` handler, blocking calls stall the event loop, serialising all in-flight requests behind the one that is executing. Fixed with `await run_in_threadpool(service.predict, parsed)`, which delegates inference to FastAPI's default anyio thread pool so the event loop remains free for other requests.

The service-layer lock was already correct: only the pointer read (`model = self._current_model`) was held under the lock, not the inference call itself. The serialisation was purely in the HTTP layer.

**Load no longer holds any lock during I/O:**
The original held `_lock` for the entire `load_model()` call (including the slow `repository.load()` I/O). I separated concerns into two locks:
- `_load_lock` serialises concurrent `load_model()` calls, held for the full duration of loading.
- `_swap_lock` (formerly `_lock`, renamed for clarity) protects the model pointer swap only — held for microseconds, never during I/O or inference.

This means a model load can never stall in-flight predict calls, even if the load takes minutes.

**Failed load never replaces a good model:**
The original code was already safe here (exception exits before the assignment). I made it explicit: the swap is only performed after a successful load. A `ModelLoadError` is raised on failure with a dedicated metric (`model_load_failure`).

**Version tracking per slot:**
Added `_current_version` / `_previous_version` alongside the model slots. The response now reports the version that actually served the request, which changes correctly after a rollback.

**Semantic output validation:**
Extended `validate_response` to reject `None` predictions and recursively check for `NaN` / `Inf` in any float values. The original only checked shape. A model returning `{"score": float("nan")}` would have passed through to the caller.

**Thread-safe Metrics:**
Converted from a dataclass to a plain class with an internal `Lock`. The `counters` property now returns a snapshot. The `Metrics` interface (`.incr()`, `.counters`) is unchanged.

**Latency metric:**
Added `predict_latency_ms_total` (a cumulative sum). Mean latency = `predict_latency_ms_total / predict_success`. Useful for on-call dashboards without requiring a histogram library.

---

## What I deliberately left unchanged

- No new HTTP endpoints. `/admin/load` would be operationally useful but wasn't in the contract.
- The `validate_request` Python API is intact and extended (not replaced with a Pydantic model) because it is part of the public surface.
- Single previous-model slot. A ring buffer of N versions is over-engineering for a one-step rollback use case.
- No async model loading. The thread-pool model is consistent with how the rest of the service handles blocking I/O, and keeps the interface simple.

---

## Main tradeoffs

| Decision | Upside | Downside |
|---|---|---|
| `run_in_threadpool` for inference | Event loop stays free; concurrent requests proceed in parallel | anyio thread pool has a fixed cap; pathological load can saturate it |
| `_load_lock` serialises loads | No slot-assignment race between two simultaneous loads | A slow load blocks any concurrent `load_model()` call for its duration |
| Rollback clears previous slot | No accidental toggle back to a bad model | Operator must trigger a fresh load to establish a new rollback target |
| NaN/Inf check on every prediction | Catches a common silent failure mode from numeric models | O(predictions × fields) traversal; negligible at typical batch sizes |

---

## What I'd do next with more time

1. **Request-level tracing.** Propagate `request_id` (or a generated trace ID) through every log line and into `PredictionResponse`.
2. **Unhealthy health check.** Return `503` from `/health` when `_current_model is None`.
3. **Configurable output schema validation.** JSON Schema or a Pydantic model passed in at service construction, instead of heuristic NaN/Inf checks.
4. **Background model loading** with a status endpoint. Operators should not have to block on a slow load; a `POST /admin/load → 202 Accepted` pattern with `GET /admin/load/status` is the natural extension.
5. **Explicit in-flight tracking.** A counter of requests currently using each model slot, so we can log how long after a swap the old model is still referenced.

---

## Deployment / rollback / monitoring (notes only)

**Deploy:** Build a container image per commit. Deploy behind a load balancer with a `/health` check. Rolling deploy: bring up new instances, wait for health checks to pass, shift traffic gradually, drain old instances. Model loading should happen at startup (before the instance is registered with the LB), not lazily on first request.

**Rollback:** If predictions degrade post-deploy, call `POST /admin/rollback` on each instance via the control plane or a short script. For automation: wire the error-rate alert to a webhook that calls rollback and pages on-call.

**Alert on:**
- `model_load_failure` > 0 — any load failure is actionable
- `predict_validation_error` rate > 0 — model returning wrong shape or NaN; likely a model regression
- `predict_failure` rate > ~0.1% — unexpected exceptions from the model
- HTTP 503 rate from the load balancer — correlate with the above
- `predict_latency_ms_total / predict_success` P99 breaching the SLO
- Thread pool saturation (if instrumented) — indicates the load is exceeding the anyio default pool size
- Process RSS growth over time — potential model memory leak if old model objects are not being garbage collected
