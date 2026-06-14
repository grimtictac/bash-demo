# Issue: ModelSlot, Rollback Correctness, and Thread-Safe Metrics

## Problems

### 1. Rollback swaps instead of restoring

The original `rollback()` did a Python swap:

```python
self._current_model, self._previous_model = self._previous_model, self._current_model
```

After rollback, the bad model ends up in `_previous_model`. A second call to `rollback()` restores it. This is the opposite of what rollback should do.

### 2. Response reports wrong version after rollback

`PredictionResponse.model_version` was always set from `self._model_version` — the version passed at construction time and never updated. After a rollback to an earlier model, every response continued to report the original version. The response was lying about which model ran.

### 3. `Metrics` is not thread-safe

`incr()` was a bare read-modify-write on a plain dict:

```python
self.counters[name] = self.counters.get(name, 0) + value
```

Under concurrent load, two threads could both read the same value, both increment it, and write back — one increment would be lost.

### 4. `load_model()` has no error handling

A failed repository load propagated the raw exception with no metric increment and no structured log.

## The Fix

`ModelSlot` bundles model and version into a single frozen reference:

```python
@dataclass(frozen=True)
class ModelSlot:
    model: ModelProtocol
    version: Optional[str]
```

`InferenceService` now holds `_slot` and `_previous_slot` instead of four separate fields. A load atomically replaces the slot:

```python
self._previous_slot = self._slot
self._slot = ModelSlot(new_model, self._model_version)
```

In CPython, reference assignment is atomic — a predict thread reading `self._slot` always sees either the old slot or the new one, never a partial state.

Rollback restores and clears:

```python
self._slot = self._previous_slot
self._previous_slot = None
```

Setting `_previous_slot = None` means a second rollback returns `False` — there is nothing to go back to.

`predict()` captures the slot once at the start of the call:

```python
slot = self._slot
```

The response is built from `slot.version`, so it always reflects the model that actually ran — correct before, during, and after a rollback.

`Metrics` now uses its own lock so concurrent `incr()` calls are safe:

```python
def incr(self, name: str, value: int = 1) -> None:
    with self._lock:
        self._counters[name] = self._counters.get(name, 0) + value
```

`load_model()` wraps the repository call in a try/except, increments `model_load_failure` on error, and emits structured log lines at load start and completion.

## How to Demonstrate

A plain Python script (no server needed) can verify all three behaviours by calling `InferenceService` directly:

**Double-rollback prevention:**
Load `healthy`, then attempt to load `load_failure` (which raises). Call `rollback()` — should return `True` and restore the healthy model. Call `rollback()` a second time — should return `False`. In the original code the second call returned `True` and toggled back to an undefined state.

**load_failure metric and log:**
Load `load_failure` and assert that `service.metrics.counters["model_load_failure"] == 1` and that `model_load_success` was not incremented. The structured error log should also fire.

**Version in response:**
This requires two distinct model versions, which needs the rollover machinery — better deferred to the rollover demo.

A `demo_rollback.py` script covering the first two cases would be a natural addition alongside the other demo shims.

## What Changed

- `_current_model` / `_previous_model` → `_slot` / `_previous_slot`
- `threading.RLock` → `threading.Lock` (load path only; predict needs no lock)
- Rollback: swap → restore + clear
- `PredictionResponse.model_version`: `self._model_version` → `slot.version`
- `Metrics`: plain dataclass → class with internal lock
- `load_model()`: no error handling → try/except with metric and log
