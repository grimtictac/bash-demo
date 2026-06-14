# Issue: Model Swap Stalling In-Flight Requests

## The Problem

The original service used a single `_lock` that was held for the full duration of `load_model()` — including the slow `repository.load()` call that fetches the model from the repository. Every `predict()` call had to acquire the same lock to safely read `_current_model`.

This meant a model swap serialised against all in-flight predictions. A load that took 12 seconds would cause every concurrent request to queue behind it, turning a routine hot swap into a 12-second outage.

```python
# original load_model
with self._lock:                          # held for entire load
    new_model = self._repository.load(…)  # slow — could be seconds
    self._current_model = new_model

# original predict
with self._lock:                          # blocked until load finishes
    model = self._current_model
```

## The Fix

`ModelSlot` (introduced in issue 2) makes the lock unnecessary in `predict()`. The entire model and version are read in a single atomic reference assignment — no lock is needed to get a consistent view.

`load_model()` retains `_load_lock`, but only to serialise concurrent loads against each other. `predict()` never touches it.

```python
# after — load_model
with self._load_lock:                     # predict never acquires this
    new_model = self._repository.load(…)
    self._slot = ModelSlot(new_model, version)  # atomic swap

# after — predict
slot = self._slot  # single read, no lock — never blocks on a load
```

During a load, in-flight predictions continue against the current slot. When the load completes, `self._slot` is updated atomically. Requests that captured the slot before the swap complete with the old model; requests that start after the swap use the new one. No request is dropped or stalled.

## Proof

The service was run against a model with a simulated 12-second load time (`demo_rollover_manual.py`). A swap was triggered while requests were in flight. The client logs show uninterrupted responses on `v1` throughout the 12-second load window, followed by a clean transition to `v2` once the load completed.

```
{'predictions': [{'score': 1.0}], 'model_name': 'slow', 'model_version': 'v1'}
{'predictions': [{'score': 1.0}], 'model_name': 'slow', 'model_version': 'v1'}
{'predictions': [{'score': 1.0}], 'model_name': 'slow', 'model_version': 'v1'}

  *** version changed: v1 → v2 ***

{'predictions': [{'score': 1.0}], 'model_name': 'slow', 'model_version': 'v2'}
{'predictions': [{'score': 1.0}], 'model_name': 'slow', 'model_version': 'v2'}
```

The server logs confirm the load ran concurrently with serving — `LOAD START` and `LOAD COMPLETE` bracket 12 seconds of uninterrupted predict activity.

## No Regressions

The full test suite (62 tests) passes. `_load_lock` remains in place, so two simultaneous `load_model()` calls cannot race on the slot assignment.
