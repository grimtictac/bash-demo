# Issue: Model Swap Stalling In-Flight Requests

## The Problem

The original service held a single `_lock` for the full duration of `load_model()` — including the slow `repository.load()` call that fetches the model. Every `predict()` call had to acquire the same lock to safely read `_current_model`.

This meant a model swap serialised against all in-flight predictions. A load that took 12 seconds would cause every concurrent request to queue behind it — a 12-second outage on every hot swap.

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

`ModelSlot` (introduced in the locking-and-rollback change) makes any lock unnecessary in `predict()`. The model and version are read in a single atomic reference assignment — no lock is needed for a consistent view.

`load_model()` retains `_load_lock`, but only to serialise concurrent loads against each other. `predict()` never touches it.

```python
# after — load_model
with self._load_lock:                           # predict never acquires this
    new_model = self._repository.load(…)        # slow, but doesn't block predict
    self._slot = ModelSlot(new_model, version)  # atomic swap

# after — predict
slot = self._slot  # single read, no lock — never blocks on a load
```

During a load, in-flight predictions continue against the current slot. When the load completes, `self._slot` is updated atomically. Requests that captured the slot before the swap complete with the old model; requests that start after use the new one. No request is dropped or stalled.

## Running the Demos

```bash
source .venv/bin/activate
```

### Manual demo

Watch the version change in real time as you trigger a swap.

**Terminal 1:**
```bash
uvicorn demos.rollover.server_hot_swap:app
```

To stop the server:
```bash
pkill -f uvicorn
```

**Terminal 2:**
```bash
python demos/rollover/client_version_watch.py
```

**Terminal 3** (while the harness is running):
```bash
curl -X POST "http://localhost:8000/demo/load?version=v2"
```

The load takes 12 seconds. Responses continue uninterrupted on `v1` throughout, then cleanly transition to `v2`.

---

### Stall demo (broken behaviour)

`server_stall_on_swap.py` reproduces the original bug — a single lock held for the full load duration. Run it in place of `server:app` and trigger a swap; the client freezes for 12 seconds.

**Terminal 1:**
```bash
uvicorn demos.rollover.server_stall_on_swap:app
```

To stop the server:
```bash
pkill -f uvicorn
```

**Terminal 2:**
```bash
python demos/rollover/client_version_watch.py
```

**Terminal 3:**
```bash
curl -X POST "http://localhost:8000/demo/load?version=v2"
```

The client output pauses completely for the 12-second load duration, then resumes.
