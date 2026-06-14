"""Broken server shim — demonstrates the original locking bug.

Uses a single _lock held for the full duration of load_model(), including
the slow repository.load() call. predict() must also acquire _lock to read
_current_model, so any in-flight or incoming request blocks for the entire
12-second load.

Boot with: uvicorn demo_stall_on_swap:app

Use the same demo_rollover_manual.py harness and trigger a swap:

    curl -X POST "http://localhost:8000/demo/load?version=v2"

You will see the client freeze for ~12 seconds while the load holds the lock.
"""

import threading
import time
from typing import Any, Dict, Optional, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from scenarios import ScenarioRepository


class SlowLoadRepository(ScenarioRepository):
    def load(self, model_name, version=None):
        time.sleep(12)
        return super().load(model_name, version)


# --- original broken state ---
_lock = threading.Lock()
_current_model = None
_current_version = None
_repo = ScenarioRepository()   # fast for initial load
_slow_repo = SlowLoadRepository()


def _load(version: str, repo) -> None:
    global _current_model, _current_version
    with _lock:                              # held for the entire load
        _current_model = repo.load("slow", version)
        _current_version = version


# initial fast load
_load("v1", _repo)

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    def _predict():
        with _lock:                          # blocks if a load is in progress
            model = _current_model
            version = _current_version
        result = model.predict([{"a": 1}])
        return {"predictions": list(result), "model_name": "slow", "model_version": version}
    return JSONResponse(await run_in_threadpool(_predict))


@app.post("/demo/load")
async def demo_load(version: str):
    await run_in_threadpool(_load, version, _slow_repo)
    return {"loaded": version}


@app.get("/health")
async def health():
    return {"status": "ok"}
