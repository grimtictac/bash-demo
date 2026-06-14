"""Validation demo server.

Extends ScenarioRepository with two additional broken models:
  - "nan"   returns {"score": float("nan")} — passes shape check, fails _check_finite
  - "none"  returns [None, ...] — passes shape check, fails None check

Boot with: uvicorn demo_validation:app

Use demo_validation_harness.py to drive it, or swap models manually:

    curl -X POST "http://localhost:8000/demo/load?model=nan"
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
         -d '{"records": [{"a": 1}]}'
"""

import app as app_module
from fastapi.concurrency import run_in_threadpool
from scenarios import ScenarioRepository

svc = app_module.InferenceService(
    ScenarioRepository(), model_name="healthy"
)
svc.load_model()
app_module.service = svc

_app = app_module.app


@_app.post("/demo/load")
async def demo_load(model: str):
    svc._model_name = model
    await run_in_threadpool(svc.load_model)
    return {"loaded": model}


app = _app
