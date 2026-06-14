"""Demo shim: SlowModel with the original serialised (blocking) handler.

Deliberately calls service.predict() synchronously in the async handler,
blocking the event loop so all requests are processed one at a time.

Use this to demonstrate the serialisation problem without switching branches.

    uvicorn demo_slow_serialised:app   ← serialised (broken)
    uvicorn demo_slow:app              ← parallel (fixed)
"""

import app as app_module
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from scenarios import ScenarioRepository

svc = app_module.InferenceService(ScenarioRepository(), model_name="slow")
svc.load_model()

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
        parsed = svc.validate_request(payload)
        response = svc.predict(parsed)          # blocking — no run_in_threadpool
        return JSONResponse(content=response.to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except app_module.PredictionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
