"""Demo shim: SlowModel with a versioned reload endpoint.

Boot with: uvicorn demos.rollover.server_hot_swap:app

Adds POST /demo/load?version=<v> so the rollover harness can trigger
a live model swap while requests are in flight.
"""

import time

import app as app_module
from fastapi.concurrency import run_in_threadpool
from scenarios import ScenarioRepository


class SlowLoadRepository(ScenarioRepository):
    """Repository that sleeps on load to simulate a slow model download."""
    def load(self, model_name, version=None):
        time.sleep(12)
        return super().load(model_name, version)


# Initial load uses the fast repo so the server starts immediately.
svc = app_module.InferenceService(
    ScenarioRepository(), model_name="slow", model_version="v1"
)
svc.load_model()

# Swap to the slow repo so subsequent /demo/load calls take 12s.
svc._repository = SlowLoadRepository()
app_module.service = svc

_fastapi_app = app_module.app


@_fastapi_app.post("/demo/load")
async def demo_load(version: str):
    svc._model_version = version
    await run_in_threadpool(svc.load_model)
    return {"loaded": version}


app = _fastapi_app
