"""Demo shim: SlowModel with the fixed parallel handler.

Requests are dispatched to the thread pool so the event loop stays free
and all concurrent requests run simultaneously.

    uvicorn demo_slow_parallel:app     ← parallel (fixed)
    uvicorn demo_slow_serialised:app   ← serialised (broken)
"""

import app
from scenarios import ScenarioRepository

app.service = app.InferenceService(ScenarioRepository(), model_name="slow")
app.service.load_model()

# uvicorn looks for `app` at module level
app = app.app
