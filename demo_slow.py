"""Demo shim: boot the original app.py against scenarios.SlowModel.

Imports app.py as-is, replaces the module-level service with one wired to
the SlowModel from scenarios.py, and re-exports the FastAPI app so uvicorn
can serve it via `uvicorn demo_slow:app`.

Used only for the concurrency demo; not part of the deliverable.
"""

import app
from scenarios import ScenarioRepository

app.service = app.InferenceService(ScenarioRepository(), model_name="slow")
app.service.load_model()

# uvicorn looks for `app` at module level
app = app.app
