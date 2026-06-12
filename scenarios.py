"""
Operational scenarios for the ML serving refactor.

These model stubs and the repository below stand in for the Snowflake model
repository in production. Use them to drive your tests and to exercise the
failure modes described in the README.

The service must serve the healthy model and fail safely on the broken ones:
  - "healthy"       well formed output, one prediction per record
  - "load_failure"  raises on load; the model that is serving must survive
  - "wrong_shape"   loads fine, returns the wrong number of predictions
  - "slow"          healthy output, but about 50ms per call; use it under
                    concurrent load to see whether the service serialises
                    inference or runs it concurrently

Other failure modes exist that are not represented here. Thinking about what
else a model can do once it has loaded is part of the exercise.

Nothing here should need to change. Wire it into your service and tests.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence


class HealthyModel:
    """One well formed prediction per input record."""

    def predict(self, features: Sequence[Dict[str, Any]]) -> Sequence[Any]:
        return [{"score": float(len(item))} for item in features]


class WrongShapeModel:
    """Returns one fewer prediction than it was given."""

    def predict(self, features: Sequence[Dict[str, Any]]) -> Sequence[Any]:
        out = [{"score": float(len(item))} for item in features]
        return out[:-1] if out else out


class SlowModel:
    """Healthy output, but each call takes about 50ms.

    Put this under concurrent load. If the service serialises inference,
    behind a lock or behind the event loop, latency will scale with the
    number of callers. If it does not, latency stays close to a single call.
    """

    def predict(self, features: Sequence[Dict[str, Any]]) -> Sequence[Any]:
        time.sleep(0.05)
        return [{"score": float(len(item))} for item in features]


class ScenarioRepository:
    """
    Stand in for the production model repository.

        load("healthy")       -> HealthyModel
        load("wrong_shape")   -> WrongShapeModel
        load("slow")          -> SlowModel
        load("load_failure")  -> raises

    The version argument is accepted and ignored, matching the production
    repository interface.
    """

    def load(self, model_name: str, version: Optional[str] = None):
        if model_name == "healthy":
            return HealthyModel()
        if model_name == "wrong_shape":
            return WrongShapeModel()
        if model_name == "slow":
            return SlowModel()
        if model_name == "load_failure":
            raise RuntimeError(f"failed to load model {model_name!r} from repository")
        raise RuntimeError(f"unknown model {model_name!r}")
