from __future__ import annotations

import copy
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = object  # type: ignore
    JSONResponse = None  # type: ignore


logger = logging.getLogger("ml_serving")
logging.basicConfig(level=logging.INFO)


class ModelLoadError(RuntimeError):
    pass


class PredictionError(RuntimeError):
    pass


class ModelProtocol(Protocol):
    def predict(self, features: Sequence[Dict[str, Any]]) -> Sequence[Any]:
        ...


class ModelRepositoryClient(Protocol):
    def load(self, model_name: str, version: Optional[str] = None) -> ModelProtocol:
        ...


@dataclass(frozen=True)
class PredictionRequest:
    records: List[Dict[str, Any]]
    request_id: Optional[str] = None


@dataclass(frozen=True)
class PredictionResponse:
    predictions: List[Any]
    model_name: str
    model_version: Optional[str]
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "predictions": self.predictions,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        return payload


@dataclass
class Metrics:
    counters: Dict[str, int] = field(default_factory=dict)

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


class InferenceService:
    def __init__(
        self,
        model_repository: ModelRepositoryClient,
        model_name: str,
        model_version: Optional[str] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self._repository = model_repository
        self._model_name = model_name
        self._model_version = model_version
        self._metrics = metrics or Metrics()
        self._lock = threading.RLock()
        self._current_model: Optional[ModelProtocol] = None
        self._previous_model: Optional[ModelProtocol] = None

    def load_model(self) -> None:
        with self._lock:
            self._metrics.incr("model_load_attempts")
            logger.info("loading model name=%s version=%s", self._model_name, self._model_version)
            new_model = self._repository.load(self._model_name, self._model_version)
            if self._current_model is not None:
                self._previous_model = self._current_model
            self._current_model = new_model
            self._metrics.incr("model_load_success")

    def rollback(self) -> bool:
        with self._lock:
            if self._previous_model is None:
                self._metrics.incr("rollback_noop")
                return False
            self._current_model, self._previous_model = self._previous_model, self._current_model
            self._metrics.incr("rollback_success")
            logger.warning("rolled back to previous model name=%s version=%s", self._model_name, self._model_version)
            return True

    def validate_request(self, payload: Any) -> PredictionRequest:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("field 'records' must be a non-empty list")
        validated: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"records[{idx}] must be an object")
            validated.append(copy.deepcopy(record))
        request_id = payload.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("field 'request_id' must be a string when provided")
        return PredictionRequest(records=validated, request_id=request_id)

    def validate_response(self, predictions: Any, expected_len: int) -> List[Any]:
        if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes, bytearray)):
            raise PredictionError("model returned an invalid prediction payload")
        result = list(predictions)
        if len(result) != expected_len:
            raise PredictionError("prediction count does not match input count")
        return result

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        with self._lock:
            model = self._current_model
        if model is None:
            self._metrics.incr("predict_unavailable")
            raise PredictionError("model is not loaded")

        self._metrics.incr("predict_attempts")
        logger.info("predict request_id=%s records=%d", request.request_id, len(request.records))
        try:
            raw_predictions = model.predict(request.records)
            predictions = self.validate_response(raw_predictions, len(request.records))
            self._metrics.incr("predict_success")
            return PredictionResponse(
                predictions=predictions,
                model_name=self._model_name,
                model_version=self._model_version,
                request_id=request.request_id,
            )
        except PredictionError:
            self._metrics.incr("predict_validation_error")
            raise
        except Exception as exc:
            self._metrics.incr("predict_failure")
            logger.exception("prediction failed request_id=%s", request.request_id)
            raise PredictionError("prediction failed") from exc

    @property
    def metrics(self) -> Metrics:
        return self._metrics


class DummyRepository:
    def load(self, model_name: str, version: Optional[str] = None) -> ModelProtocol:
        class EchoModel:
            def predict(self, features: Sequence[Dict[str, Any]]) -> Sequence[Any]:
                return [{"score": len(item)} for item in features]

        return EchoModel()


service = InferenceService(DummyRepository(), model_name="retail_optimizer", model_version=None)
service.load_model()

if FastAPI is not None:
    app = FastAPI(title="ML Serving Service", version="0.1.0")

    @app.post("/predict")
    async def predict(request: Request):
        try:
            payload = await request.json()
            parsed = service.validate_request(payload)
            response = service.predict(parsed)
            return JSONResponse(content=response.to_dict())
        except ValueError as exc:
            logger.info("bad request: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except PredictionError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/admin/rollback")
    async def rollback():
        if not service.rollback():
            raise HTTPException(status_code=409, detail="rollback unavailable")
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok", "metrics": service.metrics.counters}
else:
    app = None


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    request = service.validate_request(payload)
    return service.predict(request).to_dict()


def load_model() -> None:
    service.load_model()


def rollback_model() -> bool:
    return service.rollback()


def get_metrics() -> Dict[str, int]:
    return dict(service.metrics.counters)


__all__ = [
    "InferenceService",
    "PredictionError",
    "PredictionRequest",
    "PredictionResponse",
    "ModelLoadError",
    "Metrics",
    "app",
    "predict",
    "load_model",
    "rollback_model",
    "get_metrics",
]
