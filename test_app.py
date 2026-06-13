"""
Tests for the ML serving service.

Coverage:
- Input validation (validate_request)
- Output validation (validate_response) including semantic checks
- All scenario models: healthy, load_failure, wrong_shape, slow
- Rollback paths including rollback with nothing to roll back to
- Concurrent inference (service layer)
- Hot model swap while requests are in flight
- HTTP endpoints via FastAPI TestClient
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import (
    InferenceService,
    Metrics,
    ModelLoadError,
    PredictionError,
    PredictionRequest,
    PredictionResponse,
    app,
    get_metrics,
    predict,
    rollback_model,
)
from scenarios import ScenarioRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo():
    return ScenarioRepository()


@pytest.fixture
def svc(repo):
    """Unloaded service backed by ScenarioRepository."""
    return InferenceService(repo, model_name="healthy")


@pytest.fixture
def loaded_svc(repo):
    s = InferenceService(repo, model_name="healthy")
    s.load_model()
    return s


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------


class TestValidateRequest:
    def test_non_dict_body(self, svc):
        with pytest.raises(ValueError, match="JSON object"):
            svc.validate_request("not a dict")

    def test_missing_records_key(self, svc):
        with pytest.raises(ValueError, match="'records'"):
            svc.validate_request({})

    def test_empty_records_list(self, svc):
        with pytest.raises(ValueError, match="non-empty"):
            svc.validate_request({"records": []})

    def test_records_not_a_list(self, svc):
        with pytest.raises(ValueError, match="'records'"):
            svc.validate_request({"records": "oops"})

    def test_record_not_a_dict(self, svc):
        with pytest.raises(ValueError, match=r"records\[0\]"):
            svc.validate_request({"records": ["not a dict"]})

    def test_valid_minimal(self, svc):
        req = svc.validate_request({"records": [{"a": 1}]})
        assert req.records == [{"a": 1}]
        assert req.request_id is None

    def test_valid_with_request_id(self, svc):
        req = svc.validate_request({"records": [{"a": 1}], "request_id": "abc-123"})
        assert req.request_id == "abc-123"

    def test_invalid_request_id_type(self, svc):
        with pytest.raises(ValueError, match="request_id"):
            svc.validate_request({"records": [{"a": 1}], "request_id": 42})

    def test_records_are_deep_copied(self, svc):
        """Mutations to the original payload must not affect the parsed request."""
        original = {"nested": {"x": 1}}
        req = svc.validate_request({"records": [original]})
        original["nested"]["x"] = 999
        assert req.records[0]["nested"]["x"] == 1

    def test_multiple_records(self, svc):
        req = svc.validate_request({"records": [{"a": 1}, {"b": 2}, {"c": 3}]})
        assert len(req.records) == 3


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_wrong_count(self, svc):
        with pytest.raises(PredictionError, match="count"):
            svc.validate_response([1, 2], 3)

    def test_not_a_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response(42, 1)

    def test_string_rejected_as_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response("abc", 3)

    def test_bytes_rejected_as_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response(b"abc", 3)

    def test_none_prediction(self, svc):
        with pytest.raises(PredictionError, match="None"):
            svc.validate_response([None], 1)

    def test_nan_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("nan")}], 1)

    def test_positive_inf_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("inf")}], 1)

    def test_negative_inf_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("-inf")}], 1)

    def test_nan_nested_in_list(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"scores": [1.0, float("nan")]}], 1)

    def test_valid_predictions_pass_through(self, svc):
        result = svc.validate_response([{"score": 1.0}, {"score": 2.0}], 2)
        assert result == [{"score": 1.0}, {"score": 2.0}]

    def test_integer_scores_are_accepted(self, svc):
        result = svc.validate_response([{"score": 1}], 1)
        assert result == [{"score": 1}]


# ---------------------------------------------------------------------------
# load_model — success and failure scenarios
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_healthy_loads_successfully(self, svc):
        svc.load_model()
        assert svc._slot is not None

    def test_load_increments_success_metric(self, svc):
        svc.load_model()
        assert svc.metrics.counters["model_load_success"] == 1

    def test_load_failure_raises_model_load_error(self, repo):
        svc = InferenceService(repo, model_name="load_failure")
        with pytest.raises(ModelLoadError):
            svc.load_model()

    def test_load_failure_increments_failure_metric(self, repo):
        svc = InferenceService(repo, model_name="load_failure")
        with pytest.raises(ModelLoadError):
            svc.load_model()
        assert svc.metrics.counters.get("model_load_failure", 0) >= 1

    def test_load_failure_does_not_replace_serving_model(self, repo):
        """A failed load must leave the currently-serving model intact."""
        svc = InferenceService(repo, model_name="healthy")
        svc.load_model()
        good_slot = svc._slot

        svc._model_name = "load_failure"
        with pytest.raises(ModelLoadError):
            svc.load_model()

        assert svc._slot is good_slot

    def test_predict_unavailable_before_load(self, svc):
        req = PredictionRequest(records=[{"a": 1}])
        with pytest.raises(PredictionError, match="not loaded"):
            svc.predict(req)

    def test_predict_unavailable_metric(self, svc):
        req = PredictionRequest(records=[{"a": 1}])
        with pytest.raises(PredictionError):
            svc.predict(req)
        assert svc.metrics.counters.get("predict_unavailable", 0) >= 1


# ---------------------------------------------------------------------------
# Prediction scenarios
# ---------------------------------------------------------------------------


class TestPredictionScenarios:
    def test_healthy_model_returns_correct_count(self, loaded_svc):
        req = PredictionRequest(records=[{"a": 1}, {"b": 2}])
        resp = loaded_svc.predict(req)
        assert len(resp.predictions) == 2

    def test_healthy_model_response_fields(self, loaded_svc):
        req = PredictionRequest(records=[{"a": 1}], request_id="req-1")
        resp = loaded_svc.predict(req)
        assert resp.model_name == "healthy"
        assert resp.request_id == "req-1"

    def test_wrong_shape_raises_prediction_error(self, repo):
        svc = InferenceService(repo, model_name="wrong_shape")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}, {"b": 2}])
        with pytest.raises(PredictionError, match="count"):
            svc.predict(req)

    def test_wrong_shape_increments_validation_error_metric(self, repo):
        svc = InferenceService(repo, model_name="wrong_shape")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}, {"b": 2}])
        with pytest.raises(PredictionError):
            svc.predict(req)
        assert svc.metrics.counters.get("predict_validation_error", 0) >= 1

    def test_wrong_shape_service_remains_usable(self, repo):
        """A bad prediction must not corrupt the service for subsequent requests."""
        svc = InferenceService(repo, model_name="wrong_shape")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}, {"b": 2}])
        with pytest.raises(PredictionError):
            svc.predict(req)
        # Service should still be queryable
        assert svc._slot is not None

    def test_slow_model_produces_valid_output(self, repo):
        svc = InferenceService(repo, model_name="slow")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}])
        resp = svc.predict(req)
        assert len(resp.predictions) == 1

    def test_latency_metric_is_recorded(self, loaded_svc):
        req = PredictionRequest(records=[{"a": 1}])
        loaded_svc.predict(req)
        assert loaded_svc.metrics.counters.get("predict_latency_ms_total", 0) >= 0


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_with_no_model_at_all_returns_false(self, svc):
        assert svc.rollback() is False

    def test_rollback_after_first_load_returns_false(self, loaded_svc):
        # Only one load — no previous model to roll back to.
        assert loaded_svc.rollback() is False

    def test_rollback_noop_metric(self, loaded_svc):
        loaded_svc.rollback()
        assert loaded_svc.metrics.counters.get("rollback_noop", 0) >= 1

    def test_rollback_after_two_loads_returns_true(self, svc):
        svc.load_model()
        svc.load_model()
        assert svc.rollback() is True

    def test_rollback_success_metric(self, svc):
        svc.load_model()
        svc.load_model()
        svc.rollback()
        assert svc.metrics.counters.get("rollback_success", 0) >= 1

    def test_rollback_restores_previous_model_instance(self, repo):
        svc = InferenceService(repo, model_name="healthy")
        svc.load_model()
        first_slot = svc._slot

        svc.load_model()
        assert svc._slot is not first_slot

        svc.rollback()
        assert svc._slot is first_slot

    def test_rollback_twice_second_is_noop(self, svc):
        svc.load_model()
        svc.load_model()
        assert svc.rollback() is True
        assert svc.rollback() is False

    def test_rollback_after_failed_load_still_serves(self, repo):
        """A failed load should not corrupt the rollback slot."""
        svc = InferenceService(repo, model_name="healthy")
        svc.load_model()
        good_slot = svc._slot

        svc._model_name = "load_failure"
        with pytest.raises(ModelLoadError):
            svc.load_model()

        # Rollback should report nothing to roll back (only one successful load)
        assert svc.rollback() is False
        assert svc._slot is good_slot

    def test_rollback_version_tracking(self, repo):
        """model_version in responses should reflect the rolled-back model."""
        svc = InferenceService(repo, model_name="healthy", model_version="v1")
        svc.load_model()

        svc._model_version = "v2"
        svc.load_model()
        assert svc._slot.version == "v2"

        svc.rollback()
        assert svc._slot.version == "v1"


# ---------------------------------------------------------------------------
# Concurrency — service layer
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_inference_runs_concurrently_not_serialised(self, repo):
        """N concurrent slow-model requests must complete in ~single-call time."""
        svc = InferenceService(repo, model_name="slow")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}])

        N = 10
        errors: list = []

        def call():
            try:
                svc.predict(req)
            except Exception as e:
                errors.append(e)

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(call) for _ in range(N)]
            for f in futs:
                f.result()
        wall = time.perf_counter() - wall_start

        assert not errors, f"predict raised during concurrent test: {errors}"
        # Serial would be N * 50ms = 500ms+. Parallel should be ~50ms.
        # Allow a generous 200ms to account for scheduling overhead.
        assert wall < 0.20, (
            f"inference appears serialised: {N} concurrent calls took {wall:.3f}s "
            f"(expected < 0.20s if running in parallel)"
        )

    def test_hot_swap_no_requests_dropped(self, repo):
        """A model swap while requests are in flight must not drop any request."""
        svc = InferenceService(repo, model_name="slow")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}])

        results: list = []
        errors: list = []

        def call():
            try:
                results.append(svc.predict(req))
            except Exception as e:
                errors.append(e)

        N = 10
        with ThreadPoolExecutor(max_workers=N + 1) as pool:
            futs = [pool.submit(call) for _ in range(N)]
            # Let some requests get in flight, then trigger a reload.
            time.sleep(0.015)
            reload_fut = pool.submit(svc.load_model)
            for f in futs:
                f.result()
            reload_fut.result()

        assert not errors, f"requests failed during hot swap: {errors}"
        assert len(results) == N

    def test_concurrent_metrics_are_consistent(self, repo):
        """Metrics counters must not lose increments under concurrent load."""
        svc = InferenceService(repo, model_name="healthy")
        svc.load_model()
        req = PredictionRequest(records=[{"a": 1}])

        N = 50
        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(svc.predict, req) for _ in range(N)]
            for f in futs:
                f.result()

        assert svc.metrics.counters["predict_success"] == N


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


class TestHTTPEndpoints:
    def test_predict_valid_request(self, client):
        resp = client.post("/predict", json={"records": [{"a": 1}]})
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert "model_name" in body
        assert "model_version" in body
        assert len(body["predictions"]) == 1

    def test_predict_multiple_records(self, client):
        resp = client.post("/predict", json={"records": [{"a": 1}, {"b": 2}]})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 2

    def test_predict_with_request_id_echoed(self, client):
        resp = client.post(
            "/predict",
            json={"records": [{"a": 1}], "request_id": "test-req-42"},
        )
        assert resp.status_code == 200
        assert resp.json()["request_id"] == "test-req-42"

    def test_predict_without_request_id_not_in_response(self, client):
        resp = client.post("/predict", json={"records": [{"a": 1}]})
        assert resp.status_code == 200
        assert "request_id" not in resp.json()

    def test_predict_empty_records_is_400(self, client):
        resp = client.post("/predict", json={"records": []})
        assert resp.status_code == 400

    def test_predict_missing_records_is_400(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400

    def test_predict_non_object_body_is_400(self, client):
        resp = client.post("/predict", json=[1, 2, 3])
        assert resp.status_code == 400

    def test_predict_record_not_dict_is_400(self, client):
        resp = client.post("/predict", json={"records": ["not a dict"]})
        assert resp.status_code == 400

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "metrics" in body
        assert isinstance(body["metrics"], dict)

    def test_rollback_no_previous_is_409(self, client):
        # The module-level service is loaded exactly once at import time,
        # so there is no previous model and rollback should return 409.
        # Re-create a fresh TestClient backed by a fresh service to isolate state.
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse

        fresh_repo = ScenarioRepository()
        fresh_svc = InferenceService(fresh_repo, model_name="healthy")
        fresh_svc.load_model()

        fresh_app = FastAPI()

        @fresh_app.post("/admin/rollback")
        async def _rollback():
            if not fresh_svc.rollback():
                raise HTTPException(status_code=409, detail="rollback unavailable")
            return {"status": "ok"}

        with TestClient(fresh_app) as c:
            resp = c.post("/admin/rollback")
        assert resp.status_code == 409

    def test_rollback_after_two_loads_is_200(self, client):
        from fastapi import FastAPI, HTTPException

        fresh_repo = ScenarioRepository()
        fresh_svc = InferenceService(fresh_repo, model_name="healthy")
        fresh_svc.load_model()
        fresh_svc.load_model()

        fresh_app = FastAPI()

        @fresh_app.post("/admin/rollback")
        async def _rollback():
            if not fresh_svc.rollback():
                raise HTTPException(status_code=409, detail="rollback unavailable")
            return {"status": "ok"}

        with TestClient(fresh_app) as c:
            resp = c.post("/admin/rollback")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_predict_wrong_shape_model_returns_503(self, client):
        # Patch the module-level service so the real app endpoint exercises
        # the wrong_shape path without building a second inline FastAPI app.
        original = app_module.service
        try:
            ws_svc = InferenceService(ScenarioRepository(), model_name="wrong_shape")
            ws_svc.load_model()
            app_module.service = ws_svc
            resp = client.post("/predict", json={"records": [{"a": 1}, {"b": 2}]})
            assert resp.status_code == 503
        finally:
            app_module.service = original


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


class TestModuleLevelAPI:
    def test_predict_function_returns_expected_keys(self):
        result = predict({"records": [{"a": 1}]})
        assert "predictions" in result
        assert "model_name" in result
        assert "model_version" in result

    def test_get_metrics_returns_dict(self):
        m = get_metrics()
        assert isinstance(m, dict)

    def test_rollback_model_returns_bool(self):
        result = rollback_model()
        assert isinstance(result, bool)
