from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app as app_module
from app import InferenceService, PredictionError, PredictionRequest
from scenarios import ScenarioRepository


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
        fresh_svc = InferenceService(ScenarioRepository(), model_name="healthy")
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
        fresh_svc = InferenceService(ScenarioRepository(), model_name="healthy")
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
        original = app_module.service
        try:
            ws_svc = InferenceService(ScenarioRepository(), model_name="wrong_shape")
            ws_svc.load_model()
            app_module.service = ws_svc
            resp = client.post("/predict", json={"records": [{"a": 1}, {"b": 2}]})
            assert resp.status_code == 503
        finally:
            app_module.service = original


class TestModuleLevelAPI:
    def test_predict_function_returns_expected_keys(self):
        from app import predict
        result = predict({"records": [{"a": 1}]})
        assert "predictions" in result
        assert "model_name" in result
        assert "model_version" in result

    def test_get_metrics_returns_dict(self):
        from app import get_metrics
        m = get_metrics()
        assert isinstance(m, dict)

    def test_rollback_model_returns_bool(self):
        from app import rollback_model
        result = rollback_model()
        assert isinstance(result, bool)
