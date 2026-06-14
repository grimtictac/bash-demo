from __future__ import annotations

import pytest

from app import InferenceService, ModelLoadError, PredictionError, PredictionRequest
from scenarios import ScenarioRepository


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


class TestRollback:
    def test_rollback_with_no_model_at_all_returns_false(self, svc):
        assert svc.rollback() is False

    def test_rollback_after_first_load_returns_false(self, loaded_svc):
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
