from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import InferenceService, PredictionRequest
from scenarios import ScenarioRepository


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
        # Serial would be N * 150ms = 1500ms+. Parallel should be ~150ms.
        # Allow a generous 500ms to account for scheduling overhead.
        assert wall < 0.50, (
            f"inference appears serialised: {N} concurrent calls took {wall:.3f}s "
            f"(expected < 0.50s if running in parallel)"
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
