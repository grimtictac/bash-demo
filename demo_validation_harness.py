"""Validation demo harness.

Loads each model in turn and fires a predict request, showing that broken
model output is rejected before it reaches the caller.

Usage: python demo_validation_harness.py
       (server: uvicorn demo_validation:app)
"""

import httpx

PREDICT_URL = "http://localhost:8000/predict"
LOAD_URL = "http://localhost:8000/demo/load"
PAYLOAD = {"records": [{"a": 1}]}


def load(model: str) -> None:
    httpx.post(LOAD_URL, params={"model": model}, timeout=10).raise_for_status()
    print(f"\n--- model: {model} ---")


def predict() -> None:
    r = httpx.post(PREDICT_URL, json=PAYLOAD, timeout=10)
    print(f"  {r.status_code}  {r.json()}")


load("healthy")
predict()

load("wrong_shape")
predict()

load("nan")
predict()

load("none")
predict()

load("healthy")
predict()
