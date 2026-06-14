from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import InferenceService, app
from scenarios import ScenarioRepository


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
