"""Demo: output validation catches NaN and None that the original service let through.

    python -m demos.output_validation.validation
"""

from app import InferenceService, ModelSlot, PredictionError, PredictionRequest
from scenarios import ScenarioRepository


class NanModel:
    def predict(self, features):
        return [{"score": float("nan")} for _ in features]


class NoneModel:
    def predict(self, features):
        return [None for _ in features]


class UnvalidatedService(InferenceService):
    """Reproduces the original shape-only validate_response."""

    def validate_response(self, predictions, expected_len):
        result = list(predictions)
        if len(result) != expected_len:
            raise PredictionError("prediction count does not match input count")
        return result


request = PredictionRequest(records=[{"a": 1}])

for label, model in [("NaN score", NanModel()), ("None prediction", NoneModel())]:
    print(f"\n=== {label} ===")

    svc = UnvalidatedService(ScenarioRepository(), model_name="healthy")
    svc.load_model()
    svc._slot = ModelSlot(model, "broken")
    result = svc.predict(request)
    print()
    print(f"  ❌ BEFORE  →  200 OK   predictions={result.predictions}   ← bad data silently returned")

    svc2 = InferenceService(ScenarioRepository(), model_name="healthy")
    svc2.load_model()
    svc2._slot = ModelSlot(model, "fixed")
    try:
        svc2.predict(request)
        print("  ✅ AFTER   →  200 OK  (unexpected)")
    except PredictionError as exc:
        print(f"  ✅ AFTER   →  503      {exc}")
    print()
