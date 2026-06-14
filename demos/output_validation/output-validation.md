# Issue: Semantic Output Validation

## The Problem

The original `validate_response` only checked that the model returned the right number of predictions — one per input record. It said nothing about the content of those predictions.

A model that loads and runs cleanly can still produce output that is wrong in ways shape-checking cannot catch:

- A prediction of `None` — structurally valid, meaningless to a caller
- A score of `NaN` or `Inf` — a common silent failure mode in numeric models, caused by division by zero, overflow, or an untrained model returning uninitialised weights

Both would have passed the original check and been returned to the caller as a `200 OK`.

```python
# before — shape only
def validate_response(self, predictions, expected_len):
    result = list(predictions)
    if len(result) != expected_len:
        raise PredictionError("prediction count does not match input count")
    return result
```

## The Fix

`validate_response` now rejects both:

```python
for i, pred in enumerate(result):
    if pred is None:
        raise PredictionError(f"predictions[{i}] is None")
    _check_finite(pred, f"predictions[{i}]")
```

`_check_finite` walks the prediction recursively — handling dicts, lists, and nested structures — and raises on any `float` value that is not finite:

```python
def _check_finite(value, path):
    if isinstance(value, float) and not math.isfinite(value):
        raise PredictionError(f"non-finite value at {path}")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_finite(v, f"{path}[{i}]")
```

Any bad prediction raises `PredictionError`, which the HTTP handler returns as a `503` with the exact path of the offending value. The `predict_validation_error` metric is incremented so on-call can alert on it.

## Running the Demo

```bash
source .venv/bin/activate
```

```
python -m demos.output_validation.validation
```

Expected output (log lines omitted for clarity):

```
=== NaN score ===

  ❌ BEFORE  →  200 OK   predictions=[{'score': nan}]   ← bad data silently returned
  ✅ AFTER   →  503      non-finite value at predictions[0].score (got nan)


=== None prediction ===

  ❌ BEFORE  →  200 OK   predictions=[None]   ← bad data silently returned
  ✅ AFTER   →  503      predictions[0] is None
```

## What it catches

| Scenario | Error |
|---|---|
| Model returns one fewer prediction | `503` — count mismatch |
| Model returns `None` for a prediction | `503` — `predictions[i] is None` |
| Model returns `{"score": float("nan")}` | `503` — non-finite at `predictions[0].score` |
| Model returns `{"score": float("inf")}` | `503` — non-finite at `predictions[0].score` |
| Nested bad value `{"details": {"score": float("nan")}}` | `503` — non-finite at `predictions[0].details.score` |
