# Tests

## Running

```bash
python -m pytest tests/ -v
```

## Structure

| File | What it covers |
|---|---|
| `test_input_validation.py` | `validate_request` — malformed payloads, type errors, deep copy |
| `test_output_validation.py` | `validate_response` — count mismatch, `None`, `NaN`, `Inf` |
| `test_locking_and_rollback.py` | Load success/failure, rollback correctness, slot semantics |
| `test_concurrency.py` | Parallel inference, hot swap under load, thread-safe metrics |
| `test_http.py` | HTTP endpoints, prediction scenarios, module-level API |
