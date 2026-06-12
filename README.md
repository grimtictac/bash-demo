# Production ML Serving Refactor

## How to read this

You may use AI tools, and we expect that you will. That is how the team works, so use whatever you would use on the job. We are not only grading the volume or polish of what comes back. We are grading whether you understand it, can defend it, and would own it in production.

For that reason this take-home is not a pass or fail gate on its own. It is the starting point for the interview. In the session we will review your submission together, ask you to justify the choices in it, and push on how your design would change under different assumptions and requirements. Submit work you understand line by line, not work you have only skimmed.

## The task

You are given a small ML serving service in `app.py`. It loads a model from a model repository, validates inputs and outputs, serves predictions over HTTP, and supports rollback. It runs, but it is not ready to operate. Make it production ready without changing what it exposes to callers.

This is a mission-critical workload. Treat it as something you and your team will be on call for: a bad prediction reaching a caller, or the service going down during a model swap, is a real incident. Hold yourself to that bar.

## It runs under load, and models change while it runs

The service serves concurrent traffic, and models are reloaded without taking it down. Your refactor has to hold up under both at once.

- A hot model swap can happen while requests are in flight. No request should be dropped, stalled behind the swap, or answered by the wrong model. Be ready to explain why your swap is safe for requests already in progress.
- Inference must run concurrently, not one request at a time. `scenarios.py` includes a slow model that takes about 50ms per call. Under concurrent load the service should stay close to single-request latency, not multiply it by the number of callers. A design that serialises inference, whether behind a lock or behind the event loop, will not meet this.

## Time box

About 2 to 3 hours. If the scope is more than you can finish well, do the most important parts and say what you left.

## The contract you must not break

This is the public surface. Other callers and services depend on it. Keep all of it stable.

HTTP:
- `POST /predict` accepts `{"records": [ {...}, ... ], "request_id": "optional string"}` and returns `{"predictions": [...], "model_name": "...", "model_version": "... or null", "request_id": "... if it was supplied"}`.
- `POST /admin/rollback` and `GET /health` continue to exist.

Importable Python API, same names and signatures:
- `predict(payload: dict) -> dict`
- `load_model() -> None`
- `rollback_model() -> bool`
- `get_metrics() -> dict[str, int]`
- the `InferenceService` class and the exceptions exported in `__all__`

You may change anything behind that surface. If you think part of the contract is wrong, do not change it silently. Flag it in your notes and propose the fix.

## Scenarios you must handle

`scenarios.py` contains four model stubs and a stand in repository: a healthy model, one that fails to load, one that returns the wrong number of predictions, and a slow one for exercising concurrency. Serve the healthy one and fail safely and observably on the broken ones, without taking the service down and without returning bad predictions to a caller.

There are other failure modes we care about that are not in the file. Part of the exercise is thinking about what else can go wrong with a model that loads cleanly.

## Deliverables

1. The refactored service. Coherent, runnable, contract preserved.
2. Tests covering input validation, the scenarios above, and the rollback paths, including rollback when there is nothing to roll back to. Pytest is fine.
3. A test or small harness that drives the service under concurrent load and during a reload, and shows it holds up.
4. Short notes (`DECISIONS.md`, about a page). These are notes for our conversation, not a graded document. Be ready to talk to every point. Cover:
   - **Assumptions and open questions.** We have deliberately not handed you every requirement. State the assumptions you made, and list the questions you would ask the team before building this for real.
   - What you changed and why, what you deliberately left, the main tradeoffs, and what you would do next with more time.
   - How you would deploy this, roll it back, and monitor it, and what you would alert on. Notes only — no Terraform or pipelines. We go deep on this in the conversation.

## Out of scope

You do not need to deploy anything, write Terraform, or wire up real Snowflake or AWS. The model repository stays behind its interface. Deployment architecture, the team wide deployment standard, and the real data layer are covered in the design conversation, not here.

## What we look at

- That you can explain every choice in your own words.
- Validation of model output by meaning, not only shape. A bad prediction should never reach a caller.
- Loading and rollback that fail safe. A bad load or a bad model never replaces a good one that is serving.
- Behaviour under concurrency. Hot swaps that do not drop, stall, or mismatch requests, inference that runs concurrently rather than serialised, and a believable account of why the swap is safe for in-flight requests.
- Tests that prove the failure modes are handled and that the service holds up under load.
- Observability useful to an on call engineer, with no sensitive record data in the logs.
- The public contract left intact.
