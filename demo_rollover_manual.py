"""Manual rollover demo.

Fires predict requests in a continuous loop and logs the model version
returned. Trigger the swap yourself from another terminal:

    curl -X POST "http://localhost:8000/demo/load?version=v2"

Watch the version change in the output.

Usage: python demo_rollover_manual.py
       (server: uvicorn demo_rollover:app)
"""

import time

import httpx

PREDICT_URL = "http://localhost:8000/predict"
PAYLOAD = {"records": [{"a": 1}]}
PAUSE_BETWEEN_REQUESTS = 0.85  # seconds

current_version = None

print("firing requests — Ctrl+C to stop")
print("swap command:  curl -X POST 'http://localhost:8000/demo/load?version=v2'")
print()

while True:
    try:
        r = httpx.post(PREDICT_URL, json=PAYLOAD, timeout=5)
        r.raise_for_status()
        body = r.json()
        version = body["model_version"]

        if version != current_version:
            if current_version is not None:
                print(f"\n  *** version changed: {current_version} → {version} ***\n")
            current_version = version

        print(body)
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\nstopped")
        break
    except Exception as exc:
        print(f"ERROR: {exc}")
