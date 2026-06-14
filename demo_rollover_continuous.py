"""Continuous rollover demo.

N worker threads fire predict requests back-to-back for RUN_SECONDS.
A swap to v2 is triggered at SWAP_AFTER seconds.

Shows the natural transition from v1 to v2 in the result stream:
  - requests in-flight at swap time complete with v1
  - requests that start after the swap return v2
  - no requests are dropped or error

Usage: python demo_rollover_continuous.py
       (server: uvicorn demo_rollover:app)
"""

import threading
import time

import httpx

PREDICT_URL = "http://localhost:8000/predict"
LOAD_URL = "http://localhost:8000/demo/load"
PAYLOAD = {"records": [{"a": 1}]}

N_WORKERS = 5
RUN_SECONDS = 15
SWAP_AFTER = 5


results: list = []
results_lock = threading.Lock()
stop_event = threading.Event()


def worker():
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            r = httpx.post(PREDICT_URL, json=PAYLOAD, timeout=5)
            r.raise_for_status()
            version = r.json()["model_version"]
        except Exception:
            version = "ERROR"
        with results_lock:
            results.append((t0, time.perf_counter(), version))


def do_swap(delay: float, version: str) -> tuple:
    time.sleep(delay)
    t_sent = time.perf_counter()
    httpx.post(LOAD_URL, params={"version": version}, timeout=5).raise_for_status()
    t_done = time.perf_counter()
    return t_sent, t_done


# warm-up
httpx.post(PREDICT_URL, json=PAYLOAD, timeout=5).raise_for_status()

t_start = time.perf_counter()

workers = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
for w in workers:
    w.start()

swap_result: list = []
swap_thread = threading.Thread(
    target=lambda: swap_result.append(do_swap(SWAP_AFTER, "v2")), daemon=True
)
swap_thread.start()
swap_thread.join()
t_swap_sent, t_swap_done = swap_result[0]
t_swap_sent_ms = (t_swap_sent - t_start) * 1000
t_swap_done_ms = (t_swap_done - t_start) * 1000
print(f"[{t_swap_sent_ms:.0f}ms – {t_swap_done_ms:.0f}ms] swap → v2  (slot assigned somewhere in this window)")

time.sleep(RUN_SECONDS - SWAP_AFTER)
stop_event.set()
for w in workers:
    w.join()

# sort by completion time
results.sort(key=lambda r: r[1])

print(f"\n{'started':>8}   {'done':>8}   {'version':>8}   {'expected':>8}")
print("-" * 50)
swap_marked = False
for t0, t1, version in results:
    started_ms = (t0 - t_start) * 1000
    done_ms = (t1 - t_start) * 1000
    # classify: before swap window → must be v1; after swap window → must be v2
    # inside the window the slot assignment could have landed either side, so either is valid
    if t0 < t_swap_sent:
        expected = "v1"
        ok = "OK" if version == "v1" else "MISMATCH"
    elif t0 > t_swap_done:
        expected = "v2"
        ok = "OK" if version == "v2" else "MISMATCH"
    else:
        expected = "v1 or v2"
        ok = "OK"
    if not swap_marked and done_ms >= t_swap_sent_ms:
        print(f"{'':>8}   {'':>8}   {'─ swap ─':>8}")
        swap_marked = True
    print(f"{started_ms:>7.0f}ms   {done_ms:>7.0f}ms   {version:>8}   {ok:>8}")

print()
counts: dict = {}
for _, _, v in results:
    counts[v] = counts.get(v, 0) + 1
total = len(results)
errors = counts.get("ERROR", 0)
mismatches = sum(
    1 for t0, _, v in results
    if (t0 < t_swap_sent and v == "v2") or (t0 > t_swap_done and v == "v1")
)
print(f"total : {total}  v1: {counts.get('v1', 0)}  v2: {counts.get('v2', 0)}  errors: {errors}  mismatches: {mismatches}")
if errors == 0 and mismatches == 0:
    print("VERDICT: clean rollover — every request was served by the model active when it started")
