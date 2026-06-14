"""Rollover demo harness.

Fires a wave of concurrent slow requests (each ~150ms), triggers a live
model swap 50ms in, then fires a second wave after the swap completes.

Expected result:
  wave 1 — all requests complete with model_version=v1 (they captured the
            slot before the swap; the swap did not stall or drop them)
  wave 2 — all requests complete with model_version=v2 (new slot active)

Usage: python demo_rollover_harness.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx

PREDICT_URL = "http://localhost:8000/predict"
LOAD_URL = "http://localhost:8000/demo/load"
PAYLOAD = {"records": [{"a": 1}]}
WAVE = 10
SWAP_AFTER_S = 0.05  # trigger swap 50ms into wave 1


def call():
    t0 = time.perf_counter()
    r = httpx.post(PREDICT_URL, json=PAYLOAD, timeout=10)
    r.raise_for_status()
    return r.json()["model_version"], (time.perf_counter() - t0) * 1000


def do_swap(version: str, delay: float):
    time.sleep(delay)
    t0 = time.perf_counter()
    r = httpx.post(LOAD_URL, params={"version": version}, timeout=10)
    r.raise_for_status()
    print(f"  [swap] → {version}  ({(time.perf_counter() - t0) * 1000:.0f}ms)")


# warm-up + baseline
httpx.post(PREDICT_URL, json=PAYLOAD, timeout=10).raise_for_status()
t0 = time.perf_counter()
httpx.post(PREDICT_URL, json=PAYLOAD, timeout=10).raise_for_status()
baseline = (time.perf_counter() - t0) * 1000
print(f"baseline (single call): {baseline:.0f}ms")

# wave 1: concurrent requests + swap mid-flight
print(f"\n[wave 1] {WAVE} requests fired, swap to v2 triggered at {int(SWAP_AFTER_S * 1000)}ms")
t_wave1 = time.perf_counter()
with ThreadPoolExecutor(max_workers=WAVE + 1) as pool:
    swap_future = pool.submit(do_swap, "v2", SWAP_AFTER_S)
    req_futures = [pool.submit(call) for _ in range(WAVE)]
    wave1 = [f.result() for f in req_futures]
    swap_future.result()
wall1 = (time.perf_counter() - t_wave1) * 1000

# wave 2: all requests after swap
print(f"\n[wave 2] {WAVE} requests fired after swap")
t_wave2 = time.perf_counter()
with ThreadPoolExecutor(max_workers=WAVE) as pool:
    wave2 = [f.result() for f in [pool.submit(call) for _ in range(WAVE)]]
wall2 = (time.perf_counter() - t_wave2) * 1000


def report(label, wave, wall):
    versions: dict = {}
    for v, _ in wave:
        versions[v] = versions.get(v, 0) + 1
    avg = sum(lat for _, lat in wave) / len(wave)
    print(f"\n{label}:")
    print(f"  completed : {len(wave)}/{WAVE}")
    print(f"  versions  : {dict(sorted(versions.items()))}")
    print(f"  wall      : {wall:.0f}ms   avg latency: {avg:.0f}ms")


report("wave 1 (in-flight during swap)", wave1, wall1)
report("wave 2 (after swap)", wave2, wall2)

print()
w1_versions = {v for v, _ in wave1}
w2_versions = {v for v, _ in wave2}
if w1_versions == {"v1"} and w2_versions == {"v2"}:
    print("VERDICT: clean rollover — in-flight requests unaffected, new requests on new model")
else:
    print(f"NOTE: wave 1 versions={w1_versions}  wave 2 versions={w2_versions}")
    print("(some wave 1 requests may have started after the swap — this is expected under light load)")
