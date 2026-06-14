"""Concurrency demo harness.

Fires N concurrent POSTs against /predict and reports wall-clock total
vs sum of individual latencies.

If inference parallelises:  wall-clock ~= single call latency
If it serialises:           wall-clock ~= N * single call latency

Usage: python demo_harness.py [N] [URL]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/predict"

    payload = {"records": [{"a": 1}]}

    # Single warm-up call to establish a baseline before the concurrent run.
    httpx.post(url, json=payload, timeout=30).raise_for_status()
    t0 = time.perf_counter()
    httpx.post(url, json=payload, timeout=30).raise_for_status()
    baseline = time.perf_counter() - t0

    latencies: list[float] = []

    def call(_: int) -> None:
        t0 = time.perf_counter()
        r = httpx.post(url, json=payload, timeout=30)
        latencies.append(time.perf_counter() - t0)
        r.raise_for_status()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(call, range(n)))
    wall = time.perf_counter() - start

    ratio = wall / baseline if baseline > 0 else 1
    print(f"calls            : {n}")
    print(f"single call      : {baseline * 1000:>7.1f} ms  (baseline)")
    print(f"wall clock total : {wall * 1000:>7.1f} ms  ({ratio:.1f}x baseline)")
    print(f"sum of latencies : {sum(latencies) * 1000:>7.1f} ms")
    print(f"avg per call     : {sum(latencies) / n * 1000:>7.1f} ms")
    print(f"max single call  : {max(latencies) * 1000:>7.1f} ms")


if __name__ == "__main__":
    main()
