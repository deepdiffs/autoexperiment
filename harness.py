"""
Experiment harness — fixed configuration and utilities.
The agent does NOT modify this file.

Roofline experiment for Mac Studio M3 Ultra running Qwen3.6-27B via llama.cpp.
Drives llama-bench, computes achieved memory bandwidth as a fraction of the
empirical peak (measured once via MLX in setup) and the spec peak (800 GB/s).

One-time setup:
    uv run harness.py
This measures peak memory bandwidth and caches it to peak_bw.json.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRIC_NAME = "bandwidth_utilization_pct"  # % of measured peak bandwidth (higher = closer to roofline)
METRIC_GOAL = "maximize"
TIME_BUDGET = 60                            # seconds per experiment run

# M3 Ultra theoretical peak (819.2 GB/s, rounded). Used as the spec ceiling.
PEAK_BW_SPEC_GBS = 800.0

# The model under test. Must already exist at ~/models/.
MODEL_PATH = "/Users/jj/models/Qwen3.6-27B-UD-Q8_K_XL.gguf"

# Cached peak-bandwidth measurement (written by setup, read by experiment.py).
import os
PEAK_BW_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peak_bw.json")

# ---------------------------------------------------------------------------
# Utilities (imported by experiment.py)
# ---------------------------------------------------------------------------

import json
import time


class Timer:
    """Wall-clock timer for enforcing time budgets."""

    def __init__(self, budget=TIME_BUDGET):
        self.budget = budget
        self._start = time.time()

    def elapsed(self):
        return time.time() - self._start

    def remaining(self):
        return max(0, self.budget - self.elapsed())

    def expired(self):
        return self.elapsed() >= self.budget


def print_results(**metrics):
    """Print results in the standard format for the agent to parse.

    Usage:
        print_results(bandwidth_utilization_pct=72.5, time_seconds=42.3)

    Prints:
        ---
        bandwidth_utilization_pct: 72.500000
        time_seconds: 42.300000
    """
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def load_peak_bw():
    """Return the measured peak bandwidth in GB/s (run harness.py first)."""
    if not os.path.exists(PEAK_BW_FILE):
        raise FileNotFoundError(
            f"{PEAK_BW_FILE} not found. Run `uv run harness.py` first to measure peak bandwidth."
        )
    with open(PEAK_BW_FILE) as f:
        data = json.load(f)
    return float(data["peak_bw_measured_gbs"])


def model_size_bytes():
    """Bytes on disk for the GGUF — proxy for bytes-per-decode-token."""
    return os.path.getsize(MODEL_PATH)


# ---------------------------------------------------------------------------
# One-time setup: measure peak memory bandwidth via MLX
# ---------------------------------------------------------------------------

def measure_peak_bandwidth_mlx(buf_gb=4.0, reps=10, warmup=3):
    """Measure GPU-accessible memory bandwidth on Apple Silicon via MLX.

    Allocates a large float32 buffer in unified memory and times a streaming
    sum. MLX dispatches to Metal, so this hits the same memory subsystem
    llama.cpp uses with --n-gpu-layers 99.

    Returns peak bandwidth in GB/s (best-of-reps).
    """
    import mlx.core as mx

    n_elements = int((buf_gb * 1024**3) / 4)  # float32 = 4 bytes
    arr = mx.random.normal(shape=(n_elements,), dtype=mx.float32)
    mx.eval(arr)
    bytes_read = arr.nbytes

    # Warmup
    for _ in range(warmup):
        s = mx.sum(arr)
        mx.eval(s)

    timings = []
    for _ in range(reps):
        t0 = time.perf_counter()
        s = mx.sum(arr)
        mx.eval(s)
        timings.append(time.perf_counter() - t0)

    best_t = min(timings)
    median_t = sorted(timings)[len(timings) // 2]
    return {
        "peak_bw_measured_gbs": bytes_read / best_t / 1e9,
        "median_bw_gbs": bytes_read / median_t / 1e9,
        "buffer_bytes": bytes_read,
        "best_seconds": best_t,
        "reps": reps,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Roofline experiment setup")
    print("=" * 60)
    print()
    print(f"Metric:       {METRIC_NAME} ({METRIC_GOAL})")
    print(f"Time budget:  {TIME_BUDGET}s per run")
    print(f"Model:        {MODEL_PATH}")
    print(f"Spec peak BW: {PEAK_BW_SPEC_GBS} GB/s")
    print()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"ERROR: model not found at {MODEL_PATH}")

    size_gb = model_size_bytes() / 1024**3
    print(f"Model size:   {size_gb:.2f} GB")
    print()

    print("Measuring peak memory bandwidth via MLX (4 GB buffer, 10 reps)...")
    result = measure_peak_bandwidth_mlx()
    print(f"  best:   {result['peak_bw_measured_gbs']:.1f} GB/s")
    print(f"  median: {result['median_bw_gbs']:.1f} GB/s")
    print(f"  spec:   {PEAK_BW_SPEC_GBS:.1f} GB/s")
    print(f"  best is {100 * result['peak_bw_measured_gbs'] / PEAK_BW_SPEC_GBS:.1f}% of spec")
    print()

    with open(PEAK_BW_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {PEAK_BW_FILE}")
    print()
    print("Setup complete. Run your first experiment with: uv run experiment.py")
