"""
Experiment harness — fixed configuration and utilities.
The agent does NOT modify this file.

Usage:
    uv run harness.py    # verify the mlx_lm.server is up and the model is loaded
"""

# ---------------------------------------------------------------------------
# Configuration (customize these for your experiment)
# ---------------------------------------------------------------------------

METRIC_NAME = "e2e_tokens_per_second"   # completion_tokens / total wall time
METRIC_GOAL = "maximize"
TIME_BUDGET = 120                       # seconds per invocation

# Cap: experiment.py is invoked at most this many times.
MAX_RUNS = 100

# mlx_lm.server (this Mac Studio). Default port is mlx_lm's own default (8000).
# Override with MLX_BASE_URL if you start the server on another port.
# Start the server manually in another terminal, e.g.:
#   mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-MLX-8bit --port 8000
MLX_BASE_URL_DEFAULT = "http://localhost:8000/v1"

# The single MLX model under test.
MODEL_ID = "unsloth/Qwen3.6-35B-A3B-MLX-8bit"

# ---------------------------------------------------------------------------
# Utilities (imported by experiment.py)
# ---------------------------------------------------------------------------

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
        print_results(e2e_tokens_per_second=12.3, time_seconds=42.3)

    Prints:
        ---
        e2e_tokens_per_second: 12.300000
        time_seconds: 42.300000
    """
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# Connectivity check (one-time). Run with: uv run harness.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import httpx

    base_url = os.environ.get("MLX_BASE_URL", MLX_BASE_URL_DEFAULT)

    print(f"Metric:      {METRIC_NAME} ({METRIC_GOAL})")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Max runs:    {MAX_RUNS}")
    print()

    print(f"Endpoint: {base_url}")
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR: cannot reach server: {e}")
        print()
        print("  Start it in another terminal with:")
        print(f"    mlx_lm.server --model {MODEL_ID} --port 8000")
        raise SystemExit(1)

    entries = r.json().get("data", [])
    models = sorted(m["id"] for m in entries)
    print(f"Models at endpoint ({len(models)}):")
    for m in models:
        print(f"  - {m}")
    print()

    if MODEL_ID not in models:
        # mlx_lm.server typically loads the model on first request, so missing
        # from /v1/models is not necessarily fatal — but warn loudly.
        print(f"  WARNING: {MODEL_ID} not listed at /v1/models.")
        print("  mlx_lm.server may load it lazily on first /v1/chat/completions call.")
        print("  Attempting a tiny chat completion to confirm...")
    else:
        print(f"  {MODEL_ID}: present")

    # Smoke-test a tiny completion to confirm the model actually generates.
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        resp = r.json()
        usage = resp.get("usage") or {}
        print()
        print(f"  smoke test OK: completion_tokens={usage.get('completion_tokens')}")
    except Exception as e:
        print(f"  ERROR running smoke test: {e}")
        raise SystemExit(1)

    print()
    print("Ready. Run the first experiment with: uv run experiment.py")
