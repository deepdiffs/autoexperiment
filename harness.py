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
TIME_BUDGET = 240                       # seconds per invocation (two models back-to-back)

# Cap: experiment.py is invoked at most this many times. Each invocation runs
# both models with the same config, yielding MAX_RUNS TSV rows per model.
MAX_RUNS = 100

# mlx_lm.server (this Mac Studio). Default port is mlx_lm's own default (8000).
# Override with MLX_BASE_URL if you start the server on another port.
# Start the server manually in another terminal, e.g.:
#   mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-MLX-8bit --port 8000
# The server can swap between loaded models on demand; both MODEL_A and MODEL_B
# must be present on disk (pre-downloaded). No restart needed between models.
MLX_BASE_URL_DEFAULT = "http://localhost:8000/v1"

# The two MLX models compared head-to-head on every invocation, in this order.
MODEL_A = "unsloth/Qwen3.6-35B-A3B-MLX-8bit"   # MoE, 35B total / ~3B active
MODEL_B = "unsloth/Qwen3.6-27B-MLX-8bit"       # dense, 27B
MODEL_IDS = (MODEL_A, MODEL_B)

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
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=15, trust_env=False)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR: cannot reach server: {e}")
        print()
        print("  Start it in another terminal with:")
        print(f"    mlx_lm.server --model {MODEL_A} --port 8000")
        raise SystemExit(1)

    entries = r.json().get("data", [])
    models = {m["id"] for m in entries}

    print(f"Target models ({len(MODEL_IDS)}):")
    for model_id in MODEL_IDS:
        if model_id not in models:
            # mlx_lm.server typically loads the model on first request, so missing
            # from /v1/models is not necessarily fatal — but warn loudly.
            print(f"  - {model_id}: NOT LISTED at /v1/models (may load lazily on first request)")
        else:
            print(f"  - {model_id}: present")

    # Smoke-test a tiny completion against each model.
    for model_id in MODEL_IDS:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=180, trust_env=False)
            r.raise_for_status()
            resp = r.json()
            usage = resp.get("usage") or {}
            print(f"  smoke test OK for {model_id}: completion_tokens={usage.get('completion_tokens')}")
        except Exception as e:
            print(f"  ERROR running smoke test for {model_id}: {e}")
            raise SystemExit(1)

    print()
    print("Ready. Run the first experiment with: uv run experiment.py")
