"""
Experiment harness — fixed configuration and utilities.
The agent does NOT modify this file.

Usage:
    uv run harness.py    # verify connectivity and list available models
"""

# ---------------------------------------------------------------------------
# Configuration (customize these for your experiment)
# ---------------------------------------------------------------------------

METRIC_NAME = "tokens_per_second"   # end-to-end completion_tokens / total wall time
METRIC_GOAL = "maximize"
TIME_BUDGET = 120                   # seconds per run

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
        print_results(tokens_per_second=9.5, time_seconds=42.3)

    Prints:
        ---
        tokens_per_second: 9.500000
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

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_MASTER_KEY")

    print(f"Metric:      {METRIC_NAME} ({METRIC_GOAL})")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    if not base_url or not api_key:
        print("ERROR: set LITELLM_BASE_URL and LITELLM_MASTER_KEY in the environment.")
        raise SystemExit(1)

    print(f"Endpoint: {base_url}")
    r = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"x-api-key": api_key, "Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    r.raise_for_status()
    models = sorted(m["id"] for m in r.json().get("data", []))
    print(f"Models at endpoint: {models}")
    print()

    ok = True
    for required in ("gemma4-26b", "supergemma4-26b"):
        present = required in models
        ok = ok and present
        tag = "OK" if present else "MISSING"
        print(f"  {required}: {tag}")

    if not ok:
        raise SystemExit(1)

    print()
    print("Ready. Run the first experiment with: uv run experiment.py")
