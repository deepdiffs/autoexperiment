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

# llama.cpp server (this Mac Studio). Override with LLAMA_BASE_URL if needed.
LLAMA_BASE_URL_DEFAULT = "http://localhost:8080/v1"

# The two models under benchmark. These must be loaded on the llama.cpp server.
MODEL_GEMMA = "gemma-4-26B-A4B-it-UD-Q4_K_M"
MODEL_SUPERGEMMA = "supergemma4-26b-uncensored-fast-v2-Q4_K_M"

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

    base_url = os.environ.get("LLAMA_BASE_URL", LLAMA_BASE_URL_DEFAULT)

    print(f"Metric:      {METRIC_NAME} ({METRIC_GOAL})")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    print(f"Endpoint: {base_url}")
    r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=15)
    r.raise_for_status()
    entries = r.json().get("data", [])
    models = sorted(m["id"] for m in entries)
    # llama.cpp /v1/models also exposes load status under m["status"]["value"]
    loaded = {m["id"] for m in entries if (m.get("status") or {}).get("value") == "loaded"}
    print(f"Models at endpoint ({len(models)}):")
    for m in models:
        tag = "loaded" if m in loaded else "unloaded"
        print(f"  - {m} [{tag}]")
    print()

    ok = True
    for required in (MODEL_GEMMA, MODEL_SUPERGEMMA):
        present = required in models
        is_loaded = required in loaded
        ok = ok and present
        if present and is_loaded:
            tag = "OK (loaded)"
        elif present:
            tag = "present but NOT loaded"
        else:
            tag = "MISSING"
        print(f"  {required}: {tag}")

    if not ok:
        raise SystemExit(1)

    print()
    print("Ready. Run the first experiment with: uv run experiment.py")
