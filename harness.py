"""
Experiment harness — fixed configuration and utilities.
The agent does NOT modify this file.

Customize these settings for your specific experiment before starting the agent.

Usage:
    # If your experiment needs one-time setup, add it here and run:
    uv run harness.py
"""

# ---------------------------------------------------------------------------
# Configuration (customize these for your experiment)
# ---------------------------------------------------------------------------

METRIC_NAME = "score"       # name of the primary metric to optimize
METRIC_GOAL = "minimize"    # "minimize" or "maximize"
TIME_BUDGET = 300           # experiment time budget in seconds (5 minutes)

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
        print_results(score=0.95, time_seconds=42.3)

    Prints:
        ---
        score: 0.950000
        time_seconds: 42.300000
    """
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# One-time setup (optional — add your data prep, downloads, etc. here)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("No setup required. Ready to experiment.")
    print()
    print(f"  Metric:      {METRIC_NAME} ({METRIC_GOAL})")
    print(f"  Time budget: {TIME_BUDGET}s")
    print()
    print("Run your first experiment with: uv run experiment.py")
