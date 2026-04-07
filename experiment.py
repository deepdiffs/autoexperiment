"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Everything is fair game: algorithm, parameters, approach, implementation.
The only constraint: call print_results() at the end with the metric
defined in harness.py.
"""

from harness import METRIC_NAME, TIME_BUDGET, Timer, print_results

# ---------------------------------------------------------------------------
# Your experiment code here
# ---------------------------------------------------------------------------

timer = Timer()

# TODO: Replace this with your actual experiment.
# The agent will iterate on this code to optimize the metric.
result = 0.0

# ---------------------------------------------------------------------------
# Results (the agent parses this output)
# ---------------------------------------------------------------------------

print_results(**{
    METRIC_NAME: result,
    "time_seconds": timer.elapsed(),
})
