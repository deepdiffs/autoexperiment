"""
Experiment harness — fixed configuration and utilities.
The agent does NOT modify this file.

Usage:
    uv run harness.py    # verify connectivity and list available models
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRIC_NAME = "tokens_per_second"   # end-to-end completion_tokens / total wall time
METRIC_GOAL = "maximize"
TIME_BUDGET = 1800                  # seconds per run (four models x two stream modes per invocation)

# Cap: experiment.py is invoked at most this many times. Each invocation tests
# all four models with the same config, so this yields MAX_RUNS data points per model.
MAX_RUNS = 25

# LMStudio server (this Mac Studio). Override with LMSTUDIO_BASE_URL if needed.
LMSTUDIO_BASE_URL_DEFAULT = "http://localhost:1234/v1"

# The four models compared head-to-head on every run — a 2x2 matrix of
# architecture x backend. They should be present in LMStudio's /v1/models
# listing. LMStudio JIT-loads on first request, so they do not need to be
# pre-loaded.
#
#   27B dense      GGUF Q8_K_XL   vs   MLX 8-bit
#   35B A3B MoE    GGUF Q8_K_XL   vs   MLX 8-bit
MODEL_27B_GGUF = "qwen3.6-27b@q8_k_xl"
MODEL_27B_MLX  = "qwen3.6-27b-mlx"
MODEL_35B_GGUF = "qwen3.6-35b-a3b@q8_k_xl"
MODEL_35B_MLX  = "qwen3.6-35b-a3b-mlx"

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
#
# LMStudio exposes two listings:
#   /v1/models            — OpenAI-compatible. GGUF quant variants appear as
#                           `base-id@quant` (e.g. `qwen3.6-27b@q8_k_xl`).
#                           These are the exact IDs we pass to chat completions.
#   /api/v0/models        — LMStudio native. Reports `state` (loaded /
#                           not-loaded), `quantization`, `compatibility_type`
#                           (gguf / mlx). Collapses GGUF quant variants under
#                           a base ID, so it is only used for nice display.
#
# Presence check uses /v1/models since those IDs must match exactly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import httpx

    base_url = os.environ.get("LMSTUDIO_BASE_URL", LMSTUDIO_BASE_URL_DEFAULT)
    root = base_url.rstrip("/").removesuffix("/v1")

    print(f"Metric:      {METRIC_NAME} ({METRIC_GOAL})")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Max runs:    {MAX_RUNS}")
    print()

    # /api/v0/models is the ground truth: reports state, quantization, and
    # compatibility_type. Entry IDs are base IDs (no @quant suffix).
    print(f"Endpoint: {root}/api/v0/models")
    r = httpx.get(f"{root}/api/v0/models", timeout=15)
    r.raise_for_status()
    native_by_id = {m["id"]: m for m in r.json().get("data", [])}

    print(f"Models at endpoint ({len(native_by_id)}):")
    for mid in sorted(native_by_id):
        m = native_by_id[mid]
        print(
            f"  - {mid} [{m.get('state','?')}] "
            f"({m.get('compatibility_type','?')}, {m.get('quantization','?')})"
        )
    print()

    def parse_required(mid):
        # "qwen3.6-27b@q8_k_xl" -> ("qwen3.6-27b", "q8_k_xl")
        # "qwen3.6-27b-mlx"     -> ("qwen3.6-27b-mlx", None)
        if "@" in mid:
            base, quant = mid.split("@", 1)
            return base, quant.lower()
        return mid, None

    required = [MODEL_27B_GGUF, MODEL_27B_MLX, MODEL_35B_GGUF, MODEL_35B_MLX]
    ok = True
    for mid in required:
        base, want_quant = parse_required(mid)
        entry = native_by_id.get(base)
        if not entry:
            print(f"  {mid}: MISSING (base id {base!r} not found)")
            ok = False
            continue
        if want_quant is not None:
            have_quant = (entry.get("quantization") or "").lower()
            if have_quant != want_quant:
                print(
                    f"  {mid}: QUANT MISMATCH "
                    f"(server has {have_quant!r}, expected {want_quant!r})"
                )
                ok = False
                continue
        print(f"  {mid}: OK ({entry.get('compatibility_type','?')}, "
              f"{entry.get('quantization','?')}, {entry.get('state','?')})")

    if not ok:
        raise SystemExit(1)

    print()
    print("Note: LMStudio JIT-loads models on first request; 'not-loaded' at")
    print("startup is fine — the per-model warmup trial in run_model() absorbs")
    print("load cost.")
    print()
    print("Ready. Run the first experiment with: uv run experiment.py")
