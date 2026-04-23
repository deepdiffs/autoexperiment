"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Benchmark end-to-end tokens/sec of four Qwen models served by a local
LMStudio server on this Mac Studio (default: http://localhost:1234/v1).

Head-to-head 2x2 paradigm: every invocation runs the EXACT SAME request
against all four models in the canonical order below, and prints one result
block per model. The agent tunes config knobs below; each knob change is
evaluated against all four models so results are directly comparable across
architecture (27B dense vs 35B A3B MoE) and backend (GGUF vs MLX).

                 GGUF Q8_K_XL                 MLX 8-bit
  27B dense      qwen3.6-27b@q8_k_xl          qwen3.6-27b-mlx
  35B A3B MoE    qwen3.6-35b-a3b@q8_k_xl      qwen3.6-35b-a3b-mlx

Note: these Qwen models emit reasoning tokens (delta.reasoning_content or
usage.completion_tokens_details.reasoning_tokens) in addition to visible
content. The primary metric is TOTAL completion_tokens / total wall time,
which matches real user-perceived throughput.
"""

import json
import os
import statistics
import time

import httpx

from harness import (
    METRIC_NAME,
    TIME_BUDGET,
    LMSTUDIO_BASE_URL_DEFAULT,
    MODEL_27B_GGUF,
    MODEL_27B_MLX,
    MODEL_35B_GGUF,
    MODEL_35B_MLX,
    Timer,
    print_results,
)

# ---------------------------------------------------------------------------
# Configuration — the agent tunes these.
# The SAME config is applied to every model on every invocation.
# ---------------------------------------------------------------------------

# The four models compared head-to-head, in this order. Do not remove any —
# the whole point of this experiment is to run each config against all four.
# Canonical order: small -> large; within each size, GGUF -> MLX.
MODELS = [
    MODEL_27B_GGUF,
    MODEL_27B_MLX,
    MODEL_35B_GGUF,
    MODEL_35B_MLX,
]

PROMPT_TYPE = "qa"
MAX_TOKENS = 512
NUM_TRIALS = 5
TEMPERATURE = 0.7
TOP_K = 40

# Observe BOTH streaming modes per invocation. Non-streaming measures only
# end-to-end wall-clock throughput (ttft and decode_tok_s are unobservable and
# fall back to totals). Streaming additionally yields real ttft and decode
# tok/s. Running both per config lets us quantify any streaming overhead.
STREAM_MODES = [True, False]

# Correctness guard — each trial must produce at least this many output tokens
# (total, including reasoning). Guards against empty/broken responses.
MIN_COMPLETION_TOKENS = 50

# Prompt bank.
SUMMARIZATION_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum."
) * 3

SUMMARIZATION_TEXT_LONG = SUMMARIZATION_TEXT * 10
SUMMARIZATION_TEXT_SHORT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
)

CODE_SNIPPET = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n"
    "    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n"
)

PROMPTS = {
    "summarization": (
        "Summarize the following text in three sentences.\n\n"
        + SUMMARIZATION_TEXT
    ),
    "summarization_short": (
        "Summarize the following text in one sentence.\n\n"
        + SUMMARIZATION_TEXT_SHORT
    ),
    "summarization_long": (
        "Summarize the following text in three sentences.\n\n"
        + SUMMARIZATION_TEXT_LONG
    ),
    "open_ended": "Write a short story about a robot learning to paint.",
    "open_ended_long": (
        "Write a long, detailed short story about a robot named Arti who lives "
        "alone in an abandoned art studio and slowly teaches itself to paint by "
        "studying the dusty canvases and tools left behind. Develop the setting, "
        "the robot's internal monologue, and include specific scenes of failure "
        "and discovery. End with an ambiguous note about whether art requires a soul."
    ),
    "qa": "Explain how photosynthesis works, step by step.",
    "qa_short": "What is photosynthesis?",
    "qa_long": (
        "Explain in extensive detail how photosynthesis works, covering the light-dependent "
        "reactions, the Calvin cycle, the role of chlorophyll a and b, stroma versus thylakoid, "
        "oxygen evolution at photosystem II, ATP synthesis, and how C3, C4, and CAM plants differ."
    ),
    "code": (
        "Explain what the following Python function does and analyze its time complexity:\n\n"
        + CODE_SNIPPET
    ),
    "code_gen": "Write a Python function that implements Dijkstra's shortest-path algorithm using a heap.",
    "math": (
        "Solve the following step by step: "
        "A train leaves Station A at 60 km/h heading east. Two hours later, another train leaves "
        "Station A at 90 km/h in the same direction. How many hours after the second train "
        "departs will it overtake the first?"
    ),
    "math_hard": (
        "Prove that the sum of the first n odd positive integers equals n^2, using mathematical "
        "induction. Show each step carefully."
    ),
    "multilingual_es": "Explica paso a paso cómo funciona la fotosíntesis en español.",
    "multilingual_zh": "请逐步解释光合作用的工作原理。",
    "multilingual_fr": "Expliquez étape par étape comment fonctionne la photosynthèse en français.",
}


# ---------------------------------------------------------------------------
# HTTP trial
# ---------------------------------------------------------------------------

def one_trial(client, url, headers, prompt, model, max_tokens, temperature, top_k=None, stream=True):
    """Run one chat completion and return timing + usage stats."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if top_k is not None:
        body["top_k"] = top_k

    start = time.perf_counter()
    first_any_token_at = None
    first_content_at = None
    last_any_token_at = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_tokens = None
    cached_prompt_tokens = None

    if stream:
        with client.stream("POST", url, headers=headers, json=body, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)

                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    saw_any = False
                    if delta.get("reasoning_content"):
                        saw_any = True
                    if delta.get("content"):
                        saw_any = True
                        if first_content_at is None:
                            first_content_at = time.perf_counter()
                    if saw_any:
                        now = time.perf_counter()
                        if first_any_token_at is None:
                            first_any_token_at = now
                        last_any_token_at = now

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = details.get("reasoning_tokens")
                    pt_details = usage.get("prompt_tokens_details") or {}
                    cached_prompt_tokens = pt_details.get("cached_tokens")
    else:
        r = client.post(url, headers=headers, json=body, timeout=300)
        r.raise_for_status()
        resp = r.json()
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")
        pt_details = usage.get("prompt_tokens_details") or {}
        cached_prompt_tokens = pt_details.get("cached_tokens")

    end = time.perf_counter()
    total_s = end - start

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    content_tokens = max(completion_tokens - reasoning_tokens, 0)

    e2e_tok_s = (completion_tokens / total_s) if total_s > 0 else 0.0

    if first_any_token_at is not None and first_any_token_at > start:
        ttft_s = first_any_token_at - start
        prefill_tok_s = (prompt_tokens / ttft_s) if prompt_tokens else 0.0
    else:
        ttft_s = total_s
        prefill_tok_s = 0.0

    if (
        first_any_token_at is not None
        and last_any_token_at is not None
        and (last_any_token_at - first_any_token_at) > 0.05
    ):
        decode_window_s = last_any_token_at - first_any_token_at
        decode_tok_s = (completion_tokens - 1) / decode_window_s if completion_tokens > 1 else 0.0
    else:
        decode_tok_s = 0.0

    return {
        "total_s": total_s,
        "ttft_ms": ttft_s * 1000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "cached_prompt_tokens": cached_prompt_tokens or 0,
        "e2e_tok_s": e2e_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
    }


def trim_mean(xs):
    """Drop min and max, mean the middle. Falls back to plain mean for small n."""
    if not xs:
        return 0.0
    if len(xs) <= 2:
        return statistics.mean(xs)
    return statistics.mean(sorted(xs)[1:-1])


def run_model(client, url, headers, prompt, model, timer, stream):
    """Run warmup + NUM_TRIALS timed trials against a single model+stream combo.

    Returns the aggregated per-model result dict (or None if time ran out
    before any trial completed)."""
    tag = f"{model}|{'stream' if stream else 'nostream'}"
    print(f"[{tag}] warmup...", flush=True)
    _ = one_trial(client, url, headers, prompt, model, MAX_TOKENS, TEMPERATURE, TOP_K, stream)

    trials = []
    for i in range(NUM_TRIALS):
        if timer.remaining() < 5:
            print(
                f"[{tag}] time budget nearly exhausted after {i} trials; stopping early",
                flush=True,
            )
            break
        t = one_trial(client, url, headers, prompt, model, MAX_TOKENS, TEMPERATURE, TOP_K, stream)
        if t["completion_tokens"] < MIN_COMPLETION_TOKENS:
            raise RuntimeError(
                f"[{tag}] trial {i + 1} produced only {t['completion_tokens']} completion tokens "
                f"(< {MIN_COMPLETION_TOKENS}); invalid run"
            )
        print(
            f"[{tag}]  trial {i + 1}: e2e={t['e2e_tok_s']:.2f} tok/s "
            f"decode={t['decode_tok_s']:.2f} tok/s "
            f"ttft={t['ttft_ms']:.0f}ms "
            f"total={t['completion_tokens']} tok "
            f"(reasoning={t['reasoning_tokens']}, content={t['content_tokens']})",
            flush=True,
        )
        trials.append(t)

    if not trials:
        return None

    return {
        "model": model,
        "stream": stream,
        "e2e_tok_s": trim_mean([t["e2e_tok_s"] for t in trials]),
        "decode_tok_s": trim_mean([t["decode_tok_s"] for t in trials]),
        "prefill_tok_s": trim_mean([t["prefill_tok_s"] for t in trials]),
        "ttft_ms": trim_mean([t["ttft_ms"] for t in trials]),
        "total_s_per_trial": trim_mean([t["total_s"] for t in trials]),
        "prompt_tokens": trials[0]["prompt_tokens"],
        "completion_tokens_avg": trim_mean([t["completion_tokens"] for t in trials]),
        "reasoning_tokens_avg": trim_mean([t["reasoning_tokens"] for t in trials]),
        "content_tokens_avg": trim_mean([t["content_tokens"] for t in trials]),
        "cached_prompt_tokens": trials[0]["cached_prompt_tokens"],
        "trials_completed": len(trials),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

timer = Timer()

base_url = os.environ.get("LMSTUDIO_BASE_URL", LMSTUDIO_BASE_URL_DEFAULT).rstrip("/")
url = f"{base_url}/chat/completions"
headers = {}

prompt = PROMPTS[PROMPT_TYPE]

print(
    f"config: models={MODELS} prompt_type={PROMPT_TYPE} "
    f"max_tokens={MAX_TOKENS} trials={NUM_TRIALS} temp={TEMPERATURE} "
    f"stream_modes={STREAM_MODES}",
    flush=True,
)

aggregates = []
crashes = []
with httpx.Client(trust_env=False) as client:
    for stream_mode in STREAM_MODES:
        for model in MODELS:
            try:
                agg = run_model(client, url, headers, prompt, model, timer, stream_mode)
            except Exception as e:
                label = f"{model}|{'stream' if stream_mode else 'nostream'}"
                print(f"[{label}] CRASH: {type(e).__name__}: {e}", flush=True)
                crashes.append((model, stream_mode, f"{type(e).__name__}: {e}"))
                continue
            if agg is None:
                label = f"{model}|{'stream' if stream_mode else 'nostream'}"
                print(f"[{label}] CRASH: no timed trials completed", flush=True)
                crashes.append((model, stream_mode, "no timed trials completed"))
                continue
            aggregates.append(agg)

# ---------------------------------------------------------------------------
# Results — one block per (model, stream_mode). The agent parses each block
# and writes one row to results.tsv per block. We encode stream mode into the
# model label via a "|stream" / "|nostream" suffix so it becomes a distinct
# row in results.tsv without needing a schema change.
# ---------------------------------------------------------------------------

for agg in aggregates:
    label = f"{agg['model']}|{'stream' if agg['stream'] else 'nostream'}"
    print_results(**{
        METRIC_NAME: agg["e2e_tok_s"],
        "decode_tok_s": agg["decode_tok_s"],
        "prefill_tok_s": agg["prefill_tok_s"],
        "ttft_ms": agg["ttft_ms"],
        "total_s_per_trial": agg["total_s_per_trial"],
        "prompt_tokens": agg["prompt_tokens"],
        "cached_prompt_tokens": agg["cached_prompt_tokens"],
        "completion_tokens_avg": agg["completion_tokens_avg"],
        "reasoning_tokens_avg": agg["reasoning_tokens_avg"],
        "content_tokens_avg": agg["content_tokens_avg"],
        "trials_completed": agg["trials_completed"],
        "model": label,
        "stream": agg["stream"],
        "prompt_type": PROMPT_TYPE,
        "max_tokens": MAX_TOKENS,
    })

# Easy-to-grep summary: one RESULT line per (model, stream_mode).
print()
for agg in aggregates:
    label = f"{agg['model']}|{'stream' if agg['stream'] else 'nostream'}"
    print(f"RESULT model={label} {METRIC_NAME}={agg['e2e_tok_s']:.6f}")
for model, stream_mode, reason in crashes:
    label = f"{model}|{'stream' if stream_mode else 'nostream'}"
    print(f"CRASH model={label} reason={reason}")

print(f"time_seconds: {timer.elapsed():.6f}")
