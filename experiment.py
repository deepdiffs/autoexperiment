"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Benchmark end-to-end tokens/sec of two Qwen models served by a local
llama.cpp server on this Mac Studio (default: http://localhost:8080/v1).

Head-to-head paradigm: every invocation runs the EXACT SAME request against
MODEL_QWEN_BF16 and then MODEL_QWEN_Q8, and prints one result block per
model. The agent tunes config knobs below; each knob change is evaluated
against both models so results are directly comparable.

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
    LLAMA_BASE_URL_DEFAULT,
    MODEL_QWEN_BF16,
    MODEL_QWEN_Q8,
    Timer,
    print_results,
)

# ---------------------------------------------------------------------------
# Configuration — the agent tunes these.
# The SAME config is applied to both models on every invocation.
# ---------------------------------------------------------------------------

# The two models compared head-to-head, in this order. Do not remove either —
# the whole point of this experiment is to run each config against both.
MODELS = [MODEL_QWEN_BF16, MODEL_QWEN_Q8]

PROMPT_TYPE = "code_gen"
MAX_TOKENS = 1024
NUM_TRIALS = 3
TEMPERATURE = 0.0
TOP_K = None
STREAM = True

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


def run_model(client, url, headers, prompt, model, timer):
    """Run warmup + NUM_TRIALS timed trials against a single model.

    Returns the aggregated per-model result dict (or None if time ran out
    before any trial completed)."""
    print(f"[{model}] warmup...", flush=True)
    _ = one_trial(client, url, headers, prompt, model, MAX_TOKENS, TEMPERATURE, TOP_K, STREAM)

    trials = []
    for i in range(NUM_TRIALS):
        if timer.remaining() < 5:
            print(
                f"[{model}] time budget nearly exhausted after {i} trials; stopping early",
                flush=True,
            )
            break
        t = one_trial(client, url, headers, prompt, model, MAX_TOKENS, TEMPERATURE, TOP_K, STREAM)
        if t["completion_tokens"] < MIN_COMPLETION_TOKENS:
            raise RuntimeError(
                f"[{model}] trial {i + 1} produced only {t['completion_tokens']} completion tokens "
                f"(< {MIN_COMPLETION_TOKENS}); invalid run"
            )
        print(
            f"[{model}]  trial {i + 1}: e2e={t['e2e_tok_s']:.2f} tok/s "
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

base_url = os.environ.get("LLAMA_BASE_URL", LLAMA_BASE_URL_DEFAULT).rstrip("/")
url = f"{base_url}/chat/completions"
headers = {}

prompt = PROMPTS[PROMPT_TYPE]

print(
    f"config: models={MODELS} prompt_type={PROMPT_TYPE} "
    f"max_tokens={MAX_TOKENS} trials={NUM_TRIALS} temp={TEMPERATURE}",
    flush=True,
)

aggregates = []
with httpx.Client(trust_env=False) as client:
    for model in MODELS:
        agg = run_model(client, url, headers, prompt, model, timer)
        if agg is None:
            raise RuntimeError(f"no timed trials completed for model {model}")
        aggregates.append(agg)

# ---------------------------------------------------------------------------
# Results — one block per model. The agent parses these separately and writes
# one row to results.tsv per model.
# ---------------------------------------------------------------------------

for agg in aggregates:
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
        "model": agg["model"],
        "prompt_type": PROMPT_TYPE,
        "max_tokens": MAX_TOKENS,
    })

# Easy-to-grep summary: one RESULT line per model.
print()
for agg in aggregates:
    print(f"RESULT model={agg['model']} {METRIC_NAME}={agg['e2e_tok_s']:.6f}")

print(f"time_seconds: {timer.elapsed():.6f}")
