"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Benchmark two MLX-served Qwen models head-to-head via mlx_lm.server's
OpenAI-compatible HTTP endpoint (default: http://localhost:8000/v1).

Per invocation: the SAME config is run against MODEL_A and then MODEL_B.
For each model: 1 warmup + NUM_TRIALS timed trials of the same chat completion.
One result block and one RESULT line are printed per model. The agent logs
two rows in results.tsv per invocation (one per model).
"""

import json
import os
import statistics
import time

import httpx

from harness import (
    METRIC_NAME,
    TIME_BUDGET,
    MLX_BASE_URL_DEFAULT,
    MODEL_IDS,
    Timer,
    print_results,
    log_result,
    log_progress,
)

# ---------------------------------------------------------------------------
# Configuration — the agent tunes these.
# ---------------------------------------------------------------------------

PROMPT_TYPE = "qa_long"
MAX_TOKENS = 128
NUM_TRIALS = 5
TEMPERATURE = 0.0
TOP_P = None
TOP_K = None
STREAM = True

# Extra mlx_lm-specific knobs forwarded into the request body. mlx_lm.server
# accepts arbitrary extra fields and passes the recognized ones through to its
# generate path. Set to None to omit the field.
REPETITION_PENALTY = None        # e.g. 1.05
REPETITION_CONTEXT_SIZE = None   # e.g. 20
MIN_P = None                     # e.g. 0.05
MIN_TOKENS_TO_KEEP = None        # for sampling

# Speculative decoding via a smaller draft model. Most mlx_lm.server versions
# accept "draft_model" + "num_draft_tokens" in the request body; older versions
# ignore unknown fields. Set DRAFT_MODEL to a model id (already loadable by
# mlx_lm) to enable. Leave None to disable.
DRAFT_MODEL = None
NUM_DRAFT_TOKENS = None

# Correctness guard — each trial must produce at least this many output tokens.
# Guards against empty/broken responses.
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

def build_body(prompt, model_id):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": STREAM,
    }
    if STREAM:
        body["stream_options"] = {"include_usage": True}
    if TOP_P is not None:
        body["top_p"] = TOP_P
    if TOP_K is not None:
        body["top_k"] = TOP_K
    if REPETITION_PENALTY is not None:
        body["repetition_penalty"] = REPETITION_PENALTY
    if REPETITION_CONTEXT_SIZE is not None:
        body["repetition_context_size"] = REPETITION_CONTEXT_SIZE
    if MIN_P is not None:
        body["min_p"] = MIN_P
    if MIN_TOKENS_TO_KEEP is not None:
        body["min_tokens_to_keep"] = MIN_TOKENS_TO_KEEP
    if DRAFT_MODEL is not None:
        body["draft_model"] = DRAFT_MODEL
    if NUM_DRAFT_TOKENS is not None:
        body["num_draft_tokens"] = NUM_DRAFT_TOKENS
    return body


def one_trial(client, url, prompt, model_id):
    """Run one chat completion and return timing + usage stats."""
    body = build_body(prompt, model_id)

    start = time.perf_counter()
    first_token_at = None       # any visible/reasoning token (TTFT)
    first_content_at = None     # first non-reasoning content token
    last_token_at = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_tokens_seen = 0   # streaming-side counter (server may not report)
    content_tokens_seen = 0
    extra_usage = {}

    if STREAM:
        with client.stream("POST", url, json=body, timeout=300) as r:
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
                    # mlx_lm.server splits Qwen reasoning into delta.reasoning
                    # (vs llama.cpp's delta.reasoning_content). Treat both
                    # reasoning and content as "tokens" for TTFT/decode purposes.
                    saw_token = False
                    if delta.get("reasoning") or delta.get("reasoning_content"):
                        saw_token = True
                        reasoning_tokens_seen += 1
                    if delta.get("content"):
                        saw_token = True
                        content_tokens_seen += 1
                        if first_content_at is None:
                            first_content_at = time.perf_counter()
                    if saw_token:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        last_token_at = now

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    # Stash any non-standard usage fields mlx_lm may surface.
                    for k, v in usage.items():
                        if k not in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            extra_usage[k] = v
    else:
        r = client.post(url, json=body, timeout=300)
        r.raise_for_status()
        resp = r.json()
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        for k, v in usage.items():
            if k not in ("prompt_tokens", "completion_tokens", "total_tokens"):
                extra_usage[k] = v

    end = time.perf_counter()
    total_s = end - start

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    e2e_tok_s = (completion_tokens / total_s) if total_s > 0 else 0.0

    if first_token_at is not None and first_token_at > start:
        ttft_s = first_token_at - start
        prefill_tok_s = (prompt_tokens / ttft_s) if prompt_tokens else 0.0
    else:
        ttft_s = total_s
        prefill_tok_s = 0.0

    if (
        first_token_at is not None
        and last_token_at is not None
        and (last_token_at - first_token_at) > 0.05
        and completion_tokens > 1
    ):
        decode_window_s = last_token_at - first_token_at
        decode_tok_s = (completion_tokens - 1) / decode_window_s
    else:
        decode_tok_s = 0.0

    if first_content_at is not None and first_content_at > start:
        ttft_content_s = first_content_at - start
    else:
        ttft_content_s = 0.0

    return {
        "total_s": total_s,
        "ttft_ms": ttft_s * 1000.0,
        "ttft_content_ms": ttft_content_s * 1000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_seen": reasoning_tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "e2e_tok_s": e2e_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "extra_usage": extra_usage,
    }


def trim_mean(xs):
    """Drop min and max, mean the middle. Falls back to plain mean for small n."""
    if not xs:
        return 0.0
    if len(xs) <= 2:
        return statistics.mean(xs)
    return statistics.mean(sorted(xs)[1:-1])


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(client, url, prompt, model_id, timer):
    """Warmup + NUM_TRIALS on one model. Returns an agg dict or raises."""
    print(f"--- model: {model_id} ---", flush=True)
    print("  warmup...", flush=True)
    _ = one_trial(client, url, prompt, model_id)

    trials = []
    for i in range(NUM_TRIALS):
        if timer.remaining() < 5:
            print(f"  time budget nearly exhausted after {i} trials; stopping early", flush=True)
            break
        t = one_trial(client, url, prompt, model_id)
        if t["completion_tokens"] < MIN_COMPLETION_TOKENS:
            raise RuntimeError(
                f"{model_id} trial {i + 1} produced only {t['completion_tokens']} "
                f"completion tokens (< {MIN_COMPLETION_TOKENS}); invalid run"
            )
        print(
            f"  trial {i + 1}: e2e={t['e2e_tok_s']:.2f} tok/s "
            f"decode={t['decode_tok_s']:.2f} tok/s "
            f"prefill={t['prefill_tok_s']:.1f} tok/s "
            f"ttft={t['ttft_ms']:.0f}ms (content={t['ttft_content_ms']:.0f}ms) "
            f"out={t['completion_tokens']} tok in={t['prompt_tokens']} tok "
            f"(reasoning_chunks={t['reasoning_tokens_seen']}, content_chunks={t['content_tokens_seen']})",
            flush=True,
        )
        trials.append(t)

    if not trials:
        raise RuntimeError(f"{model_id}: no timed trials completed")

    agg = {
        "model": model_id,
        "e2e_tokens_per_second": trim_mean([t["e2e_tok_s"] for t in trials]),
        "decode_tok_s": trim_mean([t["decode_tok_s"] for t in trials]),
        "prefill_tok_s": trim_mean([t["prefill_tok_s"] for t in trials]),
        "ttft_ms": trim_mean([t["ttft_ms"] for t in trials]),
        "ttft_content_ms": trim_mean([t["ttft_content_ms"] for t in trials]),
        "total_s_per_trial": trim_mean([t["total_s"] for t in trials]),
        "prompt_tokens": trials[0]["prompt_tokens"],
        "completion_tokens_avg": trim_mean([t["completion_tokens"] for t in trials]),
        "reasoning_chunks_avg": trim_mean([t["reasoning_tokens_seen"] for t in trials]),
        "content_chunks_avg": trim_mean([t["content_tokens_seen"] for t in trials]),
        "trials_completed": len(trials),
    }

    # Merge any non-standard usage fields the server returned.
    extra_keys = set()
    for t in trials:
        extra_keys.update(t["extra_usage"].keys())
    for k in extra_keys:
        vals = [t["extra_usage"].get(k) for t in trials if k in t["extra_usage"]]
        nums = [v for v in vals if isinstance(v, (int, float))]
        if nums and len(nums) == len(vals):
            agg[f"mlx_{k}"] = trim_mean(nums)
        else:
            agg[f"mlx_{k}"] = vals[-1]

    return agg


# ---------------------------------------------------------------------------
# Run (head-to-head: same config applied to each model in MODEL_IDS)
# ---------------------------------------------------------------------------

timer = Timer()

base_url = os.environ.get("MLX_BASE_URL", MLX_BASE_URL_DEFAULT).rstrip("/")
url = f"{base_url}/chat/completions"
prompt = PROMPTS[PROMPT_TYPE]

# Build a config-snapshot description string used for BOTH rows of this
# invocation. Lists every non-None knob; one-knob-per-run rule means adjacent
# invocations' descriptions differ in exactly one key=value pair.
description_parts = [
    f"prompt_type={PROMPT_TYPE}",
    f"max_tokens={MAX_TOKENS}",
    f"num_trials={NUM_TRIALS}",
    f"temperature={TEMPERATURE}",
    f"stream={STREAM}",
]
for name, val in (
    ("top_p", TOP_P),
    ("top_k", TOP_K),
    ("repetition_penalty", REPETITION_PENALTY),
    ("repetition_context_size", REPETITION_CONTEXT_SIZE),
    ("min_p", MIN_P),
    ("min_tokens_to_keep", MIN_TOKENS_TO_KEEP),
    ("draft_model", DRAFT_MODEL),
    ("num_draft_tokens", NUM_DRAFT_TOKENS),
):
    if val is not None:
        description_parts.append(f"{name}={val}")
description = " ".join(description_parts)

print(f"config: models={list(MODEL_IDS)} {description}", flush=True)

aggregates = []
with httpx.Client(trust_env=False) as client:
    for model_id in MODEL_IDS:
        agg = run_model(client, url, prompt, model_id, timer)
        aggregates.append(agg)
        # Log to results.tsv + progress.log immediately so the human can
        # tail -f the files during the run and see each model's results
        # land as soon as its trials finish.
        log_result(model_id, agg["e2e_tokens_per_second"], "keep", description)
        log_progress(model_id, agg["e2e_tokens_per_second"], "keep", description)

# ---------------------------------------------------------------------------
# Results — one block per model. The agent parses these separately and writes
# one row to results.tsv per model.
# ---------------------------------------------------------------------------

for agg in aggregates:
    print_results(**{
        METRIC_NAME: agg["e2e_tokens_per_second"],
        "decode_tok_s": agg["decode_tok_s"],
        "prefill_tok_s": agg["prefill_tok_s"],
        "ttft_ms": agg["ttft_ms"],
        "ttft_content_ms": agg["ttft_content_ms"],
        "total_s_per_trial": agg["total_s_per_trial"],
        "prompt_tokens": agg["prompt_tokens"],
        "completion_tokens_avg": agg["completion_tokens_avg"],
        "reasoning_chunks_avg": agg["reasoning_chunks_avg"],
        "content_chunks_avg": agg["content_chunks_avg"],
        "trials_completed": agg["trials_completed"],
        "model": agg["model"],
        "prompt_type": PROMPT_TYPE,
        "max_tokens": MAX_TOKENS,
        "stream": STREAM,
        **{k: v for k, v in agg.items() if k.startswith("mlx_")},
    })

# Easy-to-grep summary: one RESULT line per model.
print()
for agg in aggregates:
    print(f"RESULT model={agg['model']} {METRIC_NAME}={agg['e2e_tokens_per_second']:.6f}")
print(f"time_seconds: {timer.elapsed():.6f}")
