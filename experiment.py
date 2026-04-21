"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Benchmark end-to-end tokens/sec of a Gemma model served by a remote
llama.cpp + LiteLLM endpoint. Streams chat-completion requests and reports
the trimmed mean across timed trials.

Note: these Gemma models emit reasoning tokens (delta.reasoning_content or
usage.completion_tokens_details.reasoning_tokens) in addition to visible
content. The primary metric is TOTAL completion_tokens / total wall time,
which matches real user-perceived throughput.
"""

import json
import os
import statistics
import time

import httpx

from harness import METRIC_NAME, TIME_BUDGET, Timer, print_results

# ---------------------------------------------------------------------------
# Configuration — the agent tunes these
# ---------------------------------------------------------------------------

MODEL = "gemma4-26b"            # one of: "gemma4-26b", "supergemma4-26b"
PROMPT_TYPE = "summarization"   # one of: "summarization", "open_ended", "qa"
MAX_TOKENS = 200                # generation length cap (total; includes reasoning)
NUM_TRIALS = 5                  # timed trials (+ 1 untimed warmup)
TEMPERATURE = 0.7

# Correctness guard — each trial must produce at least this many output tokens
# (total, including reasoning). Guards against empty/broken responses.
MIN_COMPLETION_TOKENS = 50

# Prompt bank. The baseline phase covers all three.
SUMMARIZATION_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum."
) * 3

PROMPTS = {
    "summarization": (
        "Summarize the following text in three sentences.\n\n"
        + SUMMARIZATION_TEXT
    ),
    "open_ended": "Write a short story about a robot learning to paint.",
    "qa": "Explain how photosynthesis works, step by step.",
}


# ---------------------------------------------------------------------------
# HTTP trial
# ---------------------------------------------------------------------------

def one_trial(client, url, headers, prompt, model, max_tokens, temperature):
    """Stream one chat completion and return timing + usage stats.

    Tracks first-token time across both `delta.content` and
    `delta.reasoning_content` (reasoning-model extension). Returns
    end-to-end and (when measurable) decode-only throughput.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_any_token_at = None     # first reasoning OR content token
    first_content_at = None       # first visible content token
    last_any_token_at = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_tokens = None
    cached_prompt_tokens = None

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

    end = time.perf_counter()
    total_s = end - start

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    content_tokens = max(completion_tokens - reasoning_tokens, 0)

    # e2e: total tokens / total time (always valid, always primary).
    e2e_tok_s = (completion_tokens / total_s) if total_s > 0 else 0.0

    # Prefill: prompt tokens per time-to-first-streamed-token-of-any-kind.
    if first_any_token_at is not None and first_any_token_at > start:
        ttft_s = first_any_token_at - start
        prefill_tok_s = (prompt_tokens / ttft_s) if prompt_tokens else 0.0
    else:
        ttft_s = total_s  # stream fully buffered — can't separate
        prefill_tok_s = 0.0

    # Decode: only meaningful if we saw chunks with a real time window between
    # first and last streamed token. If the proxy buffers, last - first ≈ 0
    # and we report 0.0 rather than a bogus huge number.
    if (
        first_any_token_at is not None
        and last_any_token_at is not None
        and (last_any_token_at - first_any_token_at) > 0.05  # 50ms window min
    ):
        decode_window_s = last_any_token_at - first_any_token_at
        # completion_tokens includes the very first token; we have (N-1)
        # inter-token gaps in the window.
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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

timer = Timer()

base_url = os.environ["LITELLM_BASE_URL"].rstrip("/")
api_key = os.environ["LITELLM_MASTER_KEY"]
url = f"{base_url}/chat/completions"
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
    "Authorization": f"Bearer {api_key}",
}

prompt = PROMPTS[PROMPT_TYPE]

print(
    f"config: model={MODEL} prompt_type={PROMPT_TYPE} "
    f"max_tokens={MAX_TOKENS} trials={NUM_TRIALS} temp={TEMPERATURE}",
    flush=True,
)

trials = []
with httpx.Client() as client:
    print("warmup...", flush=True)
    _ = one_trial(client, url, headers, prompt, MODEL, MAX_TOKENS, TEMPERATURE)

    for i in range(NUM_TRIALS):
        if timer.remaining() < 5:
            print(
                f"time budget nearly exhausted after {i} trials; stopping early",
                flush=True,
            )
            break
        t = one_trial(client, url, headers, prompt, MODEL, MAX_TOKENS, TEMPERATURE)
        if t["completion_tokens"] < MIN_COMPLETION_TOKENS:
            raise RuntimeError(
                f"trial {i + 1} produced only {t['completion_tokens']} completion tokens "
                f"(< {MIN_COMPLETION_TOKENS}); invalid run"
            )
        print(
            f"  trial {i + 1}: e2e={t['e2e_tok_s']:.2f} tok/s "
            f"decode={t['decode_tok_s']:.2f} tok/s "
            f"ttft={t['ttft_ms']:.0f}ms "
            f"total={t['completion_tokens']} tok "
            f"(reasoning={t['reasoning_tokens']}, content={t['content_tokens']})",
            flush=True,
        )
        trials.append(t)

if not trials:
    raise RuntimeError("no timed trials completed")

e2e_tok_s = trim_mean([t["e2e_tok_s"] for t in trials])
decode_tok_s = trim_mean([t["decode_tok_s"] for t in trials])
prefill_tok_s = trim_mean([t["prefill_tok_s"] for t in trials])
ttft_ms = trim_mean([t["ttft_ms"] for t in trials])
total_s_per_trial = trim_mean([t["total_s"] for t in trials])
prompt_tokens = trials[0]["prompt_tokens"]
completion_tokens_avg = trim_mean([t["completion_tokens"] for t in trials])
reasoning_tokens_avg = trim_mean([t["reasoning_tokens"] for t in trials])
content_tokens_avg = trim_mean([t["content_tokens"] for t in trials])
cached_prompt_tokens = trials[0]["cached_prompt_tokens"]

# ---------------------------------------------------------------------------
# Results (the agent parses this output)
# ---------------------------------------------------------------------------

print_results(**{
    METRIC_NAME: e2e_tok_s,
    "decode_tok_s": decode_tok_s,
    "prefill_tok_s": prefill_tok_s,
    "ttft_ms": ttft_ms,
    "total_s_per_trial": total_s_per_trial,
    "prompt_tokens": prompt_tokens,
    "cached_prompt_tokens": cached_prompt_tokens,
    "completion_tokens_avg": completion_tokens_avg,
    "reasoning_tokens_avg": reasoning_tokens_avg,
    "content_tokens_avg": content_tokens_avg,
    "trials_completed": len(trials),
    "model": MODEL,
    "prompt_type": PROMPT_TYPE,
    "max_tokens": MAX_TOKENS,
    "time_seconds": timer.elapsed(),
})
