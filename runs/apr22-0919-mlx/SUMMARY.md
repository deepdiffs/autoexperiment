# apr22-0919-mlx — MLX Qwen3.6 head-to-head throughput

Branch: `autoexperiment/apr22-0919-mlx`
Date: 2026-04-22
Status: stopped early by user at 22/100 invocations

## Setup

Head-to-head throughput benchmark of two MLX models served locally via `mlx_lm.server` on Apple Silicon (Mac Studio):

- **Model A**: `unsloth/Qwen3.6-35B-A3B-MLX-8bit` — MoE, 35B total / ~3B active, 8-bit
- **Model B**: `unsloth/Qwen3.6-27B-MLX-8bit` — dense 27B, 8-bit

Metric: `e2e_tokens_per_second` (completion_tokens / total_s), trim-mean across 1 warmup + 5 timed trials per model per invocation. Thinking mode enabled throughout.

## Coverage

22 invocations, 43 result rows (one crash for Model B at run 5, one transient SIGTERM for Model B at run 21).

Swept axes (one knob per commit):

- **Prompt type**: qa, summarization, open_ended, qa_short, qa_long, summarization_long, code, math, multilingual_zh, code_gen
- **max_tokens**: 128, 256, 512, 1024, 2048
- **temperature**: 0.0, 0.6, 1.0
- **top_p**: 0.95
- **top_k**: 20
- **min_p**: 0.0, 0.05
- **repetition_penalty**: 1.05

## Headline numbers

| model | range (e2e tok/s) | typical |
|---|---|---|
| 35B-A3B MoE (8-bit) | 74.5 – 80.5 | ~79 |
| 27B dense (8-bit) | 19.8 – 20.3 | ~20 |

The MoE is ~4× faster end-to-end than the dense 27B on this hardware — consistent with ~3B active parameters vs. 27B.

## Observations

- **Model rank is stable across every config tested.** MoE always wins by ~4×; no sampling/prompt combination flipped or narrowed the gap meaningfully.
- **Sampling knobs barely move throughput.** temperature 0.0 → 1.0, top_p, top_k, min_p, and repetition_penalty=1.05 all landed within ~1–2% of the baseline for each model. Sampler overhead is negligible vs. decode.
- **max_tokens=128 drops 35B-A3B to ~74 tok/s** (from ~79 at 512+). Short outputs don't amortize prefill/warmup cost. 27B barely shifts (~19.8 vs ~20.2).
- **Long outputs are fine on MoE, risky on 27B within the budget.** max_tokens=2048 with open_ended: 35B-A3B = 79.9, 27B crashed — ran out of the 240s budget after Model A consumed ~150s. A future run should either bump the budget or separate the two models when sweeping long outputs.
- **Prompt-type variation (code / math / zh / summarization) had minor effect** on throughput (~77.8–80.5 for MoE, ~19.9–20.3 for 27B). Reasoning-vs-content ratio varied but headline tok/s absorbed it.
- **Peak 35B-A3B measurement: 80.50 tok/s** at run 22 (bac8b4a, code_gen prompt, temperature=1.0 top_p=0.95 top_k=20 min_p=0.05 repetition_penalty=1.05). Likely within noise of the ~79 baseline — not a real gain.

## Crashes / gaps

- **run 5 (ea90513, max_tokens=2048, open_ended)** — 27B timed out. Logged as `crash`.
- **run 21 (08adedc, min_p=0.05, code_gen)** — 27B SIGTERMed during warmup (process was killed manually; not a model problem). Logged as `crash`.
- **run 22 (bac8b4a, repetition_penalty=1.05)** — Model A completed; Model B never ran because the invocation was interrupted when the user asked to stop. Only one row in `results.tsv`; no 27B entry.

## Files

- `results.tsv` — raw TSV, one header + 43 data rows
- `progress.log` — timestamped per-(run, model) line
- `run.log` — last invocation's stdout/stderr (truncated — each invocation overwrites)

## Ideas for a follow-up experiment

- Separate the two models across invocations when sweeping long `max_tokens`, so Model B has its own full budget.
- Try the smaller quant variants on disk (35B-A3B 3-bit / 4-bit, 27B 4-bit / 6-bit) — would need `MODEL_IDS` unlocked in `harness.py`.
- Measure cold-swap cost between Model A and Model B (currently folded into warmup).
- Push past 2048 output tokens on 35B-A3B alone — Unsloth recommends up to 32k for real use; see where decode actually saturates.
- Try `presence_penalty` to confirm whether mlx_lm.server passes it to the sampler.
