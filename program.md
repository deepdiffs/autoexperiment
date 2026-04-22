# autoexperiment — MLX Qwen3.6 head-to-head throughput exploration

Sweep generation configurations against **two** MLX-served Qwen models head-to-head and log throughput / latency metrics for each. The agent's job is to *characterize* how the two models respond to the same inputs and knobs — not to climb a single number. Every valid datapoint is a keep.

## Domain context

You are benchmarking two MLX models head-to-head:

- **Model A**: `unsloth/Qwen3.6-35B-A3B-MLX-8bit` — MoE, 35B total / ~3B active, 8-bit MLX.
- **Model B**: `unsloth/Qwen3.6-27B-MLX-8bit` — dense 27B, 8-bit MLX.
- **Serving**: `mlx_lm.server` running locally on this Mac Studio (Apple Silicon + Metal), exposing an OpenAI-compatible `/v1` API.
- **Default endpoint**: `http://localhost:8000/v1`. Override with `MLX_BASE_URL`.

Both models are pre-downloaded. `mlx_lm.server` swaps weights between models on demand — no restart needed — so each invocation runs the same config against Model A then Model B back-to-back.

The server is **already running**, started manually by the human in another terminal. You interact with it exclusively via HTTP. You cannot restart it, change launch flags, or change server-side config — only the request body.

The model ids are defined in `harness.py` as `MODEL_A`, `MODEL_B`, and `MODEL_IDS` (the ordered tuple). Import `MODEL_IDS` from `experiment.py` and iterate over it — do not hard-code the strings.

**Qwen 3.6 is a reasoning model.** mlx_lm.server (≥0.31) streams reasoning tokens under `delta.reasoning` and visible answer tokens under `delta.content`. `usage.completion_tokens` covers both. The harness counts a chunk in either field as a "token" for TTFT and decode-rate purposes, since the user feels reasoning streaming the same way they feel content streaming. `ttft_content_ms` is reported separately so you can see when the model finally exits the reasoning phase. At modest `max_tokens` (e.g. 512) on a "step by step" prompt, the model frequently never reaches `content` at all — that's normal; flag it in the description but don't treat it as a crash.

## Metrics

Per invocation, `experiment.py` runs 1 warmup + `NUM_TRIALS` timed trials **per model**, for both models, with the same config. It reports trim-mean aggregates per model (one block + one `RESULT` line per model):

| metric | meaning | notes |
|---|---|---|
| `e2e_tokens_per_second` | `completion_tokens / total_s` | **Primary metric** — the headline number per `METRIC_NAME` |
| `decode_tok_s` | `(completion_tokens - 1) / (last_token_at - first_token_at)` | Pure decode rate; only meaningful with `STREAM=True` |
| `prefill_tok_s` | `prompt_tokens / ttft_s` | Prefill throughput |
| `ttft_ms` | time to first streamed token (reasoning OR content) | True latency-to-first-byte for the user |
| `ttft_content_ms` | time to first visible content token | 0 if the model never exits reasoning within `MAX_TOKENS` |
| `reasoning_chunks_avg`, `content_chunks_avg` | streamed chunk counts per delta type | Useful to see how much of `MAX_TOKENS` was spent thinking vs answering |
| `total_s_per_trial` | wall time per trial | Useful for budget planning |
| `completion_tokens_avg`, `prompt_tokens` | output / input token counts | |
| `mlx_*` | any non-standard `usage` fields the server returns | Pass-through; differs by mlx_lm version |

`METRIC_NAME` / `METRIC_GOAL` say "maximize `e2e_tokens_per_second`" for plumbing reasons, but treat this as a **comparison/exploration experiment**, not a single-number optimization: every valid configuration is logged and kept.

## Setup

1. **Confirm the server is up**: `uv run harness.py` checks `/v1/models` and runs a small smoke completion. If it fails, tell the human to start `mlx_lm.server`.
2. **Agree on a run tag**: propose a tag that uniquely identifies this run. Pick one of:
   - `<date>-<HHMM>` — e.g. `apr21-1430`. Always unique, good default.
   - `<date>-<ordinal>` — e.g. `apr21-3`. Check `git branch --list 'autoexperiment/<date>-*'` and increment.
   - `<date>-<topic>` — e.g. `apr21-mlx-mxfp4-sweep`.
3. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
4. **Read the in-scope files**:
   - `README.md` — repository context.
   - `harness.py` — fixed configuration (metric, goal, time budget, model id, `MAX_RUNS`) and utilities. Do not modify.
   - `experiment.py` — the file you modify.
5. **Initialize logs fresh** on this branch:
   - `results.tsv` — create with just the header row: `commit\tmetric_value\tstatus\tdescription`.
   - `progress.log` — truncate (`: > progress.log`).
6. **Confirm and go**.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py`. Tune `PROMPT_TYPE`, `MAX_TOKENS`, `NUM_TRIALS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `STREAM`, prompt texts.
- Tune mlx_lm-specific knobs already plumbed in: `REPETITION_PENALTY`, `REPETITION_CONTEXT_SIZE`, `MIN_P`, `MIN_TOKENS_TO_KEEP`, `DRAFT_MODEL`, `NUM_DRAFT_TOKENS`.
- Add additional request-body fields that mlx_lm.server accepts. Unknown fields are ignored gracefully; if they're recognized they take effect.

**What you CANNOT do:**
- Change `MODEL_A`, `MODEL_B`, or `MODEL_IDS` — the models are fixed in `harness.py`. Do not skip a model or reorder on a whim; the head-to-head comparability depends on running both every invocation.
- Modify `harness.py` (read-only).
- Install new packages or add dependencies. Only what's already in `pyproject.toml`.
- Issue parallel/concurrent requests. Trials must be strictly sequential.
- Call any endpoint other than `/v1/chat/completions` or `/v1/models`.
- Restart or reconfigure the mlx_lm.server.
- Return fewer than 50 completion tokens per trial. The experiment raises on this (invalid run → crash).
- Skip the warmup trial — it absorbs prompt-cache priming and any first-call cost on this configuration.

**The goal**: collect a matrix of (config, model) → throughput/latency measurements across a wide variety of configs, so the user can characterize how each model behaves under different inputs and how the two compare side-by-side.

**Hard cap**: stop after `MAX_RUNS` (100) invocations of `experiment.py`. Each invocation writes **2 rows** to `results.tsv` (one per model), so expect ~200 rows total. Check the count of **unique commits** in `results.tsv` (minus header) before each new run — that's the invocation count, not the row count.

## Baseline phase

Before exploring, establish **3 baseline invocations** — one per `PROMPT_TYPE`, defaults otherwise:

| invocation | PROMPT_TYPE      | other knobs                        |
|------------|------------------|------------------------------------|
| 1          | `qa`             | defaults                           |
| 2          | `summarization`  | defaults                           |
| 3          | `open_ended`     | defaults                           |

That's 3 invocations / 3 TSV rows, counting toward the 100-invocation cap.

Tag each baseline row's description clearly, e.g. `baseline qa`, `baseline summarization`.

## Strategy guidance (after baseline)

**Try:**
- `MAX_TOKENS` sweeps (128 / 256 / 512 / 1024 / 2048) — how does decode throughput scale with generation length? Where does it saturate?
- Prompt-length sweeps (short / medium / long via `qa_short` / `qa` / `qa_long` and `summarization_short` / `summarization` / `summarization_long`) — characterize prefill vs decode regimes.
- Prompt content variation — code, math, multilingual. Reasoning-token output ratio likely varies.
- `TEMPERATURE` (0.0 vs > 0), `TOP_P`, `TOP_K`, `MIN_P` — sampling overhead.
- `STREAM=True` vs `STREAM=False` — non-streaming may report different `decode_tok_s` (it'll be 0 since we only get usage at the end) but `e2e_tok_s` is comparable.
- `REPETITION_PENALTY` and `REPETITION_CONTEXT_SIZE` — overhead on the sampler.
- Repeated identical requests in succession — does prefix caching kick in? Compare run-to-run prefill_tok_s.
- One variable changed per run, so deltas are attributable.
- Keep at least 1 warmup + 3 timed trials. 5 is the default and usually fine.

**Avoid:**
- Concurrency / `asyncio.gather` / threads firing multiple requests at once.
- Non-chat endpoints (`/v1/completions`, `/v1/embeddings`, custom routes).
- Any attempt to reconfigure or restart the server.
- Obviously-bad knobs (e.g. `MAX_TOKENS` absurdly high, extreme temperatures) — don't burn budget confirming the obvious.
- Changing `harness.py`, `MODEL_A`/`MODEL_B`/`MODEL_IDS`, or adding dependencies.
- `DRAFT_MODEL` unless the human has confirmed a draft model is available locally — it will likely 4xx if the draft isn't loadable.

**Tips:**
- If trial-to-trial variance exceeds ~20% of the mean, flag it in the description.
- The first trial of a brand-new prompt may show a prefill spike due to no prompt-cache hit. The warmup trial inside `experiment.py` absorbs this.
- 8-bit MLX on M-series ASi typically lands in the tens-of-tok/s range for 35B-A3B; treat anything wildly outside that as suspicious.

## Output format

Each invocation emits **one `---` block per model**, then **one `RESULT` line per model**, then one `time_seconds` line. Example (abridged):

```
---
e2e_tokens_per_second: 38.500000
decode_tok_s: 41.200000
ttft_ms: 248.000000
model: unsloth/Qwen3.6-35B-A3B-MLX-8bit
prompt_type: qa
max_tokens: 512
...
---
e2e_tokens_per_second: 44.800000
decode_tok_s: 46.100000
ttft_ms: 212.000000
model: unsloth/Qwen3.6-27B-MLX-8bit
prompt_type: qa
max_tokens: 512
...

RESULT model=unsloth/Qwen3.6-35B-A3B-MLX-8bit e2e_tokens_per_second=38.500000
RESULT model=unsloth/Qwen3.6-27B-MLX-8bit e2e_tokens_per_second=44.800000
time_seconds: 148.700000
```

Extract both metrics with:

```
grep "^RESULT " run.log
```

You should see exactly two `RESULT ` lines per successful invocation. If you see zero or one, the run is broken.

## Logging results

Log every experiment to `results.tsv` (tab-separated — commas break in descriptions).

Header + 5 columns:

```
commit	model	metric_value	status	description
```

1. git commit hash (short, 7 chars). **Same commit appears on both rows for a single invocation.**
2. full model id as served by mlx_lm (e.g. `unsloth/Qwen3.6-35B-A3B-MLX-8bit`). Must match exactly one of `MODEL_IDS`.
3. `e2e_tokens_per_second` value for that model (0.000000 for crashes).
4. status: `keep` for any valid measurement, `crash` for failures. **Do not use `discard`** — exploration, not optimization.
5. description — a **complete snapshot of every tunable knob's value for this run**, in `key=value` form, space-separated. Always include: `prompt_type`, `max_tokens`, `num_trials`, `temperature`, `stream`. Include any other knob that is not `None`: `top_p`, `top_k`, `repetition_penalty`, `repetition_context_size`, `min_p`, `min_tokens_to_keep`, `draft_model`, `num_draft_tokens`. Prefix baseline rows with `baseline ` so they're easy to grep. Never use commas (they break TSV parsing downstream).

**Every invocation produces 2 rows** — one per model — both sharing the same commit hash and the same description (the config is identical across the two models). Write them in the same order as `MODEL_IDS`. If one model crashes and the other succeeds, log both rows (one `crash`, one `keep`).

**One-knob-per-run rule.** Each new commit must change **exactly one** knob vs. the previous commit (or vs. the baseline for the very first exploration step). The description columns across two consecutive invocations should differ in exactly one `key=value` pair. This makes every delta in the primary metric attributable to a single variable. If you want to change two things, do two separate commits and two separate invocations.

Example:

```
commit	model	metric_value	status	description
a1b2c3d	unsloth/Qwen3.6-35B-A3B-MLX-8bit	79.556526	keep	baseline prompt_type=qa max_tokens=512 num_trials=5 temperature=0.0 stream=True
a1b2c3d	unsloth/Qwen3.6-27B-MLX-8bit	20.297485	keep	baseline prompt_type=qa max_tokens=512 num_trials=5 temperature=0.0 stream=True
b2c3d4e	unsloth/Qwen3.6-35B-A3B-MLX-8bit	35.100000	keep	baseline prompt_type=summarization max_tokens=512 num_trials=5 temperature=0.0 stream=True
b2c3d4e	unsloth/Qwen3.6-27B-MLX-8bit	18.400000	keep	baseline prompt_type=summarization max_tokens=512 num_trials=5 temperature=0.0 stream=True
c3d4e5f	unsloth/Qwen3.6-35B-A3B-MLX-8bit	80.200000	keep	prompt_type=qa max_tokens=1024 num_trials=5 temperature=0.0 stream=True
c3d4e5f	unsloth/Qwen3.6-27B-MLX-8bit	20.800000	keep	prompt_type=qa max_tokens=1024 num_trials=5 temperature=0.0 stream=True
d4e5f6g	unsloth/Qwen3.6-35B-A3B-MLX-8bit	0.000000	crash	prompt_type=qa max_tokens=512 num_trials=5 temperature=0.0 stream=True draft_model=qwen-0.5B
d4e5f6g	unsloth/Qwen3.6-27B-MLX-8bit	0.000000	crash	prompt_type=qa max_tokens=512 num_trials=5 temperature=0.0 stream=True draft_model=qwen-0.5B
```

## Progress log (for human monitoring)

Append **one line per model per invocation** to `progress.log` (so 2 lines per invocation) so the human can `tail -f progress.log` in another terminal:

```
[YYYY-MM-DD HH:MM:SS] run=<n> commit=<short> model=<shortname> metric=<value> status=<keep|crash> | <description>
```

`run` increments once per invocation (both model lines share the same `run=N`). Emit the Model A line, then the Model B line.

Do **not** commit `progress.log` — keep it untracked like `results.tsv`. Append with `>>`, never rewrite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr21-mlx-sweep`).

LOOP UNTIL `MAX_RUNS` (100) INVOCATIONS COMPLETED:

1. Look at git state: the current branch/commit.
2. Count invocations so far (**unique commits** in `results.tsv` minus header — remember each invocation adds 2 rows with the same commit). If ≥ 100, stop.
3. Modify `experiment.py` with one experimental idea (one knob change).
4. `git commit`.
5. Run: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT tee).
6. Parse the results: `grep "^RESULT " run.log` yields **two** lines (one per model). Extract `e2e_tokens_per_second` for each, keyed by `model=`.
7. If fewer than 2 `RESULT` lines → one or both models crashed. `tail -n 80 run.log` for the trace. Trivial fix → fix and retry. Fundamentally broken → log `crash` rows for the affected model(s) and move on.
8. Append **two lines** to `progress.log` (one per model) — see "Progress log" section.
9. Append **two rows** to `results.tsv` (one per model, same commit) — see "Logging results" section.
10. **Do NOT git-reset.** Every valid datapoint stays, regardless of whether the number went up or down vs. the previous run.

**Never run trials in parallel** — always sequential.

**Timeout**: each invocation ≈ `TIME_BUDGET` seconds (240s) for both models combined. If a run exceeds 2× budget, kill it and treat as failure.

**Crashes**: trivial fixes → fix and re-run; fundamentally broken ideas → log `crash` and move on.

**STOPPING**: when 100 invocations are logged, stop and tell the human you're done. This is exploration with a hard cap — do not loop forever.
