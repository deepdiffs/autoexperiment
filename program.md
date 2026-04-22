# autoexperiment — MLX Qwen3.6-35B-A3B-8bit throughput exploration

Sweep generation configurations against a single MLX-served Qwen model and log throughput / latency metrics across the config space. The agent's job is to *characterize* how the model responds to different inputs and knobs — not to climb a single number. Every valid datapoint is a keep.

## Domain context

You are benchmarking a single MLX model:

- **Model**: `unsloth/Qwen3.6-35B-A3B-MLX-8bit` (8-bit MLX quantization of Qwen 3.6 35B-A3B reasoning model)
- **Serving**: `mlx_lm.server` running locally on this Mac Studio (Apple Silicon + Metal), exposing an OpenAI-compatible `/v1` API.
- **Default endpoint**: `http://localhost:8000/v1`. Override with `MLX_BASE_URL`.

The server is **already running**, started manually by the human in another terminal:

```
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-MLX-8bit --port 8000
```

You interact with it exclusively via HTTP. You cannot restart it, change launch flags, or change server-side config — only the request body.

The model id is defined in `harness.py` as `MODEL_ID`; import and use that constant from `experiment.py` rather than hard-coding the string. Because the model stays loaded in the persistent server, there is no per-invocation load overhead — the warmup trial only needs to absorb prompt-cache priming, not weight loading.

**Qwen 3.6 is a reasoning model.** mlx_lm.server (≥0.31) streams reasoning tokens under `delta.reasoning` and visible answer tokens under `delta.content`. `usage.completion_tokens` covers both. The harness counts a chunk in either field as a "token" for TTFT and decode-rate purposes, since the user feels reasoning streaming the same way they feel content streaming. `ttft_content_ms` is reported separately so you can see when the model finally exits the reasoning phase. At modest `max_tokens` (e.g. 512) on a "step by step" prompt, the model frequently never reaches `content` at all — that's normal; flag it in the description but don't treat it as a crash.

## Metrics

Per invocation, `experiment.py` runs 1 warmup + `NUM_TRIALS` timed trials of the same chat completion and reports trim-mean aggregates:

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
- Change `MODEL_ID` — the model is fixed at `unsloth/Qwen3.6-35B-A3B-MLX-8bit`.
- Modify `harness.py` (read-only).
- Install new packages or add dependencies. Only what's already in `pyproject.toml`.
- Issue parallel/concurrent requests. Trials must be strictly sequential.
- Call any endpoint other than `/v1/chat/completions` or `/v1/models`.
- Restart or reconfigure the mlx_lm.server.
- Return fewer than 50 completion tokens per trial. The experiment raises on this (invalid run → crash).
- Skip the warmup trial — it absorbs prompt-cache priming and any first-call cost on this configuration.

**The goal**: collect a matrix of (config) → throughput/latency measurements across a wide variety of configs, so the user can characterize how this model behaves under different inputs.

**Hard cap**: stop after `MAX_RUNS` invocations of `experiment.py` (100). Check the count of unique commits in `results.tsv` (minus header) before each new run.

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
- Changing `harness.py` or `MODEL_ID`, or adding dependencies.
- `DRAFT_MODEL` unless the human has confirmed a draft model is available locally — it will likely 4xx if the draft isn't loadable.

**Tips:**
- If trial-to-trial variance exceeds ~20% of the mean, flag it in the description.
- The first trial of a brand-new prompt may show a prefill spike due to no prompt-cache hit. The warmup trial inside `experiment.py` absorbs this.
- 8-bit MLX on M-series ASi typically lands in the tens-of-tok/s range for 35B-A3B; treat anything wildly outside that as suspicious.

## Output format

Each invocation emits one `---` block, one `RESULT` line, and one `time_seconds` line. Example:

```
---
e2e_tokens_per_second: 38.500000
decode_tok_s: 41.200000
prefill_tok_s: 312.500000
ttft_ms: 248.000000
total_s_per_trial: 13.300000
prompt_tokens: 78
completion_tokens_avg: 512.000000
trials_completed: 5
model: unsloth/Qwen3.6-35B-A3B-MLX-8bit
prompt_type: qa
max_tokens: 512
stream: True

RESULT e2e_tokens_per_second=38.500000
time_seconds: 71.400000
```

Extract the metric with:

```
grep "^RESULT " run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated — commas break in descriptions).

Header + 4 columns:

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars).
2. `e2e_tokens_per_second` value (0.000000 for crashes).
3. status: `keep` for any valid measurement, `crash` for failures. **Do not use `discard`** — exploration, not optimization.
4. short text description capturing the variant, e.g. `baseline qa`, `max_tokens=1024 qa`, `stream=False qa`, `temp=0.7 top_p=0.9 open_ended`.

Example:

```
commit	metric_value	status	description
a1b2c3d	38.500000	keep	baseline qa
b2c3d4e	35.100000	keep	baseline summarization
c3d4e5f	30.400000	keep	max_tokens=1024 qa
d4e5f6g	0.000000	crash	bad request body, 400 from server
```

## Progress log (for human monitoring)

Append one line per invocation to `progress.log` so the human can `tail -f progress.log` in another terminal:

```
[YYYY-MM-DD HH:MM:SS] run=<n> commit=<short> metric=<value> status=<keep|crash> | <description>
```

`run` increments once per row.

Do **not** commit `progress.log` — keep it untracked like `results.tsv`. Append with `>>`, never rewrite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr21-mlx-sweep`).

LOOP UNTIL `MAX_RUNS` (100) INVOCATIONS COMPLETED:

1. Look at git state: the current branch/commit.
2. Count invocations so far (unique commits in `results.tsv` minus header). If ≥ 100, stop.
3. Modify `experiment.py` with one experimental idea (one knob change).
4. `git commit`.
5. Run: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT tee).
6. Parse the result: `grep "^RESULT " run.log` yields one line. Extract `e2e_tokens_per_second`.
7. If the `RESULT` line is missing → invocation crashed. `tail -n 50 run.log` for the trace. Trivial fix → fix and retry. Fundamentally broken → log `crash` and move on.
8. Append one line to `progress.log` — see "Progress log" section.
9. Append one line to `results.tsv` — see "Logging results" section.
10. **Do NOT git-reset.** Every valid datapoint stays, regardless of whether the number went up or down vs. the previous run.

**Never run trials in parallel** — always sequential.

**Timeout**: each run ≈ `TIME_BUDGET` seconds (120s) + warmup overhead. If a run exceeds 2× budget, kill it and treat as failure.

**Crashes**: trivial fixes → fix and re-run; fundamentally broken ideas → log `crash` and move on.

**STOPPING**: when 100 invocations are logged, stop and tell the human you're done. This is exploration with a hard cap — do not loop forever.
