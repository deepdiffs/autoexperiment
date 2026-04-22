# autoexperiment — LMStudio 4-way head-to-head benchmark

Comparative experiment: on every invocation, test the SAME config against **four** Qwen models back-to-back and log four result blocks. The agent's job is to sweep through different configurations and collect comparison data — not to optimize a single number.

## Domain context

You are benchmarking **end-to-end tokens/sec** (`tokens_per_second` = `completion_tokens / total_wall_time`) of four Qwen 3.6 variants served locally by **LMStudio** on this Mac Studio. The 2×2 matrix is architecture × backend:

|                 | GGUF Q8_K_XL (llama.cpp backend) | MLX 8-bit                |
|-----------------|----------------------------------|--------------------------|
| 27B dense       | `qwen3.6-27b@q8_k_xl`            | `qwen3.6-27b-mlx`        |
| 35B A3B MoE     | `qwen3.6-35b-a3b@q8_k_xl`        | `qwen3.6-35b-a3b-mlx`    |

They are hosted by an **LMStudio server running directly on this machine** at `http://localhost:1234` (OpenAI-compatible `/v1` API). You interact with it exclusively via HTTP — you cannot change server-side config (no model reloads, no launch flags).

Override the base URL with `LMSTUDIO_BASE_URL` if needed; otherwise it defaults to `http://localhost:1234/v1`. The short names are defined in `harness.py` as `MODEL_27B_GGUF`, `MODEL_27B_MLX`, `MODEL_35B_GGUF`, `MODEL_35B_MLX`; import and use those constants from `experiment.py` rather than hard-coding the full IDs.

**LMStudio JIT-loads models on first request.** A model listed as `not-loaded` in `/api/v0/models` is still callable — the first request loads it from disk (slow, ~10–30s for 35B weights). The per-model warmup trial inside `run_model()` absorbs this. Only one (or a few) models stay resident at a time depending on VRAM; expect load events when switching between the four models.

**All four models are reasoning models.** Each response emits reasoning tokens (reported in `usage.completion_tokens_details.reasoning_tokens`) in addition to visible content. `completion_tokens` covers both. The primary `tokens_per_second` metric is total completion tokens per wall-clock second. `decode_tok_s` is also logged but is only meaningful when the stream isn't buffered.

## Head-to-head paradigm

**Every invocation of `experiment.py` runs the exact same request against all four models, in canonical order:** `MODEL_27B_GGUF` → `MODEL_27B_MLX` → `MODEL_35B_GGUF` → `MODEL_35B_MLX`. One warmup + N timed trials per model. This produces **four result blocks** — one per model — with identical prompt, max_tokens, trials, temperature, etc. You change the config knobs; the framework guarantees every model sees them.

This is a **comparison** experiment, not an optimization. The agent's job is to sweep configurations and log what each model does. Do **not** maintain a "best so far" score or git-reset on regressions — every data point is valuable comparison data.

## Setup

1. **Agree on a run tag**: propose a tag that uniquely identifies this run. Pick one of:
   - `<date>-<HHMM>` — e.g. `apr22-1430`. Always unique, good default.
   - `<date>-<ordinal>` — e.g. `apr22-3`. Check existing branches with `git branch --list 'autoexperiment/<date>-*'` and increment.
   - `<date>-<topic>` — e.g. `apr22-lmstudio-4way`. Use when the run has a clear focus.

   Verify the branch does not already exist: `git show-ref --verify --quiet refs/heads/autoexperiment/<tag>` must return non-zero.
2. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `harness.py` — fixed configuration (metric, goal, time budget, model IDs, `MAX_RUNS`) and utilities. Do not modify.
   - `experiment.py` — the file you modify.
4. **Verify connectivity**: `uv run harness.py` lists the models exposed by LMStudio's `/v1/models`. Confirm all four — `qwen3.6-27b@q8_k_xl`, `qwen3.6-27b-mlx`, `qwen3.6-35b-a3b@q8_k_xl`, `qwen3.6-35b-a3b-mlx` — appear.
5. **Initialize logs fresh**:
   - `results.tsv` — create with just the header row.
   - `progress.log` — truncate (delete or `: > progress.log`).
6. **Confirm and go**.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py`. Tune `PROMPT_TYPE`, `MAX_TOKENS`, `NUM_TRIALS`, `TEMPERATURE`, `TOP_K`, `STREAM`, prompt texts, and add new request-body fields to sweep.

**What you CANNOT do:**
- Change the `MODELS` list in `experiment.py` — all four models must run on every invocation, in the canonical order (`MODEL_27B_GGUF` → `MODEL_27B_MLX` → `MODEL_35B_GGUF` → `MODEL_35B_MLX`). Skipping or reordering defeats the purpose of the experiment.
- Modify `harness.py` (read-only).
- Install new packages or add dependencies. Only what's in `pyproject.toml`.
- Issue parallel/concurrent requests. Trials must be strictly sequential, and the four models must run sequentially (never overlap).
- Call any endpoint other than `/v1/chat/completions`.
- Change server-side config. You only control the HTTP request body.
- Return fewer than 50 completion tokens per trial. The experiment raises on this (invalid run → crash).

**The goal**: collect a matrix of (config, model) → `tokens_per_second` measurements across a wide variety of configs, so the user can compare how the four combinations of {size × backend} respond to the same inputs. `METRIC_NAME` / `METRIC_GOAL` still read "maximize tokens_per_second" for plumbing purposes, but you are not optimizing — you are sampling config space.

**Hard cap**: stop after `MAX_RUNS` invocations of `experiment.py` (25). That yields 25 rows per model × 4 models = 100 total rows in `results.tsv`. Check the number of distinct `experiment.py` invocations already logged (e.g. `grep -c "^RESULT model=qwen3.6-27b@q8_k_xl" progress.log` or count unique commits in `results.tsv`) before each new run.

## Baseline phase

Before exploring, establish **3 baseline invocations** — one per `PROMPT_TYPE`, using the defaults otherwise. Each invocation logs four rows (one per model):

| PROMPT_TYPE    | rows logged                                                             |
|----------------|-------------------------------------------------------------------------|
| summarization  | 1× 27B-gguf + 1× 27B-mlx + 1× 35B-gguf + 1× 35B-mlx                     |
| open_ended     | 1× 27B-gguf + 1× 27B-mlx + 1× 35B-gguf + 1× 35B-mlx                     |
| qa             | 1× 27B-gguf + 1× 27B-mlx + 1× 35B-gguf + 1× 35B-mlx                     |

That's 3 invocations and 12 TSV rows, counting toward the 25-invocation cap.

Tag each baseline row's description clearly, e.g. `qwen3.6-27b@q8_k_xl baseline qa`, `qwen3.6-35b-a3b-mlx baseline summarization`.

## Strategy guidance (after baseline)

**Try:**
- `MAX_TOKENS` sweeps (128 / 256 / 512 / 1024 / 2048) — how does decode throughput scale with generation length, and does the relative gap widen or narrow across the 2×2 matrix?
- Prompt-length sweeps (short / medium / long) — prefill- vs decode-dominated regimes, per backend per size.
- Prompt content variation — code, prose, math, multilingual. Reasoning-token ratio likely differs across models.
- `TEMPERATURE` (0.0 vs > 0) and `TOP_K` variations — sampling overhead differences between backends.
- `STREAM` True vs False — does one backend benefit more from streaming?
- Repeated identical requests to probe prefix-cache behavior (GGUF has strong prefix-cache; MLX may differ).
- One variable changed per run, so per-model deltas are attributable.
- Keep at least 1 warmup + 3 timed trials. 5 is the default and usually fine.

**Avoid:**
- Concurrency / `asyncio.gather` / threads firing multiple requests at once.
- Running fewer than all four models in a given invocation.
- Non-chat endpoints (`/v1/completions`, `/v1/embeddings`, custom routes).
- Any attempt to reconfigure the LMStudio server.
- Obviously-bad knobs (e.g. `MAX_TOKENS` absurdly high, extreme temperatures) — don't burn budget confirming the obvious.
- Changing `harness.py` or adding dependencies.

**Tips:**
- If trial-to-trial variance exceeds ~20% of the mean for any model, flag it in the description.
- Switching between models triggers JIT load/unload in LMStudio. The per-model warmup trial inside `run_model()` absorbs that; don't be surprised if a model's first trial shows an elevated TTFT.
- GGUF Q8 and MLX 8-bit are roughly comparable in memory footprint; expect their throughputs to be closer than bf16 vs q8 on the same backend. The interesting questions are *does backend choice matter more than size?* and *do MoE and dense architectures respond differently to the two backends?*

## Output format

For each invocation, `experiment.py` emits **four** `---` blocks (one per model), followed by four `RESULT` summary lines and a final `time_seconds` line. Example:

```
---
tokens_per_second: 38.100000
decode_tok_s: 40.500000
...
model: qwen3.6-27b@q8_k_xl
prompt_type: qa
max_tokens: 512
---
tokens_per_second: 55.400000
...
model: qwen3.6-27b-mlx
---
tokens_per_second: 19.800000
...
model: qwen3.6-35b-a3b@q8_k_xl
---
tokens_per_second: 41.200000
...
model: qwen3.6-35b-a3b-mlx

RESULT model=qwen3.6-27b@q8_k_xl tokens_per_second=38.100000
RESULT model=qwen3.6-27b-mlx tokens_per_second=55.400000
RESULT model=qwen3.6-35b-a3b@q8_k_xl tokens_per_second=19.800000
RESULT model=qwen3.6-35b-a3b-mlx tokens_per_second=41.200000
time_seconds: 612.400000
```

Extract the four metrics with:

```
grep "^RESULT " run.log
```

Each `RESULT` line has `model=<name>` and `tokens_per_second=<value>`. Parse all four before logging.

## Logging results

Log every experiment to `results.tsv` (tab-separated — commas break in descriptions).

Header + 4 columns (unchanged — model name lives in the description for both ease-of-reading and viz.py compatibility):

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars) — same commit for all four rows of a single invocation.
2. `tokens_per_second` value (0.000000 for crashes).
3. status: `keep` for any valid measurement, `crash` for failures. **Do not use `discard`** — this is a comparison experiment, not an optimization.
4. short text description. **Must begin with the full model name** so downstream tools can identify the row, e.g. `qwen3.6-27b@q8_k_xl baseline qa`, `qwen3.6-35b-a3b-mlx max_tokens=1024 open_ended`.

Each `experiment.py` invocation appends **four consecutive rows** in canonical order, all sharing the same commit hash.

Example:

```
commit	metric_value	status	description
a1b2c3d	38.100000	keep	qwen3.6-27b@q8_k_xl baseline qa
a1b2c3d	55.400000	keep	qwen3.6-27b-mlx baseline qa
a1b2c3d	19.800000	keep	qwen3.6-35b-a3b@q8_k_xl baseline qa
a1b2c3d	41.200000	keep	qwen3.6-35b-a3b-mlx baseline qa
b2c3d4e	36.200000	keep	qwen3.6-27b@q8_k_xl baseline summarization
b2c3d4e	52.800000	keep	qwen3.6-27b-mlx baseline summarization
b2c3d4e	18.400000	keep	qwen3.6-35b-a3b@q8_k_xl baseline summarization
b2c3d4e	39.600000	keep	qwen3.6-35b-a3b-mlx baseline summarization
c3d4e5f	0.000000	crash	bad request body, 400 from server (all models)
```

If the invocation crashes before any model produces a result, log a single `crash` row with no model prefix. If some models succeed and others crash, log the success rows for those that worked and a `crash` row for each that didn't, all at the same commit.

## Progress log (for human monitoring)

Also append one line **per model result** (so four lines per invocation) to `progress.log` so the human can `tail -f progress.log` in another terminal. Format:

```
[YYYY-MM-DD HH:MM:SS] run=<n> commit=<short> metric=<value> status=<keep|crash> | <description>
```

`run` is incremented once per *row* (four rows per invocation → four consecutive run numbers per invocation). `description` must start with the model name so `viz.py` can categorize.

Example:

```
[2026-04-22 14:23:01] run=1 commit=a1b2c3d metric=38.100000 status=keep | qwen3.6-27b@q8_k_xl baseline qa
[2026-04-22 14:23:01] run=2 commit=a1b2c3d metric=55.400000 status=keep | qwen3.6-27b-mlx baseline qa
[2026-04-22 14:23:01] run=3 commit=a1b2c3d metric=19.800000 status=keep | qwen3.6-35b-a3b@q8_k_xl baseline qa
[2026-04-22 14:23:01] run=4 commit=a1b2c3d metric=41.200000 status=keep | qwen3.6-35b-a3b-mlx baseline qa
```

Do **not** commit `progress.log` — keep it untracked like `results.tsv`. Use UTC or local time consistently. Append with `>>`, never rewrite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr22-lmstudio-4way`).

LOOP UNTIL `MAX_RUNS` (25) INVOCATIONS COMPLETED:

1. Look at git state: the current branch/commit.
2. Count invocations so far (e.g. unique commits in `results.tsv` minus header, or `grep -c "^RESULT model=qwen3.6-27b@q8_k_xl" ...`). If ≥ 25, stop.
3. Modify `experiment.py` with one experimental idea (one knob change).
4. `git commit`.
5. Run: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT tee).
6. Parse results: `grep "^RESULT " run.log` yields four lines (one per model). Extract each model's `tokens_per_second`.
7. If any `RESULT` lines are missing → partial failure. `tail -n 50 run.log` for the trace. If it's something trivial (typo, missing key), fix and retry. If fundamentally broken, log `crash` rows for the missing models and move on.
8. Append four lines to `progress.log` (one per model) — see "Progress log" section.
9. Append four lines to `results.tsv` (one per model) — see "Logging results" section.
10. **Do NOT git-reset.** This is a comparison experiment — every valid data point stays. Keep the commit regardless of whether the numbers went up or down.

**Never run the four models in parallel** — always sequential, and always in the canonical order inside `experiment.py`.

**Timeout**: each run ≈ `TIME_BUDGET` seconds (600s for four models) + overhead. If a run exceeds 2× budget (1200s), kill it and treat as failure.

**Crashes**: trivial fixes → fix and re-run; fundamentally broken ideas → log `crash` and move on.

**STOPPING**: when 25 invocations are logged, stop and tell the human you're done. Unlike the optimization variant of this template, this experiment has a hard cap — do not loop forever.
