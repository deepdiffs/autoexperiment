# autoexperiment — Qwen bf16 vs q8 head-to-head benchmark

Comparative experiment: on every single run, test the SAME config against **both** Qwen models back-to-back and log both results. The agent's job is to sweep through different configurations and collect comparison data — not to optimize a single number.

## Domain context

You are benchmarking **end-to-end tokens/sec** (`tokens_per_second` = `completion_tokens / total_wall_time`) of two Qwen 3.6 35B variants served locally on this Mac Studio:

- `qwen3.6-35b-bf16` — full-precision bfloat16 weights
- `qwen3.6-35b-q8` — 8-bit quantized weights

They are hosted by a **`llama.cpp` server running directly on this machine** at `http://localhost:8080` (OpenAI-compatible `/v1` API). The backend is **Apple Silicon + Metal**. You interact with it exclusively via HTTP — you cannot change server-side config (no SSH, no restart, no llama-server launch flags).

The server is already running with both models loaded. No authentication is required. Override the base URL with `LLAMA_BASE_URL` if needed; otherwise it defaults to `http://localhost:8080/v1`.

The short names for the two models are defined in `harness.py` as `MODEL_QWEN_BF16` and `MODEL_QWEN_Q8`; import and use those constants from `experiment.py` rather than hard-coding the full IDs.

**Both models are reasoning models.** Each response emits reasoning tokens (reported in `usage.completion_tokens_details.reasoning_tokens`) in addition to visible content. `completion_tokens` covers both. The primary `tokens_per_second` metric is total completion tokens per wall-clock second. `decode_tok_s` is also logged but is only meaningful when the stream isn't buffered.

## Head-to-head paradigm

**Every invocation of `experiment.py` runs the exact same request against both models, in order: `MODEL_QWEN_BF16` first, then `MODEL_QWEN_Q8`.** One warmup + N timed trials per model. This produces **two result blocks** — one per model — with identical prompt, max_tokens, trials, temperature, etc. You change the config knobs; the framework guarantees both models see them.

This is a **comparison** experiment, not an optimization. The agent's job is to sweep configurations and log what each model does. Do **not** maintain a "best so far" score or git-reset on regressions — every data point is valuable comparison data.

## Setup

1. **Agree on a run tag**: propose a tag that uniquely identifies this run. Pick one of:
   - `<date>-<HHMM>` — e.g. `apr21-1430`. Always unique, good default.
   - `<date>-<ordinal>` — e.g. `apr21-3`. Check existing branches with `git branch --list 'autoexperiment/<date>-*'` and increment.
   - `<date>-<topic>` — e.g. `apr21-qwen-headtohead`. Use when the run has a clear focus.

   Verify the branch does not already exist: `git show-ref --verify --quiet refs/heads/autoexperiment/<tag>` must return non-zero.
2. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `harness.py` — fixed configuration (metric, goal, time budget, model IDs, `MAX_RUNS`) and utilities. Do not modify.
   - `experiment.py` — the file you modify.
4. **Verify connectivity**: `uv run harness.py` lists the models available at the endpoint. Confirm both `qwen3.6-35b-bf16` and `qwen3.6-35b-q8` appear and are marked `loaded`.
5. **Initialize logs fresh**:
   - `results.tsv` — create with just the header row.
   - `progress.log` — truncate (delete or `: > progress.log`).
6. **Confirm and go**.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py`. Tune `PROMPT_TYPE`, `MAX_TOKENS`, `NUM_TRIALS`, `TEMPERATURE`, `TOP_K`, `STREAM`, prompt texts, and add new request-body fields to sweep.

**What you CANNOT do:**
- Change the `MODELS` list in `experiment.py` — both models must run on every invocation, in the canonical order (`MODEL_QWEN_BF16` first, then `MODEL_QWEN_Q8`). Skipping a model defeats the purpose of the experiment.
- Modify `harness.py` (read-only).
- Install new packages or add dependencies. Only what's in `pyproject.toml`.
- Issue parallel/concurrent requests. Trials must be strictly sequential, and the two models must run sequentially (never overlap).
- Call any endpoint other than `/v1/chat/completions`.
- Change server-side config. You only control the HTTP request body.
- Return fewer than 50 completion tokens per trial. The experiment raises on this (invalid run → crash).

**The goal**: collect a matrix of (config, model) → `tokens_per_second` measurements across a wide variety of configs, so the user can compare how bf16 vs q8 respond to the same inputs. `METRIC_NAME` / `METRIC_GOAL` still read "maximize tokens_per_second" for plumbing purposes, but you are not optimizing — you are sampling config space.

**Hard cap**: stop after `MAX_RUNS` invocations of `experiment.py` (100). That yields 100 bf16 rows + 100 q8 rows = 200 total rows in `results.tsv`. Check the number of distinct `experiment.py` invocations already logged (e.g. `grep -c "^RESULT model=qwen3.6-35b-bf16" progress.log` or count unique commits in `results.tsv`) before each new run.

## Baseline phase

Before exploring, establish **3 baseline invocations** — one per `PROMPT_TYPE`, using the defaults otherwise. Each invocation logs two rows (one per model):

| PROMPT_TYPE    | rows logged                              |
|----------------|------------------------------------------|
| summarization  | 1× bf16 + 1× q8                          |
| open_ended     | 1× bf16 + 1× q8                          |
| qa             | 1× bf16 + 1× q8                          |

That's 3 invocations and 6 TSV rows, counting toward the 100-invocation cap.

Tag each baseline row's description clearly, e.g. `baseline qwen3.6-35b-bf16 qa` and `baseline qwen3.6-35b-q8 qa`.

## Strategy guidance (after baseline)

**Try:**
- `MAX_TOKENS` sweeps (128 / 256 / 512 / 1024 / 2048) — how does decode throughput scale with generation length, and does the relative gap between bf16 and q8 widen or narrow?
- Prompt-length sweeps (short / medium / long) — bf16 vs q8 under prefill-dominated vs decode-dominated regimes.
- Prompt content variation — code, prose, math, multilingual. Reasoning-token ratio likely differs between models.
- `TEMPERATURE` (0.0 vs > 0) and `TOP_K` variations — sampling overhead differences.
- `STREAM` True vs False.
- Repeated identical requests to probe prefix-cache behavior.
- One variable changed per run, so per-model deltas are attributable.
- Keep at least 1 warmup + 3 timed trials. 5 is the default and usually fine.

**Avoid:**
- Concurrency / `asyncio.gather` / threads firing multiple requests at once.
- Running only one of the two models in a given invocation.
- Non-chat endpoints (`/v1/completions`, `/v1/embeddings`, custom routes).
- Any attempt to reconfigure the server.
- Obviously-bad knobs (e.g. `MAX_TOKENS` absurdly high, extreme temperatures) — don't burn budget confirming the obvious.
- Changing `harness.py` or adding dependencies.

**Tips:**
- If trial-to-trial variance exceeds ~20% of the mean for either model, flag it in the description.
- Switching between the two models may trigger a slot reload on the llama.cpp server. The per-model warmup trial inside `run_model()` absorbs that; don't be surprised if the bf16 → q8 boundary has a first-trial prefill spike.
- bf16 is ~2× the memory footprint of q8 on disk/VRAM and correspondingly slower per token. Expect q8 to win on raw tok/s; the interesting question is *by how much, under which config*.

## Output format

For each invocation, `experiment.py` now emits **two** `---` blocks (one per model), followed by two `RESULT` summary lines and a final `time_seconds` line. Example:

```
---
tokens_per_second: 18.500000
decode_tok_s: 19.200000
...
model: qwen3.6-35b-bf16
prompt_type: qa
max_tokens: 512
---
tokens_per_second: 42.300000
decode_tok_s: 44.100000
...
model: qwen3.6-35b-q8
prompt_type: qa
max_tokens: 512

RESULT model=qwen3.6-35b-bf16 tokens_per_second=18.500000
RESULT model=qwen3.6-35b-q8 tokens_per_second=42.300000
time_seconds: 103.400000
```

Extract the two metrics with:

```
grep "^RESULT " run.log
```

Each `RESULT` line has `model=<name>` and `tokens_per_second=<value>`. Parse both before logging.

## Logging results

Log every experiment to `results.tsv` (tab-separated — commas break in descriptions).

Header + 4 columns (unchanged — model name lives in the description for both ease-of-reading and viz.py compatibility):

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars) — same commit for both rows of a single invocation.
2. `tokens_per_second` value (0.000000 for crashes).
3. status: `keep` for any valid measurement, `crash` for failures. **Do not use `discard`** — this is a comparison experiment, not an optimization.
4. short text description. **Must begin with the full model name** so downstream tools can identify the row, e.g. `qwen3.6-35b-bf16 baseline qa`, `qwen3.6-35b-q8 max_tokens=1024 open_ended`.

Each `experiment.py` invocation appends **two consecutive rows** — bf16 first, then q8, both sharing the same commit hash.

Example:

```
commit	metric_value	status	description
a1b2c3d	18.500000	keep	qwen3.6-35b-bf16 baseline qa
a1b2c3d	42.300000	keep	qwen3.6-35b-q8 baseline qa
b2c3d4e	17.800000	keep	qwen3.6-35b-bf16 baseline summarization
b2c3d4e	41.100000	keep	qwen3.6-35b-q8 baseline summarization
c3d4e5f	14.200000	keep	qwen3.6-35b-bf16 max_tokens=1024 qa
c3d4e5f	36.400000	keep	qwen3.6-35b-q8 max_tokens=1024 qa
d4e5f6g	0.000000	crash	bad request body, 400 from server (both models)
```

If the invocation crashes before either model produces a result, log one `crash` row with no model prefix (or both — your call; just don't pretend it succeeded). If one model succeeds and the other crashes, log the success row(s) for the one that worked and a `crash` row for the one that didn't, both at the same commit.

## Progress log (for human monitoring)

Also append one line **per model result** (so two lines per invocation) to `progress.log` so the human can `tail -f progress.log` in another terminal. Format:

```
[YYYY-MM-DD HH:MM:SS] run=<n> commit=<short> metric=<value> status=<keep|crash> | <description>
```

`run` is incremented once per *row* (two rows per invocation → two consecutive run numbers per invocation). `description` must start with the model name so `viz.py` can categorize.

Example:

```
[2026-04-21 14:23:01] run=1 commit=a1b2c3d metric=18.500000 status=keep | qwen3.6-35b-bf16 baseline qa
[2026-04-21 14:23:01] run=2 commit=a1b2c3d metric=42.300000 status=keep | qwen3.6-35b-q8 baseline qa
[2026-04-21 14:25:44] run=3 commit=b2c3d4e metric=17.800000 status=keep | qwen3.6-35b-bf16 baseline summarization
[2026-04-21 14:25:44] run=4 commit=b2c3d4e metric=41.100000 status=keep | qwen3.6-35b-q8 baseline summarization
```

Do **not** commit `progress.log` — keep it untracked like `results.tsv`. Use UTC or local time consistently. Append with `>>`, never rewrite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr21-qwen-headtohead`).

LOOP UNTIL `MAX_RUNS` (100) INVOCATIONS COMPLETED:

1. Look at git state: the current branch/commit.
2. Count invocations so far (e.g. unique commits in `results.tsv` minus header, or `grep -c "^RESULT model=qwen3.6-35b-bf16" ...`). If ≥ 100, stop.
3. Modify `experiment.py` with one experimental idea (one knob change).
4. `git commit`.
5. Run: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT tee).
6. Parse results: `grep "^RESULT " run.log` yields two lines (bf16 and q8). Extract each model's `tokens_per_second`.
7. If both `RESULT` lines are missing → invocation crashed. `tail -n 50 run.log` for the trace. If it's something trivial (typo, missing key), fix and retry. If fundamentally broken, log `crash` and move on.
8. Append two lines to `progress.log` (one per model) — see "Progress log" section.
9. Append two lines to `results.tsv` (one per model) — see "Logging results" section.
10. **Do NOT git-reset.** This is a comparison experiment — every valid data point stays. Keep the commit regardless of whether the numbers went up or down.

**Never run both models in parallel** — always sequential, and always in the canonical order inside `experiment.py`.

**Timeout**: each run ≈ `TIME_BUDGET` seconds (240s for two models) + overhead. If a run exceeds 2× budget, kill it and treat as failure.

**Crashes**: trivial fixes → fix and re-run; fundamentally broken ideas → log `crash` and move on.

**STOPPING**: when 100 invocations are logged, stop and tell the human you're done. Unlike the optimization variant of this template, this experiment has a hard cap — do not loop forever.
