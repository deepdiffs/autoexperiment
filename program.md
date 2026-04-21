# autoexperiment — llama.cpp tokens/sec benchmark

Autonomous experiment optimization. An AI agent iterates on code, runs experiments, and keeps what works.

## Domain context

You are benchmarking **end-to-end tokens/sec** (`tokens_per_second` = `completion_tokens / total_wall_time`) of two Gemma models served remotely:

- `gemma4-26b`
- `supergemma4-26b`

They are hosted by a `llama.cpp` server fronted by a `LiteLLM` OpenAI-compatible proxy at the URL in `$LITELLM_BASE_URL`. The backend runs on **Apple Silicon + Metal**. You interact with it exclusively via HTTP — you cannot change server-side config (no SSH, no restart, no llama-server launch flags).

The server is already running with both models loaded. Authentication uses `$LITELLM_MASTER_KEY`, sent as both `x-api-key` and `Authorization: Bearer`.

**Both models are reasoning models.** Each response emits reasoning tokens (reported in `usage.completion_tokens_details.reasoning_tokens`) in addition to visible content. `completion_tokens` covers both. The primary `tokens_per_second` metric is total completion tokens per wall-clock second — this is what a user actually waits for. `decode_tok_s` is also logged but is only meaningful when the proxy doesn't buffer the stream (frequently it buffers tiny completions, in which case `decode_tok_s` will be 0.0 — don't chase that).

## Setup

1. **Agree on a run tag**: propose a tag that uniquely identifies this run (multiple runs happen per day, so date alone is not enough). Pick **one** of these formats:
   - `<date>-<HHMM>` — e.g. `apr21-1430`. Always unique, good default.
   - `<date>-<ordinal>` — e.g. `apr21-3`. Check existing branches with `git branch --list 'autoexperiment/<date>-*'` and increment.
   - `<date>-<topic>` — e.g. `apr21-maxtok-sweep`. Use when the run has a clear focus.

   Verify the branch does not already exist: `git show-ref --verify --quiet refs/heads/autoexperiment/<tag>` must return non-zero. If it exists, pick a new tag.
2. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `harness.py` — fixed configuration (metric, goal, time budget) and utilities. Do not modify.
   - `experiment.py` — the file you modify.
4. **Verify connectivity**: `uv run harness.py` lists the models available at the endpoint. Confirm both `gemma4-26b` and `supergemma4-26b` appear.
5. **Initialize results.tsv**: create `results.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py`. Tune `MODEL`, `PROMPT_TYPE`, `MAX_TOKENS`, `NUM_TRIALS`, `TEMPERATURE`, the prompt texts, and add new dimensions to sweep.

**What you CANNOT do:**
- Modify `harness.py` (read-only).
- Install new packages or add dependencies. Only what's in `pyproject.toml`.
- Issue parallel/concurrent requests. Trials must be strictly sequential.
- Call any endpoint other than `/v1/chat/completions`.
- Change server-side config. You only control the HTTP request body.
- Return fewer than 50 completion tokens per trial. The experiment raises on this (invalid run → crash).

**The goal**: maximize `tokens_per_second` (decode tokens/sec). Read `METRIC_NAME` and `METRIC_GOAL` in `harness.py` to confirm.

**Simplicity criterion**: all else being equal, simpler is better. A small tok/s improvement that adds ugly complexity is not worth it.

## Baseline phase

Before exploring, establish **6 baseline runs** — one per (model × prompt_type) cell:

| MODEL             | PROMPT_TYPE    |
|-------------------|----------------|
| gemma4-26b        | summarization  |
| gemma4-26b        | open_ended     |
| gemma4-26b        | qa             |
| supergemma4-26b   | summarization  |
| supergemma4-26b   | open_ended     |
| supergemma4-26b   | qa             |

The default `experiment.py` is pre-set to `gemma4-26b` + `open_ended`. Run that, then change the two constants and re-run five more times. Record each as its own TSV row with a clear description like `baseline <model> <prompt_type>`. Only after all six baselines are in should you move on to exploration.

## Strategy guidance (after baseline)

**Try:**
- `MAX_TOKENS` sweeps (128 / 256 / 512 / 1024) — how does decode throughput scale with generation length?
- Prompt-length sweeps — shorter and longer prompts, to study prefill vs decode separation.
- Prompt-prefix caching: reuse an identical prefix across all timed trials. llama.cpp's prompt cache should make later trials much faster on prefill — watch whether `prefill_tok_s` jumps between trial 1 and trials 2-5.
- One variable changed per run, so improvements are attributable.
- Keep at least 1 warmup + 3 timed trials.

**Avoid:**
- Concurrency / `asyncio.gather` / threads firing multiple requests at once.
- Non-chat endpoints (`/v1/completions`, `/v1/embeddings`, custom routes).
- Any attempt to reconfigure the server.
- Obviously-bad knobs (e.g. `MAX_TOKENS` absurdly high, extreme temperatures) — don't burn budget confirming the obvious.
- Changing `harness.py` or adding dependencies.

**Tips:**
- If trial-to-trial variance exceeds ~20% of the mean, flag it in the description column of `results.tsv`.
- The backend is Apple Silicon + Metal — your intuitions from CUDA hardware don't necessarily apply.
- `prefill_tok_s` and `decode_tok_s` are logged separately. A config that helps one may hurt the other — optimize the primary metric but keep an eye on both.

## Output format

`print_results()` emits:

```
---
tokens_per_second: 9.345678
prefill_tok_s: 120.500000
e2e_tok_s: 8.900000
ttft_ms: 350.000000
...
```

Extract the primary metric with:

```
grep "^tokens_per_second:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated — commas break in descriptions).

Header + 4 columns:

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars)
2. `tokens_per_second` value (0.000000 for crashes)
3. status: `keep`, `discard`, or `crash`
4. short text description, e.g. `baseline gemma4-26b open_ended max_tokens=200`

Example:

```
commit	metric_value	status	description
a1b2c3d	9.345000	keep	baseline gemma4-26b open_ended max_tokens=200
b2c3d4e	10.123000	keep	baseline supergemma4-26b open_ended max_tokens=200
c3d4e5f	11.500000	keep	max_tokens=512 with supergemma4-26b open_ended
d4e5f6g	0.000000	crash	dropped stream_options, usage never arrived
```

## Progress log (for human monitoring)

Also append one line per completed run to `progress.log` so the human can `tail -f progress.log` in another terminal and see the rolling history without being blind. Format:

```
[YYYY-MM-DD HH:MM:SS] run=<n> commit=<short> metric=<value> status=<keep|discard|crash> | <description>
```

Example:

```
[2026-04-21 14:23:01] run=1 commit=a1b2c3d metric=9.345000 status=keep | baseline gemma4-26b open_ended max_tokens=200
[2026-04-21 14:25:44] run=2 commit=b2c3d4e metric=10.123000 status=keep | baseline supergemma4-26b open_ended max_tokens=200
[2026-04-21 14:28:10] run=3 commit=0000000 metric=0.000000 status=crash | dropped stream_options, usage never arrived
```

Do **not** commit `progress.log` — keep it untracked like `results.tsv`. Use UTC or local time consistently. Append with `>>`, never rewrite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr21`).

LOOP FOREVER:

1. Look at git state: the current branch/commit.
2. Modify `experiment.py` with one experimental idea.
3. `git commit`.
4. Run: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT tee).
5. Read results: `grep "^tokens_per_second:" run.log`.
6. If grep is empty → run crashed. `tail -n 50 run.log` for the trace. If it's something trivial (typo, missing key), fix and retry. If fundamentally broken, log `crash` and move on.
7. Append one line to `progress.log` (see "Progress log" section) so the human can tail the rolling history.
8. Record in `results.tsv` (do NOT commit either file — keep both untracked).
9. If `tokens_per_second` improved (higher = better), advance: keep the commit.
10. If equal or worse, `git reset --hard HEAD~1` back.

**Timeout**: each run ≈ `TIME_BUDGET` seconds + overhead. If a run exceeds 2× budget, kill it and treat as failure.

**Crashes**: trivial fixes → fix and re-run; fundamentally broken ideas → log `crash` and move on.

**NEVER STOP**: once the loop begins (after baseline), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human expects you to continue *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the TSV, combine near-misses, try more radical changes. The loop runs until the human interrupts, period.
