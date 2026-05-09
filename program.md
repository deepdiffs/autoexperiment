# autoexperiment — Qwen3.6-27B roofline on M3 Ultra

Autonomous experiment optimization. An AI agent iterates on `experiment.py`, runs
llama-bench against the local Qwen3.6-27B Q8 GGUF, and tries to push the achieved
memory bandwidth as close to the measured roofline as possible.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may9`). The branch `autoexperiment/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these for full context:
   - `README.md` — repository context.
   - `harness.py` — fixed config (model path, metric, time budget, peak-BW measurement). Do not modify.
   - `experiment.py` — the file you modify. Your entire experiment lives here.
4. **Run setup**: `uv run harness.py` measures peak memory bandwidth via MLX and writes `peak_bw.json`. This must complete before the first experiment.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good (peak BW measurement looks plausible — expect 400–700 GB/s on M3 Ultra).

Once you get confirmation, kick off the experimentation.

## Domain context

The metric is `bandwidth_utilization_pct` — the percentage of *measured* peak memory
bandwidth that decode achieves on Qwen3.6-27B Q8.

LLM decode is memory-bound: every generated token reads (essentially) all weights
once. So effective bandwidth ≈ `model_size_bytes × tokens_per_second`. The measured
peak (from MLX in `peak_bw.json`) is the realistic ceiling for streaming reads on
Apple Silicon. We also report `bw_pct_vs_spec` against the 800 GB/s theoretical
peak as a sanity check.

Prefill is compute-bound (high arithmetic intensity) and shows up on the roofline
plot but does **not** count toward the optimization metric — only `tg` (decode)
points contribute.

Each run also writes `roofline.png` showing achieved (AI, GFLOPS) points against
the bandwidth roof and compute roof. This is a wrap-up artifact: it's overwritten
each run; the latest version reflects the current best config.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py` — this is the only file you edit. Everything is fair game: the llama-bench command, the sweep grid, parsing logic, plotting, batch/ubatch/quant/KV-dtype/flash-attn flags, etc.

**What you CANNOT do:**
- Modify `harness.py`. It is read-only. The metric definition, model path, peak-BW measurement, and time budget live there.
- Install new packages. Use only what's in `pyproject.toml` (`mlx`, `matplotlib`, `numpy`) plus the stdlib.
- Modify `~/models/`. The GGUF is shared infra. Do not move, delete, or quantize it.
- Restart or kill the LaunchDaemon llama-server. `llama-bench` runs as a separate process; it does not need the daemon.

**The goal: maximize `bandwidth_utilization_pct`.** Higher = closer to the bandwidth roof. The time budget is fixed at 60s — pick a sweep that finishes in time.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing things and getting equal results is a win.

**The first run**: establish the baseline. Run `experiment.py` as-is.

## Strategy guidance

Sweep axes worth exploring (the agent should try these):
- **`-b` / `-ub` (batch / ubatch sizes)** — affects prompt-processing parallelism.
- **`-fa 1` (flash attention)** — typically helps decode at long context.
- **`-ctk` / `-ctv` (KV cache dtype: f16, q8_0, q4_0)** — smaller KV → less BW pressure.
- **`-d` (depth, KV pre-fill)** — shows how decode BW changes as KV grows.
- **`-p` / `-n` choices** — pure decode (`-p 0 -n N`) gives the cleanest BW number.
- **`-r` (repetitions)** — more reps = lower noise, but consumes the 60s budget faster.
- **`-t` (CPU threads)** and **`-ngl` (GPU layers)** — less impactful since this model is fully GPU-resident, but worth checking.

Approaches to avoid:
- **Don't change the model** — stay on `Qwen3.6-27B-UD-Q8_K_XL.gguf`. Smaller/quantized variants game the metric (less data to move).
- **Don't fudge `model_bytes`** — `bytes/decode-token` should reflect what's actually read. If you account for KV cache reads more precisely, that's fine and welcome, but don't artificially inflate it to push the percentage up.
- **Don't disable warmup unless intentional** — first-run cache effects skew the number.
- **Don't change the metric formula in a way that disconnects from physical bandwidth** — the % must remain a meaningful fraction of GB/s observed vs GB/s achievable.

Good improvements look like: a more careful bytes-per-token model (counting KV correctly), running more reps and reporting best-of, finding a flag combo that genuinely raises tg tok/s, or recognizing that the baseline left BW on the table because of a startup-cost artifact.

## Output format

The experiment prints results using `print_results()`:

```
---
bandwidth_utilization_pct: 72.500000
bw_pct_vs_spec: 56.300000
peak_bw_measured_gbs: 620.500000
n_points: 6
time_seconds: 42.300000
```

Extract the metric:

```
grep "^bandwidth_utilization_pct:" run.log
```

## Logging results

When an experiment finishes, log it to `results.tsv` (tab-separated).

Header + 4 columns:

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars)
2. `bandwidth_utilization_pct` value (e.g. 72.500000) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this run tried

Example:

```
commit	metric_value	status	description
a1b2c3d	62.300000	keep	baseline mini-sweep p=0,512,2048
b2c3d4e	68.100000	keep	add -fa 1 flash attn, -r 3
c3d4e5f	60.000000	discard	switch ctk=q4_0 — bytes-per-token denominator dropped, % went down
```

Do not commit `results.tsv`, `peak_bw.json`, or `roofline.png` — leave them untracked.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/may9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `experiment.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT use tee)
5. Read out the results: `grep "^bandwidth_utilization_pct:" run.log`
6. If the grep output is empty, the run crashed. `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after a few attempts, log "crash" and move on.
7. Record the results in `results.tsv` (do not commit it)
8. If the metric improved, "advance" the branch and keep the commit
9. If the metric is equal or worse, `git reset --hard` back to where you started

**Timeout**: Each run should take ~60s + a few seconds overhead. If a run exceeds 2x the budget (>120s), kill it and treat it as a crash.

**Crashes**: Use judgment. Typo or missing import → fix and re-run. Fundamentally broken idea → log "crash" and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep and expects you to continue *indefinitely* until manually stopped. If you run out of ideas, think harder — re-read the code, combine previous near-misses, try more radical changes. The loop runs until the human interrupts you.
