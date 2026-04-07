# autoexperiment

Autonomous experiment optimization. An AI agent iterates on code, runs experiments, and keeps what works.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoexperiment/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoexperiment/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `harness.py` — fixed configuration (metric name, goal, time budget) and utilities. Do not modify.
   - `experiment.py` — the file you modify. Your entire experiment lives here.
4. **Run setup if needed**: If `harness.py` has a setup step, tell the human to run `uv run harness.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is run with: `uv run experiment.py`

**What you CAN do:**
- Modify `experiment.py` — this is the only file you edit. Everything is fair game: algorithm, parameters, approach, implementation, imports, structure.

**What you CANNOT do:**
- Modify `harness.py`. It is read-only. It contains the fixed metric configuration and utilities.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: optimize the metric defined in `harness.py`.** Read `METRIC_NAME` and `METRIC_GOAL` to know what you're optimizing and in which direction. Since the time budget is fixed, you don't need to worry about experiment duration — it's always the same. Everything is fair game as long as the code runs and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the experiment as is.

## Output format

The experiment prints a summary using `print_results()` from the harness:

```
---
score:        0.123456
time_seconds: 42.300000
```

You can extract the key metric from the log:

```
grep "^<METRIC_NAME>:" run.log
```

(Replace `<METRIC_NAME>` with the actual metric name from `harness.py`.)

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	metric_value	status	description
```

1. git commit hash (short, 7 chars)
2. metric value achieved (e.g. 0.123456) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	metric_value	status	description
a1b2c3d	0.500000	keep	baseline
b2c3d4e	0.420000	keep	switch to binary search
c3d4e5f	0.510000	discard	add random restarts
d4e5f6g	0.000000	crash	refactor broke imports
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoexperiment/apr6`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `experiment.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^<METRIC_NAME>:" run.log` (use the actual metric name from harness.py)
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the TSV (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the metric improved (check `METRIC_GOAL` in harness.py for direction), you "advance" the branch, keeping the git commit
9. If the metric is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck, you can rewind but do this very sparingly (if ever).

**Timeout**: Each experiment should take roughly `TIME_BUDGET` seconds (+ a few seconds for overhead). If a run exceeds 2x the budget, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: if it's something dumb and easy to fix (typo, missing import), fix it and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
