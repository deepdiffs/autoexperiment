# autoexperiment

Give an AI agent an experiment and let it optimize autonomously. It modifies the code, runs the experiment, checks if the metric improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) better results.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). This is a generalized template — the original optimized LLM training; this template works for any experiment.

## How it works

Three files:

- **`harness.py`** — fixed configuration (metric name, goal, time budget) and utilities. Not modified by the agent.
- **`experiment.py`** — the single file the agent edits. Your experiment lives here. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human**.

The agent runs `experiment.py` within a **fixed time budget**, measures the metric, and keeps or discards the change. Lower or higher is better depending on `METRIC_GOAL` in the harness.

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Customize for your experiment:
#    - Edit harness.py: set METRIC_NAME, METRIC_GOAL, TIME_BUDGET
#    - Edit experiment.py: implement your baseline experiment
#    - Optionally edit program.md: add domain-specific guidance for the agent

# 4. Run setup if your harness needs it
uv run harness.py

# 5. Test a single run
uv run experiment.py
```

## Running the agent

Spin up Claude Code (or your agent of choice) in this repo, then prompt:

```
Read program.md and let's kick off a new experiment! Let's do the setup first.
```

## Using as a template

1. **Define your metric** in `harness.py`: set `METRIC_NAME`, `METRIC_GOAL` ("minimize" or "maximize"), and `TIME_BUDGET`.
2. **Write your baseline** in `experiment.py`: implement the experiment, call `print_results()` at the end.
3. **Add domain context** to `program.md` (optional): constraints, things to try, things to avoid.
4. **Add dependencies** to `pyproject.toml` as needed.
5. Let the agent loose.

## Design choices

- **Single file to modify.** The agent only touches `experiment.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Every experiment runs for the same duration, making results directly comparable regardless of what the agent changes.
- **Self-contained.** One file, one metric. No complex configs.

## License

MIT
