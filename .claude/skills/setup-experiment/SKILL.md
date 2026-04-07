---
name: setup-experiment
description: >-
  Interactively configure this autoexperiment template for a specific experiment.
  Gathers requirements via questions (domain, metric, dependencies, baseline,
  benchmarking strategy, agent guidance) then rewrites harness.py, experiment.py,
  pyproject.toml, and program.md to implement the experiment. Use when the user
  says "setup", "configure", "initialize", "customize", "new experiment", or
  wants to adapt this template for their own use case.
---

# Setup Experiment

Guide the user through configuring this autoexperiment template for their specific experiment. The goal is to gather all requirements interactively, then rewrite the repo files to implement a fully working experiment that the autonomous agent can iterate on.

## Overview

This repo is a generic template for autonomous agent-driven experiment optimization. It has four key files:

| File | Role | Editable? |
|------|------|-----------|
| `harness.py` | Metric config, timer, print helper | Rewritten during setup |
| `experiment.py` | The experiment code the agent iterates on | Rewritten during setup |
| `pyproject.toml` | Project metadata and dependencies | Rewritten during setup |
| `program.md` | Instructions for the autonomous agent | Rewritten during setup |

## Workflow

Follow these phases sequentially. Use `AskUserQuestion` for every phase. Do NOT skip phases or assume answers.

### Phase 1: Understand the domain

Ask the user:
- What experiment they want to run (broad description)
- What domain this is in (ML training, data processing, algorithm optimization, simulation, etc.)

Then ask clarifying follow-ups based on their answer. For ML experiments, ask about the model architecture, dataset, and task. For algorithm optimization, ask about the problem being solved. Tailor your questions to the domain.

### Phase 2: Define the metric

Ask:
- What metric should be optimized (e.g., loss, accuracy, F1, throughput, latency, score)
- Whether to minimize or maximize it
- How the metric is computed (is it from a library, a custom formula, an external benchmark?)

### Phase 3: Set the time budget

Ask:
- How long each experiment run should take (the time budget in seconds)
- Explain that all runs use the same budget so improvements are directly comparable

### Phase 4: Dependencies and environment

Ask:
- What Python packages are needed (e.g., torch, numpy, scikit-learn, transformers)
- Whether any data needs to be downloaded or prepared as a one-time setup step
- Whether there are any hardware requirements (GPU, specific CPU features)

### Phase 5: Baseline implementation

Ask:
- What the initial baseline experiment should do
- Whether they have existing code to use as a starting point (if so, ask them to paste it or point to a file)
- What the expected baseline metric value is (roughly)

### Phase 6: Benchmarking and evaluation

Ask:
- How should results be validated (e.g., held-out test set, cross-validation, fixed seed)
- Are there any correctness constraints (e.g., "output must still pass these tests")
- What constitutes a meaningful improvement vs noise

### Phase 7: Agent guidance

Ask:
- What approaches should the agent try (e.g., hyperparameter tuning, architecture changes, data augmentation)
- What approaches should the agent avoid (e.g., "don't use external pretrained models", "don't change the dataset")
- Any domain-specific tips or constraints
- Any other important context

### Phase 8: Confirm and implement

Summarize everything gathered and present it to the user for confirmation. Then rewrite all four files:

#### 1. Rewrite `harness.py`

Update the configuration section:
```python
METRIC_NAME = "<user's metric>"
METRIC_GOAL = "<minimize or maximize>"
TIME_BUDGET = <user's time budget>
```

If there is one-time setup (data download, preprocessing), add it to the `if __name__ == "__main__"` block.

Do NOT change the `Timer` class or `print_results` function signatures.

#### 2. Rewrite `experiment.py`

Implement the baseline experiment. It must:
- Import from harness: `from harness import METRIC_NAME, TIME_BUDGET, Timer, print_results`
- Create a `Timer()` instance at the start
- Compute the metric
- End with `print_results(**{METRIC_NAME: result, "time_seconds": timer.elapsed()})`

#### 3. Update `pyproject.toml`

Add all required dependencies to the `dependencies` list. Keep the existing project metadata but update the description.

#### 4. Customize `program.md`

Keep the core structure (setup phase, experimentation loop, output format, logging, the experiment loop steps) but customize:
- Replace generic metric references with the actual metric name
- Add domain-specific guidance to a new "## Domain context" section
- Add the user's "try these" and "avoid these" notes to a "## Strategy guidance" section
- Add any correctness constraints or validation requirements

#### 5. Run `uv sync`

After rewriting files, run `uv sync` to install dependencies.

#### 6. Run `uv run harness.py`

If there is one-time setup code, run it.

#### 7. Run baseline

Run `uv run experiment.py` to verify the baseline works and report the initial metric value.

## Important rules

- Ask ONE phase of questions at a time. Do not overwhelm the user with all questions at once.
- Use `AskUserQuestion` with well-structured options whenever possible (not just free-text).
- If the user's answers are vague, ask targeted follow-ups before moving on.
- Preserve the core autoexperiment architecture - the agent-modifiable file is always `experiment.py`.
- The harness.py `Timer` class and `print_results` function signatures must not change.
- Keep `program.md` structure intact - the autonomous loop instructions are battle-tested.
- After implementation, always verify the baseline runs successfully.
