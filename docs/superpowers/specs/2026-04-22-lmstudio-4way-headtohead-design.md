# LMStudio 4-way head-to-head — design

## Goal

Run the same autoexperiment head-to-head paradigm as the existing
`autoexperiment/apr21-qwen-headtohead` branch (Qwen bf16 vs q8 on a local
llama.cpp server), but against a **local LMStudio server** at
`http://localhost:1234/v1`, comparing **four models** in a 2×2 matrix of
architecture × backend:

|                 | GGUF (llama.cpp backend) | MLX backend           |
|-----------------|--------------------------|-----------------------|
| 27B dense       | `qwen3.6-27b@q8_k_xl`    | `qwen3.6-27b-mlx`     |
| 35B A3B MoE     | `qwen3.6-35b-a3b@q8_k_xl`| `qwen3.6-35b-a3b-mlx` |

Every invocation of `experiment.py` runs the **exact same request** against
all four models in a canonical order and logs one result block per model.
The agent's job is to sweep config knobs, not to optimize a single number.

## Non-goals

- Not exploring LMStudio SDK features (JIT loading APIs, load-time config,
  etc.) — we stay on the OpenAI-compatible HTTP surface so results are
  directly comparable to the llama.cpp and MLX branches.
- Not changing server-side config. The agent only controls the HTTP
  request body.
- Not producing a shippable GUI or new analysis tooling beyond what
  `viz.py` already does.

## API surface

- Inference: `POST {LMSTUDIO_BASE_URL}/chat/completions` (OpenAI-compatible).
- Readiness check (harness only): `GET {LMSTUDIO_BASE_URL%/v1}/api/v0/models`
  — LMStudio's native endpoint, which reports per-model `state`
  (`loaded` / `not-loaded`) and the quantization suffix.
- Base URL default: `http://localhost:1234/v1`, overridable with
  `LMSTUDIO_BASE_URL`.

## Files changed

On a fresh branch `autoexperiment/<tag>-lmstudio-4way` cut from `main`:

**Rewritten**:

- `harness.py` — new constants (URL, 4 model IDs, `TIME_BUDGET=600`,
  `MAX_RUNS=25`), readiness check hits `/api/v0/models`.
- `experiment.py` — baseline rewrite: 4-model canonical list, new env var
  name, otherwise reuses the current branch's `one_trial` / `run_model` /
  `trim_mean` helpers and prompt bank verbatim.
- `program.md` — adapted from the current branch's head-to-head version
  (not the main-branch template): LMStudio context, 4 models, new caps,
  4-rows-per-invocation logging.

**Deleted** (stale artifacts from earlier experiments / template scaffolding
that are not used by this experiment):

- `log_run.py`
- `log_run_b64.txt`
- `patch.py`
- `runexp.py`
- `step.py`
- `results.html` (regenerable from `results.tsv` via `viz.py`)

**Initialized fresh**:

- `results.tsv` — header row only.
- `progress.log` — empty, untracked.

**Unchanged**:

- `README.md`
- `viz.py` (may need minor tweaks for 4 model series; handled lazily, not
  fixed upfront)
- `pyproject.toml`, `uv.lock`
- `srt-settings.json`

## Harness config

```python
METRIC_NAME  = "tokens_per_second"
METRIC_GOAL  = "maximize"
TIME_BUDGET  = 600                          # seconds per invocation (4 models)
MAX_RUNS     = 25                           # → 100 TSV rows (25 per model × 4)

LMSTUDIO_BASE_URL_DEFAULT = "http://localhost:1234/v1"

MODEL_27B_GGUF = "qwen3.6-27b@q8_k_xl"
MODEL_27B_MLX  = "qwen3.6-27b-mlx"
MODEL_35B_GGUF = "qwen3.6-35b-a3b@q8_k_xl"
MODEL_35B_MLX  = "qwen3.6-35b-a3b-mlx"
```

`harness.py` as a script (`uv run harness.py`):

1. Prints metric, goal, time budget.
2. `GET /api/v0/models` and lists all entries with their `state`.
3. Asserts all four model IDs are present. `SystemExit(1)` if any missing.
   (Loaded-state is informational only — LMStudio JIT-loads on first
   request, so "not-loaded" at startup is fine.)

## Experiment config (agent-tunable)

```python
MODELS = [
    MODEL_27B_GGUF,   # smallest GGUF first
    MODEL_27B_MLX,    # same size, MLX
    MODEL_35B_GGUF,   # scale up
    MODEL_35B_MLX,
]

PROMPT_TYPE = "qa"
MAX_TOKENS = 512
NUM_TRIALS = 5
TEMPERATURE = 0.0
TOP_K = None
STREAM = False
MIN_COMPLETION_TOKENS = 50
```

**Canonical order rationale**: small → large; within each size, GGUF → MLX.
Cheapest model runs first, so bad configs crash fast. Order is fixed across
invocations so trial-order artifacts don't confound model comparisons.

**Prompt bank, `one_trial`, `run_model`, `trim_mean`, and the streaming /
non-streaming usage-accounting logic are copied verbatim from the current
branch's `experiment.py`.** LMStudio returns OpenAI-shaped usage blocks
including `completion_tokens_details.reasoning_tokens`, so the metric
plumbing works unchanged.

## Runtime characteristics

- LMStudio JIT-loads a model on first request. The per-model warmup trial
  inside `run_model()` absorbs that load cost.
- No preemptive load/unload calls — pure HTTP.
- Four models per invocation means up to four JIT load events per run
  (unless LMStudio keeps previous models resident). Budget includes
  headroom for this.
- `NUM_TRIALS=5` → 4 × (1 warmup + 5 timed) = 24 requests per invocation.
  Estimated wall time ~10–12 min/invocation; 25 invocations ≈ 4–5 h total.

## Output format

Per invocation, four `---` blocks (one per model), then four `RESULT`
lines, then `time_seconds`. Example:

```
---
tokens_per_second: 38.100000
model: qwen3.6-27b@q8_k_xl
...
---
tokens_per_second: 55.400000
model: qwen3.6-27b-mlx
...
---
tokens_per_second: 19.800000
model: qwen3.6-35b-a3b@q8_k_xl
...
---
tokens_per_second: 41.200000
model: qwen3.6-35b-a3b-mlx
...

RESULT model=qwen3.6-27b@q8_k_xl tokens_per_second=38.100000
RESULT model=qwen3.6-27b-mlx tokens_per_second=55.400000
RESULT model=qwen3.6-35b-a3b@q8_k_xl tokens_per_second=19.800000
RESULT model=qwen3.6-35b-a3b-mlx tokens_per_second=41.200000
time_seconds: 612.400000
```

## Logging

- `results.tsv` — header `commit\tmetric_value\tstatus\tdescription`,
  **four consecutive rows per invocation**, same short commit hash on all
  four. Description must start with the full model ID (for `viz.py`
  categorization).
- `progress.log` — one line per model result, so four lines per
  invocation. `run=<n>` increments once per row. Appended with `>>`.
- Status uses `keep` (valid) or `crash` (failed). Never `discard` — this is
  comparison data, not optimization.

Example:

```
a1b2c3d  38.100000  keep  qwen3.6-27b@q8_k_xl baseline qa
a1b2c3d  55.400000  keep  qwen3.6-27b-mlx baseline qa
a1b2c3d  19.800000  keep  qwen3.6-35b-a3b@q8_k_xl baseline qa
a1b2c3d  41.200000  keep  qwen3.6-35b-a3b-mlx baseline qa
```

## Baseline phase

Three baseline invocations, one per `PROMPT_TYPE`, all else default:

| PROMPT_TYPE   | rows logged |
|---------------|-------------|
| summarization | 4           |
| open_ended    | 4           |
| qa            | 4           |

3 invocations → 12 TSV rows, counted against the 25-invocation cap.
Remaining 22 invocations for sweeps.

## Strategy guidance (post-baseline)

Same sweep ideas as current branch, now observed across the 2×2 matrix:

- `MAX_TOKENS` sweeps (128 / 256 / 512 / 1024 / 2048) — does the
  GGUF-vs-MLX gap depend on generation length? Does the 27B-vs-35B gap?
- Prompt-length sweeps (short / medium / long) — prefill-dominated vs
  decode-dominated regimes, per backend per size.
- Prompt content variation — code, prose, math, multilingual. Reasoning-
  token ratio likely differs across models.
- `TEMPERATURE`, `TOP_K`, `STREAM` variations — sampling overhead
  differences between backends.
- Repeated identical requests for prefix-cache probing.
- One variable changed per run. Keep at least 1 warmup + 3 timed trials.

**Avoid**: concurrency, running <4 models per invocation, non-chat
endpoints, server reconfiguration, obviously-bad knobs.

## Run loop

1. Count invocations in `results.tsv` (unique commits minus header, or
   `grep -c "^RESULT model=qwen3.6-27b@q8_k_xl" ...`). If ≥ 25, stop.
2. Modify `experiment.py` with one knob change.
3. `git commit`.
4. `uv run experiment.py > run.log 2>&1`.
5. `grep "^RESULT " run.log` — expect 4 lines. Missing any → crash.
6. Append 4 lines to `progress.log`, 4 rows to `results.tsv`.
7. **Do NOT git-reset.** Every valid data point stays.

Timeout: kill a run that exceeds `2 × TIME_BUDGET` (1200s) and log
`crash`.

## Branch setup (one-time, pre-run)

1. `git checkout main && git pull`.
2. `git checkout -b autoexperiment/<tag>-lmstudio-4way`.
3. Delete stale files: `log_run.py`, `log_run_b64.txt`, `patch.py`,
   `runexp.py`, `step.py`, `results.html`.
4. Write new `harness.py`, `experiment.py`, `program.md` per this spec.
5. `: > results.tsv` then write header; leave `progress.log` empty and
   untracked.
6. `uv run harness.py` to verify all four models are discoverable.
7. Commit: `setup: lmstudio 4-way head-to-head baseline`.

## Open items (deferred, not blockers)

- `viz.py` currently categorizes by prefix matching on two model names;
  it may need small tweaks for four series. Verify after the first
  baseline invocation logs data; fix lazily if needed.
- Whether LMStudio keeps multiple models resident vs unloads the previous
  is unknown; the warmup trial absorbs either behavior. Note any
  anomalies in `results.tsv` description column.
