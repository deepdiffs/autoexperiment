"""
Experiment file — the agent modifies ONLY this file.
Usage: uv run experiment.py

Goal: maximize the achieved memory bandwidth as a fraction of the measured
peak. Each run also writes roofline.png so we can visualize where the points
land relative to the bandwidth roof and the compute roof.
"""

import json
import shutil
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import (
    METRIC_NAME,
    MODEL_PATH,
    PEAK_BW_SPEC_GBS,
    TIME_BUDGET,
    Timer,
    load_peak_bw,
    model_size_bytes,
    print_results,
)

HERE = Path(__file__).resolve().parent
ROOFLINE_PNG = HERE / "roofline.png"

# Approx FLOPs ceiling for M3 Ultra Metal (32-core GPU, ~27 TFLOPS fp16).
# Used only as the compute roof on the roofline plot, not the optimization metric.
PEAK_COMPUTE_TFLOPS = 27.0
N_PARAMS = 27e9  # Qwen3.6-27B


def run_llama_bench(p_values, n_values, batch=2048, ubatch=512, ctk="f16", ctv="f16", fa=0, reps=2, poll=100):
    """Invoke llama-bench once with comma-separated p/n sweeps; return parsed JSON results.

    --poll 100 = busy-wait for GPU completion (lowest dispatch latency).
    """
    cmd = [
        "llama-bench",
        "-m", MODEL_PATH,
        "-p", ",".join(str(p) for p in p_values),
        "-n", ",".join(str(n) for n in n_values),
        "-b", str(batch),
        "-ub", str(ubatch),
        "-ctk", ctk,
        "-ctv", ctv,
        "-fa", str(fa),
        "-r", str(reps),
        "--poll", str(poll),
        "-o", "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {proc.stderr[-500:]}")
    return json.loads(proc.stdout)


def bw_utilization(tg_tokens_per_sec, model_bytes, peak_gbs):
    """% of peak memory bandwidth achieved by decode at this tok/s.
    Decode reads ~all weights once per token, so effective_BW ≈ model_bytes * tg.
    """
    effective_gbs = model_bytes * tg_tokens_per_sec / 1e9
    return 100.0 * effective_gbs / peak_gbs


def plot_roofline(points, peak_bw_gbs, peak_bw_spec_gbs, peak_compute_tflops, out_path):
    """Plot roofline: arithmetic intensity (x) vs achieved performance (y) in GFLOPS."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ai = [10 ** (i / 4) for i in range(-8, 21)]  # 0.001 -> ~32k FLOPs/byte
    bw_roof_gflops = [a * peak_bw_gbs for a in ai]
    bw_roof_spec_gflops = [a * peak_bw_spec_gbs for a in ai]
    compute_roof_gflops = peak_compute_tflops * 1000  # TFLOPS -> GFLOPS

    ax.plot(ai, bw_roof_gflops, "b-", lw=2, label=f"BW roof (measured: {peak_bw_gbs:.0f} GB/s)")
    ax.plot(ai, bw_roof_spec_gflops, "b--", lw=1, alpha=0.5, label=f"BW roof (spec: {peak_bw_spec_gbs:.0f} GB/s)")
    ax.axhline(compute_roof_gflops, color="r", lw=2, label=f"Compute roof ({peak_compute_tflops:.0f} TFLOPS)")

    if points:
        xs = [p["ai"] for p in points]
        ys = [p["gflops"] for p in points]
        labels = [p["label"] for p in points]
        ax.scatter(xs, ys, s=80, c="green", edgecolors="black", zorder=5)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Qwen3.6-27B Q8 on M3 Ultra — Roofline")
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(1, max(compute_roof_gflops, max(ys) if points else 1) * 2)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    if not shutil.which("llama-bench"):
        raise SystemExit("llama-bench not found on PATH (brew install llama.cpp)")

    timer = Timer()
    peak_bw = load_peak_bw()

    # Baseline mini-sweep. llama-bench expands `-p A,B,C -n X,Y` into separate
    # tests; entries with n_prompt>0,n_gen=0 are prefill (pp), n_gen>0,n_prompt=0
    # are decode (tg). We probe a couple prefill lengths and a couple decode lengths.
    p_values = [0, 2048]
    n_values = [0, 128]

    try:
        entries = run_llama_bench(p_values=p_values, n_values=n_values, reps=3)
    except Exception as e:
        print(f"# llama-bench failed: {e}")
        entries = []

    points = []
    bw_utilizations = []
    tg_median_pct = None
    tg_stddev_pct = None

    for entry in entries:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        samples = [float(s) for s in (entry.get("samples_ts") or [entry.get("avg_ts", 0.0)])]
        tps_max = max(samples)
        tps_median = statistics.median(samples)
        tps_stddev = statistics.stdev(samples) if len(samples) >= 2 else 0.0
        model_bytes = int(entry.get("model_size", model_size_bytes()))
        n_params = int(entry.get("model_n_params", N_PARAMS))
        if tps_max <= 0:
            continue

        if n_gen > 0 and n_prompt == 0:
            bw_pct = bw_utilization(tps_max, model_bytes, peak_bw)
            bw_utilizations.append(bw_pct)
            tg_median_pct = bw_utilization(tps_median, model_bytes, peak_bw)
            tg_stddev_pct = bw_utilization(tps_stddev, model_bytes, peak_bw)
            ai = 2 / (model_bytes / n_params)  # ~2 FLOPs per weight read
            gflops = tps_max * 2 * n_params / 1e9
            label = f"tg{n_gen} ({tps_max:.1f} t/s, {bw_pct:.0f}%)"
            points.append({"ai": ai, "gflops": gflops, "label": label})
            print(
                f"# tg n={n_gen:<5} max={tps_max:.2f} med={tps_median:.2f} "
                f"sd={tps_stddev:.2f} t/s -> {bw_pct:.1f}% of measured peak BW "
                f"(median: {tg_median_pct:.1f}%)"
            )
        elif n_prompt > 0 and n_gen == 0:
            ai = n_prompt * 2  # AI grows ~linearly with prefill length
            gflops = tps_max * 2 * n_params / 1e9
            label = f"pp{n_prompt} ({tps_max:.0f} t/s)"
            points.append({"ai": ai, "gflops": gflops, "label": label})
            print(f"# pp p={n_prompt:<5} max={tps_max:.2f} med={tps_median:.2f} t/s")

    plot_roofline(points, peak_bw, PEAK_BW_SPEC_GBS, PEAK_COMPUTE_TFLOPS, ROOFLINE_PNG)
    print(f"# wrote {ROOFLINE_PNG.name}")

    best_bw_pct = max(bw_utilizations) if bw_utilizations else 0.0
    bw_pct_vs_spec = best_bw_pct * load_peak_bw() / PEAK_BW_SPEC_GBS

    print_results(**{
        METRIC_NAME: best_bw_pct,
        "bw_pct_vs_spec": bw_pct_vs_spec,
        "tg_median_pct": tg_median_pct or 0.0,
        "tg_stddev_pct": tg_stddev_pct or 0.0,
        "peak_bw_measured_gbs": peak_bw,
        "n_points": len(points),
        "time_seconds": timer.elapsed(),
    })


if __name__ == "__main__":
    main()
