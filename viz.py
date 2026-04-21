"""
Visualize experiment results as a self-contained HTML chart.

Reads progress.log and results.tsv, writes results.html next to them.
Open results.html in a browser — Chart.js loads from CDN.

Usage:
    uv run viz.py            # writes results.html
    uv run viz.py --open     # also opens it in the default browser
"""

import html
import json
import re
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRESS_LOG = HERE / "progress.log"
RESULTS_TSV = HERE / "results.tsv"
OUT_HTML = HERE / "results.html"

LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+"
    r"run=(?P<run>\d+)\s+"
    r"commit=(?P<commit>[0-9a-f]+)\s+"
    r"metric=(?P<metric>[0-9.]+)\s+"
    r"status=(?P<status>\w+)\s*\|\s*"
    r"(?P<desc>.*)$"
)

MODELS = ("supergemma4-26b", "gemma4-26b")
PROMPT_TYPES = ("qa", "open_ended", "summarization")
MAX_TOKENS_RE = re.compile(r"max_tokens?=(\d+)", re.IGNORECASE)


def parse_progress(path):
    rows = []
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        desc = m["desc"].strip()
        rows.append({
            "ts": m["ts"],
            "run": int(m["run"]),
            "commit": m["commit"],
            "metric": float(m["metric"]),
            "status": m["status"],
            "description": desc,
            "model": next((x for x in MODELS if x in desc), ""),
            "prompt_type": next((x for x in PROMPT_TYPES if x in desc), ""),
            "max_tokens": _first_int(MAX_TOKENS_RE, desc),
            "baseline": desc.startswith("baseline "),
        })
    rows.sort(key=lambda r: r["run"])
    return rows


def _first_int(regex, s):
    m = regex.search(s)
    return int(m.group(1)) if m else None


def running_best(rows, goal="maximize"):
    """Best-so-far across kept runs (status == 'keep'), else carry previous."""
    cmp = (lambda a, b: a > b) if goal == "maximize" else (lambda a, b: a < b)
    best_vals = []
    best = None
    for r in rows:
        if r["status"] == "keep" and (best is None or cmp(r["metric"], best)):
            best = r["metric"]
        best_vals.append(best)
    return best_vals


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 24px 32px 48px;
    background: #fafafa;
    color: #222;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .summary {{
    display: flex; gap: 24px; flex-wrap: wrap;
    background: #fff; border: 1px solid #e5e5e5; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 16px; font-size: 13px;
  }}
  .summary b {{ color: #000; }}
  .chart-wrap {{
    background: #fff; border: 1px solid #e5e5e5; border-radius: 6px;
    padding: 16px; height: 520px;
  }}
  table {{
    border-collapse: collapse; width: 100%;
    background: #fff; margin-top: 16px; font-size: 12px;
    border: 1px solid #e5e5e5; border-radius: 6px; overflow: hidden;
  }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }}
  th {{ background: #f7f7f7; font-weight: 600; }}
  tr.keep td {{ background: #f6fbf6; }}
  tr.discard td {{ color: #888; }}
  tr.crash td {{ background: #fdf3f3; color: #a00; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .dot.keep {{ background: #2e8b57; }}
  .dot.discard {{ background: #999; }}
  .dot.crash {{ background: #d33; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="sub">{subtitle}</div>

<div class="summary">
  <div><b>Runs:</b> {n_runs}</div>
  <div><b>Kept:</b> {n_keep}</div>
  <div><b>Discarded:</b> {n_discard}</div>
  <div><b>Crashes:</b> {n_crash}</div>
  <div><b>Best {metric_name}:</b> {best_metric} <span style="color:#666">(run {best_run}, <code>{best_commit}</code>)</span></div>
  <div><b>Baseline best:</b> {baseline_best}</div>
  <div><b>Lift:</b> +{lift_pct}%</div>
</div>

<div class="chart-wrap"><canvas id="chart"></canvas></div>

<table>
  <thead>
    <tr>
      <th>#</th><th>status</th><th>metric</th><th>commit</th><th>model</th><th>prompt</th><th>max_tok</th><th>description</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<script>
const DATA = {data_json};

const keepPts = [];
const discardPts = [];
const crashPts = [];
const bestLine = [];

for (const r of DATA.rows) {{
  const pt = {{ x: r.run, y: r.metric, meta: r }};
  if (r.status === 'keep') keepPts.push(pt);
  else if (r.status === 'crash') crashPts.push(pt);
  else discardPts.push(pt);
}}
for (let i = 0; i < DATA.rows.length; i++) {{
  const r = DATA.rows[i];
  const best = DATA.running_best[i];
  if (best !== null) bestLine.push({{ x: r.run, y: best }});
}}

const annotations = [];
// Vertical separator between baseline and exploration
const lastBaseline = DATA.rows.filter(r => r.baseline).slice(-1)[0];
if (lastBaseline) annotations.push(lastBaseline.run + 0.5);

const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'best so far',
        type: 'line',
        data: bestLine,
        borderColor: '#2e8b57',
        backgroundColor: 'rgba(46,139,87,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        stepped: true,
        order: 0,
      }},
      {{
        label: 'keep',
        data: keepPts,
        backgroundColor: '#2e8b57',
        borderColor: '#2e8b57',
        pointRadius: 5,
        pointHoverRadius: 7,
        order: 1,
      }},
      {{
        label: 'discard',
        data: discardPts,
        backgroundColor: '#bbbbbb',
        borderColor: '#999999',
        pointRadius: 4,
        pointHoverRadius: 6,
        order: 2,
      }},
      {{
        label: 'crash',
        data: crashPts,
        backgroundColor: '#d33',
        borderColor: '#a00',
        pointRadius: 5,
        pointHoverRadius: 7,
        order: 1,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'nearest', intersect: true }},
    scales: {{
      x: {{
        type: 'linear',
        title: {{ display: true, text: 'run #' }},
        ticks: {{ stepSize: 5 }},
      }},
      y: {{
        title: {{ display: true, text: DATA.metric_name + ' (higher is better)' }},
      }},
    }},
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{
        callbacks: {{
          title: items => {{
            const r = items[0].raw.meta;
            return r ? `run ${{r.run}} — ${{r.status}}` : '';
          }},
          label: items => {{
            const r = items.raw.meta;
            if (!r) return `best so far: ${{items.raw.y.toFixed(3)}}`;
            const lines = [
              `${{DATA.metric_name}}: ${{r.metric.toFixed(4)}}`,
              `commit: ${{r.commit}}`,
              `time: ${{r.ts}}`,
            ];
            if (r.model) lines.push(`model: ${{r.model}}`);
            if (r.prompt_type) lines.push(`prompt: ${{r.prompt_type}}`);
            if (r.max_tokens !== null) lines.push(`max_tokens: ${{r.max_tokens}}`);
            lines.push(`— ${{r.description}}`);
            return lines;
          }}
        }}
      }},
      annotation: {{}},
    }},
  }},
  plugins: [
    {{
      id: 'baselineDivider',
      afterDraw(chart) {{
        if (!annotations.length) return;
        const {{ ctx, chartArea: {{ top, bottom }}, scales: {{ x }} }} = chart;
        for (const xv of annotations) {{
          const px = x.getPixelForValue(xv);
          ctx.save();
          ctx.strokeStyle = 'rgba(0,0,0,0.15)';
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(px, top);
          ctx.lineTo(px, bottom);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(0,0,0,0.45)';
          ctx.font = '11px -apple-system, sans-serif';
          ctx.fillText('← baselines | exploration →', px - 80, top + 12);
          ctx.restore();
        }}
      }}
    }}
  ]
}});
</script>
</body>
</html>
"""


def row_html(r):
    cls = r["status"]
    return (
        f'<tr class="{cls}">'
        f'<td>{r["run"]}</td>'
        f'<td><span class="dot {cls}"></span>{cls}</td>'
        f'<td>{r["metric"]:.4f}</td>'
        f'<td><code>{html.escape(r["commit"])}</code></td>'
        f'<td>{html.escape(r["model"])}</td>'
        f'<td>{html.escape(r["prompt_type"])}</td>'
        f'<td>{r["max_tokens"] if r["max_tokens"] is not None else ""}</td>'
        f'<td>{html.escape(r["description"])}</td>'
        f'</tr>'
    )


def render(rows, metric_name="tokens_per_second"):
    best_by = running_best(rows, goal="maximize")
    keeps = [r for r in rows if r["status"] == "keep"]
    discards = [r for r in rows if r["status"] == "discard"]
    crashes = [r for r in rows if r["status"] == "crash"]

    baselines = [r for r in rows if r["baseline"]]
    baseline_best = max((r["metric"] for r in baselines), default=0.0)
    best_row = max(keeps, key=lambda r: r["metric"]) if keeps else None
    best_metric = best_row["metric"] if best_row else 0.0
    lift_pct = ((best_metric - baseline_best) / baseline_best * 100.0) if baseline_best else 0.0

    data = {
        "metric_name": metric_name,
        "rows": rows,
        "running_best": best_by,
    }

    return HTML_TEMPLATE.format(
        title="autoexperiment — tokens/sec results",
        subtitle=f"{len(rows)} runs — branch visualization",
        metric_name=metric_name,
        n_runs=len(rows),
        n_keep=len(keeps),
        n_discard=len(discards),
        n_crash=len(crashes),
        best_metric=f"{best_metric:.4f}" if best_row else "—",
        best_run=best_row["run"] if best_row else "—",
        best_commit=best_row["commit"] if best_row else "—",
        baseline_best=f"{baseline_best:.4f}" if baselines else "—",
        lift_pct=f"{lift_pct:.2f}",
        rows_html="\n    ".join(row_html(r) for r in rows),
        data_json=json.dumps(data),
    )


def main():
    rows = parse_progress(PROGRESS_LOG)
    if not rows:
        print(f"no parsable rows in {PROGRESS_LOG}", file=sys.stderr)
        sys.exit(1)
    OUT_HTML.write_text(render(rows))
    print(f"wrote {OUT_HTML} ({len(rows)} rows)")
    if "--open" in sys.argv:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
