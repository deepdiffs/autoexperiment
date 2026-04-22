"""
Visualize head-to-head experiment results as a self-contained HTML chart.

Reads progress.log, writes results.html next to it. Each invocation produces
two rows (bf16 + q8); this renders them as two series sharing an invocation #
on the x-axis so the per-config delta is obvious.

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
OUT_HTML = HERE / "results.html"

LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+"
    r"run=(?P<run>\d+)\s+"
    r"commit=(?P<commit>[0-9a-f]+)\s+"
    r"metric=(?P<metric>[0-9.]+)\s+"
    r"status=(?P<status>\w+)\s*\|\s*"
    r"(?P<desc>.*)$"
)

MODEL_BF16 = "qwen3.6-35b-bf16"
MODEL_Q8 = "qwen3.6-35b-q8"
MODELS = (MODEL_BF16, MODEL_Q8)
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
            "baseline": "baseline" in desc.lower(),
        })
    rows.sort(key=lambda r: r["run"])

    # Assign invocation # by commit order (each invocation shares one commit).
    invocation_by_commit = {}
    for r in rows:
        if r["commit"] not in invocation_by_commit:
            invocation_by_commit[r["commit"]] = len(invocation_by_commit) + 1
        r["invocation"] = invocation_by_commit[r["commit"]]
    return rows


def _first_int(regex, s):
    m = regex.search(s)
    return int(m.group(1)) if m else None


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
  tr.keep td {{ background: #fbfbfb; }}
  tr.crash td {{ background: #fdf3f3; color: #a00; }}
  tr.bf16 td.model {{ color: #1f6feb; font-weight: 600; }}
  tr.q8 td.model {{ color: #2e8b57; font-weight: 600; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .dot.keep {{ background: #2e8b57; }}
  .dot.crash {{ background: #d33; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="sub">{subtitle}</div>

<div class="summary">
  <div><b>Invocations:</b> {n_invocations}</div>
  <div><b>Rows:</b> {n_rows} ({n_keep} keep, {n_crash} crash)</div>
  <div><b>bf16 mean:</b> {bf16_mean} tok/s <span style="color:#666">(min {bf16_min}, max {bf16_max})</span></div>
  <div><b>q8 mean:</b> {q8_mean} tok/s <span style="color:#666">(min {q8_min}, max {q8_max})</span></div>
  <div><b>q8 / bf16:</b> {ratio}×</div>
</div>

<div class="chart-wrap"><canvas id="chart"></canvas></div>

<table>
  <thead>
    <tr>
      <th>#</th><th>inv</th><th>status</th><th>metric</th><th>commit</th><th>model</th><th>prompt</th><th>max_tok</th><th>description</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<script>
const DATA = {data_json};

const bf16Pts = [];
const q8Pts = [];
const crashPts = [];

for (const r of DATA.rows) {{
  const pt = {{ x: r.invocation, y: r.metric, meta: r }};
  if (r.status === 'crash') crashPts.push(pt);
  else if (r.model === 'qwen3.6-35b-bf16') bf16Pts.push(pt);
  else if (r.model === 'qwen3.6-35b-q8') q8Pts.push(pt);
}}

const annotations = [];
const lastBaseline = DATA.rows.filter(r => r.baseline).slice(-1)[0];
if (lastBaseline) annotations.push(lastBaseline.invocation + 0.5);

const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'qwen3.6-35b-bf16',
        data: bf16Pts,
        backgroundColor: '#1f6feb',
        borderColor: '#1f6feb',
        pointRadius: 4,
        pointHoverRadius: 7,
      }},
      {{
        label: 'qwen3.6-35b-q8',
        data: q8Pts,
        backgroundColor: '#2e8b57',
        borderColor: '#2e8b57',
        pointRadius: 4,
        pointHoverRadius: 7,
      }},
      {{
        label: 'crash',
        data: crashPts,
        backgroundColor: '#d33',
        borderColor: '#a00',
        pointStyle: 'crossRot',
        pointRadius: 7,
        pointHoverRadius: 9,
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
        title: {{ display: true, text: 'invocation #' }},
        ticks: {{ stepSize: 5 }},
      }},
      y: {{
        title: {{ display: true, text: 'tokens_per_second' }},
      }},
    }},
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{
        callbacks: {{
          title: items => {{
            const r = items[0].raw.meta;
            return r ? `invocation ${{r.invocation}} — ${{r.model || r.status}}` : '';
          }},
          label: items => {{
            const r = items.raw.meta;
            if (!r) return '';
            const lines = [
              `tokens_per_second: ${{r.metric.toFixed(4)}}`,
              `commit: ${{r.commit}}`,
              `time: ${{r.ts}}`,
            ];
            if (r.prompt_type) lines.push(`prompt: ${{r.prompt_type}}`);
            if (r.max_tokens !== null) lines.push(`max_tokens: ${{r.max_tokens}}`);
            lines.push(`— ${{r.description}}`);
            return lines;
          }}
        }}
      }},
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
    model_cls = "bf16" if r["model"] == MODEL_BF16 else ("q8" if r["model"] == MODEL_Q8 else "")
    return (
        f'<tr class="{cls} {model_cls}">'
        f'<td>{r["run"]}</td>'
        f'<td>{r["invocation"]}</td>'
        f'<td><span class="dot {cls}"></span>{cls}</td>'
        f'<td>{r["metric"]:.4f}</td>'
        f'<td><code>{html.escape(r["commit"])}</code></td>'
        f'<td class="model">{html.escape(r["model"])}</td>'
        f'<td>{html.escape(r["prompt_type"])}</td>'
        f'<td>{r["max_tokens"] if r["max_tokens"] is not None else ""}</td>'
        f'<td>{html.escape(r["description"])}</td>'
        f'</tr>'
    )


def _stats(values):
    if not values:
        return ("—", "—", "—", 0.0)
    return (f"{sum(values)/len(values):.2f}", f"{min(values):.2f}", f"{max(values):.2f}", sum(values)/len(values))


def render(rows):
    keeps = [r for r in rows if r["status"] == "keep"]
    crashes = [r for r in rows if r["status"] == "crash"]
    n_invocations = max((r["invocation"] for r in rows), default=0)

    bf16_vals = [r["metric"] for r in keeps if r["model"] == MODEL_BF16]
    q8_vals = [r["metric"] for r in keeps if r["model"] == MODEL_Q8]
    bf16_mean_s, bf16_min_s, bf16_max_s, bf16_mean = _stats(bf16_vals)
    q8_mean_s, q8_min_s, q8_max_s, q8_mean = _stats(q8_vals)
    ratio = f"{(q8_mean / bf16_mean):.3f}" if bf16_mean else "—"

    data = {"rows": rows}

    return HTML_TEMPLATE.format(
        title="autoexperiment — Qwen bf16 vs q8 head-to-head",
        subtitle=f"{n_invocations} invocations, {len(rows)} rows — branch visualization",
        n_invocations=n_invocations,
        n_rows=len(rows),
        n_keep=len(keeps),
        n_crash=len(crashes),
        bf16_mean=bf16_mean_s, bf16_min=bf16_min_s, bf16_max=bf16_max_s,
        q8_mean=q8_mean_s, q8_min=q8_min_s, q8_max=q8_max_s,
        ratio=ratio,
        rows_html="\n    ".join(row_html(r) for r in rows),
        data_json=json.dumps(data),
    )


def main():
    rows = parse_progress(PROGRESS_LOG)
    if not rows:
        print(f"no parsable rows in {PROGRESS_LOG}", file=sys.stderr)
        sys.exit(1)
    OUT_HTML.write_text(render(rows))
    print(f"wrote {OUT_HTML} ({len(rows)} rows, {max(r['invocation'] for r in rows)} invocations)")
    if "--open" in sys.argv:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
