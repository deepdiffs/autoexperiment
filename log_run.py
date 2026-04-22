import subprocess, datetime, re, sys

desc_suffix = sys.argv[1] if len(sys.argv) > 1 else "unknown"

results = []
for line in open("run.log"):
    m = re.match(r"^RESULT model=(\S+) tokens_per_second=([\d.]+)", line)
    if m:
        results.append((m.group(1), float(m.group(2))))

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

shortnames = {
    "Qwen3.6-35B-A3B-BF16-00001-of-00002": "qwen3.6-35b-bf16",
    "Qwen3.6-35B-A3B-UD-Q8_K_XL": "qwen3.6-35b-q8",
}

with open("progress.log") as f:
    existing = [l for l in f if l.strip()]
next_run = len(existing) + 1

ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

full_by_short = {}
for full, tps in results:
    short = shortnames.get(full, full)
    full_by_short[short] = tps

all_models = ["qwen3.6-35b-bf16", "qwen3.6-35b-q8"]

TAB = chr(9)
NL = chr(10)

if len(results) == 0:
    with open("results.tsv", "a") as tsv, open("progress.log", "a") as plog:
        desc = "crash " + desc_suffix
        tsv.write(commit + TAB + "0.000000" + TAB + "crash" + TAB + desc + NL)
        plog.write("[" + ts + "] run=" + str(next_run) + " commit=" + commit + " metric=0.000000 status=crash | " + desc + NL)
    print("logged crash")
    sys.exit(0)

with open("results.tsv", "a") as tsv, open("progress.log", "a") as plog:
    for short in all_models:
        if short in full_by_short:
            tps = full_by_short[short]
            desc = short + " " + desc_suffix
            status = "keep"
            tsv.write(commit + TAB + ("%.6f" % tps) + TAB + status + TAB + desc + NL)
            plog.write("[" + ts + "] run=" + str(next_run) + " commit=" + commit + " metric=" + ("%.6f" % tps) + " status=" + status + " | " + desc + NL)
            next_run += 1
        else:
            desc = short + " crash " + desc_suffix
            tsv.write(commit + TAB + "0.000000" + TAB + "crash" + TAB + desc + NL)
            plog.write("[" + ts + "] run=" + str(next_run) + " commit=" + commit + " metric=0.000000 status=crash | " + desc + NL)
            next_run += 1

print("logged", len(results), "results")
