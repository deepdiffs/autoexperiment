import sys, re
path = "experiment.py"
s = open(path).read()
for arg in sys.argv[1:]:
    if arg.startswith("--marker="):
        marker = arg[len("--marker="):]
        s = re.sub(r"#\s*baseline:.*", "# " + marker, s, count=1)
        continue
    key, _, val = arg.partition("=")
    pat = r"^" + re.escape(key) + r"\s*=.*$"
    new = key + " = " + val
    s2, n = re.subn(pat, new, s, count=1, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit("did not find assignment for " + key)
    s = s2
open(path, "w").write(s)
print("patched:", sys.argv[1:])
