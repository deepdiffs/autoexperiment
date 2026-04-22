import sys, subprocess, re

args = sys.argv[1:]
if not args:
    raise SystemExit("usage: step.py DESC KEY=VAL [KEY=VAL ...]")
desc = args[0]
patches = args[1:]

path = "experiment.py"
s = open(path).read()
for p in patches:
    key, _, val = p.partition("=")
    pat = r"^" + re.escape(key) + r"\s*=.*$"
    new = key + " = " + val
    s2, n = re.subn(pat, new, s, count=1, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit("did not find assignment for " + key)
    s = s2
open(path, "w").write(s)

subprocess.run(["git","add","experiment.py"], check=True)
subprocess.run(["git","commit","--allow-empty","-m",desc], check=True)

print("[step] running experiment: " + desc)
r = subprocess.run(["uv","run","experiment.py"], capture_output=True, text=True, timeout=540)
open("run.log","w").write(r.stdout + r.stderr)
print("exit=", r.returncode)

subprocess.run(["python3","log_run.py", desc], check=True)
subprocess.run(["grep","^RESULT","run.log"])
