import subprocess, sys
r = subprocess.run(["uv","run","experiment.py"], capture_output=True, text=True, timeout=540)
open("run.log","w").write(r.stdout + r.stderr)
print("exit=", r.returncode)
