"""Parse RLinf embodied training log -> deterministic-eval success series.

Layout per eval epoch: the timing table row containing `eval=<seconds>` is
followed by the train-rollout metrics table and THEN the eval metrics table;
the LAST `success_once=` within the next 40 lines after the `eval=` row is the
eval value. Prints one float per eval, oldest first, then `epochs=<N>`.
"""
import re
import sys

path = sys.argv[1]
lines = open(path, errors="replace").read().splitlines()
ev_rows = [i for i, ln in enumerate(lines) if re.search(r"eval=\d", ln)]
succ = re.compile(r"success_once=([0-9.]+)")
for i in ev_rows:
    vals = []
    for ln in lines[i : i + 41]:
        vals += succ.findall(ln)
    if vals:
        print(vals[-1])
epochs = sum(1 for ln in lines if "rollout/generate_one_epoch" in ln)
print(f"epochs={epochs}")
