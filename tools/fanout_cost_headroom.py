import csv
import glob
import os
import re

WT = "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/post-r3-options-research/data/reports"
FAN = "/root/.claude/jobs/960bda86/tmp/fanout"
INSTS = ["mnq", "si", "ym", "rty", "hg", "es", "gc", "pl"]

# Measured all-in RT costs where this repo actually measured them; None = never measured.
MEASURED = {"hg": 4.00, "pl": 34.00, "mnq": 6.00}


def pf(p):
    g = sum(x for x in p if x > 0)
    l = -sum(x for x in p if x < 0)
    return g / l if l else float("inf")


def cstar(gross, target):
    lo, hi = 0.0, 1000.0
    if pf(gross) < target:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if pf([g - mid for g in gross]) >= target:
            lo = mid
        else:
            hi = mid
    return lo


print(f"{'inst':5} {'N':>5} {'grossPF':>8} {'gross$/trd':>11} "
      f"{'c*@1.00':>9} {'c*@1.10':>9} {'measured':>9} {'netPF@meas':>11} {'net$/trd':>9}")
for inst in INSTS:
    log = f"{FAN}/{inst}.log"
    m = re.search(r"Trades\s+→\s+(\S+)", open(log).read())
    rows = list(csv.DictReader(open(os.path.join(WT, os.path.basename(m.group(1))))))
    g = [float(r["pnl"]) for r in rows]
    n = len(g)
    avg = sum(g) / n
    c0, c1 = cstar(g, 1.00), cstar(g, 1.10)
    mc = MEASURED.get(inst)
    if mc is not None:
        npf = pf([x - mc for x in g])
        navg = avg - mc
        print(f"{inst:5} {n:5d} {pf(g):8.3f} {avg:11.2f} {c0:9.2f} {c1:9.2f} "
              f"{mc:9.2f} {npf:11.3f} {navg:9.2f}")
    else:
        print(f"{inst:5} {n:5d} {pf(g):8.3f} {avg:11.2f} {c0:9.2f} {c1:9.2f} "
              f"{'—':>9} {'—':>11} {'—':>9}")
