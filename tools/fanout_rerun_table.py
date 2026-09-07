import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

MAIN = "/root/Silver-Bullet-ML-BMAD/data/reports"
WT = "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/post-r3-options-research/data/reports"
FAN = "/root/.claude/jobs/960bda86/tmp/fanout"

INSTS = ["mnq", "si", "ym", "rty", "hg", "es", "gc", "pl"]


def summ(path):
    t = open(path).read()
    def g(pat):
        m = re.search(pat, t)
        return m.group(1) if m else None
    return {
        "n": int(g(r"Total trades\s*:\s*(\d+)") or 0),
        "pf": float(g(r"Profit factor\s*:\s*([\d.]+)") or 0),
        "pnl": float((g(r"Net P&L\s*:\s*\$([+\-\d,\.]+)") or "0").replace(",", "").replace("+", "")),
        "wr": g(r"Win rate\s*:\s*([\d.]+)%"),
    }


print("=== ORIGINAL fan-out runs (2026-06-25/26), identified by N/PF ===")
orig = {}
for p in sorted(glob.glob(f"{MAIN}/backtest_1year_2026062[56]_*.txt")):
    s = summ(p)
    stamp = os.path.basename(p).replace("backtest_1year_", "").replace(".txt", "")
    csvp = p.replace(".txt", ".csv")
    cm = datetime.fromtimestamp(os.path.getmtime(csvp)).strftime("%m-%d %H:%M")
    tm = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
    print(f"  {stamp}: N={s['n']:4d} PF={s['pf']:.3f} pnl=${s['pnl']:>9,.0f} "
          f"| txt {tm}  csv {cm}" + ("   <-- CSV REWRITTEN AFTER TXT" if cm != tm else ""))
    orig[stamp] = (s, csvp)


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["pnl"] = float(r["pnl"])
        r["xd"] = datetime.fromisoformat(r["exit_time"]).date()
    return rows


def stats(rows):
    p = [r["pnl"] for r in rows]
    g = sum(x for x in p if x > 0)
    l = -sum(x for x in p if x < 0)
    pf = g / l if l else float("inf")
    day = defaultdict(float)
    for r in rows:
        day[r["xd"]] += r["pnl"]
    top3 = sum(sorted(day.values(), reverse=True)[:3])
    tot = sum(p)
    tail = (top3 / tot * 100) if tot else float("nan")
    wr = sum(1 for x in p if x > 0) / len(p) * 100
    return len(p), pf, tot, wr, tail, sum(p) - top3


# daily series for correlation
def daily(rows):
    d = defaultdict(float)
    for r in rows:
        d[r["xd"]] += r["pnl"]
    return d


def corr(a, b):
    ks = sorted(set(a) | set(b))
    x = [a.get(k, 0.0) for k in ks]
    y = [b.get(k, 0.0) for k in ks]
    n = len(ks)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    vy = sum((yi - my) ** 2 for yi in y) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


print("\n=== RE-RUN on current code (2026-09-07), same window/mode ===")
newrows = {}
for inst in INSTS:
    log = f"{FAN}/{inst}.log"
    m = re.search(r"Trades\s+→\s+(\S+)", open(log).read())
    path = os.path.join(WT, os.path.basename(m.group(1)))
    newrows[inst] = load(path)

mnq_daily = daily(newrows["mnq"])
print(f"{'inst':5} {'N':>5} {'grossPF':>8} {'gross$':>10} {'WR%':>6} "
      f"{'tail3%':>8} {'ex-top3$':>10} {'corrMNQ':>8}")
res = {}
for inst in INSTS:
    n, pf, tot, wr, tail, ext = stats(newrows[inst])
    c = 1.0 if inst == "mnq" else corr(mnq_daily, daily(newrows[inst]))
    res[inst] = (n, pf, tot, wr, tail, ext, c)
    print(f"{inst:5} {n:5d} {pf:8.3f} {tot:10,.0f} {wr:6.1f} {tail:8.0f} {ext:10,.0f} {c:8.3f}")
