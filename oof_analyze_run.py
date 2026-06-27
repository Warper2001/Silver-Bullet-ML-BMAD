"""Walk-forward out-of-fold analysis of the SL x TP sweep (in-sample only).

Reports, for an instrument:
  - the full-window SL x TP response surface (in-sample net$ + PF) — context
  - Ceiling A (best-in-hindsight over OOF folds) — the OVERFIT MIRAGE
  - Walk-forward OOF (pick best config on PAST folds, score on next) — HONEST
  - Frozen SL2/TP8 over the same OOF folds — the BASELINE TO BEAT
  - headroom = OOF - Frozen  (>0 ⇒ durable tuning headroom worth a future prereg)

Usage: .venv/bin/python oof_analyze.py <hg|pl>
"""
import sys, csv
from datetime import datetime, timezone
from pathlib import Path

INST = sys.argv[1]
DIR = Path(f"data/reports/oof_sweep/{INST}")
START = datetime(2025, 5, 19, tzinfo=timezone.utc)
END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
K = 5
FROZEN = ("2.0", "8.0")

configs = {}
for f in sorted(DIR.glob("SL*_TP*.csv")):
    sl, tp = f.stem.replace("SL", "").split("_TP")
    rows = [(datetime.fromisoformat(r["entry_time"]), float(r["pnl"])) for r in csv.DictReader(open(f))]
    configs[(sl, tp)] = rows

if not configs:
    print(f"No sweep CSVs in {DIR} yet."); sys.exit(0)

span = END - START
edges = [START + span * i / K for i in range(K + 1)]
def fold_of(dt):
    for i in range(K):
        if edges[i] <= dt < edges[i + 1]:
            return i
    return K - 1

def net(rows, folds):  return sum(p for dt, p in rows if fold_of(dt) in folds)
def n(rows, folds):    return sum(1 for dt, p in rows if fold_of(dt) in folds)
def pf(rows, folds):
    ps = [p for dt, p in rows if fold_of(dt) in folds]
    gp = sum(x for x in ps if x > 0); gl = abs(sum(x for x in ps if x < 0))
    return gp / gl if gl else float("inf")

ALL = set(range(K))
# 1. full-window response surface
print(f"=== {INST.upper()} SL×TP response surface (FULL in-sample window — this is Ceiling A territory, overfit) ===")
sls = sorted({c[0] for c in configs}, key=float); tps = sorted({c[1] for c in configs}, key=float)
print("          " + "".join(f"TP{t:>7}" for t in tps))
for sl in sls:
    cells = []
    for tp in tps:
        r = configs.get((sl, tp))
        cells.append(f"${net(r, ALL):>6,.0f}" if r else "    n/a")
    mark = "  (SL/TP rows; ★=frozen)"
    print(f"  SL{sl:>4}  " + "".join(f"{c:>9}" for c in cells) + (mark if sl == FROZEN[0] else ""))

# 2. Walk-forward OOF (test folds 1..K-1; train = all earlier folds)
print(f"\n=== Walk-forward OOF (test folds 1..{K-1}, train on prior folds only) ===")
oof_net = oof_n = 0
for k in range(1, K):
    train = set(range(k))
    best = max(configs, key=lambda c: net(configs[c], train))
    tnet, tn = net(configs[best], {k}), n(configs[best], {k})
    oof_net += tnet; oof_n += tn
    print(f"  fold {k}: best-on-train = SL{best[0]}/TP{best[1]}  → test net ${tnet:>7,.0f}  (n={tn})")

OOF = set(range(1, K))
fz_net, fz_n, fz_pf = net(configs[FROZEN], OOF), n(configs[FROZEN], OOF), pf(configs[FROZEN], OOF)
ceilA = max(configs, key=lambda c: net(configs[c], OOF))
ceilA_net = net(configs[ceilA], OOF)

print(f"\n=== VERDICT ({INST.upper()}, OOF period = folds 1..{K-1}, in-sample) ===")
print(f"  Walk-forward OOF (honest)      : net ${oof_net:>8,.0f}  n={oof_n}")
print(f"  Frozen SL2/TP8 (baseline)      : net ${fz_net:>8,.0f}  n={fz_n}  PF {fz_pf:.3f}")
print(f"  Ceiling A (best-in-hindsight SL{ceilA[0]}/TP{ceilA[1]}, MIRAGE): net ${ceilA_net:>8,.0f}")
print(f"  --> durable headroom (OOF − frozen): ${oof_net - fz_net:>+8,.0f}")
print(f"  --> overfit gap (CeilingA − OOF)    : ${ceilA_net - oof_net:>+8,.0f}")
if oof_net > fz_net:
    print("  READ: positive OOF headroom → MAY justify a SEPARATE future pre-registered tuning campaign.")
else:
    print("  READ: no OOF headroom over frozen → tuning the exit shape does NOT durably help; keep frozen.")
print("  (Small-sample caveat: ~20 trades/fold; treat as a coarse signal, not a precise estimate.)")
