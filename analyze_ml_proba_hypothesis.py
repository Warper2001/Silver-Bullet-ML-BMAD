#!/usr/bin/env python3
"""analyze_ml_proba_hypothesis.py — frozen verdict tool for the ml_proba ordinal hypothesis.

Pre-registration: _bmad-output/preregistration_ml_proba_ordinal.md
DO NOT MODIFY after the pre-registration commit — any change voids the prospective test.

Question: does an ML-gated bot's per-trade `ml_proba` carry ORDINAL information about
realized PnL (do higher-probability trades earn more, in rank), measured PROSPECTIVELY
on trades entered strictly AFTER the seal commit? Rank correlation (Spearman) is the
PRIMARY metric on purpose: the in-sample binary "high vs low" lift is a fat-tail
artifact (in-sample rho=+0.06, p=0.79) that a rank test correctly discounts.

Decision (per model, primary = trader-s26):
  - Need N >= 30 prospective trades.
  - PASS  : Spearman rho(ml_proba, net_pnl) > 0 with one-sided p < 0.05
            AND top-tercile mean net PnL > bottom-tercile mean net PnL.
  - FAIL  : otherwise at N=30.
  - Early bearish stop: ONE interim look at N>=15 — if rho <= 0, HALT and declare FAIL.
  - No restricting to winning days/regimes, no excluding trades, no changing the cut,
    no swapping the metric, no pooling models to rescue a fail.

Usage:
  .venv/bin/python analyze_ml_proba_hypothesis.py \
      --trader trader-s26 --cutoff 2026-06-20 [--db data/trades.db]
"""
import argparse
import sqlite3
import statistics as st
import sys

SEAL_NOTE = "prospective = entry timestamp strictly after the seal; use --cutoff (>=)."


def load(db, trader, cutoff):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in con.execute(
            "SELECT timestamp, pnl, ml_proba FROM trades "
            "WHERE trader_id=? AND ml_proba IS NOT NULL AND pnl IS NOT NULL "
            "AND substr(timestamp,1,10) >= ? ORDER BY timestamp",
            (trader, cutoff),
        )
    ]
    con.close()
    return rows


def tercile_block(rows):
    s = sorted(rows, key=lambda r: r["ml_proba"])
    n = len(s)
    t = n // 3
    bot, top = s[:t], s[2 * t:]
    bot_e = st.mean(r["pnl"] for r in bot) if bot else float("nan")
    top_e = st.mean(r["pnl"] for r in top) if top else float("nan")
    return bot_e, top_e, len(bot), len(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/trades.db")
    ap.add_argument("--trader", default="trader-s26")
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD; trades with entry date >= cutoff count")
    args = ap.parse_args()

    rows = load(args.db, args.trader, args.cutoff)
    n = len(rows)
    print(f"== ml_proba ordinal hypothesis — {args.trader} ==")
    print(f"prospective cutoff (entry date >= ): {args.cutoff}   [{SEAL_NOTE}]")
    print(f"N prospective trades: {n}")
    if n == 0:
        print("No prospective trades yet. PENDING.")
        return

    probas = [r["ml_proba"] for r in rows]
    pnls = [r["pnl"] for r in rows]
    wr = sum(1 for p in pnls if p > 0) / n
    print(f"ml_proba range {min(probas):.3f}-{max(probas):.3f}  WR {wr*100:.0f}%  "
          f"E$/trade {st.mean(pnls):+.1f}  sumPnL {sum(pnls):+.1f}")

    try:
        from scipy.stats import spearmanr
        rho, p_two = spearmanr(probas, pnls)
        # one-sided p for the directional H1 (rho > 0)
        p_one = (p_two / 2) if rho > 0 else (1 - p_two / 2)
    except Exception as e:  # pragma: no cover
        print(f"scipy unavailable ({e}); cannot compute Spearman — install scipy.")
        sys.exit(2)

    bot_e, top_e, nb, nt = tercile_block(rows)
    print(f"Spearman rho={rho:+.3f}  two-sided p={p_two:.3f}  one-sided p(rho>0)={p_one:.3f}")
    print(f"bottom-tercile E$={bot_e:+.1f} (n={nb})   top-tercile E$={top_e:+.1f} (n={nt})")

    # descriptive only (post-hoc 0.63 cut) — NOT the decision metric
    hi = [r for r in rows if r["ml_proba"] >= 0.63]
    lo = [r for r in rows if r["ml_proba"] < 0.63]
    for lbl, g in (("ml_proba>=0.63", hi), ("ml_proba<0.63", lo)):
        if g:
            print(f"  [descriptive] {lbl:<16} n={len(g):>2} E$={st.mean(x['pnl'] for x in g):+7.1f}")

    print("\n-- VERDICT --")
    if n < 15:
        print(f"PENDING — need N>=30 (interim look allowed at N>=15). Have {n}.")
    elif n < 30:
        if rho <= 0:
            print("EARLY-STOP FAIL — at interim look (N>=15) rho<=0; no ordinal information.")
        else:
            print(f"PENDING — interim rho={rho:+.3f}>0, continue to N=30. Have {n}.")
    else:
        passed = (rho > 0) and (p_one < 0.05) and (top_e > bot_e)
        if passed:
            print("PASS — ml_proba carries ordinal information (rho>0, one-sided p<0.05, "
                  "top-tercile E$>bottom). Motivates a SEPARATE config-change prereg to "
                  "DERIVE an optimal threshold on a train/holdout split (do not assert).")
        else:
            print(f"FAIL — rho={rho:+.3f}, one-sided p={p_one:.3f}, "
                  f"top_e>bot_e={top_e>bot_e}. 06-15 was variance/regime; ml_proba is not "
                  "a selectivity dial beyond its current binary gate.")


if __name__ == "__main__":
    main()
