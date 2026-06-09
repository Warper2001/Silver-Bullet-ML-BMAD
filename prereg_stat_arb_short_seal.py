#!/usr/bin/env python3
"""prereg_stat_arb_short_seal.py — Tamper-evident pre-registration for ES/MNQ Stat Arb Short-Only.

Strategy: fade MNQ outperformance of ES (short MNQ when 5-bar cumulative divergence
exceeds +20 pts, targeting reversion to zero divergence, stop at 1× divergence beyond entry).

Data-observation disclosure (mandatory under methodology):
  study_stat_arb_large_div.py was run on in-sample data BEFORE this pre-registration.
  That study measured a strong directional asymmetry:
    Long  (MNQ underperformed ES): WR=47.9%, AvgP&L=-$8.01  (negative EV)
    Short (MNQ overperformed  ES): WR=57.4%, AvgP&L=+$5.84  (positive EV)
  The structural justification (Nasdaq-specific spikes revert; ES-specific moves persist)
  was identified AFTER observing this asymmetry. The short-only Gate 0 diagnostic
  (study_stat_arb_short_only.py) was then run to validate the direction-filtered strategy.
  Both studies used in-sample data. This pre-registration therefore discloses
  direction-selection informed by in-sample data exploration.

Hashes:
  hash_a: SHA-256 of stat_arb_short_config.yaml (proves parameters not changed at backtest time)
  hash_b: SHA-256 of study_stat_arb_short_only.py (proves Gate 0 simulation logic)
  hash_c: git HEAD at pre-registration commit (proves document committed before backtest)

Usage:
    python prereg_stat_arb_short_seal.py
    git add -f _bmad-output/preregistration_stat_arb_short_combine.md \\
               stat_arb_short_config.yaml prereg_stat_arb_short_seal.py \\
               study_stat_arb_large_div.py study_stat_arb_short_only.py
    git commit -m "pre-register ES/MNQ stat arb short-only combine strategy"
"""

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT        = Path(__file__).parent
CONFIG_PATH = ROOT / "stat_arb_short_config.yaml"
STUDY_PATH  = ROOT / "study_stat_arb_short_only.py"
OUTPUT_PATH = ROOT / "_bmad-output" / "preregistration_stat_arb_short_combine.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return True


def main() -> int:
    for path in (CONFIG_PATH, STUDY_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    if _git_is_dirty():
        print("WARNING: Working tree is dirty — commit this document first to make hash_c meaningful.")

    hash_a = _sha256(CONFIG_PATH)
    hash_b = _sha256(STUDY_PATH)
    hash_c = _git_head()
    today  = date.today().isoformat()
    config_snapshot = CONFIG_PATH.read_text().strip()

    doc = f"""\
# Pre-Registration: ES/MNQ Stat Arb — Short-Only Combine Strategy

**Generated:** {today}
**Experiment ID:** stat-arb-short-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure

This strategy was designed and filtered AFTER running diagnostic studies on in-sample data
(2025-05-01 → 2026-02-28). The following findings directly informed parameter choices:

| Finding | Source Study | Impact on Strategy |
|---|---|---|
| 5-bar cumulative ES/MNQ divergence has positive EV at 1:1 R/R before direction split | `study_es_mnq_stat_arb.py` | Confirms the spread-reversion signal exists |
| Strong directional asymmetry: Long WR=47.9% (−$8.01), Short WR=57.4% (+$5.84) at THRESH=20 | `study_stat_arb_large_div.py` | Direction filter — short only |
| Short-only Gate 0 PASS: WR=58.0%, PF=1.27, freq=2.96/d, worst-mo=38.5% | `study_stat_arb_short_only.py` | Confirms feasibility of direction-filtered strategy |
| THRESH=20 at STOP=1.0× is positive EV; STOP=2.0× is negative EV across all thresholds | `study_stat_arb_short_only.py` | Stop multiplier locked at 1.0× |

**Structural justification for short-only direction:**
When MNQ outperforms ES, the driver is typically Nasdaq-specific (tech/AI catalysts,
single-stock moves propagating to the index). These outperformance episodes partially
revert as order flow normalises and the divergence gets arbitraged. When MNQ
underperforms ES, the driver is often ES-specific macro strength (defensive rotation,
energy, value) — MNQ genuinely does not participate, so there is no reversion signal
to fade.

**Consequence:** The in-sample backtest is confirmatory, NOT discovery.
The direction was selected after observing the in-sample asymmetry.
A strong in-sample result is expected and does NOT constitute independent validation.
**The OOS holdout (≥ 2026-03-01) is the primary validity test.**

---

## Hypothesis

**H₁ (alternative):** ES/MNQ stat arb short-only — shorting MNQ when its 5-bar
cumulative return divergence from beta-predicted ES exceeds +20 pts, with a stop at
1× divergence and target at 1× divergence recovery — generates positive expectancy
on MNQ 1-minute RTH bars with WR ≥ 56%, PF ≥ 1.20, frequency ≥ 1.0/day, and
worst-month WR ≥ 35% in the 2025-05-01 → 2026-02-28 in-sample period; and retains
PF ≥ 1.10 OOS.

**H₀ (null):** Fading MNQ outperformance of ES at a fixed divergence threshold has
no edge (PF ≤ 1.0 OOS); the in-sample short-direction edge is a noise artifact of
the exploration that produced the direction-selection decision.

---

## Strategy Logic (Frozen)

| Element | Rule |
|---|---|
| Instruments | MNQ (entry/exit) + ES (signal construction only) |
| Beta estimation | Rolling 60-bar OLS: β = Cov(ΔMNQ, ΔES) / Var(ΔES), clipped [0, 10], forward-filled |
| Divergence | 5-bar cumulative: div = Σ₅ΔMNQ − β × Σ₅ΔES |
| Direction | **SHORT ONLY** — enter when div > +20 pts (MNQ outperformed ES) |
| Entry | Close of the triggering bar (MNQ close price) |
| TP | Entry − divergence (MNQ gives back exactly 1× the divergence) |
| Stop | Entry + 1.0 × divergence (divergence widens further against us) |
| Stop cap | Skip trade if stop distance > $150/contract; enforce at entry |
| Hold max | 30 bars (~30 min); forced market-close if neither TP nor stop hit |
| Session close | 15:55 ET: force-close all open positions at close price |
| Trade sequencing | One trade at a time; no new entry while trade is active |
| RTH only | 09:30–15:55 ET; no overnight or pre-market |
| Daily halt | Halt new entries if session P&L ≤ −$300 for the day |

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 1 — In-Sample Full Backtest (2025-05-01 → 2026-02-28)

Gate 0 has already passed on the primary spec. Gate 1 runs the same simulation with
the full combine-accounting overlay (trailing DD path, qualifying day count, daily halt).

| Criterion | Minimum | Action if below |
|---|---|---|
| Win rate | ≥ 56% | STOP — no OOS access |
| Avg net P&L/trade | > $0 | STOP |
| Profit factor | ≥ 1.20 | STOP |
| Max trailing DD (in-sample) | ≤ $1,500 | STOP |
| Frequency | ≥ 1.0 setups/day | STOP |
| N trades | ≥ 80 | STOP |
| Worst-month WR | ≥ 35% | STOP |
| Qualifying sessions / last 20 | ≥ 6 | WARNING |
| Largest single day as % of total P&L | ≤ 50% | WARNING |

### Gate 2 — OOS Holdout (≥ 2026-03-01, one-shot, requires Gate 1 pass)

| Criterion | Minimum |
|---|---|
| OOS WR | ≥ 53% |
| OOS avg net P&L/trade | > $0 |
| OOS profit factor | ≥ 1.10 |
| OOS PF retention vs in-sample | ≥ 75% |
| N OOS trades | ≥ 20 |

**OOS stopping rule (live, if deployed):** Halt combine trading if PF < 1.05
after first 25 OOS trades.

Accessing the OOS holdout before Gate 1 passes voids this pre-registration.

---

## Strategy Parameters Snapshot

```yaml
{config_snapshot}
```

---

## In-Sample and Holdout Data Ranges

- **Development data (in-sample):** 2025-05-01 → 2026-02-28
  - MNQ: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - MNQ: `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`
  - ES:  `data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv`
- **Sealed holdout (DO NOT ACCESS until Gate 1 passes):** 2026-03-01 → present
  - `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`

Note: ES data is required for signal construction only; MNQ is the traded instrument.
OOS holdout currently contains MNQ only. ES OOS data must be fetched (TradeStation)
before running the OOS backtest.

---

## Integrity Hashes

| Hash | File | Value |
|---|---|---|
| (a) stat_arb_short_config.yaml SHA-256 | `stat_arb_short_config.yaml` | `{hash_a}` |
| (b) study_stat_arb_short_only.py SHA-256 | `study_stat_arb_short_only.py` | `{hash_b}` |
| (c) Git HEAD at seal time | — | `{hash_c}` |

*Hash (a): Proves strategy parameters (threshold, stop, TP, session rules) unchanged.*
*Hash (b): Proves Gate 0 simulation logic (beta estimation, divergence calc, P&L formula) unchanged.*
*Hash (c): Commit this document before any full backtest to prove pre-reg preceded data access.*

---

## Scope Constraint

This pre-registration covers **backtest-validation only**.
No live ProjectX trader, combine account setup, or position-size decision is made
until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.

## Combine Strategy Search Context

This is the **first Gate 0 PASS** after eight failed strategy families:
ORB (4), VWAP Reversion (2), PBC (2), ES/MNQ Stat Arb all-directions, Vol Compression
(1-min + 15-min), Volume Profile POC Fade, Lunch-Window Oscillation.
The direction-asymmetry discovery (long fails, short passes) is the novel structural
finding that separates this candidate from the prior stat arb attempt.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc)
    print(f"SEAL PASS — pre-registration written to {OUTPUT_PATH}")
    print(f"  hash_a (config):  {hash_a[:16]}…")
    print(f"  hash_b (study):   {hash_b[:16]}…")
    print(f"  hash_c (git HEAD): {hash_c[:16]}…")
    print()
    print("Next steps:")
    print("  git add -f _bmad-output/preregistration_stat_arb_short_combine.md \\")
    print("             stat_arb_short_config.yaml prereg_stat_arb_short_seal.py \\")
    print("             study_stat_arb_large_div.py study_stat_arb_short_only.py \\")
    print("             study_lunch_window_oscillation.py study_vol_compression_15min.py")
    print('  git commit -m "pre-register ES/MNQ stat arb short-only combine strategy"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
