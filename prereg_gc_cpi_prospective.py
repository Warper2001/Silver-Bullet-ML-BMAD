#!/usr/bin/env python3
"""prereg_gc_cpi_prospective.py — Tamper-evident pre-registration for GC/MGC CPI post-catalyst prospective test.

Strategy: after each monthly CPI release (08:30 ET), wait 10 minutes for the initial
spike to settle, then enter MGC in the direction of gold's net move if the move exceeds
2× ATR(14). Stop at 1.5×ATR, target at 2:1 R/R.

Data-observation disclosure (MANDATORY — methodology requires full transparency):
  study_gc_post_catalyst.py was run on HISTORICAL in-sample data (2025-05-01 → 2026-05-19)
  before this pre-registration was written. That study measured a dramatic event-type split:

    CPI  (N=7):  WR=57.1%, PF=1.77, avg P&L=+$22.37  ← POSITIVE EDGE
    NFP  (N=8):  WR=25.0%, PF=0.30, avg P&L=-$39.54  ← LOSING
    FOMC (N=6):  WR=33.3%, PF=0.50, avg P&L=-$28.60  ← LOSING
    All  (N=21): WR=38.1%, PF=0.668, avg P&L=-$15.78 ← FAIL

  The "CPI only" filter was selected AFTER observing this in-sample split.
  This is a post-hoc subgroup selection. The in-sample result (PF=1.77) cannot be
  treated as independent validation — it is the OBSERVATION that motivated this test.

  Consequence: the prospective test is the PRIMARY validity check.
  A strong prospective result would provide genuine independent evidence.
  A weak or negative result means CPI was a noise artefact.

  The exact parameters (WAIT=10, MIN_MOVE=2×ATR, SM=1.5×, TP=2:1) were declared as
  the "primary spec" BEFORE running study_gc_post_catalyst.py. They were NOT tuned to
  optimise the CPI subgroup. This is the only methodological protection this test has
  against overfitting.

Hashes:
  hash_a: SHA-256 of gc_cpi_config.yaml (proves parameters not changed after pre-reg)
  hash_b: SHA-256 of study_gc_post_catalyst.py (proves in-sample logic unchanged)
  hash_c: git HEAD at pre-registration commit (proves doc committed before any prospective data)

Usage:
    python prereg_gc_cpi_prospective.py
    git add -f _bmad-output/preregistration_gc_cpi_prospective.md \\
               gc_cpi_config.yaml prereg_gc_cpi_prospective.py \\
               study_gc_post_catalyst.py track_gc_cpi_event.py \\
               data/reports/gc_cpi_prospective_log.csv
    git commit -m "pre-register GC/MGC CPI post-catalyst prospective test"
"""
import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT        = Path(__file__).parent
CONFIG_PATH = ROOT / "gc_cpi_config.yaml"
STUDY_PATH  = ROOT / "study_gc_post_catalyst.py"
OUTPUT_PATH = ROOT / "_bmad-output" / "preregistration_gc_cpi_prospective.md"


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
# Pre-Registration: GC/MGC CPI Post-Catalyst — Prospective Test

**Generated:** {today}
**Experiment ID:** gc-cpi-prospective-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure (CRITICAL — read before interpreting any result)

This test was designed **after** running `study_gc_post_catalyst.py` on historical
in-sample data (2025-05-01 → 2026-05-19). The following observations directly
motivated the "CPI only" filter:

| Event | N | WR | PF | Avg P&L | Decision |
|---|---|---|---|---|---|
| CPI | 7 | 57.1% | 1.77 | +$22 | Motivates this test |
| NFP | 8 | 25.0% | 0.30 | −$40 | Excluded (losing) |
| FOMC | 6 | 33.3% | 0.50 | −$29 | Excluded (losing) |
| **All (primary spec)** | **21** | **38.1%** | **0.668** | **−$16** | **Gate 0 FAIL** |

**The CPI subgroup was selected POST-HOC.** The in-sample CPI PF of 1.77 (N=7)
has ZERO independent validity. Any backtest that reproduces this result merely
confirms what was already observed — it adds no information.

**This pre-registration is the one mechanism that creates genuine validation.**
If the prospective PF (on events that occurred AFTER this commit date) exceeds
the thresholds below, that constitutes independent evidence of edge.
If it does not, CPI was a noise artefact of N=7 in-sample.

**The only methodological protection in this test:** the exact parameters
(WAIT=10, MIN_MOVE=2×ATR, SM=1.5×, TP=2:1) were declared as the "primary spec"
BEFORE running the historical study. They were not tuned to optimise the CPI
subgroup. This is explicitly documented in `study_gc_post_catalyst.py` docstring.

---

## Hypothesis

**H₁ (alternative):** Monthly CPI releases create a directional gold price move
that persists for at least 2 hours and generates positive expectancy when entered
10 minutes after the release, provided the initial move exceeds 2× ATR(14).
Specifically: prospective PF > 1.20, EV > $0, WR ≥ 40% over N ≥ 10 qualifying events.

**H₀ (null):** The in-sample CPI PF of 1.77 is a noise artefact of N=7 post-hoc
selection; prospective PF ≤ 1.0.

---

## Strategy Logic (Frozen — do NOT adjust after any prospective event)

| Element | Rule |
|---|---|
| Event scope | **CPI only** (monthly BLS Consumer Price Index release) |
| Release time | 08:30 ET (standard BLS schedule) |
| Pre-event reference | Close of 1-min GC bar immediately before 08:30 ET |
| Wait period | 10 bars = 10 minutes after 08:30 ET bar |
| Direction | Sign of (close[T+10] − close[T−1]): LONG if positive, SHORT if negative |
| Qualification | Skip if |net move| < 2.0 × ATR(14) at T+10 bar |
| Entry | Close of the 10th 1-min bar after 08:30 ET |
| ATR | Rolling 14-bar mean of (high − low) on GC 1-min bars (full session, not RTH-only) |
| Stop | 1.5 × ATR(14) at entry bar, clamped to $150/contract (= 15 pts on MGC) |
| TP | 2.0 × stop_dist in entry direction |
| Hold max | 120 bars (2 hours); close at market if neither TP nor stop hit |
| No session close | GC trades 23h/day; no forced session exit (except HOLD_MAX) |
| Contract | 1 MGC (Micro Gold, $10/pt, 10 troy oz) |
| Commission | $4.80 round-trip |

---

## Prospective Tracking Protocol

### Which events are in scope

All **CPI releases on or after the date of the pre-registration commit** that:
- Are released at 08:30 ET by the BLS
- Have GC 1-min data available for the event time and the following 130 bars

Events before the pre-registration commit date are **in-sample** and are excluded
from the prospective count regardless of outcome.

### How to log each event

After each CPI release:
1. Download fresh GC 1-min data covering the event date plus 3 hours
   (via `download_gc_1min.py` with updated end date and contract)
2. Run: `.venv/bin/python track_gc_cpi_event.py --date YYYY-MM-DD`
   The script will verify the config hash, compute the trade, and append to
   `data/reports/gc_cpi_prospective_log.csv`
3. Commit the updated log: `git add data/reports/gc_cpi_prospective_log.csv && git commit`
4. Check the early stopping rule (printed by the tracker)

### No adjustments allowed

After any prospective event is observed, the following are PROHIBITED:
- Changing WAIT_BARS, MIN_MOVE_ATR, STOP_ATR_MULT, TP_MULT, or any other parameter
- Restricting to a direction (long/short) based on what has been observed
- Excluding events retroactively
- "Testing" alternative parameters on the same prospective events

Any such change voids this pre-registration. Start a new one.

---

## Decision Rules (Pre-committed, Immutable After Seal)

### Early Stopping Rule (bearish)

After **N_qualifying ≥ 5** prospective events:
- If running prospective PF < **0.70** → HALT, declare FAIL, do not wait for N=10
- Strategy is clearly not working and continued testing wastes live capital

### Final Verdict (after N_qualifying = 10)

| Criterion | Threshold | Action if not met |
|---|---|---|
| N qualifying events | ≥ 10 | Continue tracking |
| Profit factor | ≥ 1.20 | FAIL — no combine deployment |
| Avg net P&L/trade | > $0 | FAIL |
| Win rate | ≥ 40.0% | FAIL |

**If all thresholds met:** PASS. The CPI post-catalyst strategy may advance to
full combine backtest (Gate 1) and, if that passes, a sealed OOS test (Gate 2).

**If any threshold missed:** FAIL. The CPI subgroup was a noise artefact. Record
the result and update memory. Do not re-run on another subgroup of CPI events.

### Combine deployment gate (only if prospective PASS)

Gate 1 (if prospective PASS): full combine accounting overlay backtest on all
historical + prospective GC data, applying trailing DD tracking, daily halt,
qualifying-day count, consistency cap. Minimum: MaxDD ≤ $2,000.

Gate 2 (if Gate 1 PASS): one-shot prospective OOS test — forward 20 CPI events
collected after Gate 1 pre-registration, subject to the same Go/No-Go rules.

---

## Strategy Parameters Snapshot

```yaml
{config_snapshot}
```

---

## In-Sample Context

| Period | N events | N qualifying | WR | PF | Avg P&L |
|---|---|---|---|---|---|
| 2025-05-01 → 2026-05-19 (historical) | 13 CPI | 7 | 57.1% | 1.77 | +$22.37 |

**This table is the observation that motivated the test, not validation.**
The next CPI event after the pre-registration commit begins the prospective count.

---

## Integrity Hashes

| Hash | File | Value |
|---|---|---|
| (a) gc_cpi_config.yaml SHA-256 | `gc_cpi_config.yaml` | `{hash_a}` |
| (b) study_gc_post_catalyst.py SHA-256 | `study_gc_post_catalyst.py` | `{hash_b}` |
| (c) Git HEAD at seal time | — | `{hash_c}` |

*Hash (a): Proves parameters unchanged between pre-registration and any prospective event.*
*Hash (b): Proves the in-sample simulation logic that motivated this test.*
*Hash (c): Any prospective CPI event that occurred before this commit is excluded.*

---

## Scope Constraint

This pre-registration covers prospective tracking only. No live MGC trading on
a funded combine account until Gate 1 and Gate 2 pass. S25
(tier2_streaming_working.py on account 23884932) continues unchanged.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc)
    print(f"SEAL PASS — pre-registration written to {OUTPUT_PATH}")
    print(f"  hash_a (config):  {hash_a[:16]}…")
    print(f"  hash_b (study):   {hash_b[:16]}…")
    print(f"  hash_c (git HEAD): {hash_c[:16]}…")
    print()
    print("Next steps:")
    print("  git add -f _bmad-output/preregistration_gc_cpi_prospective.md \\")
    print("             gc_cpi_config.yaml prereg_gc_cpi_prospective.py \\")
    print("             study_gc_post_catalyst.py track_gc_cpi_event.py \\")
    print("             data/reports/gc_cpi_prospective_log.csv")
    print('  git commit -m "pre-register GC/MGC CPI post-catalyst prospective test"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
