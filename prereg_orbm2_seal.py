#!/usr/bin/env python3
"""prereg_orbm2_seal.py — Generate tamper-evident pre-registration for ORBM-2.

ORBM-2 (ORB Breakout Momentum v2) enters IN THE DIRECTION of the ORB extension.
This is the inverse of SORM (which faded extensions) and ORBM-1 (which entered
against the fade direction by mistake). It is informed by — but distinct from —
the Phase A diagnostic studies.

Computes:
  hash_a: SHA-256 of orbm2_config.yaml bytes (proves parameters not changed)
  hash_b: SHA-256 of src/research/sorm_core.py bytes (proves shared signal logic not changed)
  hash_c: git HEAD commit SHA (proves document committed before backtest data touched)

Data-observation disclosure (mandatory under methodology):
  Phase A studies (study_orb_reversion_rate.py, study_orb_control_window.py,
  study_orb_continuation_target.py, study_orb_noise_stop_rate.py) were run on
  in-sample data (2025-01-01 → 2026-02-28) BEFORE this pre-registration.
  Findings that informed ORBM-2 design:
    - 74.2% ORB continuation rate (vs. 56.8% control window; +17.4 ppt ORB-specific)
    - Opposite-boundary stop geometry killed 81% of setups (median $295/contract)
    - NSR=44% with ORBM-1 v2 stop; SHORT direction 56% WR vs. LONG 29% WR
    - Lowering threshold to 0.25× expected to raise frequency ~1.5-2.5/session
  ORBM-2 is therefore a HYPOTHESIS GENERATED FROM in-sample exploration.
  The in-sample backtest is confirmatory, NOT discovery. OOS holdout is the
  primary validity test.

Usage:
    python prereg_orbm2_seal.py
    git add -f _bmad-output/preregistration_orbm2_combine.md orbm2_config.yaml
    git commit -m "pre-register ORBM-2 combine strategy"
"""

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "orbm2_config.yaml"
CORE_PATH   = ROOT / "src" / "research" / "sorm_core.py"
OUTPUT_PATH = ROOT / "_bmad-output" / "preregistration_orbm2_combine.md"


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
    for path in (CONFIG_PATH, CORE_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    if _git_is_dirty():
        print("WARNING: Working tree is dirty — commit this document first to make hash_c meaningful")

    hash_a = _sha256(CONFIG_PATH)
    hash_b = _sha256(CORE_PATH)
    hash_c = _git_head()
    today = date.today().isoformat()

    config_snapshot = CONFIG_PATH.read_text().strip()

    doc = f"""\
# Pre-Registration: ORBM-2 Combine Strategy

**Generated:** {today}
**Experiment ID:** orbm2-combine-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure

This strategy was designed AFTER running Phase A diagnostic studies on in-sample data
(2025-01-01 → 2026-02-28). The following findings from in-sample exploration directly
informed the ORBM-2 parameter choices:

| Finding | Source | Impact on ORBM-2 |
|---|---|---|
| 74.2% ORB continuation rate | `study_orb_control_window.py` | Enter WITH extension (not against it) |
| +17.4 ppt ORB-specific excess over control | `study_orb_control_window.py` | Confirms ORB context is load-bearing |
| Opposite-boundary stop: 81% setups skipped, median $295/contract | `study_orb_continuation_target.py` | Stop at ORB boundary (0.25×ORB_size from entry) instead |
| NSR=44% with ORBM-1 v2; SHORT WR=56%, LONG WR=29% | `study_orb_noise_stop_rate.py` | Symmetric entry with stop at boundary; lower threshold for frequency |
| 0.25× threshold expected frequency: ~1.5–2.5/session | Design inference | extension_threshold: 0.25 |

**Consequence:** The in-sample backtest is confirmatory, not discovery. A strong
in-sample result is expected and does NOT constitute independent validation.
**The OOS holdout (≥ 2026-03-01) is the primary validity test.**

---

## Hypothesis

**H₁ (alternative):** ORBM-2 — entering IN THE DIRECTION of ORB extensions at
0.25×ORB_size threshold, stopping at the ORB boundary, and taking profit at 1.5R —
generates positive expectancy on MNQ 1-minute bars with frequency ≥ 1.0 setups/day,
WR ≥ 55%, PF ≥ 1.40, and MaxDD ≤ $1,500 in the 2025-01-01 → 2026-02-28
in-sample period; and retains PF ≥ 1.30 OOS.

**H₀ (null):** Continuation trades at a tight ORB-boundary stop have no edge
(PF ≤ 1.0 OOS); the in-sample result is a noise artifact of the explored threshold.

---

## Strategy Logic (Frozen)

| Element | Rule |
|---|---|
| ORB window | 09:30–09:44 ET (bars starting before 09:45) |
| Extension threshold | Close ≥ 0.25 × ORB_size beyond boundary |
| Extension window | 09:45–10:45 ET (no new entries after) |
| Entry direction | LONG for upward extension; SHORT for downward extension |
| Entry price | Extension bar's close (market order at next bar open equivalent) |
| Stop | ORB boundary ± 1 tick (structural invalidation) |
| Stop distance | ≈ 0.25 × ORB_size from entry |
| Stop cap | Skip trade if stop > 75 pts ($150/contract) |
| Take profit | 1.5R from entry in continuation direction (single target) |
| Hard close | 11:30 ET — all positions closed at market |
| Max trades/session | 1 (first qualifying extension only) |
| Daily loss halt | Halt new signals if session P&L ≤ -$200 |
| Daily profit halt | Halt new signals if session P&L ≥ +$750 |
| ORB minimum size | Skip sessions with ORB < 5 pts (no real range) |
| Sizing — small stop | stop_pts < 50 → 2 contracts |
| Sizing — large stop | stop_pts 50–75 → 1 contract |
| Sizing — skip | stop_pts > 75 → no trade |

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 1 — In-Sample Backtest (2025-01-01 → 2026-02-28)

| Criterion | Minimum | Target | Action if below minimum |
|---|---|---|---|
| Win rate | ≥ 55% | ≥ 60% | STOP — no OOS access |
| Profit factor | ≥ 1.40 | ≥ 1.80 | STOP |
| Max backtest drawdown | ≤ $1,500 | ≤ $1,000 | STOP |
| Frequency | ≥ 1.0 setups/day | ≥ 1.5/day | STOP |
| N trades | ≥ 80 | ≥ 120 | STOP |
| Qualifying sessions / 20 | ≥ 6 | ≥ 8 | STOP |
| Largest day as % of total P&L | ≤ 40% | ≤ 30% | WARNING |
| Per-trade Sharpe | ≥ 0.20 | ≥ 0.30 | WARNING |

### Gate 2 — OOS Holdout (≥ 2026-03-01, one-shot, requires Gate 1 pass)

| Criterion | Minimum | Target |
|---|---|---|
| OOS profit factor | ≥ 1.30 | ≥ 1.50 |
| OOS PF retention vs in-sample | ≥ 75% | ≥ 85% |
| OOS win rate | ≥ 52% | ≥ 56% |
| N OOS trades | ≥ 20 | — |

**OOS stopping rule (live, if deployed):** Halt combine account trading if PF < 1.10
after first 25 OOS trades.

---

## Strategy Parameters Snapshot

```yaml
{config_snapshot}
```

---

## In-Sample and Holdout Data Ranges

- **Development data (in-sample):** 2025-01-01 → 2026-02-28 (UTC)
- **Data files:**
  - `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` (filtered to < 2026-03-01)
- **Sealed holdout (DO NOT TOUCH until Gate 1 passes):** 2026-03-01 → present

Accessing holdout before Gate 1 passes voids this pre-registration.

---

## Integrity Hashes

| Hash | Path | Value |
|---|---|---|
| (a) orbm2_config.yaml SHA-256 | `orbm2_config.yaml` | `{hash_a}` |
| (b) sorm_core.py SHA-256 | `src/research/sorm_core.py` | `{hash_b}` |
| (c) Git HEAD at seal time | — | `{hash_c}` |

*Hash (a): Proves config parameters unchanged between pre-registration and backtest run.*
*Hash (b): Proves shared signal logic (ORB build, extension detect) unchanged.*
*Hash (c): Commit this document first; then `git rev-parse HEAD` in the backtest script confirms pre-reg preceded data access.*

---

## Scope Constraint

This pre-registration covers the **backtest-validation phase only**.
No live ProjectX trader or combine account trading is built until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc)
    print(f"SEAL PASS — pre-registration written to {OUTPUT_PATH}")
    print(f"  hash_a (orbm2_config):  {hash_a[:16]}…")
    print(f"  hash_b (sorm_core):     {hash_b[:16]}…")
    print(f"  hash_c (git HEAD):      {hash_c[:16]}…")
    print()
    print("WARNING: Working tree is dirty. Commit BEFORE running any backtest.")
    print()
    print("Next steps:")
    print("  git add -f _bmad-output/preregistration_orbm2_combine.md orbm2_config.yaml prereg_orbm2_seal.py")
    print('  git commit -m "pre-register ORBM-2 combine strategy"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
