#!/usr/bin/env python3
"""prereg_sorm_seal.py — Generate tamper-evident pre-registration for SORM v1.

Computes:
  hash_a: SHA-256 of sorm_config.yaml bytes (proves parameters not changed)
  hash_b: SHA-256 of src/research/sorm_core.py bytes (proves signal logic not changed)
  hash_c: git HEAD commit SHA (proves document committed before data touched)

Usage:
    python prereg_sorm_seal.py
    git add -f _bmad-output/preregistration_sorm_combine.md
    git commit -m "pre-register SORM combine strategy v1"
"""

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "sorm_config.yaml"
CORE_PATH = ROOT / "src" / "research" / "sorm_core.py"
OUTPUT_PATH = ROOT / "_bmad-output" / "preregistration_sorm_combine.md"


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
        print("WARNING: Working tree is dirty — sealed commit will not match HEAD")

    hash_a = _sha256(CONFIG_PATH)
    hash_b = _sha256(CORE_PATH)
    hash_c = _git_head()
    today = date.today().isoformat()

    # Read config for display (human-readable snapshot)
    config_snapshot = CONFIG_PATH.read_text().strip()

    doc = f"""\
# Pre-Registration: SORM Combine Strategy v1

**Generated:** {today}
**Experiment ID:** sorm-combine-v1

---

## Hypothesis

**H₁ (alternative):** The Session Open Range Mean Reversion strategy generates
positive expectancy on MNQ 1-minute bars: extensions beyond the 09:30–09:44 ET
opening range by ≥ 50% of ORB size (09:45–10:45 ET window) revert to the ORB
midpoint in >55% of cases, and a filtered trade (RSI 30–70, stop ≤ $200/contract)
produces PF ≥ 1.40 with MaxDD ≤ $2,500 out-of-sample.

**H₀ (null):** Extension events do not revert to the ORB midpoint at a rate
above chance; the strategy has no edge (PF ≤ 1.0 OOS, reversion rate ≤ 50%).

---

## Go / No-Go Decision Rules (Pre-committed, Immutable After Seal)

### Gate 0 — Reversion Rate Study (run FIRST, before full backtest)

| Criterion | Threshold | Action |
|---|---|---|
| Reversion rate (extensions → orb_mid) | ≥ 55% | Proceed to full backtest |
| Reversion rate | 50–54% | Reconsider parameters before proceeding |
| Reversion rate | < 50% | STOP — no live code built |

### Gate 1 — In-Sample Backtest

| Criterion | Threshold |
|---|---|
| Win rate | ≥ 55% |
| Profit factor (in-sample) | ≥ 1.40 |
| Max drawdown | ≤ $2,500 (target ≤ $1,500) |
| Frequency | 1.0–3.5 trades/day |
| Risk per trade | $100–$200 avg |
| Minimum trades | N ≥ 150 |

### Gate 2 — OOS Holdout (one-shot, ≥ 2026-03-01)

| Criterion | Threshold |
|---|---|
| OOS profit factor | ≥ 1.40 |
| OOS PF retention vs in-sample | ≥ 80% of in-sample PF |
| Stopping rule | Halt live if PF < 1.10 after first 30 OOS trades |

---

## Strategy Parameters Snapshot

```yaml
{config_snapshot}
```

---

## In-Sample Data Range

- **Development data:** 2025-01-01 → 2026-02-28 (UTC)
- **Data files:**
  - `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
  - `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` (filtered to < 2026-03-01)
- **Sealed holdout (DO NOT TOUCH until Gate 1 passes):** 2026-03-01 → present

---

## Integrity Hashes

| Hash | Path | Value |
|---|---|---|
| (a) sorm_config.yaml SHA-256 | `sorm_config.yaml` | `{hash_a}` |
| (b) sorm_core.py SHA-256 | `src/research/sorm_core.py` | `{hash_b}` |
| (c) Git HEAD commit | — | `{hash_c}` |

*Hash (a): SHA-256 of `sorm_config.yaml` file bytes — proves parameters unchanged.*
*Hash (b): SHA-256 of `src/research/sorm_core.py` source bytes — proves signal logic unchanged.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*

---

## Scope Constraint

This pre-registration covers the **backtest-validation phase only**.
No live ProjectX trader is built until Gate 2 passes.
S25 (tier2_streaming_working.py) continues running unchanged on Topstep account 23884932.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc)
    print(f"SEAL PASS — pre-registration written to {OUTPUT_PATH}")
    print(f"  hash_a (config):      {hash_a[:16]}…")
    print(f"  hash_b (sorm_core):   {hash_b[:16]}…")
    print(f"  hash_c (git HEAD):    {hash_c[:16]}…")
    print()
    print("Next steps:")
    print("  git add -f _bmad-output/preregistration_sorm_combine.md")
    print('  git commit -m "pre-register SORM combine strategy v1"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
