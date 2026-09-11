"""Shared per-bar decision log for the Tier-2 family of bots.

WHY THIS MODULE EXISTS
----------------------
Three bots append to ONE file, `logs/tier2_bar_decisions.csv`:
`yank_streaming_working.py`, `tier2_streaming_working.py`, `btc_combine_streaming.py`.
Each carried its own copy of the writer, and the copies drifted. Consequences, all real:

  * Only YANK had the "don't log during startup backfill" guard, so the other two
    re-logged historical bars on every restart. The file reached 24,066,757 rows /
    1.66 GB before anyone noticed (archived + truncated 2026-09-10).
  * No row said WHICH bot wrote it, so the file could not be attributed at all.
  * `action` was one of HOLD/SKIP/ENTER with no reason, so a SKIP collapsed
    "volatility regime blocked", "no sweep", "no CHoCH", "no FVG", "FVG in the wrong
    direction", "LR regime filter rejected" and "ML threshold rejected" into one
    indistinguishable value. Diagnosing YANK's 24-day silence in September 2026
    required grepping a 275 MB text log, because the CSV could not answer it.

One writer, one schema, one guard.

SCHEMA MIGRATION
----------------
`FIELDS` is the canonical header. If the file on disk has a different header (an older
schema, or a newer one from a half-deployed change), it is rotated into `logs/archive/`
and a fresh file is started. That is deliberate: silently appending 10 columns of data
under a 7-column header produces a file that parses without error and means nothing.
"""
from __future__ import annotations

import csv
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repo root: this file is <root>/src/research/decision_log.py
_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = _ROOT / "logs" / "tier2_bar_decisions.csv"
ARCHIVE_DIR = _ROOT / "logs" / "archive"

FIELDS = [
    "bar_timestamp",
    "trader_id",          # added 2026-09-11 -- three bots share this file
    "h1_sweep_active",
    "kill_zone_active",
    "vol_regime_blocked",
    "vol_regime_pct",     # added 2026-09-11 -- the value behind the block, previously
                          # computed and then thrown away; a blocked bar said nothing
                          # about HOW blocked it was
    "m15_confirmed",
    "fvg_detected",
    "action",             # HOLD | SKIP | ENTER
    "rejection_reason",   # added 2026-09-11 -- see REASONS below; "" when action != SKIP
]

# Canonical reason codes. Keep these stable: analyze_filter_funnel.py groups on them.
# Ordered as the gate chain evaluates, so a funnel report reads top-to-bottom.
REASONS = (
    "data_stale",           # feed went stale; no decision possible
    "in_trade",             # already holding a position (action=HOLD)
    "warmup",               # < 20 bars; ATR/volume features not yet computable
    "flatten_window",       # Topstep 15:08-17:00 CT no-new-entry window
    "tuesday",              # day-of-week exclusion
    "daily_breaker",        # daily loss limit reached
    "seasonality",          # month in TIER2_BLOCKED_MONTHS
    "vol_regime",           # H1 ATR above the regime threshold percentile
    "no_sweep_or_choch",    # direction gate: no active H1 sweep and/or no M15 CHoCH
    "no_fvg",               # sweep+CHoCH present, but no qualifying FVG on this bar
    "fvg_wrong_direction",  # FVG found but direction/sweep mismatch (near-miss)
    "lr_regime",            # LR counter-trend regime filter rejected the signal
    "ml_threshold",         # meta-labeling probability below threshold
    "entered",              # not a rejection; action=ENTER
)


def _needs_rotation(path: Path) -> bool:
    """True if the file exists with a header that is not the current FIELDS."""
    try:
        with path.open("r", newline="") as f:
            header = next(csv.reader(f), None)
    except Exception:
        return False  # unreadable -> leave it alone; append will surface the error
    return header is not None and header != FIELDS


def _rotate(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{path.stem}_{stamp}_preschema.csv"
    shutil.move(str(path), str(dest))
    logger.warning(
        "decision log schema changed -- rotated the previous file to %s and started "
        "a fresh one. The old rows are intact there; they use the older column set.",
        dest,
    )


def append_decision(
    *,
    trader_id: str,
    bar_timestamp: datetime,
    action: str,
    rejection_reason: str = "",
    h1_sweep_active: bool = False,
    kill_zone_active: bool = False,
    vol_regime_blocked: bool = False,
    vol_regime_pct: Optional[float] = None,
    m15_confirmed: bool = False,
    fvg_detected: bool = False,
    is_backfill: bool = False,
) -> None:
    """Append one per-bar decision row. Never raises.

    `is_backfill` MUST be passed by every caller. Historical replay bars are re-logged on
    every restart; without this guard the file grows without bound (it reached 1.66 GB).
    The guard lives here so no copy of it can be forgotten again.
    """
    if is_backfill:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _needs_rotation(LOG_PATH):
            _rotate(LOG_PATH)
        write_header = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({
                "bar_timestamp": bar_timestamp.isoformat(),
                "trader_id": trader_id,
                "h1_sweep_active": h1_sweep_active,
                "kill_zone_active": kill_zone_active,
                "vol_regime_blocked": vol_regime_blocked,
                "vol_regime_pct": "" if vol_regime_pct is None else round(float(vol_regime_pct), 4),
                "m15_confirmed": m15_confirmed,
                "fvg_detected": fvg_detected,
                "action": action,
                "rejection_reason": rejection_reason,
            })
    except Exception as exc:  # never let telemetry kill a trading loop
        logger.warning("Filter decision log write failed: %s", exc)
