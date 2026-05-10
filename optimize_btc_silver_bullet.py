#!/usr/bin/env python3
"""BTC Silver Bullet Parameter Optimization.

Grid search over kill zone subsets, target cap R:R, stop width multiplier,
and NY AM time variants. Detection runs once; only execution varies per combo.

Train:   Nov 8, 2024 – Nov 8, 2025 (12 months)
Holdout: Nov 8, 2025 – May 2026    (6 months, touched once after selection)

Output:
  data/reports/btc_optimization_grid_{ts}.csv
  data/reports/btc_optimization_report_{ts}.txt
"""

import sys
import logging
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from backtest_btc_silver_bullet import (
    load_csv_as_bars,
    detect_swing_points,
    detect_mss_events,
    detect_fvg_setups,
    detect_confluence,
    _load_kill_zones,
    _in_kill_zone,
    _cdt_date,
    _find_next_liquidity_pool,
    round_tick,
    BTC_CONTRACT_VALUE,
    COMMISSION,
    POSITION_SIZE,
    LIMIT_CANCEL_BARS,
    MAX_HOLD_BARS,
    MIN_RR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy per-setup debug from base module during grid loop
logging.getLogger("backtest_btc_silver_bullet").setLevel(logging.WARNING)

SPLIT_TS = datetime(2025, 11, 8, tzinfo=timezone.utc)
MIN_TRADES = 30

# Grid axes
NY_AM_VARIANTS = [
    ("09:00-10:00", 9, 0, 10, 0),
    ("09:30-10:00", 9, 30, 10, 0),
    ("09:00-10:30", 9, 0, 10, 30),
]
TARGET_CAPS = [0, 2.0, 3.0, 4.0]   # 0 = uncapped swing pool
STOP_MULTS = [0.5, 0.75, 1.0]       # multiplier on FVG gap size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_active_zones(
    subset_names: list[str],
    base_zones: list[tuple],
    ny_variant: tuple,
) -> list[tuple]:
    """Return kill zone list for a subset, substituting the NY AM variant."""
    _, ny_sh, ny_sm, ny_eh, ny_em = ny_variant
    result = []
    for name, sh, sm, eh, em in base_zones:
        if name not in subset_names:
            continue
        if name == "NY AM":
            result.append(("NY AM", ny_sh, ny_sm, ny_eh, ny_em))
        else:
            result.append((name, sh, sm, eh, em))
    return result


def filter_setups_by_zones(setups: list, kill_zones: list) -> list:
    """Non-mutating kill zone filter; returns new dicts with killzone_window set."""
    result = []
    for setup in setups:
        in_kz, name = _in_kill_zone(setup["timestamp"], kill_zones)
        if in_kz:
            result.append({**setup, "killzone_window": name})
    return result


def execute_param(
    bars: list,
    setups: list,
    stop_mult: float,
    target_cap_rr: float,
) -> list:
    """Parameterized execution — stop_mult and target_cap_rr variants.

    Identical to execute_backtest() in backtest_btc_silver_bullet.py except:
    - stop_loss uses stop_mult instead of hardcoded 0.5
    - target_price is capped at target_cap_rr × stop_distance when > 0
    """
    trades = []
    window_trades: dict[str, set] = {}

    for setup in setups:
        try:
            setup_idx = setup["index"]
            if setup_idx >= len(bars) - LIMIT_CANCEL_BARS - 1:
                continue

            direction = setup["direction"]
            window = setup.get("killzone_window", "unknown")
            fvg_midpoint = round_tick(
                (setup["entry_zone_top"] + setup["entry_zone_bottom"]) / 2
            )

            setup_date = _cdt_date(bars[setup_idx].timestamp)
            fired_dates = window_trades.setdefault(window, set())
            if setup_date in fired_dates:
                continue

            # Wait for FVG touch (limit fill simulation)
            fill_idx = None
            for i in range(
                setup_idx + 1,
                min(setup_idx + LIMIT_CANCEL_BARS + 1, len(bars)),
            ):
                bar = bars[i]
                if direction == "bullish" and bar.low <= fvg_midpoint:
                    fill_idx = i
                    break
                elif direction == "bearish" and bar.high >= fvg_midpoint:
                    fill_idx = i
                    break

            if fill_idx is None:
                continue

            entry_price = fvg_midpoint
            fill_bar = bars[fill_idx]

            gap_size = setup["entry_zone_top"] - setup["entry_zone_bottom"]
            if direction == "bullish":
                stop_loss = round_tick(setup["entry_zone_bottom"] - gap_size * stop_mult)
            else:
                stop_loss = round_tick(setup["entry_zone_top"] + gap_size * stop_mult)

            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                continue

            target_price = _find_next_liquidity_pool(
                bars, fill_idx, direction, entry_price
            )
            if target_price is None:
                continue

            target_price = round_tick(target_price)
            target_distance = abs(target_price - entry_price)
            if target_distance < MIN_RR * stop_distance:
                continue

            # Apply target cap
            if target_cap_rr > 0:
                cap_dist = target_cap_rr * stop_distance
                if direction == "bullish":
                    target_price = min(target_price, round_tick(entry_price + cap_dist))
                else:
                    target_price = max(target_price, round_tick(entry_price - cap_dist))
                target_distance = abs(target_price - entry_price)

            # Guard: fill bar immediately breaches stop
            if direction == "bullish" and fill_bar.low <= stop_loss:
                continue
            if direction == "bearish" and fill_bar.high >= stop_loss:
                continue

            # Bar-by-bar exit simulation
            exit_idx = fill_idx
            exit_price = 0.0
            pnl = 0.0
            exit_reason = None

            for j in range(
                fill_idx + 1,
                min(fill_idx + MAX_HOLD_BARS + 1, len(bars)),
            ):
                exit_bar = bars[j]
                if direction == "bullish":
                    if exit_bar.low <= stop_loss:
                        pnl = (stop_loss - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif exit_bar.high >= target_price:
                        pnl = (target_price - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break
                else:
                    if exit_bar.high >= stop_loss:
                        pnl = (entry_price - stop_loss) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif exit_bar.low <= target_price:
                        pnl = (entry_price - target_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break

                if j - fill_idx >= MAX_HOLD_BARS:
                    if direction == "bullish":
                        pnl = (exit_bar.close - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                    else:
                        pnl = (entry_price - exit_bar.close) * POSITION_SIZE * BTC_CONTRACT_VALUE
                    exit_price = exit_bar.close
                    exit_reason = "time_exit"
                    exit_idx = j
                    break

            if exit_reason is None:
                continue

            pnl -= COMMISSION
            fired_dates.add(setup_date)

            trades.append({
                "pnl": pnl,
                "exit_reason": exit_reason,
                "killzone_window": window,
                "bars_held": exit_idx - fill_idx,
            })

        except Exception as exc:
            logger.debug(f"Trade error: {exc}")
            continue

    return trades


def quick_metrics(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "pnl": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    return {
        "trades": len(trades),
        "pf": gp / gl if gl > 0 else float("inf"),
        "wr": len(wins) / len(trades),
        "pnl": sum(pnls),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("BTC Silver Bullet Parameter Optimization")
    logger.info(f"Train: Nov 2024 – Nov 8 2025 | Holdout: Nov 8 2025 – end")

    base_zones = _load_kill_zones("config_kraken.yaml")
    zone_names = [z[0] for z in base_zones]

    # ── Load data ────────────────────────────────────────────────────────────
    bars = load_csv_as_bars("data/kraken/PF_XBTUSD_1min.csv")
    if len(bars) < 1000:
        logger.error("Insufficient data")
        return

    # ── Detect once on all bars ───────────────────────────────────────────
    logger.info("Running pattern detection (runs once)...")
    swing_highs, swing_lows = detect_swing_points(bars)
    mss_events = detect_mss_events(bars, swing_highs, swing_lows)
    fvg_setups = detect_fvg_setups(bars)
    all_setups = detect_confluence(mss_events, fvg_setups)

    # ── Split by timestamp ────────────────────────────────────────────────
    train_setups = [s for s in all_setups if s["timestamp"] < SPLIT_TS]
    holdout_setups = [s for s in all_setups if s["timestamp"] >= SPLIT_TS]
    logger.info(
        f"Train setups: {len(train_setups):,} | Holdout setups: {len(holdout_setups):,}"
    )

    # ── Build grid ────────────────────────────────────────────────────────
    window_subsets = [
        list(combo)
        for r in range(1, len(zone_names) + 1)
        for combo in combinations(zone_names, r)
    ]
    total = len(window_subsets) * len(NY_AM_VARIANTS) * len(TARGET_CAPS) * len(STOP_MULTS)
    logger.info(f"Grid: {total} combinations — this may take 15–30 min...")

    results = []

    for subset_names in tqdm(window_subsets, desc="Subsets"):
        ny_am_in_subset = "NY AM" in subset_names
        for ny_variant in NY_AM_VARIANTS:
            ny_label = ny_variant[0]
            active_zones = build_active_zones(subset_names, base_zones, ny_variant)
            train_filtered = filter_setups_by_zones(train_setups, active_zones)

            for target_cap in TARGET_CAPS:
                for stop_mult in STOP_MULTS:
                    trades = execute_param(
                        bars, train_filtered, stop_mult, target_cap
                    )
                    m = quick_metrics(trades)
                    results.append({
                        "windows": "+".join(subset_names),
                        "ny_am_variant": ny_label if ny_am_in_subset else "N/A",
                        "target_cap_rr": target_cap,
                        "stop_mult": stop_mult,
                        "train_trades": m["trades"],
                        "train_pf": round(m["pf"], 4) if m["pf"] != float("inf") else 9999.0,
                        "train_wr": round(m["wr"], 4),
                        "train_pnl": round(m["pnl"], 2),
                        "holdout_trades": None,
                        "holdout_pf": None,
                        "holdout_wr": None,
                        "holdout_pnl": None,
                        "qualified": m["trades"] >= MIN_TRADES,
                    })

    df = pd.DataFrame(results)
    qualified = df[df["qualified"]]

    if qualified.empty:
        logger.error(
            f"No combos produced ≥ {MIN_TRADES} trades on train set — "
            "check kill zone config or widen parameters."
        )
        return

    logger.info(f"Qualified combos: {len(qualified)} / {total}")

    # ── Select best by train PF ───────────────────────────────────────────
    best_idx = qualified["train_pf"].idxmax()
    best = df.loc[best_idx]
    logger.info(
        f"Best: {best['windows']} | NY={best['ny_am_variant']} | "
        f"cap={best['target_cap_rr']} | stop={best['stop_mult']} | "
        f"train_PF={best['train_pf']:.3f}"
    )

    # ── Holdout validation (best config only) ────────────────────────────
    best_subset = best["windows"].split("+")
    ny_variant_used = next(
        (v for v in NY_AM_VARIANTS if v[0] == best["ny_am_variant"]),
        NY_AM_VARIANTS[0],
    )
    best_zones = build_active_zones(best_subset, base_zones, ny_variant_used)
    holdout_filtered = filter_setups_by_zones(holdout_setups, best_zones)
    holdout_trades = execute_param(
        bars, holdout_filtered, float(best["stop_mult"]), float(best["target_cap_rr"])
    )
    hm = quick_metrics(holdout_trades)

    df.loc[best_idx, "holdout_trades"] = hm["trades"]
    df.loc[best_idx, "holdout_pf"] = round(hm["pf"], 4) if hm["pf"] != float("inf") else 9999.0
    df.loc[best_idx, "holdout_wr"] = round(hm["wr"], 4)
    df.loc[best_idx, "holdout_pnl"] = round(hm["pnl"], 2)

    # ── Write outputs ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"btc_optimization_grid_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Grid CSV: {csv_path}")

    # ── Text report ──────────────────────────────────────────────────────
    top10 = qualified.sort_values("train_pf", ascending=False).head(10)

    lines = [
        "=" * 72,
        "BTC SILVER BULLET PARAMETER OPTIMIZATION REPORT",
        f"Train: Nov 2024 – Nov 8 2025 | Holdout: Nov 8 2025 – May 2026",
        f"Grid: {total} combos | Qualified (≥{MIN_TRADES} trades): {len(qualified)}",
        "=" * 72,
        "",
        "TOP 10 TRAIN CONFIGURATIONS",
        "-" * 72,
    ]
    for _, row in top10.iterrows():
        pf_str = f"{row['train_pf']:.3f}" if row["train_pf"] < 999 else "inf"
        lines.append(
            f"  {row['windows']:<32} NY={row['ny_am_variant']:<14} "
            f"cap={str(row['target_cap_rr']):<4} stop={row['stop_mult']:.2f} | "
            f"PF={pf_str}  WR={row['train_wr']:.1%}  "
            f"N={row['train_trades']:>4}  P&L=${row['train_pnl']:>9,.2f}"
        )

    hm_pf_str = f"{hm['pf']:.3f}" if hm["pf"] < 999 else "inf"
    lines += [
        "",
        "=" * 72,
        "BEST CONFIG — HOLDOUT VALIDATION (OUT-OF-SAMPLE)",
        "=" * 72,
        f"  Windows:      {best['windows']}",
        f"  NY AM window: {best['ny_am_variant']}",
        f"  Target cap:   {'uncapped' if best['target_cap_rr'] == 0 else str(best['target_cap_rr']) + 'R'}",
        f"  Stop mult:    {best['stop_mult']}×gap",
        "",
        f"  TRAIN   | N={best['train_trades']:>4}  PF={best['train_pf']:.3f}"
        f"  WR={best['train_wr']:.1%}  P&L=${best['train_pnl']:>9,.2f}",
        f"  HOLDOUT | N={hm['trades']:>4}  PF={hm_pf_str}"
        f"  WR={hm['wr']:.1%}  P&L=${hm['pnl']:>9,.2f}",
        "",
    ]

    if hm["trades"] < MIN_TRADES:
        lines.append(
            "  WARNING: Holdout has insufficient trades — possible overfit to train period."
        )
    elif hm["pf"] >= 1.5:
        lines.append(
            "  RECOMMENDATION: Config holds up OOS — proceed to ML training with these params."
        )
    elif hm["pf"] >= 1.2:
        lines.append(
            "  RECOMMENDATION: Moderate OOS edge — acceptable foundation for ML training."
        )
    else:
        lines.append(
            "  CAUTION: OOS profit factor < 1.2 — review before committing to ML training."
        )

    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report)

    txt_path = out_dir / f"btc_optimization_report_{ts}.txt"
    txt_path.write_text(report)
    logger.info(f"Report: {txt_path}")


if __name__ == "__main__":
    main()
