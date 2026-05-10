#!/usr/bin/env python3
"""BTC Silver Bullet ML Training Pipeline.

Optimized config: NY AM 09:00-10:00 CDT, stop_mult=0.75, uncapped target.
Train: Nov 2024 – Nov 8 2025 | Holdout: Nov 8 2025 – end.

Outputs: models/btc_xgb/model.joblib, threshold.json, feature_importance.csv
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from backtest_btc_silver_bullet import (
    COMMISSION,
    LIMIT_CANCEL_BARS,
    MAX_HOLD_BARS,
    MIN_RR,
    BTC_CONTRACT_VALUE,
    POSITION_SIZE,
    _cdt_date,
    _find_next_liquidity_pool,
    detect_confluence,
    detect_fvg_setups,
    detect_mss_events,
    detect_swing_points,
    load_csv_as_bars,
    round_tick,
)
from optimize_btc_silver_bullet import filter_setups_by_zones

# ── Constants ──────────────────────────────────────────────────────────────────
SPLIT_TS = datetime(2025, 11, 8, tzinfo=timezone.utc)
STOP_MULT = 0.75
NY_AM_ZONE = [("NY AM", 9, 0, 10, 0)]

DATA_PATH = "data/kraken/PF_XBTUSD_1min.csv"
MODEL_DIR = Path("models/btc_xgb")
MIN_TRAIN_TRADES = 100
MIN_AUC = 0.52
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.05)
MIN_FILTERED_PER_FOLD = 10

VOL_REGIME_LOOKBACK = 252   # calendar days lookback for ATR% percentile (BTC trades 24/7)
VOL_REGIME_LOW = 0.20       # below this percentile → "low" (skip)
VOL_REGIME_HIGH = 0.80      # above this percentile → "extreme" (skip)
VOL_REGIME_MIN_HISTORY = 30 # days needed before filtering begins

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("backtest_btc_silver_bullet").setLevel(logging.WARNING)
logging.getLogger("optimize_btc_silver_bullet").setLevel(logging.WARNING)


# ── CDT helper ─────────────────────────────────────────────────────────────────

def _cdt(ts_utc: datetime) -> datetime:
    return ts_utc + timedelta(hours=-5)


# ── Execution: returns full trade context for feature extraction ───────────────

def execute_for_ml(
    bars: list,
    setups: list,
    vol_regime_map: dict | None = None,
) -> list:
    """execute_param(stop_mult=0.75, target_cap_rr=0) with full trade metadata.

    vol_regime_map: optional {cdt_date_str: 'low'|'medium'|'extreme'}.
    When provided, setups on non-'medium' days are skipped.
    """
    trades = []
    window_trades: dict = {}

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

            if vol_regime_map is not None:
                if vol_regime_map.get(setup_date, "medium") != "medium":
                    continue

            fired_dates = window_trades.setdefault(window, set())
            if setup_date in fired_dates:
                continue

            fill_idx = None
            for i in range(setup_idx + 1, min(setup_idx + LIMIT_CANCEL_BARS + 1, len(bars))):
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
                stop_loss = round_tick(setup["entry_zone_bottom"] - gap_size * STOP_MULT)
            else:
                stop_loss = round_tick(setup["entry_zone_top"] + gap_size * STOP_MULT)

            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                continue

            target_price = _find_next_liquidity_pool(bars, fill_idx, direction, entry_price)
            if target_price is None:
                continue

            target_price = round_tick(target_price)
            target_distance = abs(target_price - entry_price)
            if target_distance < MIN_RR * stop_distance:
                continue

            if direction == "bullish" and fill_bar.low <= stop_loss:
                continue
            if direction == "bearish" and fill_bar.high >= stop_loss:
                continue

            exit_idx = fill_idx
            exit_price = 0.0
            pnl = 0.0
            exit_reason = None

            for j in range(fill_idx + 1, min(fill_idx + MAX_HOLD_BARS + 1, len(bars))):
                eb = bars[j]
                if direction == "bullish":
                    if eb.low <= stop_loss:
                        pnl = (stop_loss - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif eb.high >= target_price:
                        pnl = (target_price - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break
                else:
                    if eb.high >= stop_loss:
                        pnl = (entry_price - stop_loss) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif eb.low <= target_price:
                        pnl = (entry_price - target_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break

                if j - fill_idx >= MAX_HOLD_BARS:
                    pnl = (
                        (eb.close - entry_price) if direction == "bullish"
                        else (entry_price - eb.close)
                    ) * POSITION_SIZE * BTC_CONTRACT_VALUE
                    exit_price = eb.close
                    exit_reason = "time_exit"
                    exit_idx = j
                    break

            if exit_reason is None:
                continue

            pnl -= COMMISSION
            fired_dates.add(setup_date)

            trades.append({
                "fill_bar_idx": fill_idx,
                "setup_bar_idx": setup_idx,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "gap_size": gap_size,
                "direction": direction,
                "rr_target": round(target_distance / stop_distance, 2),
                "pnl": pnl,
                "exit_reason": exit_reason,
                "killzone_window": window,
                "mss_event": setup.get("mss_event"),
                "fvg_event": setup.get("fvg_event"),
            })

        except Exception as exc:
            logger.debug(f"Trade error: {exc}")
            continue

    logger.info(f"Trades collected: {len(trades)}")
    return trades


# ── ATR(14) ───────────────────────────────────────────────────────────────────

def compute_atr14(bars: list, idx: int, period: int = 14) -> float:
    if idx < period:
        return bars[idx].close * 0.005
    trs = [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        for i in range(idx - period + 1, idx + 1)
    ]
    return sum(trs) / period


# ── Volatility regime classification ─────────────────────────────────────────

def build_vol_regime_map(bars: list) -> dict[str, str]:
    """Return {cdt_date_str: 'low'|'medium'|'extreme'} for every CDT calendar date.

    Uses the first bar of each CDT date as the ATR(14)/close anchor.
    Regime is classified by 252-day rolling percentile rank of ATR%.
    Dates with fewer than VOL_REGIME_MIN_HISTORY prior days default to 'medium'.
    """
    # Step 1: find first bar index per CDT date
    date_first_bar: dict[str, int] = {}
    for i, bar in enumerate(bars):
        d = _cdt(bar.timestamp).date().isoformat()
        if d not in date_first_bar:
            date_first_bar[d] = i

    sorted_dates = sorted(date_first_bar.keys())

    # Step 2: compute ATR% at each day's first bar
    date_atr_pct: dict[str, float] = {}
    for d in sorted_dates:
        idx = date_first_bar[d]
        atr = compute_atr14(bars, idx)
        close = bars[idx].close
        date_atr_pct[d] = atr / close if close > 0 else 0.0

    # Step 3: rolling 252-day percentile rank → regime label
    regime_map: dict[str, str] = {}
    for i, d in enumerate(sorted_dates):
        lookback_start = max(0, i - VOL_REGIME_LOOKBACK)
        prior_vals = [date_atr_pct[sorted_dates[j]] for j in range(lookback_start, i)]

        if len(prior_vals) < VOL_REGIME_MIN_HISTORY:
            regime_map[d] = "medium"
            continue

        current = date_atr_pct[d]
        rank = sum(1 for v in prior_vals if v < current) / len(prior_vals)

        if rank > VOL_REGIME_HIGH:
            regime_map[d] = "extreme"
        elif rank < VOL_REGIME_LOW:
            regime_map[d] = "low"
        else:
            regime_map[d] = "medium"

    return regime_map


# ── Feature extraction ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "fvg_gap_size",
    "stop_distance",
    "rr_at_entry",
    "direction",
    "atr14_rel_gap",
    "vol_expansion",
    "momentum_20",
    "h1_momentum",
    "trend_slope_50",
    "session_time_frac",
    "day_of_week",
    "swing_dist_atr",
    "volume_ratio",
    "mss_break_pct",
    "mss_to_fvg_bars",
    "price_vs_prior_day_pct",
]


def build_daily_ranges(bars: list, volumes: np.ndarray) -> dict:
    """Compute {cdt_date_str: (day_high, day_low)} for prior-day features."""
    from collections import defaultdict
    daily: dict = defaultdict(lambda: [float("-inf"), float("inf")])
    for i, bar in enumerate(bars):
        d = _cdt(bar.timestamp).date().isoformat()
        if bar.high > daily[d][0]:
            daily[d][0] = bar.high
        if bar.low < daily[d][1]:
            daily[d][1] = bar.low
    return dict(daily)


def extract_features(
    trade: dict,
    bars: list,
    swing_highs: list,
    swing_lows: list,
    volumes: np.ndarray,
    daily_ranges: dict,
    sorted_dates: list,
) -> dict | None:
    fill_idx = trade["fill_bar_idx"]
    if fill_idx < 100:
        return None

    gap_size = trade["gap_size"]
    stop_distance = abs(trade["entry_price"] - trade["stop_loss"])
    atr14 = compute_atr14(bars, fill_idx)
    atr14_100 = compute_atr14(bars, max(fill_idx - 100, 14), 14)

    momentum_20 = (bars[fill_idx].close - bars[fill_idx - 20].close) / bars[fill_idx - 20].close
    h1_momentum = (bars[fill_idx].close - bars[fill_idx - 60].close) / bars[fill_idx - 60].close

    closes_50 = [bars[j].close for j in range(fill_idx - 49, fill_idx + 1)]
    slope_raw = np.polyfit(range(50), closes_50, 1)[0]
    trend_slope_50 = slope_raw / bars[fill_idx].close

    cdt = _cdt(bars[fill_idx].timestamp)
    session_time_frac = (cdt.hour - 9) + cdt.minute / 60.0
    day_of_week = cdt.weekday()

    cur_price = bars[fill_idx].close
    near_high = min(
        (abs(cur_price - s["price"]) for s in swing_highs if s["index"] < fill_idx),
        default=atr14 * 20,
    )
    near_low = min(
        (abs(cur_price - s["price"]) for s in swing_lows if s["index"] < fill_idx),
        default=atr14 * 20,
    )
    swing_dist_atr = min(near_high, near_low) / (atr14 + 1e-9)

    # Real volume ratio: 20-bar avg / 100-bar avg (is current volume elevated?)
    vol_20 = float(np.mean(volumes[max(0, fill_idx - 20): fill_idx])) if fill_idx >= 20 else 1e-9
    vol_100 = float(np.mean(volumes[max(0, fill_idx - 100): fill_idx])) if fill_idx >= 100 else vol_20
    volume_ratio = vol_20 / (vol_100 + 1e-9)

    # Volatility expansion: current ATR vs ATR from 100 bars ago
    vol_expansion = atr14 / (atr14_100 + 1e-9)

    # MSS setup quality features
    mss = trade.get("mss_event")
    mss_break_pct = 0.0
    mss_to_fvg_bars = 5  # default mid-range
    if mss:
        swing_px = mss["swing_point"]["price"]
        mss_break_pct = abs(mss["breakout_price"] - swing_px) / (swing_px + 1e-9)
        fvg_idx = trade.get("setup_bar_idx", fill_idx)
        mss_to_fvg_bars = min(fvg_idx - mss["index"], 10)

    # Prior day high/low position (where is entry in prior day's range?)
    cdt_date_str = cdt.date().isoformat()
    price_vs_prior_day_pct = 0.5  # default mid-range
    try:
        date_pos = sorted_dates.index(cdt_date_str)
        if date_pos > 0:
            prior_date = sorted_dates[date_pos - 1]
            pdh, pdl = daily_ranges[prior_date]
            day_range = pdh - pdl
            if day_range > 0:
                price_vs_prior_day_pct = (cur_price - pdl) / day_range
    except (ValueError, KeyError):
        pass

    return {
        "fvg_gap_size":           gap_size,
        "stop_distance":          stop_distance,
        "rr_at_entry":            trade["rr_target"],
        "direction":              1 if trade["direction"] == "bullish" else 0,
        "atr14_rel_gap":          atr14 / (gap_size + 1e-9),
        "vol_expansion":          vol_expansion,
        "momentum_20":            momentum_20,
        "h1_momentum":            h1_momentum,
        "trend_slope_50":         trend_slope_50,
        "session_time_frac":      session_time_frac,
        "day_of_week":            day_of_week,
        "swing_dist_atr":         swing_dist_atr,
        "volume_ratio":           volume_ratio,
        "mss_break_pct":          mss_break_pct,
        "mss_to_fvg_bars":        float(mss_to_fvg_bars),
        "price_vs_prior_day_pct": price_vs_prior_day_pct,
    }


# ── Profit factor ─────────────────────────────────────────────────────────────

def profit_factor(pnls: list[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


# ── Threshold tuning on CV OOS predictions ────────────────────────────────────

def tune_threshold(oos_probs: list, oos_labels: list, oos_pnls: list) -> tuple[float, float]:
    best_threshold, best_pf = 0.5, 0.0
    n_folds = 5
    any_qualified = False

    for threshold in THRESHOLD_RANGE:
        filtered_pnls = [
            pnl for prob, pnl in zip(oos_probs, oos_pnls) if prob >= threshold
        ]
        if len(filtered_pnls) < MIN_FILTERED_PER_FOLD * n_folds:
            continue
        any_qualified = True
        pf = profit_factor(filtered_pnls)
        if pf > best_pf:
            best_pf = pf
            best_threshold = threshold

    if not any_qualified:
        logger.warning(
            f"No threshold in {THRESHOLD_RANGE[0]:.2f}–{THRESHOLD_RANGE[-1]:.2f} met "
            f"minimum {MIN_FILTERED_PER_FOLD * n_folds} filtered trades — "
            "using fallback threshold=0.5 (unvalidated)"
        )

    return float(best_threshold), float(best_pf)


# ── Metrics display ───────────────────────────────────────────────────────────

def _metrics(pnls: list[float]) -> str:
    if not pnls:
        return "trades=0"
    wins = [p for p in pnls if p > 0]
    pf = profit_factor(pnls)
    wr = len(wins) / len(pnls)
    total = sum(pnls)
    return (
        f"trades={len(pnls):3d}  WR={wr:.1%}  PF={pf:.3f}  "
        f"P&L=${total:+,.0f}  avg=${total/len(pnls):+.2f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("BTC Silver Bullet ML Training Pipeline")
    logger.info(f"Config: NY AM 09:00-10:00 CDT | stop_mult={STOP_MULT} | target_cap=uncapped")
    logger.info(f"Split: train < {SPLIT_TS.date()} | holdout ≥ {SPLIT_TS.date()}")

    # ── Load bars + volume array ──────────────────────────────────────────────
    logger.info("Loading bars...")
    bars = load_csv_as_bars(DATA_PATH)
    logger.info(f"Bars loaded: {len(bars):,}")

    # Load raw volume (DollarBar.volume is set to 0 by load_csv_as_bars)
    raw_df = pd.read_csv(DATA_PATH, usecols=["volume"])
    volumes = raw_df["volume"].to_numpy(dtype=np.float64)
    volumes = volumes[: len(bars)]  # guard against length mismatch

    logger.info("Building daily price ranges (for prior-day features)...")
    daily_ranges = build_daily_ranges(bars, volumes)
    sorted_dates = sorted(daily_ranges.keys())

    logger.info("Building volatility regime map...")
    vol_regime_map = build_vol_regime_map(bars)
    n_medium = sum(1 for v in vol_regime_map.values() if v == "medium")
    n_low = sum(1 for v in vol_regime_map.values() if v == "low")
    n_extreme = sum(1 for v in vol_regime_map.values() if v == "extreme")
    n_total = len(vol_regime_map)
    logger.info(
        f"Regime days — medium: {n_medium} ({n_medium/n_total:.0%})  "
        f"low: {n_low} ({n_low/n_total:.0%})  "
        f"extreme: {n_extreme} ({n_extreme/n_total:.0%})  "
        f"total: {n_total}"
    )

    logger.info("Detecting patterns (runs once on all bars)...")
    swing_highs, swing_lows = detect_swing_points(bars)
    mss_events = detect_mss_events(bars, swing_highs, swing_lows)
    fvg_setups = detect_fvg_setups(bars)
    confluences = detect_confluence(mss_events, fvg_setups)
    logger.info(f"Confluences: {len(confluences)}")

    # ── Apply NY AM kill zone filter ──────────────────────────────────────────
    all_setups = filter_setups_by_zones(confluences, NY_AM_ZONE)
    logger.info(f"NY AM setups: {len(all_setups)}")

    # ── Split setups by timestamp ─────────────────────────────────────────────
    train_setups = [s for s in all_setups if s["timestamp"] < SPLIT_TS]
    holdout_setups = [s for s in all_setups if s["timestamp"] >= SPLIT_TS]
    logger.info(f"Train setups: {len(train_setups)} | Holdout setups: {len(holdout_setups)}")

    # ── Execute with vol regime gate ──────────────────────────────────────────
    logger.info("Executing backtest on train and holdout setups (regime gate active)...")
    train_trades = execute_for_ml(bars, train_setups, vol_regime_map=vol_regime_map)
    holdout_trades = execute_for_ml(bars, holdout_setups, vol_regime_map=vol_regime_map)

    logger.info(f"Train trades after regime filter: {len(train_trades)} | Holdout trades: {len(holdout_trades)}")

    if len(train_trades) < MIN_TRAIN_TRADES:
        logger.error(
            f"HALT: only {len(train_trades)} train trades after regime filter — minimum {MIN_TRAIN_TRADES} required. "
            "Consider widening VOL_REGIME_LOW/HIGH thresholds (e.g. 0.10/0.90) or check kill zone config."
        )
        sys.exit(1)

    # ── Extract features ──────────────────────────────────────────────────────
    logger.info("Extracting features for train trades...")
    train_records = []
    for trade in train_trades:
        feats = extract_features(trade, bars, swing_highs, swing_lows, volumes, daily_ranges, sorted_dates)
        if feats is None:
            logger.debug(f"Skipped trade at fill_bar_idx={trade['fill_bar_idx']} (insufficient history)")
            continue
        feats["label"] = 1 if trade["pnl"] > 0 else 0
        feats["pnl"] = trade["pnl"]
        train_records.append(feats)

    logger.info(f"Train records with features: {len(train_records)}")

    if len(train_records) < MIN_TRAIN_TRADES:
        logger.error(f"HALT: only {len(train_records)} usable train records after feature extraction.")
        sys.exit(1)

    train_df = pd.DataFrame(train_records)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    pnls_train = train_df["pnl"].tolist()

    # ── TimeSeriesSplit CV: AUC + OOS predictions for threshold tuning ────────
    logger.info("Running TimeSeriesSplit(5) cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    oos_probs, oos_labels, oos_pnls = [], [], []
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        pnl_val = [pnls_train[i] for i in val_idx]

        n_pos_tr = int(y_tr.sum())
        n_neg_tr = len(y_tr) - n_pos_tr
        spw = n_neg_tr / max(n_pos_tr, 1)
        model_fold = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        )
        model_fold.fit(X_tr, y_tr)
        probs = model_fold.predict_proba(X_val)[:, 1]

        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
            auc = 0.5
            logger.warning(f"  Fold {fold+1}: single class in validation set — AUC set to 0.5")
        fold_aucs.append(auc)
        oos_probs.extend(probs.tolist())
        oos_labels.extend(y_val.tolist())
        oos_pnls.extend(pnl_val)

        logger.info(f"  Fold {fold+1}: AUC={auc:.4f}  val_trades={len(y_val)}")

    cv_mean_auc = float(np.mean(fold_aucs))
    logger.info(f"CV mean AUC: {cv_mean_auc:.4f}")

    if cv_mean_auc < MIN_AUC:
        logger.warning(
            f"CV AUC={cv_mean_auc:.4f} < {MIN_AUC} — model has weak discriminative power. "
            "Continuing to report results. Re-evaluate features or feature engineering."
        )

    # ── Threshold tuning ──────────────────────────────────────────────────────
    logger.info("Tuning probability threshold on CV OOS predictions...")
    best_threshold, best_cv_pf = tune_threshold(oos_probs, oos_labels, oos_pnls)
    logger.info(f"Best threshold: {best_threshold:.2f}  CV OOS PF: {best_cv_pf:.3f}")

    # Compare CV OOS unfiltered vs filtered
    cv_oos_pf_unfiltered = profit_factor(oos_pnls)
    cv_oos_filtered = [p for prob, p in zip(oos_probs, oos_pnls) if prob >= best_threshold]
    cv_oos_pf_filtered = profit_factor(cv_oos_filtered) if cv_oos_filtered else 0.0
    pass_rate_cv = len(cv_oos_filtered) / len(oos_pnls) if oos_pnls else 0.0

    # ── Train final model on full train set ───────────────────────────────────
    logger.info("Training final model on full train set...")
    n_pos_full = int(y_train.sum())
    n_neg_full = len(y_train) - n_pos_full
    spw_full = n_neg_full / max(n_pos_full, 1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=spw_full,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    train_probs_full = model.predict_proba(X_train)[:, 1]
    train_auc_full = roc_auc_score(y_train, train_probs_full)

    # ── Holdout evaluation ────────────────────────────────────────────────────
    logger.info("Extracting features for holdout trades...")
    holdout_records = []
    for trade in holdout_trades:
        feats = extract_features(trade, bars, swing_highs, swing_lows, volumes, daily_ranges, sorted_dates)
        if feats is None:
            continue
        feats["pnl"] = trade["pnl"]
        holdout_records.append(feats)

    logger.info(f"Holdout records with features: {len(holdout_records)}")

    holdout_pf_unfiltered = profit_factor([r["pnl"] for r in holdout_records]) if holdout_records else 0.0
    holdout_pf_filtered = 0.0
    holdout_pass_rate = 0.0
    holdout_filtered_pnls: list[float] = []

    if holdout_records:
        X_holdout = pd.DataFrame(holdout_records)[FEATURE_COLS].values
        holdout_probs = model.predict_proba(X_holdout)[:, 1]
        holdout_filtered_pnls = [
            r["pnl"] for r, p in zip(holdout_records, holdout_probs) if p >= best_threshold
        ]
        holdout_pass_rate = len(holdout_filtered_pnls) / len(holdout_records)

        if holdout_filtered_pnls:
            holdout_pf_filtered = profit_factor(holdout_filtered_pnls)
        else:
            logger.warning(
                f"No holdout trades passed filter (threshold={best_threshold:.2f})"
            )

    # ── Save artifacts ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "model.joblib")
    logger.info(f"Model saved: {MODEL_DIR}/model.joblib")

    threshold_data = {
        "threshold": best_threshold,
        "cv_mean_auc": cv_mean_auc,
        "cv_oos_pf_unfiltered": round(cv_oos_pf_unfiltered, 4),
        "cv_oos_pf_filtered": round(cv_oos_pf_filtered, 4),
        "cv_pass_rate": round(pass_rate_cv, 4),
        "train_auc_full": round(train_auc_full, 4),
    }
    with open(MODEL_DIR / "threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Threshold saved: {MODEL_DIR}/threshold.json")

    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    logger.info(f"Feature importance saved: {MODEL_DIR}/feature_importance.csv")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BTC SILVER BULLET ML TRAINING REPORT")
    print("=" * 70)
    print(f"\nConfig:  NY AM 09:00-10:00 CDT | stop_mult={STOP_MULT} | uncapped")
    print(f"Model:   XGBoost | n_estimators=200 | max_depth=3 | lr=0.05 | scale_pos_weight={spw_full:.1f}")
    print(f"CV:      TimeSeriesSplit(5)")
    print(f"CV AUC:  {cv_mean_auc:.4f}  (train AUC full fit: {train_auc_full:.4f})")
    print(f"Threshold: {best_threshold:.2f}  (CV OOS PF: {best_cv_pf:.3f})")
    print()

    print("── TRAIN (in-sample, TimeSeriesSplit CV OOS) ──────────────────────")
    print(f"  Unfiltered: {_metrics(oos_pnls)}")
    print(f"  Filtered:   {_metrics(cv_oos_filtered)}  (pass rate: {pass_rate_cv:.1%})")
    print()

    print("── HOLDOUT (out-of-sample) ─────────────────────────────────────────")
    holdout_all_pnls = [r["pnl"] for r in holdout_records]
    print(f"  Unfiltered: {_metrics(holdout_all_pnls)}")
    if holdout_filtered_pnls:
        print(f"  Filtered:   {_metrics(holdout_filtered_pnls)}  (pass rate: {holdout_pass_rate:.1%})")
    else:
        print(f"  Filtered:   No trades passed filter (threshold={best_threshold:.2f})")
    print()

    print("── FEATURE IMPORTANCE ──────────────────────────────────────────────")
    for _, row in feat_imp.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:<22} {row['importance']:.4f}  {bar}")
    print()

    print("── ARTIFACTS ───────────────────────────────────────────────────────")
    print(f"  {MODEL_DIR}/model.joblib")
    print(f"  {MODEL_DIR}/threshold.json")
    print(f"  {MODEL_DIR}/feature_importance.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
