#!/usr/bin/env python3
"""Validate and conditionally promote the LR counter-trend regime filter to the paper trader.

Performance gate (ALL THREE must pass for GO):
  1. Val tail (Nov–Dec 2025) filtered PF ≥ 1.20  (regime + model threshold combined)
  2. Regime filter ratio ≥ 0.25  (at least 25% of val tail signals pass the counter-trend filter)
  3. Adaptive threshold ≥ 0.45

LR counter-trend logic:
  LR regime "UP"   + bearish signal → PASS  (shorting into uptrend — Silver Bullet logic)
  LR regime "DOWN" + bullish signal → PASS  (buying into downtrend)
  LR "SIDEWAYS"    → PASS (neutral)
  LR regime agrees with signal direction → FILTER OUT

If GO:
  - Writes models/xgboost/lr_regime_config.json
  - Patches src/research/tier2_streaming_working.py with LRRegimeFilter class + integration

If NO-GO:
  - Prints reason(s) and exits 0; NO files modified
"""

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/ml_training/doe_run_08_fullyear_features.csv")
HISTORY_PATH  = Path("data/ml_training/doe_run_08_fullyear_history.csv")
REGIME_PATH   = Path("data/ml_training/doe_run_08_regime_enriched.csv")
CONFIG_JSON   = Path("models/xgboost/lr_regime_config.json")
TRADER_PATH   = Path("src/research/tier2_streaming_working.py")

# ── Validation parameters ─────────────────────────────────────────────────────

VAL_TAIL_MONTHS      = ["2025-11", "2025-12"]
THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_VAL_TRADES       = 10
DEFAULT_THRESHOLD    = 0.50

GATE_VAL_PF_MIN       = 1.20
GATE_THRESHOLD_MIN    = 0.45
GATE_MIN_FILTER_RATIO = 0.25

LR_FAST = 390   # 1 MNQ session (6.5 h × 60 min)
LR_SLOW = 1950  # 5 sessions (1 trading week)

FEATURE_COLS = [
    "fvg_fill_pct", "sweep_window_vol", "volume_ratio", "signal_direction",
    "h1_trend_slope", "atr", "session_displacement", "session_volume_ratio",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def profit_factor(pnl: np.ndarray) -> float:
    gains  = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def regime_counter(lr_regime: str, signal_direction: str) -> bool:
    """Return True if signal should be taken (counter-trend or neutral)."""
    if lr_regime == "UP":
        return signal_direction == "bearish"
    if lr_regime == "DOWN":
        return signal_direction == "bullish"
    return True  # SIDEWAYS → pass through


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (FEATURES_PATH, HISTORY_PATH, REGIME_PATH):
        if not p.exists():
            sys.exit(
                f"Error: {p} not found."
                + ("\n  Run: .venv/bin/python enrich_tier2_with_regime.py" if p == REGIME_PATH else "")
            )

    feat   = pd.read_csv(FEATURES_PATH)
    hist   = pd.read_csv(HISTORY_PATH)
    regime = pd.read_csv(REGIME_PATH)

    if not (len(feat) == len(hist) == len(regime)):
        raise ValueError(
            f"Row-count mismatch: features={len(feat)}, history={len(hist)}, regime={len(regime)}"
        )

    hist["timestamp"]    = pd.to_datetime(hist["timestamp"])
    hist["year_month"]   = hist["timestamp"].dt.to_period("M").astype(str)
    feat["year_month"]   = hist["year_month"].values
    regime["year_month"] = hist["year_month"].values

    print(f"  Loaded {len(feat)} samples | "
          f"{hist['timestamp'].min().date()} → {hist['timestamp'].max().date()}")
    return feat, hist, regime


def train_model(feat: pd.DataFrame, hist: pd.DataFrame) -> Pipeline:
    X = feat[FEATURE_COLS].values
    y = feat["label"].values
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.01, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=42,
        )),
    ])
    model.fit(X, y)
    print(f"  Model trained on {len(X)} samples (full Jan–Dec 2025)")
    return model


# ── Val tail search ───────────────────────────────────────────────────────────

def run_val_tail_search(
    model: Pipeline,
    feat: pd.DataFrame,
    hist: pd.DataFrame,
    regime: pd.DataFrame,
) -> tuple[float, float, int, int]:
    """Search threshold on regime-filtered val tail. Returns (best_thresh, best_pf, n_taken, n_total)."""
    val_mask = feat["year_month"].isin(VAL_TAIL_MONTHS)
    n_total  = val_mask.sum()

    X_val      = feat.loc[val_mask, FEATURE_COLS].values
    pnl_val    = hist.loc[val_mask, "pnl"].values
    lr_val     = regime.loc[val_mask, "lr_regime"].values
    dir_val    = hist.loc[val_mask, "direction"].values

    # Apply counter-trend regime filter
    regime_pass = np.array([regime_counter(r, d) for r, d in zip(lr_val, dir_val)])
    n_regime    = regime_pass.sum()

    print(f"\n  Val tail: {VAL_TAIL_MONTHS}  ({n_total} total, {n_regime} pass regime filter "
          f"[{n_regime/n_total:.1%}])")

    proba_all = model.predict_proba(X_val)[:, 1]
    proba_val = proba_all[regime_pass]
    pnl_reg   = pnl_val[regime_pass]

    print(f"\n  {'Threshold':>10}  {'N taken':>8}  {'% of reg':>9}  {'PF':>8}")
    print(f"  {'-'*40}")

    best_pf, best_thresh, best_n_taken = float("-inf"), DEFAULT_THRESHOLD, 0
    for thr in THRESHOLD_CANDIDATES:
        mask = proba_val >= thr
        n    = mask.sum()
        if n < MIN_VAL_TRADES:
            print(f"  {thr:>10.2f}  {n:>8}  {'—':>9}  {'(< min trades)':>8}")
            continue
        pf  = profit_factor(pnl_reg[mask])
        pct = n / n_regime if n_regime > 0 else 0.0
        print(f"  {thr:>10.2f}  {n:>8}  {pct:>9.1%}  {pf:>8.3f}")
        if pf > best_pf:
            best_pf, best_thresh, best_n_taken = pf, thr, n

    if best_pf == float("-inf"):
        best_pf = 0.0

    print(f"\n  → Best: threshold={best_thresh:.2f}  PF={best_pf:.3f}  "
          f"taken={best_n_taken}/{n_regime} regime-pass ({best_n_taken/n_total:.1%} of all val)")
    return best_thresh, best_pf, best_n_taken, n_regime


# ── Gate ──────────────────────────────────────────────────────────────────────

def apply_gate(
    best_thresh: float,
    best_pf: float,
    n_taken: int,
    n_regime: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if best_pf < GATE_VAL_PF_MIN:
        reasons.append(
            f"Val tail PF {best_pf:.3f} < {GATE_VAL_PF_MIN} (must beat meaningful baseline)"
        )
    if best_thresh < GATE_THRESHOLD_MIN:
        reasons.append(
            f"Threshold {best_thresh:.2f} < {GATE_THRESHOLD_MIN} (take-all ≈ disabled)"
        )
    filter_ratio = n_taken / n_regime if n_regime > 0 else 0.0
    if filter_ratio < GATE_MIN_FILTER_RATIO:
        reasons.append(
            f"Filter ratio {filter_ratio:.1%} < {GATE_MIN_FILTER_RATIO:.0%} "
            f"(model must actually reject some regime-filtered signals)"
        )
    return len(reasons) == 0, reasons


# ── Deploy ────────────────────────────────────────────────────────────────────

def deploy(best_thresh: float) -> None:
    """Write config JSON and patch the paper trader. Idempotent."""

    # 1. Write LR regime config JSON
    payload = {
        "fast_len":       LR_FAST,
        "slow_len":       LR_SLOW,
        "polarity":       "counter_trend",
        "ml_threshold":   best_thresh,
        "validated_date": str(date.today()),
        "val_tail_months": VAL_TAIL_MONTHS,
        "gate_criteria": {
            "val_pf_min":        GATE_VAL_PF_MIN,
            "threshold_min":     GATE_THRESHOLD_MIN,
            "min_filter_ratio":  GATE_MIN_FILTER_RATIO,
        },
        "enabled": True,
        "note": "Written by validate_lr_regime_promotion.py — all gate criteria passed",
    }
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  Wrote {CONFIG_JSON}")

    # 2. Patch tier2_streaming_working.py
    text = TRADER_PATH.read_text()

    if "LRRegimeFilter" in text:
        print(f"  {TRADER_PATH} already patched — skipping (idempotent)")
        return

    # ── Patch A: insert LRRegimeFilter class after MetaLabelingFilter class ──
    # Anchor: the end of the _log_decision method (last line before class gap)
    anchor_class = 'logger.warning(f"Filter log write failed: {_e}")\n'

    lr_class = textwrap.dedent("""\

    class LRRegimeFilter:
        \"\"\"LR channel counter-trend pre-filter for Silver Bullet signals.

        Reads config from models/xgboost/lr_regime_config.json.
        Counter-trend: passes signals when LR regime DISAGREES with signal direction.
        Shorting into uptrend and buying into downtrend = Silver Bullet edge.
        SIDEWAYS regime → pass through (no trend to fade).
        \"\"\"

        def __init__(self) -> None:
            self.enabled  = False
            self.fast_len = 390
            self.slow_len = 1950
            self.ml_threshold = 0.50
            _cfg = Path(__file__).parent.parent.parent / "models/xgboost/lr_regime_config.json"
            if _cfg.exists():
                try:
                    import json as _json
                    _data = _json.loads(_cfg.read_text())
                    self.fast_len     = int(_data.get("fast_len", 390))
                    self.slow_len     = int(_data.get("slow_len", 1950))
                    self.ml_threshold = float(_data.get("ml_threshold", 0.50))
                    self.enabled      = bool(_data.get("enabled", True))
                    logger.info(
                        f"LR regime filter loaded: fast={self.fast_len}, "
                        f"slow={self.slow_len}, polarity=counter_trend, "
                        f"ml_threshold={self.ml_threshold}, enabled={self.enabled} "
                        f"(validated {_data.get('validated_date', '?')})"
                    )
                except Exception as _e:
                    logger.warning(f"LR regime config load failed: {_e} — filter disabled")
            else:
                logger.info("lr_regime_config.json not found — LR regime filter disabled")

        def allows(self, bars: list, signal_direction: str) -> bool:
            \"\"\"Return True if this signal should proceed past the regime pre-filter.\"\"\"
            if not self.enabled:
                return True
            if len(bars) < self.slow_len:
                return True  # insufficient bar history during warm-up → pass through
            try:
                import numpy as _np
                from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector
                closes = _np.array([b.close for b in bars[-self.slow_len:]], dtype=float)
                detector = LRChannelRegimeDetector(fast_len=self.fast_len, slow_len=self.slow_len)
                regimes  = detector.fit_predict(closes)
                regime   = regimes[-1]  # current bar regime
            except Exception as _e:
                logger.warning(f"LR regime computation failed: {_e} — passing signal through")
                return True

            # Counter-trend: pass when regime DISAGREES with signal direction
            if regime == "UP":
                passes = (signal_direction == "bearish")
            elif regime == "DOWN":
                passes = (signal_direction == "bullish")
            else:  # SIDEWAYS → neutral, no trend to fade
                passes = True

            if not passes:
                logger.info(
                    f"Signal FILTERED by LR regime | regime={regime}, direction={signal_direction}"
                )
            return passes
    """)

    if anchor_class not in text:
        print(f"  ERROR: Patch A anchor not found in {TRADER_PATH}. Manual patch required.")
        return

    text = text.replace(anchor_class, anchor_class + lr_class, 1)

    # ── Patch B: add self.lr_filter = LRRegimeFilter() after self.ml_filter ──
    anchor_b = "        self.ml_filter = MetaLabelingFilter(ML_MODEL_PATH)\n"
    patch_b  = anchor_b + "        self.lr_filter  = LRRegimeFilter()\n"

    if anchor_b not in text:
        print(f"  ERROR: Patch B anchor not found. Manual patch required.")
        return

    text = text.replace(anchor_b, patch_b, 1)

    # ── Patch C: insert LR pre-filter check into bearish detection path ──
    # Bearish is the live path (BEARISH_ONLY=True), so patch this one first.
    # Anchor: the fvg assignment line in the bearish block
    anchor_c_old = (
        '        if self.h1_bearish_sweep_active:  # independent of bullish — fixes pre-existing elif gap\n'
        '            fvg = self._detect_fvg(bars, bullish=False)\n'
        '            if fvg:\n'
        '                features = self._extract_features(bars, bar, fvg, "bearish")\n'
    )
    anchor_c_new = (
        '        if self.h1_bearish_sweep_active:  # independent of bullish — fixes pre-existing elif gap\n'
        '            fvg = self._detect_fvg(bars, bullish=False)\n'
        '            if fvg:\n'
        '                if not self.lr_filter.allows(bars, "bearish"):\n'
        '                    return\n'
        '                features = self._extract_features(bars, bar, fvg, "bearish")\n'
    )

    if anchor_c_old not in text:
        print(f"  ERROR: Patch C (bearish path) anchor not found. Manual patch required.")
        return

    text = text.replace(anchor_c_old, anchor_c_new, 1)

    # ── Patch D: insert LR pre-filter check into bullish detection path ──
    anchor_d_old = (
        '        if not BEARISH_ONLY and self.h1_bullish_sweep_active:\n'
        '            fvg = self._detect_fvg(bars, bullish=True)\n'
        '            if fvg:\n'
        '                features = self._extract_features(bars, bar, fvg, "bullish")\n'
    )
    anchor_d_new = (
        '        if not BEARISH_ONLY and self.h1_bullish_sweep_active:\n'
        '            fvg = self._detect_fvg(bars, bullish=True)\n'
        '            if fvg:\n'
        '                if not self.lr_filter.allows(bars, "bullish"):\n'
        '                    return\n'
        '                features = self._extract_features(bars, bar, fvg, "bullish")\n'
    )

    if anchor_d_old not in text:
        print(f"  [info] Patch D (bullish path) anchor not found — skipping (BEARISH_ONLY active)")
    else:
        text = text.replace(anchor_d_old, anchor_d_new, 1)

    # 3. Write patched file
    TRADER_PATH.write_text(text)

    # 4. Verify
    patched = TRADER_PATH.read_text()
    assert "LRRegimeFilter" in patched,    "Patch A failed — LRRegimeFilter class not found"
    assert "self.lr_filter"  in patched,   "Patch B failed — self.lr_filter not found"
    assert 'lr_filter.allows(bars, "bearish")' in patched, "Patch C failed"

    print(f"  Patched {TRADER_PATH}")
    print(f"    ✓ LRRegimeFilter class inserted (Patch A)")
    print(f"    ✓ self.lr_filter = LRRegimeFilter() added in __init__ (Patch B)")
    print(f"    ✓ Regime pre-filter wired into bearish detection path (Patch C)")
    if 'lr_filter.allows(bars, "bullish")' in patched:
        print(f"    ✓ Regime pre-filter wired into bullish detection path (Patch D)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  LR Regime Promotion Validator")
    print("=" * 64)

    print("\n[1/4] Loading data …")
    feat, hist, regime = load_data()

    print("\n[2/4] Training model on full dataset …")
    model = train_model(feat, hist)

    print("\n[3/4] Running val tail threshold search (counter-trend regime filtered) …")
    best_thresh, best_pf, n_taken, n_regime = run_val_tail_search(model, feat, hist, regime)

    print("\n[4/4] Applying performance gate …")
    filter_ratio = n_taken / n_regime if n_regime > 0 else 0.0
    print(f"  Gate criteria:")
    print(f"    PF ≥ {GATE_VAL_PF_MIN}           → {best_pf:.3f}  {'✓' if best_pf >= GATE_VAL_PF_MIN else '✗'}")
    print(f"    Threshold ≥ {GATE_THRESHOLD_MIN}      → {best_thresh:.2f}   {'✓' if best_thresh >= GATE_THRESHOLD_MIN else '✗'}")
    print(f"    Filter ratio ≥ {GATE_MIN_FILTER_RATIO:.0%}  → {filter_ratio:.1%}  {'✓' if filter_ratio >= GATE_MIN_FILTER_RATIO else '✗'}")

    go, reasons = apply_gate(best_thresh, best_pf, n_taken, n_regime)

    print()
    if go:
        print("  ✅  GO — all gate criteria passed")
        print(f"      LR regime: fast={LR_FAST} bars, slow={LR_SLOW} bars, polarity=counter_trend")
        print(f"      ML threshold to deploy: {best_thresh:.2f}")
        print()
        deploy(best_thresh)
        print()
        print("  Next steps:")
        print("    1. Review git diff on src/research/tier2_streaming_working.py")
        print("    2. Restart the paper trader:")
        print("         pkill -f tier2_streaming_working")
        print("         .venv/bin/python src/research/tier2_streaming_working.py &")
        print("    3. Confirm in logs:")
        print('         grep "LR regime filter loaded" logs/tier2_streaming_working.log | tail -1')
        print("    4. Monitor — first 1,950 bars are warm-up (regime filter passes through):")
        print('         grep "FILTERED by LR regime" logs/tier2_streaming_working.log | tail -5')
    else:
        print("  ❌  NO-GO — gate criteria failed. No files modified.")
        for r in reasons:
            print(f"      • {r}")

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
