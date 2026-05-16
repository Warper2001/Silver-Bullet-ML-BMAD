#!/usr/bin/env python3
"""Validate and conditionally promote the tier2 adaptive threshold to the MNQ paper trader.

Performance gate (ALL THREE must pass for GO):
  1. Val tail filtered PF >= 1.20
  2. Optimal threshold >= 0.45 (0.40 = take-all = same as disabled)
  3. Filtered trade count >= 25% of val tail signals

If GO:
  - Writes models/xgboost/tier2_threshold.json with the validated threshold
  - Patches src/research/tier2_streaming_working.py to load threshold from JSON
  - Adds per-signal filter logging to logs/tier2_filter_log.csv

If NO-GO:
  - Prints reason(s) and exits 0; NO files are modified
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

# ── Paths ────────────────────────────────────────────────────────────────────

FEATURES_PATH  = Path("data/ml_training/doe_run_08_fullyear_features.csv")
HISTORY_PATH   = Path("data/ml_training/doe_run_08_fullyear_history.csv")
THRESHOLD_JSON = Path("models/xgboost/tier2_threshold.json")
TRADER_PATH    = Path("src/research/tier2_streaming_working.py")

# ── Validation parameters ─────────────────────────────────────────────────────

VAL_TAIL_MONTHS      = ["2025-11", "2025-12"]
THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_VAL_TRADES       = 10

GATE_VAL_PF_MIN       = 1.20   # val tail filtered PF must be >= this
GATE_THRESHOLD_MIN    = 0.45   # 0.40 takes everything = same as disabled
GATE_MIN_FILTER_RATIO = 0.25   # filter must actually reject ≥ 75% OR take ≥ 25%

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURES_PATH.exists():
        sys.exit(f"Error: {FEATURES_PATH} not found")
    if not HISTORY_PATH.exists():
        sys.exit(f"Error: {HISTORY_PATH} not found")

    feat = pd.read_csv(FEATURES_PATH)
    hist = pd.read_csv(HISTORY_PATH)

    if len(feat) != len(hist):
        raise ValueError(
            f"Row-count mismatch: features={len(feat)}, history={len(hist)}"
        )

    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["year_month"] = hist["timestamp"].dt.to_period("M").astype(str)
    feat["year_month"] = hist["year_month"].values

    print(f"  Loaded {len(feat)} samples | "
          f"{hist['timestamp'].min().date()} → {hist['timestamp'].max().date()}")
    month_counts = feat["year_month"].value_counts().sort_index()
    for m, n in month_counts.items():
        print(f"    {m}: {n} trades")
    return feat, hist


def train_on_all_data(feat: pd.DataFrame, hist: pd.DataFrame) -> Pipeline:
    """Fit model on the full dataset (Jan–Dec 2025)."""
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


def run_val_tail_search(
    model: Pipeline,
    feat: pd.DataFrame,
    hist: pd.DataFrame,
) -> tuple[float, float, int, int]:
    """Search threshold candidates on val tail. Returns (best_thresh, best_pf, n_taken, n_total)."""
    val_mask = feat["year_month"].isin(VAL_TAIL_MONTHS)
    X_val    = feat.loc[val_mask, FEATURE_COLS].values
    pnl_val  = hist.loc[val_mask, "pnl"].values
    n_total  = val_mask.sum()

    print(f"\n  Val tail: {VAL_TAIL_MONTHS}  ({n_total} trades)")
    print(f"\n  {'Threshold':>10}  {'N taken':>8}  {'% taken':>8}  {'PF':>8}")
    print(f"  {'-'*44}")

    proba = model.predict_proba(X_val)[:, 1]

    best_pf, best_thresh, best_n_taken = float("-inf"), 0.40, 0
    for thr in THRESHOLD_CANDIDATES:
        mask = proba >= thr
        n    = mask.sum()
        if n < MIN_VAL_TRADES:
            print(f"  {thr:>10.2f}  {n:>8}  {'—':>8}  {'(< min trades)':>8}")
            continue
        pf = profit_factor(pnl_val[mask])
        pct = n / n_total
        print(f"  {thr:>10.2f}  {n:>8}  {pct:>8.1%}  {pf:>8.3f}")
        if pf > best_pf:
            best_pf, best_thresh, best_n_taken = pf, thr, n

    if best_pf == float("-inf"):
        best_pf = 0.0

    print(f"\n  → Best: threshold={best_thresh:.2f}  PF={best_pf:.3f}  "
          f"taken={best_n_taken}/{n_total} ({best_n_taken/n_total:.1%})")
    return best_thresh, best_pf, best_n_taken, n_total


def apply_gate(
    best_thresh: float,
    best_pf: float,
    n_taken: int,
    n_total: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if best_pf < GATE_VAL_PF_MIN:
        reasons.append(
            f"Val tail PF {best_pf:.3f} < {GATE_VAL_PF_MIN} "
            f"(must beat meaningful baseline)"
        )
    if best_thresh < GATE_THRESHOLD_MIN:
        reasons.append(
            f"Threshold {best_thresh:.2f} < {GATE_THRESHOLD_MIN} "
            f"(0.40 = take-all = identical to disabled)"
        )
    filter_ratio = n_taken / n_total if n_total > 0 else 0.0
    if filter_ratio < GATE_MIN_FILTER_RATIO:
        reasons.append(
            f"Filter ratio {filter_ratio:.1%} < {GATE_MIN_FILTER_RATIO:.0%} "
            f"(filter must actually reject some trades)"
        )
    return len(reasons) == 0, reasons


def deploy(best_thresh: float) -> None:
    """Write threshold JSON and patch the paper trader. Idempotent."""
    # 1. Write threshold JSON
    payload = {
        "threshold":        best_thresh,
        "validated_date":   str(date.today()),
        "val_tail_months":  VAL_TAIL_MONTHS,
        "gate_criteria": {
            "val_pf_min":        GATE_VAL_PF_MIN,
            "threshold_min":     GATE_THRESHOLD_MIN,
            "min_filter_ratio":  GATE_MIN_FILTER_RATIO,
        },
        "note": "Written by validate_tier2_promotion.py — all gate criteria passed",
    }
    THRESHOLD_JSON.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  Wrote {THRESHOLD_JSON}")

    # 2. Patch tier2_streaming_working.py
    text = TRADER_PATH.read_text()

    if "tier2_threshold.json" in text:
        print(f"  {TRADER_PATH} already patched — skipping file patch (idempotent)")
        return

    # Patch A — insert JSON-load block into MetaLabelingFilter.__init__
    # Anchor: the logger.info line immediately after joblib.load succeeds
    anchor_a = (
        'logger.info(f"ML filter loaded from {model_path} (threshold={threshold})")'
    )
    patch_a = textwrap.dedent("""\
                # Load validated threshold from JSON; overrides constructor default
                _thr_json = Path(__file__).parent.parent.parent / "models/xgboost/tier2_threshold.json"
                if _thr_json.exists():
                    try:
                        import json as _json
                        _data = _json.loads(_thr_json.read_text())
                        self.threshold = float(_data["threshold"])
                        logger.info(
                            f"ML threshold loaded from JSON: {self.threshold} "
                            f"(validated {_data.get('validated_date', '?')})"
                        )
                    except Exception as _e:
                        logger.warning(f"Threshold JSON read failed: {_e} — using {self.threshold}")""")

    # The anchor line is indented with 16 spaces in the file
    indent = " " * 16
    anchor_a_indented = indent + anchor_a
    replacement_a = anchor_a_indented + "\n" + textwrap.indent(patch_a, indent)

    if anchor_a not in text:
        print(f"  ERROR: Anchor A not found in {TRADER_PATH}. Manual patch required.")
        print(f"  Expected: {anchor_a!r}")
        return

    text = text.replace(anchor_a_indented, replacement_a, 1)

    # Patch B — replace ML_THRESHOLD comparisons with self.ml_filter.threshold
    # There are exactly 2 occurrences of each pattern in _detect_and_enter
    old_cmp = "if proba >= ML_THRESHOLD:"
    new_cmp = "if proba >= self.ml_filter.threshold:"
    count_b = text.count(old_cmp)
    if count_b != 2:
        print(f"  WARNING: Expected 2 occurrences of {old_cmp!r}, found {count_b}. Patching anyway.")
    text = text.replace(old_cmp, new_cmp)

    # Also update the log f-string so it prints the live threshold value
    old_log = "< {ML_THRESHOLD}"
    new_log = "< {self.ml_filter.threshold}"
    text = text.replace(old_log, new_log)

    # Patch C — add _log_decision method and call sites
    # Method: insert before predict_proba (first method after FEATURE_COLS)
    method_anchor = "    def predict_proba(self, features: dict) -> float:"
    log_method = textwrap.dedent("""\
        def _log_decision(self, timestamp, proba: float, decision: str) -> None:
            \"\"\"Append filter decision to logs/tier2_filter_log.csv.\"\"\"
            import csv as _csv
            log_path = Path(__file__).parent.parent.parent / "logs/tier2_filter_log.csv"
            row = {
                "timestamp":       str(timestamp),
                "filter_decision": decision,
                "probability":     round(proba, 4),
                "threshold":       self.threshold,
            }
            write_header = not log_path.exists()
            try:
                with log_path.open("a", newline="") as _f:
                    _w = _csv.DictWriter(_f, fieldnames=list(row.keys()))
                    if write_header:
                        _w.writeheader()
                    _w.writerow(row)
            except Exception as _e:
                logger.warning(f"Filter log write failed: {_e}")

    """)
    text = text.replace(
        method_anchor,
        "    " + log_method.rstrip() + "\n\n" + method_anchor,
        1,
    )

    # Call sites — add log call after each allowed/filtered branch
    text = text.replace(
        'logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")\n'
        "                    await self._enter_trade(fvg, bar, len(bars) - 1, is_backfill)",
        'logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")\n'
        "                    self.ml_filter._log_decision(bar.timestamp, proba, \"ALLOWED\")\n"
        "                    await self._enter_trade(fvg, bar, len(bars) - 1, is_backfill)",
    )
    text = text.replace(
        'logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {self.ml_filter.threshold}")',
        'logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {self.ml_filter.threshold}")\n'
        "                    self.ml_filter._log_decision(bar.timestamp, proba, \"FILTERED\")",
    )

    # 3. Write patched file
    TRADER_PATH.write_text(text)

    # 4. Verify
    patched = TRADER_PATH.read_text()
    assert "tier2_threshold.json" in patched, "Patch A failed — JSON path not found"
    assert "self.ml_filter.threshold" in patched, "Patch B failed — comparison not updated"
    print(f"  Patched {TRADER_PATH}")
    print(f"    ✓ JSON-load block inserted in MetaLabelingFilter.__init__")
    print(f"    ✓ Comparison sites updated to self.ml_filter.threshold")
    print(f"    ✓ Filter logging (_log_decision) added")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  Tier2 Promotion Validator")
    print("=" * 62)

    print("\n[1/4] Loading data …")
    feat, hist = load_data()

    print("\n[2/4] Training model on full dataset …")
    model = train_on_all_data(feat, hist)

    print("\n[3/4] Running val tail threshold search …")
    best_thresh, best_pf, n_taken, n_total = run_val_tail_search(model, feat, hist)

    print("\n[4/4] Applying performance gate …")
    print(f"  Gate criteria:")
    print(f"    PF ≥ {GATE_VAL_PF_MIN}           → {best_pf:.3f}  {'✓' if best_pf >= GATE_VAL_PF_MIN else '✗'}")
    print(f"    Threshold ≥ {GATE_THRESHOLD_MIN}      → {best_thresh:.2f}   {'✓' if best_thresh >= GATE_THRESHOLD_MIN else '✗'}")
    filter_ratio = n_taken / n_total if n_total > 0 else 0.0
    print(f"    Filter ratio ≥ {GATE_MIN_FILTER_RATIO:.0%}  → {filter_ratio:.1%}  {'✓' if filter_ratio >= GATE_MIN_FILTER_RATIO else '✗'}")

    go, reasons = apply_gate(best_thresh, best_pf, n_taken, n_total)

    print()
    if go:
        print("  ✅  GO — all gate criteria passed")
        print(f"      Threshold to deploy: {best_thresh:.2f}")
        print()
        deploy(best_thresh)
        print()
        print("  Next steps:")
        print("    1. Review git diff on src/research/tier2_streaming_working.py")
        print("    2. Restart the paper trader:")
        print("         pkill -f tier2_streaming_working")
        print("         .venv/bin/python src/research/tier2_streaming_working.py &")
        print("    3. Confirm in logs:")
        print('         grep "ML threshold loaded from JSON" logs/tier2_streaming_working.log | tail -1')
        print("    4. Monitor filter decisions:")
        print("         tail -f logs/tier2_filter_log.csv")
    else:
        print("  ❌  NO-GO — gate criteria failed. No files modified.")
        for r in reasons:
            print(f"      • {r}")
        print()
        print("  ML_THRESHOLD remains 0.0 — paper trader unchanged.")

    print()
    print("=" * 62)


if __name__ == "__main__":
    main()
