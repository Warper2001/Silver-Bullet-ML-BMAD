#!/usr/bin/env python3
"""Amendment 4 harness: re-run the sealed §5 gate of
_bmad-output/preregistration_yank_bidirectional_m15_choch.md UNCHANGED on front-month bars.

The only change from Amendment 1's run is the input bars. Engine code, StrategyConfig
(effective), gate definitions and bars-window are all as sealed and are pinned below.

  1. G0 (reproduction): run the sealed gate on the ORIGINAL bars and require Amendment 1's and
     Amendment 2's recorded numbers to reproduce. If it does not, STOP: nothing corrected is run.
  2. Corrected run: the same gate on (2025 front-month rebuild) + (Jan-Feb 2026 raw MNQH26).

One-shot: refuses to run if the output directory already holds a results.json, so the run
cannot be repeated with tweaks. No sealed-holdout file is opened, and no bar at or after
2026-03-01 is retained (asserted). The raw JSON is scanned past that point, never kept.

Usage (from the main checkout):
    .venv/bin/python tools/yank_bidir_frontmonth_rerun.py --preregistration <sealing-sha> \
        --front2025 <mnq_1min_2025_frontmonth.csv> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEALED_DOC = "_bmad-output/preregistration_yank_bidirectional_m15_choch.md"
HARNESS_REL = "tools/yank_bidir_frontmonth_rerun.py"
AMENDMENT_MARK = "## Amendment 4"
YAML_REF = "41d693d"  # the commit Amendment 1's gate ran at
RAW_JSON = Path("/root/mnq_historical.json")

# Everything the corrected run depends on, pinned before the run.
PINNED_FILES = {
    "src/research/backtest_engine.py": "4c72b05b8adf1b0840bc1d604e0d5bfdd561f51cd7b5f3838f99b509e614d911",
    "src/research/strategy_core.py": "96e087d8154a99b31da9a7f0239da9dbf49129126cb4553da5d8543870931201",
    "src/research/config_loader.py": "b290e3f10337573f0982dde22f833d6256e903a291c488a1dc24cfbc1f6f1a92",
    "tools/yank_gap_ceiling_backtest.py": "a28398ee80fa29858b78ee04d4bf73c58eac3351767808824ab1e1e0ff99202b",
    "tools/yank_bidir_m15_choch_g5_gate.py": "6c71dd3a0eccbeb94a2beeb5cde6771f3d91fadcb71fb975f25bc24b71f2e72f",
    "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv": "3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822",  # noqa: E501
}
PIN_YAML_SHA = "293c9d23c69f667564954e30bbb6360ca8af658288978ffa62de1b3c21391abb"
PIN_FRONT2025_SHA = "f1fe5b36abba90681d8b1439a3975f94e4b4d1040093c7c0368629a807d219d4"
PIN_RAW_JSON_SHA = "e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924"
# The combined corrected series the engine reads (337,745 rows, 2025-01-01 23:01Z .. 2026-02-27 22:00Z).
PIN_CORRECTED_BARS_SHA = (
    "a850c51275ab1e4ccedcba87ac3c34eb1260a36dfc2f38495797c65b905b7d48"
)

DERIV_START = pd.Timestamp("2025-01-01", tz="UTC")
DERIV_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-03-01", tz="UTC")
JAN1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FRONT = "MNQH26"  # the front month for every Jan-Feb 2026 session
POINT_NOTIONAL = 20.0  # notional = close * volume * 20, as both existing CSVs
MAX_GAP_ATR_RATIO = 0.426
HALF_SPLIT = pd.Timestamp("2025-08-01", tz="UTC")  # Amendment 2's disclosure split

# Amendment 1 / 2 recorded results on the ORIGINAL bars: the G0 reproduction targets.
G0_EXPECTED: dict[str, Any] = {
    "baseline": {"n": 46, "pf": 1.053},
    "bullish": {"n": 23, "pf": 1.430, "net": 2566.75},
    "g3_diff_pct": 3.32,
    "worst_month": ("2026-01", 17.4),
    "h1": {"n": 10, "net": -261.25},
    "h2": {"n": 13, "net": 2828.00},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


# ── gates that read no bars ──────────────────────────────────────────────────────────────


def check_sealing_commit(root: Path, sha: str, harness_bytes: bytes) -> None:
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise SystemExit(f"refusing: {sha!r} is not a commit SHA")
    doc = git(root, "show", f"{sha}:{SEALED_DOC}")
    if doc.returncode != 0:
        raise SystemExit(f"refusing: commit {sha} does not contain {SEALED_DOC}")
    if AMENDMENT_MARK.encode() not in doc.stdout:
        raise SystemExit(f"refusing: {sha}:{SEALED_DOC} holds no '{AMENDMENT_MARK}'")
    blob = git(root, "show", f"{sha}:{HARNESS_REL}")
    if blob.returncode != 0 or blob.stdout != harness_bytes:
        raise SystemExit(
            f"refusing: {HARNESS_REL} differs from the copy sealed in {sha}"
        )


def check_pins(root: Path, front2025: Path) -> dict[str, str]:
    got: dict[str, str] = {}
    for rel, want in PINNED_FILES.items():
        p = root / rel
        if not p.exists():
            raise SystemExit(f"refusing: {rel} missing")
        got[rel] = sha256(p)
        if got[rel] != want:
            raise SystemExit(f"refusing: {rel} sha256 {got[rel]} != pinned {want}")
    for name, p, want in (
        ("front2025", front2025, PIN_FRONT2025_SHA),
        ("raw_json", RAW_JSON, PIN_RAW_JSON_SHA),
    ):
        got[name] = sha256(p)
        if got[name] != want:
            raise SystemExit(f"refusing: {name} sha256 {got[name]} != pinned {want}")
    return got


def sealed_yaml(root: Path, out_dir: Path) -> Path:
    r = git(root, "show", f"{YAML_REF}:strategy_config.yaml")
    if r.returncode != 0:
        raise SystemExit(f"refusing: cannot read {YAML_REF}:strategy_config.yaml")
    p = out_dir / f"strategy_config_{YAML_REF}.yaml"
    p.write_bytes(r.stdout)
    if sha256(p) != PIN_YAML_SHA:
        raise SystemExit(
            f"refusing: {YAML_REF} YAML sha256 {sha256(p)} != pinned {PIN_YAML_SHA}"
        )
    return p


# ── corrected bars ───────────────────────────────────────────────────────────────────────


def build_corrected(front2025: Path, raw_json: Path, out_csv: Path) -> dict[str, Any]:
    """2025 front-month rebuild + Jan-Feb 2026 raw MNQH26 (1-minute), hard-cut before the holdout."""
    from tools.yank_frontmonth_revalidation import read_raw

    b25 = pd.read_csv(front2025, parse_dates=["timestamp"])
    b25["timestamp"] = b25["timestamp"].dt.tz_convert("UTC")
    b25 = b25[
        (b25["timestamp"] >= DERIV_START)
        & (b25["timestamp"] < pd.Timestamp("2026-01-01", tz="UTC"))
    ]

    raw = read_raw(raw_json, JAN1, HOLDOUT_START.to_pydatetime())
    labels = {r[6] for r in raw}
    if labels != {FRONT}:
        raise SystemExit(
            f"refusing: Jan-Feb 2026 raw records carry labels {sorted(labels)}, expected only {FRONT}"
        )
    s1 = pd.DataFrame(
        [
            (
                pd.Timestamp(r[0]),
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[4] * r[5] * POINT_NOTIONAL,
            )
            for r in raw
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume", "notional"],
    )
    bars = (
        pd.concat([b25, s1], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    bars = bars[(bars["timestamp"] >= DERIV_START) & (bars["timestamp"] <= DERIV_END)]
    assert_before_holdout(bars["timestamp"])
    bars.to_csv(out_csv, index=False)
    return {
        "rows_2025": int(len(b25)),
        "rows_2026_janfeb": int(len(s1)),
        "rows_total": int(len(bars)),
        "first": str(bars["timestamp"].min()),
        "last": str(bars["timestamp"].max()),
        "s1_labels": sorted(labels),
        "out_sha256": sha256(out_csv),
    }


def assert_before_holdout(ts: pd.Series) -> None:
    if len(ts) == 0 or ts.max() >= HOLDOUT_START:
        raise SystemExit("refusing: bars are empty or extend into the sealed holdout")


# ── the sealed gate, verbatim logic ──────────────────────────────────────────────────────


def summarize(trades: Sequence[Any]) -> dict[str, float]:
    from src.research.strategy_core import calc_profit_factor

    if not trades:
        return {"n": 0, "pf": float("nan"), "net": 0.0}
    pnls = [t.pnl_usd for t in trades]
    return {"n": len(trades), "pf": calc_profit_factor(pnls), "net": float(sum(pnls))}


def evaluate_gates(baseline: Sequence[Any], bidir: Sequence[Any]) -> dict[str, Any]:
    """G1-G4 exactly as tools/yank_bidir_m15_choch_g5_gate.py (sealed §5)."""
    base_s = summarize(baseline)
    bear = [t for t in bidir if t.direction == "BEARISH"]
    bull = [t for t in bidir if t.direction == "BULLISH"]
    bear_s, bull_s = summarize(bear), summarize(bull)

    g1 = bull_s["n"] >= 15
    g2 = bool(bull_s["n"] and bull_s["pf"] == bull_s["pf"] and bull_s["pf"] > 1.3)
    g3, g3_diff = False, float("nan")
    if (
        base_s["pf"] == base_s["pf"]
        and base_s["pf"] > 0
        and bear_s["pf"] == bear_s["pf"]
    ):
        g3_diff = abs(bear_s["pf"] - base_s["pf"]) / base_s["pf"] * 100.0
        g3 = g3_diff <= 10.0
    months = Counter(t.timestamp_entry.strftime("%Y-%m") for t in bull)
    worst, g4_pct, g4 = None, float("nan"), False
    if bull_s["n"] > 0:
        worst, cnt = months.most_common(1)[0]
        g4_pct = cnt / bull_s["n"] * 100.0
        g4 = g4_pct <= 40.0
    return {
        "baseline": base_s,
        "bearish": bear_s,
        "bullish": bull_s,
        "n_bidir": len(bidir),
        "months": dict(sorted(months.items())),
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G3_diff_pct": g3_diff,
        "G4": g4,
        "G4_worst_month": worst,
        "G4_worst_pct": g4_pct,
        "all_pass": g1 and g2 and g3 and g4,
    }


def verdict(g: dict[str, Any]) -> str:
    """Sealed §5 reading: G1/G2 fail = H0; G3 fail = defect, not judged; G4 fail = artifact."""
    if not (g["G1"] and g["G2"]):
        return "H0 (Response B): G1/G2 failed"
    if not g["G3"]:
        return "NOT JUDGED: G3 failed (bearish behaviour changed; sealed §5 calls this a defect, not a verdict)"
    if not g["G4"]:
        return "H1 NOT CONFIRMED: G4 failed (bullish trades concentrated in one month)"
    return "ALL FOUR GATES PASS on this input"


def half_split(bull: Sequence[Any]) -> dict[str, dict[str, float]]:
    def cut(sel: Sequence[Any]) -> dict[str, float]:
        return summarize(list(sel))

    def ts(t: Any) -> pd.Timestamp:
        e = pd.Timestamp(t.timestamp_entry)
        return e.tz_convert("UTC") if e.tzinfo else pd.Timestamp(e, tz="UTC")

    return {
        "H1_2025-01..07": cut([t for t in bull if ts(t) < HALF_SPLIT]),
        "H2_2025-08..2026-02": cut([t for t in bull if ts(t) >= HALF_SPLIT]),
    }


def dispersion(bull: Sequence[Any]) -> dict[str, float]:
    """Descriptive only. No edge claim may be made from N this small (sealed in Amendment 4)."""
    x = [t.pnl_usd for t in bull]
    n = len(x)
    if n < 3:
        return {"n": n}
    mean = sum(x) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": mean / se if se else float("nan"),
        "mde_80pct_power_one_sided_05": (1.645 + 0.842) * se,
    }


def g0_check(
    orig: dict[str, Any], halves: dict[str, dict[str, float]]
) -> dict[str, Any]:
    e = G0_EXPECTED
    checks = {
        "baseline_n": orig["baseline"]["n"] == e["baseline"]["n"],
        "baseline_pf": round(orig["baseline"]["pf"], 3) == e["baseline"]["pf"],
        "bullish_n": orig["bullish"]["n"] == e["bullish"]["n"],
        "bullish_pf": round(orig["bullish"]["pf"], 3) == e["bullish"]["pf"],
        "bullish_net": round(orig["bullish"]["net"], 2) == e["bullish"]["net"],
        "g3_diff_pct": round(orig["G3_diff_pct"], 2) == e["g3_diff_pct"],
        "worst_month": (orig["G4_worst_month"], round(orig["G4_worst_pct"], 1))
        == e["worst_month"],
        "h1": (halves["H1_2025-01..07"]["n"], round(halves["H1_2025-01..07"]["net"], 2))
        == (e["h1"]["n"], e["h1"]["net"]),
        "h2": (
            halves["H2_2025-08..2026-02"]["n"],
            round(halves["H2_2025-08..2026-02"]["net"], 2),
        )
        == (e["h2"]["n"], e["h2"]["net"]),
    }
    return {"checks": checks, "pass": all(checks.values())}


def trades_frame(trades: Sequence[Any]) -> pd.DataFrame:
    cols = [
        "timestamp_entry",
        "timestamp_exit",
        "direction",
        "entry_price",
        "exit_price",
        "pnl_usd",
        "exit_reason",
    ]
    return pd.DataFrame([{c: getattr(t, c, None) for c in cols} for t in trades])


# ── run ──────────────────────────────────────────────────────────────────────────────────


def run_gate(
    csv_path: Path, yaml_path: Path
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    from src.research.backtest_engine import BacktestEngine
    from src.research.config_loader import load_strategy_config

    live_cfg = load_strategy_config(yaml_path)
    base_cfg = dataclasses.replace(live_cfg, max_gap_atr_ratio=MAX_GAP_ATR_RATIO)
    bidir_cfg = dataclasses.replace(
        base_cfg, bearish_only=False
    )  # m15_confirmation stays True
    baseline = BacktestEngine(str(csv_path), base_cfg).run()
    bidir = BacktestEngine(str(csv_path), bidir_cfg).run()
    return evaluate_gates(baseline, bidir), baseline, bidir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preregistration", required=True, help="sealing commit SHA of Amendment 4"
    )
    ap.add_argument("--front2025", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir.resolve()
    if (out / "results.json").exists():
        raise SystemExit(
            "refusing: results.json exists. This gate is one-shot; it is not re-run with tweaks."
        )
    out.mkdir(parents=True, exist_ok=True)

    check_sealing_commit(
        ROOT, args.preregistration, Path(__file__).resolve().read_bytes()
    )
    pins = check_pins(ROOT, args.front2025.resolve())
    yaml_path = sealed_yaml(ROOT, out)
    res: dict[str, Any] = {"sealing_commit": args.preregistration, "pins": pins}

    # Outcome-blind: build the corrected bars and verify the pinned hash before any engine run.
    corr_csv = out / "corrected_bars.csv"
    res["corrected_bars"] = build_corrected(
        args.front2025.resolve(), RAW_JSON, corr_csv
    )
    if res["corrected_bars"]["out_sha256"] != PIN_CORRECTED_BARS_SHA:
        raise SystemExit(
            f"refusing: corrected bars sha256 {res['corrected_bars']['out_sha256']}"
            f" != pinned {PIN_CORRECTED_BARS_SHA}"
        )

    # G0: the sealed gate on the ORIGINAL bars must reproduce Amendment 1 / 2.
    from tools.yank_gap_ceiling_backtest import load_and_write_temp_csv

    # The loader is annotated `-> Path` but returns (path, first_ts, last_ts, n_bars).
    loaded: Any = load_and_write_temp_csv()
    tmp_csv, dmin, dmax, n_bars = loaded
    try:
        orig_g, orig_base, orig_bidir = run_gate(tmp_csv, yaml_path)
    finally:
        Path(tmp_csv).unlink(missing_ok=True)
    orig_halves = half_split([t for t in orig_bidir if t.direction == "BULLISH"])
    g0 = g0_check(orig_g, orig_halves)
    res["original"] = {
        "window": [str(dmin), str(dmax)],
        "n_bars": n_bars,
        "gates": orig_g,
        "halves": orig_halves,
        "verdict": verdict(orig_g),
    }
    res["G0"] = g0
    trades_frame([t for t in orig_bidir]).to_csv(
        out / "original_bidir_trades.csv", index=False
    )
    if not g0["pass"]:
        (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res["G0"], indent=2))
        raise SystemExit(
            "G0 FAILED: the sealed gate did not reproduce on original bars. Nothing corrected was run."
        )

    # Corrected run (bars were built and hash-verified before any engine ran).
    corr_g, corr_base, corr_bidir = run_gate(corr_csv, yaml_path)
    corr_bull = [t for t in corr_bidir if t.direction == "BULLISH"]
    orig_bull_keys = {
        pd.Timestamp(t.timestamp_entry).date()
        for t in orig_bidir
        if t.direction == "BULLISH"
    }
    corr_bull_keys = {pd.Timestamp(t.timestamp_entry).date() for t in corr_bull}
    res["corrected"] = {
        "gates": corr_g,
        "halves": half_split(corr_bull),
        "dispersion": dispersion(corr_bull),
        "verdict": verdict(corr_g),
        "bullish_entry_dates_in_both": len(orig_bull_keys & corr_bull_keys),
        "bullish_entry_dates_original_only": sorted(
            str(d) for d in orig_bull_keys - corr_bull_keys
        ),
        "bullish_entry_dates_corrected_only": sorted(
            str(d) for d in corr_bull_keys - orig_bull_keys
        ),
    }
    trades_frame(corr_bidir).to_csv(out / "corrected_bidir_trades.csv", index=False)
    trades_frame(corr_base).to_csv(out / "corrected_baseline_trades.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(
        json.dumps(
            {
                "G0": g0["pass"],
                "original": res["original"]["verdict"],
                "corrected": res["corrected"]["verdict"],
            },
            indent=2,
        )
    )
    return 0 if corr_g["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
