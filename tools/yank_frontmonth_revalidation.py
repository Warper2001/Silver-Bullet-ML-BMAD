#!/usr/bin/env python3
"""Front-month re-validation of YANK seal 138cab1 (preregistration_yank_frontmonth_revalidation.md).

Two subcommands:

  run    one faithful replay (one arm, one data mode) over 2025-05-19 -> 2026-05-19.
         Reads holdout-period bars, so it refuses to start unless --preregistration names a
         commit that contains the sealed document AND this exact harness file; it then
         appends an ACCESS_LOG row before any holdout-period byte is read.
  score  G0 reproduction check plus the inherited 138cab1 decision rule. Reads only run
         outputs and the June 2026 seal CSVs; touches no bars.

Data modes (sealed doc section 2):
  original   mnq_1min_2025.csv from 2025-05-19 + mnq_1min_2026_ytd.csv to 2026-05-19, exactly
             as backtest_tier2_1year_validation.py loads them (G0 must reproduce June's rows).
  corrected  same 2025 bars (S0); 2026 replaced by S1 (raw MNQH26, Jan-Feb), S2 (raw MNQH26,
             sessions 03-02..03-11, any session carrying another label dropped whole), S3
             (mnq_1min_2026_ytd.csv from the 03-12 roll session, shifted by -delta).

Config: the engine reads STRATEGY_CONFIG_PATH, which this harness points at 138cab1's own
strategy_config.yaml (sha256 verified), so the sealed config loads with no in-memory pin.

Run from the repo root of a checkout whose models/xgboost holds the pinned files, e.g.:
  .venv/bin/python tools/yank_frontmonth_revalidation.py run --preregistration <sha> \
      --mode original --ml-threshold 0.50 --out-dir <dir>
  .venv/bin/python tools/yank_frontmonth_revalidation.py score --out-dir <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HARNESS_REL = "tools/yank_frontmonth_revalidation.py"
SEALED_DOC = "_bmad-output/preregistration_yank_frontmonth_revalidation.md"
SEAL_REF = "138cab1"
SEAL_YAML_SHA = "faef1c740ed753449796dc948cce15f41722e0b8b60dbb99df3d82e37d1d8b52"
PINNED = {
    "models/xgboost/tier2_meta_labeling_model.pkl": "f58530e1d08b8436e6785b7151cc20e5296c71acd06a44ae5af5388c666f80e2",
    "models/xgboost/tier2_threshold.json": "f15753530508e2fe73f8f22dfdf7f87bd37647e190caa98e1624e58a5ef04e7e",
    "models/xgboost/lr_regime_config.json": "6410625050fe90846fcd2543b426b7bf20d118d36a487a030bafb3eca25a5a24",
}
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
RAW_JSON = Path("/root/mnq_historical.json")
DB1 = "data/processed/dollar_bars/1_minute"
JUNE_CSV = {"ml": MAIN / "data/reports/backtest_1year_20260615_181838.csv",
            "noml": MAIN / "data/reports/backtest_1year_20260615_185354.csv"}
GUARDED = ("data/trades.db", "logs/tier2_trade_log.csv", "logs/yank_ml_canary.csv")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
JAN1 = datetime(2026, 1, 1, tzinfo=UTC)
CUTOFF = datetime(2026, 3, 1, tzinfo=UTC)
ROLL_OPEN = datetime(2026, 3, 11, 18, 0, tzinfo=ET).astimezone(UTC)   # first MNQM26 session (trade date 03-12)
FRONT, NEXT = "MNQH26", "MNQM26"
POINT_NOTIONAL = 20.0          # notional = close * volume * 20, both existing CSVs
# inherited verbatim from seal 138cab1
PF_MIN, N_MIN = 1.20, 25

_FIELD = re.compile(r'^\s*"(High|Low|Open|Close|TimeStamp|TotalVolume|Contract)":\s*"?([^",]*)"?,?\s*$')


# ── small helpers ─────────────────────────────────────────────────────────────────────────

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def session_key(ts_utc: datetime):
    """Globex trade date: 18:00 ET opens the next day's session."""
    return (ts_utc.astimezone(ET).replace(tzinfo=None) + timedelta(hours=6)).date()


def pf(x: np.ndarray) -> float | None:
    gl = -x[x < 0].sum()
    return float(x[x > 0].sum() / gl) if gl > 0 else None


# ── gate: the sealing commit must hold the sealed doc and this exact harness ─────────────

def check_sealing_commit(root: Path, sha: str, harness_bytes: bytes) -> None:
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise SystemExit(f"refusing: {sha!r} is not a commit SHA")
    doc = git(root, "cat-file", "-e", f"{sha}:{SEALED_DOC}")
    if doc.returncode != 0:
        raise SystemExit(f"refusing: commit {sha} does not contain {SEALED_DOC}")
    blob = git(root, "show", f"{sha}:{HARNESS_REL}")
    if blob.returncode != 0 or blob.stdout != harness_bytes:
        raise SystemExit(f"refusing: {HARNESS_REL} differs from the copy sealed in {sha}")


def check_pinned(root: Path) -> dict:
    got = {}
    for rel, want in PINNED.items():
        p = root / rel
        if not p.exists():
            raise SystemExit(f"refusing: {rel} missing under {root}")
        got[rel] = sha256(p)
        if got[rel] != want:
            raise SystemExit(f"refusing: {rel} sha256 {got[rel]} != pinned {want}")
    return got


def seal_yaml(root: Path, out_dir: Path) -> Path:
    r = git(root, "show", f"{SEAL_REF}:strategy_config.yaml")
    if r.returncode != 0:
        raise SystemExit(f"refusing: cannot read {SEAL_REF}:strategy_config.yaml")
    p = out_dir / f"strategy_config_{SEAL_REF}.yaml"
    p.write_bytes(r.stdout)
    if sha256(p) != SEAL_YAML_SHA:
        raise SystemExit(f"refusing: {SEAL_REF} YAML sha256 {sha256(p)} != {SEAL_YAML_SHA}")
    return p


# ── corrected 2026 bars (sealed doc section 2) ─────────────────────────────────────────

def read_raw(raw_path: Path, lo: datetime, hi: datetime) -> list[tuple]:
    """(ts_utc, open, high, low, close, volume, contract) for raw records stamped in [lo, hi)."""
    lo_s, hi_s = lo.strftime("%Y-%m-%dT%H:%M:%SZ"), hi.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, cur = [], {}
    with open(raw_path) as f:
        for line in f:
            m = _FIELD.match(line)
            if not m:
                continue
            k, v = m.groups()
            cur[k] = v
            if k == "Contract":
                ts = cur["TimeStamp"]
                if lo_s <= ts < hi_s:
                    rows.append((datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC),
                                 float(cur["Open"]), float(cur["High"]), float(cur["Low"]), float(cur["Close"]),
                                 int(float(cur["TotalVolume"])), v))
                cur = {}
    return rows


def build_corrected_2026(raw_path: Path, csv_2026: Path, end: datetime, out_csv: Path) -> dict:
    raw = read_raw(raw_path, JAN1, ROLL_OPEN)
    s1 = [r for r in raw if r[0] < CUTOFF]
    bad = {r[6] for r in s1} - {FRONT}
    if bad:
        raise SystemExit(f"refusing: S1 (Jan-Feb) carries labels other than {FRONT}: {sorted(bad)}")
    s2_all = [r for r in raw if r[0] >= CUTOFF]
    by_sess: dict = {}
    for r in s2_all:
        by_sess.setdefault(session_key(r[0]), []).append(r)
    kept = {d: rs for d, rs in by_sess.items() if all(r[6] == FRONT for r in rs)}
    dropped = sorted(str(d) for d in by_sess if d not in kept)
    s2 = [r for d in sorted(kept) for r in kept[d]]

    # S3 + the rows needed for delta, from the 2026 CSV
    s3, csv_close = [], {}
    last_s2 = max(kept) if kept else None
    with open(csv_2026) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            if last_s2 is not None and ts < ROLL_OPEN and session_key(ts) == last_s2:
                csv_close[ts] = float(row["close"])
            if ROLL_OPEN <= ts <= end:
                s3.append((ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                           int(float(row["volume"]))))
    common = [csv_close[r[0]] - r[4] for r in kept.get(last_s2, []) if r[0] in csv_close] if last_s2 else []
    delta = float(np.median(common)) if common else 0.0

    def out(ts, o, h, lo, c, v):
        return [ts.isoformat(), o, h, lo, c, v, c * v * POINT_NOTIONAL]

    rows = [out(*r[:6]) for r in s1 + s2]
    rows += [out(ts, o - delta, h - delta, lo - delta, c - delta, v) for ts, o, h, lo, c, v in s3]
    rows.sort(key=lambda x: x[0])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume", "notional"])
        w.writerows(rows)
    return {"S1_rows": len(s1), "S2_rows": len(s2), "S3_rows": len(s3),
            "S2_sessions_kept": sorted(str(d) for d in kept), "S2_sessions_dropped": dropped,
            "delta_pts": delta, "delta_common_minutes": len(common), "delta_unshifted": not common,
            "roll_open_utc": ROLL_OPEN.isoformat(), "out_csv": str(out_csv), "out_sha256": sha256(out_csv)}


# ── run ─────────────────────────────────────────────────────────────────────────────────

def guard(roots: list[Path]) -> dict:
    snap = {}
    for r in roots:
        for p in GUARDED:
            q = r / p
            snap[str(q)] = (q.stat().st_size, q.stat().st_mtime) if q.exists() else None
    return snap


async def run(args) -> dict:
    root = Path(args.engine_root).resolve()
    here = Path(__file__).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{'ml' if args.ml_threshold > 0 else 'noml'}"

    # 1. gates that read no bars
    check_sealing_commit(root, args.preregistration, here.read_bytes())
    pinned = check_pinned(root)
    yaml_path = seal_yaml(root, out)
    os.environ["STRATEGY_CONFIG_PATH"] = str(yaml_path)
    os.chdir(root)
    sys.path.insert(0, str(root))
    import backtest_tier2_1year_validation as btv  # noqa: E402
    import src.research.tier2_streaming_working as tier2  # noqa: E402
    from src.research.config_loader import load_strategy_config  # noqa: E402

    btv.ACCESS_LOG_PATH = Path(args.access_log).resolve()
    btv.verify_preregistration(args.preregistration)
    effective = tier2._build_strategy_config() if hasattr(tier2, "_build_strategy_config") else None
    sealed_cfg = load_strategy_config(yaml_path)
    if effective is not None and effective != sealed_cfg:
        raise SystemExit("refusing: effective StrategyConfig differs from the 138cab1 snapshot")

    # 2. log the access before any holdout-period byte is read
    btv.append_access_log(args.preregistration, sys.argv)

    # 3. bars
    data_root = Path(args.data_root)
    csv25, csv26 = data_root / DB1 / "mnq_1min_2025.csv", data_root / DB1 / "mnq_1min_2026_ytd.csv"
    bars = btv.load_bars(csv25, start=btv.START_DATE, end=None)
    seg = {"S0_bars": len(bars)}
    if args.mode == "original":
        extra = btv.load_bars(csv26, start=None, end=btv.END_DATE)
        seg["2026_csv_bars"] = len(extra)
    else:
        seg.update(build_corrected_2026(Path(args.raw_json), csv26, btv.END_DATE, out / "corrected_2026.csv"))
        extra = btv.load_bars(out / "corrected_2026.csv", start=None, end=btv.END_DATE)
    bars += extra
    bars.sort(key=lambda b: b.timestamp)
    print(f"[{tag}] {len(bars):,} bars {bars[0].timestamp} -> {bars[-1].timestamp}", flush=True)

    # 4. replay
    g0 = guard([root, MAIN])
    symbol = getattr(btv, "INSTRUMENTS", {}).get("mnq", {}).get("symbol", "MNQM26")   # G0' engine may predate INSTRUMENTS
    trades = await btv.run_backtest(bars, ml_threshold=args.ml_threshold, symbol=symbol)
    g1 = guard([root, MAIN])
    report, trade_rows, _eq = btv.build_report(trades, btv.START_DATE, btv.END_DATE)
    (out / f"report_{tag}.txt").write_text(report)
    with open(out / f"trades_{tag}.csv", "w", newline="") as f:
        if trade_rows:
            w = csv.DictWriter(f, fieldnames=trade_rows[0].keys())
            w.writeheader()
            w.writerows(trade_rows)
    meta = {"tag": tag, "preregistration": args.preregistration, "ml_threshold": args.ml_threshold,
            "engine_root": str(root), "engine_head": git(root, "rev-parse", "HEAD").stdout.decode().strip(),
            "harness_sha256": sha256(here), "pinned": pinned, "seal_yaml_sha256": sha256(yaml_path),
            "effective_config": {k: str(v) for k, v in dataclasses.asdict(sealed_cfg).items()},
            "bars": len(bars), "segments": seg, "trades": len(trades),
            "guard_changed": [k for k in g0 if g0[k] != g1[k]]}
    (out / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("tag", "bars", "trades", "guard_changed")}), flush=True)
    return meta


# ── score ───────────────────────────────────────────────────────────────────────────────

def read_rows(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def arm_stats(rows: list[dict]) -> dict:
    x = np.array([float(r["pnl"]) for r in rows])
    n = len(x)
    if n == 0:
        return {"n": 0, "pnl": 0.0, "pf": None}
    days = [datetime.fromisoformat(r["entry_time"]).astimezone(ET).date() for r in rows]
    dev = x - x.mean()
    cl = {}
    for d, v in zip(days, dev):
        cl[d] = cl.get(d, 0.0) + v
    se = float(np.sqrt(sum(v * v for v in cl.values())) / n) if n > 1 else float("nan")
    return {"n": n, "pnl": float(x.sum()), "pf": pf(x), "mean": float(x.mean()), "wins": int((x > 0).sum()),
            "t_cluster": float(x.mean() / se) if se and se > 0 else None,
            "ci95_mean_cluster": [float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)] if se == se else None}


def segment(rows: list[dict], lo: datetime | None, hi: datetime | None) -> list[dict]:
    out = []
    for r in rows:
        t = datetime.fromisoformat(r["entry_time"])
        if (lo is None or t >= lo) and (hi is None or t < hi):
            out.append(r)
    return out


def decide(ml: dict, noml: dict) -> str:
    if ml["n"] < N_MIN:
        return "INCONCLUSIVE -> ML disabled (ml_threshold 0.0)"
    if ml["pf"] is not None and noml["pf"] is not None and ml["pf"] > noml["pf"] and ml["pf"] >= PF_MIN:
        return "KEEP ml_threshold 0.50"
    return "REVERT ml_threshold to 0.0"


def score(args) -> dict:
    out = Path(args.out_dir)
    res: dict = {"G0": {}}
    for arm in ("ml", "noml"):
        p = out / f"trades_original_{arm}.csv"
        res["G0"][arm] = {"present": p.exists(),
                          "reproduces_june_row_for_row": p.exists() and read_rows(p) == read_rows(JUNE_CSV[arm])}
    g0 = all(v["reproduces_june_row_for_row"] for v in res["G0"].values())
    res["G0_passed"] = g0
    segs = {"S1": (JAN1, CUTOFF), "S2": (CUTOFF, ROLL_OPEN), "S3": (ROLL_OPEN, None), "2026": (JAN1, None)}
    for mode in ("original", "corrected"):
        for arm in ("ml", "noml"):
            p = out / f"trades_{mode}_{arm}.csv"
            if p.exists():
                rows = read_rows(p)
                res[f"{mode}_{arm}"] = {k: arm_stats(segment(rows, lo, hi)) for k, (lo, hi) in segs.items()}
    if not g0:
        res["verdict"] = "NOT INTERPRETED: G0 failed (run G0' with the engine at 138cab1, else STOP)"
    elif "corrected_ml" in res and "corrected_noml" in res:
        ml, noml = res["corrected_ml"]["2026"], res["corrected_noml"]["2026"]
        res["verdict"] = decide(ml, noml)
        res["halt_review_recommended"] = ml["pnl"] < 0 and noml["pnl"] < 0
    else:
        res["verdict"] = "PENDING: corrected runs missing"
    (out / "score.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--preregistration", required=True)
    r.add_argument("--mode", choices=("original", "corrected"), required=True)
    r.add_argument("--ml-threshold", type=float, required=True, choices=(0.5, 0.0))
    r.add_argument("--out-dir", required=True)
    r.add_argument("--engine-root", default=str(Path(__file__).resolve().parents[1]))
    r.add_argument("--data-root", default=str(MAIN))
    r.add_argument("--raw-json", default=str(RAW_JSON))
    r.add_argument("--access-log", default=str(MAIN / "data/sealed_holdout/ACCESS_LOG.md"))
    s = sub.add_parser("score")
    s.add_argument("--out-dir", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
