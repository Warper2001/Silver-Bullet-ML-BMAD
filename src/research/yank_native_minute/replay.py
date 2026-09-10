"""Private execution of hash-pinned frozen sources, with interval annotations."""

import ast
import builtins
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import sys
import types
from pathlib import Path

REPLAY_ROOT = Path("/root/Silver-Bullet-ML-BMAD-yank-replay")
MODEL = Path("/root/Silver-Bullet-ML-BMAD/models/xgboost/tier2_meta_labeling_model.pkl")
PREFIX = "_yank_native_frozen"


def load_frozen(pins):
    source = {}
    for path, expected in pins["sources"].items():
        raw = (REPLAY_ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("frozen source hash mismatch: " + path)
        source[path] = raw
    # Modules use a private package and a narrow import remapping. The exact pinned
    # source bytes execute unchanged; src.research.__init__ is never imported.
    for name in [PREFIX, PREFIX + ".signals"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    original_import = builtins.__import__

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.research.yank_replay.models":
            return sys.modules[PREFIX + ".models"]
        return original_import(name, globals, locals, fromlist, level)

    def execute(name, path):
        module = types.ModuleType(name)
        module.__file__ = str(REPLAY_ROOT / path)
        module.__package__ = name.rpartition(".")[0]
        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": private_import,
        }
        sys.modules[name] = module
        exec(compile(source[path], module.__file__, "exec"), module.__dict__)
        return module

    accounting = execute(PREFIX + ".models", "src/research/yank_replay/models.py")
    reference = execute(
        PREFIX + ".signals.reference", "src/research/yank_signals/reference.py"
    )
    model = execute(PREFIX + ".signals.model", "src/research/yank_signals/model.py")
    engine = execute(PREFIX + ".signals.engine", "src/research/yank_signals/engine.py")
    # Only the pure independent ledger reconciler is needed from evidence.py.
    # Exclude its CSV reader, publisher and all original execution entrypoints.
    path = "src/research/yank_signals/evidence.py"
    tree = ast.parse(source[path])
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("reconcile", "_reconcile")
    ]
    ns = dict(
        Decimal=Decimal,
        localcontext=localcontext,
        accounting_context=accounting.accounting_context,
    )
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(REPLAY_ROOT / path), "exec"
        ),
        ns,
    )
    return engine, model, reference, ns["reconcile"]


def replay(bars, pins, holds):
    if holds:
        return dict(
            outcome="HOLD",
            reasons=sorted(holds),
            arms={},
            research_status="HOLD_VALIDATION",
        )
    engine, model_module, reference, reconcile = load_frozen(pins)
    model = model_module.PinnedModel(MODEL)
    smoke = {k: 0.0 for k in model_module.FEATURE_COLS}
    smoke["signal_direction"] = "bearish"
    model.predict(smoke)
    converted = [
        engine.Bar(
            datetime.fromtimestamp(b["start_ns"] // 1_000_000_000, timezone.utc),
            *(p / 1_000_000_000 for p in b["ohlcv"][:4]),
            b["ohlcv"][4],
        )
        for b in bars
    ]
    by_time = {b.timestamp.isoformat(): raw for b, raw in zip(converted, bars)}
    watermark = 0
    decision_times = {}
    indices = {}
    for index, (bar, raw) in enumerate(zip(converted, bars)):
        watermark = max(watermark, raw["availability_ns"])
        decision_times[bar.timestamp.isoformat()] = watermark
        indices[bar.timestamp.isoformat()] = index
    readiness = {}
    days = []
    hours = []
    h1 = []
    volatility_count = 0
    for i, b in enumerate(converted):
        day = b.timestamp.astimezone(engine.NY).date().isoformat()
        if not days or days[-1] != day:
            days.append(day)
        hour = b.timestamp.replace(minute=0, second=0, microsecond=0)
        if not hours or hours[-1] != hour:
            hours.append(hour)
            if len(h1) >= 2 and i + 1 >= 60:
                frame = engine.frame(h1)
                high, low, close = (
                    frame[k].to_numpy() for k in ("high", "low", "close")
                )
                pc = engine.np.roll(close, 1).astype(float)
                pc[0] = engine.np.nan
                tr = engine.np.where(
                    engine.np.isnan(pc),
                    high - low,
                    engine.np.maximum(
                        high - low, engine.np.maximum(abs(high - pc), abs(low - pc))
                    ),
                )
                hist = [
                    v
                    for v in engine.pd.Series(tr)
                    .rolling(20, min_periods=5)
                    .mean()
                    .dropna()
                    if v > 0
                ][-120:]
                volatility_count = len(hist)
            h1.append(engine.Bar(hour, b.open, b.high, b.low, b.close, b.volume))
        else:
            h = h1[-1]
            h1[-1] = engine.Bar(
                hour,
                h.open,
                max(h.high, b.high),
                min(h.low, b.low),
                b.close,
                h.volume + b.volume,
            )
        readiness[b.timestamp.isoformat()] = dict(
            observed_closes=i + 1,
            lr_full_history=i + 1 >= 1950,
            preceding_observed_days=min(20, len(days) - 1),
            adr_full_history=len(days) > 20,
            completed_observed_h1_buckets=max(0, len(hours) - 1),
            volatility_positive_atr_observations=volatility_count,
            volatility_minimum_history=volatility_count >= 20,
            volatility_full_history=volatility_count >= 120,
        )
    arms = {}
    for name, adapter in [("no-ml", None), ("ml050", model)]:
        result = engine.ReplayEngine(adapter, classification="DEVELOPMENT_REPLAY").run(
            converted
        )
        result["independent_reconciliation"] = reconcile(result)
        for row in result["gates"]:
            if row["outcome"] == "ORDER":
                index = indices[row["timestamp"]]
                if (
                    index + 1 < len(bars)
                    and decision_times[row["timestamp"]] > bars[index + 1]["start_ns"]
                ):
                    return dict(
                        outcome="HOLD",
                        reasons=["ORDER_AVAILABILITY_OVERLAPS_NEXT_BAR"],
                        unsafe_order={
                            "arm": name,
                            "signal_start_ns": bars[index]["start_ns"],
                            "availability_ns": decision_times[row["timestamp"]],
                            "next_start_ns": bars[index + 1]["start_ns"],
                        },
                        arms={},
                        research_status="HOLD_VALIDATION",
                    )
        for row in result["gates"]:
            raw = by_time[row["timestamp"]]
            row.update(
                interval_start_ns=raw["start_ns"],
                interval_end_ns=raw["end_ns"],
                bar_available_ns=raw["availability_ns"],
                decision_available_ns=decision_times[row["timestamp"]],
                cold_start=readiness[row["timestamp"]],
            )
        for row in result["events"]:
            raw = by_time[row["timestamp"]]
            row.update(
                interval_start_ns=raw["start_ns"],
                interval_end_ns=raw["end_ns"],
                bar_available_ns=raw["availability_ns"],
            )
            if row["kind"] == "FILL":
                signal = by_time[row["signal_time"]]
                if (
                    decision_times[row["signal_time"]] > raw["start_ns"]
                    or signal["start_ns"] >= raw["start_ns"]
                ):
                    raise ValueError("unsafe modeled fill interval")
                row["timing_qualification"] = (
                    "MODELED_WITHIN_BAR_INTERVAL_NOT_EXACT_FILL_TIME"
                )
            elif row["kind"] == "EXIT":
                row["timing_qualification"] = (
                    "MODELED_WITHIN_BAR_INTERVAL_INTRABAR_PATH_UNKNOWN"
                )
        for trade in result["trades"]:
            trade["timing"] = {
                key: {
                    "interval_start_ns": by_time[trade[key]]["start_ns"],
                    "interval_end_ns": by_time[trade[key]]["end_ns"],
                    "bar_available_ns": by_time[trade[key]]["availability_ns"],
                    "decision_available_ns": decision_times[trade[key]],
                }
                for key in ("signal_time", "fill_time", "exit_time")
            }
        terminal_bar = bars[-1]
        result["terminal"]["mark_interval_end_ns"] = terminal_bar["end_ns"]
        result["terminal"]["mark_available_ns"] = decision_times[
            converted[-1].timestamp.isoformat()
        ]
        arms[name] = result
    config = asdict(reference.FROZEN_CONFIG)
    config = {
        k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in config.items()
    }
    return dict(
        outcome=(
            "REPLAY_COMPLETE"
            if all(a["summary"]["outcome"] == "REPLAY_COMPLETE" for a in arms.values())
            else "HOLD"
        ),
        research_status="HOLD_VALIDATION",
        arms=arms,
        effective_config=config,
        model=model.metadata,
    )
