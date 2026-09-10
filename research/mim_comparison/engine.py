"""Common execution model, explicitly distinct reference and actual realized P&L."""

from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
import pandas as pd
from .references import author_vectorized


@dataclass(frozen=True)
class Arm:
    name: str
    published_gap: bool = False
    confirmation: bool = False
    neutral_exit: bool = False
    guarded: bool = True

    def digest(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


ARMS = [
    Arm("A"),
    Arm("B", True, True, True),
    Arm("C", guarded=False),
    Arm("D", True, True, True, False),
    Arm("gap", True),
    Arm("confirmation", confirmation=True),
    Arm("exit", neutral_exit=True),
]


def simulate(session, sigma, arm, delay=2, cost=2.24, quantity=1):
    bars = session.bars
    o = float(bars.open.iloc[0])
    prev = session.previous_close
    p = 0
    entry = 0.0
    anchor = 0.0
    reference = 0.0
    gross = 0.0
    turnover = 0
    exposure = 0
    pending = None
    ledger = []
    decisions = []
    deactivated = False
    published = author_vectorized(bars, prev, sigma)
    volumes = bars.volume.to_numpy(float)
    deployed_vwap = np.divide(
        np.cumsum(bars.close.to_numpy(float) * volumes),
        np.cumsum(volumes),
        out=bars.close.to_numpy(float).copy(),
        where=np.cumsum(volumes) != 0,
    )

    def event(i, delta, fill, ref, reason):
        nonlocal p, entry, anchor, reference, gross, turnover, deactivated
        old = p
        if old:
            gross += old * (fill - entry) * 2 * quantity
            reference += old * (ref - anchor) * 2  # deployed one-contract guard
            if reference <= -1000:
                deactivated = True
        turnover += abs(delta - old) * quantity
        ledger.append(
            {
                "day": session.day,
                "contract": session.contract,
                "arm": arm.name,
                "arm_hash": arm.digest(),
                "event_timestamp": bars.timestamp.iloc[i].isoformat(),
                "modeled_fill_timestamp": (
                    (
                        bars.timestamp.iloc[i]
                        if reason == "EOD_CLOSE_PROXY"
                        else bars.timestamp.iloc[i] - pd.Timedelta(minutes=1)
                    ).isoformat()
                    if reason != "CAT_STOP"
                    else None
                ),
                "fill_time_basis": (
                    "intrabar_unknown_within_event_minute"
                    if reason == "CAT_STOP"
                    else (
                        "session_close_proxy"
                        if reason == "EOD_CLOSE_PROXY"
                        else "bar_open"
                    )
                ),
                "receipt_timestamp": (
                    str(bars.received_at.iloc[i]) if "received_at" in bars else None
                ),
                "signal_price": float(ref),
                "fill": float(fill),
                "quantity": int(abs(delta - old) * quantity),
                "position_after": delta * quantity,
                "costs": abs(delta - old) * quantity * cost / 2,
                "reason": reason,
                "eligible": True,
                "exclusion": None,
                "reference_realized_gross": reference,
            }
        )
        p = delta
        entry = fill
        anchor = ref

    for i, row in enumerate(bars.itertuples(index=False)):
        c = float(row.close)
        if pending and pending[0] == i:
            _, target, ref, reason = pending
            # A stop may have flattened the position during latency: no stale exit or resurrection.
            if target != p and not (target and arm.guarded and deactivated):
                event(i, target, float(row.open), ref, reason)
            pending = None
        if p:
            exposure += abs(p) * quantity
            stop = anchor - p * 250
            if arm.guarded and (
                (p == 1 and row.low <= stop) or (p == -1 and row.high >= stop)
            ):
                fill = (
                    min(float(row.open), stop) if p == 1 else max(float(row.open), stop)
                )
                event(i, 0, fill, stop, "CAT_STOP")
        minute = int(row.minute)
        if minute == 960:
            if p:
                event(i, 0, c, c, "EOD_CLOSE_PROXY")
            pending = None
            continue
        if (minute - 570) % 30 or minute < 600:
            continue
        s = float(sigma[i])
        if not np.isfinite(s):
            continue
        ub = (
            max(o, prev) * (1 + s)
            if arm.published_gap
            else o * (1 + s) + max(prev - o, 0)
        )
        lb = (
            min(o, prev) * (1 - s)
            if arm.published_gap
            else o * (1 - s) - max(o - prev, 0)
        )
        direction = 1 if c > ub else -1 if c < lb else 0
        if arm.confirmation and not (
            (direction == 1 and c > published["vwap"][i])
            or (direction == -1 and c < published["vwap"][i])
        ):
            direction = 0
        target = p
        reason = "HOLD"
        if (p == 1 and c < lb) or (p == -1 and c > ub):
            target = 0
            reason = "BAND_STOP"
        if arm.neutral_exit and direction == 0:
            target = 0
            reason = "SAMPLED_NEUTRAL"
        # Model signal-close reference exit before deciding reentry, including threshold equality.
        projected = reference + (p * (c - anchor) * 2 if p and target != p else 0)
        if (
            direction
            and direction != target
            and not (arm.guarded and (deactivated or projected <= -1000))
        ):
            target = direction
            reason = "REVERSAL" if p else "ENTRY"
        decisions.append(
            {
                "day": session.day,
                "contract": session.contract,
                "arm": arm.name,
                "arm_hash": arm.digest(),
                "event_timestamp": row.timestamp.isoformat(),
                "receipt_timestamp": str(getattr(row, "received_at", None)),
                "signal_price": c,
                "sigma": s,
                "upper": ub,
                "lower": lb,
                "vwap": (
                    (
                        float(published["vwap"][i])
                        if np.isfinite(published["vwap"][i])
                        else None
                    )
                    if arm.confirmation
                    else float(deployed_vwap[i])
                ),
                "position": p,
                "target": target,
                "reference_realized_gross": reference,
                "reason": reason,
                "eligible": True,
                "exclusion": None,
            }
        )
        if target != p and i + delay < len(bars):
            pending = (i + delay, target, c, reason)
    return (
        {
            "day": session.day,
            "contract": session.contract,
            "arm": arm.name,
            "arm_hash": arm.digest(),
            "delay": delay,
            "cost": cost,
            "quantity": quantity,
            "gross": gross,
            "costs": turnover * cost / 2,
            "net": gross - turnover * cost / 2,
            "turnover": turnover,
            "exposure_contract_minutes": exposure,
            "eligible": True,
            "exclusion": None,
        },
        ledger,
        decisions,
    )


def sizing_quantity(equity, opened, returns, day_index, dynamic=True):
    # Author returns[d-15:d-1] excludes yesterday; 14 values, sample stdev.
    lag = np.asarray(returns[max(0, day_index - 15) : max(0, day_index - 1)], float)
    vol = float(np.std(lag, ddof=1)) if len(lag) == 14 else float("nan")
    leverage = min(4.0, 0.02 / vol) if np.isfinite(vol) and vol > 0 else 4.0
    if not dynamic:
        leverage = 1.0
    return max(0, round(max(0.0, equity) * leverage / (opened * 2))), vol, leverage
