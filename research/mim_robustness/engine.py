"""Common execution model, explicitly distinct reference and actual realized P&L."""

from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
import pandas as pd
from research.mim_comparison.references import author_vectorized
from .features import features, gate
from .artifacts import CONFIG


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


ARMS = [Arm(name) for name in CONFIG["arms"]]


def simulate(session, sigma, arm, delay=2, cost=2.24, quantity=1, feature_values=None):
    values = (
        features(session, np.asarray(sigma))
        if feature_values is None
        else feature_values
    )
    ready = True
    reset_bar = -1
    armed_bar = -1
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
        nonlocal p, entry, anchor, reference, gross, turnover, deactivated, ready, reset_bar
        old = p
        if arm.name == "F" and old and reason in ("CAT_STOP", "BAND_STOP", "REVERSAL"):
            ready = False
            reset_bar = i
        if old:
            gross += old * (fill - entry) * 2 * quantity
            reference += old * (ref - anchor) * 2  # deployed one-contract guard
            if reference <= CONFIG["reference_guard"]:
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
            # Preserve baseline: stale flat exits skip, accepted nonzero pending
            # reversals may still fill after an intervening protective stop.
            if target != p and not (target and arm.guarded and deactivated):
                event(i, target, float(row.open), ref, reason)
            pending = None
        if p:
            exposure += abs(p) * quantity
            stop = anchor - p * CONFIG["stop_points"]
            if arm.guarded and (
                (p == 1 and row.low <= stop) or (p == -1 and row.high >= stop)
            ):
                fill = (
                    min(float(row.open), stop) if p == 1 else max(float(row.open), stop)
                )
                event(i, 0, fill, stop, "CAT_STOP")
        if arm.name == "F" and not p and not ready and i > reset_bar:
            if values["lower"][i] <= c <= values["upper"][i]:
                ready = True
                armed_bar = i
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
        exiting_band = reason == "BAND_STOP"
        gate_pass = (
            gate(arm.name, direction, i, values, ready and i > armed_bar)
            if direction
            else False
        )
        if arm.name == "F" and exiting_band:
            gate_pass = False
        entry_attempt = bool(direction and direction != target)
        # Model signal-close reference exit before deciding reentry, including threshold equality.
        projected = reference + (p * (c - anchor) * 2 if p and target != p else 0)
        if (
            direction
            and gate_pass
            and direction != target
            and not (
                arm.guarded and (deactivated or projected <= CONFIG["reference_guard"])
            )
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
                "direction": direction,
                "entry_attempt": entry_attempt,
                "guard_deactivated": deactivated,
                "projected_reference_gross": projected,
                "risk_blocked": bool(
                    entry_attempt
                    and arm.guarded
                    and (deactivated or projected <= CONFIG["reference_guard"])
                ),
                "entry_disposition": (
                    "no_entry_attempt"
                    if not entry_attempt
                    else (
                        "daily_guard"
                        if arm.guarded
                        and (deactivated or projected <= CONFIG["reference_guard"])
                        else "filter_rejected" if not gate_pass else "accepted"
                    )
                ),
                "gate_pass": gate_pass,
                "gate_reason": (
                    "pass"
                    if gate_pass
                    else ("no_breakout" if not direction else "entry_gate_rejected")
                ),
                "reset_ready": ready,
                "slope": float(values["slope"][i]),
                "r2": float(values["r2"][i]),
                "efficiency": float(values["efficiency"][i]),
                "displacement": float(values["displacement"][i]),
                "persistence_long": bool(values["persistence_long"][i]),
                "persistence_short": bool(values["persistence_short"][i]),
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
