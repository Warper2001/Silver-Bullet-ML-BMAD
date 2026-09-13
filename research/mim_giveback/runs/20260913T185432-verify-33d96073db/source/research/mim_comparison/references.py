"""Two deliberately independent reference implementations; no production imports."""

import numpy as np
import pandas as pd


def author_vectorized(bars, previous_close, sigma):
    o = float(bars.open.iloc[0])
    upper = max(o, previous_close) * (1 + np.asarray(sigma))
    lower = min(o, previous_close) * (1 - np.asarray(sigma))
    volume = bars.volume.to_numpy(float)
    vwap = np.divide(
        np.cumsum((bars.high + bars.low + bars.close).to_numpy() / 3 * volume),
        np.cumsum(volume),
        out=np.full(len(bars), np.nan),
        where=np.cumsum(volume) != 0,
    )
    signal = np.where(
        (bars.close.to_numpy() > upper) & (bars.close.to_numpy() > vwap),
        1,
        np.where(
            (bars.close.to_numpy() < lower) & (bars.close.to_numpy() < vwap), -1, 0
        ),
    )
    # Original start label offset +1: original 09:59 min_from_open=30 is end 10:00.
    sample = (bars.minute.to_numpy() - 570) % 30 == 0
    exposure = (
        pd.Series(np.where(sample, signal, np.nan))
        .ffill()
        .fillna(0)
        .shift(1)
        .fillna(0)
        .to_numpy()
    )
    gross = float(
        np.sum(exposure * np.diff(bars.close.to_numpy(), prepend=bars.close.iloc[0]))
    )
    return {
        "upper": upper,
        "lower": lower,
        "vwap": vwap,
        "signal": signal,
        "sample": sample,
        "exposure": exposure,
        "gross_points": gross,
    }


def author_sigma(moves):
    """Executable rolling14,min13,shift1; first session moves are missing."""
    frame = pd.DataFrame(moves, dtype=float)
    if len(frame):
        frame.iloc[0] = np.nan
    return frame.rolling(14, min_periods=13).mean().shift(1).to_numpy()


def deployed_event(
    close,
    opened,
    previous_close,
    sigma,
    position,
    realized,
    *,
    mark=True,
    eod=False,
    broker="normal",
    anchor=None,
):
    """Pure transcription of deployed on_bar decisions with controlled broker events.

    Rejected exits retain position; unknown stop status does not imply fill.
    External flatten uses entry reference as deployed prev_ref_price does.
    """
    upper = opened * (1 + sigma) + max(previous_close - opened, 0)
    lower = opened * (1 - sigma) - max(opened - previous_close, 0)
    actions = []
    if not mark:
        return position, realized, actions

    def record(px, why):
        nonlocal position, realized
        if anchor is None:
            raise ValueError("Position requires signal anchor")
        realized += position * (px - anchor) * 2
        position = 0
        actions.append(why)

    if eod:
        if position and broker != "exit_rejected":
            record(close, "EOD")
        return position, realized, actions
    if position and broker == "external_flatten":
        record(anchor, "EXTERNAL_FLATTEN")
    elif position and broker == "stop_filled":
        record(anchor - position * 250, "CAT_STOP")
    elif (
        position
        and ((position == 1 and close < lower) or (position == -1 and close > upper))
        and broker != "exit_rejected"
    ):
        record(close, "STOP")
    if realized > -1000:
        direction = 1 if close > upper else -1 if close < lower else 0
        if direction and direction != position:
            if position and broker != "exit_rejected":
                record(close, "REVERSAL")
            if broker != "entry_rejected":
                position = direction
                actions.append("ENTER_LONG" if direction == 1 else "ENTER_SHORT")
                if broker == "stop_rejected":
                    position = 0
                    actions.append("CAT_STOP_REJECTED")
    return position, realized, actions


def log_reconciliation(frame):
    """Rounded operational decisions cannot establish exact threshold agreement."""
    records = []
    for row in frame.to_dict("records"):
        try:
            c, u, l, s, o = (
                float(row[k]) for k in ("close", "ub", "lb", "sigma", "open_d")
            )
            # UB/LB logged to cents; sigma to six decimals. Conservative uncertainty envelope.
            tolerance = 0.005 + abs(o) * 0.0000005
            ambiguous = min(abs(c - u), abs(c - l)) <= tolerance
            records.append(
                {
                    "timestamp": row["ts_et"],
                    "recorded_action": row.get("action"),
                    "classification": (
                        "rounded_threshold_ambiguity"
                        if ambiguous
                        else "rounded_log_only_not_full_precision_parity"
                    ),
                    "long_breakout": bool(c > u),
                    "short_breakout": bool(c < l),
                    "tolerance": tolerance,
                    "contract_provenance": "unavailable",
                    "broker_reconstruction": "unavailable",
                }
            )
        except (ValueError, KeyError, TypeError):
            records.append(
                {
                    "timestamp": row.get("ts_et"),
                    "classification": "unavailable_depth_or_fields",
                }
            )
    return records
