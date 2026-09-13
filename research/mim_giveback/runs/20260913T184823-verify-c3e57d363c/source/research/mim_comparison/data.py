"""Contract provenance, end-labelled ET normalization and causal session selection."""

from dataclasses import dataclass
from datetime import date, timedelta
import re
import numpy as np
import pandas as pd

ET = "America/New_York"


@dataclass
class Session:
    day: str
    contract: str
    bars: pd.DataFrame
    previous_close: float | None


def expiry(symbol):
    match = re.fullmatch(r"MNQ([HMUZ])(\d{2})", str(symbol))
    if not match:
        raise ValueError(f"Unestablished quarterly MNQ contract provenance: {symbol}")
    month = {"H": 3, "M": 6, "U": 9, "Z": 12}[match[1]]
    first = date(2000 + int(match[2]), month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def load(path, labels):
    if labels not in ("start", "end"):
        raise ValueError("Explicit start/end timestamp declaration required")
    frame = pd.read_csv(path)
    required = {"contract", "timestamp", "open", "high", "low", "close", "volume"}
    if not required <= set(frame):
        raise ValueError(
            "Missing contract provenance or required OHLCV columns: "
            + str(sorted(required - set(frame)))
        )
    for name in frame.contract.unique():
        expiry(name)
    # Naive timestamps are refused rather than silently treated as UTC.
    if not frame.timestamp.astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$").all():
        raise ValueError("Timestamps must declare UTC offset")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(ET)
    if labels == "start":
        frame["timestamp"] += pd.Timedelta(minutes=1)
    for name in ["open", "high", "low", "close", "volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["day"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    frame["minute"] = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
    return frame.loc[frame.minute.between(571, 960)].copy()


def audit_select(frame):
    groups, exclusions, valid = {}, [], {}
    for (day, contract), raw in frame.groupby(["day", "contract"], sort=True):
        bars = raw.sort_values("timestamp").reset_index(drop=True)
        values = bars[["open", "high", "low", "close", "volume"]].to_numpy()
        reasons = []
        if bars.timestamp.duplicated().any():
            reasons.append("duplicate_minutes")
        if len(bars) != 390 or set(bars.minute) != set(range(571, 961)):
            reasons.append("missing_minutes_or_early_close")
        if not bars.timestamp.eq(bars.timestamp.dt.floor("min")).all():
            reasons.append("nonminute_timestamp")
        if (
            not np.isfinite(values).all()
            or (bars[["open", "high", "low", "close"]] <= 0).any().any()
            or (bars.volume < 0).any()
        ):
            reasons.append("invalid_numeric")
        if (
            (bars.high < bars[["open", "close", "low"]].max(axis=1))
            | (bars.low > bars[["open", "close", "high"]].min(axis=1))
        ).any():
            reasons.append("invalid_ohlc")
        if date.fromisoformat(day).weekday() >= 5:
            reasons.append("weekend_rth_closed")
        if date.fromisoformat(day) >= expiry(contract):
            reasons.append("expired_at_session_open")
        groups.setdefault(day, {})[contract] = bars
        if reasons:
            exclusions.append(
                {
                    "day": day,
                    "contract": contract,
                    "eligible": False,
                    "exclusion": ",".join(reasons),
                }
            )
        else:
            valid[day, contract] = bars
    selected = []
    days = sorted(groups)
    if days:
        for absent in pd.bdate_range(days[0], days[-1]):
            key = absent.strftime("%Y-%m-%d")
            if key not in groups:
                exclusions.append(
                    {
                        "day": key,
                        "contract": None,
                        "eligible": False,
                        "exclusion": "missing_or_exchange_closed_unverified",
                    }
                )
    previous_day = None
    for day in sorted(groups):
        if date.fromisoformat(day).weekday() >= 5:
            continue
        candidates = []
        expected_previous = (pd.Timestamp(day) - pd.offsets.BDay(1)).strftime(
            "%Y-%m-%d"
        )
        if previous_day and previous_day == expected_previous:
            for contract in groups[previous_day]:
                # Availability can be audited afterward; never replace a selected bad day with another contract.
                prev = valid.get((previous_day, contract))
                if prev is not None and date.fromisoformat(day) < expiry(contract):
                    candidates.append(
                        (-float(prev.volume.sum()), expiry(contract), contract)
                    )
        if not candidates:
            exclusions.append(
                {
                    "day": day,
                    "contract": None,
                    "eligible": False,
                    "exclusion": "no_previous_session_contract_volume",
                }
            )
        else:
            contract = min(candidates)[2]
            if (day, contract) in valid:
                selected.append(
                    Session(
                        day,
                        contract,
                        valid[day, contract],
                        float(valid[previous_day, contract].close.iloc[-1]),
                    )
                )
            else:
                exclusions.append(
                    {
                        "day": day,
                        "contract": contract,
                        "eligible": False,
                        "exclusion": "selected_contract_incomplete_no_fallback",
                    }
                )
        previous_day = day
    return selected, exclusions
