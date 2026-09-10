"""Independent new-bar comparison against the prior native research measurement."""

import json, hashlib, sys
from pathlib import Path

ORACLE = Path(
    "/root/Silver-Bullet-ML-BMAD/_bmad-output/planning-artifacts/research/technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/mismatches-bars.json"
)
EXPECTED = "03732c05f1d75787e841a8ab98816aeb0264f784acbe08f2ea35bab2c29fb212"
M = 60_000_000_000


def require(x, msg):
    if not x:
        raise ValueError(msg)


def read(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def validate_capture_bar(bar):
    require(bar["end_ns"] == bar["start_ns"] + M, "capture interval width")
    require(bar["availability_ns"] >= bar["end_ns"], "capture availability before end")
    require(bar["incomplete_event"] is False, "incomplete capture event")


def interval_status(status, start, end):
    # Independent segmentation of positive-duration intervals, including changes
    # exactly at the left boundary and excluding the right boundary.
    prior = [value for ts, value in status if ts <= start]
    at_start = prior[-1] if prior else None
    current = at_start
    seen = set()
    position = start
    for ts, value in status:
        if start < ts < end:
            if ts > position:
                seen.add(current)
            current = value
            position = ts
    seen.add(current)
    classification = (
        "UNKNOWN"
        if None in seen
        else "MIXED" if len(seen) > 1 else "TRADING" if True in seen else "NONTRADING"
    )
    return at_start, current, classification


def check(capture, exchange, coverage):
    data = ORACLE.read_bytes()
    require(hashlib.sha256(data).hexdigest() == EXPECTED, "oracle hash")
    oracle = json.loads(data)
    counts = {}
    raw_digest_data = Path("/tmp/yank-minute-raw-digests.json").read_bytes()
    require(
        hashlib.sha256(raw_digest_data).hexdigest()
        == "8b427bcb214ddef3b2db039c7acde3a5f1f007c4e04c914ef13ac9523accdeb2",
        "independent raw digest oracle hash",
    )
    raw_digests = json.loads(raw_digest_data)
    for clock, path in [("capture", capture), ("exchange_diagnostic", exchange)]:
        bars = read(path)
        refs = oracle[clock]
        require(len(bars) == len(refs) == 13440, clock + " bar count")
        starts = []
        for b in bars:
            if clock == "capture":
                validate_capture_bar(b)
            start = b["start_ns"]
            starts.append(start)
            expected = refs.get(str(start // M + 1))
            require(expected is not None, clock + " minute absent from oracle")
            require(
                b["trade_sha256"] == raw_digests[clock][str(start)]["sha256"],
                "raw trade byte hash",
            )
            require(
                b["trade_count"] == raw_digests[clock][str(start)]["count"],
                "raw trade count",
            )
            require(b["ohlcv"] == expected["raw"], clock + " OHLCV at " + str(start))
            for side in ("first", "last"):
                for field in (
                    "file",
                    "record_index",
                    "sequence",
                    "ts_recv_ns",
                    "ts_event_ns",
                ):
                    require(
                        b[side][field] == expected[side][field],
                        clock + " " + side + " " + field,
                    )
            require(b["end_ns"] == start + M, clock + " interval width")
        require(starts == sorted(set(starts)), clock + " ordering/unique")
        counts[clock] = len(bars)
    rows = read(coverage)
    require(
        [r["start_ns"] for r in rows]
        == list(range(1747612800000000000, 1748649600000000000, M)),
        "full coverage grid",
    )
    captured = {r["start_ns"]: r for r in read(capture)}
    for r in rows:
        require(r["end_ns"] == r["start_ns"] + M, "coverage width")
        b = captured.get(r["start_ns"])
        require(r["ohlcv"] == (None if b is None else b["ohlcv"]), "coverage OHLC")
        require(
            r["trade_count"] == (0 if b is None else b["trade_count"]),
            "coverage trade count",
        )
    import databento_dbn as dbn
    import zstandard

    status = []
    native = Path(
        "/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907/native"
    )
    for path in sorted(native.glob("*/*.status.dbn.zst")):
        with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as rd:
            decoded = dbn.DBNDecoder().write_and_decode(rd.read())[1:]
        status.extend((int(x.ts_recv), x.is_trading) for x in decoded)
    status.sort(key=lambda x: x[0])
    source_dates = {
        p.name.split("-")[2].split(".")[0] for p in native.glob("*/*.mbo.dbn.zst")
    }
    from datetime import datetime, timezone

    for row in rows:
        at_start, at_end, classification = interval_status(
            status, row["start_ns"], row["end_ns"]
        )
        require(row["interval_status"] == classification, "official interval status")
        for field, value in [("status_at_start", at_start), ("status_at_end", at_end)]:
            actual = row[field]
            require(
                (None if actual is None else actual["is_trading"]) is value,
                "official " + field,
            )
        if row["ohlcv"] is None:
            expected = {
                "UNKNOWN": "NO_TRADE_STATUS_UNKNOWN",
                "MIXED": "NO_TRADE_MIXED_STATUS",
                "TRADING": "NO_TRADE_OBSERVED_TRADING",
                "NONTRADING": "NO_TRADE_OBSERVED_NONTRADING",
            }[classification]
            require(
                row["coverage"] == expected,
                "official status classification at " + str(row["start_ns"]),
            )
        date = datetime.fromtimestamp(
            row["start_ns"] // 1_000_000_000, timezone.utc
        ).strftime("%Y%m%d")
        require(
            row["mbo_source_file_present"] == (date in source_dates),
            "source date availability",
        )
    require(
        sum(b["trade_count"] for b in captured.values()) == 7526752,
        "trade count vs independent event scan",
    )
    delayed = [b for b in captured.values() if b["availability_ns"] > b["end_ns"]]
    require(len(delayed) == 1, "delayed bar count")
    require(
        delayed[0]["availability_ns"] == 1748349300000002948,
        "independent terminator availability",
    )
    require(
        delayed[0]["completion_ref"]["record_index"] == 4106159,
        "independent terminator source",
    )
    return {
        "status": "PASS_INDEPENDENT_NATIVE_COMPARISON",
        "oracle_sha256": EXPECTED,
        "matched_rows": counts,
        "coverage_rows": len(rows),
        "trade_count": 7526752,
        "delayed_bars": 1,
        "raw_trade_digests_checked": 26880,
        "official_status_coverage": "PASS",
        "method": "All OHLCV and first/last native references compared to hash-pinned prior independent native research; trade/event totals cross-checked by fresh full native probe.",
    }


if __name__ == "__main__":
    print(json.dumps(check(*sys.argv[1:]), sort_keys=True, indent=2))
