"""Read-only native trade-print evidence for the reviewed May 28 replay cases."""

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import zstandard
import databento_dbn as dbn

ROOT = Path("/root/Silver-Bullet-ML-BMAD-yank-minute")
DATA = Path("/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907")
REPLAY = ROOT / "data/yank/native-minute-reviewed-a/replay.json"
EXPECTED_REPLAY = "a6a10e53cd541b083260c8a0a03dc391b458122226e8111de99118359c63382b"
CHUNK_RECORDS = 65536
D = np.dtype(
    {
        "names": [
            "length",
            "rtype",
            "instrument",
            "event",
            "price",
            "size",
            "flags",
            "action",
            "recv",
            "sequence",
        ],
        "formats": ["u1", "u1", "<u4", "<u8", "<i8", "<u4", "u1", "u1", "<u8", "<u4"],
        "offsets": [0, 1, 4, 8, 24, 32, 36, 38, 40, 52],
        "itemsize": 56,
    }
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for raw in iter(lambda: f.read(8 << 20), b""):
            h.update(raw)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp(ns):
    return (
        datetime.fromtimestamp(ns // 10**9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{ns%10**9:09d}Z"
    )


def chunks(path):
    with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        head = stream.read(8)
        require(head[:4] == b"DBN\x03" and len(head) == 8, "DBNv3 header")
        n = struct.unpack("<I", head[4:])[0]
        meta = stream.read(n)
        require(len(meta) == n, "metadata framing")
        require(
            len(dbn.DBNDecoder().write_and_decode(head + meta)) == 1, "metadata decode"
        )
        index = 0
        carry = b""
        while raw := stream.read(56 * CHUNK_RECORDS):
            raw = carry + raw
            complete = len(raw) // 56 * 56
            carry = raw[complete:]
            raw = raw[:complete]
            a = np.frombuffer(raw, dtype=D)
            require(
                bool(
                    np.all(
                        (a["length"] == 14)
                        & (a["rtype"] == 160)
                        & (a["instrument"] == 42009475)
                    )
                ),
                "record identity",
            )
            yield index, raw, a
            index += len(a)
        require(not carry, "truncated record")


def main():
    require(sha(REPLAY) == EXPECTED_REPLAY, "reviewed replay hash")
    replay = json.loads(REPLAY.read_text())
    pins = json.loads((ROOT / "src/research/yank_native_minute/pins.json").read_text())
    pin = next(p for p in pins["files"] if p["file"].endswith("20250528.mbo.dbn.zst"))
    path = DATA / pin["file"]
    require(
        path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
        "native hash",
    )
    unique = {}
    for arm, a in replay["arms"].items():
        for t in a["trades"]:
            key = (
                t["signal_time"],
                t["entry_price"],
                t["sl_price"],
                t["tp_price"],
                t["exit_time"],
            )
            if key not in unique:
                unique[key] = {"trade": t, "arms": []}
            unique[key]["arms"].append(arm)
    cases = []
    for group in unique.values():
        t = group["trade"]
        for delay in [0, 100, 500]:
            cases.append(
                dict(
                    arms=group["arms"],
                    signal_time=t["signal_time"],
                    modeled_fill_time=t["fill_time"],
                    modeled_exit_time=t["exit_time"],
                    contracts=t["contracts"],
                    limit_nanos=int(t["entry_price"] * 10**9),
                    stop_nanos=int(t["sl_price"] * 10**9),
                    target_nanos=int(t["tp_price"] * 10**9),
                    delay_ms=delay,
                    arrival_ns=t["timing"]["signal_time"]["decision_available_ns"]
                    + delay * 10**6,
                    end_ns=t["timing"]["exit_time"]["interval_end_ns"],
                    hits={},
                    at_limit_volume=0,
                    strictly_above_volume=0,
                    at_or_above_trade_count=0,
                )
            )
    refs = {}
    pending = []
    last_end = -1
    previous_recv = 0
    total = 0

    def ref(i, raw, a, j, start, end):
        index = i + int(j)
        if index not in refs:
            row = a[j]
            refs[index] = dict(
                record_index=index,
                sequence=int(row["sequence"]),
                ts_recv_ns=int(row["recv"]),
                ts_recv_utc=stamp(int(row["recv"])),
                ts_event_ns=int(row["event"]),
                ts_event_utc=stamp(int(row["event"])),
                price_nanos=int(row["price"]),
                size=int(row["size"]),
                action=chr(int(row["action"])),
                flags=int(row["flags"]),
                raw_hex=raw[int(j) * 56 : (int(j) + 1) * 56].hex(),
                event_start_record=start,
                event_end_record=end,
            )
            if end is None:
                pending.append(index)
        return index

    for i, raw, a in chunks(path):
        total += len(a)
        ends = np.flatnonzero(a["flags"] & 128)
        if len(ends):
            for index in pending:
                refs[index]["event_end_record"] = i + int(ends[0])
            pending.clear()
        observed = a["recv"][(a["flags"] & 32) == 0]
        observed = observed[observed != 2**64 - 1]
        if len(observed):
            require(
                int(observed[0]) >= previous_recv
                and bool(np.all(observed[1:] >= observed[:-1])),
                "capture regression",
            )
            previous_recv = int(observed[-1])
        is_trade = (a["action"] == ord("T")) & ((a["flags"] & 32) == 0)
        for c in cases:
            ix = np.flatnonzero(
                is_trade & (a["recv"] >= c["arrival_ns"]) & (a["recv"] < c["end_ns"])
            )
            if not len(ix):
                continue
            require(
                not bool(np.any(a["flags"][ix] & 8)), "bad capture flag on evidence"
            )
            prices = a["price"][ix]
            c["at_limit_volume"] += int(a["size"][ix[prices == c["limit_nanos"]]].sum())
            c["strictly_above_volume"] += int(
                a["size"][ix[prices > c["limit_nanos"]]].sum()
            )
            c["at_or_above_trade_count"] += int(
                np.count_nonzero(prices >= c["limit_nanos"])
            )
            conditions = {
                "first_touch": prices == c["limit_nanos"],
                "first_through": prices > c["limit_nanos"],
                "first_stop_since_arrival": prices >= c["stop_nanos"],
                "first_target_since_arrival": prices <= c["target_nanos"],
            }
            for label, mask in conditions.items():
                if label not in c["hits"] and bool(np.any(mask)):
                    j = int(ix[np.flatnonzero(mask)[0]])
                    pos = int(np.searchsorted(ends, j))
                    c["hits"][label] = ref(
                        i,
                        raw,
                        a,
                        j,
                        (i + int(ends[pos - 1]) + 1) if pos else last_end + 1,
                        (i + int(ends[pos])) if pos < len(ends) else None,
                    )
            if "first_through" in c["hits"]:
                entry = c["hits"]["first_through"]
                for label, mask in [
                    ("first_stop_at_or_after_through", prices >= c["stop_nanos"]),
                    ("first_target_at_or_after_through", prices <= c["target_nanos"]),
                ]:
                    eligible = mask & ((ix + i) >= entry)
                    if label not in c["hits"] and bool(np.any(eligible)):
                        j = int(ix[np.flatnonzero(eligible)[0]])
                        pos = int(np.searchsorted(ends, j))
                        c["hits"][label] = ref(
                            i,
                            raw,
                            a,
                            j,
                            (i + int(ends[pos - 1]) + 1) if pos else last_end + 1,
                            (i + int(ends[pos])) if pos < len(ends) else None,
                        )
        if len(ends):
            last_end = i + int(ends[-1])
    require(not pending, "unterminated supporting event")
    # Resolve selected event endpoints directly from native positions in a second pass.
    endpoints = {r["event_end_record"] for r in refs.values()}
    boundary_refs = {}
    for i, raw, a in chunks(path):
        for index in sorted(endpoints):
            if i <= index < i + len(a):
                j = index - i
                v = a[j]
                require(int(v["flags"]) & 128, "event endpoint LAST flag")
                boundary_refs[index] = dict(
                    record_index=index,
                    flags=int(v["flags"]),
                    ts_recv_ns=int(v["recv"]),
                    ts_event_ns=int(v["event"]),
                    sequence=int(v["sequence"]),
                    action=chr(int(v["action"])),
                    price_nanos=int(v["price"]),
                    size=int(v["size"]),
                    raw_hex=raw[j * 56 : (j + 1) * 56].hex(),
                )
    require(set(boundary_refs) == endpoints, "event endpoint coverage")
    for r in refs.values():
        end = boundary_refs[r["event_end_record"]]
        require(end["ts_recv_ns"] >= r["ts_recv_ns"], "event endpoint timing")
    # Independently decode selected trade and LAST bytes with the official decoder.
    for r in list(refs.values()) + list(boundary_refs.values()):
        decoder = dbn.DBNDecoder(has_metadata=False)
        values = decoder.write_and_decode(bytes.fromhex(r["raw_hex"]))
        require(len(values) == 1, "official reference count")
        v = values[0]
        require(
            int(v.ts_recv) == r["ts_recv_ns"]
            and int(v.ts_event) == r["ts_event_ns"]
            and int(v.price) == r["price_nanos"]
            and int(v.size) == r["size"]
            and str(v.action) == r["action"]
            and int(v.sequence) == r["sequence"],
            "official reference fields",
        )
    for c in cases:
        entry = refs.get(c["hits"].get("first_through"))
        stop = refs.get(c["hits"].get("first_stop_at_or_after_through"))
        c["entry_evidence"] = (
            "TRADE_THROUGH_SUPPORT"
            if entry
            else "TOUCH_ONLY" if "first_touch" in c["hits"] else "NO_ENTRY_SUPPORT"
        )
        c["exchange_clock_entry_precedes_stop"] = (
            None if not stop else entry["ts_event_ns"] < stop["ts_event_ns"]
        )
        c["entry_stop_ordering"] = (
            "NO_SUBSEQUENT_STOP_PRINT"
            if not stop
            else (
                "AMBIGUOUS_EQUAL_TIME_OR_SAME_EVENT"
                if entry["ts_recv_ns"] == stop["ts_recv_ns"]
                or entry["event_end_record"] == stop["event_end_record"]
                else "THROUGH_PRECEDES_STOP_IN_CAPTURE_AND_NATIVE_ORDER"
            )
        )
    require(
        sha(path) == pin["sha256"] and sha(REPLAY) == EXPECTED_REPLAY, "inputs changed"
    )
    print(
        json.dumps(
            dict(
                research_status="HOLD_VALIDATION",
                scope="Trade-print timeline from signal completion through modeled exit minute; no book/queue reconstruction or revised PNL",
                native_file=pin["file"],
                native_sha256=pin["sha256"],
                replay_sha256=EXPECTED_REPLAY,
                script_sha256=sha(Path(__file__)),
                records_scanned=total,
                official_decoder_reference_checks=len(refs) + len(boundary_refs),
                event_endpoints=[boundary_refs[k] for k in sorted(boundary_refs)],
                cases=cases,
                references=[refs[k] for k in sorted(refs)],
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
