"""Bounded DBNv3 capture-clock aggregation; no network or live imports."""

import collections
import hashlib
import struct
from datetime import datetime, timezone, timedelta

import databento_dbn as dbn
import numpy as np
import zstandard

MINUTE = 60_000_000_000
START = 1747612800000000000
END = 1748649600000000000
INSTRUMENT = 42009475
# Cover every byte: NumPy structured fancy indexing need not preserve padding.
# These fields retain exact native56-byte records through every filtered copy.
DTYPE = np.dtype(
    {
        "names": [
            "length",
            "rtype",
            "publisher",
            "id",
            "event",
            "order_id",
            "price",
            "size",
            "flags",
            "channel",
            "action",
            "side",
            "recv",
            "ts_in_delta",
            "seq",
        ],
        "formats": [
            "u1",
            "u1",
            "<u2",
            "<u4",
            "<u8",
            "<u8",
            "<i8",
            "<u4",
            "u1",
            "u1",
            "u1",
            "u1",
            "<u8",
            "<i4",
            "<u4",
        ],
        "offsets": [0, 1, 2, 4, 8, 16, 24, 32, 36, 37, 38, 39, 40, 48, 52],
        "itemsize": 56,
    }
)


def iso(ns):
    return datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc).isoformat()


def exact_read(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise ValueError("truncated native framing")
        parts.extend(part)
    return bytes(parts)


def metadata(stream, schema, day):
    head = exact_read(stream, 8)
    if head[:4] != b"DBN\x03":
        raise ValueError("expected DBNv3")
    length = struct.unpack("<I", head[4:])[0]
    if not 100 <= length <= 1_000_000:
        raise ValueError("invalid metadata length")
    raw = head + exact_read(stream, length)
    values = dbn.DBNDecoder().write_and_decode(raw)
    if len(values) != 1:
        raise ValueError("invalid metadata framing")
    meta = values[0]
    start = (
        int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
        * 1_000_000_000
    )
    if (
        str(meta.schema) != schema
        or meta.dataset != "GLBX.MDP3"
        or list(meta.symbols) != ["MNQM5"]
        or meta.start != start
        or meta.end != start + 1440 * MINUTE
        or meta.ts_out
        or meta.partial
        or meta.not_found
        or meta.mappings
        != {
            "MNQM5": [
                {
                    "start_date": datetime.strptime(day, "%Y%m%d").date(),
                    "end_date": datetime.strptime(day, "%Y%m%d").date()
                    + timedelta(days=1),
                    "symbol": "42009475",
                }
            ]
        }
    ):
        raise ValueError("native metadata identity/range mismatch")
    return raw, start


def chunks(stream, records=65536):
    if records <= 0:
        raise ValueError("invalid chunk size")
    pending = b""
    while True:
        raw = stream.read(records * 56)
        if not raw:
            break
        raw = pending + raw
        size = len(raw) // 56 * 56
        if size:
            yield np.frombuffer(raw[:size], dtype=DTYPE)
        pending = raw[size:]
    if pending:
        raise ValueError("truncated MBO record")


def reference(row, file, index):
    return {
        "file": file,
        "record_index": int(index),
        "sequence": int(row["seq"]),
        "ts_recv_ns": int(row["recv"]),
        "ts_event_ns": int(row["event"]),
    }


class Builder:
    def __init__(self, start=START, end=END):
        self.start, self.end = start, end
        self.bars = {}
        self.exchange = {}
        self.counts = collections.Counter()
        self.pending = []
        self.holds = set()
        self.files = []
        self.previous_capture = None

    def aggregate(self, rows, indices, file, clock, table):
        minutes = rows[clock] // MINUTE * MINUTE
        for minute in np.unique(minutes):
            m = int(minute)
            if not self.start <= m < self.end:
                self.counts[clock + "_outside_range"] += int(
                    np.count_nonzero(minutes == minute)
                )
                continue
            selected = np.flatnonzero(minutes == minute)
            a = rows[selected]
            prices = a["price"]
            first = reference(a[0], file, indices[selected[0]])
            last = reference(a[-1], file, indices[selected[-1]])
            if m not in table:
                table[m] = dict(
                    start_ns=m,
                    end_ns=m + MINUTE,
                    ohlcv=[
                        int(prices[0]),
                        int(prices.max()),
                        int(prices.min()),
                        int(prices[-1]),
                        0,
                    ],
                    trade_count=0,
                    first=first,
                    last=last,
                    digest=hashlib.sha256(),
                    availability_ns=m + MINUTE,
                    completion_ref=None,
                    incomplete_event=False,
                )
            bar = table[m]
            v = bar["ohlcv"]
            v[1], v[2], v[3] = (
                max(v[1], int(prices.max())),
                min(v[2], int(prices.min())),
                int(prices[-1]),
            )
            v[4] += sum(map(int, a["size"]))
            bar["last"] = last
            bar["trade_count"] += len(a)
            # Raw fixed-width records in native order; digest independent of chunks.
            bar["digest"].update(a.tobytes())

    def completion(self, minute, row, file, index):
        bar = self.bars.get(minute)
        if bar is None:
            return
        recv = int(row["recv"])
        if recv == 0 or recv >= (1 << 63) or int(row["flags"]) & 8:
            bar["incomplete_event"] = True
            self.holds.add("INVALID_TRADE_EVENT_TERMINATOR_TIME")
        elif recv >= bar["availability_ns"]:
            bar["availability_ns"] = recv
            bar["completion_ref"] = reference(row, file, index)

    def consume(self, a, file, offset):
        if not np.all(
            (a["length"] == 14)
            & (a["rtype"] == 160)
            & (a["id"] == INSTRUMENT)
            & (a["publisher"] == 1)
        ):
            raise ValueError("unexpected MBO framing/schema/instrument/publisher")
        if not np.isin(a["action"], list(b"ACMRTFN")).all():
            raise ValueError("unexpected MBO action")
        capture = a["recv"][(a["flags"] & 32) == 0]
        if len(capture):
            valid_capture = capture[(capture > 0) & (capture < (1 << 63))]
            if len(valid_capture):
                if np.any(valid_capture[1:] < valid_capture[:-1]) or (
                    self.previous_capture is not None
                    and int(valid_capture[0]) < self.previous_capture
                ):
                    self.holds.add("NONMONOTONE_NATIVE_CAPTURE")
                self.previous_capture = int(valid_capture[-1])
        self.counts["records"] += len(a)
        snapshot = (a["flags"] & 32) != 0
        self.counts["snapshot_records"] += int(snapshot.sum())
        self.counts["F_records"] += int(np.count_nonzero(a["action"] == ord("F")))
        self.counts["bad_capture_flag_records"] += int(np.count_nonzero(a["flags"] & 8))
        for action in b"ACMRTFN":
            self.counts["action_" + chr(action)] += int(
                np.count_nonzero(a["action"] == action)
            )
        terms = np.flatnonzero(((a["flags"] & 128) != 0) & ~snapshot)
        if self.pending and len(terms):
            for minute in self.pending:
                self.completion(minute, a[terms[0]], file, offset + int(terms[0]))
            self.pending.clear()
        selected = np.flatnonzero((a["action"] == ord("T")) & ~snapshot)
        trades = a[selected]
        invalid = (
            (trades["recv"] < self.start)
            | (trades["recv"] >= self.end)
            | (trades["event"] == 0)
            | (trades["event"] >= (1 << 63))
            | ((trades["flags"] & 8) != 0)
        )
        if invalid.any():
            self.holds.add("INVALID_TRADE_TIMESTAMP")
            self.counts["invalid_trade_timestamps"] += int(invalid.sum())
        valid = ~invalid
        selected, trades = selected[valid], trades[valid]
        if (
            (trades["price"] <= 0)
            | (trades["price"] == (1 << 63) - 1)
            | (trades["price"] % 250_000_000 != 0)
        ).any():
            raise ValueError("invalid/off-tick trade price")
        self.counts["zero_size_T_records"] += int(np.count_nonzero(trades["size"] == 0))
        self.counts["included_T_records"] += len(trades)
        self.counts["cross_clock_minute_T_records"] += int(
            np.count_nonzero(trades["recv"] // MINUTE != trades["event"] // MINUTE)
        )
        self.aggregate(trades, selected + offset, file, "recv", self.bars)
        self.aggregate(trades, selected + offset, file, "event", self.exchange)
        if len(selected):
            positions = np.searchsorted(terms, selected)
            used = np.unique(positions[positions < len(terms)])
            bad_terms = set()
            for pos in used:
                row = a[terms[pos]]
                if (
                    int(row["recv"]) == 0
                    or int(row["recv"]) >= (1 << 63)
                    or int(row["flags"]) & 8
                ):
                    bad_terms.add(int(pos))
            if bad_terms:
                self.holds.add("INVALID_TRADE_EVENT_TERMINATOR_TIME")
                for trade, pos in zip(trades, positions):
                    if int(pos) in bad_terms:
                        self.bars[int(trade["recv"] // MINUTE) * MINUTE][
                            "incomplete_event"
                        ] = True
            minutes = trades["recv"] // MINUTE * MINUTE
            for minute in np.unique(minutes):
                ps = positions[minutes == minute]
                completed = ps[ps < len(terms)]
                if len(completed):
                    ix = int(terms[completed[-1]])
                    self.completion(int(minute), a[ix], file, offset + ix)
                if np.any(ps == len(terms)):
                    self.pending.append(int(minute))

    def read_mbo(self, path, name, chunk_records=65536):
        before = self.counts["records"]
        with (
            path.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as stream,
        ):
            metadata(stream, "mbo", path.name.split("-")[2].split(".")[0])
            offset = 0
            for a in chunks(stream, chunk_records):
                self.consume(a, name, offset)
                offset += len(a)
        # Daily snapshots break continuity: never borrow the next day's terminator.
        self.close_file()
        self.files.append({"file": name, "records": self.counts["records"] - before})

    def close_file(self):
        if self.pending:
            self.holds.add("MISSING_TRADE_EVENT_TERMINATOR")
            for minute in self.pending:
                self.bars[minute]["incomplete_event"] = True
            self.pending.clear()

    def finish(self):
        self.close_file()
        for table in (self.bars, self.exchange):
            for bar in table.values():
                if "digest" in bar:
                    bar["trade_sha256"] = bar.pop("digest").hexdigest()
        ordered = [self.bars[m] for m in sorted(self.bars)]
        delayed = []
        for bar, nxt in zip(ordered, ordered[1:]):
            if bar["availability_ns"] > nxt["start_ns"]:
                delayed.append(
                    {
                        "start_ns": bar["start_ns"],
                        "availability_ns": bar["availability_ns"],
                        "next_start_ns": nxt["start_ns"],
                        "completion_ref": bar["completion_ref"],
                    }
                )
        if not ordered:
            self.holds.add("NO_USABLE_TRADE_BARS")
        return ordered, delayed


def read_auxiliary(path, name, schema):
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source) as stream,
    ):
        raw, start = metadata(stream, schema, path.name.split("-")[2].split(".")[0])
        body = stream.read(1_000_001)
        if len(body) > 1_000_000:
            raise ValueError("auxiliary file exceeds bounded limit")
    pos = 0
    while pos < len(body):
        length = body[pos] * 4
        if length < 16 or pos + length > len(body):
            raise ValueError("truncated auxiliary record")
        pos += length
    records = dbn.DBNDecoder().write_and_decode(raw + body)[1:]
    result = []
    for index, r in enumerate(records):
        expected = dbn.InstrumentDefMsg if schema == "definition" else dbn.StatusMsg
        if (
            not isinstance(r, expected)
            or r.instrument_id != INSTRUMENT
            or r.publisher_id != 1
        ):
            raise ValueError("invalid auxiliary identity/schema")
        if not start <= r.ts_recv < start + 1440 * MINUTE:
            raise ValueError("invalid auxiliary capture time")
        if not 0 < r.ts_event < (1 << 63):
            raise ValueError("invalid auxiliary exchange time")
        row = dict(
            file=name,
            record_index=index,
            ts_recv_ns=r.ts_recv,
            ts_event_ns=r.ts_event,
            initial_state=r.ts_recv == start and r.ts_event < start,
        )
        if schema == "definition":
            if (
                r.raw_symbol != "MNQM5"
                or r.min_price_increment != 250_000_000
                or r.unit_of_measure_qty != 2_000_000_000
                or r.currency != "USD"
            ):
                raise ValueError("definition contract mismatch")
            row.update(
                symbol=r.raw_symbol,
                tick_nanos=r.min_price_increment,
                point_value_nanos=r.unit_of_measure_qty,
            )
        else:
            row.update(
                action=int(r.action),
                reason=int(r.reason),
                trading_event=int(r.trading_event),
                is_trading=r.is_trading,
                is_quoting=r.is_quoting,
            )
        result.append(row)
    if not result:
        raise ValueError("empty auxiliary evidence")
    return result


def coverage(builder, statuses):
    states = sorted(
        statuses, key=lambda s: (s["ts_recv_ns"], s["file"], s["record_index"])
    )
    cursor, state = 0, None
    source_days = {
        item["file"].split("/")[-1].split("-")[2].split(".")[0]: item["file"]
        for item in builder.files
    }
    rows = []
    for minute in range(builder.start, builder.end, MINUTE):
        transitions = []
        while cursor < len(states) and states[cursor]["ts_recv_ns"] <= minute:
            state = states[cursor]
            if state["ts_recv_ns"] == minute:
                transitions.append(state)
            cursor += 1
        start_state = state
        position = minute
        interval_states = set()
        while cursor < len(states) and states[cursor]["ts_recv_ns"] < minute + MINUTE:
            transition = states[cursor]
            if transition["ts_recv_ns"] > position:
                interval_states.add(None if state is None else state["is_trading"])
            position = transition["ts_recv_ns"]
            state = transition
            transitions.append(state)
            cursor += 1
        interval_states.add(None if state is None else state["is_trading"])
        interval_status = (
            "UNKNOWN"
            if None in interval_states
            else (
                "MIXED"
                if len(interval_states) > 1
                else "TRADING" if True in interval_states else "NONTRADING"
            )
        )
        bar = builder.bars.get(minute)
        label = (
            "TRADED"
            if bar
            else {
                "UNKNOWN": "NO_TRADE_STATUS_UNKNOWN",
                "MIXED": "NO_TRADE_MIXED_STATUS",
                "TRADING": "NO_TRADE_OBSERVED_TRADING",
                "NONTRADING": "NO_TRADE_OBSERVED_NONTRADING",
            }[interval_status]
        )
        day = datetime.fromtimestamp(minute // 1_000_000_000, timezone.utc).strftime(
            "%Y%m%d"
        )
        rows.append(
            dict(
                start_ns=minute,
                end_ns=minute + MINUTE,
                mbo_source_file=source_days.get(day),
                mbo_source_file_present=day in source_days,
                ohlcv=None if bar is None else bar["ohlcv"],
                trade_count=0 if bar is None else bar["trade_count"],
                coverage=label,
                status_at_start=start_state,
                interval_status=interval_status,
                status_at_end=state,
                status_transitions=transitions,
            )
        )
    return rows
