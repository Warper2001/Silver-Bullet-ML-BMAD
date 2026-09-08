"""Offline displayed-book observations for the pinned May 28 research timeline."""

import importlib.util
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("timeline", HERE / "inspect-may28.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
RECORD = struct.Struct("<BBHIQQqIBBBBQiI")


class Book:
    def __init__(self):
        self.orders = {}
        self.ready = False
        self.started = False
        self.snapshot_end = None
        self.last_end = None
        self.last_recv = 0
        self.count = 0

    def apply(self, r, index):
        (
            length,
            rtype,
            publisher,
            instrument,
            event,
            oid,
            price,
            size,
            flags,
            channel,
            action,
            side,
            recv,
            delta,
            seq,
        ) = r
        t.require(length == 14 and rtype == 160 and instrument == 42009475, "identity")
        t.require(not flags & 64, "unsupported top-of-book record")
        if flags & 32:
            t.require(not self.ready, "unexpected later snapshot")
            if not self.started:
                t.require(
                    index == 0 and action == ord("R"), "missing initial snapshot clear"
                )
                self.started = True
            else:
                t.require(action == ord("A"), "invalid snapshot action")
        else:
            t.require(self.ready, "incomplete initial snapshot")
            t.require(
                not flags & 8 and 1748390400000000000 <= recv < 1748476800000000000,
                "invalid observed capture timestamp",
            )
            t.require(
                0 < event <= recv and recv >= self.last_recv,
                "invalid exchange time or capture regression",
            )
            self.last_recv = recv
        if action == ord("R"):
            self.orders.clear()
        elif action in (ord("A"), ord("M"), ord("C")):
            t.require(
                side in (ord("A"), ord("B")) and 0 < price < 2**63 - 1 and size > 0,
                "invalid book value",
            )
            if action == ord("A"):
                t.require(oid not in self.orders, "duplicate add")
                self.orders[oid] = (side, price, size)
            else:
                t.require(oid in self.orders, "update without order")
                old_side, old_price, old_size = self.orders[oid]
                t.require(side == old_side, "side mismatch")
                if action == ord("M"):
                    self.orders[oid] = (side, price, size)
                else:
                    t.require(
                        price == old_price and size <= old_size, "cancel mismatch"
                    )
                    if size == old_size:
                        del self.orders[oid]
                    else:
                        self.orders[oid] = (side, price, old_size - size)
        else:
            t.require(action in (ord("T"), ord("F"), ord("N")), "unsupported action")
        if flags & 128:
            self.last_end = dict(
                record_index=index,
                ts_recv_ns=recv,
                ts_event_ns=event,
                sequence=seq,
                raw_hex=RECORD.pack(*r).hex(),
            )
            if flags & 32:
                self.ready = True
                self.snapshot_end = index
        self.count += 1

    def observe(self, limit, contracts, next_index):
        if (
            not self.ready
            or self.last_end is None
            or self.last_end["record_index"] != next_index - 1
        ):
            return dict(
                status="UNASSESSABLE_INCOMPLETE_EVENT",
                last_complete_event=self.last_end,
            )
        bids = {}
        asks = {}
        for side, price, size in self.orders.values():
            levels = bids if side == ord("B") else asks
            levels[price] = levels.get(price, 0) + size
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        t.require(
            best_bid is None or best_ask is None or best_bid < best_ask,
            "locked/crossed completed book",
        )
        return dict(
            status="OBSERVED_COMPLETE_EVENT",
            last_complete_event=self.last_end,
            resting_orders=len(self.orders),
            best_bid_nanos=best_bid,
            best_bid_size=bids.get(best_bid, 0),
            best_ask_nanos=best_ask,
            best_ask_size=asks.get(best_ask, 0),
            spread_nanos=(
                None if best_bid is None or best_ask is None else best_ask - best_bid
            ),
            ask_size_at_limit=asks.get(limit, 0),
            bid_size_at_or_above_limit=sum(
                size for price, size in bids.items() if price >= limit
            ),
            short_limit_marketable=best_bid is not None and best_bid >= limit,
            opposing_display_covers_quantity=sum(
                size for price, size in bids.items() if price >= limit
            )
            >= contracts,
        )


def main():
    evidence_path = HERE / "may28-timeline.json"
    t.require(
        t.sha(evidence_path)
        == "7b1f8546feb84d76b81466fe204a2b58b52e917164aecc7eeacbebe871de060d",
        "timeline pin",
    )
    evidence = json.loads(evidence_path.read_text())
    source = t.DATA / evidence["native_file"]
    t.require(t.sha(source) == evidence["native_sha256"], "native pin")
    refs = {r["record_index"]: r for r in evidence["references"]}
    requests = []
    for c in evidence["cases"]:
        requests.append(
            dict(
                kind="arrival",
                signal_time=c["signal_time"],
                arms=c["arms"],
                delay_ms=c["delay_ms"],
                time_ns=c["arrival_ns"],
                limit=c["limit_nanos"],
                contracts=c["contracts"],
            )
        )
        if c["delay_ms"] == 0:
            through = refs[c["hits"]["first_through"]]
            requests.append(
                dict(
                    kind="before_first_through_event",
                    signal_time=c["signal_time"],
                    arms=c["arms"],
                    record_index=through["event_start_record"],
                    limit=c["limit_nanos"],
                    contracts=c["contracts"],
                )
            )
    book = Book()
    observations = []
    pending = list(requests)
    for base, raw, a in t.chunks(source):
        for offset, r in enumerate(RECORD.iter_unpack(raw)):
            index = base + offset
            if pending and not r[8] & 32:
                due = [
                    q
                    for q in pending
                    if (
                        index == q["record_index"]
                        if "record_index" in q
                        else r[12] >= q["time_ns"]
                    )
                ]
                for q in due:
                    observations.append(
                        dict(
                            request=q,
                            book=book.observe(q["limit"], q["contracts"], index),
                            first_unapplied_record=dict(
                                record_index=index,
                                ts_recv_ns=r[12],
                                ts_event_ns=r[4],
                                sequence=r[14],
                                raw_hex=RECORD.pack(*r).hex(),
                            ),
                            equal_capture_time_boundary=q.get("time_ns") == r[12],
                        )
                    )
                    pending.remove(q)
                if not pending:
                    break
            try:
                book.apply(r, index)
            except ValueError as exc:
                raise ValueError(f"record {index}: {exc}") from exc
        if not pending:
            break
    t.require(not pending, "unavailable requested observations")
    # Official decoder verifies the raw records bracketing every observation.
    checked = set()
    for o in observations:
        for ref in (o["book"]["last_complete_event"], o["first_unapplied_record"]):
            if ref is None:
                continue
            v = t.dbn.DBNDecoder(has_metadata=False).write_and_decode(
                bytes.fromhex(ref["raw_hex"])
            )
            t.require(
                len(v) == 1
                and int(v[0].ts_recv) == ref["ts_recv_ns"]
                and int(v[0].ts_event) == ref["ts_event_ns"]
                and int(v[0].sequence) == ref["sequence"],
                "official boundary decode",
            )
            checked.add(ref["record_index"])
    t.require(t.sha(source) == evidence["native_sha256"], "input changed")
    print(
        json.dumps(
            dict(
                research_status="HOLD_VALIDATION",
                scope="Observed displayed book before arrivals and first entry-through events; no simulated matching or PNL revision",
                native_file=evidence["native_file"],
                native_sha256=evidence["native_sha256"],
                timeline_sha256=t.sha(evidence_path),
                script_sha256=t.sha(Path(__file__)),
                reader_sha256=t.sha(HERE / "inspect-may28.py"),
                snapshot_end_record=book.snapshot_end,
                records_applied=book.count,
                official_boundary_records_checked=len(checked),
                observations=observations,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
