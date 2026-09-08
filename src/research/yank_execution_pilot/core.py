"""Integer-price book and completed-event evidence primitives."""
from sortedcontainers import SortedDict
from dataclasses import dataclass
from datetime import datetime, timezone

NS = 1_000_000_000
MINUTE = 60 * NS
DAY = 1440 * MINUTE
SCALE = NS
UNDEF = 2**63 - 1


def ns(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('naive timestamp')
    return int(d.timestamp()) * NS + d.microsecond * 1000


def iso(value):
    seconds, nanos = divmod(int(value), NS)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


@dataclass(frozen=True, slots=True)
class Record:
    index: int
    ts: int
    exchange: int
    action: str
    side: str
    order_id: int
    price: int
    size: int
    flags: int
    sequence: int

    def ref(self, source, event):
        return dict(source=source, record_index=self.index, event_id=event,
                    ts_recv=iso(self.ts), ts_recv_ns=self.ts,
                    ts_event=iso(self.exchange), ts_event_ns=self.exchange,
                    sequence=self.sequence, action=self.action, side=self.side,
                    price=self.price / SCALE, size=self.size, flags=self.flags)


class Book:
    def __init__(self):
        self.orders = {}
        self.levels = {'B': SortedDict(), 'A': SortedDict()}
        self.initialized = False
        self.valid = False
        self.snapshot = False

    def _remove(self, order):
        side, price, size = order
        self.levels[side][price] -= size
        if self.levels[side][price] == 0:
            del self.levels[side][price]

    def apply(self, r):
        """Return gap reason, never infer an order missing from source evidence."""
        snap = bool(r.flags & 32)
        if r.flags & ~(128 | 32 | 8):
            self.valid = False
            return 'unsupported_flags'
        if r.flags & 8 and not snap:
            self.valid = False
            return 'bad_live_capture_timestamp'
        if snap and r.action not in ('R', 'A'):
            self.valid = False
            return 'unsupported_snapshot_action'
        if r.action == 'R':
            self.orders.clear()
            self.levels = {'B': SortedDict(), 'A': SortedDict()}
            self.initialized = True
            self.valid = True
            self.snapshot = snap
            return None
        if not self.initialized:
            return 'missing_initial_snapshot'
        if snap and not self.snapshot:
            self.valid = False
            return 'snapshot_without_reset'
        if r.action in ('T', 'F', 'N'):
            return None
        if r.action not in ('A', 'M', 'C'):
            self.valid = False
            return 'unsupported_action'
        old = self.orders.get(r.order_id)
        error = None
        if r.side not in ('A', 'B') or not 0 < r.price < UNDEF or r.size <= 0:
            error = 'invalid_order_fields'
        elif r.action == 'A' and old is not None:
            error = 'duplicate_add'
        elif r.action in ('M', 'C') and old is None:
            error = 'unknown_order_update'
        elif old and (old[0] != r.side or (r.action == 'C' and (r.size > old[2] or r.price != old[1]))):
            error = 'malformed_update'
        if error:
            self.valid = False
            return error
        if old:
            self._remove(old)
        size = old[2] - r.size if r.action == 'C' else r.size
        if size:
            self.orders[r.order_id] = (r.side, r.price, size)
            self.levels[r.side][r.price] = self.levels[r.side].get(r.price, 0) + size
        else:
            del self.orders[r.order_id]
        return None

    def complete(self):
        self.snapshot = False
        bid = self.levels['B'].peekitem(-1)[0] if self.levels['B'] else None
        ask = self.levels['A'].peekitem(0)[0] if self.levels['A'] else None
        if bid is None or ask is None:
            return 'one_sided_or_empty_book'
        if bid >= ask:
            return 'locked_or_crossed_book'
        return None if self.valid else 'invalid_book'

    def quote(self, limit, quantity):
        bid = self.levels['B'].peekitem(-1)[0] if self.levels['B'] else None
        ask = self.levels['A'].peekitem(0)[0] if self.levels['A'] else None
        available = sum(s for p, s in self.levels['B'].items() if p >= limit)
        return dict(bid=bid / SCALE if bid else None, ask=ask / SCALE if ask else None,
                    spread=(ask-bid) / SCALE if bid and ask else None,
                    bid_size=self.levels['B'].get(bid, 0), ask_size=self.levels['A'].get(ask, 0),
                    displayed_ask_size_at_limit=self.levels['A'].get(limit, 0),
                    displayed_bid_size_at_or_above_limit=available,
                    marketable=bid is not None and bid >= limit,
                    insufficient_displayed_size=available < abs(quantity), displayed_size_scope='immediately_executable_bids_for_short_limit; same-price asks are queue context only')


def opportunities(labels, signal, convention, end):
    future = [t for t in labels if t > signal][:240]
    if not future:
        raise ValueError('no scheduled pending opportunities')
    offset = MINUTE if convention == 'start' else 0
    return dict(signal_completion=signal + offset,
                first_opportunity=future[0], last_opportunity=future[-1],
                scheduled_count=len(future),
                expiry=min(future[-1] + offset, end),
                clipped=len(future) < 240 or future[-1] + offset > end)


def outcome(through, touch, gaps, covered):
    if gaps or not covered:
        return 'unassessable'
    if through:
        return 'supported'
    return 'touch-only' if touch else 'unsupported'


def ordering(entry, barrier):
    if entry is None or barrier is None:
        return 'unassessable_missing_crossing'
    if entry.get('completed_event_valid') is False or barrier.get('completed_event_valid') is False:
        return 'unassessable_invalid_completed_event'
    if entry['event_id'] == barrier['event_id'] or entry['ts_recv_ns'] == barrier['ts_recv_ns']:
        return 'ambiguous_same_event_or_equal_capture_time'
    if entry['ts_recv_ns'] > barrier['ts_recv_ns']:
        return 'barrier_before_possible_entry'
    return 'entry_evidence_before_barrier_not_fill_proof'
