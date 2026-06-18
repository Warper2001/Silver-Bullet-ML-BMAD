"""capture_recon.py — offline capture leak-detector (pure core).

Reconciles a bot's THEORETICAL fills (backtest replay) against its REALIZED
live fills, attributes the per-trade P&L gap to leak buckets, and ranks each
bucket by its contribution to **MaxGiveback from the high-water mark** — the
path-dependent quantity a Topstep trailing-drawdown ratchet actually punishes.

Design objective (see _bmad-output plan / party-mode roundtable):
    The headline metric is NOT average per-trade slippage (a mean, which the
    ratchet ignores). It is counterfactual ΔMaxGiveback: how much shallower the
    worst peak-to-trough valley would have been had a given leak source been
    zeroed. A leak that is small on average but concentrated inside a drawdown
    is what feeds the ratchet.

PURITY CONTRACT: this module imports only the stdlib. No engine imports, no
file/logging side effects. All bot-specific loading (which DOES pull in
side-effectful engines) lives in ``capture_loaders.py``. Keeping this core pure
means the AC test-suite loads instantly and never touches ``logs/``.

Conservation law (the spine): for every matched pair,
    sum(bucket_components) + unattributed == realized_pnl - theoretical_pnl
to the cent. SLIPPAGE + PARTIAL_FILL decompose the price/qty delta exactly
(the algebra cancels — verified), so a nonzero UNATTRIBUTED means a loader fed
P&L inconsistent with its own prices, and the report self-marks UNTRUSTED.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

# Cent tolerance for conservation / trust gates.
CENT = 0.01

# MNQ point value (USD per point per contract); matches strategy_core.POINT_VALUE_USD.
DEFAULT_POINT_VALUE = 2.0


class Bucket(str, Enum):
    """Leak attribution buckets. Every dollar of (realized - theoretical) P&L
    lands in exactly one (plus the UNATTRIBUTED residual)."""

    SLIPPAGE = "SLIPPAGE"
    LATENCY = "LATENCY"
    PARTIAL_FILL = "PARTIAL_FILL"
    MISSED = "MISSED"
    LABELING_OFFSET = "LABELING_OFFSET"
    UNATTRIBUTED = "UNATTRIBUTED"


@dataclass(frozen=True)
class Trade:
    """Normalized trade record — both THEORETICAL and REALIZED loaders emit this.

    ``signal_bar_ts`` is the entry/signal bar timestamp, tz-aware UTC, floored
    to the minute. It is the *basis* of the join key — NOT the wall-clock fill
    time (that is the artifact source AC4 guards against).
    """

    signal_bar_ts: datetime
    side: int  # +1 long / -1 short
    entry_px: float
    exit_px: float
    qty: int
    pnl_usd: float
    exit_reason: str
    order_id: str | None = None
    regime: str | None = None  # optional vol-regime label attached by loaders


@dataclass(frozen=True)
class ReconConfig:
    bot_id: str
    point_value_usd: float = DEFAULT_POINT_VALUE
    cost_usd_per_contract: float = 0.0  # round-trip cost already baked into pnl_usd if 0
    label_offset_min: int = 0  # the ProjectX/TS +1-min convention; 0 == aligned
    bar_seconds: int = 60  # 60s poll => latency is integer bars
    latency_window_bars: int = 2  # max bar shift the matcher will pair across
    labeling_offset_frac: float = 0.9  # >= this fraction sharing one shift => systematic
    top_n_fat_tail_days: int = 3  # MIM-NB edge ~3 fat-tail days of 163
    thin_n: int = 20  # below this matched-N, print a "statistically thin" banner
    confound_note: str | None = None  # e.g. YANK bar-source confound


# --------------------------------------------------------------------------- #
# Join key
# --------------------------------------------------------------------------- #
def _floor_minute(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(second=0, microsecond=0)


def signal_key(ts: datetime, side: int, *, label_offset_min: int = 0) -> tuple[datetime, int]:
    """Deterministic theoretical<->realized join key.

    Key = (entry bar floored to the UTC minute, side). ``label_offset_min`` is
    the SINGLE place the ProjectX/TS labeling convention is reconciled
    (projectx_bars.PX_LABEL_OFFSET_MIN == 1; ProjectX labels by open-time, TS by
    close-time). The offset shifts the *theoretical* timestamp before keying so
    a correctly-aligned feed lands on the same key as realized.
    """
    return (_floor_minute(ts) + timedelta(minutes=label_offset_min), side)


def _bar_index(ts: datetime, cfg: ReconConfig) -> int:
    """Integer bar index (epoch minutes / bar) — for bar-shift arithmetic."""
    epoch = _floor_minute(ts)
    return int(epoch.timestamp()) // cfg.bar_seconds


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
@dataclass
class MatchedPair:
    theo: Trade | None
    real: Trade | None
    bar_shift: int = 0  # real_bar - theo_bar (after offset), 0 when exact
    dd_state: str | None = None  # 'below_hwm' / 'at_hwm', filled by giveback pass

    @property
    def key_ts(self) -> datetime:
        t = self.real or self.theo
        assert t is not None
        return _floor_minute(t.signal_bar_ts)

    @property
    def regime(self) -> str:
        t = self.real or self.theo
        assert t is not None
        return t.regime or "unknown"


def _dedup(trades: list[Trade]) -> tuple[list[Trade], int]:
    """Collapse duplicate trades. Key on order_id when present, else
    (floored signal_bar_ts, side). Returns (deduped, n_dropped)."""
    seen: set = set()
    out: list[Trade] = []
    dropped = 0
    for t in trades:
        k = ("oid", t.order_id) if t.order_id else ("ts", _floor_minute(t.signal_bar_ts), t.side)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        out.append(t)
    return out, dropped


@dataclass
class MatchResult:
    pairs: list[MatchedPair]
    n_theo: int
    n_real: int
    dropped_dupes: int
    collisions: int  # many-to-one join attempts (poisons trust)


def match_trades(theo: list[Trade], real: list[Trade], cfg: ReconConfig) -> MatchResult:
    """Outer join theoretical<->realized.

    Pass 1: exact key match (offset applied to theoretical). Pass 2: among the
    still-unmatched, pair within +/- ``latency_window_bars`` (same side, nearest
    bar) so a shifted fill surfaces as LATENCY rather than two MISSED rows.
    Leftovers become MISSED (theoretical-only or realized-only).
    """
    theo, d1 = _dedup(theo)
    real, d2 = _dedup(real)

    collisions = 0
    pairs: list[MatchedPair] = []

    # index realized by exact key
    real_by_key: dict[tuple[datetime, int], list[Trade]] = {}
    for r in real:
        real_by_key.setdefault((_floor_minute(r.signal_bar_ts), r.side), []).append(r)

    used_real: set[int] = set()
    unmatched_theo: list[Trade] = []

    # Pass 1 — exact
    for t in theo:
        k = signal_key(t.signal_bar_ts, t.side, label_offset_min=cfg.label_offset_min)
        bucket = real_by_key.get(k)
        cand = None
        if bucket:
            for r in bucket:
                if id(r) not in used_real:
                    cand = r
                    break
            if cand is not None and len([r for r in bucket if id(r) not in used_real]) > 1:
                collisions += 1  # multiple realized for one key — ambiguous
        if cand is not None:
            used_real.add(id(cand))
            pairs.append(MatchedPair(theo=t, real=cand, bar_shift=0))
        else:
            unmatched_theo.append(t)

    # Pass 2 — windowed (latency / labeling shift)
    remaining_real = [r for r in real if id(r) not in used_real]
    real_by_side: dict[int, list[Trade]] = {}
    for r in remaining_real:
        real_by_side.setdefault(r.side, []).append(r)

    still_theo: list[Trade] = []
    for t in unmatched_theo:
        # offset is in minutes; convert to bars (1 bar == bar_seconds)
        theo_bar = _bar_index(t.signal_bar_ts, cfg) + (cfg.label_offset_min * 60) // cfg.bar_seconds
        best = None
        best_shift = None
        for r in real_by_side.get(t.side, []):
            if id(r) in used_real:
                continue
            shift = _bar_index(r.signal_bar_ts, cfg) - theo_bar
            if abs(shift) >= 1 and abs(shift) <= cfg.latency_window_bars:
                if best is None or abs(shift) < abs(best_shift):
                    best, best_shift = r, shift
        if best is not None:
            used_real.add(id(best))
            pairs.append(MatchedPair(theo=t, real=best, bar_shift=best_shift))
        else:
            still_theo.append(t)

    # Leftovers => MISSED
    for t in still_theo:
        pairs.append(MatchedPair(theo=t, real=None))
    for r in real:
        if id(r) not in used_real:
            pairs.append(MatchedPair(theo=None, real=r))

    pairs.sort(key=lambda p: (p.key_ts, p.regime))
    return MatchResult(pairs=pairs, n_theo=len(theo), n_real=len(real),
                       dropped_dupes=d1 + d2, collisions=collisions)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def _price_components(theo: Trade, real: Trade, cfg: ReconConfig) -> tuple[float, float]:
    """Return (slippage_usd, partial_fill_usd) decomposing the price/qty delta.

    Verified to satisfy: slippage + partial == real.pnl_priced - theo.pnl_priced
    where pnl_priced = side*(exit-entry)*pt*qty - cost*qty. The two terms cancel
    algebraically so conservation is exact.
    """
    pt = cfg.point_value_usd
    cost = cfg.cost_usd_per_contract
    qd = real.qty - theo.qty
    side = theo.side
    theo_unit = side * (theo.exit_px - theo.entry_px) * pt - cost
    partial = theo_unit * qd
    slippage = side * ((theo.entry_px - real.entry_px) + (real.exit_px - theo.exit_px)) * pt * real.qty
    return slippage, partial


@dataclass
class Attribution:
    components: dict  # id(pair) -> {Bucket: usd}
    bucket_totals: dict
    total_real_minus_theo: float
    unattributed: float
    trusted: bool
    untrusted_reasons: list = field(default_factory=list)


def _systematic_shift(pairs: list[MatchedPair], cfg: ReconConfig) -> int | None:
    """Return the bar-shift value that is shared by >= labeling_offset_frac of
    the both-present pairs (a systematic labeling-offset bug), else None.
    Distinguishes LABELING_OFFSET (all trades shift the same way) from LATENCY
    (sporadic single-trade shifts)."""
    both = [p for p in pairs if p.theo is not None and p.real is not None]
    # Require a few matched pairs before declaring a SYSTEMATIC offset; with only
    # one or two pairs a single shifted fill is more honestly sporadic LATENCY.
    if len(both) < 3:
        return None
    shifts = Counter(p.bar_shift for p in both)
    for shift, n in shifts.items():
        if shift != 0 and n >= cfg.labeling_offset_frac * len(both):
            return shift
    return None


def attribute(match: MatchResult, cfg: ReconConfig) -> Attribution:
    pairs = match.pairs
    sys_shift = _systematic_shift(pairs, cfg)

    components: dict = {}
    totals = {b: 0.0 for b in Bucket}
    total_delta = 0.0

    for p in pairs:
        comp = {b: 0.0 for b in Bucket}
        if p.theo is not None and p.real is None:
            # theoretical took a trade we never realized
            comp[Bucket.MISSED] = -p.theo.pnl_usd
            total_delta += -p.theo.pnl_usd
        elif p.real is not None and p.theo is None:
            # we realized a trade theoretical never took
            comp[Bucket.MISSED] = p.real.pnl_usd
            total_delta += p.real.pnl_usd
        else:
            delta = p.real.pnl_usd - p.theo.pnl_usd
            total_delta += delta
            if p.bar_shift != 0 and sys_shift is not None and p.bar_shift == sys_shift:
                # systematic shift => the whole delta is the labeling-offset bug
                comp[Bucket.LABELING_OFFSET] = delta
            elif p.bar_shift != 0:
                # sporadic shift => latency; reference bar moved, whole delta to LATENCY
                comp[Bucket.LATENCY] = delta
            else:
                slip, partial = _price_components(p.theo, p.real, cfg)
                comp[Bucket.SLIPPAGE] = slip
                comp[Bucket.PARTIAL_FILL] = partial
                resid = delta - slip - partial
                comp[Bucket.UNATTRIBUTED] = resid
        components[id(p)] = comp
        for b, v in comp.items():
            totals[b] += v

    unattr = totals[Bucket.UNATTRIBUTED]
    reasons: list[str] = []
    if abs(unattr) > CENT:
        reasons.append(f"unattributed ${unattr:+.2f} exceeds 1c (loader P&L inconsistent with prices)")
    if match.collisions:
        reasons.append(f"{match.collisions} many-to-one join collision(s)")
    # conservation invariant
    summed = sum(totals[b] for b in Bucket)
    if abs(summed - total_delta) > CENT:
        reasons.append(f"conservation broken: sum(buckets)=${summed:+.2f} != delta=${total_delta:+.2f}")

    return Attribution(components=components, bucket_totals=totals,
                       total_real_minus_theo=total_delta, unattributed=unattr,
                       trusted=not reasons, untrusted_reasons=reasons)


# --------------------------------------------------------------------------- #
# Giveback (the headline path-dependent metric)
# --------------------------------------------------------------------------- #
def max_giveback(equity_curve: list[float]) -> float:
    """Maximum running-peak-minus-equity over the path (>= 0). Mirrors the
    calc_max_drawdown running-peak pattern (strategy_core.py:911)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    mg = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        gb = peak - v
        if gb > mg:
            mg = gb
    return mg


def _equity_events(pairs: list[MatchedPair]) -> list[tuple[datetime, float, MatchedPair]]:
    """Realized-trade P&L events in chronological (entry) order, for the curve."""
    ev = []
    for p in pairs:
        if p.real is not None:
            ev.append((_floor_minute(p.real.signal_bar_ts), p.real.pnl_usd, p))
    ev.sort(key=lambda e: e[0])
    return ev


@dataclass
class GivebackAttribution:
    baseline_giveback: float
    deepest_valley_ts: datetime | None
    deepest_valley_trades: list[Trade]
    delta_by_bucket: dict  # Bucket -> baseline - counterfactual giveback


def _curve_giveback(events: list[tuple[datetime, float]]) -> tuple[float, datetime | None, int, int]:
    """Cumsum events (sorted), return (max_giveback, valley_ts, peak_idx, trough_idx)."""
    if len(events) < 2:
        return 0.0, None, 0, 0
    cum = 0.0
    eq = []
    for _, pnl in events:
        cum += pnl
        eq.append(cum)
    peak = eq[0]
    peak_idx = 0
    mg = 0.0
    valley_ts = None
    best_peak_idx = 0
    best_trough_idx = 0
    for i, v in enumerate(eq):
        if v > peak:
            peak = v
            peak_idx = i
        gb = peak - v
        if gb > mg:
            mg = gb
            valley_ts = events[i][0]
            best_peak_idx = peak_idx
            best_trough_idx = i
    return mg, valley_ts, best_peak_idx, best_trough_idx


def delta_max_giveback(pairs: list[MatchedPair], attr: Attribution, cfg: ReconConfig) -> GivebackAttribution:
    """Counterfactual ΔMaxGiveback per bucket: rebuild the realized equity curve
    with each bucket's per-trade leak zeroed and recompute MaxGiveback. Positive
    Δ => removing that leak would have shallowed the worst valley. Ranks leaks by
    PATH damage, not mean P&L."""
    base_events = _equity_events(pairs)
    base_simple = [(ts, pnl) for ts, pnl, _ in base_events]
    baseline, valley_ts, p_idx, t_idx = _curve_giveback(base_simple)

    valley_trades: list[Trade] = []
    if valley_ts is not None and base_events:
        for ts, _, p in base_events[p_idx:t_idx + 1]:
            if p.real is not None:
                valley_trades.append(p.real)

    # tag dd_state on each pair (below running HWM at entry?)
    cum = 0.0
    peak = None
    for ts, pnl, p in base_events:
        if peak is None:
            peak = cum
        p.dd_state = "below_hwm" if cum < peak else "at_hwm"
        cum += pnl
        if cum > peak:
            peak = cum

    delta_by_bucket: dict = {}
    for bucket in (Bucket.SLIPPAGE, Bucket.LATENCY, Bucket.PARTIAL_FILL,
                   Bucket.LABELING_OFFSET, Bucket.MISSED):
        cf: list[tuple[datetime, float]] = []
        for ts, pnl, p in base_events:
            comp = attr.components[id(p)].get(bucket, 0.0)
            cf.append((ts, pnl - comp))  # remove the leak's effect on this realized trade
        if bucket == Bucket.MISSED:
            # also add back theoretical-only trades we missed entirely
            for p in pairs:
                if p.theo is not None and p.real is None:
                    cf.append((_floor_minute(p.theo.signal_bar_ts), p.theo.pnl_usd))
                if p.real is not None and p.theo is None:
                    # remove a realized trade theoretical never took
                    cf = [(ts, v) for (ts, v) in cf
                          if not (ts == _floor_minute(p.real.signal_bar_ts) and abs(v - p.real.pnl_usd) < CENT)]
        cf.sort(key=lambda e: e[0])
        cf_gb, *_ = _curve_giveback(cf)
        delta_by_bucket[bucket] = baseline - cf_gb

    return GivebackAttribution(baseline_giveback=baseline, deepest_valley_ts=valley_ts,
                               deepest_valley_trades=valley_trades, delta_by_bucket=delta_by_bucket)


# --------------------------------------------------------------------------- #
# Slicing (regime x dd-state) and fat-tail isolation
# --------------------------------------------------------------------------- #
def slice_giveback(pairs: list[MatchedPair], attr: Attribution, cfg: ReconConfig) -> dict:
    """Net leak ($) per (regime, dd_state) cell, per bucket. dd_state must have
    been filled by delta_max_giveback first."""
    cells: dict = {}
    for p in pairs:
        regime = p.regime
        dd = p.dd_state or "unknown"
        cell = cells.setdefault((regime, dd), {b: 0.0 for b in Bucket})
        for b, v in attr.components[id(p)].items():
            cell[b] += v
    return cells


def fat_tail_days(pairs: list[MatchedPair], cfg: ReconConfig) -> dict:
    """Isolate the top-N largest favorable-realized-P&L days and report
    realized vs theoretical capture on those days specifically (MIM-NB's edge is
    ~3 fat-tail days of 163, so a leak concentrated there is the whole story)."""
    by_day_real: dict[date, float] = {}
    by_day_theo: dict[date, float] = {}
    for p in pairs:
        if p.real is not None:
            d = _floor_minute(p.real.signal_bar_ts).date()
            by_day_real[d] = by_day_real.get(d, 0.0) + p.real.pnl_usd
        if p.theo is not None:
            d = _floor_minute(p.theo.signal_bar_ts).date()
            by_day_theo[d] = by_day_theo.get(d, 0.0) + p.theo.pnl_usd
    top = sorted(by_day_real, key=lambda d: by_day_real[d], reverse=True)[:cfg.top_n_fat_tail_days]
    out = {}
    for d in top:
        real = by_day_real.get(d, 0.0)
        theo = by_day_theo.get(d, 0.0)
        out[d] = {"realized": real, "theoretical": theo,
                  "capture_pct": (real / theo * 100.0) if theo else float("nan")}
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class ReconReport:
    cfg: ReconConfig
    match: MatchResult
    attribution: Attribution
    giveback: GivebackAttribution
    sliced: dict
    fat_tail: dict

    @property
    def n_matched(self) -> int:
        return sum(1 for p in self.match.pairs if p.theo is not None and p.real is not None)

    @property
    def trusted(self) -> bool:
        return self.attribution.trusted


def run_recon(theo: list[Trade], real: list[Trade], cfg: ReconConfig) -> ReconReport:
    match = match_trades(theo, real, cfg)
    attr = attribute(match, cfg)
    gb = delta_max_giveback(match.pairs, attr, cfg)
    sliced = slice_giveback(match.pairs, attr, cfg)
    ft = fat_tail_days(match.pairs, cfg)
    return ReconReport(cfg=cfg, match=match, attribution=attr, giveback=gb, sliced=sliced, fat_tail=ft)


def render_markdown(report: ReconReport) -> str:
    cfg = report.cfg
    a = report.attribution
    gb = report.giveback
    lines: list[str] = []
    lines.append(f"# Capture Leak Report — {cfg.bot_id}")
    lines.append("")
    if not report.trusted:
        lines.append("## ⛔ UNTRUSTED")
        for r in a.untrusted_reasons:
            lines.append(f"- {r}")
        lines.append("")
    if cfg.confound_note:
        lines.append(f"> ⚠️ **CONFOUNDED:** {cfg.confound_note}")
        lines.append("")
    n = report.n_matched
    lines.append(f"- matched pairs: **{n}**  (theo {report.match.n_theo} / real {report.match.n_real}, "
                 f"dropped dupes {report.match.dropped_dupes})")
    if n < cfg.thin_n:
        lines.append(f"> 📉 **Statistically thin (N={n} < {cfg.thin_n})** — read ΔMaxGiveback as directional only.")
    lines.append(f"- total realized − theoretical P&L: **${a.total_real_minus_theo:+,.2f}**")
    lines.append(f"- unattributed residual: ${a.unattributed:+,.2f}")
    lines.append("")

    lines.append("## Leak by bucket (P&L impact)")
    lines.append("| bucket | P&L $ | ΔMaxGiveback $ |")
    lines.append("|---|---:|---:|")
    for b in (Bucket.SLIPPAGE, Bucket.LATENCY, Bucket.PARTIAL_FILL,
              Bucket.MISSED, Bucket.LABELING_OFFSET, Bucket.UNATTRIBUTED):
        d = gb.delta_by_bucket.get(b)
        dtxt = f"{d:+,.2f}" if d is not None else "—"
        lines.append(f"| {b.value} | {a.bucket_totals[b]:+,.2f} | {dtxt} |")
    lines.append("")
    lines.append(f"- baseline MaxGiveback: **${gb.baseline_giveback:,.2f}**")
    if gb.deepest_valley_ts is not None:
        vts = gb.deepest_valley_ts.astimezone(timezone.utc)
        lines.append(f"- deepest valley at (UTC): {vts:%Y-%m-%d %H:%M}, "
                     f"{len(gb.deepest_valley_trades)} trade(s) in the descent")
    lines.append("")

    lines.append("## ΔMaxGiveback drivers by regime × drawdown-state")
    lines.append("| regime | dd_state | SLIPPAGE | LATENCY | MISSED | PARTIAL | LABELING |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (regime, dd), cell in sorted(report.sliced.items()):
        lines.append(f"| {regime} | {dd} | {cell[Bucket.SLIPPAGE]:+,.0f} | {cell[Bucket.LATENCY]:+,.0f} "
                     f"| {cell[Bucket.MISSED]:+,.0f} | {cell[Bucket.PARTIAL_FILL]:+,.0f} "
                     f"| {cell[Bucket.LABELING_OFFSET]:+,.0f} |")
    lines.append("")

    if report.fat_tail:
        lines.append(f"## Fat-tail days (top {cfg.top_n_fat_tail_days} realized) — capture")
        lines.append("| day | realized $ | theoretical $ | capture % |")
        lines.append("|---|---:|---:|---:|")
        for d, row in report.fat_tail.items():
            cap = row["capture_pct"]
            captxt = f"{cap:.1f}" if cap == cap else "n/a"  # nan check
            lines.append(f"| {d} | {row['realized']:+,.2f} | {row['theoretical']:+,.2f} | {captxt} |")
        lines.append("")

    return "\n".join(lines)
