"""Unit tests for the pure capture-recon core (no engine imports, no live I/O).

ACs map to the approved plan:
  AC1 zero-slippage replay reconciles to ~0 gap & equal MaxGiveback
  AC2 systematic +1-min shift lands entirely in LABELING_OFFSET, unattributed==0
  AC3 double-count guard (dedup on order_id / (ts,side))
  AC4 sub-bar shift -> attribution unchanged; full +1 bar -> LATENCY (not slippage)
  AC6 conservation: sum(buckets)+unattributed == real-theo to the cent
  extra: valley-concentrated leak has larger ΔMaxGiveback than calm-stretch leak
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.research.capture_recon import (
    Bucket,
    ReconConfig,
    Trade,
    attribute,
    match_trades,
    max_giveback,
    run_recon,
)

PT = 2.0
BASE = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)


def _cfg(**kw) -> ReconConfig:
    return ReconConfig(bot_id="test", point_value_usd=PT, cost_usd_per_contract=0.0, **kw)


def mk(
    minute: int,
    side: int,
    entry: float,
    exit: float,
    qty: int = 1,
    *,
    order_id: str | None = None,
    regime: str | None = None,
    cost: float = 0.0,
) -> Trade:
    """Build a Trade whose pnl_usd is self-consistent with its prices."""
    pnl = side * (exit - entry) * PT * qty - cost * qty
    return Trade(
        signal_bar_ts=BASE + timedelta(minutes=minute),
        side=side,
        entry_px=entry,
        exit_px=exit,
        qty=qty,
        pnl_usd=pnl,
        exit_reason="TEST",
        order_id=order_id,
        regime=regime,
    )


# --------------------------------------------------------------------------- #
# AC1 — zero-slippage replay reconciles to ~0
# --------------------------------------------------------------------------- #
def test_ac1_zero_slippage_reconciles_to_zero():
    trades = [mk(0, 1, 100, 105), mk(10, -1, 110, 108), mk(20, 1, 120, 119)]
    theo = list(trades)
    real = list(trades)
    report = run_recon(theo, real, _cfg())

    assert report.trusted
    assert abs(report.attribution.total_real_minus_theo) < 0.01
    for b in Bucket:
        assert abs(report.attribution.bucket_totals[b]) < 0.01, b
    # equal curves => equal MaxGiveback
    real_curve, theo_curve = [], []
    cr = ct = 0.0
    for t in real:
        cr += t.pnl_usd
        real_curve.append(cr)
    for t in theo:
        ct += t.pnl_usd
        theo_curve.append(ct)
    assert max_giveback(real_curve) == max_giveback(theo_curve)


# --------------------------------------------------------------------------- #
# AC2 — systematic +1-min shift => LABELING_OFFSET, unattributed == 0
# --------------------------------------------------------------------------- #
def test_ac2_systematic_offset_lands_in_labeling_offset():
    theo = [mk(0, 1, 100, 105), mk(10, -1, 110, 108), mk(20, 1, 120, 122), mk(30, -1, 130, 129)]
    # realized are identical fills but every signal_bar_ts shifted +1 minute,
    # and we run with the offset MIS-set (cfg expects 0).
    real = [
        Trade(
            signal_bar_ts=t.signal_bar_ts + timedelta(minutes=1),
            side=t.side,
            entry_px=t.entry_px,
            exit_px=t.exit_px,
            qty=t.qty,
            pnl_usd=t.pnl_usd,
            exit_reason=t.exit_reason,
        )
        for t in theo
    ]
    report = run_recon(theo, real, _cfg(label_offset_min=0))

    totals = report.attribution.bucket_totals
    assert abs(totals[Bucket.UNATTRIBUTED]) < 0.01
    assert abs(totals[Bucket.SLIPPAGE]) < 0.01
    # the entire P&L delta is captured by the labeling-offset canary
    assert abs(totals[Bucket.LABELING_OFFSET] - report.attribution.total_real_minus_theo) < 0.01
    # and the buckets that are NOT systematic should be empty
    assert abs(totals[Bucket.LATENCY]) < 0.01


def test_ac2_correct_offset_aligns_cleanly():
    """With label_offset_min set correctly, the same data reconciles to zero."""
    theo = [mk(0, 1, 100, 105), mk(10, -1, 110, 108)]
    real = [
        Trade(signal_bar_ts=t.signal_bar_ts + timedelta(minutes=1), side=t.side,
              entry_px=t.entry_px, exit_px=t.exit_px, qty=t.qty, pnl_usd=t.pnl_usd,
              exit_reason=t.exit_reason)
        for t in theo
    ]
    report = run_recon(theo, real, _cfg(label_offset_min=1))
    for b in Bucket:
        assert abs(report.attribution.bucket_totals[b]) < 0.01, b


# --------------------------------------------------------------------------- #
# AC3 — double-count guard
# --------------------------------------------------------------------------- #
def test_ac3_dedup_on_order_id():
    t = mk(0, 1, 100, 105, order_id="A")
    dup = mk(0, 1, 100, 105, order_id="A")
    theo = [mk(0, 1, 100, 105, order_id="A")]
    real = [t, dup]
    report = run_recon(theo, real, _cfg())
    assert report.match.dropped_dupes == 1
    assert report.n_matched == 1
    # no doubled P&L
    assert abs(report.attribution.total_real_minus_theo) < 0.01


def test_ac3_dedup_on_ts_side_without_order_id():
    theo = [mk(5, -1, 110, 108)]
    real = [mk(5, -1, 110, 108), mk(5, -1, 110, 108)]  # no order_id => key on (ts,side)
    report = run_recon(theo, real, _cfg())
    assert report.match.dropped_dupes == 1
    assert report.n_matched == 1


# --------------------------------------------------------------------------- #
# AC4 — clock-skew immunity
# --------------------------------------------------------------------------- #
def test_ac4_sub_bar_shift_unchanged():
    theo = [mk(0, 1, 100, 105), mk(10, -1, 110, 108)]
    # realized fills 30s later — same minute floor => same key, no shift
    real = [
        Trade(signal_bar_ts=t.signal_bar_ts + timedelta(seconds=30), side=t.side,
              entry_px=t.entry_px, exit_px=t.exit_px, qty=t.qty, pnl_usd=t.pnl_usd,
              exit_reason=t.exit_reason)
        for t in theo
    ]
    report = run_recon(theo, real, _cfg())
    for b in Bucket:
        assert abs(report.attribution.bucket_totals[b]) < 0.01, b
    assert all(p.bar_shift == 0 for p in report.match.pairs)


def test_ac4_full_bar_shift_surfaces_as_latency():
    # one sporadic trade shifted a full bar (+1 min) among several aligned ones
    theo = [mk(0, 1, 100, 105), mk(10, -1, 110, 108), mk(20, 1, 120, 124), mk(30, -1, 130, 127)]
    real = [mk(0, 1, 100, 105), mk(10, -1, 110, 108), mk(20, 1, 120, 124)]
    # shift the 4th realized by a full bar
    shifted = Trade(signal_bar_ts=theo[3].signal_bar_ts + timedelta(minutes=1), side=-1,
                    entry_px=130, exit_px=127, qty=1, pnl_usd=theo[3].pnl_usd, exit_reason="T")
    real.append(shifted)
    report = run_recon(theo, real, _cfg())
    totals = report.attribution.bucket_totals
    # the shifted trade matched (not MISSED) and landed in LATENCY, not SLIPPAGE
    assert any(p.bar_shift != 0 for p in report.match.pairs)
    assert abs(totals[Bucket.SLIPPAGE]) < 0.01
    assert abs(totals[Bucket.MISSED]) < 0.01  # it matched within the window
    # latency delta is zero here (same prices), but the classification path is exercised;
    # make a price diff to confirm it routes to LATENCY:


def test_ac4_latency_captures_pnl_delta():
    theo = [mk(0, 1, 100, 105), mk(10, -1, 110, 108), mk(20, 1, 120, 124)]
    real = [mk(0, 1, 100, 105), mk(10, -1, 110, 108)]
    # 3rd trade realized one bar late AND at a worse exit (123 vs 124)
    shifted = mk(21, 1, 120, 123)
    real.append(shifted)
    report = run_recon(theo, real, _cfg())
    totals = report.attribution.bucket_totals
    expected_delta = shifted.pnl_usd - theo[2].pnl_usd  # = (123-120 - (124-120))*2 = -2
    assert abs(totals[Bucket.LATENCY] - expected_delta) < 0.01
    assert abs(totals[Bucket.SLIPPAGE]) < 0.01


# --------------------------------------------------------------------------- #
# AC6 — conservation law
# --------------------------------------------------------------------------- #
def test_ac6_conservation_mixed_leaks():
    theo = [mk(0, 1, 100, 105, qty=2), mk(10, -1, 110, 108, qty=2),
            mk(20, 1, 120, 125, qty=2), mk(30, -1, 130, 126, qty=2)]
    real = [
        mk(0, 1, 100.25, 104.75, qty=2),   # slippage both sides
        mk(10, -1, 110, 108, qty=1),       # partial fill
        # 3rd trade missed entirely (theoretical-only)
        mk(40, 1, 140, 138, qty=2),        # realized-only (theoretical never took)
    ]
    report = run_recon(theo, real, _cfg())
    totals = report.attribution.bucket_totals
    summed = sum(totals[b] for b in Bucket)
    assert abs(summed - report.attribution.total_real_minus_theo) < 0.01
    assert abs(totals[Bucket.UNATTRIBUTED]) < 0.01
    # we exercised slippage, partial, and missed (both directions)
    assert abs(totals[Bucket.SLIPPAGE]) > 0.0
    assert abs(totals[Bucket.PARTIAL_FILL]) > 0.0
    assert abs(totals[Bucket.MISSED]) > 0.0


def test_ac6_inconsistent_pnl_trips_trust_gate():
    """A realized trade whose logged pnl disagrees with its prices => UNATTRIBUTED
    nonzero => untrusted."""
    theo = [mk(0, 1, 100, 105)]
    bad = Trade(signal_bar_ts=BASE, side=1, entry_px=100, exit_px=105, qty=1,
                pnl_usd=999.0, exit_reason="BAD")  # pnl inconsistent with prices
    report = run_recon(theo, [bad], _cfg())
    assert not report.trusted
    assert abs(report.attribution.bucket_totals[Bucket.UNATTRIBUTED]) > 0.01


# --------------------------------------------------------------------------- #
# extra — ΔMaxGiveback ranks by PATH damage, not mean P&L
# --------------------------------------------------------------------------- #
def test_giveback_ranks_valley_leak_over_calm_leak():
    """Two equal-magnitude slippage leaks: one inside the peak->trough descent,
    one in a calm/rising stretch. The valley leak must have the larger
    ΔMaxGiveback even though both cost the same mean P&L."""
    # Build a curve: up, up, then a deep valley, then recovery.
    # Trade pnls (theoretical): +100, +100, -300, -300, +400
    def pair(minute, pnl, entry_slip=0.0):
        # represent pnl via a long trade exit; slippage via worse entry
        entry = 100.0
        exit = entry + pnl / PT  # qty 1
        return entry, exit

    theo = []
    pnls = [100, 100, -300, -300, 400]
    for i, pnl in enumerate(pnls):
        e = 100.0
        x = e + pnl / PT
        theo.append(mk(i * 10, 1, e, x))

    # Variant A: slippage on trade index 2 (start of the valley descent)
    realA = list(theo)
    # worse entry by 5 points on the valley trade => -10 usd leak
    t2 = theo[2]
    realA[2] = mk(20, 1, t2.entry_px + 5.0, t2.exit_px)
    repA = run_recon(theo, realA, _cfg())

    # Variant B: same-magnitude slippage on the final rising trade (calm stretch)
    realB = list(theo)
    t4 = theo[4]
    realB[4] = mk(40, 1, t4.entry_px + 5.0, t4.exit_px)
    repB = run_recon(theo, realB, _cfg())

    dA = repA.giveback.delta_by_bucket[Bucket.SLIPPAGE]
    dB = repB.giveback.delta_by_bucket[Bucket.SLIPPAGE]
    # same mean leak
    assert abs(repA.attribution.bucket_totals[Bucket.SLIPPAGE]
               - repB.attribution.bucket_totals[Bucket.SLIPPAGE]) < 0.01
    # but the valley leak deepened MaxGiveback more
    assert dA > dB
