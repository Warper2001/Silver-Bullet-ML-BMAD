"""Regression: YANK's trades.db write must carry ml_proba (party-mode 2026-08-18).

Root cause: ml_proba is computed correctly for every trade (ML Filter ACTIVE,
threshold=0.5) and logged correctly to logs/yank_ml_canary.csv, but the
TradeDatabase().log_trade() call in _close_active_trade never passed the
ml_proba kwarg — 1,845 of 1,846 historical YANK rows in trades.db are NULL
despite the model being live the whole time (s26/s27 populate it on every
row via the same DB helper, so the gap was call-site-specific, not a
TradeDatabase limitation). Fix is additive logging only — no gate, sizing,
or entry logic touched, so no prereg amendment is required.
"""
import inspect

from src.research.yank_streaming_working import Tier2StreamingTrader


def test_trades_db_call_passes_ml_proba():
    src = inspect.getsource(Tier2StreamingTrader._close_active_trade)
    assert "ml_proba=" in src, (
        "the trades.db log_trade() call must pass ml_proba — it silently "
        "defaulted to NULL for 1,845/1,846 YANK rows despite the model being active"
    )


def test_ml_proba_nan_maps_to_none_not_literal_nan():
    """NaN (the ActiveTrade default when no proba was captured) must become
    SQL NULL, not the float nan — sqlite3 would otherwise store nan verbatim
    and downstream `is not null` checks would misreport it as present."""
    src = inspect.getsource(Tier2StreamingTrader._close_active_trade)
    assert "np.isnan(t.ml_proba)" in src
    assert "None if np.isnan(t.ml_proba)" in src
