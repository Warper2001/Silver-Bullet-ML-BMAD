"""Topstep-session compliance boundaries for YANK (Phase A of the joint deployment).

Verifies the 15:08-17:00 CT entry block and the 15:10 CT flatten boundary convert
correctly from UTC across DST. The gate logic in _detect_and_enter / _advance_active_trade
is: ct_min = bar_ct.hour*60+bar_ct.minute; block if BLOCK_LO<=ct_min<BLOCK_HI; flatten if
ct_min>=FLATTEN_MIN.
"""
import datetime as dt
import pytz

from src.research.yank_streaming_working import (
    CT_TZ, TOPSTEP_FLATTEN_MIN, TOPSTEP_BLOCK_LO, TOPSTEP_BLOCK_HI,
)

UTC = pytz.UTC


def ct_min(utc_dt):
    c = utc_dt.astimezone(CT_TZ)
    return c.hour * 60 + c.minute


def blocked(utc_dt):
    return TOPSTEP_BLOCK_LO <= ct_min(utc_dt) < TOPSTEP_BLOCK_HI


def flattens(utc_dt):
    # only the [15:10, 17:00) CT close window — matches _advance_active_trade
    return TOPSTEP_FLATTEN_MIN <= ct_min(utc_dt) < TOPSTEP_BLOCK_HI


def test_constants():
    assert TOPSTEP_BLOCK_LO == 15 * 60 + 8
    assert TOPSTEP_FLATTEN_MIN == 15 * 60 + 10
    assert TOPSTEP_BLOCK_HI == 17 * 60


# Summer = CDT (UTC-5): 15:09 CT = 20:09 UTC. Winter = CST (UTC-6): 15:09 CT = 21:09 UTC.
def test_entry_block_summer_cdt():
    assert not blocked(UTC.localize(dt.datetime(2025, 7, 1, 20, 7)))   # 15:07 CT - allowed
    assert blocked(UTC.localize(dt.datetime(2025, 7, 1, 20, 8)))       # 15:08 CT - blocked
    assert blocked(UTC.localize(dt.datetime(2025, 7, 1, 20, 9)))       # 15:09 CT - blocked
    assert blocked(UTC.localize(dt.datetime(2025, 7, 1, 21, 30)))      # 16:30 CT - blocked
    assert not blocked(UTC.localize(dt.datetime(2025, 7, 1, 22, 0)))   # 17:00 CT - reopened
    assert not blocked(UTC.localize(dt.datetime(2025, 7, 1, 23, 0)))   # 18:00 CT - evening ok


def test_entry_block_winter_cst():
    assert not blocked(UTC.localize(dt.datetime(2025, 1, 15, 21, 7)))  # 15:07 CT
    assert blocked(UTC.localize(dt.datetime(2025, 1, 15, 21, 8)))      # 15:08 CT
    assert not blocked(UTC.localize(dt.datetime(2025, 1, 15, 23, 0)))  # 17:00 CT reopened


def test_flatten_boundary():
    # summer
    assert not flattens(UTC.localize(dt.datetime(2025, 7, 1, 20, 9)))  # 15:09 CT - hold
    assert flattens(UTC.localize(dt.datetime(2025, 7, 1, 20, 10)))     # 15:10 CT - flatten
    assert flattens(UTC.localize(dt.datetime(2025, 7, 1, 20, 30)))     # 15:30 CT - flatten
    # winter
    assert flattens(UTC.localize(dt.datetime(2025, 1, 15, 21, 10)))    # 15:10 CT
    # morning never flattens
    assert not flattens(UTC.localize(dt.datetime(2025, 7, 1, 14, 30))) # 09:30 CT


def test_evening_globex_allowed_not_flattened():
    ev = UTC.localize(dt.datetime(2025, 7, 1, 23, 0))  # 18:00 CT
    assert not blocked(ev) and not flattens(ev)
