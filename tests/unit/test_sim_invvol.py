"""Unit tests for the inverse-vol allocation paper-track (SIM_INVVOL).

Contract under test:
  1. InvVolScaler holds a CONSTANT contract count regardless of equity updates
     (it is a fixed-allocation testbed, not the equity-growing SimScaler).
  2. It duck-types the interface TSSimMirror uses (strategy / target_contracts /
     update_equity).
  3. TSSimMirror writes a SIM equity-curve CSV when equity_log_path is set
     (header once, append thereafter, contracts column from the scaler), and is
     a no-op otherwise.
"""

from pathlib import Path

from src.research.ts_sim_mirror import InvVolScaler, TSSimMirror


# --------------------------------------------------------------------------- #
# InvVolScaler
# --------------------------------------------------------------------------- #
def test_invvol_scaler_is_fixed_size():
    s = InvVolScaler("YANK", contracts=1)
    assert s.target_contracts() == 1
    # equity moving up or down must NOT change the size (unlike SimScaler)
    s.update_equity(50_000.0, 100_000.0)
    assert s.target_contracts() == 1
    s.update_equity(75_000.0)
    assert s.target_contracts() == 1
    s.update_equity(40_000.0)
    assert s.target_contracts() == 1
    assert s.equity == 40_000.0  # recorded for the equity log


def test_invvol_scaler_duck_types_scaler_interface():
    s = InvVolScaler("MIM-NB", contracts=1)
    assert s.strategy == "MIM-NB"
    assert hasattr(s, "target_contracts") and hasattr(s, "update_equity")
    # update_equity tolerates a None reading without raising
    s.update_equity(None)
    assert s.target_contracts() == 1


# --------------------------------------------------------------------------- #
# Equity-curve logging
# --------------------------------------------------------------------------- #
def test_equity_log_writes_header_then_appends(tmp_path: Path):
    log = tmp_path / "sub" / "yank_invvol_equity.csv"
    mirror = TSSimMirror(ts_auth=None, scaler=InvVolScaler("YANK", contracts=1),
                         equity_log_path=log)
    mirror._log_equity(50_000.0, 120_000.0)
    mirror._log_equity(50_250.0, 121_000.0)

    rows = log.read_text().splitlines()
    assert rows[0] == "ts_utc,equity,buying_power,contracts"
    assert len(rows) == 3  # header + 2 data rows (no duplicate header)
    # contracts column reflects the scaler's fixed size
    assert rows[1].endswith(",50000.0,120000.0,1")
    assert rows[2].endswith(",50250.0,121000.0,1")


def test_equity_log_noop_without_path(tmp_path: Path):
    mirror = TSSimMirror(ts_auth=None, scaler=InvVolScaler("YANK"), equity_log_path=None)
    # must not raise and must not create any file
    mirror._log_equity(50_000.0, 120_000.0)
    assert list(tmp_path.iterdir()) == []


def test_equity_log_handles_missing_buying_power(tmp_path: Path):
    log = tmp_path / "mim_invvol_equity.csv"
    mirror = TSSimMirror(ts_auth=None, scaler=InvVolScaler("MIM-NB", contracts=1),
                         equity_log_path=log)
    mirror._log_equity(50_000.0, None)
    last = log.read_text().splitlines()[-1]
    assert last.endswith(",50000.0,,1")  # empty buying_power field
