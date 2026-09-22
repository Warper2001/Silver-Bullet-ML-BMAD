import numpy as np
import pandas as pd
import pytest

from tools import kronos_inference_pilot as pilot


def minutes():
    index = pd.date_range("2025-02-03T14:31:00Z", periods=45, freq="min")
    return pd.DataFrame(
        {
            "timestamp": index.astype(str),
            "contract": pilot.CONTRACT,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        }
    )


def test_context_excludes_future_and_aggregates():
    frame = minutes()
    result = pilot.build_context(frame, "2025-02-03T15:00:00Z", 2)
    assert len(result) == 2
    assert result.volume.tolist() == [150.0, 150.0]
    frame.loc[30:, "close"] = -999
    pd.testing.assert_frame_equal(
        result, pilot.build_context(frame, "2025-02-03T15:00:00Z", 2)
    )


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "naive", "price", "contract", "nan"]
)
def test_rejects_bad_context(problem):
    frame = minutes()
    if problem == "missing":
        frame = frame.drop(index=2)
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif problem == "naive":
        frame["timestamp"] = frame.timestamp.str.replace("+00:00", "", regex=False)
    elif problem == "price":
        frame.loc[0, "high"] = 1
    elif problem == "contract":
        frame["contract"] = "OTHER"
    else:
        frame.loc[0, "close"] = np.nan
    with pytest.raises(pilot.AuditError):
        pilot.build_context(frame, "2025-02-03T15:00:00Z", 2)


def test_forecast_checks():
    context = pilot.build_context(minutes(), "2025-02-03T15:00:00Z", 2)
    assert pilot.validate_forecast(context, context.index) == {
        "all_finite": True,
        "invalid_candles": 0,
    }
    context.iloc[0, 1] = 1
    assert pilot.validate_forecast(context, context.index)["invalid_candles"] == 1
    context.iloc[0, 1] = np.nan
    assert not pilot.validate_forecast(context, context.index)["all_finite"]
    with pytest.raises(pilot.AuditError):
        pilot.validate_forecast(context.drop(columns="amount"), context.index)


def test_safe_destination_and_missing_offline_cache(tmp_path):
    assert pilot.check_destination(tmp_path / "report") == tmp_path / "report"
    for path in [tmp_path, tmp_path / "data" / "report"]:
        with pytest.raises(pilot.AuditError):
            pilot.check_destination(path)
    with pytest.raises(pilot.AuditError):
        pilot.prepare_source(tmp_path / "cache", True)
