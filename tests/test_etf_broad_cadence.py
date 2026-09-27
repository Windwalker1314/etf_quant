import pandas as pd
import pytest

from steadyquant.etf_broad_study import first_session_mask


def test_weekly_signal_moves_to_first_observed_session_after_holiday():
    dates = pd.to_datetime(["2026-09-18", "2026-09-22", "2026-09-23", "2026-09-28"])
    assert first_session_mask(dates, "weekly").tolist() == [True, True, False, True]
    assert first_session_mask(dates, "monthly").tolist() == [True, False, False, False]
    assert first_session_mask(dates, "daily").tolist() == [True, True, True, True]
    with pytest.raises(ValueError):
        first_session_mask(dates, "hourly")
