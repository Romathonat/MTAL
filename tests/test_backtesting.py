import numpy as np
import pandas as pd
import pytest
from mtal.analysis import compute_ema

from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.ema_cross_backtest import EMACrossBacktester


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2020-01-01", periods=150)
    prices = np.concatenate(
        [
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
            np.linspace(start=110, stop=100, num=50),  # Descend
        ]
    )
    df = pd.DataFrame(data={"date": dates, "price": prices})
    df["EMA_3"] = compute_ema(df, 3)
    df["EMA_5"] = compute_ema(df, 5)

    df.set_index("date", inplace=True)
    return df


def test_ema_cross_backtester(sample_data: pd.DataFrame):
    short_ema = 10
    long_ema = 30
    tester = EMACrossBacktester(
        short_ema=short_ema, long_ema=long_ema, data=sample_data
    )
    results = tester.run()

    assert isinstance(results, BacktestResults)
    assert results.total_return is not None
    assert results.max_drawdown is not None
    assert results.win_rate is not None
    assert results.average_return is not None
