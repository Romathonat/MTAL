import numpy as np
import pandas as pd
import pytest

from src.mtal.backtesting.vzo_rsi import VZO_RSI
from src.mtal.trainer import train_strategy


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2020-01-01", periods=200)
    prices = np.concatenate(
        [
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
        ]
    )
    df = pd.DataFrame(data={"date": dates, "Open": prices})
    df["Close"] = df["Open"].shift(-1)
    df["Open Time"] = df["date"]
    df["Close Time"] = df["date"].shift(-1)
    df["Volume"] = df["Open"] * 10
    df.drop(df.index[-1], inplace=True)
    df.set_index("date", inplace=True)

    return df


def test_trainer_no_ranges(sample_data: pd.DataFrame):
    ranges = {}

    best_params, train_results, test_results, _, _ = train_strategy(
        sample_data, VZO_RSI, ranges
    )  # type: ignore
    assert train_results is None and test_results is None


def test_trainer(sample_data: pd.DataFrame):
    ranges = {
        "span": range(1, 3),
        "grey_zone_rsi": range(1, 3),
        "grey_zone_vzo": range(1, 3),
    }

    best_params, train_results, test_results, train_df, test_df = train_strategy(
        sample_data, VZO_RSI, ranges, split=0.5
    )  # type: ignore

    assert len(best_params) == 3
    assert best_params == (1, 1, 1)
    assert train_results.trade_number == 1
    assert train_results.win_rate == 1.0
    assert train_results.exit_dates[0] == pd.Timestamp("2020-04-08 00:00:00")
    assert test_results.trade_number == 1
    assert test_results.exit_dates[0] == pd.Timestamp("2020-07-17 00:00:00")
    assert len(train_df) + 1 == len(test_df)
