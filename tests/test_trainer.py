from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.mtal.backtesting.ma_cross_backtest import MACrossBacktester
from src.mtal.backtesting.vzo_rsi import VZO_RSI
from src.mtal.trainer import train_strategy


@pytest.fixture
def sample_data():
    dates = pl.date_range(
        start=date(2020, 1, 1), end=date(2020, 7, 18), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
        ]
    )

    df = pl.DataFrame({"date": dates, "Open": prices})

    df = df.with_columns(
        pl.col("Open").shift(-1).alias("Close"),
        pl.col("date").alias("Open Time"),
        pl.col("date").shift(-1).alias("Close Time"),
        (pl.col("Open") * 10).alias("Volume"),
    )

    # Drop the last row to match pandas behavior as the shifted 'Close' creates a None at the last position
    df = df.filter(pl.col("date") != df.select(pl.max("date")).to_series()[0])

    return df


def test_trainer_no_ranges(sample_data: pl.DataFrame):
    ranges = {}

    best_params, train_results, test_results, _, _ = train_strategy(
        sample_data, VZO_RSI, ranges
    )  # type: ignore
    assert train_results is None and test_results is None


def test_trainer(sample_data: pl.DataFrame):
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
    assert train_results.exit_dates[0] == pd.Timestamp("2020-04-09 00:00:00")
    assert test_results.trade_number == 1
    assert test_results.exit_dates[0] == pd.Timestamp("2020-07-17 00:00:00")
    assert len(train_df) + 1 == len(test_df)


def test_trainer_ema_cross_one_combination(sample_data: pl.DataFrame):
    ranges = {
        "short_ma": range(3, 4),
        "long_ma": range(20, 21),
    }

    best_params, train_results, test_results, train_df, test_df = train_strategy(
        sample_data, MACrossBacktester, ranges, test_size=100
    )  # type: ignore

    assert len(best_params) == 2
    assert best_params == (3, 20)
    assert train_results.trade_number == 1
    assert train_results.win_rate == 1.0
    assert train_results.entry_dates[0] == pd.Timestamp("2020-02-28 00:00:00")
    assert train_results.entry_prices[0] == 101.63265306122449
    assert test_results.trade_number == 1
    assert test_results.exit_dates[0] == pd.Timestamp("2020-07-17 00:00:00")
    assert len(train_df) + 1 == len(test_df)


def test_trainer_fixed_size_test_window(sample_data: pl.DataFrame):
    ranges = {
        "span": range(1, 3),
        "grey_zone_rsi": range(1, 3),
        "grey_zone_vzo": range(1, 3),
    }

    best_params, train_results, test_results, train_df, test_df = train_strategy(
        sample_data, VZO_RSI, ranges, test_size=100
    )  # type: ignore

    assert len(best_params) == 3
    assert best_params == (1, 1, 1)
    assert train_results.trade_number == 1
    assert train_results.win_rate == 1.0
    assert train_results.exit_dates[0] == pd.Timestamp("2020-04-09 00:00:00")
    assert test_results.trade_number == 1
    assert test_results.exit_dates[0] == pd.Timestamp("2020-07-17 00:00:00")
    assert len(train_df) + 1 == len(test_df)
