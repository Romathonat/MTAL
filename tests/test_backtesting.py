from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.ma_cross_backtest import MACrossBacktester
from src.mtal.utils import generate_pinescript


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
        ]
    )

    dates = dates[: len(prices)]
    df = pl.DataFrame({"date": dates, "Open": prices})

    df = df.with_columns(
        pl.col("Open").shift(-1).alias("Close"),
        pl.col("date").alias("Open Time"),
        pl.col("date").shift(-1).alias("Close Time"),
        (pl.col("Open") * 10).alias("Volume"),
    )

    df = df.filter(pl.col("date") != df.select(pl.max("date")).to_series()[0])

    return df


@pytest.fixture
def sample_data_no_exit():
    dates = pl.date_range(
        start=date(2020, 1, 1), end=date(2020, 7, 18), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=110, num=50),  # Remonte
        ]
    )
    dates = dates[: len(prices)]

    df = pl.DataFrame({"date": dates, "Open": prices})

    df = df.with_columns(
        pl.col("Open").shift(-1).alias("Close"),
        pl.col("date").alias("Open Time"),
        pl.col("date").shift(-1).alias("Close Time"),
        (pl.col("Open") * 10).alias("Volume"),
    )

    df = df.filter(pl.col("date") != df.select(pl.max("date")).to_series()[0])
    return df


def test_ema_cross_backtester(sample_data: pl.DataFrame):
    short_ema = 3
    long_ema = 20
    tester = MACrossBacktester(short_ma=short_ema, long_ma=long_ema, data=sample_data)
    results = tester.run()
    assert isinstance(results, BacktestResults)
    assert results.pnl_percentage > 0
    assert results.max_drawdown is not None
    assert results.win_rate == 1
    assert results.average_return is not None
    assert results.normalized_pnl is not None
    assert len(results.value_history) == len(sample_data)
    assert len(results.b_n_h_history) == len(sample_data)
    assert results.excess_return_vs_buy_and_hold == 0.1554843910959824
    assert results.kelly_criterion == 100


def test_ema_cross_backtester_no_exit_except_ending(sample_data_no_exit: pl.DataFrame):
    short_ema = 3
    long_ema = 20
    tester = MACrossBacktester(
        short_ma=short_ema, long_ma=long_ema, data=sample_data_no_exit
    )
    results = tester.run()
    assert isinstance(results, BacktestResults)
    assert results.pnl_percentage > 0
    assert results.max_drawdown is not None
    assert results.win_rate is not None
    assert results.average_return is not None
    assert results.entry_dates[0] == datetime(2020, 2, 28, 0, 0)
    assert results.entry_prices[0] == 101.63265306122449
    assert len(results.exit_dates) == 1


def test_generate_pine_entry_exit():
    pine_script = """
//@version=5
indicator("Manual Trades", overlay=true)

var int[] entryDates = array.new_int()
var int[] exitDates = array.new_int()

array.push(entryDates, 1513555199999)
array.push(entryDates, 1517788799999)
array.push(exitDates, 1519603199999)
array.push(exitDates, 1520812799999)

for i = 0 to array.size(exitDates) - 1
    label.new(x=array.get(entryDates, i), xloc=xloc.bar_time, y=close, yloc=yloc.belowbar, color=color.green, textcolor=color.white, style=label.style_label_up)
    label.new(x=array.get(exitDates, i), xloc=xloc.bar_time, y=close, yloc=yloc.abovebar, color=color.red, textcolor=color.white, style=label.style_label_down)
    """
    entry_dates = [1513555199999, 1517788799999]
    exit_dates = [1519603199999, 1520812799999]

    assert generate_pinescript(entry_dates, exit_dates).strip() == pine_script.strip()


def test_backtester_no_trade(sample_data_no_exit: pl.DataFrame):
    short_ema = 1
    long_ema = 1
    tester = MACrossBacktester(
        short_ma=short_ema, long_ma=long_ema, data=sample_data_no_exit
    )
    results = tester.run()

    assert results.win_rate == 0
    assert results.average_return == 0
    assert results.max_drawdown == 0
    assert results.entry_dates == []


def test_ema_cross_backtester_kelly_criterion_2_trades(sample_data: pl.DataFrame):
    short_ema = 3
    long_ema = 20
    tester = MACrossBacktester(
        short_ma=short_ema, long_ma=long_ema, data=pl.concat([sample_data, sample_data])
    )

    results = tester.run()
    assert isinstance(results, BacktestResults)
    assert results.pnl_percentage > 0
    assert results.max_drawdown is not None
    assert results.win_rate == 2 / 3
    assert results.average_return is not None
    assert results.normalized_pnl is not None
    assert len(results.value_history) == len(sample_data) * 2
    assert len(results.b_n_h_history) == len(sample_data) * 2
    assert results.excess_return_vs_buy_and_hold == 0.19021549088817305
    assert results.kelly_criterion == 16.067736185383204
