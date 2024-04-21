import numpy as np
import pandas as pd
import pytest

from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.ema_cross_backtest import EMACrossBacktester
from src.mtal.utils import generate_pinescript


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
    df = pd.DataFrame(data={"date": dates, "Close": prices})

    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def sample_data_no_exit():
    dates = pd.date_range(start="2020-01-01", periods=150)
    prices = np.concatenate(
        [
            np.linspace(start=110, stop=100, num=50),  # Descend
            np.linspace(start=100, stop=130, num=100),  # Remonte
        ]
    )
    df = pd.DataFrame(data={"date": dates, "Close": prices})

    df.set_index("date", inplace=True)
    return df


def test_ema_cross_backtester(sample_data: pd.DataFrame):
    short_ema = 3
    long_ema = 20
    tester = EMACrossBacktester(
        short_ema=short_ema, long_ema=long_ema, data=sample_data
    )
    results = tester.run()

    assert isinstance(results, BacktestResults)
    assert results.pnl_percentage > 0
    assert results.max_drawdown is not None
    assert results.win_rate is not None
    assert results.average_return is not None


def test_ema_cross_backtester_no_ending(sample_data_no_exit: pd.DataFrame):
    short_ema = 3
    long_ema = 20
    tester = EMACrossBacktester(
        short_ema=short_ema, long_ema=long_ema, data=sample_data_no_exit
    )
    results = tester.run()

    assert isinstance(results, BacktestResults)
    assert results.pnl_percentage > 0
    assert results.max_drawdown is not None
    assert results.win_rate is not None
    assert results.average_return is not None


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
