from datetime import date

import numpy as np
import polars as pl
import pytest

from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.vzo_rsi import VZO_RSI
from src.mtal.backtesting.walk_forward import WalkForward


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


def test_walk_forward(sample_data: pl.DataFrame):
    ranges = {
        "span": range(1, 3),
        "grey_zone_rsi": range(1, 3),
        "grey_zone_vzo": range(1, 3),
    }
    wf = WalkForward(sample_data, VZO_RSI, ranges, k=5)

    results = wf.run()

    assert len(results) == 5
    assert all(isinstance(result[1], BacktestResults) for result in results)
    assert len(results[0][0]) == 24
    assert results[-3][1].win_rate == 1
