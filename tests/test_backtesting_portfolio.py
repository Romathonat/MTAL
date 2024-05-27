from datetime import date

import numpy as np
import polars as pl
import pytest

from src.mtal.backtesting.portfolio.rebalance import PortfolioRebalance


@pytest.fixture
def sample_data_1():
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 5, 30), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=100, stop=100, num=50),  # Descend
            np.linspace(start=150, stop=150, num=50),  # Remonte
            np.linspace(start=100, stop=100, num=50),  # Descend
        ]
    )

    dates = dates[: len(prices)]
    df = pl.DataFrame({"Date": dates, "Open": prices})

    df = df.with_columns(
        pl.col("Open").shift(-1).alias("Close"),
        pl.col("Date").alias("Open Time"),
        pl.col("Date").shift(-1).alias("Close Time"),
        (pl.col("Open") * 10).alias("Volume"),
    )

    df = df.filter(pl.col("Date") != df.select(pl.max("Date")).to_series()[0])

    return df


@pytest.fixture
def sample_data_2():
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 5, 30), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=100, stop=100, num=50),
            np.linspace(start=50, stop=50, num=50),
            np.linspace(start=100, stop=100, num=50),
        ]
    )

    dates = dates[: len(prices)]
    df = pl.DataFrame({"Date": dates, "Open": prices})

    df = df.with_columns(
        pl.col("Open").shift(-1).alias("Close"),
        pl.col("Date").alias("Open Time"),
        pl.col("Date").shift(-1).alias("Close Time"),
        (pl.col("Open") * 10).alias("Volume"),
    )

    df = df.filter(pl.col("Date") != df.select(pl.max("Date")).to_series()[0])

    return df


def test_backtesting_portfolio(
    sample_data_1: pl.DataFrame,
    sample_data_2: pl.DataFrame,
):
    assets = [sample_data_1, sample_data_2]
    weights = [50, 50]
    results = PortfolioRebalance(assets, weights, freq="M").run()
    assert results.pnl == 1333.3333333333335
    assert len(results.value_history) == len(results.date_history)
    print(sample_data_1["Date"])
    print(results.date_history)
    assert len(results.date_history) == 5


# TODO: check different asset size in time


def test_backtesting_portfolio_no_sum_equal_to_one(
    sample_data_1: pl.DataFrame,
    sample_data_2: pl.DataFrame,
):
    assets = [sample_data_1, sample_data_2]
    weights = [50, 60]

    with pytest.raises(ValueError, match="The sum of weights must be 100"):
        PortfolioRebalance(assets, weights, freq="M").run()


def test_backtesting_portfolio_with_cash(
    sample_data_1: pl.DataFrame,
    sample_data_2: pl.DataFrame,
):
    assets = [sample_data_1, sample_data_2]
    weights = [50, 30, 20]

    with pytest.raises(
        ValueError, match="The number of assets must match the number of weights"
    ):
        PortfolioRebalance(assets, weights, freq="M").run()
