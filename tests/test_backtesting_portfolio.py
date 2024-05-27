from datetime import date

import numpy as np
import polars as pl
import pytest

from src.mtal.backtesting.portfolio.rebalance import PortfolioRebalance


@pytest.fixture
def sample_data_1():
    dates = pl.date_range(
        start=date(2020, 1, 1), end=date(2020, 7, 18), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=100, stop=100, num=50),  # Descend
            np.linspace(start=150, stop=150, num=50),  # Remonte
            np.linspace(start=100, stop=100, num=50),  # Descend
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
def sample_data_2():
    dates = pl.date_range(
        start=date(2020, 1, 1), end=date(2020, 7, 18), interval="1d", eager=True
    )
    prices = np.concatenate(
        [
            np.linspace(start=100, stop=100, num=50),
            np.linspace(start=50, stop=50, num=50),
            np.linspace(start=100, stop=100, num=50),
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


def test_backtesting_portfolio(
    sample_data_1: pl.DataFrame,
    sample_data_2: pl.DataFrame,
):
    assets = [sample_data_1, sample_data_2]
    weights = [50, 50]
    results = PortfolioRebalance(assets, weights, freq="M").run()

    assert results.pnl == 1333.3333333333335


# TODO: check different asset size in time

# def test_backtesting_portfolio_no_sum_equal_to_one(
#     sample_data_1: pl.DataFrame,
#     sample_data_2: pl.DataFrame,
#     sample_data_3: pl.DataFrame,
# ):
#     results = PortfolioRebalance(sample_data_1, sample_data_2, sample_data_3, 10, 20, 70)

#     pass

# def test_backtesting_portfolio_with_cash(
#     sample_data_1: pl.DataFrame,
#     sample_data_2: pl.DataFrame,
#     sample_data_3: pl.DataFrame,
# ):
#     cash_data = sample_data_1.clone()
#     cash_data[:, "Close"] = 1000
#     results = PortfolioRebalance(sample_data_1, sample_data_2, sample_data_3, cash_data, 10, 20, 70)

#     pass