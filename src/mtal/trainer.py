from itertools import product
from typing import Tuple, Type, Union

import pandas as pd
import polars as pl

from src.mtal.backtesting.common import AbstractBacktest, BacktestResults


def train_strategy(
    data: pl.DataFrame,
    backtester_class: Type[AbstractBacktest],
    ranges: dict,
    split=0.8,
) -> Union[Tuple, BacktestResults, BacktestResults, pd.DataFrame, pd.DataFrame]:
    if not ranges:
        return None, None, None, None, None

    cutoff = int(len(data) * split)

    train_data, test_data = data[:cutoff], data[cutoff:]

    keys, values = zip(*ranges.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    results = {}

    for params in param_combinations:
        backtester = backtester_class(train_data.clone(), **params)
        train_result = backtester.run()

        results[params.values()] = train_result

    best_combination = max(
        results, key=lambda x: results[x].excess_return_vs_buy_and_hold
    )
    train_result = results[best_combination]

    params_test = dict(zip(keys, best_combination))
    backtester = backtester_class(test_data.clone(), **params_test)
    test_results = backtester.run()

    return tuple(best_combination), train_result, test_results, train_data, test_data
