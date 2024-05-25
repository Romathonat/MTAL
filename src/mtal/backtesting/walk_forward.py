from typing import List, Tuple, Type

import polars as pl

from src.mtal.backtesting.common import AbstractBacktest, BacktestResults
from src.mtal.trainer import train_strategy


class WalkForward:
    def __init__(
        self, data: pl.DataFrame, backtester: Type[AbstractBacktest], ranges: dict, k=5
    ):
        self.data = data
        self.backtester = backtester
        self.ranges = ranges
        # we create indices of k+1 segments
        self.segments, self.segment_size = self._create_segments(k + 1)

    def run(self) -> List[Tuple[pl.DataFrame, BacktestResults]]:
        # we run a complete trainer on 0, i, we test on i, i+1
        results = []
        for segment_i in self.segments[2:]:
            data = self.data[0:segment_i]
            test_size = self.segment_size
            best_params, train_results, test_results, train_df, test_df = (
                train_strategy(data, self.backtester, self.ranges, test_size=test_size)
            )
            print(best_params)
            results.append((data[-test_size:], test_results))
        return results

    def _create_segments(self, k: int) -> Tuple[List[int], int]:
        segment_size = len(self.data) // k
        segments = []
        for i in range(k):
            segments.append(i * segment_size)
        segments.append(len(self.data) - 1)

        return segments, segment_size
