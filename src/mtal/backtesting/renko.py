import polars as pl
from polars import DataFrame

from src.mtal.analysis import compute_hma, compute_renko
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class RenkoDirection(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        span_atr,
        brick_size_factor,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_renko(
            data, span_atr=span_atr, brick_size_factor=brick_size_factor
        )

    def is_enter(self, df: DataFrame):
        if df[-1, "Direction"] == 1:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if df[-1, "Direction"] == -1:
            return True
        return False


class RenkoHMACross(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma,
        long_ma,
        span_atr,
        brick_size_factor,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.ma_type = "hma"
        self.data = compute_hma(self.data, short_ma)
        self.data = compute_hma(self.data, long_ma)
        self.data = compute_renko(
            self.data, span_atr=span_atr, brick_size_factor=brick_size_factor
        )

    def is_enter(self, df: DataFrame):
        renko_ok = df[-1, "Direction"] == 1

        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if renko_ok and just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        cross_down = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if cross_down:
            return True
        return False
