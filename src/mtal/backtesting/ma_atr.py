import polars as pl
from polars import DataFrame

from src.mtal.analysis import (
    compute_ehma,
    compute_ema,
    compute_hma,
    compute_keltner_low,
    compute_vwma,
)
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class MAATR(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        long_ma=10,
        ma_type="ema",
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        if ma_type == "vwma":
            self.data = compute_vwma(self.data, short_ma)
            self.data = compute_vwma(self.data, long_ma)
        elif ma_type == "hma":
            self.data = compute_hma(self.data, short_ma)
            self.data = compute_hma(self.data, long_ma)
        elif ma_type == "ehma":
            self.data = compute_ehma(self.data, short_ma)
            self.data = compute_ehma(self.data, long_ma)
        else:
            self.data = compute_ema(self.data, short_ma)
            self.data = compute_ema(self.data, long_ma)

        self.data = compute_keltner_low(self.data)
        self.trailing_stop = 0

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 30:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 30:
            return False
        self.trailing_stop = max(self.trailing_stop, df[-1, "keltner_low"])

        if df[-1, "Close"] < self.trailing_stop:
            self.trailing_stop = 0
            return True
        return False
