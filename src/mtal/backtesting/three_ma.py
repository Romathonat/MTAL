import polars as pl
from polars import DataFrame

from src.mtal.analysis import compute_ema, compute_hma, compute_vwma
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class ThreeMA(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        mid_ma=10,
        long_ma=15,
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
            self.data = compute_vwma(self.data, mid_ma)
            self.data = compute_vwma(self.data, long_ma)
        elif ma_type == "hma":
            self.data = compute_hma(self.data, short_ma)
            self.data = compute_hma(self.data, mid_ma)
            self.data = compute_hma(self.data, long_ma)
        else:
            self.data = compute_ema(self.data, short_ma)
            self.data = compute_ema(self.data, mid_ma)
            self.data = compute_ema(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.mid_ma, prefix=self.ma_type)]  # type: ignore
        )

        uptrend = (
            df[-1, get_ma_names(self.mid_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed and uptrend:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            >= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        if just_crossed and uncrossed_before:
            return True
        return False


class ThreeMARetest(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        mid_ma=10,
        long_ma=15,
        distance_retest=5,
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
            self.data = compute_vwma(self.data, mid_ma)
            self.data = compute_vwma(self.data, long_ma)
        elif ma_type == "hma":
            self.data = compute_hma(self.data, short_ma)
            self.data = compute_hma(self.data, mid_ma)
            self.data = compute_hma(self.data, long_ma)
        else:
            self.data = compute_ema(self.data, short_ma)
            self.data = compute_ema(self.data, mid_ma)
            self.data = compute_ema(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False

        retest = (
            df[-1, "Close"] - df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        ) / df[
            -1, get_ma_names(self.long_ma, prefix=self.ma_type)
        ] < self.distance_retest  # type: ignore

        uptrend = (
            df[-1, get_ma_names(self.mid_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if retest and uptrend:
            return True

        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        just_crossed = (
            df[-1, "Close"]  # type: ignore
            < df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            >= df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True

        downtrend = (
            df[-1, get_ma_names(self.mid_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if downtrend:
            return True

        return False
