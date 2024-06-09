import polars as pl
from polars import DataFrame

from src.mtal.analysis import compute_BB, compute_keltner_high, compute_keltner_low
from src.mtal.backtesting.common import AbstractBacktest


class Keltner(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        span,
        window_ATR,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_keltner_low(self.data, span=span, window_ATR=window_ATR)
        self.data = compute_keltner_high(self.data, span=span, window_ATR=window_ATR)
        self.trailing_stop = 0

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 30:
            return False

        just_crossed = (
            df[-1, "Close"]  # type: ignore
            > df[-1, "keltner_high"]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            <= df[-2, "keltner_high"]  # type: ignore
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


class BB(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        window,
        window_dev,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_BB(self.data, window=window, window_dev=window_dev)
        self.trailing_stop = 0

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 30:
            return False

        just_crossed = (
            df[-1, "Close"]  # type: ignore
            > df[-1, "BB_hband"]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            <= df[-2, "BB_hband"]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 30:
            return False

        self.trailing_stop = max(self.trailing_stop, df[-1, "BB_lband"])

        if df[-1, "Close"] < self.trailing_stop:
            self.trailing_stop = 0
            return True
        return False


class BB_silico(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        window,
        window_dev,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_BB(self.data, window=window, window_dev=window_dev)
        self.trailing_stop = 0

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 30:
            return False

        just_crossed = (
            df[-1, "Close"]  # type: ignore
            > df[-1, "BB_lband"]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            <= df[-2, "BB_lband"]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 30:
            return False

        just_crossed_high = (
            df[-1, "Close"]  # type: ignore
            < df[-1, "BB_hband"]  # type: ignore
        )
        uncrossed_before_high = (
            df[-2, "Close"]  # type: ignore
            >= df[-2, "BB_hband"]  # type: ignore
        )

        if just_crossed_high and uncrossed_before_high:
            return True

        just_crossed_mid = (
            df[-1, "Close"]  # type: ignore
            < df[-1, "BB_mid"]  # type: ignore
        )
        uncrossed_before_mid = (
            df[-2, "Close"]  # type: ignore
            >= df[-2, "BB_mid"]  # type: ignore
        )

        if just_crossed_mid and uncrossed_before_mid:
            return True

        just_crossed_low = (
            df[-1, "Close"]  # type: ignore
            < df[-1, "BB_lband"]  # type: ignore
        )
        just_crossed_low = (
            df[-2, "Close"]  # type: ignore
            >= df[-2, "BB_lband"]  # type: ignore
        )

        if just_crossed_low and just_crossed_low:
            return True

        return False
