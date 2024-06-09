import polars as pl
from polars import DataFrame

from src.mtal.analysis import compute_ehma, compute_ema, compute_hma, compute_vwma
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class MACrossBacktester(AbstractBacktest):
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

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
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


class MACrossFakeBarBacktester(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        long_ma=10,
        bar_ratio=1000,
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

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        bar_ratio = df[-1, "Close"] - df[-1, "Open"] / df[-1, "Volume"]

        if just_crossed and uncrossed_before and bar_ratio > self.bar_ratio:
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


class MACrossAlphaBacktester(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        long_ma=10,
        ma_type="ema",
        alpha=1,
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

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
            * (1 + self.alpha / 100)  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
            * (1 + self.alpha / 100)  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]
            * (1 + self.alpha / 100)  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            >= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]
            * (1 + self.alpha / 100)  # type: ignore
        )
        if just_crossed and uncrossed_before:
            return True
        return False


class MACrossPriceAboveBacktester(AbstractBacktest):
    def __init__(
        self,
        data,
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
        else:
            self.data = compute_ema(self.data, short_ma)
            self.data = compute_ema(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        price_above = (
            df[-1, "Close"] > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed and uncrossed_before and price_above:
            return True
        return False

    def is_exit(self, df: DataFrame):
        """
        We get out if price below long ma
        """
        if len(df) < 3:
            return False

        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed:
            return True

        return False


class PriceCrossMABacktester(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        ma=5,
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
            self.data = compute_vwma(self.data, ma)
        elif ma_type == "hma":
            self.data = compute_hma(self.data, ma)
        else:
            self.data = compute_ema(self.data, ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) <= self.ma:
            return False
        just_crossed = (
            df[-1, "Close"]  # type: ignore
            > df[-1, get_ma_names(self.ma, prefix=self.ma_type)]  # type: ignore
        )

        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            <= df[-2, get_ma_names(self.ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) <= self.ma:
            return False

        just_crossed = (
            df[-1, "Close"]  # type: ignore
            < df[-1, get_ma_names(self.ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "Close"]  # type: ignore
            >= df[-2][get_ma_names(self.ma, prefix=self.ma_type)]  # type: ignore
        ).all()
        if just_crossed and uncrossed_before:
            return True
        return False


class MACrossBacktesterOnTrend(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma_down=5,
        long_ma_down=10,
        short_ma_up=5,
        long_ma_up=10,
        ma_trend=200,
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
        if ma_type == "hma":
            self.data = compute_hma(self.data, short_ma_down)
            self.data = compute_hma(self.data, long_ma_down)
            self.data = compute_hma(self.data, short_ma_up)
            self.data = compute_hma(self.data, long_ma_up)
        else:
            self.data = compute_ema(self.data, short_ma_down)
            self.data = compute_ema(self.data, long_ma_down)
            self.data = compute_ema(self.data, short_ma_up)
            self.data = compute_ema(self.data, long_ma_up)
        self.data = compute_ema(self.data, ma_trend)

    def is_enter(self, df: DataFrame):
        if len(df) < 3:
            return False

        uptrend = df[-1, "Close"] > df[-1, get_ma_names(self.ma_trend, prefix="ema")]  # type: ignore

        if uptrend:
            ma_short = get_ma_names(self.short_ma_up, prefix=self.ma_type)  # type: ignore
            ma_long = get_ma_names(self.long_ma_up, prefix=self.ma_type)  # type: ignore
        else:
            ma_short = get_ma_names(self.short_ma_down, prefix=self.ma_type)  # type: ignore
            ma_long = get_ma_names(self.long_ma_down, prefix=self.ma_type)  # type: ignore

        just_crossed = (
            df[-1, ma_short]  # type: ignore
            > df[-1, ma_long]  # type: ignore
        )
        uncrossed_before = (
            df[-2, ma_short]  # type: ignore
            <= df[-2, ma_long]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True

        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        uptrend = df[-1, "Close"] > df[-1, get_ma_names(self.ma_trend, prefix="ema")]  # type: ignore

        if uptrend:
            ma_short = get_ma_names(self.short_ma_up, prefix=self.ma_type)  # type: ignore
            ma_long = get_ma_names(self.long_ma_up, prefix=self.ma_type)  # type: ignore
        else:
            ma_short = get_ma_names(self.short_ma_down, prefix=self.ma_type)  # type: ignore
            ma_long = get_ma_names(self.long_ma_down, prefix=self.ma_type)  # type: ignore

        crossed = (
            df[-1, ma_short]  # type: ignore
            < df[-1, ma_long]  # type: ignore
        )

        if crossed:
            return True
        return False


class MACrossBacktesterNoTradeTrendDown(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma_up=5,
        long_ma_up=10,
        ma_trend=200,
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
        if ma_type == "hma":
            self.data = compute_hma(self.data, short_ma_up)
            self.data = compute_hma(self.data, long_ma_up)
        else:
            self.data = compute_ema(self.data, short_ma_up)
            self.data = compute_ema(self.data, long_ma_up)
        self.data = compute_ema(self.data, ma_trend)

    def is_enter(self, df: DataFrame):
        if len(df) < 3:
            return False

        uptrend = df[-1, "Close"] > df[-1, get_ma_names(self.ma_trend, prefix="ema")]  # type: ignore

        ma_short = get_ma_names(self.short_ma_up, prefix=self.ma_type)  # type: ignore
        ma_long = get_ma_names(self.long_ma_up, prefix=self.ma_type)  # type: ignore

        just_crossed = (
            df[-1, ma_short]  # type: ignore
            > df[-1, ma_long]  # type: ignore
        )
        uncrossed_before = (
            df[-2, ma_short]  # type: ignore
            <= df[-2, ma_long]  # type: ignore
        )

        if just_crossed and uncrossed_before and uptrend:
            return True

        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        ma_short = get_ma_names(self.short_ma_up, prefix=self.ma_type)  # type: ignore
        ma_long = get_ma_names(self.long_ma_up, prefix=self.ma_type)  # type: ignore

        cross_down = (
            df[-1, ma_short]  # type: ignore
            < df[-1, ma_long]  # type: ignore
        )

        if cross_down:
            return True
        return False
