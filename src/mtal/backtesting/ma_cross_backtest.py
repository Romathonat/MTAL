from pandas import DataFrame

from src.mtal.analysis import compute_ema, compute_hma, compute_vwma
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class MACrossBacktester(AbstractBacktest):
    def __init__(self, data, short_ma, long_ma, ma_type="ema"):
        super().__init__(data)
        self.ma_type = ma_type
        if ma_type == "vwma":
            compute_vwma(self.data, short_ma)
            compute_vwma(self.data, long_ma)
        elif ma_type == "hma":
            compute_hma(self.data, short_ma)
            compute_hma(self.data, long_ma)
        else:
            compute_ema(self.data, short_ma)
            compute_ema(self.data, long_ma)
        self.short_ema = short_ma
        self.long_ema = long_ma

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False

        just_crossed = (
            df.iloc[-1][get_ma_names(self.short_ema, prefix=self.ma_type)]
            > df.iloc[-1][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )
        uncrossed_before = (
            df.iloc[-2][get_ma_names(self.short_ema, prefix=self.ma_type)]
            <= df.iloc[-2][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        just_crossed = (
            df.iloc[-1][get_ma_names(self.short_ema, prefix=self.ma_type)]
            < df.iloc[-1][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )
        uncrossed_before = (
            df.iloc[-2][get_ma_names(self.short_ema, prefix=self.ma_type)]
            >= df.iloc[-2][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )
        if just_crossed and uncrossed_before:
            return True
        return False


class MACrossPriceAboveBacktester(AbstractBacktest):
    def __init__(self, data, short_ma=4, long_ma=6, ma_type="ema"):
        super().__init__(data)
        self.ma_type = ma_type
        if ma_type == "vwma":
            compute_vwma(self.data, short_ma)
            compute_vwma(self.data, long_ma)
        elif ma_type == "hma":
            compute_hma(self.data, short_ma)
            compute_hma(self.data, long_ma)
        else:
            compute_ema(self.data, short_ma)
            compute_ema(self.data, long_ma)
        self.short_ema = short_ma
        self.long_ema = long_ma

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False

        just_crossed = (
            df.iloc[-1][get_ma_names(self.short_ema, prefix=self.ma_type)]
            > df.iloc[-1][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )
        uncrossed_before = (
            df.iloc[-2][get_ma_names(self.short_ema, prefix=self.ma_type)]
            <= df.iloc[-2][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )
        price_above = (
            df.iloc[-1]["Close"]
            > df.iloc[-1][get_ma_names(self.long_ema, prefix=self.ma_type)]
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

        cross_down = (
            df.iloc[-1]["Close"]
            < df.iloc[-1][get_ma_names(self.long_ema, prefix=self.ma_type)]
        )

        if cross_down:
            return True

        return False
