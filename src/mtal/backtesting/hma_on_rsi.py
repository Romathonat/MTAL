from polars import DataFrame

from src.mtal.analysis import compute_ema_on_rsi, compute_hma_on_rsi, compute_rsi
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class HMA_RSI_CROSS(AbstractBacktest):
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

        self.data = compute_rsi(self.data, window=14)

        if ma_type == "hma":
            self.data = compute_hma_on_rsi(self.data, short_ma)
            self.data = compute_hma_on_rsi(self.data, long_ma)
        else:
            self.data = compute_ema_on_rsi(self.data, short_ma)
            self.data = compute_ema_on_rsi(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
        )

        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        """
        We get out if price below long ma
        """
        if len(df) < 3:
            return False

        cross_down = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_RSI")]  # type: ignore
        )

        if cross_down:
            return True

        return False
