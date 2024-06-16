import polars as pl
from polars import DataFrame

from src.mtal.analysis import compute_heikin_ashin, compute_hma
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class HeikinAshin(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_heikin_ashin(data)

    def is_enter(self, df: DataFrame):
        green = df[-1, "ha_Close"] > df[-1, "ha_Open"]

        if green:
            return True
        return False

    def is_exit(self, df: DataFrame):
        red = df[-1, "ha_Close"] <= df[-1, "ha_Open"]
        if red:
            return True
        return False


class HeikinAshinHMACross(AbstractBacktest):
    def __init__(
        self,
        data: pl.DataFrame,
        short_ma=5,
        long_ma=10,
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )
        self.data = compute_heikin_ashin(data)
        self.data = compute_hma(self.data, short_ma)
        self.data = compute_hma(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        ma_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix="hma")]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix="hma")]  # type: ignore
        )
        green_ha = df[-1, "ha_Close"] > df[-1, "ha_Open"]

        if green_ha and ma_crossed:
            return True
        return False

    def is_exit(self, df: DataFrame):
        ma_crossed_down = (
            df[-1, get_ma_names(self.short_ma, prefix="hma")]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix="hma")]  # type: ignore
        )

        red_ha = df[-1, "ha_Close"] <= df[-1, "ha_Open"]

        if red_ha and ma_crossed_down:
            return True
        return False
