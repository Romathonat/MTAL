from polars import DataFrame

from src.mtal.analysis import (
    compute_anchored_obv,
    compute_hma,
    compute_hma_on_obv,
    compute_obv,
)
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ma_names


class TWO_MA_OBV_CROSS(AbstractBacktest):
    def __init__(
        self,
        data,
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

        self.data = compute_obv(self.data)
        self.ma_type = "hma"
        self.data = compute_hma_on_obv(self.data, short_ma)
        self.data = compute_hma_on_obv(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
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
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
        )

        if cross_down:
            return True

        return False


class OBV_MA_CROSS(AbstractBacktest):
    def __init__(
        self,
        data,
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

        self.data = compute_obv(self.data)
        self.ma_type = "hma"
        self.data = compute_hma_on_obv(self.data, long_ma)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False
        just_crossed = (
            df[-1, "OBV"]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
        )
        uncrossed_before = (
            df[-2, "OBV"]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
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
            df[-1, "OBV"]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type, suffix="_on_OBV")]  # type: ignore
        )

        if cross_down:
            return True

        return False


class ANCHORED_OBV(AbstractBacktest):
    def __init__(
        self,
        data,
        reset_period="3M",
        cutoff_begin=None,
        cutoff_end=None,
    ):
        super().__init__(
            data,
            cutoff_begin=cutoff_begin,
            cutoff_end=cutoff_end,
            params=self.extract_named_params(locals()),
        )

        self.data = compute_anchored_obv(self.data, reset_period=reset_period)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False
        if df[-1, "Anchored_OBV"] > 0 and df[-2, "Anchored_OBV"] <= 0:
            return True
        return False

    def is_exit(self, df: DataFrame):
        """
        We get out if price below long ma
        """
        if len(df) < 3:
            return False

        if df[-1, "Anchored_OBV"] < 0:
            return True

        return False


class ANCHORED_OBV_HMA_CROSS(AbstractBacktest):
    def __init__(
        self,
        data,
        reset_period="3M",
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
        self.ma_type = "hma"
        self.data = compute_hma(self.data, short_ma)
        self.data = compute_hma(self.data, long_ma)
        self.data = compute_anchored_obv(self.data, reset_period=reset_period)

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open there is a cross and the price is above the long ma
        """
        if len(df) < 3:
            return False

        obv_ok = df[-1, "Anchored_OBV"] > 0
        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            > df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            <= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if obv_ok and just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        """
        We get out if price below long ma
        """
        if len(df) < 3:
            return False

        obv_not_ok = df[-1, "Anchored_OBV"] < 0

        just_crossed = (
            df[-1, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            < df[-1, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )
        uncrossed_before = (
            df[-2, get_ma_names(self.short_ma, prefix=self.ma_type)]  # type: ignore
            >= df[-2, get_ma_names(self.long_ma, prefix=self.ma_type)]  # type: ignore
        )

        if just_crossed:
            return True

        return False
