from polars import DataFrame

from src.mtal.analysis import compute_rsi, compute_vzo
from src.mtal.backtesting.common import AbstractBacktest


class VZO_RSI(AbstractBacktest):
    def __init__(self, data, span=14, grey_zone_rsi=5, grey_zone_vzo=5):
        super().__init__(data, params=self.extract_named_params(locals()))
        self.data = compute_rsi(self.data, window=span)
        self.data = compute_vzo(self.data, window=span)

    def is_enter(self, df: DataFrame):
        if len(df) < 3:
            return False

        vzo_val, rsi_val = df[-1, "VZO"], df[-1, "RSI"]

        if vzo_val > 0 + self.grey_zone_vzo * 2 and rsi_val > 50 + self.grey_zone_rsi:  # type: ignore
            return True
        return False

    def is_exit(self, df: DataFrame):
        return not self.is_enter(df)


class VZO_RSI_let_grey(AbstractBacktest):
    def __init__(self, data, span=14, grey_zone_rsi=5, grey_zone_vzo=5):
        super().__init__(data, params=self.extract_named_params(locals()))
        self.data = compute_rsi(self.data, window=span)
        self.data = compute_vzo(self.data, window=span)

    def is_enter(self, df: DataFrame):
        if len(df) < 3:
            return False

        vzo_val, rsi_val = df[-1, "VZO"], df[-1, "RSI"]

        if vzo_val > 0 + self.grey_zone_vzo and rsi_val > 50 + self.grey_zone_rsi:  # type: ignore
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        vzo_val, rsi_val = df[-1, "VZO"], df[-1, "RSI"]

        if vzo_val < 0 - self.grey_zone_vzo and rsi_val < 50 - self.grey_zone_rsi:  # type: ignore
            return True
        return False
