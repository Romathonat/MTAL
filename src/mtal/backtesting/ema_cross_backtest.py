from pandas import DataFrame

from src.mtal.analysis import compute_ema
from src.mtal.backtesting.common import AbstractBacktest
from src.mtal.utils import get_ema_names


class EMACrossBacktester(AbstractBacktest):
    def __init__(self, data, short_ema, long_ema):
        super().__init__(data)
        compute_ema(self.data, short_ema)
        compute_ema(self.data, long_ema)
        self.short_ema = short_ema
        self.long_ema = long_ema

    def is_enter(self, df: DataFrame):
        """
        We enter at the current open if the previous ema is a cross
        """
        if len(df) < 3:
            return False

        just_crossed = (
            df.iloc[-2][get_ema_names(self.short_ema)]
            > df.iloc[-2][get_ema_names(self.long_ema)]
        )
        uncrossed_before = (
            df.iloc[-3][get_ema_names(self.short_ema)]
            <= df.iloc[-3][get_ema_names(self.long_ema)]
        )
        if just_crossed and uncrossed_before:
            return True
        return False

    def is_exit(self, df: DataFrame):
        if len(df) < 3:
            return False

        just_crossed = (
            df.iloc[-2][get_ema_names(self.short_ema)]
            < df.iloc[-2][get_ema_names(self.long_ema)]
        )
        uncrossed_before = (
            df.iloc[-3][get_ema_names(self.short_ema)]
            >= df.iloc[-3][get_ema_names(self.long_ema)]
        )
        if just_crossed and uncrossed_before:
            return True
        return False
