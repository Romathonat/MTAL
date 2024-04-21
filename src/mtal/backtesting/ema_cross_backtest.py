from src.mtal.backtesting.common import AbstractBacktest, BacktestResults


class EMACrossBacktester(AbstractBacktest):
    def __init__(self, data, short_ema, long_ema):
        self.data = data
        self.short_ema = short_ema
        self.long_ema = long_ema

    def run(self):
        results = BacktestResults(
            total_return=self.data["daily_return"].sum(),
            max_drawdown=self.data["daily_return"].cumsum().min(),
            win_rate=len(self.data[self.data["daily_return"] > 0])
            / len(self.data["daily_return"]),
        )
        return results
