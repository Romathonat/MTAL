from abc import ABC, abstractmethod
from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class BacktestResults:
    pnl: float
    pnl_percentage: float
    max_drawdown: float
    win_rate: float
    average_return: float
    trade_number: float


class AbstractBacktest(ABC):
    def __init__(self, data, cash=1000) -> None:
        self.data = data
        self.price_entered = 0
        self.init_cash = cash
        self.cash = cash
        self.current_bet = 0
        self.wins = 0
        self.losses = 0
        self.cash_history = [cash]
        self.return_history = []
        self.entry_dates = []
        self.exit_dates = []
        
    def run(self):
        for i in range(1, len(self.data)):
            current_df = self.data.iloc[:i]
            if self.is_enter(current_df):
                self._entering_update(current_df)
            elif self.current_bet != 0 and self.is_exit(current_df):
                self._exiting_update(current_df)

        if self.cash == 0:
            self._exiting_update(current_df)

        pnl_percentage = (self.cash - self.init_cash) / self.init_cash

        results = BacktestResults(
            pnl=self.cash,
            pnl_percentage=pnl_percentage,
            max_drawdown=min(self.return_history),
            win_rate=self.wins / (self.wins + self.losses),
            average_return=sum(self.return_history) / len(self.return_history),
            trade_number=len(self.return_history),
        )
        return results

    def _exiting_update(self, current_df):
        variation = self._update_cash(current_df)
        self.current_bet = 0
        self.price_entered = 0
        self.cash_history.append(self.cash)
        self.return_history.append(variation)

    def _update_cash(self, current_df):
        variation = (
            current_df.iloc[-1]["Close"] - self.price_entered
        ) / self.price_entered

        if variation > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.cash = (1 + variation) * self.current_bet
        return variation

    def _entering_update(self, current_df):
        self.price_entered = current_df.iloc[-1]["Close"]
        self.current_bet, self.cash = self.cash, self.current_bet

    @abstractmethod
    def is_enter(self, df: DataFrame):
        pass

    @abstractmethod
    def is_exit(self, df: DataFrame):
        pass
