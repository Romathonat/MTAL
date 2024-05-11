from abc import ABC, abstractmethod
from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class BacktestResults:
    pnl: float
    normalized_pnl: float
    pnl_percentage: float
    max_drawdown: float
    win_rate: float
    average_return: float
    trade_number: float
    entry_dates: list
    exit_dates: list
    entry_prices: list
    exit_prices: list
    profit_pct_history: list
    profit_history: list
    value_history: list
    b_n_h_history: list
    excess_return_vs_buy_and_hold: float


class AbstractBacktest(ABC):
    def __init__(self, data, params={}, cash=1000) -> None:
        self.data = data
        self.cash = cash
        self.current_bet = 0
        self.wins = 0
        self.losses = 0
        self.cash_history = [cash]
        self.entry_dates = []
        self.exit_dates = []
        self.entry_prices = []
        self.exit_prices = []
        self.profit_pct_history = []
        self.profit_history = []
        self.value_history = [cash]
        self.b_n_h_history = [cash]

        for key, value in params.items():
            setattr(self, key, value)

    def extract_named_params(self, params):
        del params["self"]
        del params["data"]
        return params

    def run(self) -> BacktestResults:
        for i in range(2, len(self.data)):
            current_df = self.data.iloc[:i]
            if self.is_enter(current_df) and self.current_bet == 0:
                self._entering_update(current_df)
            elif self.is_exit(current_df) and self.current_bet != 0:
                self._exiting_update(current_df)
            variation_entry = self.get_variation_to_date(current_df)
            # print(current_df)
            variation_yesterday = (
                current_df.iloc[-1]["Close"] - current_df.iloc[-2]["Close"]
            ) / current_df.iloc[-2]["Close"]
            self.value_history.append(
                self.cash + (1 + variation_entry) * self.current_bet
            )
            self.b_n_h_history.append(
                self.b_n_h_history[-1] * (1 + variation_yesterday)
            )
        if self.cash == 0:
            self._exiting_update(current_df)

        pnl_percentage = (self.cash - self.cash_history[0]) / self.cash_history[0]

        max_drawdown = min(self.profit_pct_history) if self.profit_pct_history else 0
        win_rate = (
            self.wins / (self.wins + self.losses) if self.profit_pct_history else 0
        )
        average_return = (
            sum(self.profit_pct_history) / len(self.profit_pct_history)
            if self.profit_pct_history
            else 0
        )
        pnl = self.cash - self.cash_history[0]
        normalized_pnl = (
            pnl / len(self.profit_pct_history) if self.profit_pct_history else 0
        )

        results = BacktestResults(
            pnl=self.cash - self.cash_history[0],
            pnl_percentage=pnl_percentage,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            average_return=average_return,
            trade_number=len(self.profit_pct_history),
            entry_dates=self.entry_dates,
            exit_dates=self.exit_dates,
            entry_prices=self.entry_prices,
            exit_prices=self.exit_prices,
            profit_pct_history=self.profit_pct_history,
            profit_history=self.profit_history,
            normalized_pnl=normalized_pnl,
            value_history=self.value_history,
            b_n_h_history=self.b_n_h_history,
            excess_return_vs_buy_and_hold=self.get_excess_return_vs_buy_and_hold(pnl),
        )
        return results

    def get_variation_to_date(self, current_df: DataFrame):
        if self.current_bet:
            variation = (
                current_df.iloc[-1]["Close"] - self.entry_prices[-1]
            ) / self.entry_prices[-1]
        else:
            variation = 0
        return variation

    def get_excess_return_vs_buy_and_hold(self, pnl):
        buy_and_hold_perf = self.data.iloc[-1]["Close"] - self.data.iloc[0]["Close"]
        return (pnl - buy_and_hold_perf) / self.cash_history[0]

    def _entering_update(self, current_df):
        self.entry_dates.append(current_df.iloc[-1]["Close Time"])
        self.entry_prices.append(current_df.iloc[-1]["Close"])
        self.current_bet, self.cash = self.cash, self.current_bet

    def _exiting_update(self, current_df):
        self.exit_dates.append(current_df.iloc[-1]["Close Time"])
        self.exit_prices.append(current_df.iloc[-1]["Close"])

        variation = self._get_variation()
        self.profit_history.append(variation * self.current_bet)
        self.current_bet, self.cash = 0, (1 + variation) * self.current_bet

        self.profit_pct_history.append(variation)
        self.cash_history.append(self.cash)

    def _get_variation(self):
        variation = (self.exit_prices[-1] - self.entry_prices[-1]) / self.entry_prices[
            -1
        ]

        if variation > 0:
            self.wins += 1
        else:
            self.losses += 1
        return variation

    @abstractmethod
    def is_enter(self, df: DataFrame):
        pass

    @abstractmethod
    def is_exit(self, df: DataFrame):
        pass
