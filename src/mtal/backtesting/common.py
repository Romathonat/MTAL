from abc import ABC
from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class BacktestResults:
    total_return: float
    max_drawdown: float
    win_rate: float
    average_return: float

class AbstractBacktest(ABC):
    def __init__(self, cash=1000) -> None:
        self.price_entered = 0
        self.cash = cash
        self.current_bet = 0

    def run(self, df: DataFrame):
        pass

    def is_enter(self, df: DataFrame):
        pass

    def is_exit(self, df: DataFrame):
        pass
