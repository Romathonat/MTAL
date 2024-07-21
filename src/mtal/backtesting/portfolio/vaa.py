from src.mtal.analysis import compute_vaa_momentum
from src.mtal.backtesting.portfolio.rebalance import BacktestPorfolioResults


class PortfolioRebalanceOnMomentum:
    def __init__(self, assets, freq="M", value=1000) -> None:
        self.assets = assets
        self.add_obv_assets()
        weight_value = 100 / len(assets)
        self.weights = [weight_value / 100 for _ in assets]
        self.freq = freq

        asset_values = []
        asset_tick_values = []
        for i, asset in enumerate(assets):
            asset_values.append(self.weights[i] * value)
            asset_tick_values.append(
                asset[0, "Close"]
            )  # warning here we do not choose week nor monthly

        self.asset_values_history = [asset_values]
        self.asset_tick_values_history = [asset_tick_values]

        self.value_history = [value]
        self.date_history = [asset[0, "Date"]]

    def add_obv_assets(self):
        # we do not compute it for cash
        for i in range(len(self.assets) - 1):
            self.assets[i] = compute_vaa_momentum(self.assets[i])

    def run(self):
        for i in range(len(self.assets[0])):
            current_date = self.assets[0][i, "Date"]
            if self.is_start_of_period(current_date):
                self.rebalance(i)
                self.update_tick_value_history(i)

        return BacktestPorfolioResults(
            pnl=self.value_history[-1],
            value_history=self.value_history,
            date_history=self.date_history,
        )

    def update_weights(self, index):
        best_asset_i = -1
        best_max = 0
        for i, asset in enumerate(self.assets[:-1]):
            if asset[index, "VAA_Momentum"] > best_max:
                best_max = asset[index, "VAA_Momentum"]
                best_asset_i = i

        self.weights = [0 for i in self.assets]
        self.weights[best_asset_i] = 1

    def update_tick_value_history(self, i):
        asset_tick_values = []
        for asset in self.assets:
            asset_tick_values.append(asset[i, "Close"])
        self.asset_tick_values_history.append(asset_tick_values)

    def is_start_of_period(self, date):
        if self.freq == "M":
            return date.month != self.date_history[-1].month
        elif self.freq == "W":
            return (
                date.isocalendar().week != self.date_history[-1].isocalendar().week
                or date.year != self.date_history[-1].year
            )
        else:
            raise ValueError("Frequency not supported")

    def rebalance(self, i):
        assets_new_values = []

        for asset_i, asset in enumerate(self.assets):
            variation = (
                asset[i, "Close"] - self.asset_tick_values_history[-1][asset_i]
            ) / self.asset_tick_values_history[-1][asset_i]
            new_value = self.asset_values_history[-1][asset_i] * (1 + variation)
            assets_new_values.append(new_value)
        total_value = sum(assets_new_values)

        self.update_weights(i)

        # Now we rebalance, we remove the value that is in excess and put it where there is less
        assets_values_rebalanced = [
            weight * total_value for weight in self.weights
        ]  # to keep the portfolio at the right proportion

        self.asset_values_history.append(assets_values_rebalanced)
        self.value_history.append(total_value)
        self.date_history.append(self.assets[0][i, "Date"])
