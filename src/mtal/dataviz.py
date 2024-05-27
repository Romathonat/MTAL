import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mtal.analysis import compute_line
from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.portfolio.rebalance import BacktestPorfolioResults


def plot_rsi(df_rsi, limit=30):
    # Calcul de la moyenne mobile sur 14 périodes pour le volume
    df_rsi["Volume_MA14"] = df_rsi["Volume"].rolling(window=14).mean()

    fig, ax = plt.subplots(figsize=(15, 6))

    # Tracé du RSI
    ax.plot(
        df_rsi.iloc[-limit:]["Close Time"],
        df_rsi.iloc[-limit:]["RSI"],
        label="RSI",
        color="purple",
    )
    ax.axhline(70, linestyle="--", alpha=0.5, color="red")
    ax.axhline(30, linestyle="--", alpha=0.5, color="green")

    # Mise à l'échelle des volumes pour l'affichage
    base_line = 0
    scaled_volume = (
        df_rsi.iloc[-limit:]["Volume"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)  # Ajustement pour l'affichage

    # Tracé des volumes
    ax.bar(
        df_rsi.iloc[-limit:]["Close Time"],
        scaled_volume,
        width=5,
        bottom=base_line,
        color="grey",
        alpha=0.8,
    )

    # Mise à l'échelle de la moyenne mobile du volume pour l'affichage
    scaled_volume_MA = (
        df_rsi.iloc[-limit:]["Volume_MA14"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)

    # Tracé de la moyenne mobile du volume
    ax.plot(
        df_rsi.iloc[-limit:]["Close Time"],
        scaled_volume_MA + base_line,
        label="Volume MA14",
        color="orange",
        linewidth=1,
    )

    # Configuration du graphique
    ax.set_title("RSI sur 14 périodes et Volume avec Volume MA14")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()

    plt.show()


def draw_line(a, b, start_point, max_size):
    def equation(x):
        return a * x + b

    x = np.array(range(start_point, max_size))
    y = equation(x)

    filtered_x, filtered_y = [], []

    for x_, y_ in zip(x, y):
        if y_ > 0 and y_ < 100:
            filtered_x.append(x_)
            filtered_y.append(y_)
    plt.plot(filtered_x, filtered_y)


def plot_rsi_with_line(x_1, x_2, y_1, y_2, df_rsi, limit=100):
    fig, ax = plt.subplots(figsize=(15, 6))

    # Tracé du RSI
    ax.plot(
        df_rsi.iloc[-limit:].index,
        df_rsi.iloc[-limit:]["RSI"],
        label="RSI",
        color="purple",
    )
    ax.axhline(70, linestyle="--", alpha=0.5, color="red")
    ax.axhline(30, linestyle="--", alpha=0.5, color="green")

    # Mise à l'échelle des volumes pour l'affichage
    base_line = 0
    scaled_volume = (
        df_rsi.iloc[-limit:]["Volume"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)  # Ajustement pour l'affichage

    # Tracé des volumes
    ax.bar(
        df_rsi.iloc[-limit:].index,
        scaled_volume,
        width=0.8,
        bottom=base_line,
        color="grey",
        alpha=0.8,
    )

    # Mise à l'échelle de la moyenne mobile du volume pour l'affichage
    scaled_volume_MA = (
        df_rsi.iloc[-limit:]["Volume_MA"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)

    # Tracé de la moyenne mobile du volume
    ax.plot(
        df_rsi.iloc[-limit:].index,
        scaled_volume_MA + base_line,
        label="Volume MA14",
        color="orange",
        linewidth=1,
    )

    # Configuration du graphique
    ax.set_title("RSI sur 14 périodes et Volume avec Volume MA14")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()

    a, b = compute_line(x_1, x_2, y_1, y_2)
    draw_line(a, b, x_1, len(df_rsi))
    plt.scatter(
        df_rsi.index[[x_1, x_2]], [y_1, y_2], color="red", label="Points spécifiques"
    )

    plt.show()


def get_x_y_from_df(df_rsi, i_1, i_2):
    x_1 = df_rsi.index[-i_1]
    x_2 = df_rsi.index[-i_2]

    y_1 = df_rsi.loc[x_1, "RSI"]
    y_2 = df_rsi.loc[x_2, "RSI"]
    return x_2, x_1, y_2, y_1


def display_top_k_lines(lines, df_rsi, top_k=3, limit=40):
    for line in lines[:top_k]:
        plot_rsi_with_line(line.x_1, line.x_2, line.y_1, line.y_2, df_rsi, limit=limit)


def display_stock(limit, best_lines):
    for line, stock, df_rsi in best_lines:
        plot_rsi_with_line(
            line.x_1, line.x_2, line.y_1, line.y_2, df_rsi=df_rsi, limit=limit
        )
        print(
            f"Asset: {stock}, Score: {line.score}, URL: https://www.tradingview.com/chart/?symbol={stock[:-3]}&interval=1W"
        )


def display_crypto(best_lines, limit):
    for line, pair, df_rsi in best_lines:
        plot_rsi_with_line(
            line.x_1, line.x_2, line.y_1, line.y_2, df_rsi=df_rsi, limit=limit
        )
        print(
            f"Asset: {pair}, Score: {line.score}, URL: https://www.tradingview.com/chart/?symbol=BINANCE:{pair}&interval=1W"
        )


def display_strat_value_over_time(df: pd.DataFrame, results: BacktestResults):
    plt.figure(figsize=(10, 5))  # Taille de la figure
    plt.plot(df["Close Time"], results.value_history, label="Valeur du Portefeuille")
    plt.plot(
        df["Close Time"],
        results.b_n_h_history,
        label="Benchmark Buy & Hold",
        color="orange",
    )
    plt.title("Évolution de la Valeur du Portefeuille")  # Titre du graphique
    plt.xlabel("Date")  # Étiquette de l'axe des x
    plt.ylabel("Valeur")  # Étiquette de l'axe des y
    plt.legend()  # Ajouter une légende
    plt.grid(True)  # Ajouter une grille pour faciliter la lecture
    plt.tight_layout()  # Ajuster automatiquement les paramètres de la figure
    plt.show()  # Afficher le graphique


def display_strategy_results(df: pd.DataFrame, results: BacktestResults):
    display_strat_value_over_time(df, results)
    print(f"pnl: {results.pnl}")
    print(f"Win Rate: {results.win_rate}")
    print(f"Max drawdown: {results.max_drawdown}")
    print(f"Trade number: {results.trade_number}")
    print(f"Excess return compared to B&H: {results.excess_return_vs_buy_and_hold}")


def display_portfolio_value(results: BacktestPorfolioResults):
    plt.figure(figsize=(10, 5))  # Taille de la figure
    plt.plot(
        results.date_history, results.value_history, label="Valeur du Portefeuille"
    )
    plt.title("Évolution de la Valeur du Portefeuille")  # Titre du graphique
    plt.xlabel("Date")  # Étiquette de l'axe des x
    plt.ylabel("Valeur")  # Étiquette de l'axe des y
    plt.legend()  # Ajouter une légende
    plt.grid(True)  # Ajouter une grille pour faciliter la lecture
    plt.tight_layout()  # Ajuster automatiquement les paramètres de la figure
    plt.show()  # Afficher le graphique
