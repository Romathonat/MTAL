import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.mtal.analysis import compute_line
from src.mtal.backtesting.common import BacktestResults
from src.mtal.backtesting.portfolio.rebalance import BacktestPorfolioResults


def plot_rsi(df_rsi, limit=30):
    df_rsi["Volume_MA14"] = df_rsi["Volume"].rolling(window=14).mean()

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        df_rsi.iloc[-limit:]["Close Time"],
        df_rsi.iloc[-limit:]["RSI"],
        label="RSI",
        color="purple",
    )
    ax.axhline(70, linestyle="--", alpha=0.5, color="red")
    ax.axhline(30, linestyle="--", alpha=0.5, color="green")

    base_line = 0
    scaled_volume = (
        df_rsi.iloc[-limit:]["Volume"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)

    ax.bar(
        df_rsi.iloc[-limit:]["Close Time"],
        scaled_volume,
        width=5,
        bottom=base_line,
        color="grey",
        alpha=0.8,
    )

    scaled_volume_MA = (
        df_rsi.iloc[-limit:]["Volume_MA14"] / df_rsi.iloc[-limit:]["Volume"].max()
    ) * (70 * 0.3)

    ax.plot(
        df_rsi.iloc[-limit:]["Close Time"],
        scaled_volume_MA + base_line,
        label="Volume MA14",
        color="orange",
        linewidth=1,
    )

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

    ax.plot(
        df_rsi[-limit:, "index"],
        df_rsi[-limit:, "RSI"],
        label="RSI",
        color="purple",
    )
    ax.axhline(70, linestyle="--", alpha=0.5, color="red")
    ax.axhline(30, linestyle="--", alpha=0.5, color="green")

    base_line = 0
    scaled_volume = (df_rsi[-limit:, "Volume"] / df_rsi[-limit:, "Volume"].max()) * (
        70 * 0.3
    )

    ax.bar(
        df_rsi[-limit:, "index"],
        scaled_volume,
        width=0.8,
        bottom=base_line,
        color="grey",
        alpha=0.8,
    )

    scaled_volume_MA = (
        df_rsi[-limit:, "Volume_MA"] / df_rsi[-limit:, "Volume"].max()
    ) * (70 * 0.3)

    ax.plot(
        df_rsi[-limit:, "index"],
        scaled_volume_MA + base_line,
        label="Volume MA14",
        color="orange",
        linewidth=1,
    )

    ax.set_title("RSI sur 14 périodes et Volume avec Volume MA14")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()

    a, b = compute_line(x_1, x_2, y_1, y_2)
    draw_line(a, b, x_1, len(df_rsi))
    plt.scatter(
        df_rsi[[x_1, x_2], "index"], [y_1, y_2], color="red", label="Points spécifiques"
    )

    plt.show()


def get_x_y_from_df(df_rsi, i_1, i_2):
    x_1 = df_rsi[-i_1, "index"]
    x_2 = df_rsi[-i_2, "index"]

    y_1 = df_rsi[x_1, "RSI"]
    y_2 = df_rsi[x_2, "RSI"]
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
    plt.figure(figsize=(10, 5))
    plt.plot(df["Close Time"], results.value_history, label="Portfolio Value")
    plt.plot(
        df["Close Time"],
        results.b_n_h_history,
        label="Benchmark Buy & Hold",
        color="orange",
    )
    plt.title("Portfolio Value Evolution")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_strategy_results(df: pd.DataFrame, results: BacktestResults):
    display_strat_value_over_time(df, results)
    print(f"pnl: {results.pnl}")
    print(f"Win Rate: {results.win_rate}")
    print(f"Max drawdown: {results.max_drawdown}")
    print(f"Trade number: {results.trade_number}")
    print(f"Excess return compared to B&H: {results.excess_return_vs_buy_and_hold}")
    print(f"Kelly Criterion: {results.kelly_criterion}")


def display_portfolio_value(results: BacktestPorfolioResults):
    plt.figure(figsize=(10, 5))  # Taille de la figure
    plt.plot(
        results.date_history, results.value_history, label="Valeur du Portefeuille"
    )
    plt.title("Portfolio Value Evolution")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_history(df: pl.DataFrame, price: float = None, start_idx: int = None, end_idx: int = None, limit: int = 100):
    """
    Affiche le graphique des prix avec les chandeliers japonais, le volume et une ligne de prix optionnelle
    
    Args:
        df: DataFrame Polars contenant les colonnes Open, High, Low, Close, Volume
        price: Prix pour la ligne horizontale
        start_idx: Indice de début pour la ligne
        end_idx: Indice de fin pour la ligne
        limit: Nombre de périodes à afficher
    """
    # Conversion des dernières lignes en pandas pour le plotting
    df_plot = df.tail(limit).to_pandas()
    
    # Création de la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[3, 1], gridspec_kw={'hspace': 0.3})
    
    # Calcul des couleurs
    colors = ['red' if close < open else 'green' 
              for close, open in zip(df_plot['Close'], df_plot['Open'])]
    
    # Graphique des chandeliers
    ax1.vlines(range(len(df_plot)), df_plot['Low'], df_plot['High'], color=colors)
    ax1.vlines(range(len(df_plot)), df_plot['Open'], df_plot['Close'], 
              color=colors, linewidth=4)
    
    # Ajout des bandes de Bollinger
    ax1.plot(range(len(df_plot)), df_plot['BB_hband'], 
            color='gray', linestyle='--', alpha=0.7, label='Bollinger Supérieure')
    ax1.plot(range(len(df_plot)), df_plot['BB_lband'], 
            color='gray', linestyle='--', alpha=0.7, label='Bollinger Inférieure')
    
    # Configuration du graphique des prix
    ax1.set_title('Prix, Volume et Bandes de Bollinger')
    ax1.set_ylabel('Prix')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Graphique du volume
    ax2.bar(range(len(df_plot)), df_plot['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Configuration des dates en x
    dates = df_plot['Open Time'].dt.strftime('%Y-%m-%d')
    plt.xticks(range(len(df_plot)), dates.to_list(), rotation=90)
    
    if price is not None and start_idx is not None and end_idx is not None:
        # Vérifier que les indices sont dans les limites
        start_idx = max(0, start_idx)
        end_idx = min(len(df_plot) - 1, end_idx)
        
        # Tracer la ligne horizontale bleue
        ax1.hlines(y=price, 
                  xmin=start_idx, 
                  xmax=end_idx, 
                  colors='blue', 
                  linestyles='--', 
                  label=f'Prix: {price}')
        ax1.legend()
    
    plt.show()
