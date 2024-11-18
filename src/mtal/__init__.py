import time

from src.mtal.analysis import HISTORY_LIMIT, compute_rsi, get_best_valid_line
from src.mtal.data_collect import (
    get_pair_df,
    get_spot_pairs,
    get_stock_data,
    get_ticker_names,
)
from src.mtal.dataviz import display_crypto, display_stock

CRYPTO_NUMBER = 100
STOCK_NUMBER = 600


def screen_best_asset(
    limit=100, start_time="20/01/18", end_time="20/01/25", only_vs_btc=False, frequency="1w"
):
    pairs = get_spot_pairs(only_vs_btc=only_vs_btc)
    best_lines = list()

    for pair in pairs[:CRYPTO_NUMBER]:
        df = get_pair_df(
            pair=pair,
            limit=HISTORY_LIMIT,
            frequency=frequency,
            start_time=start_time,
            end_time=end_time,
        )
        df_rsi = compute_rsi(df)
        df_with_index = df_rsi.with_row_index()
        get_best_valid_line(best_lines, pair, df_with_index, limit)

    best_lines.sort(key=lambda x: x[0].score, reverse=True)
    display_crypto(best_lines, limit)


def screen_best_stocks(limit=100):
    stocks = get_ticker_names()
    best_lines = list()

    for stock in stocks[:STOCK_NUMBER]:
        df = get_stock_data(stock)
        df_rsi = compute_rsi(df)
        get_best_valid_line(best_lines, stock, df_rsi, limit)

    best_lines.sort(key=lambda x: x[0].score, reverse=True)
    display_stock(limit, best_lines)
