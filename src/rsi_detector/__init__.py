from src.rsi_detector.analysis import HISTORY_LIMIT, compute_rsi, get_best_valid_line
from src.rsi_detector.data_collect import (
    get_pair_df,
    get_spot_pairs,
    get_stock_data,
    get_ticker_names,
)
from src.rsi_detector.dataviz import display_crypto, display_stock

CRYPTO_NUMBER = 500
STOCK_NUMBER = 600


def screen_best_asset(limit=100):
    pairs = get_spot_pairs()
    best_lines = list()

    for pair in pairs[:CRYPTO_NUMBER]:
        df = get_pair_df(pair=pair, limit=HISTORY_LIMIT, frequency="1w")
        df_rsi = compute_rsi(df)
        get_best_valid_line(best_lines, pair, df_rsi, limit)

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
