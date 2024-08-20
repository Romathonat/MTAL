import time
from datetime import datetime

import polars as pl
from binance.spot import Spot

client = Spot()
AUTHORIZED_PAIRS = {"USDT", "BTC"}

MARKET_SHORTNAME = {
    "Euronext Growth Paris": "PA",
    "Euronext Brussels, Paris": "BR",
    "Euronext Paris": "PA",
    "Euronext Amsterdam, Paris": "AS",
    "Euronext Access Paris": "PA",
    "Euronext Paris, Brussels": "PA",
    "Euronext Paris, Amsterdam, Brussels": "PA",
    "Euronext Amsterdam, Brussels, Paris": "AS",
    "Euronext Paris, Amsterdam": "PA",
}

API_STOCKS_TOKEN = "<TODO>"


def get_spot_pairs(only_vs_btc=False):
    infos = client.list_all_convert_pairs()

    cryptos = {}

    for info in infos:
        if only_vs_btc and info["toAsset"] == "BTC":
            cryptos[info["fromAsset"]] = info["toAsset"]
        elif info["toAsset"] in AUTHORIZED_PAIRS:
            cryptos[info["fromAsset"]] = info["toAsset"]

    return [f"{key}{value}" for key, value in cryptos.items()]


def date_to_ms_timestamp(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%d/%m/%y")
        timestamp_s = time.mktime(date_obj.timetuple())
        timestamp_ms = int(timestamp_s * 1000)

        return timestamp_ms
    except ValueError as e:
        print(f"Erreur: {e}. Assurez-vous que la date est au format 'jj/mm/aa'.")
    return None


def get_pair_df(
    pair="BTCUSDT",
    limit=400,
    start_time="20/01/18",
    end_time="20/01/25",
    frequency="1w",
):
    columns = [
        "Open Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close Time",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
        "Ignore",
    ]

    try:
        df = pl.DataFrame(
            data=client.klines(
                pair,
                interval=frequency,
                limit=limit,
                startTime=date_to_ms_timestamp(start_time),
                endTime=date_to_ms_timestamp(end_time),
            ),
        )

        df.columns = columns

    except Exception:
        out = pl.DataFrame(schema=[(col, pl.Float64) for col in columns])
        return out

    df = df[:, 0:7]
    df = df.with_columns(pl.from_epoch("Open Time", time_unit="ms"))
    df = df.with_columns(pl.from_epoch("Close Time", time_unit="ms"))

    df = df.with_columns(
        df["Open"].cast(pl.Float64),
        df["High"].cast(pl.Float64),
        df["Low"].cast(pl.Float64),
        df["Close"].cast(pl.Float64),
        df["Volume"].cast(pl.Float64),
    )
    print(pair)
    return df


def map_market(value):
    return MARKET_SHORTNAME.get(value, value)


def get_ticker_names():
    df = pl.read_csv("./data/stock_list.csv", separator=";", infer_schema_length=10000)
    df = df.with_columns(
        (
            pl.col("Symbol")
            + "."
            + pl.col("Market").apply(map_market, return_dtype=pl.String)
        ).alias("ticker_eodhd")
    )
    return df["ticker_eodhd"]


def get_stock_data(ticker, period="w"):
    url = f"https://eodhd.com/api/eod/{ticker}?period={period}&from=2020-01-05&api_token={API_STOCKS_TOKEN}&fmt=csv"
    try:
        df = pl.read_csv(url, try_parse_dates=True)
    except:
        return pl.DataFrame()
    if not len(df):
        return pl.DataFrame()
    df = df.with_columns(df["Date"].alias("Close Time"))
    return df
