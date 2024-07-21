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


def get_spot_pairs():
    infos = client.list_all_convert_pairs()

    cryptos = {}

    for info in infos:
        if info["toAsset"] in AUTHORIZED_PAIRS:
            cryptos[info["fromAsset"]] = info["toAsset"]
    return [f"{key}{value}" for key, value in cryptos.items()]


def get_pair_df(pair="BTCUSDT", limit=400, frequency="1w"):
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
            data=client.klines(pair, interval=frequency, limit=limit),
        )
        df.columns = columns

    except Exception:
        out = pl.DataFrame()
        out.columns = columns
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
