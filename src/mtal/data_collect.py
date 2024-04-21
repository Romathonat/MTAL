import pandas as pd
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

API_STOCKS_TOKEN = "6614351c645f34.55152589"


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
        df = pd.DataFrame(data=client.klines(pair, "1w", limit=limit), columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns)

    df = df.iloc[:, 0:7]
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    return df


def get_ticker_names():
    df = pd.read_csv("./data/stock_list.csv", sep=";")
    df["ticker_eodhd"] = df["Symbol"] + "." + df["Market"].map(MARKET_SHORTNAME)
    return df["ticker_eodhd"]


def get_stock_data(ticker):
    url = f"https://eodhd.com/api/eod/{ticker}?period=w&from=2020-01-05&api_token={API_STOCKS_TOKEN}&fmt=csv"
    try:
        df = pd.read_csv(url)
    except:
        return pd.DataFrame()
    if not len(df):
        return pd.DataFrame()
    df["Close Time"] = df["Date"]
    return df
