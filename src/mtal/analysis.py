import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from ta.trend import WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import on_balance_volume

from src.mtal.utils import get_ma_names

THRESHOLD_CROSS = 3
NB_LAST_POINT_AUTHORIZED = 3
NB_PREVIOUS_POINT_NO_CROSS = 0
MAX_SLOPE_POSITIVE = 0.30
MIN_SLOPE_NEGATIVE = -0.90
HISTORY_LIMIT = 200
MINIMAL_SPACE_LINE_POINTS = 2
VOLATILITY_COMPRESSION_HISTORY = 10
VOLATILITY_COMPRESSION_THRESHOLD = 1


@dataclass
class Line:
    x_1: float
    x_2: float
    y_1: float
    y_2: float
    score: float
    a: float = 0.0
    b: float = 0.0


def compute_rsi(df: pl.DataFrame, window=14) -> pl.DataFrame:
    df_pd = df.to_pandas()

    if len(df_pd) == 0:
        return pl.DataFrame()
    df_pd["Change"] = df_pd["Close"].diff()

    df_pd["Gain"] = df_pd["Change"].mask(df_pd["Change"] < 0, 0)
    df_pd["Loss"] = -df_pd["Change"].mask(df_pd["Change"] > 0, 0)

    df_pd["Avg Gain"] = df_pd["Gain"].ewm(alpha=1 / window, adjust=False).mean()
    df_pd["Avg Loss"] = df_pd["Loss"].ewm(alpha=1 / window, adjust=False).mean()

    df_pd["RS"] = df_pd["Avg Gain"] / df_pd["Avg Loss"]
    df_pd["ema5"] = df_pd["Close"].ewm(span=5, adjust=False).mean()

    df_pd["RSI"] = 100 - (100 / (1 + df_pd["RS"]))
    df_pd["Volume_MA"] = df_pd["Volume"].rolling(window=20).mean()

    return pl.from_pandas(df_pd)


def compute_obv(df: pl.DataFrame) -> pl.DataFrame:
    df_pd = df.to_pandas()

    df_pd["OBV"] = on_balance_volume(df_pd["Close"], df_pd["Volume"])

    return pl.from_pandas(df_pd)


def compute_anchored_obv(df: pl.DataFrame, reset_period="1M"):
    df_pd = df.to_pandas()

    df_pd["Close Time"] = pd.to_datetime(df_pd["Close Time"])
    df_pd.sort_values("Close Time", inplace=True)

    # Créer un identifiant de période pour chaque groupe de réinitialisation
    df_pd["Period"] = df_pd["Close Time"].dt.to_period(reset_period)

    # Déterminer le début de chaque période de réinitialisation en fonction de l'année civile
    if reset_period == "1M":
        df_pd["Period"] = (
            df_pd["Close Time"].dt.year.astype(str)
            + "-"
            + df_pd["Close Time"].dt.month.astype(str).str.zfill(2)
        )
    elif reset_period == "3M":
        df_pd["Period"] = (
            df_pd["Close Time"].dt.year.astype(str)
            + "-"
            + ((df_pd["Close Time"].dt.month - 1) // 3 + 1).astype(str).str.zfill(2)
        )
    elif reset_period == "6M":
        df_pd["Period"] = (
            df_pd["Close Time"].dt.year.astype(str)
            + "-"
            + ((df_pd["Close Time"].dt.month - 1) // 6 + 1).astype(str).str.zfill(2)
        )
    elif reset_period == "1Y":
        df_pd["Close Time"].dt.year.astype(str)

    df_pd["Anchored_OBV"] = 0

    for period, period_data in df_pd.groupby("Period"):
        change_vector = period_data["Close"].diff()
        direction = [1 if change > 0 else -1 for change in change_vector]

        period_obv = (direction * period_data["Volume"]).cumsum()
        period_obv -= period_obv.iloc[0]

        # Mise à jour de la colonne OBV dans les données principales
        df_pd.loc[period_data.index, "Anchored_OBV"] = period_obv.astype(int)
    df_pd.drop(columns=["Period"], inplace=True)

    return pl.from_pandas(df_pd)


def compute_vaa_momentum(df: pl.DataFrame):
    df_pd = df.to_pandas()

    # Convertir les dates en datetime et trier
    df_pd["Close Time"] = pd.to_datetime(df_pd["Close Time"])
    df_pd.sort_values("Close Time", inplace=True)

    # Calculer le momentum pour 1, 3, 6 et 12 mois
    def calculate_momentum(shift_months):
        shift_weeks = 4 * shift_months
        past_prices = df_pd["Close"].shift(periods=shift_weeks)
        momentum = df_pd["Close"] - past_prices
        return momentum.fillna(0)

    # Périodes de momentum correspondant à 1M, 3M, 6M et 12M (en supposant 21 jours ouvrables par mois)
    m1 = calculate_momentum(1)
    m3 = calculate_momentum(3)
    m6 = calculate_momentum(6)
    m12 = calculate_momentum(12)

    # Calcul du VAA Momentum
    df_pd["VAA_Momentum"] = (12 * m1 + 4 * m3 + 2 * m6 + m12) / 19

    return pl.from_pandas(df_pd)


def compute_hma_on_obv(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()

    wma_half = weighted_moving_average(df["OBV"], span // 2)
    wma_full = weighted_moving_average(df["OBV"], span)

    df["data_hull"] = 2 * wma_half - wma_full
    df[get_ma_names(span, prefix="hma", suffix="_on_OBV")] = [
        int(i) if not np.isnan(i) else 0
        for i in weighted_moving_average(df["data_hull"], int(np.sqrt(span)))
    ]

    return pl.from_pandas(df)


def compute_vzo(df_in: pl.DataFrame, window=14) -> pl.DataFrame:
    df = df_in.to_pandas()
    if len(df) == 0:
        return pl.DataFrame()

    df["Price Change"] = df["Close"].diff()
    df["Volume Change"] = df["Volume"].diff()

    conditions = [
        df["Price Change"] > 0,
        df["Price Change"] < 0,
    ]

    choices = [
        df["Volume"],  # Si le prix monte, le volume total est pris
        -df["Volume"],  # Si le prix descend, le volume total est négatif
    ]

    df["Qualified Volume"] = np.select(conditions, choices)

    df["VZO Nominator"] = df["Qualified Volume"].ewm(span=window, adjust=False).mean()
    df["VZO Denominator"] = df["Volume"].ewm(span=window, adjust=False).mean()

    # df["VZO Nominator"] = ema_indicator(df["Qualified Volume"], window=window)
    # df["VZO Denominator"] = ema_indicator(df["Volume"], window=window)

    df["VZO"] = (df["VZO Nominator"] / df["VZO Denominator"]) * 100
    return pl.from_pandas(df)


def compute_ema(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()
    ema = df["Close"].ewm(span=span, adjust=False).mean()
    df[get_ma_names(span)] = ema
    return pl.from_pandas(df)


def compute_vwma(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    volume_prices = df_in["Close"] * df_in["Volume"]

    sum_volume_prices = volume_prices.rolling_sum(window_size=span)
    sum_volumes = df_in["Volume"].rolling_sum(window_size=span)

    vwma = sum_volume_prices / sum_volumes

    vwma_col_name = f"vwma_{span}"

    return df_in.with_columns(vwma.alias(vwma_col_name))


def weighted_moving_average(close, span=2):
    return WMAIndicator(close=close, window=span).wma()


def compute_hma(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()
    wma_half = weighted_moving_average(df["Close"], span // 2)
    wma_full = weighted_moving_average(df["Close"], span)
    df["data_hull"] = 2 * wma_half - wma_full
    df[get_ma_names(span, prefix="hma")] = [
        int(i) if not np.isnan(i) else 0
        for i in weighted_moving_average(df["data_hull"], int(np.sqrt(span)))
    ]

    return pl.from_pandas(df)


def compute_ehma(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()
    ema_half = df["Close"].ewm(span=span // 2, adjust=False).mean()
    ema_full = df["Close"].ewm(span=span, adjust=False).mean()
    df["data_hull"] = 2 * ema_half - ema_full
    df[get_ma_names(span, prefix="ehma")] = [
        int(i) if not np.isnan(i) else 0
        for i in weighted_moving_average(df["data_hull"], int(np.sqrt(span)))
    ]
    return pl.from_pandas(df)


def compute_atr(df_in: pl.DataFrame, span=14):
    df = df_in.to_pandas()
    df["ATR"] = AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=span
    ).average_true_range()
    return pl.from_pandas(df)


def compute_keltner_low(df_in: pl.DataFrame, span=20, window_ATR=3):
    df = df_in.to_pandas()
    df["keltner_low"] = KeltnerChannel(
        df["High"],
        df["Low"],
        df["Close"],
        window=span,
        window_atr=window_ATR,
        original_version=False,
    ).keltner_channel_lband()

    return pl.from_pandas(df)


def compute_keltner_high(df_in: pl.DataFrame, span=20, window_ATR=3):
    df = df_in.to_pandas()
    df["keltner_high"] = KeltnerChannel(
        df["High"],
        df["Low"],
        df["Close"],
        window=span,
        window_atr=window_ATR,
        original_version=False,
    ).keltner_channel_hband()

    return pl.from_pandas(df)


def compute_BB(df_in: pl.DataFrame, window: int = 20, window_dev=2):
    df = df_in.to_pandas()
    BB = BollingerBands(df["Close"], window=window, window_dev=window_dev)

    df["BB_hband"] = BB.bollinger_hband()
    df["BB_mid"] = BB.bollinger_mavg()
    df["BB_lband"] = BB.bollinger_lband()
    return pl.from_pandas(df)


def compute_heikin_ashin(df_in: pl.DataFrame):
    df = df_in.to_pandas()

    heikin_ashi_df = pd.DataFrame(
        index=df.index, columns=["Open", "High", "Low", "Close"]
    )

    heikin_ashi_df["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    heikin_ashi_df["Open"] = (df["Open"].shift(1) + df["Close"].shift(1)) / 2.0
    heikin_ashi_df["High"] = (
        heikin_ashi_df[["Open", "Close"]].join(df["High"]).max(axis=1)
    )
    heikin_ashi_df["Low"] = (
        heikin_ashi_df[["Open", "Close"]].join(df["Low"]).min(axis=1)
    )

    heikin_ashi_df.iloc[0]["Open"] = df.iloc[0]["Open"]
    heikin_ashi_df.iloc[0]["Close"] = (
        df.iloc[0]["Open"]
        + df.iloc[0]["High"]
        + df.iloc[0]["Low"]
        + df.iloc[0]["Close"]
    ) / 4.0
    heikin_ashi_df.iloc[0]["High"] = df.iloc[0]["High"]
    heikin_ashi_df.iloc[0]["Low"] = df.iloc[0]["Low"]

    df["ha_Close"] = heikin_ashi_df["Close"]
    df["ha_Open"] = heikin_ashi_df["Open"]
    df["ha_High"] = heikin_ashi_df["High"]
    df["ha_Low"] = heikin_ashi_df["Low"]

    return pl.from_pandas(df)


def compute_renko(df: pl.DataFrame, span_atr: int, brick_size_factor: float):
    df = compute_atr(df, span=span_atr)
    df_pd = df.to_pandas()

    prices = df_pd["Close"].tolist()

    brick_sizes = df_pd["ATR"] * brick_size_factor

    renko_directions = []  # +1 pour une brique montante, -1 pour une brique descendante
    renko_prices = []

    current_price = prices[0]
    last_brick = current_price

    for i, price in enumerate(prices):
        brick_size = brick_sizes[i]
        renko_direction = 0
        if price >= last_brick + brick_size:
            last_brick += brick_size
            renko_direction = 1
        elif price <= last_brick - brick_size:
            last_brick -= brick_size
            renko_direction = -1

        renko_prices.append(last_brick)
        renko_directions.append(renko_direction)

    df_pd["Renko_Price"] = renko_prices
    df_pd["Direction"] = renko_directions

    return pl.from_pandas(df_pd)


def compute_hma_on_rsi(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()
    wma_half = weighted_moving_average(df["RSI"], span // 2)
    wma_full = weighted_moving_average(df["RSI"], span)
    df["data_hull"] = 2 * wma_half - wma_full
    df[get_ma_names(span, prefix="hma", suffix="_on_RSI")] = [
        int(i) if not np.isnan(i) else 0
        for i in weighted_moving_average(df["data_hull"], int(np.sqrt(span)))
    ]
    return pl.from_pandas(df)


def compute_ema_on_rsi(df_in: pl.DataFrame, span=9) -> pl.DataFrame:
    df = df_in.to_pandas()
    ema = df["RSI"].ewm(span=span, adjust=False).mean()
    df[get_ma_names(span, prefix="ema", suffix="_on_RSI")] = ema
    return pl.from_pandas(df)


def compute_line(x_1, x_2, y_1, y_2):
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1
    return a, b


def is_valid_magic_line(x_1, x_2, y_1, y_2, df_rsi, limit=30) -> Line:
    a, b = compute_line(x_1, x_2, y_1, y_2)
    if a > MAX_SLOPE_POSITIVE or a < MIN_SLOPE_NEGATIVE:
        return Line(x_1=0, x_2=0, y_1=0, y_2=0, a=0, score=0, b=0)

    start_point = max(x_1 - NB_PREVIOUS_POINT_NO_CROSS, 0)
    df = df_rsi.filter(pl.col("index") >= start_point)
    total_points = len(df)

    points = 0
    previous_touching_point = False

    for i, (x, y) in enumerate(
        zip(
            df["index"],
            df["RSI"],
        )
    ):
        position_from_end = total_points - i - 0
        if y > a * x + b + THRESHOLD_CROSS:
            if (
                position_from_end > NB_LAST_POINT_AUTHORIZED
                or df[i, "Close"] < df[i, "ema5"]
            ):
                return Line(x_1=0, x_2=0, y_1=0, y_2=0, a=0, score=0, b=0)
            else:
                return Line(x_1=x_1, x_2=x_2, y_1=y_1, y_2=y_2, a=a, score=points, b=b)
        if (
            abs(a * x - y + b) / np.sqrt(a**2 + 1) <= THRESHOLD_CROSS
            and not previous_touching_point
        ):
            points += 1
            previous_touching_point = True
        else:
            previous_touching_point = False

    return Line(x_1=0, x_2=0, y_1=0, y_2=0, a=0, score=0, b=0)


def filter_similar_lines(lines, tolerance=1):
    filtered_lines = []

    for new_line in lines:
        similar_exists = False
        for existing_line in filtered_lines:
            if (
                abs(new_line.a - existing_line.a) < tolerance
                and abs(new_line.b - existing_line.b) < 10 * tolerance
            ):
                similar_exists = True
                break
        if not similar_exists:
            filtered_lines.append(new_line)

    return filtered_lines


def calculate_local_tops(df_rsi):
    is_higher_than_prev = df_rsi["RSI"] > df_rsi["RSI"].shift(1)
    is_higher_than_next = df_rsi["RSI"] > df_rsi["RSI"].shift(-1)

    local_tops = is_higher_than_prev & is_higher_than_next

    return local_tops


def is_invalid_setup(local_tops, idx1, row1, idx2):
    return (
        pd.isna(row1["RSI"])
        or idx2 - idx1 < MINIMAL_SPACE_LINE_POINTS
        or not local_tops[idx1]
        or not local_tops[idx2]
    )


def compute_and_validate_2_combinations(df_rsi: pl.DataFrame, limit=100):
    if not len(df_rsi):
        return None

    df_rsi = df_rsi.tail(limit)
    pairs = list(
        itertools.combinations(list(enumerate(df_rsi.iter_rows(named=True))), 2)
    )
    best_lines = list()
    local_tops = calculate_local_tops(df_rsi)
    for (idx1, row1), (idx2, row2) in pairs:
        if is_invalid_setup(local_tops, idx1, row1, idx2):
            continue

        line = is_valid_magic_line(
            row1["index"], row2["index"], row1["RSI"], row2["RSI"], df_rsi, limit=limit
        )

        if line.score > 0:
            best_lines.append(line)
    best_lines.sort(key=lambda x: x.score, reverse=True)
    best_lines_filtered = filter_similar_lines(best_lines)

    if best_lines_filtered:
        return best_lines_filtered[0]
    else:
        return None


def get_sum_line_distances(df, a, b):
    # y_line = a * df["index"] + b
    # # is_below = df["RSI"] <= y_line

    # # distances = np.where(
    # #     is_below, abs(a * df["index"] - df["RSI"] + b) / np.sqrt(a**2 + 1), 0
    # # )

    distances = abs(a * df["index"] - df["RSI"] + b) / np.sqrt(a**2 + 1)

    df = df.with_columns(pl.Series(name="distances", values=distances))

    df = df.with_columns(
        pl.Series(name="Volatility", values=df["distances"].rolling_sum(window_size=10))
    )

    df = df.with_columns(pl.col("Volatility").diff().alias("Volatility_tendency"))
    return df


def get_best_valid_line(best_lines, asset, df_rsi, limit):
    best_line = compute_and_validate_2_combinations(df_rsi[-limit:])

    if best_line:
        df_line_distance = get_sum_line_distances(df_rsi, best_line.a, best_line.b)

        # diminish_tendency = (
        #     df_line_distance[-10:, "Volatility_tendency"] < 0
        # ).mean() < VOLATILITY_COMPRESSION_THRESHOLD

        # if best_line.score > 1 and diminish_tendency:
        if best_line.score > 1:
            best_lines.append((best_line, asset, df_line_distance))
            return True
    return False
