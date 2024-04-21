import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mtal.utils import get_ema_names

THRESHOLD_CROSS = 1
NB_LAST_POINT_AUTHORIZED = 3
NB_PREVIOUS_POINT_NO_CROSS = 0
MAX_SLOPE_POSITIVE = 0.10
MIN_SLOPE_NEGATIVE = -0.50
HISTORY_LIMIT = 200
MINIMAL_SPACE_LINE_POINTS = 2
VOLATILITY_COMPRESSION_HISTORY = 10
VOLATILITY_COMPRESSION_THRESHOLD = 0.5


@dataclass
class Line:
    x_1: float
    x_2: float
    y_1: float
    y_2: float
    score: float
    a: float = 0.0
    b: float = 0.0


def compute_rsi(df):
    if len(df) == 0:
        return pd.DataFrame()

    df["Change"] = df["Close"].diff()

    df["Gain"] = df["Change"].mask(df["Change"] < 0, 0)
    df["Loss"] = -df["Change"].mask(df["Change"] > 0, 0)

    window_length = 14
    df["Avg Gain"] = df["Gain"].ewm(alpha=1 / window_length, adjust=False).mean()
    df["Avg Loss"] = df["Loss"].ewm(alpha=1 / window_length, adjust=False).mean()

    df["RS"] = df["Avg Gain"] / df["Avg Loss"]
    df["ema5"] = df["Close"].ewm(span=5, adjust=False).mean()

    df["RSI"] = 100 - (100 / (1 + df["RS"]))
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    return df


def compute_ema(df, span=9):
    ema = df["Close"].ewm(span=span, adjust=False).mean()
    df[get_ema_names(span)] = ema
    return df


def compute_line(x_1, x_2, y_1, y_2):
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1
    return a, b


def is_valid_magic_line(x_1, x_2, y_1, y_2, df_rsi, limit=30) -> Line:
    a, b = compute_line(x_1, x_2, y_1, y_2)
    if a > MAX_SLOPE_POSITIVE or a < MIN_SLOPE_NEGATIVE:
        return Line(x_1=0, x_2=0, y_1=0, y_2=0, a=0, score=0, b=0)
    start_point = max(x_1 - NB_PREVIOUS_POINT_NO_CROSS, 0)
    total_points = len(df_rsi.loc[start_point:])

    points = 0
    previous_touching_point = False

    for i, (x, y) in enumerate(
        zip(df_rsi.loc[start_point:].index, df_rsi.loc[start_point:]["RSI"])
    ):
        position_from_end = total_points - i - 1
        if y > a * x + b + THRESHOLD_CROSS:
            if (
                position_from_end > NB_LAST_POINT_AUTHORIZED
                or df_rsi.loc[x]["Close"] < df_rsi.loc[x]["ema5"]
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
        pd.isna(row1.RSI)
        or idx2 - idx1 < MINIMAL_SPACE_LINE_POINTS
        or not local_tops[idx1]
        or not local_tops[idx2]
    )


def compute_and_validate_2_combinations(df_rsi, limit=100):
    if not len(df_rsi):
        return None

    df_rsi = df_rsi.iloc[-limit:]
    pairs = list(itertools.combinations(list(df_rsi.iterrows()), 2))
    best_lines = list()
    local_tops = calculate_local_tops(df_rsi)

    for (idx1, row1), (idx2, row2) in pairs:
        if is_invalid_setup(local_tops, idx1, row1, idx2):
            continue

        line = is_valid_magic_line(idx1, idx2, row1.RSI, row2.RSI, df_rsi, limit=limit)
        if line.score > 0:
            best_lines.append(line)

    best_lines.sort(key=lambda x: x.score, reverse=True)
    best_lines_filtered = filter_similar_lines(best_lines)

    if best_lines_filtered:
        return best_lines_filtered[0]
    else:
        return None


def get_sum_line_distances(df, a, b):
    y_line = a * df.index + b
    is_below = df["RSI"] <= y_line
    df["distances"] = np.where(
        is_below, abs(a * df.index - df["RSI"] + b) / np.sqrt(a**2 + 1), 0
    )

    df["Volatility"] = df["distances"].rolling(window=10).sum()
    df["Volatility_tendency"] = df["Volatility"].diff()

    return df


def get_best_valid_line(best_lines, asset, df_rsi, limit):
    best_line = compute_and_validate_2_combinations(df_rsi.iloc[-limit:])

    if best_line:
        get_sum_line_distances(df_rsi, best_line.a, best_line.b)
        diminish_tendency = (
            df_rsi["Volatility_tendency"][-10:] < 0
        ).mean() > VOLATILITY_COMPRESSION_THRESHOLD
        if best_line.score > 1 and diminish_tendency:
            best_lines.append((best_line, asset, df_rsi))
