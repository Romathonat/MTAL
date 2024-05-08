import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.mtal.analysis import compute_ema, compute_hull_moving_average, compute_rsi
from src.mtal.utils import get_ma_names


def test_compute_rsi_empty():
    df = pd.DataFrame()

    df_rsi = compute_rsi(df)

    assert len(df_rsi) == 0


def test_compute_rsi():
    data = {
        "Date": [
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
            "2024-02-26",
            "2024-03-04",
            "2024-03-11",
            "2024-03-18",
            "2024-03-25",
            "2024-04-02",
            "2024-04-08",
        ],
        "Open": [25.4, 27.0, 26.8, 26.8, 25.8, 26.2, 26.0, 26.2, 25.2, 26.0],
        "High": [26.8, 27.0, 26.8, 26.8, 26.2, 26.6, 26.2, 26.2, 26.2, 26.2],
        "Low": [24.8, 25.4, 25.4, 25.4, 25.4, 25.4, 25.2, 25.2, 25.0, 25.0],
        "Close": [26.0, 26.4, 25.6, 25.8, 26.2, 26.2, 25.2, 26.0, 25.0, 25.0],
        "Volume": [657, 259, 383, 298, 83, 435, 420, 128, 1054, 175],
    }

    df = pd.DataFrame(data=data)

    df_rsi = compute_rsi(df)

    expected_serie = pd.Series(
        [
            np.nan,
            100.0,
            86.66666666666667,
            87.12871287128714,
            88.02267895109851,
            88.02267895109851,
            73.26819766893904,
            76.64145736407187,
            65.51280826364248,
            65.51280826364248,
        ]
    )
    assert df_rsi["RSI"].all() == expected_serie.all()


def test_compute_ema():
    data = {
        "Date": [
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
            "2024-02-26",
            "2024-03-04",
            "2024-03-11",
            "2024-03-18",
            "2024-03-25",
            "2024-04-02",
            "2024-04-08",
        ],
        "Close": [67129, 64977, 65672, 63487, 63835, 61301, 63487, 63835, 64975, 65007],
    }

    df = pd.DataFrame(data=data)

    span = 3
    df_ema = compute_ema(df, span=span)

    expected_ema = pd.Series(
        [
            67129.0,
            66053.0,
            65862.5,
            64674.75,
            64254.875,
            62777.9375,
            63132.46875,
            63483.734375,
            64229.3671875,
            64618.18359375,
        ],
        index=df.index,
    )
    assert_series_equal(df_ema[get_ma_names(span)], expected_ema, check_names=False)


def test_compute_hull_moving_average():
    data = {
        "Date": [
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
            "2024-02-26",
            "2024-03-04",
            "2024-03-11",
            "2024-03-18",
            "2024-03-25",
            "2024-04-02",
            "2024-04-08",
        ],
        "Close": [48309, 52115, 51734, 63249, 69049, 68288, 67242, 71333, 69240, 65910],
    }

    df = pd.DataFrame(data=data)

    span = 2
    df_hma = compute_hull_moving_average(df, span=span)

    expected_hma = pd.Series(
        [0, 53383, 51607, 67087, 70982, 68034, 66893, 72696, 68542, 64800],
        index=df.index,
    )

    assert_series_equal(
        df_hma[get_ma_names(span, prefix="hma")], expected_hma, check_names=False
    )
