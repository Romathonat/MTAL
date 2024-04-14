import numpy as np
import pandas as pd

from src.mtal.analysis import compute_rsi


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
