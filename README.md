# Market Technical Analysis Lab 


# Unit test reproducibility
To have a visual representation of what is happening in unit tests, you can copy code and put it into a notebook:
```python
import pandas as pd

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
        "RSI": [80, 71, 48, 69, 51, 51, 68, 64, 75, 70],
        "ema5": [
            26.153542,
            26.235695,
            26.023796,
            25.949198,
            26.032798,
            26.088532,
            25.792355,
            25.861570,
            26,
            25.382920,
        ],
        "Volume_MA": [313.15, 322.90, 332.15, 325.55, 272.85, 292, 300, 300, 300, 300],
    }

# we need a longer dataset for rolling windows
for key, item in data.items():
    data[key] = item[:-1] + item

df = pd.DataFrame(data=data)

plot_rsi_with_line(*get_x_y_from_df(df, 4, 7), df, limit=100)
```