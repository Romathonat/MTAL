import polars as pl
from src.mtal import CRYPTO_NUMBER
from src.mtal.analysis import HISTORY_LIMIT, compute_BB
from src.mtal.data_collect import (
    get_pair_df,
    get_spot_pairs,
    get_stock_data,
    get_ticker_names,
)
from src.mtal.dataviz import display_crypto

# Constants
CRYPTO_NUMBER = 200
HISTORY_LIMIT = 200
TOO_OLD_THRESHOLD = 10
TOLERANCE_THRESHOLD = 0.03
BARS_BETWEEN_TOUCHES = 3

def filter_touches(potential_touches: pl.DataFrame) -> list:
    """Filter touches to ensure minimum distance between them"""
    last_touch_idx = None
    filtered_touches = []
    
    for row in potential_touches.iter_rows(named=True):
        if last_touch_idx is None or (row["idx"] - last_touch_idx) > BARS_BETWEEN_TOUCHES:
            filtered_touches.append(row)
            last_touch_idx = row["idx"]
            
    return filtered_touches

def get_potential_touches(df: pl.DataFrame, resistance_level: float, pivot_idx: int, tolerance: float) -> pl.DataFrame:
    """Get all potential touches at a resistance level"""
    potential_touches = df.filter(
        ((pl.col("Close") >= resistance_level - tolerance) &
        (pl.col("Close") <= resistance_level + tolerance)) |
        ((pl.col("High") >= resistance_level - tolerance) &
        (pl.col("High") <= resistance_level + tolerance))
    ).select("Close Time", "Close", "Volume", "idx")
    
    # Handle breakout case
    for row in df.sort("Close Time", descending=True).iter_rows(named=True):
        if row["Close"] > resistance_level + tolerance and row["Close Time"] < pivot_idx:
            potential_touches = potential_touches.filter(pl.col("Close Time") > row["Close Time"])
            new_touch = create_touch_dataframe(row)
            potential_touches = pl.concat([new_touch, potential_touches])
            break
            
    return potential_touches

def create_touch_dataframe(row: dict) -> pl.DataFrame:
    """Create a DataFrame for a single touch point"""
    return pl.DataFrame(
        {k: [row[k]] for k in ["Close Time", "Close", "Volume", "idx"]},
        schema={
            "Close Time": pl.Datetime("ms"),
            "Close": pl.Float64,
            "Volume": pl.Float64,
            "idx": pl.Int64
        }
    )

def validate_pattern_metrics(df: pl.DataFrame, setup: dict) -> bool:
    """Validate the metrics of a potential cup and handle pattern"""
    cup_data = df.filter(
        (pl.col("idx") >= setup['begin']) &
        (pl.col("idx") <= setup['pivot'])
    )
    handle_data = df.filter(
        (pl.col("idx") > setup['pivot']) &
        (pl.col("idx") <= setup['end'])
    )
    
    max_cup_distance = abs(setup['resistance_level'] - cup_data["Close"].min())
    handle_distance = abs(setup['resistance_level'] - handle_data["Close"].min())
    
    tasse_length = setup['pivot'] - setup['begin']
    hanse_length = setup['end'] - setup['pivot']
    
    return (
        max_cup_distance * 1.2 > handle_distance and
        tasse_length > hanse_length and
        hanse_length > 3 and
        df[-1, "idx"] - setup['end'] < TOO_OLD_THRESHOLD
    )

def create_setup_dict_if_valid(df: pl.DataFrame, touches: pl.DataFrame, resistance_level: float, pivot_idx: int) -> dict:
    """
    Create a setup dictionary containing all relevant information about a potential cup and handle pattern
    
    Parameters:
    - df: Main DataFrame with price data
    - touches: DataFrame containing touch points
    - resistance_level: The resistance level being tested
    - pivot_idx: Index of the pivot point
    
    Returns:
    - dict: Setup information or None if invalid
    """
    first_touch_idx = df.filter(pl.col("Close Time") == touches[0, "Close Time"])["idx"].item()
    first_touch_price = touches[0, "Close"]

    # Check if pre-touch average is lower than first touch
    pre_touch_avg = df.filter(pl.col("idx") < first_touch_idx)["Close"].mean()
    if pre_touch_avg >= first_touch_price:
        return None

    # Get pivot and handle data
    pivot_bar_idx = df.filter(pl.col("Close Time") == pivot_idx)["idx"].item()
    handle_data = df.filter(pl.col("Close Time") > pivot_idx)
    
    if len(handle_data) == 0:
        return None



    # Calculate breakout point and volume
    tolerance = resistance_level * TOLERANCE_THRESHOLD
    breakout_point = handle_data.filter(pl.col("Close") > resistance_level + tolerance)

    # Setup end parameters
    if len(breakout_point) > 0:
        breakout_bar_idx = df.filter(pl.col("Close Time") == breakout_point[0, "Close Time"])["idx"].item()
        avg_volume = df[breakout_bar_idx-20:breakout_bar_idx, "Volume"].mean()

        breakout_volume_ratio = breakout_point[0, "Volume"] / avg_volume
        if breakout_volume_ratio < 1:
            return None

        if breakout_point[0, "Close"] < df[pivot_bar_idx, "BB_hband"]:
            return None
        end = df.filter(pl.col("Close Time") == breakout_point[0, "Close Time"])["idx"].item()
    else:
        breakout_volume_ratio = 0
        end = df[-1, "idx"]

    return {
        'resistance_level': resistance_level,
        'setup_size': pivot_bar_idx - first_touch_idx + (end - pivot_bar_idx),
        'touches': len(touches),
        'breakout_volume_ratio': breakout_volume_ratio,
        'pivot': pivot_bar_idx,
        'begin': first_touch_idx,
        'end': end
    }

def detect_cup_handle(df: pl.DataFrame):
    """Main function to detect cup and handle patterns"""
    df = df.with_columns([
        pl.col("Close").rolling_max(window_size=10).alias("local_high")
    ])

    df = compute_BB(df)
    
    potential_setups = []
    top_levels = df["local_high"].unique().sort(descending=True).head(20)[1:]
    
    for resistance_level in top_levels:
        pivot_idx = df.filter(pl.col("Close") == resistance_level)[0, "Close Time"]
        tolerance = resistance_level * TOLERANCE_THRESHOLD
        
        potential_touches = get_potential_touches(df, resistance_level, pivot_idx, tolerance)
        touches = pl.DataFrame(filter_touches(potential_touches))
        
        if len(touches) < 2:
            continue
            
        setup = create_setup_dict_if_valid(df, touches, resistance_level, pivot_idx)
        if setup and validate_pattern_metrics(df, setup):
            potential_setups.append(setup)
    
    return max(potential_setups, key=lambda x: x['resistance_level']) if potential_setups else False