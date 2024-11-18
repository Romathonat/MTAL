import polars as pl
from src.mtal import CRYPTO_NUMBER
from src.mtal.analysis import HISTORY_LIMIT
from src.mtal.data_collect import (
    get_pair_df,
    get_spot_pairs,
    get_stock_data,
    get_ticker_names,
)

from src.mtal.dataviz import display_crypto

CRYPTO_NUMBER = 200
HISTORY_LIMIT = 200

def screen_best_asset(
    limit=100, start_time="20/01/18", end_time="20/01/25", frequency="1d"
):
    pairs = get_spot_pairs()
    best_lines = list()

    for pair in pairs[:CRYPTO_NUMBER]:
        df = get_pair_df(
            pair=pair,
            limit=HISTORY_LIMIT,
            frequency=frequency,
            start_time=start_time,
            end_time=end_time,
        )

    display_crypto(best_lines, limit)

def detect_tasse_hanse(df: pl.DataFrame):
    # On cherche d'abord les points hauts locaux
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=3).alias("local_high")
    ])
    
    # On identifie les points qui touchent la résistance (avec une tolérance de 1%)
    resistance_level = df["local_high"].max()
    tolerance = resistance_level * 0.01
    
    # Trouver les points de contact avec la résistance
    touches = df.filter(
        (pl.col("high") >= resistance_level - tolerance)
    ).select("timestamp", "high", "volume")
    
    if len(touches) < 3:
        return False
        
    # Vérifier que les touches sont de plus en plus proches de la résistance
    touches = touches.sort("timestamp")
    highs = touches["high"].to_list()
    if not all(abs(highs[i] - resistance_level) >= abs(highs[i+1] - resistance_level) 
              for i in range(len(highs)-1)):
        return False
    
    # Vérifier la condition de la tasse plus grande que la hanse
    pivot_idx = touches["timestamp"].to_list()[-1]
    tasse_length = (pivot_idx - touches["timestamp"].to_list()[0])
    
    # Chercher le breakout après le pivot
    breakout_df = df.filter(pl.col("timestamp") > pivot_idx)
    if len(breakout_df) == 0:
        return False
        
    breakout_point = breakout_df.filter(
        pl.col("high") > resistance_level + tolerance
    ).first()
    
    if breakout_point is None:
        return False
        
    hanse_length = breakout_point["timestamp"] - pivot_idx
    
    # Vérifier que la tasse est plus longue que la hanse
    if tasse_length <= hanse_length:
        return False
        
    # Vérifier le volume au breakout
    avg_volume = df["volume"].mean()
    if breakout_point["volume"] <= avg_volume:
        return False
    
    return True
    