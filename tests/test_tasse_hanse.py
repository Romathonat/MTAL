import pytest
import polars as pl
from pathlib import Path
from src.mtal.froment.tasse_hanse import detect_cup_handle

@pytest.fixture
def load_test_data():
    def _load_test_data(pair: str) -> pl.DataFrame:
        path = Path("tests/data") / f"{pair.lower()}_data.parquet"
        df = pl.read_parquet(path)
        if "idx" not in df.columns:
            df = df.with_columns(pl.Series(name="idx", values=range(len(df))))
        return df
    return _load_test_data

@pytest.mark.parametrize("pair", [
    "solusdt",
    "trxusdt",
    "dogeusdt",
])
def test_detect_tasse_hanse_structure(load_test_data, pair):
    df = load_test_data(pair)
    result = detect_cup_handle(df)
    
    assert isinstance(result, dict)
    assert all(key in result for key in [
        'resistance_level',
        'setup_size',   
        'touches',
        'breakout_volume_ratio',
        'pivot',
        'begin',
        'end'
    ])
    
    # Vérifications logiques
    assert result['touches'] >= 2
    assert result['setup_size'] > 0
    assert result['begin'] < result['pivot'] < result['end']

def test_pattern_too_old(load_test_data):
    """Test sur un cas connu pour avoir un pattern"""
    pair = "aaveusdt"  # À adapter selon vos données de test
    df = load_test_data(pair)
    result = detect_cup_handle(df)
    
    assert result is False

def test_only_cup_detection(load_test_data):
    pair = "wifusdt"
    df = load_test_data(pair)
    result = detect_cup_handle(df)
    
    assert result['touches'] == 2
    assert result['breakout_volume_ratio'] == 0

# def test_volume_breakout_but_no_compression_bb(load_test_data):
#     pair = "solusdt_volume"
#     df = load_test_data(pair)
#     result = detect_cup_handle(df)
    
#     assert result['breakout_volume_ratio'] > 1
#     assert result['resistance_level'] == 184.49 
# def test_

def test_no_BB_begin_and_no_up_before(load_test_data):
    pair = "no_BB_begin"
    df = load_test_data(pair)
    result = detect_cup_handle(df)
    
    assert not result