
import pandas as pd

def generate_signals(df: pd.DataFrame, lookback: int = 50, buffer_bps: int = 10, **kwargs):
    df = df.copy()
    hi = df["high"].rolling(lookback, min_periods=lookback).max()
    lo = df["low"].rolling(lookback, min_periods=lookback).min()
    buf = (df["close"] * (buffer_bps/10000.0))
    df["buy_signal"] = df["close"] > (hi + buf)
    df["sell_signal"] = df["close"] < (lo - buf)
    return df
