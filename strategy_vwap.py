
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, window: int = 30, band_bps: int = 20, **kwargs):
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.rolling(window, min_periods=1).sum()
    cum_vol = df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)
    vwap = (cum_pv / cum_vol).fillna(df["close"])
    df["vwap"] = vwap
    band = vwap * (band_bps/10000.0)
    df["buy_signal"] = df["close"] < (vwap - band)
    df["sell_signal"] = df["close"] > (vwap + band)
    return df
