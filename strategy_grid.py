
import pandas as pd

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def generate_signals(df: pd.DataFrame, step_pct: float = 0.6, band_mult: float = 1.0, ema_slow: int = 21, **kwargs):
    df = df.copy()
    mid = ema(df["close"], ema_slow)
    band = mid * (step_pct/100.0) * band_mult
    # buy near lower band, sell near upper band
    df["buy_signal"] = df["close"] <= (mid - band)
    df["sell_signal"] = df["close"] >= (mid + band)
    return df
