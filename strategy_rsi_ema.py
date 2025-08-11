import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period - 1) / 2, adjust=False).mean()
    ma_down = down.ewm(com=(period - 1) / 2, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def generate_signals(df: pd.DataFrame, ema_fast: int, ema_slow: int, rsi_period: int, rsi_buy: int, rsi_sell: int):
    close = df["close"]
    df["ema_fast"] = ema(close, ema_fast)
    df["ema_slow"] = ema(close, ema_slow)
    df["rsi"] = rsi(close, period=rsi_period)
    # إشارة شراء: تقاطع EMA + RSI منخفض
    df["buy_signal"] = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] <= rsi_buy)
    # إشارة بيع: تقاطع عكسي أو RSI مرتفع
    df["sell_signal"] = (df["ema_fast"] < df["ema_slow"]) | (df["rsi"] >= rsi_sell)
    return df
