
import ccxt
import pandas as pd

from strategy_rsi_ema import generate_signals as strat_rsi_ema
from strategy_breakout import generate_signals as strat_breakout
from strategy_vwap import generate_signals as strat_vwap
from strategy_grid import generate_signals as strat_grid
from risk import position_size_usdt

def simulate(df, cfg):
    # simple sim identical to backtest
    df = df.copy()
    balance = 1000.0
    position = None
    entry = 0.0
    wins = losses = 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        if position is None and bool(row.get("buy_signal", False)):
            size = position_size_usdt(balance, cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["stop_loss_pct"]) / price
            position = size
            entry = price
        elif position is not None:
            sell = bool(row.get("sell_signal", False)) or (price <= entry * (1 - cfg["risk"]["stop_loss_pct"]/100.0)) or (price >= entry * (1 + cfg["risk"]["take_profit_pct"]/100.0))
            if sell:
                pnl = (price - entry) * position
                balance += pnl
                if pnl >= 0: wins += 1
                else: losses += 1
                position = None
    return {"ending_balance": round(balance,2), "trades": wins+losses, "wins": wins, "losses": losses}

def fetch_df(symbol, timeframe, limit=500):
    ex = ccxt.mexc({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def run_grid(strategy_type, symbols, timeframe, ranges: dict, base_cfg: dict):
    results = []
    if isinstance(symbols, str):
        symbols = [symbols]
    for symbol in symbols:
        df = fetch_df(symbol, timeframe, limit=500)
        if strategy_type == "rsi_ema":
        for ema_fast in ranges.get("ema_fast", [9]):
            for ema_slow in ranges.get("ema_slow", [21]):
                for rsi_buy in ranges.get("rsi_buy", [32]):
                    for rsi_sell in ranges.get("rsi_sell", [68]):
                        d = strat_rsi_ema(df.copy(), ema_fast, ema_slow, base_cfg["strategy"]["rsi_period"], rsi_buy, rsi_sell)
                        sim = simulate(d, base_cfg)
                        sim.update({"ema_fast":ema_fast,"ema_slow":ema_slow,"rsi_buy":rsi_buy,"rsi_sell":rsi_sell})
                        sim['market'] = symbol
                        results.append(sim)
    elif strategy_type == "breakout":
        for lookback in ranges.get("lookback",[50]):
            for buffer_bps in ranges.get("buffer_bps",[10]):
                d = strat_breakout(df.copy(), lookback=lookback, buffer_bps=buffer_bps)
                sim = simulate(d, base_cfg)
                sim.update({"lookback":lookback,"buffer_bps":buffer_bps})
                sim['market'] = symbol
                        results.append(sim)
    elif strategy_type == "vwap":
        for window in ranges.get("window",[30]):
            for band_bps in ranges.get("band_bps",[20]):
                d = strat_vwap(df.copy(), window=window, band_bps=band_bps)
                sim = simulate(d, base_cfg)
                sim.update({"window":window,"band_bps":band_bps})
                sim['market'] = symbol
                        results.append(sim)
    elif strategy_type == "grid":
        for step_pct in ranges.get("step_pct",[0.6]):
            for band_mult in ranges.get("band_mult",[1.0]):
                d = strat_grid(df.copy(), step_pct=step_pct, band_mult=band_mult, ema_slow=base_cfg["strategy"]["ema_slow"])
                sim = simulate(d, base_cfg)
                sim.update({"step_pct":step_pct,"band_mult":band_mult})
                sim['market'] = symbol
                        results.append(sim)
        return results
