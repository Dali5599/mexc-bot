import argparse, time
import pandas as pd
import ccxt
from strategy_rsi_ema import generate_signals
from risk import position_size_usdt

def run_backtest(market, timeframe, days):
    ex = ccxt.mexc({"enableRateLimit": True})
    limit = min(200, int(days * 24 * (60 / (1 if timeframe.endswith('m') else 60)) + 50))
    ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = generate_signals(df, 9, 21, 14, 32, 68)
    balance = 1000.0
    position = None
    entry = 0.0
    wins = losses = 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row["close"]
        if position is None and row["buy_signal"]:
            size = position_size_usdt(balance, 0.5, 0.6) / price
            position = size
            entry = price
        elif position is not None:
            if row["sell_signal"] or (price <= entry * (1 - 0.006)) or (price >= entry * (1 + 0.012)):
                pnl = (price - entry) * position
                balance += pnl
                if pnl >= 0: wins += 1
                else: losses += 1
                position = None
    return {"ending_balance": round(balance,2), "trades": wins+losses, "wins": wins, "losses": losses}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()
    res = run_backtest(args.market, args.timeframe, args.days)
    print(res)
