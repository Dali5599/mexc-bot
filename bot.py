import os, time, math
import pandas as pd
from dotenv import load_dotenv
from utils import load_config, log
from exchange import MexcSpot
from strategy_rsi_ema import generate_signals
from risk import position_size_usdt

load_dotenv()

def main():
    cfg = load_config("config.yaml")
    symbol = os.getenv("MARKET", cfg["market"])
    timeframe = cfg["timeframe"]
    ex = MexcSpot()
    os.makedirs(cfg["logs_dir"], exist_ok=True)
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = generate_signals(df, cfg["strategy"]["ema_fast"], cfg["strategy"]["ema_slow"],
                                  cfg["strategy"]["rsi_period"], cfg["strategy"]["rsi_buy"], cfg["strategy"]["rsi_sell"])
            last = df.iloc[-1]
            balance = ex.fetch_balance()
            usdt = balance.get("total", {}).get("USDT", 1000.0)
            # تحقق من السبريد (تقريبي باستخدام آخر شمعة)
            bid = last["close"] * 0.999  # تبسيط
            ask = last["close"] * 1.001
            spread_bps = (ask - bid) / ((ask + bid)/2) * 10000
            if spread_bps > cfg["max_spread_bps"]:
                log(f"سبريد مرتفع {spread_bps:.1f}bps — لا تداول الآن")
                time.sleep(10)
                continue
            # منطق الدخول/الخروج
            if last["buy_signal"]:
                notional = position_size_usdt(usdt, cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["stop_loss_pct"])
                qty = round(notional / ask, 6)
                res = ex.place_order(symbol, "buy", qty)  # أمر سوق افتراضي
                log(f"أمر شراء: {res}")
            elif last["sell_signal"]:
                # في نموذج بسيط نفترض مركز سابق مساوي لقيمة المخاطرة
                qty = round(position_size_usdt(usdt, cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["stop_loss_pct"]) / bid, 6)
                res = ex.place_order(symbol, "sell", qty)
                log(f"أمر بيع: {res}")
            time.sleep(15)
        except Exception as e:
            log(f"خطأ: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
