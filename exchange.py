import os, time
import ccxt
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

class MexcSpot:
    def __init__(self):
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.live = os.getenv("LIVE_TRADING", "False").lower() == "true"
        self.exchange = ccxt.mexc({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def fetch_balance(self):
        if self.api_key and self.api_secret:
            return self.exchange.fetch_balance()
        return {"total": {"USDT": 1000.0}}  # رصيد افتراضي للمحاكاة

    def place_order(self, symbol, side, amount, price=None):
        if not self.live:
            return {"info": "SIMULATED", "symbol": symbol, "side": side, "amount": amount, "price": price}
        if price is None:
            return self.exchange.create_market_order(symbol, side, amount)
        else:
            return self.exchange.create_limit_order(symbol, side, amount, price)
