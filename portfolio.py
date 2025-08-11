
from dataclasses import dataclass, field
import os, csv, sqlite3
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_PATH, exist_ok=True)
CSV_TRADES = os.path.join(LOG_PATH, "trades.csv")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "trades.db")

def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT,
  side TEXT,
  entry_price REAL,
  exit_price REAL,
  entry_time TEXT,
  exit_time TEXT,
  pnl REAL
)""")
    conn.commit()
    conn.close()

_ensure_db()

def _append_sqlite(trade):
    try:
        from utils import load_config
        cfg = load_config("config.yaml")
        if not ((cfg.get("logging") or {}).get("sqlite", False)):
            return
    except Exception:
        pass
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trades(symbol,side,entry_price,exit_price,entry_time,exit_time,pnl) VALUES (?,?,?,?,?,?,?)",
        (trade.symbol, trade.side, float(trade.entry_price), float(trade.exit_price),
         trade.entry_time.isoformat(), trade.exit_time.isoformat(), float(trade.pnl))
    )
    conn.commit()
    conn.close()

import os, json
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
STATE_PATH = os.path.join(DATA_DIR, "state.json")

def _serialize_trade(tr):
    return {
        "symbol": tr.symbol,
        "side": tr.side,
        "entry_price": tr.entry_price,
        "exit_price": tr.exit_price,
        "entry_time": tr.entry_time.isoformat(),
        "exit_time": tr.exit_time.isoformat(),
        "pnl": tr.pnl,
    }

def _deserialize_trade(d):
    from datetime import datetime
    return Trade(
        symbol=d["symbol"],
        side=d["side"],
        entry_price=float(d["entry_price"]),
        exit_price=float(d["exit_price"]),
        entry_time=datetime.fromisoformat(d["entry_time"]),
        exit_time=datetime.fromisoformat(d["exit_time"]),
        pnl=float(d["pnl"]),
    )

def _serialize_position(p):
    return {
        "symbol": p.symbol,
        "qty": p.qty,
        "entry_price": p.entry_price,
        "entry_time": p.entry_time.isoformat(),
        "last_skim_anchor": p.last_skim_anchor,
        "high_watermark": p.high_watermark,
        "skim_count": p.skim_count,
    }

def _deserialize_position(d):
    from datetime import datetime
    pos = Position(
        symbol=d["symbol"],
        qty=float(d["qty"]),
        entry_price=float(d["entry_price"]),
        entry_time=datetime.fromisoformat(d["entry_time"]),
    )
    pos.last_skim_anchor = d.get("last_skim_anchor")
    pos.high_watermark = d.get("high_watermark")
    pos.skim_count = int(d.get("skim_count", 0))
    return pos

def save_state(state):
    payload = {
        "open_positions": {k: _serialize_position(v) for k, v in state.open_positions.items()},
        "trades": [_serialize_trade(t) for t in state.trades],
        "stats": state.stats,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_state():
    st = PortfolioState()
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.open_positions = {k: _deserialize_position(v) for k, v in data.get("open_positions", {}).items()}
        st.trades = [_deserialize_trade(x) for x in data.get("trades", [])]
        st.stats = data.get("stats", {})
    return st

def _append_csv(trade):
    # trade: Trade
    header = ["symbol","side","entry_price","exit_price","entry_time","exit_time","pnl"]
    exists = os.path.exists(CSV_TRADES)
    with open(CSV_TRADES, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([trade.symbol, trade.side, f"{trade.entry_price:.8f}", f"{trade.exit_price:.8f}", trade.entry_time.isoformat(), trade.exit_time.isoformat(), f"{trade.pnl:.8f}"])

from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    last_skim_anchor: float = None  # reference price/anchor for next skim level
    high_watermark: float = None   # highest seen price since entry
    skim_count: int = 0
    dca_steps: int = 0

    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float

@dataclass
class PortfolioState:
    stats: dict = field(default_factory=dict)

    # symbol -> Position
    open_positions: Dict[str, Position] = field(default_factory=dict)
    # list of Trade
    trades: List[Trade] = field(default_factory=list)

    def open_position(self, symbol: str, qty: float, price: float, t: datetime):
        p = Position(symbol, qty, price, t)
        p.high_watermark = price
        self.open_positions[symbol] = p
        # update entries count
        today = datetime.utcnow().date().isoformat()
        if self.stats.get('today') != today:
            self.stats = {'today': today, 'entries': 0}
        self.stats['entries'] = int(self.stats.get('entries', 0)) + 1
        save_state(self)

    def close_position(self, symbol: str, price: float, t: datetime):
        pos = self.open_positions.get(symbol)
        if not pos:
            return None
        pnl = (price - pos.entry_price) * pos.qty
        tr = Trade(
            symbol=symbol, side="sell" if pnl>=0 else "sell",
            entry_price=pos.entry_price, exit_price=price,
            entry_time=pos.entry_time, exit_time=t, pnl=pnl
        )
        self.trades.append(tr)
        _append_csv(tr)
        _append_sqlite(tr)
        del self.open_positions[symbol]
        save_state(self)
        return tr


    def partial_close(self, symbol: str, fraction: float, price: float, t: datetime):
        pos = self.open_positions.get(symbol)
        if not pos or fraction <= 0 or fraction >= 1:
            return None
        qty_to_sell = pos.qty * fraction
        pos.qty -= qty_to_sell
        pnl = (price - pos.entry_price) * qty_to_sell
        tr = Trade(
            symbol=symbol,
            side="partial-sell",
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=t,
            pnl=pnl
        )
        self.trades.append(tr)
        _append_csv(tr)
        _append_sqlite(tr)
        if pos.qty <= 1e-9:
            del self.open_positions[symbol]
        save_state(self)
        return tr
