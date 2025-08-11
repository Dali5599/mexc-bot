
import os, json, threading, time
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import load_config
from exchange import MexcSpot
from ui_controller import BotRunner
from backtest import run_backtest
from portfolio import PortfolioState, load_state, save_state
STATE = load_state()
LIMITS_STATUS = {"blocked": False, "reason": ""}

BASE_DIR = os.path.dirname(__file__)
templates_dir = os.path.join(BASE_DIR, "templates")
locales_dir = os.path.join(BASE_DIR, "locales")

env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=select_autoescape()
)

def load_locale(lang):
    fname = {"ar":"ar.json","fr":"fr.json"}.get(lang,"en.json")
    with open(os.path.join(locales_dir, fname), "r", encoding="utf-8") as f:
        return json.load(f)

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR,"static")), name="static")


def bot_loop(stop_event):
    from performance import daily_pnl
    from datetime import datetime as _dt
    import pandas as pd
    from datetime import datetime
    from utils import log
    from strategy_rsi_ema import generate_signals as strat_rsi_ema
from strategy_breakout import generate_signals as strat_breakout
from strategy_vwap import generate_signals as strat_vwap
from strategy_grid import generate_signals as strat_grid
    from risk import position_size_usdt
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    # Multi-markets: prefer list "markets", else fallback to single "market"
    markets = cfg.get("markets") or [cfg["market"]]
    # compute unrealized PnL for open positions
    unrealized = {}
    try:
        for sym, pos in STATE.open_positions.items():
            data_last = ex.fetch_ohlcv(sym, timeframe=cfg["timeframe"], limit=1)
            px = float(data_last[-1][4]) if data_last else pos.entry_price
            upnl = (px - pos.entry_price) * pos.qty
            unrealized[sym] = {"price": px, "uPnL": upnl}
    except Exception:
        pass
    timeframe = cfg["timeframe"]
    
    # Limits from config
    limits = cfg.get("limits", {}) or {}
    max_trades = int(limits.get("max_trades_per_day", 0) or 0)
    max_loss = float(limits.get("daily_max_loss_usdt", 0) or 0.0)
    def _limits_ok():
        # realized pnl today
        try:
            pnl_daily = dict(daily_pnl())
            today = _dt.utcnow().date().isoformat()
            realized = float(pnl_daily.get(today, 0.0))
        except Exception:
            realized = 0.0
        # entries today
        stats = getattr(STATE, "stats", {}) or {}
        entries = int(stats.get("entries", 0)) if stats.get("today") == _dt.utcnow().date().isoformat() else 0
        # checks
        if max_loss > 0 and realized <= -abs(max_loss):
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Daily loss limit reached: {realized:.2f} USDT"
            return False
        if max_trades > 0 and entries >= max_trades:
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Max trades per day reached: {entries}"
            return False
        LIMITS_STATUS["blocked"] = False
        LIMITS_STATUS["reason"] = ""
        return True

    ex = MexcSpot()
    # Simple per-symbol balance share for sizing (simulation-friendly)
    try:
        bal = ex.fetch_balance()
        usdt_total = float(bal.get("total", {}).get("USDT", 1000.0))
    except Exception:
        usdt_total = 1000.0
    # Distribute risk across symbols
    per_symbol_usdt = max(10.0, usdt_total / max(1, len(markets)))
    while not stop_event.is_set():
        try:
            for symbol in markets:
                # --- Liquidity & Spread filters ---
                try:
                    ticker = ex.exchange.fetch_ticker(symbol)
                    vol_ok = True
                    liq = cfg.get('liquidity', {}) or {}
                    min_vol = float(liq.get('min_24h_volume_usdt', 0))
                    if min_vol > 0 and ticker.get('quoteVolume') is not None:
                        vol_ok = float(ticker['quoteVolume']) >= min_vol
                    ob = ex.exchange.fetch_order_book(symbol, limit=50)
                    best_bid = ob['bids'][0][0] if ob['bids'] else None
                    best_ask = ob['asks'][0][0] if ob['asks'] else None
                    if best_bid and best_ask:
                        spread_bps_ob = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 10000
                    else:
                        spread_bps_ob = 99999
                    depth_usdt = 0.0
                    # sum top 10 levels approx as depth
                    for px, qty in (ob['bids'][:10] if ob['bids'] else []):
                        depth_usdt += px*qty
                    for px, qty in (ob['asks'][:10] if ob['asks'] else []):
                        depth_usdt += px*qty
                    min_depth = float(liq.get('min_orderbook_depth_usdt', 0))
                    depth_ok = (min_depth == 0) or (depth_usdt >= min_depth)
                except Exception:
                    vol_ok = False; depth_ok = False; spread_bps_ob = 99999
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
                import pandas as pd
                df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                stype = (cfg.get('strategy', {}) or {}).get('type', 'rsi_ema')
                if stype == 'breakout':
                    df = strat_breakout(df, lookback=cfg.get('breakout',{}).get('lookback',50), buffer_bps=cfg.get('breakout',{}).get('buffer_bps',10))
                elif stype == 'vwap':
                    df = strat_vwap(df, window=cfg.get('vwap',{}).get('window',30), band_bps=cfg.get('vwap',{}).get('band_bps',20))
                elif stype == 'grid':
                    df = strat_grid(df, step_pct=cfg.get('grid',{}).get('step_pct',0.6), band_mult=cfg.get('grid',{}).get('band_mult',1.0), ema_slow=cfg['strategy']['ema_slow'])
                else:
                    df = strat_rsi_ema(df, cfg['strategy']['ema_fast'], cfg['strategy']['ema_slow'], cfg['strategy']['rsi_period'], cfg['strategy']['rsi_buy'], cfg['strategy']['rsi_sell'])
                last = df.iloc[-1]
                price = float(last["close"])
                # Entry condition
                dca_cfg = cfg.get('dca', {}) or {}
                dca_enabled = bool(dca_cfg.get('enabled', False))
                if last["buy_signal"] and symbol not in STATE.open_positions and _limits_ok():
                    # Initial entry handled below
                    notional = position_size_usdt(per_symbol_usdt, cfg["risk"]["risk_per_trade_pct"], cfg["risk"]["stop_loss_pct"])
                    qty = round(notional / price, 6)
                    if qty > 0:
                        ex.place_order(symbol, "buy", qty)
                        STATE.open_position(symbol, qty, price, last["ts"].to_pydatetime())
                # Exit condition: sell_signal OR TP/SL
                else:
                    pos = STATE.open_positions.get(symbol)
                    if pos:
                        # DCA scale-in on pullback
                        if dca_enabled:
                            add_steps = int(dca_cfg.get('add_steps', 0))
                            add_drop = float(dca_cfg.get('add_pct_drop', 1.0))/100.0
                            add_frac = float(dca_cfg.get('add_fraction_pct', 20.0))/100.0
                            if pos.dca_steps < add_steps and price <= pos.last_skim_anchor*(1 - add_drop):
                                add_qty = round(pos.qty * add_frac, 6)
                                if add_qty > 0 and _limits_ok():
                                    ex.place_order(symbol, 'buy', add_qty)
                                    # adjust average entry price (simple weighted)
                                    new_qty = pos.qty + add_qty
                                    pos.entry_price = (pos.entry_price*pos.qty + price*add_qty)/new_qty
                                    pos.qty = new_qty
                                    pos.last_skim_anchor = price
                                    pos.dca_steps += 1
                        # Time-based stop
                        tstop = int(cfg['risk'].get('time_stop_minutes', 0)) if isinstance(cfg.get('risk'), dict) else 0
                        if tstop and (last['ts'].to_pydatetime() - pos.entry_time).total_seconds() >= tstop*60:
                            ex.place_order(symbol, 'sell', pos.qty)
                            STATE.close_position(symbol, price, last['ts'].to_pydatetime())
                            continue
                        # --- Profit Skim Logic ---
                        skim_cfg = cfg.get("skim", {}) or {}
                        skim_enabled = skim_cfg.get("enabled", False)
                        if skim_enabled:
                            step_pct = float(skim_cfg.get("step_pct", 0.8)) / 100.0
                            frac_pct = float(skim_cfg.get("fraction_pct", 20.0)) / 100.0
                            trailing = bool(skim_cfg.get("trailing", True))
                            max_steps = int(skim_cfg.get("max_steps", 5))
                            # Init anchors
                            if pos.last_skim_anchor is None:
                                pos.last_skim_anchor = pos.entry_price
                            if pos.high_watermark is None:
                                pos.high_watermark = pos.entry_price
                            # Update high watermark
                            if price > pos.high_watermark:
                                pos.high_watermark = price
                            # Choose anchor for next level
                            anchor = pos.high_watermark if trailing else pos.last_skim_anchor
                            next_level = anchor * (1 + step_pct)
                            if price >= next_level and pos.qty > 0 and (pos.skim_count < max_steps):
                                # partial close amount
                                sell_qty = round(pos.qty * frac_pct, 6)
                                if sell_qty > 0:
                                    ex.place_order(symbol, "sell", sell_qty)
                                    from datetime import datetime as _dt
                                    STATE.partial_close(symbol, frac_pct, price, _dt.utcnow())
                                    pos = STATE.open_positions.get(symbol)
                                    if pos:
                        # Time-based stop
                        tstop = int(cfg['risk'].get('time_stop_minutes', 0)) if isinstance(cfg.get('risk'), dict) else 0
                        if tstop and (last['ts'].to_pydatetime() - pos.entry_time).total_seconds() >= tstop*60:
                            ex.place_order(symbol, 'sell', pos.qty)
                            STATE.close_position(symbol, price, last['ts'].to_pydatetime())
                            continue
                                        pos.last_skim_anchor = price
                                        pos.skim_count += 1

                        entry = pos.entry_price
                        sl = entry * (1 - cfg["risk"]["stop_loss_pct"]/100.0)
                        tp = entry * (1 + cfg["risk"]["take_profit_pct"]/100.0)
                        should_exit = bool(last["sell_signal"] or price <= sl or price >= tp)
                        if should_exit:
                            ex.place_order(symbol, "sell", pos.qty)
                            STATE.close_position(symbol, price, last["ts"].to_pydatetime())
            # small sleep between cycles
            import time as _t; _t.sleep(5)
        except Exception:
            import time as _t; _t.sleep(3)

    # Simple loop that mimics bot.py main loop but listens to stop_event
    import pandas as pd
    from utils import log
    from strategy_rsi_ema import generate_signals as strat_rsi_ema
from strategy_breakout import generate_signals as strat_breakout
from strategy_vwap import generate_signals as strat_vwap
from strategy_grid import generate_signals as strat_grid
    from risk import position_size_usdt
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    symbol = os.getenv("MARKET", cfg["market"])
    timeframe = cfg["timeframe"]
    
    # Limits from config
    limits = cfg.get("limits", {}) or {}
    max_trades = int(limits.get("max_trades_per_day", 0) or 0)
    max_loss = float(limits.get("daily_max_loss_usdt", 0) or 0.0)
    def _limits_ok():
        # realized pnl today
        try:
            pnl_daily = dict(daily_pnl())
            today = _dt.utcnow().date().isoformat()
            realized = float(pnl_daily.get(today, 0.0))
        except Exception:
            realized = 0.0
        # entries today
        stats = getattr(STATE, "stats", {}) or {}
        entries = int(stats.get("entries", 0)) if stats.get("today") == _dt.utcnow().date().isoformat() else 0
        # checks
        if max_loss > 0 and realized <= -abs(max_loss):
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Daily loss limit reached: {realized:.2f} USDT"
            return False
        if max_trades > 0 and entries >= max_trades:
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Max trades per day reached: {entries}"
            return False
        LIMITS_STATUS["blocked"] = False
        LIMITS_STATUS["reason"] = ""
        return True

    ex = MexcSpot()
    while not stop_event.is_set():
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = generate_signals(df, cfg["strategy"]["ema_fast"], cfg["strategy"]["ema_slow"],
                                  cfg["strategy"]["rsi_period"], cfg["strategy"]["rsi_buy"], cfg["strategy"]["rsi_sell"])
            time.sleep(8)  # keep light
        except Exception as e:
            time.sleep(5)

runner = BotRunner(bot_loop)


def sparkline_svg(values, width=180, height=40, padding=4):
    if not values: 
        return '<svg width="%d" height="%d"></svg>' % (width, height)
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1e-9
    pts = []
    n = len(values)
    inner_w = width - 2*padding
    inner_h = height - 2*padding
    for i, v in enumerate(values):
        x = padding + inner_w * (i / (n-1 if n>1 else 1))
        y = padding + inner_h * (1 - (v - lo) / rng)
        pts.append(f"{x:.2f},{y:.2f}")
    path = " ".join(pts)
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline fill="none" stroke="currentColor" stroke-width="2" points="{path}" /></svg>'

def get_lang(request: Request):

    lang = request.query_params.get("lang") or request.cookies.get("lang") or "ar"
    return lang

def render(request: Request, tpl, context):
    lang = context.get("lang") or get_lang(request)
    t = load_locale(lang)
    template = env.get_template(tpl)
    res = template.render(**context, t=t, lang=lang, now=datetime.utcnow())
    response = HTMLResponse(res)
    response.set_cookie("lang", lang, max_age=86400*30, path="/")
    return response

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    from statistics import mean
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    
    # Limits from config
    limits = cfg.get("limits", {}) or {}
    max_trades = int(limits.get("max_trades_per_day", 0) or 0)
    max_loss = float(limits.get("daily_max_loss_usdt", 0) or 0.0)
    def _limits_ok():
        # realized pnl today
        try:
            pnl_daily = dict(daily_pnl())
            today = _dt.utcnow().date().isoformat()
            realized = float(pnl_daily.get(today, 0.0))
        except Exception:
            realized = 0.0
        # entries today
        stats = getattr(STATE, "stats", {}) or {}
        entries = int(stats.get("entries", 0)) if stats.get("today") == _dt.utcnow().date().isoformat() else 0
        # checks
        if max_loss > 0 and realized <= -abs(max_loss):
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Daily loss limit reached: {realized:.2f} USDT"
            return False
        if max_trades > 0 and entries >= max_trades:
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Max trades per day reached: {entries}"
            return False
        LIMITS_STATUS["blocked"] = False
        LIMITS_STATUS["reason"] = ""
        return True

    ex = MexcSpot()
    try:
        bal = ex.fetch_balance()
        balance = round(bal.get("total", {}).get("USDT", 0.0), 2)
    except Exception:
        balance = 0.0
    live = os.getenv("LIVE_TRADING","False").lower()=="true"
    return render(request, "index.html", {"status": runner.status, "cfg": cfg, "balance": balance, "live": live, "lang": lang})

@app.post("/start")
def start_bot(request: Request):
    runner.start()
    lang = get_lang(request)
    return RedirectResponse(url=f"/?lang={lang}", status_code=303)

@app.post("/stop")
def stop_bot(request: Request):
    runner.stop()
    lang = get_lang(request)
    return RedirectResponse(url=f"/?lang={lang}", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
def get_settings(request: Request):
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    return render(request, "settings.html", {"cfg": cfg, "saved": False, "lang": lang})

@app.post("/settings")
async def post_settings(request: Request,
    market: str = Form(...),
    timeframe: str = Form(...),
    max_spread_bps: int = Form(...),
    ema_fast: int = Form(...),
    ema_slow: int = Form(...),
    rsi_period: int = Form(...),
    rsi_buy: int = Form(...),
    rsi_sell: int = Form(...),
    markets: str = Form(None),
    risk_per_trade_pct: float = Form(...),
    stop_loss_pct: float = Form(...),
    take_profit_pct: float = Form(...),
):
    import yaml
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    cfg["market"] = market
    cfg["timeframe"] = timeframe
    if markets:
        ms = [m.strip() for m in markets.split(",") if m.strip()]
        cfg["markets"] = ms
    cfg["max_spread_bps"] = int(max_spread_bps)
    cfg["strategy"]["ema_fast"] = int(ema_fast)
    cfg["strategy"]["ema_slow"] = int(ema_slow)
    cfg["strategy"]["rsi_period"] = int(rsi_period)
    cfg["strategy"]["rsi_buy"] = int(rsi_buy)
    cfg["strategy"]["rsi_sell"] = int(rsi_sell)
    cfg["risk"]["risk_per_trade_pct"] = float(risk_per_trade_pct)
    cfg["risk"]["stop_loss_pct"] = float(stop_loss_pct)
    cfg["risk"]["take_profit_pct"] = float(take_profit_pct)
        # Strategy type & params
    if strategy_type: cfg['strategy']['type'] = strategy_type
    if breakout_lookback is not None: cfg['breakout']['lookback'] = int(breakout_lookback)
    if breakout_buffer_bps is not None: cfg['breakout']['buffer_bps'] = int(breakout_buffer_bps)
    if vwap_window is not None: cfg['vwap']['window'] = int(vwap_window)
    if vwap_band_bps is not None: cfg['vwap']['band_bps'] = int(vwap_band_bps)
    if grid_step_pct is not None: cfg['grid']['step_pct'] = float(grid_step_pct)
    if grid_band_mult is not None: cfg['grid']['band_mult'] = float(grid_band_mult)
    with open('config.yaml','w',encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    lang = get_lang(request)
    return RedirectResponse(url=f"/settings?lang={lang}", status_code=303)

@app.get("/backtest", response_class=HTMLResponse)
def get_backtest(request: Request):
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    return render(request, "backtest.html", {"cfg": cfg, "result": None, "lang": lang})

@app.post("/backtest", response_class=HTMLResponse)
async def do_backtest(request: Request,
    market: str = Form(...),
    timeframe: str = Form(...),
    days: int = Form(...),
):
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    result = run_backtest(market, timeframe, int(days))
    return render(request, "backtest.html", {"cfg": cfg, "result": result, "lang": lang})


@app.post("/close")
def close_position(request: Request, symbol: str = Form(...)):
    # Close the open position at latest market price
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    
    # Limits from config
    limits = cfg.get("limits", {}) or {}
    max_trades = int(limits.get("max_trades_per_day", 0) or 0)
    max_loss = float(limits.get("daily_max_loss_usdt", 0) or 0.0)
    def _limits_ok():
        # realized pnl today
        try:
            pnl_daily = dict(daily_pnl())
            today = _dt.utcnow().date().isoformat()
            realized = float(pnl_daily.get(today, 0.0))
        except Exception:
            realized = 0.0
        # entries today
        stats = getattr(STATE, "stats", {}) or {}
        entries = int(stats.get("entries", 0)) if stats.get("today") == _dt.utcnow().date().isoformat() else 0
        # checks
        if max_loss > 0 and realized <= -abs(max_loss):
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Daily loss limit reached: {realized:.2f} USDT"
            return False
        if max_trades > 0 and entries >= max_trades:
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Max trades per day reached: {entries}"
            return False
        LIMITS_STATUS["blocked"] = False
        LIMITS_STATUS["reason"] = ""
        return True

    ex = MexcSpot()
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=cfg["timeframe"], limit=1)
        price = float(ohlcv[-1][4]) if ohlcv else None
    except Exception:
        price = None
    pos = STATE.open_positions.get(symbol)
    if pos and price:
        ex.place_order(symbol, "sell", pos.qty)
        from datetime import datetime
        STATE.close_position(symbol, price, datetime.utcnow())
    lang = get_lang(request)
    return RedirectResponse(url=f"/?lang={lang}", status_code=303)


@app.post("/close_partial")
def close_partial(request: Request, symbol: str = Form(...), percent: float = Form(...)):
    # Close part of the position by given percent
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    
    # Limits from config
    limits = cfg.get("limits", {}) or {}
    max_trades = int(limits.get("max_trades_per_day", 0) or 0)
    max_loss = float(limits.get("daily_max_loss_usdt", 0) or 0.0)
    def _limits_ok():
        # realized pnl today
        try:
            pnl_daily = dict(daily_pnl())
            today = _dt.utcnow().date().isoformat()
            realized = float(pnl_daily.get(today, 0.0))
        except Exception:
            realized = 0.0
        # entries today
        stats = getattr(STATE, "stats", {}) or {}
        entries = int(stats.get("entries", 0)) if stats.get("today") == _dt.utcnow().date().isoformat() else 0
        # checks
        if max_loss > 0 and realized <= -abs(max_loss):
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Daily loss limit reached: {realized:.2f} USDT"
            return False
        if max_trades > 0 and entries >= max_trades:
            LIMITS_STATUS["blocked"] = True
            LIMITS_STATUS["reason"] = f"Max trades per day reached: {entries}"
            return False
        LIMITS_STATUS["blocked"] = False
        LIMITS_STATUS["reason"] = ""
        return True

    ex = MexcSpot()
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=cfg["timeframe"], limit=1)
        price = float(ohlcv[-1][4]) if ohlcv else None
    except Exception:
        price = None
    pos = STATE.open_positions.get(symbol)
    if pos and price and percent > 0:
        frac = min(max(percent / 100.0, 0.01), 0.99)
        qty = round(pos.qty * frac, 6)
        if qty > 0:
            ex.place_order(symbol, "sell", qty)
            from datetime import datetime
            STATE.partial_close(symbol, frac, price, datetime.utcnow())
    lang = get_lang(request)
    return RedirectResponse(url=f"/?lang={lang}", status_code=303)


@app.get("/export_trades")
def export_trades():
    import os
    p = os.path.join(os.path.dirname(__file__), "logs", "trades.csv")
    if not os.path.exists(p):
        # create empty file with header
        with open(p, "w", encoding="utf-8") as f:
            f.write("symbol,side,entry_price,exit_price,entry_time,exit_time,pnl\n")
    return FileResponse(path=p, filename="trades.csv", media_type="text/csv")


@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    from performance import daily_pnl, weekly_pnl, per_symbol_stats, ensure_charts
    ensure_charts()
    lang = get_lang(request)
    stats_sym = per_symbol_stats()
    return render(request, "performance.html", {"stats_sym": stats_sym, "lang": lang})


@app.get("/weekly_report")
def weekly_report():
    from performance import generate_weekly_pdf
    path = generate_weekly_pdf()
    return FileResponse(path=path, filename="weekly_report.pdf", media_type="application/pdf")


@app.post("/performance", response_class=HTMLResponse)
async def performance_apply(request: Request, start_date: str = Form(None), end_date: str = Form(None), week: str = Form(None)):
    from performance import ensure_charts, ensure_equity_chart, ensure_charts_range
    from datetime import datetime as _dt
    sdt = _dt.fromisoformat(start_date) if start_date else None
    edt = _dt.fromisoformat(end_date) if end_date else None
    ensure_charts()
    ensure_equity_chart()
    ensure_charts_range(sdt, edt)
    lang = get_lang(request)
    return RedirectResponse(url=f"/performance?lang={lang}", status_code=303)

@app.get("/weekly_report")
def weekly_report(week: str = None):
    from performance import generate_weekly_pdf
    path = generate_weekly_pdf(week_key=week)
    return FileResponse(path=path, filename="weekly_report.pdf", media_type="application/pdf")


@app.get("/strategies", response_class=HTMLResponse)
def strategies_page(request: Request):
    lang = get_lang(request)
    return render(request, "strategies.html", {"lang": lang, "request": request})

@app.get("/optimizer", response_class=HTMLResponse)
def optimizer_page(request: Request):
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    # load presets list
    try:
        with open(os.path.join("data","optimizer_presets.json"),"r",encoding="utf-8") as f:
            prs = json.load(f).get("presets", [])
    except Exception:
        prs = []
    return render(request, "optimizer.html", {"cfg": cfg, "lang": lang, "results": None, "presets": prs})

@app.post("/optimizer", response_class=HTMLResponse)
async def optimizer_run(request: Request,
    strategy_type: str = Form(...),
    markets: str = Form(...),
    timeframe: str = Form(...),
    p1: str = Form(None),
    p2: str = Form(None),
    p3: str = Form(None),
    p4: str = Form(None),
):
    import pandas as pd, csv
    from optimizer import run_grid
    cfg = load_config("config.yaml")
    mkts = [m.strip() for m in (markets or '').split(',') if m.strip()]
    def _parse_range(s):
        # format: start:end:step or single value
        arr = []
        if not s: return arr
        try:
            if ":" in s:
                a,b,c = s.split(":")
                a=float(a); b=float(b); c=float(c)
                x=a
                while x <= b+1e-12:
                    # keep int if integer
                    arr.append(int(x) if abs(x-round(x))<1e-9 else round(x,4))
                    x += c
            else:
                v=float(s)
                arr=[int(v) if abs(v-round(v))<1e-9 else v]
        except Exception:
            pass
        return arr
    ranges = {}
    for k,s in [("p1",p1),("p2",p2),("p3",p3),("p4",p4)]:
        vals = _parse_range(s)
        if vals: ranges[k]=vals
    # map p1..p4 to real params by strategy
    if strategy_type=="rsi_ema":
        param_map = {"p1":"ema_fast","p2":"ema_slow","p3":"rsi_buy","p4":"rsi_sell"}
    elif strategy_type=="breakout":
        param_map = {"p1":"lookback","p2":"buffer_bps"}
    elif strategy_type=="vwap":
        param_map = {"p1":"window","p2":"band_bps"}
    else: # grid
        param_map = {"p1":"step_pct","p2":"band_mult"}
    real_ranges = {}
    for k,vals in ranges.items():
        name = param_map.get(k)
        if name: real_ranges[name]=vals
    res = run_grid(strategy_type, mkts, timeframe, real_ranges, cfg)
    # save CSV
    out_csv = os.path.join("logs","optimizer_results.csv")
    os.makedirs("logs", exist_ok=True)
    import csv
    if res:
        keys = sorted({kk for r in res for kk in r.keys()})
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(res)
    lang = get_lang(request)
    try:
        with open(os.path.join("data","optimizer_presets.json"),"r",encoding="utf-8") as f:
            prs = json.load(f).get("presets", [])
    except Exception:
        prs = []
    return render(request, "optimizer.html", {"cfg": cfg, "lang": lang, "results": res, "presets": prs})


@app.post("/optimizer/save")
async def optimizer_save(request: Request, name: str = Form(...), strategy_type: str = Form(...), markets: str = Form(...), timeframe: str = Form(...), p1: str = Form(None), p2: str = Form(None), p3: str = Form(None), p4: str = Form(None)):
    path = os.path.join("data","optimizer_presets.json")
    try:
        with open(path,"r",encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {"presets": []}
    prs = obj.get("presets", [])
    # upsert by name
    payload = {"name": name, "strategy_type": strategy_type, "markets": markets, "timeframe": timeframe, "p1": p1, "p2": p2, "p3": p3, "p4": p4}
    prs = [p for p in prs if p.get("name") != name] + [payload]
    obj["presets"] = prs
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    lang = get_lang(request)
    return RedirectResponse(url=f"/optimizer?lang={lang}", status_code=303)

@app.get("/optimizer/load", response_class=HTMLResponse)
def optimizer_load(request: Request, name: str):
    path = os.path.join("data","optimizer_presets.json")
    try:
        with open(path,"r",encoding="utf-8") as f:
            obj = json.load(f)
        preset = next((p for p in obj.get("presets", []) if p.get("name")==name), None)
    except Exception:
        preset = None
    lang = get_lang(request)
    cfg = load_config("config.yaml")
    return render(request, "optimizer.html", {"cfg": cfg, "lang": lang, "results": None, "preset": preset, "presets": obj.get("presets", []) if 'obj' in locals() else []})


@app.get("/sql", response_class=HTMLResponse)
def sql_page(request: Request):
    lang = get_lang(request)
    return render(request, "sql.html", {"rows": None, "headers": [], "lang": lang})

@app.post("/sql", response_class=HTMLResponse)
async def sql_run(request: Request, symbol: str = Form(None), from_dt: str = Form(None), to_dt: str = Form(None), min_pnl: float = Form(None), max_pnl: float = Form(None)):
    import sqlite3
    conn = sqlite3.connect(os.path.join("data","trades.db"))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = "SELECT symbol, side, entry_price, exit_price, entry_time, exit_time, pnl FROM trades WHERE 1=1"
    params = []
    if symbol:
        q += " AND symbol = ?"; params.append(symbol)
    if from_dt:
        q += " AND exit_time >= ?"; params.append(from_dt)
    if to_dt:
        q += " AND exit_time <= ?"; params.append(to_dt)
    if min_pnl is not None and str(min_pnl) != '':
        q += " AND pnl >= ?"; params.append(float(min_pnl))
    if max_pnl is not None and str(max_pnl) != '':
        q += " AND pnl <= ?"; params.append(float(max_pnl))
    q += " ORDER BY exit_time DESC LIMIT 500"
    cur.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    headers = list(rows[0].keys()) if rows else ["symbol","side","entry_price","exit_price","entry_time","exit_time","pnl"]
    conn.close()
    lang = get_lang(request)
    return render(request, "sql.html", {"rows": rows, "headers": headers, "lang": lang})

@app.post("/sql_raw", response_class=HTMLResponse)
async def sql_raw(request: Request, query: str = Form(...)):
    import sqlite3, re
    q = (query or "").strip()
    # very basic validation: must start with SELECT and disallow dangerous keywords
    if not q.lower().startswith("select"):
        q = "SELECT 'Only SELECT queries allowed' AS message"
    forbidden = ["insert","update","delete","drop","alter","create","attach","detach","pragma","vacuum"]
    if any(k in q.lower() for k in forbidden):
        q = "SELECT 'Forbidden keyword detected' AS error"
    conn = sqlite3.connect(os.path.join("data","trades.db"))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(q)
        rows = [dict(r) for r in cur.fetchall()]
        headers = list(rows[0].keys()) if rows else ["result"]
    except Exception as e:
        rows = [{"error": str(e)}]; headers = ["error"]
    conn.close()
    lang = get_lang(request)
    return render(request, "sql.html", {"rows": rows, "headers": headers, "lang": lang})
