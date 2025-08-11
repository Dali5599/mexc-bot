
import os, csv
from datetime import datetime
from collections import defaultdict

LOGS = os.path.join(os.path.dirname(__file__), "logs")
CSV_TRADES = os.path.join(LOGS, "trades.csv")

def _read_trades():
    rows = []
    if not os.path.exists(CSV_TRADES):
        return rows
    with open(CSV_TRADES, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["entry_time"] = datetime.fromisoformat(row["entry_time"])
                row["exit_time"] = datetime.fromisoformat(row["exit_time"])
                row["entry_price"] = float(row["entry_price"])
                row["exit_price"] = float(row["exit_price"])
                row["pnl"] = float(row["pnl"])
                rows.append(row)
            except Exception:
                continue
    return rows

def daily_pnl():
    rows = _read_trades()
    d = defaultdict(float)
    for tr in rows:
        day = tr["exit_time"].date().isoformat()
        d[day] += tr["pnl"]
    # return as sorted list of (date, pnl)
    return sorted(d.items(), key=lambda x: x[0])

def weekly_pnl():
    rows = _read_trades()
    d = defaultdict(float)
    for tr in rows:
        year, week, wd = tr["exit_time"].isocalendar()
        key = f"{year}-W{week:02d}"
        d[key] += tr["pnl"]
    return sorted(d.items(), key=lambda x: x[0])

def per_symbol_stats():
    rows = _read_trades()
    stats = {}
    by_sym = defaultdict(list)
    for tr in rows:
        by_sym[tr["symbol"]].append(tr)
    for sym, arr in by_sym.items():
        pnl = sum(x["pnl"] for x in arr)
        wins = sum(1 for x in arr if x["pnl"] >= 0)
        losses = len(arr) - wins
        winrate = (wins / len(arr) * 100.0) if arr else 0.0
        avg_pnl = pnl / len(arr) if arr else 0.0
        stats[sym] = {"trades": len(arr), "pnl": pnl, "wins": wins, "losses": losses, "winrate": winrate, "avg_pnl": avg_pnl}
    return stats

def ensure_charts():
    import matplotlib.pyplot as plt
    import os
    static_gen = os.path.join(os.path.dirname(__file__), "static", "gen")
    os.makedirs(static_gen, exist_ok=True)

    # Daily chart
    d = daily_pnl()
    if d:
        xs = [x for x, _ in d]
        ys = [v for _, v in d]
        plt.figure()
        plt.plot(range(len(xs)), ys)  # no colors specified
        plt.title("Daily Realized PnL")
        plt.xlabel("Day")
        plt.ylabel("PnL (USDT)")
        plt.tight_layout()
        plt.savefig(os.path.join(static_gen, "daily.png"))
        plt.close()

    # Weekly chart
    w = weekly_pnl()
    if w:
        xs = [x for x, _ in w]
        ys = [v for _, v in w]
        plt.figure()
        plt.plot(range(len(xs)), ys)
        plt.title("Weekly Realized PnL")
        plt.xlabel("Week")
        plt.ylabel("PnL (USDT)")
        plt.tight_layout()
        plt.savefig(os.path.join(static_gen, "weekly.png"))
        plt.close()


def equity_curve(start_equity=1000.0):
    rows = _read_trades()
    rows = sorted(rows, key=lambda r: r["exit_time"])
    eq = []
    eq_val = float(start_equity)
    for r in rows:
        eq_val += r["pnl"]
        eq.append((r["exit_time"], eq_val))
    return eq

def ensure_equity_chart(start_equity=1000.0):
    import matplotlib.pyplot as plt
    import os
    static_gen = os.path.join(os.path.dirname(__file__), "static", "gen")
    os.makedirs(static_gen, exist_ok=True)
    eq = equity_curve(start_equity)
    if not eq:
        return None
    xs = list(range(len(eq)))
    ys = [v for _, v in eq]
    plt.figure()
    plt.plot(xs, ys)
    plt.title("Equity Curve")
    plt.xlabel("Trades")
    plt.ylabel("Equity (USDT)")
    plt.tight_layout()
    out = os.path.join(static_gen, "equity.png")
    plt.savefig(out)
    plt.close()
    return out

def latest_week_key():
    rows = _read_trades()
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["exit_time"])
    y, w, _ = rows[-1]["exit_time"].isocalendar()
    return f"{y}-W{w:02d}"

def generate_weekly_pdf(week_key=None, start_equity=1000.0):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    import os
    static_gen = os.path.join(os.path.dirname(__file__), "static", "gen")
    os.makedirs(static_gen, exist_ok=True)

    # compute stats for the selected week
    rows = _read_trades()
    if not rows:
        outpdf = os.path.join(static_gen, "weekly_report.pdf")
        c = canvas.Canvas(outpdf, pagesize=A4)
        c.drawString(2*cm, 27*cm, "No trades available.")
        c.save()
        return outpdf

    if week_key is None:
        week_key = latest_week_key()

    # filter rows by week_key
    wk_rows = []
    total = 0.0
    wins = losses = 0
    for r in rows:
        y, w, _ = r["exit_time"].isocalendar()
        key = f"{y}-W{w:02d}"
        if key == week_key:
            wk_rows.append(r)
            total += r["pnl"]
            if r["pnl"] >= 0: wins += 1
            else: losses += 1

    # ensure charts exist
    ensure_charts()
    ensure_equity_chart(start_equity=start_equity)

    # produce PDF
    width, height = A4
    outpdf = os.path.join(static_gen, "weekly_report.pdf")
    c = canvas.Canvas(outpdf, pagesize=A4)
    c.setTitle(f"Weekly Report {week_key}")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, 28*cm, f"Weekly Report: {week_key}")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, 26.8*cm, f"Trades: {len(wk_rows)}  Wins: {wins}  Losses: {losses}  PnL: {total:.2f} USDT")

    def _img(path, x, y, w):
      try:
        from reportlab.lib.utils import ImageReader
        img = ImageReader(path)
        iw, ih = img.getSize()
        h = (w/iw)*ih
        c.drawImage(img, x, y, width=w, height=h)
        return h
      except Exception:
        return 0

    y_cursor = 25.5*cm
    y_cursor -= _img(os.path.join(static_gen, "daily.png"), 2*cm, y_cursor-6*cm, 8*cm) + 0.5*cm
    y_cursor -= _img(os.path.join(static_gen, "weekly.png"), 2*cm, y_cursor-6*cm, 8*cm) + 0.5*cm
    y_cursor -= _img(os.path.join(static_gen, "equity.png"), 2*cm, y_cursor-6*cm, 8*cm) + 0.5*cm

    # list top 10 trades for the week
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, max(2*cm, y_cursor), "Top 10 Trades (by PnL)")
    top = sorted(wk_rows, key=lambda r: r["pnl"], reverse=True)[:10]
    c.setFont("Helvetica", 10)
    y = max(1.5*cm, y_cursor - 0.7*cm)
    for r in top:
        line = f"{r['symbol']}  pnl={r['pnl']:.2f}  exit={r['exit_time'].strftime('%Y-%m-%d %H:%M')}"
        if y < 1*cm: break
        c.drawString(2*cm, y, line)
        y -= 0.5*cm

    c.showPage()
    c.save()
    return outpdf

def _filter_by_range(rows, start_dt=None, end_dt=None):
    if start_dt is None and end_dt is None:
        return rows
    out = []
    for r in rows:
        t = r["exit_time"]
        if start_dt and t < start_dt: continue
        if end_dt and t > end_dt: continue
        out.append(r)
    return out

def ensure_charts_range(start_dt=None, end_dt=None, start_equity=1000.0):
    import matplotlib.pyplot as plt
    import os
    rows = _read_trades()
    rows = sorted(rows, key=lambda r: r['exit_time'])
    rows = _filter_by_range(rows, start_dt, end_dt)
    static_gen = os.path.join(os.path.dirname(__file__), "static", "gen")
    os.makedirs(static_gen, exist_ok=True)

    # daily pnl within range
    from collections import defaultdict
    d = defaultdict(float)
    for tr in rows:
        d[tr["exit_time"].date().isoformat()] += tr["pnl"]
    if d:
        xs = list(sorted(d.keys()))
        ys = [d[k] for k in xs]
        plt.figure()
        plt.plot(range(len(xs)), ys)
        plt.title("Daily PnL (Range)")
        plt.xlabel("Day")
        plt.ylabel("PnL (USDT)")
        plt.tight_layout()
        plt.savefig(os.path.join(static_gen, "daily_range.png"))
        plt.close()

    # weekly pnl within range
    w = defaultdict(float)
    for tr in rows:
        y, wk, _ = tr["exit_time"].isocalendar()
        key = f"{y}-W{wk:02d}"
        w[key] += tr["pnl"]
    if w:
        xs = list(sorted(w.keys()))
        ys = [w[k] for k in xs]
        plt.figure()
        plt.plot(range(len(xs)), ys)
        plt.title("Weekly PnL (Range)")
        plt.xlabel("Week")
        plt.ylabel("PnL (USDT)")
        plt.tight_layout()
        plt.savefig(os.path.join(static_gen, "weekly_range.png"))
        plt.close()

    # equity curve within range
    eq_val = float(start_equity)
    xs = []
    ys = []
    for tr in rows:
        eq_val += tr["pnl"]
        xs.append(tr["exit_time"])
        ys.append(eq_val)
    if xs:
        plt.figure()
        plt.plot(range(len(xs)), ys)
        plt.title("Equity Curve (Range)")
        plt.xlabel("Trades")
        plt.ylabel("Equity (USDT)")
        plt.tight_layout()
        plt.savefig(os.path.join(static_gen, "equity_range.png"))
        plt.close()
