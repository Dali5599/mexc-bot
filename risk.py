import math

def position_size_usdt(balance_usdt: float, risk_per_trade_pct: float, stop_loss_pct: float):
    # مخاطرة نقدية = رصيد * نسبة المخاطرة
    cash_risk = balance_usdt * (risk_per_trade_pct / 100.0)
    if stop_loss_pct <= 0:
        stop_loss_pct = 0.5
    # قيمة الصفقة = المخاطرة / نسبة SL
    notional = cash_risk / (stop_loss_pct / 100.0)
    return max(10.0, notional)  # حد أدنى 10 USDT
