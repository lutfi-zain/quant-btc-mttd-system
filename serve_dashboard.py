#!/usr/bin/env python3
"""MTTD v2 Dashboard — serves pre-computed data + dashboard.html"""

import json, os, sys, numpy as np, pandas as pd
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Custom JSON encoder for numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

print("[1/3] Computing strategy data...")
from multi_principle_strategy import multi_principle_strategy

with open('data/btc_daily.json') as f:
    btc_data = json.load(f)
df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

config = {'t_entry': 0.25, 'er_entry': 0.20, 'entropy_thresh': 2.3,
          'min_hold_days': 10, 'max_hold_days': 60,
          'chikou_thresh': -0.30, 'immunity_thresh': 0.50,
          'imo_min_limit': -0.30, 'imo_exit_bull': -0.30,
          'roc_gate_limit': -0.20, 'cooldown': 5,
          'confirm_entry': 2, 'confirm_exit': 1}

result = multi_principle_strategy(df.copy(), **config)
pos = result['Pos']
daily_ret = pos.shift(1) * df['close'].pct_change() - 0.001 * pos.diff().abs() / 2
daily_ret = daily_ret.dropna()
equity = (1 + daily_ret).cumprod()

# Trades
trades = []
in_pos = False; entry_date = None; entry_price = None
for i, (date, p) in enumerate(pos.items()):
    if p == 1.0 and not in_pos:
        in_pos = True; entry_date = date; entry_price = float(df['close'].loc[date])
    elif p == 0.0 and in_pos:
        in_pos = False
        exit_price = float(df['close'].loc[date])
        ret = (exit_price / entry_price - 1) * 100
        hold = (date - entry_date).days
        trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                       'ret': round(ret, 1), 'hold': hold, 'is_win': bool(ret > 0)})

# Metrics
total_ret = float((equity.iloc[-1] - 1) * 100)
years = len(daily_ret) / 365.25
cagr = float((equity.iloc[-1] ** (1/years) - 1) * 100)
sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))
peak = equity.expanding().max()
max_dd = float(((equity - peak) / peak * 100).min())
n_win = sum(1 for t in trades if t['is_win'])
win_rate = n_win / len(trades) * 100
avg_hold = int(np.mean([t['hold'] for t in trades]))
avg_win = float(np.mean([t['ret'] for t in trades if t['is_win']]) if n_win else 0)
avg_loss = float(np.mean([t['ret'] for t in trades if not t['is_win']]) if len(trades)-n_win else 0)

# Equity curve (sampled)
step = max(1, len(equity) // 300)
eq_data = [{'date': str(equity.index[i].date()), 'value': round(float(equity.iloc[i]), 2)} for i in range(0, len(equity), step)]
if eq_data[-1]['date'] != str(equity.index[-1].date()):
    eq_data.append({'date': str(equity.index[-1].date()), 'value': round(float(equity.iloc[-1]), 2)})

# Drawdown
dd_data = [{'date': str(equity.index[i].date()), 'dd': round(float(((equity.iloc[i]/peak.iloc[i])-1)*100), 1)} for i in range(0, len(equity), step)]
if dd_data[-1]['date'] != str(equity.index[-1].date()):
    dd_data.append({'date': str(equity.index[-1].date()), 'dd': round(float(((equity.iloc[-1]/peak.iloc[-1])-1)*100), 1)})

# Monthly returns
monthly_ret = df['close'].resample('ME').last().pct_change().dropna() * 100
monthly = [{'month': str(k.date())[:7], 'ret': round(float(v), 1)} for k, v in monthly_ret.items()]

data = {
    'metrics': {
        'total_return': round(total_ret, 1), 'cagr': round(cagr, 1),
        'sharpe': round(sharpe, 2), 'win_rate': round(win_rate, 1),
        'max_dd': round(max_dd, 1), 'avg_hold': avg_hold,
        'trades': len(trades), 'wins': n_win, 'losses': len(trades) - n_win,
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'final_equity': round(float(equity.iloc[-1]), 2)
    },
    'trades': trades, 'equity': eq_data, 'drawdown': dd_data, 'monthly': monthly
}

os.makedirs('dashboard/data', exist_ok=True)
with open('dashboard/data/dashboard.json', 'w') as f:
    json.dump(data, f, indent=2, cls=NpEncoder)

print(f"  {len(trades)} trades, {win_rate:.0f}% win, Sharpe {sharpe:.2f}, CAGR {cagr:.0f}%")
print(f"  Data saved to dashboard/data/dashboard.json")

# Server
PORT = 8081
HOST = '0.0.0.0'
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

print(f"\nDashboard: http://localhost:{PORT}/dashboard.html")
print(f"API:       http://localhost:{PORT}/dashboard/data/dashboard.json")
print("Press Ctrl+C to stop")

with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
