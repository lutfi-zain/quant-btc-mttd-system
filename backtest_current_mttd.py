import os
import sys
import yaml
import re
import importlib.util
import urllib.request
import json
import pandas as pd
import numpy as np

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)

from indicators_helper import *

def fetch_series(series_name, index="day1"):
    url = f"https://bitview.space/api/series/{series_name}/{index}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def load_data():
    ohlc_res = fetch_series("price_ohlc")
    volume_res = fetch_series("transfer_volume_sum_24h_usd")
    date_res = fetch_series("date")
    ohlc_start, ohlc_end = ohlc_res['start'], ohlc_res['start'] + len(ohlc_res['data']) - 1
    vol_start, vol_end = volume_res['start'], volume_res['start'] + len(volume_res['data']) - 1
    date_start, date_end = date_res['start'], date_res['start'] + len(date_res['data']) - 1
    start_idx = max(ohlc_start, vol_start, date_start)
    end_idx = min(ohlc_end, vol_end, date_end)
    aligned_data = []
    for idx in range(start_idx, end_idx + 1):
        date_val = date_res['data'][idx - date_res['start']]
        ohlc_val = ohlc_res['data'][idx - ohlc_res['start']]
        vol_val = volume_res['data'][idx - volume_res['start']]
        aligned_data.append({
            'time': date_val,
            'open': ohlc_val[0],
            'high': ohlc_val[1],
            'low': ohlc_val[2],
            'close': ohlc_val[3],
            'volume': vol_val
        })
    df = pd.DataFrame(aligned_data)
    df.set_index('time', inplace=True)
    return df

def normalize_name(name):
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n

def detect_direction_series(res_df):
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend']:
        if col in res_df.columns:
            return res_df[col]
    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l = res_df['long_signal'].iloc[i]
            s = res_df['short_signal'].iloc[i]
            l_val = bool(l) if not pd.isna(l) else False
            s_val = bool(s) if not pd.isna(s) else False
            if l_val and not s_val:
                curr = 1.0
            elif s_val and not l_val:
                curr = -1.0
            direction.iloc[i] = curr
        return direction
    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction
    for col in res_df.columns:
        col_lower = col.lower()
        if 'direction' in col_lower or 'signal' in col_lower or 'trend' in col_lower:
            unique_vals = res_df[col].dropna().unique()
            if len(unique_vals) <= 10:
                return res_df[col]
    return None

df = load_data()

SELECTED_INDICATORS = [
    {"name": "Adaptive Regime Cloud", "category": "perpetual"},
    {"name": "Adaptive Volatility Controlled LSMA | QuantAlgo", "category": "perpetual"},
    {"name": "Polynomial Deviation Bands", "category": "perpetual"},
    {"name": "alma lag | viResearch", "category": "perpetual"},
    {"name": "lsma | viResearch", "category": "perpetual"},
    {"name": "DSMA | viResearch", "category": "perpetual"},
    {"name": "IRS`Elder Force Volume Index", "category": "perpetual"},
    {"name": "Gaussian Smooth Trend | QuantEdgeB", "category": "perpetual"},
    {"name": "DEGA RMA | QuantEdgeB", "category": "perpetual"},
    {"name": "Linear % ST | QuantEdgeB", "category": "perpetual"},
    {"name": "Quantile DEMA Trend | QuantEdgeB", "category": "perpetual"},
    {"name": "HILO Interpolation | QuantEdgeB", "category": "perpetual"},
    {"name": "MadTrend | InvestorUnknown", "category": "perpetual"},
    {"name": "Median Deviation Suite | InvestorUnknown", "category": "perpetual"},
    {"name": "Root Mean Square Deviation Trend", "category": "perpetual"}
]

sum_sigs = pd.Series(0.0, index=df.index)

for ind in SELECTED_INDICATORS:
    name = ind['name']
    cat = ind['category']
    normalized = normalize_name(name)
    py_file = os.path.join(project_root, cat, f"{normalized}.py")
    spec = importlib.util.spec_from_file_location(normalized, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, normalized)
    res_df = func(df)
    dir_series = detect_direction_series(res_df).reindex(df.index).fillna(0.0)
    sigs = pd.Series([1.0 if val > 0 else -1.0 for val in dir_series], index=df.index)
    sum_sigs += sigs

# Limit evaluation to CSV range
csv_path = "/home/ubuntu/projects/quant-technical-indicator-bank/isp-signals-btcusd-2026-06-13.csv"
df_csv = pd.read_csv(csv_path)
first_date = df_csv['Date'].iloc[0]
last_date = df_csv['Date'].iloc[-1]

df_eval = df.loc[first_date:last_date].copy()
sum_sigs_eval = sum_sigs.loc[first_date:last_date]

# Calculate metrics for binary majority vote
# agg_val = 1.0 if sum_sigs > 0 else -1.0
# Wait! In target backtest, the position can be 0.0, 0.5, or 1.0.
# If we run a binary backtest: agg_val = 1.0 (100% BTC) else 0.0 (0% BTC)
agg_val = sum_sigs_eval.apply(lambda x: 1.0 if x > 0 else 0.0)

cash = 10000.0
btc = 0.0
commission_rate = 0.001
prev_target = 0.0

daily_equity = []

for date, row in df_eval.iterrows():
    price = row['close']
    target = agg_val.loc[date]
    
    if target != prev_target:
        total_equity = cash + btc * price
        target_btc_val = total_equity * target
        current_btc_val = btc * price
        trade_val = target_btc_val - current_btc_val
        
        if trade_val > 0:
            comm = abs(trade_val) * commission_rate
            btc_change = (trade_val - comm) / price
            btc += btc_change
            cash -= trade_val
        elif trade_val < 0:
            comm = abs(trade_val) * commission_rate
            btc_change = trade_val / price
            btc += btc_change
            cash += abs(trade_val) - comm
            
        prev_target = target
    
    eq = cash + btc * price
    daily_equity.append(eq)

df_eval['equity'] = daily_equity
df_eval['returns'] = df_eval['equity'].pct_change()

mean_return = df_eval['returns'].mean()
std_return = df_eval['returns'].std()
sharpe_annual = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0.0

downside_returns = df_eval['returns'][df_eval['returns'] < 0]
std_downside = downside_returns.std()
sortino_annual = (mean_return / std_downside) * np.sqrt(365) if std_downside > 0 else 0.0

total_ret_pct = (df_eval['equity'].iloc[-1] - 10000.0) / 10000.0 * 100.0

print(f"Current Selection Binary Aggregate System:")
print(f"  Total Return: {total_ret_pct:.2f}%")
print(f"  Annualized Sharpe Ratio: {sharpe_annual:.4f}")
print(f"  Annualized Sortino Ratio: {sortino_annual:.4f}")
