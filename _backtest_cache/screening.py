#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股筛选: 市值>200亿 + 近10日大部分下跌 + 近5日大单净流入为正
数据来源: 东方财富
"""

import requests
import json
import time
import pandas as pd

proxies = {'http': None, 'https': None}

def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0

def get_fund_flow(code, market):
    """获取近5日资金流向数据（东方财富）"""
    url = 'http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get'
    params = {
        'lmt': 5, 'klt': 101,
        'secid': f'{market}.{code}',
        'fields1': 'f1,f2,f3,f7',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://quote.eastmoney.com/'
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30, proxies=proxies)
        data = resp.json()
        klines = data.get('data', {}).get('klines', [])
        
        total_big_net = 0
        inflow_days = 0
        details = []
        
        for k in klines:
            parts = k.split(',')
            if len(parts) >= 6:
                date = parts[0]
                big_net = float(parts[4])   # 大单净流入
                elg_net = float(parts[5])   # 超大单净流入
                total = big_net + elg_net
                total_big_net += total
                if total > 0:
                    inflow_days += 1
                details.append(f'{date}:{total/10000:+.0f}万')
        
        return total_big_net, inflow_days, details
    except Exception as e:
        return None, 0, [str(e)[:50]]


print("=" * 80)
print("A股筛选: 市值>200亿 + 近10日>=8天下跌 + 近5日大单净流入为正")
print("=" * 80)

# 步骤1: 获取市值>200亿的股票（东方财富）
print("\n步骤1: 获取市值>200亿的股票...")
name_map = {}
mv_map = {}
for page in range(1, 42):
    url = 'http://push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': page, 'pz': 100, 'po': 1, 'np': 1, 'fltt': 2, 'invt': 2,
        'fid': 'f20',
        'fs': 'm:0+t:6,m:0+t:13,m:1+t:2,m:1+t:23',
        'fields': 'f12,f14,f20'
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    data = resp.json()
    diff = data.get('data', {}).get('diff', [])
    for item in diff:
        code = item.get('f12', '')
        name = item.get('f14', '')
        total_mv = safe_float(item.get('f20'))
        if total_mv > 20000000000:
            name_map[code] = name
            mv_map[code] = round(total_mv / 100000000, 2)  # 亿元
    time.sleep(0.05)

print(f"  市值>200亿: {len(name_map)}只")

# 步骤2: 获取近10日K线数据，筛选>=8天下跌的（Tushare）
print("\n步骤2: 获取近10日K线数据...")
import tushare as ts
with open('D:/others/temp/earn_money/wateralways/stock-monitor/.env', 'r') as f:
    for line in f:
        if line.startswith('TUSHARE_TOKEN='):
            token = line.strip().split('=', 1)[1]
            break
ts.set_token(token)
pro = ts.pro_api()

def to_ts_code(code):
    if code.startswith('6') or code.startswith('68'):
        return code + '.SH'
    return code + '.SZ'

big_cap_codes = [to_ts_code(c) for c in name_map.keys()]

dates = ['20260518', '20260519', '20260520', '20260521', '20260522', 
         '20260525', '20260526', '20260527', '20260528', '20260529']

all_daily = []
for d in dates:
    df = pro.daily(trade_date=d)
    all_daily.append(df[df['ts_code'].isin(big_cap_codes)][['ts_code','pct_chg']])
    time.sleep(0.6)

daily_combined = pd.concat(all_daily, ignore_index=True)
down_days = daily_combined[daily_combined['pct_chg'] < 0].groupby('ts_code').size().reset_index(name='down_days')
up_days = daily_combined[daily_combined['pct_chg'] > 0].groupby('ts_code').size().reset_index(name='up_days')

stats = down_days.merge(up_days, on='ts_code', how='outer').fillna(0)
stats['down_days'] = stats['down_days'].astype(int)
stats['up_days'] = stats['up_days'].astype(int)

candidates = stats[stats['down_days'] >= 8].copy()
candidates = candidates.sort_values('down_days', ascending=False)
print(f"  近10日>=8天下跌: {len(candidates)}只")

# 步骤3: 获取近5日资金流向，筛选大单净流入为正的
print("\n步骤3: 获取近5日资金流向数据（东方财富）...")
results = []
for i, row in candidates.iterrows():
    ts_code = row['ts_code']
    raw_code = ts_code.split('.')[0]
    market = 1 if ts_code.endswith('.SH') else 0
    name = name_map.get(raw_code, raw_code)
    
    total_big_net, inflow_days, details = get_fund_flow(raw_code, market)
    time.sleep(1.2)
    
    if total_big_net is not None:
        results.append({
            'ts_code': ts_code,
            'name': name,
            'raw_code': raw_code,
            'mv_yi': mv_map.get(raw_code, 0),
            'down_days': int(row['down_days']),
            'up_days': int(row['up_days']),
            'total_big_net_wan': round(total_big_net / 10000, 1),
            'total_big_net_yi': round(total_big_net / 100000000, 3),
            'inflow_days': inflow_days,
            'details': details
        })
    
    if (len(results)) % 10 == 0 or len(results) == len(candidates):
        print(f"  进度: {len(results)}/{len(candidates)}")

# 筛选结果
filtered = [r for r in results if r['total_big_net_yi'] > 0]
filtered.sort(key=lambda x: x['total_big_net_yi'], reverse=True)

print(f"\n{'='*80}")
print(f"筛选结果")
print(f"{'='*80}")
print(f"市值>200亿: {len(name_map)}只")
print(f"近10日>=8天下跌: {len(candidates)}只")
print(f"近5日大单净流入为正: {len(filtered)}只")
print()

if not filtered:
    print("未找到符合条件的股票。")
else:
    print(f"{'代码':<12} {'名称':<10} {'市值(亿)':>10} {'下跌天数':>8} {'净流入(万)':>12} {'净流入天数':>10}")
    print("-" * 80)
    for r in filtered:
        print(f"{r['ts_code']:<12} {r['name']:<10} {r['mv_yi']:>10.2f} {r['down_days']:>8}天 {r['total_big_net_wan']:>+12.0f} {r['inflow_days']:>9}/5天")
    
    print("\n详细资金流向:")
    for r in filtered:
        print(f"\n{r['ts_code']} {r['name']}: 近5日净流入 +{r['total_big_net_wan']:.0f}万")
        for d in r['details']:
            print(f"  {d}")

# 保存
with open('D:/others/temp/earn_money/wateralways/stock-monitor/_backtest_cache/screening_result.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
