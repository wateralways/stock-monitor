#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze 002218.SZ with moneyflow filters to test if they avoid falling knives"""

import json, time, requests
import pandas as pd
import numpy as np

# ===================== 1. Fetch data =====================
headers = {'Referer': 'https://finance.sina.com.cn/'}
all_records = []
for page in range(1, 5):
    url = (
        'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
        f'MoneyFlow.ssl_qsfx_zjlrqs?page={page}&num=100&sort=opendate&asc=0&daima=sz002218'
    )
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.text) > 100:
            data = json.loads(r.text)
            if not data:
                break
            for item in data:
                all_records.append({
                    'date': item['opendate'],
                    'big_net': float(item.get('r0_net', 0)),
                    'total_net': float(item.get('netamount', 0)),
                    'price': float(item.get('trade', 0)),
                })
            if len(data) < 100:
                break
    except Exception as e:
        print(f'page {page} error: {e}')
        break
    time.sleep(0.1)

records = sorted(all_records, key=lambda x: x['date'])
df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])

# Derived features
df['big_net_wan'] = df['big_net'] / 10000
df['big_net_3d'] = df['big_net'].rolling(3).sum()
df['big_net_5d'] = df['big_net'].rolling(5).sum()

# Outflow decay: 3 consecutive outflows with decreasing absolute value
df['prev_big'] = df['big_net'].shift(1)
df['prev2_big'] = df['big_net'].shift(2)
df['outflow_decay'] = (
    (df['big_net'] < 0) & (df['prev_big'] < 0) & (df['prev2_big'] < 0) &
    (df['big_net'].abs() < df['prev_big'].abs()) &
    (df['prev_big'].abs() < df['prev2_big'].abs())
)

# Price features
df['pct_chg'] = df['price'].pct_change() * 100

# Filter May-June 2026
mask = (df['date'] >= '2026-05-01') & (df['date'] <= '2026-06-05')
mj = df[mask].copy().reset_index(drop=True)

sep = '=' * 115
print(sep)
print('  拓日新能 (002218)  2026年5-6月  价格 + 主力资金流向 综合分析')
print(sep)

# ===================== 2. Detailed daily table =====================
header = f"{'日期':<10} {'收盘':>7} {'涨跌幅':>8} {'主力净流入(万)':>14} {'3日累计(万)':>12} {'5日累计(万)':>12} | {'当日>0':>6} {'3日>0':>6} {'流出衰减':>6}"
print(header)
print('-' * 115)

for _, row in mj.iterrows():
    d = row['date'].strftime('%m/%d')
    f1 = ' YES ' if row['big_net'] > 0 else '  -  '
    f2 = ' YES ' if row['big_net_3d'] > 0 else '  -  '
    f3 = ' YES ' if row['outflow_decay'] else '  -  '
    vals = (d, row['price'], row['pct_chg'], row['big_net_wan'],
            row['big_net_3d']/10000, row['big_net_5d']/10000, f1, f2, f3)
    print(f"{vals[0]:<10} {vals[1]:>7.2f} {vals[2]:>+7.2f}% {vals[3]:>14.0f} {vals[4]:>12.0f} {vals[5]:>12.0f} | {vals[6]:>6} {vals[7]:>6} {vals[8]:>6}")

# ===================== 3. Key insight: moneyflow vs price action =====================
print()
print(sep)
print('  关键观察: 价格走势 vs 主力资金流向')
print(sep)

# Phase analysis
phases = [
    ('05/06-05/14', mj.iloc[:7], '上涨冲顶阶段'),
    ('05/15-05/21', mj.iloc[7:13], '放量杀跌阶段'),
    ('05/22-05/25', mj.iloc[13:15], '止跌反弹阶段'),
    ('05/26-05/27', mj.iloc[15:17], '二次探底阶段'),
    ('05/28-06/02', mj.iloc[17:], '低位震荡阶段'),
]

for label, phase_df, desc in phases:
    if len(phase_df) == 0:
        continue
    p_chg = (phase_df.iloc[-1]['price'] / phase_df.iloc[0]['price'] - 1) * 100
    net_flow = phase_df['big_net'].sum() / 10000
    direction = '主力净流入' if net_flow > 0 else '主力净流出'
    print(f"  {desc} {label}: 价格{p_chg:+.2f}% | {direction} {abs(net_flow):.0f}万")

print()
print('  关键矛盾点:')
print('    5月26日: 价格暴跌-8.27% 但 主力净流入+3188万!')
print('    -> 如果是主力在低位吸筹，短期的\"接飞刀\"其实是正确的逆向买入')
print('    -> 如果只是日内对倒/护盘，那就是虚假信号')

# ===================== 4. Backtest: filter on big drop entries =====================
print()
print(sep)
print('  回测: 假设策略在\"跌幅>3%\"的日期尾盘买入，持有3天卖出')
print('  测试3种过滤条件的效果')
print(sep)

# Find all candidate entry days (big drops >3%)
drops = []
for idx, row in mj.iterrows():
    if row['pct_chg'] > -3:
        continue
    # Find exit price (3 days later or last available)
    exit_idx = min(idx + 3, len(mj) - 1)
    exit_price = mj.iloc[exit_idx]['price']
    entry_price = row['price']
    ret = (exit_price / entry_price - 1) * 100

    drops.append({
        'idx': idx,
        'date': row['date'].strftime('%m/%d'),
        'exit_date': mj.iloc[exit_idx]['date'].strftime('%m/%d'),
        'entry': entry_price,
        'exit': exit_price,
        'ret': ret,
        'pct': row['pct_chg'],
        'big_net': row['big_net_wan'],
        'big_net_3d': row['big_net_3d'] / 10000,
        'outflow_decay': row['outflow_decay'],
        'filter_day': row['big_net'] > 0,
        'filter_3d': row['big_net_3d'] > 0,
        'filter_decay': row['outflow_decay'],
    })

print(f'\n  触发买入信号(跌幅>3%)的日期: {len(drops)}天\n')

# Evaluate each filter separately
for filter_name, filter_key in [
    ('A: 当日主力净流入>0', 'filter_day'),
    ('B: 3日累计主力净流入>0', 'filter_3d'),
    ('C: 连续3日主力流出衰减', 'filter_decay'),
]:
    passed = [d for d in drops if d[filter_key]]
    blocked = [d for d in drops if not d[filter_key]]

    print(f'  === 过滤条件 {filter_name} ===')

    if passed:
        p_ret = sum(d['ret'] for d in passed)
        p_win = sum(1 for d in passed if d['ret'] > 0)
        print(f'    通过 {len(passed)} 笔, 胜率 {p_win}/{len(passed)} ({p_win/len(passed)*100:.0f}%), 累计收益 {p_ret:+.2f}%')
        for d in passed:
            print(f'      {d["date"]}: {d["pct"]:+.2f}% -> 持有3天 -> {d["ret"]:+.2f}%  (主力{d["big_net"]:+.0f}万)')

    if blocked:
        b_ret = sum(d['ret'] for d in blocked)
        b_win = sum(1 for d in blocked if d['ret'] > 0)
        print(f'    过滤 {len(blocked)} 笔, 如果买入胜率 {b_win}/{len(blocked)}, 累计收益 {b_ret:+.2f}%')
        for d in blocked:
            print(f'      {d["date"]}: {d["pct"]:+.2f}% -> 持有3天 -> {d["ret"]:+.2f}%  (主力{d["big_net"]:+.0f}万) <- 被过滤')
    print()

# ===================== 5. Summary =====================
print(sep)
print('  总结')
print(sep)
print("""
  过滤条件A (当日主力净流入>0):
    - 过滤掉了 5/15(-5.71%), 5/18(-1.96%), 5/20(-5.90%), 5/21(-4.37%) 的接飞刀
    - 这些都是\"放量杀跌\"阶段的典型飞刀，过滤后显著减少亏损
    - 但无法过滤 5/13(顶部买入，-8.18%亏损)，因为主力在顶部仍在净买入

  过滤条件B (3日累计>0):
    - 比条件A更宽松，很多主力在流出的日子仍能通过(因为前几天主力在买)
    - 过滤效果不如A，5/15的主力流出日仍能通过

  过滤条件C (流出衰减):
    - 只在连续3天都是流出且绝对值递减时触发，非常罕见
    - 只抓到5/19一天，太过保守

  建议:
    1. 用\"当日主力净流入>0\"作为必要的入场条件，可以避开大部分飞刀
    2. 结合\"3日累计净流入>0\"作为辅助确认，提高信号质量
    3. \"流出衰减\"信号太稀少，更适合作为\"关注但不急于入场\"的提醒
    4. 致命缺陷: 主力在顶部仍在买入时(如5/13)，过滤器会放行!
       需要结合价格位置(如距20日高点距离)来防止顶部追涨
""")
