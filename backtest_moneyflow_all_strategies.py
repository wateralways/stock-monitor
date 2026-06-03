#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1/S2/S4/S6/S7/S8 双过滤回测: 主力净流入 + MA20斜率
对比 4 种场景: 无过滤 / 仅主力 / 仅MA20斜率 / 双过滤
期间: 2026年5月-6月
"""

import json, time, requests, sys, io
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

STOCKS = {
    "300696.SZ": {"name": "爱乐达", "sina": "sz300696", "strategies": ["S1", "S8"]},
    "000697.SZ": {"name": "ST炼石", "sina": "sz000697", "strategies": ["S1", "S2", "S4"]},
    "002928.SZ": {"name": "华夏航空", "sina": "sz002928", "strategies": ["S1"]},
    "002831.SZ": {"name": "裕同科技", "sina": "sz002831", "strategies": ["S4"]},
    "600486.SH": {"name": "扬农化工", "sina": "sh600486", "strategies": ["S4"]},
    "300627.SZ": {"name": "华测导航", "sina": "sz300627", "strategies": ["S4"]},
    "002272.SZ": {"name": "川润股份", "sina": "sz002272", "strategies": ["S4", "S8"]},
    "300499.SZ": {"name": "高澜股份", "sina": "sz300499", "strategies": ["S4", "S8"]},
    "002218.SZ": {"name": "拓日新能", "sina": "sz002218", "strategies": ["S4", "S6"]},
    "603912.SH": {"name": "佳力图", "sina": "sh603912", "strategies": ["S4"]},
    "000682.SZ": {"name": "东方电子", "sina": "sz000682", "strategies": ["S4"]},
    "300572.SZ": {"name": "安车检测", "sina": "sz300572", "strategies": ["S6", "S8"]},
    "002837.SZ": {"name": "英维克", "sina": "sz002837", "strategies": ["S7"]},
    "600418.SH": {"name": "江淮汽车", "sina": "sh600418", "strategies": ["S8"]},
    "688223.SH": {"name": "晶科能源", "sina": "sh688223", "strategies": ["S8"]},
}

# ===================== Indicators =====================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bollinger(close, period=20, std=2):
    ma = close.rolling(period).mean()
    s = close.rolling(period).std()
    upper = ma + std * s
    lower = ma - std * s
    pos = (close - lower) / (upper - lower)
    return ma, upper, lower, pos

def calc_kdj(df, n=9, k_smooth=3, d_smooth=3):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(alpha=1/k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1/d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def calc_consecutive_down(pct_chg):
    result = pd.Series(0, index=pct_chg.index, dtype=int)
    cnt = 0
    for i in range(len(pct_chg)):
        if pd.notna(pct_chg.iloc[i]) and pct_chg.iloc[i] < 0:
            cnt += 1
        else:
            cnt = 0
        result.iloc[i] = cnt
    return result

def get_adaptive_params(df):
    vol_20 = df["pct_chg"].rolling(20).std().iloc[-1]
    vol_60 = df["pct_chg"].rolling(60).std().iloc[-1] if len(df) >= 60 else vol_20
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    return {
        "rsi_entry": float(np.clip(33 - (vol_ratio - 1.0) * 6, 20, 45)),
        "bb_entry": float(np.clip(0.5 - (vol_ratio - 1.0) * 0.2, 0.1, 0.6)),
        "rsi_consec": float(np.clip(35 - (vol_ratio - 1.0) * 5, 25, 40)),
        "kdj_entry": float(np.clip(10 - (vol_ratio - 1.0) * 3, 2, 15)),
    }

# ===================== Data Fetching =====================
def fetch_price_data(ts_code, days=80):
    try:
        end_date = datetime.now() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=days)
        df = pro.daily(ts_code=ts_code,
                       start_date=start_date.strftime("%Y%m%d"),
                       end_date=end_date.strftime("%Y%m%d"))
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    except Exception as e:
        print(f"    [ERROR] fetch {ts_code}: {e}")
        return None

def fetch_moneyflow_sina_bulk(sina_code, pages=4):
    headers = {'Referer': 'https://finance.sina.com.cn/'}
    all_records = []
    for page in range(1, pages + 1):
        url = (
            'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
            f'MoneyFlow.ssl_qsfx_zjlrqs?page={page}&num=100&sort=opendate&asc=0&daima={sina_code}'
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.text) > 100:
                data = json.loads(r.text)
                if not data: break
                for item in data:
                    all_records.append({
                        'date': pd.Timestamp(item['opendate']),
                        'big_net': float(item.get('r0_net', 0)),
                    })
                if len(data) < 100: break
        except Exception: break
        time.sleep(0.15)
    return sorted(all_records, key=lambda x: x['date'])

# ===================== Signal Generation =====================
# All signal generators now also compute filter fields

def detect_s1_signals(df, stock_name):
    signals = []
    for i in range(40, len(df)):
        window = df.iloc[:i+1].copy()
        prices = window["close"]
        rsi = calc_rsi(prices, 14)
        _, _, _, bb_pos = calc_bollinger(prices, 20, 2)
        row = df.iloc[i]
        today_rsi = rsi.iloc[-1]
        today_bb = bb_pos.iloc[-1]

        ap = get_adaptive_params(window)
        if "ST" in stock_name:
            rsi_th, bb_th = 33, 0.5
        else:
            rsi_th, bb_th = ap["rsi_entry"], ap["bb_entry"]

        rsi_ok = today_rsi < rsi_th if pd.notna(today_rsi) else False
        bb_ok = today_bb < bb_th if pd.notna(today_bb) else False
        up = row["pct_chg"] > 0
        yang = row["close"] > row["open"]

        # MA20 slope
        ma20 = prices.rolling(20).mean()
        ma20_now = ma20.iloc[-1]
        ma20_5ago = ma20.iloc[-6] if len(ma20) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

        if (rsi_ok or bb_ok) and up and yang:
            signals.append({
                'idx': i, 'date': row['trade_date'], 'price': row['close'],
                'strategy': 'S1-RSI+布林带', 'stock': stock_name,
                'ma20_slope': round(ma20_slope, 2),
            })
    return signals

def detect_s2_signals(df, stock_name):
    signals = []
    k, d, j = calc_kdj(df)
    ap = get_adaptive_params(df)
    for i in range(40, len(df)):
        row = df.iloc[i]
        sig_j = j.iloc[i-3] if i >= 3 and pd.notna(j.iloc[i-3]) else None
        sig_ma20 = df["close"].rolling(20).mean().iloc[i-3] if i >= 3 else None
        sig_close = df.iloc[i-3]["close"] if i >= 3 else None

        sig_signals = []
        if sig_ma20 and sig_close and abs(sig_close - sig_ma20) / sig_ma20 < 0.02 and df.iloc[i-3]["pct_chg"] > 0:
            sig_signals.append("MA20")
        if sig_j is not None and sig_j < ap["kdj_entry"]:
            sig_signals.append("KDJ")

        if sig_signals:
            rise = (row["close"] - sig_close) / sig_close * 100
            if rise < 2.0:
                ma20 = df["close"].rolling(20).mean()
                ma20_now = ma20.iloc[i]
                ma20_5ago = ma20.iloc[i-5] if i >= 5 and pd.notna(ma20.iloc[i-5]) else ma20_now
                ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

                signals.append({
                    'idx': i, 'date': row['trade_date'], 'price': row['close'],
                    'strategy': 'S2-MA支撑+KDJ', 'stock': stock_name,
                    'ma20_slope': round(ma20_slope, 2),
                })
    return signals

def detect_s4_signals(df, stock_name):
    signals = []
    rsi = calc_rsi(df["close"], 14)
    cons = calc_consecutive_down(df["pct_chg"])
    for i in range(30, len(df)):
        row = df.iloc[i]
        ap = get_adaptive_params(df.iloc[:i+1])
        if stock_name in ("扬农化工", "拓日新能", "佳力图"):
            rsi_th = 35
        else:
            rsi_th = ap["rsi_consec"]
        rsi_ok = pd.notna(rsi.iloc[i]) and rsi.iloc[i] <= rsi_th
        cons_ok = cons.iloc[i] >= 2

        if rsi_ok and cons_ok:
            ma20 = df["close"].rolling(20).mean()
            ma20_now = ma20.iloc[i]
            ma20_5ago = ma20.iloc[i-5] if i >= 5 and pd.notna(ma20.iloc[i-5]) else ma20_now
            ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

            signals.append({
                'idx': i, 'date': row['trade_date'], 'price': row['close'],
                'strategy': 'S4-RSI+连跌', 'stock': stock_name,
                'ma20_slope': round(ma20_slope, 2),
            })
    return signals

def detect_s6_signals(df, stock_name):
    signals = []
    rsi = calc_rsi(df["close"], 14)
    bb_ma, bb_upper, bb_lower, bb_pos = calc_bollinger(df["close"], 20, 2)
    k, d, j = calc_kdj(df)
    cons = calc_consecutive_down(df["pct_chg"])
    for i in range(30, len(df)):
        row = df.iloc[i]
        score = 0
        if pd.notna(rsi.iloc[i]):
            v = rsi.iloc[i]
            if v < 25: score += 25
            elif v < 30: score += 20
            elif v < 35: score += 15
            elif v < 40: score += 10
            elif v < 45: score += 5
        if pd.notna(bb_pos.iloc[i]):
            v = bb_pos.iloc[i]
            if v < 0.1: score += 20
            elif v < 0.2: score += 16
            elif v < 0.3: score += 12
            elif v < 0.4: score += 8
            elif v < 0.5: score += 4
        if pd.notna(j.iloc[i]):
            v = j.iloc[i]
            if v < 0: score += 15
            elif v < 10: score += 12
            elif v < 20: score += 8
            elif v < 30: score += 4
        if cons.iloc[i] >= 4: score += 15
        elif cons.iloc[i] >= 3: score += 12
        elif cons.iloc[i] >= 2: score += 8
        if i >= 5:
            ret5d = (row["close"] / df.iloc[i-5]["close"] - 1) * 100
            if ret5d < -10: score += 10
            elif ret5d < -7: score += 8
            elif ret5d < -5: score += 6
            elif ret5d < -3: score += 3
        if i >= 5:
            vol_ma5 = df["vol"].iloc[i-4:i+1].mean()
            vol_ratio = row["vol"] / vol_ma5 if vol_ma5 > 0 else 1
            if vol_ratio < 0.6: score -= 8
        if score >= 50:
            ma20 = df["close"].rolling(20).mean()
            ma20_now = ma20.iloc[i]
            ma20_5ago = ma20.iloc[i-5] if i >= 5 and pd.notna(ma20.iloc[i-5]) else ma20_now
            ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

            signals.append({
                'idx': i, 'date': row['trade_date'], 'price': row['close'],
                'strategy': 'S6-评分超卖', 'stock': stock_name,
                'ma20_slope': round(ma20_slope, 2),
            })
    return signals

def detect_s7_signals(df, stock_name):
    signals = []
    k, d, j = calc_kdj(df)
    for i in range(30, len(df)):
        row = df.iloc[i]
        j_val = j.iloc[i]
        is_up = row["pct_chg"] > 0
        if pd.notna(j_val) and j_val < 10 and is_up:
            ma20 = df["close"].rolling(20).mean()
            ma20_now = ma20.iloc[i]
            ma20_5ago = ma20.iloc[i-5] if i >= 5 and pd.notna(ma20.iloc[i-5]) else ma20_now
            ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

            signals.append({
                'idx': i, 'date': row['trade_date'], 'price': row['close'],
                'strategy': 'S7-KDJ超卖', 'stock': stock_name,
                'ma20_slope': round(ma20_slope, 2),
            })
    return signals

def detect_s8_signals(df, stock_name):
    signals = []
    rsi = calc_rsi(df["close"], 14)
    for i in range(10, len(df)):
        row = df.iloc[i]
        if i < 5: continue
        ret5d = (row["close"] / df.iloc[i-5]["close"] - 1) * 100
        today_rsi = rsi.iloc[i]
        today_pct = row["pct_chg"]
        sig_a = ret5d < -5 and pd.notna(today_rsi) and today_rsi < 40 and today_pct > 0
        sig_b = ret5d < -10 and stock_name in ("高澜股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源")

        if sig_a or sig_b:
            ma20 = df["close"].rolling(20).mean()
            ma20_now = ma20.iloc[i]
            ma20_5ago = ma20.iloc[i-5] if i >= 5 and pd.notna(ma20.iloc[i-5]) else ma20_now
            ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0

            sig_type = "A" if sig_a else "B"
            signals.append({
                'idx': i, 'date': row['trade_date'], 'price': row['close'],
                'strategy': 'S8-深跌反弹', 'stock': stock_name,
                'ma20_slope': round(ma20_slope, 2),
                'ret5d': round(ret5d, 1), 'sig_type': sig_type,
            })
    return signals

# ===================== Backtest Engine =====================
def backtest_with_filters(price_df, mf_records, signals, hold_days=5):
    """Calculate returns under 4 filter scenarios for each signal"""
    results = []
    for sig in signals:
        idx = sig['idx']
        sig_date = sig['date']
        exit_idx = min(idx + hold_days, len(price_df) - 1)
        exit_price = price_df.iloc[exit_idx]['close']
        entry_price = sig['price']
        ret = (exit_price / entry_price - 1) * 100

        # Moneyflow filter: 当日主力净流入 > 0
        mf_on_date = None
        for mf in mf_records:
            if mf['date'] == sig_date:
                mf_on_date = mf['big_net']
                break
        mf_pass = mf_on_date is not None and mf_on_date > 0
        mf_nodata = mf_on_date is None

        # MA20 slope filter: slope > -1.0 (not in downtrend)
        ma20_slope = sig.get('ma20_slope', 0)
        ma_pass = ma20_slope > -1.0

        # 4 scenarios
        s_none = True       # no filter
        s_mf = mf_pass      # moneyflow only
        s_ma = ma_pass      # MA20 slope only
        s_both = mf_pass and ma_pass  # both filters

        results.append({
            **sig,
            'entry': entry_price, 'exit': exit_price,
            'exit_date': price_df.iloc[exit_idx]['trade_date'],
            'hold_days': hold_days, 'ret': ret,
            'mf_big_net': mf_on_date if mf_on_date is not None else 0,
            'mf_data': 'yes' if mf_on_date is not None else 'no',
            'mf_pass': mf_pass, 'ma_pass': ma_pass,
            's_none': s_none, 's_mf': s_mf, 's_ma': s_ma, 's_both': s_both,
        })
    return results

# ===================== Main =====================
def main():
    print("=" * 130)
    print("  S1/S2/S4/S6/S7/S8 双过滤回测: 主力净流入 + MA20斜率 — 2026年5-6月")
    print("=" * 130)

    # Step 1: Fetch price data
    print("\n[Step 1] 获取价格数据...")
    price_data = {}
    for code, info in STOCKS.items():
        name = info['name']
        print(f"  {name} ({code})...", end=" ", flush=True)
        df = fetch_price_data(code, days=80)
        if df is not None and len(df) >= 30:
            price_data[code] = df
            print(f"{len(df)}条")
        else:
            print("无数据!")
        time.sleep(0.3)
    print(f"  成功: {len(price_data)}/15")

    # Step 2: Fetch moneyflow
    print("\n[Step 2] 获取主力资金流向...")
    mf_data = {}
    for code, info in STOCKS.items():
        if code not in price_data: continue
        name = info['name']
        print(f"  {name} ({info['sina']})...", end=" ", flush=True)
        mf = fetch_moneyflow_sina_bulk(info['sina'], pages=4)
        if mf:
            mf_data[code] = mf
            print(f"{len(mf)}条")
        else:
            print("无数据!")
        time.sleep(0.3)
    print(f"  成功: {len(mf_data)}/15")

    # Step 3: Generate signals and backtest
    print("\n[Step 3] 生成信号并回测...")

    strategy_funcs = {
        'S1': detect_s1_signals, 'S2': detect_s2_signals,
        'S4': detect_s4_signals, 'S6': detect_s6_signals,
        'S7': detect_s7_signals, 'S8': detect_s8_signals,
    }
    strategy_hold = {'S1': 5, 'S2': 4, 'S4': 5, 'S6': 5, 'S7': 4, 'S8': 4}

    all_results = []

    for code, info in STOCKS.items():
        if code not in price_data or code not in mf_data:
            continue
        name = info['name']
        price_df = price_data[code]
        mf_records = mf_data[code]

        for strat in info['strategies']:
            func = strategy_funcs[strat]
            hold_days = strategy_hold[strat]
            signals = func(price_df, name)
            may_june = [s for s in signals
                       if s['date'] >= pd.Timestamp('2026-05-01')
                       and s['date'] <= pd.Timestamp('2026-06-05')]
            if may_june:
                results = backtest_with_filters(price_df, mf_records, may_june, hold_days)
                all_results.extend(results)

    # Step 4: Aggregate by scenario and strategy
    print("\n" + "=" * 130)
    print("  按策略 × 4种过滤场景 汇总")
    print("=" * 130)

    scenarios = [
        ('A: 无过滤', 's_none', '原始信号，全部买入'),
        ('B: 仅主力净流入>0', 's_mf', '当日主力净流入>0 才买入'),
        ('C: 仅MA20斜率>-1%', 's_ma', 'MA20不在下降趋势才买入'),
        ('D: 双过滤(主力+MA20)', 's_both', '两条件同时满足才买入'),
    ]

    strat_names = {
        'S1': 'RSI+布林带均值回归', 'S2': 'MA支撑+KDJ超卖',
        'S4': 'RSI+连跌中等信号', 'S6': '多因子评分超卖',
        'S7': 'KDJ超卖反弹', 'S8': '深跌反弹',
    }

    # Aggregate by strategy
    for strat in ['S1', 'S2', 'S4', 'S6', 'S7', 'S8']:
        strat_results = [r for r in all_results if r['strategy'].startswith(strat)]
        if not strat_results:
            continue

        print(f"\n{'─' * 110}")
        print(f"  {strat} {strat_names.get(strat, '')} — 共 {len(strat_results)} 笔信号")
        print(f"{'─' * 110}")
        print(f"  {'场景':<30} {'笔数':>5} {'胜率':>8} {'累计收益':>10} {'平均收益':>10} {'最大亏损':>10} {'最大盈利':>10}")
        print(f"  {'─' * 80}")

        for s_name, s_key, s_desc in scenarios:
            sigs = [r for r in strat_results if r[s_key]]
            if not sigs:
                print(f"  {s_name:<30} {'0笔':>5}")
                continue

            n = len(sigs)
            win = sum(1 for r in sigs if r['ret'] > 0)
            wr = win / n * 100
            total_ret = sum(r['ret'] for r in sigs)
            avg_ret = total_ret / n
            max_loss = min(r['ret'] for r in sigs)
            max_gain = max(r['ret'] for r in sigs)

            print(f"  {s_name:<30} {n:>5} {wr:>7.0f}% {total_ret:>+10.2f}% {avg_ret:>+10.2f}% {max_loss:>+10.2f}% {max_gain:>+10.2f}%")

    # Step 5: Overall comparison
    print(f"\n{'=' * 130}")
    print(f"  全策略 × 4种过滤场景 总览")
    print(f"{'=' * 130}")
    print(f"  {'场景':<30} {'笔数':>5} {'胜率':>8} {'累计收益':>10} {'平均收益':>10} {'最大亏损':>10} {'最大盈利':>10}")
    print(f"  {'─' * 80}")

    scenario_totals = {}
    for s_name, s_key, s_desc in scenarios:
        sigs = [r for r in all_results if r[s_key]]
        n = len(sigs)
        if n == 0:
            print(f"  {s_name:<30} {'0笔':>5}")
            continue
        win = sum(1 for r in sigs if r['ret'] > 0)
        wr = win / n * 100
        total_ret = sum(r['ret'] for r in sigs)
        avg_ret = total_ret / n
        max_loss = min(r['ret'] for r in sigs)
        max_gain = max(r['ret'] for r in sigs)

        scenario_totals[s_key] = {
            'n': n, 'win': win, 'wr': wr, 'total': total_ret,
            'avg': avg_ret, 'max_loss': max_loss, 'max_gain': max_gain,
        }

        print(f"  {s_name:<30} {n:>5} {wr:>7.0f}% {total_ret:>+10.2f}% {avg_ret:>+10.2f}% {max_loss:>+10.2f}% {max_gain:>+10.2f}%")

    # Step 6: Improvement analysis
    print(f"\n{'=' * 130}")
    print(f"  过滤效果对比分析")
    print(f"{'=' * 130}")

    base = scenario_totals.get('s_none', {})
    if base:
        base_n = base['n']
        base_total = base['total']
        base_avg = base['avg']
        base_wr = base['wr']

        for s_name, s_key, s_desc in scenarios[1:]:  # skip baseline
            s = scenario_totals.get(s_key)
            if not s: continue
            reduced = base_n - s['n']
            avoided_loss = base_total - s['total']
            avg_improve = s['avg'] - base_avg
            wr_change = s['wr'] - base_wr

            print(f"\n  {s_name}:")
            print(f"    信号从 {base_n} 减少到 {s['n']} (过滤掉 {reduced} 笔)")
            print(f"    累计收益: {base_total:+.2f}% → {s['total']:+.2f}% (改善 {avoided_loss:+.2f}%)")
            print(f"    平均收益: {base_avg:+.2f}% → {s['avg']:+.2f}% (提升 {avg_improve:+.2f}%)")
            print(f"    胜率: {base_wr:.0f}% → {s['wr']:.0f}% (变化 {wr_change:+.0f}%)")
            if avoided_loss > 0:
                print(f"    💰 成功避免了 {avoided_loss:.2f}% 的亏损!")

    # Step 7: Worst false positives (passed both filters but still lost big)
    print(f"\n{'=' * 130}")
    print(f"  双过滤仍失败的信号 (通过了主力+MA20两道关卡，但仍然亏损)")
    print(f"{'=' * 130}")
    both_fails = [r for r in all_results if r['s_both'] and r['ret'] < -3]
    both_fails.sort(key=lambda x: x['ret'])

    if both_fails:
        print(f"  {'策略':<16} {'股票':<10} {'日期':<10} {'MA20斜率':>8} {'主力(万)':>10} {'收益':>8}")
        print(f"  {'─' * 70}")
        for r in both_fails:
            print(f"  {r['strategy']:<16} {r['stock']:<10} {r['date'].strftime('%m/%d'):<10} "
                  f"{r['ma20_slope']:>+7.2f}% {r['mf_big_net']/10000:>+10.0f} {r['ret']:>+7.2f}%")
        print(f"\n  ⚠️ 以上 {len(both_fails)} 笔通过了双重过滤但仍然亏损，是过滤器的\"盲区\"")
    else:
        print("  无! 双过滤非常有效")

    # Step 8: Best filtered out (would have lost big, but filters saved us)
    print(f"\n{'=' * 130}")
    print(f"  双过滤拦截的最大飞刀 (被过滤掉的最大亏损)")
    print(f"{'=' * 130}")
    blocked = [r for r in all_results if not r['s_both'] and r['ret'] < -3]
    blocked.sort(key=lambda x: x['ret'])

    if blocked:
        print(f"  {'策略':<16} {'股票':<10} {'日期':<10} {'原因':<25} {'收益':>8}")
        print(f"  {'─' * 80}")
        for r in blocked[:15]:
            reasons = []
            if not r['mf_pass']: reasons.append('主力净流出')
            if not r['ma_pass']: reasons.append('MA20下降')
            reason = '+'.join(reasons)
            print(f"  {r['strategy']:<16} {r['stock']:<10} {r['date'].strftime('%m/%d'):<10} "
                  f"{reason:<25} {r['ret']:>+7.2f}%")

    print()

if __name__ == "__main__":
    main()
