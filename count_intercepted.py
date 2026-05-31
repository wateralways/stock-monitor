#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 S6/S7/S8 新过滤在2026年4~5月拦截了多少笔信号
"""
import sys, io, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 技术指标 (与前一个脚本相同)
def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_kdj(df, n=9):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    df2 = df.copy()
    df2["rsv"] = (df2["close"] - low_min) / (high_max - low_min) * 100
    df2["k"] = df2["rsv"].ewm(com=2, adjust=False).mean()
    df2["d"] = df2["k"].ewm(com=2, adjust=False).mean()
    df2["j"] = 3 * df2["k"] - 2 * df2["d"]
    return df2

def fetch_data(ts_code, start_date, end_date):
    try:
        warmup = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=warmup, end_date=end_date)
        if df is None or df.empty: return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def calc_mfi(df, window=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["vol"]
    mf = rmf.diff()
    pf = mf.where(mf > 0, 0).rolling(window).sum()
    nf = (-mf.where(mf < 0, 0)).rolling(window).sum()
    mr = pf / nf.replace(0, np.nan)
    return 100 - (100 / (1 + mr))

def add_indicators(df):
    df = df.copy()
    df["rsi14"] = calc_rsi(df["close"], 14)
    bb_u, bb_m, bb_l, bb_p = calc_bollinger(df["close"], 20, 2)
    df["bb_position"] = bb_p
    df["ma60"] = df["close"].rolling(60).mean()
    df = calc_kdj(df)
    df["vol_ma20"] = df["vol"].rolling(20).mean()
    return df

def calc_bollinger(prices, window=20, num_std=2):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return ma + std*num_std, ma, ma - std*num_std, (prices - (ma - std*num_std)) / ((ma + std*num_std) - (ma - std*num_std))

def vol_shrink_ratio(df, lookback=5):
    if df is None or len(df) < 20: return None
    v20 = df["vol"].rolling(20).mean().iloc[-1]
    if pd.isna(v20) or v20 <= 0: return None
    return float(df["vol"].iloc[-min(lookback, len(df)):].mean() / v20)

# ============ 信号定义 ============

# S7: KDJ超卖反弹
def s7_signal(df, idx):
    if idx < 1: return False
    row = df.iloc[idx]
    return pd.notna(row.get("j")) and row["j"] < 10 and row.get("pct_chg", 0) > 0

# S6: 评分超卖
def s6_signal(df, idx):
    if idx < 5: return False, 0
    row = df.iloc[idx]
    score = 0
    rsi = row.get("rsi14")
    if pd.notna(rsi):
        if rsi < 25: score += 20
        elif rsi < 30: score += 15
        elif rsi < 35: score += 10
        elif rsi < 40: score += 5
    bb = row.get("bb_position")
    if pd.notna(bb):
        if bb < 0.1: score += 15
        elif bb < 0.2: score += 10
        elif bb < 0.3: score += 5
    j = row.get("j")
    if pd.notna(j):
        if j < 0: score += 15
        elif j < 10: score += 12
        elif j < 20: score += 8
        elif j < 30: score += 4
    cd = row.get("consecutive_down", 0)
    if cd >= 4: score += 15
    elif cd >= 3: score += 12
    elif cd >= 2: score += 8
    if idx >= 5:
        ret5d = (row["close"] / df.iloc[idx-5]["close"] - 1) * 100
        if ret5d < -10: score += 10
        elif ret5d < -7: score += 8
        elif ret5d < -5: score += 6
        elif ret5d < -3: score += 3
    v20 = row.get("vol_ma20", 0)
    if pd.notna(v20) and v20 > 0:
        vr = df["vol"].iloc[idx] / v20
        if vr < 0.6: score -= 8
    return score >= 50, score

# S8: 深跌反弹
def s8_signal(df, idx, stock_name=""):
    if idx < 6: return False, []
    row = df.iloc[idx]
    ret5d = (row["close"] / df.iloc[idx-5]["close"] - 1) * 100
    rsi14 = row.get("rsi14")
    pct = row.get("pct_chg", 0)
    sigs = []
    if ret5d < -5 and pd.notna(rsi14) and rsi14 < 40 and pct > 0:
        sigs.append("A")
    if ret5d < -10 and stock_name in ("高澜股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源"):
        sigs.append("B")
    return len(sigs) > 0, sigs

# 新过滤检查
def check_new_filters(df, idx, stock_name, sh_trend="sideways"):
    """返回 (是否通过, 被哪个过滤拦截, 详情值)"""
    # 1. 缩量下跌
    vr = vol_shrink_ratio(df.iloc[:idx+1], lookback=5)
    if vr is not None and vr > 1.1:
        return False, "缩量下跌", vr
    
    # 2. MFI + 收盘位置 资金流向检查
    # 真反弹应满足：收盘在当日上半区或MFI正在回升
    if idx >= 14:
        mfi_series = calc_mfi(df.iloc[:idx+1], window=14)
        mfi_val = mfi_series.iloc[-1]
        mfi_prev = mfi_series.iloc[-2] if len(mfi_series) >= 2 else None
        if pd.notna(mfi_val):
            mfi_improving = pd.notna(mfi_prev) and mfi_val > mfi_prev
            row = df.iloc[idx]
            day_range = row["high"] - row["low"]
            close_position = (row["close"] - row["low"]) / day_range if day_range > 0 else 0.5
            close_mid_up = close_position > 0.5
            if not close_mid_up and not mfi_improving:
                return False, "MFI+收盘位置", round(mfi_val, 1)
    
    return True, None, None


def main():
    print("=" * 100)
    print("S6/S7/S8 新过滤拦截统计 (2026-04-01 ~ 2026-05-31)")
    print("=" * 100)
    
    # 配置
    configs = [
        ("S8 深跌反弹", {
            "300499.SZ": "高澜股份", "002272.SZ": "川润股份",
            "600418.SH": "江淮汽车", "300696.SZ": "爱乐达",
            "300572.SZ": "安车检测", "688223.SH": "晶科能源",
        }),
        ("S7 KDJ超卖反弹", {"002837.SZ": "英维克"}),
        ("S6 多因子评分超卖", {"002218.SZ": "拓日新能", "300572.SZ": "安车检测"}),
    ]
    
    grand_intercepted_vol = 0
    grand_intercepted_mf = 0
    grand_total_old = 0
    grand_total_new = 0
    
    for strategy_name, stock_dict in configs:
        print(f"\n{'─' * 80}")
        print(f"【{strategy_name}】")
        print(f"{'─' * 80}")
        
        strat_intercepted_vol = 0
        strat_intercepted_mf = 0
        strat_total_old = 0
        strat_total_new = 0
        
        for ts_code, name in stock_dict.items():
            print(f"\n  {name}({ts_code}):")
            df = fetch_data(ts_code, "20260401", "20260531")
            if df is None or len(df) < 30:
                print("    数据不足")
                continue
            df = add_indicators(df)
            
            df_start = df.index[df["trade_date"] >= "2026-04-01"]
            if len(df_start) == 0: continue
            start_idx = df_start[0]
            
            stock_old = 0
            stock_intercepted_vol = 0
            stock_intercepted_mf = 0
            stock_intercepted_unknown = 0
            
            for idx in range(start_idx, len(df)):
                # 获取 OLD 信号
                if strategy_name.startswith("S8"):
                    buy_old, sigs = s8_signal(df, idx, name)
                elif strategy_name.startswith("S7"):
                    buy_old = s7_signal(df, idx)
                elif strategy_name.startswith("S6"):
                    buy_old, _ = s6_signal(df, idx)
                
                if not buy_old:
                    continue
                
                stock_old += 1
                
                # 对新信号应用过滤
                passed, reason, detail = check_new_filters(df, idx, name)
                if not passed:
                    if reason == "缩量下跌":
                        stock_intercepted_vol += 1
                    elif reason == "MFI+收盘位置":
                        stock_intercepted_mf += 1
                    else:
                        stock_intercepted_unknown += 1
            
            total_intercepted = stock_intercepted_vol + stock_intercepted_mf
            print(f"    OLD信号: {stock_old}次")
            if stock_old > 0:
                pct = total_intercepted / stock_old * 100
                print(f"    拦截: {total_intercepted}次 ({pct:.0f}%)")
                print(f"      - 缩量下跌过滤: {stock_intercepted_vol}次")
                print(f"      - MFI+收盘位置过滤: {stock_intercepted_mf}次")
                print(f"    通过(NEW信号): {stock_old - total_intercepted}次")
            else:
                print(f"    无信号")
            
            strat_total_old += stock_old
            strat_intercepted_vol += stock_intercepted_vol
            strat_intercepted_mf += stock_intercepted_mf
        
        strat_total_intercepted = strat_intercepted_vol + strat_intercepted_mf
        if strat_total_old > 0:
            print(f"\n  ── {strategy_name} 汇总 ──")
            print(f"    OLD总信号: {strat_total_old}次")
            print(f"    拦截合计: {strat_total_intercepted}次 ({strat_total_intercepted/strat_total_old*100:.0f}%)")
            print(f"      - 缩量下跌: {strat_intercepted_vol}次")
            print(f"      - 主力资金: {strat_intercepted_mf}次")
        
        grand_total_old += strat_total_old
        grand_intercepted_vol += strat_intercepted_vol
        grand_intercepted_mf += strat_intercepted_mf
    
    # 总体汇总
    grand_total_intercepted = grand_intercepted_vol + grand_intercepted_mf
    print(f"\n" + "=" * 100)
    print(f"【总体拦截统计】")
    print(f"=" * 100)
    print(f"  OLD信号总数: {grand_total_old}次")
    if grand_total_old > 0:
        print(f"  新过滤拦截: {grand_total_intercepted}次 ({grand_total_intercepted/grand_total_old*100:.0f}%)")
        print(f"    - 缩量下跌过滤: {grand_intercepted_vol}次 ({grand_intercepted_vol/grand_total_old*100:.0f}%)")
        print(f"    - MFI+收盘位置过滤: {grand_intercepted_mf}次 ({grand_intercepted_mf/grand_total_old*100:.0f}%)")
        print(f"  NEW信号总数: {grand_total_old - grand_total_intercepted}次")
    print()

if __name__ == "__main__":
    main()
