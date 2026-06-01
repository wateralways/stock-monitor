#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6/S7/S8 (跌超反弹策略) 新增过滤回测
对比 OLD vs NEW（缩量下跌过滤 + 主力资金流向过滤）
回测区间：2026-04-01 ~ 2026-05-31
"""

import sys, io, os, json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ============ 技术指标 ============

def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger(prices, window=20, num_std=2):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    return upper, ma, lower, (prices - lower) / (upper - lower)

def calc_kdj(df, n=9):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    df2 = df.copy()
    df2["rsv"] = (df2["close"] - low_min) / (high_max - low_min) * 100
    df2["k"] = df2["rsv"].ewm(com=2, adjust=False).mean()
    df2["d"] = df2["k"].ewm(com=2, adjust=False).mean()
    df2["j"] = 3 * df2["k"] - 2 * df2["d"]
    return df2

def calc_consecutive_days(pct_chg, direction="down"):
    result = pd.Series(0, index=pct_chg.index)
    for i in range(1, len(pct_chg)):
        if direction == "down" and pct_chg.iloc[i] < 0:
            result.iloc[i] = result.iloc[i - 1] + 1
        elif direction == "up" and pct_chg.iloc[i] > 0:
            result.iloc[i] = result.iloc[i - 1] + 1
    return result

# ============ 数据获取 ============

def fetch_data(ts_code, start_date, end_date):
    try:
        warmup = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=warmup, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    except Exception as e:
        print(f"  [ERROR] fetch {ts_code}: {e}")
        return None

def fetch_moneyflow_backtest(ts_code, trade_date):
    """回测中获取资金流向数据"""
    try:
        df = pro.moneyflow(ts_code=ts_code, trade_date=trade_date)
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        return {
            "buy_lg_amount": float(row.get("buy_lg_amount", 0)),
            "sell_lg_amount": float(row.get("sell_lg_amount", 0)),
            "buy_elg_amount": float(row.get("buy_elg_amount", 0)),
            "sell_elg_amount": float(row.get("sell_elg_amount", 0)),
            "net_mf_amount": float(row.get("net_mf_amount", 0)),
        }
    except:
        return None

def add_indicators(df):
    df = df.copy()
    df["rsi14"] = calc_rsi(df["close"], 14)
    bb_u, bb_m, bb_l, bb_p = calc_bollinger(df["close"], 20, 2)
    df["bb_upper"] = bb_u; df["bb_mid"] = bb_m; df["bb_lower"] = bb_l
    df["bb_position"] = bb_p
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    df = calc_kdj(df)
    df["consecutive_down"] = calc_consecutive_days(df["pct_chg"], "down")
    df["vol_ma20"] = df["vol"].rolling(20).mean()
    return df

# ============ 信号定义 ============

def compute_vol_shrink_ratio(df, lookback=5):
    """计算过去N日均量 / 20日均量"""
    if df is None or len(df) < 20:
        return None
    vol_ma20 = df["vol"].rolling(20).mean().iloc[-1]
    if pd.isna(vol_ma20) or vol_ma20 <= 0:
        return None
    vol_recent = df["vol"].iloc[-min(lookback, len(df)):].mean()
    return float(vol_recent / vol_ma20)

def calc_mfi_for_slice(df_slice, window=14):
    tp = (df_slice["high"] + df_slice["low"] + df_slice["close"]) / 3
    rmf = tp * df_slice["vol"]
    diff = rmf.diff()
    pos = diff.where(diff > 0, 0).rolling(window).sum()
    neg = (-diff.where(diff < 0, 0)).rolling(window).sum()
    mr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mr))

# ---------- S6: 多因子评分超卖 (OLD) ----------

def s6_score_old(df, idx):
    """S6原始评分逻辑"""
    if idx < 5:
        return False, 0
    row = df.iloc[idx]
    score = 0
    # RSI
    rsi = row.get("rsi14")
    if pd.notna(rsi):
        if rsi < 25: score += 20
        elif rsi < 30: score += 15
        elif rsi < 35: score += 10
        elif rsi < 40: score += 5
    # BB位置
    bb = row.get("bb_position")
    if pd.notna(bb):
        if bb < 0.1: score += 15
        elif bb < 0.2: score += 10
        elif bb < 0.3: score += 5
    # KDJ J值
    j = row.get("j")
    if pd.notna(j):
        if j < 0: score += 15
        elif j < 10: score += 12
        elif j < 20: score += 8
        elif j < 30: score += 4
    # 连跌天数
    cd = row.get("consecutive_down", 0)
    if cd >= 4: score += 15
    elif cd >= 3: score += 12
    elif cd >= 2: score += 8
    # 5日跌幅
    if idx >= 5:
        ret5d = (row["close"] / df.iloc[idx-5]["close"] - 1) * 100
        if ret5d < -10: score += 10
        elif ret5d < -7: score += 8
        elif ret5d < -5: score += 6
        elif ret5d < -3: score += 3
    # 缩量惩罚
    vol_ma5 = df["vol"].iloc[max(0,idx-4):idx+1].mean()
    vol_ma20_val = row.get("vol_ma20", 0)
    if pd.notna(vol_ma20_val) and vol_ma20_val > 0:
        vol_ratio = df["vol"].iloc[idx] / vol_ma20_val
        if vol_ratio < 0.6:
            score -= 8
    return score >= 50, score

# ---------- S7: KDJ超卖反弹 (OLD) ----------

def s7_signal_old(df, idx):
    """S7原始逻辑: J<10 + 今日涨"""
    if idx < 1:
        return False
    row = df.iloc[idx]
    j = row.get("j")
    pct = row.get("pct_chg", 0)
    return pd.notna(j) and j < 10 and pct > 0

# ---------- S8: 深跌反弹 (OLD) ----------

def s8_signal_old(df, idx, stock_name=""):
    """S8原始逻辑: 信号A 或 信号B"""
    if idx < 6:
        return False, []
    row = df.iloc[idx]
    ret5d = (row["close"] / df.iloc[idx-5]["close"] - 1) * 100
    rsi14 = row.get("rsi14")
    pct = row.get("pct_chg", 0)
    signals = []
    # 信号A
    sig_a = ret5d < -5 and pd.notna(rsi14) and rsi14 < 40 and pct > 0
    if sig_a:
        signals.append("A")
    # 信号B
    sig_b = ret5d < -10 and stock_name in ("高澜股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源")
    if sig_b:
        signals.append("B")
    return len(signals) > 0, signals

# ---------- 新过滤封装 ----------

def apply_new_filters(df, idx, stock_name, sh_trend="sideways"):
    """
    应用两个新过滤：
    1. 缩量下跌过滤：量比>1.1（放量下跌）时拦截
    2. MFI + 收盘位置 资金流向检查：收盘在下半区且MFI无改善时拦截
    返回 (pass_filter: bool, filters_info: dict)
    """
    info = {"vol_shrink_ratio": None, "vol_expand_drop": False,
            "mfi_value": None, "main_force_weak": False}
    
    # 1. 缩量下跌
    vol_ratio = compute_vol_shrink_ratio(df.iloc[:idx+1], lookback=5)
    info["vol_shrink_ratio"] = vol_ratio
    if vol_ratio is not None and vol_ratio > 1.1:
        info["vol_expand_drop"] = True
        return False, info
    
    # 2. MFI + 收盘位置
    if idx >= 14:
        mfi_s = calc_mfi_for_slice(df.iloc[:idx+1], window=14)
        mfi_val = mfi_s.iloc[-1]
        mfi_prev = mfi_s.iloc[-2] if len(mfi_s) >= 2 else None
        if pd.notna(mfi_val):
            mfi_improving = pd.notna(mfi_prev) and mfi_val > mfi_prev
            row = df.iloc[idx]
            day_range = row["high"] - row["low"]
            close_position = (row["close"] - row["low"]) / day_range if day_range > 0 else 0.5
            close_mid_up = close_position > 0.5
            info["mfi_value"] = round(mfi_val, 1)
            if not close_mid_up and not mfi_improving:
                info["main_force_weak"] = True
                return False, info
    
    return True, info

# ============ 回测配置 ============

S8_STOCKS = {
    "300499.SZ": "高澜股份",
    "002272.SZ": "川润股份",
    "600418.SH": "江淮汽车",
    "300696.SZ": "爱乐达",
    "300572.SZ": "安车检测",
    "688223.SH": "晶科能源",
}

S6_STOCKS = {
    "002218.SZ": "拓日新能",
    "300572.SZ": "安车检测",
}

S7_STOCKS = {
    "002837.SZ": "英维克",
}

# ============ 回测引擎 ============

def backtest(df, stock_name, signal_func_new, signal_func_old,
             buy_offset=1, max_hold=5, bt_start="20260401", bt_end="20260531"):
    """
    通用回测，同时跑OLD和NEW版本
    signal_func_new(df, idx, stock_name) -> (buy_signal, info)
    signal_func_old(df, idx, stock_name) -> (buy_signal, ...)
    """
    bt_start_dt = pd.to_datetime(bt_start)
    bt_end_dt = pd.to_datetime(bt_end)
    
    start_idx = df.index[df["trade_date"] >= bt_start_dt]
    if len(start_idx) == 0:
        return [], []
    start_idx = start_idx[0]
    
    trades_old = []
    trades_new = []
    
    for version, trades, sig_func in [
        ("OLD", trades_old, signal_func_old),
        ("NEW", trades_new, signal_func_new),
    ]:
        holding = False
        buy_price = 0
        buy_date = None
        hold_days = 0
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            if row["trade_date"] > bt_end_dt:
                break
            
            if holding:
                hold_days += 1
                if hold_days >= max_hold:
                    sp = row["open"]  # 开盘卖出
                    pnl = (sp - buy_price) / buy_price * 100
                    trades.append({
                        "buy_date": buy_date,
                        "sell_date": row["trade_date"],
                        "buy_price": buy_price,
                        "sell_price": sp,
                        "pnl": pnl,
                        "hold_days": hold_days,
                        "exit": "到期",
                    })
                    holding = False
            else:
                if version == "OLD":
                    buy, _ = sig_func(df, idx, stock_name)
                else:
                    buy, _ = sig_func(df, idx, stock_name)
                
                if buy:
                    # 实际买入日
                    actual_idx = idx + buy_offset
                    if actual_idx < len(df):
                        buy_row = df.iloc[actual_idx]
                        bp = buy_row["open"]
                        if bp <= 0:
                            continue
                        buy_price = bp
                        buy_date = buy_row["trade_date"]
                        holding = True
                        hold_days = 0
    
    return trades_old, trades_new


def print_comparison(name, trades_old, trades_new):
    def stats(trades):
        if not trades:
            return {"count": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        return {"count": len(trades), "win_rate": wins/len(trades)*100,
                "total_pnl": sum(pnls), "avg_pnl": np.mean(pnls)}
    
    so = stats(trades_old)
    sn = stats(trades_new)
    
    print(f"\n  {name}:")
    print(f"    OLD: {so['count']}笔 | 胜率{so['win_rate']:.1f}% | 总收益{so['total_pnl']:+.2f}% | 均{so['avg_pnl']:+.2f}%")
    print(f"    NEW: {sn['count']}笔 | 胜率{sn['win_rate']:.1f}% | 总收益{sn['total_pnl']:+.2f}% | 均{sn['avg_pnl']:+.2f}%")
    
    if so['count'] > 0:
        imp_win = sn['win_rate'] - so['win_rate']
        imp_pnl = sn['total_pnl'] - so['total_pnl']
        imp_str = "+" if imp_win >= 0 else ""
        print(f"    改善: 胜率 {imp_str}{imp_win:.1f}% | 总收益 {imp_str}{imp_pnl:.2f}%")
    
    # 逐笔明细
    if trades_old:
        print("    OLD明细:")
        for t in trades_old:
            m = "✓" if t["pnl"] > 0 else "✗"
            print(f"      {m} {t['buy_date'].strftime('%m-%d')}→{t['sell_date'].strftime('%m-%d')} | "
                  f"买{t['buy_price']:.2f} 卖{t['sell_price']:.2f} | {t['pnl']:+.2f}%")
    if trades_new:
        print("    NEW明细:")
        for t in trades_new:
            m = "✓" if t["pnl"] > 0 else "✗"
            print(f"      {m} {t['buy_date'].strftime('%m-%d')}→{t['sell_date'].strftime('%m-%d')} | "
                  f"买{t['buy_price']:.2f} 卖{t['sell_price']:.2f} | {t['pnl']:+.2f}%")
    
    return so, sn


def main():
    print("=" * 100)
    print("S6/S7/S8 跌超反弹策略 - 新过滤效果回测")
    print("回测区间: 2026-04-01 ~ 2026-05-31")
    print("=" * 100)

    # --- S8: 深跌反弹 ---
    print("\n" + "=" * 80)
    print("【S8 深跌反弹】")
    print("=" * 80)
    
    all_s8_old, all_s8_new = [], []
    for ts_code, name in S8_STOCKS.items():
        print(f"\n  获取 {name}({ts_code}) 数据...")
        df = fetch_data(ts_code, "20260401", "20260531")
        if df is None or len(df) < 30:
            print("    [跳过] 数据不足")
            continue
        df = add_indicators(df)
        print(f"    {len(df)} 条日线")
        
        # OLD版信号
        def make_old(name):
            def fn(df, idx, stock_name):
                return s8_signal_old(df, idx, stock_name)
            return fn
        
        # NEW版信号（加过滤）
        def make_new(name):
            def fn(df, idx, stock_name):
                buy, sigs = s8_signal_old(df, idx, stock_name)
                if not buy:
                    return False, []
                pass_filter, info = apply_new_filters(df, idx, stock_name)
                return pass_filter, sigs
            return fn
        
        # 回测
        old_trades = []
        new_trades = []
        
        # 用T+1买 / T+5卖（信号A）或 T+1买 / T+6卖（信号B），简化用统一T+1/T+5
        for version, trades_list, sig_maker in [
            ("OLD", old_trades, make_old(name)),
            ("NEW", new_trades, make_new(name)),
        ]:
            holding = False
            buy_price = 0
            buy_date = None
            hold_days = 0
            max_hold = 5
            
            df_start = df.index[df["trade_date"] >= "2026-04-01"]
            if len(df_start) == 0:
                continue
            start_idx = df_start[0]
            
            for idx in range(start_idx, len(df)):
                row = df.iloc[idx]
                if holding:
                    hold_days += 1
                    if hold_days >= max_hold:
                        sp = row["open"]
                        pnl = (sp - buy_price) / buy_price * 100
                        trades_list.append({
                            "buy_date": buy_date, "sell_date": row["trade_date"],
                            "buy_price": buy_price, "sell_price": sp,
                            "pnl": pnl, "exit": "到期",
                        })
                        holding = False
                else:
                    buy, _ = sig_maker(df, idx, name)
                    if buy:
                        actual_idx = idx + 1  # T+1
                        if actual_idx < len(df):
                            buy_row = df.iloc[actual_idx]
                            bp = buy_row["open"]
                            if bp > 0:
                                buy_price = bp
                                buy_date = buy_row["trade_date"]
                                holding = True
                                hold_days = 0
        
        all_s8_old.extend(old_trades)
        all_s8_new.extend(new_trades)
        print_comparison(name, old_trades, new_trades)
    
    # S8汇总
    print("\n  ─────────────────────────────────────")
    print_comparison("S8 深跌反弹 (合计)", all_s8_old, all_s8_new)
    
    # --- S7: KDJ超卖反弹 ---
    print("\n" + "=" * 80)
    print("【S7 KDJ超卖反弹 - 英维克】")
    print("=" * 80)
    
    all_s7_old, all_s7_new = [], []
    for ts_code, name in S7_STOCKS.items():
        print(f"\n  获取 {name}({ts_code}) 数据...")
        df = fetch_data(ts_code, "20260401", "20260531")
        if df is None or len(df) < 30:
            print("    [跳过] 数据不足")
            continue
        df = add_indicators(df)
        
        # 回测
        old_trades, new_trades = [], []
        
        for version, trades_list in [("OLD", old_trades), ("NEW", new_trades)]:
            holding = False
            buy_price = 0
            buy_date = None
            hold_days = 0
            max_hold = 5
            
            df_start = df.index[df["trade_date"] >= "2026-04-01"]
            if len(df_start) == 0:
                continue
            start_idx = df_start[0]
            
            for idx in range(start_idx, len(df)):
                row = df.iloc[idx]
                if holding:
                    hold_days += 1
                    if hold_days >= max_hold:
                        sp = row["open"]
                        pnl = (sp - buy_price) / buy_price * 100
                        trades_list.append({
                            "buy_date": buy_date, "sell_date": row["trade_date"],
                            "buy_price": buy_price, "sell_price": sp,
                            "pnl": pnl, "exit": "到期",
                        })
                        holding = False
                else:
                    buy = s7_signal_old(df, idx)
                    if buy and version == "NEW":
                        pass_filter, _ = apply_new_filters(df, idx, name)
                        if not pass_filter:
                            buy = False
                    if buy:
                        actual_idx = idx + 1
                        if actual_idx < len(df):
                            buy_row = df.iloc[actual_idx]
                            bp = buy_row["open"]
                            if bp > 0:
                                buy_price = bp
                                buy_date = buy_row["trade_date"]
                                holding = True
                                hold_days = 0
        
        all_s7_old.extend(old_trades)
        all_s7_new.extend(new_trades)
        print_comparison(name, old_trades, new_trades)
    
    print("\n  ─────────────────────────────────────")
    print_comparison("S7 KDJ超卖反弹 (合计)", all_s7_old, all_s7_new)
    
    # --- S6: 多因子评分超卖 ---
    print("\n" + "=" * 80)
    print("【S6 多因子评分超卖】")
    print("=" * 80)
    
    all_s6_old, all_s6_new = [], []
    for ts_code, name in S6_STOCKS.items():
        print(f"\n  获取 {name}({ts_code}) 数据...")
        df = fetch_data(ts_code, "20260401", "20260531")
        if df is None or len(df) < 30:
            print("    [跳过] 数据不足")
            continue
        df = add_indicators(df)
        
        old_trades, new_trades = [], []
        
        for version, trades_list in [("OLD", old_trades), ("NEW", new_trades)]:
            holding = False
            buy_price = 0
            buy_date = None
            hold_days = 0
            max_hold = 5
            
            df_start = df.index[df["trade_date"] >= "2026-04-01"]
            if len(df_start) == 0:
                continue
            start_idx = df_start[0]
            
            for idx in range(start_idx, len(df)):
                row = df.iloc[idx]
                if holding:
                    hold_days += 1
                    if hold_days >= max_hold:
                        sp = row["open"]
                        pnl = (sp - buy_price) / buy_price * 100
                        trades_list.append({
                            "buy_date": buy_date, "sell_date": row["trade_date"],
                            "buy_price": buy_price, "sell_price": sp,
                            "pnl": pnl, "exit": "到期",
                        })
                        holding = False
                else:
                    buy, _ = s6_score_old(df, idx)
                    if buy and version == "NEW":
                        pass_filter, _ = apply_new_filters(df, idx, name)
                        if not pass_filter:
                            buy = False
                    if buy:
                        actual_idx = idx + 1
                        if actual_idx < len(df):
                            buy_row = df.iloc[actual_idx]
                            bp = buy_row["open"]
                            if bp > 0:
                                buy_price = bp
                                buy_date = buy_row["trade_date"]
                                holding = True
                                hold_days = 0
        
        all_s6_old.extend(old_trades)
        all_s6_new.extend(new_trades)
        print_comparison(name, old_trades, new_trades)
    
    print("\n  ─────────────────────────────────────")
    print_comparison("S6 多因子评分超卖 (合计)", all_s6_old, all_s6_new)
    
    # === 总体汇总 ===
    print("\n" + "=" * 100)
    print("【总体汇总】")
    print("=" * 100)
    print_comparison("全部跌超反弹策略 (S6+S7+S8)", all_s6_old + all_s7_old + all_s8_old,
                     all_s6_new + all_s7_new + all_s8_new)
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
