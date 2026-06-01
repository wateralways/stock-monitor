#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比不同大单流向过滤条件的效果 (2026年3月~6月)"""

import json, os, sys, io, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

TOKEN = os.environ.get("TUSHARE_TOKEN", "")
import tushare as ts
ts.set_token(TOKEN)
pro = ts.pro_api()

# ===== 数据加载 =====
with open("moneyflow_cache.json") as f:
    mf_cache = json.load(f)

ALL_STOCKS = {
    "300696.SZ": "爱乐达","000697.SZ": "ST炼石","002928.SZ": "华夏航空",
    "002831.SZ": "裕同科技","600486.SH": "扬农化工","300627.SZ": "华测导航",
    "002272.SZ": "川润股份","300499.SZ": "高澜股份","002218.SZ": "拓日新能",
    "603912.SH": "佳力图","000682.SZ": "东方电子","300572.SZ": "安车检测",
    "002837.SZ": "英维克","600418.SH": "江淮汽车","688223.SH": "晶科能源",
}
STRATEGY_STOCKS = {
    "策略1": ["300696.SZ","000697.SZ","002928.SZ"],
    "策略2": ["000697.SZ"],
    "策略4": ["002831.SZ","600486.SH","300627.SZ","002272.SZ","300499.SZ","002218.SZ",
              "603912.SH","000682.SZ","000697.SZ"],
    "策略6": ["002218.SZ","300572.SZ"],
    "策略7": ["002837.SZ"],
    "策略8": ["300499.SZ","002272.SZ","600418.SH","300696.SZ","300572.SZ","688223.SH"],
}

# ===== 获取日线 =====
print("获取日线数据...", flush=True)
daily_data = {}
for c in ALL_STOCKS:
    try:
        df = pro.daily(ts_code=c, start_date="20251201", end_date="20260601")
        if df is not None and not df.empty:
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["pct_chg"] = df["close"].pct_change() * 100
            daily_data[c] = df
    except: pass
print(f"  日线: {len(daily_data)}只", flush=True)

# ===== 技术指标 (复用backtest逻辑) =====
def calc_rsi(prices, w=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    return 100 - (100/(1+gain/loss.replace(0,np.nan)))

def calc_bollinger(prices, w=20):
    ma = prices.rolling(w).mean()
    std = prices.rolling(w).std()
    return ma+2*std, ma, ma-2*std

def calc_kdj(df, n=9):
    lm = df["low"].rolling(n).min()
    hm = df["high"].rolling(n).max()
    rsv = (df["close"]-lm)/(hm-lm)*100
    k = rsv.ewm(com=2,adjust=False).mean()
    d = k.ewm(com=2,adjust=False).mean()
    return k, d, 3*k-2*d

def calc_consec(df, direction="down"):
    r = pd.Series(0,index=df.index)
    for i in range(1,len(df)):
        if direction=="down" and df["pct_chg"].iloc[i]<0: r.iloc[i]=r.iloc[i-1]+1
        elif direction=="up" and df["pct_chg"].iloc[i]>0: r.iloc[i]=r.iloc[i-1]+1
        else: r.iloc[i]=0
    return r

def adp_rsi_consec(df_s):
    v20=df_s["pct_chg"].rolling(20).std().iloc[-1]
    v60=df_s["pct_chg"].rolling(60).std().iloc[-1]
    v20=v20 if pd.notna(v20) else 3; v60=v60 if pd.notna(v60) else 3
    vr=v20/v60 if v60>0 else 1
    return float(np.clip(35-(vr-1)*5,22,40))

# ===== 信号检测 (返回信号日期的idx) =====
def detect_signals(code, stock_name):
    """返回: [(signal_date, strategy_name, idx_in_df), ...]"""
    if code not in daily_data: return []
    df = daily_data[code]
    mf_records = mf_cache.get(code, [])
    mf_map = {}
    for r in mf_records:
        try: mf_map[pd.Timestamp(r["date"])] = r["big_net_amount"]
        except: pass

    # Add moneyflow to df
    df = df.copy()
    df["big_net"] = df["trade_date"].map(mf_map)
    df["big_net_1d"] = df["big_net"].shift(1)
    df["big_net_2d"] = df["big_net"].shift(2)
    df["big_net_3d_sum"] = df["big_net"].rolling(3).sum()
    df["big_net_5d_sum"] = df["big_net"].rolling(5).sum()
    df["pos3"] = (df["big_net"]>0).rolling(3).sum()
    df["pos5"] = (df["big_net"]>0).rolling(5).sum()
    df["neg3"] = (df["big_net"]<0).rolling(3).sum()
    df["outflow_shrinking"] = ((df["big_net"]<0)&(df["big_net"]>df["big_net_1d"])).astype(int)
    df["inflow_accelerating"] = ((df["big_net"]>0)&(df["big_net"]>df["big_net_1d"])).astype(int)
    df["net_improving"] = (df["big_net"]>df["big_net_1d"]).astype(int)

    signals = []
    test_mask = (df["trade_date"]>=pd.Timestamp("2026-03-01"))&(df["trade_date"]<=pd.Timestamp("2026-06-01"))
    test_idx = df[test_mask].index

    for idx in test_idx:
        if idx < 30: continue
        df_s = df.iloc[:idx+1]
        latest = df_s.iloc[-1]
        prices = df_s["close"]
        rsi14 = calc_rsi(prices,14).iloc[-1]
        _,_,bb_lower = calc_bollinger(prices,20)
        bb_pos = (latest["close"]-bb_lower.iloc[-1])/(2*prices.rolling(20).std().iloc[-1]) if prices.rolling(20).std().iloc[-1]>0 else 0.5

        # --- Strategy 1 ---
        if code in STRATEGY_STOCKS["策略1"]:
            is_st = "ST" in stock_name
            rsi_th = 33 if is_st else float(np.clip(33-(df_s["pct_chg"].rolling(20).std().iloc[-1]/max(df_s["pct_chg"].rolling(60).std().iloc[-1],0.01)-1)*6,18,45))
            bb_th = 0.5 if is_st else float(np.clip(0.5-(df_s["pct_chg"].rolling(20).std().iloc[-1]/max(df_s["pct_chg"].rolling(60).std().iloc[-1],0.01)-1)*0.2,0.05,0.6))
            if ((pd.notna(rsi14) and rsi14<rsi_th) or (pd.notna(bb_pos) and bb_pos<bb_th)) and latest["pct_chg"]>0 and latest["close"]>latest["open"]:
                signals.append((latest["trade_date"], stock_name, "策略1-RSI+布林带", idx))

        # --- Strategy 2 ---
        if code in STRATEGY_STOCKS["策略2"]:
            if idx >= 3+5:
                sig_day = df.iloc[idx-3]
                k,d,j = calc_kdj(df_s)
                sig_j = j.iloc[-4]
                sig_ma20 = prices.rolling(20).mean().iloc[-4]
                rise = (latest["close"]-sig_day["close"])/sig_day["close"]*100
                ma_ok = pd.notna(sig_ma20) and abs(sig_day["close"]-sig_ma20)/sig_ma20<0.02 and sig_day["pct_chg"]>0
                kdj_ok = pd.notna(sig_j) and sig_j<30
                if (ma_ok or kdj_ok) and rise<2:
                    signals.append((latest["trade_date"], stock_name, "策略2-MA+KDJ", idx))

        # --- Strategy 4 ---
        if code in STRATEGY_STOCKS["策略4"]:
            consec = calc_consec(df_s)
            rsi_th = 35 if stock_name in ("扬农化工","拓日新能","佳力图") else adp_rsi_consec(df_s)
            if pd.notna(rsi14) and rsi14<=rsi_th and consec.iloc[-1]>=2:
                signals.append((latest["trade_date"], stock_name, "策略4-RSI+连跌", idx))

        # --- Strategy 6 ---
        if code in STRATEGY_STOCKS["策略6"]:
            score = 0
            if pd.notna(rsi14):
                if rsi14<25: score+=25
                elif rsi14<30: score+=20
                elif rsi14<35: score+=15
                elif rsi14<40: score+=10
                elif rsi14<45: score+=5
            bb = (latest["close"]-bb_lower.iloc[-1])/(2*prices.rolling(20).std().iloc[-1]) if prices.rolling(20).std().iloc[-1]>0 else 0.5
            if bb<0.1: score+=20
            elif bb<0.2: score+=16
            elif bb<0.3: score+=12
            elif bb<0.4: score+=8
            elif bb<0.5: score+=4
            k,d,j = calc_kdj(df_s)
            jv = j.iloc[-1]
            if pd.notna(jv):
                if jv<0: score+=15
                elif jv<10: score+=12
                elif jv<20: score+=8
                elif jv<30: score+=4
            cd = calc_consec(df_s).iloc[-1]
            if cd>=4: score+=15
            elif cd>=3: score+=12
            elif cd>=2: score+=8
            if len(df_s)>=6:
                ret5d=(latest["close"]/df_s["close"].iloc[-6]-1)*100
                if ret5d<-10: score+=10
                elif ret5d<-7: score+=8
                elif ret5d<-5: score+=6
                elif ret5d<-3: score+=3
            if score>=50:
                signals.append((latest["trade_date"], stock_name, "策略6-评分超卖", idx))

        # --- Strategy 7 ---
        if code in STRATEGY_STOCKS["策略7"]:
            k,d,j = calc_kdj(df_s)
            if pd.notna(j.iloc[-1]) and j.iloc[-1]<10 and latest["pct_chg"]>0:
                signals.append((latest["trade_date"], stock_name, "策略7-KDJ超卖", idx))

        # --- Strategy 8 ---
        if code in STRATEGY_STOCKS["策略8"]:
            ret5d = (latest["close"]/df_s["close"].iloc[-6]-1)*100 if len(df_s)>=6 else 0
            ret5d_prev = (df_s["close"].iloc[-2]/df_s["close"].iloc[-7]-1)*100 if len(df_s)>=7 else 0
            sig_a = ret5d<-5 and pd.notna(rsi14) and rsi14<40 and latest["pct_chg"]>0
            sig_b = ret5d<-10 and stock_name in ("高澜股份","江淮汽车","爱乐达","安车检测","晶科能源")
            if sig_a or sig_b:
                signals.append((latest["trade_date"], stock_name, "策略8-深跌反弹", idx))

    return signals, df  # return df with moneyflow columns

# ===== 收集所有信号 =====
print("检测策略信号...", flush=True)
all_signals = []
all_dfs = {}
for code, name in ALL_STOCKS.items():
    sigs, df = detect_signals(code, name)
    all_signals.extend(sigs)
    all_dfs[code] = df

print(f"  总信号: {len(all_signals)}", flush=True)

# ===== 模拟交易 =====
def simulate_trade(df, sig_idx, stock_name, strategy_name):
    """买入卖出的时序规则"""
    t0_best = {("高澜股份","RSI+连跌"),("裕同科技","RSI+连跌"),("扬农化工","RSI+连跌"),
               ("华测导航","RSI+连跌"),("川润股份","RSI+连跌"),("ST炼石","RSI+连跌"),
               ("拓日新能","评分超卖"),("英维克","KDJ超卖"),("高澜股份","深跌反弹"),
               ("爱乐达","RSI+布林带"),("爱乐达","深跌反弹"),("华夏航空","RSI+布林带"),
               ("东方电子","RSI+连跌"),("晶科能源","深跌反弹")}
    t4_sell = {("川润股份","深跌反弹"),("晶科能源","深跌反弹"),("安车检测","评分超卖")}

    s_short = strategy_name.split("-")[-1] if "-" in strategy_name else strategy_name
    # Map back to the full strategy base name
    strat_base = strategy_name.replace("策略1-","").replace("策略2-","").replace("策略4-","").replace("策略6-","").replace("策略7-","").replace("策略8-","")

    buy_off = 0 if (stock_name, s_short) in t0_best or any((stock_name,b) in t0_best for b in [strat_base, s_short]) else 1
    # Simplified: use close prices
    if stock_name == "ST炼石" and "MA+KDJ" in strategy_name:
        buy_off, sell_off = 1, 6
    elif (stock_name, s_short) in t4_sell:
        sell_off = 4
        buy_off = 0 if (stock_name, s_short) in t0_best else 1
    elif any((stock_name, b) in t0_best for b in [strat_base]):
        buy_off, sell_off = 0, 5
    else:
        buy_off, sell_off = 1, 6

    buy_idx = sig_idx + buy_off
    sell_idx = sig_idx + sell_off
    if buy_idx >= len(df) or sell_idx >= len(df):
        return None

    buy_price = df.iloc[sig_idx]["close"] if buy_off==0 else df.iloc[buy_idx]["open"]
    sell_price = df.iloc[sell_idx]["close"]
    pnl = (sell_price-buy_price)/buy_price*100
    return {"pnl": round(pnl,2), "win": pnl>0, "buy_date": df.iloc[buy_idx]["trade_date"],
            "sell_date": df.iloc[sell_idx]["trade_date"], "buy_price": buy_price, "sell_price": sell_price}

# ===== 评估不同过滤条件 =====
print("评估过滤条件...\n", flush=True)

# 定义过滤条件
FILTERS = {
    "无过滤(原始)": lambda row: True,
    "A:近3天均净流入(原方案)": lambda row: row["pos3"] >= 3,
    "B:近3天≥2天净流入": lambda row: row["pos3"] >= 2,
    "C:近3天≥1天净流入": lambda row: row["pos3"] >= 1,
    "D:今日大单净流入": lambda row: row["big_net"] > 0,
    "E:3日累计净流入>0": lambda row: row["big_net_3d_sum"] > 0,
    "F:5日累计净流入>0": lambda row: row["big_net_5d_sum"] > 0,
    "G:非连续3天净流出": lambda row: row["neg3"] < 3,
    "H:今日非极端流出(>-5000万)": lambda row: row["big_net"] > -5000,
    "I:今日非极端流出(>-1亿)": lambda row: row["big_net"] > -10000,
    "J:流出在收窄+今日非巨量流出": lambda row: (row["outflow_shrinking"] == 1) or (row["big_net"] > 0),
    "K:3天净流入天数≥1 且 今日非极端流出": lambda row: row["pos3"] >= 1 and row["big_net"] > -5000,
    "L:3日累计>0 或 今日净流入": lambda row: (row["big_net_3d_sum"] > 0) or (row["big_net"] > 0),
}

results = {}
for fname, fcond in FILTERS.items():
    trades = []
    for sig_date, stock_name, strategy_name, sig_idx in all_signals:
        code = [c for c,n in ALL_STOCKS.items() if n==stock_name][0]
        df = all_dfs[code]
        row = df.iloc[sig_idx]
        if pd.isna(row.get("big_net")):
            continue  # skip if no moneyflow data
        if not fcond(row):
            continue
        trade = simulate_trade(df, sig_idx, stock_name, strategy_name)
        if trade:
            trades.append({**trade, "stock": stock_name, "strategy": strategy_name,
                          "sig_date": sig_date, "big_net_3": [row.get("big_net_2d"), row.get("big_net_1d"), row.get("big_net")]})

    if not trades:
        results[fname] = {"n": 0, "win": 0, "loss": 0, "wr": 0, "cum_pnl": 0, "avg_pnl": 0}
        continue

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    pnls = [t["pnl"] for t in trades]
    results[fname] = {
        "n": len(trades), "win": len(wins), "loss": len(losses),
        "wr": round(len(wins)/len(trades)*100, 1),
        "cum_pnl": round(sum(pnls), 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "avg_win": round(np.mean([t["pnl"] for t in wins]),2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses]),2) if losses else 0,
    }

# ===== 输出 =====
print(f"{'='*100}")
print(f"【不同大单过滤条件效果对比】2026年3月~6月")
print(f"{'='*100}\n")
print(f"{'过滤条件':<36} {'笔数':>5} {'胜率':>7} {'累计盈亏':>10} {'平均盈亏':>8} {'均盈利':>7} {'均亏损':>7}")
print(f"{'-'*90}")

# Sort: best cumulative P&L first
sorted_results = sorted(results.items(), key=lambda x: x[1]["cum_pnl"], reverse=True)
for fname, r in sorted_results:
    print(f"{fname:<36} {r['n']:>5} {r['wr']:>6.1f}% {r['cum_pnl']:>+9.2f}% {r['avg_pnl']:>+7.2f}% {r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}%")

# ===== 详细分析最优方案 =====
print(f"\n{'='*100}")
print("【深度分析：最优过滤方案】")
print(f"{'='*100}")

# Find top 3 by cum_pnl
top3 = sorted_results[:3]
for rank, (fname, r) in enumerate(top3, 1):
    print(f"\n--- 第{rank}名: {fname} ---")
    print(f"  笔数={r['n']}, 胜率={r['wr']}%, 累计={r['cum_pnl']:+.2f}%, 平均={r['avg_pnl']:+.2f}%")

    # Show what got filtered (for filters vs no-filter)
    if fname != "无过滤(原始)":
        base_trades = sorted_results[0][1]  # no filter trades
        # Actually, let's compute it properly
        base_n = results["无过滤(原始)"]["n"]
        filtered_out_n = base_n - r['n']
        if filtered_out_n > 0:
            # Get filtered trades details
            base_set = set()
            fcond = FILTERS[fname]
            all_trades_detail = []
            for sig_date, stock_name, strategy_name, sig_idx in all_signals:
                code = [c for c,n in ALL_STOCKS.items() if n==stock_name][0]
                df = all_dfs[code]
                row = df.iloc[sig_idx]
                if pd.isna(row.get("big_net")): continue
                trade = simulate_trade(df, sig_idx, stock_name, strategy_name)
                if trade:
                    passed = fcond(row)
                    all_trades_detail.append({**trade, "passed": passed, "stock": stock_name,
                        "sig_date": sig_date, "strategy": strategy_name})

            filtered_out = [t for t in all_trades_detail if not t["passed"]]
            fo_wins = sum(1 for t in filtered_out if t["win"])
            fo_losses = sum(1 for t in filtered_out if not t["win"])
            fo_pnl = sum(t["pnl"] for t in filtered_out)
            print(f"  过滤掉{filtered_out_n}笔: 盈利{fo_wins}笔/亏损{fo_losses}笔, 累计{fo_pnl:+.2f}%")
            kept = [t for t in all_trades_detail if t["passed"]]
            kept_wins = sum(1 for t in kept if t["win"])
            kept_losses = sum(1 for t in kept if not t["win"])
            kept_pnl = sum(t["pnl"] for t in kept)
            print(f"  保留{len(kept)}笔: 盈利{kept_wins}笔/亏损{kept_losses}笔, 累计{kept_pnl:+.2f}%")

print(f"\n{'='*100}")
print("分析完成")
print(f"{'='*100}")
