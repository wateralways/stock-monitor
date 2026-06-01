#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大单净流向过滤回测
对策略1/2/4/6/7/8加入"大单+超大单至少连续3天净流入"条件
回测时间: 2026年3月1日 ~ 2026年6月1日

对比:
  A) 原始策略 (不加过滤)
  B) 新策略 (加连续3天大单净流入过滤)
"""

import sys
import io
import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ========== Tushare 初始化 ==========
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not TUSHARE_TOKEN:
    print("[ERROR] TUSHARE_TOKEN not set")
    sys.exit(1)

import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ========== 所有涉及的股票 ==========
ALL_STOCKS = {
    # 策略1: RSI+布林带均值回归
    "300696.SZ": {"name": "爱乐达"},
    "000697.SZ": {"name": "ST炼石"},
    "002928.SZ": {"name": "华夏航空"},
    # 策略2: MA支撑+KDJ超卖 (ST炼石已在上面)
    # 策略4: RSI+连跌中等信号
    "002831.SZ": {"name": "裕同科技"},
    "600486.SH": {"name": "扬农化工"},
    "300627.SZ": {"name": "华测导航"},
    "002272.SZ": {"name": "川润股份"},
    "300499.SZ": {"name": "高澜股份"},
    "002218.SZ": {"name": "拓日新能"},
    "603912.SH": {"name": "佳力图"},
    "000682.SZ": {"name": "东方电子"},
    # 策略6: 多因子评分超卖
    "300572.SZ": {"name": "安车检测"},
    # 策略7: KDJ超卖反弹
    "002837.SZ": {"name": "英维克"},
    # 策略8: 深跌反弹
    "600418.SH": {"name": "江淮汽车"},
    "688223.SH": {"name": "晶科能源"},
}

# 策略-股票映射
STRATEGY_STOCKS = {
    1: ["300696.SZ", "000697.SZ", "002928.SZ"],  # RSI+布林带均值回归
    2: ["000697.SZ"],                              # MA支撑+KDJ超卖
    4: ["002831.SZ", "600486.SH", "300627.SZ", "002272.SZ",
        "300499.SZ", "002218.SZ", "603912.SH", "000682.SZ", "000697.SZ"],  # RSI+连跌
    6: ["002218.SZ", "300572.SZ"],                 # 多因子评分超卖
    7: ["002837.SZ"],                               # KDJ超卖反弹
    8: ["300499.SZ", "002272.SZ", "600418.SH", "300696.SZ",
        "300572.SZ", "688223.SH"],                 # 深跌反弹
}

# 大单固定阈值策略 (不用自适应参数)
FIXED_THRESHOLD_STOCKS = {
    "ST炼石": {"rsi_entry": 33, "bb_entry": 0.5, "rsi_consec": 35},
    "扬农化工": {"rsi_consec": 35},
    "拓日新能": {"rsi_consec": 35},
    "佳力图": {"rsi_consec": 35},
}

# 策略8信号B适用范围 (5日跌>10%)
STRATEGY8_SIGNAL_B_STOCKS = {"高澜股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源"}

# ========== 技术指标 ==========
def calc_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bollinger(prices: pd.Series, window: int = 20, num_std: int = 2):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    bb_pos = (prices - lower) / (upper - lower)
    return upper, ma, lower, bb_pos

def calc_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def calc_consecutive_days(pct_chg: pd.Series, direction: str = "down") -> pd.Series:
    result = pd.Series(0, index=pct_chg.index)
    for i in range(1, len(pct_chg)):
        if direction == "down" and pct_chg.iloc[i] < 0:
            result.iloc[i] = result.iloc[i - 1] + 1
        elif direction == "up" and pct_chg.iloc[i] > 0:
            result.iloc[i] = result.iloc[i - 1] + 1
        else:
            result.iloc[i] = 0
    return result

def calc_adaptive_rsi_entry(df_slice: pd.DataFrame) -> float:
    """简化自适应参数: 基于波动率计算RSI入场阈值"""
    vol_20 = df_slice["pct_chg"].rolling(20).std().iloc[-1]
    vol_60 = df_slice["pct_chg"].rolling(60).std().iloc[-1]
    vol_20 = vol_20 if pd.notna(vol_20) else 3.0
    vol_60 = vol_60 if pd.notna(vol_60) else 3.0
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    return float(np.clip(33 - (vol_ratio - 1.0) * 6, 18, 45))

def calc_adaptive_bb_entry(df_slice: pd.DataFrame) -> float:
    vol_20 = df_slice["pct_chg"].rolling(20).std().iloc[-1]
    vol_60 = df_slice["pct_chg"].rolling(60).std().iloc[-1]
    vol_20 = vol_20 if pd.notna(vol_20) else 3.0
    vol_60 = vol_60 if pd.notna(vol_60) else 3.0
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    return float(np.clip(0.5 - (vol_ratio - 1.0) * 0.2, 0.05, 0.6))

def calc_adaptive_rsi_consec(df_slice: pd.DataFrame) -> float:
    vol_20 = df_slice["pct_chg"].rolling(20).std().iloc[-1]
    vol_60 = df_slice["pct_chg"].rolling(60).std().iloc[-1]
    vol_20 = vol_20 if pd.notna(vol_20) else 3.0
    vol_60 = vol_60 if pd.notna(vol_60) else 3.0
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    return float(np.clip(35 - (vol_ratio - 1.0) * 5, 22, 40))


# ========== 数据获取 ==========
def fetch_daily(ts_code: str, start: str = "20260101", end: str = "20260601") -> Optional[pd.DataFrame]:
    """获取日线数据"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["pct_chg"] = df["close"].pct_change() * 100
        return df
    except Exception as e:
        print(f"  [ERROR] fetch_daily {ts_code}: {e}")
        return None

def fetch_moneyflow(ts_code: str, start: str = "20260101", end: str = "20260601") -> Optional[pd.DataFrame]:
    """获取每日资金流向数据 (大单+超大单)
    Tushare moneyflow接口限速1次/分钟，每次调用后需等待65秒
    """
    try:
        df = pro.moneyflow(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 大单+超大单 净流入量(万元)
        df["big_net_inflow"] = (
            (df.get("buy_lg_vol", 0) - df.get("sell_lg_vol", 0)) +
            (df.get("buy_elg_vol", 0) - df.get("sell_elg_vol", 0))
        ) / 100  # 股->手, 近似万元
        # 也可以用金额直接判断
        df["big_net_amount"] = (
            (df.get("buy_lg_amount", 0) - df.get("sell_lg_amount", 0)) +
            (df.get("buy_elg_amount", 0) - df.get("sell_elg_amount", 0))
        )  # 万元
        return df
    except Exception as e:
        print(f"  [ERROR] fetch_moneyflow {ts_code}: {e}")
        return None

def check_big_order_filter(mf_records: list, signal_date: pd.Timestamp, consecutive_days: int = 3) -> Tuple[bool, List[float]]:
    """
    检查信号日前连续N天(含信号日)是否大单+超大单均为净流入
    mf_records: [{date: str, big_net_amount: float}, ...] 按日期升序排列
    返回: (是否满足, 最近N天的净流入金额列表)
    """
    if not mf_records:
        return False, []

    # 转换为易查询的格式
    mf_dict = {}
    for r in mf_records:
        try:
            d = pd.Timestamp(r["date"])
            mf_dict[d] = r["big_net_amount"]
        except:
            continue

    if not mf_dict:
        return False, []

    # 找到信号日及之前的所有日期
    all_dates = sorted([d for d in mf_dict.keys() if d <= signal_date])
    if len(all_dates) < consecutive_days:
        return False, []

    # 取最近N天
    recent_dates = all_dates[-consecutive_days:]

    # 确保是连续交易日(间隔不超过4天, 考虑周末/节假日)
    for i in range(1, len(recent_dates)):
        if (recent_dates[i] - recent_dates[i-1]).days > 4:
            return False, []

    net_amounts = [mf_dict[d] for d in recent_dates]
    all_positive = all(a > 0 for a in net_amounts)
    return all_positive, net_amounts


# ========== 策略模拟 ==========
def simulate_strategy1(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略1: RSI+布林带均值回归"""
    if idx < 30:
        return None
    df_slice = df.iloc[:idx+1].copy()

    prices = df_slice["close"]
    rsi_series = calc_rsi(prices, 14)
    _, _, bb_lower, bb_pos = calc_bollinger(prices, 20, 2)

    today_rsi = rsi_series.iloc[-1]
    today_bb_pos = bb_pos.iloc[-1]
    today = df_slice.iloc[-1]

    is_st = "ST" in stock_name
    if is_st:
        rsi_th, bb_th = 33, 0.5
    else:
        rsi_th = calc_adaptive_rsi_entry(df_slice)
        bb_th = calc_adaptive_bb_entry(df_slice)

    buy_cond_rsi = pd.notna(today_rsi) and today_rsi < rsi_th
    buy_cond_bb = pd.notna(today_bb_pos) and today_bb_pos < bb_th
    buy_cond_up = today["pct_chg"] > 0
    buy_cond_yang = today["close"] > today["open"]

    signal = (buy_cond_rsi or buy_cond_bb) and buy_cond_up and buy_cond_yang

    # 趋势过滤 (简化: MA20不能明显下降)
    if not is_st:
        ma20_series = prices.rolling(20).mean()
        if len(ma20_series) >= 6:
            ma20_now = ma20_series.iloc[-1]
            ma20_5ago = ma20_series.iloc[-6]
            ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
            if ma20_slope < -1.0:
                signal = False

        ma60_series = prices.rolling(60).mean()
        if len(ma60_series) >= 60:
            ma60_now = ma60_series.iloc[-1]
            if pd.notna(ma60_now) and today["close"] < ma60_now * 0.98:
                signal = False

    if signal:
        return {
            "strategy": "策略1-RSI+布林带",
            "name": stock_name,
            "rsi": round(today_rsi, 1) if pd.notna(today_rsi) else None,
            "bb_pos": round(today_bb_pos, 3) if pd.notna(today_bb_pos) else None,
            "price": today["close"],
        }
    return None

def simulate_strategy2(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略2: MA支撑+KDJ超卖 (3天延迟入场)"""
    DELAY = 3
    THRESHOLD = 2  # 涨超2%放弃
    if idx < DELAY + 5:
        return None
    df_slice = df.iloc[:idx+1].copy()

    sig_day = df_slice.iloc[-1 - DELAY]
    latest = df_slice.iloc[-1]

    prices = df_slice["close"]
    ma20 = prices.rolling(20).mean()
    k, d, j = calc_kdj(df_slice)

    sig_signals = []
    sig_ma20 = ma20.iloc[-1 - DELAY]
    if pd.notna(sig_ma20) and abs(sig_day["close"] - sig_ma20) / sig_ma20 < 0.02 and sig_day["pct_chg"] > 0:
        sig_signals.append("MA20支撑")

    # 自适应KDJ阈值
    sig_slice = df.iloc[:idx+1-DELAY].copy()
    kdj_th = 30
    try:
        vol_20 = sig_slice["pct_chg"].rolling(20).std().iloc[-1]
        vol_60 = sig_slice["pct_chg"].rolling(60).std().iloc[-1]
        vol_20 = vol_20 if pd.notna(vol_20) else 3.0
        vol_60 = vol_60 if pd.notna(vol_60) else 3.0
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        kdj_th = float(np.clip(30 - vol_ratio * 10, 8, 35))
    except:
        pass

    sig_j = j.iloc[-1 - DELAY]
    if pd.notna(sig_j) and sig_j < kdj_th:
        sig_signals.append("KDJ超卖")

    sig_triggered = len(sig_signals) > 0
    rise = (latest["close"] - sig_day["close"]) / sig_day["close"] * 100
    not_risen = rise < THRESHOLD

    signal = sig_triggered and not_risen

    if signal:
        return {
            "strategy": "策略2-MA支撑+KDJ",
            "name": stock_name,
            "j_value": round(sig_j, 1) if pd.notna(sig_j) else None,
            "rise_from_sig": round(rise, 2),
            "price": latest["close"],
        }
    return None

def simulate_strategy4(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略4: RSI+连跌中等信号"""
    if idx < 20:
        return None
    df_slice = df.iloc[:idx+1].copy()

    prices = df_slice["close"]
    rsi_series = calc_rsi(prices, 14)
    consec = calc_consecutive_days(df_slice["pct_chg"], "down")

    today_rsi = rsi_series.iloc[-1]
    today_consec = consec.iloc[-1]
    latest = df_slice.iloc[-1]

    if stock_name in ("扬农化工", "拓日新能", "佳力图"):
        rsi_th = 35
    else:
        rsi_th = calc_adaptive_rsi_consec(df_slice)

    signal = pd.notna(today_rsi) and today_rsi <= rsi_th and today_consec >= 2

    # 趋势过滤
    if signal:
        ma60_series = prices.rolling(60).mean()
        if len(ma60_series) >= 60:
            ma60_now = ma60_series.iloc[-1]
            if pd.notna(ma60_now) and latest["close"] < ma60_now * 0.97:
                downtrend_exempt = stock_name in ("ST炼石", "拓日新能", "东方电子")
                if not downtrend_exempt:
                    signal = False

    if signal:
        return {
            "strategy": "策略4-RSI+连跌",
            "name": stock_name,
            "rsi": round(today_rsi, 1) if pd.notna(today_rsi) else None,
            "consecutive_down": int(today_consec),
            "price": latest["close"],
        }
    return None

def simulate_strategy6(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略6: 多因子评分超卖"""
    if idx < 30:
        return None
    df_slice = df.iloc[:idx+1].copy()
    latest = df_slice.iloc[-1]
    score = 0

    # RSI14
    rsi14 = calc_rsi(df_slice["close"], 14).iloc[-1]
    if pd.notna(rsi14):
        if rsi14 < 25: score += 25
        elif rsi14 < 30: score += 20
        elif rsi14 < 35: score += 15
        elif rsi14 < 40: score += 10
        elif rsi14 < 45: score += 5

    # 布林带位置
    _, _, _, bb_pos = calc_bollinger(df_slice["close"], 20, 2)
    bb = bb_pos.iloc[-1] if pd.notna(bb_pos.iloc[-1]) else 0.5
    if bb < 0.1: score += 20
    elif bb < 0.2: score += 16
    elif bb < 0.3: score += 12
    elif bb < 0.4: score += 8
    elif bb < 0.5: score += 4

    # KDJ J值
    k, d, j = calc_kdj(df_slice)
    j_val = j.iloc[-1]
    if pd.notna(j_val):
        if j_val < 0: score += 15
        elif j_val < 10: score += 12
        elif j_val < 20: score += 8
        elif j_val < 30: score += 4

    # 连续下跌
    consec = calc_consecutive_days(df_slice["pct_chg"], "down")
    cd = consec.iloc[-1]
    if cd >= 4: score += 15
    elif cd >= 3: score += 12
    elif cd >= 2: score += 8

    # 5日跌幅
    if len(df_slice) >= 6:
        ret5d = (df_slice["close"].iloc[-1] / df_slice["close"].iloc[-6] - 1) * 100
        if ret5d < -10: score += 10
        elif ret5d < -7: score += 8
        elif ret5d < -5: score += 6
        elif ret5d < -3: score += 3

    signal = score >= 50

    # 缩量下跌过滤
    if signal and len(df_slice) >= 20:
        vol_ma20 = df_slice["vol"].rolling(20).mean().iloc[-1]
        if pd.notna(vol_ma20) and vol_ma20 > 0:
            vol_shrink = df_slice["vol"].iloc[-5:].mean() / vol_ma20
            if vol_shrink > 1.1:
                signal = False

    if signal:
        return {
            "strategy": "策略6-评分超卖",
            "name": stock_name,
            "score": score,
            "rsi": round(rsi14, 1) if pd.notna(rsi14) else None,
            "price": latest["close"],
        }
    return None

def simulate_strategy7(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略7: KDJ超卖反弹"""
    if idx < 30:
        return None
    df_slice = df.iloc[:idx+1].copy()
    latest = df_slice.iloc[-1]

    k, d, j = calc_kdj(df_slice)
    j_val = j.iloc[-1]
    signal = pd.notna(j_val) and j_val < 10 and latest["pct_chg"] > 0

    # 缩量下跌过滤
    if signal and len(df_slice) >= 20:
        vol_ma20 = df_slice["vol"].rolling(20).mean().iloc[-1]
        if pd.notna(vol_ma20) and vol_ma20 > 0:
            vol_shrink = df_slice["vol"].iloc[-5:].mean() / vol_ma20
            if vol_shrink > 1.1:
                signal = False

    if signal:
        return {
            "strategy": "策略7-KDJ超卖",
            "name": stock_name,
            "j_value": round(j_val, 1) if pd.notna(j_val) else None,
            "price": latest["close"],
        }
    return None

def simulate_strategy8(df: pd.DataFrame, idx: int, stock_name: str) -> Optional[Dict]:
    """策略8: 深跌反弹"""
    if idx < 20:
        return None
    df_slice = df.iloc[:idx+1].copy()
    latest = df_slice.iloc[-1]

    rsi_series = calc_rsi(df_slice["close"], 14)
    rsi14 = rsi_series.iloc[-1]
    today_pct = latest["pct_chg"]

    # 5日跌幅
    ret5d = (latest["close"] / df_slice["close"].iloc[-6] - 1) * 100 if len(df_slice) >= 6 else 0
    prev_ret5d = (df_slice["close"].iloc[-2] / df_slice["close"].iloc[-7] - 1) * 100 if len(df_slice) >= 7 else 0

    signal_a = ret5d < -5 and pd.notna(rsi14) and rsi14 < 40 and today_pct > 0
    signal_b = ret5d < -10 and stock_name in STRATEGY8_SIGNAL_B_STOCKS

    # 爱乐达特殊处理
    if stock_name == "爱乐达":
        ma60 = df_slice["close"].rolling(60).mean().iloc[-1]
        ald_true_deep = (pd.notna(rsi14) and rsi14 <= 35) or (
            pd.notna(ma60) and latest["close"] <= ma60 * 1.08
        )
        if signal_b and prev_ret5d < -10:
            signal_b = False  # 同一轮急跌已触发过
        elif (signal_a or signal_b) and not ald_true_deep:
            signal_a = False
            signal_b = False

    signal = signal_a or signal_b

    # 下降趋势过滤
    if signal and len(df_slice) >= 60:
        ma60 = df_slice["close"].rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and latest["close"] < ma60 * 0.95:
            signal = False

    # 缩量下跌过滤
    if signal and len(df_slice) >= 20:
        vol_ma20 = df_slice["vol"].rolling(20).mean().iloc[-1]
        if pd.notna(vol_ma20) and vol_ma20 > 0:
            vol_shrink = df_slice["vol"].iloc[-5:].mean() / vol_ma20
            if vol_shrink > 1.1:
                signal = False

    if signal:
        sig_type = "A" if signal_a else "B"
        return {
            "strategy": "策略8-深跌反弹",
            "name": stock_name,
            "signal_type": sig_type,
            "ret5d": round(ret5d, 1),
            "rsi": round(rsi14, 1) if pd.notna(rsi14) else None,
            "price": latest["close"],
        }
    return None


# ========== 交易模拟 ==========
def get_strategy_timing(stock_name: str, strategy_num: int) -> Dict:
    """返回策略的买入/卖出时机"""
    strategy_names = {
        1: "RSI+布林带均值回归",
        2: "MA支撑+KDJ超卖",
        4: "RSI+连跌中等信号",
        6: "多因子评分超卖",
        7: "KDJ超卖反弹",
        8: "深跌反弹",
    }
    sname = strategy_names[strategy_num]

    # 复用原版逻辑
    t0_best = {
        ("高澜股份", "RSI+连跌中等信号"),
        ("裕同科技", "RSI+连跌中等信号"),
        ("扬农化工", "RSI+连跌中等信号"),
        ("华测导航", "RSI+连跌中等信号"),
        ("川润股份", "RSI+连跌中等信号"),
        ("拓日新能", "多因子评分超卖"),
        ("英维克", "KDJ超卖反弹"),
        ("高澜股份", "深跌反弹"),
        ("爱乐达", "RSI+布林带均值回归"),
        ("爱乐达", "深跌反弹"),
        ("华夏航空", "RSI+布林带均值回归"),
        ("东方电子", "RSI+连跌中等信号"),
        ("晶科能源", "深跌反弹"),
        ("ST炼石", "RSI+连跌中等信号"),
    }
    t4_sell = {
        ("川润股份", "深跌反弹"),
        ("晶科能源", "深跌反弹"),
        ("安车检测", "多因子评分超卖"),
    }

    if (stock_name, sname) in t4_sell:
        if (stock_name, sname) in t0_best:
            return {"buy": "T+0", "sell": "T+4", "buy_offset": 0, "sell_offset": 4}
        else:
            return {"buy": "T+1", "sell": "T+4", "buy_offset": 1, "sell_offset": 4}
    if (stock_name, sname) in t0_best:
        return {"buy": "T+0", "sell": "T+5", "buy_offset": 0, "sell_offset": 5}
    # ST炼石 MA+KDJ: T+1/T+6
    if stock_name == "ST炼石" and sname == "MA支撑+KDJ超卖":
        return {"buy": "T+1", "sell": "T+6", "buy_offset": 1, "sell_offset": 6}
    return {"buy": "T+1", "sell": "T+6", "buy_offset": 1, "sell_offset": 6}

def simulate_trade(df: pd.DataFrame, signal_idx: int, timing: Dict) -> Optional[Dict]:
    """模拟一笔交易: 在signal_idx这天触发信号, 按timing买入卖出"""
    buy_offset = timing["buy_offset"]  # 0=T+0尾盘, 1=T+1开盘
    sell_offset = timing["sell_offset"]

    buy_idx = signal_idx + buy_offset
    sell_idx = signal_idx + sell_offset

    if buy_idx >= len(df) or sell_idx >= len(df):
        return None  # 数据不足

    signal_row = df.iloc[signal_idx]
    buy_row = df.iloc[buy_idx]
    sell_row = df.iloc[sell_idx]

    if buy_offset == 0:
        buy_price = signal_row["close"]  # T+0 尾盘买入, 用当天收盘价
    else:
        buy_price = buy_row["open"]  # T+1 开盘买入

    sell_price = sell_row["close"]  # 尾盘卖出

    pnl_pct = (sell_price - buy_price) / buy_price * 100
    is_win = pnl_pct > 0

    return {
        "signal_date": signal_row["trade_date"],
        "buy_date": buy_row["trade_date"],
        "sell_date": sell_row["trade_date"],
        "buy_price": buy_price,
        "sell_price": sell_price,
        "pnl_pct": round(pnl_pct, 2),
        "win": is_win,
    }


# ========== 主回测逻辑 ==========
def run_backtest():
    print("=" * 100)
    print("大单净流向过滤回测")
    print(f"回测时间: 2026-03-01 ~ 2026-06-01")
    print(f"测试策略: 1, 2, 4, 6, 7, 8")
    print(f"过滤条件: 大单+超大单连续3天净流入(金额>0)")
    print("=" * 100)

    # ===== 1. 获取数据 =====
    print("\n[1/4] 获取日线数据...")
    daily_data = {}
    for code, info in ALL_STOCKS.items():
        print(f"  获取 {info['name']}({code}) 日线...", end=" ")
        df = fetch_daily(code, "20251201", "20260601")
        if df is not None and len(df) > 30:
            daily_data[code] = df
            print(f"{len(df)}条")
        else:
            print("失败")

    print(f"\n  日线数据: {len(daily_data)}/{len(ALL_STOCKS)} 只股票")

    print("\n[2/4] 加载资金流向缓存数据...")
    moneyflow_data = {}
    CACHE_FILE = "moneyflow_cache.json"
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        for code, records in cache.items():
            if len(records) > 10:
                moneyflow_data[code] = records
                big_pos = sum(1 for r in records if r["big_net_amount"] > 0)
                print(f"  {ALL_STOCKS.get(code, {}).get('name', code)}({code}): {len(records)}条 (净流入{big_pos}天)")
    else:
        print(f"  [WARN] 缓存文件 {CACHE_FILE} 不存在，尝试从Tushare获取...")
        # 回退到Tushare (带限速)
        mf_call_count = 0
        for code, info in ALL_STOCKS.items():
            if code not in daily_data:
                continue
            if mf_call_count > 0:
                time.sleep(65)
            print(f"  获取 {info['name']}({code}) 资金流向...", end=" ")
            mf = fetch_moneyflow(code, "20251201", "20260601")
            mf_call_count += 1
            if mf is not None and len(mf) > 10:
                moneyflow_data[code] = mf
                print(f"{len(mf)}条")
            else:
                print("无数据")

    if not moneyflow_data:
        print("[ERROR] 没有资金流向数据, 无法回测。请先运行 cache_moneyflow_eastmoney.py 或等待Tushare缓存完成。")
        return

    # ===== 3. 逐日扫描信号 =====
    print("\n[3/4] 逐日扫描策略信号...")

    # 收集所有信号
    all_signals = []  # [(date, code, stock_name, strategy_num, signal_info, big_order_ok)]

    for strategy_num, stocks in STRATEGY_STOCKS.items():
        strategy_func = {
            1: simulate_strategy1,
            2: simulate_strategy2,
            4: simulate_strategy4,
            6: simulate_strategy6,
            7: simulate_strategy7,
            8: simulate_strategy8,
        }[strategy_num]

        strategy_name = {
            1: "策略1-RSI+布林带",
            2: "策略2-MA+KDJ",
            4: "策略4-RSI+连跌",
            6: "策略6-评分超卖",
            7: "策略7-KDJ超卖",
            8: "策略8-深跌反弹",
        }[strategy_num]

        print(f"\n  {strategy_name}:")

        for code in stocks:
            if code not in daily_data:
                print(f"    {ALL_STOCKS.get(code, {}).get('name', code)}: 无日线数据")
                continue

            df = daily_data[code]
            mf_records = moneyflow_data.get(code)
            stock_name = ALL_STOCKS[code]["name"]

            # 只回测2026年3月~6月的交易日
            mask = (df["trade_date"] >= pd.Timestamp("2026-03-01")) & (df["trade_date"] <= pd.Timestamp("2026-06-01"))
            test_indices = df[mask].index

            signals_found = 0
            for idx in test_indices:
                if idx < 30:  # 需要足够的历史数据
                    continue
                signal = strategy_func(df, idx, stock_name)
                if signal is not None:
                    signal_date = df.iloc[idx]["trade_date"]
                    # 检查大单过滤
                    big_ok, net_amounts = check_big_order_filter(mf_records, signal_date, 3)
                    signals_found += 1
                    all_signals.append({
                        "date": signal_date,
                        "code": code,
                        "stock_name": stock_name,
                        "strategy_num": strategy_num,
                        "strategy_name": strategy_name,
                        "idx": idx,
                        "signal": signal,
                        "big_order_ok": big_ok,
                        "net_amounts": net_amounts,
                    })

            status = f"{signals_found}个信号" if signals_found > 0 else "无信号"
            mf_status = "有资金流" if mf_records else "无资金流"
            print(f"    {stock_name}({code}): {status} ({mf_status})")

    print(f"\n  总信号数: {len(all_signals)}")

    # ===== 4. 模拟交易并对比 =====
    print("\n[4/4] 模拟交易并对比...")

    # 分组: 原始 vs 过滤后
    original_trades = []  # 所有原始信号模拟的交易
    filtered_trades = []  # 经过大单过滤后的交易

    for sig in all_signals:
        code = sig["code"]
        df = daily_data[code]
        stock_name = sig["stock_name"]
        strategy_num = sig["strategy_num"]
        timing = get_strategy_timing(stock_name, strategy_num)

        trade = simulate_trade(df, sig["idx"], timing)
        if trade is None:
            continue

        trade_info = {
            **trade,
            "stock_name": stock_name,
            "code": code,
            "strategy_name": sig["strategy_name"],
            "strategy_num": strategy_num,
            "big_order_ok": sig["big_order_ok"],
            "net_amounts": sig["net_amounts"],
            "signal_detail": sig["signal"],
        }

        original_trades.append(trade_info)
        if sig["big_order_ok"]:
            filtered_trades.append(trade_info)

    # ===== 5. 输出对比结果 =====
    print("\n" + "=" * 100)
    print("【回测结果对比】")
    print("=" * 100)

    def calc_stats(trades: List[Dict]) -> Dict:
        if not trades:
            return {"count": 0, "wins": 0, "losses": 0, "wr": 0, "cum_pnl": 0,
                    "avg_pnl": 0, "avg_win": 0, "avg_loss": 0, "max_win": 0, "max_loss": 0}
        wins = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]
        pnls = [t["pnl_pct"] for t in trades]
        return {
            "count": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "wr": len(wins) / len(trades) * 100,
            "cum_pnl": sum(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "avg_win": float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0,
            "avg_loss": float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0,
            "max_win": max(pnls),
            "max_loss": min(pnls),
        }

    orig_stats = calc_stats(original_trades)
    filt_stats = calc_stats(filtered_trades)

    print(f"\n{'指标':<20} {'原始策略':>15} {'+大单过滤':>15} {'改善':>15}")
    print("-" * 70)
    print(f"{'交易笔数':<20} {orig_stats['count']:>15} {filt_stats['count']:>15} {filt_stats['count'] - orig_stats['count']:>+15}")
    print(f"{'胜率':<20} {orig_stats['wr']:>14.1f}% {filt_stats['wr']:>14.1f}% {filt_stats['wr'] - orig_stats['wr']:>+14.1f}%")
    print(f"{'累计盈亏':<20} {orig_stats['cum_pnl']:>+14.2f}% {filt_stats['cum_pnl']:>+14.2f}% {filt_stats['cum_pnl'] - orig_stats['cum_pnl']:>+14.2f}%")
    print(f"{'平均盈亏':<20} {orig_stats['avg_pnl']:>+14.2f}% {filt_stats['avg_pnl']:>+14.2f}% {filt_stats['avg_pnl'] - orig_stats['avg_pnl']:>+14.2f}%")
    print(f"{'平均盈利':<20} {orig_stats['avg_win']:>+14.2f}% {filt_stats['avg_win']:>+14.2f}%")
    print(f"{'平均亏损':<20} {orig_stats['avg_loss']:>+14.2f}% {filt_stats['avg_loss']:>+14.2f}%")
    print(f"{'最大盈利':<20} {orig_stats['max_win']:>+14.2f}% {filt_stats['max_win']:>+14.2f}%")
    print(f"{'最大亏损':<20} {orig_stats['max_loss']:>+14.2f}% {filt_stats['max_loss']:>+14.2f}%")

    # 按策略细分
    print(f"\n{'='*100}")
    print("【按策略细分对比】")
    print(f"{'='*100}")

    for snum in [1, 2, 4, 6, 7, 8]:
        sname = {
            1: "策略1-RSI+布林带",
            2: "策略2-MA+KDJ",
            4: "策略4-RSI+连跌",
            6: "策略6-评分超卖",
            7: "策略7-KDJ超卖",
            8: "策略8-深跌反弹",
        }[snum]

        orig_s = [t for t in original_trades if t["strategy_num"] == snum]
        filt_s = [t for t in filtered_trades if t["strategy_num"] == snum]

        o = calc_stats(orig_s)
        f = calc_stats(filt_s)

        filtered_out = [t for t in orig_s if not t["big_order_ok"]]
        fo_wins = sum(1 for t in filtered_out if t["win"])
        fo_losses = sum(1 for t in filtered_out if not t["win"])

        print(f"\n  {sname}:")
        print(f"    {'':<18} {'原始':>10} {'+大单过滤':>10}")
        print(f"    {'笔数':<18} {o['count']:>10} {f['count']:>10}")
        print(f"    {'胜率':<18} {o['wr']:>9.1f}% {f['wr']:>9.1f}%")
        print(f"    {'累计盈亏':<18} {o['cum_pnl']:>+9.2f}% {f['cum_pnl']:>+9.2f}%")
        print(f"    {'被过滤信号':<18} {len(filtered_out):>10}笔 (盈利{fo_wins}笔/亏损{fo_losses}笔)")
        if filtered_out:
            fo_pnl = sum(t["pnl_pct"] for t in filtered_out)
            print(f"    {'过滤掉的盈亏':<18} {fo_pnl:>+9.2f}% {'✅ 过滤正确' if fo_pnl < 0 else '⚠️ 误杀盈利机会'}")

    # 按股票细分
    print(f"\n{'='*100}")
    print("【按股票细分对比】")
    print(f"{'='*100}")

    all_stock_names = sorted(set(t["stock_name"] for t in original_trades))
    for stock_name in all_stock_names:
        orig_s = [t for t in original_trades if t["stock_name"] == stock_name]
        filt_s = [t for t in filtered_trades if t["stock_name"] == stock_name]

        o = calc_stats(orig_s)
        f = calc_stats(filt_s)

        filtered_out = [t for t in orig_s if not t["big_order_ok"]]

        print(f"\n  {stock_name}:")
        print(f"    {'':<18} {'原始':>10} {'+大单过滤':>10}")
        print(f"    {'笔数':<18} {o['count']:>10} {f['count']:>10}")
        print(f"    {'胜率':<18} {o['wr']:>9.1f}% {f['wr']:>9.1f}%")
        print(f"    {'累计盈亏':<18} {o['cum_pnl']:>+9.2f}% {f['cum_pnl']:>+9.2f}%")
        if filtered_out:
            fo_pnl = sum(t["pnl_pct"] for t in filtered_out)
            fo_verdict = "✅过滤正确" if fo_pnl < 0 else "⚠️误杀"
            print(f"    {'过滤'+str(len(filtered_out))+'笔盈亏':<18} {fo_pnl:>+9.2f}% {fo_verdict}")

    # 被过滤掉的信号详情
    print(f"\n{'='*100}")
    print("【被大单过滤掉的信号详情】")
    print(f"{'='*100}")
    filtered_out_all = [t for t in original_trades if not t["big_order_ok"]]
    filtered_out_all.sort(key=lambda t: t["signal_date"])

    if filtered_out_all:
        print(f"\n  {'日期':<12} {'股票':<8} {'策略':<18} {'盈亏':>8} {'大单净额(近3日)'}")
        print(f"  {'-'*80}")
        for t in filtered_out_all:
            net_str = ", ".join([f"{a:+.0f}万" for a in t.get("net_amounts", [])])
            win_mark = "✅" if t["win"] else "❌"
            print(f"  {str(t['signal_date'].date()):<12} {t['stock_name']:<8} {t['strategy_name']:<18} {t['pnl_pct']:>+7.2f}% {win_mark} {net_str}")
        fo_pnl = sum(t["pnl_pct"] for t in filtered_out_all)
        print(f"\n  过滤总计: {len(filtered_out_all)}笔, 累计盈亏{fo_pnl:+.2f}%")
    else:
        print("\n  无被过滤信号(所有信号均满足大单连续流入条件)")

    # 过滤后保留的信号详情
    print(f"\n{'='*100}")
    print("【大单过滤后保留的信号详情】")
    print(f"{'='*100}")
    filtered_in_all = [t for t in filtered_trades if t["big_order_ok"]]
    filtered_in_all.sort(key=lambda t: t["signal_date"])

    if filtered_in_all:
        print(f"\n  {'日期':<12} {'股票':<8} {'策略':<18} {'盈亏':>8} {'大单净额(近3日)'}")
        print(f"  {'-'*80}")
        for t in filtered_in_all:
            net_str = ", ".join([f"{a:+.0f}万" for a in t.get("net_amounts", [])])
            win_mark = "✅" if t["win"] else "❌"
            print(f"  {str(t['signal_date'].date()):<12} {t['stock_name']:<8} {t['strategy_name']:<18} {t['pnl_pct']:>+7.2f}% {win_mark} {net_str}")

    print("\n" + "=" * 100)
    print("回测完成")
    print("=" * 100)

    return original_trades, filtered_trades

if __name__ == "__main__":
    run_backtest()
