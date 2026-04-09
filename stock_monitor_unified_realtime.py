#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票策略实时监控系统 - 尾盘专用
在尾盘时段执行，获取实时数据进行策略分析

使用方法：在交易日14:30-15:00执行
python stock_monitor_unified_realtime.py
"""

import sys
import io
import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests
import tushare as ts

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

POSITION_FILE = "position_record.json"
TRADE_LOG_FILE = "trade_log.csv"


class DataFetcher:
    @staticmethod
    def fetch_history_data(ts_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=days)
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                print(f"    [DEBUG] No data for {ts_code}")
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            print(f"    [ERROR] fetch_history_data {ts_code}: {e}")
            return None

    @staticmethod
    def fetch_realtime_sina(sina_code: str) -> Optional[Dict]:
        try:
            url = f"https://hq.sinajs.cn/list={sina_code}"
            headers = {"Referer": "https://finance.sina.com.cn"}
            r = requests.get(url, headers=headers, timeout=10)
            data = r.text.split('"')[1]
            parts = data.split(",")
            if len(parts) < 32:
                print(f"    [DEBUG] Invalid data for {sina_code}: {len(parts)} parts")
                return None
            return {
                "name": parts[0],
                "open": float(parts[1]),
                "pre_close": float(parts[2]),
                "close": float(parts[3]),
                "high": float(parts[4]),
                "low": float(parts[5]),
                "volume": float(parts[8]),
                "amount": float(parts[9]),
                "time": parts[31],
                "date": parts[30] if len(parts) > 30 else "",
            }
        except Exception as e:
            print(f"    [ERROR] fetch_realtime_sina {sina_code}: {e}")
            return None

    @staticmethod
    def merge_realtime_data(history_df: pd.DataFrame, realtime: Dict) -> pd.DataFrame:
        if history_df is None or realtime is None:
            return history_df
        today = datetime.now()
        pct_chg = (
            (realtime["close"] - realtime["pre_close"]) / realtime["pre_close"] * 100
        )
        today_row = pd.DataFrame(
            [
                {
                    "trade_date": today,
                    "ts_code": history_df["ts_code"].iloc[0]
                    if "ts_code" in history_df.columns
                    else "",
                    "open": realtime["open"],
                    "high": realtime["high"],
                    "low": realtime["low"],
                    "close": realtime["close"],
                    "pre_close": realtime["pre_close"],
                    "change": realtime["close"] - realtime["pre_close"],
                    "pct_chg": round(pct_chg, 2),
                    "vol": realtime["volume"] / 100 if realtime["volume"] else 0,
                    "amount": realtime["amount"] / 1000 if realtime["amount"] else 0,
                }
            ]
        )
        return pd.concat([history_df, today_row], ignore_index=True)


class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_bollinger(
        prices: pd.Series, window: int = 20, num_std: int = 2
    ) -> tuple:
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        position = (prices - lower) / (upper - lower)
        return upper, ma, lower, position

    @staticmethod
    def calculate_ma(
        prices: pd.Series, windows: List[int] = [5, 10, 20, 60]
    ) -> Dict[str, pd.Series]:
        result = {}
        for w in windows:
            result[f"ma{w}"] = prices.rolling(window=w).mean()
        return result

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        low_min = df["low"].rolling(n).min()
        high_max = df["high"].rolling(n).max()
        df = df.copy()
        df["rsv"] = (df["close"] - low_min) / (high_max - low_min) * 100
        df["k"] = df["rsv"].ewm(com=2, adjust=False).mean()
        df["d"] = df["k"].ewm(com=2, adjust=False).mean()
        df["j"] = 3 * df["k"] - 2 * df["d"]
        return df

    @staticmethod
    def calculate_macd(prices: pd.Series) -> tuple:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
        vol_ma = df["vol"].rolling(window=window).mean()
        return df["vol"] / vol_ma

    @staticmethod
    def calculate_consecutive_days(
        pct_chg: pd.Series, direction: str = "down"
    ) -> pd.Series:
        result = pd.Series(0, index=pct_chg.index)
        for i in range(1, len(pct_chg)):
            if direction == "down" and pct_chg.iloc[i] < 0:
                result.iloc[i] = result.iloc[i - 1] + 1
            elif direction == "up" and pct_chg.iloc[i] > 0:
                result.iloc[i] = result.iloc[i - 1] + 1
            else:
                result.iloc[i] = 0
        return result


class AdaptiveParams:
    """根据当前波动率动态计算策略参数阈值"""

    @staticmethod
    def compute(vol_20: float, vol_60: float) -> Dict:
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        return {
            "rsi_entry": float(np.clip(33 - (vol_ratio - 1.0) * 6, 20, 45)),
            "bb_entry": float(np.clip(0.5 - (vol_ratio - 1.0) * 0.2, 0.1, 0.6)),
            "rsi_exit": float(np.clip(60 - (vol_ratio - 1.0) * 5, 50, 70)),
            "bb_exit": 0.75,
            "kdj_entry": float(np.clip(30 - (vol_ratio - 1.0) * 10, 10, 35)),
            "rsi_consec": float(np.clip(35 - (vol_ratio - 1.0) * 5, 25, 40)),
            "mom_vol_ratio": float(np.clip(1.2 + (vol_ratio - 1.0) * 0.3, 1.0, 2.0)),
            "vol_ratio_raw": round(vol_ratio, 2),
        }

    @staticmethod
    def from_df(df: pd.DataFrame) -> Dict:
        vol_20 = df["pct_chg"].rolling(20).std().iloc[-1]
        vol_60 = df["pct_chg"].rolling(60).std().iloc[-1]
        vol_20 = vol_20 if pd.notna(vol_20) else 3.0
        vol_60 = vol_60 if pd.notna(vol_60) else 3.0
        return AdaptiveParams.compute(vol_20, vol_60)


class Strategy1_RSI_Bollinger:
    NAME = "RSI+布林带均值回归"
    STOCKS = [
        {"code": "300696.SZ", "name": "爱乐达", "sina_code": "sz300696"},
        {"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"},
    ]

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], 14)
        _, _, bb_lower, bb_pos = TechnicalIndicators.calculate_bollinger(
            df["close"], 20, 2
        )
        df["bb_position"] = bb_pos
        prices = df["close"].copy()
        prices.iloc[-1] = rt["close"]
        today_rsi = TechnicalIndicators.calculate_rsi(prices, 14).iloc[-1]
        ma20 = df["close"].iloc[-20:].mean()
        std20 = df["close"].iloc[-20:].std()
        bb_upper = ma20 + 2 * std20
        bb_lower_val = ma20 - 2 * std20
        today_bb_pos = (
            (rt["close"] - bb_lower_val) / (bb_upper - bb_lower_val)
            if (bb_upper - bb_lower_val) > 0
            else 0.5
        )

        # ST炼石用旧版固定参数，其他用自适应参数
        is_st = "ST" in stock.get("name", "")
        stock_name = stock.get("name", "")
        ap = AdaptiveParams.from_df(df)
        if is_st:
            rsi_th, bb_th = 33, 0.5
        else:
            rsi_th, bb_th = ap["rsi_entry"], ap["bb_entry"]

        # 趋势过滤 (ST股和高澜股份不过滤)
        skip_trend_filter = is_st or stock_name == "高澜股份"
        ma20_series = df["close"].rolling(20).mean()
        ma20_now = ma20_series.iloc[-1]
        ma20_5ago = ma20_series.iloc[-6] if len(ma20_series) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
        ma20_downtrend = ma20_slope < -1.0

        buy_cond_rsi = today_rsi < rsi_th
        buy_cond_bb = today_bb_pos < bb_th
        buy_cond_up = rt["close"] > rt["pre_close"]
        buy_cond_yang = rt["close"] > rt["open"]
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        buy_signal = (buy_cond_rsi or buy_cond_bb) and buy_cond_up and buy_cond_yang
        if not skip_trend_filter:
            buy_signal = buy_signal and not ma20_downtrend

        sell_cond1 = today_rsi > ap["rsi_exit"]
        sell_cond2 = today_bb_pos > ap["bb_exit"]
        sell_signal = sell_cond1 or sell_cond2
        entry_price = df["low"].iloc[-10:].min()
        profit_pct = (rt["close"] - entry_price) / entry_price * 100
        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pre_close": rt["pre_close"],
            "pct_chg": today_pct,
            "rsi": today_rsi,
            "bb_position": today_bb_pos,
            "ma20_slope": round(ma20_slope, 2),
            "adaptive_params": ap,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "profit_pct": profit_pct,
        }


class Strategy2_MA_KDJ:
    NAME = "MA支撑+KDJ超卖"
    STOCKS = [{"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"}]

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df = df.rename(columns={"vol": "volume"})
        mas = TechnicalIndicators.calculate_ma(df["close"], [5, 20])
        for k, v in mas.items():
            df[k] = v
        df = TechnicalIndicators.calculate_kdj(df)
        ap = AdaptiveParams.from_df(df)
        latest = df.iloc[-1]
        signals = []
        if (
            pd.notna(latest["ma20"])
            and abs(latest["close"] - latest["ma20"]) / latest["ma20"] < 0.02
            and latest["pct_chg"] > 0
        ):
            signals.append("MA20支撑")
        if pd.notna(latest["j"]) and latest["j"] < ap["kdj_entry"]:
            signals.append("KDJ超卖")
        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "ma20": latest["ma20"],
            "j_value": latest["j"],
            "signals": signals,
            "adaptive_params": ap,
            "buy_signal": len(signals) > 0,
        }


class Strategy3_MultiFactor:
    NAME = "多因子买入策略"
    STOCKS = [
        {"code": "300499.SZ", "name": "高澜股份", "sina_code": "sz300499"},
        {"code": "002837.SZ", "name": "英维克", "sina_code": "sz002837"},
    ]
    CONFIG = {
        "vpp_volume_ratio": 1.5,
        "vpp_pct_chg": 2.0,
        "momentum_volume_ratio": 1.2,
        "breakout_volume_ratio": 2.0,
        "breakout_pct_chg": 3.0,
    }

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        mas = TechnicalIndicators.calculate_ma(df["close"], [5, 10, 20, 60])
        for k, v in mas.items():
            df[k] = v
        df["volume_ratio"] = TechnicalIndicators.calculate_volume_ratio(df, 20)
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], 14)
        macd, macd_signal = TechnicalIndicators.calculate_macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        _, bb_mid, bb_lower, bb_pos = TechnicalIndicators.calculate_bollinger(
            df["close"], 20, 2
        )
        df["bb_mid"] = bb_mid
        df["bb_lower"] = bb_lower
        df["bb_position"] = bb_pos
        df["consecutive_up"] = TechnicalIndicators.calculate_consecutive_days(
            df["pct_chg"], "up"
        )
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        signals = []
        if (
            latest["volume_ratio"] > cls.CONFIG["vpp_volume_ratio"]
            and latest["pct_chg"] > cls.CONFIG["vpp_pct_chg"]
            and latest["close"] >= latest["ma5"]
        ):
            signals.append("量价配合")
        if (
            latest["consecutive_up"] >= 1
            and latest["pct_chg"] > prev["pct_chg"]
            and latest["volume_ratio"] > cls.CONFIG["momentum_volume_ratio"]
        ):
            signals.append("动量加速")
        if (
            latest["volume_ratio"] > cls.CONFIG["breakout_volume_ratio"]
            and latest["pct_chg"] > cls.CONFIG["breakout_pct_chg"]
            and latest["close"] > latest["ma20"]
        ):
            signals.append("放量突破")
        if (
            latest["close"] < latest["bb_lower"]
            and prev["close"] >= df["bb_lower"].iloc[-2]
        ):
            signals.append("布林带触及")
        if (
            (latest["ma5"] > latest["ma10"] > latest["ma20"])
            and not (prev["ma5"] > prev["ma10"])
            and latest["macd"] > 0
        ):
            signals.append("均线多头排列")
        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "volume_ratio": latest["volume_ratio"],
            "rsi": latest["rsi"],
            "signals": signals,
            "buy_signal": len(signals) > 0,
        }


class Strategy4_RSI_ConsecutiveDown:
    NAME = "RSI+连跌中等信号"
    STOCKS = {
        "002831.SZ": {"name": "裕同科技", "sina_code": "sz002831"},
        "600486.SH": {"name": "扬农化工", "sina_code": "sh600486"},
        "300627.SZ": {"name": "华测导航", "sina_code": "sz300627"},
        "002272.SZ": {"name": "川润股份", "sina_code": "sz002272"},
        "000697.SZ": {"name": "ST炼石", "sina_code": "sz000697"},
        "300499.SZ": {"name": "高澜股份", "sina_code": "sz300499"},
        "002218.SZ": {"name": "拓日新能", "sina_code": "sz002218"},
    }

    @classmethod
    def analyze(cls, ts_code: str, stock_info: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(ts_code, days=90)
        if df is None or len(df) < 20:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock_info["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], 14)
        df["consecutive_down"] = TechnicalIndicators.calculate_consecutive_days(
            df["pct_chg"], "down"
        )
        # 扬农化工和拓日新能用固定阈值，其他用自适应
        ap = AdaptiveParams.from_df(df)
        if stock_info["name"] in ("扬农化工", "拓日新能"):
            rsi_th = 35
        else:
            rsi_th = ap["rsi_consec"]
        latest = df.iloc[-1]
        buy_signal = latest["rsi"] <= rsi_th and latest["consecutive_down"] >= 2
        return {
            "strategy": cls.NAME,
            "name": stock_info["name"],
            "code": ts_code,
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "rsi": latest["rsi"],
            "consecutive_down": int(latest["consecutive_down"]),
            "rsi_condition": latest["rsi"] <= rsi_th,
            "consecutive_condition": latest["consecutive_down"] >= 2,
            "adaptive_params": ap,
            "buy_signal": buy_signal,
        }


class Strategy5_Momentum:
    NAME = "动量策略"
    STOCKS = [
        {"code": "002272.SZ", "sina_code": "sz002272", "name": "川润股份"},
        {"code": "300696.SZ", "sina_code": "sz300696", "name": "爱乐达"},
    ]

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df["volume_ma5"] = df["vol"].rolling(5).mean()
        df["volume_ratio"] = df["vol"] / df["volume_ma5"]
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["consecutive_up"] = 0
        for i in range(1, len(df)):
            if df["pct_chg"].iloc[i] > 0:
                df.iloc[i, df.columns.get_loc("consecutive_up")] = (
                    df["consecutive_up"].iloc[i - 1] + 1
                )
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        df["rsi_6"] = 100 - (100 / (1 + rs))
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["dif"] = df["ema12"] - df["ema26"]
        df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
        df["macd"] = (df["dif"] - df["dea"]) * 2
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # 英维克用旧版固定参数，其他用自适应
        stock_name = stock.get("name", "")
        ap = AdaptiveParams.from_df(df)
        if stock_name == "英维克":
            mom_vol_th = 1.2
            osb_vol_th = 1.3
        else:
            mom_vol_th = ap["mom_vol_ratio"]
            osb_vol_th = mom_vol_th * 1.1

        signal_volume_price = (
            latest["volume_ratio"] > 1.5
            and latest["pct_chg"] > 2
            and latest["close"] >= latest["ma5"]
        )
        signal_momentum = (
            latest["consecutive_up"] >= 1
            and latest["pct_chg"] > prev["pct_chg"]
            and latest["volume_ratio"] > mom_vol_th
        )
        signal_oversold_bounce = (
            latest["rsi_6"] < 30
            and latest["pct_chg"] > 3
            and latest["volume_ratio"] > osb_vol_th
        )
        signal_breakout = (
            latest["close"] > latest["ma20"]
            and prev["close"] <= prev["ma20"]
            and latest["volume_ratio"] > 1.5
        )
        signal_macd = (
            latest["dif"] > latest["dea"]
            and prev["dif"] <= prev["dea"]
            and latest["macd"] > 0
        )

        # 趋势过滤 (英维克不过滤)
        ma20_now = latest["ma20"]
        ma20_5ago = df["ma20"].iloc[-6] if len(df) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
        ma20_downtrend = ma20_slope < -1.0

        volume_price_threshold = 1.5
        if stock_name == "川润股份":
            volume_price_threshold = 1.1
        elif stock_name == "爱乐达":
            volume_price_threshold = 1.2
        elif stock_name == "英维克":
            volume_price_threshold = 1.1

        use_volume_price = (
            signal_volume_price and latest["volume_ratio"] > volume_price_threshold
        )
        use_momentum = signal_momentum
        use_oversold_bounce = signal_oversold_bounce
        use_breakout = signal_breakout
        use_macd = signal_macd

        if stock_name == "川润股份":
            use_breakout = False
            use_macd = False
        elif stock_name == "英维克":
            use_oversold_bounce = False
            use_breakout = False
        elif stock_name == "爱乐达":
            use_volume_price = False
            use_breakout = False

        signal_count = sum(
            [
                use_volume_price,
                use_momentum,
                use_oversold_bounce,
                use_breakout,
                use_macd,
            ]
        )
        signal_strong = signal_count >= 2
        signal_momentum_only = use_momentum and signal_count == 1

        # 单动量信号在下跌趋势中不触发 (英维克不过滤)
        if stock_name != "英维克" and ma20_downtrend and signal_momentum_only:
            signal_momentum_only = False

        buy_signal = signal_strong or signal_momentum_only
        triggered_signals = []
        if use_volume_price:
            triggered_signals.append("量价配合")
        if use_momentum:
            triggered_signals.append("动量加速")
        if use_oversold_bounce:
            triggered_signals.append("超跌反弹")
        if use_breakout:
            triggered_signals.append("突破信号")
        if use_macd:
            triggered_signals.append("MACD金叉")
        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "volume_ratio": latest["volume_ratio"],
            "rsi_6": latest["rsi_6"],
            "ma20_slope": round(ma20_slope, 2),
            "adaptive_params": ap,
            "signal_count": signal_count,
            "triggered_signals": triggered_signals,
            "signal_volume_price": signal_volume_price,
            "signal_momentum": signal_momentum,
            "signal_oversold_bounce": signal_oversold_bounce,
            "signal_breakout": signal_breakout,
            "signal_macd": signal_macd,
            "buy_signal": buy_signal,
        }


class Strategy6_ScoreModel:
    NAME = "多因子评分超卖"
    STOCKS = [
        {"code": "002218.SZ", "name": "拓日新能", "sina_code": "sz002218"},
    ]
    SCORE_THRESHOLD = 50

    @classmethod
    def compute_score(cls, df: pd.DataFrame, rt: Dict, sh_pct: float = 0) -> int:
        """计算超卖评分 (0-100)"""
        score = 0
        latest = df.iloc[-1]

        # RSI14
        rsi14 = TechnicalIndicators.calculate_rsi(df["close"], 14).iloc[-1]
        if pd.notna(rsi14):
            if rsi14 < 25: score += 25
            elif rsi14 < 30: score += 20
            elif rsi14 < 35: score += 15
            elif rsi14 < 40: score += 10
            elif rsi14 < 45: score += 5

        # 布林带位置
        _, _, bb_lower, bb_pos = TechnicalIndicators.calculate_bollinger(df["close"], 20, 2)
        bb = bb_pos.iloc[-1] if pd.notna(bb_pos.iloc[-1]) else 0.5
        if bb < 0.1: score += 20
        elif bb < 0.2: score += 16
        elif bb < 0.3: score += 12
        elif bb < 0.4: score += 8
        elif bb < 0.5: score += 4

        # KDJ J值
        df_kdj = TechnicalIndicators.calculate_kdj(df)
        j = df_kdj["j"].iloc[-1]
        if pd.notna(j):
            if j < 0: score += 15
            elif j < 10: score += 12
            elif j < 20: score += 8
            elif j < 30: score += 4

        # 连续下跌
        consec = TechnicalIndicators.calculate_consecutive_days(df["pct_chg"], "down")
        cd = consec.iloc[-1]
        if cd >= 4: score += 15
        elif cd >= 3: score += 12
        elif cd >= 2: score += 8

        # 5日跌幅
        if len(df) >= 5:
            ret5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
            if ret5d < -10: score += 10
            elif ret5d < -7: score += 8
            elif ret5d < -5: score += 6
            elif ret5d < -3: score += 3

        # 大盘当日涨
        if sh_pct > 0:
            score += 5

        # 长下影线
        lower_shadow = (min(rt["close"], rt["open"]) - rt["low"]) / rt["close"] * 100
        if lower_shadow > 1.5:
            score += 5

        # 相对大盘5日弱
        if len(df) >= 5:
            rel_5d = sum(df["pct_chg"].iloc[-5:]) - sh_pct * 5  # 近似
            if rel_5d < -5:
                score += 5

        # 缩量惩罚
        vol_ma5 = df["vol"].rolling(5).mean().iloc[-1]
        if pd.notna(vol_ma5) and vol_ma5 > 0:
            vol_ratio = df["vol"].iloc[-1] / vol_ma5
            if vol_ratio < 0.6:
                score -= 8

        return score, rsi14, bb, j, cd

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)

        # 获取大盘涨跌 (简化: 用0近似，实际大盘数据在评分中权重低)
        sh_pct = 0
        try:
            import tushare as ts
            sh_df = ts.pro_api().index_daily(
                ts_code="000001.SH",
                start_date=df["trade_date"].iloc[-1].strftime("%Y%m%d"),
                end_date=df["trade_date"].iloc[-1].strftime("%Y%m%d"),
            )
            if sh_df is not None and not sh_df.empty:
                sh_pct = sh_df["pct_chg"].iloc[0]
        except Exception:
            pass

        score, rsi14, bb, j, cd = cls.compute_score(df, rt, sh_pct)
        buy_signal = score >= cls.SCORE_THRESHOLD

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100,
            "score": score,
            "rsi": rsi14,
            "bb_position": bb,
            "j_value": j,
            "consecutive_down": int(cd),
            "buy_signal": buy_signal,
        }


class Strategy7_KDJ_Bounce:
    NAME = "KDJ超卖反弹"
    STOCKS = [
        {"code": "002837.SZ", "name": "英维克", "sina_code": "sz002837"},
    ]

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df = TechnicalIndicators.calculate_kdj(df)
        latest = df.iloc[-1]
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        j_value = latest["j"]
        is_up = today_pct > 0
        buy_signal = pd.notna(j_value) and j_value < 10 and is_up

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "j_value": j_value,
            "k_value": latest["k"],
            "buy_signal": buy_signal,
        }


class Strategy8_DeepDrop:
    NAME = "深跌反弹"
    STOCKS = [
        {"code": "300499.SZ", "name": "高澜股份", "sina_code": "sz300499"},
        {"code": "002272.SZ", "name": "川润股份", "sina_code": "sz002272"},
        {"code": "600418.SH", "name": "江淮汽车", "sina_code": "sh600418"},
        {"code": "300696.SZ", "name": "爱乐达", "sina_code": "sz300696"},
    ]

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 20:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], 14)
        latest = df.iloc[-1]
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        # 5日跌幅
        ret5d = (latest["close"] / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        rsi14 = latest["rsi"]

        # 信号A: 5日跌>5% & RSI<40 & 当日涨, T+1/T+4
        signal_a = ret5d < -5 and pd.notna(rsi14) and rsi14 < 40 and today_pct > 0
        # 信号B: 5日跌>10% (高澜T+0/T+5, 江淮T+1/T+6, 爱乐达T+0/T+5)
        signal_b = ret5d < -10 and stock["name"] in ("高澜股份", "江淮汽车", "爱乐达")

        signals = []
        if signal_a:
            signals.append(f"5日跌{ret5d:.1f}%+RSI{rsi14:.0f}+涨(T+0/T+4)")
        if signal_b:
            signals.append(f"5日跌{ret5d:.1f}%(T+0/T+5)")

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "ret5d": round(ret5d, 1),
            "rsi": rsi14,
            "signals": signals,
            "buy_signal": signal_a or signal_b,
        }


def load_positions() -> List:
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_positions(positions: List):
    with open(POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def get_history_win_rate(stock_name: str, strategy_name: str) -> float:
    # 自适应参数回测胜率 (2025-01-01起, ST炼石2025-05-07起)
    win_rates = {
        ("爱乐达", "RSI+布林带均值回归"): 83.3,
        ("爱乐达", "动量策略"): 64.7,
        ("ST炼石", "RSI+布林带均值回归"): 69.6,
        ("ST炼石", "MA支撑+KDJ超卖"): 60.9,
        ("ST炼石", "RSI+连跌中等信号"): 100.0,
        ("高澜股份", "多因子买入策略"): 60.8,
        ("高澜股份", "RSI+连跌中等信号"): 56.2,
        ("英维克", "多因子买入策略"): 58.2,
        ("英维克", "KDJ超卖反弹"): 84.6,
        ("裕同科技", "RSI+连跌中等信号"): 78.6,
        ("扬农化工", "RSI+连跌中等信号"): 71.4,
        ("华测导航", "RSI+连跌中等信号"): 68.8,
        ("川润股份", "RSI+连跌中等信号"): 62.5,
        ("川润股份", "动量策略"): 63.0,
        ("川润股份", "深跌反弹"): 70.0,
        ("拓日新能", "RSI+连跌中等信号"): 85.7,
        ("拓日新能", "多因子评分超卖"): 86.4,
        ("高澜股份", "深跌反弹"): 81.8,
        ("江淮汽车", "深跌反弹"): 76.9,
        ("爱乐达", "深跌反弹"): 73.7,
    }
    return win_rates.get((stock_name, strategy_name), 0.0)


def get_trade_timing(stock_name: str, strategy_name: str) -> Dict:
    t0_best = {
        ("高澜股份", "RSI+连跌中等信号"),
        ("裕同科技", "RSI+连跌中等信号"),
        ("扬农化工", "多因子买入策略"),
        ("扬农化工", "RSI+连跌中等信号"),
        ("华测导航", "RSI+连跌中等信号"),
        ("川润股份", "MA支撑+KDJ超卖"),
        ("川润股份", "RSI+连跌中等信号"),
        ("川润股份", "动量策略"),
        ("拓日新能", "多因子评分超卖"),
        ("英维克", "KDJ超卖反弹"),
        ("高澜股份", "深跌反弹"),  # 两个子信号都是T+0买
        ("爱乐达", "RSI+布林带均值回归"),
        ("爱乐达", "深跌反弹"),  # 8A: T+0/T+3开盘卖, 8B: T+0/T+5尾盘卖
    }
    # 特殊时机：川润动量T+0/T+4，川润深跌反弹T+1/T+4
    t4_sell = {
        ("川润股份", "动量策略"),
        ("川润股份", "深跌反弹"),
    }
    if (stock_name, strategy_name) in t4_sell:
        if (stock_name, strategy_name) in t0_best:
            return {
                "buy_timing": "T+0尾盘",
                "sell_timing": "T+4尾盘",
                "buy_desc": "当日尾盘收盘前买入",
                "sell_desc": "最晚第4个交易日尾盘卖出",
            }
        else:
            return {
                "buy_timing": "T+1开盘",
                "sell_timing": "T+4开盘",
                "buy_desc": "次日开盘买入",
                "sell_desc": "最晚第4个交易日开盘卖出",
            }
    if (stock_name, strategy_name) in t0_best:
        return {
            "buy_timing": "T+0尾盘",
            "sell_timing": "T+5尾盘",
            "buy_desc": "当日尾盘收盘前买入",
            "sell_desc": "最晚第5个交易日尾盘卖出",
        }
    else:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+6开盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第6个交易日开盘卖出",
        }


def print_header():
    print("\n" + "=" * 100)
    print("股票策略监控系统 - 统一版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


def print_strategy_results(strategy_name: str, results: List[Dict]):
    if not results:
        return
    print(f"\n{'=' * 100}")
    print(f"【{strategy_name}】")
    print(f"{'=' * 100}")
    buy_signals = [r for r in results if r.get("buy_signal")]
    for r in results:
        signal_mark = "[买]" if r.get("buy_signal") else "[-]"
        name = r.get("name", "")
        code = r.get("code", "")
        price = r.get("price", 0)
        pct_chg = r.get("pct_chg", 0)
        win_rate = get_history_win_rate(name, strategy_name)
        print(
            f"  {signal_mark} {name}({code}): 价格{price:.2f}, 涨幅{pct_chg:+.2f}%, 历史胜率{win_rate:.1f}%"
        )
        if r.get("strategy") == Strategy1_RSI_Bollinger.NAME:
            ap = r.get("adaptive_params", {})
            print(
                f"       RSI={r.get('rsi', 0):.1f}, BB位置={r.get('bb_position', 0):.3f}"
                f"  [自适应: RSI<{ap.get('rsi_entry', 33):.0f} BB<{ap.get('bb_entry', 0.5):.2f} 波动比={ap.get('vol_ratio_raw', 1):.2f}x]"
            )
        elif r.get("strategy") == Strategy2_MA_KDJ.NAME:
            sigs = r.get("signals", [])
            print(f"       MA20={r.get('ma20', 0):.2f}, J值={r.get('j_value', 0):.2f}")
            if sigs:
                print(f"       触发信号: {', '.join(sigs)}")
        elif r.get("strategy") == Strategy3_MultiFactor.NAME:
            sigs = r.get("signals", [])
            print(
                f"       量比={r.get('volume_ratio', 0):.2f}, RSI={r.get('rsi', 0):.1f}"
            )
            if sigs:
                print(f"       触发策略: {', '.join(sigs)}")
        elif r.get("strategy") == Strategy4_RSI_ConsecutiveDown.NAME:
            rsi_cond = "满足" if r.get("rsi_condition") else "不满足"
            cons_cond = "满足" if r.get("consecutive_condition") else "不满足"
            print(
                f"       RSI={r.get('rsi', 0):.1f}({rsi_cond}), 连跌={r.get('consecutive_down', 0)}天({cons_cond})"
            )
            if r.get("name") == "川润股份" and r.get("buy_signal"):
                print(f"       [川润提示] 同时参考T+0/T+4卖出(胜率76.5%)")
        elif r.get("strategy") == Strategy5_Momentum.NAME:
            signals = r.get("triggered_signals", [])
            print(
                f"       量比={r.get('volume_ratio', 0):.2f}, RSI(6)={r.get('rsi_6', 0):.1f}"
            )
            print(f"       信号数: {r.get('signal_count', 0)}/5")
            if signals:
                print(f"       触发: {', '.join(signals)}")
        elif r.get("strategy") == Strategy6_ScoreModel.NAME:
            print(
                f"       评分={r.get('score', 0)}, RSI={r.get('rsi', 0):.1f}, "
                f"BB={r.get('bb_position', 0):.2f}, J={r.get('j_value', 0):.1f}, "
                f"连跌={r.get('consecutive_down', 0)}天"
            )
        elif r.get("strategy") == Strategy7_KDJ_Bounce.NAME:
            print(
                f"       J={r.get('j_value', 0):.1f}, K={r.get('k_value', 0):.1f}, "
                f"涨幅={r.get('pct_chg', 0):+.2f}%"
            )
        elif r.get("strategy") == Strategy8_DeepDrop.NAME:
            sigs = r.get("signals", [])
            rsi_s = f"{r.get('rsi', 0):.1f}" if pd.notna(r.get('rsi')) else "-"
            print(
                f"       5日跌幅={r.get('ret5d', 0):.1f}%, RSI={rsi_s}, 涨跌={r.get('pct_chg', 0):+.2f}%"
            )
            if sigs:
                print(f"       触发: {', '.join(sigs)}")
            else:
                # 显示距离触发的距离
                ret5d = r.get('ret5d', 0)
                rsi = r.get('rsi', 50)
                notes = []
                if ret5d >= -5:
                    notes.append(f"5日跌幅不足(需<-5%)")
                elif ret5d >= -10:
                    if pd.notna(rsi) and rsi >= 40:
                        notes.append(f"RSI未达标(需<40)")
                    if r.get('pct_chg', 0) <= 0:
                        notes.append(f"当日未涨")
                if notes:
                    print(f"       未触发: {', '.join(notes)}")
    print(f"\n  买入信号数量: {len(buy_signals)}/{len(results)}")


def print_summary(all_results: Dict[str, List]):
    print("\n" + "=" * 100)
    print("【汇总报告】")
    print("=" * 100)
    total_buy = 0
    total_analyzed = 0
    buy_list = []
    for strategy_name, results in all_results.items():
        for r in results:
            total_analyzed += 1
            if r.get("buy_signal"):
                total_buy += 1
                buy_list.append(r)
    print(f"\n总计分析股票: {total_analyzed} 只")
    print(f"触发买入信号: {total_buy} 只")
    if buy_list:
        print("\n买入信号股票列表:")
        print("-" * 100)
        print(
            f"{'股票':<8} {'策略':<18} {'价格':>8} {'胜率':>6} {'买入时机':<10} {'卖出时机':<10}"
        )
        print("-" * 100)
        for r in buy_list:
            name = r.get("name", "")
            strategy_name = r.get("strategy", "")
            timing = get_trade_timing(name, strategy_name)
            print(
                f"  {name:<6} {strategy_name:<16} {r.get('price', 0):>8.2f} {get_history_win_rate(name, strategy_name):>5.1f}% {timing['buy_timing']:<10} {timing['sell_timing']:<10}"
            )
        print("-" * 100)
        print("\n操作建议:")
        for i, r in enumerate(buy_list, 1):
            name = r.get("name", "")
            strategy_name = r.get("strategy", "")
            timing = get_trade_timing(name, strategy_name)
            print(f"  {i}. {name}({r.get('code', '')})")
            print(f"     买入: {timing['buy_desc']}")
            print(f"     卖出: {timing['sell_desc']} (可提前止盈/止损)")
    else:
        print("\n暂无买入信号")
    print("\n" + "=" * 100)


def main():
    now = datetime.now()
    print("\n" + "=" * 100)
    print("股票策略实时监控系统 - 尾盘专用")
    print(f"执行时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DEBUG] TUSHARE_TOKEN: {'已设置' if TUSHARE_TOKEN else '未设置'}")
    print("=" * 100)

    all_results = {}

    print("\n[策略1] RSI+布林带均值回归")
    results = []
    for stock in Strategy1_RSI_Bollinger.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy1_RSI_Bollinger.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "RSI+布林带均值回归")
            timing = get_trade_timing(stock["name"], "RSI+布林带均值回归")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy1_RSI_Bollinger.NAME] = results
    print_strategy_results(Strategy1_RSI_Bollinger.NAME, results)

    print("\n[策略2] MA支撑+KDJ超卖")
    results = []
    for stock in Strategy2_MA_KDJ.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy2_MA_KDJ.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "MA支撑+KDJ超卖")
            timing = get_trade_timing(stock["name"], "MA支撑+KDJ超卖")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy2_MA_KDJ.NAME] = results
    print_strategy_results(Strategy2_MA_KDJ.NAME, results)

    print("\n[策略3] 多因子买入策略")
    results = []
    for stock in Strategy3_MultiFactor.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy3_MultiFactor.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "多因子买入策略")
            timing = get_trade_timing(stock["name"], "多因子买入策略")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy3_MultiFactor.NAME] = results
    print_strategy_results(Strategy3_MultiFactor.NAME, results)

    print("\n[策略4] RSI+连跌中等信号")
    results = []
    for ts_code, stock_info in Strategy4_RSI_ConsecutiveDown.STOCKS.items():
        print(f"  分析 {stock_info['name']}...", end=" ")
        r = Strategy4_RSI_ConsecutiveDown.analyze(ts_code, stock_info)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock_info["name"], "RSI+连跌中等信号")
            timing = get_trade_timing(stock_info["name"], "RSI+连跌中等信号")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy4_RSI_ConsecutiveDown.NAME] = results
    print_strategy_results(Strategy4_RSI_ConsecutiveDown.NAME, results)

    print("\n[策略5] 动量策略")
    results = []
    for stock in Strategy5_Momentum.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy5_Momentum.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "动量策略")
            timing = get_trade_timing(stock["name"], "动量策略")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy5_Momentum.NAME] = results
    print_strategy_results(Strategy5_Momentum.NAME, results)

    print("\n[策略6] 多因子评分超卖")
    results = []
    for stock in Strategy6_ScoreModel.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy6_ScoreModel.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "多因子评分超卖")
            timing = get_trade_timing(stock["name"], "多因子评分超卖")
            r["win_rate"] = win_rate
            r["timing"] = timing
            score = r.get("score", 0)
            print(f"完成 - 评分{score} {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy6_ScoreModel.NAME] = results
    print_strategy_results(Strategy6_ScoreModel.NAME, results)

    print("\n[策略7] KDJ超卖反弹")
    results = []
    for stock in Strategy7_KDJ_Bounce.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy7_KDJ_Bounce.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "KDJ超卖反弹")
            timing = get_trade_timing(stock["name"], "KDJ超卖反弹")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - J={r.get('j_value', 0):.1f} {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy7_KDJ_Bounce.NAME] = results
    print_strategy_results(Strategy7_KDJ_Bounce.NAME, results)

    print("\n[策略8] 深跌反弹")
    results = []
    for stock in Strategy8_DeepDrop.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy8_DeepDrop.analyze(stock)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "深跌反弹")
            timing = get_trade_timing(stock["name"], "深跌反弹")
            r["win_rate"] = win_rate
            r["timing"] = timing
            sigs = r.get("signals", [])
            print(f"完成 - {', '.join(sigs) if sigs else '无信号'}")
        else:
            print("失败")
    all_results[Strategy8_DeepDrop.NAME] = results
    print_strategy_results(Strategy8_DeepDrop.NAME, results)

    print_summary_with_timing(all_results)


def print_summary_with_timing(all_results: Dict[str, List]):
    print("\n" + "=" * 100)
    print("【汇总报告】")
    print("=" * 100)
    total_buy = 0
    total_analyzed = 0
    buy_list = []
    for strategy_name, results in all_results.items():
        for r in results:
            total_analyzed += 1
            if r.get("buy_signal"):
                total_buy += 1
                buy_list.append(r)
    print(f"\n总计分析股票: {total_analyzed} 只")
    print(f"触发买入信号: {total_buy} 只")
    if buy_list:
        print("\n买入信号股票列表:")
        print("-" * 110)
        print(
            f"{'股票':<8} {'策略':<18} {'价格':>8} {'胜率':>6} {'买入时机':<10} {'卖出时机':<10}"
        )
        print("-" * 110)
        for r in buy_list:
            name = r.get("name", "")
            strategy_name = r.get("strategy", "")
            timing = r.get("timing", {})
            win_rate = r.get("win_rate", 0)
            print(
                f"  {name:<6} {strategy_name:<16} {r.get('price', 0):>8.2f} {win_rate:>5.1f}% {timing.get('buy_timing', ''):<10} {timing.get('sell_timing', ''):<10}"
            )
        print("-" * 110)
        print("\n操作建议:")
        for i, r in enumerate(buy_list, 1):
            name = r.get("name", "")
            strategy_name = r.get("strategy", "")
            timing = r.get("timing", {})
            print(f"  {i}. {name}({r.get('code', '')})")
            print(f"     买入: {timing.get('buy_desc', '')}")
            print(f"     卖出: {timing.get('sell_desc', '')} (可提前止盈/止损)")
    else:
        print("\n暂无买入信号")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
