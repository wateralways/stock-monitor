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

TUSHARE_TOKEN = "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
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
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception:
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
        except Exception:
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


class Strategy1_RSI_Bollinger:
    NAME = "RSI+布林带均值回归"
    STOCKS = [
        {"code": "300696.SZ", "name": "爱乐达", "sina_code": "sz300696"},
        {"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"},
        {"code": "300499.SZ", "name": "高澜股份", "sina_code": "sz300499"},
    ]
    CONFIG = {"rsi_entry": 33, "bb_entry": 0.5, "rsi_exit": 60, "bb_exit": 0.75}

    @classmethod
    def analyze(cls, stock: Dict) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
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
        buy_cond1 = (
            today_rsi < cls.CONFIG["rsi_entry"] or today_bb_pos < cls.CONFIG["bb_entry"]
        )
        buy_cond2 = rt["close"] > rt["pre_close"]
        buy_cond3 = rt["close"] > rt["open"]
        buy_signal = buy_cond1 and buy_cond2 and buy_cond3
        sell_cond1 = today_rsi > cls.CONFIG["rsi_exit"]
        sell_cond2 = today_bb_pos > cls.CONFIG["bb_exit"]
        sell_signal = sell_cond1 or sell_cond2
        entry_price = df["low"].iloc[-10:].min()
        profit_pct = (rt["close"] - entry_price) / entry_price * 100
        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pre_close": rt["pre_close"],
            "pct_chg": (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100,
            "rsi": today_rsi,
            "bb_position": today_bb_pos,
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
        latest = df.iloc[-1]
        signals = []
        if (
            pd.notna(latest["ma20"])
            and abs(latest["close"] - latest["ma20"]) / latest["ma20"] < 0.02
            and latest["pct_chg"] > 0
        ):
            signals.append("MA20支撑")
        if pd.notna(latest["j"]) and latest["j"] < 30:
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
        "002837.SZ": {"name": "英维克", "sina_code": "sz002837"},
        "000697.SZ": {"name": "ST炼石", "sina_code": "sz000697"},
        "300499.SZ": {"name": "高澜股份", "sina_code": "sz300499"},
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
        latest = df.iloc[-1]
        buy_signal = latest["rsi"] <= 35 and latest["consecutive_down"] >= 2
        return {
            "strategy": cls.NAME,
            "name": stock_info["name"],
            "code": ts_code,
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "rsi": latest["rsi"],
            "consecutive_down": int(latest["consecutive_down"]),
            "rsi_condition": latest["rsi"] <= 35,
            "consecutive_condition": latest["consecutive_down"] >= 2,
            "buy_signal": buy_signal,
        }


class Strategy5_Momentum:
    NAME = "动量策略"
    STOCKS = [
        {"code": "002272.SZ", "sina_code": "sz002272", "name": "川润股份"},
        {"code": "002837.SZ", "sina_code": "sz002837", "name": "英维克"},
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
        signal_volume_price = (
            latest["volume_ratio"] > 1.5
            and latest["pct_chg"] > 2
            and latest["close"] >= latest["ma5"]
        )
        signal_momentum = (
            latest["consecutive_up"] >= 1
            and latest["pct_chg"] > prev["pct_chg"]
            and latest["volume_ratio"] > 1.2
        )
        signal_oversold_bounce = (
            latest["rsi_6"] < 30
            and latest["pct_chg"] > 3
            and latest["volume_ratio"] > 1.3
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

        stock_name = stock.get("name", "")

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
            "signal_count": signal_count,
            "triggered_signals": triggered_signals,
            "signal_volume_price": signal_volume_price,
            "signal_momentum": signal_momentum,
            "signal_oversold_bounce": signal_oversold_bounce,
            "signal_breakout": signal_breakout,
            "signal_macd": signal_macd,
            "buy_signal": buy_signal,
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
    win_rates = {
        ("爱乐达", "RSI+布林带均值回归"): 63.16,
        ("爱乐达", "多因子买入策略"): 46.67,
        ("爱乐达", "动量策略"): 68.18,
        ("ST炼石", "RSI+布林带均值回归"): 75.00,
        ("ST炼石", "MA支撑+KDJ超卖"): 61.90,
        ("ST炼石", "多因子买入策略"): 90.00,
        ("ST炼石", "RSI+连跌中等信号"): 100.00,
        ("高澜股份", "RSI+布林带均值回归"): 57.14,
        ("高澜股份", "多因子买入策略"): 61.54,
        ("高澜股份", "RSI+连跌中等信号"): 71.43,
        ("裕同科技", "RSI+连跌中等信号"): 77.78,
        ("扬农化工", "多因子买入策略"): 75.00,
        ("扬农化工", "RSI+连跌中等信号"): 88.89,
        ("华测导航", "RSI+连跌中等信号"): 77.78,
        ("川润股份", "MA支撑+KDJ超卖"): 50.00,
        ("川润股份", "RSI+连跌中等信号"): 77.78,
        ("川润股份", "动量策略"): 68.42,
        ("英维克", "多因子买入策略"): 58.52,
        ("英维克", "RSI+连跌中等信号"): 66.67,
        ("英维克", "动量策略"): 60.87,
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
        ("英维克", "动量策略"),
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
            print(
                f"       RSI={r.get('rsi', 0):.1f}, BB位置={r.get('bb_position', 0):.3f}"
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
        elif r.get("strategy") == Strategy5_Momentum.NAME:
            signals = r.get("triggered_signals", [])
            print(
                f"       量比={r.get('volume_ratio', 0):.2f}, RSI(6)={r.get('rsi_6', 0):.1f}"
            )
            print(f"       信号数: {r.get('signal_count', 0)}/5")
            if signals:
                print(f"       触发: {', '.join(signals)}")
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
