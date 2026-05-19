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


class MarketEnvironment:
    """市场环境判断：大盘趋势、波动、风险等级"""

    _cache: Optional[Dict] = None
    _cache_time: Optional[datetime] = None

    @classmethod
    def fetch_sh_index(cls) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            df = pro.index_daily(
                ts_code="000001.SH",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            print(f"    [WARN] 获取大盘数据失败: {e}")
            return None

    @classmethod
    def get_env(cls) -> Dict:
        now = datetime.now()
        if cls._cache is not None and cls._cache_time is not None:
            if (now - cls._cache_time).seconds < 300:
                return cls._cache

        df = cls.fetch_sh_index()
        if df is None or len(df) < 20:
            return {
                "sh_trend": "unknown",
                "sh_ma20_slope": 0.0,
                "sh_5d_return": 0.0,
                "sh_today_pct": 0.0,
                "sh_vol_20": 1.0,
                "market_risk": "medium",
                "pause_all": False,
                "pause_momentum": False,
            }

        latest = df.iloc[-1]
        sh_today_pct = latest.get("pct_chg", 0)
        sh_5d_return = (latest["close"] / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        ma10 = df["close"].rolling(10).mean()
        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        ma10_now = ma10.iloc[-1] if len(ma10) >= 10 else latest["close"]
        ma20_now = ma20.iloc[-1] if len(ma20) >= 20 else latest["close"]
        ma60_now = ma60.iloc[-1] if len(ma60) >= 60 else latest["close"]
        ma20_5ago = ma20.iloc[-6] if len(ma20) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
        vol_20 = df["pct_chg"].rolling(20).std().iloc[-1]
        vol_20 = vol_20 if pd.notna(vol_20) else 1.0

        # 价格在均线上的位置
        close = latest["close"]
        above_ma20 = close > ma20_now if pd.notna(ma20_now) else True
        above_ma60 = close > ma60_now if pd.notna(ma60_now) else True
        ma10_above_ma20 = ma10_now > ma20_now if pd.notna(ma10_now) and pd.notna(ma20_now) else True

        # 趋势判断（精细化）
        if ma20_slope > 0.5:
            sh_trend = "up"
        elif ma20_slope < -0.5:
            sh_trend = "down"
        elif ma20_slope > 0.1 and above_ma20 and ma10_above_ma20:
            # 均线微微上翘 + 价格在均线上方 + 短期均线多头排列
            sh_trend = "sideways_strong"  # 强势震荡
        elif ma20_slope < -0.1 and not above_ma20 and not ma10_above_ma20:
            # 均线微微下倾 + 价格在均线下方 + 短期均线空头排列
            sh_trend = "sideways_weak"    # 弱势震荡/下行整理
        else:
            sh_trend = "sideways"         # 中性震荡

        # 风险等级
        market_risk = "medium"
        if sh_trend in ("down", "sideways_weak") and sh_5d_return < -2:
            market_risk = "high"
        elif sh_trend in ("sideways", "sideways_strong") and vol_20 > 1.5 and sh_5d_return < -1.5:
            market_risk = "high"
        elif sh_trend == "up" and sh_5d_return > 3:
            market_risk = "low"
        elif sh_trend == "sideways_strong" and sh_5d_return > 0:
            market_risk = "low"

        pause_all = sh_today_pct < -1.5 or (sh_trend in ("down", "sideways_weak") and sh_5d_return < -3)
        pause_momentum = sh_trend in ("down", "sideways_weak") and sh_5d_return < -1

        # ========== 市场阶段分析与变盘概率 ==========
        phase_info = cls._analyze_market_phase(df, close, ma20_slope, ma20_now)

        result = {
            "sh_trend": sh_trend,
            "sh_ma20_slope": round(ma20_slope, 2),
            "sh_5d_return": round(sh_5d_return, 2),
            "sh_today_pct": round(sh_today_pct, 2),
            "sh_vol_20": round(vol_20, 2),
            "sh_above_ma20": above_ma20,
            "sh_above_ma60": above_ma60,
            "sh_ma10_above_ma20": ma10_above_ma20,
            "market_risk": market_risk,
            "pause_all": pause_all,
            "pause_momentum": pause_momentum,
            **phase_info,  # 合并市场阶段信息
        }
        cls._cache = result
        cls._cache_time = now
        return result

    @classmethod
    def _analyze_market_phase(cls, df: pd.DataFrame, close: float, ma20_slope: float, ma20_now: float) -> Dict:
        """分析市场阶段和震荡结束概率"""
        if df is None or len(df) < 40:
            return {
                "market_phase": "unknown",
                "phase_cn": "数据不足",
                "consolidation_end_prob": 0,
                "bb_width_ratio": 1.0,
                "volume_ratio": 1.0,
                "ma_spread_pct": 0.0,
                "price_position_pct": 50.0,
            }

        # 1. 布林带带宽计算
        df["bb_ma20"] = df["close"].rolling(20).mean()
        df["bb_std20"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_ma20"] + 2 * df["bb_std20"]
        df["bb_lower"] = df["bb_ma20"] - 2 * df["bb_std20"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_ma20"]

        latest = df.iloc[-1]
        bb_width_now = latest["bb_width"] if pd.notna(latest["bb_width"]) else 0.05
        bb_width_avg20 = df["bb_width"].iloc[-20:].mean()
        bb_width_avg20 = bb_width_avg20 if pd.notna(bb_width_avg20) and bb_width_avg20 > 0 else 0.05
        bb_width_ratio = bb_width_now / bb_width_avg20

        # 2. 成交量萎缩程度
        vol_now = latest["vol"]
        vol_ma20 = df["vol"].rolling(20).mean().iloc[-1]
        vol_ratio = vol_now / vol_ma20 if pd.notna(vol_ma20) and vol_ma20 > 0 else 1.0

        # 3. 均线粘合程度
        ma5_now = df["close"].rolling(5).mean().iloc[-1]
        ma10_now = df["close"].rolling(10).mean().iloc[-1]
        ma20_now = df["close"].rolling(20).mean().iloc[-1]
        ma_values = [v for v in [ma5_now, ma10_now, ma20_now] if pd.notna(v) and v > 0]
        if len(ma_values) >= 2:
            ma_spread_pct = (max(ma_values) / min(ma_values) - 1) * 100
        else:
            ma_spread_pct = 0.0

        # 4. 价格在近20日区间的位置
        high20 = df["high"].iloc[-20:].max()
        low20 = df["low"].iloc[-20:].min()
        if high20 > low20:
            price_position_pct = (close - low20) / (high20 - low20) * 100
        else:
            price_position_pct = 50.0

        # ========== 市场阶段判断 ==========
        # 趋势判断优先于震荡判断，不依赖布林带宽度
        if ma20_slope > 0.5 and close > ma20_now:
            market_phase = "trending_up"
            phase_cn = "趋势上涨中"
        elif ma20_slope < -0.5 and close < ma20_now:
            market_phase = "trending_down"
            phase_cn = "趋势下跌中"
        elif -0.5 <= ma20_slope <= 0.5:
            if bb_width_ratio < 0.7:
                market_phase = "consolidating_contracting"
                phase_cn = "震荡收缩中（可能即将变盘）"
            elif bb_width_ratio > 1.3:
                market_phase = "consolidating_expanding"
                phase_cn = "震荡扩张中（刚经历变盘）"
            else:
                market_phase = "consolidating_normal"
                phase_cn = "震荡维持中"
        elif ma20_slope > 0.3:
            market_phase = "trending_up_weak"
            phase_cn = "弱势上涨中"
        elif ma20_slope < -0.3:
            market_phase = "trending_down_weak"
            phase_cn = "弱势下跌中"
        else:
            market_phase = "transitioning"
            phase_cn = "趋势转换中"

        # ========== 震荡结束概率计算 ==========
        # 只在震荡阶段计算
        prob = 0
        if "consolidating" in market_phase or market_phase == "transitioning":
            # 因子1: 布林带收缩程度 (0-40分)
            if bb_width_ratio < 0.5:
                bb_score = 40
            elif bb_width_ratio < 0.7:
                bb_score = 30
            elif bb_width_ratio < 0.9:
                bb_score = 20
            elif bb_width_ratio < 1.1:
                bb_score = 10
            else:
                bb_score = 0

            # 因子2: 成交量萎缩 (0-20分)
            if vol_ratio < 0.5:
                vol_score = 20
            elif vol_ratio < 0.7:
                vol_score = 15
            elif vol_ratio < 0.9:
                vol_score = 10
            elif vol_ratio < 1.1:
                vol_score = 5
            else:
                vol_score = 0

            # 因子3: 均线粘合 (0-20分)
            if ma_spread_pct < 0.3:
                ma_score = 20
            elif ma_spread_pct < 0.6:
                ma_score = 15
            elif ma_spread_pct < 1.0:
                ma_score = 10
            elif ma_spread_pct < 2.0:
                ma_score = 5
            else:
                ma_score = 0

            # 因子4: 价格位置 (0-20分)
            # 越接近区间中部（40%-60%），越可能是在积蓄力量准备突破
            dist_from_center = abs(price_position_pct - 50)
            if dist_from_center < 10:
                pos_score = 20
            elif dist_from_center < 20:
                pos_score = 15
            elif dist_from_center < 30:
                pos_score = 10
            elif dist_from_center < 40:
                pos_score = 5
            else:
                pos_score = 0

            prob = bb_score + vol_score + ma_score + pos_score
            prob = min(100, max(0, prob))

        return {
            "market_phase": market_phase,
            "phase_cn": phase_cn,
            "consolidation_end_prob": prob,
            "bb_width_ratio": round(bb_width_ratio, 2),
            "volume_ratio": round(vol_ratio, 2),
            "ma_spread_pct": round(ma_spread_pct, 2),
            "price_position_pct": round(price_position_pct, 1),
        }


class StrategyMatcher:
    """市场环境与策略匹配度评估"""

    # 策略分组
    MEAN_REVERSION = {
        "RSI+布林带均值回归",
        "RSI+连跌中等信号",
        "KDJ超卖反弹",
        "多因子评分超卖",
        "深跌反弹",
        "MA支撑+KDJ超卖",
    }
    TREND_BREAKOUT = {
        "多因子买入策略",
        "动量策略",
        "N字突破",
        "底部抬高+温和放量",
        "缩量涨信号触发",
    }

    @classmethod
    def get_match_scores(cls, market_env: Optional[Dict]) -> Dict:
        """返回当前市场环境下各策略类型的匹配度(0-100)和建议仓位"""
        me = market_env or {}
        trend = me.get("sh_trend", "unknown")
        risk = me.get("market_risk", "medium")
        above_ma20 = me.get("sh_above_ma20", True)
        above_ma60 = me.get("sh_above_ma60", True)
        ma10_above_ma20 = me.get("sh_ma10_above_ma20", True)

        # 默认：中性
        mr_score = 50  # 均值回归
        tb_score = 50  # 趋势突破
        position = 50
        desc = "中性市场"

        if trend == "up":
            if risk == "low":
                mr_score, tb_score, position = 60, 100, 100
                desc = "上升趋势，低风险 — 趋势策略黄金期"
            elif risk == "medium":
                mr_score, tb_score, position = 65, 80, 85
                desc = "上升趋势，中等风险 — 趋势为主，回归为辅"
            else:
                mr_score, tb_score, position = 55, 60, 70
                desc = "上升趋势，但风险升高 — 控制仓位"
        elif trend == "sideways_strong":
            # 强势震荡：价格在均线上方，均线微微多头
            if risk == "low":
                mr_score, tb_score, position = 85, 60, 80
                desc = "强势震荡（偏多）— 回踩均线做多，突破可追"
            elif risk == "medium":
                mr_score, tb_score, position = 80, 40, 65
                desc = "强势震荡，中等风险 — 偏多做反弹，突破谨慎追"
            else:
                mr_score, tb_score, position = 65, 15, 45
                desc = "强势震荡转弱 — 降低仓位，减少追涨"
        elif trend == "sideways_weak":
            # 弱势震荡：价格在均线下方，均线微微下倾
            if risk == "low":
                mr_score, tb_score, position = 70, 15, 50
                desc = "弱势震荡（偏空）— 只做超跌反弹，不做突破"
            else:
                mr_score, tb_score, position = 50, 0, 30
                desc = "弱势震荡/下行整理 — 严控仓位，空仓或极小仓位"
        elif trend == "sideways":
            # 中性震荡：价格在均线附近来回穿越
            if risk == "low":
                mr_score, tb_score, position = 90, 35, 70
                desc = "中性震荡，低风险 — 均值回归策略主场"
            elif risk == "medium":
                mr_score, tb_score, position = 80, 20, 55
                desc = "中性震荡，中等风险 — 少做突破，多做反弹"
            else:
                mr_score, tb_score, position = 55, 0, 30
                desc = "震荡转弱，风险偏高 — 严控仓位，只做左侧"
        elif trend == "down":
            if risk == "low":
                mr_score, tb_score, position = 45, 15, 40
                desc = "弱势整理 — 观望为主，极小仓位博反弹"
            else:
                mr_score, tb_score, position = 25, 0, 15
                desc = "下跌趋势，高风险 — 空仓或极小仓位博反弹"
        else:
            desc = "大盘数据缺失，按中性处理"

        return {
            "mean_reversion_score": mr_score,
            "trend_breakout_score": tb_score,
            "position_pct": position,
            "market_desc": desc,
            "trend": trend,
            "risk": risk,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
            "ma10_above_ma20": ma10_above_ma20,
        }

    @classmethod
    def get_strategy_match(cls, strategy_name: str, market_env: Optional[Dict]) -> int:
        """获取单个策略的匹配度"""
        scores = cls.get_match_scores(market_env)
        base_name = strategy_name.split("(")[0].strip() if "(" in strategy_name else strategy_name
        if base_name in cls.MEAN_REVERSION:
            return scores["mean_reversion_score"]
        elif base_name in cls.TREND_BREAKOUT:
            return scores["trend_breakout_score"]
        return 50

    @classmethod
    def format_match_bar(cls, score: int) -> str:
        """用ASCII条形图显示匹配度"""
        filled = score // 5
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        return f"[{bar}] {score}%"


class AdaptiveParams:
    """根据当前波动率和大盘环境动态计算策略参数阈值"""

    @staticmethod
    def compute(vol_20: float, vol_60: float, market_env: Optional[Dict] = None) -> Dict:
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        me = market_env or {}
        market_risk = me.get("market_risk", "medium")
        sh_trend = me.get("sh_trend", "sideways")

        # 市场环境调节因子
        if market_risk == "high":
            risk_adj = 1.0  # 更保守
        elif market_risk == "low":
            risk_adj = -0.5  # 更激进
        else:
            risk_adj = 0.0

        # 下跌趋势中进一步收紧
        if sh_trend == "down":
            risk_adj += 1.0

        return {
            "rsi_entry": float(np.clip(33 - (vol_ratio - 1.0) * 6 - risk_adj * 3, 18, 45)),
            "bb_entry": float(np.clip(0.5 - (vol_ratio - 1.0) * 0.2 - risk_adj * 0.05, 0.05, 0.6)),
            "rsi_exit": float(np.clip(60 - (vol_ratio - 1.0) * 5 + risk_adj * 3, 50, 72)),
            "bb_exit": 0.75,
            "kdj_entry": float(np.clip(30 - (vol_ratio - 1.0) * 10 - risk_adj * 5, 8, 35)),
            "rsi_consec": float(np.clip(35 - (vol_ratio - 1.0) * 5 - risk_adj * 3, 22, 40)),
            "mom_vol_ratio": float(np.clip(1.2 + (vol_ratio - 1.0) * 0.3 + risk_adj * 0.2, 1.0, 2.5)),
            "vol_ratio_raw": round(vol_ratio, 2),
            "market_risk": market_risk,
            "sh_trend": sh_trend,
        }

    @staticmethod
    def from_df(df: pd.DataFrame, market_env: Optional[Dict] = None) -> Dict:
        vol_20 = df["pct_chg"].rolling(20).std().iloc[-1]
        vol_60 = df["pct_chg"].rolling(60).std().iloc[-1]
        vol_20 = vol_20 if pd.notna(vol_20) else 3.0
        vol_60 = vol_60 if pd.notna(vol_60) else 3.0
        return AdaptiveParams.compute(vol_20, vol_60, market_env)


class Strategy1_RSI_Bollinger:
    NAME = "RSI+布林带均值回归"
    STOCKS = [
        {"code": "300696.SZ", "name": "爱乐达", "sina_code": "sz300696"},
        {"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"},
        {"code": "002928.SZ", "name": "华夏航空", "sina_code": "sz002928"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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

        # 大盘环境
        me = market_env or MarketEnvironment.get_env()

        # ST炼石用旧版固定参数，其他用自适应参数
        is_st = "ST" in stock.get("name", "")
        stock_name = stock.get("name", "")
        ap = AdaptiveParams.from_df(df, me)
        if is_st:
            rsi_th, bb_th = 33, 0.5
        else:
            rsi_th, bb_th = ap["rsi_entry"], ap["bb_entry"]

        # 趋势过滤 (ST股不过滤)
        skip_trend_filter = is_st
        ma20_series = df["close"].rolling(20).mean()
        ma60_series = df["close"].rolling(60).mean()
        ma20_now = ma20_series.iloc[-1]
        ma60_now = ma60_series.iloc[-1] if len(ma60_series) >= 60 else ma20_now
        ma20_5ago = ma20_series.iloc[-6] if len(ma20_series) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
        ma20_downtrend = ma20_slope < -1.0
        # 新增: 个股明显下降趋势过滤 (close < ma60)
        stock_downtrend = rt["close"] < ma60_now * 0.98 if pd.notna(ma60_now) else False

        buy_cond_rsi = today_rsi < rsi_th
        buy_cond_bb = today_bb_pos < bb_th
        buy_cond_up = rt["close"] > rt["pre_close"]
        buy_cond_yang = rt["close"] > rt["open"]
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        buy_signal = (buy_cond_rsi or buy_cond_bb) and buy_cond_up and buy_cond_yang
        if not skip_trend_filter:
            buy_signal = buy_signal and not ma20_downtrend and not stock_downtrend

        # 大盘高风险时禁止买入
        if me.get("pause_all"):
            buy_signal = False

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
            "ma60_now": round(ma60_now, 2) if pd.notna(ma60_now) else None,
            "stock_downtrend": stock_downtrend,
            "adaptive_params": ap,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "profit_pct": profit_pct,
            "market_risk": me.get("market_risk", "medium"),
            "pause_all": me.get("pause_all", False),
        }


class Strategy2_MA_KDJ:
    NAME = "MA支撑+KDJ超卖"
    STOCKS = [{"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"}]
    DELAY = 3  # ST炼石延迟3天入场
    DELAY_THRESHOLD = 2  # 涨超2%放弃

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        me = market_env or MarketEnvironment.get_env()
        ap = AdaptiveParams.from_df(df, me)
        latest = df.iloc[-1]

        # 延迟入场: 检查3天前是否触发信号 + 至今未涨超2%
        if len(df) < cls.DELAY + 5:
            return None
        sig_day = df.iloc[-1 - cls.DELAY]
        sig_signals = []
        if (
            pd.notna(sig_day.get("ma20"))
            and abs(sig_day["close"] - sig_day["ma20"]) / sig_day["ma20"] < 0.02
            and sig_day["pct_chg"] > 0
        ):
            sig_signals.append("MA20支撑")
        sig_ap = AdaptiveParams.from_df(df.iloc[:-cls.DELAY], me)
        if pd.notna(sig_day.get("j")) and sig_day["j"] < sig_ap["kdj_entry"]:
            sig_signals.append("KDJ超卖")

        sig_triggered = len(sig_signals) > 0
        rise = (latest["close"] - sig_day["close"]) / sig_day["close"] * 100
        not_risen = rise < cls.DELAY_THRESHOLD

        buy_signal = sig_triggered and not_risen

        # 大盘高风险时禁止买入
        if me.get("pause_all"):
            buy_signal = False

        display_signals = []
        if sig_triggered:
            display_signals.append(f"3天前触发({','.join(sig_signals)})")
        if not_risen:
            display_signals.append(f"未涨({rise:+.1f}%<{cls.DELAY_THRESHOLD}%)")
        else:
            display_signals.append(f"已涨{rise:+.1f}%放弃")

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "ma20": latest["ma20"],
            "j_value": latest["j"],
            "signals": display_signals,
            "adaptive_params": ap,
            "buy_signal": buy_signal,
            "market_risk": me.get("market_risk", "medium"),
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
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        me = market_env or MarketEnvironment.get_env()
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

        buy_signal = len(signals) > 0

        # 个股趋势过滤: 60日均线下方禁止买入
        ma60_now = latest["ma60"] if pd.notna(latest.get("ma60")) else None
        if ma60_now is not None and latest["close"] < ma60_now * 0.98:
            buy_signal = False

        # 大盘高风险或暂停动量时禁止买入
        if me.get("pause_all") or me.get("pause_momentum"):
            buy_signal = False

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": latest["close"],
            "pct_chg": latest["pct_chg"],
            "volume_ratio": latest["volume_ratio"],
            "rsi": latest["rsi"],
            "signals": signals,
            "buy_signal": buy_signal,
            "market_risk": me.get("market_risk", "medium"),
            "pause_reason": "大盘暂停" if me.get("pause_all") or me.get("pause_momentum") else "",
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
        "603912.SH": {"name": "佳力图", "sina_code": "sh603912"},
        "000682.SZ": {"name": "东方电子", "sina_code": "sz000682"},
    }

    @classmethod
    def analyze(cls, ts_code: str, stock_info: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        me = market_env or MarketEnvironment.get_env()
        # 扬农化工和拓日新能用固定阈值，其他用自适应
        ap = AdaptiveParams.from_df(df, me)
        if stock_info["name"] in ("扬农化工", "拓日新能", "佳力图"):
            rsi_th = 35
        else:
            rsi_th = ap["rsi_consec"]
        latest = df.iloc[-1]
        buy_signal = latest["rsi"] <= rsi_th and latest["consecutive_down"] >= 2

        # 个股趋势过滤: 如果股票在60日均线下方且大盘趋势向下，禁止买入
        ma60_series = df["close"].rolling(60).mean()
        ma60_now = ma60_series.iloc[-1] if len(ma60_series) >= 60 else None
        stock_downtrend = False
        if ma60_now is not None and pd.notna(ma60_now):
            stock_downtrend = latest["close"] < ma60_now * 0.97
        if stock_downtrend and me.get("sh_trend") == "down":
            buy_signal = False

        # 大盘高风险时禁止买入
        if me.get("pause_all"):
            buy_signal = False

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
            "stock_downtrend": stock_downtrend,
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy5_Momentum:
    NAME = "动量策略"
    STOCKS = [
        {"code": "002272.SZ", "sina_code": "sz002272", "name": "川润股份"},
        {"code": "300696.SZ", "sina_code": "sz300696", "name": "爱乐达"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        df["ma60"] = df["close"].rolling(60).mean()
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

        me = market_env or MarketEnvironment.get_env()
        # 英维克用旧版固定参数，其他用自适应
        stock_name = stock.get("name", "")
        ap = AdaptiveParams.from_df(df, me)
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

        # 趋势过滤
        ma20_now = latest["ma20"]
        ma60_now = latest["ma60"] if pd.notna(latest.get("ma60")) else ma20_now
        ma20_5ago = df["ma20"].iloc[-6] if len(df) > 5 else ma20_now
        ma20_slope = (ma20_now - ma20_5ago) / ma20_5ago * 100 if ma20_5ago > 0 else 0
        ma20_downtrend = ma20_slope < -1.0
        stock_downtrend = latest["close"] < ma60_now * 0.98 if pd.notna(ma60_now) else False

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

        # 单动量信号在下跌趋势中不触发
        if ma20_downtrend and signal_momentum_only:
            signal_momentum_only = False

        buy_signal = signal_strong or signal_momentum_only

        # 个股或大盘下降趋势中禁止买入
        if stock_downtrend or me.get("pause_momentum") or me.get("pause_all"):
            buy_signal = False

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
            "stock_downtrend": stock_downtrend,
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy6_ScoreModel:
    NAME = "多因子评分超卖"
    STOCKS = [
        {"code": "002218.SZ", "name": "拓日新能", "sina_code": "sz002218"},
        {"code": "300572.SZ", "name": "安车检测", "sina_code": "sz300572"},
    ]
    SCORE_THRESHOLD = 50

    @classmethod
    def compute_score(cls, df: pd.DataFrame, rt: Dict, sh_pct: float = 0, market_env: Optional[Dict] = None) -> int:
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
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=100)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        me = market_env or MarketEnvironment.get_env()

        # 获取大盘涨跌
        sh_pct = me.get("sh_today_pct", 0)

        score, rsi14, bb, j, cd = cls.compute_score(df, rt, sh_pct, me)
        buy_signal = score >= cls.SCORE_THRESHOLD

        # 大盘高风险时提高阈值
        if me.get("market_risk") == "high":
            buy_signal = score >= cls.SCORE_THRESHOLD + 5

        # 个股60日均线下方且大盘下行时禁止买入
        ma60 = df["close"].rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and rt["close"] < ma60 * 0.97 and me.get("sh_trend") == "down":
            buy_signal = False

        # 大盘暂停
        if me.get("pause_all"):
            buy_signal = False

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
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy7_KDJ_Bounce:
    NAME = "KDJ超卖反弹"
    STOCKS = [
        {"code": "002837.SZ", "name": "英维克", "sina_code": "sz002837"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        me = market_env or MarketEnvironment.get_env()

        j_value = latest["j"]
        is_up = today_pct > 0
        buy_signal = pd.notna(j_value) and j_value < 10 and is_up

        # 个股60日均线下方且大盘下行时禁止买入
        ma60 = df["close"].rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and rt["close"] < ma60 * 0.97 and me.get("sh_trend") == "down":
            buy_signal = False

        if me.get("pause_all"):
            buy_signal = False

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "j_value": j_value,
            "k_value": latest["k"],
            "buy_signal": buy_signal,
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy8_DeepDrop:
    NAME = "深跌反弹"
    STOCKS = [
        {"code": "300499.SZ", "name": "高澜股份", "sina_code": "sz300499"},
        {"code": "002272.SZ", "name": "川润股份", "sina_code": "sz002272"},
        {"code": "600418.SH", "name": "江淮汽车", "sina_code": "sh600418"},
        {"code": "300696.SZ", "name": "爱乐达", "sina_code": "sz300696"},
        {"code": "300572.SZ", "name": "安车检测", "sina_code": "sz300572"},
        {"code": "688223.SH", "name": "晶科能源", "sina_code": "sh688223"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
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
        me = market_env or MarketEnvironment.get_env()

        # 5日跌幅
        ret5d = (latest["close"] / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        rsi14 = latest["rsi"]

        # 信号A: 5日跌>5% & RSI<40 & 当日涨, T+1/T+4
        signal_a = ret5d < -5 and pd.notna(rsi14) and rsi14 < 40 and today_pct > 0
        # 信号B: 5日跌>10%
        signal_b = ret5d < -10 and stock["name"] in ("高澜股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源")

        buy_signal = signal_a or signal_b

        # 大盘大幅下跌时，深跌反弹策略更容易失效，提高RSI要求
        if me.get("sh_trend") == "down" and me.get("sh_5d_return", 0) < -2:
            if pd.notna(rsi14) and rsi14 >= 35:
                buy_signal = False

        # 个股处于明显下降趋势中禁止买入
        ma60 = df["close"].rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and latest["close"] < ma60 * 0.95:
            buy_signal = False

        if me.get("pause_all"):
            buy_signal = False

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
            "buy_signal": buy_signal,
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy9_HigherLowVolume:
    NAME = "底部抬高+温和放量"
    STOCKS = [
        {"code": "002272.SZ", "name": "川润股份", "sina_code": "sz002272"},
        {"code": "002831.SZ", "name": "裕同科技", "sina_code": "sz002831"},
        {"code": "002218.SZ", "name": "拓日新能", "sina_code": "sz002218"},
        {"code": "000697.SZ", "name": "ST炼石", "sina_code": "sz000697"},
        {"code": "002928.SZ", "name": "华夏航空", "sina_code": "sz002928"},
        # 英维克该策略胜率仅44.4%，暂停使用
        # {"code": "002837.SZ", "name": "英维克", "sina_code": "sz002837"},
        {"code": "603912.SH", "name": "佳力图", "sina_code": "sh603912"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], 14)
        latest = df.iloc[-1]
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100
        me = market_env or MarketEnvironment.get_env()

        # 底部抬高: 近5日低点 > 近15日低点
        low5 = df["low"].iloc[-5:].min()
        low15 = df["low"].iloc[-15:].min()
        higher_low = low5 > low15

        # 温和放量: 近3日量均 > 近10日量均 * 1.2
        vol3 = df["vol"].iloc[-3:].mean()
        vol10 = df["vol"].iloc[-10:].mean()
        vol_expand = vol3 > vol10 * 1.2 if vol10 > 0 else False
        vol_ratio = vol3 / vol10 if vol10 > 0 else 0

        # RSI在中性偏多区 (45-65)
        rsi14 = latest["rsi"]
        rsi_ok = pd.notna(rsi14) and 45 <= rsi14 <= 65

        # 今日收阳
        today_bullish = rt["close"] > rt["open"]

        buy_signal = higher_low and vol_expand and rsi_ok and today_bullish

        # 个股趋势过滤: 60日均线下方禁止买入
        ma60 = df["close"].rolling(60).mean().iloc[-1]
        if pd.notna(ma60) and rt["close"] < ma60 * 0.98:
            buy_signal = False

        # 大盘暂停动量
        if me.get("pause_all") or me.get("pause_momentum"):
            buy_signal = False

        conditions = []
        if higher_low:
            conditions.append(f"底部抬高(5日低{low5:.2f}>15日低{low15:.2f})")
        if vol_expand:
            conditions.append(f"温和放量({vol_ratio:.2f}x)")
        if rsi_ok:
            conditions.append(f"RSI{rsi14:.0f}中性")
        if today_bullish:
            conditions.append(f"收阳")

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "rsi": rsi14,
            "low5": low5,
            "low15": low15,
            "vol_ratio_3_10": vol_ratio,
            "higher_low": higher_low,
            "vol_expand": vol_expand,
            "rsi_ok": rsi_ok,
            "today_bullish": today_bullish,
            "conditions": conditions,
            "buy_signal": buy_signal,
            "market_risk": me.get("market_risk", "medium"),
        }


class Strategy10_NBreakout:
    """N字突破: 回调后再次突破前高, 趋势延续的加速信号
    【警告】该策略在震荡市中假突破极多，历史胜率低(高澜47.6%, 华测62.5%)，建议谨慎使用
    """
    NAME = "N字突破"
    STOCKS = [
        # 高澜N字突破胜率仅47.6%，暂停使用
        # {"code": "300627.SZ", "name": "华测导航", "sina_code": "sz300627"},
        # {"code": "300499.SZ", "name": "高澜股份", "sina_code": "sz300499"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
        if df is None or len(df) < 30:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        # 找 B 点 (近15-3天最高点) 和 A 点 (B之前的低点)
        window_b = df.iloc[-15:-3]
        if len(window_b) < 10:
            return None
        b_high = window_b["high"].max()
        b_idx_local = window_b["high"].idxmax()
        b_pos = window_b.index.get_loc(b_idx_local)
        a_low = window_b.iloc[:max(b_pos, 1)]["low"].min()

        # 近4天低点 (C点回调)
        last_few = df.iloc[-4:]
        c_low = last_few["low"].min()

        uptrend_ok = (b_high - a_low) / a_low >= 0.05 if a_low > 0 else False
        pullback_pct = (b_high - c_low) / (b_high - a_low) * 100 if b_high > a_low else 0
        pullback_ok = 25 <= pullback_pct <= 70 and c_low < b_high

        today_break = rt["close"] > b_high
        today_bullish = rt["close"] > rt["open"]
        vol_ma5 = df["vol"].iloc[-6:-1].mean()
        vol_ok = df["vol"].iloc[-1] >= vol_ma5 * 1.0 if vol_ma5 > 0 else False

        buy_signal = uptrend_ok and pullback_ok and today_break and today_bullish and vol_ok

        conditions = []
        if uptrend_ok:
            conditions.append(f"A→B涨{(b_high-a_low)/a_low*100:.1f}%")
        if pullback_ok:
            conditions.append(f"回调{pullback_pct:.0f}%")
        if today_break:
            conditions.append(f"破B高{b_high:.2f}")
        if today_bullish:
            conditions.append("收阳")
        if vol_ok:
            conditions.append(f"量比{df['vol'].iloc[-1]/vol_ma5:.2f}")

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "a_low": a_low,
            "b_high": b_high,
            "c_low": c_low,
            "pullback_pct": pullback_pct,
            "conditions": conditions,
            "buy_signal": buy_signal,
        }


class Strategy11_ShrinkVolumeRise:
    """川润专用-缩量涨信号触发: 量比<0.8 + 涨>1% + MA20以上
    【警告】该策略历史胜率仅52.6%，在震荡市中容易诱多后补跌，建议谨慎使用
    """
    NAME = "缩量涨信号触发"
    STOCKS = [
        # 川润缩量涨策略胜率低，暂停使用
        # {"code": "002272.SZ", "name": "川润股份", "sina_code": "sz002272"},
    ]

    @classmethod
    def analyze(cls, stock: Dict, market_env: Optional[Dict] = None) -> Optional[Dict]:
        df = DataFetcher.fetch_history_data(stock["code"], days=60)
        if df is None or len(df) < 25:
            return None
        rt = DataFetcher.fetch_realtime_sina(stock["sina_code"])
        if rt is None:
            return None
        df = DataFetcher.merge_realtime_data(df, rt)
        today_pct = (rt["close"] - rt["pre_close"]) / rt["pre_close"] * 100

        # 5日均量
        vol_ma5 = df["vol"].iloc[-6:-1].mean() if len(df) >= 6 else df["vol"].mean()
        vol_ratio = df["vol"].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1

        # MA20
        ma20 = df["close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else None

        # 条件
        shrink_ok = vol_ratio < 0.8
        rise_ok = today_pct > 1
        ma20_ok = pd.notna(ma20) and rt["close"] > ma20

        buy_signal = shrink_ok and rise_ok and ma20_ok

        conditions = []
        if shrink_ok:
            conditions.append(f"缩量(量比{vol_ratio:.2f}<0.8)")
        if rise_ok:
            conditions.append(f"涨{today_pct:+.2f}%>1%")
        if ma20_ok:
            conditions.append(f"MA20上方({rt['close']:.2f}>{ma20:.2f})")

        return {
            "strategy": cls.NAME,
            "name": stock["name"],
            "code": stock["code"],
            "price": rt["close"],
            "pct_chg": today_pct,
            "vol_ratio": vol_ratio,
            "ma20": ma20 if pd.notna(ma20) else 0,
            "shrink_ok": shrink_ok,
            "rise_ok": rise_ok,
            "ma20_ok": ma20_ok,
            "conditions": conditions,
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


_DYNAMIC_WIN_RATES_CACHE: Optional[Dict] = None


def _load_dynamic_win_rates() -> Dict:
    """从 docs/win_rates.json 读取最新胜率。文件由 generate_trades_page.py 每次运行时重建。"""
    global _DYNAMIC_WIN_RATES_CACHE
    if _DYNAMIC_WIN_RATES_CACHE is not None:
        return _DYNAMIC_WIN_RATES_CACHE
    result: Dict = {}
    path = os.path.join("docs", "win_rates.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, info in data.get("rates", {}).items():
                if "|" not in key:
                    continue
                stock, strategy = key.split("|", 1)
                result[(stock, strategy)] = float(info.get("win_rate", 0))
        except Exception as e:
            print(f"[WARN] 读取 {path} 失败: {e}")
    _DYNAMIC_WIN_RATES_CACHE = result
    return result


def get_history_win_rate(stock_name: str, strategy_name: str) -> float:
    # 优先使用 docs/win_rates.json 中动态计算的胜率 (每次跑交易明细页时更新)
    dynamic = _load_dynamic_win_rates()
    if (stock_name, strategy_name) in dynamic:
        return dynamic[(stock_name, strategy_name)]

    # 回落: 研究期写死的快照胜率 (首次部署或 docs/win_rates.json 缺失时使用)
    # 自适应参数回测胜率 (2025-01-01起, ST炼石2025-05-07起)
    win_rates = {
        ("爱乐达", "RSI+布林带均值回归"): 83.3,
        ("爱乐达", "动量策略"): 64.7,
        ("ST炼石", "RSI+布林带均值回归"): 69.6,
        ("ST炼石", "MA支撑+KDJ超卖"): 66.0,
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
        ("华夏航空", "RSI+布林带均值回归"): 83.3,
        ("佳力图", "RSI+连跌中等信号"): 78.6,
        ("东方电子", "RSI+连跌中等信号"): 76.9,
        ("安车检测", "多因子评分超卖"): 85.2,
        ("安车检测", "深跌反弹"): 75.0,
        ("晶科能源", "深跌反弹"): 85.7,
        ("川润股份", "底部抬高+温和放量"): 75.0,
        ("裕同科技", "底部抬高+温和放量"): 100.0,
        ("拓日新能", "底部抬高+温和放量"): 70.6,
        ("ST炼石", "底部抬高+温和放量"): 83.3,
        ("华夏航空", "底部抬高+温和放量"): 85.7,
        ("英维克", "底部抬高+温和放量"): 60.0,
        ("佳力图", "底部抬高+温和放量"): 70.0,
        ("华测导航", "N字突破"): 76.9,
        ("高澜股份", "N字突破"): 68.4,
        ("川润股份", "缩量涨信号触发"): 50.0,
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
        ("华夏航空", "RSI+布林带均值回归"),  # T+0/T+5
        ("东方电子", "RSI+连跌中等信号"),  # T+0/T+5
        ("晶科能源", "深跌反弹"),  # 8A: T+0/T+4, 8B: T+0/T+4
    }
    # 特殊时机：川润动量T+0/T+4，川润深跌反弹T+1/T+4，晶科能源深跌T+0/T+4
    # 拓日新能/ST炼石 底部抬高 T+1/T+4 (方案A调整, 非t0_best -> T+1开盘/T+4开盘)
    t4_sell = {
        ("川润股份", "动量策略"),
        ("川润股份", "深跌反弹"),
        ("晶科能源", "深跌反弹"),
        ("拓日新能", "底部抬高+温和放量"),
        ("ST炼石", "底部抬高+温和放量"),
        ("华测导航", "N字突破"),
    }
    # 特殊时机：佳力图RSI+连跌 T+1/T+5 (方案A已将ST炼石底部抬高改为T+1/T+4)
    t1_t5 = {
        ("佳力图", "RSI+连跌中等信号"),
    }
    # 特殊时机：华夏航空 底部抬高 T+1/T+6
    t1_t6 = {
        ("华夏航空", "底部抬高+温和放量"),
    }
    # 特殊时机：高澜股份 N字突破 T+1/T+7, 川润缩量涨 T+1/T+7
    t1_t7 = {
        ("高澜股份", "N字突破"),
        ("川润股份", "缩量涨信号触发"),
    }
    # 特殊时机：英维克 底部抬高 T+1/T+8
    t1_t8 = {
        ("英维克", "底部抬高+温和放量"),
    }
    # 特殊时机：川润/裕同/佳力图 底部抬高 T+1/T+9
    t1_t9 = {
        ("川润股份", "底部抬高+温和放量"),
        ("裕同科技", "底部抬高+温和放量"),
        ("佳力图", "底部抬高+温和放量"),
    }
    if (stock_name, strategy_name) in t1_t6:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+6尾盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第6个交易日尾盘卖出",
        }
    if (stock_name, strategy_name) in t1_t8:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+8尾盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第8个交易日尾盘卖出",
        }
    if (stock_name, strategy_name) in t1_t5:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+5尾盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第5个交易日尾盘卖出",
        }
    if (stock_name, strategy_name) in t1_t7:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+7尾盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第7个交易日尾盘卖出",
        }
    if (stock_name, strategy_name) in t1_t9:
        return {
            "buy_timing": "T+1开盘",
            "sell_timing": "T+9尾盘",
            "buy_desc": "次日开盘买入",
            "sell_desc": "最晚第9个交易日尾盘卖出",
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
    print("股票策略监控系统 - 统一版 (改进版v2)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


def print_market_env(me: Dict):
    print(f"\n【市场环境】")
    trend = me.get('sh_trend', 'unknown')
    trend_cn = {
        "up": "上升趋势", "down": "下降趋势",
        "sideways": "中性震荡", "sideways_strong": "强势震荡（偏多）",
        "sideways_weak": "弱势震荡（偏空）", "unknown": "未知"
    }.get(trend, trend)
    print(f"  大盘趋势: {trend_cn} (MA20斜率{me.get('sh_ma20_slope', 0):.2f}%)")
    print(f"  上证位置: {'MA20上方' if me.get('sh_above_ma20') else 'MA20下方'}  {'MA10>MA20(多头)' if me.get('sh_ma10_above_ma20') else 'MA10<MA20(空头)'}")
    print(f"  上证5日涨跌: {me.get('sh_5d_return', 0):+.2f}%  今日: {me.get('sh_today_pct', 0):+.2f}%")
    print(f"  市场波动率: {me.get('sh_vol_20', 0):.2f}%  风险等级: {me.get('market_risk', 'medium')}")

    # 市场阶段与变盘概率
    phase = me.get('market_phase', 'unknown')
    phase_cn = me.get('phase_cn', '未知')
    prob = me.get('consolidation_end_prob', 0)
    if phase != "unknown":
        print(f"\n  [市场阶段] {phase_cn}")
        if "consolidating" in phase or phase == "transitioning":
            bar = "█" * (prob // 5) + "░" * (20 - prob // 5)
            print(f"  [变盘概率] [{bar}] {prob}%")
            print(f"    布林带收缩: {me.get('bb_width_ratio', 1.0):.2f}x  量能: {me.get('volume_ratio', 1.0):.2f}x")
            print(f"    均线离散: {me.get('ma_spread_pct', 0):.2f}%  价格位置: {me.get('price_position_pct', 50):.1f}%")
            if prob >= 70:
                print(f"    [!] 震荡可能即将结束，关注方向选择，建议控制仓位等待明朗")
            elif prob >= 50:
                print(f"    [注意] 震荡有收缩迹象，保持警惕")

    if me.get("pause_all"):
        print(f"\n  [!] 大盘风险高，已自动暂停部分买入信号")
    if me.get("pause_momentum"):
        print(f"  [!] 震荡/下行市，已自动暂停动量类策略")
    print("-" * 100)


def print_strategy_results(strategy_name: str, results: List[Dict], market_env: Optional[Dict] = None):
    if not results:
        # 对暂停的策略给出说明
        if strategy_name == Strategy10_NBreakout.NAME:
            print(f"\n{'='*100}")
            print(f"【{strategy_name}】[已暂停]")
            print(f"  原因: 该策略在震荡市中假突破极多，高澜历史胜率仅47.6%，华测62.5%")
            print(f"  建议: 等待趋势明朗后恢复")
        elif strategy_name == Strategy11_ShrinkVolumeRise.NAME:
            print(f"\n{'='*100}")
            print(f"【{strategy_name}】[已暂停]")
            print(f"  原因: 川润缩量涨策略历史胜率仅52.6%，震荡市容易诱多后补跌")
            print(f"  建议: 等待趋势明朗后恢复")
        return
    print(f"\n{'=' * 100}")
    print(f"【{strategy_name}】")
    # 显示当前策略的市场匹配度
    match_score = StrategyMatcher.get_strategy_match(strategy_name, None)
    match_bar = StrategyMatcher.format_match_bar(match_score)
    print(f"  [市场环境匹配度] {match_bar}")
    print(f"{'=' * 100}")
    buy_signals = [r for r in results if r.get("buy_signal")]
    for r in results:
        signal_mark = "[买]" if r.get("buy_signal") else "[-]"
        name = r.get("name", "")
        code = r.get("code", "")
        price = r.get("price", 0)
        pct_chg = r.get("pct_chg", 0)
        win_rate = get_history_win_rate(name, strategy_name)
        # 如果被大盘过滤阻止，显示原因
        blocked = not r.get("buy_signal") and (r.get("pause_all") or r.get("pause_momentum") or r.get("stock_downtrend"))
        block_note = ""
        if blocked:
            if r.get("pause_all"):
                block_note = " [被大盘风险阻止]"
            elif r.get("pause_momentum"):
                block_note = " [被动量暂停阻止]"
            elif r.get("stock_downtrend"):
                block_note = " [被个股趋势阻止]"
        print(
            f"  {signal_mark} {name}({code}): 价格{price:.2f}, 涨幅{pct_chg:+.2f}%, 历史胜率{win_rate:.1f}%{block_note}"
        )
        if r.get("strategy") == Strategy1_RSI_Bollinger.NAME:
            ap = r.get("adaptive_params", {})
            print(
                f"       RSI={r.get('rsi', 0):.1f}, BB位置={r.get('bb_position', 0):.3f}"
                f"  [自适应: RSI<{ap.get('rsi_entry', 33):.0f} BB<{ap.get('bb_entry', 0.5):.2f} 波动比={ap.get('vol_ratio_raw', 1):.2f}x 风险={ap.get('market_risk', 'medium')}]"
            )
            if r.get("stock_downtrend"):
                print(f"       [过滤] 个股处于下降趋势(价格<MA60)")
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
            if r.get("pause_reason"):
                print(f"       [过滤] {r.get('pause_reason')}")
        elif r.get("strategy") == Strategy4_RSI_ConsecutiveDown.NAME:
            rsi_cond = "满足" if r.get("rsi_condition") else "不满足"
            cons_cond = "满足" if r.get("consecutive_condition") else "不满足"
            print(
                f"       RSI={r.get('rsi', 0):.1f}({rsi_cond}), 连跌={r.get('consecutive_down', 0)}天({cons_cond})"
            )
            if r.get("stock_downtrend"):
                print(f"       [过滤] 个股处于下降趋势+大盘下行，阻止买入")
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
            if r.get("stock_downtrend"):
                print(f"       [过滤] 个股处于下降趋势(价格<MA60)")
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
        elif r.get("strategy") == Strategy9_HigherLowVolume.NAME:
            rsi_s = f"{r.get('rsi', 0):.1f}" if pd.notna(r.get('rsi')) else "-"
            print(
                f"       RSI={rsi_s}, 5日低={r.get('low5', 0):.2f}, 15日低={r.get('low15', 0):.2f}, "
                f"量比(3/10)={r.get('vol_ratio_3_10', 0):.2f}"
            )
            conditions = r.get("conditions", [])
            if conditions:
                print(f"       满足: {', '.join(conditions)}")
        elif r.get("strategy") == Strategy11_ShrinkVolumeRise.NAME:
            print(
                f"       量比={r.get('vol_ratio', 0):.2f}, MA20={r.get('ma20', 0):.2f}, "
                f"涨幅={r.get('pct_chg', 0):+.2f}%"
            )
            conditions = r.get("conditions", [])
            if conditions:
                print(f"       满足: {', '.join(conditions)}")
        elif r.get("strategy") == Strategy10_NBreakout.NAME:
            print(
                f"       A低={r.get('a_low', 0):.2f}, B高={r.get('b_high', 0):.2f}, "
                f"C低={r.get('c_low', 0):.2f}, 回调{r.get('pullback_pct', 0):.0f}%"
            )
            conditions = r.get("conditions", [])
            if conditions:
                print(f"       满足: {', '.join(conditions)}")
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
    print("股票策略实时监控系统 - 尾盘专用 (改进版v2)")
    print(f"执行时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DEBUG] TUSHARE_TOKEN: {'已设置' if TUSHARE_TOKEN else '未设置'}")
    print("=" * 100)

    # 先获取市场环境
    market_env = MarketEnvironment.get_env()
    print_market_env(market_env)

    all_results = {}

    print("\n[策略1] RSI+布林带均值回归")
    results = []
    for stock in Strategy1_RSI_Bollinger.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy1_RSI_Bollinger.analyze(stock, market_env)
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
    print_strategy_results(Strategy1_RSI_Bollinger.NAME, results, market_env)

    print("\n[策略2] MA支撑+KDJ超卖")
    results = []
    for stock in Strategy2_MA_KDJ.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy2_MA_KDJ.analyze(stock, market_env)
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
    print_strategy_results(Strategy2_MA_KDJ.NAME, results, market_env)

    print("\n[策略3] 多因子买入策略")
    results = []
    for stock in Strategy3_MultiFactor.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy3_MultiFactor.analyze(stock, market_env)
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
    print_strategy_results(Strategy3_MultiFactor.NAME, results, market_env)

    print("\n[策略4] RSI+连跌中等信号")
    results = []
    for ts_code, stock_info in Strategy4_RSI_ConsecutiveDown.STOCKS.items():
        print(f"  分析 {stock_info['name']}...", end=" ")
        r = Strategy4_RSI_ConsecutiveDown.analyze(ts_code, stock_info, market_env)
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
    print_strategy_results(Strategy4_RSI_ConsecutiveDown.NAME, results, market_env)

    print("\n[策略5] 动量策略")
    results = []
    for stock in Strategy5_Momentum.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy5_Momentum.analyze(stock, market_env)
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
    print_strategy_results(Strategy5_Momentum.NAME, results, market_env)

    print("\n[策略6] 多因子评分超卖")
    results = []
    for stock in Strategy6_ScoreModel.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy6_ScoreModel.analyze(stock, market_env)
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
    print_strategy_results(Strategy6_ScoreModel.NAME, results, market_env)

    print("\n[策略7] KDJ超卖反弹")
    results = []
    for stock in Strategy7_KDJ_Bounce.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy7_KDJ_Bounce.analyze(stock, market_env)
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
    print_strategy_results(Strategy7_KDJ_Bounce.NAME, results, market_env)

    print("\n[策略8] 深跌反弹")
    results = []
    for stock in Strategy8_DeepDrop.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy8_DeepDrop.analyze(stock, market_env)
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
    print_strategy_results(Strategy8_DeepDrop.NAME, results, market_env)

    print("\n[策略9] 底部抬高+温和放量")
    results = []
    for stock in Strategy9_HigherLowVolume.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy9_HigherLowVolume.analyze(stock, market_env)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "底部抬高+温和放量")
            timing = get_trade_timing(stock["name"], "底部抬高+温和放量")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy9_HigherLowVolume.NAME] = results
    print_strategy_results(Strategy9_HigherLowVolume.NAME, results, market_env)

    print("\n[策略10] N字突破")
    results = []
    for stock in Strategy10_NBreakout.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy10_NBreakout.analyze(stock, market_env)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "N字突破")
            timing = get_trade_timing(stock["name"], "N字突破")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy10_NBreakout.NAME] = results
    print_strategy_results(Strategy10_NBreakout.NAME, results, market_env)

    print("\n[策略11] 川润专用-缩量涨信号触发")
    results = []
    for stock in Strategy11_ShrinkVolumeRise.STOCKS:
        print(f"  分析 {stock['name']}...", end=" ")
        r = Strategy11_ShrinkVolumeRise.analyze(stock, market_env)
        if r:
            results.append(r)
            win_rate = get_history_win_rate(stock["name"], "缩量涨信号触发")
            timing = get_trade_timing(stock["name"], "缩量涨信号触发")
            r["win_rate"] = win_rate
            r["timing"] = timing
            print(f"完成 - 量比{r.get('vol_ratio',0):.2f} {'买入信号' if r.get('buy_signal') else '无信号'}")
        else:
            print("失败")
    all_results[Strategy11_ShrinkVolumeRise.NAME] = results
    print_strategy_results(Strategy11_ShrinkVolumeRise.NAME, results, market_env)

    print_summary_with_timing(all_results, market_env)


def print_summary_with_timing(all_results: Dict[str, List], market_env: Optional[Dict] = None):
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

    # ========== 市场环境-策略匹配度报告 ==========
    me = market_env or {}
    match = StrategyMatcher.get_match_scores(me)

    print("\n" + "=" * 100)
    print("【市场环境 - 策略匹配度】")
    print("=" * 100)
    trend_cn = {
        "up": "上升趋势", "down": "下降趋势",
        "sideways": "中性震荡", "sideways_strong": "强势震荡（偏多）",
        "sideways_weak": "弱势震荡（偏空）", "unknown": "未知"
    }.get(match['trend'], match['trend'])
    print(f"\n  当前环境: {match['market_desc']}")
    print(f"  大盘趋势: {trend_cn}  |  风险等级: {match['risk']}")
    pos_info = []
    if match.get('above_ma20'):
        pos_info.append("收盘价>MA20")
    else:
        pos_info.append("收盘价<MA20")
    if match.get('ma10_above_ma20'):
        pos_info.append("MA10>MA20")
    else:
        pos_info.append("MA10<MA20")
    print(f"  技术特征: {'  '.join(pos_info)}")
    print()
    print(f"  {'均值回归策略':<16} {StrategyMatcher.format_match_bar(match['mean_reversion_score'])}")
    print(f"     (RSI+布林带、RSI+连跌、KDJ超卖、评分超卖、深跌反弹、MA+KDJ)")
    print()
    print(f"  {'趋势突破策略':<16} {StrategyMatcher.format_match_bar(match['trend_breakout_score'])}")
    print(f"     (多因子买入、动量策略、N字突破、底部抬高、缩量涨)")
    print()
    print(f"  {'建议总仓位':<16} {StrategyMatcher.format_match_bar(match['position_pct'])}")

    # 对买入信号列表标注匹配度
    if buy_list:
        print("\n  当前买入信号的环境适配度:")
        print("  " + "-" * 80)
        for r in buy_list:
            s_match = StrategyMatcher.get_strategy_match(r.get("strategy", ""), me)
            s_bar = "█" * (s_match // 10) + "░" * (10 - s_match // 10)
            warn = " ⚠️ 策略与环境不匹配" if s_match < 50 else ""
            print(f"    {r.get('name',''):<6} {r.get('strategy',''):<16} [{s_bar}] {s_match}%{warn}")
        print("  " + "-" * 80)

    # 市场阶段与变盘概率提示
    phase = me.get('market_phase', '')
    prob = me.get('consolidation_end_prob', 0)
    if phase and phase != 'unknown':
        phase_cn = me.get('phase_cn', '')
        print(f"\n【市场阶段】{phase_cn}")
        if prob > 0:
            bar = "█" * (prob // 5) + "░" * (20 - prob // 5)
            print(f"  震荡结束概率: [{bar}] {prob}%")
            if prob >= 70:
                print(f"  [!] 震荡即将结束，变盘在即！建议：")
                print(f"     - 控制仓位，不要重仓赌方向")
                print(f"     - 等待方向明确后再加大仓位")
                print(f"     - 若已有持仓，设好止损，谨防方向做错")
            elif prob >= 50:
                print(f"  [注意] 震荡有收缩迹象，保持警惕，适度减仓")
        elif "trending" in phase:
            print(f"  [当前处于趋势中，顺势操作]")

    # 市场环境提示
    if me.get("market_risk") == "high":
        print("\n[!] 当前市场处于高风险状态，建议控制仓位，谨慎操作")
    if me.get("pause_all"):
        print("[!] 大盘当日大跌或短期跌幅较大，部分买入信号已被自动过滤")
    if me.get("pause_momentum"):
        print("[!] 当前为震荡/下行市，动量类策略信号已被自动过滤")

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
            s_match = StrategyMatcher.get_strategy_match(strategy_name, me)
            match_note = " [策略与环境高度匹配]" if s_match >= 80 else ""
            if s_match < 50:
                match_note = " [⚠️ 策略与环境不匹配，建议降低仓位或观望]"
            print(f"  {i}. {name}({r.get('code', '')}){match_note}")
            print(f"     买入: {timing.get('buy_desc', '')}")
            print(f"     卖出: {timing.get('sell_desc', '')} (可提前止盈/止损)")
        # 止损提示
        print("\n【风险控制】")
        pos = match['position_pct']
        print(f"  1. 建议总仓位控制在 {pos}% 以下")
        print(f"  2. 单只仓位不超过总资金20%")
        print(f"  3. 买入后若亏损超过-5%，建议次日开盘止损")
        print(f"  4. 大盘若继续下跌，可提前止盈/止损，不必等到固定卖出日")
        if pos <= 30:
            print(f"  5. [重要] 当前市场环境恶劣，建议空仓或极小仓位参与")
    else:
        print("\n暂无买入信号")
        pos = match['position_pct']
        if pos <= 30:
            print(f"\n[建议] 市场环境恶劣(匹配度低)，空仓观望")
        elif pos <= 50:
            print(f"\n[建议] 市场环境一般，谨慎参与")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
