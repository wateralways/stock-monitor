#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测: 对比新旧主力资金过滤逻辑
- 旧逻辑: 最新日或前日主力净流入 > 0
- 新逻辑: 1)最新日流入>0  OR  2)3日累计>0  OR  3)流出趋势减弱

使用 Tushare moneyflow 接口获取历史资金流向数据 (限速1次/分钟)
"""

import sys
import io
import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not TUSHARE_TOKEN:
    print("[ERROR] TUSHARE_TOKEN not set")
    sys.exit(1)

import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ========== 旧过滤逻辑 ==========
def old_filter(mf_records: List[Dict]) -> Tuple[bool, str]:
    """旧版: 最新日或前日主力净流入 > 0"""
    if not mf_records or len(mf_records) < 2:
        return True, "数据不足，放行"

    today_mf = mf_records[-1]
    yesterday_mf = mf_records[-2]

    today_inflow = today_mf["big_net_amount"] > 0
    yesterday_inflow = yesterday_mf["big_net_amount"] > 0

    if today_inflow or yesterday_inflow:
        return True, "OK"
    else:
        return False, "近2日均净流出"

# ========== 新过滤逻辑 ==========
def new_filter(mf_records: List[Dict]) -> Tuple[bool, str]:
    """新版: 三条件"""
    if not mf_records or len(mf_records) < 2:
        return True, "数据不足，放行"

    latest_mf = mf_records[-1]

    # 条件1: 最新一天主力净流入 > 0
    if latest_mf["big_net_amount"] > 0:
        return True, "条件1:最新日流入"

    # 条件2: 近3天累计 > 0
    recent_3 = mf_records[-3:] if len(mf_records) >= 3 else mf_records
    total_3d = sum(r["big_net_amount"] for r in recent_3)
    if total_3d > 0:
        return True, "条件2:3日累计>0"

    # 条件3: 流出趋势减弱
    if len(recent_3) >= 3:
        amounts = [r["big_net_amount"] for r in recent_3]
        improving_steps = sum(1 for i in range(1, len(amounts)) if amounts[i] > amounts[i-1])
        if improving_steps >= 1 and amounts[-1] > amounts[0]:
            return True, "条件3:流出减弱"

    return False, "不满足"


# ========== 数据获取 ==========
CODE_TO_TSCODE = {
    "sz300696": "300696.SZ", "sz000697": "000697.SZ", "sz002928": "002928.SZ",
    "sz002831": "002831.SZ", "sh600486": "600486.SH", "sz300627": "300627.SZ",
    "sz002272": "002272.SZ", "sz300499": "300499.SZ", "sz002218": "002218.SZ",
    "sh603912": "603912.SH", "sz000682": "000682.SZ", "sz300572": "300572.SZ",
    "sz002837": "002837.SZ", "sh600418": "600418.SH", "sh688223": "688223.SH",
}

def fetch_moneyflow_ts(ts_code: str, start: str, end: str) -> List[Dict]:
    """获取历史资金流向 (Tushare)"""
    try:
        df = pro.moneyflow(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return []
        df = df.sort_values("trade_date").reset_index(drop=True)
        records = []
        for _, row in df.iterrows():
            big_net = (row.get("buy_lg_amount", 0) - row.get("sell_lg_amount", 0) +
                       row.get("buy_elg_amount", 0) - row.get("sell_elg_amount", 0))
            records.append({
                "date": str(row["trade_date"]),
                "big_net_amount": big_net,  # 万元
            })
        return records
    except Exception as e:
        print(f"  [ERROR] fetch_moneyflow_ts {ts_code}: {e}")
        return []


def main():
    # 加载历史交易数据
    trades_file = "docs/trades_data.json"
    if not os.path.exists(trades_file):
        print(f"[ERROR] {trades_file} not found")
        return

    with open(trades_file, "r", encoding="utf-8") as f:
        trades_data = json.load(f)

    # 筛选博反弹策略的交易
    MEAN_REVERSION_STRATEGIES = [
        "RSI+布林带均值回归",
        "MA支撑+KDJ超卖",
        "RSI+连跌中等信号",
        "多因子评分超卖",
        "KDJ超卖反弹",
        "深跌反弹",
    ]

    # 收集所有需要查询的 (ts_code, date) 组合
    queries = defaultdict(set)  # ts_code -> set of dates
    trades_list = []

    for key, trades in trades_data.items():
        for strategy_name in MEAN_REVERSION_STRATEGIES:
            if strategy_name in key:
                for t in trades:
                    ts_code = key.split("|")[0]
                    signal_date = t["signal"]  # YYYY-MM-DD
                    queries[ts_code].add(signal_date)
                    trades_list.append({
                        "ts_code": ts_code,
                        "strategy": strategy_name,
                        "signal_date": signal_date,
                        "pnl": t["pnl"],
                        "win": t["win"],
                        "pending": t.get("pending", False),
                    })

    print(f"共 {len(trades_list)} 笔博反弹策略交易")
    print(f"涉及 {len(queries)} 只股票\n")

    # 获取每只股票的资金流向数据
    all_mf_data = {}  # ts_code -> [{date, big_net_amount}, ...]

    for i, (ts_code, dates) in enumerate(queries.items()):
        min_date = min(dates)
        max_date = max(dates)
        # 前后扩展几天用于3日窗口
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=10)
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=5)
        start = min_date_obj.strftime("%Y%m%d")
        end = max_date_obj.strftime("%Y%m%d")

        print(f"[{i+1}/{len(queries)}] 获取 {ts_code} 资金流向 ({start}~{end})...")
        records = fetch_moneyflow_ts(ts_code, start, end)
        all_mf_data[ts_code] = records

        if i < len(queries) - 1:
            print(f"  等待65秒 (API限速)...")
            time.sleep(65)

    # ========== 对比分析 ==========
    old_pass = 0
    new_pass = 0
    old_block_new_pass = 0  # 旧版阻止但新版放行 (关键!)
    both_block = 0
    both_pass = 0

    affected_trades = []  # 旧版阻止但新版放行的交易

    for trade in trades_list:
        ts_code = trade["ts_code"]
        signal_date = trade["signal_date"]
        mf_records = all_mf_data.get(ts_code, [])

        # 取信号日前3天(含信号日)的资金数据
        signal_dt = pd.Timestamp(signal_date)
        before_signal = [r for r in mf_records
                         if pd.Timestamp(r["date"]) <= signal_dt
                         and (signal_dt - pd.Timestamp(r["date"])).days <= 4]

        if len(before_signal) < 2:
            continue  # 数据不足，无法判断

        # 取最近3天
        recent = before_signal[-3:] if len(before_signal) >= 3 else before_signal

        old_ok, old_reason = old_filter(recent)
        new_ok, new_reason = new_filter(recent)

        if old_ok:
            old_pass += 1
        if new_ok:
            new_pass += 1
        if old_ok and new_ok:
            both_pass += 1
        if not old_ok and not new_ok:
            both_block += 1
        if not old_ok and new_ok:
            old_block_new_pass += 1
            affected_trades.append({**trade, "old_reason": old_reason, "new_reason": new_reason,
                                     "recent_flows": [f"{r['big_net_amount']/10000:.0f}万" for r in recent]})

    total = old_pass + (len(trades_list) - old_pass)  # approximation
    analyzed = old_pass + (len(trades_list) - old_pass)  # those with sufficient data

    print("\n" + "=" * 70)
    print("  新旧过滤逻辑对比结果")
    print("=" * 70)
    print(f"  总交易数(有资金数据): {len(trades_list)}")
    print(f"  有效分析数: {old_pass + (both_block + old_block_new_pass)}")
    print()
    print(f"  旧版过滤:")
    print(f"    放行: {old_pass} 笔")
    print(f"    阻止: {both_block + old_block_new_pass} 笔")
    print()
    print(f"  新版过滤:")
    print(f"    放行: {new_pass} 笔 (+{old_block_new_pass} 笔 vs 旧版)")
    print(f"    阻止: {both_block} 笔")
    print()

    if old_block_new_pass > 0:
        print(f"  ⚠️ 旧版阻止但新版放行: {old_block_new_pass} 笔")
        print(f"  {'─' * 60}")
        wins = 0
        losses = 0
        for t in affected_trades:
            outcome = "✅ WIN" if t["win"] else "❌ LOSS"
            if t["win"]:
                wins += 1
            else:
                losses += 1
            print(f"  {t['signal_date']} {t['ts_code']:12s} {t['strategy']:20s} "
                  f"pnl={t['pnl']:+.2f}% {outcome}")
            print(f"    资金流向: {t['recent_flows']}")
            print(f"    旧版: {t['old_reason']} → 新版: {t['new_reason']}")

        print(f"\n  📊 这 {old_block_new_pass} 笔新增放行交易:")
        print(f"    Win: {wins}, Loss: {losses}")
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        print(f"    胜率: {win_rate:.1f}%")
        total_pnl = sum(t["pnl"] for t in affected_trades)
        print(f"    累计盈亏: {total_pnl:+.2f}%")

        if win_rate < 50:
            print(f"\n  🔴 警告: 新增放行的交易胜率较低 ({win_rate:.1f}%)，建议调整过滤条件")
        else:
            print(f"\n  🟢 新增放行的交易表现尚可 ({win_rate:.1f}%)")

    print(f"\n  整体影响评估:")
    print(f"    旧版通过率: {old_pass/len(trades_list)*100:.1f}%")
    print(f"    新版通过率: {new_pass/len(trades_list)*100:.1f}%")
    print(f"    通过率变化: +{(new_pass-old_pass)/len(trades_list)*100:.1f}%")

if __name__ == "__main__":
    main()
