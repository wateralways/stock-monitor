#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速回测: 使用新浪API获取历史资金流向，对比新旧过滤逻辑
Sina API无频率限制，比Tushare快得多
"""

import json, time, requests, sys, io
from datetime import datetime, timedelta
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ========== 旧过滤 ==========
def old_filter(recent_3):
    """旧版: 最新日或前日主力净流入 > 0"""
    if len(recent_3) < 2:
        return True, "数据不足"
    return (recent_3[-1] > 0 or recent_3[-2] > 0), \
           f"最新={recent_3[-1]/10000:.0f}万 前日={recent_3[-2]/10000:.0f}万"

# ========== 新过滤 ==========
def new_filter(recent_3):
    """新版: 三条件"""
    if len(recent_3) < 2:
        return True, "数据不足"

    # 条件1: 最新一天 > 0
    if recent_3[-1] > 0:
        return True, "条件1"

    # 条件2: 3日累计 > 0
    total_3d = sum(recent_3)
    if total_3d > 0:
        return True, "条件2"

    # 条件3: 流出减弱
    if len(recent_3) >= 3:
        improving = sum(1 for i in range(1, len(recent_3)) if recent_3[i] > recent_3[i-1])
        if improving >= 1 and recent_3[-1] > recent_3[0]:
            return True, "条件3"

    return False, "不满足"

# ========== 数据获取 ==========
CODES = {
    "300696.SZ": "sz300696", "000697.SZ": "sz000697", "002928.SZ": "sz002928",
    "002831.SZ": "sz002831", "600486.SH": "sh600486", "300627.SZ": "sz300627",
    "002272.SZ": "sz002272", "300499.SZ": "sz300499", "002218.SZ": "sz002218",
    "603912.SH": "sh603912", "000682.SZ": "sz000682", "300572.SZ": "sz300572",
    "002837.SZ": "sz002837", "600418.SH": "sh600418", "688223.SH": "sh688223",
}

def fetch_moneyflow_sina_hist(sina_code, pages=3):
    """从新浪获取历史资金流向，每页100条"""
    headers = {"Referer": "https://finance.sina.com.cn/"}
    all_records = {}
    for page in range(1, pages + 1):
        url = (
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"MoneyFlow.ssl_qsfx_zjlrqs?page={page}&num=100&sort=opendate&asc=0&daima={sina_code}"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.text) > 100:
                data = json.loads(r.text)
                if not data:
                    break
                for item in data:
                    all_records[item["opendate"]] = float(item.get("r0_net", 0))
        except Exception as e:
            print(f"  [ERROR] {sina_code} page={page}: {e}")
            break
        time.sleep(0.3)  # 礼貌延迟
    return all_records  # {date: big_net_amount}


def main():
    print("=== 主力资金过滤 新旧逻辑对比回测 ===\n")

    # 加载历史交易
    with open("docs/trades_data.json", "r", encoding="utf-8") as f:
        trades_data = json.load(f)

    MR_STRATEGIES = [
        "RSI+布林带均值回归", "MA支撑+KDJ超卖", "RSI+连跌中等信号",
        "多因子评分超卖", "KDJ超卖反弹", "深跌反弹",
    ]

    # 收集所有需要查询的股票
    needed_codes = set()
    for key in trades_data:
        for s in MR_STRATEGIES:
            if s in key:
                needed_codes.add(key.split("|")[0])

    print(f"需要查询 {len(needed_codes)} 只股票的资金流向数据\n")

    # 获取所有股票的历史资金数据
    all_mf_data = {}
    for i, ts_code in enumerate(needed_codes):
        sina_code = CODES.get(ts_code)
        if not sina_code:
            continue
        print(f"[{i+1}/{len(needed_codes)}] 获取 {ts_code} ({sina_code})...")
        records = fetch_moneyflow_sina_hist(sina_code, pages=3)
        all_mf_data[ts_code] = records
        print(f"  获取到 {len(records)} 条记录 ({min(records.keys()) if records else 'N/A'} ~ {max(records.keys()) if records else 'N/A'})")

    print("\n" + "=" * 70)
    print("  逐笔对比分析")
    print("=" * 70)

    # 对比分析
    results = {
        "both_pass": [],      # 新旧都放行
        "both_block": [],     # 新旧都阻止
        "old_block_new_pass": [],  # 旧阻止新放行 ← 关键
        "insufficient_data": [],   # 数据不足
    }

    for key, trades in trades_data.items():
        ts_code = key.split("|")[0]
        strategy = "|".join(key.split("|")[1:])

        is_mr = any(s in key for s in MR_STRATEGIES)
        if not is_mr:
            continue

        mf_data = all_mf_data.get(ts_code, {})
        if not mf_data:
            continue

        for t in trades:
            signal_date = t["signal"]
            if t.get("pending"):
                continue

            # 找信号日前3个交易日的资金数据
            signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
            available_dates = sorted(mf_data.keys())
            # 找到 <= signal_date 的最近3个交易日
            before = [d for d in available_dates if d <= signal_date]
            if len(before) < 2:
                results["insufficient_data"].append({**t, "ts_code": ts_code, "strategy": strategy})
                continue

            recent_dates = before[-3:] if len(before) >= 3 else before
            recent_flows = [mf_data[d] for d in recent_dates]

            old_ok, _ = old_filter(recent_flows)
            new_ok, new_reason = new_filter(recent_flows)

            trade_info = {
                **t, "ts_code": ts_code, "strategy": strategy,
                "signal_date": signal_date,
                "recent_flows": [f"{x/10000:.0f}万" for x in recent_flows],
                "new_reason": new_reason,
            }

            if old_ok and new_ok:
                results["both_pass"].append(trade_info)
            elif not old_ok and not new_ok:
                results["both_block"].append(trade_info)
            elif not old_ok and new_ok:
                results["old_block_new_pass"].append(trade_info)
            # old_ok and not new_ok should not happen (new is superset)

    # ========== 输出结果 ==========
    total = sum(len(v) for v in results.values())
    print(f"\n总交易笔数(有资金数据): {total}")
    print(f"  数据不足: {len(results['insufficient_data'])}")
    print(f"  新旧都放行: {len(results['both_pass'])}")
    print(f"  新旧都阻止: {len(results['both_block'])}")
    print(f"  ⚠️ 旧阻止新放行: {len(results['old_block_new_pass'])} ← 关键差异\n")

    if results["old_block_new_pass"]:
        print("=" * 70)
        print("  旧版阻止但新版放行的交易明细")
        print("=" * 70)
        wins = 0
        losses = 0
        total_pnl = 0
        for t in sorted(results["old_block_new_pass"], key=lambda x: x["signal_date"]):
            outcome = "WIN" if t["win"] else "LOSS"
            if t["win"]:
                wins += 1
            else:
                losses += 1
            total_pnl += t["pnl"]
            print(f"  {t['signal_date']} {t['ts_code']:12s} {t['strategy'][:30]:30s} "
                  f"pnl={t['pnl']:+6.2f}% {outcome:5s} 原因={t['new_reason']} "
                  f"资金={t['recent_flows']}")

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        print(f"\n  📊 新增放行: Win={wins} Loss={losses}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  累计盈亏: {total_pnl:+.2f}%")
        print(f"  平均盈亏: {total_pnl/(wins+losses):+.2f}% per trade")

        if win_rate >= 60 and total_pnl > 0:
            print(f"\n  🟢 新增放行交易质量良好! 胜率>{60}%且累计正收益")
        elif win_rate >= 50:
            print(f"\n  🟡 新增放行交易勉强可接受，需观察")
        else:
            print(f"\n  🔴 新增放行交易质量差，建议调整过滤条件")

    # 对比 only
    both_block_wins = sum(1 for t in results["both_block"] if t["win"])
    both_block_total = len(results["both_block"])
    if both_block_total > 0:
        print(f"\n  旧版已阻止的交易: {both_block_total}笔, 胜率={both_block_wins/both_block_total*100:.1f}%")
        print(f"  (这些是无论新旧都被过滤的交易，作为参考基准)")

    print(f"\n  总结:")
    print(f"  旧版通过率: {len(results['both_pass'])/total*100:.1f}%")
    print(f"  新版通过率: {(len(results['both_pass'])+len(results['old_block_new_pass']))/total*100:.1f}%")
    old_pass = len(results['both_pass'])
    new_pass = len(results['both_pass']) + len(results['old_block_new_pass'])
    print(f"  通过率变化: +{(new_pass-old_pass)/total*100:.1f}%")

if __name__ == "__main__":
    main()
