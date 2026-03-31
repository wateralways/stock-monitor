#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成历史交易明细 HTML 页面 (docs/trades.html)
"""

import sys
import io
import time
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TUSHARE_TOKEN = "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# ============================================================
# 技术指标 + 自适应参数 (与策略代码一致)
# ============================================================
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger(prices, window=20, num_std=2):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    position = (prices - lower) / (upper - lower)
    return upper, ma, lower, position

def calculate_kdj(df, n=9):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    df = df.copy()
    df["rsv"] = (df["close"] - low_min) / (high_max - low_min) * 100
    df["k"] = df["rsv"].ewm(com=2, adjust=False).mean()
    df["d"] = df["k"].ewm(com=2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df

def calculate_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_consecutive_days(pct_chg, direction="down"):
    result = pd.Series(0, index=pct_chg.index)
    for i in range(1, len(pct_chg)):
        if direction == "down" and pct_chg.iloc[i] < 0:
            result.iloc[i] = result.iloc[i - 1] + 1
        elif direction == "up" and pct_chg.iloc[i] > 0:
            result.iloc[i] = result.iloc[i - 1] + 1
        else:
            result.iloc[i] = 0
    return result

def adaptive_params(df):
    vol_20 = df["pct_chg"].rolling(20).std().iloc[-1]
    vol_60 = df["pct_chg"].rolling(60).std().iloc[-1]
    vol_20 = vol_20 if pd.notna(vol_20) else 3.0
    vol_60 = vol_60 if pd.notna(vol_60) else 3.0
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
    return {
        "rsi_entry": float(np.clip(33 - (vol_ratio - 1.0) * 6, 20, 45)),
        "bb_entry": float(np.clip(0.5 - (vol_ratio - 1.0) * 0.2, 0.1, 0.6)),
        "kdj_entry": float(np.clip(30 - (vol_ratio - 1.0) * 10, 10, 35)),
        "rsi_consec": float(np.clip(35 - (vol_ratio - 1.0) * 5, 25, 40)),
        "mom_vol_ratio": float(np.clip(1.2 + (vol_ratio - 1.0) * 0.3, 1.0, 2.0)),
    }


# ============================================================
# 策略信号函数 (与当前版一致)
# ============================================================
def check_s1(df, idx, name):
    if idx < 60: return False
    sub = df.iloc[:idx+1].copy()
    sub["rsi"] = calculate_rsi(sub["close"], 14)
    _, _, _, bb_pos = calculate_bollinger(sub["close"], 20, 2)
    sub["bb_position"] = bb_pos
    ap = adaptive_params(sub)
    skip = "ST" in name or name == "高澜股份"
    ma20 = sub["close"].rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] * 100 if len(ma20) > 5 and ma20.iloc[-6] > 0 else 0
    latest = sub.iloc[-1]
    rsi = latest["rsi"] if pd.notna(latest["rsi"]) else 50
    bb = latest["bb_position"] if pd.notna(latest["bb_position"]) else 0.5
    rsi_th = 33 if "ST" in name else ap["rsi_entry"]
    bb_th = 0.5 if "ST" in name else ap["bb_entry"]
    sig = (rsi < rsi_th or bb < bb_th) and latest["pct_chg"] > 0 and latest["close"] > latest["open"]
    if not skip: sig = sig and slope >= -1.0
    return sig

def check_s2(df, idx, name):
    if idx < 60: return False
    sub = df.iloc[:idx+1].copy()
    sub["ma20"] = sub["close"].rolling(20).mean()
    sub = calculate_kdj(sub)
    ap = adaptive_params(sub)
    l = sub.iloc[-1]
    s = []
    if pd.notna(l["ma20"]) and abs(l["close"]-l["ma20"])/l["ma20"] < 0.02 and l["pct_chg"] > 0: s.append(1)
    if pd.notna(l["j"]) and l["j"] < ap["kdj_entry"]: s.append(1)
    return len(s) > 0

def check_s3(df, idx, name):
    if idx < 60: return False
    sub = df.iloc[:idx+1].copy()
    for w in [5,10,20,60]: sub[f"ma{w}"] = sub["close"].rolling(w).mean()
    sub["volume_ratio"] = sub["vol"] / sub["vol"].rolling(20).mean()
    sub["rsi"] = calculate_rsi(sub["close"], 14)
    macd, _ = calculate_macd(sub["close"])
    sub["macd"] = macd
    _, _, bb_lower, _ = calculate_bollinger(sub["close"], 20, 2)
    sub["bb_lower"] = bb_lower
    sub["consecutive_up"] = calculate_consecutive_days(sub["pct_chg"], "up")
    l, p = sub.iloc[-1], sub.iloc[-2]
    s = []
    if l["volume_ratio"]>1.5 and l["pct_chg"]>2 and l["close"]>=l["ma5"]: s.append(1)
    if l["consecutive_up"]>=1 and l["pct_chg"]>p["pct_chg"] and l["volume_ratio"]>1.2: s.append(1)
    if l["volume_ratio"]>2 and l["pct_chg"]>3 and l["close"]>l["ma20"]: s.append(1)
    if l["close"]<l["bb_lower"] and p["close"]>=sub["bb_lower"].iloc[-2]: s.append(1)
    if l["ma5"]>l["ma10"]>l["ma20"] and not(p["ma5"]>p["ma10"]) and l["macd"]>0: s.append(1)
    return len(s) > 0

def check_s4(df, idx, name):
    if idx < 60: return False
    sub = df.iloc[:idx+1].copy()
    sub["rsi"] = calculate_rsi(sub["close"], 14)
    sub["cd"] = calculate_consecutive_days(sub["pct_chg"], "down")
    ap = adaptive_params(sub)
    l = sub.iloc[-1]
    rsi_th = 35 if name in ("扬农化工", "拓日新能") else ap["rsi_consec"]
    return l["rsi"] <= rsi_th and l["cd"] >= 2

def check_s5(df, idx, name):
    if idx < 60: return False
    sub = df.iloc[:idx+1].copy()
    sub["vr"] = sub["vol"] / sub["vol"].rolling(5).mean()
    sub["ma5"] = sub["close"].rolling(5).mean()
    sub["ma20"] = sub["close"].rolling(20).mean()
    sub["cu"] = calculate_consecutive_days(sub["pct_chg"], "up")
    delta = sub["close"].diff()
    g = delta.where(delta>0,0).rolling(6).mean()
    lo = (-delta.where(delta<0,0)).rolling(6).mean()
    sub["rsi6"] = 100-(100/(1+g/lo))
    sub["dif"] = sub["close"].ewm(span=12).mean()-sub["close"].ewm(span=26).mean()
    sub["dea"] = sub["dif"].ewm(span=9).mean()
    ap = adaptive_params(sub)
    if name == "英维克":
        mt = 1.2
        ob_th = 1.3
    else:
        mt = ap["mom_vol_ratio"]
        ob_th = mt * 1.1
    l, p = sub.iloc[-1], sub.iloc[-2]
    slope = (l["ma20"]-sub["ma20"].iloc[-6])/sub["ma20"].iloc[-6]*100 if len(sub)>5 and sub["ma20"].iloc[-6]>0 else 0
    down = slope < -1.0 if name != "英维克" else False
    svp = l["vr"]>1.5 and l["pct_chg"]>2 and l["close"]>=l["ma5"]
    sm = l["cu"]>=1 and l["pct_chg"]>p["pct_chg"] and l["vr"]>mt
    sos = l["rsi6"]<30 and l["pct_chg"]>3 and l["vr"]>ob_th
    sbr = l["close"]>l["ma20"] and p["close"]<=p["ma20"] and l["vr"]>1.5
    smc = l["dif"]>l["dea"] and p["dif"]<=p["dea"] and (l["dif"]-l["dea"])*2>0
    uvp,um,uos,ubr,umc = svp,sm,sos,sbr,smc
    vt = 1.5
    if name=="川润股份": vt=1.1; ubr=False; umc=False
    elif name=="英维克": vt=1.1; uos=False; ubr=False
    elif name=="爱乐达": vt=1.2; uvp=False; ubr=False
    if uvp: uvp = svp and l["vr"]>vt
    sc = sum([uvp,um,uos,ubr,umc])
    strong = sc >= 2
    mom_only = um and sc == 1
    if down and mom_only: mom_only = False
    return strong or mom_only


# ============================================================
# 回测引擎
# ============================================================
T0_BEST = {
    ("高澜股份","RSI+连跌中等信号"),("裕同科技","RSI+连跌中等信号"),
    ("扬农化工","RSI+连跌中等信号"),("华测导航","RSI+连跌中等信号"),
    ("川润股份","MA支撑+KDJ超卖"),("川润股份","RSI+连跌中等信号"),
    ("川润股份","动量策略"),("英维克","动量策略"),
}

def backtest(df, name, strategy, func, start):
    t0 = (name, strategy) in T0_BEST
    buy_off = 0 if t0 else 1
    sell_off = 5 if t0 else 6
    trades = []
    start_dt = pd.to_datetime(start)
    for idx in range(60, len(df)):
        sig_date = df.iloc[idx]["trade_date"]
        if sig_date < start_dt: continue
        if not func(df, idx, name): continue
        bi = idx + buy_off
        si = idx + buy_off + (sell_off - buy_off)

        # 买入日超出数据范围：信号触发但还没到买入日
        if bi >= len(df):
            trades.append({
                "signal": sig_date.strftime("%Y-%m-%d"),
                "buy": "待买入", "sell": "-",
                "bp": 0, "sp": 0, "pnl": 0,
                "win": None, "pending": True,
            })
            continue

        bp = df.iloc[idx]["close"] if buy_off == 0 else df.iloc[bi]["open"]
        buy_date = df.iloc[bi if buy_off > 0 else idx]["trade_date"].strftime("%Y-%m-%d")

        # 卖出日超出数据范围：已买入但还在持仓中
        if si >= len(df):
            # 用最新价计算浮动盈亏
            latest_price = df.iloc[-1]["close"]
            float_pnl = (latest_price - bp) / bp * 100
            trades.append({
                "signal": sig_date.strftime("%Y-%m-%d"),
                "buy": buy_date, "sell": "持仓中",
                "bp": round(bp, 2), "sp": round(latest_price, 2),
                "pnl": round(float_pnl, 2),
                "win": None, "pending": True,
            })
            continue

        sp = df.iloc[si]["close"] if t0 else df.iloc[si]["open"]
        pnl = (sp - bp) / bp * 100
        trades.append({
            "signal": sig_date.strftime("%Y-%m-%d"),
            "buy": df.iloc[bi if buy_off > 0 else idx]["trade_date"].strftime("%Y-%m-%d"),
            "sell": df.iloc[si]["trade_date"].strftime("%Y-%m-%d"),
            "bp": round(bp, 2), "sp": round(sp, 2),
            "pnl": round(pnl, 2), "win": pnl > 0, "pending": False,
        })
    return trades


# ============================================================
# HTML 生成
# ============================================================
COMBOS = [
    ("RSI+布林带均值回归", "#3498DB", check_s1, [
        ("300696.SZ", "爱乐达"), ("000697.SZ", "ST炼石"), ("300499.SZ", "高澜股份"),
    ]),
    ("MA支撑+KDJ超卖", "#9B59B6", check_s2, [
        ("000697.SZ", "ST炼石"),
    ]),
    ("多因子买入策略", "#E67E22", check_s3, [
        ("300499.SZ", "高澜股份"), ("002837.SZ", "英维克"),
    ]),
    ("RSI+连跌中等信号", "#27AE60", check_s4, [
        ("002831.SZ", "裕同科技"), ("600486.SH", "扬农化工"), ("300627.SZ", "华测导航"),
        ("002272.SZ", "川润股份"), ("000697.SZ", "ST炼石"), ("300499.SZ", "高澜股份"),
        ("002218.SZ", "拓日新能"),
    ]),
    ("动量策略", "#E74C3C", check_s5, [
        ("002272.SZ", "川润股份"), ("002837.SZ", "英维克"), ("300696.SZ", "爱乐达"),
    ]),
]


def generate_html(all_data):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>历史交易明细</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding-bottom: 30px;
        }}
        .header {{
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 22px; color: #333; margin-bottom: 4px; }}
        .header .sub {{ font-size: 13px; color: #888; }}
        .header .back {{ position: absolute; left: 15px; top: 20px; color: #667eea; text-decoration: none; font-size: 14px; }}
        .container {{ max-width: 700px; margin: 0 auto; padding: 15px; }}
        .nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }}
        .nav a {{
            background: rgba(255,255,255,0.2); color: white; padding: 6px 14px;
            border-radius: 20px; text-decoration: none; font-size: 13px;
            transition: background 0.2s;
        }}
        .nav a:hover, .nav a.active {{ background: rgba(255,255,255,0.4); }}
        .combo-card {{
            background: white;
            border-radius: 16px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .combo-header {{
            padding: 14px 18px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        .combo-title {{ font-size: 16px; font-weight: 600; }}
        .combo-stats {{ font-size: 13px; opacity: 0.9; }}
        .combo-body {{ display: none; }}
        .combo-body.open {{ display: block; }}
        .summary {{
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 10px; padding: 15px 18px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .summary-item {{ text-align: center; }}
        .summary-val {{ font-size: 20px; font-weight: 700; color: #333; }}
        .summary-val.green {{ color: #27ae60; }}
        .summary-val.red {{ color: #e74c3c; }}
        .summary-label {{ font-size: 11px; color: #999; margin-top: 2px; }}
        .trade-table {{
            width: 100%; border-collapse: collapse; font-size: 13px;
        }}
        .trade-table th {{
            padding: 10px 8px; text-align: center;
            background: #f8f9fa; color: #666; font-weight: 600; font-size: 12px;
            position: sticky; top: 0;
        }}
        .trade-table td {{
            padding: 9px 8px; text-align: center;
            border-bottom: 1px solid #f5f5f5;
        }}
        .trade-table tr:last-child td {{ border-bottom: none; }}
        .win {{ color: #e74c3c; font-weight: 600; }}
        .lose {{ color: #27ae60; font-weight: 600; }}
        .badge-win {{
            background: #fde8e8; color: #e74c3c;
            padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
        }}
        .badge-lose {{
            background: #e8f5e9; color: #27ae60;
            padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
        }}
        .badge-pending {{
            background: #fff3e0; color: #e67e22;
            padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
        }}
        .pending {{ color: #e67e22; font-weight: 600; }}
        .footer {{
            text-align: center; padding: 25px 20px;
            color: rgba(255,255,255,0.7); font-size: 13px;
        }}
        @media (max-width: 500px) {{
            .trade-table {{ font-size: 12px; }}
            .trade-table th, .trade-table td {{ padding: 7px 4px; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <a href="strategy.html" class="back">&larr; 策略说明</a>
        <h1>历史交易明细</h1>
        <div class="sub">回测区间: 2025-01-01 至今 (ST炼石: 2025-05-07起) &middot; 更新: {now}</div>
    </div>
    <div class="container">
        <div class="nav">
            <a href="index.html">实时监控</a>
            <a href="strategy.html">策略说明</a>
            <a href="#" class="active">交易明细</a>
        </div>
"""

    for strategy_name, color, _, stocks_trades in all_data:
        for stock_name, code, trades in stocks_trades:
            if not trades:
                continue
            closed = [t for t in trades if not t.get("pending")]
            pending = [t for t in trades if t.get("pending")]
            total_closed = len(closed)
            wins = sum(1 for t in closed if t["win"])
            wr = wins / total_closed * 100 if total_closed > 0 else 0
            avg_pnl = sum(t["pnl"] for t in closed) / total_closed if total_closed > 0 else 0
            total_pnl = sum(t["pnl"] for t in closed)
            total_all = len(trades)
            card_id = f"{code}_{strategy_name}".replace("+", "").replace(" ", "")

            wr_class = "green" if wr >= 60 else ("red" if wr < 50 else "")
            avg_class = "green" if avg_pnl > 0 else "red"
            pending_tag = f" + {len(pending)}持仓" if pending else ""

            html += f"""
        <div class="combo-card">
            <div class="combo-header" style="background:{color}" onclick="toggle('{card_id}')">
                <span class="combo-title">{stock_name} &middot; {strategy_name}</span>
                <span class="combo-stats">{wins}/{total_closed}{pending_tag} &middot; {wr:.0f}%</span>
            </div>
            <div class="combo-body" id="{card_id}">
                <div class="summary">
                    <div class="summary-item">
                        <div class="summary-val">{total_all}</div>
                        <div class="summary-label">总交易</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-val {wr_class}">{wr:.1f}%</div>
                        <div class="summary-label">胜率</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-val {avg_class}">{avg_pnl:+.2f}%</div>
                        <div class="summary-label">平均收益</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-val {avg_class}">{total_pnl:+.1f}%</div>
                        <div class="summary-label">累计收益</div>
                    </div>
                </div>
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>#</th><th>信号日</th><th>买入日</th><th>卖出日</th>
                            <th>买价</th><th>卖价</th><th>收益</th><th>结果</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for i, t in enumerate(trades, 1):
                is_pending = t.get("pending", False)
                if is_pending:
                    pnl_cls = "pending"
                    badge = "badge-pending"
                    result = "持仓"
                    bp_str = f"{t['bp']:.2f}" if t['bp'] > 0 else "-"
                    sp_str = f"{t['sp']:.2f}" if t['sp'] > 0 else "-"
                    pnl_str = f"{t['pnl']:+.2f}%" if t['bp'] > 0 else "-"
                    sell_str = t["sell"]
                    buy_str = t["buy"] if t["buy"] == "待买入" else t["buy"][5:]
                else:
                    pnl_cls = "win" if t["win"] else "lose"
                    badge = "badge-win" if t["win"] else "badge-lose"
                    result = "盈" if t["win"] else "亏"
                    bp_str = f"{t['bp']:.2f}"
                    sp_str = f"{t['sp']:.2f}"
                    pnl_str = f"{t['pnl']:+.2f}%"
                    sell_str = t["sell"][5:]
                    buy_str = t["buy"][5:]
                html += f"""                        <tr>
                            <td>{i}</td>
                            <td>{t["signal"][5:]}</td>
                            <td>{buy_str}</td>
                            <td>{sell_str}</td>
                            <td>{bp_str}</td>
                            <td>{sp_str}</td>
                            <td class="{pnl_cls}">{pnl_str}</td>
                            <td><span class="{badge}">{result}</span></td>
                        </tr>
"""

            html += """                    </tbody>
                </table>
            </div>
        </div>
"""

    html += """
        <div class="footer">
            <div>以上为历史回测数据，不代表未来表现</div>
            <div style="margin-top: 8px;">投资有风险，入市需谨慎</div>
        </div>
    </div>
    <script>
        function toggle(id) {
            var el = document.getElementById(id);
            el.classList.toggle('open');
        }
        // 默认展开第一个
        var first = document.querySelector('.combo-body');
        if (first) first.classList.add('open');
    </script>
</body>
</html>"""
    return html


def main():
    print("=" * 80)
    print("生成历史交易明细页面")
    print("=" * 80)

    # 获取数据
    all_codes = set()
    for _, _, _, stocks in COMBOS:
        for code, _ in stocks:
            all_codes.add(code)

    print("\n获取数据...")
    cache = {}
    for code in sorted(all_codes):
        print(f"  {code}...", end=" ", flush=True)
        try:
            df = pro.daily(ts_code=code, start_date="20240901")
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                cache[code] = df
                print(f"OK ({len(df)}条)")
            else:
                print("EMPTY")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

    # 回测
    print("\n回测中...")
    all_data = []
    for strategy_name, color, func, stocks in COMBOS:
        stock_results = []
        for code, name in stocks:
            if code not in cache:
                continue
            start = "2025-05-07" if "ST" in name else "2025-01-01"
            trades = backtest(cache[code], name, strategy_name, func, start)
            wins = sum(1 for t in trades if t["win"])
            total = len(trades)
            wr = wins/total*100 if total > 0 else 0
            print(f"  {name} + {strategy_name}: {total}笔, 胜率{wr:.1f}%")
            stock_results.append((name, code, trades))
        all_data.append((strategy_name, color, func, stock_results))

    # 生成 HTML
    html = generate_html(all_data)
    with open("docs/trades.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n已生成 docs/trades.html ({len(html)} bytes)")


if __name__ == "__main__":
    main()
