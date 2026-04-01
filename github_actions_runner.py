#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions 运行脚本
用于在 GitHub Actions 环境中运行股票监控并生成 HTML 报告
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

BEIJING_TZ = timezone(timedelta(hours=8))


def get_beijing_time():
    return datetime.now(BEIJING_TZ)


os.makedirs("docs", exist_ok=True)


def run_monitor(from_file=None):
    if from_file:
        try:
            with open(from_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read {from_file}: {e}")
            return ""

    result = subprocess.run(
        [sys.executable, "stock_monitor_unified_realtime.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"[WARN] Monitor exited with code {result.returncode}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"[STDERR] {line}")
    return result.stdout


def parse_output(output):
    import re

    strategy_configs = [
        {"name": "RSI+布林带均值回归", "short": "策略1", "color": "#3498DB"},
        {"name": "MA支撑+KDJ超卖", "short": "策略2", "color": "#9B59B6"},
        {"name": "多因子买入策略", "short": "策略3", "color": "#E67E22"},
        {"name": "RSI+连跌中等信号", "short": "策略4", "color": "#27AE60"},
        {"name": "动量策略", "short": "策略5", "color": "#E74C3C"},
        {"name": "多因子评分超卖", "short": "策略6", "color": "#8E44AD"},
        {"name": "KDJ超卖反弹", "short": "策略7", "color": "#16A085"},
        {"name": "深跌反弹", "short": "策略8", "color": "#D35400"},
    ]

    data = {
        "time": get_beijing_time().strftime("%Y-%m-%d %H:%M:%S"),
        "strategies": [],
        "summary": {"total_analyzed": 0, "total_buy": 0, "buy_list": []},
    }

    for cfg in strategy_configs:
        data["strategies"].append(
            {
                "name": cfg["name"],
                "short": cfg["short"],
                "color": cfg["color"],
                "stocks": [],
                "buy_count": 0,
                "total_count": 0,
            }
        )

    lines = output.split("\n")
    current_strategy_idx = -1
    in_summary = False

    for line in lines:
        strategy_match = re.search(r"\[\D*([12345678])\]", line)
        if strategy_match:
            strategy_num = int(strategy_match.group(1))
            current_strategy_idx = strategy_num - 1
            continue

        if "汇总" in line:
            in_summary = True
            current_strategy_idx = -1
            continue

        stock_match = re.search(
            r"\[([^\]]*)\]\s+(\S+?)\(([\d\.]+\.[A-Z]{2})\):.*?([\d.]+).*?([\-+\d.]+)%",
            line,
        )
        if stock_match and current_strategy_idx >= 0 and not in_summary:
            marker = stock_match.group(1)
            name = stock_match.group(2)
            code = stock_match.group(3)
            price = float(stock_match.group(4))
            pct_chg = float(stock_match.group(5))
            is_buy = marker != "-"

            existing = any(
                s["code"] == code
                for s in data["strategies"][current_strategy_idx]["stocks"]
            )

            if not existing:
                stock_info = {
                    "name": name,
                    "code": code,
                    "price": price,
                    "pct_chg": pct_chg,
                    "buy_signal": is_buy,
                    "details": [],
                }
                data["strategies"][current_strategy_idx]["stocks"].append(stock_info)
                if is_buy:
                    data["strategies"][current_strategy_idx]["buy_count"] += 1
                data["strategies"][current_strategy_idx]["total_count"] += 1
            continue

        if line.startswith("       ") and current_strategy_idx >= 0:
            detail = line.strip()
            if (
                detail
                and not detail.startswith("买入")
                and not detail.startswith("触发")
            ):
                stocks = data["strategies"][current_strategy_idx]["stocks"]
                if stocks:
                    stocks[-1]["details"].append(detail)

        if in_summary:
            summary_match = re.search(
                r"(\S+)\s+(\S.*\S)\s+([\d.]+)\s+([\d.]+)%?\s+(T\+\d+\S*)\s+(T\+\d+\S*)",
                line,
            )
            if summary_match and len(line) > 30:
                try:
                    data["summary"]["buy_list"].append(
                        {
                            "name": summary_match.group(1),
                            "strategy": summary_match.group(2).strip(),
                            "price": float(summary_match.group(3)),
                            "win_rate": summary_match.group(4),
                            "buy_timing": summary_match.group(5),
                            "sell_timing": summary_match.group(6),
                        }
                    )
                except:
                    pass

    data["summary"]["total_analyzed"] = sum(
        s["total_count"] for s in data["strategies"]
    )
    data["summary"]["total_buy"] = sum(s["buy_count"] for s in data["strategies"])

    return data


def generate_html(data):
    summary = data["summary"]
    time_str = data["time"]

    strategies_html = ""
    for strategy in data["strategies"]:
        has_buy = strategy["buy_count"] > 0
        stocks_html = ""

        if strategy["stocks"]:
            for stock in strategy["stocks"]:
                change_class = "up" if stock["pct_chg"] > 0 else "down"
                change_symbol = "+" if stock["pct_chg"] > 0 else ""
                buy_badge = (
                    '<span class="buy-badge">买入</span>' if stock["buy_signal"] else ""
                )
                name_class = "buy" if stock["buy_signal"] else ""

                details_html = ""
                if stock["details"]:
                    detail_text = (
                        stock["details"][0][:60] + "..."
                        if len(stock["details"][0]) > 60
                        else stock["details"][0]
                    )
                    details_html = f'<div class="stock-detail">{detail_text}</div>'

                stocks_html += f"""
                <div class="stock-item">
                    <div class="stock-info">
                        <div class="stock-name {name_class}">
                            {stock["name"]}
                            {buy_badge}
                        </div>
                        <div class="stock-code">{stock["code"]}</div>
                        {details_html}
                    </div>
                    <div class="stock-price">
                        <div class="price-value">¥{stock["price"]:.2f}</div>
                        <div class="price-change {change_class}">{change_symbol}{stock["pct_chg"]:.2f}%</div>
                    </div>
                </div>"""
        else:
            stocks_html = '<div class="empty-state"><div class="empty-icon">📭</div>暂无数据</div>'

        pulse_class = "pulse" if has_buy else ""

        strategies_html += f"""
        <div class="strategy-card">
            <div class="strategy-header" style="background: {strategy["color"]}" onclick="toggleStrategy(this)">
                <div class="strategy-title">
                    <span class="strategy-icon {pulse_class}"></span>
                    <span class="strategy-name">{strategy["short"]}: {strategy["name"]}</span>
                </div>
                <span class="strategy-badge">{strategy["buy_count"]}/{strategy["total_count"]}</span>
            </div>
            <div class="strategy-content active">
                {stocks_html}
            </div>
        </div>"""

    if summary["buy_list"]:
        rec_items_html = ""
        for rec in summary["buy_list"]:
            rec_items_html += f"""
                <div class="rec-item">
                    <div class="rec-header">
                        <span class="rec-stock">{rec["name"]}</span>
                        <span class="rec-rate">胜率 {rec["win_rate"]}%</span>
                    </div>
                    <div class="rec-strategy">{rec["strategy"]} · ¥{rec["price"]:.2f}</div>
                    <div class="rec-timing">⏰ {rec["buy_timing"]} / {rec["sell_timing"]}</div>
                </div>"""
        rec_content = rec_items_html
    else:
        rec_content = '<div class="no-signal">🎉 暂无买入信号<br><span style="font-size: 16px; opacity: 0.8;">继续观望，等待机会</span></div>'

    rec_html = f"""
        <div class="recommendation-section">
            <div class="rec-title">[买入推荐]</div>
            {rec_content}
        </div>"""

    buy_value_class = "buy" if summary["total_buy"] > 0 else ""
    signal_ratio = summary["total_buy"] / max(summary["total_analyzed"], 1) * 100

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>实时策略监控 - {time_str}</title>
    
    <!-- Open Graph / 微信分享卡片 -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://wateralways.github.io/stock-monitor/">
    <meta property="og:title" content="📈 股票策略监控 - {"买入信号: " + str(summary["total_buy"]) + "只" if summary["total_buy"] > 0 else "暂无信号"}">
    <meta property="og:description" content="分析{summary["total_analyzed"]}只股票 | 更新时间: {time_str}">
    <meta property="og:image" content="https://wateralways.github.io/stock-monitor/preview.png">
    
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="📈 股票策略监控">
    <meta name="twitter:description" content="分析{summary["total_analyzed"]}只股票 | {"买入信号: " + str(summary["total_buy"]) + "只" if summary["total_buy"] > 0 else "暂无信号"}">
    <meta name="twitter:image" content="https://wateralways.github.io/stock-monitor/preview.png">
    
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding-bottom: 30px;
        }}
        .header {{
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .header h1 {{ font-size: 24px; color: #333; margin-bottom: 5px; }}
        .header .time {{ font-size: 14px; color: #888; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 15px; }}
        .stats-card {{
            background: white;
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            text-align: center;
        }}
        .stat-item {{ padding: 10px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .stat-value.buy {{ color: #e74c3c; }}
        .stat-label {{ font-size: 13px; color: #888; margin-top: 5px; }}
        .strategy-card {{
            background: white;
            border-radius: 20px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .strategy-header {{
            padding: 18px 20px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        .strategy-title {{ display: flex; align-items: center; gap: 10px; }}
        .strategy-name {{ font-size: 16px; font-weight: 600; }}
        .strategy-badge {{
            background: rgba(255,255,255,0.3);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }}
        .strategy-content {{ padding: 15px 20px; }}
        .stock-item {{
            padding: 15px 0;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stock-item:last-child {{ border-bottom: none; }}
        .stock-info {{ flex: 1; }}
        .stock-name {{
            font-size: 17px;
            font-weight: 600;
            color: #333;
            margin-bottom: 3px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .stock-name.buy {{ color: #e74c3c; }}
        .stock-code {{ font-size: 13px; color: #999; }}
        .stock-detail {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            padding: 5px 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .buy-badge {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .stock-price {{ text-align: right; }}
        .price-value {{ font-size: 18px; font-weight: 600; color: #333; }}
        .price-change {{ font-size: 14px; font-weight: 600; margin-top: 3px; }}
        .price-change.up {{ color: #e74c3c; }}
        .price-change.down {{ color: #27ae60; }}
        .recommendation-section {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 10px 40px rgba(231, 76, 60, 0.3);
        }}
        .rec-title {{
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }}
        .rec-item {{
            background: rgba(255,255,255,0.25);
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 15px;
        }}
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .rec-stock {{ font-size: 28px; font-weight: 700; }}
        .rec-rate {{
            background: rgba(255,255,255,0.35);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 18px;
            font-weight: 700;
        }}
        .rec-strategy {{ font-size: 18px; opacity: 0.95; margin-bottom: 10px; }}
        .rec-timing {{ font-size: 16px; opacity: 0.9; }}
        .no-signal {{ text-align: center; padding: 40px; font-size: 22px; opacity: 0.9; font-weight: 600; }}
        .empty-state {{ text-align: center; padding: 40px 20px; color: #999; }}
        .footer {{ text-align: center; padding: 30px 20px; color: rgba(255,255,255,0.7); font-size: 13px; }}
        @keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
        .pulse {{ animation: pulse 2s infinite; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>实时策略监控</h1>
        <div class="time">{time_str}</div>
    </div>
    
    <div class="container">
        <div class="stats-card">
            <div class="stat-item">
                <div class="stat-value">{summary["total_analyzed"]}</div>
                <div class="stat-label">分析股票</div>
            </div>
            <div class="stat-item">
                <div class="stat-value {buy_value_class}">{summary["total_buy"]}</div>
                <div class="stat-label">买入信号</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{signal_ratio:.1f}%</div>
                <div class="stat-label">信号比例</div>
            </div>
        </div>
        
        {rec_html}
        {strategies_html}
        
        <div class="footer">
            <div><a href="strategy.html" style="color: #FFD700; text-decoration: none;">📖 查看策略说明文档</a></div>
            <div style="margin-top: 8px;"><a href="trades.html" style="color: #FFD700; text-decoration: none;">📋 查看历史交易明细</a></div>
            <div style="margin-top: 10px;">建议持仓周期: T+5 (5个交易日)</div>
            <div style="margin-top: 8px; opacity: 0.7;">⚠️ 投资有风险，策略仅供参考</div>
        </div>
    </div>
    
    <script>
        function toggleStrategy(header) {{
            const content = header.nextElementSibling;
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</body>
</html>"""

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    with open("docs/report.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    generate_preview_image(data)
    generate_strategy_doc()

    return "docs/index.html"


def generate_strategy_doc():
    strategies = [
        {
            "id": 1,
            "name": "RSI+布林带均值回归",
            "color": "#3498DB",
            "stocks": ["爱乐达", "ST炼石"],
            "win_rates": {"爱乐达": "83.3%", "ST炼石": "69.6%"},
            "entry": "RSI或布林带位置低于阈值，且当日上涨收阳。爱乐达用自适应阈值+MA20趋势过滤；ST炼石用固定阈值(RSI<33/BB<0.5)无趋势过滤",
            "exit": "RSI > 自适应卖出阈值(约50-70) 或 布林带位置 > 0.75",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "利用RSI超卖和布林带下轨支撑，捕捉短期反弹机会。爱乐达用自适应参数+MA20趋势过滤(胜率最高83.3%)；ST炼石用固定参数无过滤(回测胜率更优)。",
        },
        {
            "id": 2,
            "name": "MA支撑+KDJ超卖",
            "color": "#9B59B6",
            "stocks": ["ST炼石"],
            "win_rates": {"ST炼石": "60.9%"},
            "entry": "股价在MA20附近(±2%)且上涨，或KDJ的J值 < 自适应阈值(根据波动率动态计算，约10-35)",
            "exit": "J值 > 80 或获利5%以上",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "结合均线支撑和KDJ超卖信号，在技术支撑位附近寻找买入机会。KDJ阈值采用自适应参数，根据波动率环境动态调整。",
        },
        {
            "id": 3,
            "name": "多因子买入策略",
            "color": "#E67E22",
            "stocks": ["高澜股份", "英维克"],
            "win_rates": {"高澜股份": "60.8%", "英维克": "58.2%"},
            "entry": "量价配合(量比>1.5且涨幅>2%且站上MA5) / 动量加速(连涨且加速放量) / 放量突破(量比>2且涨幅>3%且站上MA20) / 布林带触及(跌破下轨) / 均线多头排列(MA5>MA10>MA20且MACD>0)",
            "exit": "触发任意卖出信号或持仓超过5天",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "综合多个技术指标，包括量价关系、动量、突破、布林带、均线排列等。任意一个子信号触发即产生买入信号，多信号共振时置信度更高。",
        },
        {
            "id": 4,
            "name": "RSI+连跌中等信号",
            "color": "#27AE60",
            "stocks": [
                "裕同科技",
                "扬农化工",
                "华测导航",
                "川润股份",
                "ST炼石",
                "高澜股份",
                "拓日新能",
            ],
            "win_rates": {
                "裕同科技": "78.6%",
                "扬农化工": "71.4%",
                "华测导航": "68.8%",
                "川润股份": "62.5%",
                "ST炼石": "100.0%",
                "高澜股份": "56.2%",
                "拓日新能": "85.7%",
            },
            "entry": "RSI <= 阈值(扬农化工/拓日新能固定35，其他自适应约25-40) 且连续下跌 >= 2天",
            "exit": "RSI > 50 或连续上涨2天",
            "timing": "部分股票T+0尾盘买入(高澜/裕同/扬农/华测/川润)，其余T+1开盘买入；对应T+5尾盘或T+6开盘卖出",
            "description": "捕捉连续下跌后的超跌反弹机会，RSI超卖确认下跌动能衰竭。扬农化工和拓日新能使用固定RSI<=35阈值(回测胜率更优)，其他股票采用自适应参数。不同股票根据回测结果采用不同的买卖时机。",
        },
        {
            "id": 5,
            "name": "动量策略",
            "color": "#E74C3C",
            "stocks": ["川润股份", "爱乐达"],
            "win_rates": {"川润股份": "59.3%", "爱乐达": "64.7%"},
            "entry": "5个子信号(量价配合/动量加速/超跌反弹/突破信号/MACD金叉)满足2个以上，或单独动量加速(非下跌趋势)。个股差异化：川润股份禁用突破和MACD；英维克禁用超跌反弹和突破；爱乐达禁用量价配合和突破",
            "exit": "触发任意卖出信号或持仓超过5天",
            "timing": "川润股份/英维克T+0尾盘买入，爱乐达T+1开盘买入；对应T+5尾盘或T+6开盘卖出",
            "description": "综合动量指标，捕捉上涨趋势中的加速信号。川润和爱乐达用自适应量比阈值+MA20趋势过滤；英维克用固定量比(1.2/1.3)无趋势过滤(回测胜率更优)。每只股票根据历史回测禁用容易产生假信号的子策略。",
        },
        {
            "id": 6,
            "name": "多因子评分超卖",
            "color": "#8E44AD",
            "stocks": ["拓日新能"],
            "win_rates": {"拓日新能": "86.4%"},
            "entry": "9个因子加权评分>=50分: RSI14(0-25), 布林带位置(0-20), KDJ J值(0-15), 连跌天数(0-15), 5日跌幅(0-10), 大盘涨(+5), 长下影线(+5), 相对弱势(+5), 缩量罚分(-8)",
            "exit": "持仓5个交易日后尾盘卖出",
            "timing": "T+0尾盘买入，T+5尾盘卖出",
            "description": "专为拓日新能设计的评分制超卖反弹策略。综合9个技术因子加权打分，评分越高超卖越严重。当评分>=50时触发买入，尾盘买入捕捉次日跳空高开。回测22笔交易，86.4%胜率，平均每笔+3%。缩量时扣分避免无量空跌。",
        },
        {
            "id": 7,
            "name": "KDJ超卖反弹",
            "color": "#16A085",
            "stocks": ["英维克"],
            "win_rates": {"英维克": "84.6%"},
            "entry": "KDJ的J值<10 且 当日收涨(确认止跌反弹)",
            "exit": "持仓5个交易日后尾盘卖出",
            "timing": "T+0尾盘买入，T+5尾盘卖出",
            "description": "专为英维克设计的KDJ超卖反弹策略。当J值低于10(深度超卖)且当日收涨(确认反弹启动)时买入。回测13笔交易，84.6%胜率，平均+2.6%。替代原动量策略(53.1%)，胜率提升31个百分点。",
        },
        {
            "id": 8,
            "name": "深跌反弹",
            "color": "#D35400",
            "stocks": ["高澜股份"],
            "win_rates": {"高澜(跌5%+RSI+涨)": "100%", "高澜(跌10%)": "81.8%"},
            "entry": "信号A: 5日跌>5% & RSI14<40 & 当日涨(T+0/T+4) | 信号B: 5日跌>10%(T+0/T+5)",
            "exit": "信号A: 4天后尾盘卖 | 信号B: 5天后尾盘卖",
            "timing": "信号A: T+0尾盘买/T+4尾盘卖 | 信号B: T+0尾盘买/T+5尾盘卖",
            "description": "专为高澜股份设计的深跌反弹策略。信号A(5日跌>5%+RSI超卖+当日反弹确认)5笔100%全赢+8%平均收益；信号B(5日跌>10%无条件买)22笔81.8%胜率+6.1%平均收益。替代原RSI布林带策略(50%/+0.03%)。",
        },
    ]

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>策略说明 - 股票监控系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding-bottom: 30px;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 { font-size: 22px; color: #333; }
        .header .back { position: absolute; left: 15px; top: 20px; color: #667eea; text-decoration: none; font-size: 14px; }
        .container { max-width: 600px; margin: 0 auto; padding: 15px; }
        .strategy-card {
            background: white;
            border-radius: 16px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .strategy-header {
            padding: 15px 20px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .strategy-title { font-size: 18px; font-weight: 600; }
        .strategy-body { padding: 15px 20px; }
        .info-row { margin-bottom: 12px; }
        .info-label { font-size: 13px; color: #888; margin-bottom: 3px; }
        .info-value { font-size: 15px; color: #333; line-height: 1.5; }
        .stocks-tag {
            display: inline-block;
            background: #f0f0f0;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 13px;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .win-rate { color: #27ae60; font-weight: 600; }
        .footer {
            text-align: center;
            padding: 20px;
            color: rgba(255,255,255,0.7);
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <a href="index.html" class="back">← 返回监控</a>
        <h1>📈 策略说明文档</h1>
    </div>
    <div class="container">
        <div style="text-align:center;margin-bottom:15px;">
            <a href="trades.html" style="display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:10px 24px;border-radius:25px;text-decoration:none;font-size:15px;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,0.4);">📋 查看历史交易明细</a>
        </div>
"""

    for s in strategies:
        stocks_html = "".join(
            [f'<span class="stocks-tag">{name}</span>' for name in s["stocks"]]
        )
        win_rates_html = " | ".join([f"{k}: {v}" for k, v in s["win_rates"].items()])

        html += f"""
        <div class="strategy-card">
            <div class="strategy-header" style="background: {s["color"]}">
                <span class="strategy-title">策略{s["id"]}: {s["name"]}</span>
            </div>
            <div class="strategy-body">
                <div class="info-row">
                    <div class="info-label">监控股票</div>
                    <div class="info-value">{stocks_html}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">历史胜率</div>
                    <div class="info-value win-rate">{win_rates_html}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">买入条件</div>
                    <div class="info-value">{s["entry"]}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">卖出条件</div>
                    <div class="info-value">{s["exit"]}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">交易时机</div>
                    <div class="info-value">{s["timing"]}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">策略说明</div>
                    <div class="info-value">{s["description"]}</div>
                </div>
            </div>
        </div>
"""

    html += """
        <div class="footer">
            <div>⚠️ 以上策略仅供参考，不构成投资建议</div>
            <div style="margin-top: 8px;">投资有风险，入市需谨慎</div>
        </div>
    </div>
</body>
</html>
"""

    with open("docs/strategy.html", "w", encoding="utf-8") as f:
        f.write(html)


def generate_preview_image(data):
    if not HAS_PIL:
        return

    summary = data["summary"]
    time_str = data["time"]

    img = Image.new("RGB", (1200, 630), "#667eea")

    gradient = Image.new("RGB", (1200, 630), "#764ba2")
    img = Image.blend(img, gradient, 0.5)

    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
        )
        stat_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
    except:
        title_font = ImageFont.load_default()
        stat_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text(
        (600, 80), "Stock Strategy Monitor", font=title_font, fill="white", anchor="mm"
    )

    draw.rectangle([50, 150, 1150, 400], fill="#7a8ed9", outline="white", width=2)

    stat_y = 220
    draw.text(
        (200, stat_y),
        str(summary["total_analyzed"]),
        font=stat_font,
        fill="white",
        anchor="mm",
    )
    draw.text(
        (200, stat_y + 45),
        "Analyzed",
        font=small_font,
        fill="#cccccc",
        anchor="mm",
    )

    buy_color = "#e74c3c" if summary["total_buy"] > 0 else "#27ae60"
    draw.text(
        (600, stat_y),
        str(summary["total_buy"]),
        font=stat_font,
        fill=buy_color,
        anchor="mm",
    )
    draw.text(
        (600, stat_y + 45),
        "Buy Signals",
        font=small_font,
        fill="#cccccc",
        anchor="mm",
    )

    ratio = summary["total_buy"] / max(summary["total_analyzed"], 1) * 100
    draw.text(
        (1000, stat_y), f"{ratio:.1f}%", font=stat_font, fill="white", anchor="mm"
    )
    draw.text(
        (1000, stat_y + 45),
        "Signal Ratio",
        font=small_font,
        fill="#cccccc",
        anchor="mm",
    )

    if summary["buy_list"]:
        rec_y = 450
        for rec in summary["buy_list"][:3]:
            rec_text = (
                f"{rec['name']} | {rec['strategy'][:20]}... | Win: {rec['win_rate']}%"
            )
            draw.text(
                (600, rec_y), rec_text, font=small_font, fill="#FFD700", anchor="mm"
            )
            rec_y += 40

    draw.text((600, 590), time_str, font=small_font, fill="#bbbbbb", anchor="mm")

    img.save("docs/preview.png", "PNG")
    print("Preview image generated: docs/preview.png")


def main():
    print("=" * 60)
    print("Stock Monitor - GitHub Actions Runner")
    print(f"Time: {get_beijing_time()}")
    print("=" * 60)

    from_file = None
    if "--from-file" in sys.argv:
        idx = sys.argv.index("--from-file")
        if idx + 1 < len(sys.argv):
            from_file = sys.argv[idx + 1]
            print(f"Reading monitor output from: {from_file}")

    try:
        output = run_monitor(from_file=from_file)
        if output:
            print(output[:3000])

        data = parse_output(output) if output else None

        if data and data["summary"]["total_analyzed"] > 0:
            html_file = generate_html(data)
            print(f"\n[OK] Report generated: {html_file}")
            print(f"Total stocks: {data['summary']['total_analyzed']}")
            print(f"Buy signals: {data['summary']['total_buy']}")
        else:
            print("[WARN] No valid data from monitor, generating empty report")
            empty_data = {
                "time": get_beijing_time().strftime("%Y-%m-%d %H:%M:%S"),
                "strategies": [
                    {"name": cfg["name"], "short": cfg["short"], "color": cfg["color"],
                     "stocks": [], "buy_count": 0, "total_count": 0}
                    for cfg in [
                        {"name": "RSI+布林带均值回归", "short": "策略1", "color": "#3498DB"},
                        {"name": "MA支撑+KDJ超卖", "short": "策略2", "color": "#9B59B6"},
                        {"name": "多因子买入策略", "short": "策略3", "color": "#E67E22"},
                        {"name": "RSI+连跌中等信号", "short": "策略4", "color": "#27AE60"},
                        {"name": "动量策略", "short": "策略5", "color": "#E74C3C"},
                        {"name": "多因子评分超卖", "short": "策略6", "color": "#8E44AD"},
                        {"name": "KDJ超卖反弹", "short": "策略7", "color": "#16A085"},
                        {"name": "深跌反弹", "short": "策略8", "color": "#D35400"},
                    ]
                ],
                "summary": {"total_analyzed": 0, "total_buy": 0, "buy_list": []},
            }
            html_file = generate_html(empty_data)
            print(f"[OK] Empty report generated: {html_file}")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
