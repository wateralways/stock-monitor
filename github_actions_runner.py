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
        strategy_match = re.search(r"\[\D*([12345])\]", line)
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
            "stocks": ["爱乐达", "ST炼石", "高澜股份"],
            "win_rates": {"爱乐达": "80%", "ST炼石": "79%", "高澜股份": "44%"},
            "entry": "RSI < 自适应阈值(20~45) 或 布林带位置 < 自适应阈值(0.1~0.6)，且当日上涨收阳。非ST股需MA20趋势非下跌。",
            "exit": "RSI > 自适应卖出阈值(50~70) 或 布林带位置 > 0.75",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "利用RSI超卖和布林带下轨支撑捕捉反弹。参数根据20日/60日波动率比值自动调节：高波动期收紧阈值避免接飞刀，低波动期放松阈值增加信号。ST股不做趋势过滤。",
            "adaptive": "RSI阈值 = 33 - (波动率比-1)×6 | BB阈值 = 0.5 - (波动率比-1)×0.2",
        },
        {
            "id": 2,
            "name": "MA支撑+KDJ超卖",
            "color": "#9B59B6",
            "stocks": ["ST炼石"],
            "win_rates": {"ST炼石": "50%"},
            "entry": "股价在MA20附近(±2%)且上涨，或KDJ的J值 < 自适应阈值(10~35)",
            "exit": "J值 > 80 或获利5%以上",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "结合均线支撑和KDJ超卖信号。KDJ超卖阈值随波动率自动调节，高波动期要求更深度超卖才触发。",
            "adaptive": "KDJ-J阈值 = 30 - (波动率比-1)×10",
        },
        {
            "id": 3,
            "name": "多因子买入策略",
            "color": "#E67E22",
            "stocks": ["高澜股份", "英维克"],
            "win_rates": {"高澜股份": "62%", "英维克": "59%"},
            "entry": "量价配合 / 动量加速 / 放量突破 / 布林带触及 / 均线多头排列",
            "exit": "触发任意卖出信号或持仓超过5天",
            "timing": "T+1开盘买入，T+6开盘卖出",
            "description": "综合多个技术指标，信号越多买入概率越高。",
            "adaptive": "无自适应（固定参数）",
        },
        {
            "id": 4,
            "name": "RSI+连跌中等信号",
            "color": "#27AE60",
            "stocks": [
                "裕同科技", "扬农化工", "华测导航",
                "川润股份", "英维克", "ST炼石", "高澜股份",
            ],
            "win_rates": {
                "裕同科技": "71%", "扬农化工": "80%",
                "华测导航": "62%", "川润股份": "78%",
                "英维克": "60%", "ST炼石": "100%",
                "高澜股份": "75%",
            },
            "entry": "RSI <= 自适应阈值(25~40) 且连续下跌 >= 2天",
            "exit": "RSI > 50 或连续上涨2天",
            "timing": "T+0尾盘买入(当日收盘前)，T+5尾盘卖出",
            "description": "捕捉连续下跌后的超跌反弹。RSI阈值随波动率自动调节，高波动期要求更低RSI才视为超卖。",
            "adaptive": "RSI阈值 = 35 - (波动率比-1)×5",
        },
        {
            "id": 5,
            "name": "动量策略",
            "color": "#E74C3C",
            "stocks": ["川润股份", "英维克", "爱乐达"],
            "win_rates": {"川润股份": "67%", "英维克": "62%", "爱乐达": "62%"},
            "entry": "量价配合+动量加速+超跌反弹+突破信号+MACD金叉，满足2个以上。单动量信号在MA20下跌趋势中不触发。",
            "exit": "触发任意卖出信号或持仓超过5天",
            "timing": "T+0尾盘买入(当日收盘前)，T+5尾盘卖出",
            "description": "综合动量指标捕捉上涨加速信号。量比阈值随波动率自动调节，高波动期要求更大量比确认。MA20下跌趋势中单动量信号被过滤。",
            "adaptive": "量比阈值 = 1.2 + (波动率比-1)×0.3",
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
        .intro-card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .intro-card h2 { font-size: 18px; color: #333; margin-bottom: 10px; }
        .intro-card p { font-size: 14px; color: #666; line-height: 1.6; }
        .intro-card .formula { background: #f8f9fa; padding: 8px 12px; border-radius: 8px; font-family: monospace; font-size: 13px; color: #e74c3c; margin: 8px 0; }
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
        .adaptive-tag {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 11px;
            margin-left: 5px;
        }
        .adaptive-formula {
            background: #f0f4ff;
            padding: 8px 12px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            color: #667eea;
            margin-top: 5px;
        }
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
        <a href="index.html" class="back">&larr; 返回监控</a>
        <h1>策略说明文档</h1>
    </div>
    <div class="container">
        <div class="intro-card">
            <h2>自适应参数系统</h2>
            <p>所有策略参数根据当前市场波动率自动调节，不再使用固定阈值。</p>
            <div class="formula">波动率比 = 20日波动率 / 60日波动率</div>
            <p>波动率比 &gt; 1 (波动加剧): 收紧阈值，减少信号，避免接飞刀<br>
               波动率比 &lt; 1 (波动收敛): 放松阈值，增加信号，抓住温和反弹</p>
            <p style="margin-top:8px; font-size:13px; color:#999;">回测区间: 2025-01-01起 (ST炼石: 2025-05-07起)<br>
               自适应 vs 固定参数: 累计收益 +427.9% vs +374.1% (+53.9%)</p>
        </div>
"""

    for s in strategies:
        stocks_html = "".join(
            [f'<span class="stocks-tag">{name}</span>' for name in s["stocks"]]
        )
        win_rates_html = " | ".join([f"{k}: {v}" for k, v in s["win_rates"].items()])

        adaptive_html = ""
        if s.get("adaptive") and s["adaptive"] != "无自适应（固定参数）":
            adaptive_html = f'<div class="adaptive-formula">{s["adaptive"]}</div>'
        adaptive_tag = '<span class="adaptive-tag">自适应</span>' if s.get("adaptive") and "无" not in s["adaptive"] else ""

        html += f"""
        <div class="strategy-card">
            <div class="strategy-header" style="background: {s["color"]}">
                <span class="strategy-title">策略{s["id"]}: {s["name"]}{adaptive_tag}</span>
            </div>
            <div class="strategy-body">
                <div class="info-row">
                    <div class="info-label">监控股票</div>
                    <div class="info-value">{stocks_html}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">回测胜率</div>
                    <div class="info-value win-rate">{win_rates_html}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">买入条件</div>
                    <div class="info-value">{s["entry"]}{adaptive_html}</div>
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
            <div>以上策略仅供参考，不构成投资建议</div>
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
