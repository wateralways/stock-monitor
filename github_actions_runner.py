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
from datetime import datetime

TUSHARE_TOKEN = os.environ.get(
    "TUSHARE_TOKEN", "701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d"
)

os.makedirs("docs", exist_ok=True)


def run_monitor():
    result = subprocess.run(
        [sys.executable, "stock_monitor_unified_realtime.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "TUSHARE_TOKEN": TUSHARE_TOKEN},
    )
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
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            <div>建议持仓周期: T+5 (5个交易日)</div>
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

    return "docs/index.html"


def main():
    print("=" * 60)
    print("Stock Monitor - GitHub Actions Runner")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    try:
        output = run_monitor()
        print(output)

        if output:
            data = parse_output(output)
            if data:
                html_file = generate_html(data)
                print(f"\n[OK] Report generated: {html_file}")
                print(f"Total stocks: {data['summary']['total_analyzed']}")
                print(f"Buy signals: {data['summary']['total_buy']}")
            else:
                print("[ERROR] Failed to parse output")
                sys.exit(1)
        else:
            print("[ERROR] No output from monitor")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
