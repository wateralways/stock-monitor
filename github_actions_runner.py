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

    # 解析大盘环境 JSON
    market_env = {}
    env_match = re.search(
        r"<!-- MARKET_ENV_JSON -->(.*?)<!-- /MARKET_ENV_JSON -->",
        output,
        re.DOTALL,
    )
    if env_match:
        try:
            market_env = json.loads(env_match.group(1).strip())
        except Exception as e:
            print(f"[WARN] Failed to parse market env JSON: {e}")

    strategy_configs = [
        {"name": "RSI+布林带均值回归", "short": "S1", "color": "#3498DB",
         "type": "博反弹", "type_class": "mr",
         "cond": "(RSI<自适应阈值 或 BB<阈值) + 当日涨 + 收阳 | 过滤: 主力流向 + MA20斜率"},
        {"name": "MA支撑+KDJ超卖", "short": "S2", "color": "#9B59B6",
         "type": "博反弹", "type_class": "mr",
         "cond": "3天前(MA20支撑或KDJ超卖)触发 + 3日未涨超2% | 过滤: 主力流向 | 延迟3天入场"},
        {"name": "多因子买入策略", "short": "S3", "color": "#E67E22",
         "type": "趋势突破", "type_class": "tb",
         "cond": "VPP量价齐升/动量延续/突破/MACD 多因子共振 | 过滤: 大盘趋势 + MA60"},
        {"name": "RSI+连跌中等信号", "short": "S4", "color": "#27AE60",
         "type": "博反弹", "type_class": "mr",
         "cond": "RSI<=阈值(自适应/固定35) + 连跌>=2天 | 过滤: 主力流向 + 个股趋势"},
        {"name": "动量策略", "short": "S5", "color": "#E74C3C",
         "type": "趋势突破", "type_class": "tb",
         "cond": "量价共振/动量延续/超跌反弹/突破/MACD 任一触发 | 过滤: 趋势+超热"},
        {"name": "多因子评分超卖", "short": "S6", "color": "#8E44AD",
         "type": "博反弹", "type_class": "mr",
         "cond": "RSI+BB+KDJ+连跌+5日跌幅 综合>=50分 | 过滤: 主力流向 + 缩量 + MFI"},
        {"name": "KDJ超卖反弹", "short": "S7", "color": "#16A085",
         "type": "博反弹", "type_class": "mr",
         "cond": "J值<10 + 当日上涨 | 过滤: 主力流向 + 缩量 + MFI + MA60"},
        {"name": "深跌反弹", "short": "S8", "color": "#D35400",
         "type": "博反弹", "type_class": "mr",
         "cond": "A:5日跌>5%+RSI<40+涨 | B:5日跌>10% | 过滤: 主力+MA20+缩量+MFI"},
        {"name": "底部抬高+温和放量", "short": "S9", "color": "#2C7873",
         "type": "趋势突破", "type_class": "tb",
         "cond": "底部抬高+量能放大+RSI确认+收阳 | 过滤: 大盘风险+主力流向"},
        {"name": "N字突破", "short": "S10", "color": "#6A4C93",
         "type": "趋势突破", "type_class": "tb",
         "cond": "A→B涨>=5%+B→C回调25~70%+破B高+收阳+放量"},
        {"name": "缩量涨信号触发", "short": "S11", "color": "#D4A017",
         "type": "趋势突破", "type_class": "tb",
         "cond": "量比<0.8 + 涨>1% + MA20上方 | 川润专用"},
    ]

    data = {
        "time": get_beijing_time().strftime("%Y-%m-%d %H:%M:%S"),
        "strategies": [],
        "summary": {"total_analyzed": 0, "total_buy": 0, "buy_list": []},
        "market_env": market_env,
    }

    for cfg in strategy_configs:
        data["strategies"].append(
            {
                "name": cfg["name"],
                "short": cfg["short"],
                "color": cfg["color"],
                "type": cfg["type"],
                "type_class": cfg["type_class"],
                "cond": cfg["cond"],
                "stocks": [],
                "buy_count": 0,
                "total_count": 0,
            }
        )

    lines = output.split("\n")
    current_strategy_idx = -1
    in_summary = False
    pending_pause_reason = False

    for line in lines:
        strategy_match = re.search(r"\[\D*(\d+)\]", line)
        if strategy_match:
            strategy_num = int(strategy_match.group(1))
            current_strategy_idx = strategy_num - 1
            continue

        # 检测暂停策略
        paused_match = re.search(r"【(.+?)】\[已暂停\]", line)
        if paused_match:
            name = paused_match.group(1)
            for i, s in enumerate(data["strategies"]):
                if s["name"] == name:
                    data["strategies"][i]["paused"] = True
                    pending_pause_reason = True
                    current_strategy_idx = i
                    break
            continue

        if pending_pause_reason and current_strategy_idx >= 0:
            reason_match = re.search(r"原因:\s*(.+)", line)
            if reason_match:
                data["strategies"][current_strategy_idx]["pause_reason"] = reason_match.group(1).strip()
                pending_pause_reason = False
            continue

        if "汇总" in line:
            in_summary = True
            current_strategy_idx = -1
            pending_pause_reason = False
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
        elif strategy.get("paused"):
            reason = strategy.get("pause_reason", "该策略当前已暂停")
            stocks_html = f'<div class="paused-state"><div class="paused-icon">⏸</div><div class="paused-title">已暂停</div><div class="paused-reason">{reason}</div></div>'
        else:
            stocks_html = '<div class="empty-state"><div class="empty-icon">📭</div>暂无数据</div>'

        pulse_class = "pulse" if has_buy else ""
        badge_text = "已暂停" if strategy.get("paused") else f"{strategy['buy_count']}/{strategy['total_count']}"
        header_opacity = "opacity: 0.6;" if strategy.get("paused") else ""

        type_tag = strategy.get("type", "")
        type_class = strategy.get("type_class", "")
        type_html = f'<span class="strategy-type-tag tag-{type_class}">{type_tag}</span>' if type_tag else ""
        cond = strategy.get("cond", "")

        strategies_html += f"""
        <div class="strategy-card">
            <div class="strategy-header" style="background: {strategy["color"]}; {header_opacity}" onclick="toggleStrategy(this)">
                <div class="strategy-title-col">
                    <div class="strategy-title">
                        <span class="strategy-icon {pulse_class}"></span>
                        <span class="strategy-name">{strategy["short"]}: {strategy["name"]}</span>
                        {type_html}
                    </div>
                    <div class="strategy-cond">{cond}</div>
                </div>
                <span class="strategy-badge">{badge_text}</span>
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

    # 读取卖出提醒
    sell_alerts = []
    try:
        import json as _j, os as _os
        pending_file = "docs/pending_sells.json"
        if _os.path.exists(pending_file):
            with open(pending_file, "r", encoding="utf-8") as f:
                sell_alerts = _j.load(f).get("alerts", [])
    except Exception as e:
        print(f"[WARN] Failed to load pending_sells.json: {e}")

    sell_html = ""
    if sell_alerts:
        today_items = [a for a in sell_alerts if a["days_until_sell"] == 0]
        tomorrow_items = [a for a in sell_alerts if a["days_until_sell"] == 1]

        def render_alert(a):
            pnl = a["float_pnl"]
            pnl_class = "up" if pnl > 0 else ("down" if pnl < 0 else "")
            pnl_sign = "+" if pnl > 0 else ""
            return f"""
                <div class="sell-item">
                    <div class="sell-header">
                        <span class="sell-stock">{a["stock"]}</span>
                        <span class="sell-pnl {pnl_class}">{pnl_sign}{pnl:.2f}%</span>
                    </div>
                    <div class="sell-strategy">{a["strategy"]} · {a["sell_timing"]}</div>
                    <div class="sell-detail">买入 {a["buy_date"]} @ ¥{a["buy_price"]:.2f} → 现价 ¥{a["current_price"]:.2f}</div>
                </div>"""

        blocks = []
        if today_items:
            blocks.append(f"""<div class="sell-group">
                <div class="sell-group-title">⚡ 今日卖出 ({len(today_items)})</div>
                {"".join(render_alert(a) for a in today_items)}
            </div>""")
        if tomorrow_items:
            blocks.append(f"""<div class="sell-group">
                <div class="sell-group-title">⏰ 明日卖出 ({len(tomorrow_items)})</div>
                {"".join(render_alert(a) for a in tomorrow_items)}
            </div>""")

        sell_html = f"""
        <div class="sell-section">
            <div class="sell-title">[卖出提醒]</div>
            {"".join(blocks)}
        </div>"""

    buy_value_class = "buy" if summary["total_buy"] > 0 else ""
    signal_ratio = summary["total_buy"] / max(summary["total_analyzed"], 1) * 100

    # 构建市场环境 HTML
    market_env = data.get("market_env", {})
    market_env_html = ""
    if market_env:
        trend = market_env.get("trend", "unknown")
        trend_cn = market_env.get("trend_cn", "未知")
        risk = market_env.get("risk", "medium")
        desc = market_env.get("market_desc", "")
        mr_score = market_env.get("mean_reversion_score", 50)
        tb_score = market_env.get("trend_breakout_score", 50)
        pos_pct = market_env.get("position_pct", 50)
        phase_cn = market_env.get("phase_cn", "")
        prob = market_env.get("consolidation_end_prob", 0)

        tag_class = trend if trend in ("up", "down", "sideways", "sideways_strong", "sideways_weak") else "sideways"
        risk_color = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}.get(risk, "#f39c12")
        risk_cn = {"low": "低风险", "medium": "中风险", "high": "高风险"}.get(risk, "中风险")

        def _bar(score, label):
            fill_class = "high" if score >= 70 else ("medium" if score >= 40 else "low")
            return f'<div class="match-row"><div class="match-label"><span>{label}</span><span>{score}%</span></div><div class="match-bar-bg"><div class="match-bar-fill {fill_class}" style="width:{score}%"></div></div></div>'

        phase_html = ""
        if phase_cn:
            prob_part = ""
            if prob > 0:
                alert_class = "phase-alert" if prob >= 50 else ""
                prob_part = f'<div class="{alert_class}">震荡结束概率: {prob}%</div>'
            phase_html = f'<div class="phase-info">&#128202; {phase_cn}{prob_part}</div>'

        extra_info = ""
        ma20 = market_env.get("sh_ma20_slope", 0)
        today_pct = market_env.get("sh_today_pct", 0)
        if ma20 or today_pct:
            extra_parts = []
            if ma20:
                extra_parts.append(f"MA20斜率: {ma20:+.2f}%")
            if today_pct:
                extra_parts.append(f"今日: {today_pct:+.2f}%")
            extra_info = f'<div style="font-size:12px;color:#999;margin-top:8px;">{" | ".join(extra_parts)}</div>'

        market_env_html = f'<div class="market-env-card"><div class="market-env-header"><span class="market-env-title">&#127760; 市场环境</span><div class="market-env-tags"><span class="env-tag {tag_class}">{trend_cn}</span><span class="env-tag" style="background:{risk_color}">{risk_cn}</span></div></div><div class="env-desc">{desc}</div>{_bar(mr_score, "均值回归策略匹配度")}{_bar(tb_score, "趋势突破策略匹配度")}{_bar(pos_pct, "建议总仓位")}{phase_html}{extra_info}</div>'

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
        .filter-rules {{
            background: white;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }}
        .filter-title {{
            font-size: 14px;
            font-weight: 700;
            color: #333;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .filter-grid {{ display: flex; flex-direction: column; gap: 8px; }}
        .filter-item {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 12px;
            color: #666;
            line-height: 1.5;
        }}
        .filter-icon {{ flex-shrink: 0; font-size: 14px; }}
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
        .strategy-title-col {{ display: flex; flex-direction: column; gap: 4px; flex: 1; }}
        .strategy-name {{ font-size: 16px; font-weight: 600; }}
        .strategy-cond {{
            font-size: 11px;
            opacity: 0.85;
            line-height: 1.4;
        }}
        .strategy-type-tag {{
            display: inline-block;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
            white-space: nowrap;
        }}
        .tag-mr {{ background: rgba(255,255,255,0.35); color: #fff; }}
        .tag-tb {{ background: rgba(255,255,255,0.35); color: #fff; }}
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
        .sell-section {{
            background: linear-gradient(135deg, #f39c12, #d35400);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 10px 40px rgba(243, 156, 18, 0.3);
        }}
        .sell-title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 18px;
            text-align: center;
        }}
        .sell-group {{ margin-bottom: 18px; }}
        .sell-group:last-child {{ margin-bottom: 0; }}
        .sell-group-title {{
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 10px;
            opacity: 0.95;
        }}
        .sell-item {{
            background: rgba(255,255,255,0.22);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }}
        .sell-item:last-child {{ margin-bottom: 0; }}
        .sell-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .sell-stock {{ font-size: 22px; font-weight: 700; }}
        .sell-pnl {{
            font-size: 18px;
            font-weight: 700;
            background: rgba(255,255,255,0.3);
            padding: 4px 12px;
            border-radius: 14px;
        }}
        .sell-pnl.up {{ background: rgba(231,76,60,0.7); }}
        .sell-pnl.down {{ background: rgba(39,174,96,0.7); }}
        .sell-strategy {{ font-size: 15px; opacity: 0.95; margin-bottom: 4px; }}
        .sell-detail {{ font-size: 14px; opacity: 0.88; }}
        .no-signal {{ text-align: center; padding: 40px; font-size: 22px; opacity: 0.9; font-weight: 600; }}
        .empty-state {{ text-align: center; padding: 40px 20px; color: #999; }}
        .footer {{ text-align: center; padding: 30px 20px; color: rgba(255,255,255,0.7); font-size: 13px; }}
        @keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
        .pulse {{ animation: pulse 2s infinite; }}
        .market-env-card {{ background: white; border-radius: 20px; padding: 25px; margin-bottom: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }}
        .market-env-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }}
        .market-env-title {{ font-size: 18px; font-weight: 700; color: #333; }}
        .market-env-tags {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .env-tag {{ padding: 4px 12px; border-radius: 15px; font-size: 13px; font-weight: 600; color: white; }}
        .env-tag.up {{ background: linear-gradient(135deg, #27ae60, #2ecc71); }}
        .env-tag.down {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
        .env-tag.sideways {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
        .env-tag.sideways_strong {{ background: linear-gradient(135deg, #3498db, #2980b9); }}
        .env-tag.sideways_weak {{ background: linear-gradient(135deg, #95a5a6, #7f8c8d); }}
        .env-desc {{ font-size: 14px; color: #666; margin-bottom: 15px; line-height: 1.5; }}
        .match-row {{ margin-bottom: 12px; }}
        .match-label {{ font-size: 13px; color: #888; margin-bottom: 5px; display: flex; justify-content: space-between; }}
        .match-bar-bg {{ background: #f0f0f0; border-radius: 10px; height: 12px; overflow: hidden; }}
        .match-bar-fill {{ height: 100%; border-radius: 10px; transition: width 0.5s ease; }}
        .match-bar-fill.high {{ background: linear-gradient(90deg, #27ae60, #2ecc71); }}
        .match-bar-fill.medium {{ background: linear-gradient(90deg, #f39c12, #e67e22); }}
        .match-bar-fill.low {{ background: linear-gradient(90deg, #e74c3c, #c0392b); }}
        .phase-info {{ margin-top: 12px; padding: 12px 15px; background: #f8f9fa; border-radius: 12px; font-size: 13px; color: #666; line-height: 1.6; }}
        .phase-alert {{ color: #e74c3c; font-weight: 600; }}
        .paused-state {{ text-align: center; padding: 30px 20px; color: #999; }}
        .paused-icon {{ font-size: 32px; margin-bottom: 10px; }}
        .paused-title {{ font-size: 16px; font-weight: 600; color: #e74c3c; margin-bottom: 8px; }}
        .paused-reason {{ font-size: 13px; color: #888; line-height: 1.5; max-width: 90%; margin: 0 auto; }}
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

        <div class="filter-rules">
            <div class="filter-title">过滤规则 (2026-06更新)</div>
            <div class="filter-grid">
                <div class="filter-item">
                    <span class="filter-icon">💰</span>
                    <span class="filter-text"><b>主力资金流向(方案U v2)</b>: 当日主力净流入>0 或 近3日累计>0 或 流出衰减 — S1/S2/S4/S6/S7/S8</span>
                </div>
                <div class="filter-item">
                    <span class="filter-icon">📉</span>
                    <span class="filter-text"><b>MA20下降趋势</b>: MA20斜率<-1%禁止买入，避免下降趋势接飞刀 — S1/S8</span>
                </div>
                <div class="filter-item">
                    <span class="filter-icon">📊</span>
                    <span class="filter-text"><b>缩量下跌确认</b>: 下跌中量比>1.1(放量抛售)禁止买入 — S6/S7/S8</span>
                </div>
                <div class="filter-item">
                    <span class="filter-icon">🔍</span>
                    <span class="filter-text"><b>MFI+收盘位置</b>: 收盘在当日下半区且MFI未回升，禁止买入 — S6/S7/S8</span>
                </div>
            </div>
        </div>

        {market_env_html}

        {rec_html}
        {sell_html}
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
            "stocks": ["爱乐达", "ST炼石", "华夏航空"],
            "win_rates": {"爱乐达": "83.3%", "ST炼石": "69.6%", "华夏航空": "83.3%"},
            "entry": "RSI或布林带位置低于阈值，且当日上涨收阳。爱乐达/华夏航空用自适应阈值+MA20趋势过滤；ST炼石用固定阈值(RSI<33/BB<0.5)无趋势过滤",
            "exit": "RSI > 自适应卖出阈值(约50-70) 或 布林带位置 > 0.75",
            "timing": "华夏航空T+0/T+5尾盘 | 爱乐达T+0/T+5尾盘 | ST炼石T+1/T+6",
            "filters": ["主力资金流向(方案U)", "MA20斜率<-1%禁止", "个股MA20下降趋势(非ST)"],
            "description": "利用RSI超卖和布林带下轨支撑，捕捉短期反弹机会。华夏航空83.3%胜率(18笔,平均+2.97%,T+0/T+5)；爱乐达83.3%+MA20趋势过滤；ST炼石用固定参数。",
        },
        {
            "id": 2,
            "name": "MA支撑+KDJ超卖",
            "color": "#9B59B6",
            "stocks": ["ST炼石"],
            "win_rates": {"ST炼石": "66.0%(延迟3天)"},
            "entry": "MA20支撑或KDJ超卖信号触发后，延迟3天观察: 如果未涨超2%则买入，已大涨则放弃",
            "exit": "持仓5个交易日后开盘卖出",
            "timing": "信号日+3天后确认未涨→T+1开盘买入，T+6开盘卖出",
            "filters": ["主力资金流向(方案U)", "大盘风险暂停"],

            "description": "ST炼石专用延迟入场策略。KDJ超卖触发后不急着买，等3天让二次探底完成再入场。延迟后胜率61%→66%，均盈+1.29%→+1.86%，97笔样本验证。涨超2%的信号放弃(这些信号虽然也好，但等待可获得更好的入场价)。",
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
                "佳力图",
                "东方电子",
            ],
            "win_rates": {
                "裕同科技": "78.6%",
                "佳力图": "78.6%",
                "东方电子": "76.9%",
                "扬农化工": "71.4%",
                "华测导航": "68.8%",
                "川润股份": "62.5%",
                "ST炼石": "100.0%",
                "高澜股份": "56.2%",
                "拓日新能": "85.7%",
            },
            "entry": "RSI <= 阈值(扬农化工/拓日新能/佳力图固定35，其他自适应约25-40) 且连续下跌 >= 2天",
            "exit": "RSI > 50 或连续上涨2天",
            "timing": "T+0尾盘买入(高澜/裕同/扬农/华测/川润/东方电子)；佳力图T+1/T+5；其余T+1/T+6",
            "filters": ["主力资金流向(方案U)", "个股趋势(price<MA60*0.97)", "大盘风险暂停"],

            "description": "捕捉连续下跌后的超跌反弹机会。佳力图78.6%(14笔,固定RSI<=35,T+1/T+5,平均+2.65%)；东方电子76.9%(13笔,T+0/T+5,平均+2.88%)。扬农化工/拓日新能/佳力图使用固定RSI<=35阈值。",
        },
        {
            "id": 5,
            "name": "动量策略",
            "color": "#E74C3C",
            "stocks": ["川润股份", "爱乐达"],
            "win_rates": {"川润股份": "63.0%", "爱乐达": "64.7%"},
            "entry": "5个子信号(量价配合/动量加速/超跌反弹/突破信号/MACD金叉)满足2个以上，或单独动量加速(非下跌趋势)。川润禁用突破和MACD；爱乐达禁用量价配合和突破",
            "exit": "川润持仓4天，爱乐达持仓5天",
            "timing": "川润T+0/T+4，爱乐达T+1/T+6",
            "description": "综合动量指标，捕捉上涨趋势中的加速信号。川润回测发现持仓4天比5天胜率更高(63% vs 59.3%)。",
        },
        {
            "id": 6,
            "name": "多因子评分超卖",
            "color": "#8E44AD",
            "stocks": ["拓日新能", "安车检测"],
            "win_rates": {"拓日新能": "86.4%", "安车检测": "85.2%"},
            "entry": "9个因子加权评分>=50分: RSI14(0-25), 布林带位置(0-20), KDJ J值(0-15), 连跌天数(0-15), 5日跌幅(0-10), 大盘涨(+5), 长下影线(+5), 相对弱势(+5), 缩量罚分(-8)",
            "exit": "持仓5-6个交易日后卖出",
            "timing": "拓日新能T+0/T+5尾盘 | 安车检测T+1/T+6开盘",
            "filters": ["主力资金流向(方案U)", "缩量下跌确认(量比>1.1禁止)", "MFI+收盘位置", "MA60+大盘下行禁止"],

            "description": "评分制超卖反弹策略。拓日新能86.4%胜率(22笔,T+0/T+5,平均+3%)；安车检测85.2%胜率(27笔,T+1/T+6,平均+5.17%,总收益+139.7%)。综合9个技术因子加权打分>=50触发买入。",
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
            "filters": ["主力资金流向(方案U)", "缩量下跌确认", "MFI+收盘位置", "MA60下方+大盘下行禁止"],

            "description": "专为英维克设计的KDJ超卖反弹策略。当J值低于10(深度超卖)且当日收涨(确认反弹启动)时买入。回测13笔交易，84.6%胜率，平均+2.6%。替代原动量策略(53.1%)，胜率提升31个百分点。",
        },
        {
            "id": 8,
            "name": "深跌反弹",
            "color": "#D35400",
            "stocks": ["高澜股份", "川润股份", "江淮汽车", "爱乐达", "安车检测", "晶科能源"],
            "win_rates": {"高澜(跌5%+RSI+涨)": "100%", "高澜(跌10%)": "81.8%", "川润(跌5%+RSI+涨)": "70.0%", "江淮(跌10%)": "76.9%", "爱乐达": "73.7%", "安车检测(跌5%+RSI+涨)": "75.0%", "晶科(跌5%+RSI+涨)": "85.7%", "晶科(跌10%)": "78.6%"},
            "entry": "信号A: 5日跌>5% & RSI14<40 & 当日涨 | 信号B: 5日跌>10%(高澜/江淮/爱乐达/安车检测/晶科能源)",
            "exit": "信号A: 3-4天后卖出 | 信号B: 4-6天后卖出",
            "timing": "晶科T+0/T+4 | 爱乐达信号A T+0/T+3 | 安车检测T+1/T+6 | 其他信号A T+0/T+4 | 其他信号B T+0/T+5(江淮T+1/T+6)",
            "filters": ["主力资金流向(方案U)", "MA20斜率<-1%禁止", "缩量下跌确认", "MFI+收盘位置", "个股下降趋势(price<MA60*0.95)", "爱乐达高位急跌保护"],
            "description": "深跌后的超跌反弹策略。晶科能源信号A 85.7%(7笔,T+0/T+4)、信号B 78.6%(14笔,T+0/T+4,平均+3.80%)；安车检测信号A 75.0%(8笔,T+1/T+6,平均+8.01%)。信号A适用全部6只股票，信号B仅限高澜/江淮/爱乐达/安车检测/晶科能源。2026-06新增MA20斜率过滤(-155%→-39%减亏效果)。",
        },
        {
            "id": 9,
            "name": "底部抬高+温和放量",
            "color": "#2C7873",
            "stocks": ["川润股份", "裕同科技", "拓日新能", "ST炼石", "华夏航空", "英维克", "佳力图"],
            "win_rates": {"川润股份": "75.0%(T+9)", "裕同科技": "100%(T+9)", "拓日新能": "70.6%(T+4)", "ST炼石": "83.3%(T+4)", "华夏航空": "85.7%(T+6)", "英维克": "60.0%(T+8)", "佳力图": "70.0%(T+9)"},
            "entry": "近5日低点 > 近15日低点(底部抬高) 且 近3日量均>近10日量均*1.2(温和放量) 且 RSI14在45-65(中性偏多) 且 今日收阳",
            "exit": "按差异化持仓期到期卖出",
            "timing": "川润/裕同/佳力图 T+1/T+9 | 英维克 T+1/T+8 | 华夏航空 T+1/T+6 | 拓日/ST炼石 T+1/T+4",
            "description": "捕捉横盘震荡后加速上涨的机会。区别于深跌反弹(抓超跌)，本策略识别'反弹后→横盘酝酿→温和放量突破'的形态。7只股票差异化持仓期: 川润+7.28%(8笔75%), 裕同+7.82%(7笔100%), 拓日+5.08%(17笔76.5%,T+4), ST炼石+4.22%(6笔83.3%,T+4), 华夏航空+3.97%(7笔85.7%), 英维克+2.52%(5笔80%), 佳力图+4.04%(10笔70%)。拓日/ST炼石偏'快进快出', 其他偏'趋势慢热'。",
        },
        {
            "id": 10,
            "name": "N字突破",
            "color": "#6A4C93",
            "stocks": ["华测导航", "高澜股份"],
            "win_rates": {"华测导航": "76.9%(T+4)", "高澜股份": "68.4%(T+7)"},
            "entry": "近15-3天内有一波上涨≥5%(A→B) + 回调25-70%(B→C) + 今日收盘突破B点高点 + 收阳 + 量能≥5日均量",
            "exit": "按差异化持仓期到期卖出",
            "timing": "华测导航 T+1/T+4 | 高澜股份 T+1/T+7",
            "description": "经典N字形态的趋势延续信号。当股价完成'上涨→回调→突破前高'的N字结构时买入。华测导航69.2%(13笔,T+1/T+4,均+3.92%)，高澜股份52.6%(19笔,T+1/T+7,均+7.90%,盈亏比极佳)。",
        },
        {
            "id": 11,
            "name": "缩量涨信号触发",
            "color": "#D4A017",
            "stocks": ["川润股份"],
            "win_rates": {"川润股份": "50.0%(T+7)"},
            "entry": "成交量/5日均量<0.8(缩量) 且 当日涨幅>1% 且 收盘价>MA20",
            "exit": "持仓7个交易日 或 止损-10%",
            "timing": "T+1开盘买入(涨停不买)，T+7开盘卖出 或 盘中触及-10%次日开盘止损",
            "description": "川润股份专用策略。利用缩量上涨识别主力控盘信号——成交量萎缩(量比<0.8)但价格上涨(>1%)说明抛压已尽、筹码锁定。MA20过滤确保在上升趋势中操作。胜率50%但盈亏比4.27:1(赢均+21%/亏均-5%)，靠盈亏比而非胜率盈利。2025年以来14笔累计+115%。",
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
        .container { max-width: 680px; margin: 0 auto; padding: 15px; }
        .filter-summary {
            background: white;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .filter-summary h3 { font-size: 16px; color: #333; margin-bottom: 12px; }
        .filter-summary .fs-item {
            font-size: 13px; color: #666; padding: 4px 0; line-height: 1.5;
        }
        .fs-item b { color: #333; }
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
        .strategy-type-tag {
            display: inline-block;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
            background: rgba(255,255,255,0.35);
            color: #fff;
            margin-left: 8px;
        }
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
        .filter-tag {
            display: inline-block;
            background: #fff3e0;
            color: #e67e22;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin-right: 4px;
            margin-bottom: 4px;
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
        <a href="index.html" class="back">← 返回监控</a>
        <h1>📈 策略说明文档</h1>
    </div>
    <div class="container">
        <div style="text-align:center;margin-bottom:15px;">
            <a href="trades.html" style="display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:10px 24px;border-radius:25px;text-decoration:none;font-size:15px;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,0.4);">📋 查看历史交易明细</a>
        </div>
        <div class="filter-summary">
            <h3>过滤体系 (2026-06更新)</h3>
            <div class="fs-item">💰 <b>主力资金流向(方案U v2)</b>: 当日主力净流入>0 或 近3日累计>0 或 流出趋势减弱 → S1/S2/S4/S6/S7/S8</div>
            <div class="fs-item">📉 <b>MA20斜率过滤</b>: MA20斜率<-1%时禁止买入，避免下降趋势中接飞刀 → S1/S8</div>
            <div class="fs-item">📊 <b>缩量下跌确认</b>: 下跌过程中量比>1.1(恐慌抛售)禁止买入 → S6/S7/S8</div>
            <div class="fs-item">🔍 <b>MFI+收盘位置</b>: 收盘在当日下半区且MFI未回升，禁止买入 → S6/S7/S8</div>
            <div class="fs-item" style="margin-top:8px;font-size:12px;color:#999;">回测验证(2026年5-6月): 过滤后累计减亏 S4(-59%→-15%) S8(-155%→-39%) S6(-18%→-4%)</div>
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
                </div>"""
        if s.get("filters"):
            filter_tags = "".join([f'<span class="filter-tag">🛡 {f}</span>' for f in s["filters"]])
            html += f"""
                <div class="info-row">
                    <div class="info-label">过滤条件</div>
                    <div class="info-value">{filter_tags}</div>
                </div>"""
        html += f"""
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
                        {"name": "底部抬高+温和放量", "short": "策略9", "color": "#2C7873"},
                        {"name": "N字突破", "short": "策略10", "color": "#6A4C93"},
                        {"name": "缩量涨信号触发", "short": "策略11", "color": "#D4A017"},
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
