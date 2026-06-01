#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用新浪API批量获取个股资金流向(主力=大单+超大单)并缓存"""
import json, time, requests, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CODES = {
    "300696.SZ": "sz300696", "000697.SZ": "sz000697", "002928.SZ": "sz002928",
    "002831.SZ": "sz002831", "600486.SH": "sh600486", "300627.SZ": "sz300627",
    "002272.SZ": "sz002272", "300499.SZ": "sz300499", "002218.SZ": "sz002218",
    "603912.SH": "sh603912", "000682.SZ": "sz000682", "300572.SZ": "sz300572",
    "002837.SZ": "sz002837", "600418.SH": "sh600418", "688223.SH": "sh688223",
}
NAMES = {
    "300696.SZ":"爱乐达","000697.SZ":"ST炼石","002928.SZ":"华夏航空",
    "002831.SZ":"裕同科技","600486.SH":"扬农化工","300627.SZ":"华测导航",
    "002272.SZ":"川润股份","300499.SZ":"高澜股份","002218.SZ":"拓日新能",
    "603912.SH":"佳力图","000682.SZ":"东方电子","300572.SZ":"安车检测",
    "002837.SZ":"英维克","600418.SH":"江淮汽车","688223.SH":"晶科能源",
}

def fetch_fund_flow_sina(sina_code: str, pages: int = 5) -> list:
    """从新浪获取个股资金流向 (主力净流入 = r0_net)
    每页最多100条, 获取pages*100条 ≈ 最近1-2年数据
    """
    headers = {"Referer": "https://finance.sina.com.cn/"}
    all_records = []

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
                    all_records.append({
                        "date": item["opendate"],
                        "big_net_amount": float(item.get("r0_net", 0)),
                        "total_net_amount": float(item.get("netamount", 0)),
                        "price": float(item.get("trade", 0)),
                    })
                if len(data) < 100:
                    break  # 最后一页
            else:
                break
        except Exception as e:
            print(f"      page {page} error: {e}")
            break

        time.sleep(0.1)  # 温和间隔

    return sorted(all_records, key=lambda x: x["date"])  # 升序

def main():
    cache_file = "moneyflow_cache.json"
    cache = {}
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
    except:
        pass

    need = [(c, s) for c, s in CODES.items() if c not in cache or len(cache.get(c, [])) < 10]
    print(f"已缓存: {len(cache)}只, 需要获取: {len(need)}只\n")

    for i, (ts_code, sina_code) in enumerate(need):
        name = NAMES.get(ts_code, ts_code)
        print(f"  [{i+1}/{len(need)}] {name}({ts_code})...", end=" ", flush=True)
        records = fetch_fund_flow_sina(sina_code, pages=4)  # 最多400条
        if records:
            cache[ts_code] = records
            with open(cache_file, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            pos_days = sum(1 for r in records if r["big_net_amount"] > 0)
            date_range = f"{records[0]['date']}~{records[-1]['date']}"
            print(f"{len(records)}条 (净流入{pos_days}天, {date_range}) 已保存")
        else:
            print("无数据")
        time.sleep(0.3)

    print(f"\n缓存完成! {cache_file} 共 {len(cache)} 只股票")

if __name__ == "__main__":
    main()
