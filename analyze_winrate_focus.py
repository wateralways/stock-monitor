#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚焦胜率优化：分析2026年4月、5月各过滤条件的胜率表现"""
import json, os, sys, io, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

TOKEN = os.environ.get("TUSHARE_TOKEN", "")
import tushare as ts
ts.set_token(TOKEN); pro = ts.pro_api()

with open("moneyflow_cache.json") as f: mf_cache = json.load(f)

ALL_STOCKS = {
    "300696.SZ":"爱乐达","000697.SZ":"ST炼石","002928.SZ":"华夏航空",
    "002831.SZ":"裕同科技","600486.SH":"扬农化工","300627.SZ":"华测导航",
    "002272.SZ":"川润股份","300499.SZ":"高澜股份","002218.SZ":"拓日新能",
    "603912.SH":"佳力图","000682.SZ":"东方电子","300572.SZ":"安车检测",
    "002837.SZ":"英维克","600418.SH":"江淮汽车","688223.SH":"晶科能源",
}
S_STOCKS = {
    "策略1":["300696.SZ","000697.SZ","002928.SZ"],"策略2":["000697.SZ"],
    "策略4":["002831.SZ","600486.SH","300627.SZ","002272.SZ","300499.SZ","002218.SZ","603912.SH","000682.SZ","000697.SZ"],
    "策略6":["002218.SZ","300572.SZ"],"策略7":["002837.SZ"],
    "策略8":["300499.SZ","002272.SZ","600418.SH","300696.SZ","300572.SZ","688223.SH"],
}

# 获取日线
daily_data = {}
for c in ALL_STOCKS:
    try:
        df = pro.daily(ts_code=c,start_date="20251201",end_date="20260601")
        if df is not None and not df.empty:
            df=df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"]=pd.to_datetime(df["trade_date"])
            df["pct_chg"]=df["close"].pct_change()*100
            daily_data[c]=df
    except: pass

def calc_rsi(p,w=14):
    d=p.diff();g=d.clip(lower=0).rolling(w).mean();l=(-d.clip(upper=0)).rolling(w).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))
def calc_bb(p,w=20):
    m=p.rolling(w).mean();s=p.rolling(w).std()
    return m+2*s,m,m-2*s,(p-(m-2*s))/(4*s)
def calc_kdj(df,n=9):
    lm=df["low"].rolling(n).min();hm=df["high"].rolling(n).max()
    r=(df["close"]-lm)/(hm-lm)*100;k=r.ewm(com=2,adjust=False).mean()
    d=k.ewm(com=2,adjust=False).mean();return k,d,3*k-2*d
def calc_cons(df,d="down"):
    r=pd.Series(0,index=df.index)
    for i in range(1,len(df)):
        if d=="down" and df["pct_chg"].iloc[i]<0: r.iloc[i]=r.iloc[i-1]+1
        elif d=="up" and df["pct_chg"].iloc[i]>0: r.iloc[i]=r.iloc[i-1]+1
        else: r.iloc[i]=0
    return r
def adp_rsi_c(df_s):
    v20=df_s["pct_chg"].rolling(20).std().iloc[-1];v60=df_s["pct_chg"].rolling(60).std().iloc[-1]
    v20=v20 if pd.notna(v20) else 3;v60=v60 if pd.notna(v60) else 3
    return float(np.clip(35-(v20/v60-1)*5 if v60>0 else 35,22,40))

# 收集信号
print("检测信号...",flush=True)
signals = []
dfs = {}
for code,name in ALL_STOCKS.items():
    if code not in daily_data: continue
    df = daily_data[code].copy()
    # add moneyflow
    mf_map={}
    for r in mf_cache.get(code,[]):
        try: mf_map[pd.Timestamp(r["date"])]=r["big_net_amount"]
        except: pass
    df["bn"]=df["trade_date"].map(mf_map)
    df["bn1"]=df["bn"].shift(1); df["bn2"]=df["bn"].shift(2)
    df["b3s"]=df["bn"].rolling(3).sum(); df["b5s"]=df["bn"].rolling(5).sum()
    df["p3"]=(df["bn"]>0).rolling(3).sum(); df["p5"]=(df["bn"]>0).rolling(5).sum()
    df["n3"]=(df["bn"]<0).rolling(3).sum()
    df["os"]=((df["bn"]<0)&(df["bn"]>df["bn1"])).astype(int)
    df["ia"]=((df["bn"]>0)&(df["bn"]>df["bn1"])).astype(int)
    df["ni"]=(df["bn"]>df["bn1"]).astype(int)
    dfs[code]=df

    mask=(df["trade_date"]>=pd.Timestamp("2026-03-01"))&(df["trade_date"]<=pd.Timestamp("2026-06-01"))
    for idx in df[mask].index:
        if idx<30: continue
        s=df.iloc[:idx+1]; l=s.iloc[-1]; p=s["close"]
        r14=calc_rsi(p,14).iloc[-1]; _,_,blo,_=calc_bb(p,20)
        bb_p=(l["close"]-blo.iloc[-1])/(2*p.rolling(20).std().iloc[-1]) if p.rolling(20).std().iloc[-1]>0 else 0.5
        k,d,j=calc_kdj(s); cd=calc_cons(s).iloc[-1]
        ret5=(l["close"]/s["close"].iloc[-6]-1)*100 if len(s)>=6 else 0
        ret5p=(s["close"].iloc[-2]/s["close"].iloc[-7]-1)*100 if len(s)>=7 else 0

        # S1
        if code in S_STOCKS["策略1"]:
            st="ST" in name
            rt=33 if st else float(np.clip(33-(s["pct_chg"].rolling(20).std().iloc[-1]/max(s["pct_chg"].rolling(60).std().iloc[-1],0.01)-1)*6,18,45))
            bt=0.5 if st else float(np.clip(0.5-(s["pct_chg"].rolling(20).std().iloc[-1]/max(s["pct_chg"].rolling(60).std().iloc[-1],0.01)-1)*0.2,0.05,0.6))
            if ((pd.notna(r14) and r14<rt) or (pd.notna(bb_p) and bb_p<bt)) and l["pct_chg"]>0 and l["close"]>l["open"]:
                signals.append((l["trade_date"],name,"S1",idx,code))
        # S2
        if code in S_STOCKS["策略2"] and idx>=8:
            sd=df.iloc[idx-3]; sj=j.iloc[-4]; sm=p.rolling(20).mean().iloc[-4]
            rise=(l["close"]-sd["close"])/sd["close"]*100
            if ((pd.notna(sm) and abs(sd["close"]-sm)/sm<0.02 and sd["pct_chg"]>0) or (pd.notna(sj) and sj<30)) and rise<2:
                signals.append((l["trade_date"],name,"S2",idx,code))
        # S4
        if code in S_STOCKS["策略4"]:
            rt2=35 if name in ("扬农化工","拓日新能","佳力图") else adp_rsi_c(s)
            if pd.notna(r14) and r14<=rt2 and cd>=2:
                signals.append((l["trade_date"],name,"S4",idx,code))
        # S6
        if code in S_STOCKS["策略6"]:
            sc=0
            if pd.notna(r14):
                if r14<25:sc+=25
            elif r14<30:sc+=20
            elif r14<35:sc+=15
            elif r14<40:sc+=10
            elif r14<45:sc+=5
            if bb_p<0.1:sc+=20
            elif bb_p<0.2:sc+=16
            elif bb_p<0.3:sc+=12
            elif bb_p<0.4:sc+=8
            elif bb_p<0.5:sc+=4
            if pd.notna(j.iloc[-1]):
                if j.iloc[-1]<0:sc+=15
                elif j.iloc[-1]<10:sc+=12
                elif j.iloc[-1]<20:sc+=8
                elif j.iloc[-1]<30:sc+=4
            if cd>=4:sc+=15
            elif cd>=3:sc+=12
            elif cd>=2:sc+=8
            if ret5<-10:sc+=10
            elif ret5<-7:sc+=8
            elif ret5<-5:sc+=6
            elif ret5<-3:sc+=3
            if sc>=50: signals.append((l["trade_date"],name,"S6",idx,code))
        # S7
        if code in S_STOCKS["策略7"]:
            if pd.notna(j.iloc[-1]) and j.iloc[-1]<10 and l["pct_chg"]>0:
                signals.append((l["trade_date"],name,"S7",idx,code))
        # S8
        if code in S_STOCKS["策略8"]:
            sa=ret5<-5 and pd.notna(r14) and r14<40 and l["pct_chg"]>0
            sb=ret5<-10 and name in ("高澜股份","江淮汽车","爱乐达","安车检测","晶科能源")
            if sa or sb: signals.append((l["trade_date"],name,"S8",idx,code))

print(f"总信号: {len(signals)}")

# 模拟交易
def sim_trade(df,sig_idx,sname,strat):
    s_short=strat
    t0=0; tsell=5
    # mappings
    if ("ST炼石"==sname and "S2"==strat): t0,tsell=1,6
    elif sname=="川润股份" and strat=="S8": t0,tsell=0,4
    elif sname=="晶科能源" and strat=="S8": t0,tsell=0,4
    elif sname=="安车检测" and strat=="S6": t0,tsell=1,4
    elif sname=="爱乐达" and strat in ("S1","S8"): t0,tsell=0,5
    elif sname=="华夏航空" and strat=="S1": t0,tsell=0,5
    elif sname in ("高澜股份","裕同科技","扬农化工","华测导航","川润股份","拓日新能","英维克","东方电子","晶科能源") and strat=="S4": t0,tsell=0,5
    bi=sig_idx+t0; si=sig_idx+tsell
    if bi>=len(df) or si>=len(df): return None
    bp=df.iloc[sig_idx]["close"] if t0==0 else df.iloc[bi]["open"]
    sp=df.iloc[si]["close"]
    pnl=(sp-bp)/bp*100
    return {"pnl":round(pnl,2),"win":pnl>0}

# 评估
FILTERS = {
    "无过滤": lambda r: True,
    "L:3日累计>0或今日净流入": lambda r: r["b3s"]>0 or r["bn"]>0,
    "M:今日净流入 且 非3日连续净流出": lambda r: r["bn"]>0 and r["n3"]<3,
    "N:3日累计>0 且 今日非极端流出(>-3000万)": lambda r: r["b3s"]>0 and r["bn"]>-3000,
    "O:今日净流入 或 (3日累计>0 且 流出在收窄)": lambda r: r["bn"]>0 or (r["b3s"]>0 and r["os"]==1),
    "P:3日净流入≥1天 且 今日非巨量流出(>-1亿)": lambda r: r["p3"]>=1 and r["bn"]>-10000,
    "Q:今日净流入 或 5日累计>0": lambda r: r["bn"]>0 or r["b5s"]>0,
    "R:近5天≥3天净流入": lambda r: r["p5"]>=3,
    "S:今日净流入加速(inflow accelerating)": lambda r: r["ia"]==1,
    "T:非连续3日净流出 且 3日累计> -5000万": lambda r: r["n3"]<3 and r["b3s"]>-5000,
    "U:前日净流入 或 今日净流入": lambda r: r["bn"]>0 or r["bn1"]>0,
    "V:3日净流入天数≥2": lambda r: r["p3"]>=2,
}

print("\n===== 全时段 (3月~6月) 胜率优先排序 =====")
print(f"{'条件':<42} {'总数':>4} {'胜率':>7} {'累计':>9} {'4月WR':>7} {'5月WR':>7} {'4月PnL':>8} {'5月PnL':>8}")
print("-"*95)

all_results = []
for fname,fcond in FILTERS.items():
    trades=[]; month_trades=defaultdict(list)
    for sd,sname,strat,idx,code in signals:
        df=dfs[code]; row=df.iloc[idx]
        if pd.isna(row.get("bn")): continue
        if not fcond(row): continue
        tr=sim_trade(df,idx,sname,strat)
        if tr:
            m=sd.strftime("%Y-%m")
            month_trades[m].append(tr)
            trades.append(tr)
    if not trades:
        all_results.append((fname,0,0,0,0,0,0,0))
        continue
    wins=[t for t in trades if t["win"]]
    wr=len(wins)/len(trades)*100
    cp=sum(t["pnl"] for t in trades)
    # by month
    def m_stats(ts):
        if not ts: return 0,0
        w=sum(1 for t in ts if t["win"])
        return w/len(ts)*100, sum(t["pnl"] for t in ts)
    m4wr,m4pnl=m_stats(month_trades.get("2026-04",[]))
    m5wr,m5pnl=m_stats(month_trades.get("2026-05",[]))
    all_results.append((fname,len(trades),wr,cp,m4wr,m5wr,m4pnl,m5pnl))

all_results.sort(key=lambda x:-x[2])  # sort by win rate
for r in all_results:
    print(f"{r[0]:<42} {r[1]:>4} {r[2]:>6.1f}% {r[3]:>+8.2f}% {r[4]:>6.1f}% {r[5]:>6.1f}% {r[6]:>+7.2f}% {r[7]:>+7.2f}%")

# Show best by win rate
print(f"\n===== 胜率TOP3方案详情 =====")
for rank in range(min(3, len(all_results))):
    fname,n,wr,cp,_,_,_,_ = all_results[rank]
    if n==0: continue
    print(f"\n--- TOP{rank+1}: {fname} (胜率{wr:.1f}%, {n}笔, 累计{cp:+.2f}%) ---")
    # Show trades by month
    fcond = FILTERS[fname]
    by_month = defaultdict(list)
    for sd,sname,strat,idx,code in signals:
        df=dfs[code]; row=df.iloc[idx]
        if pd.isna(row.get("bn")): continue
        if not fcond(row): continue
        tr=sim_trade(df,idx,sname,strat)
        if tr:
            m=sd.strftime("%Y-%m")
            by_month[m].append({**tr,"sig_date":sd,"stock":sname,"strat":strat})
    for m in sorted(by_month):
        ts=by_month[m]
        w=sum(1 for t in ts if t["win"])
        cp_m=sum(t["pnl"] for t in ts)
        print(f"  {m}: {len(ts)}笔, 胜率{w/len(ts)*100:.1f}%, 累计{cp_m:+.2f}%")
    # Show losing trades
    losses=[t for t in by_month.get("2026-04",[])+by_month.get("2026-05",[]) if not t["win"]]
    if losses:
        print(f"  4-5月亏损交易:")
        for t in sorted(losses, key=lambda x:x["pnl"]):
            print(f"    {t['sig_date'].strftime('%m-%d')} {t['stock']:<6} {t['strat']:<4} {t['pnl']:+.2f}%")

print("\n分析完成")
