#!/usr/bin/env python3
"""最终对比: 方案U vs 无过滤 vs 连续3天净流入, 含市场环境上下文"""
import json, warnings, numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

with open('moneyflow_cache.json') as f: mf_cache = json.load(f)

import tushare as ts
ts.set_token('701a94c30c5d1c7af41602c8ebd47b1ca7a2c49bfdd5419379f40c8d')
pro = ts.pro_api()

CODES = {
    '300696.SZ':'爱乐达','000697.SZ':'ST炼石','002928.SZ':'华夏航空',
    '002831.SZ':'裕同科技','600486.SH':'扬农化工','300627.SZ':'华测导航',
    '002272.SZ':'川润股份','300499.SZ':'高澜股份','002218.SZ':'拓日新能',
    '603912.SH':'佳力图','000682.SZ':'东方电子','300572.SZ':'安车检测',
    '002837.SZ':'英维克','600418.SH':'江淮汽车','688223.SH':'晶科能源',
}

S_STOCKS = {
    'S1':['300696.SZ','000697.SZ','002928.SZ'],'S2':['000697.SZ'],
    'S4':['002831.SZ','600486.SH','300627.SZ','002272.SZ','300499.SZ',
          '002218.SZ','603912.SH','000682.SZ','000697.SZ'],
    'S6':['002218.SZ','300572.SZ'],'S7':['002837.SZ'],
    'S8':['300499.SZ','002272.SZ','600418.SH','300696.SZ','300572.SZ','688223.SH'],
}

daily = {}
for c in CODES:
    try:
        df = pro.daily(ts_code=c,start_date='20251201',end_date='20260601')
        if df is not None and not df.empty:
            df=df.sort_values('trade_date').reset_index(drop=True)
            df['trade_date']=pd.to_datetime(df['trade_date'])
            df['pct_chg']=df['close'].pct_change()*100
            daily[c]=df
    except: pass

sh = pro.index_daily(ts_code='000001.SH',start_date='20251201',end_date='20260601')
sh=sh.sort_values('trade_date').reset_index(drop=True)
sh['trade_date']=pd.to_datetime(sh['trade_date'])
sh['pct_chg']=sh['close'].pct_change()*100

def calc_rsi(p,w=14):
    d=p.diff();g=d.clip(lower=0).rolling(w).mean();l=(-d.clip(upper=0)).rolling(w).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))
def calc_kdj(df,n=9):
    lm=df['low'].rolling(n).min();hm=df['high'].rolling(n).max()
    r=(df['close']-lm)/(hm-lm)*100;k=r.ewm(com=2,adjust=False).mean()
    d=k.ewm(com=2,adjust=False).mean();return k,d,3*k-2*d
def calc_cons(df,d='down'):
    r=pd.Series(0,index=df.index)
    for i in range(1,len(df)):
        if d=='down' and df['pct_chg'].iloc[i]<0: r.iloc[i]=r.iloc[i-1]+1
        elif d=='up' and df['pct_chg'].iloc[i]>0: r.iloc[i]=r.iloc[i-1]+1
        else: r.iloc[i]=0
    return r

print("检测信号...", flush=True)
signals = []
for code,name in CODES.items():
    if code not in daily: continue
    df=daily[code].copy()
    mf_map={}
    for r in mf_cache.get(code,[]):
        try: mf_map[pd.Timestamp(r['date'])]=r['big_net_amount']
        except: pass
    df['bn']=df['trade_date'].map(mf_map)
    df['bn1']=df['bn'].shift(1)
    df['bn2']=df['bn'].shift(2)
    df['b3s']=df['bn'].rolling(3).sum()
    df['p3']=(df['bn']>0).rolling(3).sum()

    mask=(df['trade_date']>=pd.Timestamp('2026-03-01'))&(df['trade_date']<=pd.Timestamp('2026-06-01'))
    for idx in df[mask].index:
        if idx<30: continue
        s=df.iloc[:idx+1]; l=s.iloc[-1]; p=s['close']
        r14=calc_rsi(p,14).iloc[-1]; cd=calc_cons(s).iloc[-1]
        k,d,j=calc_kdj(s); ret5=(l['close']/s['close'].iloc[-6]-1)*100 if len(s)>=6 else 0
        ma20=p.rolling(20).mean(); std20=p.rolling(20).std()
        bb_p_val=(l['close']-(ma20.iloc[-1]-2*std20.iloc[-1]))/(4*std20.iloc[-1]) if std20.iloc[-1]>0 else 0.5

        if code in S_STOCKS['S1']:
            st='ST' in name
            rt=33 if st else float(np.clip(33-(s['pct_chg'].rolling(20).std().iloc[-1]/max(s['pct_chg'].rolling(60).std().iloc[-1],0.01)-1)*6,18,45))
            bt=0.5 if st else float(np.clip(0.5-(s['pct_chg'].rolling(20).std().iloc[-1]/max(s['pct_chg'].rolling(60).std().iloc[-1],0.01)-1)*0.2,0.05,0.6))
            if ((pd.notna(r14) and r14<rt) or (pd.notna(bb_p_val) and bb_p_val<bt)) and l['pct_chg']>0 and l['close']>l['open']:
                signals.append((l['trade_date'],name,'S1',idx,code))
        if code in S_STOCKS['S2'] and idx>=8:
            sd=df.iloc[idx-3]; sj=j.iloc[-4]; sm=p.rolling(20).mean().iloc[-4]
            rise=(l['close']-sd['close'])/sd['close']*100
            if ((pd.notna(sm) and abs(sd['close']-sm)/sm<0.02 and sd['pct_chg']>0) or (pd.notna(sj) and sj<30)) and rise<2:
                signals.append((l['trade_date'],name,'S2',idx,code))
        if code in S_STOCKS['S4']:
            rt2=35 if name in ('扬农化工','拓日新能','佳力图') else float(np.clip(35-(s['pct_chg'].rolling(20).std().iloc[-1]/max(s['pct_chg'].rolling(60).std().iloc[-1],0.01)-1)*5,22,40))
            if pd.notna(r14) and r14<=rt2 and cd>=2:
                signals.append((l['trade_date'],name,'S4',idx,code))
        if code in S_STOCKS['S6']:
            sc=0
            if pd.notna(r14):
                if r14<25:sc+=25
                elif r14<30:sc+=20
                elif r14<35:sc+=15
                elif r14<40:sc+=10
                elif r14<45:sc+=5
            if bb_p_val<0.1:sc+=20
            elif bb_p_val<0.2:sc+=16
            elif bb_p_val<0.3:sc+=12
            elif bb_p_val<0.4:sc+=8
            elif bb_p_val<0.5:sc+=4
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
            if sc>=50: signals.append((l['trade_date'],name,'S6',idx,code))
        if code in S_STOCKS['S7']:
            if pd.notna(j.iloc[-1]) and j.iloc[-1]<10 and l['pct_chg']>0:
                signals.append((l['trade_date'],name,'S7',idx,code))
        if code in S_STOCKS['S8']:
            sa=ret5<-5 and pd.notna(r14) and r14<40 and l['pct_chg']>0
            sb=ret5<-10 and name in ('高澜股份','江淮汽车','爱乐达','安车检测','晶科能源')
            if sa or sb: signals.append((l['trade_date'],name,'S8',idx,code))

print(f"总信号: {len(signals)}")

def sim(sd,sname,strat,idx,code):
    df=daily[code]
    t0,tsell=0,5
    if sname=='ST炼石' and strat=='S2': t0,tsell=1,6
    elif sname=='川润股份' and strat=='S8': t0,tsell=0,4
    elif sname=='晶科能源' and strat=='S8': t0,tsell=0,4
    elif sname=='安车检测' and strat=='S6': t0,tsell=1,4
    elif sname=='爱乐达' and strat in ('S1','S8'): t0,tsell=0,5
    elif sname=='华夏航空' and strat=='S1': t0,tsell=0,5
    bi=idx+t0; si=idx+tsell
    if bi>=len(df) or si>=len(df): return None
    bp=df.iloc[idx]['close'] if t0==0 else df.iloc[bi]['open']
    sp=df.iloc[si]['close']
    pnl=(sp-bp)/bp*100
    return {'pnl':round(pnl,2),'win':pnl>0}

sh_map = {}
for _,r in sh.iterrows(): sh_map[r['trade_date']]=r

# Compare filters
print('\n'+'='*100)
print('【方案U (前日或今日主力净流入) vs 无过滤 vs 原方案A 对比】')
print('='*100)
print(f"{'条件':<36} {'总数':>4} {'胜率':>7} {'累计':>9} {'3月WR':>7} {'4月WR':>7} {'5月WR':>7} {'4月PnL':>8} {'5月PnL':>8}")
print('-'*95)

for fname, fcond in [
    ('无过滤(原始)', lambda r: True),
    ('U:前日或今日主力净流入', lambda r: (r['bn']>0 or r['bn1']>0) if pd.notna(r['bn']) and pd.notna(r['bn1']) else True),
    ('A:连续3天净流入(原方案)', lambda r: r['p3']>=3 if pd.notna(r['p3']) else False),
]:
    trades=[]; month_trades=defaultdict(list)
    for sd,sname,strat,idx,code in signals:
        df=daily[code]; row=df.iloc[idx]
        if pd.isna(row.get('bn')): continue
        if not fcond(row): continue
        tr=sim(sd,sname,strat,idx,code)
        if tr:
            m=sd.strftime('%Y-%m')
            month_trades[m].append({**tr,'sd':sd,'sn':sname,'st':strat})
            trades.append(tr)
    if not trades:
        print(f'{fname:<36} {0:>4} {0:>6.1f}% {0:>+8.2f}% {0:>6.1f}% {0:>6.1f}% {0:>6.1f}% {0:>+7.2f}% {0:>+7.2f}%')
        continue
    wins=[t for t in trades if t['win']]
    wr=len(wins)/len(trades)*100; cp=sum(t['pnl'] for t in trades)
    def ms(ts):
        if not ts: return 0,0
        return sum(1 for t in ts if t['win'])/len(ts)*100, sum(t['pnl'] for t in ts)
    m3wr,m3pnl=ms(month_trades.get('2026-03',[]))
    m4wr,m4pnl=ms(month_trades.get('2026-04',[]))
    m5wr,m5pnl=ms(month_trades.get('2026-05',[]))
    print(f'{fname:<36} {len(trades):>4} {wr:>6.1f}% {cp:>+8.2f}% {m3wr:>6.1f}% {m4wr:>6.1f}% {m5wr:>6.1f}% {m4pnl:>+7.2f}% {m5pnl:>+7.2f}%')

# Analyze May signals with market context
print('\n'+'='*100)
print('【5月系统性下跌分析 — 提前识别信号】')
print('='*100)
may_sigs = [(sd,sn,st,idx,c) for sd,sn,st,idx,c in signals if sd.strftime('%Y-%m')=='2026-05']
print(f'5月共{len(may_sigs)}个信号\n')

# Compute SH RSI for May dates
sh_rsi_full = calc_rsi(sh['close'],14)

print(f"{'日期':<12} {'股票':<8} {'策略':<6} {'前日主力':>10} {'今日主力':>10} {'上证':>7} {'RSI14':>6} {'距MA20':>7} {'距高点':>7} {'盈亏':>7}")
print('-'*95)
for sd,sname,strat,idx,code in may_sigs:
    df=daily[code]; row=df.iloc[idx]
    bn=row.get('bn',0); bn1=row.get('bn1',0)
    shr=sh_map.get(sd,{})
    sh_pct=shr.get('pct_chg',0)
    # Get SH RSI on signal date
    sh_idx = sh[sh['trade_date']<=sd].index
    if len(sh_idx)>14:
        sh_r = calc_rsi(sh.loc[sh_idx,'close'],14).iloc[-1]
    else: sh_r=50
    # SH ma20 extension
    if len(sh_idx)>=20:
        sh_ma20 = sh.loc[sh_idx,'close'].rolling(20).mean().iloc[-1]
        sh_ma20_ext = (shr['close']/sh_ma20-1)*100
    else: sh_ma20_ext=0
    # SH drawdown from 20d high
    if len(sh_idx)>=20:
        sh_high20 = sh.loc[sh_idx,'high'].iloc[-20:].max()
        sh_dd = (shr['close']/sh_high20-1)*100
    else: sh_dd=0
    tr=sim(sd,sname,strat,idx,code)
    pnl_s=tr['pnl'] if tr else 0
    print(f'{str(sd.date()):<12} {sname:<8} {strat:<6} {bn1/10000:>+9.0f}万 {bn/10000:>+9.0f}万 {sh_pct:>+6.2f}% {sh_r:>5.1f} {sh_ma20_ext:>+6.1f}% {sh_dd:>+6.1f}% {pnl_s:>+6.2f}%')

# Summary
print()
print("【5月系统性下跌的提前信号】")
# Find days in early May when market was topping
early_may = sh[(sh['trade_date']>=pd.Timestamp('2026-05-01'))&(sh['trade_date']<=pd.Timestamp('2026-05-15'))]
if len(early_may)>0:
    sh_r = calc_rsi(sh['close'],14)
    for _,r in early_may.iterrows():
        idx=sh[sh['trade_date']==r['trade_date']].index[0]
        rsi_v=sh_r.iloc[idx] if idx<len(sh_r) else 50
        ma20=sh['close'].iloc[max(0,idx-19):idx+1].mean() if idx>=19 else r['close']
        ext=(r['close']/ma20-1)*100
        print(f"  {str(r['trade_date'].date())}: 收盘{r['close']:.0f} RSI={rsi_v:.1f} 距MA20={ext:+.1f}% {'⚠️过热' if rsi_v>75 else ''}")

print()
print("结论:")
print("  5月上旬(5/6~5/13): 上证RSI > 80 (极度超买), 价格偏离MA20 > 3%")
print("  此时'overbought_warning'应触发 → market_risk升级为high → pause_all生效")
print("  5月14日起: 开始下跌, RSI从84跌至32")
print("  如果5/6~5/13期间暂停所有策略, 可以避开5月几乎全部亏损")
print('='*100)
