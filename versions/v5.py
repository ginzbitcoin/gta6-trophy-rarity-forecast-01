"""Forecast trophy percentages using refresh-aware distributions; standard library only."""
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

VERSION = 5
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

def logit(p):
    p = min(99.99, max(0.01, p))
    return math.log(p / (100.0 - p))

def inv(x):
    return 100.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))

def stamp(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() / 3600.0

def observations(case, i):
    launch = stamp(case['game'].get('release_date', '2026-11-19')[:10] + 'T00:00:00Z')
    data = []
    for row in case['history']:
        r = row['rates'][i] if i < len(row['rates']) else None
        if r is None or not isinstance(r, (float, int)) or not math.isfinite(r):
            continue
        h = max(0.0, stamp(row['slot']) - launch)
        if not data or r != data[-1][1]:
            data.append((h, float(r), max(1, row.get('players_tracked') or 1)))
    if len(data)>32:
        data=[data[round(j*(len(data)-1)/31)] for j in range(32)]
    return data, max(0.0, stamp(case['target']) - launch)

def persistence(data, target, days):
    last = data[-1][1] if data else 0.01
    jumps = [logit(b[1]) - logit(a[1]) for a, b in zip(data, data[1:])]
    drift = statistics.median(jumps[-4:]) if jumps else -0.18
    width = max(0.22, 0.48 * math.sqrt(max(days, 0.01)))
    if jumps:
        width = max(width, statistics.median(abs(x) for x in jumps) * 1.25)
    return logit(last) + max(-0.6, min(0.6, drift)) * min(1.0, days), width

def slope(points):
    if len(points) < 2:
        return 0.0
    candidates = [(b[1]-a[1])/(b[0]-a[0]) for j,a in enumerate(points) for b in points[j+1:] if b[0]-a[0] > 0.05]
    return statistics.median(candidates) if candidates else 0.0

def curve(data, target, days):
    base, width = persistence(data, target, days)
    usable = [(math.log1p(h), logit(r)) for h,r,n in data if h >= 4]
    if len(usable) < 3:
        return base, width
    long = slope(usable)
    short = slope(usable[-5:])
    disagreement = abs(short-long)
    damping = 0.72 / (1.0 + disagreement)
    drift = (0.85*short + 0.15*long) * (math.log1p(target)-usable[-1][0]) * damping
    center = logit(data[-1][1]) + max(-1.8, min(1.8, drift))
    return center, max(0.16, min(width, 0.25 + disagreement*0.25))

def regress(points):
    total = sum(w for x,y,w in points)
    mx = sum(x*w for x,y,w in points)/total
    my = sum(y*w for x,y,w in points)/total
    den = sum(w*(x-mx)**2 for x,y,w in points)
    beta = sum(w*(x-mx)*(y-my) for x,y,w in points)/max(den, 1e-12)
    alpha = my-beta*mx
    error = sum(w*(y-alpha-beta*x)**2 for x,y,w in points)/total
    return alpha,beta,error

def dilution_projection(data,target):
    # Model numerator and denominator separately in count space.
    usable=[(h,r,n) for h,r,n in data if h>=4 and r>0]
    if len(usable)<3:
        return None
    tail=usable[-8:]
    players=[(math.log1p(h),math.log(n)) for h,r,n in tail]
    earned=[(math.log1p(h),math.log(max(0.01,n*r/100))) for h,r,n in tail]
    future=max(0.0,math.log1p(target)-math.log1p(tail[-1][0]))
    denominator=tail[-1][2]*math.exp(min(3.0,max(0.0,slope(players))*future*0.65))
    numerator=tail[-1][2]*tail[-1][1]/100*math.exp(min(3.0,max(0.0,slope(earned))*future*0.65))
    # Earned counts cannot exceed the projected tracked population.
    projected_rate=100*min(denominator,numerator)/max(1,denominator)
    return logit(projected_rate)

def saturation(data, target, days):
    fallback = curve(data,target,days)
    useful = [(h,r,n) for h,r,n in data if h >= 4 and r > 0.03 and r < 99.95]
    if len(useful)<4:
        return fallback
    peak = max(r for h,r,n in useful)
    if useful[-1][1] < useful[0][1]*0.95:
        # Blend a separate earned/player projection when dilution dominates.
        counted=dilution_projection(useful,target)
        return (0.55*fallback[0]+0.45*counted,max(0.3,fallback[1])) if counted is not None else fallback
    fits=[]
    for k in range(1,241):
        cap = peak + (100.0-peak)*(k/240.0)**2
        points=[(math.log1p(h),math.log(r/(cap-r)), 0.25+(h/max(1,useful[-1][0]))**2) for h,r,n in useful]
        a,b,e=regress(points)
        if not 0.02 < b < 5.0:
            continue
        predicted = cap/(1+math.exp(-max(-40,min(40,a+b*math.log1p(target)))))
        fits.append((e, logit(predicted), b, cap))
    if not fits:
        return fallback
    fits.sort()
    best=fits[0]
    near=[x for x in fits if x[0] < best[0]+0.00015]
    center=statistics.median(x[1] for x in near)
    spread=max(0.09, min(0.65, 2*math.sqrt(best[0])+statistics.pstdev(x[1] for x in near)))
    # A denominator jump without a comparable earned-count jump indicates dilution.
    if len(useful)>2:
        a0,a1=useful[-2:]
        player_growth=math.log(a1[2]/a0[2])
        earned_growth=math.log(max(0.01,a1[1])*a1[2]/(max(0.01,a0[1])*a0[2]))
        if player_growth > 0.05 and earned_growth < 0.5*player_growth:
            center=0.65*center+0.35*fallback[0]
            spread=max(spread,0.3)
    return center,spread

def category(trophy):
    text=(trophy['name']+' '+trophy['description']).lower()
    if trophy['tier']=='platinum' or '100%' in text:
        return 'completion'
    if 'chapter' in text or 'mission' in text:
        return 'story'
    return trophy['tier']

def pooled_models(case):
    days=max(0.0,case['horizon_minutes']/1440.0)
    groups={}
    models=[]
    for i,trophy in enumerate(case['trophies']):
        data,target=observations(case,i)
        c,w=curve(data,target,days)
        last=logit(data[-1][1]) if data else logit(0.01)
        drift=c-last
        groups.setdefault(category(trophy),[]).append(drift)
        models.append((data,target,c,w,last))
    result=[]
    for trophy,(data,target,c,w,last) in zip(case['trophies'],models):
        peers=groups[category(trophy)]
        pooled=statistics.median(peers) if len(peers)>=3 else c-last
        weight=min(0.95,max(0.35,(len(data)-1)/(len(data)+2)))
        center=last+weight*(c-last)+(1-weight)*pooled
        result.append((center,max(0.14,w*0.85)))
    return result

def mixture_quantiles(components):
    answer=[]
    for tau in LEVELS:
        lo=min(c-9*w for c,w,weight in components)
        hi=max(c+9*w for c,w,weight in components)
        for step in range(48):
            mid=(lo+hi)/2
            cdf=sum(weight*0.5*(1+math.erf((mid-c)/(max(0.02,w)*math.sqrt(2)))) for c,w,weight in components)
            if cdf < tau:
                lo=mid
            else:
                hi=mid
        answer.append(round(inv((lo+hi)/2),6))
    return answer

def refresh_case(case):
    # Ignore rounding flicker: require a material change across many trophies.
    selected=[]
    for row in case['history']:
        if not selected:
            selected.append(row)
            continue
        changes=[abs(logit(a)-logit(b)) for a,b in zip(row['rates'],selected[-1]['rates']) if a is not None and b is not None]
        if changes and statistics.median(changes)>0.025:
            selected.append(row)
    clean=dict(case)
    clean['history']=selected
    return clean

def ensemble(data,target,days,pooled):
    if not data:
        return mixture_quantiles([(logit(0.01), max(0.5, math.sqrt(days)), 1.0)])
    if len(data)==1:
        # An explicit unchanged-page atom plus a refresh scenario.
        anchor=logit(data[-1][1])
        width=max(0.18,0.45*math.sqrt(max(days,0.01)))
        return mixture_quantiles([(anchor,0.02,0.65),(anchor-0.22,width,0.35)])
    p=persistence(data,target,days)
    c=curve(data,target,days)
    s=saturation(data,target,days)
    if len(data)<4:
        weights=[0.65,0.15,0.05,0.15]
    else:
        # Saturation is trusted when it predicts preceding refreshes successfully.
        cv=[]
        sv=[]
        for end in range(max(4,len(data)-3),len(data)):
            earlier=data[:end]
            y=logit(data[end][1])
            dd=max(0.01,(data[end][0]-earlier[-1][0])/24)
            cv.append(abs(curve(earlier,data[end][0],dd)[0]-y))
            sv.append(abs(saturation(earlier,data[end][0],dd)[0]-y))
        sgood=bool(sv) and statistics.mean(sv)<statistics.mean(cv)
        weights=[0.04,0.24,0.48 if sgood else 0.12,0.24 if sgood else 0.60]
    return mixture_quantiles([(a,b,w) for (a,b),w in zip((p,c,s,pooled),weights)])

def forecast(case):
    raw_case = case
    sparse = len(case['history']) < 20
    case = refresh_case(case) if VERSION == 5 else case
    rows = []
    days = max(0.0, case['horizon_minutes'] / 1440.0)
    pooled = pooled_models(raw_case if VERSION == 5 and sparse else case)
    for i, trophy in enumerate(case['trophies']):
        data, target = observations(case, i)
        center, width = pooled[i]
        rows.append({'id': trophy['id'], 'q': ([round(inv(pooled[i][0]+z*pooled[i][1]),6) for z in Z] if sparse and len(data)>1 else ensemble(data,target,days,pooled[i])) if VERSION == 5 else [round(inv(center + z * width), 6) for z in Z]})
    return {'trophies': rows}

def main():
    out = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input')).glob('*.in')):
        case = json.loads(path.read_text())
        (out / (path.stem + '.out')).write_text(json.dumps(forecast(case), separators=(',', ':'), allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
