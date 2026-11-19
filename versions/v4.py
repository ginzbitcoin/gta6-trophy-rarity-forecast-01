"""Forecast trophy percentages using refresh-aware distributions; standard library only."""
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

VERSION = 4
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
        r = row['rates'][i]
        if r is None or not isinstance(r, (float, int)) or not math.isfinite(r):
            continue
        h = max(0.0, stamp(row['slot']) - launch)
        if not data or r != data[-1][1]:
            data.append((h, float(r), max(1, row.get('players_tracked') or 1)))
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

def saturation(data, target, days):
    fallback = curve(data,target,days)
    useful = [(h,r,n) for h,r,n in data if h >= 4 and r > 0.03 and r < 99.95]
    if len(useful)<4:
        return fallback
    peak = max(r for h,r,n in useful)
    if useful[-1][1] < useful[0][1]*0.95:
        # Declining rates are forecast directly: player growth can dilute earn rates.
        return fallback
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

def forecast(case):
    rows = []
    days = max(0.0, case['horizon_minutes'] / 1440.0)
    pooled = pooled_models(case)
    for i, trophy in enumerate(case['trophies']):
        data, target = observations(case, i)
        center, width = pooled[i]
        rows.append({'id': trophy['id'], 'q': [round(inv(center + z * width), 6) for z in Z]})
    return {'trophies': rows}

def main():
    out = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input')).glob('*.in')):
        case = json.loads(path.read_text())
        (out / (path.stem + '.out')).write_text(json.dumps(forecast(case), separators=(',', ':'), allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
