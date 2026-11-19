"""Forecast trophy percentages using refresh-aware distributions; standard library only."""
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1
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

def refresh_quantiles(data,target,days):
    center,width=persistence(data,target,days)
    anchor=logit(data[-1][1]) if data else logit(0.01)
    # Read quantiles from a two-scenario CDF rather than averaging quantiles.
    unchanged=0.45 if len(data)>2 else 0.6
    quantiles=[]
    for tau in LEVELS:
        lo=min(anchor,center)-9*width
        hi=max(anchor,center)+9*width
        for step in range(48):
            mid=(lo+hi)/2
            still=0.5*(1+math.erf((mid-anchor)/(0.04*math.sqrt(2))))
            refresh=0.5*(1+math.erf((mid-center)/(width*math.sqrt(2))))
            if unchanged*still+(1-unchanged)*refresh < tau:
                lo=mid
            else:
                hi=mid
        quantiles.append(round(inv((lo+hi)/2),6))
    return quantiles

def forecast(case):
    rows = []
    days = max(0.0, case['horizon_minutes'] / 1440.0)
    for i, trophy in enumerate(case['trophies']):
        data, target = observations(case, i)
        center, width = persistence(data, target, days)
        rows.append({'id': trophy['id'], 'q': refresh_quantiles(data,target,days)})
    return {'trophies': rows}

def main():
    out = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input')).glob('*.in')):
        case = json.loads(path.read_text())
        (out / (path.stem + '.out')).write_text(json.dumps(forecast(case), separators=(',', ':'), allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
