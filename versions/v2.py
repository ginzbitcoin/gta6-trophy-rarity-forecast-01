"""Forecast trophy percentages using refresh-aware distributions; standard library only."""
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

VERSION = 2
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

def forecast(case):
    rows = []
    days = max(0.0, case['horizon_minutes'] / 1440.0)
    for i, trophy in enumerate(case['trophies']):
        data, target = observations(case, i)
        center, width = curve(data, target, days)
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
