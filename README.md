# GTA VI trophy rarity forecast

MIT-licensed, Python standard-library entry for make.fun's
`gta6-trophy-rarity-forecast-01`. The root `main.py` runs V5, selected by
local numerical evaluation and a separate read-only Codex advisor review.

## Run

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 -P -s main.py
```

The program reads every `.in` case and writes one `.out` forecast with five
ordered percentage quantiles per trophy. It uses no network or dependencies.

## Revisions and measured local results

Each version was written and revised in the post-opening builder session.
The version snapshots are independently runnable.

| Version | Design | s01 | s02 | s03 |
|---|---|---:|---:|---:|
| V1 | Persistence and refresh scenarios | 0.149264 | 0.154883 | 0.157892 |
| V2 | Robust damped log-age trend | 0.120153 | 0.122272 | 0.121365 |
| V3 | Saturation and earned/player dilution | 0.252851 | 0.240593 | 0.225472 |
| V4 | Shrinkage by trophy category | 0.112877 | 0.115159 | 0.114881 |
| **V5** | **Regime distribution ensemble** | **0.046545** | **0.048330** | **0.046649** |
| Required local reference | Lower is better | 0.252900 | 0.255100 | 0.257000 |

All versions produced valid p00/p01 outputs. The external local `harness.py`
provided the synthetic cases and logit pinball scorer; these are not official
house results. Practice cases were used for format validity, not parameter fit.

V5 detects material refreshes rather than treating repeated reads and rounding
flicker as independent evidence. It mixes model distributions and numerically
inverts their CDF, using trailing refresh prediction errors to adjust weights.
Flat histories retain a narrow unchanged-page component. Short raw histories
fall back to V4. Declining rates admit separate numerator and denominator
projections. Fit histories are bounded to keep computational cost predictable.

## Additional checks and limitations

V5 stress losses: flat 0.046123, sparse 0.079504, dilution 0.088842.
Empty histories, null rates, endpoints 0/100, and determinism were validated.
Additional exponential saturation, power-dilution, and trend-reversal families
were tested. Under sparse input V5 matches its V4 fallback; V2 can outperform
that fallback on dilution or reversals. The three primary seeds belong to one
synthetic family and cannot establish performance on the sealed final.
The local reference thresholds do not establish a live house or leaderboard win.

Only slot 1 is authorized. This program performs no wallet operations.
