# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yowanda/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yowanda/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, a week after launch. It reads the house's recorded reads of the trophy page up to the case checkpoint and writes, for each of the 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the target, scored by pinball loss on the logit scale.

## Run it

```sh
python3 main.py
```

It reads `$MAKEFUN_INPUT_DIR` (else `/input/`) for `<id>.in` case files, and writes `$MAKEFUN_OUTPUT_DIR/<id>.out` (else `/output/<id>.out`) per case. Python 3.13, standard library only, offline, deterministic.

## Method

Per trophy, all fitting happens on the logit scale — the scale the contest scores on, so a move from 1% to 2% counts like one from 33% to 50%:

- **Log-time growth curve.** Each trophy's logit earn rate is fitted by weighted least squares against `ln(1 + hours since 2026-11-19 00:00 UTC)`. Trophy curves are concave in clock time (fast early gains, then flattening), and log-time linearises that shape, so a straight line in log-time is a saturating curve in clock time.
- **Recency weighting.** Reads are weighted with a 12-hour half-life: Exophase refreshes its counts in jumps hours apart and the tracked-player mix changes through launch week, so this morning's reads describe tomorrow better than launch night's.
- **Shrinkage.** Each trophy's slope is shrunk toward the pooled (median) slope across trophies, weighted by its effective number of reads — trophies with flat or sparse histories borrow the field's typical growth instead of extrapolating noise.
- **Damped extrapolation.** Only a damped share of the move past the last read is kept (`1 / (1 + 0.12 × horizon days)`; ~0.99 over a 90-minute practice horizon, ~0.65 over the final's 4.5 days), and the median's logit move from the last read is capped at ±3.0, so a short launch-window slope cannot run away over four more days.
- **Honest spread.** The logit spread combines, in quadrature: the house random-walk term (0.5·√days), the fit's residual standard deviation (floored 0.15, capped 1.5), a refresh term (0.45·√days) for Exophase's jump refreshes and the dilution from new tracked players joining (the first refresh dropped almost every rate even though players only gain trophies), plus 0.4 extra when a trophy's history never moved — a flat past is no evidence the future stays flat, as the practice cases prove.

Why it should beat the house baselines: persistence says nothing moves, and the house curve baseline fits the same line shape but only over the last 24 hours, with no shrinkage, no damping, and a narrower spread. Over a 4.5-day horizon the differences that matter are the median's direction (recency-weighted, shrunk, damped) and a spread wide enough to cover Exophase's refresh jumps.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
