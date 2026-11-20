# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ryukaa853/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ryukaa853/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players Exophase tracks who will have earned it at the case's target, as five quantiles (5/25/50/75/95%) per trophy, scored by pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

Entered on [make.fun](https://make.fun/?ref=8tc4hwe6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards

## How it works

Everything is modelled in earner *counts*, not rates alone: a rate is earners over players tracked, and the two move for different reasons. Players keep earning trophies (counts only grow) while new players keep joining the tracker (the denominator grows, pulling every rate down at each Exophase refresh).

1. **Flat history** (players tracked never changed in any read — the practice cases). Exophase has not refreshed the page once inside the window, so the next refresh is the whole story. The launch-night refresh the rules describe is the prior: 638 → 3,868 players tracked (×6.06), 943 → 4,276 trophies earned, the one trophy at 100% staying at 100%, and only ~103 new earners over the other 54 trophies (~0.032 per new player), allocated in proportion to the square of each trophy's current earner count (early, widely-held trophies are the ones fresh players reach first). The forecast is a mixture: with weight exp(−frozen hours / 4) the refresh has not happened by the target and rates stay put; otherwise they sit log-normally (logit sd 0.45) around the diluted value.
2. **Moving history** (the final case). Per trophy, a least-squares line of logit(rate) on log(1 + hours since release), slopes shrunk toward the cross-trophy median slope, read at the target: fitting the observed rate keeps dilution and progression together as the page shows them, and the logit scale keeps saturating trophies from being extrapolated past 100%. A second, counts-based extrapolation (log earners and log players tracked separately) guards the downside: when it sits below the logit fit — dilution outrunning progression — the median moves halfway toward it and the spread widens. Spreads combine each fit's residual scatter with a base uncertainty that grows with the horizon, and the platinum is capped at the least-held other trophy.

## Properties

- Python 3.13, standard library only; no network, no clock, no randomness — two runs on the same case write byte-identical output.
- Fast: well under 1 s per case.
