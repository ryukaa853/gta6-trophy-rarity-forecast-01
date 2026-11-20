"""Forecast GTA VI trophy earn rates on Exophase (gta6-trophy-rarity-forecast-01).

Reads one case per *.in file (schema "makefun/gta6-trophies-case/v1") from
$MAKEFUN_INPUT_DIR (default /input/) and writes one <id>.out per case to
$MAKEFUN_OUTPUT_DIR (default /output/): {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}.

Model (standard library only, deterministic, no network, no clock):
everything is done with earner *counts*, not with rates alone, because a
rate is earners divided by players tracked, and the two move for different
reasons: players keep earning trophies (counts only grow) while new players
keep joining the tracker (the denominator grows, which pulls every rate
down at each Exophase refresh).

Two regimes, picked from the case's own history:

1. Flat history (players tracked never changed in any read). Exophase has
   not refreshed the page once inside the observed window, so the next
   refresh is the whole story. The launch-night refresh the rules describe
   is the prior: 638 -> 3,868 players tracked (x6.06), 943 -> 4,276 trophies
   earned, the one trophy at 100% staying at 100%, and only ~103 new
   earners spread over the other 54 trophies (~0.032 per new player).
   New earners are allocated across trophies in proportion to the square
   of each trophy's current earner count (early, widely-held trophies are
   the ones fresh players reach first; a flat allocation and a linear one
   were both tried against the house baseline's stated practice loss of
   ~1.51 and the squared allocation matched its concentration best).
   The forecast is a two-part mixture: with weight w0 the refresh has not
   happened by the target and every rate stays put; with weight 1 - w0 it
   has, and rates sit log-normally (logit sd 0.45) around the diluted
   value. w0 = exp(-frozen_hours / 4), where frozen_hours runs from the
   first read to the target: the longer the page has sat frozen, the more
   due the refresh is.

2. Moving history (the final case). Per trophy, a least-squares line of
   logit(rate) on log(1 + hours since release), fitted on the last 24
   hours of reads only, slopes shrunk toward the cross-trophy median
   slope, read at the target: fitting the observed rate keeps dilution
   and progression together as the page shows them, the logit scale
   keeps saturating trophies from being extrapolated past 100%, and the
   last-day window keeps the launch's first hours -- near-zero rates,
   explosive growth -- from steepening the line. A second, counts-based
   extrapolation (log earners and log players tracked separately) guards
   the downside: when it sits below the logit fit -- dilution outrunning
   progression -- the median moves halfway toward it and the spread
   widens. Spreads combine each fit's residual scatter with a base
   uncertainty that grows with the horizon, and the platinum is capped
   at the least-held other trophy (it needs them all).

Quantiles are normal quantiles on the logit scale (z = -1.645, -0.674, 0,
0.674, 1.645), rounded to 4 decimals half-up. No randomness anywhere, so
two runs write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Launch-night first-refresh prior (flat regime only) -------------------
# From the rules: the first refresh took players tracked 638 -> 3868 and
# trophies earned 943 -> 4276. The 100% trophy accounts for all 3230 new
# players; the other 54 trophies shared only 103 new earners.
REFRESH_PLAYERS_MULT = 3868.0 / 638.0          # ~6.06
REFRESH_NEW_EARNERS_PER_PLAYER = 103.0 / 3230.0  # ~0.032, other trophies
REFRESH_MEAN_INTERVAL_H = 4.0
REFRESH_ALLOC_EXPONENT = 2.0
REFRESH_LOGIT_SD = 0.45

# --- Trend regime -----------------------------------------------------------
TREND_SHRINK_PRIOR = 4.0   # pseudo-reads pulling each slope to the median
TREND_BASE_VAR = 0.09      # (0.30 logit)^2 of irreducible forecast spread
TREND_HORIZON_VAR_PER_DAY = 0.02
TREND_MIN_SD = 0.50
TREND_MAX_SD = 1.25
TREND_MAX_PLAYERS_MULT = 8.0


def clamp_pct(x):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO 8601 timestamp (trailing Z, or a bare date) to UTC."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual sd or None)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return a, b, math.sqrt(max(0.0, var))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def mixture_quantiles(old_rate, diluted_rate, w_no_refresh, sd):
    """Quantiles of a two-part predictive distribution on the logit scale.

    With weight w_no_refresh the rate is still old_rate (point mass); with
    the rest it is log-normal around diluted_rate with logit sd `sd`.
    Inverted on a fixed logit grid, so the result is deterministic.
    """
    if diluted_rate >= 99.9 or old_rate >= 99.99:
        return [100, 100, 100, 100, 100]
    lo = logit(0.01)
    hi = logit(99.99)
    n_grid = 1201
    step = (hi - lo) / (n_grid - 1)
    grid = [lo + step * k for k in range(n_grid)]
    lg_old = logit(old_rate)
    lg_dil = logit(diluted_rate)
    cdfs = []
    for x in grid:
        mass = w_no_refresh if x >= lg_old else 0.0
        cdfs.append(mass + (1.0 - w_no_refresh) * phi((x - lg_dil) / sd))
    out = []
    for tau in LEVELS:
        pick = n_grid - 1
        for k, cdf in enumerate(cdfs):
            if cdf >= tau:
                pick = k
                break
        out.append(round4(inv_logit(grid[pick])))
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def flat_forecast(case, last_rates, last_counts, players_now):
    """History never moved: forecast the overdue first refresh."""
    history = case["history"]
    first_at = parse_time(history[0]["at"])
    target_at = parse_time(case["target"])
    frozen_h = max(0.0, hours_between(first_at, target_at))
    w_no_refresh = math.exp(-frozen_h / REFRESH_MEAN_INTERVAL_H)
    players_pred = players_now * REFRESH_PLAYERS_MULT
    new_players = players_pred - players_now
    new_total = REFRESH_NEW_EARNERS_PER_PLAYER * new_players
    weights = []
    for i, c in enumerate(last_counts):
        if last_rates[i] is not None and last_rates[i] >= 99.99:
            weights.append(0.0)
        else:
            weights.append((max(c, 0.0)) ** REFRESH_ALLOC_EXPONENT)
    w_sum = sum(weights)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        old = last_rates[i]
        if old is None:
            old = 0.0
        if old >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        share = (weights[i] / w_sum) if w_sum > 0 else (1.0 / max(1, len(weights)))
        diluted = 100.0 * (last_counts[i] + share * new_total) / players_pred
        rows.append({
            "id": trophy["id"],
            "q": mixture_quantiles(old, diluted, w_no_refresh, REFRESH_LOGIT_SD),
        })
    return {"trophies": rows}


def trend_forecast(case, history, last_rates, last_counts, players_now):
    """History moved: extrapolate each trophy's rate in logit space.

    Primary: per trophy, a least-squares line of logit(rate) on
    log(1 + hours since release), slopes shrunk toward the cross-trophy
    median slope, read at the target. Fitting the observed rate (rather
    than earners and players separately) keeps dilution and progression
    together as the page actually shows them, and the logit scale keeps
    saturating trophies from being extrapolated past 100% -- a separate
    log-count extrapolation overshoots badly there (it once put an early
    trophy at 99.99% against a 75% truth on a synthetic final-like case).

    Guard: earners and players tracked are also extrapolated separately
    (log counts, log players). When that counts-based rate sits *below*
    the logit fit -- the signature of dilution outrunning progression --
    the median moves halfway toward it in logit space and the spread
    widens by the disagreement. When the counts rate sits above, it is
    ignored: its upside is the unreliable side (saturation).
    """
    release = parse_time(case["game"].get("release_date", history[0]["at"]))
    target_at = parse_time(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    x_target = math.log1p(max(0.0, hours_between(release, target_at)))
    # Fit on the last 24 hours of reads only. The launch's first hours
    # are a different regime -- near-zero rates, explosive growth -- and
    # a line fitted through them extrapolates far too steeply: on a
    # synthetic final-like case it put an early trophy at 92% against a
    # 60% truth. The last day is the regime the target extends.
    last_at = parse_time(history[-1]["at"])
    window_start = last_at.timestamp() - 24.0 * 3600.0
    fit_idx = [k for k, read in enumerate(history)
               if parse_time(read["at"]).timestamp() >= window_start]
    if len(fit_idx) < 6:
        fit_idx = list(range(len(history)))
    xs_all = []
    for read in history:
        t = max(0.0, hours_between(release, parse_time(read["at"])))
        xs_all.append(math.log1p(t))
    xs = [xs_all[k] for k in fit_idx]
    n_trophies = len(case["trophies"])
    # Per-trophy logit-rate fits, for the slope shrinkage pool.
    fits = []
    for i in range(n_trophies):
        txs, tys = [], []
        for pos, k in enumerate(fit_idx):
            rate = history[k]["rates"][i]
            if rate is None:
                continue
            txs.append(xs[pos])
            tys.append(logit(rate))
        fits.append((txs, tys))
    slopes = []
    for txs, tys in fits:
        if len(txs) >= 3:
            slopes.append(ols(txs, tys)[1])
    med_slope = median(slopes) if slopes else 0.0
    resid_pool = []
    for txs, tys in fits:
        if len(txs) >= 3:
            resid_pool.append(ols(txs, tys)[2])
    med_resid = median(resid_pool) if resid_pool else 0.0
    # Counts-based guard: log players tracked and log earner counts.
    p_logs = [math.log(max(1, history[k]["players_tracked"])) for k in fit_idx]
    _, p_slope, _ = ols(xs, p_logs)
    p_intercept = sum(p_logs) / len(p_logs) - p_slope * (sum(xs) / len(xs))
    players_pred = math.exp(p_intercept + p_slope * x_target)
    players_pred = min(max(players_pred, float(players_now)),
                       players_now * TREND_MAX_PLAYERS_MULT)
    med_logits = []
    spreads = []
    for i in range(n_trophies):
        txs, tys = fits[i]
        n = len(txs)
        if n >= 2:
            a, b, resid = ols(txs, tys)
        elif n == 1:
            a, b, resid = tys[0] - med_slope * txs[0], med_slope, med_resid
        else:
            a, b, resid = logit(last_rates[i]), med_slope, med_resid
        b_shrunk = (n * b + TREND_SHRINK_PRIOR * med_slope) / (n + TREND_SHRINK_PRIOR)
        a_adj = (sum(tys) / n - b_shrunk * (sum(txs) / n)) if n else a
        lg = a_adj + b_shrunk * x_target
        gap = 0.0
        # Counts-based rate for this trophy.
        cxs, cys = [], []
        for pos, k in enumerate(fit_idx):
            rate = history[k]["rates"][i]
            if rate is None:
                continue
            count = round(rate * history[k]["players_tracked"] / 100.0)
            cxs.append(xs[pos])
            cys.append(math.log(max(count, 0) + 0.5))
        if len(cxs) >= 2:
            ca, cb, _ = ols(cxs, cys)
            cb_shrunk = (len(cxs) * cb + TREND_SHRINK_PRIOR * med_slope) \
                / (len(cxs) + TREND_SHRINK_PRIOR)
            ca_adj = sum(cys) / len(cys) - cb_shrunk * (sum(cxs) / len(cxs))
            count_pred = math.exp(ca_adj + cb_shrunk * x_target) - 0.5
            count_pred = max(count_pred, float(last_counts[i]))
            lg_counts = logit(100.0 * min(count_pred, players_pred) / players_pred)
            if lg_counts < lg:
                gap = lg - lg_counts
                lg = lg - 0.5 * gap
        var = resid * resid + TREND_BASE_VAR \
            + TREND_HORIZON_VAR_PER_DAY * horizon_days + 0.09 * gap * gap
        med_logits.append(lg)
        spreads.append(min(TREND_MAX_SD, max(TREND_MIN_SD, math.sqrt(var))))
    # Platinum needs every other trophy: cap it at the least-held other.
    med_rates = [inv_logit(lg) for lg in med_logits]
    plat_idx = None
    for i, trophy in enumerate(case["trophies"]):
        if str(trophy.get("tier", "")).lower() == "platinum":
            plat_idx = i
            break
    if plat_idx is not None and n_trophies > 1:
        cap = min(med_rates[i] for i in range(n_trophies) if i != plat_idx)
        if med_rates[plat_idx] > cap:
            med_rates[plat_idx] = cap
            med_logits[plat_idx] = logit(cap)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        lg = med_logits[i]
        sd = spreads[i]
        qs = [round4(inv_logit(lg + z * sd)) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def last_counts_and_rates(case):
    """Latest non-null rate per trophy and its earner count."""
    history = case["history"]
    last = history[-1]
    players_now = last["players_tracked"]
    n = len(case["trophies"])
    rates = [None] * n
    counts = [0] * n
    for read in reversed(history):
        for i in range(n):
            if rates[i] is None and read["rates"][i] is not None:
                rates[i] = read["rates"][i]
                counts[i] = round(read["rates"][i] * read["players_tracked"] / 100.0)
        if all(r is not None for r in rates):
            break
    for i in range(n):
        if rates[i] is None:
            rates[i] = 0.0
            counts[i] = 0
    return rates, counts, players_now


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    rates, counts, players_now = last_counts_and_rates(case)
    players_seen = {read["players_tracked"] for read in case["history"]}
    flat = len(players_seen) <= 1
    if flat:
        return flat_forecast(case, rates, counts, players_now)
    return trend_forecast(case, case["history"], rates, counts, players_now)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
