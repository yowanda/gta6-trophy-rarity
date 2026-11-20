"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per run: the house's Exophase reads of the GTA VI PS5 trophy
page up to the case checkpoint, and forecasts each trophy's earn rate at the
case target as five quantiles (5/25/50/75/95%), scored by pinball loss on the
logit scale.

Method (per trophy, all in logit space, stdlib only, deterministic):

1.  Each read gives a rate r (percent of tracked players holding the trophy).
    Work with y = logit(r) = ln(p / (1 - p)), p = r / 100, r clamped to
    [0.01, 99.99].  The logit scale is the scored scale, so fitting there is
    fitting what matters: a move from 1% to 2% counts like 33% to 50%.

2.  Time is x = ln(1 + hours since 2026-11-19T00:00Z, the release day).
    Trophy curves are concave in clock time -- early trophies are earned
    fast, then the curve flattens as the remaining players are the slow
    ones -- and log-time linearises that shape, so a straight line in
    (x, y) is a saturating curve in clock time.  This is the same family
    the house curve baseline uses, but fitted better (see below).

3.  Weighted least squares of y on x over the whole history, with recency
    weights w = 0.5 ** (age_hours / 12): Exophase refreshes its counts in
    jumps hours apart and the player mix changes as launch week goes on,
    so yesterday's reads describe tomorrow worse than this morning's.
    A flat history (launch night, before the first refresh) gives a flat
    line, exactly as it should -- there is no signal to extrapolate.

4.  Shrinkage.  A single trophy's slope is noisy, so each slope is shrunk
    toward the pooled slope (the median slope across trophies with a
    usable fit), with weight n_eff / (n_eff + 3) on its own slope, where
    n_eff is the effective number of distinct reads.  Trophies with flat
    or sparse histories therefore borrow the field's typical growth
    instead of extrapolating noise.

5.  Damped extrapolation.  The line is read at the target, but only a
    damped share of the move past the last read is kept:
    damp = 1 / (1 + 0.12 * horizon_days).  Over a 90-minute practice
    horizon damp is ~0.99 (no effect); over the final's 4.5-day horizon
    it is ~0.65, guarding against a short launch-window slope running
    away over four more days.  The median's logit move from the last
    read is also capped at +/-3.0 (about a 20x change in odds), and the
    median itself is clamped to [0.01, 99.99].  The platinum (trophy 1)
    can therefore rise, but never to a fantasy number.

6.  Honest spread.  The logit spread sigma combines three terms in
    quadrature:
      * the house random-walk term, 0.5 * sqrt(horizon_days);
      * the fit's weighted residual standard deviation (floored at 0.15,
        capped at 1.5) -- trophies whose history wobbled get wider bands;
      * a refresh term, 0.45 * sqrt(horizon_days) -- Exophase refreshes
        in jumps and the tracked-player base keeps growing, which dilutes
        every rate (the first refresh dropped almost every rate even
        though players only gain trophies).  A trophy whose history never
        moved at all gets an extra 0.4: a flat past is no evidence the
        future stays flat, as the practice cases prove.
    Quantiles are q_tau = inv_logit(logit(median) + z_tau * sigma) with
    z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals.

Why this should beat the house baselines: persistence says nothing
moves, and the house curve baseline fits the same line shape but only
over the last 24 hours, with no shrinkage, no damping and a narrower
spread.  Over a 4.5-day horizon the differences that matter are the
median's direction (recency-weighted, shrunk, damped) and a spread wide
enough to cover Exophase's refresh jumps.

Usage (grader): `python3 -P -s /bot/main.py`, reads every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out per case to
$MAKEFUN_OUTPUT_DIR (else /output).  Locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day start: the origin of the log-time axis.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Recency half-life for read weights, in hours.
HALF_LIFE_H = 12.0
# Shrinkage prior strength (effective reads) for the pooled slope.
SHRINK_PRIOR = 3.0
# Damping of the extrapolated move, per horizon day.
DAMP_PER_DAY = 0.12
# Cap on the median's logit move away from the last read.
MAX_LOGIT_MOVE = 3.0
# Random-walk spread per sqrt(day), as in the house baseline.
RW_SIGMA = 0.5
# Extra spread per sqrt(day) for Exophase refresh jumps / dilution.
REFRESH_SIGMA = 0.45
# Floor / cap on the per-trophy residual spread.
RESID_FLOOR = 0.15
RESID_CAP = 1.5
# Extra spread (logit) when a trophy's history never moved.
FLAT_EXTRA = 0.4


def clamp_rate(r):
    return min(99.99, max(0.01, float(r)))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(at):
    hours = (at - T0).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, n_eff, resid_sd)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, sw, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    # Weighted residual sd (needs >= 3 points to mean anything).
    if len(xs) >= 3 and sw > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
        var /= sw * (len(xs) - 2) / len(xs)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b, sw, resid)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_time(target)

    # Per-trophy series: (x, y) pairs and the last observed rate.
    series = []
    for idx in range(len(trophies)):
        xs, ys, ws = [], [], []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            last_rate = float(rates[idx])
            at = parse_time(read.get("at") or read.get("slot"))
            age_h = max(0.0, (checkpoint - at).total_seconds() / 3600.0)
            xs.append(log_time(at))
            ys.append(logit(rates[idx]))
            ws.append(0.5 ** (age_h / HALF_LIFE_H))
        series.append((xs, ys, ws, last_rate))

    # First pass: fit every trophy, collect usable slopes for pooling.
    fits = []
    slopes = []
    for xs, ys, ws, _ in series:
        if len(xs) >= 2 and len(set(round(x, 6) for x in xs)) >= 2:
            fit = wls_line(xs, ys, ws)
            fits.append(fit)
            slopes.append(fit[1])
        else:
            fits.append(None)
    slopes.sort()
    pooled_slope = slopes[len(slopes) // 2] if slopes else 0.0

    damp = 1.0 / (1.0 + DAMP_PER_DAY * horizon_days)
    rw = RW_SIGMA * math.sqrt(horizon_days)
    refresh = REFRESH_SIGMA * math.sqrt(horizon_days)

    rows = []
    for idx, trophy in enumerate(trophies):
        xs, ys, ws, last_rate = series[idx]
        fit = fits[idx]
        if last_rate is None:
            # Trophy never observed: fall back to a low flat rate.
            last_rate = 0.5
        last_logit = logit(last_rate)

        if fit is None or len(xs) < 3:
            # Too little to fit: persistence median, pooled drift only
            # when there is at least a two-point slope to borrow.
            if fit is not None:
                b = pooled_slope
                x_last = xs[-1]
                move = b * (x_target - x_last) * damp
                move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
                med_logit = last_logit + move
            else:
                med_logit = last_logit
            resid = RESID_FLOOR
            flat = len(set(ys)) <= 1
        else:
            a, b, n_eff, resid = fit
            lam = n_eff / (n_eff + SHRINK_PRIOR)
            b_shrunk = lam * b + (1.0 - lam) * pooled_slope
            x_last = xs[-1]
            move = b_shrunk * (x_target - x_last) * damp
            move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
            med_logit = last_logit + move
            flat = len(set(ys)) <= 1

        resid_c = min(RESID_CAP, max(RESID_FLOOR, resid))
        var = rw * rw + resid_c * resid_c + refresh * refresh
        if flat:
            var += FLAT_EXTRA * FLAT_EXTRA
        sigma = math.sqrt(var)

        med = inv_logit(med_logit)
        med = min(99.99, max(0.01, med))
        med_logit = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            q = round(min(99.99, max(0.01, q)), 4)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.in"))
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        dest = out_dir / (path.stem + ".out")
        dest.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
