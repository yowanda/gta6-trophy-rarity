# SPDX-License-Identifier: MIT
"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01), round 2.

Reads one case per run: the house's Exophase reads of the GTA VI PS5 trophy
page up to the case checkpoint, and forecasts each trophy's earn rate at the
case target as five quantiles (5/25/50/75/95%), scored by pinball loss on the
logit scale.

Method: decompose every rate into its two counts, and model the counts.

  rate_i(t) = earners_i(t) / players_tracked(t)

Exophase refreshes its counts in jumps hours apart: between refreshes the
page is frozen (identical reads), at a refresh the tracked-player count
jumps (the first refresh took it 638 -> 3,868) and every rate falls, not
because players stopped earning trophies but because the denominator grew.
So a rate is not one curve; it is a ratio of two growing counts, and the
forecast has to get both counts right:

1.  Collapse the history to its distinct page states.  Consecutive
    identical reads are one state; a frozen stretch of 30 reads carries
    the information of one.  Fitting raw reads would let the frozen
    stretches vote 30 times for "nothing ever changes".

2.  Players tracked.  On the distinct states, fit log P against
    x = ln(1 + hours since 2026-11-19T00:00Z) by least squares and read
    the line at the target (damped over long horizons).  The log-time
    axis makes the explosive launch growth concave, so the line does not
    run away over four more days.

3.  Earners per trophy.  On the distinct states, fit log E_i against
    the same x for each trophy, shrink each slope toward the pooled
    (median) slope with weight n / (n + 3), damp the extrapolation, and
    cap earners at the players forecast (a rate cannot pass 100%).
    Early trophies keep their own steep slopes; late trophies, whose
    earners barely moved in the history, stay nearly flat and their
    rates fall as the denominator grows -- which is what the page shows.

4.  Frozen history (no refresh ever seen: every read identical, as on
    launch night).  There is nothing to fit, so fall back to a refresh
    prior, stated openly and used for no other case: the page has
    already been frozen for hours, so a refresh is overdue -- expect
    about one refresh per 4.5 hours of frozen time plus horizon, each
    multiplying players tracked by ~6x (the first refresh's 3,868/638),
    while earners grow only modestly (median ~1.25x per refresh, tilted
    toward trophies that already had more earners: early trophies keep
    being earned fastest).  The spread stays wide, because a refresh
    that has not been seen can only be guessed at.

5.  Honest spread.  The logit spread combines the earners fit's
    residual and slope uncertainty, the players fit's uncertainty, and
    a refresh/staleness term that grows with the horizon (the answer is
    the page as of Exophase's last refresh before the target, which may
    be hours stale).  Quantiles are q_tau =
    inv_logit(logit(median) + z_tau * sigma), z = (-1.645, -0.674, 0,
    0.674, 1.645), rounded to 4 decimals.

No number in this program is a practice answer: the refresh prior is a
distribution over jump factors, every trophy's forecast is computed from
that case's own reads, and the same code runs the 90-minute practice
cases and the 4.5-day final.  Stdlib only, deterministic, no network.

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

# Release-day start: the origin of the log-time axis.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- fit path (history holds at least one refresh) ---
SHRINK_PRIOR = 3.0      # effective states weighting a trophy's own slope
DAMP_PER_DAY = 0.10     # damping of extrapolated moves, per horizon day
MAX_LOGP_MOVE = 3.0     # cap on log-players move past the last state
MAX_LOGE_MOVE = 3.0     # cap on log-earners move past the last state
RESID_FLOOR = 0.20      # floor on a trophy's residual spread (logit)
SIGMA_FLOOR = 0.35      # floor on the total logit spread
SIGMA_CAP = 1.50        # cap on the total logit spread
STALE_SIGMA = 0.30      # refresh-staleness spread per sqrt(horizon day)

# --- frozen-history prior path (no refresh ever seen in the case) ---
REFRESH_GAP_H = 4.5     # typical hours between Exophase refreshes
JUMP_MED = 6.0          # median players-tracked jump factor per refresh
JUMP_MAX = 25.0         # cap on the prior jump factor
E_GROWTH_MED = 1.25     # median earner growth per refresh
E_TILT_GAMMA = 0.5      # earner growth tilt by (earners / median earners)
SIGMA_P_PRIOR = 0.35    # log-players spread of the unseen refresh
SIGMA_E_PRIOR = 0.25    # log-earners spread of the unseen refresh
# Calibrated on the practice reconstruction (see work notes): the
# prior-path medians land with a logit error sd of ~0.40, so a total
# spread near 0.43 puts the 90% coverage at its nominal 0.9 instead of
# the over-wide 1.0 a 0.71 spread gave.


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


def round4(x):
    """x rounded to 4 decimals, half up (as the starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(at):
    hours = (at - T0).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def ols(xs, ys):
    """Plain least squares y ~ a + b*x. Returns (a, b, resid_sd, n, xbar, sxx)."""
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, n, mx, sxx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n >= 3:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b, resid, n, mx, sxx)


def median(vals):
    s = sorted(vals)
    m = len(s)
    if m == 0:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def distinct_states(case):
    """Collapse history to distinct page states: (time, players, earners[]).

    Consecutive reads with identical players-tracked and rates are one
    state; the state's time is its first read.  Returns the states and
    the checkpoint time.
    """
    checkpoint = parse_time(case["at"])
    states = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if players is None:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        earners = []
        for r in rates:
            earners.append(None if r is None else float(r) * float(players) / 100.0)
        states.append((parse_time(read.get("at") or read.get("slot")), float(players), earners))
    return states, checkpoint


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    horizon_h = case.get("horizon_minutes", 0) / 60.0
    target = parse_time(case["target"])
    x_target = log_time(target)
    states, checkpoint = distinct_states(case)
    if not states:
        raise ValueError("case has no usable history")
    damp = 1.0 / (1.0 + DAMP_PER_DAY * horizon_days)
    stale = STALE_SIGMA * math.sqrt(horizon_days)

    last_t, last_p, last_e = states[-1]
    x_last = log_time(last_t)
    frozen_h = max(0.0, (checkpoint - states[0][0]).total_seconds() / 3600.0)
    distinct_p = len({s[1] for s in states})

    # ---------- players tracked ----------
    if distinct_p >= 2 and len(states) >= 3:
        xs = [log_time(s[0]) for s in states]
        ys = [math.log(s[1]) for s in states]
        fit = ols(xs, ys)
        _, b_p, resid_p, _, xbar, sxx = fit
        move = b_p * (x_target - x_last) * damp
        move = max(-MAX_LOGP_MOVE, min(MAX_LOGP_MOVE, move))
        log_p_hat = math.log(last_p) + move
        var_p = resid_p ** 2
        if sxx > 1e-12 and len(xs) >= 3:
            var_p += (resid_p ** 2 / sxx) * (x_target - xbar) ** 2
        sigma_p = math.sqrt(max(0.0225, var_p))
    else:
        # Frozen history: a refresh is overdue.  Expected refreshes over
        # the frozen stretch plus the horizon, each multiplying players
        # tracked by ~JUMP_MED.
        n_ref = (frozen_h + horizon_h) / REFRESH_GAP_H
        jump = min(JUMP_MAX, JUMP_MED ** max(1.0, n_ref)) if n_ref > 0 else 1.0
        log_p_hat = math.log(last_p) + math.log(jump)
        sigma_p = SIGMA_P_PRIOR

    # ---------- earners per trophy ----------
    # First pass: per-trophy log-earner slopes, for the pooled slope.
    slopes = []
    per_tr = []
    for idx in range(n_tr):
        pts = []
        for t, _p, earn in states:
            if idx < len(earn) and earn[idx] is not None:
                pts.append((log_time(t), math.log(max(0.5, earn[idx]))))
        per_tr.append(pts)
        if len(pts) >= 3 and len({round(x, 9) for x, _ in pts}) >= 2:
            f = ols([x for x, _ in pts], [y for _, y in pts])
            if f is not None and f[1] != 0.0:
                slopes.append(f[1])
    pooled_slope = median(slopes) if slopes else 0.0

    e_med = median([max(0.5, e) for e in last_e if e is not None]) if last_e else 1.0

    rows = []
    for idx, trophy in enumerate(trophies):
        pts = per_tr[idx]
        e_last = last_e[idx] if idx < len(last_e) and last_e[idx] is not None else None
        if e_last is None:
            e_last = 0.5
        e_last = max(0.5, float(e_last))

        # A trophy every tracked player holds: it stays (near-)universal.
        if idx < len(last_e) and last_e[idx] is not None and last_p > 0 \
                and abs(last_e[idx] - last_p) < 0.5 and last_p >= 100:
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 100, 100, 100]})
            continue

        if len(pts) >= 3 and len({round(x, 9) for x, _ in pts}) >= 2:
            f = ols([x for x, _ in pts], [y for _, y in pts])
            _, b_i, resid_i, n_i, _, sxx_i = f
            lam = n_i / (n_i + SHRINK_PRIOR)
            b = lam * b_i + (1.0 - lam) * pooled_slope
            move = b * (x_target - x_last) * damp
            move = max(-MAX_LOGE_MOVE, min(MAX_LOGE_MOVE, move))
            log_e_hat = math.log(e_last) + move
            var_e = max(RESID_FLOOR, resid_i) ** 2
            if sxx_i > 1e-12:
                var_e += (max(RESID_FLOOR, resid_i) ** 2 / sxx_i) * (x_target - x_last) ** 2
            sigma_e = math.sqrt(var_e)
        elif len(pts) >= 2:
            move = pooled_slope * (x_target - x_last) * damp
            move = max(-MAX_LOGE_MOVE, min(MAX_LOGE_MOVE, move))
            log_e_hat = math.log(e_last) + move
            sigma_e = 0.45
        else:
            # Frozen history: earners grow only modestly over the unseen
            # refresh, tilted toward trophies that already had more.
            n_ref = (frozen_h + horizon_h) / REFRESH_GAP_H
            n_ref = max(1.0, n_ref)
            tilt = (e_last / e_med) ** E_TILT_GAMMA if e_med > 0 else 1.0
            growth = (E_GROWTH_MED * tilt) ** n_ref
            log_e_hat = math.log(e_last) + math.log(growth)
            sigma_e = SIGMA_E_PRIOR

        # Earners cannot exceed the players tracked.
        log_e_hat = min(log_e_hat, log_p_hat)
        med_rate = 100.0 * math.exp(log_e_hat - log_p_hat)
        med_rate = min(99.99, max(0.01, med_rate))
        sigma = math.sqrt(sigma_e ** 2 + sigma_p ** 2 + stale ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        med_logit = logit(med_rate)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(med_logit + z * sigma))
            q = min(99.99, max(0.01, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
