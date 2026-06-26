"""
research/ffr/limit_entry_test.py

Market-at-close entries failed the slippage stress: +EV only survived at 0.10%
friction (bare taker, no slip), which is unrealistic when you market-buy right
after a climax bar. This tests the pre-committed fix — LIMIT entries on a shallow
pullback — head-to-head against the market baseline on the SAME out-of-sample
window (last 365 days, where the walk-forward tested).

Why limit entries might rescue it:
  - maker fee (~0.02%/side) instead of taker (~0.05%/side): ~0.06% rt saved
  - better fill price (buy the pullback, not the spike) with zero entry slippage
Why they might not — the thing this measures:
  - adverse selection: trades that never pull back don't fill, and those may be
    the strongest runners. Fill rate is reported so we see how much we skim off.

Limit placed a fraction f of the way from the climax close back toward the
climax extreme. Fill if a later bar (within W) trades through it; barrier runs
from the fill bar inclusive (pessimistic — a fill bar that also tags the stop is
a loss). Trend-aligned continuation only.
"""

import numpy as np
import pandas as pd
from geometry_test import build_frame, triggers
from sweep_continuation import barrier, boot_ci

DAY = 86_400_000
M = 4.0
HOLD = 192
MIN_STOP = 0.0015
COOLDOWN = 16
FILL_WINDOW = 8          # bars to wait for the pullback fill (2h)
TAKER = 0.0015           # realistic market round-trip (taker + slippage)
MAKER = 0.0005           # realistic limit round-trip (maker, ~no slippage)


def _aligned(down, tr):
    return (down and tr == -1) or ((not down) and tr == 1)


def market_events(df, k, target, cost, oos_lo):
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    trend, ts = df["trend"].values, df["t"].values
    n = len(df)
    out = []
    for i in triggers(df, k, M, COOLDOWN):
        if ts[i] < oos_lo:
            continue
        down = C[i] < O[i]
        if not _aligned(down, trend[i]):
            continue
        if down:
            entry, stop, long = C[i], H[i], False
            risk = stop - entry
            tgt = entry - target * risk
        else:
            entry, stop, long = C[i], L[i], True
            risk = entry - stop
            tgt = entry + target * risk
        if risk <= 0 or risk / entry < MIN_STOP:
            continue
        res = barrier(long, entry, stop, tgt, H, L, C, i + 1, min(n, i + 1 + HOLD))
        if res is None:
            continue
        out.append(res[1] - (cost * entry) / risk)
    return np.asarray(out), len(out), len(out)


def limit_events(df, k, target, f, cost, oos_lo):
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    trend, ts = df["trend"].values, df["t"].values
    n = len(df)
    out = []
    eligible = filled = 0
    for i in triggers(df, k, M, COOLDOWN):
        if ts[i] < oos_lo:
            continue
        down = C[i] < O[i]
        if not _aligned(down, trend[i]):
            continue
        eligible += 1
        if down:                                  # short: limit above close, toward high
            stop = H[i]
            limit = C[i] + f * (H[i] - C[i])
            long = False
            fill = next((j for j in range(i + 1, min(n, i + 1 + FILL_WINDOW))
                         if H[j] >= limit), None)
        else:                                      # long: limit below close, toward low
            stop = L[i]
            limit = C[i] - f * (C[i] - L[i])
            long = True
            fill = next((j for j in range(i + 1, min(n, i + 1 + FILL_WINDOW))
                         if L[j] <= limit), None)
        if fill is None:
            continue
        entry = limit
        risk = (stop - entry) if not long else (entry - stop)
        if risk <= 0 or risk / entry < MIN_STOP:
            continue
        tgt = (entry - target * risk) if not long else (entry + target * risk)
        res = barrier(long, entry, stop, tgt, H, L, C, fill, min(n, fill + HOLD))
        if res is None:
            continue
        filled += 1
        out.append(res[1] - (cost * entry) / risk)
    return np.asarray(out), eligible, filled


def line(label, net, eligible, filled):
    n = len(net)
    if n == 0:
        print(f"  {label:<30} no fills")
        return
    er = net.mean()
    lo, hi = boot_ci(net)
    flag = "+EV" if lo > 0 else ("-EV" if hi < 0 else "~0 ")
    fillpct = f"{filled}/{eligible}={filled/eligible:.0%}" if eligible else "-"
    eq = np.cumsum(net)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    print(f"  {label:<30} fill={fillpct:<11} N={n:<4} E[R]={er:+.3f} "
          f"[95% {lo:+.3f},{hi:+.3f}] {flag} | total={net.sum():+5.1f}R maxDD={dd:4.1f}R")


def main():
    df = build_frame(96, 96)
    oos_lo = int(df["t"].max()) - 365 * DAY
    print("=" * 88)
    print(f"LIMIT vs MARKET ENTRY | OOS=last 365d | target=2R | "
          f"taker={TAKER:.2%} maker={MAKER:.2%}")
    print("Does a maker pullback entry beat the slippage that killed market entries?")
    print("=" * 88)
    for k in [3.0, 3.5]:
        print(f"k={k}:")
        net, el, fl = market_events(df, k, 2.0, TAKER, oos_lo)
        line("market @ taker 0.15%", net, el, fl)
        for f in [0.15, 0.25, 0.40]:
            net, el, fl = limit_events(df, k, 2.0, f, MAKER, oos_lo)
            line(f"limit pullback f={f} @ maker", net, el, fl)
    print("=" * 88)


if __name__ == "__main__":
    main()
