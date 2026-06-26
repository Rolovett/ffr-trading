"""
research/ffr/slippage_stress.py

Walk-forward passed, but the fixed-param OOS lower CI was only ~+0.03R, and
continuation enters at the climax-bar close — exactly where live slippage is
worst. This stresses the edge against rising round-trip friction (taker + entry
+ exit slippage, lumped as a fraction of price) to find where the margin dies.

Same machinery as walk_forward.py (rolling folds, risk-adjusted selection,
trend-aligned continuation). For each friction level it reports OOS E[R] with
bootstrap CI and max drawdown for the optimised and fixed configs. The friction
at which the lower CI crosses zero is the point the edge stops being real — and
tells us how good our live execution has to be.
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
GRID_K = [3.0, 3.5, 4.0]
GRID_T = [1.5, 2.0, 2.5, 3.0]
MIN_TRAIN_N = 20


def gen(df, k, target, cost):
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    trend, ts = df["trend"].values, df["t"].values
    n = len(df)
    rows = []
    for i in triggers(df, k, M, COOLDOWN):
        down = C[i] < O[i]
        aligned = (down and trend[i] == -1) or ((not down) and trend[i] == 1)
        if not aligned:
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
        outcome, rmult = res
        rows.append((int(ts[i]), rmult - (cost * entry) / risk))
    return pd.DataFrame(rows, columns=["t", "net"])


def risk_score(net):
    net = np.asarray(net, dtype=float)
    if len(net) < MIN_TRAIN_N or net.std() == 0:
        return -np.inf
    return net.mean() / net.std()


def max_dd(net):
    eq = np.cumsum(net)
    if len(eq) == 0:
        return 0.0
    return float((np.maximum.accumulate(eq) - eq).max())


def run(df, cost, train_days=365, test_days=90):
    cache = {(k, t): gen(df, k, t, cost) for k in GRID_K for t in GRID_T}
    fixed = cache[(3.0, 2.0)]
    t0, t1 = int(df["t"].min()), int(df["t"].max())
    oos_opt, oos_fix = [], []
    s = t0
    while s + (train_days + test_days) * DAY <= t1 + DAY:
        tr_lo, tr_hi = s, s + train_days * DAY
        te_lo, te_hi = tr_hi, tr_hi + test_days * DAY
        best, best_sc = None, -np.inf
        for key, ev in cache.items():
            tr = ev[(ev.t >= tr_lo) & (ev.t < tr_hi)]
            sc = risk_score(tr.net.values)
            if sc > best_sc:
                best_sc, best = sc, key
        if best is not None:
            te = cache[best][(cache[best].t >= te_lo) & (cache[best].t < te_hi)]
            oos_opt.extend(te.net.tolist())
        tf = fixed[(fixed.t >= te_lo) & (fixed.t < te_hi)]
        oos_fix.extend(tf.net.tolist())
        s += test_days * DAY
    return np.asarray(oos_opt), np.asarray(oos_fix)


def line(label, net):
    n = len(net)
    if n == 0:
        print(f"    {label:<12} no trades")
        return
    er = net.mean()
    lo, hi = boot_ci(net)
    flag = "+EV" if lo > 0 else ("-EV" if hi < 0 else "~0 ")
    print(f"    {label:<12} N={n:<4} E[R]={er:+.3f} [95% {lo:+.3f},{hi:+.3f}] {flag} "
          f"| total={net.sum():+5.1f}R maxDD={max_dd(net):4.1f}R")


def main():
    df = build_frame(96, 96)
    print("=" * 78)
    print("SLIPPAGE STRESS — OOS edge vs round-trip friction (taker + slippage)")
    print("baseline backtest used 0.10%. find where the lower CI crosses zero.")
    print("=" * 78)
    for cost in [0.0010, 0.0015, 0.0020, 0.0025, 0.0030]:
        opt, fix = run(df, cost)
        print(f"friction = {cost:.2%} round-trip:")
        line("optimised", opt)
        line("fixed 3/2R", fix)
    print("=" * 78)


if __name__ == "__main__":
    main()
