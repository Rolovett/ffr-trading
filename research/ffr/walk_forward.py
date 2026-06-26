"""
research/ffr/walk_forward.py

THE GATE. Everything before this was in-sample and selected across many cells,
so it cannot be trusted on its own. This validates the trend-aligned
continuation edge out-of-sample:

  - Rolling folds: each fold tunes (k, target) on a TRAIN window using only past
    data, then trades the next UNSEEN test window with those params. No leakage:
    train entries are strictly before test entries.
  - Selection metric is RISK-ADJUSTED (per-trade mean/std), not raw E[R], so the
    optimiser won't chase the fat-tailed high-target configs that wreck drawdown.
  - Aggregates all out-of-sample trades into one honest ledger and reports:
    net E[R] with bootstrap 95% CI, win rate, N, max drawdown (in R) and longest
    losing streak — the risk numbers that decide "low-risk and profitable".
  - Also runs a FIXED-PARAM baseline (k=3, target=2R) over the same OOS windows.
    If a sensible fixed setting matches the optimiser, ship the fixed one — it is
    the more robust choice and has zero optimisation risk.

Reuses helpers from sweep_continuation / geometry_test. Trend-aligned only
(short in a daily downtrend, long in a daily uptrend).
"""

import argparse
import numpy as np
import pandas as pd
from cascade_detector import wilson
from geometry_test import build_frame, triggers
from sweep_continuation import barrier, boot_ci

DAY = 86_400_000
M = 4.0
HOLD = 192
MIN_STOP = 0.0015
COOLDOWN = 16
COST = 0.0010
GRID_K = [3.0, 3.5, 4.0]
GRID_T = [1.5, 2.0, 2.5, 3.0]
MIN_TRAIN_N = 20


def gen(df, k, target):
    """All trend-aligned continuation trades for (k, target), with entry time."""
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
        rows.append((int(ts[i]), rmult - (COST * entry) / risk, outcome == "win"))
    return pd.DataFrame(rows, columns=["t", "net", "win"])


def risk_score(net):
    net = np.asarray(net, dtype=float)
    if len(net) < MIN_TRAIN_N or net.std() == 0:
        return -np.inf
    return net.mean() / net.std()


def drawdown_and_streak(net_series):
    eq = np.cumsum(net_series)
    peak = np.maximum.accumulate(eq)
    max_dd = float((peak - eq).max()) if len(eq) else 0.0
    streak = worst = 0
    for x in net_series:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    return max_dd, worst


def report(name, net):
    net = np.asarray(net, dtype=float)
    n = len(net)
    if n == 0:
        print(f"{name}: no OOS trades")
        return
    wins = int((net > 0).sum())
    er = net.mean()
    lo, hi = boot_ci(net)
    dd, streak = drawdown_and_streak(net)
    tag = "  +EV (OOS)" if lo > 0 else ("  -EV (OOS)" if hi < 0 else "  (CI includes 0)")
    print(f"{name}:")
    print(f"  N={n}  WR={wins / n:.1%}  E[R]={er:+.3f} [95% {lo:+.3f},{hi:+.3f}]{tag}")
    print(f"  total={net.sum():+.1f}R  maxDD={dd:.1f}R  longest_loss_streak={streak}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_days", type=int, default=365)
    p.add_argument("--test_days", type=int, default=90)
    args = p.parse_args()

    df = build_frame(96, 96)
    cache = {(k, t): gen(df, k, t) for k in GRID_K for t in GRID_T}
    fixed = cache[(3.0, 2.0)]

    t0, t1 = int(df["t"].min()), int(df["t"].max())
    oos_opt, oos_fix = [], []
    print("=" * 80)
    print(f"WALK-FORWARD | train={args.train_days}d test={args.test_days}d "
          f"| select on risk-adjusted mean/std | grid k{GRID_K} x t{GRID_T}")
    print("-" * 80)
    print(f"{'train window':<26} {'pick':<14} {'train mean/σ':>12} {'test N':>7} {'test E[R]':>10}")

    s = t0
    fold = 0
    while s + (args.train_days + args.test_days) * DAY <= t1 + DAY:
        tr_lo, tr_hi = s, s + args.train_days * DAY
        te_lo, te_hi = tr_hi, tr_hi + args.test_days * DAY

        best, best_sc = None, -np.inf
        for key, ev in cache.items():
            tr = ev[(ev.t >= tr_lo) & (ev.t < tr_hi)]
            sc = risk_score(tr.net.values)
            if sc > best_sc:
                best_sc, best = sc, key

        if best is not None:
            te = cache[best][(cache[best].t >= te_lo) & (cache[best].t < te_hi)]
            oos_opt.extend(te.net.tolist())
            te_er = te.net.mean() if len(te) else float("nan")
            d0 = pd.to_datetime(tr_lo, unit="ms").date()
            d1 = pd.to_datetime(tr_hi, unit="ms").date()
            print(f"{str(d0)}..{str(d1):<13} k={best[0]},t={best[1]:<5} "
                  f"{best_sc:>12.3f} {len(te):>7} {te_er:>+10.3f}")

        tf = fixed[(fixed.t >= te_lo) & (fixed.t < te_hi)]
        oos_fix.extend(tf.net.tolist())
        s += args.test_days * DAY
        fold += 1

    print("-" * 80)
    print(f"Folds: {fold}")
    print()
    report("OUT-OF-SAMPLE (walk-forward optimised)", oos_opt)
    print()
    report("OUT-OF-SAMPLE (fixed k=3, target=2R)", oos_fix)
    print("=" * 80)


if __name__ == "__main__":
    main()
