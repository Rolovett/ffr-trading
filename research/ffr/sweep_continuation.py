"""
research/ffr/sweep_continuation.py

Continuation beat fade decisively: BTC perp cascades carry momentum, not
reversion. The original 'liquidation reversal' thesis is rejected by the data;
the live edge is volatility-breakout CONTINUATION off the climax bar.

This sweeps the target multiple and a coarse k grid for the continuation
geometry to locate the ROBUST expectancy plateau — the configuration that
maximises net E[R] per trade across a *range* of settings, not a single lucky
cell. Net E[R] is the metric that compounds over 100 trades, so it's the
headline; WR and N are reported alongside.

Honesty guards: no lookahead, pessimistic same-bar tie -> stop, min-stop
economic filter, cooldown spacing. Timeouts are marked-to-market at the exit
bar's close (not assumed flat), so expectancy isn't flattered by open trades.
A chosen config is only believed after the walk-forward step that follows.
"""

import argparse
import numpy as np
import pandas as pd
from cascade_detector import wilson
from geometry_test import build_frame, triggers


def barrier(long, entry, stop, target, H, L, C, start, end):
    risk = (entry - stop) if long else (stop - entry)
    if risk <= 0:
        return None
    for j in range(start, end):
        if long:
            if L[j] <= stop:
                return ("loss", -1.0)
            if H[j] >= target:
                return ("win", (target - entry) / risk)
        else:
            if H[j] >= stop:
                return ("loss", -1.0)
            if L[j] <= target:
                return ("win", (entry - target) / risk)
    last = C[end - 1]                       # timeout: mark to market
    mtm = (last - entry) / risk if long else (entry - last) / risk
    return ("timeout", mtm)


def continuation_events(df, k, m, target_R, hold, min_stop_pct, cooldown, cost):
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    trend = df["trend"].values
    n = len(df)
    rows = []
    for i in triggers(df, k, m, cooldown):
        down = C[i] < O[i]
        if down:                            # down-spike -> short continuation
            entry, stop, long = C[i], H[i], False
            risk = stop - entry
            tgt = entry - target_R * risk
        else:                               # up-spike -> long continuation
            entry, stop, long = C[i], L[i], True
            risk = entry - stop
            tgt = entry + target_R * risk
        if risk <= 0 or risk / entry < min_stop_pct:
            continue
        res = barrier(long, entry, stop, tgt, H, L, C, i + 1, min(n, i + 1 + hold))
        if res is None:
            continue
        outcome, rmult = res
        cost_R = (cost * entry) / risk
        net = rmult - cost_R
        aligned = (down and trend[i] == -1) or ((not down) and trend[i] == 1)
        rows.append(dict(side=("short" if down else "long"),
                         aligned=bool(aligned), outcome=outcome,
                         win=(outcome == "win"), net=net))
    return pd.DataFrame(rows)


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stat(ev):
    n = len(ev)
    if n == 0:
        return None
    wins = int(ev.win.sum())
    losses = int((ev.outcome == "loss").sum())
    to = int((ev.outcome == "timeout").sum())
    resolved = wins + losses
    wr = wins / resolved if resolved else float("nan")
    er = ev.net.mean()
    lo, hi = boot_ci(ev.net.values)
    return dict(N=n, WR=wr, TO=to, ER=er, lo=lo, hi=hi)


def line(label, s):
    if s is None:
        print(f"  {label:<26} N=0")
        return
    pos = "  +EV" if s["lo"] > 0 else ""
    print(f"  {label:<26} N={s['N']:<4} WR={s['WR']:5.1%} TO={s['TO']:<3} "
          f"| E[R]={s['ER']:+.3f} [95% {s['lo']:+.3f},{s['hi']:+.3f}]{pos}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=float, default=4.0)
    p.add_argument("--hold", type=int, default=192)
    p.add_argument("--min_stop_pct", type=float, default=0.0015)
    p.add_argument("--cooldown", type=int, default=16)
    p.add_argument("--cost", type=float, default=0.0010)
    args = p.parse_args()

    df = build_frame(96, 96)
    targets = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    ks = [2.0, 2.5, 3.0, 3.5, 4.0]

    print("=" * 86)
    print(f"CONTINUATION expectancy sweep | m={args.m} hold={args.hold}b "
          f"min_stop={args.min_stop_pct:.2%} cost={args.cost:.2%}")
    print("E[R] = net expectancy per trade (x100 = edge over 100 trades). "
          "'+EV' = bootstrap 95% low > 0.")

    print("-" * 86)
    print("A) TARGET SWEEP at k=3.0  (all / short / long / trend-aligned)")
    for tr in targets:
        ev = continuation_events(df, 3.0, args.m, tr, args.hold,
                                 args.min_stop_pct, args.cooldown, args.cost)
        print(f"target={tr}R:")
        line("all", stat(ev))
        line("short", stat(ev[ev.side == "short"]))
        line("long", stat(ev[ev.side == "long"]))
        line("trend-aligned", stat(ev[ev.aligned]))

    print("-" * 86)
    print("B) ROBUSTNESS GRID — E[R] for 'all', rows=k, cols=target "
          "(plateau = real, lone spike = overfit)")
    header = "     " + "".join(f"{t:>8}R" for t in targets)
    print(header)
    for k in ks:
        cells = []
        for tr in targets:
            ev = continuation_events(df, k, args.m, tr, args.hold,
                                     args.min_stop_pct, args.cooldown, args.cost)
            s = stat(ev)
            cells.append(f"{s['ER']:+.3f}" if s else "   -  ")
        print(f"k={k:<3} " + "".join(f"{c:>9}" for c in cells))
    print("=" * 86)


if __name__ == "__main__":
    main()
