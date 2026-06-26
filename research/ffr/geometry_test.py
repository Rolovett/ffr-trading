"""
research/ffr/geometry_test.py

The naive fade failed: gross win rate ~15% vs a 20% breakeven at 4R, i.e.
negative before fees, with MAE ~1.9R showing the wick is not a floor — entering
at the climax is catching the knife.

This pre-registers THREE entry geometries on the SAME cascade events and lets
the gross base rates adjudicate, instead of tuning filters on the dead one:

  1. fade_close   : baseline (enter climax close, fade direction, stop = wick).
  2. fade_reclaim : wait up to W bars for price to close back beyond the climax
                    bar, THEN fade; stop = swing extreme of the cascade window
                    (wider, fee-resilient). "Wait for the knife to stick."
  3. continuation : trade WITH the cascade (down-spike -> short); stop on the
                    far side of the climax bar. Motivated by the large MAE.

Guards:
  - Minimum stop distance (min_stop_pct of price): skip setups whose stop is so
    tight that fees exceed ~1R. These are not tradeable and were wrecking E[R].
  - No lookahead; pessimistic same-bar tie -> stop. Non-overlapping events via
    a cooldown gap between triggers.
  - Reports GROSS win rate vs breakeven (edge test, cost-independent) AND net
    E[R] after a round-trip taker cost.

Run after data_history.py. Reuses helpers from cascade_detector.py.
"""

import argparse
import numpy as np
import pandas as pd
from cascade_detector import load, add_indicators, daily_trend, wilson


def build_frame(atr_period, vol_period):
    perp = load("BTC-USDT-SWAP", "15m")
    spot = load("BTC-USDT", "15m")[["t", "c"]].rename(columns={"c": "spot_c"})
    perp = perp.merge(spot, on="t", how="left")
    perp["basis"] = (perp["c"] - perp["spot_c"]) / perp["spot_c"]
    daily = load("BTC-USDT-SWAP", "1D")
    perp = pd.merge_asof(perp.sort_values("t"), daily_trend(daily).sort_values("t"),
                         on="t", direction="backward")
    return add_indicators(perp, atr_period, vol_period)


def walk(long, entry, stop, target, H, L, start, end):
    """Bar-by-bar barrier from `start` to `end`. Pessimistic: stop wins ties."""
    risk = (entry - stop) if long else (stop - entry)
    if risk <= 0:
        return None
    mfe = mae = 0.0
    for j in range(start, end):
        if long:
            mfe = max(mfe, (H[j] - entry) / risk)
            mae = max(mae, (entry - L[j]) / risk)
            if L[j] <= stop:
                return ("loss", -1.0, mfe, mae, j - start + 1)
            if H[j] >= target:
                return ("win", (target - entry) / risk, mfe, mae, j - start + 1)
        else:
            mfe = max(mfe, (entry - L[j]) / risk)
            mae = max(mae, (H[j] - entry) / risk)
            if H[j] >= stop:
                return ("loss", -1.0, mfe, mae, j - start + 1)
            if L[j] <= target:
                return ("win", (entry - target) / risk, mfe, mae, j - start + 1)
    return ("timeout", 0.0, mfe, mae, end - start)


def triggers(df, k, m, cooldown):
    """Indices of cascade climax bars, spaced >= cooldown apart."""
    rng, atr = df["range"].values, df["atr"].values
    vmed, vol = df["vol_med"].values, df["v"].values
    O, C = df["o"].values, df["c"].values
    out, last = [], -10 ** 9
    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(vmed[i]) or atr[i] <= 0 or vmed[i] <= 0:
            continue
        if rng[i] >= k * atr[i] and vol[i] >= m * vmed[i] and C[i] != O[i]:
            if i - last >= cooldown:
                out.append(i)
                last = i
    return out


def record(rows, model, direction, res, entry, risk, cost_pct, ctx):
    if res is None:
        return
    outcome, rmult, mfe, mae, bars = res
    cost_R = (cost_pct * entry) / risk
    net = (rmult - cost_R) if outcome != "timeout" else (-cost_R)
    rows.append(dict(model=model, direction=direction, outcome=outcome,
                     gross_win=(outcome == "win"), r=rmult, net=net,
                     mfe=mfe, mae=mae, bars=bars, **ctx))


def evaluate(df, k, m, target_R, hold, W, min_stop_pct, cooldown, cost_pct):
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    n = len(df)
    rows = []
    skipped = {"fade_close": 0, "fade_reclaim": 0, "continuation": 0}
    no_entry = 0
    for i in triggers(df, k, m, cooldown):
        down = C[i] < O[i]
        ctx = dict(trend=int(df["trend"].values[i]), basis=float(df["basis"].values[i]))
        end_i = min(n, i + 1 + hold)

        # 1) fade_close
        if down:
            entry, stop = C[i], L[i]
            tgt = entry + target_R * (entry - stop)
            risk = entry - stop
            if risk / entry >= min_stop_pct:
                record(rows, "fade_close", "long",
                       walk(True, entry, stop, tgt, H, L, i + 1, end_i),
                       entry, risk, cost_pct, ctx)
            else:
                skipped["fade_close"] += 1
        else:
            entry, stop = C[i], H[i]
            tgt = entry - target_R * (stop - entry)
            risk = stop - entry
            if risk / entry >= min_stop_pct:
                record(rows, "fade_close", "short",
                       walk(False, entry, stop, tgt, H, L, i + 1, end_i),
                       entry, risk, cost_pct, ctx)
            else:
                skipped["fade_close"] += 1

        # 2) fade_reclaim — wait for close back beyond the climax bar
        w_end = min(n, i + 1 + W)
        entered = False
        if down:
            swing_low = L[i]
            for j in range(i + 1, w_end):
                swing_low = min(swing_low, L[j])
                if C[j] > H[i]:                       # reclaimed climax high
                    entry, stop = C[j], swing_low
                    risk = entry - stop
                    if risk > 0 and risk / entry >= min_stop_pct:
                        tgt = entry + target_R * risk
                        record(rows, "fade_reclaim", "long",
                               walk(True, entry, stop, tgt, H, L, j + 1, min(n, j + 1 + hold)),
                               entry, risk, cost_pct, ctx)
                        entered = True
                    else:
                        skipped["fade_reclaim"] += 1
                        entered = True
                    break
        else:
            swing_high = H[i]
            for j in range(i + 1, w_end):
                swing_high = max(swing_high, H[j])
                if C[j] < L[i]:
                    entry, stop = C[j], swing_high
                    risk = stop - entry
                    if risk > 0 and risk / entry >= min_stop_pct:
                        tgt = entry - target_R * risk
                        record(rows, "fade_reclaim", "short",
                               walk(False, entry, stop, tgt, H, L, j + 1, min(n, j + 1 + hold)),
                               entry, risk, cost_pct, ctx)
                        entered = True
                    else:
                        skipped["fade_reclaim"] += 1
                        entered = True
                    break
        if not entered:
            no_entry += 1

        # 3) continuation — trade WITH the cascade
        if down:                                       # down-spike -> short
            entry, stop = C[i], H[i]
            tgt = entry - target_R * (stop - entry)
            risk = stop - entry
            if risk / entry >= min_stop_pct:
                record(rows, "continuation", "short",
                       walk(False, entry, stop, tgt, H, L, i + 1, end_i),
                       entry, risk, cost_pct, ctx)
            else:
                skipped["continuation"] += 1
        else:                                          # up-spike -> long
            entry, stop = C[i], L[i]
            tgt = entry + target_R * (entry - stop)
            risk = entry - stop
            if risk / entry >= min_stop_pct:
                record(rows, "continuation", "long",
                       walk(True, entry, stop, tgt, H, L, i + 1, end_i),
                       entry, risk, cost_pct, ctx)
            else:
                skipped["continuation"] += 1

    return pd.DataFrame(rows), skipped, no_entry


def summarize(ev, label, target_R):
    n = len(ev)
    if n == 0:
        print(f"  {label:<30} N=0")
        return
    wins = int(ev.gross_win.sum())
    losses = int((ev.outcome == "loss").sum())
    to = int((ev.outcome == "timeout").sum())
    resolved = wins + losses
    wr = wins / resolved if resolved else 0.0
    lo, hi = wilson(wins, resolved)
    edge = "EDGE" if lo > 1.0 / (1.0 + target_R) else "    "
    print(f"  {label:<30} N={n:<4} W/L/TO={wins}/{losses}/{to} "
          f"| grossWR={wr:5.1%} [{lo:.0%}-{hi:.0%}] {edge} "
          f"| E[netR]={ev.net.mean():+.2f} | MFE50={ev.mfe.median():.2f} MAE50={ev.mae.median():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--m", type=float, default=4.0)
    p.add_argument("--target_r", type=float, default=4.0)
    p.add_argument("--hold", type=int, default=192)
    p.add_argument("--reclaim_window", type=int, default=48, help="bars to await reclaim (48=12h)")
    p.add_argument("--min_stop_pct", type=float, default=0.0015, help="skip stops tighter than this frac of price")
    p.add_argument("--cooldown", type=int, default=16)
    p.add_argument("--cost", type=float, default=0.0010)
    args = p.parse_args()

    df = build_frame(96, 96)
    ev, skipped, no_entry = evaluate(df, args.k, args.m, args.target_r, args.hold,
                                     args.reclaim_window, args.min_stop_pct,
                                     args.cooldown, args.cost)
    be = 1.0 / (1.0 + args.target_r)
    print("=" * 84)
    print(f"FFR geometry test | k={args.k} m={args.m} target={args.target_r}R "
          f"min_stop={args.min_stop_pct:.2%} reclaim_W={args.reclaim_window}b")
    print(f"Breakeven gross WR at {args.target_r}R = {be:.1%}. "
          f"'EDGE' = Wilson lower bound clears breakeven.")
    print(f"Triggers: {len(triggers(df, args.k, args.m, args.cooldown))} "
          f"| skipped(min_stop)={skipped} | reclaim no-entry={no_entry}")
    print("-" * 84)
    for model in ["fade_close", "fade_reclaim", "continuation"]:
        sub = ev[ev.model == model]
        print(f"{model}:")
        summarize(sub, "all", args.target_r)
        summarize(sub[sub.direction == "long"], "long", args.target_r)
        summarize(sub[sub.direction == "short"], "short", args.target_r)
    print("=" * 84)


if __name__ == "__main__":
    main()
