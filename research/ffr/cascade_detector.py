"""
research/ffr/cascade_detector.py

Detects forced-flow cascade events and measures their RAW forward outcomes —
deliberately BEFORE any confluence filters. The goal is the unconditional base
rate: when BTC perp prints a volume + range climax, how often does price
actually revert to a 4R target before stopping out? Then we layer simple
filters (basis, trend, wick) and watch whether the win rate actually moves.

If the raw number is near the 4R breakeven (20%) and no filter lifts it
convincingly, the edge isn't there — and this script will say so in numbers.

Method / honesty guards:
  - No lookahead: ATR and volume baselines use only bars strictly before the
    event bar (rolling().shift(1)).
  - Entry = close of the climax bar (it has closed; realistic). Stop = the
    bar's extreme (the wick). Risk R = |entry - stop|.
  - Forward barrier: does +target_R or -1R come first, walked bar-by-bar on
    intrabar highs/lows.
  - Same-bar ambiguity (a later bar spans both stop and target) is resolved
    PESSIMISTICALLY as a stop-out. No optimistic fills.
  - Events are non-overlapping: after one resolves we re-arm on the next bar.
  - Costs: a round-trip taker cost (default 0.10% of entry) is converted into
    R units and subtracted, so reported expectancy is net.

Reads CSVs from research/ffr/data/ (oldest-first; columns t,o,h,l,c,v).
"""

import os
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(inst: str, bar: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, f"ffr_{inst}_{bar}.csv"))
    return df.sort_values("t").reset_index(drop=True)


def add_indicators(df: pd.DataFrame, atr_period: int, vol_period: int) -> pd.DataFrame:
    h, l, c = df["h"], df["l"], df["c"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_period).mean().shift(1)        # prior bars only
    df["vol_med"] = df["v"].rolling(vol_period).median().shift(1)
    df["range"] = h - l
    return df


def daily_trend(df_d: pd.DataFrame, sma: int = 20) -> pd.DataFrame:
    df_d = df_d.sort_values("t").reset_index(drop=True)
    df_d["sma"] = df_d["c"].rolling(sma).mean()
    df_d["trend"] = np.where(df_d["c"] > df_d["sma"], 1,
                             np.where(df_d["c"] < df_d["sma"], -1, 0))
    return df_d[["t", "trend"]].dropna()


def wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((centre - half) / d, (centre + half) / d)


def label_events(df, k, m, target_R, hold_bars, cost_pct):
    n = len(df)
    O, H, L, C = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    rng, atr, vmed, vol = (df["range"].values, df["atr"].values,
                           df["vol_med"].values, df["v"].values)
    basis, trend = df["basis"].values, df["trend"].values
    out = []
    i = 0
    while i < n:
        if (np.isnan(atr[i]) or np.isnan(vmed[i]) or atr[i] <= 0 or vmed[i] <= 0):
            i += 1
            continue
        spike = (rng[i] >= k * atr[i]) and (vol[i] >= m * vmed[i])
        if not spike:
            i += 1
            continue
        down, up = C[i] < O[i], C[i] > O[i]
        if not (down or up):
            i += 1
            continue

        direction = "long" if down else "short"
        entry = C[i]
        if direction == "long":
            stop = L[i]
            risk = entry - stop
            wick = (min(O[i], C[i]) - L[i])
            target = entry + target_R * risk
        else:
            stop = H[i]
            risk = stop - entry
            wick = (H[i] - max(O[i], C[i]))
            target = entry - target_R * risk
        if risk <= 0:
            i += 1
            continue
        wick_frac = wick / rng[i] if rng[i] > 0 else 0.0

        outcome, rmult, bars, mfe, mae = "timeout", 0.0, 0, 0.0, 0.0
        end = min(n, i + 1 + hold_bars)
        for j in range(i + 1, end):
            bars = j - i
            if direction == "long":
                mfe = max(mfe, (H[j] - entry) / risk)
                mae = max(mae, (entry - L[j]) / risk)
                hit_stop, hit_tgt = L[j] <= stop, H[j] >= target
            else:
                mfe = max(mfe, (entry - L[j]) / risk)
                mae = max(mae, (H[j] - entry) / risk)
                hit_stop, hit_tgt = H[j] >= stop, L[j] <= target
            if hit_stop:                       # pessimistic: stop wins ties
                outcome, rmult = "loss", -1.0
                break
            if hit_tgt:
                outcome, rmult = "win", target_R
                break

        cost_R = (cost_pct * entry) / risk
        net = (rmult - cost_R) if outcome != "timeout" else (0.0 - cost_R)
        out.append(dict(t=int(df["t"].values[i]), direction=direction,
                        range_mult=rng[i] / atr[i], vol_mult=vol[i] / vmed[i],
                        wick_frac=wick_frac, basis=basis[i], trend=int(trend[i]),
                        outcome=outcome, r=rmult, net=net, bars=bars,
                        mfe=mfe, mae=mae))
        i = i + max(bars, 1) + 1                # re-arm after resolution
    return pd.DataFrame(out)


def summarize(ev: pd.DataFrame, label: str, target_R: float):
    n = len(ev)
    if n == 0:
        print(f"  {label:<34} N=0")
        return
    wins = int((ev.outcome == "win").sum())
    losses = int((ev.outcome == "loss").sum())
    to = int((ev.outcome == "timeout").sum())
    resolved = wins + losses
    wr_res = wins / resolved if resolved else 0.0
    lo, hi = wilson(wins, resolved)
    print(f"  {label:<34} N={n:<4} W/L/TO={wins}/{losses}/{to} "
          f"| WR={wr_res:5.1%} [{lo:.0%}-{hi:.0%}] "
          f"| E[netR]={ev.net.mean():+.3f} "
          f"| MFE50={ev.mfe.median():.2f} MAE50={ev.mae.median():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=3.0, help="range >= k*ATR")
    p.add_argument("--m", type=float, default=4.0, help="volume >= m*median")
    p.add_argument("--target_r", type=float, default=4.0)
    p.add_argument("--hold", type=int, default=192, help="max hold in 15m bars (192=2d)")
    p.add_argument("--cost", type=float, default=0.0010, help="round-trip taker, frac of price")
    p.add_argument("--atr_period", type=int, default=96)
    p.add_argument("--vol_period", type=int, default=96)
    args = p.parse_args()

    perp = load("BTC-USDT-SWAP", "15m")
    spot = load("BTC-USDT", "15m")[["t", "c"]].rename(columns={"c": "spot_c"})
    perp = perp.merge(spot, on="t", how="left")
    perp["basis"] = (perp["c"] - perp["spot_c"]) / perp["spot_c"]

    daily = load("BTC-USDT-SWAP", "1D")
    dtrend = daily_trend(daily)
    perp = pd.merge_asof(perp.sort_values("t"), dtrend.sort_values("t"),
                         on="t", direction="backward")

    perp = add_indicators(perp, args.atr_period, args.vol_period)
    ev = label_events(perp, args.k, args.m, args.target_r, args.hold, args.cost)

    breakeven = 1.0 / (1.0 + args.target_r)
    print("=" * 78)
    print(f"FFR cascade base-rate | k={args.k} m={args.m} target={args.target_r}R "
          f"hold={args.hold}b cost={args.cost:.2%}")
    print(f"Breakeven WR at {args.target_r}R = {breakeven:.1%}  "
          f"(WR above this = positive expectancy)")
    print(f"Total cascade events detected: {len(ev)}  "
          f"({(ev.direction=='long').sum()} long / {(ev.direction=='short').sum()} short)")
    print("-" * 78)
    if len(ev) == 0:
        print("No events — loosen k/m.")
        return

    print("RAW (no filters):")
    summarize(ev, "all", args.target_r)
    summarize(ev[ev.direction == "long"], "long (fade down-spike)", args.target_r)
    summarize(ev[ev.direction == "short"], "short (fade up-spike)", args.target_r)

    print("LONG, single filters:")
    L = ev[ev.direction == "long"]
    summarize(L[L.trend == 1], "long + uptrend (with-trend dip)", args.target_r)
    summarize(L[L.basis < 0], "long + perp discount (basis<0)", args.target_r)
    summarize(L[L.wick_frac >= 0.30], "long + lower wick>=30%", args.target_r)

    print("LONG, stacked filters:")
    summarize(L[(L.trend == 1) & (L.basis < 0)], "long + uptrend + basis<0", args.target_r)
    summarize(L[(L.trend == 1) & (L.basis < 0) & (L.wick_frac >= 0.30)],
              "long + uptrend + basis<0 + wick", args.target_r)

    print("SHORT, stacked filters (mirror):")
    S = ev[ev.direction == "short"]
    summarize(S[(S.trend == -1) & (S.basis > 0)], "short + downtrend + basis>0", args.target_r)
    print("=" * 78)


if __name__ == "__main__":
    main()
