"""
research/ffr/outcome_marker.py

Companion to live_scanner.py. Two jobs:

1. MODEL outcomes: for each signal not yet resolved, replay the real OKX forward
   path to decide fill (maker limit) -> win/loss/timeout, pessimistic ties,
   maker cost applied. Same barrier conventions as the backtest. Writes back to
   paper_signals.json.

2. RECONCILIATION: read the trader's hand-completed journal.csv (actual OKX demo
   fills) and render it next to the model in one scorecard, with the delta. This
   is what tells us whether the validated edge survives real execution + the
   trader's availability. Journal parsing is isolated so a hand-edit typo can
   never break the auto-resolver.

Also cross-checks scanner liveness via live_state.json's heartbeat and warns if
stale. Sends to Telegram when something newly resolved OR the scanner looks
stalled, so it isn't noisy in steady state.
"""

import os
import csv
import json
import time
import logging
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OKX = "https://www.okx.com/api/v5"
HEADERS = {"User-Agent": "ffr-outcome/0.1"}
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "paper_signals.json")
STATE_PATH = os.path.join(HERE, "live_state.json")
JOURNAL_PATH = os.path.join(HERE, "journal.csv")

FILL_WINDOW = 8          # bars to wait for the pullback fill (matches scanner)
HOLD = 192               # max bars to resolution (matches backtest)
TARGET_R = 2.0
MAKER = 0.0005           # round-trip maker friction
BAR_MS = 15 * 60 * 1000
STALE_MIN = 100          # scanner heartbeat older than this -> stalled warning


def _get(path, params):
    try:
        r = requests.get(f"{OKX}{path}", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        d = r.json()
        return d.get("data") if d.get("code") == "0" else None
    except Exception as e:
        log.warning("GET %s failed: %s", path, e)
        return None


def fetch_since(inst, bar, start_ms):
    """Oldest-first closed candles [ts,o,h,l,c,v] from start_ms to now."""
    rows, after, pages = {}, "", 0
    while True:
        params = {"instId": inst, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        data = _get("/market/history-candles", params)
        if not data:
            break
        oldest = int(data[-1][0])
        for r in data:
            if r[8] == "1":
                rows[int(r[0])] = [int(r[0]), float(r[1]), float(r[2]),
                                   float(r[3]), float(r[4]), float(r[5])]
        pages += 1
        if oldest <= start_ms or pages > 80:
            break
        after = str(oldest)
        time.sleep(0.2)
    return [rows[t] for t in sorted(rows) if t >= start_ms]


def resolve(sig, series, idx):
    """Return dict of resolution fields, or None if still open (not enough bars)."""
    i = idx[sig["bar_ts"]]
    short = sig["side"] == "short"
    entry, stop, target = sig["entry_015"], sig["stop"], sig["target_015"]
    n = len(series)

    fill = None
    last_fill_bar = min(n - 1, i + FILL_WINDOW)
    for j in range(i + 1, last_fill_bar + 1):
        hi, lo = series[j][2], series[j][3]
        if (short and hi >= entry) or ((not short) and lo <= entry):
            fill = j
            break
    if fill is None:
        if i + FILL_WINDOW < n:
            return {"outcome": "no_fill", "net_r": 0.0,
                    "resolved_at": series[min(i + FILL_WINDOW, n - 1)][0]}
        return None

    risk = (stop - entry) if short else (entry - stop)
    cost_r = (MAKER * entry) / risk if risk > 0 else 0.0
    end = min(n, fill + HOLD)
    for k in range(fill, end):
        hi, lo = series[k][2], series[k][3]
        if short:
            if hi >= stop:
                return {"outcome": "loss", "net_r": round(-1.0 - cost_r, 3),
                        "fill_ts": series[fill][0], "resolved_at": series[k][0]}
            if lo <= target:
                return {"outcome": "win", "net_r": round(TARGET_R - cost_r, 3),
                        "fill_ts": series[fill][0], "resolved_at": series[k][0]}
        else:
            if lo <= stop:
                return {"outcome": "loss", "net_r": round(-1.0 - cost_r, 3),
                        "fill_ts": series[fill][0], "resolved_at": series[k][0]}
            if hi >= target:
                return {"outcome": "win", "net_r": round(TARGET_R - cost_r, 3),
                        "fill_ts": series[fill][0], "resolved_at": series[k][0]}
    if end - fill >= HOLD:
        last = series[end - 1][4]
        mtm = ((entry - last) if short else (last - entry)) / risk
        return {"outcome": "timeout", "net_r": round(mtm - cost_r, 3),
                "fill_ts": series[fill][0], "resolved_at": series[end - 1][0]}
    return None


def model_metrics(sigs):
    taken = [s for s in sigs if s.get("outcome") in ("win", "loss", "timeout")]
    no_fill = [s for s in sigs if s.get("outcome") == "no_fill"]
    decided = taken + no_fill
    w = sum(s["outcome"] == "win" for s in taken)
    l = sum(s["outcome"] == "loss" for s in taken)
    to = sum(s["outcome"] == "timeout" for s in taken)
    er35l = [s["net_r"] for s in taken if s.get("ge_35")]
    return dict(
        n=len(sigs), open=sum(s.get("outcome") is None for s in sigs),
        taken=len(taken), fill=(len(taken) / len(decided) if decided else 0.0),
        w=w, l=l, to=to, wr=(w / (w + l) if (w + l) else 0.0),
        er=(sum(s["net_r"] for s in taken) / len(taken) if taken else 0.0),
        er35=(sum(er35l) / len(er35l) if er35l else 0.0), n35=len(er35l))


def _num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _yn(x):
    s = (str(x) if x is not None else "").strip().lower()
    if s in ("y", "yes", "1", "true"):
        return True
    if s in ("n", "no", "0", "false"):
        return False
    return None


def read_journal():
    rows = {}
    try:
        with open(JOURNAL_PATH, newline="") as f:
            for r in csv.DictReader(f):
                sid = (r.get("signal_id") or "").strip()
                if sid:
                    rows[sid] = r
    except Exception as e:
        log.warning("journal read failed (non-fatal): %s", e)
    return rows


def actual_metrics(journal):
    placed = filled = missed = pending = 0
    rs = []
    for r in journal.values():
        try:
            p = _yn(r.get("placed"))
            if p is True:
                placed += 1
                if _num(r.get("fill_price")) is not None:
                    filled += 1
                    rr = _num(r.get("realised_R"))
                    if rr is not None:
                        rs.append(rr)
            elif p is False:
                missed += 1
            else:
                pending += 1
        except Exception:
            pending += 1
    w = sum(1 for x in rs if x > 0)
    l = sum(1 for x in rs if x <= 0)
    return dict(placed=placed, filled=filled, missed=missed, pending=pending,
                fill=(filled / placed if placed else 0.0),
                w=w, l=l, wr=(w / (w + l) if (w + l) else 0.0),
                er=(sum(rs) / len(rs) if rs else 0.0), closed=len(rs))


def scanner_liveness():
    try:
        st = json.load(open(STATE_PATH))
    except Exception:
        return ("  scanner: ❓ no live_state yet (awaiting first run)", False)
    last = st.get("last_run_ts")
    if not last:
        return ("  scanner: ❓ heartbeat missing", False)
    age_min = (int(time.time() * 1000) - last) / 60000.0
    obs = st.get("last_obs", {}) or {}
    extra = ""
    if "range_x" in obs:
        extra = f" | last saw {obs.get('range_x')}×ATR, {obs.get('vol_x')}×med, trend {obs.get('trend')}"
    if age_min > STALE_MIN:
        return (f"  scanner: ⚠️ STALLED — last run {age_min:.0f}m ago{extra}", True)
    return (f"  scanner: ✅ last run {age_min:.0f}m ago{extra}", False)


def render(m, a, live_line):
    L = [
        "\U0001f9ea <b>FFR PAPER — SCORECARD</b>",
        f"  signals: {m['n']} | open: {m['open']} | resolved: {m['taken']}",
        f"  [model] fill {m['fill']:.0%} | W/L/TO {m['w']}/{m['l']}/{m['to']} | "
        f"WR {m['wr']:.0%} | E[R] {m['er']:+.3f}",
        f"  [model] ⭐≥3.5 E[R] {m['er35']:+.3f} (N={m['n35']})",
    ]
    if a["placed"] or a["missed"] or a["pending"]:
        L += [
            "\U0001f4d3 <b>ACTUAL (your OKX demo)</b>",
            f"  placed {a['placed']} | filled {a['filled']} | missed {a['missed']} | pending {a['pending']}",
            f"  actual fill {a['fill']:.0%} | W/L {a['w']}/{a['l']} | "
            f"WR {a['wr']:.0%} | E[R] {a['er']:+.3f} ({a['closed']} closed)",
        ]
        if a["closed"] > 0:
            L.append(f"  Δ model→actual E[R]: {a['er'] - m['er']:+.3f}")
    else:
        L.append("\U0001f4d3 actual: awaiting completed demo trades")
    L.append(live_line)
    return "\n".join(L)


def send(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if os.environ.get("DRY_RUN", "false").lower() == "true" or not token or not chat:
        log.info("(not sent)\n%s", text)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                      timeout=10).raise_for_status()
    except Exception as e:
        log.error("send failed: %s", e)


def main():
    try:
        sigs = json.load(open(LOG_PATH))
    except Exception:
        sigs = []

    live_line, stale = scanner_liveness()

    pending = [s for s in sigs if s.get("outcome") is None]
    newly = 0
    if pending:
        start = min(s["bar_ts"] for s in pending) - BAR_MS
        series = fetch_since("BTC-USDT-SWAP", "15m", start)
        idx = {c[0]: n for n, c in enumerate(series)}
        for s in pending:
            if s["bar_ts"] not in idx:
                continue
            r = resolve(s, series, idx)
            if r:
                s.update(r)
                newly += 1
                log.info("resolved %s -> %s (%.3fR)", s["signal_id"], r["outcome"], r["net_r"])
        if newly:
            json.dump(sigs, open(LOG_PATH, "w"), indent=2)

    m = model_metrics(sigs)
    a = actual_metrics(read_journal())
    card = render(m, a, live_line)
    print(card.replace("<b>", "").replace("</b>", ""))
    if newly or stale:
        send(card)


if __name__ == "__main__":
    main()
