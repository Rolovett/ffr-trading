"""
research/ffr/live_scanner.py

FFR live PAPER scanner. Detects the validated edge on the latest CLOSED 15m
OKX bar and fires a Telegram alert + logs the signal for forward validation.
PAPER ONLY — it never places orders. Its job is to accumulate the clean,
never-seen-by-the-research sample that turns "backtested" into "proven".

Validated spec (see research notes):
  trigger : 15m range >= K*ATR(96) AND volume >= M*median(96)
  filter  : continuation, trend-aligned only
            down-spike + daily downtrend -> SHORT ; up-spike + uptrend -> LONG
  entry   : maker limit, shallow pullback f of the way back to the climax extreme
  stop    : climax far extreme ; target : 2R ; min stop : 0.15% of price

Manual demo workflow: each alert states size (1R=$2k) and a place-by deadline,
and drops a stub row into journal.csv for the trader to complete by hand after
the trade closes on the OKX demo account. The limit rests AHEAD of price
(above for shorts, below for longs); price has the ~2h fill window to come to it.

Isolated from the live scanner: own state/log files, own workflow, alerts
prefixed "FFR PAPER". Reuses the repo's Telegram secrets. Heartbeat -> live_state.json
(hourly) for liveness. Pure stdlib + requests for a fast cold start.
"""

import os
import csv
import json
import time
import logging
import statistics
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OKX = "https://www.okx.com/api/v5"
HEADERS = {"User-Agent": "ffr-paper/0.1"}
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "live_state.json")
LOG_PATH = os.path.join(HERE, "paper_signals.json")
JOURNAL_PATH = os.path.join(HERE, "journal.csv")

# ── LOCKED STRATEGY CONFIG ──────────────────────────────────────────
K = 3.0                  # range >= K*ATR (alert at 3.0; log multiple for >=3.5 subset)
M = 4.0                  # volume >= M*median
ATR_PERIOD = 96
VOL_PERIOD = 96
TARGET_R = 2.0
PULLBACK_F = 0.15        # primary maker-limit pullback
MIN_STOP_PCT = 0.0015
DAILY_SMA = 20
COOLDOWN_BARS = 16       # 4h, matches the backtested event spacing
FILL_WINDOW = 8          # bars price has to pull back to the limit (~2h)
BAR_MS = 15 * 60 * 1000
R_USD = 2000.0           # 1R risk in USD for demo position sizing

JOURNAL_HEADER = ["signal_id", "alert_iso", "side", "size_btc", "entry", "stop",
                  "target", "fill_deadline_iso", "placed", "fill_price",
                  "exit_price", "exit_reason", "realised_R", "notes"]


def _get(path, params):
    try:
        r = requests.get(f"{OKX}{path}", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        d = r.json()
        return d.get("data") if d.get("code") == "0" else None
    except Exception as e:
        log.warning("GET %s failed: %s", path, e)
        return None


def closed_candles(inst, bar, limit=240):
    """Oldest-first list of CLOSED candles: [ts,o,h,l,c,v]. Drops forming bar."""
    raw = _get("/market/candles", {"instId": inst, "bar": bar, "limit": str(limit)})
    if not raw:
        return []
    rows = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
            for r in raw if r[8] == "1"]            # confirm == "1"
    return sorted(rows, key=lambda x: x[0])


def daily_trend(dailies):
    if len(dailies) < DAILY_SMA:
        return 0
    closes = [c[4] for c in dailies]
    sma = sum(closes[-DAILY_SMA:]) / DAILY_SMA
    last = closes[-1]
    return 1 if last > sma else (-1 if last < sma else 0)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_journal_stub(row):
    """Best-effort: one stub row per alert. Never allowed to break the scanner."""
    try:
        fresh = not os.path.exists(JOURNAL_PATH)
        with open(JOURNAL_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(JOURNAL_HEADER)
            w.writerow(row)
    except Exception as e:
        log.warning("journal stub write failed: %s", e)


def send_telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if os.environ.get("DRY_RUN", "false").lower() == "true" or not token or not chat:
        log.info("DRY_RUN / no creds — would send:\n%s", text)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.error("Telegram send failed: %s", e)
        return False


def evaluate():
    c15 = closed_candles("BTC-USDT-SWAP", "15m", 240)
    state = load_json(STATE_PATH, {"last_signal_ts": 0})
    now_ms = int(time.time() * 1000)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ms / 1000))
    hour = now_iso[:13]

    def heartbeat(obs, force=False):
        # Record liveness on EVERY run; persist (->commit) hourly or when forced.
        state["last_run_ts"] = now_ms
        state["last_run_iso"] = now_iso
        state["last_obs"] = obs
        if force or state.get("last_run_hour") != hour:
            state["last_run_hour"] = hour
            save_json(STATE_PATH, state)

    if len(c15) < ATR_PERIOD + 2:
        log.info("not enough 15m bars (%d)", len(c15))
        heartbeat({"note": "insufficient_bars", "bars": len(c15)})
        return

    dailies = closed_candles("BTC-USDT-SWAP", "1D", 60)
    trend = daily_trend(dailies)

    bar = c15[-1]                                   # latest CLOSED bar
    ts, o, h, l, cl, v = bar
    window = c15[-(ATR_PERIOD + 1):-1]              # the 96 bars BEFORE it
    trs = []
    prev_c = window[0][4]
    for _, _, hh, ll, cc, _ in window:
        trs.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
        prev_c = cc
    atr = sum(trs) / len(trs)
    vmed = statistics.median([w[5] for w in window])
    if atr <= 0 or vmed <= 0:
        heartbeat({"note": "bad_atr_or_vol", "bars": len(c15)})
        return

    rng = h - l
    range_mult = rng / atr
    vol_mult = v / vmed
    down = cl < o
    up = cl > o
    trig = (range_mult >= K and vol_mult >= M and (down or up))
    aligned = (down and trend == -1) or (up and trend == 1)

    heartbeat({"bars": len(c15), "bar_ts": ts,
               "range_x": round(range_mult, 2), "vol_x": round(vol_mult, 2),
               "trend": trend, "trigger": bool(trig),
               "aligned": bool(trig and aligned)})

    if not trig:
        log.info("no trigger | range=%.2fxATR vol=%.2fxmed", range_mult, vol_mult)
        return
    if not aligned:
        log.info("trigger but not trend-aligned (trend=%d, %s) — skip",
                 trend, "down" if down else "up")
        return

    if ts <= state.get("last_signal_ts", 0):
        return                                       # already handled this bar
    if ts - state.get("last_signal_ts", 0) < COOLDOWN_BARS * BAR_MS:
        log.info("within cooldown — skip")
        return

    # Build the plan
    if down:                                         # SHORT continuation
        side, emoji = "SHORT", "\U0001f534"
        extreme = h
        entry = cl + PULLBACK_F * (extreme - cl)
        stop = extreme
        risk = stop - entry
        target = entry - TARGET_R * risk
        entry_025 = cl + 0.25 * (extreme - cl)
    else:                                            # LONG continuation
        side, emoji = "LONG", "\U0001f7e2"
        extreme = l
        entry = cl - PULLBACK_F * (cl - extreme)
        stop = extreme
        risk = entry - stop
        target = entry + TARGET_R * risk
        entry_025 = cl - 0.25 * (cl - extreme)

    if risk <= 0 or risk / entry < MIN_STOP_PCT:
        log.info("stop too tight (%.3f%%) — skip", 100 * risk / entry)
        return

    sid = f"ffr_{ts}"
    risk_pct = 100 * risk / entry
    size_btc = R_USD / risk                          # 1R=$2k -> position size in BTC
    notional = size_btc * entry
    deadline_ts = ts + (FILL_WINDOW + 1) * BAR_MS    # close of last fill bar
    deadline_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline_ts / 1000))
    deadline_hhmm = time.strftime("%H:%M", time.gmtime(deadline_ts / 1000))
    action = "SELL limit (rests ABOVE price)" if down else "BUY limit (rests BELOW price)"

    msg = (
        f"\U0001f9ea <b>FFR PAPER — CONTINUATION {side}</b> {emoji}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"  Bar close: <b>${cl:,.0f}</b>\n"
        f"  Trigger: range <b>{range_mult:.1f}×ATR</b>, vol <b>{vol_mult:.1f}×med</b>"
        f"{'  ⭐≥3.5' if range_mult >= 3.5 else ''}\n"
        f"  Daily trend: <b>{'DOWN' if trend == -1 else 'UP'}</b> ✅ aligned\n"
        f"\n<b>Place now · {action}:</b>\n"
        f"  Entry: <b>${entry:,.0f}</b>  ({PULLBACK_F:.2f} pullback)\n"
        f"  Size:  <b>{size_btc:.3f} BTC</b> (~${notional:,.0f}, 1R=$2k)\n"
        f"  Stop:  <b>${stop:,.0f}</b>  (risk {risk_pct:.2f}%)\n"
        f"  Target:<b>${target:,.0f}</b>  (2R = $4k)\n"
        f"  ⏳ Fill window to <b>{deadline_hhmm} UTC</b> (~2h) — else no-fill\n"
        f"━━━━━━━━━━━━━━━\n"
        f"  <code>{sid}</code>\n"
        f"<i>Paper/demo only — logged to journal.csv for manual outcome.</i>"
    )
    send_telegram(msg)

    bar_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts / 1000))
    append_journal_stub([sid, bar_iso, side.lower(), round(size_btc, 4),
                         round(entry, 1), round(stop, 1), round(target, 1),
                         deadline_iso, "", "", "", "", "", ""])

    rec = dict(signal_id=sid, bar_ts=ts, iso=bar_iso,
               side=side.lower(), close=cl, high=h, low=l,
               range_mult=round(range_mult, 3), vol_mult=round(vol_mult, 3),
               daily_trend=trend, entry_015=round(entry, 1), entry_025=round(entry_025, 1),
               stop=round(stop, 1), target_015=round(target, 1), risk_pct=round(risk_pct, 3),
               size_btc=round(size_btc, 4), notional_usd=round(notional, 0),
               fill_deadline_iso=deadline_iso, ge_35=range_mult >= 3.5, outcome=None)
    sigs = load_json(LOG_PATH, [])
    sigs.append(rec)
    save_json(LOG_PATH, sigs)
    state["last_signal_ts"] = ts
    save_json(STATE_PATH, state)                     # force full save (incl heartbeat fields)
    log.info("SIGNAL %s %s logged (size %.4f BTC)", sid, side, size_btc)


if __name__ == "__main__":
    if os.environ.get("TEST_PING", "").lower() == "true":
        send_telegram("\U0001f9ea <b>FFR PAPER scanner online</b> — test ping. "
                      "Paper continuation signals will appear here for forward validation.")
    evaluate()                                       # always run, so one dispatch verifies everything
