"""
research/ffr/data_history.py

Deep historical data fetcher for the Forced-Flow Reversal (FFR) backtest.

WHY THIS EXISTS
  scanner/fetchers.py is built for *live* scanning — its candle calls hit
  OKX /market/candles, which is capped at the most recent 100 bars. A
  provable backtest needs years of history. This module uses OKX
  /market/history-candles (paginated) for deep candle history, plus
  paginated funding-rate-history.

SCOPE (Path A — free data)
  Deeply backtestable here:  candles (perp + spot), funding.
  NOT deeply available free:  open interest, taker volume (OKX rubik
  endpoints are shallow). Those are recorded forward by a separate
  collector, so the OI-drop / taker filters are validated on accumulating
  live data rather than deep history. This is the accepted Path A boundary.

  Geo note: OKX public endpoints are reachable from GitHub Actions / UK
  with no geo-block, so no VPN/VPS is required. (Binance & Bybit return
  451 from Actions runners — that is why this whole stack lives on OKX.)

OUTPUT
  CSV files under research/ffr/data/, oldest-first:
    ffr_<instId>_<bar>.csv     columns: t,o,h,l,c,v   (t = ms epoch)
    ffr_funding_<instId>.csv   columns: t,fundingRate

No dependencies beyond `requests`, matching scanner/fetchers.py style.
"""

import csv
import os
import time
import logging
import requests
from typing import Optional, List

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

OKX = "https://www.okx.com/api/v5"
TIMEOUT = 15
HEADERS = {"User-Agent": "ffr-backtest/0.1"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# history-candles returns <=100 rows/call and is rate-limited (~20 req/2s).
PAGE_LIMIT = 100
PAGE_SLEEP = 0.30
PAGE_CAP = 5000  # hard safety stop (5000 pages * 100 = 500k bars)


def _get(path: str, params: dict) -> Optional[dict]:
    try:
        r = requests.get(f"{OKX}{path}", params=params,
                         headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("GET %s %s failed: %s", path, params, e)
        return None


def fetch_history_candles(inst_id: str, bar: str, start_ms: int,
                          end_ms: Optional[int] = None) -> List[list]:
    """
    Candles for inst_id/bar over [start_ms, end_ms], oldest-first.
    Pages backward from newest using the `after` cursor (returns records
    earlier than `after`), de-dupes by timestamp, then trims to start_ms.

    OKX row format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    Achievable depth varies by bar; the loop stops cleanly when OKX runs
    out of history (empty batch) — check the logged bar count.
    """
    rows = {}
    after = str(end_ms) if end_ms else ""
    pages = 0
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": str(PAGE_LIMIT)}
        if after:
            params["after"] = after
        d = _get("/market/history-candles", params)
        if not d or d.get("code") != "0" or not d.get("data"):
            if d and d.get("code") != "0":
                log.warning("history-candles %s/%s code=%s msg=%s",
                            inst_id, bar, d.get("code"), d.get("msg"))
            break
        batch = d["data"]
        oldest_ts = int(batch[-1][0])
        for row in batch:
            ts = int(row[0])
            rows[ts] = [ts, float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5])]
        pages += 1
        if oldest_ts <= start_ms:
            break
        after = str(oldest_ts)
        time.sleep(PAGE_SLEEP)
        if pages > PAGE_CAP:
            log.warning("page cap hit for %s/%s", inst_id, bar)
            break
    out = [rows[t] for t in sorted(rows)
           if t >= start_ms and (end_ms is None or t <= end_ms)]
    log.info("candles %s %s: %d bars (%d pages)", inst_id, bar, len(out), pages)
    return out


def fetch_funding_history(inst_id: str, start_ms: int,
                          end_ms: Optional[int] = None) -> List[list]:
    """
    Funding settlements (OKX settles every 8h), oldest-first.
    Row: [ts, fundingRate]. Pages backward via the `after` cursor.
    """
    rows = {}
    after = str(end_ms) if end_ms else ""
    pages = 0
    while True:
        params = {"instId": inst_id, "limit": "100"}
        if after:
            params["after"] = after
        d = _get("/public/funding-rate-history", params)
        if not d or d.get("code") != "0" or not d.get("data"):
            break
        batch = d["data"]
        oldest_ts = int(batch[-1]["fundingTime"])
        for row in batch:
            ts = int(row["fundingTime"])
            rate = row.get("fundingRate") or row.get("realizedRate") or 0
            rows[ts] = [ts, float(rate)]
        pages += 1
        if oldest_ts <= start_ms:
            break
        after = str(oldest_ts)
        time.sleep(PAGE_SLEEP)
        if pages > PAGE_CAP:
            break
    out = [rows[t] for t in sorted(rows)
           if t >= start_ms and (end_ms is None or t <= end_ms)]
    log.info("funding %s: %d settlements (%d pages)", inst_id, len(out), pages)
    return out


def _save_csv(rows: list, header: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


def build_dataset(years: float = 2.0) -> None:
    """
    Default build: perp + spot candles at 15m/1H/1D plus perp funding.
    Perp (SWAP) is primary — cascade flow lives in perps; spot candles
    feed the perp-vs-spot basis filter (perp-led vs spot-led).
    """
    now = int(time.time() * 1000)
    start = now - int(years * 365 * 24 * 60 * 60 * 1000)

    jobs = [
        ("BTC-USDT-SWAP", "15m"),   # primary signal timeframe
        ("BTC-USDT-SWAP", "1H"),
        ("BTC-USDT-SWAP", "1D"),    # HTF trend gate
        ("BTC-USDT",      "15m"),   # spot, for basis
        ("BTC-USDT",      "1H"),
    ]
    for inst, bar in jobs:
        rows = fetch_history_candles(inst, bar, start, now)
        if rows:
            _save_csv(rows, ["t", "o", "h", "l", "c", "v"],
                      os.path.join(DATA_DIR, f"ffr_{inst}_{bar}.csv"))

    fr = fetch_funding_history("BTC-USDT-SWAP", start, now)
    if fr:
        _save_csv(fr, ["t", "fundingRate"],
                  os.path.join(DATA_DIR, "ffr_funding_BTC-USDT-SWAP.csv"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FFR deep-history fetcher (OKX, free)")
    p.add_argument("--years", type=float, default=2.0,
                   help="how far back to pull (default 2.0)")
    args = p.parse_args()
    build_dataset(years=args.years)
    print("Done. CSVs in", DATA_DIR)
