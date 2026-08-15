#!/usr/bin/env python3
"""
Scrape Manifold Markets for a prediction-market calibration study.

Collects resolved YES/NO binary markets and, for each one, reconstructs the
market probability at three moments in its life:

    early : created + min(3 days, lifetime)      -- far from closure
    mid   : halfway between creation and closure
    late  : closure - 3 days, floored at creation -- shortly before closure

"Closure" means the last moment the market was tradeable, i.e.
min(closeTime, resolutionTime): a market can be resolved early (trading stops
at resolution) or resolved long after trading stopped (trading stops at close).

Probabilities come from the bet stream. Every bet carries probBefore/probAfter,
so the probability at time t is the probAfter of the last bet at or before t,
or the first bet's probBefore if no bet had happened yet (= the opening price).

Everything is read-only and needs no API key. The public rate limit is
500 requests/min per IP; this script stays under it.

Output lands in data/ and the run is resumable -- re-running continues where it
left off, and raising --target extends an existing dataset instead of redoing it.

    python scrape.py --target 20000
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://api.manifold.markets/v0"
DAY_MS = 86_400_000
WINDOW_MS = 3 * DAY_MS

# Markets with at most this many unique bettors are fetched in one shot (their
# whole bet history fits in a single 1000-bet page), instead of one request per
# snapshot. Most Manifold markets are thin, so this roughly halves the run time.
THIN_BETTORS = 60

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# --------------------------------------------------------------------------
# HTTP: rate limiting + retries
# --------------------------------------------------------------------------

class RateLimiter:
    """Token bucket shared by all worker threads."""

    def __init__(self, per_minute):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_local = threading.local()
_limiter = None


def _session():
    # requests.Session isn't guaranteed thread-safe, so give each thread its own.
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "manifold-calibration-study/1.0"
        _local.s = s
    return _local.s


def api_get(path, params, tries=6):
    """GET an API path, retrying on rate limits and transient server errors."""
    last = None
    for attempt in range(tries):
        _limiter.acquire()
        try:
            r = _session().get(API + path, params=params, timeout=45)
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError(f"{path} failed after {tries} tries: {last}")


# --------------------------------------------------------------------------
# Step 1: enumerate resolved YES/NO binary markets
# --------------------------------------------------------------------------

KEEP_FIELDS = ("id", "slug", "question", "createdTime", "closeTime",
               "resolutionTime", "resolution", "totalLiquidity", "volume",
               "uniqueBettorCount", "mechanism", "probability")


def usable(m):
    """Keep only resolved-YES/NO play-money binary markets with a sane lifetime."""
    if m.get("outcomeType") != "BINARY":
        return False
    if not m.get("isResolved"):
        return False
    # Drop MKT (resolved to a probability) and CANCEL (N/A) -- we need a hard
    # binary outcome to score against.
    if m.get("resolution") not in ("YES", "NO"):
        return False
    # CASH markets are sweepstakes mirrors of MANA markets; keeping both would
    # double-count the same question.
    if m.get("token", "MANA") != "MANA":
        return False
    created, end = m.get("createdTime"), end_time(m)
    return bool(created and end and end > created)


def end_time(m):
    """Last moment the market was tradeable."""
    close, res = m.get("closeTime"), m.get("resolutionTime")
    both = [t for t in (close, res) if t]
    return min(both) if both else None


def enumerate_markets(target):
    """Page backwards through market creation time, appending to markets.jsonl."""
    path = os.path.join(DATA, "markets.jsonl")
    cursor_path = os.path.join(DATA, "cursor.json")

    seen, kept = set(), []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                seen.add(m["id"])
                kept.append(m)
    cursor = None
    exhausted = False
    if os.path.exists(cursor_path):
        state = json.load(open(cursor_path, encoding="utf-8"))
        cursor, exhausted = state.get("beforeTime"), state.get("exhausted", False)

    if len(kept) >= target:
        print(f"[markets] already have {len(kept)}, target {target} -- skipping")
        return kept
    if exhausted:
        print(f"[markets] API exhausted at {len(kept)} markets (that is all of them)")
        return kept

    print(f"[markets] have {len(kept)}, fetching up to {target}...")
    out = open(path, "a", encoding="utf-8")
    try:
        while len(kept) < target:
            params = {"term": "", "filter": "resolved", "contractType": "BINARY",
                      "sort": "newest", "limit": 1000}
            if cursor:
                params["beforeTime"] = cursor
            page = api_get("/search-markets", params)
            if not page:
                exhausted = True
                print("[markets] reached the end of the market list")
                break

            for m in page:
                if m["id"] in seen or not usable(m):
                    continue
                seen.add(m["id"])
                rec = {k: m.get(k) for k in KEEP_FIELDS}
                out.write(json.dumps(rec) + "\n")
                kept.append(rec)

            times = [m["createdTime"] for m in page if m.get("createdTime")]
            new_cursor = min(times) if times else None
            # Guard against a page that cannot advance the cursor.
            if new_cursor is None or new_cursor == cursor:
                exhausted = True
                print("[markets] cursor stopped advancing; stopping enumeration")
                break
            cursor = new_cursor
            out.flush()
            json.dump({"beforeTime": cursor, "exhausted": exhausted},
                      open(cursor_path, "w", encoding="utf-8"))
            print(f"\r[markets] kept {len(kept)}", end="", flush=True)
    finally:
        out.close()
        json.dump({"beforeTime": cursor, "exhausted": exhausted},
                  open(cursor_path, "w", encoding="utf-8"))

    print(f"\n[markets] {len(kept)} usable markets")
    return kept


# --------------------------------------------------------------------------
# Step 2: reconstruct probabilities at the three snapshot times
# --------------------------------------------------------------------------

def snapshot_times(m):
    """The three timestamps, per the study definition."""
    created, end = m["createdTime"], end_time(m)
    lifetime = end - created
    return {
        "early": created + min(WINDOW_MS, lifetime),
        "mid": created + lifetime // 2,
        "late": created + max(lifetime - WINDOW_MS, 0),
    }


def _prob_from_history(bets, t, opening):
    """Probability at time t given the full ascending bet history."""
    p = opening
    for b in bets:
        if b["createdTime"] <= t:
            p = b["probAfter"]
        else:
            break
    return p


def fetch_probs(m):
    """Return a result record for one market, or None if it can't be scored."""
    times = snapshot_times(m)
    mid = m["id"]

    probs = None
    # Fast path: thin markets fit their whole history in one request.
    if (m.get("uniqueBettorCount") or 0) <= THIN_BETTORS:
        bets = api_get("/bets", {"contractId": mid, "limit": 1000, "order": "asc"})
        if not bets:
            return None  # never traded -- no probability to speak of
        if len(bets) < 1000:
            opening = bets[0]["probBefore"]
            probs = {k: _prob_from_history(bets, t, opening) for k, t in times.items()}

    if probs is None:
        # General path: one tiny query per snapshot, newest bet at or before t.
        probs, opening = {}, None
        for key, t in times.items():
            hit = api_get("/bets", {"contractId": mid, "limit": 1, "beforeTime": t})
            if hit:
                probs[key] = hit[0]["probAfter"]
                continue
            if opening is None:
                first = api_get("/bets", {"contractId": mid, "limit": 1, "order": "asc"})
                if not first:
                    return None
                opening = first[0]["probBefore"]
            probs[key] = opening

    if any(p is None for p in probs.values()):
        return None

    lifetime = end_time(m) - m["createdTime"]
    return {
        "id": mid,
        "slug": m.get("slug"),
        "question": m.get("question"),
        "outcome": 1 if m["resolution"] == "YES" else 0,
        "p_early": probs["early"],
        "p_mid": probs["mid"],
        "p_late": probs["late"],
        "lifetime_days": lifetime / DAY_MS,
        "liquidity": m.get("totalLiquidity") or 0.0,
        "volume": m.get("volume") or 0.0,
        "bettors": m.get("uniqueBettorCount") or 0,
        "mechanism": m.get("mechanism"),
        "createdTime": m["createdTime"],
    }


def collect(markets, workers):
    """Fetch probabilities for every market not already done, appending as we go."""
    path = os.path.join(DATA, "probs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass  # tolerate a torn last line from an interrupted run

    todo = [m for m in markets if m["id"] not in done]
    print(f"[probs] {len(done)} already done, {len(todo)} to fetch")
    if not todo:
        return

    lock = threading.Lock()
    out = open(path, "a", encoding="utf-8")
    state = {"n": 0, "skipped": 0, "failed": 0, "t0": time.time()}

    def work(m):
        try:
            rec = fetch_probs(m)
        except Exception as e:
            with lock:
                state["failed"] += 1
                if state["failed"] <= 5:
                    print(f"\n[probs] {m['id']} failed: {e}")
            return
        with lock:
            state["n"] += 1
            if rec is None:
                state["skipped"] += 1
            else:
                out.write(json.dumps(rec) + "\n")
            if state["n"] % 50 == 0:
                out.flush()
                rate = state["n"] / max(time.time() - state["t0"], 1e-9)
                eta = (len(todo) - state["n"]) / max(rate, 1e-9) / 60
                print(f"\r[probs] {state['n']}/{len(todo)}  "
                      f"{rate:.1f}/s  ETA {eta:.0f} min  "
                      f"skipped {state['skipped']}  failed {state['failed']}",
                      end="", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
    finally:
        out.close()

    print(f"\n[probs] done: {state['n'] - state['skipped']} scored, "
          f"{state['skipped']} skipped (untraded), {state['failed']} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=20000,
                    help="how many markets to collect (default 20000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=400,
                    help="requests per minute; the API allows 500 (default 400)")
    args = ap.parse_args()

    global _limiter
    _limiter = RateLimiter(args.rpm)

    os.makedirs(DATA, exist_ok=True)
    markets = enumerate_markets(args.target)
    collect(markets[:args.target], args.workers)


if __name__ == "__main__":
    main()
