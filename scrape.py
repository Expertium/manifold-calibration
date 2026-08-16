#!/usr/bin/env python3
"""
Scrape Manifold Markets for a prediction-market calibration study.

Collects resolved YES/NO binary markets and, for each one, reconstructs the
market probability at three moments in its life:

    early : created + min(3 days, L/4)   -- far from closure
    mid   : halfway between creation and closure
    late  : closure - min(3 days, L/4)   -- shortly before closure

where L is the market's lifetime. The horizon is capped at a quarter of the
market's life so the three moments stay strictly ordered even for markets that
resolve within days; see snapshot_times.

"Closure" means the last moment the market was tradeable, i.e.
min(closeTime, resolutionTime): a market can be resolved early (trading stops
at resolution) or resolved long after trading stopped (trading stops at close).

Probabilities come from the bet stream. Every bet carries probBefore/probAfter,
so the probability at time t is the probAfter of the last bet at or before t,
or the first bet's probBefore if no bet had happened yet (= the opening price).

Everything is read-only and needs no API key. The public rate limit is
500 requests/min per IP; this script stays under it.

Output lands in data/ and the run is resumable -- re-running continues where it
left off, and raising TARGET extends an existing dataset instead of redoing it.

Settings live in the constants block below (no command-line arguments):
edit them there and run the file.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://api.manifold.markets/v0"
DAY_MS = 86_400_000
WINDOW_MS = 3 * DAY_MS


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
    """The three snapshot timestamps.

        early : created + min(3 days, L/4)
        mid   : created + L/2
        late  : end     - min(3 days, L/4)

    For a market living 12 days or more this is exactly "3 days after
    creation, the midpoint, and 3 days before the end". Below that the
    horizons compress symmetrically to 25% / 50% / 75% of the market's life
    instead of crossing over each other.

    The previous definition floored `late` at creation, so a market that
    resolved within 3 days was graded on the untouched 0.50 opening price:
    17% of markets piled up at exactly 0.5000 in the late panel. Capping the
    horizon removes that by construction instead of by filtering the markets
    out, which is what lets one sample serve all three snapshots without
    conditioning on anything that happened after the forecast was made.

    Capping at L/2 would look more natural but collapses all three moments
    onto the midpoint for the 28% of markets shorter than 6 days; L/4 keeps
    them strictly ordered for every market in the dataset.
    """
    created, end = m["createdTime"], end_time(m)
    lifetime = end - created
    cap = min(WINDOW_MS, lifetime // 4)
    return {
        "early": created + cap,
        "mid": created + lifetime // 2,
        "late": end - cap,
    }


def _history_until(mid, until):
    """Every bet at or before `until`, ascending, plus the opening price.

    Volume and trader counts at time t need the whole bet stream up to t, not
    just the last bet before it, so the old one-tiny-request-per-snapshot
    shortcut cannot serve them. Paging forward costs about the same anyway:
    94% of markets have <= 60 traders and fit in a single 1000-bet page.
    """
    bets, cursor, opening = [], None, None
    while True:
        params = {"contractId": mid, "limit": 1000, "order": "asc"}
        if cursor:
            params["after"] = cursor
        page = api_get("/bets", params)
        if not page:
            break
        if opening is None:
            opening = page[0]["probBefore"]
        bets.extend(b for b in page if b["createdTime"] <= until)
        # stop as soon as the stream has run past the last snapshot
        if len(page) < 1000 or page[-1]["createdTime"] > until:
            break
        cursor = page[-1]["id"]
    return bets, opening


def _bet_volume(b):
    """Mana traded by one bet, counting each matched trade only once.

    A trade between a taker and a resting limit order is recorded on *both*
    sides, so naively summing every bet's amount double-counts it: +59% on the
    busiest market checked, and not at all on markets with no limit orders.
    That error scales with activity, which is the very axis being studied, so
    it would bend the volume result. Count fills against the AMM always, and
    matched fills only from the taker's side.

    Reproduces Manifold's own `volume` exactly on markets without limit-order
    matching, and runs ~5-10% under it on the busiest ones (residual
    unexplained -- their bet streams do not sum to the reported figure under
    any simple rule).
    """
    if b.get("isRedemption"):
        return 0.0
    fills = b.get("fills") or []
    if not fills:
        return abs(b.get("amount") or 0.0)
    taker = b.get("limitProb") is None
    return sum(abs(f.get("amount") or 0.0) for f in fills
               if taker or f.get("matchedBetId") is None)


def _stats_at(bets, times, opening):
    """Probability, traded volume and trader count at each snapshot time.

    One pass: the snapshots are visited in time order and the bet index only
    moves forward, so a market with 50,000 bets still costs one sweep.

    See _bet_volume for how a trade is counted.
    """
    out, p, vol, traders, i = {}, opening, 0.0, set(), 0
    for key, t in sorted(times.items(), key=lambda kv: kv[1]):
        while i < len(bets) and bets[i]["createdTime"] <= t:
            b = bets[i]
            p = b["probAfter"]
            if not b.get("isRedemption"):
                vol += _bet_volume(b)
                traders.add(b["userId"])
            i += 1
        out[key] = (p, vol, len(traders))
    return out


def fetch_probs(m):
    """Return a result record for one market, or None if it can't be scored."""
    times = snapshot_times(m)
    mid = m["id"]

    # A short market has late < early, so page to whichever snapshot is latest.
    bets, opening = _history_until(mid, max(times.values()))
    if opening is None:
        return None  # never traded -- no probability to speak of

    stats = _stats_at(bets, times, opening)
    probs = {k: v[0] for k, v in stats.items()}
    if any(p is None for p in probs.values()):
        return None

    lifetime = end_time(m) - m["createdTime"]
    until_closed = m.get("closeTime") - m["createdTime"]
    return {
        "id": mid,
        "slug": m.get("slug"),
        "question": m.get("question"),
        "outcome": 1 if m["resolution"] == "YES" else 0,
        "p_early": probs["early"],
        "p_mid": probs["mid"],
        "p_late": probs["late"],
        # volume and trader count as of each snapshot -- unlike the final
        # `volume` / `bettors` below, these are knowable at the time the
        # forecast was made, so filtering on them carries no lookahead
        "vol_early": stats["early"][1],
        "vol_mid": stats["mid"][1],
        "vol_late": stats["late"][1],
        "traders_early": stats["early"][2],
        "traders_mid": stats["mid"][2],
        "traders_late": stats["late"][2],
        "lifetime_days": lifetime / DAY_MS,
        "until_closed_days": until_closed / DAY_MS,  # <=lifetime_days
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


# --------------------------------------------------------------------------
# settings -- edit here and run the file; there are no command-line arguments
# --------------------------------------------------------------------------

TARGET = 250000   # how many markets to collect (stops early if fewer exist)
WORKERS = 8
RPM = 450         # requests per minute; the API allows 500


def main():
    global _limiter
    _limiter = RateLimiter(RPM)

    os.makedirs(DATA, exist_ok=True)
    markets = enumerate_markets(TARGET)
    collect(markets[:TARGET], WORKERS)


if __name__ == "__main__":
    main()
