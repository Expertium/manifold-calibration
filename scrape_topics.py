#!/usr/bin/env python3
"""
Fetch topic tags (groupSlugs) for the markets in the calibration study.

Topics are only exposed on the *full* market endpoint, GET /v0/market/{id}.
The bulk enumeration endpoint returns LiteMarket, which has no groupSlugs, and
there is no batch-by-ids endpoint (both `?ids=` and `/by-ids` are rejected), so
this costs one request per market.

To keep that bounded it fetches only the markets that pass the study filters
(lifetime >= 7 days, >= 3 traders) rather than every resolved market -- about a
third fewer requests.

    python scrape_topics.py

Appends one {id, topics} line per market to data/topics.jsonl and skips ids
already present, so it can be interrupted and restarted.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import scrape  # reuse the rate limiter, session handling and retry logic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "topics.jsonl")


def wanted_ids(min_lifetime_days, min_bettors):
    ids = []
    with open(os.path.join(DATA, "probs.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("lifetime_days", 0) >= min_lifetime_days
                    and r.get("bettors", 0) >= min_bettors):
                ids.append(r["id"])
    return ids


def already_done():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-lifetime-days", type=float, default=7.0)
    ap.add_argument("--min-bettors", type=int, default=3)
    args = ap.parse_args()

    scrape._limiter = scrape.RateLimiter(args.rpm)

    ids = wanted_ids(args.min_lifetime_days, args.min_bettors)
    done = already_done()
    todo = [i for i in ids if i not in done]
    print(f"{len(ids):,} markets in scope, {len(done):,} already fetched, "
          f"{len(todo):,} to go", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    state = {"n": 0, "failed": 0, "t0": time.time()}

    def work(mid):
        try:
            m = scrape.api_get(f"/market/{mid}", None)
        except Exception:
            m = None
        rec = ({"id": mid, "topics": m.get("groupSlugs") or []} if m
               else {"id": mid, "topics": None})
        with lock:
            out.write(json.dumps(rec) + "\n")
            state["n"] += 1
            if m is None:
                state["failed"] += 1
            if state["n"] % 250 == 0:
                out.flush()
                el = time.time() - state["t0"]
                rate = state["n"] / el
                eta = (len(todo) - state["n"]) / rate / 60 if rate else 0
                print(f"[topics] {state['n']}/{len(todo)}  {rate:.1f}/s  "
                      f"ETA {eta:.0f} min  failed {state['failed']}",
                      end="\r", flush=True)

    with open(OUT, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))

    print(f"\ndone: {state['n']:,} fetched, {state['failed']:,} failed")


if __name__ == "__main__":
    main()
