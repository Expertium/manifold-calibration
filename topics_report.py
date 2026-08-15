#!/usr/bin/env python3
"""
Base rates and forecast skill broken down by Manifold topic.

Joins data/topics.jsonl (written by scrape_topics.py) onto the study markets
and reports, for every topic with enough markets, the YES base rate and the
Brier skill score at each snapshot.

    python topics_report.py
    python topics_report.py --min-markets 200 --sort base_rate

Markets carry several topics each, so the groups overlap and the counts do not
sum to the study total. A market with four tags contributes to four rows.

Skill is measured against each topic's *own* base rate, so a lopsided topic
gets no credit for that lopsidedness -- 0 means "no better than knowing this
topic resolves YES x% of the time", which is the comparison that matters when
asking whether some subjects are harder to forecast than others.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load, metrics, SNAPSHOTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")


def load_topics():
    path = os.path.join(DATA, "topics.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("topics"):
                out[r["id"]] = r["topics"]
            elif r.get("topics") == []:
                out[r["id"]] = []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-markets", type=int, default=285)
    ap.add_argument("--min-lifetime-days", type=float, default=7.0)
    ap.add_argument("--min-bettors", type=int, default=3)
    ap.add_argument("--sort", default="early",
                    choices=["late", "mid", "early", "base_rate", "n", "topic"])
    ap.add_argument("--top", type=int, default=0,
                    help="show only the N best and N worst by the sort key")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    recs = [r for r in load(os.path.join(DATA, "probs.jsonl"))
            if r["lifetime_days"] >= args.min_lifetime_days
            and r["bettors"] >= args.min_bettors]
    topics = load_topics()
    have = [r for r in recs if r["id"] in topics]
    print(f"{len(recs):,} study markets, {len(have):,} with topics fetched "
          f"({100*len(have)/max(1,len(recs)):.1f}%)")

    by_topic = defaultdict(list)
    untagged = 0
    for r in have:
        ts = topics[r["id"]]
        if not ts:
            untagged += 1
        for t in ts:
            by_topic[t].append(r)
    print(f"{len(by_topic):,} distinct topics; {untagged:,} markets carry no "
          f"topic at all")

    rows = []
    for t, rs in by_topic.items():
        if len(rs) < args.min_markets:
            continue
        o = np.array([r["outcome"] for r in rs], dtype=float)
        row = {"topic": t, "n": len(rs), "base_rate": float(o.mean())}
        for key, short, _ in SNAPSHOTS:
            p = np.array([r[key] for r in rs], dtype=float)
            m = metrics(p, o)
            row[short] = m["brier_skill"]
            row[f"{short}_brier"] = m["brier"]
        rows.append(row)

    keymap = {"topic": lambda r: r["topic"], "n": lambda r: -r["n"],
              "base_rate": lambda r: -r["base_rate"]}
    rows.sort(key=keymap.get(args.sort, lambda r: -r[args.sort]))
    print(f"{len(rows):,} topics with >= {args.min_markets} markets\n")

    show = rows
    if args.top and len(rows) > 2 * args.top:
        show = rows[:args.top] + [None] + rows[-args.top:]

    print(f"{'topic':<34}{'n':>7}{'base rate':>10}"
          f"{'skill:early':>12}{'mid':>8}{'late':>8}")
    print("-" * 79)
    for r in show:
        if r is None:
            print(f"{'...':<34}")
            continue
        print(f"{r['topic'][:33]:<34}{r['n']:>7,}{r['base_rate']:>10.3f}"
              f"{r['early']:>12.3f}{r['mid']:>8.3f}{r['late']:>8.3f}")

    with open(os.path.join(DATA, f"topics_summary{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote data/topics_summary{args.tag}.json")

    if rows:
        plot(rows, args)


def plot(rows, args):
    """Base rate vs late skill, one point per topic."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    x = [r["base_rate"] for r in rows]
    y = [r["late"] for r in rows]
    n = np.array([r["n"] for r in rows], dtype=float)
    sizes = 25 + 200 * (np.log10(n) - np.log10(n.min())) / max(
        1e-9, np.log10(n.max()) - np.log10(n.min()))
    sc = ax.scatter(x, y, s=sizes, c=y, cmap="RdYlGn", edgecolor="black",
                    lw=0.4, alpha=0.9)

    # label the extremes rather than every point, or it turns to soup
    order = sorted(range(len(rows)), key=lambda i: y[i])
    for i in order[:8] + order[-8:] + sorted(
            range(len(rows)), key=lambda i: -n[i])[:6]:
        ax.annotate(rows[i]["topic"][:24], (x[i], y[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                    alpha=0.85)

    ax.axhline(0, color="grey", lw=1, ls=":")
    ax.set_xlabel("YES base rate of the topic", fontsize=12)
    ax.set_ylabel("Brier skill score, late snapshot  (higher = better)",
                  fontsize=12)
    ax.set_title(f"Forecast skill by topic\n"
                 f"{len(rows)} Manifold topics with >= {args.min_markets} "
                 f"resolved markets   |   point size = market count",
                 fontsize=13)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02).set_label(
        "late skill", fontsize=10)
    fig.tight_layout()
    path = os.path.join(PLOTS, f"skill_by_topic{args.tag}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote plots/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
