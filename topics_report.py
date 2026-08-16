#!/usr/bin/env python3
"""
Base rates and forecast skill broken down by Manifold topic.

Joins data/topics.jsonl (written by scrape_topics.py) onto the study markets
and reports, for the N topics with the most markets, the YES base rate and the
Brier skill score at each snapshot.

Settings live in the constants block below (no command-line arguments): edit
them there and run the file.

Markets carry several topics each, so the groups overlap and the counts do not
sum to the study total. A market with four tags contributes to four rows.

Skill is measured against each topic's *own* base rate, so a lopsided topic
gets no credit for that lopsidedness -- 0 means "no better than knowing this
topic resolves YES x% of the time", which is the comparison that matters when
asking whether some subjects are harder to forecast than others.
"""

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


# --------------------------------------------------------------------------
# settings -- edit here and run the file; there are no command-line arguments
# --------------------------------------------------------------------------

TOPICS = 100              # how many topics, taking those with the most markets
MIN_LIFETIME_DAYS = 7.0   # must match the filters used in the analysis
MIN_BETTORS = 3
SORT = "early"            # "late", "mid", "early", "base_rate", "n" or "topic"
TOP = 0                   # nonzero: print only the N best and N worst rows
TAG = ""                  # suffix for output filenames


def main():
    recs = [r for r in load(os.path.join(DATA, "probs.jsonl"))
            if r["lifetime_days"] >= MIN_LIFETIME_DAYS
            and r["bettors"] >= MIN_BETTORS]
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

    biggest = sorted(by_topic.items(), key=lambda kv: -len(kv[1]))[:TOPICS]
    rows = []
    for t, rs in biggest:
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
    rows.sort(key=keymap.get(SORT, lambda r: -r[SORT]))
    print(f"top {len(rows):,} topics by market count "
          f"(smallest has {min(r['n'] for r in rows):,} markets)\n")

    show = rows
    if TOP and len(rows) > 2 * TOP:
        show = rows[:TOP] + [None] + rows[-TOP:]

    print(f"{'topic':<34}{'n':>7}{'base rate':>10}"
          f"{'skill:early':>12}{'mid':>8}{'late':>8}")
    print("-" * 79)
    for r in show:
        if r is None:
            print(f"{'...':<34}")
            continue
        print(f"{r['topic'][:33]:<34}{r['n']:>7,}{r['base_rate']:>10.3f}"
              f"{r['early']:>12.3f}{r['mid']:>8.3f}{r['late']:>8.3f}")

    with open(os.path.join(DATA, f"topics_summary{TAG}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote data/topics_summary{TAG}.json")

    if rows:
        plot(rows)


def plot(rows):
    """Base rate vs early skill, one point per topic."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    x = np.array([r["base_rate"] for r in rows], dtype=float)
    y = np.array([r["early"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    sizes = 25 + 200 * (n - n.min()) / max(1e-9, n.max() - n.min())
    sc = ax.scatter(x, y, s=sizes, c=y, cmap="RdYlGn", edgecolor="black",
                    lw=0.4, alpha=0.9)

    # Label only the notable topics, or it turns to soup: the biggest, and
    # the extremes of both axes.
    notable = set()
    for key in (-n, -y, y, -x, x):
        notable.update(np.argsort(key, kind="stable")[:5])

    ax.axhline(0, color="grey", lw=1, ls=":")
    ax.margins(0.06)

    # Place labels with collision handling: try several positions around each
    # point and keep the first that neither overlaps an already-placed label
    # nor leaves the axes. Points are labelled in order of market count, so
    # when a cluster of labels cannot all fit, the biggest topic wins.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    placed = []
    candidates = [(6, 4, "left"), (-6, 4, "right"), (6, -11, "left"),
                  (-6, -11, "right"), (0, 9, "center"), (0, -16, "center")]
    for i in sorted(notable, key=lambda i: -n[i]):
        for dx, dy, ha in candidates:
            a = ax.annotate(rows[i]["topic"], (x[i], y[i]),
                            textcoords="offset points", xytext=(dx, dy),
                            ha=ha, fontsize=7.5, alpha=0.85)
            bb = a.get_window_extent(renderer).expanded(1.1, 1.15)
            ok = (bb.x0 >= ax_box.x0 and bb.x1 <= ax_box.x1
                  and bb.y0 >= ax_box.y0 and bb.y1 <= ax_box.y1
                  and not any(bb.overlaps(p) for p in placed))
            if ok:
                placed.append(bb)
                break
            a.remove()
        # all candidate spots collide -> this label is dropped entirely
    ax.set_xlabel("YES base rate of the topic", fontsize=12)
    ax.set_ylabel("Brier skill score, early snapshot  (higher = better)",
                  fontsize=12)
    ax.set_title(f"Forecast skill by topic\n"
                 f"{len(rows)} Manifold topics with the most resolved "
                 f"markets   |   bubble size = market count", fontsize=13)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02).set_label(
        "early skill", fontsize=10)
    fig.tight_layout()
    path = os.path.join(PLOTS, f"skill_by_topic{TAG}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote plots/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
