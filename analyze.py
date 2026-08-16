#!/usr/bin/env python3
"""
Calibration analysis of Manifold prediction markets.

Reads data/probs.jsonl (written by scrape.py) and produces:

  * three reliability diagrams -- one per snapshot in a market's life
    (early / midpoint / late), each with the count histogram underneath
  * Brier and log loss as functions of total trading volume
  * a metrics table: Brier with its Murphy decomposition, and log loss

Brier score is the mean squared error of the probability, mean((p - o)^2),
where o is 1 for YES and 0 for NO. Lower is better; 0.25 is what you get by
always saying 50%. Murphy's decomposition splits it into

    Brier = reliability - resolution + uncertainty

where reliability measures calibration error (lower better), resolution
measures how far forecasts stray from the base rate (higher better), and
uncertainty is the irreducible base-rate variance o_bar*(1-o_bar).

Settings live in the constants block below main's helper functions (no
command-line arguments): edit them there and run the file.

One common sample: markets with at least MIN_TRADERS traders as of the early
snapshot. Nothing is filtered on a market's outcome, its final trader count or
how long it turned out to run -- all of which are unknowable when the forecast
is made, and selecting on them biases the calibration estimate.

That single filter suffices because scrape.snapshot_times caps each horizon at
a quarter of the market's life, so early <= mid <= late holds for every market
and each graded price necessarily comes after the cutoff. The 0.50
opening-price artifact is therefore impossible by construction rather than
filtered away afterwards.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import binomial_ci

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")

SNAPSHOTS = [
    ("p_early", "early", "min(3 days, 1/4 of market life) after creation"),
    ("p_mid",   "mid",   "midpoint between creation and resolution"),
    ("p_late",  "late",  "min(3 days, 1/4 of market life) before the end"),
]

EPS = 1e-4  # log-loss clipping; Manifold prices can reach ~0.001


def load(path):
    """Read probs.jsonl, tolerating a torn final line while a scrape is running."""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def bin_stats(p, o, n_bins):
    """Group forecasts into equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    # right-closed so that p == 1.0 lands in the last bin, not one past it
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt:
            out.append({"bin": b, "n": cnt, "p_mean": float(p[m].mean()),
                        "o_mean": float(o[m].mean()),
                        "lo": float(edges[b]), "hi": float(edges[b + 1])})
        else:
            out.append({"bin": b, "n": 0, "p_mean": float("nan"),
                        "o_mean": float("nan"),
                        "lo": float(edges[b]), "hi": float(edges[b + 1])})
    return out


def metrics(p, o, n_bins=20):
    """Brier + Murphy decomposition, log loss.

    No ECE: it is not a proper scoring rule, so it can be improved by
    forecasts that are worse. It is also sensitive to the binning it is
    computed over and biased upward in small samples. Brier and log loss are
    both proper and agree with each other here.
    """
    n = len(p)
    brier = float(np.mean((p - o) ** 2))
    base = float(o.mean())
    uncertainty = base * (1 - base)

    reliability = resolution = 0.0
    for b in bin_stats(p, o, n_bins):
        if not b["n"]:
            continue
        w = b["n"] / n
        reliability += w * (b["p_mean"] - b["o_mean"]) ** 2
        resolution += w * (b["o_mean"] - base) ** 2

    pc = np.clip(p, EPS, 1 - EPS)
    logloss = float(-np.mean(o * np.log(pc) + (1 - o) * np.log(1 - pc)))

    # Skill relative to always predicting this group's own base rate. Raw Brier
    # and log loss are not comparable across groups whose base rates differ:
    # a lopsided group has a lower irreducible floor, so its scores look better
    # no matter how good the forecasts are. Skill divides that floor out.
    # 1 = perfect, 0 = no better than knowing the base rate, < 0 = worse.
    entropy = (float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
               if 0 < base < 1 else 0.0)
    nan = float("nan")
    brier_skill = 1 - brier / uncertainty if uncertainty > 0 else nan
    logloss_skill = 1 - logloss / entropy if entropy > 0 else nan

    return {"n": n, "mean_p": float(p.mean()), "brier": brier,
            "logloss": logloss, "brier_skill": brier_skill,
            "logloss_skill": logloss_skill, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty,
            "base_rate": base}


def _logit(p):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.log(pc / (1 - pc))


def cox_calibration(p, o, iters=50):
    """Logistic recalibration: logit(P(YES)) = a + b * logit(price).

    Fitted on the individual markets, not the binned points, so the binning
    choice cannot influence it.

    b == 1 with a == 0 is perfect. b < 1 means the prices are too extreme
    (overconfident), b > 1 not extreme enough (underconfident). a != 0 is a
    directional bias: the whole curve shifted rather than reshaped, which is a
    different failure from over/underconfidence and worth reading separately.

    Returns (a, b, se_a, se_b). Solved by Newton-Raphson; the standard errors
    come from the inverse observed information at the optimum.
    """
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (mu - o))
        beta = beta - step
        if np.max(np.abs(step)) < 1e-11:
            break
    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    w = np.clip(mu * (1 - mu), 1e-12, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    return (float(beta[0]), float(beta[1]),
            float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])))


def linear_calibration(p, o):
    """Least-squares fit: P(YES) = a + b * price, in plain probability units.

    Reads straight off the plot: the fit is a literal straight line, b is how
    steep it is against the diagonal and a is where it starts. b < 1 with the
    line crossing the diagonal mid-range means prices are too extreme; a below
    0 shifts the whole line down, meaning YES is priced too high throughout.

    Compared with cox_calibration this trades statistical tidiness for
    readability -- it is not bounded to [0, 1], and it treats an error at
    p = 0.5 like one at p = 0.02, where the logistic version works in log-odds
    and so weighs the tails far more heavily. For judging a calibration plot by
    eye, that is usually the trade you want.

    Standard errors are heteroskedasticity-robust (HC0): a binary outcome has
    variance p(1-p), which varies across the range, so the textbook OLS errors
    would be wrong.
    """
    X = np.column_stack([np.ones_like(p), p])
    beta = np.linalg.lstsq(X, o, rcond=None)[0]
    resid = o - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ (X.T @ (X * (resid ** 2)[:, None])) @ xtx_inv
    return (float(beta[0]), float(beta[1]),
            float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])))


CI_METHODS = {
    "likelihood": binomial_ci.likelihood_interval,   # Orawo (2021), Section 2.3
    "hpd": binomial_ci.hpd_interval,                 # flat-prior posterior HPD
    "wilson": binomial_ci.wilson_interval,
    "clopper-pearson": binomial_ci.clopper_pearson_interval,
    "wald": binomial_ci.wald_interval,
}


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def calibration_plot(p, o, title, subtitle, path, n_bins=20, min_bin=20,
                     ci="hpd", alpha=0.05, fit="linear"):
    bins = bin_stats(p, o, n_bins)
    shown = [b for b in bins if b["n"] >= min_bin]

    xs = [b["p_mean"] for b in shown]
    ys = [b["o_mean"] for b in shown]
    if shown:
        counts = np.array([b["n"] for b in shown], dtype=float)
        successes = np.round(np.array(ys) * counts)
        los, his = CI_METHODS[ci](successes, counts, alpha=alpha)
        los, his = np.atleast_1d(los), np.atleast_1d(his)
    else:
        los, his = (), ()

    m = metrics(p, o, n_bins)

    fig, ax = plt.subplots(figsize=(10, 9))
    perfect = ax.plot([0, 1], [0, 1], color="tab:orange", lw=2,
                      label="Perfect calibration", zorder=3)[0]
    band = actual = None
    if shown:
        band = ax.fill_between(xs, los, his, color="tab:blue", alpha=0.18, lw=0,
                               label=f"{100*(1-alpha):.0f}% CI", zorder=2)
        actual = ax.plot(xs, ys, color="tab:blue", lw=2, marker="o", ms=4,
                         label="Actual calibration", zorder=4)[0]

    grid = np.linspace(1e-3, 1 - 1e-3, 500)
    fit_handles, fit_labels = [], []
    if fit in ("linear", "both"):
        la, lb, lsa, lsb = linear_calibration(p, o)
        fit_handles.append(ax.plot(grid, la + lb * grid, color="tab:purple",
                                   lw=1.8, ls="--", label="Linear fit",
                                   zorder=5)[0])
        fit_labels.append(f"linear fit:   slope {lb:.3f} $\\pm$ {lsb:.3f}"
                          f"      intercept {la:+.3f} $\\pm$ {lsa:.3f}")
    if fit in ("logistic", "both"):
        ca, cb, csa, csb = cox_calibration(p, o)
        curve = 1.0 / (1.0 + np.exp(-(ca + cb * _logit(grid))))
        fit_handles.append(ax.plot(grid, curve, color="tab:brown", lw=1.8,
                                   ls=":", label="Logistic fit", zorder=5)[0])
        fit_labels.append(f"logistic fit (log-odds):   slope {cb:.3f} "
                          f"$\\pm$ {csb:.3f}      intercept {ca:+.3f} "
                          f"$\\pm$ {csa:.3f}")

    ax.set_xlabel("Predicted probability (market price)", fontsize=13)
    ax.set_ylabel("Actual frequency of YES", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # market counts as a histogram on a twin axis, under the calibration curve
    ax2 = ax.twinx()
    counts = [b["n"] for b in bins]
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    ax2.bar(centers, counts, width=1.0 / n_bins, color="tab:blue", alpha=0.35,
            edgecolor="black", lw=0.5, zorder=1, label="Number of markets")
    ax2.set_ylabel("Number of markets", fontsize=13)
    ax2.set_ylim(0, max(counts) * 1.05 if counts else 1)

    # Order the legend the way the eye reads the plot -- reference line, the
    # measured curve, then the band around it -- rather than by draw order.
    h2, l2 = ax2.get_legend_handles_labels()
    handles = ([h for h in (perfect, actual) if h is not None] + fit_handles
               + [h for h in (band,) if h is not None])
    ax.legend(handles + h2, [h.get_label() for h in handles] + l2,
              loc="upper left", fontsize=11, framealpha=0.95)

    # stat = (f"Brier {m['brier']:.4f}    log loss {m['logloss']:.4f}    "
    #         f"base rate {m['base_rate']:.3f}")
    stat = f"Brier={m['brier']:.4f}"
    lines = "\n".join([title, subtitle, stat] + fit_labels)
    ax.set_title(lines, fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return m


def _trim(s):
    return s.rstrip("0").rstrip(".") if "." in s else s


def _mana(v, _pos=None):
    """Tick labels: 100, 577, 1.3k, 15.6k, 55.2M.

    Rounded to a couple of significant figures -- these label quantile edges as
    well as round axis ticks, and a raw edge prints as 1.34706k otherwise.
    """
    if v < 0:
        return ""
    if v >= 1_000_000:
        return _trim(f"{v / 1_000_000:.1f}") + "M"
    if v >= 1000:
        return _trim(f"{v / 1000:.1f}") + "k"
    return f"{v:.0f}"


def value_bins(x, n_bins=8, min_n=250):
    """Group markets into bins of a positive quantity, respecting ties.

    Volume is effectively continuous (~95% distinct values), so this reduces to
    ordinary equal-count quantile bins. Liquidity is the opposite -- 100 and
    1000 mana alone are ~85% of all markets -- and there quantile edges
    collapse, because the median and the 75th percentile are the same repeated
    value, leaving only two usable bins.

    Walking the distinct values and accumulating them handles both: a value
    holding a whole bin's worth of markets gets its own bin, and sparse values
    between the spikes merge until they reach the target size.

    Returns a list of (mask, lo, hi, n).
    """
    liq = x
    pos = liq[liq > 0]
    if len(pos) < min_n * 2:
        return []
    values, counts = np.unique(pos, return_counts=True)

    # A value holding a whole bin's worth of markets on its own gets its own
    # bin. Sizing the *merged* bins off the total would then be wrong: the
    # spikes are ~85% of the data, so the target would be far larger than
    # everything left over and the sparse stretches would never split. Size
    # them off the non-spike markets and the bins left to spend on them.
    solo = max(min_n, len(pos) // n_bins)
    spikes = counts >= solo
    sparse = int(counts[~spikes].sum())
    target = max(min_n, sparse // max(1, n_bins - int(spikes.sum())))

    groups, cur, cur_n = [], [], 0
    for v, c in zip(values, counts):
        # A spike must not absorb the stragglers just below it, or the "1000
        # mana exactly" bin quietly becomes "662 to 1000". Stragglers too small
        # to stand alone go back into the bin below them instead of forward
        # into the spike -- same liquidity neighbourhood, and nothing is lost.
        if cur_n > 0 and c >= solo:
            if cur_n >= min_n or not groups:
                groups.append((cur, cur_n))
            else:
                prev, prev_n = groups[-1]
                groups[-1] = (prev + cur, prev_n + cur_n)
            cur, cur_n = [], 0
        cur.append(v)
        cur_n += int(c)
        if c >= solo or cur_n >= target:
            groups.append((cur, cur_n))
            cur, cur_n = [], 0
    if cur:
        if groups and cur_n < min_n:          # fold a small tail into the last bin
            prev, prev_n = groups[-1]
            groups[-1] = (prev + cur, prev_n + cur_n)
        else:
            groups.append((cur, cur_n))

    out = []
    for vals, n in groups:
        if n < min_n:
            continue
        lo, hi = float(min(vals)), float(max(vals))
        out.append(((liq >= lo) & (liq <= hi), lo, hi, n))
    return out


def _plain(v, _pos=None):
    """Tick labels for quantities that are not mana: 7, 30, 100, 1000."""
    if v < 0:
        return ""
    return f"{v:.0f}" if v >= 10 else _trim(f"{v:.1f}")


X_AXES = {
    # cli name:  (record field, axis label, short name, tick formatter)
    "volume": ("volume",
               "Total trading volume (mana, log scale)",
               "trading volume", _mana),
    "liquidity": ("liquidity",
                  "Market liquidity (mana, log scale)",
                  "liquidity", _mana),
    "lifespan": ("lifetime_days",
                 "Market lifespan (days, log scale)",
                 "market lifespan", _plain),
}


METRIC_PANELS = [
    ("brier", "Brier score", "lower = better"),
    ("logloss", "Log loss", "lower = better"),
    ("brier_skill", "Brier skill score",
     "higher = better;  0 = no better than the bin's base rate"),
    ("logloss_skill", "Log loss skill score",
     "higher = better;  0 = no better than the bin's base rate"),
]


def metrics_vs_activity_plots(recs, plots_dir, field="volume", tag="", n_bins=8):
    """One figure per metric, plotted against market volume (or liquidity).

    Returns (rows, filenames).
    """
    rec_field, axis_label, short_name, tick_fmt = X_AXES[field]
    xv = np.array([r.get(rec_field) or 0.0 for r in recs], dtype=float)
    groups = value_bins(xv, n_bins=n_bins)
    if len(groups) < 2:
        print(f"[plot] {short_name} too concentrated to bin")
        return [], []

    outcome = np.array([r["outcome"] for r in recs], dtype=float)
    colors = {"p_early": "tab:blue", "p_mid": "tab:green", "p_late": "tab:red"}

    xs = [float(np.exp(np.mean(np.log(xv[m])))) for m, _, _, _ in groups]  # geo mean
    ns = [n for _, _, _, n in groups]

    # Collect every series first: the n labels go above whichever curve is
    # topmost at that x, which is not known until all three are computed.
    series, rows = {}, []
    for key, short, _ in SNAPSHOTS:
        pk = np.array([r[key] for r in recs], dtype=float)
        keep = common_mask(recs)
        for name, _, _ in METRIC_PANELS:
            series[(name, short)] = []
        for mask, lo, hi, n in groups:
            sel = mask & keep
            if sel.sum() < 30:          # too thin to score once filtered
                for name, _, _ in METRIC_PANELS:
                    series[(name, short)].append(np.nan)
                continue
            m = metrics(pk[sel], outcome[sel])
            for name, _, _ in METRIC_PANELS:
                series[(name, short)].append(m[name])
            rows.append({"snapshot": short, "x_field": field, "lo": lo,
                         "hi": hi, "n": int(sel.sum()), "brier": m["brier"],
                         "logloss": m["logloss"], "base_rate": m["base_rate"],
                         "uncertainty": m["uncertainty"]})

    # "Even" allows a little slack: quantile bins land within a market or two
    # of each other, since the total rarely divides exactly. Report that as a
    # single approximate size -- quoting a 1,947-1,951 range would imply a
    # spread that matters, and calling it "equal-count" would contradict it.
    even_bins = max(ns) - min(ns) <= max(1, round(0.02 * max(ns)))
    if min(ns) == max(ns):
        bin_note = f"{len(ns)} bins, {min(ns):,} markets each"
    elif even_bins:
        bin_note = f"{len(ns)} bins, ~{round(sum(ns) / len(ns)):,} markets each"
    else:
        bin_note = f"{len(ns)} bins, {min(ns):,}-{max(ns):,} markets each"

    written = []
    for name, label, direction in METRIC_PANELS:
        fig, ax = plt.subplots(figsize=(11, 7))
        for key, short, _ in SNAPSHOTS:
            ax.plot(xs, series[(name, short)], marker="o", lw=2,
                    color=colors[key], label=short)

        # Equal-count bins (volume) make a label per point pure repetition, so
        # state it once in the title instead. Tied values (liquidity) give
        # genuinely uneven bins, and there each point needs its own label.
        if not even_bins:
            tops = [max(series[(name, s)][i] for _, s, _ in SNAPSHOTS)
                    for i in range(len(xs))]
            # Angled rather than horizontal: on a log axis the middle bins sit
            # close enough that level labels run into each other and into the
            # curve. 30 degrees still collides where the curve dips and the
            # next label starts where the previous one ends; 45 clears it.
            for xi, ni, ti in zip(xs, ns, tops):
                ax.annotate(f"n={ni:,}", (xi, ti), textcoords="offset points",
                            xytext=(1, 6), ha="left", va="bottom", rotation=45,
                            rotation_mode="anchor", fontsize=7.5, alpha=0.75)
            ax.margins(y=0.18)      # headroom so the labels are not clipped

        ax.set_ylabel(f"{label}  ({direction})", fontsize=12)
        ax.grid(alpha=0.35, which="both")
        ax.set_axisbelow(True)
        ax.legend(title="snapshot", fontsize=11)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=(2, 3, 5), numticks=12))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_fmt))
        ax.xaxis.set_minor_formatter(mticker.FuncFormatter(tick_fmt))
        ax.tick_params(axis="x", which="both", labelsize=9)
        ax.set_xlabel(axis_label, fontsize=12)

        ax.set_title(f"{label} vs {short_name}\n"
                     f"Manifold resolved YES/NO binary markets   |   {bin_note}",
                     fontsize=13)
        fig.tight_layout()
        fname = f"{name}_vs_{field}{tag}.png"
        fig.savefig(os.path.join(plots_dir, fname), dpi=140)
        plt.close(fig)
        written.append(fname)

    return rows, written


def _quantile_edges(v, n_bins, split_low=1):
    """Quantile edges, deduplicated. Returns edges and a bin index per row.

    split_low subdivides the lowest bin further, by quantiles of the rows
    inside it. Volume is so skewed that the bottom sixth spans 0 to ~577 mana
    -- a wider spread than the five bins above it put together -- so it is
    worth resolving on its own terms.
    """
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if split_low > 1 and len(edges) > 1:
        inner = v[(v >= edges[0]) & (v < edges[1])]
        if len(inner) > split_low * 50:
            extra = np.quantile(inner, np.linspace(0, 1, split_low + 1))[1:-1]
            edges = np.unique(np.concatenate([edges, extra]))
    idx = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    return edges, idx


def skill_heatmap(recs, plots_dir, tag="", x_field="volume",
                  y_field="lifespan", n_bins=6, metric="brier_skill",
                  min_cell=100, split_low=3):
    """Skill score over a volume x lifespan grid, one panel per snapshot.

    Volume and lifespan are close to independent here (log-log correlation
    ~0.03), so quantile edges on each axis fill the grid fairly evenly rather
    than piling everything onto a diagonal.

    Skill rather than raw Brier because the base rate varies a lot across these
    cells; raw scores would mostly track how lopsided each cell is.
    """
    x_rec, _, x_short, x_fmt = X_AXES[x_field]
    y_rec, _, y_short, y_fmt = X_AXES[y_field]
    xv = np.array([r.get(x_rec) or 0.0 for r in recs], dtype=float)
    yv = np.array([r.get(y_rec) or 0.0 for r in recs], dtype=float)
    outcome = np.array([r["outcome"] for r in recs], dtype=float)

    x_edges, x_idx = _quantile_edges(xv, n_bins, split_low=split_low)
    y_edges, y_idx = _quantile_edges(yv, n_bins)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1

    label = dict((m[0], m[1]) for m in METRIC_PANELS)[metric]
    grids, counts = {}, np.zeros((ny, nx), dtype=int)
    for key, short, _ in SNAPSHOTS:
        pk = np.array([r[key] for r in recs], dtype=float)
        keep = common_mask(recs)
        g = np.full((ny, nx), np.nan)
        for iy in range(ny):
            for ix in range(nx):
                m = (x_idx == ix) & (y_idx == iy) & keep
                counts[iy, ix] = int(m.sum())
                if m.sum() >= min_cell:
                    g[iy, ix] = metrics(pk[m], outcome[m])[metric]
        grids[short] = g

    # One file per snapshot: three panels side by side only stay legible at
    # full width. The colour scale is computed across all three first, so the
    # separate files stay directly comparable.
    allv = np.concatenate([g[~np.isnan(g)].ravel() for g in grids.values()])
    vmin, vmax = float(allv.min()), float(allv.max())

    written = []
    for _, short, desc in SNAPSHOTS:
        g = grids[short]
        fig, ax = plt.subplots(figsize=(1.05 * nx + 3.6, 0.8 * ny + 3.4))
        im = ax.imshow(g, origin="lower", cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       aspect="auto")
        for iy in range(ny):
            for ix in range(nx):
                if np.isnan(g[iy, ix]):
                    ax.text(ix, iy, "n/a", ha="center", va="center",
                            fontsize=9, color="grey")
                else:
                    ax.text(ix, iy, f"{g[iy, ix]:.2f}", ha="center",
                            va="center", fontsize=11, color="black")
        ax.set_xticks(range(nx))
        ax.set_xticklabels([f"{x_fmt(x_edges[i])}-\n{x_fmt(x_edges[i+1])}"
                            for i in range(nx)], fontsize=9)
        ax.set_yticks(range(ny))
        ax.set_yticklabels([f"{y_fmt(y_edges[i])}-{y_fmt(y_edges[i+1])}"
                            for i in range(ny)], fontsize=10)
        ax.set_ylabel("Market lifespan (days)", fontsize=12)
        ax.set_xlabel("Total trading volume (mana)", fontsize=12)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(f"{label}   (green = better)", fontsize=10)

        ax.set_title(f"{label} -- {short}: {desc}\n"
                     f"{len(recs):,} markets, {counts.min():,}-"
                     f"{counts.max():,} per cell   |   colour scale shared "
                     f"across all three snapshots", fontsize=11)
        fig.tight_layout()
        fname = f"skill_heatmap_{short}{tag}.png"
        fig.savefig(os.path.join(plots_dir, fname), dpi=140)
        plt.close(fig)
        written.append(fname)

    return grids, written


# --------------------------------------------------------------------------
# settings -- edit here and run the file; there are no command-line arguments
# --------------------------------------------------------------------------

MIN_TRADERS = 3           # per snapshot: require this many traders *as of that
                          # snapshot*. Each snapshot is therefore judged only on
                          # markets that had real trading at the moment being
                          # graded -- see snapshot_mask.
BINS = 20                 # probability bins on the calibration plots
MIN_BIN = 20              # hide calibration bins with fewer markets than this
CI = "hpd"                # error-band method; see CI_METHODS for the options
ALPHA = 0.05              # 1 - confidence level (0.05 -> 95% bands)
FIT = "none"              # "linear", "logistic", "both" or "none"
X_AXIS = ["volume", "lifespan"]   # accuracy-vs-what plots; "liquidity" also works
HEATMAP_BINS = 6          # grid size for the volume x lifespan skill heatmap
ACTIVITY_BINS = 12        # target bin count on the accuracy-vs-activity plots
TAG = ""                  # suffix for output filenames


def common_mask(recs):
    """Markets with at least MIN_TRADERS traders as of the *early* snapshot.

    One rule, decided at the earliest moment any panel is graded, and applied
    to all three. That gives a single sample for the whole study without any
    lookahead: membership depends only on how many people had traded by then,
    which is visible at that moment, and never on the outcome, the final
    trader count, or how long the market turned out to run.

    It works because scrape.snapshot_times caps each horizon at a quarter of
    the market's life, so early <= mid <= late always holds. Every graded
    price therefore comes after this filter's cutoff, meaning a market that
    passes has real trading behind all three of its quotes -- the 0.50
    opening-price artifact cannot occur, rather than being filtered away
    afterwards.
    """
    return np.array([r.get("traders_early", 0) >= MIN_TRADERS for r in recs])


def main():

    recs = load(os.path.join(DATA, "probs.jsonl"))
    total = len(recs)
    if not recs:
        raise SystemExit("no markets in data/probs.jsonl")

    os.makedirs(PLOTS, exist_ok=True)
    keep_all = common_mask(recs)
    print(f"loaded {total:,} markets; {keep_all.sum():,} graded "
          f"({100*keep_all.mean():.1f}%) -- one common sample, markets with "
          f">= {MIN_TRADERS} traders by the early snapshot")
    o_keep = np.array([r["outcome"] for r in recs], dtype=float)[keep_all]
    print("base rate: %.4f YES" % o_keep.mean())


    o_all = np.array([r["outcome"] for r in recs], dtype=float)
    results = {}
    for key, short, desc in SNAPSHOTS:
        keep = keep_all
        p = np.array([r[key] for r in recs], dtype=float)[keep]
        o = o_all[keep]
        fname = f"calibration_{short}{TAG}.png"
        m = calibration_plot(
            p, o,
            title=f"Manifold calibration -- {short} in market life",
            subtitle=f"{keep.sum():,} of {len(recs):,} resolved YES/NO markets "
                     f"(>= {MIN_TRADERS} traders by then)   |   probability at {desc}",
            path=os.path.join(PLOTS, fname),
            n_bins=BINS, min_bin=MIN_BIN,
            ci=CI, alpha=ALPHA, fit=FIT)
        results[short] = m
        print(f"wrote plots/{fname}")

    _, hm_files = skill_heatmap(recs, PLOTS, tag=TAG,
                                n_bins=HEATMAP_BINS)
    for fname in hm_files:
        print(f"wrote plots/{fname}")

    rows = []
    for field in X_AXIS:
        r, written = metrics_vs_activity_plots(
            recs, PLOTS, field=field, tag=TAG, n_bins=ACTIVITY_BINS)
        rows += r
        for fname in written:
            print(f"wrote plots/{fname}")

    print("\n" + "=" * 94)
    print(f"{'snapshot':<10}{'n':>8}{'MeanPrice':>11}{'ActualYES':>11}{'Gap':>8}"
          f"{'Brier':>10}{'LogLoss':>10}{'Reliab':>10}{'Resol':>8}{'Uncert':>8}")
    print("-" * 94)
    for _, short, _ in SNAPSHOTS:
        m = results[short]
        print(f"{short:<10}{m['n']:>8,}{m['mean_p']:>11.4f}"
              f"{m['base_rate']:>11.4f}{m['mean_p'] - m['base_rate']:>+8.4f}"
              f"{m['brier']:>10.4f}{m['logloss']:>10.4f}"
              f"{m['reliability']:>10.5f}{m['resolution']:>8.4f}"
              f"{m['uncertainty']:>8.4f}")
    print("=" * 94)
    print("Brier = reliability - resolution + uncertainty   (lower Brier is better;")
    print("lower reliability = better calibrated; higher resolution = more informative)")
    print("Gap = MeanPrice - ActualYES: positive means YES is priced too high on average.")

    summary = {"n_total": total, "n_loaded": len(recs),
               "filters": {"min_traders_per_snapshot": MIN_TRADERS},
               "metrics": results, "metrics_by_activity": rows}
    with open(os.path.join(DATA, f"summary{TAG}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote data/summary{TAG}.json")


if __name__ == "__main__":
    main()
