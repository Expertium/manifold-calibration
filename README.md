# Manifold prediction-market calibration

Are Manifold's prediction markets calibrated — when the market says 70%, do such events
happen 70% of the time? And do markets get *better* as resolution approaches?

Four scripts:

```bash
python scrape.py --target 250000     # collect market + probability data (resumable)
python analyze.py                    # calibration plots, volume/lifespan plots, heatmaps
python scrape_topics.py              # fetch topic tags for the study markets (resumable)
python topics_report.py              # base rate + skill score per topic
```

No API key is needed for any of this -- Manifold's read endpoints are public
(rate limit 500 requests/min per IP; the scrapers default to staying under it).

## What gets collected

Every resolved **YES/NO binary** market (`MKT` = resolved-to-a-probability and
`CANCEL` = N/A are dropped).

For each market, the probability at three moments in its life:

| snapshot | when |
|---|---|
| `early` | creation + min(3 days, lifetime) |
| `mid`   | halfway between creation and closure |
| `late`  | closure − 3 days, floored at creation |

"Closure" is `min(closeTime, resolutionTime)` — the last moment the market was
actually tradeable. A market can be resolved early (trading stops at
resolution) or resolved long after trading stopped (trading stops at close), so
neither timestamp alone is right.

## How the probabilities are recovered

Manifold has no "probability history" endpoint, but every **bet** carries
`probBefore`, `probAfter` and `createdTime`. So the probability at time *t* is
the `probAfter` of the last bet at or before *t* — or the first bet's
`probBefore` (the opening price) if nobody had traded yet.

`GET /v0/bets?contractId=…&limit=1&beforeTime=t` returns exactly that last bet,
so each snapshot costs one small request. Markets with ≤60 unique traders take
a shortcut — their whole history fits in one 1000-bet page, answering all three
snapshots at once. In practice: ~1.3 requests per market.

## Filters (on by default)

`--min-lifetime-days 7`

`--min-bettors 3`

These two filters filter out roughly a third of all markets.

Pass `--min-lifetime-days 0 --min-bettors 0` to see the unfiltered version.

## Metrics

**Brier score** (`mean((p − o)²)`, lower better, 0.25 = always saying 50%) and
**log loss**, plus Murphy's decomposition
`Brier = reliability − resolution + uncertainty` and the **Brier skill score**
`1 − Brier/uncertainty` — 1 is perfect, 0 is no better than always predicting
the group's base rate, below 0 is worse. Skill is what makes groups with
different base rates comparable.

ECE is deliberately **not** reported: it is not a proper scoring rule (it can
be improved by genuinely worse forecasts), depends on an arbitrary binning,
and is biased upward in small samples.

The error band on the calibration plots comes from `binomial_ci.py`, a
standalone numpy/scipy module with five exact, vectorised confidence-interval
methods for a binomial proportion (`--ci` picks one; the default is the
highest-density interval, which has the best coverage at extreme
probabilities). Nothing in it is Manifold-specific — details and coverage
comparisons in [binomial_ci.md](binomial_ci.md).

## Topics

Topic tags (`groupSlugs`) are only exposed on the full-market endpoint, one
request per market -- the bulk endpoint omits them and there is no batch
lookup. `scrape_topics.py` fetches them for the markets that pass the study
filters and writes `data/topics.jsonl` as a permanent join table.

`topics_report.py` then reports, per topic, the YES base rate and the Brier
skill score at each snapshot. Skill is measured against each topic's *own*
base rate, so a lopsided topic gets no credit for being lopsided. Markets
carry several tags, so topic groups overlap and **topic counts do not sum to the
total number of markets**.

## Output

```
data/markets.jsonl    one line per resolved binary market
data/probs.jsonl      one line per market with the three probabilities
data/topics.jsonl     market id -> topic tags (join table)
data/*.json           summary metrics
plots/calibration_{early,mid,late}.png
plots/{brier,logloss,brier_skill,logloss_skill}_vs_{volume,lifespan}.png
plots/skill_heatmap_{early,mid,late}.png
plots/skill_by_topic.png
```

The scrapers are resumable and append-only, and the analysis scripts tolerate
a partial dataset, so everything can run while a scrape is still going.

## Caveats

- Manifold is play money. Incentives are weaker than real-money markets.
- Creators resolve their own markets, so some resolutions are subjective.
- The base rate is ~36% YES, not 50%.
- Markets are self-selected questions, not a random sample of forecastable events.
- Only resolved markets are included, so long-lifespan samples skew toward
  markets created in Manifold's earlier years.
