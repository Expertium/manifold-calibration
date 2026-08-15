# Manifold prediction-market calibration

Are Manifold's prediction markets calibrated — when the market says 70%, does it
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
`CANCEL` = N/A are dropped, since we need a hard outcome to score against; CASH
sweepstakes markets are dropped because they mirror the MANA ones and would
double-count).

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
so each snapshot costs one small request. Markets with ≤60 unique traders take a
shortcut: their entire history fits in one 1000-bet page
(`order=asc`), so one request answers all three snapshots. In practice this
averages ~1.3 requests per market.

All endpoints are public and read-only — **no API key needed**. The limit is 500
requests/min per IP; `--rpm` defaults to 450.

## Filters (on by default, and they matter)

**`--min-lifetime-days 7`.** Short markets wreck the comparison. Under 6 days
the `early` (+3d) and `late` (−3d) snapshots cross over each other; under ~3
days the `late` snapshot lands at creation, before anyone traded, so its
"probability" is just the opening price (usually 0.50) rather than a forecast.
Unfiltered, this *inverts* the headline result and makes the early snapshot look
more accurate than the late one — an artifact, not a finding.

This bites hard because resolved markets skew short-lived: a market created
recently can only be resolved already if it was brief.

**`--min-bettors 3`.** One or two traders is a single person's opinion at
whatever price the AMM happened to sit at, not a market forecast.

Pass `--min-lifetime-days 0 --min-bettors 0` to see the unfiltered version.

## Metrics

**Brier score** — mean squared error of the probability, `mean((p − o)²)` with
`o ∈ {0,1}`. Lower is better; always guessing 50% scores 0.25.

Murphy's decomposition splits it into

```
Brier = reliability − resolution + uncertainty
```

- **reliability** — calibration error (lower better). This is the part the
  calibration plots show.
- **resolution** — how far forecasts stray from the base rate, i.e. how
  *informative* they are (higher better). A forecaster who always predicts the
  base rate is perfectly calibrated but useless; resolution catches that.
- **uncertainty** — irreducible base-rate variance, `ō(1−ō)`. Not affected by
  the forecaster; it just sets the scale.

Also reported: log loss (probabilities clipped to 1e−4).

ECE is deliberately **not** reported. It is not a proper scoring rule, so a
forecaster can improve its ECE by making genuinely worse forecasts; it also
depends on an arbitrary binning and is biased upward in small samples. Brier
and log loss are both proper, and they agree with each other throughout.

## Confidence intervals (`binomial_ci.py`)

The error band on the calibration plots comes from `binomial_ci.py`, a
standalone module (numpy + scipy only) implementing the four methods from
Orawo (2021) plus the highest-density interval. Pick with `--ci`:

| `--ci` | method |
|---|---|
| `hpd` (default) | highest-density interval of the flat-prior posterior |
| `likelihood` | Orawo 2021 §2.3 — the paper's likelihood interval |
| `wilson`, `clopper-pearson`, `wald` | the usual suspects |

**`likelihood_interval`** is the paper's method exactly: the interval is
`{p : R(p) ≥ c}` where `R(p) = L(p)/L(p̂)` is the relative likelihood and the
level is fixed by Wilks' theorem at `c = exp(−z²/2) ≈ 0.1465`. Since `log R` is
concave with its peak at `p̂`, each endpoint is bracketed from the start —
lower root in `(0, p̂)`, upper in `(p̂, 1)` — so vectorised bisection solves
every interval in lockstep with no grid and no tuning. Endpoints satisfy
`R(p) = c` to ~1e-15.

Validation lives in the scratchpad scripts: it reproduces the paper's worked
example (16/17 → 0.7656, 0.9965 vs the paper's 0.7658, 0.9965) and its Table 1
coverage figures. Two numbers in that table are typos — the likelihood interval's
mean length at n=100 is printed as `0.81`, which is `0.181`, and at n=50 as
`0.216`, which is inconsistent with the paper's own pattern of the likelihood
interval being slightly *wider* than Wilson at every other n.

`x = 0` and `x = n`, which the paper says the method can't handle, have exact
one-sided closed forms: `[0, 1 − c^(1/n)]` and `[c^(1/n), 1]`.

**`hpd_interval`** is the method from
[fsrs-optimizer PR #166](https://github.com/open-spaced-repetition/fsrs-optimizer/pull/166),
computed exactly instead of on a grid. Normalising `L(p)` over `p` *is* the
Beta(k+1, n−k+1) density, so that PR's search for the horizontal cut enclosing
95% of the mass is the highest-density region of that Beta. Parametrising by
the lower-tail mass `t` makes the interval `[ppf(t), ppf(t+1−α)]`, whose width
is stationary exactly where the density matches at both ends — one more
monotone, bracketed bisection. It agrees with the grid version to 8e-06, below
that version's own 1e-5 grid step.

### Which to use

Measured coverage of nominal 95% intervals, exactly (not simulated):

| grid | method | mean coverage | worst-case | mean length |
|---|---|---|---|---|
| p ∈ (0.2, 0.8) | likelihood | 0.945–0.949 | 0.911–0.935 | — |
| | hpd | 0.938–0.948 | 0.897–0.935 | slightly shorter |
| p ∈ (0.01, 0.99) | likelihood | 0.945–0.948 | **0.820–0.850** | — |
| | hpd | 0.949–0.950 | **0.896–0.921** | slightly shorter |

On the paper's own grid the likelihood interval is the better of the two. Over
the wider range the ordering reverses sharply: the fixed Wilks level is an
asymptotic approximation that degrades at extreme `p`, while the HPD keeps
near-nominal coverage and is slightly shorter besides.

Calibration plots put their most populated bins right at 0 and 1, which is the
regime where that gap opens up, so **`hpd` is the default**. Use
`--ci likelihood` for the paper's method. Clopper-Pearson never drops below
nominal but pays ~10% extra width for it.

## Topics

Topic tags (`groupSlugs`) are only exposed on the full-market endpoint, one
request per market -- the bulk endpoint omits them and there is no batch
lookup. `scrape_topics.py` fetches them for the markets that pass the study
filters and writes `data/topics.jsonl` as a permanent join table.

`topics_report.py` then reports, per topic, the YES base rate and the Brier
skill score at each snapshot. Skill is measured against each topic's *own*
base rate, so a lopsided topic gets no credit for being lopsided. Markets
carry several tags, so topic groups overlap and counts do not sum to the
study total.

## Output

```
data/markets.jsonl   one line per market that passed the filters
data/probs.jsonl     one line per market with the three probabilities
data/cursor.json     enumeration progress (re-running resumes)
plots/calibration_{early,mid,late}.png
plots/brier_vs_liquidity.png
```

Both scripts are resumable and append-only, so a run can be interrupted and
restarted, and `analyze.py` can be run against a partial dataset while the
scrape is still going.

## Caveats

- Manifold is play money. Incentives are weaker than real-money markets.
- Creators resolve their own markets, so some resolutions are subjective.
- The base rate sits near 43% YES, not 50%.
- Markets are self-selected questions, not a random sample of forecastable events.
