# Confidence intervals for a binomial proportion (`binomial_ci.py`)

A standalone module (numpy + scipy only) that computes the error bands on the
calibration plots. Nothing in it is Manifold-specific — it implements the four
interval methods compared in [Orawo (2021), "Confidence Intervals for the
Binomial Proportion: A Comparison of Four
Methods"](https://doi.org/10.4236/ojs.2021.115047), plus the highest-density
interval, all exact and vectorised. `analyze.py` selects one with `--ci`:

| `--ci` | method |
|---|---|
| `hpd` (default) | highest-density interval of the flat-prior posterior |
| `likelihood` | Orawo 2021 §2.3 — the paper's likelihood interval |
| `wilson`, `clopper-pearson`, `wald` | the usual suspects |

All methods share the signature `interval(k, n, alpha=0.05) -> (low, high)`
and accept arrays.

## `likelihood_interval`

The paper's method exactly: the interval is `{p : R(p) ≥ c}` where
`R(p) = L(p)/L(p̂)` is the relative likelihood and the level is fixed by
Wilks' theorem at `c = exp(−z²/2) ≈ 0.1465`. Since `log R` is concave with its
peak at `p̂`, each endpoint is bracketed from the start — lower root in
`(0, p̂)`, upper in `(p̂, 1)` — so vectorised bisection solves every interval
in lockstep with no grid and no tuning. Endpoints satisfy `R(p) = c` to ~1e-15.

Validation: reproduces the paper's worked example (16/17 → 0.7656, 0.9965 vs
the paper's 0.7658, 0.9965) and its Table 1 coverage figures. Two numbers in
that table are typos — the likelihood interval's mean length at n=100 is
printed as `0.81`, which is `0.181`, and at n=50 as `0.216`, which is
inconsistent with the paper's own pattern of the likelihood interval being
slightly *wider* than Wilson at every other n.

`x = 0` and `x = n`, which the paper says the method can't handle, have exact
one-sided closed forms: `[0, 1 − c^(1/n)]` and `[c^(1/n), 1]`.

## `hpd_interval`

The highest-density interval of the flat-prior posterior. Normalising the
binomial likelihood `L(p) = p^k (1−p)^(n−k)` over `p` gives the
Beta(k+1, n−k+1) density, and the HPD region is the shortest interval holding
`1−α` of its mass — equivalently, a horizontal cut of the density with the cut
level chosen so the enclosed mass is `1−α` (where the likelihood interval
instead fixes the level by Wilks' theorem). Parametrising by the lower-tail
mass `t` makes the interval `[ppf(t), ppf(t+1−α)]`, whose width is stationary
exactly where the density matches at both ends — one more monotone, bracketed
bisection, with mass correct to ~1e-10 and endpoint densities equal to ~1e-14.
When `k` is 0 or `n` the density is monotone and the region runs one-sided to
the boundary.

## Which to use

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
regime where that gap opens up, so **`hpd` is the default**. Clopper-Pearson
never drops below nominal but pays ~10% extra width for it.
