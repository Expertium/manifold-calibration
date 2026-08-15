"""
Confidence intervals for a binomial proportion.

Implements the four methods compared in

    Orawo, L. A. (2021). "Confidence Intervals for the Binomial Proportion:
    A Comparison of Four Methods." Open Journal of Statistics, 11, 806-816.
    https://doi.org/10.4236/ojs.2021.115047

The one that matters here is `likelihood_interval` -- the paper's likelihood
(relative-likelihood) interval, implemented exactly as defined, and fast.

------------------------------------------------------------------------------
The method, straight from Section 2.3
------------------------------------------------------------------------------

For x successes in n trials the binomial likelihood, dropping the constant
binomial coefficient, is

    L(p) = p**x * (1 - p)**(n - x)

The *relative* likelihood compares p against the MLE p_hat = x/n:

    R(p) = L(p) / L(p_hat)

and the 100c% likelihood interval is the set { p : R(p) >= c }. Because
log R(p) is strictly concave with its maximum at p_hat, that set is always an
interval, and its endpoints are the two roots of

    r(p) = log R(p) = x*log(p) + (n - x)*log(1 - p) - l(p_hat) = log(c)

one on each side of p_hat. The paper says "the use of a numerical procedure is
usually necessary to solve this equation" -- that is the whole computation.

The level c is fixed, not fitted. By Wilks' theorem -2*log R(p) is
asymptotically chi-square with 1 degree of freedom, so a 100(1-alpha)%
confidence interval corresponds to

    c = exp(-chi2_{1, 1-alpha} / 2) = exp(-z_{alpha/2}**2 / 2)

which is 0.1465 at alpha = 0.05 -- the classic "14.65% likelihood interval".

------------------------------------------------------------------------------
Why this is fast
------------------------------------------------------------------------------

r(p) - log(c) is concave, positive at p_hat, and goes to -inf at both ends, so
each root is bracketed from the start: the lower one in (0, p_hat), the upper
one in (p_hat, 1). Bisection on a guaranteed bracket needs no tuning and cannot
fail, and it vectorises perfectly -- every (k, n) pair is solved in lockstep by
the same fixed number of numpy operations. 80 iterations halves the bracket
2**80 times over, which is far past float64 precision, and costs 80 array ops
regardless of how many intervals are being computed.

No grid is involved, so the endpoints are exact rather than being quantised to
a grid step, and nothing is integrated.

------------------------------------------------------------------------------
x = 0 and x = n
------------------------------------------------------------------------------

The paper notes the method "does not produce an interval when the number of
successes x is 0 or n". At those points p_hat sits on the boundary and only one
root exists -- but that root has a closed form, so no solver is needed:

    x = 0:  R(p) = (1-p)**n >= c  ->  [0, 1 - c**(1/n)]
    x = n:  R(p) = p**n     >= c  ->  [c**(1/n), 1]

These are the exact one-sided likelihood intervals, which is what the
degenerate case should give. (This is a cleaner fix than nudging the counts by
a continuity correction, since it changes nothing about the definition.)
"""

import numpy as np
from scipy.stats import beta as _beta, norm as _norm

__all__ = ["likelihood_interval", "hpd_interval", "wilson_interval",
           "clopper_pearson_interval", "wald_interval", "likelihood_level"]


def likelihood_level(alpha=0.05):
    """The relative-likelihood cutoff c matching a 100(1-alpha)% interval."""
    z = _norm.ppf(1 - alpha / 2)
    return float(np.exp(-0.5 * z * z))


def _bisect(k, n, target, lo, hi, increasing, iters=80):
    """Vectorised bisection for x*log(p) + (n-x)*log1p(-p) == target.

    `increasing` says whether the function rises with p over the bracket, which
    is True on the lower side of p_hat and False on the upper side.
    """
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        # Midpoints of (0, p_hat) and (p_hat, 1) are never 0 or 1, so the logs
        # stay finite and no endpoint is ever evaluated.
        g = k * np.log(mid) + (n - k) * np.log1p(-mid) - target
        below = (g < 0) if increasing else (g > 0)
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def likelihood_interval(k, n, alpha=0.05):
    """Likelihood confidence interval for a binomial proportion (Orawo 2021).

    `k` successes out of `n` trials. Both accept arrays and broadcast together;
    the return is a `(low, high)` pair of arrays, or of floats for scalar input.

    >>> lo, hi = likelihood_interval(16, 17)     # the paper's worked example
    >>> round(lo, 4), round(hi, 4)
    (0.7655, 0.9965)
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    k, n = np.broadcast_arrays(k, n)
    if np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n")

    c = likelihood_level(alpha)
    log_c = np.log(c)

    p_hat = np.divide(k, n, out=np.full(k.shape, np.nan), where=n > 0)

    # Interior case: solve r(p) = log(c) on each side of p_hat.
    # Clip only to keep log() finite in the unused boundary lanes; those
    # results are overwritten below.
    safe = np.clip(p_hat, 1e-300, 1 - 1e-16)
    l_max = k * np.log(safe) + (n - k) * np.log1p(-safe)
    target = l_max + log_c

    with np.errstate(divide="ignore", invalid="ignore"):
        low = _bisect(k, n, target, np.zeros_like(p_hat), p_hat, increasing=True)
        high = _bisect(k, n, target, p_hat, np.ones_like(p_hat), increasing=False)

    # Boundary cases: exact closed forms, no solver.
    at_zero = k <= 0
    at_full = k >= n
    root = np.power(c, np.divide(1.0, n, out=np.full(n.shape, np.nan), where=n > 0))
    low = np.where(at_zero, 0.0, np.where(at_full, root, low))
    high = np.where(at_full, 1.0, np.where(at_zero, 1.0 - root, high))

    bad = ~(n > 0)
    low = np.where(bad, np.nan, low)
    high = np.where(bad, np.nan, high)

    if low.ndim == 0:
        return float(low), float(high)
    return low, high


# --------------------------------------------------------------------------
# the highest-density interval
# --------------------------------------------------------------------------

def hpd_interval(k, n, alpha=0.05):
    """Highest-density interval of the flat-prior posterior.

    Normalising p**k * (1-p)**(n-k) over p gives the Beta(k+1, n-k+1) density
    (the flat-prior posterior). The HPD region is the shortest interval
    holding 1-alpha of its mass -- equivalently, a horizontal cut of the
    density with the cut level chosen so the enclosed mass is 1-alpha.

    Parametrise by the mass t left in the lower tail: the interval is
    [ppf(t), ppf(t + 1 - alpha)], whose width is stationary exactly when the
    density is equal at both ends. So solve

        pdf(ppf(t)) - pdf(ppf(t + 1 - alpha)) = 0     for t in (0, alpha)

    That difference is monotonically increasing in t, so bisection is again
    bracketed from the start and vectorises. This differs from the paper's
    likelihood interval: the cut level here is chosen so the enclosed mass is
    1 - alpha, whereas the paper fixes it at c = exp(-z**2/2) regardless.

    When k is 0 or n the mode sits on a boundary and the region is one-sided,
    which is handled directly rather than by nudging the counts.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    k, n = np.broadcast_arrays(k, n)
    if np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n")

    a = k + 1.0
    b = n - k + 1.0

    lo_t = np.zeros(a.shape, dtype=float)
    hi_t = np.full(a.shape, alpha, dtype=float)
    for _ in range(60):
        t = 0.5 * (lo_t + hi_t)
        p1 = _beta.ppf(t, a, b)
        p2 = _beta.ppf(t + 1 - alpha, a, b)
        h = _beta.pdf(p1, a, b) - _beta.pdf(p2, a, b)
        lo_t = np.where(h < 0, t, lo_t)
        hi_t = np.where(h < 0, hi_t, t)
    t = 0.5 * (lo_t + hi_t)
    low = _beta.ppf(t, a, b)
    high = _beta.ppf(t + 1 - alpha, a, b)

    # Boundary cases: the density is monotone, so the region runs to the edge.
    at_zero = k <= 0
    at_full = k >= n
    low = np.where(at_zero, 0.0, np.where(at_full, _beta.ppf(alpha, a, b), low))
    high = np.where(at_full, 1.0, np.where(at_zero, _beta.ppf(1 - alpha, a, b), high))

    if low.ndim == 0:
        return float(low), float(high)
    return low, high


# --------------------------------------------------------------------------
# the other three methods from the paper, for comparison
# --------------------------------------------------------------------------

def wald_interval(k, n, alpha=0.05):
    """p_hat +/- z * sqrt(p_hat*(1-p_hat)/n).  Section 2.1. Can overshoot [0,1]."""
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    z = _norm.ppf(1 - alpha / 2)
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    lo, hi = p - half, p + half
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def wilson_interval(k, n, alpha=0.05):
    """Score interval: [n*p + z^2/2 +/- z*sqrt(n*p*(1-p) + z^2/4)] / (n + z^2)."""
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    z = _norm.ppf(1 - alpha / 2)
    p = k / n
    denom = n + z * z
    centre = (n * p + z * z / 2) / denom
    half = z * np.sqrt(n * p * (1 - p) + z * z / 4) / denom
    lo = np.clip(centre - half, 0, 1)
    hi = np.clip(centre + half, 0, 1)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def clopper_pearson_interval(k, n, alpha=0.05):
    """Exact equal-tail interval via Beta quantiles.  Section 2.2.

    The paper writes this with F quantiles; the Beta form is the same interval
    (Theorem 3 relates the two) and is what scipy exposes directly.
    """
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    lo = _beta.ppf(alpha / 2, k, n - k + 1)
    hi = _beta.ppf(1 - alpha / 2, k + 1, n - k)
    lo = np.where(k <= 0, 0.0, lo)
    hi = np.where(k >= n, 1.0, hi)
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.where(np.isnan(hi), 1.0, hi)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi
