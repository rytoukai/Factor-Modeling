"""Three ways to combine the four factor z-scores into one composite
score: equal-weight (trivial, needs no function here), IC-weighted
(rolling), and Kalman-filtered mean-variance. See Main.ipynb sections
8-10 for how this is used.

Run `python weighting.py` directly to execute the data-independent
correctness checks at the bottom of this file.
"""

import numpy as np
import pandas as pd


def compute_forward_return(panel: pd.DataFrame, price_col: str = "price", ticker_col: str = "ticker") -> pd.Series:
    """Forward 1-month return: price[t+1] / price[t] - 1, per ticker.

    This is a label, not a feature — it's only ever used to score how
    predictive a factor *was* in the IC calculation below, never used
    directly in a stock's own composite score. shift(-1) looks *ahead* one
    row within each ticker's group, which is exactly why this column must
    never be joined back into a stock's own factor inputs.
    """
    next_price = panel.groupby(ticker_col)[price_col].shift(-1)
    return next_price / panel[price_col] - 1


def compute_ic_series(panel: pd.DataFrame, z_col: str, forward_return_col: str = "fwd_return", date_col: str = "date") -> pd.Series:
    """Cross-sectional rank IC per date: Spearman correlation between a
    factor's z-score and the forward return it's trying to predict, across
    every stock that has both non-null on that date. Rank correlation
    (Spearman), not linear (Pearson) — standard in factor investing, and
    far less sensitive to a handful of extreme monthly return outliers
    than a linear correlation would be.

    Looped explicitly over each date rather than groupby().apply(...) —
    slower, but the intermediate result for any one date is easy to
    reproduce by hand if a number ever looks wrong, same tradeoff as the
    per-ticker loop in ffill_with_reporting_lag.
    """
    ic_by_date = {}
    for date, group in panel.groupby(date_col):
        ic_by_date[date] = group[z_col].corr(group[forward_return_col], method="spearman")
    return pd.Series(ic_by_date).sort_index()


def compute_ic_weights(rolling_ic_by_factor: pd.DataFrame) -> pd.DataFrame:
    """Turn each factor's rolling (lagged) IC into a per-date weight.
    Negative IC is clipped to zero, and the positive values renormalized
    to sum to 1. Falls back to equal weight (1/n_factors) when a row's
    weight_sum is 0 — warm-up, or no factor currently has positive
    predictive power.

    .fillna(0) before clipping matters specifically for factors with a
    longer own warm-up than others (momentum and low-vol both need ~13
    months of price history before they're even defined, unlike value and
    quality) — without it, a factor still in its own warm-up has NaN
    rolling IC while its warmed-up peers have real values, the row-level
    weight_sum ends up positive from those peers alone, the whole-row
    equal-weight fallback below never triggers, and that one factor's
    weight is left as NaN instead of being cleanly excluded. Filling with
    0 first treats "no IC history yet for this factor" the same as "this
    factor currently shows no positive predictive power" — excluded from
    the blend, not propagated as a NaN that poisons any stock's composite
    later in combine_weighted."""
    n_factors = rolling_ic_by_factor.shape[1]
    clipped = rolling_ic_by_factor.fillna(0).clip(lower=0)
    weight_sum = clipped.sum(axis=1)

    weights = clipped.div(weight_sum, axis=0)
    weights = weights.where(weight_sum > 0, 1 / n_factors)
    return weights


def combine_weighted(panel: pd.DataFrame, z_cols: list[str], weight_cols: list[str]) -> pd.Series:
    """Weighted average of each stock's z-scores, using per-date,
    per-factor weights already broadcast onto `panel` (one weight column
    per z-score column, same order). Missing z-scores are excluded and the
    remaining weights renormalized, same "exclude, don't zero-fill"
    principle as the equal-weight composite.

    Uses .to_numpy() rather than DataFrame arithmetic: pandas aligns two
    DataFrames' columns by *label* during arithmetic, but z_cols and
    weight_cols intentionally have different names — dropping to numpy
    sidesteps that entirely and just multiplies position-for-position,
    which is correct here only because both lists were built in the same
    order.
    """
    z = panel[z_cols].to_numpy()
    w = panel[weight_cols].to_numpy()

    is_missing = np.isnan(z)
    w_masked = np.where(is_missing, 0.0, w)
    z_filled = np.where(is_missing, 0.0, z)

    weight_sum = w_masked.sum(axis=1)
    contribution = (z_filled * w_masked).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = contribution / weight_sum
    result = np.where(weight_sum > 0, result, np.nan)
    return pd.Series(result, index=panel.index)


def kalman_filter_series(observations: pd.Series, q_to_r_ratio: float) -> pd.Series:
    """1D Kalman filter (local level model) over a single noisy time
    series. Returns a filtered estimate of the series' underlying level at
    every point, using only that point and everything before it.

    R (observation noise) is estimated from the series' own variance; Q
    (process noise) is q_to_r_ratio * R. A missing observation still
    advances the predict step (uncertainty grows) but skips the update —
    the filter just carries its last estimate forward, the Kalman
    equivalent of forward-filling.
    """
    obs = observations.to_numpy()
    n = len(obs)
    r = np.nanvar(obs)
    q = r * q_to_r_ratio

    filtered = np.full(n, np.nan)
    first_valid = np.argmax(~np.isnan(obs))  # index of the first non-NaN observation
    x = 0.0
    # start from a diffuse prior (no real belief about the state before
    # any data): a huge initial variance means the very first observation
    # gets trusted almost completely, rather than being blended with an
    # arbitrary starting guess — this is what makes the Q=0 special case
    # below reduce exactly to a running average, as it should
    p = 1e12 * max(r, 1.0)

    for t in range(first_valid, n):
        p = p + q  # predict: no observation yet, so uncertainty just grows by Q

        y = obs[t]
        if np.isnan(y):
            filtered[t] = x
            continue

        k = p / (p + r)          # Kalman gain: how much to trust the new observation
        x = x + k * (y - x)      # update the state estimate toward the observation
        p = (1 - k) * p          # shrink uncertainty, having incorporated new info
        filtered[t] = x

    return pd.Series(filtered, index=observations.index)


def compute_factor_return_series(panel: pd.DataFrame, z_col: str, forward_return_col: str = "fwd_return", date_col: str = "date") -> pd.Series:
    """Fama-MacBeth factor return: for each date, cross-sectionally
    regress forward_return on the factor's z-score across every stock
    with both non-null (forward_return = alpha + beta * z + error); the
    slope beta is that period's return per unit of factor exposure.

    Simple-regression slope = cov(x, y) / var(x) — computed directly
    rather than pulling in a regression library, since it's one line and
    this module otherwise has no modeling dependency beyond pandas/numpy.
    Looped per date like compute_ic_series, same reasoning: easy to
    reproduce any one date's number by hand if it looks wrong.
    """
    beta_by_date = {}
    for date, group in panel.groupby(date_col):
        x = group[z_col].to_numpy()
        y = group[forward_return_col].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2 or np.var(x, ddof=1) == 0:
            beta_by_date[date] = np.nan
            continue
        beta_by_date[date] = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return pd.Series(beta_by_date).sort_index()


def rolling_covariance_entries(returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing rolling covariance for every unique pair of factor-return
    columns (i <= j, since covariance is symmetric), one column per pair.
    Reshaped this way — rather than pandas' native stacked
    .rolling().cov() output — specifically so the whole thing can be
    Kalman-filtered and .shift(1)'d column-by-column exactly like the
    mean vector."""
    cols = returns_df.columns
    entries = {}
    for i, ci in enumerate(cols):
        for cj in cols[i:]:
            entries[(ci, cj)] = returns_df[ci].rolling(window).cov(returns_df[cj])
    return pd.DataFrame(entries)


def build_cov_matrix(entry_row: pd.Series, cols: list[str]) -> np.ndarray:
    """Reassemble one date's flat (pair -> value) covariance entries back
    into a symmetric n x n matrix, in the same column order as `cols`."""
    n = len(cols)
    mat = np.full((n, n), np.nan)
    for (ci, cj), value in entry_row.items():
        i, j = cols.index(ci), cols.index(cj)
        mat[i, j] = value
        mat[j, i] = value
    return mat


def mean_variance_weights(mean_vec: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
    """Maximum-Sharpe combination weights: w ∝ Σ⁻¹μ, the classical
    tangency-portfolio formula — solved via np.linalg.solve(Σ, μ) rather
    than explicitly inverting Σ, which is both faster and more numerically
    stable. Negative weights are clipped to zero and the remainder
    renormalized to sum to 1 (long-only across factors — the optimizer can
    exclude a factor, not short it). Falls back to equal weight if the
    inputs are missing, or Σ turns out to be singular/unusable (see
    shrink_covariance: independently-smoothed covariance entries aren't
    guaranteed to form a valid matrix every period)."""
    n = len(mean_vec)
    if np.isnan(mean_vec).any() or np.isnan(cov_mat).any():
        return np.full(n, 1 / n)
    try:
        raw_weights = np.linalg.solve(cov_mat, mean_vec)
    except np.linalg.LinAlgError:
        return np.full(n, 1 / n)
    raw_weights = np.clip(raw_weights, 0, None)
    total = raw_weights.sum()
    if total <= 0:
        return np.full(n, 1 / n)
    return raw_weights / total


def shrink_covariance(cov_mat: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink a covariance matrix toward its own diagonal (i.e. toward
    treating the factors as uncorrelated), by a fixed intensity rather
    than an estimated 'optimal' one (a full Ledoit-Wolf-style optimal
    shrinkage estimate wants more history than a 4-asset, ~100-month
    series comfortably supports). Targets a real instability: a noisy,
    independently-smoothed 4x4 covariance matrix produces extreme,
    corner-solution weights when inverted directly — pulling the
    off-diagonal entries toward zero makes the matrix better-conditioned
    and the resulting weights less sensitive to estimation noise in any
    single covariance entry.

    shrinkage=0 reproduces the original unshrunk behavior exactly;
    shrinkage=1 discards all covariance information and treats the four
    factors as independent, which is verified below."""
    diagonal_target = np.diag(np.diag(cov_mat))
    return (1 - shrinkage) * cov_mat + shrinkage * diagonal_target


if __name__ == "__main__":
    # data-independent correctness checks — run with `python weighting.py`

    # with Q=0 (state assumed perfectly constant, no drift), a Kalman
    # filter has a well-known closed form: it reduces exactly to the
    # running sample mean. If this doesn't hold, kalman_filter_series has
    # a bug.
    test_series = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    filtered_no_drift = kalman_filter_series(test_series, q_to_r_ratio=0.0)
    assert np.isclose(filtered_no_drift.iloc[-1], test_series.mean())
    print("kalman_filter_series: Q=0 special case matches the running mean")

    test_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    assert np.allclose(shrink_covariance(test_cov, 0.0), test_cov)
    assert np.allclose(shrink_covariance(test_cov, 1.0), np.diag(np.diag(test_cov)))
    print("shrink_covariance: shrinkage=0 unchanged, shrinkage=1 fully diagonal")
