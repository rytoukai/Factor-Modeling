"""The four raw factor definitions (momentum, value, quality, low-vol) and
cross-sectional z-scoring. Every factor is oriented so that higher = better.
See Main.ipynb's Factor Implementation section for how this is used."""

import pandas as pd


def momentum_12_1(panel: pd.DataFrame, price_col: str = "price", ticker_col: str = "ticker") -> pd.Series:
    """12-1 month momentum: (price[t-1] / price[t-13]) - 1.

    Shifts are taken within each ticker's own price history (via groupby)
    so one ticker's prices never leak into another's calculation at the
    boundary between tickers. Assumes `panel` is sorted by [ticker, date]
    with no missing months per ticker — otherwise shift(13) grabs "13 rows
    back" rather than "13 calendar months back."
    """
    price_by_ticker = panel.groupby(ticker_col)[price_col]
    price_t_minus_1 = price_by_ticker.shift(1)
    price_t_minus_13 = price_by_ticker.shift(13)
    return price_t_minus_1 / price_t_minus_13 - 1


def value_factor(panel: pd.DataFrame, ev_ebitda_col: str = "ev_ebitda") -> pd.Series:
    """Value factor: -1 * EV/EBITDA. A lower EV/EBITDA means a cheaper
    stock, so it's negated here to match the "higher = better" convention
    used for every factor."""
    return -1 * panel[ev_ebitda_col]


def quality_factor(panel: pd.DataFrame, roe_col: str = "roe") -> pd.Series:
    """Quality factor: ROE as-is. Higher ROE already means better quality,
    so no inversion — kept as its own function so leverage or other quality
    signals can be folded in later without touching call sites."""
    return panel[roe_col]


def low_vol_factor(
    panel: pd.DataFrame,
    price_col: str = "price",
    ticker_col: str = "ticker",
    window: int = 12,
) -> pd.Series:
    """Low-volatility factor: -1 * trailing `window`-month std dev of
    monthly returns. Returns and the rolling std dev are both computed
    within each ticker's own history (via groupby) so one ticker's
    prices/returns never leak into another's calculation. Negated so a
    higher factor value means lower realized volatility ("better").
    Assumes `panel` is sorted by [ticker, date] with no missing months per
    ticker, same as momentum_12_1.
    """
    returns = panel.groupby(ticker_col)[price_col].pct_change()
    trailing_vol = (
        returns.groupby(panel[ticker_col])
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )
    return -1 * trailing_vol


def zscore_within_sector(
    panel: pd.DataFrame,
    factor_col: str,
    date_col: str = "date",
    sector_col: str = "GICS",
    eligible_col: str = "above_cap_floor",
) -> pd.Series:
    """Cross-sectional z-score of `factor_col`, computed within each
    (date, sector) group, using only cap-floor-eligible rows to set the
    group's mean/std. Ineligible rows and rows with a missing factor value
    both come back NaN — excluded from the score, not zero-filled.
    """
    # NaN out ineligible rows *before* grouping, so they don't influence
    # the mean/std either — pandas' mean()/std() skip NaN by default
    eligible_values = panel[factor_col].where(panel[eligible_col])

    group_keys = [panel[date_col], panel[sector_col]]
    group_mean = eligible_values.groupby(group_keys).transform("mean")
    group_std = eligible_values.groupby(group_keys).transform("std")

    return (eligible_values - group_mean) / group_std
